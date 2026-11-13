"""Exception handling verification for symbolic analysis."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExceptionVerifier:
    """Verifies proper exception handling patterns."""

    async def verify(self, content: str, file_path: str) -> dict[str, Any]:
        """Verify proper exception handling patterns."""
        violations = []
        exception_patterns = []

        try:
            tree = ast.parse(content)

            class ExceptionChecker(ast.NodeVisitor):
                """AST visitor to check for exception handling issues."""

                def visit_Try(self, node):
                    """Analyze try-except blocks."""
                    # Analyze try-except blocks
                    has_generic_except = False
                    has_specific_except = False
                    has_finally = bool(node.finalbody)

                    for handler in node.handlers:
                        if handler.type is None:  # bare except:
                            has_generic_except = True
                            exception_patterns.append(
                                {
                                    "type": "bare_except",
                                    "line": handler.lineno,
                                    "severity": "medium",
                                }
                            )
                        else:
                            has_specific_except = True

                    exception_patterns.append(
                        {
                            "type": "try_block",
                            "line": node.lineno,
                            "has_generic": has_generic_except,
                            "has_specific": has_specific_except,
                            "has_finally": has_finally,
                        }
                    )

                    self.generic_visit(node)

                def visit_Raise(self, node):
                    """Check raise statements."""
                    # Check raise statements
                    if node.exc is None:  # bare raise
                        exception_patterns.append({"type": "bare_raise", "line": node.lineno})
                    self.generic_visit(node)

                def visit_Call(self, node):
                    """Check for operations that commonly raise exceptions."""
                    # Check for operations that commonly raise exceptions
                    risky_operations = [
                        "int",
                        "float",
                        "open",
                        "json.loads",
                        "requests.get",
                    ]

                    if isinstance(node.func, ast.Name) and node.func.id in risky_operations:
                        exception_patterns.append(
                            {
                                "type": "risky_operation",
                                "line": node.lineno,
                                "operation": node.func.id,
                            }
                        )

                    self.generic_visit(node)

            checker = ExceptionChecker()
            checker.visit(tree)

            # Analyze exception handling patterns for violations
            violations = self._analyze_exception_patterns(exception_patterns)

        except SyntaxError as e:
            logger.warning("Syntax error in file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "syntax_error",
                    "line": getattr(e, "lineno", 0),
                    "message": f"Syntax error prevents exception handling analysis: {e}",
                    "severity": "high",
                }
            )
        except (ValueError, TypeError) as e:
            logger.error("Error parsing file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "parse_error",
                    "line": 0,
                    "message": f"Error parsing file: {e}",
                    "severity": "high",
                }
            )

        return {
            "property": "exception_handling",
            "violations": violations,
            "try_blocks": len([p for p in exception_patterns if p["type"] == "try_block"]),
            "risky_operations": len(
                [p for p in exception_patterns if p["type"] == "risky_operation"]
            ),
        }

    def _analyze_exception_patterns(self, exception_patterns: list) -> list:
        """Analyze exception handling patterns for violations."""
        violations = []

        for pattern in exception_patterns:
            if pattern["type"] == "bare_except":
                violations.append(
                    {
                        "type": "overly_broad_exception",
                        "line": pattern["line"],
                        "message": "Bare except clause catches all exceptions - too broad",
                        "severity": "medium",
                    }
                )
            elif pattern["type"] == "risky_operation":
                # Check if risky operation is in try block
                line_num = pattern["line"]
                is_in_try = any(
                    p["type"] == "try_block"
                    # Within reasonable range
                    and abs(p["line"] - line_num) <= 10
                    for p in exception_patterns
                )

                if not is_in_try:
                    violations.append(
                        {
                            "type": "unhandled_risky_operation",
                            "line": line_num,
                            "message": (
                                f"Risky operation '{pattern['operation']}' not in try-except block"
                            ),
                            "severity": "low",
                        }
                    )

        return violations
