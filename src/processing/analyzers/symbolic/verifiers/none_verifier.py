"""None checking verification for symbolic analysis."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class NoneVerifier:
    """Verifies proper None checking patterns in code."""

    async def verify(self, content: str, file_path: str) -> dict[str, Any]:
        """Verify proper None checking patterns."""
        violations = []
        patterns = []

        try:
            # Parse the AST
            tree = ast.parse(content)

            # Look for potential None access violations
            class NoneChecker(ast.NodeVisitor):
                """AST visitor to check for potential None access violations."""

                def visit_Attribute(self, node):
                    """Check for direct attribute access that might be None."""
                    # Check for direct attribute access that might be None
                    if isinstance(node.value, ast.Name):
                        # Look for patterns like obj.attr without checking obj is not None
                        line_num = node.lineno
                        patterns.append(
                            {
                                "type": "unchecked_attribute_access",
                                "line": line_num,
                                "variable": node.value.id,
                                "attribute": node.attr,
                            }
                        )
                    self.generic_visit(node)

                def visit_Subscript(self, node):
                    """Check for subscript access that might be None."""
                    # Check for subscript access that might be None
                    if isinstance(node.value, ast.Name):
                        line_num = node.lineno
                        patterns.append(
                            {
                                "type": "unchecked_subscript_access",
                                "line": line_num,
                                "variable": node.value.id,
                            }
                        )
                    self.generic_visit(node)

            checker = NoneChecker()
            checker.visit(tree)

            # Analyze patterns for potential None violations
            violations = self._analyze_patterns(content, patterns)

        except SyntaxError as e:
            logger.warning("Syntax error in file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "syntax_error",
                    "line": getattr(e, "lineno", 0),
                    "message": f"Syntax error prevents None checking analysis: {e}",
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
            "property": "none_checking",
            "violations": violations,
            "patterns_found": len(patterns),
            "safe_patterns": len(patterns) - len(violations),
        }

    def _analyze_patterns(self, content: str, patterns: list) -> list:
        """Analyze patterns for potential None violations."""
        violations = []
        lines = content.split("\n")

        for pattern in patterns:
            line_idx = pattern["line"] - 1
            if line_idx < len(lines):
                line_content = lines[line_idx].strip()

                # Check if there's None checking in the vicinity
                context_lines = lines[max(0, line_idx - 3) : line_idx + 1]

                variable = pattern.get("variable", "")
                if variable and not self._has_none_check(variable, context_lines):
                    violations.append(
                        {
                            "type": "potential_none_access",
                            "line": pattern["line"],
                            "message": (
                                f"Potential None access for variable '{variable}' without proper checking"
                            ),
                            "severity": "medium",
                            "code_snippet": line_content,
                        }
                    )

        return violations

    def _has_none_check(self, variable: str, context_lines: list) -> bool:
        """Check if variable has None checking in context lines."""
        return any(
            f"{variable} is not None" in ctx or f"if {variable}:" in ctx or f"{variable} and" in ctx
            for ctx in context_lines
        )
