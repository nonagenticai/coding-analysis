"""Bounds checking verification for symbolic analysis."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BoundsVerifier:
    """Verifies proper bounds checking for list/array access."""

    async def verify(self, content: str, file_path: str) -> dict[str, Any]:
        """Verify proper bounds checking for list/array access."""
        violations = []
        subscript_accesses = []

        try:
            tree = ast.parse(content)

            class BoundsChecker(ast.NodeVisitor):
                """AST visitor to check for potential bounds violations."""

                def visit_Subscript(self, node):
                    """Track all subscript accesses."""
                    # Track all subscript accesses
                    line_num = node.lineno
                    if isinstance(node.value, ast.Name):
                        subscript_accesses.append(
                            {
                                "line": line_num,
                                "variable": node.value.id,
                                "slice_type": type(node.slice).__name__,
                            }
                        )
                    self.generic_visit(node)

            checker = BoundsChecker()
            checker.visit(tree)

            # Analyze for potential bounds violations
            violations = self._analyze_accesses(content, subscript_accesses)

        except SyntaxError as e:
            logger.warning("Syntax error in file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "syntax_error",
                    "line": getattr(e, "lineno", 0),
                    "message": f"Syntax error prevents bounds checking analysis: {e}",
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
            "property": "bounds_checking",
            "violations": violations,
            "subscript_accesses": len(subscript_accesses),
            "safe_accesses": len(subscript_accesses) - len(violations),
        }

    def _analyze_accesses(self, content: str, subscript_accesses: list) -> list:
        """Analyze subscript accesses for potential bounds violations."""
        violations = []
        lines = content.split("\n")

        for access in subscript_accesses:
            line_idx = access["line"] - 1
            if line_idx < len(lines):
                line_content = lines[line_idx].strip()

                # Look for bounds checking in context
                context_lines = lines[max(0, line_idx - 5) : line_idx + 1]

                variable = access["variable"]
                # Check for common bounds checking patterns
                has_bounds_check = self._has_bounds_check(variable, context_lines)

                if not has_bounds_check and "[" in line_content:
                    violations.append(
                        {
                            "type": "unchecked_bounds_access",
                            "line": access["line"],
                            "message": (
                                f"Potential bounds violation for '{variable}' without length checking"
                            ),
                            "severity": "medium",
                            "code_snippet": line_content,
                        }
                    )

        return violations

    def _has_bounds_check(self, variable: str, context_lines: list) -> bool:
        """Check if variable has bounds checking in context lines."""
        return any(
            f"len({variable})" in ctx
            or f"range(len({variable}))" in ctx
            or f"if {variable}:" in ctx
            or "except IndexError" in ctx
            for ctx in context_lines
        )
