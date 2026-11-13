"""Type safety verification for symbolic analysis."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TypeVerifier:
    """Verifies type safety patterns in code."""

    async def verify(self, content: str, file_path: str) -> dict[str, Any]:
        """Verify type safety patterns."""
        violations = []
        type_issues = []

        try:
            tree = ast.parse(content)

            class TypeChecker(ast.NodeVisitor):
                """AST visitor to check for type safety issues."""

                def visit_FunctionDef(self, node):
                    """Check for missing type annotations."""
                    # Check for missing type annotations
                    if not node.returns and len(node.args.args) > 0:
                        type_issues.append(
                            {
                                "type": "missing_return_annotation",
                                "line": node.lineno,
                                "function": node.name,
                            }
                        )

                    # Check for untyped parameters
                    for arg in node.args.args:
                        if not arg.annotation:
                            type_issues.append(
                                {
                                    "type": "missing_parameter_annotation",
                                    "line": node.lineno,
                                    "function": node.name,
                                    "parameter": arg.arg,
                                }
                            )

                    self.generic_visit(node)

                def visit_Call(self, node):
                    """Check for potentially unsafe operations."""
                    # Check for potentially unsafe operations
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec"]:
                            type_issues.append(
                                {
                                    "type": "unsafe_evaluation",
                                    "line": node.lineno,
                                    "function": node.func.id,
                                }
                            )
                    self.generic_visit(node)

            checker = TypeChecker()
            checker.visit(tree)

            # Convert type issues to violations based on severity
            violations = self._analyze_type_issues(type_issues)

        except SyntaxError as e:
            logger.warning("Syntax error in file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "syntax_error",
                    "line": getattr(e, "lineno", 0),
                    "message": f"Syntax error prevents type safety analysis: {e}",
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
            "property": "type_safety",
            "violations": violations,
            "type_issues_found": len(type_issues),
            "critical_violations": len([v for v in violations if v["severity"] == "high"]),
        }

    def _analyze_type_issues(self, type_issues: list) -> list:
        """Convert type issues to violations based on severity."""
        violations = []

        for issue in type_issues:
            if issue["type"] == "unsafe_evaluation":
                violations.append(
                    {
                        "type": issue["type"],
                        "line": issue["line"],
                        "message": (
                            f"Unsafe use of {issue['function']} - security and type safety risk"
                        ),
                        "severity": "high",
                    }
                )
            elif issue["type"] in [
                "missing_return_annotation",
                "missing_parameter_annotation",
            ]:
                # Only flag as violation if this is a public function
                function_name = issue.get("function", "")
                if not function_name.startswith("_"):
                    violations.append(
                        {
                            "type": issue["type"],
                            "line": issue["line"],
                            "message": (
                                f"Missing type annotation in public function '{function_name}'"
                            ),
                            "severity": "low",
                        }
                    )

        return violations
