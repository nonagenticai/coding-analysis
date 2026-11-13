"""Resource management verification for symbolic analysis."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResourceVerifier:
    """Verifies proper resource management (files, connections, etc.)."""

    async def verify(self, content: str, file_path: str) -> dict[str, Any]:
        """Verify proper resource management (files, connections, etc.)."""
        violations = []
        resource_operations = []

        try:
            tree = ast.parse(content)

            class ResourceChecker(ast.NodeVisitor):
                """AST visitor to check for resource management issues."""

                def visit_Call(self, node):
                    """Check for resource allocation without proper management."""
                    # Check for resource allocation without proper management
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["open", "socket", "connect"]:
                            resource_operations.append(
                                {
                                    "type": "resource_allocation",
                                    "line": node.lineno,
                                    "function": node.func.id,
                                }
                            )
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in ["open", "connect", "cursor"]:
                            resource_operations.append(
                                {
                                    "type": "resource_allocation",
                                    "line": node.lineno,
                                    "function": node.func.attr,
                                }
                            )
                    self.generic_visit(node)

                def visit_With(self, node):
                    """Track proper with statement usage."""
                    # Track proper with statement usage
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            resource_operations.append(
                                {
                                    "type": "proper_resource_management",
                                    "line": node.lineno,
                                    "managed": True,
                                }
                            )
                    self.generic_visit(node)

            checker = ResourceChecker()
            checker.visit(tree)

            # Analyze resource management patterns
            violations = self._analyze_resource_patterns(content, resource_operations)

        except SyntaxError as e:
            logger.warning("Syntax error in file %s: %s", file_path, e)
            violations.append(
                {
                    "type": "syntax_error",
                    "line": getattr(e, "lineno", 0),
                    "message": f"Syntax error prevents resource management analysis: {e}",
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
            "property": "resource_management",
            "violations": violations,
            "resource_operations": len(resource_operations),
            "properly_managed": len([op for op in resource_operations if op.get("managed")]),
        }

    def _analyze_resource_patterns(self, content: str, resource_operations: list) -> list:
        """Analyze resource management patterns for violations."""
        violations = []
        lines = content.split("\n")

        for op in resource_operations:
            if op["type"] == "resource_allocation" and not op.get("managed"):
                line_idx = op["line"] - 1
                if line_idx < len(lines):
                    line_content = lines[line_idx].strip()

                    # Check if this allocation is in a with statement or has proper cleanup
                    context_lines = lines[max(0, line_idx - 2) : min(len(lines), line_idx + 10)]
                    context = "\n".join(context_lines)

                    # Look for proper cleanup patterns
                    has_proper_cleanup = self._has_proper_cleanup(line_content, context)

                    if not has_proper_cleanup:
                        violations.append(
                            {
                                "type": "resource_leak",
                                "line": op["line"],
                                "message": (
                                    f"Resource allocation '{op['function']}' without proper cleanup/context manager"
                                ),
                                "severity": "medium",
                                "code_snippet": line_content,
                            }
                        )

        return violations

    def _has_proper_cleanup(self, line_content: str, context: str) -> bool:
        """Check if resource has proper cleanup patterns."""
        return any(
            [
                "with " in line_content,  # Already in with statement
                ".close()" in context,  # Manual close
                "finally:" in context,  # Finally block cleanup
                "except:" in context and ".close()" in context,  # Exception handling with cleanup
            ]
        )
