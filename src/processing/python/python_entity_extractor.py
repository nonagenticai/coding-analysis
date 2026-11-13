"""Python entity extractor for extracting code entities from Python AST.

This module provides the PythonEntityExtractor class that implements the
BaseEntityExtractor interface for Python code analysis. It extracts classes,
functions, and methods from Python source code using the ast module.
"""

from __future__ import annotations

import ast
import ast as ast_module
import logging
from typing import Any

from ..base_extractor import BaseEntityExtractor

logger = logging.getLogger(__name__)


class PythonEntityExtractor(BaseEntityExtractor):
    """Extractor for Python code entities using the ast module."""

    async def extract_entities(  # type: ignore[invalid-method-override]  # ty: ignore[invalid-method-override]
        self, ast_tree: ast_module.Module, file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract entities (classes, functions, methods) from Python code.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file
            module_name: Name of the module being processed

        Returns:
            List of entity dictionaries
        """
        if not ast_tree:
            logger.warning("Empty AST tree for %s", file_path)
            return []

        try:
            entities: list[dict[str, Any]] = []
            self._extract_entities_from_ast(ast_tree, file_path, entities)
            return entities
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(
                "Error extracting entities from %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return []

    def _extract_entities_from_ast(
        self,
        ast_tree: ast.Module,
        file_path: str,
        entities: list[dict[str, Any]],
    ) -> None:
        """Extract entities from AST and add them to the entities list.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file
            entities: List to populate with extracted entities
        """

        class EntityVisitor(ast.NodeVisitor):
            """AST visitor for extracting entities from Python code."""

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                """Visit a class definition node and extract class information.

                Args:
                    node: The ClassDef AST node to process
                """
                # Extract class information
                class_name = node.name
                class_id = f"{file_path}::{class_name}"

                # Get docstring if available
                docstring = ast.get_docstring(node)

                # Get line numbers
                start_line = node.lineno
                end_line = max(
                    (n.lineno for n in ast.walk(node) if hasattr(n, "lineno")),
                    default=start_line,
                )

                # Create class entity
                entities.append(
                    {
                        "id": class_id,
                        "name": class_name,
                        "type": "class",
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "docstring": docstring,
                        "properties": {
                            "bases": [
                                base.id if isinstance(base, ast.Name) else "" for base in node.bases
                            ],
                            "decorators": [
                                (
                                    d.id
                                    if isinstance(d, ast.Name)
                                    else (
                                        d.func.id
                                        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                                        else ""
                                    )
                                )
                                for d in node.decorator_list
                            ],
                        },
                    }
                )

                # Continue visiting child nodes
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                """Visit an async function definition node (same extraction as sync)."""
                self.visit_FunctionDef(node)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit a function definition node and extract function information.

                Args:
                    node: The FunctionDef AST node to process
                """
                # Extract function information
                func_name = node.name

                # Determine if this is a method (inside a class)
                parent_class = None
                for ancestor in ast.walk(ast_tree):
                    if isinstance(ancestor, ast.ClassDef) and node in ancestor.body:
                        parent_class = ancestor.name
                        break

                # Create ID based on whether it's a method or function
                if parent_class:
                    func_id = f"{file_path}::{parent_class}.{func_name}"
                    func_type = "method"
                else:
                    func_id = f"{file_path}::{func_name}"
                    func_type = "function"

                # Get docstring if available
                docstring = ast.get_docstring(node)

                # Get line numbers
                start_line = node.lineno
                end_line = max(
                    (n.lineno for n in ast.walk(node) if hasattr(n, "lineno")),
                    default=start_line,
                )

                # Extract parameters
                params: list[Any] = []
                for arg in node.args.args:
                    params.append(arg.arg)

                # Create function entity
                entities.append(
                    {
                        "id": func_id,
                        "name": func_name,
                        "type": func_type,
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "docstring": docstring,
                        "properties": {
                            "parameters": params,
                            "parent_class": parent_class,
                            "decorators": [
                                (
                                    d.id
                                    if isinstance(d, ast.Name)
                                    else (
                                        d.func.id
                                        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                                        else ""
                                    )
                                )
                                for d in node.decorator_list
                            ],
                        },
                    }
                )

                # Continue visiting child nodes
                self.generic_visit(node)

        # Run the visitor on the AST
        EntityVisitor().visit(ast_tree)
