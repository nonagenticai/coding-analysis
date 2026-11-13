"""Python relation extractor for code analysis.

This module provides the PythonRelationExtractor class for extracting
relationships between entities in Python code using AST analysis.
"""

from __future__ import annotations

import ast as ast_module
import logging
import os
from typing import Any

from ..base_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class PythonRelationExtractor(RelationExtractor):
    """Extractor for Python code relationships using the ast module."""

    async def extract_all_relations(
        self, ast: ast_module.Module, file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract all relationships between entities from Python code.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file
            module_name: Name of the module being processed

        Returns:
            List of relationship dictionaries
        """
        if not ast:
            logger.warning("Empty AST tree for %s", file_path)
            return []

        try:
            # Extract different types of relationships
            basic_relationships = self._extract_basic_relationships(ast, file_path)
            function_calls = self._extract_function_calls(ast, file_path)
            imports = self._extract_imports(ast, file_path)

            # Combine all relationships
            relationships = basic_relationships + function_calls + imports
            return relationships
        except (SyntaxError, ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Error extracting relationships from %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return []

    def _extract_basic_relationships(
        self, ast: ast_module.Module, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract basic relationships between entities in Python code.

        This method identifies parent-child relationships between classes, methods,
        and functions in the Python AST.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file

        Returns:
            List of relationship dictionaries with source, target, and type
        """
        relationships: list[dict[str, Any]] = []

        class RelationshipVisitor(ast_module.NodeVisitor):
            """AST visitor for extracting basic relationships between entities."""

            def __init__(self) -> None:
                """Initialize the relationship visitor."""
                self.relationships: list[dict[str, Any]] = []

            def visit_ClassDef(self, node):
                """Visit class definition nodes to extract inheritance relationships."""
                class_name = node.name
                class_id = f"{file_path}::{class_name}"

                # Extract inheritance relationships
                for base in node.bases:
                    if isinstance(base, ast_module.Name):
                        # Basic inheritance (e.g., class MyClass(BaseClass))
                        base_name = base.id
                        relationships.append(
                            {
                                "source": class_id,
                                "target": base_name,  # Tentative target, needs resolution
                                "type": "INHERITS",
                                "tentative": True,
                            }
                        )
                    elif isinstance(base, ast_module.Attribute):
                        # Qualified inheritance (e.g., class MyClass(module.BaseClass))
                        base_name = self._get_attribute_full_name(base)
                        relationships.append(
                            {
                                "source": class_id,
                                "target": base_name,  # Tentative target, needs resolution
                                "type": "INHERITS",
                                "tentative": True,
                            }
                        )

                # Process class body for methods and attributes
                for item in node.body:
                    if isinstance(item, (ast_module.FunctionDef, ast_module.AsyncFunctionDef)):
                        method_name = item.name
                        method_id = f"{file_path}::{class_name}.{method_name}"

                        # Add CONTAINS relationship from class to method
                        relationships.append(
                            {
                                "source": class_id,
                                "target": method_id,
                                "type": "CONTAINS",
                            }
                        )

                # Continue visiting child nodes
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                """Visit async function definition nodes (same logic as sync)."""
                self.visit_FunctionDef(node)

            def visit_FunctionDef(self, node):
                """Visit function definition nodes to extract containment relationships."""
                # Check if this is a method (inside a class) or a standalone function
                parent_class = None
                for ancestor in ast_module.walk(ast):
                    if isinstance(ancestor, ast_module.ClassDef) and node in ancestor.body:
                        parent_class = ancestor.name
                        break

                # Skip if this is a method (already handled in visit_ClassDef)
                if parent_class:
                    pass
                else:
                    # This is a standalone function
                    func_name = node.name
                    func_id = f"{file_path}::{func_name}"

                    # Add MODULE_CONTAINS relationship from file to function
                    module_id = f"{file_path}"
                    relationships.append(
                        {
                            "source": module_id,
                            "target": func_id,
                            "type": "MODULE_CONTAINS",
                        }
                    )

                # Continue visiting child nodes
                self.generic_visit(node)

            def _get_attribute_full_name(self, node: ast_module.AST) -> str:
                """Helper to get the full name of an attribute node"""
                parts: list[str] = []
                curr = node
                while isinstance(curr, ast_module.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast_module.Name):
                    parts.append(curr.id)
                return ".".join(reversed(parts))

        # Run the visitor on the AST
        RelationshipVisitor().visit(ast)
        return relationships

    def _extract_function_calls(
        self, ast: ast_module.Module, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract function call relationships from Python code.

        This method identifies function calls within functions and methods.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file

        Returns:
            List of call relationship dictionaries
        """
        call_relationships: list[dict[str, Any]] = []

        class PythonCallVisitor(ast_module.NodeVisitor):
            """AST visitor for extracting function call relationships."""

            def __init__(self) -> None:
                """Initialize the call visitor."""
                self.current_class: str | None = None
                self.current_function: str | None = None
                self.call_relationships: list[dict[str, Any]] = []

            def visit_ClassDef(self, node):
                """Visit class definition nodes to track class context."""
                # Save the current class context
                old_class = self.current_class
                self.current_class = node.name

                # Visit all child nodes within this class context
                self.generic_visit(node)

                # Restore the previous class context
                self.current_class = old_class

            def visit_AsyncFunctionDef(self, node):
                """Visit async function definition nodes (same logic as sync)."""
                self.visit_FunctionDef(node)

            def visit_FunctionDef(self, node):
                """Visit function definition nodes to track function context."""
                # Save the current function context
                old_function = self.current_function
                self.current_function = node.name

                # Visit all child nodes within this function context
                self.generic_visit(node)

                # Restore the previous function context
                self.current_function = old_function

            def visit_Call(self, node):
                """Visit function call nodes to extract call relationships."""
                # Only process calls if we're inside a function/method
                if self.current_function:
                    # Get the caller ID
                    if self.current_class:
                        caller_id = f"{file_path}::{self.current_class}.{self.current_function}"
                    else:
                        caller_id = f"{file_path}::{self.current_function}"

                    # Extract the called function name
                    if isinstance(node.func, ast_module.Name):
                        # Basic function call (e.g., func())
                        called_name = node.func.id
                    elif isinstance(node.func, ast_module.Attribute):
                        # Method call or qualified function call
                        # (e.g., obj.method() or module.func())
                        called_name = self._extract_object_name(node.func)
                    else:
                        # Skip complex calls
                        self.generic_visit(node)
                        return

                    # Add CALLS relationship
                    call_relationships.append(
                        {
                            "source": caller_id,
                            "target": called_name,  # Tentative target, needs resolution
                            "type": "CALLS",
                            "tentative": True,
                        }
                    )

                # Continue visiting child nodes
                self.generic_visit(node)

            def _extract_object_name(self, node):
                """Extract name from an object expression"""
                if isinstance(node, ast_module.Name):
                    return node.id
                if isinstance(node, ast_module.Attribute):
                    obj_name = self._extract_object_name(node.value)
                    return f"{obj_name}.{node.attr}" if obj_name else node.attr
                return ""

        # Run the visitor on the AST
        PythonCallVisitor().visit(ast)
        return call_relationships

    def _extract_imports(self, ast: ast_module.Module, file_path: str) -> list[dict[str, Any]]:
        """Extract import relationships from Python code.

        This method identifies imports and creates relationships between the
        current file and imported modules.

        Args:
            ast_tree: The parsed AST tree
            file_path: Path to the source file

        Returns:
            List of import relationship dictionaries
        """
        # No need for local variable since we use visitor.relationships

        try:

            class ImportVisitor(ast_module.NodeVisitor):
                """AST visitor for extracting import relationships."""

                def __init__(self) -> None:
                    """Initialize the import visitor."""
                    self.relationships: list[dict[str, Any]] = []

                def visit_Import(self, node):
                    """Visit import nodes to extract import relationships."""
                    # Process regular imports (e.g., import module)
                    for name in node.names:
                        module_name = name.name
                        alias = name.asname if name.asname else module_name

                        try:
                            self.relationships.append(
                                {
                                    "source": file_path,
                                    "target": module_name,
                                    "type": "IMPORTS",
                                    "alias": alias,
                                }
                            )
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.error(
                                "Error processing import in %s: %s",
                                file_path,
                                str(e),
                                exc_info=True,
                            )

                def visit_ImportFrom(self, node):
                    """Visit from-import nodes to extract import relationships."""
                    # Process from-imports (e.g., from module import name)
                    if node.module:
                        module_name = node.module

                        # Handle relative imports
                        if node.level > 0:
                            # This is a relative import (e.g., from . import name)
                            # We need to resolve the actual module path
                            current_dir = os.path.dirname(file_path)
                            parent_dir = current_dir
                            for _ in range(node.level - 1):
                                parent_dir = os.path.dirname(parent_dir)

                            if module_name:
                                # from .module import name
                                module_path = os.path.join(
                                    parent_dir, module_name.replace(".", "/")
                                )
                            else:
                                # from . import name
                                module_path = parent_dir
                        else:
                            # This is an absolute import
                            module_path = module_name

                        for name in node.names:
                            imported_name = name.name
                            alias = name.asname if name.asname else imported_name

                            try:
                                self.relationships.append(
                                    {
                                        "source": file_path,
                                        "target": (
                                            f"{module_path}.{imported_name}"
                                            if imported_name != "*"
                                            else module_path
                                        ),
                                        "type": "IMPORTS",
                                        "alias": alias,
                                        "level": node.level,  # For relative imports
                                    }
                                )
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.error(
                                    "Error processing from-import in %s: %s",
                                    file_path,
                                    str(e),
                                    exc_info=True,
                                )

            # Run the visitor on the AST
            visitor = ImportVisitor()
            visitor.visit(ast)
            return visitor.relationships

        except (SyntaxError, ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Error extracting imports from %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )

        return []
