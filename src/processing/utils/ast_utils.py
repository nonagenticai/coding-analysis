"""AST utility functions for code analysis.

This module provides utilities for working with Abstract Syntax Trees (AST)
including entity ID generation, complexity calculation, and code extraction.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ASTUtils:
    """Shared AST utilities for code analysis

    Attributes:
        complexity_weights (Dict): Instance attribute complexity_weights."""

    def __init__(self) -> None:
        self.complexity_weights = {
            "i": 1,
            "for": 1,
            "while": 1,
            "except": 1,
            "with": 1,
            "and": 1,
            "or": 1,
            "lambda": 1,
            "assert": 1,
        }

    def generate_entity_id(
        self, entity_type: str, name: str, file_path: str, line_number: int = 0
    ) -> str:
        """Generate a consistent unique ID for an entity.

        Args:
            entity_type (str): The entity_type parameter.
            name (str): The name parameter.
            file_path (str): The file_path parameter.
            line_number (int, optional): The line_number parameter. Defaults to 0.

        Returns:
            str:     String ID that uniquely identifies the entity.

        Example:
            >>> generate_entity_id("entity_type_value", "example", "file_path_value", 42)
        """
        id_string = f"{entity_type}:{file_path}:{name}:{line_number}"
        return hashlib.sha256(id_string.encode("utf-8")).hexdigest()[:16]

    def generate_relationship_id(self, source_id: str, target_id: str, rel_type: str) -> str:
        """Generate a unique ID for a relationship.

        Args:
            source_id (str): The source_id parameter.
            target_id (str): The target_id parameter.
            rel_type (str): The rel_type parameter.

        Returns:
            str:     String ID that uniquely identifies the relationship.

        Example:
            >>> generate_relationship_id("example-id", "example-id", "rel_type_value")
        """
        id_string = f"{source_id}->{rel_type}->{target_id}"
        return hashlib.sha256(id_string.encode("utf-8")).hexdigest()[:16]

    def get_module_name(self, file_path: str) -> str:
        """Extract module name from file path.

        Args:
            file_path (str): The file_path parameter.

        Returns:
            str: The str result."""
        path = Path(file_path)
        if path.name == "__init__.py":
            return path.parent.name
        return path.stem

    def get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.class').

        Args:
            node (ast.Attribute): The node parameter.

        Returns:
            str: The str result."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        if isinstance(node.value, ast.Attribute):
            return f"{self.get_attribute_name(node.value)}.{node.attr}"
        return node.attr

    def extract_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module,
    ) -> str | None:
        """Extract docstring from function, class, or module.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module]): The node parameter.

        Returns:
            Optional[str]: The optional[str] result.

        Example:
            >>> extract_docstring("node_value")"""
        if not node.body:
            return None

        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            if isinstance(first_stmt.value.value, str):
                return first_stmt.value.value
        return None

    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a node.

        Args:
            node (ast.AST): The node parameter.

        Returns:
            int: The int result.

        Example:
            >>> calculate_cyclomatic_complexity("node_value")"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(
                child,
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                complexity += 1
                # Add complexity for conditions in comprehensions
                for generator in child.generators:
                    complexity += len(generator.ifs)

        return complexity

    def calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (more human-oriented than cyclomatic).

        Args:
            node (ast.AST): The node parameter.

        Returns:
            int: The int result.

        Example:
            >>> calculate_cognitive_complexity("node_value")"""
        complexity = 0

        def visit_node(node, level=0):
            """Function visit_node.

            Args:
                node: The node parameter.
                level: The level parameter. Defaults to 0.

            Example:
                >>> visit_node("node_value", "level_value")"""
            nonlocal complexity

            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1 + level
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1

            # Increase nesting for certain constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.ExceptHandler)):
                for child in ast.iter_child_nodes(node):
                    visit_node(child, level + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    visit_node(child, level)

        visit_node(node)
        return complexity

    def extract_function_parameters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[dict[str, Any]]:
        """Extract function parameters with type annotations.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The node parameter.

        Returns:
            List[Dict[str, Any]]: The list[dict[str, any]] result.

        Example:
            >>> extract_function_parameters("node_value")"""
        parameters: list[Any] = []
        # Regular arguments
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": "positional",
                "annotation": (
                    self._get_annotation_string(arg.annotation) if arg.annotation else None
                ),
                "default": None,
            }
            parameters.append(param_info)

        # Default values for regular arguments
        defaults = node.args.defaults
        if defaults:
            # Defaults apply to the last len(defaults) arguments
            for i, default in enumerate(defaults):
                param_index = len(node.args.args) - len(defaults) + i
                if param_index < len(parameters):
                    parameters[param_index]["default"] = self._get_node_value(default)

        # *args
        if node.args.vararg:
            parameters.append(
                {
                    "name": node.args.vararg.arg,
                    "type": "vararg",
                    "annotation": (
                        self._get_annotation_string(node.args.vararg.annotation)
                        if node.args.vararg.annotation
                        else None
                    ),
                    "default": None,
                }
            )

        # **kwargs
        if node.args.kwarg:
            parameters.append(
                {
                    "name": node.args.kwarg.arg,
                    "type": "kwarg",
                    "annotation": (
                        self._get_annotation_string(node.args.kwarg.annotation)
                        if node.args.kwarg.annotation
                        else None
                    ),
                    "default": None,
                }
            )

        return parameters

    def extract_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> list[str]:
        """Extract decorator names from function or class.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]): The node parameter.

        Returns:
            List[str]: The list[str] result.

        Example:
            >>> extract_decorators("node_value")"""
        decorators: list[Any] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self.get_attribute_name(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(f"{decorator.func.id}()")
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(f"{self.get_attribute_name(decorator.func)}()")
        return decorators

    def extract_base_classes(self, node: ast.ClassDef) -> list[str]:
        """Extract base class names from class definition.

        Args:
            node (ast.ClassDef): The node parameter.

        Returns:
            List[str]: The list[str] result."""
        bases: list[Any] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self.get_attribute_name(base))
        return bases

    def get_code_snippet(
        self,
        file_lines: list[str],
        start_line: int,
        end_line: int | None = None,
    ) -> str:
        """Extract code snippet from file lines.

        Args:
            file_lines (List[str]): The file_lines parameter.
            start_line (int): The start_line parameter.
            end_line (Optional[int], optional): The end_line parameter. Defaults to None.

        Returns:
            str: The str result.

        Example:
            >>> get_code_snippet("file_lines_value", "start_line_value", "end_line_value")
        """
        if not file_lines or start_line < 1:
            return ""

        start_idx = start_line - 1  # Convert to 0-based index
        end_idx = end_line if end_line else start_line

        if start_idx >= len(file_lines):
            return ""

        end_idx = min(end_idx, len(file_lines))
        return "\n".join(file_lines[start_idx:end_idx])

    def add_parent_links(self, tree: ast.AST) -> None:
        """Add parent links to AST nodes for context tracking.

        Args:
            tree (ast.AST): The tree parameter.

        Returns:
            None: The none result."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                # Use setattr to dynamically add parent_node attribute
                # ty doesn't know about this dynamic attribute, so we ignore
                # the type check
                setattr(child, "parent_node", parent)

    def _get_annotation_string(self, annotation: ast.expr) -> str:
        """Convert annotation AST node to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Attribute):
            return self.get_attribute_name(annotation)
        if isinstance(annotation, ast.Constant):
            return str(annotation.value)
        if isinstance(annotation, ast.Subscript):
            # Handle generic types like List[str], Dict[str, int]
            value = self._get_annotation_string(annotation.value)
            slice_val = annotation.slice
            if isinstance(slice_val, ast.Tuple):
                slice_parts = [self._get_annotation_string(elt) for elt in slice_val.elts]
                return f"{value}[{', '.join(slice_parts)}]"
            slice_str = self._get_annotation_string(slice_val)
            return f"{value}[{slice_str}]"
        return "Unknown"

    def _get_node_value(self, node: ast.expr | None) -> Any:
        """Extract value from AST node."""
        if node is None:
            return None

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return [self._get_node_value(elt) for elt in node.elts]
        if isinstance(node, ast.Dict):
            return self._extract_dict_value(node)
        return None

    def _extract_dict_value(self, node: ast.Dict) -> dict[str, Any]:
        """Extract dictionary value from AST Dict node."""
        result: dict[str, Any] = {}
        for k, v in zip(node.keys, node.values):
            # Handle None keys gracefully
            key = self._get_node_value(k) if k is not None else None
            value = self._get_node_value(v)
            # Only add to dict if key is not None and can be converted to string
            if key is not None:
                try:
                    result[str(key)] = value
                except (TypeError, ValueError):  # fmt: skip
                    # Skip keys that can't be converted to string
                    continue
        return result
