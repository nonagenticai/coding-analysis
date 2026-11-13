"""Module for analyzing class structures in code."""

from __future__ import annotations

import ast
from typing import Any

import astor


class ClassAnalyzer:
    """Analyzes class structures and relationships in Python code."""

    def extract_class_structure(self, file_path: str) -> dict[str, dict[str, Any]]:
        """
        Extract class definitions with methods, attributes, and inheritance.

        Returns:
            {
                "class_name": {
                    "methods": ["method1", "method2"],
                    "attributes": ["attr1", "attr2"],
                    "bases": ["BaseClass1", "BaseClass2"],
                    "file": "path/to/file.py",
                    "line": 10
                }
            }
        """
        with open(file_path) as f:
            code = f.read()

        try:
            tree = ast.parse(code)
            classes: dict[str, Any] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract base classes using astor
                    bases = [astor.to_source(base).strip() for base in node.bases]

                    # Extract methods and attributes
                    methods: list[Any] = []
                    attributes: list[Any] = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attributes.append(target.id)

                    classes[node.name] = {
                        "methods": methods,
                        "attributes": attributes,
                        "bases": bases,
                        "file": file_path,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                    }

            return classes
        except SyntaxError:
            return {
                "__error__": {
                    "type": "SyntaxError",
                    "message": "Syntax error in file",
                    "file": file_path,
                    "line": 0,
                    "methods": [],
                    "attributes": [],
                    "bases": [],
                    "docstring": None,
                }
            }
