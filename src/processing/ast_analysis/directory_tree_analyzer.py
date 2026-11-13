"""
Directory Tree Analyzer (Migrated from services/ast_analysis)

Integrates directory tree analysis with AST parsing for comprehensive code structure analysis.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DirectoryTreeAnalyzer:
    """Integrates directory structure analysis with AST parsing."""

    def __init__(self) -> None:
        self.directory_structure: dict[str, Any] = {}
        self.file_analysis: dict[str, dict[str, Any]] = {}

    def analyze_directory(
        self, directory_path: Path, extensions: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Analyze directory structure and integrate with AST analysis.

        Args:
            directory_path: Path to directory to analyze
            extensions: File extensions to include (default: ['.py'])

        Returns:
            Comprehensive directory and code analysis
        """
        if extensions is None:
            extensions = [".py"]

        try:
            self.directory_structure = self._build_directory_tree(directory_path)

            # Analyze each Python file
            for file_path in directory_path.rglob("*"):
                if file_path.suffix in extensions and file_path.is_file():
                    self.file_analysis[str(file_path)] = self._analyze_file_ast(file_path)

            return {
                "directory_structure": self.directory_structure,
                "file_analysis": self.file_analysis,
                "summary": self._generate_summary(),
            }

        except Exception as e:
            logger.error("Error analyzing directory %s: %s", directory_path, e)
            return {}

    def _build_directory_tree(self, path: Path) -> dict[str, Any]:
        """Build directory tree structure."""
        tree: dict[str, Any] = {
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "path": str(path),
            "children": [],
        }

        if path.is_dir():
            try:
                for child in sorted(path.iterdir()):
                    tree["children"].append(self._build_directory_tree(child))
            except PermissionError:
                pass  # Skip directories we can't access

        return tree

    def _analyze_file_ast(self, file_path: Path) -> dict[str, Any]:
        """Analyze individual file with AST."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            analysis: dict[str, Any] = {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions_list = analysis["functions"]
                    if isinstance(functions_list, list):
                        functions_list.append(
                            {
                                "name": node.name,
                                "line_number": node.lineno,
                                "is_async": isinstance(node, ast.AsyncFunctionDef),
                                "is_private": node.name.startswith("_"),
                            }
                        )
                elif isinstance(node, ast.ClassDef):
                    classes_list = analysis["classes"]
                    if isinstance(classes_list, list):
                        classes_list.append(
                            {
                                "name": node.name,
                                "line_number": node.lineno,
                                "bases": [
                                    base.id for base in node.bases if isinstance(base, ast.Name)
                                ],
                                "is_private": node.name.startswith("_"),
                            }
                        )
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports_list = analysis["imports"]
                            if isinstance(imports_list, list):
                                imports_list.append(
                                    {
                                        "module": alias.name,
                                        "alias": alias.asname,
                                        "type": "import",
                                    }
                                )
                    else:  # ImportFrom
                        for alias in node.names:
                            imports_list = analysis["imports"]
                            if isinstance(imports_list, list):
                                imports_list.append(
                                    {
                                        "module": node.module,
                                        "name": alias.name,
                                        "alias": alias.asname,
                                        "type": "from_import",
                                    }
                                )

            # Basic complexity calculation
            # Calculate complexity score
            functions_count = (
                len(analysis.get("functions", []))
                if isinstance(analysis.get("functions"), list)
                else 0
            )
            classes_count = (
                len(analysis.get("classes", [])) if isinstance(analysis.get("classes"), list) else 0
            )
            analysis["complexity_score"] = functions_count + classes_count * 2

            return analysis

        except Exception as e:
            logger.error("Error analyzing %s: %s", file_path, e)
            return {}

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total_functions = sum(
            len(analysis.get("functions", [])) for analysis in self.file_analysis.values()
        )
        total_classes = sum(
            len(analysis.get("classes", [])) for analysis in self.file_analysis.values()
        )
        total_files = len(self.file_analysis)

        return {
            "total_files_analyzed": total_files,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "average_functions_per_file": total_functions / max(1, total_files),
            "average_classes_per_file": total_classes / max(1, total_files),
        }
