"""Module for analyzing code complexity metrics."""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """Analyzes code complexity metrics for various programming languages.

    This class provides functionality to calculate complexity metrics such as
    cyclomatic complexity, nesting depth, and structural information for
    different programming languages including Python, JavaScript, TypeScript,
    and Java.
    """

    def __init__(self) -> None:
        """Initialize the complexity analyzer with empty cache."""
        self.complexity_cache: dict[str, dict[str, Any]] = {}

    def calculate_complexity(self, file_path: Path, content: str, language: str) -> dict[str, Any]:
        """Calculate complexity metrics for a file.

        Analyzes the code content to extract various complexity metrics including
        cyclomatic complexity, nesting depth, and structural information.

        Args:
            file_path: Path to the file being analyzed
            content: String containing the file's source code
            language: Programming language of the file content

        Returns:
            Dictionary containing various complexity metrics including:
            - cyclomatic_complexity: Overall complexity score
            - max_nesting_depth: Maximum nesting level in the code
            - function_count: Number of functions/methods
            - class_count: Number of classes
            - import_count: Number of import statements
            - loc: Lines of code
            - average_function_complexity: Average complexity per function
        """
        cache_key = f"{file_path}:{hashlib.md5(content.encode()).hexdigest()}"  # nosec B324

        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]

        metrics = {
            "cyclomatic_complexity": 0,
            "max_nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "loc": 0,
            "average_function_complexity": 0.0,
        }

        if not content:
            return metrics

        if language == "python":
            metrics = self._analyze_python_complexity(content)
        elif language in ["javascript", "typescript"]:
            metrics = self._analyze_javascript_complexity(content)
        elif language == "java":
            metrics = self._analyze_java_complexity(content)
        else:
            # Generic analysis for unknown languages
            metrics = self._analyze_generic_complexity(content)

        self.complexity_cache[cache_key] = metrics
        return metrics

    def _analyze_python_complexity(self, content: str) -> dict[str, Any]:
        """Analyze Python code complexity using AST parsing.

        Uses AST parsing to analyze Python code and extract complexity metrics
        by traversing the abstract syntax tree with a specialized visitor.

        Args:
            content: String containing Python source code

        Returns:
            Dictionary containing various complexity metrics specific to Python code
        """
        try:
            tree = ast.parse(content)

            class ComplexityVisitor(ast.NodeVisitor):
                """AST visitor for analyzing Python code complexity."""

                def __init__(self):
                    """Initialize the complexity visitor with depth tracking."""
                    self.current_depth = 0
                    self.max_depth = 0
                    self.total_complexity = 0
                    self.function_count = 0
                    self.class_count = 0
                    self.import_count = 0

                def visit_FunctionDef(self, node):
                    """Visit function definitions and analyze their complexity."""
                    self.function_count += 1
                    self.current_depth += 1
                    if self.current_depth > self.max_depth:
                        self.max_depth = self.current_depth

                    # Count branches (if/elif/else, for, while, try/except)
                    branch_count = 0
                    for child in ast.walk(node):
                        if isinstance(
                            child,
                            (
                                ast.If,
                                ast.For,
                                ast.While,
                                ast.Try,
                                ast.With,
                                ast.Match,
                            ),
                        ):
                            branch_count += 1

                    # Store function complexity
                    self.total_complexity += branch_count + 1
                    self.generic_visit(node)
                    self.current_depth -= 1

                def visit_AsyncFunctionDef(self, node):
                    """Visit async function definitions."""
                    self.visit_FunctionDef(node)

                def visit_ClassDef(self, node):
                    """Visit class definitions and process their contents."""
                    self.class_count += 1
                    self.generic_visit(node)

                def visit_Import(self, node):
                    """Visit import statements."""
                    self.import_count += 1

                def visit_ImportFrom(self, node):
                    """Visit import-from statements."""
                    self.import_count += 1

            visitor = ComplexityVisitor()
            visitor.visit(tree)

            # Count lines of code (excluding blank lines and comments)
            loc = len(
                [
                    line
                    for line in content.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
            )

            # Calculate average complexity per function
            avg_complexity = (
                visitor.total_complexity / visitor.function_count
                if visitor.function_count > 0
                else 0.0
            )

            return {
                "cyclomatic_complexity": visitor.total_complexity,
                "max_nesting_depth": visitor.max_depth,
                "function_count": visitor.function_count,
                "class_count": visitor.class_count,
                "import_count": visitor.import_count,
                "loc": loc,
                "average_function_complexity": avg_complexity,
            }
        except Exception as e:
            logger.warning("Failed to analyze Python complexity: %s", e)
            return self._get_default_metrics()

    def _analyze_javascript_complexity(self, content: str) -> dict[str, Any]:
        """Analyze JavaScript/TypeScript code complexity using regex heuristics.

        Uses regex-based heuristics to analyze JavaScript or TypeScript code
        complexity, counting control structures, functions, classes, and imports.

        Args:
            content: String containing JavaScript/TypeScript source code

        Returns:
            Dictionary containing various complexity metrics
        """
        try:
            # Simple heuristic-based complexity analysis for JS/TS
            lines = content.split("\n")
            loc = len(
                [line for line in lines if line.strip() and not line.strip().startswith("//")]
            )

            # Count functions (including arrow functions)
            function_count = len(re.findall(r"function\s+\w+\s*\(", content))
            function_count += len(re.findall(r"\w+\s*=\s*function\s*\(", content))
            function_count += len(re.findall(r"\(.*\)\s*=>", content))

            # Count classes
            class_count = len(re.findall(r"class\s+\w+", content))

            # Count imports
            import_count = len(re.findall(r"import\s+.*\s+from\s+['\"].*['\"];", content))
            import_count += len(re.findall(r"require\(['\"].*['\"]\)", content))

            # Estimate complexity based on control structures
            complexity = len(re.findall(r"if\s*\(", content))
            complexity += len(re.findall(r"else\s*\{", content))
            complexity += len(re.findall(r"for\s*\(", content))
            complexity += len(re.findall(r"while\s*\(", content))
            complexity += len(re.findall(r"switch\s*\(", content))
            complexity += len(re.findall(r"\?.*:\s*", content))  # Ternary operators

            # Estimate nesting depth
            max_depth = 0
            current_depth = 0
            for line in lines:
                current_depth += line.count("{") - line.count("}")
                max_depth = max(max_depth, current_depth)

            return {
                "cyclomatic_complexity": complexity,
                "max_nesting_depth": max_depth,
                "function_count": function_count,
                "class_count": class_count,
                "import_count": import_count,
                "loc": loc,
                "average_function_complexity": (
                    complexity / function_count if function_count > 0 else 0.0
                ),
            }
        except Exception as e:
            logger.warning("Failed to analyze JavaScript complexity: %s", e)
            return self._get_default_metrics()

    def _analyze_java_complexity(self, content: str) -> dict[str, Any]:
        """Analyze Java code complexity using regex heuristics.

        Args:
            content: String containing Java source code

        Returns:
            Dictionary containing various complexity metrics specific to Java
        """
        # For now, use similar logic to JavaScript analysis
        # Could be enhanced with Java-specific patterns
        return self._analyze_javascript_complexity(content)

    def _analyze_generic_complexity(self, content: str) -> dict[str, Any]:
        """Analyze code complexity for unknown languages using basic heuristics.

        Args:
            content: String containing source code

        Returns:
            Dictionary containing basic complexity metrics
        """
        try:
            lines = content.split("\n")
            loc = len([line for line in lines if line.strip()])

            # Basic complexity estimation based on common patterns
            complexity = content.count("if ") + content.count("while ") + content.count("for ")
            complexity += content.count("switch") + content.count("case")

            # Basic nesting estimation
            max_depth = 0
            current_depth = 0
            for line in lines:
                # Count common nesting characters
                current_depth += line.count("{") - line.count("}")
                current_depth += line.count("(") - line.count(")")
                max_depth = max(max_depth, current_depth)

            return {
                "cyclomatic_complexity": complexity,
                "max_nesting_depth": max_depth,
                "function_count": 0,  # Hard to estimate without language specifics
                "class_count": 0,
                "import_count": 0,
                "loc": loc,
                "average_function_complexity": 0.0,
            }
        except Exception as e:
            logger.warning("Failed to analyze generic complexity: %s", e)
            return self._get_default_metrics()

    def _get_default_metrics(self) -> dict[str, Any]:
        """Return default complexity metrics when analysis fails.

        Returns:
            Dictionary with default (zero) values for all complexity metrics
        """
        return {
            "cyclomatic_complexity": 0,
            "max_nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "loc": 0,
            "average_function_complexity": 0.0,
        }

    def clear_cache(self) -> None:
        """Clear the complexity analysis cache."""
        self.complexity_cache.clear()

    def get_cache_size(self) -> int:
        """Get the current size of the complexity cache.

        Returns:
            Number of cached complexity analyses
        """
        return len(self.complexity_cache)
