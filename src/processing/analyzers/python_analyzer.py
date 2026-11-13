"""
Python code processor for comprehensive analysis of Python source files.

This module provides a unified processor that consolidates functionality from
python_processor and advanced_ast_analyzer to perform comprehensive Python
code analysis including complexity metrics, security pattern detection,
and structural analysis.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

from models.core_models import CodeSecurityPattern as SecurityPattern
from models.core_models import (
    CoreAnalysisResult,
    CoreComplexityMetrics,
)

from ..extractors.dependency_analyzer import CodeDependencyExtractor
from ..extractors.entity_extractor import EntityExtractor
from ..extractors.relationship_extractor import RelationshipExtractor
from ..utils.ast_utils import ASTUtils
from .base_analyzer import BaseCodeProcessor

logger = logging.getLogger(__name__)


class PythonCodeProcessor(BaseCodeProcessor):
    """Unified Python code processor for comprehensive Python source code analysis.

    This processor consolidates functionality from python_processor and advanced_ast_analyzer
    to provide a complete Python code analysis solution. It uses the Python AST (Abstract
    Syntax Tree) to perform deep analysis including:

    - Entity extraction (classes, functions, variables, imports)
    - Relationship detection (inheritance, calls, references)
    - Dependency analysis (internal and external imports)
    - Complexity metrics (cyclomatic, cognitive, nesting depth)
    - Security pattern detection (dangerous functions, hardcoded secrets)
    - Code quality metrics (docstring coverage, comment ratio)

    The processor is designed to handle both individual files and entire directories,
    providing aggregated metrics for project-level analysis.

    Attributes:
        entity_extractor: Extracts code entities from AST
        relationship_extractor: Identifies relationships between entities
        dependency_analyzer: Analyzes import statements and dependencies
        ast_utils: Utility functions for AST manipulation and analysis
    """

    def __init__(self) -> None:
        """Initialize the Python analyzer with required components.

        Sets up extractors and utilities needed for comprehensive Python code analysis.
        All components are initialized without external dependencies, making the
        analyzer ready for immediate use.
        """
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.dependency_analyzer = CodeDependencyExtractor()
        self.ast_utils = ASTUtils()

    def get_supported_extensions(self) -> list[str]:
        """Return list of supported Python file extensions.

        Returns:
            List[str]: List containing '.py' (standard Python) and '.pyx' (Cython) extensions.
        """
        return [".py", ".pyx"]

    def get_language_name(self) -> str:
        """Return the canonical language name.

        Returns:
            str: Always returns 'python' for this analyzer.
        """
        return "python"

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Perform comprehensive analysis on a Python source file.

        This method orchestrates the complete analysis pipeline:
        1. Reads and parses the Python source code into an AST
        2. Adds parent links to the AST for context tracking
        3. Extracts entities (classes, functions, variables, imports)
        4. Identifies relationships between entities
        5. Analyzes dependencies and imports
        6. Calculates complexity metrics
        7. Detects potential security patterns
        8. Computes file-level metrics

        Args:
            file_path (str): Absolute path to the Python file to analyze.
                           Must be a valid path to an existing .py or .pyx file.

        Returns:
            CoreAnalysisResult: Comprehensive analysis results including:
                - All extracted entities and their properties
                - Relationships between entities
                - Import dependencies
                - Complexity metrics (cyclomatic, cognitive, etc.)
                - Security patterns and potential vulnerabilities
                - File metrics and metadata

        Raises:
            The method catches all exceptions and returns them in the CoreAnalysisResult:
            - SyntaxError: For Python syntax errors (includes line/column info)
            - OSError/IOError: For file access issues
            - Exception: For any other analysis errors
        """
        try:
            # Read and parse file
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            file_lines = content.splitlines()
            tree = ast.parse(content, filename=file_path)

            # Add parent links for context tracking
            self.ast_utils.add_parent_links(tree)

            # Extract all components using single AST walk
            entities = await self.entity_extractor.extract_python_entities(
                tree, file_path, file_lines
            )
            import_entities = await self.entity_extractor.extract_import_entities(tree, file_path)
            entities.extend(import_entities)

            relationships = await self.relationship_extractor.extract_relationships(tree, entities)
            dependencies = await self.dependency_analyzer.analyze_imports(tree, file_path)

            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(tree, content)

            # Detect security patterns
            security_patterns = self._detect_security_patterns(tree)

            # Calculate file-level metrics
            file_metrics = self._calculate_file_metrics(tree, content, file_path)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=len(file_lines),
                entities=entities,
                relationships=relationships,
                dependencies=dependencies,
                complexity_metrics=complexity_metrics,
                security_patterns=security_patterns,
                error=None,
                metadata={
                    "file_path": file_path,
                    "language": "python",
                    "file_metrics": file_metrics,
                    "line_count": len(file_lines),
                    "file_size": (os.path.getsize(file_path) if os.path.exists(file_path) else 0),
                },
            )

        except SyntaxError as e:
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=0,
                error=f"Syntax error in {file_path}: {e}",
                complexity_metrics=None,
                metadata={
                    "file_path": file_path,
                    "error_type": "syntax_error",
                    "line_number": e.lineno,
                    "offset": e.offset,
                },
            )
        except OSError as e:
            logger.error("Error reading file %s: %s", file_path, e)
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=0,
                error=f"File access error: {e}",
                complexity_metrics=None,
                metadata={
                    "file_path": file_path,
                    "error_type": "file_access_error",
                },
            )
        except Exception as e:
            logger.error("Unexpected error analyzing %s: %s", file_path, e)
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=0,
                error=f"Analysis error: {e}",
                complexity_metrics=None,
                metadata={
                    "file_path": file_path,
                    "error_type": "analysis_error",
                },
            )

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze all Python files in a directory recursively.

        Recursively finds and analyzes all .py files in the specified directory
        and its subdirectories. Results are aggregated to provide directory-level
        insights including total complexity metrics, all entities across files,
        and cross-file relationships.

        Args:
            directory_path (str): Absolute path to the directory to analyze.
                                Must be a valid directory path.

        Returns:
            CoreAnalysisResult: Aggregated analysis results containing:
                - Combined entities from all analyzed files
                - Cross-file relationships
                - Aggregated complexity metrics
                - Directory-level statistics
                - Per-file result summaries in metadata

        Note:
            - Skips files that fail to analyze and logs warnings
            - Continues analysis even if individual files fail
            - Aggregates metrics by summing file-level values
        """
        all_entities: list[Any] = []
        all_relationships: list[Any] = []
        all_dependencies: list[Any] = []
        all_security_patterns: list[Any] = []
        file_results: list[Any] = []
        directory = Path(directory_path)
        python_files = list(directory.rglob("*.py"))

        for file_path in python_files:
            try:
                result = await self.analyze_file(str(file_path))
                if result.success:
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                    all_dependencies.extend(result.dependencies)
                    all_security_patterns.extend(result.security_patterns)
                    file_results.append(
                        {
                            "file_path": str(file_path),
                            "entity_count": len(result.entities),
                            "relationship_count": len(result.relationships),
                            "dependency_count": len(result.dependencies),
                            "complexity": result.complexity_metrics,
                        }
                    )
                else:
                    logger.warning("Failed to analyze %s: %s", file_path, result.error)
            except OSError as e:
                logger.error("Error accessing file %s: %s", file_path, e)
            except Exception as e:
                logger.error("Unexpected error processing %s: %s", file_path, e)

        # Calculate directory-level metrics
        total_complexity = CoreComplexityMetrics(
            cyclomatic_complexity=0,
            cognitive_complexity=0,
            nested_depth=0,
            function_count=0,
            class_count=0,
            lines_of_code=0,
            comment_lines=0,
            blank_lines=0,
            docstring_coverage=0.0,
        )
        for result in file_results:
            if result.complexity_metrics:
                complexity = result.complexity_metrics
                total_complexity.cyclomatic_complexity += complexity.cyclomatic_complexity
                total_complexity.cognitive_complexity += complexity.cognitive_complexity
                total_complexity.function_count += complexity.function_count
                total_complexity.class_count += complexity.class_count
                total_complexity.lines_of_code += complexity.lines_of_code

        return CoreAnalysisResult(
            success=True,
            language="python",
            file_path=directory_path,
            parse_time_ms=0.0,
            total_lines=sum(
                result.get("complexity_metrics", {}).get("lines_of_code", 0)
                for result in file_results
            ),
            entities=all_entities,
            relationships=all_relationships,
            dependencies=all_dependencies,
            complexity_metrics=total_complexity,
            security_patterns=all_security_patterns,
            error=None,
            metadata={
                "directory_path": directory_path,
                "language": "python",
                "file_count": len(python_files),
                "analyzed_files": len(file_results),
                "file_results": file_results,
            },
        )

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a Python code snippet without requiring a file.

        Performs the same comprehensive analysis as analyze_file but on a string
        containing Python code. Useful for analyzing dynamically generated code,
        code from databases, or for real-time analysis in editors.

        Args:
            code_snippet (str): Python source code to analyze as a string.
                              Must be valid Python syntax.
            file_path (Optional[str]): Optional file path for context.
                                     Used for import resolution and error
                                     reporting. Defaults to 'snippet'.

        Returns:
            CoreAnalysisResult: Complete analysis results equivalent to file analysis,
                          with metadata indicating snippet source.

        Raises:
            Exceptions are caught and returned in CoreAnalysisResult:
            - SyntaxError: For invalid Python syntax in the snippet
            - Exception: For any other analysis errors
        """
        try:
            tree = ast.parse(code_snippet)
            snippet_lines = code_snippet.splitlines()

            # Add parent links
            self.ast_utils.add_parent_links(tree)

            # Analyze snippet as if it were a file
            entities = await self.entity_extractor.extract_python_entities(
                tree, file_path or "snippet", snippet_lines
            )
            import_entities = await self.entity_extractor.extract_import_entities(
                tree, file_path or "snippet"
            )
            entities.extend(import_entities)

            relationships = await self.relationship_extractor.extract_relationships(tree, entities)
            dependencies = await self.dependency_analyzer.analyze_imports(
                tree, file_path or "snippet"
            )

            # Calculate metrics for snippet
            complexity_metrics = self._calculate_complexity_metrics(tree, code_snippet)
            security_patterns = self._detect_security_patterns(tree)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=file_path or "snippet",
                parse_time_ms=0.0,
                total_lines=len(snippet_lines),
                entities=entities,
                relationships=relationships,
                dependencies=dependencies,
                complexity_metrics=complexity_metrics,
                security_patterns=security_patterns,
                error=None,
                metadata={
                    "source": "snippet",
                    "language": "python",
                    "line_count": len(snippet_lines),
                },
            )

        except SyntaxError as e:
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path or "snippet",
                parse_time_ms=0.0,
                total_lines=0,
                error=f"Syntax error in snippet: {e}",
                complexity_metrics=None,
                metadata={
                    "source": "snippet",
                    "error_type": "syntax_error",
                    "line_number": e.lineno,
                    "offset": e.offset,
                },
            )
        except Exception as e:
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path or "snippet",
                parse_time_ms=0.0,
                total_lines=0,
                error=f"Snippet analysis error: {e}",
                complexity_metrics=None,
                metadata={
                    "source": "snippet",
                    "error_type": "snippet_analysis_error",
                },
            )

    def _calculate_complexity_metrics(self, tree: ast.AST, content: str) -> CoreComplexityMetrics:
        """Calculate comprehensive complexity metrics for Python code.

        Analyzes the AST to compute various complexity metrics that indicate
        code maintainability and potential problem areas.

        Args:
            tree (ast.AST): Parsed AST of the Python code.
            content (str): Original source code content for line analysis.

        Returns:
            CoreComplexityMetrics: Object containing:
                - cyclomatic_complexity: Total cyclomatic complexity
                - cognitive_complexity: Total cognitive complexity
                - nested_depth: Maximum nesting depth found
                - function_count: Number of function definitions
                - class_count: Number of class definitions
                - lines_of_code: Non-empty lines
                - comment_lines: Lines starting with #
                - blank_lines: Empty lines
                - docstring_coverage: Percentage of documented functions/classes
        """
        lines = content.splitlines()

        # Calculate all values first
        cyclomatic_complexity = 0
        cognitive_complexity = 0
        nested_depth = 0
        function_count = 0
        class_count = 0

        # Calculate complexity for each function and class
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                cyclomatic_complexity += self.ast_utils.calculate_cyclomatic_complexity(node)
                cognitive_complexity += self.ast_utils.calculate_cognitive_complexity(node)
            elif isinstance(node, ast.ClassDef):
                class_count += 1
                # Calculate nested depth
                depth = self._calculate_nesting_depth(node)
                nested_depth = max(nested_depth, depth)

        # Create metrics object with all calculated values
        metrics = CoreComplexityMetrics(
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            nested_depth=nested_depth,
            function_count=function_count,
            class_count=class_count,
            lines_of_code=len([line for line in lines if line.strip()]),
            comment_lines=len([line for line in lines if line.strip().startswith("#")]),
            blank_lines=len([line for line in lines if not line.strip()]),
            docstring_coverage=self._calculate_docstring_coverage(tree),
        )

        return metrics

    def _detect_security_patterns(self, tree: ast.AST) -> list[SecurityPattern]:
        """Detect potential security vulnerabilities in Python code.

        Walks the AST to identify security-sensitive patterns including:
        - Use of dangerous functions (eval, exec)
        - Hardcoded secrets and passwords
        - Unsafe subprocess calls
        - User input without validation

        Args:
            tree (ast.AST): Parsed AST to analyze for security patterns.

        Returns:
            List[SecurityPattern]: List of detected security issues with:
                - Pattern type and severity
                - Line number location
                - Description and remediation suggestions
                - Associated CWE IDs where applicable
        """
        security_patterns: list[Any] = []
        for node in ast.walk(tree):
            patterns = self._check_node_for_security_issues(node)
            security_patterns.extend(patterns)
        return security_patterns

    def _check_node_for_security_issues(self, node: ast.AST) -> list[SecurityPattern]:
        """Check a single AST node for security vulnerabilities.

        Examines an individual AST node for security-sensitive patterns.
        Delegates to specific checking methods based on node type.

        Args:
            node (ast.AST): AST node to examine for security issues.

        Returns:
            List[SecurityPattern]: Security issues found in this node.
        """
        patterns: list[SecurityPattern] = []

        if isinstance(node, ast.Call):
            patterns.extend(self._check_function_call_security(node))
        elif isinstance(node, ast.Assign):
            patterns.extend(self._check_assignment_security(node))

        return patterns

    def _check_function_call_security(self, node: ast.Call) -> list[SecurityPattern]:
        """Check function calls for security vulnerabilities.

        Analyzes function call nodes for dangerous or security-sensitive functions
        including eval, exec, input functions, and subprocess calls.

        Args:
            node (ast.Call): Function call AST node to analyze.

        Returns:
            List[SecurityPattern]: Security patterns found in the function call,
                                 including severity level and remediation advice.
        """
        patterns: list[SecurityPattern] = []

        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ["eval", "exec"]:
                patterns.append(
                    SecurityPattern(
                        type="dangerous_function",
                        severity="high",
                        line_number=node.lineno,
                        description=f"Use of dangerous function '{func_name}'",
                        cwe_id=None,
                        metadata={
                            "function": func_name,
                            "suggestion": f"Avoid using {func_name} with untrusted input",
                        },
                    )
                )
            elif func_name in ["input", "raw_input"]:
                patterns.append(
                    SecurityPattern(
                        type="user_input",
                        severity="medium",
                        line_number=node.lineno,
                        description=f"User input function '{func_name}' detected",
                        cwe_id=None,
                        metadata={
                            "function": func_name,
                            "suggestion": "Validate and sanitize user input",
                        },
                    )
                )

        # Check for subprocess calls
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                patterns.append(
                    SecurityPattern(
                        type="subprocess_call",
                        severity="medium",
                        line_number=node.lineno,
                        description="Subprocess call detected",
                        cwe_id=None,
                        metadata={
                            "method": node.func.attr,
                            "suggestion": "Ensure proper input validation for subprocess calls",
                        },
                    )
                )

        return patterns

    def _check_assignment_security(self, node: ast.Assign) -> list[SecurityPattern]:
        """Check variable assignments for hardcoded secrets.

        Detects potential hardcoded credentials by examining variable names
        containing keywords like 'password', 'secret', 'key', or 'token'
        that are assigned string literals.

        Args:
            node (ast.Assign): Assignment AST node to analyze.

        Returns:
            List[SecurityPattern]: Detected hardcoded secret patterns with
                                 recommendations to use environment variables
                                 or secure configuration management.
        """
        patterns: list[SecurityPattern] = []

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            var_name = target.id.lower()
            if not any(keyword in var_name for keyword in ["password", "secret", "key", "token"]):
                continue

            if not isinstance(node.value, ast.Constant):
                continue

            value = str(node.value.value)

            if len(value) > 8:  # Likely not a placeholder
                patterns.append(
                    SecurityPattern(
                        type="hardcoded_secret",
                        severity="high",
                        line_number=node.lineno,
                        description=f"Potential hardcoded secret in variable '{target.id}'",
                        cwe_id=None,
                        metadata={
                            "variable": target.id,
                            "suggestion": "Use environment variables or secure configuration",
                        },
                    )
                )

        return patterns

    def _calculate_file_metrics(
        self, tree: ast.AST, content: str, file_path: str
    ) -> dict[str, Any]:
        """Calculate comprehensive file-level metrics.

        Computes various metrics about the file structure and content
        to provide insights into code organization and style.

        Args:
            tree (ast.AST): Parsed AST of the file.
            content (str): Original file content.
            file_path (str): Path to the file being analyzed.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - statement_counts: Breakdown by statement type
                - total_lines: Total line count
                - non_empty_lines: Lines with content
                - comment_ratio: Percentage of comment lines
                - file_size_bytes: File size in bytes
                - is_package: Whether this is an __init__.py file
                - has_main_guard: Whether the file has if __name__ == "__main__"
        """
        lines = content.splitlines()

        # Count different types of statements
        statement_counts = {
            "imports": 0,
            "classes": 0,
            "functions": 0,
            "assignments": 0,
            "if_statements": 0,
            "loops": 0,
            "try_blocks": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                statement_counts["imports"] += 1
            elif isinstance(node, ast.ClassDef):
                statement_counts["classes"] += 1
            elif isinstance(node, ast.FunctionDef):
                statement_counts["functions"] += 1
            elif isinstance(node, ast.Assign):
                statement_counts["assignments"] += 1
            elif isinstance(node, ast.If):
                statement_counts["if_statements"] += 1
            elif isinstance(node, (ast.For, ast.While)):
                statement_counts["loops"] += 1
            elif isinstance(node, ast.Try):
                statement_counts["try_blocks"] += 1

        return {
            "statement_counts": statement_counts,
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_ratio": (
                len([line for line in lines if line.strip().startswith("#")]) / len(lines)
                if lines
                else 0
            ),
            "file_size_bytes": len(content.encode("utf-8")),
            "is_package": file_path.endswith("__init__.py"),
            "has_main_guard": '__name__ == "__main__"' in content,
        }

    def _calculate_docstring_coverage(self, tree: ast.AST) -> float:
        """Calculate the percentage of documented functions and classes.

        Walks the AST to count functions and classes, checking how many
        have docstrings to compute documentation coverage.

        Args:
            tree (ast.AST): Parsed AST to analyze.

        Returns:
            float: Percentage (0-100) of functions and classes that have
                  docstrings. Returns 0 if no functions or classes exist.
        """
        total_items = 0
        documented_items = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_items += 1
                if self.ast_utils.extract_docstring(node):
                    documented_items += 1

        return (documented_items / total_items * 100) if total_items > 0 else 0

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate the maximum nesting depth within an AST node.

        Recursively traverses the AST to find the deepest level of nesting
        created by control flow structures (if, for, while, with, try).
        High nesting depth indicates complex, hard-to-maintain code.

        Args:
            node (ast.AST): AST node to analyze (typically a function or class).

        Returns:
            int: Maximum nesting depth found. 0 indicates no nesting,
                higher values indicate deeper nesting levels.
        """
        max_depth = 0

        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            # Increase depth for nesting constructs
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                current_depth += 1

            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)

        calculate_depth(node)
        return max_depth
