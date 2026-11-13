#!/usr/bin/env python3
"""AST Validation Service

Handles comprehensive AST parsing and structure validation for code analysis.
Provides detailed analysis of Abstract Syntax Trees including syntax validation,
structural analysis, and complexity assessment.

This service supports multiple programming languages and provides detailed
insights into code structure, syntax correctness, and complexity metrics.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

from pydantic import BaseModel

from processing.python.python_entity_extractor import (
    PythonEntityExtractor,
)

logger = logging.getLogger(__name__)

#: Languages ``analyze_proposed_code_ast`` parses; anything else is answered
#: "not supported" (and ``/api/analysis/capabilities`` reports exactly these).
AST_LANGUAGES: tuple[str, ...] = ("python",)


class ASTIssue(BaseModel):
    """Single issue found during AST analysis."""

    description: str
    line_number: int = 0


class ASTSyntaxError(BaseModel):
    """One syntax error, with its location (1-based line and column)."""

    line: int = 0
    column: int = 0
    message: str


class ASTAnalysisResult(BaseModel):
    """Model for AST analysis results.

    This model captures the comprehensive results of AST analysis including
    syntax validation, structural components, and complexity metrics.
    """

    is_valid: bool
    syntax_errors: list[str]
    imports: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    classes: list[dict[str, Any]]
    variables: list[dict[str, Any]]
    complexity_score: float
    line_count: int
    issues: list[ASTIssue] = []
    # Structured form of ``syntax_errors`` for real parse failures (the string
    # list also carries non-syntax reasons such as "language not supported").
    syntax_error_details: list[ASTSyntaxError] = []


class ASTValidationService:
    """Service for comprehensive AST parsing and validation.

    This service provides advanced AST analysis capabilities including:
    - Syntax validation and error detection
    - Structural component extraction
    - Complexity analysis and metrics
    - Code quality assessment
    - Multi-language AST support

    The service integrates with entity extractors to provide detailed
    insights into code structure and syntax correctness.
    """

    def __init__(self, entity_extractor: PythonEntityExtractor | None = None):
        """Initialize the ASTValidationService service.

        Args:
            entity_extractor (Optional[PythonEntityExtractor]): Entity extractor
        """
        self.entity_extractor = entity_extractor or PythonEntityExtractor()

    async def analyze_proposed_code_ast(
        self, code: str, filename: str, language: str = "python"
    ) -> ASTAnalysisResult:
        """
        Analyze the AST structure of proposed code.

        Args:
            code: The code content to analyze
            filename: Name of the file
            language: Programming language (currently only python supported)

        Returns:
            ASTAnalysisResult with analysis details
        """
        try:
            if language.lower() not in AST_LANGUAGES:
                return ASTAnalysisResult(
                    is_valid=False,
                    syntax_errors=[f"Language {language} not supported for AST analysis"],
                    imports=[],
                    functions=[],
                    classes=[],
                    variables=[],
                    complexity_score=0.0,
                    line_count=len(code.splitlines()),
                )

            return await self._analyze_python_ast(code)

        except Exception as e:
            logger.error("Error analyzing AST for %s: %s", filename, e)
            return ASTAnalysisResult(
                is_valid=False,
                syntax_errors=[f"AST analysis failed: {str(e)}"],
                imports=[],
                functions=[],
                classes=[],
                variables=[],
                complexity_score=0.0,
                line_count=len(code.splitlines()) if code else 0,
            )

    async def _analyze_python_ast(self, code: str) -> ASTAnalysisResult:
        """Analyze Python AST structure."""
        syntax_errors: list[str] = []
        imports: list[Any] = []
        functions: list[Any] = []
        classes: list[Any] = []
        variables: list[Any] = []
        try:
            # Parse the AST
            tree = ast.parse(code)

            # Use our entity extractor to extract entities
            entities = await self.entity_extractor.extract_entities(
                tree,
                "memory_file.py",
                "",  # Pass empty string instead of empty list
            )

            # Process the extracted entities
            for entity in entities:
                entity_type = entity.get("type")
                if entity_type == "class":
                    classes.append(
                        {
                            "name": entity.get("name"),
                            "line": entity.get("start_line"),
                            "bases": entity.get("properties", {}).get("bases", []),
                            "methods": [],  # Will be populated later
                            "decorators": entity.get("properties", {}).get("decorators", []),
                            "docstring": entity.get("docstring"),
                        }
                    )
                elif entity_type in ["function", "method"]:
                    entity_name = entity.get("name")
                    if entity_name is not None:
                        function_name = entity_name.split(".")[-1]
                    else:
                        function_name = "unknown"

                    func_info = {
                        "name": function_name,
                        "line": entity.get("start_line"),
                        "args": entity.get("properties", {}).get("parameters", []),
                        "defaults": 0,  # Not easily available
                        "is_async": False,  # Not easily available
                        "decorators": entity.get("properties", {}).get("decorators", []),
                        "docstring": entity.get("docstring"),
                        "returns": None,  # Not easily available
                    }

                    # If it's a method, add it to the class methods list
                    parent_class = entity.get("properties", {}).get("parent_class")
                    if parent_class:
                        for cls in classes:
                            if cls["name"] == parent_class:
                                cls["methods"].append(func_info["name"])
                    else:
                        functions.append(func_info)

            # Walk through the AST nodes to extract imports and variables
            for node in ast.walk(tree):
                try:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports.extend(self._extract_import_info(node))
                    elif isinstance(node, ast.Assign):
                        variables.extend(self._extract_variable_info(node))
                except Exception as e:
                    logger.warning("Error processing AST node %s: %s", type(node), e)
                    continue

            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(tree)

            return ASTAnalysisResult(
                is_valid=True,
                syntax_errors=syntax_errors,
                imports=imports,
                functions=functions,
                classes=classes,
                variables=variables,
                complexity_score=complexity_score,
                line_count=len(code.splitlines()),
            )

        except SyntaxError as e:
            logger.error("Syntax error in Python code: %s", e)
            syntax_errors.append(f"Line {e.lineno}, column {e.offset}: {e.msg}")
            return ASTAnalysisResult(
                is_valid=False,
                syntax_errors=syntax_errors,
                syntax_error_details=[
                    ASTSyntaxError(line=e.lineno or 0, column=e.offset or 0, message=e.msg or "")
                ],
                imports=[],
                functions=[],
                classes=[],
                variables=[],
                complexity_score=0.0,
                line_count=len(code.splitlines()) if code else 0,
            )

    def _extract_import_info(self, node: ast.AST) -> list[dict[str, Any]]:
        """Extract import information from AST node."""
        imports: list[Any] = []
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(
                    {
                        "module": name.name,
                        "alias": name.asname,
                        "line": node.lineno,
                        "type": "import",
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                imports.append(
                    {
                        "module": (f"{module}.{name.name}" if module else name.name),
                        "alias": name.asname,
                        "line": node.lineno,
                        "type": "from_import",
                        "level": node.level,
                    }
                )

        return imports

    def _extract_variable_info(self, node: ast.Assign) -> list[dict[str, Any]]:
        """Extract variable assignment information."""
        variables: list[Any] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                variables.append(
                    {
                        "name": target.id,
                        "line": node.lineno,
                        "type": "assignment",
                    }
                )
        return variables

    def _get_attribute_name(self, node: ast.AST) -> str:
        """Get attribute name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return "unknown"

    def _calculate_complexity_score(self, tree: ast.AST) -> float:
        """Calculate a basic complexity score based on AST structure."""
        complexity: float = 0

        for node in ast.walk(tree):
            # Control flow adds complexity
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            # Nested functions add complexity
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 0.5
            # Exception handling adds complexity
            elif isinstance(node, ast.ExceptHandler):
                complexity += 0.5

        # Normalize by number of lines (rough estimate)
        total_lines = max(1, len([n for n in ast.walk(tree) if hasattr(n, "lineno")]))
        return min(10.0, complexity / total_lines * 10)  # Scale to 0-10
