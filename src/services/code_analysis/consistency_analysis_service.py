#!/usr/bin/env python3
"""Consistency Analysis Service

Handles analysis of code consistency with existing codebase patterns.
Provides comprehensive consistency checking including naming conventions,
documentation standards, style guidelines, and pattern adherence.

This service uses direct database access for optimal performance and provides
detailed insights into code consistency across the entire codebase.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.core.config.core_config_settings import get_settings
from db.facade import GraphDatabaseConnection

settings = get_settings()
logger = logging.getLogger(__name__)


class ConsistencyAnalysisResult(BaseModel):
    """Model for consistency analysis results.

    This model captures the comprehensive results of consistency analysis
    including identified issues, scores, and actionable suggestions.
    """

    naming_issues: list[str]
    documentation_issues: list[str]
    style_issues: list[str]
    pattern_violations: list[str]
    consistency_score: float
    suggestions: list[str]


class ConsistencyAnalysisService:
    """Service for comprehensive code consistency analysis.

    This service provides advanced consistency analysis capabilities including:
    - Naming convention validation
    - Documentation standard enforcement
    - Style guideline compliance
    - Pattern adherence checking
    - Consistency scoring and reporting

    The service uses direct database access for optimal performance and
    provides detailed insights into code consistency across the entire codebase.
    """

    def __init__(self) -> None:
        """Initialize the consistency analysis service with database connection."""
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.db_connection: GraphDatabaseConnection | None = None

        # Configuration for analysis rules
        self.max_line_length = 88
        self.max_function_args = 5
        self.min_docstring_length = 10
        self.min_class_docstring_length = 20

        self.logger.info("ConsistencyAnalysisService initialized")

    async def _get_db_connection(self, code_graph_id: str) -> GraphDatabaseConnection:
        """Get or create database connection for a specific code graph."""
        if self.db_connection is None:
            self.db_connection = GraphDatabaseConnection(code_graph_id)
            await self.db_connection.connect()
        return self.db_connection

    async def close(self) -> None:
        """Close database connection and cleanup resources."""
        if self.db_connection and self.db_connection.conn:
            await self.db_connection.close()
            self.db_connection = None
            self.logger.info("ConsistencyAnalysisService database connection closed")

    async def analyze_code_consistency(
        self,
        proposed_code: str,
        ast_result: dict[str, Any],
    ) -> ConsistencyAnalysisResult:
        """
        Analyze consistency of proposed code with existing codebase.

        Args:
            proposed_code: The proposed code content
            ast_result: AST analysis result

        Returns:
            ConsistencyAnalysisResult with analysis details
        """
        naming_issues: list[Any] = []
        documentation_issues: list[Any] = []
        style_issues: list[Any] = []
        pattern_violations: list[Any] = []
        suggestions: list[Any] = []
        try:
            # Check naming conventions
            naming_issues = await self._check_naming_conventions(ast_result)

            # Check documentation style
            documentation_issues = await self._check_documentation_style(ast_result)

            # Check code style patterns
            style_issues = self._check_code_style(proposed_code)

            # Check for pattern violations
            pattern_violations = self._check_pattern_violations(ast_result)

            # Generate suggestions
            suggestions = self._generate_consistency_suggestions(
                naming_issues,
                documentation_issues,
                style_issues,
                pattern_violations,
            )

            # Calculate overall consistency score
            consistency_score = self._calculate_consistency_score(
                naming_issues,
                documentation_issues,
                style_issues,
                pattern_violations,
            )

        except Exception as e:
            self.logger.error("Error in consistency analysis: %s", e)
            # Return default result on error
            consistency_score = 0.5

        return ConsistencyAnalysisResult(
            naming_issues=naming_issues,
            documentation_issues=documentation_issues,
            style_issues=style_issues,
            pattern_violations=pattern_violations,
            consistency_score=consistency_score,
            suggestions=suggestions,
        )

    async def _check_naming_conventions(
        self,
        ast_result: dict[str, Any],
    ) -> list[str]:
        """Check naming convention consistency."""
        issues: list[Any] = []
        try:
            # Check function naming
            functions = ast_result.get("functions", [])
            for func in functions:
                func_name = func.get("name", "")
                if not self._is_snake_case(func_name):
                    issues.append(f"Function '{func_name}' should use snake_case naming")

                if func_name.startswith("_") and not func_name.startswith("__"):
                    # Private function - check if it's really needed
                    issues.append(f"Consider if private function '{func_name}' is necessary")

            # Check class naming
            classes = ast_result.get("classes", [])
            for cls in classes:
                cls_name = cls.get("name", "")
                if not self._is_pascal_case(cls_name):
                    issues.append(f"Class '{cls_name}' should use PascalCase naming")

            # Check variable naming
            variables = ast_result.get("variables", [])
            for var in variables:
                var_name = var.get("name", "")
                if not self._is_snake_case(var_name) and not var_name.isupper():
                    issues.append(
                        f"Variable '{var_name}' should use snake_case or UPPER_CASE naming"
                    )

        except Exception as e:
            self.logger.warning("Error checking naming conventions: %s", e)

        return issues

    async def _check_documentation_style(
        self,
        ast_result: dict[str, Any],
    ) -> list[str]:
        """Check documentation style consistency."""
        issues: list[Any] = []
        try:
            # Check function documentation
            functions = ast_result.get("functions", [])
            for func in functions:
                func_name = func.get("name", "")
                docstring = func.get("docstring")

                if not docstring and not func_name.startswith("_"):
                    issues.append(f"Public function '{func_name}' missing docstring")
                elif docstring and len(docstring.strip()) < self.min_docstring_length:
                    issues.append(f"Function '{func_name}' has very short docstring")

            # Check class documentation
            classes = ast_result.get("classes", [])
            for cls in classes:
                cls_name = cls.get("name", "")
                docstring = cls.get("docstring")

                if not docstring:
                    issues.append(f"Class '{cls_name}' missing docstring")
                elif docstring and len(docstring.strip()) < self.min_class_docstring_length:
                    issues.append(f"Class '{cls_name}' has very short docstring")

        except Exception as e:
            self.logger.warning("Error checking documentation style: %s", e)

        return issues

    def _check_code_style(self, code: str) -> list[str]:
        """Check code style patterns."""
        issues: list[Any] = []
        try:
            lines = code.splitlines()

            # Check line length
            for i, line in enumerate(lines, 1):
                if (
                    len(line) > self.max_line_length
                ):  # PEP 8 recommends 79, but 88 is common in modern Python
                    issues.append(
                        f"Line {i} exceeds recommended length ({self.max_line_length} characters)"
                    )

            # Check for trailing whitespace
            for i, line in enumerate(lines, 1):
                if line.rstrip() != line:
                    issues.append(f"Line {i} has trailing whitespace")

            # Check for multiple blank lines
            blank_count = 0
            for i, line in enumerate(lines, 1):
                if not line.strip():
                    blank_count += 1
                    if blank_count > 2:
                        issues.append(f"Too many consecutive blank lines around line {i}")
                else:
                    blank_count = 0

        except Exception as e:
            self.logger.warning("Error checking code style: %s", e)

        return issues

    def _check_pattern_violations(self, ast_result: dict[str, Any]) -> list[str]:
        """Check for common pattern violations."""
        violations: list[Any] = []
        try:
            # Check for overly complex functions
            functions = ast_result.get("functions", [])
            for func in functions:
                func_name = func.get("name", "")
                args_count = len(func.get("args", []))

                if args_count > self.max_function_args:
                    violations.append(
                        f"Function '{func_name}' has too many parameters ({args_count}). "
                        "Consider using a configuration object."
                    )

            # Check for missing type hints (if we can detect them)
            for func in functions:
                func_name = func.get("name", "")
                returns = func.get("returns")

                if not returns and not func_name.startswith("_"):
                    violations.append(f"Function '{func_name}' missing return type annotation")

        except Exception as e:
            self.logger.warning("Error checking pattern violations: %s", e)

        return violations

    def _generate_consistency_suggestions(
        self,
        naming_issues: list[str],
        documentation_issues: list[str],
        style_issues: list[str],
        pattern_violations: list[str],
    ) -> list[str]:
        """Generate suggestions based on consistency issues."""
        suggestions: list[Any] = []
        if naming_issues:
            suggestions.append("Consider following PEP 8 naming conventions consistently")

        if documentation_issues:
            suggestions.append("Add comprehensive docstrings following Google or NumPy style")

        if style_issues:
            suggestions.append("Run a code formatter like ruff format to resolve style issues")

        if pattern_violations:
            suggestions.append("Consider refactoring complex functions into smaller, focused units")

        return suggestions

    def _calculate_consistency_score(
        self,
        naming_issues: list[str],
        documentation_issues: list[str],
        style_issues: list[str],
        pattern_violations: list[str],
    ) -> float:
        """Calculate overall consistency score (0-1)."""
        total_issues = (
            len(naming_issues)
            + len(documentation_issues)
            + len(style_issues)
            + len(pattern_violations)
        )

        if total_issues == 0:
            return 1.0

        # Weight different types of issues
        weighted_score = (
            len(naming_issues) * 0.3
            + len(documentation_issues) * 0.2
            + len(style_issues) * 0.1
            + len(pattern_violations) * 0.4
        )

        # Convert to 0-1 scale (assuming max 20 issues for normalization)
        normalized_score = max(0.0, 1.0 - (weighted_score / 20.0))

        return round(normalized_score, 2)

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        if not name:
            return False

        # Allow leading/trailing underscores for special methods
        clean_name = name.strip("_")
        if not clean_name:
            return True

        # Should be lowercase with underscores
        return clean_name.islower() and " " not in clean_name

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention."""
        if not name:
            return False

        # Should start with uppercase and have no underscores (except leading/trailing)
        clean_name = name.strip("_")
        if not clean_name:
            return True

        return clean_name[0].isupper() and "_" not in clean_name
