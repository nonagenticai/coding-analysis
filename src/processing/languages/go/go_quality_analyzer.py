"""
Go code quality analysis module.

Provides code quality metrics, complexity analysis, and code health assessment
for Go source code. Includes cyclomatic complexity, maintainability metrics,
and Go-specific quality indicators.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityType,
)

logger = logging.getLogger(__name__)


class GoQualityAnalyzer:
    """Go code quality analysis and metrics calculation."""

    def __init__(self):
        """Initialize Go quality analyzer."""
        pass

    async def calculate_complexity_metrics(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> CoreComplexityMetrics:
        """Calculate comprehensive complexity metrics for Go code.

        Args:
            content: Source code content
            entities: List of extracted entities

        Returns:
            CoreComplexityMetrics with calculated values
        """
        logger.debug(
            "Calculating complexity metrics for Go code (%d chars, %d entities)",
            len(content),
            len(entities),
        )

        lines = content.splitlines()
        total_lines = len(lines)

        # Calculate logical lines (non-empty, non-comment)
        logical_lines = []
        comment_lines = []
        blank_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines.append(line)
            elif stripped.startswith("//") or stripped.startswith("/*"):
                comment_lines.append(line)
            else:
                logical_lines.append(line)

        # Calculate function-specific metrics
        function_count = len([e for e in entities if e.type == EntityType.FUNCTION])

        # Calculate cyclomatic complexity
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(content)

        # Calculate cognitive complexity
        cognitive_complexity = self._calculate_cognitive_complexity(content)

        # Calculate maintainability index
        maintainability_index = self._calculate_maintainability_index(
            len(logical_lines), cyclomatic_complexity, function_count
        )

        # Calculate docstring coverage (Go uses comments above declarations)
        docstring_coverage = self._calculate_comment_coverage(content, entities)

        return CoreComplexityMetrics(
            lines_of_code=total_lines,
            logical_lines=len(logical_lines),
            function_count=function_count,
            class_count=0,  # Go doesn't have classes
            comment_lines=len(comment_lines),
            blank_lines=len(blank_lines),
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            maintainability_index=maintainability_index,
            docstring_coverage=docstring_coverage,
        )

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity for Go code.

        Args:
            content: Source code content

        Returns:
            Cyclomatic complexity value
        """
        # Base complexity is 1
        complexity = 1

        # Keywords that increase complexity
        complexity_keywords = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\bfor\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bselect\b",
            r"\bgo\b",
            r"\bdefer\b",
            r"&&",
            r"\|\|",
        ]

        for keyword in complexity_keywords:
            matches = re.findall(keyword, content)
            complexity += len(matches)

        # Function declarations add to complexity
        func_pattern = r"\bfunc\b"
        func_matches = re.findall(func_pattern, content)
        complexity += len(func_matches)

        return complexity

    def _calculate_cognitive_complexity(self, content: str) -> int:
        """Calculate cognitive complexity for Go code.

        Args:
            content: Source code content

        Returns:
            Cognitive complexity value
        """
        complexity = 0
        nesting_level = 0

        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()

            # Skip comments and empty lines
            if not stripped or stripped.startswith("//"):
                continue

            # Track nesting level
            if any(keyword in stripped for keyword in ["{", "if", "for", "switch", "select"]):
                if any(keyword in stripped for keyword in ["if", "for", "switch", "select"]):
                    complexity += 1 + nesting_level
                if "{" in stripped:
                    nesting_level += 1

            if "}" in stripped:
                nesting_level = max(0, nesting_level - 1)

            # Logical operators add complexity
            if "&&" in stripped or "||" in stripped:
                complexity += 1

        return complexity

    def _calculate_maintainability_index(
        self, logical_lines: int, cyclomatic_complexity: int, function_count: int
    ) -> float:
        """Calculate maintainability index.

        Args:
            logical_lines: Number of logical lines of code
            cyclomatic_complexity: Cyclomatic complexity value
            function_count: Number of functions

        Returns:
            Maintainability index (0-100, higher is better)
        """
        import math

        if logical_lines == 0:
            return 100.0

        # Simplified maintainability index calculation
        halstead_volume = logical_lines * math.log2(max(function_count, 1))

        try:
            mi = (
                171
                - 5.2 * math.log(halstead_volume)
                - 0.23 * cyclomatic_complexity
                - 16.2 * math.log(logical_lines)
            )
            return max(0.0, min(100.0, mi))
        except (ValueError, ZeroDivisionError):  # fmt: skip
            return 50.0  # Default moderate maintainability

    def _calculate_comment_coverage(self, content: str, entities: list[CoreCodeEntity]) -> float:
        """Calculate comment coverage for documented entities.

        Args:
            content: Source code content
            entities: List of extracted entities

        Returns:
            Comment coverage percentage (0.0-1.0)
        """
        lines = content.splitlines()
        documented_entities = 0
        total_documentable = 0

        # Entities that should be documented
        documentable_types = {
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.INTERFACE,
            EntityType.TYPE_ALIAS,
            EntityType.CONSTANT,
        }

        for entity in entities:
            if entity.type in documentable_types:
                # Only count exported (public) entities
                if entity.visibility == "public":
                    total_documentable += 1

                    # Check if there's a comment above the entity
                    if entity.line_number and entity.line_number > 1:
                        line_before = lines[entity.line_number - 2].strip()
                        if line_before.startswith("//"):
                            documented_entities += 1

        if total_documentable == 0:
            return 1.0

        return documented_entities / total_documentable

    async def analyze_code_quality(
        self, content: str, entities: list[CoreCodeEntity], file_path: Path
    ) -> dict[str, Any]:
        """Perform comprehensive code quality analysis.

        Args:
            content: Source code content
            entities: List of extracted entities
            file_path: Path to the source file

        Returns:
            Dictionary with quality analysis results
        """
        logger.debug("Analyzing code quality for Go file: %s", file_path)

        # Get complexity metrics
        complexity_metrics = await self.calculate_complexity_metrics(content, entities)

        # Analyze function complexity
        function_analysis = self._analyze_function_complexity(content, entities)

        # Analyze naming conventions
        naming_analysis = self._analyze_naming_conventions(entities)

        # Analyze code patterns
        pattern_analysis = self._analyze_code_patterns(content)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            complexity_metrics, function_analysis, naming_analysis, pattern_analysis
        )

        return {
            "complexity_metrics": complexity_metrics,
            "function_analysis": function_analysis,
            "naming_analysis": naming_analysis,
            "pattern_analysis": pattern_analysis,
            "quality_score": quality_score,
            "recommendations": self._generate_recommendations(
                complexity_metrics, function_analysis, naming_analysis, pattern_analysis
            ),
        }

    def _analyze_function_complexity(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> dict[str, Any]:
        """Analyze complexity of individual functions.

        Args:
            content: Source code content
            entities: List of extracted entities

        Returns:
            Function complexity analysis results
        """
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        function_complexities: list[dict[str, Any]] = []

        for func in functions:
            # Extract function body (simplified)
            func_pattern = (
                rf"func\s+(?:\([^)]+\)\s+)?{re.escape(func.name)}\s*\([^)]*\).*?{{(.*?)}}"
            )
            func_match = re.search(func_pattern, content, re.DOTALL)

            if func_match:
                func_body = func_match.group(1)
                func_complexity = self._calculate_cyclomatic_complexity(func_body)
                func_lines = len(func_body.splitlines())

                function_complexities.append(
                    {
                        "name": func.name,
                        "complexity": func_complexity,
                        "lines": func_lines,
                        "is_method": func.properties.get("is_method", False),
                        "receiver_type": func.properties.get("receiver_type"),
                    }
                )

        # Calculate statistics
        if function_complexities:
            complexities = [f["complexity"] for f in function_complexities]
            avg_complexity = sum(complexities) / len(complexities)
            max_complexity = max(complexities)

            # Find problematic functions (complexity > 10)
            complex_functions = [f for f in function_complexities if f["complexity"] > 10]
        else:
            avg_complexity = 0
            max_complexity = 0
            complex_functions = []

        return {
            "total_functions": len(functions),
            "average_complexity": avg_complexity,
            "max_complexity": max_complexity,
            "complex_functions": complex_functions,
            "function_details": function_complexities,
        }

    def _analyze_naming_conventions(self, entities: list[CoreCodeEntity]) -> dict[str, Any]:
        """Analyze naming convention compliance.

        Args:
            entities: List of extracted entities

        Returns:
            Naming convention analysis results
        """
        issues = []
        total_entities = 0

        for entity in entities:
            total_entities += 1
            name = entity.name

            # Check Go naming conventions
            if entity.type == EntityType.FUNCTION:
                # Functions should use camelCase or PascalCase
                if not re.match(r"^[A-Za-z][A-Za-z0-9]*$", name):
                    issues.append(
                        {
                            "entity": name,
                            "type": "function",
                            "issue": "Function name should use camelCase or PascalCase",
                        }
                    )

                # Check for exported functions (should start with uppercase)
                if entity.visibility == "public" and not name[0].isupper():
                    issues.append(
                        {
                            "entity": name,
                            "type": "function",
                            "issue": "Exported function should start with uppercase letter",
                        }
                    )

            elif entity.type in [EntityType.CLASS, EntityType.INTERFACE]:
                # Types should use PascalCase
                if not re.match(r"^[A-Z][A-Za-z0-9]*$", name):
                    issues.append(
                        {
                            "entity": name,
                            "type": "type",
                            "issue": "Type name should use PascalCase",
                        }
                    )

            elif entity.type == EntityType.CONSTANT:
                # Constants can be PascalCase or UPPER_CASE
                if not re.match(r"^[A-Z][A-Za-z0-9]*$|^[A-Z][A-Z0-9_]*$", name):
                    issues.append(
                        {
                            "entity": name,
                            "type": "constant",
                            "issue": "Constant should use PascalCase or UPPER_CASE",
                        }
                    )

        compliance_rate = (
            1.0 if total_entities == 0 else (total_entities - len(issues)) / total_entities
        )

        return {
            "compliance_rate": compliance_rate,
            "total_entities": total_entities,
            "issues": issues,
            "issue_count": len(issues),
        }

    def _analyze_code_patterns(self, content: str) -> dict[str, Any]:
        """Analyze code patterns and potential issues.

        Args:
            content: Source code content

        Returns:
            Code pattern analysis results
        """
        patterns = {
            "error_handling": {
                "if_err_nil": len(re.findall(r"if\s+err\s*!=\s*nil", content)),
                "error_returns": len(re.findall(r"return.*err", content)),
            },
            "goroutines": {
                "go_statements": len(re.findall(r"\bgo\s+\w+", content)),
                "channels": len(re.findall(r"<-|chan\s+", content)),
            },
            "defer_statements": len(re.findall(r"\bdefer\s+", content)),
            "panic_statements": len(re.findall(r"\bpanic\s*\(", content)),
            "TODO_comments": len(re.findall(r"//\s*TODO", content, re.IGNORECASE)),
            "FIXME_comments": len(re.findall(r"//\s*FIXME", content, re.IGNORECASE)),
        }

        # Calculate pattern scores
        good_patterns = patterns["error_handling"]["if_err_nil"] + patterns["defer_statements"]
        concerning_patterns = patterns["panic_statements"] + patterns["TODO_comments"]

        return {
            "patterns": patterns,
            "good_pattern_count": good_patterns,
            "concerning_pattern_count": concerning_patterns,
            "concurrent_features": patterns["goroutines"]["go_statements"] > 0
            or patterns["goroutines"]["channels"] > 0,
        }

    def _calculate_quality_score(
        self,
        complexity_metrics: CoreComplexityMetrics,
        function_analysis: dict,
        naming_analysis: dict,
        pattern_analysis: dict,
    ) -> float:
        """Calculate overall quality score.

        Args:
            complexity_metrics: Complexity metrics
            function_analysis: Function analysis results
            naming_analysis: Naming analysis results
            pattern_analysis: Pattern analysis results

        Returns:
            Quality score (0-100, higher is better)
        """
        scores = []

        # Maintainability score (25% weight)
        maintainability_score = complexity_metrics.maintainability_index
        scores.append(maintainability_score * 0.25)

        # Documentation score (20% weight)
        doc_score = complexity_metrics.docstring_coverage * 100
        scores.append(doc_score * 0.20)

        # Function complexity score (25% weight)
        if function_analysis["average_complexity"] > 0:
            # Lower complexity is better (invert the score)
            func_score = max(0, 100 - (function_analysis["average_complexity"] * 5))
        else:
            func_score = 100
        scores.append(func_score * 0.25)

        # Naming convention score (15% weight)
        naming_score = naming_analysis["compliance_rate"] * 100
        scores.append(naming_score * 0.15)

        # Code pattern score (15% weight)
        pattern_score = max(0, 100 - (pattern_analysis["concerning_pattern_count"] * 10))
        scores.append(pattern_score * 0.15)

        return sum(scores)

    def _generate_recommendations(
        self,
        complexity_metrics: CoreComplexityMetrics,
        function_analysis: dict,
        naming_analysis: dict,
        pattern_analysis: dict,
    ) -> list[str]:
        """Generate code quality improvement recommendations.

        Args:
            complexity_metrics: Complexity metrics
            function_analysis: Function analysis results
            naming_analysis: Naming analysis results
            pattern_analysis: Pattern analysis results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Complexity recommendations
        if complexity_metrics.cyclomatic_complexity > 20:
            recommendations.append("Consider breaking down complex logic into smaller functions")

        if function_analysis["complex_functions"]:
            recommendations.append(
                f"Refactor {len(function_analysis['complex_functions'])} complex functions with complexity > 10"
            )

        # Documentation recommendations
        if complexity_metrics.docstring_coverage < 0.8:
            recommendations.append("Add documentation comments for exported functions and types")

        # Naming recommendations
        if naming_analysis["compliance_rate"] < 0.9:
            recommendations.append("Fix naming convention violations following Go standards")

        # Pattern recommendations
        if pattern_analysis["patterns"]["panic_statements"] > 0:
            recommendations.append("Consider using error returns instead of panic statements")

        if pattern_analysis["patterns"]["TODO_comments"] > 5:
            recommendations.append("Address TODO comments to improve code completeness")

        # Maintainability recommendations
        if complexity_metrics.maintainability_index < 50:
            recommendations.append(
                "Improve code maintainability by reducing complexity and adding documentation"
            )

        return recommendations
