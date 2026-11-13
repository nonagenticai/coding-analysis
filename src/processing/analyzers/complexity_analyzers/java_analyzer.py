"""
Java-specific complexity analyzer.

Provides complexity analysis tailored to Java's features like
inheritance, interfaces, annotations, generics, and exception handling.
"""

from __future__ import annotations

import math

from models.core_models import CoreCodeEntity
from processing.languages.base import ParseResult

from .base import (
    ComplexityLevel,
    LanguageComplexityAnalyzer,
    ServiceComplexityMetrics,
)
from .body_metrics import SourceContext


class JavaComplexityAnalyzer(LanguageComplexityAnalyzer):
    """Java-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze Java entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="java", entity_type=entity.type, entity_name=entity.name
        )

        # Basic complexity
        self.apply_body_metrics(metrics, entity, source)

        # Java-specific factors
        if entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)
        elif entity.type == "method":
            metrics.parameter_count = self._count_parameters(entity)

        # Calculate maintainability
        metrics.maintainability_index = self._calculate_maintainability_index(metrics)

        # Determine complexity level
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)

        # Generate suggestions
        metrics.suggestions = self._generate_java_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """Java-specific complexity factors."""
        return {
            "inheritance_penalty": 1.2,
            "interface_bonus": 0.8,
            "annotation_complexity": 0.1,
            "generics_penalty": 1.1,
            "exception_handling": 1.3,
        }

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for Java classes."""
        coupling = 0.0

        # Count imports and dependencies
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                coupling += 1.0

        return min(coupling / 10.0, 1.0)  # Normalize to 0-1

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count method parameters from signature."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                return len([p.strip() for p in params_str.split(",") if p.strip()])
        return 0

    def _calculate_maintainability_index(self, metrics: ServiceComplexityMetrics) -> float:
        """Calculate maintainability index (0-100)."""
        # Streamlined Microsoft maintainability index formula
        if metrics.lines_of_code == 0:
            return 100.0

        loc = max(metrics.lines_of_code, 1)
        cc = max(metrics.cyclomatic_complexity, 1)

        mi = 171 - 5.2 * math.log(loc) - 0.23 * cc - 16.2 * math.log(loc)
        return max(0.0, min(100.0, mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify complexity level."""
        if complexity <= 5:
            return ComplexityLevel.TRIVIAL
        elif complexity <= 10:
            return ComplexityLevel.BASIC
        elif complexity <= 20:
            return ComplexityLevel.MODERATE
        elif complexity <= 40:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_java_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate Java-specific refactoring suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 10:
            suggestions.append("Consider breaking down this method into smaller methods")

        if metrics.parameter_count > 5:
            suggestions.append("Consider using a parameter object or builder pattern")

        if metrics.nesting_depth > 3:
            suggestions.append("Reduce nesting depth using guard clauses or early returns")

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider dependency injection or interfaces"
            )

        if entity.type == "class" and metrics.cyclomatic_complexity > 50:
            suggestions.append("Large class detected - consider splitting responsibilities")

        return suggestions


__all__ = ["JavaComplexityAnalyzer"]
