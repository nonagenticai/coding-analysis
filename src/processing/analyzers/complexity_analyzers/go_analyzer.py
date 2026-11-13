"""
Go-specific complexity analyzer.

Provides complexity analysis tailored to Go's features like
interfaces, goroutines, channels, error handling, and pointers.
"""

from __future__ import annotations

from models.core_models import CoreCodeEntity
from processing.languages.base import ParseResult

from .base import (
    ComplexityLevel,
    LanguageComplexityAnalyzer,
    ServiceComplexityMetrics,
)
from .body_metrics import SourceContext


class GoComplexityAnalyzer(LanguageComplexityAnalyzer):
    """Go-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze Go entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="go", entity_type=entity.type, entity_name=entity.name
        )

        self.apply_body_metrics(metrics, entity, source)

        # Go-specific analysis
        if entity.type == "function":
            metrics.parameter_count = self._count_parameters(entity)

        elif entity.type == "struct":
            metrics.coupling_factor = self._analyze_interface_coupling(entity, parse_result)

        metrics.maintainability_index = self._calculate_go_maintainability(metrics)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_go_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """Go-specific complexity factors."""
        return {
            "interface_bonus": 0.8,
            "goroutine_penalty": 1.3,
            "channel_complexity": 1.2,
            "error_handling": 1.1,
            "pointer_complexity": 1.1,
        }

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count Go function parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                return len([p.strip() for p in params_str.split(",") if p.strip()])
        return 0

    def _analyze_interface_coupling(
        self, entity: CoreCodeEntity, parse_result: ParseResult
    ) -> float:
        """Analyze interface coupling."""
        interface_refs = 0

        for relation in parse_result.relationships:
            if relation.source_id == entity.id and relation.type == "implements":
                interface_refs += 1

        return min(interface_refs / 3.0, 1.0)

    def _calculate_go_maintainability(self, metrics: ServiceComplexityMetrics) -> float:
        """Calculate Go-specific maintainability."""
        base_mi = 100.0
        base_mi -= metrics.cyclomatic_complexity * 1.5
        base_mi -= metrics.parameter_count * 2
        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify Go complexity."""
        if complexity <= 5:
            return ComplexityLevel.TRIVIAL
        elif complexity <= 10:
            return ComplexityLevel.BASIC
        elif complexity <= 15:
            return ComplexityLevel.MODERATE
        elif complexity <= 25:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_go_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate Go-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 10:
            suggestions.append("Consider splitting this function into smaller functions")

        if metrics.parameter_count > 5:
            suggestions.append("Consider using a struct to group parameters")

        if entity.type == "function" and "goroutine" in (entity.signature or ""):
            suggestions.append("Review goroutine usage for potential race conditions")

        return suggestions


__all__ = ["GoComplexityAnalyzer"]
