"""
Rust-specific complexity analyzer.

Provides complexity analysis tailored to Rust's features like
lifetimes, unsafe blocks, macros, traits, and ownership semantics.
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


class RustComplexityAnalyzer(LanguageComplexityAnalyzer):
    """Rust-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze Rust entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="rust", entity_type=entity.type, entity_name=entity.name
        )

        # Rust-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # Rust-specific analysis
        if entity.type == "function":
            metrics.parameter_count = self._count_parameters(entity)
            if entity.is_async:
                metrics.risk_factors.append("async")

        elif entity.type == "struct":
            metrics.coupling_factor = self._analyze_trait_coupling(entity, parse_result)

        # Calculate maintainability with Rust-specific factors
        metrics.maintainability_index = self._calculate_rust_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_rust_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """Rust-specific complexity factors."""
        return {
            "lifetime_penalty": 1.1,
            "unsafe_penalty": 2.0,
            "macro_complexity": 1.5,
            "trait_bonus": 0.9,
            "ownership_complexity": 1.2,
        }

    def _analyze_trait_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze trait coupling for Rust types."""
        trait_impls = 0

        for relation in parse_result.relationships:
            if relation.source_id == entity.id and relation.type in [
                "implements",
                "trait_bound",
            ]:
                trait_impls += 1

        return min(trait_impls / 5.0, 1.0)

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count function parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                # Filter out 'self' parameters
                params = [
                    p.strip()
                    for p in params_str.split(",")
                    if p.strip() and not p.strip().startswith("&sel") and p.strip() != "sel"
                ]
                return len(params)
        return 0

    def _calculate_rust_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate Rust-specific maintainability."""
        base_mi = 100.0

        # Penalties for Rust-specific complexity
        if entity.properties:
            if entity.properties.get("has_unsafe", False):
                base_mi -= 20.0
            if entity.properties.get("macro_usage", 0) > 0:
                base_mi -= 10.0

        # Factor in cyclomatic complexity
        base_mi -= metrics.cyclomatic_complexity * 2

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify Rust complexity (slightly different thresholds)."""
        if complexity <= 4:
            return ComplexityLevel.TRIVIAL
        elif complexity <= 8:
            return ComplexityLevel.BASIC
        elif complexity <= 15:
            return ComplexityLevel.MODERATE
        elif complexity <= 30:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_rust_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate Rust-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 8:
            suggestions.append("Consider breaking this function into smaller functions")

        if metrics.parameter_count > 4:
            suggestions.append("Consider using a struct to group related parameters")

        if entity.properties and entity.properties.get("has_unsafe", False):
            suggestions.append("Review unsafe code blocks and consider safe alternatives")

        if metrics.coupling_factor > 0.8:
            suggestions.append("High trait coupling - consider trait composition")

        return suggestions


__all__ = ["RustComplexityAnalyzer"]
