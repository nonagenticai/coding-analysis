"""
C#-specific complexity analyzer.

Provides complexity analysis tailored to C#'s features like
LINQ, async/await, delegates, events, reflection, generics,
attributes, and properties.
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


class CSharpComplexityAnalyzer(LanguageComplexityAnalyzer):
    """C#-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze C# entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="csharp", entity_type=entity.type, entity_name=entity.name
        )

        # C#-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # C#-specific analysis
        if entity.type in ["method", "function", "constructor"]:
            metrics.parameter_count = self._count_parameters(entity)

            # Async/await complexity
            if entity.is_async:
                metrics.risk_factors.append("async")

            if entity.properties:
                # LINQ complexity
                if entity.properties.get("has_linq"):
                    metrics.risk_factors.append("linq")

                # Delegate/Event complexity
                if entity.properties.get("has_delegates"):
                    metrics.risk_factors.append("delegates")

                # Reflection usage
                if entity.properties.get("uses_reflection"):
                    metrics.risk_factors.append("reflection")

                # Extension methods
                if entity.properties.get("is_extension_method"):
                    metrics.risk_factors.append("extension_method")

        elif entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)

            if entity.properties:
                # Generics complexity
                generic_params = entity.properties.get("generic_parameters", [])
                if generic_params:
                    metrics.risk_factors.append("generic_parameters")

                # Attribute usage
                attributes = entity.properties.get("attributes", [])
                if len(attributes) > 2:
                    metrics.risk_factors.append("many_attributes")

        elif entity.type == "property":
            # Property complexity
            if entity.properties:
                if entity.properties.get("has_getter") and entity.properties.get("has_setter"):
                    metrics.risk_factors.append("property_getter_setter")

        metrics.maintainability_index = self._calculate_csharp_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_csharp_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """C#-specific complexity factors."""
        return {
            "linq_penalty": 1.3,
            "async_penalty": 1.2,
            "delegate_penalty": 1.2,
            "reflection_penalty": 1.5,
            "generics_penalty": 1.1,
            "attribute_penalty": 1.1,
            "interface_bonus": 0.9,
            "property_bonus": 0.95,
        }

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for C# classes."""
        coupling = 0.0

        # Count inheritance, interfaces, and dependencies
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                if relation.type in ["inherits", "extends"]:
                    coupling += 1.5  # Inheritance coupling
                elif relation.type == "implements":
                    coupling += 0.8  # Interface coupling (lower than inheritance)
                elif relation.type == "uses":
                    coupling += 1.0  # Dependency coupling
                else:
                    coupling += 0.5  # Other dependencies

        return min(coupling / 8.0, 1.0)  # Normalize to 0-1

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count C# method parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                # Filter out 'this' for extension methods
                params = [
                    p.strip()
                    for p in params_str.split(",")
                    if p.strip() and not p.strip().startswith("this ")
                ]
                return len(params)
        return 0

    def _calculate_csharp_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate C#-specific maintainability."""
        base_mi = 100.0

        # Penalties for C#-specific complexity
        base_mi -= metrics.cyclomatic_complexity * 2
        base_mi -= metrics.parameter_count * 1.5

        if entity.properties:
            # LINQ complexity penalty
            if entity.properties.get("has_linq"):
                linq_depth = entity.properties.get("linq_depth", 1)
                if linq_depth > 2:
                    base_mi -= (linq_depth - 2) * 8.0

            # Reflection penalty
            if entity.properties.get("uses_reflection"):
                base_mi -= 15.0

            # Async state machine complexity
            if entity.is_async and metrics.cyclomatic_complexity > 10:
                base_mi -= 10.0

            # Delegate composition penalty
            if (
                entity.properties.get("has_delegates")
                and entity.properties.get("delegate_count", 0) > 2
            ):
                base_mi -= 5.0

            # Property simplicity bonus
            if entity.type == "property" and entity.properties.get("is_auto_property"):
                base_mi += 5.0

            # Interface implementation bonus
            if entity.type == "class" and entity.properties.get("implements_interfaces"):
                base_mi += 3.0

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify C# complexity."""
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

    def _generate_csharp_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate C#-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 10:
            suggestions.append("Consider breaking this method into smaller, focused methods")

        if metrics.parameter_count > 5:
            suggestions.append("Consider using a parameter object or builder pattern")

        if entity.properties:
            # LINQ suggestions
            if entity.properties.get("has_linq"):
                linq_depth = entity.properties.get("linq_depth", 1)
                if linq_depth > 2:
                    suggestions.append(
                        "Complex LINQ query detected - consider breaking into multiple queries or using query syntax"
                    )

            # Async suggestions
            if entity.is_async and metrics.cyclomatic_complexity > 12:
                suggestions.append(
                    "Complex async method - consider extracting synchronous logic to helper methods"
                )

            # Reflection suggestions
            if entity.properties.get("uses_reflection"):
                suggestions.append(
                    "Reflection usage detected - consider using source generators or compile-time alternatives"
                )

            # Delegate suggestions
            if (
                entity.properties.get("has_delegates")
                and entity.properties.get("delegate_count", 0) > 2
            ):
                suggestions.append(
                    "Multiple delegates - consider using events or the observer pattern"
                )

            # Generic complexity
            generic_params = entity.properties.get("generic_parameters", [])
            if len(generic_params) > 2:
                suggestions.append(
                    "High generic complexity - consider splitting into simpler generic types"
                )

            # Attribute usage
            attributes = entity.properties.get("attributes", [])
            if len(attributes) > 3:
                suggestions.append(
                    "Many attributes detected - ensure each provides clear value and consider consolidation"
                )

            # Property suggestions
            if entity.type == "property" and not entity.properties.get("is_auto_property"):
                if metrics.cyclomatic_complexity > 3:
                    suggestions.append("Complex property logic - consider converting to a method")

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider dependency injection or interface-based design"
            )

        return suggestions


__all__ = ["CSharpComplexityAnalyzer"]
