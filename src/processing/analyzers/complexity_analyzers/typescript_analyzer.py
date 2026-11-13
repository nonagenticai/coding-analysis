"""
TypeScript-specific complexity analyzer.

Provides complexity analysis tailored to TypeScript's features like
type system, generics, decorators, conditional types, and mapped types.
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


class TypeScriptComplexityAnalyzer(LanguageComplexityAnalyzer):
    """TypeScript-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze TypeScript entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="typescript",
            entity_type=entity.type,
            entity_name=entity.name,
        )

        # TypeScript-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # TypeScript-specific analysis
        if entity.type in ["interface", "type_alias"]:
            metrics.coupling_factor = self._analyze_type_coupling(entity, parse_result)
        elif entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)
            if entity.properties and entity.properties.get("decorators"):
                metrics.risk_factors.append("decorators")
        elif entity.type == "function":
            metrics.parameter_count = self._count_typescript_parameters(entity)

        # Advanced TypeScript features complexity
        if entity.properties:
            if entity.properties.get("is_conditional_type"):
                metrics.risk_factors.append("conditional_type")
            if entity.properties.get("is_mapped_type"):
                metrics.risk_factors.append("mapped_type")
            if entity.properties.get("generic_parameters"):
                metrics.risk_factors.append("generic_parameters")

        metrics.maintainability_index = self._calculate_typescript_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_typescript_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """TypeScript-specific complexity factors."""
        return {
            "type_system_penalty": 1.3,
            "generic_penalty": 1.2,
            "decorator_penalty": 1.1,
            "conditional_type_penalty": 1.4,
            "mapped_type_penalty": 1.3,
            "interface_bonus": 0.9,
            "namespace_penalty": 1.1,
        }

    def _analyze_type_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze type coupling for TypeScript types."""
        coupling = 0.0

        # Count type dependencies
        for relation in parse_result.relationships:
            if relation.source_id == entity.id and relation.type in [
                "extends",
                "implements",
                "generic_constraint",
            ]:
                coupling += 1.0

        return min(coupling / 8.0, 1.0)  # Normalize to 0-1

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for TypeScript classes."""
        coupling = 0.0

        # Count inheritance and implementation relationships
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                if relation.type in ["extends", "implements"]:
                    coupling += 2.0  # High coupling
                elif relation.type == "decorated_by":
                    coupling += 0.5  # Medium coupling
                else:
                    coupling += 1.0  # Normal coupling

        return min(coupling / 10.0, 1.0)

    def _count_typescript_parameters(self, entity: CoreCodeEntity) -> int:
        """Count TypeScript function parameters."""
        if entity.properties and "parameters" in entity.properties:
            params_str = entity.properties["parameters"]
            if params_str.strip():
                # Basic parameter counting (could be updated with proper parsing)
                return len([p.strip() for p in params_str.split(",") if p.strip()])
        return 0

    def _calculate_typescript_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate TypeScript-specific maintainability."""
        base_mi = 100.0

        # Penalties for TypeScript complexity
        base_mi -= metrics.cyclomatic_complexity * 2
        base_mi -= metrics.parameter_count * 1.5

        # Type system bonuses/penalties
        if entity.properties:
            if entity.properties.get("is_exported"):
                base_mi -= 5.0  # Public APIs need more maintenance
            if entity.properties.get("generic_parameters"):
                base_mi -= len(entity.properties["generic_parameters"]) * 3.0
            if entity.properties.get("decorators"):
                base_mi -= len(entity.properties["decorators"]) * 2.0

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify TypeScript complexity."""
        if complexity <= 6:
            return ComplexityLevel.TRIVIAL
        elif complexity <= 12:
            return ComplexityLevel.BASIC
        elif complexity <= 25:
            return ComplexityLevel.MODERATE
        elif complexity <= 50:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_typescript_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate TypeScript-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 12:
            suggestions.append("Consider breaking this component into smaller, more focused pieces")

        if metrics.parameter_count > 4:
            suggestions.append("Consider using an options object or interface for parameters")

        if entity.properties:
            if (
                entity.properties.get("generic_parameters")
                and len(entity.properties["generic_parameters"]) > 3
            ):
                suggestions.append(
                    "High generic complexity - consider streamlining type parameters"
                )

            if entity.properties.get("is_conditional_type"):
                suggestions.append(
                    "Conditional types can be hard to understand - consider documenting the logic"
                )

            if entity.properties.get("is_mapped_type"):
                suggestions.append("Mapped types add complexity - ensure good documentation")

            if entity.properties.get("decorators") and len(entity.properties["decorators"]) > 2:
                suggestions.append(
                    "Multiple decorators can create complex behavior - consider composition"
                )

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider using dependency injection or abstractions"
            )

        return suggestions


__all__ = ["TypeScriptComplexityAnalyzer"]
