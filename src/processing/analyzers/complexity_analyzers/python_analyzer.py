"""
Python-specific complexity analyzer.

Provides complexity analysis tailored to Python's features like
decorators, comprehensions, metaclasses, generators, and async patterns.
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


class PythonComplexityAnalyzer(LanguageComplexityAnalyzer):
    """Python-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze Python entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="python", entity_type=entity.type, entity_name=entity.name
        )

        # Python-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # Python-specific analysis
        if entity.type in ["class", "method", "function"]:
            metrics.parameter_count = self._count_parameters(entity)

            # Check for decorators
            if entity.properties:
                decorators = entity.properties.get("decorators", [])
                if decorators:
                    metrics.risk_factors.append("decorators")

                # Comprehension complexity
                if entity.properties.get("has_comprehension"):
                    metrics.risk_factors.append("comprehension")

                # Metaclass complexity
                if entity.properties.get("has_metaclass"):
                    metrics.risk_factors.append("metaclass")

                # Generator complexity
                if entity.properties.get("is_generator"):
                    metrics.risk_factors.append("generator")

            # Async functions add complexity
            if entity.is_async:
                metrics.risk_factors.append("async")

        if entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)

        metrics.maintainability_index = self._calculate_python_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_python_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """Python-specific complexity factors."""
        return {
            "decorator_penalty": 1.2,
            "comprehension_penalty": 1.1,
            "metaclass_penalty": 2.0,
            "generator_penalty": 1.15,
            "async_penalty": 1.2,
            "duck_typing_complexity": 1.1,
            "magic_method_penalty": 1.3,
            "context_manager_bonus": 0.9,
        }

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for Python classes."""
        coupling = 0.0

        # Count inheritance and imports
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                if relation.type == "inherits":
                    coupling += 1.5  # Inheritance coupling
                elif relation.type == "imports":
                    coupling += 0.5  # Import coupling
                else:
                    coupling += 1.0  # Other dependencies

        return min(coupling / 8.0, 1.0)  # Normalize to 0-1

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count Python function/method parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                # Filter out 'self' and 'cls' parameters
                params = [
                    p.strip()
                    for p in params_str.split(",")
                    if p.strip()
                    and not p.strip().startswith("self")
                    and not p.strip().startswith("cls")
                ]
                return len(params)
        return 0

    def _calculate_python_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate Python-specific maintainability."""
        base_mi = 100.0

        # Penalties for Python-specific complexity
        base_mi -= metrics.cyclomatic_complexity * 2
        base_mi -= metrics.parameter_count * 1.5

        if entity.properties:
            # Decorator stacking penalty
            decorators = entity.properties.get("decorators", [])
            if len(decorators) > 3:
                base_mi -= (len(decorators) - 3) * 5.0

            # Magic method penalty
            if entity.name.startswith("__") and entity.name.endswith("__"):
                base_mi -= 5.0

            # Type hints bonus
            if entity.properties.get("has_type_hints"):
                base_mi += 5.0

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify Python complexity."""
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

    def _generate_python_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate Python-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 10:
            suggestions.append(
                "Consider breaking this function into smaller, more focused functions"
            )

        if metrics.parameter_count > 5:
            suggestions.append("Consider using **kwargs or a dataclass for parameter grouping")

        if entity.properties:
            decorators = entity.properties.get("decorators", [])
            if len(decorators) > 3:
                suggestions.append(
                    "Excessive decorator stacking - consider composition or simplification"
                )

            if entity.properties.get("has_comprehension") and metrics.cyclomatic_complexity > 15:
                suggestions.append(
                    "Complex comprehensions detected - consider extracting to helper functions"
                )

            if not entity.properties.get("has_type_hints") and entity.type in [
                "function",
                "method",
            ]:
                suggestions.append("Add type hints to improve code clarity and tooling")

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider dependency injection or ABC interfaces"
            )

        if entity.name.startswith("__") and entity.name.endswith("__"):
            suggestions.append(
                "Magic methods should be simple - delegate complex logic to helper methods"
            )

        return suggestions


__all__ = ["PythonComplexityAnalyzer"]
