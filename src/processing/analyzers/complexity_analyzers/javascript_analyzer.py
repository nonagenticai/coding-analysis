"""
JavaScript-specific complexity analyzer.

Provides complexity analysis tailored to JavaScript's features like
callbacks, promises, closures, prototype chains, and dynamic typing.
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


class JavaScriptComplexityAnalyzer(LanguageComplexityAnalyzer):
    """JavaScript-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze JavaScript entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="javascript", entity_type=entity.type, entity_name=entity.name
        )

        # JavaScript-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # JavaScript-specific analysis
        if entity.type in ["function", "arrow_function", "method"]:
            metrics.parameter_count = self._count_parameters(entity)

            # Callback complexity
            if entity.properties:
                if entity.properties.get("has_callbacks"):
                    metrics.risk_factors.append("callbacks")

                # Promise chain complexity
                if entity.properties.get("has_promises"):
                    metrics.risk_factors.append("promises")

                # Closure depth
                closure_depth = entity.properties.get("closure_depth", 0)
                if closure_depth > 0:
                    metrics.risk_factors.append("closures")

                # IIFE (Immediately Invoked Function Expression)
                if entity.properties.get("is_iife"):
                    metrics.risk_factors.append("iife")

            # Async functions
            if entity.is_async:
                metrics.risk_factors.append("async")

        elif entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)

        metrics.maintainability_index = self._calculate_javascript_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_javascript_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """JavaScript-specific complexity factors."""
        return {
            "callback_nesting_penalty": 1.4,
            "promise_chain_penalty": 1.2,
            "closure_penalty": 1.3,
            "prototype_complexity": 1.2,
            "async_penalty": 1.2,
            "dynamic_typing_complexity": 1.1,
            "event_listener_penalty": 1.1,
        }

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for JavaScript classes."""
        coupling = 0.0

        # Count prototype chain and dependencies
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                if relation.type in ["extends", "inherits"]:
                    coupling += 1.5  # Inheritance coupling
                elif relation.type == "imports":
                    coupling += 0.5  # Import coupling
                else:
                    coupling += 1.0  # Other dependencies

        return min(coupling / 8.0, 1.0)  # Normalize to 0-1

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count JavaScript function parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                # Filter out rest parameters and destructuring (basic check)
                params = [
                    p.strip() for p in params_str.split(",") if p.strip() and p.strip() != "..."
                ]
                return len(params)
        return 0

    def _calculate_javascript_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate JavaScript-specific maintainability."""
        base_mi = 100.0

        # Penalties for JavaScript-specific complexity
        base_mi -= metrics.cyclomatic_complexity * 2
        base_mi -= metrics.parameter_count * 1.5

        if entity.properties:
            # Callback hell penalty
            if entity.properties.get("has_callbacks") and metrics.cyclomatic_complexity > 15:
                base_mi -= 15.0

            # Deep closure penalty
            closure_depth = entity.properties.get("closure_depth", 0)
            if closure_depth > 2:
                base_mi -= (closure_depth - 2) * 5.0

            # Prototype manipulation penalty
            if entity.properties.get("modifies_prototype"):
                base_mi -= 10.0

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify JavaScript complexity."""
        if complexity <= 5:
            return ComplexityLevel.TRIVIAL
        elif complexity <= 10:
            return ComplexityLevel.BASIC
        elif complexity <= 18:
            return ComplexityLevel.MODERATE
        elif complexity <= 35:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_javascript_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate JavaScript-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 10:
            suggestions.append(
                "Consider breaking this function into smaller, single-purpose functions"
            )

        if metrics.parameter_count > 4:
            suggestions.append("Consider using an options object or destructuring for parameters")

        if entity.properties:
            if entity.properties.get("has_callbacks") and not entity.is_async:
                suggestions.append("Convert callbacks to async/await for better readability")

            if entity.properties.get("has_promises") and metrics.cyclomatic_complexity > 12:
                suggestions.append(
                    "Complex promise chain detected - consider async/await or break into smaller functions"
                )

            closure_depth = entity.properties.get("closure_depth", 0)
            if closure_depth > 2:
                suggestions.append(
                    "Deep closure nesting - consider extracting functions to reduce scope complexity"
                )

            if entity.properties.get("modifies_prototype"):
                suggestions.append("Avoid prototype modification - use composition or ES6 classes")

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider dependency injection or module patterns"
            )

        return suggestions


__all__ = ["JavaScriptComplexityAnalyzer"]
