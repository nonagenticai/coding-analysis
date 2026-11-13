"""
C++-specific complexity analyzer.

Provides complexity analysis tailored to C++'s features like
templates, multiple inheritance, raw pointers, operator overloading,
exceptions, move semantics, and virtual functions.
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


class CppComplexityAnalyzer(LanguageComplexityAnalyzer):
    """C++-specific complexity analysis."""

    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze C++ entity complexity."""

        metrics = ServiceComplexityMetrics(
            language="cpp", entity_type=entity.type, entity_name=entity.name
        )

        # C++-specific complexity factors
        self.apply_body_metrics(metrics, entity, source)

        # C++-specific analysis
        if entity.type in ["function", "method", "constructor", "destructor"]:
            metrics.parameter_count = self._count_parameters(entity)

            if entity.properties:
                # Template complexity
                if entity.properties.get("template_parameters"):
                    metrics.risk_factors.append("template_parameters")

                # Operator overloading
                if entity.properties.get("is_operator_overload"):
                    metrics.risk_factors.append("operator_overload")

                # RAII complexity
                if entity.properties.get("manages_resources"):
                    metrics.risk_factors.append("manages_resources")

                # Exception handling
                if entity.properties.get("has_exceptions"):
                    metrics.risk_factors.append("exception_handling")

                # Move semantics
                if entity.properties.get("has_move_semantics"):
                    metrics.risk_factors.append("move_semantics")

                # Raw pointer usage
                if entity.properties.get("uses_raw_pointers"):
                    metrics.risk_factors.append("raw_pointers")

        elif entity.type == "class":
            metrics.coupling_factor = self._analyze_class_coupling(entity, parse_result)

            if entity.properties:
                # Multiple inheritance penalty
                inheritance_count = entity.properties.get("inheritance_count", 0)
                if inheritance_count > 1:
                    metrics.risk_factors.append("multiple_inheritance")

                # Virtual functions
                if entity.properties.get("has_virtual_functions"):
                    metrics.risk_factors.append("virtual_functions")

        metrics.maintainability_index = self._calculate_cpp_maintainability(metrics, entity)
        metrics.complexity_level = self._classify_complexity(metrics.cyclomatic_complexity)
        metrics.suggestions = self._generate_cpp_suggestions(metrics, entity)

        return metrics

    def get_language_specific_factors(self) -> dict[str, float]:
        """C++-specific complexity factors."""
        return {
            "template_penalty": 1.5,
            "multiple_inheritance_penalty": 1.4,
            "raw_pointer_penalty": 1.3,
            "operator_overload_penalty": 1.2,
            "exception_penalty": 1.3,
            "move_semantics_penalty": 1.1,
            "virtual_function_penalty": 1.2,
            "raii_bonus": 0.9,
            "const_correctness_bonus": 0.95,
        }

    def _analyze_class_coupling(self, entity: CoreCodeEntity, parse_result: ParseResult) -> float:
        """Analyze coupling for C++ classes."""
        coupling = 0.0

        # Count inheritance, template dependencies, and includes
        for relation in parse_result.relationships:
            if relation.source_id == entity.id:
                if relation.type in ["inherits", "extends"]:
                    coupling += 2.0  # High coupling for inheritance
                elif relation.type == "template_specialization":
                    coupling += 1.5  # Template coupling
                elif relation.type == "includes":
                    coupling += 0.5  # Include coupling
                else:
                    coupling += 1.0  # Other dependencies

        return min(coupling / 10.0, 1.0)  # Normalize to 0-1

    def _count_parameters(self, entity: CoreCodeEntity) -> int:
        """Count C++ function parameters."""
        signature = entity.signature or ""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip() and params_str.strip() != "void":
                # Basic parameter counting (C++ can be complex with templates)
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                return len(params)
        return 0

    def _calculate_cpp_maintainability(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> float:
        """Calculate C++-specific maintainability."""
        base_mi = 100.0

        # Penalties for C++-specific complexity
        base_mi -= metrics.cyclomatic_complexity * 2.5
        base_mi -= metrics.parameter_count * 2.0

        if entity.properties:
            # Template metaprogramming penalty
            template_params = entity.properties.get("template_parameters", [])
            if len(template_params) > 2:
                base_mi -= (len(template_params) - 2) * 8.0

            # Multiple inheritance penalty
            inheritance_count = entity.properties.get("inheritance_count", 0)
            if inheritance_count > 1:
                base_mi -= (inheritance_count - 1) * 10.0

            # Raw pointer penalty
            if entity.properties.get("uses_raw_pointers"):
                base_mi -= 15.0

            # Smart pointer bonus
            if entity.properties.get("uses_smart_pointers"):
                base_mi += 5.0

            # Const correctness bonus
            if entity.properties.get("const_correct"):
                base_mi += 5.0

        return max(0.0, min(100.0, base_mi))

    def _classify_complexity(self, complexity: float) -> ComplexityLevel:
        """Classify C++ complexity (stricter thresholds due to language complexity)."""
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

    def _generate_cpp_suggestions(
        self, metrics: ServiceComplexityMetrics, entity: CoreCodeEntity
    ) -> list[str]:
        """Generate C++-specific suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > 8:
            suggestions.append("Consider breaking this function into smaller, focused functions")

        if metrics.parameter_count > 4:
            suggestions.append("Consider using a parameter struct or builder pattern")

        if entity.properties:
            template_params = entity.properties.get("template_parameters", [])
            if len(template_params) > 2:
                suggestions.append(
                    "High template complexity - consider simplifying or using type traits"
                )

            inheritance_count = entity.properties.get("inheritance_count", 0)
            if inheritance_count > 1:
                suggestions.append(
                    "Multiple inheritance detected - consider composition or interface segregation"
                )

            if entity.properties.get("uses_raw_pointers"):
                suggestions.append(
                    "Use smart pointers (unique_ptr/shared_ptr) instead of raw pointers for resource management"
                )

            if entity.properties.get("is_operator_overload") and metrics.cyclomatic_complexity > 5:
                suggestions.append(
                    "Operator overloads should be simple - delegate complex logic to helper methods"
                )

            if entity.properties.get("has_exceptions") and not entity.properties.get(
                "exception_safe"
            ):
                suggestions.append(
                    "Ensure exception safety - use RAII and strong exception guarantee"
                )

            if entity.type == "class" and not entity.properties.get("has_rule_of_five"):
                if entity.properties.get("manages_resources"):
                    suggestions.append(
                        "Class manages resources - implement Rule of Five (destructor, copy/move constructors and assignments)"
                    )

        if metrics.coupling_factor > 0.7:
            suggestions.append(
                "High coupling detected - consider dependency injection or abstract interfaces"
            )

        return suggestions


__all__ = ["CppComplexityAnalyzer"]
