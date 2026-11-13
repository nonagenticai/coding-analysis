"""
Base classes and shared types for complexity analysis.

This module provides the abstract base class and shared data structures
for language-specific complexity analyzers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from models.core_models import CoreCodeEntity
from processing.languages.base import ParseResult

from .body_metrics import SourceContext


class ComplexityLevel(StrEnum):
    """Complexity level classifications."""

    TRIVIAL = "trivial"  # 0-5
    BASIC = "basic"  # 6-10
    MODERATE = "moderate"  # 11-20
    COMPLEX = "complex"  # 21-40
    VERY_COMPLEX = "very_complex"  # 41+


@dataclass
class ServiceComplexityMetrics:
    """Comprehensive complexity metrics for a code entity."""

    # Basic metrics
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    lines_of_code: int = 0

    # Language-specific metrics
    nesting_depth: int = 0
    parameter_count: int = 0
    dependency_count: int = 0

    # Advanced metrics
    halstead_difficulty: float = 0.0
    maintainability_index: float = 0.0
    coupling_factor: float = 0.0

    # Classifications
    complexity_level: ComplexityLevel = ComplexityLevel.TRIVIAL
    risk_factors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Metadata
    language: str = ""
    entity_type: str = ""
    entity_name: str = ""

    # False when the body could not be parsed/located: the numeric metrics are
    # then zero and must not be read as "simple code".
    computed: bool = True
    not_computed_reason: str | None = None


@dataclass
class FileComplexityReport:
    """Complexity report for an entire file."""

    file_path: str
    language: str
    total_loc: int

    # Aggregated metrics
    average_complexity: float = 0.0
    max_complexity: float = 0.0
    total_entities: int = 0
    complex_entities: int = 0

    # Entity-level details
    entity_metrics: list[ServiceComplexityMetrics] = field(default_factory=list)
    hotspots: list[str] = field(default_factory=list)

    entities_not_computed: int = 0

    # File-level recommendations
    refactoring_suggestions: list[str] = field(default_factory=list)
    # None when the file has entities but NONE could be measured: a score built
    # from zero measurements would read as clean code.
    quality_score: float | None = 0.0
    metrics_computed: bool = True


class LanguageComplexityAnalyzer(ABC):
    """Abstract base class for language-specific complexity analysis."""

    @abstractmethod
    def analyze_entity_complexity(
        self,
        entity: CoreCodeEntity,
        parse_result: ParseResult,
        source: SourceContext | None = None,
    ) -> ServiceComplexityMetrics:
        """Analyze complexity for a specific entity.

        *source* carries the file content + parse tree; without it the
        body metrics are reported as not computed.
        """

    @abstractmethod
    def get_language_specific_factors(self) -> dict[str, float]:
        """Get language-specific complexity factors."""

    def apply_body_metrics(
        self,
        metrics: ServiceComplexityMetrics,
        entity: CoreCodeEntity,
        source: SourceContext | None,
    ) -> None:
        """Fill cyclomatic/cognitive/nesting/LOC from the entity's real body.

        Leaves the numbers at zero and sets ``computed=False`` (with a reason)
        when the body cannot be analysed - never a placeholder value.
        """
        if source is None:
            metrics.computed = False
            metrics.not_computed_reason = "source content not available"
            return
        start = getattr(entity, "line_number", None)
        body = source.metrics_for(start if isinstance(start, int) else None)
        metrics.computed = body.computed
        metrics.not_computed_reason = body.not_computed_reason
        if not body.computed:
            return
        metrics.cyclomatic_complexity = float(body.cyclomatic)
        metrics.cognitive_complexity = float(body.cognitive)
        metrics.nesting_depth = body.nesting
        metrics.lines_of_code = body.loc

    def _classify_complexity_standard(self, complexity: float) -> ComplexityLevel:
        """Standard complexity classification."""
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

    def _calculate_standard_maintainability(self, metrics: ServiceComplexityMetrics) -> float:
        """Calculate standard maintainability index (0-100)."""
        # Streamlined Microsoft maintainability index formula
        if metrics.lines_of_code == 0:
            return 100.0

        loc = max(metrics.lines_of_code, 1)
        cc = max(metrics.cyclomatic_complexity, 1)

        mi = 171 - 5.2 * math.log(loc) - 0.23 * cc - 16.2 * math.log(loc)
        return max(0.0, min(100.0, mi))

    def _count_parameters_from_signature(self, signature: str) -> int:
        """Count parameters from a function signature string."""
        if "(" in signature and ")" in signature:
            params_str = signature[signature.find("(") + 1 : signature.find(")")]
            if params_str.strip():
                return len([p.strip() for p in params_str.split(",") if p.strip()])
        return 0


__all__ = [
    "ComplexityLevel",
    "ServiceComplexityMetrics",
    "FileComplexityReport",
    "LanguageComplexityAnalyzer",
]
