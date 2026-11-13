"""
ComplexityAnalysisService — multi-language complexity analysis facade.

Wraps the per-language analyzers (Python/JavaScript/TypeScript/Java/Go/Rust/
C++/C#) behind a single entry point that produces ``FileComplexityReport``
instances for parsed files. Re-exported from the package ``__init__`` for
convenience.
"""

from __future__ import annotations

import logging

from processing.languages.base import ParseResult

from .base import (
    ComplexityLevel,
    FileComplexityReport,
    LanguageComplexityAnalyzer,
    ServiceComplexityMetrics,
)
from .body_metrics import SourceContext
from .cpp_analyzer import CppComplexityAnalyzer
from .csharp_analyzer import CSharpComplexityAnalyzer
from .go_analyzer import GoComplexityAnalyzer
from .java_analyzer import JavaComplexityAnalyzer
from .javascript_analyzer import JavaScriptComplexityAnalyzer
from .python_analyzer import PythonComplexityAnalyzer
from .rust_analyzer import RustComplexityAnalyzer
from .typescript_analyzer import TypeScriptComplexityAnalyzer

logger = logging.getLogger(__name__)


class ComplexityAnalysisService:
    """Service for analyzing code complexity across multiple languages."""

    def __init__(self) -> None:
        """Initialize the complexity analysis service."""
        self._analyzers: dict[str, LanguageComplexityAnalyzer] = {
            "python": PythonComplexityAnalyzer(),
            "javascript": JavaScriptComplexityAnalyzer(),
            "typescript": TypeScriptComplexityAnalyzer(),
            "java": JavaComplexityAnalyzer(),
            "go": GoComplexityAnalyzer(),
            "rust": RustComplexityAnalyzer(),
            "cpp": CppComplexityAnalyzer(),
            "csharp": CSharpComplexityAnalyzer(),
        }

    def analyze_file_complexity(
        self, parse_result: ParseResult, content: str | None = None
    ) -> FileComplexityReport:
        """Analyze complexity for an entire file.

        Args:
            parse_result: Parsed file result containing entities and relationships
            content: The file's source text. Per-entity metrics are computed from
                the entity bodies in it; without it every entity is reported as
                ``computed=False`` rather than with placeholder numbers.

        Returns:
            Comprehensive file complexity report
        """
        analyzer = self._analyzers.get(parse_result.language.lower())
        if not analyzer:
            # Fall back to generic analysis
            analyzer = self._create_generic_analyzer(parse_result.language)

        source = (
            SourceContext(parse_result.language, parse_result.file_path, content)
            if content is not None
            else None
        )

        report = FileComplexityReport(
            file_path=parse_result.file_path,
            language=parse_result.language,
            total_loc=parse_result.lines_of_code,
        )

        entity_complexities = []
        total_complexity = 0.0
        max_complexity = 0.0
        complex_count = 0
        computed_count = 0

        for entity in parse_result.entities:
            metrics = analyzer.analyze_entity_complexity(entity, parse_result, source)
            entity_complexities.append(metrics)
            if not metrics.computed:
                continue
            computed_count += 1

            total_complexity += metrics.cyclomatic_complexity
            max_complexity = max(max_complexity, metrics.cyclomatic_complexity)

            if metrics.complexity_level in [
                ComplexityLevel.COMPLEX,
                ComplexityLevel.VERY_COMPLEX,
            ]:
                complex_count += 1
                report.hotspots.append(f"{entity.type} {entity.name}")

        report.entity_metrics = entity_complexities
        report.total_entities = len(parse_result.entities)
        report.entities_not_computed = len(parse_result.entities) - computed_count
        report.complex_entities = complex_count
        report.average_complexity = total_complexity / max(computed_count, 1)
        report.max_complexity = max_complexity

        # Calculate quality score -- never from zero measurements.
        if report.total_entities > 0 and computed_count == 0:
            report.metrics_computed = False
            report.quality_score = None
        else:
            report.quality_score = self._calculate_quality_score(report)

        # Generate file-level suggestions
        report.refactoring_suggestions = self._generate_file_suggestions(report)

        return report

    def _create_generic_analyzer(self, language: str) -> LanguageComplexityAnalyzer:
        """Create a generic analyzer for unsupported languages."""
        logger.info(
            f"No specific analyzer for language '{language}', using Java analyzer as default"
        )
        # Use Java analyzer as a reasonable default
        return JavaComplexityAnalyzer()

    def _calculate_quality_score(self, report: FileComplexityReport) -> float:
        """Calculate overall file quality score (0-100)."""
        if report.total_entities == 0:
            return 100.0

        # Start with perfect score
        score = 100.0

        # Penalize high complexity
        complexity_penalty = (report.average_complexity - 5) * 2
        score -= max(0, complexity_penalty)

        # Penalize high percentage of complex entities
        if report.total_entities > 0:
            complex_ratio = report.complex_entities / report.total_entities
            score -= complex_ratio * 30

        # Penalize very large files
        if report.total_loc > 500:
            size_penalty = (report.total_loc - 500) / 100
            score -= size_penalty

        return max(0.0, min(100.0, score))

    def _generate_file_suggestions(self, report: FileComplexityReport) -> list[str]:
        """Generate file-level refactoring suggestions."""
        suggestions = []

        if report.total_loc > 1000:
            suggestions.append("Large file detected - consider splitting into multiple files")

        if report.complex_entities > report.total_entities * 0.3:
            suggestions.append("High ratio of complex entities - consider refactoring")

        if report.average_complexity > 15:
            suggestions.append("High average complexity - review architectural design")

        if len(report.hotspots) > 5:
            suggestions.append("Multiple complexity hotspots - prioritize refactoring efforts")

        return suggestions

    def get_supported_languages(self) -> list[str]:
        """Get list of languages with specific complexity analyzers."""
        return list(self._analyzers.keys())


# Global service instance
_complexity_service: ComplexityAnalysisService | None = None


def get_complexity_service() -> ComplexityAnalysisService:
    """Get or create the global complexity analysis service."""
    global _complexity_service

    if _complexity_service is None:
        _complexity_service = ComplexityAnalysisService()
        logger.info("Initialized complexity analysis service")

    return _complexity_service


__all__ = [
    "ComplexityAnalysisService",
    "ServiceComplexityMetrics",
    "get_complexity_service",
]
