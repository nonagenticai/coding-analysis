"""
Language-Specific Complexity Analysis Module

Canonical home for complexity analyzers. The legacy monolith
``src/services/complexity_analyzer.py`` previously hosted equivalent code; a
diff-audit confirmed this package is the canonical (newer) source for every
shared symbol — base helpers (``_classify_complexity_standard``,
``_calculate_standard_maintainability``, ``_count_parameters_from_signature``)
are tested only here, the Rust ``_count_parameters`` comment fix lives only
here, and ``ComplexityAnalysisService.analyze_file_complexity`` carries the
fuller Args/Returns docstring only here. The shim is removed in a follow-up
commit and consumers import from this package directly.

This module provides language-specific complexity analyzers for various
programming languages. Each analyzer is tailored to the unique features
and patterns of its respective language.

Supported Languages:
- Python: decorators, comprehensions, metaclasses, generators, async
- Java: inheritance, interfaces, annotations, generics, exceptions
- JavaScript: callbacks, promises, closures, prototypes, dynamic typing
- TypeScript: type system, generics, decorators, conditional/mapped types
- Rust: lifetimes, unsafe blocks, macros, traits, ownership
- Go: interfaces, goroutines, channels, error handling
- C++: templates, multiple inheritance, RAII, operator overloading
- C#: LINQ, async/await, delegates, reflection, generics

Example Usage:
    from processing.analyzers.complexity_analyzers import (
        get_complexity_analyzer,
        ComplexityAnalysisService,
        ComplexityLevel,
    )

    # Get a language-specific analyzer
    analyzer = get_complexity_analyzer("python")
    metrics = analyzer.analyze_entity_complexity(entity, parse_result)

    # Or use the unified service
    service = get_complexity_service()
    report = service.analyze_file_complexity(parse_result)
"""

from __future__ import annotations

import logging

from .base import (
    ComplexityLevel,
    FileComplexityReport,
    LanguageComplexityAnalyzer,
    ServiceComplexityMetrics,
)
from .cpp_analyzer import CppComplexityAnalyzer
from .csharp_analyzer import CSharpComplexityAnalyzer
from .go_analyzer import GoComplexityAnalyzer
from .java_analyzer import JavaComplexityAnalyzer
from .javascript_analyzer import JavaScriptComplexityAnalyzer
from .python_analyzer import PythonComplexityAnalyzer
from .rust_analyzer import RustComplexityAnalyzer
from .service import (
    ComplexityAnalysisService,
    get_complexity_service,
)
from .typescript_analyzer import TypeScriptComplexityAnalyzer

logger = logging.getLogger(__name__)

# Analyzer registry - maps language names to analyzer classes
_ANALYZER_REGISTRY: dict[str, type[LanguageComplexityAnalyzer]] = {
    "python": PythonComplexityAnalyzer,
    "java": JavaComplexityAnalyzer,
    "javascript": JavaScriptComplexityAnalyzer,
    "typescript": TypeScriptComplexityAnalyzer,
    "rust": RustComplexityAnalyzer,
    "go": GoComplexityAnalyzer,
    "cpp": CppComplexityAnalyzer,
    "c++": CppComplexityAnalyzer,
    "csharp": CSharpComplexityAnalyzer,
    "c#": CSharpComplexityAnalyzer,
}

# Cached analyzer instances
_analyzer_cache: dict[str, LanguageComplexityAnalyzer] = {}


def get_complexity_analyzer(language: str) -> LanguageComplexityAnalyzer:
    """Get a complexity analyzer for the specified language.

    Args:
        language: Programming language name (case-insensitive)

    Returns:
        Language-specific complexity analyzer instance

    Note:
        Returns Java analyzer as default for unsupported languages
    """
    lang_key = language.lower()

    if lang_key in _analyzer_cache:
        return _analyzer_cache[lang_key]

    analyzer_class = _ANALYZER_REGISTRY.get(lang_key)
    if analyzer_class is None:
        logger.info(
            f"No specific analyzer for language '{language}', using Java analyzer as default"
        )
        analyzer_class = JavaComplexityAnalyzer
        lang_key = "java"  # Cache under correct key

    analyzer = analyzer_class()
    _analyzer_cache[lang_key] = analyzer
    return analyzer


def get_supported_languages() -> list[str]:
    """Get list of languages with specific complexity analyzers.

    Returns:
        List of supported language names
    """
    # Return unique language names (excluding aliases like 'c++' and 'c#')
    return ["python", "java", "javascript", "typescript", "rust", "go", "cpp", "csharp"]


# Public API exports
__all__ = [
    # Base types
    "ComplexityLevel",
    "ServiceComplexityMetrics",
    "FileComplexityReport",
    "LanguageComplexityAnalyzer",
    # Language-specific analyzers
    "PythonComplexityAnalyzer",
    "JavaComplexityAnalyzer",
    "JavaScriptComplexityAnalyzer",
    "TypeScriptComplexityAnalyzer",
    "RustComplexityAnalyzer",
    "GoComplexityAnalyzer",
    "CppComplexityAnalyzer",
    "CSharpComplexityAnalyzer",
    # Service and factory
    "ComplexityAnalysisService",
    "get_complexity_service",
    "get_complexity_analyzer",
    "get_supported_languages",
]
