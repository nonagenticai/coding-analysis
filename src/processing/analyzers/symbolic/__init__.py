"""Symbolic Analysis Package.

This package provides symbolic analysis and advanced validation capabilities
for code analysis and security assessment.
"""

from .analyzer import SymbolicValidator
from .config import SymbolicAnalysisConfig, ValidationConfig
from .exceptions import (
    CrossHairAnalysisError,
    FileProcessingError,
    FormalVerificationError,
    PysaAnalysisError,
    SymbolicAnalysisError,
    SymbolicValidationError,
    ValidationError,
)
from .tools import CrossHairAnalyzer, FormalVerifier, PysaAnalyzer
from .utils import SafetyCalculator, SymbolicRecommendationGenerator, ValidationHelper

# Export SymbolicValidator as SymbolicAnalyzer for consistent naming
SymbolicAnalyzer = SymbolicValidator

__all__ = [
    # Main analyzer class
    "SymbolicAnalyzer",
    "SymbolicValidator",
    # Configuration
    "SymbolicAnalysisConfig",
    "ValidationConfig",
    # Exceptions
    "SymbolicAnalysisError",
    "CrossHairAnalysisError",
    "PysaAnalysisError",
    "FormalVerificationError",
    "SymbolicValidationError",
    "FileProcessingError",
    "ValidationError",
    # Tools
    "CrossHairAnalyzer",
    "FormalVerifier",
    "PysaAnalyzer",
    # Utilities
    "SymbolicRecommendationGenerator",
    "SafetyCalculator",
    "ValidationHelper",
]
