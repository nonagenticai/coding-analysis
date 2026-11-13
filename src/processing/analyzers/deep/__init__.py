"""Deep code analyzer package.

Public surface: ``DeepCodeAnalyzer`` plus the issue/result dataclasses and
``IssueType``/``Severity`` enums consumed across the analysis service.
"""

from .analyzer import DeepCodeAnalyzer
from .base import (
    CodeIssue,
    CodeRelationship,
    DeepAnalysisResult,
    DeepAnalysisServiceCodeEntity,
    IssueType,
    Severity,
)

__all__ = [
    "CodeIssue",
    "CodeRelationship",
    "DeepAnalysisResult",
    "DeepAnalysisServiceCodeEntity",
    "DeepCodeAnalyzer",
    "IssueType",
    "Severity",
]
