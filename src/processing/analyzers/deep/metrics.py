"""Metrics + scoring + recommendations for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

from .base import (
    IssueType,
    Severity,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def calculate_metrics(analyzer: DeepCodeAnalyzer) -> dict[str, Any]:
    """Calculate overall codebase metrics."""
    total_lines = 0
    total_functions = 0
    total_classes = 0
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            total_lines += len(content.splitlines())
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error calculating metrics for {file_path}: {e}")
    issue_breakdown: dict[str, int] = {}
    for issue in analyzer.issues:
        key = f"{issue.type.value}_{issue.severity.value}"
        issue_breakdown[key] = issue_breakdown.get(key, 0) + 1
    return {
        "total_files": len(analyzer.python_files),
        "total_lines": total_lines,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "total_issues": len(analyzer.issues),
        "issue_breakdown": issue_breakdown,
        "average_issues_per_file": len(analyzer.issues) / max(len(analyzer.python_files), 1),
    }


def calculate_overall_score(analyzer: DeepCodeAnalyzer) -> float:
    """Calculate overall code quality score (0-10)."""
    if not analyzer.issues:
        return 10.0
    severity_weights = {
        Severity.CRITICAL: 4.0,
        Severity.HIGH: 2.0,
        Severity.MEDIUM: 1.0,
        Severity.LOW: 0.5,
        Severity.INFO: 0.1,
    }
    total_penalty = sum(severity_weights.get(issue.severity, 1.0) for issue in analyzer.issues)
    penalty_per_file = total_penalty / max(len(analyzer.python_files), 1)
    score = max(0.0, 10.0 - penalty_per_file)
    return round(score, 2)


def generate_recommendations(analyzer: DeepCodeAnalyzer) -> list[str]:
    """Generate high-level recommendations."""
    recommendations: list[Any] = []
    type_counts: dict[Any, int] = {}
    for issue in analyzer.issues:
        type_counts[issue.type] = type_counts.get(issue.type, 0) + 1
    if type_counts.get(IssueType.SECURITY, 0) > 0:
        recommendations.append(
            "Review security practices, especially around credential storage and input validation"
        )
    if type_counts.get(IssueType.COMPLEXITY, 0) > 5:
        recommendations.append("Consider refactoring complex functions to update maintainability")
    if type_counts.get(IssueType.DOCUMENTATION, 0) > 10:
        recommendations.append("Update code documentation with docstrings and comments")
    if type_counts.get(IssueType.BUG_RISK, 0) > 0:
        recommendations.append(
            "Address potential bug risks, especially exception handling and mutable defaults"
        )
    if type_counts.get(IssueType.PERFORMANCE, 0) > 3:
        recommendations.append(
            "Optimize performance bottlenecks, especially in loops and string operations"
        )
    return recommendations
