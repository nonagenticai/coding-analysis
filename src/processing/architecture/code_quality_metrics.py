"""Module for code quality metrics calculation and scoring."""

from __future__ import annotations

import math
from typing import Any

from models.architecture_models import ArchitectureQualityIssue


class CodeQualityMetrics:
    """Handler for code quality metrics and scoring calculations."""

    def __init__(self) -> None:
        """Initialize the quality metrics calculator."""
        # Severity weights for quality score calculation
        self.severity_weights = {"critical": 25, "high": 10, "medium": 5, "low": 2}

    def calculate_quality_score(self, issues: list[ArchitectureQualityIssue]) -> float:
        """Calculate overall quality score from issues."""
        if not issues:
            return 100.0

        total_penalty = 0
        for issue in issues:
            severity = issue.severity.lower()
            penalty = self.severity_weights.get(severity, 5)
            total_penalty += penalty

        # Apply logarithmic scaling to prevent single issues from dominating
        scaled_penalty = math.log(total_penalty + 1) * 10

        # Calculate score
        score = max(0.0, 100.0 - scaled_penalty)
        return round(score, 1)

    def score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    def count_issues_by_severity(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by severity level."""
        counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for issue in issues:
            severity = issue.severity.lower()
            if severity in counts:
                counts[severity] += 1
            else:
                counts["medium"] += 1

        return counts

    def count_issues_by_tool(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by tool."""
        counts: dict[str, int] = {}

        for issue in issues:
            tool = issue.tool
            if tool not in counts:
                counts[tool] = 0
            counts[tool] += 1

        return counts

    def count_issues_by_category(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by category."""
        counts: dict[str, int] = {}

        for issue in issues:
            category = issue.category
            if category not in counts:
                counts[category] = 0
            counts[category] += 1

        return counts

    def calculate_tool_coverage(self, tools_run: list[str], available_tools: list[str]) -> float:
        """Calculate percentage of available tools that were run."""
        if not available_tools:
            return 100.0

        coverage = (len(tools_run) / len(available_tools)) * 100
        return round(coverage, 1)

    def calculate_severity_distribution(
        self, issues: list[ArchitectureQualityIssue]
    ) -> dict[str, float]:
        """Calculate percentage distribution of issues by severity."""
        severity_counts = self.count_issues_by_severity(issues)
        total_issues = sum(severity_counts.values())

        if total_issues == 0:
            return {severity: 0.0 for severity in severity_counts}

        distribution = {}
        for severity, count in severity_counts.items():
            percentage = (count / total_issues) * 100
            distribution[severity] = round(percentage, 1)

        return distribution

    def get_quality_trend_indicators(
        self,
        current_issues: list[ArchitectureQualityIssue],
        previous_issues: list[ArchitectureQualityIssue] | None = None,
    ) -> dict[str, Any]:
        """Calculate quality trend indicators compared to previous analysis."""
        current_score = self.calculate_quality_score(current_issues)
        current_counts = self.count_issues_by_severity(current_issues)

        indicators = {
            "current_score": current_score,
            "current_grade": self.score_to_grade(current_score),
            "total_issues": len(current_issues),
            "severity_counts": current_counts,
        }

        if previous_issues is not None:
            previous_score = self.calculate_quality_score(previous_issues)
            previous_counts = self.count_issues_by_severity(previous_issues)

            indicators.update(
                {
                    "score_change": round(current_score - previous_score, 1),
                    "issues_change": len(current_issues) - len(previous_issues),
                    "critical_change": current_counts["critical"] - previous_counts["critical"],
                    "high_change": current_counts["high"] - previous_counts["high"],
                    "improving": current_score > previous_score,
                }
            )

        return indicators
