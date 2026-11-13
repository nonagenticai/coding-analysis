#!/usr/bin/env python3
"""Quality metrics module."""

from __future__ import annotations

import logging
from typing import Any

from .quality_models import AppToolResult

logger = logging.getLogger(__name__)


class QualityMetricsCalculator:
    """Calculates quality metrics and scores from tool results."""

    def __init__(self):
        """Initialize quality metrics calculator."""
        self.error_weight = 10  # Points deducted per error
        self.warning_weight = 5  # Points deducted per warning
        self.info_weight = 2  # Points deducted per info issue
        self.base_score = 100.0  # Starting score for each tool

    def calculate_quality_score(self, tools_results: dict[str, AppToolResult]) -> float:
        """Calculate overall quality score from tool results."""
        if not tools_results:
            return 0.0

        total_score = 0.0
        total_tools = 0

        for tool_result in tools_results.values():
            if tool_result.success:
                tool_score = self._calculate_tool_score(tool_result)
                total_score += tool_score
                total_tools += 1

        return total_score / total_tools if total_tools > 0 else 0.0

    def _calculate_tool_score(self, tool_result: AppToolResult) -> float:
        """Calculate score for a single tool result."""
        # Calculate score based on issues found
        len(tool_result.issues)
        error_count = sum(1 for issue in tool_result.issues if issue.severity == "error")
        warning_count = sum(1 for issue in tool_result.issues if issue.severity == "warning")
        info_count = sum(
            1 for issue in tool_result.issues if issue.severity not in ["error", "warning"]
        )

        # Scoring formula (100 - penalties)
        tool_score = self.base_score
        tool_score -= error_count * self.error_weight  # -10 points per error
        tool_score -= warning_count * self.warning_weight  # -5 points per warning
        tool_score -= info_count * self.info_weight  # -2 points per other issue

        return max(0.0, tool_score)  # Don't go below 0

    def calculate_total_issues(self, tools_results: dict[str, AppToolResult]) -> int:
        """Calculate total number of issues across all tools."""
        return sum(len(result.issues) for result in tools_results.values())

    def calculate_issue_breakdown(self, tools_results: dict[str, AppToolResult]) -> dict[str, int]:
        """Calculate breakdown of issues by severity."""
        breakdown = {"error": 0, "warning": 0, "info": 0, "other": 0}

        for tool_result in tools_results.values():
            for issue in tool_result.issues:
                severity = issue.severity.lower()
                if severity in breakdown:
                    breakdown[severity] += 1
                else:
                    breakdown["other"] += 1

        return breakdown

    def calculate_tool_success_rate(self, tools_results: dict[str, AppToolResult]) -> float:
        """Calculate percentage of tools that executed successfully."""
        if not tools_results:
            return 0.0

        successful_tools = sum(1 for result in tools_results.values() if result.success)
        return (successful_tools / len(tools_results)) * 100.0

    def calculate_fixable_issues_count(self, tools_results: dict[str, AppToolResult]) -> int:
        """Calculate number of issues that can be automatically fixed."""
        fixable_count = 0

        for tool_result in tools_results.values():
            for issue in tool_result.issues:
                if issue.fixable:
                    fixable_count += 1

        return fixable_count

    def calculate_execution_stats(self, tools_results: dict[str, AppToolResult]) -> dict[str, Any]:
        """Calculate execution statistics for tool performance."""
        if not tools_results:
            return {}

        execution_times = [result.execution_time for result in tools_results.values()]

        return {
            "total_execution_time": sum(execution_times),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "tools_count": len(tools_results),
        }

    def generate_quality_summary(self, tools_results: dict[str, AppToolResult]) -> dict[str, Any]:
        """Generate comprehensive quality analysis summary."""
        overall_score = self.calculate_quality_score(tools_results)
        total_issues = self.calculate_total_issues(tools_results)
        issue_breakdown = self.calculate_issue_breakdown(tools_results)
        success_rate = self.calculate_tool_success_rate(tools_results)
        fixable_issues = self.calculate_fixable_issues_count(tools_results)
        execution_stats = self.calculate_execution_stats(tools_results)

        # Determine quality grade
        grade = self._calculate_quality_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            tools_results, issue_breakdown, fixable_issues
        )

        return {
            "overall_score": overall_score,
            "quality_grade": grade,
            "total_issues": total_issues,
            "issue_breakdown": issue_breakdown,
            "fixable_issues": fixable_issues,
            "success_rate": success_rate,
            "execution_stats": execution_stats,
            "recommendations": recommendations,
        }

    def _calculate_quality_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_recommendations(
        self,
        tools_results: dict[str, AppToolResult],
        issue_breakdown: dict[str, int],
        fixable_issues: int,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations: list[str] = []

        # Check for high error count
        if issue_breakdown["error"] > 5:
            recommendations.append(
                "High number of errors detected. Consider fixing critical issues first."
            )

        # Check for fixable issues
        if fixable_issues > 0:
            recommendations.append(
                f"{fixable_issues} issues can be automatically fixed. "
                "Consider running auto-fix tools like ruff format or Prettier."
            )

        # Check for failed tools
        failed_tools = [name for name, result in tools_results.items() if not result.success]
        if failed_tools:
            recommendations.append(
                f"Tools failed to execute: {', '.join(failed_tools)}. Check tool configuration and dependencies."
            )

        # Check for security issues (from bandit)
        security_issues = 0
        for tool_result in tools_results.values():
            if tool_result.tool == "bandit":
                security_issues = len(tool_result.issues)
                break

        if security_issues > 0:
            recommendations.append(
                f"Security scan found {security_issues} potential vulnerabilities. Review and address security issues."
            )

        # Check for complexity issues (from radon)
        complexity_issues = 0
        for tool_result in tools_results.values():
            if tool_result.tool == "radon":
                complexity_issues = len(
                    [issue for issue in tool_result.issues if "High complexity" in issue.message]
                )
                break

        if complexity_issues > 0:
            recommendations.append(
                f"Found {complexity_issues} functions with high complexity. Consider refactoring complex functions."
            )

        # General recommendations based on total issues
        total_issues = sum(issue_breakdown.values())
        if total_issues == 0:
            recommendations.append("Excellent! No quality issues detected.")
        elif total_issues < 5:
            recommendations.append("Good code quality with minimal issues.")
        elif total_issues < 15:
            recommendations.append("Moderate number of issues. Consider addressing warnings.")
        else:
            recommendations.append(
                "High number of issues detected. Comprehensive review recommended."
            )

        return recommendations


# Create global quality metrics calculator instance
quality_metrics = QualityMetricsCalculator()
