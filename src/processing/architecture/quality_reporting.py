"""Module for quality reporting and recommendation generation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict

from models.architecture_models import ArchitectureQualityIssue, QualityReport


class RecommendationTemplate(TypedDict):
    """Type definition for recommendation templates."""

    low_threshold: int
    message: str


class QualityReporting:
    """Handler for quality report generation and recommendations."""

    def __init__(self) -> None:
        """Initialize the quality reporting system."""
        self.recommendation_templates: dict[str, RecommendationTemplate] = {
            "security": {
                "low_threshold": 1,
                "message": "Resolve {count} security issues to prevent vulnerabilities",
            },
            "type-safety": {
                "low_threshold": 5,
                "message": "Add type annotations and resolve type errors to improve code reliability",
            },
            "maintainability": {
                "low_threshold": 1,
                "message": "Refactor complex functions to improve code maintainability",
            },
            "style": {
                "low_threshold": 10,
                "message": "Configure and enforce consistent code style with automated formatting",
            },
            "performance": {
                "low_threshold": 3,
                "message": "Address performance issues to improve application speed",
            },
            "correctness": {
                "low_threshold": 1,
                "message": "Fix correctness issues to prevent bugs",
            },
        }

    def generate_quality_report(
        self,
        codebase_id: str,
        issues: list[ArchitectureQualityIssue],
        overall_score: float,
        grade: str,
        summary: dict[str, int],
        execution_time: float,
        language: str = "unknown",
        additional_metrics: dict[str, Any] | None = None,
        inconclusive: bool = False,
        partial: bool = False,
        tools_attempted: list[str] | None = None,
        tools_succeeded: list[str] | None = None,
        tools_failed: list[str] | None = None,
    ) -> QualityReport:
        """Generate a comprehensive quality report.

        When ``inconclusive`` is True (no static-analysis tool completed), an
        empty issue list means "unknown", not "perfect": the recommendations
        and downstream renderers must not treat it as a pass.

        ``language`` is what the analyzer detected, carried on the report so
        consumers report the measured language instead of echoing a request
        field the caller may have left blank.
        """

        recommendations = self.generate_recommendations(
            issues, overall_score, inconclusive=inconclusive
        )

        # Add additional metrics if provided
        if additional_metrics:
            summary.update(additional_metrics)

        return QualityReport(
            job_id=f"quality-{codebase_id}",
            codebase_id=codebase_id,
            generated_at=datetime.now(UTC).isoformat(),
            overall_score=overall_score,
            grade=grade,
            language=language,
            inconclusive=inconclusive,
            partial=partial,
            tools_attempted=tools_attempted or [],
            tools_succeeded=tools_succeeded or [],
            tools_failed=tools_failed or [],
            issues=issues,
            summary=summary,
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def generate_recommendations(
        self,
        issues: list[ArchitectureQualityIssue],
        score: float,
        inconclusive: bool = False,
    ) -> list[str]:
        """Generate recommendations based on issues found."""
        if inconclusive:
            return [
                "Analysis inconclusive: no static-analysis tool completed "
                "successfully — this is NOT a pass; re-run or check tool timeouts."
            ]

        recommendations = []

        # Count issues by category
        category_counts = self._count_issues_by_category(issues)

        # Generate category-specific recommendations
        for category, count in category_counts.items():
            template = self.recommendation_templates.get(category)
            if template and count >= template["low_threshold"]:
                message = template["message"]
                if "{count}" in message:
                    message = message.format(count=count)
                recommendations.append(message)

        # Overall recommendations based on score
        if score < 60:
            recommendations.append(
                "Consider setting up pre-commit hooks to catch issues before commit"
            )
            recommendations.append("Implement continuous integration with quality gates")
        elif score < 80:
            recommendations.append("Enable stricter linting rules to maintain code quality")
            recommendations.append("Add more comprehensive test coverage")
        else:
            recommendations.append("Good code quality! Consider adding more comprehensive tests")
            recommendations.append("Consider implementing advanced static analysis tools")

        return recommendations

    def _count_issues_by_category(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by category."""
        counts: dict[str, int] = {}

        for issue in issues:
            category = issue.category
            if category not in counts:
                counts[category] = 0
            counts[category] += 1

        return counts

    def generate_summary_statistics(
        self,
        issues: list[ArchitectureQualityIssue],
        total_files_analyzed: int,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Generate comprehensive summary statistics."""

        # Count issues by severity
        severity_counts = self._count_issues_by_severity(issues)

        # Count issues by tool
        tool_counts = self._count_issues_by_tool(issues)

        # Basic summary
        summary = {
            "total_files_analyzed": total_files_analyzed,
            "total_issues": len(issues),
            "critical_issues": severity_counts.get("critical", 0),
            "high_issues": severity_counts.get("high", 0),
            "medium_issues": severity_counts.get("medium", 0),
            "low_issues": severity_counts.get("low", 0),
        }

        # Add tool-specific counts for common tools
        common_tools = ["ruff", "ruff-format", "ty", "bandit", "radon", "eslint", "tsc"]
        for tool in common_tools:
            summary[f"{tool}_issues"] = tool_counts.get(tool, 0)

        # Add additional metrics if provided
        if metrics:
            summary.update(metrics)

        return summary

    def _count_issues_by_severity(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
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

    def _count_issues_by_tool(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by tool."""
        counts: dict[str, int] = {}

        for issue in issues:
            tool = issue.tool
            if tool not in counts:
                counts[tool] = 0
            counts[tool] += 1

        return counts

    def generate_executive_summary(
        self, report: QualityReport, trends: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate an executive summary of the quality report."""

        summary: dict[str, Any] = {
            "overall_grade": report.grade,
            "overall_score": report.overall_score,
            "total_issues": len(report.issues),
            "critical_issues": report.summary.get("critical_issues", 0),
            "files_analyzed": report.summary.get("total_files_analyzed", 0),
            "execution_time": report.execution_time,
            "key_findings": [],
            "priority_actions": [],
        }

        # Generate key findings
        critical_count = report.summary.get("critical_issues", 0)
        high_count = report.summary.get("high_issues", 0)

        if critical_count > 0:
            summary["key_findings"].append(
                f"{critical_count} critical issues require immediate attention"
            )

        if high_count > 10:
            summary["key_findings"].append(f"{high_count} high-severity issues detected")

        if report.overall_score < 70:
            summary["key_findings"].append("Code quality below acceptable threshold")

        # Generate priority actions (top 3 recommendations)
        summary["priority_actions"] = report.recommendations[:3]

        # Add trend information if available
        if trends:
            summary["trends"] = trends

        return summary

    def generate_detailed_breakdown(self, issues: list[ArchitectureQualityIssue]) -> dict[str, Any]:
        """Generate detailed breakdown of issues."""

        breakdown: dict[str, Any] = {
            "by_severity": self._count_issues_by_severity(issues),
            "by_category": self._count_issues_by_category(issues),
            "by_tool": self._count_issues_by_tool(issues),
            "by_file": {},
            "top_issues": [],
        }

        # Count issues by file
        file_counts: dict[str, int] = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in file_counts:
                file_counts[file_path] = 0
            file_counts[file_path] += 1

        # Get top 10 files with most issues
        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        breakdown["by_file"] = dict(sorted_files[:10])

        # Get top critical/high severity issues
        high_priority_issues = [
            issue for issue in issues if issue.severity.lower() in ["critical", "high"]
        ]

        # Sort by severity (critical first) and take top 10
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        high_priority_issues.sort(key=lambda x: severity_order.get(x.severity.lower(), 3))

        breakdown["top_issues"] = [
            {
                "severity": issue.severity,
                "message": issue.message,
                "file": issue.file_path,
                "line": issue.line_number,
                "tool": issue.tool,
                "category": issue.category,
            }
            for issue in high_priority_issues[:10]
        ]

        return breakdown

    def format_recommendations_by_priority(
        self, recommendations: list[str]
    ) -> dict[str, list[str]]:
        """Format recommendations by priority level."""

        formatted: dict[str, list[str]] = {
            "immediate": [],
            "short_term": [],
            "long_term": [],
        }

        # Categorize recommendations by keywords
        immediate_keywords = ["critical", "security", "vulnerability", "immediate"]
        short_term_keywords = ["refactor", "type", "maintainability", "complexity"]

        for rec in recommendations:
            rec_lower = rec.lower()

            if any(keyword in rec_lower for keyword in immediate_keywords):
                formatted["immediate"].append(rec)
            elif any(keyword in rec_lower for keyword in short_term_keywords):
                formatted["short_term"].append(rec)
            else:
                formatted["long_term"].append(rec)

        return formatted

    def generate_comparison_report(
        self,
        current_report: QualityReport,
        previous_report: QualityReport | None = None,
    ) -> dict[str, Any]:
        """Generate comparison report between current and previous analysis."""

        comparison: dict[str, Any] = {
            "current": {
                "score": current_report.overall_score,
                "grade": current_report.grade,
                "total_issues": len(current_report.issues),
            }
        }

        if previous_report:
            comparison["previous"] = {
                "score": previous_report.overall_score,
                "grade": previous_report.grade,
                "total_issues": len(previous_report.issues),
            }

            comparison["changes"] = {
                "score_change": current_report.overall_score - previous_report.overall_score,
                "grade_change": f"{previous_report.grade} → {current_report.grade}",
                "issues_change": len(current_report.issues) - len(previous_report.issues),
            }

            # Determine trend
            score_change = comparison["changes"]["score_change"]
            if score_change > 0:
                comparison["trend"] = "improving"
            elif score_change < 0:
                comparison["trend"] = "declining"
            else:
                comparison["trend"] = "stable"

        return comparison
