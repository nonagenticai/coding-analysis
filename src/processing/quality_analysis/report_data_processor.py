"""
Report Data Processor - Handle data processing and metrics calculation for quality reports.
Provides data aggregation, metric calculation, and data transformation utilities.
"""

from __future__ import annotations

import logging
from typing import Any

from models.testing_models import QualityIssue, QualityResult

from .report_configuration import QualityMetrics, ReportSettings

logger = logging.getLogger(__name__)


class ReportDataProcessor:
    """
    Processes and formats data for quality reports.
    Handles metrics calculation, data aggregation, and transformations.
    """

    def __init__(self):
        """Initialize the data processor."""
        logger.info("ReportDataProcessor initialized")

    async def calculate_metrics(self, results: list[QualityResult]) -> QualityMetrics:
        """
        Calculate aggregated metrics from quality results.

        Args:
            results: List of QualityResult objects

        Returns:
            QualityMetrics object with calculated values
        """
        if not results:
            return QualityMetrics(
                total_files_analyzed=0,
                total_issues=0,
                overall_score=0.0,
                critical_issues=0,
                error_issues=0,
                warning_issues=0,
                info_issues=0,
                successful_tools=0,
                failed_tools=0,
                total_execution_time=0.0,
                average_execution_time=0.0,
            )

        # Aggregate basic metrics
        total_issues = sum(len(r.issues) for r in results)
        total_execution_time = sum(r.execution_time for r in results)

        # Count issues by severity
        severity_counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        for result in results:
            for issue in result.issues:
                severity = issue.severity.lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1

        # Count tools and success rates
        all_tools = set()
        successful_tools = 0
        failed_tools = 0

        for result in results:
            for tool_name, tool_result in result.tools_results.items():
                all_tools.add(tool_name)
                if tool_result.get("success", True):
                    successful_tools += 1
                else:
                    failed_tools += 1

        # Language distribution
        language_counts: dict[str, int] = {}
        for result in results:
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Calculate overall score (weighted average)
        if results:
            overall_score = sum(r.overall_score for r in results) / len(results)
        else:
            overall_score = 0.0

        return QualityMetrics(
            total_files_analyzed=len(results),
            total_issues=total_issues,
            overall_score=round(overall_score, 2),
            critical_issues=severity_counts["critical"],
            error_issues=severity_counts["error"],
            warning_issues=severity_counts["warning"],
            info_issues=severity_counts["info"],
            tools_used=list(all_tools),
            successful_tools=successful_tools,
            failed_tools=failed_tools,
            total_execution_time=round(total_execution_time, 2),
            average_execution_time=(
                round(total_execution_time / len(results), 2) if results else 0.0
            ),
            languages_analyzed=language_counts,
        )

    async def generate_issue_summary(
        self, results: list[QualityResult], settings: ReportSettings
    ) -> dict[str, Any]:
        """
        Generate summary of issues across all results.

        Args:
            results: List of QualityResult objects
            settings: Report generation settings

        Returns:
            Dictionary with issue summaries grouped by various criteria
        """
        tool_issues: dict[str, list[QualityIssue]] = {}
        file_issues: dict[str, list[QualityIssue]] = {}
        rule_issues: dict[str, list[QualityIssue]] = {}

        for result in results:
            for issue in result.issues:
                if issue.severity.lower() not in settings.severity_filter:
                    continue

                # Group by tool
                if issue.tool not in tool_issues:
                    tool_issues[issue.tool] = []
                tool_issues[issue.tool].append(issue)

                # Group by file
                if issue.file_path not in file_issues:
                    file_issues[issue.file_path] = []
                file_issues[issue.file_path].append(issue)

                # Group by rule
                if issue.rule_id:
                    if issue.rule_id not in rule_issues:
                        rule_issues[issue.rule_id] = []
                    rule_issues[issue.rule_id].append(issue)

        return {
            "issues_by_tool": {tool: len(issues) for tool, issues in tool_issues.items()},
            "issues_by_file": {file: len(issues) for file, issues in file_issues.items()},
            "issues_by_rule": {rule: len(issues) for rule, issues in rule_issues.items()},
            "most_problematic_files": sorted(
                file_issues.items(), key=lambda x: len(x[1]), reverse=True
            )[:10],
            "most_common_rules": sorted(rule_issues.items(), key=lambda x: len(x[1]), reverse=True)[
                :10
            ],
        }

    async def generate_tool_performance(self, results: list[QualityResult]) -> dict[str, Any]:
        """
        Generate tool performance statistics.

        Args:
            results: List of QualityResult objects

        Returns:
            Dictionary with tool performance metrics
        """
        tool_stats = {}

        for result in results:
            for tool_name, tool_result in result.tools_results.items():
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "total_runs": 0,
                        "successful_runs": 0,
                        "total_execution_time": 0.0,
                        "issues_found": 0,
                    }

                stats = tool_stats[tool_name]
                stats["total_runs"] += 1

                if tool_result.get("success", True):
                    stats["successful_runs"] += 1

                stats["total_execution_time"] += tool_result.get("execution_time", 0.0)
                stats["issues_found"] += len(tool_result.get("issues", []))

        # Calculate derived metrics
        for tool_name, stats in tool_stats.items():
            if stats["total_runs"] > 0:
                stats["success_rate"] = stats["successful_runs"] / stats["total_runs"] * 100
                stats["average_execution_time"] = (
                    stats["total_execution_time"] / stats["total_runs"]
                )
                stats["average_issues_per_run"] = stats["issues_found"] / stats["total_runs"]
            else:
                stats["success_rate"] = 0
                stats["average_execution_time"] = 0
                stats["average_issues_per_run"] = 0

        return tool_stats

    async def generate_recommendations(
        self, metrics: QualityMetrics, results: list[QualityResult]
    ) -> list[str]:
        """
        Generate actionable recommendations based on analysis results.

        Args:
            metrics: Calculated quality metrics
            results: List of QualityResult objects

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Score-based recommendations
        if metrics.overall_score < 60:
            recommendations.append(
                "⚠️ Overall quality score is below acceptable threshold. "
                "Focus on addressing critical and error-level issues first."
            )

        # Issue-based recommendations
        if metrics.critical_issues > 0:
            recommendations.append(
                f"🚨 {metrics.critical_issues} critical issues found. These should be addressed immediately."
            )

        if metrics.error_issues > metrics.total_issues * 0.3:  # More than 30% are errors
            recommendations.append(
                "❌ High proportion of error-level issues. Review code patterns and consider additional linting rules."
            )

        # Tool-based recommendations
        if metrics.failed_tools > 0:
            recommendations.append(
                f"🔧 {metrics.failed_tools} tool executions failed. Check tool configurations and dependencies."
            )

        # Performance recommendations
        if metrics.average_execution_time > 60:
            recommendations.append(
                "⏱️ Analysis execution time is high. Consider optimizing tool "
                "configurations or analyzing smaller code chunks."
            )

        # Language-specific recommendations
        for language in metrics.languages_analyzed:
            if language == "python" and any("bandit" in r.tools_results for r in results):
                if any(issue.tool == "bandit" for r in results for issue in r.issues):
                    recommendations.append(
                        "🔒 Security issues detected in Python code. Review security best practices."
                    )

        if not recommendations:
            recommendations.append(
                "✅ Code quality looks good! Consider implementing regular quality checks to maintain standards."
            )

        return recommendations

    async def filter_and_format_issues(
        self, result: QualityResult, settings: ReportSettings
    ) -> list[dict[str, Any]]:
        """
        Filter and format issues based on settings.

        Args:
            result: QualityResult object
            settings: Report generation settings

        Returns:
            List of formatted issue dictionaries
        """
        filtered_issues = []

        for issue in result.issues:
            if issue.severity.lower() not in settings.severity_filter:
                continue

            formatted_issue = {
                "tool": issue.tool,
                "severity": issue.severity,
                "message": issue.message,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "column_number": issue.column_number,
                "rule_id": issue.rule_id,
                "suggested_fix": (
                    issue.suggested_fix if settings.include_fix_suggestions else None
                ),
            }
            filtered_issues.append(formatted_issue)

            # Respect max issues limit
            if (
                settings.max_issues_per_file
                and len(filtered_issues) >= settings.max_issues_per_file
            ):
                break

        return filtered_issues

    async def calculate_comparative_metrics(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> dict[str, Any]:
        """
        Calculate comparative metrics between current and baseline results.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline results for comparison

        Returns:
            Dictionary with comparative analysis data
        """
        current_metrics = await self.calculate_metrics(current_results)
        baseline_metrics = await self.calculate_metrics(baseline_results)

        # Calculate changes
        score_change = current_metrics.overall_score - baseline_metrics.overall_score
        issue_change = current_metrics.total_issues - baseline_metrics.total_issues

        return {
            "current_metrics": current_metrics.model_dump(),
            "baseline_metrics": baseline_metrics.model_dump(),
            "score_change": round(score_change, 2),
            "score_change_percentage": (
                round(
                    (score_change / baseline_metrics.overall_score * 100),
                    2,
                )
                if baseline_metrics.overall_score > 0
                else 0
            ),
            "issue_count_change": issue_change,
            "issue_change_percentage": (
                round(
                    (issue_change / baseline_metrics.total_issues * 100),
                    2,
                )
                if baseline_metrics.total_issues > 0
                else 0
            ),
        }

    async def identify_improvements(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> list[str]:
        """
        Identify areas of improvement compared to baseline.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline results for comparison

        Returns:
            List of improvement descriptions
        """
        improvements = []

        current_metrics = await self.calculate_metrics(current_results)
        baseline_metrics = await self.calculate_metrics(baseline_results)

        if current_metrics.overall_score > baseline_metrics.overall_score:
            score_increase = current_metrics.overall_score - baseline_metrics.overall_score
            improvements.append(f"Overall quality score increased by {score_increase:.1f} points")

        if current_metrics.critical_issues < baseline_metrics.critical_issues:
            critical_reduction = baseline_metrics.critical_issues - current_metrics.critical_issues
            improvements.append(f"Critical issues reduced by {critical_reduction}")

        if current_metrics.error_issues < baseline_metrics.error_issues:
            error_reduction = baseline_metrics.error_issues - current_metrics.error_issues
            improvements.append(f"Error issues reduced by {error_reduction}")

        return improvements

    async def identify_regressions(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> list[str]:
        """
        Identify areas of regression compared to baseline.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline results for comparison

        Returns:
            List of regression descriptions
        """
        regressions = []

        current_metrics = await self.calculate_metrics(current_results)
        baseline_metrics = await self.calculate_metrics(baseline_results)

        if current_metrics.overall_score < baseline_metrics.overall_score:
            score_decrease = baseline_metrics.overall_score - current_metrics.overall_score
            regressions.append(f"Overall quality score decreased by {score_decrease:.1f} points")

        if current_metrics.critical_issues > baseline_metrics.critical_issues:
            critical_increase = current_metrics.critical_issues - baseline_metrics.critical_issues
            regressions.append(f"Critical issues increased by {critical_increase}")

        if current_metrics.error_issues > baseline_metrics.error_issues:
            error_increase = current_metrics.error_issues - baseline_metrics.error_issues
            regressions.append(f"Error issues increased by {error_increase}")

        return regressions

    def safe_float_convert(self, value: Any) -> float:
        """
        Safely convert value to float.

        Args:
            value: Value to convert

        Returns:
            Float value or 0.0 if conversion fails
        """
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):  # fmt: skip
            return 0.0


def create_data_processor() -> ReportDataProcessor:
    """Factory function to create ReportDataProcessor."""
    return ReportDataProcessor()
