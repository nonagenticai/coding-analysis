"""
Report Output Handler - Handle various output formats for quality reports.
Provides HTML, PDF, JSON, and other format export capabilities.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.testing_models import QualityResult

from .report_configuration import QualityMetrics, ReportContext, ReportSettings
from .report_data_processor import ReportDataProcessor
from .report_template_manager import ReportTemplateManager

logger = logging.getLogger(__name__)


class ReportOutputHandler:
    """
    Handles various output formats for quality reports.
    Supports HTML, JSON, CSV, and other export formats.
    """

    def __init__(
        self,
        output_directory: str | None = None,
        template_manager: ReportTemplateManager | None = None,
        data_processor: ReportDataProcessor | None = None,
    ):
        """
        Initialize the output handler.

        Args:
            output_directory: Directory for saving reports
            template_manager: Optional template manager instance
            data_processor: Optional data processor instance
        """
        self.output_directory = Path(output_directory or "/var/tmp/quality_reports")  # nosec B108
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.template_manager = template_manager or ReportTemplateManager()
        self.data_processor = data_processor or ReportDataProcessor()

        logger.info("ReportOutputHandler initialized, output: %s", self.output_directory)

    async def export_html_report(
        self,
        results: QualityResult | list[QualityResult],
        settings: ReportSettings | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Export HTML report to file.

        Args:
            results: Single or multiple QualityResult objects
            settings: Report generation settings
            filename: Optional custom filename

        Returns:
            Path to the exported HTML file
        """
        try:
            logger.info("Exporting HTML quality report")

            # Normalize input to list
            result_list = results if isinstance(results, list) else [results]
            settings = settings or ReportSettings()

            # Calculate metrics
            metrics = await self.data_processor.calculate_metrics(result_list)
            context = ReportContext(metrics=metrics, settings=settings)

            # Generate HTML content
            html_content = await self.template_manager.render_html_template(result_list, context)

            # Determine filename
            if not filename:
                filename = f"{context.report_id}.html"
            elif not filename.endswith(".html"):
                filename += ".html"

            # Save to file
            report_path = self.output_directory / filename
            report_path.write_text(html_content, encoding="utf-8")

            logger.info("HTML report exported to: %s", report_path)
            return str(report_path)

        except Exception as e:
            logger.error("Failed to export HTML report: %s", e)
            raise

    async def export_json_summary(
        self,
        results: QualityResult | list[QualityResult],
        settings: ReportSettings | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Export JSON summary to file.

        Args:
            results: Single or multiple QualityResult objects
            settings: Report generation settings
            filename: Optional custom filename

        Returns:
            Path to the exported JSON file
        """
        try:
            logger.info("Exporting JSON quality summary")

            # Normalize input to list
            result_list = results if isinstance(results, list) else [results]
            settings = settings or ReportSettings()

            # Calculate metrics
            metrics = await self.data_processor.calculate_metrics(result_list)
            context = ReportContext(metrics=metrics, settings=settings)

            # Build comprehensive summary
            summary = {
                "report_metadata": {
                    "report_id": context.report_id,
                    "generation_time": context.generation_time.isoformat(),
                    "title": context.title,
                    "description": context.description,
                    "generator_version": "1.0.0",
                },
                "quality_metrics": metrics.model_dump(),
                "analysis_results": [],
                "issue_summary": await self.data_processor.generate_issue_summary(
                    result_list, settings
                ),
                "tool_performance": await self.data_processor.generate_tool_performance(
                    result_list
                ),
                "recommendations": await self.data_processor.generate_recommendations(
                    metrics, result_list
                ),
            }

            # Add detailed results for each analysis
            for result in result_list:
                analysis_summary: dict[str, Any] = {
                    "analysis_id": result.analysis_id,
                    "language": result.language,
                    "success": result.success,
                    "overall_score": result.overall_score,
                    "total_issues": result.total_issues,
                    "execution_time": result.execution_time,
                    "created_at": result.created_at.isoformat(),
                    "tools_executed": list(result.tools_results.keys()),
                    "error": result.error,
                }

                # Add filtered issues based on settings
                if settings.severity_filter:
                    filtered_issues = await self.data_processor.filter_and_format_issues(
                        result, settings
                    )
                    analysis_summary["issues"] = filtered_issues

                summary["analysis_results"].append(analysis_summary)

            # Determine filename
            if not filename:
                filename = f"{context.report_id}_summary.json"
            elif not filename.endswith(".json"):
                filename += ".json"

            # Save to file
            summary_path = self.output_directory / filename
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str),
                encoding="utf-8",
            )

            logger.info("JSON summary exported to: %s", summary_path)
            return str(summary_path)

        except Exception as e:
            logger.error("Failed to export JSON summary: %s", e)
            raise

    async def export_csv_report(
        self,
        results: QualityResult | list[QualityResult],
        settings: ReportSettings | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Export CSV report with issue details.

        Args:
            results: Single or multiple QualityResult objects
            settings: Report generation settings
            filename: Optional custom filename

        Returns:
            Path to the exported CSV file
        """
        try:
            logger.info("Exporting CSV quality report")

            # Normalize input to list
            result_list = results if isinstance(results, list) else [results]
            settings = settings or ReportSettings()

            # Prepare CSV content
            csv_lines = [
                "Analysis ID,Language,Tool,Severity,File Path,Line,Column,Rule ID,Message,Suggested Fix"
            ]

            for result in result_list:
                filtered_issues = await self.data_processor.filter_and_format_issues(
                    result, settings
                )

                for issue in filtered_issues:
                    # Escape CSV values
                    analysis_id = self._escape_csv_value(result.analysis_id)
                    language = self._escape_csv_value(result.language)
                    tool = self._escape_csv_value(issue.get("tool", ""))
                    severity = self._escape_csv_value(issue.get("severity", ""))
                    file_path = self._escape_csv_value(issue.get("file_path", ""))
                    line = str(issue.get("line_number", ""))
                    column = str(issue.get("column_number", ""))
                    rule_id = self._escape_csv_value(issue.get("rule_id", ""))
                    message = self._escape_csv_value(issue.get("message", ""))
                    suggested_fix = self._escape_csv_value(issue.get("suggested_fix", ""))

                    csv_line = f"{analysis_id},{language},{tool},{severity},{file_path},{line},{column},{rule_id},{message},{suggested_fix}"
                    csv_lines.append(csv_line)

            # Determine filename
            if not filename:
                timestamp = int(datetime.now(UTC).timestamp())
                filename = f"quality_report_{timestamp}.csv"
            elif not filename.endswith(".csv"):
                filename += ".csv"

            # Save to file
            csv_path = self.output_directory / filename
            csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

            logger.info("CSV report exported to: %s", csv_path)
            return str(csv_path)

        except Exception as e:
            logger.error("Failed to export CSV report: %s", e)
            raise

    async def export_dashboard_data(
        self,
        results: list[QualityResult],
        historical_data: list[dict[str, Any]] | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Export dashboard data to JSON file.

        Args:
            results: List of QualityResult objects
            historical_data: Optional historical analysis data for trends
            filename: Optional custom filename

        Returns:
            Path to the exported dashboard JSON file
        """
        try:
            logger.info("Exporting dashboard data")

            if not results:
                raise ValueError("No quality results provided for dashboard generation")

            # Calculate comprehensive metrics
            metrics = await self.data_processor.calculate_metrics(results)

            # Generate dashboard data
            dashboard = {
                "dashboard_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "total_analyses": len(results),
                    "date_range": {
                        "start": min(r.created_at for r in results).isoformat(),
                        "end": max(r.created_at for r in results).isoformat(),
                    },
                },
                "key_metrics": {
                    "overall_quality_score": metrics.overall_score,
                    "total_issues": metrics.total_issues,
                    "critical_issues": metrics.critical_issues,
                    "files_analyzed": metrics.total_files_analyzed,
                    "tools_success_rate": (
                        (
                            metrics.successful_tools
                            / (metrics.successful_tools + metrics.failed_tools)
                            * 100
                        )
                        if (metrics.successful_tools + metrics.failed_tools) > 0
                        else 100.0
                    ),
                },
                "quality_alerts": await self._generate_quality_alerts(metrics, results),
            }

            # Determine filename
            if not filename:
                timestamp = int(datetime.now(UTC).timestamp())
                filename = f"dashboard_{timestamp}.json"
            elif not filename.endswith(".json"):
                filename += ".json"

            # Save dashboard data
            dashboard_path = self.output_directory / filename
            dashboard_path.write_text(
                json.dumps(dashboard, indent=2, default=str),
                encoding="utf-8",
            )

            logger.info("Dashboard data exported to: %s", dashboard_path)
            return str(dashboard_path)

        except Exception as e:
            logger.error("Failed to export dashboard data: %s", e)
            raise

    async def export_comparative_report(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
        settings: ReportSettings | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Export comparative report to JSON file.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline results for comparison
            settings: Report generation settings
            filename: Optional custom filename

        Returns:
            Path to the exported comparative report file
        """
        try:
            logger.info("Exporting comparative quality report")

            settings = settings or ReportSettings()

            # Calculate comparative metrics
            comparative_data = await self.data_processor.calculate_comparative_metrics(
                current_results, baseline_results
            )

            comparative_report = {
                "comparison_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "current_period": {
                        "analyses_count": len(current_results),
                        "date_range": {
                            "start": min(r.created_at for r in current_results).isoformat(),
                            "end": max(r.created_at for r in current_results).isoformat(),
                        },
                    },
                    "baseline_period": {
                        "analyses_count": len(baseline_results),
                        "date_range": {
                            "start": min(r.created_at for r in baseline_results).isoformat(),
                            "end": max(r.created_at for r in baseline_results).isoformat(),
                        },
                    },
                },
                "quality_changes": {
                    **comparative_data,
                    "improvement_areas": await self.data_processor.identify_improvements(
                        current_results, baseline_results
                    ),
                    "regression_areas": await self.data_processor.identify_regressions(
                        current_results, baseline_results
                    ),
                },
                "recommendations": await self._generate_comparative_recommendations(
                    comparative_data, current_results, baseline_results
                ),
            }

            # Determine filename
            if not filename:
                timestamp = int(datetime.now(UTC).timestamp())
                filename = f"comparative_report_{timestamp}.json"
            elif not filename.endswith(".json"):
                filename += ".json"

            # Save to file
            report_path = self.output_directory / filename
            report_path.write_text(
                json.dumps(comparative_report, indent=2, default=str),
                encoding="utf-8",
            )

            logger.info("Comparative report exported to: %s", report_path)
            return str(report_path)

        except Exception as e:
            logger.error("Failed to export comparative report: %s", e)
            raise

    def _escape_csv_value(self, value: Any) -> str:
        """
        Escape CSV value to handle commas and quotes.

        Args:
            value: Value to escape

        Returns:
            Escaped CSV value
        """
        if value is None:
            return ""

        str_value = str(value)
        if "," in str_value or '"' in str_value or "\n" in str_value:
            # Escape quotes by doubling them and wrap in quotes
            escaped = str_value.replace('"', '""')
            return f'"{escaped}"'
        return str_value

    async def _generate_quality_alerts(
        self, metrics: QualityMetrics, results: list[QualityResult]
    ) -> list[dict[str, Any]]:
        """Generate quality alerts based on thresholds."""
        # Note: results parameter used for future extensibility
        _ = results  # Mark as intentionally unused
        alerts = []

        # Critical issues alert
        if metrics.critical_issues > 0:
            alerts.append(
                {
                    "type": "critical",
                    "title": "Critical Issues Detected",
                    "message": f"{metrics.critical_issues} critical issues require immediate attention",
                    "action": "Review and resolve critical issues first",
                }
            )

        # Low quality score alert
        if metrics.overall_score < 70:
            alerts.append(
                {
                    "type": "warning",
                    "title": "Low Quality Score",
                    "message": f"Overall quality score of {metrics.overall_score:.1f} is below recommended threshold",
                    "action": "Focus on addressing error and warning level issues",
                }
            )

        # Tool failure alert
        if metrics.failed_tools > 0:
            alerts.append(
                {
                    "type": "info",
                    "title": "Tool Execution Failures",
                    "message": f"{metrics.failed_tools} tool executions failed",
                    "action": "Check tool configurations and dependencies",
                }
            )

        return alerts

    async def _generate_comparative_recommendations(
        self,
        comparative_data: dict[str, Any],
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> list[str]:
        """Generate recommendations based on comparative analysis."""
        # Note: current_results and baseline_results used for future extensibility
        _ = current_results  # Mark as intentionally unused
        _ = baseline_results  # Mark as intentionally unused
        recommendations = []

        score_change = comparative_data.get("score_change", 0)

        if score_change < -5:  # Significant regression
            recommendations.append(
                "⚠️ Significant quality regression detected. Review recent changes "
                "and consider reverting problematic commits."
            )
        elif score_change > 5:  # Significant improvement
            recommendations.append(
                "✅ Significant quality improvement achieved. Document successful practices for future reference."
            )

        issue_change_percentage = comparative_data.get("issue_change_percentage", 0)

        if issue_change_percentage > 20:  # 20% increase in issues
            recommendations.append(
                "📈 Issue count has increased significantly. Focus on code review processes and preventive measures."
            )

        if not recommendations:
            recommendations.append(
                "📊 Quality metrics are stable. Continue current practices and monitor trends."
            )

        return recommendations

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export formats."""
        return ["html", "json", "csv", "dashboard", "comparative"]

    def set_output_directory(self, directory: str) -> None:
        """
        Set the output directory for reports.

        Args:
            directory: Path to output directory
        """
        self.output_directory = Path(directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory set to: %s", self.output_directory)


def create_output_handler(
    output_directory: str | None = None,
    template_manager: ReportTemplateManager | None = None,
    data_processor: ReportDataProcessor | None = None,
) -> ReportOutputHandler:
    """Factory function to create ReportOutputHandler."""
    return ReportOutputHandler(
        output_directory=output_directory
        or os.getenv("QUALITY_REPORTS_DIR", "/var/tmp/quality_reports"),  # nosec B108
        template_manager=template_manager,
        data_processor=data_processor,
    )
