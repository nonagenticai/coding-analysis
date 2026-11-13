"""

Quality Report Generator - Generate comprehensive quality reports and metrics.
Provides HTML, JSON, and dashboard formats for quality analysis results.

This module serves as the main interface and orchestrator for the modular report generation system.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.testing_models import QualityResult
from processing.quality_analysis.report_chart_generator import (
    create_chart_generator,
)
from processing.quality_analysis.report_configuration import (
    ReportContext,
    ReportSettings,
    create_configuration_manager,
    create_default_settings,
)
from processing.quality_analysis.report_data_processor import (
    create_data_processor,
)
from processing.quality_analysis.report_output_handler import (
    create_output_handler,
)
from processing.quality_analysis.report_template_manager import (
    create_template_manager,
)

logger = logging.getLogger(__name__)


class QualityReportGenerator:
    """
    Generate comprehensive quality reports in multiple formats.
    Orchestrates modular components for report generation, data processing, and output handling.
    """

    def __init__(
        self,
        cache_manager: Any | None = None,
        output_directory: str | None = None,
    ):
        self.cache_manager = cache_manager

        # Initialize modular components
        self.config_manager = create_configuration_manager()
        self.data_processor = create_data_processor()
        self.template_manager = create_template_manager()
        self.chart_generator = create_chart_generator(self.data_processor)
        self.output_handler = create_output_handler(
            output_directory=output_directory,
            template_manager=self.template_manager,
            data_processor=self.data_processor,
        )

        logger.info(
            "QualityReportGenerator initialized with modular components, output: %s",
            self.output_handler.output_directory,
        )

    async def generate_html_report(
        self,
        results: QualityResult | list[QualityResult],
        settings: ReportSettings | None = None,
        theme_name: str = "default",
        style_name: str = "default",
    ) -> str:
        """
        Generate comprehensive HTML report from quality analysis results.

        Args:
            results: Single or multiple QualityResult objects
            settings: Report generation settings
            theme_name: Theme name to use for styling
            style_name: Style configuration to use

        Returns:
            HTML report content as string
        """
        try:
            logger.info("Generating HTML quality report")

            # Normalize input to list
            result_list = results if isinstance(results, list) else [results]
            settings = settings or create_default_settings()

            # Calculate metrics
            metrics = await self.data_processor.calculate_metrics(result_list)
            context = self.config_manager.create_context(
                metrics=metrics,
                settings=settings,
                theme_name=theme_name,
                style_name=style_name,
            )

            # Generate HTML content
            html_content = await self.template_manager.render_html_template(result_list, context)

            # Save to file using output handler
            report_path = await self.output_handler.export_html_report(result_list, settings)
            logger.info("HTML report saved to: %s", report_path)

            return html_content

        except Exception as e:
            logger.error("Failed to generate HTML report: %s", e)
            raise

    async def generate_json_summary(
        self,
        results: QualityResult | list[QualityResult],
        settings: ReportSettings | None = None,
    ) -> dict[str, Any]:
        """
        Generate JSON summary of quality analysis results.

        Args:
            results: Single or multiple QualityResult objects
            settings: Report generation settings

        Returns:
            Dictionary containing structured summary data
        """
        try:
            logger.info("Generating JSON quality summary")

            # Normalize input to list
            result_list = results if isinstance(results, list) else [results]
            settings = settings or create_default_settings()

            # Export JSON summary using output handler
            summary_path = await self.output_handler.export_json_summary(result_list, settings)
            logger.info("JSON summary saved to: %s", summary_path)

            # Calculate metrics and build summary for return value
            metrics = await self.data_processor.calculate_metrics(result_list)
            context = self.config_manager.create_context(metrics=metrics, settings=settings)

            summary = {
                "report_metadata": {
                    "report_id": context.report_id,
                    "generation_time": context.generation_time.isoformat(),
                    "title": context.title,
                    "description": context.description,
                    "generator_version": "2.0.0",
                },
                "quality_metrics": metrics.model_dump(),
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

            return summary

        except Exception as e:
            logger.error("Failed to generate JSON summary: %s", e)
            raise

    async def generate_metrics_dashboard(
        self,
        results: list[QualityResult],
        historical_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate dashboard-ready metrics and visualizations.

        Args:
            results: List of QualityResult objects
            historical_data: Optional historical analysis data for trends

        Returns:
            Dictionary containing dashboard data and chart configurations
        """
        try:
            logger.info("Generating metrics dashboard")

            if not results:
                raise ValueError("No quality results provided for dashboard generation")

            # Calculate comprehensive metrics
            metrics = await self.data_processor.calculate_metrics(results)

            # Generate charts using chart generator
            charts = await self.chart_generator.generate_dashboard_charts(results, historical_data)

            # Generate table data
            tables = await self._generate_dashboard_tables(results)

            # Generate dashboard data
            dashboard = {
                "dashboard_metadata": {
                    "generated_at": metrics.total_execution_time,  # Use existing timestamp
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
                "charts": charts,
                "tables": tables,
            }

            # Save dashboard data using output handler
            dashboard_path = await self.output_handler.export_dashboard_data(
                results, historical_data
            )
            logger.info("Dashboard data saved to: %s", dashboard_path)

            return dashboard

        except Exception as e:
            logger.error("Failed to generate metrics dashboard: %s", e)
            raise

    async def generate_comparative_report(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
        settings: ReportSettings | None = None,
    ) -> dict[str, Any]:
        """
        Generate comparative report showing quality improvements/degradations.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline results for comparison
            settings: Report generation settings

        Returns:
            Comparative analysis report
        """
        try:
            logger.info("Generating comparative quality report")

            settings = settings or create_default_settings()

            # Export comparative report using output handler
            report_path = await self.output_handler.export_comparative_report(
                current_results, baseline_results, settings
            )
            logger.info("Comparative report saved to: %s", report_path)

            # Calculate comparative metrics for return value
            comparative_data = await self.data_processor.calculate_comparative_metrics(
                current_results, baseline_results
            )

            comparative_report = {
                "comparison_metadata": {
                    "generated_at": comparative_data.get("current_metrics", {}).get(
                        "total_execution_time", 0
                    ),
                    "current_period": {
                        "analyses_count": len(current_results),
                    },
                    "baseline_period": {
                        "analyses_count": len(baseline_results),
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
            }

            return comparative_report

        except Exception as e:
            logger.error("Failed to generate comparative report: %s", e)
            raise

    # Helper methods for dashboard table generation

    async def _generate_dashboard_tables(self, results: list[QualityResult]) -> dict[str, Any]:
        """
        Generate dashboard table data.

        Args:
            results: List of QualityResult objects

        Returns:
            Dictionary containing table data
        """
        return {
            "top_issues": await self._generate_top_issues_table(results),
            "file_quality_ranking": await self._generate_file_ranking_table(results),
            "tool_statistics": await self._generate_tool_statistics_table(results),
        }

    # Legacy helper methods (deprecated - use modular components instead)
    # These methods are kept for backward compatibility

    async def _calculate_metrics(self, results: list[QualityResult]):
        """Calculate aggregated metrics from quality results (deprecated - use data_processor)."""
        return await self.data_processor.calculate_metrics(results)

    async def _render_html_template(
        self, results: list[QualityResult], context: ReportContext
    ) -> str:
        """Render HTML template with quality data (deprecated - use template_manager)."""
        return await self.template_manager.render_html_template(results, context)

    async def _generate_issue_summary(
        self, results: list[QualityResult], settings: ReportSettings
    ) -> dict[str, Any]:
        """Generate summary of issues across all results (deprecated - use data_processor)."""
        return await self.data_processor.generate_issue_summary(results, settings)

    async def _generate_tool_performance(self, results: list[QualityResult]) -> dict[str, Any]:
        """Generate tool performance statistics (deprecated - use data_processor)."""
        return await self.data_processor.generate_tool_performance(results)

    async def _generate_recommendations(self, metrics, results: list[QualityResult]) -> list[str]:
        """Generate actionable recommendations (deprecated - use data_processor)."""
        return await self.data_processor.generate_recommendations(metrics, results)

    # Legacy chart generation methods (deprecated - use chart_generator)

    async def _generate_severity_chart(self, results: list[QualityResult]) -> dict[str, Any]:
        """Generate severity distribution chart data (deprecated)."""
        return await self.chart_generator.generate_severity_chart(results)

    async def _generate_score_distribution_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """Generate quality score distribution chart (deprecated)."""
        return await self.chart_generator.generate_score_distribution_chart(results)

    async def _generate_tools_performance_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """Generate tools performance chart (deprecated)."""
        return await self.chart_generator.generate_tools_performance_chart(results)

    async def _generate_language_breakdown_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """Generate language breakdown chart (deprecated)."""
        return await self.chart_generator.generate_language_breakdown_chart(results)

    async def _generate_trends_chart(
        self,
        results: list[QualityResult],
        historical_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate trends chart data (deprecated)."""
        return await self.chart_generator.generate_trends_chart(results, historical_data)

    # Legacy table generation methods (kept for backward compatibility)

    async def _generate_top_issues_table(
        self, results: list[QualityResult]
    ) -> list[dict[str, Any]]:
        """Generate top issues table data."""
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)

        # Sort by severity priority
        severity_priority = {
            "critical": 4,
            "error": 3,
            "warning": 2,
            "info": 1,
        }
        sorted_issues = sorted(
            all_issues,
            key=lambda x: severity_priority.get(x.severity.lower(), 0),
            reverse=True,
        )

        return [
            {
                "severity": issue.severity,
                "tool": issue.tool,
                "message": (
                    issue.message[:100] + "..." if len(issue.message) > 100 else issue.message
                ),
                "file": issue.file_path,
                "line": issue.line_number,
                "rule": issue.rule_id,
            }
            for issue in sorted_issues[:20]  # Top 20 issues
        ]

    async def _generate_file_ranking_table(
        self, results: list[QualityResult]
    ) -> list[dict[str, Any]]:
        """Generate file quality ranking table."""
        file_stats = {}

        for result in results:
            # Streamlined file identification
            file_path = f"{result.language}:{result.analysis_id}"
            if file_path not in file_stats:
                file_stats[file_path] = {
                    "file": file_path,
                    "language": result.language,
                    "score": result.overall_score,
                    "issues": len(result.issues),
                    "critical_issues": len(
                        [i for i in result.issues if i.severity.lower() == "critical"]
                    ),
                }

        # Sort by score (descending) and then by issues (ascending)
        sorted_files = sorted(
            file_stats.values(),
            key=lambda x: (
                -self.data_processor.safe_float_convert(x["score"]),
                x["issues"],
            ),
        )

        return sorted_files[:20]  # Top 20 files

    async def _generate_tool_statistics_table(
        self, results: list[QualityResult]
    ) -> list[dict[str, Any]]:
        """Generate tool statistics table."""
        tool_performance = await self.data_processor.generate_tool_performance(results)

        return [
            {
                "tool": tool,
                "runs": stats["total_runs"],
                "success_rate": f"{stats['success_rate']:.1f}%",
                "avg_time": f"{stats['average_execution_time']:.2f}s",
                "issues_found": stats["issues_found"],
                "avg_issues": f"{stats['average_issues_per_run']:.1f}",
            }
            for tool, stats in tool_performance.items()
        ]

    # Legacy comparative analysis methods (deprecated - use data_processor)

    async def _identify_improvements(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> list[str]:
        """Identify areas of improvement (deprecated - use data_processor)."""
        return await self.data_processor.identify_improvements(current_results, baseline_results)

    async def _identify_regressions(
        self,
        current_results: list[QualityResult],
        baseline_results: list[QualityResult],
    ) -> list[str]:
        """Identify areas of regression (deprecated - use data_processor)."""
        return await self.data_processor.identify_regressions(current_results, baseline_results)

    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return self.config_manager.get_available_themes()

    def get_available_styles(self) -> list[str]:
        """Get list of available style names."""
        return self.config_manager.get_available_styles()

    def get_chart_types(self) -> list[str]:
        """Get list of available chart types."""
        return self.chart_generator.get_available_chart_types()

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export formats."""
        return self.output_handler.get_supported_formats()


# Factory function for dependency injection
def create_quality_report_generator(
    cache_manager: Any | None = None, output_directory: str | None = None
) -> QualityReportGenerator:
    """Factory function to create QualityReportGenerator with modular components."""
    return QualityReportGenerator(
        cache_manager=cache_manager,
        output_directory=output_directory
        or os.getenv("QUALITY_REPORTS_DIR", "/var/tmp/quality_reports"),  # nosec B108
    )
