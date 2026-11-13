"""
Test Reporting Module for Integration Tests.

This module provides comprehensive reporting functionality for integration
test results, including formatted reports, visualizations, and export
capabilities.

The module supports:
- Multiple report formats (markdown, HTML, JSON, plain text)
- Test result visualization and charts
- Executive summaries and detailed reports
- Historical trend reporting
- Export to various formats
- Email and notification support

Classes:
    TestReportGenerator: Main report generation coordinator
    MarkdownReportGenerator: Generates markdown format reports
    HTMLReportGenerator: Generates HTML format reports
    JSONReportGenerator: Generates JSON format reports
    VisualizationGenerator: Creates charts and visualizations

Functions:
    generate_test_report: Generate comprehensive test report
    generate_executive_summary: Generate executive summary report
    export_test_results: Export test results to various formats
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .test_analysis import QualityMetrics, TestAnalysisResult
from .test_coverage import CoverageReport
from .test_execution import TestExecutionResult, TestExecutionStatus

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    CSV = "csv"


class ReportLevel(Enum):
    """Report detail levels."""

    EXECUTIVE = "executive"  # High-level summary
    SUMMARY = "summary"  # Mid-level overview
    DETAILED = "detailed"  # Full details
    TECHNICAL = "technical"  # Technical deep-dive


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    format: ReportFormat
    level: ReportLevel
    include_charts: bool = False
    include_recommendations: bool = True
    include_historical: bool = False
    max_details: int = 100
    group_by_component: bool = True


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""

    report_id: str
    generation_time: float
    report_type: str
    format: ReportFormat
    total_pages: int = 1
    file_size_bytes: int = 0


@dataclass
class TestReport:
    """Complete test report structure."""

    metadata: ReportMetadata
    content: str
    attachments: dict[str, Any]
    summary: dict[str, Any]


class MarkdownReportGenerator:
    """Generates markdown format test reports.

    This class creates well-formatted markdown reports with tables,
    charts, and structured sections for easy reading and sharing.
    """

    def __init__(self):
        """Initialize the markdown report generator."""
        self.report_sections: list[str] = []

    def generate_comprehensive_report(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
        coverage_report: CoverageReport,
        quality_metrics: QualityMetrics,
        config: ReportConfig,
    ) -> str:
        """Generate comprehensive markdown report.

        Args:
            test_results: List of test execution results
            analysis_results: List of test analysis results
            coverage_report: Coverage analysis report
            quality_metrics: Test quality metrics
            config: Report configuration

        Returns:
            str: Formatted markdown report
        """
        sections = []

        # Report header
        sections.append(self._generate_header(test_results, quality_metrics))

        # Executive summary
        if config.level in [
            ReportLevel.EXECUTIVE,
            ReportLevel.SUMMARY,
            ReportLevel.DETAILED,
        ]:
            sections.append(
                self._generate_executive_summary(test_results, quality_metrics, coverage_report)
            )

        # Test results overview
        if config.level in [
            ReportLevel.SUMMARY,
            ReportLevel.DETAILED,
            ReportLevel.TECHNICAL,
        ]:
            sections.append(self._generate_test_results_overview(test_results, config))

        # Coverage analysis
        if config.level in [ReportLevel.DETAILED, ReportLevel.TECHNICAL]:
            sections.append(self._generate_coverage_section(coverage_report))

        # Quality analysis
        if config.level in [ReportLevel.DETAILED, ReportLevel.TECHNICAL]:
            sections.append(self._generate_quality_section(quality_metrics))

        # Component analysis
        if config.group_by_component and config.level in [
            ReportLevel.DETAILED,
            ReportLevel.TECHNICAL,
        ]:
            sections.append(self._generate_component_analysis(test_results, analysis_results))

        # Recommendations
        if config.include_recommendations:
            sections.append(
                self._generate_recommendations_section(analysis_results, coverage_report)
            )

        # Technical details
        if config.level == ReportLevel.TECHNICAL:
            sections.append(self._generate_technical_details(test_results, analysis_results))

        return "\n\n".join(sections)

    def _generate_header(
        self, test_results: list[TestExecutionResult], quality_metrics: QualityMetrics
    ) -> str:
        """Generate report header."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestExecutionStatus.COMPLETED)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        return f"""# Integration Test Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Quick Stats
- **Total Tests:** {total_tests}
- **Passed:** {passed_tests}
- **Failed:** {total_tests - passed_tests}
- **Success Rate:** {success_rate:.1f}%
- **Overall Quality Score:** {quality_metrics.overall_score:.2f}/1.0"""

    def _generate_executive_summary(
        self,
        test_results: list[TestExecutionResult],
        quality_metrics: QualityMetrics,
        coverage_report: CoverageReport,
    ) -> str:
        """Generate executive summary section."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestExecutionStatus.COMPLETED)

        # Calculate average execution time
        avg_time = statistics.mean([r.execution_time for r in test_results]) if test_results else 0

        # Determine overall status
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        if success_rate >= 95:
            status = "✅ EXCELLENT"
        elif success_rate >= 80:
            status = "✔️ GOOD"
        elif success_rate >= 60:
            status = "⚠️ NEEDS ATTENTION"
        else:
            status = "❌ CRITICAL"

        return f"""## Executive Summary

**Overall Status:** {status}

### Key Metrics
- **Test Coverage:** {coverage_report.overall_coverage:.1f}%
- **Quality Score:** {quality_metrics.overall_score:.2f}/1.0
- **Reliability Score:** {quality_metrics.reliability_score:.2f}/1.0
- **Performance Score:** {quality_metrics.performance_score:.2f}/1.0
- **Average Execution Time:** {avg_time:.2f}s

### Critical Issues
{self._format_critical_issues(test_results, coverage_report)}

### Recommendations
{self._format_top_recommendations(coverage_report)}"""

    def _format_critical_issues(
        self, test_results: list[TestExecutionResult], coverage_report: CoverageReport
    ) -> str:
        """Format critical issues for executive summary."""
        issues = []

        # Failed tests
        failed_tests = [r for r in test_results if r.status == TestExecutionStatus.FAILED]
        if failed_tests:
            issues.append(f"- {len(failed_tests)} test(s) failed")

        # Coverage gaps
        high_severity_gaps = [
            gap for gap in coverage_report.coverage_gaps if gap.severity == "high"
        ]
        if high_severity_gaps:
            issues.append(f"- {len(high_severity_gaps)} high-severity coverage gap(s)")

        # Performance issues
        slow_tests = [r for r in test_results if r.execution_time > 30.0]
        if slow_tests:
            issues.append(f"- {len(slow_tests)} slow test(s) (>30s)")

        return "\n".join(issues) if issues else "- No critical issues identified"

    def _format_top_recommendations(self, coverage_report: CoverageReport) -> str:
        """Format top recommendations for executive summary."""
        top_recommendations = coverage_report.recommendations[:3]  # Top 3
        if top_recommendations:
            return "\n".join(f"- {rec}" for rec in top_recommendations)
        else:
            return "- Continue maintaining current test quality"

    def _generate_test_results_overview(
        self, test_results: list[TestExecutionResult], config: ReportConfig
    ) -> str:
        """Generate test results overview section."""
        # Group results by component
        component_results = {}
        for result in test_results:
            if result.component not in component_results:
                component_results[result.component] = []
            component_results[result.component].append(result)

        section = ["## Test Results Overview"]

        # Summary table
        section.append("### Results by Component")
        section.append("| Component | Total | Passed | Failed | Success Rate | Avg Time |")
        section.append("|-----------|-------|--------|--------|--------------|----------|")

        for component, results in component_results.items():
            total = len(results)
            passed = sum(1 for r in results if r.status == TestExecutionStatus.COMPLETED)
            failed = total - passed
            success_rate = (passed / total * 100) if total > 0 else 0
            avg_time = statistics.mean([r.execution_time for r in results]) if results else 0

            section.append(
                f"| {component} | {total} | {passed} | {failed} | {success_rate:.1f}% | {avg_time:.2f}s |"
            )

        # Failed tests details (if any)
        failed_tests = [r for r in test_results if r.status == TestExecutionStatus.FAILED]
        if failed_tests and config.level in [
            ReportLevel.DETAILED,
            ReportLevel.TECHNICAL,
        ]:
            section.append("\n### Failed Tests")
            for test in failed_tests[: config.max_details]:
                section.append(
                    f"- **{test.component}/{test.test_name}**: {test.error_message or 'Unknown error'}"
                )

        return "\n".join(section)

    def _generate_coverage_section(self, coverage_report: CoverageReport) -> str:
        """Generate coverage analysis section."""
        section = ["## Coverage Analysis"]

        section.append(f"**Overall Coverage:** {coverage_report.overall_coverage:.1f}%")
        section.append("")

        # Coverage by type
        section.append("### Coverage by Type")
        section.append("| Coverage Type | Percentage | Level | Missing Items |")
        section.append("|---------------|------------|-------|---------------|")

        for metric in coverage_report.coverage_metrics:
            section.append(
                f"| {metric.coverage_type.value.title()} | {metric.coverage_percentage:.1f}% | "
                f"{metric.coverage_level.value.title()} | {len(metric.missing_items)} |"
            )

        # Coverage gaps
        if coverage_report.coverage_gaps:
            section.append("\n### Coverage Gaps")
            for gap in coverage_report.coverage_gaps:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(gap.severity, "⚪")
                section.append(f"- {severity_emoji} **{gap.gap_type.title()}**: {gap.description}")
                section.append(f"  - *Effort: {gap.estimated_effort.title()}*")

        return "\n".join(section)

    def _generate_quality_section(self, quality_metrics: QualityMetrics) -> str:
        """Generate quality analysis section."""
        section = ["## Quality Analysis"]

        section.append("### Quality Scores")
        section.append("| Metric | Score | Level |")
        section.append("|--------|-------|-------|")
        section.append(
            f"| Overall | {quality_metrics.overall_score:.2f} | {self._score_to_level(quality_metrics.overall_score)} |"
        )
        section.append(
            f"| Reliability | {quality_metrics.reliability_score:.2f} | {self._score_to_level(quality_metrics.reliability_score)} |"
        )
        section.append(
            f"| Performance | {quality_metrics.performance_score:.2f} | {self._score_to_level(quality_metrics.performance_score)} |"
        )
        section.append(
            f"| Coverage | {quality_metrics.coverage_score:.2f} | {self._score_to_level(quality_metrics.coverage_score)} |"
        )
        section.append(
            f"| Maintainability | {quality_metrics.maintainability_score:.2f} | {self._score_to_level(quality_metrics.maintainability_score)} |"
        )

        # Quality details
        section.append("\n### Quality Details")
        for key, value in quality_metrics.details.items():
            section.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        return "\n".join(section)

    def _generate_component_analysis(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
    ) -> str:
        """Generate component-specific analysis section."""
        section = ["## Component Analysis"]

        # Group by component
        component_analysis = {}
        for analysis in analysis_results:
            if analysis.component not in component_analysis:
                component_analysis[analysis.component] = []
            component_analysis[analysis.component].append(analysis)

        for component, analyses in component_analysis.items():
            section.append(f"\n### {component.title()} Component")

            # Component metrics
            total_tests = len(analyses)
            passed_tests = sum(1 for a in analyses if a.passed)
            avg_performance = (
                statistics.mean([a.performance_score for a in analyses]) if analyses else 0
            )
            avg_quality = statistics.mean([a.quality_score for a in analyses]) if analyses else 0

            section.append(f"- **Tests**: {passed_tests}/{total_tests} passed")
            section.append(f"- **Avg Performance Score**: {avg_performance:.2f}")
            section.append(f"- **Avg Quality Score**: {avg_quality:.2f}")

            # Key insights
            all_insights = []
            for analysis in analyses:
                all_insights.extend(analysis.insights)

            unique_insights = list(set(all_insights))[:3]  # Top 3 unique insights
            if unique_insights:
                section.append("- **Key Insights**:")
                for insight in unique_insights:
                    section.append(f"  - {insight}")

        return "\n".join(section)

    def _generate_recommendations_section(
        self,
        analysis_results: list[TestAnalysisResult],
        coverage_report: CoverageReport,
    ) -> str:
        """Generate recommendations section."""
        section = ["## Recommendations"]

        # Collect all recommendations
        all_recommendations = []

        # From coverage report
        all_recommendations.extend(coverage_report.recommendations)

        # From analysis results
        for analysis in analysis_results:
            all_recommendations.extend(analysis.recommendations)

        # Deduplicate and prioritize
        unique_recommendations = list(set(all_recommendations))

        # Categorize recommendations
        high_priority = [
            r
            for r in unique_recommendations
            if any(
                keyword in r.lower() for keyword in ["critical", "urgent", "immediately", "high"]
            )
        ]
        medium_priority = [
            r
            for r in unique_recommendations
            if r not in high_priority
            and any(keyword in r.lower() for keyword in ["improve", "optimize", "enhance"])
        ]
        low_priority = [
            r for r in unique_recommendations if r not in high_priority and r not in medium_priority
        ]

        if high_priority:
            section.append("### High Priority")
            for rec in high_priority:
                section.append(f"🔴 {rec}")

        if medium_priority:
            section.append("\n### Medium Priority")
            for rec in medium_priority:
                section.append(f"🟡 {rec}")

        if low_priority:
            section.append("\n### Low Priority")
            for rec in low_priority[:5]:  # Limit to 5
                section.append(f"🟢 {rec}")

        return "\n".join(section)

    def _generate_technical_details(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
    ) -> str:
        """Generate technical details section."""
        section = ["## Technical Details"]

        # Execution details
        section.append("### Execution Details")
        total_time = sum([r.execution_time for r in test_results])
        section.append(f"- **Total Execution Time**: {total_time:.2f}s")
        section.append(f"- **Test Count**: {len(test_results)}")
        section.append(f"- **Analysis Count**: {len(analysis_results)}")

        # Performance distribution
        execution_times = [
            r.execution_time for r in test_results if r.status == TestExecutionStatus.COMPLETED
        ]
        if execution_times:
            section.append("\n### Performance Distribution")
            section.append(f"- **Min Time**: {min(execution_times):.2f}s")
            section.append(f"- **Max Time**: {max(execution_times):.2f}s")
            section.append(f"- **Mean Time**: {statistics.mean(execution_times):.2f}s")
            section.append(f"- **Median Time**: {statistics.median(execution_times):.2f}s")

        # Error analysis
        failed_results = [r for r in test_results if r.status == TestExecutionStatus.FAILED]
        if failed_results:
            section.append("\n### Error Analysis")
            error_types = {}
            for result in failed_results:
                if result.error_message:
                    error_type = (
                        result.error_message.split(":")[0]
                        if ":" in result.error_message
                        else "Unknown"
                    )
                    error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                section.append(f"- **{error_type}**: {count} occurrence(s)")

        return "\n".join(section)

    def _score_to_level(self, score: float) -> str:
        """Convert score to level description."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Moderate"
        elif score >= 0.3:
            return "Poor"
        else:
            return "Inadequate"


class HTMLReportGenerator:
    """Generates HTML format test reports with interactive elements."""

    def __init__(self):
        """Initialize the HTML report generator."""
        pass

    def generate_html_report(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
        coverage_report: CoverageReport,
        quality_metrics: QualityMetrics,
        config: ReportConfig,
    ) -> str:
        """Generate HTML format report.

        Args:
            test_results: List of test execution results
            analysis_results: List of test analysis results
            coverage_report: Coverage analysis report
            quality_metrics: Test quality metrics
            config: Report configuration

        Returns:
            str: HTML formatted report
        """
        # Generate markdown first, then convert to HTML
        markdown_generator = MarkdownReportGenerator()
        markdown_content = markdown_generator.generate_comprehensive_report(
            test_results, analysis_results, coverage_report, quality_metrics, config
        )

        # Simple markdown to HTML conversion (in practice, use a proper library)
        html_content = self._markdown_to_html(markdown_content)

        return self._wrap_in_html_template(html_content, quality_metrics)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (simplified implementation)."""
        html = markdown

        # Convert headers
        html = html.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
        html = html.replace("\n### ", "</h2>\n<h3>").replace("\n#### ", "</h3>\n<h4>")

        # Convert tables (simplified)
        lines = html.split("\n")
        html_lines = []
        in_table = False

        for line in lines:
            if "|" in line and not line.strip().startswith("|"):
                if not in_table:
                    html_lines.append('<table class="table table-striped">')
                    in_table = True
                    html_lines.append("<thead><tr>")
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    for cell in cells:
                        html_lines.append(f"<th>{cell}</th>")
                    html_lines.append("</tr></thead><tbody>")
                elif line.strip().startswith("|---"):
                    continue  # Skip separator line
                else:
                    html_lines.append("<tr>")
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    for cell in cells:
                        html_lines.append(f"<td>{cell}</td>")
                    html_lines.append("</tr>")
            else:
                if in_table:
                    html_lines.append("</tbody></table>")
                    in_table = False
                html_lines.append(line)

        if in_table:
            html_lines.append("</tbody></table>")

        return "\n".join(html_lines)

    def _wrap_in_html_template(self, content: str, quality_metrics: QualityMetrics) -> str:
        """Wrap content in HTML template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integration Test Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .score-excellent {{ color: #28a745; }}
        .score-good {{ color: #17a2b8; }}
        .score-moderate {{ color: #ffc107; }}
        .score-poor {{ color: #fd7e14; }}
        .score-inadequate {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                {content}
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""


class JSONReportGenerator:
    """Generates JSON format test reports for programmatic consumption."""

    def __init__(self):
        """Initialize the JSON report generator."""
        pass

    def generate_json_report(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
        coverage_report: CoverageReport,
        quality_metrics: QualityMetrics,
        config: ReportConfig,
    ) -> str:
        """Generate JSON format report.

        Args:
            test_results: List of test execution results
            analysis_results: List of test analysis results
            coverage_report: Coverage analysis report
            quality_metrics: Test quality metrics
            config: Report configuration

        Returns:
            str: JSON formatted report
        """
        # Convert dataclasses to dictionaries
        report_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "report_type": "integration_test_report",
                "format": config.format.value,
                "level": config.level.value,
            },
            "summary": {
                "total_tests": len(test_results),
                "passed_tests": sum(
                    1 for r in test_results if r.status == TestExecutionStatus.COMPLETED
                ),
                "failed_tests": sum(
                    1 for r in test_results if r.status == TestExecutionStatus.FAILED
                ),
                "overall_coverage": coverage_report.overall_coverage,
                "quality_score": quality_metrics.overall_score,
            },
            "test_results": [
                {
                    "component": result.component,
                    "test_name": result.test_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "results": result.results,
                }
                for result in test_results
            ],
            "analysis_results": [
                {
                    "component": analysis.component,
                    "test_name": analysis.test_name,
                    "passed": analysis.passed,
                    "performance_score": analysis.performance_score,
                    "quality_score": analysis.quality_score,
                    "insights": analysis.insights,
                    "recommendations": analysis.recommendations,
                }
                for analysis in analysis_results
            ],
            "coverage_report": {
                "overall_coverage": coverage_report.overall_coverage,
                "coverage_metrics": [
                    {
                        "type": metric.coverage_type.value,
                        "percentage": metric.coverage_percentage,
                        "level": metric.coverage_level.value,
                        "total_items": metric.total_items,
                        "covered_items": metric.covered_items,
                        "missing_items": metric.missing_items,
                    }
                    for metric in coverage_report.coverage_metrics
                ],
                "coverage_gaps": [
                    {
                        "type": gap.gap_type,
                        "description": gap.description,
                        "severity": gap.severity,
                        "estimated_effort": gap.estimated_effort,
                        "recommendations": gap.recommendations,
                    }
                    for gap in coverage_report.coverage_gaps
                ],
                "recommendations": coverage_report.recommendations,
            },
            "quality_metrics": {
                "overall_score": quality_metrics.overall_score,
                "reliability_score": quality_metrics.reliability_score,
                "performance_score": quality_metrics.performance_score,
                "coverage_score": quality_metrics.coverage_score,
                "maintainability_score": quality_metrics.maintainability_score,
                "details": quality_metrics.details,
            },
        }

        return json.dumps(report_data, indent=2, default=str)


class TestReportGenerator:
    """Main test report generation coordinator.

    This class coordinates different report generators and provides
    a unified interface for creating test reports in various formats.
    """

    def __init__(self):
        """Initialize the test report generator."""
        self.markdown_generator = MarkdownReportGenerator()
        self.html_generator = HTMLReportGenerator()
        self.json_generator = JSONReportGenerator()

    def generate_report(
        self,
        test_results: list[TestExecutionResult],
        analysis_results: list[TestAnalysisResult],
        coverage_report: CoverageReport,
        quality_metrics: QualityMetrics,
        config: ReportConfig,
    ) -> TestReport:
        """Generate test report in specified format.

        Args:
            test_results: List of test execution results
            analysis_results: List of test analysis results
            coverage_report: Coverage analysis report
            quality_metrics: Test quality metrics
            config: Report configuration

        Returns:
            TestReport: Generated test report
        """
        start_time = time.time()

        # Generate content based on format
        if config.format == ReportFormat.MARKDOWN:
            content = self.markdown_generator.generate_comprehensive_report(
                test_results, analysis_results, coverage_report, quality_metrics, config
            )
        elif config.format == ReportFormat.HTML:
            content = self.html_generator.generate_html_report(
                test_results, analysis_results, coverage_report, quality_metrics, config
            )
        elif config.format == ReportFormat.JSON:
            content = self.json_generator.generate_json_report(
                test_results, analysis_results, coverage_report, quality_metrics, config
            )
        else:
            # Default to markdown
            content = self.markdown_generator.generate_comprehensive_report(
                test_results, analysis_results, coverage_report, quality_metrics, config
            )

        generation_time = time.time() - start_time

        # Create metadata
        metadata = ReportMetadata(
            report_id=f"test_report_{int(time.time())}",
            generation_time=generation_time,
            report_type="integration_test_report",
            format=config.format,
            file_size_bytes=len(content.encode("utf-8")),
        )

        # Create summary
        summary = {
            "total_tests": len(test_results),
            "passed_tests": sum(
                1 for r in test_results if r.status == TestExecutionStatus.COMPLETED
            ),
            "overall_quality": quality_metrics.overall_score,
            "overall_coverage": coverage_report.overall_coverage,
            "generation_time": generation_time,
        }

        return TestReport(metadata=metadata, content=content, attachments={}, summary=summary)

    def generate_executive_summary(
        self,
        test_results: list[TestExecutionResult],
        quality_metrics: QualityMetrics,
        coverage_report: CoverageReport,
    ) -> str:
        """Generate executive summary report.

        Args:
            test_results: List of test execution results
            quality_metrics: Test quality metrics
            coverage_report: Coverage analysis report

        Returns:
            str: Executive summary report
        """
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestExecutionStatus.COMPLETED)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Determine status
        if success_rate >= 95 and quality_metrics.overall_score >= 0.8:
            status = "🟢 EXCELLENT"
            status_desc = "All systems performing optimally"
        elif success_rate >= 80 and quality_metrics.overall_score >= 0.6:
            status = "🟡 GOOD"
            status_desc = "Systems performing well with minor issues"
        elif success_rate >= 60:
            status = "🟠 ATTENTION NEEDED"
            status_desc = "Some systems require attention"
        else:
            status = "🔴 CRITICAL"
            status_desc = "Critical issues require immediate action"

        return f"""# Executive Summary - Integration Testing

**Status:** {status}
**Assessment:** {status_desc}

## Key Metrics
- **Test Success Rate:** {success_rate:.1f}% ({passed_tests}/{total_tests})
- **Overall Quality Score:** {quality_metrics.overall_score:.2f}/1.0
- **Test Coverage:** {coverage_report.overall_coverage:.1f}%
- **System Reliability:** {quality_metrics.reliability_score:.2f}/1.0

## Priority Actions
{self._get_priority_actions(test_results, coverage_report)}

## Next Steps
{self._get_next_steps(quality_metrics, coverage_report)}

---
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*"""

    def _get_priority_actions(
        self, test_results: list[TestExecutionResult], coverage_report: CoverageReport
    ) -> str:
        """Get priority actions for executive summary."""
        actions = []

        # Failed tests
        failed_tests = [r for r in test_results if r.status == TestExecutionStatus.FAILED]
        if failed_tests:
            actions.append(f"1. Investigate {len(failed_tests)} failed test(s)")

        # Coverage gaps
        high_severity_gaps = [
            gap for gap in coverage_report.coverage_gaps if gap.severity == "high"
        ]
        if high_severity_gaps:
            actions.append(f"2. Address {len(high_severity_gaps)} high-severity coverage gap(s)")

        # Performance issues
        slow_tests = [r for r in test_results if r.execution_time > 30.0]
        if slow_tests:
            actions.append(f"3. Optimize {len(slow_tests)} slow-performing test(s)")

        if not actions:
            actions.append("1. Continue monitoring current test quality")
            actions.append("2. Maintain existing test coverage")

        return "\n".join(actions)

    def _get_next_steps(
        self, quality_metrics: QualityMetrics, coverage_report: CoverageReport
    ) -> str:
        """Get next steps for executive summary."""
        steps = []

        if quality_metrics.overall_score < 0.7:
            steps.append("1. Improve overall test quality through better test design")

        if coverage_report.overall_coverage < 80:
            steps.append("2. Increase test coverage in identified gap areas")

        if quality_metrics.performance_score < 0.6:
            steps.append("3. Optimize test performance and execution efficiency")

        if not steps:
            steps.append("1. Maintain current quality standards")
            steps.append("2. Continue regular monitoring and assessment")

        return "\n".join(steps)


# Convenience functions
def generate_test_report(
    test_results: list[TestExecutionResult],
    analysis_results: list[TestAnalysisResult],
    coverage_report: CoverageReport,
    quality_metrics: QualityMetrics,
    format: ReportFormat = ReportFormat.MARKDOWN,
    level: ReportLevel = ReportLevel.DETAILED,
) -> TestReport:
    """Generate comprehensive test report.

    Args:
        test_results: List of test execution results
        analysis_results: List of test analysis results
        coverage_report: Coverage analysis report
        quality_metrics: Test quality metrics
        format: Report format
        level: Report detail level

    Returns:
        TestReport: Generated test report
    """
    generator = TestReportGenerator()
    config = ReportConfig(format=format, level=level)

    return generator.generate_report(
        test_results, analysis_results, coverage_report, quality_metrics, config
    )


def generate_executive_summary(
    test_results: list[TestExecutionResult],
    quality_metrics: QualityMetrics,
    coverage_report: CoverageReport,
) -> str:
    """Generate executive summary report.

    Args:
        test_results: List of test execution results
        quality_metrics: Test quality metrics
        coverage_report: Coverage analysis report

    Returns:
        str: Executive summary report
    """
    generator = TestReportGenerator()
    return generator.generate_executive_summary(test_results, quality_metrics, coverage_report)


def export_test_results(
    test_results: list[TestExecutionResult],
    format: ReportFormat = ReportFormat.JSON,
    file_path: str | None = None,
) -> str:
    """Export test results to specified format.

    Args:
        test_results: List of test execution results
        format: Export format
        file_path: Optional file path to save to

    Returns:
        str: Exported content
    """
    if format == ReportFormat.JSON:
        content = json.dumps(
            [
                {
                    "component": result.component,
                    "test_name": result.test_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "results": result.results,
                }
                for result in test_results
            ],
            indent=2,
            default=str,
        )
    elif format == ReportFormat.CSV:
        lines = ["component,test_name,status,execution_time,error_message"]
        for result in test_results:
            error_msg = (result.error_message or "").replace(",", ";")
            lines.append(
                f"{result.component},{result.test_name},{result.status.value},{result.execution_time},{error_msg}"
            )
        content = "\n".join(lines)
    else:
        content = str(test_results)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content
