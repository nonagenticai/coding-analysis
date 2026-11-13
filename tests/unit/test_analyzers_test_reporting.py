"""Unit tests for src/processing/analyzers/test_reporting.py"""

from __future__ import annotations

import json
from typing import Any

from processing.analyzers.test_analysis import FailureCategory, QualityMetrics, TestAnalysisResult
from processing.analyzers.test_coverage import (
    CoverageGap,
    CoverageLevel,
    CoverageMetric,
    CoverageReport,
    CoverageType,
)
from processing.analyzers.test_execution import TestExecutionResult, TestExecutionStatus
from processing.analyzers.test_reporting import (
    HTMLReportGenerator,
    JSONReportGenerator,
    MarkdownReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportLevel,
    ReportMetadata,
    TestReport,
    TestReportGenerator,
    export_test_results,
    generate_executive_summary,
    generate_test_report,
)

# ---------------------------------------------------------------------------
# Helpers to build fixture data
# ---------------------------------------------------------------------------


def _make_test_result(
    component: str = "gateway",
    test_name: str = "test_something",
    status: TestExecutionStatus = TestExecutionStatus.COMPLETED,
    execution_time: float = 1.5,
    error_message: str | None = None,
    results: dict[str, Any] | None = None,
) -> TestExecutionResult:
    return TestExecutionResult(
        component=component,
        test_name=test_name,
        status=status,
        execution_time=execution_time,
        results=results or {},
        error_message=error_message,
    )


def _make_analysis_result(
    component: str = "gateway",
    test_name: str = "test_something",
    passed: bool = True,
    performance_score: float = 0.85,
    quality_score: float = 0.9,
    insights: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> TestAnalysisResult:
    return TestAnalysisResult(
        component=component,
        test_name=test_name,
        passed=passed,
        execution_time=1.0,
        failure_category=None if passed else FailureCategory.RUNTIME_ERROR,
        performance_score=performance_score,
        quality_score=quality_score,
        insights=insights or ["insight1"],
        recommendations=recommendations or [],
    )


def _make_quality_metrics(
    overall: float = 0.85,
    reliability: float = 0.9,
    performance: float = 0.8,
    coverage: float = 0.75,
    maintainability: float = 0.7,
) -> QualityMetrics:
    return QualityMetrics(
        overall_score=overall,
        reliability_score=reliability,
        performance_score=performance,
        coverage_score=coverage,
        maintainability_score=maintainability,
        details={"test_count": 10, "failure_rate": 0.1},
    )


def _make_coverage_metric(
    ctype: CoverageType = CoverageType.FUNCTIONAL,
    pct: float = 80.0,
) -> CoverageMetric:
    return CoverageMetric(
        coverage_type=ctype,
        total_items=100,
        covered_items=int(pct),
        coverage_percentage=pct,
        coverage_level=CoverageLevel.GOOD,
        missing_items=["item1", "item2"],
        details={},
    )


def _make_coverage_gap(severity: str = "high") -> CoverageGap:
    return CoverageGap(
        gap_type="missing_tests",
        description="Some functions are not tested",
        severity=severity,
        affected_areas=["area1"],
        recommendations=["Add tests for area1"],
        estimated_effort="medium",
    )


def _make_coverage_report(
    overall: float = 75.0,
    gaps: list[CoverageGap] | None = None,
    recommendations: list[str] | None = None,
) -> CoverageReport:
    return CoverageReport(
        overall_coverage=overall,
        coverage_metrics=[_make_coverage_metric()],
        coverage_gaps=gaps or [],
        recommendations=recommendations or ["Add more tests"],
        trend_analysis={},
        details={},
    )


# ---------------------------------------------------------------------------
# ReportFormat / ReportLevel enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_report_format_values(self):
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.PLAIN_TEXT.value == "plain_text"
        assert ReportFormat.CSV.value == "csv"

    def test_report_level_values(self):
        assert ReportLevel.EXECUTIVE.value == "executive"
        assert ReportLevel.SUMMARY.value == "summary"
        assert ReportLevel.DETAILED.value == "detailed"
        assert ReportLevel.TECHNICAL.value == "technical"


# ---------------------------------------------------------------------------
# ReportConfig / ReportMetadata / TestReport dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_report_config_defaults(self):
        cfg = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.DETAILED)
        assert cfg.include_charts is False
        assert cfg.include_recommendations is True
        assert cfg.include_historical is False
        assert cfg.max_details == 100
        assert cfg.group_by_component is True

    def test_report_metadata(self):
        meta = ReportMetadata(
            report_id="r1",
            generation_time=0.5,
            report_type="test",
            format=ReportFormat.JSON,
            total_pages=2,
            file_size_bytes=1024,
        )
        assert meta.report_id == "r1"
        assert meta.file_size_bytes == 1024

    def test_test_report(self):
        meta = ReportMetadata(
            report_id="r1",
            generation_time=0.1,
            report_type="t",
            format=ReportFormat.MARKDOWN,
        )
        report = TestReport(
            metadata=meta,
            content="hello",
            attachments={"a": 1},
            summary={"total": 5},
        )
        assert report.content == "hello"
        assert report.attachments == {"a": 1}


# ---------------------------------------------------------------------------
# MarkdownReportGenerator
# ---------------------------------------------------------------------------


class TestMarkdownReportGenerator:
    def setup_method(self):
        self.gen = MarkdownReportGenerator()
        self.test_results = [
            _make_test_result(component="gateway", test_name="test_a"),
            _make_test_result(
                component="gateway",
                test_name="test_b",
                status=TestExecutionStatus.FAILED,
                error_message="AssertionError: bad value",
            ),
            _make_test_result(component="analysis", test_name="test_c", execution_time=35.0),
        ]
        self.analysis_results = [
            _make_analysis_result(component="gateway", test_name="test_a"),
            _make_analysis_result(
                component="gateway",
                test_name="test_b",
                passed=False,
                recommendations=["Fix the critical bug immediately"],
            ),
            _make_analysis_result(component="analysis", test_name="test_c"),
        ]
        self.coverage_report = _make_coverage_report(
            gaps=[_make_coverage_gap("high"), _make_coverage_gap("low")],
            recommendations=["Improve coverage", "Add edge case tests"],
        )
        self.quality_metrics = _make_quality_metrics()

    # -- _generate_header ---------------------------------------------------

    def test_generate_header_contains_stats(self):
        header = self.gen._generate_header(self.test_results, self.quality_metrics)
        assert "Integration Test Report" in header
        assert "Total Tests:** 3" in header
        assert "Passed:** 2" in header
        assert "Failed:** 1" in header
        assert "Overall Quality Score" in header

    def test_generate_header_zero_tests(self):
        header = self.gen._generate_header([], self.quality_metrics)
        assert "Total Tests:** 0" in header
        assert "Success Rate:** 0.0%" in header

    # -- _generate_executive_summary ----------------------------------------

    def test_executive_summary_excellent(self):
        results = [_make_test_result() for _ in range(20)]
        qm = _make_quality_metrics(overall=0.95)
        cr = _make_coverage_report(overall=90.0)
        summary = self.gen._generate_executive_summary(results, qm, cr)
        assert "EXCELLENT" in summary
        assert "Executive Summary" in summary

    def test_executive_summary_good(self):
        results = [_make_test_result() for _ in range(10)]
        results.append(_make_test_result(status=TestExecutionStatus.FAILED, error_message="err"))
        summary = self.gen._generate_executive_summary(
            results, self.quality_metrics, self.coverage_report
        )
        assert "GOOD" in summary

    def test_executive_summary_needs_attention(self):
        passed = [_make_test_result() for _ in range(7)]
        failed = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="e")
            for _ in range(3)
        ]
        # 70% success rate -> NEEDS ATTENTION
        summary = self.gen._generate_executive_summary(
            passed + failed, self.quality_metrics, self.coverage_report
        )
        assert "NEEDS ATTENTION" in summary

    def test_executive_summary_critical(self):
        failed = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="e")
            for _ in range(10)
        ]
        summary = self.gen._generate_executive_summary(
            failed, self.quality_metrics, self.coverage_report
        )
        assert "CRITICAL" in summary

    # -- _format_critical_issues --------------------------------------------

    def test_format_critical_issues_with_failures(self):
        result = self.gen._format_critical_issues(self.test_results, self.coverage_report)
        assert "test(s) failed" in result
        assert "high-severity coverage gap" in result
        assert "slow test(s)" in result  # one test at 35s > 30s

    def test_format_critical_issues_no_issues(self):
        results = [_make_test_result()]
        cr = _make_coverage_report()
        result = self.gen._format_critical_issues(results, cr)
        assert "No critical issues" in result

    # -- _format_top_recommendations ----------------------------------------

    def test_format_top_recommendations_with_recs(self):
        result = self.gen._format_top_recommendations(self.coverage_report)
        assert "Improve coverage" in result

    def test_format_top_recommendations_empty(self):
        cr = CoverageReport(
            overall_coverage=80.0,
            coverage_metrics=[],
            coverage_gaps=[],
            recommendations=[],
            trend_analysis={},
            details={},
        )
        result = self.gen._format_top_recommendations(cr)
        assert "Continue maintaining" in result

    # -- _generate_test_results_overview ------------------------------------

    def test_test_results_overview_grouped(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.DETAILED)
        overview = self.gen._generate_test_results_overview(self.test_results, config)
        assert "Results by Component" in overview
        assert "gateway" in overview
        assert "analysis" in overview
        assert "Failed Tests" in overview

    def test_test_results_overview_summary_level(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.SUMMARY)
        overview = self.gen._generate_test_results_overview(self.test_results, config)
        # summary level should NOT include failed test details
        assert "Failed Tests" not in overview

    # -- _generate_coverage_section -----------------------------------------

    def test_coverage_section(self):
        section = self.gen._generate_coverage_section(self.coverage_report)
        assert "Coverage Analysis" in section
        assert "Coverage by Type" in section
        assert "Coverage Gaps" in section

    def test_coverage_section_no_gaps(self):
        cr = _make_coverage_report()
        section = self.gen._generate_coverage_section(cr)
        assert "Coverage Gaps" not in section

    # -- _generate_quality_section ------------------------------------------

    def test_quality_section(self):
        section = self.gen._generate_quality_section(self.quality_metrics)
        assert "Quality Analysis" in section
        assert "Quality Scores" in section
        assert "Quality Details" in section
        assert "Test Count" in section  # from details dict

    # -- _generate_component_analysis ---------------------------------------

    def test_component_analysis(self):
        section = self.gen._generate_component_analysis(self.test_results, self.analysis_results)
        assert "Component Analysis" in section
        assert "Gateway" in section
        assert "Analysis" in section

    # -- _generate_recommendations_section ----------------------------------

    def test_recommendations_section_prioritized(self):
        analysis_with_recs = [
            _make_analysis_result(recommendations=["Fix critical issue now"]),
            _make_analysis_result(recommendations=["Improve test speed"]),
            _make_analysis_result(recommendations=["Add some logging"]),
        ]
        section = self.gen._generate_recommendations_section(
            analysis_with_recs, self.coverage_report
        )
        assert "Recommendations" in section
        assert "High Priority" in section
        assert "Medium Priority" in section

    def test_recommendations_section_low_only(self):
        analysis_with_recs = [
            _make_analysis_result(recommendations=["Add some logging"]),
        ]
        cr = _make_coverage_report(recommendations=["Add some tests"])
        section = self.gen._generate_recommendations_section(analysis_with_recs, cr)
        assert "Low Priority" in section

    # -- _generate_technical_details ----------------------------------------

    def test_technical_details(self):
        section = self.gen._generate_technical_details(self.test_results, self.analysis_results)
        assert "Technical Details" in section
        assert "Total Execution Time" in section
        assert "Performance Distribution" in section
        assert "Error Analysis" in section  # because we have a failed test

    def test_technical_details_no_failures(self):
        results = [_make_test_result(), _make_test_result()]
        section = self.gen._generate_technical_details(results, [])
        assert "Error Analysis" not in section

    def test_technical_details_no_completed(self):
        results = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="err"),
        ]
        section = self.gen._generate_technical_details(results, [])
        # No completed tests -> no performance distribution
        assert "Performance Distribution" not in section

    # -- _score_to_level ----------------------------------------------------

    def test_score_to_level(self):
        assert self.gen._score_to_level(0.95) == "Excellent"
        assert self.gen._score_to_level(0.9) == "Excellent"
        assert self.gen._score_to_level(0.75) == "Good"
        assert self.gen._score_to_level(0.55) == "Moderate"
        assert self.gen._score_to_level(0.35) == "Poor"
        assert self.gen._score_to_level(0.1) == "Inadequate"

    # -- generate_comprehensive_report (integration of all sections) --------

    def test_comprehensive_report_executive(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.EXECUTIVE)
        report = self.gen.generate_comprehensive_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert "Integration Test Report" in report
        assert "Executive Summary" in report
        # executive level does NOT include test results overview
        assert "Test Results Overview" not in report

    def test_comprehensive_report_detailed(self):
        config = ReportConfig(
            format=ReportFormat.MARKDOWN,
            level=ReportLevel.DETAILED,
            include_recommendations=True,
            group_by_component=True,
        )
        report = self.gen.generate_comprehensive_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert "Executive Summary" in report
        assert "Test Results Overview" in report
        assert "Coverage Analysis" in report
        assert "Quality Analysis" in report
        assert "Component Analysis" in report
        assert "Recommendations" in report

    def test_comprehensive_report_technical(self):
        config = ReportConfig(
            format=ReportFormat.MARKDOWN,
            level=ReportLevel.TECHNICAL,
            include_recommendations=False,
            group_by_component=False,
        )
        report = self.gen.generate_comprehensive_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert "Technical Details" in report
        # recommendations disabled
        assert "Recommendations" not in report
        # group_by_component=False so no component analysis
        assert "Component Analysis" not in report

    def test_comprehensive_report_summary(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.SUMMARY)
        report = self.gen.generate_comprehensive_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert "Executive Summary" in report
        assert "Test Results Overview" in report
        # summary level does NOT have coverage/quality sections
        assert "Coverage Analysis" not in report


# ---------------------------------------------------------------------------
# HTMLReportGenerator
# ---------------------------------------------------------------------------


class TestHTMLReportGenerator:
    def setup_method(self):
        self.gen = HTMLReportGenerator()
        self.test_results = [_make_test_result()]
        self.analysis_results = [_make_analysis_result()]
        self.coverage_report = _make_coverage_report()
        self.quality_metrics = _make_quality_metrics()
        self.config = ReportConfig(format=ReportFormat.HTML, level=ReportLevel.SUMMARY)

    def test_generate_html_report_structure(self):
        html = self.gen.generate_html_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            self.config,
        )
        assert "<!DOCTYPE html>" in html
        assert "<html lang=" in html
        assert "Integration Test Report" in html
        assert "bootstrap" in html

    def test_markdown_to_html_converts_headers(self):
        md = "# Hello\n## World\n### Sub"
        html = self.gen._markdown_to_html(md)
        assert "<h1>" in html

    def test_wrap_in_html_template(self):
        content = "<p>Test content</p>"
        html = self.gen._wrap_in_html_template(content, self.quality_metrics)
        assert "<!DOCTYPE html>" in html
        assert "Test content" in html
        assert "score-excellent" in html


# ---------------------------------------------------------------------------
# JSONReportGenerator
# ---------------------------------------------------------------------------


class TestJSONReportGenerator:
    def setup_method(self):
        self.gen = JSONReportGenerator()
        self.test_results = [
            _make_test_result(),
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="oops"),
        ]
        self.analysis_results = [_make_analysis_result()]
        self.coverage_report = _make_coverage_report(
            gaps=[_make_coverage_gap()],
            recommendations=["rec1"],
        )
        self.quality_metrics = _make_quality_metrics()
        self.config = ReportConfig(format=ReportFormat.JSON, level=ReportLevel.DETAILED)

    def test_generate_json_report_parses(self):
        raw = self.gen.generate_json_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            self.config,
        )
        data = json.loads(raw)
        assert data["summary"]["total_tests"] == 2
        assert data["summary"]["passed_tests"] == 1
        assert data["summary"]["failed_tests"] == 1
        assert data["summary"]["quality_score"] == 0.85
        assert data["metadata"]["format"] == "json"
        assert data["metadata"]["level"] == "detailed"

    def test_json_report_test_results_list(self):
        raw = self.gen.generate_json_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            self.config,
        )
        data = json.loads(raw)
        assert len(data["test_results"]) == 2
        assert data["test_results"][0]["status"] == "completed"
        assert data["test_results"][1]["status"] == "failed"

    def test_json_report_coverage(self):
        raw = self.gen.generate_json_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            self.config,
        )
        data = json.loads(raw)
        assert data["coverage_report"]["overall_coverage"] == 75.0
        assert len(data["coverage_report"]["coverage_gaps"]) == 1

    def test_json_report_quality_metrics(self):
        raw = self.gen.generate_json_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            self.config,
        )
        data = json.loads(raw)
        qm = data["quality_metrics"]
        assert qm["overall_score"] == 0.85
        assert qm["reliability_score"] == 0.9


# ---------------------------------------------------------------------------
# TestReportGenerator (coordinator)
# ---------------------------------------------------------------------------


class TestTestReportGenerator:
    def setup_method(self):
        self.gen = TestReportGenerator()
        self.test_results = [
            _make_test_result(),
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="fail"),
        ]
        self.analysis_results = [_make_analysis_result()]
        self.coverage_report = _make_coverage_report(
            gaps=[_make_coverage_gap()],
        )
        self.quality_metrics = _make_quality_metrics()

    def test_generate_report_markdown(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.SUMMARY)
        report = self.gen.generate_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert isinstance(report, TestReport)
        assert report.metadata.format == ReportFormat.MARKDOWN
        assert report.metadata.report_type == "integration_test_report"
        assert report.summary["total_tests"] == 2
        assert report.summary["passed_tests"] == 1
        assert "Integration Test Report" in report.content

    def test_generate_report_html(self):
        config = ReportConfig(format=ReportFormat.HTML, level=ReportLevel.SUMMARY)
        report = self.gen.generate_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert "<!DOCTYPE html>" in report.content
        assert report.metadata.format == ReportFormat.HTML

    def test_generate_report_json(self):
        config = ReportConfig(format=ReportFormat.JSON, level=ReportLevel.DETAILED)
        report = self.gen.generate_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        data = json.loads(report.content)
        assert data["summary"]["total_tests"] == 2
        assert report.metadata.format == ReportFormat.JSON

    def test_generate_report_default_format(self):
        config = ReportConfig(format=ReportFormat.PLAIN_TEXT, level=ReportLevel.SUMMARY)
        report = self.gen.generate_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        # Falls through to markdown default
        assert "Integration Test Report" in report.content

    def test_generate_report_metadata_fields(self):
        config = ReportConfig(format=ReportFormat.MARKDOWN, level=ReportLevel.SUMMARY)
        report = self.gen.generate_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            config,
        )
        assert report.metadata.report_id.startswith("test_report_")
        assert report.metadata.generation_time >= 0
        assert report.metadata.file_size_bytes > 0
        assert report.attachments == {}

    # -- generate_executive_summary -----------------------------------------

    def test_executive_summary_excellent_status(self):
        results = [_make_test_result() for _ in range(20)]
        qm = _make_quality_metrics(overall=0.9)
        cr = _make_coverage_report(overall=90.0)
        summary = self.gen.generate_executive_summary(results, qm, cr)
        assert "EXCELLENT" in summary
        assert "Executive Summary" in summary

    def test_executive_summary_good_status(self):
        results = [_make_test_result() for _ in range(9)]
        results.append(_make_test_result(status=TestExecutionStatus.FAILED, error_message="e"))
        qm = _make_quality_metrics(overall=0.7)
        cr = _make_coverage_report()
        summary = self.gen.generate_executive_summary(results, qm, cr)
        assert "GOOD" in summary

    def test_executive_summary_attention_needed(self):
        passed = [_make_test_result() for _ in range(7)]
        failed = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="e")
            for _ in range(3)
        ]
        qm = _make_quality_metrics(overall=0.5)
        cr = _make_coverage_report()
        summary = self.gen.generate_executive_summary(passed + failed, qm, cr)
        assert "ATTENTION NEEDED" in summary

    def test_executive_summary_critical_status(self):
        failed = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="e")
            for _ in range(10)
        ]
        qm = _make_quality_metrics(overall=0.3)
        cr = _make_coverage_report()
        summary = self.gen.generate_executive_summary(failed, qm, cr)
        assert "CRITICAL" in summary

    def test_executive_summary_zero_tests(self):
        qm = _make_quality_metrics()
        cr = _make_coverage_report()
        summary = self.gen.generate_executive_summary([], qm, cr)
        assert "0/0" in summary

    # -- _get_priority_actions -----------------------------------------------

    def test_priority_actions_with_issues(self):
        results = [
            _make_test_result(status=TestExecutionStatus.FAILED, error_message="err"),
            _make_test_result(execution_time=50.0),
        ]
        cr = _make_coverage_report(gaps=[_make_coverage_gap("high")])
        actions = self.gen._get_priority_actions(results, cr)
        assert "failed test" in actions
        assert "coverage gap" in actions
        assert "slow" in actions

    def test_priority_actions_no_issues(self):
        results = [_make_test_result()]
        cr = _make_coverage_report()
        actions = self.gen._get_priority_actions(results, cr)
        assert "Continue monitoring" in actions

    # -- _get_next_steps ----------------------------------------------------

    def test_next_steps_low_quality(self):
        qm = _make_quality_metrics(overall=0.5, performance=0.4)
        cr = _make_coverage_report(overall=60.0)
        steps = self.gen._get_next_steps(qm, cr)
        assert "quality" in steps.lower()
        assert "coverage" in steps.lower()
        assert "performance" in steps.lower()

    def test_next_steps_everything_good(self):
        qm = _make_quality_metrics(overall=0.9, performance=0.9)
        cr = _make_coverage_report(overall=90.0)
        steps = self.gen._get_next_steps(qm, cr)
        assert "Maintain" in steps


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def setup_method(self):
        self.test_results = [_make_test_result()]
        self.analysis_results = [_make_analysis_result()]
        self.coverage_report = _make_coverage_report()
        self.quality_metrics = _make_quality_metrics()

    def test_generate_test_report_default(self):
        report = generate_test_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
        )
        assert isinstance(report, TestReport)
        assert report.metadata.format == ReportFormat.MARKDOWN
        assert report.summary["total_tests"] == 1

    def test_generate_test_report_json(self):
        report = generate_test_report(
            self.test_results,
            self.analysis_results,
            self.coverage_report,
            self.quality_metrics,
            format=ReportFormat.JSON,
            level=ReportLevel.SUMMARY,
        )
        data = json.loads(report.content)
        assert "summary" in data

    def test_generate_executive_summary_func(self):
        result = generate_executive_summary(
            self.test_results,
            self.quality_metrics,
            self.coverage_report,
        )
        assert "Executive Summary" in result

    def test_export_test_results_json(self):
        content = export_test_results(self.test_results, format=ReportFormat.JSON)
        data = json.loads(content)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["status"] == "completed"

    def test_export_test_results_csv(self):
        results = [
            _make_test_result(),
            _make_test_result(
                status=TestExecutionStatus.FAILED,
                error_message="some,error",
            ),
        ]
        content = export_test_results(results, format=ReportFormat.CSV)
        lines = content.split("\n")
        assert lines[0] == "component,test_name,status,execution_time,error_message"
        assert len(lines) == 3
        # commas in error_message should be replaced with semicolons
        assert "some;error" in lines[2]

    def test_export_test_results_default_format(self):
        content = export_test_results(self.test_results, format=ReportFormat.PLAIN_TEXT)
        # Falls through to str(test_results)
        assert isinstance(content, str)

    def test_export_test_results_to_file(self, tmp_path):
        file_path = str(tmp_path / "export.json")
        content = export_test_results(
            self.test_results,
            format=ReportFormat.JSON,
            file_path=file_path,
        )
        with open(file_path) as f:
            written = f.read()
        assert written == content

    def test_export_test_results_csv_none_error(self):
        results = [_make_test_result(error_message=None)]
        content = export_test_results(results, format=ReportFormat.CSV)
        lines = content.split("\n")
        # error_message column should be empty string
        assert lines[1].endswith(",")
