"""Round 4 tests for QualityReportGenerator (79 uncov lines)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from models.testing_models import QualityIssue, QualityResult
from processing.quality_analysis.report_generator import (
    QualityReportGenerator,
    create_quality_report_generator,
)


def _make_result(
    analysis_id="a1",
    success=True,
    language="python",
    score=85.0,
    issues=None,
    tools_results=None,
    exec_time=1.5,
):
    return QualityResult(
        analysis_id=analysis_id,
        success=success,
        language=language,
        overall_score=score,
        total_issues=len(issues) if issues else 0,
        issues=issues or [],
        tools_results=tools_results or {},
        execution_time=exec_time,
        created_at=datetime.now(UTC),
    )


def _make_issue(
    severity="warning", tool="ruff", rule_id="E001", msg="test issue", file="test.py", line=1
):
    return QualityIssue(
        severity=severity,
        tool=tool,
        rule_id=rule_id,
        message=msg,
        file_path=file,
        line_number=line,
    )


class TestQualityReportGeneratorInit:
    def test_init_defaults(self):
        gen = QualityReportGenerator()
        assert gen.cache_manager is None
        assert gen.config_manager is not None
        assert gen.data_processor is not None
        assert gen.template_manager is not None
        assert gen.chart_generator is not None
        assert gen.output_handler is not None

    def test_init_with_output_dir(self, tmp_path):
        gen = QualityReportGenerator(output_directory=str(tmp_path))
        assert gen.output_handler is not None


class TestQualityReportGeneratorMethods:
    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    def test_get_available_themes(self, gen):
        themes = gen.get_available_themes()
        assert "default" in themes
        assert "dark" in themes
        assert "corporate" in themes

    def test_get_available_styles(self, gen):
        styles = gen.get_available_styles()
        assert "compact" in styles
        assert "detailed" in styles
        assert "minimal" in styles

    def test_get_chart_types(self, gen):
        chart_types = gen.get_chart_types()
        assert "severity_distribution" in chart_types
        assert "quality_score_distribution" in chart_types
        assert "tools_performance" in chart_types
        assert "language_breakdown" in chart_types
        assert "quality_trends" in chart_types
        assert "issue_trends" in chart_types
        assert "execution_time" in chart_types
        assert len(chart_types) == 7

    def test_get_supported_formats(self, gen):
        formats = gen.get_supported_formats()
        assert "html" in formats
        assert "json" in formats
        assert "csv" in formats
        assert "dashboard" in formats
        assert "comparative" in formats
        assert len(formats) == 5


class TestTopIssuesTable:
    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_top_issues_table_empty(self, gen):
        result = await gen._generate_top_issues_table([_make_result()])
        assert result == []

    @pytest.mark.asyncio
    async def test_top_issues_table_with_issues(self, gen):
        issues = [
            _make_issue(severity="critical", msg="critical issue"),
            _make_issue(severity="warning", msg="warning issue"),
            _make_issue(severity="info", msg="info issue"),
            _make_issue(severity="error", msg="error issue"),
        ]
        result = await gen._generate_top_issues_table([_make_result(issues=issues)])
        assert len(result) == 4
        # Critical should come first
        assert result[0]["severity"] == "critical"
        assert result[1]["severity"] == "error"

    @pytest.mark.asyncio
    async def test_top_issues_truncates_message(self, gen):
        long_msg = "x" * 200
        issues = [_make_issue(msg=long_msg)]
        result = await gen._generate_top_issues_table([_make_result(issues=issues)])
        assert len(result) == 1
        assert result[0]["message"].endswith("...")
        assert len(result[0]["message"]) <= 103


class TestFileRankingTable:
    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_file_ranking_table_empty(self, gen):
        result = await gen._generate_file_ranking_table([])
        assert result == []

    @pytest.mark.asyncio
    async def test_file_ranking_table(self, gen):
        results = [
            _make_result(analysis_id="a1", score=90.0, issues=[_make_issue()]),
            _make_result(analysis_id="a2", score=70.0, issues=[_make_issue(severity="critical")]),
        ]
        table = await gen._generate_file_ranking_table(results)
        assert len(table) == 2
        # Higher score should come first
        assert table[0]["score"] >= table[1]["score"]


class TestToolStatisticsTable:
    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_tool_statistics_table(self, gen):
        results = [
            _make_result(
                tools_results={"ruff": {"success": True, "issues": [], "execution_time": 0.5}}
            )
        ]
        table = await gen._generate_tool_statistics_table(results)
        assert len(table) == 1
        assert table[0]["tool"] == "ruff"
        assert table[0]["runs"] == 1
        assert table[0]["success_rate"] == "100.0%"
        assert table[0]["avg_time"] == "0.50s"
        assert table[0]["issues_found"] == 0
        assert table[0]["avg_issues"] == "0.0"


class TestDashboardTables:
    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_generate_dashboard_tables(self, gen):
        results = [_make_result()]
        tables = await gen._generate_dashboard_tables(results)
        assert "top_issues" in tables
        assert "file_quality_ranking" in tables
        assert "tool_statistics" in tables


class TestLegacyMethods:
    """Test deprecated methods that delegate to modular components."""

    @pytest.fixture
    def gen(self):
        return QualityReportGenerator()

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, gen):
        issues = [
            _make_issue(severity="critical"),
            _make_issue(severity="error"),
            _make_issue(severity="warning"),
        ]
        results = [_make_result(score=85.0, issues=issues, exec_time=1.5)]
        metrics = await gen._calculate_metrics(results)
        assert metrics.total_files_analyzed == 1
        assert metrics.total_issues == 3
        assert metrics.overall_score == 85.0
        assert metrics.critical_issues == 1
        assert metrics.error_issues == 1
        assert metrics.warning_issues == 1
        assert metrics.info_issues == 0
        assert metrics.total_execution_time == 1.5
        assert metrics.average_execution_time == 1.5

    @pytest.mark.asyncio
    async def test_calculate_metrics_multiple_results(self, gen):
        results = [
            _make_result(score=80.0, exec_time=2.0),
            _make_result(score=90.0, exec_time=1.0),
        ]
        metrics = await gen._calculate_metrics(results)
        assert metrics.total_files_analyzed == 2
        assert metrics.overall_score == 85.0
        assert metrics.total_execution_time == 3.0
        assert metrics.average_execution_time == 1.5

    @pytest.mark.asyncio
    async def test_generate_issue_summary(self, gen):
        from processing.quality_analysis.report_configuration import create_default_settings

        issues = [
            _make_issue(severity="warning", tool="ruff", rule_id="E001", file="a.py"),
            _make_issue(severity="error", tool="ty", rule_id="type-error", file="b.py"),
            _make_issue(severity="warning", tool="ruff", rule_id="E001", file="a.py"),
        ]
        results = [_make_result(issues=issues)]
        settings = create_default_settings()
        summary = await gen._generate_issue_summary(results, settings)
        assert summary["issues_by_tool"]["ruff"] == 2
        assert summary["issues_by_tool"]["ty"] == 1
        assert summary["issues_by_file"]["a.py"] == 2
        assert summary["issues_by_file"]["b.py"] == 1
        assert summary["issues_by_rule"]["E001"] == 2
        assert summary["issues_by_rule"]["type-error"] == 1
        assert len(summary["most_problematic_files"]) >= 1
        # a.py has 2 issues, so it should be the most problematic
        assert summary["most_problematic_files"][0][0] == "a.py"

    @pytest.mark.asyncio
    async def test_generate_tool_performance(self, gen):
        results = [
            _make_result(
                tools_results={
                    "ruff": {"success": True, "issues": [{"msg": "x"}], "execution_time": 0.5},
                    "ty": {"success": False, "issues": [], "execution_time": 1.0},
                }
            )
        ]
        perf = await gen._generate_tool_performance(results)
        assert "ruff" in perf
        assert "ty" in perf
        assert perf["ruff"]["total_runs"] == 1
        assert perf["ruff"]["successful_runs"] == 1
        assert perf["ruff"]["success_rate"] == 100.0
        assert perf["ruff"]["issues_found"] == 1
        assert perf["ruff"]["average_execution_time"] == 0.5
        assert perf["ty"]["successful_runs"] == 0
        assert perf["ty"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_recommendations_good_code(self, gen):
        """High-quality code with no issues should get positive recommendation."""
        results = [_make_result(score=95.0)]
        metrics = await gen._calculate_metrics(results)
        recs = await gen._generate_recommendations(metrics, results)
        assert len(recs) >= 1
        # No critical issues, high score -> should get the "looks good" message
        assert any("looks good" in r for r in recs)

    @pytest.mark.asyncio
    async def test_generate_recommendations_critical_issues(self, gen):
        """Code with critical issues should get critical-issue recommendations."""
        issues = [_make_issue(severity="critical")]
        results = [_make_result(score=50.0, issues=issues)]
        metrics = await gen._calculate_metrics(results)
        recs = await gen._generate_recommendations(metrics, results)
        assert any("critical" in r.lower() for r in recs)
        assert any("below acceptable" in r.lower() or "threshold" in r.lower() for r in recs)

    @pytest.mark.asyncio
    async def test_generate_severity_chart(self, gen):
        issues = [
            _make_issue(severity="critical"),
            _make_issue(severity="warning"),
            _make_issue(severity="warning"),
        ]
        results = [_make_result(issues=issues)]
        chart = await gen._generate_severity_chart(results)
        assert chart["type"] == "pie"
        assert chart["title"] == "Issue Distribution by Severity"
        labels = chart["data"]["labels"]
        values = chart["data"]["values"]
        # The labels are always critical/error/warning/info in that order
        assert labels == ["critical", "error", "warning", "info"]
        assert values[labels.index("critical")] == 1
        assert values[labels.index("warning")] == 2
        assert values[labels.index("error")] == 0
        assert values[labels.index("info")] == 0

    @pytest.mark.asyncio
    async def test_generate_score_distribution_chart(self, gen):
        results = [_make_result(score=85.0)]
        chart = await gen._generate_score_distribution_chart(results)
        assert chart["type"] == "bar"
        assert chart["title"] == "Quality Score Distribution"
        labels = chart["data"]["labels"]
        data_values = chart["data"]["datasets"][0]["data"]
        # Score 85.0 falls in 80-89 range
        idx_80_89 = labels.index("80-89")
        assert data_values[idx_80_89] == 1
        # All other ranges should be 0
        for i, val in enumerate(data_values):
            if i != idx_80_89:
                assert val == 0

    @pytest.mark.asyncio
    async def test_generate_tools_performance_chart(self, gen):
        results = [
            _make_result(
                tools_results={"ruff": {"success": True, "issues": [], "execution_time": 0.5}}
            )
        ]
        chart = await gen._generate_tools_performance_chart(results)
        assert chart["type"] == "bar"
        assert chart["title"] == "Tool Success Rates"
        assert "ruff" in chart["data"]["labels"]
        assert chart["data"]["datasets"][0]["data"][0] == 100.0

    @pytest.mark.asyncio
    async def test_generate_language_breakdown_chart(self, gen):
        results = [
            _make_result(language="python"),
            _make_result(language="python"),
            _make_result(language="javascript"),
        ]
        chart = await gen._generate_language_breakdown_chart(results)
        assert chart["type"] == "pie"
        assert chart["title"] == "Analysis by Language"
        labels = chart["data"]["labels"]
        values = chart["data"]["datasets"][0]["data"]
        python_idx = labels.index("python")
        js_idx = labels.index("javascript")
        assert values[python_idx] == 2
        assert values[js_idx] == 1

    @pytest.mark.asyncio
    async def test_generate_trends_chart(self, gen):
        results = [_make_result(score=85.0)]
        chart = await gen._generate_trends_chart(results)
        assert chart["type"] == "line"
        assert chart["title"] == "Quality Score Trend"
        assert chart["data"]["datasets"][0]["data"] == [85.0]
        assert len(chart["data"]["labels"]) == 1

    @pytest.mark.asyncio
    async def test_identify_improvements(self, gen):
        current = [_make_result(score=90.0)]
        baseline = [_make_result(score=70.0)]
        improvements = await gen._identify_improvements(current, baseline)
        assert len(improvements) >= 1
        assert any("increased" in imp.lower() for imp in improvements)
        assert any("20.0" in imp for imp in improvements)

    @pytest.mark.asyncio
    async def test_identify_regressions(self, gen):
        current = [_make_result(score=70.0)]
        baseline = [_make_result(score=90.0)]
        regressions = await gen._identify_regressions(current, baseline)
        assert len(regressions) >= 1
        assert any("decreased" in reg.lower() for reg in regressions)
        assert any("20.0" in reg for reg in regressions)

    @pytest.mark.asyncio
    async def test_identify_improvements_no_change(self, gen):
        """No improvement when scores are equal."""
        current = [_make_result(score=80.0)]
        baseline = [_make_result(score=80.0)]
        improvements = await gen._identify_improvements(current, baseline)
        assert improvements == []

    @pytest.mark.asyncio
    async def test_identify_regressions_no_change(self, gen):
        """No regression when scores are equal."""
        current = [_make_result(score=80.0)]
        baseline = [_make_result(score=80.0)]
        regressions = await gen._identify_regressions(current, baseline)
        assert regressions == []


class TestFactoryFunction:
    def test_create_report_generator(self):
        gen = create_quality_report_generator()
        assert isinstance(gen, QualityReportGenerator)

    def test_create_report_generator_with_dir(self, tmp_path):
        gen = create_quality_report_generator(output_directory=str(tmp_path))
        assert isinstance(gen, QualityReportGenerator)
