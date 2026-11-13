"""Deep unit tests for ReportDataProcessor (report_data_processor.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from models.testing_models import QualityIssue, QualityResult
from processing.quality_analysis.report_configuration import QualityMetrics, ReportSettings
from processing.quality_analysis.report_data_processor import (
    ReportDataProcessor,
    create_data_processor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)


def _issue(
    tool: str = "ruff",
    severity: str = "warning",
    message: str = "unused import",
    file_path: str = "main.py",
    line_number: int | None = 1,
    column_number: int | None = None,
    rule_id: str | None = "F401",
    suggested_fix: str | None = None,
) -> QualityIssue:
    return QualityIssue(
        tool=tool,
        severity=severity,
        message=message,
        file_path=file_path,
        line_number=line_number,
        column_number=column_number,
        rule_id=rule_id,
        suggested_fix=suggested_fix,
    )


def _result(
    issues: list[QualityIssue] | None = None,
    tools_results: dict[str, dict[str, Any]] | None = None,
    overall_score: float = 85.0,
    execution_time: float = 1.5,
    language: str = "python",
    analysis_id: str = "a1",
) -> QualityResult:
    return QualityResult(
        analysis_id=analysis_id,
        success=True,
        overall_score=overall_score,
        total_issues=len(issues) if issues else 0,
        issues=issues or [],
        tools_results=tools_results or {},
        execution_time=execution_time,
        created_at=_NOW,
        language=language,
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateDataProcessor:
    def test_returns_instance(self) -> None:
        proc = create_data_processor()
        assert isinstance(proc, ReportDataProcessor)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_empty_results(self, processor: ReportDataProcessor) -> None:
        metrics = await processor.calculate_metrics([])
        assert metrics.total_files_analyzed == 0
        assert metrics.total_issues == 0
        assert metrics.overall_score == 0.0
        assert metrics.critical_issues == 0
        assert metrics.error_issues == 0
        assert metrics.warning_issues == 0
        assert metrics.info_issues == 0
        assert metrics.successful_tools == 0
        assert metrics.failed_tools == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    @pytest.mark.asyncio
    async def test_single_result_no_issues(self, processor: ReportDataProcessor) -> None:
        r = _result(issues=[], overall_score=90.0, execution_time=2.0)
        metrics = await processor.calculate_metrics([r])
        assert metrics.total_files_analyzed == 1
        assert metrics.total_issues == 0
        assert metrics.overall_score == 90.0
        assert metrics.total_execution_time == 2.0
        assert metrics.average_execution_time == 2.0

    @pytest.mark.asyncio
    async def test_severity_counts(self, processor: ReportDataProcessor) -> None:
        issues = [
            _issue(severity="critical"),
            _issue(severity="critical"),
            _issue(severity="error"),
            _issue(severity="warning"),
            _issue(severity="info"),
            _issue(severity="info"),
        ]
        r = _result(issues=issues, overall_score=50.0)
        metrics = await processor.calculate_metrics([r])
        assert metrics.critical_issues == 2
        assert metrics.error_issues == 1
        assert metrics.warning_issues == 1
        assert metrics.info_issues == 2
        assert metrics.total_issues == 6

    @pytest.mark.asyncio
    async def test_unknown_severity_ignored(self, processor: ReportDataProcessor) -> None:
        """Severities not in the known set are silently skipped."""
        issues = [_issue(severity="notice")]
        r = _result(issues=issues)
        metrics = await processor.calculate_metrics([r])
        assert metrics.critical_issues == 0
        assert metrics.error_issues == 0
        assert metrics.warning_issues == 0
        assert metrics.info_issues == 0
        assert metrics.total_issues == 1  # still counted

    @pytest.mark.asyncio
    async def test_tools_success_and_failure(self, processor: ReportDataProcessor) -> None:
        tools = {
            "ruff": {"success": True},
            "bandit": {"success": False},
            "ty": {"success": True},
        }
        r = _result(tools_results=tools)
        metrics = await processor.calculate_metrics([r])
        assert metrics.successful_tools == 2
        assert metrics.failed_tools == 1
        assert set(metrics.tools_used) == {"ruff", "bandit", "ty"}

    @pytest.mark.asyncio
    async def test_tool_default_success_true(self, processor: ReportDataProcessor) -> None:
        """Tool entries missing 'success' key default to True."""
        tools = {"eslint": {"issues": []}}
        r = _result(tools_results=tools)
        metrics = await processor.calculate_metrics([r])
        assert metrics.successful_tools == 1
        assert metrics.failed_tools == 0

    @pytest.mark.asyncio
    async def test_language_distribution(self, processor: ReportDataProcessor) -> None:
        results = [
            _result(language="python", analysis_id="a1"),
            _result(language="python", analysis_id="a2"),
            _result(language="javascript", analysis_id="a3"),
        ]
        metrics = await processor.calculate_metrics(results)
        assert metrics.languages_analyzed == {"python": 2, "javascript": 1}

    @pytest.mark.asyncio
    async def test_overall_score_averaged(self, processor: ReportDataProcessor) -> None:
        results = [
            _result(overall_score=80.0, analysis_id="a1"),
            _result(overall_score=60.0, analysis_id="a2"),
        ]
        metrics = await processor.calculate_metrics(results)
        assert metrics.overall_score == 70.0

    @pytest.mark.asyncio
    async def test_execution_time_aggregation(self, processor: ReportDataProcessor) -> None:
        results = [
            _result(execution_time=3.0, analysis_id="a1"),
            _result(execution_time=7.0, analysis_id="a2"),
        ]
        metrics = await processor.calculate_metrics(results)
        assert metrics.total_execution_time == 10.0
        assert metrics.average_execution_time == 5.0


# ---------------------------------------------------------------------------
# generate_issue_summary
# ---------------------------------------------------------------------------


class TestGenerateIssueSummary:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.fixture
    def default_settings(self) -> ReportSettings:
        return ReportSettings()

    @pytest.mark.asyncio
    async def test_empty_results(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        summary = await processor.generate_issue_summary([], default_settings)
        assert summary["issues_by_tool"] == {}
        assert summary["issues_by_file"] == {}
        assert summary["issues_by_rule"] == {}
        assert summary["most_problematic_files"] == []
        assert summary["most_common_rules"] == []

    @pytest.mark.asyncio
    async def test_groups_by_tool_file_rule(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        issues = [
            _issue(tool="ruff", file_path="a.py", rule_id="F401"),
            _issue(tool="ruff", file_path="a.py", rule_id="F401"),
            _issue(tool="bandit", file_path="b.py", rule_id="B101"),
        ]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], default_settings)
        assert summary["issues_by_tool"] == {"ruff": 2, "bandit": 1}
        assert summary["issues_by_file"] == {"a.py": 2, "b.py": 1}
        assert summary["issues_by_rule"] == {"F401": 2, "B101": 1}

    @pytest.mark.asyncio
    async def test_severity_filter_excludes(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings(severity_filter=["critical", "error"])
        issues = [
            _issue(severity="critical"),
            _issue(severity="warning"),
            _issue(severity="info"),
        ]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], settings)
        assert summary["issues_by_tool"]["ruff"] == 1  # only critical

    @pytest.mark.asyncio
    async def test_rule_id_none_excluded(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        issues = [_issue(rule_id=None)]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], default_settings)
        assert summary["issues_by_rule"] == {}

    @pytest.mark.asyncio
    async def test_most_problematic_files_limited_to_10(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        issues = [_issue(file_path=f"file_{i}.py") for i in range(15)]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], default_settings)
        assert len(summary["most_problematic_files"]) == 10

    @pytest.mark.asyncio
    async def test_most_problematic_files_sorted_desc(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        issues = [
            _issue(file_path="a.py"),
            _issue(file_path="b.py"),
            _issue(file_path="b.py", rule_id="X"),
            _issue(file_path="b.py", rule_id="Y"),
        ]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], default_settings)
        top = summary["most_problematic_files"]
        assert top[0][0] == "b.py"
        assert len(top[0][1]) == 3

    @pytest.mark.asyncio
    async def test_most_common_rules_limited_to_10(
        self, processor: ReportDataProcessor, default_settings: ReportSettings
    ) -> None:
        issues = [_issue(rule_id=f"R{i}") for i in range(15)]
        r = _result(issues=issues)
        summary = await processor.generate_issue_summary([r], default_settings)
        assert len(summary["most_common_rules"]) == 10


# ---------------------------------------------------------------------------
# generate_tool_performance
# ---------------------------------------------------------------------------


class TestGenerateToolPerformance:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_empty_results(self, processor: ReportDataProcessor) -> None:
        stats = await processor.generate_tool_performance([])
        assert stats == {}

    @pytest.mark.asyncio
    async def test_single_tool_single_run(self, processor: ReportDataProcessor) -> None:
        tools = {"ruff": {"success": True, "execution_time": 2.0, "issues": ["a", "b"]}}
        r = _result(tools_results=tools)
        stats = await processor.generate_tool_performance([r])
        assert stats["ruff"]["total_runs"] == 1
        assert stats["ruff"]["successful_runs"] == 1
        assert stats["ruff"]["total_execution_time"] == 2.0
        assert stats["ruff"]["issues_found"] == 2
        assert stats["ruff"]["success_rate"] == 100.0
        assert stats["ruff"]["average_execution_time"] == 2.0
        assert stats["ruff"]["average_issues_per_run"] == 2.0

    @pytest.mark.asyncio
    async def test_failed_tool(self, processor: ReportDataProcessor) -> None:
        tools = {"ty": {"success": False, "execution_time": 0.5, "issues": []}}
        r = _result(tools_results=tools)
        stats = await processor.generate_tool_performance([r])
        assert stats["ty"]["successful_runs"] == 0
        assert stats["ty"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_runs_same_tool(self, processor: ReportDataProcessor) -> None:
        results = [
            _result(
                tools_results={"ruff": {"success": True, "execution_time": 1.0, "issues": ["x"]}},
                analysis_id="a1",
            ),
            _result(
                tools_results={
                    "ruff": {"success": False, "execution_time": 3.0, "issues": ["x", "y"]}
                },
                analysis_id="a2",
            ),
        ]
        stats = await processor.generate_tool_performance(results)
        assert stats["ruff"]["total_runs"] == 2
        assert stats["ruff"]["successful_runs"] == 1
        assert stats["ruff"]["success_rate"] == 50.0
        assert stats["ruff"]["total_execution_time"] == 4.0
        assert stats["ruff"]["average_execution_time"] == 2.0
        assert stats["ruff"]["issues_found"] == 3
        assert stats["ruff"]["average_issues_per_run"] == 1.5

    @pytest.mark.asyncio
    async def test_missing_keys_default(self, processor: ReportDataProcessor) -> None:
        """execution_time and issues default to 0/[] when not present."""
        tools = {"tool": {"success": True}}
        r = _result(tools_results=tools)
        stats = await processor.generate_tool_performance([r])
        assert stats["tool"]["total_execution_time"] == 0.0
        assert stats["tool"]["issues_found"] == 0


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    def _metrics(self, **overrides: Any) -> QualityMetrics:
        defaults: dict[str, Any] = {
            "total_files_analyzed": 1,
            "total_issues": 10,
            "overall_score": 80.0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 5,
            "info_issues": 5,
            "successful_tools": 3,
            "failed_tools": 0,
            "total_execution_time": 10.0,
            "average_execution_time": 3.3,
            "languages_analyzed": {},
        }
        defaults.update(overrides)
        return QualityMetrics(**defaults)

    @pytest.mark.asyncio
    async def test_good_quality_no_issues(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(overall_score=90.0, total_issues=0, warning_issues=0, info_issues=0)
        recs = await processor.generate_recommendations(m, [])
        assert len(recs) == 1
        assert "looks good" in recs[0]

    @pytest.mark.asyncio
    async def test_low_score_recommendation(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(overall_score=50.0)
        recs = await processor.generate_recommendations(m, [])
        assert any("below acceptable" in r for r in recs)

    @pytest.mark.asyncio
    async def test_critical_issues_recommendation(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(critical_issues=3)
        recs = await processor.generate_recommendations(m, [])
        assert any("3 critical issues" in r for r in recs)

    @pytest.mark.asyncio
    async def test_high_error_proportion(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(total_issues=10, error_issues=5)
        recs = await processor.generate_recommendations(m, [])
        assert any("error-level issues" in r for r in recs)

    @pytest.mark.asyncio
    async def test_failed_tools_recommendation(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(failed_tools=2)
        recs = await processor.generate_recommendations(m, [])
        assert any("2 tool executions failed" in r for r in recs)

    @pytest.mark.asyncio
    async def test_high_execution_time(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(average_execution_time=120.0)
        recs = await processor.generate_recommendations(m, [])
        assert any("execution time is high" in r for r in recs)

    @pytest.mark.asyncio
    async def test_bandit_security_issues(self, processor: ReportDataProcessor) -> None:
        m = self._metrics(languages_analyzed={"python": 1})
        result = _result(
            tools_results={"bandit": {"success": True}},
            issues=[_issue(tool="bandit", severity="warning")],
        )
        recs = await processor.generate_recommendations(m, [result])
        assert any("Security issues" in r for r in recs)

    @pytest.mark.asyncio
    async def test_bandit_present_but_no_bandit_issues(
        self, processor: ReportDataProcessor
    ) -> None:
        """bandit tool present in results, but no issues from bandit -> no security rec."""
        m = self._metrics(languages_analyzed={"python": 1})
        result = _result(
            tools_results={"bandit": {"success": True}},
            issues=[_issue(tool="ruff", severity="warning")],
        )
        recs = await processor.generate_recommendations(m, [result])
        assert not any("Security issues" in r for r in recs)

    @pytest.mark.asyncio
    async def test_non_python_language_no_bandit_check(
        self, processor: ReportDataProcessor
    ) -> None:
        m = self._metrics(languages_analyzed={"javascript": 1})
        recs = await processor.generate_recommendations(m, [])
        # Should not crash and should not reference bandit
        assert not any("Security issues" in r for r in recs)


# ---------------------------------------------------------------------------
# filter_and_format_issues
# ---------------------------------------------------------------------------


class TestFilterAndFormatIssues:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_basic_formatting(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings()
        issue = _issue(
            tool="ruff",
            severity="warning",
            message="unused",
            file_path="x.py",
            line_number=10,
            column_number=5,
            rule_id="F401",
            suggested_fix="remove import",
        )
        r = _result(issues=[issue])
        formatted = await processor.filter_and_format_issues(r, settings)
        assert len(formatted) == 1
        f = formatted[0]
        assert f["tool"] == "ruff"
        assert f["severity"] == "warning"
        assert f["message"] == "unused"
        assert f["file_path"] == "x.py"
        assert f["line_number"] == 10
        assert f["column_number"] == 5
        assert f["rule_id"] == "F401"
        assert f["suggested_fix"] == "remove import"

    @pytest.mark.asyncio
    async def test_severity_filter(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings(severity_filter=["error"])
        issues = [
            _issue(severity="error"),
            _issue(severity="warning"),
        ]
        r = _result(issues=issues)
        formatted = await processor.filter_and_format_issues(r, settings)
        assert len(formatted) == 1
        assert formatted[0]["severity"] == "error"

    @pytest.mark.asyncio
    async def test_fix_suggestions_excluded(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings(include_fix_suggestions=False)
        issue = _issue(suggested_fix="do something")
        r = _result(issues=[issue])
        formatted = await processor.filter_and_format_issues(r, settings)
        assert formatted[0]["suggested_fix"] is None

    @pytest.mark.asyncio
    async def test_max_issues_per_file(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings(max_issues_per_file=3)
        issues = [_issue(message=f"issue {i}") for i in range(10)]
        r = _result(issues=issues)
        formatted = await processor.filter_and_format_issues(r, settings)
        assert len(formatted) == 3

    @pytest.mark.asyncio
    async def test_max_issues_none_means_no_limit(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings(max_issues_per_file=None)
        issues = [_issue(message=f"issue {i}") for i in range(20)]
        r = _result(issues=issues)
        formatted = await processor.filter_and_format_issues(r, settings)
        assert len(formatted) == 20

    @pytest.mark.asyncio
    async def test_empty_issues(self, processor: ReportDataProcessor) -> None:
        settings = ReportSettings()
        r = _result(issues=[])
        formatted = await processor.filter_and_format_issues(r, settings)
        assert formatted == []


# ---------------------------------------------------------------------------
# calculate_comparative_metrics
# ---------------------------------------------------------------------------


class TestCalculateComparativeMetrics:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_score_improvement(self, processor: ReportDataProcessor) -> None:
        current = [_result(overall_score=90.0, analysis_id="c1")]
        baseline = [_result(overall_score=70.0, analysis_id="b1")]
        comp = await processor.calculate_comparative_metrics(current, baseline)
        assert comp["score_change"] == 20.0
        assert comp["score_change_percentage"] == pytest.approx(28.57, abs=0.01)
        assert comp["issue_count_change"] == 0

    @pytest.mark.asyncio
    async def test_score_regression(self, processor: ReportDataProcessor) -> None:
        current = [_result(overall_score=50.0, analysis_id="c1")]
        baseline = [_result(overall_score=80.0, analysis_id="b1")]
        comp = await processor.calculate_comparative_metrics(current, baseline)
        assert comp["score_change"] == -30.0

    @pytest.mark.asyncio
    async def test_issue_count_change(self, processor: ReportDataProcessor) -> None:
        current = [_result(issues=[_issue(), _issue(rule_id="X")], analysis_id="c1")]
        baseline = [_result(issues=[_issue()], analysis_id="b1")]
        comp = await processor.calculate_comparative_metrics(current, baseline)
        assert comp["issue_count_change"] == 1
        assert comp["issue_change_percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_baseline_zero_score(self, processor: ReportDataProcessor) -> None:
        current = [_result(overall_score=50.0, analysis_id="c1")]
        baseline: list[QualityResult] = []
        comp = await processor.calculate_comparative_metrics(current, baseline)
        assert comp["score_change_percentage"] == 0  # division guard

    @pytest.mark.asyncio
    async def test_baseline_zero_issues(self, processor: ReportDataProcessor) -> None:
        current = [_result(issues=[], analysis_id="c1")]
        baseline: list[QualityResult] = []
        comp = await processor.calculate_comparative_metrics(current, baseline)
        assert comp["issue_change_percentage"] == 0

    @pytest.mark.asyncio
    async def test_both_empty(self, processor: ReportDataProcessor) -> None:
        comp = await processor.calculate_comparative_metrics([], [])
        assert comp["score_change"] == 0.0
        assert comp["issue_count_change"] == 0


# ---------------------------------------------------------------------------
# identify_improvements
# ---------------------------------------------------------------------------


class TestIdentifyImprovements:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_score_improvement(self, processor: ReportDataProcessor) -> None:
        current = [_result(overall_score=90.0, analysis_id="c1")]
        baseline = [_result(overall_score=70.0, analysis_id="b1")]
        imps = await processor.identify_improvements(current, baseline)
        assert any("increased by 20.0" in i for i in imps)

    @pytest.mark.asyncio
    async def test_critical_reduction(self, processor: ReportDataProcessor) -> None:
        current = [_result(issues=[_issue(severity="critical")], analysis_id="c1")]
        baseline = [
            _result(
                issues=[_issue(severity="critical"), _issue(severity="critical", rule_id="X")],
                analysis_id="b1",
            )
        ]
        imps = await processor.identify_improvements(current, baseline)
        assert any("Critical issues reduced by 1" in i for i in imps)

    @pytest.mark.asyncio
    async def test_error_reduction(self, processor: ReportDataProcessor) -> None:
        current = [_result(issues=[], analysis_id="c1")]
        baseline = [_result(issues=[_issue(severity="error")], analysis_id="b1")]
        imps = await processor.identify_improvements(current, baseline)
        assert any("Error issues reduced by 1" in i for i in imps)

    @pytest.mark.asyncio
    async def test_no_improvements(self, processor: ReportDataProcessor) -> None:
        r = _result(analysis_id="a1")
        imps = await processor.identify_improvements([r], [r])
        assert imps == []


# ---------------------------------------------------------------------------
# identify_regressions
# ---------------------------------------------------------------------------


class TestIdentifyRegressions:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    @pytest.mark.asyncio
    async def test_score_regression(self, processor: ReportDataProcessor) -> None:
        current = [_result(overall_score=50.0, analysis_id="c1")]
        baseline = [_result(overall_score=80.0, analysis_id="b1")]
        regs = await processor.identify_regressions(current, baseline)
        assert any("decreased by 30.0" in r for r in regs)

    @pytest.mark.asyncio
    async def test_critical_increase(self, processor: ReportDataProcessor) -> None:
        current = [
            _result(
                issues=[_issue(severity="critical"), _issue(severity="critical", rule_id="X")],
                analysis_id="c1",
            )
        ]
        baseline = [_result(issues=[_issue(severity="critical")], analysis_id="b1")]
        regs = await processor.identify_regressions(current, baseline)
        assert any("Critical issues increased by 1" in r for r in regs)

    @pytest.mark.asyncio
    async def test_error_increase(self, processor: ReportDataProcessor) -> None:
        current = [_result(issues=[_issue(severity="error")], analysis_id="c1")]
        baseline = [_result(issues=[], analysis_id="b1")]
        regs = await processor.identify_regressions(current, baseline)
        assert any("Error issues increased by 1" in r for r in regs)

    @pytest.mark.asyncio
    async def test_no_regressions(self, processor: ReportDataProcessor) -> None:
        r = _result(analysis_id="a1")
        regs = await processor.identify_regressions([r], [r])
        assert regs == []


# ---------------------------------------------------------------------------
# safe_float_convert
# ---------------------------------------------------------------------------


class TestSafeFloatConvert:
    @pytest.fixture
    def processor(self) -> ReportDataProcessor:
        return ReportDataProcessor()

    def test_none(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert(None) == 0.0

    def test_empty_string(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert("") == 0.0

    def test_valid_int(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert(42) == 42.0

    def test_valid_float(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert(3.14) == 3.14

    def test_valid_string_number(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert("7.5") == 7.5

    def test_invalid_string(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert("abc") == 0.0

    def test_list_value(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert([1, 2]) == 0.0

    def test_dict_value(self, processor: ReportDataProcessor) -> None:
        assert processor.safe_float_convert({"a": 1}) == 0.0
