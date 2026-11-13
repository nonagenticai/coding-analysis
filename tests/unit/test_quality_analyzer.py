"""Unit tests for processing.architecture.quality_analyzer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from models.architecture_models import ArchitectureQualityIssue, QualityReport
from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    tool: str = "ruff",
    severity: str = "medium",
    rule_id: str = "E001",
    message: str = "test issue",
) -> ArchitectureQualityIssue:
    return ArchitectureQualityIssue(
        id=f"{tool}-{rule_id}-0",
        severity=severity,
        message=message,
        rule_id=rule_id,
        rule_name=rule_id,
        file_path="src/app.py",
        line_number=1,
        column_number=None,
        tool=tool,
        category="style",
        description=message,
    )


def _make_quality_report(**overrides) -> QualityReport:
    defaults = dict(
        job_id="test-job-1",
        codebase_id="test-codebase",
        generated_at="2026-01-01T00:00:00Z",
        overall_score=85.0,
        grade="B",
        issues=[],
        summary={"total_issues": 0, "total_files": 10},
        recommendations=["Keep up the good work"],
        execution_time=1.5,
    )
    defaults.update(overrides)
    return QualityReport(**defaults)  # type: ignore[invalid-argument-type]


@pytest.fixture(autouse=True)
def _no_container_materialization():
    """By default, materialization yields nothing, so ``analyze`` falls back to the
    local repo path and the existing local-fs assertions hold (and no real network
    call is made to coding-containers). Tests that exercise the container path
    override this with their own inner ``patch`` of ``materialize_from_container``."""
    with patch(
        "processing.architecture.quality_analyzer.materialize_from_container",
        AsyncMock(return_value=None),
    ):
        yield


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_all_subcomponents(self):
        """ArchitectureQualityAnalyzer should instantiate all five sub-components."""
        with (
            patch("processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"),
            patch("processing.architecture.quality_analyzer.CodeQualityMetrics"),
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch("processing.architecture.quality_analyzer.QualityReporting"),
            patch("processing.architecture.quality_analyzer.ToolExecution"),
        ):
            analyzer = ArchitectureQualityAnalyzer()

        assert analyzer.pattern_analyzer is not None
        assert analyzer.metrics_calculator is not None
        assert analyzer.complexity_analyzer is not None
        assert analyzer.performance_analyzer is not None
        assert analyzer.reporter is not None
        assert analyzer.tool_executor is not None

    def test_subcomponents_are_correct_types(self):
        """Each sub-component should be instantiated by the correct class."""
        # The sibling modules are stubs (MagicMock), so we verify by patching
        # with known sentinel classes and confirming instantiation.
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch(
                "processing.architecture.quality_analyzer.ComplexityAnalysis"
            ) as mock_complexity_cls,
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            analyzer = ArchitectureQualityAnalyzer()

        assert analyzer.pattern_analyzer is mock_pattern_cls.return_value
        assert analyzer.metrics_calculator is mock_metrics_cls.return_value
        assert analyzer.complexity_analyzer is mock_complexity_cls.return_value
        assert analyzer.performance_analyzer is mock_perf_cls.return_value
        assert analyzer.reporter is mock_reporting_cls.return_value
        assert analyzer.tool_executor is mock_tool_exec_cls.return_value


# ---------------------------------------------------------------------------
# analyze() -- success path
# ---------------------------------------------------------------------------


class TestAnalyzeSuccess:
    @pytest.fixture
    def analyzer_with_mocks(self):
        """Build an ArchitectureQualityAnalyzer with all sub-components mocked."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch(
                "processing.architecture.quality_analyzer.ComplexityAnalysis"
            ) as mock_complexity_cls,
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            # Configure pattern analyzer
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff", "ty"]
            mock_pattern.is_tool_enabled.return_value = True
            mock_pattern.get_tool_config.return_value = {"command": ["ruff", "check", "."]}
            mock_pattern.count_files_by_language.return_value = 42

            # Configure tool executor
            mock_tool_exec = mock_tool_exec_cls.return_value
            test_issue = _make_issue()
            mock_tool_exec.run_tool = AsyncMock(return_value=([test_issue], {"success": True}))

            # Configure performance analyzer
            mock_perf = mock_perf_cls.return_value
            mock_perf.analyze_performance_patterns.return_value = []

            # Configure metrics calculator
            mock_metrics = mock_metrics_cls.return_value
            mock_metrics.calculate_quality_score.return_value = 85.0
            mock_metrics.score_to_grade.return_value = "B"

            # Configure reporter
            mock_reporter = mock_reporting_cls.return_value
            mock_reporter.generate_summary_statistics.return_value = {
                "total_issues": 2,
                "total_files": 42,
            }
            expected_report = _make_quality_report(
                issues=[test_issue, test_issue],
                overall_score=85.0,
                grade="B",
            )
            mock_reporter.generate_quality_report.return_value = expected_report

            analyzer = ArchitectureQualityAnalyzer()
            yield (
                analyzer,
                {
                    "pattern": mock_pattern,
                    "metrics": mock_metrics,
                    "complexity": mock_complexity_cls.return_value,
                    "performance": mock_perf,
                    "reporter": mock_reporter,
                    "tool_exec": mock_tool_exec,
                    "expected_report": expected_report,
                    "test_issue": test_issue,
                },
            )

    async def test_returns_quality_report(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        report = await analyzer.analyze(Path("/repo"), codebase_id="codebase-1")

        assert report is not None
        assert isinstance(report, QualityReport)
        assert report.overall_score == 85.0
        assert report.grade == "B"

    async def test_detects_language(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        mocks["pattern"].detect_primary_language.assert_called_once_with(Path("/repo"))

    async def test_gets_available_tools(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        mocks["pattern"].get_available_tools.assert_called_once_with("python")

    async def test_runs_each_enabled_tool(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        # Two tools: ruff and ty -- each should be run once
        assert mocks["tool_exec"].run_tool.await_count == 2

    async def test_runs_performance_analysis(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        mocks["performance"].analyze_performance_patterns.assert_called_once_with(
            Path("/repo"), "python"
        )

    async def test_calculates_score(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        mocks["metrics"].calculate_quality_score.assert_called_once()

    async def test_generates_summary(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"))

        mocks["reporter"].generate_summary_statistics.assert_called_once()

    async def test_generates_report(self, analyzer_with_mocks):
        analyzer, mocks = analyzer_with_mocks
        await analyzer.analyze(Path("/repo"), codebase_id="codebase-1")

        mocks["reporter"].generate_quality_report.assert_called_once()
        call_kwargs = mocks["reporter"].generate_quality_report.call_args.kwargs
        assert call_kwargs["codebase_id"] == "codebase-1"
        assert call_kwargs["overall_score"] == 85.0
        assert call_kwargs["grade"] == "B"

    async def test_tenant_is_threaded_into_the_container_calls(self, analyzer_with_mocks):
        """Both container hops must run as the tenant that owns the workspace.

        Without ``X-Tenant-Id`` coding-containers resolves the caller as
        ``svc-coding-analysis``'s service account and its BOLA guard 404s the
        exec, so every tool "fails" in ~1.2 ms.
        """
        tenant = "0195f48b-d000-7000-8000-000000000003"
        analyzer, mocks = analyzer_with_mocks
        with patch(
            "processing.architecture.quality_analyzer.materialize_from_container",
            AsyncMock(return_value=None),
        ) as mock_materialize:
            await analyzer.analyze(Path("/repo"), codebase_id="ws-1", tenant_id=tenant)

        assert mock_materialize.await_args.kwargs.get("tenant_id") == tenant, (
            "materialize_from_container was called with NO tenant -- the "
            "container `find`/read is 404'd by the BOLA guard"
        )
        assert mocks["tool_exec"].run_tool.await_args.kwargs.get("tenant_id") == tenant, (
            "run_tool was called with NO tenant -- every static-analysis exec is "
            "404'd by the BOLA guard and the report comes back inconclusive"
        )

    async def test_report_carries_the_detected_language(self, analyzer_with_mocks):
        """The report names what detect_primary_language() actually found."""
        analyzer, mocks = analyzer_with_mocks
        mocks["pattern"].detect_primary_language.return_value = "python"
        await analyzer.analyze(Path("/repo"), codebase_id="ws-1")

        call_kwargs = mocks["reporter"].generate_quality_report.call_args.kwargs
        assert call_kwargs["language"] == "python"

    async def test_disabled_tool_is_skipped(self):
        """If is_tool_enabled returns False for a tool, it should not be run."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff", "ty"]
            mock_pattern.is_tool_enabled.side_effect = lambda lang, tool: tool == "ruff"
            mock_pattern.get_tool_config.return_value = {"command": ["ruff", "check"]}
            mock_pattern.count_files_by_language.return_value = 10

            mock_tool_exec = mock_tool_exec_cls.return_value
            mock_tool_exec.run_tool = AsyncMock(return_value=([], {"success": True}))

            mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
            mock_metrics_cls.return_value.calculate_quality_score.return_value = 100.0
            mock_metrics_cls.return_value.score_to_grade.return_value = "A"
            mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
            mock_reporting_cls.return_value.generate_quality_report.return_value = (
                _make_quality_report()
            )

            analyzer = ArchitectureQualityAnalyzer()
            await analyzer.analyze(Path("/repo"))

            # Only ruff should have been run, not ty
            assert mock_tool_exec.run_tool.await_count == 1

    async def test_tool_with_non_list_command_is_skipped(self):
        """If get_tool_config returns a non-list command, it should be skipped."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff"]
            mock_pattern.is_tool_enabled.return_value = True
            mock_pattern.get_tool_config.return_value = {"command": "not-a-list"}
            mock_pattern.count_files_by_language.return_value = 10

            mock_tool_exec = mock_tool_exec_cls.return_value
            mock_tool_exec.run_tool = AsyncMock(return_value=([], {"success": True}))

            mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
            mock_metrics_cls.return_value.calculate_quality_score.return_value = 100.0
            mock_metrics_cls.return_value.score_to_grade.return_value = "A"
            mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
            mock_reporting_cls.return_value.generate_quality_report.return_value = (
                _make_quality_report()
            )

            analyzer = ArchitectureQualityAnalyzer()
            await analyzer.analyze(Path("/repo"))

            mock_tool_exec.run_tool.assert_not_awaited()

    async def test_abstains_when_no_tool_succeeds(self):
        """If every tool fails to run, report an inconclusive verdict (-1.0 / N/A),
        not a vacuous perfect score. calculate_quality_score must NOT be consulted."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff", "ty"]
            mock_pattern.is_tool_enabled.return_value = True
            mock_pattern.get_tool_config.return_value = {"command": ["ruff", "check"]}
            mock_pattern.count_files_by_language.return_value = 10

            # Every tool fails (e.g. not installed / container exec 400).
            mock_tool_exec = mock_tool_exec_cls.return_value
            mock_tool_exec.run_tool = AsyncMock(
                return_value=([], {"success": False, "error": "not found"})
            )

            mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
            mock_metrics = mock_metrics_cls.return_value
            mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
            mock_reporting_cls.return_value.generate_quality_report.return_value = (
                _make_quality_report()
            )

            analyzer = ArchitectureQualityAnalyzer()
            await analyzer.analyze(Path("/repo"), codebase_id="cb-1")

            mock_metrics.calculate_quality_score.assert_not_called()
            call_kwargs = mock_reporting_cls.return_value.generate_quality_report.call_args.kwargs
            assert call_kwargs["overall_score"] == -1.0
            assert call_kwargs["grade"] == "N/A"
            # Inconclusive is the 0-succeeded end of the spectrum: NOT partial,
            # but the provenance still names every attempted (and failed) tool.
            assert call_kwargs["inconclusive"] is True
            assert call_kwargs["partial"] is False
            assert call_kwargs["tools_attempted"] == ["ruff", "ty"]
            assert call_kwargs["tools_succeeded"] == []
            assert call_kwargs["tools_failed"] == ["ruff", "ty"]

    async def test_partial_when_some_tools_fail(self):
        """Some (not all) tools succeed -> partial=True with the per-tool split,
        and a real score is still computed from the tools that ran."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff", "ty"]
            mock_pattern.is_tool_enabled.return_value = True
            mock_pattern.get_tool_config.return_value = {"command": ["ruff", "check"]}
            mock_pattern.count_files_by_language.return_value = 10

            # ruff succeeds, ty errors.
            mock_tool_exec = mock_tool_exec_cls.return_value
            mock_tool_exec.run_tool = AsyncMock(
                side_effect=[
                    ([], {"success": True}),
                    ([], {"success": False, "error": "ty not found"}),
                ]
            )

            mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
            mock_metrics = mock_metrics_cls.return_value
            mock_metrics.calculate_quality_score.return_value = 90.0
            mock_metrics.score_to_grade.return_value = "A"
            mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
            mock_reporting_cls.return_value.generate_quality_report.return_value = (
                _make_quality_report()
            )

            analyzer = ArchitectureQualityAnalyzer()
            await analyzer.analyze(Path("/repo"), codebase_id="cb-1")

            # A real score IS computed (not the inconclusive sentinel).
            mock_metrics.calculate_quality_score.assert_called_once()
            call_kwargs = mock_reporting_cls.return_value.generate_quality_report.call_args.kwargs
            assert call_kwargs["inconclusive"] is False
            assert call_kwargs["partial"] is True
            assert call_kwargs["tools_attempted"] == ["ruff", "ty"]
            assert call_kwargs["tools_succeeded"] == ["ruff"]
            assert call_kwargs["tools_failed"] == ["ty"]


# ---------------------------------------------------------------------------
# analyze() -- error path
# ---------------------------------------------------------------------------


class TestAnalyzeError:
    async def test_returns_none_on_exception(self):
        """analyze() should catch exceptions and return None."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch("processing.architecture.quality_analyzer.CodeQualityMetrics"),
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch("processing.architecture.quality_analyzer.QualityReporting"),
            patch("processing.architecture.quality_analyzer.ToolExecution"),
        ):
            mock_pattern_cls.return_value.detect_primary_language.side_effect = RuntimeError("boom")

            analyzer = ArchitectureQualityAnalyzer()
            result = await analyzer.analyze(Path("/repo"), codebase_id="test")

        assert result is None

    async def test_returns_none_on_tool_execution_failure(self):
        """If tool execution raises, analyze() should return None."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch("processing.architecture.quality_analyzer.CodeQualityMetrics"),
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch("processing.architecture.quality_analyzer.QualityReporting"),
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = ["ruff"]
            mock_pattern.is_tool_enabled.return_value = True
            mock_pattern.get_tool_config.return_value = {"command": ["ruff", "check"]}

            mock_tool_exec_cls.return_value.run_tool = AsyncMock(
                side_effect=Exception("tool crashed")
            )

            analyzer = ArchitectureQualityAnalyzer()
            result = await analyzer.analyze(Path("/repo"))

        assert result is None


# ---------------------------------------------------------------------------
# _ruff_severity_to_standard
# ---------------------------------------------------------------------------


class TestRuffSeverityToStandard:
    @pytest.fixture
    def analyzer(self):
        with (
            patch("processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"),
            patch("processing.architecture.quality_analyzer.CodeQualityMetrics"),
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch("processing.architecture.quality_analyzer.QualityReporting"),
            patch("processing.architecture.quality_analyzer.ToolExecution"),
        ):
            return ArchitectureQualityAnalyzer()

    def test_f_codes_return_critical(self, analyzer):
        assert analyzer._ruff_severity_to_standard("F401") == "critical"
        assert analyzer._ruff_severity_to_standard("F811") == "critical"
        assert analyzer._ruff_severity_to_standard("F") == "critical"

    def test_e_codes_return_high(self, analyzer):
        assert analyzer._ruff_severity_to_standard("E501") == "high"
        assert analyzer._ruff_severity_to_standard("E123") == "high"
        assert analyzer._ruff_severity_to_standard("E") == "high"

    def test_w_codes_return_medium(self, analyzer):
        assert analyzer._ruff_severity_to_standard("W291") == "medium"
        assert analyzer._ruff_severity_to_standard("W605") == "medium"
        assert analyzer._ruff_severity_to_standard("W") == "medium"

    def test_other_codes_return_low(self, analyzer):
        assert analyzer._ruff_severity_to_standard("I001") == "low"
        assert analyzer._ruff_severity_to_standard("N801") == "low"
        assert analyzer._ruff_severity_to_standard("UP006") == "low"
        assert analyzer._ruff_severity_to_standard("C901") == "low"

    def test_empty_string_returns_low(self, analyzer):
        assert analyzer._ruff_severity_to_standard("") == "low"


# ---------------------------------------------------------------------------
# _eslint_severity_to_standard
# ---------------------------------------------------------------------------


class TestEslintSeverityToStandard:
    @pytest.fixture
    def analyzer(self):
        with (
            patch("processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"),
            patch("processing.architecture.quality_analyzer.CodeQualityMetrics"),
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch("processing.architecture.quality_analyzer.QualityReporting"),
            patch("processing.architecture.quality_analyzer.ToolExecution"),
        ):
            return ArchitectureQualityAnalyzer()

    def test_severity_2_returns_high(self, analyzer):
        assert analyzer._eslint_severity_to_standard(2) == "high"

    def test_severity_1_returns_medium(self, analyzer):
        assert analyzer._eslint_severity_to_standard(1) == "medium"

    def test_severity_0_returns_medium(self, analyzer):
        """Severity 0 (off) should fall through to 'medium'."""
        assert analyzer._eslint_severity_to_standard(0) == "medium"

    def test_any_non_2_severity_returns_medium(self, analyzer):
        assert analyzer._eslint_severity_to_standard(3) == "medium"
        assert analyzer._eslint_severity_to_standard(-1) == "medium"


# ---------------------------------------------------------------------------
# Legacy delegation methods
# ---------------------------------------------------------------------------


class TestLegacyDelegation:
    @pytest.fixture
    def analyzer(self):
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis"),
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution") as mock_tool_exec_cls,
        ):
            analyzer = ArchitectureQualityAnalyzer()

            # Store mocks for assertions
            object.__setattr__(analyzer, "_mock_pattern", mock_pattern_cls.return_value)
            object.__setattr__(analyzer, "_mock_metrics", mock_metrics_cls.return_value)
            object.__setattr__(analyzer, "_mock_reporter", mock_reporting_cls.return_value)
            object.__setattr__(analyzer, "_mock_tool_exec", mock_tool_exec_cls.return_value)

            yield analyzer

    def test_detect_primary_language_delegates(self, analyzer):
        analyzer._mock_pattern.detect_primary_language.return_value = "go"
        result = analyzer._detect_primary_language(Path("/repo"))
        assert result == "go"
        analyzer._mock_pattern.detect_primary_language.assert_called_once_with(Path("/repo"))

    def test_calculate_quality_score_delegates(self, analyzer):
        issues = [_make_issue()]
        analyzer._mock_metrics.calculate_quality_score.return_value = 92.5
        result = analyzer.calculate_quality_score(issues)
        assert result == 92.5
        analyzer._mock_metrics.calculate_quality_score.assert_called_once_with(issues)

    def test_score_to_grade_delegates(self, analyzer):
        analyzer._mock_metrics.score_to_grade.return_value = "A"
        result = analyzer._score_to_grade(95.0)
        assert result == "A"

    def test_count_issues_by_severity_delegates(self, analyzer):
        issues = [_make_issue()]
        analyzer._mock_metrics.count_issues_by_severity.return_value = {"medium": 1}
        result = analyzer._count_issues_by_severity(issues)
        assert result == {"medium": 1}

    def test_count_issues_by_tool_delegates(self, analyzer):
        issues = [_make_issue()]
        analyzer._mock_metrics.count_issues_by_tool.return_value = {"ruff": 1}
        result = analyzer._count_issues_by_tool(issues)
        assert result == {"ruff": 1}

    def test_count_files_delegates(self, analyzer):
        analyzer._mock_pattern.count_files_by_language.return_value = 50
        result = analyzer._count_files(Path("/repo"), "python")
        assert result == 50

    def test_generate_recommendations_delegates(self, analyzer):
        issues = [_make_issue()]
        analyzer._mock_reporter.generate_recommendations.return_value = ["Fix issues"]
        result = analyzer._generate_recommendations(issues, 70.0)
        assert result == ["Fix issues"]

    async def test_run_tool_legacy_delegates(self, analyzer):
        analyzer._mock_tool_exec.run_tool = AsyncMock(return_value=([], {"success": True}))
        issues, result = await analyzer._run_tool("ruff", [], Path("/repo"), "python")
        assert result["success"] is True
        analyzer._mock_tool_exec.run_tool.assert_awaited_once_with(
            "ruff", [], Path("/repo"), "python"
        )


# ---------------------------------------------------------------------------
# A2: container-materialized language detection / perf scanning
# ---------------------------------------------------------------------------


class TestContainerMaterialization:
    """The repo lives inside the analysis container, not on the analyzer pod's
    local fs. Detection / perf scanning must run against files mirrored out of the
    container — otherwise a non-Python repo is misdetected as the python default."""

    async def test_detection_runs_against_materialized_dir_and_cleans_up(self):
        # Stand in for the container copy: a real temp dir holding a JS file. The
        # detection/perf scanners must run against THIS, not the local /repo.
        import shutil
        import tempfile

        materialized = Path(tempfile.mkdtemp(prefix="qa-test-"))
        (materialized / "app.js").write_text("console.log('hi')\n")
        seen: dict[str, object] = {}

        try:
            with (
                patch(
                    "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
                ) as mock_pattern_cls,
                patch(
                    "processing.architecture.quality_analyzer.CodeQualityMetrics"
                ) as mock_metrics_cls,
                patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
                patch(
                    "processing.architecture.quality_analyzer.PerformanceAnalysis"
                ) as mock_perf_cls,
                patch(
                    "processing.architecture.quality_analyzer.QualityReporting"
                ) as mock_reporting_cls,
                patch("processing.architecture.quality_analyzer.ToolExecution"),
                patch(
                    "processing.architecture.quality_analyzer.materialize_from_container",
                    AsyncMock(return_value=materialized),
                ) as mock_materialize,
            ):
                mock_pattern = mock_pattern_cls.return_value

                def _detect(path):
                    # Detection must see the materialized temp dir (which actually
                    # contains app.js), NOT the bare local /repo that doesn't exist.
                    seen["detect_path"] = Path(path)
                    seen["app_js_present"] = (Path(path) / "app.js").exists()
                    return "javascript"

                mock_pattern.detect_primary_language.side_effect = _detect
                mock_pattern.get_available_tools.return_value = []
                mock_pattern.count_files_by_language.return_value = 1

                mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
                mock_metrics_cls.return_value.calculate_quality_score.return_value = 80.0
                mock_metrics_cls.return_value.score_to_grade.return_value = "B"
                mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
                mock_reporting_cls.return_value.generate_quality_report.return_value = (
                    _make_quality_report()
                )

                analyzer = ArchitectureQualityAnalyzer()
                report = await analyzer.analyze(Path("/repo"), codebase_id="container-1")

            assert report is not None
            # Materialization was driven by the container id and the source globs.
            assert mock_materialize.await_args.args[1] == "container-1"
            # Detection / perf ran against the container copy, not the local /repo.
            assert seen["detect_path"] == materialized
            assert seen["app_js_present"] is True
            perf_path = mock_perf_cls.return_value.analyze_performance_patterns.call_args.args[0]
            assert perf_path == materialized
            # ...and the temp dir is cleaned up by analyze()'s finally.
            assert not materialized.exists()
        finally:
            shutil.rmtree(materialized, ignore_errors=True)

    async def test_empty_codebase_id_skips_container_and_uses_local(self):
        """With no container id (local callers / tests) the local path is used
        directly and the container is never consulted."""
        with (
            patch(
                "processing.architecture.quality_analyzer.ArchitecturePatternAnalysis"
            ) as mock_pattern_cls,
            patch(
                "processing.architecture.quality_analyzer.CodeQualityMetrics"
            ) as mock_metrics_cls,
            patch("processing.architecture.quality_analyzer.ComplexityAnalysis"),
            patch("processing.architecture.quality_analyzer.PerformanceAnalysis") as mock_perf_cls,
            patch(
                "processing.architecture.quality_analyzer.QualityReporting"
            ) as mock_reporting_cls,
            patch("processing.architecture.quality_analyzer.ToolExecution"),
            patch(
                "processing.architecture.quality_analyzer.materialize_from_container",
                AsyncMock(),
            ) as mock_materialize,
        ):
            mock_pattern = mock_pattern_cls.return_value
            mock_pattern.detect_primary_language.return_value = "python"
            mock_pattern.get_available_tools.return_value = []
            mock_pattern.count_files_by_language.return_value = 0
            mock_perf_cls.return_value.analyze_performance_patterns.return_value = []
            mock_metrics_cls.return_value.calculate_quality_score.return_value = 0.0
            mock_reporting_cls.return_value.generate_summary_statistics.return_value = {}
            mock_reporting_cls.return_value.generate_quality_report.return_value = (
                _make_quality_report()
            )

            analyzer = ArchitectureQualityAnalyzer()
            await analyzer.analyze(Path("/repo"), codebase_id="")

        mock_materialize.assert_not_awaited()
        mock_pattern.detect_primary_language.assert_called_once_with(Path("/repo"))
