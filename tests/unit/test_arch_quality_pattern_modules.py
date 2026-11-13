#!/usr/bin/env python3
"""Unit tests for architecture quality and pattern modules.

Covers:
- quality_reporting.py (QualityReporting)
- quality_analyzer.py (ArchitectureQualityAnalyzer)
- architecture_pattern_analyzer.py (ArchitecturePatternAnalyzer)
- architecture_pattern_analysis.py (ArchitecturePatternAnalysis)
- pattern_validation_scoring.py (PatternValidationScorer)
- pattern_reporting_utils.py (PatternReportingUtils)
- cloud_pattern_helpers.py (CloudPatternHelpers)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from models.architecture_models import (
    ArchitectureQualityIssue,
    DetectedPattern,
    QualityReport,
    RepoConfig,
)
from processing.architecture.architecture_pattern_analysis import ArchitecturePatternAnalysis
from processing.architecture.architecture_pattern_analyzer import ArchitecturePatternAnalyzer
from processing.architecture.cloud_pattern_helpers import CloudPatternHelpers
from processing.architecture.code_pattern_recognition import CodePatternRecognizer
from processing.architecture.pattern_reporting_utils import PatternReportingUtils
from processing.architecture.pattern_validation_scoring import PatternValidationScorer
from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer
from processing.architecture.quality_reporting import QualityReporting

# ===================================================================
# Helpers
# ===================================================================


def _make_issue(
    *,
    severity: str = "medium",
    category: str = "style",
    tool: str = "ruff",
    file_path: str = "app.py",
    line_number: int = 1,
    message: str = "some issue",
) -> ArchitectureQualityIssue:
    return ArchitectureQualityIssue(
        id="issue-1",
        severity=severity,
        message=message,
        rule_id="R001",
        rule_name="test-rule",
        file_path=file_path,
        line_number=line_number,
        tool=tool,
        category=category,
    )


def _make_pattern(
    *,
    pattern_type: str = "Singleton",
    confidence: float = 0.8,
    components: list[str] | None = None,
    name: str = "Singleton Pattern",
    location: str = "module.py",
    description: str = "Detected singleton",
) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=pattern_type,
        confidence=confidence,
        components=components or ["node_a"],
        description=description,
        name=name,
        location=location,
    )


def _make_report(
    *,
    score: float = 85.0,
    grade: str = "B",
    issues: list[ArchitectureQualityIssue] | None = None,
    summary: dict | None = None,
) -> QualityReport:
    return QualityReport(
        job_id="quality-test",
        codebase_id="test-codebase",
        generated_at="2026-01-01T00:00:00",
        overall_score=score,
        grade=grade,
        issues=issues or [],
        summary=summary or {"total_files_analyzed": 10, "total_issues": 0},
        recommendations=["rec1", "rec2", "rec3", "rec4"],
        execution_time=1.5,
    )


def _build_graph_with_nodes(*node_specs: tuple) -> nx.DiGraph:
    """Build a DiGraph from (node_id, data_dict) tuples."""
    g = nx.DiGraph()
    for node_id, data in node_specs:
        g.add_node(node_id, **data)
    return g


# ===================================================================
# QualityReporting tests
# ===================================================================


class TestQualityReporting:
    def setup_method(self):
        self.reporter = QualityReporting()

    # -- generate_quality_report -----------------------------------------

    def test_generate_quality_report_basic(self):
        issues = [_make_issue()]
        report = self.reporter.generate_quality_report(
            codebase_id="abc",
            issues=issues,
            overall_score=90.0,
            grade="A",
            summary={"total_files_analyzed": 5},
            execution_time=2.0,
        )
        assert report.codebase_id == "abc"
        assert report.overall_score == 90.0
        assert report.grade == "A"
        assert report.job_id == "quality-abc"
        assert len(report.recommendations) > 0

    def test_generate_quality_report_with_additional_metrics(self):
        report = self.reporter.generate_quality_report(
            codebase_id="x",
            issues=[],
            overall_score=75.0,
            grade="C",
            summary={"total_files_analyzed": 3},
            execution_time=1.0,
            additional_metrics={"custom_metric": 42},
        )
        assert report.summary["custom_metric"] == 42

    def test_generate_quality_report_inconclusive_abstain(self):
        # Regression (Defect A): a run where no static-analysis tool succeeded is
        # threaded through as inconclusive=True with the -1.0/N/A sentinels and an
        # honest recommendation — never a vacuous pass.
        report = self.reporter.generate_quality_report(
            codebase_id="abc",
            issues=[],
            overall_score=-1.0,
            grade="N/A",
            summary={"total_files_analyzed": 0},
            execution_time=0.5,
            inconclusive=True,
        )
        assert report.inconclusive is True
        assert report.grade == "N/A"
        assert report.overall_score == -1.0
        assert any("inconclusive" in r.lower() for r in report.recommendations)
        assert not any("Great job" in r for r in report.recommendations)

    def test_generate_quality_report_conclusive_defaults_not_inconclusive(self):
        report = self.reporter.generate_quality_report(
            codebase_id="abc",
            issues=[],
            overall_score=90.0,
            grade="A",
            summary={"total_files_analyzed": 5},
            execution_time=2.0,
        )
        assert report.inconclusive is False

    # -- generate_recommendations ----------------------------------------

    def test_recommendations_inconclusive_is_honest(self):
        recs = self.reporter.generate_recommendations([], -1.0, inconclusive=True)
        assert any("inconclusive" in r.lower() for r in recs)
        assert any("NOT a pass" in r for r in recs)
        assert not any("Good code quality" in r for r in recs)

    def test_recommendations_high_score(self):
        recs = self.reporter.generate_recommendations([], 95.0)
        assert any("Good code quality" in r for r in recs)

    def test_recommendations_mid_score(self):
        recs = self.reporter.generate_recommendations([], 75.0)
        assert any("stricter linting" in r for r in recs)

    def test_recommendations_low_score(self):
        recs = self.reporter.generate_recommendations([], 50.0)
        assert any("pre-commit" in r for r in recs)

    def test_recommendations_with_security_issues(self):
        issues = [_make_issue(category="security") for _ in range(3)]
        recs = self.reporter.generate_recommendations(issues, 50.0)
        assert any("security" in r.lower() for r in recs)

    def test_recommendations_with_type_safety_issues(self):
        issues = [_make_issue(category="type-safety") for _ in range(6)]
        recs = self.reporter.generate_recommendations(issues, 80.0)
        assert any("type" in r.lower() for r in recs)

    def test_recommendations_category_below_threshold_skipped(self):
        # style threshold is 10, only 2 issues
        issues = [_make_issue(category="style") for _ in range(2)]
        recs = self.reporter.generate_recommendations(issues, 85.0)
        assert not any("style" in r.lower() and "formatting" in r.lower() for r in recs)

    # -- _count_issues_by_category ---------------------------------------

    def test_count_issues_by_category(self):
        issues = [
            _make_issue(category="security"),
            _make_issue(category="security"),
            _make_issue(category="style"),
        ]
        counts = self.reporter._count_issues_by_category(issues)
        assert counts["security"] == 2
        assert counts["style"] == 1

    # -- generate_summary_statistics -------------------------------------

    def test_generate_summary_statistics(self):
        issues = [
            _make_issue(severity="critical", tool="ruff"),
            _make_issue(severity="high", tool="ty"),
            _make_issue(severity="low", tool="ruff"),
        ]
        summary = self.reporter.generate_summary_statistics(issues, 20)
        assert summary["total_files_analyzed"] == 20
        assert summary["total_issues"] == 3
        assert summary["critical_issues"] == 1
        assert summary["high_issues"] == 1
        assert summary["low_issues"] == 1
        assert summary["ruff_issues"] == 2
        assert summary["ty_issues"] == 1

    def test_generate_summary_statistics_with_metrics(self):
        summary = self.reporter.generate_summary_statistics([], 5, {"extra": 99})
        assert summary["extra"] == 99

    def test_generate_summary_statistics_unknown_severity_buckets_to_medium(self):
        issues = [_make_issue(severity="UNKNOWN")]
        summary = self.reporter.generate_summary_statistics(issues, 1)
        assert summary["medium_issues"] == 1

    # -- _count_issues_by_severity / _count_issues_by_tool ---------------

    def test_count_issues_by_severity(self):
        issues = [_make_issue(severity="critical"), _make_issue(severity="low")]
        counts = self.reporter._count_issues_by_severity(issues)
        assert counts["critical"] == 1
        assert counts["low"] == 1

    def test_count_issues_by_tool(self):
        issues = [_make_issue(tool="ruff"), _make_issue(tool="ruff"), _make_issue(tool="bandit")]
        counts = self.reporter._count_issues_by_tool(issues)
        assert counts["ruff"] == 2
        assert counts["bandit"] == 1

    # -- generate_executive_summary --------------------------------------

    def test_executive_summary_basic(self):
        report = _make_report(score=85.0, grade="B")
        summary = self.reporter.generate_executive_summary(report)
        assert summary["overall_grade"] == "B"
        assert summary["overall_score"] == 85.0
        assert len(summary["priority_actions"]) <= 3

    def test_executive_summary_with_critical_issues(self):
        report = _make_report(
            score=60.0,
            summary={
                "total_files_analyzed": 10,
                "total_issues": 5,
                "critical_issues": 3,
                "high_issues": 15,
            },
        )
        summary = self.reporter.generate_executive_summary(report)
        assert any("critical" in f for f in summary["key_findings"])
        assert any("high-severity" in f for f in summary["key_findings"])
        assert any("below" in f for f in summary["key_findings"])

    def test_executive_summary_with_trends(self):
        report = _make_report()
        summary = self.reporter.generate_executive_summary(report, trends={"direction": "up"})
        assert summary["trends"]["direction"] == "up"

    # -- generate_detailed_breakdown -------------------------------------

    def test_detailed_breakdown(self):
        issues = [
            _make_issue(severity="critical", file_path="a.py"),
            _make_issue(severity="high", file_path="b.py"),
            _make_issue(severity="low", file_path="a.py"),
        ]
        breakdown = self.reporter.generate_detailed_breakdown(issues)
        assert "by_severity" in breakdown
        assert "by_category" in breakdown
        assert "by_tool" in breakdown
        assert "by_file" in breakdown
        assert "top_issues" in breakdown
        assert breakdown["by_file"]["a.py"] == 2

    def test_detailed_breakdown_empty(self):
        breakdown = self.reporter.generate_detailed_breakdown([])
        assert breakdown["by_file"] == {}
        assert breakdown["top_issues"] == []

    # -- format_recommendations_by_priority ------------------------------

    def test_format_recommendations_by_priority(self):
        recs = [
            "Fix critical security vulnerability",
            "Refactor complex functions",
            "Add more tests",
        ]
        formatted = self.reporter.format_recommendations_by_priority(recs)
        assert "Fix critical security vulnerability" in formatted["immediate"]
        assert "Refactor complex functions" in formatted["short_term"]
        assert "Add more tests" in formatted["long_term"]

    def test_format_recommendations_empty(self):
        formatted = self.reporter.format_recommendations_by_priority([])
        assert formatted["immediate"] == []
        assert formatted["short_term"] == []
        assert formatted["long_term"] == []

    # -- generate_comparison_report --------------------------------------

    def test_comparison_report_no_previous(self):
        current = _make_report(score=80.0, grade="B")
        comparison = self.reporter.generate_comparison_report(current)
        assert comparison["current"]["score"] == 80.0
        assert "previous" not in comparison

    def test_comparison_report_with_previous_improving(self):
        current = _make_report(score=90.0, grade="A")
        previous = _make_report(score=70.0, grade="C")
        comparison = self.reporter.generate_comparison_report(current, previous)
        assert comparison["changes"]["score_change"] == 20.0
        assert comparison["trend"] == "improving"

    def test_comparison_report_declining(self):
        current = _make_report(score=60.0, grade="D")
        previous = _make_report(score=80.0, grade="B")
        comparison = self.reporter.generate_comparison_report(current, previous)
        assert comparison["trend"] == "declining"

    def test_comparison_report_stable(self):
        current = _make_report(score=80.0, grade="B")
        previous = _make_report(score=80.0, grade="B")
        comparison = self.reporter.generate_comparison_report(current, previous)
        assert comparison["trend"] == "stable"


# ===================================================================
# ArchitectureQualityAnalyzer tests
# ===================================================================


class TestArchitectureQualityAnalyzer:
    def setup_method(self):
        self.analyzer = ArchitectureQualityAnalyzer()

    # -- Legacy delegation methods ---------------------------------------

    def test_detect_primary_language_delegates(self):
        with patch.object(
            self.analyzer.pattern_analyzer, "detect_primary_language", return_value="go"
        ):
            result = self.analyzer._detect_primary_language(Path("/fake"))
            assert result == "go"

    def test_calculate_quality_score_delegates(self):
        with patch.object(
            self.analyzer.metrics_calculator,
            "calculate_quality_score",
            return_value=88.0,
        ):
            result = self.analyzer.calculate_quality_score([])
            assert result == 88.0

    def test_score_to_grade_delegates(self):
        with patch.object(self.analyzer.metrics_calculator, "score_to_grade", return_value="A"):
            result = self.analyzer._score_to_grade(95.0)
            assert result == "A"

    def test_count_issues_by_severity_delegates(self):
        expected = {"critical": 1}
        with patch.object(
            self.analyzer.metrics_calculator,
            "count_issues_by_severity",
            return_value=expected,
        ):
            result = self.analyzer._count_issues_by_severity([])
            assert result == expected

    def test_count_issues_by_tool_delegates(self):
        expected = {"ruff": 2}
        with patch.object(
            self.analyzer.metrics_calculator,
            "count_issues_by_tool",
            return_value=expected,
        ):
            result = self.analyzer._count_issues_by_tool([])
            assert result == expected

    def test_count_files_delegates(self):
        with patch.object(
            self.analyzer.pattern_analyzer, "count_files_by_language", return_value=42
        ):
            result = self.analyzer._count_files(Path("/fake"), "python")
            assert result == 42

    def test_generate_recommendations_delegates(self):
        expected = ["do something"]
        with patch.object(
            self.analyzer.reporter, "generate_recommendations", return_value=expected
        ):
            result = self.analyzer._generate_recommendations([], 80.0)
            assert result == expected

    # -- Severity mapping methods ----------------------------------------

    def test_ruff_severity_fatal(self):
        assert self.analyzer._ruff_severity_to_standard("F401") == "critical"

    def test_ruff_severity_error(self):
        assert self.analyzer._ruff_severity_to_standard("E501") == "high"

    def test_ruff_severity_warning(self):
        assert self.analyzer._ruff_severity_to_standard("W291") == "medium"

    def test_ruff_severity_other(self):
        assert self.analyzer._ruff_severity_to_standard("C901") == "low"

    def test_eslint_severity_error(self):
        assert self.analyzer._eslint_severity_to_standard(2) == "high"

    def test_eslint_severity_warning(self):
        assert self.analyzer._eslint_severity_to_standard(1) == "medium"

    # -- analyze (async) -------------------------------------------------

    @pytest.mark.asyncio
    async def test_analyze_happy_path(self):
        repo_path = Path("/fake/repo")

        with (
            patch.object(
                self.analyzer.pattern_analyzer,
                "detect_primary_language",
                return_value="python",
            ),
            patch.object(
                self.analyzer.pattern_analyzer,
                "get_available_tools",
                return_value=["ruff"],
            ),
            patch.object(self.analyzer.pattern_analyzer, "is_tool_enabled", return_value=True),
            patch.object(
                self.analyzer.pattern_analyzer,
                "get_tool_config",
                return_value={"command": ["ruff", "check"]},
            ),
            patch.object(
                self.analyzer.tool_executor,
                "run_tool",
                new_callable=AsyncMock,
                return_value=([], {"status": "ok"}),
            ),
            patch.object(
                self.analyzer.performance_analyzer,
                "analyze_performance_patterns",
                return_value=[],
            ),
            patch.object(
                self.analyzer.metrics_calculator,
                "calculate_quality_score",
                return_value=90.0,
            ),
            patch.object(self.analyzer.metrics_calculator, "score_to_grade", return_value="A"),
            patch.object(
                self.analyzer.pattern_analyzer,
                "count_files_by_language",
                return_value=10,
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_summary_statistics",
                return_value={"total_files_analyzed": 10, "total_issues": 0},
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_quality_report",
                return_value=_make_report(),
            ),
        ):
            report = await self.analyzer.analyze(repo_path, "test-codebase")
            assert report is not None
            assert report.overall_score == 85.0

    @pytest.mark.asyncio
    async def test_analyze_exception_returns_none(self):
        with patch.object(
            self.analyzer.pattern_analyzer,
            "detect_primary_language",
            side_effect=RuntimeError("boom"),
        ):
            result = await self.analyzer.analyze(Path("/fake"), "id")
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_tool_command_not_list_skipped(self):
        with (
            patch.object(
                self.analyzer.pattern_analyzer,
                "detect_primary_language",
                return_value="python",
            ),
            patch.object(
                self.analyzer.pattern_analyzer,
                "get_available_tools",
                return_value=["ruff"],
            ),
            patch.object(self.analyzer.pattern_analyzer, "is_tool_enabled", return_value=True),
            patch.object(
                self.analyzer.pattern_analyzer,
                "get_tool_config",
                return_value={"command": "not-a-list"},
            ),
            patch.object(
                self.analyzer.performance_analyzer,
                "analyze_performance_patterns",
                return_value=[],
            ),
            patch.object(
                self.analyzer.metrics_calculator,
                "calculate_quality_score",
                return_value=90.0,
            ),
            patch.object(self.analyzer.metrics_calculator, "score_to_grade", return_value="A"),
            patch.object(
                self.analyzer.pattern_analyzer,
                "count_files_by_language",
                return_value=5,
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_summary_statistics",
                return_value={"total_files_analyzed": 5, "total_issues": 0},
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_quality_report",
                return_value=_make_report(),
            ),
        ):
            report = await self.analyzer.analyze(Path("/fake"), "id")
            assert report is not None

    # -- analyze_complexity / analyze_performance ------------------------

    def test_analyze_complexity(self):
        with (
            patch.object(
                self.analyzer.complexity_analyzer,
                "run_radon_complexity",
                return_value=([], {}),
            ),
            patch.object(
                self.analyzer.pattern_analyzer,
                "detect_architecture_patterns",
                return_value={},
            ),
            patch.object(
                self.analyzer.pattern_analyzer,
                "analyze_dependency_structure",
                return_value={},
            ),
        ):
            result = self.analyzer.analyze_complexity(Path("/fake"))
            assert "radon_analysis" in result
            assert "architecture_patterns" in result
            assert "dependency_structure" in result

    def test_analyze_performance(self):
        with (
            patch.object(
                self.analyzer.performance_analyzer,
                "analyze_performance_patterns",
                return_value=[],
            ),
            patch.object(
                self.analyzer.performance_analyzer,
                "analyze_memory_usage_patterns",
                return_value={},
            ),
            patch.object(
                self.analyzer.performance_analyzer,
                "run_profiling_analysis",
                return_value={},
            ),
        ):
            result = self.analyzer.analyze_performance(Path("/fake"))
            assert "performance_patterns" in result
            assert "memory_analysis" in result

    # -- generate_detailed_report ----------------------------------------

    @pytest.mark.asyncio
    async def test_generate_detailed_report_none_when_analysis_fails(self):
        with patch.object(self.analyzer, "analyze", new_callable=AsyncMock, return_value=None):
            result = await self.analyzer.generate_detailed_report(Path("/fake"), "id")
            assert result is None

    @pytest.mark.asyncio
    async def test_generate_detailed_report_with_performance(self):
        mock_report = _make_report()
        with (
            patch.object(
                self.analyzer, "analyze", new_callable=AsyncMock, return_value=mock_report
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_detailed_breakdown",
                return_value={},
            ),
            patch.object(
                self.analyzer.reporter,
                "generate_executive_summary",
                return_value={},
            ),
            patch.object(
                self.analyzer.pattern_analyzer,
                "detect_primary_language",
                return_value="python",
            ),
            patch.object(self.analyzer, "analyze_performance", return_value={"p": 1}),
            patch.object(self.analyzer, "analyze_complexity", return_value={"c": 2}),
        ):
            result = await self.analyzer.generate_detailed_report(
                Path("/fake"), "id", include_performance=True
            )
            assert result is not None
            assert "performance_analysis" in result
            assert "complexity_analysis" in result

    # -- Legacy tool delegation methods ----------------------------------

    @pytest.mark.asyncio
    async def test_run_ruff_delegates(self):
        with patch.object(
            self.analyzer.tool_executor,
            "run_tool",
            new_callable=AsyncMock,
            return_value=([], {}),
        ):
            result = await self.analyzer._run_ruff(Path("/fake"))
            assert result == ([], {})

    @pytest.mark.asyncio
    async def test_run_ty_delegates(self):
        with patch.object(
            self.analyzer.tool_executor,
            "run_tool",
            new_callable=AsyncMock,
            return_value=([], {}),
        ):
            result = await self.analyzer._run_ty(Path("/fake"))
            assert result == ([], {})

    @pytest.mark.asyncio
    async def test_run_bandit_delegates(self):
        with patch.object(
            self.analyzer.tool_executor,
            "run_tool",
            new_callable=AsyncMock,
            return_value=([], {}),
        ):
            result = await self.analyzer._run_bandit(Path("/fake"))
            assert result == ([], {})

    @pytest.mark.asyncio
    async def test_run_radon_delegates(self):
        with patch.object(
            self.analyzer.complexity_analyzer,
            "run_radon_complexity",
            return_value=([], {}),
        ):
            result = await self.analyzer._run_radon(Path("/fake"))
            assert result == ([], {})

    @pytest.mark.asyncio
    async def test_run_eslint_delegates(self):
        with patch.object(
            self.analyzer.tool_executor,
            "run_tool",
            new_callable=AsyncMock,
            return_value=([], {}),
        ):
            result = await self.analyzer._run_eslint(Path("/fake"), "javascript")
            assert result == ([], {})


# ===================================================================
# ArchitecturePatternAnalysis tests
# ===================================================================


class TestArchitecturePatternAnalysis:
    def setup_method(self):
        self.analysis = ArchitecturePatternAnalysis()

    # -- detect_primary_language ----------------------------------------

    def test_detect_primary_language_python(self, tmp_path):
        (tmp_path / "main.py").touch()
        (tmp_path / "utils.py").touch()
        assert self.analysis.detect_primary_language(tmp_path) == "python"

    def test_detect_primary_language_javascript(self, tmp_path):
        (tmp_path / "index.js").touch()
        (tmp_path / "app.js").touch()
        (tmp_path / "util.js").touch()
        assert self.analysis.detect_primary_language(tmp_path) == "javascript"

    def test_detect_primary_language_default_when_empty(self, tmp_path):
        assert self.analysis.detect_primary_language(tmp_path) == "python"

    # -- get_available_tools --------------------------------------------

    def test_get_available_tools_python(self):
        tools = self.analysis.get_available_tools("python")
        assert "ruff" in tools
        assert "ty" in tools

    def test_get_available_tools_unknown(self):
        assert self.analysis.get_available_tools("brainfuck") == []

    # -- get_tool_config ------------------------------------------------

    def test_get_tool_config_exists(self):
        config = self.analysis.get_tool_config("python", "ruff")
        assert "command" in config
        assert config["enabled"] is True

    def test_get_tool_config_missing(self):
        config = self.analysis.get_tool_config("python", "nonexistent")
        assert config == {}

    def test_get_tool_config_missing_language(self):
        config = self.analysis.get_tool_config("haskell", "ruff")
        assert config == {}

    # -- is_tool_enabled ------------------------------------------------

    def test_is_tool_enabled_true(self):
        assert self.analysis.is_tool_enabled("python", "ruff") is True

    def test_is_tool_enabled_false_for_missing(self):
        assert self.analysis.is_tool_enabled("python", "nope") is False

    # -- get_security_tools ---------------------------------------------

    def test_get_security_tools_python(self):
        tools = self.analysis.get_security_tools("python")
        assert "semgrep" in tools
        assert "sonarqube" in tools

    def test_get_security_tools_unknown_language(self):
        assert self.analysis.get_security_tools("cobol") == []

    # -- count_files_by_language ----------------------------------------

    def test_count_files_by_language(self, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        assert self.analysis.count_files_by_language(tmp_path, "python") == 2

    def test_count_files_by_language_unknown(self, tmp_path):
        assert self.analysis.count_files_by_language(tmp_path, "fortran") == 0

    # -- detect_architecture_patterns -----------------------------------

    def test_detect_architecture_patterns_mvc(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "views").mkdir()
        (tmp_path / "controllers").mkdir()
        # Need files not just dirs to match rglob
        (tmp_path / "models" / "x.py").touch()
        (tmp_path / "views" / "x.py").touch()
        (tmp_path / "controllers" / "x.py").touch()
        patterns = self.analysis.detect_architecture_patterns(tmp_path)
        assert patterns["mvc"] is True

    def test_detect_architecture_patterns_no_match(self, tmp_path):
        (tmp_path / "random_dir").mkdir()
        patterns = self.analysis.detect_architecture_patterns(tmp_path)
        assert patterns["mvc"] is False
        assert patterns["microservices"] is False

    # -- analyze_dependency_structure -----------------------------------

    def test_analyze_dependency_structure_python(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").touch()
        result = self.analysis.analyze_dependency_structure(tmp_path, "python")
        assert "module_count" in result
        assert "max_depth" in result

    def test_analyze_dependency_structure_javascript(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        result = self.analysis.analyze_dependency_structure(tmp_path, "javascript")
        assert "package_count" in result
        assert result["package_count"] == 1

    def test_analyze_dependency_structure_other_language(self, tmp_path):
        result = self.analysis.analyze_dependency_structure(tmp_path, "rust")
        assert result["circular_dependencies"] is False

    # -- severity_mappings exist ----------------------------------------

    def test_severity_mappings_populated(self):
        assert "python" in self.analysis.severity_mappings
        assert "javascript" in self.analysis.severity_mappings
        assert "go" in self.analysis.severity_mappings
        assert "rust" in self.analysis.severity_mappings


# ===================================================================
# ArchitecturePatternAnalyzer tests
# ===================================================================


class TestArchitecturePatternAnalyzer:
    def setup_method(self):
        self.recognizer = CodePatternRecognizer()
        self.analyzer = ArchitecturePatternAnalyzer(self.recognizer)

    def _graph_with_code(self, node_id: str, code: str, **extra) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(node_id, code=code, **extra)
        return g

    # -- detect_all_architecture_patterns --------------------------------

    def test_detect_all_empty_graph(self):
        g = nx.DiGraph()
        patterns = self.analyzer.detect_all_architecture_patterns(g)
        assert patterns == []

    # -- _detect_sidecar_pattern ----------------------------------------

    def test_detect_sidecar_explicit(self):
        g = self._graph_with_code("sidecar_node", "class SidecarHelper:\n    pass")
        patterns = self.analyzer._detect_sidecar_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "Sidecar"

    def test_detect_sidecar_no_match(self):
        g = self._graph_with_code("plain_node", "class PlainService:\n    pass")
        patterns = self.analyzer._detect_sidecar_pattern(g)
        assert len(patterns) == 0

    # -- _detect_ambassador_pattern -------------------------------------

    def test_detect_ambassador_explicit(self):
        g = self._graph_with_code(
            "ambassador_proxy", "class AmbassadorProxy:\n    ambassador logic"
        )
        patterns = self.analyzer._detect_ambassador_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "Ambassador"
        assert patterns[0].confidence == 0.9

    def test_detect_ambassador_proxy_keyword(self):
        g = self._graph_with_code("proxy_node", "def proxy_handler():\n    proxy request")
        patterns = self.analyzer._detect_ambassador_pattern(g)
        assert len(patterns) >= 1

    # -- _detect_adapter_pattern ----------------------------------------

    def test_detect_adapter_explicit(self):
        g = self._graph_with_code("adapter_node", "class DataAdapter:\n    pass")
        patterns = self.analyzer._detect_adapter_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "Adapter"

    def test_detect_adapter_structural(self):
        g = nx.DiGraph()
        g.add_node("adapter_cls", code="class Adapter:\n    pass", type="class")
        g.add_node("interface", code="class IService:\n    pass", type="interface")
        g.add_node("service", code="class Service:\n    pass", type="class")
        g.add_edge("adapter_cls", "interface", type="implements")
        g.add_edge("adapter_cls", "service", type="calls")
        patterns = self.analyzer._detect_adapter_pattern(g)
        # Should find both explicit (name) and structural
        structural = [p for p in patterns if "Inferred" in p.description]
        assert len(structural) >= 1

    # -- _detect_circuit_breaker_pattern --------------------------------

    def test_detect_circuit_breaker_explicit(self):
        g = self._graph_with_code("cb", "class CircuitBreaker:\n    circuitbreaker logic")
        patterns = self.analyzer._detect_circuit_breaker_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].confidence == 0.9

    def test_detect_circuit_breaker_library(self):
        g = self._graph_with_code("cb", "import resilience4j\ndef retry_handler(): pass")
        patterns = self.analyzer._detect_circuit_breaker_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].confidence == 0.85

    def test_detect_circuit_breaker_generic_fallback(self):
        g = self._graph_with_code("fb", "def fallback_handler():\n    pass")
        patterns = self.analyzer._detect_circuit_breaker_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].confidence == 0.7

    # -- _detect_service_mesh_pattern -----------------------------------

    def test_detect_service_mesh_explicit(self):
        g = self._graph_with_code("mesh_node", "class IstioMesh:\n    mesh config")
        patterns = self.analyzer._detect_service_mesh_pattern(g)
        assert any(p.pattern_type == "ServiceMesh" for p in patterns)

    def test_detect_service_mesh_control_and_data_planes(self):
        g = nx.DiGraph()
        g.add_node(
            "ctrl",
            code="class ControlPlane:\n    config manage orchestrate",
            type="service",
        )
        g.add_node(
            "data",
            code="class DataPlane:\n    proxy route traffic forward",
            type="service",
        )
        patterns = self.analyzer._detect_service_mesh_pattern(g)
        complete_mesh = [p for p in patterns if "control and data" in p.description.lower()]
        assert len(complete_mesh) >= 1
        assert complete_mesh[0].confidence == 0.85

    # -- _detect_api_gateway_pattern ------------------------------------

    def test_detect_api_gateway_explicit(self):
        g = self._graph_with_code("gw", "class ApiGateway:\n    apigateway routing")
        patterns = self.analyzer._detect_api_gateway_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].confidence == 0.9

    def test_detect_api_gateway_router_in_id(self):
        g = self._graph_with_code("router_main", "def router():\n    gateway logic")
        patterns = self.analyzer._detect_api_gateway_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].confidence == 0.85

    # -- Delegate methods -----------------------------------------------

    def test_is_likely_helper_delegates(self):
        with patch.object(self.analyzer._cloud_helpers, "is_likely_helper", return_value=True):
            assert self.analyzer._is_likely_helper({}, {}) is True

    def test_has_proxy_behavior_delegates(self):
        with patch.object(self.analyzer._cloud_helpers, "has_proxy_behavior", return_value=False):
            assert self.analyzer._has_proxy_behavior({}) is False


# ===================================================================
# PatternValidationScorer tests
# ===================================================================


class TestPatternValidationScorer:
    def setup_method(self):
        self.scorer = PatternValidationScorer()

    # -- initialize_ml_calculator ---------------------------------------

    def test_initialize_ml_no_config(self):
        self.scorer.initialize_ml_calculator(None)
        assert self.scorer.is_ml_enabled() is False

    def test_initialize_ml_no_settings(self):
        config = MagicMock(spec=RepoConfig)
        config.ml_confidence_settings = None
        self.scorer.initialize_ml_calculator(config)
        assert self.scorer.is_ml_enabled() is False

    def test_initialize_ml_disabled(self):
        config = MagicMock(spec=RepoConfig)
        config.ml_confidence_settings = {"enabled": False}
        self.scorer.initialize_ml_calculator(config)
        assert self.scorer.is_ml_enabled() is False

    def test_initialize_ml_enabled_success(self):
        config = MagicMock(spec=RepoConfig)
        config.ml_confidence_settings = {"enabled": True}
        with patch(
            "processing.architecture.pattern_validation_scoring.MLConfidenceCalculator"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            self.scorer.initialize_ml_calculator(config)
            assert self.scorer.is_ml_enabled() is True
            assert self.scorer.get_ml_calculator() is not None

    def test_initialize_ml_import_error(self):
        config = MagicMock(spec=RepoConfig)
        config.ml_confidence_settings = {"enabled": True}
        with patch(
            "processing.architecture.pattern_validation_scoring.MLConfidenceCalculator",
            side_effect=ImportError("no ml"),
        ):
            self.scorer.initialize_ml_calculator(config)
            assert self.scorer.is_ml_enabled() is False

    # -- adjust_confidence_with_ml --------------------------------------

    def test_adjust_confidence_ml_disabled(self):
        pattern = _make_pattern(confidence=0.8)
        result = self.scorer.adjust_confidence_with_ml(pattern, nx.DiGraph())
        assert result.confidence == 0.8

    def test_adjust_confidence_ml_success(self):
        self.scorer._ml_enabled = True
        self.scorer._ml_calculator = MagicMock()
        self.scorer._ml_calculator.calculate_confidence.return_value = 0.95
        pattern = _make_pattern(confidence=0.8)
        result = self.scorer.adjust_confidence_with_ml(pattern, nx.DiGraph())
        assert result.confidence == 0.95

    def test_adjust_confidence_ml_failure(self):
        self.scorer._ml_enabled = True
        self.scorer._ml_calculator = MagicMock()
        self.scorer._ml_calculator.calculate_confidence.side_effect = RuntimeError("fail")
        pattern = _make_pattern(confidence=0.8)
        result = self.scorer.adjust_confidence_with_ml(pattern, nx.DiGraph())
        assert result.confidence == 0.8  # Returns original

    # -- validate_pattern_components ------------------------------------

    def test_validate_components_all_exist(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        pattern = _make_pattern(components=["a", "b"])
        assert self.scorer.validate_pattern_components(pattern, g) is True

    def test_validate_components_missing(self):
        g = nx.DiGraph()
        g.add_node("a")
        pattern = _make_pattern(components=["a", "missing"])
        assert self.scorer.validate_pattern_components(pattern, g) is False

    def test_validate_components_empty(self):
        pattern = _make_pattern(components=[])
        assert self.scorer.validate_pattern_components(pattern, nx.DiGraph()) is False

    # -- calculate_pattern_cohesion -------------------------------------

    def test_cohesion_single_component(self):
        pattern = _make_pattern(components=["a"])
        assert self.scorer.calculate_pattern_cohesion(pattern, nx.DiGraph()) == 1.0

    def test_cohesion_fully_connected(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        pattern = _make_pattern(components=["a", "b"])
        assert self.scorer.calculate_pattern_cohesion(pattern, g) == 1.0

    def test_cohesion_no_connections(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        pattern = _make_pattern(components=["a", "b"])
        assert self.scorer.calculate_pattern_cohesion(pattern, g) == 0.0

    def test_cohesion_partial(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        pattern = _make_pattern(components=["a", "b"])
        cohesion = self.scorer.calculate_pattern_cohesion(pattern, g)
        assert 0.0 < cohesion < 1.0

    # -- calculate_pattern_coupling -------------------------------------

    def test_coupling_no_components(self):
        pattern = _make_pattern(components=[])
        assert self.scorer.calculate_pattern_coupling(pattern, nx.DiGraph()) == 0.0

    def test_coupling_no_external(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        pattern = _make_pattern(components=["a", "b"])
        assert self.scorer.calculate_pattern_coupling(pattern, g) == 0.0

    def test_coupling_all_external(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("ext")
        g.add_edge("a", "ext")
        pattern = _make_pattern(components=["a"])
        coupling = self.scorer.calculate_pattern_coupling(pattern, g)
        assert coupling == 1.0

    def test_coupling_mixed(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("ext")
        g.add_edge("a", "b")  # internal
        g.add_edge("a", "ext")  # external
        pattern = _make_pattern(components=["a", "b"])
        coupling = self.scorer.calculate_pattern_coupling(pattern, g)
        assert 0.0 < coupling < 1.0

    # -- score_pattern_quality ------------------------------------------

    def test_score_pattern_quality_basic(self):
        g = nx.DiGraph()
        g.add_node("a")
        pattern = _make_pattern(components=["a"], confidence=0.9)
        score = self.scorer.score_pattern_quality(pattern, g)
        assert 0.0 <= score <= 1.0

    def test_score_quality_bounded(self):
        g = nx.DiGraph()
        g.add_node("a")
        pattern = _make_pattern(components=["a"], confidence=1.0)
        score = self.scorer.score_pattern_quality(pattern, g)
        assert score <= 1.0

    # -- validate_pattern_structure -------------------------------------

    def test_validate_structure_valid(self):
        g = nx.DiGraph()
        g.add_node("a")
        pattern = _make_pattern(components=["a"], confidence=0.9)
        result = self.scorer.validate_pattern_structure(pattern, g)
        assert result["is_valid"] is True
        assert result["components_exist"] is True
        assert len(result["validation_errors"]) == 0

    def test_validate_structure_missing_components(self):
        g = nx.DiGraph()
        pattern = _make_pattern(components=["missing"])
        result = self.scorer.validate_pattern_structure(pattern, g)
        assert result["is_valid"] is False
        assert result["components_exist"] is False
        assert len(result["validation_errors"]) > 0

    def test_validate_structure_low_quality(self):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("ext1")
        g.add_node("ext2")
        g.add_node("ext3")
        g.add_edge("a", "ext1")
        g.add_edge("a", "ext2")
        g.add_edge("a", "ext3")
        pattern = _make_pattern(components=["a"], confidence=0.1)
        result = self.scorer.validate_pattern_structure(pattern, g)
        # Low confidence + high coupling = low quality
        if not result["is_valid"]:
            assert any("quality score" in e for e in result["validation_errors"])

    # -- adjust_confidence_by_structure ---------------------------------

    def test_adjust_confidence_by_structure_valid(self):
        g = nx.DiGraph()
        g.add_node("a")
        pattern = _make_pattern(components=["a"], confidence=0.8)
        adjusted = self.scorer.adjust_confidence_by_structure(pattern, g)
        assert 0.0 <= adjusted.confidence <= 1.0

    def test_adjust_confidence_by_structure_invalid(self):
        g = nx.DiGraph()
        pattern = _make_pattern(components=["missing"], confidence=0.8)
        adjusted = self.scorer.adjust_confidence_by_structure(pattern, g)
        # Invalid patterns get reduced confidence
        assert adjusted.confidence < 0.8

    # -- is_ml_enabled / get_ml_calculator ------------------------------

    def test_is_ml_enabled_default(self):
        assert self.scorer.is_ml_enabled() is False

    def test_get_ml_calculator_default(self):
        assert self.scorer.get_ml_calculator() is None


# ===================================================================
# PatternReportingUtils tests
# ===================================================================


class TestPatternReportingUtils:
    def setup_method(self):
        self.utils = PatternReportingUtils()

    # -- get_pattern_statistics -----------------------------------------

    def test_pattern_statistics(self):
        patterns = [
            _make_pattern(name="Singleton Pattern"),
            _make_pattern(name="Singleton Pattern"),
            _make_pattern(name="Factory Pattern"),
        ]
        stats = self.utils.get_pattern_statistics(patterns)
        assert stats["Singleton Pattern"] == 2
        assert stats["Factory Pattern"] == 1

    def test_pattern_statistics_empty(self):
        assert self.utils.get_pattern_statistics([]) == {}

    def test_pattern_statistics_none_name(self):
        pattern = _make_pattern(name=None)
        stats = self.utils.get_pattern_statistics([pattern])
        assert stats == {}

    # -- filter_patterns_by_confidence -----------------------------------

    def test_filter_by_confidence(self):
        patterns = [
            _make_pattern(confidence=0.9),
            _make_pattern(confidence=0.5),
            _make_pattern(confidence=0.8),
        ]
        filtered = self.utils.filter_patterns_by_confidence(patterns, 0.7)
        assert len(filtered) == 2

    def test_filter_by_confidence_default(self):
        patterns = [_make_pattern(confidence=0.6)]
        filtered = self.utils.filter_patterns_by_confidence(patterns)
        assert len(filtered) == 0

    # -- filter_patterns_by_type ----------------------------------------

    def test_filter_by_type(self):
        patterns = [
            _make_pattern(pattern_type="Singleton"),
            _make_pattern(pattern_type="Factory"),
            _make_pattern(pattern_type="Observer"),
        ]
        filtered = self.utils.filter_patterns_by_type(patterns, ["Singleton", "Observer"])
        assert len(filtered) == 2

    # -- group_patterns_by_type -----------------------------------------

    def test_group_by_type(self):
        patterns = [
            _make_pattern(pattern_type="A"),
            _make_pattern(pattern_type="B"),
            _make_pattern(pattern_type="A"),
        ]
        grouped = self.utils.group_patterns_by_type(patterns)
        assert len(grouped["A"]) == 2
        assert len(grouped["B"]) == 1

    # -- get_patterns_by_location ---------------------------------------

    def test_patterns_by_location(self):
        patterns = [
            _make_pattern(location="file1.py"),
            _make_pattern(location="file2.py"),
            _make_pattern(location="file1.py"),
        ]
        by_loc = self.utils.get_patterns_by_location(patterns)
        assert len(by_loc["file1.py"]) == 2
        assert len(by_loc["file2.py"]) == 1

    def test_patterns_by_location_none_excluded(self):
        pattern = _make_pattern(location=None)
        by_loc = self.utils.get_patterns_by_location([pattern])
        assert len(by_loc) == 0

    # -- get_confidence_distribution ------------------------------------

    def test_confidence_distribution(self):
        patterns = [
            _make_pattern(confidence=0.1),
            _make_pattern(confidence=0.3),
            _make_pattern(confidence=0.5),
            _make_pattern(confidence=0.7),
            _make_pattern(confidence=0.9),
        ]
        dist = self.utils.get_confidence_distribution(patterns)
        assert dist["0.0-0.2"] == 1
        assert dist["0.2-0.4"] == 1
        assert dist["0.4-0.6"] == 1
        assert dist["0.6-0.8"] == 1
        assert dist["0.8-1.0"] == 1

    def test_confidence_distribution_empty(self):
        dist = self.utils.get_confidence_distribution([])
        assert sum(dist.values()) == 0

    # -- get_pattern_component_stats ------------------------------------

    def test_component_stats(self):
        patterns = [
            _make_pattern(components=["a"]),
            _make_pattern(components=["a", "b", "c"]),
        ]
        stats = self.utils.get_pattern_component_stats(patterns)
        assert stats["total_patterns"] == 2
        assert stats["total_components"] == 4
        assert stats["avg_components_per_pattern"] == 2.0
        assert stats["max_components"] == 3
        assert stats["min_components"] == 1
        assert stats["patterns_with_single_component"] == 1
        assert stats["patterns_with_multiple_components"] == 1

    def test_component_stats_empty(self):
        stats = self.utils.get_pattern_component_stats([])
        assert stats["total_patterns"] == 0

    # -- get_top_patterns_by_confidence ---------------------------------

    def test_top_patterns(self):
        patterns = [
            _make_pattern(confidence=0.5),
            _make_pattern(confidence=0.9),
            _make_pattern(confidence=0.7),
        ]
        top = self.utils.get_top_patterns_by_confidence(patterns, limit=2)
        assert len(top) == 2
        assert top[0].confidence == 0.9
        assert top[1].confidence == 0.7

    # -- get_patterns_summary -------------------------------------------

    def test_patterns_summary(self):
        patterns = [
            _make_pattern(pattern_type="A", confidence=0.9, name="Pattern A"),
            _make_pattern(pattern_type="B", confidence=0.7, name="Pattern B"),
        ]
        summary = self.utils.get_patterns_summary(patterns)
        assert summary["overview"]["total_patterns"] == 2
        assert summary["overview"]["unique_types"] == 2
        assert summary["overview"]["avg_confidence"] == 0.8
        assert summary["overview"]["high_confidence_patterns"] == 1
        assert "A" in summary["by_type"]
        assert len(summary["top_patterns"]) == 2

    def test_patterns_summary_empty(self):
        summary = self.utils.get_patterns_summary([])
        assert summary["overview"]["total_patterns"] == 0
        assert summary["overview"]["avg_confidence"] == 0.0

    # -- generate_pattern_report ----------------------------------------

    def test_generate_pattern_report(self):
        patterns = [_make_pattern(confidence=0.85)]
        report = self.utils.generate_pattern_report(patterns)
        assert "Pattern Detection Report" in report
        assert "Total Patterns Detected: 1" in report
        assert "Component Statistics:" in report

    def test_generate_pattern_report_empty(self):
        report = self.utils.generate_pattern_report([])
        assert "Total Patterns Detected: 0" in report

    # -- export_patterns_to_dict ----------------------------------------

    def test_export_to_dict(self):
        patterns = [_make_pattern(pattern_type="X", confidence=0.7)]
        exported = self.utils.export_patterns_to_dict(patterns)
        assert len(exported) == 1
        assert exported[0]["pattern_type"] == "X"
        assert exported[0]["confidence"] == 0.7

    # -- find_pattern_overlaps ------------------------------------------

    def test_find_overlaps(self):
        patterns = [
            _make_pattern(components=["a", "b"]),
            _make_pattern(components=["b", "c"]),
            _make_pattern(components=["x", "y"]),
        ]
        overlaps = self.utils.find_pattern_overlaps(patterns)
        assert len(overlaps) == 1
        assert "b" in overlaps[0]["common_components"]

    def test_find_overlaps_none(self):
        patterns = [
            _make_pattern(components=["a"]),
            _make_pattern(components=["b"]),
        ]
        overlaps = self.utils.find_pattern_overlaps(patterns)
        assert len(overlaps) == 0


# ===================================================================
# CloudPatternHelpers tests
# ===================================================================


class TestCloudPatternHelpers:
    def setup_method(self):
        self.recognizer = CodePatternRecognizer()
        self.helpers = CloudPatternHelpers(self.recognizer)

    # -- is_likely_helper -----------------------------------------------

    def test_is_likely_helper_by_name(self):
        main_data = {"code": "class MainService:\n    " + "x" * 200}
        helper_data = {"code": "class X:\n    pass", "id": "logging_agent"}
        assert self.helpers.is_likely_helper(main_data, helper_data) is True

    def test_is_likely_helper_by_cross_cutting(self):
        main_data = {"code": "class MainService:\n    " + "x" * 200}
        helper_data = {"code": "class X:\n    log metrics trace", "id": "util"}
        assert self.helpers.is_likely_helper(main_data, helper_data) is True

    def test_is_likely_helper_too_large(self):
        main_data = {"code": "small"}
        helper_data = {"code": "much larger code body than the main", "id": "big"}
        assert self.helpers.is_likely_helper(main_data, helper_data) is False

    def test_is_likely_helper_no_indicators(self):
        main_data = {"code": "class MainService:\n    " + "x" * 200}
        helper_data = {"code": "class X:\n    unrelated stuff", "id": "something"}
        assert self.helpers.is_likely_helper(main_data, helper_data) is False

    # -- has_proxy_behavior ---------------------------------------------

    def test_has_proxy_behavior_true(self):
        node_data = {"code": "def handler():\n    forward(request)"}
        assert self.helpers.has_proxy_behavior(node_data) is True

    def test_has_proxy_behavior_false(self):
        node_data = {"code": "def handler():\n    return 42"}
        assert self.helpers.has_proxy_behavior(node_data) is False

    def test_has_proxy_behavior_multiple_indicators(self):
        node_data = {"code": "proxy route delegate transport"}
        assert self.helpers.has_proxy_behavior(node_data) is True

    # -- detect_kubernetes_mesh_patterns --------------------------------

    def test_k8s_mesh_no_match(self):
        g = nx.DiGraph()
        g.add_node("plain", code="class Plain:\n    pass")
        patterns = self.helpers.detect_kubernetes_mesh_patterns(g)
        assert len(patterns) == 0

    def test_k8s_mesh_match(self):
        g = nx.DiGraph()
        g.add_node(
            "k8s_mesh",
            code="apiVersion: apps/v1\nkind: deployment\nistio mesh proxy envoy",
        )
        patterns = self.helpers.detect_kubernetes_mesh_patterns(g)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "ServiceMesh"
        assert patterns[0].confidence == 0.9

    # -- detect_kubernetes_gateway_patterns -----------------------------

    def test_k8s_gateway_no_match(self):
        g = nx.DiGraph()
        g.add_node("plain", code="class Plain:\n    pass")
        patterns = self.helpers.detect_kubernetes_gateway_patterns(g)
        assert len(patterns) == 0

    def test_k8s_gateway_match(self):
        g = nx.DiGraph()
        g.add_node(
            "k8s_gw",
            code="kind: ingress\napiVersion: networking.k8s.io\nroute path host backend service",
        )
        patterns = self.helpers.detect_kubernetes_gateway_patterns(g)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "ApiGateway"
        assert patterns[0].confidence == 0.85

    # -- analyze_structural_patterns ------------------------------------

    def test_structural_patterns_sidecar(self):
        g = nx.DiGraph()
        g.add_node(
            "main_svc",
            code="class MainService:\n    " + "x" * 200,
            type="service",
        )
        g.add_node(
            "helper_svc",
            code="class X:\n    log metrics",
            type="service",
            id="log_agent",
        )
        g.add_edge("main_svc", "helper_svc", type="uses")
        patterns = self.helpers.analyze_structural_patterns(g, "main_svc", g.nodes["main_svc"])
        sidecar = [p for p in patterns if p.pattern_type == "Sidecar"]
        assert len(sidecar) >= 1

    def test_structural_patterns_ambassador(self):
        g = nx.DiGraph()
        g.add_node(
            "proxy_svc",
            code="def handler():\n    forward proxy route",
            type="proxy",
        )
        g.add_node("client1", code="client", type="client")
        g.add_node("client2", code="client", type="client")
        g.add_node("external", code="service", type="service")
        g.add_edge("client1", "proxy_svc")
        g.add_edge("client2", "proxy_svc")
        g.add_edge("proxy_svc", "external")
        patterns = self.helpers.analyze_structural_patterns(g, "proxy_svc", g.nodes["proxy_svc"])
        ambassador = [p for p in patterns if p.pattern_type == "Ambassador"]
        assert len(ambassador) >= 1

    def test_structural_patterns_gateway(self):
        g = nx.DiGraph()
        g.add_node(
            "gateway",
            code="def handle():\n    route auth transform filter",
            type="service",
        )
        g.add_node("client", code="client", type="client")
        g.add_edge("client", "gateway")
        for i in range(6):
            backend = f"backend_{i}"
            g.add_node(backend, code="service", type="service")
            g.add_edge("gateway", backend)
        patterns = self.helpers.analyze_structural_patterns(g, "gateway", g.nodes["gateway"])
        gw_patterns = [p for p in patterns if p.pattern_type == "ApiGateway"]
        assert len(gw_patterns) >= 1

    # -- find_mesh_control_plane_nodes ----------------------------------

    def test_find_control_plane_nodes(self):
        g = nx.DiGraph()
        g.add_node(
            "ctrl",
            code="class ControlPlane:\n    config manage orchestrate",
        )
        g.add_node("other", code="class Other:\n    pass")
        nodes = self.helpers.find_mesh_control_plane_nodes(g)
        assert "ctrl" in nodes
        assert "other" not in nodes

    def test_find_control_plane_none(self):
        g = nx.DiGraph()
        g.add_node("plain", code="class Plain:\n    pass")
        assert self.helpers.find_mesh_control_plane_nodes(g) == []

    # -- find_mesh_data_plane_nodes -------------------------------------

    def test_find_data_plane_nodes(self):
        g = nx.DiGraph()
        g.add_node(
            "dp",
            code="class DataPlane:\n    proxy route traffic forward",
        )
        g.add_node("other", code="class Other:\n    pass")
        nodes = self.helpers.find_mesh_data_plane_nodes(g)
        assert "dp" in nodes

    def test_find_data_plane_none(self):
        g = nx.DiGraph()
        g.add_node("plain", code="class Plain:\n    pass")
        assert self.helpers.find_mesh_data_plane_nodes(g) == []

    # -- _detect_structural_sidecar (no helpers found) ------------------

    def test_structural_sidecar_no_helpers(self):
        g = nx.DiGraph()
        g.add_node("svc", code="class Svc:\n    pass", type="service")
        patterns = self.helpers._detect_structural_sidecar(g, "svc", g.nodes["svc"])
        assert len(patterns) == 0

    # -- _detect_structural_ambassador no match -------------------------

    def test_structural_ambassador_no_clients(self):
        g = nx.DiGraph()
        g.add_node("proxy", code="forward proxy", type="proxy")
        g.add_node("ext", code="service", type="service")
        g.add_edge("proxy", "ext")
        # Only 0 in_neighbors -> no ambassador
        patterns = self.helpers._detect_structural_ambassador(g, "proxy", g.nodes["proxy"])
        assert len(patterns) == 0

    def test_structural_ambassador_one_client_not_enough(self):
        g = nx.DiGraph()
        g.add_node("proxy", code="forward proxy", type="proxy")
        g.add_node("client", code="client", type="client")
        g.add_node("ext", code="service", type="service")
        g.add_edge("client", "proxy")
        g.add_edge("proxy", "ext")
        # Only 1 client -> not enough for ambassador
        patterns = self.helpers._detect_structural_ambassador(g, "proxy", g.nodes["proxy"])
        assert len(patterns) == 0

    # -- _detect_structural_gateway no match ----------------------------

    def test_structural_gateway_not_enough_backends(self):
        g = nx.DiGraph()
        g.add_node("gw", code="route auth", type="service")
        g.add_node("backend", code="service", type="service")
        g.add_edge("gw", "backend")
        patterns = self.helpers._detect_structural_gateway(g, "gw", g.nodes["gw"])
        assert len(patterns) == 0

    def test_structural_gateway_no_routing_code(self):
        g = nx.DiGraph()
        g.add_node("gw", code="plain service no routing keywords", type="service")
        for i in range(6):
            g.add_node(f"b{i}", code="service", type="service")
            g.add_edge("gw", f"b{i}")
        patterns = self.helpers._detect_structural_gateway(g, "gw", g.nodes["gw"])
        assert len(patterns) == 0
