#!/usr/bin/env python3
"""Unit tests for quality analysis core modules.

Covers:
- quality_models.py (QualityServiceRequest, QualityServiceIssue, AppToolResult, QualityResult)
- quality_config.py (QualityConfig, detect_language, create_temp_dir)
- quality_metrics.py (QualityMetricsCalculator)
- quality_analysis_core.py (QualityAnalysisCore)
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub services.quality.tool_integration — this module does not exist in the
# codebase but is referenced by quality_analysis_core.py at import time.
# ---------------------------------------------------------------------------
if "services.quality.tool_integration" not in sys.modules:
    _ti_module = types.ModuleType("services.quality.tool_integration")
    _ti_module.tool_executor = MagicMock()
    sys.modules["services.quality.tool_integration"] = _ti_module

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from services.quality.quality_config import (
    QualityConfig,
    create_temp_dir,
    detect_language,
)
from services.quality.quality_metrics import QualityMetricsCalculator
from services.quality.quality_models import (
    AppToolResult,
    QualityResult,
    QualityServiceIssue,
    QualityServiceRequest,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_issue(
    tool: str = "ruff",
    severity: str = "error",
    message: str = "some error",
    fixable: bool = False,
) -> QualityServiceIssue:
    return QualityServiceIssue(
        tool=tool,
        severity=severity,
        message=message,
        line=1,
        column=0,
        rule_id="E001",
        fixable=fixable,
    )


def _make_tool_result(
    tool: str = "ruff",
    success: bool = True,
    issues: list[QualityServiceIssue] | None = None,
    execution_time: float = 1.0,
) -> AppToolResult:
    return AppToolResult(
        tool=tool,
        success=success,
        issues=issues or [],
        execution_time=execution_time,
    )


# ===================================================================
# quality_models.py tests
# ===================================================================


class TestQualityServiceRequest:
    """Tests for QualityServiceRequest model."""

    def test_defaults(self):
        req = QualityServiceRequest()
        assert req.code_path is None
        assert req.code_content is None
        assert req.language is None
        assert req.tools == []
        assert req.fix_issues is False
        assert req.custom_rules is None

    def test_with_code_content(self):
        req = QualityServiceRequest(
            code_content="print('hello')",
            language="python",
            tools=["ruff", "ruff-format"],
        )
        assert req.code_content == "print('hello')"
        assert req.language == "python"
        assert req.tools == ["ruff", "ruff-format"]

    def test_with_code_path(self):
        req = QualityServiceRequest(code_path="/tmp/test.py")
        assert req.code_path == "/tmp/test.py"

    def test_with_fix_issues(self):
        req = QualityServiceRequest(fix_issues=True)
        assert req.fix_issues is True

    def test_with_custom_rules(self):
        rules = {"max_line_length": 120}
        req = QualityServiceRequest(custom_rules=rules)
        assert req.custom_rules == rules


class TestQualityServiceIssue:
    """Tests for QualityServiceIssue model."""

    def test_minimal(self):
        issue = QualityServiceIssue(tool="ruff", severity="error", message="bad code")
        assert issue.tool == "ruff"
        assert issue.severity == "error"
        assert issue.message == "bad code"
        assert issue.line is None
        assert issue.column is None
        assert issue.rule_id is None
        assert issue.fixable is False

    def test_full(self):
        issue = QualityServiceIssue(
            tool="eslint",
            severity="warning",
            message="unused var",
            line=10,
            column=5,
            rule_id="no-unused-vars",
            fixable=True,
        )
        assert issue.line == 10
        assert issue.column == 5
        assert issue.rule_id == "no-unused-vars"
        assert issue.fixable is True


class TestAppToolResult:
    """Tests for AppToolResult model."""

    def test_defaults(self):
        result = AppToolResult(tool="ruff-format", success=True, execution_time=0.5)
        assert result.tool == "ruff-format"
        assert result.success is True
        assert result.issues == []
        assert result.execution_time == 0.5
        assert result.output == ""
        assert result.error == ""

    def test_with_issues_and_error(self):
        issue = _make_issue()
        result = AppToolResult(
            tool="ruff",
            success=False,
            issues=[issue],
            execution_time=2.0,
            output="raw output",
            error="tool failed",
        )
        assert len(result.issues) == 1
        assert result.error == "tool failed"
        assert result.output == "raw output"


class TestQualityResult:
    """Tests for QualityResult model."""

    def test_minimal(self):
        result = QualityResult(
            analysis_id="abc-123",
            success=True,
            language="python",
            overall_score=95.0,
            total_issues=2,
            execution_time=3.0,
            created_at="2026-01-01T00:00:00",
        )
        assert result.analysis_id == "abc-123"
        assert result.success is True
        assert result.language == "python"
        assert result.overall_score == 95.0
        assert result.total_issues == 2
        assert result.tools_results == {}
        assert result.error is None

    def test_with_error(self):
        result = QualityResult(
            analysis_id="abc-456",
            success=False,
            language="unknown",
            overall_score=0.0,
            total_issues=0,
            execution_time=0.1,
            created_at="2026-01-01T00:00:00",
            error="analysis failed",
        )
        assert result.error == "analysis failed"


# ===================================================================
# quality_config.py tests
# ===================================================================


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_python_extension(self):
        assert detect_language("main.py") == "python"

    def test_javascript_extension(self):
        assert detect_language("app.js") == "javascript"

    def test_jsx_extension(self):
        assert detect_language("component.jsx") == "javascript"

    def test_typescript_extension(self):
        assert detect_language("app.ts") == "typescript"

    def test_tsx_extension(self):
        assert detect_language("component.tsx") == "typescript"

    def test_unknown_extension_defaults_python(self):
        assert detect_language("file.txt") == "python"

    def test_content_detection_python(self):
        assert detect_language("", "def hello():\n    pass") == "python"

    def test_content_detection_javascript(self):
        assert detect_language("", "const x = 42;") == "javascript"

    def test_content_detection_typescript(self):
        assert detect_language("", "interface Foo: string\nfunction bar() {}") == "typescript"

    def test_empty_defaults_python(self):
        assert detect_language("") == "python"

    def test_file_path_takes_priority_over_content(self):
        # Even if content looks like JS, file extension wins
        assert detect_language("test.py", "const x = 10;") == "python"


class TestQualityConfig:
    """Tests for QualityConfig class."""

    def setup_method(self):
        self.cfg = QualityConfig()

    def test_default_values(self):
        assert self.cfg.max_analysis_time == 3600
        assert self.cfg.quality_cache_ttl == 7200
        assert self.cfg.enable_autofix is False
        assert self.cfg.test_workspace == "/app/test_workspace"
        assert self.cfg.quality_cache_dir == "/app/quality_cache"

    def test_get_tool_timeout_known(self):
        assert self.cfg.get_tool_timeout("ruff") == 300
        assert self.cfg.get_tool_timeout("ty") == 900
        assert self.cfg.get_tool_timeout("ruff-format") == 300
        assert self.cfg.get_tool_timeout("bandit") == 900

    def test_get_tool_timeout_unknown(self):
        # Unknown tool should return default timeout
        assert self.cfg.get_tool_timeout("nonexistent_tool") == 900

    def test_get_tools_for_language_python(self):
        tools = self.cfg.get_tools_for_language("python")
        assert "ruff-format" in tools
        assert "ruff" in tools
        assert "ty" in tools
        assert "bandit" in tools
        assert "vulture" in tools
        assert "radon" in tools
        assert "ruff" in tools

    def test_get_tools_for_language_javascript(self):
        tools = self.cfg.get_tools_for_language("javascript")
        assert "eslint" in tools
        assert "prettier" in tools

    def test_get_tools_for_language_typescript(self):
        tools = self.cfg.get_tools_for_language("typescript")
        assert "eslint" in tools
        assert "prettier" in tools
        assert "tsc" in tools

    def test_get_tools_for_language_unknown(self):
        tools = self.cfg.get_tools_for_language("ruby")
        assert tools == {}

    def test_get_tool_config_existing(self):
        tool_cfg = self.cfg.get_tool_config("python", "ruff-format")
        assert tool_cfg is not None
        assert "command" in tool_cfg
        assert "description" in tool_cfg

    def test_get_tool_config_nonexistent(self):
        assert self.cfg.get_tool_config("python", "nonexistent") is None
        assert self.cfg.get_tool_config("ruby", "nonexistent") is None

    def test_get_supported_languages(self):
        languages = self.cfg.get_supported_languages()
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages

    def test_is_tool_available(self):
        assert self.cfg.is_tool_available("python", "ruff") is True
        assert self.cfg.is_tool_available("python", "eslint") is False
        assert self.cfg.is_tool_available("javascript", "eslint") is True

    def test_get_service_info(self):
        info = self.cfg.get_service_info()
        assert info["service"] == "quality-analyzer"
        assert info["version"] == "1.0.0"
        assert "python" in info["supported_languages"]
        assert "multi_language_analysis" in info["features"]
        assert "configuration" in info
        assert info["configuration"]["max_analysis_time"] == 3600


class TestCreateTempDir:
    """Tests for create_temp_dir async function."""

    async def test_creates_directory_with_configs(self, tmp_path):
        with patch("services.quality.quality_config.QUALITY_CACHE_DIR", str(tmp_path)):
            result = await create_temp_dir("test-analysis-123")

        assert result.exists()
        assert result.is_dir()
        assert result.name == "test-analysis-123"

        # Should create ESLint config
        eslint_path = result / ".eslintrc.json"
        assert eslint_path.exists()
        eslint_cfg = json.loads(eslint_path.read_text())
        assert "env" in eslint_cfg
        assert "rules" in eslint_cfg

        # Should create TypeScript config
        ts_path = result / "tsconfig.json"
        assert ts_path.exists()
        ts_cfg = json.loads(ts_path.read_text())
        assert "compilerOptions" in ts_cfg

    async def test_idempotent(self, tmp_path):
        with patch("services.quality.quality_config.QUALITY_CACHE_DIR", str(tmp_path)):
            result1 = await create_temp_dir("same-id")
            result2 = await create_temp_dir("same-id")
        assert result1 == result2
        assert result1.exists()


# ===================================================================
# quality_metrics.py tests
# ===================================================================


class TestQualityMetricsCalculator:
    """Tests for QualityMetricsCalculator."""

    def setup_method(self):
        self.calc = QualityMetricsCalculator()

    # -- calculate_quality_score --

    def test_quality_score_empty(self):
        assert self.calc.calculate_quality_score({}) == 0.0

    def test_quality_score_no_issues(self):
        results = {"ruff": _make_tool_result("ruff", success=True, issues=[])}
        score = self.calc.calculate_quality_score(results)
        assert score == 100.0

    def test_quality_score_with_errors(self):
        issues = [_make_issue(severity="error") for _ in range(3)]
        results = {"ruff": _make_tool_result("ruff", success=True, issues=issues)}
        score = self.calc.calculate_quality_score(results)
        # 100 - 3*10 = 70
        assert score == 70.0

    def test_quality_score_with_warnings(self):
        issues = [_make_issue(severity="warning") for _ in range(4)]
        results = {"ruff": _make_tool_result("ruff", success=True, issues=issues)}
        score = self.calc.calculate_quality_score(results)
        # 100 - 4*5 = 80
        assert score == 80.0

    def test_quality_score_with_info(self):
        issues = [_make_issue(severity="info") for _ in range(5)]
        results = {"ruff": _make_tool_result("ruff", success=True, issues=issues)}
        score = self.calc.calculate_quality_score(results)
        # 100 - 5*2 = 90
        assert score == 90.0

    def test_quality_score_floors_at_zero(self):
        issues = [_make_issue(severity="error") for _ in range(15)]
        results = {"ruff": _make_tool_result("ruff", success=True, issues=issues)}
        score = self.calc.calculate_quality_score(results)
        # 100 - 15*10 = -50 => clamped to 0
        assert score == 0.0

    def test_quality_score_skips_failed_tools(self):
        results = {
            "ruff": _make_tool_result("ruff", success=False, issues=[]),
            "ruff-format": _make_tool_result("ruff-format", success=True, issues=[]),
        }
        score = self.calc.calculate_quality_score(results)
        # Only ruff-format counts, which has 0 issues => 100
        assert score == 100.0

    def test_quality_score_all_failed(self):
        results = {
            "ruff": _make_tool_result("ruff", success=False),
            "ruff-format": _make_tool_result("ruff-format", success=False),
        }
        score = self.calc.calculate_quality_score(results)
        assert score == 0.0

    def test_quality_score_multiple_tools_averaged(self):
        results = {
            "ruff": _make_tool_result(
                "ruff",
                success=True,
                issues=[_make_issue(severity="error")],
            ),
            "ruff-format": _make_tool_result("ruff-format", success=True, issues=[]),
        }
        score = self.calc.calculate_quality_score(results)
        # ruff: 100-10=90, ruff-format: 100 => avg = 95
        assert score == 95.0

    # -- calculate_total_issues --

    def test_total_issues_empty(self):
        assert self.calc.calculate_total_issues({}) == 0

    def test_total_issues(self):
        results = {
            "ruff": _make_tool_result("ruff", issues=[_make_issue(), _make_issue()]),
            "ruff-format": _make_tool_result("ruff-format", issues=[_make_issue()]),
        }
        assert self.calc.calculate_total_issues(results) == 3

    # -- calculate_issue_breakdown --

    def test_issue_breakdown(self):
        results = {
            "tool1": _make_tool_result(
                "tool1",
                issues=[
                    _make_issue(severity="error"),
                    _make_issue(severity="warning"),
                    _make_issue(severity="info"),
                    _make_issue(severity="convention"),
                ],
            ),
        }
        breakdown = self.calc.calculate_issue_breakdown(results)
        assert breakdown["error"] == 1
        assert breakdown["warning"] == 1
        assert breakdown["info"] == 1
        assert breakdown["other"] == 1

    def test_issue_breakdown_empty(self):
        breakdown = self.calc.calculate_issue_breakdown({})
        assert breakdown == {"error": 0, "warning": 0, "info": 0, "other": 0}

    # -- calculate_tool_success_rate --

    def test_success_rate_empty(self):
        assert self.calc.calculate_tool_success_rate({}) == 0.0

    def test_success_rate_all_success(self):
        results = {
            "a": _make_tool_result(success=True),
            "b": _make_tool_result(success=True),
        }
        assert self.calc.calculate_tool_success_rate(results) == 100.0

    def test_success_rate_partial(self):
        results = {
            "a": _make_tool_result(success=True),
            "b": _make_tool_result(success=False),
        }
        assert self.calc.calculate_tool_success_rate(results) == 50.0

    # -- calculate_fixable_issues_count --

    def test_fixable_issues_count(self):
        results = {
            "tool1": _make_tool_result(
                issues=[
                    _make_issue(fixable=True),
                    _make_issue(fixable=False),
                    _make_issue(fixable=True),
                ]
            ),
        }
        assert self.calc.calculate_fixable_issues_count(results) == 2

    def test_fixable_issues_count_none(self):
        results = {"tool1": _make_tool_result(issues=[_make_issue(fixable=False)])}
        assert self.calc.calculate_fixable_issues_count(results) == 0

    # -- calculate_execution_stats --

    def test_execution_stats_empty(self):
        assert self.calc.calculate_execution_stats({}) == {}

    def test_execution_stats(self):
        results = {
            "a": _make_tool_result(execution_time=2.0),
            "b": _make_tool_result(execution_time=4.0),
        }
        stats = self.calc.calculate_execution_stats(results)
        assert stats["total_execution_time"] == 6.0
        assert stats["average_execution_time"] == 3.0
        assert stats["min_execution_time"] == 2.0
        assert stats["max_execution_time"] == 4.0
        assert stats["tools_count"] == 2

    # -- _calculate_quality_grade --

    def test_grade_a(self):
        assert self.calc._calculate_quality_grade(95.0) == "A"
        assert self.calc._calculate_quality_grade(90.0) == "A"

    def test_grade_b(self):
        assert self.calc._calculate_quality_grade(85.0) == "B"
        assert self.calc._calculate_quality_grade(80.0) == "B"

    def test_grade_c(self):
        assert self.calc._calculate_quality_grade(75.0) == "C"
        assert self.calc._calculate_quality_grade(70.0) == "C"

    def test_grade_d(self):
        assert self.calc._calculate_quality_grade(65.0) == "D"
        assert self.calc._calculate_quality_grade(60.0) == "D"

    def test_grade_f(self):
        assert self.calc._calculate_quality_grade(59.0) == "F"
        assert self.calc._calculate_quality_grade(0.0) == "F"

    # -- generate_quality_summary --

    def test_quality_summary_no_issues(self):
        results = {"ruff": _make_tool_result(success=True, issues=[])}
        summary = self.calc.generate_quality_summary(results)
        assert summary["overall_score"] == 100.0
        assert summary["quality_grade"] == "A"
        assert summary["total_issues"] == 0
        assert summary["fixable_issues"] == 0
        assert summary["success_rate"] == 100.0
        assert "recommendations" in summary
        assert any("Excellent" in r or "No quality issues" in r for r in summary["recommendations"])

    def test_quality_summary_with_errors(self):
        issues = [_make_issue(severity="error") for _ in range(6)]
        results = {"ruff": _make_tool_result(success=True, issues=issues)}
        summary = self.calc.generate_quality_summary(results)
        assert summary["total_issues"] == 6
        assert summary["issue_breakdown"]["error"] == 6
        assert any("High number of errors" in r for r in summary["recommendations"])

    def test_quality_summary_with_fixable(self):
        issues = [_make_issue(fixable=True)]
        results = {"ruff": _make_tool_result(success=True, issues=issues)}
        summary = self.calc.generate_quality_summary(results)
        assert summary["fixable_issues"] == 1
        assert any("automatically fixed" in r for r in summary["recommendations"])

    def test_quality_summary_failed_tools(self):
        results = {"ruff": _make_tool_result(success=False)}
        summary = self.calc.generate_quality_summary(results)
        assert any("failed to execute" in r.lower() for r in summary["recommendations"])

    # -- _generate_recommendations --

    def test_recommendations_bandit_security(self):
        bandit_issues = [_make_issue(tool="bandit", severity="warning")]
        results = {"bandit": _make_tool_result(tool="bandit", success=True, issues=bandit_issues)}
        breakdown = self.calc.calculate_issue_breakdown(results)
        fixable = self.calc.calculate_fixable_issues_count(results)
        recs = self.calc._generate_recommendations(results, breakdown, fixable)
        assert any("Security scan" in r for r in recs)

    def test_recommendations_radon_complexity(self):
        radon_issues = [
            _make_issue(tool="radon", severity="warning", message="High complexity in func"),
        ]
        results = {"radon": _make_tool_result(tool="radon", success=True, issues=radon_issues)}
        breakdown = self.calc.calculate_issue_breakdown(results)
        fixable = self.calc.calculate_fixable_issues_count(results)
        recs = self.calc._generate_recommendations(results, breakdown, fixable)
        assert any("high complexity" in r.lower() for r in recs)

    def test_recommendations_moderate_issues(self):
        issues = [_make_issue(severity="warning") for _ in range(8)]
        results = {"tool": _make_tool_result(success=True, issues=issues)}
        breakdown = self.calc.calculate_issue_breakdown(results)
        fixable = 0
        recs = self.calc._generate_recommendations(results, breakdown, fixable)
        assert any("Moderate" in r for r in recs)


# ===================================================================
# quality_analysis_core.py tests
# ===================================================================


class TestQualityAnalysisCore:
    """Tests for QualityAnalysisCore."""

    def setup_method(self):
        from services.quality.quality_analysis_core import QualityAnalysisCore

        self.core = QualityAnalysisCore()

    def test_generate_analysis_id(self):
        aid = self.core.generate_analysis_id()
        assert isinstance(aid, str)
        assert len(aid) > 0

    async def test_cleanup_temp_dir(self, tmp_path):
        sub = tmp_path / "cleanup_test"
        sub.mkdir()
        (sub / "file.txt").write_text("data")
        assert sub.exists()

        await self.core.cleanup_temp_dir(sub)
        assert not sub.exists()

    async def test_cleanup_temp_dir_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        # Should not raise
        await self.core.cleanup_temp_dir(nonexistent)

    async def test_validate_request_no_code(self):
        req = QualityServiceRequest()
        with pytest.raises(ValueError, match="Either 'code_path' or 'code_content'"):
            await self.core.validate_request(req)

    async def test_validate_request_nonexistent_path(self, tmp_path):
        req = QualityServiceRequest(code_path=str(tmp_path / "nope.py"))
        with pytest.raises(ValueError, match="does not exist"):
            await self.core.validate_request(req)

    async def test_validate_request_unsupported_language(self, tmp_path):
        # Create a real file so path validation passes
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        req = QualityServiceRequest(code_path=str(f), language="ruby")
        with pytest.raises(ValueError, match="Unsupported language"):
            await self.core.validate_request(req)

    async def test_validate_request_invalid_tools(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        req = QualityServiceRequest(code_path=str(f), language="python", tools=["nonexistent_tool"])
        with pytest.raises(ValueError, match="Invalid tools"):
            await self.core.validate_request(req)

    async def test_validate_request_valid_content(self):
        req = QualityServiceRequest(code_content="print('hi')", language="python")
        # Should not raise
        await self.core.validate_request(req)

    async def test_prepare_analysis_file_content(self, tmp_path):
        req = QualityServiceRequest(code_content="x = 1\n")
        file_path = await self.core._prepare_analysis_file(req, tmp_path, "python")
        assert file_path.suffix == ".py"
        assert file_path.read_text() == "x = 1\n"

    async def test_prepare_analysis_file_js(self, tmp_path):
        req = QualityServiceRequest(code_content="const x = 1;")
        file_path = await self.core._prepare_analysis_file(req, tmp_path, "javascript")
        assert file_path.suffix == ".js"

    async def test_prepare_analysis_file_ts(self, tmp_path):
        req = QualityServiceRequest(code_content="let x: number = 1;")
        file_path = await self.core._prepare_analysis_file(req, tmp_path, "typescript")
        assert file_path.suffix == ".ts"

    async def test_prepare_analysis_file_copy_path(self, tmp_path):
        # Create source file
        src = tmp_path / "source.py"
        src.write_text("hello = 'world'")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        req = QualityServiceRequest(code_path=str(src))
        file_path = await self.core._prepare_analysis_file(req, dest_dir, "python")
        assert file_path.name == "source.py"
        assert file_path.read_text() == "hello = 'world'"

    async def test_prepare_analysis_file_no_code_raises(self, tmp_path):
        req = QualityServiceRequest()
        with pytest.raises(ValueError, match="Either 'code_path' or 'code_content'"):
            await self.core._prepare_analysis_file(req, tmp_path, "python")

    async def test_analyze_code_quality_success(self, tmp_path):
        """Test the main analyze_code_quality method with mocked dependencies."""
        tool_result = _make_tool_result("ruff", success=True, issues=[])

        with (
            patch.object(
                self.core,
                "generate_analysis_id",
                return_value="test-id-123",
            ),
            patch(
                "services.quality.quality_analysis_core.create_temp_dir",
                new_callable=AsyncMock,
                return_value=tmp_path,
            ),
            patch.object(
                self.core,
                "_prepare_analysis_file",
                new_callable=AsyncMock,
                return_value=tmp_path / "analysis.py",
            ),
            patch.object(
                self.core,
                "_execute_quality_tools",
                new_callable=AsyncMock,
                return_value={"ruff": tool_result},
            ),
            patch.object(
                self.core,
                "cleanup_temp_dir",
                new_callable=AsyncMock,
            ),
        ):
            req = QualityServiceRequest(code_content="x = 1", language="python")
            result = await self.core.analyze_code_quality(req)

        assert result.analysis_id == "test-id-123"
        assert result.success is True
        assert result.language == "python"
        assert result.overall_score == 100.0
        assert result.error is None

    async def test_analyze_code_quality_no_tools(self, tmp_path):
        """Test analyze_code_quality when no tools are available for language."""
        with (
            patch(
                "services.quality.quality_analysis_core.create_temp_dir",
                new_callable=AsyncMock,
                return_value=tmp_path,
            ),
            patch.object(
                self.core,
                "_prepare_analysis_file",
                new_callable=AsyncMock,
                return_value=tmp_path / "analysis.py",
            ),
            patch.object(
                self.core,
                "cleanup_temp_dir",
                new_callable=AsyncMock,
            ),
            patch.object(
                self.core.config,
                "get_tools_for_language",
                return_value={},
            ),
        ):
            req = QualityServiceRequest(code_content="x = 1", language="python")
            result = await self.core.analyze_code_quality(req)

        assert result.success is False
        assert "No quality tools" in (result.error or "")

    async def test_analyze_code_quality_exception(self, tmp_path):
        """Test analyze_code_quality handles unexpected exceptions."""
        with (
            patch(
                "services.quality.quality_analysis_core.create_temp_dir",
                new_callable=AsyncMock,
                side_effect=RuntimeError("disk full"),
            ),
            patch.object(
                self.core,
                "cleanup_temp_dir",
                new_callable=AsyncMock,
            ),
        ):
            req = QualityServiceRequest(code_content="x = 1", language="python")
            result = await self.core.analyze_code_quality(req)

        assert result.success is False
        assert "disk full" in (result.error or "")

    async def test_get_analysis_summary(self):
        results = {"ruff": _make_tool_result(success=True)}
        summary = await self.core.get_analysis_summary(results)
        assert "overall_score" in summary
        assert "quality_grade" in summary

    async def test_execute_quality_tools_parallel(self):
        """Test _execute_quality_tools runs tools and collects results."""
        expected_result = _make_tool_result("ruff", success=True)
        self.core.tool_executor.run_quality_tool = AsyncMock(return_value=expected_result)

        tools_to_run = ["ruff"]
        available_tools = {"ruff": {"command": ["ruff"], "description": "lint"}}

        results = await self.core._execute_quality_tools(
            tools_to_run, available_tools, "/tmp/test.py", "python"
        )

        assert "ruff" in results
        assert results["ruff"].success is True

    async def test_execute_quality_tools_exception_in_tool(self):
        """Test _execute_quality_tools handles per-tool exceptions from gather."""
        # Make the tool raise an exception
        self.core.tool_executor.run_quality_tool = AsyncMock(
            side_effect=RuntimeError("tool crashed")
        )

        tools_to_run = ["ruff"]
        available_tools = {"ruff": {"command": ["ruff"], "description": "lint"}}

        results = await self.core._execute_quality_tools(
            tools_to_run, available_tools, "/tmp/test.py", "python"
        )

        assert "ruff" in results
        assert results["ruff"].success is False
        assert "tool crashed" in results["ruff"].error
