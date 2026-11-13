"""Round 4 tests for processing/quality_analysis modules.

Covers: http_quality_runner, fix_suggester, quality_runner, tool_registry,
report_generator, report_chart_generator, report_template_manager.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from models.testing_models import QualityIssue, QualityRequest, QualityResult

# ---------------------------------------------------------------------------
# SessionMonitor tests
# ---------------------------------------------------------------------------


class TestSessionMonitor:
    """Tests for SessionMonitor class."""

    def _make_monitor(self):
        from processing.quality_analysis.http_quality_runner import SessionMonitor

        return SessionMonitor()

    @pytest.mark.asyncio
    async def test_register_session_increments_stats(self):
        monitor = self._make_monitor()
        session = MagicMock()
        await monitor.register_session(session)
        assert monitor.stats.total_sessions_created == 1
        assert monitor.stats.active_sessions >= 0

    @pytest.mark.asyncio
    async def test_record_session_closed(self):
        monitor = self._make_monitor()
        await monitor.record_session_closed()
        assert monitor.stats.total_sessions_closed == 1

    @pytest.mark.asyncio
    async def test_record_session_error_categorises_errors(self):
        monitor = self._make_monitor()
        await monitor.record_session_error("session is closed")
        assert monitor.error_patterns["session_closed"] == 1
        await monitor.record_session_error("connection refused")
        assert monitor.error_patterns["connection_refused"] == 1
        await monitor.record_session_error("timeout waiting")
        assert monitor.error_patterns["timeout"] == 1
        await monitor.record_session_error("pool exhausted")
        assert monitor.error_patterns["connection_pool_issue"] == 1
        await monitor.record_session_error("HTTP 502 bad gateway")
        assert monitor.error_patterns["service_unavailable"] == 1
        await monitor.record_session_error("random error")
        assert monitor.error_patterns["other"] == 1

    @pytest.mark.asyncio
    async def test_record_session_error_trims_to_25(self):
        monitor = self._make_monitor()
        # Push 51 errors to trigger trim
        for i in range(51):
            await monitor.record_session_error(f"error {i}")
        assert len(monitor.stats.session_errors) == 25

    def test_get_health_summary(self):
        monitor = self._make_monitor()
        summary = monitor.get_health_summary()
        assert "active_sessions" in summary
        assert "error_patterns" in summary

    def test_categorize_error_503(self):
        monitor = self._make_monitor()
        assert monitor._categorize_error("HTTP 503 Service Unavailable") == "service_unavailable"


# ---------------------------------------------------------------------------
# HTTPQualityRunner tests
# ---------------------------------------------------------------------------


class TestHTTPQualityRunner:
    """Tests for HTTPQualityRunner class."""

    def _make_runner(self):
        from processing.quality_analysis.http_quality_runner import HTTPQualityRunner

        return HTTPQualityRunner()

    def test_init_creates_runner_with_defaults(self):
        runner = self._make_runner()
        assert runner.config is not None
        assert runner.monitor is not None

    def test_load_and_validate_config_defaults(self):
        runner = self._make_runner()
        assert runner.config.total_timeout == 3600
        assert runner.config.connect_timeout == 60
        assert runner.config.max_concurrent_connections == 50

    def test_load_and_validate_config_from_env(self):
        with patch.dict(
            os.environ,
            {
                "QUALITY_HTTP_TIMEOUT": "100",
                "QUALITY_MAX_CONNECTIONS": "10",
                "QUALITY_MAX_CONCURRENT_REQUESTS": "5",
            },
        ):
            runner = self._make_runner()
            assert runner.config.total_timeout == 100
            assert runner.config.max_concurrent_connections == 10

    def test_calculate_health_score_healthy(self):
        runner = self._make_runner()
        basic_health = {"status": "healthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 0, "total_created": 1},
            "service_connectivity": {"reachable": True, "response_time_ms": 100},
            "resource_usage": {"memory_usage_mb": 200},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        assert score == 100.0

    def test_calculate_health_score_unhealthy(self):
        runner = self._make_runner()
        basic_health = {"status": "unhealthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 5, "total_created": 10},
            "service_connectivity": {"reachable": False},
            "resource_usage": {"memory_usage_mb": 2000},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        assert score < 50

    def test_calculate_config_score_default(self):
        runner = self._make_runner()
        score = runner._calculate_config_score()
        # Default config should be reasonable
        assert 70 <= score <= 100

    def test_calculate_config_score_low_timeout(self):
        runner = self._make_runner()
        runner.config.total_timeout = 100
        score = runner._calculate_config_score()
        assert score <= 95  # Should be penalized

    def test_calculate_config_score_high_timeout(self):
        runner = self._make_runner()
        runner.config.total_timeout = 8000
        score = runner._calculate_config_score()
        assert score <= 90  # Should be penalized

    def test_calculate_config_score_low_retries(self):
        runner = self._make_runner()
        runner.config.max_retries = 1
        score = runner._calculate_config_score()
        assert score < 100

    def test_calculate_config_score_high_retries(self):
        runner = self._make_runner()
        runner.config.max_retries = 10
        score = runner._calculate_config_score()
        assert score < 100

    def test_create_failed_quality_result(self):
        runner = self._make_runner()
        start = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-123",
            start_time=start,
            language="python",
            error_message="Service down",
        )
        assert result.success is False
        assert result.error == "Service down"
        assert result.language == "python"

    def test_create_failed_quality_result_empty_error(self):
        runner = self._make_runner()
        start = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-123",
            start_time=start,
            language="python",
            error_message="",
        )
        assert "no specific error" in result.error.lower()

    def test_create_failed_quality_result_none_error(self):
        runner = self._make_runner()
        start = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-123",
            start_time=start,
            language="python",
            error_message=None,
        )
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.asyncio
    async def test_run_quality_analysis_disabled(self):
        with patch.dict(os.environ, {"ENABLE_QUALITY_ANALYSIS": "false"}):
            runner = self._make_runner()
            request = QualityRequest(
                container_name="test", language="python", code_content="print('hi')"
            )
            result = await runner.run_quality_analysis(request)
            assert result.success is False
            assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_close(self):
        runner = self._make_runner()
        await runner.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_get_resource_metrics_without_psutil(self):
        runner = self._make_runner()
        with patch("processing.quality_analysis.http_quality_runner.PSUTIL_AVAILABLE", False):
            result = await runner._get_resource_metrics()
            assert "error" in result

    @pytest.mark.asyncio
    async def test_get_performance_metrics_zero_created(self):
        runner = self._make_runner()
        result = await runner._get_performance_metrics()
        assert "session_efficiency_percent" in result
        assert result["session_efficiency_percent"] == 100.0


# ---------------------------------------------------------------------------
# create_quality_tool_runner factory
# ---------------------------------------------------------------------------


class TestCreateQualityToolRunnerFactory:
    def test_factory_function(self):
        from processing.quality_analysis.http_quality_runner import create_quality_tool_runner

        runner = create_quality_tool_runner(MagicMock())
        assert runner is not None


# ---------------------------------------------------------------------------
# AutoFixSuggester tests
# ---------------------------------------------------------------------------


class TestAutoFixSuggester:
    """Tests for AutoFixSuggester/FixSuggester."""

    def _make_suggester(self):
        from processing.quality_analysis.fix_suggester import AutoFixSuggester

        return AutoFixSuggester()

    def test_init(self):
        s = self._make_suggester()
        assert s.max_fixes_per_file == 50
        assert s.enable_experimental_fixes is False

    @pytest.mark.asyncio
    async def test_suggest_fixes_empty(self):
        s = self._make_suggester()
        result = await s.suggest_fixes([])
        assert result == []

    def test_group_issues_by_file(self):
        s = self._make_suggester()
        issue1 = QualityIssue(
            tool="ruff",
            severity="warning",
            message="unused import",
            file_path="a.py",
            line_number=1,
            column_number=None,
            rule_id="F401",
            suggested_fix=None,
            fix_confidence=None,
        )
        issue2 = QualityIssue(
            tool="ruff",
            severity="error",
            message="syntax error",
            file_path="b.py",
            line_number=2,
            column_number=None,
            rule_id="E999",
            suggested_fix=None,
            fix_confidence=None,
        )
        grouped = s._group_issues_by_file([issue1, issue2])
        assert "a.py" in grouped
        assert "b.py" in grouped
        assert len(grouped["a.py"]) == 1

    def test_calculate_fix_priority_critical_security(self):
        from processing.quality_analysis.fix_models import FixCategory, FixRule, FixStrategy

        s = self._make_suggester()
        issue = QualityIssue(
            tool="bandit",
            severity="critical",
            message="sql injection",
            file_path="x.py",
            line_number=1,
            column_number=None,
            rule_id="B608",
            suggested_fix=None,
            fix_confidence=None,
        )
        rule = FixRule(
            rule_id="B608",
            tool="bandit",
            rule_pattern=r".*",
            category=FixCategory.SECURITY,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
            requires_review=True,
        )
        priority = s._calculate_fix_priority(issue, rule)
        assert priority == 1  # max priority for critical + security

    def test_calculate_fix_priority_info_formatting(self):
        from processing.quality_analysis.fix_models import FixCategory, FixRule, FixStrategy

        s = self._make_suggester()
        issue = QualityIssue(
            tool="ruff-format",
            severity="info",
            message="format",
            file_path="x.py",
            line_number=1,
            column_number=None,
            rule_id="format",
            suggested_fix=None,
            fix_confidence=None,
        )
        rule = FixRule(
            rule_id="format",
            tool="ruff-format",
            rule_pattern=r".*",
            category=FixCategory.FORMATTING,
            strategy=FixStrategy.SUGGESTED,
            confidence=0.5,
            requires_review=False,
        )
        priority = s._calculate_fix_priority(issue, rule)
        assert priority == 10

    def test_filter_applicable_fixes_no_filter(self):
        s = self._make_suggester()
        fixes = [MagicMock(strategy="automatic", confidence=0.5, breaking_change=False)]
        result = s._filter_applicable_fixes(fixes, filter_by_confidence=False)
        assert result == fixes

    def test_filter_applicable_fixes_with_confidence(self):
        from processing.quality_analysis.fix_models import FixStrategy

        s = self._make_suggester()
        good_fix = MagicMock(strategy=FixStrategy.AUTOMATIC, confidence=0.95, breaking_change=False)
        bad_fix = MagicMock(strategy=FixStrategy.AUTOMATIC, confidence=0.3, breaking_change=False)
        breaking_fix = MagicMock(
            strategy=FixStrategy.AUTOMATIC, confidence=0.95, breaking_change=True
        )
        result = s._filter_applicable_fixes(
            [good_fix, bad_fix, breaking_fix], filter_by_confidence=True
        )
        assert good_fix in result
        assert bad_fix not in result
        assert breaking_fix not in result

    def test_create_empty_result(self):
        s = self._make_suggester()
        result = s._create_empty_result("app-001", 1000.0)
        assert result.successful_applications == 0

    def test_create_error_result(self):
        s = self._make_suggester()
        mock_fix = MagicMock(fix_id="fix-1")
        result = s._create_error_result("app-002", [mock_fix], 1000.0, "boom")
        assert result.failed_applications == 1
        assert "boom" in result.errors

    @pytest.mark.asyncio
    async def test_validate_fixes_empty_code(self):
        s = self._make_suggester()
        result = await s.validate_fixes("original", "", language="python")
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_fixes_shorter_code_warning(self):
        s = self._make_suggester()
        original = "x" * 200
        fixed = "y" * 10
        result = await s.validate_fixes(original, fixed, language="python")
        assert any("shorter" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_apply_line_based_fix(self):
        s = self._make_suggester()
        content = "line1\nline2\nline3"
        result = await s._apply_line_based_fix(content, 2, "fixed_line2")
        assert "fixed_line2" in result

    @pytest.mark.asyncio
    async def test_apply_line_based_fix_out_of_range(self):
        s = self._make_suggester()
        content = "line1\nline2"
        result = await s._apply_line_based_fix(content, 999, "nope")
        assert result == content

    @pytest.mark.asyncio
    async def test_validate_fix_syntax_no_code(self):
        s = self._make_suggester()
        fix = MagicMock(fixed_code=None, file_path="test.py")
        result = await s._validate_fix_syntax(fix)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_fix_syntax_unknown_lang(self):
        s = self._make_suggester()
        fix = MagicMock(fixed_code="code", file_path="test.rb")
        result = await s._validate_fix_syntax(fix)
        assert result is True

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_empty(self):
        s = self._make_suggester()
        result = await s.suggest_bulk_fixes([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_apply_automated_fixes_empty_after_filter(self):
        s = self._make_suggester()
        result = await s.apply_automated_fixes([], filter_by_confidence=True)
        assert result.total_fixes_attempted == 0

    def test_fix_application_context_str(self):
        from processing.quality_analysis.fix_suggester import FixApplicationContext

        ctx = FixApplicationContext(
            applicable_fixes=[],
            dry_run=False,
            create_backup=True,
            application_id="id-1",
            start_time=0.0,
        )
        assert "id-1" in str(ctx)


# ---------------------------------------------------------------------------
# create_auto_fix_suggester factory
# ---------------------------------------------------------------------------


class TestCreateAutoFixSuggesterFactory:
    def test_factory_default(self):
        from processing.quality_analysis.fix_suggester import create_auto_fix_suggester

        s = create_auto_fix_suggester()
        assert s.max_fixes_per_file == 50

    def test_factory_with_env(self):
        from processing.quality_analysis.fix_suggester import create_auto_fix_suggester

        with patch.dict(
            os.environ, {"ENABLE_EXPERIMENTAL_FIXES": "true", "MAX_FIXES_PER_FILE": "10"}
        ):
            s = create_auto_fix_suggester()
            assert s.enable_experimental_fixes is True
            assert s.max_fixes_per_file == 10


# ---------------------------------------------------------------------------
# QualityToolRunner tests
# ---------------------------------------------------------------------------


class TestQualityToolRunner:
    """Tests for QualityToolRunner."""

    def _make_runner(self):
        from processing.quality_analysis.quality_runner import QualityToolRunner

        mock_docker = MagicMock()
        return QualityToolRunner(mock_docker)

    def test_detect_language_from_request(self):
        runner = self._make_runner()
        req = QualityRequest(container_name="test", language="JavaScript")
        assert runner._detect_language(req) == "javascript"

    def test_detect_language_from_path(self):
        runner = self._make_runner()
        req = QualityRequest(container_name="test", code_path="/tmp/test.ts")
        assert runner._detect_language(req) == "typescript"

    def test_detect_language_default(self):
        runner = self._make_runner()
        req = QualityRequest(container_name="test")
        assert runner._detect_language(req) == "python"

    def test_get_file_extension(self):
        runner = self._make_runner()
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("unknown") == ".py"

    def test_calculate_quality_score_no_issues(self):
        runner = self._make_runner()
        assert runner._calculate_quality_score([], {}) == 10.0

    def test_calculate_quality_score_with_issues(self):
        runner = self._make_runner()
        issues = [
            QualityIssue(
                tool="ruff",
                severity="error",
                message="err",
                file_path="f.py",
                line_number=1,
                column_number=None,
                rule_id="E001",
                suggested_fix=None,
                fix_confidence=None,
            ),
            QualityIssue(
                tool="ruff",
                severity="warning",
                message="warn",
                file_path="f.py",
                line_number=2,
                column_number=None,
                rule_id="W001",
                suggested_fix=None,
                fix_confidence=None,
            ),
        ]
        score = runner._calculate_quality_score(issues, {})
        assert 0 <= score <= 10
        assert score < 10  # There are issues

    def test_map_ruff_severity(self):
        runner = self._make_runner()
        assert runner._map_ruff_severity("F811") == "error"
        assert runner._map_ruff_severity("W605") == "warning"
        assert runner._map_ruff_severity("I001") == "info"

    def test_select_tools_with_filter(self):
        runner = self._make_runner()
        available = runner.tool_registry.list_available_tools("python")
        selected = runner._select_tools("python", ["ruff", "nonexistent"])
        for t in selected:
            assert t in available

    def test_get_target_filename_from_path(self):
        runner = self._make_runner()
        req = QualityRequest(container_name="test", code_path="/tmp/mycode.py")
        assert runner._get_target_filename(req, "python") == "mycode.py"

    def test_get_target_filename_default(self):
        runner = self._make_runner()
        req = QualityRequest(container_name="test", code_content="print(1)")
        assert runner._get_target_filename(req, "python") == "main.py"

    def test_initialize_container_images(self):
        runner = self._make_runner()
        images = runner._initialize_container_images()
        assert "python" in images
        assert "javascript" in images


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def _make_registry(self):
        from processing.quality_analysis.tool_registry import ToolRegistry

        return ToolRegistry()

    def test_list_supported_languages(self):
        r = self._make_registry()
        langs = r.list_supported_languages()
        assert "python" in langs

    def test_get_tool_info_nonexistent(self):
        r = self._make_registry()
        assert r.get_tool_info("python", "nonexistent_tool") is None

    def test_get_installation_commands(self):
        r = self._make_registry()
        cmds = r.get_installation_commands("python")
        assert isinstance(cmds, list)

    def test_get_tools_for_language(self):
        r = self._make_registry()
        tools = r.get_tools_for_language("python")
        assert isinstance(tools, dict)

    def test_execute_tool_not_found(self):
        r = self._make_registry()
        result = r.execute_tool("python", "nonexistent_xyzzy", "/tmp")
        assert "error" in result

    def test_validate_tool_availability_nonexistent(self):
        r = self._make_registry()
        assert r.validate_tool_availability("python", "nonexistent_xyzzy") is False

    def test_get_configuration_templates(self):
        r = self._make_registry()
        templates = r.get_configuration_templates("python")
        assert isinstance(templates, dict)


# ---------------------------------------------------------------------------
# ToolRegistry singleton and convenience functions
# ---------------------------------------------------------------------------


class TestToolRegistrySingleton:
    def test_get_tool_registry(self):
        from processing.quality_analysis.tool_registry import get_tool_registry

        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_convenience_get_tools_for_language(self):
        from processing.quality_analysis.tool_registry import get_tools_for_language

        tools = get_tools_for_language("python")
        assert isinstance(tools, dict)

    def test_convenience_get_installation_commands(self):
        from processing.quality_analysis.tool_registry import get_installation_commands

        cmds = get_installation_commands("python")
        assert isinstance(cmds, list)

    def test_convenience_list_supported_languages(self):
        from processing.quality_analysis.tool_registry import list_supported_languages

        langs = list_supported_languages()
        assert "python" in langs


# ---------------------------------------------------------------------------
# ReportChartGenerator tests
# ---------------------------------------------------------------------------


class TestReportChartGenerator:
    def _make_result(self, **kwargs):
        defaults = dict(
            analysis_id="a1",
            success=True,
            language="python",
            tools_results={},
            overall_score=85.0,
            total_issues=0,
            issues=[],
            execution_time=1.0,
            created_at=datetime.now(UTC),
        )
        defaults.update(kwargs)
        return QualityResult(**defaults)

    def _make_generator(self):
        from processing.quality_analysis.report_chart_generator import ReportChartGenerator

        return ReportChartGenerator()

    @pytest.mark.asyncio
    async def test_generate_severity_chart(self):
        gen = self._make_generator()
        issue = QualityIssue(
            tool="ruff",
            severity="warning",
            message="x",
            file_path="f.py",
            line_number=1,
            column_number=None,
            rule_id="W",
            suggested_fix=None,
            fix_confidence=None,
        )
        r = self._make_result(issues=[issue], total_issues=1)
        chart = await gen.generate_severity_chart([r])
        assert chart["type"] == "pie"
        assert chart["data"]["values"][2] == 1  # warning index

    @pytest.mark.asyncio
    async def test_generate_score_distribution_chart(self):
        gen = self._make_generator()
        results = [
            self._make_result(overall_score=95.0),
            self._make_result(overall_score=72.0),
            self._make_result(overall_score=50.0),
        ]
        chart = await gen.generate_score_distribution_chart(results)
        assert chart["type"] == "bar"

    @pytest.mark.asyncio
    async def test_generate_trends_chart_with_historical(self):
        gen = self._make_generator()
        historical = [
            {"timestamp": "2024-01-01", "overall_score": 80},
            {"timestamp": "2024-02-01", "overall_score": 85},
        ]
        chart = await gen.generate_trends_chart([], historical_data=historical)
        assert chart["type"] == "line"
        assert len(chart["data"]["labels"]) == 2

    @pytest.mark.asyncio
    async def test_generate_language_breakdown_chart(self):
        gen = self._make_generator()
        results = [
            self._make_result(language="python"),
            self._make_result(language="python"),
            self._make_result(language="javascript"),
        ]
        chart = await gen.generate_language_breakdown_chart(results)
        assert chart["type"] == "pie"

    @pytest.mark.asyncio
    async def test_generate_execution_time_chart(self):
        gen = self._make_generator()
        r = self._make_result(
            tools_results={"ruff": {"execution_time": 2.5}, "ty": {"execution_time": 5.0}}
        )
        chart = await gen.generate_execution_time_chart([r])
        assert chart["type"] == "bar"

    @pytest.mark.asyncio
    async def test_generate_issue_trends_chart(self):
        gen = self._make_generator()
        issue = QualityIssue(
            tool="t",
            severity="critical",
            message="m",
            file_path="f.py",
            line_number=1,
            column_number=None,
            rule_id="C",
            suggested_fix=None,
            fix_confidence=None,
        )
        r = self._make_result(issues=[issue])
        chart = await gen.generate_issue_trends_chart([r])
        assert chart["type"] == "line"

    def test_get_available_chart_types(self):
        gen = self._make_generator()
        types = gen.get_available_chart_types()
        assert "severity_distribution" in types

    def test_get_chart_color_scheme(self):
        gen = self._make_generator()
        assert gen.get_chart_color_scheme("severity_distribution")[0] == "#e74c3c"
        assert gen.get_chart_color_scheme("quality_score_distribution")[0] == "#27ae60"
        assert len(gen.get_chart_color_scheme("unknown")) == 6


# ---------------------------------------------------------------------------
# ReportTemplateManager tests
# ---------------------------------------------------------------------------


class TestReportTemplateManager:
    def _make_manager(self):
        from processing.quality_analysis.report_template_manager import ReportTemplateManager

        return ReportTemplateManager()

    def test_init(self):
        m = self._make_manager()
        assert m is not None

    def test_get_css_styles(self):
        m = self._make_manager()
        css = m._get_css_styles()
        assert "font-family" in css

    def test_get_available_templates(self):
        m = self._make_manager()
        templates = m.get_available_templates()
        assert "html_report" in templates

    @pytest.mark.asyncio
    async def test_render_basic_html(self):
        m = self._make_manager()
        result = QualityResult(
            analysis_id="a1",
            success=True,
            language="python",
            tools_results={},
            overall_score=90.0,
            total_issues=0,
            issues=[],
            execution_time=1.0,
            created_at=datetime.now(UTC),
        )
        metrics = MagicMock(overall_score=90.0, total_issues=0, total_files_analyzed=1)
        context = MagicMock(
            title="Test Report",
            description="desc",
            generation_time=datetime.now(UTC),
            metrics=metrics,
        )
        html = m._render_basic_html([result], context)
        assert "Test Report" in html
        assert "90.0" in html

    @pytest.mark.asyncio
    async def test_render_custom_template_without_jinja(self):
        from processing.quality_analysis.report_template_manager import ReportTemplateManager

        with patch("processing.quality_analysis.report_template_manager.HAS_JINJA2", False):
            m = ReportTemplateManager()
            result = await m.render_custom_template("Hello {{ name }}", name="World")
            # Without jinja2, returns template as-is
            assert "{{ name }}" in result


# ---------------------------------------------------------------------------
# QualityReportGenerator tests
# ---------------------------------------------------------------------------


class TestQualityReportGenerator:
    def test_init(self):
        from processing.quality_analysis.report_generator import QualityReportGenerator

        gen = QualityReportGenerator(output_directory="/tmp/test_reports")
        assert str(gen.output_handler.output_directory) == "/tmp/test_reports"

    def test_get_available_themes(self):
        from processing.quality_analysis.report_generator import QualityReportGenerator

        gen = QualityReportGenerator()
        themes = gen.get_available_themes()
        assert isinstance(themes, list)

    def test_get_supported_formats(self):
        from processing.quality_analysis.report_generator import QualityReportGenerator

        gen = QualityReportGenerator()
        formats = gen.get_supported_formats()
        assert isinstance(formats, list)

    def test_get_chart_types(self):
        from processing.quality_analysis.report_generator import QualityReportGenerator

        gen = QualityReportGenerator()
        chart_types = gen.get_chart_types()
        assert "severity_distribution" in chart_types

    def test_factory_function(self):
        from processing.quality_analysis.report_generator import create_quality_report_generator

        gen = create_quality_report_generator()
        assert gen is not None
