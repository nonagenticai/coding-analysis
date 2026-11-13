"""Unit tests for quality analysis modules.

Covers:
- http_quality_runner.py (SessionMonitor, HTTPQualityRunner)
- fix_suggester.py (AutoFixSuggester, FixApplicationContext)
- quality_runner.py (QualityToolRunner)
- java_quality_analyzer.py (JavaQualityAnalyzer)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree.ElementTree import Element, SubElement

import pytest

# Now import the modules under test
from models.testing_models import QualityIssue, QualityRequest, QualityResult
from processing.quality_analysis.fix_models import (
    FixApplicationResult,
    FixCategory,
    FixRule,
    FixStrategy,
    QualityAnalysisResolveSuggestions,
)
from processing.quality_analysis.fix_suggester import (
    AutoFixSuggester,
    FixApplicationContext,
    create_auto_fix_suggester,
)
from processing.quality_analysis.http_quality_runner import (
    HTTPQualityRunner,
    SessionMonitor,
    create_quality_tool_runner,
)
from processing.quality_analysis.java_quality_analyzer import JavaQualityAnalyzer
from processing.quality_analysis.quality_runner import QualityToolRunner
from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quality_issue(**overrides):
    defaults = {
        "tool": "ruff",
        "severity": "warning",
        "message": "Missing docstring",
        "file_path": "test.py",
        "line_number": 1,
        "column_number": 0,
        "rule_id": "D103",
        "suggested_fix": None,
        "fix_confidence": None,
    }
    defaults.update(overrides)
    return QualityIssue(**defaults)


def _make_quality_request(**overrides):
    defaults = {
        "container_name": "test-container",
        "code": "print('hello')",
        "language": "python",
        "tools": ["ruff"],
    }
    defaults.update(overrides)
    return QualityRequest(**defaults)


def _make_quality_result(**overrides):
    defaults = {
        "analysis_id": "test-123",
        "success": True,
        "overall_score": 8.0,
        "total_issues": 0,
        "issues": [],
        "tools_results": {},
        "execution_time": 1.0,
        "created_at": datetime.now(UTC),
        "language": "python",
    }
    defaults.update(overrides)
    return QualityResult(**defaults)


def _make_fix_suggestion(**overrides):
    defaults = {
        "fix_id": "fix-1",
        "issue_id": None,
        "strategy": FixStrategy.AUTOMATIC,
        "category": FixCategory.CODE_STYLE,
        "confidence": 0.9,
        "priority": 3,
        "tool": "ruff",
        "rule_id": "C0111",
        "severity": "warning",
        "title": "Add docstring",
        "description": "Add missing docstring",
        "file_path": "test.py",
        "line_number": 1,
        "requires_review": False,
        "breaking_change": False,
    }
    defaults.update(overrides)
    return QualityAnalysisResolveSuggestions(**defaults)


def _make_quality_tool(**overrides):
    defaults = {
        "name": "ruff",
        "category": ToolCategory.LINTER,
        "command": ["ruff", "--output-format=json"],
        "output_format": "json",
        "description": "Python linter",
        "timeout": 30,
    }
    defaults.update(overrides)
    return QualityTool(**defaults)


# ===========================================================================
# SessionMonitor tests
# ===========================================================================


class TestSessionMonitor:
    """Tests for SessionMonitor."""

    async def test_register_session(self):
        monitor = SessionMonitor()
        session = MagicMock()
        await monitor.register_session(session)
        assert monitor.stats.total_sessions_created == 1
        assert monitor.stats.active_sessions == 1

    async def test_record_session_closed(self):
        monitor = SessionMonitor()
        await monitor.record_session_closed()
        assert monitor.stats.total_sessions_closed == 1

    async def test_record_session_error(self):
        monitor = SessionMonitor()
        await monitor.record_session_error("test error")
        assert monitor.stats.failed_sessions == 1
        assert len(monitor.stats.session_errors) == 1

    async def test_record_session_error_trims_list(self):
        monitor = SessionMonitor()
        for i in range(55):
            await monitor.record_session_error(f"error {i}")
        # After 50 the list is trimmed to 25
        assert len(monitor.stats.session_errors) <= 30

    async def test_categorize_error_session_closed(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Session is closed") == "session_closed"

    async def test_categorize_error_connection_refused(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Connection refused") == "connection_refused"

    async def test_categorize_error_timeout(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Read timeout") == "timeout"

    async def test_categorize_error_pool(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Connection pool exhausted") == "connection_pool_issue"

    async def test_categorize_error_502(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("HTTP 502 Bad Gateway") == "service_unavailable"

    async def test_categorize_error_503(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("HTTP 503 Service Unavailable") == "service_unavailable"

    async def test_categorize_error_other(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Unknown issue") == "other"

    async def test_get_health_summary(self):
        monitor = SessionMonitor()
        session = MagicMock()
        await monitor.register_session(session)
        await monitor.record_session_error("test error")

        summary = monitor.get_health_summary()
        assert summary["active_sessions"] == 1
        assert summary["total_created"] == 1
        assert summary["failed_sessions"] == 1
        assert summary["recent_errors"] == 1
        assert "timeout" not in summary["error_patterns"]


# ===========================================================================
# HTTPQualityRunner tests
# ===========================================================================


class TestHTTPQualityRunner:
    """Tests for HTTPQualityRunner."""

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_init_defaults(self):
        runner = HTTPQualityRunner()
        # service_url is sourced from Settings; conftest stub starts with http.
        assert runner.config.service_url.startswith("http")
        assert runner.cache_manager is None

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_load_and_validate_config_env(self):
        # service_url is sourced from cached Settings; patch the field directly.
        from config.settings import get_settings

        s = get_settings()
        original_url = s.QUALITY_ANALYSIS_SERVICE_URL
        try:
            object.__setattr__(s, "QUALITY_ANALYSIS_SERVICE_URL", "http://localhost:9000")
            with patch.dict(
                "os.environ",
                {
                    "QUALITY_HTTP_TIMEOUT": "600",
                    "QUALITY_HTTP_CONNECT_TIMEOUT": "30",
                    "QUALITY_MAX_CONNECTIONS": "20",
                    "QUALITY_MAX_CONCURRENT_REQUESTS": "10",
                    "QUALITY_MAX_CONCURRENT_BATCHES": "5",
                    "QUALITY_CONNECTION_RETRY_ATTEMPTS": "2",
                    "QUALITY_CONNECTION_KEEPALIVE": "60",
                    "QUALITY_MAX_REQUESTS_PER_SESSION": "500",
                },
            ):
                runner = HTTPQualityRunner()
                assert runner.config.service_url == "http://localhost:9000"
                assert runner.config.total_timeout == 600
                assert runner.config.max_concurrent_connections == 20
                assert runner.config.max_concurrent_requests == 10
        finally:
            object.__setattr__(s, "QUALITY_ANALYSIS_SERVICE_URL", original_url)

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_create_session(self):
        runner = HTTPQualityRunner()
        with patch("processing.quality_analysis.http_quality_runner.TCPConnector"):
            with patch("processing.quality_analysis.http_quality_runner.ClientSession") as mock_cs:
                mock_session = MagicMock()
                mock_session.closed = False
                mock_cs.return_value = mock_session
                session = await runner._create_session()
                assert session == mock_session
                assert runner.monitor.stats.total_sessions_created == 1

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_ensure_session_creates_new(self):
        runner = HTTPQualityRunner()
        with patch.object(runner, "_create_session", new_callable=AsyncMock) as mock_create:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_create.return_value = mock_session
            session = await runner._ensure_session()
            assert session == mock_session
            mock_create.assert_awaited_once()

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_ensure_session_reuses_existing(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_session.closed = False
        runner._session_manager.session = mock_session
        runner._session_manager.created_at = datetime.now(UTC)
        runner._session_manager.request_count = 0

        session = await runner._ensure_session()
        assert session == mock_session

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_ensure_session_replaces_closed(self):
        runner = HTTPQualityRunner()
        old_session = MagicMock()
        old_session.closed = True
        runner._session_manager.session = old_session
        runner._session_manager.created_at = datetime.now(UTC)

        with patch.object(runner, "_create_session", new_callable=AsyncMock) as mock_create:
            new_session = MagicMock()
            new_session.closed = False
            mock_create.return_value = new_session
            session = await runner._ensure_session()
            assert session == new_session

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_cleanup_session(self):
        runner = HTTPQualityRunner()
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()
        await runner._cleanup_session(session)
        session.close.assert_awaited_once()

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_cleanup_session_already_closed(self):
        runner = HTTPQualityRunner()
        session = MagicMock()
        session.closed = True
        await runner._cleanup_session(session)
        # Should not try to close it again

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_close(self):
        runner = HTTPQualityRunner()
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()
        runner._session_manager.session = session
        await runner.close()
        assert runner._session_manager.session is None

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_health_check_with_retry_success(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner._health_check_with_retry(max_retries=1, delay=0)
            assert result is True

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_health_check_with_retry_failure(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner._health_check_with_retry(max_retries=1, delay=0)
            assert result is False

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_run_quality_analysis_disabled(self):
        runner = HTTPQualityRunner()
        request = _make_quality_request()
        with patch.dict("os.environ", {"ENABLE_QUALITY_ANALYSIS": "false"}):
            result = await runner.run_quality_analysis(request)
            assert result.success is False
            assert "disabled" in result.error.lower()

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_run_quality_analysis_unhealthy(self):
        runner = HTTPQualityRunner()
        request = _make_quality_request()
        with patch.dict("os.environ", {"ENABLE_QUALITY_ANALYSIS": "true"}, clear=False):
            with patch.object(
                runner, "_health_check_with_retry", new_callable=AsyncMock, return_value=False
            ):
                result = await runner.run_quality_analysis(request)
                assert result.success is False
                assert "unavailable" in result.error.lower()

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_run_quality_analysis_success(self):
        runner = HTTPQualityRunner()
        request = _make_quality_request()
        expected_result = _make_quality_result(success=True, overall_score=9.0)
        with patch.dict("os.environ", {"ENABLE_QUALITY_ANALYSIS": "true"}, clear=False):
            with patch.object(
                runner, "_health_check_with_retry", new_callable=AsyncMock, return_value=True
            ):
                with patch.object(
                    runner, "_ensure_session", new_callable=AsyncMock, return_value=MagicMock()
                ):
                    with patch.object(
                        runner,
                        "_execute_request_with_retry",
                        new_callable=AsyncMock,
                        return_value=expected_result,
                    ):
                        result = await runner.run_quality_analysis(request)
                        assert result.success is True

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_run_single_tool_success(self):
        runner = HTTPQualityRunner()
        tool_result_data = _make_quality_result(
            success=True, tools_results={"ruff": {"issues": [], "score": 10}}
        )
        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=MagicMock()
        ):
            with patch.object(
                runner,
                "_execute_request_with_retry",
                new_callable=AsyncMock,
                return_value=tool_result_data,
            ):
                result = await runner.run_single_tool("ruff", "python", "x=1")
                assert result == {"issues": [], "score": 10}

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_run_single_tool_not_in_results(self):
        runner = HTTPQualityRunner()
        tool_result_data = _make_quality_result(success=True, tools_results={})
        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=MagicMock()
        ):
            with patch.object(
                runner,
                "_execute_request_with_retry",
                new_callable=AsyncMock,
                return_value=tool_result_data,
            ):
                result = await runner.run_single_tool("ty", "python", "x=1")
                assert result["success"] is False

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_get_available_tools(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"available_tools": {"python": ["ruff", "ty"]}})
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner.get_available_tools()
            assert "python" in result

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_get_available_tools_filtered(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"available_tools": {"python": ["ruff"], "javascript": ["eslint"]}}
        )
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner.get_available_tools(language="python")
            assert "python" in result
            assert "javascript" not in result

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_create_failed_quality_result(self):
        runner = HTTPQualityRunner()
        start = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="a1", start_time=start, language="python", error_message="boom"
        )
        assert result.success is False
        assert result.error == "boom"
        assert result.language == "python"

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_create_failed_quality_result_empty_error(self):
        runner = HTTPQualityRunner()
        result = runner._create_failed_quality_result(
            analysis_id="a2",
            start_time=datetime.now(UTC),
            language="python",
            error_message="",
        )
        assert "no specific error" in result.error.lower()

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_calculate_health_score_healthy(self):
        runner = HTTPQualityRunner()
        basic = {"status": "healthy"}
        diag = {
            "session_health": {"failed_sessions": 0, "total_created": 1},
            "service_connectivity": {"reachable": True, "response_time_ms": 50},
            "resource_usage": {"memory_usage_mb": 200},
        }
        score = runner._calculate_health_score(basic, diag)
        assert score == 100.0

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_calculate_health_score_unhealthy(self):
        runner = HTTPQualityRunner()
        basic = {"status": "unhealthy"}
        diag = {
            "session_health": {"failed_sessions": 5, "total_created": 10},
            "service_connectivity": {"reachable": False},
            "resource_usage": {"memory_usage_mb": 2000},
        }
        score = runner._calculate_health_score(basic, diag)
        assert score < 50

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_calculate_config_score_default(self):
        runner = HTTPQualityRunner()
        score = runner._calculate_config_score()
        assert 0 <= score <= 100

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_get_resource_metrics_psutil(self):
        runner = HTTPQualityRunner()
        # psutil is not installed, so mock the psutil module access
        mock_mem = MagicMock()
        mock_mem.rss = 100 * 1024 * 1024
        mock_process = MagicMock()
        mock_process.memory_info.return_value = mock_mem
        mock_process.cpu_percent.return_value = 5.0
        mock_process.open_files.return_value = []
        mock_process.net_connections.return_value = []
        mock_process.num_threads.return_value = 4
        mock_psutil = MagicMock()
        mock_psutil.Process.return_value = mock_process
        with (
            patch("processing.quality_analysis.http_quality_runner.PSUTIL_AVAILABLE", True),
            patch.dict("sys.modules", {"psutil": mock_psutil}),
            patch(
                "processing.quality_analysis.http_quality_runner.psutil", mock_psutil, create=True
            ),
        ):
            metrics = await runner._get_resource_metrics()
            assert "memory_usage_mb" in metrics

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_get_resource_metrics_no_psutil(self):
        runner = HTTPQualityRunner()
        with patch("processing.quality_analysis.http_quality_runner.PSUTIL_AVAILABLE", False):
            metrics = await runner._get_resource_metrics()
            assert "error" in metrics

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_get_performance_metrics(self):
        runner = HTTPQualityRunner()
        metrics = await runner._get_performance_metrics()
        assert "session_efficiency_percent" in metrics

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_health_check_healthy(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "ok"})
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner.health_check()
            assert result["status"] == "healthy"

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    async def test_health_check_unhealthy(self):
        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch.object(
            runner, "_ensure_session", new_callable=AsyncMock, return_value=mock_session
        ):
            result = await runner.health_check()
            assert result["status"] == "unhealthy"

    @patch("processing.quality_analysis.http_quality_runner.PROMETHEUS_AVAILABLE", False)
    def test_create_quality_tool_runner_factory(self):
        runner = create_quality_tool_runner(MagicMock())
        assert isinstance(runner, HTTPQualityRunner)


# ===========================================================================
# AutoFixSuggester tests
# ===========================================================================


class TestFixApplicationContext:
    def test_str(self):
        ctx = FixApplicationContext(
            applicable_fixes=[_make_fix_suggestion()],
            dry_run=False,
            create_backup=True,
            application_id="app-1",
            start_time=0.0,
        )
        assert "app-1" in str(ctx)
        assert "1" in str(ctx)


class TestAutoFixSuggester:
    """Tests for AutoFixSuggester."""

    def test_init(self):
        suggester = AutoFixSuggester()
        assert suggester.max_fixes_per_file == 50
        assert suggester.enable_experimental_fixes is False

    async def test_suggest_fixes_empty(self):
        suggester = AutoFixSuggester()
        result = await suggester.suggest_fixes([])
        assert result == []

    async def test_suggest_fixes_generates_suggestions(self):
        suggester = AutoFixSuggester()
        issues = [
            _make_quality_issue(tool="ruff", rule_id="D103", severity="warning"),
        ]
        suggestions = await suggester.suggest_fixes(issues)
        assert len(suggestions) >= 1

    async def test_suggest_fixes_limits_per_file(self):
        suggester = AutoFixSuggester(max_fixes_per_file=2)
        issues = [
            _make_quality_issue(tool="ruff", rule_id="D103", file_path="a.py", severity="warning"),
            _make_quality_issue(tool="ruff", rule_id="D103", file_path="a.py", severity="error"),
            _make_quality_issue(tool="ruff", rule_id="D103", file_path="a.py", severity="info"),
        ]
        suggestions = await suggester.suggest_fixes(issues)
        # Each issue may produce 1+ suggestion; filter by file
        a_suggestions = [s for s in suggestions if s.file_path == "a.py"]
        assert len(a_suggestions) <= 2

    async def test_suggest_fixes_sorts_by_priority(self):
        suggester = AutoFixSuggester()
        issues = [
            _make_quality_issue(tool="ruff", rule_id="D103", severity="info"),
            _make_quality_issue(tool="ruff", rule_id="D103", severity="critical"),
        ]
        suggestions = await suggester.suggest_fixes(issues)
        if len(suggestions) >= 2:
            # Higher priority (lower number) should come first
            assert suggestions[0].priority <= suggestions[-1].priority

    def test_group_issues_by_file(self):
        suggester = AutoFixSuggester()
        issues = [
            _make_quality_issue(file_path="a.py"),
            _make_quality_issue(file_path="b.py"),
            _make_quality_issue(file_path="a.py"),
        ]
        grouped = suggester._group_issues_by_file(issues)
        assert len(grouped["a.py"]) == 2
        assert len(grouped["b.py"]) == 1

    def test_calculate_fix_priority_critical(self):
        suggester = AutoFixSuggester()
        issue = _make_quality_issue(severity="critical")
        rule = FixRule(
            rule_id="test",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.SECURITY,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 1  # critical(1) - security adjustment(-2) = max(1, -1) = 1

    def test_calculate_fix_priority_info(self):
        suggester = AutoFixSuggester()
        issue = _make_quality_issue(severity="info")
        rule = FixRule(
            rule_id="test",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.FORMATTING,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 10  # info(8) + formatting(+2) = 10

    def test_calculate_fix_priority_bug_fix(self):
        suggester = AutoFixSuggester()
        issue = _make_quality_issue(severity="error")
        rule = FixRule(
            rule_id="test",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.BUG_FIX,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 1  # error(2) - bug_fix(-1) = 1

    def test_filter_applicable_fixes_no_filter(self):
        suggester = AutoFixSuggester()
        fixes = [_make_fix_suggestion(confidence=0.5)]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=False)
        assert len(result) == 1

    def test_filter_applicable_fixes_by_confidence(self):
        suggester = AutoFixSuggester()
        fixes = [
            _make_fix_suggestion(
                fix_id="f1", confidence=0.95, strategy=FixStrategy.AUTOMATIC, breaking_change=False
            ),
            _make_fix_suggestion(
                fix_id="f2", confidence=0.5, strategy=FixStrategy.AUTOMATIC, breaking_change=False
            ),
        ]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=True)
        assert len(result) == 1
        assert result[0].fix_id == "f1"

    def test_filter_applicable_fixes_excludes_breaking(self):
        suggester = AutoFixSuggester()
        fixes = [
            _make_fix_suggestion(
                confidence=0.95, strategy=FixStrategy.AUTOMATIC, breaking_change=True
            ),
        ]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=True)
        assert len(result) == 0

    def test_filter_applicable_fixes_excludes_manual(self):
        suggester = AutoFixSuggester()
        fixes = [
            _make_fix_suggestion(
                confidence=0.95, strategy=FixStrategy.MANUAL, breaking_change=False
            ),
        ]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=True)
        assert len(result) == 0

    async def test_apply_automated_fixes_empty(self):
        suggester = AutoFixSuggester()
        result = await suggester.apply_automated_fixes([])
        assert isinstance(result, FixApplicationResult)
        assert result.successful_applications == 0

    async def test_apply_automated_fixes_dry_run(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(
            confidence=0.95,
            strategy=FixStrategy.AUTOMATIC,
            file_path="test.py",
            fixed_code=None,
        )
        result = await suggester.apply_automated_fixes(
            [fix], dry_run=True, filter_by_confidence=True
        )
        assert isinstance(result, FixApplicationResult)

    async def test_validate_fixes_valid_python(self):
        suggester = AutoFixSuggester()
        result = await suggester.validate_fixes(
            original="x = 1\n",
            resolved="x = 2\n",
            language="python",
        )
        assert result.is_valid is True
        assert result.syntax_valid is True

    async def test_validate_fixes_empty_resolved(self):
        suggester = AutoFixSuggester()
        result = await suggester.validate_fixes(
            original="x = 1\n",
            resolved="   ",
            language="python",
        )
        assert result.is_valid is False

    async def test_validate_fixes_javascript_mismatched_braces(self):
        suggester = AutoFixSuggester()
        result = await suggester.validate_fixes(
            original="function f() { }",
            resolved="function f() { { }",
            language="javascript",
        )
        # Should have warnings about mismatched braces
        assert len(result.validation_errors) > 0 or not result.is_safe

    async def test_validate_fixes_shorter_code_warns(self):
        suggester = AutoFixSuggester()
        result = await suggester.validate_fixes(
            original="x = 1\ny = 2\nz = 3\nw = 4\nq = 5\n",
            resolved="x = 1\n",
            language="python",
        )
        assert len(result.warnings) > 0

    async def test_suggest_bulk_fixes_empty(self):
        suggester = AutoFixSuggester()
        result = await suggester.suggest_bulk_fixes([])
        assert result == {}

    async def test_suggest_bulk_fixes_with_results(self):
        suggester = AutoFixSuggester()
        issues = [_make_quality_issue(tool="ruff", rule_id="D103")]
        qr = _make_quality_result(issues=issues)
        result = await suggester.suggest_bulk_fixes([qr])
        assert isinstance(result, dict)

    async def test_validate_fix_syntax_no_code(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(fixed_code=None, file_path="test.py")
        result = await suggester._validate_fix_syntax(fix)
        assert result is True

    async def test_validate_fix_syntax_python(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(fixed_code="x = 1", file_path="test.py")
        result = await suggester._validate_fix_syntax(fix)
        assert result is True

    async def test_validate_fix_syntax_unknown_lang(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(fixed_code="hello", file_path="test.rb")
        result = await suggester._validate_fix_syntax(fix)
        assert result is True

    async def test_validate_fix_syntax_js(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(fixed_code="const x = 1;", file_path="test.js")
        result = await suggester._validate_fix_syntax(fix)
        assert result is True

    async def test_validate_fix_syntax_tsx(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(fixed_code="const x = 1;", file_path="test.tsx")
        result = await suggester._validate_fix_syntax(fix)
        assert result is True

    async def test_apply_line_based_fix(self):
        suggester = AutoFixSuggester()
        content = "line1\nline2\nline3"
        result = await suggester._apply_line_based_fix(content, 2, "fixed_line2")
        assert "fixed_line2" in result

    async def test_apply_line_based_fix_out_of_range(self):
        suggester = AutoFixSuggester()
        content = "line1\nline2"
        result = await suggester._apply_line_based_fix(content, 100, "fix")
        assert result == content

    async def test_apply_code_fixes_with_replacement(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(
            original_code="old_code",
            fixed_code="new_code",
            category=FixCategory.BUG_FIX,
        )
        result = await suggester._apply_code_fixes("some old_code here", [fix])
        assert "new_code" in result

    async def test_apply_code_fixes_line_based(self):
        suggester = AutoFixSuggester()
        fix = _make_fix_suggestion(
            original_code=None,
            fixed_code="new_line",
            line_number=2,
            category=FixCategory.BUG_FIX,
        )
        result = await suggester._apply_code_fixes("line1\nline2\nline3", [fix])
        assert "new_line" in result

    async def test_group_similar_fixes(self):
        suggester = AutoFixSuggester()
        fixes = [_make_fix_suggestion(), _make_fix_suggestion(fix_id="fix-2")]
        result = await suggester._group_similar_fixes(fixes)
        assert len(result) == 2

    async def test_prioritize_fixes_by_impact(self):
        suggester = AutoFixSuggester()
        f1 = _make_fix_suggestion(
            fix_id="f1",
            category=FixCategory.SECURITY,
            priority=1,
            confidence=0.95,
            strategy=FixStrategy.AUTOMATIC,
        )
        f2 = _make_fix_suggestion(
            fix_id="f2",
            category=FixCategory.FORMATTING,
            priority=8,
            confidence=0.5,
            strategy=FixStrategy.MANUAL,
        )
        result = await suggester._prioritize_fixes_by_impact([f2, f1])
        assert result[0].fix_id == "f1"  # higher impact first

    def test_create_empty_result(self):
        suggester = AutoFixSuggester()
        import time

        result = suggester._create_empty_result("app-1", time.time())
        assert result.total_fixes_attempted == 0

    def test_create_error_result(self):
        suggester = AutoFixSuggester()
        import time

        fixes = [_make_fix_suggestion()]
        result = suggester._create_error_result("app-1", fixes, time.time(), "boom")
        assert result.failed_applications == 1
        assert "boom" in result.errors

    def test_factory_function(self):
        s = create_auto_fix_suggester()
        assert isinstance(s, AutoFixSuggester)

    def test_factory_function_with_params(self):
        s = create_auto_fix_suggester(max_fixes_per_file=10, enable_experimental_fixes=True)
        assert s.max_fixes_per_file == 10
        assert s.enable_experimental_fixes is True

    async def test_create_file_backup_fail(self):
        suggester = AutoFixSuggester()
        result = await suggester._create_file_backup("/nonexistent/path/to/file.py")
        assert result is False


# ===========================================================================
# QualityToolRunner tests
# ===========================================================================


class TestQualityToolRunner:
    """Tests for QualityToolRunner."""

    def _make_runner(self):
        docker = MagicMock()
        docker.create_test_container = AsyncMock(return_value="container-123")
        docker.start_container = AsyncMock()
        docker.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        docker.create_file_in_container = AsyncMock(return_value=True)
        docker.stop_container = AsyncMock()
        docker.remove_container_by_id = AsyncMock()
        return QualityToolRunner(docker)

    def test_init(self):
        runner = self._make_runner()
        assert runner.docker_service is not None

    def test_detect_language_from_request(self):
        runner = self._make_runner()
        req = _make_quality_request(language="python")
        assert runner._detect_language(req) == "python"

    def test_detect_language_from_path(self):
        runner = self._make_runner()
        req = _make_quality_request(language=None, code_path="app.js")
        assert runner._detect_language(req) == "javascript"

    def test_detect_language_ts(self):
        runner = self._make_runner()
        req = _make_quality_request(language=None, code_path="app.tsx")
        assert runner._detect_language(req) == "typescript"

    def test_detect_language_default(self):
        runner = self._make_runner()
        req = _make_quality_request(language=None, code_path=None)
        assert runner._detect_language(req) == "python"

    def test_get_file_extension(self):
        runner = self._make_runner()
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("typescript") == ".ts"
        assert runner._get_file_extension("unknown") == ".py"

    def test_get_target_filename_with_path(self):
        runner = self._make_runner()
        req = _make_quality_request(code_path="/some/dir/app.py")
        assert runner._get_target_filename(req, "python") == "app.py"

    def test_get_target_filename_without_path(self):
        runner = self._make_runner()
        req = _make_quality_request(code_path=None)
        assert runner._get_target_filename(req, "python") == "main.py"

    def test_calculate_quality_score_no_issues(self):
        runner = self._make_runner()
        assert runner._calculate_quality_score([], {}) == 10.0

    def test_calculate_quality_score_with_issues(self):
        runner = self._make_runner()
        issues = [
            _make_quality_issue(severity="critical"),
            _make_quality_issue(severity="warning"),
        ]
        score = runner._calculate_quality_score(issues, {})
        assert 0 <= score <= 10.0
        assert score < 10.0

    def test_map_ruff_severity(self):
        runner = self._make_runner()
        assert runner._map_ruff_severity("F401") == "error"
        assert runner._map_ruff_severity("PLE0101") == "error"
        assert runner._map_ruff_severity("E501") == "warning"
        assert runner._map_ruff_severity("PLW0602") == "warning"
        assert runner._map_ruff_severity("I001") == "info"
        assert runner._map_ruff_severity("PLR0913") == "info"

    def test_select_tools_all(self):
        runner = self._make_runner()
        with patch.object(
            runner.tool_registry, "list_available_tools", return_value=["ruff", "ty"]
        ):
            result = runner._select_tools("python", None)
            assert result == ["ruff", "ty"]

    def test_select_tools_filtered(self):
        runner = self._make_runner()
        with patch.object(
            runner.tool_registry, "list_available_tools", return_value=["ruff", "ty"]
        ):
            result = runner._select_tools("python", ["ruff", "ruff-format"])
            assert result == ["ruff"]

    def test_initialize_container_images(self):
        runner = self._make_runner()
        images = runner._initialize_container_images()
        assert "python" in images
        assert "javascript" in images

    def test_get_available_tools(self):
        runner = self._make_runner()
        with patch.object(runner.tool_registry, "list_available_tools", return_value=["ruff"]):
            result = runner.get_available_tools("python")
            assert result == ["ruff"]

    def test_get_tool_info(self):
        runner = self._make_runner()
        with patch.object(runner.tool_registry, "get_tool_info", return_value={"name": "ruff"}):
            result = runner.get_tool_info("ruff", "python")
            assert result["name"] == "ruff"

    async def test_run_quality_analysis_no_tools(self):
        runner = self._make_runner()
        req = _make_quality_request(language="python")
        with patch.object(runner, "_select_tools", return_value=[]):
            result = await runner.run_quality_analysis(req)
            assert result.success is False
            assert "No quality tools" in result.error

    async def test_run_quality_analysis_success(self):
        runner = self._make_runner()
        req = _make_quality_request(language="python")
        with patch.object(runner, "_select_tools", return_value=["ruff"]):
            with patch.object(
                runner, "_execute_tools_parallel", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = {"ruff": {"success": True, "issues": []}}
                result = await runner.run_quality_analysis(req)
                assert result.overall_score == 10.0

    async def test_cleanup_container(self):
        runner = self._make_runner()
        await runner._cleanup_container("container-123")
        runner.docker_service.stop_container.assert_awaited_once_with("container-123")
        runner.docker_service.remove_container_by_id.assert_awaited_once_with("container-123")

    async def test_parse_json_output_ruff(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ruff", output_format="json")
        output = '[{"code": "E999", "message": "bad", "filename": "t.py", "location": {"row": 5, "column": 2}}]'
        issues = await runner._parse_json_output(tool, output, "test.py")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].line_number == 5

    async def test_parse_json_output_eslint(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="eslint", output_format="json")
        output = '[{"messages": [{"severity": 2, "message": "no-var", "line": 3, "column": 1, "ruleId": "no-var"}]}]'
        issues = await runner._parse_json_output(tool, output, "test.js")
        assert len(issues) == 1
        assert issues[0].severity == "error"

    async def test_parse_json_output_bandit(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="bandit", output_format="json")
        output = '{"results": [{"issue_severity": "HIGH", "issue_text": "eval usage", "line_number": 10, "test_id": "B307"}]}'
        issues = await runner._parse_json_output(tool, output, "test.py")
        assert len(issues) == 1
        assert issues[0].severity == "high"

    async def test_parse_json_output_empty(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ruff", output_format="json")
        issues = await runner._parse_json_output(tool, "", "test.py")
        assert issues == []

    async def test_parse_json_output_invalid(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ruff", output_format="json")
        issues = await runner._parse_json_output(tool, "not json", "test.py")
        assert issues == []

    async def test_parse_json_output_ty(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ty", output_format="json")
        issues = await runner._parse_json_output(
            tool,
            '[{"check_name":"invalid-assignment","description":"invalid-assignment: Bad type","severity":"major","location":{"path":"t.py","positions":{"begin":{"line":10,"column":1}}}}]',
            "test.py",
        )
        assert len(issues) == 1
        assert issues[0].line_number == 10

    async def test_parse_text_output_formatter(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="prettier", output_format="text")
        issues = await runner._parse_text_output(tool, "[warn] test.js", "", "test.js")
        assert len(issues) == 1
        assert issues[0].tool == "prettier"

    async def test_parse_diff_output(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ruff-format", output_format="dif")
        issues = await runner._parse_diff_output(tool, "--- a/test.py\n+++ b/test.py", "test.py")
        assert len(issues) == 1
        assert issues[0].rule_id == "format_dif"

    async def test_parse_diff_output_empty(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="ruff-format", output_format="dif")
        issues = await runner._parse_diff_output(tool, "", "test.py")
        assert issues == []

    async def test_parse_tool_output_unknown_format(self):
        runner = self._make_runner()
        tool = _make_quality_tool(name="custom", output_format="yaml")
        issues = await runner._parse_tool_output(tool, "data", "", "test.py")
        assert issues == []

    async def test_execute_single_tool_success(self):
        runner = self._make_runner()
        tool = _make_quality_tool()
        runner.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "[]", "stderr": ""}
        )
        result = await runner._execute_single_tool("container-123", tool, "test.py")
        assert result["success"] is True

    async def test_execute_single_tool_failure(self):
        runner = self._make_runner()
        tool = _make_quality_tool()
        runner.docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("exec failed"))
        result = await runner._execute_single_tool("container-123", tool, "test.py")
        assert result["success"] is False

    async def test_run_single_tool_no_tool(self):
        runner = self._make_runner()
        with patch.object(runner.tool_registry, "get_tool", return_value=None):
            with pytest.raises(ValueError, match="not available"):
                await runner.run_single_tool("nonexistent", "python", "x=1")

    async def test_ensure_tools_installed_success(self):
        runner = self._make_runner()
        with patch.object(
            runner.tool_registry,
            "get_installation_commands",
            return_value=["uv pip install --system ruff"],
        ):
            await runner._ensure_tools_installed("cid", "python", ["ruff"])
            runner.docker_service.exec_in_container.assert_awaited()

    async def test_ensure_tools_installed_failure(self):
        runner = self._make_runner()
        runner.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stderr": "failed"}
        )
        with patch.object(
            runner.tool_registry,
            "get_installation_commands",
            return_value=["uv pip install --system ruff"],
        ):
            # Should not raise
            await runner._ensure_tools_installed("cid", "python", ["ruff"])

    async def test_prepare_code_files_from_code(self):
        runner = self._make_runner()
        req = _make_quality_request(code="x = 1")
        await runner._prepare_code_files("cid", req, "python")
        runner.docker_service.create_file_in_container.assert_awaited()

    async def test_prepare_code_files_no_code(self):
        runner = self._make_runner()
        req = _make_quality_request(code=None, code_path=None)
        runner.docker_service.create_file_in_container = AsyncMock(return_value=False)
        with pytest.raises(ValueError, match="No code files"):
            await runner._prepare_code_files("cid", req, "python")


# ===========================================================================
# JavaQualityAnalyzer tests
# ===========================================================================


class TestJavaQualityAnalyzer:
    """Tests for JavaQualityAnalyzer."""

    def _make_analyzer(self):
        docker = MagicMock()
        docker.create_test_container = AsyncMock(return_value="java-cid")
        docker.start_container = AsyncMock()
        docker.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "", "stderr": ""}
        )
        docker.remove_container_by_id = AsyncMock()
        analyzer = JavaQualityAnalyzer(docker)
        return analyzer

    def test_init(self):
        analyzer = self._make_analyzer()
        assert analyzer.docker_service is not None
        assert isinstance(analyzer.config, dict)

    def test_get_tools_for_profile_standard(self):
        analyzer = self._make_analyzer()
        tools = analyzer._get_tools_for_profile("standard")
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_get_tools_for_profile_unknown(self):
        analyzer = self._make_analyzer()
        tools = analyzer._get_tools_for_profile("nonexistent")
        assert isinstance(tools, list)

    def test_prepare_tool_command_checkstyle(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="checkstyle", command=["checkstyle"])
        cmd = analyzer._prepare_tool_command(tool, "/project")
        assert "/workspace" in cmd

    def test_prepare_tool_command_pmd(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="pmd", command=["pmd", "check", "-d", "."])
        cmd = analyzer._prepare_tool_command(tool, "/project")
        assert "/workspace" in cmd

    def test_prepare_tool_command_spotbugs(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="spotbugs", command=["spotbugs"])
        cmd = analyzer._prepare_tool_command(tool, "/project")
        assert "/workspace/target/classes" in cmd

    def test_prepare_tool_command_google_java_format(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="google-java-format", command=["google-java-format"])
        cmd = analyzer._prepare_tool_command(tool, "/project")
        assert "/workspace/**/*.java" in cmd

    def test_parse_checkstyle_xml(self):
        analyzer = self._make_analyzer()
        root = Element("checkstyle")
        file_elem = SubElement(root, "file", name="App.java")
        SubElement(
            file_elem,
            "error",
            severity="warning",
            message="Missing Javadoc",
            line="10",
            column="5",
            source="JavadocMethodCheck",
        )
        issues = analyzer._parse_checkstyle_xml(root)
        assert len(issues) == 1
        assert issues[0].tool == "checkstyle"
        assert issues[0].line_number == 10

    def test_parse_pmd_xml(self):
        analyzer = self._make_analyzer()
        root = Element("pmd")
        file_elem = SubElement(root, "file", name="App.java")
        violation = SubElement(
            file_elem,
            "violation",
            priority="3",
            beginline="15",
            begincolumn="8",
            rule="UnusedVariable",
        )
        violation.text = "Unused variable 'x'"
        issues = analyzer._parse_pmd_xml(root)
        assert len(issues) == 1
        assert issues[0].tool == "pmd"
        assert issues[0].line_number == 15
        assert "Unused" in issues[0].message

    def test_parse_spotbugs_xml(self):
        analyzer = self._make_analyzer()
        root = Element("BugCollection")
        bug = SubElement(root, "BugInstance", type="NP_NULL_ON_SOME_PATH", priority="2")
        SubElement(bug, "SourceLine", sourcepath="App.java", start="20")
        msg = SubElement(bug, "ShortMessage")
        msg.text = "Possible null pointer dereference"
        issues = analyzer._parse_spotbugs_xml(root)
        assert len(issues) == 1
        assert issues[0].tool == "spotbugs"
        assert issues[0].line_number == 20

    def test_parse_spotbugs_xml_no_source_line(self):
        analyzer = self._make_analyzer()
        root = Element("BugCollection")
        bug = SubElement(root, "BugInstance", type="NP_NULL", priority="2")
        msg = SubElement(bug, "ShortMessage")
        msg.text = "Issue"
        issues = analyzer._parse_spotbugs_xml(root)
        assert len(issues) == 1
        assert issues[0].line_number == 0

    def test_parse_json_output(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="json")
        issues = analyzer._parse_json_output(tool, '{"data": []}')
        assert issues == []

    def test_parse_json_output_invalid(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="json")
        issues = analyzer._parse_json_output(tool, "not json")
        assert issues == []

    def test_parse_text_output(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="text")
        output = "App.java:10:Missing semicolon"
        issues = analyzer._parse_text_output(tool, output)
        assert len(issues) == 1
        assert issues[0].line_number == 10

    def test_parse_text_output_empty(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="text")
        issues = analyzer._parse_text_output(tool, "")
        assert issues == []

    def test_parse_text_output_no_line_number(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="text")
        issues = analyzer._parse_text_output(tool, "file:abc:message here")
        assert len(issues) == 1
        assert issues[0].line_number == 0

    def test_parse_tool_output_xml(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="checkstyle", output_format="xml")
        xml_str = '<checkstyle><file name="App.java"><error severity="warning" message="test" line="1" column="1" source="Check"/></file></checkstyle>'
        issues = analyzer._parse_tool_output(tool, xml_str, "")
        assert len(issues) == 1

    def test_parse_tool_output_xml_invalid(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="checkstyle", output_format="xml")
        # Invalid XML should fall back to text parsing
        issues = analyzer._parse_tool_output(tool, "not xml", "")
        assert isinstance(issues, list)

    def test_parse_tool_output_text_fallback(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="custom", output_format="text")
        issues = analyzer._parse_tool_output(tool, "file:1:msg", "")
        assert len(issues) >= 1

    def test_create_quality_result(self):
        analyzer = self._make_analyzer()
        issues = [_make_quality_issue()]
        tool_results = {"ruff": {"status": "success", "issues_count": 1}}
        result = analyzer._create_quality_result(issues, tool_results, "standard", "/proj")
        assert result.success is True
        assert result.language == "java"

    def test_create_quality_result_high_issues(self):
        analyzer = self._make_analyzer()
        issues = [_make_quality_issue() for _ in range(60)]
        result = analyzer._create_quality_result(issues, {}, "standard", "/proj")
        assert result.overall_score == 0.0

    def test_calculate_issue_metrics(self):
        analyzer = self._make_analyzer()
        issues = [
            _make_quality_issue(severity="warning", tool="checkstyle"),
            _make_quality_issue(severity="error", tool="pmd"),
            _make_quality_issue(severity="warning", tool="checkstyle"),
        ]
        metrics = analyzer._calculate_issue_metrics(issues)
        assert metrics["by_severity"]["warning"] == 2
        assert metrics["by_severity"]["error"] == 1
        assert metrics["by_tool"]["checkstyle"] == 2

    async def test_generate_quality_report_json(self):
        analyzer = self._make_analyzer()
        qr = _make_quality_result(language="java")
        report = await analyzer.generate_quality_report(qr, output_format="json")
        import json

        data = json.loads(report)
        assert "analysis_id" in data

    async def test_generate_quality_report_html(self):
        analyzer = self._make_analyzer()
        qr = _make_quality_result(language="java")
        report = await analyzer.generate_quality_report(qr, output_format="html")
        assert "<html>" in report

    async def test_generate_quality_report_markdown(self):
        analyzer = self._make_analyzer()
        qr = _make_quality_result(language="java")
        report = await analyzer.generate_quality_report(qr, output_format="markdown")
        assert "# Java Quality Analysis Report" in report

    async def test_generate_quality_report_default(self):
        analyzer = self._make_analyzer()
        qr = _make_quality_result(language="java")
        report = await analyzer.generate_quality_report(qr, output_format="txt")
        assert isinstance(report, str)

    def test_generate_html_report_with_issues(self):
        analyzer = self._make_analyzer()
        issues = [
            _make_quality_issue(severity="error", tool="checkstyle"),
            _make_quality_issue(severity="warning", tool="pmd"),
        ]
        qr = _make_quality_result(issues=issues, language="java")
        html = analyzer._generate_html_report(qr)
        assert "checkstyle" in html or "pmd" in html

    def test_generate_markdown_report_with_issues(self):
        analyzer = self._make_analyzer()
        issues = [
            _make_quality_issue(severity="error", tool="spotbugs"),
        ]
        qr = _make_quality_result(issues=issues, language="java")
        md = analyzer._generate_markdown_report(qr)
        assert "spotbugs" in md

    async def test_analyze_java_project(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_get_tools_for_profile", return_value=["checkstyle"]):
            with patch.object(
                analyzer,
                "_run_java_tool",
                new_callable=AsyncMock,
                return_value=[_make_quality_issue(tool="checkstyle")],
            ):
                with patch.object(analyzer, "_install_java_tools", new_callable=AsyncMock):
                    result = await analyzer.analyze_java_project("/project", profile="standard")
                    assert result.success is True
                    assert result.language == "java"

    async def test_analyze_java_project_tool_error(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_get_tools_for_profile", return_value=["checkstyle"]):
            with patch.object(
                analyzer,
                "_run_java_tool",
                new_callable=AsyncMock,
                side_effect=OSError("tool failed"),
            ):
                with patch.object(analyzer, "_install_java_tools", new_callable=AsyncMock):
                    result = await analyzer.analyze_java_project("/project")
                    # Should still succeed overall, with error in tool results
                    assert "checkstyle" in result.tools_results
                    assert result.tools_results["checkstyle"]["status"] == "error"

    async def test_run_java_tool_no_tool(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer.tool_registry, "get_tool", return_value=None):
            issues = await analyzer._run_java_tool("cid", "nonexistent", "/proj")
            assert issues == []

    async def test_run_java_tool_exec_error(self):
        analyzer = self._make_analyzer()
        tool = _make_quality_tool(name="checkstyle", output_format="xml")
        with patch.object(analyzer.tool_registry, "get_tool", return_value=tool):
            analyzer.docker_service.exec_in_container = AsyncMock(
                side_effect=OSError("exec failed")
            )
            issues = await analyzer._run_java_tool("cid", "checkstyle", "/proj")
            assert issues == []

    async def test_install_java_tools_success(self):
        analyzer = self._make_analyzer()
        with patch.object(
            analyzer.tool_registry,
            "get_installation_commands",
            return_value=["apt install checkstyle"],
        ):
            await analyzer._install_java_tools("cid")
            analyzer.docker_service.exec_in_container.assert_awaited()

    async def test_install_java_tools_failure(self):
        analyzer = self._make_analyzer()
        analyzer.docker_service.exec_in_container = AsyncMock(side_effect=OSError("install failed"))
        with patch.object(
            analyzer.tool_registry,
            "get_installation_commands",
            return_value=["apt install checkstyle"],
        ):
            # Should not raise
            await analyzer._install_java_tools("cid")
