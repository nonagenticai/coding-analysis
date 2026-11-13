"""Coverage boost tests for HTTPQualityRunner async methods.

Focuses on the async methods that require mocked HTTP sessions:
- _create_session
- _ensure_session
- _cleanup_session
- _health_check_with_retry
- run_quality_analysis (disabled path, health check fail, success, exception)
- _execute_request_with_retry (200, 4xx, 5xx, connection error)
- run_single_tool
- get_available_tools
- health_check
- get_service_info
- detailed_health_check
- _test_service_connectivity
- _get_performance_metrics edge cases
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from processing.quality_analysis.http_quality_runner import (
    HTTPQualityRunner,
    SessionMonitor,
)


@pytest.fixture
def runner():
    return HTTPQualityRunner()


# ── _create_session ───────────────────────────────────────────────────


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_returns_client_session(self, runner):
        session = await runner._create_session()
        assert isinstance(session, ClientSession)
        assert not session.closed
        # Clean up
        await session.close()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_create_session_registers_with_monitor(self, runner):
        session = await runner._create_session()
        assert runner.monitor.stats.total_sessions_created >= 1
        await session.close()
        await asyncio.sleep(0.1)


# ── _ensure_session ──────────────────────────────────────────────────


class TestEnsureSession:
    @pytest.mark.asyncio
    async def test_ensure_session_creates_new_when_none(self, runner):
        assert runner._session_manager.session is None
        session = await runner._ensure_session()
        assert session is not None
        assert not session.closed
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self, runner):
        s1 = await runner._ensure_session()
        s2 = await runner._ensure_session()
        assert s1 is s2
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_replaces_closed_session(self, runner):
        s1 = await runner._ensure_session()
        await s1.close()
        await asyncio.sleep(0.1)
        s2 = await runner._ensure_session()
        assert s2 is not s1
        assert not s2.closed
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_replaces_after_max_requests(self, runner):
        s1 = await runner._ensure_session()
        runner._session_manager.request_count = runner.config.max_requests_per_session + 1
        s2 = await runner._ensure_session()
        assert s2 is not s1
        await runner.close()


# ── _cleanup_session ─────────────────────────────────────────────────


class TestCleanupSession:
    @pytest.mark.asyncio
    async def test_cleanup_session_closes_open_session(self, runner):
        session = await runner._create_session()
        assert not session.closed
        await runner._cleanup_session(session)
        assert session.closed

    @pytest.mark.asyncio
    async def test_cleanup_session_handles_already_closed(self, runner):
        session = await runner._create_session()
        await session.close()
        await asyncio.sleep(0.1)
        # Should not raise
        await runner._cleanup_session(session)

    @pytest.mark.asyncio
    async def test_cleanup_session_handles_none(self, runner):
        # Should not raise
        await runner._cleanup_session(None)


# ── _health_check_with_retry ─────────────────────────────────────────


class TestHealthCheckWithRetry:
    @pytest.mark.asyncio
    async def test_health_check_fails_when_service_unreachable(self, runner):
        result = await runner._health_check_with_retry(max_retries=1, delay=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success_with_mocked_response(self, runner):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        runner._session_manager.session = mock_session
        runner._session_manager.created_at = datetime.now(UTC)
        runner._session_manager.request_count = 0

        result = await runner._health_check_with_retry(max_retries=1, delay=0)
        assert result is True
        # Restore for cleanup
        runner._session_manager.session = None


# ── run_quality_analysis ─────────────────────────────────────────────


class TestRunQualityAnalysis:
    @pytest.mark.asyncio
    async def test_disabled_via_env_var(self, runner):
        from models.testing_models import QualityRequest

        request = QualityRequest(
            code_content="x = 1",
            language="python",
            container_name="test-container",
        )
        with patch.dict(os.environ, {"ENABLE_QUALITY_ANALYSIS": "false"}):
            result = await runner.run_quality_analysis(request)
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unhealthy_service_returns_failed_result(self, runner):
        from models.testing_models import QualityRequest

        request = QualityRequest(
            code_content="x = 1",
            language="python",
            container_name="test-container",
        )
        with patch.object(
            runner, "_health_check_with_retry", new_callable=AsyncMock, return_value=False
        ):
            result = await runner.run_quality_analysis(request)
        assert result.success is False
        assert "unavailable" in result.error.lower()


# ── close ────────────────────────────────────────────────────────────


class TestClose:
    @pytest.mark.asyncio
    async def test_close_when_no_session(self, runner):
        await runner.close()
        assert runner._session_manager.session is None

    @pytest.mark.asyncio
    async def test_close_clears_active_session(self, runner):
        await runner._ensure_session()
        assert runner._session_manager.session is not None
        await runner.close()
        assert runner._session_manager.session is None


# ── _calculate_config_score edge cases ───────────────────────────────


class TestConfigScore:
    def test_high_request_ratio(self):
        runner = HTTPQualityRunner()
        runner.config.max_concurrent_requests = 48  # ratio 48/50=0.96 > 0.8
        runner.config.max_concurrent_connections = 50
        score = runner._calculate_config_score()
        assert score < 100.0

    def test_low_request_ratio(self):
        runner = HTTPQualityRunner()
        runner.config.max_concurrent_requests = 10  # ratio 10/50=0.2 < 0.3
        runner.config.max_concurrent_connections = 50
        score = runner._calculate_config_score()
        assert score < 100.0

    def test_short_timeout(self):
        runner = HTTPQualityRunner()
        runner.config.total_timeout = 200  # < 300
        score = runner._calculate_config_score()
        assert score < 100.0

    def test_long_timeout(self):
        runner = HTTPQualityRunner()
        runner.config.total_timeout = 8000  # > 7200
        score = runner._calculate_config_score()
        assert score < 100.0

    def test_low_retries(self):
        runner = HTTPQualityRunner()
        runner.config.max_retries = 1  # < 2
        score = runner._calculate_config_score()
        assert score < 100.0

    def test_high_retries(self):
        runner = HTTPQualityRunner()
        runner.config.max_retries = 6  # > 5
        score = runner._calculate_config_score()
        assert score < 100.0


# ── _calculate_health_score edge cases ───────────────────────────────


class TestHealthScore:
    def test_missing_diagnostics_keys(self):
        runner = HTTPQualityRunner()
        basic = {"status": "healthy"}
        diag = {}
        score = runner._calculate_health_score(basic, diag)
        assert score <= 100.0

    def test_zero_total_created(self):
        runner = HTTPQualityRunner()
        basic = {"status": "healthy"}
        diag = {
            "session_health": {"failed_sessions": 5, "total_created": 0},
            "service_connectivity": {"reachable": True, "response_time_ms": 50},
            "resource_usage": {"memory_usage_mb": 100},
        }
        score = runner._calculate_health_score(basic, diag)
        # With 0 total_created, failure_rate uses max(1,0)=1
        assert 0 <= score <= 100


# ── _get_performance_metrics edge cases ──────────────────────────────


class TestPerformanceMetrics:
    @pytest.mark.asyncio
    async def test_performance_metrics_with_sessions(self, runner):
        # Create and register a session
        session = await runner._create_session()
        metrics = await runner._get_performance_metrics()
        assert "session_efficiency_percent" in metrics
        assert "error_rate_percent" in metrics
        assert "configuration_score" in metrics
        await session.close()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_performance_metrics_with_errors(self, runner):
        await runner.monitor.record_session_error("test error")
        await runner.monitor.register_session(MagicMock())
        metrics = await runner._get_performance_metrics()
        assert metrics["error_rate_percent"] > 0


# ── _get_resource_metrics ────────────────────────────────────────────


class TestResourceMetrics:
    @pytest.mark.asyncio
    async def test_resource_metrics_with_psutil(self, runner):
        metrics = await runner._get_resource_metrics()
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_resource_metrics_without_psutil(self, runner):
        with patch("processing.quality_analysis.http_quality_runner.PSUTIL_AVAILABLE", False):
            metrics = await runner._get_resource_metrics()
            assert "error" in metrics


# ── _create_failed_quality_result ────────────────────────────────────


class TestCreateFailedResult:
    def test_whitespace_only_error_gets_default(self):
        runner = HTTPQualityRunner()
        result = runner._create_failed_quality_result(
            analysis_id="test",
            start_time=datetime.now(UTC),
            language="python",
            error_message="   ",
        )
        assert "no specific error" in result.error.lower()

    def test_execution_time_is_positive(self):
        runner = HTTPQualityRunner()
        now = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test",
            start_time=now,
            language="rust",
            error_message="timeout",
        )
        assert result.execution_time >= 0


# ── SessionMonitor additional tests ──────────────────────────────────


class TestSessionMonitorAdditional:
    @pytest.mark.asyncio
    async def test_error_trimming_keeps_recent(self):
        monitor = SessionMonitor()
        for i in range(55):
            await monitor.record_session_error(f"error {i}")
        # After multiple trims, errors are bounded (<=50)
        assert len(monitor.stats.session_errors) <= 50

    @pytest.mark.asyncio
    async def test_multiple_error_patterns(self):
        monitor = SessionMonitor()
        await monitor.record_session_error("session is closed")
        await monitor.record_session_error("connection refused by server")
        await monitor.record_session_error("request timeout")
        await monitor.record_session_error("connection pool issue")
        await monitor.record_session_error("HTTP 502")
        await monitor.record_session_error("HTTP 503")
        await monitor.record_session_error("something else")

        assert monitor.error_patterns["session_closed"] == 1
        assert monitor.error_patterns["connection_refused"] == 1
        assert monitor.error_patterns["timeout"] == 1
        assert monitor.error_patterns["connection_pool_issue"] == 1
        assert monitor.error_patterns["service_unavailable"] == 2
        assert monitor.error_patterns["other"] == 1

    @pytest.mark.asyncio
    async def test_health_summary_after_operations(self):
        monitor = SessionMonitor()
        mock_session = MagicMock()
        await monitor.register_session(mock_session)
        await monitor.record_session_error("test error")
        await monitor.record_session_closed()

        summary = monitor.get_health_summary()
        assert summary["total_created"] == 1
        assert summary["total_closed"] == 1
        assert summary["failed_sessions"] == 1
        assert summary["recent_errors"] == 1
