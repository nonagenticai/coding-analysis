"""Tests for HTTPQualityRunner, SessionMonitor, and supporting models.

Verifies actual config values, score calculations, metric keys, and factory behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from processing.quality_analysis.http_quality_models import (
    HTTPQualityModel,
    QualityServiceConfig,
    RequestMetrics,
    SessionManager,
    SessionStats,
)
from processing.quality_analysis.http_quality_runner import (
    HTTPQualityRunner,
    SessionMonitor,
    create_quality_tool_runner,
)

# --- SessionMonitor tests ---


class TestSessionMonitor:
    @pytest.fixture
    def monitor(self):
        return SessionMonitor()

    @pytest.mark.asyncio
    async def test_register_session(self, monitor):
        mock_session = MagicMock()
        await monitor.register_session(mock_session)
        assert monitor.stats.total_sessions_created == 1
        assert monitor.stats.active_sessions >= 0

    @pytest.mark.asyncio
    async def test_record_session_closed(self, monitor):
        await monitor.record_session_closed()
        assert monitor.stats.total_sessions_closed == 1

    @pytest.mark.asyncio
    async def test_record_session_error(self, monitor):
        await monitor.record_session_error("connection timeout")
        assert monitor.stats.failed_sessions == 1
        assert len(monitor.stats.session_errors) == 1
        assert "timeout" in monitor.error_patterns

    @pytest.mark.asyncio
    async def test_error_trimming_at_50(self, monitor):
        for i in range(55):
            await monitor.record_session_error(f"error {i}")
        assert len(monitor.stats.session_errors) <= 50

    def test_categorize_error_session_closed(self, monitor):
        assert monitor._categorize_error("Session is closed") == "session_closed"

    def test_categorize_error_connection_refused(self, monitor):
        assert monitor._categorize_error("Connection refused by host") == "connection_refused"

    def test_categorize_error_timeout(self, monitor):
        assert monitor._categorize_error("Request timeout after 30s") == "timeout"

    def test_categorize_error_pool(self, monitor):
        assert monitor._categorize_error("Connection pool exhausted") == "connection_pool_issue"

    def test_categorize_error_502(self, monitor):
        assert monitor._categorize_error("HTTP 502 Bad Gateway") == "service_unavailable"

    def test_categorize_error_503(self, monitor):
        assert monitor._categorize_error("HTTP 503 Service Unavailable") == "service_unavailable"

    def test_categorize_error_other(self, monitor):
        assert monitor._categorize_error("unknown error xyz") == "other"

    def test_get_health_summary(self, monitor):
        summary = monitor.get_health_summary()
        assert summary["active_sessions"] == 0
        assert summary["total_created"] == 0
        assert summary["total_closed"] == 0
        assert summary["failed_sessions"] == 0
        assert summary["recent_errors"] == 0
        assert summary["error_patterns"] == {}


# --- QualityServiceConfig tests ---


class TestQualityServiceConfig:
    def test_valid_config(self):
        config = QualityServiceConfig(
            service_url="http://localhost:8688/api/quality",
            total_timeout=3600,
            connect_timeout=60,
            max_concurrent_connections=50,
            max_concurrent_requests=32,
            max_concurrent_batches=12,
            max_retries=3,
            keepalive_timeout=120,
        )
        assert config.service_url == "http://localhost:8688/api/quality"

    def test_invalid_connections(self):
        with pytest.raises(ValueError, match="max_concurrent_connections"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=0,
                max_concurrent_requests=1,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
            )

    def test_requests_exceed_connections(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=20,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
            )

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="total_timeout"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=10,
                connect_timeout=5,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
            )

    def test_connect_exceeds_total(self):
        with pytest.raises(ValueError, match="connect_timeout must be less"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=60,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
            )

    def test_invalid_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=11,
                keepalive_timeout=120,
            )

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="service_url"):
            QualityServiceConfig(
                service_url="not-a-url",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
            )

    def test_invalid_keepalive(self):
        with pytest.raises(ValueError, match="keepalive_timeout"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=5,
            )

    def test_invalid_max_requests_per_session(self):
        with pytest.raises(ValueError, match="max_requests_per_session"):
            QualityServiceConfig(
                service_url="http://localhost:8688/api/quality",
                total_timeout=3600,
                connect_timeout=60,
                max_concurrent_connections=10,
                max_concurrent_requests=10,
                max_concurrent_batches=1,
                max_retries=3,
                keepalive_timeout=120,
                max_requests_per_session=50,
            )


# --- HTTPQualityModel tests ---


class TestHTTPQualityModel:
    def _make_config(self):
        return QualityServiceConfig(
            service_url="http://localhost:8688/api/quality",
            total_timeout=3600,
            connect_timeout=60,
            max_concurrent_connections=50,
            max_concurrent_requests=32,
            max_concurrent_batches=12,
            max_retries=3,
            keepalive_timeout=120,
        )

    def test_init(self):
        model = HTTPQualityModel(config=self._make_config())
        assert model.session_manager is not None
        assert model.request_metrics == []

    def test_add_request_metric(self):
        model = HTTPQualityModel(config=self._make_config())
        metric = RequestMetrics(request_id="r1", start_time=1.0, end_time=2.0, success=True)
        model.add_request_metric(metric)
        assert len(model.request_metrics) == 1

    def test_add_request_metric_trimming(self):
        model = HTTPQualityModel(config=self._make_config())
        for i in range(1050):
            model.add_request_metric(RequestMetrics(request_id=f"r{i}", start_time=float(i)))
        assert len(model.request_metrics) <= 1000

    def test_get_success_rate_empty(self):
        model = HTTPQualityModel(config=self._make_config())
        assert model.get_success_rate() == 1.0

    def test_get_success_rate(self):
        model = HTTPQualityModel(config=self._make_config())
        model.add_request_metric(RequestMetrics(request_id="r1", start_time=1.0, success=True))
        model.add_request_metric(RequestMetrics(request_id="r2", start_time=2.0, success=False))
        assert model.get_success_rate() == 0.5

    def test_get_average_response_time_empty(self):
        model = HTTPQualityModel(config=self._make_config())
        assert model.get_average_response_time() == 0.0

    def test_get_average_response_time(self):
        model = HTTPQualityModel(config=self._make_config())
        model.add_request_metric(
            RequestMetrics(request_id="r1", start_time=1.0, end_time=3.0, success=True)
        )
        model.add_request_metric(
            RequestMetrics(request_id="r2", start_time=2.0, end_time=5.0, success=True)
        )
        avg = model.get_average_response_time()
        assert avg == 2.5  # (2 + 3) / 2

    def test_get_average_response_time_no_completed(self):
        model = HTTPQualityModel(config=self._make_config())
        model.add_request_metric(RequestMetrics(request_id="r1", start_time=1.0, end_time=None))
        assert model.get_average_response_time() == 0.0

    def test_cleanup_old_metrics(self):
        model = HTTPQualityModel(config=self._make_config())
        old = RequestMetrics(request_id="old", start_time=0.0)
        recent = RequestMetrics(request_id="new", start_time=datetime.now(UTC).timestamp())
        model.add_request_metric(old)
        model.add_request_metric(recent)
        model.cleanup_old_metrics(max_age_seconds=3600)
        assert len(model.request_metrics) == 1
        assert model.request_metrics[0].request_id == "new"


# --- HTTPQualityRunner tests ---


class TestHTTPQualityRunner:
    def test_init_config_has_correct_defaults(self):
        """Verify the runner's config is loaded with correct default values from env defaults."""
        runner = HTTPQualityRunner()
        # service_url is sourced from Settings; conftest stub starts with http.
        assert runner.config.service_url.startswith("http")
        assert runner.config.total_timeout == 3600
        assert runner.config.connect_timeout == 60
        assert runner.config.max_concurrent_connections == 50
        assert runner.config.max_concurrent_requests == 32
        assert runner.config.max_concurrent_batches == 12
        assert runner.config.max_retries == 3
        assert runner.config.keepalive_timeout == 120
        assert runner.monitor is not None

    def test_load_and_validate_config_returns_valid_config(self):
        """Config returned has correct types and passes validation."""
        runner = HTTPQualityRunner()
        config = runner._load_and_validate_config()
        assert config.service_url.startswith("http")
        assert config.total_timeout > config.connect_timeout
        assert config.max_concurrent_requests <= config.max_concurrent_connections

    def test_calculate_config_score_default_config(self):
        """Default config should score high (good configuration)."""
        runner = HTTPQualityRunner()
        score = runner._calculate_config_score()
        # Default: ratio=32/50=0.64, timeout=3600, retries=3 => score should be ~90
        # ratio 0.64 is in the 0.3-0.8 sweet spot, timeout 300-7200 is fine, retries 2-5 fine
        assert score == 100.0

    def test_calculate_health_score_healthy(self):
        runner = HTTPQualityRunner()
        basic_health = {"status": "healthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 0, "total_created": 1},
            "service_connectivity": {"reachable": True, "response_time_ms": 100},
            "resource_usage": {"memory_usage_mb": 200},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        assert score == 100.0

    def test_calculate_health_score_unhealthy_status_loses_50_points(self):
        runner = HTTPQualityRunner()
        basic_health = {"status": "unhealthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 5, "total_created": 10},
            "service_connectivity": {"reachable": False},
            "resource_usage": {"memory_usage_mb": 2000},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        # unhealthy=-50, unreachable=-30, 50% failures=-15, high memory=-10 => 100-50-30-15-10 = max(0, -5) = 0
        assert score < 10.0

    def test_calculate_health_score_slow_connectivity(self):
        runner = HTTPQualityRunner()
        basic_health = {"status": "healthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 0, "total_created": 1},
            "service_connectivity": {"reachable": True, "response_time_ms": 6000},
            "resource_usage": {"memory_usage_mb": 200},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        # healthy=100, slow >5000ms => -10 => 90
        assert score == 90.0

    def test_calculate_health_score_high_memory(self):
        runner = HTTPQualityRunner()
        basic_health = {"status": "healthy"}
        diagnostics = {
            "session_health": {"failed_sessions": 0, "total_created": 1},
            "service_connectivity": {"reachable": True, "response_time_ms": 100},
            "resource_usage": {"memory_usage_mb": 1500},
        }
        score = runner._calculate_health_score(basic_health, diagnostics)
        # healthy=100, memory >1000 => -10 => 90
        assert score == 90.0

    def test_create_failed_quality_result(self):
        runner = HTTPQualityRunner()
        now = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-123",
            start_time=now,
            language="python",
            error_message="Service unavailable",
        )
        assert result.success is False
        assert result.analysis_id == "test-123"
        assert result.language == "python"
        assert result.error == "Service unavailable"
        assert result.overall_score == 0.0
        assert result.total_issues == 0
        assert result.tools_results == {}

    def test_create_failed_quality_result_empty_error_gets_default(self):
        """Empty error message is replaced with a descriptive default."""
        runner = HTTPQualityRunner()
        now = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-456",
            start_time=now,
            language="java",
            error_message="",
        )
        assert "no specific error" in result.error.lower()

    def test_create_failed_quality_result_none_error_gets_default(self):
        """None error message is replaced with a descriptive default."""
        runner = HTTPQualityRunner()
        now = datetime.now(UTC)
        result = runner._create_failed_quality_result(
            analysis_id="test-789",
            start_time=now,
            language="go",
            error_message=None,
        )
        assert "no specific error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_close_clears_session(self):
        runner = HTTPQualityRunner()
        await runner.close()
        assert runner._session_manager.session is None

    @pytest.mark.asyncio
    async def test_get_resource_metrics_returns_dict_with_known_keys(self):
        """Resource metrics should return dict with either real metrics or an error key."""
        runner = HTTPQualityRunner()
        metrics = await runner._get_resource_metrics()
        assert isinstance(metrics, dict)
        if "error" not in metrics:
            assert "memory_usage_mb" in metrics
            assert "cpu_percent" in metrics
            assert "threads" in metrics
            assert metrics["memory_usage_mb"] > 0
            assert metrics["threads"] >= 1

    @pytest.mark.asyncio
    async def test_get_performance_metrics_returns_derived_values(self):
        """Performance metrics should include computed efficiency and error rate."""
        runner = HTTPQualityRunner()
        metrics = await runner._get_performance_metrics()
        assert isinstance(metrics, dict)
        if "error" not in metrics:
            assert metrics["session_efficiency_percent"] == 100.0  # no sessions created yet
            assert metrics["error_rate_percent"] == 0.0
            assert metrics["active_sessions"] == 0
            assert "configuration_score" in metrics


class TestCreateQualityToolRunner:
    def test_factory_function_returns_configured_runner(self):
        """Factory function returns an HTTPQualityRunner with valid default config."""
        runner = create_quality_tool_runner(None)
        assert isinstance(runner, HTTPQualityRunner)
        # service_url is sourced from Settings; conftest stub starts with http.
        assert runner.config.service_url.startswith("http")
        assert runner.config.max_retries == 3


# --- SessionStats tests ---


class TestSessionStats:
    def test_defaults(self):
        stats = SessionStats()
        assert stats.total_sessions_created == 0
        assert stats.active_sessions == 0
        assert stats.failed_sessions == 0
        assert stats.session_errors == []


class TestSessionManager:
    def test_defaults(self):
        sm = SessionManager()
        assert sm.session is None
        assert sm.request_count == 0


# --- Additional coverage for HTTPQualityRunner ---


class TestHTTPQualityRunnerConfigError:
    """Cover lines 215-217: config validation error path."""

    def test_invalid_env_config_raises(self):
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {"QUALITY_HTTP_TIMEOUT": "not_a_number"},
        ):
            with pytest.raises(ValueError):
                HTTPQualityRunner()


class TestHTTPQualityRunnerSessionCreationError:
    """Cover lines 248-252: session creation error."""

    @pytest.mark.asyncio
    async def test_create_session_failure(self):
        from unittest.mock import patch

        runner = HTTPQualityRunner()
        with patch(
            "processing.quality_analysis.http_quality_runner.TCPConnector",
            side_effect=OSError("cannot create connector"),
        ):
            with pytest.raises(OSError, match="cannot create connector"):
                await runner._create_session()
        # Verify error was recorded
        assert runner.monitor.stats.failed_sessions >= 1
        await runner.close()


class TestHTTPQualityRunnerCleanupError:
    """Cover lines 295-297: session cleanup error."""

    @pytest.mark.asyncio
    async def test_cleanup_session_error(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock(side_effect=OSError("close failed"))
        # Should not raise
        await runner._cleanup_session(mock_session)
        assert runner.monitor.stats.failed_sessions >= 1
        await runner.close()


class TestHTTPQualityRunnerAnalysisErrorPath:
    """Cover lines 420-430: exception in run_quality_analysis."""

    @pytest.mark.asyncio
    async def test_run_quality_analysis_exception(self):
        from unittest.mock import AsyncMock

        from models.testing_models import QualityRequest

        runner = HTTPQualityRunner()
        runner._health_check_with_retry = AsyncMock(side_effect=OSError("network down"))
        request = QualityRequest(
            container_name="test",
            code="x = 1",
            language="python",
        )
        result = await runner.run_quality_analysis(request)
        assert result.success is False
        assert "network down" in result.error
        await runner.close()


class TestHTTPQualityRunnerRetryExhaustion:
    """Cover lines 512-541: connection error exhausting all retries."""

    @pytest.mark.asyncio
    async def test_execute_request_all_retries_fail(self):
        from unittest.mock import AsyncMock, MagicMock

        from aiohttp import ClientError

        runner = HTTPQualityRunner()
        runner.config.max_retries = 2
        mock_session = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=ClientError("connection lost"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_ctx)

        # Also mock _ensure_session to prevent real session creation
        runner._ensure_session = AsyncMock(return_value=mock_session)
        runner._cleanup_session = AsyncMock()

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-retry",
            start_time=datetime.now(UTC),
        )
        assert result.success is False
        assert "failed" in result.error.lower()
        await runner.close()


class TestHTTPQualityRunnerSingleToolError:
    """Cover lines 603-606: single tool execution error."""

    @pytest.mark.asyncio
    async def test_run_single_tool_error(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("no session"))
        result = await runner.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is False
        assert "no session" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerGetAvailableToolsError:
    """Cover lines 636-642: get_available_tools error path."""

    @pytest.mark.asyncio
    async def test_get_available_tools_error(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("cannot connect"))
        result = await runner.get_available_tools("python")
        assert result == {}
        await runner.close()


class TestHTTPQualityRunnerDetailedHealthCheck:
    """Cover lines 652-693: detailed_health_check."""

    @pytest.mark.asyncio
    async def test_detailed_health_check_error(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner.health_check = AsyncMock(side_effect=OSError("network down"))
        result = await runner.detailed_health_check()
        assert result["status"] == "error"
        assert "network down" in result["error"]
        await runner.close()

    @pytest.mark.asyncio
    async def test_detailed_health_check_success(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner.health_check = AsyncMock(return_value={"status": "healthy"})
        runner._get_resource_metrics = AsyncMock(return_value={"memory_usage_mb": 100})
        runner._test_service_connectivity = AsyncMock(
            return_value={"reachable": True, "response_time_ms": 50}
        )
        runner._get_performance_metrics = AsyncMock(
            return_value={
                "session_efficiency_percent": 100.0,
                "error_rate_percent": 0.0,
                "active_sessions": 0,
                "connection_pool_utilization": 0.0,
                "configuration_score": 100.0,
            }
        )
        result = await runner.detailed_health_check()
        assert "health_score" in result
        assert result["health_score"] == 100.0
        assert "detailed_diagnostics" in result
        await runner.close()


class TestHTTPQualityRunnerHealthCheckError:
    """Cover lines 733-735: health_check exception path."""

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("session error"))
        result = await runner.health_check()
        assert result["status"] == "error"
        assert "session error" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerGetServiceInfo:
    """Cover lines 753-767: get_service_info error path."""

    @pytest.mark.asyncio
    async def test_get_service_info_error(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("no connection"))
        with pytest.raises(OSError, match="no connection"):
            await runner.get_service_info()
        await runner.close()


class TestHTTPQualityRunnerResourceMetricsError:
    """Cover lines 786-787: _get_resource_metrics OSError."""

    @pytest.mark.asyncio
    async def test_resource_metrics_oserror(self):
        from unittest.mock import MagicMock

        import processing.quality_analysis.http_quality_runner as mod

        runner = HTTPQualityRunner()
        original_psutil = getattr(mod, "psutil", None)
        original_available = mod.PSUTIL_AVAILABLE
        try:
            mock_psutil_mod = MagicMock()
            mock_psutil_mod.Process.side_effect = OSError("permission denied")
            mod.PSUTIL_AVAILABLE = True
            mod.psutil = mock_psutil_mod
            result = await runner._get_resource_metrics()
            assert "error" in result
            assert "permission denied" in result["error"]
        finally:
            mod.PSUTIL_AVAILABLE = original_available
            if original_psutil is not None:
                mod.psutil = original_psutil
        await runner.close()


class TestHTTPQualityRunnerTestServiceConnectivity:
    """Cover lines 791-809: _test_service_connectivity."""

    @pytest.mark.asyncio
    async def test_connectivity_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner._test_service_connectivity()
        assert result["reachable"] is True
        assert "response_time_ms" in result
        assert result["status_code"] == 200
        await runner.close()

    @pytest.mark.asyncio
    async def test_connectivity_failure(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("unreachable"))
        result = await runner._test_service_connectivity()
        assert result["reachable"] is False
        assert "unreachable" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerPerformanceMetricsError:
    """Cover lines 843-844: _get_performance_metrics exception."""

    @pytest.mark.asyncio
    async def test_performance_metrics_error(self):
        from unittest.mock import MagicMock

        runner = HTTPQualityRunner()
        # Force an error by making get_health_summary raise
        runner.monitor.get_health_summary = MagicMock(side_effect=ValueError("bad math"))
        result = await runner._get_performance_metrics()
        assert "error" in result
        assert "bad math" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerCreateSession:
    """Cover lines 231-246: _create_session success path."""

    @pytest.mark.asyncio
    async def test_create_session_success(self):
        runner = HTTPQualityRunner()
        session = await runner._create_session()
        assert session is not None
        assert not session.closed
        assert runner.monitor.stats.total_sessions_created == 1
        await session.close()
        await runner.close()


class TestHTTPQualityRunnerEnsureSession:
    """Cover lines 256-284: _ensure_session."""

    @pytest.mark.asyncio
    async def test_ensure_session_creates_new(self):
        runner = HTTPQualityRunner()
        session = await runner._ensure_session()
        assert session is not None
        assert runner._session_manager.session is session
        assert runner._session_manager.request_count == 0
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self):
        runner = HTTPQualityRunner()
        session1 = await runner._ensure_session()
        session2 = await runner._ensure_session()
        assert session1 is session2
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_replaces_closed(self):
        from unittest.mock import MagicMock

        runner = HTTPQualityRunner()
        # Create a mock "closed" session
        mock_closed = MagicMock()
        mock_closed.closed = True
        runner._session_manager.session = mock_closed
        runner._session_manager.created_at = datetime.now(UTC)

        session = await runner._ensure_session()
        assert session is not None
        assert session is not mock_closed
        await runner.close()

    @pytest.mark.asyncio
    async def test_ensure_session_replaces_max_requests(self):
        runner = HTTPQualityRunner()
        # First create a session
        await runner._ensure_session()
        # Set request count to exceed max
        runner._session_manager.request_count = runner.config.max_requests_per_session + 1
        session2 = await runner._ensure_session()
        # Should have created a new session
        assert session2 is not None
        await runner.close()


class TestHTTPQualityRunnerCleanupSessionSuccess:
    """Cover lines 289-294: _cleanup_session success path."""

    @pytest.mark.asyncio
    async def test_cleanup_session_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        await runner._cleanup_session(mock_session)
        mock_session.close.assert_called_once()
        assert runner.monitor.stats.total_sessions_closed == 1
        await runner.close()


class TestHTTPQualityRunnerHealthCheckWithRetry:
    """Cover lines 299-314: _health_check_with_retry."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner._health_check_with_retry(max_retries=1, delay=0)
        assert result is True
        await runner.close()

    @pytest.mark.asyncio
    async def test_health_check_all_fail(self):
        from unittest.mock import AsyncMock

        runner = HTTPQualityRunner()
        runner._ensure_session = AsyncMock(side_effect=OSError("no connect"))
        result = await runner._health_check_with_retry(max_retries=1, delay=0)
        assert result is False
        await runner.close()

    @pytest.mark.asyncio
    async def test_health_check_non_200(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 503
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner._health_check_with_retry(max_retries=1, delay=0)
        assert result is False
        await runner.close()


class TestHTTPQualityRunnerRunQualityAnalysis:
    """Cover lines 341-430: run_quality_analysis."""

    @pytest.mark.asyncio
    async def test_run_quality_analysis_disabled(self):
        from unittest.mock import patch

        from models.testing_models import QualityRequest

        runner = HTTPQualityRunner()
        with patch.dict("os.environ", {"ENABLE_QUALITY_ANALYSIS": "false"}):
            request = QualityRequest(
                container_name="test",
                code="x = 1",
                language="python",
            )
            result = await runner.run_quality_analysis(request)
            assert result.success is False
            assert "disabled" in result.error.lower()
        await runner.close()

    @pytest.mark.asyncio
    async def test_run_quality_analysis_unhealthy_service(self):
        from unittest.mock import AsyncMock

        from models.testing_models import QualityRequest

        runner = HTTPQualityRunner()
        runner._health_check_with_retry = AsyncMock(return_value=False)
        request = QualityRequest(
            container_name="test",
            code="x = 1",
            language="python",
        )
        result = await runner.run_quality_analysis(request)
        assert result.success is False
        assert "unavailable" in result.error.lower()
        await runner.close()

    @pytest.mark.asyncio
    async def test_run_quality_analysis_success(self):
        from unittest.mock import AsyncMock, MagicMock

        from models.testing_models import QualityRequest, QualityResult

        runner = HTTPQualityRunner()
        runner._health_check_with_retry = AsyncMock(return_value=True)
        mock_session = MagicMock()
        runner._ensure_session = AsyncMock(return_value=mock_session)

        # Mock the entire _execute_request_with_retry to return a success result
        mock_result = QualityResult(
            analysis_id="test-123",
            success=True,
            language="python",
            tools_results={"ruff": {"score": 9.5}},
            overall_score=95.0,
            total_issues=0,
            execution_time=1.0,
            created_at=datetime.now(UTC),
        )
        runner._execute_request_with_retry = AsyncMock(return_value=mock_result)

        request = QualityRequest(
            container_name="test",
            code="x = 1",
            language="python",
        )
        result = await runner.run_quality_analysis(request)
        assert result.success is True
        assert result.overall_score == 95.0
        await runner.close()


class TestHTTPQualityRunnerExecuteRequestWithRetry:
    """Cover lines 448-549: _execute_request_with_retry."""

    @pytest.mark.asyncio
    async def test_execute_request_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "analysis_id": "test-1",
                "success": True,
                "language": "python",
                "tools_results": {},
                "overall_score": 80.0,
                "total_issues": 0,
                "execution_time": 1.0,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_ctx)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-1",
            start_time=datetime.now(UTC),
        )
        assert result.success is True
        assert result.overall_score == 80.0
        await runner.close()

    @pytest.mark.asyncio
    async def test_execute_request_4xx_error_no_retry(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_ctx)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-400",
            start_time=datetime.now(UTC),
        )
        # 4xx should return a failed result (break, not retry)
        assert result.success is False
        await runner.close()

    @pytest.mark.asyncio
    async def test_execute_request_5xx_retries_then_fails(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        runner.config.max_retries = 2
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_ctx)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-500",
            start_time=datetime.now(UTC),
        )
        assert result.success is False
        assert "500" in result.error
        await runner.close()


class TestHTTPQualityRunnerRunSingleTool:
    """Cover lines 569-606: run_single_tool."""

    @pytest.mark.asyncio
    async def test_run_single_tool_success(self):
        from unittest.mock import AsyncMock, MagicMock

        from models.testing_models import QualityResult

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        runner._ensure_session = AsyncMock(return_value=mock_session)
        mock_result = QualityResult(
            analysis_id="single-1",
            success=True,
            language="python",
            tools_results={"ruff": {"issues": [], "score": 10}},
            overall_score=100.0,
            total_issues=0,
            execution_time=0.5,
            created_at=datetime.now(UTC),
        )
        runner._execute_request_with_retry = AsyncMock(return_value=mock_result)
        result = await runner.run_single_tool("ruff", "python", "x = 1")
        assert result["score"] == 10
        await runner.close()

    @pytest.mark.asyncio
    async def test_run_single_tool_not_in_results(self):
        from unittest.mock import AsyncMock, MagicMock

        from models.testing_models import QualityResult

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        runner._ensure_session = AsyncMock(return_value=mock_session)
        mock_result = QualityResult(
            analysis_id="single-2",
            success=True,
            language="python",
            tools_results={},  # Tool not in results
            overall_score=100.0,
            total_issues=0,
            execution_time=0.5,
            created_at=datetime.now(UTC),
        )
        runner._execute_request_with_retry = AsyncMock(return_value=mock_result)
        result = await runner.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is False
        assert "not found" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerGetAvailableTools:
    """Cover lines 624-642: get_available_tools success paths."""

    @pytest.mark.asyncio
    async def test_get_available_tools_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"available_tools": {"python": ["ruff", "ty"], "javascript": ["eslint"]}}
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.get_available_tools()
        assert "python" in result
        assert "ruff" in result["python"]
        await runner.close()

    @pytest.mark.asyncio
    async def test_get_available_tools_filtered_by_language(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"available_tools": {"python": ["ruff"], "javascript": ["eslint"]}}
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.get_available_tools("python")
        assert "python" in result
        assert "javascript" not in result
        await runner.close()

    @pytest.mark.asyncio
    async def test_get_available_tools_non_200(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.get_available_tools()
        assert result == {}
        await runner.close()


class TestHTTPQualityRunnerHealthCheckSuccess:
    """Cover lines 707-735: health_check success and unhealthy paths."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.health_check()
        assert result["status"] == "healthy"
        assert "session_health" in result
        await runner.close()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 503
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.health_check()
        assert result["status"] == "unhealthy"
        assert "503" in result["error"]
        await runner.close()


class TestHTTPQualityRunnerGetServiceInfoSuccess:
    """Cover lines 753-767: get_service_info success path."""

    @pytest.mark.asyncio
    async def test_get_service_info_success(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"name": "quality-service", "version": "1.0"})
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        result = await runner.get_service_info()
        assert result["name"] == "quality-service"
        await runner.close()

    @pytest.mark.asyncio
    async def test_get_service_info_http_error(self):
        from unittest.mock import AsyncMock, MagicMock

        runner = HTTPQualityRunner()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Error")
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_ctx)
        runner._ensure_session = AsyncMock(return_value=mock_session)

        with pytest.raises(RuntimeError, match="500"):
            await runner.get_service_info()
        await runner.close()
