"""Metrics module for HTTP Quality Runner."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics collection disabled")


# Dummy metrics for when prometheus is not available
class DummyMetric:
    """Dummy metric class for when prometheus is not available."""

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        """Increment metric - no-op for dummy."""

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        """Observe metric - no-op for dummy."""

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        """Set metric - no-op for dummy."""

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        """Set info metric - no-op for dummy."""

    def labels(self, *_args: Any, **_kwargs: Any) -> DummyMetric:
        """Return labels - returns self for dummy."""
        return self


# Declare metrics variables
quality_requests_total: Any
quality_duration_seconds: Any
quality_sessions_active: Any
quality_sessions_created_total: Any
quality_sessions_errors_total: Any
quality_connection_pool_size: Any
quality_retries_total: Any
quality_service_info: Any

# Initialize metrics based on prometheus availability
if PROMETHEUS_AVAILABLE:
    # Request metrics
    quality_requests_total = Counter(
        "quality_requests_total",
        "Total number of quality analysis requests",
        ["language", "status"],
    )

    quality_duration_seconds = Histogram(
        "quality_duration_seconds",
        "Quality analysis request duration in seconds",
        ["language"],
    )

    # Session metrics
    quality_sessions_active = Gauge("quality_sessions_active", "Number of active HTTP sessions")

    quality_sessions_created_total = Counter(
        "quality_sessions_created_total",
        "Total number of HTTP sessions created",
    )

    quality_sessions_errors_total = Counter(
        "quality_sessions_errors_total",
        "Total number of session errors",
        ["error_type"],
    )

    # Connection pool metrics
    quality_connection_pool_size = Gauge(
        "quality_connection_pool_size", "Current connection pool size"
    )

    quality_retries_total = Counter(
        "quality_retries_total", "Total number of request retries", ["attempt"]
    )

    # Service info
    quality_service_info = Info(
        "quality_service_info",
        "Information about the quality service configuration",
    )
else:
    # Use dummy metrics when prometheus is not available
    quality_requests_total = DummyMetric()
    quality_duration_seconds = DummyMetric()
    quality_sessions_active = DummyMetric()
    quality_sessions_created_total = DummyMetric()
    quality_sessions_errors_total = DummyMetric()
    quality_connection_pool_size = DummyMetric()
    quality_retries_total = DummyMetric()
    quality_service_info = DummyMetric()


class HTTPQualityMetrics:
    """HTTP Quality Metrics collector and handler.

    Manages Prometheus metrics for HTTP-based quality analysis operations,
    with fallback to dummy metrics when Prometheus is not available.
    """

    def __init__(self) -> None:
        """Initialize the HTTP Quality Metrics collector."""
        self.requests_total = quality_requests_total
        self.duration_seconds = quality_duration_seconds
        self.sessions_active = quality_sessions_active
        self.sessions_created_total = quality_sessions_created_total
        self.sessions_errors_total = quality_sessions_errors_total
        self.connection_pool_size = quality_connection_pool_size
        self.retries_total = quality_retries_total
        self.service_info = quality_service_info

        logger.info(
            "HTTPQualityMetrics initialized with prometheus_available=%s",
            PROMETHEUS_AVAILABLE,
        )

    def record_request(self, language: str, status: str) -> None:
        """Record a quality analysis request."""
        self.requests_total.labels(language=language, status=status).inc()

    def record_duration(self, language: str, duration: float) -> None:
        """Record request duration in seconds."""
        self.duration_seconds.labels(language=language).observe(duration)

    def set_active_sessions(self, count: int) -> None:
        """Set the number of active sessions."""
        self.sessions_active.set(count)

    def record_session_created(self) -> None:
        """Record a new session creation."""
        self.sessions_created_total.inc()

    def record_session_error(self, error_type: str) -> None:
        """Record a session error."""
        self.sessions_errors_total.labels(error_type=error_type).inc()

    def set_connection_pool_size(self, size: int) -> None:
        """Set the connection pool size."""
        self.connection_pool_size.set(size)

    def record_retry(self, attempt: str) -> None:
        """Record a retry attempt."""
        self.retries_total.labels(attempt=attempt).inc()

    def set_service_info(self, info: dict[str, str]) -> None:
        """Set service information."""
        self.service_info.info(info)
