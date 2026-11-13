"""Models for HTTP Quality Runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aiohttp import ClientSession


@dataclass
class SessionStats:
    """Statistics for session monitoring."""

    total_sessions_created: int = 0
    total_sessions_closed: int = 0
    active_sessions: int = 0
    failed_sessions: int = 0
    session_errors: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RequestMetrics:
    """Metrics for individual requests."""

    request_id: str
    start_time: float
    end_time: float | None = None
    success: bool = False
    error_message: str | None = None
    retry_count: int = 0
    session_reused: bool = False


@dataclass
class QualityServiceConfig:
    """Configuration for quality service with validation."""

    service_url: str
    total_timeout: int
    connect_timeout: int
    max_concurrent_connections: int
    max_concurrent_requests: int
    max_concurrent_batches: int
    max_retries: int
    keepalive_timeout: int
    max_requests_per_session: int = 1000

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.max_concurrent_connections < 1:
            errors.append("max_concurrent_connections must be >= 1")

        if self.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be >= 1")

        if self.max_concurrent_requests > self.max_concurrent_connections:
            errors.append("max_concurrent_requests cannot exceed max_concurrent_connections")

        if self.max_concurrent_batches < 1:
            errors.append("max_concurrent_batches must be >= 1")

        if self.total_timeout < 30:
            errors.append("total_timeout must be >= 30 seconds")

        if self.connect_timeout < 5:
            errors.append("connect_timeout must be >= 5 seconds")

        if self.connect_timeout >= self.total_timeout:
            errors.append("connect_timeout must be less than total_timeout")

        if self.max_retries < 0 or self.max_retries > 10:
            errors.append("max_retries must be between 0 and 10")

        if self.keepalive_timeout < 30:
            errors.append("keepalive_timeout must be >= 30 seconds")

        if self.max_requests_per_session < 100:
            errors.append("max_requests_per_session must be >= 100")

        if not self.service_url or not self.service_url.startswith(("http://", "https://")):
            errors.append("service_url must be a valid HTTP/HTTPS URL")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


@dataclass
class SessionManager:
    """Container for session management attributes."""

    session: ClientSession | None = None
    created_at: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_count: int = 0


@dataclass
class HTTPQualityModel:
    """Main model for HTTP quality analysis operations.

    Provides a unified interface for HTTP-based quality analysis requests
    and response handling, incorporating session management and metrics collection.
    """

    config: QualityServiceConfig
    session_manager: SessionManager = field(default_factory=SessionManager)
    session_stats: SessionStats = field(default_factory=SessionStats)
    request_metrics: list[RequestMetrics] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the HTTP quality model after creation."""
        # Validate the configuration
        self.config.validate()

    def add_request_metric(self, metric: RequestMetrics) -> None:
        """Add a request metric to the collection."""
        self.request_metrics.append(metric)

        # Keep only the last 1000 metrics to prevent memory growth
        if len(self.request_metrics) > 1000:
            self.request_metrics = self.request_metrics[-1000:]

    def get_success_rate(self) -> float:
        """Calculate the success rate of recent requests."""
        if not self.request_metrics:
            return 1.0

        successful = sum(1 for metric in self.request_metrics if metric.success)
        return successful / len(self.request_metrics)

    def get_average_response_time(self) -> float:
        """Calculate the average response time in seconds."""
        if not self.request_metrics:
            return 0.0

        completed_metrics = [
            metric for metric in self.request_metrics if metric.end_time is not None
        ]

        if not completed_metrics:
            return 0.0

        total_time = sum(
            metric.end_time - metric.start_time
            for metric in completed_metrics
            if metric.end_time is not None
        )
        return total_time / len(completed_metrics)

    def cleanup_old_metrics(self, max_age_seconds: int = 3600) -> None:
        """Remove metrics older than the specified age."""
        current_time = datetime.now(UTC).timestamp()
        cutoff_time = current_time - max_age_seconds

        self.request_metrics = [
            metric for metric in self.request_metrics if metric.start_time > cutoff_time
        ]
