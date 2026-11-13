"""Health API models stub.

TODO: Implement health-related API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SystemHealth(BaseModel):
    """System health."""

    status: str = "unknown"
    uptime: float = 0.0


class ResourceMetrics(BaseModel):
    """Resource metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0


class DatabaseHealth(BaseModel):
    """Database health."""

    connected: bool = False
    latency_ms: float = 0.0


class ContainerHealth(BaseModel):
    """Container health."""

    container_id: str = ""
    healthy: bool = False
    details: dict[str, Any] = {}


class ContainerHealthResponse(BaseModel):
    """Container health response."""

    containers: list[ContainerHealth] = []


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = "unknown"
    version: str = ""
    checks: dict[str, Any] = {}


class DetailedHealthResponse(BaseModel):
    """Detailed health response."""

    system: SystemHealth = SystemHealth()
    database: DatabaseHealth = DatabaseHealth()
    resources: ResourceMetrics = ResourceMetrics()


class MetricsResponse(BaseModel):
    """Metrics response."""

    metrics: dict[str, Any] = {}
