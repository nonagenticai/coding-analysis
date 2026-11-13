"""Python test suite configuration stub.

TODO: Implement test configuration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TestConfig(BaseModel):
    """Test configuration."""

    test_timeout: int = 30
    max_retries: int = 3
    isolation_level: str = "container"
    allowed_imports: list[str] = []
    blocked_imports: list[str] = []
    max_memory_mb: int = 512
    max_cpu_time_s: int = 60
    working_directory: str = "/tmp/tests"  # nosec B108
    environment: dict[str, Any] = {}
