"""Unit tests for app/core/container_metadata_registry.py.

Covers:
- ContainerMetadataRegistry.get_metadata
- ContainerMetadataRegistry.register
- ContainerMetadataRegistry.get_container
- get_container_registry singleton
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import pytest

from app.core.container_metadata_registry import (
    ContainerMetadataRegistry,
    get_container_registry,
)


@pytest.fixture(autouse=True)
def _ensure_real_container_manager():
    """Evict stub for app.core.container_manager so ContainerInfo is importable."""
    mod = sys.modules.get("app.core.container_manager")
    if mod is not None and not hasattr(mod, "ContainerInfo"):
        del sys.modules["app.core.container_manager"]
    yield
    # Clean up after to avoid leaking the real module into later test files
    # that might expect a stub.


class TestContainerMetadataRegistry:
    """Tests for ContainerMetadataRegistry."""

    def test_get_metadata_returns_none(self):
        reg = ContainerMetadataRegistry()
        assert reg.get_metadata("any-id") is None

    def test_register_does_not_raise(self):
        reg = ContainerMetadataRegistry()
        reg.register("test-id", {"key": "value"})

    async def test_get_container_returns_info(self):
        reg = ContainerMetadataRegistry()
        mock_client = AsyncMock()
        mock_client.get_container = AsyncMock(
            return_value={
                "container_name": "test-container",
                "codebase_path": "/app/code",
                "work_dir": "/app/code",
                "status": "running",
            }
        )

        with patch(
            "clients.coding_containers.get_coding_containers_client",
            return_value=mock_client,
        ):
            result = await reg.get_container("test-container")

        assert result is not None
        assert result.container_name == "test-container"
        assert result.codebase_path == "/app/code"
        assert result.status == "running"

    async def test_get_container_returns_none_when_not_found(self):
        reg = ContainerMetadataRegistry()
        mock_client = AsyncMock()
        mock_client.get_container = AsyncMock(return_value=None)

        with patch(
            "clients.coding_containers.get_coding_containers_client",
            return_value=mock_client,
        ):
            result = await reg.get_container("nonexistent")

        assert result is None


class TestGetContainerRegistry:
    """Tests for get_container_registry singleton."""

    def test_returns_instance(self):
        import app.core.container_metadata_registry as mod

        mod._registry = None
        reg = get_container_registry()
        assert isinstance(reg, ContainerMetadataRegistry)
        mod._registry = None

    def test_returns_same_instance(self):
        import app.core.container_metadata_registry as mod

        mod._registry = None
        r1 = get_container_registry()
        r2 = get_container_registry()
        assert r1 is r2
        mod._registry = None
