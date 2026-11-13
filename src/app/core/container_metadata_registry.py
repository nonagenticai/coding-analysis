"""Container metadata registry stub for coding-analysis.

Full implementation lives in coding-containers service.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ContainerMetadataRegistry:
    """Minimal container metadata interface for analysis endpoints."""

    def get_metadata(self, container_id: str) -> dict | None:
        return None

    def register(self, container_id: str, metadata: dict) -> None:
        pass

    async def get_container(self, container_name: str):
        """Get container info via coding-containers service."""
        from app.core.container_manager import ContainerInfo
        from clients.coding_containers import get_coding_containers_client

        client = get_coding_containers_client()
        data = await client.get_container(container_name)
        if data is None:
            return None
        return ContainerInfo(
            container_name=data.get("container_name", container_name),
            codebase_path=data.get("codebase_path", "/app/data/repo"),
            work_dir=data.get("work_dir", data.get("codebase_path", "/app/data/repo")),
            status=data.get("status", "unknown"),
            extra=data,
        )


_registry: ContainerMetadataRegistry | None = None


def get_container_registry() -> ContainerMetadataRegistry:
    """Get the global container metadata registry instance."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = ContainerMetadataRegistry()
    return _registry
