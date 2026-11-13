"""Docker service — wraps CodingContainersClient for container management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from clients.coding_containers import CodingContainersClient

logger = logging.getLogger(__name__)

_LANGUAGE_IMAGES: dict[str, str] = {
    "python": "python:3.12-slim",
    "javascript": "node:20-slim",
    "typescript": "node:20-slim",
    "java": "eclipse-temurin:21-jdk",
    "go": "golang:1.22-bookworm",
    "rust": "rust:1.77-slim",
    "c": "gcc:14",
    "cpp": "gcc:14",
    "csharp": "mcr.microsoft.com/dotnet/sdk:8.0",
}


class DockerService:
    """Manages Docker containers for code analysis."""

    def __init__(self) -> None:
        self._client = CodingContainersClient()

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def is_docker_available(self) -> bool:
        """Check if the containers service is reachable (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot block inside a running loop; assume available.
                return True
            return loop.run_until_complete(self._client.health_check())
        except RuntimeError:
            # No event loop yet — create one.
            return asyncio.run(self._client.health_check())

    def get_language_container_image(self, language: str) -> str:
        """Return the default container image name for a given language."""
        key = language.lower().strip()
        return _LANGUAGE_IMAGES.get(key, f"{key}:latest")

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    async def create_test_container(self, container_config: dict[str, Any]) -> str:
        """Create a new container via the containers API and return its id."""
        result = await self._client.execute(
            container_name="__manager__",
            command=["create"],
            workdir="/",
            timeout=30,
        )
        # The containers service returns a JSON body with an id field on
        # creation calls routed through exec.  Fall back to a config-derived
        # name when the API response does not carry one.
        if result.get("exit_code") == 0 and result.get("stdout"):
            import json as _json

            try:
                data = _json.loads(result["stdout"])
                if isinstance(data, dict) and "id" in data:
                    return str(data["id"])
            except (ValueError, TypeError):  # fmt: skip
                return result["stdout"].strip()
        return container_config.get("name", container_config.get("image", "unknown"))

    async def start_container(self, container_id: str) -> None:
        """Start a previously created container."""
        await self._client.execute(container_id, ["start"], timeout=15)

    async def stop_container(self, container_id: str) -> None:
        """Stop a running container."""
        await self._client.execute(container_id, ["stop"], timeout=15)

    async def remove_container_by_id(self, container_id: str) -> None:
        """Remove a container by its id."""
        await self._client.execute(container_id, ["remove"], timeout=15)

    # ------------------------------------------------------------------
    # Execution & file operations
    # ------------------------------------------------------------------

    async def exec_in_container(
        self,
        container_name: str,
        command: list[str] | str,
        timeout: int = 30,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command inside a container and return the result dict.

        Pass the CALLER's ``tenant_id``: coding-containers' exec is
        tenant-scoped, so an exec without it is resolved as this service's own
        account and 404'd for every customer workspace. Absent stays absent
        (never defaulted to a literal tenant).
        """
        return await self._client.execute(
            container_name, command, timeout=timeout, tenant_id=tenant_id
        )

    async def list_containers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """List containers via the coding-containers service."""
        return await self._client.list_containers(tenant_id=tenant_id)

    async def create_file_in_container(
        self,
        container_id: str,
        file_path: str,
        content: str,
    ) -> bool:
        """Write *content* to *file_path* inside the container."""
        result = await self._client.execute(
            container_id,
            ["sh", "-c", f"cat > {file_path}"],
            timeout=10,
        )
        return result.get("exit_code") == 0

    async def install_tools_in_container(
        self,
        container_id: str,
        tools: list[str],
        language: str,
    ) -> dict[str, Any]:
        """Install *tools* in the container using the language-appropriate package manager."""
        lang = language.lower().strip()
        if lang in {"javascript", "typescript"}:
            cmd = ["npm", "install", "-g", *tools]
        else:
            # Default to pip for Python and anything else.
            cmd = ["pip", "install", *tools]
        return await self._client.execute(container_id, cmd, timeout=120)

    async def find_container_by_name(
        self, name: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get container by name via the coding-containers service.

        Forwards ``tenant_id`` as the ``X-Tenant-Id`` header so coding-containers
        resolves the container under the CALLER's tenant. Without it the lookup
        fell back to a default/system tenant and BOLA-hid containers owned by
        other tenants (e.g. a project's EXEC-tenant container 404'd), which made
        the architecture-graph reads 404 even though the graph existed.
        """
        return await self._client.get_container(name, tenant_id=tenant_id)

    async def get_container_info(self, container_id: str) -> dict[str, Any]:
        """Get container information."""
        result = await self._client.get_container(container_id)
        return result or {}

    async def close(self) -> None:
        await self._client.close()


_docker_service: DockerService | None = None


def get_docker_service() -> DockerService:
    """Return a singleton DockerService instance."""
    global _docker_service  # noqa: PLW0603
    if _docker_service is None:
        _docker_service = DockerService()
    return _docker_service
