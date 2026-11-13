"""Container manager for coding-analysis.

Delegates container operations to the coding-containers microservice
via the HTTP client adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from clients.coding_containers import get_coding_containers_client

logger = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    """Lightweight container info object matching attributes callers expect."""

    container_name: str
    codebase_path: str = "/app/data/repo"
    work_dir: str = "/app/data/repo"
    status: str = "unknown"
    repo_url: str = ""
    branch: str = "main"
    extra: dict = field(default_factory=dict)


class ContainerManager:
    """Container manager that delegates to the coding-containers service."""

    async def get_container(
        self, container_name: str, tenant_id: str | None = None
    ) -> ContainerInfo | None:
        """Look up a container by name via coding-containers service.

        Returns ContainerInfo or None if not found.
        Raises ValueError on lookup errors (callers catch this).
        """
        client = get_coding_containers_client()
        data = await client.get_container(container_name, tenant_id=tenant_id)
        if data is None:
            return None
        return ContainerInfo(
            container_name=data.get("container_name", container_name),
            codebase_path=data.get("codebase_path", "/app/data/repo"),
            work_dir=data.get("work_dir", data.get("codebase_path", "/app/data/repo")),
            status=data.get("status", "unknown"),
            repo_url=data.get("repo_url", ""),
            branch=data.get("branch", "main"),
            extra=data,
        )

    async def list_containers(self, tenant_id: str | None = None) -> list[ContainerInfo]:
        """List all containers via coding-containers service."""
        client = get_coding_containers_client()
        raw = await client.list_containers(tenant_id=tenant_id)
        return [
            ContainerInfo(
                container_name=c.get("container_name", c.get("name", "")),
                codebase_path=c.get("codebase_path", "/app/data/repo"),
                work_dir=c.get("work_dir", c.get("codebase_path", "/app/data/repo")),
                status=c.get("status", "unknown"),
                extra=c,
            )
            for c in raw
        ]

    async def has_container(self, container_name: str) -> bool:
        """Check if a container exists."""
        return (await self.get_container(container_name)) is not None

    async def execute_command(
        self,
        container_name: str,
        command: str,
        workdir: str = "/workspace",
        timeout: int = 30,
    ) -> dict:
        """Execute a command in a container via the coding-containers service.

        Returns dict with 'success', 'output', and 'error' keys.
        """
        client = get_coding_containers_client()
        result = await client.execute(
            container_name=container_name,
            command=command,
            workdir=workdir,
            timeout=timeout,
        )
        exit_code = result.get("exit_code", -1)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        return {
            "success": exit_code == 0,
            "output": stdout if stdout else stderr,
            "error": stderr if exit_code != 0 else "",
        }

    async def get_container_files(self, container_id: str, path: str = "/") -> list[dict]:
        """Get file listing from a container.

        Args:
            container_id: Name/ID of the container.
            path: Directory path inside the container.

        Returns:
            List of file info dicts.
        """
        client = get_coding_containers_client()
        return await client.get_files(container_id, path)

    async def read_container_file(self, container_id: str, path: str) -> str:
        """Read a file from a container.

        Args:
            container_id: Name/ID of the container.
            path: File path inside the container.

        Returns:
            File content as string, or empty string if not found.
        """
        client = get_coding_containers_client()
        content = await client.read_file(container_id, path)
        return content if content is not None else ""


@lru_cache
def get_container_manager() -> ContainerManager:
    return ContainerManager()
