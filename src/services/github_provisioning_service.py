"""GitHub provisioning service — clones repos and reads files via CodingContainersClient."""

from __future__ import annotations

import logging
from typing import Any

from clients.coding_containers import CodingContainersClient

logger = logging.getLogger(__name__)

_CLONE_CONTAINER = "git-provisioner"


class GitHubProvisioningService:
    """Provides access to GitHub repositories for analysis."""

    def __init__(self) -> None:
        self._client = CodingContainersClient()

    async def provision_repository(self, repo_url: str) -> str:
        """Clone a repository inside the containers service and return the local path."""
        # Derive a target directory from the repo URL.
        repo_name = repo_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        target_dir = f"/workspace/{repo_name}"

        result = await self._client.execute(
            _CLONE_CONTAINER,
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            workdir="/workspace",
            timeout=120,
        )
        if result.get("exit_code") != 0:
            stderr = result.get("stderr", "")
            logger.error("git clone failed for %s: %s", repo_url, stderr)
            raise RuntimeError(f"Failed to clone repository: {stderr}")

        return target_dir

    async def get_file_content(self, repo_path: str, file_path: str) -> str:
        """Read a single file from a provisioned repository."""
        full_path = f"{repo_path.rstrip('/')}/{file_path.lstrip('/')}"
        content = await self._client.read_file(_CLONE_CONTAINER, full_path)
        if content is None:
            raise FileNotFoundError(f"File not found: {full_path}")
        return content

    async def get_repository_files(self, repo_path: str) -> list[dict[str, Any]]:
        """List files in a provisioned repository."""
        return await self._client.get_files(_CLONE_CONTAINER, path=repo_path)

    async def list_containers(self, tenant_id: str | None = None) -> list:
        """List containers via coding-containers service."""
        from app.core.container_manager import ContainerInfo

        raw = await self._client.list_containers(tenant_id=tenant_id)
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

    async def close(self) -> None:
        await self._client.close()


_github_provisioning_service: GitHubProvisioningService | None = None


def get_github_provisioning_service() -> GitHubProvisioningService:
    """Return a singleton GitHubProvisioningService instance."""
    global _github_provisioning_service  # noqa: PLW0603
    if _github_provisioning_service is None:
        _github_provisioning_service = GitHubProvisioningService()
    return _github_provisioning_service
