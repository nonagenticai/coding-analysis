"""Unit tests for services.github_provisioning_service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.github_provisioning_service import (
    GitHubProvisioningService,
    get_github_provisioning_service,
)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.execute = AsyncMock(
        return_value={"exit_code": 0, "stdout": "", "stderr": "", "duration": 1.0}
    )
    client.read_file = AsyncMock(return_value="file content")
    client.get_files = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


@pytest.fixture
def svc(mock_client):
    service = GitHubProvisioningService()
    service._client = mock_client
    return service


# ------------------------------------------------------------------
# provision_repository
# ------------------------------------------------------------------


class TestProvisionRepository:
    async def test_clones_repo_and_returns_path(self, svc, mock_client):
        path = await svc.provision_repository("https://github.com/org/myrepo.git")
        assert path == "/workspace/myrepo"
        mock_client.execute.assert_awaited_once()
        call_args = mock_client.execute.call_args
        cmd = call_args[0][1]
        assert "git" in cmd
        assert "clone" in cmd
        assert "https://github.com/org/myrepo.git" in cmd
        assert "/workspace/myrepo" in cmd

    async def test_strips_trailing_slash(self, svc, mock_client):
        path = await svc.provision_repository("https://github.com/org/myrepo/")
        assert path == "/workspace/myrepo"

    async def test_handles_url_without_git_suffix(self, svc, mock_client):
        path = await svc.provision_repository("https://github.com/org/plain-repo")
        assert path == "/workspace/plain-repo"

    async def test_raises_on_clone_failure(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 128,
            "stdout": "",
            "stderr": "fatal: repo not found",
            "duration": 0.5,
        }
        with pytest.raises(RuntimeError, match="Failed to clone"):
            await svc.provision_repository("https://github.com/org/bad")


# ------------------------------------------------------------------
# get_file_content
# ------------------------------------------------------------------


class TestGetFileContent:
    async def test_reads_file(self, svc, mock_client):
        mock_client.read_file.return_value = "print('hello')"
        content = await svc.get_file_content("/workspace/repo", "src/main.py")
        assert content == "print('hello')"
        mock_client.read_file.assert_awaited_once_with(
            "git-provisioner", "/workspace/repo/src/main.py"
        )

    async def test_raises_on_missing_file(self, svc, mock_client):
        mock_client.read_file.return_value = None
        with pytest.raises(FileNotFoundError):
            await svc.get_file_content("/workspace/repo", "missing.py")


# ------------------------------------------------------------------
# get_repository_files
# ------------------------------------------------------------------


class TestGetRepositoryFiles:
    async def test_lists_files(self, svc, mock_client):
        expected = [{"name": "a.py"}, {"name": "b.py"}]
        mock_client.get_files.return_value = expected
        result = await svc.get_repository_files("/workspace/repo")
        assert result == expected
        mock_client.get_files.assert_awaited_once_with("git-provisioner", path="/workspace/repo")


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------


class TestSingleton:
    def test_returns_instance(self):
        with patch("services.github_provisioning_service._github_provisioning_service", None):
            svc = get_github_provisioning_service()
            assert isinstance(svc, GitHubProvisioningService)
