"""Unit tests for services.docker_service."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.docker_service import DockerService, get_docker_service


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.execute = AsyncMock(
        return_value={"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.1}
    )
    client.get_files = AsyncMock(return_value=[])
    client.list_containers = AsyncMock(return_value=[])
    client.get_container = AsyncMock(return_value=None)
    client.read_file = AsyncMock(return_value="content")
    client.close = AsyncMock()
    return client


@pytest.fixture
def svc(mock_client):
    service = DockerService()
    service._client = mock_client
    return service


# ------------------------------------------------------------------
# exec_in_container
# ------------------------------------------------------------------


class TestExecInContainer:
    async def test_delegates_to_client(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "hello",
            "stderr": "",
            "duration": 0.5,
        }
        result = await svc.exec_in_container("ctr", ["echo", "hello"], timeout=10)
        assert result["stdout"] == "hello"
        mock_client.execute.assert_awaited_once_with(
            "ctr", ["echo", "hello"], timeout=10, tenant_id=None
        )

    async def test_default_timeout(self, svc, mock_client):
        await svc.exec_in_container("ctr", "ls")
        mock_client.execute.assert_awaited_once_with("ctr", "ls", timeout=30, tenant_id=None)

    async def test_forwards_tenant(self, svc, mock_client):
        # coding-containers' exec is tenant-scoped; the caller's tenant must ride
        # on the exec or a customer workspace is 404'd.
        await svc.exec_in_container("ctr", "ls", tenant_id="tenant-a")
        mock_client.execute.assert_awaited_once_with("ctr", "ls", timeout=30, tenant_id="tenant-a")


# ------------------------------------------------------------------
# list_containers
# ------------------------------------------------------------------


class TestListContainers:
    async def test_returns_files_list(self, svc, mock_client):
        expected = [{"name": "ctr1"}, {"name": "ctr2"}]
        mock_client.list_containers.return_value = expected
        result = await svc.list_containers()
        assert result == expected
        mock_client.list_containers.assert_awaited_once_with(tenant_id=None)

    async def test_passes_tenant_id(self, svc, mock_client):
        await svc.list_containers(tenant_id="t-1")
        mock_client.list_containers.assert_awaited_once_with(tenant_id="t-1")


# ------------------------------------------------------------------
# is_docker_available
# ------------------------------------------------------------------


class TestIsDockerAvailable:
    def test_returns_true_in_running_loop(self, svc, mock_client):
        # Inside pytest-asyncio the loop is already running, so the
        # sync helper returns True without blocking.
        assert svc.is_docker_available() is True


# ------------------------------------------------------------------
# create_test_container
# ------------------------------------------------------------------


class TestCreateTestContainer:
    async def test_returns_id_from_json_stdout(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": json.dumps({"id": "abc-123"}),
            "stderr": "",
            "duration": 0.2,
        }
        cid = await svc.create_test_container({"image": "python:3.12"})
        assert cid == "abc-123"

    async def test_returns_raw_stdout_when_not_json(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "raw-id-456",
            "stderr": "",
            "duration": 0.1,
        }
        cid = await svc.create_test_container({"image": "python:3.12"})
        assert cid == "raw-id-456"

    async def test_falls_back_to_config_name(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "err",
            "duration": 0.1,
        }
        cid = await svc.create_test_container({"name": "fallback"})
        assert cid == "fallback"

    async def test_falls_back_to_image_when_no_name(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "err",
            "duration": 0.1,
        }
        cid = await svc.create_test_container({"image": "node:20"})
        assert cid == "node:20"


# ------------------------------------------------------------------
# get_language_container_image
# ------------------------------------------------------------------


class TestGetLanguageContainerImage:
    def test_known_language(self, svc):
        assert svc.get_language_container_image("python") == "python:3.12-slim"
        assert svc.get_language_container_image("javascript") == "node:20-slim"
        assert svc.get_language_container_image("Java") == "eclipse-temurin:21-jdk"

    def test_unknown_language(self, svc):
        assert svc.get_language_container_image("ruby") == "ruby:latest"


# ------------------------------------------------------------------
# find_container_by_name
# ------------------------------------------------------------------


class TestFindContainerByName:
    async def test_found(self, svc, mock_client):
        mock_client.get_container.return_value = {"name": "beta"}
        result = await svc.find_container_by_name("beta")
        assert result == {"name": "beta"}
        mock_client.get_container.assert_awaited_once_with("beta", tenant_id=None)

    async def test_not_found(self, svc, mock_client):
        mock_client.get_container.return_value = None
        result = await svc.find_container_by_name("missing")
        assert result is None
        mock_client.get_container.assert_awaited_once_with("missing", tenant_id=None)

    async def test_forwards_tenant(self, svc, mock_client):
        mock_client.get_container.return_value = {"name": "beta"}
        await svc.find_container_by_name("beta", tenant_id="tenant-9")
        mock_client.get_container.assert_awaited_once_with("beta", tenant_id="tenant-9")


# ------------------------------------------------------------------
# container lifecycle helpers
# ------------------------------------------------------------------


class TestContainerLifecycle:
    async def test_start_container(self, svc, mock_client):
        await svc.start_container("ctr-1")
        mock_client.execute.assert_awaited_once_with("ctr-1", ["start"], timeout=15)

    async def test_stop_container(self, svc, mock_client):
        await svc.stop_container("ctr-1")
        mock_client.execute.assert_awaited_once_with("ctr-1", ["stop"], timeout=15)

    async def test_remove_container_by_id(self, svc, mock_client):
        await svc.remove_container_by_id("ctr-1")
        mock_client.execute.assert_awaited_once_with("ctr-1", ["remove"], timeout=15)


# ------------------------------------------------------------------
# create_file_in_container
# ------------------------------------------------------------------


class TestCreateFileInContainer:
    async def test_returns_true_on_success(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration": 0.1,
        }
        ok = await svc.create_file_in_container("ctr", "/app/test.py", "print('hi')")
        assert ok is True

    async def test_returns_false_on_failure(self, svc, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "err",
            "duration": 0.1,
        }
        ok = await svc.create_file_in_container("ctr", "/app/test.py", "code")
        assert ok is False


# ------------------------------------------------------------------
# install_tools_in_container
# ------------------------------------------------------------------


class TestInstallTools:
    async def test_python_uses_pip(self, svc, mock_client):
        await svc.install_tools_in_container("ctr", ["pytest", "ruff"], "python")
        cmd = mock_client.execute.call_args[0][1]
        assert cmd[0] == "pip"
        assert "pytest" in cmd
        assert "ruff" in cmd

    async def test_javascript_uses_npm(self, svc, mock_client):
        await svc.install_tools_in_container("ctr", ["eslint"], "javascript")
        cmd = mock_client.execute.call_args[0][1]
        assert cmd[:3] == ["npm", "install", "-g"]
        assert "eslint" in cmd

    async def test_typescript_uses_npm(self, svc, mock_client):
        await svc.install_tools_in_container("ctr", ["tsc"], "TypeScript")
        cmd = mock_client.execute.call_args[0][1]
        assert cmd[0] == "npm"


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------


class TestSingleton:
    def test_get_docker_service_returns_instance(self):
        with patch("services.docker_service._docker_service", None):
            svc = get_docker_service()
            assert isinstance(svc, DockerService)
