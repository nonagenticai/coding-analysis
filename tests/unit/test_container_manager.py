"""Unit tests for app.core.container_manager and app.core.redis_service modules."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.core.container_manager import ContainerManager, get_container_manager

# ---------------------------------------------------------------------------
# get_container_files()
# ---------------------------------------------------------------------------


class TestGetContainerFiles:
    """Test ContainerManager.get_container_files()."""

    async def test_delegates_to_client_get_files(self, mock_container_client):
        expected_files = [{"name": "main.py", "size": 256}, {"name": "utils.py", "size": 128}]
        mock_container_client.get_files = AsyncMock(return_value=expected_files)

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.get_container_files("ctr-123", path="/workspace")

        assert result == expected_files
        mock_container_client.get_files.assert_awaited_once_with("ctr-123", "/workspace")

    async def test_default_path_is_root(self, mock_container_client):
        mock_container_client.get_files = AsyncMock(return_value=[])

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.get_container_files("ctr-abc")

        assert result == []
        mock_container_client.get_files.assert_awaited_once_with("ctr-abc", "/")

    async def test_returns_empty_list_on_client_error(self, mock_container_client):
        mock_container_client.get_files = AsyncMock(return_value=[])

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.get_container_files("ctr-err")

        assert result == []


# ---------------------------------------------------------------------------
# read_container_file()
# ---------------------------------------------------------------------------


class TestReadContainerFile:
    """Test ContainerManager.read_container_file()."""

    async def test_returns_content_when_found(self, mock_container_client):
        mock_container_client.read_file = AsyncMock(return_value="print('hi')")

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.read_container_file("ctr-1", "/workspace/main.py")

        assert result == "print('hi')"
        mock_container_client.read_file.assert_awaited_once_with("ctr-1", "/workspace/main.py")

    async def test_returns_empty_string_when_none(self, mock_container_client):
        mock_container_client.read_file = AsyncMock(return_value=None)

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.read_container_file("ctr-1", "/missing.py")

        assert result == ""

    async def test_returns_empty_string_content(self, mock_container_client):
        mock_container_client.read_file = AsyncMock(return_value="")

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.read_container_file("ctr-1", "/empty.py")

        assert result == ""

    async def test_returns_multiline_content(self, mock_container_client):
        content = "line1\nline2\nline3\n"
        mock_container_client.read_file = AsyncMock(return_value=content)

        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.read_container_file("ctr-1", "/multi.py")

        assert result == content


# ---------------------------------------------------------------------------
# get_container_manager() singleton
# ---------------------------------------------------------------------------


class TestGetContainerManager:
    """Test the module-level singleton factory."""

    def test_returns_container_manager_instance(self):
        get_container_manager.cache_clear()
        manager = get_container_manager()
        assert isinstance(manager, ContainerManager)

    def test_returns_same_instance(self):
        get_container_manager.cache_clear()
        a = get_container_manager()
        b = get_container_manager()
        assert a is b
        get_container_manager.cache_clear()


# ---------------------------------------------------------------------------
# Uncovered methods: get_container, list_containers, has_container,
# execute_command (lines 42-46, 58-60, 73, 86-96)
# ---------------------------------------------------------------------------

from app.core.container_manager import ContainerInfo


class TestGetContainer:
    """Test ContainerManager.get_container() (lines 42-54)."""

    async def test_returns_container_info(self, mock_container_client):
        mock_container_client.get_container = AsyncMock(
            return_value={
                "container_name": "my-ctr",
                "codebase_path": "/app/code",
                "work_dir": "/app/code",
                "status": "running",
                "repo_url": "https://github.com/test",
                "branch": "develop",
            }
        )
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.get_container("my-ctr", tenant_id="t1")

        assert isinstance(result, ContainerInfo)
        assert result.container_name == "my-ctr"
        assert result.codebase_path == "/app/code"
        assert result.status == "running"
        assert result.repo_url == "https://github.com/test"
        assert result.branch == "develop"

    async def test_returns_none_when_not_found(self, mock_container_client):
        mock_container_client.get_container = AsyncMock(return_value=None)
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.get_container("missing", tenant_id="t1")

        assert result is None


class TestListContainers:
    """Test ContainerManager.list_containers() (lines 58-69)."""

    async def test_returns_list_of_container_info(self, mock_container_client):
        mock_container_client.list_containers = AsyncMock(
            return_value=[
                {"container_name": "ctr-1", "status": "running"},
                {"name": "ctr-2", "codebase_path": "/custom"},
            ]
        )
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.list_containers(tenant_id="t1")

        assert len(result) == 2
        assert result[0].container_name == "ctr-1"
        assert result[0].status == "running"
        assert result[1].container_name == "ctr-2"
        assert result[1].codebase_path == "/custom"


class TestHasContainer:
    """Test ContainerManager.has_container() (line 73)."""

    async def test_returns_true_when_exists(self, mock_container_client):
        mock_container_client.get_container = AsyncMock(return_value={"container_name": "exists"})
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            assert await manager.has_container("exists") is True

    async def test_returns_false_when_missing(self, mock_container_client):
        mock_container_client.get_container = AsyncMock(return_value=None)
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            assert await manager.has_container("missing") is False


class TestExecuteCommand:
    """Test ContainerManager.execute_command() (lines 86-100)."""

    async def test_successful_command(self, mock_container_client):
        mock_container_client.execute = AsyncMock(
            return_value={"exit_code": 0, "stdout": "hello world", "stderr": ""}
        )
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.execute_command("ctr-1", "echo hello", workdir="/app")

        assert result["success"] is True
        assert result["output"] == "hello world"
        assert result["error"] == ""

    async def test_failed_command(self, mock_container_client):
        mock_container_client.execute = AsyncMock(
            return_value={"exit_code": 1, "stdout": "", "stderr": "command not found"}
        )
        with patch(
            "app.core.container_manager.get_coding_containers_client",
            return_value=mock_container_client,
        ):
            manager = ContainerManager()
            result = await manager.execute_command("ctr-1", "bad_cmd")

        assert result["success"] is False
        assert result["error"] == "command not found"


# ---------------------------------------------------------------------------
# RedisService tests (src/app/core/redis_service.py)
# Covers lines: 24, 40-41, 44-48, 51-53
# ---------------------------------------------------------------------------


from app.core.redis_service import RedisService, get_redis_service


class TestRedisService:
    """Test RedisService methods."""

    async def test_client_property_initially_none(self):
        """Client property is None before connect (line 24)."""
        svc = RedisService(url="redis://localhost:6379/0")
        assert svc.client is None

    async def test_get_client_creates_connection(self):
        """get_client creates Redis connection on first call."""
        svc = RedisService(url="redis://localhost:6379/0")
        with patch("app.core.redis_service.redis.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_from_url.return_value = mock_client
            client = await svc.get_client()
            assert client is mock_client
            assert svc.client is mock_client

    async def test_delete_keys(self):
        """delete method delegates to client.delete (lines 40-41)."""
        svc = RedisService()
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(return_value=2)
        svc._client = mock_client
        result = await svc.delete("key1", "key2")
        assert result == 2
        mock_client.delete.assert_awaited_once_with("key1", "key2")

    async def test_async_scan_keys(self):
        """async_scan_keys scans for matching keys (lines 44-48)."""
        svc = RedisService()
        mock_client = AsyncMock()

        async def mock_scan_iter(match=None):
            for key in ["prefix:a", "prefix:b"]:
                yield key

        mock_client.scan_iter = mock_scan_iter
        svc._client = mock_client
        result = await svc.async_scan_keys("prefix:*")
        assert result == ["prefix:a", "prefix:b"]

    async def test_close(self):
        """close shuts down the client (lines 51-53)."""
        svc = RedisService()
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        svc._client = mock_client
        await svc.close()
        mock_client.close.assert_awaited_once()
        assert svc._client is None

    async def test_close_when_no_client(self):
        """close with no client does nothing."""
        svc = RedisService()
        await svc.close()  # Should not raise

    def test_get_redis_service_singleton(self):
        """get_redis_service returns same instance."""
        get_redis_service.cache_clear()
        a = get_redis_service()
        b = get_redis_service()
        assert a is b
        get_redis_service.cache_clear()
