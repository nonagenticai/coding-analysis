"""Unit tests for processing.task_queue, processing.task_validator,
and processing.task_result_processor modules."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from processing.task_queue import ProcessingTaskQueue, TaskQueueConfig
from processing.task_result_processor import (
    AnalysisMetadataManager,
    FileProcessingResultHandler,
    TaskResultProcessor,
)
from processing.task_validator import (
    FileSystemValidator,
    TaskErrorHandler,
    TaskMonitor,
    TaskParameterValidator,
)

# ===========================================================================
# TaskQueueConfig
# ===========================================================================


class TestTaskQueueConfig:
    def test_defaults(self):
        cfg = TaskQueueConfig()
        assert cfg.max_retries == 3
        assert cfg.retry_delay == 1.0
        assert cfg.connection_timeout == 10.0
        assert cfg.operation_timeout == 5.0
        assert cfg.namespace == "code_graph:"

    def test_custom_values(self):
        cfg = TaskQueueConfig(max_retries=5, namespace="custom:")
        assert cfg.max_retries == 5
        assert cfg.namespace == "custom:"


# ===========================================================================
# ProcessingTaskQueue
# ===========================================================================


class TestProcessingTaskQueue:
    """Tests for ProcessingTaskQueue."""

    def _make_queue(self, redis_service=None, config=None):
        rs = redis_service or MagicMock()
        return ProcessingTaskQueue(rs, config)

    # --- init ---

    def test_init_sets_keys(self):
        q = self._make_queue()
        assert "queue" in q.keys
        assert "processing" in q.keys
        assert "results_prefix" in q.keys

    # --- _get_redis_client_with_retry ---

    @pytest.mark.asyncio
    async def test_get_redis_client_success(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()

        rs = MagicMock()
        rs.client = mock_client  # awaitable property simulated

        q = self._make_queue(redis_service=rs, config=TaskQueueConfig(max_retries=1))
        # Patch wait_for to just return the client directly
        with patch("processing.task_queue.asyncio.wait_for", side_effect=[mock_client, None]):
            client = await q._get_redis_client_with_retry()
            assert client is mock_client

    @pytest.mark.asyncio
    async def test_get_redis_client_returns_none_when_no_service(self):
        q = self._make_queue(config=TaskQueueConfig(max_retries=1))
        q.redis_service = None
        result = await q._get_redis_client_with_retry()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_client_retry_on_timeout(self):
        rs = MagicMock()
        rs.client = AsyncMock()

        q = self._make_queue(
            redis_service=rs, config=TaskQueueConfig(max_retries=2, retry_delay=0.01)
        )

        with patch("processing.task_queue.asyncio.wait_for", side_effect=TimeoutError("timeout")):
            with patch("processing.task_queue.asyncio.sleep", new_callable=AsyncMock):
                result = await q._get_redis_client_with_retry()
                assert result is None

    # --- add_task ---

    @pytest.mark.asyncio
    async def test_add_task_redis_success(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.rpush = AsyncMock()

        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(return_value=mock_client)

        task = {"id": "task-1", "type": "analysis"}
        with patch("processing.task_queue.asyncio.wait_for", new_callable=AsyncMock) as mock_wf:
            mock_wf.return_value = None
            result = await q.add_task(task)
            assert result is True

    @pytest.mark.asyncio
    async def test_add_task_falls_back_to_local(self):
        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(return_value=None)

        task = {"id": "task-2", "type": "analysis"}
        result = await q.add_task(task)
        assert result is True
        assert q.local_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_add_task_catches_error_falls_back(self):
        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(side_effect=ConnectionError("fail"))

        task = {"id": "task-3", "type": "analysis"}
        result = await q.add_task(task)
        assert result is True
        assert q.local_queue.qsize() == 1

    # --- get_next_task ---

    @pytest.mark.asyncio
    async def test_get_next_task_from_redis(self):
        q = self._make_queue()
        expected = {"id": "t1", "type": "test"}
        q._get_task_from_redis = AsyncMock(return_value=expected)
        q._get_task_from_local_queue = AsyncMock(return_value=None)

        result = await q.get_next_task(timeout=1)
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_next_task_falls_back_to_local(self):
        q = self._make_queue()
        expected = {"id": "t2", "type": "local"}
        q._get_task_from_redis = AsyncMock(return_value=None)
        q._get_task_from_local_queue = AsyncMock(return_value=expected)

        result = await q.get_next_task(timeout=1)
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_next_task_returns_none_when_empty(self):
        q = self._make_queue()
        q._get_task_from_redis = AsyncMock(return_value=None)
        q._get_task_from_local_queue = AsyncMock(return_value=None)

        result = await q.get_next_task(timeout=1)
        assert result is None

    # --- _get_task_from_redis ---

    @pytest.mark.asyncio
    async def test_get_task_from_redis_success(self):
        task_data = {"id": "r1", "type": "test"}
        mock_client = AsyncMock()

        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(return_value=mock_client)

        with patch("processing.task_queue.asyncio.wait_for") as mock_wf:
            # First call: blpop returns (key, json)
            # Second call: hset
            mock_wf.side_effect = [
                (b"queue", json.dumps(task_data).encode()),
                None,
            ]
            result = await q._get_task_from_redis(timeout=1)
            assert result == task_data

    @pytest.mark.asyncio
    async def test_get_task_from_redis_no_client(self):
        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(return_value=None)
        result = await q._get_task_from_redis(timeout=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_from_redis_timeout(self):
        mock_client = AsyncMock()
        q = self._make_queue()
        q._get_redis_client_with_retry = AsyncMock(return_value=mock_client)

        with patch("processing.task_queue.asyncio.wait_for", side_effect=TimeoutError):
            result = await q._get_task_from_redis(timeout=1)
            assert result is None

    # --- _get_task_from_local_queue ---

    @pytest.mark.asyncio
    async def test_get_task_from_local_queue_timeout(self):
        q = self._make_queue()
        result = await q._get_task_from_local_queue(timeout=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_from_local_queue_has_item(self):
        q = self._make_queue()
        task = {"id": "local1"}
        await q.local_queue.put(task)
        result = await q._get_task_from_local_queue(timeout=1)
        assert result == task

    # --- update_task_status ---

    @pytest.mark.asyncio
    async def test_update_task_status_with_redis(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.update_task_status("task-1", "running")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_task_status_fallback(self):
        rs = MagicMock()
        rs.client = None

        q = self._make_queue(redis_service=rs)
        result = await q.update_task_status("task-1", "running")
        assert result is True
        assert "task-1:status" in q.local_results

    @pytest.mark.asyncio
    async def test_update_task_status_no_redis_service(self):
        q = self._make_queue()
        q.redis_service = None
        result = await q.update_task_status("task-x", "done")
        assert result is True
        assert "task-x:status" in q.local_results

    # --- complete_task ---

    @pytest.mark.asyncio
    async def test_complete_task_with_redis(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.hdel = AsyncMock(return_value=1)

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.complete_task("task-1", {"data": "value"})
        assert result is True

    @pytest.mark.asyncio
    async def test_complete_task_fallback(self):
        rs = MagicMock()
        rs.client = None

        q = self._make_queue(redis_service=rs)
        result = await q.complete_task("task-1", {"data": "value"})
        assert result is True

    # --- fail_task ---

    @pytest.mark.asyncio
    async def test_fail_task_with_redis(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.hdel = AsyncMock(return_value=1)

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.fail_task("task-1", "some error")
        assert result is True

    @pytest.mark.asyncio
    async def test_fail_task_fallback(self):
        rs = MagicMock()
        rs.client = None

        q = self._make_queue(redis_service=rs)
        result = await q.fail_task("task-1", "some error")
        assert result is True

    # --- get_task_status ---

    @pytest.mark.asyncio
    async def test_get_task_status_from_redis(self):
        status = {"status": "running", "updated_at": 123.0}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=json.dumps(status))

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_task_status("task-1")
        assert result == status

    @pytest.mark.asyncio
    async def test_get_task_status_fallback(self):
        rs = MagicMock()
        rs.client = None

        q = self._make_queue(redis_service=rs)
        q.local_results["task-1:status"] = {"status": "done"}
        result = await q.get_task_status("task-1")
        assert result == {"status": "done"}

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_task_status("nonexistent")
        assert result is None

    # --- get_task_result ---

    @pytest.mark.asyncio
    async def test_get_task_result_from_redis(self):
        result_data = {"success": True, "output": "hello"}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=json.dumps(result_data))

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_task_result("task-1")
        assert result == result_data

    @pytest.mark.asyncio
    async def test_get_task_result_not_found(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_task_result("nonexistent")
        assert result is None

    # --- get_all_tasks ---

    @pytest.mark.asyncio
    async def test_get_all_tasks_with_results(self):
        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=[b"code_graph:task:1", b"code_graph:task:2"])

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_all_tasks()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_all_tasks_empty(self):
        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=[])

        rs = MagicMock()
        rs.client = mock_client

        q = self._make_queue(redis_service=rs)
        result = await q.get_all_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_tasks_no_redis(self):
        q = self._make_queue()
        q.redis_service = None
        result = await q.get_all_tasks()
        assert result == []


# ===========================================================================
# TaskParameterValidator
# ===========================================================================


class TestTaskParameterValidator:
    # --- validate_codebase_analysis_params ---

    def test_valid_codebase_analysis(self):
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/some/path", "scan_depth": 2, "code_graph_id": "graph1"}
        )
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["validated_params"]["codebase_path"] == "/some/path"
        assert result["validated_params"]["scan_depth"] == 2

    def test_missing_codebase_path(self):
        result = TaskParameterValidator.validate_codebase_analysis_params({})
        assert result["is_valid"] is False
        assert any("codebase_path" in e for e in result["errors"])

    def test_invalid_scan_depth(self):
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/path", "scan_depth": -1}
        )
        assert result["is_valid"] is False
        assert any("scan_depth" in e for e in result["errors"])

    def test_invalid_scan_depth_string(self):
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/path", "scan_depth": "deep"}
        )
        assert result["is_valid"] is False

    def test_empty_code_graph_id(self):
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/path", "code_graph_id": "   "}
        )
        assert result["is_valid"] is False
        assert any("code_graph_id" in e for e in result["errors"])

    def test_defaults_applied(self):
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/path", "code_graph_id": "g1"}
        )
        assert result["is_valid"] is True
        assert result["validated_params"]["scan_depth"] == 1
        assert result["validated_params"]["code_graph_id"] == "g1"

    def test_missing_code_graph_id_is_invalid(self):
        # No "default" fallback: an untagged task must fail validation rather
        # than silently funnel into a shared graph (cross-tenant bleed).
        result = TaskParameterValidator.validate_codebase_analysis_params(
            {"codebase_path": "/path"}
        )
        assert result["is_valid"] is False
        assert any("code_graph_id" in e for e in result["errors"])

    # --- validate_centrality_calculation_params ---

    def test_valid_centrality_params(self):
        result = TaskParameterValidator.validate_centrality_calculation_params(
            {"code_graph_id": "graph1"}
        )
        assert result["is_valid"] is True

    def test_centrality_missing_code_graph_id_is_invalid(self):
        result = TaskParameterValidator.validate_centrality_calculation_params({})
        assert result["is_valid"] is False
        assert any("code_graph_id" in e for e in result["errors"])

    def test_centrality_invalid_code_graph_id(self):
        result = TaskParameterValidator.validate_centrality_calculation_params(
            {"code_graph_id": ""}
        )
        assert result["is_valid"] is False

    def test_centrality_invalid_node_type(self):
        result = TaskParameterValidator.validate_centrality_calculation_params({"node_type": 123})
        assert result["is_valid"] is False
        assert any("node_type" in e for e in result["errors"])

    def test_centrality_valid_node_type_none(self):
        result = TaskParameterValidator.validate_centrality_calculation_params(
            {"code_graph_id": "g1", "node_type": None}
        )
        assert result["is_valid"] is True

    def test_centrality_valid_node_type_string(self):
        result = TaskParameterValidator.validate_centrality_calculation_params(
            {"code_graph_id": "g1", "node_type": "function"}
        )
        assert result["is_valid"] is True

    # --- validate_multi_language_analysis_params ---

    def test_valid_multi_language_params(self):
        result = TaskParameterValidator.validate_multi_language_analysis_params(
            {"codebase_path": "/path", "code_graph_id": "g1"}
        )
        assert result["is_valid"] is True

    def test_multi_language_missing_path(self):
        result = TaskParameterValidator.validate_multi_language_analysis_params({})
        assert result["is_valid"] is False
        assert any("codebase_path" in e for e in result["errors"])

    def test_multi_language_invalid_code_graph_id(self):
        result = TaskParameterValidator.validate_multi_language_analysis_params(
            {"codebase_path": "/path", "code_graph_id": ""}
        )
        assert result["is_valid"] is False


# ===========================================================================
# TaskErrorHandler
# ===========================================================================


class TestTaskErrorHandler:
    def test_categorize_file_system_error(self):
        assert TaskErrorHandler.categorize_error(FileNotFoundError()) == "FILE_SYSTEM"
        assert TaskErrorHandler.categorize_error(OSError()) == "FILE_SYSTEM"
        assert TaskErrorHandler.categorize_error(PermissionError()) == "FILE_SYSTEM"

    def test_categorize_data_processing_error(self):
        assert TaskErrorHandler.categorize_error(ValueError()) == "DATA_PROCESSING"
        assert TaskErrorHandler.categorize_error(KeyError()) == "DATA_PROCESSING"
        assert TaskErrorHandler.categorize_error(TypeError()) == "DATA_PROCESSING"

    def test_categorize_runtime_error(self):
        assert TaskErrorHandler.categorize_error(RuntimeError()) == "RUNTIME"
        assert TaskErrorHandler.categorize_error(MemoryError()) == "RUNTIME"
        assert TaskErrorHandler.categorize_error(RecursionError()) == "RUNTIME"

    def test_categorize_import_error(self):
        assert TaskErrorHandler.categorize_error(ImportError()) == "IMPORT_MODULE"
        assert TaskErrorHandler.categorize_error(ModuleNotFoundError()) == "IMPORT_MODULE"

    def test_categorize_unknown_error(self):
        assert TaskErrorHandler.categorize_error(SystemExit()) == "UNKNOWN"
        assert TaskErrorHandler.categorize_error(StopIteration()) == "UNKNOWN"

    def test_create_error_response(self):
        err = ValueError("bad value")
        resp = TaskErrorHandler.create_error_response(err, context="testing")
        assert resp["success"] is False
        assert resp["error"] == "bad value"
        assert resp["error_category"] == "DATA_PROCESSING"
        assert resp["error_type"] == "ValueError"
        assert resp["context"] == "testing"

    def test_create_error_response_no_context(self):
        resp = TaskErrorHandler.create_error_response(RuntimeError("boom"))
        assert resp["context"] == ""

    def test_log_error_filesystem(self):
        with patch("processing.task_validator.logger") as mock_logger:
            TaskErrorHandler.log_error(FileNotFoundError("x"), "ctx", "scan")
            mock_logger.error.assert_called_once()

    def test_log_error_runtime(self):
        with patch("processing.task_validator.logger") as mock_logger:
            TaskErrorHandler.log_error(RuntimeError("x"), "ctx", "scan")
            mock_logger.error.assert_called_once()

    def test_log_error_data_warning(self):
        with patch("processing.task_validator.logger") as mock_logger:
            TaskErrorHandler.log_error(ValueError("x"), "ctx", "scan")
            mock_logger.warning.assert_called_once()


# ===========================================================================
# TaskMonitor
# ===========================================================================


class TestTaskMonitor:
    def test_initial_state(self):
        m = TaskMonitor()
        assert m._total_tasks == 0
        assert m._successful_tasks == 0
        assert m._failed_tasks == 0

    def test_record_task_start(self):
        m = TaskMonitor()
        m.record_task_start("analysis", {"id": "1"})
        assert m._total_tasks == 1

    def test_record_task_success(self):
        m = TaskMonitor()
        m.record_task_start("analysis", {})
        m.record_task_success("analysis", 1.5)
        assert m._successful_tasks == 1

    def test_record_task_failure(self):
        m = TaskMonitor()
        m.record_task_start("analysis", {})
        m.record_task_failure("analysis", ValueError("bad"), 2.0)
        assert m._failed_tasks == 1
        assert "DATA_PROCESSING" in m._error_categories

    def test_task_stats_property(self):
        m = TaskMonitor()
        m.record_task_start("a", {})
        m.record_task_success("a", 1.0)
        stats = m.task_stats
        assert stats["total_tasks"] == 1
        assert stats["successful_tasks"] == 1
        assert stats["failed_tasks"] == 0

    def test_get_statistics_empty(self):
        m = TaskMonitor()
        result = m.get_statistics()
        assert result == {"message": "No tasks executed yet"}

    def test_get_statistics_with_data(self):
        m = TaskMonitor()
        m.record_task_start("a", {})
        m.record_task_start("b", {})
        m.record_task_success("a", 1.0)
        m.record_task_failure("b", RuntimeError("x"), 2.0)

        stats = m.get_statistics()
        assert stats["total_tasks"] == 2
        assert stats["successful_tasks"] == 1
        assert stats["failed_tasks"] == 1
        assert stats["success_rate"] == 50.0
        assert stats["failure_rate"] == 50.0

    def test_reset_statistics(self):
        m = TaskMonitor()
        m.record_task_start("a", {})
        m.record_task_success("a", 1.0)
        m.reset_statistics()
        assert m._total_tasks == 0
        assert m._successful_tasks == 0
        assert m._failed_tasks == 0
        assert m._error_categories == {}


# ===========================================================================
# FileSystemValidator
# ===========================================================================


class TestFileSystemValidator:
    def test_valid_path(self, tmp_path):
        result = FileSystemValidator.validate_codebase_path(str(tmp_path))
        assert result["is_valid"] is True
        assert "absolute_path" in result
        assert "size_info" in result

    def test_nonexistent_path(self):
        result = FileSystemValidator.validate_codebase_path("/nonexistent/path/xyz")
        assert result["is_valid"] is False
        assert "does not exist" in result["error"]

    def test_file_not_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = FileSystemValidator.validate_codebase_path(str(f))
        assert result["is_valid"] is False
        assert "not a directory" in result["error"]

    def test_directory_info(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x")
        result = FileSystemValidator.validate_codebase_path(str(tmp_path))
        assert result["is_valid"] is True
        info = result["size_info"]
        assert info["total_files"] >= 1
        assert info["total_directories"] >= 1


# ===========================================================================
# TaskResultProcessor
# ===========================================================================


class TestTaskResultProcessor:
    @pytest.mark.asyncio
    async def test_get_node_count(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(42,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        count = await proc._get_node_count(mock_conn, "g1")
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_node_count_none(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        count = await proc._get_node_count(mock_conn, "g1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_edge_count(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(10,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        count = await proc._get_edge_count(mock_conn, "g1")
        assert count == 10

    @pytest.mark.asyncio
    async def test_build_analysis_result_shallow(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(5,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        import time

        start = time.time()
        result = await proc.build_analysis_result(
            mock_conn, 10, scan_depth=1, start_time=start, code_graph_id="g1"
        )
        assert result["success"] is True
        assert result["processed_files"] == 10
        assert result["node_count"] == 5
        assert result["edge_count"] == 5
        assert result["important_nodes"] == []
        assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_build_analysis_result_deep_scan(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(5,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        import time

        start = time.time()

        # Patch calculate_pagerank on the stub module so the local import picks it up
        mock_pagerank = AsyncMock(return_value={"n1": 0.5, "n2": 0.3})
        with patch.object(
            sys.modules["processing.task_execution_engine"],
            "calculate_pagerank",
            mock_pagerank,
        ):
            # Mock _get_important_nodes
            proc._get_important_nodes = AsyncMock(return_value=[{"id": "n1"}])
            result = await proc.build_analysis_result(
                mock_conn, 10, scan_depth=2, start_time=start, code_graph_id="g1"
            )
            assert result["success"] is True
            assert len(result["important_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_get_important_nodes(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        # psycopg returns JSONB properties as a dict already (no json.loads)
        row = ("node1", "function", {"name": "foo"})
        mock_cursor.fetchone = AsyncMock(return_value=row)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        pagerank = {"node1": 0.9, "node2": 0.1}
        result = await proc._get_important_nodes(mock_conn, "g1", pagerank)
        assert len(result) >= 1
        assert result[0]["id"] == "node1"
        assert result[0]["pagerank"] == 0.9
        assert result[0]["name"] == "foo"

    @pytest.mark.asyncio
    async def test_get_important_nodes_null_properties(self):
        proc = TaskResultProcessor()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # NULL JSONB column -> source coerces to {} and the node is still returned
        mock_cursor.fetchone = AsyncMock(return_value=("n1", "func", None))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await proc._get_important_nodes(mock_conn, "g1", {"n1": 0.5})
        assert len(result) == 1
        assert result[0]["id"] == "n1"
        assert result[0]["name"] == ""

    def test_build_multi_language_result(self):
        proc = TaskResultProcessor()
        import time

        factory = MagicMock()
        factory.get_supported_languages.return_value = ["python", "javascript"]

        analysis_data = {
            "supported_files": ["a.py", "b.js"],
            "language_stats": {"python": 1, "javascript": 1},
            "language_analysis_results": {"python": {}, "javascript": {}},
            "multi_language_factory": factory,
        }

        start = time.time()
        result = proc.build_multi_language_result(analysis_data, start)
        assert result["success"] is True
        assert result["total_files_analyzed"] == 2
        assert "python" in result["languages_detected"]
        assert result["supported_languages"] == ["python", "javascript"]


# ===========================================================================
# AnalysisMetadataManager
# ===========================================================================


class TestAnalysisMetadataManager:
    @pytest.mark.asyncio
    async def test_store_language_analysis(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()

        await mgr.store_language_analysis(mock_conn, "g1", "python", {"files": 5})
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO nodes" in call_args[0]
        # postgres upsert: params are (code_graph_id, node_id, type, Jsonb(props))
        assert call_args[1][0] == "g1"
        assert call_args[1][1] == "lang_analysis_python_g1"
        assert call_args[1][2] == "language_analysis"

    @pytest.mark.asyncio
    async def test_store_analysis_metadata(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        factory = MagicMock()
        factory.get_supported_extensions.return_value = [".py", ".js"]
        factory.get_supported_languages.return_value = ["python", "javascript"]

        ctx = {
            "conn": mock_conn,
            "code_graph_id": "g1",
            "supported_files": ["a.py", "b.js"],
            "language_stats": {"python": 1, "javascript": 1},
            "multi_language_factory": factory,
        }

        await mgr.store_analysis_metadata(ctx)
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analysis_metadata_found(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # psycopg returns JSONB as a dict already (no json.loads in source)
        metadata = {"code_graph_id": "g1", "total_files": 5}
        mock_cursor.fetchone = AsyncMock(return_value=(metadata,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await mgr.get_analysis_metadata(mock_conn, "g1")
        assert result == metadata

    @pytest.mark.asyncio
    async def test_get_analysis_metadata_not_found(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await mgr.get_analysis_metadata(mock_conn, "nonexistent")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_analysis_metadata_null_properties(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # NULL JSONB column -> source returns {} (row[0] falsy)
        mock_cursor.fetchone = AsyncMock(return_value=(None,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await mgr.get_analysis_metadata(mock_conn, "g1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_language_analysis_found(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # psycopg returns JSONB as a dict already (no json.loads in source)
        data = {"language": "python", "analysis_results": {}}
        mock_cursor.fetchone = AsyncMock(return_value=(data,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await mgr.get_language_analysis(mock_conn, "g1", "python")
        assert result == data

    @pytest.mark.asyncio
    async def test_get_language_analysis_not_found(self):
        mgr = AnalysisMetadataManager()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await mgr.get_language_analysis(mock_conn, "g1", "rust")
        assert result == {}


# ===========================================================================
# FileProcessingResultHandler
# ===========================================================================


class TestFileProcessingResultHandler:
    @pytest.mark.asyncio
    async def test_store_file_node(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        with (
            patch("os.path.basename", return_value="test.py"),
            patch("os.path.exists", return_value=False),
            patch("os.path.getsize", return_value=0),
        ):
            await handler._store_file_node(mock_conn, "g1", "/path/test.py", "python")
            mock_conn.execute.assert_called_once()
            args = mock_conn.execute.call_args[0]
            assert "INSERT INTO nodes" in args[0]
            # postgres upsert: params (code_graph_id, file_path, type, Jsonb(props))
            assert args[1][0] == "g1"
            assert args[1][1] == "/path/test.py"
            assert args[1][2] == "file"

    @pytest.mark.asyncio
    async def test_store_functions(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        storage_ctx = {
            "conn": mock_conn,
            "code_graph_id": "g1",
            "file_path": "/path/test.py",
            "language": "python",
        }

        functions = [
            {"name": "foo", "line_number": 1, "signature": "def foo()", "complexity": 3},
            {"name": "bar", "line_number": 10},
        ]

        await handler._store_functions(storage_ctx, functions)
        # 2 functions * 2 calls each (node + edge) = 4 calls
        assert mock_conn.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_store_classes_and_methods(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        storage_ctx = {
            "conn": mock_conn,
            "code_graph_id": "g1",
            "file_path": "/path/test.py",
            "language": "python",
        }

        classes = [
            {
                "name": "MyClass",
                "line_number": 5,
                "methods": [
                    {"name": "method1", "line_number": 6},
                ],
            }
        ]

        await handler._store_classes_and_methods(storage_ctx, classes)
        # class node + class edge + method node + method edge = 4
        assert mock_conn.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_store_dependencies(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        storage_ctx = {
            "conn": mock_conn,
            "code_graph_id": "g1",
            "file_path": "/path/test.py",
            "language": "python",
        }

        deps = [{"target": "os"}, {"target": "sys", "type": "import"}]
        await handler._store_dependencies(storage_ctx, deps)
        # 2 deps * 2 calls each (node + edge) = 4 calls
        assert mock_conn.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_store_processing_results_success(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        processing_data = {
            "file_path": "/path/test.py",
            "language": "python",
            "result": {
                "functions": [{"name": "foo", "line_number": 1}],
                "classes": [],
                "dependencies": [{"target": "os"}],
            },
        }

        with (
            patch("os.path.basename", return_value="test.py"),
            patch("os.path.exists", return_value=False),
            patch("os.path.getsize", return_value=0),
        ):
            await handler.store_processing_results(mock_conn, "g1", processing_data)
            mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_processing_results_error(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))

        processing_data = {
            "file_path": "/path/test.py",
            "language": "python",
            "result": {"functions": [], "classes": [], "dependencies": []},
        }

        with pytest.raises(Exception, match="db error"):
            await handler.store_processing_results(mock_conn, "g1", processing_data)
        mock_conn.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_class_methods(self):
        handler = FileProcessingResultHandler()
        mock_conn = AsyncMock()

        storage_ctx = {
            "conn": mock_conn,
            "code_graph_id": "g1",
            "file_path": "/path/test.py",
            "language": "python",
        }

        cls = {
            "name": "Foo",
            "methods": [
                {"name": "m1", "line_number": 1, "visibility": "public"},
                {"name": "m2", "line_number": 5},
            ],
        }

        await handler._store_class_methods(storage_ctx, "/path/test.py::Foo", cls)
        # 2 methods * 2 calls each = 4
        assert mock_conn.execute.call_count == 4


# ===========================================================================
# FileSystemValidator extended (lines 266, 277-278, 296-297)
# ===========================================================================


class TestFileSystemValidatorExtended:
    def test_validate_codebase_path_nonexistent(self):
        result = FileSystemValidator.validate_codebase_path("/nonexistent/path/abc")
        assert result["is_valid"] is False
        assert "does not exist" in result["error"]

    def test_validate_codebase_path_is_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = FileSystemValidator.validate_codebase_path(str(f))
        assert result["is_valid"] is False
        assert "not a directory" in result["error"]

    def test_validate_codebase_path_valid_directory(self, tmp_path):
        (tmp_path / "file.py").write_text("x = 1")
        result = FileSystemValidator.validate_codebase_path(str(tmp_path))
        assert result["is_valid"] is True
        assert "absolute_path" in result
        assert result["size_info"]["total_files"] >= 1

    def test_validate_codebase_path_exception(self):
        """Exception during validation returns error dict (lines 277-278)."""
        with patch("pathlib.Path", side_effect=Exception("path error")):
            result = FileSystemValidator.validate_codebase_path("/whatever/nonexistent")
            assert result["is_valid"] is False
            assert "Error validating" in result["error"]

    def test_get_directory_info_exception(self, tmp_path):
        """Exception in _get_directory_info returns error dict (lines 296-297)."""
        with patch.object(type(tmp_path), "rglob", side_effect=PermissionError("denied")):
            result = FileSystemValidator._get_directory_info(tmp_path)
            assert "error" in result


# DatabaseValidator tests removed: the class (validate_database_connection /
# _get_database_stats) was an aiosqlite file-DB validator dropped in the
# postgres cutover. Schema is now Alembic-owned and rows are scoped by
# code_graph_id, so there is no file-level DB to validate.
