"""Round 4 tests for processing/test_execution and other modules.

Covers: security_manager, test_runner,
task_validator, task_queue, analyzers/test_execution, multi_language_analysis_processor,
codebase_analysis_processor, python_processor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from models.testing_models import SecurityLevel

# ---------------------------------------------------------------------------
# ExecutionSecurityManager tests
# ---------------------------------------------------------------------------


class TestExecutionSecurityManager:
    """Tests for ExecutionSecurityManager."""

    def _make_config(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        return TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=50000,
            allowed_modules=["pytest", "math", "os"],
        )

    def _make_manager(self):
        from processing.test_execution.security_manager import ExecutionSecurityManager

        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            return ExecutionSecurityManager(self._make_config())

    def test_init(self):
        mgr = self._make_manager()
        assert mgr.dangerous_patterns is not None
        assert "python" in mgr.dangerous_patterns
        assert "javascript" in mgr.dangerous_patterns

    def test_get_container_security_config_high(self):
        mgr = self._make_manager()
        config = mgr.get_container_security_config(SecurityLevel.HIGH)
        assert config["mem_limit"] == "256m"
        assert config["cpu_quota"] == 25000

    def test_get_container_security_config_medium(self):
        mgr = self._make_manager()
        config = mgr.get_container_security_config(SecurityLevel.MEDIUM)
        assert config["mem_limit"] == "512m"

    def test_get_container_security_config_low(self):
        mgr = self._make_manager()
        config = mgr.get_container_security_config(SecurityLevel.LOW)
        assert config["mem_limit"] == "1g"

    def test_validate_container_runtime_security_valid(self):
        mgr = self._make_manager()
        config = {
            "read_only": True,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "512m",
        }
        assert mgr.validate_container_runtime_security(config) is True

    def test_validate_container_runtime_security_missing_feature(self):
        mgr = self._make_manager()
        config = {"read_only": True}  # Missing other features
        assert mgr.validate_container_runtime_security(config) is False

    def test_validate_container_runtime_security_root_user(self):
        mgr = self._make_manager()
        config = {
            "read_only": True,
            "network_disabled": True,
            "user": "root",
            "security_opt": ["x"],
            "mem_limit": "512m",
        }
        assert mgr.validate_container_runtime_security(config) is False

    def test_validate_request_structure_empty(self):
        mgr = self._make_manager()
        request = MagicMock(code=None, test_code=None, language="python", timeout=None)
        with pytest.raises(ValueError, match="must contain"):
            mgr._validate_request_structure(request)

    def test_validate_request_structure_no_language(self):
        mgr = self._make_manager()
        request = MagicMock(code="x", test_code=None, language=None, timeout=None)
        with pytest.raises(ValueError, match="Language must be specified"):
            mgr._validate_request_structure(request)

    def test_validate_request_structure_unsupported_language(self):
        mgr = self._make_manager()
        request = MagicMock(code="x", test_code=None, language="rust", timeout=None)
        with pytest.raises(ValueError, match="Unsupported language"):
            mgr._validate_request_structure(request)

    def test_validate_request_structure_timeout_exceeded(self):
        mgr = self._make_manager()
        request = MagicMock(code="x", test_code=None, language="python", timeout=999)
        with pytest.raises(ValueError, match="Timeout exceeds"):
            mgr._validate_request_structure(request)

    @pytest.mark.asyncio
    async def test_validate_container_specific_security_escape(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="container escape"):
            await mgr._validate_container_specific_security("docker run exploit", "python")

    @pytest.mark.asyncio
    async def test_validate_python_security_dangerous_import(self):
        mgr_config = self._make_config()
        mgr_config.allowed_modules = ["pytest"]  # os not allowed
        from processing.test_execution.security_manager import ExecutionSecurityManager

        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            mgr = ExecutionSecurityManager(mgr_config)
        with pytest.raises(ValueError, match="Dangerous import"):
            await mgr._validate_python_security("import subprocess")

    @pytest.mark.asyncio
    async def test_validate_python_security_dangerous_function(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Dangerous function"):
            await mgr._validate_python_security("eval('1+1')")

    @pytest.mark.asyncio
    async def test_validate_javascript_security(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError):
            await mgr._validate_javascript_security("require('child_process')")

    @pytest.mark.asyncio
    async def test_scan_for_security_violations(self):
        mgr = self._make_manager()
        violations = await mgr.scan_for_security_violations(
            "import os\nos.system('rm -rf /')", "python"
        )
        assert any(v["type"] == "dangerous_pattern" for v in violations)

    @pytest.mark.asyncio
    async def test_check_suspicious_imports(self):
        mgr = self._make_manager()
        violations = await mgr._check_suspicious_imports("__import__('os')", "python")
        assert len(violations) > 0

    @pytest.mark.asyncio
    async def test_check_filesystem_access(self):
        mgr = self._make_manager()
        violations = await mgr._check_filesystem_access("open('/etc/passwd')", "python")
        assert len(violations) > 0

    def test_sanitize_string_field(self):
        mgr = self._make_manager()
        result = mgr._sanitize_string_field("<script>alert('xss')</script>hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_sanitize_string_field_empty(self):
        mgr = self._make_manager()
        assert mgr._sanitize_string_field("") == ""

    def test_sanitize_string_field_truncate(self):
        mgr = self._make_manager()
        result = mgr._sanitize_string_field("x" * 2000, max_length=100)
        assert len(result) == 100

    @pytest.mark.asyncio
    async def test_sanitize_code_content_python(self):
        mgr = self._make_manager()
        code = "import subprocess\nresult = subprocess.run(['ls'])"
        sanitized = await mgr._sanitize_code_content(code, "python")
        assert "SANITIZED" in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_code_content_javascript(self):
        mgr = self._make_manager()
        code = "const cp = require('child_process')"
        sanitized = await mgr._sanitize_code_content(code, "javascript")
        assert "SANITIZED" in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_code_content_truncation(self):
        mgr = self._make_manager()
        mgr.config.max_code_length = 50
        code = "x" * 200
        sanitized = await mgr._sanitize_code_content(code, "python")
        assert "truncated" in sanitized.lower()

    @pytest.mark.asyncio
    async def test_create_sanitization_report_no_changes(self):
        mgr = self._make_manager()
        req = MagicMock(code="safe code", test_code=None, timeout=10)
        report = await mgr.create_sanitization_report(req, req)
        assert report["sanitization_applied"] is False

    @pytest.mark.asyncio
    async def test_create_sanitization_report_with_changes(self):
        mgr = self._make_manager()
        original = MagicMock(code="original", test_code=None, timeout=30)
        sanitized = MagicMock(code="sanitized", test_code=None, timeout=20)
        report = await mgr.create_sanitization_report(original, sanitized)
        assert report["changes_count"] > 0

    def test_factory_function(self):
        from processing.test_execution.security_manager import create_test_security_manager

        config = self._make_config()
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            mgr = create_test_security_manager(config)
        assert mgr is not None


# ---------------------------------------------------------------------------
# ContainerTestRunner tests
# ---------------------------------------------------------------------------


class TestContainerTestRunner:
    def test_should_use_orchestrator(self):
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = MagicMock()
        mock_config = MagicMock(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_imports=["pytest"],
        )
        runner = ContainerTestRunner(mock_docker, mock_config)
        go_request = MagicMock(language="go")
        python_request = MagicMock(language="python")
        assert runner._should_use_orchestrator(go_request) is True
        assert runner._should_use_orchestrator(python_request) is False

    def test_get_memory_limit(self):
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = MagicMock()
        mock_config = MagicMock(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_imports=["pytest"],
        )
        runner = ContainerTestRunner(mock_docker, mock_config)
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"

    def test_get_cpu_quota(self):
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = MagicMock()
        mock_config = MagicMock(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_imports=["pytest"],
        )
        runner = ContainerTestRunner(mock_docker, mock_config)
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000

    def test_get_file_extension(self):
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = MagicMock()
        mock_config = MagicMock(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_imports=["pytest"],
        )
        runner = ContainerTestRunner(mock_docker, mock_config)
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("unknown") == ".py"

    def test_get_test_file_extension(self):
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = MagicMock()
        mock_config = MagicMock(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_imports=["pytest"],
        )
        runner = ContainerTestRunner(mock_docker, mock_config)
        assert runner._get_test_file_extension("python") == ".py"
        assert runner._get_test_file_extension("javascript") == ".test.js"
        assert runner._get_test_file_extension("typescript") == ".test.ts"


# ---------------------------------------------------------------------------
# TaskParameterValidator tests
# ---------------------------------------------------------------------------


class TestTaskParameterValidator:
    def test_validate_codebase_analysis_valid(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_codebase_analysis_params(
            {
                "codebase_path": "/tmp/test",
                "scan_depth": 2,
                "code_graph_id": "my-graph",
            }
        )
        assert result["is_valid"] is True

    def test_validate_codebase_analysis_missing_path(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_codebase_analysis_params({})
        assert result["is_valid"] is False
        assert any("codebase_path" in e for e in result["errors"])

    def test_validate_codebase_analysis_bad_depth(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_codebase_analysis_params(
            {
                "codebase_path": "/tmp/test",
                "scan_depth": -1,
            }
        )
        assert result["is_valid"] is False

    def test_validate_codebase_analysis_bad_graph_id(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_codebase_analysis_params(
            {
                "codebase_path": "/tmp/test",
                "code_graph_id": "",
            }
        )
        assert result["is_valid"] is False

    def test_validate_centrality_calculation_valid(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_centrality_calculation_params(
            {
                "code_graph_id": "test",
                "node_type": "function",
            }
        )
        assert result["is_valid"] is True

    def test_validate_centrality_bad_node_type(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_centrality_calculation_params(
            {
                "code_graph_id": "test",
                "node_type": 123,
            }
        )
        assert result["is_valid"] is False

    def test_validate_multi_language_analysis_valid(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_multi_language_analysis_params(
            {
                "codebase_path": "/tmp/test",
                "code_graph_id": "test",
            }
        )
        assert result["is_valid"] is True

    def test_validate_multi_language_analysis_missing_path(self):
        from processing.task_validator import TaskParameterValidator

        result = TaskParameterValidator.validate_multi_language_analysis_params({})
        assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# TaskErrorHandler tests
# ---------------------------------------------------------------------------


class TestTaskErrorHandler:
    def test_categorize_file_system(self):
        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(FileNotFoundError("x")) == "FILE_SYSTEM"

    def test_categorize_database(self):
        # postgres cutover: DATABASE errors are now psycopg.Error, not sqlite3.Error
        import psycopg

        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(psycopg.Error("x")) == "DATABASE"

    def test_categorize_data_processing(self):
        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(ValueError("x")) == "DATA_PROCESSING"

    def test_categorize_import(self):
        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(ImportError("x")) == "IMPORT_MODULE"

    def test_categorize_runtime(self):
        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(RuntimeError("x")) == "RUNTIME"

    def test_categorize_unknown(self):
        from processing.task_validator import TaskErrorHandler

        assert TaskErrorHandler.categorize_error(StopIteration()) == "UNKNOWN"

    def test_create_error_response(self):
        from processing.task_validator import TaskErrorHandler

        result = TaskErrorHandler.create_error_response(ValueError("bad value"), "test_context")
        assert result["success"] is False
        assert result["error_category"] == "DATA_PROCESSING"
        assert result["context"] == "test_context"

    def test_log_error_file_system(self):
        from processing.task_validator import TaskErrorHandler

        # Should not raise
        TaskErrorHandler.log_error(FileNotFoundError("nope"), "testing", "analysis")

    def test_log_error_runtime(self):
        from processing.task_validator import TaskErrorHandler

        TaskErrorHandler.log_error(RuntimeError("crash"), "testing", "analysis")


# ---------------------------------------------------------------------------
# TaskMonitor tests
# ---------------------------------------------------------------------------


class TestTaskMonitor:
    def test_record_and_stats(self):
        from processing.task_validator import TaskMonitor

        monitor = TaskMonitor()
        monitor.record_task_start("analysis", {})
        monitor.record_task_success("analysis", 1.5)
        stats = monitor.get_statistics()
        assert stats["total_tasks"] == 1
        assert stats["success_rate"] == 100.0

    def test_record_failure(self):
        from processing.task_validator import TaskMonitor

        monitor = TaskMonitor()
        monitor.record_task_start("analysis", {})
        monitor.record_task_failure("analysis", ValueError("bad"), 0.5)
        stats = monitor.get_statistics()
        assert stats["failed_tasks"] == 1
        assert "DATA_PROCESSING" in stats["error_categories"]

    def test_reset(self):
        from processing.task_validator import TaskMonitor

        monitor = TaskMonitor()
        monitor.record_task_start("x", {})
        monitor.record_task_success("x", 1.0)
        monitor.reset_statistics()
        stats = monitor.get_statistics()
        assert stats["message"] == "No tasks executed yet"

    def test_task_stats_property(self):
        from processing.task_validator import TaskMonitor

        monitor = TaskMonitor()
        stats = monitor.task_stats
        assert stats["total_tasks"] == 0


# ---------------------------------------------------------------------------
# FileSystemValidator tests
# ---------------------------------------------------------------------------


class TestFileSystemValidator:
    def test_validate_existing_dir(self, tmp_path):
        from processing.task_validator import FileSystemValidator

        result = FileSystemValidator.validate_codebase_path(str(tmp_path))
        assert result["is_valid"] is True

    def test_validate_nonexistent_path(self):
        from processing.task_validator import FileSystemValidator

        result = FileSystemValidator.validate_codebase_path("/nonexistent/path/xyz123")
        assert result["is_valid"] is False

    def test_validate_file_not_dir(self, tmp_path):
        from processing.task_validator import FileSystemValidator

        f = tmp_path / "file.txt"
        f.write_text("x")
        result = FileSystemValidator.validate_codebase_path(str(f))
        assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# ProcessingTaskQueue tests (local fallback only, no Redis)
# ---------------------------------------------------------------------------


class TestProcessingTaskQueueLocalFallback:
    def _make_queue(self):
        from processing.task_queue import ProcessingTaskQueue, TaskQueueConfig

        # redis_service = None triggers local fallback path
        config = TaskQueueConfig(max_retries=1, retry_delay=0.01)
        return ProcessingTaskQueue(redis_service=None, config=config)

    @pytest.mark.asyncio
    async def test_add_task_local_fallback(self):
        q = self._make_queue()
        result = await q.add_task({"id": "task1", "type": "analysis"})
        assert result is True

    @pytest.mark.asyncio
    async def test_get_task_from_local_queue(self):
        q = self._make_queue()
        await q.local_queue.put({"id": "task1"})
        task = await q._get_task_from_local_queue(timeout=1)
        assert task["id"] == "task1"

    @pytest.mark.asyncio
    async def test_get_task_from_local_queue_timeout(self):
        q = self._make_queue()
        task = await q._get_task_from_local_queue(timeout=0)
        assert task is None

    @pytest.mark.asyncio
    async def test_update_task_status_local(self):
        q = self._make_queue()
        result = await q.update_task_status("task1", "running")
        assert result is True
        assert q.local_results["task1:status"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_complete_task_local(self):
        q = self._make_queue()
        result = await q.complete_task("task1", {"output": "done"})
        assert result is True

    @pytest.mark.asyncio
    async def test_fail_task_local(self):
        q = self._make_queue()
        result = await q.fail_task("task1", "something broke")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_task_status_local(self):
        q = self._make_queue()
        await q.update_task_status("task1", "running")
        status = await q.get_task_status("task1")
        assert status["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_task_result_local(self):
        q = self._make_queue()
        await q.complete_task("task1", {"output": "done"})
        result = await q.get_task_result("task1")
        assert result["output"] == "done"

    @pytest.mark.asyncio
    async def test_get_all_tasks_no_redis(self):
        q = self._make_queue()
        tasks = await q.get_all_tasks()
        assert tasks == []
