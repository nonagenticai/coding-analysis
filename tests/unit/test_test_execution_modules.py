"""Unit tests for test_execution modules:

- security_manager.ExecutionSecurityManager
- test_runner.ContainerTestRunner
- timeout_manager.TimeoutManager / TimeoutConfig / factory functions
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Models are available after conftest stubs run
# ---------------------------------------------------------------------------
from models.testing_models import (
    SecurityLevel,
    TestRequest,
    TestResult,
)

# ---------------------------------------------------------------------------
# timeout_manager has no heavy external deps — import directly
# ---------------------------------------------------------------------------
from processing.test_execution.timeout_manager import (
    ExecutionParams,
    TimeoutConfig,
    TimeoutManager,
    TimeoutType,
    create_container_timeout_manager,
    create_production_timeout_manager,
    create_test_timeout_manager,
)

# ===================================================================
# TIMEOUT MANAGER TESTS
# ===================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig validation."""

    def test_defaults(self):
        cfg = TimeoutConfig()
        assert cfg.default_timeout_seconds == 30
        assert cfg.max_timeout_seconds == 300
        assert cfg.graceful_shutdown_timeout == 10
        assert cfg.container_start_timeout == 60
        assert cfg.container_stop_timeout == 30
        assert cfg.cleanup_on_timeout is True
        assert cfg.retry_cleanup_attempts == 3

    def test_custom_values(self):
        cfg = TimeoutConfig(
            default_timeout_seconds=60,
            max_timeout_seconds=600,
            graceful_shutdown_timeout=20,
        )
        assert cfg.default_timeout_seconds == 60
        assert cfg.max_timeout_seconds == 600
        assert cfg.graceful_shutdown_timeout == 20

    def test_max_less_than_default_raises(self):
        with pytest.raises(ValueError, match="max_timeout_seconds"):
            TimeoutConfig(default_timeout_seconds=100, max_timeout_seconds=50)

    def test_out_of_range_default(self):
        with pytest.raises(ValueError):
            TimeoutConfig(default_timeout_seconds=0)

    def test_out_of_range_max(self):
        with pytest.raises(ValueError):
            TimeoutConfig(max_timeout_seconds=5000)


class TestTimeoutManager:
    """Tests for TimeoutManager."""

    def test_init_default_config(self):
        mgr = TimeoutManager()
        assert mgr.config.default_timeout_seconds == 30

    def test_init_custom_config(self):
        cfg = TimeoutConfig(default_timeout_seconds=45, max_timeout_seconds=300)
        mgr = TimeoutManager(cfg)
        assert mgr.config.default_timeout_seconds == 45

    def test_resolve_timeout_explicit(self):
        mgr = TimeoutManager()
        result = mgr._resolve_timeout(10, TimeoutType.TEST_EXECUTION)
        assert result == 10

    def test_resolve_timeout_exceeds_max(self):
        cfg = TimeoutConfig(default_timeout_seconds=30, max_timeout_seconds=60)
        mgr = TimeoutManager(cfg)
        result = mgr._resolve_timeout(120, TimeoutType.TEST_EXECUTION)
        assert result == 60  # capped to max

    def test_resolve_timeout_none_uses_default(self):
        mgr = TimeoutManager()
        result = mgr._resolve_timeout(None, TimeoutType.TEST_EXECUTION)
        assert result == mgr.config.default_timeout_seconds

    def test_resolve_timeout_container_setup(self):
        mgr = TimeoutManager()
        result = mgr._resolve_timeout(None, TimeoutType.CONTAINER_SETUP)
        assert result == mgr.config.container_start_timeout

    def test_resolve_timeout_container_cleanup(self):
        mgr = TimeoutManager()
        result = mgr._resolve_timeout(None, TimeoutType.CONTAINER_CLEANUP)
        assert result == mgr.config.container_stop_timeout

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        mgr = TimeoutManager()

        async def quick_task():
            return "done"

        result = await mgr.execute_with_timeout(quick_task())
        assert result.success is True
        assert result.timed_out is False
        assert result.exit_code == 0
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_with_timeout_timeout(self):
        mgr = TimeoutManager()

        async def slow_task():
            await asyncio.sleep(100)

        params = ExecutionParams(timeout_seconds=1, task_id="slow-1")
        result = await mgr.execute_with_timeout(slow_task(), params)
        assert result.success is False
        assert result.timed_out is True
        assert result.exit_code == 124

    @pytest.mark.asyncio
    async def test_execute_with_timeout_error(self):
        mgr = TimeoutManager()

        async def failing_task():
            raise ValueError("boom")

        result = await mgr.execute_with_timeout(failing_task())
        assert result.success is False
        assert result.timed_out is False
        assert result.error_message == "boom"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_calls_cleanup(self):
        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        mgr = TimeoutManager()

        async def slow():
            await asyncio.sleep(100)

        params = ExecutionParams(timeout_seconds=1, task_id="cleanup-test", cleanup_handler=cleanup)
        result = await mgr.execute_with_timeout(slow(), params)
        assert result.timed_out is True
        assert result.cleanup_performed is True
        assert cleanup_called is True

    @pytest.mark.asyncio
    async def test_get_active_tasks_empty(self):
        mgr = TimeoutManager()
        active = await mgr.get_active_tasks()
        assert active == {}

    @pytest.mark.asyncio
    async def test_cleanup_all_tasks_empty(self):
        mgr = TimeoutManager()
        count = await mgr.cleanup_all_tasks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self):
        mgr = TimeoutManager()
        result = await mgr.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_context_success(self):
        mgr = TimeoutManager()
        async with mgr.timeout_context(timeout_seconds=5):
            pass  # completes within timeout

    @pytest.mark.asyncio
    async def test_timeout_context_timeout(self):
        mgr = TimeoutManager()
        with pytest.raises(TimeoutError):
            async with mgr.timeout_context(timeout_seconds=1):
                await asyncio.sleep(100)


class TestTimeoutFactoryFunctions:
    """Tests for factory functions."""

    def test_create_test_timeout_manager(self):
        mgr = create_test_timeout_manager()
        assert isinstance(mgr, TimeoutManager)
        assert mgr.config.default_timeout_seconds == 30
        assert mgr.config.max_timeout_seconds == 300

    def test_create_container_timeout_manager(self):
        mgr = create_container_timeout_manager()
        assert isinstance(mgr, TimeoutManager)
        assert mgr.config.default_timeout_seconds == 60
        assert mgr.config.retry_cleanup_attempts == 5

    def test_create_production_timeout_manager(self):
        mgr = create_production_timeout_manager()
        assert isinstance(mgr, TimeoutManager)
        assert mgr.config.default_timeout_seconds == 60
        assert mgr.config.max_timeout_seconds == 900


# ===================================================================
# SECURITY MANAGER TESTS
# ===================================================================


class TestExecutionSecurityManager:
    """Tests for ExecutionSecurityManager."""

    @pytest.fixture
    def security_config(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        return TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=10000,
            allowed_modules=["pytest", "math"],
        )

    @pytest.fixture
    def security_manager(self, security_config):
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import (
                ExecutionSecurityManager,
            )

            return ExecutionSecurityManager(security_config)

    def test_get_container_security_config_high(self, security_manager):
        config = security_manager.get_container_security_config(SecurityLevel.HIGH)
        assert config["mem_limit"] == "256m"
        assert config["read_only"] is True
        assert config["network_disabled"] is True
        assert config["cpu_quota"] == 25000

    def test_get_container_security_config_medium(self, security_manager):
        config = security_manager.get_container_security_config(SecurityLevel.MEDIUM)
        assert config["mem_limit"] == "512m"
        assert config["cpu_quota"] == 50000

    def test_get_container_security_config_low(self, security_manager):
        config = security_manager.get_container_security_config(SecurityLevel.LOW)
        assert config["mem_limit"] == "1g"
        assert config["cpu_quota"] == 100000
        assert config["pids_limit"] == 200

    def test_validate_container_runtime_security_valid(self, security_manager):
        container_config = {
            "read_only": True,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "512m",
        }
        assert security_manager.validate_container_runtime_security(container_config) is True

    def test_validate_container_runtime_security_missing_feature(self, security_manager):
        container_config = {
            "read_only": True,
            "network_disabled": True,
            # missing user, security_opt, mem_limit
        }
        assert security_manager.validate_container_runtime_security(container_config) is False

    def test_validate_container_runtime_security_root_user(self, security_manager):
        container_config = {
            "read_only": True,
            "network_disabled": True,
            "user": "root",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "512m",
        }
        assert security_manager.validate_container_runtime_security(container_config) is False

    def test_validate_request_structure_no_code(self, security_manager):
        request = MagicMock()
        request.code = None
        request.test_code = None
        request.language = "python"
        request.timeout = None
        with pytest.raises(ValueError, match="must contain"):
            security_manager._validate_request_structure(request)

    def test_validate_request_structure_no_language(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.language = ""
        request.timeout = None
        with pytest.raises(ValueError, match="Language must be specified"):
            security_manager._validate_request_structure(request)

    def test_validate_request_structure_unsupported_language(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.language = "go"
        request.timeout = None
        with pytest.raises(ValueError, match="Unsupported language"):
            security_manager._validate_request_structure(request)

    def test_validate_request_structure_timeout_exceeded(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.language = "python"
        request.timeout = 999
        with pytest.raises(ValueError, match="Timeout exceeds maximum"):
            security_manager._validate_request_structure(request)

    def test_validate_request_structure_valid(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.language = "python"
        request.timeout = 10
        # Should not raise
        security_manager._validate_request_structure(request)

    @pytest.mark.asyncio
    async def test_scan_for_security_violations_python(self, security_manager):
        code = "os.system('rm -rf /')\nsubprocess.run(['ls'])"
        violations = await security_manager.scan_for_security_violations(code, "python")
        assert len(violations) > 0
        pattern_types = [v["type"] for v in violations]
        assert "dangerous_pattern" in pattern_types

    @pytest.mark.asyncio
    async def test_scan_for_security_violations_clean(self, security_manager):
        code = "def add(a, b):\n    return a + b\n"
        violations = await security_manager.scan_for_security_violations(code, "python")
        # Clean code should have no dangerous_pattern violations
        dangerous = [v for v in violations if v["type"] == "dangerous_pattern"]
        assert len(dangerous) == 0

    @pytest.mark.asyncio
    async def test_scan_for_security_violations_javascript(self, security_manager):
        code = "process.env.SECRET\neval('1+1')"
        violations = await security_manager.scan_for_security_violations(code, "javascript")
        assert len(violations) > 0

    @pytest.mark.asyncio
    async def test_validate_container_specific_security_escape_attempt(self, security_manager):
        code = "docker run --rm -it ubuntu"
        with pytest.raises(ValueError, match="container escape"):
            await security_manager._validate_container_specific_security(code, "python")

    @pytest.mark.asyncio
    async def test_validate_python_security_dangerous_import(self, security_manager):
        code = "import subprocess"
        with pytest.raises(ValueError, match="Dangerous import"):
            await security_manager._validate_python_security(code)

    @pytest.mark.asyncio
    async def test_validate_python_security_dangerous_function(self, security_manager):
        code = "eval('1+1')"
        with pytest.raises(ValueError, match="Dangerous function"):
            await security_manager._validate_python_security(code)

    @pytest.mark.asyncio
    async def test_validate_javascript_security(self, security_manager):
        code = "require('child_process')"
        with pytest.raises(ValueError, match="Dangerous JavaScript pattern"):
            await security_manager._validate_javascript_security(code)

    def test_sanitize_string_field_empty(self, security_manager):
        assert security_manager._sanitize_string_field("") == ""

    def test_sanitize_string_field_script_tags(self, security_manager):
        val = "hello <script>alert(1)</script> world"
        result = security_manager._sanitize_string_field(val)
        assert "<script>" not in result
        assert "hello" in result
        assert "world" in result

    def test_sanitize_string_field_truncation(self, security_manager):
        long_val = "a" * 2000
        result = security_manager._sanitize_string_field(long_val, max_length=100)
        assert len(result) == 100

    def test_sanitize_string_field_javascript_protocol(self, security_manager):
        val = "javascript: alert(1)"
        result = security_manager._sanitize_string_field(val)
        assert "javascript:" not in result.lower()

    def test_dangerous_patterns_initialized(self, security_manager):
        patterns = security_manager.dangerous_patterns
        assert "python" in patterns
        assert "javascript" in patterns
        assert "typescript" in patterns
        assert "shell_injection" in patterns["python"]

    @pytest.mark.asyncio
    async def test_check_suspicious_imports_dynamic(self, security_manager):
        code = "__import__('os')"
        violations = await security_manager._check_suspicious_imports(code, "python")
        assert len(violations) > 0
        assert violations[0]["type"] == "suspicious_import"

    @pytest.mark.asyncio
    async def test_check_filesystem_access_python(self, security_manager):
        code = "open('/etc/passwd')"
        violations = await security_manager._check_filesystem_access(code, "python")
        assert len(violations) > 0
        assert violations[0]["type"] == "filesystem_access"

    @pytest.mark.asyncio
    async def test_sanitize_code_content_null_bytes(self, security_manager):
        code = "hello\x00world"
        result = await security_manager._sanitize_code_content(code, "python")
        assert "\x00" not in result

    @pytest.mark.asyncio
    async def test_sanitize_code_content_normalizes_newlines(self, security_manager):
        code = "line1\r\nline2\rline3"
        result = await security_manager._sanitize_code_content(code, "python")
        assert "\r" not in result
        assert "line1\nline2\nline3" == result

    @pytest.mark.asyncio
    async def test_create_sanitization_report_no_changes(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.timeout = 10
        report = await security_manager.create_sanitization_report(request, request)
        assert report["sanitization_applied"] is False
        assert report["changes_count"] == 0


# ===================================================================
# TEST RUNNER TESTS
# ===================================================================


class TestContainerTestRunner:
    """Tests for ContainerTestRunner from test_runner.py."""

    @pytest.fixture
    def docker_service(self):
        svc = MagicMock()
        svc.get_language_container_image = MagicMock(return_value="python:3.12")
        svc.create_test_container = AsyncMock(return_value="container-123")
        svc.start_container = AsyncMock()
        svc.create_file_in_container = AsyncMock(return_value=True)
        svc.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "1 passed", "stderr": ""}
        )
        svc.install_tools_in_container = AsyncMock()
        svc.stop_container = AsyncMock()
        svc.remove_container_by_id = AsyncMock()
        return svc

    @pytest.fixture
    def config(self):
        from python_test_suite.src.config import TestConfig

        return TestConfig()

    @pytest.fixture
    def runner(self, docker_service, config):
        with patch("processing.test_execution.test_runner.create_test_orchestrator") as mock_orch:
            mock_orch.return_value = MagicMock()
            from processing.test_execution.test_runner import ContainerTestRunner

            return ContainerTestRunner(docker_service, config)

    def test_should_use_orchestrator_go(self, runner):
        request = MagicMock()
        request.language = "go"
        assert runner._should_use_orchestrator(request) is True

    def test_should_use_orchestrator_rust(self, runner):
        request = MagicMock()
        request.language = "rust"
        assert runner._should_use_orchestrator(request) is True

    def test_should_use_orchestrator_python(self, runner):
        request = MagicMock()
        request.language = "python"
        assert runner._should_use_orchestrator(request) is False

    def test_get_file_extension_python(self, runner):
        assert runner._get_file_extension("python") == ".py"

    def test_get_file_extension_javascript(self, runner):
        assert runner._get_file_extension("javascript") == ".js"

    def test_get_file_extension_typescript(self, runner):
        assert runner._get_file_extension("typescript") == ".ts"

    def test_get_file_extension_unknown(self, runner):
        assert runner._get_file_extension("cobol") == ".py"

    def test_get_test_file_extension_python(self, runner):
        assert runner._get_test_file_extension("python") == ".py"

    def test_get_test_file_extension_javascript(self, runner):
        assert runner._get_test_file_extension("javascript") == ".test.js"

    def test_get_test_file_extension_typescript(self, runner):
        assert runner._get_test_file_extension("typescript") == ".test.ts"

    def test_get_memory_limit_high(self, runner):
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"

    def test_get_memory_limit_medium(self, runner):
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"

    def test_get_memory_limit_low(self, runner):
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"

    def test_get_cpu_quota_high(self, runner):
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000

    def test_get_cpu_quota_medium(self, runner):
        assert runner._get_cpu_quota(SecurityLevel.MEDIUM) == 50000

    @pytest.mark.asyncio
    async def test_validate_request_no_code(self, runner):
        request = MagicMock()
        request.code = None
        request.test_code = None
        request.code_path = None
        with pytest.raises(ValueError, match="must contain"):
            await runner._validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_request_code_too_long(self, runner):
        request = MagicMock()
        request.code = "x" * 60000
        request.test_code = ""
        request.code_path = None
        # Disable security_manager to test length check
        runner.security_manager = None
        with pytest.raises(ValueError, match="Code length exceeds"):
            await runner._validate_request(request)

    @pytest.mark.asyncio
    async def test_cleanup_container(self, runner, docker_service):
        await runner._cleanup_container("container-abc")
        docker_service.stop_container.assert_awaited_once_with("container-abc")
        docker_service.remove_container_by_id.assert_awaited_once_with("container-abc")

    @pytest.mark.asyncio
    async def test_cleanup_container_error(self, runner, docker_service):
        docker_service.stop_container.side_effect = RuntimeError("fail")
        # Should not raise
        await runner._cleanup_container("container-abc")

    @pytest.mark.asyncio
    async def test_execute_test_error_returns_failed_result(self, runner, docker_service):
        """Test that execute_test catches errors and returns a failed TestResult."""
        request = TestRequest(
            code="def add(a,b): return a+b",
            test_code="def test_add(): assert add(1,2)==3",
            language="python",
        )
        # Force orchestrator off and make validation raise
        runner.orchestrator = None
        runner.security_manager = None

        # Make _create_test_container raise
        docker_service.create_test_container.side_effect = RuntimeError("docker down")
        result = await runner.execute_test(request)
        assert isinstance(result, TestResult)
        assert result.success is False
        assert "docker down" in result.stderr


class TestSecurityManagerValidateAndSanitize:
    """Cover lines 51-83: validate_and_sanitize full flow."""

    @pytest.fixture
    def security_config(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        return TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=10000,
            allowed_modules=["pytest", "math"],
        )

    @pytest.fixture
    def security_manager(self, security_config):
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            mgr = ExecutionSecurityManager(security_config)
            # Mock testpy_security.validate_code to return object with .passed
            mock_result = MagicMock()
            mock_result.passed = True
            mock_result.error = None
            mgr.testpy_security.validate_code = MagicMock(return_value=mock_result)
            return mgr

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_valid_request(self, security_manager):
        """Full validation and sanitization of a clean request."""
        request = TestRequest(
            code="def add(a, b): return a + b",
            test_code="assert add(1, 2) == 3",
            language="python",
        )
        result = await security_manager.validate_and_sanitize(request)
        assert result.code is not None
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_failed_validation(self, security_manager):
        """Validation fails when testpy_security returns passed=False."""
        mock_result = MagicMock()
        mock_result.passed = False
        mock_result.error = "dangerous code"
        security_manager.testpy_security.validate_code = MagicMock(return_value=mock_result)
        request = TestRequest(
            code="eval('1+1')",
            test_code="assert True",
            language="python",
        )
        with pytest.raises(ValueError, match="Security validation failed"):
            await security_manager.validate_and_sanitize(request)

    @pytest.mark.asyncio
    async def test_validate_and_sanitize_code_too_long(self, security_manager):
        """Validation fails on overly long code."""
        request = TestRequest(
            code="x = 1\n" * 5000,
            test_code="assert True",
            language="python",
        )
        with pytest.raises(ValueError, match="Security validation failed"):
            await security_manager.validate_and_sanitize(request)


class TestSecurityManagerNotRunningLocally:
    """Cover line 154 (not network_disabled warning) and 157 (not read_only warning)."""

    @pytest.fixture
    def security_manager(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=30, max_memory_mb=512, max_code_length=10000
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            return ExecutionSecurityManager(config)

    def test_validate_container_not_read_only(self, security_manager):
        """Cover line 154: not read_only triggers warning but still passes."""
        config = {
            "read_only": False,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "512m",
        }
        assert security_manager.validate_container_runtime_security(config) is True

    def test_validate_container_not_network_disabled(self, security_manager):
        """Cover line 157: not network_disabled triggers warning but still passes."""
        config = {
            "read_only": True,
            "network_disabled": False,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": "512m",
        }
        assert security_manager.validate_container_runtime_security(config) is True


class TestSecurityManagerResourceValidation:
    """Cover lines 300-308: _validate_resource_requirements."""

    @pytest.fixture
    def security_manager(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=100,
            allowed_modules=["pytest"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            return ExecutionSecurityManager(config)

    def test_resource_code_too_long(self, security_manager):
        request = MagicMock()
        request.code = "x" * 200
        request.test_code = ""
        request.timeout = None
        with pytest.raises(ValueError, match="Code length exceeds limit"):
            security_manager._validate_resource_requirements(request)

    def test_resource_timeout_exceeds(self, security_manager):
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = ""
        request.timeout = 999
        with pytest.raises(ValueError, match="Timeout exceeds limit"):
            security_manager._validate_resource_requirements(request)


class TestSecurityManagerSanitizeRequest:
    """Cover lines 314-330: _sanitize_request."""

    @pytest.fixture
    def security_manager(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=10000,
            allowed_modules=["pytest", "math"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            return ExecutionSecurityManager(config)

    @pytest.mark.asyncio
    async def test_sanitize_request_with_code(self, security_manager):
        request = TestRequest(
            code="x = 1\x00hidden",
            test_code="assert True",
            language="python",
        )
        result = await security_manager._sanitize_request(request)
        assert "\x00" not in (result.code or "")

    @pytest.mark.asyncio
    async def test_sanitize_request_metadata(self, security_manager):
        """Test sanitization reduces timeout to max limit."""
        request = TestRequest(
            code="x = 1",
            test_code=None,
            language="python",
            timeout=25,  # within TestRequest validator but tests _sanitize_request_metadata
        )
        # Directly test the metadata sanitization with an override
        request_copy = request.model_copy()
        # Force timeout above limit by bypassing pydantic validation
        object.__setattr__(request_copy, "timeout", 100)
        result = await security_manager._sanitize_request_metadata(request_copy)
        assert result.timeout == 30


class TestSecurityManagerSanitizeRequestMetadata:
    """Cover lines 582-602: _sanitize_request_metadata."""

    @pytest.fixture
    def security_manager(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=10000,
            allowed_modules=["pytest"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            return ExecutionSecurityManager(config)

    @pytest.mark.asyncio
    async def test_sanitize_metadata_timeout_reduced(self, security_manager):
        """Timeout exceeding max is reduced."""
        request = TestRequest(
            code="x = 1",
            language="python",
            timeout=10,
        )
        # Force timeout above the security config limit
        object.__setattr__(request, "timeout", 100)
        result = await security_manager._sanitize_request_metadata(request)
        assert result.timeout == 30

    @pytest.mark.asyncio
    async def test_sanitize_metadata_timeout_adjustment(self, security_manager):
        """Timeout exceeding max is reduced in metadata sanitization."""
        request = TestRequest(
            code="x = 1",
            language="python",
            timeout=10,
        )
        # Force timeout above limit by bypassing pydantic
        object.__setattr__(request, "timeout", 50)
        result = await security_manager._sanitize_request_metadata(request)
        assert result.timeout == 30


class TestSecurityManagerCreateSanitizationReport:
    """Cover lines for create_sanitization_report with changes."""

    @pytest.fixture
    def security_manager(self):
        from python_test_suite.src.security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_code_length=10000,
            allowed_modules=["pytest"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as mock_lm:
            mock_lm.get_instance.return_value = MagicMock()
            from processing.test_execution.security_manager import ExecutionSecurityManager

            return ExecutionSecurityManager(config)

    @pytest.mark.asyncio
    async def test_report_with_code_changes(self, security_manager):
        original = MagicMock()
        original.code = "import os\nos.system('rm -rf /')"
        original.test_code = "assert True"
        original.timeout = 10

        sanitized = MagicMock()
        sanitized.code = "# SANITIZED: import os\n# SANITIZED: os.system('rm -rf /')"
        sanitized.test_code = "assert True"
        sanitized.timeout = 10

        report = await security_manager.create_sanitization_report(original, sanitized)
        assert report["sanitization_applied"] is True
        assert report["changes_count"] >= 1

    @pytest.mark.asyncio
    async def test_report_with_timeout_change(self, security_manager):
        original = MagicMock()
        original.code = "x = 1"
        original.test_code = None
        original.timeout = 999

        sanitized = MagicMock()
        sanitized.code = "x = 1"
        sanitized.test_code = None
        sanitized.timeout = 30

        report = await security_manager.create_sanitization_report(original, sanitized)
        assert report["sanitization_applied"] is True
        timeout_changes = [c for c in report["changes"] if c["field"] == "timeout"]
        assert len(timeout_changes) == 1
