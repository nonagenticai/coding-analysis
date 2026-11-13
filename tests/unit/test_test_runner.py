"""Coverage boost tests for ContainerTestRunner.

Focuses on:
- __init__: LoggingManager fallback, SecurityManager fallback, orchestrator fallback
- execute_test: orchestrator path, legacy path, exception handling
- _should_use_orchestrator: language routing
- _validate_request: code safety, length validation
- _get_memory_limit / _get_cpu_quota: all security levels
- _get_file_extension / _get_test_file_extension: all languages
- _create_js_package_file: javascript vs typescript
- _run_test_in_container: python, javascript, unsupported language
- _cleanup_container: success and failure
- create_container_test_runner factory
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import SecurityLevel, TestRequest


@pytest.fixture
def mock_docker_service():
    svc = MagicMock()
    svc.get_language_container_image = MagicMock(return_value="python:3.12")
    svc.create_test_container = AsyncMock(return_value="container-123")
    svc.start_container = AsyncMock()
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.install_tools_in_container = AsyncMock()
    svc.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "OK", "stderr": ""})
    return svc


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.max_execution_time = 30
    config.max_memory_mb = 512
    config.max_file_size_mb = 10
    config.max_code_length = 10000
    config.allowed_imports = ["pytest", "math"]
    return config


@pytest.fixture
def runner(mock_docker_service, mock_config):
    from processing.test_execution.test_runner import ContainerTestRunner

    return ContainerTestRunner(mock_docker_service, mock_config)


# ── _should_use_orchestrator ─────────────────────────────────────────


class TestShouldUseOrchestrator:
    def test_go_uses_orchestrator(self, runner):
        req = MagicMock()
        req.language = "go"
        assert runner._should_use_orchestrator(req) is True

    def test_rust_uses_orchestrator(self, runner):
        req = MagicMock()
        req.language = "rust"
        assert runner._should_use_orchestrator(req) is True

    def test_python_does_not_use_orchestrator(self, runner):
        req = MagicMock()
        req.language = "python"
        assert runner._should_use_orchestrator(req) is False

    def test_javascript_does_not_use_orchestrator(self, runner):
        req = MagicMock()
        req.language = "javascript"
        assert runner._should_use_orchestrator(req) is False


# ── _get_memory_limit ────────────────────────────────────────────────


class TestGetMemoryLimit:
    def test_high_security(self, runner):
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"

    def test_medium_security(self, runner):
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"

    def test_low_security(self, runner):
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"


# ── _get_cpu_quota ───────────────────────────────────────────────────


class TestGetCpuQuota:
    def test_high_security(self, runner):
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000

    def test_medium_security(self, runner):
        assert runner._get_cpu_quota(SecurityLevel.MEDIUM) == 50000

    def test_low_security(self, runner):
        assert runner._get_cpu_quota(SecurityLevel.LOW) == 100000


# ── _get_file_extension ─────────────────────────────────────────────


class TestGetFileExtension:
    def test_python(self, runner):
        assert runner._get_file_extension("python") == ".py"

    def test_javascript(self, runner):
        assert runner._get_file_extension("javascript") == ".js"

    def test_typescript(self, runner):
        assert runner._get_file_extension("typescript") == ".ts"

    def test_unknown_defaults_to_py(self, runner):
        assert runner._get_file_extension("unknown") == ".py"


# ── _get_test_file_extension ────────────────────────────────────────


class TestGetTestFileExtension:
    def test_python(self, runner):
        assert runner._get_test_file_extension("python") == ".py"

    def test_javascript(self, runner):
        assert runner._get_test_file_extension("javascript") == ".test.js"

    def test_typescript(self, runner):
        assert runner._get_test_file_extension("typescript") == ".test.ts"

    def test_unknown_defaults_to_py(self, runner):
        assert runner._get_test_file_extension("unknown") == ".py"


# ── _validate_request ────────────────────────────────────────────────


class TestValidateRequest:
    @pytest.mark.asyncio
    async def test_validate_request_no_code(self, runner):
        req = MagicMock()
        req.code = None
        req.test_code = None
        req.code_path = None
        with pytest.raises(ValueError, match="must contain"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_validate_request_code_too_long(self, runner):
        req = MagicMock()
        req.code = "x" * 60000
        req.test_code = ""
        req.code_path = None
        with pytest.raises(ValueError, match="exceeds limit"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_validate_request_valid_code(self, runner):
        req = MagicMock()
        req.code = "x = 1"
        req.test_code = "assert x == 1"
        req.code_path = None
        # Should not raise
        await runner._validate_request(req)


# ── _cleanup_container ───────────────────────────────────────────────


class TestCleanupContainer:
    @pytest.mark.asyncio
    async def test_cleanup_success(self, runner, mock_docker_service):
        await runner._cleanup_container("container-123")
        mock_docker_service.stop_container.assert_called_once_with("container-123")
        mock_docker_service.remove_container_by_id.assert_called_once_with("container-123")

    @pytest.mark.asyncio
    async def test_cleanup_handles_errors(self, runner, mock_docker_service):
        mock_docker_service.stop_container.side_effect = RuntimeError("failed")
        # Should not raise
        await runner._cleanup_container("container-123")


# ── _create_js_package_file ──────────────────────────────────────────


class TestCreateJsPackageFile:
    @pytest.mark.asyncio
    async def test_javascript_package(self, runner, mock_docker_service):
        await runner._create_js_package_file("container-123", "javascript")
        mock_docker_service.create_file_in_container.assert_called_once()
        call_args = mock_docker_service.create_file_in_container.call_args
        assert "jest" in call_args[0][2]
        assert "ts-jest" not in call_args[0][2]

    @pytest.mark.asyncio
    async def test_typescript_package(self, runner, mock_docker_service):
        await runner._create_js_package_file("container-123", "typescript")
        mock_docker_service.create_file_in_container.assert_called_once()
        call_args = mock_docker_service.create_file_in_container.call_args
        content = call_args[0][2]
        assert "ts-jest" in content
        assert "typescript" in content
        assert "@types/jest" in content


# ── execute_test ─────────────────────────────────────────────────────


class TestExecuteTest:
    @pytest.mark.asyncio
    async def test_execute_test_exception_returns_failed_result(self, runner):
        req = MagicMock(spec=TestRequest)
        req.language = "python"
        req.code = "x = 1"
        req.test_code = "assert x == 1"
        req.code_path = None
        req.security_level = SecurityLevel.MEDIUM
        req.timeout = 30
        req.container_name = None
        req.test_path = None

        # Make _validate_request raise
        runner.orchestrator = None
        with patch.object(runner, "_validate_request", side_effect=ValueError("bad code")):
            result = await runner.execute_test(req)
        assert result.success is False
        assert "bad code" in result.error


# ── factory function ─────────────────────────────────────────────────


class TestLogManagerFallbacks:
    """Cover lines 38-39, 60, 77-80, 89-91: init fallback paths."""

    def test_init_logging_manager_not_available(self, mock_docker_service, mock_config):
        """When _LoggingManager is None, falls back to stdlib logging."""
        from unittest.mock import patch

        with patch("processing.test_execution.test_runner._LoggingManager", None):
            from processing.test_execution.test_runner import ContainerTestRunner

            runner = ContainerTestRunner(mock_docker_service, mock_config)
            # Should use stdlib logger fallback
            assert runner.testpy_logger is not None

    def test_init_logging_manager_raises(self, mock_docker_service, mock_config):
        """When LoggingManager.get_instance() raises, falls back to stdlib logging."""
        from unittest.mock import MagicMock, patch

        mock_lm = MagicMock()
        mock_lm.get_instance.side_effect = RuntimeError("init error")
        with patch("processing.test_execution.test_runner._LoggingManager", mock_lm):
            from processing.test_execution.test_runner import ContainerTestRunner

            runner = ContainerTestRunner(mock_docker_service, mock_config)
            assert runner.testpy_logger is not None

    def test_init_security_manager_fails(self, mock_docker_service, mock_config):
        """When SecurityManager init raises, falls back to None."""
        from unittest.mock import patch

        with patch(
            "processing.test_execution.test_runner.SecurityManager",
            side_effect=TypeError("bad args"),
        ):
            from processing.test_execution.test_runner import ContainerTestRunner

            runner = ContainerTestRunner(mock_docker_service, mock_config)
            assert runner.security_manager is None

    def test_init_orchestrator_fails(self, mock_docker_service, mock_config):
        """When create_test_orchestrator raises, falls back to None."""
        from unittest.mock import patch

        with patch(
            "processing.test_execution.test_runner.create_test_orchestrator",
            side_effect=ImportError("no module"),
        ):
            from processing.test_execution.test_runner import ContainerTestRunner

            runner = ContainerTestRunner(mock_docker_service, mock_config)
            assert runner.orchestrator is None


class TestExecuteWithOrchestrator:
    """Cover lines 103-104, 141-162: orchestrator execution path."""

    @pytest.mark.asyncio
    async def test_execute_with_orchestrator(self, runner, mock_docker_service):
        """Test _execute_with_orchestrator creates proper result."""
        from unittest.mock import AsyncMock, MagicMock

        from models.testing_models import TestRequest

        mock_orch_result = MagicMock()
        mock_orch_result.success = True
        mock_orch_result.stdout = "tests passed"
        mock_orch_result.stderr = ""
        mock_orch_result.exit_code = 0
        mock_orch_result.execution_time_ms = 1500.0
        mock_orch_result.coverage_report = None
        mock_orch_result.container_id = "orch-container-1"
        mock_orch_result.test_type = MagicMock()
        mock_orch_result.test_type.value = "unit"
        mock_orch_result.performance_metrics = {}
        mock_orch_result.security_issues = []

        runner.orchestrator = MagicMock()
        runner.orchestrator.execute_tests = AsyncMock(return_value=mock_orch_result)

        from datetime import UTC, datetime

        request = TestRequest(
            code="fn main() {}",
            test_code="fn test_main() {}",
            language="go",
        )
        result = await runner._execute_with_orchestrator(request, "test-123", datetime.now(UTC))
        assert result.success is True
        assert result.stdout == "tests passed"
        assert result.execution_time == 1.5  # 1500ms / 1000

    @pytest.mark.asyncio
    async def test_execute_with_orchestrator_no_orchestrator(self, runner):
        """When orchestrator is None, raises RuntimeError."""
        from datetime import UTC, datetime

        from models.testing_models import TestRequest

        runner.orchestrator = None
        request = TestRequest(code="x", test_code="y", language="go")
        with pytest.raises(RuntimeError, match="not available"):
            await runner._execute_with_orchestrator(request, "t1", datetime.now(UTC))


class TestRunTestInContainer:
    """Cover lines 276-279, 285-288, 296, 300, 337-342, 370-374."""

    @pytest.mark.asyncio
    async def test_run_test_python(self, runner, mock_docker_service):
        """Test python test execution in container."""
        from models.testing_models import TestRequest

        request = TestRequest(code="x = 1", test_code="assert x == 1", language="python")
        result = await runner._run_test_in_container("container-123", request, "test-1")
        assert result.success is True
        assert result.stdout == "OK"
        mock_docker_service.install_tools_in_container.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_test_javascript(self, runner, mock_docker_service):
        """Test javascript test execution in container."""
        from models.testing_models import TestRequest

        request = TestRequest(
            code="const x = 1;",
            test_code="test('x', () => { expect(x).toBe(1); });",
            language="javascript",
        )
        result = await runner._run_test_in_container("container-123", request, "test-1")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_test_unsupported_language(self, runner, mock_docker_service):
        """Test unsupported language raises ValueError."""
        # Use a MagicMock to bypass TestRequest validation since 'cobol' is not accepted
        request = MagicMock()
        request.language = "cobol"
        request.timeout = 30
        with pytest.raises(ValueError, match="Unsupported language"):
            await runner._run_test_in_container("container-123", request, "test-1")

    @pytest.mark.asyncio
    async def test_run_test_timeout(self, runner, mock_docker_service):
        """Cover lines 370-374: timeout during test execution."""
        from models.testing_models import TestRequest

        mock_docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("timed out"))
        request = TestRequest(code="x = 1", test_code="assert True", language="python")
        result = await runner._run_test_in_container("container-123", request, "test-1")
        assert result.success is False
        assert result.exit_code == 124
        assert "timed out" in result.error.lower()


class TestSetupTestEnvironment:
    """Cover lines that handle code_path and test_path file reading."""

    @pytest.mark.asyncio
    async def test_setup_with_code_and_test(self, runner, mock_docker_service):
        """Basic setup with inline code and test_code."""
        from models.testing_models import TestRequest

        request = TestRequest(
            code="x = 1",
            test_code="assert x == 1",
            language="python",
        )
        await runner._setup_test_environment("container-123", request)
        # Should create 2 files
        assert mock_docker_service.create_file_in_container.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_js_creates_package_json(self, runner, mock_docker_service):
        """JavaScript setup creates package.json."""
        from models.testing_models import TestRequest

        request = TestRequest(
            code="const x = 1;",
            test_code="test('x', () => {});",
            language="javascript",
        )
        await runner._setup_test_environment("container-123", request)
        # 2 code files + 1 package.json
        assert mock_docker_service.create_file_in_container.call_count == 3

    @pytest.mark.asyncio
    async def test_setup_file_creation_failure(self, runner):
        """Cover the file creation failure path."""
        runner.docker_service.create_file_in_container = AsyncMock(return_value=False)
        # Use MagicMock to avoid model_validator setting test_path='.'
        request = MagicMock()
        request.code = "x = 1"
        request.test_code = None
        request.code_path = None
        request.test_path = None
        request.language = "python"
        with pytest.raises(RuntimeError, match="Failed to create file"):
            await runner._setup_test_environment("container-123", request)


class TestFactory:
    def test_create_container_test_runner(self, mock_docker_service, mock_config):
        from processing.test_execution.test_runner import create_container_test_runner

        runner = create_container_test_runner(mock_docker_service, mock_config)
        assert runner is not None
        assert runner.docker_service is mock_docker_service
