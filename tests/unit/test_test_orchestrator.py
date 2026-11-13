"""Unit tests for TestOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.test_execution.test_orchestrator import (
    _LANGUAGE_COMMANDS,
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestLanguage,
    TestOrchestrator,
    create_test_orchestrator,
)


@pytest.fixture
def docker_service():
    svc = MagicMock()
    svc.create_test_container = AsyncMock(return_value="container-abc")
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(
        return_value={"exit_code": 0, "stdout": "1 passed", "stderr": ""}
    )
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def config():
    return MagicMock()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_create_test_orchestrator_returns_instance(docker_service, config):
    orch = create_test_orchestrator(docker_service, config)
    assert isinstance(orch, TestOrchestrator)


def test_create_test_orchestrator_stores_docker_service(docker_service, config):
    orch = create_test_orchestrator(docker_service, config)
    assert orch.docker_service is docker_service


def test_create_test_orchestrator_default_type(docker_service, config):
    orch = create_test_orchestrator(docker_service, config)
    assert orch.orchestrator_type == OrchestratorType.DOCKER


def test_create_test_orchestrator_custom_type(docker_service, config):
    orch = create_test_orchestrator(docker_service, config, OrchestratorType.LOCAL)
    assert orch.orchestrator_type == OrchestratorType.LOCAL


# ---------------------------------------------------------------------------
# execute_tests — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tests_success(docker_service, config):
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="def add(a, b): return a + b",
        test_files={"test_add.py": "from main import add\ndef test_add(): assert add(1,2)==3"},
        timeout_seconds=30,
    )

    result = await orch.execute_tests(request)

    assert isinstance(result, TestExecutionResult)
    assert result.success is True
    assert result.stdout == "1 passed"
    assert result.exit_code == 0
    assert result.container_id == "container-abc"
    assert result.test_type == OrchestratorType.UNIT


@pytest.mark.asyncio
async def test_execute_tests_failure_exit_code(docker_service, config):
    docker_service.exec_in_container = AsyncMock(
        return_value={"exit_code": 1, "stdout": "", "stderr": "FAILED"}
    )
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(language=TestLanguage.PYTHON)

    result = await orch.execute_tests(request)

    assert result.success is False
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Language-to-command mapping
# ---------------------------------------------------------------------------


def test_python_command():
    assert _LANGUAGE_COMMANDS["python"][0] == "pytest"


def test_javascript_command():
    assert _LANGUAGE_COMMANDS["javascript"] == ["npx", "jest"]


def test_go_command():
    assert _LANGUAGE_COMMANDS["go"] == ["go", "test", "./..."]


def test_rust_command():
    assert _LANGUAGE_COMMANDS["rust"] == ["cargo", "test"]


def test_csharp_command():
    assert _LANGUAGE_COMMANDS["csharp"] == ["dotnet", "test"]


def test_cpp_command():
    assert _LANGUAGE_COMMANDS["cpp"] == ["make", "test"]


def test_java_command():
    assert _LANGUAGE_COMMANDS["java"] == ["mvn", "test"]


@pytest.mark.asyncio
async def test_correct_command_used_for_go(docker_service, config):
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(language=TestLanguage.GO, source_code="package main")

    await orch.execute_tests(request)

    docker_service.exec_in_container.assert_awaited_once()
    call_args = docker_service.exec_in_container.call_args
    assert call_args[0][1] == ["go", "test", "./..."]


# ---------------------------------------------------------------------------
# Container cleanup on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_cleanup_on_exec_failure(docker_service, config):
    docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("container died"))
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(language=TestLanguage.PYTHON)

    result = await orch.execute_tests(request)

    # Should still return a result (not raise)
    assert result.success is False
    assert "container died" in result.stderr
    # Cleanup must be called despite the error
    docker_service.stop_container.assert_awaited_once_with("container-abc")
    docker_service.remove_container_by_id.assert_awaited_once_with("container-abc")


@pytest.mark.asyncio
async def test_container_cleanup_on_success(docker_service, config):
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(language=TestLanguage.PYTHON)

    await orch.execute_tests(request)

    docker_service.stop_container.assert_awaited_once_with("container-abc")
    docker_service.remove_container_by_id.assert_awaited_once_with("container-abc")


# ---------------------------------------------------------------------------
# Legacy execute() method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_execute_returns_dict(docker_service, config):
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(language=TestLanguage.PYTHON)

    result = await orch.execute(request)

    assert isinstance(result, dict)
    assert "success" in result


# ---------------------------------------------------------------------------
# Source + test files are written to container
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_code_written_to_container(docker_service, config):
    orch = TestOrchestrator(docker_service, config)
    request = TestExecutionRequest(
        language=TestLanguage.PYTHON,
        source_code="x = 1",
        test_files={"test_x.py": "assert x == 1"},
    )

    await orch.execute_tests(request)

    # At least one write call for source code
    assert docker_service.create_file_in_container.await_count >= 2
