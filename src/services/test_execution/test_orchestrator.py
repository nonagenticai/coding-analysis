"""Test orchestrator service for executing tests in Docker containers."""

from __future__ import annotations

import enum
import logging
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OrchestratorType(enum.StrEnum):
    """Type of test orchestrator."""

    DOCKER = "docker"
    LOCAL = "local"
    KUBERNETES = "kubernetes"
    UNIT = "unit"
    INTEGRATION = "integration"


class TestLanguage(enum.StrEnum):
    __test__ = False
    """Supported test languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    CPP = "cpp"


# Language → test command mapping
_LANGUAGE_COMMANDS: dict[str, list[str]] = {
    "python": ["pytest", "-v", "--tb=short"],
    "javascript": ["npx", "jest"],
    "typescript": ["npx", "jest"],
    "java": ["mvn", "test"],
    "go": ["go", "test", "./..."],
    "rust": ["cargo", "test"],
    "csharp": ["dotnet", "test"],
    "cpp": ["make", "test"],
}

# Language → source file extension
_LANGUAGE_EXTENSIONS: dict[str, str] = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "csharp": ".cs",
    "cpp": ".cpp",
}


class TestExecutionRequest(BaseModel):
    __test__ = False
    """Request to execute tests."""

    # New fields expected by test_runner.py
    language: TestLanguage = TestLanguage.PYTHON
    test_type: OrchestratorType = OrchestratorType.UNIT
    source_code: str = ""
    container_name: str | None = None
    test_files: dict[str, str] = {}
    timeout_seconds: int = 300
    security_level: Any = None

    # Legacy fields kept for backwards compat
    code: str = ""
    test_code: str = ""
    timeout: int = 30
    environment: dict[str, Any] = {}


class TestExecutionResult(BaseModel):
    __test__ = False
    """Result of a test execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    container_id: str = ""
    test_type: OrchestratorType = OrchestratorType.UNIT
    performance_metrics: dict[str, Any] = {}
    security_issues: list[Any] = []
    coverage_report: dict[str, Any] | None = None


class TestOrchestrator:
    __test__ = False
    """Orchestrator for test execution across containers."""

    def __init__(
        self,
        docker_service: Any,
        config: Any,
        orchestrator_type: OrchestratorType = OrchestratorType.DOCKER,
    ) -> None:
        """Initialize test orchestrator."""
        self.docker_service = docker_service
        self.config = config
        self.orchestrator_type = orchestrator_type
        logger.debug("TestOrchestrator initialized (type=%s)", orchestrator_type)

    async def execute_tests(self, request: TestExecutionRequest) -> TestExecutionResult:
        """Execute tests for the given request.

        1. Map language to test command.
        2. Create container via docker_service.
        3. Write source + test files into container.
        4. Execute test command.
        5. Return TestExecutionResult.
        6. Cleanup container in finally block.
        """
        language = str(request.language).lower()
        test_cmd = _LANGUAGE_COMMANDS.get(language, ["pytest"])
        container_id = ""
        start_ts = time.monotonic()

        try:
            # 1. Determine container image / config
            container_config: dict[str, Any] = {
                "language": language,
                "working_dir": "/app/test",
                "network_disabled": True,
            }
            if request.container_name:
                container_config["name"] = request.container_name

            # 2. Create container
            container_id = await self.docker_service.create_test_container(container_config)

            # 3. Write source code
            if request.source_code:
                ext = _LANGUAGE_EXTENSIONS.get(language, ".py")
                await self.docker_service.create_file_in_container(
                    container_id,
                    f"/app/test/main{ext}",
                    request.source_code,
                )

            # Write all test files
            for filename, content in (request.test_files or {}).items():
                await self.docker_service.create_file_in_container(
                    container_id,
                    f"/app/test/{filename}",
                    content,
                )

            # 4. Execute test command
            exec_result = await self.docker_service.exec_in_container(
                container_id,
                test_cmd,
                timeout=request.timeout_seconds,
            )

            elapsed_ms = (time.monotonic() - start_ts) * 1000.0

            return TestExecutionResult(
                success=exec_result.get("exit_code", 1) == 0,
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", ""),
                exit_code=exec_result.get("exit_code", 1),
                execution_time_ms=elapsed_ms,
                container_id=container_id,
                test_type=request.test_type,
                performance_metrics={"duration_ms": elapsed_ms},
                security_issues=[],
                coverage_report=None,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_ts) * 1000.0
            logger.error("Test execution failed: %s", exc)
            return TestExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time_ms=elapsed_ms,
                container_id=container_id,
                test_type=request.test_type,
                performance_metrics={},
                security_issues=[],
                coverage_report=None,
            )

        finally:
            # 6. Cleanup container
            if container_id:
                try:
                    await self.docker_service.stop_container(container_id)
                    await self.docker_service.remove_container_by_id(container_id)
                except Exception as cleanup_exc:
                    logger.warning("Container cleanup failed for %s: %s", container_id, cleanup_exc)

    async def execute(self, request: TestExecutionRequest) -> dict[str, Any]:
        """Legacy execute method — delegates to execute_tests()."""
        result = await self.execute_tests(request)
        return result.model_dump()


def create_test_orchestrator(
    docker_service: Any,
    config: Any,
    orchestrator_type: OrchestratorType = OrchestratorType.DOCKER,
) -> TestOrchestrator:
    """Create a test orchestrator instance.

    Args:
        docker_service: DockerService for container operations.
        config: Test configuration.
        orchestrator_type: Type of orchestrator to create.

    Returns:
        TestOrchestrator instance.
    """
    return TestOrchestrator(
        docker_service=docker_service,
        config=config,
        orchestrator_type=orchestrator_type,
    )
