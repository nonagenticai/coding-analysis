"""
Container-adapted test runner using Python test suite patterns.
Executes tests in isolated Docker containers while leveraging existing TestPy components.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid7

from models.testing_models import (
    SecurityLevel,
    TestRequest,
    TestResult,
)
from python_test_suite.src.config import TestConfig
from python_test_suite.src.security_manager import (
    SecurityManager,
    TestValidationSecurityConfig,
)
from services.docker_service import DockerService
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestLanguage,
    TestOrchestrator,
    create_test_orchestrator,
)

_LoggingManager: type | None = None
try:
    from python_test_suite.src.logging_manager import LoggingManager as _LM  # noqa: N814

    _LoggingManager = _LM
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ContainerTestRunner:
    """
    Test runner adapted for container execution while maintaining TestPy security patterns.
    Now enhanced with orchestrator integration for multi-language support.
    """

    def __init__(self, docker_service: DockerService, config: TestConfig):
        self.docker_service = docker_service
        self.config = config

        # Initialize TestPy components for local validation
        self.testpy_logger: Any
        try:
            if _LoggingManager is not None:
                self.testpy_logger = _LoggingManager.get_instance()
            else:
                self.testpy_logger = logging.getLogger("testpy")
        except (ImportError, AttributeError, RuntimeError):  # fmt: skip
            # Fallback if LoggingManager not available
            self.testpy_logger = logging.getLogger("testpy")

        self.security_manager: SecurityManager | None
        try:
            # SecurityManager expects SecurityConfig, not TestConfig
            # Create a minimal security config from test config
            security_config = TestValidationSecurityConfig(
                max_execution_time=getattr(config, "max_execution_time", 30),
                max_memory_mb=getattr(config, "max_memory_mb", 512),
                max_file_size_mb=getattr(config, "max_file_size_mb", 10),
                max_code_length=getattr(config, "max_code_length", 10000),
                allowed_modules=getattr(config, "allowed_imports", ["pytest", "math"]),
            )
            self.security_manager = SecurityManager(self.testpy_logger, security_config)
        except (ImportError, AttributeError, RuntimeError, TypeError):  # fmt: skip
            # Fallback if SecurityManager initialization fails
            self.security_manager = None
            logger.warning("TestPy SecurityManager not available, using basic validation")

        # Note: TestPyRunner was removed as it was not used in the codebase

        # Initialize orchestrator for multi-language support
        self.orchestrator: TestOrchestrator | None
        try:
            self.orchestrator = create_test_orchestrator(docker_service, config)
            logger.info("Test orchestrator initialized successfully")
        except (ImportError, AttributeError, RuntimeError, TypeError):  # fmt: skip
            self.orchestrator = None
            logger.warning("Test orchestrator initialization failed, using legacy execution")

    async def execute_test(self, request: TestRequest) -> TestResult:
        """Execute test in secure container environment with orchestrator support."""
        test_id = str(uuid7())
        start_time = datetime.now(UTC)

        try:
            logger.info("Starting container test execution %s", test_id)

            # Try orchestrator first for enhanced multi-language support
            if self.orchestrator and self._should_use_orchestrator(request):
                logger.info("Using orchestrator for %s test execution", request.language)
                return await self._execute_with_orchestrator(request, test_id, start_time)

            # Fallback to legacy execution
            logger.info("Using legacy execution for %s test", request.language)
            return await self._execute_legacy(request, test_id, start_time)

        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Container test execution %s failed: %s", test_id, e)
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                created_at=start_time,
                language=request.language,
                security_level=request.security_level,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error=str(e),
            )

    def _should_use_orchestrator(self, request: TestRequest) -> bool:
        """Determine if orchestrator should be used for this request."""
        # Use orchestrator for Go and Rust (have dedicated containers)
        orchestrator_languages = ["go", "rust"]
        return request.language in orchestrator_languages

    async def _execute_with_orchestrator(
        self, request: TestRequest, test_id: str, start_time: datetime
    ) -> TestResult:
        """Execute test using the orchestrator."""
        # Convert TestRequest to TestExecutionRequest
        test_files = {}
        if request.test_code:
            test_extension = self._get_test_file_extension(request.language)
            test_files[f"test_main{test_extension}"] = request.test_code

        orchestrator_request = TestExecutionRequest(
            language=TestLanguage(request.language),
            test_type=OrchestratorType.UNIT,  # Default to unit tests
            source_code=request.code or "",
            container_name=request.container_name,
            test_files=test_files,
            timeout_seconds=request.timeout or 300,
            security_level=request.security_level,
        )

        # Execute through orchestrator
        if self.orchestrator is None:
            raise RuntimeError("TestOrchestrator not available")
        orchestrator_result = await self.orchestrator.execute_tests(orchestrator_request)

        # Convert back to TestResult
        return TestResult(
            test_id=test_id,
            success=orchestrator_result.success,
            stdout=orchestrator_result.stdout,
            stderr=orchestrator_result.stderr,
            exit_code=orchestrator_result.exit_code,
            execution_time=orchestrator_result.execution_time_ms / 1000.0,  # Convert to seconds
            created_at=start_time,
            language=request.language,
            security_level=request.security_level,
            coverage_data=orchestrator_result.coverage_report,
            test_metadata={
                "container_id": orchestrator_result.container_id,
                "test_type": orchestrator_result.test_type.value,
                "performance_metrics": orchestrator_result.performance_metrics,
                "security_issues": orchestrator_result.security_issues,
            },
            test_type=orchestrator_result.test_type.value,
            container_name=orchestrator_result.container_id,
            quality_report=None,
            error=None,
        )

    async def _execute_legacy(
        self, request: TestRequest, test_id: str, _start_time: datetime
    ) -> TestResult:
        """Execute test using legacy container execution."""
        # Pre-execution validation using TestPy security
        await self._validate_request(request)

        # Create secure container
        docker_id = await self._create_test_container(request, test_id)

        try:
            # Setup test environment in container
            await self._setup_test_environment(docker_id, request)

            # Execute test using adapted TestPy patterns
            result = await self._run_test_in_container(docker_id, request, test_id)

            return result

        finally:
            # Always cleanup container
            await self._cleanup_container(docker_id)

    async def _validate_request(self, request: TestRequest) -> None:
        """Validate request using TestPy security manager."""
        try:
            # Basic validation
            if not request.code and not request.test_code and not request.code_path:
                raise ValueError("Request must contain code, test_code, or code_path")

            # Validate code content using TestPy if available
            if self.security_manager:
                if request.code:
                    # Use check_code_safety method if available
                    checker = getattr(self.security_manager, "check_code_safety", None) or getattr(
                        self.security_manager, "validate_code", None
                    )
                    if checker:
                        checker(request.code)

                if request.test_code:
                    checker = getattr(self.security_manager, "check_code_safety", None) or getattr(
                        self.security_manager, "validate_code", None
                    )
                    if checker:
                        checker(request.test_code)

            # Basic length validation
            total_code = (request.code or "") + (request.test_code or "")
            if len(total_code) > 50000:  # Basic limit
                raise ValueError(f"Code length exceeds limit: {len(total_code)} > 50000")

        except Exception as e:
            logger.error("Request validation failed: %s", e)
            raise ValueError(f"Security validation failed: {str(e)}") from e

    async def _create_test_container(self, request: TestRequest, test_id: str) -> str:
        """Create secure container for test execution."""
        # Get appropriate container image
        image = self.docker_service.get_language_container_image(request.language)

        # Configure container security settings
        container_config = {
            "image": image,
            "working_dir": "/app/test",
            "mem_limit": self._get_memory_limit(request.security_level),
            "cpu_quota": self._get_cpu_quota(request.security_level),
            "network_disabled": True,
            "read_only": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": {"/tmp": "noexec,nosuid,size=100m"},  # nosec B108
            "labels": {"test_id": test_id, "service": "test_execution"},
            "language": request.language,
        }

        docker_id = await self.docker_service.create_test_container(container_config)
        await self.docker_service.start_container(docker_id)

        logger.info("Created test container %s for test %s", docker_id, test_id)
        return docker_id

    async def _setup_test_environment(self, docker_id: str, request: TestRequest) -> None:
        """Setup test files and environment in container."""
        # Prepare test files
        files_to_create = []

        # Handle code file
        if request.code:
            extension = self._get_file_extension(request.language)
            files_to_create.append((f"main{extension}", request.code))
        elif request.code_path and Path(request.code_path).exists():
            code_content = Path(request.code_path).read_text(encoding="utf-8")
            filename = Path(request.code_path).name
            files_to_create.append((filename, code_content))

        # Handle test file
        if request.test_code:
            test_extension = self._get_test_file_extension(request.language)
            files_to_create.append((f"test_main{test_extension}", request.test_code))
        elif request.test_path and Path(request.test_path).exists():
            test_content = Path(request.test_path).read_text(encoding="utf-8")
            test_filename = Path(request.test_path).name
            files_to_create.append((test_filename, test_content))

        # Create files in container
        for filename, content in files_to_create:
            success = await self.docker_service.create_file_in_container(
                docker_id, f"/app/test/{filename}", content
            )
            if not success:
                raise RuntimeError(f"Failed to create file {filename} in container")

        # Create package.json for JavaScript/TypeScript projects
        if request.language in ["javascript", "typescript"]:
            await self._create_js_package_file(docker_id, request.language)

        logger.debug("Created %d test files in container %s", len(files_to_create), docker_id)

    async def _create_js_package_file(self, docker_id: str, language: str) -> None:
        """Create package.json for JavaScript/TypeScript testing."""
        package_json: dict[str, Any] = {
            "name": "test-project",
            "version": "1.0.0",
            "scripts": {"test": ("jest" if language == "javascript" else "jest --preset ts-jest")},
            "devDependencies": {"jest": "^29.0.0"},
        }

        if language == "typescript":
            # devDependencies is a dict, not a Collection
            dev_deps = package_json["devDependencies"]
            dev_deps["ts-jest"] = "^29.0.0"
            dev_deps["@types/jest"] = "^29.0.0"
            dev_deps["typescript"] = "^5.0.0"

        package_content = json.dumps(package_json, indent=2)

        await self.docker_service.create_file_in_container(
            docker_id, "/app/test/package.json", package_content
        )

    async def _run_test_in_container(
        self, docker_id: str, request: TestRequest, test_id: str
    ) -> TestResult:
        """Execute test in container using language-specific test runner."""
        start_time = datetime.now(UTC)

        # Build test command based on language
        if request.language == "python":
            # Install pytest if not available
            await self.docker_service.install_tools_in_container(docker_id, ["pytest"], "python")
            command = ["python", "-m", "pytest", "-v", "--tb=short", "."]
        elif request.language in ["javascript", "typescript"]:
            # Install dependencies and run tests
            await self.docker_service.exec_in_container(docker_id, ["npm", "install"], timeout=60)
            command = ["npm", "test"]
        else:
            raise ValueError(f"Unsupported language: {request.language}")

        # Execute test with timeout
        try:
            result = await self.docker_service.exec_in_container(
                docker_id, command, timeout=request.timeout or 30
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return TestResult(
                test_id=test_id,
                success=result["exit_code"] == 0,
                stdout=result["stdout"],
                stderr=result["stderr"],
                exit_code=result["exit_code"],
                execution_time=execution_time,
                created_at=start_time,
                language=request.language,
                security_level=request.security_level,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=docker_id,
                quality_report=None,
                error=None,
            )

        except TimeoutError:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            logger.warning("Test execution timed out for %s", test_id)

            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr="Test execution timed out",
                exit_code=124,  # Standard timeout exit code
                execution_time=execution_time,
                created_at=start_time,
                language=request.language,
                security_level=request.security_level,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=docker_id,
                quality_report=None,
                error="Test execution timed out",
            )

    async def _cleanup_container(self, docker_id: str) -> None:
        """Cleanup test container."""
        try:
            await self.docker_service.stop_container(docker_id)
            await self.docker_service.remove_container_by_id(docker_id)
            logger.debug("Cleaned up container %s", docker_id)
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Failed to cleanup container %s: %s", docker_id, e)

    def _get_memory_limit(self, security_level: SecurityLevel) -> str:
        """Get memory limit based on security level."""
        limits = {
            SecurityLevel.HIGH: "256m",
            SecurityLevel.MEDIUM: "512m",
            SecurityLevel.LOW: "1g",
        }
        return limits.get(security_level, "512m")

    def _get_cpu_quota(self, security_level: SecurityLevel) -> int:
        """Get CPU quota based on security level."""
        quotas = {
            SecurityLevel.HIGH: 25000,  # 0.25 CPU
            SecurityLevel.MEDIUM: 50000,  # 0.5 CPU
            SecurityLevel.LOW: 100000,  # 1.0 CPU
        }
        return quotas.get(security_level, 50000)

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
        }
        return extensions.get(language, ".py")

    def _get_test_file_extension(self, language: str) -> str:
        """Get test file extension for language."""
        if language == "python":
            return ".py"
        if language in ["javascript", "typescript"]:
            return ".test.js" if language == "javascript" else ".test.ts"
        return ".py"


# Factory function for dependency injection
def create_container_test_runner(
    docker_service: DockerService, config: TestConfig
) -> ContainerTestRunner:
    """Factory function to create ContainerTestRunner."""
    return ContainerTestRunner(docker_service, config)
