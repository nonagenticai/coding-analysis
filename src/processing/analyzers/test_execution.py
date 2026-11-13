"""
Test Execution Module for Integration Tests.

This module provides functionality for executing integration tests across
different analysis components. It coordinates test execution, manages
test environments, and handles component-specific testing logic.

The module supports:
- Sequential and parallel test execution
- Component-specific test runners
- Test environment setup and teardown
- Error handling and recovery
- Test timing and performance measurement

Classes:
    ComponentTestRunner: Executes tests for individual analysis components
    IntegrationTestExecutor: Coordinates execution of complete test suites
    TestEnvironmentManager: Manages test execution environments

Functions:
    execute_component_tests: Execute tests for a specific component
    run_integration_test_suite: Run complete integration test suite
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from models.static_analysis_models import (
    CodeChange,
    DocumentationType,
    OutputFormat,
    ValidationType,
)
from models.static_analysis_models import StaticAnalysisChangeType as ChangeType
from models.static_analysis_models import (
    StaticAnalysisSecurityLevel as SecurityLevel,
)

# Import analysis components
from ..ast_analysis.function_extractor import FunctionExtractor
from .call_graph_analyzer import CallGraphConfig, CallGraphGenerator
from .documentation_engine import DocumentationConfig, DocumentationEngine
from .symbolic_analyzer import SymbolicValidator, ValidationConfig
from .test_discovery import TestFileManager, TestScenario

logger = logging.getLogger(__name__)


class TestExecutionStatus(Enum):
    """Status values for test execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestExecutionResult:
    """Results from test execution."""

    component: str
    test_name: str
    status: TestExecutionStatus
    execution_time: float
    results: dict[str, Any]
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ComponentTestConfig:
    """Configuration for component testing."""

    component_name: str
    test_scenarios: list[TestScenario]
    timeout: float = 30.0
    retry_count: int = 0
    parallel_execution: bool = False


class ComponentTestRunner:
    """Executes tests for individual analysis components.

    This class handles the execution of tests for specific analysis
    components like call graph analysis, signature extraction, etc.
    Each component has specialized test execution logic.
    """

    def __init__(self, file_manager: TestFileManager):
        """Initialize the component test runner.

        Args:
            file_manager: Test file manager for temporary file handling
        """
        self.file_manager = file_manager
        self.execution_results: list[TestExecutionResult] = []

    async def run_call_graph_tests(self, test_scenario: TestScenario) -> TestExecutionResult:
        """Execute call graph analysis tests.

        Args:
            test_scenario: Test scenario for call graph analysis

        Returns:
            TestExecutionResult: Results from call graph testing
        """
        start_time = time.time()

        try:
            logger.info("Executing call graph analysis tests")

            # Create test file
            test_file_path = self.file_manager.create_test_file(test_scenario.test_code, ".py")

            # Initialize analyzer
            analyzer = CallGraphGenerator()

            # Test single file analysis
            file_result = await analyzer.analyze_file(test_file_path)

            # Test comprehensive analysis
            test_project = self.file_manager.create_integration_test_project()

            config = CallGraphConfig(
                output_format=OutputFormat.JSON,
                include_external=False,
                max_depth=5,
            )

            comprehensive_result = await analyzer.generate_comprehensive_call_graph(
                test_project.root_dir, config
            )

            # Compile results
            results = {
                "single_file": {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_call_graph": "call_graph" in file_result.metadata,
                },
                "comprehensive": {
                    "success": bool(comprehensive_result.analysis_id),
                    "has_nodes": len(comprehensive_result.call_graph.get("nodes", [])) > 0,
                    "has_edges": len(comprehensive_result.call_graph.get("edges", [])) > 0,
                    "has_metrics": bool(comprehensive_result.metrics),
                    "found_orphaned": len(comprehensive_result.orphaned_functions) > 0,
                    "total_functions": comprehensive_result.metrics.total_functions,
                },
            }

            execution_time = time.time() - start_time

            return TestExecutionResult(
                component="call_graph",
                test_name=test_scenario.name,
                status=TestExecutionStatus.COMPLETED,
                execution_time=execution_time,
                results=results,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in call graph tests: {e}")

            return TestExecutionResult(
                component="call_graph",
                test_name=test_scenario.name,
                status=TestExecutionStatus.FAILED,
                execution_time=execution_time,
                results={},
                error_message=str(e),
            )

    async def run_signature_analysis_tests(
        self, test_scenario: TestScenario
    ) -> TestExecutionResult:
        """Execute signature analysis tests.

        Args:
            test_scenario: Test scenario for signature analysis

        Returns:
            TestExecutionResult: Results from signature analysis testing
        """
        start_time = time.time()

        try:
            logger.info("Executing signature analysis tests")

            # Create test file
            test_file_path = self.file_manager.create_test_file(test_scenario.test_code, ".py")

            # Initialize extractor
            extractor = FunctionExtractor(enable_type_inference=True, analyze_usage_patterns=True)

            # Test basic signature extraction
            signatures = extractor.extract_function_signatures(test_file_path)

            # Test comprehensive signature analysis
            comprehensive = extractor.extract_comprehensive_signatures(test_file_path)

            # Compile results
            results = {
                "basic": {
                    "success": not signatures.get("error"),
                    "function_count": len(
                        [k for k in signatures if k not in ["error", "debug_info"]]
                    ),
                    "has_docstrings": any(
                        sig.get("docstring") for sig in signatures.values() if isinstance(sig, dict)
                    ),
                },
                "comprehensive": {
                    "success": not comprehensive.get("error"),
                    "has_analysis_id": bool(comprehensive.get("analysis_id")),
                    "function_count": len(comprehensive.get("functions", [])),
                    "type_coverage": comprehensive.get("type_coverage", 0.0),
                    "has_external_deps": len(comprehensive.get("external_dependencies", [])) > 0,
                    "has_usage_examples": any(
                        func.get("usage_examples") for func in comprehensive.get("functions", [])
                    ),
                    "has_complexity_metrics": any(
                        func.get("complexity_metrics")
                        for func in comprehensive.get("functions", [])
                    ),
                },
            }

            execution_time = time.time() - start_time

            return TestExecutionResult(
                component="signature_analysis",
                test_name=test_scenario.name,
                status=TestExecutionStatus.COMPLETED,
                execution_time=execution_time,
                results=results,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in signature analysis tests: {e}")

            return TestExecutionResult(
                component="signature_analysis",
                test_name=test_scenario.name,
                status=TestExecutionStatus.FAILED,
                execution_time=execution_time,
                results={},
                error_message=str(e),
            )

    async def run_symbolic_validation_tests(
        self, test_scenario: TestScenario
    ) -> TestExecutionResult:
        """Execute symbolic validation tests.

        Args:
            test_scenario: Test scenario for symbolic validation

        Returns:
            TestExecutionResult: Results from symbolic validation testing
        """
        start_time = time.time()

        try:
            logger.info("Executing symbolic validation tests")

            # Create test file
            test_file_path = self.file_manager.create_test_file(test_scenario.test_code, ".py")

            # Initialize analyzer
            analyzer = SymbolicValidator()

            # Test single file analysis
            file_result = await analyzer.analyze_file(test_file_path)

            # Test code change validation
            code_changes = [
                CodeChange(
                    file_path="test.py",
                    original_code=test_scenario.test_code,
                    modified_code=test_scenario.test_code.replace(
                        "return a / b", "return a / b if b != 0 else 0"
                    ),
                    change_type=ChangeType.MODIFIED,
                )
            ]

            config = ValidationConfig(
                validation_types=[
                    ValidationType.SYMBOLIC_EXECUTION,
                    ValidationType.TAINT_ANALYSIS,
                ],
                security_level=SecurityLevel.MEDIUM,
            )

            validation_result = await analyzer.validate_code_changes(code_changes, config)

            # Compile results
            results = {
                "single_file": {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_analysis": "symbolic_analysis" in file_result.metadata,
                },
                "code_changes": {
                    "success": bool(validation_result.analysis_id),
                    "has_results": len(validation_result.validation_results) > 0,
                    "safety_score": validation_result.overall_safety_score,
                    "has_recommendations": len(validation_result.recommendations) > 0,
                    "blocking_issues": len(validation_result.blocking_issues),
                },
            }

            execution_time = time.time() - start_time

            return TestExecutionResult(
                component="symbolic_validation",
                test_name=test_scenario.name,
                status=TestExecutionStatus.COMPLETED,
                execution_time=execution_time,
                results=results,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in symbolic validation tests: {e}")

            return TestExecutionResult(
                component="symbolic_validation",
                test_name=test_scenario.name,
                status=TestExecutionStatus.FAILED,
                execution_time=execution_time,
                results={},
                error_message=str(e),
            )

    async def run_documentation_tests(self, test_scenario: TestScenario) -> TestExecutionResult:
        """Execute documentation generation tests.

        Args:
            test_scenario: Test scenario for documentation testing

        Returns:
            TestExecutionResult: Results from documentation testing
        """
        start_time = time.time()

        try:
            logger.info("Executing documentation generation tests")

            # Create test project
            test_project = self.file_manager.create_integration_test_project()
            main_file = test_project.files["src/main.py"]

            # Initialize engine
            engine = DocumentationEngine()

            # Test single file analysis
            file_result = await engine.analyze_file(main_file)

            # Test project documentation
            project_result = await engine.generate_project_documentation(test_project.root_dir)

            # Test comprehensive documentation
            config = DocumentationConfig(
                doc_types=[
                    DocumentationType.ARCHITECTURE,
                    DocumentationType.API_REFERENCE,
                ],
                include_diagrams=True,
            )

            comprehensive_result = await engine.generate_comprehensive_documentation(
                test_project.root_dir, config
            )

            # Compile results
            results = {
                "single_file": {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_documentation": "documentation" in file_result.metadata,
                },
                "project": {
                    "success": not project_result.get("error"),
                    "has_structure": "project_structure" in project_result,
                    "has_api_docs": "api_documentation" in project_result,
                    "coverage": project_result.get("coverage", 0.0),
                },
                "comprehensive": {
                    "success": bool(comprehensive_result.analysis_id),
                    "has_sections": len(comprehensive_result.documentation) > 0,
                    "quality_score": comprehensive_result.quality_score,
                    "generated_files": len(comprehensive_result.generated_files),
                },
            }

            execution_time = time.time() - start_time

            return TestExecutionResult(
                component="documentation",
                test_name=test_scenario.name,
                status=TestExecutionStatus.COMPLETED,
                execution_time=execution_time,
                results=results,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in documentation tests: {e}")

            return TestExecutionResult(
                component="documentation",
                test_name=test_scenario.name,
                status=TestExecutionStatus.FAILED,
                execution_time=execution_time,
                results={},
                error_message=str(e),
            )

    async def run_performance_tests(self, test_scenario: TestScenario) -> TestExecutionResult:
        """Execute performance tests across all components.

        Args:
            test_scenario: Test scenario for performance testing

        Returns:
            TestExecutionResult: Results from performance testing
        """
        start_time = time.time()

        try:
            logger.info("Executing performance tests")

            # Create large test file
            test_file_path = self.file_manager.create_test_file(test_scenario.test_code, ".py")

            performance_results = {}

            # Test call graph performance
            component_start = time.time()
            call_graph_generator = CallGraphGenerator()
            await call_graph_generator.analyze_file(test_file_path)
            performance_results["call_graph_time"] = time.time() - component_start

            # Test signature analysis performance
            component_start = time.time()
            function_extractor = FunctionExtractor()
            function_extractor.extract_comprehensive_signatures(test_file_path)
            performance_results["signature_analysis_time"] = time.time() - component_start

            # Test symbolic analysis performance
            component_start = time.time()
            symbolic_validator = SymbolicValidator()
            await symbolic_validator.analyze_file(test_file_path)
            performance_results["symbolic_analysis_time"] = time.time() - component_start

            # Test documentation performance
            component_start = time.time()
            doc_engine = DocumentationEngine()
            await doc_engine.analyze_file(test_file_path)
            performance_results["documentation_time"] = time.time() - component_start

            # Compile results
            results = {
                "individual_times": performance_results,
                "total_time": sum(performance_results.values()),
                "acceptable_performance": all(
                    t < 30.0 for t in performance_results.values()
                ),  # 30 second limit
                "file_size_lines": len(test_scenario.test_code.split("\n")),
            }

            execution_time = time.time() - start_time

            return TestExecutionResult(
                component="performance",
                test_name=test_scenario.name,
                status=TestExecutionStatus.COMPLETED,
                execution_time=execution_time,
                results=results,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error in performance tests: {e}")

            return TestExecutionResult(
                component="performance",
                test_name=test_scenario.name,
                status=TestExecutionStatus.FAILED,
                execution_time=execution_time,
                results={},
                error_message=str(e),
            )

    async def run_component_tests(self, config: ComponentTestConfig) -> list[TestExecutionResult]:
        """Run tests for a specific component.

        Args:
            config: Configuration for component testing

        Returns:
            List of test execution results
        """
        results = []

        component_runners = {
            "call_graph": self.run_call_graph_tests,
            "signature_analysis": self.run_signature_analysis_tests,
            "symbolic_validation": self.run_symbolic_validation_tests,
            "documentation": self.run_documentation_tests,
            "performance": self.run_performance_tests,
        }

        runner = component_runners.get(config.component_name)
        if not runner:
            raise ValueError(f"Unknown component: {config.component_name}")

        if config.parallel_execution and len(config.test_scenarios) > 1:
            # Run tests in parallel
            tasks = [runner(scenario) for scenario in config.test_scenarios]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed test results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = TestExecutionResult(
                        component=config.component_name,
                        test_name=config.test_scenarios[i].name,
                        status=TestExecutionStatus.FAILED,
                        execution_time=0.0,
                        results={},
                        error_message=str(result),
                    )
        else:
            # Run tests sequentially
            for scenario in config.test_scenarios:
                try:
                    # Apply timeout
                    result = await asyncio.wait_for(runner(scenario), timeout=config.timeout)
                    results.append(result)
                except TimeoutError:
                    results.append(
                        TestExecutionResult(
                            component=config.component_name,
                            test_name=scenario.name,
                            status=TestExecutionStatus.FAILED,
                            execution_time=config.timeout,
                            results={},
                            error_message=f"Test timed out after {config.timeout} seconds",
                        )
                    )
                except Exception as e:
                    results.append(
                        TestExecutionResult(
                            component=config.component_name,
                            test_name=scenario.name,
                            status=TestExecutionStatus.FAILED,
                            execution_time=0.0,
                            results={},
                            error_message=str(e),
                        )
                    )

        validated_results = cast(list[TestExecutionResult], list(results))
        self.execution_results.extend(validated_results)
        return validated_results


class IntegrationTestExecutor:
    """Coordinates execution of complete integration test suites.

    This class manages the overall test execution workflow, coordinating
    between different components and handling test dependencies.
    """

    def __init__(self):
        """Initialize the integration test executor."""
        self.file_manager = TestFileManager()
        self.component_runner = ComponentTestRunner(self.file_manager)
        self.execution_history: list[TestExecutionResult] = []

    async def run_integration_tests(
        self, test_configs: list[ComponentTestConfig]
    ) -> dict[str, Any]:
        """Run complete integration test suite.

        Args:
            test_configs: List of component test configurations

        Returns:
            Dict containing complete test results
        """
        logger.info("Starting integration test execution")
        start_time = time.time()

        all_results = {}

        try:
            with self.file_manager:  # Ensures cleanup
                for config in test_configs:
                    logger.info(f"Running tests for component: {config.component_name}")

                    component_results = await self.component_runner.run_component_tests(config)
                    all_results[config.component_name] = component_results

                    # Store in execution history
                    self.execution_history.extend(component_results)

            total_time = time.time() - start_time

            # Compile summary
            summary = self._compile_test_summary(all_results, total_time)

            return {
                "results": all_results,
                "summary": summary,
                "execution_time": total_time,
            }

        except Exception as e:
            logger.exception(f"Error during integration test execution: {e}")
            return {
                "results": all_results,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    async def run_end_to_end_integration(self) -> dict[str, Any]:
        """Run end-to-end integration test combining all components.

        Returns:
            Dict containing end-to-end test results
        """
        logger.info("Running end-to-end integration test")
        start_time = time.time()

        try:
            with self.file_manager:
                # Create test project
                test_project = self.file_manager.create_integration_test_project()
                main_file = test_project.files["src/main.py"]

                # Run integration components
                integration_results = await self._run_integration_components(
                    test_project.root_dir, main_file
                )

                execution_time = time.time() - start_time

                return {
                    "individual_results": integration_results,
                    "all_components_working": all(
                        result.get("success", False) for result in integration_results.values()
                    ),
                    "project_files_count": len(list(Path(test_project.root_dir).rglob("*.py"))),
                    "execution_time": execution_time,
                }

        except Exception as e:
            logger.exception(f"Error in end-to-end integration: {e}")
            return {"error": str(e), "execution_time": time.time() - start_time}

    async def _run_integration_components(self, temp_dir: str, main_file: str) -> dict[str, Any]:
        """Run integration test components.

        Args:
            temp_dir: Temporary directory path
            main_file: Main file path

        Returns:
            Dict containing component results
        """
        results = {}

        # 1. Call graph analysis
        call_graph_generator = CallGraphGenerator()
        call_graph_result = await call_graph_generator.generate_comprehensive_call_graph(temp_dir)
        results["call_graph"] = {
            "success": bool(call_graph_result.analysis_id),
            "functions_found": call_graph_result.metrics.total_functions,
        }

        # 2. Signature analysis
        extractor = FunctionExtractor()
        signature_result = extractor.extract_comprehensive_signatures(main_file, temp_dir)
        results["signatures"] = {
            "success": not signature_result.get("error"),
            "functions_analyzed": len(signature_result.get("functions", [])),
        }

        # 3. Symbolic validation
        symbolic_validator = SymbolicValidator()
        code_changes = [
            CodeChange(
                file_path=main_file,
                original_code="# original",
                modified_code="# modified",
                change_type=ChangeType.MODIFIED,
            )
        ]
        validation_result = await symbolic_validator.validate_code_changes(code_changes)
        # `overall_safety_score` is None when the validator abstained — no
        # analyzer produced a result, so there is no score. Report that absence
        # as None: coercing it to 0 would publish the worst possible score for a
        # change nobody measured, which is a fabricated verdict of its own.
        # (The former `int()` was also lossy — it floored every 0-1 score below
        # 1.0 to 0 — so a real 0.8 and a real 0.0 reported identically. The two
        # sibling call sites in this file and in integration_tests.py already
        # pass the score through unrounded; this now matches them.)
        results["validation"] = {
            "success": bool(validation_result.analysis_id),
            "safety_score": validation_result.overall_safety_score,
            "safety_score_computed": validation_result.overall_safety_score is not None,
        }

        # 4. Documentation generation
        doc_engine = DocumentationEngine()
        doc_result = await doc_engine.generate_comprehensive_documentation(temp_dir)
        results["documentation"] = {
            "success": bool(doc_result.analysis_id),
            "quality_score": int(doc_result.quality_score),
        }

        return results

    def _compile_test_summary(
        self, all_results: dict[str, Any], total_time: float
    ) -> dict[str, Any]:
        """Compile summary of test results.

        Args:
            all_results: All test results by component
            total_time: Total execution time

        Returns:
            Dict containing test summary
        """
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for component_results in all_results.values():
            for result in component_results:
                total_tests += 1
                if result.status == TestExecutionStatus.COMPLETED:
                    passed_tests += 1
                else:
                    failed_tests += 1

        return {
            "total_time": total_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "all_tests_passed": failed_tests == 0,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "timestamp": time.time(),
        }


class TestEnvironmentManager:
    """Manages test execution environments and resources.

    This class handles setup and teardown of test environments,
    resource allocation, and environment isolation for test execution.
    """

    def __init__(self):
        """Initialize the test environment manager."""
        self.active_environments: dict[str, Any] = {}
        self.resource_limits = {
            "max_execution_time": 300.0,  # 5 minutes
            "max_memory_mb": 1024,  # 1GB
            "max_temp_files": 100,
        }

    async def setup_test_environment(self, test_id: str) -> dict[str, Any]:
        """Setup a test execution environment.

        Args:
            test_id: Unique test identifier

        Returns:
            Dict containing environment configuration
        """
        logger.info(f"Setting up test environment: {test_id}")

        environment = {
            "test_id": test_id,
            "start_time": time.time(),
            "file_manager": TestFileManager(),
            "temp_files": [],
            "resource_usage": {
                "memory_mb": 0,
                "execution_time": 0.0,
                "temp_files_count": 0,
            },
        }

        self.active_environments[test_id] = environment
        return environment

    async def teardown_test_environment(self, test_id: str):
        """Teardown a test execution environment.

        Args:
            test_id: Test identifier to teardown
        """
        if test_id in self.active_environments:
            logger.info(f"Tearing down test environment: {test_id}")

            environment = self.active_environments[test_id]

            # Cleanup file manager
            if "file_manager" in environment:
                environment["file_manager"].cleanup_test_files()

            # Remove from active environments
            del self.active_environments[test_id]

    def check_resource_limits(self, test_id: str) -> bool:
        """Check if test is within resource limits.

        Args:
            test_id: Test identifier to check

        Returns:
            bool: True if within limits, False otherwise
        """
        if test_id not in self.active_environments:
            return False

        environment = self.active_environments[test_id]
        usage = environment["resource_usage"]

        # Check execution time
        current_time = time.time() - environment["start_time"]
        if current_time > self.resource_limits["max_execution_time"]:
            return False

        # Check temp files count
        if usage["temp_files_count"] > self.resource_limits["max_temp_files"]:
            return False

        return True

    async def cleanup_all_environments(self):
        """Cleanup all active test environments."""
        logger.info("Cleaning up all test environments")

        for test_id in list(self.active_environments.keys()):
            await self.teardown_test_environment(test_id)


# Convenience functions
async def execute_component_tests(
    component_name: str, test_scenarios: list[TestScenario], **kwargs
) -> list[TestExecutionResult]:
    """Execute tests for a specific component.

    Args:
        component_name: Name of component to test
        test_scenarios: List of test scenarios
        **kwargs: Additional configuration options

    Returns:
        List of test execution results
    """
    file_manager = TestFileManager()
    runner = ComponentTestRunner(file_manager)

    config = ComponentTestConfig(
        component_name=component_name, test_scenarios=test_scenarios, **kwargs
    )

    try:
        with file_manager:
            return await runner.run_component_tests(config)
    except Exception as e:
        logger.exception(f"Error executing component tests: {e}")
        return []


async def run_integration_test_suite(
    test_configs: list[ComponentTestConfig],
) -> dict[str, Any]:
    """Run complete integration test suite.

    Args:
        test_configs: List of component test configurations

    Returns:
        Dict containing complete test results
    """
    executor = IntegrationTestExecutor()
    return await executor.run_integration_tests(test_configs)
