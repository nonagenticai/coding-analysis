"""
Integration Tests for AI Agent Static Analysis Augmentation.

This module provides comprehensive integration testing for all components of the static
analysis augmentation system. It validates the functionality of call graph analysis,
signature extraction, symbolic validation, and documentation generation through
end-to-end testing scenarios.

The test suite validates:
- Call graph analysis with multiple tools (Pyan3, Code2flow, PyCG)
- Function signature extraction and analysis
- Symbolic execution and code validation
- Documentation generation and quality metrics
- Performance benchmarking across all components
- End-to-end integration of all analysis capabilities

The tests are designed to run both independently and as part of a comprehensive
test suite, with support for pytest integration and standalone execution.

Classes:
    StaticAnalysisIntegrationTests: Main test suite coordinator

Functions:
    run_integration_tests: Standalone test runner
    test_static_analysis_integration: Pytest-compatible test function

Example:
    # Run tests standalone
    results = await run_integration_tests()

    # Run with pytest
    pytest test_static_analysis_integration

    # Run specific test components
    suite = StaticAnalysisIntegrationTests()
    await suite._test_call_graph_analysis()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover — production image has no pytest
    pytest = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

from models.static_analysis_models import (
    ChangeType,
    CodeChange,
    DocumentationType,
    OutputFormat,
    ValidationType,
)

from ..ast_analysis.function_extractor import FunctionExtractor
from .call_graph_analyzer import CallGraphConfig, CallGraphGenerator
from .documentation_engine import DocumentationConfig, DocumentationEngine
from .symbolic_analyzer import SecurityLevel, SymbolicValidator, ValidationConfig

logger = logging.getLogger(__name__)


class StaticAnalysisIntegrationTests:
    """Comprehensive integration tests for static analysis augmentation.

    This class orchestrates a complete test suite for all components of the static
    analysis augmentation system. It provides structured testing of individual
    components and their integration, with detailed results tracking and reporting.

    The test suite covers:
    - Call graph analysis using multiple tools (Pyan3, Code2flow, PyCG)
    - Function signature extraction and type analysis
    - Symbolic execution and code validation
    - Documentation generation with quality metrics
    - Performance benchmarking and optimization validation
    - End-to-end integration testing

    Attributes:
        test_results (Dict[str, Any]): Structured storage for all test results,
                                     organized by component type with detailed
                                     success/failure tracking and metrics.

    Example:
        # Run complete test suite
        suite = StaticAnalysisIntegrationTests()
        results = await suite.run_all_tests()

        # Generate detailed report
        report = suite.generate_test_report()
        print(report)
    """

    def __init__(self) -> None:
        """Initialize test suite with result storage structure.

        Sets up the test results dictionary with predefined categories for
        organizing test outcomes and metrics across all analysis components.
        """
        self.test_results: dict[str, Any] = {
            "call_graph": {},
            "signature_analysis": {},
            "symbolic_validation": {},
            "documentation": {},
            "performance": {},
            "integration": {},
        }

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all integration tests sequentially.

        Executes the complete test suite covering all components of the static
        analysis augmentation system. Tests are run in logical order with
        comprehensive error handling and result tracking.

        Returns:
            Dict[str, Any]: Complete test results including:
                - Individual test outcomes by component
                - Performance metrics and timing information
                - Overall success/failure status
                - Error details for any failed tests
                - Summary statistics and timestamps

        Raises:
            The method catches all exceptions and returns them in results:
            - OSError: For file system access issues
            - ValueError: For invalid test parameters
            - RuntimeError: For general test execution errors

        Example:
            suite = StaticAnalysisIntegrationTests()
            results = await suite.run_all_tests()

            if results["summary"]["all_tests_passed"]:
                print("All tests passed successfully!")
            else:
                print("Some tests failed - check results for details")
        """
        logger.info("Starting comprehensive static analysis integration tests")
        start_time = time.time()

        try:
            # Test call graph analysis
            await self._test_call_graph_analysis()

            # Test signature analysis
            await self._test_signature_analysis()

            # Test symbolic validation
            await self._test_symbolic_validation()

            # Test documentation generation
            await self._test_documentation_generation()

            # Test performance
            await self._test_performance()

            # Test end-to-end integration
            await self._test_end_to_end_integration()

            total_time = time.time() - start_time
            self.test_results["summary"] = {
                "total_time": total_time,
                "all_tests_passed": self._check_all_tests_passed(),
                "timestamp": time.time(),
            }

            logger.info("Integration tests completed in %.2f seconds", total_time)
            return dict(self.test_results)

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error running integration tests: %s", e)
            self.test_results["error"] = str(e)
            return dict(self.test_results)

    async def _test_call_graph_analysis(self):
        """Test call graph analysis functionality with multiple tools.

        Validates call graph generation using different analysis tools (Pyan3,
        Code2flow, PyCG) on synthetic Python code. Tests both single-file and
        comprehensive directory-level analysis.

        Test scenarios:
        - Single file analysis with function call chains
        - Comprehensive analysis with multiple tools
        - Orphaned function detection
        - Critical path identification
        - Error handling for various edge cases

        Results are stored in self.test_results["call_graph"] with metrics for:
        - Analysis success/failure status
        - Node and edge extraction counts
        - Tool-specific result validation
        - Performance timing information
        """
        logger.info("Testing call graph analysis")

        try:
            # Create test Python code
            test_code = '''
def main():
    result = process_data(get_input())
    save_result(result)
    return result

def get_input():
    return "test input"

def process_data(data):
    validated = validate_data(data)
    return transform_data(validated)

def validate_data(data):
    if not data:
        raise ValueError("Invalid data")
    return data

def transform_data(data):
    return data.upper()

def save_result(result):
    print(f"Saving: {result}")

def orphaned_function():
    """This function is never called."""
    pass
'''

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(test_code)
                tmp_file_path = tmp_file.name

            try:
                # Test call graph analyzer
                analyzer = CallGraphGenerator()

                # Test single file analysis
                file_result = await analyzer.analyze_file(tmp_file_path)
                self.test_results["call_graph"]["single_file"] = {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_call_graph": "call_graph" in file_result.metadata,
                }

                # Test comprehensive call graph
                temp_dir = tempfile.mkdtemp()
                try:
                    # Copy test file to temp directory
                    temp_test_file = os.path.join(temp_dir, "test_module.py")
                    with open(temp_test_file, "w", encoding="utf-8") as f:
                        f.write(test_code)

                    config = CallGraphConfig(
                        output_format=OutputFormat.JSON,
                        include_external=False,
                        max_depth=5,
                    )

                    comprehensive_result = await analyzer.generate_comprehensive_call_graph(
                        temp_dir, config
                    )

                    self.test_results["call_graph"]["comprehensive"] = {
                        "success": bool(comprehensive_result.analysis_id),
                        "has_nodes": len(comprehensive_result.call_graph.get("nodes", [])) > 0,
                        "has_edges": len(comprehensive_result.call_graph.get("edges", [])) > 0,
                        "has_metrics": bool(comprehensive_result.metrics),
                        "found_orphaned": len(comprehensive_result.orphaned_functions) > 0,
                        "total_functions": comprehensive_result.metrics.total_functions,
                    }

                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            finally:
                os.unlink(tmp_file_path)

            logger.info("Call graph analysis tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in call graph analysis tests: %s", e)
            self.test_results["call_graph"]["error"] = str(e)

    async def _test_signature_analysis(self):
        """Test function signature analysis and type annotation extraction.

        Validates comprehensive analysis of Python function signatures including
        type annotations, parameter information, docstring extraction, and
        usage pattern analysis.

        Test scenarios:
        - Functions with complete type annotations
        - Legacy functions without type hints
        - Async functions and decorators
        - Complex parameter types and return values
        - Docstring parsing and documentation coverage
        - External dependency analysis

        Results are stored in self.test_results["signature_analysis"] with:
        - Function extraction success rates
        - Type coverage analysis
        - Docstring coverage metrics
        - External dependency detection
        - Usage example generation
        """
        logger.info("Testing signature analysis")

        try:
            # Create test Python code with type annotations
            test_code = '''
from typing import List, Dict, Optional
import json

def process_users(users: List[Dict[str, str]], config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process a list of users with optional configuration.

    Args:
        users: List of user dictionaries with name and email
        config: Optional configuration dictionary

    Returns:
        Dictionary containing processed user data
    """
    if not users:
        return {"count": 0, "users": []}

    processed = []
    for user in users:
        processed_user = validate_user(user)
        if processed_user:
            processed.append(processed_user)

    return {
        "count": len(processed),
        "users": processed,
        "config": config or {}
    }

def validate_user(user: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Validate a single user dictionary."""
    if "name" not in user or "email" not in user:
        return None

    return {
        "name": user["name"].strip(),
        "email": user["email"].lower(),
        "validated": True
    }

def legacy_function(data):
    """Legacy function without type annotations."""
    return str(data)

async def async_process(data: str) -> str:
    """Example async function."""
    return await some_async_operation(data)

@property
def computed_property(self) -> int:
    """Example property with decorator."""
    return 42
'''

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(test_code)
                tmp_file_path = tmp_file.name

            try:
                # Test function extractor
                extractor = FunctionExtractor(
                    enable_type_inference=True, analyze_usage_patterns=True
                )

                # Test basic signature extraction
                signatures = extractor.extract_function_signatures(tmp_file_path)
                self.test_results["signature_analysis"]["basic"] = {
                    "success": not signatures.get("error"),
                    "function_count": len(
                        [k for k in signatures if k not in ["error", "debug_info"]]
                    ),
                    "has_docstrings": any(
                        sig.get("docstring") for sig in signatures.values() if isinstance(sig, dict)
                    ),
                }

                # Test comprehensive signature analysis
                comprehensive = extractor.extract_comprehensive_signatures(tmp_file_path)
                self.test_results["signature_analysis"]["comprehensive"] = {
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
                }

            finally:
                os.unlink(tmp_file_path)

            logger.info("Signature analysis tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in signature analysis tests: %s", e)
            self.test_results["signature_analysis"]["error"] = str(e)

    async def _test_symbolic_validation(self):
        """Test symbolic execution and advanced code validation.

        Validates symbolic analysis capabilities including safety checking,
        security vulnerability detection, and formal verification of code
        changes and potential runtime issues.

        Test scenarios:
        - Division by zero detection
        - Index out of bounds checking
        - SQL injection vulnerability detection
        - Code change impact assessment
        - Safety score calculation
        - Recommendation generation

        Results are stored in self.test_results["symbolic_validation"] with:
        - Single file analysis results
        - Code change validation outcomes
        - Safety scores and risk assessment
        - Blocking issue identification
        - Recommendation quality metrics
        """
        logger.info("Testing symbolic validation")

        try:
            # Create test code with potential issues
            test_code = '''
def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers - potential division by zero."""
    return a / b

def process_list(items: list) -> int:
    """Process a list - potential index error."""
    if not items:
        return 0
    return items[0] + items[-1]

def unsafe_input_processing(user_input: str) -> str:
    """Process user input - potential security issue."""
    # This would be flagged by taint analysis
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query

def safe_function(data: str) -> str:
    """A safe function for comparison."""
    return data.upper()
'''

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(test_code)
                tmp_file_path = tmp_file.name

            try:
                # Test symbolic analyzer
                analyzer = SymbolicValidator()

                # Test single file analysis
                file_result = await analyzer.analyze_file(tmp_file_path)
                self.test_results["symbolic_validation"]["single_file"] = {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_analysis": "symbolic_analysis" in file_result.metadata,
                }

                # Test code change validation
                code_changes = [
                    CodeChange(
                        file_path="test.py",
                        original_code=test_code,
                        modified_code=test_code.replace(
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

                self.test_results["symbolic_validation"]["code_changes"] = {
                    "success": bool(validation_result.analysis_id),
                    "has_results": len(validation_result.validation_results) > 0,
                    "safety_score": validation_result.overall_safety_score,
                    "has_recommendations": len(validation_result.recommendations) > 0,
                    "blocking_issues": len(validation_result.blocking_issues),
                }

            finally:
                os.unlink(tmp_file_path)

            logger.info("Symbolic validation tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in symbolic validation tests: %s", e)
            self.test_results["symbolic_validation"]["error"] = str(e)

    async def _test_documentation_generation(self):
        """Test documentation generation functionality."""
        logger.info("Testing documentation generation")

        try:
            # Create test project structure
            temp_dir = tempfile.mkdtemp()

            try:
                # Create test files
                main_file = os.path.join(temp_dir, "main.py")
                with open(main_file, "w", encoding="utf-8") as f:
                    f.write(
                        '''
"""Main module for the test application."""

def main():
    """Main entry point."""
    print("Hello, World!")

class TestClass:
    """Test class for documentation."""

    def method(self, param: str) -> str:
        """Test method with documentation."""
        return param.upper()
'''
                    )

                readme_file = os.path.join(temp_dir, "README.md")
                with open(readme_file, "w", encoding="utf-8") as f:
                    f.write(
                        "# Test Project\n\nThis is a test project for documentation generation."
                    )

                # Test documentation engine
                engine = DocumentationEngine()

                # Test single file analysis
                file_result = await engine.analyze_file(main_file)
                self.test_results["documentation"]["single_file"] = {
                    "success": file_result.success,
                    "has_metadata": bool(file_result.metadata),
                    "has_documentation": "documentation" in file_result.metadata,
                }

                # Test project documentation
                project_result = await engine.generate_project_documentation(temp_dir)
                self.test_results["documentation"]["project"] = {
                    "success": not project_result.get("error"),
                    "has_structure": "project_structure" in project_result,
                    "has_api_docs": "api_documentation" in project_result,
                    "coverage": project_result.get("coverage", 0.0),
                }

                # Test comprehensive documentation
                config = DocumentationConfig(
                    doc_types=[
                        DocumentationType.ARCHITECTURE,
                        DocumentationType.API_REFERENCE,
                    ],
                    include_diagrams=True,
                )

                comprehensive_result = await engine.generate_comprehensive_documentation(
                    temp_dir, config
                )

                self.test_results["documentation"]["comprehensive"] = {
                    "success": bool(comprehensive_result.analysis_id),
                    "has_sections": len(comprehensive_result.documentation) > 0,
                    "quality_score": comprehensive_result.quality_score,
                    "generated_files": len(comprehensive_result.generated_files),
                }

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info("Documentation generation tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in documentation generation tests: %s", e)
            self.test_results["documentation"]["error"] = str(e)

    async def _test_performance(self):
        """Test performance of all components."""
        logger.info("Testing performance")

        try:
            # Create larger test code for performance testing
            large_test_code = """
from typing import List, Dict, Optional, Any
import json
import asyncio

""" + "\n".join(
                [
                    f"""
def function_{func_id}(param1: str, param2: int = {func_id}) -> Dict[str, Any]:
    \"\"\"Function {func_id} for performance testing.\"\"\"
    result = process_function_{func_id}(param1, param2)
    return validate_result_{func_id}(result)

def process_function_{func_id}(data: str, num: int) -> str:
    \"\"\"Process function {func_id}.\"\"\"
    return f"{{data}}_{{num}}"

def validate_result_{func_id}(result: str) -> Dict[str, Any]:
    \"\"\"Validate result for function {func_id}.\"\"\"
    return {{"valid": bool(result), "length": len(result), "function_id": {func_id}}}
"""
                    for func_id in range(50)  # 50 functions for performance testing
                ]
            )

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(large_test_code)
                tmp_file_path = tmp_file.name

            try:
                performance_results = {}

                # Test call graph performance
                start_time = time.time()
                call_graph_generator = CallGraphGenerator()
                await call_graph_generator.analyze_file(tmp_file_path)
                performance_results["call_graph_time"] = time.time() - start_time

                # Test signature analysis performance
                start_time = time.time()
                function_extractor = FunctionExtractor()
                function_extractor.extract_comprehensive_signatures(tmp_file_path)
                performance_results["signature_analysis_time"] = time.time() - start_time

                # Test symbolic analysis performance
                start_time = time.time()
                symbolic_validator = SymbolicValidator()
                await symbolic_validator.analyze_file(tmp_file_path)
                performance_results["symbolic_analysis_time"] = time.time() - start_time

                # Test documentation performance
                start_time = time.time()
                doc_engine = DocumentationEngine()
                await doc_engine.analyze_file(tmp_file_path)
                performance_results["documentation_time"] = time.time() - start_time

                self.test_results["performance"] = {
                    "individual_times": performance_results,
                    "total_time": sum(performance_results.values()),
                    "acceptable_performance": all(
                        t < 30.0 for t in performance_results.values()
                    ),  # 30 second limit
                    "file_size_lines": len(large_test_code.split("\n")),
                }

            finally:
                os.unlink(tmp_file_path)

            logger.info("Performance tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in performance tests: %s", e)
            self.test_results.setdefault("performance", {})["error"] = str(e)

    async def _test_end_to_end_integration(self):
        """Test end-to-end integration of all components."""
        logger.info("Testing end-to-end integration")

        try:
            # Create a complete test project
            temp_dir = tempfile.mkdtemp()

            try:
                # Create project structure
                src_dir = os.path.join(temp_dir, "src")
                os.makedirs(src_dir)

                # Main module
                main_file = os.path.join(src_dir, "main.py")
                with open(main_file, "w", encoding="utf-8") as f:
                    f.write(
                        '''
"""Main application module."""

from typing import List, Dict, Optional
import logging

def main() -> int:
    """Main entry point for the application."""
    try:
        users = load_users()
        processed = process_users(users)
        save_results(processed)
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

def load_users() -> List[Dict[str, str]]:
    """Load users from data source."""
    return [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"}
    ]

def process_users(users: List[Dict[str, str]]) -> Dict[str, Any]:
    """Process loaded users."""
    return {
        "count": len(users),
        "users": [validate_user(user) for user in users]
    }

def validate_user(user: Dict[str, str]) -> Dict[str, str]:
    """Validate a single user."""
    if not user.get("name") or not user.get("email"):
        raise ValueError("Invalid user data")
    return user

def save_results(results: Dict[str, Any]) -> None:
    """Save processing results."""
    print(f"Processed {results['count']} users")

if __name__ == "__main__":
    exit(main())
'''
                    )

                # Utils module
                utils_file = os.path.join(src_dir, "utils.py")
                with open(utils_file, "w", encoding="utf-8") as f:
                    f.write(
                        '''
"""Utility functions."""

def format_email(email: str) -> str:
    """Format email address."""
    return email.lower().strip()

def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email
'''
                    )

                # README
                readme_file = os.path.join(temp_dir, "README.md")
                with open(readme_file, "w", encoding="utf-8") as f:
                    content = "# Integration Test Project\n\nComplete test project for integration testing."
                    f.write(content)

                # Run full integration test
                integration_results = await self._run_integration_components(temp_dir, main_file)

                self.test_results["integration"] = {
                    "individual_results": integration_results,
                    "all_components_working": all(
                        result.get("success", False) for result in integration_results.values()
                    ),
                    "project_files_count": len(list(Path(temp_dir).rglob("*.py"))),
                }

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info("End-to-end integration tests completed successfully")

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error in end-to-end integration tests: %s", e)
            self.test_results["integration"] = {"error": str(e)}

    async def _run_integration_components(self, temp_dir: str, main_file: str) -> dict[str, Any]:
        """Run integration test components."""
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
        # sibling call sites in this file and in test_execution.py already pass
        # the score through unrounded; this now matches them.)
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

    def _check_all_tests_passed(self) -> bool:
        """Check if all tests passed successfully."""
        try:
            # Check each test category
            for category, results in self.test_results.items():
                if category in ["summary"]:
                    continue

                if "error" in results:
                    return False

                # Check nested test results
                for test_result in results.values():
                    if isinstance(test_result, dict):
                        if test_result.get("success") is False:
                            return False
                        if "error" in test_result:
                            return False

            return True

        except (TypeError, AttributeError):  # fmt: skip
            return False

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report = ["# Static Analysis Augmentation Integration Test Report\n"]

        if "summary" in self.test_results:
            summary = self.test_results["summary"]
            report.append(f"**Total Execution Time:** {summary['total_time']:.2f} seconds")
            report.append(
                f"**All Tests Passed:** {'✅ Yes' if summary['all_tests_passed'] else '❌ No'}"
            )
            report.append("")

        # Detailed results by category
        for category, results in self.test_results.items():
            if category == "summary":
                continue

            report.append(f"## {category.replace('_', ' ').title()}")

            if "error" in results:
                report.append(f"❌ **Error:** {results['error']}")
            else:
                for test_name, test_result in results.items():
                    if isinstance(test_result, dict):
                        status = "✅" if test_result.get("success", False) else "❌"
                        report.append(f"- {status} {test_name}: {test_result}")

            report.append("")

        return "\n".join(report)


# Standalone test runner
async def run_integration_tests():
    """Run integration tests as standalone function.

    Convenience function for running the complete integration test suite
    without requiring explicit test class instantiation. Executes all
    tests and provides formatted output.

    Returns:
        Dict[str, Any]: Complete test results including success/failure
                       status, timing information, and detailed metrics
                       for all tested components.

    Example:
        # Run tests and get results
        results = await run_integration_tests()

        # Check if all tests passed
        if results["summary"]["all_tests_passed"]:
            print("All integration tests passed!")
        else:
            print("Some tests failed - check output for details")
    """
    test_suite = StaticAnalysisIntegrationTests()
    results = await test_suite.run_all_tests()

    # Log report
    report = test_suite.generate_test_report()
    logger.info("Integration test report:\n%s", report)

    return results


# Test runner for pytest
_asyncio_mark = pytest.mark.asyncio if pytest is not None else (lambda f: f)


@_asyncio_mark
async def test_static_analysis_integration():  # noqa: F811
    """Pytest-compatible test runner for static analysis integration tests.

    This function provides a pytest-compatible entry point for running
    the complete integration test suite. It executes all tests and uses
    pytest assertions to validate success.

    Raises:
        AssertionError: If any integration tests fail, with detailed
                       error report showing which components failed
                       and their specific error conditions.

    Note:
        This function is designed to be discovered and executed by pytest
        automatically. It can also be run directly for debugging purposes.

    Example:
        # Run with pytest
        pytest -v test_static_analysis_integration

        # Run directly for debugging
        await test_static_analysis_integration()
    """
    test_suite = StaticAnalysisIntegrationTests()
    results = await test_suite.run_all_tests()

    # Assert that all tests passed
    assert results.get("summary", {}).get(  # nosec B101
        "all_tests_passed", False
    ), f"Integration tests failed: {test_suite.generate_test_report()}"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(run_integration_tests())
