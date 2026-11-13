"""
C++ test execution and results module.

This module handles test discovery, execution, and result analysis
for C++ projects using various testing frameworks.
"""

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityType,
)


class CppTestExecutor:
    """Test executor for C++ projects supporting multiple testing frameworks."""

    def __init__(self):
        """Initialize the test executor."""
        self.supported_frameworks = {
            "gtest": {
                "patterns": ["*_test.cpp", "*Test.cpp", "test_*.cpp"],
                "executable_patterns": ["*_test", "*Test", "test_*"],
                "includes": ["gtest/gtest.h", "gmock/gmock.h"],
            },
            "catch2": {
                "patterns": ["*_test.cpp", "*Test.cpp", "test_*.cpp"],
                "executable_patterns": ["*_test", "*Test", "test_*"],
                "includes": ["catch2/catch.hpp", "catch.hpp"],
            },
            "doctest": {
                "patterns": ["*_test.cpp", "*Test.cpp", "test_*.cpp"],
                "executable_patterns": ["*_test", "*Test", "test_*"],
                "includes": ["doctest/doctest.h", "doctest.h"],
            },
            "cppunit": {
                "patterns": ["*_test.cpp", "*Test.cpp", "test_*.cpp"],
                "executable_patterns": ["*_test", "*Test", "test_*"],
                "includes": ["cppunit/TestCase.h"],
            },
            "boost_test": {
                "patterns": ["*_test.cpp", "*Test.cpp", "test_*.cpp"],
                "executable_patterns": ["*_test", "*Test", "test_*"],
                "includes": ["boost/test/unit_test.hpp"],
            },
        }

    def discover_test_files(self, project_root: Path) -> dict[str, list[Path]]:
        """Discover test files in the project."""
        discovered_tests = {framework: [] for framework in self.supported_frameworks}

        # Find all C++ source files
        cpp_files = []
        for pattern in ["**/*.cpp", "**/*.cc", "**/*.cxx"]:
            cpp_files.extend(project_root.rglob(pattern))

        # Analyze each file to determine testing framework
        for cpp_file in cpp_files:
            framework = self._detect_test_framework(cpp_file)
            if framework:
                discovered_tests[framework].append(cpp_file)

        # Remove empty entries
        return {k: v for k, v in discovered_tests.items() if v}

    def _detect_test_framework(self, file_path: Path) -> str | None:
        """Detect which testing framework a file uses."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        # Check include patterns
        for framework, config in self.supported_frameworks.items():
            for include in config["includes"]:
                if f'#include "{include}"' in content or f"#include <{include}>" in content:
                    return framework

        # Check for framework-specific patterns
        if "TEST(" in content or "TEST_F(" in content:
            return "gtest"
        elif "TEST_CASE(" in content or "SCENARIO(" in content:
            return "catch2"
        elif "DOCTEST_" in content:
            return "doctest"
        elif "CPPUNIT_" in content:
            return "cppunit"
        elif "BOOST_AUTO_TEST" in content:
            return "boost_test"

        return None

    def discover_test_executables(self, project_root: Path) -> list[Path]:
        """Discover test executables in the project."""
        executables = []

        # Look in common build directories
        build_dirs = [
            "build",
            "cmake-build-debug",
            "cmake-build-release",
            "bin",
            "Debug",
            "Release",
        ]

        for build_dir in build_dirs:
            build_path = project_root / build_dir
            if build_path.exists():
                # Find executable files
                for item in build_path.rglob("*"):
                    if item.is_file() and self._is_test_executable(item):
                        executables.append(item)

        return executables

    def _is_test_executable(self, file_path: Path) -> bool:
        """Check if a file is likely a test executable."""
        name = file_path.name.lower()

        # Check naming patterns
        test_patterns = ["test", "_test", "tests", "_tests"]
        if any(pattern in name for pattern in test_patterns):
            return True

        # Check if file is executable (Unix-like systems)
        try:
            return file_path.stat().st_mode & 0o111 != 0
        except Exception:
            return False

    def extract_test_entities(self, test_file: Path, framework: str) -> list[CoreCodeEntity]:
        """Extract test entities from a test file."""
        entities = []

        try:
            content = test_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return entities

        if framework == "gtest":
            entities.extend(self._extract_gtest_entities(test_file, content))
        elif framework == "catch2":
            entities.extend(self._extract_catch2_entities(test_file, content))
        elif framework == "doctest":
            entities.extend(self._extract_doctest_entities(test_file, content))
        elif framework == "cppunit":
            entities.extend(self._extract_cppunit_entities(test_file, content))
        elif framework == "boost_test":
            entities.extend(self._extract_boost_test_entities(test_file, content))

        return entities

    def _extract_gtest_entities(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Extract Google Test entities."""
        entities = []

        # Extract TEST and TEST_F macros
        test_patterns = [
            r"TEST\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            r"TEST_F\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            r"TEST_P\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
        ]

        for pattern in test_patterns:
            for match in re.finditer(pattern, content):
                test_suite = match.group(1).strip()
                test_name = match.group(2).strip()
                line_num = content[: match.start()].count("\n") + 1

                entities.append(
                    CoreCodeEntity(
                        id=f"gtest:{test_suite}:{test_name}:{file_path}",
                        name=f"{test_suite}.{test_name}",
                        type=EntityType.FUNCTION,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=line_num,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=match.group(0),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "test_framework": "gtest",
                            "test_suite": test_suite,
                            "test_name": test_name,
                            "is_test": True,
                        },
                    )
                )

        return entities

    def _extract_catch2_entities(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Extract Catch2 test entities."""
        entities = []

        # Extract TEST_CASE and SCENARIO macros
        test_patterns = [
            r'TEST_CASE\s*\(\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\)',
            r'SCENARIO\s*\(\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\)',
        ]

        for pattern in test_patterns:
            for match in re.finditer(pattern, content):
                test_name = match.group(1).strip()
                test_tags = match.group(2) if len(match.groups()) > 1 else ""
                line_num = content[: match.start()].count("\n") + 1

                entities.append(
                    CoreCodeEntity(
                        id=f"catch2:{test_name}:{file_path}",
                        name=test_name,
                        type=EntityType.FUNCTION,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=line_num,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=match.group(0),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "test_framework": "catch2",
                            "test_name": test_name,
                            "test_tags": test_tags,
                            "is_test": True,
                        },
                    )
                )

        return entities

    def _extract_doctest_entities(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Extract doctest entities."""
        entities = []

        # Extract TEST_CASE and similar macros
        test_patterns = [
            r'TEST_CASE\s*\(\s*"([^"]+)"\s*\)',
            r'DOCTEST_TEST_CASE\s*\(\s*"([^"]+)"\s*\)',
        ]

        for pattern in test_patterns:
            for match in re.finditer(pattern, content):
                test_name = match.group(1).strip()
                line_num = content[: match.start()].count("\n") + 1

                entities.append(
                    CoreCodeEntity(
                        id=f"doctest:{test_name}:{file_path}",
                        name=test_name,
                        type=EntityType.FUNCTION,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=line_num,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=match.group(0),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "test_framework": "doctest",
                            "test_name": test_name,
                            "is_test": True,
                        },
                    )
                )

        return entities

    def _extract_cppunit_entities(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Extract CppUnit test entities."""
        entities = []

        # Extract test methods (simplified)
        test_pattern = r"void\s+(test[A-Za-z0-9_]*)\s*\(\s*\)"

        for match in re.finditer(test_pattern, content):
            test_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"cppunit:{test_name}:{file_path}",
                    name=test_name,
                    type=EntityType.FUNCTION,
                    language="cpp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=match.group(0),
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "test_framework": "cppunit",
                        "test_name": test_name,
                        "is_test": True,
                    },
                )
            )

        return entities

    def _extract_boost_test_entities(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Extract Boost.Test entities."""
        entities = []

        # Extract BOOST_AUTO_TEST_CASE and similar macros
        test_patterns = [
            r"BOOST_AUTO_TEST_CASE\s*\(\s*([^)]+)\s*\)",
            r"BOOST_FIXTURE_TEST_CASE\s*\(\s*([^,]+)\s*,",
        ]

        for pattern in test_patterns:
            for match in re.finditer(pattern, content):
                test_name = match.group(1).strip()
                line_num = content[: match.start()].count("\n") + 1

                entities.append(
                    CoreCodeEntity(
                        id=f"boost_test:{test_name}:{file_path}",
                        name=test_name,
                        type=EntityType.FUNCTION,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=line_num,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=match.group(0),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "test_framework": "boost_test",
                            "test_name": test_name,
                            "is_test": True,
                        },
                    )
                )

        return entities

    def run_tests(self, test_executable: Path, output_format: str = "xml") -> dict[str, Any]:
        """Run a test executable and capture results."""
        results: dict[str, Any] = {
            "executable": str(test_executable),
            "success": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "test_results": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "execution_time": 0.0,
            },
        }

        try:
            # Prepare command based on framework
            cmd = [str(test_executable)]

            # Add output format flags
            if "gtest" in test_executable.name.lower():
                if output_format == "xml":
                    cmd.extend(["--gtest_output=xml:-"])
                elif output_format == "json":
                    cmd.extend(["--gtest_output=json:-"])
            elif "catch" in test_executable.name.lower():
                if output_format == "xml":
                    cmd.extend(["-r", "junit"])
                elif output_format == "json":
                    cmd.extend(["-r", "json"])

            # Execute the test
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=test_executable.parent,
            )

            results["exit_code"] = process.returncode
            results["stdout"] = process.stdout
            results["stderr"] = process.stderr
            results["success"] = process.returncode == 0

            # Parse test results
            if output_format == "xml" and process.stdout:
                results["test_results"] = self._parse_xml_results(process.stdout)
            elif output_format == "json" and process.stdout:
                results["test_results"] = self._parse_json_results(process.stdout)
            else:
                results["test_results"] = self._parse_text_results(process.stdout)

            # Calculate summary
            results["summary"] = self._calculate_test_summary(results["test_results"])

        except subprocess.TimeoutExpired:
            results["stderr"] = "Test execution timed out"
        except Exception as e:
            results["stderr"] = f"Test execution failed: {str(e)}"

        return results

    def _parse_xml_results(self, xml_output: str) -> list[dict[str, Any]]:
        """Parse XML test results."""
        test_results = []

        try:
            root = ET.fromstring(xml_output)  # nosec B314

            # Handle Google Test XML format
            for testcase in root.findall(".//testcase"):
                result = {
                    "name": testcase.get("name", ""),
                    "classname": testcase.get("classname", ""),
                    "time": float(testcase.get("time", 0)),
                    "status": "passed",
                    "message": "",
                }

                # Check for failures or errors
                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None:
                    result["status"] = "failed"
                    result["message"] = failure.get("message", "")
                elif error is not None:
                    result["status"] = "error"
                    result["message"] = error.get("message", "")

                test_results.append(result)

        except ET.ParseError:
            pass

        return test_results

    def _parse_json_results(self, json_output: str) -> list[dict[str, Any]]:
        """Parse JSON test results."""
        test_results = []

        try:
            data = json.loads(json_output)

            # Handle Google Test JSON format
            if "testsuites" in data:
                for testsuite in data["testsuites"]:
                    for testcase in testsuite.get("tests", []):
                        result = {
                            "name": testcase.get("name", ""),
                            "classname": testsuite.get("name", ""),
                            "time": testcase.get("time", 0),
                            "status": testcase.get("status", "").lower(),
                            "message": "",
                        }

                        if "failures" in testcase:
                            result["status"] = "failed"
                            result["message"] = testcase["failures"][0].get("failure", "")

                        test_results.append(result)

        except json.JSONDecodeError:
            pass

        return test_results

    def _parse_text_results(self, text_output: str) -> list[dict[str, Any]]:
        """Parse plain text test results."""
        test_results = []

        # Simple text parsing for common patterns
        lines = text_output.splitlines()

        for line in lines:
            # Google Test pattern
            if "[  PASSED  ]" in line or "[  FAILED  ]" in line:
                parts = line.split()
                if len(parts) >= 3:
                    status = "passed" if "PASSED" in line else "failed"
                    name = parts[2] if len(parts) > 2 else ""

                    test_results.append(
                        {
                            "name": name,
                            "classname": "",
                            "time": 0,
                            "status": status,
                            "message": "",
                        }
                    )

        return test_results

    def _calculate_test_summary(self, test_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate test execution summary."""
        summary = {
            "total_tests": len(test_results),
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "execution_time": 0.0,
        }

        for result in test_results:
            status = result.get("status", "").lower()
            if status == "passed":
                summary["passed_tests"] += 1
            elif status in ["failed", "error"]:
                summary["failed_tests"] += 1
            elif status == "skipped":
                summary["skipped_tests"] += 1

            summary["execution_time"] += result.get("time", 0)

        return summary

    def generate_test_report(self, test_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            "overview": {
                "total_executables": len(test_results),
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_skipped": 0,
                "total_time": 0.0,
                "success_rate": 0.0,
            },
            "details": test_results,
            "failed_tests": [],
            "slowest_tests": [],
        }

        # Aggregate statistics
        all_tests = []
        for executable_result in test_results:
            summary = executable_result.get("summary", {})
            report["overview"]["total_tests"] += summary.get("total_tests", 0)
            report["overview"]["total_passed"] += summary.get("passed_tests", 0)
            report["overview"]["total_failed"] += summary.get("failed_tests", 0)
            report["overview"]["total_skipped"] += summary.get("skipped_tests", 0)
            report["overview"]["total_time"] += summary.get("execution_time", 0)

            # Collect all individual test results
            all_tests.extend(executable_result.get("test_results", []))

        # Calculate success rate
        if report["overview"]["total_tests"] > 0:
            report["overview"]["success_rate"] = (
                report["overview"]["total_passed"] / report["overview"]["total_tests"] * 100
            )

        # Find failed tests
        report["failed_tests"] = [
            test for test in all_tests if test.get("status") in ["failed", "error"]
        ]

        # Find slowest tests
        report["slowest_tests"] = sorted(all_tests, key=lambda x: x.get("time", 0), reverse=True)[
            :10
        ]

        return report
