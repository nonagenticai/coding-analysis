"""
Go test execution and results module.

Provides functionality for executing Go tests, parsing test results,
and analyzing test coverage. Handles both unit tests and benchmark tests.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GoTestResult:
    """Result of a Go test execution."""

    test_name: str
    package: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    output: str
    error_message: str | None = None


@dataclass
class GoBenchmarkResult:
    """Result of a Go benchmark execution."""

    benchmark_name: str
    package: str
    iterations: int
    ns_per_op: float
    mb_per_sec: float | None = None
    allocs_per_op: int | None = None


@dataclass
class GoTestSuite:
    """Complete test suite results."""

    package: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    coverage_percentage: float | None = None
    test_results: list[GoTestResult] | None = None
    benchmark_results: list[GoBenchmarkResult] | None = None


class GoTestExecutor:
    """Go test execution and analysis functionality."""

    def __init__(self):
        """Initialize Go test executor."""
        self.go_binary = "go"

    async def discover_test_files(self, project_root: Path) -> list[Path]:
        """Discover Go test files in the project.

        Args:
            project_root: Root directory to search for test files

        Returns:
            List of test file paths
        """
        test_files = []

        try:
            # Find all *_test.go files
            for test_file in project_root.rglob("*_test.go"):
                if test_file.is_file():
                    test_files.append(test_file)
        except Exception as e:
            logger.warning("Error discovering test files in %s: %s", project_root, e)

        logger.debug("Found %d test files in %s", len(test_files), project_root)
        return test_files

    async def run_tests(
        self,
        package_path: Path,
        verbose: bool = False,
        coverage: bool = False,
        timeout: int = 300,
    ) -> GoTestSuite:
        """Run Go tests for a specific package.

        Args:
            package_path: Path to the Go package
            verbose: Enable verbose test output
            coverage: Enable coverage analysis
            timeout: Test execution timeout in seconds

        Returns:
            GoTestSuite with test results
        """
        logger.debug("Running Go tests for package: %s", package_path)

        # Build test command
        cmd = [self.go_binary, "test"]

        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend(["-cover", "-coverprofile=coverage.out"])

        # Add JSON output for easier parsing
        cmd.extend(["-json", "."])

        try:
            # Execute tests
            result = await self._execute_command(cmd, package_path, timeout)

            if result.returncode == 0:
                # Parse test results
                test_suite = self._parse_test_output(result.stdout, str(package_path))

                # Parse coverage if enabled
                if coverage:
                    coverage_percent = await self._parse_coverage(package_path)
                    test_suite.coverage_percentage = coverage_percent

                return test_suite
            else:
                logger.error("Test execution failed for %s: %s", package_path, result.stderr)
                return GoTestSuite(
                    package=str(package_path),
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    skipped_tests=0,
                    total_duration=0.0,
                    test_results=[],
                )

        except Exception as e:
            logger.error("Error running tests for %s: %s", package_path, e)
            return GoTestSuite(
                package=str(package_path),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                total_duration=0.0,
                test_results=[],
            )

    async def run_benchmarks(
        self, package_path: Path, benchmark_pattern: str = ".*", timeout: int = 600
    ) -> list[GoBenchmarkResult]:
        """Run Go benchmarks for a specific package.

        Args:
            package_path: Path to the Go package
            benchmark_pattern: Regex pattern to match benchmark names
            timeout: Benchmark execution timeout in seconds

        Returns:
            List of benchmark results
        """
        logger.debug("Running Go benchmarks for package: %s", package_path)

        cmd = [
            self.go_binary,
            "test",
            "-bench",
            benchmark_pattern,
            "-benchmem",  # Include memory allocation stats
            "-run",
            "^$",  # Don't run regular tests
            ".",
        ]

        try:
            result = await self._execute_command(cmd, package_path, timeout)

            if result.returncode == 0:
                return self._parse_benchmark_output(result.stdout, str(package_path))
            else:
                logger.error("Benchmark execution failed for %s: %s", package_path, result.stderr)
                return []

        except Exception as e:
            logger.error("Error running benchmarks for %s: %s", package_path, e)
            return []

    async def analyze_test_coverage(self, package_path: Path) -> dict[str, Any]:
        """Analyze test coverage for a Go package.

        Args:
            package_path: Path to the Go package

        Returns:
            Dictionary with coverage analysis results
        """
        logger.debug("Analyzing test coverage for package: %s", package_path)

        # Run tests with coverage
        cmd = [
            self.go_binary,
            "test",
            "-cover",
            "-coverprofile=coverage.out",
            "-covermode=count",
            ".",
        ]

        try:
            result = await self._execute_command(cmd, package_path)

            if result.returncode == 0:
                # Parse coverage profile
                coverage_data = await self._parse_coverage_profile(package_path / "coverage.out")

                # Generate coverage report
                coverage_report = await self._generate_coverage_report(package_path)

                return {
                    "overall_percentage": coverage_data.get("overall_percentage", 0.0),
                    "file_coverage": coverage_data.get("file_coverage", {}),
                    "uncovered_lines": coverage_data.get("uncovered_lines", []),
                    "coverage_report": coverage_report,
                }
            else:
                logger.error("Coverage analysis failed for %s: %s", package_path, result.stderr)
                return {"overall_percentage": 0.0, "file_coverage": {}}

        except Exception as e:
            logger.error("Error analyzing coverage for %s: %s", package_path, e)
            return {"overall_percentage": 0.0, "file_coverage": {}}

    async def _execute_command(
        self, cmd: list[str], cwd: Path, timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """Execute a command in the specified directory.

        Args:
            cmd: Command and arguments to execute
            cwd: Working directory
            timeout: Execution timeout in seconds

        Returns:
            CompletedProcess result
        """
        logger.debug("Executing command: %s in %s", " ".join(cmd), cwd)

        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)

    def _parse_test_output(self, output: str, package: str) -> GoTestSuite:
        """Parse Go test JSON output.

        Args:
            output: Test output string
            package: Package name

        Returns:
            GoTestSuite with parsed results
        """
        test_results = []
        total_duration = 0.0
        passed = 0
        failed = 0
        skipped = 0

        try:
            # Parse JSON lines
            for line in output.strip().split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)

                        if data.get("Action") == "pass" and "Test" in data:
                            test_name = data["Test"]
                            duration = data.get("Elapsed", 0.0)

                            test_results.append(
                                GoTestResult(
                                    test_name=test_name,
                                    package=package,
                                    status="PASS",
                                    duration=duration,
                                    output=data.get("Output", ""),
                                )
                            )
                            passed += 1
                            total_duration += duration

                        elif data.get("Action") == "fail" and "Test" in data:
                            test_name = data["Test"]
                            duration = data.get("Elapsed", 0.0)

                            test_results.append(
                                GoTestResult(
                                    test_name=test_name,
                                    package=package,
                                    status="FAIL",
                                    duration=duration,
                                    output=data.get("Output", ""),
                                    error_message=data.get("Output", ""),
                                )
                            )
                            failed += 1
                            total_duration += duration

                        elif data.get("Action") == "skip" and "Test" in data:
                            test_name = data["Test"]

                            test_results.append(
                                GoTestResult(
                                    test_name=test_name,
                                    package=package,
                                    status="SKIP",
                                    duration=0.0,
                                    output=data.get("Output", ""),
                                )
                            )
                            skipped += 1

                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue

        except Exception as e:
            logger.warning("Error parsing test output: %s", e)

        return GoTestSuite(
            package=package,
            total_tests=len(test_results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            total_duration=total_duration,
            test_results=test_results,
        )

    def _parse_benchmark_output(self, output: str, package: str) -> list[GoBenchmarkResult]:
        """Parse Go benchmark output.

        Args:
            output: Benchmark output string
            package: Package name

        Returns:
            List of benchmark results
        """
        benchmark_results = []

        # Benchmark result pattern
        # BenchmarkName-4    1000000    1234 ns/op    5678 B/op    9 allocs/op
        benchmark_pattern = r"Benchmark(\w+)-\d+\s+(\d+)\s+(\d+(?:\.\d+)?)\s+ns/op(?:\s+(\d+(?:\.\d+)?)\s+B/op)?(?:\s+(\d+)\s+allocs/op)?"

        for line in output.splitlines():
            match = re.match(benchmark_pattern, line.strip())
            if match:
                name = match.group(1)
                iterations = int(match.group(2))
                ns_per_op = float(match.group(3))
                bytes_per_op = float(match.group(4)) if match.group(4) else None
                allocs_per_op = int(match.group(5)) if match.group(5) else None

                # Calculate MB/sec if we have bytes per op
                mb_per_sec = None
                if bytes_per_op and ns_per_op > 0:
                    mb_per_sec = (bytes_per_op * 1000) / ns_per_op  # Convert to MB/sec

                benchmark_results.append(
                    GoBenchmarkResult(
                        benchmark_name=name,
                        package=package,
                        iterations=iterations,
                        ns_per_op=ns_per_op,
                        mb_per_sec=mb_per_sec,
                        allocs_per_op=allocs_per_op,
                    )
                )

        return benchmark_results

    async def _parse_coverage(self, package_path: Path) -> float | None:
        """Parse coverage percentage from coverage output.

        Args:
            package_path: Path to the package

        Returns:
            Coverage percentage if available
        """
        coverage_file = package_path / "coverage.out"

        if not coverage_file.exists():
            return None

        try:
            # Run go tool cover to get percentage
            cmd = [self.go_binary, "tool", "cover", "-func", str(coverage_file)]
            result = await self._execute_command(cmd, package_path)

            if result.returncode == 0:
                # Look for total coverage line
                for line in result.stdout.splitlines():
                    if line.startswith("total:"):
                        # Extract percentage
                        percentage_match = re.search(r"(\d+\.\d+)%", line)
                        if percentage_match:
                            return float(percentage_match.group(1))

        except Exception as e:
            logger.warning("Error parsing coverage: %s", e)

        return None

    async def _parse_coverage_profile(self, coverage_file: Path) -> dict[str, Any]:
        """Parse detailed coverage profile.

        Args:
            coverage_file: Path to coverage profile file

        Returns:
            Dictionary with detailed coverage data
        """
        if not coverage_file.exists():
            return {}

        try:
            content = coverage_file.read_text()
            file_coverage = {}
            uncovered_lines = []

            for line in content.splitlines():
                if line.startswith("mode:"):
                    continue

                # Parse coverage line: filename:start.column,end.column count
                parts = line.split()
                if len(parts) >= 2:
                    location = parts[0]
                    count = int(parts[1])

                    # Extract filename and line range
                    if ":" in location:
                        filename, range_part = location.rsplit(":", 1)

                        if filename not in file_coverage:
                            file_coverage[filename] = {
                                "covered_lines": 0,
                                "total_lines": 0,
                            }

                        # Parse line range
                        if "," in range_part:
                            start_part, end_part = range_part.split(",")
                            start_line = int(start_part.split(".")[0])
                            end_line = int(end_part.split(".")[0])

                            lines_in_range = end_line - start_line + 1
                            file_coverage[filename]["total_lines"] += lines_in_range

                            if count > 0:
                                file_coverage[filename]["covered_lines"] += lines_in_range
                            else:
                                # Add uncovered lines
                                for line_num in range(start_line, end_line + 1):
                                    uncovered_lines.append(f"{filename}:{line_num}")

            # Calculate overall percentage
            total_lines = sum(data["total_lines"] for data in file_coverage.values())
            covered_lines = sum(data["covered_lines"] for data in file_coverage.values())

            overall_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

            return {
                "overall_percentage": overall_percentage,
                "file_coverage": file_coverage,
                "uncovered_lines": uncovered_lines,
            }

        except Exception as e:
            logger.warning("Error parsing coverage profile: %s", e)
            return {}

    async def _generate_coverage_report(self, package_path: Path) -> str:
        """Generate human-readable coverage report.

        Args:
            package_path: Path to the package

        Returns:
            Coverage report string
        """
        coverage_file = package_path / "coverage.out"

        if not coverage_file.exists():
            return "No coverage data available"

        try:
            # Generate HTML coverage report
            cmd = [self.go_binary, "tool", "cover", "-html", str(coverage_file)]
            result = await self._execute_command(cmd, package_path)

            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error generating coverage report: {result.stderr}"

        except Exception as e:
            logger.warning("Error generating coverage report: %s", e)
            return f"Error generating coverage report: {e}"

    def get_test_statistics(self, test_suites: list[GoTestSuite]) -> dict[str, Any]:
        """Get aggregate test statistics from multiple test suites.

        Args:
            test_suites: List of test suite results

        Returns:
            Dictionary with aggregate statistics
        """
        total_tests = sum(suite.total_tests for suite in test_suites)
        total_passed = sum(suite.passed_tests for suite in test_suites)
        total_failed = sum(suite.failed_tests for suite in test_suites)
        total_skipped = sum(suite.skipped_tests for suite in test_suites)
        total_duration = sum(suite.total_duration for suite in test_suites)

        # Calculate average coverage
        coverage_values = [
            suite.coverage_percentage
            for suite in test_suites
            if suite.coverage_percentage is not None
        ]
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0

        return {
            "total_packages": len(test_suites),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "total_duration": total_duration,
            "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0.0,
            "average_coverage": avg_coverage,
            "packages_with_failures": len([s for s in test_suites if s.failed_tests > 0]),
        }
