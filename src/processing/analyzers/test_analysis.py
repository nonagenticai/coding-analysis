"""
Test Result Analysis Module for Integration Tests.

This module provides functionality for analyzing test results, identifying
patterns and issues, and extracting insights from test execution data.

The module supports:
- Test result validation and verification
- Failure pattern analysis
- Performance trend analysis
- Test quality metrics calculation
- Regression detection
- Success/failure correlation analysis

Classes:
    TestResultAnalyzer: Analyzes individual test results
    TestPatternDetector: Detects patterns in test execution
    TestQualityAnalyzer: Calculates test quality metrics
    RegressionDetector: Identifies test regressions

Functions:
    analyze_test_results: Analyze a collection of test results
    detect_failure_patterns: Detect common failure patterns
    calculate_quality_metrics: Calculate test quality scores
"""

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .test_execution import TestExecutionResult, TestExecutionStatus

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of test failures."""

    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    ASSERTION_ERROR = "assertion_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class TestTrend(Enum):
    """Test performance trends."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class TestAnalysisResult:
    """Results from test analysis."""

    component: str
    test_name: str
    passed: bool
    execution_time: float
    failure_category: FailureCategory | None
    performance_score: float
    quality_score: float
    insights: list[str]
    recommendations: list[str]


@dataclass
class TestPatternMatch:
    """Represents a detected test pattern."""

    pattern_type: str
    description: str
    affected_tests: list[str]
    confidence: float
    recommendations: list[str]


@dataclass
class QualityMetrics:
    """Test quality metrics."""

    overall_score: float
    reliability_score: float
    performance_score: float
    coverage_score: float
    maintainability_score: float
    details: dict[str, Any]


class TestResultAnalyzer:
    """Analyzes individual test results for insights and quality assessment.

    This class examines test execution results to extract meaningful
    insights, identify issues, and provide recommendations for improvement.
    """

    def __init__(self):
        """Initialize the test result analyzer."""
        self.analysis_cache: dict[str, TestAnalysisResult] = {}

    def analyze_test_result(self, result: TestExecutionResult) -> TestAnalysisResult:
        """Analyze a single test execution result.

        Args:
            result: Test execution result to analyze

        Returns:
            TestAnalysisResult: Analysis of the test result
        """
        cache_key = f"{result.component}_{result.test_name}_{result.status}"

        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        # Determine if test passed
        passed = result.status == TestExecutionStatus.COMPLETED

        # Categorize failure if applicable
        failure_category = None
        if not passed:
            failure_category = self._categorize_failure(result)

        # Calculate performance score
        performance_score = self._calculate_performance_score(result)

        # Calculate quality score
        quality_score = self._calculate_quality_score(result)

        # Generate insights
        insights = self._generate_insights(result, performance_score, quality_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            result, failure_category, performance_score, quality_score
        )

        analysis = TestAnalysisResult(
            component=result.component,
            test_name=result.test_name,
            passed=passed,
            execution_time=result.execution_time,
            failure_category=failure_category,
            performance_score=performance_score,
            quality_score=quality_score,
            insights=insights,
            recommendations=recommendations,
        )

        self.analysis_cache[cache_key] = analysis
        return analysis

    def _categorize_failure(self, result: TestExecutionResult) -> FailureCategory:
        """Categorize the type of test failure.

        Args:
            result: Failed test execution result

        Returns:
            FailureCategory: Category of the failure
        """
        if not result.error_message:
            return FailureCategory.UNKNOWN

        error_msg = result.error_message.lower()

        # Check for timeout
        if "timeout" in error_msg or "timed out" in error_msg:
            return FailureCategory.TIMEOUT

        # Check for syntax errors
        if "syntax error" in error_msg or "syntaxerror" in error_msg:
            return FailureCategory.SYNTAX_ERROR

        # Check for import errors
        if "import error" in error_msg or "modulenotfounderror" in error_msg:
            return FailureCategory.IMPORT_ERROR

        # Check for assertion errors
        if "assertion" in error_msg or "assert" in error_msg:
            return FailureCategory.ASSERTION_ERROR

        # Check for resource errors
        if "memory" in error_msg or "disk" in error_msg or "file" in error_msg:
            return FailureCategory.RESOURCE_ERROR

        # Check for configuration errors
        if "config" in error_msg or "setting" in error_msg:
            return FailureCategory.CONFIGURATION_ERROR

        # Default to runtime error
        return FailureCategory.RUNTIME_ERROR

    def _calculate_performance_score(self, result: TestExecutionResult) -> float:
        """Calculate performance score for test result.

        Args:
            result: Test execution result

        Returns:
            float: Performance score (0.0 - 1.0)
        """
        if result.status != TestExecutionStatus.COMPLETED:
            return 0.0

        # Base score on execution time
        execution_time = result.execution_time

        # Define performance thresholds (in seconds)
        excellent_threshold = 1.0
        good_threshold = 5.0
        acceptable_threshold = 15.0
        poor_threshold = 30.0

        if execution_time <= excellent_threshold:
            return 1.0
        elif execution_time <= good_threshold:
            return 0.8
        elif execution_time <= acceptable_threshold:
            return 0.6
        elif execution_time <= poor_threshold:
            return 0.4
        else:
            return 0.2

    def _calculate_quality_score(self, result: TestExecutionResult) -> float:
        """Calculate quality score for test result.

        Args:
            result: Test execution result

        Returns:
            float: Quality score (0.0 - 1.0)
        """
        if result.status != TestExecutionStatus.COMPLETED:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Check if test has proper results structure
        if result.results:
            max_score += 0.3
            if isinstance(result.results, dict) and len(result.results) > 0:
                score += 0.3

        # Check for comprehensive test coverage
        if result.results and isinstance(result.results, dict):
            # Look for different test aspects
            test_aspects = ["single_file", "comprehensive", "basic", "advanced"]
            covered_aspects = sum(1 for aspect in test_aspects if aspect in result.results)

            max_score += 0.4
            aspect_score = min(covered_aspects / len(test_aspects), 1.0) * 0.4
            score += aspect_score

        # Check for success indicators in results
        if result.results and isinstance(result.results, dict):
            success_indicators = 0
            total_indicators = 0

            for key, value in result.results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        total_indicators += 1
                        if sub_key == "success" and sub_value:
                            success_indicators += 1
                        elif "has_" in sub_key and sub_value:
                            success_indicators += 1
                        elif (
                            sub_key.endswith("_count")
                            and isinstance(sub_value, int)
                            and sub_value > 0
                        ):
                            success_indicators += 1

            if total_indicators > 0:
                max_score += 0.3
                success_rate = success_indicators / total_indicators
                score += success_rate * 0.3

        return score / max_score if max_score > 0 else 0.0

    def _generate_insights(
        self,
        result: TestExecutionResult,
        performance_score: float,
        quality_score: float,
    ) -> list[str]:
        """Generate insights from test analysis.

        Args:
            result: Test execution result
            performance_score: Calculated performance score
            quality_score: Calculated quality score

        Returns:
            List of insight strings
        """
        insights = []

        # Performance insights
        if performance_score >= 0.8:
            insights.append("Excellent performance - test executes quickly")
        elif performance_score >= 0.6:
            insights.append("Good performance - acceptable execution time")
        elif performance_score >= 0.4:
            insights.append("Moderate performance - could be optimized")
        else:
            insights.append("Poor performance - significant optimization needed")

        # Quality insights
        if quality_score >= 0.8:
            insights.append("High quality test with comprehensive coverage")
        elif quality_score >= 0.6:
            insights.append("Good test quality with adequate coverage")
        elif quality_score >= 0.4:
            insights.append("Moderate test quality - room for improvement")
        else:
            insights.append("Low test quality - needs significant enhancement")

        # Component-specific insights
        if result.component == "call_graph":
            if result.results and isinstance(result.results, dict):
                if result.results.get("comprehensive", {}).get("found_orphaned"):
                    insights.append("Successfully detected orphaned functions")
                if result.results.get("comprehensive", {}).get("total_functions", 0) > 10:
                    insights.append("Analyzed complex codebase with many functions")

        elif result.component == "signature_analysis":
            if result.results and isinstance(result.results, dict):
                type_coverage = result.results.get("comprehensive", {}).get("type_coverage", 0.0)
                if type_coverage > 0.8:
                    insights.append("Excellent type annotation coverage")
                elif type_coverage > 0.5:
                    insights.append("Good type annotation coverage")

        elif result.component == "symbolic_validation":
            if result.results and isinstance(result.results, dict):
                safety_score = result.results.get("code_changes", {}).get("safety_score", 0)
                if safety_score > 80:
                    insights.append("High safety score - code is well-validated")
                elif safety_score > 60:
                    insights.append("Acceptable safety score")

        return insights

    def _generate_recommendations(
        self,
        result: TestExecutionResult,
        failure_category: FailureCategory | None,
        performance_score: float,
        quality_score: float,
    ) -> list[str]:
        """Generate recommendations for improvement.

        Args:
            result: Test execution result
            failure_category: Category of failure (if any)
            performance_score: Calculated performance score
            quality_score: Calculated quality score

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Failure-specific recommendations
        if failure_category:
            if failure_category == FailureCategory.TIMEOUT:
                recommendations.append("Increase timeout limits or optimize test performance")
            elif failure_category == FailureCategory.IMPORT_ERROR:
                recommendations.append("Check dependencies and import paths")
            elif failure_category == FailureCategory.SYNTAX_ERROR:
                recommendations.append("Review test code syntax and fix errors")
            elif failure_category == FailureCategory.RESOURCE_ERROR:
                recommendations.append("Check resource availability and limits")

        # Performance recommendations
        if performance_score < 0.6:
            recommendations.append("Optimize test execution time")
            if result.execution_time > 15.0:
                recommendations.append("Consider breaking down large tests into smaller units")

        # Quality recommendations
        if quality_score < 0.6:
            recommendations.append("Improve test coverage and validation")
            recommendations.append("Add more comprehensive test scenarios")

        # Component-specific recommendations
        if result.component == "performance":
            if result.results and isinstance(result.results, dict):
                if not result.results.get("acceptable_performance", True):
                    recommendations.append(
                        "Performance tests indicate bottlenecks - investigate slow components"
                    )

        return recommendations


class TestPatternDetector:
    """Detects patterns in test execution results.

    This class analyzes collections of test results to identify
    common patterns, recurring issues, and systematic problems.
    """

    def __init__(self):
        """Initialize the test pattern detector."""
        self.detected_patterns: list[TestPatternMatch] = []

    def detect_patterns(self, results: list[TestExecutionResult]) -> list[TestPatternMatch]:
        """Detect patterns in test execution results.

        Args:
            results: List of test execution results

        Returns:
            List of detected patterns
        """
        patterns = []

        # Detect failure patterns
        patterns.extend(self._detect_failure_patterns(results))

        # Detect performance patterns
        patterns.extend(self._detect_performance_patterns(results))

        # Detect component patterns
        patterns.extend(self._detect_component_patterns(results))

        self.detected_patterns = patterns
        return patterns

    def _detect_failure_patterns(
        self, results: list[TestExecutionResult]
    ) -> list[TestPatternMatch]:
        """Detect failure patterns in test results.

        Args:
            results: List of test execution results

        Returns:
            List of failure patterns
        """
        patterns = []

        # Group failures by error message patterns
        failure_groups = defaultdict(list)
        for result in results:
            if result.status == TestExecutionStatus.FAILED and result.error_message:
                # Extract error type from message
                error_pattern = self._extract_error_pattern(result.error_message)
                failure_groups[error_pattern].append(result.test_name)

        # Identify significant patterns
        total_tests = len(results)
        for pattern, affected_tests in failure_groups.items():
            if len(affected_tests) >= 2:  # At least 2 tests with same pattern
                confidence = len(affected_tests) / total_tests

                patterns.append(
                    TestPatternMatch(
                        pattern_type="failure_pattern",
                        description=f"Common failure pattern: {pattern}",
                        affected_tests=affected_tests,
                        confidence=confidence,
                        recommendations=[
                            f"Investigate common cause of {pattern} failures",
                            "Fix root cause to resolve multiple test failures",
                        ],
                    )
                )

        return patterns

    def _detect_performance_patterns(
        self, results: list[TestExecutionResult]
    ) -> list[TestPatternMatch]:
        """Detect performance patterns in test results.

        Args:
            results: List of test execution results

        Returns:
            List of performance patterns
        """
        patterns = []

        # Group by component and analyze performance
        component_times = defaultdict(list)
        for result in results:
            if result.status == TestExecutionStatus.COMPLETED:
                component_times[result.component].append(result.execution_time)

        for component, times in component_times.items():
            if len(times) >= 2:
                avg_time = statistics.mean(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0

                # Detect slow components
                if avg_time > 10.0:  # Slow threshold
                    patterns.append(
                        TestPatternMatch(
                            pattern_type="performance_pattern",
                            description=f"Slow performance in {component} component",
                            affected_tests=[f"{component}_tests"],
                            confidence=0.8,
                            recommendations=[
                                f"Optimize {component} component performance",
                                "Profile execution to identify bottlenecks",
                            ],
                        )
                    )

                # Detect high variance
                if std_dev > avg_time * 0.5:  # High variance
                    patterns.append(
                        TestPatternMatch(
                            pattern_type="performance_pattern",
                            description=f"Inconsistent performance in {component} component",
                            affected_tests=[f"{component}_tests"],
                            confidence=0.7,
                            recommendations=[
                                f"Investigate performance variability in {component}",
                                "Ensure consistent test conditions",
                            ],
                        )
                    )

        return patterns

    def _detect_component_patterns(
        self, results: list[TestExecutionResult]
    ) -> list[TestPatternMatch]:
        """Detect component-specific patterns.

        Args:
            results: List of test execution results

        Returns:
            List of component patterns
        """
        patterns = []

        # Group by component
        component_results = defaultdict(list)
        for result in results:
            component_results[result.component].append(result)

        for component, comp_results in component_results.items():
            # Calculate success rate
            total = len(comp_results)
            passed = sum(1 for r in comp_results if r.status == TestExecutionStatus.COMPLETED)
            success_rate = passed / total if total > 0 else 0

            # Detect unreliable components
            if success_rate < 0.7 and total >= 3:
                patterns.append(
                    TestPatternMatch(
                        pattern_type="reliability_pattern",
                        description=f"Low reliability in {component} component ({success_rate:.1%} success rate)",
                        affected_tests=[r.test_name for r in comp_results],
                        confidence=0.9,
                        recommendations=[
                            f"Investigate {component} component stability",
                            "Review component implementation for robustness",
                        ],
                    )
                )

        return patterns

    def _extract_error_pattern(self, error_message: str) -> str:
        """Extract error pattern from error message.

        Args:
            error_message: Full error message

        Returns:
            str: Simplified error pattern
        """
        # Common error patterns to extract
        patterns = [
            r"(\w+Error)",  # Exception types
            r"(timeout|timed out)",  # Timeout errors
            r"(No such file|FileNotFoundError)",  # File errors
            r"(Permission denied)",  # Permission errors
            r"(Memory|memory)",  # Memory errors
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1)

        # If no pattern matches, return first few words
        words = error_message.split()[:3]
        return " ".join(words)


class TestQualityAnalyzer:
    """Analyzes test quality metrics and provides quality assessment.

    This class evaluates the overall quality of test suites based on
    various metrics including reliability, performance, and coverage.
    """

    def __init__(self):
        """Initialize the test quality analyzer."""
        self.quality_history: list[QualityMetrics] = []

    def calculate_quality_metrics(self, results: list[TestExecutionResult]) -> QualityMetrics:
        """Calculate comprehensive quality metrics for test results.

        Args:
            results: List of test execution results

        Returns:
            QualityMetrics: Calculated quality metrics
        """
        if not results:
            return QualityMetrics(
                overall_score=0.0,
                reliability_score=0.0,
                performance_score=0.0,
                coverage_score=0.0,
                maintainability_score=0.0,
                details={},
            )

        # Calculate individual metrics
        reliability_score = self._calculate_reliability_score(results)
        performance_score = self._calculate_performance_score(results)
        coverage_score = self._calculate_coverage_score(results)
        maintainability_score = self._calculate_maintainability_score(results)

        # Calculate overall score (weighted average)
        overall_score = (
            reliability_score * 0.4
            + performance_score * 0.3
            + coverage_score * 0.2
            + maintainability_score * 0.1
        )

        # Compile details
        details = {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.status == TestExecutionStatus.COMPLETED),
            "failed_tests": sum(1 for r in results if r.status == TestExecutionStatus.FAILED),
            "avg_execution_time": statistics.mean([r.execution_time for r in results]),
            "components_tested": len(set(r.component for r in results)),
            "test_scenarios": len(set(r.test_name for r in results)),
        }

        metrics = QualityMetrics(
            overall_score=overall_score,
            reliability_score=reliability_score,
            performance_score=performance_score,
            coverage_score=coverage_score,
            maintainability_score=maintainability_score,
            details=details,
        )

        self.quality_history.append(metrics)
        return metrics

    def _calculate_reliability_score(self, results: list[TestExecutionResult]) -> float:
        """Calculate reliability score based on test success rate.

        Args:
            results: List of test execution results

        Returns:
            float: Reliability score (0.0 - 1.0)
        """
        if not results:
            return 0.0

        passed = sum(1 for r in results if r.status == TestExecutionStatus.COMPLETED)
        return passed / len(results)

    def _calculate_performance_score(self, results: list[TestExecutionResult]) -> float:
        """Calculate performance score based on execution times.

        Args:
            results: List of test execution results

        Returns:
            float: Performance score (0.0 - 1.0)
        """
        if not results:
            return 0.0

        completed_results = [r for r in results if r.status == TestExecutionStatus.COMPLETED]
        if not completed_results:
            return 0.0

        avg_time = statistics.mean([r.execution_time for r in completed_results])

        # Score based on average execution time
        if avg_time <= 2.0:
            return 1.0
        elif avg_time <= 5.0:
            return 0.8
        elif avg_time <= 10.0:
            return 0.6
        elif avg_time <= 20.0:
            return 0.4
        else:
            return 0.2

    def _calculate_coverage_score(self, results: list[TestExecutionResult]) -> float:
        """Calculate coverage score based on components and scenarios tested.

        Args:
            results: List of test execution results

        Returns:
            float: Coverage score (0.0 - 1.0)
        """
        if not results:
            return 0.0

        # Expected components for comprehensive testing
        expected_components = {
            "call_graph",
            "signature_analysis",
            "symbolic_validation",
            "documentation",
            "performance",
        }

        tested_components = set(r.component for r in results)
        component_coverage = len(tested_components & expected_components) / len(expected_components)

        # Test scenario diversity
        unique_scenarios = len(set(r.test_name for r in results))
        scenario_score = min(unique_scenarios / 10, 1.0)  # Normalize to max 10 scenarios

        return (component_coverage + scenario_score) / 2

    def _calculate_maintainability_score(self, results: list[TestExecutionResult]) -> float:
        """Calculate maintainability score based on test characteristics.

        Args:
            results: List of test execution results

        Returns:
            float: Maintainability score (0.0 - 1.0)
        """
        if not results:
            return 0.0

        score = 0.0

        # Check for consistent test structure
        components_with_results = 0
        total_components = len(set(r.component for r in results))

        for component in set(r.component for r in results):
            component_results = [r for r in results if r.component == component]
            has_structured_results = any(
                isinstance(r.results, dict) and len(r.results) > 0 for r in component_results
            )
            if has_structured_results:
                components_with_results += 1

        if total_components > 0:
            score += (components_with_results / total_components) * 0.6

        # Check for error handling
        failed_with_messages = sum(
            1 for r in results if r.status == TestExecutionStatus.FAILED and r.error_message
        )
        total_failed = sum(1 for r in results if r.status == TestExecutionStatus.FAILED)

        if total_failed > 0:
            error_handling_score = failed_with_messages / total_failed
            score += error_handling_score * 0.4
        else:
            score += 0.4  # No failures is good for maintainability

        return min(score, 1.0)


class RegressionDetector:
    """Detects regressions in test performance and quality.

    This class compares current test results with historical data
    to identify performance regressions and quality degradations.
    """

    def __init__(self):
        """Initialize the regression detector."""
        self.historical_data: list[dict[str, Any]] = []

    def detect_regressions(
        self,
        current_results: list[TestExecutionResult],
        historical_results: list[list[TestExecutionResult]] | None = None,
    ) -> dict[str, Any]:
        """Detect regressions in test results.

        Args:
            current_results: Current test execution results
            historical_results: Optional historical test results for comparison

        Returns:
            Dict containing regression analysis
        """
        regressions = {
            "performance_regressions": [],
            "quality_regressions": [],
            "new_failures": [],
            "overall_trend": TestTrend.STABLE,
        }

        if not historical_results or len(historical_results) == 0:
            return regressions

        # Compare with most recent historical data
        recent_historical = historical_results[-1]

        # Detect performance regressions
        perf_regressions = self._detect_performance_regressions(current_results, recent_historical)
        regressions["performance_regressions"] = perf_regressions

        # Detect quality regressions
        quality_regressions = self._detect_quality_regressions(current_results, recent_historical)
        regressions["quality_regressions"] = quality_regressions

        # Detect new failures
        new_failures = self._detect_new_failures(current_results, recent_historical)
        regressions["new_failures"] = new_failures

        # Determine overall trend
        regressions["overall_trend"] = self._determine_overall_trend(
            current_results, historical_results
        )

        return regressions

    def _detect_performance_regressions(
        self, current: list[TestExecutionResult], historical: list[TestExecutionResult]
    ) -> list[dict[str, Any]]:
        """Detect performance regressions between current and historical results.

        Args:
            current: Current test results
            historical: Historical test results

        Returns:
            List of detected performance regressions
        """
        regressions = []

        # Create lookup for historical times by test
        historical_times = {}
        for result in historical:
            if result.status == TestExecutionStatus.COMPLETED:
                key = f"{result.component}_{result.test_name}"
                historical_times[key] = result.execution_time

        # Compare current times with historical
        for result in current:
            if result.status == TestExecutionStatus.COMPLETED:
                key = f"{result.component}_{result.test_name}"
                if key in historical_times:
                    historical_time = historical_times[key]
                    current_time = result.execution_time

                    # Check for significant slowdown (>50% increase)
                    if current_time > historical_time * 1.5:
                        regression_factor = current_time / historical_time
                        regressions.append(
                            {
                                "test": key,
                                "component": result.component,
                                "historical_time": historical_time,
                                "current_time": current_time,
                                "regression_factor": regression_factor,
                                "severity": ("high" if regression_factor > 2.0 else "medium"),
                            }
                        )

        return regressions

    def _detect_quality_regressions(
        self, current: list[TestExecutionResult], historical: list[TestExecutionResult]
    ) -> list[dict[str, Any]]:
        """Detect quality regressions between current and historical results.

        Args:
            current: Current test results
            historical: Historical test results

        Returns:
            List of detected quality regressions
        """
        regressions = []

        # Calculate success rates by component
        current_success_rates = self._calculate_success_rates_by_component(current)
        historical_success_rates = self._calculate_success_rates_by_component(historical)

        for component in current_success_rates:
            if component in historical_success_rates:
                current_rate = current_success_rates[component]
                historical_rate = historical_success_rates[component]

                # Check for significant drop in success rate (>20% decrease)
                if current_rate < historical_rate - 0.2:
                    regressions.append(
                        {
                            "component": component,
                            "historical_success_rate": historical_rate,
                            "current_success_rate": current_rate,
                            "degradation": historical_rate - current_rate,
                            "severity": (
                                "high" if (historical_rate - current_rate) > 0.4 else "medium"
                            ),
                        }
                    )

        return regressions

    def _detect_new_failures(
        self, current: list[TestExecutionResult], historical: list[TestExecutionResult]
    ) -> list[dict[str, Any]]:
        """Detect new test failures not present in historical data.

        Args:
            current: Current test results
            historical: Historical test results

        Returns:
            List of new failures
        """
        # Get passed tests from historical data
        historical_passed = set()
        for result in historical:
            if result.status == TestExecutionStatus.COMPLETED:
                key = f"{result.component}_{result.test_name}"
                historical_passed.add(key)

        # Find current failures that were previously passing
        new_failures = []
        for result in current:
            if result.status == TestExecutionStatus.FAILED:
                key = f"{result.component}_{result.test_name}"
                if key in historical_passed:
                    new_failures.append(
                        {
                            "test": key,
                            "component": result.component,
                            "test_name": result.test_name,
                            "error_message": result.error_message,
                        }
                    )

        return new_failures

    def _determine_overall_trend(
        self,
        current: list[TestExecutionResult],
        historical: list[list[TestExecutionResult]],
    ) -> TestTrend:
        """Determine overall trend across multiple historical periods.

        Args:
            current: Current test results
            historical: List of historical test result sets

        Returns:
            TestTrend: Overall trend assessment
        """
        if len(historical) < 2:
            return TestTrend.STABLE

        # Calculate success rates over time
        success_rates = []
        for results in historical:
            total = len(results)
            passed = sum(1 for r in results if r.status == TestExecutionStatus.COMPLETED)
            success_rates.append(passed / total if total > 0 else 0)

        # Add current success rate
        current_total = len(current)
        current_passed = sum(1 for r in current if r.status == TestExecutionStatus.COMPLETED)
        current_rate = current_passed / current_total if current_total > 0 else 0
        success_rates.append(current_rate)

        # Analyze trend
        if len(success_rates) >= 3:
            # Check for consistent improvement or degradation
            improvements = 0
            degradations = 0

            for i in range(1, len(success_rates)):
                if success_rates[i] > success_rates[i - 1] + 0.05:  # 5% threshold
                    improvements += 1
                elif success_rates[i] < success_rates[i - 1] - 0.05:
                    degradations += 1

            if improvements > degradations and improvements >= 2:
                return TestTrend.IMPROVING
            elif degradations > improvements and degradations >= 2:
                return TestTrend.DEGRADING
            elif abs(max(success_rates) - min(success_rates)) > 0.3:
                return TestTrend.VOLATILE

        return TestTrend.STABLE

    def _calculate_success_rates_by_component(
        self, results: list[TestExecutionResult]
    ) -> dict[str, float]:
        """Calculate success rates by component.

        Args:
            results: List of test execution results

        Returns:
            Dict mapping component names to success rates
        """
        component_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})

        for result in results:
            component_stats[result.component]["total"] += 1
            if result.status == TestExecutionStatus.COMPLETED:
                component_stats[result.component]["passed"] += 1

        success_rates = {}
        for component, stats in component_stats.items():
            success_rates[component] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0

        return success_rates


# Convenience functions
def analyze_test_results(
    results: list[TestExecutionResult],
) -> list[TestAnalysisResult]:
    """Analyze a collection of test results.

    Args:
        results: List of test execution results

    Returns:
        List of test analysis results
    """
    analyzer = TestResultAnalyzer()
    return [analyzer.analyze_test_result(result) for result in results]


def detect_failure_patterns(
    results: list[TestExecutionResult],
) -> list[TestPatternMatch]:
    """Detect common failure patterns in test results.

    Args:
        results: List of test execution results

    Returns:
        List of detected failure patterns
    """
    detector = TestPatternDetector()
    return detector.detect_patterns(results)


def calculate_quality_metrics(results: list[TestExecutionResult]) -> QualityMetrics:
    """Calculate test quality metrics.

    Args:
        results: List of test execution results

    Returns:
        QualityMetrics: Calculated quality metrics
    """
    analyzer = TestQualityAnalyzer()
    return analyzer.calculate_quality_metrics(results)
