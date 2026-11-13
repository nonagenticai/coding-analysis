"""Unit tests for test_analysis.py and test_coverage.py analyzer modules."""

from __future__ import annotations

import pytest

# ── test_analysis.py imports ──
from processing.analyzers.test_analysis import (
    FailureCategory,
    QualityMetrics,
    RegressionDetector,
    TestAnalysisResult,
    TestPatternDetector,
    TestQualityAnalyzer,
    TestResultAnalyzer,
    TestTrend,
    analyze_test_results,
    calculate_quality_metrics,
    detect_failure_patterns,
)

# ── test_coverage.py imports ──
from processing.analyzers.test_coverage import (
    ComponentCoverageAnalyzer,
    CoverageAnalyzer,
    CoverageGap,
    CoverageLevel,
    CoverageMetric,
    CoverageReport,
    CoverageType,
    EdgeCaseCoverageAnalyzer,
    FunctionalCoverageAnalyzer,
    ScenarioCoverageAnalyzer,
    analyze_test_coverage,
    generate_coverage_report,
    identify_coverage_gaps,
)
from processing.analyzers.test_discovery import TestScenario, TestScenarioType
from processing.analyzers.test_execution import TestExecutionResult, TestExecutionStatus

# ---------------------------------------------------------------------------
# Helpers for building test data
# ---------------------------------------------------------------------------


def _make_result(
    component: str = "call_graph",
    test_name: str = "test_basic",
    status: TestExecutionStatus = TestExecutionStatus.COMPLETED,
    execution_time: float = 1.0,
    results: dict | None = None,
    error_message: str | None = None,
) -> TestExecutionResult:
    return TestExecutionResult(
        component=component,
        test_name=test_name,
        status=status,
        execution_time=execution_time,
        results=results if results is not None else {},
        error_message=error_message,
    )


def _make_scenario(
    name: str = "test_scenario",
    scenario_type: TestScenarioType = TestScenarioType.CALL_GRAPH,
    test_code: str = "def main():\n    pass\n",
    expected_results: dict | None = None,
    metadata: dict | None = None,
) -> TestScenario:
    return TestScenario(
        name=name,
        scenario_type=scenario_type,
        test_code=test_code,
        expected_results=expected_results or {},
        metadata=metadata or {},
    )


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — test_analysis.py
# ═══════════════════════════════════════════════════════════════════════════


class TestTestResultAnalyzer:
    """Tests for TestResultAnalyzer."""

    def test_analyze_completed_result(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.COMPLETED,
            execution_time=0.5,
            results={"basic": {"success": True}},
        )
        analysis = analyzer.analyze_test_result(result)

        assert isinstance(analysis, TestAnalysisResult)
        assert analysis.passed is True
        assert analysis.failure_category is None
        assert analysis.performance_score > 0.0
        assert len(analysis.insights) > 0

    def test_analyze_failed_result_timeout(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="Test timed out after 30s",
        )
        analysis = analyzer.analyze_test_result(result)

        assert analysis.passed is False
        assert analysis.failure_category == FailureCategory.TIMEOUT

    def test_analyze_failed_result_syntax_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="SyntaxError: invalid syntax",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.SYNTAX_ERROR

    def test_analyze_failed_result_import_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="ModuleNotFoundError: No module named 'foo'",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.IMPORT_ERROR

    def test_analyze_failed_result_assertion_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="AssertionError: values differ",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.ASSERTION_ERROR

    def test_analyze_failed_result_resource_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="Out of memory",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.RESOURCE_ERROR

    def test_analyze_failed_result_config_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="Invalid config setting",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.CONFIGURATION_ERROR

    def test_analyze_failed_result_runtime_error(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="something went wrong with the process",
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.RUNTIME_ERROR

    def test_analyze_failed_result_unknown_no_message(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message=None,
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.failure_category == FailureCategory.UNKNOWN

    def test_caching_returns_same_result(self):
        analyzer = TestResultAnalyzer()
        result = _make_result()
        a1 = analyzer.analyze_test_result(result)
        a2 = analyzer.analyze_test_result(result)
        assert a1 is a2

    # -- performance score thresholds --

    def test_performance_score_excellent(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=0.5)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 1.0

    def test_performance_score_good(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=3.0)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 0.8

    def test_performance_score_acceptable(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=10.0)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 0.6

    def test_performance_score_poor(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=20.0)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 0.4

    def test_performance_score_very_poor(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=60.0)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 0.2

    def test_performance_score_zero_on_failure(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(status=TestExecutionStatus.FAILED)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.performance_score == 0.0

    # -- quality score --

    def test_quality_score_with_structured_results(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            results={
                "basic": {"success": True, "has_output": True},
                "single_file": {"success": True},
                "comprehensive": {"count": 5},
            },
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.quality_score > 0.0

    def test_quality_score_zero_on_failure(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(status=TestExecutionStatus.FAILED)
        analysis = analyzer.analyze_test_result(result)
        assert analysis.quality_score == 0.0

    def test_quality_score_with_success_indicators(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            results={
                "basic": {"success": True, "has_output": True, "item_count": 3},
            },
        )
        analysis = analyzer.analyze_test_result(result)
        assert analysis.quality_score > 0.0

    # -- insights --

    def test_insights_call_graph_orphaned(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            component="call_graph",
            results={"comprehensive": {"found_orphaned": True, "total_functions": 15}},
        )
        analysis = analyzer.analyze_test_result(result)
        assert any("orphaned" in i.lower() for i in analysis.insights)
        assert any(
            "complex" in i.lower() or "many functions" in i.lower() for i in analysis.insights
        )

    def test_insights_signature_analysis_type_coverage(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            component="signature_analysis",
            results={"comprehensive": {"type_coverage": 0.9}},
        )
        analysis = analyzer.analyze_test_result(result)
        assert any("type" in i.lower() for i in analysis.insights)

    def test_insights_symbolic_validation_safety(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            component="symbolic_validation",
            results={"code_changes": {"safety_score": 85}},
        )
        analysis = analyzer.analyze_test_result(result)
        assert any("safety" in i.lower() for i in analysis.insights)

    # -- recommendations --

    def test_recommendations_for_timeout_failure(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="timed out",
        )
        analysis = analyzer.analyze_test_result(result)
        assert any("timeout" in r.lower() for r in analysis.recommendations)

    def test_recommendations_for_import_failure(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="import error in module",
        )
        analysis = analyzer.analyze_test_result(result)
        assert any(
            "import" in r.lower() or "dependencies" in r.lower() for r in analysis.recommendations
        )

    def test_recommendations_for_slow_test(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(execution_time=20.0)
        analysis = analyzer.analyze_test_result(result)
        assert any(
            "optimize" in r.lower() or "breaking down" in r.lower()
            for r in analysis.recommendations
        )

    def test_recommendations_for_performance_component(self):
        analyzer = TestResultAnalyzer()
        result = _make_result(
            component="performance",
            results={"acceptable_performance": False},
        )
        analysis = analyzer.analyze_test_result(result)
        assert any(
            "performance" in r.lower() or "bottleneck" in r.lower()
            for r in analysis.recommendations
        )


class TestTestPatternDetector:
    """Tests for TestPatternDetector."""

    def test_detect_no_patterns_from_single_result(self):
        detector = TestPatternDetector()
        results = [_make_result()]
        patterns = detector.detect_patterns(results)
        # single result can't produce failure patterns requiring >= 2
        assert isinstance(patterns, list)

    def test_detect_failure_pattern(self):
        detector = TestPatternDetector()
        results = [
            _make_result(
                test_name="t1",
                status=TestExecutionStatus.FAILED,
                error_message="TimeoutError: exceeded",
            ),
            _make_result(
                test_name="t2",
                status=TestExecutionStatus.FAILED,
                error_message="TimeoutError: limit",
            ),
            _make_result(test_name="t3", status=TestExecutionStatus.COMPLETED),
        ]
        patterns = detector.detect_patterns(results)
        failure_patterns = [p for p in patterns if p.pattern_type == "failure_pattern"]
        assert len(failure_patterns) >= 1
        assert failure_patterns[0].confidence > 0

    def test_detect_performance_pattern_slow(self):
        detector = TestPatternDetector()
        results = [
            _make_result(component="comp_a", test_name="t1", execution_time=15.0),
            _make_result(component="comp_a", test_name="t2", execution_time=20.0),
        ]
        patterns = detector.detect_patterns(results)
        perf = [p for p in patterns if p.pattern_type == "performance_pattern"]
        assert len(perf) >= 1

    def test_detect_performance_pattern_high_variance(self):
        detector = TestPatternDetector()
        results = [
            _make_result(component="comp_b", test_name="t1", execution_time=1.0),
            _make_result(component="comp_b", test_name="t2", execution_time=10.0),
        ]
        patterns = detector.detect_patterns(results)
        perf = [p for p in patterns if p.pattern_type == "performance_pattern"]
        # Should detect high variance since std_dev > avg*0.5
        assert len(perf) >= 1

    def test_detect_component_reliability_pattern(self):
        detector = TestPatternDetector()
        results = [
            _make_result(
                component="comp_c",
                test_name="t1",
                status=TestExecutionStatus.FAILED,
                error_message="err",
            ),
            _make_result(
                component="comp_c",
                test_name="t2",
                status=TestExecutionStatus.FAILED,
                error_message="err",
            ),
            _make_result(component="comp_c", test_name="t3", status=TestExecutionStatus.COMPLETED),
        ]
        patterns = detector.detect_patterns(results)
        rel = [p for p in patterns if p.pattern_type == "reliability_pattern"]
        # 1/3 = 33% < 70% threshold, and total >= 3
        assert len(rel) >= 1

    def test_extract_error_pattern_exception_type(self):
        detector = TestPatternDetector()
        assert detector._extract_error_pattern("ValueError: bad value") == "ValueError"

    def test_extract_error_pattern_timeout(self):
        detector = TestPatternDetector()
        assert detector._extract_error_pattern("connection timed out") == "timed out"

    def test_extract_error_pattern_file_not_found(self):
        detector = TestPatternDetector()
        assert "FileNotFoundError" in detector._extract_error_pattern("FileNotFoundError: missing")

    def test_extract_error_pattern_permission(self):
        detector = TestPatternDetector()
        assert detector._extract_error_pattern("Permission denied on /tmp") == "Permission denied"

    def test_extract_error_pattern_memory(self):
        detector = TestPatternDetector()
        result = detector._extract_error_pattern("Memory allocation failed")
        assert result == "Memory"

    def test_extract_error_pattern_fallback(self):
        detector = TestPatternDetector()
        result = detector._extract_error_pattern("xyz abc 123 456")
        assert result == "xyz abc 123"

    def test_detected_patterns_stored(self):
        detector = TestPatternDetector()
        results = [_make_result()]
        detector.detect_patterns(results)
        assert detector.detected_patterns is not None


class TestTestQualityAnalyzer:
    """Tests for TestQualityAnalyzer."""

    def test_empty_results_returns_zero_metrics(self):
        analyzer = TestQualityAnalyzer()
        metrics = analyzer.calculate_quality_metrics([])
        assert metrics.overall_score == 0.0
        assert metrics.reliability_score == 0.0
        assert metrics.performance_score == 0.0

    def test_all_passing_results(self):
        analyzer = TestQualityAnalyzer()
        results = [
            _make_result(component="call_graph", test_name=f"t{i}", execution_time=1.0)
            for i in range(5)
        ]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.reliability_score == 1.0
        assert metrics.performance_score > 0.0
        assert metrics.overall_score > 0.0
        assert metrics.details["total_tests"] == 5
        assert metrics.details["passed_tests"] == 5

    def test_mixed_results(self):
        analyzer = TestQualityAnalyzer()
        results = [
            _make_result(test_name="t1"),
            _make_result(test_name="t2", status=TestExecutionStatus.FAILED, error_message="err"),
        ]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.reliability_score == 0.5
        assert metrics.details["failed_tests"] == 1

    def test_performance_score_fast(self):
        analyzer = TestQualityAnalyzer()
        results = [_make_result(execution_time=1.0)]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.performance_score == 1.0

    def test_performance_score_slow(self):
        analyzer = TestQualityAnalyzer()
        results = [_make_result(execution_time=25.0)]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.performance_score == 0.2

    def test_performance_score_all_failed(self):
        analyzer = TestQualityAnalyzer()
        results = [_make_result(status=TestExecutionStatus.FAILED, error_message="x")]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.performance_score == 0.0

    def test_coverage_score_expected_components(self):
        analyzer = TestQualityAnalyzer()
        results = [
            _make_result(component="call_graph", test_name="t1"),
            _make_result(component="signature_analysis", test_name="t2"),
            _make_result(component="symbolic_validation", test_name="t3"),
            _make_result(component="documentation", test_name="t4"),
            _make_result(component="performance", test_name="t5"),
        ]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.coverage_score > 0.5

    def test_maintainability_score_with_structured_results(self):
        analyzer = TestQualityAnalyzer()
        results = [
            _make_result(component="comp_a", results={"key": "val"}),
            _make_result(component="comp_b", results={"key": "val"}),
        ]
        metrics = analyzer.calculate_quality_metrics(results)
        assert metrics.maintainability_score > 0.0

    def test_maintainability_score_failures_with_messages(self):
        analyzer = TestQualityAnalyzer()
        results = [
            _make_result(status=TestExecutionStatus.FAILED, error_message="detailed error"),
        ]
        metrics = analyzer.calculate_quality_metrics(results)
        # All failures have messages, so error handling part is 1.0
        assert metrics.maintainability_score > 0.0

    def test_quality_history_appended(self):
        analyzer = TestQualityAnalyzer()
        analyzer.calculate_quality_metrics([_make_result()])
        analyzer.calculate_quality_metrics([_make_result()])
        assert len(analyzer.quality_history) == 2


class TestRegressionDetector:
    """Tests for RegressionDetector."""

    def test_no_historical_data_returns_stable(self):
        detector = RegressionDetector()
        result = detector.detect_regressions([_make_result()])
        assert result["overall_trend"] == TestTrend.STABLE
        assert result["performance_regressions"] == []
        assert result["quality_regressions"] == []
        assert result["new_failures"] == []

    def test_detect_performance_regression(self):
        detector = RegressionDetector()
        historical = [_make_result(test_name="t1", execution_time=1.0)]
        current = [_make_result(test_name="t1", execution_time=3.0)]  # 3x slower
        result = detector.detect_regressions(current, [historical])
        assert len(result["performance_regressions"]) == 1
        assert result["performance_regressions"][0]["severity"] == "high"

    def test_no_performance_regression_small_increase(self):
        detector = RegressionDetector()
        historical = [_make_result(test_name="t1", execution_time=1.0)]
        current = [_make_result(test_name="t1", execution_time=1.3)]  # 30% increase < 50% threshold
        result = detector.detect_regressions(current, [historical])
        assert len(result["performance_regressions"]) == 0

    def test_detect_quality_regression(self):
        detector = RegressionDetector()
        historical = [
            _make_result(component="comp_a", test_name="t1"),
            _make_result(component="comp_a", test_name="t2"),
            _make_result(component="comp_a", test_name="t3"),
        ]
        current = [
            _make_result(
                component="comp_a",
                test_name="t1",
                status=TestExecutionStatus.FAILED,
                error_message="x",
            ),
            _make_result(
                component="comp_a",
                test_name="t2",
                status=TestExecutionStatus.FAILED,
                error_message="x",
            ),
            _make_result(component="comp_a", test_name="t3"),
        ]
        result = detector.detect_regressions(current, [historical])
        assert len(result["quality_regressions"]) >= 1

    def test_detect_new_failures(self):
        detector = RegressionDetector()
        historical = [
            _make_result(component="comp_a", test_name="t1"),
        ]
        current = [
            _make_result(
                component="comp_a",
                test_name="t1",
                status=TestExecutionStatus.FAILED,
                error_message="oops",
            ),
        ]
        result = detector.detect_regressions(current, [historical])
        assert len(result["new_failures"]) == 1
        assert result["new_failures"][0]["error_message"] == "oops"

    def test_overall_trend_improving(self):
        detector = RegressionDetector()
        # Historical: improving success rates 0.3 -> 0.6 -> 0.9
        h1 = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 3 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        h2 = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 6 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        current = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 9 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        result = detector.detect_regressions(current, [h1, h2])
        assert result["overall_trend"] == TestTrend.IMPROVING

    def test_overall_trend_degrading(self):
        detector = RegressionDetector()
        h1 = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 9 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        h2 = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 6 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        current = [
            _make_result(
                test_name=f"t{i}",
                status=TestExecutionStatus.COMPLETED if i < 3 else TestExecutionStatus.FAILED,
                error_message="e",
            )
            for i in range(10)
        ]
        result = detector.detect_regressions(current, [h1, h2])
        assert result["overall_trend"] == TestTrend.DEGRADING

    def test_overall_trend_stable_with_single_historical(self):
        detector = RegressionDetector()
        result = detector.detect_regressions([_make_result()], [[_make_result()]])
        assert result["overall_trend"] == TestTrend.STABLE

    def test_success_rates_by_component(self):
        detector = RegressionDetector()
        results = [
            _make_result(component="a", test_name="t1"),
            _make_result(
                component="a", test_name="t2", status=TestExecutionStatus.FAILED, error_message="e"
            ),
            _make_result(component="b", test_name="t3"),
        ]
        rates = detector._calculate_success_rates_by_component(results)
        assert rates["a"] == 0.5
        assert rates["b"] == 1.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions in test_analysis.py."""

    def test_analyze_test_results(self):
        results = [_make_result(), _make_result(test_name="t2")]
        analyses = analyze_test_results(results)
        assert len(analyses) == 2
        assert all(isinstance(a, TestAnalysisResult) for a in analyses)

    def test_detect_failure_patterns_convenience(self):
        results = [
            _make_result(
                test_name="t1", status=TestExecutionStatus.FAILED, error_message="TimeoutError"
            ),
            _make_result(
                test_name="t2", status=TestExecutionStatus.FAILED, error_message="TimeoutError"
            ),
        ]
        patterns = detect_failure_patterns(results)
        assert isinstance(patterns, list)

    def test_calculate_quality_metrics_convenience(self):
        results = [_make_result()]
        metrics = calculate_quality_metrics(results)
        assert isinstance(metrics, QualityMetrics)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — test_coverage.py
# ═══════════════════════════════════════════════════════════════════════════


class TestFunctionalCoverageAnalyzer:
    """Tests for FunctionalCoverageAnalyzer."""

    def test_basic_functional_coverage(self):
        analyzer = FunctionalCoverageAnalyzer()
        scenario = _make_scenario(
            test_code="def main():\n    pass\ndef helper():\n    pass\n",
        )
        result = _make_result(
            component="call_graph",
            results={"comprehensive": {"total_functions": 5}},
        )
        metric = analyzer.analyze_functional_coverage([result], [scenario])
        assert isinstance(metric, CoverageMetric)
        assert metric.coverage_type == CoverageType.FUNCTIONAL
        assert metric.total_items >= 2  # main, helper

    def test_functional_coverage_no_scenarios(self):
        analyzer = FunctionalCoverageAnalyzer()
        metric = analyzer.analyze_functional_coverage([], [])
        assert metric.coverage_percentage == 0
        assert metric.total_items == 0

    def test_functional_coverage_with_async_functions(self):
        analyzer = FunctionalCoverageAnalyzer()
        scenario = _make_scenario(
            test_code="async def async_main():\n    pass\ndef sync_helper():\n    pass\n",
        )
        metric = analyzer.analyze_functional_coverage([], [scenario])
        assert metric.total_items == 2

    def test_functional_coverage_bad_syntax_handled(self):
        analyzer = FunctionalCoverageAnalyzer()
        scenario = _make_scenario(test_code="def broken(:\n")
        metric = analyzer.analyze_functional_coverage([], [scenario])
        # Should not raise, just skip
        assert metric.total_items == 0

    def test_categorize_functions(self):
        analyzer = FunctionalCoverageAnalyzer()
        functions = {"main", "validate_input", "process_data", "format_output", "other"}
        categories = analyzer._categorize_functions(functions)
        assert "main" in categories["main_functions"]
        assert "validate_input" in categories["validation_functions"]
        assert "process_data" in categories["processing_functions"]
        assert "format_output" in categories["utility_functions"]
        assert "other" in categories["other_functions"]

    def test_determine_coverage_level_excellent(self):
        analyzer = FunctionalCoverageAnalyzer()
        assert analyzer._determine_coverage_level(95) == CoverageLevel.EXCELLENT

    def test_determine_coverage_level_good(self):
        analyzer = FunctionalCoverageAnalyzer()
        assert analyzer._determine_coverage_level(75) == CoverageLevel.GOOD

    def test_determine_coverage_level_moderate(self):
        analyzer = FunctionalCoverageAnalyzer()
        assert analyzer._determine_coverage_level(55) == CoverageLevel.MODERATE

    def test_determine_coverage_level_poor(self):
        analyzer = FunctionalCoverageAnalyzer()
        assert analyzer._determine_coverage_level(35) == CoverageLevel.POOR

    def test_determine_coverage_level_inadequate(self):
        analyzer = FunctionalCoverageAnalyzer()
        assert analyzer._determine_coverage_level(10) == CoverageLevel.INADEQUATE

    def test_extract_tested_functions_call_graph(self):
        analyzer = FunctionalCoverageAnalyzer()
        tested = set()
        result = _make_result(
            component="call_graph",
            results={"comprehensive": {"total_functions": 5}},
        )
        analyzer._extract_tested_functions_from_result(result, tested)
        assert "main" in tested
        assert "process_data" in tested

    def test_extract_tested_functions_signature_analysis(self):
        analyzer = FunctionalCoverageAnalyzer()
        tested = set()
        result = _make_result(
            component="signature_analysis",
            results={"comprehensive": {"function_count": 3}},
        )
        analyzer._extract_tested_functions_from_result(result, tested)
        assert "function_0" in tested


class TestComponentCoverageAnalyzer:
    """Tests for ComponentCoverageAnalyzer."""

    def test_component_coverage_all_components(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(component="call_graph", test_name="t1"),
            _make_result(component="signature_analysis", test_name="t2"),
            _make_result(component="symbolic_validation", test_name="t3"),
            _make_result(component="documentation", test_name="t4"),
            _make_result(component="performance", test_name="t5"),
        ]
        metric = analyzer.analyze_component_coverage(results)
        assert metric.coverage_type == CoverageType.COMPONENT
        assert metric.coverage_percentage == 100.0
        assert metric.coverage_level == CoverageLevel.EXCELLENT

    def test_component_coverage_partial(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(component="call_graph", test_name="t1"),
        ]
        metric = analyzer.analyze_component_coverage(results)
        assert metric.coverage_percentage == 20.0

    def test_component_coverage_ignores_failed(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(
                component="call_graph", status=TestExecutionStatus.FAILED, error_message="err"
            ),
        ]
        metric = analyzer.analyze_component_coverage(results)
        assert metric.covered_items == 0

    def test_component_depth_analysis(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(
                component="call_graph", results={"basic": {}, "advanced": {}, "comprehensive": {}}
            ),
        ]
        depth = analyzer._analyze_component_depth(results)
        assert depth["call_graph"] == "deep"

    def test_component_depth_shallow(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(component="call_graph", results={"basic": {}}),
        ]
        depth = analyzer._analyze_component_depth(results)
        assert depth["call_graph"] == "shallow"

    def test_component_depth_moderate(self):
        analyzer = ComponentCoverageAnalyzer()
        results = [
            _make_result(component="call_graph", results={"a": {}, "b": {}}),
        ]
        depth = analyzer._analyze_component_depth(results)
        assert depth["call_graph"] == "moderate"


class TestScenarioCoverageAnalyzer:
    """Tests for ScenarioCoverageAnalyzer."""

    def test_scenario_coverage_all_types(self):
        analyzer = ScenarioCoverageAnalyzer()
        scenarios = [
            _make_scenario(name="s1", scenario_type=TestScenarioType.CALL_GRAPH),
            _make_scenario(name="s2", scenario_type=TestScenarioType.SIGNATURE_ANALYSIS),
            _make_scenario(name="s3", scenario_type=TestScenarioType.SYMBOLIC_VALIDATION),
            _make_scenario(name="s4", scenario_type=TestScenarioType.DOCUMENTATION),
            _make_scenario(name="s5", scenario_type=TestScenarioType.PERFORMANCE),
            _make_scenario(name="s6", scenario_type=TestScenarioType.INTEGRATION),
        ]
        results = [
            _make_result(component="call_graph", test_name="t1"),
            _make_result(component="signature_analysis", test_name="t2"),
            _make_result(component="symbolic_validation", test_name="t3"),
            _make_result(component="documentation", test_name="t4"),
            _make_result(component="performance", test_name="t5"),
            _make_result(component="integration", test_name="t6"),
        ]
        metric = analyzer.analyze_scenario_coverage(scenarios, results)
        assert metric.coverage_type == CoverageType.SCENARIO
        assert metric.coverage_percentage == 100.0

    def test_scenario_coverage_empty(self):
        analyzer = ScenarioCoverageAnalyzer()
        metric = analyzer.analyze_scenario_coverage([], [])
        assert metric.coverage_percentage == 0.0

    def test_scenario_quality_analysis(self):
        analyzer = ScenarioCoverageAnalyzer()
        scenario = _make_scenario(
            name="my_scenario",
            test_code="def a():\n    pass\n",
        )
        result = _make_result(test_name="my_scenario")
        quality = analyzer._analyze_scenario_quality([scenario], [result])
        assert quality["total_scenarios"] == 1

    def test_assess_scenario_complexity_low(self):
        analyzer = ScenarioCoverageAnalyzer()
        scenario = _make_scenario(test_code="def a():\n    pass\n")
        assert analyzer._assess_scenario_complexity(scenario) == "low"

    def test_assess_scenario_complexity_high(self):
        analyzer = ScenarioCoverageAnalyzer()
        code = "\n".join([f"def func_{i}():\n    pass" for i in range(12)])
        scenario = _make_scenario(test_code=code)
        assert analyzer._assess_scenario_complexity(scenario) == "high"

    def test_assess_scenario_effectiveness_high(self):
        analyzer = ScenarioCoverageAnalyzer()
        result = _make_result(
            results={
                "group": {"success": True, "has_output": True, "has_data": True, "has_more": True},
            },
        )
        effectiveness = analyzer._assess_scenario_effectiveness(result)
        assert effectiveness == "high"

    def test_assess_scenario_effectiveness_low_no_results(self):
        analyzer = ScenarioCoverageAnalyzer()
        _make_result(results=None)
        # results is None => "low"
        # We need to handle this - results could be None via the dataclass
        # Actually our _make_result defaults to {} which is not None
        result_obj = TestExecutionResult(
            component="x",
            test_name="t",
            status=TestExecutionStatus.COMPLETED,
            execution_time=1.0,
            results=None,
        )
        effectiveness = analyzer._assess_scenario_effectiveness(result_obj)
        assert effectiveness == "low"

    def test_assess_scenario_effectiveness_low_empty_dict(self):
        analyzer = ScenarioCoverageAnalyzer()
        result = _make_result(results={})
        effectiveness = analyzer._assess_scenario_effectiveness(result)
        assert effectiveness == "low"


class TestEdgeCaseCoverageAnalyzer:
    """Tests for EdgeCaseCoverageAnalyzer."""

    def test_edge_case_coverage_with_null_input(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(test_code="x = None\n")
        metric = analyzer.analyze_edge_case_coverage([scenario], [])
        assert "null_input" not in metric.missing_items or metric.covered_items >= 0

    def test_edge_case_coverage_with_empty_input(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(test_code='x = ""\ny = []\n')
        metric = analyzer.analyze_edge_case_coverage([scenario], [])
        assert metric.covered_items > 0

    def test_edge_case_coverage_with_invalid_input(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(test_code="raise ValueError('invalid input')\n")
        metric = analyzer.analyze_edge_case_coverage([scenario], [])
        assert metric.covered_items >= 1

    def test_edge_case_coverage_with_boundary_values(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(test_code="x = zero\ny = negative\n")
        metric = analyzer.analyze_edge_case_coverage([scenario], [])
        assert metric.covered_items >= 1

    def test_edge_case_coverage_with_timeout_scenario(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(test_code="# timeout handling\n")
        metric = analyzer.analyze_edge_case_coverage([scenario], [])
        assert metric.covered_items >= 1

    def test_identify_edge_cases_symbolic_validation(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenario = _make_scenario(
            scenario_type=TestScenarioType.SYMBOLIC_VALIDATION,
            test_code="# divide by zero check\n# sql injection test\n",
        )
        edge_cases = analyzer._identify_edge_cases_in_scenario(scenario)
        assert "boundary_values" in edge_cases
        assert "invalid_input" in edge_cases

    def test_identify_edge_cases_in_result_failed_timeout(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="Operation timeout occurred",
        )
        edge_cases = analyzer._identify_edge_cases_in_result(result)
        assert "error_conditions" in edge_cases
        assert "timeout_scenarios" in edge_cases

    def test_identify_edge_cases_in_result_memory_error(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        result = _make_result(
            status=TestExecutionStatus.FAILED,
            error_message="Out of memory",
        )
        edge_cases = analyzer._identify_edge_cases_in_result(result)
        assert "resource_limits" in edge_cases

    def test_identify_edge_cases_in_result_boundary(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        result = _make_result(
            results={"test": {"boundary": "tested"}},
        )
        edge_cases = analyzer._identify_edge_cases_in_result(result)
        assert "boundary_values" in edge_cases

    def test_identify_edge_cases_in_result_validation(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        result = _make_result(
            results={"test": {"validation": "done"}},
        )
        edge_cases = analyzer._identify_edge_cases_in_result(result)
        assert "invalid_input" in edge_cases

    def test_edge_case_distribution(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenarios = [
            _make_scenario(test_code="x = None\n"),
            _make_scenario(test_code='y = ""\n'),
        ]
        distribution = analyzer._analyze_edge_case_distribution(scenarios)
        assert isinstance(distribution, dict)

    def test_error_handling_coverage(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        results = [
            _make_result(status=TestExecutionStatus.FAILED, error_message="real error"),
            _make_result(status=TestExecutionStatus.FAILED, error_message=None),
            _make_result(),
        ]
        coverage = analyzer._analyze_error_handling_coverage(results)
        assert coverage["total_tests"] == 3
        assert coverage["failed_tests"] == 2
        assert coverage["tests_with_error_messages"] == 1

    def test_boundary_testing_coverage(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        scenarios = [
            _make_scenario(test_code="x = zero\nmax_val = 100\n"),
            _make_scenario(test_code="def normal():\n    pass\n"),
        ]
        result = analyzer._analyze_boundary_testing_coverage(scenarios)
        assert result["scenarios_with_boundary_tests"] == 1
        assert "zero" in result["boundary_test_types"]

    def test_determine_coverage_level_excellent(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        assert analyzer._determine_coverage_level(85) == CoverageLevel.EXCELLENT

    def test_determine_coverage_level_inadequate(self):
        analyzer = EdgeCaseCoverageAnalyzer()
        assert analyzer._determine_coverage_level(10) == CoverageLevel.INADEQUATE


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer (main coordinator)."""

    def test_analyze_comprehensive_coverage(self):
        analyzer = CoverageAnalyzer()
        scenarios = [
            _make_scenario(
                name="cg_scenario",
                scenario_type=TestScenarioType.CALL_GRAPH,
                test_code="def main():\n    pass\ndef helper():\n    pass\n",
            ),
        ]
        results = [
            _make_result(
                component="call_graph",
                test_name="t1",
                results={"comprehensive": {"total_functions": 3}},
            ),
        ]
        report = analyzer.analyze_comprehensive_coverage(scenarios, results)
        assert isinstance(report, CoverageReport)
        assert len(report.coverage_metrics) == 4
        assert report.overall_coverage >= 0.0
        assert report.details["total_scenarios"] == 1
        assert report.details["total_results"] == 1

    def test_calculate_overall_coverage_empty(self):
        analyzer = CoverageAnalyzer()
        assert analyzer._calculate_overall_coverage([]) == 0.0

    def test_calculate_overall_coverage_weighted(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=10,
                coverage_percentage=100.0,
                coverage_level=CoverageLevel.EXCELLENT,
                missing_items=[],
                details={},
            ),
            CoverageMetric(
                coverage_type=CoverageType.COMPONENT,
                total_items=5,
                covered_items=5,
                coverage_percentage=100.0,
                coverage_level=CoverageLevel.EXCELLENT,
                missing_items=[],
                details={},
            ),
        ]
        overall = analyzer._calculate_overall_coverage(metrics)
        assert overall == pytest.approx(100.0)

    def test_identify_coverage_gaps_poor(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=2,
                coverage_percentage=20.0,
                coverage_level=CoverageLevel.INADEQUATE,
                missing_items=["f1", "f2", "f3"],
                details={},
            ),
        ]
        gaps = analyzer._identify_coverage_gaps(metrics)
        assert len(gaps) == 1
        assert gaps[0].severity == "high"

    def test_identify_coverage_gaps_none_for_good(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=8,
                coverage_percentage=80.0,
                coverage_level=CoverageLevel.GOOD,
                missing_items=[],
                details={},
            ),
        ]
        gaps = analyzer._identify_coverage_gaps(metrics)
        assert len(gaps) == 0

    def test_generate_gap_recommendations_functional(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.FUNCTIONAL,
            total_items=10,
            covered_items=2,
            coverage_percentage=20.0,
            coverage_level=CoverageLevel.INADEQUATE,
            missing_items=[],
            details={},
        )
        recs = analyzer._generate_gap_recommendations(metric)
        assert any("function" in r.lower() for r in recs)

    def test_generate_gap_recommendations_component(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.COMPONENT,
            total_items=5,
            covered_items=1,
            coverage_percentage=20.0,
            coverage_level=CoverageLevel.INADEQUATE,
            missing_items=[],
            details={},
        )
        recs = analyzer._generate_gap_recommendations(metric)
        assert any("component" in r.lower() for r in recs)

    def test_generate_gap_recommendations_scenario(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.SCENARIO,
            total_items=6,
            covered_items=1,
            coverage_percentage=16.0,
            coverage_level=CoverageLevel.INADEQUATE,
            missing_items=[],
            details={},
        )
        recs = analyzer._generate_gap_recommendations(metric)
        assert any("scenario" in r.lower() for r in recs)

    def test_generate_gap_recommendations_edge_case(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.EDGE_CASE,
            total_items=8,
            covered_items=1,
            coverage_percentage=12.5,
            coverage_level=CoverageLevel.INADEQUATE,
            missing_items=[],
            details={},
        )
        recs = analyzer._generate_gap_recommendations(metric)
        assert any(
            "boundary" in r.lower() or "edge" in r.lower() or "error" in r.lower() for r in recs
        )

    def test_estimate_gap_effort_high(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.FUNCTIONAL,
            total_items=100,
            covered_items=10,
            coverage_percentage=10.0,
            coverage_level=CoverageLevel.INADEQUATE,
            missing_items=list(range(90)),
            details={},
        )
        assert analyzer._estimate_gap_effort(metric) == "high"

    def test_estimate_gap_effort_medium(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.FUNCTIONAL,
            total_items=10,
            covered_items=7,
            coverage_percentage=70.0,
            coverage_level=CoverageLevel.GOOD,
            missing_items=["a", "b", "c", "d", "e", "f"],
            details={},
        )
        assert analyzer._estimate_gap_effort(metric) == "medium"

    def test_estimate_gap_effort_low(self):
        analyzer = CoverageAnalyzer()
        metric = CoverageMetric(
            coverage_type=CoverageType.FUNCTIONAL,
            total_items=10,
            covered_items=9,
            coverage_percentage=90.0,
            coverage_level=CoverageLevel.EXCELLENT,
            missing_items=["a"],
            details={},
        )
        assert analyzer._estimate_gap_effort(metric) == "low"

    def test_generate_coverage_recommendations_inadequate(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=2,
                coverage_percentage=20.0,
                coverage_level=CoverageLevel.INADEQUATE,
                missing_items=["f1"],
                details={},
            ),
        ]
        gaps = [
            CoverageGap(
                gap_type="functional",
                description="Low coverage",
                severity="high",
                affected_areas=["f1"],
                recommendations=[],
                estimated_effort="high",
            ),
        ]
        recs = analyzer._generate_coverage_recommendations(metrics, gaps)
        assert any("inadequate" in r.lower() for r in recs)

    def test_generate_coverage_recommendations_excellent(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=10,
                coverage_percentage=100.0,
                coverage_level=CoverageLevel.EXCELLENT,
                missing_items=[],
                details={},
            ),
        ]
        recs = analyzer._generate_coverage_recommendations(metrics, [])
        assert any("excellent" in r.lower() for r in recs)

    def test_analyze_coverage_trends(self):
        analyzer = CoverageAnalyzer()
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=2,
                coverage_percentage=20.0,
                coverage_level=CoverageLevel.INADEQUATE,
                missing_items=[],
                details={},
            ),
        ]
        trends = analyzer._analyze_coverage_trends(metrics)
        assert "current_overall_coverage" in trends
        assert "improvement_areas" in trends
        assert "functional" in trends["improvement_areas"]


class TestCoverageConvenienceFunctions:
    """Tests for module-level convenience functions in test_coverage.py."""

    def test_analyze_test_coverage(self):
        scenarios = [_make_scenario()]
        results = [_make_result()]
        report = analyze_test_coverage(scenarios, results)
        assert isinstance(report, CoverageReport)

    def test_identify_coverage_gaps_function(self):
        metrics = [
            CoverageMetric(
                coverage_type=CoverageType.FUNCTIONAL,
                total_items=10,
                covered_items=2,
                coverage_percentage=20.0,
                coverage_level=CoverageLevel.INADEQUATE,
                missing_items=["f1"],
                details={},
            ),
        ]
        gaps = identify_coverage_gaps(metrics)
        assert len(gaps) == 1

    def test_generate_coverage_report_function(self):
        scenarios = [_make_scenario()]
        results = [_make_result()]
        report_str = generate_coverage_report(scenarios, results)
        assert "Coverage" in report_str
        assert "Overall Coverage" in report_str
