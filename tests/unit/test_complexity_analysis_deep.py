"""Unit tests for processing.architecture.complexity_analysis — comprehensive coverage.

Covers all public and private methods in ComplexityAnalysis:
  - run_radon_complexity (success, JSON parse error, timeout, FileNotFoundError)
  - calculate_cyclomatic_complexity (python, js/ts, unknown language)
  - _analyze_python_complexity (multiple functions, edge cases)
  - _analyze_js_complexity (function keyword, arrow, const arrow, braces)
  - _extract_js_function_name (all branches)
  - _calculate_complexity_metrics (empty, single, distribution buckets)
  - analyze_nesting_depth (python, js, unknown language)
  - _analyze_python_nesting (deep nesting, empty, long line truncation)
  - _analyze_js_nesting (deep nesting, negative depth guard)
  - calculate_maintainability_index (zero LOC, normal, edge clamps)
  - get_complexity_recommendations (all recommendation branches)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import processing.architecture.complexity_analysis as _ca_mod
from processing.architecture.complexity_analysis import ComplexityAnalysis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return ComplexityAnalysis()


# ---------------------------------------------------------------------------
# run_radon_complexity
# ---------------------------------------------------------------------------


class TestRunRadonComplexity:
    """Tests for run_radon_complexity."""

    def test_successful_run_high_complexity(self, analyzer):
        radon_output = json.dumps(
            {
                "src/foo.py": [
                    {
                        "name": "big_func",
                        "complexity": 25,
                        "lineno": 10,
                        "col_offset": 0,
                    }
                ],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].severity == "high"
        assert "big_func" in issues[0].message

    def test_successful_run_medium_complexity(self, analyzer):
        radon_output = json.dumps(
            {
                "src/bar.py": [
                    {
                        "name": "medium_func",
                        "complexity": 12,
                        "lineno": 5,
                    }
                ],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].severity == "medium"

    def test_below_threshold_filtered_out(self, analyzer):
        radon_output = json.dumps(
            {
                "src/simple.py": [
                    {
                        "name": "simple_func",
                        "complexity": 5,
                        "lineno": 1,
                    }
                ],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 0

    def test_empty_stdout(self, analyzer):
        proc = MagicMock(stdout="", returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is False
        assert len(issues) == 0

    def test_json_decode_error(self, analyzer):
        proc = MagicMock(stdout="NOT_VALID_JSON{{{", returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is False
        assert len(issues) == 0

    def test_timeout_expired(self, analyzer):
        with patch.object(
            _ca_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("radon", 60),
        ):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is False
        assert len(issues) == 0

    def test_file_not_found(self, analyzer):
        with patch.object(_ca_mod.subprocess, "run", side_effect=FileNotFoundError):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is False
        assert len(issues) == 0

    def test_multiple_files_and_functions(self, analyzer):
        radon_output = json.dumps(
            {
                "a.py": [
                    {"name": "f1", "complexity": 10, "lineno": 1, "col_offset": 0},
                    {"name": "f2", "complexity": 22, "lineno": 20, "col_offset": 4},
                ],
                "b.py": [
                    {"name": "f3", "complexity": 15, "lineno": 3},
                ],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 3
        severities = [i.severity for i in issues]
        assert "high" in severities
        assert "medium" in severities

    def test_complexity_exactly_10(self, analyzer):
        """Boundary: complexity == 10 should be included."""
        radon_output = json.dumps(
            {
                "x.py": [{"name": "boundary", "complexity": 10, "lineno": 1}],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].severity == "medium"

    def test_complexity_exactly_20(self, analyzer):
        """Boundary: complexity == 20 should be severity 'high'."""
        radon_output = json.dumps(
            {
                "x.py": [{"name": "edge", "complexity": 20, "lineno": 1}],
            }
        )
        proc = MagicMock(stdout=radon_output, returncode=0)
        with patch.object(_ca_mod.subprocess, "run", return_value=proc):
            issues, result = analyzer.run_radon_complexity(Path("/repo"))

        assert len(issues) == 1
        assert issues[0].severity == "high"


# ---------------------------------------------------------------------------
# calculate_cyclomatic_complexity
# ---------------------------------------------------------------------------


class TestCalculateCyclomaticComplexity:
    """Tests for calculate_cyclomatic_complexity dispatch."""

    def test_python_dispatch(self, analyzer):
        code = "def foo():\n    if x:\n        pass\n"
        result = analyzer.calculate_cyclomatic_complexity(code, "python")
        assert result["total_functions"] == 1

    def test_javascript_dispatch(self, analyzer):
        code = "function bar() {\n    if (x) {}\n}\n"
        result = analyzer.calculate_cyclomatic_complexity(code, "javascript")
        assert result["total_functions"] >= 1

    def test_typescript_dispatch(self, analyzer):
        code = "function baz() {\n    if (x) {}\n}\n"
        result = analyzer.calculate_cyclomatic_complexity(code, "typescript")
        assert result["total_functions"] >= 1

    def test_unknown_language_returns_defaults(self, analyzer):
        result = analyzer.calculate_cyclomatic_complexity("some code", "ruby")
        assert result["total_functions"] == 0
        assert result["avg_complexity"] == 0.0


# ---------------------------------------------------------------------------
# _analyze_python_complexity
# ---------------------------------------------------------------------------


class TestAnalyzePythonComplexity:
    """Tests for _analyze_python_complexity."""

    def test_empty_code(self, analyzer):
        result = analyzer._analyze_python_complexity("")
        assert result["total_functions"] == 0

    def test_single_simple_function(self, analyzer):
        code = "def simple():\n    return 1\n"
        result = analyzer._analyze_python_complexity(code)
        assert result["total_functions"] == 1
        assert result["functions"][0]["complexity"] == 1

    def test_function_with_branching(self, analyzer):
        code = (
            "def branchy():\n"
            "    if a:\n"
            "        pass\n"
            "    elif b:\n"
            "        pass\n"
            "    for x in y:\n"
            "        pass\n"
            "    while z:\n"
            "        pass\n"
        )
        result = analyzer._analyze_python_complexity(code)
        assert result["total_functions"] == 1
        assert result["functions"][0]["complexity"] == 5  # 1 base + 4 keywords

    def test_multiple_functions(self, analyzer):
        code = "def first():\n    if x:\n        pass\ndef second():\n    pass\n"
        result = analyzer._analyze_python_complexity(code)
        assert result["total_functions"] == 2
        assert result["functions"][0]["name"] == "first"
        assert result["functions"][0]["complexity"] == 2
        assert result["functions"][1]["name"] == "second"
        assert result["functions"][1]["complexity"] == 1

    def test_async_def_recognized(self, analyzer):
        code = "async def handler():\n    if cond:\n        pass\n"
        result = analyzer._analyze_python_complexity(code)
        assert result["total_functions"] == 1
        # The parser replaces "def " first, leaving "async handler"
        assert result["functions"][0]["name"] == "async handler"
        assert result["functions"][0]["complexity"] == 2

    def test_and_or_increase_complexity(self, analyzer):
        code = "def logic():\n    if a and b:\n        pass\n    if c or d:\n        pass\n"
        result = analyzer._analyze_python_complexity(code)
        # any() adds +1 per line, not per keyword: line 2 = +1, line 4 = +1; base = 1
        assert result["functions"][0]["complexity"] == 3

    def test_except_increases_complexity(self, analyzer):
        code = "def risky():\n    try:\n        pass\n    except ValueError:\n        pass\n"
        result = analyzer._analyze_python_complexity(code)
        assert result["functions"][0]["complexity"] == 2  # base + except

    def test_no_functions(self, analyzer):
        code = "x = 1\ny = 2\nif x:\n    pass\n"
        result = analyzer._analyze_python_complexity(code)
        assert result["total_functions"] == 0


# ---------------------------------------------------------------------------
# _analyze_js_complexity
# ---------------------------------------------------------------------------


class TestAnalyzeJsComplexity:
    """Tests for _analyze_js_complexity."""

    def test_function_keyword(self, analyzer):
        code = "function foo() {\n    if (x) {}\n}\n"
        result = analyzer._analyze_js_complexity(code)
        assert result["total_functions"] >= 1

    def test_arrow_function(self, analyzer):
        code = "const bar = () => {\n    if (x) {}\n}\n"
        result = analyzer._analyze_js_complexity(code)
        assert result["total_functions"] >= 1

    def test_multiple_complexity_keywords(self, analyzer):
        code = (
            "function complex() {\n"
            "    if (a) {}\n"
            "    else if (b) {}\n"
            "    while (c) {}\n"
            "    for (let i = 0; i < 10; i++) {}\n"
            "    switch (d) {}\n"
            "    try {} catch (e) {}\n"
            "    if (a && b) {}\n"
            "    if (a || b) {}\n"
            "}\n"
        )
        result = analyzer._analyze_js_complexity(code)
        assert result["total_functions"] >= 1
        func = result["functions"][0]
        assert func["complexity"] > 1

    def test_empty_code(self, analyzer):
        result = analyzer._analyze_js_complexity("")
        assert result["total_functions"] == 0

    def test_multiple_functions(self, analyzer):
        code = "function first() {\n    if (x) {}\n}\nfunction second() {\n    return 1;\n}\n"
        result = analyzer._analyze_js_complexity(code)
        assert result["total_functions"] == 2

    def test_anonymous_arrow(self, analyzer):
        code = "x = () => {\n    if (y) {}\n}\n"
        result = analyzer._analyze_js_complexity(code)
        assert result["total_functions"] >= 1


# ---------------------------------------------------------------------------
# _extract_js_function_name
# ---------------------------------------------------------------------------


class TestExtractJsFunctionName:
    """Tests for _extract_js_function_name."""

    def test_function_keyword(self, analyzer):
        assert analyzer._extract_js_function_name("function myFunc(a, b)") == "myFunc"

    def test_const_arrow(self, analyzer):
        assert analyzer._extract_js_function_name("const handler = (x) => {") == "handler"

    def test_arrow_without_const(self, analyzer):
        assert analyzer._extract_js_function_name("x = (a) => {") == "anonymous"

    def test_no_pattern(self, analyzer):
        assert analyzer._extract_js_function_name("let x = 5;") == "unknown"


# ---------------------------------------------------------------------------
# _calculate_complexity_metrics
# ---------------------------------------------------------------------------


class TestCalculateComplexityMetrics:
    """Tests for _calculate_complexity_metrics."""

    def test_empty_functions_list(self, analyzer):
        result = analyzer._calculate_complexity_metrics([])
        assert result["total_functions"] == 0
        assert result["avg_complexity"] == 0.0
        assert result["max_complexity"] == 0
        assert result["high_complexity_functions"] == 0

    def test_single_low_function(self, analyzer):
        result = analyzer._calculate_complexity_metrics([{"name": "f", "complexity": 3}])
        assert result["total_functions"] == 1
        assert result["avg_complexity"] == 3.0
        assert result["max_complexity"] == 3
        assert result["complexity_distribution"]["low"] == 1

    def test_distribution_buckets(self, analyzer):
        funcs = [
            {"name": "low", "complexity": 2},  # low (<=5)
            {"name": "med", "complexity": 8},  # medium (6-10)
            {"name": "high", "complexity": 13},  # high (11-15)
            {"name": "crit", "complexity": 20},  # critical (16+)
        ]
        result = analyzer._calculate_complexity_metrics(funcs)
        assert result["total_functions"] == 4
        assert result["complexity_distribution"]["low"] == 1
        assert result["complexity_distribution"]["medium"] == 1
        assert result["complexity_distribution"]["high"] == 1
        assert result["complexity_distribution"]["critical"] == 1
        assert result["high_complexity_functions"] == 2  # high + critical
        assert result["max_complexity"] == 20
        assert result["avg_complexity"] == round((2 + 8 + 13 + 20) / 4, 2)

    def test_boundary_values(self, analyzer):
        funcs = [
            {"name": "a", "complexity": 5},  # low boundary
            {"name": "b", "complexity": 6},  # medium boundary
            {"name": "c", "complexity": 10},  # medium boundary
            {"name": "d", "complexity": 11},  # high boundary
            {"name": "e", "complexity": 15},  # high boundary
            {"name": "f", "complexity": 16},  # critical boundary
        ]
        result = analyzer._calculate_complexity_metrics(funcs)
        assert result["complexity_distribution"]["low"] == 1
        assert result["complexity_distribution"]["medium"] == 2
        assert result["complexity_distribution"]["high"] == 2
        assert result["complexity_distribution"]["critical"] == 1

    def test_all_low(self, analyzer):
        funcs = [{"name": f"f{i}", "complexity": i} for i in range(1, 6)]
        result = analyzer._calculate_complexity_metrics(funcs)
        assert result["complexity_distribution"]["low"] == 5
        assert result["high_complexity_functions"] == 0

    def test_functions_list_in_result(self, analyzer):
        funcs = [{"name": "x", "complexity": 7}]
        result = analyzer._calculate_complexity_metrics(funcs)
        assert "functions" in result
        assert result["functions"] == funcs


# ---------------------------------------------------------------------------
# analyze_nesting_depth
# ---------------------------------------------------------------------------


class TestAnalyzeNestingDepth:
    """Tests for analyze_nesting_depth dispatch."""

    def test_python_dispatch(self, analyzer):
        code = "if True:\n    pass\n"
        result = analyzer.analyze_nesting_depth(code, "python")
        assert "max_depth" in result

    def test_javascript_dispatch(self, analyzer):
        code = "if (true) {\n    x;\n}\n"
        result = analyzer.analyze_nesting_depth(code, "javascript")
        assert "max_depth" in result

    def test_typescript_dispatch(self, analyzer):
        code = "if (true) {\n    x;\n}\n"
        result = analyzer.analyze_nesting_depth(code, "typescript")
        assert "max_depth" in result

    def test_unknown_language(self, analyzer):
        result = analyzer.analyze_nesting_depth("code", "go")
        assert result["max_depth"] == 0
        assert result["avg_depth"] == 0.0
        assert result["deep_nesting_lines"] == []


# ---------------------------------------------------------------------------
# _analyze_python_nesting
# ---------------------------------------------------------------------------


class TestAnalyzePythonNesting:
    """Tests for _analyze_python_nesting."""

    def test_empty_code(self, analyzer):
        result = analyzer._analyze_python_nesting("")
        assert result["max_depth"] == 0
        assert result["avg_depth"] == 0.0

    def test_flat_code(self, analyzer):
        code = "x = 1\ny = 2\nz = 3\n"
        result = analyzer._analyze_python_nesting(code)
        assert result["max_depth"] == 0
        assert result["deep_nesting_lines"] == []

    def test_shallow_nesting(self, analyzer):
        code = "def foo():\n    if True:\n        pass\n"
        result = analyzer._analyze_python_nesting(code)
        assert result["max_depth"] == 2
        assert result["deep_nesting_lines"] == []

    def test_deep_nesting_detected(self, analyzer):
        # depth 5 = 20 spaces
        code = "                    deeply_nested_line\n"
        result = analyzer._analyze_python_nesting(code)
        assert result["max_depth"] == 5
        assert len(result["deep_nesting_lines"]) == 1
        assert result["deep_nesting_lines"][0]["depth"] == 5
        assert result["deep_nesting_lines"][0]["line"] == 1

    def test_long_line_truncation(self, analyzer):
        long_content = "a" * 80
        code = " " * 20 + long_content + "\n"  # 20 spaces = depth 5
        result = analyzer._analyze_python_nesting(code)
        assert len(result["deep_nesting_lines"]) == 1
        entry = result["deep_nesting_lines"][0]
        assert entry["content"].endswith("...")
        assert len(entry["content"]) == 53  # 50 chars + "..."

    def test_short_deep_line_no_truncation(self, analyzer):
        code = " " * 20 + "short\n"  # depth 5
        result = analyzer._analyze_python_nesting(code)
        entry = result["deep_nesting_lines"][0]
        assert entry["content"] == "short"
        assert not entry["content"].endswith("...")

    def test_empty_lines_skipped(self, analyzer):
        code = "x = 1\n\n\ny = 2\n"
        result = analyzer._analyze_python_nesting(code)
        assert result["max_depth"] == 0
        assert result["avg_depth"] == 0.0


# ---------------------------------------------------------------------------
# _analyze_js_nesting
# ---------------------------------------------------------------------------


class TestAnalyzeJsNesting:
    """Tests for _analyze_js_nesting."""

    def test_empty_code(self, analyzer):
        result = analyzer._analyze_js_nesting("")
        assert result["max_depth"] == 0
        assert result["avg_depth"] == 0.0

    def test_flat_code(self, analyzer):
        code = "let x = 1;\nlet y = 2;\n"
        result = analyzer._analyze_js_nesting(code)
        assert result["max_depth"] == 0

    def test_brace_nesting(self, analyzer):
        code = "function foo() {\n    if (x) {\n        while (y) {\n        }\n    }\n}\n"
        result = analyzer._analyze_js_nesting(code)
        assert result["max_depth"] == 3

    def test_deep_nesting_detected(self, analyzer):
        code = (
            "if (a) {\n"
            "    if (b) {\n"
            "        if (c) {\n"
            "            if (d) {\n"
            "                if (e) {\n"
            "                    x;\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = analyzer._analyze_js_nesting(code)
        assert result["max_depth"] == 5
        assert len(result["deep_nesting_lines"]) > 0

    def test_negative_depth_guarded(self, analyzer):
        code = "}\n}\n}\n"
        result = analyzer._analyze_js_nesting(code)
        assert result["max_depth"] == 0

    def test_long_line_truncation(self, analyzer):
        long_line = "a" * 80
        code = "{\n{\n{\n{\n{\n" + long_line + "\n}\n}\n}\n}\n}\n"
        result = analyzer._analyze_js_nesting(code)
        deep_entries = [e for e in result["deep_nesting_lines"] if e["depth"] > 4]
        assert len(deep_entries) > 0
        # Find the entry with the long content (not the "{" brace lines)
        long_entries = [e for e in deep_entries if len(e["content"]) > 1]
        assert len(long_entries) > 0
        assert long_entries[0]["content"].endswith("...")

    def test_empty_lines_skipped(self, analyzer):
        code = "{\n\n\n}\n"
        result = analyzer._analyze_js_nesting(code)
        assert result["max_depth"] >= 0


# ---------------------------------------------------------------------------
# calculate_maintainability_index
# ---------------------------------------------------------------------------


class TestCalculateMaintainabilityIndex:
    """Tests for calculate_maintainability_index."""

    def test_zero_loc_returns_100(self, analyzer):
        assert analyzer.calculate_maintainability_index({}, 0) == 100.0

    def test_simple_low_complexity(self, analyzer):
        metrics = {"avg_complexity": 2, "max_complexity": 3}
        result = analyzer.calculate_maintainability_index(metrics, 50)
        # 100 - min(50, 4) - min(30, 3) - min(20, 0.5) = 92.5
        assert result == 92.5

    def test_high_complexity_clamped(self, analyzer):
        metrics = {"avg_complexity": 50, "max_complexity": 100}
        result = analyzer.calculate_maintainability_index(metrics, 5000)
        # 100 - 50 - 30 - 20 = 0
        assert result == 0.0

    def test_missing_keys_use_defaults(self, analyzer):
        result = analyzer.calculate_maintainability_index({}, 100)
        # avg=1, max=1: 100 - 2 - 1 - 1 = 96
        assert result == 96.0

    def test_moderate_values(self, analyzer):
        metrics = {"avg_complexity": 10, "max_complexity": 15}
        result = analyzer.calculate_maintainability_index(metrics, 200)
        # 100 - 20 - 15 - 2 = 63
        assert result == 63.0

    def test_result_never_below_zero(self, analyzer):
        metrics = {"avg_complexity": 100, "max_complexity": 100}
        result = analyzer.calculate_maintainability_index(metrics, 100000)
        assert result == 0.0


# ---------------------------------------------------------------------------
# get_complexity_recommendations
# ---------------------------------------------------------------------------


class TestGetComplexityRecommendations:
    """Tests for get_complexity_recommendations."""

    def test_no_recommendations_for_clean_code(self, analyzer):
        metrics = {
            "high_complexity_functions": 0,
            "max_complexity": 5,
            "avg_complexity": 3,
            "complexity_distribution": {"critical": 0},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert len(recs) == 0

    def test_max_complexity_over_20(self, analyzer):
        metrics = {
            "high_complexity_functions": 2,
            "max_complexity": 25,
            "avg_complexity": 5,
            "complexity_distribution": {"critical": 0},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert any("Refactor" in r for r in recs)

    def test_avg_complexity_over_8(self, analyzer):
        metrics = {
            "high_complexity_functions": 0,
            "max_complexity": 10,
            "avg_complexity": 12,
            "complexity_distribution": {"critical": 0},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert any("breaking down" in r for r in recs)

    def test_high_complexity_count_over_5(self, analyzer):
        metrics = {
            "high_complexity_functions": 8,
            "max_complexity": 10,
            "avg_complexity": 5,
            "complexity_distribution": {"critical": 0},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert any("architectural" in r for r in recs)

    def test_critical_count_above_0(self, analyzer):
        metrics = {
            "high_complexity_functions": 0,
            "max_complexity": 10,
            "avg_complexity": 5,
            "complexity_distribution": {"critical": 3},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert any("critically complex" in r for r in recs)
        assert any("3" in r for r in recs)

    def test_all_recommendations_triggered(self, analyzer):
        metrics = {
            "high_complexity_functions": 10,
            "max_complexity": 30,
            "avg_complexity": 15,
            "complexity_distribution": {"critical": 5},
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert len(recs) == 4

    def test_missing_keys_handled(self, analyzer):
        recs = analyzer.get_complexity_recommendations({})
        assert isinstance(recs, list)
        assert len(recs) == 0

    def test_missing_distribution(self, analyzer):
        metrics = {
            "high_complexity_functions": 0,
            "max_complexity": 5,
            "avg_complexity": 3,
        }
        recs = analyzer.get_complexity_recommendations(metrics)
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for ComplexityAnalysis initialization."""

    def test_thresholds_set(self):
        ca = ComplexityAnalysis()
        assert ca.complexity_thresholds["low"] == 5
        assert ca.complexity_thresholds["medium"] == 10
        assert ca.complexity_thresholds["high"] == 15
        assert ca.complexity_thresholds["critical"] == 20
