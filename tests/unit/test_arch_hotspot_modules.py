"""Unit tests for processing.architecture.hotspot modules.

Covers:
- hotspot_reporter.py (HotspotReport, HotspotReporter)
- bottleneck_detector.py (BottleneckMetrics, BottleneckDetector)
- change_frequency_analyzer.py (ChangeMetrics, GitAnalyzer)
- complexity_analyzer.py (ComplexityAnalyzer)
- hotspot_analyzer.py (HotspotAnalyzer)
- detection_algorithms.py (HotspotDetector)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from models.architecture_models import HotspotPrediction

# BottleneckDetector.__init__ calls GitAnalyzer(repo_path) which runs git;
# import the module but create instances with repo_path=None to avoid subprocess.
from processing.architecture.hotspot.bottleneck_detector import (
    BottleneckDetector,
    BottleneckMetrics,
)
from processing.architecture.hotspot.change_frequency_analyzer import (
    ChangeMetrics,
    GitAnalyzer,
)
from processing.architecture.hotspot.complexity_analyzer import ComplexityAnalyzer
from processing.architecture.hotspot.detection_algorithms import HotspotDetector

# HotspotAnalyzer imports BottleneckDetector/GitAnalyzer in __init__; we test
# it with repo_path=None so no git subprocess is triggered.
from processing.architecture.hotspot.hotspot_analyzer import HotspotAnalyzer
from processing.architecture.hotspot.hotspot_reporter import (
    HotspotReport,
    HotspotReporter,
)

# ===========================================================================
# Helpers / fixtures
# ===========================================================================


def _make_hotspot(
    component_id: str = "comp_a",
    score: float = 0.85,
    confidence: float = 0.8,
    predicted_issues: list[str] | None = None,
    risk_factors: list[str] | None = None,
    recommended_actions: list[str] | None = None,
) -> HotspotPrediction:
    return HotspotPrediction(
        component_id=component_id,
        score=score,
        confidence=confidence,
        predicted_issues=predicted_issues or ["bug-prone"],
        risk_factors=risk_factors or ["High complexity (80%)"],
        recommended_actions=recommended_actions or ["Refactor code"],
    )


def _make_graph(*nodes_with_data: tuple[str, dict], edges: list[tuple[str, str]] | None = None):
    """Build a small directed graph for testing."""
    g = nx.DiGraph()
    for nid, data in nodes_with_data:
        g.add_node(nid, **data)
    if edges:
        g.add_edges_from(edges)
    return g


# ===========================================================================
# detection_algorithms.py  (HotspotDetector)
# ===========================================================================


class TestHotspotDetector:
    def setup_method(self):
        self.detector = HotspotDetector()

    # -- calculate_hotspot_score --

    def test_score_all_zeros(self):
        features = {k: 0.0 for k in self.detector.feature_weights}
        assert self.detector.calculate_hotspot_score(features) == 0.0

    def test_score_all_ones(self):
        features = {k: 1.0 for k in self.detector.feature_weights}
        score = self.detector.calculate_hotspot_score(features)
        assert score == pytest.approx(1.0)

    def test_score_partial_features(self):
        features = {"complexity": 0.8, "change_frequency": 0.6}
        score = self.detector.calculate_hotspot_score(features)
        expected = 0.8 * 0.3 + 0.6 * 0.25
        assert score == pytest.approx(expected)

    def test_score_clamps_above_one(self):
        features = {k: 2.0 for k in self.detector.feature_weights}
        assert self.detector.calculate_hotspot_score(features) == 1.0

    # -- normalize_feature --

    def test_normalize_zero(self):
        result = self.detector.normalize_feature(0, 50)
        assert 0.0 <= result <= 1.0

    def test_normalize_at_max(self):
        result = self.detector.normalize_feature(50, 50)
        # sigmoid(4*(1 - 0.5)) = sigmoid(2) ~ 0.88
        assert result > 0.8

    def test_normalize_large_value(self):
        result = self.detector.normalize_feature(200, 50)
        assert result > 0.95

    # -- predict_issue_types --

    def test_predict_issues_high_bug(self):
        features = {"bug_frequency": 0.9}
        assert "bug-prone" in self.detector.predict_issue_types(features)

    def test_predict_issues_high_complexity(self):
        features = {"complexity": 0.8}
        assert "high-complexity" in self.detector.predict_issue_types(features)

    def test_predict_issues_high_change(self):
        features = {"change_frequency": 0.8}
        assert "unstable" in self.detector.predict_issue_types(features)

    def test_predict_issues_knowledge_siloed(self):
        features = {"author_count": 0.9}
        assert "knowledge-siloed" in self.detector.predict_issue_types(features)

    def test_predict_issues_high_churn(self):
        features = {"code_churn": 0.8}
        assert "high-churn" in self.detector.predict_issue_types(features)

    def test_predict_issues_nothing(self):
        features = {k: 0.1 for k in ["bug_frequency", "complexity", "change_frequency"]}
        assert self.detector.predict_issue_types(features) == []

    # -- identify_risk_factors --

    def test_risk_factors_high(self):
        features = {
            "bug_frequency": 0.9,
            "complexity": 0.8,
            "change_frequency": 0.7,
            "author_count": 0.6,
            "code_churn": 0.75,
        }
        factors = self.detector.identify_risk_factors(features)
        assert len(factors) == 5

    def test_risk_factors_none(self):
        features = {k: 0.1 for k in self.detector.feature_weights}
        assert self.detector.identify_risk_factors(features) == []

    # -- generate_recommendations --

    def test_recommendations_high_complexity(self):
        recs = self.detector.generate_recommendations({"complexity": 0.9}, 0.7)
        assert any("Refactor" in r for r in recs)

    def test_recommendations_high_bug(self):
        recs = self.detector.generate_recommendations({"bug_frequency": 0.9}, 0.7)
        assert any("test" in r.lower() for r in recs)

    def test_recommendations_high_score(self):
        recs = self.detector.generate_recommendations({}, 0.85)
        assert any("splitting" in r.lower() for r in recs)

    def test_recommendations_change_and_authors(self):
        features = {"change_frequency": 0.9, "author_count": 0.9}
        recs = self.detector.generate_recommendations(features, 0.5)
        assert any("architectural" in r.lower() for r in recs)

    def test_recommendations_author_only(self):
        recs = self.detector.generate_recommendations({"author_count": 0.9}, 0.5)
        assert any("Document" in r for r in recs)

    def test_recommendations_fallback(self):
        recs = self.detector.generate_recommendations({}, 0.3)
        assert recs == ["Monitor this file for future degradation"]

    # -- create_hotspot_prediction --

    def test_create_prediction(self):
        features = {"complexity": 0.9, "bug_frequency": 0.8}
        pred = self.detector.create_hotspot_prediction("comp1", 0.85, features)
        assert pred.component_id == "comp1"
        assert pred.score == 0.85
        assert pred.confidence == 0.85  # capped at score (< 0.95)
        assert len(pred.predicted_issues) > 0
        assert len(pred.risk_factors) > 0
        assert len(pred.recommended_actions) > 0

    def test_create_prediction_caps_confidence(self):
        features = {"complexity": 1.0}
        pred = self.detector.create_hotspot_prediction("c", 0.99, features)
        assert pred.confidence == 0.95

    # -- is_hotspot --

    def test_is_hotspot_above_threshold(self):
        assert self.detector.is_hotspot(0.6) is True

    def test_is_hotspot_below_threshold(self):
        assert self.detector.is_hotspot(0.3) is False

    def test_is_hotspot_at_threshold(self):
        assert self.detector.is_hotspot(0.5) is False

    def test_is_hotspot_custom_threshold(self):
        assert self.detector.is_hotspot(0.4, threshold=0.3) is True

    # -- detect_language --

    def test_detect_python(self):
        assert self.detector.detect_language("foo/bar.py") == "python"

    def test_detect_js(self):
        assert self.detector.detect_language("index.js") == "javascript"

    def test_detect_ts(self):
        assert self.detector.detect_language("app.ts") == "typescript"

    def test_detect_tsx(self):
        assert self.detector.detect_language("comp.tsx") == "typescript"

    def test_detect_java(self):
        assert self.detector.detect_language("Main.java") == "java"

    def test_detect_go(self):
        assert self.detector.detect_language("main.go") == "go"

    def test_detect_rust(self):
        assert self.detector.detect_language("lib.rs") == "rust"

    def test_detect_unknown(self):
        assert self.detector.detect_language("readme.txt") == "unknown"


# ===========================================================================
# change_frequency_analyzer.py  (ChangeMetrics, GitAnalyzer)
# ===========================================================================


class TestChangeMetrics:
    def setup_method(self):
        self.metrics = ChangeMetrics()

    def test_defaults(self):
        assert self.metrics.change_count == 0
        assert self.metrics.unique_authors == set()
        assert self.metrics.bug_fix_count == 0
        assert self.metrics.lines_added == 0
        assert self.metrics.lines_removed == 0

    def test_add_commit_normal(self):
        dt = datetime(2025, 1, 1, tzinfo=UTC)
        self.metrics.add_commit("abc123", "alice", dt, "add feature", 10, 5)
        assert self.metrics.change_count == 1
        assert "alice" in self.metrics.unique_authors
        assert self.metrics.lines_added == 10
        assert self.metrics.lines_removed == 5
        assert self.metrics.first_change_date == dt
        assert self.metrics.last_change_date == dt
        assert self.metrics.bug_fix_count == 0

    def test_add_commit_bugfix(self):
        dt = datetime(2025, 2, 1, tzinfo=UTC)
        self.metrics.add_commit("def456", "bob", dt, "fix crash in parser", 3, 1)
        assert self.metrics.bug_fix_count == 1

    def test_add_multiple_commits_date_tracking(self):
        dt1 = datetime(2025, 1, 1, tzinfo=UTC)
        dt2 = datetime(2025, 6, 1, tzinfo=UTC)
        self.metrics.add_commit("a", "alice", dt2, "second", 0, 0)
        self.metrics.add_commit("b", "bob", dt1, "first", 0, 0)
        assert self.metrics.first_change_date == dt1
        assert self.metrics.last_change_date == dt2

    def test_is_bug_fix_keywords(self):
        assert self.metrics._is_bug_fix("Fix login crash") is True
        assert self.metrics._is_bug_fix("resolve issue #42") is True
        assert self.metrics._is_bug_fix("Error handling improved") is True
        assert self.metrics._is_bug_fix("add new feature") is False

    def test_get_age_days_no_date(self):
        assert self.metrics.get_age_days() == 0

    def test_get_age_days_with_date(self):
        self.metrics.first_change_date = datetime.now(UTC) - timedelta(days=30)
        assert 29 <= self.metrics.get_age_days() <= 31

    def test_get_code_churn(self):
        self.metrics.lines_added = 100
        self.metrics.lines_removed = 50
        assert self.metrics.get_code_churn() == 150


class TestGitAnalyzer:
    """Test GitAnalyzer WITHOUT actually running git."""

    def test_init_no_repo(self):
        analyzer = GitAnalyzer(repo_path=None)
        assert analyzer.metrics_cache == {}

    @patch("processing.architecture.hotspot.change_frequency_analyzer.subprocess.run")
    def test_init_with_repo_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc|alice|alice@x.com|1700000000|add feature\n10\t5\tsrc/main.py\n",
        )
        analyzer = GitAnalyzer(repo_path=Path("/fake/repo"))
        assert "src/main.py" in analyzer.metrics_cache

    @patch("processing.architecture.hotspot.change_frequency_analyzer.subprocess.run")
    def test_init_with_repo_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=60)
        analyzer = GitAnalyzer(repo_path=Path("/fake/repo"))
        assert analyzer.metrics_cache == {}

    @patch("processing.architecture.hotspot.change_frequency_analyzer.subprocess.run")
    def test_init_with_repo_called_process_error(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        analyzer = GitAnalyzer(repo_path=Path("/fake/repo"))
        assert analyzer.metrics_cache == {}

    def test_get_change_frequency_missing(self):
        analyzer = GitAnalyzer(repo_path=None)
        assert analyzer.get_change_frequency("nonexist.py") == 0

    def test_get_change_frequency_cached(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.change_count = 7
        analyzer.metrics_cache["foo.py"] = m
        assert analyzer.get_change_frequency("foo.py") == 7

    def test_get_unique_authors(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.unique_authors = {"a", "b", "c"}
        analyzer.metrics_cache["f.py"] = m
        assert analyzer.get_unique_authors("f.py") == 3
        assert analyzer.get_unique_authors("missing.py") == 0

    def test_get_bug_fix_count(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.bug_fix_count = 3
        analyzer.metrics_cache["f.py"] = m
        assert analyzer.get_bug_fix_count("f.py") == 3
        assert analyzer.get_bug_fix_count("missing.py") == 0

    def test_get_code_churn(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.lines_added = 100
        m.lines_removed = 50
        analyzer.metrics_cache["f.py"] = m
        assert analyzer.get_code_churn("f.py") == 150
        assert analyzer.get_code_churn("x.py") == 0

    def test_get_age_days(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.first_change_date = datetime.now(UTC) - timedelta(days=10)
        analyzer.metrics_cache["f.py"] = m
        assert 9 <= analyzer.get_age_days("f.py") <= 11
        assert analyzer.get_age_days("x.py") == 0

    def test_get_recency_score_recent(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.last_change_date = datetime.now(UTC) - timedelta(days=1)
        analyzer.metrics_cache["f.py"] = m
        score = analyzer.get_recency_score("f.py")
        assert score > 0.9

    def test_get_recency_score_old(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.last_change_date = datetime.now(UTC) - timedelta(days=365)
        analyzer.metrics_cache["f.py"] = m
        score = analyzer.get_recency_score("f.py")
        assert score < 0.1

    def test_get_recency_score_missing(self):
        analyzer = GitAnalyzer(repo_path=None)
        assert analyzer.get_recency_score("nope.py") == 0.0

    def test_get_recency_score_no_last_change(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        analyzer.metrics_cache["f.py"] = m
        assert analyzer.get_recency_score("f.py") == 0.0

    def test_get_author_list(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.unique_authors = {"alice", "bob"}
        analyzer.metrics_cache["f.py"] = m
        authors = analyzer.get_author_list("f.py")
        assert set(authors) == {"alice", "bob"}
        assert analyzer.get_author_list("x.py") == []

    def test_get_commit_list(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        m.commits = ["aaa", "bbb"]
        analyzer.metrics_cache["f.py"] = m
        commits = analyzer.get_commit_list("f.py")
        assert commits == ["aaa", "bbb"]
        # Verify it's a copy
        commits.append("ccc")
        assert len(analyzer.metrics_cache["f.py"].commits) == 2
        assert analyzer.get_commit_list("x.py") == []

    def test_get_metrics(self):
        analyzer = GitAnalyzer(repo_path=None)
        m = ChangeMetrics()
        analyzer.metrics_cache["f.py"] = m
        assert analyzer.get_metrics("f.py") is m
        assert analyzer.get_metrics("x.py") is None

    def test_get_files_with_changes(self):
        analyzer = GitAnalyzer(repo_path=None)
        analyzer.metrics_cache["a.py"] = ChangeMetrics()
        analyzer.metrics_cache["b.py"] = ChangeMetrics()
        assert set(analyzer.get_files_with_changes()) == {"a.py", "b.py"}

    def test_clear_cache(self):
        analyzer = GitAnalyzer(repo_path=None)
        analyzer.metrics_cache["a.py"] = ChangeMetrics()
        analyzer.clear_cache()
        assert analyzer.get_cache_size() == 0

    def test_get_cache_size(self):
        analyzer = GitAnalyzer(repo_path=None)
        assert analyzer.get_cache_size() == 0
        analyzer.metrics_cache["a.py"] = ChangeMetrics()
        assert analyzer.get_cache_size() == 1

    def test_parse_git_log_binary_file(self):
        """Binary files show '-' for added/removed."""
        analyzer = GitAnalyzer(repo_path=None)
        log_output = "abc|alice|a@x|1700000000|commit msg\n-\t-\timage.png\n"
        analyzer._parse_git_log(log_output)
        assert "image.png" in analyzer.metrics_cache
        m = analyzer.metrics_cache["image.png"]
        assert m.lines_added == 0
        assert m.lines_removed == 0

    def test_parse_git_log_multiple_commits(self):
        analyzer = GitAnalyzer(repo_path=None)
        log_output = (
            "aaa|alice|a@x|1700000000|add feature\n"
            "5\t2\tsrc/a.py\n"
            "\n"
            "bbb|bob|b@x|1700100000|fix bug in a\n"
            "3\t1\tsrc/a.py\n"
            "10\t0\tsrc/b.py\n"
        )
        analyzer._parse_git_log(log_output)
        assert analyzer.metrics_cache["src/a.py"].change_count == 2
        assert analyzer.metrics_cache["src/b.py"].change_count == 1
        assert analyzer.metrics_cache["src/a.py"].bug_fix_count == 1

    def test_process_file_change_short_line(self):
        """Lines with < 3 tab-separated parts should be skipped."""
        analyzer = GitAnalyzer(repo_path=None)
        analyzer._process_file_change("bad line", "aaa", "alice", datetime.now(UTC), "msg")
        assert len(analyzer.metrics_cache) == 0

    def test_process_file_change_none_commit(self):
        """Should still create metrics entry but not add commit data."""
        analyzer = GitAnalyzer(repo_path=None)
        analyzer._process_file_change("5\t3\tfile.py", None, None, None, None)
        assert "file.py" in analyzer.metrics_cache
        assert analyzer.metrics_cache["file.py"].change_count == 0


# ===========================================================================
# complexity_analyzer.py  (ComplexityAnalyzer)
# ===========================================================================


class TestComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()

    def test_empty_content(self):
        result = self.analyzer.calculate_complexity(Path("x.py"), "", "python")
        assert result["cyclomatic_complexity"] == 0
        assert result["loc"] == 0

    def test_python_simple(self):
        code = "def foo():\n    return 1\n"
        result = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert result["function_count"] == 1
        assert result["loc"] > 0

    def test_python_with_branches(self):
        code = (
            "def bar(x):\n"
            "    if x > 0:\n"
            "        for i in range(x):\n"
            "            pass\n"
            "    else:\n"
            "        while True:\n"
            "            break\n"
        )
        result = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert result["cyclomatic_complexity"] >= 4  # 1 base + if + for + while
        assert result["max_nesting_depth"] >= 1

    def test_python_class_and_import(self):
        code = (
            "import os\nfrom sys import argv\n\nclass Foo:\n    def method(self):\n        pass\n"
        )
        result = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert result["class_count"] == 1
        assert result["function_count"] == 1
        assert result["import_count"] == 2

    def test_python_async_function(self):
        code = "async def afoo():\n    pass\n"
        result = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert result["function_count"] == 1

    def test_python_syntax_error_fallback(self):
        code = "def broken(:\n"
        result = self.analyzer.calculate_complexity(Path("bad.py"), code, "python")
        # Falls back to _get_default_metrics
        assert result["cyclomatic_complexity"] == 0

    def test_javascript_analysis(self):
        code = (
            'import { foo } from "bar";\n'
            "function greet(name) {\n"
            "  if (name) {\n"
            "    return name;\n"
            "  }\n"
            "}\n"
            "const fn = () => {};\n"
        )
        result = self.analyzer.calculate_complexity(Path("x.js"), code, "javascript")
        assert result["function_count"] >= 1
        assert result["import_count"] >= 1
        assert result["cyclomatic_complexity"] >= 1  # at least the if

    def test_typescript_analysis(self):
        code = "class Foo { method() { if (true) { } } }\n"
        result = self.analyzer.calculate_complexity(Path("x.ts"), code, "typescript")
        assert result["class_count"] >= 1

    def test_java_delegates_to_js(self):
        code = "class Main { void run() { if (true) {} } }\n"
        result = self.analyzer.calculate_complexity(Path("Main.java"), code, "java")
        assert result["class_count"] >= 1

    def test_generic_analysis(self):
        code = "if something\nfor x in y\nwhile true\nswitch\ncase\n"
        result = self.analyzer.calculate_complexity(Path("x.rb"), code, "ruby")
        assert result["cyclomatic_complexity"] >= 3

    def test_cache_hit(self):
        code = "def f():\n    pass\n"
        r1 = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        r2 = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert r1 is r2

    def test_cache_miss_different_content(self):
        r1 = self.analyzer.calculate_complexity(Path("x.py"), "x = 1\n", "python")
        r2 = self.analyzer.calculate_complexity(Path("x.py"), "y = 2\n", "python")
        assert r1 is not r2

    def test_clear_cache(self):
        self.analyzer.calculate_complexity(Path("x.py"), "x = 1\n", "python")
        assert self.analyzer.get_cache_size() == 1
        self.analyzer.clear_cache()
        assert self.analyzer.get_cache_size() == 0

    def test_get_default_metrics(self):
        m = self.analyzer._get_default_metrics()
        assert m["cyclomatic_complexity"] == 0
        assert m["average_function_complexity"] == 0.0

    def test_average_function_complexity(self):
        code = "def a():\n    if True:\n        pass\ndef b():\n    for x in y:\n        pass\n"
        result = self.analyzer.calculate_complexity(Path("x.py"), code, "python")
        assert result["function_count"] == 2
        # Each function has 1 (base) + 1 (branch) = 2, total = 4, avg = 2.0
        assert result["average_function_complexity"] == pytest.approx(2.0)


# ===========================================================================
# bottleneck_detector.py  (BottleneckMetrics, BottleneckDetector)
# ===========================================================================


class TestBottleneckMetrics:
    def test_defaults(self):
        m = BottleneckMetrics()
        assert m.coupling_score == 0.0
        assert m.dependency_count == 0
        assert m.centrality_score == 0.0

    def test_composite_score_zeros(self):
        m = BottleneckMetrics()
        assert m.get_composite_score() == 0.0

    def test_composite_score_clamps(self):
        m = BottleneckMetrics()
        m.coupling_score = 5.0
        m.dependency_count = 500
        m.centrality_score = 5.0
        m.hotpath_score = 5.0
        m.resource_intensity = 5.0
        m.scalability_risk = 5.0
        assert m.get_composite_score() == 1.0

    def test_composite_score_weighted(self):
        m = BottleneckMetrics()
        m.coupling_score = 1.0
        m.dependency_count = 50  # normalizes to 1.0
        m.centrality_score = 1.0
        m.hotpath_score = 1.0
        m.resource_intensity = 1.0
        m.scalability_risk = 1.0
        score = m.get_composite_score()
        assert score == pytest.approx(1.0)


class TestBottleneckDetector:
    def setup_method(self):
        self.detector = BottleneckDetector(repo_path=None)

    def test_detect_bottlenecks_empty_graph(self):
        g = nx.DiGraph()
        result = self.detector.detect_bottlenecks(g)
        assert result == []

    def test_detect_bottlenecks_no_file_path(self):
        g = nx.DiGraph()
        g.add_node("a", file_path="")
        result = self.detector.detect_bottlenecks(g)
        assert result == []

    def test_detect_bottlenecks_below_threshold(self):
        g = _make_graph(
            ("a", {"file_path": "a.py"}),
            ("b", {"file_path": "b.py"}),
        )
        result = self.detector.detect_bottlenecks(g, threshold=0.99)
        assert result == []

    def test_detect_bottlenecks_hub_node(self):
        """A node with many connections should score higher."""
        nodes = [("hub", {"file_path": "hub.py"})]
        edges = []
        for i in range(20):
            nid = f"n{i}"
            nodes.append((nid, {"file_path": f"{nid}.py"}))
            edges.append((nid, "hub"))
        g = _make_graph(*nodes, edges=edges)
        result = self.detector.detect_bottlenecks(g, threshold=0.1)
        assert len(result) > 0
        # Hub should be among the top-scored
        hub_results = [r for r in result if r["component_id"] == "hub"]
        assert len(hub_results) == 1
        assert hub_results[0]["score"] > 0.1

    def test_identify_bottleneck_risks_high_coupling(self):
        m = BottleneckMetrics()
        m.coupling_score = 0.8
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("coupling" in r.lower() for r in risks)

    def test_identify_bottleneck_risks_many_deps(self):
        m = BottleneckMetrics()
        m.dependency_count = 25
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("dependencies" in r.lower() for r in risks)

    def test_identify_bottleneck_risks_critical_path(self):
        m = BottleneckMetrics()
        m.centrality_score = 0.6
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("Critical path" in r for r in risks)

    def test_identify_bottleneck_risks_frequently_executed(self):
        m = BottleneckMetrics()
        m.hotpath_score = 0.7
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("executed" in r.lower() for r in risks)

    def test_identify_bottleneck_risks_resource_intensive(self):
        m = BottleneckMetrics()
        m.resource_intensity = 0.7
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("Resource" in r for r in risks)

    def test_identify_bottleneck_risks_scalability(self):
        m = BottleneckMetrics()
        m.scalability_risk = 0.8
        risks = self.detector._identify_bottleneck_risks(m)
        assert any("Scalability" in r for r in risks)

    def test_identify_bottleneck_risks_none(self):
        m = BottleneckMetrics()
        assert self.detector._identify_bottleneck_risks(m) == []

    def test_generate_bottleneck_recommendations_coupling(self):
        m = BottleneckMetrics()
        m.coupling_score = 0.8
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert any("coupling" in r.lower() for r in recs)

    def test_generate_bottleneck_recommendations_deps(self):
        m = BottleneckMetrics()
        m.dependency_count = 25
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert any("dependency" in r.lower() for r in recs)

    def test_generate_bottleneck_recommendations_centrality(self):
        m = BottleneckMetrics()
        m.centrality_score = 0.6
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert any("caching" in r.lower() or "optimize" in r.lower() for r in recs)

    def test_generate_bottleneck_recommendations_resource(self):
        m = BottleneckMetrics()
        m.resource_intensity = 0.7
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert any("Profile" in r for r in recs)

    def test_generate_bottleneck_recommendations_scalability(self):
        m = BottleneckMetrics()
        m.scalability_risk = 0.8
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert any("scaling" in r.lower() for r in recs)

    def test_generate_bottleneck_recommendations_fallback(self):
        m = BottleneckMetrics()
        recs = self.detector._generate_bottleneck_recommendations(m)
        assert recs == ["Monitor performance metrics for this component"]

    def test_analyze_dependency_chains_simple(self):
        g = _make_graph(
            ("a", {"file_path": "a.py"}),
            ("b", {"file_path": "b.py"}),
            ("c", {"file_path": "c.py"}),
            ("d", {"file_path": "d.py"}),
            edges=[("a", "b"), ("b", "c"), ("c", "d")],
        )
        chains = self.detector.analyze_dependency_chains(g, max_depth=5)
        assert len(chains) > 0
        # Should find chains of length >= 3
        assert any(c["length"] >= 3 for c in chains)

    def test_analyze_dependency_chains_empty_graph(self):
        g = nx.DiGraph()
        chains = self.detector.analyze_dependency_chains(g)
        assert chains == []

    def test_find_longest_paths_no_successors(self):
        g = _make_graph(("a", {}))
        paths = self.detector._find_longest_paths(g, "a", 5)
        assert paths == []

    def test_find_longest_paths_cycle_avoidance(self):
        g = nx.DiGraph()
        g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        paths = self.detector._find_longest_paths(g, "a", 5)
        # Should stop before revisiting 'a'
        for path in paths:
            assert len(path) == len(set(path))

    def test_calculate_chain_risk_short(self):
        g = nx.DiGraph()
        g.add_node("a")
        assert self.detector._calculate_chain_risk(g, ["a"]) == 0.0

    def test_calculate_chain_risk_long(self):
        nodes = [f"n{i}" for i in range(10)]
        g = nx.DiGraph()
        for n in nodes:
            g.add_node(n)
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1])
        risk = self.detector._calculate_chain_risk(g, nodes)
        assert 0.0 < risk <= 1.0

    def test_calculate_centrality_scores(self):
        g = _make_graph(("a", {}), ("b", {}), ("c", {}), edges=[("a", "b"), ("b", "c")])
        scores = self.detector._calculate_centrality_scores(g)
        assert "a" in scores
        assert "b" in scores
        assert "c" in scores

    def test_calculate_centrality_scores_fallback(self):
        """If centrality calculation fails, falls back to degree-based."""
        g = _make_graph(("a", {}), ("b", {}), edges=[("a", "b")])
        with patch("networkx.betweenness_centrality", side_effect=Exception("fail")):
            scores = self.detector._calculate_centrality_scores(g)
        assert "a" in scores
        assert "b" in scores


# ===========================================================================
# hotspot_reporter.py  (HotspotReport, HotspotReporter)
# ===========================================================================


class TestHotspotReport:
    def test_init(self):
        report = HotspotReport("test-codebase")
        assert report.codebase_id == "test-codebase"
        assert report.hotspots == []
        assert report.bottlenecks == []
        assert report.summary_stats == {}
        assert report.analysis_metadata == {}
        assert isinstance(report.timestamp, datetime)

    def test_add_hotspots(self):
        report = HotspotReport("cb")
        hotspots = [_make_hotspot("a"), _make_hotspot("b")]
        report.add_hotspots(hotspots)
        assert len(report.hotspots) == 2

    def test_add_bottlenecks(self):
        report = HotspotReport("cb")
        bottlenecks = [{"component_id": "x", "score": 0.7}]
        report.add_bottlenecks(bottlenecks)
        assert report.bottlenecks == bottlenecks

    def test_set_summary_stats(self):
        report = HotspotReport("cb")
        stats = {"total_hotspots": 5}
        report.set_summary_stats(stats)
        assert report.summary_stats == stats

    def test_set_metadata(self):
        report = HotspotReport("cb")
        meta = {"version": "1.0"}
        report.set_metadata(meta)
        assert report.analysis_metadata == meta

    def test_to_dict(self):
        report = HotspotReport("cb")
        h = _make_hotspot("comp1")
        report.add_hotspots([h])
        report.add_bottlenecks([{"x": 1}])
        report.set_summary_stats({"total": 1})
        report.set_metadata({"ver": "1"})
        d = report.to_dict()
        assert d["codebase_id"] == "cb"
        assert len(d["hotspots"]) == 1
        assert d["hotspots"][0]["component_id"] == "comp1"
        assert d["bottlenecks"] == [{"x": 1}]
        assert d["summary_stats"] == {"total": 1}
        assert d["analysis_metadata"] == {"ver": "1"}
        assert "timestamp" in d


class TestHotspotReporter:
    def setup_method(self):
        self.reporter = HotspotReporter()

    def _make_report_with_hotspots(self):
        hotspots = [
            _make_hotspot("high", 0.9, 0.9, ["bug-prone"], ["High complexity (90%)"], ["Refactor"]),
            _make_hotspot(
                "med", 0.6, 0.6, ["unstable"], ["Frequently changed (60%)"], ["Test more"]
            ),
            _make_hotspot("low", 0.3, 0.3, [], [], []),
        ]
        bottlenecks = [
            {
                "component_id": "bottleneck1",
                "score": 0.8,
                "risk_factors": ["High coupling"],
                "recommendations": ["Reduce coupling"],
            }
        ]
        return self.reporter.generate_report("test-cb", hotspots, bottlenecks, {"version": "1.0"})

    def test_generate_report_basic(self):
        report = self.reporter.generate_report("cb1", [])
        assert report.codebase_id == "cb1"
        assert report.summary_stats["total_hotspots"] == 0

    def test_generate_report_with_data(self):
        report = self._make_report_with_hotspots()
        assert report.summary_stats["total_hotspots"] == 3
        assert report.summary_stats["high_risk_hotspots"] == 1  # score >= 0.8
        assert report.summary_stats["total_bottlenecks"] == 1
        assert report.analysis_metadata == {"version": "1.0"}

    def test_generate_report_no_bottlenecks(self):
        report = self.reporter.generate_report("cb", [_make_hotspot()])
        assert report.bottlenecks == []

    def test_summary_stats_score_distribution(self):
        hotspots = [
            _make_hotspot("a", 0.9, predicted_issues=["bug-prone", "unstable"]),
            _make_hotspot("b", 0.7, predicted_issues=["bug-prone"]),
        ]
        report = self.reporter.generate_report("cb", hotspots)
        stats = report.summary_stats
        assert stats["average_hotspot_score"] == pytest.approx(0.8)
        assert stats["max_hotspot_score"] == 0.9
        assert stats["issue_type_distribution"]["bug-prone"] == 2
        assert stats["issue_type_distribution"]["unstable"] == 1

    def test_summary_stats_risk_factor_distribution(self):
        hotspots = [
            _make_hotspot("a", risk_factors=["High complexity (90%)", "Frequently changed (70%)"]),
            _make_hotspot("b", risk_factors=["High complexity (80%)"]),
        ]
        report = self.reporter.generate_report("cb", hotspots)
        risk_dist = report.summary_stats["risk_factor_distribution"]
        assert risk_dist["High complexity"] == 2
        assert risk_dist["Frequently changed"] == 1

    def test_generate_text_report(self):
        report = self._make_report_with_hotspots()
        text = self.reporter.generate_text_report(report)
        assert "HOTSPOT ANALYSIS REPORT" in text
        assert "test-cb" in text
        assert "SUMMARY" in text
        assert "TOP HOTSPOTS" in text
        assert "PERFORMANCE BOTTLENECKS" in text
        assert "bottleneck1" in text
        assert "High coupling" in text

    def test_generate_text_report_no_bottlenecks(self):
        report = self.reporter.generate_report("cb", [_make_hotspot()])
        text = self.reporter.generate_text_report(report)
        assert "PERFORMANCE BOTTLENECKS" not in text

    def test_generate_text_report_empty(self):
        report = self.reporter.generate_report("empty", [])
        text = self.reporter.generate_text_report(report)
        assert "Total Hotspots: 0" in text

    def test_generate_text_report_issue_distribution(self):
        h = _make_hotspot(predicted_issues=["bug-prone", "high-complexity"])
        report = self.reporter.generate_report("cb", [h])
        text = self.reporter.generate_text_report(report)
        assert "ISSUE TYPE DISTRIBUTION" in text

    def test_save_report_json(self, tmp_path):
        report = self._make_report_with_hotspots()
        out_path = tmp_path / "report.json"
        self.reporter.save_report_json(report, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["codebase_id"] == "test-cb"

    def test_save_report_json_error(self):
        report = self._make_report_with_hotspots()
        # Pass a path that can't be written to
        bad_path = Path("/nonexistent_dir_xyz/report.json")
        # Should log error but not raise
        self.reporter.save_report_json(report, bad_path)

    def test_save_report_text(self, tmp_path):
        report = self._make_report_with_hotspots()
        out_path = tmp_path / "report.txt"
        self.reporter.save_report_text(report, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "HOTSPOT ANALYSIS REPORT" in content

    def test_save_report_text_error(self):
        report = self._make_report_with_hotspots()
        bad_path = Path("/nonexistent_dir_xyz/report.txt")
        self.reporter.save_report_text(report, bad_path)

    def test_generate_visualization_data(self):
        report = self._make_report_with_hotspots()
        viz = self.reporter.generate_visualization_data(report)
        assert "hotspot_scores" in viz
        assert "score_distribution" in viz
        assert "recommendation_frequency" in viz
        assert "bottleneck_scores" in viz

    def test_generate_visualization_data_no_bottlenecks(self):
        report = self.reporter.generate_report("cb", [_make_hotspot()])
        viz = self.reporter.generate_visualization_data(report)
        assert "bottleneck_scores" not in viz

    def test_score_distribution(self):
        hotspots = [
            _make_hotspot("a", score=0.1),
            _make_hotspot("b", score=0.3),
            _make_hotspot("c", score=0.5),
            _make_hotspot("d", score=0.7),
            _make_hotspot("e", score=0.9),
        ]
        dist = self.reporter._calculate_score_distribution(hotspots)
        assert dist["0.0-0.2"] == 1
        assert dist["0.2-0.4"] == 1
        assert dist["0.4-0.6"] == 1
        assert dist["0.6-0.8"] == 1
        assert dist["0.8-1.0"] == 1

    def test_recommendation_frequency(self):
        hotspots = [
            _make_hotspot(recommended_actions=["Refactor code", "Add tests"]),
            _make_hotspot(recommended_actions=["Document design", "Split module"]),
            _make_hotspot(recommended_actions=["Monitor performance", "Architect changes"]),
            _make_hotspot(recommended_actions=["Decompose component"]),
            _make_hotspot(recommended_actions=["Something else entirely"]),
        ]
        freq = self.reporter._calculate_recommendation_frequency(hotspots)
        assert freq.get("Refactoring", 0) >= 1
        assert freq.get("Testing", 0) >= 1
        assert freq.get("Documentation", 0) >= 1
        assert freq.get("Decomposition", 0) >= 2  # split + decompose
        assert freq.get("Monitoring", 0) >= 1
        assert freq.get("Architecture", 0) >= 1
        assert freq.get("Other", 0) >= 1

    def test_create_dashboard_summary(self):
        report = self._make_report_with_hotspots()
        summary = self.reporter.create_dashboard_summary(report)
        assert summary["codebase_id"] == "test-cb"
        assert "health_score" in summary
        assert "risk_level" in summary
        assert summary["needs_attention"] is True
        assert summary["total_hotspots"] == 3
        assert summary["bottleneck_count"] == 1

    def test_dashboard_risk_levels(self):
        # LOW risk
        report = self.reporter.generate_report("cb", [])
        summary = self.reporter.create_dashboard_summary(report)
        assert summary["risk_level"] == "LOW"

        # MEDIUM risk (> 5% high risk ratio)
        hotspots = [_make_hotspot(f"h{i}", score=0.9) for i in range(8)]
        report = self.reporter.generate_report("cb", hotspots)
        summary = self.reporter.create_dashboard_summary(report)
        assert summary["risk_level"] in ("MEDIUM", "HIGH")

    def test_dashboard_high_risk(self):
        # HIGH risk (> 10% high risk ratio)
        hotspots = [_make_hotspot(f"h{i}", score=0.9) for i in range(15)]
        report = self.reporter.generate_report("cb", hotspots)
        summary = self.reporter.create_dashboard_summary(report)
        assert summary["risk_level"] == "HIGH"


# ===========================================================================
# hotspot_analyzer.py  (HotspotAnalyzer)
# ===========================================================================


class TestHotspotAnalyzer:
    def setup_method(self):
        # No repo_path => no git subprocess
        self.analyzer = HotspotAnalyzer(repo_path=None)

    def test_init_no_repo(self):
        assert self.analyzer.repo_path is None
        assert self.analyzer.git_analyzer is None
        assert isinstance(self.analyzer.complexity_analyzer, ComplexityAnalyzer)
        assert isinstance(self.analyzer.hotspot_detector, HotspotDetector)
        assert isinstance(self.analyzer.bottleneck_detector, BottleneckDetector)
        assert isinstance(self.analyzer.reporter, HotspotReporter)

    def test_analyze_empty_graph(self):
        g = nx.DiGraph()
        result = self.analyzer.analyze(g, "test-cb")
        assert result == []

    def test_analyze_no_file_path(self):
        g = nx.DiGraph()
        g.add_node("n1", other="data")
        result = self.analyzer.analyze(g, "test-cb")
        assert result == []

    def test_analyze_returns_sorted_hotspots(self):
        """With no git/repo, complexity is 0, so scores will be low.
        We mock the score calculation to test sorting."""
        g = _make_graph(
            ("a", {"file_path": "a.py"}),
            ("b", {"file_path": "b.py"}),
        )

        scores = {"a.py": (0.9, {"complexity": 0.9}), "b.py": (0.7, {"complexity": 0.7})}
        with patch.object(
            self.analyzer, "_calculate_hotspot_score", side_effect=lambda fp: scores[fp]
        ):
            result = self.analyzer.analyze(g, "test-cb")

        assert len(result) == 2
        assert result[0].score >= result[1].score

    def test_analyze_limits_to_50(self):
        """Verify top-50 limit."""
        g = nx.DiGraph()
        for i in range(60):
            g.add_node(f"n{i}", file_path=f"f{i}.py")

        with patch.object(
            self.analyzer,
            "_calculate_hotspot_score",
            return_value=(0.8, {"complexity": 0.8}),
        ):
            result = self.analyzer.analyze(g, "test-cb")

        assert len(result) <= 50

    @patch(
        "processing.architecture.hotspot.hotspot_analyzer.GitAnalyzer",
    )
    def test_analyze_reinitializes_git_on_new_path(self, mock_git_cls):
        mock_git_cls.return_value = MagicMock()
        g = nx.DiGraph()
        new_path = Path("/new/repo")
        self.analyzer.analyze(g, "test-cb", repo_path=new_path)
        assert self.analyzer.repo_path == new_path
        mock_git_cls.assert_called_once_with(new_path)

    def test_calculate_hotspot_score_no_git_no_repo(self):
        score, features = self.analyzer._calculate_hotspot_score("test.py")
        assert score == 0.0
        assert features["change_frequency"] == 0.0
        assert features["complexity"] == 0.0

    def test_analyze_with_report(self):
        g = _make_graph(
            ("a", {"file_path": "a.py"}),
        )
        with patch.object(
            self.analyzer,
            "_calculate_hotspot_score",
            return_value=(0.8, {"complexity": 0.8}),
        ):
            report = self.analyzer.analyze_with_report(g, "test-cb")

        assert isinstance(report, HotspotReport)
        assert report.codebase_id == "test-cb"
        assert report.analysis_metadata["total_nodes"] == 1

    def test_analyze_with_report_bottleneck_failure(self):
        """Bottleneck failure should not crash the analysis."""
        g = _make_graph(("a", {"file_path": "a.py"}))
        with (
            patch.object(
                self.analyzer,
                "_calculate_hotspot_score",
                return_value=(0.3, {}),
            ),
            patch.object(
                self.analyzer.bottleneck_detector,
                "detect_bottlenecks",
                side_effect=Exception("fail"),
            ),
        ):
            report = self.analyzer.analyze_with_report(g, "test-cb")

        assert isinstance(report, HotspotReport)

    def test_analyze_with_report_no_bottlenecks_flag(self):
        g = nx.DiGraph()
        report = self.analyzer.analyze_with_report(g, "cb", include_bottlenecks=False)
        assert report.analysis_metadata["bottleneck_analysis_enabled"] is False

    def test_detect_bottlenecks_delegates(self):
        g = nx.DiGraph()
        with patch.object(
            self.analyzer.bottleneck_detector, "detect_bottlenecks", return_value=[]
        ) as mock_detect:
            result = self.analyzer.detect_bottlenecks(g, threshold=0.5)
        mock_detect.assert_called_once_with(g, 0.5)
        assert result == []

    def test_analyze_dependency_chains_delegates(self):
        g = nx.DiGraph()
        with patch.object(
            self.analyzer.bottleneck_detector,
            "analyze_dependency_chains",
            return_value=[],
        ) as mock_chains:
            result = self.analyzer.analyze_dependency_chains(g, max_depth=3)
        mock_chains.assert_called_once_with(g, 3)
        assert result == []

    def test_save_report(self, tmp_path):
        report = HotspotReport("cb")
        report.set_summary_stats({})
        self.analyzer.save_report(report, tmp_path, formats=["json", "text"])
        assert (tmp_path / "hotspot_report_cb.json").exists()
        assert (tmp_path / "hotspot_report_cb.txt").exists()

    def test_save_report_default_formats(self, tmp_path):
        report = HotspotReport("cb")
        report.set_summary_stats({})
        self.analyzer.save_report(report, tmp_path)
        assert (tmp_path / "hotspot_report_cb.json").exists()
        assert (tmp_path / "hotspot_report_cb.txt").exists()

    def test_save_report_json_only(self, tmp_path):
        report = HotspotReport("cb")
        report.set_summary_stats({})
        self.analyzer.save_report(report, tmp_path, formats=["json"])
        assert (tmp_path / "hotspot_report_cb.json").exists()
        assert not (tmp_path / "hotspot_report_cb.txt").exists()

    def test_get_visualization_data(self):
        report = HotspotReport("cb")
        report.set_summary_stats({})
        viz = self.analyzer.get_visualization_data(report)
        assert "hotspot_scores" in viz

    def test_get_dashboard_summary(self):
        report = HotspotReport("cb")
        report.set_summary_stats({"high_risk_hotspots": 0, "total_hotspots": 0})
        summary = self.analyzer.get_dashboard_summary(report)
        assert summary["codebase_id"] == "cb"
        assert "health_score" in summary
