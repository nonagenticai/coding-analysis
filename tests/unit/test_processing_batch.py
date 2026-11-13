"""Tests for processing modules.

Covers real behavior of:
- processing.architecture.code_quality_metrics.CodeQualityMetrics
- processing.architecture.code_pattern_recognition.CodePatternRecognizer
- processing.architecture.component_extractor.ComponentExtractor (instantiation, helpers)
- processing.architecture.dependency_graph.DependencyGraphBuilder
- processing.analyzers.analyzer_factory.AnalyzerFactory
- processing.contract_analysis.parsers.base (get_parser, PY_LANGUAGE)
"""

from __future__ import annotations

import re

import networkx as nx
import pytest

from models.architecture_models import ArchitectureQualityIssue, Component
from processing.architecture.code_pattern_recognition import CodePatternRecognizer
from processing.architecture.code_quality_metrics import CodeQualityMetrics

# ---------------------------------------------------------------------------
# Helper to build ArchitectureQualityIssue instances quickly
# ---------------------------------------------------------------------------


def _issue(
    severity: str = "medium",
    tool: str = "ruff",
    category: str = "style",
    rule_id: str = "W001",
) -> ArchitectureQualityIssue:
    return ArchitectureQualityIssue(
        id="i1",
        severity=severity,
        message="msg",
        rule_id=rule_id,
        rule_name=rule_id,
        file_path="f.py",
        line_number=1,
        tool=tool,
        category=category,
    )


# ---------------------------------------------------------------------------
# CodeQualityMetrics
# ---------------------------------------------------------------------------


class TestCodeQualityMetrics:
    """Test real scoring, grading, and counting behaviour."""

    def test_no_issues_gives_perfect_score(self):
        m = CodeQualityMetrics()
        assert m.calculate_quality_score([]) == 100.0

    def test_critical_issues_lower_score_more(self):
        m = CodeQualityMetrics()
        low_score = m.calculate_quality_score([_issue("low")])
        crit_score = m.calculate_quality_score([_issue("critical")])
        assert crit_score < low_score

    def test_many_issues_never_go_below_zero(self):
        m = CodeQualityMetrics()
        issues = [_issue("critical") for _ in range(100)]
        assert m.calculate_quality_score(issues) >= 0.0

    @pytest.mark.parametrize(
        ("score", "grade"),
        [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (50, "F"), (0, "F")],
    )
    def test_score_to_grade(self, score: float, grade: str):
        m = CodeQualityMetrics()
        assert m.score_to_grade(score) == grade

    def test_count_issues_by_severity(self):
        m = CodeQualityMetrics()
        issues = [_issue("critical"), _issue("high"), _issue("low"), _issue("low")]
        counts = m.count_issues_by_severity(issues)
        assert counts["critical"] == 1
        assert counts["high"] == 1
        assert counts["low"] == 2
        assert counts["medium"] == 0

    def test_unknown_severity_counted_as_medium(self):
        m = CodeQualityMetrics()
        counts = m.count_issues_by_severity([_issue("info")])
        assert counts["medium"] == 1

    def test_count_issues_by_tool(self):
        m = CodeQualityMetrics()
        issues = [_issue(tool="ruff"), _issue(tool="ruff"), _issue(tool="ty")]
        counts = m.count_issues_by_tool(issues)
        assert counts["ruff"] == 2
        assert counts["ty"] == 1

    def test_count_issues_by_category(self):
        m = CodeQualityMetrics()
        issues = [_issue(category="style"), _issue(category="security")]
        counts = m.count_issues_by_category(issues)
        assert counts["style"] == 1
        assert counts["security"] == 1

    def test_tool_coverage_all_run(self):
        m = CodeQualityMetrics()
        assert m.calculate_tool_coverage(["a", "b"], ["a", "b"]) == 100.0

    def test_tool_coverage_partial(self):
        m = CodeQualityMetrics()
        assert m.calculate_tool_coverage(["a"], ["a", "b"]) == 50.0

    def test_tool_coverage_no_available(self):
        m = CodeQualityMetrics()
        assert m.calculate_tool_coverage([], []) == 100.0

    def test_severity_distribution_percentages(self):
        m = CodeQualityMetrics()
        issues = [_issue("critical"), _issue("high"), _issue("high"), _issue("low")]
        dist = m.calculate_severity_distribution(issues)
        assert dist["critical"] == 25.0
        assert dist["high"] == 50.0
        assert dist["low"] == 25.0

    def test_severity_distribution_no_issues(self):
        m = CodeQualityMetrics()
        dist = m.calculate_severity_distribution([])
        assert all(v == 0.0 for v in dist.values())

    def test_quality_trend_without_previous(self):
        m = CodeQualityMetrics()
        indicators = m.get_quality_trend_indicators([_issue("low")])
        assert "current_score" in indicators
        assert "current_grade" in indicators
        assert "score_change" not in indicators

    def test_quality_trend_with_previous(self):
        m = CodeQualityMetrics()
        current = [_issue("low")]
        previous = [_issue("critical"), _issue("critical")]
        indicators = m.get_quality_trend_indicators(current, previous)
        assert indicators["improving"] is True
        assert indicators["score_change"] > 0


# ---------------------------------------------------------------------------
# CodePatternRecognizer
# ---------------------------------------------------------------------------


class TestCodePatternRecognizer:
    """Test real pattern matching, code extraction, and classification."""

    def test_get_compiled_pattern_caches(self):
        r = CodePatternRecognizer()
        p1 = r.get_compiled_pattern(r"class\s+\w+")
        p2 = r.get_compiled_pattern(r"class\s+\w+")
        assert p1 is p2

    def test_get_compiled_pattern_case_insensitive(self):
        r = CodePatternRecognizer()
        p = r.get_compiled_pattern(r"class\s+\w+")
        assert p.search("CLASS Foo")

    @pytest.mark.parametrize(
        ("node_data", "expected_code"),
        [
            ({"code": "def foo(): pass"}, "def foo(): pass"),
            ({"docstring": "some doc"}, "some doc"),
            ({"properties": {"source_code": "x=1"}}, "x=1"),
            ({"id": "node42"}, "node42"),
            ({}, ""),
        ],
    )
    def test_get_node_code(self, node_data: dict, expected_code: str):
        r = CodePatternRecognizer()
        assert r.get_node_code(node_data) == expected_code

    def test_node_matches_pattern(self):
        r = CodePatternRecognizer()
        pattern = re.compile(r"singleton", re.IGNORECASE)
        assert r.node_matches_pattern({"code": "class Singleton: pass"}, pattern)
        assert not r.node_matches_pattern({"code": "class Foo: pass"}, pattern)

    def test_get_node_type(self):
        r = CodePatternRecognizer()
        assert r.get_node_type({"type": "Class"}) == "class"
        assert r.get_node_type({}) == "unknown"

    def test_extract_class_info_with_class(self):
        r = CodePatternRecognizer()
        node = {"code": "class MyClass(Base, Mixin):\n    def method(self): pass"}
        info = r.extract_class_info(node)
        assert info["is_class"] is True
        assert info["class_name"] == "MyClass"
        assert "method" in info["methods"]
        assert "Base" in info["inherits_from"]

    def test_extract_class_info_without_class(self):
        r = CodePatternRecognizer()
        node = {"code": "x = 1\ny = 2"}
        info = r.extract_class_info(node)
        assert info["is_class"] is False

    def test_calculate_code_complexity(self):
        r = CodePatternRecognizer()
        simple = {"code": "x = 1"}
        complex_code = {
            "code": "if x:\n  for i in range(10):\n    while True:\n      try:\n        pass\n      except:\n        pass"
        }
        assert r.calculate_code_complexity(simple) == 0
        assert r.calculate_code_complexity(complex_code) > 0

    @pytest.mark.parametrize(
        ("node_data", "expected"),
        [
            ({"code": "def test_foo(): assert True"}, True),
            ({"id": "test_module.py", "code": ""}, True),
            ({"code": "x = 1", "id": "main.py"}, False),
        ],
    )
    def test_is_test_code(self, node_data: dict, expected: bool):
        r = CodePatternRecognizer()
        assert r.is_test_code(node_data) is expected

    @pytest.mark.parametrize(
        ("node_data", "expected"),
        [
            ({"code": "settings = {}", "id": "config.py"}, True),
            ({"code": "x = 1", "id": "main.py"}, False),
        ],
    )
    def test_is_configuration_code(self, node_data: dict, expected: bool):
        r = CodePatternRecognizer()
        assert r.is_configuration_code(node_data) is expected

    def test_detect_custom_patterns_on_graph(self):
        r = CodePatternRecognizer()
        g = nx.DiGraph()
        g.add_node("n1", code="class Singleton:\n    _instance = None")
        g.add_node("n2", code="x = 1")
        patterns = [{"name": "Singleton", "match": r"Singleton"}]
        detected = r.detect_custom_patterns(g, patterns)
        assert len(detected) == 1
        assert detected[0].pattern_type == "Singleton"
        assert "n1" in detected[0].components

    def test_find_nodes_by_pattern(self):
        r = CodePatternRecognizer()
        g = nx.DiGraph()
        g.add_node("n1", code="class Foo: pass")
        g.add_node("n2", code="x = 1")
        matches = r.find_nodes_by_pattern(g, r"class\s+\w+")
        assert len(matches) == 1
        assert matches[0][0] == "n1"

    def test_has_edge_type(self):
        r = CodePatternRecognizer()
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", type="imports")
        assert r.has_edge_type(g, "a", "imports") is True
        assert r.has_edge_type(g, "a", "calls") is False
        assert r.has_edge_type(g, "b", "imports") is True  # incoming

    def test_get_connected_nodes_by_edge_type(self):
        r = CodePatternRecognizer()
        g = nx.DiGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", type="imports")
        g.add_edge("a", "c", type="calls")
        connected = r.get_connected_nodes_by_edge_type(g, "a", "imports")
        assert "b" in connected
        assert "c" not in connected


# ---------------------------------------------------------------------------
# ComponentExtractor (instantiation + helper methods)
# ---------------------------------------------------------------------------


class TestComponentExtractor:
    """Test ComponentExtractor construction and utility methods."""

    def test_instantiation_defaults(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        assert ext.max_workers >= 1
        assert ext.min_parallel_threshold == 1000
        assert ext.use_processes is False

    def test_get_component_name_from_id(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        assert ext._get_component_name("component_api_gateway") == "Api Gateway"
        assert ext._get_component_name("custom-id") == "custom-id"

    def test_matches_pattern(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        assert ext._matches_pattern("src/api/routes.py", "src/api/*")
        assert not ext._matches_pattern("tests/test.py", "src/api/*")

    def test_get_component_id_from_pattern_glob(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        cid = ext._get_component_id_from_pattern("src/apps/auth/main.py", "src/apps/**")
        assert "auth" in cid

    def test_get_component_id_from_pattern_simple(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        cid = ext._get_component_id_from_pattern("src/lib/utils.py", "src/lib/*")
        assert "lib" in cid

    def test_get_extraction_stats(self):
        from processing.architecture.component_extractor import ComponentExtractor

        ext = ComponentExtractor()
        comps = [
            Component(id="c1", name="C1", files=["a.py", "b.py"]),
            Component(id="c2", name="C2", files=["c.py"]),
        ]
        stats = ext.get_extraction_stats(comps)
        assert stats.components_found == 2
        assert stats.edges_scanned == 3


# ---------------------------------------------------------------------------
# DependencyGraphBuilder
# ---------------------------------------------------------------------------


class TestDependencyGraphBuilder:
    """Test DependencyGraphBuilder init."""

    def test_instantiation_with_timeout(self):
        from processing.architecture.dependency_graph import DependencyGraphBuilder

        builder = DependencyGraphBuilder(timeout=60)
        assert builder.timeout == 60

    def test_default_timeout(self):
        from processing.architecture.dependency_graph import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        assert builder.timeout == 30


# ---------------------------------------------------------------------------
# AnalyzerFactory
# ---------------------------------------------------------------------------


class TestAnalyzerFactory:
    """Test AnalyzerFactory methods with real logic."""

    def test_get_supported_languages(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        langs = factory.get_supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs

    def test_get_supported_extensions(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        exts = factory.get_supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("main.py", True),
            ("index.js", True),
            ("app.tsx", True),
            ("readme.md", False),
        ],
    )
    def test_is_supported_file(self, path: str, expected: bool):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        assert factory.is_supported_file(path) is expected

    def test_dts_file_is_supported(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        assert factory.is_supported_file("types.d.ts") is True

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("main.py", "python"),
            ("index.js", "javascript"),
            ("app.ts", "typescript"),
            ("readme.md", None),
        ],
    )
    def test_get_language_for_file(self, path: str, expected: str | None):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        assert factory.get_language_for_file(path) == expected

    def test_get_analyzer_info(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        info = factory.get_analyzer_info()
        assert "python" in info
        assert "class" in info["python"]
        assert "extensions" in info["python"]

    def test_get_analyzer_returns_instance(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer("python")
        assert analyzer is not None

    def test_get_analyzer_unknown_returns_none(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        assert factory.get_analyzer("brainfuck") is None

    def test_get_analyzer_caches_instance(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        a1 = factory.get_analyzer("python")
        a2 = factory.get_analyzer("python")
        assert a1 is a2

    def test_clear_cache(self):
        from processing.analyzers.analyzer_factory import AnalyzerFactory

        factory = AnalyzerFactory()
        factory.get_analyzer("python")
        factory.clear_cache()
        assert len(factory._analyzers) == 0


# ---------------------------------------------------------------------------
# contract_analysis parsers base
# ---------------------------------------------------------------------------


class TestContractAnalysisParsersBase:
    """Test tree-sitter parser setup."""

    def test_py_language_is_valid(self):
        from processing.contract_analysis.parsers.base import PY_LANGUAGE

        assert PY_LANGUAGE is not None

    def test_get_parser_returns_parser(self):
        from processing.contract_analysis.parsers.base import get_parser

        parser = get_parser()
        assert parser is not None

    def test_parser_can_parse_python(self):
        from processing.contract_analysis.parsers.base import get_parser

        parser = get_parser()
        tree = parser.parse(b"def foo(): pass\n")
        assert tree.root_node is not None
        assert tree.root_node.type == "module"
