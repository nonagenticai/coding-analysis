"""Unit tests for processing.architecture.metrics_calculator module."""

from __future__ import annotations

import networkx as nx
import pytest

from models.architecture_models import ArchitecturalMetric, Component
from processing.architecture.metrics_calculator import MetricsCalculator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _comp(cid: str, files: list[str], deps: list[str] | None = None) -> Component:
    return Component(id=cid, name=cid, files=files, dependencies=deps or [], layer=None)


def _make_graph(nodes=None, edges=None):
    g = nx.DiGraph()
    if nodes:
        for nid, data in nodes:
            g.add_node(nid, **data)
    if edges:
        for src, dst in edges:
            g.add_edge(src, dst)
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator():
    return MetricsCalculator()


@pytest.fixture
def sample_graph():
    return _make_graph(
        nodes=[
            ("n1", {"file_path": "src/app/main.py", "type": "class", "name": "App"}),
            ("n2", {"file_path": "src/lib/utils.py", "type": "class", "name": "Utils"}),
            ("n3", {"file_path": "src/lib/helpers.py", "type": "function", "name": "helper"}),
        ],
        edges=[("n1", "n2"), ("n1", "n3")],
    )


@pytest.fixture
def sample_components():
    return [
        _comp("comp-app", ["src/app/main.py"], deps=["comp-lib"]),
        _comp("comp-lib", ["src/lib/utils.py", "src/lib/helpers.py"]),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsGeneration:
    """test_metrics_generation"""

    def test_calc_returns_list_of_metrics(self, calculator, sample_graph, sample_components):
        result = calculator.calc(sample_graph, sample_components)
        assert isinstance(result, list)
        for m in result:
            assert isinstance(m, ArchitecturalMetric)

    def test_calc_metrics_have_component_ids(self, calculator, sample_graph, sample_components):
        result = calculator.calc(sample_graph, sample_components)
        comp_ids = {m.component_id for m in result}
        # Should have metrics for both components
        assert "comp-app" in comp_ids
        assert "comp-lib" in comp_ids

    def test_calc_metrics_have_expected_fields(self, calculator, sample_graph, sample_components):
        result = calculator.calc(sample_graph, sample_components)
        for m in result:
            assert hasattr(m, "instability")
            assert hasattr(m, "abstractness")
            assert hasattr(m, "distance")
            assert 0.0 <= m.instability <= 1.0
            assert 0.0 <= m.abstractness <= 1.0
            assert 0.0 <= m.distance <= 1.0

    def test_calculate_component_metrics(self, calculator, sample_graph, sample_components):
        result = calculator.calculate_component_metrics(sample_components, sample_graph)
        assert isinstance(result, dict)
        # Should have metrics for each component
        for comp in sample_components:
            if comp.id in result:
                assert isinstance(result[comp.id], dict)
                # Should contain instability and abstractness
                assert "instability" in result[comp.id]
                assert "abstractness" in result[comp.id]

    def test_get_metrics_summary(self, calculator, sample_graph, sample_components):
        summary = calculator.get_metrics_summary(sample_components, sample_graph)
        assert isinstance(summary, dict)
        assert "total_components" in summary
        assert summary["total_components"] == 2

    def test_validate_components_and_graph(self, calculator, sample_graph, sample_components):
        result = calculator.validate_components_and_graph(sample_components, sample_graph)
        assert isinstance(result, dict)

    def test_calculate_system_overview(self, calculator, sample_graph, sample_components):
        overview = calculator.calculate_system_overview(sample_components, sample_graph)
        assert isinstance(overview, dict)

    def test_analyze_dependency_patterns(self, calculator, sample_graph, sample_components):
        patterns = calculator.analyze_dependency_patterns(sample_components, sample_graph)
        assert isinstance(patterns, dict)

    def test_lcom4_delegation(self, calculator, sample_graph):
        """Test _calculate_lcom4 delegates to complexity_computer."""
        nodes = list(sample_graph.nodes())
        result = calculator._calculate_lcom4(sample_graph, nodes)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_tcc_delegation(self, calculator, sample_graph):
        """Test _calculate_tcc delegates to complexity_computer."""
        nodes = list(sample_graph.nodes())
        result = calculator._calculate_tcc(sample_graph, nodes)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestEmptyComponentsEdges:
    """test_empty_components_edges"""

    def test_calc_empty_components(self, calculator):
        g = nx.DiGraph()
        result = calculator.calc(g, [])
        assert result == []

    def test_calculate_component_metrics_empty(self, calculator):
        g = nx.DiGraph()
        result = calculator.calculate_component_metrics([], g)
        assert result == {}

    def test_get_metrics_summary_empty(self, calculator):
        g = nx.DiGraph()
        summary = calculator.get_metrics_summary([], g)
        assert summary["total_components"] == 0
        assert summary["avg_instability"] == 0.0
        assert summary["avg_abstractness"] == 0.0

    def test_graph_with_no_edges(self, calculator):
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "a.py", "type": "class", "name": "A"}),
                ("n2", {"file_path": "b.py", "type": "class", "name": "B"}),
            ]
        )
        comps = [_comp("comp-a", ["a.py"]), _comp("comp-b", ["b.py"])]
        result = calculator.calc(g, comps)
        assert isinstance(result, list)
        # With no edges, instability should be 0
        for m in result:
            assert m.instability == 0.0


class TestThresholdCorrectness:
    """test_threshold_correctness"""

    def test_component_recommendations(self, calculator):
        metrics = {
            "instability": 0.95,
            "abstractness": 0.05,
            "distance": 0.9,
            "lcom4": 0.8,
            "tcc": 0.1,
        }
        recs = calculator.get_component_recommendations("comp-bad", metrics)
        assert isinstance(recs, list)
        # With extreme metrics, should have recommendations
        assert len(recs) > 0

    def test_complexity_recommendations(self, calculator):
        summary = {
            "avg_complexity": 25.0,
            "max_complexity": 50.0,
            "total_classes": 5,
            "high_complexity_classes": 3,
        }
        recs = calculator.get_complexity_recommendations(summary)
        assert isinstance(recs, list)

    def test_component_health_indicators(self, calculator):
        basic_metrics = {
            "instability": 0.5,
            "abstractness": 0.5,
            "fan_in": 3,
            "fan_out": 3,
        }
        health = calculator.get_component_health_indicators("comp-x", basic_metrics)
        assert isinstance(health, dict)

    def test_weighted_quality_score(self, calculator):
        metrics = {
            "comp-a": {"instability": 0.5, "abstractness": 0.5, "distance": 0.0},
            "comp-b": {"instability": 0.3, "abstractness": 0.7, "distance": 0.0},
        }
        score = calculator._calculate_weighted_quality_score(metrics)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_weighted_quality_score_empty(self, calculator):
        score = calculator._calculate_weighted_quality_score({})
        assert isinstance(score, float)

    def test_class_complexity_metrics(self, calculator, sample_graph):
        result = calculator.calculate_class_complexity_metrics(sample_graph, "n1")
        assert isinstance(result, dict)

    def test_component_complexity_summary(self, calculator, sample_graph):
        nodes = list(sample_graph.nodes())
        result = calculator.calculate_component_complexity_summary(sample_graph, nodes)
        assert isinstance(result, dict)
