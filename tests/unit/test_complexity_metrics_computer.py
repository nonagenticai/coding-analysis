"""Tests for processing.architecture.complexity_metrics_computer — covers
ComplexityMetricsComputer: LCOM4, TCC, class complexity metrics, recommendations."""

from __future__ import annotations

import networkx as nx
import pytest

from processing.architecture.complexity_metrics_computer import ComplexityMetricsComputer


def _build_class_graph():
    """Build a graph with a class, methods, and fields for testing."""
    g = nx.DiGraph()
    # Class node
    g.add_node("ClassA", type="class")
    # Methods
    g.add_node("method1", type="method")
    g.add_node("method2", type="method")
    g.add_node("method3", type="method")
    # Fields
    g.add_node("field1", type="field")
    g.add_node("field2", type="field")
    # Contains edges (class -> members)
    g.add_edge("ClassA", "method1", type="contains")
    g.add_edge("ClassA", "method2", type="contains")
    g.add_edge("ClassA", "method3", type="contains")
    g.add_edge("ClassA", "field1", type="contains")
    g.add_edge("ClassA", "field2", type="contains")
    # Method accesses field edges
    g.add_edge("method1", "field1", type="accesses")
    g.add_edge("method2", "field1", type="accesses")
    g.add_edge("method2", "field2", type="accesses")
    g.add_edge("method3", "field2", type="accesses")
    # Method calls
    g.add_edge("method1", "method2", type="calls")
    return g


class TestComplexityMetricsComputer:
    """Tests for ComplexityMetricsComputer."""

    def setup_method(self):
        self.computer = ComplexityMetricsComputer()

    # -----------------------------------------------------------------------
    # _is_class_node
    # -----------------------------------------------------------------------

    def test_is_class_node_class(self):
        assert self.computer._is_class_node({"type": "class"}) is True

    def test_is_class_node_interface(self):
        assert self.computer._is_class_node({"type": "interface"}) is True

    def test_is_class_node_abstract(self):
        assert self.computer._is_class_node({"type": "abstract_class"}) is True

    def test_is_class_node_trait(self):
        assert self.computer._is_class_node({"type": "trait"}) is True

    def test_is_class_node_struct(self):
        assert self.computer._is_class_node({"type": "struct"}) is True

    def test_is_class_node_function(self):
        assert self.computer._is_class_node({"type": "function"}) is False

    def test_is_class_node_empty(self):
        assert self.computer._is_class_node({}) is False

    # -----------------------------------------------------------------------
    # _get_class_methods_and_fields
    # -----------------------------------------------------------------------

    def test_get_class_methods_and_fields(self):
        g = _build_class_graph()
        methods, fields = self.computer._get_class_methods_and_fields(g, "ClassA")
        assert set(methods) == {"method1", "method2", "method3"}
        assert set(fields) == {"field1", "field2"}

    def test_get_class_methods_and_fields_empty(self):
        g = nx.DiGraph()
        g.add_node("ClassB", type="class")
        methods, fields = self.computer._get_class_methods_and_fields(g, "ClassB")
        assert methods == []
        assert fields == []

    def test_get_class_methods_with_constructor(self):
        g = nx.DiGraph()
        g.add_node("ClassC", type="class")
        g.add_node("ctor", type="constructor")
        g.add_node("prop", type="property")
        g.add_edge("ClassC", "ctor", type="contains")
        g.add_edge("ClassC", "prop", type="contains")
        methods, fields = self.computer._get_class_methods_and_fields(g, "ClassC")
        assert methods == ["ctor"]
        assert fields == ["prop"]

    # -----------------------------------------------------------------------
    # calculate_lcom4
    # -----------------------------------------------------------------------

    def test_lcom4_no_classes(self):
        g = nx.DiGraph()
        g.add_node("func1", type="function")
        assert self.computer.calculate_lcom4(g, ["func1"]) == 0.0

    def test_lcom4_single_method_class(self):
        g = nx.DiGraph()
        g.add_node("ClassA", type="class")
        g.add_node("m1", type="method")
        g.add_edge("ClassA", "m1", type="contains")
        # One method: skipped
        assert self.computer.calculate_lcom4(g, ["ClassA", "m1"]) == 0.0

    def test_lcom4_connected_class(self):
        g = _build_class_graph()
        nodes = list(g.nodes)
        lcom4 = self.computer.calculate_lcom4(g, nodes)
        # method1 & method2 share field1, method2 & method3 share field2
        # So all connected => 1 component => LCOM4 = 0
        assert lcom4 == pytest.approx(0.0)

    def test_lcom4_disconnected_methods(self):
        g = nx.DiGraph()
        g.add_node("ClassA", type="class")
        g.add_node("m1", type="method")
        g.add_node("m2", type="method")
        g.add_node("f1", type="field")
        g.add_node("f2", type="field")
        g.add_edge("ClassA", "m1", type="contains")
        g.add_edge("ClassA", "m2", type="contains")
        g.add_edge("ClassA", "f1", type="contains")
        g.add_edge("ClassA", "f2", type="contains")
        # m1 accesses f1 only, m2 accesses f2 only — no shared fields
        g.add_edge("m1", "f1", type="accesses")
        g.add_edge("m2", "f2", type="accesses")
        nodes = list(g.nodes)
        lcom4 = self.computer.calculate_lcom4(g, nodes)
        # 2 components, 2 methods: (2-1)/(2-1) = 1.0
        assert lcom4 == pytest.approx(1.0)

    # -----------------------------------------------------------------------
    # calculate_tcc
    # -----------------------------------------------------------------------

    def test_tcc_no_classes(self):
        g = nx.DiGraph()
        g.add_node("func1", type="function")
        assert self.computer.calculate_tcc(g, ["func1"]) == 0.0

    def test_tcc_connected_class(self):
        g = _build_class_graph()
        nodes = list(g.nodes)
        tcc = self.computer.calculate_tcc(g, nodes)
        # 3 methods, max_pairs = 3
        # field1: {m1,m2} -> 1 pair, field2: {m2,m3} -> 1 pair = 2 pairs
        # TCC = 2/3
        assert tcc == pytest.approx(2 / 3)

    def test_tcc_disconnected(self):
        g = nx.DiGraph()
        g.add_node("ClassA", type="class")
        g.add_node("m1", type="method")
        g.add_node("m2", type="method")
        g.add_node("f1", type="field")
        g.add_node("f2", type="field")
        g.add_edge("ClassA", "m1", type="contains")
        g.add_edge("ClassA", "m2", type="contains")
        g.add_edge("ClassA", "f1", type="contains")
        g.add_edge("ClassA", "f2", type="contains")
        g.add_edge("m1", "f1", type="accesses")
        g.add_edge("m2", "f2", type="accesses")
        tcc = self.computer.calculate_tcc(g, list(g.nodes))
        assert tcc == pytest.approx(0.0)

    # -----------------------------------------------------------------------
    # calculate_class_complexity_metrics
    # -----------------------------------------------------------------------

    def test_class_complexity_metrics(self):
        g = _build_class_graph()
        metrics = self.computer.calculate_class_complexity_metrics(g, "ClassA")
        assert metrics["method_count"] == 3
        assert metrics["field_count"] == 2
        assert "lcom4" in metrics
        assert "tcc" in metrics
        assert "method_field_ratio" in metrics
        assert "method_coupling" in metrics
        assert "complexity_score" in metrics

    def test_class_complexity_not_a_class(self):
        g = nx.DiGraph()
        g.add_node("func1", type="function")
        metrics = self.computer.calculate_class_complexity_metrics(g, "func1")
        assert metrics == {}

    def test_class_complexity_no_fields(self):
        g = nx.DiGraph()
        g.add_node("ClassA", type="class")
        g.add_node("m1", type="method")
        g.add_node("m2", type="method")
        g.add_edge("ClassA", "m1", type="contains")
        g.add_edge("ClassA", "m2", type="contains")
        metrics = self.computer.calculate_class_complexity_metrics(g, "ClassA")
        assert metrics["field_count"] == 0
        assert metrics["method_field_ratio"] == 2  # method_count when no fields

    def test_class_complexity_with_internal_calls(self):
        g = _build_class_graph()
        metrics = self.computer.calculate_class_complexity_metrics(g, "ClassA")
        # method1 calls method2: 1 internal call out of 3 methods
        assert metrics["method_coupling"] == pytest.approx(1 / 3)

    # -----------------------------------------------------------------------
    # _calculate_class_complexity_score
    # -----------------------------------------------------------------------

    def test_complexity_score_low(self):
        score = self.computer._calculate_class_complexity_score(2, 1, 0.0, 1.0, 0.5)
        assert 0.0 <= score <= 1.0
        assert score < 0.3  # Low complexity

    def test_complexity_score_high(self):
        score = self.computer._calculate_class_complexity_score(30, 10, 0.9, 0.1, 0.1)
        assert score > 0.5  # High complexity

    def test_complexity_score_low_coupling(self):
        score = self.computer._calculate_class_complexity_score(5, 3, 0.3, 0.7, 0.1)
        assert 0.0 <= score <= 1.0

    def test_complexity_score_high_coupling(self):
        score = self.computer._calculate_class_complexity_score(5, 3, 0.3, 0.7, 0.9)
        assert 0.0 <= score <= 1.0

    def test_complexity_score_good_coupling(self):
        score = self.computer._calculate_class_complexity_score(5, 3, 0.3, 0.7, 0.5)
        assert 0.0 <= score <= 1.0

    # -----------------------------------------------------------------------
    # calculate_component_complexity_summary
    # -----------------------------------------------------------------------

    def test_component_summary_no_classes(self):
        g = nx.DiGraph()
        g.add_node("func1", type="function")
        summary = self.computer.calculate_component_complexity_summary(g, ["func1"])
        assert summary["class_count"] == 0

    def test_component_summary_with_class(self):
        g = _build_class_graph()
        nodes = list(g.nodes)
        summary = self.computer.calculate_component_complexity_summary(g, nodes)
        assert summary["class_count"] == 1
        assert summary["total_methods"] == 3
        assert summary["total_fields"] == 2
        assert "avg_lcom4" in summary
        assert "avg_tcc" in summary
        assert "class_metrics" in summary

    # -----------------------------------------------------------------------
    # get_complexity_recommendations
    # -----------------------------------------------------------------------

    def test_recommendations_high_lcom4(self):
        summary = {
            "avg_lcom4": 0.8,
            "avg_tcc": 0.5,
            "avg_complexity_score": 0.3,
            "complex_classes": [],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert any("LCOM4" in r for r in recs)

    def test_recommendations_low_tcc(self):
        summary = {
            "avg_lcom4": 0.2,
            "avg_tcc": 0.1,
            "avg_complexity_score": 0.3,
            "complex_classes": [],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert any("TCC" in r for r in recs)

    def test_recommendations_high_complexity(self):
        summary = {
            "avg_lcom4": 0.2,
            "avg_tcc": 0.8,
            "avg_complexity_score": 0.8,
            "complex_classes": [],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert any("complexity score" in r.lower() for r in recs)

    def test_recommendations_complex_classes(self):
        summary = {
            "avg_lcom4": 0.2,
            "avg_tcc": 0.8,
            "avg_complexity_score": 0.3,
            "complex_classes": ["A", "B"],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert any("A" in r for r in recs)

    def test_recommendations_complex_classes_many(self):
        summary = {
            "avg_lcom4": 0.2,
            "avg_tcc": 0.8,
            "avg_complexity_score": 0.3,
            "complex_classes": [f"C{i}" for i in range(10)],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert any("more" in r for r in recs)

    def test_recommendations_all_good(self):
        summary = {
            "avg_lcom4": 0.2,
            "avg_tcc": 0.8,
            "avg_complexity_score": 0.3,
            "complex_classes": [],
        }
        recs = self.computer.get_complexity_recommendations(summary)
        assert recs == []
