"""Unit tests for architecture metrics modules.

Covers:
- ml_confidence_calculator.py
- runtime_trace_ingestor.py
- code_metrics_calculator.py
- metrics_aggregator.py
- architecture_metrics_analyzer.py
"""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from models.architecture_models import (
    ArchitectureRuntimeEdge,
    Component,
    DetectedPattern,
    RepoConfig,
)
from processing.architecture.architecture_metrics_analyzer import (
    ArchitectureMetricsAnalyzer,
)
from processing.architecture.code_metrics_calculator import CodeMetricsCalculator
from processing.architecture.metrics_aggregator import MetricsAggregator
from processing.architecture.ml_confidence_calculator import MLConfidenceCalculator
from processing.architecture.performance_metrics_tracker import (
    ArchitectureMetricTimeSeries,
)
from processing.architecture.runtime_trace_ingestor import (
    RuntimeSpan,
    RuntimeTraceIngestor,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_component(
    cid: str,
    files: list[str] | None = None,
    deps: list[str] | None = None,
) -> Component:
    return Component(
        id=cid,
        name=cid,
        files=files or [f"{cid}.py"],
        dependencies=deps or [],
    )


def _simple_graph() -> nx.DiGraph:
    """Build a small directed graph with node attributes."""
    g = nx.DiGraph()
    g.add_node("a", file="comp_a.py", type="class", component_id="comp_a")
    g.add_node("b", file="comp_a.py", type="function", component_id="comp_a")
    g.add_node("c", file="comp_b.py", type="class", component_id="comp_b", is_abstract=True)
    g.add_node("d", file="comp_b.py", type="method", component_id="comp_b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    return g


# ============================================================================
# MLConfidenceCalculator tests
# ============================================================================


class TestMLConfidenceCalculatorInit:
    def test_init_no_config(self):
        calc = MLConfidenceCalculator(config=None)
        assert calc.initialized is False
        assert calc.models == {}

    def test_init_ml_disabled_config(self):
        cfg = RepoConfig(ml_confidence_settings={"enabled": False})
        calc = MLConfidenceCalculator(config=cfg)
        assert calc.initialized is False

    def test_is_ml_enabled_no_config(self):
        calc = MLConfidenceCalculator()
        assert calc._is_ml_enabled() is False

    def test_is_ml_enabled_no_ml_settings(self):
        cfg = RepoConfig()
        calc = MLConfidenceCalculator(config=cfg)
        assert calc._is_ml_enabled() is False

    def test_is_ml_enabled_settings_present(self):
        cfg = RepoConfig(ml_confidence_settings={"enabled": True})
        calc = MLConfidenceCalculator.__new__(MLConfidenceCalculator)
        calc.config = cfg
        assert calc._is_ml_enabled() is True

    def test_initialize_creates_dir_and_loads(self, tmp_path):
        cfg = RepoConfig(ml_confidence_settings={"enabled": True, "thresholds": {"Singleton": 0.9}})
        calc = MLConfidenceCalculator.__new__(MLConfidenceCalculator)
        calc.config = cfg
        calc.model_directory = tmp_path / "models"
        calc.models = {}
        calc.initialized = False
        calc.confidence_thresholds = {"Singleton": 0.7, "Default": 0.7}
        calc._initialize()
        assert calc.initialized is True
        assert calc.confidence_thresholds["Singleton"] == 0.9

    def test_initialize_handles_exception(self, tmp_path):
        cfg = RepoConfig(ml_confidence_settings={"enabled": True})
        calc = MLConfidenceCalculator.__new__(MLConfidenceCalculator)
        calc.config = cfg
        calc.model_directory = tmp_path / "models"
        calc.models = {}
        calc.initialized = False
        calc.confidence_thresholds = {"Default": 0.7}
        with patch.object(calc, "_load_models", side_effect=OSError("boom")):
            calc._initialize()
        assert calc.initialized is False


class TestMLConfidenceCalculatorFeatures:
    @pytest.fixture()
    def calc(self):
        c = MLConfidenceCalculator()
        return c

    def test_calculate_average_degree(self, calc):
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        result = calc._calculate_average_degree(["a", "b"], g)
        # a has degree 1 (out to b), b has degree 2 (in from a, out to c)
        assert result == 1.5

    def test_calculate_average_degree_no_valid_nodes(self, calc):
        g = nx.DiGraph()
        assert calc._calculate_average_degree(["x"], g) == 0.0

    def test_calculate_subgraph_density_single_node(self, calc):
        g = nx.DiGraph()
        g.add_node("a")
        assert calc._calculate_subgraph_density(["a"], g) == 0.0

    def test_calculate_subgraph_density_multiple(self, calc):
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        density = calc._calculate_subgraph_density(["a", "b"], g)
        assert density == pytest.approx(1.0)

    def test_calculate_clustering_empty(self, calc):
        g = nx.DiGraph()
        assert calc._calculate_clustering_coefficient(["x"], g) == 0.0

    def test_calculate_edge_counts(self, calc):
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("c", "a")
        result = calc._calculate_edge_counts(["a"], g)
        assert result == [1.0, 1.0]  # 1 predecessor (c), 1 successor (b)

    def test_calculate_edge_counts_missing_nodes(self, calc):
        g = nx.DiGraph()
        result = calc._calculate_edge_counts(["missing"], g)
        assert result == [0.0, 0.0]

    def test_calculate_internal_external_ratio(self, calc):
        g = nx.DiGraph()
        g.add_edge("a", "b")  # internal
        g.add_edge("a", "c")  # external (c not in components)
        ratio = calc._calculate_internal_external_ratio(["a", "b"], g)
        assert ratio > 0

    def test_get_text_features(self, calc):
        g = nx.DiGraph()
        g.add_node("SingletonManager", code="get instance singleton")
        features = calc._get_text_features(["SingletonManager"], g)
        assert len(features) == 13  # 12 pattern keyword groups + avg name length
        # Singleton keywords should be detected
        assert features[0] > 0  # Singleton group

    def test_get_text_features_no_code(self, calc):
        g = nx.DiGraph()
        g.add_node("plain_node")
        features = calc._get_text_features(["plain_node"], g)
        assert len(features) == 13

    def test_get_text_features_nonexistent_node(self, calc):
        g = nx.DiGraph()
        features = calc._get_text_features(["missing"], g)
        assert len(features) == 13
        # avg name length should still be computed from component list
        assert features[-1] == 7.0  # len("missing") = 7

    def test_get_structural_features(self, calc):
        g = nx.DiGraph()
        g.add_node("a", loc=100, complexity=5)
        g.add_node("b", code="line1\nline2\nline3")
        features = calc._get_structural_features(["a", "b"], g)
        assert features[0] == 2.0  # component count
        # avg size: (100 + 3) / 2 = 51.5
        assert features[1] == pytest.approx(51.5)

    def test_get_structural_features_single_node(self, calc):
        g = nx.DiGraph()
        g.add_node("a", loc=50, complexity=3)
        features = calc._get_structural_features(["a"], g)
        assert features[0] == 1.0
        assert features[3] == 0.0  # variance with single node

    def test_extract_features(self, calc):
        g = nx.DiGraph()
        g.add_node("a")
        g.add_edge("a", "a")  # self loop
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.8,
            components=["a"],
            description="test",
        )
        features = calc.extract_features(pattern, g)
        assert isinstance(features, list)
        assert all(isinstance(f, float) for f in features)

    def test_extract_features_empty_returns_default(self, calc):
        g = nx.DiGraph()
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.8,
            components=[],
            description="test",
        )
        with patch.object(calc, "_get_relationship_features", return_value=[]):
            with patch.object(calc, "_get_text_features", return_value=[]):
                with patch.object(calc, "_get_structural_features", return_value=[]):
                    result = calc.extract_features(pattern, g)
        assert result == [0.0] * 10


class TestMLConfidenceCalculatorCalculate:
    def test_calculate_confidence_not_initialized(self):
        calc = MLConfidenceCalculator()
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.8,
            components=["a"],
            description="test",
        )
        g = nx.DiGraph()
        assert calc.calculate_confidence(pattern, g) == 0.8

    def test_calculate_confidence_model_missing(self):
        calc = MLConfidenceCalculator()
        calc.initialized = True
        calc.models = {}
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.8,
            components=["a"],
            description="test",
        )
        g = nx.DiGraph()
        assert calc.calculate_confidence(pattern, g) == 0.8

    def test_calculate_confidence_with_model(self):
        calc = MLConfidenceCalculator()
        calc.initialized = True
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.9]]
        calc.models = {"Singleton": mock_model}
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.6,
            components=["a"],
            description="test",
        )
        g = nx.DiGraph()
        g.add_node("a")
        result = calc.calculate_confidence(pattern, g)
        # blended = 0.9 * 0.7 + 0.6 * 0.3 = 0.63 + 0.18 = 0.81
        assert result == pytest.approx(0.81, abs=0.01)

    def test_calculate_confidence_error_fallback(self):
        calc = MLConfidenceCalculator()
        calc.initialized = True
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = ValueError("fail")
        calc.models = {"Singleton": mock_model}
        pattern = DetectedPattern(
            pattern_type="Singleton",
            confidence=0.75,
            components=["a"],
            description="test",
        )
        g = nx.DiGraph()
        g.add_node("a")
        assert calc.calculate_confidence(pattern, g) == 0.75


class TestMLConfidenceCalculatorTrain:
    def test_train_model_empty_data(self):
        calc = MLConfidenceCalculator()
        calc.train_model("Singleton", [])
        assert "Singleton" not in calc.models

    def test_train_model_success(self, tmp_path):
        calc = MLConfidenceCalculator()
        calc.model_directory = tmp_path
        training_data = [
            ([0.1, 0.2, 0.3], 1),
            ([0.4, 0.5, 0.6], 0),
            ([0.7, 0.8, 0.9], 1),
        ]
        # Mock sklearn if not available
        mock_clf = MagicMock()
        with patch(
            "processing.architecture.ml_confidence_calculator.RandomForestClassifier",
            return_value=mock_clf,
        ):
            with patch(
                "processing.architecture.ml_confidence_calculator.SKLEARN_AVAILABLE",
                True,
            ):
                calc.train_model("Factory", training_data)
        assert "Factory" in calc.models

    def test_train_model_exception(self, tmp_path):
        calc = MLConfidenceCalculator()
        calc.model_directory = tmp_path
        training_data = [([0.1], 1)]
        with patch(
            "processing.architecture.ml_confidence_calculator.RandomForestClassifier",
            side_effect=ValueError("bad data"),
        ):
            calc.train_model("Factory", training_data)
        assert "Factory" not in calc.models


class TestMLConfidenceCalculatorLoadModels:
    def test_load_models_no_files(self, tmp_path):
        calc = MLConfidenceCalculator()
        calc.model_directory = tmp_path
        calc.confidence_thresholds = {"Singleton": 0.7, "Default": 0.7}
        calc._load_models()
        assert calc.models == {}

    def test_load_models_corrupt_file(self, tmp_path):
        calc = MLConfidenceCalculator()
        calc.model_directory = tmp_path
        calc.confidence_thresholds = {"Singleton": 0.7, "Default": 0.7}
        (tmp_path / "singleton_classifier.pkl").write_bytes(b"not a pickle")
        calc._load_models()
        assert "Singleton" not in calc.models


# ============================================================================
# RuntimeTraceIngestor tests
# ============================================================================


class TestRuntimeTraceIngestorFormatDetection:
    @pytest.fixture()
    def ingestor(self):
        return RuntimeTraceIngestor()

    def test_supported_formats(self, ingestor):
        assert "jaeger" in ingestor.supported_formats
        assert "zipkin" in ingestor.supported_formats
        assert "opentelemetry" in ingestor.supported_formats

    def test_is_jaeger_format(self, ingestor):
        data = {"data": [{"spans": []}]}
        assert ingestor._is_jaeger_format(data) is True
        assert ingestor._is_jaeger_format({"data": []}) is False
        assert ingestor._is_jaeger_format([]) is False

    def test_is_zipkin_format(self, ingestor):
        data = [{"traceID": "abc", "spanID": "123"}]
        assert ingestor._is_zipkin_format(data) is True
        assert ingestor._is_zipkin_format([]) is False
        assert ingestor._is_zipkin_format({}) is False

    def test_is_opentelemetry_format(self, ingestor):
        assert ingestor._is_opentelemetry_format({"resourceSpans": []}) is True
        assert ingestor._is_opentelemetry_format({"other": []}) is False


class TestRuntimeTraceIngestorParsing:
    @pytest.fixture()
    def ingestor(self):
        return RuntimeTraceIngestor()

    def test_create_span_from_jaeger(self, ingestor):
        span_data = {
            "spanID": "s1",
            "operationName": "GET /api",
            "startTime": 1_700_000_000_000_000,
            "duration": 5_000_000,
            "process": {"serviceName": "gateway"},
            "tags": [{"key": "http.method", "value": "GET"}],
        }
        span = ingestor._create_span_from_jaeger(span_data, "trace1")
        assert span is not None
        assert span.span_id == "s1"
        assert span.trace_id == "trace1"
        assert span.service_name == "gateway"
        assert span.duration_ms == 5000.0
        assert span.tags["http.method"] == "GET"

    def test_create_span_from_jaeger_missing_key(self, ingestor):
        result = ingestor._create_span_from_jaeger({}, "t1")
        assert result is None

    def test_create_span_from_zipkin(self, ingestor):
        span_data = {
            "id": "s1",
            "traceId": "t1",
            "name": "POST /data",
            "timestamp": 1_700_000_000_000_000,
            "duration": 2_000_000,
            "localEndpoint": {"serviceName": "data-svc"},
            "tags": {"db.type": "postgres"},
        }
        span = ingestor._create_span_from_zipkin(span_data)
        assert span is not None
        assert span.service_name == "data-svc"
        assert span.duration_ms == 2000.0

    def test_create_span_from_zipkin_missing_key(self, ingestor):
        assert ingestor._create_span_from_zipkin({}) is None

    def test_create_span_from_opentelemetry(self, ingestor):
        span_data = {
            "spanId": "s1",
            "traceId": "t1",
            "name": "rpc.call",
            "startTimeUnixNano": "1700000000000000000",
            "endTimeUnixNano": "1700000003000000000",
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "my-svc"}},
                {"key": "http.status", "value": {"intValue": 200}},
            ],
        }
        span = ingestor._create_span_from_opentelemetry(span_data)
        assert span is not None
        assert span.service_name == "my-svc"
        assert span.duration_ms == 3000.0
        assert span.tags["http.status"] == 200

    def test_create_span_from_opentelemetry_missing_key(self, ingestor):
        assert ingestor._create_span_from_opentelemetry({}) is None

    def test_parse_jaeger_traces(self, ingestor):
        data = {
            "data": [
                {
                    "traceID": "t1",
                    "spans": [
                        {
                            "spanID": "s1",
                            "operationName": "op",
                            "startTime": 1_700_000_000_000_000,
                            "duration": 1_000_000,
                            "process": {"serviceName": "svc"},
                        }
                    ],
                }
            ]
        }
        spans = ingestor._parse_jaeger_traces(data)
        assert len(spans) == 1

    def test_parse_zipkin_traces(self, ingestor):
        data = [
            {
                "id": "s1",
                "traceId": "t1",
                "name": "op",
                "timestamp": 1_700_000_000_000_000,
                "duration": 1_000_000,
                "localEndpoint": {"serviceName": "svc"},
            }
        ]
        spans = ingestor._parse_zipkin_traces(data)
        assert len(spans) == 1

    def test_parse_opentelemetry_traces(self, ingestor):
        data = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "spanId": "s1",
                                    "traceId": "t1",
                                    "name": "op",
                                    "startTimeUnixNano": "1700000000000000000",
                                    "endTimeUnixNano": "1700000001000000000",
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        spans = ingestor._parse_opentelemetry_traces(data)
        assert len(spans) == 1


class TestRuntimeTraceIngestorGraph:
    @pytest.fixture()
    def ingestor(self):
        return RuntimeTraceIngestor()

    def _make_span(self, sid, tid, svc, parent=None, op="op", dur=100.0):
        now = datetime.datetime(2024, 1, 1, 12, 0, 0)
        return RuntimeSpan(
            span_id=sid,
            trace_id=tid,
            operation_name=op,
            service_name=svc,
            start_time=now,
            end_time=now + datetime.timedelta(milliseconds=dur),
            duration_ms=dur,
            parent_span_id=parent,
        )

    def test_build_runtime_graph_no_spans(self, ingestor):
        g = ingestor._build_runtime_graph([])
        assert g.number_of_nodes() == 0

    def test_build_runtime_graph_parent_child(self, ingestor):
        spans = [
            self._make_span("s1", "t1", "gateway"),
            self._make_span("s2", "t1", "backend", parent="s1"),
        ]
        g = ingestor._build_runtime_graph(spans)
        assert g.has_edge("gateway", "backend")
        edge = g.edges["gateway", "backend"]
        assert edge["call_count"] == 1
        assert edge["runtime"] is True

    def test_build_runtime_graph_same_service(self, ingestor):
        """Parent and child in same service should NOT create an edge."""
        spans = [
            self._make_span("s1", "t1", "gateway"),
            self._make_span("s2", "t1", "gateway", parent="s1"),
        ]
        g = ingestor._build_runtime_graph(spans)
        assert g.number_of_edges() == 0

    def test_build_runtime_graph_increments_call_count(self, ingestor):
        spans = [
            self._make_span("s1", "t1", "gateway"),
            self._make_span("s2", "t1", "backend", parent="s1"),
            self._make_span("s3", "t1", "gateway"),
            self._make_span("s4", "t1", "backend", parent="s3"),
        ]
        g = ingestor._build_runtime_graph(spans)
        assert g.edges["gateway", "backend"]["call_count"] == 2

    def test_extract_runtime_edges(self, ingestor):
        g = nx.DiGraph()
        g.add_edge("svcA", "svcB", call_count=5, total_duration=500.0)
        edges = ingestor._extract_runtime_edges(g, [])
        assert len(edges) == 1
        edge = edges[0]
        assert edge.source == "svcA"
        assert edge.target == "svcB"
        assert edge.avg_duration_ms == 100.0
        assert edge.confidence == 0.5  # min(1.0, 5/10)

    def test_extract_runtime_edges_high_confidence(self, ingestor):
        g = nx.DiGraph()
        g.add_edge("a", "b", call_count=20, total_duration=200.0)
        edges = ingestor._extract_runtime_edges(g, [])
        assert edges[0].confidence == 1.0

    def test_merge_graphs_new_edge(self, ingestor):
        static = nx.DiGraph()
        static.add_node("svcA")
        static.add_node("svcB")
        runtime = nx.DiGraph()
        edge = ArchitectureRuntimeEdge(
            source="svcA",
            target="svcB",
            call_count=3,
            avg_duration_ms=50.0,
            error_rate=0.0,
            confidence=0.3,
        )
        merged = ingestor._merge_graphs(static, runtime, [edge])
        assert merged.has_edge("svcA", "svcB")
        assert merged.edges["svcA", "svcB"]["runtime_only"] is True

    def test_merge_graphs_augment_existing(self, ingestor):
        static = nx.DiGraph()
        static.add_edge("svcA", "svcB", weight=1)
        runtime = nx.DiGraph()
        edge = ArchitectureRuntimeEdge(
            source="svcA",
            target="svcB",
            call_count=10,
            avg_duration_ms=25.0,
            error_rate=0.01,
            confidence=1.0,
        )
        merged = ingestor._merge_graphs(static, runtime, [edge])
        data = merged.edges["svcA", "svcB"]
        assert data["has_runtime_data"] is True
        assert data["runtime_call_count"] == 10
        assert data["weight"] == 1  # original attr preserved

    def test_get_runtime_metrics_empty(self, ingestor):
        assert ingestor.get_runtime_metrics([]) == {}

    def test_get_runtime_metrics(self, ingestor):
        edges = [
            ArchitectureRuntimeEdge(
                source="a",
                target="b",
                call_count=10,
                avg_duration_ms=50.0,
                error_rate=0.05,
                confidence=1.0,
            ),
            ArchitectureRuntimeEdge(
                source="b",
                target="c",
                call_count=5,
                avg_duration_ms=100.0,
                error_rate=0.1,
                confidence=0.5,
            ),
        ]
        metrics = ingestor.get_runtime_metrics(edges)
        assert metrics["total_runtime_edges"] == 2
        assert metrics["total_calls"] == 15
        assert len(metrics["hottest_paths"]) <= 5
        assert len(metrics["slowest_paths"]) <= 5


class TestRuntimeTraceIngestorAsync:
    @pytest.fixture()
    def ingestor(self):
        return RuntimeTraceIngestor()

    async def test_parse_trace_file_jaeger(self, ingestor, tmp_path):
        trace_data = {
            "data": [
                {
                    "traceID": "t1",
                    "spans": [
                        {
                            "spanID": "s1",
                            "operationName": "op",
                            "startTime": 1_700_000_000_000_000,
                            "duration": 1_000_000,
                            "process": {"serviceName": "svc"},
                        }
                    ],
                }
            ]
        }
        f = tmp_path / "jaeger.json"
        f.write_text(json.dumps(trace_data))
        spans = await ingestor._parse_trace_file(f)
        assert len(spans) == 1

    async def test_parse_trace_file_unknown_format(self, ingestor, tmp_path):
        f = tmp_path / "unknown.json"
        f.write_text(json.dumps({"foo": "bar"}))
        spans = await ingestor._parse_trace_file(f)
        assert spans == []

    async def test_parse_trace_file_invalid_json(self, ingestor, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        spans = await ingestor._parse_trace_file(f)
        assert spans == []

    async def test_ingest_traces_happy_path(self, ingestor, tmp_path):
        trace_data = {
            "data": [
                {
                    "traceID": "t1",
                    "spans": [
                        {
                            "spanID": "s1",
                            "operationName": "op",
                            "startTime": 1_700_000_000_000_000,
                            "duration": 1_000_000,
                            "process": {"serviceName": "gateway"},
                        },
                        {
                            "spanID": "s2",
                            "operationName": "op2",
                            "startTime": 1_700_000_001_000_000,
                            "duration": 500_000,
                            "process": {"serviceName": "backend"},
                            "parentSpanID": "s1",
                        },
                    ],
                }
            ]
        }
        f = tmp_path / "traces.json"
        f.write_text(json.dumps(trace_data))
        static = nx.DiGraph()
        merged, edges = await ingestor.ingest_traces("codebase1", [f], static)
        assert isinstance(merged, nx.DiGraph)
        assert len(edges) >= 1

    async def test_ingest_traces_error_returns_static(self, ingestor, tmp_path):
        f = tmp_path / "missing.json"
        static = nx.DiGraph()
        static.add_node("x")
        merged, edges = await ingestor.ingest_traces("codebase1", [f], static)
        assert merged.has_node("x")
        assert edges == []


# ============================================================================
# CodeMetricsCalculator tests
# ============================================================================


class TestCodeMetricsCalculatorValidation:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_validate_empty_components(self, calc):
        g = nx.DiGraph()
        g.add_node("x")
        result = calc.validate_components_and_graph([], g)
        assert result["valid"] is False
        assert "No components" in result["errors"][0]

    def test_validate_empty_graph(self, calc):
        comp = _make_component("a")
        g = nx.DiGraph()
        result = calc.validate_components_and_graph([comp], g)
        assert result["valid"] is False
        assert "no nodes" in result["errors"][0]

    def test_validate_duplicate_ids(self, calc):
        comp1 = _make_component("a")
        comp2 = _make_component("a", files=["other.py"])
        g = nx.DiGraph()
        g.add_node("x")
        result = calc.validate_components_and_graph([comp1, comp2], g)
        assert result["valid"] is False
        assert any("Duplicate" in e for e in result["errors"])

    def test_validate_no_files_warning(self, calc):
        # Component requires at least one file, so use a real file
        comp = _make_component("a", files=["placeholder.py"])
        g = nx.DiGraph()
        g.add_node("x")
        result = calc.validate_components_and_graph([comp], g)
        assert result["valid"] is True

    def test_validate_orphaned_nodes(self, calc):
        comp = _make_component("a", files=["comp_a.py"])
        g = nx.DiGraph()
        g.add_node("n1", file="comp_a.py")
        g.add_node("n2", file="other.py")  # orphan
        result = calc.validate_components_and_graph([comp], g)
        assert result["orphaned_nodes"] == 1
        assert any("could not be mapped" in w for w in result["warnings"])


class TestCodeMetricsCalculatorMapping:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_map_nodes_to_components(self, calc):
        comp_a = _make_component("a", files=["comp_a.py"])
        comp_b = _make_component("b", files=["comp_b.py"])
        g = _simple_graph()
        mapping = calc.map_nodes_to_components([comp_a, comp_b], g)
        assert "a" in mapping
        assert "b" in mapping
        assert len(mapping["a"]) == 2  # nodes a, b
        assert len(mapping["b"]) == 2  # nodes c, d

    def test_map_nodes_no_file_attribute(self, calc):
        comp = _make_component("a", files=["file.py"])
        g = nx.DiGraph()
        g.add_node("no_file_node")  # no 'file' attr
        mapping = calc.map_nodes_to_components([comp], g)
        assert mapping["a"] == []


class TestCodeMetricsCalculatorBasicMetrics:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_basic_metrics_empty(self, calc):
        assert calc.calculate_basic_component_metrics([], nx.DiGraph()) == {}

    def test_basic_metrics(self, calc):
        comp = _make_component("comp_a", files=["comp_a.py"], deps=["comp_b"])
        g = _simple_graph()
        metrics = calc.calculate_basic_component_metrics([comp], g)
        assert "comp_a" in metrics
        m = metrics["comp_a"]
        assert m["file_count"] == 1
        assert m["node_count"] == 2
        assert m["dependency_count"] == 1
        assert m["has_nodes"] is True
        assert m["has_dependencies"] is True

    def test_analyze_node_types(self, calc):
        g = nx.DiGraph()
        g.add_node("n1", type="class")
        g.add_node("n2", type="function")
        g.add_node("n3", type="class")
        result = calc._analyze_node_types(g, ["n1", "n2", "n3"])
        assert result == {"class": 2, "function": 1}

    def test_analyze_node_types_unknown(self, calc):
        g = nx.DiGraph()
        g.add_node("n1")
        result = calc._analyze_node_types(g, ["n1"])
        assert result == {"unknown": 1}


class TestCodeMetricsCalculatorSize:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_size_metrics(self, calc):
        comp_a = _make_component("comp_a", files=["comp_a.py"])
        comp_b = _make_component("comp_b", files=["comp_b.py"])
        g = _simple_graph()
        metrics = calc.calculate_component_size_metrics([comp_a, comp_b], g)
        assert "comp_a" in metrics
        assert "comp_b" in metrics
        assert "size_score" in metrics["comp_a"]
        assert "is_large_component" in metrics["comp_a"]

    def test_size_metrics_empty(self, calc):
        assert calc.calculate_component_size_metrics([], nx.DiGraph()) == {}


class TestCodeMetricsCalculatorHealth:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_health_excellent(self, calc):
        metrics = {
            "file_count": 5,
            "node_count": 20,
            "dependency_count": 3,
            "nodes_per_file": 4.0,
            "node_types": {"class": 5, "function": 10, "method": 5},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert result["overall_health"] == "excellent"
        assert result["component_id"] == "comp"

    def test_health_no_files(self, calc):
        metrics = {
            "file_count": 0,
            "node_count": 0,
            "dependency_count": 0,
            "nodes_per_file": 0,
            "node_types": {},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert "no files" in result["issues"][0]

    def test_health_single_file(self, calc):
        metrics = {
            "file_count": 1,
            "node_count": 3,
            "dependency_count": 2,
            "nodes_per_file": 3.0,
            "node_types": {"class": 1, "function": 2},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("only one file" in i for i in result["issues"])

    def test_health_high_density(self, calc):
        metrics = {
            "file_count": 2,
            "node_count": 50,
            "dependency_count": 2,
            "nodes_per_file": 25.0,
            "node_types": {"class": 1, "function": 20, "method": 10},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("High node density" in i for i in result["issues"])

    def test_health_low_density(self, calc):
        metrics = {
            "file_count": 5,
            "node_count": 5,
            "dependency_count": 2,
            "nodes_per_file": 1.0,
            "node_types": {"class": 3, "function": 2},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("Low node density" in i for i in result["issues"])

    def test_health_high_dependencies(self, calc):
        metrics = {
            "file_count": 3,
            "node_count": 10,
            "dependency_count": 15,
            "nodes_per_file": 3.3,
            "node_types": {"class": 3, "function": 5, "method": 2},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("High dependency" in i for i in result["issues"])

    def test_health_no_deps_with_code(self, calc):
        metrics = {
            "file_count": 3,
            "node_count": 10,
            "dependency_count": 0,
            "nodes_per_file": 3.3,
            "node_types": {"class": 3, "function": 5, "method": 2},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("no dependencies" in i for i in result["issues"])

    def test_health_classes_no_methods(self, calc):
        metrics = {
            "file_count": 3,
            "node_count": 5,
            "dependency_count": 2,
            "nodes_per_file": 1.7,
            "node_types": {"class": 5},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert any("no methods" in i for i in result["issues"])

    def test_health_poor(self, calc):
        metrics = {
            "file_count": 0,
            "node_count": 0,
            "dependency_count": 15,
            "nodes_per_file": 0,
            "node_types": {},
        }
        result = calc.get_component_health_indicators("comp", metrics)
        assert result["overall_health"] == "poor"


class TestCodeMetricsCalculatorSystemOverview:
    @pytest.fixture()
    def calc(self):
        return CodeMetricsCalculator()

    def test_system_overview(self, calc):
        comp_a = _make_component("comp_a", files=["comp_a.py"], deps=["comp_b"])
        comp_b = _make_component("comp_b", files=["comp_b.py"])
        g = _simple_graph()
        overview = calc.calculate_system_overview([comp_a, comp_b], g)
        assert overview["total_components"] == 2
        assert overview["total_files"] == 2
        assert "health_distribution" in overview
        assert "validation_summary" in overview


# ============================================================================
# MetricsAggregator tests
# ============================================================================


class TestMetricsAggregatorCompareCommits:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def test_compare_no_commits(self, aggregator):
        result = aggregator.compare_metrics_across_commits(
            "comp", [], lambda c, g: {}, lambda m: 0.5
        )
        assert len(result) == 6
        for ts in result.values():
            assert len(ts.data_points) == 0

    def test_compare_single_commit(self, aggregator):
        comp = _make_component("comp", files=["f.py"])
        g = nx.DiGraph()
        ts = datetime.datetime(2024, 1, 1)
        commit_data = [("abc123", ts, [comp], g)]

        def calc_metrics(comps, graph):
            return {
                "comp": {
                    "instability": 0.5,
                    "abstractness": 0.3,
                    "distance": 0.2,
                    "lcom4": 1.0,
                    "tcc": 0.8,
                }
            }

        def calc_quality(m):
            return 0.7

        result = aggregator.compare_metrics_across_commits(
            "comp", commit_data, calc_metrics, calc_quality
        )
        assert result["instability"].data_points[0].value == 0.5

    def test_compare_component_not_found(self, aggregator):
        comp = _make_component("other", files=["f.py"])
        ts = datetime.datetime(2024, 1, 1)
        commit_data = [("abc123", ts, [comp], nx.DiGraph())]
        result = aggregator.compare_metrics_across_commits(
            "comp", commit_data, lambda c, g: {}, lambda m: 0.5
        )
        for ts_obj in result.values():
            assert len(ts_obj.data_points) == 0

    def test_compare_metrics_not_calculated(self, aggregator):
        comp = _make_component("comp", files=["f.py"])
        ts = datetime.datetime(2024, 1, 1)
        commit_data = [("abc123", ts, [comp], nx.DiGraph())]
        result = aggregator.compare_metrics_across_commits(
            "comp", commit_data, lambda c, g: {}, lambda m: 0.5
        )
        for ts_obj in result.values():
            assert len(ts_obj.data_points) == 0


class TestMetricsAggregatorEvolutionReport:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def test_empty_time_series(self, aggregator):
        result = aggregator.get_architecture_evolution_report("comp", {})
        assert result["metrics_analyzed"] == 0
        assert result["has_degradation"] is False
        assert result["overall_quality_trend"] == "unknown"

    def test_report_with_data(self, aggregator):
        ts = ArchitectureMetricTimeSeries("instability", "comp")
        for i in range(5):
            ts.add_data_point(
                timestamp=datetime.datetime(2024, 1, i + 1),
                commit_id=f"c{i}",
                value=0.3 + i * 0.05,
                raw_data={},
            )
        result = aggregator.get_architecture_evolution_report("comp", {"instability": ts})
        assert result["metrics_analyzed"] == 1
        assert result["commits_analyzed"] >= 5
        assert "instability" in result["trends"]

    def test_report_overall_trend_from_weighted_quality(self, aggregator):
        ts = ArchitectureMetricTimeSeries("weighted_quality", "comp")
        for i in range(5):
            ts.add_data_point(
                timestamp=datetime.datetime(2024, 1, i + 1),
                commit_id=f"c{i}",
                value=0.8 - i * 0.1,  # degrading
                raw_data={},
            )
        result = aggregator.get_architecture_evolution_report("comp", {"weighted_quality": ts})
        assert result["overall_quality_trend"] is not None


class TestMetricsAggregatorHistorical:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def test_get_historical_data(self, aggregator):
        ts = ArchitectureMetricTimeSeries("instability", "comp")
        ts.add_data_point(
            timestamp=datetime.datetime(2024, 1, 1),
            commit_id="c1",
            value=0.5,
        )
        ts.add_data_point(
            timestamp=datetime.datetime(2024, 1, 2),
            commit_id="c2",
            value=0.6,
        )
        result = aggregator.get_historical_data_for_component(
            "comp", {"instability": ts}, metric_name="instability"
        )
        assert "instability" in result["metrics"]

    def test_get_historical_data_all_metrics(self, aggregator):
        ts1 = ArchitectureMetricTimeSeries("instability", "comp")
        ts2 = ArchitectureMetricTimeSeries("distance", "comp")
        for t in [ts1, ts2]:
            t.add_data_point(timestamp=datetime.datetime(2024, 1, 1), commit_id="c1", value=0.5)
        result = aggregator.get_historical_data_for_component(
            "comp", {"instability": ts1, "distance": ts2}
        )
        assert "instability" in result["metrics"]
        assert "distance" in result["metrics"]


class TestMetricsAggregatorDecay:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def _make_degrading_ts(self, metric, comp_id):
        ts = ArchitectureMetricTimeSeries(metric, comp_id)
        for i in range(10):
            ts.add_data_point(
                timestamp=datetime.datetime(2024, 1, i + 1),
                commit_id=f"c{i}",
                value=0.1 + i * 0.08,
                raw_data={},
            )
        return ts

    def test_detect_decay_empty(self, aggregator):
        result = aggregator.detect_architectural_decay([], {})
        assert result["system_decay_score"] == 0.0

    def test_detect_decay_with_degradation(self, aggregator):
        ts_collection = {
            "comp1": {
                "distance": self._make_degrading_ts("distance", "comp1"),
                "instability": self._make_degrading_ts("instability", "comp1"),
            }
        }
        result = aggregator.detect_architectural_decay(["comp1"], ts_collection)
        assert result["components_analyzed"] == 1
        assert "decay_by_component" in result

    def test_detect_decay_component_missing(self, aggregator):
        result = aggregator.detect_architectural_decay(["missing"], {})
        assert result["system_decay_score"] == 0.0


class TestMetricsAggregatorSummary:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def test_summary_empty_components(self, aggregator):
        result = aggregator.generate_system_metrics_summary([], {}, lambda m: 0.0)
        assert result["total_components"] == 0

    def test_summary_with_data(self, aggregator):
        comp = _make_component("a", files=["a.py"])
        metrics = {
            "a": {
                "instability": 0.5,
                "abstractness": 0.3,
                "distance": 0.2,
                "lcom4": 1.0,
                "tcc": 0.8,
            }
        }
        result = aggregator.generate_system_metrics_summary([comp], metrics, lambda m: 0.75)
        assert result["total_components"] == 1
        assert result["avg_instability"] == 0.5
        assert result["weighted_quality_score"] == 0.75


class TestMetricsAggregatorDashboard:
    @pytest.fixture()
    def aggregator(self):
        return MetricsAggregator()

    def test_dashboard_empty(self, aggregator):
        result = aggregator.generate_component_evolution_dashboard([], {})
        assert result["summary"]["total_components"] == 0

    def test_dashboard_with_component(self, aggregator):
        ts = ArchitectureMetricTimeSeries("instability", "comp1")
        for i in range(5):
            ts.add_data_point(
                timestamp=datetime.datetime(2024, 1, i + 1),
                commit_id=f"c{i}",
                value=0.5,
                raw_data={},
            )
        ts_collection = {"comp1": {"instability": ts}}
        result = aggregator.generate_component_evolution_dashboard(["comp1"], ts_collection)
        assert result["summary"]["components_with_data"] == 1
        assert "comp1" in result["components"]


# ============================================================================
# ArchitectureMetricsAnalyzer tests
# ============================================================================


class TestArchitectureMetricsAnalyzerCalc:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_calc_empty(self, analyzer):
        result = analyzer.calc(nx.DiGraph(), [])
        assert result == []

    def test_calc_single_component(self, analyzer):
        comp = _make_component("a", files=["a.py"], deps=["b"])
        g = nx.DiGraph()
        g.add_node("n1", component_id="a", type="class")
        result = analyzer.calc(g, [comp])
        assert len(result) == 1
        assert result[0].component_id == "a"
        assert result[0].instability is not None
        assert result[0].abstractness is not None
        assert result[0].distance is not None


class TestArchitectureMetricsAnalyzerFan:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_fan_metrics(self, analyzer):
        comp_a = _make_component("a", deps=["b"])
        comp_b = _make_component("b", files=["b.py"])
        g = nx.DiGraph()
        fan_in, fan_out = analyzer._calculate_fan_metrics(comp_a, [comp_a, comp_b], g)
        assert fan_out == 1
        assert fan_in == 0

    def test_fan_in(self, analyzer):
        comp_a = _make_component("a", deps=["b"])
        comp_b = _make_component("b", files=["b.py"])
        g = nx.DiGraph()
        fan_in, fan_out = analyzer._calculate_fan_metrics(comp_b, [comp_a, comp_b], g)
        assert fan_in == 1
        assert fan_out == 0


class TestArchitectureMetricsAnalyzerInstability:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_instability_no_deps(self, analyzer):
        assert analyzer._calculate_instability(0, 0) == 0.0

    def test_instability_balanced(self, analyzer):
        assert analyzer._calculate_instability(5, 5) == 0.5

    def test_instability_all_outgoing(self, analyzer):
        assert analyzer._calculate_instability(0, 10) == 1.0

    def test_instability_all_incoming(self, analyzer):
        assert analyzer._calculate_instability(10, 0) == 0.0


class TestArchitectureMetricsAnalyzerAbstractness:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_abstractness_no_nodes(self, analyzer):
        comp = _make_component("a")
        g = nx.DiGraph()
        assert analyzer._calculate_abstractness(comp, g) == 0.0

    def test_abstractness_with_abstract(self, analyzer):
        comp = _make_component("a")
        g = nx.DiGraph()
        g.add_node("n1", component_id="a", is_abstract=True)
        g.add_node("n2", component_id="a", is_abstract=False)
        result = analyzer._calculate_abstractness(comp, g)
        assert result == 0.5

    def test_abstractness_interface(self, analyzer):
        comp = _make_component("a")
        g = nx.DiGraph()
        g.add_node("n1", component_id="a", type="interface")
        result = analyzer._calculate_abstractness(comp, g)
        assert result == 1.0

    def test_abstractness_abstract_methods(self, analyzer):
        comp = _make_component("a")
        g = nx.DiGraph()
        g.add_node("n1", component_id="a", methods=[{"is_abstract": True}])
        result = analyzer._calculate_abstractness(comp, g)
        assert result == 1.0


class TestArchitectureMetricsAnalyzerDistance:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_distance_on_main_sequence(self, analyzer):
        # A + I = 1 => distance = 0
        assert analyzer._calculate_distance(0.5, 0.5) == 0.0

    def test_distance_zone_of_pain(self, analyzer):
        # Both concrete and stable: A=0, I=0 => distance = 1
        assert analyzer._calculate_distance(0.0, 0.0) == 1.0


class TestArchitectureMetricsAnalyzerComponentMetrics:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_component_metrics_empty(self, analyzer):
        assert analyzer.calculate_component_metrics([], nx.DiGraph()) == {}

    def test_component_metrics(self, analyzer):
        comp_a = _make_component("comp_a", files=["comp_a.py"], deps=["comp_b"])
        comp_b = _make_component("comp_b", files=["comp_b.py"])
        g = _simple_graph()
        metrics = analyzer.calculate_component_metrics([comp_a, comp_b], g)
        assert "comp_a" in metrics
        assert "comp_b" in metrics
        assert "instability" in metrics["comp_a"]
        assert "fan_in" in metrics["comp_a"]


class TestArchitectureMetricsAnalyzerQuality:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_weighted_quality_empty(self, analyzer):
        assert analyzer._calculate_weighted_quality_score({}) == 0.0

    def test_weighted_quality_ideal(self, analyzer):
        metrics = {"a": {"instability": 0.5, "abstractness": 0.5, "distance": 0.0}}
        score = analyzer._calculate_weighted_quality_score(metrics)
        assert score > 0.0

    def test_weighted_quality_poor(self, analyzer):
        metrics = {"a": {"instability": 0.0, "abstractness": 0.0, "distance": 1.0}}
        score = analyzer._calculate_weighted_quality_score(metrics)
        # Score should be lower for poor architecture
        ideal = {"a": {"instability": 0.5, "abstractness": 0.5, "distance": 0.0}}
        ideal_score = analyzer._calculate_weighted_quality_score(ideal)
        assert score < ideal_score


class TestArchitectureMetricsAnalyzerDependencyPatterns:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_analyze_empty(self, analyzer):
        assert analyzer.analyze_dependency_patterns([], nx.DiGraph()) == {}

    def test_analyze_with_components(self, analyzer):
        comp_a = _make_component("comp_a", files=["comp_a.py"], deps=["comp_b"])
        comp_b = _make_component("comp_b", files=["comp_b.py"])
        g = _simple_graph()
        result = analyzer.analyze_dependency_patterns([comp_a, comp_b], g)
        assert result["total_components"] == 2
        assert "dependency_cycles" in result
        assert "main_sequence_violations" in result

    def test_analyze_dependency_cycles(self, analyzer):
        comp_a = _make_component("comp_a", files=["a.py"], deps=["comp_b"])
        comp_b = _make_component("comp_b", files=["b.py"], deps=["comp_a"])
        g = nx.DiGraph()
        g.add_node("n1", file="a.py")
        g.add_node("n2", file="b.py")
        result = analyzer.analyze_dependency_patterns([comp_a, comp_b], g)
        assert result["cycle_count"] > 0


class TestArchitectureMetricsAnalyzerRecommendations:
    @pytest.fixture()
    def analyzer(self):
        return ArchitectureMetricsAnalyzer()

    def test_no_recommendations(self, analyzer):
        metrics = {"instability": 0.5, "abstractness": 0.5, "distance": 0.0}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert result == []

    def test_zone_of_uselessness(self, analyzer):
        metrics = {"instability": 0.8, "abstractness": 0.9, "distance": 0.7}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert any("Uselessness" in r for r in result)

    def test_zone_of_pain(self, analyzer):
        metrics = {"instability": 0.1, "abstractness": 0.1, "distance": 0.8}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert any("Pain" in r for r in result)

    def test_high_instability(self, analyzer):
        metrics = {"instability": 0.9, "abstractness": 0.5, "distance": 0.1}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert any("high instability" in r for r in result)

    def test_stable_concrete(self, analyzer):
        metrics = {"instability": 0.1, "abstractness": 0.1, "distance": 0.4}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert any("stable but concrete" in r for r in result)

    def test_highly_abstract_and_unstable(self, analyzer):
        metrics = {"instability": 0.9, "abstractness": 0.9, "distance": 0.8}
        result = analyzer.get_component_recommendations("comp", metrics)
        assert any("highly abstract and unstable" in r for r in result)
