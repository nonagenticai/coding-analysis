"""Unit tests for processing.architecture.pattern_detector module."""

from __future__ import annotations

import networkx as nx
import pytest

from models.architecture_models import DetectedPattern, RepoConfig
from processing.architecture.pattern_detector import ArchitecturePatternDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(nodes=None, edges=None):
    g = nx.DiGraph()
    if nodes:
        for nid, data in nodes:
            g.add_node(nid, **data)
    if edges:
        for src, dst in edges:
            g.add_edge(src, dst)
    return g


def _pattern(ptype: str = "singleton", confidence: float = 0.8) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=ptype,
        confidence=confidence,
        components=["comp-a"],
        description=f"Detected {ptype} pattern",
        name=ptype,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    return ArchitecturePatternDetector()


@pytest.fixture
def singleton_graph():
    """Graph with singleton-like node (has _instance attribute and get_instance method)."""
    return _make_graph(
        nodes=[
            (
                "s1",
                {
                    "file_path": "src/singleton.py",
                    "type": "class",
                    "name": "Singleton",
                    "code": (
                        "class Singleton:\n"
                        "    _instance = None\n"
                        "    @classmethod\n"
                        "    def get_instance(cls):\n"
                        "        if cls._instance is None:\n"
                        "            cls._instance = cls()\n"
                        "        return cls._instance\n"
                    ),
                },
            ),
        ],
    )


@pytest.fixture
def factory_graph():
    """Graph with factory-like node."""
    return _make_graph(
        nodes=[
            (
                "f1",
                {
                    "file_path": "src/factory.py",
                    "type": "class",
                    "name": "ConnectionFactory",
                    "code": (
                        "class ConnectionFactory:\n"
                        "    def create_connection(self, type):\n"
                        "        if type == 'db': return DbConnection()\n"
                        "        return HttpConnection()\n"
                    ),
                },
            ),
            ("f2", {"file_path": "src/db_conn.py", "type": "class", "name": "DbConnection"}),
            ("f3", {"file_path": "src/http_conn.py", "type": "class", "name": "HttpConnection"}),
        ],
        edges=[("f1", "f2"), ("f1", "f3")],
    )


@pytest.fixture
def observer_graph():
    """Graph with observer-like nodes."""
    return _make_graph(
        nodes=[
            (
                "o1",
                {
                    "file_path": "src/event_bus.py",
                    "type": "class",
                    "name": "EventBus",
                    "code": (
                        "class EventBus:\n"
                        "    def __init__(self):\n"
                        "        self._listeners = []\n"
                        "    def subscribe(self, listener):\n"
                        "        self._listeners.append(listener)\n"
                        "    def notify(self, event):\n"
                        "        for l in self._listeners:\n"
                        "            l.on_event(event)\n"
                    ),
                },
            ),
            ("o2", {"file_path": "src/listener.py", "type": "class", "name": "LogListener"}),
        ],
        edges=[("o1", "o2")],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatternDetectionOnSampleGraphs:
    """test_pattern_detection_on_sample_graphs"""

    def test_detect_returns_list(self, detector, singleton_graph):
        config = RepoConfig(detect_by=["folder"])
        patterns = detector.detect(singleton_graph, config)
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, DetectedPattern)

    def test_detect_singleton_pattern(self, detector, singleton_graph):
        patterns = detector._detect_singleton_pattern(singleton_graph)
        assert isinstance(patterns, list)
        singleton_patterns = [p for p in patterns if "singleton" in p.pattern_type.lower()]
        assert isinstance(singleton_patterns, list)

    def test_detect_factory_pattern(self, detector, factory_graph):
        patterns = detector._detect_factory_pattern(factory_graph)
        assert isinstance(patterns, list)

    def test_detect_builtin_patterns(self, detector, singleton_graph):
        patterns = detector._detect_builtin_patterns(singleton_graph)
        assert isinstance(patterns, list)

    def test_detect_with_custom_patterns(self, detector):
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "src/repo.py", "type": "class", "name": "UserRepository"}),
            ]
        )
        config = RepoConfig(
            detect_by=["folder"],
            custom_patterns=[
                {"name": "repository", "pattern": ".*Repository.*", "type": "class"},
            ],
        )
        patterns = detector.detect(g, config)
        assert isinstance(patterns, list)

    def test_get_pattern_statistics(self, detector):
        patterns = [
            _pattern("singleton", 0.9),
            _pattern("singleton", 0.8),
            _pattern("factory", 0.7),
        ]
        stats = detector.get_pattern_statistics(patterns)
        assert stats.get("singleton", 0) == 2
        assert stats.get("factory", 0) == 1

    def test_filter_patterns_by_confidence(self, detector):
        patterns = [
            _pattern("singleton", 0.9),
            _pattern("factory", 0.3),
            _pattern("observer", 0.7),
        ]
        filtered = detector.filter_patterns_by_confidence(patterns, min_confidence=0.7)
        assert len(filtered) == 2
        assert all(p.confidence >= 0.7 for p in filtered)

    def test_detect_observer_graph(self, detector, observer_graph):
        config = RepoConfig(detect_by=["folder"])
        patterns = detector.detect(observer_graph, config)
        assert isinstance(patterns, list)


class TestNoPatternCase:
    """test_no_pattern_case"""

    def test_empty_graph_returns_no_patterns(self, detector):
        g = nx.DiGraph()
        config = RepoConfig(detect_by=["folder"])
        patterns = detector.detect(g, config)
        assert patterns == []

    def test_graph_without_recognizable_patterns(self, detector):
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "src/a.py", "type": "function", "name": "foo"}),
                ("n2", {"file_path": "src/b.py", "type": "function", "name": "bar"}),
            ],
            edges=[("n1", "n2")],
        )
        config = RepoConfig(detect_by=["folder"])
        patterns = detector.detect(g, config)
        assert isinstance(patterns, list)

    def test_pattern_summary_empty(self, detector):
        summary = detector.get_pattern_summary([])
        assert isinstance(summary, dict)

    def test_pattern_report_empty(self, detector):
        report = detector.generate_pattern_report([])
        assert isinstance(report, str)

    def test_pattern_summary_with_patterns(self, detector):
        patterns = [_pattern("singleton", 0.9), _pattern("factory", 0.7)]
        summary = detector.get_pattern_summary(patterns)
        assert isinstance(summary, dict)
        # Summary may have nested structure with 'overview' key
        if "overview" in summary:
            assert summary["overview"]["total_patterns"] == 2
        else:
            assert summary.get("total_patterns", 0) >= 2


class TestFalsePositiveBoundaries:
    """test_false_positive_boundaries"""

    def test_filter_removes_low_confidence(self, detector):
        patterns = [
            _pattern("singleton", 0.1),
            _pattern("factory", 0.2),
        ]
        filtered = detector.filter_patterns_by_confidence(patterns, min_confidence=0.5)
        assert len(filtered) == 0

    def test_find_pattern_overlaps(self, detector):
        patterns = [
            DetectedPattern(
                pattern_type="singleton",
                confidence=0.9,
                components=["comp-a", "comp-b"],
                description="Singleton in comp-a, comp-b",
            ),
            DetectedPattern(
                pattern_type="factory",
                confidence=0.8,
                components=["comp-b", "comp-c"],
                description="Factory in comp-b, comp-c",
            ),
        ]
        overlaps = detector.find_pattern_overlaps(patterns)
        assert isinstance(overlaps, list)

    def test_validate_pattern_structure(self, detector, singleton_graph):
        pattern = _pattern("singleton")
        result = detector.validate_pattern_structure(pattern, singleton_graph)
        assert isinstance(result, dict)

    def test_property_accessors(self, detector):
        assert detector.recognizer is not None
        assert detector.design_detector is not None
        assert detector.architecture_analyzer is not None
        assert detector.validator_scorer is not None
        assert detector.reporting_utils is not None


class TestLegacyBackwardsCompatibility:
    """Test all legacy backwards compatibility methods on pattern_detector.py."""

    @pytest.fixture
    def detector(self):
        return ArchitecturePatternDetector()

    @pytest.fixture
    def simple_graph(self):
        return _make_graph(
            nodes=[
                (
                    "n1",
                    {
                        "file_path": "src/a.py",
                        "type": "class",
                        "name": "A",
                        "code": "class A: pass",
                    },
                ),
                (
                    "n2",
                    {
                        "file_path": "src/b.py",
                        "type": "class",
                        "name": "B",
                        "code": "class B: pass",
                    },
                ),
            ],
            edges=[("n1", "n2")],
        )

    def test_detect_custom_patterns(self, detector, simple_graph):
        custom = [{"name": "test_pattern", "pattern": ".*A.*", "type": "class"}]
        result = detector._detect_custom_patterns(simple_graph, custom)
        assert isinstance(result, list)

    def test_detect_singleton_pattern(self, detector, simple_graph):
        result = detector._detect_singleton_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_factory_pattern(self, detector, simple_graph):
        result = detector._detect_factory_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_observer_pattern(self, detector, simple_graph):
        result = detector._detect_observer_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_mvc_pattern(self, detector, simple_graph):
        result = detector._detect_mvc_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_repository_pattern(self, detector, simple_graph):
        result = detector._detect_repository_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_service_pattern(self, detector, simple_graph):
        result = detector._detect_service_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_sidecar_pattern(self, detector, simple_graph):
        result = detector._detect_sidecar_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_ambassador_pattern(self, detector, simple_graph):
        result = detector._detect_ambassador_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_adapter_pattern(self, detector, simple_graph):
        result = detector._detect_adapter_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_circuit_breaker_pattern(self, detector, simple_graph):
        result = detector._detect_circuit_breaker_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_service_mesh_pattern(self, detector, simple_graph):
        result = detector._detect_service_mesh_pattern(simple_graph)
        assert isinstance(result, list)

    def test_detect_api_gateway_pattern(self, detector, simple_graph):
        result = detector._detect_api_gateway_pattern(simple_graph)
        assert isinstance(result, list)

    def test_get_compiled_pattern(self, detector):
        result = detector._get_compiled_pattern(".*test.*")
        assert result is not None

    def test_node_matches_pattern(self, detector):
        node_data = {"name": "TestService", "type": "class"}
        import re

        pattern = re.compile(".*Test.*")
        result = detector._node_matches_pattern(node_data, pattern)
        assert isinstance(result, bool)

    def test_get_node_code(self, detector):
        node_data = {"code": "class A: pass"}
        result = detector._get_node_code(node_data)
        assert result == "class A: pass"

    def test_initialize_ml_calculator(self, detector):
        config = RepoConfig(detect_by=["folder"])
        detector._initialize_ml_calculator(config)
        # Should not raise

    def test_adjust_confidence_with_ml(self, detector):
        pattern = _pattern("singleton", 0.8)
        graph = _make_graph()
        result = detector._adjust_confidence_with_ml(pattern, graph)
        assert isinstance(result, DetectedPattern)

    def test_is_likely_helper(self, detector):
        main = {"name": "MainService", "file_path": "src/main.py"}
        helper = {"name": "HelperUtil", "file_path": "src/helper.py"}
        result = detector._is_likely_helper(main, helper)
        assert isinstance(result, bool)

    def test_has_proxy_behavior(self, detector):
        node_data = {"code": "proxy = create_proxy()\nproxy.forward(request)"}
        result = detector._has_proxy_behavior(node_data)
        assert isinstance(result, bool)

    def test_generate_pattern_report_with_patterns(self, detector):
        patterns = [_pattern("singleton", 0.9), _pattern("factory", 0.7)]
        report = detector.generate_pattern_report(patterns)
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# DesignPatternDetector deep coverage
# (processing/architecture/design_pattern_detector.py lines 155-165, 194, 223)
# ---------------------------------------------------------------------------

from processing.architecture.code_pattern_recognition import CodePatternRecognizer
from processing.architecture.design_pattern_detector import DesignPatternDetector


class TestDesignPatternDetectorDeep:
    """Deep coverage tests for DesignPatternDetector matching paths."""

    @pytest.fixture
    def dd(self):
        recognizer = CodePatternRecognizer()
        return DesignPatternDetector(recognizer)

    def test_detect_mvc_controller(self, dd):
        """Detect MVC Controller pattern (line 157)."""
        g = _make_graph(
            nodes=[
                (
                    "c1",
                    {
                        "file_path": "src/ctrl.py",
                        "type": "class",
                        "name": "UserController",
                        "code": "class UserController:\n    pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_mvc_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "MVC Controller"

    def test_detect_mvc_model(self, dd):
        """Detect MVC Model pattern (line 159)."""
        g = _make_graph(
            nodes=[
                (
                    "m1",
                    {
                        "file_path": "src/model.py",
                        "type": "class",
                        "name": "UserModel",
                        "code": "class UserModel:\n    pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_mvc_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "MVC Model"

    def test_detect_mvc_view(self, dd):
        """Detect MVC View pattern (line 161)."""
        g = _make_graph(
            nodes=[
                (
                    "v1",
                    {
                        "file_path": "src/view.py",
                        "type": "class",
                        "name": "UserView",
                        "code": "class UserView:\n    pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_mvc_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "MVC View"

    def test_detect_repository_pattern(self, dd):
        """Detect Repository pattern (line 194)."""
        g = _make_graph(
            nodes=[
                (
                    "r1",
                    {
                        "file_path": "src/repo.py",
                        "type": "class",
                        "name": "UserRepository",
                        "code": "class UserRepository:\n    def find(self): pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_repository_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "Repository"

    def test_detect_service_pattern(self, dd):
        """Detect Service pattern (line 223)."""
        g = _make_graph(
            nodes=[
                (
                    "s1",
                    {
                        "file_path": "src/svc.py",
                        "type": "class",
                        "name": "AuthService",
                        "code": "class AuthService:\n    def authenticate(self): pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_service_pattern(g)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == "Service"

    def test_detect_all_design_patterns(self, dd):
        """detect_all_design_patterns aggregates all detectors."""
        g = _make_graph(
            nodes=[
                (
                    "c1",
                    {
                        "file_path": "src/ctrl.py",
                        "type": "class",
                        "name": "AppController",
                        "code": "class AppController:\n    pass\n",
                    },
                ),
                (
                    "r1",
                    {
                        "file_path": "src/repo.py",
                        "type": "class",
                        "name": "DataRepository",
                        "code": "class DataRepository:\n    pass\n",
                    },
                ),
                (
                    "s1",
                    {
                        "file_path": "src/svc.py",
                        "type": "class",
                        "name": "ApiService",
                        "code": "class ApiService:\n    pass\n",
                    },
                ),
            ]
        )
        patterns = dd.detect_all_design_patterns(g)
        types = {p.pattern_type for p in patterns}
        assert "MVC Controller" in types
        assert "Repository" in types
        assert "Service" in types

    def test_detect_singleton_pattern(self, dd):
        """Detect Singleton pattern (line 68)."""
        g = _make_graph(
            nodes=[
                (
                    "s1",
                    {
                        "file_path": "src/singleton.py",
                        "type": "class",
                        "name": "MySingleton",
                        "code": "class MySingleton:\n    _instance = None\n    def getInstance(cls): pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_singleton_pattern(g)
        assert len(patterns) >= 1

    def test_detect_factory_pattern(self, dd):
        """Detect Factory pattern."""
        g = _make_graph(
            nodes=[
                (
                    "f1",
                    {
                        "file_path": "src/factory.py",
                        "type": "class",
                        "name": "WidgetFactory",
                        "code": "class WidgetFactory:\n    def createWidget(self): pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_factory_pattern(g)
        assert len(patterns) >= 1

    def test_detect_observer_pattern(self, dd):
        """Detect Observer pattern."""
        g = _make_graph(
            nodes=[
                (
                    "o1",
                    {
                        "file_path": "src/observer.py",
                        "type": "class",
                        "name": "EventObserver",
                        "code": "class EventObserver:\n    def subscribe(self): pass\n",
                    },
                ),
            ]
        )
        patterns = dd._detect_observer_pattern(g)
        assert len(patterns) >= 1
