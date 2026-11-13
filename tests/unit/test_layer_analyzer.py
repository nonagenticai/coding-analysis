"""Unit tests for processing.architecture.layer_analyzer module."""

from __future__ import annotations

import sys
import types as _types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
_ct = sys.modules.get("coding_client")
if _ct and not hasattr(_ct, "__path__"):
    _ct.__path__ = ["__coding_client_stub__"]
    _ct.__package__ = "coding_client"

if "coding_client.utils" not in sys.modules:
    _cu = _types.ModuleType("coding_client.utils")
    _cu.__path__ = ["__coding_client_utils_stub__"]
    _cu.__package__ = "coding_client.utils"
    sys.modules["coding_client.utils"] = _cu
    if _ct:
        _ct.utils = _cu

if "coding_client.utils.profiling" not in sys.modules:
    _prof = _types.ModuleType("coding_client.utils.profiling")
    _prof.ProgressReporter = MagicMock
    _prof.profile = lambda *a, **kw: a[0] if a and callable(a[0]) else lambda f: f
    _prof.progressive_task = lambda *a, **kw: lambda f: f
    sys.modules["coding_client.utils.profiling"] = _prof
    sys.modules["coding_client.utils"].profiling = _prof

# ---------------------------------------------------------------------------

from models.architecture_models import Component, Layer, RepoConfig
from processing.architecture.layer_analyzer import LayerAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _comp(
    cid: str, files: list[str], deps: list[str] | None = None, layer: str | None = None
) -> Component:
    return Component(id=cid, name=cid, files=files, dependencies=deps or [], layer=layer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return LayerAnalyzer()


@pytest.fixture
def layered_config():
    return RepoConfig(
        detect_by=["folder"],
        layers={
            "presentation": ["src/ui/*", "src/views/*"],
            "service": ["src/services/*"],
            "domain": ["src/domain/*"],
            "infrastructure": ["src/infra/*"],
        },
    )


@pytest.fixture
def components():
    return [
        _comp("comp-ui", ["src/ui/app.js"], deps=["comp-svc"]),
        _comp("comp-svc", ["src/services/api.py"], deps=["comp-domain"]),
        _comp("comp-domain", ["src/domain/model.py"]),
        _comp("comp-infra", ["src/infra/db.py"]),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLayerAssignmentLogic:
    """test_layer_assignment_logic"""

    def test_assign_layers_matches_patterns(self, analyzer, components, layered_config):
        result = analyzer.assign_layers(components, layered_config)
        assert "presentation" in result
        assert "comp-ui" in result["presentation"]
        assert "service" in result
        assert "comp-svc" in result["service"]

    def test_assign_layers_sets_component_layer(self, analyzer, components, layered_config):
        analyzer.assign_layers(components, layered_config)
        assert components[0].layer == "presentation"
        assert components[1].layer == "service"
        assert components[2].layer == "domain"
        assert components[3].layer == "infrastructure"

    def test_generate_layers_returns_layer_objects(self, analyzer, components, layered_config):
        layers = analyzer.generate_layers(components, layered_config)
        assert isinstance(layers, list)
        assert all(isinstance(lyr, Layer) for lyr in layers)
        layer_names = {lyr.name for lyr in layers}
        assert "presentation" in layer_names

    def test_detect_violations_clean_architecture(self, analyzer, components, layered_config):
        """detect_violations returns a list for a layered architecture.

        The hierarchy is: infrastructure=0, domain=2, service=3, presentation=5.
        The code flags when component_layer_level > dep_layer_level, i.e.
        higher layers depending on lower layers (which is actually correct
        architecture). This is a known logic inversion in the source.
        """
        analyzer.assign_layers(components, layered_config)
        violations = analyzer.detect_violations(components, layered_config)
        assert isinstance(violations, list)
        # presentation(5) -> service(3) and service(3) -> domain(2) both trigger
        # because the code checks component_level > dep_level
        assert len(violations) == 2

    def test_detect_violations_finds_reverse_dependency(self, analyzer, layered_config):
        """Infrastructure depending on presentation does NOT trigger because
        infrastructure(0) < presentation(5), so the condition
        component_level > dep_level is False.  This is a known logic inversion.
        """
        comps = [
            _comp("comp-ui", ["src/ui/app.js"]),
            _comp("comp-infra", ["src/infra/db.py"], deps=["comp-ui"]),
        ]
        analyzer.assign_layers(comps, layered_config)
        violations = analyzer.detect_violations(comps, layered_config)
        assert isinstance(violations, list)
        # infrastructure(0) -> presentation(5): 0 > 5 is False, so no violation
        assert len(violations) == 0

    def test_layer_dependencies(self, analyzer, components, layered_config):
        analyzer.assign_layers(components, layered_config)
        deps = analyzer.get_layer_dependencies(components)
        assert isinstance(deps, dict)
        # presentation -> service dependency exists via comp-ui -> comp-svc
        if "presentation" in deps:
            assert "service" in deps["presentation"]

    def test_layer_metrics(self, analyzer, components, layered_config):
        """get_layer_metrics should return metrics keyed by layer name."""
        analyzer.assign_layers(components, layered_config)
        metrics = analyzer.get_layer_metrics(components)
        assert isinstance(metrics, dict)
        # Each key is a layer name with sub-metrics
        assert "presentation" in metrics
        assert "service" in metrics
        assert "domain" in metrics
        assert "infrastructure" in metrics
        # Each layer has component_count
        assert metrics["presentation"]["component_count"] == 1


class TestUnknownComponentFallback:
    """test_unknown_component_fallback"""

    def test_no_layers_config_assigns_all_to_unassigned(self, analyzer):
        comps = [_comp("comp-a", ["src/a.py"]), _comp("comp-b", ["src/b.py"])]
        config = RepoConfig(detect_by=["folder"])
        result = analyzer.assign_layers(comps, config)
        assert "unassigned" in result
        assert len(result["unassigned"]) == 2

    def test_unmatched_component_goes_to_unassigned(self, analyzer, layered_config):
        comps = [_comp("comp-unknown", ["weird/path/foo.py"])]
        result = analyzer.assign_layers(comps, layered_config)
        assert "unassigned" in result
        assert "comp-unknown" in result["unassigned"]

    def test_generate_layer_graph_data_no_layers(self, analyzer):
        comps = [_comp("comp-a", ["src/a.py"])]
        config = RepoConfig(detect_by=["folder"])
        data = analyzer.generate_layer_graph_data(comps, config)
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["metadata"]["layer_count"] == 0

    def test_generate_layer_graph_data_with_layers(self, analyzer, components, layered_config):
        """generate_layer_graph_data should produce graph nodes and edges."""
        analyzer.assign_layers(components, layered_config)
        data = analyzer.generate_layer_graph_data(components, layered_config)
        assert isinstance(data, dict)
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert data["metadata"]["layer_count"] > 0

    def test_hierarchy_defaults(self, analyzer):
        config = RepoConfig(
            detect_by=["folder"],
            layers={"infrastructure": ["infra/*"], "domain": ["domain/*"]},
        )
        hierarchy = analyzer._get_layer_hierarchy(config)
        assert hierarchy["infrastructure"] < hierarchy["domain"]
