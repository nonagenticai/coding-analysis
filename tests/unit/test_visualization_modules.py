"""Unit tests for visualization_modules package.

Covers: chart_generation, export_rendering, export_utilities,
        data_utilities, shared_models, configuration, graph_visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.visualization_modules.chart_generation import ChartGenerator
from app.api.endpoints.visualization_modules.configuration import (
    VisualizationConfig,
    get_visualization_config,
)
from app.api.endpoints.visualization_modules.data_utilities import (
    parse_files,
    prepare_chart_data,
    transform_graph_data,
    validate_file_paths,
    validate_visualization_request,
    validate_visualization_type,
)
from app.api.endpoints.visualization_modules.export_rendering import (
    VisualizationExporter,
)
from app.api.endpoints.visualization_modules.export_utilities import (
    ExportUtilities,
    export_to_format,
    get_supported_formats,
    validate_export_format,
)
from app.api.endpoints.visualization_modules.shared_models import (
    ChartType,
    DependencyVisualizerServiceVisualizationEdge,
    DependencyVisualizerServiceVisualizationNode,
    ExportFormat,
    VisualizationCapabilitiesResponse,
    VisualizationRequest,
    VisualizationResult,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight data objects to mimic graph nodes / edges
# ---------------------------------------------------------------------------


@dataclass
class FakeNode:
    id: str
    label: str
    type: str
    language: str
    complexity: float = 5.0
    importance: float = 0.5
    file_path: str | None = None
    color: str | None = None
    size: int = 1
    group: str | None = None
    shape: str = "box"
    properties: dict = field(default_factory=dict)


@dataclass
class FakeEdge:
    id: str
    source: str
    target: str
    type: str
    label: str | None = None
    weight: float = 1.0
    color: str | None = None
    style: str = "solid"
    properties: dict = field(default_factory=dict)


def _make_nodes(n: int = 3, **overrides) -> list[FakeNode]:
    nodes = []
    langs = ["python", "java", "typescript"]
    types_ = ["class", "function", "module"]
    for i in range(n):
        kw = {
            "id": f"node_{i}",
            "label": f"Node{i}",
            "type": types_[i % len(types_)],
            "language": langs[i % len(langs)],
            "complexity": float(i + 1),
            "file_path": f"/src/file_{i}.py",
        }
        kw.update(overrides)
        nodes.append(FakeNode(**kw))
    return nodes


def _make_edges(n: int = 2, **overrides) -> list[FakeEdge]:
    edges = []
    for i in range(n):
        kw = {
            "id": f"edge_{i}",
            "source": f"node_{i}",
            "target": f"node_{i + 1}",
            "type": "dependency",
        }
        kw.update(overrides)
        edges.append(FakeEdge(**kw))
    return edges


# ============================================================================
# ChartGenerator tests (chart_generation.py)
# ============================================================================


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    def setup_method(self):
        self.gen = ChartGenerator()

    def test_supported_chart_types(self):
        assert "complexity_heatmap" in self.gen.supported_chart_types
        assert "dependency_matrix" in self.gen.supported_chart_types
        assert "language_distribution" in self.gen.supported_chart_types
        assert "call_hierarchy" in self.gen.supported_chart_types
        assert "architectural_layers" in self.gen.supported_chart_types
        assert len(self.gen.supported_chart_types) == 5

    # -- complexity heatmap --------------------------------------------------

    async def test_generate_complexity_heatmap_happy_path(self):
        nodes = _make_nodes(3)
        result = await self.gen.generate_complexity_heatmap(nodes)
        assert result["type"] == "complexity_heatmap"
        assert len(result["data"]) == 3
        # sorted descending by complexity
        assert result["data"][0]["complexity"] >= result["data"][1]["complexity"]
        assert result["options"]["color_scale"] == "RdYlBu"
        assert result["layout"]["width"] == 800

    async def test_generate_complexity_heatmap_custom_options(self):
        nodes = _make_nodes(2)
        options = {"color_scale": "Viridis", "max_items": 10, "width": 1200}
        result = await self.gen.generate_complexity_heatmap(nodes, options)
        assert result["options"]["color_scale"] == "Viridis"
        assert result["options"]["max_items"] == 10
        assert result["layout"]["width"] == 1200

    async def test_generate_complexity_heatmap_empty_nodes(self):
        result = await self.gen.generate_complexity_heatmap([])
        assert result["data"] == []
        assert result["type"] == "complexity_heatmap"

    async def test_generate_complexity_heatmap_nodes_without_attrs(self):
        """Nodes missing complexity/language are skipped."""

        class Bare:
            id = "x"
            label = "X"

        result = await self.gen.generate_complexity_heatmap([Bare()])
        assert result["data"] == []

    # -- dependency matrix ---------------------------------------------------

    async def test_generate_dependency_matrix_happy_path(self):
        nodes = _make_nodes(3)
        edges = [FakeEdge("e0", "node_0", "node_1", "dependency", weight=2)]
        result = await self.gen.generate_dependency_matrix(nodes, edges)
        assert result["type"] == "dependency_matrix"
        matrix = result["data"]["matrix"]
        assert len(matrix) == 3
        assert matrix[0][1] == 2  # node_0 -> node_1
        assert matrix[0][0] == 0  # no self-edge

    async def test_generate_dependency_matrix_no_edges(self):
        nodes = _make_nodes(2)
        result = await self.gen.generate_dependency_matrix(nodes, [])
        matrix = result["data"]["matrix"]
        assert all(cell == 0 for row in matrix for cell in row)

    async def test_generate_dependency_matrix_custom_options(self):
        nodes = _make_nodes(2)
        result = await self.gen.generate_dependency_matrix(
            nodes, [], options={"show_labels": False, "cell_size": 30}
        )
        assert result["options"]["show_labels"] is False
        assert result["layout"]["cell_size"] == 30

    # -- language distribution -----------------------------------------------

    async def test_generate_language_distribution_happy_path(self):
        nodes = _make_nodes(4)
        result = await self.gen.generate_language_distribution(nodes)
        assert result["type"] == "language_distribution"
        assert "python" in result["data"]["counts"]
        assert len(result["data"]["languages"]) > 0
        assert result["options"]["chart_type"] == "pie"

    async def test_generate_language_distribution_empty_nodes(self):
        result = await self.gen.generate_language_distribution([])
        assert result["data"]["counts"] == {}
        assert result["data"]["average_complexity"] == {}

    async def test_generate_language_distribution_custom_options(self):
        nodes = _make_nodes(2)
        result = await self.gen.generate_language_distribution(
            nodes, options={"chart_type": "bar", "width": 700}
        )
        assert result["options"]["chart_type"] == "bar"
        assert result["layout"]["width"] == 700

    # -- call hierarchy ------------------------------------------------------

    async def test_generate_call_hierarchy_happy_path(self):
        nodes = _make_nodes(3)
        edges = [
            FakeEdge("e0", "node_0", "node_1", "call"),
            FakeEdge("e1", "node_0", "node_2", "invoke"),
        ]
        result = await self.gen.generate_call_hierarchy(nodes, edges)
        assert result["type"] == "call_hierarchy"
        assert result["data"]["root_node"] == "node_0"
        tree = result["data"]["tree"]
        assert tree["id"] == "node_0"
        assert len(tree["children"]) == 2

    async def test_generate_call_hierarchy_explicit_root(self):
        nodes = _make_nodes(3)
        edges = [FakeEdge("e0", "node_0", "node_1", "call")]
        result = await self.gen.generate_call_hierarchy(nodes, edges, root_node_id="node_1")
        assert result["data"]["root_node"] == "node_1"

    async def test_generate_call_hierarchy_no_call_edges(self):
        nodes = _make_nodes(2)
        edges = [FakeEdge("e0", "node_0", "node_1", "import")]  # not a call edge
        result = await self.gen.generate_call_hierarchy(nodes, edges)
        assert result["data"]["tree"] == {}
        assert result["data"]["root_node"] is None

    async def test_generate_call_hierarchy_cyclic(self):
        """Cycles are broken by the visited set."""
        nodes = _make_nodes(2)
        edges = [
            FakeEdge("e0", "node_0", "node_1", "call"),
            FakeEdge("e1", "node_1", "node_0", "call"),
        ]
        result = await self.gen.generate_call_hierarchy(nodes, edges, root_node_id="node_0")
        tree = result["data"]["tree"]
        assert tree["id"] == "node_0"
        # node_1's subtree should not recurse back to node_0 with children
        child = tree["children"][0]
        assert child["id"] == "node_1"
        # node_0 reappears but with no children (visited guard)
        grandchild = child["children"][0]
        assert grandchild["children"] == []

    # -- architectural layers ------------------------------------------------

    async def test_generate_architectural_layers_happy_path(self):
        nodes = [
            FakeNode(
                "n0", "UserController", "class", "java", file_path="/src/controller/user.java"
            ),
            FakeNode("n1", "UserService", "class", "java", file_path="/src/service/user.java"),
            FakeNode(
                "n2", "UserRepository", "class", "java", file_path="/src/repository/user.java"
            ),
            FakeNode("n3", "AppConfig", "class", "java", file_path="/src/config/app.java"),
            FakeNode("n4", "SomeHelper", "class", "java", file_path="/src/unknown/helper.java"),
        ]
        edges = [
            FakeEdge("e0", "n0", "n1", "dependency"),
            FakeEdge("e1", "n1", "n2", "dependency"),
        ]
        result = await self.gen.generate_architectural_layers(nodes, edges)
        assert result["type"] == "architectural_layers"
        layers = result["data"]["layers"]
        assert any(n["id"] == "n0" for n in layers["presentation"])
        assert any(n["id"] == "n1" for n in layers["service"])
        assert any(n["id"] == "n2" for n in layers["data"])
        assert any(n["id"] == "n3" for n in layers["infrastructure"])
        # helper label contains "helper", so it goes in infrastructure
        # n4 -> file_path doesn't match any layer, but label might
        deps = result["data"]["dependencies"]
        assert "presentation->service" in deps

    async def test_generate_architectural_layers_uncategorized(self):
        nodes = [FakeNode("n0", "Xyz", "class", "java", file_path="/src/xyz.java")]
        result = await self.gen.generate_architectural_layers(nodes, [])
        assert len(result["data"]["uncategorized"]) == 1

    async def test_generate_architectural_layers_custom_patterns(self):
        nodes = [FakeNode("n0", "Router", "class", "go", file_path="/src/router.go")]
        opts = {"layer_patterns": {"api": ["router"]}}
        result = await self.gen.generate_architectural_layers(nodes, [], options=opts)
        assert any(n["id"] == "n0" for n in result["data"]["layers"]["api"])


# ============================================================================
# ExportUtilities tests (export_utilities.py)
# ============================================================================


class TestExportUtilities:
    """Tests for ExportUtilities helper class."""

    def test_get_language_color_known(self):
        assert ExportUtilities.get_language_color("python") == "#4ecdc4"
        assert ExportUtilities.get_language_color("Java") == "#ff6b6b"
        assert ExportUtilities.get_language_color("RUST") == "#fd79a8"

    def test_get_language_color_unknown(self):
        color = ExportUtilities.get_language_color("brainfuck")
        assert color.startswith("#")
        assert len(color) == 7

    def test_get_cytoscape_style(self):
        styles = ExportUtilities.get_cytoscape_style()
        assert isinstance(styles, list)
        assert len(styles) >= 3
        selectors = [s["selector"] for s in styles]
        assert "node" in selectors
        assert "edge" in selectors

    def test_get_node_attributes_simple(self):
        node = FakeNode("n0", "X", "class", "python", complexity=5)
        attrs = ExportUtilities.get_node_attributes(node, "#abc")
        assert 'fillcolor="#abc"' in attrs
        assert "penwidth" not in attrs  # complexity <= 10

    def test_get_node_attributes_high_complexity(self):
        node = FakeNode("n0", "X", "class", "python", complexity=15)
        attrs = ExportUtilities.get_node_attributes(node, "#abc")
        assert "penwidth=3" in attrs

    def test_get_node_attributes_custom_shape(self):
        node = FakeNode("n0", "X", "class", "python")
        node.shape = "ellipse"
        attrs = ExportUtilities.get_node_attributes(node, "#abc")
        assert "shape=ellipse" in attrs

    def test_get_edge_attributes_empty(self):
        edge = FakeEdge("e0", "a", "b", "dep", weight=1)
        edge.color = None
        edge.style = "solid"
        attrs = ExportUtilities.get_edge_attributes(edge)
        assert attrs == ""

    def test_get_edge_attributes_colored_heavy(self):
        edge = FakeEdge("e0", "a", "b", "dep", weight=3, color="#ff0000", style="dashed")
        attrs = ExportUtilities.get_edge_attributes(edge)
        assert 'color="#ff0000"' in attrs
        assert "penwidth=3" in attrs
        assert "style=dashed" in attrs

    def test_get_edge_attributes_weight_capped_at_5(self):
        edge = FakeEdge("e0", "a", "b", "dep", weight=99)
        attrs = ExportUtilities.get_edge_attributes(edge)
        assert "penwidth=5" in attrs

    def test_sanitize_mermaid_id(self):
        assert ExportUtilities.sanitize_mermaid_id("a.b/c-d") == "a_b_c_d"
        assert ExportUtilities.sanitize_mermaid_id("simple") == "simple"

    def test_get_mermaid_styling(self):
        nodes = _make_nodes(3)
        styling = ExportUtilities.get_mermaid_styling(nodes)
        assert len(styling) > 0
        assert any("classDef" in line for line in styling)
        assert any("class " in line for line in styling)

    def test_convert_nodes_to_dict(self):
        nodes = _make_nodes(2)
        result = ExportUtilities.convert_nodes_to_dict(nodes)
        assert len(result) == 2
        assert result[0]["id"] == "node_0"
        assert result[0]["language"] == "python"
        assert "complexity" in result[0]

    def test_convert_edges_to_dict(self):
        edges = _make_edges(2)
        result = ExportUtilities.convert_edges_to_dict(edges)
        assert len(result) == 2
        assert result[0]["source"] == "node_0"
        assert result[0]["type"] == "dependency"

    # -- to_dot, to_mermaid, to_graphml (object-based) -----------------------

    def test_to_dot_with_objects(self):
        utils = ExportUtilities()
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        dot = utils.to_dot(nodes, edges, "TestGraph")
        assert 'digraph "TestGraph"' in dot
        assert '"node_0"' in dot
        assert '"node_0" -> "node_1"' in dot

    def test_to_dot_with_dicts(self):
        utils = ExportUtilities()
        nodes = [{"id": "a", "label": "A"}]
        edges = [{"source": "a", "target": "b"}]
        dot = utils.to_dot(nodes, edges)
        assert '"a" [label="A"]' in dot
        assert '"a" -> "b"' in dot

    def test_to_mermaid_with_objects(self):
        utils = ExportUtilities()
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        mermaid = utils.to_mermaid(nodes, edges)
        assert "graph LR" in mermaid
        assert "node_0[Node0]" in mermaid
        assert "node_0 --> node_1" in mermaid

    def test_to_mermaid_with_dicts(self):
        utils = ExportUtilities()
        nodes = [{"id": "a.b", "label": "AB"}]
        edges = [{"source": "a.b", "target": "c-d"}]
        mermaid = utils.to_mermaid(nodes, edges)
        assert "a_b[AB]" in mermaid
        assert "a_b --> c_d" in mermaid

    def test_to_graphml(self):
        utils = ExportUtilities()
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        graphml = utils.to_graphml(nodes, edges, "G")
        assert "<?xml version" in graphml
        assert '<node id="node_0"/>' in graphml
        assert '<edge id="e0" source="node_0" target="node_1"/>' in graphml


# -- Module-level functions --------------------------------------------------


class TestExportUtilitiesFunctions:
    """Tests for module-level export_utilities functions."""

    def test_get_supported_formats(self):
        fmts = get_supported_formats()
        assert "json" in fmts
        assert "dot" in fmts
        assert "mermaid" in fmts
        assert "graphml" in fmts
        assert "png" in fmts

    def test_validate_export_format_valid(self):
        assert validate_export_format("json") is True
        assert validate_export_format("DOT") is True
        assert validate_export_format("Mermaid") is True

    def test_validate_export_format_invalid(self):
        assert validate_export_format("html") is False
        assert validate_export_format("") is False

    def test_export_to_format_json(self):
        data = {"nodes": [], "edges": []}
        result = export_to_format(data, "json")
        assert result is data  # returns as-is

    def test_export_to_format_dot(self):
        data = {"nodes": [{"id": "a", "label": "A"}], "edges": [], "title": "T"}
        result = export_to_format(data, "dot")
        assert result["format"] == "dot"
        assert "digraph" in result["content"]

    def test_export_to_format_mermaid(self):
        data = {"nodes": [{"id": "a", "label": "A"}], "edges": []}
        result = export_to_format(data, "mermaid")
        assert result["format"] == "mermaid"
        assert "graph LR" in result["content"]

    def test_export_to_format_graphml(self):
        data = {"nodes": [{"id": "a"}], "edges": [], "title": "G"}
        result = export_to_format(data, "graphml")
        assert result["format"] == "graphml"
        assert "<graphml" in result["content"]

    def test_export_to_format_unsupported(self):
        data = {"nodes": []}
        result = export_to_format(data, "png")
        assert result["format"] == "png"
        assert "client-side rendering" in result["message"]


# ============================================================================
# VisualizationExporter tests (export_rendering.py)
# ============================================================================


class TestVisualizationExporter:
    """Tests for VisualizationExporter class."""

    def setup_method(self):
        self.exporter = VisualizationExporter()

    def test_supported_formats(self):
        assert "cytoscape" in self.exporter.supported_formats
        assert "d3" in self.exporter.supported_formats
        assert "graphviz" in self.exporter.supported_formats
        assert "mermaid" in self.exporter.supported_formats
        assert "json" in self.exporter.supported_formats

    # -- Cytoscape -----------------------------------------------------------

    async def test_export_to_cytoscape_happy(self):
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        raw = await self.exporter.export_to_cytoscape(nodes, edges)
        data = json.loads(raw)
        assert "elements" in data
        assert len(data["elements"]["nodes"]) == 2
        assert len(data["elements"]["edges"]) == 1
        assert "style" in data

    async def test_export_to_cytoscape_with_styling(self):
        node = FakeNode("n0", "X", "class", "python", color="#ff0000", size=3)
        edge = FakeEdge("e0", "n0", "n1", "dep", color="#00ff00", style="dashed")
        raw = await self.exporter.export_to_cytoscape([node], [edge])
        data = json.loads(raw)
        n = data["elements"]["nodes"][0]
        assert n["style"]["background-color"] == "#ff0000"
        assert n["style"]["width"] == 60  # 3 * 20
        e = data["elements"]["edges"][0]
        assert e["style"]["line-color"] == "#00ff00"
        assert e["style"]["line-style"] == "dashed"

    async def test_export_to_cytoscape_empty(self):
        raw = await self.exporter.export_to_cytoscape([], [])
        data = json.loads(raw)
        assert data["elements"]["nodes"] == []
        assert data["elements"]["edges"] == []

    # -- D3 ------------------------------------------------------------------

    async def test_export_to_d3_happy(self):
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        raw = await self.exporter.export_to_d3(nodes, edges)
        data = json.loads(raw)
        assert len(data["nodes"]) == 2
        assert len(data["links"]) == 1
        assert data["metadata"]["node_count"] == 2
        assert data["metadata"]["link_count"] == 1

    async def test_export_to_d3_node_fields(self):
        node = FakeNode("n0", "Lbl", "function", "go", group="grp")
        raw = await self.exporter.export_to_d3([node], [])
        data = json.loads(raw)
        n = data["nodes"][0]
        assert n["name"] == "Lbl"
        assert n["group"] == "grp"

    # -- Graphviz ------------------------------------------------------------

    async def test_export_to_graphviz_digraph(self):
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        dot = await self.exporter.export_to_graphviz(nodes, edges)
        assert dot.startswith("digraph dependency_graph")
        assert "rankdir=TB" in dot
        assert "->" in dot

    async def test_export_to_graphviz_graph_type(self):
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        dot = await self.exporter.export_to_graphviz(
            nodes, edges, options={"graph_type": "graph", "group_by_language": False}
        )
        assert dot.startswith("graph dependency_graph")
        assert "--" in dot

    async def test_export_to_graphviz_no_grouping(self):
        nodes = _make_nodes(2)
        dot = await self.exporter.export_to_graphviz(
            nodes, [], options={"group_by_language": False}
        )
        assert "subgraph" not in dot

    async def test_export_to_graphviz_with_grouping(self):
        nodes = _make_nodes(3)
        dot = await self.exporter.export_to_graphviz(nodes, [])
        assert "subgraph cluster_" in dot

    # -- Mermaid -------------------------------------------------------------

    async def test_export_to_mermaid_happy(self):
        nodes = _make_nodes(3)
        edges = _make_edges(2)
        mermaid = await self.exporter.export_to_mermaid(nodes, edges)
        assert mermaid.startswith("graph TD")
        assert "node_0" in mermaid

    async def test_export_to_mermaid_shapes(self):
        nodes = [
            FakeNode("n0", "Cls", "class", "java"),
            FakeNode("n1", "Fn", "function", "python"),
            FakeNode("n2", "Mod", "module", "go"),
            FakeNode("n3", "Other", "variable", "rust"),
        ]
        mermaid = await self.exporter.export_to_mermaid(nodes, [])
        assert '["Cls"]' in mermaid
        assert '("Fn")' in mermaid
        assert '[["Mod"]]' in mermaid
        assert '["Other"]' in mermaid  # default shape

    async def test_export_to_mermaid_edge_types(self):
        edges = [
            FakeEdge("e0", "a", "b", "inheritance"),
            FakeEdge("e1", "b", "c", "call"),
            FakeEdge("e2", "c", "d", "dependency"),
            FakeEdge("e3", "d", "e", "other"),
        ]
        # Provide minimal nodes so sanitization works
        nodes = [FakeNode(f"n{i}", f"N{i}", "class", "java") for i in range(5)]
        nodes[0].id = "a"
        nodes[1].id = "b"
        nodes[2].id = "c"
        nodes[3].id = "d"
        nodes[4].id = "e"
        mermaid = await self.exporter.export_to_mermaid(nodes, edges)
        assert "a -->" in mermaid
        assert "b -.->" in mermaid  # call edge

    async def test_export_to_mermaid_edge_label(self):
        edges = [FakeEdge("e0", "a", "b", "dep", label="uses")]
        nodes = [FakeNode("a", "A", "class", "java"), FakeNode("b", "B", "class", "java")]
        mermaid = await self.exporter.export_to_mermaid(nodes, edges)
        assert "|uses|" in mermaid

    async def test_export_to_mermaid_no_styling(self):
        nodes = _make_nodes(1)
        mermaid = await self.exporter.export_to_mermaid(nodes, [], options={"add_styling": False})
        assert "classDef" not in mermaid

    async def test_export_to_mermaid_direction(self):
        nodes = _make_nodes(1)
        mermaid = await self.exporter.export_to_mermaid(
            nodes, [], options={"direction": "LR", "diagram_type": "flowchart"}
        )
        assert mermaid.startswith("flowchart LR")

    # -- JSON ----------------------------------------------------------------

    async def test_export_to_json_happy(self):
        nodes = _make_nodes(2)
        edges = _make_edges(1)
        raw = await self.exporter.export_to_json(nodes, edges, metadata={"version": "1"})
        data = json.loads(raw)
        assert data["metadata"]["version"] == "1"
        assert data["statistics"]["node_count"] == 2
        assert data["statistics"]["edge_count"] == 1

    async def test_export_to_json_no_metadata(self):
        raw = await self.exporter.export_to_json([], [])
        data = json.loads(raw)
        assert data["metadata"] == {}
        assert data["statistics"]["node_count"] == 0


# ============================================================================
# data_utilities tests
# ============================================================================


class TestDataUtilities:
    """Tests for data_utilities module functions."""

    # -- validate_visualization_type -----------------------------------------

    def test_validate_visualization_type_valid(self):
        result = validate_visualization_type("graph")
        assert result.value == "graph"

    def test_validate_visualization_type_invalid(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_visualization_type("invalid_type")
        assert exc_info.value.status_code == 400
        assert "Invalid visualization type" in str(exc_info.value.detail)

    # -- validate_file_paths -------------------------------------------------

    def test_validate_file_paths_returns_confined_relative_paths(self):
        result = validate_file_paths(["src/a.py", "./src/a.py", "/workspace/b.py"])
        assert result == ["src/a.py", "b.py"]

    @pytest.mark.parametrize(
        "bad",
        [
            "/etc/passwd",
            "/app/src/main.py",
            "../../etc/passwd",
            "src/../../etc/shadow",
            "/workspace/../etc/passwd",
            "a\x00.py",
        ],
    )
    def test_validate_file_paths_refuses_traversal(self, bad):
        # This used to check only Path(p).exists() on THIS pod and parse
        # whatever it found -- any file the service could read.
        with pytest.raises(HTTPException) as exc_info:
            validate_file_paths(["ok.py", bad])
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "INVALID_WORKSPACE_PATH"

    def test_validate_file_paths_never_touches_local_disk(self, tmp_path):
        # An existing LOCAL absolute path is not thereby acceptable.
        f = tmp_path / "secret.txt"
        f.write_text("s3cr3t")
        with pytest.raises(HTTPException) as exc_info:
            validate_file_paths([str(f)])
        assert exc_info.value.status_code == 400

    # -- prepare_chart_data --------------------------------------------------

    def test_prepare_chart_data_complexity_heatmap(self):
        result = prepare_chart_data({"values": [1, 2]}, "complexity_heatmap")
        assert result["chart_type"] == "complexity_heatmap"
        assert result["options"]["color_scale"] == "viridis"
        assert result["options"]["show_values"] is True

    def test_prepare_chart_data_dependency_matrix(self):
        result = prepare_chart_data({}, "dependency_matrix")
        assert result["options"]["symmetric"] is True
        assert result["options"]["show_diagonal"] is False

    def test_prepare_chart_data_language_distribution(self):
        result = prepare_chart_data({}, "language_distribution")
        assert result["options"]["chart_style"] == "pie"

    def test_prepare_chart_data_unknown_type(self):
        result = prepare_chart_data({"x": 1}, "unknown")
        assert result["chart_type"] == "unknown"
        assert result["options"] == {}

    # -- transform_graph_data ------------------------------------------------

    def test_transform_graph_data_vis(self):
        graph = {
            "nodes": [{"id": "a", "label": "A", "type": "class", "tooltip": "tip"}],
            "edges": [{"source": "a", "target": "b", "type": "dep"}],
        }
        result = transform_graph_data(graph, "vis")
        assert result["nodes"][0]["group"] == "class"
        assert result["nodes"][0]["title"] == "tip"
        assert result["edges"][0]["from"] == "a"
        assert result["edges"][0]["arrows"] == "to"

    def test_transform_graph_data_d3(self):
        graph = {
            "nodes": [{"id": "a"}],
            "edges": [{"source": "a", "target": "b", "type": "dep"}],
        }
        result = transform_graph_data(graph, "d3")
        assert "links" in result
        assert result["links"][0]["source"] == "a"

    def test_transform_graph_data_unsupported(self):
        graph = {"nodes": []}
        result = transform_graph_data(graph, "unknown_format")
        assert result is graph

    # -- validate_visualization_request --------------------------------------

    def test_validate_visualization_request_happy(self):
        data = {"type": "dependency_graph", "file_paths": ["/a.py"]}
        result = validate_visualization_request(data)
        assert result["type"] == "dependency_graph"
        assert result["export_format"] == "json"

    def test_validate_visualization_request_missing_type(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_visualization_request({})
        assert exc_info.value.status_code == 400
        assert "type is required" in str(exc_info.value.detail)

    def test_validate_visualization_request_invalid_type(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_visualization_request({"type": "bogus"})
        assert exc_info.value.status_code == 400

    def test_validate_visualization_request_invalid_format(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_visualization_request({"type": "dependency_graph", "export_format": "html"})
        assert exc_info.value.status_code == 400
        assert "Invalid export format" in str(exc_info.value.detail)

    def test_validate_visualization_request_defaults(self):
        result = validate_visualization_request({"type": "call_graph"})
        assert result["file_paths"] == []
        assert result["options"] == {}
        assert result["container_name"] is None

    # -- create_visualization_result -----------------------------------------

    def test_create_visualization_result(self):
        from app.api.endpoints.visualization_modules.data_utilities import (
            create_visualization_result,
        )

        # Build a mock graph object
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.label = "MyClass"
        mock_node.type.value = "class"
        mock_node.language = "python"
        mock_node.file_path = "/src/a.py"
        mock_node.size = 2
        mock_node.color = "#fff"
        mock_node.shape = "box"
        mock_node.group = "g"
        mock_node.complexity = 5.0
        mock_node.importance = 0.8
        mock_node.properties = {"k": "v"}

        mock_edge = MagicMock()
        mock_edge.id = "e1"
        mock_edge.source = "n1"
        mock_edge.target = "n2"
        mock_edge.type.value = "dependency"
        mock_edge.label = "lbl"
        mock_edge.weight = 2.0
        mock_edge.color = "#aaa"
        mock_edge.style = "solid"
        mock_edge.properties = {}

        mock_graph = MagicMock()
        mock_graph.id = "g1"
        mock_graph.title = "Test"
        mock_graph.type.value = "dependency_graph"
        mock_graph.nodes = [mock_node]
        mock_graph.edges = [mock_edge]
        mock_graph.languages = {"python"}
        mock_graph.statistics = {"total": 1}
        mock_graph.layout_options = {"algo": "x"}

        result = create_visualization_result(mock_graph, {"json": "{}"})
        assert result.id == "g1"
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "n1"
        assert len(result.edges) == 1
        assert result.edges[0].source == "n1"
        assert result.exports == {"json": "{}"}


# ============================================================================
# shared_models tests
# ============================================================================


class TestSharedModels:
    """Tests for Pydantic models in shared_models.py."""

    def test_chart_type_enum(self):
        assert ChartType.COMPLEXITY_HEATMAP == "complexity_heatmap"
        assert ChartType.DEPENDENCY_MATRIX == "dependency_matrix"
        assert ChartType.CALL_HIERARCHY == "call_hierarchy"

    def test_export_format_enum(self):
        assert ExportFormat.JSON == "json"
        assert ExportFormat.DOT == "dot"
        assert ExportFormat.MERMAID == "mermaid"
        assert ExportFormat.VIS == "vis"

    def test_visualization_node_defaults(self):
        node = DependencyVisualizerServiceVisualizationNode(
            id="n1", label="X", type="class", language="python"
        )
        assert node.size == 1
        assert node.complexity == 0.0
        assert node.shape == "box"
        assert node.properties == {}
        assert node.file_path is None

    def test_visualization_node_full(self):
        node = DependencyVisualizerServiceVisualizationNode(
            id="n1",
            label="X",
            type="class",
            language="java",
            file_path="/a.java",
            size=3,
            color="#ff0000",
            shape="ellipse",
            group="svc",
            complexity=12.5,
            importance=0.9,
            properties={"pkg": "com.x"},
        )
        assert node.color == "#ff0000"
        assert node.group == "svc"

    def test_visualization_edge_defaults(self):
        edge = DependencyVisualizerServiceVisualizationEdge(
            id="e1", source="a", target="b", type="dep"
        )
        assert edge.weight == 1.0
        assert edge.style == "solid"
        assert edge.label is None

    def test_visualization_request_defaults(self):
        req = VisualizationRequest(container_name="ws", file_paths=["a.py"])
        assert req.visualization_type == "dependency_graph"
        assert req.export_formats == ["cytoscape", "json"]
        assert req.options is None

    def test_visualization_result(self):
        result = VisualizationResult(
            id="v1",
            title="Test",
            type="dependency_graph",
            nodes=[],
            edges=[],
            languages=["python"],
            statistics={"total": 0},
            layout_options={},
            exports={"json": "{}"},
        )
        assert result.id == "v1"
        assert result.languages == ["python"]

    def test_visualization_capabilities_response(self):
        resp = VisualizationCapabilitiesResponse(
            visualization_types=["dependency_graph"],
            node_types=["class"],
            edge_types=["dependency"],
            export_formats=["json"],
            layout_algorithms=["force-directed"],
            supported_languages=["python"],
        )
        assert resp.visualization_types == ["dependency_graph"]


# ============================================================================
# configuration tests
# ============================================================================


class TestConfiguration:
    """Tests for configuration module."""

    def test_visualization_config_defaults(self):
        config = VisualizationConfig()
        assert config.max_nodes == 1000
        assert config.max_edges == 5000
        assert config.default_layout == "hierarchical"
        assert "json" in config.supported_formats
        assert config.cache_ttl == 3600
        assert config.chunk_size == 100

    def test_get_visualization_config(self):
        config = get_visualization_config()
        assert "default_options" in config
        assert config["default_options"]["layout"] == "hierarchical"
        assert "export_formats" in config
        assert "json" in config["export_formats"]
        assert "layout_algorithms" in config
        assert "hierarchical" in config["layout_algorithms"]
        assert "performance" in config
        assert config["performance"]["max_nodes_interactive"] == 500
        assert "visualization_types" in config
        assert "dependency_graph" in config["visualization_types"]

    async def test_get_visualization_capabilities_endpoint(self):
        """Test the capabilities endpoint function directly."""
        from app.api.endpoints.visualization_modules.configuration import (
            get_visualization_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test"
        result = await get_visualization_capabilities(user=mock_user)
        assert isinstance(result, VisualizationCapabilitiesResponse)
        assert len(result.visualization_types) > 0
        assert len(result.node_types) > 0
        assert len(result.edge_types) > 0
        assert "python" in result.supported_languages

    async def test_get_visualization_examples_endpoint(self):
        """Test the examples endpoint function directly."""
        from app.api.endpoints.visualization_modules.configuration import (
            get_visualization_examples,
        )

        mock_user = MagicMock()
        result = await get_visualization_examples(user=mock_user)
        assert "examples" in result
        assert "best_practices" in result
        assert "tips" in result
        assert "basic_dependency_graph" in result["examples"]
        assert len(result["tips"]) > 0


# ============================================================================
# graph_visualization tests
# ============================================================================


_VIZ_VALIDATE = (
    "app.api.endpoints.visualization_modules.graph_visualization"
    ".validate_and_get_container_for_tenant"
)
_VIZ_WS_CLIENT = "app.api.utils.workspace_files.get_coding_containers_client"


def _viz_container():
    return SimpleNamespace(codebase_path="/workspace", validated=True)


def _viz_ws_client(files):
    client = MagicMock()

    async def _read_file(container_name, file_path, tenant_id=None):
        return files.get(file_path)

    client.read_file = AsyncMock(side_effect=_read_file)
    return client


class TestGraphVisualization:
    """Tests for the create_dependency_visualization endpoint."""

    async def test_create_dependency_visualization_empty_paths(self):
        from app.api.endpoints.visualization_modules.graph_visualization import (
            create_dependency_visualization,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test"
        request = VisualizationRequest(container_name="ws", file_paths=[])
        # Empty file_paths should raise 400
        # The endpoint checks `if not request.file_paths`
        # Since file_paths=[] is falsy, it should raise
        with pytest.raises(HTTPException) as exc_info:
            await create_dependency_visualization(
                request=request, user=mock_user, tenant_id="tenant-a"
            )
        assert exc_info.value.status_code == 400

    async def test_create_dependency_visualization_success(self):
        from app.api.endpoints.visualization_modules.graph_visualization import (
            create_dependency_visualization,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "tester"

        # Mock the entire pipeline
        mock_graph = MagicMock()
        mock_graph.id = "g1"
        mock_graph.title = "Test Graph"
        mock_graph.type.value = "dependency_graph"
        mock_graph.languages = {"python"}
        mock_graph.statistics = {"total": 1}
        mock_graph.layout_options = {}

        mock_gnode = MagicMock()
        mock_gnode.id = "n1"
        mock_gnode.label = "A"
        mock_gnode.type.value = "class"
        mock_gnode.language = "python"
        mock_gnode.file_path = "/a.py"
        mock_gnode.size = 1
        mock_gnode.color = None
        mock_gnode.shape = "box"
        mock_gnode.group = None
        mock_gnode.complexity = 1.0
        mock_gnode.importance = 0.5
        mock_gnode.properties = {}
        mock_graph.nodes = [mock_gnode]
        mock_graph.edges = []

        mock_visualizer = AsyncMock()
        mock_visualizer.create_dependency_graph = AsyncMock(return_value=mock_graph)
        mock_visualizer.export_to_formats = AsyncMock(return_value={"json": "{}"})

        request = VisualizationRequest(container_name="ws", file_paths=["src/test.py"])
        client = _viz_ws_client({"/workspace/src/test.py": "import os\n"})

        with (
            patch(
                "app.api.endpoints.visualization_modules.graph_visualization"
                ".validate_visualization_type"
            ) as mock_vt,
            patch(_VIZ_VALIDATE, new=AsyncMock(return_value=_viz_container())),
            patch(_VIZ_WS_CLIENT, return_value=client),
            patch(
                "app.api.endpoints.visualization_modules.graph_visualization.parse_files",
                new_callable=AsyncMock,
            ) as mock_pf,
            patch(
                "app.api.endpoints.visualization_modules.graph_visualization"
                ".get_dependency_visualizer"
            ) as mock_gdv,
        ):
            mock_vt.return_value = "dependency_graph"
            mock_pf.return_value = [MagicMock()]
            mock_gdv.return_value = mock_visualizer

            result = await create_dependency_visualization(
                request=request, user=mock_user, tenant_id="tenant-a"
            )

        assert result.status.value == "success"
        assert result.result.id == "g1"
        assert len(result.result.nodes) == 1
        # The parser saw the WORKSPACE copy, fetched under the caller's tenant.
        client.read_file.assert_awaited_once_with(
            "ws", "/workspace/src/test.py", tenant_id="tenant-a"
        )
        (parsed,) = mock_pf.await_args.args[0]
        assert parsed.parts[-2:] == ("src", "test.py")
        # ... and the local copy is gone afterwards.
        assert not parsed.exists()

    @pytest.mark.parametrize(
        "bad", ["/etc/passwd", "../../etc/passwd", "/app/src/main.py", "src/../../x.py"]
    )
    async def test_traversal_refused_before_any_read(self, bad):
        from app.api.endpoints.visualization_modules.graph_visualization import (
            create_dependency_visualization,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "tester"
        client = _viz_ws_client({})
        with (
            patch(_VIZ_VALIDATE, new=AsyncMock(return_value=_viz_container())),
            patch(_VIZ_WS_CLIENT, return_value=client),
            patch(
                "app.api.endpoints.visualization_modules.graph_visualization.parse_files",
                new_callable=AsyncMock,
            ) as mock_pf,
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_dependency_visualization(
                request=VisualizationRequest(
                    container_name="ws", file_paths=[bad], visualization_type="graph"
                ),
                user=mock_user,
                tenant_id="tenant-a",
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "INVALID_WORKSPACE_PATH"
        client.read_file.assert_not_awaited()
        mock_pf.assert_not_awaited()

    async def test_cross_tenant_container_refused_before_any_read(self):
        from app.api.endpoints.visualization_modules.graph_visualization import (
            create_dependency_visualization,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "tester"
        client = _viz_ws_client({"/workspace/a.py": "x = 1\n"})
        with (
            patch(
                _VIZ_VALIDATE,
                new=AsyncMock(side_effect=HTTPException(status_code=404, detail="not found")),
            ),
            patch(_VIZ_WS_CLIENT, return_value=client),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_dependency_visualization(
                request=VisualizationRequest(
                    container_name="ws", file_paths=["a.py"], visualization_type="graph"
                ),
                user=mock_user,
                tenant_id="tenant-b",
            )
        assert exc_info.value.status_code == 404
        client.read_file.assert_not_awaited()

    async def test_unreadable_workspace_files_are_422(self):
        from app.api.endpoints.visualization_modules.graph_visualization import (
            create_dependency_visualization,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "tester"
        with (
            patch(_VIZ_VALIDATE, new=AsyncMock(return_value=_viz_container())),
            patch(_VIZ_WS_CLIENT, return_value=_viz_ws_client({})),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_dependency_visualization(
                request=VisualizationRequest(
                    container_name="ws", file_paths=["a.py"], visualization_type="graph"
                ),
                user=mock_user,
                tenant_id="tenant-a",
            )
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["error_code"] == "WORKSPACE_UNRESOLVABLE"


# ============================================================================
# parse_files tests
# ============================================================================

_GET_FACTORY = "app.api.endpoints.visualization_modules.data_utilities.get_multi_language_factory"
_FILE_SERVICE = "app.api.endpoints.visualization_modules.data_utilities.FileService"


class TestParseFiles:
    """Tests for parse_files function."""

    @pytest.mark.asyncio
    async def test_successful_parse(self, tmp_path):
        f1 = tmp_path / "main.py"
        f1.write_text("x = 1\n")

        mock_parse_result = MagicMock()
        mock_parse_result.success = True

        mock_processor = MagicMock()
        mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)

        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
        ):
            results = await parse_files([f1])
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_processor_skips_file(self, tmp_path):
        f1 = tmp_path / "data.xyz"
        f1.write_text("data\n")

        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=None)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
            pytest.raises(HTTPException) as exc_info,
        ):
            await parse_files([f1])
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_processor_without_parse_file_with_metrics(self, tmp_path):
        f1 = tmp_path / "main.py"
        f1.write_text("x = 1\n")

        mock_processor = MagicMock(spec=[])  # empty spec - no parse_file_with_metrics
        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
            pytest.raises(HTTPException) as exc_info,
        ):
            await parse_files([f1])
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_failure_skips_file(self, tmp_path):
        f1 = tmp_path / "bad.py"
        f1.write_text("bad code\n")

        mock_parse_result = MagicMock()
        mock_parse_result.success = False
        mock_parse_result.errors = ["syntax error"]

        mock_processor = MagicMock()
        mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)

        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
            pytest.raises(HTTPException) as exc_info,
        ):
            await parse_files([f1])
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_os_error_skips_file(self, tmp_path):
        f1 = tmp_path / "missing.py"
        # Don't create the file -- reading it will raise OSError

        mock_processor = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
            pytest.raises(HTTPException) as exc_info,
        ):
            await parse_files([f1])
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_all_files_fail_raises_422(self, tmp_path):
        """When no files parse successfully, raises 422."""
        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=None)

        with (
            patch(_GET_FACTORY, return_value=mock_factory),
            patch(_FILE_SERVICE),
            pytest.raises(HTTPException) as exc_info,
        ):
            await parse_files([tmp_path / "a.py"])
        assert exc_info.value.status_code == 422
