"""Unit tests for services.dependency_visualizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from services.dependency_visualizer import (
    DependencyVisualizer,
    EdgeType,
    NodeType,
    VisualizationType,
    get_dependency_visualizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PACKAGES = {
    "requests": {"name": "requests", "version": "==2.31.0", "extras": []},
    "flask": {"name": "flask", "version": ">=3.0", "extras": []},
    "uvicorn": {"name": "uvicorn", "version": ">=0.30", "extras": ["standard"]},
}

ONE_PACKAGE = {
    "numpy": {"name": "numpy", "version": ">=2.0", "extras": []},
}


def _make_visualizer(packages: dict) -> DependencyVisualizer:
    """Return a DependencyVisualizer whose DependencyParser is mocked."""
    return DependencyVisualizer()


async def _create(packages: dict, viz_type: VisualizationType) -> dict:
    """Run create_visualization with a mocked DependencyParser."""
    with patch(
        "services.dependency_parser.DependencyParser",
        autospec=True,
    ) as mock_parser:
        instance = mock_parser.return_value
        instance.get_available_packages = AsyncMock(return_value=packages)
        vis = DependencyVisualizer()
        return await vis.create_visualization("/fake/path", viz_type)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    async def test_has_nodes_edges_metadata(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result

    async def test_metadata_fields(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        meta = result["metadata"]
        assert meta["total_packages"] == len(SAMPLE_PACKAGES)
        assert meta["visualization_type"] == "graph"
        assert meta["codebase_path"] == "/fake/path"

    async def test_node_structure(self):
        result = await _create(ONE_PACKAGE, VisualizationType.GRAPH)
        node = result["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert node["type"] == NodeType.PACKAGE
        assert "version" in node

    async def test_node_count_matches_packages(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        assert len(result["nodes"]) == len(SAMPLE_PACKAGES)

    async def test_edge_structure(self):
        # uvicorn has extras=["standard"] — standard not in packages, so no edges expected
        packages = {
            "parent": {"name": "parent", "version": "", "extras": ["child"]},
            "child": {"name": "child", "version": "", "extras": []},
        }
        result = await _create(packages, VisualizationType.GRAPH)
        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["source"] == "parent"
        assert edge["target"] == "child"
        assert edge["type"] == EdgeType.DEPENDS_ON


# ---------------------------------------------------------------------------
# Empty codebase
# ---------------------------------------------------------------------------


class TestEmptyCodebase:
    async def test_no_packages_returns_empty_nodes_and_edges(self):
        result = await _create({}, VisualizationType.GRAPH)
        assert result["nodes"] == []
        assert result["edges"] == []

    async def test_metadata_total_packages_is_zero(self):
        result = await _create({}, VisualizationType.MATRIX)
        assert result["metadata"]["total_packages"] == 0

    async def test_empty_matrix_is_empty_list(self):
        result = await _create({}, VisualizationType.MATRIX)
        assert result["matrix"] == []
        assert result["labels"] == []

    async def test_empty_tree_has_empty_tree(self):
        result = await _create({}, VisualizationType.TREE)
        assert result["tree"] == []


# ---------------------------------------------------------------------------
# GRAPH visualization
# ---------------------------------------------------------------------------


class TestGraphVisualization:
    async def test_adjacency_contains_all_node_ids(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        for node in result["nodes"]:
            assert node["id"] in result["adjacency"]

    async def test_adjacency_is_dict(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        assert isinstance(result["adjacency"], dict)

    async def test_no_edges_when_no_extras_match(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.GRAPH)
        # uvicorn[standard] — "standard" not a package name in SAMPLE_PACKAGES
        assert result["edges"] == []


# ---------------------------------------------------------------------------
# TREE visualization
# ---------------------------------------------------------------------------


class TestTreeVisualization:
    async def test_tree_key_present(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.TREE)
        assert "tree" in result

    async def test_tree_nodes_have_id_and_children(self):
        result = await _create(ONE_PACKAGE, VisualizationType.TREE)
        tree = result["tree"]
        assert len(tree) == 1
        assert tree[0]["id"] == "numpy"
        assert tree[0]["children"] == []

    async def test_tree_children_relationship(self):
        packages = {
            "parent": {"name": "parent", "version": "", "extras": ["child"]},
            "child": {"name": "child", "version": "", "extras": []},
        }
        result = await _create(packages, VisualizationType.TREE)
        tree = result["tree"]
        # Only "parent" is a root (child has a parent)
        root_ids = [t["id"] for t in tree]
        assert "parent" in root_ids
        assert "child" not in root_ids
        parent_node = next(t for t in tree if t["id"] == "parent")
        assert any(c["id"] == "child" for c in parent_node["children"])


# ---------------------------------------------------------------------------
# MATRIX visualization
# ---------------------------------------------------------------------------


class TestMatrixVisualization:
    async def test_matrix_is_square(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.MATRIX)
        n = len(SAMPLE_PACKAGES)
        matrix = result["matrix"]
        assert len(matrix) == n
        for row in matrix:
            assert len(row) == n

    async def test_labels_match_nodes(self):
        result = await _create(ONE_PACKAGE, VisualizationType.MATRIX)
        assert result["labels"] == ["numpy"]

    async def test_matrix_dependency_recorded(self):
        packages = {
            "parent": {"name": "parent", "version": "", "extras": ["child"]},
            "child": {"name": "child", "version": "", "extras": []},
        }
        result = await _create(packages, VisualizationType.MATRIX)
        labels = result["labels"]
        matrix = result["matrix"]
        pi = labels.index("parent")
        ci = labels.index("child")
        assert matrix[pi][ci] == 1

    async def test_matrix_no_self_loops_by_default(self):
        result = await _create(ONE_PACKAGE, VisualizationType.MATRIX)
        assert result["matrix"][0][0] == 0


# ---------------------------------------------------------------------------
# FORCE_DIRECTED visualization
# ---------------------------------------------------------------------------


class TestForceDirectedVisualization:
    async def test_nodes_have_x_y_positions(self):
        result = await _create(ONE_PACKAGE, VisualizationType.FORCE_DIRECTED)
        for node in result["nodes"]:
            assert "x" in node
            assert "y" in node

    async def test_positions_are_numeric(self):
        result = await _create(ONE_PACKAGE, VisualizationType.FORCE_DIRECTED)
        node = result["nodes"][0]
        assert isinstance(node["x"], float)
        assert isinstance(node["y"], float)

    async def test_adjacency_present(self):
        result = await _create(SAMPLE_PACKAGES, VisualizationType.FORCE_DIRECTED)
        assert "adjacency" in result

    async def test_positions_are_deterministic(self):
        r1 = await _create(ONE_PACKAGE, VisualizationType.FORCE_DIRECTED)
        r2 = await _create(ONE_PACKAGE, VisualizationType.FORCE_DIRECTED)
        assert r1["nodes"][0]["x"] == r2["nodes"][0]["x"]
        assert r1["nodes"][0]["y"] == r2["nodes"][0]["y"]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_dependency_visualizer_returns_instance(self):
        with patch("services.dependency_visualizer._dependency_visualizer", None):
            vis = get_dependency_visualizer()
            assert isinstance(vis, DependencyVisualizer)

    def test_returns_same_instance_on_repeated_calls(self):
        with patch("services.dependency_visualizer._dependency_visualizer", None):
            a = get_dependency_visualizer()
            # Patch the module-level var to simulate singleton state
            with patch("services.dependency_visualizer._dependency_visualizer", a):
                b = get_dependency_visualizer()
            assert a is b
