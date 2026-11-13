"""Unit tests for code_graph_metrics.py and code_graph_metrics_extension.py.

Covers:
- CodeGraphMetricsService.get_graph_metrics
- CodeGraphMetricsService.get_node_centrality_metrics
- CodeGraphMetricsService.calculate_centrality_metrics
- CodeGraphMetricsService.get_function_network_metrics
- Extension: _calculate_betweenness_centrality
- Extension: _find_shortest_paths_through_node
- Extension: _find_total_shortest_paths
- Extension: _bfs_distance
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from services.code_graph_metrics import CodeGraphMetricsService
from services.code_graph_metrics_extension import (
    _bfs_distance,
    _find_shortest_paths_through_node,
    _find_total_shortest_paths,
)


def _make_mock_db_conn(
    node_count=10,
    edge_count=15,
    node_types=None,
    rel_types=None,
    nodes_list=None,
    edges_list=None,
):
    """Create a mock db_connection with configurable data."""
    if node_types is None:
        node_types = {"function": 5, "class": 3, "module": 2}
    if rel_types is None:
        rel_types = {"calls": 8, "imports": 7}

    conn = MagicMock()
    db = MagicMock()
    db.conn = conn

    cursor_counts = AsyncMock()
    cursor_counts.fetchone = AsyncMock(
        return_value=(
            node_count,
            edge_count,
            len(node_types),
            len(rel_types),
        )
    )

    cursor_types = AsyncMock()
    cursor_types.fetchall = AsyncMock(return_value=list(node_types.items()))

    cursor_rels = AsyncMock()
    cursor_rels.fetchall = AsyncMock(return_value=list(rel_types.items()))

    conn.execute = AsyncMock(side_effect=[cursor_counts, cursor_types, cursor_rels])

    return db


def _make_centrality_mock_db(out_degree=3, in_degree=5, total_nodes=20):
    """Create a mock for centrality queries."""
    conn = MagicMock()
    db = MagicMock()
    db.conn = conn

    cursor = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=(out_degree, in_degree, total_nodes))
    conn.execute = AsyncMock(return_value=cursor)

    return db


def _make_network_mock_db(out_degree, in_degree, total_nodes, node_ids, edges):
    """Mock db for get_function_network_metrics.

    Serves, in call order: (1) the single centrality query used by
    ``get_node_centrality_metrics`` (``fetchone`` -> out/in/total), then
    (2) the node-id query and (3) the edge query issued by
    ``_calculate_pagerank_and_closeness`` (``fetchall`` each).
    """
    conn = MagicMock()
    db = MagicMock()
    db.conn = conn

    cursor_centrality = AsyncMock()
    cursor_centrality.fetchone = AsyncMock(return_value=(out_degree, in_degree, total_nodes))

    cursor_nodes = AsyncMock()
    cursor_nodes.fetchall = AsyncMock(return_value=[(nid,) for nid in node_ids])

    cursor_edges = AsyncMock()
    cursor_edges.fetchall = AsyncMock(return_value=list(edges))

    conn.execute = AsyncMock(side_effect=[cursor_centrality, cursor_nodes, cursor_edges])

    return db


def _make_entity_type_mock_db(node_ids=None, out_degree=2, in_degree=1, total_nodes=10):
    """Create a mock for calculate_centrality_metrics."""
    if node_ids is None:
        node_ids = ["fn1", "fn2"]

    conn = MagicMock()
    db = MagicMock()
    db.conn = conn

    cursor_nodes = AsyncMock()
    # Source reads node[0] positionally, so rows are single-column tuples.
    node_rows = [(nid,) for nid in node_ids]
    cursor_nodes.fetchall = AsyncMock(return_value=node_rows)

    centrality_cursors = []
    for _ in node_ids:
        c = AsyncMock()
        c.fetchone = AsyncMock(return_value=(out_degree, in_degree, total_nodes))
        centrality_cursors.append(c)

    conn.execute = AsyncMock(side_effect=[cursor_nodes] + centrality_cursors)

    return db


class TestCodeGraphMetricsService:
    """Tests for CodeGraphMetricsService."""

    async def test_get_graph_metrics_success(self):
        """get_graph_metrics should return OverallGraphMetrics with calculated values."""
        db = _make_mock_db_conn(node_count=10, edge_count=15)
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 10
        assert result.edge_count == 15
        assert result.density == 15 / (10 * 9)
        assert result.average_degree == (2 * 15) / 10
        assert result.node_types == {"function": 5, "class": 3, "module": 2}
        assert result.relationship_types == {"calls": 8, "imports": 7}
        assert result.calculation_time_seconds >= 0.0

    async def test_get_graph_metrics_no_connection(self):
        """get_graph_metrics should return zero metrics when conn is None."""
        db = MagicMock()
        db.conn = None
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.density == 0.0
        assert result.average_degree == 0.0
        assert result.node_types == {}
        assert result.relationship_types == {}
        assert result.calculation_time_seconds == 0.0

    async def test_get_graph_metrics_query_exception(self):
        """get_graph_metrics should return fallback on exception."""
        db = MagicMock()
        db.conn = MagicMock()
        db.conn.execute = AsyncMock(side_effect=RuntimeError("DB error"))
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.density == 0.0
        assert result.average_degree == 0.0
        assert result.node_types == {}
        assert result.relationship_types == {}

    async def test_get_graph_metrics_none_result(self):
        """get_graph_metrics should handle None result from fetchone."""
        db = MagicMock()
        db.conn = MagicMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=None)
        db.conn.execute = AsyncMock(return_value=cursor)
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 0
        assert result.density == 0.0
        assert result.average_degree == 0.0

    async def test_get_graph_metrics_zero_nodes(self):
        """get_graph_metrics should handle zero nodes."""
        db = _make_mock_db_conn(node_count=0, edge_count=0, node_types={}, rel_types={})
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.average_degree == 0.0
        assert result.density == 0.0
        assert result.node_types == {}
        assert result.relationship_types == {}

    async def test_get_graph_metrics_single_node(self):
        """get_graph_metrics should handle density with single node."""
        db = _make_mock_db_conn(node_count=1, edge_count=0, node_types={"module": 1}, rel_types={})
        svc = CodeGraphMetricsService(db)

        result = await svc.get_graph_metrics()

        assert result.node_count == 1
        assert result.edge_count == 0
        assert result.density == 0.0
        assert result.average_degree == 0.0
        assert result.node_types == {"module": 1}
        assert result.relationship_types == {}


class TestNodeCentralityMetrics:
    """Tests for get_node_centrality_metrics."""

    async def test_get_node_centrality_success(self):
        """Should return centrality metrics for a valid node."""
        db = _make_centrality_mock_db(out_degree=3, in_degree=5, total_nodes=20)
        svc = CodeGraphMetricsService(db)

        result = await svc.get_node_centrality_metrics("fn1")

        assert result["out_degree"] == 3
        assert result["in_degree"] == 5
        assert result["total_degree"] == 8
        assert result["degree_centrality"] == 8 / 19

    async def test_get_node_centrality_no_connection(self):
        """Should return fallback when conn is None."""
        db = MagicMock()
        db.conn = None
        svc = CodeGraphMetricsService(db)

        result = await svc.get_node_centrality_metrics("fn1")

        assert result == {
            "degree_centrality": 0.0,
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": 0,
        }

    async def test_get_node_centrality_exception(self):
        """Should return fallback on exception."""
        db = MagicMock()
        db.conn = MagicMock()
        db.conn.execute = AsyncMock(side_effect=RuntimeError("error"))
        svc = CodeGraphMetricsService(db)

        result = await svc.get_node_centrality_metrics("fn1")

        assert result == {
            "degree_centrality": 0.0,
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": 0,
        }

    async def test_get_node_centrality_none_result(self):
        """Should handle None result from fetchone."""
        db = MagicMock()
        db.conn = MagicMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=None)
        db.conn.execute = AsyncMock(return_value=cursor)
        svc = CodeGraphMetricsService(db)

        result = await svc.get_node_centrality_metrics("fn1")

        assert result == {
            "degree_centrality": 0.0,
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": 0,
        }


class TestCalculateCentralityMetrics:
    """Tests for calculate_centrality_metrics."""

    async def test_calculate_centrality_for_entity_type(self):
        """Should return centrality for all nodes of a given type."""
        db = _make_entity_type_mock_db(
            node_ids=["fn1", "fn2"], out_degree=2, in_degree=1, total_nodes=10
        )
        svc = CodeGraphMetricsService(db)

        result = await svc.calculate_centrality_metrics("function")

        expected_total = 2 + 1
        expected_centrality = expected_total / max(1, 10 - 1)
        assert set(result.keys()) == {"fn1", "fn2"}
        for node_id in ("fn1", "fn2"):
            assert result[node_id]["out_degree"] == 2
            assert result[node_id]["in_degree"] == 1
            assert result[node_id]["total_degree"] == expected_total
            assert result[node_id]["degree_centrality"] == expected_centrality

    async def test_calculate_centrality_no_connection(self):
        """Should return empty dict when conn is None."""
        db = MagicMock()
        db.conn = None
        svc = CodeGraphMetricsService(db)

        result = await svc.calculate_centrality_metrics("function")

        assert result == {}

    async def test_calculate_centrality_exception(self):
        """Should return empty dict on exception."""
        db = MagicMock()
        db.conn = MagicMock()
        db.conn.execute = AsyncMock(side_effect=TypeError("bad"))
        svc = CodeGraphMetricsService(db)

        result = await svc.calculate_centrality_metrics("function")

        assert result == {}


class TestFunctionNetworkMetrics:
    """Tests for get_function_network_metrics."""

    async def test_get_function_network_metrics_success(self):
        """closeness/pagerank must be REAL networkx values, not degree multiples."""
        import networkx as nx

        node_ids = ["my_func", "b", "c", "d"]
        edges = [("b", "my_func"), ("c", "my_func"), ("my_func", "d"), ("c", "b")]
        # out_degree(my_func)=1, in_degree=2, total_nodes=4 -> matches the graph.
        db = _make_network_mock_db(
            out_degree=1, in_degree=2, total_nodes=4, node_ids=node_ids, edges=edges
        )
        svc = CodeGraphMetricsService(db)
        # Source awaits _calculate_betweenness_centrality, so it must be async.
        svc._calculate_betweenness_centrality = AsyncMock(return_value=0.5)

        result = await svc.get_function_network_metrics("my_func")

        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        graph.add_edges_from(edges)
        expected_pr = nx.pagerank(graph)["my_func"]
        expected_cc = nx.closeness_centrality(graph, u="my_func")

        dc = (1 + 2) / (4 - 1)
        ni = dc * 0.4 + 0.5 * 0.3 + expected_cc * 0.2 + expected_pr * 0.1

        assert result["in_degree"] == 2
        assert result["out_degree"] == 1
        assert result["total_degree"] == 3
        assert result["degree_centrality"] == dc
        assert result["betweenness_centrality"] == 0.5
        # Honest contract: these are the actual graph-theoretic metrics, NOT dc*k.
        assert result["closeness_centrality"] == expected_cc
        assert result["pagerank"] == expected_pr
        assert result["closeness_centrality"] != dc * 0.6
        assert result["pagerank"] != dc * 0.7
        assert abs(result["network_importance"] - ni) < 1e-12

    async def test_get_function_network_metrics_error(self):
        """Should return fallback on error."""
        db = MagicMock()
        db.conn = MagicMock()
        db.conn.execute = AsyncMock(side_effect=ValueError("error"))
        svc = CodeGraphMetricsService(db)

        result = await svc.get_function_network_metrics("broken_func")

        assert result == {
            "degree_centrality": 0.0,
            "betweenness_centrality": 0.0,
            "closeness_centrality": 0.0,
            "pagerank": 0.0,
            "network_importance": 0.0,
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": 0,
        }

    async def test_network_importance_weighting(self):
        """Network importance is a weighted combination of the REAL centrality metrics."""
        import networkx as nx

        node_ids = ["my_func", "b", "c"]
        edges = [("b", "my_func"), ("my_func", "c"), ("c", "b")]
        db = _make_network_mock_db(
            out_degree=1, in_degree=1, total_nodes=3, node_ids=node_ids, edges=edges
        )
        svc = CodeGraphMetricsService(db)
        # Source awaits _calculate_betweenness_centrality, so it must be async.
        svc._calculate_betweenness_centrality = AsyncMock(return_value=0.0)

        result = await svc.get_function_network_metrics("my_func")

        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        graph.add_edges_from(edges)
        expected_pr = nx.pagerank(graph)["my_func"]
        expected_cc = nx.closeness_centrality(graph, u="my_func")

        dc = (1 + 1) / (3 - 1)
        expected_ni = dc * 0.4 + 0.0 * 0.3 + expected_cc * 0.2 + expected_pr * 0.1
        assert result["degree_centrality"] == dc
        assert result["closeness_centrality"] == expected_cc
        assert result["pagerank"] == expected_pr
        assert abs(result["network_importance"] - expected_ni) < 1e-12


class TestBfsDistance:
    """Tests for _bfs_distance."""

    def test_same_node(self):
        graph = {"A": ["B"], "B": ["A"]}
        assert _bfs_distance(graph, "A", "A") == 0

    def test_direct_neighbor(self):
        graph = {"A": ["B"], "B": []}
        assert _bfs_distance(graph, "A", "B") == 1

    def test_two_hops(self):
        graph = {"A": ["B"], "B": ["C"], "C": []}
        assert _bfs_distance(graph, "A", "C") == 2

    def test_unreachable(self):
        graph = {"A": [], "B": []}
        assert _bfs_distance(graph, "A", "B") == 999999

    def test_empty_graph(self):
        assert _bfs_distance({}, "A", "B") == 999999

    def test_exception_handling(self):
        """Should return 999999 on error."""
        assert _bfs_distance(None, "A", "B") == 999999


class TestFindShortestPathsThroughNode:
    """Tests for _find_shortest_paths_through_node."""

    def test_path_through_node(self):
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = _find_shortest_paths_through_node(graph, "A", "C", "B")
        assert result == 1

    def test_no_path_through_node(self):
        graph = {"A": ["C"], "B": [], "C": []}
        result = _find_shortest_paths_through_node(graph, "A", "C", "B")
        assert result == 0

    def test_unreachable_target(self):
        graph = {"A": [], "B": [], "C": []}
        result = _find_shortest_paths_through_node(graph, "A", "C", "B")
        assert result == 0

    def test_exception_handling(self):
        result = _find_shortest_paths_through_node(None, "A", "C", "B")
        assert result == 0


class TestFindTotalShortestPaths:
    """Tests for _find_total_shortest_paths."""

    def test_reachable(self):
        graph = {"A": ["B"], "B": []}
        result = _find_total_shortest_paths(graph, "A", "B")
        assert result == 1

    def test_unreachable(self):
        graph = {"A": [], "B": []}
        result = _find_total_shortest_paths(graph, "A", "B")
        assert result == 0

    def test_exception_handling(self):
        result = _find_total_shortest_paths(None, "A", "B")
        assert result == 0


class TestCalculateBetweennessCentrality:
    """Tests for _calculate_betweenness_centrality via CodeGraphMetricsService.

    The method is now async and reads from postgres: it awaits two queries
    (``SELECT id FROM nodes`` then ``SELECT source, target FROM edges``), both
    scoped by ``code_graph_id``, and reads rows positionally (``row[0]`` /
    ``source, target``).
    """

    @staticmethod
    def _make_db(make_pg_conn, node_rows, edge_rows):
        db = MagicMock()
        db.code_graph_id = "g1"
        db.conn = make_pg_conn(
            routes={
                "SELECT id FROM nodes WHERE code_graph_id = %s": node_rows,
                "SELECT source, target FROM edges WHERE code_graph_id = %s": edge_rows,
            }
        )
        return db

    async def test_betweenness_with_small_graph(self, make_pg_conn):
        """Betweenness centrality of middle node in A->B->C chain should be 1.0."""
        db = self._make_db(
            make_pg_conn,
            node_rows=[("A",), ("B",), ("C",)],
            edge_rows=[("A", "B"), ("B", "C")],
        )
        svc = CodeGraphMetricsService(db)

        result = await svc._calculate_betweenness_centrality("B")
        assert result == 1.0
        # both queries scoped to the code_graph_id
        for _sql, params in db.conn.executed:
            assert params == ("g1",)

    async def test_betweenness_fewer_than_three_nodes(self, make_pg_conn):
        """With fewer than 3 nodes, betweenness should be 0."""
        db = self._make_db(
            make_pg_conn,
            node_rows=[("A",), ("B",)],
            edge_rows=[],
        )
        svc = CodeGraphMetricsService(db)

        result = await svc._calculate_betweenness_centrality("A")
        assert result == 0.0

    async def test_betweenness_error_returns_zero(self):
        """Should return 0.0 on error."""
        db = MagicMock()
        db.code_graph_id = "g1"
        db.conn = MagicMock()
        db.conn.execute = AsyncMock(side_effect=AttributeError("bad"))
        svc = CodeGraphMetricsService(db)

        result = await svc._calculate_betweenness_centrality("A")
        assert result == 0.0

    async def test_betweenness_isolated_node(self, make_pg_conn):
        """An isolated node in a graph should have 0 betweenness."""
        db = self._make_db(
            make_pg_conn,
            node_rows=[("A",), ("B",), ("C",)],
            edge_rows=[("A", "B")],
        )
        svc = CodeGraphMetricsService(db)

        result = await svc._calculate_betweenness_centrality("C")
        assert result == 0.0
