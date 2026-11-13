"""Unit tests for graph_database_populator module (postgres pool).

After the sqlite->postgres cutover this module batch-populates the shared
PostgreSQL code-graph store via ``db.session.get_raw_connection()``. Every row
is scoped to one ``code_graph_id`` (== container name); the schema is owned by
Alembic, so ``_create_database_schema`` and the per-file SQLite PRAGMA tuning
were removed. ``_batch_insert_nodes``/``_batch_insert_edges`` now take a leading
``code_graph_id`` argument and ``executemany`` parameterised INSERT ... ON
CONFLICT statements, wrapping JSONB properties in ``Jsonb(...)``.

These tests drive the shared ``FakePgConn`` (see ``tests/conftest.py``) and
assert the SQL shape + scoping recorded on ``conn.executed``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

# Load the module directly from its file path to avoid triggering
# services/__init__.py which imports many heavy dependencies.
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "services"
    / "code_graph"
    / "graph_database_populator.py"
)
_spec = importlib.util.spec_from_file_location("graph_database_populator", _MODULE_PATH)
_gdp = importlib.util.module_from_spec(_spec)
sys.modules["graph_database_populator"] = _gdp
_spec.loader.exec_module(_gdp)

_batch_insert_edges = _gdp._batch_insert_edges
_batch_insert_nodes = _gdp._batch_insert_nodes
get_database_stats = _gdp.get_database_stats
optimize_database_performance = _gdp.optimize_database_performance
populate_graph_database = _gdp.populate_graph_database
verify_database_integrity = _gdp.verify_database_integrity

# Patch target: the module-loaded ``get_raw_connection`` name.
_GET_CONN = "graph_database_populator.get_raw_connection"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_inserts(conn):
    return [
        (sql, params)
        for sql, params in conn.executed
        if "INSERT INTO nodes" in " ".join(sql.split())
    ]


def _edge_inserts(conn):
    return [
        (sql, params)
        for sql, params in conn.executed
        if "INSERT INTO edges" in " ".join(sql.split())
    ]


# _create_database_schema tests removed: the schema is now Alembic-owned and
# this module no longer creates tables or tunes per-file SQLite pragmas.


# ---------------------------------------------------------------------------
# _batch_insert_nodes (now takes a leading code_graph_id; executemany on the pool)
# ---------------------------------------------------------------------------


class TestBatchInsertNodes:
    async def test_empty_list_returns_zero(self, make_pg_conn):
        conn = make_pg_conn()
        result = await _batch_insert_nodes(conn, "cg", [])
        assert result == 0
        assert _node_inserts(conn) == []

    async def test_inserts_single_node(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [{"id": "n1", "type": "function", "properties": {"name": "foo"}}]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1
        inserts = _node_inserts(conn)
        assert len(inserts) == 1
        # executemany param is the LIST of tuples; each tuple is
        # (code_graph_id, id, type, Jsonb(properties)).
        batch = inserts[0][1]
        assert len(batch) == 1
        assert batch[0][0] == "cg"
        assert batch[0][1] == "n1"
        assert batch[0][2] == "function"
        assert batch[0][3].obj == {"name": "foo"}

    async def test_deduplicates_nodes(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [
            {"id": "n1", "type": "function", "properties": {}},
            {"id": "n1", "type": "function", "properties": {}},
            {"id": "n2", "type": "class", "properties": {}},
        ]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 2
        batch = _node_inserts(conn)[0][1]
        assert [row[1] for row in batch] == ["n1", "n2"]

    async def test_node_with_no_properties_gets_empty_dict(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [{"id": "n1", "type": "file"}]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1
        batch = _node_inserts(conn)[0][1]
        assert batch[0][3].obj == {}

    async def test_node_properties_enriched_with_name(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [{"id": "n1", "type": "function", "name": "my_func", "properties": {}}]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1
        props = _node_inserts(conn)[0][1][0][3].obj
        assert props["name"] == "my_func"

    async def test_node_properties_enriched_with_file_path(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [
            {
                "id": "n1",
                "type": "function",
                "file_path": "/src/main.py",
                "properties": {},
            }
        ]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1
        props = _node_inserts(conn)[0][1][0][3].obj
        assert props["file_path"] == "/src/main.py"

    async def test_existing_name_not_overwritten(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [
            {
                "id": "n1",
                "type": "function",
                "name": "override_me",
                "properties": {"name": "keep_me"},
            }
        ]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1
        props = _node_inserts(conn)[0][1][0][3].obj
        assert props["name"] == "keep_me"

    async def test_node_with_name_and_file_path_in_properties(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [
            {
                "id": "n1",
                "type": "class",
                "name": "Cls",
                "file_path": "/a.py",
                "properties": {"name": "Cls", "file_path": "/a.py"},
            }
        ]
        await _batch_insert_nodes(conn, "cg", nodes)
        props = _node_inserts(conn)[0][1][0][3].obj
        assert props["name"] == "Cls"
        assert props["file_path"] == "/a.py"

    async def test_multiple_unique_nodes(self, make_pg_conn):
        conn = make_pg_conn()
        nodes = [{"id": f"n{i}", "type": "function", "properties": {}} for i in range(10)]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 10

    async def test_node_file_path_defaults_to_empty_string(self, make_pg_conn):
        """When neither node nor properties carry file_path, no enrichment happens."""
        conn = make_pg_conn()
        nodes = [{"id": "n1", "type": "function", "properties": {}}]
        result = await _batch_insert_nodes(conn, "cg", nodes)
        assert result == 1

    async def test_insert_sql_is_parameterized_with_conflict(self, make_pg_conn):
        conn = make_pg_conn()
        await _batch_insert_nodes(conn, "cg", [{"id": "n1", "type": "function", "properties": {}}])
        sql = " ".join(_node_inserts(conn)[0][0].split())
        assert "INSERT INTO nodes (code_graph_id, id, type, properties)" in sql
        assert "ON CONFLICT (code_graph_id, tenant_id, id) DO NOTHING" in sql


# ---------------------------------------------------------------------------
# _batch_insert_edges
# ---------------------------------------------------------------------------


class TestBatchInsertEdges:
    async def test_empty_list_returns_zero(self, make_pg_conn):
        conn = make_pg_conn()
        result = await _batch_insert_edges(conn, "cg", [])
        assert result == 0
        assert _edge_inserts(conn) == []

    async def test_inserts_single_edge(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [
            {
                "source_id": "n1",
                "target_id": "n2",
                "type": "calls",
                "properties": {"line": 10},
            }
        ]
        result = await _batch_insert_edges(conn, "cg", edges)
        assert result == 1
        batch = _edge_inserts(conn)[0][1]
        assert len(batch) == 1
        # (code_graph_id, source, target, relationship, Jsonb(properties))
        assert batch[0][0] == "cg"
        assert batch[0][1] == "n1"
        assert batch[0][2] == "n2"
        assert batch[0][3] == "calls"
        assert batch[0][4].obj == {"line": 10}

    async def test_deduplicates_edges(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [
            {"source_id": "n1", "target_id": "n2", "type": "calls"},
            {"source_id": "n1", "target_id": "n2", "type": "calls"},
            {"source_id": "n1", "target_id": "n2", "type": "imports"},
        ]
        result = await _batch_insert_edges(conn, "cg", edges)
        assert result == 2

    async def test_edge_properties_serialized(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [
            {
                "source_id": "n1",
                "target_id": "n2",
                "type": "calls",
                "properties": {"weight": 3},
            }
        ]
        await _batch_insert_edges(conn, "cg", edges)
        props = _edge_inserts(conn)[0][1][0][4].obj
        assert props["weight"] == 3

    async def test_edge_without_properties(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [{"source_id": "n1", "target_id": "n2", "type": "calls"}]
        await _batch_insert_edges(conn, "cg", edges)
        props = _edge_inserts(conn)[0][1][0][4].obj
        assert props == {}

    async def test_multiple_unique_edges(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [
            {"source_id": f"n{i}", "target_id": f"n{i + 1}", "type": "calls"} for i in range(5)
        ]
        result = await _batch_insert_edges(conn, "cg", edges)
        assert result == 5

    async def test_edge_sql_statement(self, make_pg_conn):
        conn = make_pg_conn()
        edges = [{"source_id": "a", "target_id": "b", "type": "imports"}]
        await _batch_insert_edges(conn, "cg", edges)
        sql = " ".join(_edge_inserts(conn)[0][0].split())
        assert "INSERT INTO edges (code_graph_id, source, target, relationship, properties)" in sql
        assert (
            "ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO NOTHING" in sql
        )


# ---------------------------------------------------------------------------
# populate_graph_database (postgres pool, scoped by code_graph_id == container)
# ---------------------------------------------------------------------------


class TestPopulateGraphDatabase:
    async def test_success_with_nodes_and_edges(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(2,)],
                "COUNT(*) FROM edges": [(1,)],
            }
        )
        analysis_data = {
            "nodes": [
                {"id": "n1", "type": "function", "properties": {}},
                {"id": "n2", "type": "class", "properties": {}},
            ],
            "edges": [
                {"source_id": "n1", "target_id": "n2", "type": "calls"},
            ],
        }
        with patch(_GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)

        assert result is True
        assert conn.commits == 1
        # every node write is scoped to the container's code_graph_id
        node_batch = _node_inserts(conn)[0][1]
        assert all(row[0] == "test-container" for row in node_batch)
        edge_batch = _edge_inserts(conn)[0][1]
        assert all(row[0] == "test-container" for row in edge_batch)

    async def test_empty_nodes_and_edges(self, make_pg_conn):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(0,)], "COUNT(*) FROM edges": [(0,)]})
        with patch(_GET_CONN, return_value=conn):
            result = await populate_graph_database("test", {"nodes": [], "edges": []})
        assert result is True
        assert _node_inserts(conn) == []
        assert _edge_inserts(conn) == []

    async def test_missing_nodes_key_defaults_to_empty(self, make_pg_conn):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(0,)], "COUNT(*) FROM edges": [(0,)]})
        with patch(_GET_CONN, return_value=conn):
            result = await populate_graph_database("test", {})
        assert result is True

    async def test_rollback_on_inner_exception(self, make_pg_conn):
        conn = make_pg_conn()

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("insert fail")

        # Make the batch insert blow up after the connection is acquired so the
        # inner try rolls back and the outer try returns False.
        conn.executemany = _boom
        with patch(_GET_CONN, return_value=conn):
            result = await populate_graph_database(
                "fail-container",
                {"nodes": [{"id": "n1", "type": "function", "properties": {}}], "edges": []},
            )
        assert result is False
        assert conn.rollbacks == 1

    async def test_returns_false_on_connection_error(self):
        with patch(_GET_CONN, side_effect=RuntimeError("no pool")):
            result = await populate_graph_database("bad", {})
        assert result is False


# ---------------------------------------------------------------------------
# verify_database_integrity
# ---------------------------------------------------------------------------


class TestVerifyDatabaseIntegrity:
    async def test_success_no_orphaned_edges(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(5,)],
                "COUNT(*) FROM edges": [(3,)],
                # orphaned-edge count query
                "WHERE e.code_graph_id": [(0,)],
                # node-type breakdown
                "GROUP BY type": [("function", 3), ("class", 2)],
            }
        )
        with patch(_GET_CONN, return_value=conn):
            result = await verify_database_integrity("test-container")

        assert result["status"] == "success"
        assert result["node_count"] == 5
        assert result["edge_count"] == 3
        assert result["orphaned_edges"] == 0
        assert result["node_types"] == {"function": 3, "class": 2}
        assert result["code_graph_id"] == "test-container"

    async def test_cleans_orphaned_edges(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(4,)],
                "COUNT(*) FROM edges": [(5,)],
                "WHERE e.code_graph_id": [(2,)],  # orphaned > 0 triggers cleanup
                "GROUP BY type": [("function", 2)],
            }
        )
        with patch(_GET_CONN, return_value=conn):
            result = await verify_database_integrity("test-container")

        assert result["status"] == "success"
        assert result["orphaned_edges"] == 0  # reset after cleanup
        assert conn.commits == 1
        # a DELETE was issued for the orphaned edges
        assert any("DELETE FROM edges" in " ".join(sql.split()) for sql, _ in conn.executed)

    async def test_exception_returns_error(self):
        with patch(_GET_CONN, side_effect=RuntimeError("boom")):
            result = await verify_database_integrity("bad")
        assert "error" in result
        assert "boom" in result["error"]

    async def test_node_types_dict(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(9,)],
                "COUNT(*) FROM edges": [(4,)],
                "WHERE e.code_graph_id": [(0,)],
                "GROUP BY type": [("function", 5), ("class", 3), ("module", 1)],
            }
        )
        with patch(_GET_CONN, return_value=conn):
            result = await verify_database_integrity("test")
        assert result["node_types"] == {"function": 5, "class": 3, "module": 1}

    async def test_queries_are_scoped_to_code_graph_id(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(0,)],
                "COUNT(*) FROM edges": [(0,)],
                "WHERE e.code_graph_id": [(0,)],
                "GROUP BY type": [],
            }
        )
        with patch(_GET_CONN, return_value=conn):
            await verify_database_integrity("my-graph")
        node_count_params = next(
            params
            for sql, params in conn.executed
            if "COUNT(*) FROM nodes" in " ".join(sql.split())
        )
        assert node_count_params == ("my-graph",)


# ---------------------------------------------------------------------------
# optimize_database_performance (now a no-op under shared PostgreSQL)
# ---------------------------------------------------------------------------


class TestOptimizeDatabasePerformance:
    async def test_is_noop_returns_true(self):
        # No connection is opened: VACUUM/ANALYZE can't run inside the pooled
        # PgBouncer transaction-mode connection; maintenance is autovacuum's job.
        with patch(_GET_CONN, side_effect=AssertionError("must not open a connection")):
            result = await optimize_database_performance("test-container")
        assert result is True


# ---------------------------------------------------------------------------
# get_database_stats
# ---------------------------------------------------------------------------


class TestGetDatabaseStats:
    async def test_success(self, make_pg_conn):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(10,)],
                "COUNT(*) FROM edges": [(20,)],
                # relationship-type breakdown
                "GROUP BY relationship": [("calls", 15), ("imports", 5)],
                # top connected nodes: (id, type, name, connection_count)
                "ORDER BY connection_count DESC": [
                    ("n1", "function", "main", 8),
                    ("n2", "class", "Foo", 5),
                ],
            }
        )
        with patch(_GET_CONN, return_value=conn):
            result = await get_database_stats("test-container")

        assert result["container_name"] == "test-container"
        assert result["code_graph_id"] == "test-container"
        assert result["node_count"] == 10
        assert result["edge_count"] == 20
        # file size is always 0 under shared postgres (no per-graph file)
        assert result["file_size_bytes"] == 0
        assert result["file_size_mb"] == 0
        assert result["relationship_types"] == {"calls": 15, "imports": 5}
        assert len(result["top_connected_nodes"]) == 2
        assert result["top_connected_nodes"][0]["name"] == "main"
        assert result["top_connected_nodes"][1]["connections"] == 5

    async def test_exception_returns_error(self):
        with patch(_GET_CONN, side_effect=RuntimeError("stats fail")):
            result = await get_database_stats("bad")
        assert "error" in result
        assert "stats fail" in result["error"]
