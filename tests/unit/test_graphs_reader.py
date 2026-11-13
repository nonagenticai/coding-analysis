"""Tests for the coding-analysis graph reader endpoints.

These endpoints are the read side of the shared PostgreSQL code-graph store
that ``populate_graph_database`` writes (rows scoped by ``code_graph_id`` ==
container name). They replace coding-gateway's direct file access — the gateway
now proxies to them over HTTP.

The reader talks to PostgreSQL through the ``db.session.get_raw_connection()``
pool with positional tuple rows. These unit tests patch that pool with the
shared ``FakePgConn`` (see ``tests/conftest.py``) and route results by SQL
substring. Because WHERE-clause filtering happens in SQL (not Python), the
fake is configured per test to return the rows the endpoint should shape into
its response; the assertions verify the endpoint's row→model wiring and counts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GET_CONN = "app.api.endpoints.graphs_reader.get_raw_connection"
_VALIDATE = "app.api.endpoints.graphs_reader.validate_and_get_container_for_tenant"


@pytest.fixture(autouse=True)
def _bypass_container_guard():
    """By default, let the per-container tenant-ownership guard pass.

    Each endpoint calls ``validate_and_get_container_for_tenant`` (registry
    lookup) before touching the DB. Tests that exercise row→model wiring patch
    it to a no-op; the cross-tenant rejection path is covered explicitly in
    ``TestTenantIsolation``.
    """
    with patch(_VALIDATE, new=AsyncMock(return_value=None)):
        yield


# Fixture graph rows (positional tuples, matching the SELECT column order).
# nodes:  (id, type, properties)   — properties is JSONB -> a dict
# edges:  (source, target, relationship, properties, weight)
_NODE_ROWS = [
    ("n1", "file", {"name": "main.py"}),
    ("n2", "function", {"name": "run"}),
    ("n3", "class", {"name": "App"}),
]
_EDGE_ROWS = [
    ("n1", "n2", "contains", {"start_line": 10}, 1.0),
    ("n1", "n3", "contains", None, 1.0),
]


@pytest.fixture
def graphs_client():
    """Mount the graphs_reader router with tenant auth bypassed.

    The connection pool is patched per test (see ``client_with_conn``); this
    fixture only builds the app + client.
    """
    from app.api.endpoints.graphs_reader import router
    from auth.tenant import require_tenant

    app = FastAPI()
    # Bypass tenant auth for these unit tests.
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    app.include_router(router, prefix="/api/analysis/graphs")

    return TestClient(app)


@pytest.fixture
def routed_conn(make_pg_conn):
    """Build a FakePgConn whose graph 'exists' (the 404 probe finds a row)."""

    def _make(routes, *, exists=True):
        full = dict(routes)
        # ``_require_graph_or_404`` probes "SELECT 1 FROM nodes ... LIMIT 1".
        full.setdefault("SELECT 1 FROM nodes", [(1,)] if exists else [])
        return make_pg_conn(routes=full)

    return _make


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestGraphMetadata:
    def test_existing_container_counts(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(3,)],
                "COUNT(*) FROM edges": [(2,)],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/demo-container")
        assert r.status_code == 200
        assert r.json() == {
            "container_name": "demo-container",
            "exists": True,
            "node_count": 3,
            "edge_count": 2,
        }

    def test_missing_container_404(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn({}, exists=False)
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/does-not-exist")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_name_pattern_rejects_slashes(self, graphs_client):
        r = graphs_client.get("/api/analysis/graphs/..%2Fetc%2Fpasswd")
        # Starlette decodes %2F back to / which then never matches the route;
        # either way we must not leak paths out of the graphs dir.
        assert r.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class TestListNodes:
    def test_returns_all_nodes_with_properties_decoded(
        self, graphs_client, routed_conn, monkeypatch
    ):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(3,)],
                "SELECT id, type, properties FROM nodes": list(_NODE_ROWS),
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/demo-container/nodes")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["offset"] == 0
        assert len(body["nodes"]) == 3
        n1 = next(n for n in body["nodes"] if n["id"] == "n1")
        assert n1["type"] == "file"
        assert n1["properties"] == {"name": "main.py"}

    def test_filter_by_type(self, graphs_client, routed_conn, monkeypatch):
        # SQL applies the type filter; the fake returns the already-filtered set.
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(1,)],
                "SELECT id, type, properties FROM nodes": [_NODE_ROWS[1]],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/nodes", params={"node_type": "function"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert [n["id"] for n in body["nodes"]] == ["n2"]
        # The user-supplied type stays bound, not interpolated.
        data_sql, data_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "SELECT id, type, properties FROM nodes" in " ".join(sql.split())
        )
        assert "type = %s" in data_sql
        assert "function" in data_params

    def test_pagination(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(3,)],
                "SELECT id, type, properties FROM nodes": [_NODE_ROWS[1]],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/nodes",
            params={"limit": 1, "offset": 1},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["limit"] == 1
        assert body["offset"] == 1
        assert len(body["nodes"]) == 1
        # limit/offset are bound as the final two params.
        data_params = next(
            params
            for sql, params in conn.executed
            if "SELECT id, type, properties FROM nodes" in " ".join(sql.split())
        )
        assert data_params[-2:] == [1, 1]

    def test_filter_by_name_pattern_case_insensitive_substring(
        self, graphs_client, routed_conn, monkeypatch
    ):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(1,)],
                "SELECT id, type, properties FROM nodes": [_NODE_ROWS[0]],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/nodes",
            params={"name_pattern": "MAIN"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["nodes"][0]["id"] == "n1"
        # name_pattern lowercased + wrapped in %...% and bound (no interpolation).
        data_params = next(
            params
            for sql, params in conn.executed
            if "SELECT id, type, properties FROM nodes" in " ".join(sql.split())
        )
        assert "%main%" in data_params

    def test_name_pattern_combined_with_type(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(0,)],
                "SELECT id, type, properties FROM nodes": [],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/nodes",
            params={"node_type": "function", "name_pattern": "nomatch"},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_name_pattern_literal_percent_does_not_wildcard(
        self, graphs_client, routed_conn, monkeypatch
    ):
        # "%" must be escaped to a literal; our seed names contain no literal
        # percent sign, so the result is empty. Proves user input can't smuggle
        # wildcards past the outer %...% wrap.
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(0,)],
                "SELECT id, type, properties FROM nodes": [],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/nodes",
            params={"name_pattern": "%"},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 0
        # The literal "%" is escaped before binding, so the bound value is "%\%%".
        data_params = next(
            params
            for sql, params in conn.executed
            if "SELECT id, type, properties FROM nodes" in " ".join(sql.split())
        )
        assert any("\\%" in str(p) for p in data_params)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestListEdges:
    def test_all_edges_with_null_properties_decoded_as_empty(
        self, graphs_client, routed_conn, monkeypatch
    ):
        conn = routed_conn(
            {
                "COUNT(*) FROM edges": [(2,)],
                "SELECT source, target, relationship, properties, weight FROM edges": list(
                    _EDGE_ROWS
                ),
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/demo-container/edges")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        e_with_props = next(e for e in body["edges"] if e["target"] == "n2")
        assert e_with_props["properties"] == {"start_line": 10}
        e_without = next(e for e in body["edges"] if e["target"] == "n3")
        # NULL properties decode to an empty dict.
        assert e_without["properties"] == {}

    def test_filter_by_source(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM edges": [(2,)],
                "SELECT source, target, relationship, properties, weight FROM edges": list(
                    _EDGE_ROWS
                ),
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/demo-container/edges", params={"source": "n1"})
        assert r.status_code == 200
        assert r.json()["total"] == 2
        data_sql, data_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "SELECT source, target, relationship, properties, weight FROM edges"
            in " ".join(sql.split())
        )
        assert "source = %s" in data_sql
        assert "n1" in data_params

    def test_filter_by_relationship_and_target(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM edges": [(1,)],
                "SELECT source, target, relationship, properties, weight FROM edges": [
                    _EDGE_ROWS[1]
                ],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/edges",
            params={"relationship": "contains", "target": "n3"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["edges"][0]["source"] == "n1"
        data_sql, data_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "SELECT source, target, relationship, properties, weight FROM edges"
            in " ".join(sql.split())
        )
        assert "target = %s" in data_sql and "relationship = %s" in data_sql
        assert "n3" in data_params and "contains" in data_params


# ---------------------------------------------------------------------------
# Augmented (one-shot nodes + edges)
# ---------------------------------------------------------------------------


class TestAugmented:
    def test_returns_nodes_and_edges_and_counts(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(3,)],
                "COUNT(*) FROM edges": [(2,)],
                "SELECT id, type, properties FROM nodes": list(_NODE_ROWS),
                "SELECT source, target, relationship, properties, weight FROM edges": list(
                    _EDGE_ROWS
                ),
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/demo-container/augmented")
        assert r.status_code == 200
        body = r.json()
        assert body["container_name"] == "demo-container"
        assert body["node_count"] == 3
        assert body["edge_count"] == 2
        assert {n["id"] for n in body["nodes"]} == {"n1", "n2", "n3"}
        assert {(e["source"], e["target"]) for e in body["edges"]} == {
            ("n1", "n2"),
            ("n1", "n3"),
        }

    def test_node_limit_caps_response(self, graphs_client, routed_conn, monkeypatch):
        # The LIMIT is applied in SQL; the fake returns the already-capped set,
        # while the COUNT(*) still reflects the full graph.
        conn = routed_conn(
            {
                "COUNT(*) FROM nodes": [(3,)],
                "COUNT(*) FROM edges": [(2,)],
                "SELECT id, type, properties FROM nodes": _NODE_ROWS[:2],
                "SELECT source, target, relationship, properties, weight FROM edges": list(
                    _EDGE_ROWS
                ),
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get(
            "/api/analysis/graphs/demo-container/augmented",
            params={"node_limit": 2},
        )
        assert r.status_code == 200
        body = r.json()
        # Counts reflect the full graph; returned arrays are capped.
        assert body["node_count"] == 3
        assert len(body["nodes"]) == 2
        # node_limit is bound on the nodes data query.
        node_params = next(
            params
            for sql, params in conn.executed
            if "SELECT id, type, properties FROM nodes" in " ".join(sql.split())
        )
        assert node_params[-1] == 2

    def test_missing_container_404(self, graphs_client, routed_conn, monkeypatch):
        conn = routed_conn({}, exists=False)
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs/missing/augmented")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Management: list
# ---------------------------------------------------------------------------


def _executed_sql(conn, needle):
    """Return (sql, params) of the first executed statement containing needle."""
    return next((sql, params) for sql, params in conn.executed if needle in " ".join(sql.split()))


class TestListGraphs:
    def test_lists_graphs_with_node_and_edge_counts(self, graphs_client, make_pg_conn, monkeypatch):
        # list_graphs is scoped to the caller's tenant (``WHERE tenant_id = %s``),
        # so only the caller's ("test-tenant") graphs are returned.
        conn = make_pg_conn(
            routes={
                "FROM nodes WHERE tenant_id": [
                    ("c1", 3, "test-tenant"),
                    ("c2", 1, "test-tenant"),
                ],
                "FROM edges GROUP BY code_graph_id": [("c1", 2)],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        by_name = {g["container_name"]: g for g in body["graphs"]}
        assert by_name["c1"] == {
            "container_name": "c1",
            "node_count": 3,
            "edge_count": 2,
            "tenant_id": "test-tenant",
        }
        # c2 has no edges -> edge_count defaults to 0.
        assert by_name["c2"]["edge_count"] == 0
        assert by_name["c2"]["tenant_id"] == "test-tenant"

    def test_empty_store(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(
            routes={
                "FROM nodes GROUP BY code_graph_id": [],
                "FROM edges GROUP BY code_graph_id": [],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.get("/api/analysis/graphs")
        assert r.status_code == 200
        assert r.json() == {"graphs": [], "total": 0}


# ---------------------------------------------------------------------------
# Management: initialize
# ---------------------------------------------------------------------------


class TestInitializeGraph:
    def test_creates_new_graph(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(0,)]})
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.post("/api/analysis/graphs/new-container", json={})
        assert r.status_code == 201
        body = r.json()
        assert body == {
            "container_name": "new-container",
            "created": True,
            "repo_node_id": "repo_new-container",
        }
        # The repository node is inserted with the request tenant + repository type.
        insert_sql, insert_params = _executed_sql(conn, "INSERT INTO nodes")
        assert "ON CONFLICT (code_graph_id, tenant_id, id) DO NOTHING" in insert_sql
        assert insert_params[0] == "new-container"
        assert insert_params[1] == "repo_new-container"
        assert insert_params[2] == "repository"
        assert insert_params[4] == "test-tenant"

    def test_no_body_uses_defaults(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(0,)]})
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.post("/api/analysis/graphs/new-container")
        assert r.status_code == 201
        assert r.json()["created"] is True

    def test_conflict_when_graph_exists(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(5,)]})
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.post("/api/analysis/graphs/existing", json={})
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"].lower()
        # No insert happened.
        assert not any("INSERT INTO nodes" in " ".join(sql.split()) for sql, _ in conn.executed)

    def test_force_rebuild_clears_then_inserts(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(5,)]})
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.post("/api/analysis/graphs/existing", json={"force_rebuild": True})
        assert r.status_code == 201
        # created reflects pre-existing graph (rebuilt, not freshly created).
        assert r.json()["created"] is False
        executed = [" ".join(sql.split()) for sql, _ in conn.executed]
        assert any("DELETE FROM edges WHERE code_graph_id" in s for s in executed)
        assert any("DELETE FROM nodes WHERE code_graph_id" in s for s in executed)
        assert any("INSERT INTO nodes" in s for s in executed)
        # Edges deleted before nodes (FK ordering).
        del_edges = next(i for i, s in enumerate(executed) if "DELETE FROM edges" in s)
        del_nodes = next(i for i, s in enumerate(executed) if "DELETE FROM nodes" in s)
        assert del_edges < del_nodes


# ---------------------------------------------------------------------------
# Management: delete
# ---------------------------------------------------------------------------


class TestDeleteGraph:
    def test_deletes_existing_graph(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(3,)],
                "COUNT(*) FROM edges": [(2,)],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.delete("/api/analysis/graphs/c1")
        assert r.status_code == 200
        assert r.json() == {
            "container_name": "c1",
            "deleted": True,
            "nodes_deleted": 3,
            "edges_deleted": 2,
        }
        executed = [" ".join(sql.split()) for sql, _ in conn.executed]
        del_edges = next(i for i, s in enumerate(executed) if "DELETE FROM edges" in s)
        del_nodes = next(i for i, s in enumerate(executed) if "DELETE FROM nodes" in s)
        assert del_edges < del_nodes

    def test_404_when_graph_absent(self, graphs_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(
            routes={
                "COUNT(*) FROM nodes": [(0,)],
                "COUNT(*) FROM edges": [(0,)],
            }
        )
        monkeypatch.setattr(_GET_CONN, lambda: conn, raising=True)
        r = graphs_client.delete("/api/analysis/graphs/missing")
        assert r.status_code == 404
        # Nothing was deleted.
        assert not any("DELETE FROM" in " ".join(sql.split()) for sql, _ in conn.executed)


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    """The per-container guard rejects access to another tenant's container."""

    def test_metadata_cross_tenant_forbidden(self, graphs_client):
        from fastapi import HTTPException

        guard = AsyncMock(
            side_effect=HTTPException(
                status_code=403, detail="Access denied: container does not belong to tenant"
            )
        )
        with patch(_VALIDATE, new=guard):
            r = graphs_client.get("/api/analysis/graphs/someone-elses-container")
        assert r.status_code == 403

    def test_delete_cross_tenant_forbidden(self, graphs_client):
        from fastapi import HTTPException

        guard = AsyncMock(side_effect=HTTPException(status_code=403, detail="denied"))
        with patch(_VALIDATE, new=guard):
            r = graphs_client.delete("/api/analysis/graphs/someone-elses-container")
        assert r.status_code == 403
