"""Integration tests for the graph-metrics endpoints against a real PostgreSQL.

Drives the HTTP endpoints (real facade + real ``CodeGraphMetricsService``) over a
migrated postgres, with two graphs sharing a node id, and asserts the metrics for
one graph never include the other's rows — the same cross-graph isolation guard
the rest of the migration relies on.

Skipped unless ``TEST_DATABASE_URL`` points at a reachable postgres (the CI
``postgres:18`` service provides it). Marked ``integration``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# This suite covers ROW-LEVEL cross-graph isolation over real postgres. The
# per-container tenant-ownership guard is a live HTTP call to coding-containers
# (absent in CI) and its cross-tenant rejection path is covered by the unit
# tests' ``TestTenantIsolation`` — so bypass it here exactly as every unit suite
# does (tests/unit/test_graph_metrics.py). The ``does-not-exist`` 404 still comes
# from the row-existence probe (``_require_graph_or_404``), not this guard.
_VALIDATE = "app.api.endpoints.graph_metrics.validate_and_get_container_for_tenant"


@pytest.fixture(autouse=True)
def _bypass_container_guard():
    with patch(_VALIDATE, new=AsyncMock(return_value=None)):
        yield


_DSN = os.getenv("TEST_DATABASE_URL")
if not _DSN:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _migrated_db():
    os.environ["DATABASE_URL"] = _DSN
    os.environ["PGBOUNCER_URL"] = ""  # local PG: no pgbouncer, use direct URL

    from app.core.config.core_config_settings import get_settings

    get_settings.cache_clear()

    from alembic.config import Config

    from alembic import command

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    command.upgrade(cfg, "head")
    yield
    command.downgrade(cfg, "base")


@pytest.fixture(autouse=True)
async def _clean_tables():
    from db import session

    async with session.get_raw_connection() as conn:
        await conn.execute(
            "TRUNCATE edges, nodes, analysis_sessions, graph_versions RESTART IDENTITY CASCADE"
        )
        await conn.commit()
    yield


@pytest.fixture
def client():
    from app.api.endpoints.graph_metrics import router
    from auth.tenant import require_tenant

    app = FastAPI()
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    app.include_router(router, prefix="/api/analysis/graphs")
    return TestClient(app)


async def _seed_graph(code_graph_id, nodes, edges):
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection(code_graph_id)
    await db.connect()
    try:
        for node_id, node_type, props in nodes:
            db.add_node(node_id, node_type, props)
        for source, target, rel in edges:
            db.add_edge(source, target, rel)
        await db.flush_pending_operations()
    finally:
        await db.close()


async def _seed_two_graphs():
    # graph-a: 3 nodes / 2 edges, including an edge INTO the shared node.
    await _seed_graph(
        "graph-a",
        nodes=[
            ("a1", "function", {"name": "a_one"}),
            ("a2", "class", {"name": "a_two"}),
            ("shared", "function", {"name": "a_shared"}),
        ],
        edges=[("a1", "a2", "calls"), ("a2", "shared", "calls")],
    )
    # graph-b: 2 nodes / 1 edge; same "shared" id, distinct neighbours.
    await _seed_graph(
        "graph-b",
        nodes=[
            ("b1", "function", {"name": "b_one"}),
            ("shared", "function", {"name": "b_shared"}),
        ],
        edges=[("b1", "shared", "calls")],
    )


async def test_overall_metrics_scoped_to_graph(client):
    await _seed_two_graphs()

    r = client.get("/api/analysis/graphs/graph-b/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["node_count"] == 2  # graph-b only (graph-a has 3)
    assert body["edge_count"] == 1  # graph-b only (graph-a has 2)
    assert body["node_types"] == {"function": 2}
    assert body["relationship_types"] == {"calls": 1}


async def test_entity_metrics_scoped_to_graph(client):
    await _seed_two_graphs()

    # In graph-b, "shared" has exactly one incoming edge (b1->shared) and no
    # outgoing — graph-a's a2->shared edge must not be counted.
    r = client.get("/api/analysis/graphs/graph-b/metrics/entity/shared")
    assert r.status_code == 200
    body = r.json()
    assert body["in_degree"] == 1
    assert body["out_degree"] == 0
    assert body["total_degree"] == 1
    # total_nodes in graph-b is 2 -> max_degree 1 -> degree_centrality 1.0
    assert body["degree_centrality"] == pytest.approx(1.0)


async def test_centrality_scoped_to_graph(client):
    await _seed_two_graphs()

    r = client.get(
        "/api/analysis/graphs/graph-b/metrics/centrality", params={"entity_type": "function"}
    )
    assert r.status_code == 200
    body = r.json()
    # Only graph-b's function nodes (b1, shared); graph-a's a1 must not appear.
    assert set(body["metrics"]) == {"b1", "shared"}


async def test_missing_graph_404(client):
    await _seed_two_graphs()
    r = client.get("/api/analysis/graphs/does-not-exist/metrics")
    assert r.status_code == 404
