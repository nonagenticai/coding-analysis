"""Tests for the coding-analysis graph-metrics endpoints.

These endpoints are a thin HTTP layer over the postgres-backed
``CodeGraphMetricsService`` (already unit-tested in its own module). They expose
graph metrics so coding-gateway can proxy here instead of opening SQLite.

These unit tests cover the HTTP layer: tenant auth, the 404 probe, and the
service-result → response-model mapping. The metric *math* (counts, centrality,
betweenness) lives in the service and is exercised end-to-end against real
postgres in ``tests/integration/test_graph_metrics_integration.py``. Here the
service is faked and the facade's connection is the shared ``FakePgConn`` (so the
404 probe routes by SQL substring).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import graph_metrics as mod
from models.core_models import OverallGraphMetrics

_VALIDATE = "app.api.endpoints.graph_metrics.validate_and_get_container_for_tenant"


@pytest.fixture(autouse=True)
def _bypass_container_guard():
    """Let the per-container tenant-ownership guard pass by default; the
    cross-tenant rejection path is covered in ``TestTenantIsolation``."""
    with patch(_VALIDATE, new=AsyncMock(return_value=None)):
        yield


class _FakeDb:
    """Stand-in for the facade: the endpoint only needs connect/get_connection/close."""

    def __init__(self, conn, code_graph_id: str) -> None:
        self.conn = conn
        self.code_graph_id = code_graph_id

    async def connect(self) -> None:
        return None

    def get_connection(self):
        return self.conn

    async def close(self) -> None:
        return None


class _FakeService:
    """Callable stand-in for ``CodeGraphMetricsService`` (the ``(db)`` ctor returns self)."""

    def __init__(self, *, overall=None, centrality=None, entity=None) -> None:
        self._overall = overall
        self._centrality = centrality or {}
        self._entity = entity or {}

    def __call__(self, db):
        return self

    async def get_graph_metrics(self):
        return self._overall

    async def calculate_centrality_metrics(self, entity_type):
        return self._centrality

    async def get_function_network_metrics(self, name):
        return self._entity


@pytest.fixture
def client():
    from auth.tenant import require_tenant

    app = FastAPI()
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    app.include_router(mod.router, prefix="/api/analysis/graphs")
    return TestClient(app)


@pytest.fixture
def wire(monkeypatch, make_pg_conn):
    """Patch the facade + service in the endpoint module; route the 404 probe."""

    def _wire(*, exists=True, overall=None, centrality=None, entity=None):
        conn = make_pg_conn(routes={"SELECT 1 FROM nodes": [(1,)] if exists else []})
        monkeypatch.setattr(mod, "GraphDatabaseConnection", lambda cn, **k: _FakeDb(conn, cn))
        monkeypatch.setattr(
            mod,
            "CodeGraphMetricsService",
            _FakeService(overall=overall, centrality=centrality, entity=entity),
        )
        return conn

    return _wire


# ---------------------------------------------------------------------------
# Overall graph metrics
# ---------------------------------------------------------------------------


class TestGraphMetrics:
    def test_returns_service_result(self, client, wire):
        wire(
            overall=OverallGraphMetrics(
                node_count=4,
                edge_count=3,
                density=0.25,
                average_degree=1.5,
                node_types={"file": 1, "function": 3},
                relationship_types={"calls": 3},
                calculation_time_seconds=0.01,
            )
        )
        r = client.get("/api/analysis/graphs/demo/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["node_count"] == 4
        assert body["edge_count"] == 3
        assert body["density"] == pytest.approx(0.25)
        assert body["node_types"] == {"file": 1, "function": 3}

    def test_missing_graph_404(self, client, wire):
        wire(exists=False)
        r = client.get("/api/analysis/graphs/nope/metrics")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Centrality for all nodes of a type
# ---------------------------------------------------------------------------


class TestCentralityMetrics:
    def test_maps_per_node_dict_to_models(self, client, wire):
        wire(
            centrality={
                "f1": {
                    "degree_centrality": 0.75,
                    "in_degree": 1,
                    "out_degree": 2,
                    "total_degree": 3,
                },
                "f2": {
                    "degree_centrality": 0.0,
                    "in_degree": 0,
                    "out_degree": 0,
                    "total_degree": 0,
                },
            }
        )
        r = client.get(
            "/api/analysis/graphs/demo/metrics/centrality", params={"entity_type": "function"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["entity_type"] == "function"
        assert set(body["metrics"]) == {"f1", "f2"}
        assert body["metrics"]["f1"]["degree_centrality"] == pytest.approx(0.75)
        assert body["metrics"]["f1"]["total_degree"] == 3

    def test_requires_entity_type(self, client, wire):
        wire()
        r = client.get("/api/analysis/graphs/demo/metrics/centrality")
        assert r.status_code == 422

    def test_missing_graph_404(self, client, wire):
        wire(exists=False)
        r = client.get(
            "/api/analysis/graphs/nope/metrics/centrality", params={"entity_type": "function"}
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Entity network metrics
# ---------------------------------------------------------------------------


class TestEntityNetworkMetrics:
    def test_maps_service_dict_to_model(self, client, wire):
        wire(
            entity={
                "degree_centrality": 1.0,
                "betweenness_centrality": 1.0,
                "closeness_centrality": 0.6,
                "pagerank": 0.7,
                "network_importance": 0.89,
                "in_degree": 1,
                "out_degree": 1,
                "total_degree": 2,
            }
        )
        r = client.get("/api/analysis/graphs/demo/metrics/entity/b")
        assert r.status_code == 200
        body = r.json()
        assert body["entity_name"] == "b"
        assert body["betweenness_centrality"] == pytest.approx(1.0)
        assert body["network_importance"] == pytest.approx(0.89)
        assert body["total_degree"] == 2

    def test_missing_graph_404(self, client, wire):
        wire(exists=False)
        r = client.get("/api/analysis/graphs/nope/metrics/entity/x")
        assert r.status_code == 404


class TestTenantIsolation:
    """The per-container guard rejects metrics for another tenant's container."""

    def test_metrics_cross_tenant_forbidden(self, client):
        from fastapi import HTTPException

        guard = AsyncMock(side_effect=HTTPException(status_code=403, detail="denied"))
        with patch(_VALIDATE, new=guard):
            r = client.get("/api/analysis/graphs/someone-elses-container/metrics")
        assert r.status_code == 403
