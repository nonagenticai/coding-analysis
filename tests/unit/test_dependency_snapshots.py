"""Tests for the dependency-report snapshot reader + writer.

The reader (``app/api/endpoints/dependency_snapshots.py``) is the graph-style
fallback for the fr Dependency Report: it serves the last report persisted at
analysis time so the UI degrades gracefully once the job container is gone. The
writer (``services/code_graph/dependency_snapshot.py``) fetches the live report
from coding-security and upserts it — best-effort, never raising.

Like ``test_graphs_reader``, these patch ``db.session.get_raw_connection`` (via
each module's imported name) with the shared ``FakePgConn`` and route by SQL
substring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_READER_CONN = "app.api.endpoints.dependency_snapshots.get_raw_connection"
_WRITER_MOD = "services.code_graph.dependency_snapshot"

_REPORT = {
    "container_name": "demo-container",
    "packages": [{"name": "requests", "version": "2.31.0", "ecosystem": "pypi"}],
    "vulnerabilities": [],
}
_COMPUTED_AT = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_client():
    """Mount the reader router with tenant auth pinned to ``test-tenant``."""
    from app.api.endpoints.dependency_snapshots import router
    from auth.tenant import require_tenant

    app = FastAPI()
    app.dependency_overrides[require_tenant] = lambda: "test-tenant"
    app.include_router(router, prefix="/api/analysis/dependency-report")
    return TestClient(app)


class TestDependencySnapshotReader:
    def test_returns_stored_snapshot(self, snapshot_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(
            routes={
                "FROM dependency_report_snapshots": [(_REPORT, _COMPUTED_AT)],
            }
        )
        monkeypatch.setattr(_READER_CONN, lambda: conn, raising=True)
        r = snapshot_client.get("/api/analysis/dependency-report/demo-container")
        assert r.status_code == 200
        body = r.json()
        assert body["container_name"] == "demo-container"
        assert body["report"]["packages"][0]["name"] == "requests"
        assert body["computed_at"].startswith("2026-07-01T12:00:00")
        # Query is scoped by both container and the caller's tenant.
        sql, params = conn.executed[0]
        assert "code_graph_id = %s AND tenant_id = %s" in " ".join(sql.split())
        assert params == ("demo-container", "test-tenant")

    def test_missing_snapshot_404(self, snapshot_client, make_pg_conn, monkeypatch):
        conn = make_pg_conn(routes={"FROM dependency_report_snapshots": []})
        monkeypatch.setattr(_READER_CONN, lambda: conn, raising=True)
        r = snapshot_client.get("/api/analysis/dependency-report/never-analyzed")
        assert r.status_code == 404
        assert "no stored dependency report" in r.json()["detail"].lower()

    def test_name_pattern_rejects_slashes(self, snapshot_client):
        r = snapshot_client.get("/api/analysis/dependency-report/..%2Fetc%2Fpasswd")
        assert r.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.calls: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        self.calls.append((url, headers))
        return self._resp


def _patch_settings(monkeypatch, url="http://coding-security.svc:8693"):
    monkeypatch.setattr(
        f"{_WRITER_MOD}.get_settings",
        lambda: SimpleNamespace(CODING_SECURITY_URL=url),
        raising=True,
    )


def _patch_client(monkeypatch, resp):
    client = _FakeClient(resp)
    monkeypatch.setattr(f"{_WRITER_MOD}.httpx.AsyncClient", lambda *a, **k: client, raising=True)
    return client


class TestDependencySnapshotWriter:
    @pytest.mark.asyncio
    async def test_upserts_report_on_success(self, make_pg_conn, monkeypatch):
        from services.code_graph.dependency_snapshot import store_dependency_report_snapshot

        _patch_settings(monkeypatch)
        client = _patch_client(monkeypatch, _FakeResp(200, _REPORT))
        conn = make_pg_conn()
        monkeypatch.setattr(f"{_WRITER_MOD}.get_raw_connection", lambda: conn, raising=True)

        ok = await store_dependency_report_snapshot("demo-container", "tenant-a")
        assert ok is True
        # Called security with the tenant header.
        url, headers = client.calls[0]
        assert url.endswith("/api/dependencies/demo-container/report")
        assert headers == {"X-Tenant-Id": "tenant-a"}
        # Upserted with ON CONFLICT + committed.
        sql, params = conn.executed[0]
        assert "INSERT INTO dependency_report_snapshots" in " ".join(sql.split())
        assert "ON CONFLICT (code_graph_id, tenant_id) DO UPDATE" in " ".join(sql.split())
        assert params[0] == "demo-container"
        assert params[1] == "tenant-a"
        assert conn.commits == 1

    @pytest.mark.asyncio
    async def test_non_200_is_swallowed_no_write(self, make_pg_conn, monkeypatch):
        from services.code_graph.dependency_snapshot import store_dependency_report_snapshot

        _patch_settings(monkeypatch)
        _patch_client(monkeypatch, _FakeResp(404, {"detail": "container gone"}))
        conn = make_pg_conn()
        monkeypatch.setattr(f"{_WRITER_MOD}.get_raw_connection", lambda: conn, raising=True)

        ok = await store_dependency_report_snapshot("demo-container", "tenant-a")
        assert ok is False
        assert conn.executed == []
        assert conn.commits == 0

    @pytest.mark.asyncio
    async def test_missing_security_url_is_noop(self, make_pg_conn, monkeypatch):
        from services.code_graph.dependency_snapshot import store_dependency_report_snapshot

        _patch_settings(monkeypatch, url="")
        conn = make_pg_conn()
        monkeypatch.setattr(f"{_WRITER_MOD}.get_raw_connection", lambda: conn, raising=True)

        ok = await store_dependency_report_snapshot("demo-container", "tenant-a")
        assert ok is False
        assert conn.executed == []


class TestDependencySnapshotWriterRequiresTenant:
    """A snapshot is keyed by tenant, so an unknown tenant must skip, not guess.

    Previously a falsy tenant sent NO headers -- reaching coding-security only via
    its trusted-network (source-IP) fallback -- and then wrote the row under a NULL
    tenant. Unlike the fire-and-forget scan kickoffs, this one must NOT fall back to
    DEFAULT_TENANT_ID: that would file one tenant's dependency report against another.
    """

    @pytest.mark.parametrize("tenant_id", [None, ""])
    @pytest.mark.asyncio
    async def test_skips_without_calling_security_or_writing(
        self, make_pg_conn, monkeypatch, tenant_id
    ):
        from services.code_graph.dependency_snapshot import store_dependency_report_snapshot

        _patch_settings(monkeypatch)
        client = _patch_client(monkeypatch, _FakeResp(200, _REPORT))
        conn = make_pg_conn()
        monkeypatch.setattr(f"{_WRITER_MOD}.get_raw_connection", lambda: conn, raising=True)

        ok = await store_dependency_report_snapshot("demo-container", tenant_id)

        assert ok is False
        assert client.calls == [], "must not call coding-security without a tenant"
        assert conn.executed == [], "must not write a row under an unknown tenant"
        assert conn.commits == 0
