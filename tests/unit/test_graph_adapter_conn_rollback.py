"""Regression tests for the architecture graph-adapter DB cascade bug.

A failed query in ``_get_all_entities`` / ``_get_all_relationships`` used to leave
the long-lived pooled connection (held by the per-code_graph singleton) in an
aborted transaction, so every subsequent architecture request failed with
"current transaction is aborted" until the pod restarted. These tests assert the
connection is rolled back on error so it is returned to a usable state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from db.facade import GraphDatabaseConnection
from processing.architecture.graph_adapter import GraphAdapter


def _service_with_conn(conn: MagicMock, code_graph_id: str) -> MagicMock:
    """A service whose ``db`` is a REAL facade holding ``conn`` (alive)."""
    conn.closed = False
    conn.broken = False
    db = GraphDatabaseConnection(code_graph_id)
    db.conn = conn
    service = MagicMock()
    service.db = db
    return service


def _adapter_with_failing_conn() -> tuple[GraphAdapter, MagicMock]:
    conn = MagicMock()
    # mimic the original failure: Postgres cannot infer the IS NULL param type
    conn.execute = AsyncMock(
        side_effect=RuntimeError("could not determine data type of parameter $3")
    )
    conn.rollback = AsyncMock()
    return GraphAdapter(_service_with_conn(conn, "cg-1")), conn


async def test_get_all_entities_rolls_back_on_error() -> None:
    adapter, conn = _adapter_with_failing_conn()
    with pytest.raises(RuntimeError):
        await adapter._get_all_entities("cb-1", "t1")
    conn.rollback.assert_awaited_once()


async def test_get_all_relationships_rolls_back_on_error() -> None:
    adapter, conn = _adapter_with_failing_conn()
    with pytest.raises(RuntimeError):
        await adapter._get_all_relationships("cb-1", "t1")
    conn.rollback.assert_awaited_once()


def _adapter_with_rows(rows: list[tuple]) -> tuple[GraphAdapter, MagicMock]:
    class _Cursor:
        def __aiter__(self):
            async def _gen():
                for r in rows:
                    yield r

            return _gen()

    conn = MagicMock()
    conn.execute = AsyncMock(return_value=_Cursor())
    conn.rollback = AsyncMock()
    # The old singleton sentinel must be IGNORED now — rows are scoped by the
    # per-request codebase_id (== container name), not db.code_graph_id.
    return GraphAdapter(_service_with_conn(conn, "__shared__")), conn


async def test_get_all_entities_scopes_by_codebase_id_and_tenant() -> None:
    adapter, conn = _adapter_with_rows(
        [("n1", "function", "python", "app.py", 10, {"x": 1}, "doc")]
    )

    entities = await adapter._get_all_entities("cb-1", "tenant-a")

    assert len(entities) == 1
    assert entities[0]["id"] == "n1"
    sql, params = conn.execute.call_args.args
    # Scoped by BOTH code_graph_id and tenant (NULL-tolerant for legacy rows) so a
    # colliding container name in another tenant can't leak its graph.
    assert params == ("cb-1", "tenant-a")
    assert "code_graph_id = %s" in sql
    assert "(tenant_id = %s OR tenant_id IS NULL)" in sql
    # The dead `properties->>'codebase_id'` sub-filter must be gone.
    assert "codebase_id" not in sql


async def test_get_all_relationships_scopes_by_codebase_id_and_tenant() -> None:
    adapter, conn = _adapter_with_rows([("s1", "t1", "calls", {})])

    rels = await adapter._get_all_relationships("cb-1", "tenant-a")

    assert len(rels) == 1
    assert rels[0]["source"] == "s1"
    sql, params = conn.execute.call_args.args
    assert params == ("cb-1", "tenant-a")
    assert "e.code_graph_id = %s" in sql
    assert "(e.tenant_id = %s OR e.tenant_id IS NULL)" in sql
    assert "codebase_id" not in sql


async def test_get_all_entities_coalesces_from_properties() -> None:
    """The ingestion writers store name/file_path/line_number in the properties
    JSON and leave the dedicated columns NULL. The adapter must fall back to
    properties so the extractors see real values (else 0 components)."""
    adapter, _ = _adapter_with_rows(
        [
            (
                "n1",
                "class",
                None,  # language column NULL
                None,  # file_path column NULL
                None,  # line_number column NULL
                {
                    "name": "RoleResponse",
                    "file_path": "src/auth_api.py",
                    "line_number": 42,
                    "language": "python",
                },
                None,
            )
        ]
    )

    entities = await adapter._get_all_entities("cb-1", "tenant-a")

    assert len(entities) == 1
    e = entities[0]
    assert e["name"] == "RoleResponse"
    assert e["file_path"] == "src/auth_api.py"
    assert e["line_number"] == 42
    assert e["language"] == "python"


async def test_load_graph_defaults_missing_file_path_to_empty(monkeypatch) -> None:
    """A node with no file_path (e.g. the repo root) must load as "" so the
    extractors' Path(file_path) calls never hit Path(None)."""
    import networkx as nx

    adapter, _ = _adapter_with_rows([])
    monkeypatch.setattr(
        adapter,
        "_get_all_entities",
        AsyncMock(
            return_value=[{"id": "repo", "type": "repository", "language": None, "file_path": None}]
        ),
    )
    monkeypatch.setattr(adapter, "_get_all_relationships", AsyncMock(return_value=[]))
    adapter._service.entity_mapper = object()

    graph: nx.DiGraph = await adapter.load_graph("cb-1", "tenant-a")

    assert graph.nodes["repo"]["file_path"] == ""


async def test_get_all_entities_heals_a_closed_singleton_connection(monkeypatch) -> None:
    """Prod 2026-10-11: the singleton's conn was closed server-side and every
    architecture GET 500'd "the connection is closed" until restart."""
    import db.facade as facade

    dead = MagicMock()
    dead.closed = True
    dead.broken = False
    fresh = MagicMock()
    fresh.closed = False
    fresh.broken = False

    class _Cursor:
        def __aiter__(self):
            async def _gen():
                yield ("n1", "function", "python", "a.py", 1, {}, None)

            return _gen()

    fresh.execute = AsyncMock(return_value=_Cursor())
    fresh.rollback = AsyncMock()
    returned: list = []

    class _Pool:
        async def getconn(self):
            return fresh

        async def putconn(self, c):
            returned.append(c)

    async def _get_pool():
        return _Pool()

    monkeypatch.setattr(facade, "get_pool", _get_pool)
    db = GraphDatabaseConnection("__shared__")
    db.conn = dead
    service = MagicMock()
    service.db = db

    entities = await GraphAdapter(service)._get_all_entities("cb-1", "t1")

    assert [e["id"] for e in entities] == ["n1"]
    assert returned == [dead]
    assert db.conn is fresh
    dead.execute.assert_not_called()
