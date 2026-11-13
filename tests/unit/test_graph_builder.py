"""Unit tests for the reusable code-graph build entry point.

Covers idempotency (skip when already populated), the build delegation to the
deep-analysis path, and force_rebuild clearing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.endpoints.analysis_modules import analysis_endpoints
from services.code_graph import graph_builder


class _FakeConnCM:
    """Minimal async context manager yielding a fake psycopg connection."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_):
        return False


def _conn_returning_count(count: int) -> MagicMock:
    """Build a fake connection whose single SELECT returns ``count``."""
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=(count,))
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    return conn


@pytest.mark.asyncio
async def test_graph_is_populated_true_when_parsed_nodes_exist(monkeypatch):
    conn = _conn_returning_count(7)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    assert await graph_builder.graph_is_populated("c1", "tenant-a") is True


@pytest.mark.asyncio
async def test_graph_is_populated_false_when_only_repo_node(monkeypatch):
    conn = _conn_returning_count(0)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    assert await graph_builder.graph_is_populated("c1", "tenant-a") is False


@pytest.mark.asyncio
async def test_graph_is_populated_scopes_by_tenant(monkeypatch):
    """The populated-check is scoped to the tenant (NULL-tolerant) so a different
    tenant's graph under a colliding container name does not wrongly skip a build."""
    conn = _conn_returning_count(0)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    await graph_builder.graph_is_populated("c1", "tenant-a")

    sql, params = conn.execute.call_args.args
    assert params == ("c1", "tenant-a")
    assert "code_graph_id = %s" in sql
    assert "(tenant_id = %s OR tenant_id IS NULL)" in sql


@pytest.mark.asyncio
async def test_clear_parsed_nodes_scopes_deletes_strictly_to_tenant(monkeypatch):
    """A destructive DELETE must only ever touch the caller's own attributed rows
    — strict ``tenant_id = %s`` (NOT NULL-tolerant), never another tenant's data."""
    conn = _conn_returning_count(0)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    await graph_builder._clear_parsed_nodes("c1", "tenant-a")

    # Both DELETEs (edges + parsed nodes) bind the tenant and use strict equality.
    for call in conn.execute.call_args_list:
        sql, params = call.args
        assert "DELETE FROM" in sql
        assert "tenant_id = %s" in sql
        assert "tenant_id IS NULL" not in sql
        assert params == ("c1", "tenant-a")
    conn.commit.assert_awaited()


@pytest.mark.asyncio
async def test_build_skips_when_already_populated(monkeypatch):
    conn = _conn_returning_count(5)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    result = await graph_builder.build_code_graph("c1", "tenant-1")

    assert result == {"nodes": 0, "edges": 0, "skipped": True}


@pytest.mark.asyncio
async def test_build_delegates_to_analyze_and_returns_counts(monkeypatch):
    # Empty graph -> not populated -> build proceeds.
    conn = _conn_returning_count(0)
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    fake_response = SimpleNamespace(
        result=SimpleNamespace(metadata={"graph_nodes": 12, "graph_edges": 9, "files_analyzed": 4})
    )
    mock_analyze = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(analysis_endpoints, "analyze_code", mock_analyze)

    result = await graph_builder.build_code_graph("my-repo", "tenant-1")

    assert result == {"nodes": 12, "edges": 9, "skipped": False}
    mock_analyze.assert_awaited_once()
    # Tenant is threaded through to the deep-analysis call.
    _, kwargs = mock_analyze.await_args
    assert kwargs.get("tenant_id") == "tenant-1"


@pytest.mark.asyncio
async def test_force_rebuild_clears_then_builds(monkeypatch):
    conn = _conn_returning_count(5)  # already populated, but force_rebuild ignores that
    monkeypatch.setattr(graph_builder, "get_raw_connection", lambda: _FakeConnCM(conn))

    fake_response = SimpleNamespace(
        result=SimpleNamespace(metadata={"graph_nodes": 3, "graph_edges": 1})
    )
    monkeypatch.setattr(analysis_endpoints, "analyze_code", AsyncMock(return_value=fake_response))

    result = await graph_builder.build_code_graph("my-repo", "tenant-1", force_rebuild=True)

    assert result == {"nodes": 3, "edges": 1, "skipped": False}
    # The clear path issued DELETE statements + commit on the fake connection.
    assert conn.execute.await_count >= 2
    conn.commit.assert_awaited()
