"""Unit tests for the analysis-owned src/db package (no live database).

These cover the schema metadata, DSN resolution, the facade batching/lifecycle
logic, and the entity-mapper SQL/parameter shapes using fakes. Real round-trip
behaviour (alembic apply, ON CONFLICT, composite-FK enforcement, cross-graph
isolation) is covered by tests/integration/test_db_package_integration.py.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db import entity_mapper, facade, session
from db.models import Base

# ---------------------------------------------------------------------------
# models / schema metadata
# ---------------------------------------------------------------------------


def test_schema_tables_present():
    assert set(Base.metadata.tables) == {
        "nodes",
        "edges",
        "analysis_sessions",
        "graph_versions",
        "bug_hunts",
        "bug_hunt_findings",
        "quality_results",
    }


def test_quality_results_schema_matches_migration_0007():
    table = Base.metadata.tables["quality_results"]
    assert [column.name for column in table.primary_key.columns] == ["id"]
    assert not table.c.tenant_id.nullable
    assert not table.c.container_name.nullable
    assert table.c.commit_sha.nullable  # unknown commit is NULL, never a branch name
    assert table.c.overall_score.nullable  # failed / inconclusive runs have no score
    # No unique constraint on the sha: re-analysing a commit is a new row.
    assert not [c for c in table.constraints if c.__class__.__name__ == "UniqueConstraint"]
    assert not any(index.unique for index in table.indexes)
    names = {index.name for index in table.indexes}
    assert names == {
        "idx_quality_results_tenant_container_created",
        "idx_quality_results_tenant_repo_sha",
    }


def test_bug_hunt_report_schema_is_normalized_and_path_indexed():
    hunts = Base.metadata.tables["bug_hunts"]
    findings = Base.metadata.tables["bug_hunt_findings"]

    assert [column.name for column in hunts.primary_key.columns] == ["id"]
    assert {
        "code_graph_id",
        "tenant_id",
        "status",
        "request_payload",
        "worker_id",
        "lease_expires_at",
    } <= set(hunts.columns.keys())
    assert [column.name for column in findings.primary_key.columns] == ["id"]
    assert {"hunt_id", "file_path", "file_name", "finding", "suggested_fixes"} <= set(
        findings.columns.keys()
    )
    path_index = next(
        index for index in findings.indexes if index.name == "idx_bug_finding_hunt_path"
    )
    assert list(path_index.columns.keys()) == ["hunt_id", "tenant_id", "file_path"]
    assert path_index.dialect_options["postgresql"]["ops"] == {"file_path": "text_pattern_ops"}


def test_nodes_tenant_scoped_identity():
    nodes = Base.metadata.tables["nodes"]
    # Mapper-level PK stays (code_graph_id, id) — logical only; the real DB
    # identity is the tenant-scoped unique index (migration 0002).
    assert [c.name for c in nodes.primary_key.columns] == ["code_graph_id", "id"]
    assert len(nodes.columns) == 21
    for col in ("language", "is_static", "signature", "docstring", "content_hash"):
        assert col in nodes.columns
    uq = next(i for i in nodes.indexes if i.name == "uq_node_identity")
    assert uq.unique
    assert list(uq.columns.keys()) == ["code_graph_id", "tenant_id", "id"]
    assert uq.dialect_options["postgresql"]["nulls_not_distinct"] is True


def test_edges_tenant_scoped_identity_no_fks():
    edges = Base.metadata.tables["edges"]
    assert [c.name for c in edges.primary_key.columns] == ["id"]
    # Old composite UNIQUE / FKs were dropped in migration 0002.
    uniques = [c for c in edges.constraints if c.__class__.__name__ == "UniqueConstraint"]
    assert uniques == []
    fks = [c for c in edges.constraints if c.__class__.__name__ == "ForeignKeyConstraint"]
    assert fks == []
    # Identity is now a tenant-scoped NULLS NOT DISTINCT unique index.
    uq = next(i for i in edges.indexes if i.name == "uq_edge_identity")
    assert uq.unique
    assert list(uq.columns.keys()) == [
        "code_graph_id",
        "tenant_id",
        "source",
        "target",
        "relationship",
    ]
    assert uq.dialect_options["postgresql"]["nulls_not_distinct"] is True


# ---------------------------------------------------------------------------
# session DSN helpers
# ---------------------------------------------------------------------------


def test_normalize_dsn_strips_sqlalchemy_driver():
    assert (
        session._normalize_dsn("postgresql+psycopg://u:p@h:5432/d") == "postgresql://u:p@h:5432/d"
    )
    assert session._normalize_dsn("postgresql://u:p@h:5432/d") == "postgresql://u:p@h:5432/d"


def test_resolve_dsn_prefers_pgbouncer(monkeypatch):
    monkeypatch.setattr(
        session,
        "get_settings",
        lambda: SimpleNamespace(
            pgbouncer_url="postgresql+psycopg://pgb/d", database_url="postgresql://direct/d"
        ),
    )
    assert session._resolve_dsn() == "postgresql://pgb/d"


def test_resolve_dsn_falls_back_to_database_url(monkeypatch):
    monkeypatch.setattr(
        session,
        "get_settings",
        lambda: SimpleNamespace(pgbouncer_url="", database_url="postgresql://direct/d"),
    )
    assert session._resolve_dsn() == "postgresql://direct/d"


def test_resolve_dsn_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr(
        session,
        "get_settings",
        lambda: SimpleNamespace(pgbouncer_url="", database_url=""),
    )
    with pytest.raises(RuntimeError, match="No database URL configured"):
        session._resolve_dsn()


# ---------------------------------------------------------------------------
# facade batching + lifecycle (fakes)
# ---------------------------------------------------------------------------


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple] = []

    def transaction(self):
        return _FakeTxn()

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))


class _FakePool:
    def __init__(self):
        self.conn = _FakeConn()
        self.released = None

    async def getconn(self):
        return self.conn

    async def putconn(self, conn):
        self.released = conn


def test_add_node_batches_with_content_hash():
    db = facade.GraphDatabaseConnection("graph-a", tenant_id="t1")
    db.add_node("n1", "function", {"name": "f"})
    assert len(db._pending_operations) == 1
    kind, node_id, node_type, props, content_hash, tenant = db._pending_operations[0]
    assert (kind, node_id, node_type, tenant) == ("node", "n1", "function", "t1")
    assert len(content_hash) == 64  # sha256 hexdigest


def test_add_edge_defaults_properties_and_tenant():
    db = facade.GraphDatabaseConnection("graph-a")
    db.add_edge("n1", "n2", "calls", tenant_id="explicit")
    kind, src, tgt, rel, props, tenant = db._pending_operations[0]
    assert (kind, src, tgt, rel, props, tenant) == ("edge", "n1", "n2", "calls", {}, "explicit")


async def test_flush_writes_in_order_and_clears(monkeypatch):
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = _FakeConn()
    db.add_node("n1", "function", {"name": "f"})
    db.add_edge("n1", "n2", "calls")
    await db.flush_pending_operations()
    assert len(db.conn.calls) == 2
    assert "INSERT INTO nodes" in db.conn.calls[0][0]
    assert "INSERT INTO edges" in db.conn.calls[1][0]
    assert db.conn.calls[0][1][0] == "graph-a"  # code_graph_id scoped
    assert db._pending_operations == []


async def test_flush_noop_when_empty():
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = _FakeConn()
    await db.flush_pending_operations()
    assert db.conn.calls == []


async def test_connect_get_close_lifecycle(monkeypatch):
    pool = _FakePool()

    async def _get_pool():
        return pool

    monkeypatch.setattr(facade, "get_pool", _get_pool)
    db = facade.GraphDatabaseConnection("graph-a")
    with pytest.raises(RuntimeError):
        db.get_connection()
    await db.connect()
    assert db.get_connection() is pool.conn
    await db.close()
    assert db.conn is None
    assert pool.released is pool.conn


class _FakeConnInfo:
    def __init__(self, status):
        self.transaction_status = status


class _FakeConnWithTxn:
    def __init__(self, status):
        self.info = _FakeConnInfo(status)
        self.rolled_back = False

    async def rollback(self):
        self.rolled_back = True


async def test_close_rolls_back_lingering_transaction(monkeypatch):
    """A connection returned INTRANS is rolled back before going to the pool."""
    from psycopg.pq import TransactionStatus

    pool = _FakePool()
    monkeypatch.setattr(facade, "get_pool", AsyncMock(return_value=pool))

    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = _FakeConnWithTxn(TransactionStatus.INTRANS)
    conn = db.conn

    await db.close()

    assert conn.rolled_back is True
    assert pool.released is conn
    assert db.conn is None


async def test_close_does_not_roll_back_idle_connection(monkeypatch):
    """An IDLE connection (writer committed) is returned without a rollback."""
    from psycopg.pq import TransactionStatus

    pool = _FakePool()
    monkeypatch.setattr(facade, "get_pool", AsyncMock(return_value=pool))

    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = _FakeConnWithTxn(TransactionStatus.IDLE)
    conn = db.conn

    await db.close()

    assert conn.rolled_back is False
    assert pool.released is conn


# ---------------------------------------------------------------------------
# entity mapper SQL/parameter shapes (fakes)
# ---------------------------------------------------------------------------


def _fake_entity():
    return SimpleNamespace(
        id="ent-1",
        type=SimpleNamespace(value="function"),
        language="python",
        file_path="src/x.py",
        line_number=10,
        column_start=0,
        column_end=4,
        visibility="public",
        is_static=True,
        is_abstract=False,
        is_final=False,
        is_async=True,
        signature="def f()",
        docstring="doc",
        properties={"content_hash": "abc", "k": "v"},
        metadata={"m": 1},
        return_type=None,
        parameter_types=[],
        created_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
    )


async def test_store_entity_params_and_flags():
    conn = _FakeConn()
    mapper = entity_mapper.MultiLanguageEntityMapper(conn, "graph-a", tenant_id="t1")
    await mapper.store_entity(_fake_entity())
    sql, params = conn.calls[0]
    assert "INSERT INTO nodes" in sql and "ON CONFLICT (code_graph_id, tenant_id, id)" in sql
    assert params[0] == "graph-a"  # code_graph_id
    assert params[1] == "ent-1"
    assert params[2] == "function"  # type.value
    # bool flags -> 1/0 ints (is_static=True, is_abstract=False, is_async=True)
    assert params[9] == 1 and params[10] == 0 and params[12] == 1
    assert params[16] == "abc"  # content_hash from properties
    assert params[-1] == "t1"  # tenant_id


async def test_store_relationship_with_and_without_location():
    conn = _FakeConn()
    mapper = entity_mapper.MultiLanguageEntityMapper(conn, "graph-a")
    loc = SimpleNamespace(file_path="src/x.py", line_start=3, column_start=2)
    rel = SimpleNamespace(
        source_id="a",
        target_id="b",
        type=SimpleNamespace(value="calls"),
        weight=2.0,
        context="ctx",
        location=loc,
        properties={"p": 1},
        created_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
    )
    await mapper.store_relationship(rel)
    sql, params = conn.calls[0]
    assert "INSERT INTO edges" in sql
    assert params[0:4] == ("graph-a", "a", "b", "calls")
    assert params[6:9] == ("src/x.py", 3, 2)  # location_file/line/column

    rel2 = SimpleNamespace(
        source_id="a",
        target_id="b",
        type=SimpleNamespace(value="calls"),
        weight=1.0,
        context=None,
        location=None,
        properties=None,
        created_at=None,
    )
    await mapper.store_relationship(rel2)
    _, params2 = conn.calls[1]
    assert params2[6:9] == (None, None, None)
    assert params2[9] is None  # properties None -> no Jsonb wrapper


# --- stale-connection self-heal (prod 2026-10-11: "the connection is closed") ---


class _StaleConn:
    def __init__(self, name, *, closed=False, broken=False):
        self.name = name
        self.closed = closed
        self.broken = broken
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        self.closed = True


class _SeqPool:
    def __init__(self, *conns):
        self._conns = list(conns)
        self.returned: list = []

    async def getconn(self):
        return self._conns.pop(0)

    async def putconn(self, conn):
        self.returned.append(conn)


def _patch_pool(monkeypatch, pool):
    async def _get_pool():
        return pool

    monkeypatch.setattr(facade, "get_pool", _get_pool)


async def test_ensure_connection_replaces_closed_conn(monkeypatch):
    dead, fresh = _StaleConn("dead", closed=True), _StaleConn("fresh")
    pool = _SeqPool(fresh)
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = dead

    assert await db.ensure_connection() is fresh
    assert pool.returned == [dead]  # dead one handed back for the pool to discard
    assert db.conn is fresh


async def test_ensure_connection_replaces_broken_conn(monkeypatch):
    broken, fresh = _StaleConn("broken", broken=True), _StaleConn("fresh")
    pool = _SeqPool(fresh)
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = broken

    assert await db.ensure_connection() is fresh
    assert pool.returned == [broken]


async def test_ensure_connection_keeps_live_conn(monkeypatch):
    """Control: a healthy connection is NOT churned."""
    live = _StaleConn("live")
    pool = _SeqPool()
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = live

    assert await db.ensure_connection() is live
    assert pool.returned == []


async def test_run_read_retries_once_on_operational_error(monkeypatch):
    import psycopg

    first, second = _StaleConn("first"), _StaleConn("second")
    pool = _SeqPool(second)
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = first
    seen: list = []

    async def _fn(conn):
        seen.append(conn)
        if conn is first:
            raise psycopg.OperationalError("the connection is closed")
        return "ok"

    assert await db.run_read(_fn) == "ok"
    assert seen == [first, second]
    assert first.close_calls == 1  # suspect conn closed before it goes back
    assert pool.returned == [first]
    assert db.conn is second


async def test_run_read_second_failure_propagates(monkeypatch):
    import psycopg

    first, second = _StaleConn("first"), _StaleConn("second")
    pool = _SeqPool(second)
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = first
    calls = 0

    async def _fn(conn):
        nonlocal calls
        calls += 1
        raise psycopg.InterfaceError("still closed")

    with pytest.raises(psycopg.InterfaceError):
        await db.run_read(_fn)
    assert calls == 2  # exactly one retry


async def test_run_read_does_not_retry_non_connection_errors(monkeypatch):
    """Control: a SQL/logic error is not a stale connection — no retry."""
    first = _StaleConn("first")
    pool = _SeqPool()
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = first
    calls = 0

    async def _fn(conn):
        nonlocal calls
        calls += 1
        raise ValueError("bad query")

    with pytest.raises(ValueError):
        await db.run_read(_fn)
    assert calls == 1
    assert db.conn is first


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: __import__("psycopg").errors.QueryCanceled("canceling statement due to timeout"),
        lambda: __import__("psycopg").errors.LockNotAvailable("lock timeout"),
    ],
)
async def test_run_read_does_not_retry_query_level_operational_errors(monkeypatch, exc_factory):
    """QueryCanceled (statement_timeout) etc. are not stale connections: no retry."""
    first = _StaleConn("first")
    pool = _SeqPool()
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = first
    calls = 0
    exc = exc_factory()

    async def _fn(conn):
        nonlocal calls
        calls += 1
        raise exc

    with pytest.raises(type(exc)):
        await db.run_read(_fn)
    assert calls == 1
    assert db.conn is first
    assert pool.returned == []


async def test_run_read_retries_connection_class_sqlstate(monkeypatch):
    """Control: an admin-shutdown (57P01) OperationalError is retried once."""
    import psycopg

    first, second = _StaleConn("first"), _StaleConn("second")
    pool = _SeqPool(second)
    _patch_pool(monkeypatch, pool)
    db = facade.GraphDatabaseConnection("graph-a")
    db.conn = first

    async def _fn(conn):
        if conn is first:
            raise psycopg.errors.AdminShutdown("terminating connection")
        return "ok"

    assert await db.run_read(_fn) == "ok"
