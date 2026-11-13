"""PostgreSQL-backed drop-in for ``coding_client.db.GraphDatabaseConnection``.

The legacy class was keyed by a per-graph SQLite file path. This facade is
keyed by a ``code_graph_id`` discriminator and is backed by the shared psycopg
pool, so many graphs live in one PostgreSQL database. It exposes only the
surface the analysis call sites actually use:

- ``connect()`` / ``get_connection()`` / ``close()`` and the ``conn`` attribute
  (a raw psycopg ``AsyncConnection`` the call sites run SQL against), and
- the batch write API ``add_node`` / ``add_edge`` / ``flush_pending_operations``
  (only ``services.code_graph.structure_storage`` uses it).

Every write is scoped to ``(code_graph_id, tenant_id)``; the tenant-scoped unique
indexes (migration 0002) keep two tenants sharing a ``code_graph_id`` from
colliding while a forgotten scope still fails loud rather than leaking.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import psycopg
from psycopg import AsyncConnection
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from .session import get_pool

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Errors that mean "this server connection is gone" (Postgres / PgBouncer closed
# it, idle-in-transaction timeout, network reset). A read retried on a fresh
# connection is safe; a write is not, so only ``run_read`` retries.
_STALE_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
)

# SQLSTATEs that mean the SERVER connection went away (class 08 = connection
# exception; 57P01-03 = admin/crash shutdown, cannot connect now). Any other
# server-reported SQLSTATE -- notably 57014 QueryCanceled (statement_timeout) --
# is a property of the query, and re-running it would just repeat the failure.
_CONNECTION_SQLSTATES = frozenset({"57P01", "57P02", "57P03"})


def _is_connection_failure(exc: BaseException, conn: AsyncConnection) -> bool:
    """True when *exc* is a stale/broken-connection failure worth one retry."""
    if isinstance(exc, psycopg.errors.QueryCanceled):
        return False
    if isinstance(exc, psycopg.InterfaceError):
        return True
    if getattr(conn, "closed", False) or getattr(conn, "broken", False):
        return True
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate is None:
        # Client-side OperationalError with no server SQLSTATE: the connection
        # failed before/while talking to the server.
        return True
    return sqlstate.startswith("08") or sqlstate in _CONNECTION_SQLSTATES


_ADD_NODE_SQL = """
    INSERT INTO nodes (code_graph_id, id, type, properties, content_hash, tenant_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (code_graph_id, tenant_id, id) DO UPDATE SET
        type = EXCLUDED.type,
        properties = EXCLUDED.properties,
        content_hash = EXCLUDED.content_hash,
        updated_at = now()
"""

_ADD_EDGE_SQL = """
    INSERT INTO edges (code_graph_id, source, target, relationship, properties, tenant_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO UPDATE SET
        properties = EXCLUDED.properties
"""


class GraphDatabaseConnection:
    """Manages a pooled psycopg connection scoped to one code graph."""

    def __init__(self, code_graph_id: str, tenant_id: str | None = None) -> None:
        self.code_graph_id = code_graph_id
        self.tenant_id = tenant_id
        self.conn: AsyncConnection | None = None
        self._pending_operations: list[tuple] = []

    async def connect(self) -> None:
        """Check out a connection from the shared pool (idempotent)."""
        if self.conn is None:
            pool = await get_pool()
            self.conn = await pool.getconn()
            logger.debug("Acquired connection for code_graph %s", self.code_graph_id)

    def get_connection(self) -> AsyncConnection:
        """Return the active psycopg connection."""
        if self.conn is None:
            raise RuntimeError("Database connection not established. Call connect() first.")
        return self.conn

    @staticmethod
    def _is_dead(conn: AsyncConnection) -> bool:
        return bool(getattr(conn, "closed", False) or getattr(conn, "broken", False))

    async def _discard_connection(self) -> None:
        """Hand the current (dead or suspect) connection back to the pool.

        ``pool.putconn`` discards a closed/broken connection instead of reusing it.
        A connection that is still open but just failed is closed first so the
        pool cannot hand the same suspect server connection to the next caller.
        """
        conn, self.conn = self.conn, None
        if conn is None:
            return
        with contextlib.suppress(Exception):
            if not self._is_dead(conn):
                await conn.close()
        with contextlib.suppress(Exception):
            pool = await get_pool()
            await pool.putconn(conn)
        logger.info("Discarded stale connection for code_graph %s", self.code_graph_id)

    async def ensure_connection(self) -> AsyncConnection:
        """Return a LIVE connection, replacing a closed/broken one.

        The architecture endpoints reach this facade through a process-wide
        singleton that holds one connection for the life of the pod. When
        Postgres/PgBouncer closes that server connection, ``get_connection()``
        keeps returning the dead object and every request 500s with "the
        connection is closed" until restart. Read paths call this instead.
        """
        if self.conn is not None and self._is_dead(self.conn):
            await self._discard_connection()
        if self.conn is None:
            await self.connect()
        assert self.conn is not None
        return self.conn

    async def run_read(self, fn: Callable[[AsyncConnection], Awaitable[_T]]) -> _T:
        """Run a READ-ONLY ``fn(conn)``, retrying once on a fresh connection.

        A connection can look alive (``closed`` is False) yet fail on first use
        because the server side was already terminated. On a CONNECTION-class
        failure (``InterfaceError``, or an ``OperationalError`` that left the
        connection closed/broken, carries no server SQLSTATE, or a class-08 /
        57P0x one) the connection is discarded and ``fn`` is retried exactly
        once; a second failure propagates. Query-level errors such as
        ``QueryCanceled`` (statement timeout) are never retried. Never use this
        for writes — a retried write could apply twice.
        """
        conn = await self.ensure_connection()
        try:
            return await fn(conn)
        except _STALE_CONNECTION_ERRORS as exc:
            if not _is_connection_failure(exc, conn):
                raise
            logger.warning(
                "Read on code_graph %s failed on a stale connection (%s); retrying once",
                self.code_graph_id,
                exc,
            )
            await self._discard_connection()
            conn = await self.ensure_connection()
            return await fn(conn)

    async def close(self) -> None:
        """Return the connection to the pool.

        Read-only analysis paths run SELECTs that implicitly open a transaction
        (autocommit is off) without an explicit commit/rollback. Returning such a
        connection leaves it ``INTRANS``, which the pool then has to roll back —
        logging ``rolling back returned connection [INTRANS]`` and, under
        PgBouncer transaction pooling, risking dead pooled connections. Roll back
        any lingering transaction here so the connection returns clean. The write
        path commits via ``conn.transaction()`` (status ``IDLE`` afterwards), so
        this is a no-op for it.
        """
        if self.conn is not None:
            pool = await get_pool()
            try:
                if self.conn.info.transaction_status != TransactionStatus.IDLE:
                    await self.conn.rollback()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug(
                    "Pre-return rollback failed for code_graph %s: %s",
                    self.code_graph_id,
                    exc,
                )
            await pool.putconn(self.conn)
            self.conn = None
            logger.debug("Released connection for code_graph %s", self.code_graph_id)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict,
        tenant_id: str | None = None,
    ) -> None:
        """Queue a node insert for the next ``flush_pending_operations``."""
        content_hash = hashlib.sha256(json.dumps(properties).encode()).hexdigest()
        self._pending_operations.append(
            ("node", node_id, node_type, properties, content_hash, tenant_id or self.tenant_id)
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        properties: dict | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Queue an edge insert for the next ``flush_pending_operations``."""
        self._pending_operations.append(
            (
                "edge",
                source_id,
                target_id,
                relationship,
                properties or {},
                tenant_id or self.tenant_id,
            )
        )

    async def flush_pending_operations(self) -> None:
        """Write all queued node/edge operations in one transaction.

        Insertion order is preserved so nodes land before the edges that
        reference them. Any failure rolls back the whole batch (matching the
        legacy transactional flush).
        """
        if not self._pending_operations:
            return
        if self.conn is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        operations = self._pending_operations
        self._pending_operations = []
        async with self.conn.transaction():
            for op in operations:
                if op[0] == "node":
                    _, node_id, node_type, properties, content_hash, tenant_id = op
                    await self.conn.execute(
                        _ADD_NODE_SQL,
                        (
                            self.code_graph_id,
                            node_id,
                            node_type,
                            Jsonb(properties),
                            content_hash,
                            tenant_id,
                        ),
                    )
                elif op[0] == "edge":
                    _, source_id, target_id, relationship, properties, tenant_id = op
                    await self.conn.execute(
                        _ADD_EDGE_SQL,
                        (
                            self.code_graph_id,
                            source_id,
                            target_id,
                            relationship,
                            Jsonb(properties),
                            tenant_id,
                        ),
                    )
        logger.info(
            "Flushed %d pending operations for code_graph %s",
            len(operations),
            self.code_graph_id,
        )


# Legacy alias retained by some call sites / tests.
CodeGraphDatabase = GraphDatabaseConnection
