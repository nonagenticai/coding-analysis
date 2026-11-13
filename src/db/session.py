"""Single async psycopg connection pool for the analysis code-graph store.

The analysis service talks to PostgreSQL through raw psycopg (the call sites
use ``conn.execute(...)`` directly), so the runtime owns exactly one
``AsyncConnectionPool`` rather than a SQLAlchemy engine — this avoids
double-pooling and keeps ``.conn`` a real psycopg connection that the existing
call sites expect.

PgBouncer transaction-mode safety:
- ``prepare_threshold = None`` disables server-side prepared statements, which
  break under transaction pooling.
- The runtime DSN prefers ``pgbouncer_url``; Alembic migrations connect
  directly to ``database_url`` (out of band, see ``alembic/env.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.core.config.core_config_settings import get_settings

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


def _normalize_dsn(url: str) -> str:
    """Return a libpq DSN psycopg accepts (strip any SQLAlchemy ``+driver``)."""
    if url.startswith("postgresql+"):
        # e.g. postgresql+psycopg://... -> postgresql://...
        scheme, rest = url.split("://", 1)
        return "postgresql://" + rest
    return url


def _resolve_dsn() -> str:
    """Resolve the runtime DSN, preferring PgBouncer over the direct URL."""
    settings = get_settings()
    url = settings.pgbouncer_url or settings.database_url
    if not url:
        raise RuntimeError(
            "No database URL configured: set PGBOUNCER_URL or DATABASE_URL "
            "(coding-analysis owns the 'coding_analysis' PostgreSQL database)."
        )
    return _normalize_dsn(url)


async def _configure_connection(conn: AsyncConnection) -> None:
    """Disable prepared statements so the pool is PgBouncer-txn-mode safe."""
    conn.prepare_threshold = None


async def get_pool() -> AsyncConnectionPool:
    """Return the process-wide pool, opening it lazily on first use."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            pool = AsyncConnectionPool(
                conninfo=_resolve_dsn(),
                open=False,
                configure=_configure_connection,
                # VALIDATE A CONNECTION BEFORE HANDING IT OUT. Without this the pool will serve a
                # socket whose far end is gone, and it will do so FOREVER -- it has no other reason
                # to discard it.
                #
                # MEASURED 2026-09-01: the `pgbouncer` Deployment rolled at 09:22:50Z (new
                # ConfigMap -> new ReplicaSet). This pod was 2d20h old with 0 restarts, so every
                # pooled connection still pointed at pgbouncer pods that no longer existed. The
                # first architecture call after the roll got
                #     consuming input failed: server closed the connection unexpectedly
                # and every call after that got `the connection is closed`, failing in ~35 ms
                # BEFORE any query ran. Architecture endpoints served 6x 500 / 0x 200 for the next
                # three hours and did NOT self-heal; only a pod restart cleared it. There is no
                # backend death in the postgres log, because nothing died -- the proxy moved.
                #
                # `check_connection` issues a cheap liveness check and, on failure, the pool
                # discards that connection and returns another, so a proxy roll costs one probe
                # instead of an outage.
                check=AsyncConnectionPool.check_connection,
                min_size=1,
                max_size=10,
            )
            await pool.open(wait=True)
            _pool = pool
    return _pool


@asynccontextmanager
async def get_raw_connection(timeout: float | None = None) -> AsyncIterator[AsyncConnection]:
    """Yield a raw psycopg connection from the pool for a single operation.

    ``timeout`` caps the wait for a free pooled connection (seconds); ``None``
    keeps the pool default.
    """
    pool = await get_pool()
    async with pool.connection(timeout=timeout) as conn:
        yield conn


async def close_pool() -> None:
    """Close the process-wide pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
