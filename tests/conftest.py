"""Shared test fixtures for coding-analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is on sys.path for imports
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set test environment BEFORE importing anything from the app
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CODING_CONTAINERS_URL", "http://localhost:8691")
# Inter-service URL stubs for tests; production values come from the gitops
# Deployment env block. Settings.model_post_init is bypassed when
# ENVIRONMENT="test" but individual modules still read these at construction.
os.environ.setdefault("LICENSE_INTROSPECT_URL", "http://localhost:8544/license/introspect")
os.environ.setdefault("QUALITY_ANALYSIS_SERVICE_URL", "http://localhost:8086/api/quality")
os.environ.setdefault("PYTHON_TEST_SERVICE_URL", "http://localhost:8087")
os.environ.setdefault("JAVASCRIPT_TEST_SERVICE_URL", "http://localhost:8085")
os.environ.setdefault("JAVA_TEST_SERVICE_URL", "http://localhost:8088")
os.environ.setdefault("RUST_TEST_SERVICE_URL", "http://localhost:8089")
os.environ.setdefault("CSHARP_TEST_SERVICE_URL", "http://localhost:8090")
os.environ.setdefault("GO_TEST_SERVICE_URL", "http://localhost:8091")

# Pre-load modules that are commonly stubbed by test files.
# Only import modules whose stubs are harmful and break other tests.
# Do NOT import endpoint modules here — that causes different test pollution.
import processing.analyzers.deep  # noqa: E402, F401
import services  # noqa: E402, F401
import services.analysis_status  # noqa: E402, F401
import services.github_provisioning_service  # noqa: E402, F401

# ── Credentialled WebSocket handshakes (coding-client >= 1.15.0) ─────────────
# Through coding-client 1.14.0 a WS handshake with NO Authorization at all was
# served on whatever X-Tenant-Id it happened to carry, so a suite exercising the
# quality WS *behaviour* could name a tenant with a bare header. 1.15.0 closes
# that: an uncredentialed handshake is refused, because a header nobody verified
# proves nothing. See src/app/api/endpoints/quality.py:quality_websocket.
#
# Those behaviour suites now have to present a licence, exactly as fr's quality
# proxy does in production. `ws_licence_headers()` builds the handshake and the
# `ws_licence` fixture is what makes it introspect.
#
# The token ENCODES the tenant so each test keeps naming its own, and the licence
# comes back with that same tenant — the "agreeing header" shape, which needs no
# on-behalf allowlist. Nothing here widens the boundary: an Authorization header
# the helper did not build still introspects to `active: false` and is refused.

_WS_LICENCE_PREFIX = "licence-for::"
#: svc-coding-gateway — the subject this service really sees on WS upgrades.
WS_LICENCE_SUBJECT = "019ffa19-db80-700e-9840-8c17e23b35d5"


def ws_licence_headers(tenant_id: str, **extra: str) -> dict[str, str]:
    """Headers for a handshake `quality_websocket` will ACCEPT for *tenant_id*.

    Requires the `ws_licence` fixture to be active in the same test, and the
    WebSocket double's ``.headers`` to be a real ``starlette.datastructures.
    Headers`` — the SDK reads ``authorization`` in lower case, so a plain dict
    is narrower than the real object and reads as uncredentialed.
    """
    return {
        "Authorization": f"Bearer {_WS_LICENCE_PREFIX}{tenant_id}",
        "X-Tenant-Id": tenant_id,
        **extra,
    }


@pytest.fixture
def ws_licence(monkeypatch):
    """Make tokens minted by :func:`ws_licence_headers` introspect as live.

    Deliberately NOT autouse. A global "every Bearer is valid" stub would make
    the auth boundary itself untestable — tests/unit/test_ws_identity.py exists
    to prove uncredentialed and non-allowlisted handshakes are refused, and it
    would pass vacuously. Modules opt in with::

        pytestmark = pytest.mark.usefixtures("ws_licence")
    """
    from coding_client.auth.tenant import clear_license_caches

    def _post(url, json=None, **kwargs):  # noqa: A002 - httpx's own kwarg name
        token = (json or {}).get("token", "")
        resp = MagicMock()
        resp.status_code = 200
        if token.startswith(_WS_LICENCE_PREFIX):
            resp.json.return_value = {
                "active": True,
                "tenant_id": token[len(_WS_LICENCE_PREFIX) :],
                "subject_id": WS_LICENCE_SUBJECT,
            }
        else:
            resp.json.return_value = {"active": False, "reason": "token_invalid"}
        return resp

    client = AsyncMock()
    client.post = AsyncMock(side_effect=_post)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    # Without this, `_introspect_license` short-circuits to None before any HTTP
    # call and every handshake here would be uncredentialed — the stub would be
    # inert and the suites would fail for a reason unrelated to what they test.
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://test-licenses/license/introspect")
    # These suites drive a real identity resolution, so the production posture
    # has to hold: no bare-header trust, no trusted-network bypass, no default.
    monkeypatch.setenv("TRUST_TENANT_HEADER", "false")

    clear_license_caches()
    with patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=client):
        yield
    clear_license_caches()


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Provide a Settings instance with test defaults."""
    from config.settings import Settings

    return Settings(
        ENVIRONMENT="test",
        ENABLE_AUTH=False,
        ENABLE_DOCS=True,
        REDIS_URL="redis://localhost:6379/0",
        CODING_CONTAINERS_URL="http://test-containers:8691",
        LOG_LEVEL="DEBUG",
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Fake psycopg connection (sqlite->postgres cutover)
# ---------------------------------------------------------------------------
#
# Source code talks to PostgreSQL through the ``db.session.get_raw_connection()``
# async-context-manager pool. Two access styles exist in the codebase:
#
#   1. ``cur = await conn.execute(sql, params)`` then ``await cur.fetchone() /
#      .fetchall()`` / ``cur.rowcount`` / ``async for row in cur`` (tuple rows).
#   2. ``cur = conn.cursor(row_factory=dict_row)`` (sync) then
#      ``await cur.execute(sql, params)`` / ``await cur.fetchall()`` (dict rows).
#
# ``FakePgConn`` supports both. Tests route fetch results by SQL substring via
# ``routes={"COUNT(*) FROM nodes": [(3,)]}`` (whitespace-insensitive); anything
# unmatched yields ``default_rows`` with ``default_rowcount``. Every executed
# ``(sql, params)`` is recorded on ``conn.executed`` so a test can assert the
# real SQL shape (``%s`` params, ``code_graph_id`` present, ``Jsonb()`` wraps).
#
# Patch the *consuming module's* name, since modules do
# ``from db.session import get_raw_connection``:
#     with patch("<module>.get_raw_connection", return_value=conn): ...
# ``FakePgConn`` is itself the async CM (``__aenter__`` returns self), so it is
# reusable across multiple ``get_raw_connection()`` calls in one test.


def _norm_sql(sql: str) -> str:
    return " ".join(str(sql).split())


class FakePgCursor:
    """Fake psycopg async cursor (tuple- or dict-row)."""

    def __init__(self, rows=(), rowcount=1, conn=None):
        self._rows = list(rows)
        self.rowcount = rowcount
        self._conn = conn

    async def execute(self, sql, params=None):
        # cursor-style execute (used by the conn.cursor(row_factory=dict_row) path)
        if self._conn is not None:
            self._conn.executed.append((sql, params))
            self._rows, self.rowcount = self._conn._lookup(sql)
        return self

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)

    async def fetchmany(self, size=1):
        out, self._rows = self._rows[:size], self._rows[size:]
        return out

    def __aiter__(self):
        rows = list(self._rows)

        async def _gen():
            for r in rows:
                yield r

        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def close(self):
        return None


class FakePgConn:
    """Fake psycopg async connection used to unit-test the postgres call sites."""

    def __init__(self, routes=None, default_rows=(), default_rowcount=1):
        self._routes = routes or {}
        self._default_rows = list(default_rows)
        self._default_rowcount = default_rowcount
        self.executed: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0

    def _lookup(self, sql):
        norm = _norm_sql(sql)
        for needle, val in self._routes.items():
            if _norm_sql(needle) in norm:
                if isinstance(val, dict):
                    return (
                        list(val.get("rows", self._default_rows)),
                        val.get("rowcount", self._default_rowcount),
                    )
                return list(val), self._default_rowcount
        return list(self._default_rows), self._default_rowcount

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        rows, rowcount = self._lookup(sql)
        return FakePgCursor(rows, rowcount, conn=self)

    async def executemany(self, sql, params_seq):
        params_seq = list(params_seq)
        self.executed.append((sql, params_seq))
        rows, rowcount = self._lookup(sql)
        return FakePgCursor(rows, rowcount, conn=self)

    def cursor(self, *args, **kwargs):
        return FakePgCursor([], self._default_rowcount, conn=self)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def make_pg_conn():
    """Factory for a :class:`FakePgConn`. See module docstring for routing."""

    def _make(routes=None, default_rows=(), default_rowcount=1):
        return FakePgConn(
            routes=routes, default_rows=default_rows, default_rowcount=default_rowcount
        )

    return _make


# ---------------------------------------------------------------------------
# HTTP / container client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_container_client():
    """Mock CodingContainersClient with all async methods."""
    client = AsyncMock()
    client.execute = AsyncMock(
        return_value={"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.1}
    )
    client.get_files = AsyncMock(return_value=[])
    client.read_file = AsyncMock(return_value="file content")
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_container_manager():
    """Mock ContainerManager."""
    manager = AsyncMock()
    manager.get_container_files = AsyncMock(return_value=[])
    manager.read_container_file = AsyncMock(return_value="")
    return manager


# ---------------------------------------------------------------------------
# File service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_file_service():
    """Mock FileService."""
    svc = MagicMock()
    svc.read_file = MagicMock(return_value="")
    svc.file_exists = MagicMock(return_value=True)
    svc.list_files = MagicMock(return_value=[])
    return svc


# ---------------------------------------------------------------------------
# Sample Python code for AST tests
# ---------------------------------------------------------------------------

SAMPLE_PYTHON_CODE = '''\
"""Sample module docstring."""

import os
import sys
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable
from . import sibling_module
from ..parent import helper

class BaseClass:
    """A base class."""
    pass

class ChildClass(BaseClass):
    """A child class with composition."""

    def __init__(self):
        self.helper = Helper()
        self.name: str = "test"

    def do_work(self, x: int) -> str:
        """Method that calls other things."""
        result = os.path.join("/tmp", str(x))
        self.helper.process(result)
        return result

    async def async_method(self):
        await self.helper.async_process()

class AnotherClass:
    """Uses ChildClass."""

    def run(self):
        child = ChildClass()
        child.do_work(42)

def standalone_function(a, b, c=None):
    """A standalone function."""
    return a + b

async def async_standalone():
    pass
'''


@pytest.fixture
def sample_python_code():
    """Multi-class Python source for AST tests."""
    return SAMPLE_PYTHON_CODE


# ---------------------------------------------------------------------------
# FastAPI test app fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app():
    """Create a FastAPI TestClient with auth disabled.

    Auth is off because ENABLE_AUTH=false (set at module scope above) means
    main.py never registers AuthMiddleware. This fixture used to also set
    SKIP_AUTH=true; measured 2026-08-15, removing that changed nothing — the
    whole suite (12406 passed) is green without it, so it was never the
    mechanism, only a second belt beside the braces. SKIP_AUTH is gone from
    coding-client as of 1.16.0.
    """

    # Patch lifespan to avoid real Redis connections
    import redis.asyncio as aioredis

    with patch("main.aioredis.from_url") as mock_redis_factory:
        mock_redis = AsyncMock(spec=aioredis.Redis)
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_redis_factory.return_value = mock_redis

        # Need httpx for async test client
        from httpx import ASGITransport, AsyncClient

        from main import app

        transport = ASGITransport(app=app)

        async def _make_client():
            return AsyncClient(transport=transport, base_url="http://test")

        yield app, _make_client
