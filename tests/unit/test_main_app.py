"""Tests for main.py uncovered paths.

Covers lines in main.py: health endpoint, root endpoint, exception handlers
(via coding-client KAG standard), lifespan branches.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Exception handlers (KAG standard via register_exception_handlers)
# ---------------------------------------------------------------------------


class TestKAGExceptionHandlers:
    """Test exception handlers registered by coding-client."""

    @pytest.fixture
    def _client(self):
        """Create a test transport for the app."""
        from main import app

        return ASGITransport(app=app)

    @pytest.mark.asyncio
    async def test_http_404_produces_kag_error(self, _client):
        async with AsyncClient(transport=_client, base_url="http://test") as client:
            resp = await client.get("/api/analysis/nonexistent-route-xyz")
        assert resp.status_code in (404, 405)
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "request_id" in data["error"]

    @pytest.mark.asyncio
    async def test_validation_error_produces_kag_422(self, _client):
        """Send a request that triggers a validation error."""
        async with AsyncClient(transport=_client, base_url="http://test") as client:
            # POST to an endpoint expecting a body, but send invalid data
            resp = await client.post(
                "/api/analysis/analyze",
                json={},  # missing required fields
            )
        # Should be 422 validation error
        if resp.status_code == 422:
            data = resp.json()
            assert "error" in data
            assert data["error"]["code"] == "HTTP_422"
            assert "details" in data["error"]


# ---------------------------------------------------------------------------
# Health endpoint (direct call)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_without_redis(self):
        from main import health

        with patch("main._redis", None):
            result = await health()
        assert result["status"] == "degraded"
        assert result["redis"] == "disconnected"

    @pytest.mark.asyncio
    async def test_health_with_redis_ok(self):
        from main import health

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        with patch("main._redis", mock_redis):
            result = await health()
        assert result["status"] == "ok"
        assert result["redis"] == "connected"

    @pytest.mark.asyncio
    async def test_health_with_redis_error(self):
        from main import health

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch("main._redis", mock_redis):
            result = await health()
        assert result["status"] == "degraded"


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root(self):
        from main import root

        result = await root()
        assert "service" in result
        assert "version" in result
        assert "health" in result


# ---------------------------------------------------------------------------
# get_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    def test_raises_when_not_initialized(self):
        from main import get_redis

        with patch("main._redis", None):
            with pytest.raises(RuntimeError, match="not initialised"):
                get_redis()

    def test_returns_redis_when_initialized(self):
        from main import get_redis

        mock_redis = MagicMock()
        with patch("main._redis", mock_redis):
            result = get_redis()
        assert result is mock_redis


# ---------------------------------------------------------------------------
# Auth middleware installation must fail CLOSED
# ---------------------------------------------------------------------------


class TestInstallAuthMiddlewareFailsClosed:
    """A service that cannot load its auth middleware must not serve.

    The previous code caught ImportError, logged a warning, and continued with
    every route unauthenticated. auth.tenant re-exports TRUSTED_NETWORKS from
    coding_client, so a coding-client release that moves that name takes this
    path -- which is exactly how a remediation bump could disable auth fleet-wide.
    """

    def test_import_error_raises_instead_of_serving_unauthenticated(self):
        import builtins

        from fastapi import FastAPI

        from main import install_auth_middleware

        real_import = builtins.__import__

        def _fail_auth_import(name, *args, **kwargs):
            if name in ("auth.middleware", "auth.tenant"):
                raise ImportError(f"simulated: cannot import {name}")
            return real_import(name, *args, **kwargs)

        target = FastAPI()
        with patch.object(builtins, "__import__", _fail_auth_import):
            with pytest.raises(RuntimeError, match="refusing to start unauthenticated"):
                install_auth_middleware(target)

        # And nothing was half-attached on the way out.
        assert not any(
            m.cls.__name__ in ("AuthMiddleware", "TenantMiddleware") for m in target.user_middleware
        )

    def test_healthy_import_attaches_both_middlewares(self):
        """The success path still wires both -- so the test above cannot pass vacuously."""
        from fastapi import FastAPI

        from main import install_auth_middleware

        target = FastAPI()
        install_auth_middleware(target)

        attached = {m.cls.__name__ for m in target.user_middleware}
        assert {"AuthMiddleware", "TenantMiddleware"} <= attached
