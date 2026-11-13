"""Unit tests for main.py -- FastAPI app, health, root, middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test GET /health."""

    async def test_health_redis_connected(self):
        mock_redis = AsyncMock(spec=aioredis.Redis)
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        with (
            patch("main.aioredis.from_url", return_value=mock_redis),
        ):
            # Simulate lifespan setting _redis
            import main as main_module
            from main import app

            original_redis = main_module._redis
            main_module._redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")

                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert data["service"] == "coding-analysis"
                assert data["version"] == "1.0.0"
                assert data["redis"] == "connected"
                assert "timestamp" in data
                assert "environment" in data
            finally:
                main_module._redis = original_redis

    async def test_health_redis_disconnected(self):
        with (
            patch("main.aioredis.from_url", return_value=AsyncMock()),
        ):
            import main as main_module
            from main import app

            original_redis = main_module._redis
            main_module._redis = None
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")

                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["redis"] == "disconnected"
            finally:
                main_module._redis = original_redis

    async def test_health_redis_ping_fails(self):
        mock_redis = AsyncMock(spec=aioredis.Redis)
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("redis down"))
        mock_redis.aclose = AsyncMock()

        with (
            patch("main.aioredis.from_url", return_value=mock_redis),
        ):
            import main as main_module
            from main import app

            original_redis = main_module._redis
            main_module._redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")

                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["redis"] == "disconnected"
            finally:
                main_module._redis = original_redis


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Test GET /."""

    async def test_root_returns_service_info(self):
        with (
            patch("main.aioredis.from_url", return_value=AsyncMock()),
        ):
            from main import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/")

            assert resp.status_code == 200
            data = resp.json()
            assert data["service"] == "coding-analysis"
            assert data["version"] == "1.0.0"
            assert data["health"] == "/health"

    async def test_root_docs_field_reflects_settings(self):
        with (
            patch("main.aioredis.from_url", return_value=AsyncMock()),
        ):
            from main import app, settings

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/")

            data = resp.json()
            if settings.docs_enabled:
                assert data["docs"] == "/docs"
            else:
                assert data["docs"] is None


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------


class TestAppConfiguration:
    """Test FastAPI app metadata and middleware."""

    def test_app_title(self):
        from main import app

        assert app.title == "coding-analysis"

    def test_app_version(self):
        from main import app

        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self):
        from main import app

        middleware_classes = [getattr(m.cls, "__name__", "") for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_app_has_health_route(self):
        from main import app

        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in routes

    def test_app_has_root_route(self):
        from main import app

        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in routes

    def test_app_has_api_routes(self):
        from main import app

        # FastAPI >= 0.162 nests included routers (app.routes holds one
        # _IncludedRouter), so read the served paths from the OpenAPI schema.
        api_routes = [p for p in app.openapi()["paths"] if p.startswith("/api/")]
        assert len(api_routes) > 0


# ---------------------------------------------------------------------------
# get_redis()
# ---------------------------------------------------------------------------


class TestGetRedis:
    """Test the get_redis() helper."""

    def test_get_redis_raises_when_none(self):
        import main as main_module
        from main import get_redis

        original = main_module._redis
        main_module._redis = None
        try:
            with pytest.raises(RuntimeError, match="Redis is not initialised"):
                get_redis()
        finally:
            main_module._redis = original

    def test_get_redis_returns_connection_when_set(self):
        import main as main_module
        from main import get_redis

        original = main_module._redis
        mock_redis = AsyncMock(spec=aioredis.Redis)
        main_module._redis = mock_redis
        try:
            result = get_redis()
            assert result is mock_redis
        finally:
            main_module._redis = original


# ---------------------------------------------------------------------------
# Lifespan tests (lines 45-81)
# ---------------------------------------------------------------------------


class TestLifespan:
    """Test the lifespan startup/shutdown cycle."""

    async def test_lifespan_success_full_cycle(self):
        """Lifespan connects Redis, yields, then cleans up (lines 45-81)."""
        import main as main_module

        mock_redis = AsyncMock(spec=aioredis.Redis)
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        with (
            patch("main.aioredis.from_url", return_value=mock_redis),
        ):
            original_redis = main_module._redis
            main_module._redis = None
            try:
                async with main_module.lifespan(main_module.app):
                    assert main_module._redis is mock_redis
                    mock_redis.ping.assert_awaited_once()
                mock_redis.aclose.assert_awaited_once()
            finally:
                main_module._redis = original_redis

    async def test_lifespan_redis_connection_failure(self):
        """Redis failure sets _redis to None (lines 71-73)."""
        import main as main_module

        with (
            patch(
                "main.aioredis.from_url",
                side_effect=Exception("Redis connection refused"),
            ),
        ):
            original_redis = main_module._redis
            main_module._redis = None
            try:
                async with main_module.lifespan(main_module.app):
                    assert main_module._redis is None
            finally:
                main_module._redis = original_redis

    async def test_lifespan_shutdown_with_no_redis(self):
        """Shutdown when _redis is None (line 78 not entered)."""
        import main as main_module

        with (
            patch(
                "main.aioredis.from_url",
                side_effect=Exception("connection refused"),
            ),
        ):
            original_redis = main_module._redis
            main_module._redis = None
            try:
                async with main_module.lifespan(main_module.app):
                    pass
                # No error during shutdown
            finally:
                main_module._redis = original_redis


class TestLifespanExtractionSelfCheck:
    async def test_startup_fails_when_the_extraction_worker_cannot_start(self):
        import main as main_module
        from processing.contract_analysis import isolation

        async def broken(limits=None):
            raise RuntimeError("contract extraction worker cannot start")

        with (
            patch("main.aioredis.from_url", side_effect=Exception("no redis")),
            patch.object(isolation, "self_check", broken),
        ):
            original_redis = main_module._redis
            try:
                with pytest.raises(RuntimeError, match="cannot start"):
                    async with main_module.lifespan(main_module.app):
                        pass
            finally:
                main_module._redis = original_redis
