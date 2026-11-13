"""Cache manager service.

Redis-backed caching with in-memory fallback.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


def cached(
    func: Callable[..., Any] | str | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Caching decorator stub. Currently passes through without caching.

    Supports both:
        @cached
        @cached("cache_key", ttl_seconds=300)
    """
    # Called as @cached("key", ...) or @cached(key="key", ...)
    if func is None or isinstance(func, str):
        return lambda f: _wrap(f)

    # Called as @cached (no arguments)
    return _wrap(func)


def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function with a pass-through (no-op cache)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kw: Any) -> Any:
        return func(*args, **kw)

    return wrapper


class RedisCacheManager:
    """Redis cache manager with in-memory fallback.

    Attempts to use Redis for all cache operations. If Redis is unavailable
    or any operation fails, transparently falls back to the in-memory dict.
    """

    def __init__(self, redis_url: str | None = None, **kwargs: Any) -> None:
        """Initialize Redis cache manager.

        ``redis_url`` is required; if not passed it falls back to
        ``Settings.REDIS_URL`` (gitops-provided). No hardcoded fallback —
        missing config fails loud at startup in production.
        """
        if not redis_url:
            from config.settings import get_settings

            redis_url = get_settings().REDIS_URL
        if not redis_url:
            raise RuntimeError(
                "RedisCacheManager requires REDIS_URL; set it in the Deployment env block."
            )
        self.redis_url = redis_url
        self._cache: dict[str, Any] = {}
        self._redis: aioredis.Redis | None = None
        try:
            self._redis = aioredis.Redis.from_url(redis_url, decode_responses=True)
            logger.debug("RedisCacheManager connected to %s", redis_url)
        except Exception:
            logger.warning("RedisCacheManager: failed to create Redis client, using in-memory only")
            self._redis = None

    async def get(self, key: str) -> Any:
        """Get a value from cache.

        Tries Redis first; falls back to the in-memory dict on any error.
        """
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
                return None
            except Exception:
                logger.debug("Redis get failed for key=%s, falling back to memory", key)
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache with an optional TTL (seconds).

        Writes to Redis when available and always mirrors into the in-memory dict.
        """
        serialized = json.dumps(value)
        self._cache[key] = value
        if self._redis is not None:
            try:
                await self._redis.set(key, serialized, ex=ttl)
            except Exception:
                logger.debug("Redis set failed for key=%s, stored in memory only", key)

    async def delete(self, key: str) -> None:
        """Delete a value from cache."""
        self._cache.pop(key, None)
        if self._redis is not None:
            try:
                await self._redis.delete(key)
            except Exception:
                logger.debug("Redis delete failed for key=%s", key)

    async def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        if self._redis is not None:
            try:
                await self._redis.flushdb()
            except Exception:
                logger.debug("Redis flushdb failed, in-memory cache cleared")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                logger.debug("Redis close failed")
