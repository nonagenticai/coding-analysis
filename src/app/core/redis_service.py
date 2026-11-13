"""Redis service for caching and pub/sub in coding-analysis."""

from __future__ import annotations

import logging
from functools import lru_cache

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisService:
    """Async Redis client wrapper.

    ``url`` defaults to ``Settings.REDIS_URL`` (sourced from the gitops
    Deployment env block). No hardcoded fallback.
    """

    def __init__(self, url: str | None = None):
        if not url:
            from config.settings import get_settings

            url = get_settings().REDIS_URL
        if not url:
            raise RuntimeError(
                "RedisService requires REDIS_URL; set it in the Deployment env block."
            )
        self.url = url
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis | None:
        """Return the underlying Redis client (may be None before connect)."""
        return self._client

    async def get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
        return self._client

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        client = await self.get_client()
        return bool(await client.set(key, value, ex=ex))

    async def get(self, key: str) -> str | None:
        client = await self.get_client()
        value = await client.get(key)
        return value.decode() if isinstance(value, bytes) else value

    async def delete(self, *keys: str) -> int:
        client = await self.get_client()
        return await client.delete(*keys)

    async def async_scan_keys(self, pattern: str) -> list[str]:
        client = await self.get_client()
        keys: list[str] = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        return keys

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


@lru_cache
def get_redis_service() -> RedisService:
    return RedisService()
