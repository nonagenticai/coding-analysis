"""Cache API models stub.

TODO: Implement cache-related API models.
"""

from __future__ import annotations

from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""

    total_keys: int = 0
    memory_used: str = "0B"
    hit_rate: float = 0.0


class CacheClearResponse(BaseModel):
    """Cache clear response."""

    cleared: bool = False
    keys_removed: int = 0


class CacheKeysResponse(BaseModel):
    """Cache keys response."""

    keys: list[str] = []
    count: int = 0
