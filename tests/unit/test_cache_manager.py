"""Unit tests for services.cache_manager."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from services.cache_manager import RedisCacheManager, cached

# ---------------------------------------------------------------------------
# cached() decorator
# ---------------------------------------------------------------------------


class TestCachedDecorator:
    def test_passthrough_no_args(self):
        @cached
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_passthrough_with_key_arg(self):
        @cached("my_key")
        def greet(name):
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_passthrough_with_none(self):
        @cached(None)
        def identity(x):
            return x

        assert identity(42) == 42

    def test_preserves_function_name(self):
        @cached
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_preserves_function_name_with_key(self):
        @cached("key")
        def another_func():
            pass

        assert another_func.__name__ == "another_func"

    def test_passthrough_returns_result(self):
        calls = []

        @cached
        def side_effect():
            calls.append(1)
            return "done"

        result = side_effect()
        assert result == "done"
        assert len(calls) == 1

    async def test_passthrough_async_function(self):
        @cached
        async def async_add(a, b):
            return a + b

        result = await async_add(3, 4)
        assert result == 7


# ---------------------------------------------------------------------------
# RedisCacheManager — happy path (Redis available)
# ---------------------------------------------------------------------------


def _make_manager_with_mock_redis():
    """Return (manager, mock_redis) with Redis client replaced by an AsyncMock."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.flushdb = AsyncMock()
    mock_redis.aclose = AsyncMock()

    with patch(
        "services.cache_manager.aioredis.Redis.from_url",
        return_value=mock_redis,
    ):
        manager = RedisCacheManager("redis://localhost:6379")

    return manager, mock_redis


class TestRedisCacheManagerHappyPath:
    async def test_get_returns_none_when_redis_miss(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.get.return_value = None
        result = await manager.get("missing_key")
        assert result is None

    async def test_get_deserializes_json(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.get.return_value = json.dumps({"hello": "world"})
        result = await manager.get("k")
        assert result == {"hello": "world"}

    async def test_set_serializes_json_and_calls_redis(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        await manager.set("k", {"x": 1}, ttl=60)
        mock_redis.set.assert_awaited_once_with("k", json.dumps({"x": 1}), ex=60)

    async def test_set_without_ttl_passes_none(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        await manager.set("k", "value")
        mock_redis.set.assert_awaited_once_with("k", json.dumps("value"), ex=None)

    async def test_set_also_stores_in_memory(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        await manager.set("k", 42)
        assert manager._cache["k"] == 42

    async def test_delete_calls_redis_and_removes_from_memory(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        manager._cache["k"] = "v"
        await manager.delete("k")
        mock_redis.delete.assert_awaited_once_with("k")
        assert "k" not in manager._cache

    async def test_delete_missing_key_is_noop(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        await manager.delete("nonexistent")  # should not raise
        mock_redis.delete.assert_awaited_once()

    async def test_clear_calls_flushdb_and_clears_memory(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        manager._cache["a"] = 1
        manager._cache["b"] = 2
        await manager.clear()
        mock_redis.flushdb.assert_awaited_once()
        assert manager._cache == {}

    async def test_close_calls_aclose(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        await manager.close()
        mock_redis.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# RedisCacheManager — JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    async def test_list_roundtrip(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        value = [1, "two", 3.0]
        mock_redis.get.return_value = json.dumps(value)
        result = await manager.get("k")
        assert result == value

    async def test_nested_dict_roundtrip(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        value = {"a": {"b": [1, 2, 3]}, "c": True}
        mock_redis.get.return_value = json.dumps(value)
        result = await manager.get("k")
        assert result == value

    async def test_integer_roundtrip(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.get.return_value = json.dumps(99)
        result = await manager.get("k")
        assert result == 99


# ---------------------------------------------------------------------------
# RedisCacheManager — Redis failure falls back to in-memory
# ---------------------------------------------------------------------------


class TestRedisFallback:
    def test_init_falls_back_when_redis_raises(self):
        with patch(
            "services.cache_manager.aioredis.Redis.from_url",
            side_effect=Exception("connection refused"),
        ):
            manager = RedisCacheManager("redis://bad-host:9999")
        assert manager._redis is None

    async def test_get_falls_back_to_memory_on_redis_error(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.get.side_effect = Exception("timeout")
        manager._cache["k"] = "fallback_value"
        result = await manager.get("k")
        assert result == "fallback_value"

    async def test_set_stores_in_memory_on_redis_error(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.set.side_effect = Exception("timeout")
        await manager.set("k", "value")
        assert manager._cache["k"] == "value"

    async def test_delete_removes_from_memory_on_redis_error(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.delete.side_effect = Exception("timeout")
        manager._cache["k"] = "v"
        await manager.delete("k")
        assert "k" not in manager._cache

    async def test_clear_clears_memory_on_redis_error(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.flushdb.side_effect = Exception("timeout")
        manager._cache["a"] = 1
        await manager.clear()
        assert manager._cache == {}

    async def test_close_is_noop_when_redis_none(self):
        manager, _ = _make_manager_with_mock_redis()
        manager._redis = None
        await manager.close()  # should not raise

    async def test_get_returns_none_when_redis_none_and_key_missing(self):
        manager, _ = _make_manager_with_mock_redis()
        manager._redis = None
        result = await manager.get("nonexistent")
        assert result is None

    async def test_get_returns_memory_value_when_redis_none(self):
        manager, _ = _make_manager_with_mock_redis()
        manager._redis = None
        manager._cache["k"] = {"stored": True}
        result = await manager.get("k")
        assert result == {"stored": True}

    async def test_close_does_not_raise_on_aclose_error(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.aclose.side_effect = Exception("close error")
        await manager.close()  # should not raise
