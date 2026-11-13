"""Unit tests for AnalysisCache service."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Purge sys.modules pollution from earlier test files.
#
# test_lang_processors_misc.py (and others) stub "services.analysis_cache"
# as a MagicMock during module-level collection.  Because this test file's
# fixture does a deferred import, the MagicMock is already in sys.modules
# by the time the fixture runs.  Force-clean it so we get the REAL module.
# ---------------------------------------------------------------------------
for _key in list(sys.modules):
    if _key == "services.analysis_cache" and isinstance(sys.modules[_key], MagicMock):
        del sys.modules[_key]

# Ensure "services" is a proper package pointing at the real source directory
# (earlier tests may have replaced it with a bare ModuleType or MagicMock).
_src_services = str(Path(__file__).resolve().parents[2] / "src" / "services")
if "services" in sys.modules:
    _svc_mod = sys.modules["services"]
    if not getattr(_svc_mod, "__path__", None):
        _svc_mod.__path__ = [_src_services]
    elif _src_services not in _svc_mod.__path__:
        _svc_mod.__path__.insert(0, _src_services)
else:
    _svc_mod = _types.ModuleType("services")  # noqa: F841
    _svc_mod.__path__ = [_src_services]
    _svc_mod.__package__ = "services"
    sys.modules["services"] = _svc_mod


@pytest.fixture
def cache(tmp_path):
    """Create an AnalysisCache with a temp directory."""
    # Re-check at fixture time in case another test polluted after collection
    for _key in list(sys.modules):
        if _key == "services.analysis_cache" and isinstance(sys.modules[_key], MagicMock):
            del sys.modules[_key]

    from services.analysis_cache import AnalysisCache, AnalysisCacheConfig

    config = AnalysisCacheConfig(
        cache_dir=tmp_path / "cache",
        max_cache_size_mb=10,
        max_entries=100,
        ttl_hours=24,
        cleanup_interval_minutes=1,
    )
    return AnalysisCache(config=config)


def _make_mock_parse_result(
    success: bool = True,
    language: str = "python",
    parser_type: str = "tree_sitter",
):
    """Create a mock ParseResult object."""
    mock = MagicMock()
    mock.success = success
    mock.language = language
    mock.entities = []
    mock.relationships = []
    mock.lines_of_code = 10
    mock.complexity_score = 1.5
    mock.parse_time_ms = 12.3
    mock.file_path = "/fake/test.py"
    mock.ast = None
    mock.errors = []
    mock.warnings = []
    # parser_used should have a .value attribute
    mock.parser_used = MagicMock()
    mock.parser_used.value = parser_type
    return mock


@pytest.mark.asyncio
async def test_initialize_creates_tables(cache, tmp_path):
    """initialize() creates the SQLite database file and tables."""
    await cache.initialize()

    assert cache._initialized is True
    assert cache.db_path.exists()

    # Cleanup background task
    await cache.shutdown()


@pytest.mark.asyncio
async def test_initialize_idempotent(cache):
    """Calling initialize() twice is safe."""
    await cache.initialize()
    await cache.initialize()  # second call should no-op

    assert cache._initialized is True
    await cache.shutdown()


@pytest.mark.asyncio
async def test_get_cached_result_miss(cache):
    """get_cached_result returns None when file is not cached."""
    await cache.initialize()

    result = await cache.get_cached_result(Path("/not/cached/file.py"), "some content")

    assert result is None
    assert cache.stats.metrics.cache_misses == 1

    await cache.shutdown()


@pytest.mark.asyncio
async def test_get_cached_result_hit(cache):
    """cache_result then get_cached_result returns a result (cache hit)."""
    await cache.initialize()

    file_path = Path("/test/file.py")
    content = "def hello(): pass\n"
    parse_result = _make_mock_parse_result()

    # Store first
    await cache.cache_result(file_path, content, parse_result)

    # Patch _is_cache_valid to always return True (avoids file system checks)
    with patch.object(cache, "_is_cache_valid", new=AsyncMock(return_value=True)):
        # Also patch _cache_entry_to_parse_result to avoid ParseResult import chain
        mock_pr = _make_mock_parse_result()
        with patch.object(
            cache, "_cache_entry_to_parse_result", new=AsyncMock(return_value=mock_pr)
        ):
            result = await cache.get_cached_result(file_path, content)

    assert result is not None
    assert cache.stats.metrics.cache_hits >= 1

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_result_stores_entry(cache):
    """cache_result stores the entry in cache_index."""
    await cache.initialize()

    file_path = Path("/test/module.py")
    content = "x = 1\n"
    parse_result = _make_mock_parse_result()

    await cache.cache_result(file_path, content, parse_result)

    assert str(file_path) in cache.cache_index

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_result_unsuccessful_not_stored(cache):
    """cache_result does not store entries where success=False."""
    await cache.initialize()

    file_path = Path("/test/bad.py")
    content = "invalid python ##"
    parse_result = _make_mock_parse_result(success=False)

    await cache.cache_result(file_path, content, parse_result)

    assert str(file_path) not in cache.cache_index

    await cache.shutdown()


@pytest.mark.asyncio
async def test_get_stats_empty(cache):
    """get_stats returns correct structure when cache is empty."""
    await cache.initialize()

    stats = await cache.get_stats()

    assert "total_entries" in stats
    assert "total_hits" in stats
    assert "total_misses" in stats
    assert "hit_rate" in stats
    assert "miss_rate" in stats
    assert "cache_size_mb" in stats
    assert "languages" in stats
    assert stats["total_entries"] == 0

    await cache.shutdown()


@pytest.mark.asyncio
async def test_get_stats_with_entries(cache):
    """get_stats reflects actual cached entries."""
    await cache.initialize()

    file_path = Path("/test/stats.py")
    content = "y = 2\n"
    parse_result = _make_mock_parse_result(language="python")

    await cache.cache_result(file_path, content, parse_result)

    stats = await cache.get_stats()

    assert stats["total_entries"] == 1
    assert "python" in stats["languages"]

    await cache.shutdown()


@pytest.mark.asyncio
async def test_clear_cache_all(cache):
    """clear_cache() removes all entries."""
    await cache.initialize()

    for i in range(3):
        fp = Path(f"/test/file{i}.py")
        pr = _make_mock_parse_result()
        await cache.cache_result(fp, f"code{i}", pr)

    assert len(cache.cache_index) == 3

    cleared = await cache.clear_cache()

    assert cleared == 3
    assert len(cache.cache_index) == 0

    await cache.shutdown()


@pytest.mark.asyncio
async def test_clear_cache_by_language(cache):
    """clear_cache(language='python') removes only python entries."""
    await cache.initialize()

    py_pr = _make_mock_parse_result(language="python")
    js_pr = _make_mock_parse_result(language="javascript")

    await cache.cache_result(Path("/test/a.py"), "py code", py_pr)
    await cache.cache_result(Path("/test/b.js"), "js code", js_pr)

    cleared = await cache.clear_cache(language="python")

    assert cleared == 1
    assert str(Path("/test/a.py")) not in cache.cache_index
    assert str(Path("/test/b.js")) in cache.cache_index

    await cache.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_background_task(cache):
    """shutdown() cancels the background cleanup task."""
    await cache.initialize()

    assert cache._cleanup_task is not None
    assert not cache._cleanup_task.done()

    await cache.shutdown()

    assert cache._cleanup_task.done()


@pytest.mark.asyncio
async def test_get_cache_info_not_cached(cache):
    """get_cache_info returns None for uncached file."""
    await cache.initialize()

    info = await cache.get_cache_info(Path("/not/there.py"))

    assert info is None

    await cache.shutdown()


@pytest.mark.asyncio
async def test_get_cache_info_cached(cache):
    """get_cache_info returns dict with correct fields after caching."""
    await cache.initialize()

    file_path = Path("/test/info.py")
    content = "def info(): pass\n"
    parse_result = _make_mock_parse_result()

    await cache.cache_result(file_path, content, parse_result)

    info = await cache.get_cache_info(file_path)

    assert info is not None
    assert info["cached"] is True
    assert info["language"] == "python"
    assert "cached_at" in info
    assert "access_count" in info
    assert "entities_count" in info

    await cache.shutdown()


def test_calculate_content_hash_consistent(cache):
    """_calculate_content_hash produces same hash for same content."""
    content = "def foo(): pass\n"

    h1 = cache._calculate_content_hash(content)
    h2 = cache._calculate_content_hash(content)

    assert h1 == h2
    assert len(h1) == 16  # truncated SHA-256


def test_calculate_content_hash_different_for_different_content(cache):
    """_calculate_content_hash produces different hashes for different content."""
    h1 = cache._calculate_content_hash("content A")
    h2 = cache._calculate_content_hash("content B")

    assert h1 != h2


@pytest.mark.asyncio
async def test_get_cache_stats_returns_stats(cache):
    """get_cache_stats returns CacheStats object."""
    from services.analysis_cache import CacheStats

    await cache.initialize()

    stats = await cache.get_cache_stats()

    assert isinstance(stats, CacheStats)

    await cache.shutdown()
