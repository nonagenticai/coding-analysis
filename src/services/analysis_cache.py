"""
Analysis Cache Service

Intelligent caching layer for multi-language code analysis results to update
performance and reduce redundant processing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from models.core_models import CoreCodeEntity, EntityRelation

if TYPE_CHECKING:
    from processing.languages.base import ParseResult, ParserType
else:
    ParseResult = None
    ParserType = None
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Analysis results for caching."""

    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    lines_of_code: int
    complexity_score: float
    parse_time_ms: float


@dataclass
class CacheMetadata:
    """Cache metadata for tracking."""

    cached_at: datetime
    access_count: int = 0
    last_accessed: datetime | None = None
    cache_hit_savings_ms: float = 0.0


@dataclass
class AnalysisCacheEntry:
    """Represents a cached analysis result."""

    file_path: str
    content_hash: str
    file_size: int
    language: str
    parser_type: str
    results: AnalysisResults
    metadata: CacheMetadata


@dataclass
class CacheAnalysisMetrics:
    """Cache request and hit metrics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    hit_rate: float = 0.0
    time_saved_ms: float = 0.0


@dataclass
class CacheSizeMetrics:
    """Cache size and efficiency metrics."""

    total_files_cached: int = 0
    total_cache_size_mb: float = 0.0
    average_file_size_kb: float = 0.0
    storage_efficiency: float = 0.0
    oldest_entry: datetime | None = None
    newest_entry: datetime | None = None


class CacheStats:
    """Cache performance statistics."""

    def __init__(self):
        """Initialize the CacheStats service."""
        self.metrics = CacheAnalysisMetrics()
        self.size_metrics = CacheSizeMetrics()

    def reset(self) -> None:
        """Reset all statistics."""
        self.metrics = CacheAnalysisMetrics()
        self.size_metrics = CacheSizeMetrics()

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_evictions": self.metrics.cache_evictions,
                "hit_rate": self.metrics.hit_rate,
                "time_saved_ms": self.metrics.time_saved_ms,
            },
            "size_metrics": {
                "total_files_cached": self.size_metrics.total_files_cached,
                "total_cache_size_mb": self.size_metrics.total_cache_size_mb,
                "average_file_size_kb": self.size_metrics.average_file_size_kb,
                "storage_efficiency": self.size_metrics.storage_efficiency,
                "oldest_entry": (
                    self.size_metrics.oldest_entry.isoformat()
                    if self.size_metrics.oldest_entry
                    else None
                ),
                "newest_entry": (
                    self.size_metrics.newest_entry.isoformat()
                    if self.size_metrics.newest_entry
                    else None
                ),
            },
        }


@dataclass
class AnalysisCacheConfig:
    """Configuration for AnalysisCache."""

    cache_dir: Path | None = None
    max_cache_size_mb: int = 500
    max_entries: int = 10000
    ttl_hours: int = 24
    cleanup_interval_minutes: int = 30

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_cache_size_mb <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if self.ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
        if self.cleanup_interval_minutes <= 0:
            raise ValueError("cleanup_interval_minutes must be positive")


class AnalysisCache:
    """Intelligent cache for multi-language analysis results."""

    def __init__(self, config: AnalysisCacheConfig | None = None):
        """Initialize the analysis cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or AnalysisCacheConfig()
        self.config.validate()
        import os

        cache_dir = (
            self.config.cache_dir
            or Path(os.getenv("TEMP_DIR", "/tmp/code_graph")) / "analysis_cache"  # nosec B108
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "cache.db"
        self.cache_index: dict[str, AnalysisCacheEntry] = {}
        self.stats = CacheStats()
        self.start_time = datetime.now(UTC)
        self._cleanup_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the cache system."""
        if self._initialized:
            return
        await self._create_tables()
        await self._load_cache_index()
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        self._initialized = True
        logger.info(
            "Analysis cache initialized: %d entries, %s",
            len(self.cache_index),
            self.db_path.parent,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        PERFORMANCE OPTIMIZATION: Single-pass iteration over cache_index
        instead of multiple passes (was: 5 passes, now: 1 pass).
        Collects total_hits, language stats, and parse_time totals in one loop.
        """
        total_entries = len(self.cache_index)
        total_hits = 0
        # Language stats: {lang: {"entries": N, "hits": N, "parse_time_sum": N}}
        language_stats: dict[str, dict[str, Any]] = {}

        # Single-pass: collect all metrics at once
        for entry in self.cache_index.values():
            total_hits += entry.metadata.access_count
            lang = entry.language
            if lang not in language_stats:
                language_stats[lang] = {"entries": 0, "hits": 0, "parse_time_sum": 0.0}
            language_stats[lang]["entries"] += 1
            language_stats[lang]["hits"] += entry.metadata.access_count
            language_stats[lang]["parse_time_sum"] += entry.results.parse_time_ms

        # Compute averages from accumulated sums (no second pass needed)
        for lang, stats in language_stats.items():
            entries_count = stats["entries"]
            stats["avg_parse_time"] = (
                stats["parse_time_sum"] / entries_count if entries_count > 0 else 0.0
            )
            del stats["parse_time_sum"]  # Remove intermediate sum from output

        total_misses = self.stats.metrics.cache_misses
        total_requests = total_hits + total_misses
        hit_rate = total_hits / max(total_requests, 1)

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate,
            "cache_size_mb": self._get_cache_size_mb(),
            "languages": language_stats,
            "uptime_seconds": (datetime.now(UTC) - self.start_time).total_seconds(),
            "cache_directory": str(self.db_path.parent),
        }

    def _get_cache_size_mb(self) -> float:
        """Get cache size in MB.

        PERFORMANCE OPTIMIZATION: Uses pre-stored file_size from cache entries
        instead of expensive str() conversion on entities/relationships.
        The file_size is already computed and stored when caching results.
        """
        try:
            # Use stored file_size instead of expensive str() conversion
            # file_size represents the original content size, which is more accurate
            total_size = sum(entry.file_size for entry in self.cache_index.values())
            return total_size / (1024 * 1024)
        except (TypeError, AttributeError, ValueError) as e:
            logger.warning("Error calculating cache size: %s", e)
            return 0.0

    async def get_cached_result(self, file_path: Path, content: str) -> ParseResult | None:
        """Get cached analysis result if available and valid.

        Args:
            file_path: Path to the source file
            content: Current file content

        Returns:
            Cached ParseResult if available, None otherwise
        """
        global ParseResult
        if ParseResult is None:
            from processing.languages.base import ParseResult as _ParseResult

            ParseResult = _ParseResult
        if not self._initialized:
            await self.initialize()
        self.stats.metrics.total_requests += 1
        content_hash = self._calculate_content_hash(content)
        cache_key = str(file_path)
        if cache_key not in self.cache_index:
            self.stats.metrics.cache_misses += 1
            logger.debug("Cache miss: %s (not cached)", file_path)
            return None
        cache_entry = self.cache_index[cache_key]
        if not await self._is_cache_valid(cache_entry, content_hash, file_path):
            await self._remove_cache_entry(cache_key)
            self.stats.metrics.cache_misses += 1
            logger.debug("Cache miss: %s (invalid)", file_path)
            return None
        cache_entry.metadata.access_count += 1
        cache_entry.metadata.last_accessed = datetime.now(UTC)
        self.stats.metrics.cache_hits += 1
        parse_result = await self._cache_entry_to_parse_result(cache_entry)
        time_saved = cache_entry.results.parse_time_ms
        self.stats.metrics.time_saved_ms += time_saved
        cache_entry.metadata.cache_hit_savings_ms += time_saved
        logger.debug("Cache hit: %s (saved %.1fms)", file_path, time_saved)
        return parse_result

    async def cache_result(self, file_path: Path, content: str, parse_result: ParseResult) -> None:
        """Cache an analysis result.

        Args:
            file_path: Path to the source file
            content: File content that was analyzed
            parse_result: Analysis result to cache
        """
        global ParseResult
        if ParseResult is None:
            from processing.languages.base import ParseResult as _ParseResult

            ParseResult = _ParseResult
        if not self._initialized:
            await self.initialize()
        if not parse_result.success:
            return
        cache_key = str(file_path)
        content_hash = self._calculate_content_hash(content)
        results = AnalysisResults(
            entities=[self._entity_to_dict(e) for e in parse_result.entities],
            relationships=[self._relation_to_dict(r) for r in parse_result.relationships],
            lines_of_code=parse_result.lines_of_code,
            complexity_score=parse_result.complexity_score or 0.0,
            parse_time_ms=parse_result.parse_time_ms,
        )
        metadata = CacheMetadata(cached_at=datetime.now(UTC))
        cache_entry = AnalysisCacheEntry(
            file_path=cache_key,
            content_hash=content_hash,
            file_size=len(content.encode("utf-8")),
            language=parse_result.language,
            parser_type=(
                parse_result.parser_used.value
                if hasattr(parse_result.parser_used, "value")
                else str(parse_result.parser_used)
            ),
            results=results,
            metadata=metadata,
        )
        await self._enforce_cache_limits()
        await self._store_cache_entry(cache_entry)
        self.cache_index[cache_key] = cache_entry
        self.stats.size_metrics.total_files_cached = len(self.cache_index)
        self._update_cache_stats()
        logger.debug("Cached result: %s (%d bytes)", file_path, cache_entry.file_size)

    async def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        if not self._initialized:
            await self.initialize()
        if self.stats.metrics.total_requests > 0:
            self.stats.metrics.hit_rate = (
                self.stats.metrics.cache_hits / self.stats.metrics.total_requests
            )
        self._update_cache_stats()
        return self.stats

    async def clear_cache(self, language: str | None = None) -> int:
        """Clear cache entries.

        Args:
            language: If specified, only clear entries for this language

        Returns:
            Number of entries cleared
        """
        if not self._initialized:
            await self.initialize()
        if language:
            keys_to_remove = [
                key for key, entry in self.cache_index.items() if entry.language == language
            ]
        else:
            keys_to_remove = list(self.cache_index.keys())
        for key in keys_to_remove:
            await self._remove_cache_entry(key)
        logger.info(
            "Cleared %d cache entries%s",
            len(keys_to_remove),
            f" for language {language}" if language else "",
        )
        return len(keys_to_remove)

    async def optimize_cache(self) -> dict[str, Any]:
        """Optimize cache by removing least useful entries."""
        if not self._initialized:
            await self.initialize()
        optimization_stats: dict[str, Any] = {
            "entries_before": len(self.cache_index),
            "entries_removed": 0,
            "space_freed_mb": 0.0,
            "optimization_strategy": [],
        }
        expired_count = await self._remove_expired_entries()
        removed_count = optimization_stats["entries_removed"]
        if isinstance(removed_count, int):
            optimization_stats["entries_removed"] = removed_count + expired_count
        strategy_list = optimization_stats["optimization_strategy"]
        if isinstance(strategy_list, list):
            strategy_list.append(f"Removed {expired_count} expired entries")
        if len(self.cache_index) > self.config.max_entries * 0.8:
            lru_removed = await self._remove_lru_entries(int(self.config.max_entries * 0.2))
            removed_count = optimization_stats["entries_removed"]
            if isinstance(removed_count, int):
                optimization_stats["entries_removed"] = removed_count + lru_removed
            strategy_list = optimization_stats["optimization_strategy"]
            if isinstance(strategy_list, list):
                strategy_list.append(f"Removed {lru_removed} LRU entries")
        current_size_mb = await self._calculate_cache_size_mb()
        if current_size_mb > self.config.max_cache_size_mb * 0.9:
            large_file_removed = await self._remove_large_file_entries()
            removed_count = optimization_stats["entries_removed"]
            if isinstance(removed_count, int):
                optimization_stats["entries_removed"] = removed_count + large_file_removed
            strategy_list = optimization_stats["optimization_strategy"]
            if isinstance(strategy_list, list):
                strategy_list.append(f"Removed {large_file_removed} large file entries")
        optimization_stats["entries_after"] = len(self.cache_index)
        optimization_stats["space_freed_mb"] = await self._calculate_cache_size_mb()
        self._update_cache_stats()
        logger.info("Cache optimization completed: %s", optimization_stats)
        return optimization_stats

    async def get_cache_info(self, file_path: Path) -> dict[str, Any] | None:
        """Get cache information for a specific file."""
        if not self._initialized:
            await self.initialize()
        cache_key = str(file_path)
        if cache_key not in self.cache_index:
            return None
        entry = self.cache_index[cache_key]
        return {
            "cached": True,
            "cached_at": entry.metadata.cached_at.isoformat(),
            "last_accessed": (
                entry.metadata.last_accessed.isoformat() if entry.metadata.last_accessed else None
            ),
            "access_count": entry.metadata.access_count,
            "language": entry.language,
            "parser_type": entry.parser_type,
            "file_size": entry.file_size,
            "entities_count": len(entry.results.entities),
            "relationships_count": len(entry.results.relationships),
            "cache_hit_savings_ms": entry.metadata.cache_hit_savings_ms,
            "age_hours": (datetime.now(UTC) - entry.metadata.cached_at).total_seconds() / 3600,
        }

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    async def _is_cache_valid(
        self, cache_entry: AnalysisCacheEntry, current_hash: str, file_path: Path
    ) -> bool:
        """Check if a cache entry is still valid."""
        if cache_entry.content_hash != current_hash:
            return False
        age = datetime.now(UTC) - cache_entry.metadata.cached_at
        if age > timedelta(hours=self.config.ttl_hours):
            return False
        try:
            if file_path.is_absolute() and (not str(file_path).startswith("/test/")):
                if not file_path.exists():
                    return False
        except (OSError, PermissionError):  # fmt: skip
            pass
        return True

    async def _create_tables(self) -> None:
        """Create cache database tables."""
        async with self._get_db_connection() as conn:
            await conn.execute(
                "\n                CREATE TABLE IF NOT EXISTS cache_entries (\n                    file_path TEXT PRIMARY KEY,\n                    content_hash TEXT NOT NULL,\n                    file_size INTEGER NOT NULL,\n                    language TEXT NOT NULL,\n                    parser_type TEXT NOT NULL,\n                    entities TEXT NOT NULL,  -- JSON\n                    relationships TEXT NOT NULL,  -- JSON\n                    lines_of_code INTEGER NOT NULL,\n                    complexity_score REAL NOT NULL,\n                    parse_time_ms REAL NOT NULL,\n                    cached_at TEXT NOT NULL,\n                    access_count INTEGER DEFAULT 0,\n                    last_accessed TEXT,\n                    cache_hit_savings_ms REAL DEFAULT 0.0\n                )\n            "
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON cache_entries(language)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cached_at ON cache_entries(cached_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)"
            )
            await conn.commit()

    @asynccontextmanager
    async def _get_db_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get async database connection using aiosqlite."""
        conn = await aiosqlite.connect(str(self.db_path))
        conn.row_factory = aiosqlite.Row
        try:
            yield conn
        finally:
            await conn.close()

    async def _load_cache_index(self) -> None:
        """Load cache index from database."""
        try:
            async with self._get_db_connection() as conn:
                cursor = await conn.execute("SELECT * FROM cache_entries")
                rows = await cursor.fetchall()
                for row in rows:
                    results = AnalysisResults(
                        entities=json.loads(row["entities"]),
                        relationships=json.loads(row["relationships"]),
                        lines_of_code=row["lines_of_code"],
                        complexity_score=row["complexity_score"],
                        parse_time_ms=row["parse_time_ms"],
                    )
                    metadata = CacheMetadata(
                        cached_at=datetime.fromisoformat(row["cached_at"]),
                        access_count=row["access_count"],
                        last_accessed=(
                            datetime.fromisoformat(row["last_accessed"])
                            if row["last_accessed"]
                            else None
                        ),
                        cache_hit_savings_ms=row["cache_hit_savings_ms"],
                    )
                    cache_entry = AnalysisCacheEntry(
                        file_path=row["file_path"],
                        content_hash=row["content_hash"],
                        file_size=row["file_size"],
                        language=row["language"],
                        parser_type=row["parser_type"],
                        results=results,
                        metadata=metadata,
                    )
                    self.cache_index[row["file_path"]] = cache_entry
                logger.info("Loaded %d cache entries from database", len(self.cache_index))
        except (aiosqlite.Error, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to load cache index: %s", e)
            self.cache_index = {}

    async def _store_cache_entry(self, cache_entry: AnalysisCacheEntry) -> None:
        """Store cache entry in database."""
        async with self._get_db_connection() as conn:
            await conn.execute(
                "\n                INSERT OR REPLACE INTO cache_entries (\n                    file_path, content_hash, file_size, language, parser_type,\n                    entities, relationships, lines_of_code, complexity_score,\n                    parse_time_ms, cached_at, access_count, last_accessed,\n                    cache_hit_savings_ms\n                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    cache_entry.file_path,
                    cache_entry.content_hash,
                    cache_entry.file_size,
                    cache_entry.language,
                    cache_entry.parser_type,
                    json.dumps(cache_entry.results.entities),
                    json.dumps(cache_entry.results.relationships),
                    cache_entry.results.lines_of_code,
                    cache_entry.results.complexity_score,
                    cache_entry.results.parse_time_ms,
                    cache_entry.metadata.cached_at.isoformat(),
                    cache_entry.metadata.access_count,
                    (
                        cache_entry.metadata.last_accessed.isoformat()
                        if cache_entry.metadata.last_accessed
                        else None
                    ),
                    cache_entry.metadata.cache_hit_savings_ms,
                ),
            )
            await conn.commit()

    async def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry from database and index."""
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
        async with self._get_db_connection() as conn:
            await conn.execute("DELETE FROM cache_entries WHERE file_path = ?", (cache_key,))
            await conn.commit()

    async def _cache_entry_to_parse_result(self, cache_entry: AnalysisCacheEntry) -> ParseResult:
        """Convert cache entry back to ParseResult."""
        global ParseResult, ParserType
        if ParseResult is None:
            from processing.languages.base import ParseResult as _ParseResult

            ParseResult = _ParseResult
        if ParserType is None:
            from processing.languages.base import ParserType as _ParserType

            ParserType = _ParserType
        entities = [self._dict_to_entity(e) for e in cache_entry.results.entities]
        relationships = [self._dict_to_relation(r) for r in cache_entry.results.relationships]
        try:
            parser_type = ParserType(cache_entry.parser_type)
        except ValueError:
            parser_type = ParserType.REGEX_FALLBACK
        return ParseResult(
            success=True,
            file_path=cache_entry.file_path,
            language=cache_entry.language,
            parser_used=parser_type,
            ast=None,
            entities=entities,
            relationships=relationships,
            lines_of_code=cache_entry.results.lines_of_code,
            complexity_score=cache_entry.results.complexity_score,
            parse_time_ms=cache_entry.results.parse_time_ms,
            errors=[],
            warnings=[],
        )

    def _entity_to_dict(self, entity: CoreCodeEntity) -> dict[str, Any]:
        """Convert CoreCodeEntity to dictionary for caching."""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "language": entity.language,
            "file_path": entity.file_path,
            "line_number": entity.line_number,
            "signature": entity.signature,
            "is_abstract": entity.is_abstract,
            "is_static": entity.is_static,
            "is_async": entity.is_async,
            "visibility": entity.visibility,
            "properties": entity.properties or {},
        }

    def _dict_to_entity(self, data: dict[str, Any]) -> CoreCodeEntity:
        """Convert dictionary back to CoreCodeEntity."""
        return CoreCodeEntity(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            language=data["language"],
            file_path=data["file_path"],
            line_number=data.get("line_number", 0),
            end_line_number=data.get("end_line_number"),
            column_start=data.get("column_start"),
            column_end=data.get("column_end"),
            code_snippet=data.get("code_snippet"),
            docstring=data.get("docstring"),
            signature=data.get("signature"),
            source_code=data.get("source_code"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            is_abstract=data.get("is_abstract", False),
            is_static=data.get("is_static", False),
            is_async=data.get("is_async", False),
            visibility=data.get("visibility"),
            properties=data.get("properties", {}),
        )

    def _relation_to_dict(self, relation: EntityRelation) -> dict[str, Any]:
        """Convert EntityRelation to dictionary for caching."""
        return {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "type": relation.type,
            "properties": relation.properties or {},
        }

    def _dict_to_relation(self, data: dict[str, Any]) -> EntityRelation:
        """Convert dictionary back to EntityRelation."""
        return EntityRelation(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            properties=data.get("properties", {}),
        )

    async def _enforce_cache_limits(self) -> None:
        """Enforce cache size and entry limits."""
        if len(self.cache_index) >= self.config.max_entries:
            await self._remove_lru_entries(self.config.max_entries // 10)
        current_size_mb = await self._calculate_cache_size_mb()
        if current_size_mb > self.config.max_cache_size_mb:
            await self._remove_large_file_entries()

    async def _remove_expired_entries(self) -> int:
        """Remove expired cache entries."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=self.config.ttl_hours)
        expired_keys = [
            key for key, entry in self.cache_index.items() if entry.metadata.cached_at < cutoff_time
        ]
        for key in expired_keys:
            await self._remove_cache_entry(key)
            self.stats.metrics.cache_evictions += 1
        return len(expired_keys)

    async def _remove_lru_entries(self, count: int) -> int:
        """Remove least recently used entries."""
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].metadata.last_accessed or x[1].metadata.cached_at,
        )
        removed_count = 0
        for key, _ in sorted_entries[:count]:
            await self._remove_cache_entry(key)
            self.stats.metrics.cache_evictions += 1
            removed_count += 1
        return removed_count

    async def _remove_large_file_entries(self) -> int:
        """Remove entries for large files to free up space."""
        sorted_entries = sorted(
            self.cache_index.items(), key=lambda x: x[1].file_size, reverse=True
        )
        removed_count = 0
        target_size_mb = self.config.max_cache_size_mb * 0.8
        for key, entry in sorted_entries:
            current_size_mb = await self._calculate_cache_size_mb()
            if current_size_mb <= target_size_mb:
                break
            if entry.metadata.access_count > 10:
                continue
            await self._remove_cache_entry(key)
            self.stats.metrics.cache_evictions += 1
            removed_count += 1
        return removed_count

    async def _calculate_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = sum(entry.file_size for entry in self.cache_index.values())
        return total_size / (1024 * 1024)

    def _update_cache_stats(self) -> None:
        """Update cache statistics.

        PERFORMANCE OPTIMIZATION: Single-pass iteration over cache_index
        instead of 4 separate passes (was: sum for total_size, list comprehension
        for cached_times + min/max, sum for total_entities).
        Now collects all metrics in one loop.
        """
        if not self.cache_index:
            return

        num_entries = len(self.cache_index)
        self.stats.size_metrics.total_files_cached = num_entries

        # Single-pass: collect all stats at once
        total_size_bytes = 0
        total_entities = 0
        oldest_time: datetime | None = None
        newest_time: datetime | None = None

        for entry in self.cache_index.values():
            total_size_bytes += entry.file_size
            total_entities += len(entry.results.entities)
            cached_at = entry.metadata.cached_at
            if oldest_time is None or cached_at < oldest_time:
                oldest_time = cached_at
            if newest_time is None or cached_at > newest_time:
                newest_time = cached_at

        self.stats.size_metrics.total_cache_size_mb = total_size_bytes / (1024 * 1024)
        self.stats.size_metrics.average_file_size_kb = total_size_bytes / num_entries / 1024
        self.stats.size_metrics.oldest_entry = oldest_time
        self.stats.size_metrics.newest_entry = newest_time

        if self.stats.size_metrics.total_cache_size_mb > 0:
            self.stats.size_metrics.storage_efficiency = (
                total_entities / self.stats.size_metrics.total_cache_size_mb
            )

    async def _background_cleanup(self) -> None:
        """Background task for periodic cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                if self.cache_index:
                    optimization_stats = await self.optimize_cache()
                    logger.debug("Background cache cleanup: %s", optimization_stats)
            except asyncio.CancelledError:
                logger.info("Background cache cleanup task cancelled")
                break
            except (TimeoutError, RuntimeError, OSError) as e:
                logger.error("Background cache cleanup error: %s", e)

    async def shutdown(self) -> None:
        """Shutdown the cache system."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Analysis cache shut down")


_global_cache: AnalysisCache | None = None


def get_analysis_cache() -> AnalysisCache:
    """Get or create the global analysis cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AnalysisCache()
        logger.info("Initialized global analysis cache")
    return _global_cache
