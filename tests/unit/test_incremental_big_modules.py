"""Unit tests covering five incremental modules:

- incremental_analysis_service.py
- incremental_update_config.py
- incremental_updater.py
- incremental_update_change_detector.py
- incremental_update_processor.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stubs -- only models.github_models (does not exist in the codebase)
# ---------------------------------------------------------------------------
if "models.github_models" not in sys.modules:
    _ghm = _types.ModuleType("models.github_models")
    _ghm.CodeSnippet = MagicMock
    _ghm.RepositoryContext = MagicMock
    sys.modules["models.github_models"] = _ghm

# ---------------------------------------------------------------------------
# Patch settings before importing modules under test
# ---------------------------------------------------------------------------
import app.core.config.core_config_settings as _settings_mod

_mock_settings = MagicMock()
_mock_settings.CODE_GRAPH_DATABASE_DIR = "/tmp/test_graphs"

# Stub out recovery / validator so that incremental_updater imports cleanly
if "services.incremental.incremental_update_recovery" not in sys.modules:
    _rec = _types.ModuleType("services.incremental.incremental_update_recovery")
    _rec.UpdateRecoveryManager = MagicMock
    sys.modules["services.incremental.incremental_update_recovery"] = _rec

if "services.incremental.incremental_update_validator" not in sys.modules:
    _val = _types.ModuleType("services.incremental.incremental_update_validator")
    _val.UpdateValidator = MagicMock
    sys.modules["services.incremental.incremental_update_validator"] = _val

with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
    import services.incremental.incremental_update_change_detector as _cd_mod
    import services.incremental.incremental_update_config as _cfg_mod
    import services.incremental.incremental_update_processor as _proc_mod
    import services.incremental.incremental_updater as _upd_mod

    importlib.reload(_cd_mod)
    importlib.reload(_cfg_mod)
    importlib.reload(_proc_mod)
    importlib.reload(_upd_mod)

# Also import the analysis service (no settings dep)
from services.incremental.incremental_analysis_service import (
    ChangeType,
    DiffFormat,
    DiffHunk,
    FileAnalysisHistory,
    FileDiff,
    IncrementalAnalysisService,
    get_incremental_analysis_service,
)

ChangeDetector = _cd_mod.ChangeDetector
UpdateConfig = _cfg_mod.UpdateConfig
VersionManager = _cfg_mod.VersionManager
DatabaseUtilities = _cfg_mod.DatabaseUtilities
UpdateUtilities = _cfg_mod.UpdateUtilities
get_update_config = _cfg_mod.get_update_config
IncrementalUpdateProcessor = _proc_mod.IncrementalUpdateProcessor
IncrementalUpdater = _upd_mod.IncrementalUpdater

from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
    EntityType,
    RelationType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_db_conn():
    """Mock GraphDatabaseConnection facade for the postgres call sites.

    Source ``await``s the facade ``connect()`` / ``close()`` and the raw
    ``conn.execute(...)`` / ``conn.commit()``; ``conn.execute`` returns a cursor.
    """
    conn = MagicMock()
    conn.connect = AsyncMock()
    conn.close = AsyncMock()
    conn.conn = MagicMock()
    conn.conn.execute = AsyncMock(return_value=_make_cursor())
    conn.conn.commit = AsyncMock()
    conn.conn.close = AsyncMock()
    return conn


def _make_cursor(rows=None, rowcount=0):
    """Fake psycopg async cursor (tuple rows, positional access).

    Source ``await``s ``fetchone()`` / ``fetchall()``, reads ``rowcount`` (sync),
    and iterates ``async for row in cursor`` in ``_get_related_nodes``.
    """
    rows = list(rows) if rows else []
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=rows[0] if rows else None)
    cursor.fetchall = AsyncMock(return_value=rows)
    cursor.rowcount = rowcount

    def _aiter(_self):
        async def _gen():
            for r in rows:
                yield r

        return _gen()

    cursor.__aiter__ = _aiter
    return cursor


def _make_entity(
    entity_id="e1",
    name="foo",
    line=10,
    end_line=20,
    entity_type=EntityType.FUNCTION,
):
    return CoreCodeEntity(
        id=entity_id,
        name=name,
        type=entity_type,
        language="python",
        file_path="test.py",
        line_number=line,
        end_line_number=end_line,
    )


def _make_relation(source="e1", target="e2", rel_type=RelationType.CALLS):
    return EntityRelation(source_id=source, target_id=target, type=rel_type)


def _make_analysis_result(file_path="test.py", entities=None, relationships=None):
    return CoreAnalysisResult(
        success=True,
        file_path=file_path,
        language="python",
        entities=entities or [],
        relationships=relationships or [],
        dependencies=[],
        complexity_metrics=CoreComplexityMetrics(),
        security_patterns=[],
    )


# ============================================================================
# 1. IncrementalAnalysisService tests
# ============================================================================


class TestDiffModels:
    """Test DiffHunk, FileDiff, ChangeType, DiffFormat."""

    def test_change_type_values(self):
        assert ChangeType.ADDED == "added"
        assert ChangeType.DELETED == "deleted"
        assert ChangeType.MODIFIED == "modified"
        assert ChangeType.UNCHANGED == "unchanged"

    def test_diff_format_values(self):
        assert DiffFormat.UNIFIED == "unified"
        assert DiffFormat.INLINE == "inline"
        assert DiffFormat.STRUCTURED == "structured"

    def test_diff_hunk_defaults(self):
        h = DiffHunk(old_start=1, old_lines=5, new_start=1, new_lines=8)
        assert h.content == ""
        assert h.context_before == []
        assert h.context_after == []

    def test_file_diff_affected_line_ranges(self):
        hunks = [
            DiffHunk(old_start=10, old_lines=5, new_start=10, new_lines=8),
            DiffHunk(old_start=30, old_lines=2, new_start=33, new_lines=3),
        ]
        fd = FileDiff(file_path="a.py", hunks=hunks)
        ranges = fd.get_affected_line_ranges()
        assert len(ranges) == 2
        # first hunk: max(1,10-3)=7, end=10+8+3=21
        assert ranges[0] == (7, 21)
        # second hunk: max(1,33-3)=30, end=33+3+3=39
        assert ranges[1] == (30, 39)

    def test_file_diff_affected_line_ranges_near_start(self):
        hunks = [DiffHunk(old_start=1, old_lines=1, new_start=1, new_lines=2)]
        fd = FileDiff(file_path="a.py", hunks=hunks)
        ranges = fd.get_affected_line_ranges()
        assert ranges[0][0] == 1  # clamped to 1

    def test_file_diff_change_summary_added(self):
        hunks = [DiffHunk(old_start=1, old_lines=2, new_start=1, new_lines=5)]
        fd = FileDiff(file_path="a.py", hunks=hunks)
        summary = fd.get_change_summary()
        assert summary["hunks"] == 1
        assert summary["lines_added"] == 3
        assert summary["lines_deleted"] == 0
        assert summary["lines_changed"] == 5

    def test_file_diff_change_summary_deleted(self):
        hunks = [DiffHunk(old_start=1, old_lines=5, new_start=1, new_lines=2)]
        fd = FileDiff(file_path="a.py", hunks=hunks)
        summary = fd.get_change_summary()
        assert summary["lines_added"] == 0
        assert summary["lines_deleted"] == 3

    def test_file_diff_change_summary_mixed(self):
        hunks = [
            DiffHunk(old_start=1, old_lines=2, new_start=1, new_lines=5),
            DiffHunk(old_start=10, old_lines=6, new_start=13, new_lines=3),
        ]
        fd = FileDiff(file_path="a.py", hunks=hunks)
        summary = fd.get_change_summary()
        assert summary["hunks"] == 2
        assert summary["lines_added"] == 3
        assert summary["lines_deleted"] == 3

    def test_file_diff_empty_hunks(self):
        fd = FileDiff(file_path="a.py", hunks=[])
        assert fd.get_affected_line_ranges() == []
        summary = fd.get_change_summary()
        assert summary["hunks"] == 0
        assert summary["lines_added"] == 0


class TestFileAnalysisHistory:
    """Test FileAnalysisHistory."""

    def test_init(self):
        h = FileAnalysisHistory("foo.py", max_versions=5)
        assert h.file_path == "foo.py"
        assert h.max_versions == 5
        assert h.versions == []
        assert h.current_version is None

    def test_add_version_basic(self):
        h = FileAnalysisHistory("foo.py")
        result = _make_analysis_result()
        v = h.add_version(result, "content")
        assert v.version == 1
        assert h.current_version is v
        assert len(h.versions) == 1
        assert v.file_size == len("content")
        assert v.line_count == 1

    def test_add_version_increments(self):
        h = FileAnalysisHistory("foo.py")
        v1 = h.add_version(_make_analysis_result(), "a")
        v2 = h.add_version(_make_analysis_result(), "b")
        assert v1.version == 1
        assert v2.version == 2
        assert h.current_version is v2

    def test_add_version_trims(self):
        h = FileAnalysisHistory("foo.py", max_versions=3)
        for i in range(5):
            h.add_version(_make_analysis_result(), f"content_{i}")
        assert len(h.versions) == 3
        assert h.versions[0].version == 3

    def test_get_current_none(self):
        h = FileAnalysisHistory("foo.py")
        assert h.get_current() is None

    def test_get_version_found(self):
        h = FileAnalysisHistory("foo.py")
        h.add_version(_make_analysis_result(), "a")
        h.add_version(_make_analysis_result(), "b")
        v = h.get_version(1)
        assert v is not None
        assert v.version == 1

    def test_get_version_not_found(self):
        h = FileAnalysisHistory("foo.py")
        assert h.get_version(99) is None

    def test_entity_cache(self):
        h = FileAnalysisHistory("foo.py")
        entity = _make_entity()
        result = _make_analysis_result(entities=[entity])
        h.add_version(result, "content")
        assert h.get_entity("e1") is entity
        assert h.get_entity("nonexistent") is None

    def test_relation_cache(self):
        h = FileAnalysisHistory("foo.py")
        rel = _make_relation()
        result = _make_analysis_result(relationships=[rel])
        h.add_version(result, "content")
        assert h.has_relation("e1", "e2", RelationType.CALLS)
        assert not h.has_relation("e1", "e2", RelationType.IMPORTS)

    def test_hash_content(self):
        h = FileAnalysisHistory("foo.py")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert h._hash_content("hello") == expected

    def test_entity_cache_empty_result(self):
        h = FileAnalysisHistory("foo.py")
        result = _make_analysis_result(entities=[])
        h.add_version(result, "content")
        assert h.entity_cache == {}

    def test_relation_cache_empty_result(self):
        h = FileAnalysisHistory("foo.py")
        result = _make_analysis_result(relationships=[])
        h.add_version(result, "content")
        assert h.relation_cache == set()


class TestIncrementalAnalysisService:
    """Tests for the main IncrementalAnalysisService class."""

    def test_init_defaults(self):
        svc = IncrementalAnalysisService()
        assert svc.max_history == 10
        assert svc.enable_caching is True
        assert svc.debounce_ms == 500
        assert svc.stats["full_analyses"] == 0

    def test_init_custom(self):
        svc = IncrementalAnalysisService(
            max_history_per_file=5, enable_caching=False, debounce_ms=100
        )
        assert svc.max_history == 5
        assert svc.enable_caching is False
        assert svc.debounce_ms == 100

    @pytest.mark.asyncio
    async def test_analyze_file_full(self):
        svc = IncrementalAnalysisService()
        result = await svc.analyze_file_full("test.py", "x = 1\n")
        assert result.success is True
        assert result.file_path == "test.py"
        assert result.language == "python"
        assert svc.stats["full_analyses"] == 1
        history = svc.get_file_history("test.py")
        assert history is not None
        assert len(history.versions) == 1

    @pytest.mark.asyncio
    async def test_analyze_file_full_explicit_language(self):
        svc = IncrementalAnalysisService()
        result = await svc.analyze_file_full("file.txt", "code", language="rust")
        assert result.language == "rust"

    @pytest.mark.asyncio
    async def test_analyze_file_diff_no_history_full_fallback(self):
        svc = IncrementalAnalysisService(debounce_ms=0)
        diff = FileDiff(file_path="new.py", new_content="x = 1\n")
        result = await svc.analyze_file_diff(diff, force_full=True)
        assert result.success is True
        assert svc.stats["full_analyses"] == 1

    @pytest.mark.asyncio
    async def test_analyze_file_diff_no_content(self):
        svc = IncrementalAnalysisService(debounce_ms=0)
        diff = FileDiff(file_path="new.py")
        result = await svc.analyze_file_diff(diff, force_full=True)
        assert result.success is False  # empty result

    @pytest.mark.asyncio
    async def test_analyze_file_diff_cache_hit(self):
        svc = IncrementalAnalysisService(debounce_ms=0)
        content = "x = 1\n"
        await svc.analyze_file_full("t.py", content)
        diff = FileDiff(file_path="t.py", new_content=content)
        result = await svc.analyze_file_diff(diff)
        assert result.success is True
        assert svc.stats["cache_hits"] == 1
        assert svc.stats["incremental_analyses"] == 1

    @pytest.mark.asyncio
    async def test_analyze_file_diff_incremental(self):
        svc = IncrementalAnalysisService(debounce_ms=0)
        old = "line1\nline2\nline3\n"
        new = "line1\nline2_changed\nline3\n"
        await svc.analyze_file_full("t.py", old)
        hunks = [DiffHunk(old_start=2, old_lines=1, new_start=2, new_lines=1)]
        diff = FileDiff(file_path="t.py", old_content=old, new_content=new, hunks=hunks)
        result = await svc.analyze_file_diff(diff)
        assert result.success is True
        assert svc.stats["incremental_analyses"] == 1
        assert svc.stats["cache_hits"] == 0

    @pytest.mark.asyncio
    async def test_analyze_file_diff_with_debounce(self):
        """Debounced analysis returns current result when history exists."""
        svc = IncrementalAnalysisService(debounce_ms=500)
        await svc.analyze_file_full("t.py", "x = 1")
        diff = FileDiff(file_path="t.py", new_content="x = 2")
        result = await svc.analyze_file_diff(diff)
        # Should return the existing result (debounce returns current)
        assert result.success is True
        assert svc.stats["debounced_requests"] == 1

    @pytest.mark.asyncio
    async def test_debounce_no_history_awaits_task(self):
        """Debounced analysis with no history awaits the task."""
        svc = IncrementalAnalysisService(debounce_ms=1)
        diff = FileDiff(file_path="brand_new.py", new_content="x = 1")
        result = await svc.analyze_file_diff(diff)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_extract_changed_regions_from_hunks(self):
        svc = IncrementalAnalysisService()
        content = "\n".join(f"line{i}" for i in range(20))
        hunks = [DiffHunk(old_start=5, old_lines=2, new_start=5, new_lines=3)]
        diff = FileDiff(file_path="t.py", new_content=content, hunks=hunks)
        regions = await svc._extract_changed_regions(diff)
        assert len(regions) == 1
        assert regions[0][0] == 5
        assert regions[0][1] == 8

    @pytest.mark.asyncio
    async def test_extract_changed_regions_from_content_diff(self):
        svc = IncrementalAnalysisService()
        old = "line1\nline2\nline3"
        new = "line1\nchanged\nline3"
        diff = FileDiff(file_path="t.py", old_content=old, new_content=new)
        regions = await svc._extract_changed_regions(diff)
        assert len(regions) >= 1

    @pytest.mark.asyncio
    async def test_compute_diff_regions(self):
        svc = IncrementalAnalysisService()
        old = "a\nb\nc"
        new = "a\nX\nc"
        regions = await svc._compute_diff_regions(old, new)
        assert len(regions) >= 1
        # Region should contain "X"
        assert "X" in regions[0][2]

    @pytest.mark.asyncio
    async def test_analyze_content_snippet(self):
        svc = IncrementalAnalysisService()
        result = await svc._analyze_content_snippet("t.py", "x=1", None, line_offset=10)
        assert result.success is True
        assert result.metadata["line_offset"] == 10
        assert result.metadata["is_snippet"] is True

    @pytest.mark.asyncio
    async def test_analyze_regions(self):
        svc = IncrementalAnalysisService()
        regions = [(5, 10, "code_here")]
        results = await svc._analyze_regions("t.py", "full content", regions, "python")
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_analyze_regions_exception_skips(self):
        svc = IncrementalAnalysisService()
        with patch.object(svc, "_analyze_content_snippet", side_effect=RuntimeError("boom")):
            results = await svc._analyze_regions("t.py", "content", [(1, 5, "code")], "python")
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_merge_results(self):
        svc = IncrementalAnalysisService()
        e1 = _make_entity("e1", "foo", 1, 5)
        e2 = _make_entity("e2", "bar", 50, 60)
        rel = _make_relation("e1", "e2")
        base = _make_analysis_result(entities=[e1, e2], relationships=[rel])

        partial_entity = _make_entity("e3", "baz", 3, 4)
        partial = _make_analysis_result(entities=[partial_entity])

        diff = FileDiff(file_path="t.py")
        regions = [(1, 6, "code")]
        merged = await svc._merge_results(base, [partial], diff, regions)

        assert merged.success is True
        # e1 is in affected range (1-6) so removed; e2 preserved
        entity_ids = {e.id for e in merged.entities}
        assert "e2" in entity_ids
        assert "e3" in entity_ids

    @pytest.mark.asyncio
    async def test_merge_results_preserves_relations(self):
        svc = IncrementalAnalysisService()
        e1 = _make_entity("e1", "foo", 50, 60)
        e2 = _make_entity("e2", "bar", 70, 80)
        rel = _make_relation("e1", "e2")
        base = _make_analysis_result(entities=[e1, e2], relationships=[rel])
        diff = FileDiff(file_path="t.py")
        regions = [(1, 5, "code")]
        merged = await svc._merge_results(base, [], diff, regions)
        assert len(merged.relationships) == 1

    def test_is_entity_in_affected_region_overlap(self):
        svc = IncrementalAnalysisService()
        e = _make_entity(line=10, end_line=15)
        assert svc._is_entity_in_affected_region(e, {(12, 18)}) is True
        assert svc._is_entity_in_affected_region(e, {(1, 5)}) is False
        assert svc._is_entity_in_affected_region(e, {(14, 20)}) is True

    def test_is_entity_in_affected_region_no_line(self):
        svc = IncrementalAnalysisService()
        e = _make_entity(line=0, end_line=None)
        # line_number=0 is falsy
        assert svc._is_entity_in_affected_region(e, {(1, 10)}) is False

    def test_filter_valid_relationships(self):
        svc = IncrementalAnalysisService()
        e1 = _make_entity("e1", "a", 1, 5)
        e2 = _make_entity("e2", "b", 10, 15)
        r1 = _make_relation("e1", "e2")
        r2 = _make_relation("e1", "e3")  # e3 not in valid list
        valid = svc._filter_valid_relationships([r1, r2], [e1, e2], set())
        assert len(valid) == 1
        assert valid[0].source_id == "e1"
        assert valid[0].target_id == "e2"

    def test_merge_complexity_metrics_base(self):
        svc = IncrementalAnalysisService()
        base = CoreComplexityMetrics(cyclomatic_complexity=5.0)
        result = svc._merge_complexity_metrics(base, [])
        assert result.cyclomatic_complexity == 5.0

    def test_merge_complexity_metrics_no_base(self):
        svc = IncrementalAnalysisService()
        partial = CoreComplexityMetrics(cyclomatic_complexity=3.0)
        result = svc._merge_complexity_metrics(None, [partial])
        assert result.cyclomatic_complexity == 3.0

    def test_merge_complexity_metrics_empty(self):
        svc = IncrementalAnalysisService()
        result = svc._merge_complexity_metrics(None, [])
        assert result.cyclomatic_complexity == 0.0

    def test_detect_language(self):
        svc = IncrementalAnalysisService()
        assert svc._detect_language("foo.py") == "python"
        assert svc._detect_language("bar.js") == "javascript"
        assert svc._detect_language("baz.ts") == "typescript"
        assert svc._detect_language("qux.tsx") == "typescript"
        assert svc._detect_language("app.java") == "java"
        assert svc._detect_language("main.go") == "go"
        assert svc._detect_language("lib.rs") == "rust"
        assert svc._detect_language("code.cpp") == "cpp"
        assert svc._detect_language("code.cc") == "cpp"
        assert svc._detect_language("code.c") == "c"
        assert svc._detect_language("code.h") == "c"
        assert svc._detect_language("code.cs") == "csharp"
        assert svc._detect_language("code.rb") == "ruby"
        assert svc._detect_language("code.php") == "php"
        assert svc._detect_language("code.xyz") == "unknown"

    def test_create_empty_result(self):
        svc = IncrementalAnalysisService()
        r = svc._create_empty_result("file.py", "test error")
        assert r.success is False
        assert r.error == "test error"
        assert r.language == "python"

    def test_parse_unified_diff_single_file(self):
        svc = IncrementalAnalysisService()
        diff_text = (
            "--- a/foo.py\t2024-01-01\n"
            "+++ b/foo.py\t2024-01-02\n"
            "@@ -10,5 +10,8 @@\n"
            " context\n"
            "-old line\n"
            "+new line\n"
            "+added line\n"
        )
        diffs = svc.parse_unified_diff(diff_text)
        assert len(diffs) == 1
        assert diffs[0].file_path == "foo.py"
        assert len(diffs[0].hunks) == 1
        assert diffs[0].hunks[0].old_start == 10
        assert diffs[0].hunks[0].new_lines == 8

    def test_parse_unified_diff_two_files(self):
        svc = IncrementalAnalysisService()
        diff_text = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+new\n"
            "--- a/bar.py\n"
            "+++ b/bar.py\n"
            "@@ -5,3 +5,4 @@\n"
            "+stuff\n"
        )
        diffs = svc.parse_unified_diff(diff_text)
        assert len(diffs) == 2
        assert diffs[0].file_path == "foo.py"
        assert diffs[1].file_path == "bar.py"

    def test_parse_unified_diff_no_a_prefix(self):
        svc = IncrementalAnalysisService()
        diff_text = "--- plain.py\n+++ plain.py\n@@ -1,1 +1,2 @@\n+x\n"
        diffs = svc.parse_unified_diff(diff_text)
        assert diffs[0].file_path == "plain.py"

    def test_parse_hunk_header_valid(self):
        svc = IncrementalAnalysisService()
        h = svc._parse_hunk_header("@@ -10,5 +10,8 @@ def foo():")
        assert h is not None
        assert h.old_start == 10
        assert h.old_lines == 5
        assert h.new_start == 10
        assert h.new_lines == 8

    def test_parse_hunk_header_invalid(self):
        svc = IncrementalAnalysisService()
        assert svc._parse_hunk_header("not a hunk") is None

    def test_create_diff_from_strings(self):
        svc = IncrementalAnalysisService()
        old = "line1\nline2\nline3"
        new = "line1\nCHANGED\nline3"
        diff = svc.create_diff_from_strings("t.py", old, new)
        assert diff.file_path == "t.py"
        assert diff.old_content == old
        assert diff.new_content == new
        assert diff.format == DiffFormat.INLINE
        assert len(diff.hunks) >= 1

    def test_create_diff_from_strings_insert(self):
        svc = IncrementalAnalysisService()
        old = "a\nb"
        new = "a\nINSERTED\nb"
        diff = svc.create_diff_from_strings("t.py", old, new)
        assert len(diff.hunks) >= 1
        # One hunk should contain "+"
        assert any("+" in h.content for h in diff.hunks)

    def test_create_diff_from_strings_delete(self):
        svc = IncrementalAnalysisService()
        old = "a\nDELETE_ME\nb"
        new = "a\nb"
        diff = svc.create_diff_from_strings("t.py", old, new)
        assert len(diff.hunks) >= 1
        assert any("-" in h.content for h in diff.hunks)

    def test_get_file_history(self):
        svc = IncrementalAnalysisService()
        assert svc.get_file_history("nope.py") is None

    def test_get_statistics(self):
        svc = IncrementalAnalysisService()
        stats = svc.get_statistics()
        assert stats["tracked_files"] == 0
        assert stats["total_versions"] == 0
        assert stats["pending_analyses"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_after_analysis(self):
        svc = IncrementalAnalysisService()
        await svc.analyze_file_full("a.py", "x=1")
        await svc.analyze_file_full("b.py", "y=2")
        stats = svc.get_statistics()
        assert stats["tracked_files"] == 2
        assert stats["total_versions"] == 2
        assert stats["full_analyses"] == 2

    def test_clear_history_specific_file(self):
        svc = IncrementalAnalysisService()
        svc._get_or_create_history("a.py")
        svc._get_or_create_history("b.py")
        svc.clear_history("a.py")
        assert "a.py" not in svc.file_histories
        assert "b.py" in svc.file_histories

    def test_clear_history_all(self):
        svc = IncrementalAnalysisService()
        svc._get_or_create_history("a.py")
        svc._get_or_create_history("b.py")
        svc.clear_history()
        assert len(svc.file_histories) == 0

    @pytest.mark.asyncio
    async def test_get_dependency_impact_empty(self):
        svc = IncrementalAnalysisService()
        assert svc.get_dependency_impact("nope.py") == []

    @pytest.mark.asyncio
    async def test_get_dependency_impact_with_relations(self):
        svc = IncrementalAnalysisService()
        e1 = _make_entity("e1", "foo", 1, 5)
        e2 = _make_entity("e2", "bar", 1, 5)
        r = _make_relation("e2", "e1")

        result_a = _make_analysis_result("a.py", entities=[e1])
        result_b = _make_analysis_result("b.py", entities=[e2], relationships=[r])

        svc._get_or_create_history("a.py").add_version(result_a, "c1")
        svc._get_or_create_history("b.py").add_version(result_b, "c2")

        impacted = svc.get_dependency_impact("a.py")
        assert "b.py" in impacted

    @pytest.mark.asyncio
    async def test_get_dependency_impact_no_entities(self):
        svc = IncrementalAnalysisService()
        result = _make_analysis_result("a.py", entities=[])
        svc._get_or_create_history("a.py").add_version(result, "c")
        assert svc.get_dependency_impact("a.py") == []

    def test_get_or_create_history_creates(self):
        svc = IncrementalAnalysisService()
        h = svc._get_or_create_history("new.py")
        assert isinstance(h, FileAnalysisHistory)
        assert h.file_path == "new.py"

    def test_get_or_create_history_returns_existing(self):
        svc = IncrementalAnalysisService()
        h1 = svc._get_or_create_history("x.py")
        h2 = svc._get_or_create_history("x.py")
        assert h1 is h2


class TestGetIncrementalAnalysisService:
    """Test the singleton factory."""

    def test_singleton(self):
        import services.incremental.incremental_analysis_service as mod

        mod._service_instance = None
        s1 = get_incremental_analysis_service()
        s2 = get_incremental_analysis_service()
        assert s1 is s2
        mod._service_instance = None  # reset


# ============================================================================
# 2. UpdateConfig tests
# ============================================================================


class TestUpdateConfig:
    """Tests for UpdateConfig."""

    def test_init(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            assert cfg._config is not None

    def test_get_top_level(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            us = cfg.get("update_settings")
            assert isinstance(us, dict)
            assert "max_parallel_files" in us

    def test_get_dot_notation(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            assert cfg.get("update_settings.max_parallel_files") == 10

    def test_get_default(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            assert cfg.get("nonexistent.key", 42) == 42

    def test_set_existing(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("update_settings.max_parallel_files", 20)
            assert cfg.get("update_settings.max_parallel_files") == 20

    def test_set_new_nested(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("custom.nested.key", "val")
            assert cfg.get("custom.nested.key") == "val"

    def test_validate_config_valid(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            result = cfg.validate_config()
            assert result["valid"] is True
            assert result["errors"] == []

    def test_validate_config_invalid_parallel(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("update_settings.max_parallel_files", -1)
            result = cfg.validate_config()
            assert result["valid"] is False
            assert any("max_parallel_files" in e for e in result["errors"])

    def test_validate_config_invalid_processing_time(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("update_settings.max_processing_time", 0)
            result = cfg.validate_config()
            assert result["valid"] is False

    def test_validate_config_invalid_file_size(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("file_tracking.max_file_size", -100)
            result = cfg.validate_config()
            assert result["valid"] is False

    def test_validate_config_invalid_timeout(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("database_settings.connection_timeout", "not_a_number")
            result = cfg.validate_config()
            assert result["valid"] is False


class TestVersionManager:
    """Tests for VersionManager."""

    @pytest.mark.asyncio
    async def test_create_version(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            vid = await vm.create_version("test desc")
            assert vid.startswith("v_")
            mock_conn.conn.execute.assert_awaited_once()
            mock_conn.conn.commit.assert_awaited_once()
            # source closes the facade (db_conn.close), not the raw conn
            mock_conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_version_found(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rows=[("v_123", "desc", "2024-01-01")])
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            v = await vm.get_version("v_123")
            assert v is not None
            assert v["id"] == "v_123"

    @pytest.mark.asyncio
    async def test_get_version_not_found(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rows=None)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            v = await vm.get_version("v_999")
            assert v is None

    @pytest.mark.asyncio
    async def test_list_versions(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        rows = [("v_1", "first", "2024-01"), ("v_2", "second", "2024-02")]
        cursor = _make_cursor(rows=rows)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            versions = await vm.list_versions()
            assert len(versions) == 2

    @pytest.mark.asyncio
    async def test_list_versions_with_limit(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rows=[("v_1", "d", "t")])
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            await vm.list_versions(limit=5)
            sql, params = mock_conn.conn.execute.await_args.args
            # limit is bound as a %s parameter, not interpolated into the SQL
            assert "LIMIT %s" in " ".join(sql.split())
            assert params[-1] == 5

    @pytest.mark.asyncio
    async def test_delete_version_found(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rowcount=1)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            assert await vm.delete_version("v_1") is True

    @pytest.mark.asyncio
    async def test_delete_version_not_found(self):
        vm = VersionManager("test_graph")
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rowcount=0)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(vm, "_get_db_connection", return_value=mock_conn):
            assert await vm.delete_version("v_none") is False


class TestDatabaseUtilities:
    """Tests for DatabaseUtilities."""

    @pytest.mark.asyncio
    async def test_get_database_stats(self):
        dbu = DatabaseUtilities("test_graph")
        mock_conn = _make_mock_db_conn()
        # node count, edge count, version count, node types, edge types
        mock_conn.conn.execute.side_effect = [
            _make_cursor(rows=[(100,)]),
            _make_cursor(rows=[(50,)]),
            _make_cursor(rows=[(3,)]),
            _make_cursor(rows=[("function", 40), ("class", 60)]),
            _make_cursor(rows=[("calls", 30), ("imports", 20)]),
        ]
        with patch.object(dbu, "_get_db_connection", return_value=mock_conn):
            stats = await dbu.get_database_stats()
            assert stats["node_count"] == 100
            assert stats["edge_count"] == 50
            assert stats["version_count"] == 3
            # No per-graph file under shared PostgreSQL; file_size is always 0.
            assert stats["file_size"] == 0

    @pytest.mark.asyncio
    async def test_get_database_stats_no_file(self):
        dbu = DatabaseUtilities("test_graph")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = [
            _make_cursor(rows=[(0,)]),
            _make_cursor(rows=[(0,)]),
            _make_cursor(rows=[(0,)]),
            _make_cursor(rows=[]),
            _make_cursor(rows=[]),
        ]
        with patch.object(dbu, "_get_db_connection", return_value=mock_conn):
            stats = await dbu.get_database_stats()
            assert stats["file_size"] == 0

    @pytest.mark.asyncio
    async def test_vacuum_database_noop(self):
        # Under shared PostgreSQL vacuum is a no-op (autovacuum handles space;
        # explicit VACUUM can't run inside a pooled PgBouncer txn-mode conn).
        dbu = DatabaseUtilities("test_graph")
        result = await dbu.vacuum_database()
        assert result["success"] is True
        assert result["space_reclaimed"] == 0
        assert result["size_before"] == 0
        assert result["size_after"] == 0

    @pytest.mark.asyncio
    async def test_analyze_database(self):
        dbu = DatabaseUtilities("test_graph")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = [
            _make_cursor(rows=[]),  # indexes
            _make_cursor(rows=[(15000,)]),  # node count
            _make_cursor(rows=[(15000,)]),  # edge count
        ]
        with patch.object(dbu, "_get_db_connection", return_value=mock_conn):
            analysis = await dbu.analyze_database()
            assert analysis["statistics"]["node_count"] == 15000
            assert len(analysis["recommendations"]) >= 1

    @pytest.mark.asyncio
    async def test_analyze_database_small(self):
        dbu = DatabaseUtilities("test_graph")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = [
            _make_cursor(rows=[("idx_nodes_type",), ("idx_edges_source",)]),  # indexes
            _make_cursor(rows=[(50,)]),  # node count
            _make_cursor(rows=[(30,)]),  # edge count
        ]
        with patch.object(dbu, "_get_db_connection", return_value=mock_conn):
            analysis = await dbu.analyze_database()
            assert analysis["recommendations"] == []


class TestUpdateUtilities:
    """Tests for UpdateUtilities static methods."""

    def test_generate_node_id(self):
        nid = UpdateUtilities.generate_node_id("function", "foo", "test.py")
        assert nid.startswith("function_")
        assert len(nid) == len("function_") + 8

    def test_generate_node_id_deterministic(self):
        a = UpdateUtilities.generate_node_id("class", "Bar", "x.py")
        b = UpdateUtilities.generate_node_id("class", "Bar", "x.py")
        assert a == b

    def test_calculate_file_checksum_md5(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        cs = UpdateUtilities.calculate_file_checksum(f, "md5")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert cs == expected

    def test_calculate_file_checksum_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        cs = UpdateUtilities.calculate_file_checksum(f, "sha256")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert cs == expected

    def test_calculate_file_checksum_unsupported(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="Unsupported"):
            UpdateUtilities.calculate_file_checksum(f, "sha512")

    def test_calculate_file_checksum_missing_file(self):
        cs = UpdateUtilities.calculate_file_checksum(Path("/nonexistent/file.txt"))
        assert cs.startswith("fallback_")

    def test_is_supported_file_valid(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
        assert UpdateUtilities.is_supported_file(f, cfg) is True

    def test_is_supported_file_wrong_extension(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
        assert UpdateUtilities.is_supported_file(f, cfg) is False

    def test_is_supported_file_too_large(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x")
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
            cfg.set("file_tracking.max_file_size", 0)  # 0 bytes max
        assert UpdateUtilities.is_supported_file(f, cfg) is False

    def test_is_supported_file_excluded_dir(self, tmp_path):
        d = tmp_path / "__pycache__"
        d.mkdir()
        f = d / "test.py"
        f.write_text("x")
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
        assert UpdateUtilities.is_supported_file(f, cfg) is False

    def test_is_supported_file_nonexistent(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            cfg = UpdateConfig()
        assert UpdateUtilities.is_supported_file(Path("/no/such.py"), cfg) is False

    def test_format_duration_seconds(self):
        assert UpdateUtilities.format_duration(30.5) == "30.50s"

    def test_format_duration_minutes(self):
        assert UpdateUtilities.format_duration(120) == "2.0m"

    def test_format_duration_hours(self):
        assert UpdateUtilities.format_duration(7200) == "2.0h"

    def test_format_file_size_bytes(self):
        assert UpdateUtilities.format_file_size(500) == "500.0B"

    def test_format_file_size_kb(self):
        assert UpdateUtilities.format_file_size(2048) == "2.0KB"

    def test_format_file_size_mb(self):
        assert UpdateUtilities.format_file_size(5 * 1024 * 1024) == "5.0MB"

    def test_format_file_size_gb(self):
        assert UpdateUtilities.format_file_size(3 * 1024**3) == "3.0GB"

    def test_validate_version_id_valid(self):
        assert UpdateUtilities.validate_version_id("v_12345") is True
        assert UpdateUtilities.validate_version_id("v_abc-def_123") is True

    def test_validate_version_id_invalid(self):
        assert UpdateUtilities.validate_version_id("12345") is False
        assert UpdateUtilities.validate_version_id("v_") is False
        assert UpdateUtilities.validate_version_id("v_abc!@#") is False

    def test_create_statistics_summary_basic(self):
        stats = {
            "added_nodes": 10,
            "modified_nodes": 5,
            "deleted_nodes": 2,
            "added_edges": 8,
            "deleted_edges": 3,
            "files_processed": 4,
            "processing_time": 2.0,
        }
        summary = UpdateUtilities.create_statistics_summary(stats)
        assert summary["total_operations"] == 28
        assert summary["node_operations"] == 17
        assert summary["edge_operations"] == 11
        assert summary["operations_per_second"] == 14.0
        assert summary["files_per_second"] == 2.0

    def test_create_statistics_summary_zero_time(self):
        summary = UpdateUtilities.create_statistics_summary({"processing_time": 0})
        assert summary["operations_per_second"] == 0
        assert summary["files_per_second"] == 0

    def test_create_statistics_summary_empty(self):
        summary = UpdateUtilities.create_statistics_summary({})
        assert summary["total_operations"] == 0


class TestGetUpdateConfig:
    """Test the singleton factory for UpdateConfig."""

    def test_singleton(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            _cfg_mod._config_instance = None
            c1 = get_update_config()
            c2 = get_update_config()
            assert c1 is c2
            _cfg_mod._config_instance = None


# ============================================================================
# 3. ChangeDetector tests
# ============================================================================


class TestChangeDetector:
    """Tests for ChangeDetector."""

    def test_init(self):
        cd = ChangeDetector()
        assert cd.checksums_cache == {}
        assert cd.last_update_time == {}

    @pytest.mark.asyncio
    async def test_detect_changes_empty_repo(self, tmp_path):
        cd = ChangeDetector()
        changes = await cd.detect_changes(tmp_path)
        assert changes["added"] == []
        assert changes["modified"] == []
        assert changes["deleted"] == []

    @pytest.mark.asyncio
    async def test_detect_changes_new_file(self, tmp_path):
        (tmp_path / "hello.py").write_text("x = 1")
        cd = ChangeDetector()
        changes = await cd.detect_changes(tmp_path)
        assert len(changes["added"]) == 1
        assert changes["added"][0].name == "hello.py"

    @pytest.mark.asyncio
    async def test_detect_changes_deleted_file(self, tmp_path):
        cd = ChangeDetector()
        cd.checksums_cache = {"gone.py": "abc123"}
        changes = await cd.detect_changes(tmp_path)
        assert len(changes["deleted"]) == 1

    @pytest.mark.asyncio
    async def test_detect_changes_modified_file(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("x = 1")
        cd = ChangeDetector()
        # Simulate previous scan
        rel = str(f.relative_to(tmp_path))
        cd.checksums_cache[rel] = "old_checksum"
        cd.last_update_time[rel] = 0  # force mtime check
        changes = await cd.detect_changes(tmp_path)
        assert len(changes["modified"]) == 1

    def test_should_skip_directory_hidden(self):
        cd = ChangeDetector()
        assert cd._should_skip_directory(Path("/repo/.git/objects")) is True

    def test_should_skip_directory_venv(self):
        cd = ChangeDetector()
        assert cd._should_skip_directory(Path("/repo/venv/lib")) is True

    def test_should_skip_directory_normal(self):
        cd = ChangeDetector()
        assert cd._should_skip_directory(Path("/repo/src/lib")) is False

    def test_is_trackable_file(self):
        cd = ChangeDetector()
        assert cd._is_trackable_file("main.py") is True
        assert cd._is_trackable_file("main.js") is False
        assert cd._is_trackable_file("README.md") is False

    @pytest.mark.asyncio
    async def test_check_file_status_new(self, tmp_path):
        f = tmp_path / "new.py"
        f.write_text("x")
        cd = ChangeDetector()
        changes = {"added": [], "modified": [], "deleted": []}
        await cd._check_file_status(f, f.relative_to(tmp_path), changes)
        assert len(changes["added"]) == 1

    @pytest.mark.asyncio
    async def test_check_file_status_known(self, tmp_path):
        f = tmp_path / "known.py"
        f.write_text("x")
        cd = ChangeDetector()
        rel = str(f.relative_to(tmp_path))
        cd.checksums_cache[rel] = cd._calculate_file_checksum(f)
        cd.last_update_time[rel] = 0
        changes = {"added": [], "modified": [], "deleted": []}
        await cd._check_file_status(f, f.relative_to(tmp_path), changes)
        assert len(changes["added"]) == 0

    @pytest.mark.asyncio
    async def test_save_checksums(self, tmp_path):
        cd = ChangeDetector()
        cd.checksums_cache = {"test.py": "abc123"}
        await cd.save_checksums(tmp_path)
        saved = json.loads((tmp_path / ".code_graph" / "checksums.json").read_text())
        assert saved["test.py"] == "abc123"

    @pytest.mark.asyncio
    async def test_load_checksums(self, tmp_path):
        cg = tmp_path / ".code_graph"
        cg.mkdir()
        (cg / "checksums.json").write_text('{"a.py": "hash1"}')
        cd = ChangeDetector()
        await cd._load_checksums(tmp_path)
        assert cd.checksums_cache["a.py"] == "hash1"

    @pytest.mark.asyncio
    async def test_load_checksums_corrupt_json(self, tmp_path):
        cg = tmp_path / ".code_graph"
        cg.mkdir()
        (cg / "checksums.json").write_text("NOT JSON")
        cd = ChangeDetector()
        await cd._load_checksums(tmp_path)
        assert cd.checksums_cache == {}

    def test_calculate_file_checksum(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello")
        cd = ChangeDetector()
        cs = cd._calculate_file_checksum(f)
        assert cs == hashlib.md5(b"hello").hexdigest()

    def test_calculate_file_checksum_missing_file(self):
        cd = ChangeDetector()
        cs = cd._calculate_file_checksum(Path("/nonexistent"))
        assert cs.startswith("fallback_")

    def test_update_file_tracking(self, tmp_path):
        f = tmp_path / "track.py"
        f.write_text("x")
        cd = ChangeDetector()
        cd.update_file_tracking(f, tmp_path)
        rel = str(f.relative_to(tmp_path))
        assert rel in cd.checksums_cache
        assert rel in cd.last_update_time

    def test_update_file_tracking_error(self, tmp_path):
        cd = ChangeDetector()
        # Path not relative to repo_path
        cd.update_file_tracking(Path("/completely/different"), tmp_path)
        # Should not crash, just warn

    def test_get_file_status_new(self, tmp_path):
        f = tmp_path / "new.py"
        f.write_text("x")
        cd = ChangeDetector()
        assert cd.get_file_status(f, tmp_path) == "new"

    def test_get_file_status_unchanged(self, tmp_path):
        f = tmp_path / "same.py"
        f.write_text("x")
        cd = ChangeDetector()
        cd.checksums_cache[str(f.relative_to(tmp_path))] = cd._calculate_file_checksum(f)
        assert cd.get_file_status(f, tmp_path) == "unchanged"

    def test_get_file_status_modified(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("x")
        cd = ChangeDetector()
        cd.checksums_cache[str(f.relative_to(tmp_path))] = "old_hash"
        assert cd.get_file_status(f, tmp_path) == "modified"

    def test_get_file_status_error(self):
        cd = ChangeDetector()
        assert cd.get_file_status(Path("/totally/different"), Path("/repo")) == "unknown"

    def test_get_change_statistics(self):
        cd = ChangeDetector()
        changes = {
            "added": [Path("a.py"), Path("b.py")],
            "modified": [Path("c.py")],
            "deleted": [],
        }
        stats = cd.get_change_statistics(changes)
        assert stats["total_changes"] == 3
        assert stats["added_files"] == 2
        assert stats["modified_files"] == 1
        assert stats["deleted_files"] == 0

    def test_clear_cache(self):
        cd = ChangeDetector()
        cd.checksums_cache = {"a": "b"}
        cd.last_update_time = {"a": 1.0}
        cd.clear_cache()
        assert cd.checksums_cache == {}
        assert cd.last_update_time == {}


# ============================================================================
# 4. IncrementalUpdateProcessor tests
# ============================================================================


class TestIncrementalUpdateProcessor:
    """Tests for IncrementalUpdateProcessor."""

    def _make_processor(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            proc = IncrementalUpdateProcessor(
                "test_graph",
                code_processor=MagicMock(),
                file_analyzer=MagicMock(),
            )
        return proc

    @pytest.mark.asyncio
    async def test_process_file_changes_empty(self):
        proc = self._make_processor()
        changes = {"added": [], "modified": [], "deleted": []}
        with patch.object(
            proc, "_collect_affected_nodes", new_callable=AsyncMock, return_value=set()
        ):
            stats = await proc.process_file_changes(Path("/repo"), changes, "v_1")
        assert stats["files_processed"] == 0
        assert stats["version_id"] == "v_1"

    @pytest.mark.asyncio
    async def test_process_deleted_files(self):
        proc = self._make_processor()
        with patch.object(proc, "_delete_nodes_by_file", new_callable=AsyncMock, return_value=3):
            stats = {"deleted_nodes": 0, "files_processed": 0}
            await proc._process_deleted_files([Path("del.py")], stats)
            assert stats["deleted_nodes"] == 3
            assert stats["files_processed"] == 1

    @pytest.mark.asyncio
    async def test_process_deleted_files_error(self):
        proc = self._make_processor()
        with patch.object(
            proc, "_delete_nodes_by_file", new_callable=AsyncMock, side_effect=OSError("fail")
        ):
            stats = {"deleted_nodes": 0, "files_processed": 0}
            await proc._process_deleted_files([Path("err.py")], stats)
            assert stats["files_processed"] == 0

    @pytest.mark.asyncio
    async def test_process_added_files(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo", "line": 1, "related_entities": []}
        with (
            patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=[entity],
            ),
            patch.object(proc, "_add_entity_node", new_callable=AsyncMock, return_value="node_1"),
            patch.object(proc, "_add_entity_relationships", new_callable=AsyncMock, return_value=2),
        ):
            stats = {"added_nodes": 0, "added_edges": 0, "files_processed": 0}
            await proc._process_added_files(Path("/repo"), [Path("new.py")], stats)
            assert stats["added_nodes"] == 1
            assert stats["added_edges"] == 2

    @pytest.mark.asyncio
    async def test_process_added_files_node_none(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo", "related_entities": []}
        with (
            patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=[entity],
            ),
            patch.object(proc, "_add_entity_node", new_callable=AsyncMock, return_value=None),
        ):
            stats = {"added_nodes": 0, "added_edges": 0, "files_processed": 0}
            await proc._process_added_files(Path("/repo"), [Path("new.py")], stats)
            assert stats["added_nodes"] == 0

    @pytest.mark.asyncio
    async def test_process_modified_files(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "bar", "related_entities": []}
        existing_node = {"id": "function_12345678"}
        with (
            patch.object(
                proc,
                "_get_nodes_by_file",
                new_callable=AsyncMock,
                return_value=[existing_node],
            ),
            patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=[entity],
            ),
            patch.object(proc, "_generate_node_id", return_value="function_12345678"),
            patch.object(proc, "_update_entity_node", new_callable=AsyncMock),
            patch.object(
                proc,
                "_update_entity_relationships",
                new_callable=AsyncMock,
                return_value={"added": 1, "deleted": 0},
            ),
        ):
            stats = {
                "added_nodes": 0,
                "modified_nodes": 0,
                "deleted_nodes": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "files_processed": 0,
            }
            await proc._process_modified_files(Path("/repo"), [Path("mod.py")], stats)
            assert stats["modified_nodes"] == 1
            assert stats["added_edges"] == 1
            assert stats["files_processed"] == 1

    @pytest.mark.asyncio
    async def test_process_modified_files_new_entity(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "new_func", "related_entities": []}
        with (
            patch.object(proc, "_get_nodes_by_file", new_callable=AsyncMock, return_value=[]),
            patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=[entity],
            ),
            patch.object(proc, "_generate_node_id", return_value="function_new"),
            patch.object(
                proc, "_add_entity_node", new_callable=AsyncMock, return_value="function_new"
            ),
            patch.object(
                proc,
                "_update_entity_relationships",
                new_callable=AsyncMock,
                return_value={"added": 0, "deleted": 0},
            ),
        ):
            stats = {
                "added_nodes": 0,
                "modified_nodes": 0,
                "deleted_nodes": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "files_processed": 0,
            }
            await proc._process_modified_files(Path("/repo"), [Path("new.py")], stats)
            assert stats["added_nodes"] == 1

    @pytest.mark.asyncio
    async def test_process_modified_files_deleted_node(self):
        proc = self._make_processor()
        existing_node = {"id": "function_old"}
        with (
            patch.object(
                proc,
                "_get_nodes_by_file",
                new_callable=AsyncMock,
                return_value=[existing_node],
            ),
            patch.object(
                proc, "_analyze_file_for_entities", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(proc, "_delete_node", new_callable=AsyncMock),
        ):
            stats = {
                "added_nodes": 0,
                "modified_nodes": 0,
                "deleted_nodes": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "files_processed": 0,
            }
            await proc._process_modified_files(Path("/repo"), [Path("mod.py")], stats)
            assert stats["deleted_nodes"] == 1

    @pytest.mark.asyncio
    async def test_analyze_file_for_entities(self):
        proc = self._make_processor()
        mock_entity = MagicMock()
        mock_entity.type = "function"
        mock_entity.name = "foo"
        mock_entity.start_line = 1
        mock_entity.start_column = 0
        mock_entity.end_line = 5
        mock_entity.end_column = 0
        mock_entity.docstring = "doc"
        mock_entity.signature = "def foo():"
        mock_result = MagicMock()
        mock_result.entities = [mock_entity]
        proc.file_analyzer.analyze_file = AsyncMock(return_value=mock_result)
        entities = await proc._analyze_file_for_entities(Path("test.py"))
        assert len(entities) == 1
        assert entities[0]["name"] == "foo"

    @pytest.mark.asyncio
    async def test_analyze_file_for_entities_error(self):
        proc = self._make_processor()
        proc.file_analyzer.analyze_file = AsyncMock(side_effect=RuntimeError("fail"))
        entities = await proc._analyze_file_for_entities(Path("err.py"))
        assert entities == []

    @pytest.mark.asyncio
    async def test_add_entity_node(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo", "line": 1}
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            with patch.object(proc, "_add_node", new_callable=AsyncMock, return_value="func_abc"):
                nid = await proc._add_entity_node(entity, Path("test.py"))
                assert nid is not None

    @pytest.mark.asyncio
    async def test_add_entity_node_error(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo"}
        with patch.object(proc, "_add_node", new_callable=AsyncMock, side_effect=RuntimeError("x")):
            nid = await proc._add_entity_node(entity, Path("test.py"))
            assert nid is None

    @pytest.mark.asyncio
    async def test_update_entity_node(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo", "line": 1}
        with patch.object(proc, "_update_node", new_callable=AsyncMock) as mock_update:
            await proc._update_entity_node(entity, "node_1")
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_entity_relationships(self):
        proc = self._make_processor()
        entity = {
            "type": "function",
            "name": "foo",
            "related_entities": [
                {"type": "class", "name": "Bar", "relationship": "calls"},
                {"type": "function", "name": "baz", "file": "other.py", "relationship": "imports"},
            ],
        }
        with patch.object(proc, "_add_edge", new_callable=AsyncMock):
            count = await proc._add_entity_relationships(entity, "src_id", Path("test.py"))
            assert count == 2

    @pytest.mark.asyncio
    async def test_add_entity_relationships_none(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo"}
        count = await proc._add_entity_relationships(entity, "src_id", Path("t.py"))
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_entity_relationships(self):
        proc = self._make_processor()
        entity = {
            "related_entities": [
                {"type": "class", "name": "C", "relationship": "calls"},
            ],
        }
        existing_edges = [
            {"target_id": "old_target", "relationship_type": "imports"},
        ]
        with (
            patch.object(
                proc,
                "_get_edges_for_node",
                new_callable=AsyncMock,
                return_value=existing_edges,
            ),
            patch.object(proc, "_add_edge", new_callable=AsyncMock),
            patch.object(proc, "_delete_edge", new_callable=AsyncMock),
        ):
            result = await proc._update_entity_relationships(entity, "n1", Path("t.py"))
            assert result["added"] == 1
            assert result["deleted"] == 1

    @pytest.mark.asyncio
    async def test_update_entity_relationships_error(self):
        proc = self._make_processor()
        entity = {"related_entities": []}
        with patch.object(
            proc,
            "_get_edges_for_node",
            new_callable=AsyncMock,
            side_effect=RuntimeError("fail"),
        ):
            result = await proc._update_entity_relationships(entity, "n1", Path("t.py"))
            assert result == {"added": 0, "deleted": 0}

    def test_generate_node_id(self):
        proc = self._make_processor()
        nid = proc._generate_node_id("function", "foo", "test.py")
        assert nid.startswith("function_")
        assert len(nid) == len("function_") + 8

    @pytest.mark.asyncio
    async def test_add_node_db(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            result = await proc._add_node("n1", "function", "foo", {"line": 1})
            assert result == "n1"
            mock_conn.conn.execute.assert_called_once()
            mock_conn.conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_node_db(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        # JSONB properties come back from psycopg as a dict (no json.loads)
        cursor = _make_cursor(rows=[({"line": 1},)])
        mock_conn.conn.execute.return_value = cursor
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._update_node("n1", {"line": 2})
            assert mock_conn.conn.execute.await_count == 2  # SELECT + UPDATE

    @pytest.mark.asyncio
    async def test_update_node_db_not_found(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rows=None)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._update_node("n1", {"line": 2})
            assert mock_conn.conn.execute.await_count == 1  # only SELECT

    # Deleted test_update_node_db_bad_json: JSONB returns a dict, so there is no
    # json-decode path on the existing-properties read anymore.

    @pytest.mark.asyncio
    async def test_delete_node_db(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._delete_node("n1")
            assert mock_conn.conn.execute.await_count == 2  # edges + nodes
            mock_conn.conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_edge_db(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._add_edge("s1", "t1", "calls", {})
            mock_conn.conn.execute.assert_awaited_once()
            mock_conn.conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_edge_db(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._delete_edge("s1", "t1", "calls")
            mock_conn.conn.execute.assert_awaited_once()
            mock_conn.conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_nodes_by_file(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        # row[3] (properties) is a JSONB dict already
        rows = [("n1", "function", "foo", {"file": "test.py"})]
        cursor = _make_cursor(rows=rows)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            nodes = await proc._get_nodes_by_file("test.py")
            assert len(nodes) == 1
            assert nodes[0]["id"] == "n1"

    # Deleted test_get_nodes_by_file_bad_json: source no longer json.loads the
    # JSONB properties column, so there is no bad-json branch.

    @pytest.mark.asyncio
    async def test_get_edges_for_node(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        rows = [("s1", "t1", "calls", {"weight": 1})]
        cursor = _make_cursor(rows=rows)
        mock_conn.conn.execute.return_value = cursor
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            edges = await proc._get_edges_for_node("s1")
            assert len(edges) == 1
            assert edges[0]["source_id"] == "s1"

    @pytest.mark.asyncio
    async def test_delete_nodes_by_file(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        cursor_select = _make_cursor(rows=[("n1",), ("n2",)])
        mock_conn.conn.execute.return_value = cursor_select
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            count = await proc._delete_nodes_by_file("test.py")
            assert count == 2
            mock_conn.conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_nodes_by_file_none(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        cursor = _make_cursor(rows=[])
        mock_conn.conn.execute.return_value = cursor
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            count = await proc._delete_nodes_by_file("test.py")
            assert count == 0

    @pytest.mark.asyncio
    async def test_get_related_nodes_empty(self):
        proc = self._make_processor()
        result = await proc._get_related_nodes([])
        assert result == set()

    @pytest.mark.asyncio
    async def test_get_related_nodes(self):
        proc = self._make_processor()
        mock_conn = _make_mock_db_conn()
        cursor1 = _make_cursor(rows=[("rel1",)])
        cursor2 = _make_cursor(rows=[("rel2",)])
        mock_conn.conn.execute.side_effect = [cursor1, cursor2]
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            related = await proc._get_related_nodes([{"id": "n1"}])
            assert "rel1" in related
            assert "rel2" in related

    @pytest.mark.asyncio
    async def test_collect_affected_nodes_added(self):
        proc = self._make_processor()
        entity = {"type": "function", "name": "foo"}
        with (
            patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=[entity],
            ),
            patch.object(proc, "_generate_node_id", return_value="func_123"),
        ):
            changes = {"added": [Path("new.py")], "modified": [], "deleted": []}
            affected = await proc._collect_affected_nodes(changes)
            assert "func_123" in affected

    @pytest.mark.asyncio
    async def test_collect_affected_nodes_deleted(self):
        proc = self._make_processor()
        with (
            patch.object(
                proc,
                "_get_nodes_by_file",
                new_callable=AsyncMock,
                return_value=[{"id": "del_node"}],
            ),
            patch.object(
                proc, "_get_related_nodes", new_callable=AsyncMock, return_value={"rel_1"}
            ),
        ):
            changes = {"added": [], "modified": [], "deleted": [Path("gone.py")]}
            affected = await proc._collect_affected_nodes(changes)
            assert "del_node" in affected
            assert "rel_1" in affected


# ============================================================================
# 5. IncrementalUpdater tests
# ============================================================================


class TestIncrementalUpdater:
    """Tests for IncrementalUpdater (orchestrator)."""

    def _make_updater(self):
        with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
            updater = IncrementalUpdater(
                "test_graph",
                code_processor=MagicMock(),
                file_analyzer=MagicMock(),
            )
        return updater

    def test_init(self):
        u = self._make_updater()
        assert u.code_graph_id == "test_graph"
        assert u.current_version_id is None

    @pytest.mark.asyncio
    async def test_detect_changes_invalid_path(self):
        u = self._make_updater()
        with pytest.raises(ValueError, match="Invalid repository path"):
            await u.detect_changes(Path("/nonexistent/path"))

    @pytest.mark.asyncio
    async def test_detect_changes_valid(self, tmp_path):
        u = self._make_updater()
        expected = {"added": [], "modified": [], "deleted": []}
        u.change_detector.detect_changes = AsyncMock(return_value=expected)
        result = await u.detect_changes(tmp_path)
        assert result == expected

    @pytest.mark.asyncio
    async def test_update_graph_no_changes(self):
        u = self._make_updater()
        changes = {"added": [], "modified": [], "deleted": []}
        u.version_manager.create_version = AsyncMock(return_value="v_no_change")
        vid, stats = await u.update_graph(Path("/repo"), changes=changes)
        assert vid == "v_no_change"
        assert stats["files_processed"] == 0

    @pytest.mark.asyncio
    async def test_update_graph_with_changes(self):
        u = self._make_updater()
        changes = {"added": [Path("new.py")], "modified": [], "deleted": []}
        u.config.get = MagicMock(return_value=False)  # disable validation/backup
        u.version_manager.create_version = AsyncMock(return_value="v_1")
        u.processor.process_file_changes = AsyncMock(
            return_value={"files_processed": 1, "added_nodes": 1}
        )
        u.change_detector.save_checksums = AsyncMock()
        vid, stats = await u.update_graph(Path("/repo"), changes=changes)
        assert vid == "v_1"
        assert stats["files_processed"] == 1

    @pytest.mark.asyncio
    async def test_update_graph_detects_changes_if_none(self, tmp_path):
        u = self._make_updater()
        u.change_detector.detect_changes = AsyncMock(
            return_value={"added": [], "modified": [], "deleted": []}
        )
        u.version_manager.create_version = AsyncMock(return_value="v_auto")
        vid, stats = await u.update_graph(tmp_path)
        assert vid == "v_auto"
        u.change_detector.detect_changes.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_graph_validation_failure(self):
        u = self._make_updater()
        changes = {"added": [Path("x.py")], "modified": [], "deleted": []}
        u.config.get = MagicMock(
            side_effect=lambda k, default=None: {
                "validation_settings.pre_update_checks": True,
                "recovery_settings.auto_backup_before_update": False,
                "validation_settings.consistency_checks": False,
                "validation_settings.post_update_verification": False,
                "recovery_settings.auto_recovery_enabled": False,
            }.get(k, default)
        )
        u.validator.validate_pre_update = AsyncMock(
            return_value={"valid": False, "errors": ["bad"], "warnings": []}
        )
        with pytest.raises(ValueError, match="Pre-update validation failed"):
            await u.update_graph(Path("/repo"), changes=changes)

    @pytest.mark.asyncio
    async def test_compare_versions(self):
        u = self._make_updater()
        u.validator.compare_versions = AsyncMock(return_value={"valid": True, "diffs": []})
        result = await u.compare_versions("v1", "v2")
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_rollback_to_version_success(self):
        u = self._make_updater()
        u.recovery_manager.rollback_to_version = AsyncMock(return_value=True)
        assert await u.rollback_to_version("v_1") is True

    @pytest.mark.asyncio
    async def test_rollback_to_version_failure(self):
        u = self._make_updater()
        u.recovery_manager.rollback_to_version = AsyncMock(return_value=False)
        assert await u.rollback_to_version("v_1") is False

    @pytest.mark.asyncio
    async def test_rollback_to_version_exception(self):
        u = self._make_updater()
        u.recovery_manager.rollback_to_version = AsyncMock(side_effect=RuntimeError("x"))
        assert await u.rollback_to_version("v_1") is False

    @pytest.mark.asyncio
    async def test_get_database_stats(self):
        u = self._make_updater()
        u.db_utilities.get_database_stats = AsyncMock(
            return_value={"node_count": 5, "edge_count": 3}
        )
        stats = await u.get_database_stats()
        assert stats["node_count"] == 5
        assert "configuration" in stats

    @pytest.mark.asyncio
    async def test_get_database_stats_error(self):
        u = self._make_updater()
        u.db_utilities.get_database_stats = AsyncMock(side_effect=RuntimeError("fail"))
        stats = await u.get_database_stats()
        assert "error" in stats

    @pytest.mark.asyncio
    async def test_vacuum_database(self):
        u = self._make_updater()
        u.db_utilities.vacuum_database = AsyncMock(
            return_value={"success": True, "space_reclaimed": 100}
        )
        result = await u.vacuum_database()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_vacuum_database_error(self):
        u = self._make_updater()
        u.db_utilities.vacuum_database = AsyncMock(side_effect=RuntimeError("fail"))
        result = await u.vacuum_database()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_list_backups(self):
        u = self._make_updater()
        u.recovery_manager.list_backups = AsyncMock(return_value=[{"id": "b1"}])
        backups = await u.list_backups()
        assert len(backups) == 1

    @pytest.mark.asyncio
    async def test_list_backups_error(self):
        u = self._make_updater()
        u.recovery_manager.list_backups = AsyncMock(side_effect=RuntimeError("fail"))
        assert await u.list_backups() == []

    @pytest.mark.asyncio
    async def test_restore_from_backup(self):
        u = self._make_updater()
        u.recovery_manager.restore_from_backup = AsyncMock(return_value=True)
        assert await u.restore_from_backup("b1") is True

    @pytest.mark.asyncio
    async def test_restore_from_backup_failure(self):
        u = self._make_updater()
        u.recovery_manager.restore_from_backup = AsyncMock(return_value=False)
        assert await u.restore_from_backup("b1") is False

    @pytest.mark.asyncio
    async def test_restore_from_backup_exception(self):
        u = self._make_updater()
        u.recovery_manager.restore_from_backup = AsyncMock(side_effect=RuntimeError("x"))
        assert await u.restore_from_backup("b1") is False

    @pytest.mark.asyncio
    async def test_get_recovery_log(self):
        u = self._make_updater()
        u.recovery_manager.get_recovery_log = AsyncMock(return_value=[{"op": "backup"}])
        log = await u.get_recovery_log()
        assert len(log) == 1

    @pytest.mark.asyncio
    async def test_get_recovery_log_error(self):
        u = self._make_updater()
        u.recovery_manager.get_recovery_log = AsyncMock(side_effect=RuntimeError("x"))
        assert await u.get_recovery_log() == []

    @pytest.mark.asyncio
    async def test_analyze_performance(self):
        u = self._make_updater()
        u.db_utilities.analyze_database = AsyncMock(
            return_value={"statistics": {"node_count": 5000}, "recommendations": []}
        )
        result = await u.analyze_performance()
        assert "database_analysis" in result
        assert "configuration_validation" in result

    @pytest.mark.asyncio
    async def test_analyze_performance_error(self):
        u = self._make_updater()
        u.db_utilities.analyze_database = AsyncMock(side_effect=RuntimeError("x"))
        result = await u.analyze_performance()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_update_failure_auto_recovery_disabled(self):
        u = self._make_updater()
        u.config.get = MagicMock(return_value=False)
        await u._handle_update_failure(["err"])
        # Should not attempt recovery

    @pytest.mark.asyncio
    async def test_handle_update_failure_with_version(self):
        u = self._make_updater()
        u.config.get = MagicMock(return_value=True)
        u.current_version_id = "v_fail"
        u.recovery_manager.recover_from_failed_update = AsyncMock(return_value={"success": True})
        await u._handle_update_failure(["err"])
        u.recovery_manager.recover_from_failed_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_update_failure_recovery_fails(self):
        u = self._make_updater()
        u.config.get = MagicMock(return_value=True)
        u.current_version_id = "v_fail"
        u.recovery_manager.recover_from_failed_update = AsyncMock(
            side_effect=RuntimeError("recovery failed")
        )
        await u._handle_update_failure(["err"])
        # Should not raise

    def test_backward_compat_calculate_file_checksum(self, tmp_path):
        u = self._make_updater()
        f = tmp_path / "test.py"
        f.write_text("hello")
        cs = u._calculate_file_checksum(f)
        expected = hashlib.md5(b"hello").hexdigest()
        assert cs == expected

    def test_backward_compat_generate_node_id(self):
        u = self._make_updater()
        nid = u._generate_node_id("function", "foo", "test.py")
        assert nid.startswith("function_")

    @pytest.mark.asyncio
    async def test_backward_compat_create_version(self):
        u = self._make_updater()
        u.version_manager.create_version = AsyncMock(return_value="v_back")
        vid = await u._create_version("desc")
        assert vid == "v_back"

    @pytest.mark.asyncio
    async def test_backward_compat_get_version(self):
        u = self._make_updater()
        u.version_manager.get_version = AsyncMock(return_value={"id": "v_1"})
        v = await u._get_version("v_1")
        assert v["id"] == "v_1"

    @pytest.mark.asyncio
    async def test_backward_compat_delegations(self):
        u = self._make_updater()
        u.processor._get_nodes_by_file = AsyncMock(return_value=[])
        u.processor._add_node = AsyncMock(return_value="n1")
        u.processor._update_node = AsyncMock()
        u.processor._add_edge = AsyncMock()
        u.processor._delete_node = AsyncMock()
        u.processor._delete_edge = AsyncMock()
        u.processor._get_edges_for_node = AsyncMock(return_value=[])
        u.processor._delete_nodes_by_file = AsyncMock(return_value=0)
        u.processor._get_related_nodes = AsyncMock(return_value=set())

        assert await u._get_nodes_by_file("f") == []
        assert await u._add_node("n1", "t", "n", {}) == "n1"
        await u._update_node("n1", {})
        await u._add_edge("s", "t", "r", {})
        await u._delete_node("n1")
        await u._delete_edge("s", "t", "r")
        assert await u._get_edges_for_node("n1") == []
        assert await u._delete_nodes_by_file("f") == 0
        assert await u._get_related_nodes([]) == set()
