"""Unit tests for services/code_analysis/pattern_search_service.py.

Covers:
- PatternSearchService.search_patterns (all branches)
- PatternSearchService.search_code_patterns_utility
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from services.code_analysis.pattern_search_service import PatternSearchService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_service(rows=None, conn_none=False, connect_fails=False):
    """Create a PatternSearchService with mocked dependencies."""
    analyzer = MagicMock()
    db_conn = MagicMock()
    processor_factory = MagicMock()

    if conn_none:
        db_conn.conn = None
        if connect_fails:
            db_conn.connect = AsyncMock()
            # After connect, conn is still None
        else:

            async def _connect():
                db_conn.conn = MagicMock()
                cursor = AsyncMock()
                cursor.fetchall = AsyncMock(return_value=rows or [])
                db_conn.conn.execute = AsyncMock(return_value=cursor)

            db_conn.connect = AsyncMock(side_effect=_connect)
    else:
        conn = MagicMock()
        cursor = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=rows or [])
        conn.execute = AsyncMock(return_value=cursor)
        db_conn.conn = conn

    svc = PatternSearchService(analyzer, db_conn, processor_factory)
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchPatterns:
    """Tests for search_patterns method."""

    async def test_search_matches_by_name(self):
        """Should find matches based on name property."""
        # JSONB columns come back as already-decoded dicts (no json.loads in source).
        props = {"name": "my_function", "file_path": "main.py"}
        rows = [("node1", "function", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "my_func"

        result = await svc.search_patterns("cg1", request)

        assert result.count >= 1
        assert result.code_graph_id == "cg1"
        assert any(m["method"] == "name_search" for m in result.matches)

    async def test_search_matches_by_code_snippet(self):
        """Should find matches based on code_snippet property."""
        props = {
            "code_snippet": "def my_function():\n    return 42",
            "file_path": "main.py",
        }
        rows = [("node1", "function", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "my_function"

        result = await svc.search_patterns("cg1", request)

        assert result.count >= 1
        assert any(m["method"] == "property_search" for m in result.matches)

    async def test_search_case_insensitive(self):
        """Pattern matching should be case-insensitive."""
        props = {"name": "MyClass", "file_path": "model.py"}
        rows = [("node1", "class", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "myclass"

        result = await svc.search_patterns("cg1", request)

        assert result.count >= 1

    async def test_search_no_matches(self):
        """Should return empty result when no matches found."""
        props = {"name": "unrelated", "file_path": "other.py"}
        rows = [("node1", "function", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "nonexistent_pattern"

        result = await svc.search_patterns("cg1", request)

        assert result.count == 0
        assert result.matches == []

    async def test_search_empty_db(self):
        """Should return empty result for empty database."""
        svc = _make_service(rows=[])

        request = MagicMock()
        request.pattern = "anything"

        result = await svc.search_patterns("cg1", request)

        assert result.count == 0
        assert result.matches == []

    # test_search_json_decode_error removed: properties is now a JSONB column
    # decoded to a dict by psycopg; the source no longer json.loads() a string,
    # so a malformed-JSON-string branch no longer exists to exercise.

    async def test_search_null_properties(self):
        """Should handle None/null properties."""
        rows = [("node1", "function", None)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "test"

        result = await svc.search_patterns("cg1", request)

        assert result.count == 0

    async def test_search_long_code_snippet_truncated(self):
        """Long code snippets should be truncated in match_text."""
        long_code = "x" * 300
        props = {
            "code_snippet": long_code,
            "file_path": "long.py",
        }
        rows = [("node1", "function", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "x"

        result = await svc.search_patterns("cg1", request)

        match = [m for m in result.matches if m["method"] == "property_search"]
        assert len(match) > 0
        assert match[0]["match_text"].endswith("...")
        assert len(match[0]["match_text"]) <= 204  # 200 + "..."

    async def test_search_conn_none_triggers_connect(self):
        """Should call connect() when conn is None."""
        props = {"name": "found", "file_path": "f.py"}
        rows = [("n1", "function", props)]
        svc = _make_service(rows=rows, conn_none=True, connect_fails=False)

        request = MagicMock()
        request.pattern = "found"

        result = await svc.search_patterns("cg1", request)

        assert result.count >= 1

    async def test_search_conn_none_connect_fails(self):
        """Should return error result when connect() fails to establish connection."""
        svc = _make_service(conn_none=True, connect_fails=True)

        request = MagicMock()
        request.pattern = "test"

        result = await svc.search_patterns("cg1", request)

        assert result.error is not None

    async def test_search_db_exception(self):
        """Should return error result on database exception."""
        analyzer = MagicMock()
        db_conn = MagicMock()
        db_conn.conn = MagicMock()
        db_conn.conn.execute = AsyncMock(side_effect=RuntimeError("DB crash"))
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        request = MagicMock()
        request.pattern = "test"

        result = await svc.search_patterns("cg1", request)

        assert result.error is not None
        assert result.count == 0

    async def test_search_both_name_and_snippet_match(self):
        """Should count both name and snippet matches separately."""
        props = {
            "name": "test_func",
            "code_snippet": "def test_func(): pass",
            "file_path": "test.py",
        }
        rows = [("node1", "function", props)]
        svc = _make_service(rows=rows)

        request = MagicMock()
        request.pattern = "test_func"

        result = await svc.search_patterns("cg1", request)

        assert result.count == 2  # Both name_search and property_search


class TestSearchCodePatternsUtility:
    """Tests for search_code_patterns_utility."""

    async def test_utility_returns_patterns(self):
        """Should return patterns from analyzer."""
        analyzer = MagicMock()
        analyzer.find_pattern_in_file = MagicMock(return_value=[{"line": 1, "text": "match"}])
        db_conn = MagicMock()
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        result = await svc.search_code_patterns_utility("pattern", "/tmp/file.py")

        assert result["count"] == 1
        assert result["patterns"] == [{"line": 1, "text": "match"}]

    async def test_utility_returns_empty(self):
        """Should return empty when no patterns found."""
        analyzer = MagicMock()
        analyzer.find_pattern_in_file = MagicMock(return_value=[])
        db_conn = MagicMock()
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        result = await svc.search_code_patterns_utility("nothing", "/tmp/file.py")

        assert result["count"] == 0

    async def test_utility_with_language(self):
        """Should pass language to analyzer."""
        analyzer = MagicMock()
        analyzer.find_pattern_in_file = MagicMock(return_value=[])
        db_conn = MagicMock()
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        await svc.search_code_patterns_utility("pat", "/tmp/f.py", language="python")

        analyzer.find_pattern_in_file.assert_called_once_with("/tmp/f.py", "pat", "python")

    async def test_utility_error_handling(self):
        """Should return error on exception."""
        analyzer = MagicMock()
        analyzer.find_pattern_in_file = MagicMock(side_effect=RuntimeError("fail"))
        db_conn = MagicMock()
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        result = await svc.search_code_patterns_utility("pat", "/tmp/f.py")

        assert "error" in result
