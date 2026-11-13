"""Unit tests for services.incremental.incremental_update_validator module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Clean polluted modules before importing.
# test_incremental_big_modules.py stubs
# "services.incremental.incremental_update_validator" as a bare ModuleType
# (with UpdateValidator = MagicMock but no 'settings' attribute) during
# collection. Purge it so we import the real module.
# ---------------------------------------------------------------------------
for _key in list(sys.modules):
    if _key == "services.incremental.incremental_update_validator":
        del sys.modules[_key]

# Also ensure "services" / "services.incremental" are proper packages
_src_services = str(Path(__file__).resolve().parents[2] / "src" / "services")
if "services" in sys.modules:
    _svc = sys.modules["services"]
    if not getattr(_svc, "__path__", None):
        _svc.__path__ = [_src_services]
    elif _src_services not in _svc.__path__:
        _svc.__path__.insert(0, _src_services)

_src_incr = str(Path(__file__).resolve().parents[2] / "src" / "services" / "incremental")
if "services.incremental" in sys.modules:
    _si = sys.modules["services.incremental"]
    if not getattr(_si, "__path__", None):
        _si.__path__ = [_src_incr]
    elif _src_incr not in _si.__path__:
        _si.__path__.insert(0, _src_incr)

# ---------------------------------------------------------------------------

from services.incremental.incremental_update_validator import UpdateValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE_PATH = "services.incremental.incremental_update_validator"


def _async_cursor(rows, rowcount=0):
    """Build a fake psycopg async cursor.

    The postgres source ``await``s ``execute`` then ``await``s
    ``cursor.fetchone()`` / ``fetchall()``, reads ``cursor.rowcount`` (sync),
    and iterates ``async for row in cursor`` (snapshot paths). Rows are tuples
    (psycopg positional access ``row[0]``); JSONB columns are dicts already.
    """
    rows = list(rows)
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


def _make_facade_conn():
    """A bare GraphDatabaseConnection facade mock with awaitable connect/close.

    Used where the SQL itself is bypassed (e.g. ``_get_version_snapshot`` is
    patched) but the source still ``await``s ``db_conn.connect()/.close()``.
    """
    db_conn = MagicMock()
    db_conn.connect = AsyncMock()
    db_conn.close = AsyncMock()
    return db_conn


def _make_mock_db_conn(
    node_count=5,
    edge_count=3,
    invalid_edges=None,
    orphaned_nodes=None,
    duplicate_groups=None,
    self_ref_edges=None,
    version_row=None,
    nodes_rows=None,
    edges_rows=None,
):
    """Build a mock GraphDatabaseConnection facade whose ``conn.execute`` is an
    awaitable that routes to pre-set async cursors based on the SQL query."""
    db_conn = MagicMock()
    db_conn.connect = AsyncMock()
    db_conn.close = AsyncMock()
    conn = db_conn.conn

    async def execute_side_effect(sql, params=None):
        query = sql.strip().upper()

        # Duplicate nodes (must be checked before generic COUNT(*) FROM NODES)
        if "GROUP BY" in query and "HAVING COUNT" in query:
            return _async_cursor(duplicate_groups or [])

        # COUNT(*) FROM nodes
        if "COUNT(*)" in query and "FROM NODES" in query:
            return _async_cursor([(node_count,)])

        # COUNT(*) FROM edges
        if "COUNT(*)" in query and "FROM EDGES" in query:
            return _async_cursor([(edge_count,)])

        # Invalid edges (node-edge consistency)
        if "LEFT JOIN NODES N1" in query and "LEFT JOIN NODES N2" in query:
            return _async_cursor(invalid_edges or [])

        # Orphaned nodes
        if "LEFT JOIN EDGES E1" in query and "LEFT JOIN EDGES E2" in query:
            return _async_cursor(orphaned_nodes or [])

        # Self-referencing edges (postgres puts code_graph_id before source=target)
        if "SOURCE = TARGET" in query:
            return _async_cursor(self_ref_edges or [])

        # Version lookup
        if "GRAPH_VERSIONS" in query:
            return _async_cursor([version_row] if version_row else [])

        # Nodes query (for snapshot)
        if "FROM NODES" in query and "SELECT ID" in query.replace(",", ""):
            return _async_cursor(nodes_rows or [])

        # Edges query (for snapshot)
        if "FROM EDGES" in query and "SOURCE" in query:
            return _async_cursor(edges_rows or [])

        # fallback
        return _async_cursor([])

    conn.execute = AsyncMock(side_effect=execute_side_effect)
    return db_conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    """Create an UpdateValidator with patched settings."""
    with patch(f"{MODULE_PATH}.settings") as mock_settings:
        mock_settings.CODE_GRAPH_DATABASE_DIR = "/tmp/test_graphs"
        v = UpdateValidator("test-graph-id")
        yield v


@pytest.fixture
def mock_db(validator):
    """Patch _get_db_connection to return a controllable mock."""
    db_conn = _make_mock_db_conn()
    with patch.object(validator, "_get_db_connection", return_value=db_conn):
        yield db_conn


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------


class TestUpdateValidatorInit:
    def test_constructor_stores_code_graph_id(self):
        with patch(f"{MODULE_PATH}.settings"):
            v = UpdateValidator("my-id")
        assert v.code_graph_id == "my-id"


# ---------------------------------------------------------------------------
# Tests: _validate_database_connection
# ---------------------------------------------------------------------------


class TestValidateDatabaseConnection:
    async def test_success(self, validator):
        db_conn = _make_mock_db_conn(node_count=10, edge_count=7)
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_database_connection()
        assert result["valid"] is True
        assert result["errors"] == []
        # source closes the facade (db_conn.close), not the raw conn
        db_conn.close.assert_called_once()

    async def test_failure(self, validator):
        with patch.object(validator, "_get_db_connection", side_effect=Exception("db down")):
            result = await validator._validate_database_connection()
        assert result["valid"] is False
        assert any("Database connection failed" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Tests: _validate_file_access
# ---------------------------------------------------------------------------


class TestValidateFileAccess:
    async def test_added_files_exist(self, validator, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("pass")
        changes = {"added": [f]}
        result = await validator._validate_file_access(changes)
        assert result["valid"] is True

    async def test_added_file_missing(self, validator, tmp_path):
        missing = tmp_path / "nope.py"
        changes = {"added": [missing]}
        result = await validator._validate_file_access(changes)
        assert result["valid"] is False
        assert any("File not found" in e for e in result["errors"])

    async def test_deleted_file_still_exists_gives_warning(self, validator, tmp_path):
        f = tmp_path / "old.py"
        f.write_text("pass")
        changes = {"deleted": [f]}
        result = await validator._validate_file_access(changes)
        assert result["valid"] is True
        assert len(result["warnings"]) == 1

    async def test_deleted_file_gone_ok(self, validator, tmp_path):
        gone = tmp_path / "gone.py"
        changes = {"deleted": [gone]}
        result = await validator._validate_file_access(changes)
        assert result["valid"] is True
        assert result["warnings"] == []

    async def test_empty_changes(self, validator):
        result = await validator._validate_file_access({})
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Tests: _validate_repository_state
# ---------------------------------------------------------------------------


class TestValidateRepositoryState:
    async def test_valid_repo_with_code_graph_dir(self, validator, tmp_path):
        (tmp_path / ".code_graph").mkdir()
        result = await validator._validate_repository_state(tmp_path)
        assert result["valid"] is True
        assert result["warnings"] == []

    async def test_valid_repo_without_code_graph_dir(self, validator, tmp_path):
        result = await validator._validate_repository_state(tmp_path)
        assert result["valid"] is True
        assert len(result["warnings"]) == 1
        assert ".code_graph" in result["warnings"][0]

    async def test_repo_path_does_not_exist(self, validator, tmp_path):
        missing = tmp_path / "nope"
        result = await validator._validate_repository_state(missing)
        assert result["valid"] is False

    async def test_repo_path_is_a_file(self, validator, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = await validator._validate_repository_state(f)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: _validate_change_conflicts
# ---------------------------------------------------------------------------


class TestValidateChangeConflicts:
    async def test_no_conflicts(self, validator):
        changes = {
            "added": [Path("/a.py")],
            "modified": [Path("/b.py")],
            "deleted": [Path("/c.py")],
        }
        result = await validator._validate_change_conflicts(changes)
        assert result["valid"] is True

    async def test_file_in_two_categories(self, validator):
        dup = Path("/dup.py")
        changes = {"added": [dup], "modified": [dup]}
        result = await validator._validate_change_conflicts(changes)
        assert result["valid"] is False
        assert len(result["errors"]) == 1

    async def test_empty_changes(self, validator):
        result = await validator._validate_change_conflicts({})
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Tests: _validate_node_edge_consistency
# ---------------------------------------------------------------------------


class TestValidateNodeEdgeConsistency:
    async def test_consistent(self, validator, mock_db):
        result = await validator._validate_node_edge_consistency()
        assert result["valid"] is True

    async def test_invalid_edges_detected(self, validator):
        db_conn = _make_mock_db_conn(
            invalid_edges=[("src1", "tgt1", "calls"), ("src2", "tgt2", "imports")]
        )
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_node_edge_consistency()
        assert result["valid"] is False
        assert len(result["errors"]) == 2

    async def test_db_error(self, validator):
        db_conn = MagicMock()
        db_conn.connect = AsyncMock()
        db_conn.close = AsyncMock()
        db_conn.conn.execute = AsyncMock(side_effect=Exception("query fail"))
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_node_edge_consistency()
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: _validate_orphaned_nodes
# ---------------------------------------------------------------------------


class TestValidateOrphanedNodes:
    async def test_no_orphans(self, validator, mock_db):
        result = await validator._validate_orphaned_nodes()
        assert result["valid"] is True
        assert result["warnings"] == []

    async def test_orphans_detected(self, validator):
        db_conn = _make_mock_db_conn(orphaned_nodes=[("id1", "orphan_func", "function")])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_orphaned_nodes()
        assert len(result["warnings"]) == 1
        assert "orphan_func" in result["warnings"][0]


# ---------------------------------------------------------------------------
# Tests: _validate_duplicate_nodes
# ---------------------------------------------------------------------------


class TestValidateDuplicateNodes:
    async def test_no_duplicates(self, validator, mock_db):
        result = await validator._validate_duplicate_nodes()
        assert result["valid"] is True

    async def test_duplicates_detected(self, validator):
        db_conn = _make_mock_db_conn(duplicate_groups=[("id1", "MyClass", "class", 3)])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_duplicate_nodes()
        assert result["valid"] is False
        assert "MyClass" in result["errors"][0]


# ---------------------------------------------------------------------------
# Tests: _validate_processing_statistics
# ---------------------------------------------------------------------------


class TestValidateProcessingStatistics:
    async def test_valid_stats(self, validator):
        stats = {"files_processed": 10, "processing_time": 5.0}
        result = await validator._validate_processing_statistics(stats)
        assert result["valid"] is True
        assert result["warnings"] == []

    async def test_negative_value_warning(self, validator):
        stats = {"files_processed": -1}
        result = await validator._validate_processing_statistics(stats)
        assert len(result["warnings"]) == 1

    async def test_long_processing_time_warning(self, validator):
        stats = {"processing_time": 7200}
        result = await validator._validate_processing_statistics(stats)
        assert any("long processing" in w.lower() for w in result["warnings"])


# ---------------------------------------------------------------------------
# Tests: _validate_relationship_integrity
# ---------------------------------------------------------------------------


class TestValidateRelationshipIntegrity:
    async def test_no_self_refs(self, validator, mock_db):
        result = await validator._validate_relationship_integrity()
        assert result["valid"] is True

    async def test_valid_self_ref_calls(self, validator):
        db_conn = _make_mock_db_conn(self_ref_edges=[("n1", "n1", "calls")])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_relationship_integrity()
        # "calls" self-reference is allowed
        assert result["valid"] is True
        assert result["errors"] == []

    async def test_invalid_self_ref(self, validator):
        db_conn = _make_mock_db_conn(self_ref_edges=[("n1", "n1", "inherits")])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_relationship_integrity()
        assert len(result["errors"]) == 1
        assert "self-reference" in result["errors"][0].lower()

    async def test_db_error(self, validator):
        db_conn = MagicMock()
        db_conn.connect = AsyncMock()
        db_conn.close = AsyncMock()
        db_conn.conn.execute = AsyncMock(side_effect=Exception("boom"))
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_relationship_integrity()
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: _generate_performance_metrics
# ---------------------------------------------------------------------------


class TestGeneratePerformanceMetrics:
    async def test_basic_metrics(self, validator):
        stats = {
            "processing_time": 10,
            "files_processed": 5,
            "added_nodes": 3,
            "modified_nodes": 2,
            "deleted_nodes": 1,
            "added_edges": 4,
            "deleted_edges": 0,
        }
        metrics = await validator._generate_performance_metrics(stats)
        assert metrics["avg_time_per_file"] == 2.0
        assert metrics["total_operations"] == 10
        assert metrics["operations_per_second"] == 1.0

    async def test_zero_files(self, validator):
        stats = {"processing_time": 5, "files_processed": 0}
        metrics = await validator._generate_performance_metrics(stats)
        assert metrics["avg_time_per_file"] == 0

    async def test_no_operations(self, validator):
        stats = {"processing_time": 5, "files_processed": 1}
        metrics = await validator._generate_performance_metrics(stats)
        assert metrics["total_operations"] == 0
        assert metrics["operations_per_second"] == 0

    async def test_empty_stats(self, validator):
        metrics = await validator._generate_performance_metrics({})
        assert metrics["total_operations"] == 0
        assert metrics["operations_per_second"] == 0


# ---------------------------------------------------------------------------
# Tests: _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    async def test_version_found(self, validator):
        db_conn = _make_mock_db_conn(version_row=("v1", "Initial version", "2024-01-01T00:00:00"))
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._get_version("v1")
        assert result is not None
        assert result["id"] == "v1"
        assert result["description"] == "Initial version"

    async def test_version_not_found(self, validator):
        db_conn = _make_mock_db_conn(version_row=None)
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._get_version("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _nodes_differ
# ---------------------------------------------------------------------------


class TestNodesDiffer:
    def test_identical_nodes(self, validator):
        n = {"name": "foo", "type": "function", "file_path": "/a.py", "properties": {}}
        assert validator._nodes_differ(n, n) is False

    def test_name_changed(self, validator):
        n1 = {"name": "foo", "type": "function", "file_path": "/a.py", "properties": {}}
        n2 = {"name": "bar", "type": "function", "file_path": "/a.py", "properties": {}}
        assert validator._nodes_differ(n1, n2) is True

    def test_type_changed(self, validator):
        n1 = {"name": "foo", "type": "function", "file_path": "/a.py", "properties": {}}
        n2 = {"name": "foo", "type": "class", "file_path": "/a.py", "properties": {}}
        assert validator._nodes_differ(n1, n2) is True

    def test_property_changed(self, validator):
        n1 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"signature": "def foo()"},
        }
        n2 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"signature": "def foo(x)"},
        }
        assert validator._nodes_differ(n1, n2) is True

    def test_irrelevant_property_unchanged(self, validator):
        n1 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"something_else": 1},
        }
        n2 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"something_else": 2},
        }
        assert validator._nodes_differ(n1, n2) is False


# ---------------------------------------------------------------------------
# Tests: _get_node_changes
# ---------------------------------------------------------------------------


class TestGetNodeChanges:
    def test_no_changes(self, validator):
        n = {"name": "foo", "type": "function", "file_path": "/a.py", "properties": {}}
        assert validator._get_node_changes(n, n) == []

    def test_name_change(self, validator):
        n1 = {"name": "foo", "type": "function", "file_path": "/a.py", "properties": {}}
        n2 = {"name": "bar", "type": "function", "file_path": "/a.py", "properties": {}}
        changes = validator._get_node_changes(n1, n2)
        assert len(changes) == 1
        assert "name" in changes[0]

    def test_property_change(self, validator):
        n1 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"visibility": "public"},
        }
        n2 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"visibility": "private"},
        }
        changes = validator._get_node_changes(n1, n2)
        assert len(changes) == 1
        assert "visibility" in changes[0]

    def test_multiple_changes(self, validator):
        n1 = {
            "name": "foo",
            "type": "function",
            "file_path": "/a.py",
            "properties": {"complexity": 1},
        }
        n2 = {"name": "bar", "type": "class", "file_path": "/b.py", "properties": {"complexity": 5}}
        changes = validator._get_node_changes(n1, n2)
        # name, type, file_path, complexity = 4 changes
        assert len(changes) == 4


# ---------------------------------------------------------------------------
# Tests: validate_pre_update (integration of sub-validators)
# ---------------------------------------------------------------------------


class TestValidatePreUpdate:
    async def test_all_checks_pass(self, validator, tmp_path):
        f = tmp_path / "new.py"
        f.write_text("pass")
        changes = {"added": [f]}

        db_conn = _make_mock_db_conn()
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_pre_update(tmp_path, changes)

        assert result["valid"] is True
        assert "database_connectivity" in result["checks_performed"]
        assert "file_accessibility" in result["checks_performed"]
        assert "repository_state" in result["checks_performed"]
        assert "change_conflicts" in result["checks_performed"]
        assert result["timestamp"]

    async def test_db_failure_makes_invalid(self, validator, tmp_path):
        changes = {"added": []}
        with patch.object(validator, "_get_db_connection", side_effect=Exception("db fail")):
            result = await validator.validate_pre_update(tmp_path, changes)
        assert result["valid"] is False

    async def test_missing_file_makes_invalid(self, validator, tmp_path):
        missing = tmp_path / "gone.py"
        changes = {"added": [missing]}
        db_conn = _make_mock_db_conn()
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_pre_update(tmp_path, changes)
        assert result["valid"] is False

    async def test_conflicting_changes_makes_invalid(self, validator, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("pass")
        changes = {"added": [f], "modified": [f]}
        db_conn = _make_mock_db_conn()
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_pre_update(tmp_path, changes)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: validate_during_update
# ---------------------------------------------------------------------------


class TestValidateDuringUpdate:
    async def test_all_checks_pass(self, validator):
        db_conn = _make_mock_db_conn()
        stats = {"files_processed": 5, "processing_time": 2}
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_during_update("v1", stats)
        assert result["valid"] is True
        assert "node_edge_relationship" in result["consistency_checks"]
        assert "orphaned_nodes" in result["consistency_checks"]
        assert "duplicate_nodes" in result["consistency_checks"]
        assert "processing_statistics" in result["consistency_checks"]

    async def test_duplicate_nodes_makes_invalid(self, validator):
        db_conn = _make_mock_db_conn(duplicate_groups=[("id1", "Dup", "class", 2)])
        stats = {}
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_during_update("v1", stats)
        assert result["valid"] is False

    async def test_negative_stat_no_top_level_warning(self, validator):
        """Stats warnings only propagate when stats_check['valid'] is False.
        _validate_processing_statistics always returns valid=True, so warnings
        from negative values do NOT appear in the top-level result."""
        db_conn = _make_mock_db_conn()
        stats = {"errors": -3}
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_during_update("v1", stats)
        # The stats sub-validator logs warnings but validate_during_update only
        # propagates them when valid=False, which never happens here.
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Tests: validate_post_update
# ---------------------------------------------------------------------------


class TestValidatePostUpdate:
    async def test_all_pass(self, validator):
        db_conn = _make_mock_db_conn(version_row=("v1", "desc", "2024-01-01T00:00:00"))
        stats = {
            "processing_time": 10,
            "files_processed": 5,
            "added_nodes": 1,
            "modified_nodes": 0,
            "deleted_nodes": 0,
            "added_edges": 1,
            "deleted_edges": 0,
        }
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_post_update("v1", stats)
        assert result["valid"] is True
        assert "comprehensive_integrity" in result["integrity_checks"]
        assert "version_consistency" in result["integrity_checks"]
        assert "relationship_integrity" in result["integrity_checks"]
        assert "performance_metrics" in result

    async def test_version_not_found_makes_invalid(self, validator):
        db_conn = _make_mock_db_conn(version_row=None)
        stats = {}
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.validate_post_update("v1", stats)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: compare_versions
# ---------------------------------------------------------------------------


class TestCompareVersions:
    async def test_both_versions_missing(self, validator):
        db_conn = _make_mock_db_conn(version_row=None)
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator.compare_versions("v1", "v2")
        assert result["valid"] is False
        assert "not found" in result["error"]

    async def test_successful_comparison(self, validator):
        """Test compare_versions with both versions found and snapshot data."""
        # We mock _get_version and _perform_detailed_version_comparison
        v1 = {"id": "v1", "created_at": "2024-01-01", "description": "first"}
        v2 = {"id": "v2", "created_at": "2024-01-02", "description": "second"}
        comparison = {
            "v1_stats": {"node_count": 5, "edge_count": 3},
            "v2_stats": {"node_count": 7, "edge_count": 4},
            "changes": {"nodes": {"added": 2}, "edges": {"added": 1}},
            "details": {},
            "metrics": {},
        }

        with (
            patch.object(validator, "_get_version", side_effect=[v1, v2]),
            patch.object(
                validator,
                "_perform_detailed_version_comparison",
                return_value=comparison,
            ),
        ):
            result = await validator.compare_versions("v1", "v2")

        assert result["valid"] is True
        assert result["version1"]["id"] == "v1"
        assert result["version2"]["id"] == "v2"
        assert result["version1"]["node_count"] == 5
        assert result["version2"]["node_count"] == 7

    async def test_one_version_missing(self, validator):
        v1 = {"id": "v1", "created_at": "2024-01-01", "description": "first"}
        with patch.object(validator, "_get_version", side_effect=[v1, None]):
            result = await validator.compare_versions("v1", "v2")
        assert result["valid"] is False
        assert result["version1_exists"] is True
        assert result["version2_exists"] is False


# ---------------------------------------------------------------------------
# Tests: _validate_version_consistency
# ---------------------------------------------------------------------------


class TestValidateVersionConsistency:
    async def test_version_exists(self, validator):
        with patch.object(
            validator,
            "_get_version",
            return_value={"id": "v1", "description": "d", "created_at": "t"},
        ):
            result = await validator._validate_version_consistency("v1")
        assert result["valid"] is True

    async def test_version_not_found(self, validator):
        with patch.object(validator, "_get_version", return_value=None):
            result = await validator._validate_version_consistency("v1")
        assert result["valid"] is False

    async def test_version_lookup_raises(self, validator):
        with patch.object(validator, "_get_version", side_effect=Exception("db boom")):
            result = await validator._validate_version_consistency("v1")
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Tests: _validate_graph_integrity
# ---------------------------------------------------------------------------


class TestValidateGraphIntegrity:
    async def test_all_clean(self, validator):
        db_conn = _make_mock_db_conn()
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_graph_integrity()
        assert result["valid"] is True

    async def test_invalid_edges_propagate(self, validator):
        db_conn = _make_mock_db_conn(invalid_edges=[("s", "t", "calls")])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_graph_integrity()
        assert result["valid"] is False

    async def test_orphans_give_warnings(self, validator):
        db_conn = _make_mock_db_conn(orphaned_nodes=[("id1", "orphan", "function")])
        with patch.object(validator, "_get_db_connection", return_value=db_conn):
            result = await validator._validate_graph_integrity()
        assert result["valid"] is True
        assert len(result["warnings"]) > 0


# ---------------------------------------------------------------------------
# Tests: _perform_detailed_version_comparison
# ---------------------------------------------------------------------------


class TestPerformDetailedVersionComparison:
    async def test_added_and_unchanged_nodes(self, validator):
        """Detects nodes present in v2 but not v1, and unchanged nodes."""
        v1 = {"id": "v1"}
        v2 = {"id": "v2"}
        v1_snap = {
            "nodes": [
                {"id": "n1", "name": "A", "type": "class", "file_path": "/a.py", "properties": {}}
            ],
            "edges": [],
        }
        v2_snap = {
            "nodes": [
                {"id": "n1", "name": "A", "type": "class", "file_path": "/a.py", "properties": {}},
                {
                    "id": "n2",
                    "name": "B",
                    "type": "function",
                    "file_path": "/b.py",
                    "properties": {},
                },
            ],
            "edges": [],
        }
        with (
            patch.object(validator, "_get_db_connection") as mock_db_fn,
            patch.object(validator, "_get_version_snapshot", side_effect=[v1_snap, v2_snap]),
        ):
            mock_db_fn.return_value = _make_facade_conn()
            result = await validator._perform_detailed_version_comparison(v1, v2)
        assert result["changes"]["nodes"]["added"] == 1
        assert result["changes"]["nodes"]["unchanged"] == 1
        assert result["details"]["added_nodes"][0]["id"] == "n2"

    async def test_removed_nodes(self, validator):
        v1 = {"id": "v1"}
        v2 = {"id": "v2"}
        v1_snap = {
            "nodes": [
                {"id": "n1", "name": "A", "type": "class", "file_path": "/a.py", "properties": {}},
                {
                    "id": "n2",
                    "name": "B",
                    "type": "function",
                    "file_path": "/b.py",
                    "properties": {},
                },
            ],
            "edges": [],
        }
        v2_snap = {
            "nodes": [
                {"id": "n1", "name": "A", "type": "class", "file_path": "/a.py", "properties": {}}
            ],
            "edges": [],
        }
        with (
            patch.object(validator, "_get_db_connection") as mock_db_fn,
            patch.object(validator, "_get_version_snapshot", side_effect=[v1_snap, v2_snap]),
        ):
            mock_db_fn.return_value = _make_facade_conn()
            result = await validator._perform_detailed_version_comparison(v1, v2)
        assert result["changes"]["nodes"]["removed"] == 1
        assert result["details"]["removed_nodes"][0]["id"] == "n2"

    async def test_modified_nodes(self, validator):
        v1 = {"id": "v1"}
        v2 = {"id": "v2"}
        v1_snap = {
            "nodes": [
                {
                    "id": "n1",
                    "name": "A",
                    "type": "class",
                    "file_path": "/a.py",
                    "properties": {"signature": "class A"},
                }
            ],
            "edges": [],
        }
        v2_snap = {
            "nodes": [
                {
                    "id": "n1",
                    "name": "A",
                    "type": "class",
                    "file_path": "/a.py",
                    "properties": {"signature": "class A(Base)"},
                }
            ],
            "edges": [],
        }
        with (
            patch.object(validator, "_get_db_connection") as mock_db_fn,
            patch.object(validator, "_get_version_snapshot", side_effect=[v1_snap, v2_snap]),
        ):
            mock_db_fn.return_value = _make_facade_conn()
            result = await validator._perform_detailed_version_comparison(v1, v2)
        assert result["changes"]["nodes"]["modified"] == 1

    async def test_edge_changes(self, validator):
        v1 = {"id": "v1"}
        v2 = {"id": "v2"}
        v1_snap = {
            "nodes": [],
            "edges": [{"source_id": "n1", "target_id": "n2", "relationship_type": "calls"}],
        }
        v2_snap = {
            "nodes": [],
            "edges": [{"source_id": "n1", "target_id": "n3", "relationship_type": "imports"}],
        }
        with (
            patch.object(validator, "_get_db_connection") as mock_db_fn,
            patch.object(validator, "_get_version_snapshot", side_effect=[v1_snap, v2_snap]),
        ):
            mock_db_fn.return_value = _make_facade_conn()
            result = await validator._perform_detailed_version_comparison(v1, v2)
        assert result["changes"]["edges"]["added"] == 1
        assert result["changes"]["edges"]["removed"] == 1

    async def test_error_returns_empty_comparison(self, validator):
        with (
            patch.object(validator, "_get_db_connection") as mock_db_fn,
            patch.object(validator, "_get_version_snapshot", side_effect=ValueError("db error")),
        ):
            mock_db_fn.return_value = _make_facade_conn()
            result = await validator._perform_detailed_version_comparison(
                {"id": "v1"}, {"id": "v2"}
            )
        assert result["v1_stats"]["node_count"] == 0
        assert result["metrics"]["stability_score"] == 1.0


# ---------------------------------------------------------------------------
# Tests: _get_version_snapshot
# ---------------------------------------------------------------------------


class TestGetVersionSnapshot:
    async def test_reads_nodes_and_edges(self, validator):
        # JSONB columns come back from psycopg as dicts already (no json.loads);
        # source iterates the cursor with `async for row in cursor`.
        db_conn = MagicMock()
        node_rows = [
            ("n1", "Foo", "class", {"file": "/a.py"}),
            ("n2", "bar", "function", None),
        ]
        edge_rows = [("n1", "n2", "calls", {"weight": 1})]
        call_count = [0]

        async def execute_side_effect(sql, params=None):
            call_count[0] += 1
            return _async_cursor(node_rows if call_count[0] == 1 else edge_rows)

        db_conn.conn.execute = AsyncMock(side_effect=execute_side_effect)
        result = await validator._get_version_snapshot("v1", db_conn)
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["file_path"] == "/a.py"
        assert len(result["edges"]) == 1

    async def test_non_dict_properties_yields_empty_file_path(self, validator):
        # Defensive path: if a properties value is not a dict, file_path is "".
        db_conn = MagicMock()
        call_count = [0]

        async def execute_side_effect(sql, params=None):
            call_count[0] += 1
            rows = [("n1", "Foo", "class", "not-a-dict")] if call_count[0] == 1 else []
            return _async_cursor(rows)

        db_conn.conn.execute = AsyncMock(side_effect=execute_side_effect)
        result = await validator._get_version_snapshot("v1", db_conn)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["file_path"] == ""

    async def test_db_error_returns_empty_snapshot(self, validator):
        db_conn = MagicMock()
        db_conn.conn.execute.side_effect = OSError("db error")
        result = await validator._get_version_snapshot("v1", db_conn)
        assert result["nodes"] == []
        assert result["edges"] == []
