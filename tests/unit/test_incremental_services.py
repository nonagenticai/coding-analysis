"""Unit tests for incremental_update_recovery and incremental_update_processor."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub models.github_models (does not exist in the codebase)
if "models.github_models" not in sys.modules:
    _ghm = _types.ModuleType("models.github_models")
    _ghm.CodeSnippet = MagicMock
    _ghm.RepositoryContext = MagicMock
    sys.modules["models.github_models"] = _ghm

# ---------------------------------------------------------------------------
# Patch the settings at module level BEFORE importing modules under test so
# that the module-level `settings = get_settings()` picks up our mock.
# ---------------------------------------------------------------------------
import importlib

import app.core.config.core_config_settings as _settings_mod

_mock_settings = MagicMock()
_mock_settings.CODE_GRAPH_DATABASE_DIR = "/tmp/test_graphs"

with patch.object(_settings_mod, "get_settings", return_value=_mock_settings):
    # Force fresh import of the modules under test
    import services.incremental.incremental_update_processor as _processor_mod
    import services.incremental.incremental_update_recovery as _recovery_mod

    importlib.reload(_recovery_mod)
    importlib.reload(_processor_mod)

UpdateRecoveryManager = _recovery_mod.UpdateRecoveryManager
IncrementalUpdateProcessor = _processor_mod.IncrementalUpdateProcessor


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_db_conn():
    """Create a mock GraphDatabaseConnection facade for the postgres call sites.

    Source obtains the facade via ``_get_db_connection()``, then ``await``s
    ``connect()`` / ``conn.execute(...)`` / ``conn.commit()`` and ``close()``.
    ``conn.execute`` is async and returns a cursor; ``conn.commit`` is async;
    ``connect`` / ``close`` (the facade-level lifecycle) are async.
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
    """Create a fake psycopg async cursor (tuple rows; positional access).

    The source ``await``s ``cursor.fetchone()`` / ``fetchall()`` and reads
    ``cursor.rowcount`` (sync); ``_get_related_nodes`` iterates ``async for``.
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


# ============================================================================
# UpdateRecoveryManager tests
# ============================================================================


class TestUpdateRecoveryManagerInit:
    """Test basic construction and attributes."""

    def test_init_sets_attributes(self):
        # Under shared PostgreSQL there is no per-graph file/backup dir anymore;
        # recovery is DB-level cleanup scoped to one code_graph_id.
        mgr = UpdateRecoveryManager("graph-123")
        assert mgr.code_graph_id == "graph-123"
        assert mgr.recovery_log == []

    def test_get_db_connection(self):
        mgr = UpdateRecoveryManager("graph-123")
        with patch.object(_recovery_mod, "GraphDatabaseConnection") as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr._get_db_connection()
            mock_cls.assert_called_once_with("graph-123")


class TestCreateBackup:
    """Tests for create_backup (a no-op under shared PostgreSQL).

    Code graphs are regenerable rows in the shared DB, not per-graph files, so
    backups copy nothing; create_backup returns a synthetic, non-empty id so
    callers' control flow is unchanged.
    """

    async def test_create_backup_with_label(self):
        mgr = UpdateRecoveryManager("graph-1")
        backup_id = await mgr.create_backup("test_label")
        assert "graph-1" in backup_id
        assert "test_label" in backup_id

    async def test_create_backup_no_label(self):
        mgr = UpdateRecoveryManager("graph-2")
        backup_id = await mgr.create_backup()
        assert backup_id.startswith("graph-2_")

    async def test_create_backup_never_raises_without_source(self):
        # There is no source file to find; backup is purely synthetic.
        mgr = UpdateRecoveryManager("missing-graph")
        backup_id = await mgr.create_backup()
        assert backup_id.startswith("missing-graph_")


class TestListBackups:
    """list_backups returns [] — no file-level backups under shared PostgreSQL."""

    async def test_list_backups_empty(self):
        mgr = UpdateRecoveryManager("graph-1")
        result = await mgr.list_backups()
        assert result == []


class TestRestoreFromBackup:
    """restore_from_backup is unsupported under shared PostgreSQL (returns False)."""

    async def test_restore_returns_false_and_logs(self):
        mgr = UpdateRecoveryManager("graph-1")
        result = await mgr.restore_from_backup("any-backup")
        assert result is False
        assert len(mgr.recovery_log) == 1
        assert mgr.recovery_log[-1]["success"] is False
        assert mgr.recovery_log[-1]["operation"] == "restore_from_backup"


class TestRollbackToVersion:
    """Tests for rollback_to_version (DB-level under shared PostgreSQL)."""

    async def test_rollback_version_not_found(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        # _get_version_info -> fetchone returns no row
        mock_conn.conn.execute = AsyncMock(return_value=_make_cursor(rows=None))

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            result = await mgr.rollback_to_version("v-missing")

        assert result is False

    async def test_rollback_success(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()

        # _get_version_info returns a row, then _perform_version_rollback DELETEs
        version_cursor = _make_cursor(rows=[("v1", "description", "2026-01-01")])
        delete_cursor = _make_cursor(rowcount=1)
        mock_conn.conn.execute = AsyncMock(side_effect=[version_cursor, delete_cursor])

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            result = await mgr.rollback_to_version("v1")

        assert result is True
        assert mgr.recovery_log[-1]["success"] is True
        # the DELETE removes versions newer than the target's created_at
        delete_sql, params = mock_conn.conn.execute.await_args_list[-1].args
        assert "DELETE FROM graph_versions" in " ".join(delete_sql.split())
        assert params[0] == "graph-1"

    async def test_rollback_exception(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(
            mgr, "_get_version_info", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await mgr.rollback_to_version("v1")
        assert result is False
        assert mgr.recovery_log[-1]["success"] is False
        assert "error" in mgr.recovery_log[-1]


class TestRecoverFromFailedUpdate:
    """Tests for recover_from_failed_update."""

    async def test_unknown_strategy(self):
        mgr = UpdateRecoveryManager("graph-1")
        result = await mgr.recover_from_failed_update("v-fail", recovery_strategy="unknown")
        assert result["success"] is False
        assert any("Unknown recovery strategy" in e for e in result["errors"])

    async def test_auto_strategy_calls_auto_recovery(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_auto_recovery", new_callable=AsyncMock) as mock_auto:
            mock_auto.return_value = {
                "success": True,
                "strategy_used": "auto",
                "actions_taken": ["auto done"],
                "errors": [],
                "timestamp": "now",
            }
            result = await mgr.recover_from_failed_update("v-fail", recovery_strategy="auto")
            mock_auto.assert_awaited_once()
            assert result["success"] is True

    async def test_rollback_strategy(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_rollback_recovery", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = {
                "success": True,
                "strategy_used": "rollback",
                "actions_taken": [],
                "errors": [],
                "timestamp": "now",
            }
            await mgr.recover_from_failed_update("v-fail", recovery_strategy="rollback")
            mock_rb.assert_awaited_once()

    async def test_repair_strategy(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_repair_recovery", new_callable=AsyncMock) as mock_repair:
            mock_repair.return_value = {
                "success": True,
                "strategy_used": "repair",
                "actions_taken": ["repaired"],
                "errors": [],
                "timestamp": "now",
            }
            await mgr.recover_from_failed_update("v-fail", recovery_strategy="repair")
            mock_repair.assert_awaited_once()

    async def test_exception_during_recovery(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(
            mgr, "_auto_recovery", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await mgr.recover_from_failed_update("v-fail", recovery_strategy="auto")
            assert result["success"] is False
            assert any("boom" in e for e in result["errors"])


class TestCleanupPartialUpdate:
    """Tests for cleanup_partial_update."""

    async def test_cleanup_success(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()

        # DELETE FROM graph_versions returns rowcount = 1
        delete_cursor = _make_cursor(rowcount=1)
        mock_conn.conn.execute.return_value = delete_cursor

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            with patch.object(mgr, "_clean_orphaned_nodes", new_callable=AsyncMock, return_value=3):
                with patch.object(
                    mgr, "_clean_invalid_edges", new_callable=AsyncMock, return_value=2
                ):
                    result = await mgr.cleanup_partial_update("v-partial")

        assert result["success"] is True
        assert len(result["cleaned_items"]) == 3  # version + orphaned + invalid

    async def test_cleanup_no_records(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        delete_cursor = _make_cursor(rowcount=0)
        mock_conn.conn.execute.return_value = delete_cursor

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            with patch.object(mgr, "_clean_orphaned_nodes", new_callable=AsyncMock, return_value=0):
                with patch.object(
                    mgr, "_clean_invalid_edges", new_callable=AsyncMock, return_value=0
                ):
                    result = await mgr.cleanup_partial_update("v-clean")

        assert result["success"] is True
        assert result["cleaned_items"] == []

    async def test_cleanup_db_exception(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_get_db_connection", side_effect=RuntimeError("db error")):
            result = await mgr.cleanup_partial_update("v-err")

        assert result["success"] is False
        assert len(result["errors"]) > 0


class TestGetRecoveryLog:
    """Tests for get_recovery_log."""

    async def test_empty_log(self):
        mgr = UpdateRecoveryManager("graph-1")
        log = await mgr.get_recovery_log()
        assert log == []

    async def test_log_returns_copy(self):
        mgr = UpdateRecoveryManager("graph-1")
        mgr.recovery_log.append({"operation": "test"})
        log = await mgr.get_recovery_log()
        assert len(log) == 1
        log.append({"operation": "extra"})
        assert len(mgr.recovery_log) == 1  # original unchanged


class TestValidateBackupIntegrity:
    """validate_backup_integrity always reports invalid — file-level backups are
    unsupported under shared PostgreSQL."""

    async def test_always_invalid(self):
        mgr = UpdateRecoveryManager("graph-1")
        result = await mgr.validate_backup_integrity("any-backup")
        assert result["valid"] is False
        assert result["backup_id"] == "any-backup"
        assert any("unsupported under shared PostgreSQL" in e for e in result["errors"])


class TestAutoRecovery:
    """Tests for _auto_recovery."""

    async def test_auto_repair_succeeds(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_attempt_repair", new_callable=AsyncMock, return_value=True):
            result = await mgr._auto_recovery(
                "v-fail",
                {"success": False, "actions_taken": [], "errors": []},
            )
        assert result["success"] is True
        assert "Automatic repair successful" in result["actions_taken"]

    async def test_auto_repair_fails_rollback_succeeds(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_attempt_repair", new_callable=AsyncMock, return_value=False):
            with patch.object(
                mgr, "_find_previous_version", new_callable=AsyncMock, return_value="v-prev"
            ):
                with patch.object(
                    mgr, "rollback_to_version", new_callable=AsyncMock, return_value=True
                ):
                    result = await mgr._auto_recovery(
                        "v-fail",
                        {"success": False, "actions_taken": [], "errors": []},
                    )
        assert result["success"] is True

    async def test_auto_no_previous_version(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_attempt_repair", new_callable=AsyncMock, return_value=False):
            with patch.object(
                mgr, "_find_previous_version", new_callable=AsyncMock, return_value=None
            ):
                result = await mgr._auto_recovery(
                    "v-fail",
                    {"success": False, "actions_taken": [], "errors": []},
                )
        assert result["success"] is False
        assert any("No previous version" in e for e in result["errors"])


class TestRollbackRecovery:
    """Tests for _rollback_recovery."""

    async def test_rollback_no_previous(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_find_previous_version", new_callable=AsyncMock, return_value=None):
            result = await mgr._rollback_recovery(
                "v-fail",
                {"success": False, "actions_taken": [], "errors": []},
            )
        assert result["success"] is False

    async def test_rollback_succeeds(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(
            mgr, "_find_previous_version", new_callable=AsyncMock, return_value="v-prev"
        ):
            with patch.object(
                mgr, "rollback_to_version", new_callable=AsyncMock, return_value=True
            ):
                result = await mgr._rollback_recovery(
                    "v-fail",
                    {"success": False, "actions_taken": [], "errors": []},
                )
        assert result["success"] is True

    async def test_rollback_fails(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(
            mgr, "_find_previous_version", new_callable=AsyncMock, return_value="v-prev"
        ):
            with patch.object(
                mgr, "rollback_to_version", new_callable=AsyncMock, return_value=False
            ):
                result = await mgr._rollback_recovery(
                    "v-fail",
                    {"success": False, "actions_taken": [], "errors": []},
                )
        assert result["success"] is False
        assert any("Rollback failed" in e for e in result["errors"])


class TestRepairRecovery:
    """Tests for _repair_recovery."""

    async def test_repair_success(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_attempt_repair", new_callable=AsyncMock, return_value=True):
            result = await mgr._repair_recovery(
                "v-fail",
                {"success": False, "actions_taken": [], "errors": []},
            )
        assert result["success"] is True
        assert "Data repair successful" in result["actions_taken"]

    async def test_repair_failure(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_attempt_repair", new_callable=AsyncMock, return_value=False):
            result = await mgr._repair_recovery(
                "v-fail",
                {"success": False, "actions_taken": [], "errors": []},
            )
        assert result["success"] is False
        assert any("repair failed" in e.lower() for e in result["errors"])


class TestAttemptRepair:
    """Tests for _attempt_repair."""

    async def test_repair_runs_cleanup(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            with patch.object(mgr, "_clean_orphaned_nodes", new_callable=AsyncMock, return_value=2):
                with patch.object(
                    mgr, "_clean_invalid_edges", new_callable=AsyncMock, return_value=1
                ):
                    result = await mgr._attempt_repair("v-fail")

        assert result is True
        mock_conn.conn.commit.assert_awaited_once()
        # source closes the facade (db_conn.close), not the raw conn
        mock_conn.close.assert_awaited_once()

    async def test_repair_exception_returns_false(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_get_db_connection", side_effect=RuntimeError("db down")):
            result = await mgr._attempt_repair("v-fail")
        assert result is False


class TestFindPreviousVersion:
    """Tests for _find_previous_version."""

    async def test_found(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=[("v-prev",)])

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            result = await mgr._find_previous_version("v-current")
        assert result == "v-prev"

    async def test_not_found(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=None)

        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            result = await mgr._find_previous_version("v-current")
        assert result is None

    async def test_exception(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_get_db_connection", side_effect=RuntimeError("err")):
            result = await mgr._find_previous_version("v-x")
        assert result is None


class TestCleanOrphanedNodes:
    """Tests for _clean_orphaned_nodes."""

    async def test_clean_returns_count(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rowcount=5)
        result = await mgr._clean_orphaned_nodes(mock_conn)
        assert result == 5

    async def test_clean_exception_returns_zero(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = RuntimeError("sql err")
        result = await mgr._clean_orphaned_nodes(mock_conn)
        assert result == 0


class TestCleanInvalidEdges:
    """Tests for _clean_invalid_edges."""

    async def test_clean_returns_count(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rowcount=3)
        result = await mgr._clean_invalid_edges(mock_conn)
        assert result == 3

    async def test_clean_exception_returns_zero(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = RuntimeError("sql err")
        result = await mgr._clean_invalid_edges(mock_conn)
        assert result == 0


class TestGetVersionInfo:
    """Tests for _get_version_info."""

    async def test_version_found(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=[("v1", "initial", "2026-01-01")])
        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            info = await mgr._get_version_info("v1")
        assert info is not None
        assert info["id"] == "v1"
        assert info["description"] == "initial"

    async def test_version_not_found(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=None)
        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            info = await mgr._get_version_info("v-missing")
        assert info is None

    async def test_version_exception(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_get_db_connection", side_effect=RuntimeError("err")):
            info = await mgr._get_version_info("v-err")
        assert info is None


class TestPerformVersionRollback:
    """Tests for _perform_version_rollback."""

    async def test_rollback_success(self):
        mgr = UpdateRecoveryManager("graph-1")
        mock_conn = _make_mock_db_conn()
        with patch.object(mgr, "_get_db_connection", return_value=mock_conn):
            result = await mgr._perform_version_rollback("v1", {"created_at": "2026-01-01"})
        assert result is True
        mock_conn.conn.commit.assert_called_once()

    async def test_rollback_exception(self):
        mgr = UpdateRecoveryManager("graph-1")
        with patch.object(mgr, "_get_db_connection", side_effect=RuntimeError("err")):
            result = await mgr._perform_version_rollback("v1", {"created_at": "2026-01-01"})
        assert result is False


# ============================================================================
# IncrementalUpdateProcessor tests
# ============================================================================


class TestIncrementalUpdateProcessorInit:
    """Test basic construction."""

    def test_init_with_defaults(self):
        proc = IncrementalUpdateProcessor("graph-1")
        assert proc.code_graph_id == "graph-1"
        # code_processor and file_analyzer should be set
        assert proc.code_processor is not None
        assert proc.file_analyzer is not None

    def test_init_with_custom_deps(self):
        mock_cp = MagicMock()
        mock_fa = MagicMock()
        proc = IncrementalUpdateProcessor("graph-1", code_processor=mock_cp, file_analyzer=mock_fa)
        assert proc.code_processor is mock_cp
        assert proc.file_analyzer is mock_fa

    def test_get_db_connection(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(_processor_mod, "GraphDatabaseConnection") as mock_cls:
            mock_cls.return_value = MagicMock()
            proc._get_db_connection()
            mock_cls.assert_called_once()


class TestGenerateNodeId:
    """Tests for _generate_node_id."""

    def test_deterministic(self):
        proc = IncrementalUpdateProcessor("graph-1")
        id1 = proc._generate_node_id("function", "my_func", "src/main.py")
        id2 = proc._generate_node_id("function", "my_func", "src/main.py")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        proc = IncrementalUpdateProcessor("graph-1")
        id1 = proc._generate_node_id("function", "my_func", "src/main.py")
        id2 = proc._generate_node_id("class", "my_func", "src/main.py")
        assert id1 != id2

    def test_format(self):
        proc = IncrementalUpdateProcessor("graph-1")
        node_id = proc._generate_node_id("function", "test", "file.py")
        assert node_id.startswith("function_")
        assert len(node_id) == len("function_") + 8  # 8 hex chars


class TestProcessFileChanges:
    """Tests for process_file_changes."""

    async def test_process_returns_stats(self):
        proc = IncrementalUpdateProcessor("graph-1")

        with patch.object(
            proc, "_collect_affected_nodes", new_callable=AsyncMock, return_value=set()
        ):
            with patch.object(proc, "_process_deleted_files", new_callable=AsyncMock):
                with patch.object(proc, "_process_added_files", new_callable=AsyncMock):
                    with patch.object(proc, "_process_modified_files", new_callable=AsyncMock):
                        changes = {"added": [], "modified": [], "deleted": []}
                        stats = await proc.process_file_changes(Path("/repo"), changes, "v1")

        assert stats["version_id"] == "v1"
        assert stats["added_nodes"] == 0
        assert stats["processing_time"] >= 0

    async def test_process_calls_in_order(self):
        proc = IncrementalUpdateProcessor("graph-1")
        call_order = []

        async def mock_collect(*a, **kw):
            call_order.append("collect")
            return set()

        async def mock_deleted(*a, **kw):
            call_order.append("deleted")

        async def mock_added(*a, **kw):
            call_order.append("added")

        async def mock_modified(*a, **kw):
            call_order.append("modified")

        with patch.object(proc, "_collect_affected_nodes", side_effect=mock_collect):
            with patch.object(proc, "_process_deleted_files", side_effect=mock_deleted):
                with patch.object(proc, "_process_added_files", side_effect=mock_added):
                    with patch.object(proc, "_process_modified_files", side_effect=mock_modified):
                        changes = {
                            "added": [Path("a.py")],
                            "modified": [Path("m.py")],
                            "deleted": [Path("d.py")],
                        }
                        await proc.process_file_changes(Path("/repo"), changes, "v1")

        assert call_order == ["collect", "deleted", "added", "modified"]


class TestProcessDeletedFiles:
    """Tests for _process_deleted_files."""

    async def test_deletes_and_updates_stats(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(proc, "_delete_nodes_by_file", new_callable=AsyncMock, return_value=3):
            stats = {"deleted_nodes": 0, "files_processed": 0}
            await proc._process_deleted_files([Path("old.py")], stats)

        assert stats["deleted_nodes"] == 3
        assert stats["files_processed"] == 1

    async def test_exception_continues(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(
            proc, "_delete_nodes_by_file", new_callable=AsyncMock, side_effect=OSError("err")
        ):
            stats = {"deleted_nodes": 0, "files_processed": 0}
            await proc._process_deleted_files([Path("bad.py"), Path("good.py")], stats)
        # Both should have attempted, neither succeeded
        assert stats["deleted_nodes"] == 0


class TestProcessAddedFiles:
    """Tests for _process_added_files."""

    async def test_adds_nodes_and_edges(self):
        proc = IncrementalUpdateProcessor("graph-1")

        entities = [
            {
                "type": "function",
                "name": "foo",
                "related_entities": [{"type": "class", "name": "Bar", "relationship": "calls"}],
            }
        ]

        with patch.object(
            proc, "_analyze_file_for_entities", new_callable=AsyncMock, return_value=entities
        ):
            with patch.object(
                proc, "_add_entity_node", new_callable=AsyncMock, return_value="node-1"
            ):
                with patch.object(
                    proc, "_add_entity_relationships", new_callable=AsyncMock, return_value=1
                ):
                    stats = {"added_nodes": 0, "added_edges": 0, "files_processed": 0}
                    await proc._process_added_files(Path("/repo"), [Path("new.py")], stats)

        assert stats["added_nodes"] == 1
        assert stats["added_edges"] == 1
        assert stats["files_processed"] == 1

    async def test_add_entity_node_returns_none(self):
        proc = IncrementalUpdateProcessor("graph-1")

        entities = [{"type": "function", "name": "foo"}]

        with patch.object(
            proc, "_analyze_file_for_entities", new_callable=AsyncMock, return_value=entities
        ):
            with patch.object(proc, "_add_entity_node", new_callable=AsyncMock, return_value=None):
                stats = {"added_nodes": 0, "added_edges": 0, "files_processed": 0}
                await proc._process_added_files(Path("/repo"), [Path("fail.py")], stats)

        assert stats["added_nodes"] == 0
        assert stats["added_edges"] == 0


class TestProcessModifiedFiles:
    """Tests for _process_modified_files."""

    async def test_update_existing_node(self):
        proc = IncrementalUpdateProcessor("graph-1")

        node_id = proc._generate_node_id("function", "foo", "mod.py")
        existing_nodes = [{"id": node_id, "type": "function", "name": "foo"}]
        entities = [{"type": "function", "name": "foo"}]

        with patch.object(
            proc, "_get_nodes_by_file", new_callable=AsyncMock, return_value=existing_nodes
        ):
            with patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=entities,
            ):
                with patch.object(proc, "_update_entity_node", new_callable=AsyncMock):
                    with patch.object(
                        proc,
                        "_update_entity_relationships",
                        new_callable=AsyncMock,
                        return_value={"added": 0, "deleted": 0},
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
        assert stats["files_processed"] == 1

    async def test_deletes_removed_nodes(self):
        proc = IncrementalUpdateProcessor("graph-1")

        existing_nodes = [{"id": "old-node", "type": "function", "name": "removed"}]
        entities = []  # No entities in new version

        with patch.object(
            proc, "_get_nodes_by_file", new_callable=AsyncMock, return_value=existing_nodes
        ):
            with patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=entities,
            ):
                with patch.object(proc, "_delete_node", new_callable=AsyncMock) as mock_del:
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
        mock_del.assert_awaited_once_with("old-node")


class TestAnalyzeFileForEntities:
    """Tests for _analyze_file_for_entities."""

    async def test_returns_entities(self):
        proc = IncrementalUpdateProcessor("graph-1")

        mock_entity = MagicMock()
        mock_entity.type = "function"
        mock_entity.name = "foo"
        mock_entity.start_line = 1
        mock_entity.start_column = 0
        mock_entity.end_line = 5
        mock_entity.end_column = 0
        mock_entity.docstring = "A function"
        mock_entity.signature = "def foo():"

        mock_result = MagicMock()
        mock_result.entities = [mock_entity]

        proc.file_analyzer = MagicMock()
        proc.file_analyzer.analyze_file = AsyncMock(return_value=mock_result)

        entities = await proc._analyze_file_for_entities(Path("test.py"))
        assert len(entities) == 1
        assert entities[0]["type"] == "function"
        assert entities[0]["name"] == "foo"
        assert entities[0]["line"] == 1

    async def test_no_entities_attribute(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_result = MagicMock(spec=[])  # no entities attr
        proc.file_analyzer = MagicMock()
        proc.file_analyzer.analyze_file = AsyncMock(return_value=mock_result)

        entities = await proc._analyze_file_for_entities(Path("test.py"))
        assert entities == []

    async def test_exception_returns_empty(self):
        proc = IncrementalUpdateProcessor("graph-1")
        proc.file_analyzer = MagicMock()
        proc.file_analyzer.analyze_file = AsyncMock(side_effect=RuntimeError("parse err"))

        entities = await proc._analyze_file_for_entities(Path("bad.py"))
        assert entities == []


class TestAddEntityNode:
    """Tests for _add_entity_node."""

    async def test_success(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(proc, "_add_node", new_callable=AsyncMock, return_value="node-1"):
            entity = {"type": "function", "name": "foo", "line": 1}
            result = await proc._add_entity_node(entity, Path("test.py"))
        assert result is not None

    async def test_exception_returns_none(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(
            proc, "_add_node", new_callable=AsyncMock, side_effect=RuntimeError("err")
        ):
            entity = {"type": "function", "name": "foo"}
            result = await proc._add_entity_node(entity, Path("test.py"))
        assert result is None


class TestUpdateEntityNode:
    """Tests for _update_entity_node."""

    async def test_calls_update_node(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(proc, "_update_node", new_callable=AsyncMock) as mock_update:
            entity = {"type": "function", "name": "foo", "line": 10}
            await proc._update_entity_node(entity, "node-1")
        mock_update.assert_awaited_once()

    async def test_exception_doesnt_raise(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(
            proc, "_update_node", new_callable=AsyncMock, side_effect=RuntimeError("err")
        ):
            entity = {"type": "function", "name": "foo"}
            # Should not raise
            await proc._update_entity_node(entity, "node-1")


class TestAddEntityRelationships:
    """Tests for _add_entity_relationships."""

    async def test_adds_edges(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(proc, "_add_edge", new_callable=AsyncMock):
            entity = {
                "type": "function",
                "name": "foo",
                "related_entities": [
                    {"type": "class", "name": "Bar", "relationship": "calls"},
                    {"type": "module", "name": "baz", "relationship": "imports"},
                ],
            }
            count = await proc._add_entity_relationships(entity, "node-1", Path("f.py"))
        assert count == 2

    async def test_no_related(self):
        proc = IncrementalUpdateProcessor("graph-1")
        entity = {"type": "function", "name": "foo"}
        count = await proc._add_entity_relationships(entity, "node-1", Path("f.py"))
        assert count == 0

    async def test_edge_exception_continues(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(
            proc, "_add_edge", new_callable=AsyncMock, side_effect=RuntimeError("err")
        ):
            entity = {
                "type": "function",
                "name": "foo",
                "related_entities": [
                    {"type": "class", "name": "Bar", "relationship": "calls"},
                ],
            }
            count = await proc._add_entity_relationships(entity, "node-1", Path("f.py"))
        assert count == 0


class TestUpdateEntityRelationships:
    """Tests for _update_entity_relationships."""

    async def test_adds_new_edges(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(proc, "_get_edges_for_node", new_callable=AsyncMock, return_value=[]):
            with patch.object(proc, "_add_edge", new_callable=AsyncMock):
                entity = {
                    "type": "function",
                    "name": "foo",
                    "related_entities": [{"type": "class", "name": "Bar", "relationship": "calls"}],
                }
                result = await proc._update_entity_relationships(entity, "node-1", Path("f.py"))
        assert result["added"] == 1
        assert result["deleted"] == 0

    async def test_deletes_old_edges(self):
        proc = IncrementalUpdateProcessor("graph-1")
        existing_edges = [
            {
                "source_id": "node-1",
                "target_id": "old-target",
                "relationship_type": "calls",
                "properties": {},
            }
        ]
        with patch.object(
            proc, "_get_edges_for_node", new_callable=AsyncMock, return_value=existing_edges
        ):
            with patch.object(proc, "_delete_edge", new_callable=AsyncMock):
                entity = {"type": "function", "name": "foo"}
                result = await proc._update_entity_relationships(entity, "node-1", Path("f.py"))
        assert result["deleted"] == 1
        assert result["added"] == 0

    async def test_exception_returns_zeros(self):
        proc = IncrementalUpdateProcessor("graph-1")
        with patch.object(
            proc, "_get_edges_for_node", new_callable=AsyncMock, side_effect=RuntimeError("err")
        ):
            entity = {"type": "function", "name": "foo"}
            result = await proc._update_entity_relationships(entity, "node-1", Path("f.py"))
        assert result == {"added": 0, "deleted": 0}


class TestDatabaseOperations:
    """Tests for low-level DB operations on IncrementalUpdateProcessor."""

    async def test_add_node(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            result = await proc._add_node("n1", "function", "foo", {"line": 1})
        assert result == "n1"
        mock_conn.conn.execute.assert_awaited_once()
        mock_conn.conn.commit.assert_awaited_once()
        # source closes the facade (db_conn.close), not the raw conn
        mock_conn.close.assert_awaited_once()
        # write is scoped to the code_graph_id and INSERTs into nodes
        sql, params = mock_conn.conn.execute.await_args.args
        assert "INSERT INTO nodes" in " ".join(sql.split())
        assert params[0] == "graph-1"

    async def test_update_node_existing(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        # JSONB properties come back from psycopg as a dict (no json.loads)
        mock_conn.conn.execute.return_value = _make_cursor(rows=[({"line": 1},)])
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._update_node("n1", {"line": 5})
        # Should have called execute twice: SELECT + UPDATE
        assert mock_conn.conn.execute.await_count == 2
        mock_conn.close.assert_awaited_once()

    async def test_update_node_not_found(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=None)
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._update_node("n1", {"line": 5})
        # Only SELECT, no UPDATE
        assert mock_conn.conn.execute.await_count == 1

    async def test_add_edge(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._add_edge("src", "tgt", "calls", {})
        mock_conn.conn.execute.assert_awaited_once()
        mock_conn.conn.commit.assert_awaited_once()
        sql, params = mock_conn.conn.execute.await_args.args
        assert "INSERT INTO edges" in " ".join(sql.split())
        assert params[0] == "graph-1"

    async def test_delete_node(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._delete_node("n1")
        # Should delete edges then node
        assert mock_conn.conn.execute.await_count == 2
        mock_conn.conn.commit.assert_awaited_once()

    async def test_delete_edge(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            await proc._delete_edge("src", "tgt", "calls")
        mock_conn.conn.execute.assert_awaited_once()
        mock_conn.conn.commit.assert_awaited_once()

    async def test_get_nodes_by_file(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        # row[3] (properties) is a JSONB dict already; no json.loads in source
        mock_conn.conn.execute.return_value = _make_cursor(
            rows=[("n1", "function", "foo", {"file": "test.py"})]
        )
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            nodes = await proc._get_nodes_by_file("test.py")
        assert len(nodes) == 1
        assert nodes[0]["id"] == "n1"
        assert nodes[0]["properties"] == {"file": "test.py"}

    # Deleted test_get_nodes_by_file_bad_json: source no longer json.loads the
    # properties column (JSONB returns a dict), so there is no bad-json branch.

    async def test_get_edges_for_node(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=[("src", "tgt", "calls", {})])
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            edges = await proc._get_edges_for_node("src")
        assert len(edges) == 1
        assert edges[0]["source_id"] == "src"

    # Deleted test_get_edges_bad_json: source no longer json.loads the edge
    # properties column (JSONB returns a dict), so there is no bad-json branch.

    async def test_delete_nodes_by_file_none(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.return_value = _make_cursor(rows=[])
        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            count = await proc._delete_nodes_by_file("nonexistent.py")
        assert count == 0

    async def test_delete_nodes_by_file_with_nodes(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()

        call_count = {"n": 0}

        def execute_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: SELECT ids
                return _make_cursor(rows=[("n1",), ("n2",)])
            else:
                # Subsequent: DELETE calls
                return _make_cursor(rowcount=1)

        mock_conn.conn.execute.side_effect = execute_side_effect

        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            count = await proc._delete_nodes_by_file("test.py")
        assert count == 2

    async def test_get_related_nodes_empty(self):
        proc = IncrementalUpdateProcessor("graph-1")
        result = await proc._get_related_nodes([])
        assert result == set()

    async def test_get_related_nodes_found(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()

        call_count = {"n": 0}

        def execute_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_cursor(rows=[("related-1",)])
            else:
                return _make_cursor(rows=[("related-2",)])

        mock_conn.conn.execute.side_effect = execute_side_effect

        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            result = await proc._get_related_nodes([{"id": "n1"}])
        assert "related-1" in result
        assert "related-2" in result

    async def test_get_related_nodes_exception(self):
        proc = IncrementalUpdateProcessor("graph-1")
        mock_conn = _make_mock_db_conn()
        mock_conn.conn.execute.side_effect = ValueError("sql err")

        with patch.object(proc, "_get_db_connection", return_value=mock_conn):
            result = await proc._get_related_nodes([{"id": "n1"}])
        assert result == set()


class TestCollectAffectedNodes:
    """Tests for _collect_affected_nodes."""

    async def test_empty_changes(self):
        proc = IncrementalUpdateProcessor("graph-1")
        changes = {"added": [], "modified": [], "deleted": []}
        result = await proc._collect_affected_nodes(changes)
        assert result == set()

    async def test_added_files(self):
        proc = IncrementalUpdateProcessor("graph-1")
        entities = [{"type": "function", "name": "foo"}]

        with patch.object(
            proc,
            "_analyze_file_for_entities",
            new_callable=AsyncMock,
            return_value=entities,
        ):
            changes = {"added": [Path("new.py")], "modified": [], "deleted": []}
            result = await proc._collect_affected_nodes(changes)
        assert len(result) >= 1

    async def test_modified_files_include_existing(self):
        proc = IncrementalUpdateProcessor("graph-1")
        existing_nodes = [{"id": "existing-1"}]
        entities = [{"type": "function", "name": "foo"}]

        with patch.object(
            proc, "_get_nodes_by_file", new_callable=AsyncMock, return_value=existing_nodes
        ):
            with patch.object(
                proc,
                "_analyze_file_for_entities",
                new_callable=AsyncMock,
                return_value=entities,
            ):
                changes = {"added": [], "modified": [Path("mod.py")], "deleted": []}
                result = await proc._collect_affected_nodes(changes)
        assert "existing-1" in result

    async def test_deleted_files(self):
        proc = IncrementalUpdateProcessor("graph-1")
        existing_nodes = [{"id": "del-1"}]

        with patch.object(
            proc, "_get_nodes_by_file", new_callable=AsyncMock, return_value=existing_nodes
        ):
            with patch.object(
                proc, "_get_related_nodes", new_callable=AsyncMock, return_value={"rel-1"}
            ):
                changes = {"added": [], "modified": [], "deleted": [Path("old.py")]}
                result = await proc._collect_affected_nodes(changes)
        assert "del-1" in result
        assert "rel-1" in result

    async def test_related_entities_in_added(self):
        proc = IncrementalUpdateProcessor("graph-1")
        entities = [
            {
                "type": "function",
                "name": "foo",
                "related_entities": [{"type": "class", "name": "Bar", "file": "other.py"}],
            }
        ]

        with patch.object(
            proc,
            "_analyze_file_for_entities",
            new_callable=AsyncMock,
            return_value=entities,
        ):
            changes = {"added": [Path("new.py")], "modified": [], "deleted": []}
            result = await proc._collect_affected_nodes(changes)
        # Should have node_id for foo AND related_id for Bar
        assert len(result) >= 2
