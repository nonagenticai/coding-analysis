"""Update Rollback and Recovery Module.

This module provides functionality for rolling back failed updates and
recovering from errors during incremental update operations. It manages
backup creation, restoration, and error recovery mechanisms.

The module supports:
- Automatic backup creation before updates
- Rollback to previous states
- Error recovery and cleanup
- Transaction-like update operations
- Data integrity preservation during recovery
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from db.facade import GraphDatabaseConnection

logger = logging.getLogger(__name__)


class UpdateRecoveryManager:
    """Handles rollback and recovery operations for incremental updates.

    Code graphs live in the shared PostgreSQL database (one row-set per
    ``code_graph_id``), not in per-graph SQLite files, and are regenerable from
    source. File-based backup/restore is therefore unsupported here; recovery is
    DB-level cleanup (orphan/invalid-edge removal, version trimming) scoped to
    one ``code_graph_id``.
    """

    def __init__(self, code_graph_id: str):
        """Initialize the recovery manager.

        Args:
            code_graph_id: The code graph ID for database access
        """
        self.code_graph_id = code_graph_id
        self.recovery_log: list[dict[str, Any]] = []

    def _get_db_connection(self):
        """Get database connection for the code graph."""
        return GraphDatabaseConnection(self.code_graph_id)

    async def create_backup(self, backup_label: str | None = None) -> str:
        """Return a synthetic backup id; file-level backups are not used.

        Code graphs are regenerable rows in the shared PostgreSQL database, not
        per-graph files, so there is nothing to copy. Callers (rollback, restore)
        still receive a non-empty id so their control flow is unchanged.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_id = f"{self.code_graph_id}_{timestamp}"
        if backup_label:
            backup_id += f"_{backup_label}"
        logger.info(
            "Backup is a no-op under shared PostgreSQL (graph %s is regenerable); id=%s",
            self.code_graph_id,
            backup_id,
        )
        return backup_id

    async def list_backups(self) -> list[dict[str, Any]]:
        """No file-level backups exist under shared PostgreSQL."""
        return []

    async def restore_from_backup(self, backup_id: str) -> bool:
        """File-level restore is unsupported under shared PostgreSQL."""
        logger.warning(
            "restore_from_backup(%s) is unsupported under shared PostgreSQL; "
            "regenerate the graph from source instead",
            backup_id,
        )
        self.recovery_log.append(
            {
                "operation": "restore_from_backup",
                "backup_id": backup_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "success": False,
                "error": "file-level restore unsupported under shared PostgreSQL",
            }
        )
        return False

    async def rollback_to_version(self, version_id: str) -> bool:
        """Rollback the graph to a specific version state.

        Args:
            version_id: Version ID to rollback to

        Returns:
            True if rollback was successful, False otherwise
        """
        try:
            # Create backup before rollback
            backup_id = await self.create_backup(f"pre_rollback_to_{version_id}")

            # Get version details
            version_info = await self._get_version_info(version_id)
            if not version_info:
                logger.error("Version %s not found", version_id)
                return False

            # Perform rollback by removing all changes after the target version
            success = await self._perform_version_rollback(version_id, version_info)

            # Log the rollback operation
            recovery_entry = {
                "operation": "rollback_to_version",
                "target_version": version_id,
                "backup_id": backup_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "success": success,
            }
            self.recovery_log.append(recovery_entry)

            if success:
                logger.info("Successfully rolled back to version %s", version_id)
            else:
                logger.error("Failed to rollback to version %s", version_id)

            return success

        except Exception as e:
            logger.error("Error during rollback to version %s: %s", version_id, e)

            recovery_entry = {
                "operation": "rollback_to_version",
                "target_version": version_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "success": False,
                "error": str(e),
            }
            self.recovery_log.append(recovery_entry)
            return False

    async def recover_from_failed_update(
        self, failed_version_id: str, recovery_strategy: str = "auto"
    ) -> dict[str, Any]:
        """Recover from a failed update operation.

        Args:
            failed_version_id: Version ID of the failed update
            recovery_strategy: Strategy to use ("auto", "rollback", "repair")

        Returns:
            Dictionary with recovery results
        """
        recovery_result: dict[str, Any] = {
            "success": False,
            "strategy_used": recovery_strategy,
            "actions_taken": [],
            "errors": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            if recovery_strategy == "auto":
                # Automatic recovery: try repair first, then rollback
                recovery_result = await self._auto_recovery(failed_version_id, recovery_result)
            elif recovery_strategy == "rollback":
                # Rollback to previous version
                recovery_result = await self._rollback_recovery(failed_version_id, recovery_result)
            elif recovery_strategy == "repair":
                # Attempt to repair the current state
                recovery_result = await self._repair_recovery(failed_version_id, recovery_result)
            else:
                recovery_result["errors"].append(f"Unknown recovery strategy: {recovery_strategy}")

        except Exception as e:
            logger.error("Error during recovery from failed update %s: %s", failed_version_id, e)
            recovery_result["errors"].append(f"Recovery exception: {e}")

        # Log the recovery attempt
        self.recovery_log.append(
            {
                "operation": "recover_from_failed_update",
                "failed_version": failed_version_id,
                "strategy": recovery_strategy,
                "result": recovery_result,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return recovery_result

    async def cleanup_partial_update(self, version_id: str) -> dict[str, Any]:
        """Clean up artifacts from a partial/failed update.

        Args:
            version_id: Version ID of the partial update

        Returns:
            Cleanup results
        """
        cleanup_result: dict[str, Any] = {
            "success": True,
            "cleaned_items": [],
            "errors": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                # Remove version record if it exists
                cursor = await db_conn.conn.execute(
                    "DELETE FROM graph_versions WHERE code_graph_id = %s AND id = %s",
                    (self.code_graph_id, version_id),
                )
                if cursor.rowcount > 0:
                    cleanup_result["cleaned_items"].append(f"Removed version record: {version_id}")

                # Remove any orphaned nodes that might have been created
                orphaned_count = await self._clean_orphaned_nodes(db_conn)
                if orphaned_count > 0:
                    cleanup_result["cleaned_items"].append(
                        f"Removed {orphaned_count} orphaned nodes"
                    )

                # Remove any invalid edges
                invalid_edges_count = await self._clean_invalid_edges(db_conn)
                if invalid_edges_count > 0:
                    cleanup_result["cleaned_items"].append(
                        f"Removed {invalid_edges_count} invalid edges"
                    )

                await db_conn.conn.commit()
            finally:
                await db_conn.close()

            logger.info("Cleaned up partial update for version %s", version_id)

        except Exception as e:
            cleanup_result["success"] = False
            cleanup_result["errors"].append(f"Cleanup error: {e}")
            logger.error("Error during cleanup of version %s: %s", version_id, e)

        return cleanup_result

    async def get_recovery_log(self) -> list[dict[str, Any]]:
        """Get the recovery operation log.

        Returns:
            List of recovery operations performed
        """
        return self.recovery_log.copy()

    async def validate_backup_integrity(self, backup_id: str) -> dict[str, Any]:
        """Validate the integrity of a backup.

        Args:
            backup_id: ID of the backup to validate

        Returns:
            Validation results
        """
        return {
            "valid": False,
            "backup_id": backup_id,
            "checks": [],
            "errors": ["file-level backups are unsupported under shared PostgreSQL"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _auto_recovery(self, failed_version_id: str, recovery_result: dict) -> dict:
        """Perform automatic recovery."""
        # Try repair first
        repair_success = await self._attempt_repair(failed_version_id)
        if repair_success:
            recovery_result["success"] = True
            recovery_result["actions_taken"].append("Automatic repair successful")
            return recovery_result

        # If repair fails, try rollback
        recovery_result["actions_taken"].append("Repair failed, attempting rollback")

        # Find previous version to rollback to
        previous_version = await self._find_previous_version(failed_version_id)
        if previous_version:
            rollback_success = await self.rollback_to_version(previous_version)
            if rollback_success:
                recovery_result["success"] = True
                recovery_result["actions_taken"].append(
                    f"Rolled back to version {previous_version}"
                )
            else:
                recovery_result["errors"].append("Rollback failed")
        else:
            recovery_result["errors"].append("No previous version found for rollback")

        return recovery_result

    async def _rollback_recovery(self, failed_version_id: str, recovery_result: dict) -> dict:
        """Perform rollback recovery."""
        previous_version = await self._find_previous_version(failed_version_id)
        if previous_version:
            rollback_success = await self.rollback_to_version(previous_version)
            if rollback_success:
                recovery_result["success"] = True
                recovery_result["actions_taken"].append(
                    f"Rolled back to version {previous_version}"
                )
            else:
                recovery_result["errors"].append("Rollback failed")
        else:
            recovery_result["errors"].append("No previous version found for rollback")

        return recovery_result

    async def _repair_recovery(self, failed_version_id: str, recovery_result: dict) -> dict:
        """Perform repair recovery."""
        repair_success = await self._attempt_repair(failed_version_id)
        if repair_success:
            recovery_result["success"] = True
            recovery_result["actions_taken"].append("Data repair successful")
        else:
            recovery_result["errors"].append("Data repair failed")

        return recovery_result

    async def _attempt_repair(self, failed_version_id: str) -> bool:
        """Attempt to repair data inconsistencies."""
        try:
            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                # Remove orphaned nodes
                orphaned_count = await self._clean_orphaned_nodes(db_conn)

                # Remove invalid edges
                invalid_edges_count = await self._clean_invalid_edges(db_conn)

                await db_conn.conn.commit()
            finally:
                await db_conn.close()

            logger.info(
                "Repair completed: removed %d orphaned nodes, %d invalid edges",
                orphaned_count,
                invalid_edges_count,
            )

            return True

        except Exception as e:
            logger.error("Repair attempt failed: %s", e)
            return False

    async def _find_previous_version(self, current_version_id: str) -> str | None:
        """Find the previous version before the current one."""
        try:
            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                cursor = await db_conn.conn.execute(
                    """
                    SELECT id FROM graph_versions
                    WHERE code_graph_id = %s AND created_at < (
                        SELECT created_at FROM graph_versions
                        WHERE code_graph_id = %s AND id = %s
                    )
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (self.code_graph_id, self.code_graph_id, current_version_id),
                )

                row = await cursor.fetchone()
            finally:
                await db_conn.close()

            return row[0] if row else None

        except Exception as e:
            logger.error("Error finding previous version: %s", e)
            return None

    async def _get_version_info(self, version_id: str) -> dict[str, Any] | None:
        """Get detailed information about a version."""
        try:
            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                cursor = await db_conn.conn.execute(
                    "SELECT id, description, created_at FROM graph_versions "
                    "WHERE code_graph_id = %s AND id = %s",
                    (self.code_graph_id, version_id),
                )
                row = await cursor.fetchone()
            finally:
                await db_conn.close()

            if row:
                return {
                    "id": row[0],
                    "description": row[1],
                    "created_at": row[2],
                }
            return None

        except Exception as e:
            logger.error("Error getting version info for %s: %s", version_id, e)
            return None

    async def _perform_version_rollback(self, version_id: str, version_info: dict) -> bool:
        """Perform the actual rollback to a specific version."""
        try:
            # This is a simplified implementation
            # In a real system, you'd need to track version-specific changes
            # and reverse them appropriately

            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                # Remove all versions created after the target version
                await db_conn.conn.execute(
                    """
                    DELETE FROM graph_versions
                    WHERE code_graph_id = %s AND created_at > %s
                """,
                    (self.code_graph_id, version_info["created_at"]),
                )

                # Additional rollback logic would go here
                # This might involve restoring nodes/edges to their state at the
                # target version

                await db_conn.conn.commit()
            finally:
                await db_conn.close()

            return True

        except Exception as e:
            logger.error("Error performing version rollback: %s", e)
            return False

    async def _clean_orphaned_nodes(self, db_conn) -> int:
        """Remove nodes that have no edges and are likely orphaned."""
        try:
            cursor = await db_conn.conn.execute(
                """
                DELETE FROM nodes
                WHERE code_graph_id = %s AND id IN (
                    SELECT n.id FROM nodes n
                    LEFT JOIN edges e1 ON n.code_graph_id = e1.code_graph_id
                        AND n.id = e1.source
                    LEFT JOIN edges e2 ON n.code_graph_id = e2.code_graph_id
                        AND n.id = e2.target
                    WHERE n.code_graph_id = %s
                    AND e1.source IS NULL AND e2.target IS NULL
                    AND n.type NOT IN ('module', 'package')  -- Keep structural nodes
                )
            """,
                (self.code_graph_id, self.code_graph_id),
            )
            return cursor.rowcount

        except Exception as e:
            logger.error("Error cleaning orphaned nodes: %s", e)
            return 0

    async def _clean_invalid_edges(self, db_conn) -> int:
        """Remove edges that reference non-existent nodes."""
        try:
            cursor = await db_conn.conn.execute(
                """
                DELETE FROM edges
                WHERE code_graph_id = %s AND (
                    source NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s)
                    OR target NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s)
                )
            """,
                (self.code_graph_id, self.code_graph_id, self.code_graph_id),
            )
            return cursor.rowcount

        except Exception as e:
            logger.error("Error cleaning invalid edges: %s", e)
            return 0
