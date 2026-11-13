"""Incremental Update Service for Code Graph Management.

This module provides the main orchestration for incrementally updating code graphs
based on file changes in a repository. It coordinates between specialized modules
for change detection, processing, validation, recovery, and configuration.

The service supports:
- Change detection (added, modified, deleted files)
- Incremental graph updates with version tracking
- Version comparison and analysis
- Update validation and verification
- Rollback and recovery capabilities
- Comprehensive configuration management

This refactored version delegates specific responsibilities to focused modules
while maintaining the same public API for backward compatibility.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from processing.analyzers.file_analyzer import FileAnalyzer
from processing.processors.code_processor import CodeProcessor

# Import the focused modules
from services.incremental.incremental_update_change_detector import ChangeDetector
from services.incremental.incremental_update_config import (
    DatabaseUtilities,
    UpdateUtilities,
    VersionManager,
    get_update_config,
)
from services.incremental.incremental_update_processor import IncrementalUpdateProcessor
from services.incremental.incremental_update_recovery import UpdateRecoveryManager
from services.incremental.incremental_update_validator import UpdateValidator

logger = logging.getLogger(__name__)


class IncrementalUpdater:
    """Handles incremental updates to the Code Graph.

    This is the main orchestrator that coordinates between specialized modules
    to perform incremental updates while maintaining data integrity and providing
    comprehensive error handling and recovery capabilities.
    """

    def __init__(
        self,
        code_graph_id: str,
        code_processor: CodeProcessor | None = None,
        file_analyzer: FileAnalyzer | None = None,
    ):
        """Initialize the incremental updater with specialized components.

        Args:
            code_graph_id: The code graph ID for database access
            code_processor: The code processor for parsing files
            file_analyzer: The file analyzer for extracting entities
        """
        self.code_graph_id = code_graph_id
        self.code_processor = code_processor or CodeProcessor()
        self.file_analyzer = file_analyzer or FileAnalyzer()

        # Initialize specialized modules
        self.change_detector = ChangeDetector()
        self.processor = IncrementalUpdateProcessor(code_graph_id, code_processor, file_analyzer)
        self.validator = UpdateValidator(code_graph_id)
        self.recovery_manager = UpdateRecoveryManager(code_graph_id)
        self.version_manager = VersionManager(code_graph_id)
        self.db_utilities = DatabaseUtilities(code_graph_id)
        self.config = get_update_config()

        # Track current operation state
        self.current_version_id: str | None = None
        self.operation_start_time: float | None = None

    async def detect_changes(self, repo_path: Path) -> dict[str, list[Path]]:
        """Detect changes in the repository since last update.

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary with 'added', 'modified', and 'deleted' files
        """
        logger.info("Starting change detection for repository: %s", repo_path)

        # Validate repository path
        if not repo_path.exists() or not repo_path.is_dir():
            raise ValueError(f"Invalid repository path: {repo_path}")

        try:
            changes = await self.change_detector.detect_changes(repo_path)

            # Log change summary
            total_changes = sum(len(file_list) for file_list in changes.values())
            logger.info(
                "Change detection completed: %d total changes (%d added, %d modified, %d deleted)",
                total_changes,
                len(changes["added"]),
                len(changes["modified"]),
                len(changes["deleted"]),
            )

            return changes

        except Exception as e:
            logger.error("Change detection failed: %s", e, exc_info=True)
            raise

    async def update_graph(
        self, repo_path: Path, changes: dict[str, list[Path]] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Update the graph based on detected changes.

        Args:
            repo_path: Path to the repository
            changes: Optional dictionary with detected changes. If None, changes will be detected.

        Returns:
            Tuple of (version_id, stats)
        """
        self.operation_start_time = time.time()

        try:
            # Detect changes if not provided
            if changes is None:
                changes = await self.detect_changes(repo_path)

            # Check if there are any changes to process
            total_changes = sum(len(file_list) for file_list in changes.values())
            if total_changes == 0:
                logger.info("No changes detected, skipping update")
                return await self._create_no_change_version(), {"files_processed": 0}

            # Pre-update validation
            if self.config.get("validation_settings.pre_update_checks", True):
                validation_result = await self.validator.validate_pre_update(repo_path, changes)
                if not validation_result["valid"]:
                    raise ValueError(f"Pre-update validation failed: {validation_result['errors']}")

                if validation_result["warnings"]:
                    logger.warning("Pre-update warnings: %s", validation_result["warnings"])

            # Create backup if enabled
            backup_id = None
            if self.config.get("recovery_settings.auto_backup_before_update", True):
                try:
                    backup_id = await self.recovery_manager.create_backup("pre_update")
                    logger.info("Created backup: %s", backup_id)
                except Exception as e:
                    logger.warning("Failed to create backup: %s", e)

            # Create version for this update
            description = f"Incremental update at {datetime.now(UTC).isoformat()}"
            self.current_version_id = await self.version_manager.create_version(description)

            logger.info("Starting incremental update with version: %s", self.current_version_id)

            # Process the changes
            stats = await self.processor.process_file_changes(
                repo_path, changes, self.current_version_id
            )

            # In-progress validation
            if self.config.get("validation_settings.consistency_checks", True):
                progress_validation = await self.validator.validate_during_update(
                    self.current_version_id, stats
                )
                if not progress_validation["valid"]:
                    logger.warning(
                        "In-progress validation issues: %s",
                        progress_validation["errors"],
                    )

            # Post-update verification
            if self.config.get("validation_settings.post_update_verification", True):
                final_validation = await self.validator.validate_post_update(
                    self.current_version_id, stats
                )
                if not final_validation["valid"]:
                    # Attempt recovery if validation fails
                    await self._handle_update_failure(final_validation["errors"])
                    raise ValueError(f"Post-update validation failed: {final_validation['errors']}")

            # Save updated checksums
            await self.change_detector.save_checksums(repo_path)

            # Add metadata to stats
            stats.update(
                {
                    "version_id": self.current_version_id,
                    "backup_id": backup_id,
                    "total_processing_time": time.time() - self.operation_start_time,
                    "validation_passed": True,
                }
            )

            logger.info(
                "Update completed successfully: version %s, %d files processed in %s",
                self.current_version_id,
                stats.get("files_processed", 0),
                UpdateUtilities.format_duration(stats["total_processing_time"]),
            )

            return self.current_version_id, stats

        except Exception as e:
            logger.error("Update failed: %s", e, exc_info=True)
            await self._handle_update_failure([str(e)])
            raise
        finally:
            self.current_version_id = None
            self.operation_start_time = None

    async def compare_versions(self, version1: str, version2: str) -> dict[str, Any]:
        """Compare two versions of the graph.

        Args:
            version1: First version ID
            version2: Second version ID

        Returns:
            Comparison results
        """
        logger.info("Comparing versions: %s vs %s", version1, version2)

        try:
            comparison_result = await self.validator.compare_versions(version1, version2)

            if not comparison_result.get("valid", False):
                logger.error("Version comparison failed: %s", comparison_result.get("error"))

            return comparison_result

        except Exception as e:
            logger.error("Version comparison failed: %s", e, exc_info=True)
            raise

    async def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific version.

        Args:
            version_id: Version ID to rollback to

        Returns:
            True if rollback was successful
        """
        logger.info("Initiating rollback to version: %s", version_id)

        try:
            success = await self.recovery_manager.rollback_to_version(version_id)

            if success:
                logger.info("Rollback to version %s completed successfully", version_id)
            else:
                logger.error("Rollback to version %s failed", version_id)

            return success

        except Exception as e:
            logger.error("Rollback failed: %s", e, exc_info=True)
            return False

    async def get_database_stats(self) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Returns:
            Database statistics dictionary
        """
        try:
            stats = await self.db_utilities.get_database_stats()

            # Add configuration info
            stats["configuration"] = {
                "auto_backup_enabled": self.config.get(
                    "recovery_settings.auto_backup_before_update"
                ),
                "validation_enabled": self.config.get("validation_settings.pre_update_checks"),
                "supported_extensions": self.config.get("file_tracking.supported_extensions"),
            }

            return stats

        except Exception as e:
            logger.error("Failed to get database stats: %s", e)
            return {"error": str(e)}

    async def vacuum_database(self) -> dict[str, Any]:
        """Vacuum the database to reclaim space.

        Returns:
            Vacuum operation results
        """
        logger.info("Starting database vacuum operation")

        try:
            result = await self.db_utilities.vacuum_database()

            if result["success"]:
                logger.info(
                    "Database vacuum completed: reclaimed %s",
                    UpdateUtilities.format_file_size(result["space_reclaimed"]),
                )
            else:
                logger.error("Database vacuum failed: %s", result.get("error"))

            return result

        except Exception as e:
            logger.error("Database vacuum failed: %s", e)
            return {"success": False, "error": str(e)}

    async def list_backups(self) -> list[dict[str, Any]]:
        """List available backups.

        Returns:
            List of backup metadata
        """
        try:
            return await self.recovery_manager.list_backups()
        except Exception as e:
            logger.error("Failed to list backups: %s", e)
            return []

    async def restore_from_backup(self, backup_id: str) -> bool:
        """Restore from a specific backup.

        Args:
            backup_id: Backup ID to restore from

        Returns:
            True if restoration was successful
        """
        logger.info("Initiating restore from backup: %s", backup_id)

        try:
            success = await self.recovery_manager.restore_from_backup(backup_id)

            if success:
                logger.info("Restore from backup %s completed successfully", backup_id)
            else:
                logger.error("Restore from backup %s failed", backup_id)

            return success

        except Exception as e:
            logger.error("Restore failed: %s", e, exc_info=True)
            return False

    async def get_recovery_log(self) -> list[dict[str, Any]]:
        """Get the recovery operation log.

        Returns:
            List of recovery operations
        """
        try:
            return await self.recovery_manager.get_recovery_log()
        except Exception as e:
            logger.error("Failed to get recovery log: %s", e)
            return []

    async def analyze_performance(self) -> dict[str, Any]:
        """Analyze update performance and provide recommendations.

        Returns:
            Performance analysis results
        """
        try:
            # Get database analysis
            db_analysis = await self.db_utilities.analyze_database()

            # Get configuration recommendations
            config_validation = self.config.validate_config()

            # Build performance recommendations list
            performance_recommendations: list[str] = []
            if db_analysis["statistics"]["node_count"] > 10000:
                performance_recommendations.append(
                    "Consider enabling database indexing for large node count"
                )
            if not self.config.get("database_settings.enable_wal_mode", False):
                performance_recommendations.append(
                    "Enable WAL mode for better concurrent access performance"
                )

            analysis = {
                "database_analysis": db_analysis,
                "configuration_validation": config_validation,
                "performance_recommendations": performance_recommendations,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            return analysis

        except Exception as e:
            logger.error("Performance analysis failed: %s", e)
            return {"error": str(e)}

    # Private helper methods
    async def _create_no_change_version(self) -> str:
        """Create a version entry when no changes are detected."""
        description = f"No changes detected at {datetime.now(UTC).isoformat()}"
        return await self.version_manager.create_version(description)

    async def _handle_update_failure(self, errors: list[str]) -> None:
        """Handle update failure with recovery if enabled."""
        if not self.config.get("recovery_settings.auto_recovery_enabled", True):
            logger.info("Auto-recovery disabled, skipping recovery attempt")
            return

        if self.current_version_id:
            logger.info("Attempting recovery for failed update: %s", self.current_version_id)

            try:
                recovery_result = await self.recovery_manager.recover_from_failed_update(
                    self.current_version_id, "auto"
                )

                if recovery_result["success"]:
                    logger.info("Auto-recovery completed successfully")
                else:
                    logger.error("Auto-recovery failed: %s", recovery_result["errors"])

            except Exception as e:
                logger.error("Recovery attempt failed: %s", e, exc_info=True)

    # Backward compatibility methods (delegated to focused modules)
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum (backward compatibility)."""
        algorithm = self.config.get("update_settings.checksum_algorithm", "md5")
        if not isinstance(algorithm, str):
            algorithm = "md5"
        return UpdateUtilities.calculate_file_checksum(file_path, algorithm)

    def _generate_node_id(self, entity_type: str, entity_name: str, file_path: str) -> str:
        """Generate node ID (backward compatibility)."""
        return UpdateUtilities.generate_node_id(entity_type, entity_name, file_path)

    async def _analyze_file_for_entities(self, file_path: Path) -> list[dict[str, Any]]:
        """Analyze file for entities (backward compatibility)."""
        return await self.processor._analyze_file_for_entities(file_path)

    def _get_db_connection(self):
        """Get database connection (backward compatibility)."""
        return self.processor._get_db_connection()

    # Delegate remaining database operations to processor for backward compatibility
    async def _create_version(self, description: str) -> str:
        """Create version (backward compatibility)."""
        return await self.version_manager.create_version(description)

    async def _get_nodes_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get nodes by file (backward compatibility)."""
        return await self.processor._get_nodes_by_file(file_path)

    async def _add_node(
        self, node_id: str, node_type: str, name: str, properties: dict[str, Any]
    ) -> str:
        """Add node (backward compatibility)."""
        return await self.processor._add_node(node_id, node_type, name, properties)

    async def _update_node(self, node_id: str, properties: dict[str, Any]) -> None:
        """Update node (backward compatibility)."""
        return await self.processor._update_node(node_id, properties)

    async def _add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Add edge (backward compatibility)."""
        return await self.processor._add_edge(source_id, target_id, relationship_type, properties)

    async def _delete_node(self, node_id: str) -> None:
        """Delete node (backward compatibility)."""
        return await self.processor._delete_node(node_id)

    async def _delete_edge(self, source_id: str, target_id: str, relationship_type: str) -> None:
        """Delete edge (backward compatibility)."""
        return await self.processor._delete_edge(source_id, target_id, relationship_type)

    async def _get_edges_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Get edges for node (backward compatibility)."""
        return await self.processor._get_edges_for_node(node_id)

    async def _delete_nodes_by_file(self, file_path: str) -> int:
        """Delete nodes by file (backward compatibility)."""
        return await self.processor._delete_nodes_by_file(file_path)

    async def _get_version(self, version_id: str) -> dict[str, Any] | None:
        """Get version (backward compatibility)."""
        return await self.version_manager.get_version(version_id)

    async def _get_related_nodes(self, nodes: list[dict[str, Any]]) -> set[str]:
        """Get related nodes (backward compatibility)."""
        return await self.processor._get_related_nodes(nodes)
