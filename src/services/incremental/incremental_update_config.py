"""Update Configuration and Utilities Module.

This module provides configuration management and utility functions for
incremental updates. It handles settings, version management, database
operations, and common utility functions used across update modules.

The module supports:
- Configuration management and validation
- Version creation and tracking
- Database utility operations
- Common helper functions
- Update metrics and statistics
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config.core_config_settings import get_settings
from db.facade import GraphDatabaseConnection

settings = get_settings()
logger = logging.getLogger(__name__)


class UpdateConfig:
    """Configuration management for incremental updates."""

    def __init__(self):
        """Initialize the configuration."""
        self.settings = get_settings()
        self._config = self._load_default_config()

    def _load_default_config(self) -> dict[str, Any]:
        """Load default configuration settings."""
        return {
            "update_settings": {
                "max_parallel_files": 10,
                "max_processing_time": 3600,  # 1 hour
                "auto_backup": True,
                "validation_enabled": True,
                "recovery_enabled": True,
                "checksum_algorithm": "md5",
                "batch_size": 100,
            },
            "validation_settings": {
                "pre_update_checks": True,
                "post_update_verification": True,
                "consistency_checks": True,
                "performance_monitoring": True,
                "error_threshold": 0.1,  # 10% error rate threshold
            },
            "recovery_settings": {
                "auto_backup_before_update": True,
                "max_backup_retention_days": 30,
                "auto_recovery_enabled": True,
                "rollback_timeout": 300,  # 5 minutes
            },
            "file_tracking": {
                "supported_extensions": [".py", ".js", ".ts", ".java", ".cpp", ".hpp"],
                "excluded_directories": [
                    ".git",
                    ".venv",
                    "venv",
                    "env",
                    ".env",
                    "__pycache__",
                    "node_modules",
                    ".pytest_cache",
                ],
                "max_file_size": 10 * 1024 * 1024,  # 10MB
            },
            "database_settings": {
                "connection_timeout": 30,
                "query_timeout": 60,
                "batch_commit_size": 1000,
                "enable_wal_mode": True,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def validate_config(self) -> dict[str, Any]:
        """Validate the current configuration.

        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Validate update settings
        max_parallel = self.get("update_settings.max_parallel_files")
        if not isinstance(max_parallel, int) or max_parallel < 1:
            validation_result["errors"].append("max_parallel_files must be a positive integer")
            validation_result["valid"] = False

        max_time = self.get("update_settings.max_processing_time")
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            validation_result["errors"].append("max_processing_time must be a positive number")
            validation_result["valid"] = False

        # Validate file tracking settings
        max_file_size = self.get("file_tracking.max_file_size")
        if not isinstance(max_file_size, int) or max_file_size <= 0:
            validation_result["errors"].append("max_file_size must be a positive integer")
            validation_result["valid"] = False

        # Validate database settings
        timeout = self.get("database_settings.connection_timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            validation_result["errors"].append("connection_timeout must be a positive number")
            validation_result["valid"] = False

        return validation_result


class VersionManager:
    """Manages version creation and tracking."""

    def __init__(self, code_graph_id: str):
        """Initialize the version manager.

        Args:
            code_graph_id: The code graph ID for database access
        """
        self.code_graph_id = code_graph_id

    def _get_db_connection(self):
        """Get database connection for the code graph."""
        return GraphDatabaseConnection(self.code_graph_id)

    async def create_version(self, description: str) -> str:
        """Create a new version using direct database access.

        Args:
            description: Description of the version

        Returns:
            Generated version ID
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            version_id = f"v_{int(time.time())}"
            await db_conn.conn.execute(
                """
                INSERT INTO graph_versions (code_graph_id, id, description)
                VALUES (%s, %s, %s)
                """,
                (self.code_graph_id, version_id, description),
            )
            await db_conn.conn.commit()
            return version_id
        finally:
            await db_conn.close()

    async def get_version(self, version_id: str) -> dict[str, Any] | None:
        """Get version details.

        Args:
            version_id: Version ID to retrieve

        Returns:
            Version details dictionary or None if not found
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            cursor = await db_conn.conn.execute(
                "SELECT id, description, created_at FROM graph_versions "
                "WHERE code_graph_id = %s AND id = %s",
                (self.code_graph_id, version_id),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "description": row[1],
                    "created_at": row[2],
                }
            return None
        finally:
            await db_conn.close()

    async def list_versions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """List all versions.

        Args:
            limit: Maximum number of versions to return

        Returns:
            List of version dictionaries
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            query = (
                "SELECT id, description, created_at FROM graph_versions "
                "WHERE code_graph_id = %s ORDER BY created_at DESC"
            )
            params: list[Any] = [self.code_graph_id]
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cursor = await db_conn.conn.execute(query, params)
            versions = []
            for row in await cursor.fetchall():
                versions.append(
                    {
                        "id": row[0],
                        "description": row[1],
                        "created_at": row[2],
                    }
                )
            return versions
        finally:
            await db_conn.close()

    async def delete_version(self, version_id: str) -> bool:
        """Delete a version.

        Args:
            version_id: Version ID to delete

        Returns:
            True if version was deleted, False if not found
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            cursor = await db_conn.conn.execute(
                "DELETE FROM graph_versions WHERE code_graph_id = %s AND id = %s",
                (self.code_graph_id, version_id),
            )
            await db_conn.conn.commit()
            return cursor.rowcount > 0
        finally:
            await db_conn.close()


class DatabaseUtilities:
    """Database utility operations for incremental updates."""

    def __init__(self, code_graph_id: str):
        """Initialize database utilities.

        Args:
            code_graph_id: The code graph ID for database access
        """
        self.code_graph_id = code_graph_id

    def _get_db_connection(self):
        """Get database connection for the code graph."""
        return GraphDatabaseConnection(self.code_graph_id)

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            cgid = self.code_graph_id
            stats = {}

            # Node count
            cursor = await db_conn.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (cgid,)
            )
            stats["node_count"] = (await cursor.fetchone())[0]

            # Edge count
            cursor = await db_conn.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (cgid,)
            )
            stats["edge_count"] = (await cursor.fetchone())[0]

            # Version count
            cursor = await db_conn.conn.execute(
                "SELECT COUNT(*) FROM graph_versions WHERE code_graph_id = %s", (cgid,)
            )
            stats["version_count"] = (await cursor.fetchone())[0]

            # Node types distribution
            cursor = await db_conn.conn.execute(
                """
                SELECT type, COUNT(*) as count
                FROM nodes
                WHERE code_graph_id = %s
                GROUP BY type
                ORDER BY count DESC
            """,
                (cgid,),
            )
            stats["node_types"] = dict(await cursor.fetchall())

            # Edge types distribution
            cursor = await db_conn.conn.execute(
                """
                SELECT relationship, COUNT(*) as count
                FROM edges
                WHERE code_graph_id = %s
                GROUP BY relationship
                ORDER BY count DESC
            """,
                (cgid,),
            )
            stats["edge_types"] = dict(await cursor.fetchall())

            # No per-graph file under shared PostgreSQL.
            stats["file_size"] = 0

            stats["timestamp"] = datetime.now(UTC).isoformat()

            return stats
        finally:
            await db_conn.close()

    async def vacuum_database(self) -> dict[str, Any]:
        """No-op under shared PostgreSQL.

        Space reclamation is handled by autovacuum; an explicit ``VACUUM``
        cannot run inside the pooled (PgBouncer transaction-mode) connection.
        """
        return {
            "success": True,
            "size_before": 0,
            "size_after": 0,
            "space_reclaimed": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def analyze_database(self) -> dict[str, Any]:
        """Analyze database for optimization opportunities.

        Returns:
            Analysis results
        """
        db_conn = self._get_db_connection()
        try:
            await db_conn.connect()
            cgid = self.code_graph_id
            analysis = {
                "recommendations": [],
                "statistics": {},
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Check for indexes
            cursor = await db_conn.conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename IN ('nodes', 'edges')"
            )
            indexes = [row[0] for row in await cursor.fetchall()]
            analysis["statistics"]["index_count"] = len(indexes)

            # Check table sizes
            cursor = await db_conn.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (cgid,)
            )
            node_count = (await cursor.fetchone())[0]

            cursor = await db_conn.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (cgid,)
            )
            edge_count = (await cursor.fetchone())[0]

            analysis["statistics"]["node_count"] = node_count
            analysis["statistics"]["edge_count"] = edge_count

            # Generate recommendations
            if node_count > 10000 and "idx_node_cg_type" not in indexes:
                analysis["recommendations"].append(
                    "Consider adding index on nodes.type for better query performance"
                )

            if edge_count > 10000 and "idx_edge_cg_source" not in indexes:
                analysis["recommendations"].append(
                    "Consider adding index on edges.source for better query performance"
                )

            if node_count > 50000:
                analysis["recommendations"].append(
                    "Consider partitioning large tables for better performance"
                )

            return analysis
        finally:
            await db_conn.close()


class UpdateUtilities:
    """Common utility functions for incremental updates."""

    @staticmethod
    def generate_node_id(entity_type: str, entity_name: str, file_path: str) -> str:
        """Generate a consistent node ID.

        Args:
            entity_type: Type of the entity
            entity_name: Name of the entity
            file_path: Path to the file containing the entity

        Returns:
            Generated node ID
        """
        content = f"{entity_type}:{entity_name}:{file_path}"
        hash_obj = hashlib.md5(content.encode())  # nosec B324
        return f"{entity_type}_{hash_obj.hexdigest()[:8]}"

    @staticmethod
    def calculate_file_checksum(file_path: Path, algorithm: str = "md5") -> str:
        """Calculate checksum of a file.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use

        Returns:
            Checksum as hex string
        """
        if algorithm.lower() == "md5":
            hash_obj = hashlib.md5()  # nosec B324
        elif algorithm.lower() == "sha256":
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to calculate checksum for %s: %s", file_path, e)
            # Return a timestamp-based fallback checksum
            return f"fallback_{int(time.time())}"

        return hash_obj.hexdigest()

    @staticmethod
    def is_supported_file(file_path: Path, config: UpdateConfig) -> bool:
        """Check if a file is supported for tracking.

        Args:
            file_path: Path to the file
            config: Update configuration

        Returns:
            True if file is supported
        """
        # Check file extension
        supported_extensions = config.get("file_tracking.supported_extensions", [".py"])
        if file_path.suffix not in supported_extensions:
            return False

        # Check file size
        max_size = config.get("file_tracking.max_file_size", 10 * 1024 * 1024)
        try:
            if file_path.stat().st_size > max_size:
                return False
        except OSError:
            return False

        # Check if in excluded directory
        excluded_dirs = config.get("file_tracking.excluded_directories", [])
        for part in file_path.parts:
            if part in excluded_dirs:
                return False

        return True

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in bytes to human-readable string.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        size: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    @staticmethod
    def validate_version_id(version_id: str) -> bool:
        """Validate a version ID format.

        Args:
            version_id: Version ID to validate

        Returns:
            True if valid
        """
        # Basic validation - should start with 'v_' and contain only valid characters
        if not version_id.startswith("v_"):
            return False

        remainder = version_id[2:]
        if not remainder.replace("_", "").replace("-", "").isalnum():
            return False

        return True

    @staticmethod
    def create_statistics_summary(stats: dict[str, Any]) -> dict[str, Any]:
        """Create a summary of update statistics.

        Args:
            stats: Raw statistics dictionary

        Returns:
            Summary statistics
        """
        summary = {
            "total_operations": (
                stats.get("added_nodes", 0)
                + stats.get("modified_nodes", 0)
                + stats.get("deleted_nodes", 0)
                + stats.get("added_edges", 0)
                + stats.get("deleted_edges", 0)
            ),
            "node_operations": (
                stats.get("added_nodes", 0)
                + stats.get("modified_nodes", 0)
                + stats.get("deleted_nodes", 0)
            ),
            "edge_operations": (stats.get("added_edges", 0) + stats.get("deleted_edges", 0)),
            "files_processed": stats.get("files_processed", 0),
            "processing_time": stats.get("processing_time", 0.0),
        }

        # Calculate rates
        if summary["processing_time"] > 0:
            summary["operations_per_second"] = (
                summary["total_operations"] / summary["processing_time"]
            )
            summary["files_per_second"] = summary["files_processed"] / summary["processing_time"]
        else:
            summary["operations_per_second"] = 0
            summary["files_per_second"] = 0

        # Add formatted strings
        summary["formatted_duration"] = UpdateUtilities.format_duration(summary["processing_time"])

        return summary


# Singleton instance for global configuration
_config_instance = None


def get_update_config() -> UpdateConfig:
    """Get the global update configuration instance.

    Returns:
        UpdateConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = UpdateConfig()
    return _config_instance
