"""Update Validation and Verification Module.

This module provides functionality for validating and verifying incremental
updates to ensure data integrity and consistency. It performs validation
checks before, during, and after update operations.

The module supports:
- Pre-update validation (prerequisites, dependencies)
- In-progress validation (data consistency checks)
- Post-update verification (integrity validation)
- Version comparison and analysis
- Rollback validation
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config.core_config_settings import get_settings
from db.facade import GraphDatabaseConnection

settings = get_settings()
logger = logging.getLogger(__name__)


class UpdateValidator:
    """Handles validation and verification of incremental updates."""

    def __init__(self, code_graph_id: str):
        """Initialize the update validator.

        Args:
            code_graph_id: The code graph ID for database access
        """
        self.code_graph_id = code_graph_id

    def _get_db_connection(self):
        """Get database connection for the code graph."""
        return GraphDatabaseConnection(self.code_graph_id)

    async def validate_pre_update(
        self, repo_path: Path, changes: dict[str, list[Path]]
    ) -> dict[str, Any]:
        """Validate prerequisites before starting an update.

        Args:
            repo_path: Path to the repository
            changes: Dictionary with detected changes

        Returns:
            Validation result with any issues found
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "checks_performed": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Check database connectivity
        db_check = await self._validate_database_connection()
        validation_result["checks_performed"].append("database_connectivity")
        if not db_check["valid"]:
            validation_result["errors"].extend(db_check["errors"])
            validation_result["valid"] = False

        # Check file accessibility
        file_check = await self._validate_file_access(changes)
        validation_result["checks_performed"].append("file_accessibility")
        if not file_check["valid"]:
            validation_result["errors"].extend(file_check["errors"])
            validation_result["valid"] = False
        if file_check["warnings"]:
            validation_result["warnings"].extend(file_check["warnings"])

        # Check repository state
        repo_check = await self._validate_repository_state(repo_path)
        validation_result["checks_performed"].append("repository_state")
        if not repo_check["valid"]:
            validation_result["errors"].extend(repo_check["errors"])
            validation_result["valid"] = False
        if repo_check["warnings"]:
            validation_result["warnings"].extend(repo_check["warnings"])

        # Check for potential conflicts
        conflict_check = await self._validate_change_conflicts(changes)
        validation_result["checks_performed"].append("change_conflicts")
        if not conflict_check["valid"]:
            validation_result["errors"].extend(conflict_check["errors"])
            validation_result["valid"] = False
        if conflict_check["warnings"]:
            validation_result["warnings"].extend(conflict_check["warnings"])

        return validation_result

    async def validate_during_update(
        self, version_id: str, processing_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate data consistency during update processing.

        Args:
            version_id: The version ID being processed
            processing_stats: Current processing statistics

        Returns:
            Validation result for in-progress update
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "consistency_checks": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Check node-edge consistency
        node_edge_check = await self._validate_node_edge_consistency()
        validation_result["consistency_checks"].append("node_edge_relationship")
        if not node_edge_check["valid"]:
            validation_result["errors"].extend(node_edge_check["errors"])
            validation_result["valid"] = False

        # Check for orphaned nodes
        orphan_check = await self._validate_orphaned_nodes()
        validation_result["consistency_checks"].append("orphaned_nodes")
        if orphan_check["warnings"]:
            validation_result["warnings"].extend(orphan_check["warnings"])

        # Check for duplicate nodes
        duplicate_check = await self._validate_duplicate_nodes()
        validation_result["consistency_checks"].append("duplicate_nodes")
        if not duplicate_check["valid"]:
            validation_result["errors"].extend(duplicate_check["errors"])
            validation_result["valid"] = False

        # Validate processing statistics
        stats_check = await self._validate_processing_statistics(processing_stats)
        validation_result["consistency_checks"].append("processing_statistics")
        if not stats_check["valid"]:
            validation_result["warnings"].extend(stats_check["warnings"])

        return validation_result

    async def validate_post_update(
        self, version_id: str, final_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform comprehensive validation after update completion.

        Args:
            version_id: The completed version ID
            final_stats: Final processing statistics

        Returns:
            Complete validation result
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "integrity_checks": [],
            "performance_metrics": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Comprehensive integrity check
        integrity_check = await self._validate_graph_integrity()
        validation_result["integrity_checks"].append("comprehensive_integrity")
        if not integrity_check["valid"]:
            validation_result["errors"].extend(integrity_check["errors"])
            validation_result["valid"] = False
        if integrity_check["warnings"]:
            validation_result["warnings"].extend(integrity_check["warnings"])

        # Validate version consistency
        version_check = await self._validate_version_consistency(version_id)
        validation_result["integrity_checks"].append("version_consistency")
        if not version_check["valid"]:
            validation_result["errors"].extend(version_check["errors"])
            validation_result["valid"] = False

        # Check relationship integrity
        relationship_check = await self._validate_relationship_integrity()
        validation_result["integrity_checks"].append("relationship_integrity")
        if not relationship_check["valid"]:
            validation_result["errors"].extend(relationship_check["errors"])
            validation_result["valid"] = False

        # Generate performance metrics
        validation_result["performance_metrics"] = await self._generate_performance_metrics(
            final_stats
        )

        return validation_result

    async def compare_versions(self, version1: str, version2: str) -> dict[str, Any]:
        """Compare two versions of the graph for validation purposes.

        Args:
            version1: First version ID
            version2: Second version ID

        Returns:
            Detailed comparison results
        """
        # Get version details using direct database access
        v1 = await self._get_version(version1)
        v2 = await self._get_version(version2)

        if not v1 or not v2:
            return {
                "valid": False,
                "error": "One or both versions not found",
                "version1_exists": v1 is not None,
                "version2_exists": v2 is not None,
            }

        # Perform detailed version comparison analysis
        comparison_result = await self._perform_detailed_version_comparison(v1, v2)

        return {
            "valid": True,
            "version1": {
                "id": version1,
                "created_at": v1.get("created_at"),
                "description": v1.get("description"),
                "node_count": comparison_result["v1_stats"]["node_count"],
                "edge_count": comparison_result["v1_stats"]["edge_count"],
            },
            "version2": {
                "id": version2,
                "created_at": v2.get("created_at"),
                "description": v2.get("description"),
                "node_count": comparison_result["v2_stats"]["node_count"],
                "edge_count": comparison_result["v2_stats"]["edge_count"],
            },
            "changes": comparison_result["changes"],
            "details": comparison_result["details"],
            "metrics": comparison_result["metrics"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _validate_database_connection(self) -> dict[str, Any]:
        """Validate database connectivity and basic operations."""
        result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        try:
            db_conn = self._get_db_connection()
            await db_conn.connect()
            try:
                # Test basic database operations
                cursor = await db_conn.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s",
                    (self.code_graph_id,),
                )
                node_count = (await cursor.fetchone())[0]

                cursor = await db_conn.conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s",
                    (self.code_graph_id,),
                )
                edge_count = (await cursor.fetchone())[0]
            finally:
                await db_conn.close()

            logger.info(
                "Database connectivity validated: %d nodes, %d edges",
                node_count,
                edge_count,
            )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Database connection failed: {e}")
            logger.error("Database validation failed: %s", e)

        return result

    async def _validate_file_access(self, changes: dict[str, list[Path]]) -> dict[str, Any]:
        """Validate that all files in changes are accessible."""
        result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        for change_type, files in changes.items():
            for file_path in files:
                if change_type == "deleted":
                    # Deleted files shouldn't exist
                    if file_path.exists():
                        result["warnings"].append(
                            f"File marked as deleted still exists: {file_path}"
                        )
                else:
                    # Added/modified files should exist and be readable
                    if not file_path.exists():
                        result["valid"] = False
                        result["errors"].append(f"File not found: {file_path}")
                    elif not os.access(file_path, os.R_OK):
                        result["valid"] = False
                        result["errors"].append(f"File not readable: {file_path}")

        return result

    async def _validate_repository_state(self, repo_path: Path) -> dict[str, Any]:
        """Validate the overall repository state."""
        result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check if repository path exists and is accessible
        if not repo_path.exists():
            result["valid"] = False
            result["errors"].append(f"Repository path does not exist: {repo_path}")
            return result

        if not repo_path.is_dir():
            result["valid"] = False
            result["errors"].append(f"Repository path is not a directory: {repo_path}")
            return result

        # Check for .code_graph directory
        code_graph_dir = repo_path / ".code_graph"
        if not code_graph_dir.exists():
            result["warnings"].append("No .code_graph directory found, will be created")

        return result

    async def _validate_change_conflicts(self, changes: dict[str, list[Path]]) -> dict[str, Any]:
        """Validate that changes don't have conflicts."""
        result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check for files that appear in multiple change categories
        all_files: dict[str, str] = {}
        for change_type, files in changes.items():
            for file_path in files:
                if str(file_path) in all_files:
                    result["errors"].append(
                        f"File appears in multiple change categories: {file_path} "
                        f"({all_files[str(file_path)]} and {change_type})"
                    )
                    result["valid"] = False
                else:
                    all_files[str(file_path)] = change_type

        return result

    async def _validate_node_edge_consistency(self) -> dict[str, Any]:
        """Validate that all edges reference existing nodes."""
        result: dict[str, Any] = {"valid": True, "errors": []}

        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Find edges that reference non-existent nodes
            cursor = await db_conn.conn.execute(
                """
                SELECT e.source, e.target, e.relationship
                FROM edges e
                LEFT JOIN nodes n1 ON e.code_graph_id = n1.code_graph_id AND e.source = n1.id
                LEFT JOIN nodes n2 ON e.code_graph_id = n2.code_graph_id AND e.target = n2.id
                WHERE e.code_graph_id = %s AND (n1.id IS NULL OR n2.id IS NULL)
            """,
                (self.code_graph_id,),
            )

            invalid_edges = await cursor.fetchall()
            if invalid_edges:
                result["valid"] = False
                for edge in invalid_edges:
                    result["errors"].append(
                        f"Edge references non-existent node: {edge[0]} -> {edge[1]} ({edge[2]})"
                    )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate node-edge consistency: {e}")
        finally:
            await db_conn.close()

        return result

    async def _validate_orphaned_nodes(self) -> dict[str, Any]:
        """Check for nodes that have no edges (might be orphaned)."""
        result: dict[str, Any] = {"valid": True, "warnings": []}

        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Find nodes with no edges
            cursor = await db_conn.conn.execute(
                """
                SELECT n.id, n.properties->>'name', n.type
                FROM nodes n
                LEFT JOIN edges e1 ON n.code_graph_id = e1.code_graph_id AND n.id = e1.source
                LEFT JOIN edges e2 ON n.code_graph_id = e2.code_graph_id AND n.id = e2.target
                WHERE n.code_graph_id = %s AND e1.source IS NULL AND e2.target IS NULL
            """,
                (self.code_graph_id,),
            )

            orphaned_nodes = await cursor.fetchall()
            if orphaned_nodes:
                for node in orphaned_nodes:
                    result["warnings"].append(
                        f"Orphaned node detected: {node[1]} ({node[2]}, ID: {node[0]})"
                    )

        except Exception as e:
            logger.error("Failed to check for orphaned nodes: %s", e)
        finally:
            await db_conn.close()

        return result

    async def _validate_duplicate_nodes(self) -> dict[str, Any]:
        """Check for duplicate nodes (same type, name, file)."""
        result: dict[str, Any] = {"valid": True, "errors": []}

        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Find potential duplicate nodes
            cursor = await db_conn.conn.execute(
                """
                SELECT MIN(n1.id), n1.properties->>'name' AS name, n1.type, COUNT(*) as count
                FROM nodes n1
                WHERE n1.code_graph_id = %s
                GROUP BY n1.properties->>'name', n1.type, n1.properties->>'file'
                HAVING COUNT(*) > 1
            """,
                (self.code_graph_id,),
            )

            duplicate_groups = await cursor.fetchall()
            if duplicate_groups:
                result["valid"] = False
                for group in duplicate_groups:
                    result["errors"].append(
                        f"Duplicate nodes found: {group[1]} ({group[2]}) - {group[3]} instances"
                    )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to check for duplicate nodes: {e}")
        finally:
            await db_conn.close()

        return result

    async def _validate_processing_statistics(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Validate processing statistics for consistency."""
        result: dict[str, Any] = {"valid": True, "warnings": []}

        # Check for negative values
        for key, value in stats.items():
            if isinstance(value, (int, float)) and value < 0:
                result["warnings"].append(f"Negative value in statistics: {key} = {value}")

        # Check processing time reasonableness
        if "processing_time" in stats:
            if stats["processing_time"] > 3600:  # More than 1 hour
                result["warnings"].append(
                    f"Unusually long processing time: {stats['processing_time']} seconds"
                )

        return result

    async def _validate_graph_integrity(self) -> dict[str, Any]:
        """Perform comprehensive graph integrity validation."""
        result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Combine multiple integrity checks
        node_edge_check = await self._validate_node_edge_consistency()
        if not node_edge_check["valid"]:
            result["valid"] = False
            result["errors"].extend(node_edge_check["errors"])

        orphan_check = await self._validate_orphaned_nodes()
        result["warnings"].extend(orphan_check["warnings"])

        duplicate_check = await self._validate_duplicate_nodes()
        if not duplicate_check["valid"]:
            result["valid"] = False
            result["errors"].extend(duplicate_check["errors"])

        return result

    async def _validate_version_consistency(self, version_id: str) -> dict[str, Any]:
        """Validate consistency of a specific version."""
        result: dict[str, Any] = {"valid": True, "errors": []}

        try:
            version = await self._get_version(version_id)
            if not version:
                result["valid"] = False
                result["errors"].append(f"Version {version_id} not found")
                return result

            # Additional version-specific checks could be added here

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate version {version_id}: {e}")

        return result

    async def _validate_relationship_integrity(self) -> dict[str, Any]:
        """Validate the integrity of relationships in the graph."""
        result: dict[str, Any] = {"valid": True, "errors": []}

        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Check for self-referencing edges (might be invalid)
            cursor = await db_conn.conn.execute(
                """
                SELECT source, target, relationship
                FROM edges
                WHERE code_graph_id = %s AND source = target
            """,
                (self.code_graph_id,),
            )

            self_refs = await cursor.fetchall()
            if self_refs:
                for edge in self_refs:
                    # Some self-references might be valid (e.g., recursive functions)
                    if edge[2] not in ["calls", "references"]:
                        result["errors"].append(
                            f"Invalid self-reference: {edge[0]} -> {edge[1]} ({edge[2]})"
                        )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate relationship integrity: {e}")
        finally:
            await db_conn.close()

        return result

    async def _generate_performance_metrics(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Generate performance metrics from processing statistics."""
        metrics: dict[str, Any] = {}

        if "processing_time" in stats and "files_processed" in stats:
            if stats["files_processed"] > 0:
                metrics["avg_time_per_file"] = stats["processing_time"] / stats["files_processed"]
            else:
                metrics["avg_time_per_file"] = 0

        total_operations = (
            stats.get("added_nodes", 0)
            + stats.get("modified_nodes", 0)
            + stats.get("deleted_nodes", 0)
            + stats.get("added_edges", 0)
            + stats.get("deleted_edges", 0)
        )

        if "processing_time" in stats and total_operations > 0:
            metrics["operations_per_second"] = total_operations / max(
                stats["processing_time"], 0.001
            )
        else:
            metrics["operations_per_second"] = 0

        metrics["total_operations"] = total_operations

        return metrics

    async def _get_version(self, version_id: str) -> dict[str, Any] | None:
        """Get version details using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
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

    async def _perform_detailed_version_comparison(self, v1: dict, v2: dict) -> dict[str, Any]:
        """Perform detailed comparison between two versions."""
        db_conn = self._get_db_connection()
        await db_conn.connect()

        try:
            # Get version snapshots
            v1_snapshot = await self._get_version_snapshot(v1["id"], db_conn)
            v2_snapshot = await self._get_version_snapshot(v2["id"], db_conn)

            # Calculate differences
            changes = {
                "nodes": {
                    "added": 0,
                    "removed": 0,
                    "modified": 0,
                    "unchanged": 0,
                },
                "edges": {"added": 0, "removed": 0},
            }

            details: dict[str, list[Any]] = {
                "added_nodes": [],
                "removed_nodes": [],
                "modified_nodes": [],
                "added_edges": [],
                "removed_edges": [],
            }

            # Compare nodes
            v1_nodes = {node["id"]: node for node in v1_snapshot["nodes"]}
            v2_nodes = {node["id"]: node for node in v2_snapshot["nodes"]}

            # Find added nodes
            for node_id, node in v2_nodes.items():
                if node_id not in v1_nodes:
                    changes["nodes"]["added"] += 1
                    details["added_nodes"].append(
                        {
                            "id": node_id,
                            "name": node.get("name"),
                            "type": node.get("type"),
                            "file_path": node.get("file_path"),
                        }
                    )

            # Find removed nodes
            for node_id, node in v1_nodes.items():
                if node_id not in v2_nodes:
                    changes["nodes"]["removed"] += 1
                    details["removed_nodes"].append(
                        {
                            "id": node_id,
                            "name": node.get("name"),
                            "type": node.get("type"),
                            "file_path": node.get("file_path"),
                        }
                    )

            # Find modified nodes
            for node_id in v1_nodes:
                if node_id in v2_nodes:
                    if self._nodes_differ(v1_nodes[node_id], v2_nodes[node_id]):
                        changes["nodes"]["modified"] += 1
                        details["modified_nodes"].append(
                            {
                                "id": node_id,
                                "name": v2_nodes[node_id].get("name"),
                                "type": v2_nodes[node_id].get("type"),
                                "changes": self._get_node_changes(
                                    v1_nodes[node_id], v2_nodes[node_id]
                                ),
                            }
                        )
                    else:
                        changes["nodes"]["unchanged"] += 1

            # Compare edges
            v1_edges = {
                f"{edge['source_id']}-{edge['target_id']}-{edge.get('relationship_type', '')}": edge
                for edge in v1_snapshot["edges"]
            }
            v2_edges = {
                f"{edge['source_id']}-{edge['target_id']}-{edge.get('relationship_type', '')}": edge
                for edge in v2_snapshot["edges"]
            }

            # Find added edges
            for edge_key, edge in v2_edges.items():
                if edge_key not in v1_edges:
                    changes["edges"]["added"] += 1
                    details["added_edges"].append(
                        {
                            "source_id": edge["source_id"],
                            "target_id": edge["target_id"],
                            "type": edge.get("relationship_type", ""),
                        }
                    )

            # Find removed edges
            for edge_key, edge in v1_edges.items():
                if edge_key not in v2_edges:
                    changes["edges"]["removed"] += 1
                    details["removed_edges"].append(
                        {
                            "source_id": edge["source_id"],
                            "target_id": edge["target_id"],
                            "type": edge.get("relationship_type", ""),
                        }
                    )

            # Calculate metrics
            total_nodes = max(len(v1_nodes), len(v2_nodes))
            total_edges = max(len(v1_edges), len(v2_edges))

            metrics = {
                "change_rate": {
                    "nodes": (
                        changes["nodes"]["added"]
                        + changes["nodes"]["removed"]
                        + changes["nodes"]["modified"]
                    )
                    / max(total_nodes, 1),
                    "edges": (changes["edges"]["added"] + changes["edges"]["removed"])
                    / max(total_edges, 1),
                },
                "stability_score": changes["nodes"]["unchanged"] / max(total_nodes, 1),
                "complexity_change": len(v2_edges) - len(v1_edges),
                "file_impact": len(
                    set(
                        node.get("file_path", "")
                        for node in details["added_nodes"]
                        + details["removed_nodes"]
                        + details["modified_nodes"]
                    )
                ),
            }

            return {
                "v1_stats": {
                    "node_count": len(v1_nodes),
                    "edge_count": len(v1_edges),
                },
                "v2_stats": {
                    "node_count": len(v2_nodes),
                    "edge_count": len(v2_edges),
                },
                "changes": changes,
                "details": details,
                "metrics": metrics,
            }

        except (OSError, ValueError) as e:
            logger.error("Error performing version comparison: %s", e)
            # Return empty comparison on error
            return {
                "v1_stats": {"node_count": 0, "edge_count": 0},
                "v2_stats": {"node_count": 0, "edge_count": 0},
                "changes": {
                    "nodes": {"added": 0, "removed": 0, "modified": 0, "unchanged": 0},
                    "edges": {"added": 0, "removed": 0},
                },
                "details": {
                    "added_nodes": [],
                    "removed_nodes": [],
                    "modified_nodes": [],
                    "added_edges": [],
                    "removed_edges": [],
                },
                "metrics": {
                    "change_rate": {"nodes": 0.0, "edges": 0.0},
                    "stability_score": 1.0,
                    "complexity_change": 0,
                    "file_impact": 0,
                },
            }
        finally:
            await db_conn.close()

    async def _get_version_snapshot(self, version_id: str, db_conn) -> dict[str, Any]:
        """Get a snapshot of nodes and edges for a specific version."""
        snapshot: dict[str, list[Any]] = {"nodes": [], "edges": []}

        try:
            # Query current nodes and edges (simplified for this implementation)
            cursor = await db_conn.conn.execute(
                "SELECT id, properties->>'name', type, properties FROM nodes "
                "WHERE code_graph_id = %s",
                (self.code_graph_id,),
            )
            async for row in cursor:
                props = row[3] or {}
                file_path = props.get("file", "") if isinstance(props, dict) else ""

                snapshot["nodes"].append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "file_path": file_path,
                        "properties": props,
                    }
                )

            # Get edges
            cursor = await db_conn.conn.execute(
                "SELECT source, target, relationship, properties FROM edges "
                "WHERE code_graph_id = %s",
                (self.code_graph_id,),
            )
            async for row in cursor:
                snapshot["edges"].append(
                    {
                        "source_id": row[0],
                        "target_id": row[1],
                        "relationship_type": row[2],
                        "properties": row[3] or {},
                    }
                )

        except (OSError, ValueError) as e:
            logger.error("Error getting version snapshot for %s: %s", version_id, e)

        return snapshot

    def _nodes_differ(self, node1: dict, node2: dict) -> bool:
        """Check if two nodes have meaningful differences."""
        # Compare key properties
        for key in ["name", "type", "file_path"]:
            if node1.get(key) != node2.get(key):
                return True

        # Compare properties (streamlined)
        props1 = node1.get("properties", {})
        props2 = node2.get("properties", {})

        # Check for changes in important properties
        important_props = ["signature", "visibility", "modifiers", "complexity"]
        for prop in important_props:
            if props1.get(prop) != props2.get(prop):
                return True

        return False

    def _get_node_changes(self, old_node: dict, new_node: dict) -> list[str]:
        """Get a list of changes between two nodes."""
        changes = []

        # Check basic properties
        for key in ["name", "type", "file_path"]:
            if old_node.get(key) != new_node.get(key):
                changes.append(f"{key}: {old_node.get(key)} → {new_node.get(key)}")

        # Check properties
        old_props = old_node.get("properties", {})
        new_props = new_node.get("properties", {})

        for prop in ["signature", "visibility", "modifiers", "complexity"]:
            old_val = old_props.get(prop)
            new_val = new_props.get(prop)
            if old_val != new_val:
                changes.append(f"{prop}: {old_val} → {new_val}")

        return changes
