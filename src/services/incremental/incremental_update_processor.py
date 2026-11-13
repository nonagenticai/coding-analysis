"""Incremental Update Processing Module.

This module handles the core processing logic for incremental updates to the
code graph. It manages the analysis of changed files, updating nodes and edges,
and maintaining graph consistency during incremental updates.

The module supports:
- File analysis and entity extraction
- Node and edge management in the graph database
- Relationship tracking and updates
- Processing statistics and metrics
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from app.core.config.core_config_settings import get_settings
from db.facade import GraphDatabaseConnection
from processing.analyzers.file_analyzer import FileAnalyzer
from processing.processors.code_processor import CodeProcessor

settings = get_settings()
logger = logging.getLogger(__name__)


class IncrementalUpdateProcessor:
    """Handles the core processing logic for incremental updates."""

    def __init__(
        self,
        code_graph_id: str,
        code_processor: CodeProcessor | None = None,
        file_analyzer: FileAnalyzer | None = None,
        tenant_id: str | None = None,
    ):
        """Initialize the update processor.

        Args:
            code_graph_id: The code graph ID for database access
            code_processor: The code processor for parsing files
            file_analyzer: The file analyzer for extracting entities
            tenant_id: Owning tenant, persisted on written rows so the
                tenant-scoped identity (migration 0002) attributes them; None
                writes a NULL tenant (collapses via NULLS NOT DISTINCT).
        """
        self.code_graph_id = code_graph_id
        self.tenant_id = tenant_id
        self.code_processor = code_processor or CodeProcessor()
        self.file_analyzer = file_analyzer or FileAnalyzer()

    def _get_db_connection(self):
        """Get database connection for the code graph."""
        return GraphDatabaseConnection(self.code_graph_id, self.tenant_id)

    async def process_file_changes(
        self,
        repo_path: Path,
        changes: dict[str, list[Path]],
        version_id: str,
    ) -> dict[str, Any]:
        """Process file changes and update the graph.

        Args:
            repo_path: Path to the repository
            changes: Dictionary with added, modified, and deleted files
            version_id: Version ID for this update

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "added_nodes": 0,
            "modified_nodes": 0,
            "deleted_nodes": 0,
            "added_edges": 0,
            "modified_edges": 0,
            "deleted_edges": 0,
            "files_processed": 0,
            "processing_time": 0.0,
            "version_id": version_id,
        }

        start_time = time.time()

        # Collect all nodes that will be affected by the changes
        await self._collect_affected_nodes(changes)

        # Process changes in order: deletions first, then additions, then modifications
        await self._process_deleted_files(changes["deleted"], stats)
        await self._process_added_files(repo_path, changes["added"], stats)
        await self._process_modified_files(repo_path, changes["modified"], stats)

        stats["processing_time"] = time.time() - start_time
        return stats

    async def _collect_affected_nodes(self, changes: dict[str, list[Path]]) -> set[str]:
        """Collect all nodes that will be affected by the changes.

        Args:
            changes: Dictionary with file changes

        Returns:
            Set of node IDs that will be affected
        """
        affected_nodes: set[str] = set()

        # Collect nodes from added and modified files
        for file_path in changes["added"] + changes["modified"]:
            try:
                # Get existing nodes for this file if it's a modification
                if file_path in changes["modified"]:
                    existing_nodes = await self._get_nodes_by_file(str(file_path))
                    for node in existing_nodes:
                        affected_nodes.add(node["id"])

                # Analyze file to get entities
                code_entities = await self._analyze_file_for_entities(file_path)

                # Add all entity IDs to affected nodes
                for entity in code_entities:
                    node_id = self._generate_node_id(entity["type"], entity["name"], str(file_path))
                    affected_nodes.add(node_id)

                    # Add related nodes that might be affected
                    if "related_entities" in entity:
                        for related in entity["related_entities"]:
                            related_id = self._generate_node_id(
                                related["type"],
                                related["name"],
                                related.get("file", str(file_path)),
                            )
                            affected_nodes.add(related_id)

            except (OSError, ValueError) as e:
                logger.error("Error collecting affected nodes for %s: %s", file_path, e)

        # Collect nodes from deleted files
        for file_path in changes["deleted"]:
            try:
                existing_nodes = await self._get_nodes_by_file(str(file_path))
                for node in existing_nodes:
                    affected_nodes.add(node["id"])

                # Get related nodes that might be affected by this deletion
                related_nodes = await self._get_related_nodes(existing_nodes)
                for node_id in related_nodes:
                    affected_nodes.add(node_id)

            except (OSError, ValueError) as e:
                logger.error(
                    "Error collecting affected nodes for deleted file %s: %s",
                    file_path,
                    e,
                )

        return affected_nodes

    async def _process_deleted_files(
        self, deleted_files: list[Path], stats: dict[str, Any]
    ) -> None:
        """Process deleted files by removing their nodes and edges.

        Args:
            deleted_files: List of deleted file paths
            stats: Statistics dictionary to update
        """
        for file_path in deleted_files:
            try:
                deleted_count = await self._delete_nodes_by_file(str(file_path))
                stats["deleted_nodes"] += deleted_count
                stats["files_processed"] += 1
                logger.info("Deleted %d nodes from %s", deleted_count, file_path)
            except (OSError, ValueError) as e:
                logger.error("Error processing deleted file %s: %s", file_path, e)

    async def _process_added_files(
        self, repo_path: Path, added_files: list[Path], stats: dict[str, Any]
    ) -> None:
        """Process added files by creating new nodes and edges.

        Args:
            repo_path: Repository root path
            added_files: List of added file paths
            stats: Statistics dictionary to update
        """
        for file_path in added_files:
            try:
                # Analyze the file
                code_entities = await self._analyze_file_for_entities(file_path)

                # Add nodes and edges
                for entity in code_entities:
                    node_id = await self._add_entity_node(entity, file_path)
                    if node_id:
                        stats["added_nodes"] += 1

                        # Add edges for relationships
                        edge_count = await self._add_entity_relationships(
                            entity, node_id, file_path
                        )
                        stats["added_edges"] += edge_count

                stats["files_processed"] += 1
                logger.info("Processed added file %s", file_path)

            except (OSError, ValueError) as e:
                logger.error("Error processing added file %s: %s", file_path, e)

    async def _process_modified_files(
        self, repo_path: Path, modified_files: list[Path], stats: dict[str, Any]
    ) -> None:
        """Process modified files by updating existing nodes and edges.

        Args:
            repo_path: Repository root path
            modified_files: List of modified file paths
            stats: Statistics dictionary to update
        """
        for file_path in modified_files:
            try:
                # Get existing nodes for this file
                existing_nodes = await self._get_nodes_by_file(str(file_path))
                existing_node_ids = {node["id"] for node in existing_nodes}

                # Analyze the file to get new entities
                code_entities = await self._analyze_file_for_entities(file_path)

                # Track new node IDs
                new_node_ids = set()

                # Add or update nodes and edges
                for entity in code_entities:
                    node_id = self._generate_node_id(entity["type"], entity["name"], str(file_path))
                    new_node_ids.add(node_id)

                    # Check if node exists
                    if node_id in existing_node_ids:
                        # Update existing node
                        await self._update_entity_node(entity, node_id)
                        stats["modified_nodes"] += 1
                    else:
                        # Add new node
                        new_id = await self._add_entity_node(entity, file_path)
                        if new_id:
                            stats["added_nodes"] += 1

                    # Handle relationships
                    edge_stats = await self._update_entity_relationships(entity, node_id, file_path)
                    stats["added_edges"] += edge_stats["added"]
                    stats["deleted_edges"] += edge_stats["deleted"]

                # Delete nodes that no longer exist
                for node_id in existing_node_ids:
                    if node_id not in new_node_ids:
                        await self._delete_node(node_id)
                        stats["deleted_nodes"] += 1

                stats["files_processed"] += 1
                logger.info("Processed modified file %s", file_path)

            except (OSError, ValueError) as e:
                logger.error("Error processing modified file %s: %s", file_path, e)

    async def _analyze_file_for_entities(self, file_path: Path) -> list[dict[str, Any]]:
        """Analyze file and return entities in the expected format.

        Args:
            file_path: Path to the file to analyze

        Returns:
            List of entity dictionaries
        """
        try:
            # Use the analyzer to get analysis results
            analysis_result = await self.file_analyzer.analyze_file(str(file_path))

            # Convert CoreAnalysisResult to the expected entity format
            entities: list[dict[str, Any]] = []
            if hasattr(analysis_result, "entities") and analysis_result.entities:
                for entity in analysis_result.entities:
                    entities.append(
                        {
                            "type": getattr(entity, "type", "unknown"),
                            "name": getattr(entity, "name", ""),
                            "line": getattr(entity, "start_line", 0),
                            "column": getattr(entity, "start_column", 0),
                            "end_line": getattr(entity, "end_line", 0),
                            "end_column": getattr(entity, "end_column", 0),
                            "docstring": getattr(entity, "docstring", ""),
                            "signature": getattr(entity, "signature", ""),
                            "related_entities": [],  # Could be enhanced later
                        }
                    )
            return entities
        except Exception as e:
            logger.warning("Failed to analyze file %s: %s", file_path, e)
            return []

    async def _add_entity_node(self, entity: dict[str, Any], file_path: Path) -> str | None:
        """Add a node for an entity.

        Args:
            entity: Entity dictionary
            file_path: Path to the file containing the entity

        Returns:
            Node ID if successful, None otherwise
        """
        try:
            node_id = self._generate_node_id(entity["type"], entity["name"], str(file_path))

            await self._add_node(
                node_id=node_id,
                node_type=entity["type"],
                name=entity["name"],
                properties={
                    "file": str(file_path),
                    "line": entity.get("line", 0),
                    "column": entity.get("column", 0),
                    "end_line": entity.get("end_line", 0),
                    "end_column": entity.get("end_column", 0),
                    "docstring": entity.get("docstring", ""),
                    "signature": entity.get("signature", ""),
                },
            )
            return node_id
        except Exception as e:
            logger.error("Failed to add entity node: %s", e)
            return None

    async def _update_entity_node(self, entity: dict[str, Any], node_id: str) -> None:
        """Update an existing entity node.

        Args:
            entity: Entity dictionary
            node_id: ID of the node to update
        """
        try:
            await self._update_node(
                node_id=node_id,
                properties={
                    "line": entity.get("line", 0),
                    "column": entity.get("column", 0),
                    "end_line": entity.get("end_line", 0),
                    "end_column": entity.get("end_column", 0),
                    "docstring": entity.get("docstring", ""),
                    "signature": entity.get("signature", ""),
                },
            )
        except Exception as e:
            logger.error("Failed to update entity node %s: %s", node_id, e)

    async def _add_entity_relationships(
        self, entity: dict[str, Any], node_id: str, file_path: Path
    ) -> int:
        """Add relationships for an entity.

        Args:
            entity: Entity dictionary
            node_id: ID of the source node
            file_path: Path to the file containing the entity

        Returns:
            Number of edges added
        """
        edge_count = 0

        if "related_entities" in entity:
            for related in entity["related_entities"]:
                try:
                    related_id = self._generate_node_id(
                        related["type"],
                        related["name"],
                        related.get("file", str(file_path)),
                    )

                    await self._add_edge(
                        source_id=node_id,
                        target_id=related_id,
                        relationship_type=related["relationship"],
                        properties={},
                    )
                    edge_count += 1
                except Exception as e:
                    logger.error("Failed to add relationship edge: %s", e)

        return edge_count

    async def _update_entity_relationships(
        self, entity: dict[str, Any], node_id: str, file_path: Path
    ) -> dict[str, int]:
        """Update relationships for an entity.

        Args:
            entity: Entity dictionary
            node_id: ID of the source node
            file_path: Path to the file containing the entity

        Returns:
            Dictionary with counts of added and deleted edges
        """
        edge_stats = {"added": 0, "deleted": 0}

        try:
            # Get existing edges for this node
            existing_edges = await self._get_edges_for_node(node_id)
            existing_edge_targets = {
                (edge["target_id"], edge["relationship_type"]) for edge in existing_edges
            }

            # Track new edges
            new_edges = set()

            if "related_entities" in entity:
                for related in entity["related_entities"]:
                    related_id = self._generate_node_id(
                        related["type"],
                        related["name"],
                        related.get("file", str(file_path)),
                    )

                    edge_key = (related_id, related["relationship"])
                    new_edges.add(edge_key)

                    # Add edge if it doesn't exist
                    if edge_key not in existing_edge_targets:
                        await self._add_edge(
                            source_id=node_id,
                            target_id=related_id,
                            relationship_type=related["relationship"],
                            properties={},
                        )
                        edge_stats["added"] += 1

            # Remove edges that no longer exist
            for edge_key in existing_edge_targets:
                if edge_key not in new_edges:
                    target_id, rel_type = edge_key
                    await self._delete_edge(
                        source_id=node_id,
                        target_id=target_id,
                        relationship_type=rel_type,
                    )
                    edge_stats["deleted"] += 1

        except Exception as e:
            logger.error("Failed to update relationships for %s: %s", node_id, e)

        return edge_stats

    def _generate_node_id(self, entity_type: str, entity_name: str, file_path: str) -> str:
        """Generate a node ID using the same pattern as the original graph manager.

        Args:
            entity_type: Type of the entity
            entity_name: Name of the entity
            file_path: Path to the file containing the entity

        Returns:
            Generated node ID
        """
        # Create a hash based on type, name, and file path for uniqueness
        content = f"{entity_type}:{entity_name}:{file_path}"
        hash_obj = hashlib.md5(content.encode())  # nosec B324
        return f"{entity_type}_{hash_obj.hexdigest()[:8]}"

    # Database operation methods (delegated to separate module in practice)
    async def _add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        properties: dict[str, Any],
    ) -> str:
        """Add a node using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            node_properties = {**properties, "name": name}
            await db_conn.conn.execute(
                """
                INSERT INTO nodes (code_graph_id, id, type, properties, tenant_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (code_graph_id, tenant_id, id) DO UPDATE SET
                    type = EXCLUDED.type,
                    properties = EXCLUDED.properties,
                    updated_at = now()
                """,
                (self.code_graph_id, node_id, node_type, Jsonb(node_properties), self.tenant_id),
            )
            await db_conn.conn.commit()
            return node_id
        finally:
            await db_conn.close()

    async def _update_node(self, node_id: str, properties: dict[str, Any]) -> None:
        """Update a node using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Get existing properties first
            cursor = await db_conn.conn.execute(
                "SELECT properties FROM nodes WHERE code_graph_id = %s AND id = %s",
                (self.code_graph_id, node_id),
            )
            row = await cursor.fetchone()
            if row:
                existing_props = row[0] or {}
                existing_props.update(properties)

                await db_conn.conn.execute(
                    "UPDATE nodes SET properties = %s, updated_at = now() "
                    "WHERE code_graph_id = %s AND id = %s",
                    (Jsonb(existing_props), self.code_graph_id, node_id),
                )
                await db_conn.conn.commit()
        finally:
            await db_conn.close()

    async def _add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Add an edge using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            await db_conn.conn.execute(
                """
                INSERT INTO edges (code_graph_id, source, target, relationship, properties,
                                   tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO UPDATE SET
                    properties = EXCLUDED.properties
                """,
                (
                    self.code_graph_id,
                    source_id,
                    target_id,
                    relationship_type,
                    Jsonb(properties),
                    self.tenant_id,
                ),
            )
            await db_conn.conn.commit()
        finally:
            await db_conn.close()

    async def _delete_node(self, node_id: str) -> None:
        """Delete a node using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # Delete edges first
            await db_conn.conn.execute(
                "DELETE FROM edges WHERE code_graph_id = %s AND (source = %s OR target = %s)",
                (self.code_graph_id, node_id, node_id),
            )
            # Delete node
            await db_conn.conn.execute(
                "DELETE FROM nodes WHERE code_graph_id = %s AND id = %s",
                (self.code_graph_id, node_id),
            )
            await db_conn.conn.commit()
        finally:
            await db_conn.close()

    async def _delete_edge(self, source_id: str, target_id: str, relationship_type: str) -> None:
        """Delete an edge using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            await db_conn.conn.execute(
                "DELETE FROM edges WHERE code_graph_id = %s AND source = %s "
                "AND target = %s AND relationship = %s",
                (self.code_graph_id, source_id, target_id, relationship_type),
            )
            await db_conn.conn.commit()
        finally:
            await db_conn.close()

    async def _get_nodes_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get nodes for a specific file using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            cursor = await db_conn.conn.execute(
                """
                SELECT id, type, properties->>'name', properties
                FROM nodes
                WHERE code_graph_id = %s AND properties->>'file' = %s
                """,
                (self.code_graph_id, file_path),
            )
            nodes = []
            for row in await cursor.fetchall():
                nodes.append(
                    {
                        "id": row[0],
                        "type": row[1],
                        "name": row[2],
                        "properties": row[3] or {},
                    }
                )
            return nodes
        finally:
            await db_conn.close()

    async def _get_edges_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Get edges for a specific node using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            cursor = await db_conn.conn.execute(
                """
                SELECT source, target, relationship, properties
                FROM edges
                WHERE code_graph_id = %s AND source = %s
                """,
                (self.code_graph_id, node_id),
            )
            edges = []
            for row in await cursor.fetchall():
                edges.append(
                    {
                        "source_id": row[0],
                        "target_id": row[1],
                        "relationship_type": row[2],
                        "properties": row[3] or {},
                    }
                )
            return edges
        finally:
            await db_conn.close()

    async def _delete_nodes_by_file(self, file_path: str) -> int:
        """Delete nodes for a specific file using direct database access."""
        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            # First get the node IDs to delete
            cursor = await db_conn.conn.execute(
                """
                SELECT id FROM nodes
                WHERE code_graph_id = %s AND properties->>'file' = %s
                """,
                (self.code_graph_id, file_path),
            )
            node_ids = [row[0] for row in await cursor.fetchall()]

            if not node_ids:
                return 0

            # Delete edges first (foreign key constraints)
            for node_id in node_ids:
                await db_conn.conn.execute(
                    "DELETE FROM edges WHERE code_graph_id = %s AND (source = %s OR target = %s)",
                    (self.code_graph_id, node_id, node_id),
                )

            # Then delete nodes
            placeholders = ",".join(["%s"] * len(node_ids))
            await db_conn.conn.execute(
                f"DELETE FROM nodes WHERE code_graph_id = %s AND id IN ({placeholders})",  # nosec B608
                [self.code_graph_id, *node_ids],
            )

            await db_conn.conn.commit()
            return len(node_ids)
        finally:
            await db_conn.close()

    async def _get_related_nodes(self, nodes: list[dict[str, Any]]) -> set[str]:
        """Get nodes that are related to the given nodes through dependencies."""
        related_nodes: set[str] = set()

        if not nodes:
            return related_nodes

        db_conn = self._get_db_connection()
        await db_conn.connect()
        try:
            node_ids = [node["id"] for node in nodes]
            placeholders = ",".join(["%s"] * len(node_ids))

            # Get nodes that depend on these nodes (incoming edges)
            incoming_query = f"""
                SELECT DISTINCT source FROM edges
                WHERE code_graph_id = %s AND target IN ({placeholders})
                AND relationship IN ('calls', 'imports', 'extends', 'implements', 'uses')
                """  # nosec B608 -- placeholders are %s marks, not user input
            cursor = await db_conn.conn.execute(incoming_query, [self.code_graph_id, *node_ids])
            async for row in cursor:
                related_nodes.add(row[0])

            # Get nodes that these nodes depend on (outgoing edges)
            outgoing_query = f"""
                SELECT DISTINCT target FROM edges
                WHERE code_graph_id = %s AND source IN ({placeholders})
                AND relationship IN ('calls', 'imports', 'extends', 'implements', 'uses')
                """  # nosec B608 -- placeholders are %s marks, not user input
            cursor = await db_conn.conn.execute(outgoing_query, [self.code_graph_id, *node_ids])
            async for row in cursor:
                related_nodes.add(row[0])

            return related_nodes

        except (OSError, ValueError) as e:
            logger.error("Error getting related nodes: %s", e)
            return set()
        finally:
            await db_conn.close()
