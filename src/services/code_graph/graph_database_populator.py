"""Graph database populator module.

Batch population of the shared PostgreSQL code-graph store. Every row is scoped
to one ``code_graph_id`` (== the container name); the schema is owned by Alembic,
so this module no longer creates tables or tunes per-file SQLite pragmas.
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from db.session import get_raw_connection

logger = logging.getLogger(__name__)


async def populate_graph_database(container_name: str, analysis_data: dict[str, Any]) -> bool:
    """Graph database population with batch processing."""
    code_graph_id = container_name
    try:
        logger.info(
            f"🚀 Starting optimized graph database population for code graph: {code_graph_id}"
        )
        nodes_data = analysis_data.get("nodes", [])
        edges_data = analysis_data.get("edges", [])
        logger.info(f"📊 Processing {len(nodes_data)} nodes, {len(edges_data)} edges")
        async with get_raw_connection() as conn:
            try:
                nodes_inserted = await _batch_insert_nodes(conn, code_graph_id, nodes_data)
                edges_inserted = await _batch_insert_edges(conn, code_graph_id, edges_data)
                await conn.commit()
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
                )
                node_result = await cursor.fetchone()
                final_node_count = node_result[0] if node_result else 0
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (code_graph_id,)
                )
                edge_result = await cursor.fetchone()
                final_edge_count = edge_result[0] if edge_result else 0
                logger.info("✅ Database population completed successfully!")
                logger.info(f"   📊 Inserted {nodes_inserted} nodes, {edges_inserted} edges")
                logger.info(
                    f"   📊 Final counts: {final_node_count} nodes, {final_edge_count} edges"
                )
                return True
            except Exception:
                await conn.rollback()
                raise
    except Exception as e:
        logger.error(f"❌ Database population failed: {e}")
        logger.exception("Database population error details:")
        return False


async def _batch_insert_nodes(
    conn: AsyncConnection, code_graph_id: str, nodes_data: list[dict]
) -> int:
    """Batch insert nodes with deduplication (name/file_path live in properties)."""
    if not nodes_data:
        return 0

    batch_data = []
    seen_node_ids: set[str] = set()

    for node in nodes_data:
        node_id = node["id"]
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)

        properties = node.get("properties")
        if properties is None:
            properties = {}
        else:
            needs_name = "name" not in properties and "name" in node
            needs_path = "file_path" not in properties and "file_path" in node
            if needs_name or needs_path:
                properties = properties.copy()
                if needs_name:
                    properties["name"] = node["name"]
                if needs_path:
                    properties["file_path"] = node.get("file_path", "")

        batch_data.append((code_graph_id, node_id, node["type"], Jsonb(properties)))

    if batch_data:
        await conn.executemany(
            "INSERT INTO nodes (code_graph_id, id, type, properties) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (code_graph_id, tenant_id, id) DO NOTHING",
            batch_data,
        )
        return len(batch_data)
    return 0


async def _batch_insert_edges(
    conn: AsyncConnection, code_graph_id: str, edges_data: list[dict]
) -> int:
    """Batch insert edges with deduplication."""
    if not edges_data:
        return 0
    batch_data = []
    seen_edge_combinations = set()
    for edge in edges_data:
        edge_key = (edge["source_id"], edge["target_id"], edge["type"])
        if edge_key in seen_edge_combinations:
            continue
        seen_edge_combinations.add(edge_key)
        batch_data.append(
            (
                code_graph_id,
                edge["source_id"],
                edge["target_id"],
                edge["type"],
                Jsonb(edge.get("properties", {})),
            )
        )
    if batch_data:
        await conn.executemany(
            "INSERT INTO edges (code_graph_id, source, target, relationship, properties) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO NOTHING",
            batch_data,
        )
        return len(batch_data)
    return 0


async def verify_database_integrity(container_name: str) -> dict[str, Any]:
    """Verify the integrity of a populated graph (scoped to one code_graph_id)."""
    code_graph_id = container_name
    try:
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
            )
            node_result = await cursor.fetchone()
            node_count = node_result[0] if node_result else 0
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (code_graph_id,)
            )
            edge_result = await cursor.fetchone()
            edge_count = edge_result[0] if edge_result else 0
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM edges e
                WHERE e.code_graph_id = %s
                  AND (e.source NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s)
                       OR e.target NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s))
            """,
                (code_graph_id, code_graph_id, code_graph_id),
            )
            orphaned_result = await cursor.fetchone()
            orphaned_edges = orphaned_result[0] if orphaned_result else 0
            if orphaned_edges > 0:
                logger.info(f"🧹 Cleaning up {orphaned_edges} orphaned edges...")
                await conn.execute(
                    """
                    DELETE FROM edges
                    WHERE code_graph_id = %s
                      AND (source NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s)
                           OR target NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s))
                    """,
                    (code_graph_id, code_graph_id, code_graph_id),
                )
                await conn.commit()
                logger.info(f"✅ Removed {orphaned_edges} orphaned edges")
                orphaned_edges = 0
            cursor = await conn.execute(
                """
                SELECT type, COUNT(*) as count
                FROM nodes
                WHERE code_graph_id = %s
                GROUP BY type
                ORDER BY count DESC
            """,
                (code_graph_id,),
            )
            node_types = await cursor.fetchall()
            return {
                "status": "success",
                "node_count": node_count,
                "edge_count": edge_count,
                "orphaned_edges": orphaned_edges,
                "node_types": {node_type: count for node_type, count in node_types},
                "code_graph_id": code_graph_id,
            }
    except Exception as e:
        logger.error(f"Database integrity check failed: {e}")
        return {"error": str(e)}


async def optimize_database_performance(container_name: str) -> bool:
    """No-op under shared PostgreSQL.

    ``VACUUM``/``ANALYZE`` cannot run inside the pooled (PgBouncer transaction-
    mode) connection, and space/statistics maintenance is handled by autovacuum.
    """
    logger.info(
        "Database optimization is a no-op under shared PostgreSQL (container: %s)",
        container_name,
    )
    return True


async def get_database_stats(container_name: str) -> dict[str, Any]:
    """Get comprehensive statistics about a code graph."""
    code_graph_id = container_name
    try:
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
            )
            node_result = await cursor.fetchone()
            node_count = node_result[0] if node_result else 0
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (code_graph_id,)
            )
            edge_result = await cursor.fetchone()
            edge_count = edge_result[0] if edge_result else 0
            cursor = await conn.execute(
                """
                SELECT relationship, COUNT(*) as count
                FROM edges
                WHERE code_graph_id = %s
                GROUP BY relationship
                ORDER BY count DESC
            """,
                (code_graph_id,),
            )
            relationship_types = await cursor.fetchall()
            cursor = await conn.execute(
                """
                SELECT n.id, n.type, n.properties->>'name' as name,
                       COUNT(e.source) + COUNT(e2.target) as connection_count
                FROM nodes n
                LEFT JOIN edges e
                    ON n.code_graph_id = e.code_graph_id AND n.id = e.source
                LEFT JOIN edges e2
                    ON n.code_graph_id = e2.code_graph_id AND n.id = e2.target
                WHERE n.code_graph_id = %s
                GROUP BY n.id, n.type, name
                ORDER BY connection_count DESC
                LIMIT 10
            """,
                (code_graph_id,),
            )
            top_connected_nodes = await cursor.fetchall()
            return {
                "container_name": container_name,
                "code_graph_id": code_graph_id,
                "file_size_bytes": 0,
                "file_size_mb": 0,
                "node_count": node_count,
                "edge_count": edge_count,
                "relationship_types": {rel_type: count for rel_type, count in relationship_types},
                "top_connected_nodes": [
                    {
                        "id": row[0],
                        "type": row[1],
                        "name": row[2],
                        "connections": row[3],
                    }
                    for row in top_connected_nodes
                ],
            }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}
