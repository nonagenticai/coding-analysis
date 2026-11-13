"""
Task execution engine for different types of background tasks.
Core execution logic and base classes for task processing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import psycopg
from psycopg import AsyncConnection

from app.core.config import get_settings
from db.facade import GraphDatabaseConnection

settings = get_settings()
logger = logging.getLogger(__name__)


async def calculate_degree_centrality(
    conn: AsyncConnection, code_graph_id: str, node_type: str | None = None
) -> dict[str, float]:
    """Calculate degree centrality directly using async SQL."""
    # Get nodes
    if node_type:
        cursor = await conn.execute(
            "SELECT id FROM nodes WHERE code_graph_id = %s AND type = %s",
            (code_graph_id, node_type),
        )
    else:
        cursor = await conn.execute(
            "SELECT id FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
        )

    node_ids = [row[0] for row in await cursor.fetchall()]

    if not node_ids:
        return {}
    centrality_scores: dict[str, Any] = {}
    for node_id in node_ids:
        # Count outgoing edges
        out_cursor = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s AND source = %s",
            (code_graph_id, node_id),
        )
        out_degree_row = await out_cursor.fetchone()
        out_degree = out_degree_row[0] if out_degree_row else 0

        # Count incoming edges
        in_cursor = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s AND target = %s",
            (code_graph_id, node_id),
        )
        in_degree_row = await in_cursor.fetchone()
        in_degree = in_degree_row[0] if in_degree_row else 0

        centrality_scores[node_id] = out_degree + in_degree
    # Normalize scores
    max_degree = max(centrality_scores.values()) if centrality_scores else 1
    if max_degree > 0:
        for node_id in centrality_scores:
            centrality_scores[node_id] = centrality_scores[node_id] / max_degree
    return centrality_scores


async def calculate_pagerank(
    conn: AsyncConnection,
    code_graph_id: str,
    node_type: str | None = None,
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[str, float]:
    """Calculate PageRank directly using async SQL."""
    # Get nodes
    if node_type:
        cursor = await conn.execute(
            "SELECT id FROM nodes WHERE code_graph_id = %s AND type = %s",
            (code_graph_id, node_type),
        )
    else:
        cursor = await conn.execute(
            "SELECT id FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
        )

    nodes = [row[0] for row in await cursor.fetchall()]
    n = len(nodes)

    if n == 0:
        return {}
    # Initialize PageRank algorithm data structures
    node_to_index = {node: i for i, node in enumerate(nodes)}
    scores = [1.0 / n] * n
    outgoing_counts = await _calculate_outgoing_counts(conn, code_graph_id, nodes)
    incoming_links = await _build_incoming_links(conn, code_graph_id, nodes, node_to_index)

    # PageRank iteration
    for _ in range(iterations):
        scores = _calculate_pagerank_iteration(scores, incoming_links, outgoing_counts, damping, n)
    return {nodes[i]: scores[i] for i in range(n)}


async def _calculate_outgoing_counts(
    conn: AsyncConnection, code_graph_id: str, nodes: list[str]
) -> list[int]:
    """Calculate outgoing edge counts for each node."""
    outgoing_counts = []
    for node in nodes:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s AND source = %s",
            (code_graph_id, node),
        )
        count_row = await cursor.fetchone()
        outgoing_counts.append(count_row[0] if count_row else 0)
    return outgoing_counts


async def _build_incoming_links(
    conn: AsyncConnection,
    code_graph_id: str,
    nodes: list[str],
    node_to_index: dict[str, int],
) -> list[list[int]]:
    """Build adjacency lists for incoming links."""
    incoming_links: list[list[int]] = [[] for _ in range(len(nodes))]
    for source_node in nodes:
        source_idx = node_to_index[source_node]
        cursor = await conn.execute(
            "SELECT target FROM edges WHERE code_graph_id = %s AND source = %s",
            (code_graph_id, source_node),
        )
        async for (target,) in cursor:
            if target in node_to_index:
                target_idx = node_to_index[target]
                incoming_links[target_idx].append(source_idx)
    return incoming_links


def _calculate_pagerank_iteration(
    scores: list[float],
    incoming_links: list[list[int]],
    outgoing_counts: list[int],
    damping: float,
    n: int,
) -> list[float]:
    """Calculate a single PageRank iteration."""
    new_scores = [(1.0 - damping) / n] * n

    for i in range(n):
        for source_idx in incoming_links[i]:
            if outgoing_counts[source_idx] > 0:
                new_scores[i] += damping * scores[source_idx] / outgoing_counts[source_idx]
    return new_scores


class BaseTaskProcessor:
    """Base class for task processors."""

    async def process(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process a task and return the result."""
        raise NotImplementedError

    def get_processor_type(self) -> str:
        """Get the processor type identifier."""
        return self.__class__.__name__


class CentralityCalculationProcessor(BaseTaskProcessor):
    """Processor for centrality calculation with direct database access."""

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported centrality metrics."""
        return ["degree_centrality", "pagerank"]

    async def process(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process centrality calculation task."""
        code_graph_id = task_data.get("code_graph_id", "default")
        node_type = task_data.get("node_type")

        # Initialize database connection directly
        db_conn = GraphDatabaseConnection(code_graph_id)

        try:
            # Establish async connection
            await db_conn.connect()
            conn = db_conn.get_connection()

            start_time = time.time()

            # Calculate centrality metrics with direct async functions
            centrality = await calculate_degree_centrality(conn, code_graph_id, node_type)
            pagerank = await calculate_pagerank(conn, code_graph_id, node_type)

            # Get top nodes
            top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]

            top_pagerank_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]

            return {
                "success": True,
                "duration_seconds": time.time() - start_time,
                "top_central_nodes": [
                    {"id": node_id, "centrality": score} for node_id, score in top_central_nodes
                ],
                "top_pagerank_nodes": [
                    {"id": node_id, "pagerank": score} for node_id, score in top_pagerank_nodes
                ],
            }

        except (FileNotFoundError, OSError) as e:
            logger.error("Database error during centrality calculation: %s", e)
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
            }
        except psycopg.Error as e:
            logger.error("Database error calculating centrality: %s", e)
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
            }
        except (RuntimeError, MemoryError, ValueError, KeyError) as e:
            logger.error("Runtime/data error calculating centrality: %s", e)
            return {
                "success": False,
                "error": f"Runtime/data error: {str(e)}",
            }
        finally:
            await db_conn.close()
