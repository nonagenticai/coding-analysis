"""Module for calculating and managing code graph metrics."""

import logging
import time
from typing import Any

import networkx as nx

from db.facade import GraphDatabaseConnection
from models.core_models import OverallGraphMetrics
from services.cache_manager import cached

logger = logging.getLogger(__name__)


class CodeGraphMetricsService:
    """Service for calculating and retrieving metrics for code graph entities."""

    def __init__(self, db_connection: GraphDatabaseConnection):
        self.db_connection = db_connection

    @cached("graph_metrics", ttl_seconds=300)
    async def get_graph_metrics(self) -> OverallGraphMetrics:
        """Get comprehensive graph metrics using single optimized SQL query."""
        try:
            if self.db_connection.conn is None:
                raise ValueError("Database connection is not available")

            start_time = time.time()

            # Single comprehensive query - MUCH more efficient
            cgid = self.db_connection.code_graph_id
            cursor = await self.db_connection.conn.execute(
                """
                SELECT
                    COUNT(*) as node_count,
                    (SELECT COUNT(*) FROM edges WHERE code_graph_id = %s) as edge_count,
                    (SELECT COUNT(DISTINCT type) FROM nodes WHERE code_graph_id = %s)
                        as node_type_count,
                    (SELECT COUNT(DISTINCT relationship) FROM edges WHERE code_graph_id = %s)
                        as relationship_type_count
                FROM nodes
                WHERE code_graph_id = %s
            """,
                (cgid, cgid, cgid, cgid),
            )

            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No data returned from metrics query")
            (
                node_count,
                edge_count,
                node_type_count,
                relationship_type_count,
            ) = result

            # Get type breakdowns in separate optimized queries
            cursor = await self.db_connection.conn.execute(
                "SELECT type, COUNT(*) FROM nodes WHERE code_graph_id = %s GROUP BY type",
                (cgid,),
            )
            node_types = {row[0]: row[1] for row in await cursor.fetchall()}

            cursor = await self.db_connection.conn.execute(
                "SELECT relationship, COUNT(*) FROM edges WHERE code_graph_id = %s "
                "GROUP BY relationship",
                (cgid,),
            )
            relationship_types = {row[0]: row[1] for row in await cursor.fetchall()}

            # Calculate derived metrics
            max_edges = node_count * (node_count - 1) if node_count > 1 else 1
            density = edge_count / max_edges if max_edges > 0 else 0.0
            avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0.0
            calculation_time = time.time() - start_time

            return OverallGraphMetrics(
                node_count=node_count or 0,
                edge_count=edge_count or 0,
                density=density,
                average_degree=avg_degree,
                node_types=node_types,
                relationship_types=relationship_types,
                calculation_time_seconds=calculation_time,
            )
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return OverallGraphMetrics(
                node_count=0,
                edge_count=0,
                density=0.0,
                average_degree=0.0,
                node_types={},
                relationship_types={},
                calculation_time_seconds=0.0,
            )

    async def get_node_centrality_metrics(self, node_id: str) -> dict[str, float]:
        """Calculate centrality metrics for a specific node using single optimized SQL query."""
        try:
            if self.db_connection.conn is None:
                raise ValueError("Database connection is not available")

            # Single query to get both in/out degree and total nodes - MUCH more efficient
            cgid = self.db_connection.code_graph_id
            cursor = await self.db_connection.conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM edges WHERE code_graph_id = %s AND source = %s)
                        as out_degree,
                    (SELECT COUNT(*) FROM edges WHERE code_graph_id = %s AND target = %s)
                        as in_degree,
                    (SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s) as total_nodes
            """,
                (cgid, node_id, cgid, node_id, cgid),
            )

            result = await cursor.fetchone()
            if result is None:
                raise ValueError("No data returned from centrality query")
            out_degree, in_degree, total_nodes = result

            total_degree = out_degree + in_degree
            max_degree = max(1, total_nodes - 1)
            degree_centrality = total_degree / max_degree

            return {
                "degree_centrality": degree_centrality,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
            }
        except Exception as e:
            logger.error(f"Error calculating centrality for node {node_id}: {e}")
            return {
                "degree_centrality": 0.0,
                "in_degree": 0,
                "out_degree": 0,
                "total_degree": 0,
            }

    async def calculate_centrality_metrics(self, entity_type: str) -> dict[str, dict[str, float]]:
        """Calculate centrality metrics for all nodes of a specific type using direct SQL."""
        try:
            if self.db_connection.conn is None:
                raise ValueError("Database connection is not available")

            cursor = await self.db_connection.conn.execute(
                "SELECT id FROM nodes WHERE code_graph_id = %s AND type = %s",
                (self.db_connection.code_graph_id, entity_type),
            )
            nodes = await cursor.fetchall()

            centrality_metrics: dict[str, Any] = {}
            for node in nodes:
                centrality_metrics[node[0]] = await self.get_node_centrality_metrics(node[0])

            return centrality_metrics
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error calculating centrality metrics for {entity_type}: {e}")
            return {}

    async def get_function_network_metrics(self, function_name: str) -> dict[str, Any]:
        """Calculate network metrics for a function."""
        try:
            # Get centrality metrics for the function
            centrality_metrics = await self.get_node_centrality_metrics(function_name)

            # Calculate comprehensive centrality metrics using graph theory algorithms
            degree_centrality = centrality_metrics.get("degree_centrality", 0.0)

            # Calculate actual betweenness centrality using shortest paths
            betweenness_centrality = await self._calculate_betweenness_centrality(function_name)
            # Real PageRank and closeness centrality over the container-scoped graph
            pagerank, closeness_centrality = await self._calculate_pagerank_and_closeness(
                function_name
            )

            # Calculate network importance as weighted average
            network_importance = (
                degree_centrality * 0.4
                + betweenness_centrality * 0.3
                + closeness_centrality * 0.2
                + pagerank * 0.1
            )

            return {
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality,
                "closeness_centrality": closeness_centrality,
                "pagerank": pagerank,
                "network_importance": network_importance,
                "in_degree": centrality_metrics.get("in_degree", 0),
                "out_degree": centrality_metrics.get("out_degree", 0),
                "total_degree": centrality_metrics.get("total_degree", 0),
            }
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error calculating network metrics for function {function_name}: {e}")
            return {
                "degree_centrality": 0.0,
                "betweenness_centrality": 0.0,
                "closeness_centrality": 0.0,
                "pagerank": 0.0,
                "network_importance": 0.0,
                "in_degree": 0,
                "out_degree": 0,
                "total_degree": 0,
            }

    async def _calculate_pagerank_and_closeness(self, node_id: str) -> tuple[float, float]:
        """Compute real PageRank and closeness centrality for a node.

        Builds a directed ``networkx`` graph from the same container-scoped
        ``nodes``/``edges`` rows the betweenness computation reads, then returns
        the node's genuine ``nx.pagerank`` and ``nx.closeness_centrality`` values.
        Returns ``(0.0, 0.0)`` when the node is absent from the graph, the graph
        is empty, or PageRank fails to converge.
        """
        try:
            if self.db_connection.conn is None:
                raise ValueError("Database connection is not available")

            cgid = self.db_connection.code_graph_id
            cursor = await self.db_connection.conn.execute(
                "SELECT id FROM nodes WHERE code_graph_id = %s", (cgid,)
            )
            all_nodes = [row[0] for row in await cursor.fetchall()]
            if node_id not in all_nodes:
                return 0.0, 0.0

            cursor = await self.db_connection.conn.execute(
                "SELECT source, target FROM edges WHERE code_graph_id = %s", (cgid,)
            )
            edges = await cursor.fetchall()

            graph = nx.DiGraph()
            graph.add_nodes_from(all_nodes)
            for source, target in edges:
                if source in graph and target in graph:
                    graph.add_edge(source, target)

            try:
                pagerank_scores = nx.pagerank(graph)
                pagerank = float(pagerank_scores.get(node_id, 0.0))
            except nx.PowerIterationFailedConvergence:
                pagerank = 0.0

            closeness = float(nx.closeness_centrality(graph, u=node_id))
            return pagerank, closeness
        except (ValueError, TypeError, AttributeError, nx.NetworkXException) as e:
            logger.debug("Error calculating pagerank/closeness for %s: %s", node_id, e)
            return 0.0, 0.0


# Wire the betweenness-centrality extension methods onto the class. This import
# must run *after* the class above is defined: the extension does
# ``setattr(CodeGraphMetricsService, ...)`` and imports the class back, so doing
# it at the top of the module would hit a partially-initialised module and fail
# silently (leaving ``_calculate_betweenness_centrality`` unbound).
from services import code_graph_metrics_extension  # noqa: E402,F401
