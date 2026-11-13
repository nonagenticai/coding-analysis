"""Extension methods for CodeGraphMetricsService"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from services.code_graph_metrics import CodeGraphMetricsService


async def _calculate_betweenness_centrality(self, node_id: str) -> float:
    """Calculate betweenness centrality for a node using shortest paths algorithm."""
    try:
        cgid = self.db_connection.code_graph_id
        cursor = await self.db_connection.conn.execute(
            "SELECT id FROM nodes WHERE code_graph_id = %s", (cgid,)
        )
        all_nodes = [row[0] for row in await cursor.fetchall()]
        if len(all_nodes) < 3:
            return 0.0
        cursor = await self.db_connection.conn.execute(
            "SELECT source, target FROM edges WHERE code_graph_id = %s", (cgid,)
        )
        edges = await cursor.fetchall()
        graph: dict[str, list[str]] = {}
        for node in all_nodes:
            graph[node] = []
        for source, target in edges:
            if source in graph and target in graph:
                graph[source].append(target)
                if target not in graph[source]:
                    graph[target].append(source)
        betweenness = 0.0
        total_pairs = 0
        for source in all_nodes:
            if source == node_id:
                continue
            for target in all_nodes:
                if target in (node_id, source):
                    continue
                paths_through_node = _find_shortest_paths_through_node(
                    graph, source, target, node_id
                )
                total_paths = _find_total_shortest_paths(graph, source, target)
                if total_paths > 0:
                    betweenness += paths_through_node / total_paths
                    total_pairs += 1
        if total_pairs > 0:
            betweenness = betweenness / total_pairs
        return min(1.0, betweenness)
    except (AttributeError, KeyError, ValueError) as e:
        logger = logging.getLogger(__name__)
        logger.debug("Error calculating betweenness centrality for %s: %s", node_id, e)
        return 0.0


def _find_shortest_paths_through_node(
    graph: dict[str, Any], source: str, target: str, through_node: str
) -> int:
    """Find number of shortest paths from source to target that go through through_node."""
    try:
        queue = deque([(source, 0)])
        visited = {source}
        shortest_distance = float("inf")
        while queue:
            node, dist = queue.popleft()
            if node == target:
                shortest_distance = dist
                break
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        if shortest_distance == float("inf"):
            return 0
        dist_to_node = _bfs_distance(graph, source, through_node)
        dist_from_node = _bfs_distance(graph, through_node, target)
        if dist_to_node + dist_from_node == shortest_distance:
            return 1
        return 0
    except (AttributeError, KeyError, ValueError):  # fmt: skip
        return 0


def _find_total_shortest_paths(graph: dict[str, Any], source: str, target: str) -> int:
    """Find total number of shortest paths from source to target."""
    try:
        distance = _bfs_distance(graph, source, target)
        return 1 if distance < 999999 else 0
    except (AttributeError, KeyError, ValueError):  # fmt: skip
        return 0


def _bfs_distance(graph: dict[str, Any], source: str, target: str) -> int:
    """Calculate shortest distance between two nodes using BFS."""
    try:
        if source == target:
            return 0
        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            node, dist = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999999
    except (AttributeError, KeyError, ValueError):  # fmt: skip
        return 999999


setattr(
    CodeGraphMetricsService,
    "_calculate_betweenness_centrality",
    _calculate_betweenness_centrality,
)
setattr(
    CodeGraphMetricsService,
    "_find_shortest_paths_through_node",
    staticmethod(_find_shortest_paths_through_node),
)
setattr(
    CodeGraphMetricsService,
    "_find_total_shortest_paths",
    staticmethod(_find_total_shortest_paths),
)
setattr(CodeGraphMetricsService, "_bfs_distance", staticmethod(_bfs_distance))
