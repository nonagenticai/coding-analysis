"""Dependency visualizer service.

Generates visual dependency graphs from parsed package information.
"""

from __future__ import annotations

import logging
import random
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class VisualizationType(StrEnum):
    """Types of dependency visualizations."""

    GRAPH = "graph"
    TREE = "tree"
    MATRIX = "matrix"
    FORCE_DIRECTED = "force_directed"


class NodeType(StrEnum):
    """Types of nodes in the dependency graph."""

    MODULE = "module"
    PACKAGE = "package"
    CLASS = "class"
    FUNCTION = "function"


class EdgeType(StrEnum):
    """Types of edges in the dependency graph."""

    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    DEPENDS_ON = "depends_on"


class DependencyVisualizer:
    """Generates visual dependency graphs."""

    async def create_visualization(
        self,
        codebase_path: str,
        visualization_type: VisualizationType = VisualizationType.GRAPH,
    ) -> dict[str, Any]:
        """Create a dependency visualization.

        Scans the codebase at *codebase_path*, builds nodes and edges from the
        discovered packages, and formats the result according to
        *visualization_type*.
        """
        from services.dependency_parser import DependencyParser

        parser = DependencyParser(codebase_path)
        packages = await parser.get_available_packages()

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for name, info in packages.items():
            nodes.append(
                {
                    "id": name,
                    "label": info.get("name", name),
                    "type": NodeType.PACKAGE,
                    "version": info.get("version", ""),
                }
            )
            # Extras are treated as dependencies on sibling packages
            for extra in info.get("extras", []):
                extra_lower = extra.lower()
                if extra_lower in packages:
                    edges.append(
                        {
                            "source": name,
                            "target": extra_lower,
                            "type": EdgeType.DEPENDS_ON,
                        }
                    )

        if visualization_type == VisualizationType.GRAPH:
            result = self._build_graph(nodes, edges)
        elif visualization_type == VisualizationType.TREE:
            result = self._build_tree(nodes, edges)
        elif visualization_type == VisualizationType.MATRIX:
            result = self._build_matrix(nodes, edges)
        else:  # FORCE_DIRECTED
            result = self._build_force_directed(nodes, edges)

        result["metadata"] = {
            "total_packages": len(packages),
            "visualization_type": str(visualization_type),
            "codebase_path": codebase_path,
        }
        return result

    # ------------------------------------------------------------------
    # Format builders
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Adjacency-list style graph."""
        adjacency: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            adjacency.setdefault(edge["source"], []).append(edge["target"])
        return {"nodes": nodes, "edges": edges, "adjacency": adjacency}

    def _build_tree(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Hierarchical parent-children tree.

        Nodes with no incoming edges are roots; others are children of their
        dependency sources.
        """
        children: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        has_parent: set[str] = set()
        for edge in edges:
            children[edge["source"]].append(edge["target"])
            has_parent.add(edge["target"])

        roots = [n["id"] for n in nodes if n["id"] not in has_parent]

        def build_node(node_id: str) -> dict[str, Any]:
            return {
                "id": node_id,
                "children": [build_node(c) for c in children.get(node_id, [])],
            }

        tree = [build_node(r) for r in roots]
        return {"nodes": nodes, "edges": edges, "tree": tree}

    def _build_matrix(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """NxN dependency matrix where matrix[i][j]=1 means node i depends on node j."""
        n = len(nodes)
        index = {node["id"]: i for i, node in enumerate(nodes)}
        matrix: list[list[int]] = [[0] * n for _ in range(n)]
        for edge in edges:
            src = index.get(edge["source"])
            tgt = index.get(edge["target"])
            if src is not None and tgt is not None:
                matrix[src][tgt] = 1
        labels = [node["id"] for node in nodes]
        return {"nodes": nodes, "edges": edges, "matrix": matrix, "labels": labels}

    def _build_force_directed(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Same as GRAPH but each node carries a random (x, y) position."""
        rng = random.Random(42)  # deterministic seed for reproducibility
        positioned_nodes = [
            {**node, "x": rng.uniform(0, 1000), "y": rng.uniform(0, 1000)} for node in nodes
        ]
        adjacency: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            adjacency.setdefault(edge["source"], []).append(edge["target"])
        return {"nodes": positioned_nodes, "edges": edges, "adjacency": adjacency}


_dependency_visualizer: DependencyVisualizer | None = None


def get_dependency_visualizer() -> DependencyVisualizer:
    """Return a singleton DependencyVisualizer instance."""
    global _dependency_visualizer  # noqa: PLW0603
    if _dependency_visualizer is None:
        _dependency_visualizer = DependencyVisualizer()
    return _dependency_visualizer
