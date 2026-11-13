"""
Chart and Diagram Generation

Advanced chart generation logic and specialized diagram creation utilities.
This module handles specialized visualization generation beyond basic dependency graphs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


class ChartGenerator:
    """Generator for specialized charts and diagrams."""

    def __init__(self):
        """Initialize the chart generator."""
        self.supported_chart_types = [
            "complexity_heatmap",
            "dependency_matrix",
            "language_distribution",
            "call_hierarchy",
            "architectural_layers",
        ]

    async def generate_complexity_heatmap(
        self, nodes: list[Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate a complexity heatmap visualization.

        Args:
            nodes: List of visualization nodes with complexity data
            options: Optional configuration for heatmap generation

        Returns:
            Heatmap visualization data
        """
        try:
            options = options or {}

            # Extract complexity data from nodes
            complexity_data = []
            for node in nodes:
                if hasattr(node, "complexity") and hasattr(node, "language"):
                    complexity_data.append(
                        {
                            "id": node.id,
                            "label": node.label,
                            "complexity": node.complexity,
                            "language": node.language,
                            "file_path": getattr(node, "file_path", None),
                        }
                    )

            # Sort by complexity for better visualization
            complexity_data.sort(key=lambda x: x["complexity"], reverse=True)

            # Generate heatmap configuration
            heatmap_config = {
                "type": "complexity_heatmap",
                "data": complexity_data,
                "options": {
                    "color_scale": options.get("color_scale", "RdYlBu"),
                    "max_items": options.get("max_items", 50),
                    "group_by_language": options.get("group_by_language", True),
                    "threshold_levels": options.get("threshold_levels", [5, 10, 20]),
                },
                "layout": {
                    "width": options.get("width", 800),
                    "height": options.get("height", 600),
                    "margin": options.get(
                        "margin", {"top": 50, "right": 50, "bottom": 50, "left": 100}
                    ),
                },
            }

            return heatmap_config

        except Exception as e:
            logger.error("Error generating complexity heatmap: %s", str(e))
            raise

    async def generate_dependency_matrix(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a dependency matrix visualization.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional configuration for matrix generation

        Returns:
            Dependency matrix visualization data
        """
        try:
            options = options or {}

            # Create node lookup for efficient access
            node_lookup = {node.id: node for node in nodes}

            # Build dependency matrix
            matrix_data = []
            node_ids = [node.id for node in nodes]

            for source_id in node_ids:
                row = []
                for target_id in node_ids:
                    # Find edge between source and target
                    edge_weight = 0
                    for edge in edges:
                        if edge.source == source_id and edge.target == target_id:
                            edge_weight = getattr(edge, "weight", 1)
                            break
                    row.append(edge_weight)
                matrix_data.append(row)

            # Generate matrix configuration
            matrix_config = {
                "type": "dependency_matrix",
                "data": {
                    "matrix": matrix_data,
                    "labels": [node_lookup[nid].label for nid in node_ids],
                    "node_ids": node_ids,
                },
                "options": {
                    "show_labels": options.get("show_labels", True),
                    "color_scale": options.get("color_scale", "Blues"),
                    "cluster_nodes": options.get("cluster_nodes", False),
                    "show_values": options.get("show_values", True),
                },
                "layout": {
                    "width": options.get("width", 600),
                    "height": options.get("height", 600),
                    "cell_size": options.get("cell_size", 20),
                },
            }

            return matrix_config

        except Exception as e:
            logger.error("Error generating dependency matrix: %s", str(e))
            raise

    async def generate_language_distribution(
        self, nodes: list[Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate a language distribution chart.

        Args:
            nodes: List of visualization nodes
            options: Optional configuration for chart generation

        Returns:
            Language distribution chart data
        """
        try:
            options = options or {}

            # Count nodes by language
            language_counts = {}
            complexity_by_language = {}

            for node in nodes:
                if hasattr(node, "language"):
                    language = node.language
                    language_counts[language] = language_counts.get(language, 0) + 1

                    if hasattr(node, "complexity"):
                        if language not in complexity_by_language:
                            complexity_by_language[language] = []
                        complexity_by_language[language].append(node.complexity)

            # Calculate average complexity per language
            avg_complexity = {}
            for lang, complexities in complexity_by_language.items():
                avg_complexity[lang] = sum(complexities) / len(complexities) if complexities else 0

            # Generate chart configuration
            chart_config = {
                "type": "language_distribution",
                "data": {
                    "counts": language_counts,
                    "average_complexity": avg_complexity,
                    "languages": list(language_counts.keys()),
                },
                "options": {
                    "chart_type": options.get("chart_type", "pie"),  # pie, bar, donut
                    "show_complexity": options.get("show_complexity", True),
                    "color_scheme": options.get("color_scheme", "category10"),
                    "show_percentages": options.get("show_percentages", True),
                },
                "layout": {
                    "width": options.get("width", 500),
                    "height": options.get("height", 500),
                    "margin": options.get(
                        "margin", {"top": 20, "right": 20, "bottom": 20, "left": 20}
                    ),
                },
            }

            return chart_config

        except Exception as e:
            logger.error("Error generating language distribution: %s", str(e))
            raise

    async def generate_call_hierarchy(
        self,
        nodes: list[Any],
        edges: list[Any],
        root_node_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a call hierarchy tree visualization.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            root_node_id: Optional root node to start hierarchy from
            options: Optional configuration for hierarchy generation

        Returns:
            Call hierarchy visualization data
        """
        try:
            options = options or {}

            # Build adjacency list for call relationships
            call_graph = {}
            for edge in edges:
                if getattr(edge, "type", "") in ["call", "invoke", "dependency"]:
                    source = edge.source
                    target = edge.target
                    if source not in call_graph:
                        call_graph[source] = []
                    call_graph[source].append(target)

            # Find root node if not specified
            if not root_node_id:
                # Find node with most outgoing calls
                max_calls = 0
                for node_id, targets in call_graph.items():
                    if len(targets) > max_calls:
                        max_calls = len(targets)
                        root_node_id = node_id

            # Build hierarchy tree
            def build_tree(node_id: str, visited: set, depth: int = 0) -> dict[str, Any]:
                if node_id in visited or depth > options.get("max_depth", 5):
                    return {"id": node_id, "children": []}

                visited.add(node_id)
                children = []

                for child_id in call_graph.get(node_id, []):
                    children.append(build_tree(child_id, visited.copy(), depth + 1))

                return {
                    "id": node_id,
                    "children": children,
                    "depth": depth,
                }

            hierarchy_tree = build_tree(root_node_id or "", set()) if root_node_id else {}

            # Generate hierarchy configuration
            hierarchy_config = {
                "type": "call_hierarchy",
                "data": {
                    "tree": hierarchy_tree,
                    "root_node": root_node_id,
                    "call_graph": call_graph,
                },
                "options": {
                    "layout_type": options.get("layout_type", "tree"),  # tree, radial, dendrogram
                    "max_depth": options.get("max_depth", 5),
                    "show_node_labels": options.get("show_node_labels", True),
                    "collapse_threshold": options.get("collapse_threshold", 3),
                },
                "layout": {
                    "width": options.get("width", 800),
                    "height": options.get("height", 600),
                    "node_size": options.get("node_size", 10),
                    "link_distance": options.get("link_distance", 100),
                },
            }

            return hierarchy_config

        except Exception as e:
            logger.error("Error generating call hierarchy: %s", str(e))
            raise

    async def generate_architectural_layers(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate an architectural layers visualization.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional configuration for layers generation

        Returns:
            Architectural layers visualization data
        """
        try:
            options = options or {}

            # Define common architectural layers
            layer_patterns = options.get(
                "layer_patterns",
                {
                    "presentation": ["controller", "view", "ui", "frontend", "web"],
                    "service": ["service", "business", "logic", "application"],
                    "data": ["repository", "dao", "model", "entity", "database"],
                    "infrastructure": ["config", "util", "helper", "common", "shared"],
                },
            )

            # Categorize nodes into layers
            layers = {layer: [] for layer in layer_patterns.keys()}
            uncategorized = []

            for node in nodes:
                categorized = False
                node_path = getattr(node, "file_path", "") or ""
                node_label = getattr(node, "label", "").lower()

                for layer, patterns in layer_patterns.items():
                    if any(
                        pattern in node_path.lower() or pattern in node_label
                        for pattern in patterns
                    ):
                        layers[layer].append(node)
                        categorized = True
                        break

                if not categorized:
                    uncategorized.append(node)

            # Calculate layer dependencies
            layer_dependencies = {}
            for edge in edges:
                source_layer = None
                target_layer = None

                # Find which layers the source and target belong to
                for layer, layer_nodes in layers.items():
                    if any(n.id == edge.source for n in layer_nodes):
                        source_layer = layer
                    if any(n.id == edge.target for n in layer_nodes):
                        target_layer = layer

                if source_layer and target_layer and source_layer != target_layer:
                    dep_key = f"{source_layer}->{target_layer}"
                    layer_dependencies[dep_key] = layer_dependencies.get(dep_key, 0) + 1

            # Generate layers configuration
            layers_config = {
                "type": "architectural_layers",
                "data": {
                    "layers": {
                        k: [{"id": n.id, "label": n.label} for n in v] for k, v in layers.items()
                    },
                    "uncategorized": [{"id": n.id, "label": n.label} for n in uncategorized],
                    "dependencies": layer_dependencies,
                    "layer_order": list(layer_patterns.keys()),
                },
                "options": {
                    "show_violations": options.get("show_violations", True),
                    "highlight_cross_layer": options.get("highlight_cross_layer", True),
                    "layer_spacing": options.get("layer_spacing", 100),
                    "show_dependency_counts": options.get("show_dependency_counts", True),
                },
                "layout": {
                    "width": options.get("width", 1000),
                    "height": options.get("height", 700),
                    "layer_height": options.get("layer_height", 150),
                },
            }

            return layers_config

        except Exception as e:
            logger.error("Error generating architectural layers: %s", str(e))
            raise
