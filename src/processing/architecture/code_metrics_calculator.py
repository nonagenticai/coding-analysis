"""Module for basic code metrics calculation and validation."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from models.architecture_models import Component

logger = logging.getLogger(__name__)


class CodeMetricsCalculator:
    """Calculates basic code metrics for components and validates graph structure.

    This class provides fundamental metric calculation capabilities including:
    - Component node mapping and validation
    - Basic file and node counting
    - Graph structure validation
    - Component relationship analysis
    """

    def validate_components_and_graph(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, Any]:
        """Validate that components and graph are properly structured for metric calculation.

        Args:
            components: List of components to validate
            graph: Code graph to validate

        Returns:
            Dictionary with validation results and basic statistics
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "component_count": len(components),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "components_with_nodes": 0,
            "orphaned_nodes": 0,
        }

        if not components:
            validation_result["valid"] = False
            validation_result["errors"].append("No components provided for analysis")
            return validation_result

        if graph.number_of_nodes() == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Graph contains no nodes")
            return validation_result

        # Check component structure
        component_ids = set()
        for component in components:
            if not component.id:
                validation_result["errors"].append("Component found with empty ID")
                validation_result["valid"] = False
                continue

            if component.id in component_ids:
                validation_result["errors"].append(f"Duplicate component ID: {component.id}")
                validation_result["valid"] = False
            else:
                component_ids.add(component.id)

            if not component.files:
                validation_result["warnings"].append(f"Component {component.id} has no files")

        # Map nodes to components
        component_node_mapping = self.map_nodes_to_components(components, graph)

        # Count components with nodes
        validation_result["components_with_nodes"] = len(
            [comp_id for comp_id, nodes in component_node_mapping.items() if nodes]
        )

        # Count orphaned nodes (nodes not belonging to any component)
        all_mapped_nodes = set()
        for nodes in component_node_mapping.values():
            all_mapped_nodes.update(nodes)

        validation_result["orphaned_nodes"] = graph.number_of_nodes() - len(all_mapped_nodes)

        if validation_result["orphaned_nodes"] > 0:
            validation_result["warnings"].append(
                f"{validation_result['orphaned_nodes']} nodes could not be mapped to any component"
            )

        return validation_result

    def map_nodes_to_components(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, list[str]]:
        """Map graph nodes to their corresponding components.

        Args:
            components: List of components
            graph: Code graph

        Returns:
            Dictionary mapping component IDs to lists of node IDs
        """
        component_nodes_map: dict[str, list[str]] = {}

        # Initialize mapping for all components
        for component in components:
            component_nodes_map[component.id] = []

        # Map nodes to components based on file paths
        logger.info(f"Mapping {graph.number_of_nodes()} nodes to {len(components)} components")

        for node, data in graph.nodes(data=True):
            file_path = data.get("file")
            if not file_path:
                continue

            # Find which component this node belongs to
            for component in components:
                if any(f in file_path for f in component.files):
                    component_nodes_map[component.id].append(node)
                    break

        # Log mapping statistics
        for component_id, nodes in component_nodes_map.items():
            logger.debug(f"Component {component_id}: {len(nodes)} nodes")

        return component_nodes_map

    def calculate_basic_component_metrics(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, dict[str, Any]]:
        """Calculate basic metrics for each component.

        Args:
            components: List of components to analyze
            graph: Code graph

        Returns:
            Dictionary of component IDs to their basic metrics
        """
        if not components:
            return {}

        component_nodes_map = self.map_nodes_to_components(components, graph)
        component_dict = {c.id: c for c in components}

        basic_metrics = {}

        for component_id, nodes in component_nodes_map.items():
            component = component_dict[component_id]

            # Basic counts
            node_count = len(nodes)
            file_count = len(component.files)
            dependency_count = len(component.dependencies)

            # Node type analysis
            node_types = self._analyze_node_types(graph, nodes)

            # Calculate basic ratios and statistics
            nodes_per_file = node_count / file_count if file_count > 0 else 0

            basic_metrics[component_id] = {
                "file_count": file_count,
                "node_count": node_count,
                "dependency_count": dependency_count,
                "nodes_per_file": nodes_per_file,
                "node_types": node_types,
                "has_nodes": node_count > 0,
                "has_dependencies": dependency_count > 0,
            }

        return basic_metrics

    def _analyze_node_types(self, graph: nx.DiGraph, nodes: list[str]) -> dict[str, int]:
        """Analyze the types of nodes in a component.

        Args:
            graph: Code graph
            nodes: List of node IDs to analyze

        Returns:
            Dictionary of node type counts
        """
        type_counts = {}

        for node in nodes:
            node_data = graph.nodes.get(node, {})
            node_type = node_data.get("type", "unknown").lower()

            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        return type_counts

    def calculate_component_size_metrics(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, dict[str, float]]:
        """Calculate size-related metrics for components.

        Args:
            components: List of components to analyze
            graph: Code graph

        Returns:
            Dictionary of component size metrics
        """
        basic_metrics = self.calculate_basic_component_metrics(components, graph)
        size_metrics = {}

        # Calculate overall statistics for normalization
        all_file_counts = [m["file_count"] for m in basic_metrics.values()]
        all_node_counts = [m["node_count"] for m in basic_metrics.values()]

        if not all_file_counts or not all_node_counts:
            return {}

        max_files = max(all_file_counts)
        max_nodes = max(all_node_counts)
        avg_files = sum(all_file_counts) / len(all_file_counts)
        avg_nodes = sum(all_node_counts) / len(all_node_counts)

        for component_id, metrics in basic_metrics.items():
            file_count = metrics["file_count"]
            node_count = metrics["node_count"]

            # Size ratios (relative to system maximum)
            file_size_ratio = file_count / max_files if max_files > 0 else 0
            node_size_ratio = node_count / max_nodes if max_nodes > 0 else 0

            # Size deviation from average
            file_deviation = abs(file_count - avg_files) / avg_files if avg_files > 0 else 0
            node_deviation = abs(node_count - avg_nodes) / avg_nodes if avg_nodes > 0 else 0

            # Combined size score
            size_score = (file_size_ratio + node_size_ratio) / 2

            size_metrics[component_id] = {
                "file_size_ratio": file_size_ratio,
                "node_size_ratio": node_size_ratio,
                "file_deviation_from_avg": file_deviation,
                "node_deviation_from_avg": node_deviation,
                "size_score": size_score,
                "is_large_component": size_score > 0.7,
                "is_small_component": size_score < 0.1,
            }

        return size_metrics

    def get_component_health_indicators(
        self, component_id: str, basic_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate health indicators for a component based on basic metrics.

        Args:
            component_id: ID of the component
            basic_metrics: Basic metrics for the component

        Returns:
            Dictionary with health indicators
        """
        health_indicators = {
            "component_id": component_id,
            "overall_health": "unknown",
            "issues": [],
            "strengths": [],
        }

        file_count = basic_metrics.get("file_count", 0)
        node_count = basic_metrics.get("node_count", 0)
        dependency_count = basic_metrics.get("dependency_count", 0)
        nodes_per_file = basic_metrics.get("nodes_per_file", 0)
        node_types = basic_metrics.get("node_types", {})

        # Analyze file structure
        if file_count == 0:
            health_indicators["issues"].append("Component has no files")
        elif file_count == 1:
            health_indicators["issues"].append(
                "Component has only one file - consider if this should be split or merged"
            )
        else:
            health_indicators["strengths"].append(f"Component has {file_count} files")

        # Analyze node structure
        if node_count == 0:
            health_indicators["issues"].append("Component has no analyzable code nodes")
        elif nodes_per_file > 20:
            health_indicators["issues"].append(
                f"High node density ({nodes_per_file:.1f} nodes per file) - consider splitting files"
            )
        elif nodes_per_file < 2 and file_count > 1:
            health_indicators["issues"].append("Low node density - files might be too granular")
        else:
            health_indicators["strengths"].append(
                f"Reasonable node density ({nodes_per_file:.1f} nodes per file)"
            )

        # Analyze dependencies
        if dependency_count == 0:
            if node_count > 5:  # Only warn for components with substantial code
                health_indicators["issues"].append(
                    "Component has no dependencies - might be isolated"
                )
        elif dependency_count > 10:
            health_indicators["issues"].append(
                f"High dependency count ({dependency_count}) - consider reducing coupling"
            )
        else:
            health_indicators["strengths"].append(
                f"Reasonable dependency count ({dependency_count})"
            )

        # Analyze node type diversity
        class_count = node_types.get("class", 0)
        function_count = node_types.get("function", 0) + node_types.get("method", 0)

        if class_count == 0 and function_count == 0:
            health_indicators["issues"].append("Component has no classes or functions")
        elif class_count > 0 and function_count == 0:
            health_indicators["issues"].append(
                "Component has classes but no methods - might indicate incomplete analysis"
            )

        # Overall health assessment
        issue_count = len(health_indicators["issues"])
        strength_count = len(health_indicators["strengths"])

        if issue_count == 0:
            health_indicators["overall_health"] = "excellent"
        elif issue_count <= 2 and strength_count >= issue_count:
            health_indicators["overall_health"] = "good"
        elif issue_count < 4:
            health_indicators["overall_health"] = "fair"
        else:
            health_indicators["overall_health"] = "poor"

        return health_indicators

    def calculate_system_overview(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, Any]:
        """Calculate system-level overview metrics.

        Args:
            components: List of all components
            graph: Complete code graph

        Returns:
            Dictionary with system overview data
        """
        validation = self.validate_components_and_graph(components, graph)
        basic_metrics = self.calculate_basic_component_metrics(components, graph)
        size_metrics = self.calculate_component_size_metrics(components, graph)

        # System totals
        total_files = sum(len(c.files) for c in components)
        total_dependencies = sum(len(c.dependencies) for c in components)

        # Component distribution
        large_components = [
            comp_id
            for comp_id, metrics in size_metrics.items()
            if metrics.get("is_large_component", False)
        ]
        small_components = [
            comp_id
            for comp_id, metrics in size_metrics.items()
            if metrics.get("is_small_component", False)
        ]

        # Health distribution
        health_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for comp_id, metrics in basic_metrics.items():
            health = self.get_component_health_indicators(comp_id, metrics)
            overall_health = health["overall_health"]
            if overall_health in health_distribution:
                health_distribution[overall_health] += 1

        return {
            "total_components": len(components),
            "total_files": total_files,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "total_dependencies": total_dependencies,
            "avg_files_per_component": (total_files / len(components) if components else 0),
            "avg_dependencies_per_component": (
                total_dependencies / len(components) if components else 0
            ),
            "large_components": large_components,
            "small_components": small_components,
            "health_distribution": health_distribution,
            "validation_summary": {
                "valid": validation["valid"],
                "error_count": len(validation["errors"]),
                "warning_count": len(validation["warnings"]),
                "orphaned_nodes": validation["orphaned_nodes"],
            },
        }
