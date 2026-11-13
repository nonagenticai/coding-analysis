"""Module for analyzing architectural metrics using Robert C. Martin's principles."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from models.architecture_models import ArchitecturalMetric, Component

logger = logging.getLogger(__name__)


class ArchitectureMetricsAnalyzer:
    """Analyzes architectural metrics following Robert C. Martin's principles.

    This class provides methods to calculate:
    - Robert C. Martin's metrics (instability, abstractness, distance from main sequence)
    - Fan-in and fan-out metrics
    - Component-level coupling metrics

    These metrics help assess the quality of software architecture and guide refactoring
    decisions to update maintainability and extensibility.
    """

    def calc(self, graph: nx.DiGraph, components: list[Component]) -> list[ArchitecturalMetric]:
        """Calculate instability, abstractness, and distance metrics.

        This method calculates:
        - Fan-in and fan-out for each component
        - Instability metric (fan_out / (fan_in + fan_out))
        - Abstractness metric (abstract_classes / total_classes)
        - Distance from main sequence (|A + I - 1|)

        Args:
            graph: The code graph to analyze
            components: List of components to calculate metrics for

        Returns:
            List of ArchitecturalMetric objects with calculated values
        """
        metrics = []

        for component in components:
            # Calculate fan metrics
            fan_in, fan_out = self._calculate_fan_metrics(component, components, graph)

            # Calculate instability
            instability = self._calculate_instability(fan_in, fan_out)

            # Calculate abstractness
            abstractness = self._calculate_abstractness(component, graph)

            # Calculate distance from main sequence
            distance = self._calculate_distance(instability, abstractness)

            metrics.append(
                ArchitecturalMetric(
                    component_id=component.id,
                    metric_type="robert_martin_metrics",
                    value=distance,
                    description=f"Robert C. Martin metrics - Instability: {instability:.2f}, Abstractness: {abstractness:.2f}, Distance: {distance:.2f}",
                    instability=instability,
                    abstractness=abstractness,
                    distance=distance,
                )
            )

        return metrics

    def _calculate_fan_metrics(
        self,
        component: Component,
        all_components: list[Component],
        _graph: nx.DiGraph,  # Reserved for future enhancements
    ) -> tuple[int, int]:
        """Calculate fan-in and fan-out for a component.

        Args:
            component: The component to analyze
            all_components: List of all components in the system
            _graph: Reserved for future graph-based analysis

        Returns:
            Tuple of (fan_in, fan_out) counts
        """
        fan_in = 0
        fan_out = 0

        # Calculate fan-in (dependencies on this component)
        for other in all_components:
            if component.id in other.dependencies:
                fan_in += 1

        # Calculate fan-out (this component's dependencies)
        fan_out = len(component.dependencies)

        return fan_in, fan_out

    def _calculate_instability(self, fan_in: int, fan_out: int) -> float:
        """Calculate instability metric: fan_out / (fan_in + fan_out).

        Args:
            fan_in: Number of incoming dependencies
            fan_out: Number of outgoing dependencies

        Returns:
            Instability metric between 0 (stable) and 1 (unstable)

        Note:
            Returns 0.0 if there are no dependencies (fan_in + fan_out = 0)
        """
        total = fan_in + fan_out
        return fan_out / total if total > 0 else 0.0

    def _calculate_abstractness(self, component: Component, graph: nx.DiGraph) -> float:
        """Calculate abstractness metric: abstract_classes / total_classes.

        Args:
            component: The component to analyze
            graph: The code graph containing node data

        Returns:
            Abstractness metric between 0 (concrete) and 1 (abstract)

        Note:
            Returns 0.0 if component has no classes
        """
        nodes = self._get_component_nodes(component, graph)
        abstract_count = sum(1 for node in nodes if self._is_abstract_node(node))
        total_count = len(nodes)
        return abstract_count / total_count if total_count > 0 else 0.0

    def _calculate_distance(self, instability: float, abstractness: float) -> float:
        """Calculate distance from main sequence: |A + I - 1|.

        Args:
            instability: Instability metric (0-1)
            abstractness: Abstractness metric (0-1)

        Returns:
            Distance from main sequence (0 is optimal)
        """
        return abs(abstractness + instability - 1)

    def _get_component_nodes(self, component: Component, graph: nx.DiGraph) -> list[dict[str, Any]]:
        """Get all nodes that belong to a component.

        Args:
            component: The component to get nodes for
            graph: The code graph containing node data

        Returns:
            List of node data dictionaries for the component
        """
        return [
            node_data
            for node_id, node_data in graph.nodes(data=True)
            if node_data.get("component_id") == component.id
        ]

    def _is_abstract_node(self, node_data: dict) -> bool:
        """Check if a node represents an abstract class or interface.

        Args:
            node_data: Node data to check

        Returns:
            True if node is abstract, False otherwise
        """
        # Check for explicit abstract flag
        if node_data.get("is_abstract", False):
            return True

        # Check for interface type
        if node_data.get("type") == "interface":
            return True

        # Check for abstract methods
        if any(method.get("is_abstract", False) for method in node_data.get("methods", [])):
            return True

        return False

    def calculate_component_metrics(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, dict[str, float]]:
        """Calculate all metrics for each component.

        Args:
            components: List of components to analyze
            graph: Directed graph representing code relationships

        Returns:
            Dictionary of component IDs to their metric values
        """
        if not components:
            return {}

        component_ids = {c.id: c for c in components}
        component_nodes_map: dict[
            str, list[str]
        ] = {}  # Map component ID to list of nodes in the component

        # Determine node to component mapping
        logger.info(f"Mapping {graph.number_of_nodes()} nodes to {len(components)} components")
        for node, data in graph.nodes(data=True):
            file_path = data.get("file")
            if not file_path:
                continue

            # Find which component this node belongs to
            for component in components:
                if any(f in file_path for f in component.files):
                    if component.id not in component_nodes_map:
                        component_nodes_map[component.id] = []
                    component_nodes_map[component.id].append(node)
                    break

        # Calculate metrics for each component
        metrics = {}
        for component_id, nodes in component_nodes_map.items():
            fan_in, fan_out = self._calculate_fan_metrics(
                component_ids[component_id], components, graph
            )
            instability = self._calculate_instability(fan_in, fan_out)
            abstractness = self._calculate_abstractness(component_ids[component_id], graph)
            distance = self._calculate_distance(instability, abstractness)

            metrics[component_id] = {
                "fan_in": fan_in,
                "fan_out": fan_out,
                "instability": instability,
                "abstractness": abstractness,
                "distance": distance,
                "file_count": len(component_ids[component_id].files),
            }

        return metrics

    def _calculate_weighted_quality_score(self, metrics: dict[str, dict[str, float]]) -> float:
        """Calculate a weighted quality score based on all metrics.

        This combines all metrics with appropriate weights to produce a single quality score.
        Higher scores represent better architectural quality.

        Args:
            metrics: Dictionary of component metrics

        Returns:
            A weighted quality score between 0 and 1
        """
        if not metrics:
            return 0.0

        # Define metric weights (must sum to 1.0)
        weights = {
            "distance": 0.4,  # Distance from main sequence
            "instability": 0.3,  # Instability
            "abstractness": 0.3,  # Abstractness
        }

        # Normalize metrics to 0-1 range where 1 is best
        scores = []
        for _, metric in metrics.items():
            component_score = 0.0
            metrics_count = 0

            # Distance from main sequence (lower is better, so subtract from 1)
            if "distance" in metric:
                # Normalize: 0 distance = 1.0 score, 1.0 distance = 0.0 score
                component_score += weights["distance"] * (1.0 - min(1.0, metric["distance"]))
                metrics_count += 1

            # Instability (should be balanced, so 0.5 is ideal)
            if "instability" in metric:
                # Score is highest when instability is 0.5
                instability_score = 1.0 - (abs(metric["instability"] - 0.5) * 2.0)
                component_score += weights["instability"] * instability_score
                metrics_count += 1

            # Abstractness (should be appropriate for its position)
            if "abstractness" in metric and "instability" in metric:
                # Distance from main sequence already captures this relationship
                # but we reward appropriate abstractness based on instability
                expected_abstractness = 1.0 - metric["instability"]
                abstractness_score = 1.0 - abs(metric["abstractness"] - expected_abstractness)
                component_score += weights["abstractness"] * abstractness_score
                metrics_count += 1

            # Normalize score if we have metrics
            if metrics_count > 0:
                scores.append(component_score / metrics_count)

        # Return average score across all components
        return sum(scores) / len(scores) if scores else 0.0

    def analyze_dependency_patterns(
        self, components: list[Component], graph: nx.DiGraph
    ) -> dict[str, Any]:
        """Analyze dependency patterns across components.

        Args:
            components: List of components to analyze
            graph: Code graph containing dependency relationships

        Returns:
            Dictionary with dependency analysis results
        """
        if not components:
            return {}

        component_metrics = self.calculate_component_metrics(components, graph)

        # Analyze instability distribution
        instabilities = [m["instability"] for m in component_metrics.values()]
        high_instability_components = [
            comp_id
            for comp_id, metrics in component_metrics.items()
            if metrics["instability"] > 0.7
        ]

        # Analyze abstractness distribution
        abstractnesses = [m["abstractness"] for m in component_metrics.values()]
        concrete_components = [
            comp_id
            for comp_id, metrics in component_metrics.items()
            if metrics["abstractness"] < 0.2
        ]

        # Analyze main sequence violations
        main_sequence_violations = [
            comp_id for comp_id, metrics in component_metrics.items() if metrics["distance"] > 0.3
        ]

        # Calculate dependency cycles
        dependency_graph = nx.DiGraph()
        for component in components:
            dependency_graph.add_node(component.id)
            for dep in component.dependencies:
                dependency_graph.add_edge(component.id, dep)

        try:
            cycles = list(nx.simple_cycles(dependency_graph))
        except Exception:
            cycles = []

        return {
            "total_components": len(components),
            "avg_instability": (sum(instabilities) / len(instabilities) if instabilities else 0.0),
            "avg_abstractness": (
                sum(abstractnesses) / len(abstractnesses) if abstractnesses else 0.0
            ),
            "high_instability_components": high_instability_components,
            "concrete_components": concrete_components,
            "main_sequence_violations": main_sequence_violations,
            "dependency_cycles": cycles,
            "cycle_count": len(cycles),
        }

    def get_component_recommendations(
        self, component_id: str, metrics: dict[str, float]
    ) -> list[str]:
        """Generate recommendations for improving a component's architecture.

        Args:
            component_id: ID of the component to analyze
            metrics: Component metrics dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        instability = metrics.get("instability", 0.0)
        abstractness = metrics.get("abstractness", 0.0)
        distance = metrics.get("distance", 0.0)

        # Distance from main sequence recommendations
        if distance > 0.3:
            if abstractness + instability > 1.3:  # Zone of Uselessness
                recommendations.append(
                    f"Component {component_id} is in the Zone of Uselessness - "
                    "consider reducing abstractness or dependencies"
                )
            elif abstractness + instability < 0.7:  # Zone of Pain
                recommendations.append(
                    f"Component {component_id} is in the Zone of Pain - "
                    "consider increasing abstractness or reducing incoming dependencies"
                )

        # Instability recommendations
        if instability > 0.8:
            recommendations.append(
                f"Component {component_id} has high instability ({instability:.2f}) - "
                "consider reducing outgoing dependencies"
            )
        elif instability < 0.2 and abstractness < 0.2:
            recommendations.append(
                f"Component {component_id} is very stable but concrete - ensure this is appropriate for its role"
            )

        # Abstractness recommendations
        if abstractness > 0.8 and instability > 0.8:
            recommendations.append(
                f"Component {component_id} is highly abstract and unstable - "
                "verify this component provides sufficient value"
            )

        return recommendations
