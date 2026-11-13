"""Module for computing complexity metrics including LCOM4 and TCC."""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class ComplexityMetricsComputer:
    """Computes complexity metrics for code cohesion and coupling analysis.

    This class provides methods to calculate:
    - LCOM4 (Lack of Cohesion of Methods) metric
    - TCC (Tight Class Cohesion) metric
    - Class-level complexity analysis
    """

    def calculate_lcom4(self, graph: nx.DiGraph, component_nodes: list[str]) -> float:
        """Calculate LCOM4 (Lack of Cohesion of Methods) metric.

        LCOM4 measures the number of connected components in the graph of methods.
        Methods are connected if they share instance variables.
        Lower values indicate better cohesion (more connected methods).

        Args:
            graph: The code graph containing method and field relationships
            component_nodes: List of node IDs belonging to the component

        Returns:
            LCOM4 value normalized between 0 and 1 (lower is better cohesion)
        """
        # Extract classes from the component
        class_nodes = [n for n in component_nodes if self._is_class_node(graph.nodes[n])]
        if not class_nodes:
            return 0.0

        lcom4_values = []
        for class_node in class_nodes:
            # Get methods and fields of this class
            methods, fields = self._get_class_methods_and_fields(graph, class_node)
            if not methods or len(methods) <= 1:
                continue

            # Create a graph connecting methods that share fields
            method_graph: nx.Graph = nx.Graph()
            method_graph.add_nodes_from(methods)

            # Connect methods that share fields
            field_to_methods = defaultdict(set)
            for method in methods:
                # Find fields accessed by this method
                for _, target, edge_data in graph.out_edges(method, data=True):
                    if target in fields and edge_data.get("type") == "accesses":
                        field_to_methods[target].add(method)

            # Add edges between methods sharing fields
            for methods_sharing_field in field_to_methods.values():
                if len(methods_sharing_field) >= 2:
                    for m1, m2 in itertools.combinations(methods_sharing_field, 2):
                        method_graph.add_edge(m1, m2)

            # Calculate connected components
            connected_components = list(nx.connected_components(method_graph))

            # LCOM4 is the number of connected components
            # Normalize: 1 component = 0.0, all separate = 1.0
            if len(methods) > 1:  # avoid division by zero
                lcom4 = (len(connected_components) - 1) / (len(methods) - 1)
                lcom4_values.append(lcom4)

        # Return average LCOM4 across all classes in the component
        return sum(lcom4_values) / len(lcom4_values) if lcom4_values else 0.0

    def calculate_tcc(self, graph: nx.DiGraph, component_nodes: list[str]) -> float:
        """Calculate TCC (Tight Class Cohesion) metric.

        TCC measures the relative number of directly connected methods.
        It represents the ratio of the actual number of connected method pairs to
        the maximum possible number of connected method pairs.
        Higher values indicate better cohesion.

        Args:
            graph: The code graph containing method relationships
            component_nodes: List of node IDs belonging to the component

        Returns:
            TCC value between 0 and 1 (higher is better cohesion)
        """
        # Extract classes from the component
        class_nodes = [n for n in component_nodes if self._is_class_node(graph.nodes[n])]
        if not class_nodes:
            return 0.0

        tcc_values = []
        for class_node in class_nodes:
            # Get methods and fields of this class
            methods, fields = self._get_class_methods_and_fields(graph, class_node)
            if not methods or len(methods) <= 1:
                continue

            # Calculate maximum possible method pairs
            max_pairs = len(methods) * (len(methods) - 1) / 2
            if max_pairs == 0:
                continue

            # Map fields to the methods that use them
            field_to_methods = defaultdict(set)
            for method in methods:
                for _, target, edge_data in graph.out_edges(method, data=True):
                    if target in fields and edge_data.get("type") == "accesses":
                        field_to_methods[target].add(method)

            # Count connected method pairs (sharing at least one field)
            connected_pairs = 0
            for field, connected_methods in field_to_methods.items():
                if len(connected_methods) >= 2:
                    connected_pairs += len(list(itertools.combinations(connected_methods, 2)))

            # TCC = connected pairs / possible pairs
            tcc = connected_pairs / max_pairs
            tcc_values.append(tcc)

        # Return average TCC across all classes in the component
        return sum(tcc_values) / len(tcc_values) if tcc_values else 0.0

    def _is_class_node(self, node_data: dict) -> bool:
        """Check if a node represents a class.

        Args:
            node_data: Node attributes

        Returns:
            True if node is a class, False otherwise
        """
        node_type = node_data.get("type", "").lower()
        return node_type in [
            "class",
            "interface",
            "abstract_class",
            "trait",
            "struct",
        ]

    def _get_class_methods_and_fields(
        self, graph: nx.DiGraph, class_node: str
    ) -> tuple[list[str], list[str]]:
        """Extract methods and fields for a class from the graph.

        Args:
            graph: Code relationship graph
            class_node: Node ID of the class

        Returns:
            Tuple of (method_nodes, field_nodes)
        """
        methods = []
        fields = []

        # Look for outgoing contains edges to find members
        for _, member_node, edge_data in graph.out_edges(class_node, data=True):
            if edge_data.get("type") == "contains":
                member_type = graph.nodes[member_node].get("type", "").lower()

                # Classify as method or field
                if member_type in [
                    "method",
                    "function",
                    "constructor",
                    "procedure",
                ]:
                    methods.append(member_node)
                elif member_type in [
                    "field",
                    "property",
                    "attribute",
                    "variable",
                    "constant",
                ]:
                    fields.append(member_node)

        logger.debug(
            "Found %d methods and %d fields for class %s",
            len(methods),
            len(fields),
            class_node,
        )
        return methods, fields

    def calculate_class_complexity_metrics(
        self, graph: nx.DiGraph, class_node: str
    ) -> dict[str, float]:
        """Calculate comprehensive complexity metrics for a single class.

        Args:
            graph: Code relationship graph
            class_node: Node ID of the class

        Returns:
            Dictionary of complexity metrics for the class
        """
        if not self._is_class_node(graph.nodes[class_node]):
            return {}

        methods, fields = self._get_class_methods_and_fields(graph, class_node)

        # Basic complexity metrics
        method_count = len(methods)
        field_count = len(fields)

        # Calculate LCOM4 for this class
        lcom4 = self.calculate_lcom4(graph, [class_node])

        # Calculate TCC for this class
        tcc = self.calculate_tcc(graph, [class_node])

        # Calculate method-to-field ratio
        method_field_ratio = method_count / field_count if field_count > 0 else method_count

        # Calculate method coupling (methods calling other methods in same class)
        internal_method_calls = 0
        for method in methods:
            for _, target, edge_data in graph.out_edges(method, data=True):
                if target in methods and edge_data.get("type") == "calls":
                    internal_method_calls += 1

        method_coupling = internal_method_calls / method_count if method_count > 0 else 0.0

        # Calculate field usage distribution
        field_usage = defaultdict(int)
        for method in methods:
            for _, target, edge_data in graph.out_edges(method, data=True):
                if target in fields and edge_data.get("type") == "accesses":
                    field_usage[target] += 1

        # Calculate field usage variance (higher variance = less cohesive)
        if field_usage:
            usage_values = list(field_usage.values())
            mean_usage = sum(usage_values) / len(usage_values)
            variance = sum((x - mean_usage) ** 2 for x in usage_values) / len(usage_values)
            field_usage_variance = variance / (mean_usage**2) if mean_usage > 0 else 0.0
        else:
            field_usage_variance = 0.0

        return {
            "method_count": method_count,
            "field_count": field_count,
            "lcom4": lcom4,
            "tcc": tcc,
            "method_field_ratio": method_field_ratio,
            "method_coupling": method_coupling,
            "field_usage_variance": field_usage_variance,
            "complexity_score": self._calculate_class_complexity_score(
                method_count, field_count, lcom4, tcc, method_coupling
            ),
        }

    def _calculate_class_complexity_score(
        self,
        method_count: int,
        field_count: int,
        lcom4: float,
        tcc: float,
        method_coupling: float,
    ) -> float:
        """Calculate an overall complexity score for a class.

        Args:
            method_count: Number of methods in the class
            field_count: Number of fields in the class
            lcom4: LCOM4 metric value
            tcc: TCC metric value
            method_coupling: Internal method coupling ratio

        Returns:
            Complexity score between 0 and 1 (higher means more complex)
        """
        # Normalize size factors
        size_complexity = min(1.0, (method_count + field_count) / 20.0)

        # Cohesion factor (lower cohesion = higher complexity)
        cohesion_complexity = (lcom4 + (1.0 - tcc)) / 2.0

        # Coupling factor (higher internal coupling can be good or bad)
        # Moderate coupling (0.3-0.7) is often optimal
        if method_coupling < 0.3:
            coupling_complexity = 0.3 - method_coupling  # Too little coupling
        elif method_coupling > 0.7:
            coupling_complexity = method_coupling - 0.7  # Too much coupling
        else:
            coupling_complexity = 0.0  # Good coupling level

        # Weighted combination
        complexity_score = (
            0.4 * size_complexity + 0.4 * cohesion_complexity + 0.2 * coupling_complexity
        )

        return min(1.0, complexity_score)

    def calculate_component_complexity_summary(
        self, graph: nx.DiGraph, component_nodes: list[str]
    ) -> dict[str, Any]:
        """Calculate a comprehensive complexity summary for a component.

        Args:
            graph: Code relationship graph
            component_nodes: List of node IDs belonging to the component

        Returns:
            Dictionary with component complexity analysis
        """
        class_nodes = [n for n in component_nodes if self._is_class_node(graph.nodes[n])]

        if not class_nodes:
            return {
                "class_count": 0,
                "avg_lcom4": 0.0,
                "avg_tcc": 0.0,
                "avg_complexity_score": 0.0,
                "complex_classes": [],
                "cohesive_classes": [],
            }

        # Calculate metrics for each class
        class_metrics = {}
        for class_node in class_nodes:
            class_metrics[class_node] = self.calculate_class_complexity_metrics(graph, class_node)

        # Aggregate metrics
        lcom4_values = [m["lcom4"] for m in class_metrics.values() if m.get("lcom4") is not None]
        tcc_values = [m["tcc"] for m in class_metrics.values() if m.get("tcc") is not None]
        complexity_scores = [m["complexity_score"] for m in class_metrics.values()]

        avg_lcom4 = sum(lcom4_values) / len(lcom4_values) if lcom4_values else 0.0
        avg_tcc = sum(tcc_values) / len(tcc_values) if tcc_values else 0.0
        avg_complexity = (
            sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
        )

        # Identify complex and cohesive classes
        complex_classes = [
            class_node
            for class_node, metrics in class_metrics.items()
            if metrics["complexity_score"] > 0.7
        ]

        cohesive_classes = [
            class_node
            for class_node, metrics in class_metrics.items()
            if metrics.get("tcc", 0.0) > 0.7 and metrics.get("lcom4", 1.0) < 0.3
        ]

        return {
            "class_count": len(class_nodes),
            "avg_lcom4": avg_lcom4,
            "avg_tcc": avg_tcc,
            "avg_complexity_score": avg_complexity,
            "complex_classes": complex_classes,
            "cohesive_classes": cohesive_classes,
            "class_metrics": class_metrics,
            "total_methods": sum(m["method_count"] for m in class_metrics.values()),
            "total_fields": sum(m["field_count"] for m in class_metrics.values()),
        }

    def get_complexity_recommendations(self, complexity_summary: dict[str, Any]) -> list[str]:
        """Generate recommendations based on complexity analysis.

        Args:
            complexity_summary: Component complexity summary from calculate_component_complexity_summary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        avg_lcom4 = complexity_summary.get("avg_lcom4", 0.0)
        avg_tcc = complexity_summary.get("avg_tcc", 0.0)
        avg_complexity = complexity_summary.get("avg_complexity_score", 0.0)
        complex_classes = complexity_summary.get("complex_classes", [])

        # LCOM4 recommendations
        if avg_lcom4 > 0.5:
            recommendations.append(
                f"High LCOM4 ({avg_lcom4:.2f}) indicates poor class cohesion. "
                "Consider splitting classes or improving method-field relationships."
            )

        # TCC recommendations
        if avg_tcc < 0.3:
            recommendations.append(
                f"Low TCC ({avg_tcc:.2f}) suggests weak method cohesion. "
                "Review method responsibilities and field usage patterns."
            )

        # Overall complexity recommendations
        if avg_complexity > 0.6:
            recommendations.append(
                f"High average complexity score ({avg_complexity:.2f}). "
                "Consider refactoring to reduce component complexity."
            )

        # Specific class recommendations
        if complex_classes:
            recommendations.append(
                f"Classes requiring attention due to high complexity: {', '.join(complex_classes[:5])}"
                + (f" and {len(complex_classes) - 5} more" if len(complex_classes) > 5 else "")
            )

        return recommendations
