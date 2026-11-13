"""Module for calculating architectural metrics for code analysis."""

from __future__ import annotations

import datetime
import logging
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx: Any = None

from models.architecture_models import ArchitecturalMetric, Component
from processing.architecture.architecture_metrics_analyzer import (
    ArchitectureMetricsAnalyzer,
)
from processing.architecture.code_metrics_calculator import CodeMetricsCalculator
from processing.architecture.complexity_metrics_computer import (
    ComplexityMetricsComputer,
)
from processing.architecture.metrics_aggregator import MetricsAggregator
from processing.architecture.performance_metrics_tracker import (
    MetricTimeSeries,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate architectural metrics following software engineering principles.

    This class provides methods to calculate various architectural and design metrics:
    - Robert C. Martin's metrics (instability, abstractness, distance from main sequence)
    - Cohesion metrics (LCOM4, TCC)
    - Component-level coupling metrics
    - Performance tracking over time
    - System-wide aggregation and reporting

    These metrics help assess the quality of software architecture and guide refactoring
    decisions to update maintainability and extensibility.
    """

    def __init__(self):
        """Initialize the metrics calculator with focused sub-calculators."""
        self.architecture_analyzer = ArchitectureMetricsAnalyzer()
        self.code_calculator = CodeMetricsCalculator()
        self.complexity_computer = ComplexityMetricsComputer()
        self.aggregator = MetricsAggregator()

    def calc(self, graph: Any, components: list[Component]) -> list[ArchitecturalMetric]:
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
        return self.architecture_analyzer.calc(graph, components)

    def calculate_component_metrics(
        self, components: list[Component], graph: Any
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

        # Get basic architectural metrics
        arch_metrics = self.architecture_analyzer.calculate_component_metrics(components, graph)

        # Get component node mapping
        component_nodes_map = self.code_calculator.map_nodes_to_components(components, graph)

        # Add complexity metrics
        for component_id, nodes in component_nodes_map.items():
            if component_id in arch_metrics:
                # Add LCOM4 and TCC metrics
                lcom4 = self.complexity_computer.calculate_lcom4(graph, nodes)
                tcc = self.complexity_computer.calculate_tcc(graph, nodes)

                arch_metrics[component_id]["lcom4"] = lcom4
                arch_metrics[component_id]["tcc"] = tcc

        return arch_metrics

    def get_metrics_summary(self, components: list[Component], graph: Any) -> dict[str, float]:
        """Get a summary of all architectural metrics.

        Args:
            components: List of components to analyze
            graph: Directed graph representing code relationships

        Returns:
            Dictionary of summarized metric values
        """
        if not components:
            return {
                "total_components": 0,
                "avg_instability": 0.0,
                "avg_abstractness": 0.0,
                "avg_distance": 0.0,
                "max_distance": 0.0,
                "components_in_main_sequence": 0,
                "avg_cohesion_lcom4": 0.0,
                "avg_cohesion_tcc": 0.0,
            }

        component_metrics = self.calculate_component_metrics(components, graph)

        return self.aggregator.generate_system_metrics_summary(
            components,
            component_metrics,
            self.architecture_analyzer._calculate_weighted_quality_score,
        )

    def _calculate_lcom4(self, graph: Any, component_nodes: list[str]) -> float:
        """Calculate LCOM4 (Lack of Cohesion of Methods) metric.

        Args:
            graph: The code graph containing method and field relationships
            component_nodes: List of node IDs belonging to the component

        Returns:
            LCOM4 value normalized between 0 and 1 (lower is better cohesion)
        """
        return self.complexity_computer.calculate_lcom4(graph, component_nodes)

    def _calculate_tcc(self, graph: Any, component_nodes: list[str]) -> float:
        """Calculate TCC (Tight Class Cohesion) metric.

        Args:
            graph: The code graph containing method relationships
            component_nodes: List of node IDs belonging to the component

        Returns:
            TCC value between 0 and 1 (higher is better cohesion)
        """
        return self.complexity_computer.calculate_tcc(graph, component_nodes)

    def _calculate_weighted_quality_score(self, metrics: dict[str, dict[str, float]]) -> float:
        """Calculate a weighted quality score based on all metrics.

        Args:
            metrics: Dictionary of component metrics

        Returns:
            A weighted quality score between 0 and 1
        """
        return self.architecture_analyzer._calculate_weighted_quality_score(metrics)

    # ----------------------------------------
    # Historical Analysis & Time Series Methods
    # ----------------------------------------

    def compare_metrics_across_commits(
        self,
        component_id: str,
        commit_data: list[tuple[str, datetime.datetime, list[Component], nx.DiGraph]],
    ) -> dict[str, MetricTimeSeries]:
        """Compare metrics for a component across multiple git commits.

        Args:
            component_id: ID of the component to analyze
            commit_data: List of tuples (commit_id, timestamp, components, graph) for each commit

        Returns:
            Dictionary of metric name to MetricTimeSeries objects
        """
        return self.aggregator.compare_metrics_across_commits(
            component_id,
            commit_data,
            self.calculate_component_metrics,
            self._calculate_weighted_quality_score,
        )

    def get_architecture_evolution_report(
        self, component_id: str, time_series_data: dict[str, MetricTimeSeries]
    ) -> dict[str, Any]:
        """Generate a comprehensive report on architectural evolution for a component.

        Args:
            component_id: ID of the component being analyzed
            time_series_data: Dictionary of metric name to MetricTimeSeries objects

        Returns:
            Dictionary with architecture evolution analysis
        """
        return self.aggregator.get_architecture_evolution_report(component_id, time_series_data)

    def get_historical_data_for_component(
        self,
        component_id: str,
        git_history_data: dict[str, Any],
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """Get historical data for visualization of component metrics over time.

        Args:
            component_id: ID of the component to analyze
            git_history_data: Dictionary with git history data from compare_metrics_across_commits
            metric_name: Optional specific metric to retrieve data for

        Returns:
            Dictionary with formatted data for visualization
        """
        return self.aggregator.get_historical_data_for_component(
            component_id, git_history_data, metric_name
        )

    def detect_architectural_decay(
        self,
        component_ids: list[str],
        time_series_collection: dict[str, dict[str, MetricTimeSeries]],
    ) -> dict[str, Any]:
        """Detect architectural decay across multiple components over time.

        Args:
            component_ids: List of component IDs to analyze
            time_series_collection: Dict mapping component IDs to their metric time series data

        Returns:
            Dictionary with decay analysis results
        """
        return self.aggregator.detect_architectural_decay(component_ids, time_series_collection)

    # ----------------------------------------
    # Validation and Basic Metrics Methods
    # ----------------------------------------

    def validate_components_and_graph(
        self, components: list[Component], graph: Any
    ) -> dict[str, Any]:
        """Validate that components and graph are properly structured for metric calculation.

        Args:
            components: List of components to validate
            graph: Code graph to validate

        Returns:
            Dictionary with validation results and basic statistics
        """
        return self.code_calculator.validate_components_and_graph(components, graph)

    def calculate_system_overview(self, components: list[Component], graph: Any) -> dict[str, Any]:
        """Calculate system-level overview metrics.

        Args:
            components: List of all components
            graph: Complete code graph

        Returns:
            Dictionary with system overview data
        """
        return self.code_calculator.calculate_system_overview(components, graph)

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
        return self.code_calculator.get_component_health_indicators(component_id, basic_metrics)

    # ----------------------------------------
    # Complexity Analysis Methods
    # ----------------------------------------

    def calculate_class_complexity_metrics(self, graph: Any, class_node: str) -> dict[str, float]:
        """Calculate comprehensive complexity metrics for a single class.

        Args:
            graph: Code relationship graph
            class_node: Node ID of the class

        Returns:
            Dictionary of complexity metrics for the class
        """
        return self.complexity_computer.calculate_class_complexity_metrics(graph, class_node)

    def calculate_component_complexity_summary(
        self, graph: Any, component_nodes: list[str]
    ) -> dict[str, Any]:
        """Calculate a comprehensive complexity summary for a component.

        Args:
            graph: Code relationship graph
            component_nodes: List of node IDs belonging to the component

        Returns:
            Dictionary with component complexity analysis
        """
        return self.complexity_computer.calculate_component_complexity_summary(
            graph, component_nodes
        )

    def get_complexity_recommendations(self, complexity_summary: dict[str, Any]) -> list[str]:
        """Generate recommendations based on complexity analysis.

        Args:
            complexity_summary: Component complexity summary

        Returns:
            List of recommendation strings
        """
        return self.complexity_computer.get_complexity_recommendations(complexity_summary)

    # ----------------------------------------
    # Architecture Analysis Methods
    # ----------------------------------------

    def analyze_dependency_patterns(
        self, components: list[Component], graph: Any
    ) -> dict[str, Any]:
        """Analyze dependency patterns across components.

        Args:
            components: List of components to analyze
            graph: Code graph containing dependency relationships

        Returns:
            Dictionary with dependency analysis results
        """
        return self.architecture_analyzer.analyze_dependency_patterns(components, graph)

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
        return self.architecture_analyzer.get_component_recommendations(component_id, metrics)
