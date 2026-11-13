"""Module for detecting performance bottlenecks and scoring algorithms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx

from .change_frequency_analyzer import GitAnalyzer
from .complexity_analyzer import ComplexityAnalyzer
from .detection_algorithms import HotspotDetector

logger = logging.getLogger(__name__)


class BottleneckMetrics:
    """Represents performance bottleneck metrics for a component."""

    def __init__(self) -> None:
        """Initialize bottleneck metrics with default values."""
        self.coupling_score: float = 0.0
        self.dependency_count: int = 0
        self.centrality_score: float = 0.0
        self.hotpath_score: float = 0.0
        self.resource_intensity: float = 0.0
        self.scalability_risk: float = 0.0

    def get_composite_score(self) -> float:
        """Calculate composite bottleneck score.

        Returns:
            Weighted composite score representing bottleneck likelihood
        """
        weights = {
            "coupling": 0.25,
            "dependency": 0.20,
            "centrality": 0.20,
            "hotpath": 0.15,
            "resource": 0.10,
            "scalability": 0.10,
        }

        score = (
            self.coupling_score * weights["coupling"]
            + (self.dependency_count / 50.0) * weights["dependency"]  # Normalize
            + self.centrality_score * weights["centrality"]
            + self.hotpath_score * weights["hotpath"]
            + self.resource_intensity * weights["resource"]
            + self.scalability_risk * weights["scalability"]
        )

        return min(1.0, max(0.0, score))


class BottleneckDetector:
    """Detects performance bottlenecks using graph analysis and metrics.

    This class identifies potential performance bottlenecks by analyzing
    structural patterns in the codebase graph, including high coupling,
    centrality measures, dependency patterns, and performance-critical paths.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        """Initialize the bottleneck detector.

        Args:
            repo_path: Optional path to the repository for enhanced analysis
        """
        self.repo_path = repo_path
        self.git_analyzer = GitAnalyzer(repo_path) if repo_path else None
        self.complexity_analyzer = ComplexityAnalyzer()
        self.hotspot_detector = HotspotDetector()

    def detect_bottlenecks(self, graph: nx.DiGraph, threshold: float = 0.6) -> list[dict[str, Any]]:
        """Detect potential performance bottlenecks in the code graph.

        Args:
            graph: NetworkX directed graph representing the codebase
            threshold: Minimum score threshold for bottleneck classification

        Returns:
            List of dictionaries containing bottleneck information
        """
        logger.info("Starting bottleneck detection analysis")

        bottlenecks = []
        centrality_scores = self._calculate_centrality_scores(graph)

        for node_id, node_data in graph.nodes(data=True):
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Calculate bottleneck metrics
            metrics = self._calculate_bottleneck_metrics(
                graph, node_id, file_path, centrality_scores
            )

            composite_score = metrics.get_composite_score()

            if composite_score >= threshold:
                bottleneck_info = {
                    "component_id": node_id,
                    "file_path": file_path,
                    "score": round(composite_score, 3),
                    "metrics": {
                        "coupling_score": round(metrics.coupling_score, 3),
                        "dependency_count": metrics.dependency_count,
                        "centrality_score": round(metrics.centrality_score, 3),
                        "hotpath_score": round(metrics.hotpath_score, 3),
                        "resource_intensity": round(metrics.resource_intensity, 3),
                        "scalability_risk": round(metrics.scalability_risk, 3),
                    },
                    "risk_factors": self._identify_bottleneck_risks(metrics),
                    "recommendations": self._generate_bottleneck_recommendations(metrics),
                }
                bottlenecks.append(bottleneck_info)

        # Sort by score descending
        bottlenecks.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Identified {len(bottlenecks)} potential bottlenecks")

        return bottlenecks

    def _calculate_centrality_scores(self, graph: nx.DiGraph) -> dict[str, float]:
        """Calculate various centrality measures for graph nodes.

        Args:
            graph: NetworkX directed graph

        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        centrality_scores = {}

        try:
            # Betweenness centrality - identifies bridge nodes
            betweenness = nx.betweenness_centrality(graph)

            # Closeness centrality - identifies central nodes
            closeness = nx.closeness_centrality(graph)

            # PageRank - identifies important nodes
            pagerank = nx.pagerank(graph)

            # Combine centrality measures
            for node in graph.nodes():
                composite_centrality = (
                    betweenness.get(node, 0.0) * 0.4
                    + closeness.get(node, 0.0) * 0.3
                    + pagerank.get(node, 0.0) * 0.3
                )
                centrality_scores[node] = composite_centrality

        except Exception as e:
            logger.warning(f"Failed to calculate centrality scores: {e}")
            # Fallback to degree centrality
            for node in graph.nodes():
                centrality_scores[node] = graph.degree(node) / max(1, len(graph.nodes()))

        return centrality_scores

    def _calculate_bottleneck_metrics(
        self,
        graph: nx.DiGraph,
        node_id: str,
        file_path: str,
        centrality_scores: dict[str, float],
    ) -> BottleneckMetrics:
        """Calculate comprehensive bottleneck metrics for a node.

        Args:
            graph: NetworkX directed graph
            node_id: Node identifier in the graph
            file_path: File path for the node
            centrality_scores: Pre-calculated centrality scores

        Returns:
            BottleneckMetrics object with calculated metrics
        """
        metrics = BottleneckMetrics()

        # Coupling score based on in/out degree
        in_degree = graph.in_degree(node_id)
        out_degree = graph.out_degree(node_id)
        total_degree = in_degree + out_degree
        max_possible = len(graph.nodes()) - 1
        metrics.coupling_score = total_degree / max(1, max_possible)

        # Dependency count
        metrics.dependency_count = total_degree

        # Centrality score
        metrics.centrality_score = centrality_scores.get(node_id, 0.0)

        # Hot path score (based on change frequency if available)
        if self.git_analyzer:
            change_freq = self.git_analyzer.get_change_frequency(file_path)
            metrics.hotpath_score = self.hotspot_detector.normalize_feature(change_freq, 50)
        else:
            metrics.hotpath_score = 0.0

        # Resource intensity (based on complexity if file exists)
        if self.repo_path and Path(self.repo_path / file_path).exists():
            try:
                content = Path(self.repo_path / file_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
                language = self.hotspot_detector.detect_language(file_path)
                complexity = self.complexity_analyzer.calculate_complexity(
                    Path(file_path), content, language
                )

                # Estimate resource intensity from complexity and size
                loc = complexity.get("loc", 0)
                cyclomatic = complexity.get("cyclomatic_complexity", 0)
                metrics.resource_intensity = self.hotspot_detector.normalize_feature(
                    loc * 0.1 + cyclomatic * 2, 100
                )
            except Exception as e:
                logger.debug(f"Failed to analyze resource intensity for {file_path}: {e}")
                metrics.resource_intensity = 0.0

        # Scalability risk (combination of coupling and complexity)
        metrics.scalability_risk = metrics.coupling_score * 0.6 + metrics.resource_intensity * 0.4

        return metrics

    def _identify_bottleneck_risks(self, metrics: BottleneckMetrics) -> list[str]:
        """Identify specific bottleneck risk factors.

        Args:
            metrics: BottleneckMetrics object with calculated values

        Returns:
            List of human-readable risk factor descriptions
        """
        risk_factors = []

        if metrics.coupling_score > 0.7:
            risk_factors.append(f"High coupling ({int(metrics.coupling_score * 100)}%)")

        if metrics.dependency_count > 20:
            risk_factors.append(f"Many dependencies ({metrics.dependency_count})")

        if metrics.centrality_score > 0.5:
            risk_factors.append(f"Critical path component ({int(metrics.centrality_score * 100)}%)")

        if metrics.hotpath_score > 0.6:
            risk_factors.append(f"Frequently executed ({int(metrics.hotpath_score * 100)}%)")

        if metrics.resource_intensity > 0.6:
            risk_factors.append(f"Resource intensive ({int(metrics.resource_intensity * 100)}%)")

        if metrics.scalability_risk > 0.7:
            risk_factors.append(f"Scalability concern ({int(metrics.scalability_risk * 100)}%)")

        return risk_factors

    def _generate_bottleneck_recommendations(self, metrics: BottleneckMetrics) -> list[str]:
        """Generate actionable recommendations for addressing bottlenecks.

        Args:
            metrics: BottleneckMetrics object with calculated values

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if metrics.coupling_score > 0.7:
            recommendations.append(
                "Reduce coupling by extracting interfaces and using dependency injection"
            )

        if metrics.dependency_count > 20:
            recommendations.append("Consider dependency inversion to reduce direct dependencies")

        if metrics.centrality_score > 0.5:
            recommendations.append("Add caching or optimize this critical path component")

        if metrics.resource_intensity > 0.6:
            recommendations.append("Profile and optimize resource-intensive operations")

        if metrics.scalability_risk > 0.7:
            recommendations.append("Design for horizontal scaling and consider async processing")

        if not recommendations:
            recommendations.append("Monitor performance metrics for this component")

        return recommendations

    def analyze_dependency_chains(
        self, graph: nx.DiGraph, max_depth: int = 5
    ) -> list[dict[str, Any]]:
        """Analyze long dependency chains that could create bottlenecks.

        Args:
            graph: NetworkX directed graph
            max_depth: Maximum depth to analyze for chains

        Returns:
            List of dependency chain analysis results
        """
        logger.info("Analyzing dependency chains")

        chains = []

        for node in graph.nodes():
            try:
                # Find longest paths from this node
                paths = self._find_longest_paths(graph, node, max_depth)

                for path in paths:
                    if len(path) >= 3:  # Only consider chains of 3+ components
                        chain_info = {
                            "start_node": path[0],
                            "end_node": path[-1],
                            "length": len(path),
                            "path": path,
                            "risk_score": self._calculate_chain_risk(graph, path),
                        }
                        chains.append(chain_info)

            except Exception as e:
                logger.debug(f"Failed to analyze chains for node {node}: {e}")

        # Sort by risk score and length
        chains.sort(key=lambda x: (x["risk_score"], x["length"]), reverse=True)

        return chains[:20]  # Return top 20 chains

    def _find_longest_paths(
        self, graph: nx.DiGraph, start_node: str, max_depth: int
    ) -> list[list[str]]:
        """Find longest dependency paths from a starting node.

        Args:
            graph: NetworkX directed graph
            start_node: Starting node for path finding
            max_depth: Maximum depth to explore

        Returns:
            List of paths (each path is a list of node IDs)
        """
        paths = []

        def dfs(current_node: str, path: list[str], depth: int) -> None:
            if depth >= max_depth:
                return

            path.append(current_node)

            successors = list(graph.successors(current_node))
            if not successors:
                # End of path - save it
                if len(path) > 1:
                    paths.append(path.copy())
            else:
                for successor in successors:
                    if successor not in path:  # Avoid cycles
                        dfs(successor, path, depth + 1)

            path.pop()

        dfs(start_node, [], 0)
        return paths

    def _calculate_chain_risk(self, graph: nx.DiGraph, path: list[str]) -> float:
        """Calculate risk score for a dependency chain.

        Args:
            graph: NetworkX directed graph
            path: List of node IDs representing the dependency chain

        Returns:
            Risk score between 0.0 and 1.0
        """
        if len(path) < 2:
            return 0.0

        # Factors that increase chain risk:
        # 1. Length of chain
        # 2. Coupling of nodes in chain
        # 3. Complexity of nodes (if available)

        length_risk = min(1.0, len(path) / 10.0)  # Normalize to 10 nodes

        coupling_risk = 0.0
        for node in path:
            node_coupling = (graph.in_degree(node) + graph.out_degree(node)) / max(
                1, len(graph.nodes())
            )
            coupling_risk += node_coupling
        coupling_risk = coupling_risk / len(path)  # Average coupling

        # Combine risk factors
        total_risk = length_risk * 0.6 + coupling_risk * 0.4

        return min(1.0, max(0.0, total_risk))
