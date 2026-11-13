"""Module for analyzing code hotspots to identify potential issues."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx

from models.architecture_models import HotspotPrediction
from processing.architecture.hotspot.bottleneck_detector import BottleneckDetector
from processing.architecture.hotspot.change_frequency_analyzer import GitAnalyzer
from processing.architecture.hotspot.complexity_analyzer import ComplexityAnalyzer
from processing.architecture.hotspot.detection_algorithms import HotspotDetector
from processing.architecture.hotspot.hotspot_reporter import (
    HotspotReport,
    HotspotReporter,
)

"""
Hotspot Analyzer Module

This module provides functionality for analyzing code hotspots in a codebase.
It orchestrates multiple analysis components including change frequency analysis,
complexity analysis, bottleneck detection, and hotspot reporting to provide
comprehensive insights into code maintainability and technical debt.
"""

logger = logging.getLogger(__name__)


class HotspotAnalyzer:
    """Comprehensive hotspot analyzer combining multiple metrics for risk assessment.

    This is the main orchestrator class that combines Git history analysis and static
    code complexity metrics to identify potential hotspots in a codebase. Hotspots
    are files or components that exhibit patterns suggesting they may become sources
    of maintenance problems, bugs, or technical debt.

    The analyzer employs a multi-dimensional scoring approach that considers:
    - Code complexity metrics (cyclomatic complexity, nesting depth)
    - Change frequency patterns from version control history
    - Bug fix frequency as an indicator of quality issues
    - Developer knowledge distribution (author count and turnover)
    - Code churn metrics (volume of changes over time)
    - Recency weighting to prioritize current concerns

    Each metric is normalized and weighted according to empirically-derived importance
    factors. The resulting composite score ranges from 0.0 to 1.0, with higher scores
    indicating greater likelihood of being a problematic hotspot.

    Beyond scoring, the analyzer provides interpretable results including:
    - Predicted issue types (bug-prone, high-complexity, unstable)
    - Specific risk factors with quantified severity
    - Actionable recommendations tailored to the identified problems
    - Confidence levels based on data availability and signal strength
    - Performance bottleneck detection
    - Comprehensive reporting and visualization

    Attributes:
        repo_path (Optional[Path]): Path to the repository being analyzed
        git_analyzer (GitAnalyzer): Component for Git history analysis
        complexity_analyzer (ComplexityAnalyzer): Component for code complexity analysis
        hotspot_detector (HotspotDetector): Component for hotspot detection algorithms
        bottleneck_detector (BottleneckDetector): Component for bottleneck detection
        reporter (HotspotReporter): Component for generating reports and visualizations
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        """
        Initialize the hotspot analyzer with repository path and analysis components.

        Creates a new HotspotAnalyzer instance and initializes the underlying Git
        and complexity analysis components. If a repository path is provided, Git
        analysis begins immediately to build the change metrics cache.

        Args:
            repo_path (Optional[Path]): Path to the Git repository to analyze.
                If None, only complexity analysis will be available and Git-based
                metrics will return default values.

        Raises:
            ValueError: If repo_path is provided but is not a valid Git repository
            OSError: If repo_path is provided but is not accessible
        """
        self.repo_path = repo_path
        self.git_analyzer = GitAnalyzer(repo_path) if repo_path else None
        self.complexity_analyzer = ComplexityAnalyzer()
        self.hotspot_detector = HotspotDetector()
        self.bottleneck_detector = BottleneckDetector(repo_path)
        self.reporter = HotspotReporter()

    def analyze(
        self,
        graph: nx.DiGraph,
        codebase_id: str,
        repo_path: Path | None = None,
        include_bottlenecks: bool = True,
    ) -> list[HotspotPrediction]:
        """
        Analyze a code graph to identify and rank potential hotspots.

        Performs comprehensive hotspot analysis on the provided code graph by
        examining each file node and calculating a composite risk score based on
        multiple metrics. The analysis combines static code complexity with
        dynamic Git history patterns to identify files that may require attention.

        The method processes all file nodes in the graph, calculates hotspot scores,
        and generates detailed predictions for files that exceed the significance
        threshold. Results are ranked by score to prioritize the most concerning
        hotspots.

        Args:
            graph (nx.DiGraph): NetworkX directed graph representing the codebase
                structure. Nodes should contain 'file_path' attributes identifying
                the files to analyze.
            codebase_id (str): Unique identifier for the codebase being analyzed.
                Used for logging and tracking purposes.
            repo_path (Optional[Path]): Alternative repository path for this analysis.
                If provided, will reinitialize Git analysis for this specific path.
            include_bottlenecks (bool): Whether to include bottleneck analysis.

        Returns:
            List[HotspotPrediction]: List of hotspot predictions sorted by descending
                score. Each prediction includes score, confidence, predicted issues,
                risk factors, and actionable recommendations. Limited to top 50 results.

        Raises:
            ValueError: If the graph contains invalid node data
            OSError: If file access fails during complexity analysis
        """
        logger.info(f"Starting comprehensive hotspot analysis for codebase: {codebase_id}")

        # Reinitialize git analyzer if new repo path provided
        if repo_path and repo_path != self.repo_path:
            self.git_analyzer = GitAnalyzer(repo_path)
            self.repo_path = repo_path
            # Update bottleneck detector with new repo path
            self.bottleneck_detector = BottleneckDetector(repo_path)

        hotspots: list[HotspotPrediction] = []
        file_scores: dict[str, float] = {}

        # Analyze each file in the graph
        for node_id, node_data in graph.nodes(data=True):
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Calculate hotspot score using all available metrics
            score, features = self._calculate_hotspot_score(file_path)
            file_scores[file_path] = score

            # Create hotspot prediction if score is significant
            if self.hotspot_detector.is_hotspot(score):
                hotspot = self.hotspot_detector.create_hotspot_prediction(
                    component_id=node_id,
                    score=score,
                    features=features,
                )
                hotspots.append(hotspot)

        # Sort by score and return top hotspots
        hotspots.sort(key=lambda h: h.score, reverse=True)
        logger.info(f"Identified {len(hotspots)} hotspots out of {len(file_scores)} files")

        return hotspots[:50]  # Return top 50 hotspots

    def analyze_with_report(
        self,
        graph: nx.DiGraph,
        codebase_id: str,
        repo_path: Path | None = None,
        include_bottlenecks: bool = True,
    ) -> HotspotReport:
        """
        Perform comprehensive hotspot analysis and generate a detailed report.

        This method combines hotspot analysis with bottleneck detection and
        generates a comprehensive report suitable for stakeholder consumption.

        Args:
            graph (nx.DiGraph): NetworkX directed graph representing the codebase
            codebase_id (str): Unique identifier for the codebase being analyzed
            repo_path (Optional[Path]): Alternative repository path for analysis
            include_bottlenecks (bool): Whether to include bottleneck analysis

        Returns:
            HotspotReport: Comprehensive report containing all analysis results
        """
        logger.info(f"Starting comprehensive analysis with reporting for: {codebase_id}")

        # Perform hotspot analysis
        hotspots = self.analyze(graph, codebase_id, repo_path, include_bottlenecks)

        # Perform bottleneck analysis if requested
        bottlenecks = []
        if include_bottlenecks:
            try:
                bottlenecks = self.bottleneck_detector.detect_bottlenecks(graph)
                logger.info(f"Detected {len(bottlenecks)} potential bottlenecks")
            except Exception as e:
                logger.warning(f"Bottleneck analysis failed: {e}")

        # Generate analysis metadata
        metadata = {
            "repository_path": str(repo_path) if repo_path else None,
            "total_nodes": len(graph.nodes()),
            "total_edges": len(graph.edges()),
            "git_analysis_enabled": self.git_analyzer is not None,
            "bottleneck_analysis_enabled": include_bottlenecks,
        }

        # Generate comprehensive report
        report = self.reporter.generate_report(
            codebase_id=codebase_id,
            hotspots=hotspots,
            bottlenecks=bottlenecks,
            analysis_metadata=metadata,
        )

        logger.info(f"Generated comprehensive report for {codebase_id}")
        return report

    def _calculate_hotspot_score(
        self,
        file_path: str,
    ) -> tuple[float, dict[str, float]]:
        """Calculate comprehensive hotspot score for a specific file.

        Computes a weighted composite score by combining multiple normalized metrics
        that correlate with maintenance problems and technical debt. The calculation
        integrates both static code analysis and dynamic Git history patterns to
        provide a holistic risk assessment.

        Args:
            file_path (str): Relative path to the file from repository root.

        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing:
                - score (float): Composite hotspot score between 0.0 and 1.0
                - features (Dict[str, float]): Dictionary of individual normalized
                  feature values used in the calculation
        """
        features = {}

        # Git-based features
        if self.git_analyzer:
            features["change_frequency"] = self.hotspot_detector.normalize_feature(
                self.git_analyzer.get_change_frequency(file_path), 50
            )
            features["bug_frequency"] = self.hotspot_detector.normalize_feature(
                self.git_analyzer.get_bug_fix_count(file_path), 10
            )
            features["author_count"] = self.hotspot_detector.normalize_feature(
                self.git_analyzer.get_unique_authors(file_path), 10
            )
            features["code_churn"] = self.hotspot_detector.normalize_feature(
                self.git_analyzer.get_code_churn(file_path), 1000
            )
            features["recency"] = self.git_analyzer.get_recency_score(file_path)
        else:
            # Default values if no git analysis
            features["change_frequency"] = 0.0
            features["bug_frequency"] = 0.0
            features["author_count"] = 0.0
            features["code_churn"] = 0.0
            features["recency"] = 0.0

        # Complexity features
        if self.repo_path and Path(self.repo_path / file_path).exists():
            try:
                content = Path(self.repo_path / file_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
                language = self.hotspot_detector.detect_language(file_path)
                complexity_metrics = self.complexity_analyzer.calculate_complexity(
                    Path(file_path), content, language
                )

                # Normalize complexity score
                complexity_score = (
                    self.hotspot_detector.normalize_feature(
                        complexity_metrics["cyclomatic_complexity"], 20
                    )
                    * 0.4
                    + self.hotspot_detector.normalize_feature(
                        complexity_metrics.get("cognitive_complexity", 0), 30
                    )
                    * 0.3
                    + self.hotspot_detector.normalize_feature(
                        complexity_metrics["max_nesting_depth"], 5
                    )
                    * 0.3
                )
                features["complexity"] = complexity_score

            except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
                logger.debug(f"Failed to analyze complexity for {file_path}: {e}")
                features["complexity"] = 0.0
        else:
            features["complexity"] = 0.0

        # Calculate weighted score using hotspot detector
        score = self.hotspot_detector.calculate_hotspot_score(features)
        return score, features

    def detect_bottlenecks(self, graph: nx.DiGraph, threshold: float = 0.6) -> list[dict[str, Any]]:
        """Detect potential performance bottlenecks in the code graph.

        Args:
            graph: NetworkX directed graph representing the codebase
            threshold: Minimum score threshold for bottleneck classification

        Returns:
            List of dictionaries containing bottleneck information
        """
        return self.bottleneck_detector.detect_bottlenecks(graph, threshold)

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
        return self.bottleneck_detector.analyze_dependency_chains(graph, max_depth)

    def save_report(
        self,
        report: HotspotReport,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> None:
        """Save the hotspot report in specified formats.

        Args:
            report: HotspotReport to save
            output_dir: Directory where to save the report files
            formats: List of formats to save ('json', 'text'). Defaults to both.
        """
        if formats is None:
            formats = ["json", "text"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if "json" in formats:
            json_path = output_dir / f"hotspot_report_{report.codebase_id}.json"
            self.reporter.save_report_json(report, json_path)

        if "text" in formats:
            text_path = output_dir / f"hotspot_report_{report.codebase_id}.txt"
            self.reporter.save_report_text(report, text_path)

    def get_visualization_data(self, report: HotspotReport) -> dict[str, Any]:
        """Get data structures suitable for visualization tools.

        Args:
            report: HotspotReport to convert for visualization

        Returns:
            Dictionary containing visualization-ready data
        """
        return self.reporter.generate_visualization_data(report)

    def get_dashboard_summary(self, report: HotspotReport) -> dict[str, Any]:
        """Create a summary suitable for dashboard display.

        Args:
            report: HotspotReport to summarize

        Returns:
            Dictionary containing dashboard-ready summary data
        """
        return self.reporter.create_dashboard_summary(report)
