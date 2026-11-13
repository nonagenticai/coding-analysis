"""Design pattern detection module for common software design patterns.

This module provides the DesignPatternDetector class which analyzes code graphs to identify
traditional design patterns including Singleton, Factory, Observer, MVC, Repository,
and Service patterns.
"""

from __future__ import annotations

import logging

import networkx as nx

from models.architecture_models import DetectedPattern

from .code_pattern_recognition import CodePatternRecognizer

logger = logging.getLogger(__name__)


class DesignPatternDetector:
    """Detect traditional design patterns in code graphs."""

    def __init__(self, recognizer: CodePatternRecognizer) -> None:
        """Initialize the design pattern detector.

        Args:
            recognizer: Code pattern recognition utility instance
        """
        self._recognizer = recognizer

    def detect_all_design_patterns(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect all built-in design patterns.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected design patterns
        """
        patterns = []

        # Detect common design patterns
        patterns.extend(self._detect_singleton_pattern(graph))
        patterns.extend(self._detect_factory_pattern(graph))
        patterns.extend(self._detect_observer_pattern(graph))
        patterns.extend(self._detect_mvc_pattern(graph))
        patterns.extend(self._detect_repository_pattern(graph))
        patterns.extend(self._detect_service_pattern(graph))

        return patterns

    def _detect_singleton_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Singleton pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Singleton patterns
        """
        patterns = []
        singleton_regex = self._recognizer.get_compiled_pattern(
            r"class\s+\w*Singleton\w*|getInstance\s*\("
        )

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, singleton_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Singleton",
                        confidence=0.8,
                        components=[node_id],
                        description="Singleton design pattern detected",
                        name="Singleton",
                        location=node_id,
                    )
                )

        return patterns

    def _detect_factory_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Factory pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Factory patterns
        """
        patterns = []
        factory_regex = self._recognizer.get_compiled_pattern(
            r"class\s+\w*Factory\w*|create\w*\s*\("
        )

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, factory_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Factory",
                        confidence=0.7,
                        components=[node_id],
                        description="Factory design pattern detected",
                        name="Factory",
                        location=node_id,
                    )
                )

        return patterns

    def _detect_observer_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Observer pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Observer patterns
        """
        patterns = []
        observer_regex = self._recognizer.get_compiled_pattern(
            r"class\s+\w*Observer\w*|notify\w*\s*\(|subscribe\s*\("
        )

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, observer_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Observer",
                        confidence=0.7,
                        components=[node_id],
                        description="Observer design pattern detected",
                        name="Observer",
                        location=node_id,
                    )
                )

        return patterns

    def _detect_mvc_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect MVC pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected MVC patterns
        """
        patterns = []
        mvc_regex = self._recognizer.get_compiled_pattern(r"class\s+\w*(Controller|Model|View)\w*")

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, mvc_regex):
                # Determine which part of MVC
                code = self._recognizer.get_node_code(node_data)
                if "Controller" in code:
                    pattern_name = "MVC Controller"
                elif "Model" in code:
                    pattern_name = "MVC Model"
                elif "View" in code:
                    pattern_name = "MVC View"
                else:
                    pattern_name = "MVC Component"

                patterns.append(
                    DetectedPattern(
                        pattern_type=pattern_name,
                        confidence=0.8,
                        components=[node_id],
                        description=f"{pattern_name} pattern detected",
                        name=pattern_name,
                        location=node_id,
                    )
                )

        return patterns

    def _detect_repository_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Repository pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Repository patterns
        """
        patterns = []
        repo_regex = self._recognizer.get_compiled_pattern(
            r"class\s+\w*Repository\w*|interface\s+\w*Repository\w*"
        )

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, repo_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Repository",
                        confidence=0.8,
                        components=[node_id],
                        description="Repository design pattern detected",
                        name="Repository",
                        location=node_id,
                    )
                )

        return patterns

    def _detect_service_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Service pattern.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Service patterns
        """
        patterns = []
        service_regex = self._recognizer.get_compiled_pattern(
            r"class\s+\w*Service\w*|interface\s+\w*Service\w*"
        )

        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, service_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Service",
                        confidence=0.7,
                        components=[node_id],
                        description="Service layer pattern detected",
                        name="Service",
                        location=node_id,
                    )
                )

        return patterns
