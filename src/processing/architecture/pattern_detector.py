"""Pattern detection module for architectural and design patterns in code graphs.

This module provides the ArchitecturePatternDetector class which analyzes code graphs
to identify various architectural patterns including design patterns (Singleton, Factory,
Observer, MVC, Repository, Service) and cloud-native patterns (Sidecar, Ambassador,
Adapter, Circuit Breaker, Service Mesh, API Gateway).

The module has been refactored into focused components:
- CodePatternRecognizer: Core pattern recognition utilities
- DesignPatternDetector: Traditional design pattern detection
- ArchitecturePatternAnalyzer: Cloud-native pattern analysis
- PatternValidationScorer: ML-based confidence scoring and validation
- PatternReportingUtils: Statistics, filtering, and reporting
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import networkx as nx

from models.architecture_models import DetectedPattern, RepoConfig
from processing.architecture.architecture_pattern_analyzer import (
    ArchitecturePatternAnalyzer,
)
from processing.architecture.code_pattern_recognition import CodePatternRecognizer
from processing.architecture.design_pattern_detector import DesignPatternDetector
from processing.architecture.pattern_reporting_utils import PatternReportingUtils
from processing.architecture.pattern_validation_scoring import (
    PatternValidationScorer,
)

logger = logging.getLogger(__name__)


class ArchitecturePatternDetector:
    """Detect architectural and design patterns in code graphs."""

    def __init__(self) -> None:
        """Initialize the pattern detector with specialized components."""
        # Initialize core components
        self._recognizer = CodePatternRecognizer()
        self._design_detector = DesignPatternDetector(self._recognizer)
        self._architecture_analyzer = ArchitecturePatternAnalyzer(self._recognizer)
        self._validator_scorer = PatternValidationScorer()
        self._reporting_utils = PatternReportingUtils()

    def detect(self, graph: nx.DiGraph, config: RepoConfig) -> list[DetectedPattern]:
        """Detect patterns based on configuration and built-in heuristics.

        This method:
        - Initializes ML confidence calculator if configured
        - Detects custom patterns from configuration
        - Detects built-in design patterns
        - Detects cloud-native architectural patterns
        - Applies ML-based confidence adjustments if enabled
        - Validates pattern structure and adjusts confidence accordingly

        Args:
            graph: The code graph to analyze for patterns
            config: Repository configuration with pattern detection settings

        Returns:
            List of detected patterns with their confidence scores and components
        """
        patterns = []

        # Initialize ML confidence calculator if configured
        self._validator_scorer.initialize_ml_calculator(config)

        # Detect custom patterns from configuration
        if config.custom_patterns:
            patterns.extend(self._recognizer.detect_custom_patterns(graph, config.custom_patterns))

        # Detect built-in design patterns
        patterns.extend(self._design_detector.detect_all_design_patterns(graph))

        # Detect cloud-native architectural patterns
        patterns.extend(self._architecture_analyzer.detect_all_architecture_patterns(graph))

        # Apply ML-based confidence adjustments if enabled
        if self._validator_scorer.is_ml_enabled():
            patterns = [
                self._validator_scorer.adjust_confidence_with_ml(pattern, graph)
                for pattern in patterns
            ]

        # Apply structural validation and confidence adjustment
        patterns = [
            self._validator_scorer.adjust_confidence_by_structure(pattern, graph)
            for pattern in patterns
        ]

        return patterns

    # Backwards compatibility methods - delegate to appropriate components

    def get_pattern_statistics(self, patterns: list[DetectedPattern]) -> dict[str, int]:
        """Get statistics about detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with pattern names as keys and their counts as values
        """
        return self._reporting_utils.get_pattern_statistics(patterns)

    def filter_patterns_by_confidence(
        self, patterns: list[DetectedPattern], min_confidence: float = 0.7
    ) -> list[DetectedPattern]:
        """Filter patterns by minimum confidence threshold.

        Args:
            patterns: List of detected patterns
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of patterns with confidence above the threshold
        """
        return self._reporting_utils.filter_patterns_by_confidence(patterns, min_confidence)

    # Additional utility methods for backwards compatibility and convenience

    def _detect_custom_patterns(
        self, graph: nx.DiGraph, custom_patterns: Sequence[dict]
    ) -> list[DetectedPattern]:
        """Detect patterns defined in configuration.

        Args:
            graph: The code graph to analyze for patterns
            custom_patterns: List of custom pattern configurations

        Returns:
            List of detected custom patterns
        """
        return self._recognizer.detect_custom_patterns(graph, custom_patterns)

    def _detect_builtin_patterns(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect common architectural patterns.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected built-in patterns
        """
        patterns = []

        # Detect design patterns
        patterns.extend(self._design_detector.detect_all_design_patterns(graph))

        # Detect architectural patterns
        patterns.extend(self._architecture_analyzer.detect_all_architecture_patterns(graph))

        return patterns

    # Legacy methods for backwards compatibility - redirect to new pattern detection

    def _detect_singleton_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Singleton pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_singleton_pattern(graph)

    def _detect_factory_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Factory pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_factory_pattern(graph)

    def _detect_observer_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Observer pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_observer_pattern(graph)

    def _detect_mvc_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect MVC pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_mvc_pattern(graph)

    def _detect_repository_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Repository pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_repository_pattern(graph)

    def _detect_service_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Service pattern - legacy method for backwards compatibility."""
        return self._design_detector._detect_service_pattern(graph)

    def _detect_sidecar_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Sidecar pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_sidecar_pattern(graph)

    def _detect_ambassador_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Ambassador pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_ambassador_pattern(graph)

    def _detect_adapter_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Adapter pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_adapter_pattern(graph)

    def _detect_circuit_breaker_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Circuit Breaker pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_circuit_breaker_pattern(graph)

    def _detect_service_mesh_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Service Mesh pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_service_mesh_pattern(graph)

    def _detect_api_gateway_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect API Gateway pattern - legacy method for backwards compatibility."""
        return self._architecture_analyzer._detect_api_gateway_pattern(graph)

    # Legacy utility methods for backwards compatibility

    def _get_compiled_pattern(self, pattern_str: str):
        """Get compiled pattern - legacy method for backwards compatibility."""
        return self._recognizer.get_compiled_pattern(pattern_str)

    def _node_matches_pattern(self, node_data: dict, pattern) -> bool:
        """Check node matches pattern - legacy method for backwards compatibility."""
        return self._recognizer.node_matches_pattern(node_data, pattern)

    def _get_node_code(self, node_data: dict) -> str:
        """Get node code - legacy method for backwards compatibility."""
        return self._recognizer.get_node_code(node_data)

    def _initialize_ml_calculator(self, config: RepoConfig) -> None:
        """Initialize ML calculator - legacy method for backwards compatibility."""
        self._validator_scorer.initialize_ml_calculator(config)

    def _adjust_confidence_with_ml(
        self, pattern: DetectedPattern, graph: nx.DiGraph
    ) -> DetectedPattern:
        """Adjust confidence with ML - legacy method for backwards compatibility."""
        return self._validator_scorer.adjust_confidence_with_ml(pattern, graph)

    # Helper methods for cloud patterns - delegate to architecture analyzer

    def _is_likely_helper(self, main_node_data: dict, helper_node_data: dict) -> bool:
        """Check if likely helper - legacy method for backwards compatibility."""
        return self._architecture_analyzer._is_likely_helper(main_node_data, helper_node_data)

    def _has_proxy_behavior(self, node_data: dict) -> bool:
        """Check proxy behavior - legacy method for backwards compatibility."""
        return self._architecture_analyzer._has_proxy_behavior(node_data)

    # Extended reporting and analysis methods

    def get_pattern_summary(self, patterns: list[DetectedPattern]) -> dict:
        """Get comprehensive pattern summary.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with comprehensive pattern analysis
        """
        return self._reporting_utils.get_patterns_summary(patterns)

    def generate_pattern_report(self, patterns: list[DetectedPattern]) -> str:
        """Generate formatted pattern report.

        Args:
            patterns: List of detected patterns

        Returns:
            Formatted text report
        """
        return self._reporting_utils.generate_pattern_report(patterns)

    def find_pattern_overlaps(self, patterns: list[DetectedPattern]) -> list[dict]:
        """Find overlapping patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            List of pattern overlap information
        """
        return self._reporting_utils.find_pattern_overlaps(patterns)

    def validate_pattern_structure(self, pattern: DetectedPattern, graph: nx.DiGraph) -> dict:
        """Validate pattern structure.

        Args:
            pattern: Pattern to validate
            graph: Code graph

        Returns:
            Validation results
        """
        return self._validator_scorer.validate_pattern_structure(pattern, graph)

    # Property accessors for component access

    @property
    def recognizer(self) -> CodePatternRecognizer:
        """Get the code pattern recognizer component."""
        return self._recognizer

    @property
    def design_detector(self) -> DesignPatternDetector:
        """Get the design pattern detector component."""
        return self._design_detector

    @property
    def architecture_analyzer(self) -> ArchitecturePatternAnalyzer:
        """Get the architecture pattern analyzer component."""
        return self._architecture_analyzer

    @property
    def validator_scorer(self) -> PatternValidationScorer:
        """Get the pattern validation scorer component."""
        return self._validator_scorer

    @property
    def reporting_utils(self) -> PatternReportingUtils:
        """Get the pattern reporting utilities component."""
        return self._reporting_utils
