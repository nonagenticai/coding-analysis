"""Pattern validation and scoring module for ML-based confidence adjustment.

This module provides the PatternValidationScorer class which handles pattern validation
and ML-based confidence scoring for detected patterns.
"""

from __future__ import annotations

import logging

import networkx as nx

from models.architecture_models import DetectedPattern, RepoConfig
from processing.architecture.ml_confidence_calculator import (
    MLConfidenceCalculator,
)

logger = logging.getLogger(__name__)


class PatternValidationScorer:
    """Handle pattern validation and ML-based confidence scoring."""

    def __init__(self) -> None:
        """Initialize the pattern validation scorer."""
        self._ml_calculator: MLConfidenceCalculator | None = None
        self._ml_enabled = False

    def initialize_ml_calculator(self, config: RepoConfig) -> None:
        """Initialize ML confidence calculator if enabled in config.

        Args:
            config: Repository configuration with ML settings

        Raises:
            ImportError: If required ML dependencies are missing
            ValueError: If ML configuration is invalid
        """
        if not config:
            return

        # Check if ML confidence settings are present and enabled
        ml_settings = getattr(config, "ml_confidence_settings", None)
        if ml_settings and ml_settings.get("enabled", False):
            try:
                self._ml_calculator = MLConfidenceCalculator(config)
                self._ml_enabled = True
                logger.info("ML-based confidence scoring enabled")
            except (ImportError, TypeError, ValueError, AttributeError) as e:
                logger.error("Failed to initialize ML confidence calculator: %s", e)
                self._ml_enabled = False

    def adjust_confidence_with_ml(
        self, pattern: DetectedPattern, graph: nx.DiGraph
    ) -> DetectedPattern:
        """Adjust pattern confidence using ML-based scoring.

        Args:
            pattern: The detected pattern to adjust confidence for
            graph: The code graph containing the pattern components

        Returns:
            A new DetectedPattern with adjusted confidence score

        Note:
            If ML calculation fails, returns the original pattern unchanged
        """
        if not self._ml_enabled or not self._ml_calculator:
            return pattern

        try:
            # Calculate ML-based confidence
            ml_confidence = self._ml_calculator.calculate_confidence(pattern, graph)

            # Create a new pattern with adjusted confidence
            return DetectedPattern(
                pattern_type=pattern.pattern_type,
                confidence=ml_confidence,
                components=pattern.components,
                description=pattern.description,
                name=pattern.name,
                location=pattern.location,
            )
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("ML confidence adjustment failed for %s: %s", pattern.pattern_type, e)
            return pattern

    def validate_pattern_components(self, pattern: DetectedPattern, graph: nx.DiGraph) -> bool:
        """Validate that pattern components exist in the graph.

        Args:
            pattern: The pattern to validate
            graph: The code graph to check against

        Returns:
            True if all components exist in the graph, False otherwise
        """
        if not pattern.components:
            logger.warning("Pattern %s has no components", pattern.pattern_type)
            return False

        for component in pattern.components:
            if component not in graph.nodes:
                logger.warning(
                    "Pattern %s component %s not found in graph",
                    pattern.pattern_type,
                    component,
                )
                return False

        return True

    def calculate_pattern_cohesion(self, pattern: DetectedPattern, graph: nx.DiGraph) -> float:
        """Calculate how cohesive the pattern components are.

        Args:
            pattern: The pattern to analyze
            graph: The code graph containing the components

        Returns:
            Cohesion score between 0.0 and 1.0 (higher = more cohesive)
        """
        if not pattern.components or len(pattern.components) < 2:
            return 1.0  # Single component is perfectly cohesive

        # Count connections between pattern components
        total_possible_connections = len(pattern.components) * (len(pattern.components) - 1)
        actual_connections = 0

        for i, component1 in enumerate(pattern.components):
            for j, component2 in enumerate(pattern.components):
                if i != j and graph.has_edge(component1, component2):
                    actual_connections += 1

        if total_possible_connections == 0:
            return 1.0

        return actual_connections / total_possible_connections

    def calculate_pattern_coupling(self, pattern: DetectedPattern, graph: nx.DiGraph) -> float:
        """Calculate how coupled the pattern is to external components.

        Args:
            pattern: The pattern to analyze
            graph: The code graph containing the components

        Returns:
            Coupling score between 0.0 and 1.0 (higher = more coupled)
        """
        if not pattern.components:
            return 0.0

        external_connections = 0
        total_connections = 0

        for component in pattern.components:
            # Count all edges from this component
            out_edges = list(graph.out_edges(component))
            in_edges = list(graph.in_edges(component))

            total_connections += len(out_edges) + len(in_edges)

            # Count external connections (to nodes not in the pattern)
            for _, target in out_edges:
                if target not in pattern.components:
                    external_connections += 1

            for source, _ in in_edges:
                if source not in pattern.components:
                    external_connections += 1

        if total_connections == 0:
            return 0.0

        return external_connections / total_connections

    def score_pattern_quality(self, pattern: DetectedPattern, graph: nx.DiGraph) -> float:
        """Calculate an overall quality score for a pattern.

        Args:
            pattern: The pattern to score
            graph: The code graph containing the components

        Returns:
            Quality score between 0.0 and 1.0 (higher = better quality)
        """
        # Base score is the pattern's confidence
        quality_score = pattern.confidence

        # Adjust based on cohesion (higher cohesion is better)
        cohesion = self.calculate_pattern_cohesion(pattern, graph)
        quality_score *= 0.7 + 0.3 * cohesion  # Weight cohesion positively

        # Adjust based on coupling (lower coupling is better for most patterns)
        coupling = self.calculate_pattern_coupling(pattern, graph)
        quality_score *= 1.0 - 0.2 * coupling  # Penalize high coupling

        # Ensure score stays within bounds
        return max(0.0, min(1.0, quality_score))

    def validate_pattern_structure(self, pattern: DetectedPattern, graph: nx.DiGraph) -> dict:
        """Perform comprehensive validation of pattern structure.

        Args:
            pattern: The pattern to validate
            graph: The code graph containing the components

        Returns:
            Dictionary with validation results and metrics
        """
        results = {
            "is_valid": False,
            "components_exist": False,
            "cohesion_score": 0.0,
            "coupling_score": 0.0,
            "quality_score": 0.0,
            "validation_errors": [],
        }

        # Check if components exist
        results["components_exist"] = self.validate_pattern_components(pattern, graph)
        if not results["components_exist"]:
            results["validation_errors"].append("Some pattern components not found in graph")
            return results

        # Calculate metrics
        results["cohesion_score"] = self.calculate_pattern_cohesion(pattern, graph)
        results["coupling_score"] = self.calculate_pattern_coupling(pattern, graph)
        results["quality_score"] = self.score_pattern_quality(pattern, graph)

        # Determine if pattern is valid based on thresholds
        min_quality_threshold = 0.3
        results["is_valid"] = results["quality_score"] >= min_quality_threshold

        if not results["is_valid"]:
            results["validation_errors"].append(
                f"Pattern quality score {results['quality_score']:.2f} below threshold {min_quality_threshold}"
            )

        return results

    def adjust_confidence_by_structure(
        self, pattern: DetectedPattern, graph: nx.DiGraph
    ) -> DetectedPattern:
        """Adjust pattern confidence based on structural analysis.

        Args:
            pattern: The pattern to adjust
            graph: The code graph containing the components

        Returns:
            Pattern with adjusted confidence based on structural metrics
        """
        validation_results = self.validate_pattern_structure(pattern, graph)

        if not validation_results["is_valid"]:
            # Reduce confidence for invalid patterns
            adjusted_confidence = pattern.confidence * 0.5
        else:
            # Adjust confidence based on quality score
            quality_factor = validation_results["quality_score"]
            adjusted_confidence = pattern.confidence * (0.7 + 0.3 * quality_factor)

        # Ensure confidence stays within bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        return DetectedPattern(
            pattern_type=pattern.pattern_type,
            confidence=adjusted_confidence,
            components=pattern.components,
            description=pattern.description,
            name=pattern.name,
            location=pattern.location,
        )

    def is_ml_enabled(self) -> bool:
        """Check if ML-based confidence scoring is enabled.

        Returns:
            True if ML scoring is enabled, False otherwise
        """
        return self._ml_enabled

    def get_ml_calculator(self) -> MLConfidenceCalculator | None:
        """Get the ML confidence calculator instance.

        Returns:
            ML calculator instance if available, None otherwise
        """
        return self._ml_calculator
