"""Module for hotspot detection algorithms and core detection logic."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from models.architecture_models import HotspotPrediction

logger = logging.getLogger(__name__)


class HotspotDetector:
    """Core hotspot detection algorithms and scoring logic.

    This class implements the main algorithms for detecting code hotspots
    based on multiple metrics and feature weights. It handles the scoring,
    prediction, and classification of potential hotspots.
    """

    def __init__(self) -> None:
        """Initialize the hotspot detector with default feature weights."""
        # Feature weights for scoring
        self.feature_weights = {
            "complexity": 0.3,
            "change_frequency": 0.25,
            "bug_frequency": 0.2,
            "author_count": 0.1,
            "code_churn": 0.1,
            "recency": 0.05,
        }

    def calculate_hotspot_score(
        self,
        features: dict[str, float],
    ) -> float:
        """Calculate comprehensive hotspot score based on normalized features.

        Computes a weighted composite score by combining multiple normalized metrics
        that correlate with maintenance problems and technical debt. The calculation
        integrates both static code analysis and dynamic Git history patterns to
        provide a holistic risk assessment.

        Args:
            features (Dict[str, float]): Dictionary of normalized feature values
                between 0.0 and 1.0, representing different risk dimensions.

        Returns:
            float: Composite hotspot score between 0.0 and 1.0, where higher
                values indicate greater likelihood of being problematic.
        """
        # Calculate weighted score
        score = sum(
            features.get(feature, 0.0) * weight for feature, weight in self.feature_weights.items()
        )

        return min(1.0, max(0.0, score))

    def normalize_feature(self, value: float, max_expected: float) -> float:
        """Normalize a raw metric value to 0-1 range using sigmoid transformation.

        Applies a sigmoid function to transform raw metric values into a normalized
        score between 0.0 and 1.0. This approach provides smooth transitions,
        handles outliers gracefully, and prevents any single extreme value from
        dominating the analysis.

        Args:
            value (float): The raw metric value to normalize.
            max_expected (float): The expected maximum value for this metric type.

        Returns:
            float: Normalized value between 0.0 and 1.0.
        """
        # Use sigmoid function for smooth normalization
        return float(1.0 / (1.0 + np.exp(-4.0 * (value / max_expected - 0.5))))

    def predict_issue_types(self, features: dict[str, float]) -> list[str]:
        """Predict likely issue types based on normalized feature analysis.

        Analyzes the normalized feature values to predict specific types of issues
        that might be present in the file. Each prediction is based on individual
        feature thresholds that correlate with different maintenance problems.

        Args:
            features (Dict[str, float]): Dictionary of normalized feature values
                between 0.0 and 1.0.

        Returns:
            List[str]: List of predicted issue type identifiers.
        """
        issues = []

        if features.get("bug_frequency", 0) > 0.7:
            issues.append("bug-prone")

        if features.get("complexity", 0) > 0.7:
            issues.append("high-complexity")

        if features.get("change_frequency", 0) > 0.7:
            issues.append("unstable")

        if features.get("author_count", 0) > 0.7:
            issues.append("knowledge-siloed")

        if features.get("code_churn", 0) > 0.7:
            issues.append("high-churn")

        return issues

    def identify_risk_factors(self, features: dict[str, float]) -> list[str]:
        """Identify and quantify specific risk factors for human consumption.

        Generates human-readable descriptions of risk factors identified through
        feature analysis. Each risk factor includes a quantified severity level
        to help users understand the magnitude of the concern.

        Args:
            features (Dict[str, float]): Dictionary of normalized feature values
                between 0.0 and 1.0.

        Returns:
            List[str]: List of human-readable risk factor descriptions.
        """
        risk_factors = []

        if features.get("bug_frequency", 0) > 0.5:
            risk_factors.append(f"High bug frequency ({int(features['bug_frequency'] * 100)}%)")

        if features.get("complexity", 0) > 0.5:
            risk_factors.append(f"High complexity ({int(features['complexity'] * 100)}%)")

        if features.get("change_frequency", 0) > 0.5:
            risk_factors.append(f"Frequently changed ({int(features['change_frequency'] * 100)}%)")

        if features.get("author_count", 0) > 0.5:
            risk_factors.append(f"Many contributors ({int(features['author_count'] * 100)}%)")

        if features.get("code_churn", 0) > 0.5:
            risk_factors.append(f"High code churn ({int(features['code_churn'] * 100)}%)")

        return risk_factors

    def generate_recommendations(self, features: dict[str, float], score: float) -> list[str]:
        """Generate actionable recommendations based on hotspot analysis results.

        Creates a prioritized list of specific, actionable recommendations for
        addressing the issues identified in a hotspot file.

        Args:
            features (Dict[str, float]): Dictionary of normalized feature values.
            score (float): Overall composite hotspot score.

        Returns:
            List[str]: Ordered list of actionable recommendations.
        """
        recommendations = []

        if features.get("complexity", 0) > 0.7:
            recommendations.append("Refactor to reduce complexity and update maintainability")

        if features.get("bug_frequency", 0) > 0.7:
            recommendations.append("Add comprehensive unit tests and increase code review scrutiny")

        if features.get("change_frequency", 0) > 0.7 and features.get("author_count", 0) > 0.7:
            recommendations.append("Consider architectural updates to stabilize this component")

        if features.get("author_count", 0) > 0.7:
            recommendations.append("Document design decisions and share knowledge across team")

        if score > 0.8:
            recommendations.append(
                "Consider splitting this file into smaller, more focused modules"
            )

        if not recommendations:
            recommendations.append("Monitor this file for future degradation")

        return recommendations

    def create_hotspot_prediction(
        self,
        component_id: str,
        score: float,
        features: dict[str, float],
    ) -> HotspotPrediction:
        """Create a complete hotspot prediction object.

        Args:
            component_id (str): Unique identifier for the component.
            score (float): Hotspot score between 0.0 and 1.0.
            features (Dict[str, float]): Dictionary of normalized feature values.

        Returns:
            HotspotPrediction: Complete prediction object with all analysis results.
        """
        confidence = min(0.95, score)  # Cap confidence at 95%

        return HotspotPrediction(
            component_id=component_id,
            score=round(score, 3),
            confidence=round(confidence, 3),
            predicted_issues=self.predict_issue_types(features),
            risk_factors=self.identify_risk_factors(features),
            recommended_actions=self.generate_recommendations(features, score),
        )

    def is_hotspot(self, score: float, threshold: float = 0.5) -> bool:
        """Determine if a score qualifies as a hotspot.

        Args:
            score (float): Hotspot score to evaluate.
            threshold (float): Threshold above which to consider a hotspot.

        Returns:
            bool: True if the score exceeds the threshold.
        """
        return score > threshold

    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension.

        Args:
            file_path (str): Path to the file, including extension.

        Returns:
            str: Standardized language identifier or "unknown".
        """
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cs": "csharp",
        }
        return language_map.get(ext, "unknown")
