"""
Similarity Metrics and Scoring

Metrics calculation, scoring algorithms, and threshold management for similarity analysis.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class SimilarityMetrics:
    """Similarity metrics calculation and scoring utilities."""

    @staticmethod
    def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """
        Calculate cosine similarity between two feature vectors.

        Args:
            vector_a: First feature vector
            vector_b: Second feature vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        # Calculate magnitudes
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))

        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            set_a: First set of features
            set_b: Second set of features

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not set_a and not set_b:
            return 1.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def edit_distance_similarity(text_a: str, text_b: str) -> float:
        """
        Calculate similarity based on edit distance (Levenshtein distance).

        Args:
            text_a: First text string
            text_b: Second text string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text_a and not text_b:
            return 1.0

        max_len = max(len(text_a), len(text_b))
        if max_len == 0:
            return 1.0

        edit_dist = SimilarityMetrics._levenshtein_distance(text_a, text_b)
        return 1.0 - (edit_dist / max_len)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return SimilarityMetrics._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row: list[int] = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def weighted_similarity_score(
        structural: float,
        semantic: float,
        syntactic: float,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate weighted similarity score from multiple dimensions.

        Args:
            structural: Structural similarity score
            semantic: Semantic similarity score
            syntactic: Syntactic similarity score
            weights: Weight distribution for each dimension

        Returns:
            Weighted similarity score (0.0 to 1.0)
        """
        if weights is None:
            weights = {"structural": 0.4, "semantic": 0.4, "syntactic": 0.2}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        # Normalize weights
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        score = (
            structural * normalized_weights.get("structural", 0)
            + semantic * normalized_weights.get("semantic", 0)
            + syntactic * normalized_weights.get("syntactic", 0)
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def confidence_score(
        similarity_score: float, feature_count: int, code_length_ratio: float = 1.0
    ) -> float:
        """
        Calculate confidence score for a similarity match.

        Args:
            similarity_score: Base similarity score
            feature_count: Number of matching features
            code_length_ratio: Ratio of code lengths (smaller/larger)

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from similarity score
        base_confidence = similarity_score

        # Adjust for feature count (more features = higher confidence)
        feature_factor = min(1.0, feature_count / 10.0)

        # Adjust for code length similarity (similar lengths = higher confidence)
        length_factor = code_length_ratio

        # Calculate weighted confidence
        confidence = base_confidence * 0.6 + feature_factor * 0.25 + length_factor * 0.15

        return max(0.0, min(1.0, confidence))


class SimilarityThresholds:
    """Threshold management for different similarity types."""

    # Default thresholds for different similarity types
    DEFAULT_THRESHOLDS = {
        "exact_duplicate": 0.95,
        "near_duplicate": 0.85,
        "structural_similar": 0.75,
        "semantic_similar": 0.70,
        "syntactic_similar": 0.65,
        "cross_language": 0.60,
        "refactoring_candidate": 0.80,
        "code_clone": 0.85,
    }

    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    }

    @staticmethod
    def get_threshold(similarity_type: str) -> float:
        """Get threshold for a specific similarity type."""
        return SimilarityThresholds.DEFAULT_THRESHOLDS.get(similarity_type, 0.5)

    @staticmethod
    def categorize_by_score(score: float) -> str:
        """Categorize similarity based on score."""
        if score >= SimilarityThresholds.DEFAULT_THRESHOLDS["exact_duplicate"]:
            return "exact_duplicate"
        elif score >= SimilarityThresholds.DEFAULT_THRESHOLDS["near_duplicate"]:
            return "near_duplicate"
        elif score >= SimilarityThresholds.DEFAULT_THRESHOLDS["structural_similar"]:
            return "structural_similar"
        elif score >= SimilarityThresholds.DEFAULT_THRESHOLDS["semantic_similar"]:
            return "semantic_similar"
        elif score >= SimilarityThresholds.DEFAULT_THRESHOLDS["syntactic_similar"]:
            return "syntactic_similar"
        elif score >= SimilarityThresholds.DEFAULT_THRESHOLDS["cross_language"]:
            return "cross_language"
        else:
            return "low_similarity"

    @staticmethod
    def categorize_confidence(confidence: float) -> str:
        """Categorize confidence level."""
        if confidence >= SimilarityThresholds.CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif confidence >= SimilarityThresholds.CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        elif confidence >= SimilarityThresholds.CONFIDENCE_THRESHOLDS["low"]:
            return "low"
        else:
            return "very_low"

    @staticmethod
    def adjust_threshold_for_scope(base_threshold: float, scope: str) -> float:
        """Adjust threshold based on analysis scope."""
        adjustments = {
            "file_level": 0.05,  # More lenient for file-level
            "function_level": 0.0,  # Standard threshold
            "class_level": 0.02,  # Slightly more lenient
            "block_level": -0.05,  # More strict for block-level
        }

        adjustment = adjustments.get(scope, 0.0)
        return max(0.0, min(1.0, base_threshold + adjustment))

    @staticmethod
    def get_refactoring_priority(similarity_score: float, confidence: float, code_size: int) -> str:
        """Determine refactoring priority based on metrics."""
        if similarity_score >= 0.9 and confidence >= 0.8 and code_size > 10:
            return "critical"
        elif similarity_score >= 0.85 and confidence >= 0.7 and code_size > 5:
            return "high"
        elif similarity_score >= 0.8 and confidence >= 0.6:
            return "medium"
        elif similarity_score >= 0.7:
            return "low"
        else:
            return "none"


class SimilarityScoring:
    """Advanced scoring algorithms for similarity analysis."""

    @staticmethod
    def calculate_composite_score(
        metrics: dict[str, float],
        scope: str = "function_level",
        cross_language: bool = False,
    ) -> tuple[float, float]:
        """
        Calculate composite similarity score and confidence.

        Args:
            metrics: Dictionary of similarity metrics
            scope: Analysis scope
            cross_language: Whether this is cross-language comparison

        Returns:
            Tuple of (similarity_score, confidence_score)
        """
        structural = metrics.get("structural", 0.0)
        semantic = metrics.get("semantic", 0.0)
        syntactic = metrics.get("syntactic", 0.0)

        # Adjust weights based on scope and cross-language
        if cross_language:
            # For cross-language, prioritize structural and semantic
            weights = {"structural": 0.5, "semantic": 0.4, "syntactic": 0.1}
        else:
            # Standard weights
            weights = {"structural": 0.4, "semantic": 0.4, "syntactic": 0.2}

        # Calculate weighted score
        similarity_score = SimilarityMetrics.weighted_similarity_score(
            structural, semantic, syntactic, weights
        )

        # Calculate confidence
        feature_count = int(metrics.get("feature_count", 0))
        code_length_ratio = metrics.get("code_length_ratio", 1.0)

        confidence = SimilarityMetrics.confidence_score(
            similarity_score, feature_count, code_length_ratio
        )

        return similarity_score, confidence

    @staticmethod
    def score_code_clone(
        text_similarity: float,
        structure_similarity: float,
        identifier_similarity: float,
    ) -> float:
        """
        Score code clone similarity.

        Args:
            text_similarity: Raw text similarity
            structure_similarity: Structural similarity
            identifier_similarity: Identifier similarity

        Returns:
            Clone similarity score (0.0 to 1.0)
        """
        # Weight different aspects of code clones
        weights = {"text": 0.4, "structure": 0.4, "identifiers": 0.2}

        score = (
            text_similarity * weights["text"]
            + structure_similarity * weights["structure"]
            + identifier_similarity * weights["identifiers"]
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def score_cross_language_pattern(
        algorithmic_similarity: float,
        structural_similarity: float,
        api_similarity: float,
    ) -> float:
        """
        Score cross-language pattern similarity.

        Args:
            algorithmic_similarity: Algorithm similarity
            structural_similarity: Structural similarity
            api_similarity: API/interface similarity

        Returns:
            Cross-language pattern score (0.0 to 1.0)
        """
        # Weight aspects relevant to cross-language patterns
        weights = {"algorithmic": 0.5, "structural": 0.3, "api": 0.2}

        score = (
            algorithmic_similarity * weights["algorithmic"]
            + structural_similarity * weights["structural"]
            + api_similarity * weights["api"]
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def normalize_score_for_languages(score: float, source_lang: str, target_lang: str) -> float:
        """
        Normalize similarity score based on language pair characteristics.

        Args:
            score: Raw similarity score
            source_lang: Source language
            target_lang: Target language

        Returns:
            Normalized similarity score
        """
        # Language similarity matrix (how similar languages are)
        language_similarity = {
            ("java", "csharp"): 1.0,
            ("javascript", "typescript"): 0.95,
            ("c", "cpp"): 0.9,
            ("java", "kotlin"): 0.85,
            ("javascript", "java"): 0.7,
            ("python", "javascript"): 0.65,
            ("go", "rust"): 0.6,
            ("python", "java"): 0.6,
        }

        # Get language pair similarity (bidirectional)
        sorted_langs = sorted([source_lang.lower(), target_lang.lower()])
        lang_pair: tuple[str, str] = (sorted_langs[0], sorted_langs[1])
        lang_factor = language_similarity.get(lang_pair, 0.5)

        # Adjust score based on language similarity
        # More similar languages should have higher baseline similarity
        adjusted_score = score * (0.7 + 0.3 * lang_factor)

        return max(0.0, min(1.0, adjusted_score))

    @staticmethod
    def calculate_duplicate_code_ratio(
        matches: list[dict[str, Any]], total_code_lines: int, threshold: float = 0.85
    ) -> float:
        """
        Calculate ratio of duplicate code in the analyzed codebase.

        Args:
            matches: List of similarity matches
            total_code_lines: Total lines of code analyzed
            threshold: Threshold for considering code as duplicate

        Returns:
            Duplicate code ratio (0.0 to 1.0)
        """
        if total_code_lines == 0:
            return 0.0

        duplicate_lines = set()

        for match in matches:
            if match.get("score", 0) >= threshold:
                # Add source lines
                source_start, source_end = match.get("source_lines", (0, 0))
                for line in range(source_start, source_end + 1):
                    duplicate_lines.add(f"{match.get('source_file', '')}:{line}")

                # Add target lines
                target_start, target_end = match.get("target_lines", (0, 0))
                for line in range(target_start, target_end + 1):
                    duplicate_lines.add(f"{match.get('target_file', '')}:{line}")

        return len(duplicate_lines) / total_code_lines


class SimilarityMetric:
    """Enumeration of available similarity metrics."""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    EXACT = "exact"
    COSINE = "cosine"
    JACCARD = "jaccard"
    EDIT_DISTANCE = "edit_distance"

    @classmethod
    def get_all_metrics(cls) -> list[str]:
        """Get list of all available metrics."""
        return [
            cls.STRUCTURAL,
            cls.SEMANTIC,
            cls.SYNTACTIC,
            cls.EXACT,
            cls.COSINE,
            cls.JACCARD,
            cls.EDIT_DISTANCE,
        ]

    @classmethod
    def get_default_metrics(cls) -> list[str]:
        """Get list of default metrics for analysis."""
        return [cls.STRUCTURAL, cls.SEMANTIC, cls.SYNTACTIC, cls.EXACT]


def calculate_metrics(
    code_a: str,
    code_b: str,
    language: str = "python",
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Calculate multiple similarity metrics between two code segments.

    Args:
        code_a: First code segment
        code_b: Second code segment
        language: Programming language
        metrics: List of metrics to calculate (defaults to all)

    Returns:
        Dictionary of metric name to score mappings
    """
    if metrics is None:
        metrics = SimilarityMetric.get_default_metrics()

    results = {}

    try:
        from .similarity_algorithms import CodeFeatureExtractor, SimilarityAlgorithms

        # Extract features for different metrics
        if any(
            m
            in [
                SimilarityMetric.STRUCTURAL,
                SimilarityMetric.SEMANTIC,
                SimilarityMetric.SYNTACTIC,
            ]
            for m in metrics
        ):
            structural_features_a = CodeFeatureExtractor.extract_structural_features(
                code_a, language
            )
            structural_features_b = CodeFeatureExtractor.extract_structural_features(
                code_b, language
            )

            semantic_features_a = CodeFeatureExtractor.extract_semantic_features(code_a, language)
            semantic_features_b = CodeFeatureExtractor.extract_semantic_features(code_b, language)

            syntactic_features_a = CodeFeatureExtractor.extract_syntactic_features(code_a)
            syntactic_features_b = CodeFeatureExtractor.extract_syntactic_features(code_b)

        # Calculate requested metrics
        for metric in metrics:
            if metric == SimilarityMetric.STRUCTURAL:
                results[metric] = SimilarityAlgorithms.compare_structural_similarity(
                    structural_features_a, structural_features_b
                )
            elif metric == SimilarityMetric.SEMANTIC:
                results[metric] = SimilarityAlgorithms.compare_semantic_similarity(
                    semantic_features_a, semantic_features_b
                )
            elif metric == SimilarityMetric.SYNTACTIC:
                results[metric] = SimilarityAlgorithms.compare_syntactic_similarity(
                    syntactic_features_a, syntactic_features_b
                )
            elif metric == SimilarityMetric.EXACT:
                results[metric] = SimilarityAlgorithms.detect_exact_duplicates(code_a, code_b)
            elif metric == SimilarityMetric.EDIT_DISTANCE:
                results[metric] = SimilarityMetrics.edit_distance_similarity(code_a, code_b)
            elif metric == SimilarityMetric.COSINE:
                # Convert code to feature vectors for cosine similarity
                vector_a = _code_to_vector(code_a)
                vector_b = _code_to_vector(code_b)
                results[metric] = SimilarityMetrics.cosine_similarity(vector_a, vector_b)
            elif metric == SimilarityMetric.JACCARD:
                # Convert code to token sets for Jaccard similarity
                tokens_a = set(code_a.split())
                tokens_b = set(code_b.split())
                results[metric] = SimilarityMetrics.jaccard_similarity(tokens_a, tokens_b)
            else:
                logger.warning(f"Unknown metric: {metric}")
                results[metric] = 0.0

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Return zero scores for all requested metrics on error
        for metric in metrics:
            results[metric] = 0.0

    return results


def get_metric_weights(
    analysis_type: str = "general", language: str = "python"
) -> dict[str, float]:
    """
    Get metric weights for different analysis types.

    Args:
        analysis_type: Type of analysis (general, refactoring, clone_detection, etc.)
        language: Programming language

    Returns:
        Dictionary of metric weights
    """
    weight_profiles = {
        "general": {
            SimilarityMetric.STRUCTURAL: 0.3,
            SimilarityMetric.SEMANTIC: 0.3,
            SimilarityMetric.SYNTACTIC: 0.2,
            SimilarityMetric.EXACT: 0.2,
        },
        "refactoring": {
            SimilarityMetric.STRUCTURAL: 0.4,
            SimilarityMetric.SEMANTIC: 0.4,
            SimilarityMetric.SYNTACTIC: 0.1,
            SimilarityMetric.EXACT: 0.1,
        },
        "clone_detection": {
            SimilarityMetric.EXACT: 0.4,
            SimilarityMetric.STRUCTURAL: 0.3,
            SimilarityMetric.SEMANTIC: 0.2,
            SimilarityMetric.SYNTACTIC: 0.1,
        },
        "cross_language": {
            SimilarityMetric.SEMANTIC: 0.5,
            SimilarityMetric.STRUCTURAL: 0.3,
            SimilarityMetric.SYNTACTIC: 0.1,
            SimilarityMetric.EXACT: 0.1,
        },
    }

    # Language-specific adjustments
    language_adjustments = {
        "python": {},  # No adjustments for Python (baseline)
        "javascript": {SimilarityMetric.SYNTACTIC: 0.1},  # JS syntax more variable
        "java": {SimilarityMetric.STRUCTURAL: 0.05},  # Java more structured
    }

    # Get base weights
    base_weights = weight_profiles.get(analysis_type, weight_profiles["general"]).copy()

    # Apply language adjustments
    if language in language_adjustments:
        for metric, adjustment in language_adjustments[language].items():
            if metric in base_weights:
                base_weights[metric] += adjustment

    # Normalize weights to sum to 1.0
    total_weight = sum(base_weights.values())
    if total_weight > 0:
        for metric in base_weights:
            base_weights[metric] /= total_weight

    return base_weights


def _code_to_vector(code: str) -> list[float]:
    """Convert code to a simple feature vector for cosine similarity."""
    # Simple feature extraction - count character frequencies
    features = [0.0] * 128  # ASCII character range

    for char in code:
        if ord(char) < 128:
            features[ord(char)] += 1.0

    # Normalize by code length
    code_len = len(code)
    if code_len > 0:
        features = [f / code_len for f in features]

    return features
