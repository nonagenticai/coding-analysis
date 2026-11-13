"""Tests for similarity_metrics module — covers SimilarityMetrics, SimilarityThresholds,
SimilarityScoring, SimilarityMetric, calculate_metrics, get_metric_weights, _code_to_vector."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from app.api.endpoints.similarity_modules.similarity_metrics import (
    SimilarityMetric,
    SimilarityMetrics,
    SimilarityScoring,
    SimilarityThresholds,
    _code_to_vector,
    calculate_metrics,
    get_metric_weights,
)

# ---------------------------------------------------------------------------
# SimilarityMetrics tests
# ---------------------------------------------------------------------------


class TestSimilarityMetrics:
    """Tests for SimilarityMetrics class."""

    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert SimilarityMetrics.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert SimilarityMetrics.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_empty_vectors(self):
        assert SimilarityMetrics.cosine_similarity([], []) == 0.0

    def test_cosine_similarity_different_lengths(self):
        assert SimilarityMetrics.cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_similarity_zero_vectors(self):
        assert SimilarityMetrics.cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_cosine_similarity_one_zero_vector(self):
        assert SimilarityMetrics.cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_cosine_similarity_partial(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert SimilarityMetrics.cosine_similarity(a, b) == pytest.approx(expected)

    def test_jaccard_similarity_identical_sets(self):
        s = {"a", "b", "c"}
        assert SimilarityMetrics.jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_jaccard_similarity_disjoint_sets(self):
        assert SimilarityMetrics.jaccard_similarity({"a"}, {"b"}) == pytest.approx(0.0)

    def test_jaccard_similarity_both_empty(self):
        assert SimilarityMetrics.jaccard_similarity(set(), set()) == 1.0

    def test_jaccard_similarity_partial_overlap(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        assert SimilarityMetrics.jaccard_similarity(a, b) == pytest.approx(2 / 4)

    def test_edit_distance_similarity_identical(self):
        assert SimilarityMetrics.edit_distance_similarity("abc", "abc") == pytest.approx(1.0)

    def test_edit_distance_similarity_totally_different(self):
        assert SimilarityMetrics.edit_distance_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_edit_distance_similarity_one_edit(self):
        # "abc" -> "aXc": 1 edit out of 3 max len => 1 - 1/3 = 2/3
        assert SimilarityMetrics.edit_distance_similarity("abc", "aXc") == pytest.approx(2 / 3)

    def test_edit_distance_similarity_both_empty(self):
        assert SimilarityMetrics.edit_distance_similarity("", "") == 1.0

    def test_edit_distance_similarity_one_empty(self):
        assert SimilarityMetrics.edit_distance_similarity("abc", "") == pytest.approx(0.0)

    def test_levenshtein_distance_same(self):
        assert SimilarityMetrics._levenshtein_distance("kitten", "kitten") == 0

    def test_levenshtein_distance_classic(self):
        assert SimilarityMetrics._levenshtein_distance("kitten", "sitting") == 3

    def test_levenshtein_distance_empty_second(self):
        assert SimilarityMetrics._levenshtein_distance("abc", "") == 3

    def test_levenshtein_distance_s1_shorter_than_s2(self):
        # Triggers the recursive swap branch
        assert SimilarityMetrics._levenshtein_distance("a", "abc") == 2

    def test_weighted_similarity_default_weights(self):
        score = SimilarityMetrics.weighted_similarity_score(0.8, 0.6, 0.4)
        expected = 0.8 * 0.4 + 0.6 * 0.4 + 0.4 * 0.2
        assert score == pytest.approx(expected)

    def test_weighted_similarity_custom_weights(self):
        score = SimilarityMetrics.weighted_similarity_score(
            1.0, 0.0, 0.0, {"structural": 1.0, "semantic": 0.0, "syntactic": 0.0}
        )
        assert score == pytest.approx(1.0)

    def test_weighted_similarity_zero_total_weight(self):
        score = SimilarityMetrics.weighted_similarity_score(
            0.5, 0.5, 0.5, {"structural": 0.0, "semantic": 0.0, "syntactic": 0.0}
        )
        assert score == 0.0

    def test_weighted_similarity_clamped(self):
        score = SimilarityMetrics.weighted_similarity_score(1.5, 1.5, 1.5)
        assert score <= 1.0

    def test_confidence_score_basic(self):
        c = SimilarityMetrics.confidence_score(0.9, 10, 0.95)
        assert 0.0 <= c <= 1.0

    def test_confidence_score_zero_features(self):
        c = SimilarityMetrics.confidence_score(0.5, 0, 1.0)
        assert 0.0 <= c <= 1.0

    def test_confidence_score_many_features(self):
        c = SimilarityMetrics.confidence_score(0.9, 20, 1.0)
        assert c > 0.5


# ---------------------------------------------------------------------------
# SimilarityThresholds tests
# ---------------------------------------------------------------------------


class TestSimilarityThresholds:
    """Tests for SimilarityThresholds class."""

    def test_get_threshold_known(self):
        assert SimilarityThresholds.get_threshold("exact_duplicate") == 0.95

    def test_get_threshold_unknown_returns_default(self):
        assert SimilarityThresholds.get_threshold("nonexistent") == 0.5

    def test_categorize_exact_duplicate(self):
        assert SimilarityThresholds.categorize_by_score(0.96) == "exact_duplicate"

    def test_categorize_near_duplicate(self):
        assert SimilarityThresholds.categorize_by_score(0.88) == "near_duplicate"

    def test_categorize_structural_similar(self):
        assert SimilarityThresholds.categorize_by_score(0.77) == "structural_similar"

    def test_categorize_semantic_similar(self):
        assert SimilarityThresholds.categorize_by_score(0.72) == "semantic_similar"

    def test_categorize_syntactic_similar(self):
        assert SimilarityThresholds.categorize_by_score(0.66) == "syntactic_similar"

    def test_categorize_cross_language(self):
        assert SimilarityThresholds.categorize_by_score(0.61) == "cross_language"

    def test_categorize_low_similarity(self):
        assert SimilarityThresholds.categorize_by_score(0.3) == "low_similarity"

    def test_categorize_confidence_high(self):
        assert SimilarityThresholds.categorize_confidence(0.9) == "high"

    def test_categorize_confidence_medium(self):
        assert SimilarityThresholds.categorize_confidence(0.65) == "medium"

    def test_categorize_confidence_low(self):
        assert SimilarityThresholds.categorize_confidence(0.45) == "low"

    def test_categorize_confidence_very_low(self):
        assert SimilarityThresholds.categorize_confidence(0.2) == "very_low"

    def test_adjust_threshold_file_level(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.8, "file_level")
        assert result == pytest.approx(0.85)

    def test_adjust_threshold_function_level(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.8, "function_level")
        assert result == pytest.approx(0.8)

    def test_adjust_threshold_class_level(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.8, "class_level")
        assert result == pytest.approx(0.82)

    def test_adjust_threshold_block_level(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.8, "block_level")
        assert result == pytest.approx(0.75)

    def test_adjust_threshold_unknown_scope(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.8, "unknown")
        assert result == pytest.approx(0.8)

    def test_adjust_threshold_clamps_to_zero(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.02, "block_level")
        assert result == pytest.approx(0.0)

    def test_adjust_threshold_clamps_to_one(self):
        result = SimilarityThresholds.adjust_threshold_for_scope(0.98, "file_level")
        assert result == pytest.approx(1.0)

    def test_refactoring_priority_critical(self):
        assert SimilarityThresholds.get_refactoring_priority(0.95, 0.85, 20) == "critical"

    def test_refactoring_priority_high(self):
        assert SimilarityThresholds.get_refactoring_priority(0.88, 0.75, 8) == "high"

    def test_refactoring_priority_medium(self):
        assert SimilarityThresholds.get_refactoring_priority(0.82, 0.65, 3) == "medium"

    def test_refactoring_priority_low(self):
        assert SimilarityThresholds.get_refactoring_priority(0.72, 0.5, 2) == "low"

    def test_refactoring_priority_none(self):
        assert SimilarityThresholds.get_refactoring_priority(0.5, 0.3, 1) == "none"


# ---------------------------------------------------------------------------
# SimilarityScoring tests
# ---------------------------------------------------------------------------


class TestSimilarityScoring:
    """Tests for SimilarityScoring class."""

    def test_composite_score_standard(self):
        metrics = {
            "structural": 0.8,
            "semantic": 0.7,
            "syntactic": 0.6,
            "feature_count": 5,
            "code_length_ratio": 0.9,
        }
        score, confidence = SimilarityScoring.calculate_composite_score(metrics)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_composite_score_cross_language(self):
        metrics = {"structural": 0.8, "semantic": 0.7, "syntactic": 0.6, "feature_count": 5}
        score_xl, _ = SimilarityScoring.calculate_composite_score(metrics, cross_language=True)
        score_std, _ = SimilarityScoring.calculate_composite_score(metrics, cross_language=False)
        # Cross-language weights structural+semantic more, syntactic less
        assert score_xl != score_std

    def test_composite_score_zero_metrics(self):
        metrics = {"structural": 0.0, "semantic": 0.0, "syntactic": 0.0}
        score, confidence = SimilarityScoring.calculate_composite_score(metrics)
        assert score == pytest.approx(0.0)

    def test_score_code_clone(self):
        score = SimilarityScoring.score_code_clone(0.9, 0.8, 0.7)
        expected = 0.9 * 0.4 + 0.8 * 0.4 + 0.7 * 0.2
        assert score == pytest.approx(expected)

    def test_score_code_clone_clamped(self):
        assert SimilarityScoring.score_code_clone(1.5, 1.5, 1.5) == 1.0
        assert SimilarityScoring.score_code_clone(-1.0, -1.0, -1.0) == 0.0

    def test_score_cross_language_pattern(self):
        score = SimilarityScoring.score_cross_language_pattern(0.9, 0.7, 0.5)
        expected = 0.9 * 0.5 + 0.7 * 0.3 + 0.5 * 0.2
        assert score == pytest.approx(expected)

    def test_normalize_score_known_pair(self):
        # sorted("c","cpp") = ("c","cpp") which IS in the table with factor 0.9
        score = SimilarityScoring.normalize_score_for_languages(0.8, "c", "cpp")
        expected = 0.8 * (0.7 + 0.3 * 0.9)
        assert score == pytest.approx(expected)

    def test_normalize_score_reversed_pair(self):
        # Should be bidirectional (sorted key)
        s1 = SimilarityScoring.normalize_score_for_languages(0.8, "c", "cpp")
        s2 = SimilarityScoring.normalize_score_for_languages(0.8, "cpp", "c")
        assert s1 == pytest.approx(s2)

    def test_normalize_score_unknown_pair(self):
        score = SimilarityScoring.normalize_score_for_languages(0.8, "cobol", "fortran")
        # lang_factor=0.5, adjusted = 0.8 * (0.7 + 0.3*0.5) = 0.8*0.85 = 0.68
        assert score == pytest.approx(0.68)

    def test_normalize_score_js_ts(self):
        score = SimilarityScoring.normalize_score_for_languages(0.8, "javascript", "typescript")
        expected = 0.8 * (0.7 + 0.3 * 0.95)
        assert score == pytest.approx(expected)

    def test_normalize_score_java_csharp_sorted_mismatch(self):
        # NOTE: table has ("java","csharp") but sorted is ("csharp","java") — falls back to 0.5
        score = SimilarityScoring.normalize_score_for_languages(1.0, "java", "csharp")
        expected = 1.0 * (0.7 + 0.3 * 0.5)  # 0.85 (default factor)
        assert score == pytest.approx(expected)

    def test_normalize_score_python_java(self):
        # sorted("java","python") = ("java","python") which IS NOT in table
        # table has ("python","java") but sorted would be ("java","python")
        score = SimilarityScoring.normalize_score_for_languages(1.0, "python", "java")
        expected = 1.0 * (0.7 + 0.3 * 0.5)  # falls back to 0.5
        assert score == pytest.approx(expected)

    def test_duplicate_code_ratio_empty(self):
        assert SimilarityScoring.calculate_duplicate_code_ratio([], 100) == 0.0

    def test_duplicate_code_ratio_zero_total(self):
        assert SimilarityScoring.calculate_duplicate_code_ratio([], 0) == 0.0

    def test_duplicate_code_ratio_with_matches(self):
        matches = [
            {
                "score": 0.9,
                "source_file": "a.py",
                "source_lines": (1, 5),
                "target_file": "b.py",
                "target_lines": (1, 5),
            },
        ]
        ratio = SimilarityScoring.calculate_duplicate_code_ratio(matches, 100)
        # 5 source lines + 5 target lines = 10 unique lines
        assert ratio == pytest.approx(10 / 100)

    def test_duplicate_code_ratio_below_threshold(self):
        matches = [
            {
                "score": 0.5,
                "source_file": "a.py",
                "source_lines": (1, 5),
                "target_file": "b.py",
                "target_lines": (1, 5),
            },
        ]
        ratio = SimilarityScoring.calculate_duplicate_code_ratio(matches, 100)
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# SimilarityMetric tests
# ---------------------------------------------------------------------------


class TestSimilarityMetric:
    """Tests for SimilarityMetric class."""

    def test_get_all_metrics(self):
        metrics = SimilarityMetric.get_all_metrics()
        assert "structural" in metrics
        assert "semantic" in metrics
        assert "edit_distance" in metrics
        assert len(metrics) == 7

    def test_get_default_metrics(self):
        defaults = SimilarityMetric.get_default_metrics()
        assert len(defaults) == 4
        assert "structural" in defaults
        assert "exact" in defaults

    def test_metric_constants(self):
        assert SimilarityMetric.STRUCTURAL == "structural"
        assert SimilarityMetric.COSINE == "cosine"
        assert SimilarityMetric.JACCARD == "jaccard"


# ---------------------------------------------------------------------------
# calculate_metrics tests
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_default_metrics(self):
        results = calculate_metrics("def foo(): pass", "def bar(): pass")
        assert "structural" in results
        assert "semantic" in results
        assert "syntactic" in results
        assert "exact" in results

    def test_specific_metrics(self):
        results = calculate_metrics("hello", "hello", metrics=["edit_distance", "jaccard"])
        assert "edit_distance" in results
        assert "jaccard" in results
        assert results["edit_distance"] == pytest.approx(1.0)
        assert results["jaccard"] == pytest.approx(1.0)

    def test_cosine_metric(self):
        results = calculate_metrics("abc", "abc", metrics=["cosine"])
        assert "cosine" in results
        assert results["cosine"] == pytest.approx(1.0)

    def test_unknown_metric(self):
        results = calculate_metrics("abc", "xyz", metrics=["nonexistent"])
        assert results["nonexistent"] == 0.0

    def test_error_returns_zeros(self):
        # Force an error via mocking
        with patch(
            "app.api.endpoints.similarity_modules.similarity_metrics.SimilarityMetrics.edit_distance_similarity",
            side_effect=RuntimeError("boom"),
        ):
            results = calculate_metrics("a", "b", metrics=["edit_distance"])
            assert results["edit_distance"] == 0.0


# ---------------------------------------------------------------------------
# get_metric_weights tests
# ---------------------------------------------------------------------------


class TestGetMetricWeights:
    """Tests for get_metric_weights function."""

    def test_general_weights(self):
        weights = get_metric_weights("general")
        assert sum(weights.values()) == pytest.approx(1.0)
        assert "structural" in weights

    def test_refactoring_weights(self):
        weights = get_metric_weights("refactoring")
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_clone_detection_weights(self):
        weights = get_metric_weights("clone_detection")
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_cross_language_weights(self):
        weights = get_metric_weights("cross_language")
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_unknown_type_uses_general(self):
        weights = get_metric_weights("imaginary_type")
        general = get_metric_weights("general")
        assert weights == general

    def test_javascript_adjustment(self):
        w_python = get_metric_weights("general", "python")
        w_js = get_metric_weights("general", "javascript")
        # JS adjusts syntactic weight
        assert w_python != w_js

    def test_java_adjustment(self):
        w_python = get_metric_weights("general", "python")
        w_java = get_metric_weights("general", "java")
        assert w_python != w_java


# ---------------------------------------------------------------------------
# _code_to_vector tests
# ---------------------------------------------------------------------------


class TestCodeToVector:
    """Tests for _code_to_vector helper."""

    def test_empty_code(self):
        v = _code_to_vector("")
        assert len(v) == 128
        assert all(x == 0.0 for x in v)

    def test_single_character(self):
        v = _code_to_vector("a")
        assert v[ord("a")] == pytest.approx(1.0)

    def test_repeated_character(self):
        v = _code_to_vector("aaa")
        assert v[ord("a")] == pytest.approx(1.0)  # 3/3

    def test_non_ascii_ignored(self):
        v = _code_to_vector("\x80\xff")
        assert all(x == 0.0 for x in v)

    def test_vector_length(self):
        v = _code_to_vector("def foo(): pass")
        assert len(v) == 128
