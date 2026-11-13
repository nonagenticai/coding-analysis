"""Tests for similarity_processing module — covers SimilarityProcessor,
process_similarity_request, batch_process_similarities, and helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.api.endpoints.similarity_modules.similarity_processing import (
    SimilarityProcessor,
    _calculate_confidence,
    _generate_basic_recommendations,
    batch_process_similarities,
    process_similarity_request,
)

# ---------------------------------------------------------------------------
# Helper: create fake SimilarityMatch
# ---------------------------------------------------------------------------


def _make_fake_match(**overrides):
    """Create a fake SimilarityMatch with sensible defaults."""
    defaults = dict(
        id="match-1",
        similarity_type=MagicMock(value="structural"),
        scope=MagicMock(value="function"),
        score=0.85,
        confidence=0.8,
        source_file="a.py",
        source_entity="func_a",
        source_language="python",
        source_lines=(1, 10),
        target_file="b.py",
        target_entity="func_b",
        target_language="python",
        target_lines=(5, 15),
        matching_features=["params", "returns"],
        differences=["naming"],
        structural_similarity=0.9,
        semantic_similarity=0.8,
        syntactic_similarity=0.7,
        is_cross_language=False,
        summary="Similar functions",
    )
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_fake_result(**overrides):
    """Create a fake SimilarityAnalysisResult."""
    match = _make_fake_match()
    defaults = dict(
        total_files_analyzed=5,
        total_entities_compared=20,
        total_matches_found=3,
        analysis_time_seconds=1.5,
        matches_by_type={MagicMock(value="structural"): [match]},
        matches_by_score={"high": [match]},
        cross_language_matches=[],
        language_distribution={"python": 5},
        average_similarity_score=0.85,
        highest_similarity_match=match,
        refactoring_opportunities=[],
        duplicate_code_ratio=0.1,
        cross_language_patterns=[],
    )
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ---------------------------------------------------------------------------
# SimilarityProcessor tests
# ---------------------------------------------------------------------------


class TestSimilarityProcessor:
    """Tests for SimilarityProcessor class."""

    def test_convert_match_to_model(self):
        match = _make_fake_match()
        model = SimilarityProcessor.convert_match_to_model(match)
        assert model.id == "match-1"
        assert model.score == 0.85
        assert model.source_file == "a.py"
        assert model.is_cross_language is False

    def test_convert_result_to_model(self):
        result = _make_fake_result()
        model = SimilarityProcessor.convert_result_to_model(result)
        assert model.total_files_analyzed == 5
        assert model.average_similarity_score == 0.85
        assert model.highest_similarity_match is not None

    def test_convert_result_no_highest_match(self):
        result = _make_fake_result(highest_similarity_match=None)
        model = SimilarityProcessor.convert_result_to_model(result)
        assert model.highest_similarity_match is None

    def test_process_clone_results_empty(self):
        result = SimilarityProcessor.process_clone_results([], ["a.py", "b.py"], 0.85)
        assert result["total_clones"] == 0
        assert result["statistics"]["duplicate_lines"] == 0

    def test_process_clone_results_with_clones(self):
        clone = _make_fake_match(score=0.96, confidence=0.9)
        result = SimilarityProcessor.process_clone_results([clone], ["a.py", "b.py"], 0.85)
        assert result["total_clones"] == 1
        assert result["clone_categories"]["exact_clones"] == 1

    def test_process_clone_results_near_clone(self):
        clone = _make_fake_match(score=0.88, confidence=0.75)
        result = SimilarityProcessor.process_clone_results([clone], ["a.py", "b.py"], 0.85)
        assert result["clone_categories"]["near_clones"] == 1
        assert result["clone_categories"]["exact_clones"] == 0

    def test_process_clone_results_structural_clone(self):
        clone = _make_fake_match(score=0.78)
        result = SimilarityProcessor.process_clone_results([clone], ["a.py"], 0.85)
        assert result["clone_categories"]["structural_clones"] == 1

    def test_process_cross_language_results_empty(self):
        result = SimilarityProcessor.process_cross_language_results([], ["a.py", "b.rs"], 0.6)
        assert result["total_patterns"] == 0
        assert result["languages_involved"] == 0

    def test_process_cross_language_results_with_patterns(self):
        pattern = _make_fake_match(
            source_language="python",
            target_language="rust",
            is_cross_language=True,
            score=0.75,
        )
        result = SimilarityProcessor.process_cross_language_results(
            [pattern], ["a.py", "b.rs"], 0.6
        )
        assert result["total_patterns"] == 1
        assert result["languages_involved"] == 2
        assert "python-rust" in result["language_pairs"]

    def test_clone_refactoring_recommendations_critical(self):
        clones = [_make_fake_match(id=f"c-{i}", score=0.97, confidence=0.85) for i in range(3)]
        recs = SimilarityProcessor._generate_clone_refactoring_recommendations(clones)
        priorities = [r["priority"] for r in recs]
        assert "critical" in priorities

    def test_clone_refactoring_recommendations_high(self):
        clones = [_make_fake_match(id=f"c-{i}", score=0.88, confidence=0.75) for i in range(2)]
        recs = SimilarityProcessor._generate_clone_refactoring_recommendations(clones)
        priorities = [r["priority"] for r in recs]
        assert "high" in priorities

    def test_clone_refactoring_recommendations_immediate_consolidation(self):
        clone = _make_fake_match(
            score=0.92, confidence=0.85, source_lines=(1, 20), target_lines=(1, 20)
        )
        recs = SimilarityProcessor._generate_clone_refactoring_recommendations([clone])
        types = [r["type"] for r in recs]
        assert "immediate_consolidation" in types

    def test_calculate_clone_statistics_empty(self):
        stats = SimilarityProcessor._calculate_clone_statistics([], ["a.py"])
        assert stats["duplicate_lines"] == 0
        assert stats["files_with_clones"] == 0
        assert stats["average_clone_size"] == 0

    def test_calculate_clone_statistics_with_clones(self):
        from app.api.endpoints.similarity_modules.similarity_processing import SimilarityMatchModel

        clone = SimilarityMatchModel(
            id="c-1",
            similarity_type="structural",
            scope="function",
            score=0.9,
            confidence=0.8,
            source_file="a.py",
            source_entity="func",
            source_language="python",
            source_lines=(1, 5),
            target_file="b.py",
            target_entity="func",
            target_language="python",
            target_lines=(10, 15),
            matching_features=["params"],
            differences=[],
            structural_similarity=0.9,
            semantic_similarity=0.8,
            syntactic_similarity=0.7,
            is_cross_language=False,
            summary="test",
        )
        stats = SimilarityProcessor._calculate_clone_statistics([clone], ["a.py", "b.py"])
        assert stats["duplicate_lines"] > 0
        assert stats["files_with_clones"] == 2
        assert stats["average_clone_size"] == 5
        assert stats["max_clone_size"] == 5
        assert stats["min_clone_size"] == 5

    def test_cross_language_insights_empty(self):
        insights = SimilarityProcessor._generate_cross_language_insights([], {})
        assert insights == {}

    def test_cross_language_insights_with_patterns(self):
        from app.api.endpoints.similarity_modules.similarity_processing import SimilarityMatchModel

        pattern = SimilarityMatchModel(
            id="p-1",
            similarity_type="semantic",
            scope="function",
            score=0.85,
            confidence=0.8,
            source_file="a.py",
            source_entity=None,
            source_language="python",
            source_lines=(1, 10),
            target_file="b.rs",
            target_entity=None,
            target_language="rust",
            target_lines=(1, 10),
            matching_features=["algorithm", "structure"],
            differences=[],
            structural_similarity=0.8,
            semantic_similarity=0.9,
            syntactic_similarity=0.5,
            is_cross_language=True,
            summary="Similar algorithm",
        )
        patterns = [pattern] * 4
        language_pairs = {"python-rust": patterns}
        insights = SimilarityProcessor._generate_cross_language_insights(patterns, language_pairs)
        assert "language_consistency" in insights
        assert "common_patterns" in insights
        assert "migration_readiness" in insights
        # 4 patterns with score 0.85 > 0.8 → migration ready
        assert len(insights["migration_readiness"]) > 0

    def test_migration_recommendations_with_high_sim(self):
        from app.api.endpoints.similarity_modules.similarity_processing import SimilarityMatchModel

        pattern = SimilarityMatchModel(
            id="p-1",
            similarity_type="semantic",
            scope="function",
            score=0.92,
            confidence=0.8,
            source_file="a.py",
            source_entity=None,
            source_language="python",
            source_lines=(1, 10),
            target_file="b.java",
            target_entity=None,
            target_language="java",
            target_lines=(1, 10),
            matching_features=["algorithm"],
            differences=[],
            structural_similarity=0.9,
            semantic_similarity=0.9,
            syntactic_similarity=0.5,
            is_cross_language=True,
            summary="High similarity",
        )
        patterns = [pattern] * 3
        language_pairs = {"java-python": patterns}
        recs = SimilarityProcessor._generate_migration_recommendations(patterns, language_pairs)
        assert any(r["type"] == "cross_language_standardization" for r in recs)
        assert any(r["type"] == "code_porting_opportunity" for r in recs)

    def test_migration_recommendations_empty(self):
        recs = SimilarityProcessor._generate_migration_recommendations([], {})
        assert recs == []

    def test_migration_recommendations_five_or_more_high_priority(self):
        from app.api.endpoints.similarity_modules.similarity_processing import SimilarityMatchModel

        pattern = SimilarityMatchModel(
            id="p-1",
            similarity_type="semantic",
            scope="function",
            score=0.92,
            confidence=0.8,
            source_file="a.py",
            source_entity=None,
            source_language="python",
            source_lines=(1, 10),
            target_file="b.java",
            target_entity=None,
            target_language="java",
            target_lines=(1, 10),
            matching_features=["algorithm"],
            differences=[],
            structural_similarity=0.9,
            semantic_similarity=0.9,
            syntactic_similarity=0.5,
            is_cross_language=True,
            summary="High similarity",
        )
        patterns = [pattern] * 6
        language_pairs = {"java-python": patterns}
        recs = SimilarityProcessor._generate_migration_recommendations(patterns, language_pairs)
        cross_recs = [r for r in recs if r["type"] == "cross_language_standardization"]
        assert cross_recs[0]["priority"] == "high"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for _calculate_confidence and _generate_basic_recommendations."""

    def test_calculate_confidence_all_consistent(self):
        results = {
            "structural": 0.8,
            "semantic": 0.8,
            "syntactic": 0.8,
            "exact": 0.8,
            "overall": 0.8,
        }
        c = _calculate_confidence(results)
        assert 0.0 <= c <= 1.0

    def test_calculate_confidence_all_zero(self):
        results = {
            "structural": 0.0,
            "semantic": 0.0,
            "syntactic": 0.0,
            "exact": 0.0,
            "overall": 0.0,
        }
        c = _calculate_confidence(results)
        assert c == pytest.approx(0.0)

    def test_calculate_confidence_inconsistent(self):
        results = {
            "structural": 1.0,
            "semantic": 0.0,
            "syntactic": 0.5,
            "exact": 0.5,
            "overall": 0.5,
        }
        c = _calculate_confidence(results)
        assert 0.0 <= c <= 1.0

    def test_recommendations_very_high(self):
        recs = _generate_basic_recommendations({"overall": 0.97}, "general")
        assert any("shared function" in r for r in recs)

    def test_recommendations_high(self):
        recs = _generate_basic_recommendations({"overall": 0.88}, "general")
        assert any("structural" in r.lower() for r in recs)

    def test_recommendations_moderate(self):
        recs = _generate_basic_recommendations({"overall": 0.72}, "general")
        assert any("design patterns" in r for r in recs)

    def test_recommendations_some(self):
        recs = _generate_basic_recommendations({"overall": 0.55}, "general")
        assert any("common requirements" in r for r in recs)

    def test_recommendations_low(self):
        recs = _generate_basic_recommendations({"overall": 0.2}, "general")
        assert any("distinct" in r for r in recs)

    def test_recommendations_refactoring_high(self):
        recs = _generate_basic_recommendations({"overall": 0.85}, "refactoring")
        assert any("refactoring potential" in r.lower() for r in recs)

    def test_recommendations_refactoring_low(self):
        recs = _generate_basic_recommendations({"overall": 0.5}, "refactoring")
        assert any("limited" in r.lower() for r in recs)

    def test_recommendations_clone_detection_high(self):
        recs = _generate_basic_recommendations({"overall": 0.92}, "clone_detection")
        assert any("investigate immediately" in r.lower() for r in recs)

    def test_recommendations_clone_detection_medium(self):
        recs = _generate_basic_recommendations({"overall": 0.75}, "clone_detection")
        assert any("manual review" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# process_similarity_request tests
# ---------------------------------------------------------------------------


class TestProcessSimilarityRequest:
    """Tests for process_similarity_request function."""

    def test_missing_code(self):
        result = process_similarity_request({"code_a": "", "code_b": "hello"})
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_missing_code_b(self):
        result = process_similarity_request({"code_a": "hello", "code_b": ""})
        assert result["success"] is False

    def test_success(self):
        result = process_similarity_request(
            {
                "code_a": "def foo(): pass",
                "code_b": "def bar(): pass",
                "language": "python",
            }
        )
        assert result["success"] is True
        assert "overall_score" in result
        assert "category" in result
        assert "recommendations" in result

    def test_custom_analysis_type(self):
        result = process_similarity_request(
            {"code_a": "x = 1", "code_b": "x = 1", "language": "python"},
            analysis_type="refactoring",
        )
        assert result["success"] is True
        assert result["analysis_type"] == "refactoring"

    def test_error_handling(self):
        with patch(
            "app.api.endpoints.similarity_modules.similarity_algorithms.calculate_similarity",
            side_effect=RuntimeError("boom"),
        ):
            result = process_similarity_request({"code_a": "a", "code_b": "b"})
            assert result["success"] is False
            assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# batch_process_similarities tests
# ---------------------------------------------------------------------------


class TestBatchProcessSimilarities:
    """Tests for batch_process_similarities function."""

    def test_empty_batch(self):
        result = batch_process_similarities([])
        assert result["batch_summary"]["total_requests"] == 0
        assert result["results"] == []

    def test_single_request(self):
        requests = [{"code_a": "def f(): pass", "code_b": "def g(): pass", "id": "r1"}]
        result = batch_process_similarities(requests)
        assert result["batch_summary"]["total_requests"] == 1
        assert result["batch_summary"]["successful"] >= 0
        assert len(result["results"]) == 1

    def test_multiple_requests(self):
        requests = [{"code_a": "a = 1", "code_b": "b = 2", "id": f"r{i}"} for i in range(5)]
        result = batch_process_similarities(requests)
        assert result["batch_summary"]["total_requests"] == 5
        assert len(result["results"]) == 5

    def test_mixed_success_failure(self):
        requests = [
            {"code_a": "a = 1", "code_b": "b = 2", "id": "ok"},
            {"code_a": "", "code_b": "", "id": "fail"},
        ]
        result = batch_process_similarities(requests)
        assert result["batch_summary"]["total_requests"] == 2
        assert result["batch_summary"]["failed"] >= 1

    def test_chunking_with_max_concurrent(self):
        requests = [{"code_a": "a", "code_b": "b", "id": f"r{i}"} for i in range(8)]
        result = batch_process_similarities(requests, max_concurrent=3)
        assert result["batch_summary"]["total_requests"] == 8
        assert len(result["results"]) == 8

    def test_error_in_batch(self):
        # Re-import to ensure we patch the same module instance that
        # batch_process_similarities uses (guards against sys.modules pollution).
        import importlib

        mod = importlib.import_module("app.api.endpoints.similarity_modules.similarity_processing")
        with patch.object(mod, "process_similarity_request", side_effect=RuntimeError("boom")):
            result = mod.batch_process_similarities([{"code_a": "a", "code_b": "b", "id": "r1"}])
            # Individual request errors are caught
            assert len(result["results"]) == 1
            assert result["results"][0]["success"] is False
