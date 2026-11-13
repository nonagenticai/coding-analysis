"""
Similarity Result Processing

Result processing, data conversion, and response formatting for similarity analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from services.similarity_detector import (
    SimilarityAnalysisResult,
    SimilarityMatch,
)

from ...models.common_api_models import BaseResponse, ResponseStatus
from .similarity_metrics import SimilarityThresholds

logger = logging.getLogger(__name__)


class SimilarityMatchModel(BaseModel):
    """Similarity match response model."""

    id: str
    similarity_type: str
    scope: str
    score: float
    confidence: float

    source_file: str
    source_entity: str | None = None
    source_language: str
    source_lines: tuple[int, int]

    target_file: str
    target_entity: str | None = None
    target_language: str
    target_lines: tuple[int, int]

    matching_features: list[str]
    differences: list[str]

    structural_similarity: float
    semantic_similarity: float
    syntactic_similarity: float

    is_cross_language: bool
    summary: str


class SimilarityAnalysisResponseModel(BaseModel):
    """Similarity analysis result response model."""

    total_files_analyzed: int
    total_entities_compared: int
    total_matches_found: int
    analysis_time_seconds: float

    matches_by_type: dict[str, list[SimilarityMatchModel]]
    matches_by_score: dict[str, list[SimilarityMatchModel]]
    cross_language_matches: list[SimilarityMatchModel]

    language_distribution: dict[str, int]
    average_similarity_score: float
    highest_similarity_match: SimilarityMatchModel | None = None

    refactoring_opportunities: list[dict[str, Any]]
    duplicate_code_ratio: float
    cross_language_patterns: list[dict[str, Any]]


class SimilarityAnalysisResponse(BaseResponse):
    """Response from similarity analysis operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: SimilarityAnalysisResponseModel = Field(..., description="Similarity analysis result")


class SimilarityProcessor:
    """Process and format similarity analysis results."""

    @staticmethod
    def convert_match_to_model(match: SimilarityMatch) -> SimilarityMatchModel:
        """Convert SimilarityMatch to response model."""
        return SimilarityMatchModel(
            id=match.id,
            similarity_type=match.similarity_type.value,
            scope=match.scope.value,
            score=match.score,
            confidence=match.confidence,
            source_file=match.source_file,
            source_entity=match.source_entity,
            source_language=match.source_language,
            source_lines=match.source_lines,
            target_file=match.target_file,
            target_entity=match.target_entity,
            target_language=match.target_language,
            target_lines=match.target_lines,
            matching_features=match.matching_features,
            differences=match.differences,
            structural_similarity=match.structural_similarity,
            semantic_similarity=match.semantic_similarity,
            syntactic_similarity=match.syntactic_similarity,
            is_cross_language=match.is_cross_language,
            summary=match.summary,
        )

    @staticmethod
    def convert_result_to_model(
        result: SimilarityAnalysisResult,
    ) -> SimilarityAnalysisResponseModel:
        """Convert SimilarityAnalysisResult to response model."""

        # Convert matches by type
        matches_by_type = {}
        for sim_type, matches in result.matches_by_type.items():
            matches_by_type[sim_type.value] = [
                SimilarityProcessor.convert_match_to_model(m) for m in matches
            ]

        # Convert matches by score
        matches_by_score = {}
        for score_level, matches in result.matches_by_score.items():
            matches_by_score[score_level] = [
                SimilarityProcessor.convert_match_to_model(m) for m in matches
            ]

        # Convert cross-language matches
        cross_language_matches = [
            SimilarityProcessor.convert_match_to_model(m) for m in result.cross_language_matches
        ]

        # Convert highest similarity match
        highest_match = None
        if result.highest_similarity_match:
            highest_match = SimilarityProcessor.convert_match_to_model(
                result.highest_similarity_match
            )

        return SimilarityAnalysisResponseModel(
            total_files_analyzed=result.total_files_analyzed,
            total_entities_compared=result.total_entities_compared,
            total_matches_found=result.total_matches_found,
            analysis_time_seconds=result.analysis_time_seconds,
            matches_by_type=matches_by_type,
            matches_by_score=matches_by_score,
            cross_language_matches=cross_language_matches,
            language_distribution=result.language_distribution,
            average_similarity_score=result.average_similarity_score,
            highest_similarity_match=highest_match,
            refactoring_opportunities=result.refactoring_opportunities,
            duplicate_code_ratio=result.duplicate_code_ratio,
            cross_language_patterns=result.cross_language_patterns,
        )

    @staticmethod
    def process_clone_results(
        clones: list[SimilarityMatch],
        valid_files: list[str],
        clone_threshold: float,
    ) -> dict[str, Any]:
        """Process code clone detection results."""
        # Convert to response format
        clone_data = [SimilarityProcessor.convert_match_to_model(clone) for clone in clones]

        # Categorize clones
        exact_clones = [c for c in clone_data if c.score >= 0.95]
        near_clones = [c for c in clone_data if 0.85 <= c.score < 0.95]
        structural_clones = [c for c in clone_data if 0.75 <= c.score < 0.85]

        # Generate refactoring recommendations
        refactoring_recommendations = (
            SimilarityProcessor._generate_clone_refactoring_recommendations(clone_data)
        )

        # Calculate clone statistics
        clone_statistics = SimilarityProcessor._calculate_clone_statistics(clone_data, valid_files)

        return {
            "message": f"Found {len(clones)} code clones across {len(valid_files)} files",
            "total_clones": len(clones),
            "files_analyzed": len(valid_files),
            "clone_categories": {
                "exact_clones": len(exact_clones),
                "near_clones": len(near_clones),
                "structural_clones": len(structural_clones),
            },
            "clones": clone_data,
            "statistics": {
                "average_similarity": (
                    sum(c.score for c in clone_data) / len(clone_data) if clone_data else 0.0
                ),
                "highest_similarity": (max(c.score for c in clone_data) if clone_data else 0.0),
                "clone_threshold_used": clone_threshold,
                **clone_statistics,
            },
            "refactoring_recommendations": refactoring_recommendations,
        }

    @staticmethod
    def process_cross_language_results(
        patterns: list[SimilarityMatch],
        valid_files: list[str],
        pattern_threshold: float,
    ) -> dict[str, Any]:
        """Process cross-language pattern detection results."""
        # Convert to response format
        pattern_data = [SimilarityProcessor.convert_match_to_model(pattern) for pattern in patterns]

        # Group by language pairs
        language_pairs: dict[str, list[SimilarityMatchModel]] = {}
        for pattern in pattern_data:
            lang_pair_str = "-".join(sorted([pattern.source_language, pattern.target_language]))
            if lang_pair_str not in language_pairs:
                language_pairs[lang_pair_str] = []
            language_pairs[lang_pair_str].append(pattern)

        # Calculate statistics
        total_languages = len(
            set(p.source_language for p in pattern_data)
            | set(p.target_language for p in pattern_data)
        )

        # Generate insights
        insights = SimilarityProcessor._generate_cross_language_insights(
            pattern_data, language_pairs
        )

        # Generate migration recommendations
        migration_recommendations = SimilarityProcessor._generate_migration_recommendations(
            pattern_data, language_pairs
        )

        return {
            "message": (
                f"Found {len(patterns)} cross-language patterns across {total_languages} languages"
            ),
            "total_patterns": len(patterns),
            "files_analyzed": len(valid_files),
            "languages_involved": total_languages,
            "language_pairs": {
                pair: len(patterns_list) for pair, patterns_list in language_pairs.items()
            },
            "patterns": pattern_data,
            "statistics": {
                "average_similarity": (
                    sum(p.score for p in pattern_data) / len(pattern_data) if pattern_data else 0.0
                ),
                "pattern_threshold_used": pattern_threshold,
                "strongest_pattern": (
                    max(pattern_data, key=lambda p: p.score).summary if pattern_data else None
                ),
            },
            "insights": {
                "most_similar_languages": (
                    max(language_pairs.items(), key=lambda x: len(x[1]))[0]
                    if language_pairs
                    else None
                ),
                "pattern_count_by_pair": {k: len(v) for k, v in language_pairs.items()},
                **insights,
            },
            "migration_recommendations": migration_recommendations,
        }

    @staticmethod
    def _generate_clone_refactoring_recommendations(
        clones: list[SimilarityMatchModel],
    ) -> list[dict[str, Any]]:
        """Generate refactoring recommendations for code clones."""
        recommendations = []

        # Group clones by priority
        critical_clones = [c for c in clones if c.score >= 0.95 and c.confidence >= 0.8]
        high_clones = [c for c in clones if c.score >= 0.85 and c.confidence >= 0.7]

        if critical_clones:
            recommendations.append(
                {
                    "priority": "critical",
                    "type": "extract_function",
                    "description": "Extract common functionality into shared functions",
                    "affected_clones": len(critical_clones),
                    "estimated_effort": "low",
                    "benefits": [
                        "Eliminate code duplication",
                        "Improve maintainability",
                        "Reduce bug potential",
                    ],
                    "clone_ids": [c.id for c in critical_clones[:5]],  # Limit to first 5
                }
            )

        if high_clones:
            recommendations.append(
                {
                    "priority": "high",
                    "type": "refactor_common_patterns",
                    "description": "Refactor similar code patterns into reusable components",
                    "affected_clones": len(high_clones),
                    "estimated_effort": "medium",
                    "benefits": [
                        "Improve code consistency",
                        "Enable future reuse",
                        "Simplify testing",
                    ],
                    "clone_ids": [c.id for c in high_clones[:3]],
                }
            )

        # Add specific recommendations based on clone types
        for clone in clones[:10]:  # Analyze top 10 clones
            if clone.score >= 0.9:
                recommendations.append(
                    {
                        "priority": SimilarityThresholds.get_refactoring_priority(
                            clone.score,
                            clone.confidence,
                            clone.source_lines[1] - clone.source_lines[0],
                        ),
                        "type": "immediate_consolidation",
                        "description": f"Consolidate identical code in {clone.source_file} and {clone.target_file}",
                        "files": [clone.source_file, clone.target_file],
                        "lines": {
                            clone.source_file: clone.source_lines,
                            clone.target_file: clone.target_lines,
                        },
                        "similarity_score": clone.score,
                        "confidence": clone.confidence,
                    }
                )

        return recommendations

    @staticmethod
    def _calculate_clone_statistics(
        clones: list[SimilarityMatchModel],
        files: list[str],
    ) -> dict[str, Any]:
        """Calculate detailed clone statistics."""
        if not clones:
            return {
                "duplicate_lines": 0,
                "total_duplicate_ratio": 0.0,
                "files_with_clones": 0,
                "average_clone_size": 0,
            }

        # Calculate duplicate lines
        duplicate_line_ranges = set()
        for clone in clones:
            # Add source lines
            for line in range(clone.source_lines[0], clone.source_lines[1] + 1):
                duplicate_line_ranges.add(f"{clone.source_file}:{line}")
            # Add target lines
            for line in range(clone.target_lines[0], clone.target_lines[1] + 1):
                duplicate_line_ranges.add(f"{clone.target_file}:{line}")

        # Calculate files with clones
        files_with_clones = set()
        for clone in clones:
            files_with_clones.add(clone.source_file)
            files_with_clones.add(clone.target_file)

        # Calculate average clone size
        clone_sizes = []
        for clone in clones:
            size = clone.source_lines[1] - clone.source_lines[0] + 1
            clone_sizes.append(size)

        average_clone_size = sum(clone_sizes) / len(clone_sizes) if clone_sizes else 0

        return {
            "duplicate_lines": len(duplicate_line_ranges),
            "files_with_clones": len(files_with_clones),
            "average_clone_size": average_clone_size,
            "max_clone_size": max(clone_sizes) if clone_sizes else 0,
            "min_clone_size": min(clone_sizes) if clone_sizes else 0,
        }

    @staticmethod
    def _generate_cross_language_insights(
        patterns: list[SimilarityMatchModel],
        language_pairs: dict[str, list[SimilarityMatchModel]],
    ) -> dict[str, Any]:
        """Generate insights from cross-language pattern analysis."""
        insights: dict[str, Any] = {}

        if not patterns:
            return insights

        # Language consistency analysis
        language_consistency = {}
        for lang_pair, pair_patterns in language_pairs.items():
            avg_score = sum(p.score for p in pair_patterns) / len(pair_patterns)
            language_consistency[lang_pair] = {
                "average_similarity": avg_score,
                "pattern_count": len(pair_patterns),
                "consistency_level": (
                    "high" if avg_score > 0.8 else "medium" if avg_score > 0.6 else "low"
                ),
            }

        insights["language_consistency"] = language_consistency

        # Pattern type analysis
        pattern_types: dict[str, int] = {}
        for pattern in patterns:
            for feature in pattern.matching_features:
                pattern_types[feature] = pattern_types.get(feature, 0) + 1

        insights["common_patterns"] = dict(
            sorted(pattern_types.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Migration readiness
        migration_ready_pairs = []
        for lang_pair, pair_patterns in language_pairs.items():
            high_similarity_patterns = [p for p in pair_patterns if p.score > 0.8]
            if len(high_similarity_patterns) >= 3:
                migration_ready_pairs.append(
                    {
                        "language_pair": lang_pair,
                        "ready_patterns": len(high_similarity_patterns),
                        "readiness_score": sum(p.score for p in high_similarity_patterns)
                        / len(high_similarity_patterns),
                    }
                )

        insights["migration_readiness"] = sorted(
            migration_ready_pairs, key=lambda x: x["readiness_score"], reverse=True
        )

        return insights

    @staticmethod
    def _generate_migration_recommendations(
        patterns: list[SimilarityMatchModel],
        language_pairs: dict[str, list[SimilarityMatchModel]],
    ) -> list[dict[str, Any]]:
        """Generate code migration recommendations."""
        recommendations = []

        # Find high-similarity cross-language patterns
        for lang_pair, pair_patterns in language_pairs.items():
            high_sim_patterns = [p for p in pair_patterns if p.score > 0.8]

            if len(high_sim_patterns) >= 2:
                languages = lang_pair.split("-")
                recommendations.append(
                    {
                        "type": "cross_language_standardization",
                        "priority": "high" if len(high_sim_patterns) >= 5 else "medium",
                        "description": f"Standardize implementations between {languages[0]} and {languages[1]}",
                        "languages": languages,
                        "similar_patterns": len(high_sim_patterns),
                        "average_similarity": sum(p.score for p in high_sim_patterns)
                        / len(high_sim_patterns),
                        "recommended_actions": [
                            "Create common interface specifications",
                            "Align naming conventions",
                            "Standardize error handling patterns",
                        ],
                        "pattern_examples": [
                            {
                                "source": p.source_file,
                                "target": p.target_file,
                                "similarity": p.score,
                                "summary": p.summary,
                            }
                            for p in high_sim_patterns[:3]
                        ],
                    }
                )

        # Find potential code porting opportunities
        exact_patterns = [p for p in patterns if p.score > 0.9]
        if exact_patterns:
            recommendations.append(
                {
                    "type": "code_porting_opportunity",
                    "priority": "medium",
                    "description": "High-similarity patterns suggest easy porting opportunities",
                    "pattern_count": len(exact_patterns),
                    "recommended_actions": [
                        "Identify missing functionality in target languages",
                        "Create automated translation scripts",
                        "Establish cross-language testing strategies",
                    ],
                    "examples": [
                        {
                            "source_language": p.source_language,
                            "target_language": p.target_language,
                            "similarity": p.score,
                            "description": p.summary,
                        }
                        for p in exact_patterns[:5]
                    ],
                }
            )

        return recommendations


def process_similarity_request(
    request_data: dict[str, Any], analysis_type: str = "general"
) -> dict[str, Any]:
    """
    Process a single similarity analysis request.

    Args:
        request_data: Request data containing code segments and options
        analysis_type: Type of similarity analysis to perform

    Returns:
        Processed similarity analysis results
    """
    try:
        from .similarity_algorithms import calculate_similarity

        # Extract request parameters
        code_a = request_data.get("code_a", "")
        code_b = request_data.get("code_b", "")
        language = request_data.get("language", "python")

        if not code_a or not code_b:
            return {
                "success": False,
                "error": "Both code_a and code_b are required",
                "results": {},
            }

        # Calculate similarity
        similarity_results = calculate_similarity(code_a, code_b, language)

        # Process results based on analysis type
        processed_results = {
            "success": True,
            "analysis_type": analysis_type,
            "language": language,
            "similarity_scores": similarity_results,
            "overall_score": similarity_results.get("overall", 0.0),
            "confidence": _calculate_confidence(similarity_results),
            "category": SimilarityThresholds.categorize_by_score(
                similarity_results.get("overall", 0.0)
            ),
            "recommendations": _generate_basic_recommendations(similarity_results, analysis_type),
        }

        return processed_results

    except Exception as e:
        logger.error(f"Error processing similarity request: {e}")
        return {"success": False, "error": str(e), "results": {}}


def batch_process_similarities(
    requests: list[dict[str, Any]],
    analysis_type: str = "general",
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Process multiple similarity analysis requests in batch.

    Args:
        requests: List of similarity analysis requests
        analysis_type: Type of similarity analysis to perform
        max_concurrent: Maximum number of concurrent processes

    Returns:
        List of processed similarity analysis results
    """
    results = []

    try:
        # Process requests in chunks to avoid overwhelming the system
        chunk_size = min(max_concurrent, len(requests))

        for i in range(0, len(requests), chunk_size):
            chunk = requests[i : i + chunk_size]

            # Process chunk
            chunk_results = []
            for request in chunk:
                try:
                    result = process_similarity_request(request, analysis_type)
                    result["request_id"] = request.get("id", f"request_{i}")
                    chunk_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing request {request.get('id', 'unknown')}: {e}")
                    chunk_results.append(
                        {
                            "success": False,
                            "error": str(e),
                            "request_id": request.get("id", f"request_{i}"),
                            "results": {},
                        }
                    )

            results.extend(chunk_results)

        # Generate batch summary
        successful_results: list[dict[str, Any]] = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]

        batch_summary = {
            "total_requests": len(requests),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "average_similarity": (
                sum(r.get("overall_score", 0.0) for r in successful_results)
                / len(successful_results)
                if successful_results
                else 0.0
            ),
            "high_similarity_count": len(
                [r for r in successful_results if r.get("overall_score", 0.0) > 0.8]
            ),
        }

        return {"batch_summary": batch_summary, "results": results}

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return {
            "batch_summary": {
                "total_requests": len(requests),
                "successful": 0,
                "failed": len(requests),
                "error": str(e),
            },
            "results": [],
        }


def _calculate_confidence(similarity_results: dict[str, float]) -> float:
    """Calculate confidence score from similarity results."""
    scores = [
        similarity_results.get("structural", 0.0),
        similarity_results.get("semantic", 0.0),
        similarity_results.get("syntactic", 0.0),
        similarity_results.get("exact", 0.0),
    ]

    # Confidence based on consistency of scores
    score_variance = sum((s - similarity_results.get("overall", 0.0)) ** 2 for s in scores) / len(
        scores
    )
    base_confidence = similarity_results.get("overall", 0.0)

    # Lower confidence if scores are inconsistent
    consistency_factor = max(0.0, 1.0 - score_variance)

    return min(1.0, base_confidence * (0.7 + 0.3 * consistency_factor))


def _generate_basic_recommendations(
    similarity_results: dict[str, float], analysis_type: str
) -> list[str]:
    """Generate basic recommendations based on similarity scores."""
    recommendations = []
    overall_score = similarity_results.get("overall", 0.0)

    if overall_score > 0.95:
        recommendations.append("Consider extracting common code into a shared function")
        recommendations.append("Review for potential code duplication")
    elif overall_score > 0.85:
        recommendations.append("Investigate structural similarities for refactoring opportunities")
        recommendations.append("Consider creating shared utilities or base classes")
    elif overall_score > 0.7:
        recommendations.append("Code shows moderate similarity - review for design patterns")
        recommendations.append("Consider establishing coding standards for consistency")
    elif overall_score > 0.5:
        recommendations.append("Some similarity detected - may indicate common requirements")
        recommendations.append("Review for potential shared interfaces or contracts")
    else:
        recommendations.append("Low similarity - code appears to be functionally distinct")

    # Analysis type specific recommendations
    if analysis_type == "refactoring":
        if overall_score > 0.8:
            recommendations.append("High refactoring potential - prioritize for cleanup")
        else:
            recommendations.append("Limited refactoring benefits expected")
    elif analysis_type == "clone_detection":
        if overall_score > 0.9:
            recommendations.append("Potential code clone detected - investigate immediately")
        elif overall_score > 0.7:
            recommendations.append("Possible code clone - requires manual review")

    return recommendations
