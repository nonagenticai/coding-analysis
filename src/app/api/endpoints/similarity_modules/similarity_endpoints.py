"""
Similarity Detection API Endpoints

Main API endpoints for similarity analysis operations.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic_core import to_jsonable_python

from auth.tenant import require_tenant
from services.similarity_detector import (
    SimilarityScope,
    get_similarity_detector,
)

from ...models.common_api_models import BaseResponse
from ...models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from ...models.similarity_api_models import (
    CloneDetectionResponse,
    CrossLanguagePatternResponse,
    SimilarityCapabilitiesResponse,
)
from .similarity_config import (
    CloneDetectionRequest,
    CrossLanguagePatternRequest,
    SimilarityAnalysisRequest,
    SimilarityConfig,
)
from .similarity_processing import SimilarityAnalysisResponse, SimilarityProcessor

logger = logging.getLogger(__name__)


def _as_json_value(value: Any) -> Any:
    """Model objects -> plain JSON values for the ``dict``-typed response fields.

    The processors return ``SimilarityMatchModel`` instances; the response models
    declare ``list[dict]``, so a run that FOUND a clone failed validation with a
    500 while a run that found none passed. Only the empty case was ever green.
    """
    return to_jsonable_python(value)


def _as_json_list(values: list[Any]) -> list[dict[str, Any]]:
    return [_as_json_value(v) for v in values]


router = APIRouter()


@router.post(
    "/analyze",
    response_model=BaseResponse,
    operation_id="analyze_similarity",
    summary="🔍 Analyze Code Similarity",
    description="""
    🎯 **Comprehensive Code Similarity Analysis** - Detect similarities, patterns, and duplicates across multiple files and languages

    ## Analysis Scopes
    - **File Level**: Compares entire files for overall similarity
    - **Function Level** (Recommended): Compares individual functions and methods
    - **Class Level**: Compares classes, structs, and interfaces
    - **Block Level**: Compares code blocks and segments

    ## Similarity Types Detected
    - **Exact Duplicates** (0.95+): Identical or nearly identical code
    - **Near Duplicates** (0.85+): Very similar code with minor differences
    - **Structural Similarity** (0.75+): Similar code structure and patterns
    - **Semantic Similarity** (0.70+): Similar functionality or purpose
    - **Syntactic Similarity** (0.65+): Similar coding style and conventions
    - **Cross-Language Patterns** (0.60+): Similar implementations across languages

    ## Features
    - Multi-dimensional comparison (structural, semantic, syntactic)
    - Advanced metrics (cosine similarity, Jaccard similarity)
    - Smart filtering and configurable thresholds
    - Cross-language detection and analysis
    - Performance optimization for large codebases
    - Actionable refactoring insights and recommendations

    ## Use Cases
    - Code quality improvement and refactoring planning
    - Cross-language development and standardization
    - Code review assistance and pattern analysis
    - Architecture analysis and component identification
    """,
    responses={
        200: {"description": "Similarity analysis completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Files not found or inaccessible"},
        422: {"description": "Analysis failed due to processing error"},
        500: {"description": "Internal similarity analysis error"},
    },
)
async def analyze_similarity(
    request: SimilarityAnalysisRequest = Body(..., description="Similarity analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> SimilarityAnalysisResponse:
    """Perform comprehensive code similarity analysis."""

    file_paths: list[str] | None = None
    try:
        # Validate request
        SimilarityConfig.validate_analysis_request(request)

        # Resolve file paths
        file_paths = await SimilarityConfig.resolve_file_paths(
            request.container_name,
            request.file_paths,
            request.max_files,
            tenant_id=tenant_id,
            repo=request.repo,
        )

        if len(file_paths) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 valid files for similarity analysis",
            )

        # Validate scope
        try:
            scope = SimilarityScope(request.scope)
        except ValueError as exc:
            valid_scopes = [s.value for s in SimilarityScope]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scope. Valid scopes: {', '.join(valid_scopes)}",
            ) from exc

        # Get similarity detector
        detector = get_similarity_detector()

        # Perform similarity analysis
        result = await detector.analyze_similarity(
            file_paths=file_paths,
            scope=scope,
            min_similarity_score=request.min_similarity_score,
            detect_cross_language=request.detect_cross_language,
            max_files=request.max_files,
        )

        # Convert to response model
        response_model = SimilarityProcessor.convert_result_to_model(result)

        logger.info(
            "Similarity analysis completed: %s matches found",
            result.total_matches_found,
        )

        return SimilarityAnalysisResponse(
            message=(
                f"Analyzed {result.total_files_analyzed} files, found {result.total_matches_found} similarity matches"
            ),
            result=response_model,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.exception("Similarity analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}") from e
    finally:
        SimilarityConfig.release_files(file_paths)


@router.post(
    "/detect-clones",
    response_model=CloneDetectionResponse,
    operation_id="detect_clones",
    summary="📋 Detect Code Clones",
    description="""
    🔍 **Code Clone Detection** - Find exact and near-duplicate code segments

    Specialized detection of code clones - segments of code that are identical or nearly identical.
    This is a focused subset of similarity analysis optimized for finding copy-paste code.

    ## Clone Types Detected
    - **Type 1 Clones** (0.95-1.0): Exact duplicates - identical code except for whitespace and comments
    - **Type 2 Clones** (0.85-0.95): Near duplicates - identical structure with variable/function name changes
    - **Type 3 Clones** (0.75-0.85): Structural clones - similar code with minor statement additions/deletions

    ## Features
    - Fast clone detection optimized for large codebases
    - Line-precise location information
    - Confidence scoring for each clone pair
    - Language-agnostic detection

    ## Use Cases
    - Code quality assessment and refactoring planning
    - Technical debt identification and code review assistance
    """,
    responses={
        200: {"description": "Clone detection completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Need at least 2 valid files for clone detection"},
        500: {"description": "Internal clone detection error"},
    },
)
async def detect_clones(
    request: CloneDetectionRequest = Body(..., description="Clone detection request"),
    tenant_id: str = Depends(require_tenant),
) -> CloneDetectionResponse:
    """Detect code clones and near-duplicates."""

    resolved: list[str] | None = None
    try:
        if not request.file_paths and not request.container_name:
            raise HTTPException(status_code=400, detail="Provide file_paths or container_name")

        # Resolve and validate file paths
        if request.file_paths:
            valid_files = SimilarityConfig.validate_file_paths(request.file_paths)
        elif request.container_name:
            resolved = await SimilarityConfig.resolve_file_paths(
                request.container_name, None, None, tenant_id=tenant_id, repo=request.repo
            )
            valid_files = SimilarityConfig.validate_file_paths(resolved)
        else:
            valid_files = []

        if len(valid_files) < 2:
            return CloneDetectionResponse(
                message="Insufficient files for clone detection",
                total_clones=0,
                files_analyzed=len(valid_files),
                clone_threshold=request.clone_threshold,
                clones=[],
                summary={},
            )

        # Get similarity detector
        detector = get_similarity_detector()

        # Detect clones
        clones = await detector.detect_code_clones(
            file_paths=valid_files,
            clone_threshold=request.clone_threshold,
        )

        # Process results
        response = SimilarityProcessor.process_clone_results(
            clones, valid_files, request.clone_threshold
        )

        logger.info("Clone detection completed: %s clones found", len(clones))

        return CloneDetectionResponse(
            message=f"Clone detection completed: {response['total_clones']} clones found",
            total_clones=response["total_clones"],
            files_analyzed=response["files_analyzed"],
            clone_threshold=request.clone_threshold,
            clones=_as_json_list(response.get("clones", [])),
            summary=_as_json_value(
                {
                    **response.get("summary", {}),
                    **{
                        key: response[key]
                        for key in ("clone_categories", "statistics", "refactoring_recommendations")
                        if key in response
                    },
                }
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Clone detection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Clone detection failed: {str(e)}") from e
    finally:
        SimilarityConfig.release_files(resolved)


@router.post(
    "/cross-language-patterns",
    response_model=CrossLanguagePatternResponse,
    operation_id="find_cross_language_patterns",
    summary="🌐 Find Cross-Language Patterns",
    description="""
    🔗 **Cross-Language Pattern Detection** - Discover similar implementations across different programming languages

    Identifies similar algorithmic patterns, design implementations, and architectural approaches
    across different programming languages in polyglot codebases.

    ## Pattern Types Detected
    - **Algorithmic Patterns**: Similar sorting/searching implementations, data structure operations
    - **Design Patterns**: Singleton, Factory, Observer pattern implementations across languages
    - **API Patterns**: Similar REST endpoint implementations, database access patterns
    - **Data Processing Patterns**: File I/O operations, JSON/XML parsing, configuration management

    ## Features
    - Language-agnostic analysis that abstracts syntax differences
    - Pattern insights for consistency analysis and learning opportunities
    - Migration pathway suggestions and implementation quality comparison

    ## Use Cases
    - Code migration and ensuring feature parity across platforms
    - Learning and training - compare language idioms and best practices
    - Architecture consistency - ensure consistent API designs and validate patterns
    """,
    responses={
        200: {"description": "Cross-language pattern detection completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Need at least 2 valid files for cross-language pattern detection"},
        500: {"description": "Internal cross-language pattern detection error"},
    },
)
async def find_cross_language_patterns(
    request: CrossLanguagePatternRequest = Body(..., description="Cross-language pattern request"),
    tenant_id: str = Depends(require_tenant),
) -> CrossLanguagePatternResponse:
    """Find similar patterns across different programming languages."""

    resolved: list[str] | None = None
    try:
        if not request.file_paths and not request.container_name:
            raise HTTPException(status_code=400, detail="Provide file_paths or container_name")

        # Resolve and validate file paths
        if request.file_paths:
            valid_files = SimilarityConfig.validate_file_paths(request.file_paths)
        elif request.container_name:
            resolved = await SimilarityConfig.resolve_file_paths(
                request.container_name, None, None, tenant_id=tenant_id, repo=request.repo
            )
            valid_files = SimilarityConfig.validate_file_paths(resolved)
        else:
            valid_files = []

        if len(valid_files) < 2:
            return CrossLanguagePatternResponse(
                message="Insufficient files for cross-language pattern detection",
                total_patterns=0,
                files_analyzed=len(valid_files),
                pattern_threshold=request.pattern_threshold,
                patterns=[],
                language_pairs={},
                summary={},
            )

        # Get similarity detector
        detector = get_similarity_detector()

        # Find cross-language patterns
        patterns = await detector.find_cross_language_patterns(
            file_paths=valid_files,
            pattern_threshold=request.pattern_threshold,
        )

        # Process results
        response = SimilarityProcessor.process_cross_language_results(
            patterns, valid_files, request.pattern_threshold
        )

        logger.info(
            "Cross-language pattern detection completed: %s patterns found",
            len(patterns),
        )

        return CrossLanguagePatternResponse(
            message=f"Cross-language pattern detection completed: {response['total_patterns']} patterns found",
            total_patterns=response["total_patterns"],
            files_analyzed=response["files_analyzed"],
            pattern_threshold=request.pattern_threshold,
            patterns=_as_json_list(response.get("patterns", [])),
            language_pairs=response.get("language_pairs", {}),
            summary=_as_json_value(
                {
                    **response.get("summary", {}),
                    **{
                        key: response[key]
                        for key in (
                            "languages_involved",
                            "statistics",
                            "insights",
                            "migration_recommendations",
                        )
                        if key in response
                    },
                }
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Cross-language pattern detection failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Cross-language pattern detection failed: {str(e)}",
        ) from e
    finally:
        SimilarityConfig.release_files(resolved)


@router.get(
    "/capabilities",
    response_model=SimilarityCapabilitiesResponse,
    operation_id="get_similarity_capabilities",
    summary="📋 Similarity Detection Capabilities",
    description="""
    🔍 **Similarity Detection System Capabilities** - Discover available analysis types and features

    Provides comprehensive information about the code similarity detection system's capabilities,
    including supported analysis types, similarity algorithms, and configuration options.

    ## Response Information
    - **Similarity Types**: Available similarity detection types
    - **Analysis Scopes**: Supported analysis granularity levels
    - **Algorithms**: Similarity calculation methods used
    - **Language Support**: Programming languages with similarity support
    - **Thresholds**: Default and recommended similarity thresholds
    - **Performance**: System limitations and optimization features

    ## Use Cases
    - **Planning**: Understand available analysis options
    - **Integration**: Determine compatibility with workflows
    - **Configuration**: Set appropriate thresholds and parameters
    - **Development**: Plan custom similarity workflows
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_similarity_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> SimilarityCapabilitiesResponse:
    """Get similarity detection system capabilities."""

    try:
        capabilities = SimilarityConfig.get_capabilities()

        # algorithms is a dict[str, str] in config; response model expects list[str]
        raw_algorithms = capabilities.get("algorithms", {})
        algorithm_list = (
            list(raw_algorithms.keys()) if isinstance(raw_algorithms, dict) else raw_algorithms
        )

        return SimilarityCapabilitiesResponse(
            message="Similarity detection capabilities retrieved successfully",
            similarity_types=capabilities.get("similarity_types", []),
            analysis_scopes=capabilities.get("analysis_scopes", []),
            algorithms=[str(a) for a in algorithm_list],
            supported_languages=capabilities.get("supported_languages", []),
            thresholds=capabilities.get("default_thresholds", {}),
            performance_limits=capabilities.get("limitations", {}),
        )

    except Exception as e:
        logger.error("Error retrieving similarity capabilities: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error retrieving capabilities: {str(e)}"
        ) from e
