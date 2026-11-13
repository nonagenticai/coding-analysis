"""Complexity module."""

from __future__ import annotations

"""
Code Complexity Analysis API Endpoints

Advanced complexity analysis endpoints that provide language-specific
complexity metrics and quality assessments.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.responses import (
    bad_request,
    error_response,
    internal_error,
    not_found,
)
from app.core.container_manager import get_container_manager
from app.core.dependencies import get_multi_language_factory
from app.core.file_service import FileService
from auth.tenant import require_tenant
from processing.analyzers.complexity_analyzers import (
    ComplexityLevel,
    get_complexity_service,
)
from processing.code_document_processor import CodeDocumentProcessor
from processing.languages.base import LanguageProcessor
from processing.multi_language_factory import MultiLanguageProcessorFactory
from processing.workspace_repos import RepoSelector, repo_root
from services.docker_service import DockerService

from ..models.common_api_models import BaseResponse, ResponseStatus
from ..models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error422Response,
    Error500Response,
    ValidationErrorResponse,
)
from ..utils.workspace_files import (
    InvalidWorkspacePathError,
    invalid_workspace_path_http,
    normalize_workspace_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _resolve_container_files(
    container_name: str,
    file_paths: list[str] | None = None,
    *,
    tenant_id: str | None,
    repo: str | None = None,
) -> list[Path]:
    """Resolve code files from a container to readable LOCAL paths.

    The code lives inside the container (e.g. ``/workspace``), which this
    analysis server cannot read from its own filesystem. Read the files via
    container exec (the same mechanism the quality path uses) and materialize
    them into a per-container temp directory so the existing local-FS analysers
    keep working unchanged.

    ``tenant_id`` is the CALLER's tenant and is required (keyword-only, no
    default) so a call site cannot forget it: coding-containers' exec and
    lookup are tenant-scoped, and an untenanted call is resolved as this
    service's own account and 404'd for every customer workspace -- the same
    defect coding-analysis#91 fixed for the quality path.

    Requested paths are confined to the workspace (repo-relative, or absolute
    under the workspace root; never ``..``): the local copy is written at
    ``local_root / rel``, so an unconfined path would also escape the local
    temp dir on this pod.
    """
    try:
        # Get container manager and find container
        container_manager = get_container_manager()

        try:
            container = await container_manager.get_container(container_name, tenant_id)
        except ValueError:
            raise not_found("Container", container_name)
        if container is None:
            raise not_found("Container", container_name)

        # The selected repo's root: the workspace root (primary) or .repos/<name>.
        base = repo_root(str(container.codebase_path).rstrip("/") or "/workspace", repo)
        docker_service = DockerService()

        if file_paths:
            try:
                rel_paths = [normalize_workspace_path(fp, base) for fp in file_paths]
            except InvalidWorkspacePathError as exc:
                raise invalid_workspace_path_http(exc) from exc
            container_files = [f"{base}/{rel}" for rel in rel_paths]
        else:
            container_files = await _list_container_code_files(
                docker_service, container_name, base, tenant_id=tenant_id
            )

        # Materialize into a deterministic per-container dir (cleared each call so
        # disk use stays bounded to one workspace copy per container).
        local_root = Path(tempfile.gettempdir()) / "coding-analysis-ws" / container_name
        shutil.rmtree(local_root, ignore_errors=True)
        local_root.mkdir(parents=True, exist_ok=True)

        resolved_paths: list[Path] = []
        for cfile in container_files:
            try:
                rel = normalize_workspace_path(cfile, base)
            except InvalidWorkspacePathError:
                logger.warning("Skipping %s: outside workspace %s", cfile, base)
                continue
            result = await docker_service.exec_in_container(
                container_name, ["cat", cfile], timeout=30, tenant_id=tenant_id
            )
            if result.get("exit_code") != 0 or result.get("stdout") is None:
                logger.warning("Could not read %s from container %s", cfile, container_name)
                continue
            local_path = local_root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(result["stdout"], encoding="utf-8")
            resolved_paths.append(local_path)

        if not resolved_paths:
            raise not_found("Files", f"no readable code files in container '{container_name}'")

        return resolved_paths

    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        logger.error(f"Error resolving container files for {container_name}: {e}")
        raise internal_error(f"Failed to resolve container files: {str(e)}") from e


async def _list_container_code_files(
    docker_service: DockerService,
    container_name: str,
    base: str,
    *,
    tenant_id: str | None,
) -> list[str]:
    """List code-file paths inside the container via ``find`` (exec)."""
    name_args: list[str] = []
    for i, ext in enumerate(sorted(CODE_EXTENSIONS)):
        if i:
            name_args.append("-o")
        name_args.extend(["-name", f"*{ext}"])
    # Hidden paths BELOW ``base`` are skipped (.git, .venv, and the source
    # checkouts under .repos/). Anchored at ``base``: the old unanchored
    # ``*/.*`` also matched ``base`` itself when it is ``<ws>/.repos/<name>``.
    cmd = [
        "find",
        base,
        "-type",
        "f",
        "(",
        *name_args,
        ")",
        "-not",
        "-path",
        f"{base}/.*",
        "-not",
        "-path",
        f"{base}/*/.*",
    ]
    result = await docker_service.exec_in_container(
        container_name, cmd, timeout=30, tenant_id=tenant_id
    )
    if result.get("exit_code") != 0:
        return []
    return [ln.strip() for ln in (result.get("stdout") or "").splitlines() if ln.strip()]


CODE_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".rs",
    ".go",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".kt",
    ".scala",
    ".swift",
    ".m",
    ".mm",
}


def _find_code_files(directory: Path) -> list[Path]:
    """Find all code files in a directory."""
    code_files = []
    for file_path in directory.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in CODE_EXTENSIONS
            and not any(part.startswith(".") for part in file_path.parts)
        ):
            code_files.append(file_path)

    return sorted(code_files)


def _validate_file_path(file_path_str: str) -> Path:
    """Validate and return Path object for file path string."""
    if len(file_path_str) > 255:
        raise bad_request("File path too long. Maximum length is 255 characters.")

    try:
        file_path = Path(file_path_str)
    except (ValueError, OSError) as e:
        raise bad_request(f"Invalid file path: {str(e)}") from e

    if not file_path.exists():
        raise not_found("File", file_path_str)

    return file_path


async def _get_processor_for_analysis(
    factory: MultiLanguageProcessorFactory,
    file_path: Path,
    language: str | None = None,
) -> LanguageProcessor | CodeDocumentProcessor:
    """Get appropriate processor for file analysis."""
    if language:
        processor = await factory.get_processor(language)
    else:
        processor = await factory.get_processor_for_file(file_path)

    if not processor:
        raise error_response(
            status_code=422,
            message=f"No processor available for file: {file_path}",
            error_code="PROCESSOR_NOT_FOUND",
        )

    return processor


async def _parse_file_content(processor: Any, file_path: Path, content: str) -> Any:
    """Parse file content using appropriate processor method."""
    if hasattr(processor, "parse_file_with_metrics"):
        parse_result = await processor.parse_file_with_metrics(file_path, content)
    elif hasattr(processor, "parse_file"):
        parse_result = await processor.parse_file(file_path, content)
    else:
        raise error_response(
            status_code=422,
            message=f"Processor for file {file_path} does not support parsing",
            error_code="PARSING_NOT_SUPPORTED",
        )
    if not parse_result.success:
        raise error_response(
            status_code=422,
            message=f"Failed to parse file: {'; '.join(parse_result.errors)}",
            error_code="PARSE_FAILED",
        )

    return parse_result


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _convert_metrics_to_response(
    metrics: Any, include_suggestions: bool = True
) -> ComplexityMetricsResponse:
    """Convert complexity metrics to response model."""
    return ComplexityMetricsResponse(
        cyclomatic_complexity=metrics.cyclomatic_complexity,
        cognitive_complexity=metrics.cognitive_complexity,
        lines_of_code=metrics.lines_of_code,
        nesting_depth=metrics.nesting_depth,
        parameter_count=metrics.parameter_count,
        dependency_count=metrics.dependency_count,
        halstead_difficulty=metrics.halstead_difficulty,
        maintainability_index=metrics.maintainability_index,
        coupling_factor=metrics.coupling_factor,
        complexity_level=metrics.complexity_level.value,
        risk_factors=metrics.risk_factors,
        suggestions=(metrics.suggestions if include_suggestions else []),
        language=metrics.language,
        entity_type=metrics.entity_type,
        entity_name=metrics.entity_name,
        computed=getattr(metrics, "computed", True) is not False,
        not_computed_reason=_optional_str(getattr(metrics, "not_computed_reason", None)),
    )


def _create_file_complexity_response(
    complexity_report: Any,
    entity_metrics: list[ComplexityMetricsResponse],
    include_suggestions: bool = True,
) -> FileComplexityResponse:
    """Create file complexity response from complexity report."""
    return FileComplexityResponse(
        file_path=complexity_report.file_path,
        language=complexity_report.language,
        total_loc=complexity_report.total_loc,
        average_complexity=complexity_report.average_complexity,
        max_complexity=complexity_report.max_complexity,
        total_entities=complexity_report.total_entities,
        complex_entities=complexity_report.complex_entities,
        entities_not_computed=_optional_int(getattr(complexity_report, "entities_not_computed", 0)),
        entity_metrics=entity_metrics,
        hotspots=complexity_report.hotspots,
        refactoring_suggestions=(
            complexity_report.refactoring_suggestions if include_suggestions else []
        ),
        quality_score=complexity_report.quality_score,
        metrics_computed=getattr(complexity_report, "metrics_computed", True) is not False,
    )


def _initialize_batch_summary(total_files: int) -> dict[str, Any]:
    """Initialize batch analysis summary statistics."""
    return {
        "total_files": total_files,
        "analyzed_files": 0,
        "failed_files": 0,
        "below_threshold": 0,
        "average_quality": 0.0,
        "total_loc": 0,
        "total_entities": 0,
        "total_quality": 0.0,
        "scored_files": 0,
        "not_computed_files": 0,
        "languages": set(),
        "all_hotspots": [],
        "quality_distribution": {
            "excellent": 0,  # 90+
            "good": 0,  # 70-89
            "fair": 0,  # 50-69
            "poor": 0,  # <50
        },
    }


def _update_quality_distribution(quality_distribution: dict[str, int], quality: float) -> None:
    """Update quality distribution counts based on quality score."""
    if quality >= 90:
        quality_distribution["excellent"] += 1
    elif quality >= 70:
        quality_distribution["good"] += 1
    elif quality >= 50:
        quality_distribution["fair"] += 1
    else:
        quality_distribution["poor"] += 1


async def _process_single_file_for_batch(
    file_path_str: str,
    factory: MultiLanguageProcessorFactory,
    complexity_service: Any,
    include_suggestions: bool = True,
) -> FileComplexityResponse | None:
    """Process a single file for batch complexity analysis."""
    try:
        # Validate file path length
        if len(file_path_str) > 255:
            logger.warning("File path too long: %s...", file_path_str[:50])
            return None

        # Validate file path and existence
        file_path = _validate_file_path(file_path_str)
        # Get processor and parse file
        processor = await _get_processor_for_analysis(factory, file_path)
        content = file_path.read_text(encoding="utf-8")
        parse_result = await _parse_file_content(processor, file_path, content)

        # Analyze complexity
        complexity_report = complexity_service.analyze_file_complexity(parse_result, content)

        # Convert to response
        entity_metrics = [
            _convert_metrics_to_response(metrics, include_suggestions)
            for metrics in complexity_report.entity_metrics
        ]

        return _create_file_complexity_response(
            complexity_report, entity_metrics, include_suggestions
        )

    except (
        ValueError,
        OSError,
        FileNotFoundError,
        UnicodeDecodeError,
    ) as e:
        logger.error("Error analyzing %s: %s", file_path_str, e)
        return None
    except (ImportError, AttributeError, TypeError) as e:
        logger.error("Processing error analyzing %s: %s", file_path_str, e)
        return None
    except Exception as e:
        # Catch-all for batch processing resilience - log and continue with other files
        # This includes HTTPException from response helpers
        logger.exception("Unexpected error analyzing %s: %s", file_path_str, e)
        return None


def _update_batch_statistics(
    file_response: FileComplexityResponse,
    quality_threshold: float,
    summary_stats: dict[str, Any],
) -> None:
    """Update batch analysis summary statistics."""
    summary_stats["analyzed_files"] += 1
    summary_stats["total_loc"] += file_response.total_loc
    summary_stats["total_entities"] += file_response.total_entities
    summary_stats["languages"].add(file_response.language)

    # Collect hotspots
    for hotspot in file_response.hotspots:
        summary_stats["all_hotspots"].append(f"{Path(file_response.file_path).name}: {hotspot}")

    quality = file_response.quality_score
    if quality is None:
        # Nothing in the file could be measured: not a score, not "clean".
        summary_stats["not_computed_files"] = summary_stats.get("not_computed_files", 0) + 1
        return
    summary_stats["scored_files"] = summary_stats.get("scored_files", 0) + 1
    summary_stats["total_quality"] = summary_stats.get("total_quality", 0.0) + quality

    if quality < quality_threshold:
        summary_stats["below_threshold"] += 1

    _update_quality_distribution(summary_stats["quality_distribution"], quality)


class ComplexityAnalysisRequest(RepoSelector):
    """Request for complexity analysis."""

    container_name: str = Field(..., description="Container name to analyze")
    file_paths: list[str] | None = Field(
        None,
        description="Specific file paths to analyze (optional, analyzes all files if not provided)",
    )
    language: str | None = Field(None, description="Language hint (auto-detected if not provided)")
    include_suggestions: bool = Field(True, description="Include refactoring suggestions")

    # Pydantic configuration for OpenAPI schema generation
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-java-project",
                "file_paths": ["/app/data/repo/src/main/java/MyClass.java"],
                "language": "java",
                "include_suggestions": True,
            },
        }
    )


class BatchComplexityRequest(RepoSelector):
    """Request for batch complexity analysis."""

    container_name: str = Field(..., description="Container name to analyze")
    file_paths: list[str] | None = Field(
        None,
        description="List of specific file paths to analyze (optional, analyzes all files if not provided)",
    )
    include_file_suggestions: bool = Field(True, description="Include file-level suggestions")
    quality_threshold: float = Field(70.0, description="Quality score threshold for flagging")

    # Pydantic configuration for OpenAPI schema generation
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-java-project",
                "file_paths": [
                    "/app/data/repo/src/main/java/service/UserService.java",
                    "/app/data/repo/src/main/java/model/User.java",
                    "/app/data/repo/src/main/java/controller/UserController.java",
                ],
                "include_file_suggestions": True,
                "quality_threshold": 70.0,
            },
        }
    )


class ComplexityMetricsResponse(BaseModel):
    """Response model for complexity metrics."""

    cyclomatic_complexity: float
    cognitive_complexity: float
    lines_of_code: int
    nesting_depth: int
    parameter_count: int
    dependency_count: int
    halstead_difficulty: float
    maintainability_index: float
    coupling_factor: float
    complexity_level: str
    risk_factors: list[str]
    suggestions: list[str]
    language: str
    entity_type: str
    entity_name: str
    computed: bool = Field(
        True,
        description="False when the entity body could not be parsed/located; "
        "the numeric metrics are then 0 and must not be read as simple code",
    )
    not_computed_reason: str | None = None


class FileComplexityResponse(BaseModel):
    """Response model for file complexity analysis."""

    file_path: str
    language: str
    total_loc: int
    average_complexity: float
    max_complexity: float
    total_entities: int
    complex_entities: int
    entities_not_computed: int = 0
    entity_metrics: list[ComplexityMetricsResponse]
    hotspots: list[str]
    refactoring_suggestions: list[str]
    quality_score: float | None = Field(
        ...,
        description="None when the file has entities but none could be measured",
    )
    metrics_computed: bool = Field(
        True,
        description="False when no entity in the file could be measured; the "
        "aggregate numbers are then not evidence of simple code",
    )


class ComplexityAnalysisResponse(BaseResponse):
    """Response from complexity analysis operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: FileComplexityResponse = Field(..., description="Complexity analysis results")


class BatchComplexityResponse(BaseResponse):
    """Response from batch complexity analysis."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    results: list[FileComplexityResponse] = Field(
        ..., description="Complexity analysis results for all files"
    )
    summary: dict[str, Any] = Field(..., description="Batch analysis summary")


class ComplexityCapabilitiesResponse(BaseModel):
    """Response model for complexity analysis capabilities."""

    supported_languages: list[str]
    complexity_levels: list[str] = Field(
        ..., description="Available complexity level classifications"
    )
    available_metrics: list[str] = Field(..., description="List of available complexity metrics")
    language_specific_features: dict[str, list[str]] = Field(
        ..., description="Language-specific analysis features"
    )


@router.post(
    "/analyze-file",
    response_model=ComplexityAnalysisResponse,
    operation_id="analyze_file_complexity",
    summary="📊 Container Complexity Analysis",
    description="""
    🎯 **Advanced Container Complexity Analysis** - Deep complexity assessment with language-specific metrics

    ## Purpose
    Performs comprehensive complexity analysis on files within a container, providing detailed metrics
    and actionable refactoring suggestions based on the specific programming language.

    ## How to Use
    **POST** `/api/analysis/complexity/analyze-file`

    **Container Analysis:**
    ```json
    {
        "container_name": "my-java-project",
        "file_paths": ["/app/data/repo/src/main/java/MyClass.java"],
        "language": "java",
        "include_suggestions": true
    }
    ```

    **Analyze All Files in Container:**
    ```json
    {
        "container_name": "my-project",
        "include_suggestions": true
    }
    ```

    ## Language-Specific Analysis

    ### 🔧 **Java Analysis**
    - Object-oriented complexity patterns
    - Inheritance and interface coupling
    - Exception handling complexity
    - Generics and annotation impact

    ### 🦀 **Rust Analysis**
    - Ownership and lifetime complexity
    - Unsafe code analysis
    - Trait composition patterns
    - Async/await complexity

    ### 🐹 **Go Analysis**
    - Interface implementation patterns
    - Goroutine and channel complexity
    - Error handling patterns
    - Pointer usage analysis

    ## Complexity Metrics

    ### 📈 **Core Metrics**
    - **Cyclomatic Complexity**: Decision point count
    - **Cognitive Complexity**: Human readability focused
    - **Maintainability Index**: Overall maintainability score
    - **Coupling Factor**: Dependency strength analysis

    ### 🎛️ **Detailed Metrics**
    - **Nesting Depth**: Control structure depth
    - **Parameter Count**: Method parameter analysis
    - **Halstead Metrics**: Code vocabulary complexity
    - **Lines of Code**: Size-based metrics

    ## Response Information
    Returns comprehensive analysis including:
    - Entity-level complexity breakdown
    - Complexity hotspot identification
    - Language-specific refactoring suggestions
    - Quality score assessment
    - Risk factor identification

    ## Use Cases
    - **Code Review**: Identify complex code requiring attention
    - **Refactoring Planning**: Prioritize refactoring efforts
    - **Quality Gates**: Enforce complexity thresholds in CI/CD
    - **Technical Debt**: Measure and track code quality
    """,
    responses={
        200: {"description": "Complexity analysis completed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid container name or parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {"model": Error404Response, "description": "Container not found"},
        422: {
            "model": Error422Response,
            "description": "File analysis failed",
        },
        500: {
            "model": Error500Response,
            "description": "Internal analysis error",
        },
    },
)
async def analyze_file_complexity(
    request: ComplexityAnalysisRequest = Body(..., description="Complexity analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> ComplexityAnalysisResponse:
    """Analyze complexity for files in a container."""

    try:
        # Resolve file paths from container
        file_paths = await _resolve_container_files(
            request.container_name, request.file_paths, tenant_id=tenant_id, repo=request.repo
        )

        if not file_paths:
            raise not_found("Code files", f"container '{request.container_name}'")

        # For single file analysis, just take the first file if multiple provided
        if request.file_paths and len(request.file_paths) == 1:
            target_file = file_paths[0]
        else:
            # If no specific files provided or multiple files, analyze the first found file
            target_file = file_paths[0]

        # Get processor factory and analyze file
        factory = get_multi_language_factory()

        # Get appropriate processor
        processor = await _get_processor_for_analysis(factory, target_file, request.language)

        # Read and parse file
        content = target_file.read_text(encoding="utf-8")
        parse_result = await _parse_file_content(processor, target_file, content)

        # Perform complexity analysis
        complexity_service = get_complexity_service()
        complexity_report = complexity_service.analyze_file_complexity(parse_result, content)

        # Convert to response model
        entity_metrics = [
            _convert_metrics_to_response(metrics, request.include_suggestions)
            for metrics in complexity_report.entity_metrics
        ]

        file_response = _create_file_complexity_response(
            complexity_report, entity_metrics, request.include_suggestions
        )

        logger.info(
            "Complexity analysis completed for %s in container %s: quality score %s",
            target_file.name,
            request.container_name,
            complexity_report.quality_score,
        )

        return ComplexityAnalysisResponse(
            message=f"Complexity analysis completed for {target_file.name} in container {request.container_name}",
            result=file_response,
            request_id=str(
                hash(f"complexity_{request.container_name}_{target_file.name}_{time.time()}")
            ),
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
        logger.exception(
            "Complexity analysis failed for container %s: %s", request.container_name, e
        )
        raise internal_error(f"Complexity analysis failed: {str(e)}") from e


@router.post(
    "/analyze-batch",
    response_model=BatchComplexityResponse,
    operation_id="analyze_batch_complexity",
    summary="📊 Batch Container Complexity Analysis",
    description="""
    🚀 **Batch Container Complexity Analysis** - Analyze multiple files in a container for complexity patterns and quality assessment

    ## Purpose
    Perform complexity analysis on multiple files simultaneously within a container, providing aggregated
    insights and comparative analysis across a codebase.

    ## How to Use
    **POST** `/api/analysis/complexity/analyze-batch`

    **Analyze All Files in Container:**
    ```json
    {
        "container_name": "my-java-project",
        "include_file_suggestions": true,
        "quality_threshold": 70.0
    }
    ```

    **Analyze Specific Files in Container:**
    ```json
    {
        "container_name": "my-java-project",
        "file_paths": [
            "/app/data/repo/src/main/java/service/UserService.java",
            "/app/data/repo/src/main/java/model/User.java"
        ],
        "include_file_suggestions": true,
        "quality_threshold": 70.0
    }
    ```

    ## Batch Analysis Features

    ### 📈 **Aggregated Metrics**
    - Overall codebase quality score
    - Complexity distribution analysis
    - Hotspot identification across files
    - Language-specific patterns

    ### 🎯 **Quality Assessment**
    - Files below quality threshold
    - Refactoring priority ranking
    - Technical debt estimation
    - Maintainability trends

    ### 📊 **Comparative Analysis**
    - Cross-file complexity comparison
    - Language-specific benchmarks
    - Entity complexity rankings
    - Coupling analysis

    ## Response Information
    Returns comprehensive batch analysis including:
    - Individual file complexity reports
    - Aggregated summary statistics
    - Quality threshold violations
    - Prioritized refactoring recommendations

    ## Use Cases
    - **Codebase Assessment**: Overall quality evaluation
    - **Refactoring Planning**: Identify worst files first
    - **Quality Monitoring**: Track complexity trends
    - **Technical Debt**: Quantify maintenance burden
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Container not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def analyze_batch_complexity(
    request: BatchComplexityRequest = Body(..., description="Batch complexity analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> BatchComplexityResponse:
    """Analyze complexity for multiple files in a container."""

    try:
        # Resolve file paths from container
        file_paths = await _resolve_container_files(
            request.container_name, request.file_paths, tenant_id=tenant_id, repo=request.repo
        )

        if not file_paths:
            raise not_found("Code files", f"container '{request.container_name}'")

        # Initialize services
        FileService()
        factory = get_multi_language_factory()
        complexity_service = get_complexity_service()

        results: list[FileComplexityResponse] = []
        summary_stats = _initialize_batch_summary(len(file_paths))

        for file_path in file_paths:
            file_response = await _process_single_file_for_batch(
                str(file_path),
                factory,
                complexity_service,
                request.include_file_suggestions,
            )

            if file_response:
                results.append(file_response)
                _update_batch_statistics(file_response, request.quality_threshold, summary_stats)
            else:
                summary_stats["failed_files"] += 1

        # Finalize summary
        analyzed_files = summary_stats["analyzed_files"]
        scored_files = summary_stats["scored_files"]
        average_quality: float | None = None
        if scored_files > 0:
            average_quality = summary_stats["total_quality"] / scored_files

        # Update final summary values
        summary_stats.update(
            {
                "average_quality": average_quality,
                "languages": list(summary_stats["languages"]),
                "top_hotspots": summary_stats["all_hotspots"][:10],  # Top 10 hotspots
            }
        )

        # Remove internal fields
        summary_stats.pop("total_quality", None)
        summary_stats.pop("all_hotspots", None)

        logger.info(
            "Batch complexity analysis completed for container %s: %d files analyzed",
            request.container_name,
            analyzed_files,
        )

        return BatchComplexityResponse(
            message=f"Batch complexity analysis completed for container {request.container_name}: {analyzed_files} files analyzed",
            results=results,
            summary=summary_stats,
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
        logger.exception(
            "Batch complexity analysis failed for container %s: %s",
            request.container_name,
            e,
        )
        raise internal_error(f"Batch complexity analysis failed: {str(e)}") from e


@router.get(
    "/capabilities",
    response_model=ComplexityCapabilitiesResponse,
    operation_id="get_complexity_capabilities",
    summary="📋 Complexity Analysis Capabilities",
    description="""
    🔍 **Complexity Analysis Capabilities** - Discover available complexity analysis features

    ## Purpose
    Provides information about the complexity analysis engine's capabilities,
    including supported languages, available metrics, and language-specific features.

    ## Response Information
    - **Supported Languages**: Languages with specific complexity analyzers
    - **Complexity Levels**: Available complexity classifications
    - **Available Metrics**: All complexity metrics that can be calculated
    - **Language Features**: Language-specific analysis capabilities
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def get_complexity_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> ComplexityCapabilitiesResponse:
    """Get complexity analysis system capabilities."""

    try:
        complexity_service = get_complexity_service()

        capabilities = ComplexityCapabilitiesResponse(
            supported_languages=complexity_service.get_supported_languages(),
            complexity_levels=[level.value for level in ComplexityLevel],
            available_metrics=[
                "cyclomatic_complexity",
                "cognitive_complexity",
                "lines_of_code",
                "nesting_depth",
                "parameter_count",
                "dependency_count",
                "halstead_difficulty",
                "maintainability_index",
                "coupling_factor",
            ],
            language_specific_features={
                "java": [
                    "Inheritance coupling analysis",
                    "Interface implementation tracking",
                    "Exception handling complexity",
                    "Generics complexity assessment",
                    "Annotation impact analysis",
                ],
                "rust": [
                    "Ownership complexity analysis",
                    "Lifetime parameter tracking",
                    "Unsafe code detection",
                    "Trait composition analysis",
                    "Async/await complexity",
                ],
                "go": [
                    "Interface implementation patterns",
                    "Goroutine complexity analysis",
                    "Channel usage patterns",
                    "Error handling assessment",
                    "Pointer usage analysis",
                ],
            },
        )

        return capabilities

    except Exception as e:
        logger.error("Error retrieving complexity capabilities: %s", e)
        raise internal_error(f"Error retrieving capabilities: {str(e)}") from e
