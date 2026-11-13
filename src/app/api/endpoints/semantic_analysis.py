"""

Semantic Analysis API Endpoints

Advanced semantic analysis endpoints for detecting cross-language relationships
and architectural patterns.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid7

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.responses import (
    bad_request,
    internal_error,
)
from auth.tenant import require_tenant
from processing.semantic_analysis import SemanticRelationType
from processing.workspace_repos import RepoSelector
from services.semantic_analysis_service import (
    get_semantic_analysis_service,
)

from ..models.common_api_models import BaseResponse, ResponseStatus
from ..models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from ..models.semantic_analysis_api_models import (
    SemanticCapabilitiesResponse,
    SemanticPatternsResponse,
)
from ..utils.container_validation import validate_and_get_container_for_tenant
from ..utils.workspace_files import (
    MaterializedFiles,
    materialize_tenant_workspace,
    materialized_files,
    release_materialized,
)

logger = logging.getLogger(__name__)

router = APIRouter()


_SEMANTIC_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".java",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
)
_SEMANTIC_MAX_FILES = 50


async def _resolve_container_files(
    container_name: str,
    tenant_id: str,
    file_paths: list[str] | None = None,
    *,
    repo: str | None = None,
) -> MaterializedFiles:
    """Resolve a container's code files to readable LOCAL paths.

    The workspace (``codebase_path``, e.g. ``/workspace``) exists only inside the
    tenant's workspace pod. The old implementation ``os.path.exists``-checked it
    on THIS pod and 404'd "Container codebase path '/workspace' not found" for
    every real workspace (prod 2026-10-11, cv-ems-p4), with a hard-coded
    fallback to an unrelated local repo for one container name. Now: exact,
    tenant-scoped lookup and a materialised copy of the live tree; no fallback.

    The caller must ``.release()`` the result when done.
    """
    container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
    if file_paths:
        root = await materialize_tenant_workspace(
            container_name,
            tenant_id,
            rel_paths=file_paths,
            container_info=container_info,
            repo=repo,
        )
    else:
        root = await materialize_tenant_workspace(
            container_name,
            tenant_id,
            name_patterns=tuple(f"*{ext}" for ext in _SEMANTIC_EXTENSIONS),
            container_info=container_info,
            repo=repo,
        )
    paths = [Path(p) for p in materialized_files(root)]
    if not file_paths and len(paths) > _SEMANTIC_MAX_FILES:
        logger.warning("Found %d files, limiting to first %d", len(paths), _SEMANTIC_MAX_FILES)
        paths = paths[:_SEMANTIC_MAX_FILES]
    logger.info("Resolved %d files in container '%s'", len(paths), container_name)
    return MaterializedFiles(paths, materialized_root=root)


def _validate_direct_file_paths(file_paths: list[str]) -> list[Path]:
    """Validate and convert direct file paths to Path objects."""
    resolved_paths: list[Path] = []
    for file_path_str in file_paths:
        path_obj = Path(file_path_str)
        if not path_obj.exists():
            raise bad_request(f"File not found: {file_path_str}")
        resolved_paths.append(path_obj)
    return resolved_paths


class SemanticAnalysisRequest(RepoSelector):
    """Request for semantic analysis of a codebase."""

    container_name: str = Field(
        ...,
        description="Container name to analyze (required for container scoping)",
    )
    file_paths: list[str] | None = Field(
        None,
        description="Specific file paths within container to analyze (optional)",
    )
    analysis_type: str | None = Field(
        "full",
        description="Type of analysis to perform: 'full' or 'basic'",
    )
    analysis_options: dict[str, Any] | None = Field(
        None, description="Analysis configuration options"
    )
    include_insights: bool = Field(True, description="Include semantic insights in response")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "multi-lang-project",
                "file_paths": [
                    "main.rs",
                    "lib.c",
                    "api.java",
                ],
                "analysis_type": "full",
                "analysis_options": {
                    "detect_ffi": True,
                    "detect_api_calls": True,
                    "detect_data_flow": True,
                },
                "include_insights": True,
            },
        }
    )


class CrossLanguageRelationship(BaseModel):
    """Cross-language relationship information."""

    source_language: str
    target_language: str
    relationship_type: str
    source_entity: str
    target_entity: str
    confidence: float = 1.0


class SemanticPattern(BaseModel):
    """Semantic pattern detection result."""

    pattern: str
    count: int


class ComplexityHotspot(BaseModel):
    """Entity with high relationship complexity."""

    entity_name: str
    entity_type: str
    language: str
    relationship_count: int
    file_path: str


class CouplingPair(BaseModel):
    """Language coupling information."""

    languages: list[str]
    coupling_strength: int


class SemanticInsights(BaseModel):
    """Semantic analysis insights."""

    language_distribution: dict[str, dict[str, Any]]
    cross_language_interactions: list[CrossLanguageRelationship]
    architectural_patterns: list[SemanticPattern]
    complexity_hotspots: list[ComplexityHotspot]
    coupling_analysis: dict[str, list[CouplingPair]]


class PRValidationSemanticAnalysisResult(BaseModel):
    """Semantic analysis result."""

    analysis_summary: dict[str, Any]
    semantic_insights: SemanticInsights | None = None
    entities_count: int
    basic_relationships_count: int
    semantic_relationships_count: int
    execution_time: float
    entities: list[dict[str, Any]] = Field(
        default=[], description="Extracted entities (classes, functions, variables)"
    )
    basic_relationships: list[dict[str, Any]] = Field(
        default=[], description="Basic relationships (calls, imports)"
    )
    semantic_relationships: dict[str, list[dict[str, Any]]] = Field(
        default={}, description="Semantic relationships by type"
    )


class SemanticAnalysisResponse(BaseResponse):
    """Response from semantic analysis operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: PRValidationSemanticAnalysisResult = Field(..., description="Semantic analysis results")


def _build_semantic_insights(insights_data: dict[str, Any]) -> SemanticInsights:
    """Build semantic insights from analysis data."""
    return SemanticInsights(
        language_distribution=insights_data["language_distribution"],
        cross_language_interactions=[
            CrossLanguageRelationship(**interaction)
            for interaction in (insights_data.get("cross_language_interactions") or [])
            if isinstance(interaction, dict)
        ],
        architectural_patterns=[
            SemanticPattern(**pattern)
            for pattern in (insights_data.get("architectural_patterns") or [])
            if isinstance(pattern, dict)
        ],
        complexity_hotspots=[
            ComplexityHotspot(**hotspot)
            for hotspot in (insights_data.get("complexity_hotspots") or [])
            if isinstance(hotspot, dict)
        ],
        coupling_analysis=insights_data["coupling_analysis"],
    )


@router.post(
    "/analyze-codebase",
    response_model=SemanticAnalysisResponse,
    operation_id="analyze_codebase_semantics",
    summary="🧠 Advanced Semantic Analysis",
    description="""
    🚀 **Advanced Semantic Codebase Analysis** - Deep relationship detection across multiple languages

    ## Purpose
    This endpoint performs sophisticated semantic analysis on multi-language codebases to detect:
    - Cross-language relationships (FFI calls, API interactions)
    - Architectural patterns (microservices, event-driven design)
    - Data flow and state mutations
    - Type system relationships and polymorphic usage

    ## How to Use
    **POST** `/api/analysis/semantic/analyze-codebase`

    **Container-based Analysis (Required):**

    All semantic analysis must be scoped to a container. The container_name is required
    to provide proper context for analyzing cross-language relationships.

    ```json
    {
        "container_name": "multi-lang-project",
        "analysis_options": {
            "detect_ffi": true,
            "detect_api_calls": true,
            "detect_data_flow": true
        },
        "include_insights": true
    }
    ```

    **Specific Files in Container:**
    ```json
    {
        "container_name": "multi-lang-project",
        "file_paths": [
            "rust_module.rs",
            "c_library.c",
            "java_service.java",
            "js_frontend.js"
        ],
        "analysis_options": {
            "detect_ffi": true,
            "detect_api_calls": true,
            "detect_data_flow": true
        },
        "include_insights": true
    }
    ```

    ## Analysis Capabilities

    ### 🔗 **Cross-Language Relationships**
    - **FFI Calls**: Rust ↔ C, Java ↔ C++, Go ↔ C
    - **API Endpoints**: REST, GraphQL, RPC calls
    - **Service Communications**: Microservice interactions
    - **Database Access**: ORM and direct database operations

    ### 🏗️ **Architectural Patterns**
    - **Service-Oriented**: Microservices, service calls
    - **Event-Driven**: Event emission and consumption
    - **Data Flow**: Variable flow and state mutations
    - **Resource Management**: Memory sharing, lock dependencies

    ### 🎯 **Type System Analysis**
    - **Generic Constraints**: Type parameter bounds
    - **Polymorphic Calls**: Virtual method dispatch
    - **Type Conversions**: Casting and type transformations
    - **Interface Bindings**: Interface implementations

    ### 📊 **Semantic Insights**
    - **Language Distribution**: Entity counts per language
    - **Coupling Analysis**: Cross-language dependency strength
    - **Complexity Hotspots**: Highly connected entities
    - **Architectural Overview**: Pattern usage statistics

    ## Response Format
    Returns comprehensive analysis with:
    - Entity and relationship counts
    - Cross-language interaction mappings
    - Architectural pattern detection
    - Complexity and coupling metrics
    - Execution performance data

    ## Use Cases
    - **Architecture Review**: Understand cross-language dependencies
    - **Refactoring Planning**: Identify tightly coupled components
    - **Security Audit**: Find FFI and external interface usage
    - **Performance Analysis**: Detect data flow bottlenecks
    """,
    responses={
        200: {"description": "Semantic analysis completed successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Container not found"},
        422: {"description": "File analysis errors"},
        500: {"description": "Internal analysis error"},
    },
)
async def analyze_codebase_semantics(
    request: SemanticAnalysisRequest = Body(..., description="Semantic analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> SemanticAnalysisResponse:
    """Perform advanced semantic analysis on a multi-language codebase."""

    file_paths: list[Path] | None = None
    try:
        start_time = time.time()
        analysis_id = str(uuid7())

        # Resolve (exact, tenant-scoped -- 404/403 inside) and materialise the
        # live workspace tree.
        file_paths = await _resolve_container_files(
            request.container_name, tenant_id, request.file_paths, repo=request.repo
        )

        # Return empty analysis if no supported files found
        if not file_paths:
            return SemanticAnalysisResponse(
                message=f"No supported files found in container '{request.container_name}'",
                result=PRValidationSemanticAnalysisResult(
                    analysis_summary={
                        "status": "no_supported_files",
                        "container": request.container_name,
                    },
                    entities_count=0,
                    basic_relationships_count=0,
                    semantic_relationships_count=0,
                    execution_time=time.time() - start_time,
                ),
                request_id=analysis_id,
            )

        # Get semantic analysis service
        service = get_semantic_analysis_service()

        # Perform semantic analysis
        logger.info("Starting semantic analysis of %d files", len(file_paths))
        analysis_results = await service.analyze_codebase_semantics(file_paths, analysis_id)

        execution_time = time.time() - start_time

        # Build response with actual data (not just counts)
        result = PRValidationSemanticAnalysisResult(
            analysis_summary=analysis_results["analysis_summary"],
            entities_count=len(analysis_results["entities"]),
            basic_relationships_count=len(analysis_results["basic_relationships"]),
            semantic_relationships_count=len(analysis_results["semantic_relationships"]),
            execution_time=execution_time,
            entities=analysis_results.get("entities", []),
            basic_relationships=analysis_results.get("basic_relationships", []),
            semantic_relationships=analysis_results.get("semantic_relationships", {}),
        )

        # Add semantic insights if requested and not basic analysis
        if request.include_insights and request.analysis_type != "basic":
            insights_data = await service.get_semantic_insights(analysis_results)
            result.semantic_insights = _build_semantic_insights(insights_data)

        logger.info(
            "Semantic analysis completed in %.2fs: %s",
            execution_time,
            result.analysis_summary,
        )

        return SemanticAnalysisResponse(
            message=f"Semantic analysis completed for {len(file_paths)} files",
            result=result,
            request_id=analysis_id,
        )

    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        logger.exception("Semantic analysis failed: %s", e)
        raise internal_error(f"Semantic analysis failed: {str(e)}") from e
    finally:
        release_materialized(file_paths)


@router.get(
    "/capabilities",
    response_model=SemanticCapabilitiesResponse,
    operation_id="get_semantic_capabilities",
    summary="📋 Semantic Analysis Capabilities",
    description="""
    🔍 **Discover Semantic Analysis Capabilities** - Learn what advanced analysis features are available

    ## Purpose
    This endpoint provides information about the semantic analysis engine's capabilities,
    including supported relationship types, analysis patterns, and cross-language features.

    ## Response Overview
    Returns comprehensive information about:
    - Supported semantic relationship types
    - Cross-language analysis patterns
    - Architectural pattern detection
    - Type system analysis features
    - Performance characteristics
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
async def get_semantic_analysis_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> SemanticCapabilitiesResponse:
    """Get semantic analysis system capabilities."""

    try:
        semantic_relationship_types = [
            {
                "type": rel_type.value,
                "description": _get_relationship_description(rel_type),
            }
            for rel_type in SemanticRelationType
        ]

        cross_language_patterns = [
            "rust_c_ffi",
            "java_jni",
            "go_cgo",
            "rest_api_calls",
            "database_access",
            "type_conversions",
        ]

        analysis_features = {
            "ffi_detection": True,
            "api_endpoint_mapping": True,
            "service_call_analysis": True,
            "data_flow_tracking": True,
            "type_system_analysis": True,
            "architectural_patterns": True,
            "memory_sharing_detection": True,
            "event_driven_patterns": True,
            "polymorphic_analysis": True,
            "cross_language_coupling": True,
        }

        supported_languages = [
            "java",
            "rust",
            "go",
            "csharp",
            "cpp",
            "c",
            "python",
            "javascript",
            "typescript",
        ]

        performance_characteristics = {
            "typical_analysis_time": "2-5 seconds per 100 files",
            "memory_usage": "< 100MB for typical codebases",
            "scalability": "Tested on codebases up to 10,000 files",
        }

        return SemanticCapabilitiesResponse(
            message="Semantic analysis capabilities retrieved successfully",
            semantic_relationship_types=semantic_relationship_types,
            cross_language_patterns=cross_language_patterns,
            analysis_features=analysis_features,
            supported_languages=supported_languages,
            performance_characteristics=performance_characteristics,
        )

    except Exception as e:
        logger.error("Error retrieving semantic analysis capabilities: %s", e)
        raise internal_error(f"Error retrieving capabilities: {str(e)}") from e


@router.get(
    "/patterns",
    response_model=SemanticPatternsResponse,
    operation_id="get_semantic_patterns",
    summary="🔍 Semantic Patterns",
    description="""
    📊 **Available Semantic Analysis Patterns** - Browse detection patterns and their descriptions

    Returns detailed information about all available semantic analysis patterns,
    including cross-language relationship patterns, architectural patterns,
    and type system analysis patterns.
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
async def get_semantic_patterns(
    pattern_type: str | None = Query(
        None,
        description="Filter by pattern type: cross_language, architectural, type_system",
    ),
    tenant_id: str = Depends(require_tenant),
) -> SemanticPatternsResponse:
    """Get information about available semantic analysis patterns."""

    try:
        cross_language_patterns = [
            {
                "name": "rust_c_ffi",
                "description": "Rust calling C functions via FFI",
                "source_languages": ["rust"],
                "target_languages": ["c", "cpp"],
                "detection_method": "extern block and unsafe call analysis",
            },
            {
                "name": "java_jni",
                "description": "Java Native Interface calls",
                "source_languages": ["java"],
                "target_languages": ["c", "cpp"],
                "detection_method": "native method declarations and JNI signatures",
            },
            {
                "name": "go_cgo",
                "description": "Go calling C code via CGO",
                "source_languages": ["go"],
                "target_languages": ["c", "cpp"],
                "detection_method": "CGO comments and C function calls",
            },
            {
                "name": "rest_api_calls",
                "description": "REST API endpoint interactions",
                "source_languages": ["*"],
                "target_languages": ["*"],
                "detection_method": "HTTP client calls and endpoint annotations",
            },
        ]

        architectural_patterns = [
            {
                "name": "service_oriented",
                "description": "Microservice and service call patterns",
                "detection_indicators": [
                    "@Service annotations",
                    "@RestController patterns",
                    "Service injection patterns",
                ],
            },
            {
                "name": "event_driven",
                "description": "Event emission and consumption patterns",
                "detection_indicators": [
                    "Event emitter/listener patterns",
                    "Message queue interactions",
                    "Observer pattern usage",
                ],
            },
            {
                "name": "data_access",
                "description": "Database and persistence patterns",
                "detection_indicators": [
                    "ORM entity annotations",
                    "SQL query patterns",
                    "Database connection patterns",
                ],
            },
        ]

        type_system_patterns = [
            {
                "name": "generic_constraints",
                "description": "Generic type parameter bounds and constraints",
                "languages": ["java", "csharp", "rust", "cpp"],
            },
            {
                "name": "polymorphic_calls",
                "description": "Virtual method dispatch and interface calls",
                "languages": ["java", "csharp", "cpp", "python"],
            },
            {
                "name": "type_conversions",
                "description": "Type casting and conversion patterns",
                "languages": ["*"],
            },
        ]

        # Filter by pattern type if specified
        if pattern_type is not None:
            # Validate that pattern_type is not empty
            if not pattern_type.strip():
                raise bad_request("pattern_type cannot be empty string")

            if pattern_type == "cross_language":
                return SemanticPatternsResponse(
                    message="Cross-language patterns retrieved successfully",
                    cross_language_patterns=cross_language_patterns,
                    architectural_patterns=[],
                    type_system_patterns=[],
                )
            elif pattern_type == "architectural":
                return SemanticPatternsResponse(
                    message="Architectural patterns retrieved successfully",
                    cross_language_patterns=[],
                    architectural_patterns=architectural_patterns,
                    type_system_patterns=[],
                )
            elif pattern_type == "type_system":
                return SemanticPatternsResponse(
                    message="Type system patterns retrieved successfully",
                    cross_language_patterns=[],
                    architectural_patterns=[],
                    type_system_patterns=type_system_patterns,
                )
            else:
                raise bad_request(
                    f"Unknown pattern type: {pattern_type}. Available: cross_language, architectural, type_system"
                )

        return SemanticPatternsResponse(
            message="All semantic patterns retrieved successfully",
            cross_language_patterns=cross_language_patterns,
            architectural_patterns=architectural_patterns,
            type_system_patterns=type_system_patterns,
        )

    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        logger.error("Error retrieving semantic patterns: %s", e)
        raise internal_error(f"Error retrieving patterns: {str(e)}") from e


def _get_relationship_description(rel_type) -> str:
    """Get human-readable description for a relationship type."""
    descriptions = {
        "ffi_call": "Foreign function interface call between languages",
        "api_endpoint": "REST or GraphQL API endpoint interaction",
        "rpc_call": "Remote procedure call",
        "data_dependency": "Data flows from source to target",
        "state_mutation": "Source modifies state of target",
        "event_emission": "Source emits events consumed by target",
        "service_call": "Microservice or service method call",
        "database_access": "Database operation or query",
        "configuration_use": "Configuration parameter usage",
        "type_constraint": "Generic type parameter constraint",
        "type_conversion": "Type casting or conversion",
        "type_alias_resolution": "Type alias usage and resolution",
        "polymorphic_call": "Virtual method call or interface dispatch",
        "interface_binding": "Interface implementation binding",
        "mixin_composition": "Mixin or trait composition",
        "higher_order_usage": "Higher-order function usage",
        "closure_capture": "Variable capture in closure",
        "monad_binding": "Monadic operation binding",
        "resource_acquisition": "Resource management operation",
        "memory_sharing": "Shared memory or reference access",
        "lock_dependency": "Synchronization dependency",
    }

    return descriptions.get(rel_type.value, f"Advanced relationship: {rel_type.value}")
