"""Pydantic models for code graph representations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Request Models


class CodebaseInitRequest(BaseModel):
    """Request model for initializing a new Code Graph for a codebase."""

    container_name: str = Field(
        ...,
        min_length=1,
        description="Container name from which to initialize the graph",
        pattern=r"^[a-zA-Z][a-zA-Z0-9\-_]{0,62}$",
        json_schema_extra={"example": "codegraph-my-project"},
    )
    codebase_path: str | None = Field(
        None,
        min_length=1,
        description="Path to the codebase directory (internal to the container). If not provided, will be automatically resolved from the container configuration.",
        pattern=r"^/[a-zA-Z0-9/_\-\.]+$",
        json_schema_extra={"example": "/app/repositories/my-project"},
    )
    include_tests: bool | None = Field(
        None,
        description="Whether to include test files in the graph analysis. If not specified, uses the default system setting.",
    )
    exclude_patterns: list[str] | None = Field(
        None,
        description="Additional patterns to exclude from analysis (e.g., ['**/legacy/**', 'temp_*']). These are added to the default ignore patterns.",
        json_schema_extra={"example": ["**/node_modules/**", "**/dist/**"]},
    )
    include_patterns: list[str] | None = Field(
        None,
        description="Specific patterns to include even if they would normally be excluded (e.g., ['important_test_*.py']). These override exclude patterns.",
        json_schema_extra={"example": ["**/important_*.py"]},
    )
    force_rebuild: bool | None = Field(
        False,
        description="Force rebuild of the graph even if it already exists. This will delete the existing database and create a new one.",
    )

    @field_validator("codebase_path")
    @classmethod
    def validate_codebase_path(cls, v: str | None) -> str | None:
        """Validate codebase path is safe and not a system directory."""
        if v is None:
            return v

        if not v or not v.strip():
            raise ValueError("Codebase path cannot be empty")

        v = v.strip()

        # Prevent dangerous paths
        dangerous_paths = [
            "/",
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/var",
            "/sys",
            "/proc",
            "/dev",
            "/boot",
            "/root",
        ]
        if v in dangerous_paths or any(v.startswith(f"{p}/") for p in dangerous_paths[:5]):
            raise ValueError(f"Invalid codebase path: {v}. Cannot analyze system directories.")

        # Ensure path doesn't contain suspicious patterns
        if ".." in v or "~" in v:
            raise ValueError("Codebase path cannot contain '..' or '~'")

        return v


class CodeGraphAnalysisRequest(BaseModel):
    """Request model for triggering analysis on a Code Graph."""

    analysis_depth: str = Field(
        "full",
        description="Depth of analysis to perform",
        examples=["basic_structure", "python_ast", "full"],
    )

    @field_validator("analysis_depth")
    @classmethod
    def validate_analysis_depth(cls, v: str) -> str:
        """Validate analysis depth value."""
        valid_depths = ["basic_structure", "python_ast", "full"]
        if v not in valid_depths:
            raise ValueError(f"analysis_depth must be one of: {valid_depths}")
        return v


class PatternSearchRequest(BaseModel):
    """Request model for searching code patterns."""

    pattern: str = Field(..., description="Regular expression or AST pattern to search for")
    language: str | None = Field(None, description="Programming language for AST-based search")
    file_paths: list[str] | None = Field(
        None,
        description="Specific files to search (if not provided, searches all files)",
    )


class EntityQueryRequest(BaseModel):
    """Request model for querying entities in the Code Graph."""

    entity_type: str = Field(..., description="Type of entity to query (e.g., 'function', 'class')")
    name_pattern: str | None = Field(None, description="Optional pattern to match entity names")
    limit: int = Field(100, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class RelatedNodesRequest(BaseModel):
    """Request model for retrieving nodes related to a specific node."""

    node_id: str = Field(..., description="ID of the node to find related nodes for")
    relationship_type: str | None = Field(None, description="Optional filter by relationship type")
    direction: str = Field("all", description="'outgoing', 'incoming', or 'all'")


class RawQueryRequest(BaseModel):
    """Request model for executing a raw SQL query against the Code Graph."""

    sql_query: str = Field(..., description="SQL query to execute (use with caution)")


class NodeQuery(BaseModel):
    """Model for querying nodes in the knowledge graph."""

    type: str | None = Field(None, description="Filter by node type")
    properties: dict[str, Any] | None = Field(None, description="Filter by node properties")
    text_search: str | None = Field(None, description="Full-text search in properties")
    with_centrality: bool | None = Field(False, description="Include centrality metrics")
    with_incoming: bool | None = Field(False, description="Include incoming edges count")
    with_outgoing: bool | None = Field(False, description="Include outgoing edges count")
    sort_by: str | None = Field(None, description="Property to sort by")
    sort_direction: str | None = Field("desc", description="Sort direction")
    limit: int | None = Field(100, description="Limit for pagination", gt=0, le=10000)
    offset: int | None = Field(0, description="Offset for pagination", ge=0)

    @field_validator("sort_direction")
    @classmethod
    def validate_sort_direction(cls, v: str | None) -> str:
        """Validate sort direction."""
        if v and v.lower() not in ["asc", "desc"]:
            raise ValueError("sort_direction must be 'asc' or 'desc'")
        return v.lower() if v else "desc"

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str | None) -> str | None:
        """Validate node type is not empty string."""
        if v is not None and not v.strip():
            raise ValueError("Node type cannot be empty string")
        return v.strip() if v else None


# Response Models


class NodeResponse(BaseModel):
    """Response model for a node in the Code Graph."""

    id: str
    type: str
    properties: dict[str, Any]
    centrality: dict[str, float] | None = None  # Centrality metrics (e.g., pagerank, degree)
    incoming_edges: int | None = None  # Count of incoming edges
    outgoing_edges: int | None = None  # Count of outgoing edges

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "node-123",
                "type": "function",
                "properties": {"name": "process_data", "complexity": 5},
                "centrality": {"pagerank": 0.23, "degree": 0.15},
            }
        }
    )


class EdgeResponse(BaseModel):
    """Response model for an edge in the Code Graph."""

    source: str
    target: str
    relationship: str
    properties: dict[str, Any]
    direction: str | None = None


class InitResponse(BaseModel):
    """Response model for Code Graph initialization."""

    message: str
    container_name: str
    graph_db_path: str
    nodes_created: int
    edges_created: int
    status: str


class CodeGraphAnalysisResponse(BaseModel):
    """Response model for Code Graph analysis triggering."""

    code_graph_id: str
    status: str
    analysis_depth: str


class PatternSearchResponse(BaseModel):
    """Response model for pattern search results."""

    matches: list[dict[str, Any]]
    count: int
    search_time_ms: float | None = None
    code_graph_id: str | None = None
    error: str | None = None


class EntityQueryResponse(BaseModel):
    """Response model for entity query results."""

    entities: list[NodeResponse]
    total: int
    limit: int
    offset: int


class RelatedNodesResponse(BaseModel):
    """Response model for related nodes query."""

    nodes: list[NodeResponse]
    edges: list[EdgeResponse]


class RawQueryResponse(BaseModel):
    """Response model for raw SQL query results."""

    results: list[dict[str, Any]]
    row_count: int


# Architecture Extension Models


class ArchitectureNodeResponse(NodeResponse):
    """Extended node response with architecture data."""

    architecture_data: dict[str, Any] | None = None
    hotspot_score: float | None = None
    quality_issues: list[dict[str, Any]] | None = None
    dependencies: list[str] | None = None
    component_id: str | None = None
    layer: str | None = None


class RuntimeEdgeResponse(EdgeResponse):
    """Extended edge response with runtime data."""

    runtime_data: dict[str, Any] | None = None
    call_count: int | None = None
    avg_duration_ms: float | None = None
    error_rate: float | None = None
    confidence: float | None = None


class ArchitectureCodeGraphResponse(BaseModel):
    """Response model for architecture code graph."""

    nodes: list[ArchitectureNodeResponse]
    edges: list[RuntimeEdgeResponse]
    architecture_metadata: dict[str, Any] | None = None
    quality_summary: dict[str, Any] | None = None
    dependency_summary: dict[str, Any] | None = None


class RuntimeGraphResponse(BaseModel):
    """Response model for runtime code graph."""

    nodes: list[NodeResponse]
    edges: list[RuntimeEdgeResponse]
    runtime_metadata: dict[str, Any] | None = None
    trace_summary: dict[str, Any] | None = None


class CodeGraphIntegrationRequest(BaseModel):
    """Request model for integrating architecture data with code graph."""

    include_architecture: bool = Field(True, description="Include architecture analysis data")
    include_dependencies: bool = Field(True, description="Include dependency analysis data")
    include_quality: bool = Field(True, description="Include quality analysis data")
    include_runtime: bool = Field(False, description="Include runtime trace data")
    quality_threshold: float | None = Field(None, description="Minimum quality score to include")
    hotspot_threshold: float | None = Field(None, description="Minimum hotspot score to include")


# Rebuild all models to resolve forward references
CodebaseInitRequest.model_rebuild()
CodeGraphAnalysisRequest.model_rebuild()
PatternSearchRequest.model_rebuild()
EntityQueryRequest.model_rebuild()
RelatedNodesRequest.model_rebuild()
RawQueryRequest.model_rebuild()
CodeGraphIntegrationRequest.model_rebuild()
InitResponse.model_rebuild()
