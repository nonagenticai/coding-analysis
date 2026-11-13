"""
Shared Models for Visualization API

Common Pydantic models and type definitions used across visualization modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...models.common_api_models import BaseResponse, ResponseStatus


class ChartType(StrEnum):
    """Supported chart types for visualization."""

    COMPLEXITY_HEATMAP = "complexity_heatmap"
    DEPENDENCY_MATRIX = "dependency_matrix"
    LANGUAGE_DISTRIBUTION = "language_distribution"
    CALL_HIERARCHY = "call_hierarchy"
    ARCHITECTURAL_LAYERS = "architectural_layers"


class ExportFormat(StrEnum):
    """Supported export formats for visualizations."""

    JSON = "json"
    CYTOSCAPE = "cytoscape"
    D3 = "d3"
    GRAPHVIZ = "graphviz"
    DOT = "dot"
    MERMAID = "mermaid"
    VIS = "vis"
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    GRAPHML = "graphml"


class VisualizationRequest(BaseModel):
    """Request for creating a dependency visualization."""

    container_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Workspace (container) whose files are visualised. Files are read through "
            "coding-containers under the caller's tenant; this pod's filesystem is never read."
        ),
    )
    file_paths: list[str] = Field(
        ...,
        max_length=200,
        description=(
            "Repo-relative paths inside the workspace (e.g. 'src/main.py'). Absolute paths "
            "outside the workspace root and any '..' segment are refused."
        ),
    )
    visualization_type: str = Field(
        "dependency_graph", description="Type of visualization to create"
    )
    options: dict[str, Any] | None = Field(None, description="Visualization options and filters")
    export_formats: list[str] = Field(
        ["cytoscape", "json"], description="Export formats to generate"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-workspace",
                "file_paths": [
                    "src/main.rs",
                    "src/api/Api.java",
                    "web/client.ts",
                ],
                "visualization_type": "dependency_graph",
                "options": {
                    "languages": ["rust", "java", "typescript"],
                    "group_by_language": True,
                    "min_complexity": 0,
                    "layout": {"algorithm": "force-directed", "spacing": 100},
                },
                "export_formats": ["cytoscape", "d3", "json"],
            },
        }
    )


class DependencyVisualizerServiceVisualizationNode(BaseModel):
    """Visualization node information."""

    id: str
    label: str
    type: str
    language: str
    file_path: str | None = None
    size: int = 1
    color: str | None = None
    shape: str = "box"
    group: str | None = None
    complexity: float = 0.0
    importance: float = 0.0
    properties: dict[str, Any] = {}


class DependencyVisualizerServiceVisualizationEdge(BaseModel):
    """Visualization edge information."""

    id: str
    source: str
    target: str
    type: str
    label: str | None = None
    weight: float = 1.0
    color: str | None = None
    style: str = "solid"
    properties: dict[str, Any] = {}


# Type aliases for backwards compatibility
VisualizationNode = DependencyVisualizerServiceVisualizationNode
VisualizationEdge = DependencyVisualizerServiceVisualizationEdge


class VisualizationResult(BaseModel):
    """Visualization analysis result."""

    id: str
    title: str
    type: str
    nodes: list[VisualizationNode]
    edges: list[VisualizationEdge]
    languages: list[str]
    statistics: dict[str, Any]
    layout_options: dict[str, Any]
    exports: dict[str, str]


class VisualizationResponse(BaseResponse):
    """Response from visualization operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: VisualizationResult = Field(..., description="Visualization result")


class VisualizationCapabilitiesResponse(BaseModel):
    """Response model for visualization capabilities."""

    visualization_types: list[str]
    node_types: list[str]
    edge_types: list[str]
    export_formats: list[str]
    layout_algorithms: list[str]
    supported_languages: list[str]
