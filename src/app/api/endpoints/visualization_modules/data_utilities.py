"""
Data Utilities for Visualization

Data processing, validation, and transformation utilities for visualization operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException

from app.core.dependencies import get_multi_language_factory
from app.core.file_service import FileService
from processing.languages.base import LanguageProcessor
from services.dependency_visualizer import VisualizationType

from ...utils.workspace_files import (
    InvalidWorkspacePathError,
    invalid_workspace_path_http,
    normalize_workspace_path,
)
from .shared_models import (
    DependencyVisualizerServiceVisualizationEdge,
    DependencyVisualizerServiceVisualizationNode,
    VisualizationResult,
)

logger = logging.getLogger(__name__)


def validate_visualization_type(visualization_type: str) -> VisualizationType:
    """Validate and return the visualization type."""
    try:
        return VisualizationType(visualization_type)
    except ValueError as exc:
        valid_types = [t.value for t in VisualizationType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid visualization type. Valid types: {', '.join(valid_types)}",
        ) from exc


def validate_file_paths(file_paths: list[str], root: str = "/workspace") -> list[str]:
    """Confine requested paths to the workspace and return them repo-relative.

    This used to check only ``Path(p).exists()`` -- on THIS pod -- and then
    parse whatever it found, so any file readable by the analysis service
    (``/etc/passwd``, mounted secrets, its own source) could be fed to the
    parsers by naming it. Paths are now repo-relative workspace paths; the
    content is fetched through coding-containers, never from local disk.

    Raises:
        HTTPException: 400 (``INVALID_WORKSPACE_PATH``) for any path that is
            absolute outside *root*, contains ``..``, or is otherwise unconfined.
    """
    try:
        return list(dict.fromkeys(normalize_workspace_path(p, root) for p in file_paths))
    except InvalidWorkspacePathError as exc:
        raise invalid_workspace_path_http(exc) from exc


async def parse_files(valid_files: list[Path]) -> list[Any]:
    """Parse all valid files and return parse results."""
    FileService()
    factory = get_multi_language_factory()
    parse_results = []

    for file_path in valid_files:
        try:
            processor = await factory.get_processor_for_file(file_path)
            if not processor:
                logger.warning("No processor for file: %s", file_path)
                continue

            content = file_path.read_text(encoding="utf-8")

            if hasattr(processor, "parse_file_with_metrics"):
                language_processor = cast(LanguageProcessor, processor)
                parse_result = await language_processor.parse_file_with_metrics(file_path, content)
            else:
                logger.warning(
                    "Processor for %s does not support parse_file_with_metrics",
                    file_path,
                )
                continue

            if parse_result.success:
                parse_results.append(parse_result)
            else:
                logger.warning("Parse failed for %s: %s", file_path, "; ".join(parse_result.errors))

        except (OSError, UnicodeDecodeError) as e:
            logger.error("Error processing %s: %s", file_path, str(e))
            continue

    if not parse_results:
        raise HTTPException(status_code=422, detail="No files could be successfully parsed")

    return parse_results


def create_visualization_result(graph: Any, exports: dict[str, str]) -> VisualizationResult:
    """Create visualization result from graph and exports."""
    nodes = []
    for node in graph.nodes:
        viz_node = DependencyVisualizerServiceVisualizationNode(
            id=node.id,
            label=node.label,
            type=node.type.value,
            language=node.language,
            file_path=node.file_path,
            size=node.size,
            color=node.color,
            shape=node.shape,
            group=node.group,
            complexity=node.complexity,
            importance=node.importance,
            properties=node.properties or {},
        )
        nodes.append(viz_node)

    edges = []
    for edge in graph.edges:
        viz_edge = DependencyVisualizerServiceVisualizationEdge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            type=edge.type.value,
            label=edge.label,
            weight=edge.weight,
            color=edge.color,
            style=edge.style,
            properties=edge.properties or {},
        )
        edges.append(viz_edge)

    return VisualizationResult(
        id=graph.id,
        title=graph.title,
        type=graph.type.value,
        nodes=nodes,
        edges=edges,
        languages=list(graph.languages),
        statistics=graph.statistics,
        layout_options=graph.layout_options or {},
        exports=exports,
    )


def prepare_chart_data(data: dict[str, Any], chart_type: str) -> dict[str, Any]:
    """
    Prepare data for chart visualization.

    Args:
        data: Raw data to prepare
        chart_type: Type of chart to generate

    Returns:
        Prepared chart data
    """
    options: dict[str, Any] = {}
    prepared_data: dict[str, Any] = {
        "chart_type": chart_type,
        "data": data,
        "options": options,
    }

    # Chart-specific data preparation
    if chart_type == "complexity_heatmap":
        options["color_scale"] = "viridis"
        options["show_values"] = True
    elif chart_type == "dependency_matrix":
        options["symmetric"] = True
        options["show_diagonal"] = False
    elif chart_type == "language_distribution":
        options["chart_style"] = "pie"
        options["show_percentages"] = True

    return prepared_data


def transform_graph_data(graph_data: dict[str, Any], target_format: str = "vis") -> dict[str, Any]:
    """
    Transform graph data for different visualization libraries.

    Args:
        graph_data: Original graph data
        target_format: Target format (vis, d3, cytoscape, etc.)

    Returns:
        Transformed graph data
    """
    if target_format == "vis":
        # Transform for vis.js
        return {
            "nodes": [
                {
                    "id": node.get("id"),
                    "label": node.get("label"),
                    "group": node.get("type"),
                    "title": node.get("tooltip", ""),
                }
                for node in graph_data.get("nodes", [])
            ],
            "edges": [
                {
                    "from": edge.get("source"),
                    "to": edge.get("target"),
                    "label": edge.get("type", ""),
                    "arrows": "to",
                }
                for edge in graph_data.get("edges", [])
            ],
        }
    elif target_format == "d3":
        # Transform for D3.js
        return {
            "nodes": graph_data.get("nodes", []),
            "links": [
                {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "type": edge.get("type", ""),
                }
                for edge in graph_data.get("edges", [])
            ],
        }
    else:
        # Return original data for unsupported formats
        return graph_data


def validate_visualization_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate visualization request data.

    Args:
        request_data: Request data to validate

    Returns:
        Validated request data

    Raises:
        HTTPException: If validation fails
    """
    # Required fields
    if "type" not in request_data:
        raise HTTPException(status_code=400, detail="Visualization type is required")

    # Validate visualization type
    valid_types = [
        "dependency_graph",
        "call_graph",
        "inheritance_tree",
        "complexity_heatmap",
        "dependency_matrix",
        "language_distribution",
    ]
    if request_data["type"] not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid visualization type. Valid types: {valid_types}",
        )

    # Set defaults
    validated_data = {
        "type": request_data["type"],
        "container_name": request_data.get("container_name"),
        "file_paths": request_data.get("file_paths", []),
        "options": request_data.get("options", {}),
        "export_format": request_data.get("export_format", "json"),
    }

    # Validate export format if specified
    valid_formats = ["json", "png", "svg", "pdf", "dot", "graphml"]
    if validated_data["export_format"] not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid export format. Valid formats: {valid_formats}",
        )

    return validated_data
