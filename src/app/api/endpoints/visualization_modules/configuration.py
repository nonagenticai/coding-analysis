"""
Visualization Configuration and Capabilities

Configuration management, system capabilities, and example templates for visualization.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException

from auth.tenant import get_current_user_with_tenant
from services.dependency_visualizer import (
    EdgeType,
    NodeType,
    VisualizationType,
)

from ...models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from .shared_models import VisualizationCapabilitiesResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class VisualizationConfig:
    """Configuration class for visualization settings."""

    def __init__(self):
        """Initialize visualization configuration."""
        self.max_nodes = 1000
        self.max_edges = 5000
        self.default_layout = "hierarchical"
        self.supported_formats = [
            "json",
            "cytoscape",
            "d3",
            "graphviz",
            "mermaid",
            "vis",
        ]
        self.cache_ttl = 3600
        self.chunk_size = 100


@router.get(
    "/capabilities",
    response_model=VisualizationCapabilitiesResponse,
    operation_id="analysis_get_visualization_capabilities",
    summary="📋 Visualization Capabilities",
    description="""
    🔍 **Visualization System Capabilities** - Discover available visualization types and features

    ## Purpose
    Provides comprehensive information about the dependency visualization system's capabilities,
    including supported visualization types, export formats, and configuration options.

    ## Response Information
    - **Visualization Types**: Available diagram types and their purposes
    - **Node Types**: Entity types that can be visualized
    - **Edge Types**: Relationship types that can be displayed
    - **Export Formats**: Available output formats for visualization
    - **Layout Algorithms**: Supported layout algorithms for different diagram types
    - **Supported Languages**: Programming languages with visualization support

    ## Use Cases
    - **Planning**: Understand available visualization options
    - **Integration**: Determine compatibility with existing tools
    - **Documentation**: Reference for visualization capabilities
    - **Development**: Plan custom visualization workflows
    """,
    responses={
        200: {
            "description": "✅ Visualization capabilities retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "visualization_types": [
                            "dependency_graph",
                            "component_diagram",
                            "layer_architecture",
                            "call_flow",
                            "cross_language_map",
                        ],
                        "node_types": [
                            "class",
                            "interface",
                            "function",
                            "method",
                            "variable",
                            "module",
                            "package",
                            "namespace",
                        ],
                        "edge_types": [
                            "dependency",
                            "inheritance",
                            "implementation",
                            "call",
                            "reference",
                            "import",
                            "extends",
                            "composition",
                        ],
                        "export_formats": [
                            "cytoscape",
                            "d3",
                            "graphviz",
                            "mermaid",
                            "json",
                        ],
                        "layout_algorithms": [
                            "force-directed",
                            "hierarchical",
                            "circular",
                            "grid",
                            "tree",
                            "layered",
                        ],
                        "supported_languages": [
                            "java",
                            "rust",
                            "go",
                            "python",
                            "javascript",
                            "typescript",
                            "csharp",
                            "cpp",
                            "c",
                        ],
                    }
                }
            },
        },
        400: {
            "model": Error400Response,
            "description": "❌ Bad request - Invalid request format or parameters",
        },
        404: {
            "model": Error404Response,
            "description": "🔍 Not found - Capabilities service or resource not found",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "⚠️ Unprocessable entity - Capabilities retrieval failed",
        },
        500: {
            "model": Error500Response,
            "description": "💥 Internal server error - Unexpected error retrieving capabilities",
        },
    },
)
async def get_visualization_capabilities(
    user: User = Depends(get_current_user_with_tenant),
) -> VisualizationCapabilitiesResponse:
    """Get visualization system capabilities. Requires authentication."""

    try:
        capabilities = VisualizationCapabilitiesResponse(
            visualization_types=[t.value for t in VisualizationType],
            node_types=[t.value for t in NodeType],
            edge_types=[t.value for t in EdgeType],
            export_formats=["cytoscape", "d3", "graphviz", "mermaid", "json"],
            layout_algorithms=[
                "force-directed",
                "hierarchical",
                "circular",
                "grid",
                "tree",
                "layered",
            ],
            supported_languages=[
                "java",
                "rust",
                "go",
                "python",
                "javascript",
                "typescript",
                "csharp",
                "cpp",
                "c",
            ],
        )

        return capabilities

    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error retrieving visualization capabilities: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Error retrieving capabilities: {str(e)}"
        ) from e


@router.get(
    "/examples",
    response_model=dict[str, Any],
    operation_id="analysis_get_visualization_examples",
    summary="📚 Visualization Examples",
    description="""
    📖 **Visualization Examples and Templates** - Get example configurations and use cases

    ## Purpose
    Provides ready-to-use examples and templates for common visualization scenarios,
    helping users quickly create effective dependency visualizations.

    ## Example Categories
    - **Basic Dependency Graph**: Basic file dependency visualization
    - **Cross-Language Analysis**: Multi-language project visualization
    - **Component Architecture**: High-level component relationships
    - **Layer Architecture**: Layered application structure
    - **Complexity Analysis**: Focus on complex entities and relationships

    ## Response Information
    - Complete request examples for each visualization type
    - Recommended options and configurations
    - Best practices for different use cases
    - Sample output descriptions
    """,
    responses={
        200: {
            "description": "✅ Visualization examples retrieved successfully",
        },
        400: {
            "model": Error400Response,
            "description": "❌ Bad request - Invalid request format or parameters",
        },
        404: {
            "model": Error404Response,
            "description": "🔍 Not found - Examples service or templates not found",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "⚠️ Unprocessable entity - Examples generation failed",
        },
        500: {
            "model": Error500Response,
            "description": "💥 Internal server error - Unexpected error retrieving examples",
        },
    },
)
async def get_visualization_examples(
    user: User = Depends(get_current_user_with_tenant),
) -> dict[str, Any]:
    """Get visualization examples and templates. Requires authentication."""

    try:
        examples = {
            "basic_dependency_graph": {
                "description": "Basic dependency graph for a single-language project",
                "request": {
                    "file_paths": ["/path/to/src/**/*.java"],
                    "visualization_type": "dependency_graph",
                    "options": {
                        "group_by_language": False,
                        "show_file_nodes": True,
                    },
                    "export_formats": ["cytoscape", "json"],
                },
                "use_case": "Understanding basic project structure and dependencies",
            },
            "cross_language_analysis": {
                "description": "Multi-language project with cross-language relationships",
                "request": {
                    "file_paths": [
                        "/path/to/backend/**/*.java",
                        "/path/to/frontend/**/*.ts",
                        "/path/to/service/**/*.rs",
                    ],
                    "visualization_type": "cross_language_map",
                    "options": {
                        "group_by_language": True,
                        "highlight_cross_language": True,
                        "languages": ["java", "typescript", "rust"],
                    },
                    "export_formats": ["cytoscape", "d3", "mermaid"],
                },
                "use_case": "Analyzing polyglot architecture and language boundaries",
            },
            "component_architecture": {
                "description": "High-level component and module relationships",
                "request": {
                    "file_paths": ["/path/to/project/**/*"],
                    "visualization_type": "component_diagram",
                    "options": {
                        "node_types": ["class", "interface", "module"],
                        "min_complexity": 5.0,
                        "layout": {
                            "algorithm": "hierarchical",
                            "direction": "LR",
                        },
                    },
                    "export_formats": ["graphviz", "mermaid"],
                },
                "use_case": "Architectural documentation and component design review",
            },
            "complexity_focused": {
                "description": "Visualization focusing on complex entities and relationships",
                "request": {
                    "file_paths": ["/path/to/src/**/*"],
                    "visualization_type": "dependency_graph",
                    "options": {
                        "min_complexity": 10.0,
                        "max_nodes": 50,
                        "highlight_hotspots": True,
                        "layout": {
                            "algorithm": "force-directed",
                            "gravity": 0.2,
                        },
                    },
                    "export_formats": ["cytoscape", "json"],
                },
                "use_case": "Identifying refactoring targets and complexity hotspots",
            },
            "api_integration_map": {
                "description": "Visualization of API integrations and service boundaries",
                "request": {
                    "file_paths": [
                        "/path/to/api/**/*.java",
                        "/path/to/client/**/*.ts",
                        "/path/to/gateway/**/*.go",
                    ],
                    "visualization_type": "call_flow",
                    "options": {
                        "edge_types": ["api_call", "rpc_call"],
                        "group_by_service": True,
                        "show_external_apis": True,
                    },
                    "export_formats": ["d3", "mermaid"],
                },
                "use_case": "Understanding service architecture and API dependencies",
            },
        }

        response = {
            "examples": examples,
            "best_practices": {
                "large_codebases": "Use filtering options to focus on specific areas",
                "cross_language": "Enable language grouping for better organization",
                "documentation": "Use mermaid format for markdown integration",
                "interactive_analysis": "Use cytoscape format for web dashboards",
                "complexity_analysis": "Apply complexity thresholds to highlight problem areas",
            },
            "tips": [
                "Start with basic dependency graph to understand overall structure",
                "Use component diagrams for high-level architectural views",
                "Apply filters to reduce complexity for large codebases",
                "Combine multiple export formats for different use cases",
                "Use cross-language maps to identify integration points",
            ],
        }

        return response

    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error retrieving visualization examples: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving examples: {str(e)}") from e


def get_visualization_config() -> dict[str, Any]:
    """
    Get visualization configuration settings.

    Returns:
        Dictionary containing visualization configuration
    """
    return {
        "default_options": {
            "layout": "hierarchical",
            "group_by_language": True,
            "show_file_nodes": True,
            "max_nodes": 100,
            "min_complexity": 0.0,
            "highlight_hotspots": True,
            "edge_thickness": "dynamic",
            "node_size": "dynamic",
        },
        "export_formats": {
            "json": {"description": "Raw JSON data", "supports_streaming": True},
            "cytoscape": {
                "description": "Cytoscape.js format",
                "supports_streaming": True,
            },
            "d3": {"description": "D3.js format", "supports_streaming": True},
            "graphviz": {
                "description": "GraphViz DOT format",
                "supports_streaming": False,
            },
            "mermaid": {
                "description": "Mermaid diagram syntax",
                "supports_streaming": False,
            },
            "vis": {"description": "vis.js format", "supports_streaming": True},
        },
        "layout_algorithms": {
            "hierarchical": {
                "description": "Top-down or left-right hierarchy",
                "best_for": "tree structures",
            },
            "force-directed": {
                "description": "Physics-based layout",
                "best_for": "complex networks",
            },
            "circular": {
                "description": "Nodes in a circle",
                "best_for": "cyclic dependencies",
            },
            "grid": {
                "description": "Regular grid layout",
                "best_for": "equal importance nodes",
            },
            "tree": {"description": "Tree layout", "best_for": "hierarchical data"},
            "layered": {
                "description": "Layered architecture",
                "best_for": "architectural views",
            },
        },
        "performance": {
            "max_nodes_interactive": 500,
            "max_nodes_static": 5000,
            "max_edges_interactive": 1000,
            "max_edges_static": 10000,
            "chunk_size": 100,
            "cache_ttl_seconds": 3600,
        },
        "visualization_types": {
            "dependency_graph": "File and module dependencies",
            "call_graph": "Function and method calls",
            "inheritance_tree": "Class inheritance hierarchy",
            "component_diagram": "High-level components",
            "cross_language_map": "Cross-language relationships",
            "complexity_heatmap": "Code complexity visualization",
            "layer_architecture": "Architectural layers",
        },
    }
