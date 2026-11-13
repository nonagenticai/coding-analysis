"""
Graph Visualization Endpoints

Main dependency graph creation and visualization endpoints.
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Body, Depends, HTTPException

from auth.tenant import get_current_user_with_tenant, require_tenant
from services.dependency_visualizer import get_dependency_visualizer

from ...models.common_api_models import ResponseStatus
from ...models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from ...utils.container_validation import validate_and_get_container_for_tenant
from ...utils.workspace_files import (
    WorkspaceUnresolvableError,
    materialize_workspace,
    remove_materialized,
    workspace_root,
    workspace_unresolvable_http,
)
from .data_utilities import (
    create_visualization_result,
    parse_files,
    validate_file_paths,
    validate_visualization_type,
)
from .shared_models import VisualizationRequest, VisualizationResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/create-graph",
    response_model=VisualizationResponse,
    operation_id="create_dependency_visualization",
    summary="📊 Create Dependency Visualization",
    description="""
    🎨 **Create Cross-Language Dependency Visualization** - Generate interactive dependency graphs and architectural diagrams

    ## Purpose
    Creates comprehensive visual representations of code dependencies and relationships
    across multiple programming languages, enabling architectural understanding and analysis.

    ## Visualization Types

    ### 🕸️ **Dependency Graph**
    - Shows all entities and their relationships
    - Interactive force-directed layout
    - Color-coded by language and complexity
    - Ideal for understanding overall architecture

    ### 🏗️ **Component Diagram**
    - Groups entities by modules/packages
    - Hierarchical layout with clear separation
    - Shows high-level component relationships
    - Perfect for architectural documentation

    ### 🏛️ **Layer Architecture**
    - Organizes code by architectural layers
    - Top-down hierarchical view
    - Shows layer dependencies and violations
    - Great for layered architecture analysis

    ### 🔄 **Call Flow**
    - Focuses on function/method call relationships
    - Sequential flow visualization
    - Shows execution paths and patterns
    - Useful for performance analysis

    ### 🌐 **Cross-Language Map**
    - Highlights interactions between languages
    - Shows FFI calls, API endpoints, service boundaries
    - Language-specific grouping and styling
    - Essential for polyglot architecture analysis

    ## Visualization Features

    ### 🎯 **Smart Filtering**
    - Filter by programming language
    - Complexity-based filtering
    - Node type selection (classes, functions, etc.)
    - Custom property-based filters

    ### 🎨 **Visual Encoding**
    - **Node Size**: Based on complexity or importance
    - **Node Color**: Language-specific color schemes
    - **Edge Weight**: Relationship strength
    - **Edge Style**: Relationship type (solid, dashed, bold)

    ### 📈 **Interactive Elements**
    - Hover for detailed entity information
    - Click to focus on specific relationships
    - Zoom and pan for large codebases
    - Search and highlight capabilities

    ## Export Formats

    ### 🌐 **Cytoscape.js**
    - Interactive web-based visualization
    - Rich styling and layout options
    - Built-in user interaction capabilities
    - Perfect for web applications

    ### 📊 **D3.js**
    - Flexible data format for custom visualizations
    - Compatible with D3 force simulation
    - Extensive customization possibilities
    - Ideal for custom web dashboards

    ### 🔄 **Graphviz DOT**
    - Industry-standard graph description language
    - High-quality static diagram generation
    - Compatible with many tools and editors
    - Excellent for documentation

    ### 📋 **Mermaid**
    - Markdown-compatible diagram syntax
    - Easy integration with documentation
    - GitHub/GitLab native support
    - Perfect for README files and wikis

    ### 📄 **JSON**
    - Raw structured data format
    - Easy integration with other tools
    - Custom processing and analysis
    - Data exchange and storage

    ## Request Options

    ### 🎛️ **Filtering Options**
    ```json
    {
      "languages": ["rust", "java", "typescript"],
      "node_types": ["class", "interface", "function"],
      "min_complexity": 5.0,
      "max_nodes": 100
    }
    ```

    ### 🔧 **Layout Options**
    ```json
    {
      "layout": {
        "algorithm": "force-directed",
        "spacing": 100,
        "gravity": 0.1,
        "iterations": 100
      }
    }
    ```

    ### 🎨 **Styling Options**
    ```json
    {
      "group_by_language": true,
      "show_file_nodes": false,
      "edge_bundling": true,
      "highlight_cross_language": true
    }
    ```

    ## Response Information
    - Complete graph structure (nodes and edges)
    - Multiple export format outputs
    - Comprehensive statistics and metrics
    - Layout configuration for rendering
    - Language distribution and complexity analysis

    ## Use Cases
    - **Architecture Review**: Understand system structure and dependencies
    - **Refactoring Planning**: Identify tightly coupled components
    - **Documentation**: Generate architectural diagrams automatically
    - **Code Analysis**: Visualize complexity and relationship patterns
    - **Onboarding**: Help new developers understand codebase structure
    - **Technical Debt**: Identify problematic dependency patterns
    """,
    responses={
        200: {
            "description": "✅ Visualization created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Visualization created with 45 nodes and 67 edges",
                        "timestamp": "2025-01-12T10:30:00.000Z",
                        "result": {
                            "id": "viz_550e8400",
                            "title": "Dependency Graph Analysis",
                            "type": "dependency_graph",
                            "nodes": [
                                {
                                    "id": "node_1",
                                    "label": "UserService",
                                    "type": "class",
                                    "language": "java",
                                    "file_path": "/src/main/java/UserService.java",
                                    "size": 3,
                                    "color": "#FF6B6B",
                                    "shape": "box",
                                    "group": "service",
                                    "complexity": 12.5,
                                    "importance": 0.8,
                                    "properties": {"package": "com.example.service"},
                                }
                            ],
                            "edges": [
                                {
                                    "id": "edge_1",
                                    "source": "node_1",
                                    "target": "node_2",
                                    "type": "dependency",
                                    "label": "calls",
                                    "weight": 2.0,
                                    "color": "#4ECDC4",
                                    "style": "solid",
                                    "properties": {"call_count": 5},
                                }
                            ],
                            "languages": ["java", "typescript", "rust"],
                            "statistics": {
                                "total_nodes": 45,
                                "total_edges": 67,
                                "average_complexity": 8.3,
                                "language_distribution": {
                                    "java": 25,
                                    "typescript": 12,
                                    "rust": 8,
                                },
                            },
                            "layout_options": {
                                "algorithm": "force-directed",
                                "spacing": 100,
                                "gravity": 0.1,
                            },
                            "exports": {
                                "cytoscape": '{ "nodes": [...], "edges": [...] }',
                                "json": '{ "graph": {...} }',
                            },
                        },
                    }
                }
            },
        },
        400: {
            "model": Error400Response,
            "description": "❌ Bad request - Invalid request parameters",
        },
        404: {
            "model": Error404Response,
            "description": "🔍 Not found - Files not found or inaccessible",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "⚠️ Unprocessable entity - Visualization generation failed",
        },
        500: {
            "model": Error500Response,
            "description": "💥 Internal server error - Unexpected visualization error",
        },
    },
)
async def create_dependency_visualization(
    request: VisualizationRequest = Body(..., description="Visualization request"),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> VisualizationResponse:
    """Create a cross-language dependency visualization.

    Requires authentication. ``file_paths`` are repo-relative paths inside the
    caller's workspace ``container_name``; their content is read through
    coding-containers under the caller's tenant. This pod's own filesystem is
    never read (it used to be: any existing absolute path was parsed).
    """

    local_root = None
    try:
        logger.info(f"Creating dependency visualization for user {user.preferred_username}")
        if not request.file_paths:
            raise HTTPException(status_code=400, detail="At least one file path is required")

        # Validate visualization type
        vis_type = validate_visualization_type(request.visualization_type)

        # Ownership first: a container the caller's tenant does not own is a
        # 404/403 before any path is even considered.
        container_info = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id
        )
        root = workspace_root(container_info)

        # Confine every path to the workspace (400 on any escape) ...
        rel_paths = validate_file_paths(request.file_paths, root)

        # ... and read them through coding-containers, never from local disk.
        try:
            local_root = await materialize_workspace(
                request.container_name, tenant_id=tenant_id, root=root, rel_paths=rel_paths
            )
        except WorkspaceUnresolvableError as exc:
            raise workspace_unresolvable_http(exc) from exc
        valid_files = [local_root / rel for rel in rel_paths if (local_root / rel).is_file()]

        # Parse all files
        parse_results = await parse_files(valid_files)

        # Create visualization
        visualizer = get_dependency_visualizer()
        graph = await visualizer.create_dependency_graph(
            parse_results, vis_type, request.options or {}
        )

        # Export to requested formats
        exports = await visualizer.export_to_formats(graph, request.export_formats)

        # Convert to response model
        result = create_visualization_result(graph, exports)

        logger.info(
            "Visualization created: %s with %d nodes, %d edges",
            graph.id,
            len(graph.nodes),
            len(graph.edges),
        )

        return VisualizationResponse(
            status=ResponseStatus.SUCCESS,
            message=(
                f"Visualization created with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            ),
            result=result,
        )

    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, OSError, UnicodeDecodeError) as e:
        logger.exception("Visualization creation failed: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Visualization creation failed: {str(e)}"
        ) from e
    finally:
        remove_materialized(local_root)
