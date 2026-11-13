"""Visualization modules package.

This package contains modular components for data visualization and graph rendering.
"""

from .chart_generation import router as chart_router

# Re-export endpoint functions from configuration
# Configuration
from .configuration import (
    VisualizationConfig,
    get_visualization_capabilities,
    get_visualization_config,
    get_visualization_examples,
)

# Re-export data utilities functions
# Utilities
from .data_utilities import (
    create_visualization_result,
    parse_files,
    prepare_chart_data,
    transform_graph_data,
    validate_file_paths,
    validate_visualization_request,
    validate_visualization_type,
)
from .export_rendering import router as export_router
from .export_utilities import (
    export_to_format,
    get_supported_formats,
    validate_export_format,
)
from .graph_visualization import router as graph_router

# Models
from .shared_models import (
    ChartType,
    DependencyVisualizerServiceVisualizationEdge,
    DependencyVisualizerServiceVisualizationNode,
    ExportFormat,
    VisualizationCapabilitiesResponse,
    VisualizationEdge,  # Add missing alias
    VisualizationNode,  # Add missing alias
    VisualizationRequest,
    VisualizationResponse,
    VisualizationResult,
)

__all__ = [
    # Routers
    "chart_router",
    "export_router",
    "graph_router",
    # Configuration
    "VisualizationConfig",
    "get_visualization_config",
    "get_visualization_capabilities",
    "get_visualization_examples",
    # Data Utilities
    "validate_visualization_type",
    "validate_file_paths",
    "parse_files",
    "create_visualization_result",
    "prepare_chart_data",
    "transform_graph_data",
    "validate_visualization_request",
    # Export Utilities
    "export_to_format",
    "get_supported_formats",
    "validate_export_format",
    # Models
    "ChartType",
    "ExportFormat",
    "VisualizationRequest",
    "VisualizationResponse",
    "VisualizationResult",
    "VisualizationCapabilitiesResponse",
    "DependencyVisualizerServiceVisualizationNode",
    "DependencyVisualizerServiceVisualizationEdge",
    "VisualizationEdge",  # Add missing alias
    "VisualizationNode",  # Add missing alias
]
