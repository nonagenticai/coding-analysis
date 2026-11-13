"""Models package.

This package contains all data models used throughout the application.
"""

# Import all models to ensure they are registered and forward references are resolved
from .code_graph_models import (
    ArchitectureCodeGraphResponse,
    ArchitectureNodeResponse,
    CodebaseInitRequest,
    CodeGraphAnalysisRequest,
    CodeGraphAnalysisResponse,
    CodeGraphIntegrationRequest,
    EdgeResponse,
    EntityQueryRequest,
    EntityQueryResponse,
    InitResponse,
    NodeQuery,
    NodeResponse,
    PatternSearchRequest,
    PatternSearchResponse,
    RawQueryRequest,
    RawQueryResponse,
    RelatedNodesRequest,
    RelatedNodesResponse,
    RuntimeEdgeResponse,
    RuntimeGraphResponse,
)

# Re-export commonly used models from submodules
__all__ = [
    "CodebaseInitRequest",
    "CodeGraphAnalysisRequest",
    "PatternSearchRequest",
    "EntityQueryRequest",
    "RelatedNodesRequest",
    "RawQueryRequest",
    "NodeQuery",
    "NodeResponse",
    "EdgeResponse",
    "InitResponse",
    "CodeGraphAnalysisResponse",
    "PatternSearchResponse",
    "EntityQueryResponse",
    "RelatedNodesResponse",
    "RawQueryResponse",
    "ArchitectureNodeResponse",
    "RuntimeEdgeResponse",
    "ArchitectureCodeGraphResponse",
    "RuntimeGraphResponse",
    "CodeGraphIntegrationRequest",
]
