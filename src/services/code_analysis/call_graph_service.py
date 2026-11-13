"""Call Graph Analysis Service

Provides comprehensive call graph analysis capabilities for building and analyzing
function call graphs, including dependency tracking and graph traversal operations.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from models.static_analysis_models import (
    CallGraphEdgeLocal,
    CallGraphNodeLocal,
    StaticAnalysisCallGraphEdge,
    StaticAnalysisCallGraphNode,
)
from services.code_structure_analyzer import CodeStructureAnalyzer

logger = logging.getLogger(__name__)


class CallGraph(BaseModel):
    """Complete call graph representation model.

    This model represents a complete function call graph including all nodes,
    edges, and metadata about the graph structure and analysis parameters.
    """

    root: str = Field(..., description="Root function name")
    nodes: list[StaticAnalysisCallGraphNode] = Field(..., description="Nodes in the call graph")
    edges: list[StaticAnalysisCallGraphEdge] = Field(..., description="Edges in the call graph")
    direction: str = Field(..., description="Direction of the call graph (outgoing/incoming)")
    depth: int = Field(..., description="Maximum depth of the call graph")


class CallGraphService:
    """Service for comprehensive call graph analysis operations.

    This service provides advanced call graph analysis capabilities including:
    - Function call graph construction
    - Dependency analysis and traversal
    - Graph metrics and analysis

    The service integrates with code structure analyzers to provide
    detailed insights into function call patterns and dependencies.
    """

    def __init__(self, analyzer: CodeStructureAnalyzer):
        """Initialize the CallGraphService service.

        Args:
            analyzer (CodeStructureAnalyzer): Analyzer
        """
        self.analyzer = analyzer

    async def build_call_graph(
        self,
        function_name: str,
        depth: int = 2,
        direction: str = "outgoing",
        code_graph_id: str | None = None,
    ) -> CallGraph:
        """Build a comprehensive call graph for a function.

        Args:
            function_name: Name of the function to analyze
            depth: Maximum depth of analysis (default: 2)
            direction: Direction of analysis ('outgoing' or 'incoming')
            code_graph_id: Optional code graph identifier

        Returns:
            CallGraph: Complete call graph with nodes and edges

        Note:
            This method analyzes the function's call patterns to build a
            comprehensive graph showing dependencies and relationships.
        """
        try:
            # Get call graph data from analyzer
            graph_data = self.analyzer.analyze_call_graph(function_name, depth, direction)

            # Convert to our model format
            nodes: list[Any] = []
            for node_data in graph_data.get("nodes", []):
                nodes.append(
                    CallGraphNodeLocal(
                        id=node_data.get("id", ""),
                        name=node_data.get("name", ""),
                        type=node_data.get("type", "function"),
                        depth=node_data.get("depth", 0),
                        properties=node_data.get("properties", {}),
                    )
                )

            edges: list[Any] = []
            for edge_data in graph_data.get("edges", []):
                edges.append(
                    CallGraphEdgeLocal(
                        source=edge_data.get("source", ""),
                        target=edge_data.get("target", ""),
                        type=edge_data.get("type", "calls"),
                        properties=edge_data.get("properties", {}),
                    )
                )

            return CallGraph(
                root=function_name,
                nodes=nodes,
                edges=edges,
                direction=direction,
                depth=depth,
            )

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error building call graph for %s: %s", function_name, e)
            # Return empty graph
            return CallGraph(
                root=function_name,
                nodes=[],
                edges=[],
                direction=direction,
                depth=depth,
            )

    async def get_function_calls(self, code_graph_id: str, function_name: str) -> dict[str, Any]:
        """Get function calls for a specific function."""
        try:
            calls = self.analyzer.find_function_calls(function_name)
            # Ensure we return a dict structure
            if isinstance(calls, list):
                return {
                    "function_name": function_name,
                    "calls": calls,
                    "total_calls": len(calls),
                }
            return calls
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error getting function calls for %s: %s", function_name, e)
            return {"error": str(e), "calls": []}

    async def find_class_references(self, code_graph_id: str, class_name: str) -> dict[str, Any]:
        """Find references to a specific class."""
        try:
            references = self.analyzer.find_class_references(class_name)
            # Ensure we return a dict structure
            if isinstance(references, list):
                return {
                    "class_name": class_name,
                    "references": references,
                    "total_references": len(references),
                }
            return references
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error finding class references for %s: %s", class_name, e)
            return {"error": str(e), "references": []}

    async def find_import_usage(self, code_graph_id: str, module_name: str) -> dict[str, Any]:
        """Find usage of a specific import/module."""
        try:
            usage = self.analyzer.find_import_usage(module_name)
            # Ensure we return a dict structure
            if isinstance(usage, list):
                return {
                    "module_name": module_name,
                    "usage": usage,
                    "total_usage": len(usage),
                }
            return usage
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error finding import usage for %s: %s", module_name, e)
            return {"error": str(e), "usage": []}

    async def get_dependency_graph(
        self, code_graph_id: str, node_id: str, max_depth: int = 3
    ) -> dict[str, Any]:
        """Get dependency graph for a node."""
        try:
            return self.analyzer.analyze_dependency_graph(node_id, max_depth)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error getting dependency graph for %s: %s", node_id, e)
            return {"error": str(e), "dependencies": []}
