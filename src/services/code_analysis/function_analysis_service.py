"""Function Analysis Service

Provides comprehensive function analysis capabilities including call relationship extraction,
metrics calculation, and network analysis for code functions.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from services.code_graph_metrics import CodeGraphMetricsService
from services.code_structure_analyzer import CodeStructureAnalyzer

logger = logging.getLogger(__name__)


class FunctionCallRelation(BaseModel):
    """Model for representing function call relationships.

    This model captures the details of function calls including source and target
    functions, location information, and entity types.
    """

    source: str = Field(..., description="Caller function")
    target: str = Field(..., description="Callee function")
    line_number: int = Field(..., description="Line number of the call")
    file_path: str = Field(..., description="File path containing the call")
    source_type: str = Field(default="function", description="Type of the caller")
    target_type: str = Field(default="function", description="Type of the callee")


class FunctionMetrics(BaseModel):
    """Model for basic function metrics.

    Provides essential metrics about function behavior including call patterns,
    complexity measures, and importance scores.
    """

    function_name: str = Field(..., description="Name of the function")
    outgoing_calls: int = Field(..., description="Number of outgoing function calls")
    incoming_calls: int = Field(..., description="Number of incoming function calls")
    unique_callees: int = Field(..., description="Number of unique functions called")
    unique_callers: int = Field(..., description="Number of unique functions calling this function")
    cyclomatic_complexity: int = Field(..., description="Cyclomatic complexity of the function")
    call_based_importance: float = Field(..., description="Importance score based on call patterns")
    error: str | None = Field(None, description="Error message if calculation failed")


class ComprehensiveFunctionMetrics(BaseModel):
    """Comprehensive function metrics model.

    Combines basic structural metrics with advanced network analysis metrics
    to provide a complete picture of function importance and behavior.
    """

    function_name: str = Field(..., description="Name of the function")
    # Structural metrics
    outgoing_calls: int = Field(..., description="Number of outgoing function calls")
    incoming_calls: int = Field(..., description="Number of incoming function calls")
    unique_callees: int = Field(..., description="Number of unique functions called")
    unique_callers: int = Field(..., description="Number of unique functions calling this function")
    cyclomatic_complexity: int = Field(..., description="Cyclomatic complexity of the function")
    call_based_importance: float = Field(..., description="Importance based on call patterns")
    # Network metrics
    degree_centrality: float = Field(..., description="Degree centrality in the code graph")
    betweenness_centrality: float = Field(
        ..., description="Betweenness centrality in the code graph"
    )
    closeness_centrality: float = Field(..., description="Closeness centrality in the code graph")
    pagerank: float = Field(..., description="PageRank score in the code graph")
    network_importance: float = Field(..., description="Network-based importance")


class FunctionAnalysisService:
    """Service for comprehensive function analysis operations.

    This service provides advanced function analysis capabilities including:
    - Function call relationship extraction
    - Structural and network metrics calculation
    - Comprehensive function behavior analysis

    The service integrates with code structure analyzers and metrics services
    to provide detailed insights into function behavior and importance.
    """

    def __init__(
        self,
        analyzer: CodeStructureAnalyzer,
        metrics_service: CodeGraphMetricsService,
    ):
        """Initialize the FunctionAnalysisService service.

        Args:
            analyzer (CodeStructureAnalyzer): Analyzer
            metrics_service (CodeGraphMetricsService): Metrics service
        """
        self.analyzer = analyzer
        self.metrics_service = metrics_service

    async def extract_function_calls(
        self, code: str, file_path: str = "unnamed.py"
    ) -> list[FunctionCallRelation]:
        """Extract function call relationships from source code.

        Args:
            code: Source code string to analyze
            file_path: Path to the file containing the code

        Returns:
            List[FunctionCallRelation]: List of function call relationships found

        Note:
            This method analyzes the provided code string to identify function
            calls and their relationships, returning structured data about each call.
        """
        try:
            # Use the analyzer to extract function calls from code string
            result = self.analyzer.extract_call_relationships_from_string(code, file_path)

            # Convert to our model format
            relations: list[Any] = []
            for call in result:
                relations.append(
                    FunctionCallRelation(
                        source=call.get("source", "unknown"),
                        target=call.get("target", "unknown"),
                        line_number=call.get("line_number", 0),
                        file_path=file_path,
                        source_type=call.get("source_type", "function"),
                        target_type=call.get("target_type", "function"),
                    )
                )

            return relations

        except Exception as e:
            logger.error(f"Error extracting function calls: {e}")
            return []

    async def get_function_metrics(self, function_name: str) -> FunctionMetrics:
        """Get basic structural metrics for a function.

        Args:
            function_name: Name of the function to analyze

        Returns:
            FunctionMetrics: Basic metrics including calls, complexity, and importance

        Note:
            This method calculates essential function metrics including call patterns,
            complexity measures, and importance scores based on structural analysis.
        """
        try:
            # Get metrics from analyzer
            metrics = self.analyzer.calculate_function_metrics(function_name)

            return FunctionMetrics(
                function_name=function_name,
                outgoing_calls=metrics.get("outgoing_calls", 0),
                incoming_calls=metrics.get("incoming_calls", 0),
                unique_callees=metrics.get("unique_callees", 0),
                unique_callers=metrics.get("unique_callers", 0),
                cyclomatic_complexity=metrics.get("cyclomatic_complexity", 0),
                call_based_importance=metrics.get("call_based_importance", 0.0),
                error=None,
            )

        except Exception as e:
            logger.error(f"Error getting function metrics for {function_name}: {e}")
            return FunctionMetrics(
                function_name=function_name,
                outgoing_calls=0,
                incoming_calls=0,
                unique_callees=0,
                unique_callers=0,
                cyclomatic_complexity=0,
                call_based_importance=0.0,
                error=str(e),
            )

    async def get_comprehensive_metrics(
        self, function_name: str, code_graph_id: str
    ) -> ComprehensiveFunctionMetrics:
        """Get comprehensive function metrics including network analysis.

        Args:
            function_name: Name of the function to analyze
            code_graph_id: ID of the code graph for network analysis

        Returns:
            ComprehensiveFunctionMetrics: Complete metrics including network measures

        Note:
            This method combines basic structural metrics with advanced network analysis
            to provide a complete picture of function importance and behavior within
            the broader codebase context.
        """
        try:
            # Get basic metrics
            basic_metrics = await self.get_function_metrics(function_name)

            # Get network metrics
            network_metrics = await self.metrics_service.get_function_network_metrics(function_name)

            return ComprehensiveFunctionMetrics(
                function_name=function_name,
                outgoing_calls=basic_metrics.outgoing_calls,
                incoming_calls=basic_metrics.incoming_calls,
                unique_callees=basic_metrics.unique_callees,
                unique_callers=basic_metrics.unique_callers,
                cyclomatic_complexity=basic_metrics.cyclomatic_complexity,
                call_based_importance=basic_metrics.call_based_importance,
                degree_centrality=network_metrics.get("degree_centrality", 0.0),
                betweenness_centrality=network_metrics.get("betweenness_centrality", 0.0),
                closeness_centrality=network_metrics.get("closeness_centrality", 0.0),
                pagerank=network_metrics.get("pagerank", 0.0),
                network_importance=network_metrics.get("network_importance", 0.0),
            )

        except Exception as e:
            logger.error(f"Error getting comprehensive metrics for {function_name}: {e}")
            # Return basic metrics with zero network metrics
            basic_metrics = await self.get_function_metrics(function_name)
            return ComprehensiveFunctionMetrics(
                function_name=function_name,
                outgoing_calls=basic_metrics.outgoing_calls,
                incoming_calls=basic_metrics.incoming_calls,
                unique_callees=basic_metrics.unique_callees,
                unique_callers=basic_metrics.unique_callers,
                cyclomatic_complexity=basic_metrics.cyclomatic_complexity,
                call_based_importance=basic_metrics.call_based_importance,
                degree_centrality=0.0,
                betweenness_centrality=0.0,
                closeness_centrality=0.0,
                pagerank=0.0,
                network_importance=0.0,
            )
