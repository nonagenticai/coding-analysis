"""Response models for metrics modules endpoints.

This module contains Pydantic response models for all metrics-related API endpoints,
including performance metrics, system metrics, analytics dashboards, and metrics aggregation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...types import SAFE_STRING_PATTERN
from .common_api_models import BaseResponse, ResponseStatus

# ==================== Performance Metrics Models ====================


class PerformanceSummaryResponse(BaseModel):
    """Response model for multi-language performance summary."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_statistics": {},
                "language_breakdown": {},
                "parser_performance": {},
                "processing_trends": {},
                "resource_utilization": {},
                "error_analysis": {},
                "recommendations": [],
            }
        }
    )

    overall_statistics: dict[str, Any] = Field(
        ..., description="Cross-language performance aggregates"
    )
    language_breakdown: dict[str, Any] = Field(..., description="Per-language detailed statistics")
    parser_performance: dict[str, Any] = Field(
        ..., description="Strategy-specific performance metrics"
    )
    processing_trends: dict[str, Any] = Field(..., description="Time-based performance patterns")
    resource_utilization: dict[str, Any] = Field(..., description="Memory and CPU usage statistics")
    error_analysis: dict[str, Any] = Field(..., description="Common error patterns and frequencies")
    recommendations: list[str] = Field(..., description="Performance optimization suggestions")


class LanguagePerformanceStatsResponse(BaseModel):
    """Response model for language-specific performance statistics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "python",
                "total_files_processed": 150,
                "total_processing_time_ms": 45000.0,
                "average_processing_time_ms": 300.0,
                "success_rate": 0.95,
                "total_entities": 1500,
                "total_relationships": 3000,
                "total_lines_of_code": 45000,
                "parser_strategy_usage": {"tree_sitter": 120, "ast": 30},
                "error_count": 5,
                "entities_per_file": 10.0,
                "lines_per_file": 300.0,
                "processing_speed_loc_per_ms": 1.0,
            }
        }
    )

    language: str = Field(
        ..., description="Programming language identifier", pattern=SAFE_STRING_PATTERN
    )
    total_files_processed: int = Field(..., description="Number of files analyzed")
    total_processing_time_ms: float = Field(..., description="Cumulative processing time")
    average_processing_time_ms: float = Field(..., description="Average time per file")
    success_rate: float = Field(..., description="Percentage of successful operations (0.0-1.0)")
    total_entities: int = Field(..., description="Number of code entities extracted")
    total_relationships: int = Field(..., description="Number of relationships identified")
    total_lines_of_code: int = Field(..., description="Total lines processed")
    parser_strategy_usage: dict[str, int] = Field(
        ..., description="Breakdown of parser strategies used"
    )
    error_count: int = Field(..., description="Number of processing errors")
    entities_per_file: float = Field(..., description="Average entities extracted per file")
    lines_per_file: float = Field(..., description="Average lines of code per file")
    processing_speed_loc_per_ms: float = Field(
        ..., description="Processing speed in lines per millisecond"
    )


class ParserRankingItem(BaseModel):
    """Individual parser ranking item."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "parser": "tree_sitter",
                "rank": 1,
                "performance_score": 95.5,
                "usage_count": 500,
                "success_rate": 0.98,
                "average_processing_time": 250.0,
                "languages_supported": ["python", "javascript", "typescript"],
            }
        }
    )

    parser: str = Field(..., description="Parser strategy name", pattern=SAFE_STRING_PATTERN)
    rank: int = Field(..., description="Performance ranking (1 = best)")
    performance_score: float = Field(..., description="Calculated efficiency score")
    usage_count: int = Field(..., description="Number of times used")
    success_rate: float = Field(..., description="Percentage of successful operations")
    average_processing_time: float = Field(..., description="Average time per operation")
    languages_supported: list[str] = Field(..., description="Languages this parser handles")


class ParserPerformanceReportResponse(BaseModel):
    """Response model for parser performance comparison."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "parser_rankings": [
                    {
                        "parser": "tree_sitter",
                        "rank": 1,
                        "performance_score": 95.5,
                        "usage_count": 500,
                        "success_rate": 0.98,
                        "average_processing_time": 250.0,
                        "languages_supported": ["python", "javascript"],
                    }
                ],
                "summary": {
                    "total_parsers": 3,
                    "best_performer": "tree_sitter",
                    "total_operations": 1000,
                    "average_success_rate": 0.95,
                },
            }
        }
    )

    parser_rankings: list[ParserRankingItem] = Field(
        ..., description="List of parsers sorted by performance"
    )
    summary: dict[str, Any] = Field(..., description="Overall parser statistics")


class TopPerformingLanguageItem(BaseModel):
    """Individual top performing language item."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "language": "python",
                "average_processing_time_ms": 200.0,
                "success_rate": 0.98,
                "performance_score": 490.0,
                "total_files_processed": 150,
                "total_entities_extracted": 1500,
                "preferred_parser": "tree_sitter",
            }
        }
    )

    rank: int = Field(..., description="Position in performance ranking")
    language: str = Field(
        ..., description="Programming language identifier", pattern=SAFE_STRING_PATTERN
    )
    average_processing_time_ms: float = Field(..., description="Average processing time per file")
    success_rate: float = Field(..., description="Percentage of successful operations (0.0-1.0)")
    performance_score: float = Field(
        ..., description="Calculated performance score (higher = better)"
    )
    total_files_processed: int = Field(
        ..., description="Number of files analyzed for this language"
    )
    total_entities_extracted: int = Field(
        ..., description="Total entities extracted from this language"
    )
    preferred_parser: str | None = Field(
        None,
        description="Most effective parser strategy for this language",
        pattern=SAFE_STRING_PATTERN,
    )


class ExportMetricsResponse(BaseResponse):
    """Response model for metrics export."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Multi-language performance metrics exported successfully",
                "timestamp": "2025-12-25T10:00:00Z",
                "request_id": "req_12345678",
                "export_timestamp": "2025-12-25T10:00:00Z",
                "data": {},
            }
        }
    )

    status: ResponseStatus = ResponseStatus.SUCCESS
    export_timestamp: str | None = Field(
        None,
        description="ISO timestamp of when export was generated",
        pattern=SAFE_STRING_PATTERN,
    )
    data: dict[str, Any] = Field(..., description="Complete metrics data")


class ResetMetricsResponse(BaseResponse):
    """Response model for metrics reset."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "All multi-language performance metrics have been reset",
                "timestamp": "2025-12-25T10:00:00Z",
                "request_id": "req_12345678",
            }
        }
    )

    status: ResponseStatus = ResponseStatus.SUCCESS


# ==================== System Metrics Models ====================


class EntityMetricsResponse(BaseModel):
    """Response model for entity-level centrality metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "centrality_measures": {
                    "degree": 0.8,
                    "betweenness": 0.65,
                    "closeness": 0.72,
                    "pagerank": 0.05,
                },
                "importance_scores": {"overall": 0.75},
                "language_specific": {},
                "timestamp": "2025-12-25T10:00:00Z",
            }
        }
    )

    centrality_measures: dict[str, float] = Field(
        ..., description="Degree, betweenness, closeness, PageRank"
    )
    importance_scores: dict[str, Any] = Field(..., description="Various importance metrics")
    language_specific: dict[str, Any] = Field(..., description="Language-dependent metrics")
    timestamp: str = Field(
        ..., description="When metrics were calculated", pattern=SAFE_STRING_PATTERN
    )


class GraphSummaryResponse(BaseModel):
    """Response model for graph-wide summary metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_counts": {"function": 100, "class": 50, "module": 20},
                "relationship_counts": {"calls": 300, "imports": 150},
                "language_distribution": {"python": 100, "javascript": 70},
                "graph_density": 0.15,
                "analysis_timestamp": "2025-12-25T10:00:00Z",
            }
        }
    )

    entity_counts: dict[str, int] = Field(..., description="Breakdown by entity type")
    relationship_counts: dict[str, int] = Field(..., description="Breakdown by relationship type")
    language_distribution: dict[str, int] = Field(..., description="Entities by language")
    graph_density: float = Field(..., description="Overall graph density")
    analysis_timestamp: str = Field(
        ..., description="When analysis was performed", pattern=SAFE_STRING_PATTERN
    )


class CentralityCalculationResponse(BaseModel):
    """Response model for centrality calculation operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "completed",
                "entities_processed": 250,
                "timestamp": "2025-12-25T10:00:00Z",
            }
        }
    )

    status: str = Field(..., description="Operation status", pattern=SAFE_STRING_PATTERN)
    entities_processed: int = Field(..., description="Count of entities analyzed")
    timestamp: str = Field(
        ..., description="When calculation completed", pattern=SAFE_STRING_PATTERN
    )


class EntityTypeMetricsResponse(BaseModel):
    """Response model for entity type-specific metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_123": {
                    "degree": 0.8,
                    "betweenness": 0.65,
                    "closeness": 0.72,
                    "pagerank": 0.05,
                    "importance": 0.75,
                }
            }
        }
    )

    # This is a dynamic dictionary keyed by entity ID
    # Using __root__ pattern for dynamic dict responses
    __pydantic_extra__: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Entity metrics keyed by entity ID"
    )


# ==================== Analytics Dashboard Models ====================


class TopEntityItem(BaseModel):
    """Individual top entity item."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_name": "main_processor",
                "metric_value": 0.85,
                "entity_type": "function",
                "rank": 1,
            }
        }
    )

    entity_name: str = Field(..., description="Name of the entity", pattern=SAFE_STRING_PATTERN)
    metric_value: float = Field(..., description="Calculated metric value")
    entity_type: str = Field(..., description="Type of entity", pattern=SAFE_STRING_PATTERN)
    rank: int = Field(..., description="Position in the ranking")


class MetricsComparisonResponse(BaseModel):
    """Response model for entity type comparison."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "function": [
                    {
                        "entity_name": "process_data",
                        "metric_value": 0.85,
                        "entity_type": "function",
                        "rank": 1,
                    }
                ],
                "class": [
                    {
                        "entity_name": "DataProcessor",
                        "metric_value": 0.78,
                        "entity_type": "class",
                        "rank": 1,
                    }
                ],
            }
        }
    )

    # Dynamic dict mapping entity types to their top entities
    __pydantic_extra__: dict[str, list[TopEntityItem]] = Field(
        default_factory=dict, description="Entity types mapped to their top entities"
    )


class DashboardSummaryResponse(BaseModel):
    """Response model for comprehensive dashboard summary."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "graph_overview": {},
                "top_entities_by_metric": {
                    "pagerank": [],
                    "betweenness": [],
                    "degree": [],
                    "importance": [],
                },
                "metrics_available": ["pagerank", "betweenness", "degree"],
                "last_updated": "2025-12-25T10:00:00Z",
                "status": "active",
            }
        }
    )

    graph_overview: dict[str, Any] = Field(..., description="Basic graph statistics")
    top_entities_by_metric: dict[str, list[TopEntityItem]] = Field(
        ..., description="Top entities for each major metric"
    )
    metrics_available: list[str] = Field(..., description="Available metrics")
    last_updated: str | None = Field(
        None, description="Last update timestamp", pattern=SAFE_STRING_PATTERN
    )
    status: str = Field(
        ...,
        description="Dashboard status (active/limited)",
        pattern=SAFE_STRING_PATTERN,
    )


class EntityRankingsResponse(BaseModel):
    """Response model for entity rankings across multiple metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rankings": {
                    "pagerank": [
                        {
                            "entity_name": "main",
                            "metric_value": 0.05,
                            "entity_type": "function",
                            "rank": 1,
                        }
                    ]
                },
                "cross_metric_analysis": {"main_processor": ["pagerank", "betweenness"]},
                "summary": {
                    "total_metrics_analyzed": 3,
                    "entities_per_metric": {"pagerank": 10},
                    "cross_metric_entities_count": 5,
                    "most_versatile_entities": [],
                },
                "requested_metrics": ["pagerank", "betweenness"],
            }
        }
    )

    rankings: dict[str, list[TopEntityItem]] = Field(
        ..., description="Rankings for each requested metric"
    )
    cross_metric_analysis: dict[str, list[str]] = Field(
        ..., description="Entities appearing in multiple top lists"
    )
    summary: dict[str, Any] = Field(..., description="Overview of ranking patterns")
    requested_metrics: list[str] = Field(..., description="Metrics that were analyzed")


# ==================== Metrics Aggregation Models ====================


class ContainerEntityMetricsResponse(BaseModel):
    """Response model for container-specific entity metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "centrality_measures": {
                    "degree": 0.8,
                    "betweenness": 0.65,
                    "closeness": 0.72,
                    "pagerank": 0.05,
                },
                "relationship_data": {},
                "container_context": {"container_name": "project_1"},
                "calculation_timestamp": "2025-12-25T10:00:00Z",
                "entity_metadata": {},
            }
        }
    )

    centrality_measures: dict[str, float] = Field(..., description="All centrality scores")
    relationship_data: dict[str, Any] = Field(
        ..., description="Connected entities and relationship types"
    )
    container_context: dict[str, Any] = Field(
        ..., description="Container-specific analysis metadata"
    )
    calculation_timestamp: str = Field(
        ..., description="When metrics were computed", pattern=SAFE_STRING_PATTERN
    )
    entity_metadata: dict[str, Any] = Field(
        ..., description="Additional information about the entity"
    )


class ContainerOverviewMetricsResponse(BaseModel):
    """Response model for container overview metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_info": {"name": "project_1", "status": "running"},
                "graph_statistics": {"total_entities": 250, "total_relationships": 600},
                "centrality_distributions": {},
                "top_entities_summary": {},
                "entity_type_breakdown": {},
                "performance_metrics": {},
                "health_indicators": {},
            }
        }
    )

    container_info: dict[str, Any] = Field(..., description="Container metadata and status")
    graph_statistics: dict[str, Any] = Field(
        ..., description="Entity counts, relationship counts, graph density"
    )
    centrality_distributions: dict[str, Any] = Field(
        ..., description="Statistical distribution of centrality scores"
    )
    top_entities_summary: dict[str, Any] = Field(
        ..., description="Highest-ranking entities across all metrics"
    )
    entity_type_breakdown: dict[str, Any] = Field(
        ..., description="Counts and statistics by entity type"
    )
    performance_metrics: dict[str, Any] = Field(
        ..., description="Container-specific analysis performance data"
    )
    health_indicators: dict[str, Any] = Field(
        ..., description="Graph connectivity and completeness metrics"
    )


class ContainerHealthMetricsResponse(BaseModel):
    """Response model for container health metrics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "project_1",
                "status": "healthy",
                "container_state": "running",
                "last_check": "2025-12-25T10:00:00Z",
                "graph_loaded": True,
                "entity_count": 250,
                "processing_capabilities": {
                    "centrality_calculation": True,
                    "metrics_aggregation": True,
                    "multi_language_support": True,
                },
                "performance_indicators": {
                    "response_time_healthy": True,
                    "data_consistency": True,
                    "processing_queue_status": "normal",
                },
                "health_score": 95.0,
            }
        }
    )

    container_name: str = Field(..., description="Container name", pattern=SAFE_STRING_PATTERN)
    status: str = Field(..., description="Current operational status", pattern=SAFE_STRING_PATTERN)
    container_state: str = Field(..., description="Container state", pattern=SAFE_STRING_PATTERN)
    last_check: str | None = Field(
        None, description="Last health check timestamp", pattern=SAFE_STRING_PATTERN
    )
    graph_loaded: bool = Field(..., description="Whether graph data is loaded")
    entity_count: int = Field(..., description="Number of entities in graph")
    processing_capabilities: dict[str, bool] = Field(..., description="Available analysis features")
    performance_indicators: dict[str, Any] = Field(..., description="Response times and throughput")
    health_score: float = Field(..., description="Overall health score (0-100)")


class ContainerRefreshResponse(BaseModel):
    """Response model for container metrics refresh operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "completed",
                "container_name": "project_1",
                "refresh_timestamp": "2025-12-25T10:00:00Z",
                "entities_processed": 250,
                "metrics_updated": [
                    "degree_centrality",
                    "betweenness_centrality",
                    "closeness_centrality",
                    "pagerank",
                    "importance_score",
                ],
                "operation_summary": "Metrics refresh completed",
            }
        }
    )

    status: str = Field(..., description="Refresh operation status", pattern=SAFE_STRING_PATTERN)
    container_name: str = Field(
        ..., description="Container that was refreshed", pattern=SAFE_STRING_PATTERN
    )
    refresh_timestamp: str | None = Field(
        None, description="When refresh was completed", pattern=SAFE_STRING_PATTERN
    )
    entities_processed: int = Field(..., description="Number of entities recalculated")
    metrics_updated: list[str] = Field(..., description="List of metrics that were refreshed")
    operation_summary: str = Field(
        ..., description="Summary of the operation", pattern=SAFE_STRING_PATTERN
    )


class ContainerComparisonResponse(BaseModel):
    """Response model for multi-container comparison."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "containers_analyzed": ["default"],
                "comparative_metrics": {
                    "default": {
                        "entity_count": {},
                        "relationship_count": {},
                        "graph_density": 0.15,
                        "languages": {},
                    }
                },
                "summary_statistics": {
                    "total_containers": 1,
                    "average_entities_per_container": 250,
                    "most_active_container": "default",
                    "comparison_timestamp": "2025-12-25T10:00:00Z",
                },
                "performance_comparison": {
                    "fastest_container": "default",
                    "largest_graph": "default",
                    "most_diverse_languages": "default",
                },
                "recommendations": [],
            }
        }
    )

    containers_analyzed: list[str] = Field(
        ..., description="List of containers included in comparison"
    )
    comparative_metrics: dict[str, dict[str, Any]] = Field(
        ..., description="Side-by-side metrics comparison"
    )
    summary_statistics: dict[str, Any] = Field(
        ..., description="Aggregate statistics across containers"
    )
    performance_comparison: dict[str, str] = Field(
        ..., description="Container performance analysis"
    )
    recommendations: list[str] = Field(..., description="Insights based on comparison")
