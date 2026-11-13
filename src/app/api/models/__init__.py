"""API v2 models package.

This package contains all Pydantic models for API v2 requests and responses.
"""

# Analysis models
from .analysis_api_models import (
    AnalysisMetrics,
    AnalysisRequest,
    ApiAnalysisResponse,
    ApiAnalysisResult,
)

# Cache models
from .cache_api_models import (
    CacheClearResponse,
    CacheKeysResponse,
    CacheStatsResponse,
)

# Common models
from .common_api_models import (
    BaseResponse,
    PaginationParams,
    SuccessResponse,
)

# Container models
from .container_api_models import (
    ContainerDetailResponse,
    ContainerListResponse,
    ContainerOperation,
    ContainerOperationRequest,
    ContainerOperationResponse,
    ContainerStatusResponse,
)

# Error models
from .error_models import (  # Backward compatibility aliases
    APIError,
    APIValidationErrorDetail,
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error405Response,
    Error406Response,
    Error408Response,
    Error409Response,
    Error422Response,
    Error429Response,
    Error500Response,
    Error503Response,
    InternalServerError,
    NotFoundError,
    StandardErrorResponse,
    ValidationError,
    ValidationErrorResponse,
)

# GitHub models
from .github_api_models import (
    CreatePRRequest,
    CreatePRResponse,
    ForkRequest,
    ForkResponse,
    RepositoryInfo,
)

# Graph models
from .graph_api_models import (
    EdgesResponse,
    GraphDeletionResponse,
    GraphDetailResponse,
    GraphEdge,
    GraphInfo,
    GraphListResponse,
    GraphNode,
    GraphQuery,
    GraphQueryResponse,
    NodesResponse,
)

# Health models
from .health_api_models import (
    ContainerHealth,
    ContainerHealthResponse,
    DatabaseHealth,
    DetailedHealthResponse,
    HealthCheckResponse,
    MetricsResponse,
    ResourceMetrics,
    SystemHealth,
)

# Intelligent Testing models
from .intelligent_testing_models import (
    ContainerScopedSmartTestSelectionRequest,
    ContainerScopedTestPatternRequest,
    ContainerScopedTestRecommendationRequest,
    SmartTestSelectionResponse,
    TestImpactAnalysisRequest,
    TestImpactAnalysisResponse,
    TestPatternAnalysisResponse,
    TestPrioritizationRequest,
    TestPrioritizationResponse,
    TestRecommendationResponse,
)

# Metrics response models
from .metrics_response_models import (
    CentralityCalculationResponse,
    ContainerComparisonResponse,
    ContainerEntityMetricsResponse,
    ContainerHealthMetricsResponse,
    ContainerOverviewMetricsResponse,
    ContainerRefreshResponse,
    DashboardSummaryResponse,
    EntityMetricsResponse,
    EntityRankingsResponse,
    EntityTypeMetricsResponse,
    ExportMetricsResponse,
    GraphSummaryResponse,
    LanguagePerformanceStatsResponse,
    ParserPerformanceReportResponse,
    ParserRankingItem,
    PerformanceSummaryResponse,
    ResetMetricsResponse,
    TopEntityItem,
    TopPerformingLanguageItem,
)

# Repository scanner models
from .repo_scanner_api_models import (
    APIScanAction,
    ApiValidationResult,
    RepoScannerDependencyScanResponse,
    RepoScannerImportInfo,
    RepoScannerImportScanResponse,
    RepoScannerScanRequest,
    RepoScannerScanResponse,
    RepoScannerValidationScanResponse,
    ScannedDependency,
    VersionInfo,
    VersionScanResponse,
)

# Validation models
from .validation_api_models import (
    CheckClassConsistencyResponse,
    CheckConsistencyResponse,
    CheckFunctionConsistencyResponse,
    ValidateClassReferenceRequest,
    ValidateClassReferenceResponse,
    ValidateCodeSnippetRequest,
    ValidateCodeSnippetResponse,
    ValidateFileReferenceRequest,
    ValidateFileReferenceResponse,
    ValidateFunctionReferenceRequest,
    ValidateFunctionReferenceResponse,
)

# Base validation models
from .validation_base_models import (
    BaseValidationRequest,
    BaseValidationResponse,
    ValidationIssue,
    ValidationMetrics,
)
from .validation_config_models import (
    ValidationConfig,
)

# WebSocket models
from .websocket_models import (
    RealtimeAnalysisRequest,
    RealtimeAnalysisResponse,
    RealtimeErrorResponse,
)

# PRValidationRequest, CodeValidationRequest, PRValidationResponse, CodeValidationResponse were removed - they don't exist
# AnalysisReport and ImpactAnalysis were removed - they don't exist

__all__ = [
    # Analysis
    "AnalysisRequest",
    "ApiAnalysisResponse",
    "ApiAnalysisResult",
    "AnalysisMetrics",
    # Cache
    "CacheStatsResponse",
    "CacheClearResponse",
    "CacheKeysResponse",
    # Common
    "BaseResponse",
    "PaginationParams",
    "SuccessResponse",
    # Container
    "ContainerOperationRequest",
    "ContainerOperationResponse",
    "ContainerListResponse",
    "ContainerDetailResponse",
    "ContainerStatusResponse",
    "ContainerOperation",
    # Error
    "StandardErrorResponse",
    "ValidationErrorResponse",
    "Error404Response",
    "Error500Response",
    "APIValidationErrorDetail",
    "Error400Response",
    "Error401Response",
    "Error403Response",
    "Error405Response",
    "Error406Response",
    "Error408Response",
    "Error409Response",
    "Error422Response",
    "Error429Response",
    "Error503Response",
    # Backward compatibility aliases
    "APIError",
    "ValidationError",
    "NotFoundError",
    "InternalServerError",
    # GitHub
    "ForkRequest",
    "ForkResponse",
    "CreatePRRequest",
    "CreatePRResponse",
    "RepositoryInfo",
    # Graph
    "GraphQuery",
    "GraphListResponse",
    "GraphDetailResponse",
    "GraphNode",
    "GraphEdge",
    "GraphInfo",
    "NodesResponse",
    "EdgesResponse",
    "GraphQueryResponse",
    "GraphDeletionResponse",
    # Health
    "SystemHealth",
    "ResourceMetrics",
    "DatabaseHealth",
    "ContainerHealth",
    "ContainerHealthResponse",
    "HealthCheckResponse",
    "DetailedHealthResponse",
    "MetricsResponse",
    # Repository scanner
    "RepoScannerScanRequest",
    "RepoScannerScanResponse",
    "RepoScannerDependencyScanResponse",
    "RepoScannerImportScanResponse",
    "RepoScannerValidationScanResponse",
    "VersionScanResponse",
    "ApiValidationResult",
    "ScannedDependency",
    "RepoScannerImportInfo",
    "VersionInfo",
    "APIScanAction",
    # Validation
    "ValidateFileReferenceRequest",
    "ValidateFunctionReferenceRequest",
    "ValidateClassReferenceRequest",
    "ValidateCodeSnippetRequest",
    "ValidateFileReferenceResponse",
    "ValidateFunctionReferenceResponse",
    "ValidateClassReferenceResponse",
    "ValidateCodeSnippetResponse",
    "CheckConsistencyResponse",
    "CheckFunctionConsistencyResponse",
    "CheckClassConsistencyResponse",
    "ValidationConfig",
    "ValidationMetrics",
    "BaseValidationRequest",
    "BaseValidationResponse",
    "ValidationIssue",
    # WebSocket
    "RealtimeAnalysisRequest",
    "RealtimeAnalysisResponse",
    "RealtimeErrorResponse",
    # Intelligent Testing
    "ContainerScopedTestPatternRequest",
    "ContainerScopedTestRecommendationRequest",
    "ContainerScopedSmartTestSelectionRequest",
    "TestPatternAnalysisResponse",
    "TestRecommendationResponse",
    "SmartTestSelectionResponse",
    "TestImpactAnalysisRequest",
    "TestImpactAnalysisResponse",
    "TestPrioritizationRequest",
    "TestPrioritizationResponse",
    # Metrics response models
    "CentralityCalculationResponse",
    "ContainerComparisonResponse",
    "ContainerEntityMetricsResponse",
    "ContainerHealthMetricsResponse",
    "ContainerOverviewMetricsResponse",
    "ContainerRefreshResponse",
    "DashboardSummaryResponse",
    "EntityMetricsResponse",
    "EntityRankingsResponse",
    "EntityTypeMetricsResponse",
    "ExportMetricsResponse",
    "GraphSummaryResponse",
    "LanguagePerformanceStatsResponse",
    "ParserPerformanceReportResponse",
    "ParserRankingItem",
    "PerformanceSummaryResponse",
    "ResetMetricsResponse",
    "TopEntityItem",
    "TopPerformingLanguageItem",
]
