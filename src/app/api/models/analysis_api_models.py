"""Analysis api models module."""

from __future__ import annotations

"""Code Analysis API Models.

This module contains Pydantic models for all code analysis operations in the V2 API.
These models support various types of analysis including basic syntax checking, security
scanning, performance analysis, and deep code inspection.

The models provide:
    - Type-safe request validation with comprehensive field documentation
    - Standardized analysis result formats across all analysis types
    - Batch processing capabilities for multiple code samples
    - Progress tracking for long-running analysis operations
    - Structured issue reporting with severity levels and remediation suggestions

Example:
    ```python
    analysis_request = AnalysisRequest(
        analysis_type=PydanticAIAnalysisType.SECURITY,
        code="import pickle\ndata = pickle.load(file)",
        language="python"
    )
    ```
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from processing.workspace_repos import RepoSelector

from ...types import SAFE_STRING_PATTERN
from .common_api_models import BaseResponse, ResponseStatus


class PydanticAIAnalysisType(StrEnum):
    """Enumeration of supported code analysis types.

    This enum defines all available analysis modes that can be performed on code.
    Each type provides different levels of inspection and focuses on specific
    code quality aspects.

    Attributes:
        BASIC: Basic syntax and structure validation
        AUGMENTED: Extended analysis with additional context
        AST: Abstract syntax tree analysis for deep code understanding
        DEEP: Comprehensive analysis including complexity and maintainability
        SECURITY: Security vulnerability scanning and best practices
        PERFORMANCE: Performance bottleneck detection and optimization suggestions
    """

    BASIC = "basic"
    AUGMENTED = "augmented"
    AST = "ast"
    DEEP = "deep"
    SECURITY = "security"
    PERFORMANCE = "performance"


class AnalysisRequest(BaseModel):
    """Request model for initiating code analysis operations.

    This model defines the structure for requesting various types of code analysis.
    It supports both direct code analysis and container-based repository analysis.

    Attributes:
        analysis_type (PydanticAIAnalysisType): The type of analysis to perform
        code (Optional[str]): Source code to analyze directly
        container_name (str): Container name for repository-based analysis (required)
        language (Optional[str]): Programming language for language-specific analysis
        options (Optional[Dict[str, Any]]): Analysis-specific configuration options

    Note:
        container_name is required and must not be empty.
        When 'container_name' is used, analysis is performed on the entire repository.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        protected_namespaces=(),
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "analysis_type": "security",
                "container_name": "my-python-project",
                "language": "python",
                "options": {
                    "check_imports": True,
                    "check_sql_injection": True,
                },
            }
        },
    )

    analysis_type: PydanticAIAnalysisType = Field(..., description="Type of analysis to perform")
    code: str | None = Field(None, description="Code to analyze (for direct analysis)")
    container_name: str = Field(
        ...,
        min_length=1,
        description="Container name for repository context (required)",
    )
    language: str | None = Field(None, description="Programming language")
    options: dict[str, Any] | None = Field(None, description="Analysis-specific options")

    @field_validator("container_name")
    @classmethod
    def validate_container_name(cls, v: str) -> str:
        """Validate that container_name is not empty after stripping whitespace.

        Args:
            v: The container name to validate

        Returns:
            str: The validated container name

        Raises:
            ValueError: If container_name is empty or contains only whitespace
        """
        if not v or not v.strip():
            raise ValueError("container_name must not be empty")
        return v.strip()


class BatchAnalysisRequest(RepoSelector):
    """Request model for batch analysis of files in a container.

    This model defines the structure for requesting batch analysis operations
    on multiple files within a container, providing efficient processing with
    configurable options and performance optimization.

    Attributes:
        container_name (str): Container name containing files to analyze
        file_paths (Optional[list[str]]): Specific file paths to analyze (optional)
        analysis_type (str): Type of analysis to perform on all files
        batch_options (Optional[Dict[str, Any]]): Batch processing configuration

    Note:
        If file_paths is not provided, all supported files in the container will be analyzed.
        Batch options can control concurrency, timeouts, and result aggregation.

    Example:
        ```python
        # Analyze all files in container
        batch_request = BatchAnalysisRequest(
            container_name="my-python-project",
            analysis_type="security",
            batch_options={
                "max_concurrent": 5,
                "timeout_per_file": 30,
                "aggregate_results": True
            }
        )

        # Analyze specific files
        specific_batch = BatchAnalysisRequest(
            container_name="web-app",
            file_paths=["src/main.py", "src/auth.py", "src/api.py"],
            analysis_type="deep",
            batch_options={"max_concurrent": 3}
        )
        ```
    """

    container_name: str = Field(..., description="Container name to analyze")
    file_paths: list[str] | None = Field(
        None,
        description="Specific file paths within container to analyze (optional, analyzes all files if not provided)",
    )
    analysis_type: str = Field("augmented", description="Type of analysis to perform")
    batch_options: dict[str, Any] | None = Field(None, description="Batch processing options")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-python-project",
                "file_paths": [
                    "/app/data/repo/src/main.py",
                    "/app/data/repo/src/utils.py",
                    "/app/data/repo/tests/test_main.py",
                ],
                "analysis_type": "augmented",
                "batch_options": {
                    "max_concurrent": 5,
                    "timeout_per_file": 30,
                    "include_relationships": True,
                    "aggregate_results": True,
                },
            },
        }
    )


class AnalysisIssue(BaseModel):
    """Represents a single code issue discovered during analysis.

    This model encapsulates all information about a specific problem or concern
    found in the analyzed code, including location details and severity assessment.

    Attributes:
        type (str): Category of issue (syntax, security, performance, etc.)
        severity (str): Issue severity level (low, medium, high, critical)
        message (str): Human-readable description of the issue
        line (int): Line number where the issue occurs
        column (int): Column position of the issue (defaults to 1)
        file (str): File path containing the issue
        rule (Optional[str]): Specific rule or check that identified the issue

    Example:
        ```python
        issue = AnalysisIssue(
            type="security",
            severity="high",
            message="Potential SQL injection vulnerability",
            line=42,
            file="database.py",
            rule="B608"
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "security",
                "severity": "high",
                "message": "Potential security risk: Using pickle to deserialize data from untrusted sources",
                "line": 6,
                "column": 16,
                "file": "data_loader.py",
                "rule": "B301",
            }
        }
    )

    type: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Issue type (e.g., 'syntax', 'security', 'performance')",
    )
    severity: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Issue severity ('low', 'medium', 'high', 'critical')",
    )
    message: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Human-readable issue description"
    )
    line: int = Field(..., description="Line number where issue occurs")
    column: int = Field(1, description="Column number where issue occurs")
    file: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="File path where issue was found"
    )
    rule: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Rule ID that triggered the issue",
    )


class AnalysisMetrics(BaseModel):
    """Quantitative metrics derived from code analysis.

    This model aggregates various quantitative measurements about the analyzed code,
    providing insights into code quality, complexity, and maintainability.

    Attributes:
        lines_of_code (int): Total lines of code analyzed
        complexity (float): Cyclomatic complexity score indicating code complexity
        maintainability (float): Maintainability index score (0-100, higher is better)
        test_coverage (Optional[float]): Test coverage percentage if available
        technical_debt (Optional[float]): Estimated technical debt in hours

    Note:
        All numeric values are validated to ensure they fall within expected ranges.
        Maintainability scores range from 0 (lowest) to 100 (highest quality).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lines_of_code": 150,
                "complexity": 12.5,
                "maintainability": 72.3,
                "test_coverage": 85.5,
                "technical_debt": 4.2,
            }
        }
    )

    lines_of_code: int = Field(default=0, ge=0, description="Total lines of code")
    complexity: float = Field(default=0.0, ge=0.0, description="Cyclomatic complexity score")
    maintainability: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maintainability index (0-100)",
    )
    test_coverage: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Test coverage percentage"
    )
    technical_debt: float | None = Field(
        default=None, ge=0.0, description="Technical debt estimation in hours"
    )


class ApiAnalysisResult(BaseModel):
    """Comprehensive result of a code analysis operation.

    This model encapsulates the complete output of an analysis session,
    including all discovered issues, calculated metrics, and recommendations.

    Attributes:
        analysis_type (PydanticAIAnalysisType): Type of analysis that was performed
        success (bool): Whether the analysis completed without errors
        issues (list[AnalysisIssue]): All issues discovered during analysis
        metrics (AnalysisMetrics): Quantitative code quality metrics
        suggestions (list[str]): Actionable recommendations for code updates
        execution_time (float): Analysis duration in seconds
        metadata (Optional[Dict[str, Any]]): Additional analysis-specific data

    Note:
        A successful analysis (success=True) may still contain issues.
        The success field indicates the analysis process completed, not code quality.
    """

    analysis_type: PydanticAIAnalysisType = Field(..., description="Type of analysis performed")
    success: bool = Field(..., description="Whether analysis completed successfully")
    issues: list[AnalysisIssue] = Field(
        default_factory=list, description="Issues found during analysis"
    )
    metrics: AnalysisMetrics = Field(
        default_factory=AnalysisMetrics, description="Code quality metrics"
    )
    suggestions: list[str] = Field(default_factory=list, description="Update suggestions")
    execution_time: float = Field(..., ge=0.0, description="Analysis execution time in seconds")
    metadata: dict[str, Any] | None = Field(None, description="Additional analysis metadata")

    model_config = ConfigDict(use_enum_values=True)


class ApiAnalysisResponse(BaseResponse):
    """Response from analysis operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: ApiAnalysisResult = Field(..., description="Analysis result data")


class AnalysisCapabilities(BaseModel):
    """System capabilities for code analysis operations.

    This model describes what analysis types and languages are supported
    by the current system configuration, along with available features.

    Attributes:
        supported_types (list[PydanticAIAnalysisType]): Analysis types available in the system
        supported_languages (list[str]): Programming languages that can be analyzed
        features (Dict[str, bool]): Feature flags indicating available capabilities

    Example:
        ```python
        capabilities = AnalysisCapabilities(
            supported_types=[PydanticAIAnalysisType.BASIC, PydanticAIAnalysisType.SECURITY],
            supported_languages=["python", "javascript", "go"],
            features={"batch_analysis": True, "real_time": False}
        )
        ```
    """

    supported_types: list[PydanticAIAnalysisType] = Field(
        ..., description="List of supported analysis types"
    )
    supported_languages: list[str] = Field(
        ...,
        description=(
            "Languages at least one analysis type has language-SPECIFIC logic for "
            "(the union of languages_by_type)"
        ),
    )
    features: dict[str, bool] = Field(..., description="Available feature flags")
    languages_by_type: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per analysis type, the languages it has language-specific logic for. "
            "An empty list = language-agnostic text heuristics only."
        ),
    )
    language_agnostic_types: list[str] = Field(
        default_factory=list,
        description="Analysis types that are text heuristics and understand no language",
    )

    model_config = ConfigDict(use_enum_values=True)


class CapabilitiesResponse(BaseResponse):
    """Response for analysis capabilities query."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    capabilities: AnalysisCapabilities = Field(..., description="Analysis system capabilities")


class AnalysisStatus(StrEnum):
    """Enumeration of possible analysis operation states.

    This enum tracks the lifecycle of analysis operations, particularly
    useful for long-running or asynchronous analysis tasks.

    Attributes:
        PENDING: Analysis request received but not yet started
        RUNNING: Analysis is currently in progress
        COMPLETED: Analysis finished successfully
        FAILED: Analysis terminated due to an error
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisPhase(StrEnum):
    """Analysis operation phases for detailed progress tracking.

    This enum defines the specific phases that an analysis operation
    goes through during execution, providing granular progress tracking.

    Attributes:
        INITIALIZING: Setting up analysis environment and parameters
        PARSING: Parsing and tokenizing the input code
        EXTRACTING: Extracting code structure and metadata
        ANALYZING: Performing the core analysis operations
        GENERATING_REPORT: Compiling results and generating output
        FINALIZING: Cleanup and final result preparation
    """

    INITIALIZING = "initializing"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    GENERATING_REPORT = "generating_report"
    FINALIZING = "finalizing"


class AnalysisStatusInfo(BaseModel):
    """Detailed status information for tracking analysis operations.

    This model provides comprehensive tracking information for analysis operations,
    including progress indicators and timing information.

    Attributes:
        analysis_id (str): Unique identifier for the analysis operation
        status (AnalysisStatus): Current state of the analysis
        progress (float): Completion percentage (0.0 to 100.0)
        estimated_completion (Optional[str]): ISO timestamp of estimated completion
        started_at (Optional[str]): ISO timestamp when analysis began
        completed_at (Optional[str]): ISO timestamp when analysis finished
        metadata (Optional[Dict[str, Any]]): Additional operation-specific data

    Note:
        Timestamps are provided in ISO 8601 format with timezone information.
        Progress is only meaningful for RUNNING status operations.
    """

    analysis_id: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Unique analysis operation identifier",
    )
    status: AnalysisStatus = Field(..., description="Current operation status")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
    estimated_completion: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="ISO timestamp of estimated completion",
    )
    started_at: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="ISO timestamp when analysis started",
    )
    completed_at: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="ISO timestamp when analysis completed",
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional status metadata")

    model_config = ConfigDict(use_enum_values=True)


class AnalysisStatusResponse(BaseResponse):
    """Response for analysis status query."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    analysis_status: AnalysisStatusInfo = Field(
        ..., description="Current analysis operation status"
    )


# Additional utility models for extended functionality


class AnalysisFilter(BaseModel):
    """Filter criteria for refining analysis results.

    This model allows users to filter and focus on specific subsets of analysis
    results based on various criteria such as severity or file location.

    Attributes:
        severity_levels (Optional[list[str]]): Include only issues with these severities
        issue_types (Optional[list[str]]): Include only these types of issues
        file_patterns (Optional[list[str]]): Include only files matching these patterns
        line_range (Optional[Dict[str, int]]): Filter by line number range

    Example:
        ```python
        filter_criteria = AnalysisFilter(
            severity_levels=["high", "critical"],
            file_patterns=["*.py", "src/**/*.js"]
        )
        ```
    """

    severity_levels: list[str] | None = Field(None, description="Filter by severity levels")
    issue_types: list[str] | None = Field(None, description="Filter by issue types")
    file_patterns: list[str] | None = Field(None, description="Filter by file patterns")
    line_range: dict[str, int] | None = Field(None, description="Filter by line range (start, end)")


class AnalysisBatch(BaseModel):
    """Request model for analyzing multiple code samples in a single operation.

    This model enables efficient batch processing of multiple analysis requests,
    reducing overhead and providing consolidated results.

    Attributes:
        items (list[AnalysisRequest]): Individual analysis requests to process
        batch_options (Optional[Dict[str, Any]]): Configuration for batch processing

    Note:
        Batch processing may be performed in parallel for improved performance.
        Each item in the batch is processed according to its individual settings.

    Example:
        ```python
        batch = AnalysisBatch(
            items=[
                AnalysisRequest(
                    analysis_type="basic",
                    container_name="my-python-project",
                    code="def hello(): pass",
                ),
                AnalysisRequest(analysis_type="security", container_name="app")
            ],
            batch_options={"parallel": True, "timeout": 300}
        )
        ```
    """

    items: list[AnalysisRequest] = Field(..., description="List of analysis requests")
    batch_options: dict[str, Any] | None = Field(None, description="Batch-specific options")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "analysis_type": "basic",
                        "container_name": "my-python-project",
                        "code": "def add(a, b):\n    return a + b",
                        "language": "python",
                    },
                    {
                        "analysis_type": "security",
                        "container_name": "my-webapp",
                        "language": "javascript",
                    },
                ],
                "batch_options": {"parallel": True, "timeout": 300},
            }
        },
    )


class AnalysisBatchResult(BaseModel):
    """Comprehensive results from a batch analysis operation.

    This model aggregates results from multiple analysis operations performed
    as part of a single batch request, providing both individual and summary data.

    Attributes:
        batch_id (str): Unique identifier for the batch operation
        total_items (int): Total number of analysis requests in the batch
        completed_items (int): Number of successfully completed analyses
        failed_items (int): Number of analyses that failed to complete
        results (list[ApiAnalysisResult]): Individual analysis results
        summary (AnalysisMetrics): Aggregated metrics across all analyses
        execution_time (float): Total time for the entire batch operation

    Note:
        The summary metrics represent averages or totals across all successful
        analyses in the batch. Failed analyses are excluded from summary calculations.
    """

    batch_id: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Unique batch identifier",
    )
    total_items: int = Field(..., ge=0, description="Total number of items in batch")
    completed_items: int = Field(..., ge=0, description="Number of completed items")
    failed_items: int = Field(..., ge=0, description="Number of failed items")
    results: list[ApiAnalysisResult] = Field(
        default_factory=list, description="Individual analysis results"
    )
    summary: AnalysisMetrics = Field(
        default_factory=AnalysisMetrics, description="Aggregated metrics"
    )
    execution_time: float = Field(..., ge=0.0, description="Total batch execution time")


class AnalysisBatchResponse(BaseResponse):
    """Response for batch analysis operation."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: AnalysisBatchResult = Field(..., description="Batch analysis results")


# Bug Hunter Specific Models


class BugCategory(StrEnum):
    """Enumeration of bug categories for bug hunting operations.

    This enum categorizes different types of bugs that can be detected
    during static analysis, providing detailed classification for better
    remediation and prevention strategies.

    Attributes:
        BUG: Standard software bug or defect
        SECURITY: Security vulnerability or unsafe code pattern
        PERFORMANCE: Performance bottleneck or inefficiency
        LOGIC: Logic error or incorrect algorithm implementation
        EDGE_CASE: Edge case or corner case not properly handled
    """

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    EDGE_CASE = "edge_case"


class BugDetail(BaseModel):
    """Detailed information about a detected bug.

    This model provides comprehensive information about a specific bug discovered
    during analysis, including its location, severity, classification, and
    recommended fix strategies.

    Attributes:
        category (BugCategory): Bug classification category
        severity (str): Severity level (low, medium, high, critical)
        file_path (str): Path to the file containing the bug
        line_number (int): Line number where the bug occurs
        description (str): Detailed description of the bug
        suggested_fix (str): Recommended fix for the bug
        confidence (float): Confidence score of detection (0.0 to 1.0)

    Example:
        ```python
        bug = BugDetail(
            category=BugCategory.SECURITY,
            severity="high",
            file_path="src/auth.py",
            line_number=42,
            description="SQL injection vulnerability in user query",
            suggested_fix="Use parameterized queries instead of string interpolation",
            confidence=0.95
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "security",
                "severity": "high",
                "file_path": "src/database.py",
                "line_number": 25,
                "description": "Potential SQL injection vulnerability detected",
                "suggested_fix": "Use prepared statements or ORM for database queries",
                "confidence": 0.92,
            }
        }
    )

    category: BugCategory = Field(..., description="Bug classification category")
    severity: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Bug severity ('low', 'medium', 'high', 'critical')",
    )
    file_path: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Path to file containing the bug"
    )
    line_number: int = Field(..., ge=1, description="Line number where bug occurs")
    description: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Detailed bug description"
    )
    suggested_fix: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Recommended fix for the bug"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")


class BugFixResult(BaseModel):
    """Result of bug fix generation and application.

    This model represents the outcome of attempting to generate and apply
    a fix for a detected bug, including the fix content and validation status.

    Attributes:
        bug_id (str): Unique identifier for the bug
        fix_applied (bool): Whether the fix was successfully applied
        fix_content (str): The generated fix code or description
        validation_status (str): Status of fix validation (pending, valid, invalid)

    Example:
        ```python
        result = BugFixResult(
            bug_id="bug_12345",
            fix_applied=True,
            fix_content="query = db.execute(sql, (user_id,))",
            validation_status="valid"
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bug_id": "bug_001",
                "fix_applied": True,
                "fix_content": "cursor.execute(query, (user_id,))",
                "validation_status": "valid",
            }
        }
    )

    bug_id: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Unique bug identifier",
    )
    fix_applied: bool = Field(..., description="Whether the fix was successfully applied")
    fix_content: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="The generated fix code or description",
    )
    validation_status: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Fix validation status (pending, valid, invalid)",
    )


class BugHunterStatusResponse(BaseModel):
    """Status information for bug hunting operations.

    This model provides status and metrics information for the bug hunter system,
    including container context, operation timing, and bug statistics.

    Attributes:
        container_name (str): Name of the container being analyzed
        last_hunt_timestamp (Optional[str]): ISO timestamp of last hunting operation
        bugs_found_count (int): Total number of bugs found
        bugs_fixed_count (int): Total number of bugs fixed
        analysis_status (str): Current analysis status (idle, running, completed, failed)

    Example:
        ```python
        status = BugHunterStatusResponse(
            container_name="my-app",
            last_hunt_timestamp="2025-12-08T10:30:00Z",
            bugs_found_count=12,
            bugs_fixed_count=8,
            analysis_status="completed"
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "python-project",
                "last_hunt_timestamp": "2025-12-08T15:45:30Z",
                "bugs_found_count": 15,
                "bugs_fixed_count": 10,
                "analysis_status": "completed",
            }
        }
    )

    container_name: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Container name being analyzed",
    )
    last_hunt_timestamp: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="ISO timestamp of last hunting operation",
    )
    bugs_found_count: int = Field(default=0, ge=0, description="Total number of bugs found")
    bugs_fixed_count: int = Field(default=0, ge=0, description="Total number of bugs fixed")
    analysis_status: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Current analysis status (idle, running, completed, failed)",
    )
