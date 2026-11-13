"""Repository scanner models for comprehensive code analysis operations.

This module provides data models for the repository scanning and analysis API endpoint,
supporting various types of code analysis including dependency scanning, import analysis,
and code validation. The models ensure consistent request/response structures for
comprehensive code analysis operations in containerized environments.

The repository scanner supports:
- Dependency analysis with validation and security checking
- Import statement analysis and resolution
- Code validation across multiple file types
- Full repository scanning with configurable options
- Detailed error reporting and issue tracking

Key Model Categories:
    Request Models: ScanRequest with configurable action types and options
    Response Models: Specialized responses for different scan types
    Data Models: DependencyValidation, ImportInfo, ValidationError for scan results
    Union Types: ScanResponseUnion for type-safe response handling

Scan Types:
    - DEPENDENCIES: Analyze package dependencies and versions
    - IMPORTS: Scan import statements and module usage
    - VALIDATION: Validate code syntax and structure
    - FULL_SCAN: Comprehensive analysis combining all scan types

Example:
    ```python
    from repo_scanner_models import RepoScannerScanRequest, RepoScanAction, RepoScannerDependencyScanResponse

    # Create scan request
    request = RepoScannerScanRequest(
        container_name="my-analysis-container",
        action=RepoScanAction.DEPENDENCIES,
        target_files=["src/main.py", "app/models.py"],
        options={"include_dev_dependencies": True}
    )

    # Process scan response
    response = RepoScannerDependencyScanResponse(
        container_name="my-analysis-container",
        action="dependencies",
        scan_duration=2.5,
        file_count=15,
        request_id="scan-123",
        dependencies=[
            DependencyValidation(
                name="fastapi",
                version="0.104.1",
                source="pip",
                valid=True
            )
        ],
        dependency_count=42,
        invalid_dependencies=[]
    )
    ```
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .common_responses_models import LegacyBaseResponse

# ================================================================================================
# 📋 REQUEST MODELS
# ================================================================================================


class RepoScanAction(StrEnum):
    """Available scan action types."""

    DEPENDENCIES = "dependencies"
    IMPORTS = "imports"
    VALIDATION = "validation"
    FULL_SCAN = "full_scan"


class RepoScannerScanRequest(BaseModel):
    """Request model for repository scanning operations."""

    container_name: str = Field(
        description="Name of the Docker container to scan",
        examples=["my-repo-container-123"],
    )

    action: RepoScanAction = Field(
        description="Type of scan to perform",
        examples=[RepoScanAction.DEPENDENCIES],
    )

    target_files: list[str] | None = Field(
        default=None,
        description="Specific files to scan (optional, defaults to all Python files)",
        examples=[["src/main.py", "app/models.py"]],
    )

    options: dict[str, Any] | None = Field(
        default=None,
        description="Additional scan options and parameters",
        examples=[{"max_files": 100, "include_tests": True}],
    )


# ================================================================================================
# 📤 RESPONSE MODELS
# ================================================================================================


class DependencyValidation(BaseModel):
    """Information about a single dependency."""

    name: str = Field(description="Package name", examples=["fastapi"])
    version: str = Field(description="Package version", examples=["0.104.1"])
    source: str = Field(description="Source of dependency info", examples=["pip"])
    valid: bool = Field(description="Whether the dependency is valid", examples=[True])


class RepoScannerImportInfo(BaseModel):
    """Information about a single import statement."""

    type: str = Field(description="Type of import", examples=["from_import"])
    module: str = Field(description="Module name", examples=["fastapi"])
    name: str = Field(description="Imported name", examples=["FastAPI"])
    alias: str | None = Field(default=None, description="Import alias", examples=["app"])
    file: str = Field(description="File containing the import", examples=["src/main.py"])
    line: int = Field(description="Line number", examples=[1])


class ScannerValidationError(BaseModel):
    """Code validation error information."""

    type: str = Field(..., description="Error type", examples=["SyntaxError"])
    message: str = Field(..., description="Error message", examples=["invalid syntax"])
    line: int | None = Field(None, description="Line number", examples=[10])
    offset: int | None = Field(None, description="Character offset", examples=[5])


class ValidationWarning(BaseModel):
    """Code validation warning information."""

    type: str = Field(..., description="Warning type", examples=["LongLine"])
    message: str = Field(
        ...,
        description="Warning message",
        examples=["Line too long (130 characters)"],
    )
    line: int = Field(..., description="Line number", examples=[15])


class RepoScannerValidationResult(BaseModel):
    """Validation result for a single file."""

    file: str = Field(description="File path", examples=["src/main.py"])
    valid: bool = Field(description="Whether the file is valid", examples=[True])
    syntax_valid: bool | None = Field(
        default=None, description="Whether syntax is valid", examples=[True]
    )
    errors: list[ScannerValidationError] = Field(default=[], description="Validation errors")
    warnings: list[ValidationWarning] = Field(default=[], description="Validation warnings")
    line_count: int = Field(description="Number of lines in file", examples=[150])


class BaseScanResponse(LegacyBaseResponse):
    """Base response model for all scan operations."""

    container_name: str = Field(description="Container name that was scanned")
    action: str = Field(description="Type of scan performed")
    scan_duration: float = Field(description="Scan duration in seconds", examples=[2.5])
    file_count: int = Field(description="Number of files processed", examples=[25])
    request_id: str = Field(description="Unique request identifier")


class RepoScannerDependencyScanResponse(BaseScanResponse):
    """Response model for dependency scan operations."""

    dependencies: list[DependencyValidation] = Field(
        default=[], description="List of discovered dependencies"
    )

    dependency_count: int = Field(description="Total number of dependencies found", examples=[42])

    invalid_dependencies: list[dict[str, Any]] = Field(
        default=[], description="List of invalid or problematic dependencies"
    )


class RepoScannerImportScanResponse(BaseScanResponse):
    """Response model for import scan operations."""

    imports: list[RepoScannerImportInfo] = Field(
        default=[], description="List of discovered import statements"
    )

    import_count: int = Field(description="Total number of imports found", examples=[156])

    invalid_imports: list[dict[str, Any]] = Field(
        default=[], description="List of invalid or problematic imports"
    )


class RepoScannerValidationScanResponse(BaseScanResponse):
    """Response model for code validation operations."""

    validation_results: list[RepoScannerValidationResult] = Field(
        default=[], description="Detailed validation results for each file"
    )

    valid_files: int = Field(..., description="Number of valid files", examples=[20])

    invalid_files: int = Field(..., description="Number of invalid files", examples=[2])


class RepoScannerScanResponse(BaseScanResponse):
    """Response model for full scan operations."""

    results: dict[str, Any] = Field(
        ...,
        description="Combined results from all scan types",
        examples=[
            {
                "dependencies": {"dependency_count": 42, "dependencies": []},
                "imports": {"import_count": 156, "imports": []},
                "validation": {"valid_files": 20, "invalid_files": 2},
                "summary": {
                    "total_dependencies": 42,
                    "total_imports": 156,
                    "valid_files": 20,
                    "invalid_files": 2,
                },
            }
        ],
    )


# ================================================================================================
# 🔄 UNION RESPONSE TYPE
# ================================================================================================

ScanResponseUnion = (
    RepoScannerDependencyScanResponse
    | RepoScannerImportScanResponse
    | RepoScannerValidationScanResponse
    | RepoScannerScanResponse
)
