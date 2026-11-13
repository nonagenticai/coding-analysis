"""Repository Scanner API Endpoints Module.

This module contains the FastAPI endpoint definitions for repository scanning functionality.
It handles HTTP request/response processing and orchestrates scanning operations.

The module provides:
- Main scan_repository endpoint with comprehensive documentation
- Request validation and error handling
- Response model mapping based on scan type
- Integration with core scanning functionality
"""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.container_manager import get_container_manager
from auth.tenant import require_tenant
from models.repo_scanner_models import RepoScanAction

from ...models.common_api_models import ResponseStatus
from ...models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
)
from ...models.repo_scanner_api_models import (
    RepoScannerDependencyScanResponse,
    RepoScannerImportScanResponse,
    RepoScannerScanRequest,
    RepoScannerScanResponse,
    RepoScannerValidationScanResponse,
)
from .scanner_core import (
    scan_dependencies,
    scan_imports,
    validate_code,
    validate_scanned_imports,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/scan",
    # Simplified to avoid Union TypeAdapter issues - using Any for now
    response_model=None,
    operation_id="scan_repository",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid scan action or parameters",
        },
        404: {"model": Error404Response, "description": "Container not found"},
        500: {
            "model": Error500Response,
            "description": "Internal server error during scanning",
        },
    },
    summary="🔍 Repository Code Scanner",
    description="""
    🚀 **Advanced Repository Analysis** - Comprehensive code scanning and analysis for dockerized repositories

    ## Purpose
    This powerful endpoint provides deep analysis of code repositories running in Docker containers.
    Performs various types of scans including dependency analysis, import validation, code quality checks,
    and version compatibility assessment. Essential for understanding code health, dependencies, and potential issues.

    ## Available Scan Actions

    ### 📦 **Dependencies Analysis** (`"dependencies"`)
    - **Purpose**: Analyze project dependencies and package management
    - **Checks**: pip packages, poetry dependencies, requirements files
    - **Outputs**: Package names, versions, sources, validity status
    - **Use Cases**: Dependency auditing, license compliance, security scanning

    ### 📥 **Import Analysis** (`"imports"`)
    - **Purpose**: Analyze Python import statements across codebase
    - **Checks**: Module imports, from imports, import aliases
    - **Outputs**: Import details, file locations, validity checks
    - **Use Cases**: Dependency mapping, refactoring planning, unused import detection

    ### ✅ **Code Validation** (`"validation"`)
    - **Purpose**: Validate Python code syntax and basic quality
    - **Checks**: Syntax errors, parsing issues, file structure
    - **Outputs**: Validation status, error details, line counts
    - **Use Cases**: Code quality assurance, CI/CD pipeline checks

    ### 🔬 **Full Scan** (`"full_scan"`)
    - **Purpose**: Comprehensive analysis combining all scan types
    - **Checks**: Everything above in one operation
    - **Outputs**: Complete repository health report
    - **Use Cases**: Complete repository audits, health assessments
    """,
)
async def scan_repository(
    request: RepoScannerScanRequest,
    tenant_id: str = Depends(require_tenant),
) -> (
    RepoScannerDependencyScanResponse
    | RepoScannerImportScanResponse
    | RepoScannerValidationScanResponse
    | RepoScannerScanResponse
):
    """Perform comprehensive repository scanning and analysis in dockerized environments.

    This is the main API endpoint for repository analysis, providing multiple
    scanning modes for different aspects of code analysis. All operations are
    performed in secure, isolated Docker containers to ensure reproducible
    results and security isolation.

    The endpoint supports four distinct scanning modes:
    1. Dependencies: Analyze package dependencies and requirements
    2. Imports: Analyze Python import statements and relationships
    3. Validation: Validate code syntax and basic quality metrics
    4. Full Scan: Combined analysis of all the above

    Args:
        request (RepoScannerScanRequest): Scan configuration containing:
            - container_name (str): Target Docker container name
            - action (str): Scan type ('dependencies', 'imports', 'validation', 'full_scan')
            - target_files (Optional[List[str]]): Specific files to analyze (for imports/validation)

    Returns:
        Union[RepoScannerDependencyScanResponse, RepoScannerImportScanResponse, RepoScannerValidationScanResponse, RepoScannerScanResponse]:
        Response type depends on the requested action:

        RepoScannerDependencyScanResponse (for 'dependencies'):
            - dependencies: List of found dependencies with versions and sources
            - dependency_count: Total number of dependencies
            - invalid_dependencies: Dependencies that couldn't be resolved

        RepoScannerImportScanResponse (for 'imports'):
            - imports: List of import statements with metadata
            - import_count: Total number of imports found
            - invalid_imports: Imports that failed validation

        RepoScannerValidationScanResponse (for 'validation'):
            - validation_results: Per-file validation results
            - valid_files: Number of syntactically correct files
            - invalid_files: Number of files with issues

        RepoScannerScanResponse (for 'full_scan'):
            - results: Combined results from all scan types
            - Includes summary statistics and comprehensive analysis

    HTTP Status Codes:
        200: Scan completed successfully
        400: Invalid request (bad action, container not running)
        404: Container not found
        422: Request validation failed
        500: Internal server error during scanning

    Raises:
        HTTPException:
            - 400: If scan action is invalid or container not running
            - 404: If specified container doesn't exist
            - 500: If scanning process encounters unexpected errors

    Container Requirements:
        - Container must exist and be in 'running' status
        - Container should have Python runtime for code analysis
        - Container must have accessible /workspace directory
        - Sufficient permissions for script execution

    Security Features:
        - All operations performed in isolated containers
        - No direct filesystem access from API server
        - Script execution with restricted privileges
        - Comprehensive input validation and sanitization

    Example Usage:
        ```python
        # Dependency scan
        request = RepoScannerScanRequest(
            container_name="my-python-app",
            action="dependencies"
        )

        # Import analysis with specific files
        request = RepoScannerScanRequest(
            container_name="my-python-app",
            action="imports",
            target_files=["src/main.py", "src/utils.py"]
        )

        # Full repository analysis
        request = RepoScannerScanRequest(
            container_name="my-python-app",
            action="full_scan"
        )
        ```

    Performance Considerations:
        - Full scans may take longer on large repositories
        - Target file specification improves performance for focused analysis
        - Results include timing information for performance monitoring
        - Container resource usage depends on repository size

    Error Recovery:
        - Individual file failures don't prevent overall scan completion
        - Graceful handling of missing dependencies or malformed files
        - Detailed error reporting for troubleshooting
        - Fallback behaviors for partial scan results
    """
    start_time = datetime.now(UTC)

    try:
        container_name = request.container_name
        action = request.action

        logger.info(f"Starting {action} scan for container: {container_name}")

        # Validate container exists and is running
        container_manager = get_container_manager()
        containers = await container_manager.list_containers(tenant_id=tenant_id)
        container_info = None

        for container in containers:
            if container_name == container.container_name:
                container_info = container
                break

        if not container_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container '{container_name}' not found",
            )

        if container_info.status != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Container '{container_name}' is not running (status: {container_info.status})",
            )

        # Perform the requested scan
        scan_results = {}

        if action == RepoScanAction.DEPENDENCIES:
            scan_results = await scan_dependencies(container_name, tenant_id=tenant_id)

            end_time = datetime.now(UTC)
            scan_duration = (end_time - start_time).total_seconds()

            # A failed dependency scan returns an ``error`` key with an empty
            # dependency list. Surface that as a degraded/error result instead of
            # reporting an empty scan as a clean success (silent degradation).
            scan_error = scan_results.get("error")
            if scan_error:
                logger.warning("Dependency scan for %s degraded: %s", container_name, scan_error)

            return RepoScannerDependencyScanResponse(
                status=ResponseStatus.ERROR if scan_error else ResponseStatus.SUCCESS,
                message=(
                    f"Dependency scan degraded: {scan_error}"
                    if scan_error
                    else "Dependency scan completed successfully"
                ),
                container_name=container_name,
                action=action,
                scan_duration=scan_duration,
                file_count=0,
                dependencies=scan_results.get("dependencies", []),
                dependency_count=scan_results.get("dependency_count", 0),
                invalid_dependencies=scan_results.get("invalid_dependencies", []),
                request_id=str(hash(f"scan_{action}_{container_name}")),
            )

        elif action == RepoScanAction.IMPORTS:
            scan_results = await scan_imports(
                container_name, request.target_files, tenant_id=tenant_id
            )

            # Validate imports using the ImportValidationService
            validated_results = await validate_scanned_imports(
                container_name, scan_results.get("imports", []), tenant_id=tenant_id
            )

            end_time = datetime.now(UTC)
            scan_duration = (end_time - start_time).total_seconds()

            return RepoScannerImportScanResponse(
                status=ResponseStatus.SUCCESS,
                message="Import scan completed successfully",
                container_name=container_name,
                action=action,
                scan_duration=scan_duration,
                file_count=(len(request.target_files) if request.target_files else 0),
                imports=validated_results.get("imports", []),
                import_count=validated_results.get("import_count", 0),
                invalid_imports=validated_results.get("invalid_imports", []),
                request_id=str(hash(f"scan_{action}_{container_name}")),
            )

        elif action == RepoScanAction.VALIDATION:
            scan_results = await validate_code(
                container_name, request.target_files, tenant_id=tenant_id
            )

            end_time = datetime.now(UTC)
            scan_duration = (end_time - start_time).total_seconds()

            return RepoScannerValidationScanResponse(
                status=ResponseStatus.SUCCESS,
                message="Code validation completed successfully",
                container_name=container_name,
                action=action,
                scan_duration=scan_duration,
                file_count=len(scan_results.get("validation_results", [])),
                validation_results=scan_results.get("validation_results", []),
                valid_files=scan_results.get("valid_files", 0),
                invalid_files=scan_results.get("invalid_files", 0),
                request_id=str(hash(f"scan_{action}_{container_name}")),
            )

        elif action == RepoScanAction.FULL_SCAN:
            # Perform all scan types
            deps_results = await scan_dependencies(container_name, tenant_id=tenant_id)
            imports_results = await scan_imports(
                container_name, request.target_files, tenant_id=tenant_id
            )
            validation_results = await validate_code(
                container_name, request.target_files, tenant_id=tenant_id
            )

            end_time = datetime.now(UTC)
            scan_duration = (end_time - start_time).total_seconds()

            # Surface any per-scan errors so a degraded full scan is not read as
            # a clean success (silent degradation).
            scan_errors = {
                name: res["error"]
                for name, res in (
                    ("dependencies", deps_results),
                    ("imports", imports_results),
                    ("validation", validation_results),
                )
                if isinstance(res, dict) and res.get("error")
            }
            if scan_errors:
                logger.warning("Full scan for %s degraded: %s", container_name, scan_errors)

            # Combine all results
            combined_results = {
                "dependencies": deps_results,
                "imports": imports_results,
                "validation": validation_results,
                "errors": scan_errors,
                "summary": {
                    "total_dependencies": deps_results.get("dependency_count", 0),
                    "total_imports": imports_results.get("import_count", 0),
                    "valid_files": validation_results.get("valid_files", 0),
                    "invalid_files": validation_results.get("invalid_files", 0),
                },
            }

            return RepoScannerScanResponse(
                status=ResponseStatus.ERROR if scan_errors else ResponseStatus.SUCCESS,
                message=(
                    f"Full repository scan degraded: {', '.join(scan_errors)} scan(s) failed"
                    if scan_errors
                    else "Full repository scan completed successfully"
                ),
                container_name=container_name,
                action=action,
                scan_duration=scan_duration,
                file_count=len(validation_results.get("validation_results", [])),
                results=combined_results,
                request_id=str(hash(f"scan_{action}_{container_name}")),
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scan action: {action}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during %s scan: %s", request.action, str(e))
        logger.error(f"Traceback: {traceback.format_exc()}")

        end_time = datetime.now(UTC)
        scan_duration = (end_time - start_time).total_seconds()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}",
        )
