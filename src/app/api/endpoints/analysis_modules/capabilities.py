"""
Capabilities and Status Endpoints

Analysis capabilities discovery and status tracking endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, status

from auth import User
from auth.tenant import get_current_user_with_tenant

from ...models.analysis_api_models import (
    AnalysisCapabilities,
    AnalysisStatusResponse,
    CapabilitiesResponse,
)
from ...models.analysis_api_models import PydanticAIAnalysisType as AnalysisType
from ...models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from .core_analysis import ANALYSIS_LANGUAGES, ANALYSIS_OUTPUTS, ANALYSIS_RUNNERS

logger = logging.getLogger(__name__)


def _any_runner_outputs(part: str) -> bool:
    """Some dispatched runner fills in *part* (``core_analysis.ANALYSIS_OUTPUTS``)."""
    return any(part in ANALYSIS_OUTPUTS.get(kind, frozenset()) for kind in ANALYSIS_RUNNERS)


def compute_analysis_capabilities() -> AnalysisCapabilities:
    """``AnalysisCapabilities`` derived from THE /analyze dispatch table."""
    languages_by_type = {
        str(kind.value): sorted(ANALYSIS_LANGUAGES.get(kind, ())) for kind in ANALYSIS_RUNNERS
    }
    features = {
        "syntax_analysis": bool(languages_by_type.get("ast")),
        "complexity_metrics": _any_runner_outputs("complexity_metrics"),
        "security_scanning": AnalysisType.SECURITY in ANALYSIS_RUNNERS,
        "performance_analysis": AnalysisType.PERFORMANCE in ANALYSIS_RUNNERS,
        # No runner computes test coverage or a technical-debt score.
        "test_coverage": False,
        "technical_debt": False,
        "resolve_suggestions": _any_runner_outputs("suggestions"),
        "cross_file_dependencies": False,  # inline code is a single snippet
    }
    return AnalysisCapabilities(
        supported_types=list(ANALYSIS_RUNNERS),
        supported_languages=sorted(
            {lang for langs in languages_by_type.values() for lang in langs}
        ),
        features=features,
        languages_by_type=languages_by_type,
        language_agnostic_types=[kind for kind, langs in languages_by_type.items() if not langs],
    )


router = APIRouter()


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="📋 Analysis Capabilities",
    description="""
    What ``POST /api/analysis/analyze`` (inline ``code``) actually does, computed from
    its dispatch table (``core_analysis.ANALYSIS_RUNNERS`` / ``ANALYSIS_LANGUAGES``),
    never from a hand-written list.

    - ``supported_types``: every analysis type with a runner.
    - ``languages_by_type``: per type, the languages it has language-SPECIFIC logic
      for. Today ``augmented``, ``ast`` and ``deep`` are Python-only; ``basic`` adds
      a few per-language line rules (python, javascript, typescript, java) to
      language-agnostic line checks.
    - ``language_agnostic_types``: ``security`` and ``performance`` are substring
      heuristics that accept any language and understand none.
    - ``supported_languages``: the union of ``languages_by_type``.
    - ``features``: a flag is true only if some runner produces it; ``test_coverage``
      and ``technical_debt`` are false because no runner computes them. There is no
      cross-file dependency tracking on this route (a snippet is one file).
    """,
    responses={
        200: {"description": "Capabilities retrieved successfully"},
        500: {"model": Error500Response, "description": "Failed to get capabilities"},
    },
    operation_id="get_analysis_capabilities",
)
async def get_analysis_capabilities(
    user: User = Depends(get_current_user_with_tenant),
) -> CapabilitiesResponse:
    """Get information about available analysis capabilities. Requires authentication."""
    try:
        logger.info(f"Getting analysis capabilities for user {user.preferred_username}")
        capabilities = compute_analysis_capabilities()

        return CapabilitiesResponse(
            message="Analysis capabilities retrieved",
            capabilities=capabilities,
        )

    except (RuntimeError, ImportError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capabilities: {str(e)}",
        ) from e


@router.get(
    "/status/{analysis_id}",
    response_model=AnalysisStatusResponse,
    summary="📊 Analysis Status",
    description="""
    🔍 **Track Analysis Progress** - Monitor your analysis operations in real-time

    ## Purpose
    This endpoint allows you to monitor the progress of long-running analysis operations,
    providing real-time status updates, progress indicators, and completion estimates.
    Perfect for tracking deep analysis, repository scans, or complex security audits.

    ## How to Use
    **GET** `/api/analysis/status/{analysis_id}`

    **Basic Status Check:**
    ```bash
    curl -X GET "https://gateway.example.com/api/analysis/status/abc123-def456"
    ```

    ## Parameters
    - **analysis_id** (path, required): The unique identifier for your analysis operation
      - Returned from the `/analyze` endpoint when analysis is started
      - Format: Usually a UUID like "abc123-def456-ghi789"
      - Case-sensitive and must match exactly

    ## Response Examples

    ### 🏃 **Analysis in Progress:**
    ```json
    {
        "message": "Status for analysis 'abc123-def456'",
        "analysis_status": {
            "analysis_id": "abc123-def456",
            "status": "running",
            "progress": 65.0,
            "phase": "analyzing",
            "started_at": "2025-06-02T10:30:00Z",
            "estimated_completion": "2025-06-02T10:32:15Z",
            "details": {
                "current_task": "Security pattern analysis",
                "files_processed": 12,
                "total_files": 18
            }
        },
        "request_id": "abc123-def456"
    }
    ```

    ### ✅ **Analysis Completed:**
    ```json
    {
        "message": "Status for analysis 'abc123-def456'",
        "analysis_status": {
            "analysis_id": "abc123-def456",
            "status": "completed",
            "progress": 100.0,
            "phase": "finished",
            "started_at": "2025-06-02T10:30:00Z",
            "completed_at": "2025-06-02T10:32:08Z",
            "execution_time": 128.5,
            "details": {
                "result_available": true,
                "issues_found": 5,
                "suggestions_generated": 8
            }
        },
        "request_id": "abc123-def456"
    }
    ```

    ### ❌ **Analysis Failed:**
    ```json
    {
        "message": "Status for analysis 'abc123-def456'",
        "analysis_status": {
            "analysis_id": "abc123-def456",
            "status": "failed",
            "progress": 45.0,
            "phase": "parsing",
            "started_at": "2025-06-02T10:30:00Z",
            "failed_at": "2025-06-02T10:31:22Z",
            "details": {
                "error": "Syntax error in input code",
                "error_phase": "parsing",
                "recovery_suggestions": [
                    "Check code syntax and try again",
                    "Verify code is complete and valid"
                ]
            }
        },
        "request_id": "abc123-def456"
    }
    ```

    ### 🔍 **Analysis Not Found:**
    ```json
    {
        "status": "error",
        "message": "Analysis operation 'invalid-id' not found",
        "timestamp": "2025-06-02T11:45:30Z",
        "request_id": "req_analysis404",
        "path": "/api/analysis/status/invalid-id",
        "error_code": "NOT_FOUND",
        "details": {}
    }
    ```

    ## Status Values Explained

    ### 📋 **Status Types**
    - **pending**: Analysis queued but not yet started
    - **running**: Analysis actively in progress
    - **completed**: Analysis finished successfully
    - **failed**: Analysis encountered an error and stopped
    - **cancelled**: Analysis was cancelled by user or system

    ### 🎯 **Analysis Phases**
    - **initializing**: Setting up analysis environment
    - **parsing**: Parsing and validating code syntax
    - **extracting**: Extracting code structure and entities
    - **analyzing**: Performing main analysis logic
    - **generating_report**: Compiling results and suggestions
    - **finished**: Analysis complete and results available

    ### 📊 **Progress Indicators**
    - **progress**: Percentage completion (0.0 to 100.0)
    - **started_at**: ISO timestamp when analysis began
    - **estimated_completion**: Predicted completion time (when available)
    - **execution_time**: Total time taken (for completed analyses)

    ## Monitoring Workflows

    ### 🔄 **Polling Pattern**
    ```javascript
    const checkStatus = async (analysisId) => {
        const response = await fetch(`/api/analysis/status/${analysisId}`);
        const data = await response.json();

        if (data.analysis_status.status === 'running') {
            console.log(`Progress: ${data.analysis_status.progress}%`);
            setTimeout(() => checkStatus(analysisId), 2000); // Poll every 2 seconds
        } else {
            console.log('Analysis complete:', data.analysis_status.status);
        }
    };
    ```

    ### ⏱️ **Recommended Polling Intervals**
    - **Fast analysis** (basic, augmented): Every 1-2 seconds
    - **Deep analysis**: Every 3-5 seconds
    - **Repository analysis**: Every 5-10 seconds
    - **Security audits**: Every 3-5 seconds

    ## Best Practices

    ### ✅ **Do This**
    - Store analysis_id immediately after starting analysis
    - Implement reasonable polling intervals based on analysis type
    - Handle both success and failure cases gracefully
    - Show progress indicators to users for long-running operations
    - Cache status results briefly to avoid excessive API calls

    ### ❌ **Avoid This**
    - Polling more frequently than every second
    - Ignoring failed analysis status
    - Polling indefinitely without timeout
    - Making concurrent status requests for the same analysis_id
    - Blocking UI while waiting for analysis completion

    ## Error Handling

    ### 🔍 **Not Found (404)**
    - Analysis ID doesn't exist or has expired
    - Check that you're using the correct analysis_id
    - Analysis results may have been cleaned up

    ### ⚠️ **Server Error (500)**
    - Analysis service temporarily unavailable
    - Retry after a brief delay
    - Check system status if problems persist
    """,
    responses={
        200: {"description": "Analysis status retrieved successfully"},
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Analysis not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
    operation_id="get_analysis_status",
)
async def get_analysis_status(
    analysis_id: str = Path(..., description="Analysis operation ID"),
    user: User = Depends(get_current_user_with_tenant),
) -> AnalysisStatusResponse:
    """Get the status of a specific analysis operation. Requires authentication."""
    try:
        logger.info(f"Getting analysis status {analysis_id} for user {user.preferred_username}")
        # This import is now local to the function to break the circular dependency
        from services.analysis_status import get_analysis_status_service

        status_service = get_analysis_status_service()
        status_info = await status_service.get_status(analysis_id)

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis operation '{analysis_id}' not found",
            )

        return AnalysisStatusResponse(
            message=f"Status for analysis '{analysis_id}'",
            analysis_status=status_info,
            request_id=analysis_id,
        )

    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Return 404 for any lookup failure (Redis down, import error, etc.)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis operation '{analysis_id}' not found",
        ) from e
