"""
Bug Hunter Endpoint

Clean, modern bug detection and analysis API for identifying and fixing code issues.
Provides comprehensive bug hunting and analysis operations with proper error handling.

API Paths:
- POST /api/bug-hunter/{container_name}/hunt - Queue a bug hunt
- GET /api/bug-hunter/{container_name}/hunts/{hunt_id}/status - Poll one hunt
- GET /api/bug-hunter/{container_name}/hunts/{hunt_id}/report - Get its summary
- GET /api/bug-hunter/{container_name}/hunts/{hunt_id}/findings - Query bugs by path
- POST /api/bug-hunter/{container_name}/hunts/{hunt_id}/findings/{finding_id}/recheck
  - Re-run the detector on one finding's file at the hunted commit
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid7

import httpx
import redis.exceptions
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from app.api.models.bug_hunter_api_models import (
    BugHuntAcceptedResponse,
    BugHunterAnalysisStatus,
    BugHunterStartHuntRequest,
    BugHunterStatusResponse,
    BugHuntFinding,
    BugHuntFindingRecheckResponse,
    BugHuntFlaggedFile,
    BugHuntJobStatus,
    BugHuntPathFindingsResponse,
    BugHuntRequest,
    BugHuntStatusResponse,
    BugHuntSummaryResponse,
)
from app.api.models.common_api_models import SuccessResponse
from app.api.responses import (
    bad_request,
    conflict,
    error_response,
    internal_error,
    not_found,
    validation_error,
)
from app.core.container_manager import get_container_manager
from app.core.redis_service import get_redis_service
from auth.tenant import require_tenant
from config.settings import get_settings
from models.analysis_models import (
    UNRATED_SEVERITY,
    BugHuntingRequest,
    BugHuntingResponse,
    ErrorDetailDict,
    FixDetailDict,
)
from models.bug_models import ErrorType, ParsedError
from services.bug_analysis import BugAnalyzer
from services.bug_hunt_report_store import BugHuntReportStore, normalize_query_path
from services.bug_hunter import BugHunter
from services.workspace_hunt import (
    WORKSPACE_ROOT,
    HuntUnverifiableError,
    RecheckUnverifiableError,
    WorkspaceHuntResult,
    expected_agent_sha,
    run_workspace_hunt,
    run_workspace_recheck,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory bug hunting state tracker (fallback when Redis is unavailable)
# Format: {container_name: {status_info}}
_bug_hunt_state: dict[str, dict[str, Any]] = {}
_bug_hunt_lock = asyncio.Lock()

_BUG_HUNT_REDIS_PREFIX = "bug_hunt_state:"
_BUG_HUNT_REDIS_TTL = 86400  # 24 hours


def _default_hunt_state(container_name: str) -> dict[str, Any]:
    """Return the default hunt state for a container."""
    return {
        "container_name": container_name,
        "last_hunt_timestamp": None,
        "bugs_found_count": 0,
        "bugs_fixed_count": 0,
        "analysis_status": BugHunterAnalysisStatus.IDLE.value,
        "last_error": None,
        "total_bugs_found": 0,
        "total_bugs_fixed": 0,
    }


async def _get_or_initialize_hunt_state(container_name: str) -> dict[str, Any]:
    """
    Get or initialize the bug hunt state for a container.

    Tries Redis first, falls back to in-memory dict if Redis is unavailable.

    Args:
        container_name: Name of the container

    Returns:
        Dictionary containing the hunt state for the container
    """
    # Try Redis first
    try:
        redis_service = get_redis_service()
        redis_key = f"{_BUG_HUNT_REDIS_PREFIX}{container_name}"
        cached = await redis_service.get(redis_key)
        if cached:
            state = json.loads(cached)
            # Also keep in-memory copy in sync
            async with _bug_hunt_lock:
                _bug_hunt_state[container_name] = state
            return state
    except (
        ConnectionError,
        redis.exceptions.ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
    ) as e:
        logger.debug("Redis not available for bug hunt state: %s", e)

    # Fall back to in-memory state
    async with _bug_hunt_lock:
        if container_name not in _bug_hunt_state:
            _bug_hunt_state[container_name] = _default_hunt_state(container_name)
        return _bug_hunt_state[container_name]


async def _persist_hunt_state(container_name: str, state: dict[str, Any]) -> None:
    """Persist bug hunt state to Redis (best-effort) and in-memory dict.

    Args:
        container_name: Name of the container
        state: The state dict to persist
    """
    # Always update in-memory
    async with _bug_hunt_lock:
        _bug_hunt_state[container_name] = state

    # Best-effort Redis persistence
    try:
        redis_service = get_redis_service()
        redis_key = f"{_BUG_HUNT_REDIS_PREFIX}{container_name}"
        await redis_service.set(
            redis_key,
            json.dumps(state, default=str),
            ex=_BUG_HUNT_REDIS_TTL,
        )
    except (
        ConnectionError,
        redis.exceptions.ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
    ) as e:
        logger.debug("Failed to persist bug hunt state to Redis: %s", e)


async def get_bug_hunter_status(
    container_name: str = Path(..., min_length=1, description="Name of the container"),
    tenant_id: str = Depends(require_tenant),
) -> BugHunterStatusResponse:
    """Get the current bug hunting status for a container.

    Args:
        container_name: Name of the container to check status for

    Returns:
        BugHunterStatusResponse with current status information

    Raises:
        HTTPException: If container is not found or other errors occur
    """
    try:
        logger.info(f"Getting bug hunter status for container: {container_name}")

        # Verify the container exists with tenant validation
        try:
            container_manager = get_container_manager()
            container = await container_manager.get_container(container_name, tenant_id=tenant_id)
            if not container:
                logger.warning(f"Container not found: {container_name}")
                raise not_found("Container", container_name)
        except ValueError:
            logger.warning(f"Container not found: {container_name}")
            raise not_found("Container", container_name)

        # Get or initialize the hunt state
        state = await _get_or_initialize_hunt_state(container_name)

        # Create response
        response = BugHunterStatusResponse(
            container_name=state["container_name"],
            last_hunt_timestamp=state["last_hunt_timestamp"],
            bugs_found_count=state["bugs_found_count"],
            bugs_fixed_count=state["bugs_fixed_count"],
            analysis_status=state["analysis_status"],
            last_error=state["last_error"],
            total_bugs_found=state["total_bugs_found"],
            total_bugs_fixed=state["total_bugs_fixed"],
        )

        logger.info(
            f"Successfully retrieved status for {container_name}: "
            f"status={response.analysis_status}, bugs_found={response.bugs_found_count}"
        )
        return response

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error(f"Error getting bug hunter status: {e}")
        raise internal_error(f"Error retrieving bug hunter status: {str(e)}")


async def start_bug_hunt(
    container_name: str = Path(..., min_length=1, description="Name of the container"),
    request: BugHunterStartHuntRequest | None = None,
    tenant_id: str = Depends(require_tenant),
) -> SuccessResponse:
    """Start a bug hunting session for a container.

    Args:
        container_name: Name of the container to hunt bugs in
        request: Hunt configuration parameters

    Returns:
        SuccessResponse indicating hunt has been started

    Raises:
        HTTPException: If container is not found or other errors occur
    """
    try:
        logger.info(f"Starting bug hunt for container: {container_name}")

        if request is None:
            request = BugHunterStartHuntRequest()

        # Verify the container exists with tenant validation
        try:
            container_manager = get_container_manager()
            container = await container_manager.get_container(container_name, tenant_id=tenant_id)
            if not container:
                logger.warning(f"Container not found: {container_name}")
                raise not_found("Container", container_name)
        except ValueError:
            logger.warning(f"Container not found: {container_name}")
            raise not_found("Container", container_name)

        # Update the hunt state
        state = await _get_or_initialize_hunt_state(container_name)
        state["analysis_status"] = BugHunterAnalysisStatus.HUNTING.value
        state["last_hunt_timestamp"] = datetime.now(UTC).isoformat()
        state["bugs_found_count"] = 0
        state["bugs_fixed_count"] = 0
        state["last_error"] = None
        await _persist_hunt_state(container_name, state)

        logger.info(
            f"Bug hunt started for {container_name} with analysis_type={request.analysis_type}"
        )

        return SuccessResponse(message=f"Bug hunting session started for {container_name}")

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error(f"Error starting bug hunt: {e}")
        raise internal_error(f"Error starting bug hunt: {str(e)}")


def get_bug_analyzer() -> BugAnalyzer:
    """Dependency to get BugAnalyzer instance.

    Returns:
        BugAnalyzer: Service instance for bug analysis
    """
    return BugAnalyzer()


async def hunt_bugs(
    container_name: str = Path(
        ...,
        min_length=1,
        description="Name of the container to analyze",
        examples=["fastapi-backend"],
    ),
    request: BugHuntingRequest | None = None,
    bug_analyzer: BugAnalyzer = Depends(get_bug_analyzer),
    tenant_id: str = Depends(require_tenant),
    hunt_id: str | None = None,
) -> BugHuntingResponse:
    """Hunt for bugs in a container's codebase.

    Analyzes error logs and the container's codebase to identify bugs
    and generate automated fixes with comprehensive assessment.

    Args:
        container_name: Name of the container to analyze
        request: BugHuntingRequest with error logs and options
        bug_analyzer: Injected BugAnalyzer service instance

    Returns:
        BugHuntingResponse containing:
        - Identified errors with context
        - Generated fixes with explanations
        - Root cause and risk analysis
        - Effort estimation
        - Optional GitHub PR information

    Raises:
        HTTPException:
            - 400: Invalid request or bad parameters
            - 404: Container not found
            - 500: Analysis failed or internal error

    Examples:
        Basic analysis:
        >>> response = await hunt_bugs("my-container", request)
        >>> print(f"Found {response.errors_found} errors")
        >>> print(f"Generated {response.fixes_generated} fixes")

        Check fix confidence:
        >>> for fix in response.resolves:
        ...     if fix.confidence > 0.8:
        ...         print(f"High confidence fix: {fix.description}")
    """
    import time

    # Validate request
    if request is None:
        logger.warning("Bug hunting request received with no body")
        raise bad_request("Request body required for bug hunting")

    # Validate container exists with tenant validation
    try:
        container_manager = get_container_manager()
        try:
            container = await container_manager.get_container(container_name, tenant_id=tenant_id)
            if container is None:
                raise not_found("Container", container_name)
            logger.debug(
                f"Container validation successful: {container_name} for tenant: {tenant_id}"
            )
        except ValueError as e:
            logger.warning(f"Container validation failed: {container_name} - {e}")
            raise not_found("Container", container_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error validating container {container_name}: {e}")
        raise internal_error("Failed to validate container")

    # Log the request
    logger.info(
        f"Bug hunting initiated for container '{container_name}' "
        f"with {len(request.error_logs or '')} bytes of error logs"
    )

    try:
        # Start timer for analysis duration
        start_time = time.time()

        error_logs = _decode_error_logs(request)

        # Prepare analysis context if provided
        context: dict[str, Any] = dict(request.context) if request.context else {}  # type: ignore[no-matching-overload]
        context.setdefault("container_name", container_name)
        context.setdefault("error_timestamp", str(time.time()))

        bug_hunter_service: BugHunter | None = None
        workspace_hunt: WorkspaceHuntResult | None = None
        github_token: str | None = None
        if not error_logs.strip():
            # A codebase scan runs INSIDE the tenant workspace (the hunt agent);
            # only verified findings come back. This pod has no checkout.
            workspace_hunt = await run_workspace_hunt(
                container_name=container_name,
                tenant_id=tenant_id,
                hunt_id=hunt_id or str(uuid7()),
            )
            # No fix generation: proposing a fix needs the file's source, and
            # the source never leaves the workspace.
            analyzer = (
                bug_analyzer if hasattr(bug_analyzer, "analyze_bugs_and_fixes") else BugAnalyzer()
            )
            analysis_result = analyzer.analyze_bugs_and_fixes(
                [_finding_to_parsed_error(finding) for finding in workspace_hunt.findings], []
            )
        else:
            # Reuse the already-awaited, tenant-validated container.
            codebase_path = getattr(container, "work_dir", None) or f"/tmp/{container_name}"  # nosec B108

            # Get GitHub token if PR creation requested
            github_token = request.GITHUB_ACCESS_TOKEN if request.create_pr else None

            # Error-log analysis: parsing + fix generation, no repository scan.
            bug_hunter_service = BugHunter(codebase_path=codebase_path, github_token=github_token)
            analysis_result = await asyncio.to_thread(
                bug_hunter_service.analyze_error_logs, error_logs, context
            )

        # Convert ParsedError objects to ErrorDetailDict for response
        if workspace_hunt is not None and len(analysis_result.errors) != len(
            workspace_hunt.findings
        ):
            raise HuntUnverifiableError(
                "unparseable_result",
                f"{len(analysis_result.errors)} errors for {len(workspace_hunt.findings)} findings",
            )
        error_details: list[ErrorDetailDict] = []
        for i, err in enumerate(analysis_result.errors):
            error_detail: ErrorDetailDict = {
                "error_id": f"err_{i}",
                "error_type": (
                    err.error_type.value
                    if hasattr(err.error_type, "value")
                    else str(err.error_type)
                ),
                "error_message": err.message,
                "stack_trace": str(err.stack_trace) if err.stack_trace else "",
                "file_path": err.file_path or "",
                "line_number": err.line_number or 0,
                "function_name": err.function_name or "",
                "error_category": (
                    err.error_type.value
                    if hasattr(err.error_type, "value")
                    else str(err.error_type)
                ),
                # ParsedError carries no severity, so there is nothing to rate it
                # from. "unrated" says so; a literal "medium" was a constant
                # presented as data (coding-analysis#112).
                "severity": UNRATED_SEVERITY,
                "timestamp": str(time.time()),
                "frequency": 1,
                "impact_score": err.confidence,
                "suggested_fix": "",
                "related_errors": [],
            }
            if workspace_hunt is not None:
                # analyze_bugs_and_fixes keeps the agent's findings in order, one
                # error per finding, so index i is finding i (checked below).
                error_detail["fingerprint"] = str(workspace_hunt.findings[i]["fingerprint"])
            error_details.append(error_detail)

        # Convert CodeFix objects to FixDetailDict for response
        fix_details: list[FixDetailDict] = []
        for i, fix in enumerate(analysis_result.resolves):
            fix_detail: FixDetailDict = {
                "fix_id": f"fix_{i}",
                "fix_type": "code_fix",
                "file_path": fix.file_path,
                "line_number": fix.line_number or 0,
                "original_code": fix.original_code,
                "fixed_code": fix.fixed_code,
                "description": fix.explanation,
                "confidence": fix.confidence,
                "risk_level": "low" if fix.confidence > 0.7 else "medium",
                "test_impact": fix.test_suggestion or "",
                "dependencies": [],
                "validation_steps": [],
                "rollback_plan": "Revert to original code",
            }
            fix_details.append(fix_detail)

        # Handle PR creation if requested
        pr_info = None
        if bug_hunter_service and request.create_pr and request.github_repo and github_token:
            try:
                repo_parts = request.github_repo.split("/")
                if len(repo_parts) == 2:
                    pr_result = await bug_hunter_service.create_pull_request(
                        repo_owner=repo_parts[0],
                        repo_name=repo_parts[1],
                        resolves=analysis_result.resolves,
                    )
                    if pr_result.get("success"):
                        pr_info = pr_result.get("data")
            except Exception as e:
                logger.warning(f"Failed to create PR: {e}")

        analysis_time = time.time() - start_time
        logger.info(
            f"Bug hunting analysis completed for '{container_name}' "
            f"in {analysis_time:.2f} seconds: "
            f"{len(analysis_result.errors)} errors, {len(analysis_result.resolves)} fixes"
        )

        response = BugHuntingResponse(
            errors_found=len(analysis_result.errors),
            fixes_generated=len(analysis_result.resolves),
            root_cause_analysis=analysis_result.root_cause_analysis,
            fix_summary=analysis_result.fix_summary,
            estimated_effort=analysis_result.estimated_effort,
            risk_assessment=analysis_result.risk_assessment,
            errors=error_details,
            resolves=fix_details,
            pull_request=pr_info,
            analysis_time=analysis_time,
            # Coverage provenance is measured only by the workspace scan, by the
            # same agent process that read the bytes. A log-only hunt examined
            # no tree, so every field stays None (unmeasured).
            scanned_files=workspace_hunt.scanned_files if workspace_hunt else None,
            tracked_source_files=(workspace_hunt.tracked_source_files if workspace_hunt else None),
            commit_sha=workspace_hunt.commit_sha if workspace_hunt else None,
            worktree_dirty=workspace_hunt.worktree_dirty if workspace_hunt else None,
            excluded_dirs=workspace_hunt.excluded_dirs if workspace_hunt else None,
            excluded_entry_kinds=(workspace_hunt.excluded_entry_kinds if workspace_hunt else None),
            agent_sha=workspace_hunt.agent_sha if workspace_hunt else None,
        )

        logger.debug(
            f"Bug hunting response prepared: errors={response.errors_found}, fixes={response.fixes_generated}"
        )

        return response

    except (HTTPException, HuntUnverifiableError):
        # HTTP errors as-is; an unverifiable hunt keeps its precise reason so
        # the worker records it verbatim as the failure.
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error during bug hunting for container '{container_name}': {e}"
        )
        raise internal_error(f"Bug hunting analysis failed: {str(e)}")


def _decode_error_logs(request: BugHuntingRequest) -> str:
    """The request's error logs as text (base64 form decoded); ``""`` when none."""
    error_logs = request.error_logs or ""
    if request.error_logs_base64:
        try:
            import base64

            error_logs = base64.b64decode(request.error_logs_base64).decode("utf-8")
            logger.debug("Decoded base64 error logs")
        except Exception as e:
            logger.warning(f"Failed to decode base64 error logs, using as plain text: {e}")
            error_logs = request.error_logs_base64
    return error_logs


def _finding_to_parsed_error(finding: dict[str, Any]) -> ParsedError:
    """One agent finding as a ``ParsedError``; a malformed one fails the hunt."""
    try:
        line = finding.get("line_number")
        return ParsedError(
            error_type=ErrorType(finding["error_type"]),
            message=str(finding["message"]),
            file_path=str(finding["file_path"]),
            line_number=line if isinstance(line, int) and not isinstance(line, bool) else None,
            function_name=None,
            stack_trace=[],
            confidence=float(finding["confidence"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HuntUnverifiableError("unparseable_result", f"finding {exc!s:.200}") from exc


def get_bug_hunt_report_store() -> BugHuntReportStore:
    """Return the stateless PostgreSQL report store dependency."""
    return BugHuntReportStore()


CALLBACK_TIMEOUT_S = 10.0
CALLBACK_ATTEMPTS = 3
CALLBACK_URL_MAX_LEN = 2048

# What may follow the configured prefix: plain path segments only (an n8n resume
# URL is ``<prefix><executionId>`` with an optional ``/<suffix>``). No dots, no
# percent-encoding, no backslashes, no empty segments. This is the traversal
# guard: httpx collapses ``.``/``..`` segments before sending, so
# ``<prefix>../webhook/x`` passes a raw startswith() and is POSTed to
# ``/webhook/x`` — an unauthenticated n8n production webhook.
_CALLBACK_TAIL = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")


def callback_url_allowed(url: str | None) -> bool:
    """True iff ``url`` is an http(s) URL under a configured callback prefix.

    Fail closed: no configured prefixes means no callback is ever allowed. The
    origin (scheme + host:port) must match a prefix EXACTLY — a raw string prefix
    would let ``http://h:5678`` admit ``http://h:56789`` or ``http://h:5678@evil``
    — and the path must be the prefix's path followed only by plain segments
    (see ``_CALLBACK_TAIL``). A query string (n8n's resume token) is allowed; a
    fragment is not.
    """
    if not url or len(url) > CALLBACK_URL_MAX_LEN:
        return False
    # Whitespace/control characters: urlsplit strips some of them, so the check
    # would pass a URL httpx then refuses to send (a hunt queued for nothing).
    if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:  # e.g. "http://[::1/x" — a malformed URL is a 422, never a 500
        return False
    if (
        parts.scheme not in ("http", "https")
        or not parts.netloc
        or "@" in parts.netloc
        or parts.fragment
    ):
        return False
    for prefix in get_settings().bug_hunt_callback_prefixes:
        allowed = urlsplit(prefix)
        base = allowed.path if allowed.path.endswith("/") else allowed.path + "/"
        if (
            parts.scheme == allowed.scheme
            and parts.netloc.lower() == allowed.netloc.lower()
            and parts.path.startswith(base)
            and _CALLBACK_TAIL.fullmatch(parts.path[len(base) :])
        ):
            return True
    return False


async def _send_completion_callback(job: dict[str, Any], url: str | None) -> None:
    """POST a hunt's final status to ``url`` (its ``callback_url``), once. Never raises.

    The hunt's own state is already persisted; a callback that cannot be
    delivered is logged, never turned into a hunt failure. The URL is
    re-checked here as well as at queue time, so narrowing the configured
    prefixes also stops hunts that were queued before the change.
    """
    if not url:
        return
    hunt_id = job["hunt_id"]
    if not callback_url_allowed(url):
        logger.warning("Bug hunt %s: callback_url no longer allowed; not sent", hunt_id)
        return
    body = {
        "hunt_id": hunt_id,
        "container_name": job["container_name"],
        "status": str(getattr(job["status"], "value", job["status"])),
        "errors_found": job.get("errors_found"),
        "fixes_generated": job.get("fixes_generated"),
        "error_message": job.get("error_message"),
        "completed_at": job["completed_at"].isoformat() if job.get("completed_at") else None,
        "links": _hunt_links(job["container_name"], hunt_id),
    }
    async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_S, follow_redirects=False) as client:
        for attempt in range(1, CALLBACK_ATTEMPTS + 1):
            try:
                resp = await client.post(url, json=body)
                if 200 <= resp.status_code < 300:
                    logger.info(
                        "Bug hunt %s: callback delivered (HTTP %s, attempt %s)",
                        hunt_id,
                        resp.status_code,
                        attempt,
                    )
                    return
                if resp.status_code < 500:
                    # Final, not retried: a 4xx will not change on retry (e.g. a Wait
                    # node left on GET answers POST with 404 and the resume is lost).
                    logger.warning(
                        "Bug hunt %s: callback refused (HTTP %s); not retried",
                        hunt_id,
                        resp.status_code,
                    )
                    return
                logger.warning(
                    "Bug hunt %s: callback HTTP %s (attempt %s)", hunt_id, resp.status_code, attempt
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Bug hunt %s: callback failed (attempt %s): %s", hunt_id, attempt, exc
                )
            if attempt < CALLBACK_ATTEMPTS:
                await asyncio.sleep(2**attempt)
    logger.error(
        "Bug hunt %s: callback not delivered after %s attempts", hunt_id, CALLBACK_ATTEMPTS
    )


async def _notify_terminal(
    store: BugHuntReportStore,
    *,
    hunt_id: str,
    container_name: str,
    tenant_id: str,
    callback_url: str | None,
) -> None:
    """Re-read the now-terminal hunt and send its callback. Never raises."""
    if not callback_url:
        return
    try:
        job = await store.get_hunt(
            hunt_id=hunt_id, container_name=container_name, tenant_id=tenant_id
        )
        if job is not None:
            await _send_completion_callback(job, callback_url)
    except Exception:
        logger.exception("Bug hunt %s: completion callback errored", hunt_id)


def _hunt_links(container_name: str, hunt_id: str) -> dict[str, str]:
    base = f"/api/bug-hunter/{container_name}/hunts/{hunt_id}"
    return {
        "status": f"{base}/status",
        "summary": f"{base}/report",
        "findings": f"{base}/findings?path=",
    }


def _status_response(job: dict[str, Any]) -> BugHuntStatusResponse:
    return BugHuntStatusResponse(
        hunt_id=job["hunt_id"],
        container_name=job["container_name"],
        status=job["status"],
        errors_found=job["errors_found"],
        fixes_generated=job["fixes_generated"],
        error_message=job["error_message"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        updated_at=job["updated_at"],
    )


async def _get_visible_hunt(
    store: BugHuntReportStore,
    *,
    hunt_id: str,
    container_name: str,
    tenant_id: str,
) -> dict[str, Any]:
    job = await store.get_hunt(
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
    )
    if job is None:
        raise not_found("Bug hunt", hunt_id)
    return job


async def _renew_hunt_lease(
    *,
    hunt_id: str,
    worker_id: str,
    store: BugHuntReportStore,
) -> None:
    """Keep a long-running job leased to its worker process."""
    while True:
        await asyncio.sleep(30)
        renewed = await store.renew_lease(hunt_id=hunt_id, worker_id=worker_id)
        if not renewed:
            logger.warning("Worker %s lost lease for bug hunt %s", worker_id, hunt_id)
            return


async def _run_claimed_hunt(
    *,
    job: dict[str, Any],
    worker_id: str,
    store: BugHuntReportStore,
) -> None:
    """Execute and persist a job atomically claimed from PostgreSQL."""
    hunt_id = job["hunt_id"]
    container_name = job["container_name"]
    tenant_id = job["tenant_id"]
    callback_url = (job.get("request_payload") or {}).get("callback_url")
    terminal = False  # True once THIS worker moved the hunt to completed/failed
    heartbeat = asyncio.create_task(
        _renew_hunt_lease(hunt_id=hunt_id, worker_id=worker_id, store=store)
    )
    try:
        request = BugHuntingRequest.model_validate(job.get("request_payload") or {})
        container = await get_container_manager().get_container(container_name, tenant_id=tenant_id)
        if container is None:
            raise FileNotFoundError(f"Container '{container_name}' not found")
        if _decode_error_logs(request).strip():
            codebase_path = getattr(container, "work_dir", None) or getattr(
                container,
                "codebase_path",
                f"/tmp/{container_name}",  # nosec B108
            )
        else:
            # The scan ran in the workspace; findings are relative to its root.
            codebase_path = WORKSPACE_ROOT
        report = await hunt_bugs(
            container_name=container_name,
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id=tenant_id,
            hunt_id=hunt_id,
        )
        await store.complete_hunt(
            hunt_id=hunt_id,
            container_name=container_name,
            tenant_id=tenant_id,
            report=report,
            codebase_path=codebase_path,
            worker_id=worker_id,
        )
        terminal = True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Bug hunt %s failed for %s", hunt_id, container_name)
        try:
            terminal = await store.fail_hunt(
                hunt_id=hunt_id,
                container_name=container_name,
                tenant_id=tenant_id,
                error_message=str(getattr(exc, "detail", exc)),
                worker_id=worker_id,
            )
        except Exception:
            logger.exception("Failed to persist failure state for bug hunt %s", hunt_id)
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
    # Only the worker that made the terminal transition calls back, so a hunt
    # whose lease moved to another worker is reported exactly once.
    if terminal:
        await _notify_terminal(
            store,
            hunt_id=hunt_id,
            container_name=container_name,
            tenant_id=tenant_id,
            callback_url=callback_url,
        )


async def run_bug_hunt_worker(worker_id: str, poll_interval: float = 2.0) -> None:
    """Continuously claim durable Bug Hunter jobs using PostgreSQL leases."""
    store = BugHuntReportStore()
    while True:
        try:
            job = await store.claim_next_hunt(worker_id=worker_id)
            if job is None:
                await asyncio.sleep(poll_interval)
                continue
            await _run_claimed_hunt(job=job, worker_id=worker_id, store=store)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Bug hunt worker %s failed while polling", worker_id)
            await asyncio.sleep(max(poll_interval, 5.0))


def start_bug_hunt_workers(count: int = 2) -> list[asyncio.Task[None]]:
    """Start a bounded set of database-backed Bug Hunter workers."""
    return [asyncio.create_task(run_bug_hunt_worker(f"bug-hunter-{uuid7()}")) for _ in range(count)]


async def stop_bug_hunt_workers(workers: list[asyncio.Task[None]]) -> None:
    """Stop worker loops; active job leases expire and can be reclaimed."""
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


@router.post(
    "/{container_name}/hunt",
    response_model=BugHuntAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="queue_bug_hunt",
    summary="Queue an asynchronous bug hunt",
    description=(
        "Queues repository analysis and immediately returns a hunt ID. Poll the status link, "
        "then use the summary and path-scoped findings links after completion. An empty JSON "
        "body scans the codebase; error_logs can be supplied to analyze runtime failures."
    ),
)
async def queue_bug_hunt(
    response: Response,
    container_name: str = Path(..., min_length=1, description="Name of the container"),
    request: BugHuntRequest | None = None,
    tenant_id: str = Depends(require_tenant),
    store: BugHuntReportStore = Depends(get_bug_hunt_report_store),
) -> BugHuntAcceptedResponse:
    """Persist and enqueue a bug hunt without holding the HTTP request open."""
    request = request or BugHuntRequest()
    if request.callback_url is not None and not callback_url_allowed(request.callback_url):
        # Refused before anything is persisted: the worker would POST to this URL
        # from inside the cluster, so only configured prefixes are accepted.
        raise validation_error(
            "callback_url is not under an allowed callback prefix", field="callback_url"
        )

    container_manager = get_container_manager()
    try:
        container = await container_manager.get_container(container_name, tenant_id=tenant_id)
    except ValueError as exc:
        raise not_found("Container", container_name) from exc
    if container is None:
        raise not_found("Container", container_name)

    hunt_id = str(uuid7())
    request_payload = request.model_dump(mode="json")
    try:
        job = await store.create_hunt(
            hunt_id=hunt_id,
            container_name=container_name,
            tenant_id=tenant_id,
            request_payload=request_payload,
        )
    except Exception as exc:
        logger.exception("Failed to create bug hunt job for %s", container_name)
        raise internal_error("Failed to queue bug hunt") from exc

    links = _hunt_links(container_name, hunt_id)
    response.headers["Location"] = links["status"]
    response.headers["Retry-After"] = "2"
    return BugHuntAcceptedResponse(
        hunt_id=hunt_id,
        container_name=container_name,
        status=BugHuntJobStatus.QUEUED,
        created_at=job["created_at"],
        links=links,
    )


@router.get(
    "/{container_name}/hunts/{hunt_id}/status",
    response_model=BugHuntStatusResponse,
    operation_id="get_bug_hunt_job_status",
    summary="Get one bug hunt's status",
)
async def get_bug_hunt_job_status(
    container_name: str,
    hunt_id: str,
    tenant_id: str = Depends(require_tenant),
    store: BugHuntReportStore = Depends(get_bug_hunt_report_store),
) -> BugHuntStatusResponse:
    job = await _get_visible_hunt(
        store,
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
    )
    return _status_response(job)


@router.get(
    "/{container_name}/hunts/{hunt_id}/report",
    response_model=BugHuntSummaryResponse,
    operation_id="get_bug_hunt_summary_report",
    summary="Get a bug hunt summary and flagged files",
)
async def get_bug_hunt_summary_report(
    container_name: str,
    hunt_id: str,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    tenant_id: str = Depends(require_tenant),
    store: BugHuntReportStore = Depends(get_bug_hunt_report_store),
) -> BugHuntSummaryResponse:
    job = await _get_visible_hunt(
        store,
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
    )
    if job["status"] != BugHuntJobStatus.COMPLETED:
        raise conflict("Bug hunt", f"Bug hunt '{hunt_id}' is {job['status']}; report is not ready")

    total_files, flagged_files = await store.get_file_summary(
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
    )
    status_fields = _status_response(job).model_dump()
    return BugHuntSummaryResponse(
        **status_fields,
        root_cause_analysis=job["root_cause_analysis"] or "",
        fix_summary=job["fix_summary"] or "",
        estimated_effort=job["estimated_effort"] or "",
        risk_assessment=job["risk_assessment"] or "",
        analysis_time=job["analysis_time"] or 0.0,
        # Straight from storage, None preserved: an unmeasured value must reach
        # the consumer as null so it refuses, never as 0.
        scanned_files=job.get("scanned_files"),
        tracked_source_files=job.get("tracked_source_files"),
        commit_sha=job.get("commit_sha"),
        worktree_dirty=job.get("worktree_dirty"),
        excluded_dirs=job.get("excluded_dirs"),
        excluded_entry_kinds=job.get("excluded_entry_kinds"),
        agent_sha=job.get("agent_sha"),
        total_flagged_files=total_files,
        flagged_files=[BugHuntFlaggedFile.model_validate(item) for item in flagged_files],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{container_name}/hunts/{hunt_id}/findings",
    response_model=BugHuntPathFindingsResponse,
    operation_id="get_bug_hunt_findings_for_path",
    summary="Get detailed bugs for a file or directory subtree",
    description=(
        "The path is relative to the repository root. A file path returns that file; a "
        "directory path returns bugs in that directory and all subdirectories. Empty path "
        "selects the whole repository."
    ),
)
async def get_bug_hunt_findings_for_path(
    container_name: str,
    hunt_id: str,
    path: str = Query("", description="Repository-relative file or directory path"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    tenant_id: str = Depends(require_tenant),
    store: BugHuntReportStore = Depends(get_bug_hunt_report_store),
) -> BugHuntPathFindingsResponse:
    job = await _get_visible_hunt(
        store,
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
    )
    if job["status"] != BugHuntJobStatus.COMPLETED:
        raise conflict(
            "Bug hunt", f"Bug hunt '{hunt_id}' is {job['status']}; findings are not ready"
        )

    try:
        normalized_path = normalize_query_path(path)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc

    total_findings, matched_files, findings = await store.get_findings_for_path(
        hunt_id=hunt_id,
        container_name=container_name,
        tenant_id=tenant_id,
        path=normalized_path,
        limit=limit,
        offset=offset,
    )
    return BugHuntPathFindingsResponse(
        hunt_id=hunt_id,
        container_name=container_name,
        path=normalized_path,
        commit_sha=job.get("commit_sha"),
        scanned_files=job.get("scanned_files"),
        tracked_source_files=job.get("tracked_source_files"),
        worktree_dirty=job.get("worktree_dirty"),
        excluded_dirs=job.get("excluded_dirs"),
        excluded_entry_kinds=job.get("excluded_entry_kinds"),
        agent_sha=job.get("agent_sha"),
        matched_files=matched_files,
        total_findings=total_findings,
        findings=[BugHuntFinding.model_validate(item) for item in findings],
        limit=limit,
        offset=offset,
    )


def _recheck_refusal(status_code: int, error_code: str, message: str) -> HTTPException:
    return error_response(status_code=status_code, message=message, error_code=error_code)


@router.post(
    "/{container_name}/hunts/{hunt_id}/findings/{finding_id}/recheck",
    response_model=BugHuntFindingRecheckResponse,
    operation_id="recheck_bug_hunt_finding",
    summary="Re-run the detector on one finding's file at the hunted commit",
    description=(
        "Reads the finding's file AT the hunt's commit_sha inside the tenant workspace (no "
        "checkout, never the working tree) and re-runs the same detector. reproduced=true iff "
        "a finding with the same fingerprint is present there. Any failure to obtain a "
        "verified scan is an error status (502 recheck_unverifiable), never reproduced=false."
    ),
    responses={
        404: {"description": "Hunt or finding not visible to this tenant"},
        409: {"description": "HUNT_NOT_COMPLETED, or AGENT_DRIFT (a different agent is deployed)"},
        422: {"description": "HUNT_INCOMPLETE or FINDING_NOT_FINGERPRINTED"},
        502: {"description": "RECHECK_UNVERIFIABLE: recheck_unverifiable: <reason>"},
    },
)
async def recheck_bug_hunt_finding(
    container_name: str,
    hunt_id: str,
    finding_id: int = Path(..., ge=1),
    tenant_id: str = Depends(require_tenant),
    store: BugHuntReportStore = Depends(get_bug_hunt_report_store),
) -> BugHuntFindingRecheckResponse:
    # Tenant-scoped like every read here: another tenant's hunt is a 404.
    job = await _get_visible_hunt(
        store, hunt_id=hunt_id, container_name=container_name, tenant_id=tenant_id
    )
    if job["status"] != BugHuntJobStatus.COMPLETED:
        raise _recheck_refusal(
            status.HTTP_409_CONFLICT,
            "HUNT_NOT_COMPLETED",
            f"Bug hunt '{hunt_id}' is {job['status']}; only a completed hunt can be rechecked",
        )
    commit_sha = job.get("commit_sha")
    scanned, tracked = job.get("scanned_files"), job.get("tracked_source_files")
    hunt_agent_sha = job.get("agent_sha")
    if not commit_sha:
        raise _recheck_refusal(
            422, "HUNT_INCOMPLETE", "hunt_incomplete: commit_sha is null (no commit was scanned)"
        )
    if scanned is None or tracked is None or scanned != tracked:
        raise _recheck_refusal(
            422,
            "HUNT_INCOMPLETE",
            f"hunt_incomplete: scanned_files {scanned!r} != tracked_source_files {tracked!r}",
        )
    if not hunt_agent_sha:
        raise _recheck_refusal(
            422, "HUNT_INCOMPLETE", "hunt_incomplete: agent_sha is null (detector unknown)"
        )

    finding = await store.get_finding(
        hunt_id=hunt_id, container_name=container_name, tenant_id=tenant_id, finding_id=finding_id
    )
    if finding is None:
        raise not_found("Bug hunt finding", str(finding_id))
    fingerprint = finding.get("fingerprint")
    if not fingerprint:
        raise _recheck_refusal(
            422, "FINDING_NOT_FINGERPRINTED", f"finding {finding_id} has no fingerprint"
        )

    deployed_sha = expected_agent_sha()
    if deployed_sha != hunt_agent_sha:
        # A different detector reproducing nothing is not a refutation.
        raise _recheck_refusal(
            status.HTTP_409_CONFLICT,
            "AGENT_DRIFT",
            f"agent_drift: hunt ran agent {hunt_agent_sha}, deployed agent is {deployed_sha}",
        )

    try:
        result = await run_workspace_recheck(
            container_name=container_name,
            tenant_id=tenant_id,
            commit_sha=str(commit_sha),
            path=str(finding["file_path"]),
            recheck_id=str(uuid7()),
        )
    except RecheckUnverifiableError as exc:
        # Infrastructure broke or the file could not be examined at that commit.
        # That is NOT "does not reproduce", so it is never a 200.
        raise error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            message=str(exc),
            error_code="RECHECK_UNVERIFIABLE",
            details={"reason": exc.reason},
        ) from exc

    matched = next((f for f in result.findings if f.get("fingerprint") == fingerprint), None)
    return BugHuntFindingRecheckResponse(
        hunt_id=hunt_id,
        container_name=container_name,
        finding_id=finding_id,
        path=result.path,
        reproduced=matched is not None,
        finding_fingerprint=str(fingerprint),
        matched_finding=matched,
        commit_sha=result.commit_sha,
        agent_sha=result.agent_sha,
        findings_at_commit=len(result.findings),
        reason=(
            "fingerprint_present_at_commit"
            if matched is not None
            else "fingerprint_absent_at_commit"
        ),
    )


# Module initialization helper
async def initialize_bug_hunt_state(container_name: str) -> None:
    """Initialize bug hunt state for a container.

    This function is called when a new container is created or discovered.
    It ensures the bug hunt tracking state is properly initialized.

    Args:
        container_name: Name of the container to initialize
    """
    await _get_or_initialize_hunt_state(container_name)
    logger.debug(f"Initialized bug hunt state for container: {container_name}")


# Helper function to update hunt state (used by services)
async def update_hunt_state(
    container_name: str,
    status: BugHunterAnalysisStatus | None = None,
    bugs_found: int | None = None,
    bugs_fixed: int | None = None,
    error: str | None = None,
) -> None:
    """Update the bug hunt state for a container.

    This function is called by bug hunting services to update progress.

    Args:
        container_name: Name of the container
        status: New analysis status
        bugs_found: Number of bugs found in current hunt
        bugs_fixed: Number of bugs fixed
        error: Error message if operation failed
    """
    state = await _get_or_initialize_hunt_state(container_name)

    if status is not None:
        state["analysis_status"] = status.value if hasattr(status, "value") else status

    if bugs_found is not None:
        state["bugs_found_count"] = bugs_found
        # Update total
        state["total_bugs_found"] = state.get("total_bugs_found", 0) + bugs_found

    if bugs_fixed is not None:
        state["bugs_fixed_count"] = bugs_fixed
        # Update total
        state["total_bugs_fixed"] = state.get("total_bugs_fixed", 0) + bugs_fixed

    if error is not None:
        state["last_error"] = error
        state["analysis_status"] = BugHunterAnalysisStatus.ERROR.value

    await _persist_hunt_state(container_name, state)
    logger.debug(f"Updated bug hunt state for {container_name}: {state}")
