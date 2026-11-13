"""API quality module."""

from __future__ import annotations

"""Quality Analysis API Module.

This module provides comprehensive FastAPI endpoints for code quality analysis and reporting
across multiple programming languages. It serves as the primary interface for triggering
quality scans, retrieving analysis results, managing quality data, and providing insights
into code quality metrics.

The module supports:
    - Multi-language quality analysis (Python, JavaScript, TypeScript, Java, Go, Rust)
    - Asynchronous quality scan execution with task tracking
    - Comprehensive filtering and search capabilities for quality issues
    - Quality trend analysis and historical data tracking
    - Intelligent caching system for performance optimization
    - Quality grading and scoring with actionable recommendations

Key Features:
    - Language-specific tool configuration and execution
    - Severity-based issue classification (critical, high, moderate, low)
    - File-level and project-level quality metrics
    - Background task processing for large codebases
    - Cache management with configurable expiration policies
    - RESTful API design with comprehensive error handling

Quality Analysis Process:
    1. Code repository scanning and file discovery
    2. Language-specific tool execution (linters, type checkers, security scanners)
    3. Issue aggregation and severity classification
    4. Quality score calculation and grade assignment
    5. Result caching and performance optimization
    6. Recommendation generation based on issue patterns

Supported Quality Tools:
    - Python: ruff (check + format), ty, bandit, radon
    - JavaScript/TypeScript: eslint, tsc, prettier
    - Java: checkstyle, pmd, spotbugs
    - Go: golint, go vet, staticcheck
    - Rust: clippy, rustfmt

Example Usage:
    ```python
    # Trigger quality analysis
    response = await client.post("/api/quality/analyze",
                                params={"codebase_id": "my-repo"})

    # Get analysis results
    results = await client.get(f"/api/quality/results/{task_id}")

    # Filter issues by severity
    issues = await client.get("/api/quality/my-repo/issues",
                             params={"severity": "high", "limit": 50})
    ```

Note:
    All endpoints support comprehensive error responses (400, 404, 422, 500)
    and follow OpenAPI 3.0 specification for documentation generation.
"""
import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid7

from coding_client.auth.websocket import Outcome, resolve_websocket_identity
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.responses import (
    bad_request,
    internal_error,
    not_found,
    validation_error,
)
from app.core.redis_service import get_redis_service
from auth.tenant import require_tenant
from models.architecture_models import (
    ArchitectureQualityIssue as ArchQualityIssue,
)
from models.architecture_models import (
    QualityGateCheck,
    QualityGateResult,
)
from models.testing_models import QualityIssue
from processing.architecture.architecture_pattern_analysis import (
    LANGUAGE_ALIASES,
    LANGUAGE_EXTENSIONS,
    enabled_tool_details_by_language,
    enabled_tools_by_language,
)
from processing.architecture.quality_analyzer import (
    ArchitectureQualityAnalyzer as QualityAnalyzer,
)
from processing.architecture.quality_analyzer import (
    ScopeMatchesNoFilesError,
    ToolsNotApplicableError,
)
from processing.workspace_repos import PRIMARY_REPO, RepoSelector
from services.quality_result_store import (
    InvalidTenantIdError,
    QualityResultStore,
    build_row,
    derive_status,
    get_quality_result_store,
    resolve_commit_sha,
    validate_tenant_id,
)

from ..models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from ..utils.container_validation import validate_and_get_container_for_tenant

logger = logging.getLogger(__name__)
router = APIRouter()

# Every severity value the quality tools actually emit, lower-cased. The /issues
# filter used to accept only low|moderate|high|critical, so the far more common
# tool outputs (error / warning / info from ruff/eslint/ty, medium from
# bandit & the architecture analyzer) could never be selected -- a filter on
# "warning" was a 422 and every real warning was invisible to it.
#   - critical/high/medium/low: architecture analyzer + bandit (issue_severity)
#   - moderate: legacy API vocabulary
#   - error/warning/info: eslint, ruff, ty, radon (tool_integration.py)
#   - unknown: security tooling when a finding carries no severity
QUALITY_ISSUE_SEVERITIES: frozenset[str] = frozenset(
    {
        "critical",
        "high",
        "medium",
        "moderate",
        "low",
        "error",
        "warning",
        "info",
        "unknown",
    }
)


#: Upper bound on stored runs read for one /trend window.
TREND_MAX_ROWS = 500


def _serialize_quality_issue(issue: Any) -> dict[str, Any]:
    """One issue shape for every quality endpoint (mirrors /analyze-sync).

    ``tool`` and ``rule_id`` are what make an issue actionable; the /issues and
    /latest endpoints used to drop both.
    """
    return {
        "tool": issue.tool,
        "severity": issue.severity,
        "message": issue.message,
        "file_path": issue.file_path,
        "line_number": issue.line_number,
        "column_number": issue.column_number,
        "rule_id": issue.rule_id,
        "repo": getattr(issue, "repo", None),
    }


def _report_provenance(report: Any) -> dict[str, Any]:
    """Inconclusive/partial/per-tool provenance fields, as /analyze-sync emits them."""
    return {
        "inconclusive": bool(report.inconclusive),
        "partial": report.partial,
        "tools_attempted": report.tools_attempted,
        "tools_succeeded": report.tools_succeeded,
        "tools_failed": report.tools_failed,
    }


#: Upper bound on ``QualityAnalysisSyncRequest.paths``. Each entry becomes a
#: ``find`` start point and a tool argument, so the list must stay small.
MAX_SCOPE_PATHS = 200


def _normalize_scope_path(raw: str) -> str:
    """Validate and normalise one workspace-relative scope path.

    Rejects (ValueError → 422) absolute paths, any ``..`` component, NUL bytes
    and backslashes, and an entry that normalises to the workspace root (``.``
    / empty): omit ``paths`` to analyse the whole workspace.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("scope path must be a non-empty string")
    path = raw.strip()
    if "\x00" in path or "\\" in path:
        raise ValueError(f"scope path contains an illegal character: {raw!r}")
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise ValueError(f"scope path must be workspace-relative, not absolute: {raw!r}")
    parts = [part for part in path.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"scope path must not contain '..': {raw!r}")
    if not parts:
        raise ValueError(
            f"scope path {raw!r} is the workspace root; omit 'paths' to analyse everything"
        )
    return "/".join(parts)


INLINE_CODE_NEEDS_WORKSPACE = (
    "inline code_content needs a workspace: send container_name (+ paths) "
    "so tools run in the workspace"
)


def _check_requested_tools(tools: list[str] | None) -> list[str] | None:
    """422 on an empty ``tools`` list or a name NO language's registry knows.

    Checked before any container work, against every language in the analyzer's
    own registry (:func:`enabled_tools_by_language`) -- not the request's
    ``language``, because the run uses the DETECTED language. Whether a known
    name applies to the detected language is decided after detection
    (``ToolsNotApplicableError``). Removed tools (mypy, pylint, ...) are simply
    unknown. Returns the de-duplicated list, or ``None`` (= default set).
    """
    if tools is None:
        return None
    registry = enabled_tools_by_language()
    accepted = sorted({t for names in registry.values() for t in names})
    details: dict[str, Any] = {"tools": tools, "accepted_by_language": registry}
    if not tools:
        raise validation_error(
            f"tools must not be empty; omit it for the default set. accepted: {accepted}",
            "tools",
            {**details, "unknown": [], "accepted": accepted},
        )
    requested = list(dict.fromkeys(tools))
    unknown = [t for t in requested if t not in accepted]
    if unknown:
        raise validation_error(
            f"unknown tools: {unknown}; accepted: {accepted}",
            "tools",
            {**details, "unknown": unknown, "accepted": accepted},
        )
    return requested


class QualityAnalysisSyncRequest(RepoSelector):
    """Request body for synchronous quality analysis (``repo``: :class:`RepoSelector`)."""

    code_content: str | None = None
    code_path: str | None = None
    language: str | None = None
    tools: list[str] | None = None
    container_name: str | None = None
    fix_issues: bool = False
    custom_rules: dict[str, Any] | None = None
    #: Optional files/directories, relative to the selected ``repo``'s root, to
    #: restrict a ``container_name`` run to. ``None`` = the whole repo.
    paths: list[str] | None = Field(default=None, min_length=1, max_length=MAX_SCOPE_PATHS)
    #: Directory, relative to the selected repo's root, that project tools run
    #: in -- the crate (or cargo workspace) dir for ``language: "rust"``.
    workdir: str | None = Field(
        default=None,
        description=(
            "Crate directory, relative to the selected repo's root, that cargo runs in "
            "(e.g. 'crates/core'). Requires language 'rust'. Default: the repo root."
        ),
    )

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(_normalize_scope_path(p) for p in value))

    @property
    def rust_override(self) -> bool:
        """``language: "rust"`` is the only value honoured (forces the Rust tools).

        Any other value is accepted and ignored, as before: the run uses the
        detected language.
        """
        return (self.language or "").strip().lower() == "rust"

    @field_validator("workdir")
    @classmethod
    def _validate_workdir(cls, value: str | None) -> str | None:
        if value is None or value.strip() in ("", ".", "./"):
            return None
        return _normalize_scope_path(value)

    @model_validator(mode="after")
    def _paths_need_a_container(self) -> QualityAnalysisSyncRequest:
        if self.paths is not None and not self.container_name:
            raise ValueError("'paths' scopes a container run and requires 'container_name'")
        if self.repo != PRIMARY_REPO and not self.container_name:
            raise ValueError("'repo' selects a workspace repo and requires 'container_name'")
        if self.workdir is not None and not self.rust_override:
            raise ValueError("'workdir' is the cargo crate dir and requires language 'rust'")
        return self


def _calculate_grade(overall_score: float) -> str:
    """Calculate letter grade from numerical quality score.

    Converts a numerical quality score (0-100) into a standard letter grade
    using common academic grading thresholds. This provides an intuitive
    representation of code quality that is easily understood by developers.

    Grade Scale:
        - A: 90-100 (Excellent quality, minimal issues)
        - B: 80-89 (Good quality, minor issues)
        - C: 70-79 (Acceptable quality, moderate issues)
        - D: 60-69 (Poor quality, significant issues)
        - F: 0-59 (Failing quality, critical issues)

    Args:
        overall_score: Numerical quality score between 0 and 100.
            Typically calculated as a weighted average of various quality metrics
            including linting violations, complexity measures, test coverage,
            and security vulnerabilities.

    Returns:
        str: Single letter grade representing quality level.

    Example:
        >>> _calculate_grade(95.5)
        'A'
        >>> _calculate_grade(72.0)
        'C'
        >>> _calculate_grade(45.2)
        'F'

    Note:
        The grade calculation is designed to align with industry standards
        for code quality assessment and can be used for dashboard displays,
        reporting, and quality gate enforcement.
    """
    if overall_score >= 90:
        return "A"
    elif overall_score >= 80:
        return "B"
    elif overall_score >= 70:
        return "C"
    elif overall_score >= 60:
        return "D"
    else:
        return "F"


def _render_grade(result: Any) -> str:
    """Grade for API payloads, honoring an inconclusive abstain.

    When a quality run is inconclusive (no static-analysis tool completed), the
    analyzer sets ``grade='N/A'`` and a ``-1.0`` sentinel score. Recomputing a
    letter grade from ``-1.0`` via :func:`_calculate_grade` would render a
    misleading ``F`` (and a self-contradicting "great job" message downstream),
    so surface the honest abstain grade verbatim instead.
    """
    if getattr(result, "inconclusive", False):
        return result.grade
    return _calculate_grade(result.overall_score)


def _build_summary(issues: list[ArchQualityIssue] | list[QualityIssue]) -> dict[str, int]:
    """Build comprehensive summary statistics from quality issues.

    Analyzes a collection of quality issues and generates aggregate statistics
    that provide insights into the overall distribution and severity of problems
    found during code quality analysis.

    The summary includes total issue counts and breakdown by severity levels,
    which helps prioritize remediation efforts and understand quality patterns.

    Args:
        issues: List of ArchQualityIssue objects representing detected quality
            problems. Each issue should have severity and other metadata attributes.

    Returns:
        Dict[str, int]: Summary statistics containing:
            - total_issues: Total number of issues found
            - critical: Number of critical severity issues
            - high: Number of high severity issues
            - moderate: Number of moderate severity issues
            - low: Number of low severity issues
            - unknown: Number of issues with unrecognized severity

    Example:
        >>> issues = [
        ...     QualityIssue(severity="high", message="Type error"),
        ...     QualityIssue(severity="low", message="Style issue"),
        ...     QualityIssue(severity="critical", message="Security flaw")
        ... ]
        >>> _build_summary(issues)
        {'total_issues': 3, 'high': 1, 'low': 1, 'critical': 1}

    Note:
        Issues without a severity attribute or with unrecognized severity values
        are categorized as 'unknown' to ensure data integrity in reporting.
    """
    summary = {"total_issues": len(issues)}
    severity_counts: dict[str, int] = {}
    for issue in issues:
        severity = issue.severity if hasattr(issue, "severity") else "unknown"
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    summary.update(severity_counts)
    return summary


def _generate_recommendations(
    issues: list[ArchQualityIssue] | list[QualityIssue], inconclusive: bool = False
) -> list[str]:
    """Generate actionable recommendations based on quality issues analysis.

    Analyzes the collection of quality issues and generates prioritized,
    actionable recommendations to help developers address code quality problems
    effectively. The recommendations are based on issue severity distribution
    and follow best practices for quality improvement workflows.

    Recommendation Strategy:
        - No issues: Positive reinforcement message
        - Issues present: Prioritized action plan based on severity
        - Critical issues: Immediate attention required
        - High issues: Next priority after critical
        - Clear guidance on issue counts and priorities

    Args:
        issues: List of ArchQualityIssue objects representing detected problems.
            Each issue should contain severity metadata for proper classification.

    Returns:
        List[str]: Ordered List[Any] of actionable recommendations, with most
            critical actions first. Each recommendation is a human-readable
            string that provides specific guidance on addressing quality issues.

    Example:
        >>> issues = [
        ...     QualityIssue(severity="critical", message="SQL injection risk"),
        ...     QualityIssue(severity="high", message="Memory leak potential"),
        ...     QualityIssue(severity="low", message="Style violation")
        ... ]
        >>> recommendations = _generate_recommendations(issues)
        >>> print(recommendations[0])
        'Address 3 quality issues to update code quality.'
        >>> print(recommendations[1])
        'Prioritize 1 critical issues first.'

    Note:
        Recommendations are generated in priority order to guide development
        teams through systematic quality improvement. Critical and high-severity
        issues receive specific mention to emphasize their importance.
    """
    recommendations = []
    if inconclusive:
        # No static-analysis tool completed: an empty issue list is "unknown",
        # not "clean". Never emit a congratulatory pass here.
        recommendations.append(
            "Analysis inconclusive: no static-analysis tool completed "
            "successfully — this is NOT a pass; re-run or check tool timeouts."
        )
    elif not issues:
        recommendations.append("Great job! No issues detected.")
    else:
        recommendations.append(f"Address {len(issues)} quality issues to update code quality.")
        critical_count = sum(1 for issue in issues if getattr(issue, "severity", "") == "critical")
        high_count = sum(1 for issue in issues if getattr(issue, "severity", "") == "high")
        if critical_count > 0:
            recommendations.append(f"Prioritize {critical_count} critical issues first.")
        if high_count > 0:
            recommendations.append(f"Address {high_count} high-severity issues.")
    return recommendations


def get_quality_analyzer():
    """FastAPI dependency function to provide QualityAnalyzer instance.

    Creates and returns a QualityAnalyzer instance for dependency injection
    in FastAPI endpoints. This follows the dependency injection pattern
    recommended by FastAPI for managing service instances and enabling
    easy testing and mock substitution.

    The QualityAnalyzer is responsible for orchestrating quality analysis
    across multiple programming languages and integrating various quality
    tools (linters, type checkers, security scanners, formatters).

    Returns:
        QualityAnalyzer: Configured analyzer instance ready for use in
            quality analysis operations. The analyzer includes language
            detection, tool execution, result aggregation, and scoring logic.

    Example:
        ```python
        @router.post("/analyze")
        async def analyze_code(
            analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
            tenant_id: str = Depends(require_tenant),
        ):
            result = await analyzer.analyze(repo_path, codebase_id, tenant_id=tenant_id)
            return result
        ```

    Note:
        This dependency can be overridden in tests by providing a mock
        QualityAnalyzer instance, enabling isolated unit testing of
        API endpoints without executing actual quality analysis tools.
    """
    return QualityAnalyzer()


class QualityLanguage(BaseModel):
    """One language the quality analysis runs tools for."""

    name: str = Field(..., description="Canonical language name")
    extensions: list[str] = Field(..., description="Source extensions counted for detection")
    tools: list[str] = Field(..., description="Tools a run for this language executes")


class QualityLanguagesResponse(BaseModel):
    languages: list[QualityLanguage]


class QualityTool(BaseModel):
    """A tool the quality analysis runs (enabled in its registry and dispatched)."""

    name: str
    category: str
    description: str
    path_scoping: Literal["paths", "whole_project_findings_filtered", "skipped"] = Field(
        ...,
        description=(
            "On a path-scoped request: 'paths' = pointed at the scoped paths, "
            "'whole_project_findings_filtered' = runs project-wide and findings are "
            "filtered to the scope, 'skipped' = not run"
        ),
    )


class QualityLanguageToolsResponse(BaseModel):
    language: str = Field(..., description="Canonical language name (aliases resolved)")
    tools: list[QualityTool]


class QualityInfoResponse(BaseModel):
    service: str
    version: str
    available_tools: dict[str, list[str]] = Field(
        ..., description="Per language, the tool names analyze-sync accepts"
    )


@router.get(
    "/languages",
    operation_id="list_quality_languages",
    response_model=QualityLanguagesResponse,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_supported_languages() -> dict[str, Any]:
    """Get List[Any] of supported programming languages for quality analysis.

    Returns information about all programming languages supported by the quality
    analysis system, including their file extensions and available analysis tools.

    Returns:
        A dictionary containing information about supported languages

    Raises:
        HTTPException: If the language information cannot be retrieved
    """
    # The analyzer's own registry: the languages it detects and the tools
    # analyze-sync accepts for each.
    return {
        "languages": [
            {
                "name": language,
                "extensions": list(LANGUAGE_EXTENSIONS.get(language, ())),
                "tools": names,
            }
            for language, names in enabled_tools_by_language().items()
        ]
    }


@router.get(
    "/extensions",
    operation_id="list_quality_extensions",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_supported_extensions() -> dict[str, Any]:
    """Get all supported file extensions for quality analysis.

    Returns a flat list of file extensions supported across all languages.
    """
    languages_data = await get_supported_languages()
    extensions: list[str] = []
    for lang in languages_data.get("languages", []):
        extensions.extend(lang.get("extensions", []))
    return {"extensions": sorted(set(extensions))}


@router.get(
    "/{language}/tools",
    operation_id="get_quality_language_tools",
    response_model=QualityLanguageToolsResponse,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_language_tools(language: str) -> dict[str, Any]:
    """Get available quality tools for a specific language.

    Every tool a run for the language executes, with its category, description
    and ``path_scoping`` (how it behaves on a ``paths``-scoped request).

    Args:
        language: The language (e.g. 'python', 'javascript'; aliases such as
            'py', 'ts', 'rs', 'c#' are accepted and the canonical name returned)

    Returns:
        A dictionary containing tool details for the specified language

    Raises:
        HTTPException: If the language is not supported or other errors occur
    """
    # The analyzer's own registry: exactly the names analyze-sync accepts.
    key = language.strip().lower()
    canonical = LANGUAGE_ALIASES.get(key, key)
    tools = enabled_tool_details_by_language().get(canonical, [])
    if not tools:
        raise not_found("Language", language)
    return {"language": canonical, "tools": tools}


async def _persist_result(
    store: QualityResultStore,
    report: Any,
    *,
    tenant_id: str,
    container_name: str,
    repo_url: str | None,
    commit_sha: str | None,
    error_message: str | None = None,
) -> str | None:
    """Store one analysis run; return its id, or ``None`` if persisting failed.

    NEVER raises. ``POST /analyze-sync`` is an auto-run verifier in the live
    conveyor, so a database hiccup here must not turn a completed analysis into
    a failed verification — the caller reports ``persisted: false`` instead.
    """
    try:
        return await store.save(
            report,
            tenant_id=tenant_id,
            container_name=container_name,
            repo_url=repo_url,
            commit_sha=commit_sha,
            error_message=error_message,
        )
    except Exception as e:  # noqa: BLE001 - persistence is best-effort by contract
        logger.error(
            "Failed to persist quality result for %s: %s", container_name, e, exc_info=True
        )
        return None


@router.post(
    "/analyze",
    operation_id="trigger_quality_analysis",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Container not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def trigger_quality_analysis(
    container_name: str,
    background_tasks: BackgroundTasks,
    analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Trigger quality analysis for a container.

    Initiates an asynchronous quality analysis task for the specified container's codebase.
    The analysis runs in the background and generates a task ID that can be used
    to retrieve the results later.

    Args:
        container_name: The name of the container to analyze
        background_tasks: FastAPI background tasks handler
        analyzer: The QualityAnalyzer instance to use for analysis

    Returns:
        A dictionary containing the task ID and status information

    Raises:
        HTTPException: If the analysis cannot be started or other errors occur
    """
    try:
        # Validate container exists and get container info
        container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
        repo_path = Path(container_info.codebase_path)

        # Note: codebase_path is inside the container, not directly accessible from this service.
        # The analyzer handles file access through the container runtime.
        task_id = str(uuid7())

        repo_url = getattr(container_info, "repo_url", None)

        async def analysis_task() -> None:
            # Resolved BEFORE analysing so the stored row names the tree that
            # was measured. Unknown → None, never a branch name.
            commit_sha = await resolve_commit_sha(container_name, tenant_id)
            try:
                result = await analyzer.analyze(repo_path, container_name, tenant_id=tenant_id)
                result_id = await _persist_result(
                    store,
                    result or None,
                    tenant_id=tenant_id,
                    container_name=container_name,
                    repo_url=repo_url,
                    commit_sha=commit_sha,
                )
                logger.info(
                    "Quality analysis completed for container %s with result: %s",
                    container_name,
                    result,
                )
                # Store result in Redis keyed by task_id
                if result:
                    try:
                        redis_service = get_redis_service()
                        task_result = {
                            "status": "completed",
                            "task_id": task_id,
                            "container_name": container_name,
                            "result_id": result_id,
                            "commit_sha": commit_sha,
                            "persisted": result_id is not None,
                            "result": {
                                "overall_score": result.overall_score,
                                "grade": _render_grade(result),
                                "inconclusive": result.inconclusive,
                                "partial": result.partial,
                                "tools_attempted": result.tools_attempted,
                                "tools_succeeded": result.tools_succeeded,
                                "tools_failed": result.tools_failed,
                                "issues_count": len(result.issues),
                                "issues": [
                                    {
                                        "tool": issue.tool,
                                        "severity": issue.severity,
                                        "message": issue.message,
                                        "file_path": issue.file_path,
                                        "line_number": issue.line_number,
                                        "column_number": issue.column_number,
                                        "rule_id": issue.rule_id,
                                        "repo": issue.repo,
                                    }
                                    for issue in result.issues
                                ],
                                "summary": _build_summary(result.issues),
                                "recommendations": _generate_recommendations(
                                    result.issues, inconclusive=result.inconclusive
                                ),
                                "execution_time": result.execution_time,
                                "scan_timestamp": result.generated_at,
                                "cached": False,
                            },
                        }
                        await redis_service.set(
                            f"quality_task:{task_id}",
                            json.dumps(task_result, default=str),
                            ex=86400,  # 24h TTL
                        )
                    except (ConnectionError, OSError, RuntimeError) as redis_err:
                        logger.warning("Failed to store task result in Redis: %s", redis_err)
                else:
                    try:
                        redis_service = get_redis_service()
                        task_result = {
                            "status": "failed",
                            "task_id": task_id,
                            "error": "Analysis failed to produce results",
                            "result_id": result_id,
                            "commit_sha": commit_sha,
                            "persisted": result_id is not None,
                        }
                        await redis_service.set(
                            f"quality_task:{task_id}",
                            json.dumps(task_result),
                            ex=86400,
                        )
                    except (ConnectionError, OSError, RuntimeError) as redis_err:
                        logger.warning("Failed to store task failure in Redis: %s", redis_err)
            except (ValueError, AttributeError, KeyError, RuntimeError) as e:
                logger.error("Error during quality analysis: %s", e, exc_info=True)
                failed_id = await _persist_result(
                    store,
                    None,
                    tenant_id=tenant_id,
                    container_name=container_name,
                    repo_url=repo_url,
                    commit_sha=commit_sha,
                    error_message=str(e),
                )
                try:
                    redis_service = get_redis_service()
                    task_result = {
                        "status": "failed",
                        "task_id": task_id,
                        "error": str(e),
                        "result_id": failed_id,
                        "commit_sha": commit_sha,
                        "persisted": failed_id is not None,
                    }
                    await redis_service.set(
                        f"quality_task:{task_id}",
                        json.dumps(task_result),
                        ex=86400,
                    )
                except (ConnectionError, OSError, RuntimeError) as redis_err:
                    logger.warning("Failed to store task error in Redis: %s", redis_err)

        background_tasks.add_task(analysis_task)
        return {
            "message": "Quality analysis started",
            "task_id": task_id,
            "container_name": container_name,
            "status": "queued",
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(
            "Failed to trigger quality analysis for %s: %s",
            container_name,
            e,
            exc_info=True,
        )
        raise internal_error(str(e)) from e


@router.post(
    "/analyze-sync",
    operation_id="run_quality_analysis_sync",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def run_quality_analysis_sync(
    request: QualityAnalysisSyncRequest,
    analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Run synchronous quality analysis and return results directly.

    When ``container_name`` is given the run is also STORED (``quality_results``)
    and the response carries ``result_id`` / ``commit_sha`` / ``persisted``.
    Persistence never fails the response: this route is an auto-run verifier,
    and every pre-existing key is unchanged.

    Inline ``code_content`` without ``container_name`` is a 422
    (``INLINE_CODE_NEEDS_WORKSPACE``) and executes nothing: every tool runs via
    coding-containers exec inside the workspace container, and this pod has
    no tools of its own, so an ad-hoc run could only ever come back
    "inconclusive: 0/N tools succeeded". Send ``container_name`` (+ ``paths``).
    With ``container_name`` set, ``code_content`` is ignored.

    ``paths`` (with ``container_name``) scopes the run to those
    workspace-relative files/directories: only they are fetched from the
    container and fed to the tools, and only their issues are reported. The
    response echoes the scope as ``scope`` (``null`` for a whole-workspace run).
    A scope that matches no source file is a 422 ("paths match no files"),
    never an empty-but-successful report. Absolute paths, ``..`` and more than
    ``MAX_SCOPE_PATHS`` entries are 422 at validation.

    ``repo`` picks the workspace repository (default ``primary``: the workspace
    root with ``.repos/`` excluded; ``.repos/<name>``: that source checkout).
    Every issue carries its ``repo``; the response echoes ``repo``.

    ``language: "rust"`` overrides detection and always runs
    ``cargo fmt --all -- --check`` and ``cargo clippy --workspace --all-targets
    --message-format=json`` in ``workdir`` (a crate dir relative to the repo;
    default its root) as developer feedback. With ``paths`` those two still run
    on the whole crate and only findings on the scoped paths are reported; a
    clippy run that exited non-zero with no finding left in scope is a failed
    tool, not a clean one.

    Store decision: a SCOPED run is NOT stored (``persisted: false``), nor is a
    run of a non-primary ``repo`` or of a ``workdir``. Any other ``language``
    value is echoed and ignored (the detected language is used). The
    ``quality_results`` row has no scope column, and ``/{c}/latest``,
    ``/{c}/gate`` and ``/{c}/history`` read the newest row for the container
    as its full-workspace result — storing a one-file run would silently
    replace the container's gate verdict with a partial one.

    ``tools`` restricts the run to exactly those tools (echoed as
    ``tools_requested``; ``null`` = default set). An empty list, a name the
    tool registry does not know (e.g. the removed ``mypy``) or one that does
    not apply to the detected language is a 422 on field ``tools`` naming the
    unknown entries and the accepted ones -- never a 200 that ran a different
    set. Like a scoped run, a run with ``tools`` set is NOT stored, even when it
    names the whole default set: it skips the performance scan, so the gate and
    history would grade it against default rows it does not match.
    """
    start_time = datetime.now(UTC)
    try:
        if not request.container_name:
            if request.code_content:
                # Ad-hoc mode cannot work: every tool runs via coding-containers
                # exec, and there is no container to exec into (this pod has no
                # tools installed either). Refuse it up front -- never run it and
                # report "inconclusive" for something that could not succeed.
                raise validation_error(INLINE_CODE_NEEDS_WORKSPACE, "code_content")
            raise bad_request("Either container_name or code_content is required")

        requested_tools = _check_requested_tools(request.tools)

        codebase_id = request.container_name
        container_info = await validate_and_get_container_for_tenant(codebase_id, tenant_id)
        repo_path = Path(container_info.codebase_path)
        repo_url = getattr(container_info, "repo_url", None)
        commit_sha = await resolve_commit_sha(codebase_id, tenant_id)

        # Only non-default knobs are passed, so an analyzer (or test double)
        # that predates them keeps working for the unchanged request shape.
        analyze_kwargs: dict[str, Any] = {"tenant_id": tenant_id}
        if request.paths:
            analyze_kwargs["scope_paths"] = request.paths
        if requested_tools is not None:
            analyze_kwargs["tools"] = requested_tools
        if request.repo != PRIMARY_REPO:
            analyze_kwargs["repo"] = request.repo
        if request.rust_override:
            analyze_kwargs["language"] = "rust"
        if request.workdir:
            analyze_kwargs["workdir"] = request.workdir
        try:
            report = await analyzer.analyze(repo_path, codebase_id, **analyze_kwargs)
        except ScopeMatchesNoFilesError as e:
            unmatched = e.unmatched if e.unmatched is not None else request.paths
            raise validation_error(
                # unmatched == []: every path exists, none holds a source file
                f"paths match no files: {unmatched}" if unmatched else str(e),
                "paths",
                {"paths": request.paths, "unmatched": unmatched},
            ) from e
        except ToolsNotApplicableError as e:
            raise validation_error(
                str(e),
                "tools",
                {
                    "tools": request.tools,
                    "unknown": e.unknown,
                    "accepted": e.accepted,
                    "language": e.language,
                },
            ) from e

        result_id: str | None = None
        # Scoped runs and tool-restricted runs are deliberately not stored -- see
        # the docstring. Neither is a run of a source repo or of one crate: the
        # stored row is the container's (primary repo's) whole-workspace verdict.
        if (
            request.container_name
            and not request.paths
            and requested_tools is None
            and request.repo == PRIMARY_REPO
            and not request.workdir
        ):
            result_id = await _persist_result(
                store,
                report or None,
                tenant_id=tenant_id,
                container_name=request.container_name,
                repo_url=repo_url,
                commit_sha=commit_sha,
            )
        stored = {
            "status": derive_status(report or None),
            "result_id": result_id,
            "commit_sha": commit_sha,
            "persisted": result_id is not None,
            "scope": request.paths,
            # scope paths that exist but hold no source file: given to no tool
            "scope_not_analysed": report.scope_not_analysed if report else [],
            "repo": request.repo,
            "workdir": request.workdir,
            "tools_requested": requested_tools,
        }

        if not report:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            return {
                "analysis_id": str(uuid7()),
                "success": False,
                # No report means nothing was measured: an empty issue list here
                # is NOT a clean result.
                "inconclusive": True,
                "partial": False,
                "tools_attempted": [],
                "tools_succeeded": [],
                "tools_failed": [],
                "language": request.language or "unknown",
                "tools_results": {},
                "overall_score": 0.0,
                "total_issues": 0,
                "issues": [],
                "execution_time": execution_time,
                "created_at": start_time.isoformat(),
                "error": "Analysis produced no results",
                **stored,
            }

        issues_list = [_serialize_quality_issue(issue) for issue in report.issues]

        # Honor the analyzer's honest abstain: when no static-analysis tool
        # completed (inconclusive), overall_score is the -1.0 sentinel — never
        # present it as a measured score, and never claim unqualified success
        # for a scan that measured nothing. Mirrors the inconclusive rendering
        # threaded into /results, /summary, /gate and the WebSocket paths.
        inconclusive = bool(report.inconclusive)
        # Per-tool provenance: which tools ran and how they fared. `tools_results`
        # was historically a hardcoded {} here — reconstruct a lightweight, honest
        # per-tool success map from the report's provenance lists (the full raw
        # per-tool output isn't carried on the report).
        tools_results = {
            name: {"success": name in report.tools_succeeded} for name in report.tools_attempted
        }
        return {
            "analysis_id": report.job_id,
            "success": not inconclusive,
            "inconclusive": inconclusive,
            "partial": report.partial,
            "tools_attempted": report.tools_attempted,
            "tools_succeeded": report.tools_succeeded,
            "tools_failed": report.tools_failed,
            # Report what the analyzer DETECTED, falling back to the request only
            # when the caller pinned a language explicitly. This used to echo the
            # request field unconditionally, so an unspecified language surfaced
            # as "unknown" beside a tools_attempted list of six Python tools.
            "language": request.language or report.language or "unknown",
            "tools_results": tools_results,
            "overall_score": None if inconclusive else report.overall_score,
            "total_issues": len(report.issues),
            "issues": issues_list,
            "execution_time": report.execution_time,
            "created_at": report.generated_at,
            "error": (
                "Analysis inconclusive: no static-analysis tool completed "
                "successfully — this is NOT a pass; re-run or check tool timeouts."
                if inconclusive
                else None
            ),
            **stored,
        }

    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Sync quality analysis failed: %s", e, exc_info=True)
        raise internal_error(str(e)) from e


@router.get("/health", operation_id="quality_health_check")
async def quality_health() -> dict[str, str]:
    """Health check for quality analysis service."""
    return {"status": "healthy", "service": "quality-analysis"}


@router.get("/info", operation_id="get_quality_info", response_model=QualityInfoResponse)
async def quality_info() -> dict[str, Any]:
    """Get quality analysis service information."""
    return {
        "service": "quality-analysis",
        "version": "1.0.0",
        # Same registry analyze-sync validates ``tools`` against.
        "available_tools": enabled_tools_by_language(),
    }


@router.get(
    "/results/{task_id}",
    operation_id="get_quality_results",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_analysis_results(
    task_id: str,
    analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Get quality analysis results by task ID.

    Retrieves the results of a previously triggered quality analysis task.
    If the task is still running, returns the current status. If completed,
    returns the full analysis results including issues, metrics, and recommendations.

    Args:
        task_id: The ID of the analysis task to retrieve results for
        analyzer: The QualityAnalyzer instance to use

    Returns:
        A dictionary containing the analysis results or status information

    Raises:
        HTTPException: If the task is not found or other errors occur
    """
    try:
        # Look up task result in Redis by task_id
        try:
            redis_service = get_redis_service()
            cached_task = await redis_service.get(f"quality_task:{task_id}")
            if cached_task:
                logger.info("Found task result in Redis for task %s", task_id)
                return json.loads(cached_task)
        except (ConnectionError, OSError, RuntimeError) as redis_err:
            logger.warning("Redis lookup failed for task %s: %s", task_id, redis_err)

        # Task not found in Redis — it may still be running or Redis was unavailable
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Analysis is still running or results have expired. "
            "Task results are stored for 24 hours after completion.",
        }
    except HTTPException:
        raise
    except (ValueError, KeyError, RuntimeError, AttributeError, OSError) as e:
        logger.error(
            "Error retrieving analysis results for task %s: %s",
            task_id,
            e,
            exc_info=True,
        )
        raise internal_error(f"Failed to retrieve results for task {task_id}") from e


# ---------------------------------------------------------------------------
# Stored-result read path (coding-analysis#106)
#
# Every GET below reads ``quality_results`` through QualityResultStore and
# NEVER calls analyzer.analyze(): viewing a Quality tab used to re-run every
# lint tool in the tenant's container on each request, and kept nothing.
#
# Authorization is ``require_tenant`` + a strict ``tenant_id`` row match in the
# store — deliberately NOT ``validate_and_get_container_for_tenant`` (same
# contract as ``dependency_snapshots.py``), so a stored result stays readable
# after its container is deprovisioned. Nothing stored → 200 with
# ``status: "not_analyzed"``; never a fabricated score or grade.
# ---------------------------------------------------------------------------

NOT_ANALYZED = "not_analyzed"
#: Stored gate runs that may be reused for the same clean commit.
REUSABLE_GATE_STATUSES = frozenset({"complete", "partial"})
NOT_ANALYZED_MESSAGE = (
    "No stored quality result for this container: nothing was measured. "
    "Run POST /quality/analyze to produce one."
)
INCONCLUSIVE_MESSAGE = (
    "Analysis inconclusive: no static-analysis tool completed "
    "successfully — this is NOT a pass; re-run or check tool timeouts."
)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _stored_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Identity of the stored run a response was rendered from."""
    return {
        "status": row["status"],
        "result_id": row["id"],
        "commit_sha": row["commit_sha"],
        "repo_url": row["repo_url"],
        "analyzed_at": _iso(row["created_at"]),
        "issues_total": row["issues_total"],
        "issues_truncated": bool(row["issues_truncated"]),
    }


def _not_analyzed_meta() -> dict[str, Any]:
    return {
        "status": NOT_ANALYZED,
        "result_id": None,
        "commit_sha": None,
        "repo_url": None,
        "analyzed_at": None,
        "issues_total": 0,
        "issues_truncated": False,
        "message": NOT_ANALYZED_MESSAGE,
    }


def _stored_provenance(row: dict[str, Any] | None) -> dict[str, Any]:
    """``_report_provenance`` for a stored row (or for nothing stored)."""
    if row is None:
        return {
            "inconclusive": True,
            "partial": False,
            "tools_attempted": [],
            "tools_succeeded": [],
            "tools_failed": [],
        }
    return {
        # A failed run measured nothing either: it is inconclusive, not clean.
        "inconclusive": row["status"] in ("inconclusive", "failed"),
        "partial": row["status"] == "partial",
        "tools_attempted": list(row["tools_attempted"] or []),
        "tools_succeeded": list(row["tools_succeeded"] or []),
        "tools_failed": list(row["tools_failed"] or []),
    }


def _stored_score(row: dict[str, Any] | None) -> float | None:
    """Score only for runs that measured something (complete / partial)."""
    if row is None or row["status"] not in ("complete", "partial"):
        return None
    return row["overall_score"]


def _stored_grade(row: dict[str, Any] | None) -> str | None:
    """``_render_grade`` for a stored row.

    * complete → letter grade from the stored score.
    * partial → ``None``: a letter computed from a subset of tools is exactly
      the green badge a reader would trust; the score, ``partial`` and the tool
      lists are still returned so a consumer can say "graded on N of M tools".
    * inconclusive → the analyzer's honest abstain grade (``N/A``), never an F.
    * failed / nothing stored → ``None``.
    """
    if row is None:
        return None
    status = row["status"]
    if status == "complete" and row["overall_score"] is not None:
        return _calculate_grade(row["overall_score"])
    if status == "inconclusive":
        return row["grade"] or "N/A"
    return None


def _serialize_stored_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """``_serialize_quality_issue`` for a stored (JSONB) issue."""
    return {
        "tool": issue.get("tool"),
        "severity": issue.get("severity"),
        "message": issue.get("message"),
        "file_path": issue.get("file_path"),
        "line_number": issue.get("line_number"),
        "column_number": issue.get("column_number"),
        "rule_id": issue.get("rule_id"),
        "repo": issue.get("repo"),
    }


def _stored_severity_counts(row: dict[str, Any]) -> dict[str, int]:
    stats = row.get("issue_stats") or {}
    return {str(k).lower(): int(v) for k, v in (stats.get("severity_counts") or {}).items()}


def _require_tenant_uuid(tenant_id: str) -> None:
    """400 (not a database 500) when the tenant id cannot match the uuid column."""
    try:
        validate_tenant_id(tenant_id)
    except InvalidTenantIdError as e:
        raise bad_request("Invalid tenant id: expected a UUID") from e


async def _load_latest(
    store: QualityResultStore, tenant_id: str, container_name: str
) -> dict[str, Any] | None:
    _require_tenant_uuid(tenant_id)
    try:
        return await store.latest(tenant_id, container_name)
    except Exception as e:  # noqa: BLE001 - a DB failure is a 500, never "not analyzed"
        logger.error("Quality result lookup failed for %s: %s", container_name, e, exc_info=True)
        raise internal_error("Failed to read stored quality result") from e


@router.get(
    "/{container_name}/latest",
    operation_id="get_latest_quality_results",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_latest_quality_results(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Latest STORED quality result for a container (never re-runs the analyzer)."""
    row = await _load_latest(store, tenant_id, container_name)
    if row is None:
        return {
            "container_name": container_name,
            "overall_score": None,
            "grade": None,
            **_stored_provenance(None),
            "issues_count": 0,
            "issues": [],
            "summary": {},
            **_not_analyzed_meta(),
        }
    return {
        "container_name": container_name,
        "overall_score": _stored_score(row),
        "grade": _stored_grade(row),
        **_stored_provenance(row),
        "issues_count": row["issues_total"],
        "issues": [_serialize_stored_issue(issue) for issue in row["issues"] or []],
        "summary": row["summary"] or {},
        "language": row["language"],
        "execution_time": row["execution_time"],
        "error_message": row["error_message"],
        **_stored_meta(row),
    }


@router.get(
    "/{container_name}/issues",
    operation_id="get_quality_issues",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_quality_issues(
    container_name: str,
    severity: str | None = Query(
        None,
        description=(
            "Filter by severity (case-insensitive). One of: "
            + ", ".join(sorted(QUALITY_ISSUE_SEVERITIES))
        ),
    ),
    file_path: str | None = Query(None, description="Filter by file path"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of issues to return"),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Issues from the latest STORED result, with filtering.

    Filters apply to the stored issue list, which is capped at write time;
    ``issues_truncated`` / ``issues_total`` say when it is not the whole result.
    """
    severity_filter = severity.lower() if severity else None
    if severity_filter is not None and severity_filter not in QUALITY_ISSUE_SEVERITIES:
        raise bad_request(
            f"Invalid severity: {severity}",
            details={
                "field": "severity",
                "allowed_values": sorted(QUALITY_ISSUE_SEVERITIES),
            },
        )
    row = await _load_latest(store, tenant_id, container_name)
    if row is None:
        return {
            "container_name": container_name,
            **_stored_provenance(None),
            "total": 0,
            "returned": 0,
            "issues": [],
            **_not_analyzed_meta(),
        }
    issues: list[dict[str, Any]] = list(row["issues"] or [])
    if severity_filter:
        issues = [
            issue for issue in issues if str(issue.get("severity") or "").lower() == severity_filter
        ]
    if file_path:
        issues = [issue for issue in issues if file_path in str(issue.get("file_path") or "")]
    limited_issues = issues[:limit]
    return {
        "container_name": container_name,
        **_stored_provenance(row),
        "total": len(issues),
        "returned": len(limited_issues),
        "issues": [_serialize_stored_issue(issue) for issue in limited_issues],
        **_stored_meta(row),
    }


@router.get(
    "/{container_name}/summary",
    operation_id="get_quality_summary",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_quality_summary(
    container_name: str,
    top_files_limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Maximum number of top problematic files to return",
    ),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Summary of the latest STORED result (never re-runs the analyzer).

    Severity and per-file counts come from aggregates computed over the full
    issue list at write time, so they stay exact when the stored list is capped.
    """
    row = await _load_latest(store, tenant_id, container_name)
    severity_counts = dict.fromkeys(sorted(QUALITY_ISSUE_SEVERITIES), 0)
    if row is None:
        return {
            "container_name": container_name,
            "overall_score": None,
            "grade": None,
            **_stored_provenance(None),
            "total_issues": 0,
            "severity_distribution": severity_counts,
            "top_problematic_files": [],
            "total_files_with_issues": 0,
            "showing_top_n_files": 0,
            "summary": {},
            **_not_analyzed_meta(),
        }
    for key, count in _stored_severity_counts(row).items():
        if key in severity_counts:
            severity_counts[key] += count
    stats = row.get("issue_stats") or {}
    file_counts: dict[str, int] = {
        str(k): int(v) for k, v in (stats.get("file_counts") or {}).items()
    }
    top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:top_files_limit]
    return {
        "container_name": container_name,
        "overall_score": _stored_score(row),
        "grade": _stored_grade(row),
        **_stored_provenance(row),
        "total_issues": row["issues_total"],
        "severity_distribution": severity_counts,
        "top_problematic_files": [
            {"file": file_name, "issue_count": count} for file_name, count in top_files
        ],
        "total_files_with_issues": int(stats.get("files_with_issues", len(file_counts))),
        "showing_top_n_files": len(top_files),
        "summary": row["summary"] or {},
        **_stored_meta(row),
    }


def _abstaining_gate(
    row: dict[str, Any] | None,
    *,
    check_name: str,
    description: str,
    message: str,
    source: Literal["stored", "analyzed"] | None = None,
) -> QualityGateResult:
    """A gate that cannot pass because nothing (conclusive) was measured."""
    provenance = _stored_provenance(row)
    return QualityGateResult(
        passed=False,
        overall_score=0.0,
        grade=(row["grade"] if row and row["status"] == "inconclusive" and row["grade"] else "N/A"),
        inconclusive=True,
        partial=False,
        tools_attempted=provenance["tools_attempted"],
        tools_succeeded=provenance["tools_succeeded"],
        tools_failed=provenance["tools_failed"],
        gate_checks=[
            QualityGateCheck(
                name=check_name,
                description=description,
                passed=False,
                actual=0.0,
                threshold=1.0,
            )
        ],
        recommendations=[message],
        status=row["status"] if row else NOT_ANALYZED,
        result_id=row["id"] if row else None,
        commit_sha=row["commit_sha"] if row else None,
        source=source,
    )


@router.get(
    "/{container_name}/gate",
    operation_id="evaluate_quality_gate",
    response_model=QualityGateResult,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def evaluate_quality_gate(
    container_name: str,
    min_score: float = Query(
        80.0, ge=0.0, le=100.0, description="Minimum overall quality score threshold"
    ),
    max_critical: int = Query(0, ge=0, description="Maximum allowed critical issues"),
    max_high: int = Query(3, ge=0, description="Maximum allowed high severity issues"),
    min_coverage: float | None = Query(
        None, ge=0.0, le=100.0, description="Minimum test coverage percentage"
    ),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
    analyzer: QualityAnalyzer = Depends(get_quality_analyzer),
) -> QualityGateResult:
    """Evaluate quality gate criteria against the container's CURRENT commit.

    The gate is called by the upgrade procedure's ``verify_quality_gate`` step
    right after it changes code, with no analyze call first — so it must grade
    the commit that is checked out NOW, never an older stored run:

    - resolve the container's HEAD sha (bounded, 5s);
    - a stored row for (tenant, container, that sha) → grade it, no analysis
      (``source: "stored"``);
    - otherwise (no row for this sha, or the sha is unresolvable) → run the
      analyzer, persist the run (a persist failure never fails the gate) and
      grade the fresh result (``source: "analyzed"``).

    The gate passes only if ALL checks pass AND the graded run is ``complete``:
    a failed run, an inconclusive run and a partial run (graded on a subset of
    tools) can never pass. This is the ONLY quality GET that may analyze.
    """
    _require_tenant_uuid(tenant_id)
    commit_sha = await resolve_commit_sha(container_name, tenant_id)

    row: dict[str, Any] | None = None
    if commit_sha is not None:
        try:
            row = await store.latest_for_sha(tenant_id, container_name, commit_sha)
        except Exception as e:  # noqa: BLE001 - an unreadable store means "analyze", not 500
            logger.warning(
                "Stored gate lookup failed for %s@%s, analyzing instead: %s",
                container_name,
                commit_sha,
                e,
            )
            row = None

    # Only a finished grade is reusable. A failed or inconclusive run (e.g. a
    # transient tool error) must not pin the gate shut for this commit forever.
    if row is not None and row.get("status") not in REUSABLE_GATE_STATUSES:
        row = None

    source: Literal["stored", "analyzed"]
    if row is not None:
        source = "stored"
    else:
        source = "analyzed"
        container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
        repo_path = Path(container_info.codebase_path)
        repo_url = getattr(container_info, "repo_url", None)
        try:
            report = await analyzer.analyze(repo_path, container_name, tenant_id=tenant_id)
        except HTTPException:
            raise
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Quality gate analysis failed: %s", e, exc_info=True)
            raise internal_error(str(e)) from e
        result_id = await _persist_result(
            store,
            report or None,
            tenant_id=tenant_id,
            container_name=container_name,
            repo_url=repo_url,
            commit_sha=commit_sha,
        )
        # Grade exactly the row that was (or would have been) stored.
        row = build_row(
            report or None,
            tenant_id=tenant_id,
            container_name=container_name,
            repo_url=repo_url,
            commit_sha=commit_sha,
        )
        row["id"] = result_id

    result = await _grade_row(
        row,
        container_name=container_name,
        tenant_id=tenant_id,
        min_score=min_score,
        max_critical=max_critical,
        max_high=max_high,
        min_coverage=min_coverage,
    )
    result.source = source
    return result


async def _grade_row(
    row: dict[str, Any],
    *,
    container_name: str,
    tenant_id: str,
    min_score: float,
    max_critical: int,
    max_high: int,
    min_coverage: float | None,
) -> QualityGateResult:
    """Grade one stored-shape row (``build_row`` output or a DB row)."""
    if row["status"] == "failed":
        return _abstaining_gate(
            row,
            check_name="analysis_exists",
            description="The latest quality analysis must have produced a report",
            message=f"Latest quality analysis failed: {row['error_message'] or 'no report'}",
        )
    if row["status"] == "inconclusive":
        return _abstaining_gate(
            row,
            check_name="analysis_conclusive",
            description="At least one static-analysis tool must complete successfully",
            message=INCONCLUSIVE_MESSAGE,
        )

    try:
        overall_score = float(row["overall_score"] or 0.0)
        partial = row["status"] == "partial"
        counts = _stored_severity_counts(row)
        severity_counts = {
            "critical": counts.get("critical", 0),
            "high": counts.get("high", 0),
        }

        gate_checks: list[QualityGateCheck] = []
        if partial:
            attempted = len(row["tools_attempted"] or [])
            succeeded = len(row["tools_succeeded"] or [])
            gate_checks.append(
                QualityGateCheck(
                    name="analysis_complete",
                    description="Every attempted static-analysis tool must complete",
                    passed=False,
                    actual=float(succeeded),
                    threshold=float(attempted),
                )
            )

        score_check = QualityGateCheck(
            name="minimum_score",
            description=f"Overall quality score must be above {min_score}",
            passed=overall_score >= min_score,
            actual=overall_score,
            threshold=min_score,
        )
        gate_checks.append(score_check)

        critical_check = QualityGateCheck(
            name="critical_issues",
            description=f"No more than {max_critical} critical severity issue{'s' if max_critical != 1 else ''} allowed",
            passed=severity_counts["critical"] <= max_critical,
            actual=float(severity_counts["critical"]),
            threshold=float(max_critical),
        )
        gate_checks.append(critical_check)

        high_check = QualityGateCheck(
            name="high_issues",
            description=f"No more than {max_high} high severity issue{'s' if max_high != 1 else ''} allowed",
            passed=severity_counts["high"] <= max_high,
            actual=float(severity_counts["high"]),
            threshold=float(max_high),
        )
        gate_checks.append(high_check)

        if min_coverage is not None:
            # Estimate coverage based on presence of test files. This needs the
            # container, so it is the one gate input that still requires it.
            container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
            repo_path = Path(container_info.codebase_path)
            all_files = list(repo_path.rglob("*.py"))
            test_files = [
                f for f in all_files if "test" in f.name.lower() or f.parent.name == "tests"
            ]
            estimated_coverage = (
                (len(test_files) / max(len(all_files), 1)) * 100 if all_files else 0.0
            )
            gate_checks.append(
                QualityGateCheck(
                    name="test_coverage",
                    description=f"Code coverage must be above {min_coverage}%",
                    passed=estimated_coverage >= min_coverage,
                    actual=estimated_coverage,
                    threshold=min_coverage,
                )
            )

        all_passed = all(check.passed for check in gate_checks)

        recommendations: list[str] = []
        if partial:
            recommendations.append(
                "Analysis partial: only "
                f"{len(row['tools_succeeded'] or [])} of {len(row['tools_attempted'] or [])} "
                "tools completed — the gate cannot pass on a subset; re-run the analysis."
            )
        if not score_check.passed:
            score_gap = min_score - overall_score
            recommendations.append(
                f"Improve overall score by {score_gap:.1f} points to meet minimum threshold of {min_score}"
            )
        if not critical_check.passed:
            critical_count = severity_counts["critical"]
            recommendations.append(
                f"Fix {critical_count} critical security issue{'s' if critical_count != 1 else ''} to pass the quality gate"
            )
        if not high_check.passed:
            high_count = severity_counts["high"]
            high_excess = high_count - max_high
            recommendations.append(
                f"Reduce high severity issues by {high_excess} (currently {high_count}, max allowed: {max_high})"
            )
        if min_coverage is not None:
            coverage_check_result = next(
                (c for c in gate_checks if c.name == "test_coverage"), None
            )
            if coverage_check_result and not coverage_check_result.passed:
                coverage_gap = min_coverage - coverage_check_result.actual
                recommendations.append(
                    f"Increase test coverage by {coverage_gap:.1f}% to meet minimum threshold of {min_coverage}%"
                )
        if all_passed:
            recommendations.append(
                "All quality gate checks passed! Consider setting higher thresholds to drive continuous improvement."
            )
        else:
            recommendations.append(
                "Consider refactoring complex functions to improve maintainability"
            )

        provenance = _stored_provenance(row)
        return QualityGateResult(
            passed=all_passed,
            overall_score=overall_score,
            grade=_stored_grade(row) or "N/A",
            partial=partial,
            tools_attempted=provenance["tools_attempted"],
            tools_succeeded=provenance["tools_succeeded"],
            tools_failed=provenance["tools_failed"],
            gate_checks=gate_checks,
            recommendations=recommendations,
            status=row["status"],
            result_id=row["id"],
            commit_sha=row["commit_sha"],
        )
    except HTTPException:
        raise
    except (ValueError, KeyError, RuntimeError, AttributeError, OSError) as e:
        logger.error("Error during quality gate evaluation: %s", e, exc_info=True)
        raise internal_error(str(e)) from e


@router.get(
    "/{container_name}/trend",
    operation_id="get_quality_trend",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_quality_trend(
    container_name: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to include in trend"),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Quality score over time from STORED results.

    Only ``complete`` runs become trend points: a partial score is graded on a
    subset of tools and an inconclusive / failed run measured nothing, so
    plotting either would draw a dip or a spike that is not a change in the code.
    ``skipped_points`` counts the runs left out, so the omission is visible.
    """
    _require_tenant_uuid(tenant_id)
    since = datetime.now(UTC) - timedelta(days=days)
    try:
        rows = await store.history(tenant_id, container_name, limit=TREND_MAX_ROWS, since=since)
    except Exception as e:  # noqa: BLE001
        logger.error("Quality history lookup failed for %s: %s", container_name, e, exc_info=True)
        raise internal_error("Failed to read stored quality history") from e

    complete = [r for r in rows if r["status"] == "complete" and r["overall_score"] is not None]
    skipped = len(rows) - len(complete)
    trend_data = [
        {
            "timestamp": _iso(r["created_at"]),
            "score": r["overall_score"],
            "grade": _stored_grade(r),
            "commit_sha": r["commit_sha"],
            "result_id": r["id"],
        }
        for r in reversed(complete)  # oldest → newest
    ]
    if not trend_data:
        return {
            "container_name": container_name,
            "period_days": days,
            "trend": [],
            "skipped_points": skipped,
            "average_score": None,
            "update": None,
            "status": NOT_ANALYZED if not rows else "no_complete_results",
            "message": (
                "No complete quality results in this period. Only complete runs are "
                "trend points; run POST /quality/analyze to build trend data."
            ),
        }
    scores = [float(r["overall_score"]) for r in reversed(complete)]
    average_score = round(sum(scores) / len(scores), 1)
    score_update = scores[-1] - scores[0] if len(scores) > 1 else 0.0
    return {
        "container_name": container_name,
        "period_days": days,
        "trend": trend_data,
        "skipped_points": skipped,
        "average_score": average_score,
        "update": round(score_update, 1),
        "status": "ok",
    }


@router.get(
    "/{container_name}/history",
    operation_id="get_quality_history",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_quality_history(
    container_name: str,
    limit: int = Query(50, ge=1, le=500, description="Maximum number of runs to return"),
    tenant_id: str = Depends(require_tenant),
    store: QualityResultStore = Depends(get_quality_result_store),
) -> dict[str, Any]:
    """Stored quality runs for a container, newest first (every status)."""
    _require_tenant_uuid(tenant_id)
    try:
        rows = await store.history(tenant_id, container_name, limit=limit)
    except Exception as e:  # noqa: BLE001
        logger.error("Quality history lookup failed for %s: %s", container_name, e, exc_info=True)
        raise internal_error("Failed to read stored quality history") from e
    return {
        "container_name": container_name,
        "status": "ok" if rows else NOT_ANALYZED,
        "count": len(rows),
        "results": [
            {
                "id": r["id"],
                # A container name can be re-provisioned from a different repo;
                # consumers must attribute each run to the repo it graded.
                "repo_url": r.get("repo_url"),
                "commit_sha": r["commit_sha"],
                "status": r["status"],
                "overall_score": _stored_score(r),
                "grade": _stored_grade(r),
                "tools_attempted": len(r["tools_attempted"] or []),
                "tools_succeeded": len(r["tools_succeeded"] or []),
                "created_at": _iso(r["created_at"]),
            }
            for r in rows
        ],
    }


@router.websocket("/ws/{container_name}")
async def quality_websocket(
    websocket: WebSocket,
    container_name: str,
) -> None:
    """WebSocket endpoint for real-time quality analysis updates.

    Provides live updates for:
    - Quality report changes
    - Quality gate evaluation results
    - Analysis status (started, completed, failed)
    - Periodic heartbeat messages

    The frontend expects messages in this format:
    ```json
    {
        "type": "quality_report" | "quality_gate_result" | "analysis_started" | "analysis_completed" | "analysis_failed" | "heartbeat",
        "data": {...},
        "codebase_id": "container-name",
        "timestamp": "2025-10-31T12:00:00Z"
    }
    ```

    Args:
        websocket: The WebSocket connection
        container_name: The container to monitor for quality updates

    Example Connection:
        ```javascript
        const ws = new WebSocket(`${window.location.origin.replace(/^http/, 'ws')}/api/quality/ws/my-container?tenant_id=xxx`);
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Quality update:', message.type, message.data);
        };
        ```
    """
    # Accept the WebSocket handshake BEFORE doing any validation. Calling
    # `websocket.close(code=…)` before `accept()` makes Starlette/uvicorn
    # reject the upgrade with HTTP 403, and the WS protocol then has no
    # way to deliver the close code to the client — the client sees a
    # generic abnormal-close (code 1006) instead of the intended 1008.
    # Accepting first lets the subsequent close() emit a real WS Close
    # frame so consumers (e.g. fr-backend's quality proxy) can translate
    # the policy-violation code into a typed NO_CONTAINER frame for the
    # frontend.
    await websocket.accept()

    # ?tenant_id= is deliberately NOT read, and that must stay true. The ingress
    # Lua filter (gitops-production .../envoy-filters/ext-authz-enforce.yaml)
    # unconditionally strips a client x-tenant-id and re-injects only what
    # license-authz resolved. It strips HEADERS; it cannot strip a query string —
    # so honouring ?tenant_id= let anyone denied the header pass the same value
    # as a query param and read another tenant's stream. (The javascript in this
    # function's docstring still shows ?tenant_id= — it is ignored.)
    #
    # That edge argument is true NORTH-SOUTH and false POD-TO-POD. The Lua filter
    # only sees traffic arriving through the ingress gateway; an east-west caller
    # inside the mesh reaches this service directly and its X-Tenant-Id is
    # scrubbed by nothing.
    #
    # AuthMiddleware cannot cover the gap: it is a BaseHTTPMiddleware, and
    # Starlette never invokes dispatch on a `websocket` scope. So once the HTTP
    # paths stopped trusting a bare header (TRUST_TENANT_HEADER=false), this
    # handler was among the last places in the fleet where any pod could assert
    # any tenant and be believed. No config flip can close that — the handler has
    # to resolve identity itself, which is what this does.
    #
    # The javascript in this function's docstring also shows no Authorization
    # header; a browser cannot set one on a WS handshake, which is why the live
    # panel reaches this route through fr's proxy, and why THAT proxy is the
    # thing that must present svc-fr-backend's licence. It now does.
    identity = await resolve_websocket_identity(websocket)

    # ⚠️ Decided by `should_close` ALONE. Re-deriving the decision here
    # (`identity.outcome is Outcome.REFUSED`) is exactly the bypass that kept the
    # uncredentialed handshake alive through coding-client 1.14.0 — the SDK owns
    # the policy, this handler owns the close. `outcome` is read below only to
    # LABEL the log line; it never gates.
    if identity.should_close:
        # The uncredentialed residual is CLOSED as of coding-client 1.15.0.
        #
        # It was deliberate while it held: a browser CANNOT set an Authorization
        # header on a WebSocket handshake (the WebSocket API has no way to pass
        # one), so a browser-originated socket carries a credential only if a
        # server-side proxy puts one there, and refusing before the proxies were
        # converted would have broken the live quality panel rather than
        # hardening it. So NO_CREDENTIAL used to fall back to a raw
        # `X-Tenant-Id` read and WARN, naming the peer.
        #
        # That fallback was never a safety property — it was the instrument for
        # finding the callers still to convert, and a credential-less handshake
        # carrying only a forged X-Tenant-Id was BELIEVED: this route answered
        # `connection_established` for a container the caller held no licence
        # for (reproduced from inside the pod). The instrument has reported:
        # over ~18h of log retention here the WARNING fired ZERO times, and 5
        # times on coding-realtime, every one from peer=127.0.0.1 — a
        # verification probe. No real caller ever used it. Meanwhile fr now
        # presents svc-fr-backend's licence on both of its WS proxies (driven
        # live: 101 with auth, 403 without) and coding-gateway attaches its own.
        if identity.outcome is Outcome.NO_CREDENTIAL:
            client = getattr(websocket, "client", None)
            logger.warning(
                "Refused an uncredentialed quality WebSocket handshake "
                "(peer=%s:%s, container=%s). The caller presented no verifiable "
                "licence, so any X-Tenant-Id it sent proves nothing. Send "
                "Authorization: Bearer <licence>.",
                getattr(client, "host", None),
                getattr(client, "port", None),
                container_name,
            )
        # identity.reason never names the asserted tenant: this close frame is
        # visible to whoever sent the assertion, and echoing the value back
        # would confirm the guess.
        await websocket.close(
            code=1008,
            reason=identity.reason or "this credential may not act for the tenant it asserted",
        )
        return

    tenant_id = identity.tenant_id

    if not tenant_id:
        # Defence in depth: a RESOLVED identity always carries a tenant, so
        # reaching here means the licence introspected to an empty one. The
        # reason deliberately does NOT tell the caller to send X-Tenant-Id —
        # since 1.15.0 a bare header is not a credential, and advertising it
        # would send an unauthenticated caller down a path that can only refuse.
        await websocket.close(
            code=1008,
            reason="A verifiable licence is required on this handshake.",
        )
        return

    # Validate container exists
    try:
        await validate_and_get_container_for_tenant(container_name, tenant_id)
    except HTTPException as e:
        await websocket.close(code=1008, reason=f"Container validation failed: {e.detail}")
        return

    logger.info("Quality WebSocket connected for container: %s", container_name)

    # Send connection established message
    try:
        await websocket.send_json(
            {
                "type": "connection_established",
                "codebase_id": container_name,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "message": "Quality WebSocket connection established",
                    "container_name": container_name,
                },
            }
        )
    except Exception as e:
        logger.error("Failed to send connection message: %s", e)
        await websocket.close()
        return

    # Create analyzer for quality checks
    analyzer = get_quality_analyzer()

    # Heartbeat task
    async def send_heartbeat():
        """Send periodic heartbeat messages to keep connection alive."""
        while True:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "codebase_id": container_name,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            except Exception as e:
                logger.debug("Heartbeat failed: %s", e)
                break

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())

    try:
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                message_type = data.get("type")

                if message_type == "request_analysis":
                    # Send analysis started event
                    await websocket.send_json(
                        {
                            "type": "analysis_started",
                            "codebase_id": container_name,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "data": {"message": "Quality analysis started"},
                        }
                    )

                    # Run quality analysis
                    try:
                        container_info = await validate_and_get_container_for_tenant(
                            container_name, tenant_id
                        )
                        repo_path = Path(container_info.codebase_path)
                        commit_sha = await resolve_commit_sha(container_name, tenant_id)

                        # Run analysis (path is inside the container, not on this filesystem)
                        result = await analyzer.analyze(
                            repo_path, container_name, tenant_id=tenant_id
                        )

                        # Store the run like POST /analyze does; never fatal to the stream.
                        await _persist_result(
                            get_quality_result_store(),
                            result or None,
                            tenant_id=tenant_id,
                            container_name=container_name,
                            repo_url=getattr(container_info, "repo_url", None),
                            commit_sha=commit_sha,
                        )

                        # Count severity issues
                        severity_counts = {
                            "critical": 0,
                            "high": 0,
                            "moderate": 0,
                            "low": 0,
                        }
                        for issue in result.issues:
                            severity = getattr(issue, "severity", "unknown")
                            if severity in severity_counts:
                                severity_counts[severity] += 1

                        # Convert issues to dicts for JSON serialization
                        issues_list = []
                        for issue in result.issues[:100]:  # Limit to 100 issues for performance
                            issues_list.append(
                                {
                                    "file_path": issue.file_path,
                                    "line_number": issue.line_number,
                                    "severity": getattr(issue, "severity", "unknown"),
                                    "message": issue.message,
                                    "rule_id": getattr(issue, "rule_id", None),
                                    "tool": getattr(issue, "tool", "unknown"),
                                }
                            )

                        # Send quality report
                        await websocket.send_json(
                            {
                                "type": "quality_report",
                                "codebase_id": container_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "data": {
                                    "overall_score": result.overall_score,
                                    "grade": _render_grade(result),
                                    "inconclusive": result.inconclusive,
                                    "partial": result.partial,
                                    "tools_attempted": result.tools_attempted,
                                    "tools_succeeded": result.tools_succeeded,
                                    "tools_failed": result.tools_failed,
                                    "total_issues": len(result.issues),
                                    "severity_distribution": severity_counts,
                                    "issues": issues_list,
                                    "summary": result.summary,
                                },
                            }
                        )

                        # Evaluate quality gate with default thresholds
                        min_score = 80.0
                        max_critical = 0
                        max_high = 3

                        gate_checks = []

                        # Check 1: Minimum score
                        score_check = {
                            "name": "minimum_score",
                            "description": f"Overall quality score must be above {min_score}",
                            "passed": result.overall_score >= min_score,
                            "actual": result.overall_score,
                            "threshold": min_score,
                        }
                        gate_checks.append(score_check)

                        # Check 2: Critical issues
                        critical_check = {
                            "name": "critical_issues",
                            "description": f"No more than {max_critical} critical issues allowed",
                            "passed": severity_counts["critical"] <= max_critical,
                            "actual": float(severity_counts["critical"]),
                            "threshold": float(max_critical),
                        }
                        gate_checks.append(critical_check)

                        # Check 3: High issues
                        high_check = {
                            "name": "high_issues",
                            "description": f"No more than {max_high} high severity issues allowed",
                            "passed": severity_counts["high"] <= max_high,
                            "actual": float(severity_counts["high"]),
                            "threshold": float(max_high),
                        }
                        gate_checks.append(high_check)

                        all_passed = all(check["passed"] for check in gate_checks)

                        # Send quality gate result
                        await websocket.send_json(
                            {
                                "type": "quality_gate_result",
                                "codebase_id": container_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "data": {
                                    "passed": all_passed and not result.inconclusive,
                                    "overall_score": result.overall_score,
                                    "grade": _render_grade(result),
                                    "inconclusive": result.inconclusive,
                                    "partial": result.partial,
                                    "tools_attempted": result.tools_attempted,
                                    "tools_succeeded": result.tools_succeeded,
                                    "tools_failed": result.tools_failed,
                                    "gate_checks": gate_checks,
                                    "recommendations": _generate_recommendations(
                                        result.issues, inconclusive=result.inconclusive
                                    ),
                                },
                            }
                        )

                        # Send analysis completed event
                        await websocket.send_json(
                            {
                                "type": "analysis_completed",
                                "codebase_id": container_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "data": {"message": "Quality analysis completed successfully"},
                            }
                        )

                    except Exception as analysis_error:
                        logger.error("Quality analysis failed: %s", analysis_error, exc_info=True)

                        # Send analysis failed event
                        await websocket.send_json(
                            {
                                "type": "analysis_failed",
                                "codebase_id": container_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "data": {"error": str(analysis_error)},
                            }
                        )

                elif message_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "codebase_id": container_name,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

            except WebSocketDisconnect:
                logger.info("Quality WebSocket disconnected for container: %s", container_name)
                break
            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e, exc_info=True)
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "codebase_id": container_name,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "data": {"error": str(e)},
                        }
                    )
                except Exception:
                    break

    finally:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        logger.info("Quality WebSocket closed for container: %s", container_name)
