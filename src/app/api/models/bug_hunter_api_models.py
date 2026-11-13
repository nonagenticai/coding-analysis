"""Bug Hunter API Models for V2 API.

This module contains Pydantic models for Bug Hunter API endpoints.
These models provide structured request/response handling for bug analysis,
bug hunting operations, and status tracking throughout the application.

The models ensure:
    - Type-safe request/response handling with comprehensive validation
    - Standardized response formats for bug hunting operations
    - Consistent status tracking for bug analysis workflows
    - Reusable structures for bug detection and tracking

Example:
    ```python
    from bug_hunter_api_models import BugHunterStatusResponse

    status = BugHunterStatusResponse(
        container_name="my-container",
        last_hunt_timestamp=datetime.now(timezone.utc),
        bugs_found_count=5,
        bugs_fixed_count=2,
        analysis_status="idle"
    )
    ```

Note:
    All models use Pydantic v2 with ConfigDict for configuration and
    include comprehensive JSON schema examples for OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...types import SAFE_STRING_PATTERN


class BugHunterAnalysisStatus(StrEnum):
    """Enumeration of bug hunting analysis statuses.

    This enum defines all possible states for bug hunting operations,
    providing consistent status tracking throughout the bug analysis workflow.

    Attributes:
        IDLE: No active bug hunting operation in progress
        HUNTING: Currently scanning for bugs in the codebase
        ANALYZING: Analyzing discovered bugs for patterns and root causes
        FIXING: Generating and applying bug fixes
        COMPLETED: Bug hunting operation has completed successfully
        ERROR: Bug hunting operation encountered an error
    """

    IDLE = "idle"
    HUNTING = "hunting"
    ANALYZING = "analyzing"
    FIXING = "fixing"
    COMPLETED = "completed"
    ERROR = "error"


class BugHuntJobStatus(StrEnum):
    """Lifecycle states for an asynchronous bug hunt."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BugHuntRequest(BaseModel):
    """Inputs safe to persist and execute later in a worker process."""

    error_logs: str | None = Field(
        "",
        description="Runtime error output to analyze; empty scans the repository",
    )
    error_logs_base64: str | None = Field(None, description="Base64-encoded runtime error output")
    context: dict[str, Any] | None = Field(
        None, description="Optional runtime and environment context"
    )
    callback_url: str | None = Field(
        None,
        description=(
            "Optional URL POSTed once with the hunt's final status (completed or failed), "
            "e.g. an n8n Wait-node resume URL. Must start with one of the service's "
            "configured callback prefixes, else the hunt is refused (422)."
        ),
    )


class BugHuntAcceptedResponse(BaseModel):
    """Response returned immediately after a bug hunt is queued."""

    hunt_id: str = Field(..., description="Stable identifier for this hunt")
    container_name: str = Field(..., description="Container/repository being analyzed")
    status: BugHuntJobStatus = Field(BugHuntJobStatus.QUEUED)
    created_at: datetime = Field(..., description="Time the hunt was queued")
    links: dict[str, str] = Field(..., description="Status, summary report, and path-findings URLs")


class BugHuntStatusResponse(BaseModel):
    """Current state and counters for one asynchronous hunt."""

    hunt_id: str
    container_name: str
    status: BugHuntJobStatus
    errors_found: int = Field(0, ge=0)
    fixes_generated: int = Field(0, ge=0)
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime


class BugHuntFlaggedFile(BaseModel):
    """One file represented in the compact summary report."""

    file_path: str
    file_name: str
    bug_count: int = Field(..., ge=1)
    severity_counts: dict[str, int] = Field(default_factory=dict)


class BugHuntSummaryResponse(BugHuntStatusResponse):
    """Compact report metadata plus paginated flagged file paths."""

    root_cause_analysis: str
    fix_summary: str
    estimated_effort: str
    risk_assessment: str
    analysis_time: float = Field(..., ge=0)
    # Coverage provenance (services.workspace_hunt). fr's bug_report/v1 stores a
    # hunt only when scanned_files == tracked_source_files and commit_sha is set.
    # null = not measured; never 0-as-unknown, never derived from each other.
    scanned_files: int | None = Field(
        None, ge=0, description="Git-tracked source files this hunt actually scanned"
    )
    tracked_source_files: int | None = Field(
        None, ge=0, description="Git-tracked source files in the hunt's scope"
    )
    commit_sha: str | None = Field(
        None, description="Workspace HEAD; the hunt scanned exactly this commit's tree"
    )
    worktree_dirty: bool | None = Field(
        None,
        description="True when the workspace had uncommitted changes, which were NOT scanned",
    )
    # The declared scope: both counts exclude files under these directory names.
    excluded_dirs: list[str] | None = Field(
        None,
        description="Directory names excluded from both file counts (declared hunt scope)",
    )
    excluded_entry_kinds: list[str] | None = Field(
        None,
        description=(
            "Tree entry kinds excluded from both file counts (declared hunt scope): "
            "non_utf8_path / submodule / symlink"
        ),
    )
    agent_sha: str | None = Field(
        None, description="sha256 of the hunt agent's MANIFEST.sha256 that produced the findings"
    )
    total_flagged_files: int = Field(..., ge=0)
    flagged_files: list[BugHuntFlaggedFile]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)


class BugHuntFinding(BaseModel):
    """One detailed bug and generated fixes associated with its file."""

    finding_id: int
    file_path: str
    file_name: str
    line_number: int = Field(0, ge=0)
    severity: str
    bug: dict[str, Any]
    suggested_fixes: list[dict[str, Any]] = Field(default_factory=list)
    fingerprint: str | None = Field(
        None,
        description=(
            "Content fingerprint (sha256 hex) emitted by the hunt agent: the finding's "
            "identity for a recheck, stable against unrelated edits. finding_id is a row "
            "id, not an identity. null = not fingerprinted (log-analysis finding)."
        ),
    )


class BugHuntPathFindingsResponse(BaseModel):
    """Paginated bugs under a repository-relative file or directory path."""

    hunt_id: str
    container_name: str
    path: str
    # The hunt's provenance, so a consumer can check a finding's precondition
    # (complete coverage at a named commit, which agent) without a second call.
    # null = not measured, exactly as on the summary report.
    commit_sha: str | None = None
    scanned_files: int | None = Field(None, ge=0)
    tracked_source_files: int | None = Field(None, ge=0)
    worktree_dirty: bool | None = None
    excluded_dirs: list[str] | None = None
    excluded_entry_kinds: list[str] | None = None
    agent_sha: str | None = None
    recursive: bool = True
    matched_files: int = Field(..., ge=0)
    total_findings: int = Field(..., ge=0)
    findings: list[BugHuntFinding]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)


class BugHunterStatusResponse(BaseModel):
    """Response model for bug hunter status endpoint.

    This model provides comprehensive status information about a container's
    bug hunting operations, including recent activity, bug counts, and
    current analysis status.

    Attributes:
        container_name (str): Name of the container being monitored
        last_hunt_timestamp (Optional[datetime]): Timestamp of the last bug hunt operation
        bugs_found_count (int): Total number of bugs discovered in current hunt
        bugs_fixed_count (int): Total number of bugs fixed
        analysis_status (BugHunterAnalysisStatus): Current status of the analysis
        last_error (Optional[str]): Last error message if analysis_status is ERROR
        total_bugs_found (Optional[int]): Total bugs found across all hunts
        total_bugs_fixed (Optional[int]): Total bugs fixed across all hunts

    Example:
        ```python
        response = BugHunterStatusResponse(
            container_name="python-backend",
            last_hunt_timestamp=datetime.now(timezone.utc),
            bugs_found_count=3,
            bugs_fixed_count=1,
            analysis_status="idle",
            total_bugs_found=15,
            total_bugs_fixed=12
        )
        ```

    Note:
        last_hunt_timestamp can be None if no hunt has been performed yet.
        analysis_status transitions follow the workflow: idle -> hunting -> analyzing -> fixing -> completed/error.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "python-backend",
                "last_hunt_timestamp": "2025-12-08T12:30:45Z",
                "bugs_found_count": 3,
                "bugs_fixed_count": 1,
                "analysis_status": "idle",
                "last_error": None,
                "total_bugs_found": 15,
                "total_bugs_fixed": 12,
            }
        }
    )

    container_name: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Name of the container",
    )
    last_hunt_timestamp: datetime | None = Field(
        None, description="Timestamp of the last bug hunt operation"
    )
    bugs_found_count: int = Field(0, ge=0, description="Number of bugs found in current hunt")
    bugs_fixed_count: int = Field(0, ge=0, description="Number of bugs fixed")
    analysis_status: BugHunterAnalysisStatus = Field(
        BugHunterAnalysisStatus.IDLE, description="Current analysis status"
    )
    last_error: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="Last error message if any"
    )
    total_bugs_found: int | None = Field(
        None, ge=0, description="Total bugs found across all hunts"
    )
    total_bugs_fixed: int | None = Field(
        None, ge=0, description="Total bugs fixed across all hunts"
    )


class BugHunterStartHuntRequest(BaseModel):
    """Request model for starting a bug hunting session.

    This model defines the parameters for initiating a new bug hunting operation
    on a specific container's codebase.

    Attributes:
        analysis_type (str): Type of analysis to perform (e.g., "full", "quick", "targeted")
        include_patterns (Optional[List[str]]): File patterns to include in analysis
        exclude_patterns (Optional[List[str]]): File patterns to exclude from analysis
        max_depth (Optional[int]): Maximum directory depth to scan

    Example:
        ```python
        request = BugHunterStartHuntRequest(
            analysis_type="full",
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["**/test_*", "**/node_modules/**"],
            max_depth=10
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_type": "full",
                "include_patterns": ["*.py", "*.js"],
                "exclude_patterns": ["**/test_*"],
                "max_depth": 10,
            }
        }
    )

    analysis_type: str = Field(
        "full",
        pattern=SAFE_STRING_PATTERN,
        description="Type of analysis (full, quick, targeted)",
    )
    include_patterns: list[str] | None = Field(None, description="File patterns to include")
    exclude_patterns: list[str] | None = Field(None, description="File patterns to exclude")
    max_depth: int | None = Field(None, ge=1, description="Maximum directory depth")


class BugHuntFindingRecheckResponse(BaseModel):
    """A VERIFIED re-run of the detector on one finding's file at the hunted commit.

    Only returned when the file was actually scanned at ``commit_sha`` by the same
    agent that ran the hunt. Any failure to get there is an error status
    (``recheck_unverifiable``, ``agent_drift`` ...), never ``reproduced: false``.
    """

    hunt_id: str
    container_name: str
    finding_id: int
    path: str
    reproduced: bool = Field(
        ...,
        description=(
            "True iff a finding with the SAME fingerprint is present in the file at "
            "commit_sha. False only after a verified scan of that file found none."
        ),
    )
    finding_fingerprint: str
    matched_finding: dict[str, Any] | None = Field(
        None, description="The agent's finding at commit_sha with that fingerprint"
    )
    commit_sha: str
    agent_sha: str
    findings_at_commit: int = Field(..., ge=0, description="Findings in the file at commit_sha")
    reason: str = Field(
        ..., description="fingerprint_present_at_commit | fingerprint_absent_at_commit"
    )
