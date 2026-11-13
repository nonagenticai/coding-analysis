"""Request/response models for the contract analysis API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from processing.workspace_repos import PRIMARY_REPO, RepoSelector


class SourceInput(BaseModel):
    """Source code input -- provide either source (inline) or file_path."""

    source: str | None = Field(default=None, description="Inline source code string")
    file_path: str | None = Field(default=None, description="Absolute filesystem path")
    label: str = Field(default="", description="Human-readable label for error messages")
    language: str | None = Field(
        default=None,
        description=(
            "Route sources only: python (default) or typescript/javascript; "
            "inferred from file_path/label extension when omitted"
        ),
    )


class ContractPair(BaseModel):
    """One client-to-service contract to scan."""

    name: str = Field(..., description="e.g. 'AnalyzerClient -> analyzer_service'")
    client: SourceInput = Field(..., description="HTTP client source file")
    schemas: list[SourceInput] = Field(..., min_length=1, description="Service schema sources")
    routes: list[SourceInput] = Field(..., min_length=1, description="Service route sources")


class ScanConfig(BaseModel):
    """Control which checks are enabled."""

    check_routes: bool = True
    check_methods: bool = True
    check_request_fields: bool = True
    check_response_fields: bool = True
    check_types: bool = True
    check_query_params: bool = True
    check_syntax: bool = True
    check_status_codes: bool = True
    fail_on_warnings: bool = False


class ScanRequest(BaseModel):
    """Top-level scan request."""

    contracts: list[ContractPair] = Field(..., min_length=1, max_length=50)
    config: ScanConfig = Field(default_factory=ScanConfig)


class ViolationSeverity(StrEnum):
    FAIL = "FAIL"
    WARN = "WARN"


class ViolationResponse(BaseModel):
    severity: ViolationSeverity
    check: str
    message: str
    client_location: str = ""
    service_location: str = ""
    closest_match: str | None = None
    root_cause: str = ""
    suggestion: str = ""


class ContractResultResponse(BaseModel):
    name: str
    passed: bool
    violations: list[ViolationResponse] = []
    stats: dict[str, int] = {}
    parse_error: str | None = None


class ScanResponse(BaseModel):
    passed: bool
    total_contracts: int
    passed_contracts: int
    failed_contracts: int
    total_violations: int
    total_warnings: int
    contracts: list[ContractResultResponse]
    processing_time_seconds: float
    scanner_version: str = "1.0"


class ScanCapabilities(BaseModel):
    checks: list[str]
    language: str = "python"
    route_languages: list[str] = ["python", "typescript", "javascript", "rust"]
    call_site_languages: list[str] = ["python", "javascript", "typescript", "svelte", "vue", "rust"]
    parser: str = "tree-sitter"
    max_contracts_per_request: int = 50
    supports_file_paths: bool
    supports_inline_source: bool = True


# ---------------------------------------------------------------------------
# Call sites: the consumer side (POST /api/analysis/contracts/call-sites)
# ---------------------------------------------------------------------------

#: Per-file and per-request source caps, both modes (inline and workspace). A file
#: over MAX_FILE_BYTES is skipped (workspace: listed in ``files_skipped_large``;
#: inline: the request is rejected 422). Workspace reads stop at MAX_REQUEST_BYTES
#: (``truncation_reason`` ``max_bytes``); an inline body over it is rejected 422.
#: The structure routes (``models.structure_models``) use these same two values.
MAX_FILE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 20 * 1024 * 1024
MAX_INLINE_FILES = 500


class CallSiteKind(StrEnum):
    HTTP = "http"
    EVENT = "event"


class InlineSourceFile(BaseModel):
    """One file sent inline: a repo-relative path (for language + reporting) and its content."""

    path: str = Field(..., min_length=1, max_length=1024)
    source: str = Field(..., max_length=MAX_FILE_BYTES)


class WorkspaceScanRequest(RepoSelector):
    """ONE workspace (``container_name``) or a set of inline ``sources``, filtered."""

    container_name: str | None = Field(
        default=None, description="Tenant workspace to read through coding-containers"
    )
    sources: list[InlineSourceFile] | None = Field(
        default=None,
        max_length=MAX_INLINE_FILES,
        description=(
            f"Inline files instead of a workspace (each <= {MAX_FILE_BYTES} bytes UTF-8, "
            f"all together <= {MAX_REQUEST_BYTES})"
        ),
    )
    paths: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Repo-relative path prefixes to scan (default: the whole repository)",
    )
    include_tests: bool = Field(default=False, description="Also scan test/spec/fixture files")
    max_files: int = Field(default=2000, ge=1, le=10000)

    @model_validator(mode="after")
    def _repo_needs_a_workspace(self) -> WorkspaceScanRequest:
        if self.repo != PRIMARY_REPO and self.container_name is None:
            raise ValueError("'repo' selects a workspace repo and requires 'container_name'")
        return self

    @model_validator(mode="after")
    def _inline_size_caps(self) -> WorkspaceScanRequest:
        """Reject oversized inline sources (422) before anything is parsed."""
        total = 0
        for item in self.sources or ():
            size = len(item.source.encode("utf-8"))
            if size > MAX_FILE_BYTES:
                raise ValueError(
                    f"sources[{item.path}] is {size} bytes; the per-file limit is {MAX_FILE_BYTES}"
                )
            total += size
        if total > MAX_REQUEST_BYTES:
            raise ValueError(
                f"inline sources total {total} bytes; the per-request limit is {MAX_REQUEST_BYTES}"
            )
        return self


class ScanLimits(BaseModel):
    """The bounds this response was produced under."""

    max_file_bytes: int = MAX_FILE_BYTES
    max_request_bytes: int = MAX_REQUEST_BYTES
    deadline_seconds: float = Field(
        ..., description="Extraction wall-clock deadline (worker process killed past it)"
    )


class CallSitesRequest(WorkspaceScanRequest):
    """Outbound call sites of ONE workspace (``container_name``) or of inline ``sources``."""

    languages: list[str] | None = Field(
        default=None,
        description="Subset of python, javascript, typescript, svelte, vue, rust (default: all)",
    )
    kinds: list[CallSiteKind] = Field(
        default_factory=lambda: [CallSiteKind.HTTP, CallSiteKind.EVENT]
    )
    max_call_sites: int = Field(default=5000, ge=1, le=50000)


class CallSiteResponse(BaseModel):
    file: str
    line: int
    kind: CallSiteKind
    method: str | None = Field(
        default=None, description="http: GET/POST/...; null when the method is dynamic"
    )
    operation: str | None = Field(default=None, description="event: publish | subscribe")
    target: str = Field(
        ...,
        description=(
            "http: path normalised exactly like scan_contracts' routes ({param}, no trailing "
            "slash, no query); event: topic/subject with holes as {param}"
        ),
    )
    client: str
    symbol: str | None = Field(default=None, description="Enclosing function/method, dotted")
    confidence: str = Field(..., description="exact (fully literal) | template (has {param})")
    base: str | None = Field(default=None, description="Stripped base-URL expression")
    host: str | None = Field(default=None, description="host[:port] of an absolute URL")
    raw: str = Field(default="", description="The target expression as written (truncated)")
    base_source_kind: str | None = Field(
        default=None,
        description=(
            "http: where the base URL comes from: literal (absolute URL / literal-bound name), "
            "env (env var lookup), config (settings/config attribute), unknown"
        ),
    )
    base_source_name: str | None = Field(
        default=None,
        description="The env var NAME, config key, literal host, or (unknown) the expression",
    )
    base_source_default: str | None = Field(
        default=None, description="A literal default URL of the env lookup, when present"
    )
    repo: str = Field(
        default=PRIMARY_REPO,
        description="Workspace repo the file belongs to: 'primary' or '.repos/<name>'",
    )


class UnresolvedCallSiteResponse(BaseModel):
    file: str
    line: int
    kind: CallSiteKind
    method: str | None = None
    operation: str | None = None
    client: str
    symbol: str | None = None
    expression: str
    reason: str
    repo: str = Field(
        default=PRIMARY_REPO,
        description="Workspace repo the file belongs to: 'primary' or '.repos/<name>'",
    )


class CallSitesResponse(BaseModel):
    call_sites: list[CallSiteResponse]
    unresolved: list[UnresolvedCallSiteResponse]
    files_matched: int = Field(..., description="Files that matched language/path filters")
    files_scanned: int = Field(
        ..., description="Files extracted without error (excludes unparsed and not reached)"
    )
    files_unreadable: list[str] = Field(default_factory=list)
    files_unparsed: list[str] = Field(default_factory=list)
    files_unparsed_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="file -> why it was not parsed (grammar, extractor error, deadline)",
    )
    files_skipped_large: list[str] = Field(
        default_factory=list, description="Files over max_file_bytes, never read"
    )
    files_unparsed_count: int = Field(
        default=0, description="Exact number of unparsed files (the list is capped at 100)"
    )
    files_not_reached: list[str] = Field(
        default_factory=list,
        description=(
            "Loaded files the extraction never attempted because it stopped first "
            "(deadline, or the max_call_sites cap); capped at 100. Every loaded file is "
            "exactly one of: scanned (files_scanned), unparsed, not reached."
        ),
    )
    files_not_reached_count: int = Field(default=0, description="Exact number not reached")
    truncated: bool
    truncation_reason: str | None = Field(
        default=None,
        description=("Comma-separated: max_files, max_bytes, max_call_sites/max_routes, deadline"),
    )
    limits: ScanLimits | None = None
    languages: list[str]
    processing_time_seconds: float
    extractor_version: str
    repo: str = Field(
        default=PRIMARY_REPO,
        description="The workspace repo that was scanned; every file path is relative to it",
    )


# ---------------------------------------------------------------------------
# Provider routes (POST /api/analysis/contracts/routes)
# ---------------------------------------------------------------------------


class RoutesRequest(WorkspaceScanRequest):
    """Provider routes of ONE workspace (``container_name``) or of inline ``sources``."""

    languages: list[str] | None = Field(
        default=None,
        description="Subset of python, javascript, typescript, rust (default: all)",
    )
    max_routes: int = Field(default=5000, ge=1, le=50000)


class ProviderRouteResponse(BaseModel):
    file: str
    line: int
    method: str = Field(..., description="GET/POST/...; ALL when the method is not checked")
    path: str = Field(
        ...,
        description="Normalised exactly like call-site targets ({param}, no trailing slash)",
    )
    raw: str = Field(default="", description="The registered pattern as written (truncated)")
    framework: str = Field(
        ..., description="fastapi | flask | method-first | express | pathname | axum"
    )
    handler: str | None = Field(default=None, description="Handler name; null when inline")
    mount_prefix: str | None = Field(
        default=None, description="Prefix contributed by include_router/use()/route() mounts"
    )
    wildcard: bool = Field(
        default=False, description="Matches anything under `path` (`(.*)`, `/*`, startsWith)"
    )
    repo: str = Field(
        default=PRIMARY_REPO,
        description="Workspace repo the file belongs to: 'primary' or '.repos/<name>'",
    )


class UnresolvedRouteResponse(BaseModel):
    file: str
    line: int
    method: str | None = None
    expression: str
    reason: str
    repo: str = Field(
        default=PRIMARY_REPO,
        description="Workspace repo the file belongs to: 'primary' or '.repos/<name>'",
    )


class RoutesResponse(BaseModel):
    routes: list[ProviderRouteResponse]
    unresolved: list[UnresolvedRouteResponse]
    files_matched: int
    files_scanned: int
    files_unreadable: list[str] = Field(default_factory=list)
    files_unparsed: list[str] = Field(default_factory=list)
    files_unparsed_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="file -> why it was not parsed (grammar, extractor error, deadline)",
    )
    files_skipped_large: list[str] = Field(
        default_factory=list, description="Files over max_file_bytes, never read"
    )
    files_unparsed_count: int = Field(
        default=0, description="Exact number of unparsed files (the list is capped at 100)"
    )
    files_not_reached: list[str] = Field(
        default_factory=list,
        description=(
            "Loaded files the extraction never attempted because it stopped first "
            "(deadline, or the max_call_sites cap); capped at 100. Every loaded file is "
            "exactly one of: scanned (files_scanned), unparsed, not reached."
        ),
    )
    files_not_reached_count: int = Field(default=0, description="Exact number not reached")
    truncated: bool
    truncation_reason: str | None = Field(
        default=None,
        description=("Comma-separated: max_files, max_bytes, max_call_sites/max_routes, deadline"),
    )
    limits: ScanLimits | None = None
    languages: list[str]
    processing_time_seconds: float
    extractor_version: str
    repo: str = Field(
        default=PRIMARY_REPO,
        description="The workspace repo that was scanned; every file path is relative to it",
    )
