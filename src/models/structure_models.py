"""Request/response models for /api/analysis/structure/* (computed facts about code)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AliasChoices, BaseModel, Field

# Per-file and per-request read caps: the SAME values as the contracts routes
# (one definition, re-exported here for the structure routes).
from models.contract_analysis_models import MAX_FILE_BYTES as MAX_FILE_BYTES
from models.contract_analysis_models import MAX_REQUEST_BYTES as MAX_REQUEST_BYTES
from processing.workspace_repos import RepoSelector


class StructureLanguage(StrEnum):
    PYTHON = "python"
    RUST = "rust"


class _Workspace(RepoSelector):
    container_name: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("container_name", "containerName"),
        description="Tenant workspace to read through coding-containers",
    )


class _AllRepos(_Workspace):
    all_repos: bool = Field(
        default=False,
        description=(
            "Every repo of the workspace (the primary repo and each '.repos/<name>' checkout), "
            "each reported under its own `repo`; `repo` is then ignored"
        ),
    )


class LanguageScopedRequest(_Workspace):
    language: StructureLanguage = Field(..., description="python or rust")
    root: str | None = Field(
        default=None,
        description=(
            "Repo-relative directory to analyse (e.g. 'src'). Python module names are "
            "computed relative to it; default: the whole repository"
        ),
    )


class TablesRequest(_AllRepos):
    pass


class EntrypointsRequest(_AllRepos):
    pass


class ServiceGraphRequest(_Workspace):
    hints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional {ENV_VAR | config key | host[:port] : repo}. Decides which repo an "
            "outbound call targets when its base URL comes from that env var/config key or "
            "host. Repo as 'primary', '.repos/<name>' or '<name>'"
        ),
    )
    include_tests: bool = Field(default=False, description="Also scan test/fixture files")


# ---------------------------------------------------------------------------
# Shared response parts
# ---------------------------------------------------------------------------


class ParamOut(BaseModel):
    name: str
    type: str | None = None
    default: str | None = None


class SignatureOut(BaseModel):
    params: list[ParamOut] = []
    returns: str | None = None
    generics: str | None = None
    is_async: bool = False


class MemberOut(BaseModel):
    name: str
    type: str | None = None
    visibility: str | None = None


class UnresolvedOut(BaseModel):
    file: str
    line: int
    reason: str
    name: str | None = None
    macro: str | None = Field(
        default=None, description="The macro responsible, when the item is macro-generated"
    )
    expression: str | None = None
    candidates: list[str] = Field(
        default=[], description="Equally plausible targets when the item is ambiguous"
    )


class ReadStats(BaseModel):
    files_read: int
    files_skipped_large: list[str] = []
    truncated: bool = False
    truncation_reason: str | None = None


# ---------------------------------------------------------------------------
# module-graph
# ---------------------------------------------------------------------------


class ModuleNodeOut(BaseModel):
    module: str
    file: str
    loc: int
    public_symbols: int


class ModuleEdgeOut(BaseModel):
    source: str = Field(..., serialization_alias="from", description="Importing module")
    target: str = Field(..., serialization_alias="to", description="Imported module")
    kind: str = Field(..., description="import | from_import | dynamic | type_only | use | path")
    file: str
    line: int


class SliceOut(BaseModel):
    id: str = Field(..., description="Slice id, s0..sN in porting order")
    layer: int = Field(..., description="Layer in `order` (0 = depends on no internal module)")
    modules: list[str] = Field(..., description="Modules ported together (sorted)")
    cyclic: bool = Field(..., description="True when the modules form an import cycle")
    depends_on: list[str] = Field(..., description="Ids of the slices this slice depends on")
    loc: int = Field(..., description="Summed non-blank, non-comment LOC of the modules")


class ModuleGraphResponse(ReadStats):
    repo: str
    language: str
    nodes: list[ModuleNodeOut]
    edges: list[ModuleEdgeOut]
    sccs: list[list[str]]
    order: list[list[str]] = Field(
        ..., description="Layers of the SCC-condensed DAG, leaves first; an SCC shares a layer"
    )
    slices: list[SliceOut] = Field(
        default=[],
        description=(
            "Porting slices: one slice per SCC (a cycle is ONE slice), topologically "
            "ordered, leaves first; depends_on = slice ids"
        ),
    )
    unresolved: list[UnresolvedOut]
    files_unparsed: dict[str, str] = {}


# ---------------------------------------------------------------------------
# api-surface
# ---------------------------------------------------------------------------


class ApiSymbolOut(BaseModel):
    id: str
    module_path: str
    name: str
    kind: str
    visibility: str
    file: str
    line: int
    signature: SignatureOut | None = None
    members: list[MemberOut] = []
    attributes: list[str] = []
    owner: str | None = None
    trait: str | None = None


class ApiSurfaceResponse(ReadStats):
    repo: str
    language: str
    symbols: list[ApiSymbolOut]
    unresolved: list[UnresolvedOut]
    files_unparsed: dict[str, str] = {}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


class TableRefOut(BaseModel):
    repo: str
    file: str
    line: int
    via: str
    statement: str | None = None
    schema_: str | None = Field(
        default=None,
        alias="schema",
        description="Schema the reference names ('billing' for billing.invoices); null if unqualified",
    )


class TableOut(BaseModel):
    table: str = Field(..., description="Table name, lowercased, without its schema")
    schemas: list[str] = Field(default=[], description="Every schema the table is referenced under")
    defined_in: list[TableRefOut] = []
    altered_in: list[TableRefOut] = []
    read_by: list[TableRefOut] = []
    written_by: list[TableRefOut] = []
    repos: list[str] = Field(..., description="Every repo that references the table")


class UnresolvedTableOut(BaseModel):
    repo: str
    file: str
    line: int
    reason: str
    expression: str


class TablesResponse(ReadStats):
    repos: list[str]
    tables: list[TableOut]
    unresolved: list[UnresolvedTableOut]
    files_unparsed: dict[str, str] = {}


# ---------------------------------------------------------------------------
# entrypoints
# ---------------------------------------------------------------------------


class EntrypointOut(BaseModel):
    repo: str
    kind: str
    name: str | None = None
    file: str
    line: int
    detail: str | None = None


class EntrypointsResponse(ReadStats):
    repos: list[str]
    entrypoints: list[EntrypointOut]
    files_unparsed: dict[str, str] = {}


# ---------------------------------------------------------------------------
# service-graph
# ---------------------------------------------------------------------------


class RouteRefOut(BaseModel):
    method: str
    path: str
    file: str
    line: int
    handler: str | None = None


class ServiceEdgeOut(BaseModel):
    from_repo: str
    from_file: str
    from_line: int
    from_symbol: str | None = None
    method: str | None = None
    from_target: str = Field(..., description="The caller's request path (joinable form)")
    match: str = Field(..., description="exact | template | prefix | unresolved")
    to_repo: str | None = None
    to_route: RouteRefOut | None = None
    resolved_by: str | None = Field(default=None, description="hint | service_name | route_only")
    from_base: str | None = Field(
        default=None, description="Base-URL expression stripped off the caller's target"
    )
    from_host: str | None = Field(default=None, description="Literal host of an absolute URL")
    from_base_source_kind: str | None = Field(
        default=None, description="literal | env | config | unknown"
    )
    from_base_source_name: str | None = None
    candidates: list[str] = []
    reason: str | None = None


class RepoSummaryOut(BaseModel):
    repo: str
    routes: int
    http_call_sites: int
    unresolved_call_sites: int
    files_unparsed: dict[str, str] = {}
    message_publishers: int = 0
    message_consumers: int = 0


class MessageEdgeOut(BaseModel):
    from_repo: str
    from_file: str
    from_line: int
    from_symbol: str | None = None
    kind: str = Field(..., description="pubsub | stream | task")
    from_channel: str = Field(
        ...,
        description=(
            "Publisher's channel / stream key / task name: the literal, a 'prefix*' pattern, "
            "or the env var name (or expression) when unresolved"
        ),
    )
    match: str = Field(..., description="exact | pattern | unresolved")
    from_via: str = Field(..., description="Publishing API, e.g. redis.publish, celery.send_task")
    to_repo: str | None = None
    to_file: str | None = None
    to_line: int | None = None
    to_symbol: str | None = None
    to_channel: str | None = Field(default=None, description="Consumer's channel / pattern")
    to_via: str | None = None
    reason: str | None = Field(default=None, description="Why the channel is unresolved")


class MessageEndpointOut(BaseModel):
    repo: str
    file: str
    line: int
    role: str = Field(..., description="publisher | consumer")
    kind: str = Field(..., description="pubsub | stream | task")
    channel: str
    channel_kind: str = Field(..., description="exact | pattern | unresolved")
    via: str
    symbol: str | None = None
    reason: str | None = None


class UnresolvedCallOut(BaseModel):
    repo: str
    file: str
    line: int
    method: str | None = None
    expression: str
    reason: str


class ServiceGraphResponse(ReadStats):
    edges: list[ServiceEdgeOut]
    repos: list[RepoSummaryOut]
    unresolved_call_sites: list[UnresolvedCallOut]
    service_names: dict[str, str] = Field(
        default_factory=dict, description="compose/k8s service name -> the repo defining it"
    )
    service_name_conflicts: dict[str, list[str]] = {}
    message_edges: list[MessageEdgeOut] = Field(
        default_factory=list,
        description=(
            "Publisher -> consumer edges of Redis pub/sub, Redis streams and celery/rq/arq "
            "tasks, matched by channel name; a publisher with an unresolved channel is one "
            "edge with match 'unresolved' and no consumer"
        ),
    )
    unmatched_publishers: list[MessageEndpointOut] = Field(
        default_factory=list, description="Resolved publishers no consumer matched"
    )
    unmatched_consumers: list[MessageEndpointOut] = Field(
        default_factory=list, description="Consumers no publisher matched"
    )
