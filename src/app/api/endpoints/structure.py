"""Computed structural facts of workspace repos (/api/analysis/structure/*).

Facts only, no recommendations, no LLM. Every route reads the tenant workspace
through coding-containers (``materialize_from_container``; ``.repos/`` excluded
unless a repo is selected or ``all_repos`` is set), then parses the tenant
source in an isolated worker process (``contract_analysis.isolation``).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.responses import validation_error
from app.api.utils.workspace_files import (
    InvalidWorkspacePathError,
    invalid_workspace_path_http,
    normalize_workspace_path,
    workspace_root,
)
from auth.tenant import require_tenant
from clients.coding_containers import get_coding_containers_client
from models.structure_models import (
    MAX_FILE_BYTES,
    MAX_REQUEST_BYTES,
    ApiSurfaceResponse,
    EntrypointsRequest,
    EntrypointsResponse,
    LanguageScopedRequest,
    ModuleGraphResponse,
    ServiceGraphRequest,
    ServiceGraphResponse,
    TablesRequest,
    TablesResponse,
)
from processing.architecture.container_fs import (
    DEFAULT_EXCLUDE_GLOBS,
    ScopeMatchesNoFilesError,
    materialize_from_container,
)
from processing.contract_analysis import isolation
from processing.structure import jobs
from processing.structure.service_graph import split_by_repo
from processing.workspace_repos import PRIMARY_REPO, repo_root

logger = logging.getLogger(__name__)
router = APIRouter()

_SOURCE_PATTERNS = {
    # pyproject.toml / setup.cfg declare the package roots (src layout): module names.
    "python": ("*.py", "pyproject.toml", "setup.cfg"),
    "rust": ("*.rs", "Cargo.toml"),
}
_TABLE_PATTERNS = ("*.py", "*.rs", "*.sql")
_ENTRYPOINT_PATTERNS = ("*.py", "*.rs", "pyproject.toml", "Cargo.toml")
_SERVICE_PATTERNS = (
    "*.py",
    "*.rs",
    "*.ts",
    "*.js",
    "*.tsx",
    "*.mjs",
    "*.yaml",  # docker-compose services, k8s Service/Deployment/StatefulSet/DaemonSet names
    "*.yml",
)
_EXCLUDES = (*DEFAULT_EXCLUDE_GLOBS, "*/__pycache__/*", "*/.svelte-kit/*", "*/.next/*")


@dataclass
class _Read:
    files: dict[str, str]
    skipped_large: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


async def read_workspace(
    container_name: str,
    tenant_id: str,
    patterns: tuple[str, ...],
    *,
    repo: str | None = None,
    all_repos: bool = False,
    scope: str | None = None,
) -> _Read:
    """Files matching *patterns* of one repo (or every repo), root-relative, capped."""
    from app.api.utils.container_validation import validate_and_get_container_for_tenant

    container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
    ws_root = workspace_root(container_info)
    base = ws_root if all_repos else repo_root(ws_root, repo)
    scope_paths = None
    if scope:
        try:
            scope_paths = [normalize_workspace_path(scope)]
        except InvalidWorkspacePathError as exc:
            raise invalid_workspace_path_http(exc) from exc
    try:
        tmp = await materialize_from_container(
            get_coding_containers_client(),
            container_name,
            base,
            patterns,
            _EXCLUDES,
            tmp_prefix="structure-",
            tenant_id=tenant_id,
            scope_paths=scope_paths,
            exclude_repos=not all_repos,
        )
    except ScopeMatchesNoFilesError as exc:
        raise validation_error(str(exc), field="root") from exc
    out = _Read(files={}, reasons=[])
    if tmp is None:
        return out

    def load() -> None:  # disk reads: run off the event loop
        total = 0
        try:
            for path in sorted(p for p in Path(tmp).rglob("*") if p.is_file()):
                rel = path.relative_to(tmp).as_posix()
                size = path.stat().st_size
                if size > MAX_FILE_BYTES:
                    out.skipped_large.append(rel)
                    continue
                if total + size > MAX_REQUEST_BYTES:
                    if "max_bytes" not in out.reasons:
                        out.reasons.append("max_bytes")
                    break
                total += size
                out.files[rel] = path.read_text(encoding="utf-8", errors="replace")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    await asyncio.to_thread(load)
    return out


def _stats(read: _Read) -> dict:
    return {
        "files_read": len(read.files),
        "files_skipped_large": read.skipped_large[:100],
        "truncated": bool(read.reasons),
        "truncation_reason": ",".join(read.reasons) or None,
    }


async def _isolated(fn, *args, tenant_id: str, narrow: tuple[str, ...]):
    """Run *fn* in the isolated extractor. *narrow* names the request fields that
    shrink the input on THIS route (``root``/``repo``; empty when none does): a
    failed extraction is a 422 on the first of them, or on ``container_name``."""
    limits = isolation.limits_from_settings()
    try:
        async with isolation.admit(limits):
            return await isolation.run(fn, *args, limits=limits, tenant=tenant_id)
    except isolation.IsolationFailedError as exc:
        if exc.reason == "busy":
            from app.api.endpoints.contract_analysis import _busy_http

            raise _busy_http(exc, limits) from exc
        hint = (
            f" Narrow {' / '.join(f'`{f}`' for f in narrow)} and retry."
            if narrow
            else " The workspace is too large to extract in one request."
        )
        raise validation_error(
            f"Extraction failed ({exc.reason}): {exc.detail}.{hint}",
            field=narrow[0] if narrow else "container_name",
            details={"reason": exc.reason, "deadline_seconds": limits.deadline_s},
        ) from exc


def _root_prefix(root: str | None) -> str:
    if not root:
        return ""
    try:
        return normalize_workspace_path(root)
    except InvalidWorkspacePathError as exc:
        raise invalid_workspace_path_http(exc) from exc


@router.post(
    "/module-graph",
    response_model=ModuleGraphResponse,
    response_model_by_alias=True,
    operation_id="structure_module_graph",
    summary="Internal module graph with SCCs and leaves-first porting layers (Python or Rust)",
    description=(
        "Nodes = internal modules (Python: every .py file; Rust: every file-backed module "
        "reachable from a crate root via `mod`, honouring #[path]) with LOC and public-symbol "
        "count. Edges = internal dependencies (`from`, `to`, `kind`: import | from_import | "
        "dynamic | type_only | use | path). `sccs` = cycles; `order` = layers of the "
        "SCC-condensed DAG, LEAVES FIRST (layer k depends only on layers < k). "
        "`slices` = porting units: one per SCC (a cycle is ONE slice), topologically "
        "ordered leaves first, with `depends_on` slice ids and summed `loc`. "
        "`unresolved` lists dynamic imports with computed names, missing `mod` files, .rs "
        "files no crate root reaches and `use` paths that resolve to nothing known."
    ),
    responses={404: {"description": "Workspace not found"}, 422: {"description": "Bad root"}},
)
async def module_graph(
    body: LanguageScopedRequest, tenant_id: str = Depends(require_tenant)
) -> ModuleGraphResponse:
    lang = body.language.value
    root = _root_prefix(body.root)
    read = await read_workspace(
        body.container_name, tenant_id, _SOURCE_PATTERNS[lang], repo=body.repo, scope=root or None
    )
    graph = await _isolated(
        jobs.module_graph_job, read.files, lang, root, tenant_id=tenant_id, narrow=("root", "repo")
    )
    return ModuleGraphResponse.model_validate(
        {**asdict(graph), **_stats(read), "repo": body.repo or PRIMARY_REPO}
    )


@router.post(
    "/api-surface",
    response_model=ApiSurfaceResponse,
    operation_id="structure_api_surface",
    summary="Public API surface with signatures (Python or Rust)",
    description=(
        "Every public symbol {id, module_path, name, kind, signature {params [{name, type, "
        "default}], returns, generics, is_async}, visibility, file, line}. Python respects "
        "`__all__` (when literal) and leading `_`. Rust lists `pub` items (visibility as "
        "written), inherent pub methods, trait methods and every trait-impl method "
        "(`impl_fn`), struct fields / enum variants as `members`. Macro-generated items "
        "(macro_rules!, item-position macro calls, #[derive], attribute macros like #[get]) are "
        "listed under `unresolved` with the macro, never silently missing."
    ),
    responses={404: {"description": "Workspace not found"}, 422: {"description": "Bad root"}},
)
async def api_surface(
    body: LanguageScopedRequest, tenant_id: str = Depends(require_tenant)
) -> ApiSurfaceResponse:
    lang = body.language.value
    root = _root_prefix(body.root)
    read = await read_workspace(
        body.container_name, tenant_id, _SOURCE_PATTERNS[lang], repo=body.repo, scope=root or None
    )
    surface = await _isolated(
        jobs.api_surface_job, read.files, lang, root, tenant_id=tenant_id, narrow=("root", "repo")
    )
    return ApiSurfaceResponse.model_validate(
        {**asdict(surface), **_stats(read), "repo": body.repo or PRIMARY_REPO}
    )


def _by_repo(body: TablesRequest | EntrypointsRequest, read: _Read) -> dict[str, dict[str, str]]:
    if body.all_repos:
        return split_by_repo(read.files)
    return {body.repo or PRIMARY_REPO: read.files}


@router.post(
    "/tables",
    response_model=TablesResponse,
    operation_id="structure_db_tables",
    summary="Database table inventory: defined / altered / read / written, per repo",
    description=(
        "Python: SQLAlchemy `__tablename__` and `Table(...)`, ORM use of mapped classes "
        "(select/query/get read; insert/update/delete and session.add/add_all/merge write), "
        "Alembic `op.*` migrations, raw SQL string literals. Rust: SQL string literals "
        "(tagged `sqlx` inside query!/query_as!/query_scalar! and sqlx::query()). Every *.sql "
        "file (`migration` under a migrations dir). Each reference carries repo, file, line "
        "and `via`. Interpolated table names are `unresolved`."
    ),
    responses={404: {"description": "Workspace not found"}},
)
async def db_tables(
    body: TablesRequest, tenant_id: str = Depends(require_tenant)
) -> TablesResponse:
    read = await read_workspace(
        body.container_name, tenant_id, _TABLE_PATTERNS, repo=body.repo, all_repos=body.all_repos
    )
    per_repo = _by_repo(body, read)
    results = await _isolated(jobs.tables_job, per_repo, tenant_id=tenant_id, narrow=("repo",))
    merged: dict[str, dict] = {}
    unresolved: list[dict] = []
    unparsed: dict[str, str] = {}
    for repo, inv in sorted(results.items()):
        for table in inv.tables:
            entry = merged.setdefault(
                table.table,
                {
                    "table": table.table,
                    "schemas": [],
                    "defined_in": [],
                    "altered_in": [],
                    "read_by": [],
                    "written_by": [],
                },
            )
            entry["schemas"] = sorted({*entry["schemas"], *table.schemas})
            for role in ("defined_in", "altered_in", "read_by", "written_by"):
                entry[role].extend({**asdict(ref), "repo": repo} for ref in getattr(table, role))
        unresolved.extend({**asdict(u), "repo": repo} for u in inv.unresolved)
        unparsed.update({_label(repo, f): r for f, r in inv.files_unparsed.items()})
    for entry in merged.values():
        entry["repos"] = sorted(
            {
                ref["repo"]
                for role in ("defined_in", "altered_in", "read_by", "written_by")
                for ref in entry[role]
            }
        )
    return TablesResponse.model_validate(
        {
            "repos": sorted(per_repo),
            "tables": [merged[k] for k in sorted(merged)],
            "unresolved": unresolved,
            "files_unparsed": unparsed,
            **_stats(read),
        }
    )


def _label(repo: str, path: str) -> str:
    return path if repo == PRIMARY_REPO else f"{repo}/{path}"


@router.post(
    "/entrypoints",
    response_model=EntrypointsResponse,
    operation_id="structure_entrypoints",
    summary="Non-HTTP entrypoints: workers, consumers, schedulers, CLIs, websockets, SSE",
    description=(
        "Background workers and everything that runs without an HTTP route calling it: "
        "FastAPI lifespan / startup hooks and the tasks they spawn, celery/rq/arq/dramatiq "
        "tasks, celery beat, APScheduler jobs, redis stream / pub-sub / list consumers, "
        "kafka/nats/amqp consumers, websocket and SSE endpoints, `if __name__ == "
        "'__main__'`, [project.scripts] console scripts; Rust #[tokio::main], bin targets, "
        "tokio::spawn / thread::spawn, axum WebSocketUpgrade handlers, Sse, clap CLIs, "
        "tokio-cron-scheduler jobs. Each entry: repo, kind, name, file, line, detail."
    ),
    responses={404: {"description": "Workspace not found"}},
)
async def entrypoints(
    body: EntrypointsRequest, tenant_id: str = Depends(require_tenant)
) -> EntrypointsResponse:
    read = await read_workspace(
        body.container_name,
        tenant_id,
        _ENTRYPOINT_PATTERNS,
        repo=body.repo,
        all_repos=body.all_repos,
    )
    per_repo = _by_repo(body, read)
    results = await _isolated(jobs.entrypoints_job, per_repo, tenant_id=tenant_id, narrow=("repo",))
    items: list[dict] = []
    unparsed: dict[str, str] = {}
    for repo, inv in sorted(results.items()):
        items.extend({**asdict(e), "repo": repo} for e in inv.entrypoints)
        unparsed.update({_label(repo, f): r for f, r in inv.files_unparsed.items()})
    return EntrypointsResponse.model_validate(
        {
            "repos": sorted(per_repo),
            "entrypoints": items,
            "files_unparsed": unparsed,
            **_stats(read),
        }
    )


@router.post(
    "/service-graph",
    response_model=ServiceGraphResponse,
    operation_id="structure_service_graph",
    summary="Cross-repo service call graph over ALL workspace repos",
    description=(
        "Runs the call-site and route extractors on the primary repo and every "
        "'.repos/<name>' checkout, then matches each outbound HTTP call to a route of "
        "another repo. The target repo comes from `hints` ({ENV_VAR | config key | host: "
        "repo}), else a docker-compose/k8s service name defined in exactly one repo, else the "
        "only other repo whose routes match. Edges: {from_repo, from_file, from_line, "
        "from_symbol, method, from_target, from_base, from_host, from_base_source_kind, "
        "from_base_source_name, to_repo, to_route, match: exact|template|prefix|unresolved, "
        "resolved_by, candidates}. "
        "Every HTTP call site is an edge; unmatched ones have match `unresolved` and a reason. "
        "`message_edges` = message-based coupling (Python + Rust redis crate): Redis pub/sub "
        "(publish -> subscribe/psubscribe), Redis streams (xadd -> xread/xreadgroup) and task "
        "queues (celery send_task/.delay -> @task, rq enqueue -> the function, arq "
        "enqueue_job -> WorkerSettings.functions): {from_repo, from_file, from_line, "
        "from_symbol, from_channel, from_via, kind: pubsub|stream|task, to_repo, to_file, "
        "to_line, to_symbol, to_channel, to_via, match: exact|pattern|unresolved, reason}. "
        "Both edge kinds name the caller/publisher side from_* and the callee/consumer side "
        "to_*. Literal names match exactly, f-string/format! names with a constant prefix and "
        "psubscribe globs match as patterns, env-var and computed names are unresolved (with "
        "the name or expression). `unmatched_publishers` / `unmatched_consumers` list the rest."
    ),
    responses={404: {"description": "Workspace not found"}, 422: {"description": "Bad hint"}},
)
async def service_graph(
    body: ServiceGraphRequest, tenant_id: str = Depends(require_tenant)
) -> ServiceGraphResponse:
    from processing.structure.service_graph import canonical_hints

    try:
        hints = canonical_hints(body.hints)
    except ValueError as exc:
        raise validation_error(f"hints: {exc}", field="hints") from exc
    read = await read_workspace(body.container_name, tenant_id, _SERVICE_PATTERNS, all_repos=True)
    per_repo = split_by_repo(read.files)
    graph = await _isolated(
        jobs.service_graph_job, per_repo, hints, body.include_tests, tenant_id=tenant_id, narrow=()
    )
    return ServiceGraphResponse.model_validate({**asdict(graph), **_stats(read)})
