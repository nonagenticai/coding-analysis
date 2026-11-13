"""API endpoints for static contract analysis."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.responses import bad_request, service_unavailable, validation_error
from app.api.utils.workspace_files import (
    InvalidWorkspacePathError,
    WorkspaceUnresolvableError,
    invalid_workspace_path_http,
    normalize_workspace_path,
    workspace_root,
    workspace_unresolvable_http,
)
from auth.tenant import require_tenant
from clients.coding_containers import get_coding_containers_client
from models.contract_analysis_models import (
    MAX_FILE_BYTES,
    MAX_REQUEST_BYTES,
    CallSiteResponse,
    CallSitesRequest,
    CallSitesResponse,
    ContractPair,
    ContractResultResponse,
    ProviderRouteResponse,
    RoutesRequest,
    RoutesResponse,
    ScanCapabilities,
    ScanLimits,
    ScanRequest,
    ScanResponse,
    SourceInput,
    UnresolvedCallSiteResponse,
    UnresolvedRouteResponse,
    ViolationResponse,
    ViolationSeverity,
    WorkspaceScanRequest,
)
from processing.contract_analysis import call_sites as cs
from processing.contract_analysis import isolation, jobs
from processing.contract_analysis import provider_routes as pr
from processing.contract_analysis.scanner import ContractResult
from processing.workspace_repos import (
    repo_of_path,
    repo_root,
    repos_exclude_globs,
    scope_enters_repos,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_FILE_SIZE = 500_000  # bytes


_ALLOWED_BASE_DIRS = (
    "/app/provisioned_containers/",
    "/repo/",
    "/app/data/repo/",
    "/app/analysis_workspace/",
    "/app/shared/volumes/",
)


def _resolve_source(inp: SourceInput) -> str:
    """Resolve a SourceInput to its source string."""
    if inp.source is not None:
        return inp.source
    if inp.file_path is not None:
        p = Path(inp.file_path)
        if not p.is_absolute():
            raise bad_request(f"file_path must be absolute: {inp.file_path}")
        resolved = p.resolve()
        if not any(str(resolved).startswith(base) for base in _ALLOWED_BASE_DIRS):
            raise bad_request(f"file_path must be within an allowed directory: {inp.file_path}")
        if not p.exists():
            raise bad_request(f"File not found: {inp.file_path}")
        if p.stat().st_size > _MAX_FILE_SIZE:
            raise bad_request(f"File too large (>{_MAX_FILE_SIZE} bytes): {inp.file_path}")
        return p.read_text()
    label = inp.label or "input"
    raise bad_request(f"Either source or file_path is required for {label}")


def _contract_result_to_response(result: ContractResult) -> ContractResultResponse:
    """Convert internal ContractResult to API response model."""
    return ContractResultResponse(
        name=result.name,
        passed=result.passed,
        violations=[
            ViolationResponse(
                severity=ViolationSeverity(v.severity.value),
                check=v.check,
                message=v.message,
                client_location=v.client_location,
                service_location=v.service_location,
                closest_match=v.closest_match,
                root_cause=v.root_cause,
                suggestion=v.suggestion,
            )
            for v in result.violations
        ],
        stats=result.stats,
        parse_error=result.parse_error,
    )


_TS_ROUTE_LANGUAGES = {"typescript", "ts", "javascript", "js", "tsx", "jsx"}
_TS_ROUTE_SUFFIXES = (".ts", ".mts", ".cts", ".tsx", ".js", ".mjs", ".cjs", ".jsx")


def _is_ts_route(inp: SourceInput) -> bool:
    if inp.language:
        return inp.language.strip().lower() in _TS_ROUTE_LANGUAGES
    name = (inp.file_path or inp.label or "").lower()
    return name.endswith(_TS_ROUTE_SUFFIXES)


def _route_name(inp: SourceInput, index: int) -> str:
    """A repo-relative-looking name, so relative imports between route files resolve."""
    raw = inp.file_path or inp.label or f"route_{index}.ts"
    for base in _ALLOWED_BASE_DIRS:
        if raw.startswith(base):
            return raw[len(base) :]
    return raw.lstrip("/")


def _busy_http(exc: isolation.IsolationFailedError, limits: isolation.Limits) -> HTTPException:
    """503 + Retry-After: the extraction workers are saturated; nothing was extracted."""
    http = service_unavailable(f"Contract extraction is busy ({exc.detail}); retry later.")
    http.headers = {"Retry-After": str(limits.retry_after_s())}
    return http


@asynccontextmanager
async def _admitted(limits: isolation.Limits) -> AsyncIterator[None]:
    """Admission for one extraction request; any ``busy`` inside it becomes a 503."""
    try:
        async with isolation.admit(limits):
            yield
    except isolation.IsolationFailedError as exc:
        if exc.reason != "busy":
            raise
        raise _busy_http(exc, limits) from exc


async def _scan_one(
    pair: ContractPair, config, limits: isolation.Limits | None = None, tenant_id: str | None = None
) -> ContractResult:
    """Scan a single contract pair.

    TypeScript route parsing AND every check run in an isolated worker process
    (tenant source, deadline-bounded); a pair that misses the deadline fails with a
    parse_error instead of stalling the service. ``busy`` is raised (the caller
    answers 503): it says nothing about the pair.
    """
    try:
        client_source = _resolve_source(pair.client)
        schema_sources = [_resolve_source(s) for s in pair.schemas]
        route_sources = [_resolve_source(r) for r in pair.routes if not _is_ts_route(r)]
        ts_route_sources = {
            _route_name(r, i): _resolve_source(r)
            for i, r in enumerate(pair.routes)
            if _is_ts_route(r)
        }
    except Exception as e:
        return ContractResult(name=pair.name, passed=False, parse_error=str(e))

    checks = jobs.PairChecks(
        syntax=config.check_syntax,
        routes=config.check_routes,
        methods=config.check_methods,
        request_fields=config.check_request_fields,
        response_fields=config.check_response_fields,
        types=config.check_types,
        query_params=config.check_query_params,
        status_codes=config.check_status_codes,
    )
    try:
        return await isolation.run(
            jobs.scan_pair_job,
            pair.name,
            client_source,
            schema_sources,
            route_sources,
            ts_route_sources,
            checks,
            limits=limits,
            tenant=tenant_id,
        )
    except isolation.IsolationFailedError as exc:
        if exc.reason == "busy":
            raise
        return ContractResult(
            name=pair.name,
            passed=False,
            parse_error=f"Contract scan failed ({exc.reason}): {exc.detail}",
        )


@router.post("/scan", response_model=ScanResponse, operation_id="scan_contracts")
async def scan_contracts(
    body: ScanRequest,
    tenant_id: str = Depends(require_tenant),
) -> ScanResponse:
    """Scan contract pairs for API violations between clients and services."""
    start = time.monotonic()

    if not body.contracts:
        raise bad_request("At least one contract pair is required in the 'contracts' array")

    limits = isolation.limits_from_settings()
    results: list[ContractResult] = []
    async with _admitted(limits):
        for pair in body.contracts:
            results.append(await _scan_one(pair, body.config, limits, tenant_id))

    contract_responses = [_contract_result_to_response(r) for r in results]
    passed_count = sum(1 for r in results if r.passed)
    total_violations = sum(len(r.violations) for r in results)
    total_warnings = sum(sum(1 for v in r.violations if v.severity == "WARN") for r in results)

    overall_passed = all(r.passed for r in results)
    if body.config.fail_on_warnings and total_warnings > 0:
        overall_passed = False

    elapsed = time.monotonic() - start
    logger.info(
        "Contract scan complete: %d contracts, %d violations in %.2fs (tenant=%s)",
        len(results),
        total_violations,
        elapsed,
        tenant_id,
    )

    return ScanResponse(
        passed=overall_passed,
        total_contracts=len(results),
        passed_contracts=passed_count,
        failed_contracts=len(results) - passed_count,
        total_violations=total_violations,
        total_warnings=total_warnings,
        contracts=contract_responses,
        processing_time_seconds=round(elapsed, 3),
    )


@router.get(
    "/capabilities", response_model=ScanCapabilities, operation_id="get_contract_capabilities"
)
async def scan_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> ScanCapabilities:
    """Return available contract scanning capabilities."""
    return ScanCapabilities(
        checks=[
            "route_path",
            "http_method",
            "request_field",
            "response_field",
            "query_param",
            "status_code",
            "python3_syntax",
            "type_check",
        ],
        supports_file_paths=True,
    )


# ---------------------------------------------------------------------------
# POST /call-sites -- the consumer side of a contract
# ---------------------------------------------------------------------------

#: Never worth reading: build output, vendored trees, VCS metadata.
_CALL_SITE_EXCLUDES = (
    "*/node_modules/*",
    "*/.git/*",
    "*/.venv/*",
    "*/venv/*",
    "*/dist/*",
    "*/build/*",
    "*/vendor/*",
    "*/target/*",
    "*/.svelte-kit/*",
    "*/.next/*",
    "*/__pycache__/*",
    "*/coverage/*",
    "*.min.js",
)
_READ_CONCURRENCY = 8
_MAX_LISTED_PATHS = 100


def _selected_languages(
    requested: list[str] | None, supported: tuple[str, ...] = cs.SUPPORTED_LANGUAGES
) -> list[str]:
    if not requested:
        return list(supported)
    langs = []
    for lang in requested:
        key = {"py": "python", "js": "javascript", "ts": "typescript"}.get(
            lang.strip().lower(), lang.strip().lower()
        )
        if key not in supported:
            raise bad_request(f"unsupported language: {lang}; supported: {', '.join(supported)}")
        if key not in langs:
            langs.append(key)
    return langs


def _wanted(path: str, langs: list[str], prefixes: list[str], include_tests: bool) -> bool:
    lang = cs.language_for(path)
    if lang is None or lang not in langs:
        return False
    if prefixes and not any(path == p or path.startswith(p + "/") for p in prefixes):
        return False
    return include_tests or not cs.is_test_path(path)


#: ``find -size`` units are KiB rounded UP: ``-size -1025k`` lists files of at most
#: 1024 KiB, ``-size +1024k`` exactly the rest. Both partition on MAX_FILE_BYTES.
_SIZE_KIB = MAX_FILE_BYTES // 1024


def _find_argv(
    roots: list[str],
    langs: list[str],
    *,
    large: bool = False,
    extra_excludes: tuple[str, ...] = (),
) -> list[str]:
    """List candidate files: at most MAX_FILE_BYTES each, or (``large``) only bigger ones."""
    argv = ["find", *roots, "-type", "f", "("]
    exts = [ext for lang in langs for ext in cs.LANGUAGE_EXTENSIONS[lang]]
    for idx, ext in enumerate(exts):
        if idx:
            argv.append("-o")
        argv += ["-name", f"*{ext}"]
    argv.append(")")
    for excl in (*_CALL_SITE_EXCLUDES, *extra_excludes):
        argv += ["-not", "-path", excl]
    argv += ["-size", f"+{_SIZE_KIB}k" if large else f"-{_SIZE_KIB + 1}k"]
    return argv


@dataclass
class _Loaded:
    """The files one request will extract from, and what was left out and why."""

    files: list[tuple[str, str]]
    matched: int
    unreadable: list[str] = field(default_factory=list)
    skipped_large: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # max_files, max_bytes


def _relative(listed: str, root: str) -> str:
    return listed[len(root) + 1 :] if listed.startswith(root + "/") else listed.lstrip("/")


async def _workspace_sources(
    body: WorkspaceScanRequest, tenant_id: str, langs: list[str], prefixes: list[str]
) -> _Loaded:
    """Read matching files of the tenant's workspace through coding-containers.

    Files are listed with a whitelisted ``find`` scoped to the requested prefixes
    (files over MAX_FILE_BYTES excluded there and listed separately as skipped),
    filtered, sorted and capped at ``max_files`` BEFORE any content is read. Reads
    then stop once MAX_REQUEST_BYTES have been read (``max_bytes``).
    """
    from app.api.utils.container_validation import validate_and_get_container_for_tenant

    container_name = body.container_name or ""
    container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
    # The selected repo's root: the workspace root (primary) or .repos/<name>.
    root = repo_root(workspace_root(container_info), body.repo)
    roots = [f"{root}/{p}" for p in prefixes] or [root]
    # The source checkouts under .repos/ are not part of the selected repo,
    # unless a path prefix names them explicitly.
    excludes = () if scope_enters_repos(prefixes) else repos_exclude_globs(root)
    client = get_coding_containers_client()
    result, large = await asyncio.gather(
        client.execute(
            container_name,
            _find_argv(roots, langs, extra_excludes=excludes),
            workdir=root,
            timeout=60,
            tenant_id=tenant_id,
        ),
        client.execute(
            container_name,
            _find_argv(roots, langs, large=True, extra_excludes=excludes),
            workdir=root,
            timeout=60,
            tenant_id=tenant_id,
        ),
    )
    listed = [line.strip() for line in (result.get("stdout") or "").splitlines() if line.strip()]
    if not listed and result.get("exit_code") not in (0, None):
        raise workspace_unresolvable_http(
            WorkspaceUnresolvableError(
                container_name, f"file listing failed: {(result.get('stderr') or '')[:300]}"
            )
        )
    rel_paths = sorted(
        {
            rel
            for rel in (_relative(p, root) for p in listed)
            if _wanted(rel, langs, prefixes, body.include_tests)
        }
    )
    skipped_large = sorted(
        {
            rel
            for rel in (
                _relative(line.strip(), root)
                for line in (large.get("stdout") or "").splitlines()
                if line.strip()
            )
            if _wanted(rel, langs, prefixes, body.include_tests)
        }
        - set(rel_paths)
    )
    loaded = _Loaded(files=[], matched=len(rel_paths) + len(skipped_large))
    loaded.skipped_large = skipped_large
    if len(rel_paths) > body.max_files:
        loaded.reasons.append("max_files")
    rel_paths = rel_paths[: body.max_files]

    total = 0
    for start in range(0, len(rel_paths), _READ_CONCURRENCY):
        batch = rel_paths[start : start + _READ_CONCURRENCY]
        contents = await asyncio.gather(
            *(
                client.read_file(container_name, f"{root}/{rel}", tenant_id=tenant_id)
                for rel in batch
            )
        )
        for rel, content in zip(batch, contents, strict=True):
            if content is None:
                loaded.unreadable.append(rel)
                continue
            size = len(content.encode("utf-8"))
            if size > MAX_FILE_BYTES:  # grew since the listing, or find rounded
                loaded.skipped_large.append(rel)
                continue
            if total + size > MAX_REQUEST_BYTES:
                loaded.reasons.append("max_bytes")
                return loaded
            total += size
            loaded.files.append((rel, content))
    return loaded


async def _load_files(body: WorkspaceScanRequest, tenant_id: str, langs: list[str]) -> _Loaded:
    """The files of a workspace or inline request, capped (see :class:`_Loaded`)."""
    if (body.container_name is None) == (body.sources is None):
        raise bad_request("Provide exactly one of container_name or sources")
    try:
        prefixes = [normalize_workspace_path(p) for p in body.paths]
    except InvalidWorkspacePathError as exc:
        raise invalid_workspace_path_http(exc) from exc
    if body.sources is None:
        return await _workspace_sources(body, tenant_id, langs, prefixes)
    try:
        inline = [(normalize_workspace_path(f.path), f.source) for f in body.sources]
    except InvalidWorkspacePathError as exc:
        raise invalid_workspace_path_http(exc) from exc
    # Per-file and total byte caps were enforced (422) when the request was validated.
    wanted = sorted((p, s) for p, s in inline if _wanted(p, langs, prefixes, body.include_tests))
    loaded = _Loaded(files=wanted[: body.max_files], matched=len(wanted))
    if len(wanted) > body.max_files:
        loaded.reasons.append("max_files")
    return loaded


def _limits(extraction: isolation.Limits) -> ScanLimits:
    return ScanLimits(deadline_seconds=extraction.deadline_s)


@dataclass
class _Extracted:
    """What one call-site extraction produced, and exactly which files it covered."""

    sites: list[cs.CallSite]
    unresolved: list[cs.UnresolvedCallSite]
    scanned: int  # files processed without error
    unparsed: dict[str, str]  # file -> reason (attempted, failed)
    not_reached: list[str]  # never attempted: the job stopped first
    failure: str | None  # deadline | worker_error | worker_died


async def _extract_call_sites(
    files: list[tuple[str, str]],
    kinds: set[str],
    max_call_sites: int,
    limits: isolation.Limits,
    tenant_id: str | None = None,
) -> _Extracted:
    """Call sites of *files*, accounted per file.

    Runs :func:`jobs.call_sites_job` in an isolated worker process. Per-file
    results stream back, so a job killed at the deadline still returns every file
    it finished; the file it was on is reported unparsed with the reason, and the
    files after it as not reached (as are the files after an early stop at the
    ``max_call_sites`` cap). Every input file is in exactly one of: scanned,
    unparsed, not_reached. Results arrive in file order, each file's sorted, so the
    lists stay globally sorted and are capped at ``max_call_sites + 1`` as they
    arrive (enough to flag truncation). ``busy`` (nothing ran) is raised.
    """
    outcomes, failure = await isolation.stream(
        jobs.call_sites_job, files, kinds, max_call_sites, limits=limits, tenant=tenant_id
    )
    if failure is not None and failure.reason == "busy":
        raise failure
    out = _Extracted(sites=[], unresolved=[], scanned=0, unparsed={}, not_reached=[], failure=None)
    for outcome in outcomes:
        if outcome.error is not None:
            out.unparsed[outcome.path] = outcome.error
            continue
        out.scanned += 1
        if len(out.sites) <= max_call_sites:
            out.sites.extend(outcome.call_sites[: max_call_sites + 1 - len(out.sites)])
        if len(out.unresolved) <= max_call_sites:
            out.unresolved.extend(outcome.unresolved[: max_call_sites + 1 - len(out.unresolved)])
    rest = [path for path, _ in files[len(outcomes) :]]
    if failure is not None:
        out.failure = failure.reason
        if rest:
            out.unparsed[rest[0]] = f"{failure.reason}: {failure.detail}"
            rest = rest[1:]
    out.not_reached = rest
    return out


@router.post(
    "/call-sites",
    response_model=CallSitesResponse,
    operation_id="scan_call_sites",
    summary="Outbound HTTP and event call sites (the consumer side of a contract)",
    description=(
        "Deterministic, read-only tree-sitter extraction of every outbound HTTP client call "
        "(fetch-style wrappers, axios, n8n httpRequest/request, requests, httpx, aiohttp, "
        "urllib, Rust reqwest) and message publish/subscribe call (NATS, Kafka, Redis pub/sub, AMQP) in a "
        "tenant workspace (`container_name`) or in inline `sources`. HTTP targets are "
        "normalised exactly like `scan_contracts` normalises route paths, so "
        "`(method, target)` joins a provider route. Targets that cannot be resolved "
        "statically are listed under `unresolved`, never guessed. Every HTTP call site "
        "reports where its base URL comes from (`base_source_kind`: literal | env | config | "
        "unknown, `base_source_name`: env var NAME / config key / host, "
        "`base_source_default`: an env lookup's literal default URL). Test and fixture files "
        "are skipped unless `include_tests` is set."
    ),
    responses={
        400: {"description": "Neither or both of container_name/sources, bad path or language"},
        404: {"description": "Workspace not found for this tenant"},
        422: {"description": "Workspace could not be read"},
    },
)
async def scan_call_sites(
    body: CallSitesRequest,
    tenant_id: str = Depends(require_tenant),
) -> CallSitesResponse:
    """List the outbound interface call sites of one workspace or set of files."""
    start = time.monotonic()
    langs = _selected_languages(body.languages)
    limits = isolation.limits_from_settings()
    async with _admitted(limits):  # before any file is read: busy costs nothing
        loaded = await _load_files(body, tenant_id, langs)
        extracted = await _extract_call_sites(
            loaded.files, {k.value for k in body.kinds}, body.max_call_sites, limits, tenant_id
        )
    sites, unresolved, unparsed = extracted.sites, extracted.unresolved, extracted.unparsed
    reasons = list(loaded.reasons)
    if len(sites) > body.max_call_sites or len(unresolved) > body.max_call_sites:
        reasons.append("max_call_sites")
        sites = sites[: body.max_call_sites]
        unresolved = unresolved[: body.max_call_sites]
    if extracted.failure is not None:
        reasons.append(extracted.failure)

    elapsed = time.monotonic() - start
    logger.info(
        "Call-site scan: %d files, %d call sites, %d unresolved in %.2fs (tenant=%s)",
        len(loaded.files),
        len(sites),
        len(unresolved),
        elapsed,
        tenant_id,
    )
    unparsed_files = sorted(unparsed)[:_MAX_LISTED_PATHS]
    return CallSitesResponse(
        call_sites=[
            CallSiteResponse(**cs.as_dict(s), repo=repo_of_path(s.file, body.repo)) for s in sites
        ],
        unresolved=[
            UnresolvedCallSiteResponse(**cs.as_dict(u), repo=repo_of_path(u.file, body.repo))
            for u in unresolved
        ],
        files_matched=loaded.matched,
        files_scanned=extracted.scanned,
        files_unreadable=loaded.unreadable[:_MAX_LISTED_PATHS],
        files_unparsed=unparsed_files,
        files_unparsed_reasons={f: unparsed[f] for f in unparsed_files},
        files_unparsed_count=len(unparsed),
        files_not_reached=extracted.not_reached[:_MAX_LISTED_PATHS],
        files_not_reached_count=len(extracted.not_reached),
        files_skipped_large=loaded.skipped_large[:_MAX_LISTED_PATHS],
        truncated=bool(reasons),
        truncation_reason=",".join(reasons) or None,
        languages=langs,
        processing_time_seconds=round(elapsed, 3),
        extractor_version=cs.EXTRACTOR_VERSION,
        limits=_limits(limits),
        repo=body.repo or "primary",
    )


# ---------------------------------------------------------------------------
# POST /routes -- the provider side, listed (pairs with /call-sites)
# ---------------------------------------------------------------------------


async def _extract_routes(
    loaded: _Loaded, body: RoutesRequest, limits: isolation.Limits, tenant_id: str
) -> pr.ProviderRoutesResult:
    """Provider routes of the loaded files in an isolated worker; ``busy`` is raised."""
    try:
        return await isolation.run(
            jobs.provider_routes_job,
            dict(loaded.files),
            body.max_routes,
            limits=limits,
            tenant=tenant_id,
        )
    except isolation.IsolationFailedError as exc:
        if exc.reason == "busy":
            raise
        # Mounts resolve across files, so there is no meaningful partial result.
        raise validation_error(
            f"Route extraction failed ({exc.reason}): {exc.detail}. Narrow `paths` and retry.",
            field="sources" if body.sources is not None else "paths",
            details={"reason": exc.reason, "deadline_seconds": limits.deadline_s},
        ) from exc


@router.post(
    "/routes",
    response_model=RoutesResponse,
    operation_id="scan_routes",
    summary="Provider HTTP routes of a workspace (pairs with scan_call_sites)",
    description=(
        "Deterministic, read-only listing of every HTTP route a workspace PROVIDES. "
        "Python: FastAPI decorators, APIRouter(prefix=...) and include_router(prefix=...) "
        "across files; Flask @app.route(methods=[...]) / verb decorators, Blueprint(url_prefix) "
        "and register_blueprint(url_prefix). Rust: axum Router::new().route(path, get(h).post(h2)), "
        ".nest(prefix, router), .merge(router), route_service/nest_service, with routers built "
        "in other functions/files resolved by name (`:id`/`{id}` params, `*rest` wildcards). "
        "TypeScript/JavaScript: method-first routers "
        "(`router.add('GET', '/api/x/:id', h)`, Hono `app.on`), Express/Fastify/Hono verb "
        "methods, `route('/x')` chains, `fastify.route({method, url})`, `url.pathname` "
        "dispatch (method ALL), and use()/route() mounts across files. Same request shape as "
        "`scan_call_sites`. A dynamic method, path or prefix, or a mount that cannot be "
        "resolved, is listed under `unresolved`, never guessed.\n\n"
        "Joining with `scan_call_sites`: both sides normalise paths identically ({param} "
        "for every placeholder, no trailing slash, no query). A call site hits a route when "
        "the methods are equal (a route method ALL, or a call-site method null, matches any) "
        "and the paths have the same number of segments with every segment equal or "
        "{param} on either side; a `wildcard` route matches anything under its path. When "
        "several routes are hit, take the most specific: score 2 per equal segment, 1 per "
        "route {param} against a call-site literal, 0 per call-site {param} against a route "
        "literal, and the highest total wins. A tie is AMBIGUOUS: report every tied route as "
        "a candidate instead of picking one (the call site cannot be attributed to a single "
        "route). A hit that needed a call-site {param} to stand for a route literal is WEAK: "
        "it is that route only if the runtime value equals the literal. "
        "`ts_route_parser.best_route` implements this rule and returns `route`, `score`, "
        "`weak`, `ambiguous` and `candidates`."
    ),
    responses={
        400: {"description": "Neither or both of container_name/sources, bad path or language"},
        404: {"description": "Workspace not found for this tenant"},
        422: {"description": "Workspace could not be read"},
    },
)
async def scan_routes(
    body: RoutesRequest,
    tenant_id: str = Depends(require_tenant),
) -> RoutesResponse:
    """List the provider routes of one workspace or set of files."""
    start = time.monotonic()
    langs = _selected_languages(body.languages, pr.ROUTE_LANGUAGES)
    limits = isolation.limits_from_settings()
    async with _admitted(limits):  # before any file is read: busy costs nothing
        loaded = await _load_files(body, tenant_id, langs)
        result = await _extract_routes(loaded, body, limits, tenant_id)
    routes, unresolved = result.routes, result.unresolved
    reasons = list(loaded.reasons)
    if result.truncated or len(routes) > body.max_routes or len(unresolved) > body.max_routes:
        reasons.append("max_routes")
        routes = routes[: body.max_routes]
        unresolved = unresolved[: body.max_routes]

    elapsed = time.monotonic() - start
    logger.info(
        "Route scan: %d files, %d routes, %d unresolved in %.2fs (tenant=%s)",
        len(loaded.files),
        len(routes),
        len(unresolved),
        elapsed,
        tenant_id,
    )
    unparsed_files = result.unparsed[:_MAX_LISTED_PATHS]
    return RoutesResponse(
        routes=[
            ProviderRouteResponse(**pr.as_dict(r), repo=repo_of_path(r.file, body.repo))
            for r in routes
        ],
        unresolved=[
            UnresolvedRouteResponse(**pr.as_dict(u), repo=repo_of_path(u.file, body.repo))
            for u in unresolved
        ],
        files_matched=loaded.matched,
        files_scanned=len(loaded.files) - len(result.unparsed),
        files_unreadable=loaded.unreadable[:_MAX_LISTED_PATHS],
        files_unparsed=unparsed_files,
        files_unparsed_reasons={
            f: result.unparsed_reasons[f] for f in unparsed_files if f in result.unparsed_reasons
        },
        files_unparsed_count=len(result.unparsed),
        # Routes are all-or-nothing (deadline -> 422): every loaded file was parsed.
        files_not_reached=[],
        files_not_reached_count=0,
        files_skipped_large=loaded.skipped_large[:_MAX_LISTED_PATHS],
        truncated=bool(reasons),
        truncation_reason=",".join(reasons) or None,
        languages=langs,
        processing_time_seconds=round(elapsed, 3),
        extractor_version=cs.EXTRACTOR_VERSION,
        limits=_limits(limits),
        repo=body.repo or "primary",
    )
