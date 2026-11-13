"""Extraction jobs the contract endpoints run in an isolated worker process.

Module-level functions with picklable arguments and results, so
:mod:`processing.contract_analysis.isolation` can run them in a separate process
(they are pickled by reference: this module is imported in the worker, the API
module is not). Each one is also callable in-process.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from . import call_sites as cs
from . import provider_routes as pr
from .parsers.ts_route_parser import parse_ts_routes
from .parsers.types import ServiceRoutes
from .scanner import ContractResult, ContractScanner


@dataclass
class FileCallSites:
    """One file's outcome. ``error`` set = the file is unparsed, and why."""

    path: str
    call_sites: list[cs.CallSite] = field(default_factory=list)
    unresolved: list[cs.UnresolvedCallSite] = field(default_factory=list)
    error: str | None = None


def call_sites_job(
    files: list[tuple[str, str]], kinds: set[str], max_call_sites: int
) -> Iterator[FileCallSites]:
    """Per-file call sites of *files*, in order, one result per file processed.

    One file failing (any exception, e.g. a RecursionError) is reported for that
    file and never fails the rest. Once more than *max_call_sites* call sites AND
    unresolved entries have been produced, the rest cannot change the (truncated)
    response, so extraction stops; past the cap for only one of the two lists, that
    list's entries are no longer sent.
    """
    sites_seen = unresolved_seen = 0
    for path, source in files:
        if sites_seen > max_call_sites and unresolved_seen > max_call_sites:
            return
        try:
            result = cs.extract_file_call_sites(path, source)
        except Exception as exc:  # one hostile file must not fail the request
            yield FileCallSites(path=path, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        if not result.parsed:
            yield FileCallSites(path=path, error="grammar not available")
            continue
        sites: list[cs.CallSite] = [s for s in result.call_sites if s.kind in kinds]
        unresolved: list[cs.UnresolvedCallSite] = [u for u in result.unresolved if u.kind in kinds]
        sites.sort(key=cs.sort_key)
        unresolved.sort(key=cs.sort_key)
        out = FileCallSites(
            path=path,
            call_sites=sites if sites_seen <= max_call_sites else [],
            unresolved=unresolved if unresolved_seen <= max_call_sites else [],
        )
        sites_seen += len(sites)
        unresolved_seen += len(unresolved)
        yield out


def provider_routes_job(files: dict[str, str], max_routes: int) -> pr.ProviderRoutesResult:
    return pr.extract_provider_routes(files, max_routes=max_routes)


#: ``scan_contracts`` checks every client call against every route; past this
#: many TS routes (mounts multiply out) the rest are dropped with a parse error.
MAX_SCAN_TS_ROUTES = 50_000


def ts_routes_job(files: dict[str, str]) -> ServiceRoutes:
    """TypeScript/JavaScript provider routes as ``scan_contracts`` consumes them."""
    parsed = parse_ts_routes(files, max_routes=MAX_SCAN_TS_ROUTES)
    routes = parsed.as_service_routes()
    if parsed.truncated:
        routes.routes = routes.routes[:MAX_SCAN_TS_ROUTES]
        routes.parse_errors.append(
            f"TypeScript routes truncated at {MAX_SCAN_TS_ROUTES} (mounts multiply out)"
        )
    return routes


@dataclass(frozen=True)
class PairChecks:
    """Which ``scan_contracts`` checks run (``ScanConfig``, picklable)."""

    syntax: bool = True
    routes: bool = True
    methods: bool = True
    request_fields: bool = True
    response_fields: bool = True
    types: bool = True
    query_params: bool = True
    status_codes: bool = True


def scan_pair_job(
    name: str,
    client_source: str,
    schema_sources: list[str],
    route_sources: list[str],
    ts_route_sources: dict[str, str],
    checks: PairChecks,
) -> ContractResult:
    """One ``scan_contracts`` pair: TS routes parsed, then every check, in one job.

    The checks (calls x routes matching) are as tenant-controlled as the parsing,
    so they run in the worker too.
    """
    ts_routes = ts_routes_job(ts_route_sources) if ts_route_sources else None
    return ContractScanner().scan_sources(
        name=name,
        client_source=client_source,
        schema_sources=schema_sources,
        route_sources=route_sources,
        ts_routes=ts_routes,
        check_syntax_enabled=checks.syntax,
        check_routes_enabled=checks.routes,
        check_methods_enabled=checks.methods,
        check_request_fields_enabled=checks.request_fields,
        check_response_fields_enabled=checks.response_fields,
        check_types_enabled=checks.types,
        check_query_params_enabled=checks.query_params,
        check_status_codes_enabled=checks.status_codes,
    )
