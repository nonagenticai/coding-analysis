"""Cross-repo service call graph: every outbound HTTP call of one repo matched to a route of another.

Per repo the same extractors as ``/contracts/call-sites`` and ``/contracts/routes``
run (Python httpx/requests/aiohttp, TS/JS, Rust reqwest; FastAPI/Flask, TS
routers, axum). Then each HTTP call site's target repo is decided, in order:

1. ``hint``: the caller's hint map ``{ENV_VAR_OR_CONFIG_KEY_OR_HOST: repo}``
   matched against the base URL's env var / config key name (case-insensitive),
   the literal host (``svc-b:8080`` or ``svc-b``) and an env default URL's host;
2. ``service_name``: the host (or env default host) is a docker-compose service
   (with a ``build``), ``container_name``/``hostname`` of one, or a Kubernetes
   ``Service``/``Deployment``/``StatefulSet``/``DaemonSet`` name (``_K8S_KINDS``)
   defined in exactly one repo. A compose/k8s YAML file that cannot be read is
   listed in that repo's ``files_unparsed``;
3. ``route_only``: no repo named; the call's ``(method, path)`` hits routes of
   exactly ONE other repo.

Then the route is picked with the same rule as ``scan_routes`` documents
(``ts_route_parser.best_route``). ``match``: ``exact`` (every segment literal and
equal), ``template`` (a ``{param}`` on either side), ``prefix`` (a wildcard route,
or the call targets a path some route lives under), ``unresolved`` (no target
repo, an ambiguous one, or no route of the target repo matches). Nothing is
dropped: every HTTP call site is one edge.

Message-based coupling (Redis pub/sub, Redis streams, celery/rq/arq task queues;
Python and the Rust redis crate) is matched by channel name in
:mod:`processing.structure.messaging`: ``message_edges`` plus the publishers and
consumers nothing matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

import yaml

from processing.contract_analysis import call_sites as cs
from processing.contract_analysis import provider_routes as pr
from processing.contract_analysis.base_source import host_of
from processing.contract_analysis.parsers.ts_route_parser import TsRoute, best_route
from processing.workspace_repos import PRIMARY_REPO, normalize_repo, repo_of_path

from .messaging import (
    MessageEdge,
    MessageEndpoint,
    RepoMessaging,
    extract_messaging,
    match_messages,
)
from .text import as_str, get_path

_COMPOSE_NAMES = ("docker-compose", "compose")
_K8S_KINDS = {"Service", "Deployment", "StatefulSet", "DaemonSet"}


@dataclass
class RouteRef:
    method: str
    path: str
    file: str
    line: int
    handler: str | None = None


@dataclass
class ServiceEdge:
    from_repo: str
    from_file: str
    from_line: int
    from_symbol: str | None
    method: str | None
    from_target: str  # the caller's request path
    match: str  # exact | template | prefix | unresolved
    to_repo: str | None = None
    to_route: RouteRef | None = None
    resolved_by: str | None = None  # hint | service_name | route_only
    from_base: str | None = None
    from_host: str | None = None
    from_base_source_kind: str | None = None
    from_base_source_name: str | None = None
    candidates: list[str] = field(default_factory=list)  # repos or routes when ambiguous
    reason: str | None = None


@dataclass
class RepoSummary:
    repo: str
    routes: int
    http_call_sites: int
    unresolved_call_sites: int
    files_unparsed: dict[str, str] = field(default_factory=dict)
    message_publishers: int = 0
    message_consumers: int = 0


@dataclass
class UnresolvedCall:
    repo: str
    file: str
    line: int
    method: str | None
    expression: str
    reason: str


@dataclass
class ServiceGraph:
    edges: list[ServiceEdge]
    repos: list[RepoSummary]
    unresolved_call_sites: list[UnresolvedCall]
    service_names: dict[str, str]
    service_name_conflicts: dict[str, list[str]]
    message_edges: list[MessageEdge] = field(default_factory=list)
    unmatched_publishers: list[MessageEndpoint] = field(default_factory=list)
    unmatched_consumers: list[MessageEndpoint] = field(default_factory=list)


def _host_only(host: str | None) -> str | None:
    return host.split(":", 1)[0] if host else None


def service_names(
    repo_files: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, str]]]:
    """Host name -> the repo that defines it (compose / k8s), names claimed by >1 repo, and
    per repo the YAML files that could not be read (``{repo: {path: reason}}``)."""
    claims: dict[str, set[str]] = {}
    skipped: dict[str, dict[str, str]] = {}
    for repo, files in repo_files.items():
        for path, text in files.items():
            name = PurePosixPath(path).name
            if not name.endswith((".yml", ".yaml")):
                continue
            try:
                docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
            except yaml.YAMLError as exc:
                skipped.setdefault(repo, {})[path] = str(exc)
                continue
            for doc in docs:
                services = doc.get("services")
                if name.startswith(_COMPOSE_NAMES) and isinstance(services, dict):
                    for svc, body in services.items():
                        if not isinstance(body, dict) or "build" not in body:
                            continue  # an image someone else builds (postgres, redis...)
                        for alias in (svc, body.get("container_name"), body.get("hostname")):
                            if isinstance(alias, str) and alias:
                                claims.setdefault(alias, set()).add(repo)
                elif isinstance(doc.get("kind"), str) and doc["kind"] in _K8S_KINDS:
                    meta_name = as_str(get_path(doc, "metadata", "name"))
                    if meta_name:
                        claims.setdefault(meta_name, set()).add(repo)
    names = {k: next(iter(v)) for k, v in claims.items() if len(v) == 1}
    conflicts = {k: sorted(v) for k, v in claims.items() if len(v) > 1}
    return names, conflicts, skipped


@dataclass
class RepoFacts:
    routes: list[pr.ProviderRoute]
    sites: list[cs.CallSite]
    unresolved: list[cs.UnresolvedCallSite]
    unparsed: dict[str, str]
    messaging: RepoMessaging | None = None


def repo_facts(
    files: dict[str, str], include_tests: bool = False, repo: str = PRIMARY_REPO
) -> RepoFacts:
    """Routes, HTTP call sites and message endpoints of ONE repo (repo-relative paths)."""
    wanted = {
        p: s
        for p, s in files.items()
        if cs.language_for(p) is not None and (include_tests or not cs.is_test_path(p))
    }
    route_files = {p: s for p, s in wanted.items() if cs.language_for(p) in pr.ROUTE_LANGUAGES}
    routes = pr.extract_provider_routes(route_files)
    sites: list[cs.CallSite] = []
    unresolved: list[cs.UnresolvedCallSite] = []
    unparsed = dict(routes.unparsed_reasons)
    for path in sorted(wanted):
        try:
            result = cs.extract_file_call_sites(path, wanted[path])
        except (RecursionError, ValueError) as exc:
            unparsed[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        if not result.parsed:
            unparsed.setdefault(path, "grammar not available")
            continue
        sites.extend(s for s in result.call_sites if s.kind == "http")
        unresolved.extend(u for u in result.unresolved if u.kind == "http")
    messaging = extract_messaging(repo, wanted)
    for path, reason in messaging.unparsed.items():
        unparsed.setdefault(path, reason)
    return RepoFacts(routes.routes, sites, unresolved, unparsed, messaging)


def _ts(route: pr.ProviderRoute) -> TsRoute:
    return TsRoute(
        method=route.method,
        path=route.path,
        file=route.file,
        line=route.line,
        framework=route.framework,
        wildcard=route.wildcard,
        raw=route.raw,
        handler=route.handler,
        mount_prefix=route.mount_prefix,
    )


def _ref(route: TsRoute) -> RouteRef:
    return RouteRef(route.method, route.path, route.file, route.line, route.handler)


def _match(site: cs.CallSite, routes: list[TsRoute]) -> tuple[str, TsRoute | None, list[TsRoute]]:
    hit = best_route(site.method, site.target, routes)
    if hit is not None:
        route = hit.route
        if route.wildcard:
            kind = "prefix"
        elif site.target == route.path and "{param}" not in route.path:
            kind = "exact"
        else:
            kind = "template"
        return kind, route, hit.candidates if hit.ambiguous else []
    under = [
        r
        for r in routes
        if site.target != "/"
        and r.path.startswith(site.target.rstrip("/") + "/")
        and (site.method is None or r.method in ("ALL", site.method))
    ]
    if under:
        return "prefix", sorted(under, key=lambda r: (r.path, r.method))[0], []
    return "unresolved", None, []


def _hinted(site: cs.CallSite, hints: dict[str, str]) -> str | None:
    lowered = {k.lower(): v for k, v in hints.items()}
    keys = [
        site.base_source_name if site.base_source_kind in ("env", "config") else None,
        site.host,
        _host_only(site.host),
        host_of(site.base_source_default),
        _host_only(host_of(site.base_source_default)),
    ]
    for key in keys:
        if key and key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _named(site: cs.CallSite, names: dict[str, str]) -> str | None:
    for host in (site.host, host_of(site.base_source_default)):
        name = _host_only(host)
        if name and name in names:
            return names[name]
    return None


#: A call that matches more repos' routes than this is ambiguous; the rest are not listed.
_MAX_ROUTE_ONLY_HITS = 10


def _first_segment(path: str) -> str:
    return path.lstrip("/").split("/", 1)[0]


class _RouteIndex:
    """Repos that can match a target, by the target's first path segment.

    Trying every repo's routes for every call site cost O(sites x repos). A route
    can only match a target whose first segment equals its own, unless the route's
    first segment is a parameter, a wildcard, or the route is ``/``; those repos are
    candidates for every target. Repos with no routes are never candidates.
    """

    def __init__(self, routes: dict[str, list[TsRoute]]) -> None:
        self.by_first: dict[str, set[str]] = {}
        self.generic: set[str] = set()
        for repo, repo_routes in routes.items():
            for r in repo_routes:
                first = _first_segment(r.path)
                if not first or first.startswith(("{", "*", ":")) or r.wildcard:
                    self.generic.add(repo)
                else:
                    self.by_first.setdefault(first, set()).add(repo)

    def candidates(self, target: str) -> list[str]:
        return sorted(self.by_first.get(_first_segment(target), set()) | self.generic)


def build_service_graph(
    repo_files: dict[str, dict[str, str]],
    hints: dict[str, str] | None = None,
    include_tests: bool = False,
) -> ServiceGraph:
    """Match every repo's outbound HTTP calls to the other repos' routes."""
    hints = canonical_hints(hints)
    names, conflicts, yaml_errors = service_names(repo_files)
    facts = {
        repo: repo_facts(files, include_tests, repo) for repo, files in sorted(repo_files.items())
    }
    routes = {repo: [_ts(r) for r in f.routes] for repo, f in facts.items()}
    route_index = _RouteIndex(routes)
    edges: list[ServiceEdge] = []
    unresolved_calls: list[UnresolvedCall] = []
    for repo, f in facts.items():
        for u in f.unresolved:
            unresolved_calls.append(
                UnresolvedCall(repo, u.file, u.line, u.method, u.expression, u.reason)
            )
        for site in f.sites:
            edge = ServiceEdge(
                from_repo=repo,
                from_file=site.file,
                from_line=site.line,
                from_symbol=site.symbol,
                method=site.method,
                from_target=site.target,
                match="unresolved",
                from_base=site.base,
                from_host=site.host,
                from_base_source_kind=site.base_source_kind,
                from_base_source_name=site.base_source_name,
            )
            target_repo = _hinted(site, hints)
            if target_repo is not None:
                edge.resolved_by = "hint"
            else:
                target_repo = _named(site, names)
                if target_repo is not None:
                    edge.resolved_by = "service_name"
            if target_repo is not None:
                edge.to_repo = target_repo
                if target_repo not in routes:
                    edge.reason = f"target repo {target_repo!r} is not in the workspace"
                else:
                    kind, route, tied = _match(site, routes[target_repo])
                    edge.match = kind
                    edge.to_route = _ref(route) if route is not None else None
                    edge.candidates = [f"{r.method} {r.path} ({r.file}:{r.line})" for r in tied]
                    if route is None:
                        edge.reason = f"no route of {target_repo} matches"
                edges.append(edge)
                continue
            hits: dict[str, tuple[str, TsRoute | None, list[TsRoute]]] = {}
            for other in route_index.candidates(site.target):
                if other == repo:
                    continue
                kind, route, tied = _match(site, routes[other])
                if route is not None and kind != "prefix":
                    hits[other] = (kind, route, tied)
                    if len(hits) > _MAX_ROUTE_ONLY_HITS:
                        break  # ambiguous already; listing every repo would be O(repos^2)
            if len(hits) == 1:
                other, (kind, route, tied) = next(iter(hits.items()))
                edge.to_repo = other
                edge.resolved_by = "route_only"
                edge.match = kind
                edge.to_route = _ref(route) if route is not None else None
                edge.candidates = [f"{r.method} {r.path} ({r.file}:{r.line})" for r in tied]
            elif hits:
                edge.candidates = sorted(hits)[:_MAX_ROUTE_ONLY_HITS]
                edge.reason = "routes of several repos match and no hint/service name decides" + (
                    f" (more than {_MAX_ROUTE_ONLY_HITS}; the first are listed)"
                    if len(hits) > _MAX_ROUTE_ONLY_HITS
                    else ""
                )
            else:
                edge.reason = (
                    "no hint or service name names the target, and no other repo's route matches"
                )
            edges.append(edge)
    summaries = [
        RepoSummary(
            repo=repo,
            routes=len(f.routes),
            http_call_sites=len(f.sites),
            unresolved_call_sites=len(f.unresolved),
            files_unparsed={**yaml_errors.get(repo, {}), **f.unparsed},
            message_publishers=sum(
                e.role == "publisher" for e in (f.messaging.endpoints if f.messaging else [])
            ),
            message_consumers=sum(
                e.role == "consumer" for e in (f.messaging.endpoints if f.messaging else [])
            ),
        )
        for repo, f in facts.items()
    ]
    edges.sort(key=lambda e: (e.from_repo, e.from_file, e.from_line))
    messages = match_messages({repo: f.messaging for repo, f in facts.items() if f.messaging})
    return ServiceGraph(
        edges,
        summaries,
        unresolved_calls,
        dict(sorted(names.items())),
        conflicts,
        messages.edges,
        messages.unmatched_publishers,
        messages.unmatched_consumers,
    )


def split_by_repo(files: dict[str, str]) -> dict[str, dict[str, str]]:
    """Workspace-root-relative files -> ``{repo: {repo-relative path: text}}``."""
    out: dict[str, dict[str, str]] = {}
    for path, text in files.items():
        repo = repo_of_path(path)
        rel = path if repo == PRIMARY_REPO else path[len(repo) + 1 :]
        out.setdefault(repo, {})[rel] = text
    return out


def canonical_hints(hints: dict[str, str] | None) -> dict[str, str]:
    """Hint values as canonical repo selectors (``acme__b`` -> ``.repos/acme__b``)."""
    return {k: normalize_repo(v) for k, v in (hints or {}).items()}
