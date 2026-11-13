"""Every provider route of a workspace: Python (FastAPI, Flask), TypeScript/JavaScript, Rust (axum).

Python routes come from the existing :func:`parse_routes` (decorators plus the
file's own ``APIRouter(prefix=...)``). On top of that, this module applies the
prefixes of ``app.include_router(router, prefix="/api")`` across files, through
``from pkg.mod import router [as r]`` and ``from pkg import mod`` + ``mod.router``.
Flask follows Flask's own rules instead: ``register_blueprint(bp, url_prefix=...)``
REPLACES the blueprint's own ``url_prefix`` (it is not concatenated), and a nested
``parent.register_blueprint(child)`` appends the child's prefix to the parent's
effective one.
TypeScript/JavaScript routes come from :func:`parse_ts_routes`, which resolves its
own mounts.

Never guessed: a dynamic include prefix, an include target that is not a router
in the scanned files, or an ``APIRouter`` that nothing includes (when the scanned
files do include routers) goes to ``unresolved``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .call_sites import _SOURCE, _args, _callee, _literal, _raw, _short_text, _walk, language_for
from .checks.types import normalize_path
from .parsers.route_parser import parse_routes
from .parsers.ts_route_parser import (
    MountPrefixes,
    Unmountable,
    UnresolvedRoute,
    parse_ts_routes,
)

ROUTE_LANGUAGES: tuple[str, ...] = ("python", "javascript", "typescript", "rust")


@dataclass
class ProviderRoute:
    file: str
    line: int
    method: str  # GET/POST/... or ALL
    path: str  # normalised exactly like call-site targets
    framework: str  # fastapi | method-first | express | pathname | axum
    raw: str = ""
    handler: str | None = None
    mount_prefix: str | None = None
    wildcard: bool = False


@dataclass
class ProviderRoutesResult:
    routes: list[ProviderRoute]
    unresolved: list[UnresolvedRoute]
    unparsed: list[str]
    #: file -> why it was not parsed (grammar missing, or the extractor failed on it).
    unparsed_reasons: dict[str, str] = field(default_factory=dict)
    truncated: bool = False  # stopped at ``max_routes``


def as_dict(item: ProviderRoute | UnresolvedRoute) -> dict[str, Any]:
    return asdict(item)


# ---------------------------------------------------------------------------
# Python include_router mounts
# ---------------------------------------------------------------------------


@dataclass
class _PyModule:
    imports: dict[str, tuple[str, str | None]]  # local -> (module, attribute or None)
    routers: dict[str, bool]  # local -> True for APIRouter(), False for FastAPI()
    router_prefix: dict[str, str | None]  # local -> APIRouter(prefix=...) (None = dynamic)
    mounts: list[tuple[str, Any, Any, int]]  # (owner, target node, prefix node|None, line)
    flask: set[str] = field(default_factory=set)  # locals bound to Blueprint()/Flask()


def _py_module_shape(root: Any) -> _PyModule:
    shape = _PyModule(imports={}, routers={}, router_prefix={}, mounts=[])
    for node in _walk(root):
        if node.type == "import_from_statement":
            module = _short_text(node.child_by_field_name("module_name"))
            for child in node.children_by_field_name("name"):
                if child.type == "aliased_import":
                    name = _short_text(child.child_by_field_name("name"))
                    local = _short_text(child.child_by_field_name("alias"))
                else:
                    name = local = _short_text(child)
                shape.imports[local] = (module, name)
        elif node.type == "import_statement":
            for child in node.named_children:
                if child.type == "aliased_import":
                    shape.imports[_short_text(child.child_by_field_name("alias"))] = (
                        _short_text(child.child_by_field_name("name")),
                        None,
                    )
        elif node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None or left.type != "identifier" or right.type != "call":
                continue
            fn = _short_text(right.child_by_field_name("function")).split(".")[-1]
            if fn in ("APIRouter", "FastAPI", "Blueprint", "Flask"):
                shape.routers[_short_text(left)] = fn in ("APIRouter", "Blueprint")
                if fn in ("Blueprint", "Flask"):
                    shape.flask.add(_short_text(left))
                _, kw = _args(right)
                ctor_prefix = kw.get("prefix") or kw.get("url_prefix")
                shape.router_prefix[_short_text(left)] = (
                    "" if ctor_prefix is None else _literal(ctor_prefix)
                )
        elif node.type == "call":
            name, receiver, _ = _callee(node)
            if name not in ("include_router", "register_blueprint") or receiver is None:
                continue
            positional, keywords = _args(node)
            target = (
                positional[0] if positional else keywords.get("router", keywords.get("blueprint"))
            )
            if target is not None:
                shape.mounts.append(
                    (
                        receiver,
                        target,
                        keywords.get("prefix", keywords.get("url_prefix")),
                        node.start_point[0] + 1,
                    )
                )
    return shape


#: The last two path segments -> the scanned files ending in them.
_SuffixIndex = dict[str, list[str]]


def _tail(path: str) -> str:
    return "/".join(path.rsplit("/", 2)[-2:])


def _suffix_index(paths: set[str]) -> _SuffixIndex:
    """Built once, so a module lookup checks a handful of files, not every file."""
    index: _SuffixIndex = {}
    for path in sorted(paths):
        index.setdefault(_tail(path), []).append(path)
    return index


def _module_file(module: str, files: _SuffixIndex) -> str | None:
    """The ONE scanned file a dotted module names (any source root), else None."""
    rel = module.replace(".", "/")
    hits = [
        f
        for name in (f"{rel}.py", f"{rel}/__init__.py")
        for f in files.get(_tail(name), ())
        if f == name or f.endswith("/" + name)
    ]
    return hits[0] if len(hits) == 1 else None


def _relative_module(importer: str, module: str) -> str:
    """``from .x import y`` in ``a/b/c.py`` -> ``a.b.x`` (dots -> parent packages)."""
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    parent = PurePosixPath(importer).parent
    for _ in range(level - 1):
        parent = parent.parent
    rest = module.lstrip(".")
    base = ".".join(p for p in parent.parts if p)
    return f"{base}.{rest}" if rest and base else (rest or base)


def _follow(
    file: str, name: str, shapes: dict[str, _PyModule], files: _SuffixIndex, depth: int = 0
) -> tuple[str, str] | None:
    """The (file, router local) that *name* in *file* is, through re-exports."""
    shape = shapes[file]
    if name in shape.routers:
        return (file, name)
    if depth > 5 or name not in shape.imports:
        return None
    module, attr = shape.imports[name]
    module = _relative_module(file, module)
    if attr is None:
        return None
    found = _module_file(module, files)
    if found is not None:
        hit = _follow(found, attr, shapes, files, depth + 1)
        if hit is not None:
            return hit
    found = _module_file(f"{module}.{attr}", files)  # from pkg import mod -> mod.router
    if found is not None:
        return _follow(found, "router", shapes, files, depth + 1)
    return None


def _py_target(
    path: str, target: Any, shapes: dict[str, _PyModule], files: _SuffixIndex
) -> tuple[str, str] | None:
    shape = shapes[path]
    if target.type == "identifier":
        return _follow(path, _short_text(target), shapes, files)
    if target.type == "attribute":  # mod.router
        obj = _short_text(target.child_by_field_name("object"))
        attr = _short_text(target.child_by_field_name("attribute"))
        if obj in shape.imports:
            module, name = shape.imports[obj]
            module = _relative_module(path, module)
            found = _module_file(f"{module}.{name}" if name else module, files)
            if found is not None:
                return _follow(found, attr, shapes, files)
    return None


def _python_routes(
    files: dict[str, str], result: ProviderRoutesResult, max_routes: int | None = None
) -> None:
    shapes: dict[str, _PyModule] = {}
    for path in sorted(files):
        content = files[path].encode("utf-8")
        tree = _parse("python", content)
        if tree is None:
            _unparsed(result, path, "grammar not available")
            continue
        token = _SOURCE.set(content)
        try:
            shapes[path] = _py_module_shape(tree.root_node)
        except Exception as exc:  # one hostile file must not fail the whole scan
            _unparsed(result, path, f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            _SOURCE.reset(token)
    names = _suffix_index(set(shapes))

    edges: dict[Any, list[tuple[Any, str | None, Any]]] = {}
    flask_mounts: list[tuple[tuple[str, str], tuple[str, str], str | None]] = []
    for path, shape in shapes.items():
        for owner, target, prefix_node, line in shape.mounts:
            resolved = _py_target(path, target, shapes, names)
            if resolved is None:
                result.unresolved.append(
                    UnresolvedRoute(
                        file=path,
                        line=line,
                        expression=_raw(target),
                        reason="included router is not a router in the scanned files",
                    )
                )
                continue
            prefix = "" if prefix_node is None else _literal(prefix_node)
            if _is_blueprint(shapes, resolved):
                # Flask: register_blueprint(url_prefix=...) REPLACES the blueprint's
                # own url_prefix; the parent's effective prefix comes down the chain.
                own = shapes[resolved[0]].router_prefix.get(resolved[1], "")
                flask_mounts.append(
                    ((path, owner), resolved, own if prefix_node is None else prefix)
                )
                continue
            # FastAPI: the parent's own APIRouter(prefix=...) applies to what it
            # includes, and include_router(prefix=...) is concatenated with it.
            parent_prefix = shape.router_prefix.get(owner, "")
            full = None if parent_prefix is None or prefix is None else parent_prefix + prefix
            edges.setdefault(resolved, []).append(((path, owner), full, None))
    mounted = set(edges) | {child for _, child, _ in flask_mounts}
    for parent, child, segment in flask_mounts:
        # An unregistered parent blueprint still applies its own url_prefix.
        base: str | None = ""
        if _is_blueprint(shapes, parent) and parent not in mounted:
            base = shapes[parent[0]].router_prefix.get(parent[1], "")
        full = None if base is None or segment is None else _flask_join(base, segment)
        edges.setdefault(child, []).append((parent, full, None))
    any_mounts = bool(edges)
    prefixes_of = MountPrefixes(edges)

    for path in sorted(shapes):
        if max_routes is not None and len(result.routes) > max_routes:
            result.truncated = True
            return
        try:
            parsed = parse_routes(files[path])
        except Exception as exc:
            _unparsed(result, path, f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        for line, rule, expression, reason in parsed.unresolved:
            result.unresolved.append(
                UnresolvedRoute(file=path, line=line, expression=rule or expression, reason=reason)
            )
        for route in parsed.routes:
            node = (path, route.router)
            is_sub_router = shapes[path].routers.get(route.router, False)
            if any_mounts and is_sub_router and node not in edges:
                result.unresolved.append(
                    UnresolvedRoute(
                        file=path,
                        line=route.line,
                        expression=route.path,
                        reason=(
                            "Blueprint that no scanned file registers"
                            if _is_blueprint(shapes, node)
                            else "APIRouter that no scanned file includes"
                        ),
                        method=route.method,
                    )
                )
                continue
            for prefix in prefixes_of(node):
                if isinstance(prefix, Unmountable):
                    result.unresolved.append(
                        UnresolvedRoute(
                            file=path,
                            line=route.line,
                            expression=route.path,
                            reason=prefix.describe("included"),
                            method=route.method,
                        )
                    )
                    continue
                full_path = prefix + route.path
                if node in edges and _is_blueprint(shapes, node):
                    # The mount chain already carries this blueprint's effective prefix.
                    full_path = _flask_join(prefix, route.rule)
                result.routes.append(
                    ProviderRoute(
                        file=path,
                        line=route.line,
                        method=route.method.upper(),
                        path=normalize_path(full_path) or "/",
                        framework="flask" if route.router in shapes[path].flask else "fastapi",
                        raw=route.path,
                        handler=route.handler or None,
                        mount_prefix=prefix or None,
                    )
                )


def _is_blueprint(shapes: dict[str, _PyModule], node: tuple[str, str]) -> bool:
    shape = shapes[node[0]]
    return node[1] in shape.flask and shape.routers.get(node[1], False)


def _flask_join(prefix: str, rule: str) -> str:
    """Flask's own join: ``prefix.rstrip('/') + '/' + rule.lstrip('/')``."""
    if not prefix:
        return rule
    if not rule:
        return prefix
    return prefix.rstrip("/") + "/" + rule.lstrip("/")


def _unparsed(result: ProviderRoutesResult, path: str, reason: str) -> None:
    if path not in result.unparsed_reasons:
        result.unparsed.append(path)
    result.unparsed_reasons[path] = reason


def _rust_routes(
    files: dict[str, str], result: ProviderRoutesResult, max_routes: int | None = None
) -> None:
    from .rust_http import extract_axum_routes

    budget = None if max_routes is None else max(max_routes - len(result.routes), 0)
    axum = extract_axum_routes(files, budget)
    result.truncated = result.truncated or axum.truncated
    for path, reason in axum.unparsed.items():
        _unparsed(result, path, reason)
    result.unresolved.extend(axum.unresolved)
    result.routes.extend(
        ProviderRoute(
            file=r.file,
            line=r.line,
            method=r.method,
            path=r.path,
            framework=r.framework,
            raw=r.raw,
            handler=r.handler,
            mount_prefix=r.mount_prefix,
            wildcard=r.wildcard,
        )
        for r in axum.routes
    )


def extract_provider_routes(
    files: dict[str, str], max_routes: int | None = None
) -> ProviderRoutesResult:
    """Provider routes of *files* (``{repo-relative path: source}``), sorted.

    With *max_routes*, extraction stops (``truncated=True``) once more than that
    many routes are listed, so a mount graph that multiplies out stays bounded.
    """
    result = ProviderRoutesResult(routes=[], unresolved=[], unparsed=[])
    python = {p: s for p, s in files.items() if language_for(p) == "python"}
    scripts = {p: s for p, s in files.items() if language_for(p) in ("javascript", "typescript")}
    rust = {p: s for p, s in files.items() if language_for(p) == "rust"}
    if python:
        _python_routes(python, result, max_routes)
    if rust and not result.truncated:
        _rust_routes(rust, result, max_routes)
    if scripts and not result.truncated:
        budget = None if max_routes is None else max(max_routes - len(result.routes), 0)
        ts = parse_ts_routes(scripts, budget)
        result.truncated = ts.truncated
        for error in ts.parse_errors:
            path, _, reason = error.partition(": ")
            _unparsed(result, path, reason)
        result.unresolved.extend(ts.unresolved)
        result.routes.extend(
            ProviderRoute(
                file=r.file,
                line=r.line,
                method=r.method,
                path=r.path,
                framework=r.framework,
                raw=r.raw,
                handler=r.handler,
                mount_prefix=r.mount_prefix,
                wildcard=r.wildcard,
            )
            for r in ts.routes
        )
    result.routes.sort(key=lambda r: (r.file, r.line, r.method, r.path))
    result.unresolved.sort(key=lambda u: (u.file, u.line))
    result.unparsed.sort()
    return result
