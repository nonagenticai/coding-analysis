"""Provider routes from TypeScript/JavaScript sources: the TS side of ``parse_routes``.

Three registration shapes, all read from the tree-sitter tree:

* ``router.add('GET', '/api/x/:id', handler)``: a method-first router (fr's own
  ``utils/router.ts``; Hono's ``app.on('GET', ...)`` has the same shape). A
  method bound by ``for (const m of ['GET', 'POST'])`` expands to each literal.
* ``app.get('/x', handler)`` / ``router.post(...)`` / ``router.route('/x').get(h)``
  / ``fastify.route({method, url})``: Express, Fastify and Hono verb methods.
* ``app.use('/api/users', usersRouter)`` / ``app.route('/api', sub)``: a mount.
  The prefix is applied to every route registered on the mounted router, through
  same-file variables and relative imports among the files given. A mount whose
  prefix or target cannot be resolved statically sends the affected routes to
  ``unresolved``, never to a guessed path.

Paths are normalised with the SAME :func:`normalize_path` the consumer side
(``call_sites``) uses, so ``(method, target)`` joins directly. A trailing regex
wildcard (``/x(.*)``, ``/x/*``) is kept as ``wildcard=True`` on its prefix.

Mount prefixes are resolved by :class:`MountPrefixes`: memoised per router,
cycle-checked on the current path, depth- and count-capped, so a self- or
mutually-mounting router (``router.use('/p', router)``) is listed as unresolved
instead of being expanded exponentially.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from ..call_sites import (
    _EXPRESS_PARAM,
    _SOURCE,
    HTTP_VERBS,
    _args,
    _callee,
    _collect_constants,
    _literal,
    _obj_props,
    _raw,
    _receiver_token,
    _render,
    _Resolver,
    _short_text,
)
from ..checks.types import normalize_path
from .types import RouteDefinition, ServiceRoutes

TS_ROUTE_EXTENSIONS = (".ts", ".mts", ".cts", ".tsx", ".js", ".mjs", ".cjs", ".jsx")
_METHODS = {v.upper() for v in HTTP_VERBS}
_METHOD_FIRST = {"add", "on", "addRoute"}
_VERB_METHODS = HTTP_VERBS | {"all"}
_REGISTRAR = re.compile(r"^(app|router|server|fastify|api|hono|r|routes?|.*router|.*app)$")
_ROUTER_FACTORY = re.compile(r"(^|\.)(Router|Hono|express)$")
_WILDCARD = re.compile(r"(\(\.\*\)|/\*|\*)$")
#: ``for (const m of [...])`` loops looked through for a method variable.
_MAX_LOOP_HOPS = 16
#: ``.route('/x').get(h).post(h)...`` calls looked through for the ``.route()``.
_MAX_CHAIN = 10


@dataclass
class TsRoute:
    method: str  # GET/POST/... or ALL
    path: str  # normalised, joinable
    file: str
    line: int
    framework: str  # method-first | express | pathname
    wildcard: bool = False
    raw: str = ""
    handler: str | None = None  # handler identifier as written, None when inline
    mount_prefix: str | None = None  # prefix contributed by use()/route() mounts


@dataclass
class UnresolvedRoute:
    file: str
    line: int
    expression: str
    reason: str
    method: str | None = None


@dataclass
class TsRouteResult:
    routes: list[TsRoute] = field(default_factory=list)
    unresolved: list[UnresolvedRoute] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    truncated: bool = False  # stopped at ``max_routes``

    def as_service_routes(self) -> ServiceRoutes:
        return ServiceRoutes(
            routes=[RouteDefinition(method=r.method, path=r.path) for r in self.routes],
            parse_errors=list(self.parse_errors),
        )


@dataclass
class _Pending:
    owner: str  # receiver variable the route is registered on
    method: str
    path: str
    wildcard: bool
    line: int
    framework: str
    raw: str
    handler: str | None = None


@dataclass
class _Mount:
    owner: str
    prefix: str | None  # None = dynamic
    target: str  # local identifier of the mounted router
    line: int
    raw: str


@dataclass
class _FileRoutes:
    pending: list[_Pending] = field(default_factory=list)
    mounts: list[_Mount] = field(default_factory=list)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)  # local -> (spec, name)
    exports: dict[str, str] = field(default_factory=dict)  # exported name -> local
    routers: set[str] = field(default_factory=set)  # locals created by Router()/new Hono()
    unresolved: list[UnresolvedRoute] = field(default_factory=list)
    owners: set[str] = field(default_factory=set)  # receivers routes are registered on


def _is_pathname(node: Any) -> bool:
    text = _short_text(node)
    return text == "pathname" or text.endswith(".pathname")


def _grammar(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".tsx"):
        return "tsx"
    if lower.endswith((".ts", ".mts", ".cts")):
        return "typescript"
    return "javascript"


def _route_path(parts_text: str) -> tuple[str, bool] | None:
    """Normalise a registered pattern; ``None`` when it is not an absolute path."""
    if not parts_text.startswith("/"):
        return None
    text = parts_text
    wildcard = False
    match = _WILDCARD.search(text)
    if match:
        wildcard = True
        text = text[: match.start()]
    text = _EXPRESS_PARAM.sub("/{param}", text)
    text = re.sub(r"\{[^{}]*\}", "{param}", text)  # [^{}]: linear on runs of "{"
    return (normalize_path(text) or "/"), wildcard


#: The enclosing ``for_in_statement`` nodes of a walk position, innermost first,
#: as a linked chain ``(node, outer)`` carried down the walk.
_Loops = tuple[Any, Any] | None


def _walk_ctx(root: Any):
    """``(node, parent, grandparent, loops)`` in pre-order, one pass, no ``.parent``."""
    stack: list[tuple[Any, Any, Any, _Loops]] = [(root, None, None, None)]
    while stack:
        node, parent, grand, loops = stack.pop()
        yield node, parent, grand, loops
        inner = (node, loops) if node.type == "for_in_statement" else loops
        stack.extend((child, node, parent, inner) for child in reversed(node.children))


def _for_of_values(loops: _Loops, name: str) -> list[str] | None:
    """Literal values of the innermost ``for (const <name> of ['A', 'B'])`` around a call."""
    hops = 0
    while loops is not None and hops < _MAX_LOOP_HOPS:
        cur, loops = loops
        hops += 1
        left = cur.child_by_field_name("left")
        right = cur.child_by_field_name("right")
        if _short_text(left) == name and right is not None and right.type == "array":
            values = [_literal(item) for item in right.named_children]
            if values and all(v is not None for v in values):
                return [v for v in values if v is not None]
            return None
    return None


def _root_identifier(node: Any) -> str:
    """``router`` for ``router``, ``this.router`` -> ``this.router`` (owner key)."""
    return _short_text(node).strip()


class _Collector:
    def __init__(self, path: str, root: Any) -> None:
        self.path = path
        self.resolver = _Resolver(_collect_constants(root, "typescript"))
        self.out = _FileRoutes()
        self.root = root
        self.loops: _Loops = None  # enclosing for-of loops of the node being visited

    def _unresolved(self, node: Any, reason: str, method: str | None = None) -> None:
        self.out.unresolved.append(
            UnresolvedRoute(
                file=self.path,
                line=node.start_point[0] + 1,
                expression=_raw(node),
                reason=reason,
                method=method,
            )
        )

    def _path(self, node: Any) -> tuple[str, bool] | None:
        rendered = _render(self.resolver.parts(node), http=False)
        if rendered is None or rendered.base is not None:
            return None
        return _route_path(rendered.path)

    def _path_shaped(self, node: Any) -> bool:
        rendered = _render(self.resolver.parts(node), http=False)
        return rendered is not None and (rendered.base is not None or rendered.path.startswith("/"))

    def _methods(self, call: Any, node: Any) -> list[str] | None:
        value = _literal(node)
        if value is not None:
            return [value.upper()] if value.upper() in _METHODS else None
        if node.type == "identifier":
            values = _for_of_values(self.loops, _short_text(node))
            if values and all(v.upper() in _METHODS for v in values):
                return [v.upper() for v in values]
        if node.type == "array":
            values = [_literal(n) for n in node.named_children]
            if values and all(v and v.upper() in _METHODS for v in values):
                return [v.upper() for v in values if v]
        return None

    def _add(self, call, owner, methods, path_node, framework) -> None:
        resolved = self._path(path_node)
        if resolved is None:
            self._unresolved(path_node, "dynamic route path", ",".join(methods) or None)
            return
        path, wildcard = resolved
        for method in methods:
            self.out.pending.append(
                _Pending(
                    owner=owner,
                    method=method,
                    path=path,
                    wildcard=wildcard,
                    line=call.start_point[0] + 1,
                    framework=framework,
                    raw=_raw(path_node),
                    handler=_handler(call, path_node),
                )
            )

    def visit_call(self, call: Any) -> None:
        name, receiver, receiver_node = _callee(call)
        if not name or receiver is None:
            return
        positional, _ = _args(call)
        token = _receiver_token(receiver)

        # router.add('GET', '/x', h) / app.on('GET', '/x', h)
        if name in _METHOD_FIRST and len(positional) >= 3:
            methods = self._methods(call, positional[0])
            if methods is not None:
                self._add(call, receiver, methods, positional[1], "method-first")
            elif _literal(positional[0]) is None and self._path_shaped(positional[1]):
                self._unresolved(positional[0], "dynamic route method")
            return

        # router.route('/x').get(h).post(h2): the path lives on the innermost .route()
        if name.lower() in _VERB_METHODS:
            inner = receiver_node
            hops = 0  # bounded: every call of an N-long chain walking it was O(N^2)
            while inner is not None and inner.type == "call_expression" and hops < _MAX_CHAIN:
                hops += 1
                inner_name, inner_receiver, inner_node = _callee(inner)
                inner_args, _ = _args(inner)
                if inner_name == "route" and inner_receiver and len(inner_args) == 1:
                    self._add(call, inner_receiver, [name.upper()], inner_args[0], "express")
                    return
                if inner_name.lower() not in _VERB_METHODS:
                    break
                inner = inner_node

        # app.get('/x', h)
        if (
            name.lower() in _VERB_METHODS
            and (_REGISTRAR.match(token) or receiver in self.out.routers)
            and len(positional) >= 2
            and self._path_shaped(positional[0])
        ):
            self._add(call, receiver, [name.upper()], positional[0], "express")
            return

        # fastify.route({ method, url, handler })
        if name == "route" and len(positional) == 1 and positional[0].type == "object":
            props = _obj_props(positional[0])
            if "url" in props and "method" in props:
                methods = self._methods(call, props["method"])
                if methods is None:
                    self._unresolved(props["method"], "dynamic route method")
                    return
                self._add(call, receiver, methods, props["url"], "express")
            return

        # app.use('/api/x', router) / app.route('/api', sub)
        if name in ("use", "route") and len(positional) >= 2:
            target = positional[-1]
            if target.type not in ("identifier", "member_expression"):
                return
            if not self._path_shaped(positional[0]) and positional[0].type not in (
                "identifier",
                "member_expression",
                "template_string",
                "binary_expression",
            ):
                return
            resolved = self._path(positional[0])
            self.out.mounts.append(
                _Mount(
                    owner=receiver,
                    prefix=resolved[0] if resolved else None,
                    target=_short_text(target),
                    line=call.start_point[0] + 1,
                    raw=_raw(positional[0]),
                )
            )

    def collect_module_shape(self) -> None:
        for node, parent, grand, _ in _walk_ctx(self.root):
            t = node.type
            if t == "import_statement":
                source = node.child_by_field_name("source")
                spec = _literal(source) if source is not None else None
                if spec is None:
                    continue
                for clause in node.named_children:
                    if clause.type != "import_clause":
                        continue
                    for child in clause.named_children:
                        if child.type == "identifier":
                            self.out.imports[_short_text(child)] = (spec, "default")
                        elif child.type == "named_imports":
                            for item in child.named_children:
                                if item.type != "import_specifier":
                                    continue
                                imported = _short_text(item.child_by_field_name("name"))
                                alias = item.child_by_field_name("alias")
                                self.out.imports[_short_text(alias) if alias else imported] = (
                                    spec,
                                    imported,
                                )
            elif t == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if name_node is None or value is None or name_node.type != "identifier":
                    continue
                local = _short_text(name_node)
                callee = value
                if callee.type == "new_expression":
                    callee = callee.child_by_field_name("constructor")
                elif callee.type == "call_expression":
                    fn = callee.child_by_field_name("function")
                    if _short_text(fn) == "require":
                        args, _ = _args(callee)
                        spec = _literal(args[0]) if args else None
                        if spec is not None:
                            self.out.imports[local] = (spec, "default")
                        continue
                    callee = fn
                if callee is not None and _ROUTER_FACTORY.search(_short_text(callee)):
                    self.out.routers.add(local)
                if parent is not None and grand is not None and grand.type == "export_statement":
                    self.out.exports[local] = local
            elif t == "export_statement":
                value = node.child_by_field_name("value")
                if value is not None and value.type == "identifier":
                    self.out.exports["default"] = _short_text(value)
                for child in node.named_children:
                    if child.type == "export_clause":
                        for spec in child.named_children:
                            if spec.type != "export_specifier":
                                continue
                            local = _short_text(spec.child_by_field_name("name"))
                            alias = spec.child_by_field_name("alias")
                            self.out.exports[_short_text(alias) if alias else local] = local
            elif t == "assignment_expression" and _short_text(node.child_by_field_name("left")) in (
                "module.exports",
                "exports.default",
            ):
                right = node.child_by_field_name("right")
                if right is not None and right.type == "identifier":
                    self.out.exports["default"] = _short_text(right)

    def visit_pathname_check(self, node: Any) -> None:
        """``url.pathname === '/health'`` / ``url.pathname.startsWith('/api/x/')``.

        A request dispatcher that answers a path before (or instead of) a router.
        The method is not part of the check, so the route is ``ALL``.
        """
        if node.type == "binary_expression":
            if _short_text(node.child_by_field_name("operator")) not in ("===", "=="):
                return
            sides = [node.child_by_field_name("left"), node.child_by_field_name("right")]
            subject = next((n for n in sides if n is not None and _is_pathname(n)), None)
            other = next((n for n in sides if n is not None and n is not subject), None)
            if subject is None or other is None or _literal(other) is None:
                return
            self._add(node, "", ["ALL"], other, "pathname")
            return
        name, receiver, receiver_node = _callee(node)
        if name == "startsWith" and receiver_node is not None and _is_pathname(receiver_node):
            positional, _ = _args(node)
            if positional and _literal(positional[0]) is not None:
                literal = _literal(positional[0]) or ""
                resolved = _route_path(literal)
                if resolved is None:
                    return
                self.out.pending.append(
                    _Pending(
                        owner="",
                        method="ALL",
                        path=resolved[0],
                        wildcard=True,
                        line=node.start_point[0] + 1,
                        framework="pathname",
                        raw=_raw(positional[0]),
                    )
                )

    def run(self) -> _FileRoutes:
        self.collect_module_shape()
        for node, _, _, loops in _walk_ctx(self.root):
            if node.has_error:
                continue
            self.loops = loops
            if node.type == "call_expression":
                self.visit_call(node)
                self.visit_pathname_check(node)
            elif node.type == "binary_expression":
                self.visit_pathname_check(node)
        self.out.owners = {p.owner for p in self.out.pending}
        return self.out


def _resolve_spec(importer: str, spec: str, files: set[str]) -> str | None:
    if not spec.startswith("."):
        return None
    base = PurePosixPath(importer).parent / spec
    parts: list[str] = []
    for part in base.parts:
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part != ".":
            parts.append(part)
    stem = "/".join(parts)
    stem_no_ext = re.sub(r"\.(js|mjs|cjs|ts)$", "", stem)
    for candidate in (
        stem,
        *(stem_no_ext + ext for ext in TS_ROUTE_EXTENSIONS),
        *(f"{stem}/index{ext}" for ext in TS_ROUTE_EXTENSIONS),
    ):
        if candidate in files:
            return candidate
    return None


def parse_ts_routes(files: dict[str, str], max_routes: int | None = None) -> TsRouteResult:
    """Every provider route registered in *files* (``{repo-relative path: source}``).

    With *max_routes*, stops (``truncated=True``) once more than that many routes
    are listed: k routers mounted at up to 64 prefixes each multiply out.
    """
    result = TsRouteResult()
    shapes: dict[str, _FileRoutes] = {}
    for path in sorted(files):
        content = files[path].encode("utf-8")
        tree = _parse(_grammar(path), content)
        if tree is None:
            result.parse_errors.append(f"{path}: grammar not available")
            continue
        token = _SOURCE.set(content)
        try:
            shapes[path] = _Collector(path, tree.root_node).run()
        except Exception as exc:  # one hostile file must not fail the whole scan
            result.parse_errors.append(f"{path}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        finally:
            _SOURCE.reset(token)
        result.unresolved.extend(shapes[path].unresolved)

    names = set(shapes)
    # Mount edges: (file, owner) -> (file, mounted local) with its prefix.
    edges: dict[tuple[str, str], list[tuple[tuple[str, str], str | None, _Mount]]] = {}
    for path, shape in shapes.items():
        for mount in shape.mounts:
            target = _resolve_target(path, mount.target, shape, shapes, names)
            if target is None:
                result.unresolved.append(
                    UnresolvedRoute(
                        file=path,
                        line=mount.line,
                        expression=f"{mount.raw} -> {mount.target}",
                        reason="mount target is not a router in the scanned files",
                    )
                )
                continue
            edges.setdefault(target, []).append(((path, mount.owner), mount.prefix, mount))

    prefixes_of = MountPrefixes(edges)
    for path, shape in shapes.items():
        if result.truncated:
            break
        for route in shape.pending:
            if max_routes is not None and len(result.routes) > max_routes:
                result.truncated = True
                break
            node = (path, route.owner)
            orphan = (
                node not in edges
                and route.owner in shape.routers
                and route.owner in shape.exports.values()
            )
            if orphan:
                result.unresolved.append(
                    UnresolvedRoute(
                        file=path,
                        line=route.line,
                        expression=route.raw,
                        reason="exported router whose mount point is not in the scanned files",
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
                            expression=route.raw,
                            reason=prefix.describe("mounted"),
                            method=route.method,
                        )
                    )
                    continue
                full = normalize_path(prefix + route.path) or "/"
                result.routes.append(
                    TsRoute(
                        method=route.method,
                        path=full,
                        file=path,
                        line=route.line,
                        framework=route.framework,
                        wildcard=route.wildcard,
                        raw=route.raw,
                        handler=route.handler,
                        mount_prefix=prefix or None,
                    )
                )
    result.routes.sort(key=lambda r: (r.file, r.line, r.method, r.path))
    result.unresolved.sort(key=lambda u: (u.file, u.line))
    return result


@dataclass(frozen=True)
class Unmountable:
    """A mount chain that has no static prefix; ``reason`` says why."""

    reason: str  # dynamic | cycle | depth | count | complexity

    def describe(self, verb: str) -> str:
        return {
            "dynamic": f"{verb} under a dynamic prefix",
            "cycle": f"{verb} under a cyclic mount (a router mounted inside itself)",
            "depth": f"{verb} under a mount chain deeper than {_MAX_MOUNT_DEPTH}",
            "count": f"{verb} under more than {_MAX_PREFIXES} distinct prefixes (rest not listed)",
            "complexity": f"{verb} under a mount graph too complex to resolve",
        }[self.reason]


DYNAMIC = Unmountable("dynamic")
_CYCLE = Unmountable("cycle")
_TOO_DEEP = Unmountable("depth")
_TOO_MANY = Unmountable("count")
_TOO_COMPLEX = Unmountable("complexity")
_MAX_MOUNT_DEPTH = 10
_MAX_PREFIXES = 64
#: Uncached walk steps per route (only cycle-affected subgraphs are uncached).
_MAX_MOUNT_STEPS = 10_000

Prefix = str | Unmountable


class MountPrefixes:
    """Every full mount prefix of a router: ``""`` when unmounted, :class:`Unmountable` else.

    *edges* maps a mounted router to ``[(parent router, prefix or None, mount)]``.
    Each router's prefixes are computed once and memoised (unless the result
    depended on a cycle cut on the current path), a router already on the current
    path is a CYCLE, a chain past :data:`_MAX_MOUNT_DEPTH` is too deep, and at most
    :data:`_MAX_PREFIXES` distinct prefixes are listed per router. The old
    recursive enumeration re-walked every path: k self-mounts cost k^11 steps.
    """

    def __init__(self, edges: dict[Any, list[tuple[Any, str | None, Any]]]) -> None:
        self.edges = edges
        self._memo: dict[Any, list[Prefix]] = {}
        self._steps = 0

    def __call__(self, node: Any) -> list[Prefix]:
        self._steps = 0
        return list(self._walk(node, set(), 0))

    def _walk(self, cur: Any, on_path: set[Any], depth: int) -> list[Prefix]:
        if cur in self._memo:
            return self._memo[cur]
        incoming = self.edges.get(cur)
        if not incoming:
            return [""]
        # The three cuts below depend on the path taken here: never memoised.
        if cur in on_path:
            return [_CYCLE]
        if depth > _MAX_MOUNT_DEPTH:
            return [_TOO_DEEP]
        self._steps += 1
        if self._steps > _MAX_MOUNT_STEPS:
            return [_TOO_COMPLEX]
        on_path.add(cur)
        out: dict[Prefix, None] = {}
        clean = True
        for parent, prefix, _ in incoming:
            uppers = self._walk(parent, on_path, depth + 1)
            if parent not in self._memo and self.edges.get(parent):
                clean = False  # the parent's answer was a path-dependent cut
            for upper in uppers:
                if isinstance(upper, Unmountable):
                    out[upper] = None
                elif prefix is None:
                    out[DYNAMIC] = None
                else:
                    out[upper + prefix] = None
                if len(out) >= _MAX_PREFIXES:
                    out[_TOO_MANY] = None
                    break
            if _TOO_MANY in out:
                break
        on_path.discard(cur)
        result = list(out)
        if clean:
            self._memo[cur] = result
        return result


def mount_prefixes(edges: dict[Any, list[tuple[Any, str | None, Any]]], node: Any) -> list[Prefix]:
    """Every full mount prefix of *node*; see :class:`MountPrefixes`."""
    return MountPrefixes(edges)(node)


def _handler(call: Any, path_node: Any) -> str | None:
    """The last argument after the path when it names a function (not an inline one)."""
    positional, _ = _args(call)
    after = [a for a in positional if a.start_byte > path_node.end_byte]
    if after and after[-1].type in ("identifier", "member_expression"):
        return _short_text(after[-1])
    return None


def _resolve_target(
    path: str, local: str, shape: _FileRoutes, shapes: dict[str, _FileRoutes], names: set[str]
) -> tuple[str, str] | None:
    """The (file, local) a mounted identifier refers to, through one relative import."""
    if local in shape.imports:
        spec, imported = shape.imports[local]
        target_file = _resolve_spec(path, spec, names)
        if target_file is None:
            return None
        target_local = shapes[target_file].exports.get(imported)
        return (target_file, target_local) if target_local else None
    if local in shape.routers or local in shape.owners:
        return (path, local)
    return None


def _match(
    method: str | None, target: str, route: TsRoute | RouteDefinition
) -> tuple[int, bool] | None:
    """``(score, weak)`` of a consumer hit on *route*; None = no hit. See :func:`match_score`."""
    route_method = route.method.upper()
    if method is not None and route_method != "ALL" and route_method != method.upper():
        return None
    route_path = normalize_path(route.path) or "/"
    a = target.split("/")
    b = route_path.split("/")
    if getattr(route, "wildcard", False):
        if not (target == route_path or target.startswith(route_path.rstrip("/") + "/")):
            return None
        return 2 * len(b), False
    if len(a) != len(b):
        return None
    score = 0
    weak = False
    for consumer_seg, route_seg in zip(a, b, strict=True):
        if consumer_seg == route_seg:
            score += 2
        elif route_seg == "{param}":
            score += 1
        elif consumer_seg == "{param}":
            weak = True  # only a hit if the runtime value equals the route's literal
        else:
            return None
    return score, weak


def match_score(method: str | None, target: str, route: TsRoute | RouteDefinition) -> int | None:
    """How well a consumer ``(method, target)`` hits a provider route; None = no hit.

    Per segment: equal text scores 2, a route ``{param}`` against a consumer
    literal scores 1, a consumer ``{param}`` against a route literal scores 0 (it
    can only be that route if the runtime value happens to equal the literal). A
    wildcard route scores its prefix's segments. ``ALL`` and an unknown consumer
    method match any method. Pick the highest score among a consumer's hits.
    """
    hit = _match(method, target, route)
    return None if hit is None else hit[0]


def route_matches(method: str | None, target: str, route: TsRoute | RouteDefinition) -> bool:
    """Does a consumer ``(method, target)`` hit this provider route at all?"""
    return _match(method, target, route) is not None


@dataclass
class RouteMatch:
    """The join of one consumer call site against the provider routes.

    ``route`` is the highest-scoring hit (the first in route order on a tie).
    ``candidates`` holds every route tied at that score, in route order, and
    ``ambiguous`` is True when there is more than one: the call site cannot be
    attributed to a single route, so do not treat ``route`` as THE provider.
    ``weak`` is True when the hit needed a consumer ``{param}`` to stand for a
    route literal (``/api/assets/{param}`` against ``/api/assets/_assignments``):
    it is that route only if the runtime value equals the literal.
    """

    route: TsRoute
    score: int
    weak: bool = False
    ambiguous: bool = False
    candidates: list[TsRoute] = field(default_factory=list)


def best_route(method: str | None, target: str, routes: list[TsRoute]) -> RouteMatch | None:
    """The provider route a consumer call most specifically hits, if any, with tie/weak flags."""
    hits: list[tuple[TsRoute, int, bool]] = []
    for route in routes:
        hit = _match(method, target, route)
        if hit is not None:
            hits.append((route, hit[0], hit[1]))
    if not hits:
        return None
    top = max(score for _, score, _ in hits)
    tied = [(route, weak) for route, score, weak in hits if score == top]
    route, weak = tied[0]
    return RouteMatch(
        route=route,
        score=top,
        weak=weak,
        ambiguous=len(tied) > 1,
        candidates=[r for r, _ in tied],
    )
