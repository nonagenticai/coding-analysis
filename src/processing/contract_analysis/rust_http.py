"""Rust HTTP facts: reqwest call sites (consumer side) and axum routes (provider side).

Deterministic tree-sitter extraction over tenant source, same rules as the
Python/TypeScript extractors: what resolves statically is reported, anything
else is listed as unresolved with a reason, never guessed.

Call sites (reqwest)
    ``client.get(url)`` / ``.post`` / ``.put`` / ``.patch`` / ``.delete`` /
    ``.head``, ``client.request(Method::POST, url)`` and ``reqwest::get(url)``.
    A verb method counts when the receiver is a known reqwest client (named like
    one: ``client``, ``http``, ``self.http_client``; a ``let`` bound to
    ``Client::new()``/``Client::builder()..build()``; a parameter typed
    ``Client``; or the constructor expression itself) or the target is URL/path
    shaped. A call on a known client whose target cannot be read is unresolved,
    never dropped.
    Targets: string literals, ``format!`` (``{}``/``{0}``/``{name}`` holes, named
    arguments), ``const``/``static`` strings, a ``let`` bound once in the same
    function, ``&x``/``x.as_str()``/``x.clone()`` and ``base.join("/x")``. A
    leading hole followed by ``/`` is the base URL (``base``), exactly as for
    Python ``f"{B_URL}/items"``. A hole in the URL authority
    (``format!("http://{}/x", host)``) is a base expression too, never a literal
    host.

Routes (axum)
    ``Router::new().route("/p", get(h).post(h2))``, ``.nest("/api", x)``,
    ``.merge(x)``, ``.route_service``/``.nest_service``, where ``x`` is a call to a
    router-building function in the scanned files (any file) or a ``let`` bound in
    the same function. Prefixes from every ``nest`` chain from a root router
    (one that no other router nests or merges) are applied. ``:id`` and ``{id}``
    are path params; ``*rest``/``{*rest}`` make the route a wildcard. Router
    functions are identified by file + name: same-named ``routes()`` in sibling
    modules are told apart by the call's module qualifier or the caller's ``use``.

Test code (files under a test directory, ``#[test]`` functions, ``#[cfg(test)]``
items) is not scanned for either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from processing.structure.rust_facts import outer_attributes
from processing.structure.text import node_text as _t
from processing.symbols import _parse

from .call_sites import is_test_path
from .checks.types import normalize_path
from .parsers.ts_route_parser import TsRoute, UnresolvedRoute

_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
_HTTP_RECEIVER = re.compile(r"(?i)(^|_)(client|http|https|api|reqwest|session)$|client$|http$")
_NOT_HTTP_RECEIVER = re.compile(
    r"(?i)map|cache|store|env|config|settings|headers|params|query|cookies|registry|storage"
    r"|router|^app$|vec|list|set|args|row|json|value|state|extensions"
)
_METHOD_ROUTERS = {*_VERBS, "trace", "connect", "any"}
_ROUTER_METHODS = {"route", "nest", "merge", "route_service", "nest_service"}
_PASSTHROUGH_METHODS = {"layer", "with_state", "fallback", "route_layer", "into_make_service"}
_URLISH = re.compile(r"^(?:https?|wss?)://([^/?#]*)(.*)$", re.I)
_FMT_HOLE = re.compile(r"\{([^{}:]*)(?::[^{}]*)?\}")
_MAX_DEPTH = 6
_MAX_NEST_DEPTH = 32
_RAW = 200


def _line(node: Any) -> int:
    return node.start_point[0] + 1


def _walk(node: Any):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.named_children))


def string_value(node: Any) -> str | None:
    """Value of a Rust string literal (``"..."`` or ``r#"..."#``), else None."""
    if node is None:
        return None
    if node.type == "string_literal":
        out = []
        for child in node.named_children:
            if child.type == "string_content":
                out.append(_t(child))
            elif child.type == "escape_sequence":
                esc = _t(child)
                out.append({"\\n": "\n", "\\t": "\t", '\\"': '"', "\\\\": "\\"}.get(esc, esc))
        return "".join(out)
    if node.type == "raw_string_literal":
        text = _t(node)
        m = re.match(r'^r(#*)"(.*)"\1$', text, re.S)
        return m.group(2) if m else None
    return None


def token_args(token_tree: Any) -> list[str]:
    """Top-level comma-separated arguments of a macro token tree, as text."""
    if token_tree is None:
        return []
    text = _t(token_tree)
    if len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        text = text[1:-1]
    args, depth, cur, quote = [], 0, [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        args.append("".join(cur).strip())
    return args


def macro_name(node: Any) -> str:
    return _t(node.child_by_field_name("macro")).rsplit("::", 1)[-1]


def macro_token_tree(node: Any) -> Any:
    for child in node.named_children:
        if child.type == "token_tree":
            return child
    return None


def first_string_in(token_tree: Any) -> tuple[str, Any] | None:
    if token_tree is None:
        return None
    for child in token_tree.named_children:
        value = string_value(child)
        if value is not None:
            return value, child
    return None


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


@dataclass
class _Hole:
    expr: str


Parts = list[str | _Hole]


@dataclass
class _Scope:
    consts: dict[str, Any]  # file-level const/static -> value node
    lets: dict[str, list[Any]] = field(default_factory=dict)  # fn-level let -> value nodes


class _UnresolvableError(Exception):
    pass


def _resolve(node: Any, scope: _Scope, depth: int = 0) -> Parts:
    if node is None or depth > _MAX_DEPTH:
        raise _UnresolvableError("too_complex")
    value = string_value(node)
    if value is not None:
        return [value]
    kind = node.type
    if kind in ("reference_expression", "parenthesized_expression"):
        inner = node.child_by_field_name("value") or (
            node.named_children[-1] if node.named_children else None
        )
        return _resolve(inner, scope, depth + 1)
    if kind == "identifier":
        name = _t(node)
        bound = scope.lets.get(name)
        if bound is not None:
            if len(bound) == 1:
                parts = _resolve(bound[0], scope, depth + 1)
                # A let bound to a call result keeps its NAME as the hole: the base
                # source is classified from the binding (``let base = env::var(..)``).
                return [_Hole(name)] if len(parts) == 1 and isinstance(parts[0], _Hole) else parts
            return [_Hole(name)]
        if name in scope.consts:
            return _resolve(scope.consts[name], scope, depth + 1)
        return [_Hole(name)]
    if kind == "macro_invocation" and macro_name(node) == "format":
        return _format_parts(node, scope, depth)
    if kind == "call_expression":
        fn = node.child_by_field_name("function")
        args = node.child_by_field_name("arguments")
        if fn is not None and fn.type == "field_expression":
            method = _t(fn.child_by_field_name("field"))
            recv = fn.child_by_field_name("value")
            arg_nodes = list(args.named_children) if args is not None else []
            if method in ("as_str", "clone", "to_string", "to_owned", "into", "as_ref"):
                return _resolve(recv, scope, depth + 1)
            if method == "join" and len(arg_nodes) == 1:
                return [
                    *_resolve(recv, scope, depth + 1),
                    *_resolve(arg_nodes[0], scope, depth + 1),
                ]
        return [_Hole(_t(node)[:_RAW])]
    if kind in ("field_expression", "scoped_identifier"):
        return [_Hole(_t(node)[:_RAW])]
    raise _UnresolvableError("dynamic target")


def _format_parts(node: Any, scope: _Scope, depth: int) -> Parts:
    args = token_args(macro_token_tree(node))
    if not args:
        raise _UnresolvableError("empty format!")
    fmt = re.match(r'^r?#*"(.*)"#*$', args[0], re.S)
    if fmt is None:
        raise _UnresolvableError("format! string is not a literal")
    template = fmt.group(1)
    positional = [a for a in args[1:] if not re.match(r"^[A-Za-z_]\w*\s*=[^=]", a)]
    named = {
        m.group(1): m.group(2).strip()
        for a in args[1:]
        if (m := re.match(r"^([A-Za-z_]\w*)\s*=([^=].*)$", a, re.S))
    }
    parts: Parts = []
    pos = 0
    last = 0
    # ``{{``/``}}`` collapse to one sentinel char each; every offset below is into
    # the collapsed text, never the original (whose offsets are shifted by them).
    template = template.replace("{{", "\0").replace("}}", "\1")
    for m in _FMT_HOLE.finditer(template):
        lit = template[last : m.start()]
        if lit:
            parts.append(lit)
        key = m.group(1).strip()
        if key == "":
            expr = positional[pos] if pos < len(positional) else "?"
            pos += 1
        elif key.isdigit():
            idx = int(key)
            expr = positional[idx] if idx < len(positional) else "?"
        else:
            expr = named.get(key, key)
        parts.extend(_expr_parts(expr, scope, depth))
        last = m.end()
    if template[last:]:
        parts.append(template[last:])
    return [p.replace("\0", "{").replace("\1", "}") if isinstance(p, str) else p for p in parts]


def _expr_parts(expr: str, scope: _Scope, depth: int) -> Parts:
    """A format argument given as text: inline an identifier's single binding/const."""
    name = expr.strip().removeprefix("&")
    if re.match(r"^[A-Za-z_]\w*$", name):
        bound = scope.lets.get(name)
        if bound is not None and len(bound) == 1:
            try:
                parts = _resolve(bound[0], scope, depth + 1)
            except _UnresolvableError:
                return [_Hole(name)]
            return [_Hole(name)] if len(parts) == 1 and isinstance(parts[0], _Hole) else parts
        if bound is None and name in scope.consts:
            try:
                return _resolve(scope.consts[name], scope, depth + 1)
            except _UnresolvableError:
                return [_Hole(name)]
    return [_Hole(name[:_RAW])]


@dataclass
class RenderedTarget:
    path: str
    base: str | None
    host: str | None
    template: bool


def render(parts: Parts) -> RenderedTarget | None:
    """Parts -> joinable path, or None when the target is not URL/path shaped."""
    base = None
    host = None
    if parts and isinstance(parts[0], _Hole):
        if len(parts) > 1 and isinstance(parts[1], str) and parts[1].startswith("/"):
            base = parts[0].expr
            parts = parts[1:]
        else:
            return None
    text_parts: list[str] = []
    holes: list[str] = []
    template = False
    for part in parts:
        if isinstance(part, _Hole):
            text_parts.append("{param}")
            holes.append(part.expr)
            template = True
        else:
            text_parts.append(part)
    text = "".join(text_parts)
    m = _URLISH.match(text)
    if m:
        host = m.group(1) or None
        text = m.group(2) or "/"
        in_host = host.count("{param}") if host else 0
        if in_host:
            # ``format!("http://{}/x", host)``: the authority is computed, so it is a
            # base expression to classify (env/config/unknown), never a literal host.
            pieces = (host or "").split("{param}")
            if pieces == ["", ""]:
                base = holes[0]
            else:
                base = pieces[0] + "".join(
                    "{" + expr + "}" + rest for expr, rest in zip(holes, pieces[1:], strict=False)
                )
            host = None
            template = len(holes) > in_host
    if not text.startswith("/"):
        return None
    text = text.split("?", 1)[0].split("#", 1)[0]
    if "{param}" in text.split("?", 1)[0]:
        template = True
    return RenderedTarget(normalize_path(text) or "/", base, host, template)


# ---------------------------------------------------------------------------
# File-level scaffolding: consts and per-function lets
# ---------------------------------------------------------------------------


def file_consts(root: Any) -> dict[str, Any]:
    consts: dict[str, Any] = {}
    seen: dict[str, int] = {}
    for node in _walk(root):
        if node.type in ("const_item", "static_item"):
            name = _t(node.child_by_field_name("name"))
            seen[name] = seen.get(name, 0) + 1
            value = node.child_by_field_name("value")
            if value is not None:
                consts[name] = value
    return {k: v for k, v in consts.items() if seen.get(k) == 1}


def function_lets(fn_body: Any) -> dict[str, list[Any]]:
    lets: dict[str, list[Any]] = {}
    for node in _walk(fn_body):
        if node.type == "let_declaration":
            pattern = node.child_by_field_name("pattern")
            value = node.child_by_field_name("value")
            if pattern is not None and pattern.type == "identifier" and value is not None:
                lets.setdefault(_t(pattern), []).append(value)
    return lets


def cfg_requires_test(predicate: str) -> bool:
    """``cfg(test)`` or ``cfg(all(test, ..))``: ``test`` as a top-level argument of
    ``all``. Any other predicate (``any(..)``, ``not(..)``, nested ``all``) is not
    read as test-only."""
    pred = re.sub(r"\s+", "", predicate)
    if pred == "test":
        return True
    if not (pred.startswith("all(") and pred.endswith(")")):
        return False
    return "test" in pred[4:-1].split(",")


def is_test_attribute(attr_item: Any) -> bool:
    """``#[test]`` / ``#[tokio::test]`` / ``#[x::test(..)]`` or a ``#[cfg(..)]`` that
    requires a test build (``cfg(test)``, ``cfg(all(test, ..))``)."""
    attr = next((c for c in attr_item.named_children if c.type == "attribute"), None)
    if attr is None or not attr.named_children:
        return False
    path = re.sub(r"\s+", "", _t(attr.named_children[0]))
    if path == "test" or path.endswith("::test"):
        return True
    if path != "cfg":
        return False
    args = attr.child_by_field_name("arguments")
    text = _t(args) if args is not None else ""
    return text.startswith("(") and text.endswith(")") and cfg_requires_test(text[1:-1])


def is_test_item(attrs: list[Any]) -> bool:
    """An item whose outer attributes (``outer_attributes``) mark it test-only."""
    return any(is_test_attribute(a) for a in attrs)


def function_items(root: Any, include_tests: bool = True) -> list[tuple[str, Any, list[Any]]]:
    """Every ``fn`` with a body: (qualified-ish name, node, outer attribute items).

    Impl fns are ``Type::fn``. With ``include_tests=False``, ``#[test]``
    functions and everything under a ``cfg(test)``-only item are skipped: they
    are not part of the shipped service. Attributes are collected in one forward
    pass per container (``outer_attributes``).
    """
    out: list[tuple[str, Any, list[Any]]] = []

    def visit(node: Any, owner: str | None) -> None:
        attrs_of = outer_attributes(node)
        for child in node.named_children:
            attrs = attrs_of.get(child.id, [])
            if not include_tests and is_test_item(attrs):
                continue
            if child.type == "function_item":
                name = _t(child.child_by_field_name("name"))
                out.append((f"{owner}::{name}" if owner else name, child, attrs))
            elif child.type == "impl_item":
                body = child.child_by_field_name("body")
                if body is not None:
                    visit(body, _t(child.child_by_field_name("type")))
            elif child.type in ("mod_item", "declaration_list"):
                body = child.child_by_field_name("body") if child.type == "mod_item" else child
                if body is not None:
                    visit(body, owner)

    visit(root, None)
    return out


def functions(root: Any, include_tests: bool = True) -> list[tuple[str, Any]]:
    """``function_items`` without the attributes: (qualified-ish name, node)."""
    return [(name, node) for name, node, _ in function_items(root, include_tests)]


# ---------------------------------------------------------------------------
# reqwest call sites
# ---------------------------------------------------------------------------


_CLIENT_CTOR = re.compile(
    r"^(?:::)?(?:reqwest::)?(?:blocking::)?(?:Client|ClientBuilder)::(?:new|builder|default)\("
)
_CLIENT_TYPE = re.compile(r"^(?:&)?(?:mut)?(?:reqwest::)?(?:blocking::)?Client$")


def _is_client_ctor(node: Any, uses_reqwest: bool) -> bool:
    """``reqwest::Client::new()`` / ``Client::builder()...build()`` (``Client`` alone only
    when the file uses reqwest at all)."""
    text = re.sub(r"\s+", "", _t(node)[:400])
    m = _CLIENT_CTOR.match(text)
    if m is None:
        return False
    return uses_reqwest or text.startswith(("reqwest::", "::reqwest::"))


def _client_bindings(fn_node: Any, lets: dict[str, list[Any]], uses_reqwest: bool) -> set[str]:
    """Names that hold a reqwest client in one function: typed params and ``let`` bindings
    to a client constructor (every binding of the name must be one)."""
    names: set[str] = set()
    params = fn_node.child_by_field_name("parameters")
    for param in params.named_children if params is not None else []:
        if param.type != "parameter":
            continue
        pattern = param.child_by_field_name("pattern")
        ptype = param.child_by_field_name("type")
        if pattern is None or ptype is None:
            continue
        if _CLIENT_TYPE.match(re.sub(r"\s+", "", _t(ptype))) and (
            uses_reqwest or "reqwest" in _t(ptype)
        ):
            names.add(_t(pattern).removeprefix("mut ").strip())
    for name, values in lets.items():
        if values and all(_is_client_ctor(v, uses_reqwest) for v in values):
            names.add(name)
    return names


def _receiver_token(recv: Any) -> str:
    text = _t(recv)
    text = re.sub(r"\(.*", "", text) if len(text) < 300 else text[:300]
    return text.rsplit(".", 1)[-1].rsplit("::", 1)[-1]


def extract_rust_call_sites(path: str, source: str) -> Any:
    """reqwest call sites of one Rust file (a ``call_sites.FileResult``)."""
    from .base_source import BaseClassifier
    from .call_sites import CallSite, FileResult, UnresolvedCallSite

    content = source.encode("utf-8")
    tree = _parse("rust", content)
    if tree is None:
        return FileResult(parsed=False)
    result = FileResult()
    consts = file_consts(tree.root_node)
    uses_reqwest = "reqwest" in source
    classifier = BaseClassifier(source, "rust")
    for fn_name, fn_node in functions(tree.root_node, include_tests=False):
        body = fn_node.child_by_field_name("body")
        if body is None:
            continue
        scope = _Scope(consts=consts, lets=function_lets(body))
        clients = _client_bindings(fn_node, scope.lets, uses_reqwest)
        for node in _walk(body):
            if node.type != "call_expression":
                continue
            hit = _http_call(node, clients, uses_reqwest)
            if hit is None:
                continue
            method, target_node, client, http_receiver = hit
            try:
                rendered = render(_resolve(target_node, scope))
                reason = "target is not URL or path shaped"
            except _UnresolvableError as exc:
                rendered = None
                reason = str(exc)
            if rendered is None:
                if http_receiver:
                    result.unresolved.append(
                        UnresolvedCallSite(
                            file=path,
                            line=_line(node),
                            kind="http",
                            client=client,
                            symbol=fn_name,
                            expression=_t(target_node)[:_RAW],
                            reason=reason,
                            method=method,
                        )
                    )
                continue
            src = classifier.classify(rendered.base, rendered.host)
            result.call_sites.append(
                CallSite(
                    file=path,
                    line=_line(node),
                    kind="http",
                    target=rendered.path,
                    client=client,
                    symbol=fn_name,
                    confidence="template" if rendered.template else "exact",
                    method=method,
                    base=rendered.base,
                    host=rendered.host,
                    raw=_t(target_node)[:_RAW],
                    base_source_kind=src.kind if src else None,
                    base_source_name=src.name if src else None,
                    base_source_default=src.default if src else None,
                )
            )
    return result


def _http_call(
    node: Any, clients: frozenset[str] | set[str] = frozenset(), uses_reqwest: bool = False
) -> tuple[str | None, Any, str, bool] | None:
    """(method, target node, client label, receiver is a known HTTP client) of a reqwest call.

    The receiver is a known client when it is HTTP-named, a name bound to a reqwest
    client in this function (*clients*), or a client constructor expression itself
    (``Client::new().get(..)``). A call on a known client is always reported: as a
    call site, or as unresolved when its target cannot be read statically.
    """
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    arg_nodes = list(args.named_children) if args is not None else []
    if fn is None or not arg_nodes:
        return None
    if fn.type == "scoped_identifier":
        text = _t(fn)
        if re.match(r"^reqwest(::blocking)?::get$", text):
            return "GET", arg_nodes[0], "reqwest", True
        return None
    if fn.type != "field_expression":
        return None
    method = _t(fn.child_by_field_name("field"))
    recv = fn.child_by_field_name("value")
    known_client = recv is not None and (
        (recv.type == "identifier" and _t(recv) in clients) or _is_client_ctor(recv, uses_reqwest)
    )
    token = _receiver_token(recv)
    if not known_client and _NOT_HTTP_RECEIVER.search(token):
        return None
    http_receiver = known_client or bool(_HTTP_RECEIVER.search(token))
    if method in _VERBS and len(arg_nodes) == 1:
        if recv is not None and recv.type == "call_expression" and not http_receiver:
            return None  # get(h).post(h2): an axum method router, not a request
        if http_receiver or _path_shaped(arg_nodes[0]):
            return method.upper(), arg_nodes[0], "reqwest", http_receiver
        return None
    if method == "request" and len(arg_nodes) == 2 and http_receiver:
        verb = _t(arg_nodes[0]).rsplit("::", 1)[-1].upper()
        return (verb if verb.lower() in _VERBS else None), arg_nodes[1], "reqwest", True
    return None


def _path_shaped(node: Any) -> bool:
    value = string_value(node)
    if value is None and node.type == "macro_invocation" and macro_name(node) == "format":
        first = first_string_in(macro_token_tree(node))
        value = first[0] if first else None
        if value is not None and re.match(r"^\{[^{}]*\}/", value):
            return True
    if value is None:
        return False
    return value.startswith("/") or bool(_URLISH.match(value))


# ---------------------------------------------------------------------------
# axum routes
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    kind: str  # route | nest | merge
    line: int
    path: str | None = None  # route path / nest prefix (None = dynamic)
    raw: str = ""
    methods: list[tuple[str, str | None]] = field(default_factory=list)
    target: str | None = None  # function name a nest/merge points at
    target_expr: str = ""
    problem: str | None = None


@dataclass
class _RouterFn:
    file: str
    entries: list[_Entry]
    name: str = ""  # bare function name
    module: tuple[str, ...] = ()  # module path of its file (``src/api/users.rs`` -> api, users)
    uses: dict[str, list[str]] = field(default_factory=dict)  # imported name -> full path

    @property
    def key(self) -> str:
        return f"{self.file}::{self.name}"


@dataclass
class AxumResult:
    routes: list[TsRoute]
    unresolved: list[UnresolvedRoute]
    unparsed: dict[str, str]
    truncated: bool = False


def _axum_path(raw: str) -> tuple[str, bool]:
    """axum path -> (joinable path, wildcard)."""
    segs = []
    wildcard = False
    for seg in raw.split("/"):
        if seg.startswith("*") or seg.startswith("{*"):
            wildcard = True
            break
        if seg.startswith(":"):
            seg = "{" + seg[1:] + "}"
        segs.append(seg)
    return normalize_path("/".join(segs)) or "/", wildcard


def _method_router(node: Any) -> list[tuple[str, str | None]] | None:
    """``get(h).post(h2)`` -> [(GET, h), (POST, h2)]; None when not statically a method router."""
    out: list[tuple[str, str | None]] = []
    cur = node
    for _ in range(16):
        if cur is None or cur.type != "call_expression":
            return None
        fn = cur.child_by_field_name("function")
        args = cur.child_by_field_name("arguments")
        arg_nodes = list(args.named_children) if args is not None else []
        handler = _t(arg_nodes[-1])[:_RAW] if arg_nodes else None
        if fn is not None and fn.type == "field_expression":
            name = _t(fn.child_by_field_name("field")).removesuffix("_service")
            if name in _METHOD_ROUTERS:
                out.append(("ALL" if name == "any" else name.upper(), handler))
                cur = fn.child_by_field_name("value")
                continue
            if name in ("layer", "route_layer", "with_state"):
                cur = fn.child_by_field_name("value")
                continue
            return None
        name = _t(fn).rsplit("::", 1)[-1].removesuffix("_service") if fn is not None else ""
        if name in _METHOD_ROUTERS:
            out.append(("ALL" if name == "any" else name.upper(), handler))
            out.reverse()
            return out
        if name == "on" and len(arg_nodes) == 2:
            verb = _t(arg_nodes[0]).rsplit("::", 1)[-1].upper()
            out.append((verb if verb.lower() in _VERBS else "ALL", handler))
            out.reverse()
            return out
        return None
    return None


def _chain(node: Any) -> list[Any] | None:
    """The call nodes of a router builder chain, innermost first, or None."""
    calls: list[Any] = []
    cur = node
    while cur is not None and cur.type == "call_expression":
        fn = cur.child_by_field_name("function")
        if fn is None or fn.type != "field_expression":
            break
        name = _t(fn.child_by_field_name("field"))
        if name not in _ROUTER_METHODS and name not in _PASSTHROUGH_METHODS:
            return None
        calls.append(cur)
        cur = fn.child_by_field_name("value")
    if not any(
        _t(c.child_by_field_name("function").child_by_field_name("field")) in _ROUTER_METHODS
        for c in calls
    ):
        return None
    calls.append(cur)  # the chain's base expression
    calls.reverse()
    return calls


def _chain_tops(body: Any) -> list[tuple[Any, list[Any]]]:
    """(outermost call, ``_chain``) of every router builder chain in *body*.

    Each method-call chain (``a.b(..).c(..)``) is walked once from its head (a
    call that is not the receiver of another method call): the chain's top is
    its outermost call whose whole receiver chain is router/passthrough methods
    with at least one router method. Computing ``_chain`` from every call of a
    chain instead re-walked it per call (quadratic in the chain length).

    A chain passed inline to ``.nest(..)``/``.merge(..)`` of another chain is
    not a top: ``_entries`` mounts it under that prefix, and listing it again
    would add every one of its routes a second time WITHOUT the prefix.
    """
    receivers: set[int] = set()
    heads: list[Any] = []
    for node in _walk(body):
        if node.type != "call_expression":
            continue
        fn = node.child_by_field_name("function")
        if fn is not None and fn.type == "field_expression":
            recv = fn.child_by_field_name("value")
            if recv is not None:
                receivers.add(recv.id)
        if node.id not in receivers:
            heads.append(node)
    tops: list[tuple[Any, list[Any]]] = []
    for head in heads:
        calls: list[tuple[Any, str]] = []  # outermost first
        cur = head
        while cur is not None and cur.type == "call_expression":
            fn = cur.child_by_field_name("function")
            if fn is None or fn.type != "field_expression":
                break
            calls.append((cur, _t(fn.child_by_field_name("field"))))
            cur = fn.child_by_field_name("value")
        top = None
        has_router = False
        for i in range(len(calls) - 1, -1, -1):  # innermost outward
            name = calls[i][1]
            if name not in _ROUTER_METHODS and name not in _PASSTHROUGH_METHODS:
                break
            has_router = has_router or name in _ROUTER_METHODS
            if has_router:
                top = i
        if top is None:
            continue
        chain = [cur, *(c for c, _ in reversed(calls[top:]))]
        tops.append((calls[top][0], chain))
    inlined: set[int] = set()
    for _, chain in tops:
        for call in chain[1:]:
            name = _t(call.child_by_field_name("function").child_by_field_name("field"))
            if name not in ("nest", "merge"):
                continue
            args = call.child_by_field_name("arguments")
            for arg in args.named_children if args is not None else []:
                inlined.add(arg.id)
    return [(top, chain) for top, chain in tops if top.id not in inlined]


def file_module(path: str) -> tuple[str, ...]:
    """Module path of a Rust file inside its crate: ``src/api/users/mod.rs`` -> (api, users)."""
    parts = path.split("/")
    if "src" in parts:
        parts = parts[len(parts) - parts[::-1].index("src") :]
    if not parts:
        return ()
    parts[-1] = parts[-1].removesuffix(".rs")
    if parts[-1] == "mod" or (len(parts) == 1 and parts[0] in ("lib", "main")):
        parts = parts[:-1]
    return tuple(parts)


def _file_uses(root: Any) -> dict[str, list[str]]:
    from processing.structure.rust_facts import use_paths

    out: dict[str, list[str]] = {}
    for node in root.named_children:
        if node.type != "use_declaration":
            continue
        arg = node.child_by_field_name("argument")
        alias = arg.child_by_field_name("alias") if arg is not None else None
        for segs in use_paths(arg):
            if segs and segs[-1] != "*":
                out[_t(alias) if alias is not None else segs[-1]] = segs
    return out


def _router_fns(path: str, root: Any, known_fns: set[str]) -> list[_RouterFn]:
    out: list[_RouterFn] = []
    module = file_module(path)
    uses = _file_uses(root)
    for fn_name, fn_node in functions(root, include_tests=False):
        body = fn_node.child_by_field_name("body")
        if body is None:
            continue
        lets = function_lets(body)
        let_of: dict[int, str] = {}
        for name, values in lets.items():
            for value in values:
                let_of[value.id] = name
        tops = _chain_tops(body)
        consumed: set[str] = set()
        for _, calls in tops:
            base = calls[0]
            if base is not None and base.type == "identifier":
                consumed.add(_t(base))
            for call in calls[1:]:
                args = call.child_by_field_name("arguments")
                for arg in args.named_children if args is not None else []:
                    if arg.type == "identifier":
                        consumed.add(_t(arg))
        entries: list[_Entry] = []
        for top, calls in tops:
            if let_of.get(top.id) in consumed:
                continue
            entries.extend(_entries(calls, lets, known_fns, set()))
        if entries:
            out.append(
                _RouterFn(
                    file=path,
                    entries=entries,
                    name=fn_name.rsplit("::", 1)[-1],
                    module=module,
                    uses=uses,
                )
            )
    return out


def _entries(
    calls: list[Any], lets: dict[str, list[Any]], known_fns: set[str], seen: set[str]
) -> list[_Entry]:
    entries: list[_Entry] = []
    base = calls[0]
    if base is not None and base.type == "identifier" and _t(base) not in seen:
        entries.extend(_inline(_t(base), lets, known_fns, seen))
    elif base is not None and base.type == "call_expression":
        target = _fn_target(base, known_fns)
        if target is not None:
            entries.append(_Entry(kind="merge", line=_line(base), path="", target=target))
    for call in calls[1:]:
        fn = call.child_by_field_name("function")
        name = _t(fn.child_by_field_name("field"))
        args = call.child_by_field_name("arguments")
        arg_nodes = list(args.named_children) if args is not None else []
        line = _line(call)
        if name in ("route", "route_service") and len(arg_nodes) == 2:
            raw = string_value(arg_nodes[0])
            methods: list[tuple[str, str | None]] | None = (
                [("ALL", _t(arg_nodes[1])[:_RAW])]
                if name == "route_service"
                else _method_router(arg_nodes[1])
            )
            entries.append(
                _Entry(
                    kind="route",
                    line=line,
                    path=raw,
                    raw=raw if raw is not None else _t(arg_nodes[0])[:_RAW],
                    methods=methods or [],
                    problem=(
                        "dynamic route path"
                        if raw is None
                        else None
                        if methods
                        else "method router is not statically resolvable"
                    ),
                    target_expr=_t(arg_nodes[1])[:_RAW],
                )
            )
        elif name in ("nest", "nest_service") and len(arg_nodes) == 2:
            prefix = string_value(arg_nodes[0])
            entries.extend(
                _mount("nest", line, prefix, arg_nodes[1], lets, known_fns, seen, service=name)
            )
        elif name == "merge" and len(arg_nodes) == 1:
            entries.extend(_mount("merge", line, "", arg_nodes[0], lets, known_fns, seen))
    return entries


def _fn_target(node: Any, known_fns: set[str]) -> str | None:
    """The called path (``routes`` / ``users::routes``) when it names a known fn."""
    if node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    if fn is None or fn.type not in ("identifier", "scoped_identifier"):
        return None
    text = re.sub(r"\s+", "", _t(fn))
    return text if text.rsplit("::", 1)[-1] in known_fns else None


#: Most candidates named in an "ambiguous" reason (the rest are counted).
_MAX_LISTED = 10


class _RouterIndex:
    """Router functions by name, by (name, file) and by (name, module suffix).

    Resolving a reference used to filter every same-named function per
    reference: N same-named ``routes()`` each mounting ``routes()`` cost O(N^2).
    Every lookup here is a dict hit; an ambiguous reference names its candidate
    GROUP (a dict key), so marking "every candidate is mounted somewhere" is O(1).
    """

    def __init__(self, fns: list[_RouterFn]) -> None:
        self.groups: dict[tuple[str, ...], list[_RouterFn]] = {}
        for f in fns:
            self.groups.setdefault(("name", f.name), []).append(f)
            self.groups.setdefault(("file", f.name, f.file), []).append(f)
            for i in range(len(f.module) + 1):
                self.groups.setdefault(("mod", f.name, *f.module[i:]), []).append(f)

    def resolve(self, target: str, caller: _RouterFn) -> tuple[_RouterFn | None, tuple | None]:
        """(router fn, None) or (None, key of the tied candidate group) for a call path
        written in *caller*'s file; (None, None) when nothing matches.

        Same-named router functions in sibling modules are told apart by the call's
        module qualifier (``users::routes``), by the caller file's ``use`` of the
        name, or by living in the caller's own file; anything else stays ambiguous.
        """
        segs = [x for x in target.split("::") if x]
        name = segs[-1]
        candidates = self.groups.get(("name", name), [])
        if len(candidates) <= 1:
            return (candidates[0] if candidates else None), None
        qual = segs[:-1] or caller.uses.get(name, [])[:-1]
        qual = [x for x in qual if x not in ("crate", "self", "super")]
        key: tuple[str, ...] = ("mod", name, *qual) if qual else ("file", name, caller.file)
        hits = self.groups.get(key, [])
        if len(hits) == 1:
            return hits[0], None
        return None, key if hits else ("name", name)

    def describe(self, key: tuple) -> str:
        tied = sorted(self.groups.get(key, []), key=lambda c: c.key)
        where = ", ".join(f"{c.file}:{c.name}" for c in tied[:_MAX_LISTED])
        more = len(tied) - _MAX_LISTED
        return where + (f" and {more} more" if more > 0 else "")

    def member_of(self, f: _RouterFn) -> list[tuple[str, ...]]:
        """Every group key *f* belongs to."""
        keys: list[tuple[str, ...]] = [("name", f.name), ("file", f.name, f.file)]
        keys += [("mod", f.name, *f.module[i:]) for i in range(len(f.module) + 1)]
        return keys


def _inline(
    name: str, lets: dict[str, list[Any]], known_fns: set[str], seen: set[str]
) -> list[_Entry]:
    values = lets.get(name, [])
    if len(values) != 1:
        return []
    value = values[0]
    calls = _chain(value)
    if calls is not None:
        return _entries(calls, lets, known_fns, seen | {name})
    target = _fn_target(value, known_fns)
    if target is not None:
        return [_Entry(kind="merge", line=_line(value), path="", target=target)]
    return []


def _mount(
    kind: str,
    line: int,
    prefix: str | None,
    arg: Any,
    lets: dict[str, list[Any]],
    known_fns: set[str],
    seen: set[str],
    service: str = "",
) -> list[_Entry]:
    expr = _t(arg)[:_RAW]
    if service == "nest_service":
        return [
            _Entry(
                kind="route",
                line=line,
                path=(prefix.rstrip("/") + "/*") if prefix is not None else None,
                raw=prefix or expr,
                methods=[("ALL", expr)],
                problem=None if prefix is not None else "dynamic nest prefix",
            )
        ]
    if prefix is None:
        return [_Entry(kind=kind, line=line, target_expr=expr, problem="dynamic nest prefix")]
    target = _fn_target(arg, known_fns)
    if target is not None:
        return [_Entry(kind=kind, line=line, path=prefix, target=target, target_expr=expr)]
    calls = _chain(arg)
    if calls is None and arg.type == "identifier" and len(lets.get(_t(arg), [])) == 1:
        inner = _inline(_t(arg), lets, known_fns, seen)
        if inner:
            return [_prefixed(e, prefix) for e in inner]
    if calls is not None:
        return [_prefixed(e, prefix) for e in _entries(calls, lets, known_fns, seen)]
    return [
        _Entry(
            kind=kind,
            line=line,
            path=prefix,
            target_expr=expr,
            problem=f"{kind} target is not a router built in the scanned files",
        )
    ]


def _prefixed(entry: _Entry, prefix: str) -> _Entry:
    if entry.kind == "route" and entry.path is not None:
        entry.path = prefix.rstrip("/") + ("" if entry.path == "/" else entry.path)
        entry.raw = entry.path
    elif entry.kind in ("nest", "merge") and entry.path is not None:
        entry.path = prefix.rstrip("/") + entry.path
    return entry


def extract_axum_routes(files: dict[str, str], max_routes: int | None = None) -> AxumResult:
    """axum routes of *files* (``{path: rust source}``), nest prefixes applied across files.

    Files under a test directory, ``#[test]`` functions and ``#[cfg(test)]`` items
    are not scanned: a router built only for tests is not a provider route. With
    *max_routes*, emission stops at that many routes and ``truncated`` is set.
    """
    result = AxumResult(routes=[], unresolved=[], unparsed={})
    trees: dict[str, Any] = {}
    known: set[str] = set()
    for path in sorted(files):
        if is_test_path(path):
            continue
        tree = _parse("rust", files[path].encode("utf-8"))
        if tree is None:
            result.unparsed[path] = "grammar not available"
            continue
        trees[path] = tree
        known.update(
            name.rsplit("::", 1)[-1] for name, _ in functions(tree.root_node, include_tests=False)
        )
    router_fns: list[_RouterFn] = []
    for path, tree in trees.items():
        try:
            router_fns.extend(_router_fns(path, tree.root_node, known))
        except RecursionError as exc:
            result.unparsed[path] = f"{type(exc).__name__}"
    index = _RouterIndex(router_fns)
    targeted: set[str] = set()
    targeted_groups: set[tuple] = set()
    for rfn in router_fns:
        for entry in rfn.entries:
            if entry.target is not None:
                hit, tied = index.resolve(entry.target, rfn)
                if hit is not None:
                    targeted.add(hit.key)
                elif tied is not None:
                    targeted_groups.add(tied)
    roots = sorted(
        (
            r
            for r in router_fns
            if r.key not in targeted and targeted_groups.isdisjoint(index.member_of(r))
        ),
        key=lambda r: r.key,
    )

    def full() -> bool:
        if max_routes is not None and len(result.routes) >= max_routes:
            result.truncated = True
        return result.truncated

    def emit(rfn: _RouterFn, prefix: str, depth: int, stack: tuple[str, ...]) -> None:
        for entry in rfn.entries:
            if full():
                return
            if entry.problem is not None:
                result.unresolved.append(
                    UnresolvedRoute(
                        file=rfn.file,
                        line=entry.line,
                        expression=entry.raw or entry.target_expr,
                        reason=entry.problem,
                    )
                )
                continue
            if entry.kind == "route" and entry.path is not None:
                path, wildcard = _axum_path(
                    prefix.rstrip("/") + entry.path if prefix else entry.path
                )
                for method, handler in entry.methods:
                    if full():
                        return
                    result.routes.append(
                        TsRoute(
                            method=method,
                            path=path,
                            file=rfn.file,
                            line=entry.line,
                            framework="axum",
                            wildcard=wildcard,
                            raw=entry.raw,
                            handler=handler,
                            mount_prefix=prefix or None,
                        )
                    )
            elif entry.target is not None:
                hit, tied = index.resolve(entry.target, rfn)
                if tied is not None:
                    where = index.describe(tied)
                    result.unresolved.append(
                        UnresolvedRoute(
                            file=rfn.file,
                            line=entry.line,
                            expression=entry.target_expr or entry.target,
                            reason=f"router function reference is ambiguous: {where}",
                        )
                    )
                elif hit is None:
                    continue
                elif hit.key in stack or depth >= _MAX_NEST_DEPTH:
                    result.unresolved.append(
                        UnresolvedRoute(
                            file=rfn.file,
                            line=entry.line,
                            expression=entry.target,
                            reason="router nesting cycle or too deep",
                        )
                    )
                else:
                    emit(
                        hit,
                        prefix.rstrip("/") + (entry.path or ""),
                        depth + 1,
                        (*stack, hit.key),
                    )

    for root in roots:
        emit(root, "", 0, (root.key,))
    result.routes.sort(key=lambda r: (r.file, r.line, r.method, r.path))
    result.unresolved.sort(key=lambda u: (u.file, u.line))
    return result
