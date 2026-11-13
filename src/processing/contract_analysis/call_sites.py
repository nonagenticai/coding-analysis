"""Outbound interface call sites: the CONSUMER side of a contract.

``scan_contracts`` reads what a service PROVIDES (route decorators). This module
reads what code CALLS: HTTP client requests and message publish/subscribe calls,
so an integration survey can pair every consumer with its provider instead of
finding consumers by text search.

Deterministic tree-sitter extraction, read-only, no LLM. Rules, in order of how
much they are allowed to guess (they never are):

* A target that is a literal, an f-string/template literal, a ``+`` concatenation
  or a same-file constant (bound exactly once) is RESOLVED. Interpolations become
  ``{param}``; a leading interpolation followed by ``/`` is a base URL and is
  reported in ``base``, not in the path. The path is normalised with the SAME
  :func:`normalize_path` the provider side uses, so ``(method, target)`` joins
  route definitions directly.
* Anything else (a parameter, a call result, a variable bound twice) is
  UNRESOLVED and listed with its expression and a reason -- never guessed.
* An HTTP verb method (``x.get(...)``, ``x.post(...)``) counts on any receiver when
  its target is URL or path shaped: a literal starting with ``/`` or a scheme, or a
  template/concatenation whose first literal starts with one, or whose leading
  hole is followed by ``/`` (``f"{url}/health"``). On an HTTP-named receiver a
  RELATIVE target (a base URL is configured elsewhere) also counts, as ``/`` + it,
  when the verb is not ``get``, the receiver is an HTTP library (or bound in this
  file to one's client, ``async with httpx.AsyncClient(...) as client``), the target
  contains ``/`` (``api.get(`users/${id}`)``), or a Python ``.get`` passes any
  keyword (``dict.get`` takes none). A bare single word under ``.get``
  (``client.get("redirect_uris")``) is a mapping key and is never turned into a
  path: UNRESOLVED on a strongly HTTP-typed receiver (``apiClient``, ``http``),
  otherwise not a call. Python ``.get(key, default)`` on a non-library receiver
  is a mapping read. A fully dynamic target on an HTTP-named receiver is UNRESOLVED.
* Other generic names (``publish``, ``subscribe``, ``send``) only count when the
  target is topic shaped, or the receiver is named like a messaging client.
  ``store.subscribe(fn)`` is not a call.

Cost model (the input is attacker-controlled tenant source): every pass is linear
in the file, or linear times a fixed cap. Constant inlining is memoised and capped
(:data:`_MAX_PARTS`, :data:`_MAX_CHARS`, :data:`_MAX_NESTING`); a target past a cap
is UNRESOLVED with reason ``too_complex``, never expanded. Enclosing symbols are
carried down the walk, never recomputed through ``node.parent`` (which tree-sitter
answers by re-walking from the root). Text of a large node is never materialised
just to be truncated.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .checks.types import normalize_path

EXTRACTOR_VERSION = "1.0"

#: language -> file extensions. ``svelte``/``vue`` parse their ``<script>`` blocks as TS.
LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".mjs", ".cjs", ".jsx"),
    "typescript": (".ts", ".mts", ".cts", ".tsx"),
    "svelte": (".svelte",),
    "vue": (".vue",),
    "rust": (".rs",),
}
SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(LANGUAGE_EXTENSIONS)

_TEST_FILE = re.compile(
    r"(^|/)(tests?|__tests__|__mocks__|e2e|fixtures?)/"
    r"|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$"
    r"|\.(test|spec)\.[cm]?[jt]sx?$|\.d\.ts$"
)

HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
#: ``x.request("GET", url)`` / ``client.stream("POST", url)`` / ``_request(...)``.
_EXPLICIT_METHOD_CALLS = {"request", "_request", "stream", "arequest"}
_KNOWN_HTTP_LIBS = {"requests", "httpx", "aiohttp", "axios", "ky", "got", "superagent", "urllib"}
_HTTP_RECEIVER = re.compile(
    r"^(requests|httpx|aiohttp|axios|ky|got|superagent|session|client|http|https|api"
    r"|.*(client|session|http|api|axios))$"
)
#: Receivers that are HTTP clients by type, not just by a generic name like ``client``
#: or ``api``: a bare-word ``.get("users")`` on one is UNRESOLVED rather than dropped.
_STRONG_HTTP_RECEIVER = re.compile(r"^(https?|.*(http_?client|api_?client|axios|http))$")
#: Receivers whose verb methods REGISTER routes (provider side) or are not HTTP at all.
_ROUTE_REGISTRAR = re.compile(r"^(app|router|server|fastify|blueprint|bp|routes?|hono|api_router)$")
_NOT_HTTP_RECEIVER = re.compile(
    r"redis|cache|store|environ|\benv\b|config|settings|headers|params|query|cookies|kwargs"
    r"|\bmap\b|dict|registry|storage",
    re.I,
)
_MESSAGING_RECEIVER = re.compile(
    r"(nats|^nc$|^js$|jetstream|kafka|producer|consumer|redis|pubsub|publisher|subscriber"
    r"|broker|^bus$|eventbus|^ch$|channel|amqp|rabbit|^mq$)"
)
_PUBLISH_OPS = {"publish", "publish_async", "basic_publish", "sendToQueue", "xadd"}
_SEND_OPS = {"send", "send_and_wait", "produce"}  # only on a Kafka producer receiver
_PRODUCER = re.compile(r"producer|kafka")
_SUBSCRIBE_OPS = {"subscribe", "psubscribe", "pull_subscribe", "basic_consume", "consume"}
_CONSUMER_CLASSES = {"AIOKafkaConsumer", "KafkaConsumer"}
#: Calls that take one options object carrying ``url`` (+ optional ``method``).
_OPTIONS_CALLS = {"axios", "customInstance", "ky", "got", "$fetch", "ofetch"}
_FUNCTION_NODES = ("arrow_function", "function_expression", "function", "lambda")
_UPPER_VERBS = {v.upper() for v in HTTP_VERBS}

_EVENT_IMPORTS = {
    "nats": "nats",
    "kafka": "kafka",
    "aiokafka": "kafka",
    "confluent_kafka": "kafka",
    "kafkajs": "kafka",
    "redis": "redis-pubsub",
    "ioredis": "redis-pubsub",
    "aioredis": "redis-pubsub",
    "pika": "amqp",
    "aio_pika": "amqp",
    "amqplib": "amqp",
    "amqp-connection-manager": "amqp",
}

#: ``[ \t]*`` not ``\s*``: ``\s`` spans newlines, which made each blank line rescan
#: every following one (quadratic on a file of blank lines).
_PY_IMPORT = re.compile(r"^[ \t]*(?:from|import)[ \t]+([A-Za-z_][\w.]*)", re.M)
_JS_IMPORT = re.compile(r"""(?:from\s+|require\(\s*|import\(\s*)['"]([^'"]+)['"]""")
#: Svelte/Vue ``<script>`` blocks are located by :func:`_script_blocks`, a linear
#: scan. A ``<script\b[^>]*>(.*?)</script>`` regex is quadratic on a file of
#: repeated unclosed openers (512 KB took 80 s holding the GIL).
_SCRIPT_OPEN = re.compile(r"<script\b", re.I)
_SCRIPT_CLOSE = re.compile(r"</script>", re.I)
_URL_PREFIX = re.compile(r"^(https?|wss?)://([^/?#]*)(.*)$", re.I)
_EXPRESS_PARAM = re.compile(r"/:([A-Za-z_]\w*)")
#: ``[^{}]`` not ``[^}]``: the latter rescans to the end of a run of ``{`` from
#: every one of them (quadratic).
_BRACE_HOLE = re.compile(r"\{[^{}]*\}")
_RAW_LIMIT = 200
#: Caps on one target expression's expansion (constants inlined, ``+`` chains
#: flattened). Past any of them the target is unresolved ``too_complex``.
_MAX_PARTS = 256
_MAX_CHARS = 2048
_MAX_NESTING = 64
#: Receivers, callee names and symbol names longer than this are not names.
_NAME_LIMIT = 512
_SYMBOL_LIMIT = 256
BARE_WORD_REASON = "bare word target: a mapping key or a relative path, not a joinable path"
TOO_COMPLEX_REASON = "too_complex: target expression expands past the resolver's caps"
#: The bytes the current tree was parsed from, so a large node's text can be read
#: as a bounded slice instead of copied whole (``node.text`` is O(span)).
_SOURCE: ContextVar[bytes | None] = ContextVar("call_sites_source", default=None)
_QUERY_HOLE = re.compile(r"(?i)query|search|params|suffix|(?<![a-z])qs(?![a-z])|\?|URLSearchParams")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CallSite:
    file: str
    line: int
    kind: str  # http | event
    target: str
    client: str
    symbol: str | None
    confidence: str  # exact | template
    method: str | None = None  # http: GET/POST/...; None when the method is dynamic
    operation: str | None = None  # event: publish | subscribe
    base: str | None = None  # stripped base-URL expression (http)
    host: str | None = None  # host[:port] of an absolute URL (http)
    raw: str = ""
    #: Where the base URL comes from (``base_source.classify_base``): literal | env |
    #: config | unknown; the env var NAME / config key / expression; an env default URL.
    base_source_kind: str | None = None
    base_source_name: str | None = None
    base_source_default: str | None = None


@dataclass
class UnresolvedCallSite:
    file: str
    line: int
    kind: str
    client: str
    symbol: str | None
    expression: str
    reason: str
    method: str | None = None
    operation: str | None = None


@dataclass
class FileResult:
    call_sites: list[CallSite] = field(default_factory=list)
    unresolved: list[UnresolvedCallSite] = field(default_factory=list)
    parsed: bool = True


def language_for(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix.lower()
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        if suffix in exts:
            return lang
    return None


def is_test_path(path: str) -> bool:
    return bool(_TEST_FILE.search(path))


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _text(node: Any) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", "replace")


def _span(node: Any) -> int:
    return node.end_byte - node.start_byte


def _short_text(node: Any, limit: int = _NAME_LIMIT) -> str:
    """``_text`` of a name-sized node; ``""`` when the node is longer than *limit*."""
    if node is None or _span(node) > limit:
        return ""
    return _text(node)


def _head(node: Any, limit: int) -> str:
    """At least the first *limit* bytes of a node's text, without copying all of it."""
    source = _SOURCE.get()
    if _span(node) <= limit or source is None:
        return _text(node)
    return source[node.start_byte : node.start_byte + limit].decode("utf-8", "replace")


def _raw(node: Any) -> str:
    if node is None:
        return ""
    text = " ".join(_head(node, 4 * _RAW_LIMIT).split())
    return text if len(text) <= _RAW_LIMIT else text[: _RAW_LIMIT - 1] + "…"


def _clip(label: str, limit: int = 80) -> str:
    text = "".join(label[: 4 * limit].split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _walk(node: Any):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def walk_with_parent(node: Any) -> Iterator[tuple[Any, Any]]:
    """``(node, parent)`` in pre-order. Never call ``node.parent`` in a loop: it is O(depth)."""
    stack: list[tuple[Any, Any]] = [(node, None)]
    while stack:
        cur, parent = stack.pop()
        yield cur, parent
        stack.extend((child, cur) for child in reversed(cur.children))


def _args(call: Any) -> tuple[list[Any], dict[str, Any]]:
    """Positional and keyword argument nodes of a call (Python or JS)."""
    arg_list = call.child_by_field_name("arguments")
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    if arg_list is None:
        return positional, keywords
    for child in arg_list.named_children:
        if child.type == "keyword_argument":
            name = _text(child.child_by_field_name("name"))
            keywords[name] = child.child_by_field_name("value")
        elif child.type == "comment":
            continue
        else:
            positional.append(child)
    return positional, keywords


def _obj_props(node: Any) -> dict[str, Any]:
    """``{key: value}`` of a JS object literal or Python dict literal."""
    props: dict[str, Any] = {}
    if node is None or node.type not in ("object", "dictionary"):
        return props
    for child in node.named_children:
        if child.type == "pair":
            key = child.child_by_field_name("key")
            name = _short_text(key).strip("\"'`")
            props[name] = child.child_by_field_name("value")
        elif child.type == "shorthand_property_identifier":
            props[_short_text(child)] = child
    return props


def _literal(node: Any) -> str | None:
    """The value of a plain (non-interpolated) string literal, else None."""
    if node is None or node.type != "string":
        return None
    parts = []
    for child in node.children:
        if child.type == "interpolation":
            return None
        if child.type in ("string_content", "string_fragment", "escape_sequence"):
            parts.append(_text(child))
        elif child.type == "string_start" and "f" in _text(child).lower():
            if any(c.type == "interpolation" for c in node.children):
                return None
    return "".join(parts)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


@dataclass
class _Param:
    expr: str
    too_complex: bool = False


Parts = list[str | _Param]
_BINARY = ("binary_expression", "binary_operator")


class _TooComplexError(Exception):
    """A target expression expands past :data:`_MAX_PARTS`/``_CHARS``/``_NESTING``."""


class _Acc:
    """Parts under construction, bounded by the caps."""

    __slots__ = ("chars", "parts")

    def __init__(self) -> None:
        self.parts: Parts = []
        self.chars = 0

    def add(self, items: Parts) -> None:
        if len(self.parts) + len(items) > _MAX_PARTS:
            raise _TooComplexError
        self.chars += sum(len(p) if isinstance(p, str) else 1 for p in items)
        if self.chars > _MAX_CHARS:
            raise _TooComplexError
        self.parts.extend(items)


def _is_plus(node: Any) -> bool:
    return node.type in _BINARY and _short_text(node.child_by_field_name("operator")) == "+"


class _Resolver:
    """Resolve a target expression to literal parts and ``{param}`` holes.

    Linear in the expression times the caps: a ``+`` chain is flattened
    iteratively (no recursion per term), each same-file constant is expanded once
    per inlining depth and memoised, and every intermediate result is bounded by
    :data:`_MAX_PARTS` / :data:`_MAX_CHARS`, recursion by :data:`_MAX_NESTING`.
    Past a cap the result is a single ``_Param(too_complex=True)``.
    """

    def __init__(self, constants: dict[str, Any]) -> None:
        self.constants = constants
        #: (name, inlining depth) -> parts, or None when that expansion is too complex.
        self._memo: dict[tuple[str, int], Parts | None] = {}

    def parts(self, node: Any) -> Parts:
        try:
            return list(self._parts(node, 0, 0))
        except _TooComplexError:
            return [_Param(_raw(node), too_complex=True)]

    def _parts(self, node: Any, depth: int, nest: int) -> Parts:
        if nest > _MAX_NESTING:
            raise _TooComplexError
        if node is None:
            return [_Param("")]
        t = node.type
        if t in ("parenthesized_expression", "as_expression", "non_null_expression"):
            inner = node.named_children[0] if node.named_children else None
            return self._parts(inner, depth, nest + 1)
        if t == "string":
            return self._string(node, depth, nest)
        if t == "template_string":
            out = _Acc()
            for child in node.children:
                if child.type in ("string_fragment", "escape_sequence"):
                    out.add([_text(child)])
                elif child.type == "template_substitution":
                    inner = child.named_children[0] if child.named_children else None
                    out.add(self._hole(inner, depth, nest + 1))
            return out.parts
        if t == "concatenated_string":
            out = _Acc()
            for child in node.named_children:
                out.add(self._parts(child, depth, nest + 1))
            return out.parts
        if t in _BINARY:
            if not _is_plus(node):
                return [_Param(_raw(node))]
            # Flatten the left spine iteratively: `a + b + ... + z` nests leftwards,
            # one level per term, and recursing on it overflowed the stack (~1000 terms).
            operands: list[Any] = []
            cur = node
            while _is_plus(cur):
                if len(operands) >= _MAX_PARTS:
                    raise _TooComplexError
                operands.append(cur.child_by_field_name("right"))
                cur = cur.child_by_field_name("left")
            operands.append(cur)
            out = _Acc()
            for operand in reversed(operands):
                out.add(self._parts(operand, depth, nest + 1))
            return out.parts
        if t == "identifier" and depth < 5:
            bound = self._constant(node, depth, nest)
            if bound is not None:
                return bound
        if t == "call":  # "...{}".format(x) keeps its braces as holes
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                if _short_text(fn.child_by_field_name("attribute")) == "format" and obj is not None:
                    if obj.type == "string":
                        return self._string(obj, depth, nest)
        return [_Param(_raw(node))]

    def _constant(self, node: Any, depth: int, nest: int) -> Parts | None:
        """The memoised expansion of a same-file constant; None when *node* is not one."""
        name = _short_text(node)
        bound = self.constants.get(name)
        if bound is None:
            return None
        key = (name, depth)
        if key in self._memo:
            cached = self._memo[key]
            if cached is None:
                raise _TooComplexError
            return cached
        try:
            expanded = self._parts(bound, depth + 1, nest + 1)
        except _TooComplexError:
            self._memo[key] = None
            raise
        self._memo[key] = expanded
        return expanded

    def _hole(self, node: Any, depth: int, nest: int) -> Parts:
        """An interpolation: a bound constant inlines, anything else is a hole."""
        if node is not None and node.type == "identifier" and depth < 5:
            bound = self._constant(node, depth, nest)
            if bound is not None:
                return bound
        return [_Param(_raw(node))]

    def _string(self, node: Any, depth: int, nest: int) -> Parts:
        out = _Acc()
        for child in node.children:
            if child.type in ("string_content", "string_fragment", "escape_sequence"):
                out.add([_text(child)])
            elif child.type == "interpolation":
                out.add(self._hole(child.child_by_field_name("expression"), depth, nest + 1))
        return out.parts


def _too_complex(parts: Parts) -> bool:
    return any(isinstance(p, _Param) and p.too_complex for p in parts)


def _collect_constants(root: Any, language: str) -> dict[str, Any]:
    """Names bound exactly once in the file to a string-shaped expression."""
    seen: dict[str, list[Any]] = {}
    for node in _walk(root):
        name_node = value = None
        if language == "python" and node.type == "assignment":
            name_node = node.child_by_field_name("left")
            value = node.child_by_field_name("right")
        elif node.type == "variable_declarator":
            name_node = node.child_by_field_name("name")
            value = node.child_by_field_name("value")
        if name_node is None or name_node.type != "identifier":
            continue
        seen.setdefault(_text(name_node), []).append(value)
    stringy = {
        "string",
        "template_string",
        "concatenated_string",
        "binary_expression",
        "binary_operator",
    }
    return {
        name: values[0]
        for name, values in seen.items()
        if len(values) == 1 and values[0] is not None and values[0].type in stringy
    }


_CLIENT_CLASSES = {"AsyncClient", "Client", "Session", "ClientSession"}
_HTTP_CLIENT_MODULES = {"httpx", "requests", "aiohttp"}


def _client_constructor(value: Any, http_imports: bool) -> bool:
    """``httpx.AsyncClient(...)``, ``requests.Session()``, ``axios.create(...)``, and a
    bare ``AsyncClient()``/``Session()`` in a Python file importing an HTTP library."""
    while value is not None and value.type == "await":
        value = value.named_children[0] if value.named_children else None
    if value is None or value.type not in ("call", "call_expression"):
        return False
    callee = _short_text(value.child_by_field_name("function"))
    root = callee.split(".", 1)[0]
    return root in _KNOWN_HTTP_LIBS or (http_imports and callee in _CLIENT_CLASSES)


def _collect_http_clients(root: Any, http_imports: bool) -> frozenset[str]:
    """Receiver tokens bound in this file to an HTTP library client constructor.

    ``async with httpx.AsyncClient(base_url=B) as client`` makes ``client.get("items")``
    a relative request on a configured base URL, not a mapping read. One pass.
    """
    tokens: set[str] = set()
    for node in _walk(root):
        target = value = None
        if node.type == "as_pattern":
            value = node.named_children[0] if node.named_children else None
            alias = node.child_by_field_name("alias")
            target = alias.named_children[0] if alias is not None and alias.named_children else None
        elif node.type == "assignment":
            target = node.child_by_field_name("left")
            value = node.child_by_field_name("right")
        elif node.type == "variable_declarator":
            target = node.child_by_field_name("name")
            value = node.child_by_field_name("value")
        if target is not None and _client_constructor(value, http_imports):
            token = _receiver_token(_short_text(target))
            if token:
                tokens.add(token)
    return frozenset(tokens)


@dataclass
class _Rendered:
    path: str
    template: bool
    base: str | None = None
    host: str | None = None


def _render(parts: Parts, *, http: bool = True) -> _Rendered | None:
    """Literal + ``{param}`` text; None when nothing static is left."""
    parts = [p for p in parts if not (isinstance(p, str) and p == "")]
    if not parts or all(isinstance(p, _Param) for p in parts):
        return None
    base = None
    if isinstance(parts[0], _Param) and isinstance(parts[1], str) and parts[1].startswith("/"):
        base = parts[0].expr
        parts = parts[1:]
    if isinstance(parts[0], _Param):
        return None  # e.g. `${x}api/y`: the leading piece decides what this is
    text = ""
    template = False
    for i, part in enumerate(parts):
        if isinstance(part, str):
            text += part
            continue
        # A hole glued to a word at the END (`/api/x${qs}`) is a query suffix only when
        # it reads as one; `${BASE}${path}` appends a PATH, and that is not knowable.
        if http and i == len(parts) - 1 and text and not text.endswith(("/", "=")):
            if _QUERY_HOLE.search(part.expr):
                break
            return None
        text += "{param}"
        template = True
    return _Rendered(path=text, template=template, base=base)


def _http_target(rendered: _Rendered) -> _Rendered | None:
    """Split host, strip query/fragment, normalise like the provider side."""
    text = rendered.path
    match = _URL_PREFIX.match(text)
    if match:
        rendered.host = match.group(2) or None
        text = match.group(3) or "/"
    if not text.startswith("/"):
        return None
    text = re.split(r"[?#]", text, maxsplit=1)[0]
    text = re.sub(r"/{2,}", "/", text)
    text = _EXPRESS_PARAM.sub("/{param}", text)
    text = _BRACE_HOLE.sub("{param}", text)  # also "{}" from "...".format()
    rendered.template = rendered.template or "{" in text
    rendered.path = normalize_path(text) or "/"
    return rendered


def _event_target(rendered: _Rendered) -> _Rendered | None:
    text = rendered.path.strip()
    if not text or any(c.isspace() for c in text):
        return None
    rendered.path = _BRACE_HOLE.sub("{param}", text)
    rendered.template = rendered.template or "{param}" in rendered.path
    return rendered


def _path_shaped(parts: Parts) -> bool:
    rendered = _render(parts)
    return rendered is not None and (
        rendered.base is not None
        or rendered.path.startswith("/")
        or bool(_URL_PREFIX.match(rendered.path))
    )


# ---------------------------------------------------------------------------
# Call classification
# ---------------------------------------------------------------------------


def _callee(call: Any) -> tuple[str, str | None, Any]:
    """``(name, receiver text, receiver node)`` of a call."""
    fn = call.child_by_field_name("function")
    # `await f<T>(x)` parses as `(await f<T>)(x)` in tree-sitter-typescript.
    while fn is not None and fn.type in ("await_expression", "non_null_expression"):
        fn = fn.named_children[0] if fn.named_children else None
    if fn is None:
        return "", None, None
    # Bounded: a receiver like `a.b(...).c(...)....` spans the whole chain, and reading
    # its text once per call in the chain was quadratic. A receiver longer than
    # _NAME_LIMIT is not a name anything below classifies on, so it reads as "…".
    if fn.type == "member_expression":
        obj = fn.child_by_field_name("object")
        return _short_text(fn.child_by_field_name("property")), _short_text(obj) or "…", obj
    if fn.type == "attribute":
        obj = fn.child_by_field_name("object")
        return _short_text(fn.child_by_field_name("attribute")), _short_text(obj) or "…", obj
    return _short_text(fn), None, None


def _receiver_token(receiver: str | None) -> str:
    if not receiver:
        return ""
    last = re.split(r"[.\[\]()?!]+", receiver.strip())
    last = [p for p in last if p and p not in ("self", "this", "cls")]
    return last[-1].lstrip("_$").lower() if last else receiver.strip().lower()


def _imports(source: str, language: str) -> set[str]:
    pattern = _PY_IMPORT if language == "python" else _JS_IMPORT
    out = set()
    for module in pattern.findall(source):
        if module.startswith("@"):
            out.add("/".join(module.split("/")[:2]))
        else:
            out.add(re.split(r"[./]", module)[0] if language == "python" else module)
    return out


class _FileExtractor:
    def __init__(self, path: str, language: str, source: str, line_offset: int = 0) -> None:
        self.path = path
        self.language = language
        self.line_offset = line_offset
        imports = _imports(source, language)
        event_libs = sorted({v for k, v in _EVENT_IMPORTS.items() if k in imports})
        self.event_lib = event_libs[0] if len(event_libs) == 1 else None
        self.result = FileResult()
        #: Python files importing an HTTP library may construct its client by bare name.
        self.http_imports = bool(imports & _HTTP_CLIENT_MODULES) and language == "python"
        #: Receiver tokens bound in this file to an HTTP library client (see run()).
        self.http_clients: frozenset[str] = frozenset()
        self.resolver = _Resolver({})
        self.symbol: str | None = None  # enclosing symbol of the call being visited
        self.parent: Any = None  # parent node of the call being visited

    # -- symbol ------------------------------------------------------------

    def _symbol(self, node: Any) -> str | None:
        """The enclosing symbol of the call being visited (carried down by :meth:`run`)."""
        return self.symbol

    def _line(self, node: Any) -> int:
        return node.start_point[0] + 1 + self.line_offset

    # -- emitters ----------------------------------------------------------

    def _http(
        self,
        call: Any,
        method: str | None,
        target_node: Any,
        client: str,
        *,
        known: bool,
        relative_ok: bool = False,
    ) -> None:
        client = _clip(client)
        parts = self.resolver.parts(target_node)
        rendered = _render(parts)
        if (
            relative_ok
            and rendered is not None
            and rendered.base is None
            and not rendered.path.startswith("/")
            and not _URL_PREFIX.match(rendered.path)
        ):
            rendered.path = "/" + rendered.path  # axios instance with a baseURL
        target = _http_target(rendered) if rendered is not None else None
        if target is None:
            if known:
                self.result.unresolved.append(
                    UnresolvedCallSite(
                        file=self.path,
                        line=self._line(call),
                        kind="http",
                        method=method,
                        client=client,
                        symbol=self._symbol(call),
                        expression=_raw(target_node),
                        reason=TOO_COMPLEX_REASON
                        if _too_complex(parts)
                        else "dynamic target: not a literal, template or same-file constant",
                    )
                )
            return
        self.result.call_sites.append(
            CallSite(
                file=self.path,
                line=self._line(call),
                kind="http",
                method=method,
                target=target.path,
                client=client,
                symbol=self._symbol(call),
                confidence="template" if target.template else "exact",
                base=target.base,
                host=target.host,
                raw=_raw(target_node),
            )
        )

    def _unresolved_http(self, call: Any, method: str, target_node: Any, client: str) -> None:
        self.result.unresolved.append(
            UnresolvedCallSite(
                file=self.path,
                line=self._line(call),
                kind="http",
                method=method,
                client=_clip(client),
                symbol=self._symbol(call),
                expression=_raw(target_node),
                reason=BARE_WORD_REASON,
            )
        )

    def _event(
        self, call: Any, operation: str, target_node: Any, client: str, *, known: bool
    ) -> None:
        client = _clip(client)
        parts = self.resolver.parts(target_node)
        rendered = _render(parts, http=False)
        target = _event_target(rendered) if rendered is not None else None
        if target is None:
            if known:
                self.result.unresolved.append(
                    UnresolvedCallSite(
                        file=self.path,
                        line=self._line(call),
                        kind="event",
                        operation=operation,
                        client=client,
                        symbol=self._symbol(call),
                        expression=_raw(target_node),
                        reason=TOO_COMPLEX_REASON
                        if _too_complex(parts)
                        else "dynamic topic: not a literal, template or same-file constant",
                    )
                )
            return
        self.result.call_sites.append(
            CallSite(
                file=self.path,
                line=self._line(call),
                kind="event",
                operation=operation,
                target=target.path,
                client=client,
                symbol=self._symbol(call),
                confidence="template" if target.template else "exact",
                raw=_raw(target_node),
            )
        )

    # -- classification ----------------------------------------------------

    def _http_client(self, receiver: str | None, name: str) -> str:
        """The library when the receiver IS one, else the receiver as written."""
        token = _receiver_token(receiver)
        if token in _KNOWN_HTTP_LIBS:
            return token
        return name if receiver is None else receiver

    def _event_client(self, receiver: str | None) -> str:
        """Receiver name first (``nc``, ``redis``), then the file's ONE messaging import."""
        token = _receiver_token(receiver)
        if token in ("nc", "js"):
            return "nats"
        for needle, label in (
            ("nats", "nats"),
            ("jetstream", "nats"),
            ("kafka", "kafka"),
            ("redis", "redis-pubsub"),
            ("amqp", "amqp"),
            ("rabbit", "amqp"),
        ):
            if needle in token:
                return label
        if self.event_lib:
            return self.event_lib
        return receiver or "unknown"

    def _method_from_init(self, init: Any) -> str | None:
        """``fetch(url, {method})``: literal method, GET when absent, None if dynamic."""
        if init is None:
            return "GET"
        props = _obj_props(init)
        if init.type not in ("object", "dictionary"):
            return None
        if any(c.type == "spread_element" for c in init.named_children) and "method" not in props:
            return None
        node = props.get("method")
        if node is None:
            return "GET"
        value = _literal(node)
        return value.upper() if value and value.upper() in _UPPER_VERBS else None

    def visit_call(self, call: Any) -> None:
        name, receiver, _ = _callee(call)
        if not name:
            return
        positional, keywords = _args(call)
        token = _receiver_token(receiver)
        typed = self._typed_http(token)
        http_receiver = (bool(_HTTP_RECEIVER.match(token)) or typed) if receiver else False
        messaging_receiver = bool(_MESSAGING_RECEIVER.search(token)) if receiver else False

        # --- HTTP -----------------------------------------------------------
        if name in ("call", "apply") and receiver is not None and "." in receiver:
            # helpers.requestWithAuthentication.call(this, cred, options)
            receiver, name = receiver.rsplit(".", 1)
            positional = positional[1:]
            token = _receiver_token(receiver)
            typed = self._typed_http(token)
            http_receiver = bool(_HTTP_RECEIVER.match(token)) or typed
        options = next(
            (
                a
                for a in positional
                if a.type in ("object", "dictionary")
                and ("url" in _obj_props(a) or "uri" in _obj_props(a))
            ),
            None,
        )
        if options is not None and (
            "request" in name.lower() or name in _OPTIONS_CALLS or "method" in _obj_props(options)
        ):
            # httpRequest({method, url}), axios({url}), customInstance({url, method})
            props = _obj_props(options)
            self._http(
                call,
                self._method_from_init(options),
                props.get("url") or props.get("uri"),
                name if receiver is None else f"{receiver}.{name}",
                known=True,
            )
            return
        if (
            name.lower() in HTTP_VERBS
            and receiver is not None
            and not _ROUTE_REGISTRAR.match(token)
            and not _NOT_HTTP_RECEIVER.search(receiver)
            and self.parent is not None
            and self.parent.type != "decorator"
        ):
            target = positional[0] if positional else keywords.get("url")
            if target is not None and target.type not in _FUNCTION_NODES:
                parts = self.resolver.parts(target)
                rendered = _render(parts)
                client = self._http_client(receiver, name)
                if _path_shaped(parts):
                    self._http(call, name.upper(), target, client, known=http_receiver)
                elif not http_receiver or self._dict_get(name, token, positional, keywords):
                    # `cache.get("k")`, Python `x.get(key, default)`: a mapping read.
                    pass
                elif rendered is None:
                    # `client.get(url)`: an HTTP call whose target is not knowable here.
                    self._http(call, name.upper(), target, client, known=True)
                elif (
                    name.lower() != "get"
                    or typed
                    or "/" in rendered.path
                    or (self.language == "python" and keywords)
                ):
                    # A relative path on an HTTP client with a base URL: `api.post("auth/
                    # login")`, `axios.get("api/users")`, `api.get(`users/${id}`)`.
                    self._http(call, name.upper(), target, client, known=True, relative_ok=True)
                elif _STRONG_HTTP_RECEIVER.match(token):
                    # `apiClient.get("users")`: an HTTP client, but one bare word is not
                    # a path anything can join on. Listed, never guessed into `/users`.
                    self._unresolved_http(call, name.upper(), target, client)
                # Otherwise a bare word on a receiver merely NAMED like a client
                # (`client.get("redirect_uris")`, `manifest.api.get("dep-1")`): a mapping
                # key, never turned into a path.
                return
        if name in _EXPLICIT_METHOD_CALLS:
            verb_node = positional[0] if positional else keywords.get("method")
            target = positional[1] if len(positional) >= 2 else keywords.get("url")
            verb = _literal(verb_node)
            if target is not None and verb and verb.upper() in _UPPER_VERBS:
                self._http(
                    call,
                    verb.upper(),
                    target,
                    self._http_client(receiver, name),
                    known=http_receiver or self._path_like(target),
                    relative_ok=http_receiver,
                )
                return
        if self._is_fetch_like(name, receiver) and positional:
            method = self._method_from_init(positional[1] if len(positional) > 1 else None)
            self._http(
                call,
                method,
                positional[0],
                name if receiver is None else f"{receiver}.{name}",
                known=True,
            )
            return
        if name == "urlopen" and positional:
            self._http(call, "GET", positional[0], "urllib", known=True)
            return

        # --- events ---------------------------------------------------------
        if name in _CONSUMER_CLASSES:
            for topic in positional:
                self._event(call, "subscribe", topic, "kafka", known=True)
            return
        operation = None
        if name in _PUBLISH_OPS or (name in _SEND_OPS and _PRODUCER.search(token)):
            operation = "publish"
        elif name in _SUBSCRIBE_OPS and (name != "consume" or messaging_receiver):
            operation = "subscribe"
        if operation is None or receiver is None:
            return
        client = self._event_client(receiver)
        if name == "basic_publish":
            target = keywords.get("routing_key")
            if target is not None:
                self._event(call, operation, target, client, known=True)
            return
        if name == "basic_consume":
            target = keywords.get("queue")
            if target is not None:
                self._event(call, operation, target, client, known=True)
            return
        if name == "sendToQueue" and positional:
            self._event(call, operation, positional[0], client, known=True)
            return
        if not positional:
            target = keywords.get("subject") or keywords.get("topic") or keywords.get("channel")
            topics = keywords.get("topics")
            if topics is not None and topics.type == "list":
                for item in topics.named_children:
                    self._event(call, operation, item, client, known=True)
                return
            if target is not None:
                self._event(call, operation, target, client, known=True)
            return
        first = positional[0]
        if first.type in ("object", "dictionary"):  # kafkajs {topic} / {topics: [...]}
            props = _obj_props(first)
            if "topic" in props:
                self._event(call, operation, props["topic"], client, known=True)
            elif "topics" in props and props["topics"].type == "array":
                for item in props["topics"].named_children:
                    self._event(call, operation, item, client, known=True)
            return
        if first.type in ("list", "array"):
            for item in first.named_children:
                self._event(call, operation, item, client, known=True)
            return
        if first.type in (
            "arrow_function",
            "function_expression",
            "function",
            "lambda",
        ):
            return  # store.subscribe(fn), observable.subscribe(fn)
        literal_shaped = _render(self.resolver.parts(first), http=False) is not None
        if literal_shaped or messaging_receiver:
            self._event(call, operation, first, client, known=messaging_receiver)

    def _dict_get(
        self, name: str, token: str, positional: list[Any], keywords: dict[str, Any]
    ) -> bool:
        """``x.get(key, default)``: Python's mapping signature, not an HTTP client's.

        ``requests``/``httpx`` take params and options as keywords; a second
        positional argument to ``.get`` on anything that is not the library itself
        is a dict default. ``dict.get`` takes no keywords, so any keyword makes it
        HTTP. (In JS ``api.get(url, config)`` is the axios signature.)
        """
        return (
            self.language == "python"
            and name == "get"
            and len(positional) >= 2
            and not keywords
            and not self._typed_http(token)
        )

    def _typed_http(self, token: str) -> bool:
        """The receiver IS an HTTP client: the library itself, or bound to one here."""
        return token in _KNOWN_HTTP_LIBS or token in self.http_clients

    def _path_like(self, node: Any) -> bool:
        return _path_shaped(self.resolver.parts(node))

    @staticmethod
    def _is_fetch_like(name: str, receiver: str | None) -> bool:
        if receiver is None:
            return name in ("fetch", "$fetch", "ofetch", "axios", "ky", "got") or (
                name.endswith(("Fetch", "_fetch"))
            )
        return (name == "fetch" and receiver in ("window", "globalThis", "self")) or (
            name == "request" and _receiver_token(receiver) == "axios"
        )

    def run(self, root: Any) -> FileResult:
        self.resolver = _Resolver(_collect_constants(root, self.language))
        self.http_clients = _collect_http_clients(root, self.http_imports)
        call_type = "call" if self.language == "python" else "call_expression"
        for node, parent, symbol in walk_with_symbols(root):
            # A call inside a parse error is not a reliable reading of anything.
            if node.type == call_type and not node.has_error:
                self.parent, self.symbol = parent, symbol
                self.visit_call(node)
        return self.result


_SCOPE_TYPES = (
    "function_definition",
    "class_definition",
    "function_declaration",
    "generator_function_declaration",
    "class_declaration",
    "method_definition",
    "abstract_class_declaration",
)
_NAMED_VALUES = ("arrow_function", "function_expression", "function", "object", "call_expression")


def _scope_name(node: Any) -> str | None:
    """The name *node* contributes to the dotted symbol of what it encloses, if any."""
    t = node.type
    if t in _SCOPE_TYPES:
        return _short_text(node.child_by_field_name("name"), _SYMBOL_LIMIT) or None
    if t in ("public_field_definition", "field_definition"):
        name = node.child_by_field_name("name") or node.child_by_field_name("property")
        return _short_text(name, _SYMBOL_LIMIT) or None
    if t == "pair":
        return _short_text(node.child_by_field_name("key"), _SYMBOL_LIMIT).strip("\"'`") or None
    if t == "variable_declarator":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _NAMED_VALUES:
            return _short_text(node.child_by_field_name("name"), _SYMBOL_LIMIT) or None
    return None


def walk_with_symbols(root: Any) -> Iterator[tuple[Any, Any, str | None]]:
    """``(node, parent, enclosing symbol)`` in pre-order, in one pass.

    The symbol is the dotted names of the enclosing functions/classes/methods/keys,
    computed once per scope on the way down (the old per-call ``node.parent`` loop
    was O(depth) per hop in tree-sitter, so O(calls x depth^2)). A symbol longer
    than :data:`_SYMBOL_LIMIT` keeps its innermost names behind a leading ``…``.
    """
    stack: list[tuple[Any, Any, str | None]] = [(root, None, None)]
    while stack:
        node, parent, symbol = stack.pop()
        yield node, parent, symbol
        name = _scope_name(node)
        inner = symbol
        if name:
            inner = f"{symbol}.{name}" if symbol else name
            if len(inner) > _SYMBOL_LIMIT:
                inner = "…" + inner[-(_SYMBOL_LIMIT - 1) :]
        stack.extend((child, node, inner) for child in reversed(node.children))


def _grammar(language: str, path: str) -> str:
    if language == "python":
        return "python"
    if language == "javascript":
        return "javascript"
    if path.lower().endswith(".tsx"):
        return "tsx"
    return "typescript"


def _script_blocks(source: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every ``<script ...>`` body, in one forward pass.

    Each search starts where the previous one stopped, so the scan is linear. An
    opener with no ``>`` ends the scan (no later opener can close either); a body
    with no ``</script>`` runs to end of file, once.

    The unterminated case is deliberate: everything after an unclosed ``<script>``
    is parsed as TypeScript (as a browser would treat it as script), so a file with
    a missing closer still yields its calls. Cost stays linear: it is one block,
    and the parse and walk of a block are linear in its length.
    """
    blocks: list[tuple[int, int]] = []
    pos = 0
    while True:
        opener = _SCRIPT_OPEN.search(source, pos)
        if opener is None:
            return blocks
        tag_end = source.find(">", opener.end())
        if tag_end == -1:
            return blocks
        start = tag_end + 1
        closer = _SCRIPT_CLOSE.search(source, start)
        if closer is None:
            blocks.append((start, len(source)))
            return blocks
        blocks.append((start, closer.start()))
        pos = closer.end()


def extract_file_call_sites(path: str, source: str, language: str | None = None) -> FileResult:
    """Every outbound HTTP/event call site in one file.

    ``parsed`` is False only when the grammar is unavailable or the file's
    language is unknown; a file with syntax errors still yields what parsed.
    """
    lang = language or language_for(path)
    if lang is None:
        return FileResult(parsed=False)
    if lang == "rust":
        from .rust_http import extract_rust_call_sites

        return extract_rust_call_sites(path, source)
    if lang in ("svelte", "vue"):
        return _with_base_sources(_extract_script_blocks(path, source), source, lang)
    content = source.encode("utf-8")
    tree = _parse(_grammar(lang, path), content)
    if tree is None:
        return FileResult(parsed=False)
    token = _SOURCE.set(content)
    try:
        return _with_base_sources(
            _FileExtractor(path, lang, source).run(tree.root_node), source, lang
        )
    finally:
        _SOURCE.reset(token)


def _with_base_sources(result: FileResult, source: str, language: str) -> FileResult:
    """Classify every HTTP call site's base URL (literal / env / config / unknown)."""
    from .base_source import BaseClassifier

    classifier = BaseClassifier(source, language)
    for site in result.call_sites:
        if site.kind != "http":
            continue
        found = classifier.classify(site.base, site.host)
        if found is not None:
            site.base_source_kind = found.kind
            site.base_source_name = found.name
            site.base_source_default = found.default
    return result


def _extract_script_blocks(path: str, source: str) -> FileResult:
    """Svelte/Vue: each ``<script>`` body parsed as TypeScript, lines kept file-relative."""
    combined = FileResult()
    offset = counted = 0  # running line count: one pass over the file, not one per block
    for start, end in _script_blocks(source):
        body = source[start:end]
        offset += source.count("\n", counted, start)
        counted = start
        content = body.encode("utf-8")
        tree = _parse("typescript", content)
        if tree is None:
            return FileResult(parsed=False)
        token = _SOURCE.set(content)
        try:
            part = _FileExtractor(path, "typescript", body, offset).run(tree.root_node)
        finally:
            _SOURCE.reset(token)
        combined.call_sites.extend(part.call_sites)
        combined.unresolved.extend(part.unresolved)
    return combined


def sort_key(site: CallSite | UnresolvedCallSite) -> tuple[str, int, str]:
    return (site.file, site.line, site.kind)


def as_dict(site: CallSite | UnresolvedCallSite) -> dict[str, Any]:
    return asdict(site)
