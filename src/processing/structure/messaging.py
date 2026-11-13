"""Message-based coupling: publishers and consumers of channels, streams and task queues.

One :class:`MessageEndpoint` per publish / consume site; :func:`match_messages`
pairs publishers with consumers across (and within) repos by channel name.

Kinds
    ``pubsub``  Redis pub/sub. Python (in a file that imports redis/aioredis/coredis)
                ``x.publish(ch, msg)``, ``x.subscribe(ch, ...)``, ``x.psubscribe(pattern)``
                (positional channels);
                Rust (redis crate) ``.publish(ch, msg)``, ``.subscribe(ch)``,
                ``.psubscribe(p)`` and ``redis::cmd("PUBLISH"|"SUBSCRIBE"|"PSUBSCRIBE").arg(ch)``.
    ``stream``  Redis streams. ``xadd(key, ...)``, ``xread({key: id})`` /
                ``xreadgroup(group, consumer, {key: id})`` (or ``streams=``); Rust ``xadd``,
                ``xread(&[keys], ..)``, ``xreadgroup(g, c, &[keys], ..)``, ``cmd("XADD").arg(key)``.
    ``task``    Python task queues. Celery ``send_task("name")`` and ``fn.delay()`` /
                ``fn.apply_async()`` (``fn`` imported or a task of the same file) ->
                ``@task`` / ``@shared_task`` functions in a file that imports celery
                (``name=`` honoured, else ``module.fn``); rq ``enqueue(fn | "mod.fn")`` (also ``enqueue_call(func=)``,
                ``enqueue_in`` / ``enqueue_at``) in a file that imports rq -> that
                module-level function; arq ``enqueue_job("fn")`` -> ``WorkerSettings.functions``.

Channel names
    A string literal or a module-level string constant is ``exact`` (a ``psubscribe``
    glob is a ``pattern``); an f-string / ``format!`` with a constant prefix is the
    ``pattern`` ``prefix*``; ``os.getenv("X")`` / ``os.environ["X"]`` /
    ``env::var("X")`` is ``unresolved`` carrying ``X``; anything else is
    ``unresolved`` carrying the expression.

Matching
    A publisher with an unresolved channel is one edge with ``match="unresolved"``.
    Otherwise every consumer of the same kind whose channel matches is an edge
    (``exact`` or ``pattern``). A task matches only a consumer of the same framework
    (celery / rq / arq) registered under exactly that name. Resolved publishers
    without a consumer are ``unmatched_publishers``; consumers no publisher reached
    are ``unmatched_consumers`` (rq targets are any module-level function, so they
    are never listed there).
"""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .text import clip, node_text, unparse

_GLOB = re.compile(r"[*?\[]")


@dataclass
class Channel:
    name: str
    kind: str  # exact | pattern | unresolved
    reason: str | None = None


@dataclass
class MessageEndpoint:
    repo: str
    file: str
    line: int
    role: str  # publisher | consumer
    kind: str  # pubsub | stream | task
    channel: str
    channel_kind: str  # exact | pattern | unresolved
    via: str  # e.g. redis.publish, celery.send_task, rq.enqueue, redis-rs.xadd
    symbol: str | None = None  # enclosing function (the task function for a consumer)
    reason: str | None = None


@dataclass
class MessageEdge:
    from_repo: str
    from_file: str
    from_line: int
    kind: str
    from_channel: str
    match: str  # exact | pattern | unresolved
    from_via: str
    from_symbol: str | None = None
    to_repo: str | None = None
    to_file: str | None = None
    to_line: int | None = None
    to_symbol: str | None = None
    to_channel: str | None = None
    to_via: str | None = None
    reason: str | None = None


@dataclass
class MessageGraph:
    edges: list[MessageEdge]
    unmatched_publishers: list[MessageEndpoint]
    unmatched_consumers: list[MessageEndpoint]


@dataclass
class RepoMessaging:
    endpoints: list[MessageEndpoint]
    rq_targets: list[MessageEndpoint]  # module-level functions an rq job may name
    unparsed: dict[str, str]


def _literal(value: str, glob: bool) -> Channel:
    return Channel(value, "pattern" if glob and _GLOB.search(value) else "exact")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _py_channel(node: ast.AST | None, consts: dict[str, str], glob: bool = False) -> Channel:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _literal(node.value, glob)
    if isinstance(node, ast.Name) and node.id in consts:
        return _literal(consts[node.id], glob)
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value:
            return Channel(first.value + "*", "pattern")
    env = None
    if (
        isinstance(node, ast.Call)
        and node.args
        and (unparse(node.func) or "").endswith(("getenv", "environ.get"))
    ):
        env = node.args[0]
    elif isinstance(node, ast.Subscript) and (unparse(node.value) or "").endswith("environ"):
        env = node.slice
    if isinstance(env, ast.Constant) and isinstance(env.value, str):
        return Channel(env.value, "unresolved", "channel name comes from an environment variable")
    return Channel(unparse(node) or "<expression>", "unresolved", "channel name is computed")


def _imports(tree: ast.Module, *roots: str) -> bool:
    for node in ast.walk(tree):
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if any(n.split(".")[0] in roots for n in names):
            return True
    return False


def _module_of(path: str, files: Collection[str]) -> str:
    """Importable module name of *path*: walk up while the parent dir is a package
    (has ``__init__.py``), as Python's own package resolution does."""
    *dirs, stem = PurePosixPath(path).with_suffix("").parts
    module = [] if stem == "__init__" else [stem]
    while dirs and "/".join([*dirs, "__init__.py"]) in files:
        module.insert(0, dirs.pop())
    return ".".join(module)


def _task_name(module: str, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The name celery registers *fn* under (``name=``, else ``module.fn``), None when
    *fn* is not decorated ``@task`` / ``@shared_task`` / ``@app.task``."""
    for dec in fn.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        target = call.func if call is not None else dec
        if (unparse(target) or "").rsplit(".", 1)[-1] in ("task", "shared_task"):
            for kw in call.keywords if call is not None else []:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
            return f"{module}.{fn.name}" if module else fn.name
    return None


class _PyFile:
    def __init__(self, repo: str, path: str, tree: ast.Module, files: Collection[str]) -> None:
        self.repo, self.path, self.tree = repo, path, tree
        self.module = _module_of(path, files)
        self.consts = {
            s.targets[0].id: s.value.value
            for s in tree.body
            if isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        }
        self.imported: dict[str, str] = {}  # local name -> dotted name
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imported[a.asname or a.name.split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    self.imported[a.asname or a.name] = f"{node.module}.{a.name}"
        self.top_funcs = [
            s for s in tree.body if isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        # Celery tasks only in a file that imports celery (Invoke / Airflow have @task too).
        self.celery = _imports(tree, "celery")
        self.tasks = {
            f.name for f in self.top_funcs if self.celery and _task_name(self.module, f) is not None
        }
        self.funcs = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        self.redis = _imports(tree, "redis", "aioredis", "coredis")
        self.rq = _imports(tree, "rq")
        self.out: list[MessageEndpoint] = []

    def enclosing(self, line: int) -> str | None:
        inside = [f for f in self.funcs if f.lineno <= line <= (f.end_lineno or f.lineno)]
        return max(inside, key=lambda f: f.lineno).name if inside else None

    def add(self, node: ast.AST, role: str, kind: str, ch: Channel, via: str, **extra: Any) -> None:
        line = getattr(node, "lineno", 1)
        self.out.append(
            MessageEndpoint(
                repo=self.repo,
                file=self.path,
                line=line,
                role=role,
                kind=kind,
                channel=ch.name,
                channel_kind=ch.kind,
                via=via,
                symbol=extra.pop("symbol", None) or self.enclosing(line),
                reason=ch.reason,
                **extra,
            )
        )

    def reference(self, node: ast.AST) -> str | None:
        """Dotted name of a function reference, resolved through this file's imports."""
        if isinstance(node, ast.Name):
            if node.id in self.imported:
                return self.imported[node.id]
            if any(f.name == node.id for f in self.top_funcs) and self.module:
                return f"{self.module}.{node.id}"
            return node.id
        if isinstance(node, ast.Attribute):
            base = self.reference(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def celery_task(self, recv: ast.AST) -> str | None:
        """``recv`` of ``recv.delay()`` when it is an imported name or a task of this file."""
        root = recv
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            return None  # Pool().apply_async, pools[i].apply_async
        if root.id in self.imported or (recv is root and root.id in self.tasks):
            return self.reference(recv)
        return None

    def call(self, node: ast.Call) -> None:
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            return
        name, args = fn.attr, node.args
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if name == "publish" and self.redis and args:
            self.add(
                node, "publisher", "pubsub", _py_channel(args[0], self.consts), "redis.publish"
            )
        elif name in ("subscribe", "psubscribe") and self.redis:
            glob = name == "psubscribe"
            for key in args:  # channels are positional; keywords are handlers / options
                self.add(
                    node, "consumer", "pubsub", _py_channel(key, self.consts, glob), f"redis.{name}"
                )
        elif name == "xadd" and args:
            self.add(node, "publisher", "stream", _py_channel(args[0], self.consts), "redis.xadd")
        elif name in ("xread", "xreadgroup"):
            index = 2 if name == "xreadgroup" else 0
            streams = kw.get("streams") or (args[index] if len(args) > index else None)
            for key in streams.keys if isinstance(streams, ast.Dict) else [streams]:
                self.add(node, "consumer", "stream", _py_channel(key, self.consts), f"redis.{name}")
        elif name == "send_task" and args:
            self.add(
                node, "publisher", "task", _py_channel(args[0], self.consts), "celery.send_task"
            )
        elif name in ("delay", "apply_async"):
            task = self.celery_task(fn.value)
            if task is not None:
                self.add(node, "publisher", "task", Channel(task, "exact"), f"celery.{name}")
        elif name == "enqueue_job" and args:
            self.add(
                node, "publisher", "task", _py_channel(args[0], self.consts), "arq.enqueue_job"
            )
        elif name in ("enqueue", "enqueue_call", "enqueue_in", "enqueue_at") and self.rq:
            index = 1 if name in ("enqueue_in", "enqueue_at") else 0
            target = kw.get("func") or (args[index] if len(args) > index else None)
            if target is None:
                return
            ref = self.reference(target)
            ch = Channel(ref, "exact") if ref else _py_channel(target, self.consts)
            self.add(node, "publisher", "task", ch, f"rq.{name}")

    def scan(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.call(node)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                task = _task_name(self.module, node) if self.celery else None
                if task is not None:
                    self.add(
                        node,
                        "consumer",
                        "task",
                        Channel(task, "exact"),
                        "celery.task",
                        symbol=node.name,
                    )
            elif isinstance(node, ast.ClassDef) and node.name == "WorkerSettings":
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "functions" for t in stmt.targets
                        )
                        and isinstance(stmt.value, ast.List | ast.Tuple)
                    ):
                        for elt in stmt.value.elts:
                            name = (unparse(elt) or "").rsplit(".", 1)[-1]
                            self.add(
                                stmt,
                                "consumer",
                                "task",
                                Channel(name, "exact"),
                                "arq.function",
                                symbol=name,
                            )

    def rq_targets(self) -> list[MessageEndpoint]:
        return [
            MessageEndpoint(
                repo=self.repo,
                file=self.path,
                line=f.lineno,
                role="consumer",
                kind="task",
                channel=f"{self.module}.{f.name}" if self.module else f.name,
                channel_kind="exact",
                via="rq.function",
                symbol=f.name,
            )
            for f in self.top_funcs
        ]


# ---------------------------------------------------------------------------
# Rust (redis crate)
# ---------------------------------------------------------------------------

_RS_METHODS = {
    "publish": ("publisher", "pubsub"),
    "subscribe": ("consumer", "pubsub"),
    "psubscribe": ("consumer", "pubsub"),
    "xadd": ("publisher", "stream"),
    "xadd_maxlen": ("publisher", "stream"),
    "xread": ("consumer", "stream"),
    "xread_options": ("consumer", "stream"),
    "xreadgroup": ("consumer", "stream"),
}
_RS_CMDS = {
    "PUBLISH": ("publisher", "pubsub"),
    "SUBSCRIBE": ("consumer", "pubsub"),
    "PSUBSCRIBE": ("consumer", "pubsub"),
    "XADD": ("publisher", "stream"),
}


def _deref(node: Any) -> Any:
    while node is not None and node.type == "reference_expression":
        node = node.child_by_field_name("value")
    return node


def _rs_string(node: Any) -> str | None:
    if node is None or node.type not in ("string_literal", "raw_string_literal"):
        return None
    return "".join(node_text(c) for c in node.named_children if c.type == "string_content")


def _rs_channel(node: Any, consts: dict[str, str], glob: bool = False) -> Channel:
    node = _deref(node)
    literal = _rs_string(node)
    if literal is not None:
        return _literal(literal, glob)
    text = node_text(node)
    if node is not None and node.type == "identifier" and text in consts:
        return _literal(consts[text], glob)
    fmt = re.match(r'format!\s*\(\s*"([^"{]+)', text)
    if fmt:
        return Channel(fmt.group(1) + "*", "pattern")
    env = re.search(r'env::var\(\s*"([^"]+)"', text)
    if env:
        return Channel(
            env.group(1), "unresolved", "channel name comes from an environment variable"
        )
    return Channel(clip(text) or "<expression>", "unresolved", "channel name is computed")


def _rs_args(call: Any) -> list[Any]:
    args = call.child_by_field_name("arguments")
    return list(args.named_children) if args is not None else []


def _rust(repo: str, path: str, source: str) -> list[MessageEndpoint]:
    if "redis" not in source:
        return []
    tree = _parse("rust", source.encode("utf-8"))
    if tree is None:
        raise ValueError("grammar not available")
    root = tree.root_node
    consts = {
        node_text(n.child_by_field_name("name")): value
        for n in root.named_children
        if n.type in ("const_item", "static_item")
        and (value := _rs_string(n.child_by_field_name("value"))) is not None
    }
    out: list[MessageEndpoint] = []
    stack: list[tuple[Any, str | None]] = [(root, None)]
    while stack:
        node, symbol = stack.pop()
        if node.type == "function_item":
            symbol = node_text(node.child_by_field_name("name"))
        stack.extend((c, symbol) for c in node.named_children)
        fn = node.child_by_field_name("function") if node.type == "call_expression" else None
        if fn is None or fn.type != "field_expression":
            continue
        method = node_text(fn.child_by_field_name("field"))
        receiver = fn.child_by_field_name("value")
        args = _rs_args(node)
        if not args:
            continue
        if method == "arg" and receiver is not None and receiver.type == "call_expression":
            cmd_args = _rs_args(receiver)
            word = (_rs_string(cmd_args[0]) or "").upper() if cmd_args else ""
            if (
                not node_text(receiver.child_by_field_name("function")).endswith("cmd")
                or word not in _RS_CMDS
            ):
                continue
            role, kind = _RS_CMDS[word]
            channels = [_rs_channel(args[0], consts, glob=word == "PSUBSCRIBE")]
            via = f"redis-rs.cmd({word})"
        elif method in _RS_METHODS:
            role, kind = _RS_METHODS[method]
            keys = args[2] if method == "xreadgroup" and len(args) > 2 else args[0]
            array = _deref(keys)
            elements = (
                array.named_children
                if array is not None and array.type == "array_expression"
                else [keys]
            )
            channels = [_rs_channel(k, consts, glob=method == "psubscribe") for k in elements]
            via = f"redis-rs.{method}"
        else:
            continue
        for ch in channels:
            out.append(
                MessageEndpoint(
                    repo=repo,
                    file=path,
                    line=node.start_point[0] + 1,
                    role=role,
                    kind=kind,
                    channel=ch.name,
                    channel_kind=ch.kind,
                    via=via,
                    symbol=symbol,
                    reason=ch.reason,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Extraction + matching
# ---------------------------------------------------------------------------


def extract_messaging(repo: str, files: dict[str, str]) -> RepoMessaging:
    """Publishers and consumers of ONE repo's ``.py`` / ``.rs`` files."""
    endpoints: list[MessageEndpoint] = []
    rq_targets: list[MessageEndpoint] = []
    unparsed: dict[str, str] = {}
    for path in sorted(files):
        try:
            if path.endswith(".py"):
                py = _PyFile(repo, path, ast.parse(files[path], filename=path), files)
                py.scan()
                endpoints.extend(py.out)
                rq_targets.extend(py.rq_targets())
            elif path.endswith(".rs"):
                endpoints.extend(_rust(repo, path, files[path]))
        except (SyntaxError, ValueError) as exc:
            unparsed[path] = f"{type(exc).__name__}: {exc}"
    return RepoMessaging(endpoints, rq_targets, unparsed)


def _match(pub: MessageEndpoint, con: MessageEndpoint) -> str | None:
    """``exact`` / ``pattern`` when *con* receives what *pub* sends, else None."""
    if con.kind != pub.kind or con.channel_kind == "unresolved":
        return None
    if pub.kind == "task":
        # Same framework (celery / rq / arq) and the very name the task is registered
        # under; a bare or suffix-only name is not evidence of the same task.
        same = pub.via.split(".", 1)[0] == con.via.split(".", 1)[0]
        return "exact" if same and pub.channel == con.channel else None
    if pub.channel_kind == "exact" and con.channel_kind == "exact":
        return "exact" if pub.channel == con.channel else None
    if pub.channel_kind == "exact":  # consumer glob
        return "pattern" if fnmatch.fnmatchcase(pub.channel, con.channel) else None
    prefix = pub.channel.rstrip("*")
    if con.channel_kind == "exact":
        return "pattern" if con.channel.startswith(prefix) else None
    other = re.split(r"[*?\[]", con.channel, maxsplit=1)[0]
    return "pattern" if prefix.startswith(other) or other.startswith(prefix) else None


def _edge(
    pub: MessageEndpoint, match: str, con: MessageEndpoint | None = None, reason: str | None = None
) -> MessageEdge:
    return MessageEdge(
        from_repo=pub.repo,
        from_file=pub.file,
        from_line=pub.line,
        kind=pub.kind,
        from_channel=pub.channel,
        match=match,
        from_via=pub.via,
        from_symbol=pub.symbol,
        to_repo=con.repo if con else None,
        to_file=con.file if con else None,
        to_line=con.line if con else None,
        to_symbol=con.symbol if con else None,
        to_channel=con.channel if con else None,
        to_via=con.via if con else None,
        reason=reason,
    )


def match_messages(per_repo: dict[str, RepoMessaging]) -> MessageGraph:
    endpoints = [e for r in per_repo.values() for e in r.endpoints]
    consumers = [e for e in endpoints if e.role == "consumer"]
    rq_targets = [t for r in per_repo.values() for t in r.rq_targets]
    edges: list[MessageEdge] = []
    unmatched_pubs: list[MessageEndpoint] = []
    reached: set[int] = set()
    for pub in (e for e in endpoints if e.role == "publisher"):
        if pub.channel_kind == "unresolved":
            edges.append(_edge(pub, "unresolved", reason=pub.reason))
            continue
        pool = rq_targets if pub.via.startswith("rq.") else consumers
        hits = [(con, how) for con in pool if (how := _match(pub, con)) is not None]
        if not hits:
            unmatched_pubs.append(pub)
        for con, how in hits:
            reached.add(id(con))
            edges.append(_edge(pub, how, con))
    edges.sort(
        key=lambda e: (
            e.from_repo,
            e.from_file,
            e.from_line,
            e.to_repo or "",
            e.to_file or "",
            e.to_line or 0,
        )
    )
    order = lambda e: (e.repo, e.file, e.line, e.channel)  # noqa: E731
    return MessageGraph(
        edges,
        sorted(unmatched_pubs, key=order),
        sorted((c for c in consumers if id(c) not in reached), key=order),
    )
