"""Non-HTTP entrypoints: everything that runs without an HTTP route calling it.

Each entry: ``kind``, ``name`` (the function / script / binary), ``file``,
``line`` and ``detail`` (trigger, topic, stream, schedule, target) when it is a
literal. Kinds:

Python
    ``lifespan`` (``FastAPI(lifespan=fn)``), ``startup_hook``/``shutdown_hook``
    (``@app.on_event``), ``background_task`` (``asyncio.create_task`` /
    ``ensure_future`` / ``loop.create_task`` / ``BackgroundTasks``-free task
    spawns inside a lifespan or startup hook), ``celery_task`` (``@app.task``,
    ``@shared_task``, ``@celery.task``), ``celery_beat`` (``beat_schedule``),
    ``rq_worker`` (``Worker(...)``), ``arq_function``/``arq_cron``
    (``WorkerSettings.functions``/``cron_jobs``), ``dramatiq_actor``,
    ``scheduled_job`` (APScheduler ``add_job``/``@scheduled_job``),
    ``redis_stream_consumer`` (``xread``/``xreadgroup``), ``redis_pubsub_subscriber``
    (``subscribe``/``psubscribe``), ``redis_list_consumer`` (``blpop``/``brpop``/``brpoplpush``),
    ``kafka_consumer``, ``nats_subscriber``, ``amqp_consumer``, ``websocket_endpoint``
    (``@x.websocket``), ``sse_endpoint`` (``EventSourceResponse`` or
    ``media_type="text/event-stream"``), ``main_block`` (``if __name__ ==
    "__main__"``), ``console_script`` (``[project.scripts]``,
    ``[project.gui-scripts]``, ``[tool.poetry.scripts]``).
Rust
    ``async_main`` (``#[tokio::main]``/``#[actix_web::main]``/``#[async_std::main]``),
    ``binary`` (``fn main`` of a bin target: a ``[[bin]]`` path, ``src/main.rs``,
    ``src/bin/<name>.rs`` or ``src/bin/<name>/main.rs``), ``bin_target`` (each
    ``[[bin]]`` table of a Cargo.toml, on the line of its own ``name`` key),
    ``background_task`` (``tokio::spawn``/``tokio::task::spawn``/
    ``spawn_blocking``/``std::thread::spawn`` anywhere),
    ``websocket_endpoint`` (a handler taking ``WebSocketUpgrade``), ``sse_endpoint``
    (``Sse<...>`` return / ``Sse::new``), ``cli`` (``#[derive(Parser)]``/
    ``Command::new``), ``scheduled_job`` (``tokio_cron_scheduler`` ``Job::new*``),
    plus the same consumer kinds as Python for ``xread``/``subscribe`` calls.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .rust_facts import default_bin_path, outer_attributes
from .text import CLIP_LIMIT, as_dict, as_list, as_str, clip, get_path
from .text import node_text as _t
from .text import unparse as _unparse


@dataclass
class Entrypoint:
    kind: str
    name: str | None
    file: str
    line: int
    detail: str | None = None


@dataclass
class EntrypointInventory:
    entrypoints: list[Entrypoint]
    files_unparsed: dict[str, str]


_TASK_SPAWNERS = {"create_task", "ensure_future", "start_soon", "run_in_executor"}
_REDIS_STREAM = {"xread", "xreadgroup"}
_REDIS_LIST = {"blpop", "brpop", "brpoplpush", "blmove", "bzpopmin", "bzpopmax"}
_SUBSCRIBE = {"subscribe", "psubscribe", "ssubscribe"}
_KAFKA_CLASSES = {"AIOKafkaConsumer", "KafkaConsumer", "Consumer"}


def _clip(text: str | None) -> str | None:
    return None if text is None else clip(text)


def _messaging_kind(source: str) -> str | None:
    if re.search(r"^[ \t]*(?:from|import)[ \t]+(?:nats)\b", source, re.M):
        return "nats"
    if re.search(r"^[ \t]*(?:from|import)[ \t]+(?:aio_pika|pika)\b", source, re.M):
        return "amqp"
    if re.search(r"^[ \t]*(?:from|import)[ \t]+(?:redis|aioredis)\b", source, re.M):
        return "redis"
    return None


def _func_name(call: ast.Call) -> tuple[str, str | None]:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        recv = _unparse(fn.value)
        return fn.attr, recv
    if isinstance(fn, ast.Name):
        return fn.id, None
    return "", None


def _str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _called_name(node: ast.AST | None) -> str | None:
    """``worker()`` / ``self.loop()`` / ``worker`` -> its name."""
    if isinstance(node, ast.Await):
        node = node.value
    if isinstance(node, ast.Call):
        node = node.func
    return _unparse(node)


def _python(path: str, source: str, out: list[Entrypoint]) -> None:
    tree = ast.parse(source, filename=path)
    messaging = _messaging_kind(source)
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    startup_fns: set[str] = set()
    # Whole-file facts computed ONCE, never per call node.
    imports_rq = re.search(r"^[ \t]*(?:from|import)[ \t]+rq\b", source, re.M) is not None
    mentions_kafka = "kafka" in source

    def add(kind: str, name: str | None, node: ast.AST, detail: str | None = None) -> None:
        out.append(Entrypoint(kind, name, path, getattr(node, "lineno", 1), _clip(detail)))

    for node, enclosing in _walk_with_enclosing(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for dec in node.decorator_list:
                call = dec if isinstance(dec, ast.Call) else None
                target = call.func if call is not None else dec
                text = _unparse(target) or ""
                last = text.rsplit(".", 1)[-1]
                first_arg = _str(call.args[0]) if call is not None and call.args else None
                if last == "on_event" and first_arg in ("startup", "shutdown"):
                    add(f"{first_arg}_hook", node.name, node)
                    if first_arg == "startup":
                        startup_fns.add(node.name)
                elif last == "websocket" or last == "websocket_route":
                    add("websocket_endpoint", node.name, node, first_arg)
                elif last == "shared_task" or (
                    last == "task" and re.search(r"celery|app|broker", text)
                ):
                    kw = {k.arg: _unparse(k.value) for k in call.keywords} if call else {}
                    add("celery_task", node.name, node, kw.get("name"))
                elif last == "actor" and "dramatiq" in text or text == "actor":
                    add("dramatiq_actor", node.name, node)
                elif last == "scheduled_job":
                    add("scheduled_job", node.name, node, _unparse(call) if call else None)
                elif last in ("periodic_task", "repeat_every", "cron"):
                    add("scheduled_job", node.name, node, _unparse(call) if call else text)
        elif isinstance(node, ast.If):
            test = _unparse(node.test) or ""
            if re.fullmatch(r"__name__ == ['\"]__main__['\"]|['\"]__main__['\"] == __name__", test):
                calls = [_called_name(s.value) for s in node.body if isinstance(s, ast.Expr)]
                add(
                    "main_block",
                    PurePosixPath(path).stem,
                    node,
                    ", ".join(c for c in calls if c) or None,
                )
        elif isinstance(node, ast.Call):
            name, recv = _func_name(node)
            kw = {k.arg: k.value for k in node.keywords}
            if name == "FastAPI" and "lifespan" in kw:
                fn = _unparse(kw["lifespan"])
                add("lifespan", fn, node)
                if fn:
                    startup_fns.add(fn)
            elif name in _REDIS_STREAM:
                streams = node.args[0] if node.args else kw.get("streams")
                if name == "xreadgroup" and len(node.args) >= 3:
                    streams = node.args[2]
                detail = None
                if isinstance(streams, ast.Dict):
                    detail = ", ".join(k for k in (_str(k) for k in streams.keys) if k) or None
                add(
                    "redis_stream_consumer",
                    enclosing,
                    node,
                    detail or _unparse(streams),
                )
            elif name in _REDIS_LIST:
                add(
                    "redis_list_consumer",
                    enclosing,
                    node,
                    _unparse(node.args[0]) if node.args else None,
                )
            elif name in _SUBSCRIBE:
                subject = _unparse(node.args[0]) if node.args else None
                kind = {
                    "nats": "nats_subscriber",
                    "amqp": "amqp_consumer",
                    "redis": "redis_pubsub_subscriber",
                }.get(messaging or "", None)
                if kind is None and recv and re.search(r"pubsub|redis", recv, re.I):
                    kind = "redis_pubsub_subscriber"
                if kind is None and recv and re.search(r"consumer|kafka", recv, re.I):
                    kind = "kafka_consumer"
                if kind is None and recv and re.search(r"\bnc\b|nats|\bjs\b", recv, re.I):
                    kind = "nats_subscriber"
                if kind is not None:
                    add(kind, enclosing, node, subject)
            elif name in _KAFKA_CLASSES and (name != "Consumer" or mentions_kafka):
                topics = ", ".join(t for t in (_str(a) for a in node.args) if t) or None
                add("kafka_consumer", enclosing, node, topics)
            elif name in ("consume", "basic_consume", "iterator") and messaging == "amqp":
                add("amqp_consumer", enclosing, node, _unparse(kw.get("queue")))
            elif name == "Worker" and imports_rq:
                add(
                    "rq_worker",
                    enclosing,
                    node,
                    _unparse(node.args[0]) if node.args else None,
                )
            elif name == "add_job":
                target = _called_name(node.args[0]) if node.args else _unparse(kw.get("func"))
                trigger = _str(node.args[1]) if len(node.args) > 1 else _unparse(kw.get("trigger"))
                add("scheduled_job", target, node, trigger)
            elif name in ("EventSourceResponse", "ServerSentEvent"):
                add("sse_endpoint", enclosing, node, name)
            elif name == "StreamingResponse" and _str(kw.get("media_type")) == "text/event-stream":
                add("sse_endpoint", enclosing, node, "text/event-stream")
        elif isinstance(node, ast.ClassDef) and node.name == "WorkerSettings":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or not isinstance(
                    stmt.value, ast.List | ast.Tuple
                ):
                    continue
                target = next((t.id for t in stmt.targets if isinstance(t, ast.Name)), "")
                for elt in stmt.value.elts:
                    if target == "functions":
                        add("arq_function", _unparse(elt), stmt)
                    elif target == "cron_jobs":
                        fn = (
                            _unparse(elt.args[0])
                            if isinstance(elt, ast.Call) and elt.args
                            else _unparse(elt)
                        )
                        add("arq_cron", fn, stmt, _unparse(elt))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                text = _unparse(target) or ""
                if text.endswith("beat_schedule") and isinstance(node.value, ast.Dict):
                    for key, value in zip(node.value.keys, node.value.values, strict=True):
                        task = None
                        if isinstance(value, ast.Dict):
                            for k, v in zip(value.keys, value.values, strict=True):
                                if _str(k) == "task":
                                    task = _str(v)
                        add("celery_beat", task or _str(key), node, _str(key))
    for fn_name in sorted(startup_fns):
        fn = funcs.get(fn_name.rsplit(".", 1)[-1])
        if fn is None:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name, _ = _func_name(node)
                if name in _TASK_SPAWNERS and node.args:
                    add(
                        "background_task",
                        _called_name(node.args[-1]),
                        node,
                        f"spawned in {fn.name}",
                    )


def _walk_with_enclosing(tree: ast.AST) -> list[tuple[ast.AST, str | None]]:
    """Every node with the name of its innermost enclosing ``def`` (None at module level).

    One iterative pass; looking the enclosing function up per call node instead
    (a scan of every function) cost O(calls x functions).
    """
    out: list[tuple[ast.AST, str | None]] = []
    stack: list[tuple[ast.AST, str | None]] = [(tree, None)]
    while stack:
        node, enclosing = stack.pop()
        out.append((node, enclosing))
        # Only a def's BODY is inside it; its decorators, defaults and annotations
        # run in the enclosing scope.
        inner, body = enclosing, set()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inner, body = node.name, {id(stmt) for stmt in node.body}
        stack.extend(
            (child, inner if id(child) in body else enclosing)
            for child in reversed(list(ast.iter_child_nodes(node)))
        )
    return out


def _pyproject(path: str, source: str, out: list[Entrypoint]) -> None:
    data = tomllib.loads(source)
    tables = [
        ("console_script", get_path(data, "project", "scripts")),
        ("gui_script", get_path(data, "project", "gui-scripts")),
        ("console_script", get_path(data, "tool", "poetry", "scripts")),
    ]
    key_lines = _key_lines(source)
    for kind, table in tables:
        for name, target in sorted(as_dict(table).items()):
            if isinstance(target, str):
                out.append(Entrypoint(kind, name, path, key_lines.get(name, 1), target[:300]))


_TOML_KEY_LINE = re.compile(r"""^[ \t]*(?:"([^"\n]*)"|'([^'\n]*)'|([A-Za-z0-9_.-]+))[ \t]*=""")


def _key_lines(source: str) -> dict[str, int]:
    """``{key: line of its first `key =` line}`` of a TOML file, in ONE pass.

    Searching the file once per key (a regex per script / bin) cost
    O(keys x file size) when the keys are written inline, never at a line start.
    """
    out: dict[str, int] = {}
    for lineno, line in enumerate(source.split("\n"), 1):
        m = _TOML_KEY_LINE.match(line)
        if m is not None:
            out.setdefault(m.group(1) or m.group(2) or m.group(3), lineno)
    return out


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def is_implicit_bin(path: str) -> bool:
    """``src/bin/<n>.rs`` or ``src/bin/<n>/main.rs`` (Cargo's auto-discovered binaries)."""
    p = PurePosixPath(path)
    if p.suffix != ".rs":
        return False
    if p.parent.name == "bin" and p.parent.parent.name == "src":
        return True
    return p.name == "main.rs" and p.parent.parent.name == "bin" and p.parents[2].name == "src"


def bin_name(path: str) -> str:
    """Binary name of a bin root: ``src/bin/x/main.rs`` -> x, ``src/bin/y.rs`` -> y."""
    p = PurePosixPath(path)
    if p.name == "main.rs" and p.parent.parent.name == "bin":
        return p.parent.name
    return p.stem


def _rust(path: str, source: str, bins: set[str], out: list[Entrypoint]) -> None:
    from processing.contract_analysis.rust_http import function_items

    content = source.encode("utf-8")
    tree = _parse("rust", content)
    if tree is None:
        raise ValueError("grammar not available")
    root = tree.root_node
    is_bin = path in bins
    for name, fn, attr_items in function_items(root):
        attrs = [_t(a) for a in attr_items]
        line = fn.start_point[0] + 1
        main_attr = next(
            (a for a in attrs if re.search(r"#\[\s*(tokio|actix_web|actix_rt|async_std)::main", a)),
            None,
        )
        if main_attr:
            out.append(Entrypoint("async_main", name, path, line, _clip(main_attr)))
        if name == "main" and is_bin:
            out.append(Entrypoint("binary", bin_name(path), path, line))
        params = _t(fn.child_by_field_name("parameters"))
        ret = _t(fn.child_by_field_name("return_type"))
        if "WebSocketUpgrade" in params:
            out.append(Entrypoint("websocket_endpoint", name, path, line))
        if re.search(r"\bSse\s*<", ret):
            out.append(Entrypoint("sse_endpoint", name, path, line))
        body = fn.child_by_field_name("body")
        stack = [body] if body is not None else []
        while stack:
            node = stack.pop()
            stack.extend(node.named_children)
            if node.type != "call_expression":
                continue
            callee = _callee_text(node.child_by_field_name("function"), content)
            short = callee.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
            nline = node.start_point[0] + 1
            if re.fullmatch(
                r"(tokio::(task::)?|std::thread::|thread::|task::|actix_rt::|async_std::task::)?(spawn|spawn_blocking|spawn_local)",
                callee,
            ):
                args = node.child_by_field_name("arguments")
                arg = args.named_children[0] if args is not None and args.named_children else None
                out.append(Entrypoint("background_task", name, path, nline, _head(arg, content)))
            elif callee.endswith("Sse::new"):
                out.append(Entrypoint("sse_endpoint", name, path, nline))
            elif short in ("xread", "xread_options", "xreadgroup") or callee.endswith(
                "xread_options"
            ):
                out.append(
                    Entrypoint("redis_stream_consumer", name, path, nline, _arg0(node, content))
                )
            elif short in ("subscribe", "psubscribe") and re.search(r"pubsub|redis", callee, re.I):
                out.append(
                    Entrypoint("redis_pubsub_subscriber", name, path, nline, _arg0(node, content))
                )
            elif short == "subscribe" and re.search(r"nats|\bnc\b|client", callee):
                out.append(Entrypoint("nats_subscriber", name, path, nline, _arg0(node, content)))
            elif short == "subscribe" and re.search(r"consumer|kafka", callee, re.I):
                out.append(Entrypoint("kafka_consumer", name, path, nline, _arg0(node, content)))
            elif re.search(r"Job::new(_async|_cron_job|_repeated)?(_tz)?$", callee):
                out.append(Entrypoint("scheduled_job", name, path, nline, _arg0(node, content)))
            elif callee in ("Command::new", "clap::Command::new"):
                out.append(Entrypoint("cli", name, path, nline, _arg0(node, content)))
    attrs_of = outer_attributes(root)
    for item in root.named_children:
        if any(_derives_parser(_t(a)) for a in attrs_of.get(item.id, [])):
            out.append(
                Entrypoint(
                    "cli",
                    _t(item.child_by_field_name("name")),
                    path,
                    item.start_point[0] + 1,
                    "clap::Parser",
                )
            )


_PARSER_WORD = re.compile(r"\bParser\b")


def _derives_parser(attr: str) -> bool:
    """``#[derive(.., Parser, ..)]``: each ``derive(`` group is scanned once (str.find),
    never re-scanned from every later ``derive(`` as ``derive\\([^)]*`` would."""
    pos = attr.find("derive(")
    while pos != -1:
        close = attr.find(")", pos)
        group = attr[pos + 7 : close if close != -1 else len(attr)]
        if _PARSER_WORD.search(group):
            return True
        if close == -1:
            return False
        pos = attr.find("derive(", close)
    return False


#: Longest callee suffix read per call: every pattern below matches a short path.
_CALLEE_TAIL = 200


def _callee_text(fn: Any, content: bytes) -> str:
    """The last :data:`_CALLEE_TAIL` bytes of a call's function expression.

    A method call's function is its whole receiver chain (``a.b().c().d``), so
    taking its full text for every call of a chain cost O(chain^2).
    """
    if fn is None:
        return ""
    start = max(fn.start_byte, fn.end_byte - _CALLEE_TAIL)
    return content[start : fn.end_byte].decode("utf-8", "replace")


def _head(node: Any, content: bytes) -> str | None:
    """Clipped text of *node*, reading at most a few clip-lengths of its bytes (nested
    calls would otherwise copy the same inner text once per nesting level)."""
    if node is None:
        return None
    end = min(node.end_byte, node.start_byte + 4 * CLIP_LIMIT)
    return _clip(content[node.start_byte : end].decode("utf-8", "replace"))


def _arg0(call: Any, content: bytes) -> str | None:
    args = call.child_by_field_name("arguments")
    arg = args.named_children[0] if args is not None and args.named_children else None
    return _head(arg, content)


_BIN_HEADER = re.compile(r"^[ \t]*\[\[[ \t]*bin[ \t]*\]\]")
_TABLE_HEADER = re.compile(r"^[ \t]*\[")
_NAME_KEY = re.compile(r"^[ \t]*name[ \t]*=")


def _bin_name_lines(source: str) -> list[int]:
    """Line of the ``name`` key of each ``[[bin]]`` table, in order (header line if none).

    One pass over the lines with a running line number; counting newlines from
    the start of the file once per ``[[bin]]`` cost O(bins x file size).
    """
    lines: list[int] = []
    in_bin = False  # inside a [[bin]] table whose name line is not found yet
    for lineno, line in enumerate(source.split("\n"), 1):
        if _BIN_HEADER.match(line):
            lines.append(lineno)  # header line until a name key is seen
            in_bin = True
        elif _TABLE_HEADER.match(line):
            in_bin = False
        elif in_bin and _NAME_KEY.match(line):
            lines[-1] = lineno
            in_bin = False
    return lines


def _cargo(path: str, source: str, out: list[Entrypoint], files: dict[str, str]) -> set[str]:
    """``[[bin]]`` targets as entries; returns every bin root file (incl. implicit ones)."""
    data = tomllib.loads(source)
    pkg_dir = PurePosixPath(path).parent
    base = "" if str(pkg_dir) == "." else str(pkg_dir)
    roots: set[str] = set()
    package = as_dict(data.get("package"))
    name_lines = _bin_name_lines(source)
    key_lines: dict[str, int] | None = None
    for index, entry in enumerate(as_list(data.get("bin"))):
        name = as_str(get_path(entry, "name"))
        if not name:
            continue
        rel = as_str(entry.get("path")) or (
            "src/main.rs"
            if name == as_str(package.get("name"))
            else default_bin_path(name, files, base)
        )
        roots.add(f"{base}/{rel}" if base else rel)
        if index < len(name_lines):
            line = name_lines[index]
        else:
            if key_lines is None:
                key_lines = _key_lines(source)
            line = key_lines.get("name", 1)
        out.append(Entrypoint("bin_target", name, path, line, rel))
    if package:
        main = pkg_dir / "src/main.rs"
        roots.add(str(main) if str(pkg_dir) != "." else "src/main.rs")
    return roots


def entrypoint_inventory(files: dict[str, str]) -> EntrypointInventory:
    """Non-HTTP entrypoints of *files* (``.py``, ``.rs``, ``pyproject.toml``, ``Cargo.toml``)."""
    out: list[Entrypoint] = []
    unparsed: dict[str, str] = {}
    bins: set[str] = set()
    for path in sorted(files):
        name = PurePosixPath(path).name
        try:
            if name == "pyproject.toml":
                _pyproject(path, files[path], out)
            elif name == "Cargo.toml":
                bins |= _cargo(path, files[path], out, files)
        except tomllib.TOMLDecodeError as exc:
            unparsed[path] = str(exc)
        except (TypeError, AttributeError, ValueError, RecursionError) as exc:
            unparsed[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    for path in sorted(files):
        if is_implicit_bin(path):
            bins.add(path)
    for path in sorted(files):
        try:
            if path.endswith(".py"):
                _python(path, files[path], out)
            elif path.endswith(".rs"):
                _rust(path, files[path], bins, out)
        except (SyntaxError, ValueError, RecursionError) as exc:
            unparsed[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out.sort(key=lambda e: (e.file, e.line, e.kind, e.name or ""))
    return EntrypointInventory(entrypoints=out, files_unparsed=unparsed)
