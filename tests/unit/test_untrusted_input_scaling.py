"""E5: every regex / parser this service runs on tenant input stays fast on large input.

Each entry of :data:`SCANNERS` is one function that reads text a tenant wrote (a
repository file, a manifest, tool output from the tenant's workspace pod) and one
ADVERSARIAL input shape for it. The test runs the function once at a large size
(4N) and asserts its CPU time (``time.process_time``, cyclic GC paused) stays under
one generous absolute budget, :data:`BUDGET_S` (>= 10x the slowest local
measurement, 0.53 s): a quadratic shape at this size takes minutes.

Inventory (the py2rust PR's tenant-input readers): ``rust_http`` (functions /
attributes, cfg predicates, router chains, reqwest call sites, format! targets),
``base_source`` (binding index), ``call_sites`` (Python), ``provider_routes``,
``structure.entrypoints`` / ``db_tables`` /
``module_graph`` / ``python_facts`` / ``rust_facts`` / ``service_graph`` /
``text``, ``container_fs`` (find output), the
Rust gate parsers (clippy / rustfmt / cargo-audit / cargo metadata), ``graph_sccs``
and ``workspace_repos``. Not listed, with the reason: anchored matches on short
service-built strings (``_URLISH`` on a rendered target, ``_PIN``, ``_TEST_ATTR``
paths); ``json.loads`` of tool output, which is C and linear (its nesting is what
``RecursionError`` guards).
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from typing import Any

import pytest

from models.contract_analysis_models import MAX_FILE_BYTES

#: Default N: characters per input (per-file readers get at most MAX_FILE_BYTES).
N = 64 * 1024
#: CPU seconds one scanner may take on its 4N input (slowest measured locally: 0.53 s).
BUDGET_S = 6.0


def cpu(fn: Callable[[Any], Any], arg: Any) -> float:
    """CPU time of one ``fn(arg)``; the cyclic GC is paused while timing (its full
    collections grow with the live heap: the interpreter's term, not ours)."""
    gc.collect()
    gc.disable()
    start = time.process_time()
    try:
        fn(arg)
    except (ValueError, RecursionError, SyntaxError):  # refusing hostile input IS the answer
        pass
    finally:
        elapsed = time.process_time() - start
        gc.enable()
    return elapsed


def _rep(unit: str, head: str = "", tail: str = "") -> Callable[[int], str]:
    return lambda n: head + unit * max(1, n // len(unit)) + tail


# ----------------------------------------------------------------------------- the readers

from processing.architecture.container_fs import safe_relative  # noqa: E402
from processing.architecture.tool_execution import _rust as gate  # noqa: E402
from processing.contract_analysis import call_sites as cs  # noqa: E402
from processing.contract_analysis import rust_http as rh  # noqa: E402
from processing.contract_analysis.base_source import BaseClassifier, binding_index  # noqa: E402
from processing.contract_analysis.provider_routes import extract_provider_routes  # noqa: E402
from processing.graph_sccs import strongly_connected_components  # noqa: E402
from processing.structure.db_tables import table_inventory  # noqa: E402
from processing.structure.entrypoints import entrypoint_inventory  # noqa: E402
from processing.structure.module_graph import module_graph, python_source_roots  # noqa: E402
from processing.structure.python_facts import parse_python_file  # noqa: E402
from processing.structure.rust_facts import load_workspace, surface  # noqa: E402
from processing.structure.service_graph import build_service_graph, split_by_repo  # noqa: E402
from processing.structure.text import clip  # noqa: E402
from processing.symbols import _parse  # noqa: E402
from processing.workspace_repos import repo_of_path  # noqa: E402


def _rust_fns(src: str) -> Any:
    tree = _parse("rust", src.encode())
    return rh.function_items(tree.root_node, include_tests=False)


def _axum(src: str) -> Any:
    return rh.extract_axum_routes({"src/main.rs": src})


def _rs_sites(src: str) -> Any:
    return cs.extract_file_call_sites("src/client.rs", src)


def _py_sites(src: str) -> Any:
    return cs.extract_file_call_sites("app/client.py", src)


def _classify_all(src: str) -> int:
    c = BaseClassifier(src, "rust")
    return sum(1 for i in range(200) if c.classify(f"v{i}", None))


def _files(prefix: str, count_unit: int, make: Callable[[int], tuple[str, str]]):
    """A generator of ``{path: text}`` workspaces whose total size is ~n characters."""

    def gen(n: int) -> dict[str, str]:
        return dict(make(i) for i in range(max(1, n // count_unit)))

    return gen


def _py_modules(n: int) -> dict[str, str]:
    k = max(2, n // 30)
    return {f"pkg/m{i}.py": f"import pkg.m{i + 1}\n" for i in range(k)} | {
        "pkg/__init__.py": "x=1\n"
    }


def _rs_modules(n: int) -> dict[str, str]:
    k = max(2, n // 30)
    lib = "".join(f"pub mod m{i};\n" for i in range(k))
    files = {f"src/m{i}.rs": f"pub fn f{i}() {{}}\n" for i in range(k)}
    return {"Cargo.toml": '[package]\nname = "x"\n', "src/lib.rs": lib, **files}


def _chain_graph(n: int) -> tuple[list[str], dict[str, list[str]]]:
    k = max(2, n // 8)
    nodes = [f"n{i}" for i in range(k)]
    adj = {f"n{i}": [f"n{(i + 1) % k}"] for i in range(k)}  # one big cycle
    return nodes, adj


def _clippy_lines(n: int) -> str:
    line = json.dumps(
        {
            "reason": "compiler-message",
            "manifest_path": "/w/Cargo.toml",
            "message": {
                "level": "warning",
                "message": "m",
                "code": {"code": "clippy::x"},
                "spans": [{"file_name": "src/a.rs", "line_start": 3, "is_primary": True}],
            },
        }
    )
    k = max(1, n // len(line))
    return "\n".join(line.replace('"line_start": 3', f'"line_start": {i + 1}') for i in range(k))


def _audit_report(n: int) -> str:
    vuln = {"advisory": {"id": "RUSTSEC-0000-0000", "title": "t"}, "package": {"name": "p"}}
    k = max(1, n // 80)
    return json.dumps({"database": {"advisory-count": 9}, "vulnerabilities": {"list": [vuln] * k}})


SCANNERS: list[tuple[str, Callable[[Any], Any], Callable[[int], Any], str, int]] = [
    # --- rust_http: attributes, cfg, chains, call sites, format! targets
    (
        "rust_http.function_items",
        _rust_fns,
        _rep("// c\n", "fn a() {}\n", "#[cfg(test)]\nfn y() {}\n"),
        "comment lines then a test item",
        N,
    ),
    (
        "rust_http.function_items",
        _rust_fns,
        _rep("#[inline]\n", "", "fn y() {}\n"),
        "attribute run on one item",
        N,
    ),
    (
        "rust_http.function_items",
        _rust_fns,
        _rep("fn a() {}\n// c\n"),
        "items interleaved with comments",
        N,
    ),
    (
        "rust_http.cfg_requires_test",
        rh.cfg_requires_test,
        _rep("a,", "all(", "test)"),
        "long all(..) argument list",
        N,
    ),
    ("rust_http.cfg_requires_test", rh.cfg_requires_test, _rep('"'), "unpaired quotes", N),
    (
        "rust_http.extract_axum_routes",
        _axum,
        _rep('.route("/r", get(h))', "fn app() -> Router { Router::new()", " }"),
        "one long router chain",
        N,
    ),
    (
        "rust_http.extract_axum_routes",
        _axum,
        _rep('.nest("/a", Router::new()', "fn app() -> Router { Router::new()", ")}"),
        "deep inline nest",
        4 * N,
    ),
    (
        "rust_http.extract_axum_routes",
        _axum,
        _rep("fn r() -> Router { Router::new().merge(r()) }\n"),
        "many router fns",
        N,
    ),
    (
        "rust_http.call_sites",
        _rs_sites,
        _rep(
            'client.get(format!("{}/x", base));\n',
            'fn f(client: &reqwest::Client) {\nlet base = std::env::var("B").unwrap();\n',
            "}",
        ),
        "many reqwest calls",
        N,
    ),
    (
        "rust_http.call_sites",
        _rs_sites,
        _rep("{}/", 'fn f(c: &reqwest::Client) { c.get(format!("', '", a)); }'),
        "format! with many holes",
        N,
    ),
    (
        "rust_http.call_sites",
        _rs_sites,
        _rep("x.", "fn f(client: &reqwest::Client) { client.get(", "y); }"),
        "long receiver target",
        N,
    ),
    # --- base_source
    (
        "base_source.binding_index",
        lambda s: binding_index(s, "rust"),
        _rep("pub("),
        "unclosed pub( on one line",
        N,
    ),
    (
        "base_source.binding_index",
        lambda s: binding_index(s, "rust"),
        _rep("let a = 1;\n"),
        "many bindings",
        N,
    ),
    (
        "base_source.binding_index",
        lambda s: binding_index(s, "python"),
        _rep("a"),
        "one long identifier line",
        N,
    ),
    (
        "base_source.binding_index",
        lambda s: binding_index(s, "python"),
        _rep("\n"),
        "blank lines",
        N,
    ),
    (
        "base_source.BaseClassifier",
        _classify_all,
        _rep('let v1 = env::var("X").unwrap();\n'),
        "200 lookups, one index",
        N,
    ),
    # --- call_sites (python) / provider routes
    (
        "call_sites.python",
        _py_sites,
        _rep("requests.get(f'{B}/x')\n", "import os, requests\nB = os.getenv('B')\n"),
        "many calls one base",
        N,
    ),
    (
        "provider_routes.flask",
        lambda s: extract_provider_routes({"app.py": s}),
        _rep(
            "@app.route('/r', methods=['GET'])\ndef h(): pass\n",
            "from flask import Flask\napp = Flask(__name__)\n",
        ),
        "many flask routes",
        N,
    ),
    (
        "provider_routes.axum",
        lambda s: extract_provider_routes({"src/main.rs": s}),
        _rep('.route("/r", get(h))', "fn app() -> Router { Router::new()", " }"),
        "long chain",
        N,
    ),
    # --- structure.entrypoints
    (
        "entrypoints.python",
        lambda s: entrypoint_inventory({"w.py": s}),
        _rep("def f():\n    r.subscribe('a')\n", "import redis\n"),
        "many fns with subscribers",
        N,
    ),
    (
        "entrypoints.python",
        lambda s: entrypoint_inventory({"w.py": s}),
        _rep("Consumer(x)\nWorker(q)\n", "import rq\n"),
        "many Consumer/Worker calls",
        N,
    ),
    (
        "entrypoints.python",
        lambda s: entrypoint_inventory({"w.py": s}),
        _rep("\n", "", "import nats\n"),
        "blank lines before an import",
        N,
    ),
    (
        "entrypoints.rust",
        lambda s: entrypoint_inventory({"src/main.rs": s}),
        _rep(".subscribe(x)", "fn f() { client", "; }"),
        "long method chain",
        N,
    ),
    (
        "entrypoints.rust",
        lambda s: entrypoint_inventory({"src/main.rs": s}),
        lambda n: "fn f() { " + "tokio::spawn(" * (n // 13) + "x" + ")" * (n // 13) + "; }",
        "nested spawn",
        N,
    ),
    (
        "entrypoints.rust",
        lambda s: entrypoint_inventory({"src/main.rs": s}),
        _rep("derive(", "#[", ")]\nstruct A;\n"),
        "derive( run in one attribute",
        N,
    ),
    (
        "entrypoints.rust",
        lambda s: entrypoint_inventory({"src/main.rs": s}),
        _rep("// c\n#[derive(Parser)]\nstruct A;\n"),
        "many clap structs",
        N,
    ),
    (
        "entrypoints.cargo",
        lambda s: entrypoint_inventory({"Cargo.toml": s}),
        _rep('[[bin]]\nname = "b"\npath = "src/b.rs"\n', '[package]\nname = "x"\n'),
        "many [[bin]] tables",
        N,
    ),
    (
        "entrypoints.pyproject",
        lambda s: entrypoint_inventory({"pyproject.toml": s}),
        lambda n: "[project.scripts]\n" + "".join(f'c{i} = "m:f"\n' for i in range(n // 12)),
        "many scripts",
        N,
    ),
    (
        "entrypoints.pyproject",
        lambda s: entrypoint_inventory({"pyproject.toml": s}),
        lambda n: (
            "[project]\nscripts = {" + ", ".join(f'c{i} = "m:f"' for i in range(n // 14)) + "}\n"
        ),
        "inline scripts table",
        N,
    ),
    # --- db_tables
    (
        "db_tables.inventory",
        lambda s: table_inventory({"src/db.rs": s}),
        _rep(".fetch(x)", 'fn f() { sqlx::query("SELECT 1 FROM t")', "; }"),
        "long sqlx chain",
        N,
    ),
    (
        "db_tables.inventory",
        lambda s: table_inventory({"m.sql": s}),
        _rep("CREATE TABLE t (id int);\n"),
        "many statements",
        N,
    ),
    # --- module graph / facts
    ("module_graph.python", lambda f: module_graph(f, "python"), _py_modules, "import chain", N),
    (
        "module_graph.python_source_roots",
        python_source_roots,
        _files("p", 40, lambda i: (f"s{i}/src/p/__init__.py", "x=1\n")),
        "many src layouts",
        N,
    ),
    ("module_graph.rust", lambda f: module_graph(f, "rust"), _rs_modules, "many modules", N),
    (
        "python_facts.parse",
        lambda s: parse_python_file("a.py", s),
        _rep("def f(a, b=1): pass\n"),
        "many defs",
        N,
    ),
    (
        "rust_facts.surface",
        lambda s: surface(load_workspace({"src/lib.rs": s})),
        _rep("/// d\n#[derive(Debug)]\n// c\npub struct S;\n"),
        "attributed items with comments",
        4 * N,  # 4N = MAX_FILE_BYTES
    ),
    (
        "rust_facts.surface",
        lambda s: surface(load_workspace({"src/lib.rs": s})),
        _rep("#[inline]\n// c\npub fn f() {}\n", "impl S {\n", "}\n"),
        "attributed impl fns",
        4 * N,
    ),
    # --- service graph / tenant data / text / workspace repos
    (
        "service_graph.build",
        lambda f: build_service_graph(split_by_repo(f)),
        _files(
            "s",
            60,
            lambda i: (f".repos/r{i}/app.py", "import requests\nrequests.get('http://svc/x')\n"),
        ),
        "many repos",
        N,
    ),
    ("text.clip", clip, _rep(" a"), "whitespace runs", N),
    (
        "workspace_repos.repo_of_path",
        lambda paths: [repo_of_path(p) for p in paths],
        lambda n: [f".repos/r{i}/a/b.py" for i in range(n // 20)],
        "many paths",
        N,
    ),
    (
        "container_fs.find_output",
        lambda s: [safe_relative(p, "/workspace") for p in s.splitlines()],
        _rep("/workspace/a/../b\n"),
        "many paths",
        N,
    ),
    ("graph_sccs", lambda g: strongly_connected_components(*g), _chain_graph, "one big cycle", N),
    # --- the Rust gate's parsers (tool output from the tenant pod)
    (
        "rust_gate.parse_clippy_json",
        lambda s: gate.parse_clippy_json(s, "/w", "/w"),
        _clippy_lines,
        "many diagnostics",
        N,
    ),
    (
        "rust_gate.parse_rustfmt_check",
        lambda s: gate.parse_rustfmt_check(s, "/w", "/w"),
        _rep("a:", "Diff in ", ":\n"),
        "colon run in one header",
        N,
    ),
    (
        "rust_gate.parse_rustfmt_check",
        lambda s: gate.parse_rustfmt_check(s, "/w", "/w"),
        _rep("Diff in /w/a.rs:1:\n"),
        "many hunks",
        N,
    ),
    (
        "rust_gate.parse_cargo_audit_json",
        lambda s: gate.parse_cargo_audit_json(s, "Cargo.lock"),
        _audit_report,
        "many vulnerabilities",
        N,
    ),
]


@pytest.mark.parametrize(
    "name, fn, gen, shape, n",
    SCANNERS,
    ids=[f"{s[0]}[{s[3]}]" for s in SCANNERS],
)
def test_scanner_stays_under_budget(name: str, fn, gen, shape: str, n: int) -> None:
    t = cpu(fn, gen(4 * n))
    print(f"E5 {name} [{shape}]: t(4N)={t:.4f}s")
    assert t < BUDGET_S, f"{name} [{shape}]: t(4N)={t:.3f}s"


def test_every_reader_module_is_inventoried() -> None:
    """Every module of the inventory has at least one entry (a new reader adds its own)."""
    names = {s[0].split(".")[0] for s in SCANNERS}
    assert {
        "rust_http", "base_source", "call_sites", "provider_routes", "entrypoints",
        "db_tables", "module_graph", "python_facts",
        "rust_facts", "service_graph", "text", "workspace_repos",
        "container_fs", "graph_sccs", "rust_gate",
    } <= names  # fmt: skip


def test_per_file_inputs_stay_under_the_file_cap() -> None:
    """4N of the default N is what one file may hold: the shapes above are real sizes."""
    assert 4 * N <= MAX_FILE_BYTES
