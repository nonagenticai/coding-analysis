"""Library-level facts on the py2rust multi-repo fixture (module graph, surface, inventories)."""

from __future__ import annotations

from pathlib import Path

import pytest

from processing.contract_analysis import call_sites as cs
from processing.contract_analysis.base_source import classify_base
from processing.contract_analysis.provider_routes import extract_provider_routes
from processing.structure.api_surface import api_surface
from processing.structure.db_tables import table_inventory
from processing.structure.entrypoints import entrypoint_inventory
from processing.structure.module_graph import condense, layers, module_graph
from processing.structure.service_graph import build_service_graph, split_by_repo

#: A small multi-repo py2rust workspace as real files: the primary repo is an axum
#: crate, ``.repos/acme__svc_a`` / ``.repos/acme__svc_b`` are FastAPI services.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "py2rust_workspace"
SVC_A_REPO = ".repos/acme__svc_a"
SVC_B_REPO = ".repos/acme__svc_b"


def workspace_files() -> dict[str, str]:
    """Workspace-root-relative paths -> content."""
    return {
        p.relative_to(FIXTURE).as_posix(): p.read_text()
        for p in sorted(FIXTURE.rglob("*"))
        if p.is_file()
    }


SVC_A = {
    path.removeprefix(f"{SVC_A_REPO}/"): text
    for path, text in workspace_files().items()
    if path.startswith(f"{SVC_A_REPO}/")
}


@pytest.fixture
def repos():
    return split_by_repo(workspace_files())


# --------------------------------------------------------------------------- module graph


def test_layers_hand_computed():
    sccs, order = layers(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "b"), ("d", "a")])
    assert sccs == [["b", "c"]]
    assert order == [["b", "c"], ["a"], ["d"]]


def test_python_module_graph(repos):
    g = module_graph(repos[SVC_A_REPO], "python")
    assert len(g.nodes) == 12
    assert sorted((e.source, e.target, e.kind) for e in g.edges) == [
        ("app.api", "app.client", "from_import"),
        ("app.cyc_a", "app.cyc_b", "from_import"),
        ("app.cyc_b", "app.cyc_a", "from_import"),
        ("app.main", "app.api", "from_import"),
        ("app.main", "app.worker", "from_import"),
        ("app.plugins", "app.tasks", "dynamic"),
        ("app.repo", "app.models", "from_import"),
    ]
    assert g.sccs == [["app.cyc_a", "app.cyc_b"]]
    assert g.order == [
        [
            "alembic.versions.0001_init",
            "app",
            "app.client",
            "app.cyc_a",
            "app.cyc_b",
            "app.models",
            "app.tasks",
            "app.worker",
        ],
        ["app.api", "app.plugins", "app.repo"],
        ["app.main"],
    ]
    assert [(u.file, u.expression) for u in g.unresolved] == [
        ("app/plugins.py", "importlib.import_module(name)")
    ]


def test_python_cycle_negative_control(repos):
    files = dict(repos[SVC_A_REPO])
    files["app/cyc_b.py"] = "Y = 2\n"  # break the planted cycle
    assert module_graph(files, "python").sccs == []


def test_rust_module_graph(repos):
    g = module_graph(repos["primary"], "rust")
    assert [n.module for n in g.nodes] == [
        "migrate",
        "tgt_svc",
        "tgt_svc::client",
        "tgt_svc::db",
        "tgt_svc::models",
        "tgt_svc::routes",
        "tgt_svc::routes::loops",
        "tgt_svc::routes::users",
    ]
    assert next(n for n in g.nodes if n.module == "tgt_svc::models").file == "src/gen/models.rs"
    assert len(g.edges) == 7
    assert g.sccs == [] and g.unresolved == []
    assert g.order[0] == ["migrate", "tgt_svc::client", "tgt_svc::models", "tgt_svc::routes::loops"]
    assert g.order[-1] == ["tgt_svc"]


def test_rust_missing_path_module_is_unresolved(repos):
    files = dict(repos["primary"])
    files["src/main.rs"] = files["src/main.rs"].replace("gen/models.rs", "gen/nope.rs")
    g = module_graph(files, "rust")
    reasons = sorted(u.reason.split(" (")[0] for u in g.unresolved)
    assert reasons == ["file is not reachable from any crate root", "module file not found"]
    assert "tgt_svc::models" not in [n.module for n in g.nodes]


#: a -> b -> c, d <-> e (a cycle), e -> c. Real source text through the real parser.
SLICE_TREE = {
    "a.py": "import b\n\n\ndef run():\n    return b.go()\n",
    "b.py": "# comment line is not LOC\nfrom c import C\n\n\ndef go():\n    return C\n",
    "c.py": "C = 1\n",
    "d.py": "import e\n\n\ndef dd():\n    return e.ee()\n",
    "e.py": "import c\n\n\ndef ee():\n    import d\n\n    return c.C, d\n",
}


def _slices(g):
    return [(s.id, s.layer, s.modules, s.cyclic, s.depends_on, s.loc) for s in g.slices]


def test_python_slices_one_per_scc_leaves_first():
    g = module_graph(SLICE_TREE, "python")
    loc = {n.module: n.loc for n in g.nodes}
    assert loc == {"a": 3, "b": 3, "c": 1, "d": 3, "e": 4}
    assert _slices(g) == [
        ("s0", 0, ["c"], False, [], 1),
        ("s1", 1, ["b"], False, ["s0"], 3),
        ("s2", 1, ["d", "e"], True, ["s0"], 7),
        ("s3", 2, ["a"], False, ["s1"], 3),
    ]
    # order / sccs are unchanged by the new field.
    assert g.sccs == [["d", "e"]]
    assert g.order == [["c"], ["b", "d", "e"], ["a"]]


def test_slices_negative_control_breaking_the_cycle_splits_the_slice():
    files = dict(SLICE_TREE, **{"e.py": "import c\n\n\ndef ee():\n    return c.C\n"})
    g = module_graph(files, "python")
    assert _slices(g) == [
        ("s0", 0, ["c"], False, [], 1),
        ("s1", 1, ["b"], False, ["s0"], 3),
        ("s2", 1, ["e"], False, ["s0"], 3),
        ("s3", 2, ["a"], False, ["s1"], 3),
        ("s4", 2, ["d"], False, ["s2"], 3),
    ]
    assert g.sccs == []


def test_slices_are_topologically_ordered_on_the_fixture(repos):
    for files, lang in ((repos[SVC_A_REPO], "python"), (repos["primary"], "rust")):
        g = module_graph(files, lang)
        seen: set[str] = set()
        for sl in g.slices:
            assert set(sl.depends_on) <= seen, (lang, sl)
            seen.add(sl.id)
        assert sorted(m for sl in g.slices for m in sl.modules) == [n.module for n in g.nodes]
        assert [
            sorted(m for sl in g.slices if sl.layer == k for m in sl.modules)
            for k in range(len(g.order))
        ] == g.order
    cyc = [sl for sl in module_graph(repos[SVC_A_REPO], "python").slices if sl.cyclic]
    assert [sl.modules for sl in cyc] == [["app.cyc_a", "app.cyc_b"]]


def test_rust_mod_cycle_is_one_slice():
    files = {
        "Cargo.toml": '[package]\nname = "cyc"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod base;\npub mod x;\npub mod y;\npub use x::X;\n",
        "src/base.rs": "pub struct Base;\n",
        "src/x.rs": "use crate::y::Y;\nuse crate::base::Base;\npub struct X(pub Option<Box<Y>>, pub Base);\n",
        "src/y.rs": "use crate::x::X;\npub struct Y(pub Option<Box<X>>);\n",
    }
    g = module_graph(files, "rust")
    assert g.sccs == [["cyc::x", "cyc::y"]]
    assert _slices(g) == [
        ("s0", 0, ["cyc::base"], False, [], 1),
        ("s1", 1, ["cyc::x", "cyc::y"], True, ["s0"], 5),
        ("s2", 2, ["cyc"], False, ["s1"], 4),
    ]


def test_self_dependency_is_one_cyclic_slice_with_no_dependencies():
    # A module that depends on itself is an SCC of one: ONE slice, cyclic, no deps.
    sccs, order, slices = condense(["x"], [("x", "x")], {"x": 2})
    assert sccs == [["x"]] and order == [["x"]]
    assert [(s.id, s.layer, s.modules, s.cyclic, s.depends_on, s.loc) for s in slices] == [
        ("s0", 0, ["x"], True, [], 2)
    ]


def test_python_self_import_is_one_slice_with_no_dependencies():
    # The Python extractor drops ``import x`` inside x.py as an edge (it is a no-op at
    # runtime), so the real tree yields ONE slice with no deps and no cycle.
    g = module_graph({"x.py": "import x\n\nX = 1\n"}, "python")
    assert g.edges == [] and g.sccs == []
    assert _slices(g) == [("s0", 0, ["x"], False, [], 2)]


def test_module_graph_response_carries_slices():
    from dataclasses import asdict

    from models.structure_models import ModuleGraphResponse

    g = module_graph(SLICE_TREE, "python")
    body = ModuleGraphResponse.model_validate(
        {**asdict(g), "files_read": 5, "repo": "primary"}
    ).model_dump(by_alias=True)
    assert body["slices"][2] == {
        "id": "s2",
        "layer": 1,
        "modules": ["d", "e"],
        "cyclic": True,
        "depends_on": ["s0"],
        "loc": 7,
    }
    assert body["order"] == g.order and body["sccs"] == g.sccs


# --------------------------------------------------------------------------- api surface


def test_python_surface_respects_all_and_underscore(repos):
    s = api_surface(repos[SVC_A_REPO], "python")
    assert len(s.symbols) == 24
    client = [x for x in s.symbols if x.module_path == "app.client"]
    assert [x.name for x in client] == ["SERVICE_B_URL", "fetch_item"]
    fetch = client[1].signature
    assert fetch is not None and fetch.is_async and fetch.returns == "dict"
    assert [(p.name, p.type, p.default) for p in fetch.params] == [
        ("item_id", "int", None),
        ("timeout", "float", "5.0"),
    ]
    narrowed = dict(repos[SVC_A_REPO])
    narrowed["app/client.py"] = SVC_A["app/client.py"].replace(
        '["fetch_item", "SERVICE_B_URL"]', '["fetch_item", "gone"]'
    )
    n = api_surface(narrowed, "python")
    assert [x.name for x in n.symbols if x.module_path == "app.client"] == ["fetch_item"]
    assert [(u.name, u.reason) for u in n.unresolved] == [
        ("gone", "name listed in __all__ is not bound in the module")
    ]


def test_rust_surface_lists_pub_items_and_macro_items(repos):
    s = api_surface(repos["primary"], "rust")
    assert len(s.symbols) == 20
    assert [(x.name, x.kind) for x in s.symbols if x.kind == "macro"] == [("item_id!", "macro")]
    assert "tgt_svc::generated_ping" not in {x.id for x in s.symbols}
    item_out = next(x for x in s.symbols if x.id == "tgt_svc::routes::users::ItemOut")
    assert [(m.name, m.visibility) for m in item_out.members] == [
        ("id", "pub"),
        ("name", "private"),
    ]
    get_item = next(x for x in s.symbols if x.name == "get_item")
    assert get_item.signature is not None and get_item.signature.returns == "Item"
    assert sorted((u.macro or "") for u in s.unresolved) == [
        "#[tokio::main]",
        "derive(Debug, serde::Serialize)",
        "macro_rules! item_id",
        "macro_rules! make_handler",
        "make_handler!",
    ]


def test_rust_impl_and_trait_methods():
    src = (
        "pub trait Tr { fn m(&self, a: u8) -> u8; }\n"
        "pub struct S;\n"
        "impl S { pub fn new() -> Self { S } fn hidden(&self) {} }\n"
        "impl Tr for S { fn m(&self, a: u8) -> u8 { a } }\n"
    )
    s = api_surface({"src/lib.rs": src}, "rust")
    kinds = sorted((x.kind, x.name, x.visibility) for x in s.symbols)
    assert kinds == [
        ("impl_fn", "m", "pub"),  # as public as the trait; kind says it is an impl
        ("method", "m", "pub"),
        ("method", "new", "pub"),
        ("struct", "S", "pub"),
        ("trait", "Tr", "pub"),
    ]


def test_rust_one_visibility_vocabulary_for_items_and_members():
    src = (
        "pub struct T(pub u8, u16, pub(crate) String);\n"
        "pub struct N { pub a: u8, b: u8, pub(super) c: u8 }\n"
        "pub enum E { A, B(u8) }\n"
    )
    s = api_surface({"src/lib.rs": src}, "rust")
    by = {x.name: x for x in s.symbols}
    assert [(m.name, m.visibility) for m in by["T"].members] == [
        ("0", "pub"),
        ("1", "private"),
        ("2", "pub(crate)"),
    ]
    assert [m.visibility for m in by["N"].members] == ["pub", "private", "pub(super)"]
    assert {m.visibility for m in by["E"].members} == {"pub"}
    vocabulary = {m.visibility for x in s.symbols for m in x.members} | {
        x.visibility for x in s.symbols
    }
    assert all(v == "private" or v.startswith("pub") for v in vocabulary)


# --------------------------------------------------------------------------- inventories


def test_tables(repos):
    a = table_inventory(repos[SVC_A_REPO])
    orders = next(t for t in a.tables if t.table == "orders")
    assert [(r.file, r.via) for r in orders.defined_in] == [
        ("alembic/versions/0001_init.py", "alembic:create_table"),
        ("app/models.py", "orm_model"),
    ]
    assert [r.via for r in orders.altered_in] == ["alembic:add_column"]
    assert [r.via for r in orders.read_by] == ["orm:select"]
    assert sorted(r.via for r in orders.written_by) == ["orm:session.add", "sql"]
    assert [t.table for t in a.tables] == ["audit_log", "orders"]
    assert [u.reason for u in a.unresolved] == ["table name is interpolated"]
    rust = table_inventory(repos["primary"])
    assert [
        (t.table, len(t.defined_in), len(t.read_by), len(t.written_by)) for t in rust.tables
    ] == [
        ("items", 1, 1, 0),
        ("jobs", 1, 0, 1),
        ("sessions", 0, 0, 1),  # pool.execute("DELETE FROM sessions ...")
    ]


def test_tables_negative_control_docstrings_and_prose():
    src = '"""Delete from cache the old entries."""\nMSG = "please update the record"\n'
    assert table_inventory({"x.py": src}).tables == []


_PROSE = [
    "Delete from cache the old entries",
    "Update the user set to active",
    "Select a value from the list",
    "Select one from the list",
    "Create table failed for users",
    "Alter table metadata failed",
    "Drop table cache now",
    "Insert into the queue failed",
    "Truncate the log file",
    "update table metadata failed: %s",
    "Delete from orders failed.",
    "Copy the file from disk",
    "With the orders done, exit",
    "Merge into main when ready",
    "Delete from users failed",
    "Delete from users failed: %s",
    "delete from sessions done",
]


@pytest.mark.parametrize("prose", _PROSE)
def test_tables_negative_control_prose_in_log_and_error_strings(prose):
    src = f"import logging\nlog = logging.getLogger()\n\ndef f():\n    log.error({prose!r})\n"
    src += f"    raise RuntimeError(f{prose!r} + str(1))\n"
    inv = table_inventory({"x.py": src, "y.rs": f"fn f() {{ panic!({prose!r}); }}\n"})
    assert inv.tables == [] and inv.unresolved == []


_REAL_SQL = {
    "SELECT id, name FROM users WHERE id = %s": ("users", "read_by"),
    "select * from users u join orders o on o.uid = u.id": ("orders", "read_by"),
    "SELECT count(*) AS n FROM audit": ("audit", "read_by"),
    "SELECT 1 FROM items LIMIT 1": ("items", "read_by"),
    "\n  -- fetch everything\n  /* block */\n  SELECT *\n    FROM ledger\n   WHERE x = 1\n": (
        "ledger",
        "read_by",
    ),
    "WITH recent AS (SELECT * FROM events) SELECT * FROM recent": ("events", "read_by"),
    "INSERT INTO jobs (id) VALUES ($1)": ("jobs", "written_by"),
    "insert into jobs select * from staging": ("staging", "read_by"),
    "UPDATE accounts SET balance = 0 WHERE id = 1": ("accounts", "written_by"),
    "DELETE FROM sessions WHERE expired": ("sessions", "written_by"),
    "DELETE FROM sessions": ("sessions", "written_by"),
    "DELETE FROM sessions s WHERE s.expired": ("sessions", "written_by"),
    "DELETE FROM sessions AS s": ("sessions", "written_by"),
    "CREATE TABLE IF NOT EXISTS widgets (id int)": ("widgets", "defined_in"),
    "ALTER TABLE widgets ADD COLUMN x int": ("widgets", "altered_in"),
    "DROP TABLE IF EXISTS widgets, gadgets CASCADE": ("widgets", "altered_in"),
    "TRUNCATE TABLE queue RESTART IDENTITY": ("queue", "written_by"),
    "CREATE UNIQUE INDEX ix ON widgets (id)": ("widgets", "altered_in"),
    "MERGE INTO stock s USING incoming i ON s.id = i.id": ("stock", "written_by"),
}


@pytest.mark.parametrize("sql", list(_REAL_SQL))
def test_tables_real_sql_still_detected(sql):
    table, role = _REAL_SQL[sql]
    inv = table_inventory({"x.py": f"def f(conn):\n    conn.execute({sql!r})\n"})
    found = {t.table: t for t in inv.tables}
    assert table in found and getattr(found[table], role), (sql, sorted(found))
    rust = table_inventory({"y.rs": f"fn f() {{ sqlx::query({_rust_str(sql)}); }}\n"})
    assert table in {t.table for t in rust.tables}


def _rust_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def test_tables_schema_is_split_off_and_kept():
    src = (
        "def f(conn):\n"
        '    conn.execute("SELECT * FROM billing.invoices WHERE id = 1")\n'
        '    conn.execute("INSERT INTO invoices (id) VALUES (1)")\n'
        '    conn.execute(\'UPDATE "Public"."Invoices" SET paid = true WHERE id = 1\')\n'
    )
    (invoices,) = table_inventory({"x.py": src}).tables
    assert invoices.table == "invoices"
    assert invoices.schemas == ["billing", "public"]
    assert [(r.schema, r.line) for r in invoices.read_by] == [("billing", 2)]
    assert [(r.schema, r.line) for r in invoices.written_by] == [(None, 3), ("public", 4)]


def test_tables_annotated_tablename_is_a_model():
    src = (
        "from sqlalchemy.orm import Mapped\n"
        "class Invoice(Base):\n"
        '    __tablename__: str = "invoices"\n'
        "def f(session):\n"
        "    return session.query(Invoice)\n"
    )
    (invoices,) = table_inventory({"m.py": src}).tables
    assert [(r.via, r.statement) for r in invoices.defined_in] == [("orm_model", "Invoice")]
    assert [r.via for r in invoices.read_by] == ["orm:query"]
    computed = "class X(Base):\n    __tablename__: str = name()\n"
    assert [u.reason for u in table_inventory({"c.py": computed}).unresolved] == [
        "__tablename__ is computed"
    ]


def test_tables_interpolated_table_name_is_unresolved():
    src = 'def f(conn, t):\n    conn.execute(f"""\n        -- purge\n        DELETE FROM {t} WHERE x\n    """)\n'
    inv = table_inventory({"x.py": src})
    assert inv.tables == []
    assert [(u.reason, u.line) for u in inv.unresolved] == [("table name is interpolated", 4)]


def test_entrypoints(repos):
    a = entrypoint_inventory(repos[SVC_A_REPO]).entrypoints
    assert sorted((e.kind, e.name) for e in a) == [
        ("background_task", "consume_events"),
        ("celery_task", "send_email"),
        ("console_script", "svc-a"),
        ("lifespan", "lifespan"),
        ("main_block", "main"),
        ("redis_stream_consumer", "consume_events"),
        ("websocket_endpoint", "ws"),
    ]
    rust = entrypoint_inventory(repos["primary"]).entrypoints
    assert sorted(e.kind for e in rust) == [
        "async_main",
        "background_task",
        "bin_target",
        "binary",
        "binary",
    ]
    files = dict(repos[SVC_A_REPO])
    files["app/main.py"] = files["app/main.py"].replace("FastAPI(lifespan=lifespan)", "FastAPI()")
    kinds = {e.kind for e in entrypoint_inventory(files).entrypoints}
    assert "lifespan" not in kinds and "background_task" not in kinds


# --------------------------------------------------------------------------- HTTP facts


def test_axum_routes_nested_across_files(repos):
    rust = {p: s for p, s in repos["primary"].items() if p.endswith(".rs")}
    r = extract_provider_routes(rust)
    assert sorted((x.method, x.path, x.handler) for x in r.routes) == [
        ("ALL", "/assets", 'tower_http::services::ServeDir::new("assets")'),  # nest_service
        ("DELETE", "/api/items/{param}", "users::delete_item"),
        ("GET", "/a/b/c", "routes::health"),  # .nest("/a", Router::new().nest("/b", leaf()))
        ("GET", "/api/items/{param}", "users::get_item"),
        ("GET", "/health", "routes::health"),
        ("GET", "/layered", "ping"),  # let app = app.route(..): followed once
        ("GET", "/layered/more", "ping"),
        ("GET", "/loop/x", "ping"),  # loops::a() -> b() -> a(): emitted once
        ("POST", "/api/v1/jobs/{param}", "create_job"),
    ]
    assert {x.framework for x in r.routes} == {"axum"}
    assert [x.path for x in r.routes if x.wildcard] == ["/assets"]
    assert [(u.file, u.line, u.reason) for u in r.unresolved] == [
        ("src/routes/loops.rs", 11, "router nesting cycle or too deep")
    ]


def test_axum_dynamic_path_and_unknown_router_are_unresolved():
    src = (
        "fn app() -> Router { Router::new().route(path(), get(h)).merge(elsewhere::r())"
        '.nest("/x", some_var) }\n'
    )
    r = extract_provider_routes({"src/lib.rs": src})
    assert r.routes == []
    assert sorted(u.reason for u in r.unresolved) == [
        "dynamic route path",
        "merge target is not a router built in the scanned files",
        "nest target is not a router built in the scanned files",
    ]


def test_flask_routes():
    src = (
        "from flask import Flask\napp = Flask(__name__)\n\n"
        "@app.route('/a', methods=['GET', 'POST'])\ndef a():\n    return ''\n\n"
        "@app.route('/b')\ndef b():\n    return ''\n"
    )
    r = extract_provider_routes({"app.py": src})
    assert [(x.method, x.path, x.framework) for x in r.routes] == [
        ("GET", "/a", "flask"),
        ("POST", "/a", "flask"),
        ("GET", "/b", "flask"),
    ]


@pytest.mark.parametrize(
    "methods", ["METHODS", "['GET', verb]", "list_of_methods()", "['GET', 'FETCH']"]
)
def test_flask_non_literal_methods_are_unresolved_not_dropped(methods):
    src = (
        "from flask import Flask\napp = Flask(__name__)\n\n"
        f"@app.route('/dyn', methods={methods})\ndef d():\n    return ''\n\n"
        "@app.route('/ok')\ndef ok():\n    return ''\n"
    )
    r = extract_provider_routes({"app.py": src})
    assert [(x.method, x.path) for x in r.routes] == [("GET", "/ok")]
    assert [(u.file, u.line, u.expression, u.reason) for u in r.unresolved] == [
        ("app.py", 4, "/dyn", "route methods are not a literal list of HTTP methods")
    ]


def test_reqwest_call_sites(repos):
    res = cs.extract_file_call_sites("src/client.rs", repos["primary"]["src/client.rs"])
    assert [
        (s.method, s.target, s.base, s.base_source_kind, s.base_source_name) for s in res.call_sites
    ] == [("POST", "/api/orders/events/{param}", "base", "env", "SVC_A_URL")]
    # rebound(): `c` is also bound to a HashMap, so `c.get("key")` is no request.
    assert res.unresolved == []
    neg = cs.extract_file_call_sites(
        "src/x.rs", 'fn f(map: &HashMap<String, u8>) { map.get("key"); get(h).post(h2); }\n'
    )
    assert neg.call_sites == [] and neg.unresolved == []


def test_base_classifier_indexes_bindings_once_and_matches_one_off_classify():
    from processing.contract_analysis.base_source import BaseClassifier, binding_index

    rs = (
        'pub(crate) const BASE: &str = "http://svc-b:80";\n'
        '    let mut host = std::env::var("HOST").unwrap();\n'
        "let dup = 1;\nlet dup = 2;\n"
    )
    idx = binding_index(rs, "rust")
    assert idx["BASE"] == ['"http://svc-b:80";'] and len(idx["dup"]) == 2
    assert "mut" not in idx
    c = BaseClassifier(rs, "rust")
    assert c.classify("BASE", None).kind == "literal"
    assert (c.classify("host", None).kind, c.classify("host", None).name) == ("env", "HOST")
    assert c.classify("dup", None).kind == "unknown"  # bound twice: never guessed
    for base in ("BASE", "host", "dup", "settings.url", None):
        assert c.classify(base, "h") == classify_base(base, "h", rs, "rust")
    py = "self.url: str = os.getenv('SVC_URL')\nx == 1\n"
    assert BaseClassifier(py, "python").classify("url", None).name == "SVC_URL"
    assert "x" not in binding_index(py, "python")  # a comparison is not a binding


def test_python_base_sources():
    src = SVC_A["app/client.py"]
    res = cs.extract_file_call_sites("app/client.py", src)
    assert {
        (s.base_source_kind, s.base_source_name, s.base_source_default) for s in res.call_sites
    } == {("env", "SERVICE_B_URL", "http://svc-b:8002")}
    assert classify_base("settings.b_url", None, "", "python").kind == "config"
    assert classify_base("url", None, "url = a\nurl = b\n", "python").kind == "unknown"


# --------------------------------------------------------------------------- service graph


def test_service_graph(repos):
    g = build_service_graph(repos, {"SVC_A_URL": "acme__svc_a"})
    assert [
        (e.from_repo, e.method, e.from_target, e.to_repo, e.match, e.resolved_by) for e in g.edges
    ] == [
        (SVC_A_REPO, "GET", "/items/{param}", SVC_B_REPO, "template", "service_name"),
        (SVC_A_REPO, "POST", "/jobs", SVC_B_REPO, "exact", "service_name"),
        (SVC_A_REPO, "GET", "/nowhere", SVC_B_REPO, "unresolved", "service_name"),
        ("primary", "POST", "/api/orders/events/{param}", SVC_A_REPO, "template", "hint"),
    ]
    assert g.service_names == {"svc-b": SVC_B_REPO}


def test_service_graph_negative_controls(repos):
    # Hint to the wrong repo: the call is not matched to a route there.
    wrong = build_service_graph(repos, {"SVC_A_URL": "acme__svc_b"})
    rust_edge = next(e for e in wrong.edges if e.from_repo == "primary")
    assert (rust_edge.to_repo, rust_edge.match) == (SVC_B_REPO, "unresolved")
    # No hint: only svc_a has the route, so it is matched by route alone.
    plain = build_service_graph(repos)
    rust_edge = next(e for e in plain.edges if e.from_repo == "primary")
    assert (rust_edge.to_repo, rust_edge.resolved_by) == (SVC_A_REPO, "route_only")


def test_text_helpers_have_one_definition():
    from processing.contract_analysis import rust_http
    from processing.structure import entrypoints, rust_facts, text

    for module in (rust_http, entrypoints, rust_facts):
        assert module._t is text.node_text, module.__name__
    assert entrypoints._unparse is text.unparse
    assert text.clip("a   b\n c") == "a b c"
    assert text.clip("x" * 10, 5) == "xxxx…"
    assert text.unparse(None) is None


def test_route_only_match_across_many_repos_is_ambiguous_and_capped():
    files = {".repos/caller/app.py": "import requests\nrequests.get('/items')\n"}
    for i in range(14):
        files[f".repos/svc{i:02d}/main.py"] = (
            "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items')\ndef f(): pass\n"
        )
    files[".repos/other/main.py"] = (
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/orders')\ndef f(): pass\n"
    )
    g = build_service_graph(split_by_repo(files))
    (edge,) = [e for e in g.edges if e.from_repo == ".repos/caller"]
    assert edge.to_repo is None and len(edge.candidates) == 10
    assert ".repos/other" not in edge.candidates
    assert "more than 10" in (edge.reason or "")


@pytest.mark.asyncio
async def test_rust_signatures_of_the_primary_crate(tmp_path):
    from app.api.endpoints.static_analysis import _rust_signatures

    for rel, text in workspace_files().items():
        if not rel.startswith(".repos/"):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text(text)
    body = (await _rust_signatures("a", tmp_path, "tenant-a")).model_dump()
    assert body["total_functions"] == 17
    get_item = next(f for f in body["functions"] if f["name"] == "get_item")
    assert get_item["signature"] == "async fn get_item(id: i64) -> Item"
    assert get_item["complexity_metrics"] is None
    # Rust parameter kinds, not Python's positional_or_keyword.
    kinds = {p["kind"] for f in body["functions"] for p in f["parameters"]}
    assert kinds == {"self", "positional"}
    label = next(f for f in body["functions"] if f["name"] == "label")
    assert [(p["name"], p["kind"]) for p in label["parameters"]] == [("self", "self")]
    assert len(body["unresolved"]) == 5  # + macro_rules! item_id
