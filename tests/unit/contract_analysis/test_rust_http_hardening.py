"""reqwest call sites and axum routes: format! escapes, receivers, sibling routers, tests, budget."""

from __future__ import annotations

import pytest

from processing.contract_analysis import call_sites as cs
from processing.contract_analysis import rust_http as rh
from processing.contract_analysis.provider_routes import extract_provider_routes
from processing.contract_analysis.rust_http import extract_axum_routes, file_module
from processing.symbols import _parse


def _sites(src: str, path: str = "src/x.rs"):
    return cs.extract_file_call_sites(path, src)


# --------------------------------------------------------------------------- format! escapes


def _format_parts(macro: str) -> list[str]:
    tree = _parse("rust", f"fn f() {{ let _ = {macro}; }}".encode())
    node = next(n for n in rh._walk(tree.root_node) if n.type == "macro_invocation")
    return [p if isinstance(p, str) else f"<{p.expr}>" for p in rh._resolve(node, rh._Scope({}))]


def test_format_escaped_braces_keep_literal_offsets():
    assert _format_parts('format!("/x/{{a}}/{}/tail", b)') == ["/x/{a}/", "<b>", "/tail"]


def test_format_escaped_braces_before_and_after_a_hole():
    assert _format_parts('format!("/{{b}}/{id}/{{c}}/z")') == ["/{b}/", "<id>", "/{c}/z"]
    assert _format_parts('format!("{{}}{}", x)') == ["{}", "<x>"]


def test_format_host_hole_is_a_base_not_a_literal_host():
    src = (
        "use reqwest::Client;\n"
        "fn f(client: &Client) {\n"
        '    let host = std::env::var("SVC_HOST").unwrap();\n'
        '    client.get(format!("http://{}/items", host));\n'
        "}\n"
    )
    (site,) = _sites(src).call_sites
    assert site.host is None
    assert site.base == "host"
    assert site.base_source_kind == "env" and site.base_source_name == "SVC_HOST"
    assert site.confidence == "exact"


def test_format_host_and_port_holes_are_unknown_not_literal():
    src = (
        "use reqwest::Client;\n"
        'fn f(client: &Client, h: &str, p: u16) { client.get(format!("http://{}:{}/x", h, p)); }\n'
    )
    (site,) = _sites(src).call_sites
    assert site.base_source_kind == "unknown" and site.base == "{h}:{p}"


def test_literal_host_is_still_literal():
    (site,) = _sites(
        'fn f(client: &reqwest::Client) { client.get("http://svc-b:8080/x"); }\n'
    ).call_sites
    assert site.base_source_kind == "literal" and site.host == "svc-b:8080"


# --------------------------------------------------------------------------- receivers


def test_client_constructor_receiver_is_a_call_site():
    res = _sites('fn f() { reqwest::Client::new().get("http://svc/x"); }\n')
    assert [(s.method, s.target) for s in res.call_sites] == [("GET", "/x")]


def test_let_bound_client_with_unresolvable_target_is_unresolved_not_dropped():
    src = (
        "use reqwest::Client;\n"
        "fn f(u: String) {\n"
        "    let c = Client::builder().build().unwrap();\n"
        "    c.post(u.to_lowercase());\n"
        "}\n"
    )
    res = _sites(src)
    assert res.call_sites == []
    assert [(u.method, u.client) for u in res.unresolved] == [("POST", "reqwest")]


def test_client_new_receiver_with_dynamic_target_is_unresolved():
    res = _sites("fn f(u: String) { reqwest::Client::new().get(u.trim()); }\n")
    assert [u.method for u in res.unresolved] == ["GET"]


def test_typed_client_param_is_a_known_receiver():
    src = "use reqwest::Client;\nfn f(c: &Client, u: String) { c.delete(u.trim()); }\n"
    assert [u.method for u in _sites(src).unresolved] == ["DELETE"]


def test_non_client_receivers_are_still_ignored():
    src = (
        "use std::collections::HashMap;\n"
        "fn f(c: &HashMap<String, u8>, k: String) { let m = HashMap::new(); m.get(&k); c.get(&k); }\n"
    )
    res = _sites(src)
    assert res.call_sites == [] and res.unresolved == []


def test_bare_client_new_without_reqwest_is_not_a_client():
    res = _sites("fn f(k: String) { Client::new().get(k.trim()); }\n")
    assert res.call_sites == [] and res.unresolved == []


# --------------------------------------------------------------------------- sibling routers

_MAIN = (
    "mod users;\nmod orders;\n"
    "fn app() -> Router {\n"
    '    Router::new().nest("/users", users::routes()).nest("/orders", orders::routes())\n'
    "}\n"
)
_USERS = 'pub fn routes() -> Router { Router::new().route("/", get(list_users)) }\n'
_ORDERS = 'pub fn routes() -> Router { Router::new().route("/", get(list_orders)) }\n'


def test_same_named_router_fns_in_sibling_modules_are_both_mounted():
    r = extract_axum_routes(
        {"src/main.rs": _MAIN, "src/users.rs": _USERS, "src/orders/mod.rs": _ORDERS}
    )
    assert sorted((x.path, x.handler) for x in r.routes) == [
        ("/orders", "list_orders"),
        ("/users", "list_users"),
    ]
    assert r.unresolved == []


def test_router_fn_imported_by_use_is_told_apart():
    main = 'use crate::users::routes;\nfn app() -> Router { Router::new().nest("/u", routes()) }\n'
    r = extract_axum_routes({"src/main.rs": main, "src/users.rs": _USERS, "src/orders.rs": _ORDERS})
    assert ("/u", "list_users") in [(x.path, x.handler) for x in r.routes]
    assert ("/u", "list_orders") not in [(x.path, x.handler) for x in r.routes]


def test_unqualified_ambiguous_router_fn_is_unresolved():
    main = 'fn app() -> Router { Router::new().nest("/u", routes()) }\n'
    r = extract_axum_routes({"src/main.rs": main, "src/users.rs": _USERS, "src/orders.rs": _ORDERS})
    assert any("ambiguous" in u.reason for u in r.unresolved)
    assert not any(x.path.startswith("/u") for x in r.routes)


def test_file_module():
    assert file_module("src/api/users/mod.rs") == ("api", "users")
    assert file_module("crates/core/src/lib.rs") == ()
    assert file_module("src/users.rs") == ("users",)


# --------------------------------------------------------------------------- test-only routers


def test_routers_in_test_code_are_not_provider_routes():
    src = (
        'pub fn app() -> Router { Router::new().route("/real", get(h)) }\n'
        "#[cfg(test)]\nmod tests {\n"
        '    fn test_app() -> Router { Router::new().route("/cfg-test", get(h)) }\n'
        "}\n"
        '#[tokio::test]\nasync fn t() { let _ = Router::new().route("/tokio-test", get(h)); }\n'
        '#[test]\nfn u() { let _ = Router::new().route("/plain-test", get(h)); }\n'
    )
    r = extract_axum_routes(
        {
            "src/lib.rs": src,
            "tests/api.rs": 'fn app() -> Router { Router::new().route("/tests-dir", get(h)) }\n',
        }
    )
    assert [x.path for x in r.routes] == ["/real"]


def test_reqwest_in_test_fn_is_not_a_call_site():
    src = (
        '#[tokio::test]\nasync fn t() { reqwest::Client::new().get("http://x/test"); }\n'
        'fn real() { reqwest::Client::new().get("http://x/real"); }\n'
    )
    assert [s.target for s in _sites(src).call_sites] == ["/real"]


@pytest.mark.parametrize(
    "pred, test_only",
    [
        ("test", True),
        ('all(test, feature = "x")', True),
        ("all(unix, test)", True),
        ("all(unix, all(test))", False),  # nested: not read (simplest form only)
        ('any(test, feature = "x")', False),  # also built with --features x
        ("any(test, all(test, unix))", False),  # any(..) is never read as test-only
        ("not(test)", False),
        ("all(not(test), unix)", False),
        ('feature = "test"', False),
        ('test = "x"', False),
        ("debug_assertions", False),
        ("all(test", False),  # unbalanced: not read as test-only
    ],
)
def test_cfg_predicate_requires_test(pred, test_only):
    assert rh.cfg_requires_test(pred) is test_only


def test_cfg_all_test_items_are_excluded_any_test_items_are_not():
    src = (
        'pub fn app() -> Router { Router::new().route("/real", get(h)) }\n'
        '#[cfg(all(test, feature = "it"))]\n'
        'mod it { fn r() -> Router { Router::new().route("/all-test", get(h)) } }\n'
        '#[cfg(any(test, feature = "dev"))]\n'
        'pub fn dev() -> Router { Router::new().route("/any-test", get(h)) }\n'
    )
    paths = sorted(x.path for x in extract_axum_routes({"src/lib.rs": src}).routes)
    assert paths == ["/any-test", "/real"]


def test_test_attribute_is_found_through_comments_and_doc_comments():
    src = (
        "#[cfg(test)]\n// a comment\n/// a doc comment\n/* block */\n"
        'mod tests { fn r() -> Router { Router::new().route("/t", get(h)) } }\n'
        'pub fn app() -> Router { Router::new().route("/real", get(h)) }\n'
    )
    assert [x.path for x in extract_axum_routes({"src/lib.rs": src}).routes] == ["/real"]


def test_attributes_stop_at_the_next_item():
    # The cfg(test) belongs to `mod a`; `app` after it is shipped code.
    src = (
        '#[cfg(test)]\nmod a {}\npub fn app() -> Router { Router::new().route("/real", get(h)) }\n'
    )
    assert [x.path for x in extract_axum_routes({"src/lib.rs": src}).routes] == ["/real"]


def test_inline_nested_router_is_mounted_once_under_its_prefix():
    src = (
        "pub fn app() -> Router {\n"
        '    Router::new().route("/a", get(h)).nest(\n'
        '        "/api",\n'
        '        Router::new().route("/b", get(h2)).merge(Router::new().route("/c", post(h3))),\n'
        "    )\n"
        "}\n"
    )
    r = extract_axum_routes({"src/main.rs": src})
    # Negative control for the old double count: /b and /c were ALSO listed bare.
    assert sorted((x.method, x.path) for x in r.routes) == [
        ("GET", "/a"),
        ("GET", "/api/b"),
        ("POST", "/api/c"),
    ]


def test_router_chain_behind_a_non_router_call_is_still_found():
    src = 'fn app() { let r = Router::new().route("/x", get(h)).into_make_service().ok(); }\n'
    assert [x.path for x in extract_axum_routes({"src/main.rs": src}).routes] == ["/x"]


# --------------------------------------------------------------------------- budget


def test_rust_routes_honour_max_routes():
    src = (
        "fn app() -> Router {\n"
        '    Router::new().route("/a", get(a)).route("/b", get(b)).route("/c", get(c))\n'
        "}\n"
    )
    full = extract_provider_routes({"src/main.rs": src})
    assert len(full.routes) == 3 and full.truncated is False
    one = extract_provider_routes({"src/main.rs": src}, max_routes=1)
    assert len(one.routes) == 1 and one.truncated is True
    exact = extract_provider_routes({"src/main.rs": src}, max_routes=3)
    assert len(exact.routes) == 3


def test_many_same_named_router_fns_are_ambiguous_with_a_bounded_reason():
    src = (
        "".join(
            f'mod m{i} {{ pub fn r() -> Router {{ Router::new().route("/x{i}", get(h)) }} }}\n'
            for i in range(30)
        )
        + "pub fn app() -> Router { Router::new().merge(r()) }\n"
    )
    res = extract_axum_routes({"src/lib.rs": src})
    (amb,) = [u for u in res.unresolved if "ambiguous" in u.reason]
    assert amb.reason.endswith("and 20 more")
    # Every candidate of the ambiguous group counts as mounted: none is a root.
    assert [r.path for r in res.routes] == []
