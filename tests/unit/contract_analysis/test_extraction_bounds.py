"""Adversarial-input bounds for the contract extractors (round-3 DoS findings).

Every case here was super-linear (seconds to minutes, GBs of memory) before its
fix. The service is one uvicorn process; the isolation worker bounds the damage,
but each extractor must still be linear on its own.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest

from processing.contract_analysis import call_sites as cs
from processing.contract_analysis.call_sites import extract_file_call_sites
from processing.contract_analysis.jobs import call_sites_job
from processing.contract_analysis.parsers.ts_route_parser import parse_ts_routes
from processing.contract_analysis.provider_routes import extract_provider_routes

_BOUND_S = 6.0  # main runs under coverage tracing; super-linear forms took 9-30+ s


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


# -- N1: self/cyclic mounts were k^11 --------------------------------------------


@pytest.mark.parametrize("k", [5, 6, 11])
def test_ts_self_mounts_are_a_cycle_not_an_explosion(k):
    src = (
        "import express from 'express';\nconst router = express.Router();\n"
        + "".join(f"router.use('/p{i}', router);\n" for i in range(k))
        + "router.get('/x', h);\nexport default router;\n"
    )
    result, elapsed = _timed(lambda: extract_provider_routes({"a.ts": src}))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert result.routes == []
    assert [u.reason for u in result.unresolved] == [
        "mounted under a cyclic mount (a router mounted inside itself)"
    ]


@pytest.mark.parametrize("k", [5, 6, 11])
def test_python_self_includes_are_a_cycle_not_an_explosion(k):
    src = (
        "from fastapi import APIRouter\nrouter = APIRouter()\n"
        + "".join(f"router.include_router(router, prefix='/p{i}')\n" for i in range(k))
        + "@router.get('/x')\ndef h(): pass\n"
    )
    result, elapsed = _timed(lambda: extract_provider_routes({"a.py": src}))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert result.routes == []
    assert [u.reason for u in result.unresolved] == [
        "included under a cyclic mount (a router mounted inside itself)"
    ]


def test_mount_ring_keeps_the_static_prefixes_and_flags_the_cycle():
    # app -> r0 at /a; r0 -> r1 -> r2 -> r0 at /b each: r0's route is at /a/x, and
    # also under the ring (unresolvable), never expanded around it.
    src = (
        "const app = express();\n"
        "const r0 = express.Router();\nconst r1 = express.Router();\n"
        "const r2 = express.Router();\n"
        "app.use('/a', r0);\nr0.use('/b', r1);\nr1.use('/b', r2);\nr2.use('/b', r0);\n"
        "r0.get('/x', h);\n"
    )
    result = parse_ts_routes({"a.ts": src})
    assert [r.path for r in result.routes] == ["/a/x"]
    assert [u.reason for u in result.unresolved] == [
        "mounted under a cyclic mount (a router mounted inside itself)"
    ]


def test_mount_diamond_lists_every_prefix_once():
    src = (
        "const app = express();\nconst a = express.Router();\nconst b = express.Router();\n"
        "app.use('/v1', a);\napp.use('/v2', a);\na.use('/b1', b);\na.use('/b2', b);\n"
        "b.get('/x', h);\n"
    )
    result = parse_ts_routes({"a.ts": src})
    assert sorted(r.path for r in result.routes) == [
        "/v1/b1/x",
        "/v1/b2/x",
        "/v2/b1/x",
        "/v2/b2/x",
    ]
    assert result.unresolved == []


def test_mount_ladder_is_capped_per_router_and_in_total():
    # Each rung mounts the next twice: 2^i prefixes at rung i without the cap.
    n = 10
    src = (
        "const app = express();\n"
        + "".join(f"const r{i} = express.Router();\n" for i in range(n))
        + "app.use('/', r0);\n"
        + "".join(f"r{i}.use('/a', r{i + 1}); r{i}.use('/b', r{i + 1});\n" for i in range(n - 1))
        + "".join(f"r{i}.get('/x', h);\n" for i in range(n))
    )
    result, elapsed = _timed(lambda: parse_ts_routes({"a.ts": src}))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert max(Counter(r.line for r in result.routes).values()) == 64  # deep rungs capped
    assert any("distinct prefixes" in u.reason for u in result.unresolved)
    capped = parse_ts_routes({"a.ts": src}, max_routes=10)
    assert capped.truncated and 10 < len(capped.routes) <= 10 + 64


# -- N2: constant inlining fanned out W^5 ------------------------------------------


def _fanout(width: int, template: bool) -> str:
    src = "const a5 = '/x';\n"
    for lvl in range(4, -1, -1):
        if template:
            src += f"const a{lvl} = `" + "".join([f"${{a{lvl + 1}}}"] * width) + "`;\n"
        else:
            src += f"const a{lvl} = " + " + ".join([f"a{lvl + 1}"] * width) + ";\n"
    return src + "fetch(a0);\n"


@pytest.mark.parametrize("template", [False, True])
def test_constant_fanout_is_unresolved_too_complex(template):
    result, elapsed = _timed(lambda: extract_file_call_sites("a.ts", _fanout(25, template)))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert result.call_sites == []
    assert [u.reason for u in result.unresolved] == [cs.TOO_COMPLEX_REASON]


def test_small_constant_fanout_still_resolves():
    src = "const B = '/api';\nconst P = B + '/v1';\nconst U = `${P}/items/${P}`;\nfetch(U);\n"
    result = extract_file_call_sites("a.ts", src)
    assert [s.target for s in result.call_sites] == ["/api/v1/items/api/v1"]


def test_cyclic_fstring_constants_do_not_recurse_forever():
    src = "A = f'{B}/x'\nB = f'{A}/y'\nrequests.get(A)\n"
    result = extract_file_call_sites("a.py", src)  # was RecursionError via f-string depth reset
    assert result.parsed


# -- N3: per-call `.parent` walk was O(calls x depth^2) ----------------------------


def test_deeply_nested_functions_with_calls_are_linear():
    n = 2000
    src = "".join(f"function f{i}() {{ fetch('/a/{i}');\n" for i in range(n)) + "}" * n
    result, elapsed = _timed(lambda: extract_file_call_sites("a.ts", src))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert len(result.call_sites) == n
    assert result.call_sites[1].symbol == "f0.f1"
    deepest = result.call_sites[-1].symbol
    assert deepest is not None and deepest.startswith("…") and deepest.endswith(f"f{n - 1}")
    assert len(deepest) <= 256


def test_nested_calls_are_linear():
    n = 2000
    result, elapsed = _timed(
        lambda: extract_file_call_sites("a.ts", "fetch(" * n + "'/a'" + ")" * n + ";\n")
    )
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert [s.target for s in result.call_sites] == ["/a"]


def test_long_call_chains_are_linear():
    n = 10000
    result, elapsed = _timed(lambda: extract_file_call_sites("a.ts", "axios" + ".get('/x')" * n))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert len(result.call_sites) == n
    routes, elapsed = _timed(
        lambda: parse_ts_routes(
            {"a.ts": "const router = express.Router();\nrouter" + ".get('/x', h)" * n}
        )
    )
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


# -- N4: `+` chains recursed per term (RecursionError at ~1000 terms) ---------------


@pytest.mark.parametrize(("path", "call"), [("a.ts", "fetch"), ("a.py", "requests.get")])
def test_long_plus_chain_is_too_complex_not_a_recursion_error(path, call):
    src = f"{call}('/a'" + " + x" * 1100 + ")\n"
    result, elapsed = _timed(lambda: extract_file_call_sites(path, src))
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    assert [u.reason for u in result.unresolved] == [cs.TOO_COMPLEX_REASON]


def test_short_plus_chain_still_resolves():
    result = extract_file_call_sites("a.ts", "fetch('/a' + '/b' + `/${id}` + '/c');\n")
    assert [s.target for s in result.call_sites] == ["/a/b/{param}/c"]


def test_one_failing_file_does_not_fail_the_job(monkeypatch):
    real = cs.extract_file_call_sites

    def flaky(path, source):
        if path == "bad.ts":
            raise RecursionError("maximum recursion depth exceeded")
        return real(path, source)

    monkeypatch.setattr(cs, "extract_file_call_sites", flaky)
    files = [("bad.ts", "fetch('/a')"), ("good.ts", "fetch('/b')")]
    out = list(call_sites_job(files, {"http"}, 100))
    assert [(o.path, o.error) for o in out] == [
        ("bad.ts", "RecursionError: maximum recursion depth exceeded"),
        ("good.ts", None),
    ]
    assert [s.target for s in out[1].call_sites] == ["/b"]


def test_one_failing_route_file_lands_in_unparsed(monkeypatch):
    from processing.contract_analysis.parsers import ts_route_parser

    real = ts_route_parser._Collector.run

    def flaky(self):
        if self.path == "bad.ts":
            raise RecursionError("deep")
        return real(self)

    monkeypatch.setattr(ts_route_parser._Collector, "run", flaky)
    result = extract_provider_routes({"bad.ts": "app.get('/a', h)", "good.ts": "app.get('/b', h)"})
    assert [r.path for r in result.routes] == ["/b"]
    assert result.unparsed == ["bad.ts"]
    assert result.unparsed_reasons == {"bad.ts": "RecursionError: deep"}


def test_call_sites_job_stops_once_both_lists_are_past_the_cap():
    files = [(f"f{i}.ts", "fetch('/a'); fetch(x);\n") for i in range(10)]
    out = list(call_sites_job(files, {"http"}, 2))
    assert len(out) == 3  # 3 files put both lists past 2; the rest cannot change the answer
    assert sum(len(o.call_sites) for o in out) == 3


# -- test gap: per-block line recount (quadratic in blocks) -------------------------


def test_many_script_blocks_are_linear():
    source = "<script></script>\n" * 120_000  # ~2 MB; a per-block recount from 0 is ~130 G chars
    result, elapsed = _timed(lambda: extract_file_call_sites("a.svelte", source))
    assert result.parsed
    assert elapsed < 8.0, f"{elapsed:.2f}s"
