"""Outbound call-site extraction: the consumer side of scan_contracts."""

from __future__ import annotations

import time

import pytest

from processing.contract_analysis import call_sites as cs
from processing.contract_analysis.call_sites import (
    extract_file_call_sites,
    is_test_path,
    language_for,
)
from processing.contract_analysis.checks.types import normalize_path
from processing.contract_analysis.parsers.route_parser import parse_routes


def _sites(path: str, source: str, kind: str | None = None):
    result = extract_file_call_sites(path, source)
    assert result.parsed
    return [s for s in result.call_sites if kind is None or s.kind == kind], result.unresolved


def _one(sites, line: int):
    found = [s for s in sites if s.line == line]
    assert len(found) == 1, f"line {line}: {sites}"
    return found[0]


# ---------------------------------------------------------------------------
# Python HTTP
# ---------------------------------------------------------------------------

PY_CLIENT = """\
import httpx
import requests

BASE = "http://kag.example.com:9000"
USERS = "/api/users"


class KagClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self._client = httpx.AsyncClient()

    async def get_doc(self, doc_id):
        return await self._client.get(f"{self.base_url}/api/documents/{doc_id}/")

    def create(self, payload):
        return requests.post(BASE + "/api/documents", json=payload)

    def list_users(self):
        return requests.get(USERS + "?limit=10")

    def raw(self, path):
        return requests.get(path)

    def explicit(self, job):
        return self._client.request("DELETE", f"/api/jobs/{job}")

    def formatted(self, job):
        return requests.put("/api/jobs/{}/cancel".format(job))

    def not_http(self, data):
        return data.get("/api/not-a-call")


def config(d):
    return d.get("key")


@router.get("/api/provided")
async def provided():
    return {}
"""


def test_python_http_call_sites():
    sites, unresolved = _sites("svc/client.py", PY_CLIENT, "http")

    doc = _one(sites, 14)
    assert (doc.method, doc.target, doc.confidence) == ("GET", "/api/documents/{param}", "template")
    assert doc.base == "self.base_url"
    assert doc.client == "self._client"
    assert doc.symbol == "KagClient.get_doc"

    create = _one(sites, 17)
    assert (create.method, create.target, create.confidence) == ("POST", "/api/documents", "exact")
    assert create.host == "kag.example.com:9000"
    assert create.client == "requests"

    users = _one(sites, 20)
    assert users.target == "/api/users"  # constant inlined, query stripped

    explicit = _one(sites, 26)
    assert (explicit.method, explicit.target) == ("DELETE", "/api/jobs/{param}")

    formatted = _one(sites, 29)
    assert (formatted.method, formatted.target) == ("PUT", "/api/jobs/{param}/cancel")

    # A path-shaped literal on an unknown receiver counts (the documented cost of
    # literal-shape detection); a non-path key on one never does.
    assert _one(sites, 32).target == "/api/not-a-call"
    assert all(s.line != 36 for s in sites)
    # the route decorator is the PROVIDER side and never a call site
    assert all(s.target != "/api/provided" for s in sites)

    assert [(u.line, u.client, u.expression) for u in unresolved] == [(23, "requests", "path")]


def test_python_dynamic_method_and_urlopen():
    src = """\
from urllib.request import urlopen

def ping():
    return urlopen("https://status.example.com/health")
"""
    sites, _ = _sites("ping.py", src)
    assert [(s.method, s.target, s.host, s.client) for s in sites] == [
        ("GET", "/health", "status.example.com", "urllib")
    ]


def test_aiohttp_session_and_rebound_variable_is_unresolved():
    src = """\
import aiohttp

async def go(session, flag):
    url = "/api/a"
    url = "/api/b"
    async with session.post(url) as r:
        return r
"""
    sites, unresolved = _sites("a.py", src)
    assert sites == []
    assert [(u.method, u.client, u.expression) for u in unresolved] == [("POST", "session", "url")]


# Live false positives on nonagenticai/ems: `client = dict(row)` then
# `client.get("redirect_uris", [])` was reported as GET /redirect_uris.
@pytest.mark.parametrize("receiver", ["client", "session", "http", "api", "self._client"])
def test_dict_get_with_default_on_client_named_receiver_is_not_a_call(receiver):
    src = f"""\
def load(row):
    {receiver} = dict(row)
    {receiver}["redirect_uris"] = {receiver}.get("redirect_uris", [])
    scope = {receiver}.get("scope", None)
    key = "contacts"
    return {receiver}.get(key, [])
"""
    sites, unresolved = _sites("db.py", src)
    assert sites == []
    assert unresolved == []


@pytest.mark.parametrize("receiver", ["client", "session", "http", "api"])
@pytest.mark.parametrize("path", ["db.py", "db.ts"])
def test_bare_word_key_is_never_a_path(receiver, path):
    sites, unresolved = _sites(path, f'x = {receiver}.get("redirect_uris")\n')
    assert sites == []
    # `http` is an HTTP client by type: listed for a human, never guessed into a path.
    expected = [("GET", cs.BARE_WORD_REASON)] if receiver == "http" else []
    assert [(u.method, u.reason) for u in unresolved] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        # non-`get` verbs are never a mapping read
        ("a.ts", 'apiClient.post("auth/login", body);\n', ("POST", "/auth/login")),
        # known HTTP libraries
        ("a.ts", 'axios.get("api/users");\n', ("GET", "/api/users")),
        ("a.py", 'import requests\nrequests.get("api/x")\n', ("GET", "/api/x")),
        # a target containing `/` on an HTTP-named receiver
        ("a.ts", "api.get(`users/${id}`);\n", ("GET", "/users/{param}")),
        ("a.py", 'self.http.get("v1/items")\n', ("GET", "/v1/items")),
        ("a.py", 'session.get("api/users")\n', ("GET", "/api/users")),
        ("a.py", 'api_client.get("users/" + uid)\n', ("GET", "/users/{param}")),
    ],
)
def test_relative_target_on_http_receiver_is_still_a_call(path, source, expected):
    sites, unresolved = _sites(path, source)
    assert [(s.method, s.target) for s in sites] == [expected]
    assert unresolved == []


@pytest.mark.parametrize(
    ("path", "source"),
    [("a.ts", 'apiClient.get("users");\n'), ("a.py", 'self.http_client.get("health")\n')],
)
def test_bare_word_get_on_strongly_typed_http_receiver_is_unresolved(path, source):
    sites, unresolved = _sites(path, source)
    assert sites == []
    assert [(u.method, u.reason) for u in unresolved] == [("GET", cs.BARE_WORD_REASON)]


def test_bare_word_get_on_a_nested_api_property_is_not_a_call():
    sites, unresolved = _sites("m.ts", "const d = dependenciesManifest.api.get('dep-1');\n")
    assert sites == []
    assert unresolved == []


# A mapping has no `.delete`/`.post`: a bare word under a non-GET verb on an
# HTTP-named receiver (not a library) is a relative request on a base URL.
@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("a.py", 'client.delete("users")\n', ("DELETE", "/users")),
        ("a.ts", 'apiClient.post("login", body);\n', ("POST", "/login")),
        ("a.py", 'self.client.post("x", data={})\n', ("POST", "/x")),
        # no keyword, so only the verb (not the `dict.get` keyword rule) can keep it
        ("a.py", 'self.client.post("x", {})\n', ("POST", "/x")),
    ],
)
def test_bare_word_non_get_verb_on_http_named_receiver_is_a_call(path, source, expected):
    sites, unresolved = _sites(path, source)
    assert [(s.method, s.target) for s in sites] == [expected]
    assert unresolved == []


# `dict.get` takes no keyword arguments: any keyword makes a Python `.get` HTTP.
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('client.get("users", params={"a": 1})\n', "/users"),
        ('session.get("x", timeout=5)\n', "/x"),
        ('client.get("users", {"a": 1}, timeout=5)\n', "/users"),
    ],
)
def test_python_get_with_a_keyword_is_never_a_mapping_read(source, expected):
    sites, unresolved = _sites("a.py", source)
    assert [(s.method, s.target) for s in sites] == [("GET", expected)]
    assert unresolved == []


def test_python_get_with_a_keyword_on_a_mapping_receiver_is_still_not_a_call():
    sites, unresolved = _sites("a.py", 'cache.get("x", timeout=5)\n')
    assert sites == []
    assert unresolved == []


@pytest.mark.parametrize(
    "source",
    [
        "import httpx\nasync def f(B):\n"
        "    async with httpx.AsyncClient(base_url=B) as client:\n"
        '        await client.get("items")\n',
        "from httpx import AsyncClient\nasync def f(B):\n"
        "    async with AsyncClient(base_url=B) as c:\n"
        '        await c.get("items")\n',
        "import requests\nclass A:\n    def __init__(self):\n"
        "        self.s = requests.Session()\n"
        "    def f(self):\n"
        '        return self.s.get("items")\n',
    ],
)
def test_bare_word_get_on_a_client_bound_to_an_http_library_is_a_call(source):
    sites, unresolved = _sites("a.py", source)
    assert [(s.method, s.target) for s in sites] == [("GET", "/items")]
    assert unresolved == []


def test_client_constructor_is_scoped_to_the_receiver_it_binds():
    src = (
        "import httpx\nasync def f(B, row):\n"
        "    async with httpx.AsyncClient(base_url=B) as http_conn:\n"
        '        await http_conn.get("items")\n'
        "    client = dict(row)\n"
        '    return client.get("redirect_uris")\n'
    )
    sites, unresolved = _sites("a.py", src)
    assert [(s.line, s.target) for s in sites] == [(4, "/items")]
    assert unresolved == []


def test_url_shaped_targets_still_count_on_any_receiver():
    src = """\
import httpx
import requests

def go(client, thing, base, url):
    a = requests.get("https://kag.example.com/api/docs")
    b = httpx.get(f"{base}/x")
    c = client.get("/x")
    d = thing.get("/api/x")
    e = client.get(f"{url}/health")
    f = requests.get("https://kag.example.com/api/q", {"a": 1})
    g = client.get(url)
"""
    sites, unresolved = _sites("go.py", src)
    assert [(s.line, s.method, s.target, s.client) for s in sites] == [
        (5, "GET", "/api/docs", "requests"),
        (6, "GET", "/x", "httpx"),
        (7, "GET", "/x", "client"),
        (8, "GET", "/api/x", "thing"),
        (9, "GET", "/health", "client"),
        (10, "GET", "/api/q", "requests"),
    ]
    assert [(u.line, u.client, u.expression) for u in unresolved] == [(11, "client", "url")]


def test_js_two_argument_get_is_still_http():
    src = """\
export const a = (base: string, cfg: object) => api.get(`${base}/x`, cfg);
export const b = (url: string, cfg: object) => api.get(url, cfg);
"""
    sites, unresolved = _sites("a.ts", src)
    assert [(s.line, s.target, s.base) for s in sites] == [(1, "/x", "base")]
    assert [(u.line, u.expression) for u in unresolved] == [(2, "url")]


# ---------------------------------------------------------------------------
# TypeScript HTTP
# ---------------------------------------------------------------------------

TS_CLIENT = """\
import axios from 'axios';
import { authFetch } from './auth-fetch';

const BASE = '/api/assets/pm/projects';

export class Api {
  projects = {
    get: (id: string) => this.get<Project>(`/api/projects/${id}`),
    create: (data: unknown) => this.post<Project>('/api/projects', data),
  };
}

export async function remove(id: string) {
  return authFetch(`${BASE}/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function list(qs: string) {
  return await fetch(`/api/items${qs}`);
}

export async function byUrl(url: string) {
  return authFetch(url);
}

export const axiosCall = () => axios.get('https://api.example.com/v2/things/:thingId?x=1');
export const axiosObj = () => axios({ url: '/api/objs', method: 'patch' });
export const n8n = (ctx: any, baseUrl: string) =>
  ctx.helpers.request({ method: 'GET', uri: `${baseUrl}/api/metrics/aggregate`, json: true });
export const n8nAuth = (ctx: any) =>
  ctx.helpers.requestWithAuthentication.call(ctx, 'frApi', { method: 'POST', url: '/api/cards' });
const lookup = new Map<string, number>();
export const notHttp = () => lookup.get('projects');
const dynMethod = (m: string) => fetch('/api/dyn', { method: m });
"""


def test_typescript_http_call_sites():
    sites, unresolved = _sites("src/lib/api/client.ts", TS_CLIENT, "http")
    by_line = {s.line: s for s in sites}

    assert (by_line[8].method, by_line[8].target, by_line[8].client) == (
        "GET",
        "/api/projects/{param}",
        "this",
    )
    assert by_line[8].symbol == "Api.projects.get"
    assert (by_line[9].method, by_line[9].target, by_line[9].confidence) == (
        "POST",
        "/api/projects",
        "exact",
    )
    # constant base + template; method from the init object
    assert (by_line[14].method, by_line[14].target, by_line[14].client) == (
        "DELETE",
        "/api/assets/pm/projects/{param}",
        "authFetch",
    )
    # a hole glued to the end of a word is a query suffix, not a segment
    assert (by_line[18].method, by_line[18].target) == ("GET", "/api/items")
    assert by_line[18].symbol == "list"
    # express-style :param, host split, query dropped
    assert (by_line[25].target, by_line[25].host) == ("/v2/things/{param}", "api.example.com")
    assert (by_line[26].method, by_line[26].target) == ("PATCH", "/api/objs")
    assert (by_line[28].target, by_line[28].base) == ("/api/metrics/aggregate", "baseUrl")
    assert (by_line[30].method, by_line[30].target) == ("POST", "/api/cards")
    assert by_line[30].client == "ctx.helpers.requestWithAuthentication"
    assert by_line[33].method is None  # dynamic method stays unknown, never guessed
    assert 32 not in by_line  # Map.get

    assert [(u.line, u.client, u.expression) for u in unresolved] == [(22, "authFetch", "url")]


def test_awaited_generic_wrapper_call():
    src = "async function f() { return await assetFetch<T[]>('/api/research/batch', { method: 'POST' }); }\n"
    sites, _ = _sites("r.ts", src)
    assert [(s.client, s.method, s.target) for s in sites] == [
        ("assetFetch", "POST", "/api/research/batch")
    ]


def test_svelte_script_block_lines_are_file_lines():
    src = """\
<h1>title</h1>
<p>body</p>
<script lang="ts">
  async function load() {
    return authFetch(`/api/library/artifacts/${id}/download-token`, { method: 'POST' });
  }
</script>
"""
    sites, _ = _sites("routes/+page.svelte", src)
    assert [(s.line, s.method, s.target, s.symbol) for s in sites] == [
        (5, "POST", "/api/library/artifacts/{param}/download-token", "load")
    ]


def test_tsx_and_javascript_files_parse():
    tsx = "export const C = () => { fetch('/api/x'); return <div/>; };\n"
    js = "const r = require('axios');\nmodule.exports = () => r.post('/api/y');\n"
    assert [s.target for s in _sites("C.tsx", tsx)[0]] == ["/api/x"]
    assert [(s.method, s.target) for s in _sites("c.js", js)[0]] == [("POST", "/api/y")]


def test_express_route_registration_is_not_a_call():
    src = "app.get('/api/health', (req, res) => res.send('ok'));\nrouter.post('/api/x', h);\n"
    assert _sites("server.js", src)[0] == []


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

PY_EVENTS = """\
import nats

async def run(nc, tenant):
    await nc.publish("jobs.created", b"x")
    await nc.publish(f"jobs.{tenant}.done", b"x")
    await nc.subscribe("jobs.*", cb=handler)
    await nc.publish(subject_for(tenant), b"x")
    js = nc.jetstream()
    await js.pull_subscribe("orders.new", durable="w")
"""


def test_python_nats_events():
    sites, unresolved = _sites("worker.py", PY_EVENTS, "event")
    assert [(s.line, s.operation, s.target, s.client, s.confidence) for s in sites] == [
        (4, "publish", "jobs.created", "nats", "exact"),
        (5, "publish", "jobs.{param}.done", "nats", "template"),
        (6, "subscribe", "jobs.*", "nats", "exact"),
        (9, "subscribe", "orders.new", "nats", "exact"),
    ]
    assert [(u.line, u.operation, u.expression) for u in unresolved] == [
        (7, "publish", "subject_for(tenant)")
    ]


def test_python_kafka_redis_amqp_events():
    src = """\
from aiokafka import AIOKafkaConsumer

consumer = AIOKafkaConsumer("billing.events", "billing.audit")
await producer.send_and_wait("billing.events", b"x")
consumer.subscribe(topics=["a.b"])
"""
    sites, _ = _sites("k.py", src, "event")
    assert [(s.operation, s.target, s.client) for s in sites] == [
        ("subscribe", "billing.events", "kafka"),
        ("subscribe", "billing.audit", "kafka"),
        ("publish", "billing.events", "kafka"),
        ("subscribe", "a.b", "kafka"),
    ]
    redis_src = 'import redis\nr.publish("cache:invalidate", "1")\np.psubscribe("cache:*")\n'
    assert [(s.operation, s.target, s.client) for s in _sites("r.py", redis_src)[0]] == [
        ("publish", "cache:invalidate", "redis-pubsub"),
        ("subscribe", "cache:*", "redis-pubsub"),
    ]
    amqp = (
        "import pika\n"
        "channel.basic_publish(exchange='', routing_key='emails', body=b'x')\n"
        "channel.basic_consume(queue='emails', on_message_callback=cb)\n"
    )
    assert [(s.operation, s.target, s.client) for s in _sites("q.py", amqp)[0]] == [
        ("publish", "emails", "amqp"),
        ("subscribe", "emails", "amqp"),
    ]


def test_typescript_kafkajs_and_non_event_subscribe():
    src = """\
import { Kafka } from 'kafkajs';
await producer.send({ topic: 'orders', messages: [] });
await consumer.subscribe({ topics: ['orders', 'refunds'] });
store.subscribe((v) => console.log(v));
socket.send('hello world');
server.publish(`kanban:board:${boardId}`, payload);
"""
    sites, unresolved = _sites("k.ts", src, "event")
    assert [(s.operation, s.target, s.client) for s in sites] == [
        ("publish", "orders", "kafka"),
        ("subscribe", "orders", "kafka"),
        ("subscribe", "refunds", "kafka"),
        ("publish", "kanban:board:{param}", "kafka"),
    ]
    assert unresolved == []


# ---------------------------------------------------------------------------
# The join: consumer call sites pair with scan_contracts' provider routes
# ---------------------------------------------------------------------------

PROVIDER_ROUTES = """\
from fastapi import APIRouter
router = APIRouter(prefix="/api/documents")

@router.get("/{document_id}")
async def get_document(document_id: str): ...

@router.post("")
async def create_document(body: DocumentCreate): ...

@router.delete("/{document_id}/")
async def delete_document(document_id: str): ...
"""

CONSUMER_TS = """\
export const getDoc = (id: string) => authFetch(`/api/documents/${id}`);
export const create = () => authFetch('/api/documents/', { method: 'POST' });
export const del = (id: string) => this.delete(`/api/documents/${encodeURIComponent(id)}`);
export const missing = () => authFetch('/api/documents/x/archive', { method: 'POST' });
"""


def test_consumer_call_sites_join_provider_routes():
    routes = parse_routes(PROVIDER_ROUTES)
    provided = {(r.method.upper(), normalize_path(r.path) or "/") for r in routes.routes}
    assert provided == {
        ("GET", "/api/documents/{param}"),
        ("POST", "/api/documents"),
        ("DELETE", "/api/documents/{param}"),
    }
    sites, _ = _sites("docs.ts", CONSUMER_TS, "http")
    keys = [(s.method, s.target) for s in sites]
    matched = [k for k in keys if k in provided]
    assert matched == [
        ("GET", "/api/documents/{param}"),
        ("POST", "/api/documents"),
        ("DELETE", "/api/documents/{param}"),
    ]
    assert [k for k in keys if k not in provided] == [("POST", "/api/documents/x/archive")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("a.py", "python"),
        ("a.mjs", "javascript"),
        ("a.tsx", "typescript"),
        ("a.svelte", "svelte"),
        ("a.vue", "vue"),
        ("a.go", None),
    ],
)
def test_language_for(path, lang):
    assert language_for(path) == lang


@pytest.mark.parametrize(
    ("path", "is_test"),
    [
        ("tests/unit/test_x.py", True),
        ("src/x.test.ts", True),
        ("src/__tests__/x.ts", True),
        ("src/types.d.ts", True),
        ("src/conftest.py", True),
        ("src/api/client.ts", False),
        ("src/testing_utils.py", False),
    ],
)
def test_is_test_path(path, is_test):
    assert is_test_path(path) is is_test


def test_unknown_language_is_not_parsed():
    assert extract_file_call_sites("a.go", "package main").parsed is False


def test_parse_error_region_is_skipped_not_guessed():
    src = "fetch('/api/ok');\nconst x = (((;\n"
    sites, _ = _sites("e.js", src)
    assert [s.target for s in sites] == ["/api/ok"]


def test_send_counts_only_on_a_kafka_producer():
    src = (
        "const raw = await redis.send('GETDEL', [key]);\n"
        "await channel.send(approval, address);\n"
        "await kafkaProducer.send({ topic: 'audit', messages: [] });\n"
    )
    sites, unresolved = _sites("s.ts", src)
    assert [(s.operation, s.target) for s in sites] == [("publish", "audit")]
    assert unresolved == []


def test_glued_trailing_hole_is_a_query_only_when_it_reads_as_one():
    src = (
        "const BASE = '/api/admin/verification';\n"
        "export const a = (path: string) => authFetch(`${BASE}${path}`);\n"
        "export const b = (qs: string) => authFetch(`${BASE}${qs ? `?${qs}` : ''}`);\n"
        "export const c = (params: URLSearchParams) => authFetch(`/api/x${params}`);\n"
    )
    sites, unresolved = _sites("v.ts", src)
    assert [(s.line, s.target) for s in sites] == [
        (3, "/api/admin/verification"),
        (4, "/api/x"),
    ]
    assert [(u.line, u.expression) for u in unresolved] == [(2, "`${BASE}${path}`")]


def test_underscored_suffix_hole_is_a_query():
    src = "ctx.helpers.request({ method: 'GET', uri: `${baseUrl}/api/approvals${__suffix}` });\n"
    sites, _ = _sites("o.ts", src)
    assert [(s.target, s.base) for s in sites] == [("/api/approvals", "baseUrl")]


# ---------------------------------------------------------------------------
# Linear-time scanning: adversarial inputs must not stall the (single-process,
# GIL-holding) service. Bounds are generous; the quadratic forms took 4-80 s.
# ---------------------------------------------------------------------------

_BOUND_S = 6.0  # main runs under coverage tracing (~1.5x slower); quadratic forms took 4-80 s


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


@pytest.mark.parametrize("path", ["a.vue", "a.svelte"])
def test_unclosed_script_openers_scan_in_linear_time(path):
    source = "<script>" * (512 * 1024 // len("<script>"))  # 512 KB, no closer
    result, elapsed = _timed(lambda: extract_file_call_sites(path, source))
    assert result.parsed
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


def test_openers_without_a_tag_end_scan_in_linear_time():
    result, elapsed = _timed(lambda: extract_file_call_sites("a.vue", "<script " * 65536))
    assert result.parsed and result.call_sites == []
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


def test_many_script_blocks_keep_file_line_numbers():
    # Lines come from a running count (one pass), not a recount from 0 per block.
    block = "\n" * 10 + "<script>fetch('/api/x')</script>"
    result = extract_file_call_sites("a.svelte", block * 512)
    assert [s.line for s in result.call_sites][:3] == [11, 21, 31]
    assert len(result.call_sites) == 512 and result.call_sites[-1].line == 5121


def test_script_blocks_case_insensitive_and_unterminated_runs_to_eof():
    src = (
        "<template><p/></template>\n"
        "<SCRIPT lang='ts'>\n"
        "fetch('/api/one')\n"
        "</Script>\n"
        "<scripts>not a script</scripts>\n"
        "<script>\n"
        "fetch('/api/two')\n"
    )
    sites, _ = _sites("a.vue", src, "http")
    assert [(s.line, s.target) for s in sites] == [(3, "/api/one"), (7, "/api/two")]


def test_brace_run_target_normalises_in_linear_time():
    source = "fetch('/" + "{" * (256 * 1024) + "')"
    result, elapsed = _timed(lambda: extract_file_call_sites("a.ts", source))
    assert result.parsed
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


def test_brace_run_event_subject_normalises_in_linear_time():
    source = "import nats\nnc.publish('" + "{" * (256 * 1024) + "x}', b'')\n"
    result, elapsed = _timed(lambda: extract_file_call_sites("a.py", source))
    # Past the resolver's char cap: listed as too complex, never expanded.
    assert result.call_sites == []
    assert [u.reason for u in result.unresolved] == [cs.TOO_COMPLEX_REASON]
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
    # The normaliser itself stays linear on the same run (reached directly).
    rendered, elapsed = _timed(
        lambda: cs._event_target(cs._Rendered(path="{" * (256 * 1024) + "x}", template=False))
    )
    assert rendered is not None and rendered.path == "{" * (256 * 1024 - 1) + "{param}"
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


def test_normalize_path_brace_run_is_linear():
    path, elapsed = _timed(lambda: normalize_path("/" + "{" * (256 * 1024) + "id}"))
    assert path.endswith("{{{param}")
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"


def test_python_imports_over_blank_lines_scan_in_linear_time():
    # Blank lines AFTER the import: with ``^\s*`` each one rescanned to end of file.
    source = "import nats\n" + "\n" * 100_000 + "nc.publish('a.b', b'')\n"
    result, elapsed = _timed(lambda: extract_file_call_sites("a.py", source))
    assert [(s.kind, s.target) for s in result.call_sites] == [("event", "a.b")]
    assert elapsed < _BOUND_S, f"{elapsed:.2f}s"
