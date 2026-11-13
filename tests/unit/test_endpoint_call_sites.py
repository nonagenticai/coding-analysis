"""POST /api/analysis/contracts/call-sites (operationId scan_call_sites)."""

from __future__ import annotations

import asyncio
import importlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TENANT = "tenant-a"
URL = "/api/analysis/contracts/call-sites"

CLIENT_TS = "export const g = (id: string) => authFetch(`/api/projects/${id}`);\n"
WORKER_PY = 'import nats\nasync def go(nc):\n    await nc.publish("jobs.created", b"x")\n'
TEST_TS = "it('x', () => fetch('/api/mocked'));\n"


@pytest.fixture
def endpoint(monkeypatch):
    # Fresh import: another test module replaces this one in sys.modules with stubs.
    import sys

    sys.modules.pop("app.api.endpoints.contract_analysis", None)
    module = importlib.import_module("app.api.endpoints.contract_analysis")
    from auth.tenant import require_tenant

    app = FastAPI()
    app.include_router(module.router, prefix="/api/analysis/contracts")
    app.dependency_overrides[require_tenant] = lambda: TENANT
    return module, TestClient(app)


def test_inline_sources(endpoint):
    _, client = endpoint
    resp = client.post(
        URL,
        json={
            "sources": [
                {"path": "src/api.ts", "source": CLIENT_TS},
                {"path": "worker.py", "source": WORKER_PY},
                {"path": "src/api.test.ts", "source": TEST_TS},
                {"path": "README.md", "source": "fetch('/x')"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [
        (c["file"], c["kind"], c["method"] or c["operation"], c["target"])
        for c in body["call_sites"]
    ] == [
        ("src/api.ts", "http", "GET", "/api/projects/{param}"),
        ("worker.py", "event", "publish", "jobs.created"),
    ]
    assert body["files_matched"] == 2 and body["files_scanned"] == 2
    assert body["truncated"] is False and body["truncation_reason"] is None
    assert body["extractor_version"]


def test_inline_filters_kinds_languages_paths_tests(endpoint):
    _, client = endpoint
    sources = [
        {"path": "src/api.ts", "source": CLIENT_TS},
        {"path": "worker.py", "source": WORKER_PY},
        {"path": "src/api.test.ts", "source": TEST_TS},
    ]
    only_events = client.post(URL, json={"sources": sources, "kinds": ["event"]}).json()
    assert [c["target"] for c in only_events["call_sites"]] == ["jobs.created"]

    only_ts = client.post(URL, json={"sources": sources, "languages": ["ts"]}).json()
    assert only_ts["languages"] == ["typescript"]
    assert [c["file"] for c in only_ts["call_sites"]] == ["src/api.ts"]

    in_src = client.post(URL, json={"sources": sources, "paths": ["src"], "include_tests": True})
    assert [c["target"] for c in in_src.json()["call_sites"]] == [
        "/api/mocked",
        "/api/projects/{param}",
    ]


def test_truncation_is_flagged(endpoint):
    _, client = endpoint
    sources = [{"path": f"f{i}.ts", "source": "fetch('/a');fetch('/b');\n"} for i in range(3)]
    body = client.post(URL, json={"sources": sources, "max_files": 2, "max_call_sites": 3}).json()
    assert body["files_matched"] == 3 and body["files_scanned"] == 2
    assert len(body["call_sites"]) == 3
    assert body["truncated"] is True
    assert body["truncation_reason"] == "max_files,max_call_sites"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"container_name": "ws", "sources": []},
        {"sources": [], "languages": ["cobol"]},
        {"sources": [{"path": "../etc/passwd.py", "source": ""}]},
        {"sources": [], "paths": ["/etc"]},
    ],
)
def test_bad_requests_are_400(endpoint, payload):
    _, client = endpoint
    assert client.post(URL, json=payload).status_code == 400


def test_workspace_mode_lists_filters_caps_and_reads(endpoint, monkeypatch):
    module, client = endpoint
    import app.api.utils.container_validation as cv

    validate = AsyncMock(return_value=SimpleNamespace(codebase_path="/workspace"))
    monkeypatch.setattr(cv, "validate_and_get_container_for_tenant", validate)
    contents = {
        "/workspace/src/api.ts": CLIENT_TS,
        "/workspace/src/worker.py": WORKER_PY,
        "/workspace/src/gone.ts": None,
    }
    containers = MagicMock()
    listing = "\n".join([*contents, "/workspace/src/api.test.ts"]) + "\n"
    large = "/workspace/src/huge.ts\n/workspace/src/huge.test.ts\n"
    containers.execute = AsyncMock(
        side_effect=lambda name, argv, **kw: {
            "exit_code": 0,
            "stdout": large if argv[-1].startswith("+") else listing,
        }
    )
    containers.read_file = AsyncMock(side_effect=lambda name, path, tenant_id=None: contents[path])
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: containers)

    resp = client.post(URL, json={"container_name": "ws-1", "paths": ["src"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate.assert_awaited_once_with("ws-1", TENANT)
    listings = [c.args[1] for c in containers.execute.await_args_list]
    assert [argv[:2] for argv in listings] == [["find", "/workspace/src"]] * 2
    assert sorted(argv[-2:][1] for argv in listings) == ["+1024k", "-1025k"]
    assert all(c.kwargs["tenant_id"] == TENANT for c in containers.execute.await_args_list)
    read_paths = sorted(c.args[1] for c in containers.read_file.await_args_list)
    assert read_paths == sorted(contents)  # the test file and the huge file were never read
    assert all(c.kwargs["tenant_id"] == TENANT for c in containers.read_file.await_args_list)
    assert body["files_skipped_large"] == ["src/huge.ts"]
    assert body["files_matched"] == 4
    assert body["files_unreadable"] == ["src/gone.ts"]
    assert [(c["file"], c["target"]) for c in body["call_sites"]] == [
        ("src/api.ts", "/api/projects/{param}"),
        ("src/worker.py", "jobs.created"),
    ]


def test_workspace_listing_failure_is_422(endpoint, monkeypatch):
    module, client = endpoint
    import app.api.utils.container_validation as cv

    monkeypatch.setattr(
        cv, "validate_and_get_container_for_tenant", AsyncMock(return_value=SimpleNamespace())
    )
    containers = MagicMock()
    containers.execute = AsyncMock(
        return_value={"exit_code": 1, "stdout": "", "stderr": "find: '/workspace/x': No such"}
    )
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: containers)
    resp = client.post(URL, json={"container_name": "ws-1", "paths": ["x"]})
    assert resp.status_code == 422
    assert "WORKSPACE_UNRESOLVABLE" in resp.text


@pytest.fixture
def settings(monkeypatch):
    """The live settings object, patchable per test (extraction deadline, isolation)."""
    from config.settings import get_settings

    return get_settings()


@pytest.fixture
def in_process(settings, monkeypatch):
    """Extract in this process, so a test can patch the extractor (no deadline)."""
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_ISOLATED", False)


def test_unparsed_file_is_reported(endpoint, monkeypatch, in_process):
    module, client = endpoint
    from processing.contract_analysis.call_sites import FileResult

    monkeypatch.setattr(
        module.cs, "extract_file_call_sites", lambda path, source: FileResult(parsed=False)
    )
    body = client.post(URL, json={"sources": [{"path": "a.ts", "source": "x"}]}).json()
    assert body["files_unparsed"] == ["a.ts"]
    assert body["files_unparsed_reasons"] == {"a.ts": "grammar not available"}
    assert body["files_scanned"] == 0


def test_scan_contracts_routes_typescript_sources_to_the_ts_parser(
    endpoint, monkeypatch, in_process
):
    module, _ = endpoint
    from models.contract_analysis_models import ContractPair, ScanConfig, SourceInput

    scanner = MagicMock()
    monkeypatch.setattr(module.jobs, "ContractScanner", lambda: scanner)
    pair = ContractPair(
        name="c",
        client=SourceInput(source="x = 1"),
        schemas=[SourceInput(source="")],
        routes=[
            SourceInput(source="@router.get('/a')\ndef a(): ...", label="routes.py"),
            SourceInput(source="router.add('GET', '/b', h);", label="src/server.ts"),
            SourceInput(source="app.get('/c', h);", language="javascript"),
        ],
    )
    asyncio.run(module._scan_one(pair, ScanConfig()))
    kwargs = scanner.scan_sources.call_args.kwargs
    assert kwargs["route_sources"] == ["@router.get('/a')\ndef a(): ..."]
    # Parsed in the job, handed to the scanner already parsed.
    assert sorted((r.method, r.path) for r in kwargs["ts_routes"].routes) == [
        ("GET", "/b"),
        ("GET", "/c"),
    ]


def test_capabilities_advertise_typescript_routes(endpoint):
    _, client = endpoint
    body = client.get("/api/analysis/contracts/capabilities").json()
    assert body["route_languages"] == ["python", "typescript", "javascript", "rust"]
    assert "rust" in body["call_site_languages"]
    assert "svelte" in body["call_site_languages"]


ROUTES_URL = "/api/analysis/contracts/routes"
SERVER_TS = (
    "const router = new Router();\n"
    "router.add('GET', '/api/projects/:id', handleGet);\n"
    "router.add(verb, '/api/x', h);\n"
)
ITEMS_PY = (
    "from fastapi import APIRouter\n"
    'router = APIRouter(prefix="/items")\n'
    '@router.delete("/{item_id}")\n'
    "async def delete_item(item_id: str): ...\n"
)


def test_scan_routes_inline(endpoint):
    _, client = endpoint
    resp = client.post(
        ROUTES_URL,
        json={
            "sources": [
                {"path": "src/server.ts", "source": SERVER_TS},
                {"path": "app/items.py", "source": ITEMS_PY},
                {"path": "src/server.test.ts", "source": "router.add('GET', '/t', h);"},
                {"path": "ui/App.svelte", "source": "<script>fetch('/x')</script>"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [
        (r["file"], r["line"], r["method"], r["path"], r["framework"], r["handler"])
        for r in body["routes"]
    ] == [
        ("app/items.py", 3, "DELETE", "/items/{param}", "fastapi", "delete_item"),
        ("src/server.ts", 2, "GET", "/api/projects/{param}", "method-first", "handleGet"),
    ]
    assert [(u["file"], u["line"], u["reason"]) for u in body["unresolved"]] == [
        ("src/server.ts", 3, "dynamic route method")
    ]
    assert body["files_matched"] == 2 and body["files_scanned"] == 2
    assert body["languages"] == ["python", "javascript", "typescript", "rust"]
    assert body["truncated"] is False


def test_scan_routes_rejects_call_site_only_languages(endpoint):
    _, client = endpoint
    resp = client.post(ROUTES_URL, json={"sources": [], "languages": ["svelte"]})
    assert resp.status_code == 400


def test_scan_routes_truncates(endpoint):
    _, client = endpoint
    sources = [
        {"path": f"s{i}.ts", "source": "router.add('GET', '/a', h);\nrouter.add('GET', '/b', h);\n"}
        for i in range(3)
    ]
    body = client.post(
        ROUTES_URL, json={"sources": sources, "max_files": 2, "max_routes": 3}
    ).json()
    assert len(body["routes"]) == 3
    assert body["truncation_reason"] == "max_files,max_routes"


def test_scan_routes_workspace_mode(endpoint, monkeypatch):
    module, client = endpoint
    import app.api.utils.container_validation as cv

    monkeypatch.setattr(
        cv,
        "validate_and_get_container_for_tenant",
        AsyncMock(return_value=SimpleNamespace(codebase_path="/workspace")),
    )
    contents = {"/workspace/src/server.ts": SERVER_TS, "/workspace/app/items.py": ITEMS_PY}
    containers = MagicMock()
    containers.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "\n".join(contents)})
    containers.read_file = AsyncMock(side_effect=lambda name, path, tenant_id=None: contents[path])
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: containers)
    body = client.post(ROUTES_URL, json={"container_name": "ws-1"}).json()
    assert [(r["file"], r["path"]) for r in body["routes"]] == [
        ("app/items.py", "/items/{param}"),
        ("src/server.ts", "/api/projects/{param}"),
    ]
    names = containers.execute.await_args.args[1]
    assert "*.svelte" not in names and "*.py" in names


# ---------------------------------------------------------------------------
# Bounds: per-file failures, input caps, the extraction deadline
# ---------------------------------------------------------------------------


def test_one_failing_file_is_unparsed_not_a_500(endpoint, monkeypatch, in_process):
    module, client = endpoint
    real = module.jobs.cs.extract_file_call_sites

    def flaky(path, source):
        if path == "bad.ts":
            raise RecursionError("maximum recursion depth exceeded")
        return real(path, source)

    monkeypatch.setattr(module.jobs.cs, "extract_file_call_sites", flaky)
    resp = client.post(
        URL,
        json={
            "sources": [{"path": "bad.ts", "source": "x"}, {"path": "ok.ts", "source": CLIENT_TS}]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["files_unparsed"] == ["bad.ts"]
    assert body["files_unparsed_reasons"] == {
        "bad.ts": "RecursionError: maximum recursion depth exceeded"
    }
    assert [c["file"] for c in body["call_sites"]] == ["ok.ts"]
    assert body["files_scanned"] == 1


def test_long_plus_chain_is_200_with_a_too_complex_entry(endpoint):
    _, client = endpoint
    src = "fetch('/a'" + " + x" * 1100 + ");\n"
    resp = client.post(URL, json={"sources": [{"path": "a.ts", "source": src}]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [u["reason"] for u in body["unresolved"]] == [
        "too_complex: target expression expands past the resolver's caps"
    ]
    assert body["files_unparsed"] == []


def test_inline_file_over_the_per_file_cap_is_422(endpoint):
    _, client = endpoint
    from models.contract_analysis_models import MAX_FILE_BYTES

    big = "x" * (MAX_FILE_BYTES + 1)
    resp = client.post(URL, json={"sources": [{"path": "a.ts", "source": big}]})
    assert resp.status_code == 422
    multibyte = "é" * (MAX_FILE_BYTES // 2 + 1)  # under the char limit, over the byte limit
    resp = client.post(ROUTES_URL, json={"sources": [{"path": "a.ts", "source": multibyte}]})
    assert resp.status_code == 422
    assert "per-file limit" in resp.text


def test_inline_sources_over_the_request_cap_are_422(endpoint, monkeypatch):
    _, client = endpoint
    import models.contract_analysis_models as models

    monkeypatch.setattr(models, "MAX_REQUEST_BYTES", 1000)
    sources = [{"path": f"f{i}.ts", "source": "x" * 600} for i in range(2)]
    resp = client.post(URL, json={"sources": sources})
    assert resp.status_code == 422
    assert "per-request limit" in resp.text


def test_workspace_reads_stop_at_the_request_byte_cap(endpoint, monkeypatch):
    module, client = endpoint
    import app.api.utils.container_validation as cv

    monkeypatch.setattr(
        cv,
        "validate_and_get_container_for_tenant",
        AsyncMock(return_value=SimpleNamespace(codebase_path="/workspace")),
    )
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 2 * len(CLIENT_TS) + 1)
    paths = [f"/workspace/f{i}.ts" for i in range(5)]
    containers = MagicMock()
    containers.execute = AsyncMock(
        side_effect=lambda name, argv, **kw: {
            "exit_code": 0,
            "stdout": "" if argv[-1].startswith("+") else "\n".join(paths),
        }
    )
    containers.read_file = AsyncMock(return_value=CLIENT_TS)
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: containers)
    body = client.post(URL, json={"container_name": "ws-1"}).json()
    assert body["files_matched"] == 5 and body["files_scanned"] == 2
    assert body["truncated"] is True and body["truncation_reason"] == "max_bytes"
    assert body["limits"]["max_file_bytes"] == 1024 * 1024


@pytest.mark.timeout(60)
def test_deadline_kills_a_stalled_extraction_and_health_stays_responsive(
    endpoint, monkeypatch, settings
):
    module, _ = endpoint
    import httpx
    from fastapi import FastAPI

    from tests.unit.contract_analysis import isolation_jobs

    deadline = 1.5
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", deadline)
    monkeypatch.setattr(module.jobs, "call_sites_job", isolation_jobs.first_file_then_spin)
    from auth.tenant import require_tenant

    app = FastAPI()
    app.include_router(module.router, prefix="/api/analysis/contracts")
    app.dependency_overrides[require_tenant] = lambda: TENANT

    @app.get("/health")
    async def health():
        return {"ok": True}

    sources = [{"path": "a.ts", "source": CLIENT_TS}, {"path": "b.ts", "source": CLIENT_TS}]

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            started = time.monotonic()
            scan = asyncio.create_task(client.post(URL, json={"sources": sources}, timeout=30))
            pings = []
            while not scan.done():
                t0 = time.monotonic()
                assert (await client.get("/health")).status_code == 200
                pings.append(time.monotonic() - t0)
                await asyncio.sleep(0.05)
            return await scan, time.monotonic() - started, pings

    resp, elapsed, pings = asyncio.run(scenario())
    assert resp.status_code == 200, resp.text
    assert elapsed < deadline + 2.0, f"{elapsed:.2f}s"
    assert len(pings) >= 5 and max(pings) < 0.5, pings
    body = resp.json()
    assert body["truncated"] is True and body["truncation_reason"] == "deadline"
    assert [c["file"] for c in body["call_sites"]] == ["a.ts"]  # the finished file is kept
    assert body["files_unparsed"] == ["b.ts"]
    assert body["files_unparsed_reasons"]["b.ts"].startswith("deadline: ")
    assert body["limits"]["deadline_seconds"] == deadline


@pytest.mark.timeout(60)
def test_routes_deadline_is_a_clear_422(endpoint, monkeypatch, settings):
    module, client = endpoint
    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(module.jobs, "provider_routes_job", isolation_jobs.spin_routes)
    start = time.monotonic()
    resp = client.post(ROUTES_URL, json={"sources": [{"path": "a.ts", "source": SERVER_TS}]})
    assert time.monotonic() - start < 3.0
    assert resp.status_code == 422
    assert "deadline" in resp.text


@pytest.mark.timeout(60)
def test_scan_contracts_pair_deadline_fails_the_pair(endpoint, monkeypatch, settings):
    module, _ = endpoint
    from models.contract_analysis_models import ContractPair, ScanConfig, SourceInput
    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(module.jobs, "scan_pair_job", isolation_jobs.spin_scan_pair)
    pair = ContractPair(
        name="c",
        client=SourceInput(source="x = 1"),
        schemas=[SourceInput(source="")],
        routes=[SourceInput(source="app.get('/c', h);", language="javascript")],
    )
    result = asyncio.run(module._scan_one(pair, ScanConfig()))
    assert result.passed is False
    assert result.parse_error is not None and "(deadline)" in result.parse_error


# ---------------------------------------------------------------------------
# Coverage accounting, busy, admission, fairness (round-3 findings)
# ---------------------------------------------------------------------------


def _app(module, *, tenant_header: bool = False):
    from fastapi import Header

    from auth.tenant import require_tenant

    def by_header(x_test_tenant: str = Header(TENANT)) -> str:
        return x_test_tenant

    app = FastAPI()
    app.include_router(module.router, prefix="/api/analysis/contracts")
    app.dependency_overrides[require_tenant] = by_header if tenant_header else (lambda: TENANT)

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


THREE = [
    {"path": "a.ts", "source": CLIENT_TS},
    {"path": "b.ts", "source": CLIENT_TS},
    {"path": "c.ts", "source": CLIENT_TS},
]


@pytest.mark.timeout(60)
def test_deadline_counts_only_processed_files_as_scanned(endpoint, monkeypatch, settings):
    module, client = endpoint
    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(module.jobs, "call_sites_job", isolation_jobs.first_file_then_spin)
    body = client.post(URL, json={"sources": THREE}).json()
    assert body["files_matched"] == 3
    assert body["files_scanned"] == 1  # only a.ts was processed
    assert body["files_unparsed"] == ["b.ts"] and body["files_unparsed_count"] == 1
    assert body["files_unparsed_reasons"]["b.ts"].startswith("deadline: ")
    assert body["files_not_reached"] == ["c.ts"] and body["files_not_reached_count"] == 1
    assert body["truncation_reason"] == "deadline"


@pytest.mark.timeout(60)
def test_a_job_that_processed_nothing_reports_zero_scanned(endpoint, monkeypatch, settings):
    module, client = endpoint
    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(module.jobs, "call_sites_job", isolation_jobs.spin_call_sites)
    body = client.post(URL, json={"sources": THREE}).json()
    assert body["files_scanned"] == 0
    assert body["files_unparsed"] == ["a.ts"]
    assert body["files_not_reached"] == ["b.ts", "c.ts"]
    assert body["files_not_reached_count"] == 2
    assert body["call_sites"] == []


def test_files_after_the_call_site_cap_are_not_reached(endpoint, in_process):
    _, client = endpoint
    src = "fetch('/a'); fetch('/b'); fetch(u); fetch(v);\n"
    sources = [{"path": f"{n}.ts", "source": src} for n in "abc"]
    body = client.post(URL, json={"sources": sources, "max_call_sites": 1}).json()
    assert body["truncation_reason"] == "max_call_sites"
    assert body["files_scanned"] == 1
    assert body["files_not_reached"] == ["b.ts", "c.ts"]
    assert body["files_scanned"] + body["files_unparsed_count"] + body[
        "files_not_reached_count"
    ] == len(sources)


def test_complete_scan_reports_nothing_unreached(endpoint):
    _, client = endpoint
    body = client.post(URL, json={"sources": THREE}).json()
    assert body["files_scanned"] == 3
    assert body["files_not_reached"] == [] and body["files_not_reached_count"] == 0


@pytest.mark.timeout(60)
@pytest.mark.parametrize("url", [URL, "/api/analysis/contracts/routes"])
def test_no_free_worker_is_503_with_retry_after_not_an_empty_partial(
    endpoint, monkeypatch, settings, url
):
    module, _ = endpoint
    import httpx

    from processing.contract_analysis import isolation
    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_MAX_WORKERS", 1)
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_QUEUE_WAIT_SECONDS", 0.3)
    hold = isolation.Limits(deadline_s=2.0, memory_mb=0, max_workers=1)
    app = _app(module)

    async def scenario():
        spinner = asyncio.create_task(isolation.stream(isolation_jobs.pid_then_spin, limits=hold))
        await asyncio.sleep(0.1)  # the only worker is taken
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            started = time.monotonic()
            resp = await client.post(url, json={"sources": THREE}, timeout=30)
            elapsed = time.monotonic() - started
        await spinner
        return resp, elapsed

    resp, elapsed = asyncio.run(scenario())
    assert resp.status_code == 503, resp.text
    assert resp.headers["retry-after"] == "1"
    assert "busy" in resp.text
    assert elapsed < 1.5, f"{elapsed:.2f}s"


@pytest.mark.timeout(60)
def test_admission_cap_refuses_before_reading_the_workspace(endpoint, monkeypatch, settings):
    module, _ = endpoint
    import httpx

    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 1.5)
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_MAX_PENDING", 1)
    monkeypatch.setattr(module.jobs, "call_sites_job", isolation_jobs.spin_call_sites)
    containers = MagicMock()
    containers.execute = AsyncMock(return_value={"stdout": "", "exit_code": 0})
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: containers)
    app = _app(module)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = asyncio.create_task(client.post(URL, json={"sources": THREE}, timeout=30))
            await asyncio.sleep(0.3)  # admitted and running
            started = time.monotonic()
            second = await client.post(URL, json={"container_name": "ws-1"}, timeout=30)
            elapsed = time.monotonic() - started
            return await first, second, elapsed

    first, second, elapsed = asyncio.run(scenario())
    assert second.status_code == 503 and "retry-after" in second.headers
    assert elapsed < 0.3, f"{elapsed:.2f}s"
    containers.execute.assert_not_awaited()  # refused before any file was listed
    assert first.status_code == 200 and first.json()["truncation_reason"] == "deadline"


@pytest.mark.timeout(60)
def test_one_tenant_cannot_take_every_worker(endpoint, monkeypatch, settings):
    module, _ = endpoint
    import httpx

    from tests.unit.contract_analysis import isolation_jobs

    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", 2.0)
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_MAX_WORKERS", 2)
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_MAX_PER_TENANT", 1)
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_QUEUE_WAIT_SECONDS", 0.3)
    real_job = module.jobs.call_sites_job
    app = _app(module, tenant_header=True)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:

            def post(tenant):
                return client.post(
                    URL, json={"sources": THREE}, headers={"x-test-tenant": tenant}, timeout=30
                )

            monkeypatch.setattr(module.jobs, "call_sites_job", isolation_jobs.spin_call_sites)
            hog = asyncio.create_task(post("a"))
            await asyncio.sleep(0.3)  # tenant a runs one (spinning) job
            monkeypatch.setattr(module.jobs, "call_sites_job", real_job)
            again_a = await post("a")
            other = await post("b")
            return await hog, again_a, other

    hog, again_a, other = asyncio.run(scenario())
    assert again_a.status_code == 503, again_a.text
    assert other.status_code == 200 and other.json()["files_scanned"] == 3
    assert hog.status_code == 200


@pytest.mark.timeout(60)
def test_scan_contracts_runs_the_checks_off_the_event_loop(endpoint, monkeypatch, settings):
    module, _ = endpoint
    import httpx

    from tests.unit.contract_analysis import isolation_jobs

    deadline = 1.5
    monkeypatch.setattr(settings, "CONTRACT_EXTRACTION_DEADLINE_SECONDS", deadline)
    monkeypatch.setattr(module.jobs, "scan_pair_job", isolation_jobs.spin_scan_pair)
    app = _app(module)
    pair = {
        "name": "c",
        "client": {"source": "import requests\nrequests.get('/api/users')"},
        "schemas": [{"source": ""}],
        "routes": [{"source": "@app.get('/api/users')\ndef u(): ...", "label": "r.py"}],
    }

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            scan = asyncio.create_task(
                client.post("/api/analysis/contracts/scan", json={"contracts": [pair]}, timeout=30)
            )
            pings = []
            while not scan.done():
                t0 = time.monotonic()
                assert (await client.get("/health")).status_code == 200
                pings.append(time.monotonic() - t0)
                await asyncio.sleep(0.05)
            return await scan, pings

    resp, pings = asyncio.run(scenario())
    assert resp.status_code == 200, resp.text
    assert len(pings) >= 5 and max(pings) < 0.5, pings
    assert "(deadline)" in resp.json()["contracts"][0]["parse_error"]


def test_scan_contracts_results_are_identical_in_the_worker(endpoint):
    module, client = endpoint
    from processing.contract_analysis.scanner import ContractScanner

    client_src = (
        "import requests\n"
        "def f():\n"
        "    requests.get('/api/users')\n"
        "    requests.post('/api/missing', json={'x': 1})\n"
    )
    routes_src = "@app.get('/api/users')\ndef users(): ...\n"
    pair = {
        "name": "p",
        "client": {"source": client_src},
        "schemas": [{"source": "class User:\n    name: str\n"}],
        "routes": [{"source": routes_src, "label": "routes.py"}],
    }
    body = client.post("/api/analysis/contracts/scan", json={"contracts": [pair]}).json()
    direct = ContractScanner().scan_sources(
        name="p",
        client_source=client_src,
        schema_sources=["class User:\n    name: str\n"],
        route_sources=[routes_src],
    )
    got = body["contracts"][0]
    assert got["passed"] == direct.passed
    assert got["stats"] == direct.stats
    assert [(v["check"], v["message"]) for v in got["violations"]] == [
        (v.check, v.message) for v in direct.violations
    ]
    assert direct.violations, "the pair should exercise at least one violation"
