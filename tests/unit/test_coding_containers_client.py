"""Unit tests for clients.coding_containers module.

The adapter delegates to ``coding_client.clients.CodingWorkspacesClient``, which
RAISES ``ServiceError`` on any >=400 instead of returning a response. These
tests pin the adapter's own contract — the return values callers depend on —
at the WIRE level via ``httpx.MockTransport``, so a wrong query-param name or a
truncated timeout cannot pass.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import clients.coding_containers as cc_mod
from clients.coding_containers import CodingContainersClient, get_coding_containers_client

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

BASE_URL = "http://fake:8691"


def _install_transport(monkeypatch, handler, *, max_retries: int = 1):
    """Route every workspaces client the adapter builds through ``handler``.

    Keeps the adapter's real timeout wiring intact (that is what the exec-budget
    tests assert on) and shrinks retries so a raised network error does not sit
    through tenacity's exponential backoff.
    """
    real = cc_mod._WorkspacesClient

    class _Transported(real):  # type: ignore[misc, valid-type]
        async def _ensure_client(self) -> httpx.AsyncClient:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout, connect=cc_mod._CONNECT_TIMEOUT),
                    transport=httpx.MockTransport(handler),
                )
            return self._client

    monkeypatch.setattr(cc_mod, "_WorkspacesClient", _Transported)
    monkeypatch.setattr(cc_mod, "_MAX_RETRIES", max_retries)


def _recorder(response_factory):
    """Return (seen, handler). ``response_factory`` takes the request."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response_factory(request)

    return seen, handler


def _json_handler(data, status_code=200):
    return _recorder(lambda _req: httpx.Response(status_code, json=data))


def _raising_handler(exc: Exception):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise exc

    return seen, handler


@pytest.fixture
def client():
    return CodingContainersClient(base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Internals: per-budget client pool
# ---------------------------------------------------------------------------


class TestWorkspacesPool:
    def test_creates_client_on_first_call(self, client):
        assert client._clients == {}
        ws = client._workspaces()
        assert ws is not None
        assert client._clients[cc_mod._DEFAULT_TIMEOUT] is ws

    def test_reuses_existing_client_for_same_budget(self, client):
        assert client._workspaces() is client._workspaces()

    def test_distinct_budgets_get_distinct_clients(self, client):
        assert client._workspaces(30.0) is not client._workspaces(135.0)

    async def test_close_closes_and_clears_every_client(self, client):
        ws = client._workspaces()
        await ws._ensure_client()
        await client.close()
        assert client._clients == {}
        assert ws._client is None

    async def test_close_noop_when_nothing_built(self, client):
        await client.close()
        assert client._clients == {}

    def test_connect_bound_stays_short(self, client):
        """A slow tool is not a slow TCP handshake.

        ``ServiceClient`` builds ``httpx.Timeout(self.timeout)``, which would
        stretch connect to the whole read budget. The local subclass re-asserts
        the 5s connect bound the raw-httpx implementation had.
        """
        ws = client._workspaces(135.0)
        assert ws.timeout == 135.0
        # asserted at the wire in TestExecute.test_exec_timeout_honors_tool_budget


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_execute_success(self, client, monkeypatch):
        expected = {"exit_code": 0, "stdout": "hello", "stderr": "", "duration": 0.5}
        seen, handler = _json_handler(expected)
        _install_transport(monkeypatch, handler)

        result = await client.execute("my-container", ["echo", "hello"])

        assert result == expected
        assert seen[0].method == "POST"
        assert seen[0].url.path == "/api/workspaces/my-container/exec"
        assert json.loads(seen[0].content)["command"] == ["echo", "hello"]

    async def test_execute_with_custom_workdir_and_timeout(self, client, monkeypatch):
        seen, handler = _json_handler({"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.1})
        _install_transport(monkeypatch, handler)

        await client.execute("ctr", "ls -la", workdir="/app", timeout=60)

        body = json.loads(seen[0].content)
        assert body["workdir"] == "/app"
        assert body["timeout"] == 60
        # Bare string commands go through shlex.split so coding-containers'
        # whitelist (keyed on command[0] basename) still sees "ls", not /bin/sh.
        assert body["command"] == ["ls", "-la"]

    async def test_exec_timeout_honors_tool_budget(self, client, monkeypatch):
        """Regression (Defect B), re-pinned after the typed-client migration.

        ``CodingWorkspacesClient``'s timeout is fixed at CONSTRUCTION, so a
        naive delegation would silently truncate every long quality-tool run at
        the 30s default. The adapter selects the client built for this call's
        budget instead; assert the effective per-request timeout on the wire.
        """
        ok = {"exit_code": 0, "stdout": "", "stderr": "", "duration": 90.0}
        seen, handler = _json_handler(ok)
        _install_transport(monkeypatch, handler)

        await client.execute("ctr", ["ruff", "check", "."], timeout=120)

        wire_timeout = seen[0].extensions["timeout"]
        assert wire_timeout["read"] >= 120
        assert wire_timeout["read"] == 135.0  # timeout + 15s headroom
        # connect stays short — a slow tool is not a slow TCP handshake.
        assert wire_timeout["connect"] == 5.0
        # and the pooled client really was built for that budget
        assert client._clients[135.0].timeout == 135.0

    async def test_execute_default_timeout_still_short_budget(self, client, monkeypatch):
        seen, handler = _json_handler({"exit_code": 0, "stdout": "", "stderr": "", "duration": 1.0})
        _install_transport(monkeypatch, handler)

        await client.execute("ctr", ["echo", "hi"])

        assert json.loads(seen[0].content)["timeout"] == 30
        assert seen[0].extensions["timeout"]["read"] == 45.0
        assert seen[0].extensions["timeout"]["connect"] == 5.0

    async def test_metadata_calls_keep_the_30s_budget(self, client, monkeypatch):
        """A long exec must not widen the budget of ordinary metadata calls."""
        seen, handler = _json_handler([])
        _install_transport(monkeypatch, handler)

        await client.execute("ctr", ["sleep", "100"], timeout=180)
        await client.get_files("ctr")

        assert seen[0].extensions["timeout"]["read"] == 195.0
        assert seen[1].extensions["timeout"]["read"] == 30.0

    async def test_execute_empty_command_short_circuits(self, client, monkeypatch):
        seen, handler = _json_handler({})
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", "   ")

        assert result == {
            "exit_code": -1,
            "stdout": "",
            "stderr": "empty command",
            "duration": 0.0,
        }
        assert seen == []

    async def test_execute_404_returns_error_dict(self, client, monkeypatch):
        _, handler = _json_handler({"detail": "no such workspace"}, status_code=404)
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", ["fail"])

        assert result["exit_code"] == -1
        assert result["stdout"] == ""
        assert result["duration"] == 0.0
        assert result["stderr"]

    async def test_execute_500_returns_error_dict(self, client, monkeypatch):
        _, handler = _json_handler({"error": "bad"}, status_code=500)
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", ["fail"])

        assert result["exit_code"] == -1
        assert result["stdout"] == ""
        assert "500" in result["stderr"]
        assert result["duration"] == 0.0

    async def test_execute_connection_error_returns_error_dict(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("connection refused"))
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", ["cmd"])

        assert result["exit_code"] == -1
        assert "connection refused" in result["stderr"]

    async def test_execute_read_timeout_returns_error_dict(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ReadTimeout("too slow"))
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", ["cmd"])

        assert result["exit_code"] == -1
        assert "too slow" in result["stderr"]

    async def test_execute_empty_body_is_a_failed_exec(self, client, monkeypatch):
        """204/empty body must not surface as a success with no exit_code."""
        _, handler = _recorder(lambda _r: httpx.Response(204))
        _install_transport(monkeypatch, handler)

        result = await client.execute("ctr", ["cmd"])

        assert result["exit_code"] == -1


# ---------------------------------------------------------------------------
# get_files()
# ---------------------------------------------------------------------------


class TestGetFiles:
    async def test_get_files_returns_list(self, client, monkeypatch):
        files = [{"name": "a.py", "size": 100}, {"name": "b.py", "size": 200}]
        seen, handler = _json_handler(files)
        _install_transport(monkeypatch, handler)

        result = await client.get_files("ctr", path="/src")

        assert result == files
        assert seen[0].url.path == "/api/workspaces/ctr/files"

    async def test_get_files_sends_path_not_file_path(self, client, monkeypatch):
        """Regression: the route reads ``path=``.

        coding-realtime sent ``file_path=``; FastAPI ignored the unknown key,
        the route defaulted to listing ``/workspace``, and every read returned
        200 with ``content: None``. The typed client owns this name now — this
        test is the guard that it stays owned.
        """
        seen, handler = _json_handler([])
        _install_transport(monkeypatch, handler)

        await client.get_files("ctr", path="/src/app")

        params = seen[0].url.params
        assert params["path"] == "/src/app"
        assert "file_path" not in params

    async def test_get_files_reads_entries_the_key_the_route_actually_sends(
        self, client, monkeypatch
    ):
        """The route keys a directory listing under ``entries``, never ``files``.

        This is the shape ContainerFilesResponse really returns
        (path/type/content/entries). The old code read ``data.get("files")``, so
        it produced an empty list for EVERY directory -- a 200 carrying a full
        listing reported as "no files here", indistinguishable from an empty
        directory. Found by probing a live workspace whose README.md read back
        12,298 bytes through this same client while get_files answered 0.
        """
        entries = [
            {"name": "app.py", "type": "file", "size": "1240", "permissions": "-rw-r--r--"},
            {"name": "src", "type": "directory", "size": "4096", "permissions": "drwxr-xr-x"},
        ]
        _, handler = _json_handler(
            {"path": "/workspace", "type": "directory", "content": None, "entries": entries}
        )
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == entries

    async def test_get_files_still_tolerates_a_files_key(self, client, monkeypatch):
        """Kept only so the fix is strictly additive.

        Nothing in coding-containers emits this shape -- asserting on it alone is
        what let the real bug survive, since a fictional payload can satisfy a
        parser that matches nothing real.
        """
        files = [{"name": "c.py"}]
        _, handler = _json_handler({"files": files})
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == files

    async def test_get_files_empty_directory_is_distinguishable_from_a_parse_miss(
        self, client, monkeypatch
    ):
        """A genuinely empty directory returns [] -- and so did the bug.

        Negative control: this is the one case where broken and correct agree, so
        it is pinned explicitly to stop it being read as proof the parser works.
        """
        _, handler = _json_handler(
            {"path": "/workspace/empty", "type": "directory", "content": None, "entries": []}
        )
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr", path="/workspace/empty") == []

    async def test_get_files_passes_tenant_id(self, client, monkeypatch):
        seen, handler = _json_handler([])
        _install_transport(monkeypatch, handler)

        await client.get_files("ctr", path="/", tenant_id="tenant-123")

        # Tenant is forwarded as the X-Tenant-Id header (coding-containers'
        # require_tenant reads the header, not a query param), never as a param.
        assert seen[0].headers["x-tenant-id"] == "tenant-123"
        assert "tenant_id" not in seen[0].url.params

    async def test_get_files_no_tenant_id_omitted(self, client, monkeypatch):
        seen, handler = _json_handler([])
        _install_transport(monkeypatch, handler)

        await client.get_files("ctr", path="/workspace")

        assert "x-tenant-id" not in seen[0].headers
        assert "tenant_id" not in seen[0].url.params

    async def test_get_files_404_returns_empty_list(self, client, monkeypatch):
        _, handler = _json_handler({"detail": "nope"}, status_code=404)
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == []

    async def test_get_files_500_returns_empty_list(self, client, monkeypatch):
        _, handler = _json_handler({"error": "nope"}, status_code=500)
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == []

    async def test_get_files_connection_error_returns_empty_list(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("down"))
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == []


# ---------------------------------------------------------------------------
# read_file()
# ---------------------------------------------------------------------------


class TestReadFile:
    async def test_read_file_success(self, client, monkeypatch):
        seen, handler = _json_handler({"content": "print('hello')"})
        _install_transport(monkeypatch, handler)

        result = await client.read_file("ctr", "/workspace/main.py")

        assert result == "print('hello')"
        assert seen[0].url.path == "/api/workspaces/ctr/files"

    async def test_read_file_sends_path_not_file_path(self, client, monkeypatch):
        """The adapter arg is ``file_path``; the WIRE param must be ``path``."""
        seen, handler = _json_handler({"content": "x"})
        _install_transport(monkeypatch, handler)

        await client.read_file("ctr", "/workspace/main.py")

        params = seen[0].url.params
        assert params["path"] == "/workspace/main.py"
        assert "file_path" not in params

    async def test_read_file_with_tenant_id(self, client, monkeypatch):
        seen, handler = _json_handler({"content": "data"})
        _install_transport(monkeypatch, handler)

        await client.read_file("ctr", "/f.txt", tenant_id="t-1")

        assert seen[0].headers["x-tenant-id"] == "t-1"
        assert "tenant_id" not in seen[0].url.params

    async def test_read_file_404_returns_none(self, client, monkeypatch):
        _, handler = _json_handler({"error": "not found"}, status_code=404)
        _install_transport(monkeypatch, handler)

        assert await client.read_file("ctr", "/missing.py") is None

    async def test_read_file_500_returns_none(self, client, monkeypatch):
        _, handler = _json_handler({"error": "boom"}, status_code=500)
        _install_transport(monkeypatch, handler)

        assert await client.read_file("ctr", "/broken.py") is None

    async def test_read_file_connection_error_returns_none(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("refused"))
        _install_transport(monkeypatch, handler)

        assert await client.read_file("ctr", "/some.py") is None

    async def test_read_file_empty_body_returns_none(self, client, monkeypatch):
        _, handler = _recorder(lambda _r: httpx.Response(204))
        _install_transport(monkeypatch, handler)

        assert await client.read_file("ctr", "/some.py") is None


# ---------------------------------------------------------------------------
# get_container()
# ---------------------------------------------------------------------------


class TestGetContainer:
    async def test_get_container_unwraps_container_key(self, client, monkeypatch):
        seen, handler = _json_handler({"container": {"container_name": "ctr"}})
        _install_transport(monkeypatch, handler)

        result = await client.get_container("ctr")

        assert result == {"container_name": "ctr"}
        assert seen[0].url.path == "/api/workspaces/ctr"

    async def test_get_container_passes_through_bare_dict(self, client, monkeypatch):
        _, handler = _json_handler({"container_name": "ctr", "status": "running"})
        _install_transport(monkeypatch, handler)

        assert await client.get_container("ctr") == {"container_name": "ctr", "status": "running"}

    async def test_get_container_404_returns_none(self, client, monkeypatch):
        _, handler = _json_handler({"error": "nope"}, status_code=404)
        _install_transport(monkeypatch, handler)

        assert await client.get_container("ctr") is None

    async def test_get_container_500_returns_none(self, client, monkeypatch):
        _, handler = _json_handler({"error": "boom"}, status_code=500)
        _install_transport(monkeypatch, handler)

        assert await client.get_container("ctr") is None

    async def test_get_container_connection_error_returns_none(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("refused"))
        _install_transport(monkeypatch, handler)

        assert await client.get_container("ctr") is None

    async def test_get_container_tenant_id_in_header(self, client, monkeypatch):
        seen, handler = _json_handler({"container": {}})
        _install_transport(monkeypatch, handler)

        await client.get_container("ctr", tenant_id="t-9")

        # Regression: tenant must travel in the header so require_tenant resolves
        # it; a query param is silently ignored by coding-containers.
        assert seen[0].headers["x-tenant-id"] == "t-9"


# ---------------------------------------------------------------------------
# list_containers()
# ---------------------------------------------------------------------------


class TestListContainers:
    async def test_list_containers_unwraps_containers_key(self, client, monkeypatch):
        seen, handler = _json_handler({"containers": [{"container_name": "a"}]})
        _install_transport(monkeypatch, handler)

        result = await client.list_containers()

        assert result == [{"container_name": "a"}]
        assert seen[0].url.path == "/api/workspaces/"

    async def test_list_containers_accepts_bare_list(self, client, monkeypatch):
        _, handler = _json_handler([{"container_name": "a"}])
        _install_transport(monkeypatch, handler)

        assert await client.list_containers() == [{"container_name": "a"}]

    async def test_list_containers_tenant_id_in_header(self, client, monkeypatch):
        seen, handler = _json_handler({"containers": []})
        _install_transport(monkeypatch, handler)

        await client.list_containers(tenant_id="t-7")

        assert seen[0].headers["x-tenant-id"] == "t-7"

    async def test_list_containers_404_returns_empty_list(self, client, monkeypatch):
        _, handler = _json_handler({"error": "nope"}, status_code=404)
        _install_transport(monkeypatch, handler)

        assert await client.list_containers() == []

    async def test_list_containers_500_returns_empty_list(self, client, monkeypatch):
        _, handler = _json_handler({"error": "boom"}, status_code=500)
        _install_transport(monkeypatch, handler)

        assert await client.list_containers() == []

    async def test_list_containers_connection_error_returns_empty_list(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("down"))
        _install_transport(monkeypatch, handler)

        assert await client.list_containers() == []


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


class TestHealthCheck:
    async def test_health_check_200_returns_true(self, client, monkeypatch):
        seen, handler = _json_handler({"status": "ok"})
        _install_transport(monkeypatch, handler)

        assert await client.health_check() is True
        assert seen[0].url.path == "/health"
        # Health is a liveness probe, not a work call — it keeps its own short budget.
        assert seen[0].extensions["timeout"]["read"] == 5.0

    async def test_health_check_503_returns_false(self, client, monkeypatch):
        _, handler = _json_handler({"status": "down"}, status_code=503)
        _install_transport(monkeypatch, handler)

        assert await client.health_check() is False

    async def test_health_check_connection_error_returns_false(self, client, monkeypatch):
        _, handler = _raising_handler(httpx.ConnectError("refused"))
        _install_transport(monkeypatch, handler)

        assert await client.health_check() is False


# ---------------------------------------------------------------------------
# ServiceError never escapes
# ---------------------------------------------------------------------------


class TestServiceErrorNeverEscapes:
    """The typed client RAISES on >=400; every adapter method must absorb it.

    Letting a ``ServiceError`` reach a caller that used to get ``None``/``[]``
    would be a regression worse than not migrating at all.
    """

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 500, 502, 503])
    async def test_no_status_escapes_any_method(self, client, monkeypatch, status):
        _, handler = _json_handler({"detail": "x"}, status_code=status)
        _install_transport(monkeypatch, handler)

        assert await client.get_files("ctr") == []
        assert await client.read_file("ctr", "/f.py") is None
        assert await client.get_container("ctr") is None
        assert await client.list_containers() == []
        assert await client.health_check() is False
        assert (await client.execute("ctr", ["cmd"]))["exit_code"] == -1


# ---------------------------------------------------------------------------
# Base URL handling
# ---------------------------------------------------------------------------


class TestBaseUrl:
    def test_explicit_base_url(self):
        c = CodingContainersClient(base_url="http://custom:9999/")
        assert c._base_url == "http://custom:9999"

    def test_trailing_slash_stripped(self):
        c = CodingContainersClient(base_url="http://host:1234///")
        assert not c._base_url.endswith("/")

    def test_default_from_settings(self):
        c = CodingContainersClient()
        # From conftest env: CODING_CONTAINERS_URL=http://localhost:8691
        assert "8691" in c._base_url

    def test_base_url_reaches_the_delegate(self):
        c = CodingContainersClient(base_url="http://custom:9999/")
        assert c._workspaces().base_url == "http://custom:9999"


# ---------------------------------------------------------------------------
# get_coding_containers_client() singleton
# ---------------------------------------------------------------------------


class TestGetCodingContainersClient:
    def test_returns_instance(self):
        with patch("clients.coding_containers._client_instance", None):
            assert isinstance(get_coding_containers_client(), CodingContainersClient)

    def test_returns_same_instance_on_repeat(self):
        with patch("clients.coding_containers._client_instance", None):
            a = get_coding_containers_client()
            with patch("clients.coding_containers._client_instance", a):
                b = get_coding_containers_client()
            assert a is b


# ---------------------------------------------------------------------------
# On-behalf-of: the service license on outbound calls
#
# Under TRUST_TENANT_HEADER=false a bare X-Tenant-Id proves nothing, so
# coding-containers refuses an asserted tenant unless the caller also proves who
# it is. These pin the credential half — without them a silently-dropped
# Authorization looks exactly like a passing test.
# ---------------------------------------------------------------------------


class TestServiceLicenseOnOutboundCalls:
    async def test_the_service_license_reaches_the_wire(self, client, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-analysis-token")
        seen, handler = _json_handler({"containers": []})
        _install_transport(monkeypatch, handler)

        await client.list_containers(tenant_id="tenant-abc")

        assert seen[0].headers["authorization"] == "Bearer svc-analysis-token"
        # The CUSTOMER's tenant must survive alongside the credential. Putting
        # the token into X-Tenant-Id instead would collapse every customer into
        # coding-analysis's own tenant.
        assert seen[0].headers["x-tenant-id"] == "tenant-abc"

    async def test_no_token_configured_sends_no_authorization(self, client, monkeypatch):
        """Absent is absent — never a fabricated or empty credential."""
        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        seen, handler = _json_handler({"containers": []})
        _install_transport(monkeypatch, handler)

        await client.list_containers(tenant_id="tenant-abc")

        assert "authorization" not in seen[0].headers
        assert seen[0].headers["x-tenant-id"] == "tenant-abc"

    async def test_an_unscoped_call_still_proves_who_is_asking(self, client, monkeypatch):
        """No tenant is not a reason to be anonymous."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-analysis-token")
        seen, handler = _json_handler({"containers": []})
        _install_transport(monkeypatch, handler)

        await client.list_containers()

        assert seen[0].headers["authorization"] == "Bearer svc-analysis-token"
        assert "x-tenant-id" not in seen[0].headers

    async def test_the_license_reaches_the_exec_path_too(self, client, monkeypatch):
        """The long-budget client is a DIFFERENT instance — it must be identical
        in every other respect, credentials included."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-analysis-token")
        seen, handler = _json_handler({"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.0})
        _install_transport(monkeypatch, handler)

        await client.execute("ctr", ["ruff", "check"], timeout=180, tenant_id="tenant-abc")

        assert seen[0].headers["authorization"] == "Bearer svc-analysis-token"
        assert seen[0].headers["x-tenant-id"] == "tenant-abc"


class TestDependencySnapshotSendsTheLicense:
    async def test_snapshot_fetch_carries_tenant_and_license(self, monkeypatch):
        from services.code_graph import dependency_snapshot as ds

        monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-analysis-token")
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(500)  # short-circuits before the DB write

        # Bind the REAL class before patching. `ds.httpx` is the httpx module
        # itself, so patching `ds.httpx.AsyncClient` is global — a fake that
        # calls `httpx.AsyncClient(...)` in its own constructor would recurse
        # into itself (RecursionError, not a useful failure).
        real_async_client = httpx.AsyncClient

        class _Client:
            def __init__(self, *a, **kw):
                self._inner = real_async_client(transport=httpx.MockTransport(handler))

            async def __aenter__(self):
                return self._inner

            async def __aexit__(self, *exc):
                await self._inner.aclose()
                return False

        monkeypatch.setattr(ds.httpx, "AsyncClient", _Client)
        monkeypatch.setattr(
            ds, "get_settings", lambda: type("S", (), {"CODING_SECURITY_URL": "http://sec:8693"})()
        )

        assert await ds.store_dependency_report_snapshot("c1", "tenant-abc") is False
        assert seen[0].headers["authorization"] == "Bearer svc-analysis-token"
        assert seen[0].headers["x-tenant-id"] == "tenant-abc"


# ---------------------------------------------------------------------------
# Delegation is real (not a coincidence of matching URLs)
# ---------------------------------------------------------------------------


class TestDelegatesToTypedClient:
    async def test_calls_go_through_the_typed_workspaces_methods(self, client):
        ws = AsyncMock()
        ws.list_files.return_value = {"content": "x"}
        ws.get_workspace.return_value = {"container": {}}
        ws.list_workspaces.return_value = []
        ws.exec.return_value = {"exit_code": 0, "stdout": "", "stderr": "", "duration": 0.0}

        with patch.object(client, "_workspaces", return_value=ws):
            await client.read_file("ctr", "/a.py", tenant_id="t")
            await client.get_container("ctr")
            await client.list_containers()
            await client.execute("ctr", ["echo"])

        # ``path=``, not ``file_path=`` — pinned at the typed-call boundary too.
        assert ws.list_files.await_args.kwargs["path"] == "/a.py"
        assert ws.list_files.await_args.args == ("ctr",)
        assert ws.get_workspace.await_args.args == ("ctr",)
        ws.list_workspaces.assert_awaited_once()
        assert ws.exec.await_args.kwargs["command"] == ["echo"]
