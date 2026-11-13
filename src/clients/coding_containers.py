"""Adapter for the coding-containers microservice.

Internals delegate to the typed :class:`coding_client.clients.CodingWorkspacesClient`
rather than hand-rolling httpx calls and workspace URLs. The public surface of
:class:`CodingContainersClient` (method names, signatures, return values,
failure semantics) is unchanged — this is an internals swap only.

Why: hand-built workspace URLs and hand-built query params are how
coding-realtime shipped a silent bug (it sent ``file_path=`` where the route
reads ``path=``; FastAPI ignored the unknown key, the route listed
``/workspace`` instead, and every read returned 200 with ``content: None``).
The typed client owns the paths and the parameter names, so that class of
defect cannot recur here.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any

import httpx
from coding_client.auth.outbound import optional_tenant_headers
from coding_client.clients import CodingWorkspacesClient, ServiceError

from config.settings import get_settings


def _identity_headers(tenant_id: str | None) -> dict[str, str]:
    """Identity headers for a call to coding-containers.

    The tenant is the CUSTOMER's, threaded from the authenticated caller — so
    this service is acting on behalf of someone. Under
    ``TRUST_TENANT_HEADER=false`` a bare ``X-Tenant-Id`` proves nothing, so we
    also present coding-analysis's OWN license in ``Authorization``.
    coding-containers honours the asserted tenant only when
    ``svc-coding-analysis``'s subject is in its ``CODING_ONBEHALF_SUBJECTS``.

    Never a forwarded copy of an inbound credential: several of these calls run
    in background analysis with no inbound request to borrow from.

    A missing tenant stays absent rather than being defaulted. Substituting a
    literal would attribute the call to a tenant that does not own the
    container — the previous RFC1918 default here is what leaked the code graphs
    on 2026-07-29.
    """
    return optional_tenant_headers(
        tenant_id, service_token=os.getenv("CODING_SERVICE_TOKEN") or None
    )


logger = logging.getLogger(__name__)

#: Read/write/pool budget for ordinary metadata calls (was ``httpx.Timeout(30.0)``).
_DEFAULT_TIMEOUT = 30.0

#: A slow quality tool is not a slow TCP handshake — the connect bound stays
#: short regardless of how long the read budget is. ``ServiceClient`` builds
#: ``httpx.Timeout(self.timeout)``, which would stretch *connect* to the read
#: budget too, so :class:`_WorkspacesClient` re-asserts this. Remove the
#: subclass once coding-client accepts a connect bound of its own.
_CONNECT_TIMEOUT = 5.0

#: Headroom over the server-side per-command budget, so coding-containers' own
#: ``timeout`` fires FIRST and returns a real exit_code/stderr.
_EXEC_TIMEOUT_HEADROOM = 15.0

#: ``ServiceClient`` retries ConnectError / ReadTimeout / PoolTimeout this many
#: times before re-raising. Module-level so tests can shrink it.
_MAX_RETRIES = 3

_HEALTH_TIMEOUT = 5.0


class _WorkspacesClient(CodingWorkspacesClient):
    """``CodingWorkspacesClient`` with the short connect bound preserved.

    Everything else — retries, ``ServiceError`` on >=400, JSON parsing — is the
    base class's, untouched.
    """

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=_CONNECT_TIMEOUT),
            )
        return self._client


class CodingContainersClient:
    """Async client for the coding-containers service.

    Thin adapter over :class:`CodingWorkspacesClient`: it owns this repo's
    return-value contract (``None``/``[]``/an error dict instead of raised
    exceptions) and translates ``ServiceError`` back into it.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or get_settings().CODING_CONTAINERS_URL).rstrip("/")
        # Keyed by read budget. The underlying httpx timeout is fixed at client
        # construction, so a per-call budget means a per-budget client. In
        # practice that is a handful of entries (30 / 45 / 75 / 135 / 195),
        # each with its own small connection pool.
        self._clients: dict[float, _WorkspacesClient] = {}

    def _workspaces(self, timeout: float = _DEFAULT_TIMEOUT) -> _WorkspacesClient:
        """Return the pooled workspaces client for this read budget."""
        client = self._clients.get(timeout)
        if client is None:
            # A client whose httpx transport was closed rebuilds it lazily in
            # ``_ensure_client``, so only a missing entry needs constructing.
            client = _WorkspacesClient(
                base_url=self._base_url,
                timeout=timeout,
                max_retries=_MAX_RETRIES,
            )
            self._clients[timeout] = client
        return client

    async def close(self) -> None:
        """Close every underlying HTTP client."""
        for client in list(self._clients.values()):
            await client.close()
        self._clients.clear()

    # ------------------------------------------------------------------
    # Container operations
    # ------------------------------------------------------------------

    async def execute(
        self,
        container_name: str,
        command: list[str] | str,
        workdir: str = "/workspace",
        timeout: int = 30,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command inside a container.

        Args:
            container_name: Name of the container to execute in.
            command: Command to run (list or string).
            workdir: Working directory inside the container.
            timeout: Execution timeout in seconds.
            tenant_id: Optional tenant identifier sent as X-Tenant-Id header.

        Returns:
            Dict with keys: exit_code, stdout, stderr, duration.
        """
        headers = _identity_headers(tenant_id)

        # coding-containers' ExecRequest requires ``command: list[str]`` and
        # enforces a whitelist keyed on ``command[0]`` (the basename), so
        # ``/bin/sh -c …`` is rejected. Split bare shell-style strings with
        # shlex so ``find /workspace -name '*.py' -type f`` becomes
        # ``["find", "/workspace", "-name", "*.py", "-type", "f"]`` — preserves
        # quoted arg handling, avoids a shell, and keeps the whitelist honest.
        # Callers that truly need shell features (pipes, redirects) must
        # invoke an allowlisted interpreter explicitly and pass a list.
        if isinstance(command, str):
            command_list: list[str] = shlex.split(command)
            if not command_list:
                logger.warning("Refusing to exec empty command in %s", container_name)
                return {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "empty command",
                    "duration": 0.0,
                }
        else:
            command_list = list(command)

        # The default budget (30s) is tuned for fast metadata calls, but a
        # quality tool can legitimately run for its whole server-side budget
        # (60/120/180s). ``CodingWorkspacesClient``'s timeout is fixed at
        # construction, so the per-call budget is honoured by selecting the
        # client built for it — a read bound a touch looser than the tool
        # budget, so coding-containers' own ``timeout`` fires FIRST and returns
        # a real exit_code/stderr rather than the client aborting at 30s and
        # masking every slow-but-healthy run as a failure.
        client = self._workspaces(float(timeout) + _EXEC_TIMEOUT_HEADROOM)
        try:
            result = await client.exec(
                container_name,
                headers=headers,
                command=command_list,
                workdir=workdir,
                timeout=timeout,
            )
        except (ServiceError, httpx.HTTPError) as exc:
            logger.warning(
                "coding-containers exec failed for %s: %s",
                container_name,
                exc,
            )
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "duration": 0.0,
            }
        if not isinstance(result, dict):
            # 204 / empty body. Previously this raised a JSON decode error out
            # of the adapter; an exec with no result is a failed exec.
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "empty response from coding-containers",
                "duration": 0.0,
            }
        return result

    async def get_files(
        self,
        container_name: str,
        path: str = "/",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get file listing from a container.

        Args:
            container_name: Name of the container.
            path: Directory path inside the container.
            tenant_id: Optional tenant identifier for scoping.

        Returns:
            List of file info dicts, or empty list on error.
        """
        headers = _identity_headers(tenant_id)
        try:
            # ``path=`` is the parameter the route actually reads. Passing it
            # through the typed client keeps the name honest.
            data = await self._workspaces().list_files(container_name, headers=headers, path=path)
        except (ServiceError, httpx.HTTPError) as exc:
            logger.warning(
                "coding-containers get_files failed for %s: %s",
                container_name,
                exc,
            )
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # The route keys a directory listing under "entries" -- see
            # ContainerFilesResponse in coding-containers
            # (path/type/content/entries). It has never emitted "files", so the
            # old `data.get("files", [])` matched nothing and this returned an
            # empty list for EVERY directory, silently: a 200 with a full
            # listing was reported as "no files here", indistinguishable from an
            # empty directory. Measured 2026-08-25 against a live workspace whose
            # README.md read back 12,298 bytes through the same client.
            entries = data.get("entries")
            if isinstance(entries, list):
                return entries
            # Tolerated, not expected: a caller-supplied or future shape that
            # really does use "files". Kept so this is strictly additive.
            files = data.get("files", [])
            return files if isinstance(files, list) else []
        return []

    async def read_file(
        self,
        container_name: str,
        file_path: str,
        tenant_id: str | None = None,
    ) -> str | None:
        """Read a file from a container.

        Args:
            container_name: Name of the container.
            file_path: Path inside the container.
            tenant_id: Optional tenant identifier.

        Returns:
            File content as string, or None on failure/404.
        """
        headers = _identity_headers(tenant_id)
        try:
            data = await self._workspaces().list_files(
                container_name, headers=headers, path=file_path
            )
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            logger.warning(
                "coding-containers read_file failed for %s/%s: %s",
                container_name,
                file_path,
                exc,
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning(
                "coding-containers read_file failed for %s/%s: %s",
                container_name,
                file_path,
                exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        content = data.get("content")
        return content if isinstance(content, str) else None

    async def get_container(
        self,
        container_name: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get container details by name. Returns None if not found."""
        headers = _identity_headers(tenant_id)
        try:
            data = await self._workspaces().get_workspace(container_name, headers=headers)
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            logger.warning(
                "coding-containers get_container failed for %s: %s",
                container_name,
                exc,
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning(
                "coding-containers get_container failed for %s: %s",
                container_name,
                exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        container = data.get("container", data)
        return container if isinstance(container, dict) else data

    async def list_containers(
        self,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all containers. Returns empty list on error."""
        headers = _identity_headers(tenant_id)
        try:
            data = await self._workspaces().list_workspaces(headers=headers)
        except (ServiceError, httpx.HTTPError) as exc:
            logger.warning("coding-containers list_containers failed: %s", exc)
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            containers = data.get("containers", [])
            return containers if isinstance(containers, list) else []
        return []

    async def health_check(self) -> bool:
        """Check if the coding-containers service is healthy."""
        # /health is not a workspaces route; it is the service's own liveness
        # endpoint, so it goes through the base client's generic GET.
        try:
            await self._workspaces().get("/health", timeout=_HEALTH_TIMEOUT)
        except Exception:
            return False
        return True


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_client_instance: CodingContainersClient | None = None


def get_coding_containers_client() -> CodingContainersClient:
    """Return a singleton CodingContainersClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = CodingContainersClient()
    return _client_instance
