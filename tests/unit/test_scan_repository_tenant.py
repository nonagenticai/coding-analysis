"""scanRepository forwards the CALLER's tenant on every coding-containers exec.

coding-containers' ``/exec`` is tenant-scoped. An exec that arrives without
``X-Tenant-Id`` is resolved as coding-analysis's own service account, and the
BOLA guard answers 404 for every customer workspace -- the defect
coding-analysis#91 fixed for the quality path. The scan endpoint validated the
container under the caller's tenant and then dropped it on every exec.

These drive the real endpoint -> scanner -> ``execute_in_container`` chain and
assert on the one thing that reaches coding-containers: the client call.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.repo_scanner_modules.scanner_endpoints import scan_repository
from app.api.models.repo_scanner_api_models import RepoScannerScanRequest
from app.core.container_manager import ContainerInfo

TENANT_A = "0195f48b-d000-7000-8000-00000000000a"

_FD = "app.api.endpoints.repo_scanner_modules.file_discovery"
_EP = "app.api.endpoints.repo_scanner_modules.scanner_endpoints"
_CORE = "app.api.endpoints.repo_scanner_modules.scanner_core"


def _manager() -> MagicMock:
    mgr = MagicMock()
    mgr.list_containers = AsyncMock(
        return_value=[ContainerInfo(container_name="ws", status="running")]
    )
    mgr.get_container = AsyncMock(return_value=None)
    return mgr


def _cc_client() -> MagicMock:
    async def _execute(container_name, command, workdir="/workspace", timeout=30, tenant_id=None):
        text = command if isinstance(command, str) else " ".join(command)
        if text.startswith("python3"):
            return {"exit_code": 0, "stdout": json.dumps([]), "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    client = MagicMock()
    client.execute = AsyncMock(side_effect=_execute)
    return client


@pytest.mark.parametrize("action", ["dependencies", "imports", "validation", "full_scan"])
async def test_every_exec_carries_the_callers_tenant(action: str) -> None:
    mgr = _manager()
    client = _cc_client()
    with (
        patch(f"{_EP}.get_container_manager", return_value=mgr),
        patch(f"{_FD}.get_container_manager", return_value=mgr),
        patch(f"{_CORE}.get_container_manager", return_value=mgr),
        patch(f"{_FD}.get_coding_containers_client", return_value=client),
    ):
        await scan_repository(
            RepoScannerScanRequest(container_name="ws", action=action, target_files=["a.py"]),
            tenant_id=TENANT_A,
        )

    assert client.execute.await_count > 0, "the scan must reach coding-containers"
    tenants = {call.kwargs.get("tenant_id") for call in client.execute.await_args_list}
    assert tenants == {TENANT_A}
    # The container lookup that gates each exec is scoped the same way.
    for call in mgr.list_containers.await_args_list:
        assert call.kwargs["tenant_id"] == TENANT_A
