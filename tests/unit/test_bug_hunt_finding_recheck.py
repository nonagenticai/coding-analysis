"""POST /api/bug-hunter/{c}/hunts/{h}/findings/{f}/recheck.

The exec client is the fake from ``test_workspace_hunt``: it runs the REAL agent
locally against a temporary git repository. The store is an in-memory one that
scopes every read by tenant exactly as the SQL does (the SQL scoping itself is
pinned in ``tests/integration``).

The contract under test: ``reproduced`` is true iff a finding with the stored
fingerprint is present in the file AT the hunt's commit; any failure to obtain
a verified scan is an error status (502 ``recheck_unverifiable``), NEVER a 200
with ``reproduced: false``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.endpoints import bug_hunter
from auth.tenant import require_tenant
from hunt_agent import hunt as agent
from services import workspace_hunt
from tests.unit.test_workspace_hunt import CLEAN_PY, FakeExecClient, _git

TENANT = "tenant-1"
OTHER_TENANT = "tenant-2"
CONTAINER = "repo-1"
HUNT_ID = "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001"
FINDING_ID = 7
URL = f"/api/bug-hunter/{CONTAINER}/hunts/{HUNT_ID}/findings/{FINDING_ID}/recheck"


class TenantScopedStore:
    """In-memory hunts/findings; every read filters by (hunt, container, tenant)."""

    def __init__(self) -> None:
        self.hunts: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.findings: dict[tuple[int, str, str, str], dict[str, Any]] = {}

    async def get_hunt(self, *, hunt_id: str, container_name: str, tenant_id: str):
        return self.hunts.get((hunt_id, container_name, tenant_id))

    async def get_finding(
        self, *, hunt_id: str, container_name: str, tenant_id: str, finding_id: int
    ):
        return self.findings.get((finding_id, hunt_id, container_name, tenant_id))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "ok.py").write_text(CLEAN_PY, encoding="utf-8")
    (root / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def fake(repo: Path, tmp_path: Path):
    (tmp_path / "wstmp").mkdir()
    client = FakeExecClient(repo, tmp_path / "wstmp")
    with patch.object(workspace_hunt, "get_coding_containers_client", return_value=client):
        yield client


@pytest.fixture
def hunted(repo: Path) -> dict[str, Any]:
    """The real agent's hunt of the repo, as the worker would have persisted it."""
    return agent.hunt(repo)


def _seed(store: TenantScopedStore, hunted: dict[str, Any], **job_overrides: Any) -> dict:
    (finding,) = [f for f in hunted["findings"] if f["file_path"] == "bad.py"]
    now = datetime.now(UTC)
    job = {
        "hunt_id": HUNT_ID,
        "container_name": CONTAINER,
        "tenant_id": TENANT,
        "status": "completed",
        "commit_sha": hunted["commit_sha"],
        "scanned_files": hunted["scanned_files"],
        "tracked_source_files": hunted["tracked_source_files"],
        "worktree_dirty": hunted["worktree_dirty"],
        "agent_sha": workspace_hunt.expected_agent_sha(),
        "created_at": now,
        "updated_at": now,
        **job_overrides,
    }
    store.hunts[(HUNT_ID, CONTAINER, TENANT)] = job
    store.findings[(FINDING_ID, HUNT_ID, CONTAINER, TENANT)] = {
        "finding_id": FINDING_ID,
        "file_path": "bad.py",
        "line_number": finding["line_number"],
        "fingerprint": finding["fingerprint"],
        "bug": finding,
    }
    return finding


@pytest.fixture
def store() -> TenantScopedStore:
    return TenantScopedStore()


@pytest.fixture
def client(store: TenantScopedStore) -> TestClient:
    app = FastAPI()

    def tenant_from_header(request: Request) -> str:
        return request.headers["X-Tenant-Id"]

    app.dependency_overrides[require_tenant] = tenant_from_header
    app.dependency_overrides[bug_hunter.get_bug_hunt_report_store] = lambda: store
    app.include_router(bug_hunter.router, prefix="/api/bug-hunter")
    return TestClient(app)


def _post(client: TestClient, tenant: str = TENANT):
    return client.post(URL, headers={"X-Tenant-Id": tenant})


def _error(resp) -> dict:
    return resp.json()["detail"]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_finding_present_at_the_hunted_commit_is_reproduced(client, store, hunted, fake):
    finding = _seed(store, hunted)

    resp = _post(client)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reproduced"] is True
    assert body["reason"] == "fingerprint_present_at_commit"
    assert body["finding_fingerprint"] == finding["fingerprint"]
    assert body["matched_finding"] == finding
    assert body["commit_sha"] == hunted["commit_sha"]
    assert body["agent_sha"] == workspace_hunt.expected_agent_sha()
    assert body["path"] == "bad.py"

    recheck = fake.calls[0]["command"]
    assert recheck[3:] == [
        "/opt/hunt-agent/hunt.py",
        "--recheck",
        "--repo",
        "/workspace",
        "--commit",
        hunted["commit_sha"],
        "--path=bad.py",
        "--out",
        recheck[-1],
    ]
    assert recheck[-1].startswith("/tmp/hunt-agent-recheck-")
    assert all(call["tenant_id"] == TENANT for call in fake.calls)
    assert "--remove-result" in fake.calls[-1]["command"]
    assert not list(fake.tmp.glob("hunt-agent-*.json"))


def test_finding_fixed_at_the_hunted_commit_is_not_reproduced(client, store, hunted, fake, repo):
    """The hunt row names a commit where bad.py is already clean: not reproduced."""
    _seed(store, hunted)
    (repo / "bad.py").write_text(CLEAN_PY, encoding="utf-8")
    _git(repo, "commit", "-qam", "fix")
    fixed = _git(repo, "rev-parse", "HEAD")
    store.hunts[(HUNT_ID, CONTAINER, TENANT)]["commit_sha"] = fixed

    resp = _post(client)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["reproduced"], body["reason"]) == (False, "fingerprint_absent_at_commit")
    assert body["matched_finding"] is None
    assert body["commit_sha"] == fixed


def test_working_tree_fix_does_not_hide_the_finding(client, store, hunted, fake, repo):
    """The fix exists only in the workspace's working tree: still reproduced at the commit."""
    _seed(store, hunted)
    (repo / "bad.py").write_text(CLEAN_PY, encoding="utf-8")  # not committed

    resp = _post(client)

    assert resp.status_code == 200, resp.text
    assert resp.json()["reproduced"] is True


# ---------------------------------------------------------------------------
# Tenant scope and preconditions
# ---------------------------------------------------------------------------


def test_another_tenants_hunt_is_404_and_nothing_is_executed(client, store, hunted, fake):
    _seed(store, hunted)

    resp = _post(client, tenant=OTHER_TENANT)

    assert resp.status_code == 404
    assert fake.calls == []


def test_unknown_finding_is_404(client, store, hunted, fake):
    _seed(store, hunted)
    store.findings.clear()

    assert _post(client).status_code == 404
    assert fake.calls == []


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_hunt_not_completed_is_409(client, store, hunted, fake, status):
    _seed(store, hunted, status=status)

    resp = _post(client)

    assert resp.status_code == 409
    assert _error(resp)["error_code"] == "HUNT_NOT_COMPLETED"
    assert fake.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"commit_sha": None},
        {"scanned_files": 1, "tracked_source_files": 2},
        {"scanned_files": None},
        {"agent_sha": None},
    ],
)
def test_incomplete_hunt_is_422(client, store, hunted, fake, overrides):
    _seed(store, hunted, **overrides)

    resp = _post(client)

    assert resp.status_code == 422
    assert _error(resp)["error_code"] == "HUNT_INCOMPLETE"
    assert fake.calls == []


def test_finding_without_fingerprint_is_422(client, store, hunted, fake):
    _seed(store, hunted)
    store.findings[(FINDING_ID, HUNT_ID, CONTAINER, TENANT)]["fingerprint"] = None

    resp = _post(client)

    assert resp.status_code == 422
    assert _error(resp)["error_code"] == "FINDING_NOT_FINGERPRINTED"


def test_hunt_by_another_agent_is_409_agent_drift(client, store, hunted, fake):
    _seed(store, hunted, agent_sha="f" * 64)

    resp = _post(client)

    assert resp.status_code == 409
    assert _error(resp)["error_code"] == "AGENT_DRIFT"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Infrastructure failure is NEVER "not reproduced"
# ---------------------------------------------------------------------------


def _is_recheck(command: list[str]) -> bool:
    return "--recheck" in command


def _assert_unverifiable(resp, reason: str) -> None:
    assert resp.status_code == 502, resp.text
    detail = _error(resp)
    assert detail["error_code"] == "RECHECK_UNVERIFIABLE"
    assert detail["details"]["reason"] == reason
    assert detail["message"].startswith(f"recheck_unverifiable: {reason}")
    assert "reproduced" not in resp.text


def test_agent_sha_mismatch_in_the_workspace_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    fake.override = lambda cmd, r: (
        {**r, "stdout": r["stdout"].replace(agent.compute_agent_sha(), "0" * 64)}
        if _is_recheck(cmd)
        else r
    )

    _assert_unverifiable(_post(client), "agent_sha_mismatch")
    assert "--remove-result" in fake.calls[-1]["command"]


def test_timeout_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    fake.override = lambda cmd, r: {**r, "exit_code": 124} if _is_recheck(cmd) else r

    _assert_unverifiable(_post(client), "timeout")
    assert "--remove-result" in fake.calls[-1]["command"]


def test_truncated_output_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    fake.override = lambda cmd, r: (
        {**r, "truncated": True, "bytes_omitted": 5} if _is_recheck(cmd) else r
    )

    _assert_unverifiable(_post(client), "output_truncated")


def test_truncated_chunk_read_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    fake.override = lambda cmd, r: {**r, "truncated": True} if "--read-result" in cmd else r

    _assert_unverifiable(_post(client), "output_truncated")


def test_corrupted_chunk_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)

    def flip(cmd, r):
        if "--read-result" in cmd:
            text = r["stdout"]
            return {**r, "stdout": ("X" if text[0] != "X" else "Y") + text[1:]}
        return r

    fake.override = flip

    _assert_unverifiable(_post(client), "result_read_failed")


def test_transport_failure_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    fake.override = lambda cmd, r: (
        {**r, "exit_code": -1, "stdout": "", "stderr": "HTTP 404"} if _is_recheck(cmd) else r
    )

    _assert_unverifiable(_post(client), "exec_failed")


def test_commit_gone_from_the_workspace_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted, commit_sha="0" * 40)

    _assert_unverifiable(_post(client), "commit_not_found")


def test_path_absent_at_the_commit_is_unverifiable(client, store, hunted, fake):
    _seed(store, hunted)
    store.findings[(FINDING_ID, HUNT_ID, CONTAINER, TENANT)]["file_path"] = "gone.py"

    _assert_unverifiable(_post(client), "path_not_found")


def test_tree_sitter_drift_is_unverifiable(client, store, hunted, fake, monkeypatch):
    _seed(store, hunted)
    monkeypatch.setattr(
        workspace_hunt, "tree_sitter_versions", lambda: {"tree-sitter": "0.0.0-other"}
    )

    _assert_unverifiable(_post(client), "tree_sitter_version_mismatch")
