"""The worker's in-workspace bug hunt: exec contract and fail-closed verification.

The coding-containers client is replaced by a fake that EXECUTES the real agent
locally (``python -I -P src/hunt_agent/hunt.py``) against a temporary git repo,
mapping the workspace paths (``/opt/hunt-agent``, ``/workspace``, ``/tmp``) onto
local ones. Each failure test then perturbs exactly one thing on the exec path
and asserts the hunt is recorded FAILED with that reason, never completed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.bug_hunter import _run_claimed_hunt
from app.api.models.bug_hunter_api_models import BugHuntRequest
from hunt_agent import hunt as agent
from services import workspace_hunt
from services.bug_hunt_report_store import BugHuntReportStore, build_finding_rows

AGENT_DIR = Path(agent.__file__).resolve().parent
CLEAN_PY = "def add(a, b):\n    return a + b\n"
BROKEN_PY = "def broken(:\n    pass\n"


def _env_without_git() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        env=_env_without_git(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "ok.py").write_text(CLEAN_PY, encoding="utf-8")
    (root / "bad.py").write_text(BROKEN_PY, encoding="utf-8")
    (root / "vendor").mkdir()
    (root / "vendor" / "v.py").write_text(BROKEN_PY, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


class FakeExecClient:
    """Runs the real agent locally; ``override`` can replace any one response."""

    def __init__(self, workspace: Path, tmp: Path) -> None:
        self.workspace = workspace
        self.tmp = tmp
        self.calls: list[dict[str, Any]] = []
        self.override: Any = None  # callable(argv, real_response) -> response

    def _map(self, arg: str) -> str:
        if arg.startswith(workspace_hunt.HUNT_AGENT_DIR + "/"):
            return str(AGENT_DIR / arg.rsplit("/", 1)[1])
        if arg == workspace_hunt.WORKSPACE_ROOT:
            return str(self.workspace)
        if arg.startswith("/tmp/"):
            return str(self.tmp / arg[len("/tmp/") :])
        return arg

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        argv = [self._map(a) for a in kwargs["command"]]
        assert argv[:3] == ["python3", "-I", "-P"]
        proc = subprocess.run(
            [sys.executable, *argv[1:]],
            cwd=kwargs["workdir"],
            env=_env_without_git(),
            capture_output=True,
            text=True,
            check=False,
        )
        response = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration": 0.1,
            "truncated": False,
            "bytes_omitted": 0,
        }
        # The agent echoes the path it was given; un-map it as the workspace would see it.
        response["stdout"] = response["stdout"].replace(f"{self.tmp}/", "/tmp/")
        if self.override is not None:
            response = self.override(kwargs["command"], response)
        return response


@pytest.fixture
def fake(repo: Path, tmp_path: Path):
    client = FakeExecClient(repo, tmp_path / "wstmp")
    (tmp_path / "wstmp").mkdir()
    with patch.object(workspace_hunt, "get_coding_containers_client", return_value=client):
        yield client


def _store() -> MagicMock:
    store = MagicMock(spec=BugHuntReportStore)
    store.renew_lease = AsyncMock(return_value=True)
    store.complete_hunt = AsyncMock()
    store.fail_hunt = AsyncMock()
    return store


def _job() -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "hunt_id": "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001",
        "container_name": "repo-1",
        "tenant_id": "tenant-1",
        "status": "running",
        "created_at": now,
        "request_payload": BugHuntRequest().model_dump(mode="json"),
    }


async def _run(store: MagicMock) -> None:
    manager = MagicMock()
    manager.get_container = AsyncMock(
        return_value=SimpleNamespace(work_dir="/app/data/repo", tenant_id="tenant-1")
    )
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=manager):
        await _run_claimed_hunt(job=_job(), worker_id="worker-1", store=store)


def _failed_reason(store: MagicMock) -> str:
    store.complete_hunt.assert_not_awaited()
    store.fail_hunt.assert_awaited_once()
    return store.fail_hunt.await_args.kwargs["error_message"]


def _is_hunt(command: list[str]) -> bool:
    return "--repo" in command


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_persists_verified_findings_and_provenance(fake, repo, monkeypatch):
    # Small chunks so the result is read back in several verified pieces.
    monkeypatch.setattr(workspace_hunt, "RESULT_CHUNK_BYTES", 97)
    store = _store()

    await _run(store)

    store.fail_hunt.assert_not_awaited()
    store.complete_hunt.assert_awaited_once()
    kwargs = store.complete_hunt.await_args.kwargs
    assert kwargs["codebase_path"] == "/workspace"  # the workspace root, not a pod path
    report = kwargs["report"]
    assert report.commit_sha == _git(repo, "rev-parse", "HEAD")
    assert report.worktree_dirty is False
    assert (report.scanned_files, report.tracked_source_files) == (2, 2)  # vendor/ excluded
    assert report.excluded_dirs == sorted(agent.EXCLUDED_DIRS)
    assert report.excluded_entry_kinds == ["non_utf8_path", "submodule", "symlink"]
    assert report.errors_found == 1
    assert report.errors[0]["file_path"] == "bad.py"
    assert report.fixes_generated == 0  # the source never leaves the workspace

    hunt_call = fake.calls[0]
    assert hunt_call["command"] == [
        "python3",
        "-I",
        "-P",
        "/opt/hunt-agent/hunt.py",
        "--repo",
        "/workspace",
        "--out",
        "/tmp/hunt-agent-0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001.json",
    ]
    assert hunt_call["workdir"] == "/"
    assert hunt_call["timeout"] == 300
    assert all(call["tenant_id"] == "tenant-1" for call in fake.calls)
    reads = [c for c in fake.calls if "--read-result" in c["command"]]
    assert len(reads) > 1
    assert "--remove-result" in fake.calls[-1]["command"]
    assert not list(fake.tmp.glob("hunt-agent-*.json"))  # cleaned up


async def test_dirty_workspace_is_a_visible_head_scan(fake, repo):
    (repo / "ok.py").write_text(BROKEN_PY, encoding="utf-8")  # uncommitted
    store = _store()

    await _run(store)

    report = store.complete_hunt.await_args.kwargs["report"]
    assert report.worktree_dirty is True
    assert report.commit_sha == _git(repo, "rev-parse", "HEAD")
    assert [e["file_path"] for e in report.errors] == ["bad.py"]  # not ok.py


# ---------------------------------------------------------------------------
# Fail-closed: each must be FAILED with a precise reason, never completed
# ---------------------------------------------------------------------------


async def test_agent_sha_mismatch_fails(fake, monkeypatch):
    monkeypatch.setattr(workspace_hunt, "expected_agent_sha", lambda: "0" * 64)
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: agent_sha_mismatch")


async def test_truncated_output_fails(fake):
    fake.override = lambda cmd, r: (
        {**r, "truncated": True, "bytes_omitted": 10} if _is_hunt(cmd) else r
    )
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: output_truncated")


async def test_missing_truncation_flag_fails(fake):
    def drop_flag(cmd, r):
        r = dict(r)
        r.pop("truncated")
        return r

    fake.override = drop_flag
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: output_truncated")


async def test_truncated_chunk_read_fails(fake):
    fake.override = lambda cmd, r: {**r, "truncated": True} if "--read-result" in cmd else r
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: output_truncated")


async def test_timeout_fails(fake):
    fake.override = lambda cmd, r: (
        {**r, "exit_code": 124, "stdout": "", "duration": 300.0} if _is_hunt(cmd) else r
    )
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: timeout")


async def test_nonzero_exit_fails(fake):
    fake.override = lambda cmd, r: {**r, "exit_code": 3} if _is_hunt(cmd) else r
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: nonzero_exit")


async def test_exec_transport_failure_fails(fake):
    # The client's encoding of an HTTP error, e.g. 404 for another tenant's workspace.
    fake.override = lambda cmd, r: {
        "exit_code": -1,
        "stdout": "",
        "stderr": "404 Not Found",
        "duration": 0.0,
    }
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: exec_failed")


async def test_bad_summary_json_fails(fake):
    fake.override = lambda cmd, r: {**r, "stdout": "not json"} if _is_hunt(cmd) else r
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: unparseable_summary")


async def test_bad_result_json_fails(fake):
    garbage = "{not json"
    expected = workspace_hunt.expected_agent_sha()

    def corrupt(cmd, r):
        if _is_hunt(cmd):
            summary = json.loads(r["stdout"])
            summary["result_bytes"] = len(garbage)
            summary["result_sha256"] = hashlib.sha256(garbage.encode()).hexdigest()
            summary["agent_sha"] = expected
            return {**r, "stdout": json.dumps(summary)}
        if "--read-result" in cmd:
            return {**r, "stdout": garbage}
        return r

    fake.override = corrupt
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: unparseable_result")


async def test_result_bytes_not_matching_the_printed_sha_fail(fake):
    fake.override = lambda cmd, r: (
        {**r, "stdout": r["stdout"][:-1] + "!"} if "--read-result" in cmd else r
    )
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: result_read_failed")


async def test_tree_sitter_version_mismatch_fails(fake, monkeypatch):
    real = agent.tree_sitter_versions()
    monkeypatch.setattr(
        workspace_hunt, "tree_sitter_versions", lambda: {**real, "tree-sitter": "0.0.1"}
    )
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: tree_sitter_version_mismatch")


async def test_log_hunt_does_not_exec_the_agent():
    """Error-log hunts keep the log path; only codebase scans go to the workspace."""
    client = MagicMock()
    client.execute = AsyncMock()
    store = _store()
    job = _job()
    job["request_payload"] = BugHuntRequest(error_logs="Traceback ...").model_dump(mode="json")
    manager = MagicMock()
    manager.get_container = AsyncMock(return_value=SimpleNamespace(work_dir="/app/data/repo"))
    with (
        patch.object(workspace_hunt, "get_coding_containers_client", return_value=client),
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=manager),
    ):
        await _run_claimed_hunt(job=job, worker_id="worker-1", store=store)

    client.execute.assert_not_awaited()
    report = store.complete_hunt.await_args.kwargs["report"]
    assert report.commit_sha is None and report.worktree_dirty is None


# ---------------------------------------------------------------------------
# Chunk reads: every chunk must be full length, and the loop is bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", ["", "x"], ids=["empty", "short"])
async def test_short_or_empty_chunk_fails_and_never_loops(fake, chunk):
    fake.override = lambda cmd, r: {**r, "stdout": chunk} if "--read-result" in cmd else r
    store = _store()

    await _run(store)

    reason = _failed_reason(store)
    assert reason.startswith(f"hunt_unverifiable: result_read_failed: chunk at 0: got {len(chunk)}")
    assert sum("--read-result" in c["command"] for c in fake.calls) == 1


async def test_chunk_read_loop_is_bounded_even_without_the_length_check(monkeypatch):
    """Defence in depth: a regressed per-chunk check must still terminate."""
    monkeypatch.setattr(workspace_hunt, "RESULT_CHUNK_BYTES", 4)
    replies = iter([{"exit_code": 0, "stdout": "ab", "truncated": False}] * 100)

    async def short_reads(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return next(replies)

    monkeypatch.setattr(workspace_hunt, "_exec", short_reads)
    with pytest.raises(workspace_hunt.HuntUnverifiableError) as caught:
        await workspace_hunt._read_result("c", "t", "/tmp/hunt-agent-x.json", 8, "0" * 64)
    assert caught.value.reason == "result_read_failed"


# ---------------------------------------------------------------------------
# Cleanup: on EVERY path, best effort, never changes the verdict
# ---------------------------------------------------------------------------


def _removes(fake: FakeExecClient) -> int:
    return sum("--remove-result" in c["command"] for c in fake.calls)


async def test_result_is_removed_after_an_agent_sha_mismatch(fake, monkeypatch):
    monkeypatch.setattr(workspace_hunt, "expected_agent_sha", lambda: "0" * 64)
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: agent_sha_mismatch")
    assert _removes(fake) == 1
    assert not list(fake.tmp.glob("hunt-agent-*.json"))


async def test_result_is_removed_after_a_read_failure(fake):
    fake.override = lambda cmd, r: {**r, "stdout": ""} if "--read-result" in cmd else r
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: result_read_failed")
    assert _removes(fake) == 1
    assert not list(fake.tmp.glob("hunt-agent-*.json"))


async def test_result_is_removed_after_a_tree_sitter_mismatch(fake, monkeypatch):
    real = agent.tree_sitter_versions()
    monkeypatch.setattr(
        workspace_hunt, "tree_sitter_versions", lambda: {**real, "tree-sitter": "0.0.1"}
    )
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: tree_sitter_version_mismatch")
    assert _removes(fake) == 1
    assert not list(fake.tmp.glob("hunt-agent-*.json"))


def _failing_cleanup(kind: str):
    def override(cmd, r):
        if "--remove-result" not in cmd:
            return r
        if kind == "raise":
            raise RuntimeError("transport down")
        return {**r, "exit_code": 3}

    return override


@pytest.mark.parametrize("kind", ["nonzero", "raise"])
async def test_cleanup_failure_keeps_a_successful_hunt_successful(fake, kind):
    fake.override = _failing_cleanup(kind)
    store = _store()

    await _run(store)

    store.fail_hunt.assert_not_awaited()
    store.complete_hunt.assert_awaited_once()
    assert _removes(fake) == 1


@pytest.mark.parametrize("kind", ["nonzero", "raise"])
async def test_cleanup_failure_keeps_a_failed_hunts_original_reason(fake, monkeypatch, kind):
    monkeypatch.setattr(workspace_hunt, "expected_agent_sha", lambda: "0" * 64)
    fake.override = _failing_cleanup(kind)
    store = _store()

    await _run(store)

    assert _failed_reason(store).startswith("hunt_unverifiable: agent_sha_mismatch")
    assert _removes(fake) == 1


# ---------------------------------------------------------------------------
# Non-UTF-8 tracked paths
# ---------------------------------------------------------------------------


def _commit_non_utf8_path(repo: Path) -> None:
    """Commit a broken .py at a path that is not valid UTF-8 (index only: APFS refuses it)."""
    env = _env_without_git()
    oid = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=repo,
            env=env,
            input=BROKEN_PY.encode(),
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    subprocess.run(
        [
            b"git",
            b"update-index",
            b"--add",
            b"--cacheinfo",
            f"100644,{oid},".encode() + b"bad\xff.py",
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    _git(repo, "commit", "-q", "-m", "non-utf8 path")
    assert (
        b"bad\xff.py"
        in subprocess.run(
            ["git", "ls-tree", "-z", "--name-only", "HEAD"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
        ).stdout
    )


async def test_non_utf8_path_is_excluded_from_both_counts_and_the_hunt_completes(fake, repo):
    _commit_non_utf8_path(repo)
    store = _store()

    await _run(store)

    store.fail_hunt.assert_not_awaited()
    report = store.complete_hunt.await_args.kwargs["report"]
    assert (report.scanned_files, report.tracked_source_files) == (2, 2)
    assert [e["file_path"] for e in report.errors] == ["bad.py"]


async def test_a_non_utf8_finding_is_a_precise_unverifiable(fake):
    """An agent that leaks such a path fails deterministically, not at the DB insert."""

    def inject(cmd, r):
        if not _is_hunt(cmd):
            return r
        summary = json.loads(r["stdout"])
        result = fake.tmp / Path(summary["result_path"]).name
        document = json.loads(result.read_text())
        document["findings"].append({**document["findings"][0], "file_path": "bad\udcff.py"})
        data = json.dumps(document, ensure_ascii=True).encode("ascii")
        result.write_bytes(data)
        summary["result_bytes"] = len(data)
        summary["result_sha256"] = hashlib.sha256(data).hexdigest()
        return {**r, "stdout": json.dumps(summary)}

    fake.override = inject
    store = _store()

    await _run(store)

    assert _failed_reason(store) == (
        "hunt_unverifiable: unparseable_result: a finding is not valid UTF-8"
    )


# ---------------------------------------------------------------------------
# Fingerprint + agent sha reach the persisted report (coding-analysis P1.8)
# ---------------------------------------------------------------------------


async def test_findings_carry_the_agent_fingerprint_and_the_hunt_its_agent_sha(fake, repo):
    store = _store()

    await _run(store)

    report = store.complete_hunt.await_args.kwargs["report"]
    (finding,) = [f for f in agent.hunt(repo)["findings"] if f["file_path"] == "bad.py"]
    assert report.errors[0]["fingerprint"] == finding["fingerprint"]
    assert report.agent_sha == workspace_hunt.expected_agent_sha()
    (row,) = build_finding_rows(report, codebase_path="/workspace")
    assert row["fingerprint"] == finding["fingerprint"]


@pytest.mark.parametrize("fingerprint", [None, "", "XYZ", "A" * 64])
async def test_a_finding_without_a_valid_fingerprint_fails_the_hunt(fake, fingerprint):
    def inject(cmd, r):
        if not _is_hunt(cmd):
            return r
        summary = json.loads(r["stdout"])
        result = fake.tmp / Path(summary["result_path"]).name
        document = json.loads(result.read_text())
        document["findings"][0]["fingerprint"] = fingerprint
        data = json.dumps(document, ensure_ascii=True).encode("ascii")
        result.write_bytes(data)
        summary["result_bytes"] = len(data)
        summary["result_sha256"] = hashlib.sha256(data).hexdigest()
        return {**r, "stdout": json.dumps(summary)}

    fake.override = inject
    store = _store()

    await _run(store)

    assert _failed_reason(store) == (
        "hunt_unverifiable: unparseable_result: a finding has no valid fingerprint"
    )


# ---------------------------------------------------------------------------
# excluded_entry_kinds: required by fr bug_report/v1, validated fail-closed
# ---------------------------------------------------------------------------

_MISSING = object()


def test_the_service_entry_kind_enum_equals_the_agents():
    # The service pins the enum as a literal (fr's schema pins the same three);
    # it must not drift from what the vendored agent emits.
    assert workspace_hunt.ENTRY_KINDS == frozenset(agent.EXCLUDED_ENTRY_KINDS)


@pytest.mark.parametrize(
    "kinds",
    [
        _MISSING,
        None,
        "symlink",
        ["symlink", "hardlink"],
        ["symlink", "symlink"],
        ["symlink", 3],
    ],
    ids=["missing", "null", "not-a-list", "unknown-value", "duplicate", "non-string"],
)
async def test_an_undeclared_or_unknown_entry_kind_fails_the_hunt(fake, kinds):
    def inject(cmd, r):
        if not _is_hunt(cmd):
            return r
        summary = json.loads(r["stdout"])
        result = fake.tmp / Path(summary["result_path"]).name
        document = json.loads(result.read_text())
        if kinds is _MISSING:
            del document["excluded_entry_kinds"]
        else:
            document["excluded_entry_kinds"] = kinds
        data = json.dumps(document, ensure_ascii=True).encode("ascii")
        result.write_bytes(data)
        summary["result_bytes"] = len(data)
        summary["result_sha256"] = hashlib.sha256(data).hexdigest()
        return {**r, "stdout": json.dumps(summary)}

    fake.override = inject
    store = _store()

    await _run(store)

    reason = _failed_reason(store)
    assert reason.startswith("hunt_unverifiable: unparseable_result: excluded_entry_kinds"), reason
