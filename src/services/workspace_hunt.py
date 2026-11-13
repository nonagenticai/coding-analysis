"""Run the bug hunt INSIDE the tenant workspace and verify what comes back.

The coding-analysis pod has no checkout of the tenant's repository (and must
not get one). The hunt agent (``hunt_agent/hunt.py``) is baked into the
workspace image at :data:`HUNT_AGENT_ENTRY` by coding-containers, which
vendors it from this repository. This module execs it through coding-containers'
``exec_in_workspace``, reads the result file back in verified chunks through the
same exec route, and FAILS CLOSED: anything it cannot verify raises
:class:`HuntUnverifiableError`, which the worker persists as ``status=failed`` with
a precise ``error_message``. There is no partial success.

Exec contract (one hunt)::

    python3 -I -P /opt/hunt-agent/hunt.py --repo /workspace --out /tmp/hunt-agent-<hunt_id>.json
        workdir=/  timeout=300

    python3 -I -P /opt/hunt-agent/hunt.py --read-result <out> --offset N --length M
        (repeated until result_bytes are read; each chunk must be complete)

    python3 -I -P /opt/hunt-agent/hunt.py --remove-result <out>
        (best effort, in a ``finally`` once the hunt exec was issued: it runs
        on EVERY outcome, and its own failure is logged and never changes the
        verdict either way)

Recheck contract (one finding, :func:`run_workspace_recheck`)::

    python3 -I -P /opt/hunt-agent/hunt.py --recheck --repo /workspace \
        --commit <hunt commit_sha> --path=<finding path> --out /tmp/hunt-agent-recheck-<id>.json

    then the same chunked --read-result and --remove-result as a hunt. Every
    failure raises :class:`RecheckUnverifiableError` (``recheck_unverifiable:
    <reason>``); a file the agent could not examine at that commit is one too.
    Only a result whose ``status`` is ``scanned`` can say "not reproduced".

Why not the workspace file-read route (``list_workspace_files``): it refuses any
path outside the repository workspace, so it cannot read ``/tmp``; and writing
the result inside the repository would put a foreign file into the tenant's
checkout. Chunked reads through the agent keep every byte verified: each chunk
must be untruncated and full length, and the reassembled file must match the
``result_sha256`` the agent printed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import hunt_agent
from clients.coding_containers import get_coding_containers_client
from hunt_agent.hunt import (
    MANIFEST_NAME,
    RECHECK_SCANNED,
    RECHECK_STATUSES,
    tree_sitter_versions,
)

logger = logging.getLogger(__name__)

#: Where coding-containers bakes the agent into the workspace image.
HUNT_AGENT_DIR = "/opt/hunt-agent"
HUNT_AGENT_ENTRY = f"{HUNT_AGENT_DIR}/hunt.py"

#: Same flags as every other interpreter run in a workspace (coding-analysis#104).
ISOLATED_PYTHON = ("python3", "-I", "-P")
NEUTRAL_WORKDIR = "/"

#: The repository checkout inside the workspace.
WORKSPACE_ROOT = "/workspace"

#: coding-containers' ceiling for one exec (ExecRequest.timeout le=300).
HUNT_EXEC_TIMEOUT_S = 300
#: A chunk read is a file read; it gets a short budget of its own.
READ_EXEC_TIMEOUT_S = 60
#: Below the exec route's 1 MiB per-stream output ceiling, with headroom.
RESULT_CHUNK_BYTES = 512 * 1024
#: A result larger than this is refused rather than read.
MAX_RESULT_BYTES = 64 * 1024 * 1024

#: coreutils ``timeout`` exit codes (TERM / KILL) from the exec wrapper.
_TIMEOUT_EXIT_CODES = frozenset({124, 137})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HUNT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def is_fingerprint(value: object) -> bool:
    """True for a well-formed finding fingerprint (lower-case sha256 hex)."""
    return isinstance(value, str) and bool(_FINGERPRINT_RE.match(value))


class HuntUnverifiableError(RuntimeError):
    """The hunt's result cannot be trusted; the hunt must be recorded as failed."""

    PREFIX = "hunt_unverifiable"

    def __init__(self, reason: str, explanation: str = "") -> None:
        # NOT ``detail``: the worker persists ``getattr(exc, "detail", exc)``
        # (the HTTPException convention), which would drop the reason prefix.
        self.reason = reason
        self.explanation = explanation
        super().__init__(f"{self.PREFIX}: {reason}" + (f": {explanation}" if explanation else ""))


class RecheckUnverifiableError(HuntUnverifiableError):
    """A finding recheck could not be verified; it is NEVER "not reproduced"."""

    PREFIX = "recheck_unverifiable"


@dataclass(frozen=True)
class WorkspaceHuntResult:
    """A verified agent result."""

    workspace_root: str
    agent_sha: str
    commit_sha: str
    worktree_dirty: bool
    scanned_files: int
    tracked_source_files: int
    excluded_dirs: list[str]
    # Tree entry kinds out of BOTH counts (symlink / submodule / non_utf8_path),
    # each one validated against the agent's enum. fr's bug_report/v1 requires it.
    excluded_entry_kinds: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)


def expected_agent_sha() -> str:
    """sha256 of the committed ``MANIFEST.sha256``, the agent this service expects."""
    manifest = Path(hunt_agent.__file__).resolve().parent / MANIFEST_NAME
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def result_path_for(hunt_id: str) -> str:
    if not _HUNT_ID_RE.match(hunt_id):
        raise ValueError(f"hunt id is not a safe file-name component: {hunt_id!r}")
    return f"/tmp/hunt-agent-{hunt_id}.json"  # nosec B108


#: The closed enum of excluded entry kinds, pinned HERE as a literal because fr's
#: bug_report/v1 schema pins the same three: a new agent kind must fail closed
#: until the consumer knows it (a test asserts it equals the agent's set today).
#: Mirrored by migration 0011's ck_bug_hunts_excluded_entry_kinds_enum.
ENTRY_KINDS: frozenset[str] = frozenset({"non_utf8_path", "submodule", "symlink"})


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tail(text: object, limit: int = 300) -> str:
    return str(text or "").strip()[-limit:]


async def _exec(
    container_name: str, tenant_id: str, argv: list[str], timeout: int
) -> dict[str, Any]:
    """One exec; transport, timeout and truncation are all failures."""
    client = get_coding_containers_client()
    try:
        # The client already allows headroom over the server budget; this outer
        # bound only guards against a hung transport.
        async with asyncio.timeout(timeout + 30):
            resp = await client.execute(
                container_name=container_name,
                command=[*ISOLATED_PYTHON, *argv],
                workdir=NEUTRAL_WORKDIR,
                timeout=timeout,
                tenant_id=tenant_id,
            )
    except TimeoutError as exc:
        raise HuntUnverifiableError("timeout", f"exec exceeded {timeout}s") from exc
    if not isinstance(resp, dict):
        raise HuntUnverifiableError("exec_failed", "no exec response")
    exit_code = resp.get("exit_code")
    truncated = resp.get("truncated")
    duration = resp.get("duration")
    if exit_code in _TIMEOUT_EXIT_CODES or (
        exit_code != 0 and isinstance(duration, int | float) and duration >= timeout
    ):
        raise HuntUnverifiableError("timeout", f"exit {exit_code} after {timeout}s budget")
    # A clipped stream is never evidence of anything, whatever the exit status.
    if truncated is True:
        raise HuntUnverifiableError(
            "output_truncated", f"bytes_omitted={resp.get('bytes_omitted')!r}"
        )
    if exit_code == -1:
        # The client's encoding of a transport/HTTP failure (incl. a 404 for a
        # workspace this tenant does not own).
        raise HuntUnverifiableError("exec_failed", _tail(resp.get("stderr")))
    if exit_code != 0:
        raise HuntUnverifiableError(
            "nonzero_exit",
            f"exit {exit_code}: {_tail(resp.get('stdout'))} {_tail(resp.get('stderr'))}".strip(),
        )
    if truncated is not False:
        # Completeness must be ASSERTED by the route; an absent flag is unknown.
        raise HuntUnverifiableError("output_truncated", f"truncated flag is {truncated!r}")
    return resp


def _parse_json(text: object, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise HuntUnverifiableError(reason, str(exc)[:200]) from exc
    if not isinstance(value, dict):
        raise HuntUnverifiableError(reason, "not a JSON object")
    return value


def _verify_identity(document: dict[str, Any], expected_sha: str, where: str) -> None:
    reported = document.get("agent_sha")
    if reported != expected_sha:
        raise HuntUnverifiableError(
            "agent_sha_mismatch", f"{where} reported {reported!r}, expected {expected_sha}"
        )


async def _read_result(
    container_name: str, tenant_id: str, out: str, size: int, digest: str
) -> bytes:
    data = bytearray()
    # Every chunk must be full length, so the read takes exactly this many
    # execs. The bound makes the loop terminate even if a chunk check regresses.
    max_reads = math.ceil(size / RESULT_CHUNK_BYTES)
    reads = 0
    while len(data) < size:
        if reads >= max_reads:
            raise HuntUnverifiableError(
                "result_read_failed", f"result not read in {max_reads} chunks"
            )
        reads += 1
        want = min(RESULT_CHUNK_BYTES, size - len(data))
        resp = await _exec(
            container_name,
            tenant_id,
            [
                HUNT_AGENT_ENTRY,
                "--read-result",
                out,
                "--offset",
                str(len(data)),
                "--length",
                str(want),
            ],
            READ_EXEC_TIMEOUT_S,
        )
        chunk = str(resp.get("stdout") or "").encode("ascii", "replace")
        if len(chunk) != want:
            raise HuntUnverifiableError(
                "result_read_failed", f"chunk at {len(data)}: got {len(chunk)} of {want} bytes"
            )
        data.extend(chunk)
    if hashlib.sha256(data).hexdigest() != digest:
        raise HuntUnverifiableError(
            "result_read_failed", "reassembled result does not match its sha256"
        )
    return bytes(data)


def _has_surrogate(value: object) -> bool:
    """True when a string anywhere in *value* holds a lone surrogate (not storable)."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return True
        return False
    if isinstance(value, dict):
        return any(_has_surrogate(k) or _has_surrogate(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_surrogate(item) for item in value)
    return False


def _verify_detector(document: dict[str, Any], expected_sha: str) -> None:
    """Same agent bytes AND same tree-sitter/grammar versions as this service."""
    _verify_identity(document, expected_sha, "result")
    expected_versions = tree_sitter_versions()
    if (
        document.get("tree_sitter_version") != expected_versions.get("tree-sitter")
        or document.get("grammar_versions") != expected_versions
    ):
        raise HuntUnverifiableError(
            "tree_sitter_version_mismatch",
            f"agent {document.get('grammar_versions')!r}, expected {expected_versions!r}",
        )


def _validate_findings(value: object) -> list[dict[str, Any]]:
    """A list of findings, each typed and fingerprinted, all storable as UTF-8."""
    if not isinstance(value, list) or not all(
        isinstance(f, dict) and isinstance(cast("dict[str, Any]", f).get("error_type"), str)
        for f in value
    ):
        raise HuntUnverifiableError("unparseable_result", "findings is not a list of findings")
    findings = cast("list[dict[str, Any]]", value)
    if not all(is_fingerprint(f.get("fingerprint")) for f in findings):
        # The fingerprint is the finding's identity for a later recheck; a
        # finding without one could never be rechecked, so the result is refused.
        raise HuntUnverifiableError("unparseable_result", "a finding has no valid fingerprint")
    if _has_surrogate(findings):
        # e.g. a non-UTF-8 path from an agent that did not exclude it: it would
        # fail at the database as a generic error; fail it here, precisely.
        raise HuntUnverifiableError("unparseable_result", "a finding is not valid UTF-8")
    return findings


def _validate_result(
    document: dict[str, Any], expected_sha: str, workspace_root: str
) -> WorkspaceHuntResult:
    _verify_detector(document, expected_sha)

    commit_sha = document.get("commit_sha")
    dirty = document.get("worktree_dirty")
    scanned = document.get("scanned_files")
    tracked = document.get("tracked_source_files")
    excluded = document.get("excluded_dirs")
    findings = document.get("findings")
    if not isinstance(commit_sha, str) or not _COMMIT_RE.match(commit_sha):
        raise HuntUnverifiableError("unparseable_result", f"commit_sha {commit_sha!r}")
    if not isinstance(dirty, bool):
        raise HuntUnverifiableError("unparseable_result", f"worktree_dirty {dirty!r}")
    if not (_is_count(scanned) and _is_count(tracked)):
        raise HuntUnverifiableError("unparseable_result", f"counts {scanned!r}/{tracked!r}")
    if cast("int", scanned) > cast("int", tracked):
        raise HuntUnverifiableError("unparseable_result", f"scanned {scanned} > tracked {tracked}")
    if not isinstance(excluded, list) or not all(isinstance(d, str) for d in excluded):
        raise HuntUnverifiableError("unparseable_result", "excluded_dirs is not a list of names")
    kinds = document.get("excluded_entry_kinds")
    if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
        # The entry-kind scope must be DECLARED; an undeclared one is unknown.
        raise HuntUnverifiableError("unparseable_result", "excluded_entry_kinds missing")
    if not set(kinds) <= ENTRY_KINDS or len(set(kinds)) != len(kinds):
        # An unknown (or repeated) kind is a scope this service cannot name to a
        # consumer; persisting it would publish a scope nobody declared.
        raise HuntUnverifiableError(
            "unparseable_result", f"excluded_entry_kinds {sorted(map(str, kinds))!r}"
        )
    checked = _validate_findings(findings)
    if _has_surrogate(excluded):
        # e.g. a non-UTF-8 path from an agent that did not exclude it: it would
        # fail at the database as a generic error; fail it here, precisely.
        raise HuntUnverifiableError("unparseable_result", "a finding is not valid UTF-8")
    return WorkspaceHuntResult(
        workspace_root=workspace_root,
        agent_sha=expected_sha,
        commit_sha=commit_sha,
        worktree_dirty=dirty,
        scanned_files=cast("int", scanned),
        tracked_source_files=cast("int", tracked),
        excluded_dirs=[str(name) for name in excluded],
        excluded_entry_kinds=sorted(cast("list[str]", kinds)),
        findings=checked,
    )


async def run_workspace_hunt(
    *,
    container_name: str,
    tenant_id: str,
    hunt_id: str,
    workspace_root: str = WORKSPACE_ROOT,
) -> WorkspaceHuntResult:
    """Run the agent in the tenant's workspace and return its verified result.

    The tenant is passed to every exec; coding-containers re-checks workspace
    ownership on each one (another tenant's workspace is a 404, i.e. a failed
    exec here).

    Raises:
        HuntUnverifiableError: on any exec failure, timeout, truncation, non-zero
            exit, unparseable output, agent sha or tree-sitter version mismatch.
    """
    expected_sha = expected_agent_sha()
    out = result_path_for(hunt_id)

    try:
        return await _hunt_and_read(container_name, tenant_id, workspace_root, out, expected_sha)
    finally:
        await _remove_result(container_name, tenant_id, out)


async def _remove_result(container_name: str, tenant_id: str, out: str) -> None:
    """Best-effort delete of the result file; never raises, never decides the verdict."""
    try:
        await _exec(
            container_name,
            tenant_id,
            [HUNT_AGENT_ENTRY, "--remove-result", out],
            READ_EXEC_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup only; the verdict is already decided
        logger.warning("hunt result cleanup failed for %s: %s", out, exc)


async def _run_agent_and_read(
    container_name: str, tenant_id: str, argv: list[str], out: str, expected_sha: str
) -> dict[str, Any]:
    """Exec the agent with *argv* (which writes *out*), then read *out* back verified."""
    resp = await _exec(container_name, tenant_id, argv, HUNT_EXEC_TIMEOUT_S)
    summary = _parse_json(resp.get("stdout"), "unparseable_summary")
    _verify_identity(summary, expected_sha, "agent")
    size = summary.get("result_bytes")
    digest = summary.get("result_sha256")
    if summary.get("ok") is not True or summary.get("result_path") != out:
        raise HuntUnverifiableError("unparseable_summary", "summary does not name the result file")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_RESULT_BYTES:
        raise HuntUnverifiableError("unparseable_summary", f"result_bytes {size!r}")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        raise HuntUnverifiableError("unparseable_summary", "result_sha256 missing")

    raw = await _read_result(container_name, tenant_id, out, size, digest)
    return _parse_json(raw.decode("ascii"), "unparseable_result")


async def _hunt_and_read(
    container_name: str, tenant_id: str, workspace_root: str, out: str, expected_sha: str
) -> WorkspaceHuntResult:
    document = await _run_agent_and_read(
        container_name,
        tenant_id,
        [HUNT_AGENT_ENTRY, "--repo", workspace_root, "--out", out],
        out,
        expected_sha,
    )
    return _validate_result(document, expected_sha, workspace_root)


# ---------------------------------------------------------------------------
# Recheck: one finding, at the hunted commit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkspaceRecheckResult:
    """A verified recheck: the file WAS scanned at ``commit_sha``; these are its findings."""

    agent_sha: str
    commit_sha: str
    path: str
    findings: list[dict[str, Any]] = field(default_factory=list)


def _validate_recheck(
    document: dict[str, Any], expected_sha: str, commit_sha: str, path: str
) -> WorkspaceRecheckResult:
    _verify_detector(document, expected_sha)
    if document.get("commit_sha") != commit_sha or document.get("path") != path:
        raise HuntUnverifiableError(
            "unparseable_result",
            f"result names {document.get('commit_sha')!r}:{document.get('path')!r}",
        )
    status = document.get("status")
    if status not in RECHECK_STATUSES:
        raise HuntUnverifiableError("unparseable_result", f"status {status!r}")
    if status != RECHECK_SCANNED:
        # The file was NOT examined at that commit (commit gone, path absent,
        # unparseable ...). That is never evidence that the finding is gone.
        raise HuntUnverifiableError(str(status), f"{path} at {commit_sha}")
    findings = _validate_findings(document.get("findings"))
    if any(f.get("file_path") != path for f in findings):
        raise HuntUnverifiableError("unparseable_result", "a finding names another file")
    return WorkspaceRecheckResult(
        agent_sha=expected_sha, commit_sha=commit_sha, path=path, findings=findings
    )


async def run_workspace_recheck(
    *,
    container_name: str,
    tenant_id: str,
    commit_sha: str,
    path: str,
    recheck_id: str,
    workspace_root: str = WORKSPACE_ROOT,
) -> WorkspaceRecheckResult:
    """Re-run the detector on ONE file at *commit_sha* in the tenant's workspace.

    Same exec client, tenant scoping, chunked sha-verified read and cleanup as
    :func:`run_workspace_hunt`.

    Raises:
        RecheckUnverifiableError: on ANY failure to obtain a verified scan of the
            file at that commit (exec failure, timeout, truncation, agent or
            tree-sitter drift, unknown commit, absent or unparseable file).
    """
    expected_sha = expected_agent_sha()
    try:
        if not _COMMIT_RE.match(commit_sha):
            raise HuntUnverifiableError("invalid_commit", repr(commit_sha))
        out = result_path_for(f"recheck-{recheck_id}")
    except ValueError as exc:
        raise RecheckUnverifiableError("invalid_request", str(exc)) from exc
    except HuntUnverifiableError as exc:
        raise RecheckUnverifiableError(exc.reason, exc.explanation) from exc
    try:
        document = await _run_agent_and_read(
            container_name,
            tenant_id,
            [
                HUNT_AGENT_ENTRY,
                "--recheck",
                "--repo",
                workspace_root,
                "--commit",
                commit_sha,
                # ``=`` form: a path starting with "-" must stay a value.
                f"--path={path}",
                "--out",
                out,
            ],
            out,
            expected_sha,
        )
        return _validate_recheck(document, expected_sha, commit_sha, path)
    except RecheckUnverifiableError:
        raise
    except HuntUnverifiableError as exc:
        raise RecheckUnverifiableError(exc.reason, exc.explanation) from exc
    finally:
        await _remove_result(container_name, tenant_id, out)
