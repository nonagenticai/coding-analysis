"""PostgreSQL persistence for quality-analysis results (coding-analysis#106).

One ``quality_results`` row per analysis run, keyed by tenant + container and
tagged with the commit that was analysed. The write path is ``POST
/quality/analyze`` (background task) and ``POST /quality/analyze-sync``; every
``GET /quality/{c}/…`` read goes through :meth:`QualityResultStore.latest` /
:meth:`QualityResultStore.history` and NEVER runs the analyzer.

Every query is tenant-scoped (``WHERE tenant_id = %s``). ``tenant_id`` is a
native ``uuid`` column, so it is SELECTed as ``tenant_id::text`` (see the note on
``TenantId`` in ``db/models.py``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid7

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from clients.coding_containers import get_coding_containers_client
from db.session import get_raw_connection
from models.architecture_models import QualityReport

logger = logging.getLogger(__name__)

#: Issues kept per row. The full count is always recorded in ``issues_total``
#: and ``issues_truncated`` says whether anything was dropped.
MAX_STORED_ISSUES = 1000

#: Per-file aggregate entries kept (the /summary route caps ``top_files_limit``
#: at 100, so this is exactly what any reader can ask for).
MAX_STORED_FILE_COUNTS = 100

QUALITY_STATUSES: frozenset[str] = frozenset({"complete", "partial", "inconclusive", "failed"})

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Hard ceiling on ``git rev-parse HEAD`` in the container. The lookup sits on
#: the request path of ``POST /analyze-sync`` and ``GET /gate`` (both auto-run
#: verifiers), so an unresponsive container costs at most this, then the commit
#: is recorded as unknown.
COMMIT_SHA_TIMEOUT_S = 5.0

#: Wait at most this long for a pooled connection on any store call.
STORE_POOL_TIMEOUT_S = 5.0

#: Server-side bound on every store statement. ``SET LOCAL`` ONLY: connections go
#: through pgbouncer in TRANSACTION pooling, where a session-level ``SET``
#: survives the transaction and is handed to the next client on that backend.
#: ``SET LOCAL`` ends with the transaction it was issued in.
_SET_LOCAL_STATEMENT_TIMEOUT = "SET LOCAL statement_timeout = '5s'"


class InvalidTenantIdError(ValueError):
    """The tenant id is not a UUID, so it cannot match the native ``uuid`` column."""


def validate_tenant_id(value: object) -> str:
    """Return ``value`` if it parses as a UUID, else raise :class:`InvalidTenantIdError`.

    ``quality_results.tenant_id`` is a native ``uuid`` column: a non-UUID tenant
    can own no row, and binding one makes PostgreSQL raise (a 500) instead of
    telling the caller what is wrong.
    """
    if isinstance(value, str):
        try:
            UUID(value)
        except ValueError:
            pass
        else:
            return value
    raise InvalidTenantIdError(f"tenant id is not a UUID: {value!r}")


_COLUMNS: tuple[str, ...] = (
    "id",
    "tenant_id",
    "container_name",
    "repo_url",
    "commit_sha",
    "status",
    "overall_score",
    "grade",
    "language",
    "tools_attempted",
    "tools_succeeded",
    "tools_failed",
    "summary",
    "issue_stats",
    "issues",
    "issues_total",
    "issues_truncated",
    "recommendations",
    "execution_time",
    "error_message",
    "created_at",
)

_SELECT = """
    SELECT id, tenant_id::text, container_name, repo_url, commit_sha, status,
           overall_score, grade, language, tools_attempted, tools_succeeded,
           tools_failed, summary, issue_stats, issues, issues_total,
           issues_truncated, recommendations, execution_time, error_message,
           created_at
      FROM quality_results
"""

#: Every column except ``created_at`` (server default ``now()``), in this order.
_INSERT_COLUMNS: tuple[str, ...] = tuple(name for name in _COLUMNS if name != "created_at")

_INSERT = """
    INSERT INTO quality_results
        (id, tenant_id, container_name, repo_url, commit_sha, status,
         overall_score, grade, language, tools_attempted, tools_succeeded,
         tools_failed, summary, issue_stats, issues, issues_total,
         issues_truncated, recommendations, execution_time, error_message)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


# ---------------------------------------------------------------------------
# Provenance: which commit / repository was analysed
# ---------------------------------------------------------------------------


def validate_commit_sha(value: object) -> str | None:
    """Return ``value`` only if it is a full lowercase 40-hex sha, else ``None``.

    Never a branch name, an abbreviated sha or ``HEAD``: a moving pointer would
    make two different trees look like one measurement.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if _SHA_RE.fullmatch(candidate) else None


def sanitize_repo_url(value: object) -> str | None:
    """Strip userinfo (tokens) from a repository URL; empty → ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme and parts.netloc and "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return raw


async def resolve_commit_sha(
    container_name: str, tenant_id: str | None, *, workdir: str = "/workspace"
) -> str | None:
    """``git rev-parse HEAD`` inside the tenant's container, validated — ``None``
    when the working tree is dirty (the checked-out code is then not that commit).

    Bounded by :data:`COMMIT_SHA_TIMEOUT_S` end to end (a timeout is ``None``).

    Uses the same coding-containers client and ``X-Tenant-Id`` scoping that the
    tool executor uses to run lint tools (``tool_execution/_base.py``). Any
    failure — exec error, non-zero exit, output that is not a 40-hex sha — is
    ``None``: an unknown commit is recorded as unknown, never guessed.
    """
    git = ["git", "-c", f"safe.directory={workdir}"]

    async def _git(*args: str) -> dict[str, Any] | None:
        resp = await client.execute(
            container_name=container_name,
            # ⛔ No shell: coding-containers' exec allowlist refuses `sh`
            # (400 "Command 'sh' not allowed"), which silently made every sha
            # unknown. Each call is a plain, allowlisted `git`.
            command=[*git, *args],
            workdir=workdir,
            timeout=int(COMMIT_SHA_TIMEOUT_S),
            tenant_id=tenant_id,
        )
        if not isinstance(resp, dict) or resp.get("exit_code") != 0:
            return None
        return resp

    try:
        client = get_coding_containers_client()
        # Both calls share ONE deadline. A sha only names the graded code when
        # the tree is clean: code edited but not committed (the upgrade
        # procedure's apply step) must never be keyed to HEAD, or a later gate
        # would reuse a grade of different bytes. Untracked files count.
        async with asyncio.timeout(COMMIT_SHA_TIMEOUT_S):
            head = await _git("rev-parse", "HEAD")
            if head is None:
                return None
            status = await _git("status", "--porcelain")
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort, never fatal
        logger.warning("Could not resolve commit sha for %s: %s", container_name, exc)
        return None
    if status is None:
        return None  # cleanliness unknown -> the commit is unknown
    if str(status.get("stdout") or "").strip():
        logger.info("Working tree of %s is dirty; commit sha recorded as unknown", container_name)
        return None
    return validate_commit_sha(head.get("stdout"))


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def derive_status(report: QualityReport | None) -> str:
    """Map a report to the stored status.

    ``None`` (the analyzer produced nothing) → ``failed``; ``inconclusive`` (no
    tool completed) → ``inconclusive``; ``partial`` (some tools failed) →
    ``partial``; otherwise ``complete``. Only ``complete`` rows are trend points.
    """
    if report is None:
        return "failed"
    if report.inconclusive:
        return "inconclusive"
    if report.partial:
        return "partial"
    return "complete"


def _issue_stats(report: QualityReport) -> dict[str, Any]:
    """Aggregates over the FULL issue list, so truncation never skews counts."""
    severity_counts: Counter[str] = Counter(
        str(issue.severity or "unknown").lower() for issue in report.issues
    )
    file_counts: Counter[str] = Counter(
        PurePosixPath(issue.file_path).name for issue in report.issues
    )
    return {
        "severity_counts": dict(severity_counts),
        "file_counts": dict(file_counts.most_common(MAX_STORED_FILE_COUNTS)),
        "files_with_issues": len(file_counts),
    }


def build_row(
    report: QualityReport | None,
    *,
    tenant_id: str,
    container_name: str,
    repo_url: str | None,
    commit_sha: str | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build the column → value mapping for one ``quality_results`` insert."""
    status = derive_status(report)
    row: dict[str, Any] = {
        "id": str(uuid7()),
        "tenant_id": tenant_id,
        "container_name": container_name,
        "repo_url": sanitize_repo_url(repo_url),
        "commit_sha": validate_commit_sha(commit_sha),
        "status": status,
        "overall_score": None,
        "grade": None,
        "language": None,
        "tools_attempted": [],
        "tools_succeeded": [],
        "tools_failed": [],
        "summary": {},
        "issue_stats": {},
        "issues": [],
        "issues_total": 0,
        "issues_truncated": False,
        "recommendations": [],
        "execution_time": None,
        "error_message": error_message,
    }
    if report is None:
        if row["error_message"] is None:
            row["error_message"] = "Analysis produced no results"
        return row

    issues = [issue.model_dump(mode="json") for issue in report.issues]
    row.update(
        {
            # The analyzer signals "nothing measured" with a -1.0 sentinel and
            # grade "N/A"; store NULL rather than a number that is not a score.
            "overall_score": None if status == "inconclusive" else report.overall_score,
            "grade": report.grade,
            "language": report.language,
            "tools_attempted": list(report.tools_attempted),
            "tools_succeeded": list(report.tools_succeeded),
            "tools_failed": list(report.tools_failed),
            "summary": dict(report.summary),
            "issue_stats": _issue_stats(report),
            "issues": issues[:MAX_STORED_ISSUES],
            "issues_total": len(issues),
            "issues_truncated": len(issues) > MAX_STORED_ISSUES,
            "recommendations": list(report.recommendations),
            "execution_time": report.execution_time,
        }
    )
    return row


_JSON_COLUMNS = frozenset(
    {
        "tools_attempted",
        "tools_succeeded",
        "tools_failed",
        "summary",
        "issue_stats",
        "issues",
        "recommendations",
    }
)


@asynccontextmanager
async def _bounded_transaction() -> AsyncIterator[AsyncConnection]:
    """A pooled connection inside one transaction with a 5s statement timeout.

    The pool wait is capped at :data:`STORE_POOL_TIMEOUT_S`; the timeout GUC is
    ``SET LOCAL`` (see :data:`_SET_LOCAL_STATEMENT_TIMEOUT`) and so dies with the
    transaction, which commits on a clean exit and rolls back on an exception.
    """
    async with get_raw_connection(timeout=STORE_POOL_TIMEOUT_S) as conn:
        async with conn.transaction():
            await conn.execute(_SET_LOCAL_STATEMENT_TIMEOUT)
            yield conn


class QualityResultStore:
    """Tenant-scoped writer and reader for ``quality_results``."""

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(_COLUMNS, row, strict=True))

    async def save(
        self,
        report: QualityReport | None,
        *,
        tenant_id: str,
        container_name: str,
        repo_url: str | None,
        commit_sha: str | None,
        error_message: str | None = None,
    ) -> str:
        """Persist one analysis run and return its id.

        ``report=None`` stores a ``failed`` row with no score.
        """
        validate_tenant_id(tenant_id)
        row = build_row(
            report,
            tenant_id=tenant_id,
            container_name=container_name,
            repo_url=repo_url,
            commit_sha=commit_sha,
            error_message=error_message,
        )
        params = tuple(
            Jsonb(row[name]) if name in _JSON_COLUMNS else row[name] for name in _INSERT_COLUMNS
        )
        async with _bounded_transaction() as conn:
            await conn.execute(_INSERT, params)
        return row["id"]

    async def latest(self, tenant_id: str, container_name: str) -> dict[str, Any] | None:
        """Newest stored run for this tenant's container, or ``None``."""
        validate_tenant_id(tenant_id)
        async with _bounded_transaction() as conn:
            cursor = await conn.execute(
                _SELECT
                + """
                 WHERE tenant_id = %s AND container_name = %s
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
                """,
                (tenant_id, container_name),
            )
            row = await cursor.fetchone()
        return self._from_row(row) if row else None

    async def latest_for_sha(
        self, tenant_id: str, container_name: str, commit_sha: str
    ) -> dict[str, Any] | None:
        """Newest stored run of this tenant's container AT ``commit_sha``, or ``None``."""
        validate_tenant_id(tenant_id)
        async with _bounded_transaction() as conn:
            cursor = await conn.execute(
                _SELECT
                + """
                 WHERE tenant_id = %s AND container_name = %s AND commit_sha = %s
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
                """,
                (tenant_id, container_name, commit_sha),
            )
            row = await cursor.fetchone()
        return self._from_row(row) if row else None

    async def history(
        self,
        tenant_id: str,
        container_name: str,
        limit: int = 50,
        *,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Stored runs, newest first, optionally only those created at/after ``since``."""
        validate_tenant_id(tenant_id)
        async with _bounded_transaction() as conn:
            cursor = await conn.execute(
                _SELECT
                + """
                 WHERE tenant_id = %s AND container_name = %s
                   AND (%s::timestamptz IS NULL OR created_at >= %s::timestamptz)
                 ORDER BY created_at DESC, id DESC
                 LIMIT %s
                """,
                (tenant_id, container_name, since, since, limit),
            )
            rows = await cursor.fetchall()
        return [self._from_row(row) for row in rows]


def get_quality_result_store() -> QualityResultStore:
    """FastAPI dependency: the stateless PostgreSQL quality-result store."""
    return QualityResultStore()
