"""PostgreSQL persistence for asynchronous Bug Hunter reports.

Hunt metadata and narrative summaries are stored once in ``bug_hunts``.  Each
bug is stored as a separate row so callers can retrieve a single file or a
directory subtree without loading a repository-wide result document.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

from psycopg.types.json import Jsonb

from db.session import get_raw_connection
from models.analysis_models import UNRATED_SEVERITY, BugHuntingResponse

# ``tenant_id::text`` is not decoration. The column is a native ``uuid``
# (migration 0005) and psycopg3 therefore returns ``uuid.UUID`` for it — but the
# claimed hunt's tenant is carried straight into an outbound ``X-Tenant-Id``
# header (httpx rejects a non-str header value) and compared with ``!=`` against
# the container's ``tenant_id`` label, which arrives from coding-containers as
# JSON text. Casting in SQL keeps every consumer seeing exactly what it saw
# before the column changed type. Same for the two RETURNING clauses below.
_HUNT_SELECT = """
    SELECT id, code_graph_id, tenant_id::text, status, errors_found, fixes_generated,
           root_cause_analysis, fix_summary, estimated_effort, risk_assessment,
           analysis_time, error_message, created_at, started_at, completed_at,
           updated_at, scanned_files, tracked_source_files, commit_sha,
           excluded_dirs, worktree_dirty, agent_sha, excluded_entry_kinds
      FROM bug_hunts
"""

_HUNT_FIELDS = (
    "hunt_id",
    "container_name",
    "tenant_id",
    "status",
    "errors_found",
    "fixes_generated",
    "root_cause_analysis",
    "fix_summary",
    "estimated_effort",
    "risk_assessment",
    "analysis_time",
    "error_message",
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
    "scanned_files",
    "tracked_source_files",
    "commit_sha",
    "excluded_dirs",
    "worktree_dirty",
    "agent_sha",
    "excluded_entry_kinds",
)

_CLAIM_FIELDS = (*_HUNT_FIELDS, "request_payload", "worker_id")


def normalize_query_path(value: str) -> str:
    """Normalize a client-supplied repository-relative file or directory path.

    Empty string and ``.`` select the repository root. Absolute paths and parent
    traversal are rejected because report paths are repository-relative.
    """
    normalized = value.strip().replace("\\", "/")
    if normalized in {"", "."}:
        return ""
    if normalized.startswith("/"):
        raise ValueError("path must be relative to the repository root")

    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise ValueError("path must not contain parent-directory traversal")
    return "/".join(parts)


def _normalize_finding_path(value: str | None, codebase_path: str) -> str:
    """Return a stable repository-relative path for a stored finding."""
    raw = (value or "").strip().replace("\\", "/")
    root = codebase_path.rstrip("/").replace("\\", "/")
    if not raw:
        return "<unknown>"
    if root and raw == root:
        return "<repository-root>"
    if root and raw.startswith(f"{root}/"):
        raw = raw[len(root) + 1 :]
    elif raw.startswith("/"):
        # Stack traces can reference an absolute path outside the configured
        # checkout root. Keep it queryable while never exposing an absolute
        # filesystem path through the report API.
        raw = raw.lstrip("/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw or "<unknown>"


def _escape_like(value: str) -> str:
    """Escape a literal value for a PostgreSQL LIKE expression."""
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _line_number(value: object) -> int:
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def persisted_fingerprint(finding: dict[str, Any]) -> str | None:
    """The finding's agent fingerprint when well-formed, else ``None`` (never invented)."""
    value = finding.get("fingerprint")
    return value if isinstance(value, str) and _FINGERPRINT_RE.match(value) else None


def persisted_severity(finding: dict[str, Any]) -> str:
    """The severity a finding row may carry: ``unrated`` unless it was COMPUTED.

    The bug hunter has nothing to rate a finding from (``ParsedError`` carries no
    severity), so its findings are ``unrated``. A severity value is persisted only
    when the finding also carries a non-empty ``severity_basis`` naming how it was
    computed. Anything else, including an absent value or a bare "medium", is
    stored as ``unrated``, so a reader can never mistake a constant for a rating.
    Before coding-analysis#112 this read ``finding.get("severity") or "medium"``,
    which turned absence into the most plausible-looking value.
    """
    basis = finding.get("severity_basis")
    severity = finding.get("severity")
    if isinstance(basis, str) and basis.strip() and isinstance(severity, str) and severity.strip():
        return severity.strip()
    return UNRATED_SEVERITY


def build_finding_rows(
    report: BugHuntingResponse,
    *,
    codebase_path: str,
) -> list[dict[str, Any]]:
    """Convert a full in-memory report into normalized per-bug rows."""
    fixes_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fix in report.resolves:
        fix_dict = dict(fix)
        fix_path = _normalize_finding_path(
            _optional_string(fix_dict.get("file_path")), codebase_path
        )
        fix_dict["file_path"] = fix_path
        fixes_by_path[fix_path].append(fix_dict)

    rows: list[dict[str, Any]] = []
    for error in report.errors:
        finding = dict(error)
        file_path = _normalize_finding_path(
            _optional_string(finding.get("file_path")), codebase_path
        )
        finding["file_path"] = file_path
        rows.append(
            {
                "file_path": file_path,
                "file_name": PurePosixPath(file_path).name,
                "line_number": _line_number(finding.get("line_number")),
                "severity": persisted_severity(finding),
                "fingerprint": persisted_fingerprint(finding),
                "finding": finding,
                "suggested_fixes": fixes_by_path.get(file_path, []),
            }
        )
    return rows


class BugHuntReportStore:
    """Tenant-scoped reader and writer for Bug Hunter jobs and findings."""

    @staticmethod
    def _hunt_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(_HUNT_FIELDS, row, strict=True))

    async def create_hunt(
        self,
        *,
        hunt_id: str,
        container_name: str,
        tenant_id: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a queued hunt before returning control to the caller."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO bug_hunts
                    (id, code_graph_id, tenant_id, status, request_payload)
                VALUES (%s, %s, %s, 'queued', %s)
                RETURNING id, code_graph_id, tenant_id::text, status, errors_found,
                          fixes_generated, root_cause_analysis, fix_summary,
                          estimated_effort, risk_assessment, analysis_time,
                          error_message, created_at, started_at, completed_at,
                          updated_at, scanned_files, tracked_source_files, commit_sha,
                          excluded_dirs, worktree_dirty, agent_sha, excluded_entry_kinds
                """,
                (hunt_id, container_name, tenant_id, Jsonb(request_payload)),
            )
            row = await cursor.fetchone()
            await conn.commit()
        if row is None:
            raise RuntimeError("Bug hunt was not created")
        return self._hunt_from_row(row)

    async def mark_running(self, *, hunt_id: str, container_name: str, tenant_id: str) -> None:
        """Transition a queued hunt to running."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE bug_hunts
                   SET status = 'running', started_at = now(), updated_at = now(),
                       error_message = NULL
                 WHERE id = %s AND code_graph_id = %s AND tenant_id = %s
                   AND status = 'queued'
                """,
                (hunt_id, container_name, tenant_id),
            )
            await conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError(f"Queued bug hunt '{hunt_id}' was not found")

    async def claim_next_hunt(
        self, *, worker_id: str, lease_seconds: int = 90
    ) -> dict[str, Any] | None:
        """Atomically claim the oldest queued job or an expired worker lease."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                WITH candidate AS (
                    SELECT id
                      FROM bug_hunts
                     WHERE status = 'queued'
                        OR (status = 'running' AND lease_expires_at < now())
                     ORDER BY created_at
                     FOR UPDATE SKIP LOCKED
                     LIMIT 1
                )
                UPDATE bug_hunts AS hunt
                   SET status = 'running', worker_id = %s,
                       started_at = COALESCE(hunt.started_at, now()),
                       lease_expires_at = now() + (%s * interval '1 second'),
                       updated_at = now(), error_message = NULL
                  FROM candidate
                 WHERE hunt.id = candidate.id
                RETURNING hunt.id, hunt.code_graph_id, hunt.tenant_id::text, hunt.status,
                          hunt.errors_found, hunt.fixes_generated,
                          hunt.root_cause_analysis, hunt.fix_summary,
                          hunt.estimated_effort, hunt.risk_assessment,
                          hunt.analysis_time, hunt.error_message, hunt.created_at,
                          hunt.started_at, hunt.completed_at, hunt.updated_at,
                          hunt.scanned_files, hunt.tracked_source_files, hunt.commit_sha,
                          hunt.excluded_dirs, hunt.worktree_dirty, hunt.agent_sha,
                          hunt.excluded_entry_kinds,
                          hunt.request_payload, hunt.worker_id
                """,
                (worker_id, lease_seconds),
            )
            row = await cursor.fetchone()
            await conn.commit()
        return dict(zip(_CLAIM_FIELDS, row, strict=True)) if row else None

    async def renew_lease(
        self,
        *,
        hunt_id: str,
        worker_id: str,
        lease_seconds: int = 90,
    ) -> bool:
        """Extend an active worker lease; false means another worker reclaimed it."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE bug_hunts
                   SET lease_expires_at = now() + (%s * interval '1 second'),
                       updated_at = now()
                 WHERE id = %s AND worker_id = %s AND status = 'running'
                """,
                (lease_seconds, hunt_id, worker_id),
            )
            await conn.commit()
        return cursor.rowcount == 1

    async def complete_hunt(
        self,
        *,
        hunt_id: str,
        container_name: str,
        tenant_id: str,
        report: BugHuntingResponse,
        codebase_path: str,
        worker_id: str | None = None,
    ) -> None:
        """Atomically persist findings and mark a hunt completed."""
        findings = build_finding_rows(report, codebase_path=codebase_path)
        async with get_raw_connection() as conn:
            if findings:
                async with conn.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO bug_hunt_findings
                            (hunt_id, code_graph_id, tenant_id, file_path, file_name,
                             line_number, severity, fingerprint, finding, suggested_fixes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                hunt_id,
                                container_name,
                                tenant_id,
                                item["file_path"],
                                item["file_name"],
                                item["line_number"],
                                item["severity"],
                                item["fingerprint"],
                                Jsonb(item["finding"]),
                                Jsonb(item["suggested_fixes"]),
                            )
                            for item in findings
                        ],
                    )

            update_params: tuple[Any, ...] = (
                report.errors_found,
                report.fixes_generated,
                report.root_cause_analysis,
                report.fix_summary,
                report.estimated_effort,
                report.risk_assessment,
                report.analysis_time,
                report.scanned_files,
                report.tracked_source_files,
                report.commit_sha,
                report.excluded_dirs,
                report.worktree_dirty,
                report.agent_sha,
                report.excluded_entry_kinds,
                hunt_id,
                container_name,
                tenant_id,
                worker_id,
                worker_id,
            )
            cursor = await conn.execute(
                """
                UPDATE bug_hunts
                   SET status = 'completed', errors_found = %s, fixes_generated = %s,
                       root_cause_analysis = %s, fix_summary = %s,
                       estimated_effort = %s, risk_assessment = %s,
                       analysis_time = %s, scanned_files = %s,
                       tracked_source_files = %s, commit_sha = %s,
                       excluded_dirs = %s, worktree_dirty = %s, agent_sha = %s,
                       excluded_entry_kinds = %s,
                       completed_at = now(), updated_at = now(),
                       lease_expires_at = NULL
                 WHERE id = %s AND code_graph_id = %s AND tenant_id = %s
                   AND status = 'running'
                   AND (%s::text IS NULL OR worker_id = %s)
                """,
                update_params,
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Running bug hunt '{hunt_id}' was not found")
            await conn.commit()

    async def fail_hunt(
        self,
        *,
        hunt_id: str,
        container_name: str,
        tenant_id: str,
        error_message: str,
        worker_id: str | None = None,
    ) -> bool:
        """Mark a queued or running hunt failed without exposing another tenant's job.

        Returns True only when THIS call moved the row to failed (False when the
        hunt was already terminal or is leased to another worker), so a caller
        can fire its one-shot completion callback exactly once.
        """
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE bug_hunts
                   SET status = 'failed', error_message = %s,
                       completed_at = now(), updated_at = now(), lease_expires_at = NULL
                 WHERE id = %s AND code_graph_id = %s AND tenant_id = %s
                   AND status IN ('queued', 'running')
                   AND (%s::text IS NULL OR worker_id = %s)
                """,
                (
                    error_message,
                    hunt_id,
                    container_name,
                    tenant_id,
                    worker_id,
                    worker_id,
                ),
            )
            await conn.commit()
            return cursor.rowcount == 1

    async def get_hunt(
        self, *, hunt_id: str, container_name: str, tenant_id: str
    ) -> dict[str, Any] | None:
        """Return one tenant-owned hunt, or ``None`` when it is not visible."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                _HUNT_SELECT + " WHERE id = %s AND code_graph_id = %s AND tenant_id = %s",
                (hunt_id, container_name, tenant_id),
            )
            row = await cursor.fetchone()
        return self._hunt_from_row(row) if row else None

    async def get_file_summary(
        self,
        *,
        hunt_id: str,
        container_name: str,
        tenant_id: str,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return paginated flagged-file summaries for a completed hunt."""
        async with get_raw_connection() as conn:
            count_cursor = await conn.execute(
                """
                SELECT count(DISTINCT file_path)
                  FROM bug_hunt_findings
                 WHERE hunt_id = %s AND code_graph_id = %s AND tenant_id = %s
                """,
                (hunt_id, container_name, tenant_id),
            )
            count_row = await count_cursor.fetchone()
            total = int(count_row[0]) if count_row else 0

            cursor = await conn.execute(
                """
                SELECT file_path, file_name, count(*) AS bug_count,
                       array_agg(severity ORDER BY severity) AS severities
                  FROM bug_hunt_findings
                 WHERE hunt_id = %s AND code_graph_id = %s AND tenant_id = %s
                 GROUP BY file_path, file_name
                 ORDER BY file_path
                 LIMIT %s OFFSET %s
                """,
                (hunt_id, container_name, tenant_id, limit, offset),
            )
            rows = await cursor.fetchall()

        return total, [
            {
                "file_path": row[0],
                "file_name": row[1],
                "bug_count": int(row[2]),
                "severity_counts": dict(Counter(row[3] or [])),
            }
            for row in rows
        ]

    async def get_findings_for_path(
        self,
        *,
        hunt_id: str,
        container_name: str,
        tenant_id: str,
        path: str,
        limit: int,
        offset: int,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Return bugs for one file or directory, recursively by default."""
        path_pattern = f"{_escape_like(path)}/%" if path else ""
        params = (hunt_id, container_name, tenant_id, path, path, path_pattern)

        async with get_raw_connection() as conn:
            count_cursor = await conn.execute(
                """
                SELECT count(*), count(DISTINCT file_path)
                  FROM bug_hunt_findings
                 WHERE hunt_id = %s AND code_graph_id = %s AND tenant_id = %s
                   AND (
                       %s::text = '' OR file_path = %s
                       OR file_path LIKE %s ESCAPE '!'
                   )
                """,
                params,
            )
            count_row = await count_cursor.fetchone()
            total_findings = int(count_row[0]) if count_row else 0
            matched_files = int(count_row[1]) if count_row else 0

            cursor = await conn.execute(
                """
                SELECT id, file_path, file_name, line_number, severity,
                       finding, suggested_fixes, fingerprint
                  FROM bug_hunt_findings
                 WHERE hunt_id = %s AND code_graph_id = %s AND tenant_id = %s
                   AND (
                       %s::text = '' OR file_path = %s
                       OR file_path LIKE %s ESCAPE '!'
                   )
                 ORDER BY file_path, line_number, id
                 LIMIT %s OFFSET %s
                """,
                (*params, limit, offset),
            )
            rows = await cursor.fetchall()

        findings = [_finding_from_row(row) for row in rows]
        return total_findings, matched_files, findings

    async def get_finding(
        self, *, hunt_id: str, container_name: str, tenant_id: str, finding_id: int
    ) -> dict[str, Any] | None:
        """One finding of one tenant-owned hunt, or ``None`` when it is not visible."""
        async with get_raw_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, file_path, file_name, line_number, severity,
                       finding, suggested_fixes, fingerprint
                  FROM bug_hunt_findings
                 WHERE id = %s AND hunt_id = %s AND code_graph_id = %s AND tenant_id = %s
                """,
                (finding_id, hunt_id, container_name, tenant_id),
            )
            row = await cursor.fetchone()
        return _finding_from_row(row) if row else None


def _finding_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "finding_id": row[0],
        "file_path": row[1],
        "file_name": row[2],
        "line_number": row[3],
        "severity": row[4],
        "bug": row[5],
        "suggested_fixes": row[6] or [],
        "fingerprint": row[7],
    }
