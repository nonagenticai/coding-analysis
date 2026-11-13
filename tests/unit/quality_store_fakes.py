"""In-memory stand-ins for ``services.quality_result_store`` (coding-analysis#106).

``FakeQualityResultStore.save`` builds its rows with the REAL ``build_row``, so a
route test that saves a real ``QualityReport`` and then reads it back exercises
the same write-side shaping (status derivation, sentinel-score nulling, issue cap,
full-list aggregates) that production stores — only the SQL is faked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from models.architecture_models import ArchitectureQualityIssue, QualityReport
from services.quality_result_store import build_row

_T0 = datetime(2026, 9, 1, tzinfo=UTC)


def make_issue(
    severity: str = "low",
    file_path: str = "test.py",
    line: int = 1,
    msg: str = "test issue",
    *,
    tool: str = "test-tool",
    rule_id: str = "R001",
    column: int | None = 1,
) -> ArchitectureQualityIssue:
    return ArchitectureQualityIssue(
        id=f"{tool}:{file_path}:{line}",
        severity=severity,
        message=msg,
        rule_id=rule_id,
        rule_name=rule_id,
        file_path=file_path,
        line_number=line,
        column_number=column,
        tool=tool,
        category="style",
    )


def make_report(
    score: float = 85.0,
    issues: list[ArchitectureQualityIssue] | None = None,
    *,
    inconclusive: bool = False,
    partial: bool = False,
    tools_attempted: list[str] | None = None,
    tools_succeeded: list[str] | None = None,
    tools_failed: list[str] | None = None,
    grade: str = "B",
    language: str = "python",
) -> QualityReport:
    attempted = tools_attempted if tools_attempted is not None else ["ruff"]
    succeeded = tools_succeeded if tools_succeeded is not None else list(attempted)
    failed = tools_failed if tools_failed is not None else []
    return QualityReport(
        job_id="job-1",
        codebase_id="c1",
        generated_at="2026-09-01T00:00:00+00:00",
        overall_score=-1.0 if inconclusive else score,
        grade="N/A" if inconclusive else grade,
        language=language,
        inconclusive=inconclusive,
        partial=partial,
        tools_attempted=attempted,
        tools_succeeded=succeeded,
        tools_failed=failed,
        issues=issues or [],
        summary={"total_issues": len(issues or [])},
        recommendations=["rec1"],
        execution_time=1.5,
    )


class FakeQualityResultStore:
    """Tenant-scoped in-memory ``QualityResultStore``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.save_calls: list[dict[str, Any]] = []
        self.fail_saves: Exception | None = None
        self.latest_for_sha_calls: list[tuple[str, str, str]] = []

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
        self.save_calls.append(
            {
                "report": report,
                "tenant_id": tenant_id,
                "container_name": container_name,
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "error_message": error_message,
            }
        )
        if self.fail_saves is not None:
            raise self.fail_saves
        row = build_row(
            report,
            tenant_id=tenant_id,
            container_name=container_name,
            repo_url=repo_url,
            commit_sha=commit_sha,
            error_message=error_message,
        )
        row["created_at"] = _T0 + timedelta(minutes=len(self.rows))
        self.rows.append(row)
        return row["id"]

    def _visible(self, tenant_id: str, container_name: str) -> list[dict[str, Any]]:
        mine = [
            r
            for r in self.rows
            if r["tenant_id"] == tenant_id and r["container_name"] == container_name
        ]
        return sorted(mine, key=lambda r: r["created_at"], reverse=True)

    async def latest(self, tenant_id: str, container_name: str) -> dict[str, Any] | None:
        rows = self._visible(tenant_id, container_name)
        return rows[0] if rows else None

    async def latest_for_sha(
        self, tenant_id: str, container_name: str, commit_sha: str
    ) -> dict[str, Any] | None:
        self.latest_for_sha_calls.append((tenant_id, container_name, commit_sha))
        rows = [
            r for r in self._visible(tenant_id, container_name) if r["commit_sha"] == commit_sha
        ]
        return rows[0] if rows else None

    async def history(
        self,
        tenant_id: str,
        container_name: str,
        limit: int = 50,
        *,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        # ``since`` is ignored: fake rows are dated in the past relative to the
        # route's ``now() - days`` window; the SQL filter is covered separately.
        return self._visible(tenant_id, container_name)[:limit]
