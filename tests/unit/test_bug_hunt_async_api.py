"""Regression tests for the asynchronous Bug Hunter report API."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Response

from app.api.endpoints.bug_hunter import (
    _run_claimed_hunt,
    get_bug_hunt_findings_for_path,
    get_bug_hunt_job_status,
    get_bug_hunt_summary_report,
    queue_bug_hunt,
)
from app.api.models.bug_hunter_api_models import BugHuntJobStatus, BugHuntRequest
from models.analysis_models import BugHuntingResponse
from services.bug_hunt_report_store import (
    BugHuntReportStore,
    _escape_like,
    build_finding_rows,
    normalize_query_path,
)


def _job(status: str = "completed") -> dict:
    now = datetime.now(UTC)
    return {
        "hunt_id": "hunt-1",
        "container_name": "repo-1",
        "tenant_id": "tenant-1",
        "status": status,
        "errors_found": 2,
        "fixes_generated": 1,
        "root_cause_analysis": "Two related parser failures",
        "fix_summary": "Guard invalid parser input",
        "estimated_effort": "1 hour",
        "risk_assessment": "Low",
        "analysis_time": 4.2,
        "error_message": None,
        "created_at": now,
        "started_at": now,
        "completed_at": now if status in {"completed", "failed"} else None,
        "updated_at": now,
    }


def _store() -> MagicMock:
    store = MagicMock(spec=BugHuntReportStore)
    store.create_hunt = AsyncMock()
    store.mark_running = AsyncMock()
    store.claim_next_hunt = AsyncMock()
    store.renew_lease = AsyncMock(return_value=True)
    store.complete_hunt = AsyncMock()
    store.fail_hunt = AsyncMock()
    store.get_hunt = AsyncMock()
    store.get_file_summary = AsyncMock()
    store.get_findings_for_path = AsyncMock()
    return store


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        (".", ""),
        ("./src//api/", "src/api"),
        (r"src\api\routes.py", "src/api/routes.py"),
        (".github/workflows/ci.yml", ".github/workflows/ci.yml"),
    ],
)
def test_normalize_query_path(raw: str, expected: str):
    assert normalize_query_path(raw) == expected


@pytest.mark.parametrize("raw", ["/etc/passwd", "src/../secret.py", "../outside"])
def test_normalize_query_path_rejects_non_repository_paths(raw: str):
    with pytest.raises(ValueError):
        normalize_query_path(raw)


def test_escape_like_uses_single_character_postgres_escape():
    assert _escape_like("src/under_score/100%!/file.py") == ("src/under!_score/100!%!!/file.py")


def test_build_finding_rows_groups_fixes_by_relative_file_path():
    report = BugHuntingResponse(
        errors_found=1,
        fixes_generated=1,
        root_cause_analysis="root",
        fix_summary="fix",
        estimated_effort="small",
        risk_assessment="low",
        errors=[
            {
                "error_id": "err_0",
                "error_type": "syntax_error",
                "error_message": "bad syntax",
                "stack_trace": "",
                "file_path": "/repo/.github/workflows/ci.yml",
                "line_number": 4,
                "function_name": "",
                "error_category": "syntax_error",
                "severity": "high",
                "timestamp": "now",
                "frequency": 1,
                "impact_score": 0.9,
                "suggested_fix": "",
                "related_errors": [],
            }
        ],
        resolves=[
            {
                "fix_id": "fix_0",
                "fix_type": "code_fix",
                "file_path": "/repo/.github/workflows/ci.yml",
                "line_number": 4,
                "original_code": "bad",
                "fixed_code": "good",
                "description": "repair YAML",
                "confidence": 0.9,
                "risk_level": "low",
                "test_impact": "",
                "dependencies": [],
                "validation_steps": [],
                "rollback_plan": "revert",
            }
        ],
        pull_request=None,
        analysis_time=0.1,
    )

    rows = build_finding_rows(report, codebase_path="/repo")

    assert rows[0]["file_path"] == ".github/workflows/ci.yml"
    assert rows[0]["file_name"] == "ci.yml"
    assert rows[0]["suggested_fixes"][0]["fix_id"] == "fix_0"


async def test_queue_hunt_returns_202_contract_and_schedules_work():
    store = _store()
    store.create_hunt.return_value = _job("queued")
    http_response = Response()
    manager = MagicMock()
    manager.get_container = AsyncMock(
        return_value=SimpleNamespace(work_dir="/repo", codebase_path="/repo")
    )

    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=manager):
        response = await queue_bug_hunt(
            response=http_response,
            container_name="repo-1",
            request=BugHuntRequest(),
            tenant_id="tenant-1",
            store=store,
        )

    assert response.status == BugHuntJobStatus.QUEUED
    assert response.links["summary"].endswith(f"/{response.hunt_id}/report")
    assert http_response.headers["location"] == response.links["status"]
    assert http_response.headers["retry-after"] == "2"
    payload = store.create_hunt.await_args.kwargs["request_payload"]
    assert "GITHUB_ACCESS_TOKEN" not in payload


async def test_job_status_is_scoped_through_store_lookup():
    store = _store()
    store.get_hunt.return_value = _job("running")

    response = await get_bug_hunt_job_status(
        container_name="repo-1",
        hunt_id="hunt-1",
        tenant_id="tenant-1",
        store=store,
    )

    assert response.status == BugHuntJobStatus.RUNNING
    store.get_hunt.assert_awaited_once_with(
        hunt_id="hunt-1", container_name="repo-1", tenant_id="tenant-1"
    )


async def test_summary_lists_flagged_paths_without_full_findings():
    store = _store()
    store.get_hunt.return_value = _job()
    store.get_file_summary.return_value = (
        1,
        [
            {
                "file_path": "src/api/routes.py",
                "file_name": "routes.py",
                "bug_count": 2,
                "severity_counts": {"medium": 2},
            }
        ],
    )

    response = await get_bug_hunt_summary_report(
        container_name="repo-1",
        hunt_id="hunt-1",
        limit=50,
        offset=0,
        tenant_id="tenant-1",
        store=store,
    )

    assert response.total_flagged_files == 1
    assert response.flagged_files[0].file_path == "src/api/routes.py"
    assert not hasattr(response.flagged_files[0], "bug")


async def test_findings_path_includes_directory_subtree():
    store = _store()
    store.get_hunt.return_value = _job()
    store.get_findings_for_path.return_value = (2, 2, [])

    response = await get_bug_hunt_findings_for_path(
        container_name="repo-1",
        hunt_id="hunt-1",
        path="./src/api/",
        limit=100,
        offset=0,
        tenant_id="tenant-1",
        store=store,
    )

    assert response.path == "src/api"
    assert response.recursive is True
    store.get_findings_for_path.assert_awaited_once_with(
        hunt_id="hunt-1",
        container_name="repo-1",
        tenant_id="tenant-1",
        path="src/api",
        limit=100,
        offset=0,
    )


async def test_report_is_not_available_until_hunt_completes():
    store = _store()
    store.get_hunt.return_value = _job("running")

    with pytest.raises(HTTPException) as exc_info:
        await get_bug_hunt_summary_report(
            container_name="repo-1",
            hunt_id="hunt-1",
            limit=50,
            offset=0,
            tenant_id="tenant-1",
            store=store,
        )

    assert exc_info.value.status_code == 409


async def test_claimed_worker_persists_completed_report():
    store = _store()
    report = BugHuntingResponse(
        errors_found=0,
        fixes_generated=0,
        root_cause_analysis="clean",
        fix_summary="none",
        estimated_effort="none",
        risk_assessment="low",
        errors=[],
        resolves=[],
        pull_request=None,
        analysis_time=0.2,
    )

    manager = MagicMock()
    manager.get_container = AsyncMock(return_value=SimpleNamespace(work_dir="/repo"))
    job = _job("running")
    job["request_payload"] = BugHuntRequest().model_dump(mode="json")

    with (
        patch("app.api.endpoints.bug_hunter.hunt_bugs", new=AsyncMock(return_value=report)),
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=manager),
    ):
        await _run_claimed_hunt(job=job, worker_id="worker-1", store=store)

    store.complete_hunt.assert_awaited_once()
    store.fail_hunt.assert_not_awaited()


@pytest.mark.parametrize(
    "stored",
    [["non_utf8_path", "submodule", "symlink"], None],
    ids=["measured", "unmeasured"],
)
async def test_summary_and_findings_expose_excluded_entry_kinds_null_preserved(stored):
    # fr's bug_report/v1 requires the hunt's excluded_entry_kinds; an unmeasured
    # hunt must reach it as null (same semantics as excluded_dirs), never [].
    store = _store()
    store.get_hunt.return_value = {
        **_job(),
        "excluded_dirs": ["vendor"] if stored else None,
        "excluded_entry_kinds": stored,
    }
    store.get_file_summary.return_value = (0, [])
    store.get_findings_for_path.return_value = (0, 0, [])

    summary = await get_bug_hunt_summary_report(
        container_name="repo-1",
        hunt_id="hunt-1",
        limit=50,
        offset=0,
        tenant_id="tenant-1",
        store=store,
    )
    findings = await get_bug_hunt_findings_for_path(
        container_name="repo-1",
        hunt_id="hunt-1",
        path="",
        limit=50,
        offset=0,
        tenant_id="tenant-1",
        store=store,
    )

    assert summary.excluded_entry_kinds == stored
    assert summary.model_dump(mode="json")["excluded_entry_kinds"] == stored
    assert findings.excluded_entry_kinds == stored
    assert findings.excluded_dirs == (["vendor"] if stored else None)
