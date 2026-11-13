"""Tests for the quality endpoint functions.

Covers: get_supported_extensions, get_language_tools, trigger_quality_analysis
(including the persisted background task), and the STORED-result read path
(coding-analysis#106): get_latest_quality_results, get_quality_issues,
get_quality_summary, evaluate_quality_gate, get_quality_trend,
get_quality_history. Every read test asserts the analyzer is never awaited.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.quality import (
    evaluate_quality_gate,
    get_language_tools,
    get_latest_quality_results,
    get_quality_history,
    get_quality_issues,
    get_quality_summary,
    get_quality_trend,
    get_supported_extensions,
    trigger_quality_analysis,
)
from tests.unit.quality_store_fakes import FakeQualityResultStore, make_issue, make_report

_VALIDATE_CONTAINER = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_GET_QUALITY_ANALYZER = "app.api.endpoints.quality.get_quality_analyzer"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"
_GET_REDIS = "app.api.endpoints.quality.get_redis_service"
_SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
_OLD_SHA = "fedcba9876543210fedcba9876543210fedcba98"  # pragma: allowlist secret
TENANT = "0195f48b-0000-7000-8000-00000000000a"
OTHER_TENANT = "0195f48b-0000-7000-8000-00000000000b"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_mock_container(name="test-container", path="/app/data/repo"):
    mock = MagicMock()
    mock.container_name = name
    mock.codebase_path = path
    mock.status = "running"
    return mock


def _make_mock_result(score=85.0, issues=None):
    """Create a mock analysis result."""
    result = MagicMock()
    result.overall_score = score
    result.issues = issues or []
    result.summary = "Test summary"
    result.recommendations = ["rec1"]
    result.execution_time = 1.5
    result.generated_at = "2026-01-01T00:00:00"
    result.job_id = "job-1"
    result.grade = "B"
    # Mirror the real QualityReport: a conclusive run is not inconclusive. A bare
    # MagicMock would auto-create a truthy .inconclusive and trip the abstain path.
    result.inconclusive = False
    # Same trap for the provenance fields: a bare MagicMock is truthy/non-iterable
    # and would be rejected by the typed QualityGateResult(list[str]) construction
    # in /gate. A conclusive run: all attempted tools succeeded, not partial.
    result.partial = False
    result.tools_attempted = ["ruff"]
    result.tools_succeeded = ["ruff"]
    result.tools_failed = []
    return result


def _make_mock_issue(severity="low", file_path="test.py", line=1, msg="test issue"):
    issue = MagicMock()
    issue.severity = severity
    issue.message = msg
    issue.file_path = file_path
    issue.line_number = line
    issue.column_number = 1
    issue.tool = "test-tool"
    issue.rule_id = "R001"
    issue.suggested_fix = None
    issue.fix_confidence = None
    return issue


# ---------------------------------------------------------------------------
# get_supported_extensions
# ---------------------------------------------------------------------------


class TestGetSupportedExtensions:
    @pytest.mark.asyncio
    async def test_returns_extensions(self):
        result = await get_supported_extensions()
        assert "extensions" in result
        assert ".py" in result["extensions"]
        assert ".js" in result["extensions"]

    @pytest.mark.asyncio
    async def test_extensions_are_unique(self):
        result = await get_supported_extensions()
        exts = result["extensions"]
        assert len(exts) == len(set(exts))

    @pytest.mark.asyncio
    async def test_extensions_are_sorted(self):
        result = await get_supported_extensions()
        exts = result["extensions"]
        assert exts == sorted(exts)


# ---------------------------------------------------------------------------
# get_language_tools
# ---------------------------------------------------------------------------


class TestGetLanguageTools:
    @pytest.mark.asyncio
    async def test_python_tools(self):
        result = await get_language_tools("python")
        assert result["language"] == "python"
        assert len(result["tools"]) >= 4
        tool_names = [t["name"] for t in result["tools"]]
        # Only what the analysis runs (py2rust #13 + astral runners #131).
        assert "ruff" in tool_names
        assert "ty" in tool_names
        assert "flake8" not in tool_names and "mypy" not in tool_names

    @pytest.mark.asyncio
    async def test_javascript_tools(self):
        result = await get_language_tools("javascript")
        assert result["language"] == "javascript"
        tool_names = [t["name"] for t in result["tools"]]
        assert "eslint" in tool_names

    @pytest.mark.asyncio
    async def test_java_tools(self):
        result = await get_language_tools("java")
        assert result["language"] == "java"
        tool_names = [t["name"] for t in result["tools"]]
        assert "checkstyle" in tool_names

    @pytest.mark.asyncio
    async def test_unsupported_language_raises_404(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_language_tools("brainfuck")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        result = await get_language_tools("PYTHON")
        assert result["language"] == "python"  # canonical name
        assert len(result["tools"]) >= 4

    @pytest.mark.asyncio
    async def test_each_tool_has_category_and_description(self):
        result = await get_language_tools("python")
        for tool in result["tools"]:
            assert "category" in tool
            assert "description" in tool
            assert "name" in tool


# ---------------------------------------------------------------------------
# trigger_quality_analysis
# ---------------------------------------------------------------------------


class TestTriggerQualityAnalysis:
    @pytest.mark.asyncio
    @patch(_GET_QUALITY_ANALYZER)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_success(self, mock_validate, mock_get_analyzer):
        mock_validate.return_value = _make_mock_container()
        mock_analyzer = MagicMock()
        mock_get_analyzer.return_value = mock_analyzer

        bg_tasks = MagicMock()

        result = await trigger_quality_analysis(
            container_name="test",
            background_tasks=bg_tasks,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["status"] == "queued"
        assert "task_id" in result
        assert result["container_name"] == "test"

    @pytest.mark.asyncio
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_container_not_found(self, mock_validate):
        mock_validate.side_effect = HTTPException(status_code=404, detail="not found")
        bg_tasks = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await trigger_quality_analysis(
                container_name="missing",
                background_tasks=bg_tasks,
                analyzer=MagicMock(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_runtime_error_raises_500(self, mock_validate):
        mock_validate.side_effect = RuntimeError("boom")
        bg_tasks = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await trigger_quality_analysis(
                container_name="bad",
                background_tasks=bg_tasks,
                analyzer=MagicMock(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch(_GET_REDIS)
    @patch(_RESOLVE_SHA, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_background_task_persists_the_run(self, mock_validate, mock_sha, mock_redis):
        """POST /analyze stores the run (with its commit) and still writes Redis."""
        container = _make_mock_container()
        container.repo_url = (
            "https://x-access-token:SECRET@github.com/org/repo.git"  # pragma: allowlist secret
        )
        mock_validate.return_value = container
        mock_sha.return_value = _SHA
        redis = MagicMock()
        redis.set = AsyncMock()
        mock_redis.return_value = redis
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=91.0))
        store = FakeQualityResultStore()
        bg_tasks = MagicMock()

        queued = await trigger_quality_analysis(
            container_name="c1",
            background_tasks=bg_tasks,
            analyzer=analyzer,
            tenant_id=TENANT,
            store=store,
        )
        task = bg_tasks.add_task.call_args.args[0]
        await task()

        assert len(store.rows) == 1
        row = store.rows[0]
        assert row["status"] == "complete"
        assert row["commit_sha"] == _SHA
        assert row["tenant_id"] == TENANT
        assert row["repo_url"] == "https://github.com/org/repo.git"  # token stripped
        import json

        stored_task = json.loads(redis.set.await_args.args[1])
        assert stored_task["task_id"] == queued["task_id"]
        assert stored_task["result_id"] == row["id"]
        assert stored_task["commit_sha"] == _SHA
        assert stored_task["persisted"] is True

    @pytest.mark.asyncio
    @patch(_GET_REDIS)
    @patch(_RESOLVE_SHA, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_background_task_survives_a_store_failure(
        self, mock_validate, mock_sha, mock_redis
    ):
        mock_validate.return_value = _make_mock_container()
        mock_sha.return_value = None
        redis = MagicMock()
        redis.set = AsyncMock()
        mock_redis.return_value = redis
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        store = FakeQualityResultStore()
        store.fail_saves = RuntimeError("db down")
        bg_tasks = MagicMock()

        await trigger_quality_analysis(
            container_name="c1",
            background_tasks=bg_tasks,
            analyzer=analyzer,
            tenant_id=TENANT,
            store=store,
        )
        await bg_tasks.add_task.call_args.args[0]()

        import json

        stored_task = json.loads(redis.set.await_args.args[1])
        assert stored_task["status"] == "completed"
        assert stored_task["persisted"] is False
        assert stored_task["result_id"] is None

    @pytest.mark.asyncio
    @patch(_GET_REDIS)
    @patch(_RESOLVE_SHA, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_background_no_report_stores_a_failed_row(
        self, mock_validate, mock_sha, mock_redis
    ):
        mock_validate.return_value = _make_mock_container()
        mock_sha.return_value = _SHA
        mock_redis.return_value = MagicMock(set=AsyncMock())
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=None)
        store = FakeQualityResultStore()
        bg_tasks = MagicMock()

        await trigger_quality_analysis(
            container_name="c1",
            background_tasks=bg_tasks,
            analyzer=analyzer,
            tenant_id=TENANT,
            store=store,
        )
        await bg_tasks.add_task.call_args.args[0]()

        assert [r["status"] for r in store.rows] == ["failed"]
        assert store.rows[0]["overall_score"] is None
        assert store.rows[0]["grade"] is None


# ---------------------------------------------------------------------------
# Stored-result read path — no GET may run the analyzer (#106)
# ---------------------------------------------------------------------------


def _analyzer() -> MagicMock:
    """An analyzer whose analyze() every read test asserts was never awaited."""
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(side_effect=AssertionError("a GET ran the analyzer"))
    return analyzer


async def _store_with(*reports, tenant=TENANT, container="c1", sha=_SHA):
    store = FakeQualityResultStore()
    for report in reports:
        await store.save(
            report, tenant_id=tenant, container_name=container, repo_url=None, commit_sha=sha
        )
    return store


class TestReadRoutesNeverAnalyze:
    """The routes no longer even accept an analyzer: a read cannot run one."""

    @pytest.mark.parametrize(
        "fn",
        [
            get_latest_quality_results,
            get_quality_issues,
            get_quality_summary,
            get_quality_trend,
            get_quality_history,
        ],
    )
    def test_read_route_has_no_analyzer_dependency(self, fn):
        import inspect

        assert "analyzer" not in inspect.signature(fn).parameters

    @pytest.mark.parametrize(
        "path",
        [
            "/quality/c1/latest",
            "/quality/c1/issues",
            "/quality/c1/summary",
            "/quality/c1/trend",
            "/quality/c1/history",
        ],
    )
    def test_http_get_does_not_await_analyze(self, path):
        """Even with the analyzer dependency overridden, no read GET awaits it.

        ``/gate`` is deliberately absent: it grades the CURRENT commit and so
        analyzes on a miss (``TestGateGradesTheCurrentCommit``)."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from services.quality_result_store import get_quality_result_store

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock()
        store = FakeQualityResultStore()
        app = FastAPI()
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: TENANT
        app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        with patch(_VALIDATE_CONTAINER, new_callable=AsyncMock) as mock_validate:
            r = TestClient(app).get(path)
            mock_validate.assert_not_awaited()
        assert r.status_code == 200, r.text
        analyzer.analyze.assert_not_awaited()


class TestGetLatestQualityResults:
    @pytest.mark.asyncio
    async def test_stored_row_is_rendered(self):
        issue = make_issue("high", "main.py", 10, "bad code")
        store = await _store_with(make_report(score=80.0, issues=[issue]))

        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["status"] == "complete"
        assert result["commit_sha"] == _SHA
        assert result["result_id"] == store.rows[0]["id"]
        assert result["overall_score"] == 80.0
        assert result["grade"] == "B"
        assert result["issues_count"] == 1
        assert result["issues"][0] == {
            "tool": "test-tool",
            "severity": "high",
            "message": "bad code",
            "file_path": "main.py",
            "line_number": 10,
            "column_number": 1,
            "rule_id": "R001",
            "repo": None,
        }
        assert result["inconclusive"] is False

    @pytest.mark.asyncio
    async def test_nothing_stored_is_not_analyzed(self):
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=FakeQualityResultStore()
        )
        assert result["status"] == "not_analyzed"
        assert result["overall_score"] is None
        assert result["grade"] is None
        assert result["issues"] == []
        assert result["result_id"] is None

    @pytest.mark.asyncio
    async def test_newest_row_wins(self):
        store = await _store_with(make_report(score=50.0), make_report(score=95.0))
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["overall_score"] == 95.0

    @pytest.mark.asyncio
    async def test_another_tenants_row_is_invisible(self):
        store = await _store_with(make_report(score=95.0), tenant=OTHER_TENANT)
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["status"] == "not_analyzed"

    @pytest.mark.asyncio
    async def test_inconclusive_row_abstains(self):
        store = await _store_with(
            make_report(inconclusive=True, tools_succeeded=[], tools_failed=["ruff"])
        )
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["status"] == "inconclusive"
        assert result["overall_score"] is None  # the -1.0 sentinel is not a score
        assert result["grade"] == "N/A"  # honest abstain, never an F
        assert result["inconclusive"] is True

    @pytest.mark.asyncio
    async def test_partial_row_has_no_letter_grade(self):
        store = await _store_with(
            make_report(
                score=97.0,
                partial=True,
                tools_attempted=["ruff", "ty"],
                tools_succeeded=["ruff"],
                tools_failed=["ty"],
                grade="A",
            )
        )
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["status"] == "partial"
        assert result["partial"] is True
        assert result["grade"] is None
        assert result["overall_score"] == 97.0
        assert result["tools_failed"] == ["ty"]

    @pytest.mark.asyncio
    async def test_failed_row_is_not_a_score(self):
        store = await _store_with(None)
        result = await get_latest_quality_results(
            container_name="c1", tenant_id=TENANT, store=store
        )
        assert result["status"] == "failed"
        assert result["overall_score"] is None
        assert result["grade"] is None
        assert result["inconclusive"] is True

    @pytest.mark.asyncio
    async def test_store_error_is_500_not_not_analyzed(self):
        store = FakeQualityResultStore()
        store.latest = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(HTTPException) as exc_info:
            await get_latest_quality_results(container_name="c1", tenant_id=TENANT, store=store)
        assert exc_info.value.status_code == 500


class TestGetQualityIssues:
    async def _issues(self, store, **kw):
        params = {"severity": None, "file_path": None, "limit": 100}
        params.update(kw)
        return await get_quality_issues(
            container_name="c1", tenant_id=TENANT, store=store, **params
        )

    @pytest.mark.asyncio
    async def test_nothing_stored_is_not_analyzed_not_clean(self):
        result = await self._issues(FakeQualityResultStore())
        assert result["status"] == "not_analyzed"
        assert result["total"] == 0
        assert result["inconclusive"] is True

    @pytest.mark.asyncio
    async def test_issues_carry_sync_fields(self):
        issue = make_issue(
            "warning", "src/a.py", 7, "unused import", tool="ruff", rule_id="F401", column=3
        )
        result = await self._issues(await _store_with(make_report(issues=[issue])))
        assert result["issues"] == [
            {
                "tool": "ruff",
                "severity": "warning",
                "message": "unused import",
                "file_path": "src/a.py",
                "line_number": 7,
                "column_number": 3,
                "rule_id": "F401",
                "repo": None,
            }
        ]
        assert result["inconclusive"] is False
        assert result["partial"] is False
        assert result["tools_attempted"] == ["ruff"]
        assert result["tools_failed"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wanted", ["warning", "WARNING", "error", "info", "medium"])
    async def test_filters_by_tool_emitted_severity(self, wanted):
        issues = [
            make_issue("warning"),
            make_issue("warning"),
            make_issue("error"),
            make_issue("info"),
            make_issue("medium"),
            make_issue("high"),
        ]
        result = await self._issues(await _store_with(make_report(issues=issues)), severity=wanted)
        expected = {"warning": 2, "error": 1, "info": 1, "medium": 1}[wanted.lower()]
        assert result["total"] == expected
        assert all(i["severity"] == wanted.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_unknown_severity_is_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await self._issues(FakeQualityResultStore(), severity="bogus")
        assert exc_info.value.status_code == 400

    def test_severity_constant_covers_tool_outputs(self):
        from app.api.endpoints.quality import QUALITY_ISSUE_SEVERITIES

        for sev in ("error", "warning", "info", "medium", "low", "moderate", "high", "critical"):
            assert sev in QUALITY_ISSUE_SEVERITIES

    @pytest.mark.asyncio
    async def test_filters_by_file_path(self):
        issues = [
            make_issue(file_path="src/main.py"),
            make_issue(file_path="src/utils.py"),
            make_issue(file_path="tests/test_main.py"),
        ]
        result = await self._issues(
            await _store_with(make_report(issues=issues)), file_path="main.py"
        )
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        issues = [make_issue(line=i) for i in range(20)]
        result = await self._issues(await _store_with(make_report(issues=issues)), limit=5)
        assert result["returned"] == 5
        assert result["total"] == 20

    @pytest.mark.asyncio
    async def test_truncation_is_reported(self):
        from services.quality_result_store import MAX_STORED_ISSUES

        issues = [make_issue(line=i) for i in range(MAX_STORED_ISSUES + 5)]
        result = await self._issues(await _store_with(make_report(issues=issues)), limit=10)
        assert result["issues_total"] == MAX_STORED_ISSUES + 5
        assert result["issues_truncated"] is True
        assert result["total"] == MAX_STORED_ISSUES


class TestGetQualitySummary:
    @pytest.mark.asyncio
    async def test_nothing_stored_is_not_analyzed(self):
        result = await get_quality_summary(
            container_name="c1",
            top_files_limit=20,
            tenant_id=TENANT,
            store=FakeQualityResultStore(),
        )
        assert result["status"] == "not_analyzed"
        assert result["overall_score"] is None
        assert result["grade"] is None
        assert result["total_issues"] == 0

    @pytest.mark.asyncio
    async def test_summary_with_issues(self):
        issues = [
            make_issue("critical", "a.py"),
            make_issue("high", "b.py"),
            make_issue("moderate", "a.py"),
            make_issue("low", "c.py"),
        ]
        store = await _store_with(make_report(score=70.0, issues=issues))
        result = await get_quality_summary(
            container_name="c1", top_files_limit=20, tenant_id=TENANT, store=store
        )
        assert result["overall_score"] == 70.0
        assert result["grade"] == "C"
        assert result["total_issues"] == 4
        assert result["severity_distribution"]["critical"] == 1
        assert result["severity_distribution"]["high"] == 1
        assert result["top_problematic_files"][0] == {"file": "a.py", "issue_count": 2}
        assert result["total_files_with_issues"] == 3

    @pytest.mark.asyncio
    async def test_counts_survive_issue_truncation(self):
        from services.quality_result_store import MAX_STORED_ISSUES

        issues = [
            make_issue("critical", f"f{i % 3}.py", line=i) for i in range(MAX_STORED_ISSUES + 7)
        ]
        store = await _store_with(make_report(score=10.0, issues=issues))
        result = await get_quality_summary(
            container_name="c1", top_files_limit=20, tenant_id=TENANT, store=store
        )
        assert result["total_issues"] == MAX_STORED_ISSUES + 7
        assert result["severity_distribution"]["critical"] == MAX_STORED_ISSUES + 7

    @pytest.mark.asyncio
    async def test_top_files_limit(self):
        issues = [make_issue("low", f"file_{i}.py") for i in range(30)]
        store = await _store_with(make_report(issues=issues))
        result = await get_quality_summary(
            container_name="c1", top_files_limit=5, tenant_id=TENANT, store=store
        )
        assert result["showing_top_n_files"] == 5
        assert result["total_files_with_issues"] == 30

    @pytest.mark.asyncio
    async def test_partial_row_has_no_green_grade(self):
        store = await _store_with(
            make_report(
                score=99.0,
                partial=True,
                tools_attempted=["ruff", "ty", "bandit"],
                tools_succeeded=["ruff"],
                tools_failed=["ty", "bandit"],
                grade="A",
            )
        )
        result = await get_quality_summary(
            container_name="c1", top_files_limit=20, tenant_id=TENANT, store=store
        )
        assert result["grade"] is None
        assert result["partial"] is True


class TestEvaluateQualityGate:
    """Grading a stored row for the container's current commit (no analysis)."""

    @pytest.fixture(autouse=True)
    def _head_is_the_stored_sha(self):
        with patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA):
            yield

    async def _gate(self, store, analyzer=None, **kw):
        params = {"min_score": 80.0, "max_critical": 0, "max_high": 3, "min_coverage": None}
        params.update(kw)
        if analyzer is None:
            return await evaluate_quality_gate(
                container_name="c1", tenant_id=TENANT, store=store, analyzer=_analyzer(), **params
            )
        with patch(
            _VALIDATE_CONTAINER,
            new_callable=AsyncMock,
            return_value=_make_mock_container(path="/workspace"),
        ):
            return await evaluate_quality_gate(
                container_name="c1", tenant_id=TENANT, store=store, analyzer=analyzer, **params
            )

    @staticmethod
    def _rerun(report):
        """A stored failed/inconclusive run is never reused: the gate re-analyzes."""
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=report)
        return analyzer

    @pytest.mark.asyncio
    async def test_gate_passes(self):
        result = await self._gate(await _store_with(make_report(score=95.0)))
        assert result.passed is True
        assert result.overall_score == 95.0
        assert result.grade == "A"
        assert result.status == "complete"
        assert result.commit_sha == _SHA
        assert any("All quality gate checks passed" in r for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_gate_fails_low_score(self):
        result = await self._gate(await _store_with(make_report(score=50.0)))
        assert result.passed is False
        assert any("Improve overall score" in r for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_gate_fails_critical_issues(self):
        issues = [make_issue("critical", line=i) for i in range(3)]
        result = await self._gate(await _store_with(make_report(score=95.0, issues=issues)))
        assert result.passed is False
        assert any("critical" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_gate_fails_high_issues(self):
        issues = [make_issue("high", line=i) for i in range(5)]
        result = await self._gate(await _store_with(make_report(score=95.0, issues=issues)))
        assert result.passed is False
        assert any("reduce high" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_stored_gate_carries_commit_and_source(self):
        result = await self._gate(await _store_with(make_report(score=95.0)))
        assert result.source == "stored"
        assert result.commit_sha == _SHA

    @pytest.mark.asyncio
    async def test_failed_row_gate_cannot_pass(self):
        analyzer = self._rerun(None)
        result = await self._gate(await _store_with(None), analyzer)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"
        assert result.passed is False
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_inconclusive_gate_is_not_a_failing_f(self):
        report = make_report(inconclusive=True, tools_succeeded=[], tools_failed=["ruff"])
        store = await _store_with(report)
        result = await self._gate(store, self._rerun(report), min_score=0.0)
        assert result.passed is False
        assert result.inconclusive is True
        assert result.grade == "N/A"
        assert result.overall_score == 0.0
        assert [c.name for c in result.gate_checks] == ["analysis_conclusive"]

    @pytest.mark.asyncio
    async def test_partial_gate_cannot_pass_even_with_a_perfect_score(self):
        store = await _store_with(
            make_report(
                score=100.0,
                partial=True,
                tools_attempted=["ruff", "ty"],
                tools_succeeded=["ruff"],
                tools_failed=["ty"],
                grade="A",
            )
        )
        result = await self._gate(store)
        assert result.passed is False
        assert result.partial is True
        assert result.grade == "N/A"
        check = next(c for c in result.gate_checks if c.name == "analysis_complete")
        assert check.passed is False
        assert (check.actual, check.threshold) == (1.0, 2.0)


class TestGetQualityTrend:
    @pytest.mark.asyncio
    async def test_only_complete_rows_are_points(self):
        store = await _store_with(
            make_report(score=70.0),
            make_report(
                score=99.0,
                partial=True,
                tools_attempted=["ruff", "ty"],
                tools_succeeded=["ruff"],
                tools_failed=["ty"],
            ),
            make_report(inconclusive=True, tools_succeeded=[], tools_failed=["ruff"]),
            None,
            make_report(score=80.0),
        )
        result = await get_quality_trend(
            container_name="c1", days=30, tenant_id=TENANT, store=store
        )
        assert [p["score"] for p in result["trend"]] == [70.0, 80.0]  # oldest → newest
        assert result["skipped_points"] == 3
        assert result["average_score"] == 75.0
        assert result["update"] == 10.0
        assert result["trend"][0]["commit_sha"] == _SHA

    @pytest.mark.asyncio
    async def test_nothing_stored(self):
        result = await get_quality_trend(
            container_name="c1", days=7, tenant_id=TENANT, store=FakeQualityResultStore()
        )
        assert result["trend"] == []
        assert result["skipped_points"] == 0
        assert result["average_score"] is None
        assert result["status"] == "not_analyzed"

    @pytest.mark.asyncio
    async def test_passes_the_window_to_the_store(self):
        store = FakeQualityResultStore()
        store.history = AsyncMock(return_value=[])
        await get_quality_trend(container_name="c1", days=7, tenant_id=TENANT, store=store)
        kwargs = store.history.await_args.kwargs
        assert kwargs["since"] is not None
        assert store.history.await_args.args[:2] == (TENANT, "c1")


class TestGetQualityHistory:
    @pytest.mark.asyncio
    async def test_rows_newest_first_with_counts(self):
        store = await _store_with(
            make_report(score=70.0, tools_attempted=["ruff", "ty"]),
            None,
        )
        result = await get_quality_history(
            container_name="c1", limit=50, tenant_id=TENANT, store=store
        )
        assert result["count"] == 2
        first, second = result["results"]
        assert first["status"] == "failed"
        assert first["overall_score"] is None
        assert second["status"] == "complete"
        assert second["overall_score"] == 70.0
        assert second["grade"] == "C"
        assert second["tools_attempted"] == 2
        assert second["tools_succeeded"] == 2
        assert second["commit_sha"] == _SHA
        assert set(second) == {
            "id",
            "repo_url",
            "commit_sha",
            "status",
            "overall_score",
            "grade",
            "tools_attempted",
            "tools_succeeded",
            "created_at",
        }

    @pytest.mark.asyncio
    async def test_each_row_names_the_repo_it_graded(self):
        """A container name re-provisioned from another repo keeps old rows: each
        row must say which repo it graded, or a consumer attributes it wrongly."""
        store = FakeQualityResultStore()
        await store.save(
            make_report(score=90.0),
            tenant_id=TENANT,
            container_name="c1",
            repo_url="https://github.com/o/a",
            commit_sha=_OLD_SHA,
        )
        await store.save(
            make_report(score=60.0),
            tenant_id=TENANT,
            container_name="c1",
            repo_url="https://github.com/o/b",
            commit_sha=_SHA,
        )
        result = await get_quality_history(
            container_name="c1", limit=50, tenant_id=TENANT, store=store
        )
        by_sha = {r["commit_sha"]: r["repo_url"] for r in result["results"]}
        assert by_sha == {_SHA: "https://github.com/o/b", _OLD_SHA: "https://github.com/o/a"}

    @pytest.mark.asyncio
    async def test_empty(self):
        result = await get_quality_history(
            container_name="c1", limit=50, tenant_id=TENANT, store=FakeQualityResultStore()
        )
        assert result == {
            "container_name": "c1",
            "status": "not_analyzed",
            "count": 0,
            "results": [],
        }


class TestDeletedRoutes:
    """The aiosqlite scan-cache routes and the dashboard router are gone (#106)."""

    def test_cache_routes_are_not_registered(self):
        from app.api.endpoints.quality import router

        paths = {
            (route.path, tuple(sorted(getattr(route, "methods", []) or [])))
            for route in router.routes
        }
        flat = {p for p, _ in paths}
        assert "/cache/stats" not in flat
        assert "/cache/{container_name}" not in flat
        assert "/cache" not in flat
        assert "/{container_name}/cached" not in flat

    def test_dashboard_module_is_gone(self):
        import importlib.util

        assert importlib.util.find_spec("app.api.endpoints.quality_dashboard") is None
        assert importlib.util.find_spec("services.quality_scan_cache") is None
        assert importlib.util.find_spec("services.quality_dashboard_service") is None


# ---------------------------------------------------------------------------
# GET /quality/{container}/issues over HTTP
# ---------------------------------------------------------------------------


class TestGetQualityIssuesOverHttp:
    """Drive the read routes through the real router + the app's exception handlers.

    The direct-call tests above bypass FastAPI's Query validation entirely, so a
    re-added ``Query(pattern="^(low|moderate|high|critical)$")`` -- the original
    bug, which 422'd ``?severity=warning`` -- left them all green. These go
    through request parsing, the route's dependencies and
    ``register_exception_handlers`` (as src/main.py wires them), and assert on
    the body a client actually receives.
    """

    @pytest.fixture
    def client_and_store(self):
        import asyncio

        from coding_client.api.exception_handlers import register_exception_handlers
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from services.quality_result_store import get_quality_result_store

        issues = [
            make_issue(severity="warning", file_path="a.py", msg="w1"),
            make_issue(severity="error", file_path="b.py", msg="e1"),
            make_issue(severity="Warning", file_path="c.py", msg="w2"),
            make_issue(severity="info", file_path="d.py", msg="i1"),
        ]
        store = asyncio.run(_store_with(make_report(issues=issues)))
        analyzer = _analyzer()

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: TENANT
        app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        yield TestClient(app), store, analyzer
        analyzer.analyze.assert_not_awaited()

    def _replace(self, store, report):
        import asyncio

        store.rows.clear()
        asyncio.run(
            store.save(
                report, tenant_id=TENANT, container_name="c1", repo_url=None, commit_sha=_SHA
            )
        )

    @pytest.mark.parametrize("sev", ["warning", "WARNING"])
    def test_warning_filter_is_200_and_filters(self, client_and_store, sev):
        client, _, _ = client_and_store
        r = client.get("/quality/c1/issues", params={"severity": sev})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        assert sorted(i["message"] for i in body["issues"]) == ["w1", "w2"]

    def test_bogus_severity_is_400_listing_allowed_values(self, client_and_store):
        from app.api.endpoints.quality import QUALITY_ISSUE_SEVERITIES

        client, _, _ = client_and_store
        r = client.get("/quality/c1/issues", params={"severity": "bogus"})
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "HTTP_400"
        assert err["details"]["field"] == "severity"
        assert err["details"]["allowed_values"] == sorted(QUALITY_ISSUE_SEVERITIES)

    def test_no_severity_returns_all_issues(self, client_and_store):
        client, _, _ = client_and_store
        r = client.get("/quality/c1/issues")
        assert r.status_code == 200, r.text
        assert r.json()["total"] == 4

    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/quality/other/issues", {"severity": "warning"}),
            ("/quality/other/summary", {}),
            ("/quality/other/latest", {}),
        ],
    )
    def test_nothing_stored_is_200_not_analyzed(self, client_and_store, path, params):
        """Nothing measured is an explicit status, never an empty "clean" result."""
        client, _, _ = client_and_store
        r = client.get(path, params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "not_analyzed"
        assert body["inconclusive"] is True
        assert body["result_id"] is None

    # Literal, not derived from QUALITY_ISSUE_SEVERITIES: a test that compares
    # the response to the constant cannot notice the constant losing a member.
    _ALL_SEVERITIES = (
        "critical",
        "error",
        "high",
        "info",
        "low",
        "medium",
        "moderate",
        "unknown",
        "warning",
    )

    @pytest.mark.parametrize("sev", _ALL_SEVERITIES)
    def test_every_tool_severity_is_accepted(self, client_and_store, sev):
        client, _, _ = client_and_store
        r = client.get("/quality/c1/issues", params={"severity": sev})
        assert r.status_code == 200, r.text

    def test_400_allowed_values_is_the_full_literal_set(self, client_and_store):
        client, _, _ = client_and_store
        r = client.get("/quality/c1/issues", params={"severity": "bogus"})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["details"]["allowed_values"] == list(self._ALL_SEVERITIES)

    def test_summary_severity_distribution_counts_mixed_case_vocabularies(self, client_and_store):
        client, store, _ = client_and_store
        sevs = [
            "error",
            "ERROR",
            "Warning",
            "warning",
            "info",
            "medium",
            "Medium",
            "unknown",
            "critical",
            "HIGH",
            "low",
            "moderate",
            "not-a-severity",
        ]
        issues = [make_issue(severity=s, file_path=f"f{i}.py") for i, s in enumerate(sevs)]
        self._replace(store, make_report(issues=issues))
        r = client.get("/quality/c1/summary")
        assert r.status_code == 200, r.text
        assert r.json()["severity_distribution"] == {
            "critical": 1,
            "error": 2,
            "high": 1,
            "info": 1,
            "low": 1,
            "medium": 2,
            "moderate": 1,
            "unknown": 1,
            "warning": 2,
        }
        assert r.json()["total_issues"] == len(sevs)

    _PROVENANCE = {
        "inconclusive": False,
        "partial": False,
        "tools_attempted": ["ruff"],
        "tools_succeeded": ["ruff"],
        "tools_failed": [],
    }

    @pytest.mark.parametrize("path", ["/quality/c1/issues", "/quality/c1/latest"])
    def test_issue_shape_and_provenance(self, client_and_store, path):
        client, store, _ = client_and_store
        issue = make_issue(
            severity="error",
            file_path="x.py",
            line=7,
            msg="boom",
            tool="ruff",
            rule_id="F401",
            column=3,
        )
        self._replace(store, make_report(issues=[issue]))
        r = client.get(path)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["issues"] == [
            {
                "tool": "ruff",
                "severity": "error",
                "message": "boom",
                "file_path": "x.py",
                "line_number": 7,
                "column_number": 3,
                "rule_id": "F401",
                "repo": None,
            }
        ]
        assert "line" not in body["issues"][0]
        assert {k: body[k] for k in self._PROVENANCE} == self._PROVENANCE
        assert body["commit_sha"] == _SHA

    def test_gate_over_http_reports_status(self, client_and_store):
        client, _, _ = client_and_store
        with patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA):
            r = client.get("/quality/c1/gate")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "complete"
        assert body["source"] == "stored"
        assert body["commit_sha"] == _SHA


# ---------------------------------------------------------------------------
# GET /gate grades the CURRENT commit: compute-on-miss keyed by commit
# ---------------------------------------------------------------------------


class TestGateGradesTheCurrentCommit:
    """The upgrade procedure's ``verify_quality_gate`` calls /gate right after
    changing code, with no analyze first. A stored row for an OLDER commit must
    never be graded as if it were the current one."""

    async def _gate(self, store, analyzer, sha, **kw):
        params = {"min_score": 80.0, "max_critical": 0, "max_high": 3, "min_coverage": None}
        params.update(kw)
        with (
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=sha) as mock_sha,
            patch(
                _VALIDATE_CONTAINER,
                new_callable=AsyncMock,
                return_value=_make_mock_container(path="/workspace"),
            ),
        ):
            result = await evaluate_quality_gate(
                container_name="c1", tenant_id=TENANT, store=store, analyzer=analyzer, **params
            )
        mock_sha.assert_awaited_once_with("c1", TENANT)
        return result

    @pytest.mark.asyncio
    async def test_stored_row_for_current_sha_is_graded_without_analysis(self):
        store = await _store_with(make_report(score=95.0), sha=_SHA)
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock()
        result = await self._gate(store, analyzer, _SHA)
        analyzer.analyze.assert_not_awaited()
        assert result.source == "stored"
        assert result.commit_sha == _SHA
        assert result.result_id == store.rows[0]["id"]
        assert result.passed is True
        assert len(store.save_calls) == 1  # only the seed; nothing new persisted

    @pytest.mark.asyncio
    async def test_row_for_an_older_sha_is_not_graded_the_head_is_analyzed(self):
        # The stored run PASSES; the current commit regressed. Grading the stale
        # row would pass a regression.
        store = await _store_with(make_report(score=99.0), sha=_OLD_SHA)
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=40.0))
        result = await self._gate(store, analyzer, _SHA)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"
        assert result.commit_sha == _SHA
        assert result.passed is False
        assert result.overall_score == 40.0
        # Persisted, keyed by the current commit, and the gate names that row.
        assert len(store.save_calls) == 2
        assert store.save_calls[-1]["commit_sha"] == _SHA
        assert store.save_calls[-1]["tenant_id"] == TENANT
        assert result.result_id == store.rows[-1]["id"]
        assert store.latest_for_sha_calls == [(TENANT, "c1", _SHA)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stuck_status", ["failed", "inconclusive"])
    async def test_failed_or_inconclusive_row_for_this_sha_is_reanalyzed(self, stuck_status):
        # A transient tool error must not pin the gate shut for this commit.
        store = await _store_with(make_report(score=95.0), sha=_SHA)
        store.rows[0]["status"] = stuck_status
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        result = await self._gate(store, analyzer, _SHA)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unresolvable_sha_analyzes(self):
        store = await _store_with(make_report(score=99.0), sha=_SHA)
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        result = await self._gate(store, analyzer, None)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"
        assert result.commit_sha is None
        assert result.passed is True
        # No sha → no stored lookup can match it, so none is attempted.
        assert store.latest_for_sha_calls == []
        assert store.save_calls[-1]["commit_sha"] is None

    @pytest.mark.asyncio
    async def test_persist_failure_still_returns_the_gate(self):
        store = FakeQualityResultStore()
        store.fail_saves = RuntimeError("db down")
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        result = await self._gate(store, analyzer, _SHA)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"
        assert result.passed is True
        assert result.result_id is None

    @pytest.mark.asyncio
    async def test_store_read_failure_analyzes_instead_of_500(self):
        store = FakeQualityResultStore()
        store.latest_for_sha = AsyncMock(side_effect=RuntimeError("db down"))
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        result = await self._gate(store, analyzer, _SHA)
        analyzer.analyze.assert_awaited_once()
        assert result.source == "analyzed"

    @pytest.mark.asyncio
    async def test_analyzer_producing_nothing_is_a_failed_gate_not_a_500(self):
        store = FakeQualityResultStore()
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=None)
        result = await self._gate(store, analyzer, _SHA)
        assert result.passed is False
        assert result.status == "failed"
        assert result.source == "analyzed"
        assert [r["status"] for r in store.rows] == ["failed"]

    def _client(self, store, analyzer):
        from coding_client.api.exception_handlers import register_exception_handlers
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from services.quality_result_store import get_quality_result_store

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: TENANT
        app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        return TestClient(app)

    @pytest.mark.parametrize(
        ("stored_sha", "expect_source", "expect_awaits"),
        [(_SHA, "stored", 0), (_OLD_SHA, "analyzed", 1)],
    )
    def test_over_http(self, stored_sha, expect_source, expect_awaits):
        import asyncio

        store = asyncio.run(_store_with(make_report(score=95.0), sha=stored_sha))
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        with (
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA),
            patch(
                _VALIDATE_CONTAINER,
                new_callable=AsyncMock,
                return_value=_make_mock_container(path="/workspace"),
            ),
        ):
            r = self._client(store, analyzer).get("/quality/c1/gate")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == expect_source
        assert body["commit_sha"] == _SHA
        assert analyzer.analyze.await_count == expect_awaits

    def test_persist_failure_over_http_is_200(self):
        store = FakeQualityResultStore()
        store.fail_saves = RuntimeError("db down")
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=95.0))
        with (
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA),
            patch(
                _VALIDATE_CONTAINER,
                new_callable=AsyncMock,
                return_value=_make_mock_container(path="/workspace"),
            ),
        ):
            r = self._client(store, analyzer).get("/quality/c1/gate")
        assert r.status_code == 200, r.text
        assert r.json()["source"] == "analyzed"
        assert r.json()["result_id"] is None


# ---------------------------------------------------------------------------
# A non-UUID tenant id is a 400 with a clear detail, never a database 500
# ---------------------------------------------------------------------------


class TestNonUuidTenantIs400:
    @pytest.mark.parametrize(
        "path",
        [
            "/quality/c1/latest",
            "/quality/c1/issues",
            "/quality/c1/summary",
            "/quality/c1/gate",
            "/quality/c1/trend",
            "/quality/c1/history",
        ],
    )
    def test_read_routes(self, path):
        from coding_client.api.exception_handlers import register_exception_handlers
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from services.quality_result_store import get_quality_result_store

        store = FakeQualityResultStore()
        store.latest = AsyncMock(side_effect=AssertionError("store reached"))
        store.history = AsyncMock(side_effect=AssertionError("store reached"))
        store.latest_for_sha = AsyncMock(side_effect=AssertionError("store reached"))
        analyzer = _analyzer()
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: "not-a-uuid"
        app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        with patch(_RESOLVE_SHA, new_callable=AsyncMock) as mock_sha:
            r = TestClient(app).get(path)
            mock_sha.assert_not_awaited()
        assert r.status_code == 400, r.text
        assert "Invalid tenant id: expected a UUID" in r.text
        analyzer.analyze.assert_not_awaited()
