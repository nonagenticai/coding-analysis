"""Tests for quality.py uncovered paths: analyze-sync, helper functions, health/info.

Covers lines 505-513, 547-631, 637, 643, 687-772, 1095-1110, 1137-1142, 1433-1678.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.quality import (
    QualityAnalysisSyncRequest,
    _build_summary,
    _calculate_grade,
    _generate_recommendations,
    get_analysis_results,
    get_quality_analyzer,
    quality_health,
    quality_info,
    run_quality_analysis_sync,
)
from tests.unit.quality_store_fakes import FakeQualityResultStore, make_issue, make_report

_VALIDATE_CONTAINER = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"
_SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
_TENANT = "0195f48b-0000-7000-8000-00000000000a"

#: Every key /analyze-sync returned BEFORE results were persisted (b9a02a1).
#: Literal on purpose: the route is an auto-run verifier in the live conveyor
#: (it verifies applyCodeChanges / executeWorkspaceAgent), so persisting may only
#: ADD keys — a test that derived this set from the response could not notice a
#: removal or rename.
_PRE_EXISTING_SYNC_KEYS = frozenset(
    {
        "analysis_id",
        "success",
        "inconclusive",
        "partial",
        "tools_attempted",
        "tools_succeeded",
        "tools_failed",
        "language",
        "tools_results",
        "overall_score",
        "total_issues",
        "issues",
        "execution_time",
        "created_at",
        "error",
    }
)
_ADDED_SYNC_KEYS = frozenset(
    {
        "status",
        "result_id",
        "commit_sha",
        "persisted",
        "scope",
        "scope_not_analysed",
        "repo",
        "workdir",
        # #133: the tools a restricted run was asked for.
        "tools_requested",
    }
)


# ---------------------------------------------------------------------------
# _calculate_grade
# ---------------------------------------------------------------------------


class TestCalculateGrade:
    def test_grade_a(self):
        assert _calculate_grade(95.0) == "A"
        assert _calculate_grade(90.0) == "A"

    def test_grade_b(self):
        assert _calculate_grade(85.0) == "B"
        assert _calculate_grade(80.0) == "B"

    def test_grade_c(self):
        assert _calculate_grade(75.0) == "C"
        assert _calculate_grade(70.0) == "C"

    def test_grade_d(self):
        assert _calculate_grade(65.0) == "D"
        assert _calculate_grade(60.0) == "D"

    def test_grade_f(self):
        assert _calculate_grade(59.9) == "F"
        assert _calculate_grade(0.0) == "F"


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_empty_issues(self):
        result = _build_summary([])
        assert result["total_issues"] == 0

    def test_with_severity_counts(self):
        issues = []
        for sev in ["critical", "high", "high", "low"]:
            issue = MagicMock()
            issue.severity = sev
            issues.append(issue)
        result = _build_summary(issues)
        assert result["total_issues"] == 4
        assert result["critical"] == 1
        assert result["high"] == 2
        assert result["low"] == 1

    def test_issues_without_severity(self):
        issue = MagicMock(spec=[])  # No severity attribute
        result = _build_summary([issue])
        assert result["total_issues"] == 1
        assert result["unknown"] == 1


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def test_no_issues(self):
        result = _generate_recommendations([])
        assert any("Great job" in r for r in result)

    def test_with_issues(self):
        issues = [MagicMock(severity="low")]
        result = _generate_recommendations(issues)
        assert any("1 quality issues" in r for r in result)

    def test_with_critical_issues(self):
        issues = [MagicMock(severity="critical"), MagicMock(severity="critical")]
        result = _generate_recommendations(issues)
        assert any("2 critical" in r for r in result)

    def test_with_high_issues(self):
        issues = [MagicMock(severity="high")]
        result = _generate_recommendations(issues)
        assert any("1 high-severity" in r for r in result)

    def test_inconclusive_never_congratulates(self):
        # Regression (Defect A): an inconclusive run has an empty issue list but
        # must NOT be reported as a pass. No "Great job", only the honest abstain.
        result = _generate_recommendations([], inconclusive=True)
        assert not any("Great job" in r for r in result)
        assert any("inconclusive" in r.lower() for r in result)
        assert any("NOT a pass" in r for r in result)


# ---------------------------------------------------------------------------
# _render_grade
# ---------------------------------------------------------------------------


class TestRenderGrade:
    def test_conclusive_uses_letter_grade(self):
        from app.api.endpoints.quality import _render_grade

        result = MagicMock(inconclusive=False, overall_score=95.0, grade="A")
        assert _render_grade(result) == "A"

    def test_inconclusive_surfaces_abstain_grade_not_f(self):
        # Regression (Defect A): the -1.0 sentinel must not be recomputed into
        # a misleading "F"; the honest "N/A" abstain grade is surfaced instead.
        from app.api.endpoints.quality import _render_grade

        result = MagicMock(inconclusive=True, overall_score=-1.0, grade="N/A")
        rendered = _render_grade(result)
        assert rendered == "N/A"
        assert rendered != "F"


# ---------------------------------------------------------------------------
# get_quality_analyzer
# ---------------------------------------------------------------------------


class TestGetQualityAnalyzer:
    def test_returns_analyzer(self):
        result = get_quality_analyzer()
        assert result is not None


# ---------------------------------------------------------------------------
# quality_health / quality_info
# ---------------------------------------------------------------------------


class TestQualityHealthInfo:
    @pytest.mark.asyncio
    async def test_health(self):
        result = await quality_health()
        assert result["status"] == "healthy"
        assert result["service"] == "quality-analysis"

    @pytest.mark.asyncio
    async def test_info(self):
        result = await quality_info()
        assert result["service"] == "quality-analysis"
        assert "python" in result["available_tools"]


# ---------------------------------------------------------------------------
# run_quality_analysis_sync: response shape (container runs)
# ---------------------------------------------------------------------------


def _ws_container() -> MagicMock:
    container = MagicMock()
    container.codebase_path = "/workspace"
    container.repo_url = None
    return container


class TestRunQualityAnalysisSync:
    @pytest.fixture(autouse=True)
    def _no_live_sha_lookup(self):
        """Never reach a real coding-containers from a unit test."""
        with patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None) as mock_sha:
            yield mock_sha

    @pytest.fixture(autouse=True)
    def _workspace(self):
        with patch(
            _VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_ws_container()
        ) as mock_validate:
            yield mock_validate

    @pytest.mark.asyncio
    async def test_no_container_no_code_raises_400(self):
        request = QualityAnalysisSyncRequest()
        mock_analyzer = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await run_quality_analysis_sync(
                request=request,
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_report_is_inconclusive(self):
        request = QualityAnalysisSyncRequest(
            container_name="ws-1",
            language="python",
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=None)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["success"] is False
        assert result["error"] == "Analysis produced no results"
        # No report is inconclusive, never a clean empty result.
        assert result["inconclusive"] is True
        assert result["issues"] == []
        assert result["tools_attempted"] == []
        assert result["tools_succeeded"] == []
        assert result["tools_failed"] == []

    @pytest.mark.asyncio
    async def test_report_is_serialised(self):
        request = QualityAnalysisSyncRequest(
            container_name="ws-1",
            language="python",
        )

        mock_issue = MagicMock()
        mock_issue.tool = "ruff"
        mock_issue.severity = "low"
        mock_issue.message = "style issue"
        mock_issue.file_path = "code.py"
        mock_issue.line_number = 1
        mock_issue.column_number = 1
        mock_issue.rule_id = "E001"

        mock_report = MagicMock()
        mock_report.issues = [mock_issue]
        mock_report.overall_score = 92.0
        mock_report.inconclusive = False
        # Explicit provenance: a bare MagicMock auto-creates a non-iterable
        # `.tools_attempted` (and a truthy `.partial`), which would break the
        # per-tool reconstruction. Set them to a real, all-succeeded run.
        mock_report.partial = False
        mock_report.tools_attempted = ["ruff"]
        mock_report.tools_succeeded = ["ruff"]
        mock_report.tools_failed = []
        mock_report.execution_time = 0.5
        mock_report.generated_at = "2026-01-01T00:00:00"
        mock_report.job_id = "job-1"

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=mock_report)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["success"] is True
        assert result["inconclusive"] is False
        assert result["partial"] is False
        assert result["overall_score"] == 92.0
        assert result["total_issues"] == 1
        assert result["issues"][0]["tool"] == "ruff"
        assert result["tools_attempted"] == ["ruff"]
        assert result["tools_succeeded"] == ["ruff"]
        assert result["tools_failed"] == []
        assert result["tools_results"] == {"ruff": {"success": True}}

    @pytest.mark.asyncio
    async def test_inconclusive_run_is_honest(self):
        """Regression (F3): an inconclusive scan (0 tools succeeded -> -1.0
        sentinel) must NOT surface as success with a numeric score. It reports
        inconclusive=true, success=false, and overall_score=None (never -1.0 or
        the 100.0-for-empty default), mirroring /results, /summary and /gate."""
        request = QualityAnalysisSyncRequest(
            container_name="ws-1",
            language="python",
        )

        mock_report = MagicMock()
        mock_report.issues = []
        mock_report.overall_score = -1.0
        mock_report.inconclusive = True
        # Inconclusive: every attempted tool failed. Set provenance explicitly
        # (MagicMock auto-attrs are non-iterable / truthy).
        mock_report.partial = False
        mock_report.tools_attempted = ["ruff", "ty"]
        mock_report.tools_succeeded = []
        mock_report.tools_failed = ["ruff", "ty"]
        mock_report.execution_time = 0.3
        mock_report.generated_at = "2026-01-01T00:00:00"
        mock_report.job_id = "job-incon"

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=mock_report)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["inconclusive"] is True
        assert result["success"] is False
        assert result["partial"] is False
        # Never present the -1.0 sentinel (or 100.0) as a measured score.
        assert result["overall_score"] is None
        assert result["overall_score"] not in (-1.0, 100.0)
        assert "inconclusive" in result["error"].lower()
        # Provenance still names what was attempted / failed.
        assert result["tools_attempted"] == ["ruff", "ty"]
        assert result["tools_failed"] == ["ruff", "ty"]
        assert result["tools_results"] == {
            "ruff": {"success": False},
            "ty": {"success": False},
        }

    @pytest.mark.asyncio
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_container_name_path(self, mock_validate):
        mock_container = MagicMock()
        mock_container.codebase_path = "/app/data/repo"
        mock_validate.return_value = mock_container

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=None)

        request = QualityAnalysisSyncRequest(container_name="test")
        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_unknown_language_without_report(self):
        request = QualityAnalysisSyncRequest(
            container_name="ws-1",
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=None)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_requested_language_is_echoed(self):
        """An explicitly requested language is echoed even with no report."""
        for lang in ["javascript", "typescript", "java", "go", "rust"]:
            request = QualityAnalysisSyncRequest(
                container_name="ws-1",
                language=lang,
            )
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze = AsyncMock(return_value=None)
            result = await run_quality_analysis_sync(
                request=request,
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
            assert result["language"] == lang

    @pytest.mark.asyncio
    async def test_language_reports_detection_when_request_omits_it(self):
        """The endpoint must report what detect_primary_language() found.

        Echoing ``request.language`` is why a plainly-Python repo came back as
        ``language: "unknown"`` in the same payload that listed six Python tools
        under ``tools_attempted``.
        """
        request = QualityAnalysisSyncRequest(container_name="ws-1")  # no language

        mock_report = MagicMock()
        mock_report.issues = []
        mock_report.overall_score = 100.0
        mock_report.inconclusive = False
        mock_report.partial = False
        mock_report.tools_attempted = ["ruff"]
        mock_report.tools_succeeded = ["ruff"]
        mock_report.tools_failed = []
        mock_report.execution_time = 0.5
        mock_report.generated_at = "2026-01-01T00:00:00"
        mock_report.job_id = "job-lang"
        mock_report.language = "python"

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=mock_report)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["language"] == "python", (
            "the endpoint echoed the (absent) request language instead of "
            "reporting the language the analyzer actually detected"
        )

    @pytest.mark.asyncio
    async def test_explicit_request_language_is_honoured(self):
        """An explicitly requested language still wins over detection."""
        request = QualityAnalysisSyncRequest(container_name="ws-1", language="typescript")

        mock_report = MagicMock()
        mock_report.issues = []
        mock_report.overall_score = 100.0
        mock_report.inconclusive = False
        mock_report.partial = False
        mock_report.tools_attempted = ["ruff"]
        mock_report.tools_succeeded = ["ruff"]
        mock_report.tools_failed = []
        mock_report.execution_time = 0.5
        mock_report.generated_at = "2026-01-01T00:00:00"
        mock_report.job_id = "job-lang-2"
        mock_report.language = "python"

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=mock_report)

        result = await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="test-tenant",
        )
        assert result["language"] == "typescript"

    @pytest.mark.asyncio
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_analysis_runs_as_the_authenticated_tenant(self, mock_validate):
        """The tenant the request authenticated as reaches the analyzer.

        It is the only thing that lets the downstream coding-containers exec
        address the customer's workspace instead of the service account's.
        """
        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_validate.return_value = mock_container

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=None)

        request = QualityAnalysisSyncRequest(container_name="ws-1")
        await run_quality_analysis_sync(
            request=request,
            analyzer=mock_analyzer,
            tenant_id="0195f48b-d000-7000-8000-000000000003",
        )
        assert (
            mock_analyzer.analyze.await_args.kwargs.get("tenant_id")
            == "0195f48b-d000-7000-8000-000000000003"
        ), "the endpoint dropped the authenticated tenant before calling the analyzer"

    @pytest.mark.asyncio
    async def test_runtime_error_raises_500(self):
        request = QualityAnalysisSyncRequest(container_name="ws-1")
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await run_quality_analysis_sync(
                request=request,
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500


class TestAnalyzeSyncPersistence:
    """POST /analyze-sync stores container runs WITHOUT changing its contract (#106)."""

    @staticmethod
    def _container():
        container = MagicMock()
        container.codebase_path = "/workspace"
        container.repo_url = "https://github.com/org/repo.git"
        return container

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_report", [True, False])
    @patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_pre_existing_keys_all_still_present(self, mock_validate, _sha, has_report):
        mock_validate.return_value = self._container()
        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(
            return_value=make_report(issues=[make_issue()]) if has_report else None
        )
        store = FakeQualityResultStore()

        result = await run_quality_analysis_sync(
            request=QualityAnalysisSyncRequest(container_name="ws-1"),
            analyzer=analyzer,
            tenant_id=_TENANT,
            store=store,
        )
        assert _PRE_EXISTING_SYNC_KEYS <= set(result), _PRE_EXISTING_SYNC_KEYS - set(result)
        assert set(result) == _PRE_EXISTING_SYNC_KEYS | _ADDED_SYNC_KEYS
        assert result["persisted"] is True
        assert result["result_id"] == store.rows[0]["id"]
        assert result["commit_sha"] == _SHA
        assert result["status"] == ("complete" if has_report else "failed")
        assert store.rows[0]["commit_sha"] == _SHA
        assert store.rows[0]["tenant_id"] == _TENANT
        assert store.rows[0]["container_name"] == "ws-1"
        assert store.rows[0]["repo_url"] == "https://github.com/org/repo.git"

    @pytest.mark.asyncio
    @patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_store_failure_still_returns_the_analysis(self, mock_validate, _sha):
        """A DB hiccup must not red every code-change verification."""
        mock_validate.return_value = self._container()
        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report(score=88.0))
        store = FakeQualityResultStore()
        store.fail_saves = RuntimeError("connection refused")

        result = await run_quality_analysis_sync(
            request=QualityAnalysisSyncRequest(container_name="ws-1"),
            analyzer=analyzer,
            tenant_id=_TENANT,
            store=store,
        )
        assert len(store.save_calls) == 1  # it was attempted
        assert result["persisted"] is False
        assert result["result_id"] is None
        assert result["success"] is True
        assert result["overall_score"] == 88.0
        assert _PRE_EXISTING_SYNC_KEYS <= set(result)

    def test_store_failure_is_http_200_with_persisted_false(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from services.quality_result_store import get_quality_result_store

        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        store = FakeQualityResultStore()
        store.fail_saves = OSError("db down")
        app = FastAPI()
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: _TENANT
        app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=self._container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            r = TestClient(app).post("/quality/analyze-sync", json={"container_name": "ws-1"})
        assert r.status_code == 200, r.text
        assert r.json()["persisted"] is False
        assert r.json()["success"] is True


class TestInlineCodeContentNeedsAWorkspace:
    """Ad-hoc ``code_content`` (no ``container_name``) is refused, never run.

    Live 2026-10-12: it wrote the code to this pod's /tmp and exec'd every
    tool into a container named 'sync-analysis' that does not exist (404 per
    tool), so the answer was always "inconclusive: 0/6 tools succeeded".
    """

    _MESSAGE = (
        "inline code_content needs a workspace: send container_name (+ paths) "
        "so tools run in the workspace"
    )

    @pytest.mark.asyncio
    async def test_inline_code_is_422_and_runs_nothing(self):
        analyzer = AsyncMock()
        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock) as mock_validate,
            patch(_RESOLVE_SHA, new_callable=AsyncMock) as mock_sha,
            patch("tempfile.mkdtemp") as mock_mkdtemp,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_quality_analysis_sync(
                    request=QualityAnalysisSyncRequest(code_content="x = 1\n", language="python"),
                    analyzer=analyzer,
                    tenant_id=_TENANT,
                    store=store,
                )
        assert exc_info.value.status_code == 422
        assert self._MESSAGE in str(exc_info.value.detail)
        analyzer.analyze.assert_not_awaited()
        mock_validate.assert_not_awaited()
        mock_sha.assert_not_awaited()
        mock_mkdtemp.assert_not_called()
        assert store.save_calls == []

    def test_inline_code_over_http_is_422_with_zero_exec_calls(self):
        from coding_client.api.exception_handlers import register_exception_handlers
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.endpoints.quality import router
        from auth.tenant import require_tenant
        from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer
        from services.quality_result_store import get_quality_result_store

        client = MagicMock()
        client.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
        client.read_file = AsyncMock(return_value="")
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: _TENANT
        # The REAL analyzer: if the route ran anything, these clients would see it.
        app.dependency_overrides[get_quality_analyzer] = ArchitectureQualityAnalyzer
        app.dependency_overrides[get_quality_result_store] = FakeQualityResultStore
        with (
            patch(
                "processing.architecture.quality_analyzer.get_coding_containers_client",
                return_value=client,
            ),
            patch(
                "processing.architecture.tool_execution._base.get_coding_containers_client",
                return_value=client,
            ),
        ):
            r = TestClient(app).post(
                "/quality/analyze-sync", json={"code_content": "x = 1\n", "language": "python"}
            )
        assert r.status_code == 422, r.text
        assert self._MESSAGE in r.text
        assert client.execute.await_count == 0
        assert client.read_file.await_count == 0

    @pytest.mark.asyncio
    async def test_neither_container_nor_code_is_still_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(),
                analyzer=AsyncMock(),
                tenant_id=_TENANT,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_code_content_with_a_container_runs_in_the_workspace(self):
        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_ws_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            result = await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(container_name="ws-1", code_content="x = 1"),
                analyzer=analyzer,
                tenant_id=_TENANT,
                store=FakeQualityResultStore(),
            )
        assert analyzer.analyze.await_args.args[:2] == (Path("/workspace"), "ws-1")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# get_analysis_results
# ---------------------------------------------------------------------------


class TestGetAnalysisResults:
    """Tests for get_analysis_results.

    The implementation looks up task results in Redis by task_id.
    If found, returns the parsed JSON. If not found, returns pending status.
    """

    _GET_REDIS = "app.api.endpoints.quality.get_redis_service"

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_pending(self):
        mock_analyzer = AsyncMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="bad-format",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_short_task_id_returns_pending(self):
        mock_analyzer = AsyncMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_redis_hit_returns_cached_result(self):
        import json

        cached_data = {
            "status": "completed",
            "task_id": "quality_analysis_myrepo_12345",
            "result": {"overall_score": 88.0, "cached": True},
        }
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=AsyncMock(),
                tenant_id="test-tenant",
            )
        assert result["status"] == "completed"
        assert result["result"]["overall_score"] == 88.0

    @pytest.mark.asyncio
    async def test_redis_hit_with_completed_analysis(self):
        import json

        cached_data = {
            "status": "completed",
            "task_id": "quality_analysis_myrepo_12345",
            "result": {"overall_score": 90.0, "cached": False},
        }
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=AsyncMock(),
                tenant_id="test-tenant",
            )
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_redis_miss_returns_pending(self):
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=AsyncMock(),
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_value_error_raises_500(self):
        with patch(self._GET_REDIS, side_effect=ValueError("cache broke")):
            with pytest.raises(HTTPException) as exc_info:
                await get_analysis_results(
                    task_id="quality_analysis_myrepo_12345",
                    analyzer=AsyncMock(),
                    tenant_id="test-tenant",
                )
            assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_supported_languages
# ---------------------------------------------------------------------------


class TestGetSupportedLanguages:
    @pytest.mark.asyncio
    async def test_returns_languages(self):
        from app.api.endpoints.quality import get_supported_languages

        result = await get_supported_languages()
        assert "languages" in result
        lang_names = [lang["name"] for lang in result["languages"]]
        assert "python" in lang_names
        assert "java" in lang_names


# ---------------------------------------------------------------------------
# Invariant: no response may claim success while inconclusive
# ---------------------------------------------------------------------------


def _sync_report(inconclusive: bool):
    report = MagicMock()
    report.issues = []
    report.overall_score = -1.0 if inconclusive else 100.0
    report.inconclusive = inconclusive
    report.partial = False
    report.tools_attempted = ["ruff"]
    report.tools_succeeded = [] if inconclusive else ["ruff"]
    report.tools_failed = ["ruff"] if inconclusive else []
    report.execution_time = 0.1
    report.generated_at = "2026-01-01T00:00:00"
    report.job_id = "job-x"
    report.language = "python"
    return report


class TestSuccessNeverWithInconclusive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", ["no_report", "inconclusive", "conclusive"])
    async def test_sync_invariant(self, outcome):
        report = None if outcome == "no_report" else _sync_report(outcome == "inconclusive")
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value=report)

        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_ws_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            result = await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(container_name="ws-1", language="python"),
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
                store=FakeQualityResultStore(),
            )
        assert "inconclusive" in result
        assert not (result["success"] is True and result["inconclusive"] is True)
        if outcome == "conclusive":
            assert result["success"] is True and result["inconclusive"] is False
        else:
            assert result["success"] is False and result["inconclusive"] is True
