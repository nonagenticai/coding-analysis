"""Tests for quality.py endpoint handlers to improve coverage.

Targets: quality gate (over a STORED result, #106) with coverage check,
WebSocket endpoint (1433-1678), analyze task (504-513), results endpoint (692+).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from app.api.endpoints.quality import (
    _build_summary,
    _calculate_grade,
    _generate_recommendations,
    evaluate_quality_gate,
    get_analysis_results,
    quality_websocket,
)
from models.architecture_models import (
    ArchitectureQualityIssue,
    QualityReport,
)
from tests.conftest import ws_licence_headers
from tests.unit.quality_store_fakes import FakeQualityResultStore

_VALIDATE = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"
_GET_STORE = "app.api.endpoints.quality.get_quality_result_store"
TENANT = "0195f48b-0000-7000-8000-00000000000a"
_SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret


async def _stored(report):
    """A store holding exactly ``report`` as the run for the container's HEAD."""
    store = FakeQualityResultStore()
    await store.save(
        report, tenant_id=TENANT, container_name="test", repo_url=None, commit_sha=_SHA
    )
    return store


def _make_issue(severity="low", file_path="test.py", rule_id="R001"):
    return ArchitectureQualityIssue(
        id="test-1",
        severity=severity,
        message="test issue",
        rule_id=rule_id,
        rule_name="test-rule",
        file_path=file_path,
        line_number=1,
        tool="test",
        category="test",
    )


def _make_quality_report(
    score=85.0,
    issues=None,
    codebase_id="test-codebase",
    inconclusive=False,
    partial=False,
    tools_attempted=None,
    tools_succeeded=None,
    tools_failed=None,
):
    """Create a real QualityReport Pydantic model instead of a MagicMock."""
    issue_list = issues if issues is not None else []
    grade = "N/A" if inconclusive else _calculate_grade(score)
    return QualityReport(
        job_id="test-job",
        codebase_id=codebase_id,
        generated_at=datetime.now(UTC).isoformat(),
        overall_score=score,
        grade=grade,
        inconclusive=inconclusive,
        partial=partial,
        tools_attempted=tools_attempted if tools_attempted is not None else [],
        tools_succeeded=tools_succeeded if tools_succeeded is not None else [],
        tools_failed=tools_failed if tools_failed is not None else [],
        issues=issue_list,
        summary=_build_summary(issue_list),
        recommendations=_generate_recommendations(issue_list, inconclusive=inconclusive),
        execution_time=1.5,
    )


# ===================================================================
# Helper function unit tests
# ===================================================================


class TestCalculateGrade:
    """Test _calculate_grade directly for all grade boundaries."""

    def test_grade_a(self):
        assert _calculate_grade(90.0) == "A"
        assert _calculate_grade(100.0) == "A"
        assert _calculate_grade(95.5) == "A"

    def test_grade_b(self):
        assert _calculate_grade(80.0) == "B"
        assert _calculate_grade(89.9) == "B"

    def test_grade_c(self):
        assert _calculate_grade(70.0) == "C"
        assert _calculate_grade(79.9) == "C"

    def test_grade_d(self):
        assert _calculate_grade(60.0) == "D"
        assert _calculate_grade(69.9) == "D"

    def test_grade_f(self):
        assert _calculate_grade(59.9) == "F"
        assert _calculate_grade(0.0) == "F"


class TestBuildSummary:
    """Test _build_summary with real issue objects."""

    def test_empty_issues(self):
        result = _build_summary([])
        assert result == {"total_issues": 0}

    def test_mixed_severities(self):
        issues = [
            _make_issue("critical"),
            _make_issue("critical"),
            _make_issue("high"),
            _make_issue("low"),
            _make_issue("moderate"),
        ]
        result = _build_summary(issues)
        assert result["total_issues"] == 5
        assert result["critical"] == 2
        assert result["high"] == 1
        assert result["low"] == 1
        assert result["moderate"] == 1

    def test_single_severity(self):
        issues = [_make_issue("high"), _make_issue("high")]
        result = _build_summary(issues)
        assert result["total_issues"] == 2
        assert result["high"] == 2
        assert "critical" not in result


class TestGenerateRecommendations:
    """Test _generate_recommendations with real issue objects."""

    def test_no_issues_gives_positive_message(self):
        recs = _generate_recommendations([])
        assert len(recs) == 1
        assert "Great job" in recs[0]

    def test_critical_issues_mentioned(self):
        issues = [_make_issue("critical"), _make_issue("critical")]
        recs = _generate_recommendations(issues)
        assert any("2 critical" in r for r in recs)
        assert any("Address 2 quality issues" in r for r in recs)

    def test_high_issues_mentioned(self):
        issues = [_make_issue("high")]
        recs = _generate_recommendations(issues)
        assert any("1 high-severity" in r for r in recs)

    def test_low_issues_no_critical_high_mention(self):
        issues = [_make_issue("low"), _make_issue("low")]
        recs = _generate_recommendations(issues)
        assert any("Address 2 quality issues" in r for r in recs)
        # Should not mention critical or high
        assert not any("critical" in r for r in recs)
        assert not any("high" in r for r in recs)


# ===================================================================
# Quality gate with coverage check (lines 1095-1142)
# ===================================================================


class TestQualityGateWithCoverage:
    """Test evaluate_quality_gate with min_coverage parameter.

    The container's HEAD is the stored run's commit, so every case here grades
    the stored row (the compute-on-miss path is in test_quality_endpoints.py).
    """

    @pytest.fixture(autouse=True)
    def _head_is_the_stored_sha(self):
        with patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA):
            yield

    @pytest.mark.asyncio
    async def test_gate_with_coverage_check_passing(self, tmp_path):
        """When coverage check is provided and passes, validate all gate check fields."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")
        (project / "test_main.py").write_text("assert True\n")
        (project / "utils.py").write_text("y = 2\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=90.0, issues=[])
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=20.0,  # 1/3 = 33% > 20%
                tenant_id=TENANT,
                store=store,
            )
            assert gate_result.passed is True
            assert gate_result.overall_score == 90.0
            assert gate_result.grade == "A"
            # Should have 4 checks (score, critical, high, coverage)
            assert len(gate_result.gate_checks) == 4
            check_names = [c.name for c in gate_result.gate_checks]
            assert check_names == [
                "minimum_score",
                "critical_issues",
                "high_issues",
                "test_coverage",
            ]

            coverage_check = gate_result.gate_checks[3]
            assert coverage_check.passed is True
            assert coverage_check.threshold == 20.0
            # 1 test file / 3 total files = 33.3%
            assert coverage_check.actual == pytest.approx(33.333, abs=0.1)

    @pytest.mark.asyncio
    async def test_gate_with_coverage_check_failing(self, tmp_path):
        """When coverage check fails (no test files), validate actual vs threshold."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")
        (project / "utils.py").write_text("y = 2\n")
        (project / "app.py").write_text("z = 3\n")
        # No test files

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=90.0, issues=[])
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=50.0,  # 0/3 = 0% < 50%
                tenant_id=TENANT,
                store=store,
            )
            coverage_check = next(c for c in gate_result.gate_checks if c.name == "test_coverage")
            assert coverage_check.passed is False
            assert coverage_check.actual == 0.0
            assert coverage_check.threshold == 50.0
            assert gate_result.passed is False
            # Other checks should still pass
            score_check = next(c for c in gate_result.gate_checks if c.name == "minimum_score")
            assert score_check.passed is True

    @pytest.mark.asyncio
    async def test_gate_inconclusive_is_not_a_failing_f(self, tmp_path):
        """Regression (Defect A): an inconclusive run (no tool completed, -1.0
        sentinel) must surface as inconclusive — cannot pass, but is NOT reported
        as a failing 'F score' and does not violate the ge=0.0 score bound."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=-1.0, issues=[], inconclusive=True)
        store = await _stored(report)
        # A stored inconclusive run is never reused for the gate: it is
        # re-analyzed, and here the re-run is inconclusive again.
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
                analyzer=analyzer,
            )
            assert gate_result.inconclusive is True
            assert gate_result.passed is False
            assert gate_result.grade == "N/A"
            assert gate_result.grade != "F"
            # The -1.0 sentinel must not leak through the ge=0.0-bounded field.
            assert gate_result.overall_score == 0.0
            assert any("inconclusive" in r.lower() for r in gate_result.recommendations)

    @pytest.mark.asyncio
    async def test_gate_failed_score_generates_recommendation(self, tmp_path):
        """When score check fails, recommendation includes the exact score gap."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=70.0, issues=[])
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
            )
            assert gate_result.passed is False
            assert gate_result.grade == "C"
            # Recommendation should mention exact gap: 80.0 - 70.0 = 10.0
            score_rec = [r for r in gate_result.recommendations if "Improve overall score" in r]
            assert len(score_rec) == 1
            assert "10.0" in score_rec[0]
            assert "80.0" in score_rec[0]

    @pytest.mark.asyncio
    async def test_gate_critical_issues_recommendation(self, tmp_path):
        """When critical check fails, recommendation includes count and 'security' wording."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        issues = [_make_issue("critical"), _make_issue("critical")]
        report = _make_quality_report(score=90.0, issues=issues)
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
            )
            assert gate_result.passed is False
            critical_check = next(c for c in gate_result.gate_checks if c.name == "critical_issues")
            assert critical_check.passed is False
            assert critical_check.actual == 2.0
            assert critical_check.threshold == 0.0
            # Recommendation mentions fix count
            crit_rec = [r for r in gate_result.recommendations if "critical" in r.lower()]
            assert len(crit_rec) >= 1
            assert "2" in crit_rec[0]

    @pytest.mark.asyncio
    async def test_gate_high_issues_recommendation(self, tmp_path):
        """When high check fails, recommendation shows excess count."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        issues = [_make_issue("high") for _ in range(5)]
        report = _make_quality_report(score=90.0, issues=issues)
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
            )
            assert gate_result.passed is False
            high_check = next(c for c in gate_result.gate_checks if c.name == "high_issues")
            assert high_check.passed is False
            assert high_check.actual == 5.0
            assert high_check.threshold == 3.0
            # Recommendation should mention excess = 5-3=2
            high_rec = [r for r in gate_result.recommendations if "high" in r.lower()]
            assert len(high_rec) >= 1
            assert "2" in high_rec[0]  # excess amount
            assert "5" in high_rec[0]  # current count
            assert "3" in high_rec[0]  # max allowed

    @pytest.mark.asyncio
    async def test_gate_all_passing_recommendation(self, tmp_path):
        """When all checks pass, recommendation encourages higher thresholds."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=95.0, issues=[])
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
            )
            assert gate_result.passed is True
            assert gate_result.grade == "A"
            assert all(c.passed for c in gate_result.gate_checks)
            # Exactly one recommendation about passing
            assert any("All quality gate checks passed" in r for r in gate_result.recommendations)
            assert any("higher thresholds" in r for r in gate_result.recommendations)

    @pytest.mark.asyncio
    async def test_gate_over_a_failed_run_cannot_pass(self):
        """A stored ``failed`` run (analyzer produced nothing) is a gate that
        cannot pass — it used to be a 500 because the GET re-ran the analyzer."""
        store = await _stored(None)
        # A stored failed run is never reused: the gate re-analyzes, and here the
        # re-run fails again (analyzer produced nothing).
        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=None)
        container_info = MagicMock()
        container_info.codebase_path = "/workspace"
        with patch(_VALIDATE, new_callable=AsyncMock, return_value=container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=None,
                tenant_id=TENANT,
                store=store,
                analyzer=analyzer,
            )
        analyzer.analyze.assert_awaited_once()
        assert gate_result.source == "analyzed"
        assert gate_result.passed is False
        assert gate_result.status == "failed"
        assert gate_result.overall_score == 0.0
        assert [c.name for c in gate_result.gate_checks] == ["analysis_exists"]

    @pytest.mark.asyncio
    async def test_gate_coverage_failing_generates_recommendation(self, tmp_path):
        """When coverage check fails, recommendation mentions exact coverage gap."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")
        (project / "utils.py").write_text("y = 2\n")
        # No test files, so estimated coverage = 0%

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = str(project)

        report = _make_quality_report(score=90.0, issues=[])
        store = await _stored(report)

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            gate_result = await evaluate_quality_gate(
                container_name="test",
                min_score=80.0,
                max_critical=0,
                max_high=3,
                min_coverage=50.0,
                tenant_id=TENANT,
                store=store,
            )
            coverage_recs = [r for r in gate_result.recommendations if "coverage" in r.lower()]
            assert len(coverage_recs) >= 1
            # Gap should be 50.0% (50.0 - 0.0)
            assert "50.0%" in coverage_recs[0]


# ===================================================================
# Quality results endpoint (lines 687-772)
# ===================================================================


class TestGetQualityResults:
    """Test get_analysis_results endpoint handler.

    The implementation looks up task results in Redis by task_id.
    If found, returns the parsed JSON. If not found, returns pending status.
    """

    _GET_REDIS = "app.api.endpoints.quality.get_redis_service"

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_pending(self):
        """Unknown task IDs return pending status when not found in Redis."""
        mock_analyzer = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="bad-format",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"
        assert result["task_id"] == "bad-format"

    @pytest.mark.asyncio
    async def test_redis_miss_returns_pending(self):
        """When task is not in Redis, returns pending status."""
        mock_analyzer = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_x_12345",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"
        assert result["task_id"] == "quality_analysis_x_12345"

    @pytest.mark.asyncio
    async def test_redis_hit_returns_cached_result(self):
        """When task result exists in Redis, returns the parsed JSON."""
        import json

        cached_data = {
            "status": "completed",
            "task_id": "quality_analysis_myrepo_12345",
            "result": {"overall_score": 85.0, "cached": True},
        }
        mock_analyzer = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "completed"
        assert result["task_id"] == "quality_analysis_myrepo_12345"
        assert result["result"]["overall_score"] == 85.0

    @pytest.mark.asyncio
    async def test_redis_hit_empty_result(self):
        """Redis hit with completed status and empty issues."""
        import json

        cached_data = {
            "status": "completed",
            "task_id": "quality_analysis_myrepo_12345",
            "result": {"overall_score": 95.0, "issues": [], "cached": True},
        }
        mock_analyzer = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "completed"
        assert result["result"]["issues"] == []

    @pytest.mark.asyncio
    async def test_redis_hit_with_full_structure(self):
        """Redis hit returns full result structure including score and issues."""
        import json

        cached_data = {
            "status": "completed",
            "task_id": "quality_analysis_myrepo_12345",
            "result": {
                "overall_score": 80.0,
                "grade": "B",
                "issues_count": 1,
                "execution_time": 1.5,
                "cached": False,
            },
        }
        mock_analyzer = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch(self._GET_REDIS, return_value=mock_redis):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "completed"
        assert result["result"]["overall_score"] == 80.0
        assert result["result"]["grade"] == "B"

    @pytest.mark.asyncio
    async def test_redis_unavailable_returns_pending(self):
        """When Redis is unavailable, returns pending status."""
        mock_analyzer = MagicMock()
        with patch(self._GET_REDIS, side_effect=RuntimeError("no redis")):
            result = await get_analysis_results(
                task_id="quality_analysis_myrepo_12345",
                analyzer=mock_analyzer,
                tenant_id="test-tenant",
            )
        assert result["status"] == "pending"
        assert result["task_id"] == "quality_analysis_myrepo_12345"


# ===================================================================
# WebSocket endpoint (lines 1433-1678)
# ===================================================================


@pytest.mark.usefixtures("ws_licence")
class TestQualityWebSocket:
    """Tests for quality_websocket.

    Since coding-client 1.15.0 an uncredentialed WS handshake is REFUSED — a bare
    X-Tenant-Id proves nothing. These exercise the handler's BEHAVIOUR, not the
    auth boundary (tests/unit/test_ws_identity.py owns that), so every handshake
    that expects to be served carries a real licence, exactly as fr's quality
    proxy now does in production.
    """

    @pytest.mark.asyncio
    async def test_no_tenant_id_closes_connection(self):
        """When no tenant_id is provided, WebSocket is accepted then closed with code 1008.

        The handshake must be accepted before close so the 1008 code is actually
        delivered to the client as a WS Close frame (not lost in HTTP 403
        handshake rejection).
        """
        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers({})
        mock_ws.accept = AsyncMock()
        mock_ws.close = AsyncMock()

        await quality_websocket(websocket=mock_ws, container_name="test")
        mock_ws.accept.assert_awaited_once()
        mock_ws.close.assert_called_once()
        close_kwargs = mock_ws.close.call_args[1]
        assert close_kwargs["code"] == 1008
        assert "no verifiable credential" in close_kwargs["reason"]

    @pytest.mark.asyncio
    async def test_tenant_id_from_the_licence(self, ws_licence):
        """The tenant comes from the resolved identity, never from a raw header.

        Was `test_tenant_id_from_header` and asserted a bare `X-Tenant-Id` was
        honoured — which is the hole: through coding-client 1.14.0 this route
        answered `connection_established` for a container the caller held no
        licence for. The bare-header half now lives in test_ws_identity.py as a
        refusal; here the same tenant is reached the legitimate way.
        """
        from starlette.websockets import WebSocketDisconnect

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("header-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info) as mock_val:
            await quality_websocket(websocket=mock_ws, container_name="test")

        # Verify it used the header tenant_id for validation
        mock_val.assert_awaited_with("test", "header-tenant")
        # Connection was accepted
        mock_ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_container_validation_failure_closes_ws(self):
        """When container validation fails, WebSocket is accepted then closed with code 1008.

        Handshake must complete first so the 1008 reaches the client as a WS
        Close frame; without that the consumer (fr-backend) would see a
        generic 1006 abnormal-close and could not distinguish "container
        missing" from "network failure".
        """
        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch(_VALIDATE, new_callable=AsyncMock, side_effect=HTTPException(404, "not found")):
            await quality_websocket(websocket=mock_ws, container_name="test")
            mock_ws.accept.assert_awaited_once()
            mock_ws.close.assert_called_once()
            close_kwargs = mock_ws.close.call_args[1]
            assert close_kwargs["code"] == 1008
            assert "Container validation failed" in close_kwargs["reason"]

    @pytest.mark.asyncio
    async def test_accept_happens_before_validation_rejection(self):
        """Regression guard: accept() must be awaited strictly before close().

        If the order is ever flipped back, Starlette/uvicorn turns the call
        into an HTTP 403 handshake rejection and the 1008 code never reaches
        the wire. This test pins the order.
        """
        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))

        call_order: list[str] = []

        async def record_accept() -> None:
            call_order.append("accept")

        async def record_close(**_: object) -> None:
            call_order.append("close")

        mock_ws.accept = AsyncMock(side_effect=record_accept)
        mock_ws.close = AsyncMock(side_effect=record_close)

        with patch(_VALIDATE, new_callable=AsyncMock, side_effect=HTTPException(404, "missing")):
            await quality_websocket(websocket=mock_ws, container_name="test")

        assert call_order == ["accept", "close"], (
            f"accept() must precede close(); got order: {call_order!r}"
        )

    @pytest.mark.asyncio
    async def test_connection_established_message_sent(self):
        """After accept, connection_established message has correct structure."""
        from starlette.websockets import WebSocketDisconnect

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            await quality_websocket(websocket=mock_ws, container_name="mycontainer")

        mock_ws.accept.assert_awaited_once()

        first_call = mock_ws.send_json.call_args_list[0]
        sent_data = first_call[0][0]
        assert sent_data["type"] == "connection_established"
        assert sent_data["codebase_id"] == "mycontainer"
        assert "timestamp" in sent_data
        assert sent_data["data"]["container_name"] == "mycontainer"
        assert "connection established" in sent_data["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_ping_responds_with_pong(self):
        """Ping message gets a pong response with codebase_id and timestamp."""
        from starlette.websockets import WebSocketDisconnect

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "ping"}
            raise WebSocketDisconnect()

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=mock_receive)
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            await quality_websocket(websocket=mock_ws, container_name="test")

        pong_calls = [c for c in mock_ws.send_json.call_args_list if c[0][0].get("type") == "pong"]
        assert len(pong_calls) == 1
        pong_data = pong_calls[0][0][0]
        assert pong_data["codebase_id"] == "test"
        assert "timestamp" in pong_data

    @pytest.mark.asyncio
    async def test_request_analysis_sends_report(self):
        """request_analysis triggers full flow and sends report with data content."""
        from starlette.websockets import WebSocketDisconnect

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "request_analysis"}
            raise WebSocketDisconnect()

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=mock_receive)
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        high_issue = _make_issue("high")
        # A partial run (ruff+bandit ran, ty errored) still produces a real
        # score — the provenance must ride through both WS frames so a consumer
        # can say "graded on 2 of 3 tools".
        report = _make_quality_report(
            score=85.0,
            issues=[high_issue],
            partial=True,
            tools_attempted=["ruff", "bandit", "ty"],
            tools_succeeded=["ruff", "bandit"],
            tools_failed=["ty"],
        )

        mock_analyzer = MagicMock()
        mock_analyzer.analyze = AsyncMock(return_value=report)

        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info),
            patch("app.api.endpoints.quality.get_quality_analyzer", return_value=mock_analyzer),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA),
            patch(_GET_STORE, return_value=store),
        ):
            await quality_websocket(websocket=mock_ws, container_name="test")

        # The WS run is stored like POST /analyze, keyed by the licence tenant + HEAD.
        assert len(store.save_calls) == 1
        assert store.save_calls[0]["report"] is report
        assert store.save_calls[0]["tenant_id"] == "test-tenant"
        assert store.save_calls[0]["container_name"] == "test"
        assert store.save_calls[0]["commit_sha"] == _SHA
        assert store.rows[0]["status"] == "partial"

        sent_messages = {c[0][0]["type"]: c[0][0] for c in mock_ws.send_json.call_args_list}
        # Verify all expected message types were sent
        assert set(sent_messages.keys()) >= {
            "connection_established",
            "analysis_started",
            "quality_report",
            "quality_gate_result",
            "analysis_completed",
        }

        # Validate quality_report data content
        qr_data = sent_messages["quality_report"]["data"]
        assert qr_data["overall_score"] == 85.0
        assert qr_data["grade"] == "B"
        assert qr_data["total_issues"] == 1
        assert qr_data["severity_distribution"]["high"] == 1
        assert qr_data["severity_distribution"]["critical"] == 0
        assert len(qr_data["issues"]) == 1
        assert qr_data["issues"][0]["severity"] == "high"
        # Per-tool provenance rides through the quality_report frame.
        assert qr_data["partial"] is True
        assert qr_data["tools_attempted"] == ["ruff", "bandit", "ty"]
        assert qr_data["tools_succeeded"] == ["ruff", "bandit"]
        assert qr_data["tools_failed"] == ["ty"]

        # Validate quality_gate_result data content
        gate_data = sent_messages["quality_gate_result"]["data"]
        assert gate_data["passed"] is True  # score 85 >= 80, 0 crit <= 0, 1 high <= 3
        assert gate_data["overall_score"] == 85.0
        assert gate_data["grade"] == "B"
        assert len(gate_data["gate_checks"]) == 3
        # ...and through the quality_gate_result frame.
        assert gate_data["partial"] is True
        assert gate_data["tools_attempted"] == ["ruff", "bandit", "ty"]
        assert gate_data["tools_succeeded"] == ["ruff", "bandit"]
        assert gate_data["tools_failed"] == ["ty"]

        # Validate analysis_completed
        assert "completed successfully" in sent_messages["analysis_completed"]["data"]["message"]

    @pytest.mark.asyncio
    async def test_request_analysis_persist_failure_is_not_fatal(self):
        """A store failure must not turn a completed WS analysis into a failure."""
        from starlette.websockets import WebSocketDisconnect

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "request_analysis"}
            raise WebSocketDisconnect()

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=mock_receive)
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"
        mock_analyzer = MagicMock()
        mock_analyzer.analyze = AsyncMock(return_value=_make_quality_report(score=90.0))
        store = FakeQualityResultStore()
        store.fail_saves = RuntimeError("db down")

        with (
            patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info),
            patch("app.api.endpoints.quality.get_quality_analyzer", return_value=mock_analyzer),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA),
            patch(_GET_STORE, return_value=store),
        ):
            await quality_websocket(websocket=mock_ws, container_name="test")

        assert len(store.save_calls) == 1
        types = [c[0][0]["type"] for c in mock_ws.send_json.call_args_list]
        assert "analysis_completed" in types
        assert "analysis_failed" not in types

    @pytest.mark.asyncio
    async def test_analysis_error_sends_failure(self):
        """When analysis raises, analysis_failed message includes the error string."""
        from starlette.websockets import WebSocketDisconnect

        call_count = 0

        async def mock_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "request_analysis"}
            raise WebSocketDisconnect()

        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=mock_receive)
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        with (
            patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info),
            patch(
                "app.api.endpoints.quality.get_quality_analyzer",
                return_value=MagicMock(
                    analyze=AsyncMock(side_effect=RuntimeError("analysis broke"))
                ),
            ),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=_SHA),
        ):
            await quality_websocket(websocket=mock_ws, container_name="test")

        sent_messages = {c[0][0]["type"]: c[0][0] for c in mock_ws.send_json.call_args_list}
        assert "analysis_failed" in sent_messages
        assert "analysis_started" in sent_messages
        # Verify the error detail is included
        assert sent_messages["analysis_failed"]["data"]["error"] == "analysis broke"
        assert sent_messages["analysis_failed"]["codebase_id"] == "test"

    @pytest.mark.asyncio
    async def test_send_connection_established_failure_closes(self):
        """When send_json fails for connection_established, WS is closed after accept."""
        mock_ws = MagicMock()
        mock_ws.query_params = {}
        mock_ws.headers = Headers(ws_licence_headers("test-tenant"))
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=RuntimeError("send failed"))
        mock_ws.close = AsyncMock()

        mock_container_info = MagicMock()
        mock_container_info.codebase_path = "/workspace"

        with patch(_VALIDATE, new_callable=AsyncMock, return_value=mock_container_info):
            await quality_websocket(websocket=mock_ws, container_name="test")
            # Accept should have been called before the failure
            mock_ws.accept.assert_awaited_once()
            mock_ws.close.assert_called()
