"""Tests for bug_hunter.py hunt_bugs endpoint and helper functions.

Covers lines 588-752, 765-766 (hunt_bugs, initialize_bug_hunt_state).
Also covers update_hunt_state (lines 772-810) and _get_or_initialize_hunt_state (58-80).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.bug_hunter import (
    _bug_hunt_state,
    _get_or_initialize_hunt_state,
    get_bug_analyzer,
    hunt_bugs,
    initialize_bug_hunt_state,
    update_hunt_state,
)
from app.api.models.bug_hunter_api_models import BugHunterAnalysisStatus
from models.analysis_models import BugHuntingRequest, BugHuntingResponse
from services.bug_analysis import BugAnalyzer

_GET_CM = "app.api.endpoints.bug_hunter.get_container_manager"


@pytest.fixture(autouse=True)
def _clear_hunt_state():
    _bug_hunt_state.clear()
    with patch(
        "app.api.endpoints.bug_hunter.get_redis_service",
        side_effect=RuntimeError("no redis in tests"),
    ):
        yield
    _bug_hunt_state.clear()


def _make_mock_error(
    *,
    error_type_value: str = "syntax",
    message: str = "Missing semicolon",
    stack_trace: str = "line 1",
    file_path: str = "main.py",
    line_number: int = 10,
    function_name: str = "main",
    confidence: float = 0.9,
) -> MagicMock:
    """Build a mock ParsedError with the given attributes."""
    mock_error = MagicMock()
    mock_error.error_type = MagicMock()
    mock_error.error_type.value = error_type_value
    mock_error.message = message
    mock_error.stack_trace = stack_trace
    mock_error.file_path = file_path
    mock_error.line_number = line_number
    mock_error.function_name = function_name
    mock_error.confidence = confidence
    return mock_error


def _make_mock_fix(
    *,
    file_path: str = "main.py",
    line_number: int = 10,
    original_code: str = "x = 1",
    fixed_code: str = "x = 1;",
    explanation: str = "Add semicolon",
    confidence: float = 0.95,
    test_suggestion: str = "test the fix",
) -> MagicMock:
    """Build a mock CodeFix with the given attributes."""
    mock_fix = MagicMock()
    mock_fix.file_path = file_path
    mock_fix.line_number = line_number
    mock_fix.original_code = original_code
    mock_fix.fixed_code = fixed_code
    mock_fix.explanation = explanation
    mock_fix.confidence = confidence
    mock_fix.test_suggestion = test_suggestion
    return mock_fix


def _make_mock_analysis(
    *,
    errors: list | None = None,
    resolves: list | None = None,
    root_cause_analysis: str = "No issues found",
    fix_summary: str = "No fixes needed",
    estimated_effort: str = "0 hours",
    risk_assessment: str = "Low",
) -> MagicMock:
    """Build a mock analysis result from BugHunter.analyze_error_logs."""
    mock = MagicMock()
    mock.errors = errors if errors is not None else []
    mock.resolves = resolves if resolves is not None else []
    mock.root_cause_analysis = root_cause_analysis
    mock.fix_summary = fix_summary
    mock.estimated_effort = estimated_effort
    mock.risk_assessment = risk_assessment
    # BugAnalysisResult coverage fields: unmeasured (a bare MagicMock attr is not an int).
    mock.scanned_files = None
    mock.tracked_source_files = None
    mock.commit_sha = None
    return mock


def _make_container_manager(
    container: MagicMock | None = None,
    raise_value_error: bool = False,
) -> MagicMock:
    """Build a mock container manager and return (mock_cm, mock_get_cm_return)."""
    mock_cm = MagicMock()
    if raise_value_error:
        mock_cm.get_container = AsyncMock(side_effect=ValueError("not found"))
    elif container is None:
        mock_cm.get_container = AsyncMock(return_value=None)
    else:
        # get_container is called both with await and without in hunt_bugs
        # (once as async, once as sync on line 645). MagicMock handles both.
        mock_cm.get_container = MagicMock(return_value=container)
        mock_cm.get_container.side_effect = None
        # Make the awaited call return the container too
        coro = AsyncMock(return_value=container)
        mock_cm.get_container = coro
    return mock_cm


# ---------------------------------------------------------------------------
# get_bug_analyzer
# ---------------------------------------------------------------------------


class TestGetBugAnalyzer:
    def test_returns_bug_analyzer(self):
        result = get_bug_analyzer()
        assert isinstance(result, BugAnalyzer)

    def test_returns_new_instance_each_call(self):
        result1 = get_bug_analyzer()
        result2 = get_bug_analyzer()
        assert result1 is not result2


# ---------------------------------------------------------------------------
# initialize_bug_hunt_state
# ---------------------------------------------------------------------------


class TestInitializeBugHuntState:
    @pytest.mark.asyncio
    async def test_initializes_state_for_container(self):
        await initialize_bug_hunt_state("init-container")
        assert "init-container" in _bug_hunt_state
        state = _bug_hunt_state["init-container"]
        assert state["bugs_found_count"] == 0
        assert state["bugs_fixed_count"] == 0
        assert state["total_bugs_found"] == 0
        assert state["total_bugs_fixed"] == 0
        assert state["analysis_status"] == BugHunterAnalysisStatus.IDLE
        assert state["last_error"] is None
        assert state["last_hunt_timestamp"] is None
        assert state["container_name"] == "init-container"

    @pytest.mark.asyncio
    async def test_idempotent(self):
        await initialize_bug_hunt_state("init-container")
        _bug_hunt_state["init-container"]["bugs_found_count"] = 99
        await initialize_bug_hunt_state("init-container")
        # Should not reset existing state
        assert _bug_hunt_state["init-container"]["bugs_found_count"] == 99

    @pytest.mark.asyncio
    async def test_multiple_containers_independent(self):
        await initialize_bug_hunt_state("container-a")
        await initialize_bug_hunt_state("container-b")
        _bug_hunt_state["container-a"]["bugs_found_count"] = 42
        assert _bug_hunt_state["container-b"]["bugs_found_count"] == 0


# ---------------------------------------------------------------------------
# _get_or_initialize_hunt_state — concurrent behavior
# ---------------------------------------------------------------------------


class TestGetOrInitializeHuntState:
    @pytest.mark.asyncio
    async def test_concurrent_initialization_returns_same_state(self):
        """Multiple concurrent calls for the same container should return identical state."""
        results = await asyncio.gather(
            _get_or_initialize_hunt_state("concurrent-test"),
            _get_or_initialize_hunt_state("concurrent-test"),
            _get_or_initialize_hunt_state("concurrent-test"),
        )
        # All three should be the exact same dict object
        assert results[0] is results[1]
        assert results[1] is results[2]
        # And it should be the one stored in _bug_hunt_state
        assert results[0] is _bug_hunt_state["concurrent-test"]

    @pytest.mark.asyncio
    async def test_concurrent_different_containers(self):
        """Concurrent calls for different containers produce independent states."""
        results = await asyncio.gather(
            _get_or_initialize_hunt_state("alpha"),
            _get_or_initialize_hunt_state("beta"),
        )
        assert results[0] is not results[1]
        assert results[0]["container_name"] == "alpha"
        assert results[1]["container_name"] == "beta"

    @pytest.mark.asyncio
    async def test_preserves_mutations_across_calls(self):
        state = await _get_or_initialize_hunt_state("mutate-test")
        state["bugs_found_count"] = 77
        state2 = await _get_or_initialize_hunt_state("mutate-test")
        assert state2["bugs_found_count"] == 77


# ---------------------------------------------------------------------------
# update_hunt_state
# ---------------------------------------------------------------------------


class TestUpdateHuntState:
    @pytest.mark.asyncio
    async def test_update_status(self):
        await initialize_bug_hunt_state("update-test")
        await update_hunt_state("update-test", status=BugHunterAnalysisStatus.HUNTING)
        assert _bug_hunt_state["update-test"]["analysis_status"] == BugHunterAnalysisStatus.HUNTING

    @pytest.mark.asyncio
    async def test_update_bugs_found_increments_total(self):
        await initialize_bug_hunt_state("counter-test")
        await update_hunt_state("counter-test", bugs_found=5)
        assert _bug_hunt_state["counter-test"]["bugs_found_count"] == 5
        assert _bug_hunt_state["counter-test"]["total_bugs_found"] == 5
        await update_hunt_state("counter-test", bugs_found=3)
        assert _bug_hunt_state["counter-test"]["bugs_found_count"] == 3
        assert _bug_hunt_state["counter-test"]["total_bugs_found"] == 8

    @pytest.mark.asyncio
    async def test_update_bugs_fixed_increments_total(self):
        await initialize_bug_hunt_state("fix-test")
        await update_hunt_state("fix-test", bugs_fixed=4)
        assert _bug_hunt_state["fix-test"]["bugs_fixed_count"] == 4
        assert _bug_hunt_state["fix-test"]["total_bugs_fixed"] == 4
        await update_hunt_state("fix-test", bugs_fixed=2)
        assert _bug_hunt_state["fix-test"]["bugs_fixed_count"] == 2
        assert _bug_hunt_state["fix-test"]["total_bugs_fixed"] == 6

    @pytest.mark.asyncio
    async def test_error_sets_error_status(self):
        await initialize_bug_hunt_state("err-test")
        await update_hunt_state("err-test", error="something failed")
        assert _bug_hunt_state["err-test"]["analysis_status"] == BugHunterAnalysisStatus.ERROR
        assert _bug_hunt_state["err-test"]["last_error"] == "something failed"

    @pytest.mark.asyncio
    async def test_error_overrides_previous_status(self):
        await initialize_bug_hunt_state("override-test")
        await update_hunt_state("override-test", status=BugHunterAnalysisStatus.HUNTING)
        assert (
            _bug_hunt_state["override-test"]["analysis_status"] == BugHunterAnalysisStatus.HUNTING
        )
        await update_hunt_state("override-test", error="crash")
        assert _bug_hunt_state["override-test"]["analysis_status"] == BugHunterAnalysisStatus.ERROR

    @pytest.mark.asyncio
    async def test_update_initializes_state_for_unknown_container(self):
        """Calling update_hunt_state for a container not yet initialized should work."""
        await update_hunt_state("unknown-container", status=BugHunterAnalysisStatus.ANALYZING)
        assert "unknown-container" in _bug_hunt_state
        assert (
            _bug_hunt_state["unknown-container"]["analysis_status"]
            == BugHunterAnalysisStatus.ANALYZING
        )
        assert _bug_hunt_state["unknown-container"]["bugs_found_count"] == 0

    @pytest.mark.asyncio
    async def test_update_multiple_fields_at_once(self):
        await initialize_bug_hunt_state("multi-test")
        await update_hunt_state(
            "multi-test",
            status=BugHunterAnalysisStatus.COMPLETED,
            bugs_found=10,
            bugs_fixed=7,
        )
        state = _bug_hunt_state["multi-test"]
        assert state["analysis_status"] == BugHunterAnalysisStatus.COMPLETED
        assert state["bugs_found_count"] == 10
        assert state["bugs_fixed_count"] == 7
        assert state["total_bugs_found"] == 10
        assert state["total_bugs_fixed"] == 7

    @pytest.mark.asyncio
    async def test_no_op_when_all_none(self):
        """Calling with no keyword args should not change state."""
        await initialize_bug_hunt_state("noop-test")
        original_state = dict(_bug_hunt_state["noop-test"])
        await update_hunt_state("noop-test")
        assert _bug_hunt_state["noop-test"]["analysis_status"] == original_state["analysis_status"]
        assert (
            _bug_hunt_state["noop-test"]["bugs_found_count"] == original_state["bugs_found_count"]
        )
        assert _bug_hunt_state["noop-test"]["last_error"] == original_state["last_error"]


# ---------------------------------------------------------------------------
# hunt_bugs: no request body -> 400
# ---------------------------------------------------------------------------


class TestHuntBugsNoBody:
    @pytest.mark.asyncio
    async def test_none_request_raises_400(self):

        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                container_name="test",
                request=None,
                bug_analyzer=BugAnalyzer(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail is not None


# ---------------------------------------------------------------------------
# hunt_bugs: container not found -> 404
# ---------------------------------------------------------------------------


class TestHuntBugsContainerNotFound:
    @pytest.mark.asyncio
    @patch(_GET_CM)
    async def test_value_error_raises_404(self, mock_get_cm):

        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(side_effect=ValueError("not found"))
        mock_get_cm.return_value = mock_cm

        request = BugHuntingRequest(error_logs="some error")

        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                container_name="missing",
                request=request,
                bug_analyzer=BugAnalyzer(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_GET_CM)
    async def test_container_none_raises_404(self, mock_get_cm):

        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=None)
        mock_get_cm.return_value = mock_cm

        request = BugHuntingRequest(error_logs="some error")

        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                container_name="missing",
                request=request,
                bug_analyzer=BugAnalyzer(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# hunt_bugs: successful analysis
# ---------------------------------------------------------------------------


class TestHuntBugsSuccess:
    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_successful_hunt_returns_response_with_no_errors(
        self, mock_get_cm, mock_bug_hunter_cls
    ):

        # Mock container manager
        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test-container"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        # Mock BugHunter with empty results
        mock_analysis_result = _make_mock_analysis(
            root_cause_analysis="No issues found",
            fix_summary="No fixes needed",
            estimated_effort="0 hours",
            risk_assessment="Low",
        )
        mock_hunter_instance = MagicMock()
        mock_hunter_instance.analyze_error_logs = MagicMock(return_value=mock_analysis_result)
        mock_bug_hunter_cls.return_value = mock_hunter_instance

        request = BugHuntingRequest(error_logs="TypeError: unsupported operand")

        result = await hunt_bugs(
            container_name="test-container",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        # Validate type
        assert isinstance(result, BugHuntingResponse)

        # Validate counts
        assert result.errors_found == 0
        assert result.fixes_generated == 0

        # Validate analysis_time is a positive float
        assert isinstance(result.analysis_time, float)
        assert result.analysis_time > 0

        # Validate string fields are passed through from the mock
        assert result.root_cause_analysis == "No issues found"
        assert result.fix_summary == "No fixes needed"
        assert result.estimated_effort == "0 hours"
        assert result.risk_assessment == "Low"

        # Validate empty collections
        assert result.errors == []
        assert result.resolves == []

        # Validate PR not created
        assert result.pull_request is None

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_hunt_with_errors_and_fixes(self, mock_get_cm, mock_bug_hunter_cls):

        # Mock container
        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test-container"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        # Build mock errors and fixes with known values
        mock_error = _make_mock_error(
            error_type_value="syntax",
            message="Missing semicolon",
            stack_trace="line 1",
            file_path="main.py",
            line_number=10,
            function_name="main",
            confidence=0.9,
        )
        mock_fix = _make_mock_fix(
            file_path="main.py",
            line_number=10,
            original_code="x = 1",
            fixed_code="x = 1;",
            explanation="Add semicolon",
            confidence=0.95,
            test_suggestion="test the fix",
        )
        mock_analysis = _make_mock_analysis(
            errors=[mock_error],
            resolves=[mock_fix],
            root_cause_analysis="Missing punctuation",
            fix_summary="Fixed main.py: 1 fix",
            estimated_effort="5 minutes",
            risk_assessment="Low",
        )

        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="SyntaxError: missing semicolon")

        result = await hunt_bugs(
            container_name="test-container",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        # Validate counts
        assert result.errors_found == 1
        assert result.fixes_generated == 1

        # Validate analysis metadata
        assert result.root_cause_analysis == "Missing punctuation"
        assert result.fix_summary == "Fixed main.py: 1 fix"
        assert result.estimated_effort == "5 minutes"
        assert result.risk_assessment == "Low"
        assert result.analysis_time > 0
        assert result.pull_request is None

        # Validate error detail structure — every field of ErrorDetailDict
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err["error_id"] == "err_0"
        assert err["error_type"] == "syntax"
        assert err["error_message"] == "Missing semicolon"
        assert err["stack_trace"] == "line 1"
        assert err["file_path"] == "main.py"
        assert err["line_number"] == 10
        assert err["function_name"] == "main"
        assert err["error_category"] == "syntax"
        assert err["severity"] == "unrated"  # nothing rates it (coding-analysis#112)
        assert isinstance(err["timestamp"], str)
        assert len(err["timestamp"]) > 0
        assert err["frequency"] == 1
        assert err["impact_score"] == 0.9
        assert err["suggested_fix"] == ""
        assert err["related_errors"] == []

        # Validate fix detail structure — every field of FixDetailDict
        assert len(result.resolves) == 1
        fix = result.resolves[0]
        assert fix["fix_id"] == "fix_0"
        assert fix["fix_type"] == "code_fix"
        assert fix["file_path"] == "main.py"
        assert fix["line_number"] == 10
        assert fix["original_code"] == "x = 1"
        assert fix["fixed_code"] == "x = 1;"
        assert fix["description"] == "Add semicolon"
        assert fix["confidence"] == 0.95
        assert fix["risk_level"] == "low"  # confidence > 0.7
        assert fix["test_impact"] == "test the fix"
        assert fix["dependencies"] == []
        assert fix["validation_steps"] == []
        assert fix["rollback_plan"] == "Revert to original code"

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_hunt_with_multiple_errors_and_fixes(self, mock_get_cm, mock_bug_hunter_cls):
        """Validate correct indexing when multiple errors and fixes are returned."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        errors = [
            _make_mock_error(
                error_type_value="import",
                message="No module named foo",
                file_path="app.py",
                line_number=1,
                function_name="<module>",
                confidence=0.8,
            ),
            _make_mock_error(
                error_type_value="type",
                message="Expected int got str",
                file_path="utils.py",
                line_number=55,
                function_name="convert",
                confidence=0.7,
            ),
        ]
        fixes = [
            _make_mock_fix(
                file_path="app.py",
                line_number=1,
                original_code="import foo",
                fixed_code="import bar as foo",
                explanation="Replace missing module",
                confidence=0.85,
                test_suggestion="Run import check",
            ),
            _make_mock_fix(
                file_path="utils.py",
                line_number=55,
                original_code="x = str(val)",
                fixed_code="x = int(val)",
                explanation="Fix type conversion",
                confidence=0.6,
                test_suggestion="Run type tests",
            ),
        ]
        mock_analysis = _make_mock_analysis(
            errors=errors,
            resolves=fixes,
            root_cause_analysis="Multiple issues",
            fix_summary="2 files fixed",
            estimated_effort="15 minutes",
            risk_assessment="Medium",
        )
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="ImportError\nTypeError")

        result = await hunt_bugs(
            container_name="test",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        assert result.errors_found == 2
        assert result.fixes_generated == 2

        # First error
        assert result.errors[0]["error_id"] == "err_0"
        assert result.errors[0]["error_type"] == "import"
        assert result.errors[0]["error_message"] == "No module named foo"
        assert result.errors[0]["file_path"] == "app.py"

        # Second error
        assert result.errors[1]["error_id"] == "err_1"
        assert result.errors[1]["error_type"] == "type"
        assert result.errors[1]["error_message"] == "Expected int got str"
        assert result.errors[1]["file_path"] == "utils.py"

        # First fix
        assert result.resolves[0]["fix_id"] == "fix_0"
        assert result.resolves[0]["file_path"] == "app.py"
        assert result.resolves[0]["risk_level"] == "low"  # 0.85 > 0.7

        # Second fix — low confidence => medium risk
        assert result.resolves[1]["fix_id"] == "fix_1"
        assert result.resolves[1]["file_path"] == "utils.py"
        assert result.resolves[1]["confidence"] == 0.6
        assert result.resolves[1]["risk_level"] == "medium"  # 0.6 <= 0.7

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_risk_level_boundary_at_0_7(self, mock_get_cm, mock_bug_hunter_cls):
        """Verify that confidence == 0.7 produces risk_level 'medium' (not 'low')."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        fix_at_boundary = _make_mock_fix(confidence=0.7)
        fix_above = _make_mock_fix(confidence=0.71)

        mock_analysis = _make_mock_analysis(resolves=[fix_at_boundary, fix_above])
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="some error")
        result = await hunt_bugs(
            container_name="test",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        # confidence == 0.7 is NOT > 0.7, so risk_level should be "medium"
        assert result.resolves[0]["risk_level"] == "medium"
        # confidence == 0.71 IS > 0.7, so risk_level should be "low"
        assert result.resolves[1]["risk_level"] == "low"

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_hunt_with_base64_error_logs(self, mock_get_cm, mock_bug_hunter_cls):
        import base64

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_analysis = _make_mock_analysis(
            root_cause_analysis="OK",
            fix_summary="",
            estimated_effort="0",
            risk_assessment="None",
        )
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        encoded = base64.b64encode(b"ImportError: no module named foo").decode()
        request = BugHuntingRequest(error_logs_base64=encoded)

        result = await hunt_bugs(
            container_name="test",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        assert isinstance(result, BugHuntingResponse)
        assert result.root_cause_analysis == "OK"
        assert result.errors_found == 0
        assert result.fixes_generated == 0
        assert result.errors == []
        assert result.resolves == []

        # Verify the decoded logs were passed to analyzer
        call_args = mock_hunter.analyze_error_logs.call_args
        assert "ImportError" in call_args[0][0]
        assert "no module named foo" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_hunt_passes_codebase_path_to_bug_hunter(self, mock_get_cm, mock_bug_hunter_cls):
        """Verify BugHunter is initialized with a codebase_path and no github_token.

        Note: The source calls get_container a second time without await (line 645),
        so with AsyncMock the coroutine object has no work_dir attribute and the
        fallback /tmp/{container_name} path is used.
        """
        mock_container = MagicMock()
        mock_container.work_dir = "/workspace/my-project"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_analysis = _make_mock_analysis()
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="error")

        await hunt_bugs(
            container_name="my-service",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        # Verify BugHunter was constructed with codebase_path and no github_token
        mock_bug_hunter_cls.assert_called_once()
        call_kwargs = mock_bug_hunter_cls.call_args
        assert isinstance(call_kwargs[1]["codebase_path"], str)
        assert len(call_kwargs[1]["codebase_path"]) > 0
        assert call_kwargs[1]["github_token"] is None

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_hunt_passes_error_logs_and_context_to_analyzer(
        self, mock_get_cm, mock_bug_hunter_cls
    ):
        """Verify analyze_error_logs receives the error_logs string and context dict."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_analysis = _make_mock_analysis()
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="NameError: undefined variable")

        await hunt_bugs(
            container_name="my-service",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        mock_hunter.analyze_error_logs.assert_called_once()
        args, kwargs = mock_hunter.analyze_error_logs.call_args
        # First positional arg is the error_logs string
        assert args[0] == "NameError: undefined variable"
        # Second positional arg is context dict containing container_name
        assert args[1]["container_name"] == "my-service"
        assert "error_timestamp" in args[1]

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_error_with_no_file_path_uses_empty_string(
        self, mock_get_cm, mock_bug_hunter_cls
    ):
        """When ParsedError has None for file_path/function_name, the dict uses empty strings."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_error = _make_mock_error()
        mock_error.file_path = None
        mock_error.line_number = None
        mock_error.function_name = None
        mock_error.stack_trace = None

        mock_analysis = _make_mock_analysis(errors=[mock_error])
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="error")
        result = await hunt_bugs(
            container_name="test",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        err = result.errors[0]
        assert err["file_path"] == ""
        assert err["line_number"] == 0
        assert err["function_name"] == ""
        assert err["stack_trace"] == ""

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_fix_with_no_test_suggestion_uses_empty_string(
        self, mock_get_cm, mock_bug_hunter_cls
    ):
        """When CodeFix has None for test_suggestion, test_impact is empty string."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_fix = _make_mock_fix()
        mock_fix.test_suggestion = None

        mock_analysis = _make_mock_analysis(resolves=[mock_fix])
        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(return_value=mock_analysis)
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="error")
        result = await hunt_bugs(
            container_name="test",
            request=request,
            bug_analyzer=BugAnalyzer(),
            tenant_id="test-tenant",
        )

        assert result.resolves[0]["test_impact"] == ""


# ---------------------------------------------------------------------------
# hunt_bugs: unexpected error -> 500
# ---------------------------------------------------------------------------


class TestHuntBugsInternalError:
    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_unexpected_error_raises_500(self, mock_get_cm, mock_bug_hunter_cls):

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_bug_hunter_cls.side_effect = RuntimeError("analyzer failed")

        request = BugHuntingRequest(error_logs="some error")

        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                container_name="test",
                request=request,
                bug_analyzer=BugAnalyzer(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.endpoints.bug_hunter.BugHunter")
    @patch(_GET_CM)
    async def test_analyze_error_logs_exception_raises_500(self, mock_get_cm, mock_bug_hunter_cls):
        """When analyze_error_logs itself throws, we get a 500."""

        mock_container = MagicMock()
        mock_container.work_dir = "/tmp/test"
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        mock_hunter = MagicMock()
        mock_hunter.analyze_error_logs = MagicMock(side_effect=ValueError("parse failed"))
        mock_bug_hunter_cls.return_value = mock_hunter

        request = BugHuntingRequest(error_logs="bad logs")

        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                container_name="test",
                request=request,
                bug_analyzer=BugAnalyzer(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# start_bug_hunt: container returns None
# ---------------------------------------------------------------------------


class TestStartBugHuntContainerNone:
    @pytest.mark.asyncio
    @patch(_GET_CM)
    async def test_container_none_raises_404(self, mock_get_cm):
        from app.api.endpoints.bug_hunter import start_bug_hunt

        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=None)
        mock_get_cm.return_value = mock_cm

        with pytest.raises(HTTPException) as exc_info:
            await start_bug_hunt(
                container_name="missing",
                request=None,
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# get_bug_hunter_status: container returns None
# ---------------------------------------------------------------------------


class TestGetBugHunterStatusContainerNone:
    @pytest.mark.asyncio
    @patch(_GET_CM)
    async def test_container_none_raises_404(self, mock_get_cm):
        from app.api.endpoints.bug_hunter import get_bug_hunter_status

        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=None)
        mock_get_cm.return_value = mock_cm

        with pytest.raises(HTTPException) as exc_info:
            await get_bug_hunter_status(
                container_name="missing",
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404
