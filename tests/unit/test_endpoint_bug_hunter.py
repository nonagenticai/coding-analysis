"""Unit tests for app.api.endpoints.bug_hunter module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.bug_hunter import (
    _bug_hunt_state,
    _get_or_initialize_hunt_state,
    update_hunt_state,
)
from app.api.models.bug_hunter_api_models import BugHunterAnalysisStatus

# ---------------------------------------------------------------------------
# Helpers to reset module-level state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_hunt_state():
    """Clear the module-level _bug_hunt_state dict before each test."""
    _bug_hunt_state.clear()
    with patch(
        "app.api.endpoints.bug_hunter.get_redis_service",
        side_effect=RuntimeError("no redis in tests"),
    ):
        yield
    _bug_hunt_state.clear()


# ---------------------------------------------------------------------------
# _get_or_initialize_hunt_state
# ---------------------------------------------------------------------------


class TestGetOrInitializeHuntState:
    """Tests for _get_or_initialize_hunt_state."""

    async def test_initializes_with_correct_defaults(self):
        state = await _get_or_initialize_hunt_state("my-container")

        assert state["container_name"] == "my-container"
        assert state["last_hunt_timestamp"] is None
        assert state["bugs_found_count"] == 0
        assert state["bugs_fixed_count"] == 0
        assert state["analysis_status"] == BugHunterAnalysisStatus.IDLE
        assert state["last_error"] is None
        assert state["total_bugs_found"] == 0
        assert state["total_bugs_fixed"] == 0

    async def test_returns_existing_state_on_second_call(self):
        state1 = await _get_or_initialize_hunt_state("my-container")
        state1["bugs_found_count"] = 42

        state2 = await _get_or_initialize_hunt_state("my-container")
        assert state2["bugs_found_count"] == 42
        assert state1 is state2

    async def test_separate_containers_have_separate_state(self):
        state_a = await _get_or_initialize_hunt_state("container-a")
        state_b = await _get_or_initialize_hunt_state("container-b")

        state_a["bugs_found_count"] = 10
        assert state_b["bugs_found_count"] == 0


# ---------------------------------------------------------------------------
# update_hunt_state
# ---------------------------------------------------------------------------


class TestUpdateHuntState:
    """Tests for update_hunt_state."""

    async def test_update_status(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state("test", status=BugHunterAnalysisStatus.HUNTING)

        state = _bug_hunt_state["test"]
        assert state["analysis_status"] == BugHunterAnalysisStatus.HUNTING

    async def test_update_bugs_found(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state("test", bugs_found=5)

        state = _bug_hunt_state["test"]
        assert state["bugs_found_count"] == 5
        assert state["total_bugs_found"] == 5

    async def test_update_bugs_fixed(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state("test", bugs_fixed=3)

        state = _bug_hunt_state["test"]
        assert state["bugs_fixed_count"] == 3
        assert state["total_bugs_fixed"] == 3

    async def test_update_with_error_sets_error_status(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state("test", error="Something broke")

        state = _bug_hunt_state["test"]
        assert state["last_error"] == "Something broke"
        assert state["analysis_status"] == BugHunterAnalysisStatus.ERROR

    async def test_update_accumulates_totals(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state("test", bugs_found=3)
        await update_hunt_state("test", bugs_found=2)

        state = _bug_hunt_state["test"]
        # Last call sets bugs_found_count to 2, total should be 3+2=5
        assert state["bugs_found_count"] == 2
        assert state["total_bugs_found"] == 5

    async def test_update_initializes_if_needed(self):
        """update_hunt_state should initialize state for new containers."""
        await update_hunt_state("new-container", status=BugHunterAnalysisStatus.ANALYZING)

        assert "new-container" in _bug_hunt_state
        assert (
            _bug_hunt_state["new-container"]["analysis_status"] == BugHunterAnalysisStatus.ANALYZING
        )

    async def test_update_multiple_fields_at_once(self):
        await _get_or_initialize_hunt_state("test")
        await update_hunt_state(
            "test",
            status=BugHunterAnalysisStatus.COMPLETED,
            bugs_found=10,
            bugs_fixed=8,
        )

        state = _bug_hunt_state["test"]
        assert state["analysis_status"] == BugHunterAnalysisStatus.COMPLETED
        assert state["bugs_found_count"] == 10
        assert state["bugs_fixed_count"] == 8


# ---------------------------------------------------------------------------
# get_bug_hunter_status endpoint
# ---------------------------------------------------------------------------


class TestGetBugHunterStatus:
    """Tests for get_bug_hunter_status endpoint."""

    @patch("app.api.endpoints.bug_hunter.get_container_manager")
    async def test_returns_status_response(self, mock_get_cm):
        mock_container = MagicMock()
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        from app.api.endpoints.bug_hunter import get_bug_hunter_status
        from app.api.models.bug_hunter_api_models import BugHunterStatusResponse

        result = await get_bug_hunter_status(
            container_name="test-container",
            tenant_id="test-tenant",
        )
        assert isinstance(result, BugHunterStatusResponse)
        assert result.container_name == "test-container"
        assert result.analysis_status == BugHunterAnalysisStatus.IDLE

    @patch("app.api.endpoints.bug_hunter.get_container_manager")
    async def test_status_not_found_when_container_missing(self, mock_get_cm):
        mock_cm = MagicMock()
        mock_cm.get_container.side_effect = ValueError("not found")
        mock_get_cm.return_value = mock_cm

        from app.api.endpoints.bug_hunter import get_bug_hunter_status

        with pytest.raises(HTTPException) as exc_info:
            await get_bug_hunter_status(
                container_name="missing",
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# start_bug_hunt endpoint
# ---------------------------------------------------------------------------


class TestStartBugHunt:
    """Tests for start_bug_hunt endpoint."""

    @patch("app.api.endpoints.bug_hunter.get_container_manager")
    async def test_transitions_to_hunting(self, mock_get_cm):
        mock_container = MagicMock()
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=mock_container)
        mock_get_cm.return_value = mock_cm

        from app.api.endpoints.bug_hunter import start_bug_hunt

        result = await start_bug_hunt(
            container_name="test-container",
            request=None,
            tenant_id="test-tenant",
        )
        assert "started" in result.message.lower()

        state = _bug_hunt_state["test-container"]
        assert state["analysis_status"] == BugHunterAnalysisStatus.HUNTING
        assert state["bugs_found_count"] == 0
        assert state["last_error"] is None
        assert state["last_hunt_timestamp"] is not None

    @patch("app.api.endpoints.bug_hunter.get_container_manager")
    async def test_start_hunt_not_found(self, mock_get_cm):
        mock_cm = MagicMock()
        mock_cm.get_container.side_effect = ValueError("not found")
        mock_get_cm.return_value = mock_cm

        from app.api.endpoints.bug_hunter import start_bug_hunt

        with pytest.raises(HTTPException) as exc_info:
            await start_bug_hunt(
                container_name="missing",
                request=None,
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestBugHunterModels:
    """Tests for Bug Hunter Pydantic models."""

    def test_analysis_status_enum_values(self):
        assert BugHunterAnalysisStatus.IDLE == "idle"
        assert BugHunterAnalysisStatus.HUNTING == "hunting"
        assert BugHunterAnalysisStatus.ANALYZING == "analyzing"
        assert BugHunterAnalysisStatus.FIXING == "fixing"
        assert BugHunterAnalysisStatus.COMPLETED == "completed"
        assert BugHunterAnalysisStatus.ERROR == "error"

    def test_status_response_defaults(self):
        from app.api.models.bug_hunter_api_models import BugHunterStatusResponse

        resp = BugHunterStatusResponse(container_name="test")
        assert resp.bugs_found_count == 0
        assert resp.bugs_fixed_count == 0
        assert resp.analysis_status == BugHunterAnalysisStatus.IDLE
        assert resp.last_error is None
        assert resp.last_hunt_timestamp is None

    def test_start_hunt_request_defaults(self):
        from app.api.models.bug_hunter_api_models import BugHunterStartHuntRequest

        req = BugHunterStartHuntRequest()
        assert req.analysis_type == "full"
        assert req.include_patterns is None
        assert req.exclude_patterns is None
        assert req.max_depth is None
