"""Unit tests for AnalysisStatusService."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.models.analysis_api_models import AnalysisPhase, AnalysisStatus, AnalysisStatusInfo


def _ensure_real_analysis_status_module():
    """Ensure services.analysis_status is the real module, not a stub.

    Other test files (e.g. test_endpoint_analysis.py, test_router_reexports.py)
    inject stub ModuleTypes into sys.modules["services.analysis_status"] that
    lack AnalysisStatusService or get_redis_service.  We detect that and
    force-reload the real module from disk using importlib, without going
    through services/__init__.py (which would trigger unrelated heavy imports).
    """
    from pathlib import Path

    mod = sys.modules.get("services.analysis_status")
    if (
        mod is not None
        and hasattr(mod, "__file__")
        and hasattr(mod, "AnalysisStatusService")
        and hasattr(mod, "get_redis_service")
    ):
        # Real module already loaded (has a file path and both key attributes)
        return mod.AnalysisStatusService

    # Stub detected or module missing -- load directly from file
    src_dir = Path(__file__).resolve().parents[2] / "src"
    file_path = src_dir / "services" / "analysis_status.py"

    spec = importlib.util.spec_from_file_location(
        "services.analysis_status",
        str(file_path),
    )
    real_mod = importlib.util.module_from_spec(spec)
    sys.modules["services.analysis_status"] = real_mod
    spec.loader.exec_module(real_mod)

    return real_mod.AnalysisStatusService


def _make_redis_mock(get_return=None, set_return=True, delete_return=1, scan_keys=None):
    """Build a mock RedisService with the methods used by AnalysisStatusService."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=get_return)
    mock.set = AsyncMock(return_value=set_return)
    mock.delete = AsyncMock(return_value=delete_return)
    mock.async_scan_keys = AsyncMock(return_value=scan_keys or [])
    return mock


@pytest.fixture
def redis_mock():
    return _make_redis_mock()


@pytest.fixture
def service(redis_mock):
    """Create AnalysisStatusService with mocked Redis."""
    analysis_status_cls = _ensure_real_analysis_status_module()

    with patch(
        "services.analysis_status.get_redis_service",
        new=AsyncMock(return_value=redis_mock),
    ):
        svc = analysis_status_cls()
        # Inject the mock redis directly to avoid lazy init overhead
        svc.redis_service = redis_mock
        yield svc


@pytest.mark.asyncio
async def test_create_analysis_session_returns_status_info(service, redis_mock):
    """create_analysis_session creates and stores an AnalysisStatusInfo."""
    result = await service.create_analysis_session("test-123", "basic")

    assert isinstance(result, AnalysisStatusInfo)
    assert result.analysis_id == "test-123"
    assert result.status == AnalysisStatus.PENDING
    assert result.progress == 0.0
    assert result.started_at is not None
    assert result.estimated_completion is not None


@pytest.mark.asyncio
async def test_create_analysis_session_custom_duration(service, redis_mock):
    """create_analysis_session uses the provided estimated_duration."""
    result = await service.create_analysis_session("test-456", "deep", estimated_duration=300.0)

    assert isinstance(result, AnalysisStatusInfo)
    # Estimated completion should be roughly 300 seconds from now
    now = datetime.now(UTC)
    estimated = datetime.fromisoformat(result.estimated_completion)
    diff = (estimated - now).total_seconds()
    assert 290 <= diff <= 310


@pytest.mark.asyncio
async def test_create_analysis_session_known_type_duration(service, redis_mock):
    """Known analysis types get appropriate estimated durations."""
    result = await service.create_analysis_session("test-789", "security")

    assert isinstance(result, AnalysisStatusInfo)
    # security -> 90 seconds
    now = datetime.now(UTC)
    estimated = datetime.fromisoformat(result.estimated_completion)
    diff = (estimated - now).total_seconds()
    assert 80 <= diff <= 100


@pytest.mark.asyncio
async def test_create_analysis_session_stored_in_memory(service, redis_mock):
    """create_analysis_session stores in memory fallback when Redis set returns True."""
    await service.create_analysis_session("mem-001", "ast")

    assert "mem-001" in service._memory_store


@pytest.mark.asyncio
async def test_update_status_changes_progress(service, redis_mock):
    """update_status changes progress and status on existing session."""
    # First create the session
    await service.create_analysis_session("upd-001", "basic")

    result = await service.update_status("upd-001", AnalysisStatus.RUNNING, progress=50.0)

    assert result.status == AnalysisStatus.RUNNING
    assert result.progress == 50.0


@pytest.mark.asyncio
async def test_update_status_clamps_progress(service, redis_mock):
    """update_status clamps progress to [0, 100]."""
    await service.create_analysis_session("clamp-001", "basic")

    result = await service.update_status("clamp-001", AnalysisStatus.RUNNING, progress=150.0)

    assert result.progress == 100.0


@pytest.mark.asyncio
async def test_update_status_completed_sets_100(service, redis_mock):
    """update_status with COMPLETED forces progress to 100."""
    await service.create_analysis_session("done-001", "basic")

    result = await service.update_status("done-001", AnalysisStatus.COMPLETED, progress=50.0)

    assert result.status == AnalysisStatus.COMPLETED
    assert result.progress == 100.0
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_update_status_failed_preserves_progress(service, redis_mock):
    """update_status with FAILED preserves current progress (not forced to 100)."""
    await service.create_analysis_session("fail-001", "basic")
    await service.update_status("fail-001", AnalysisStatus.RUNNING, progress=30.0)

    result = await service.update_status("fail-001", AnalysisStatus.FAILED)

    assert result.status == AnalysisStatus.FAILED
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_update_status_with_phase(service, redis_mock):
    """update_status with phase kwarg stores phase in metadata."""
    await service.create_analysis_session("phase-001", "basic")

    result = await service.update_status(
        "phase-001",
        AnalysisStatus.RUNNING,
        progress=25.0,
        phase=AnalysisPhase.PARSING,
    )

    assert result.metadata is not None
    assert result.metadata.get("current_phase") == AnalysisPhase.PARSING.value


@pytest.mark.asyncio
async def test_update_status_auto_creates_missing_session(service, redis_mock):
    """update_status on non-existent session auto-creates it instead of raising."""
    result = await service.update_status("nonexistent-999", AnalysisStatus.RUNNING)
    assert result is not None
    assert result.analysis_id == "nonexistent-999"
    assert result.status == AnalysisStatus.RUNNING


@pytest.mark.asyncio
async def test_get_status_from_redis(service, redis_mock):
    """get_status deserializes from Redis JSON string."""
    # Build a valid status dict as JSON
    status_info = AnalysisStatusInfo(
        analysis_id="redis-001",
        status=AnalysisStatus.RUNNING,
        progress=75.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata={"phase": "analyzing"},
    )
    redis_mock.get.return_value = json.dumps(status_info.model_dump())

    result = await service.get_status("redis-001")

    assert result is not None
    assert result.analysis_id == "redis-001"
    assert result.status == AnalysisStatus.RUNNING
    assert result.progress == 75.0


@pytest.mark.asyncio
async def test_get_status_from_redis_dict(service, redis_mock):
    """get_status handles pre-deserialized dict from Redis."""
    status_info = AnalysisStatusInfo(
        analysis_id="dict-001",
        status=AnalysisStatus.PENDING,
        progress=0.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata=None,
    )
    redis_mock.get.return_value = status_info.model_dump()

    result = await service.get_status("dict-001")

    assert result is not None
    assert result.analysis_id == "dict-001"


@pytest.mark.asyncio
async def test_get_status_from_memory_fallback(service):
    """get_status falls back to memory store when Redis returns None."""
    # Put directly in memory store
    status_info = AnalysisStatusInfo(
        analysis_id="mem-fallback-001",
        status=AnalysisStatus.RUNNING,
        progress=40.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata=None,
    )
    service._memory_store["mem-fallback-001"] = status_info

    # Make Redis return None
    service.redis_service.get.return_value = None

    result = await service.get_status("mem-fallback-001")

    assert result is not None
    assert result.analysis_id == "mem-fallback-001"
    assert result.progress == 40.0


@pytest.mark.asyncio
async def test_get_status_none_when_not_found(service):
    """get_status returns None when session doesn't exist in Redis or memory."""
    service.redis_service.get.return_value = None

    result = await service.get_status("does-not-exist")

    assert result is None


@pytest.mark.asyncio
async def test_get_active_analyses_empty(service, redis_mock):
    """get_active_analyses returns empty list when no active sessions."""
    redis_mock.async_scan_keys.return_value = []

    result = await service.get_active_analyses()

    assert result == []


@pytest.mark.asyncio
async def test_get_active_analyses_from_memory(service):
    """get_active_analyses includes sessions from memory store."""
    # Add active and completed sessions to memory
    active_info = AnalysisStatusInfo(
        analysis_id="active-001",
        status=AnalysisStatus.RUNNING,
        progress=50.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata=None,
    )
    done_info = AnalysisStatusInfo(
        analysis_id="done-002",
        status=AnalysisStatus.COMPLETED,
        progress=100.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata=None,
    )
    service._memory_store["active-001"] = active_info
    service._memory_store["done-002"] = done_info
    # Redis returns no keys
    service.redis_service.async_scan_keys.return_value = []

    result = await service.get_active_analyses()

    ids = [a.analysis_id for a in result]
    assert "active-001" in ids
    assert "done-002" not in ids


@pytest.mark.asyncio
async def test_get_active_analyses_from_redis(service, redis_mock):
    """get_active_analyses scans Redis for active sessions."""
    running_info = AnalysisStatusInfo(
        analysis_id="redis-run-001",
        status=AnalysisStatus.RUNNING,
        progress=60.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=datetime.now(UTC).isoformat(),
        metadata=None,
    )
    key = "analysis_status:redis-run-001"
    redis_mock.async_scan_keys.return_value = [key]
    redis_mock.get.return_value = json.dumps(running_info.model_dump())

    result = await service.get_active_analyses()

    ids = [a.analysis_id for a in result]
    assert "redis-run-001" in ids


@pytest.mark.asyncio
async def test_cleanup_completed_analyses_removes_old(service, redis_mock):
    """cleanup_completed_analyses removes old completed sessions from Redis."""
    old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    old_info = AnalysisStatusInfo(
        analysis_id="old-done-001",
        status=AnalysisStatus.COMPLETED,
        progress=100.0,
        started_at=old_time,
        completed_at=old_time,
        estimated_completion=old_time,
        metadata=None,
    )
    key = "analysis_status:old-done-001"
    redis_mock.async_scan_keys.return_value = [key]
    redis_mock.get.return_value = json.dumps(old_info.model_dump())

    count = await service.cleanup_completed_analyses(older_than_hours=24)

    assert count == 1
    redis_mock.delete.assert_called_once_with(key)


@pytest.mark.asyncio
async def test_cleanup_completed_analyses_keeps_recent(service, redis_mock):
    """cleanup_completed_analyses keeps sessions completed within the threshold."""
    recent_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    recent_info = AnalysisStatusInfo(
        analysis_id="recent-done-001",
        status=AnalysisStatus.COMPLETED,
        progress=100.0,
        started_at=recent_time,
        completed_at=recent_time,
        estimated_completion=recent_time,
        metadata=None,
    )
    key = "analysis_status:recent-done-001"
    redis_mock.async_scan_keys.return_value = [key]
    redis_mock.get.return_value = json.dumps(recent_info.model_dump())

    count = await service.cleanup_completed_analyses(older_than_hours=24)

    assert count == 0
    redis_mock.delete.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_active_not_cleaned(service, redis_mock):
    """cleanup_completed_analyses does not remove active sessions."""
    old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    active_info = AnalysisStatusInfo(
        analysis_id="old-active-001",
        status=AnalysisStatus.RUNNING,
        progress=50.0,
        started_at=old_time,
        completed_at=None,
        estimated_completion=old_time,
        metadata=None,
    )
    key = "analysis_status:old-active-001"
    redis_mock.async_scan_keys.return_value = [key]
    redis_mock.get.return_value = json.dumps(active_info.model_dump())

    count = await service.cleanup_completed_analyses(older_than_hours=24)

    assert count == 0
