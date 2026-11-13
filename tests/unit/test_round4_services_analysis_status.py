"""Round 4 coverage tests for services/analysis_status.py.

Covers the missing lines: 91-93, 131, 152-155, 179-184, 216-217, 227, 231-236,
259-270, 284-285, 295, 299-304, 325-336, 369-371, 375, 383.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.models.analysis_api_models import AnalysisStatus, AnalysisStatusInfo
from services.analysis_status import (
    AnalysisStatusService,
    _AnalysisStatusServiceHolder,
    get_analysis_status_service,
)


def _make_redis_mock(**overrides):
    mock = MagicMock()
    mock.get = AsyncMock(return_value=overrides.get("get_return", None))
    mock.set = AsyncMock(return_value=overrides.get("set_return", True))
    mock.delete = AsyncMock(return_value=overrides.get("delete_return", 1))
    mock.async_scan_keys = AsyncMock(return_value=overrides.get("scan_keys", []))
    return mock


@pytest.fixture
def service():
    rmock = _make_redis_mock()
    with patch("services.analysis_status.get_redis_service", return_value=rmock):
        svc = AnalysisStatusService()
        svc.redis_service = rmock
        yield svc, rmock


# ── Lines 91-93: create_analysis_session exception path ──


@pytest.mark.asyncio
async def test_create_session_exception_propagates():
    """Cover lines 91-93: exception during session creation."""
    svc = AnalysisStatusService()
    # Patch _store_status to raise so the outer try/except is hit
    svc._store_status = AsyncMock(side_effect=RuntimeError("redis down"))
    with pytest.raises(RuntimeError, match="redis down"):
        await svc.create_analysis_session("fail-id", "basic")


# ── Line 131: metadata is None path ──


@pytest.mark.asyncio
async def test_update_status_with_none_metadata(service):
    """Cover line 131: metadata is None."""
    svc, rmock = service
    await svc.create_analysis_session("meta-none", "basic")
    # Force metadata to None
    svc._memory_store["meta-none"].metadata = None
    rmock.get.return_value = None
    result = await svc.update_status("meta-none", AnalysisStatus.RUNNING, progress=10.0)
    assert result.metadata is not None


# ── Lines 152-155: update_status exception returns fallback ──


@pytest.mark.asyncio
async def test_update_status_exception_returns_fallback(service):
    """Cover lines 152-155: exception returns fallback AnalysisStatusInfo."""
    svc, rmock = service
    # Make get_status raise
    rmock.get = AsyncMock(side_effect=RuntimeError("connection lost"))
    svc._memory_store.clear()
    # Also make _store_status fail so we hit the outer exception
    rmock.set = AsyncMock(side_effect=RuntimeError("still broken"))
    result = await svc.update_status("error-id", AnalysisStatus.RUNNING, progress=25.0)
    assert result.analysis_id == "error-id"
    assert result.status == AnalysisStatus.RUNNING
    assert result.progress == 25.0


# ── Lines 179-184: get_status with unexpected data type ──


@pytest.mark.asyncio
async def test_get_status_unexpected_data_type(service):
    """Cover lines 179-184: Redis returns non-dict, non-str data."""
    svc, rmock = service
    rmock.get.return_value = 12345  # int, unexpected type
    result = await svc.get_status("weird-data")
    assert result is None


# ── Lines 216-217: get_active_analyses scan_keys exception ──


@pytest.mark.asyncio
async def test_get_active_analyses_scan_keys_exception(service):
    """Cover lines 216-217: scan_keys raises exception."""
    svc, rmock = service
    rmock.async_scan_keys = AsyncMock(side_effect=RuntimeError("scan failed"))
    # Add an active one to memory so we get a result
    active = AnalysisStatusInfo(
        analysis_id="mem-act",
        status=AnalysisStatus.RUNNING,
        progress=50.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=None,
        metadata=None,
    )
    svc._memory_store["mem-act"] = active
    result = await svc.get_active_analyses()
    assert len(result) >= 1
    assert any(a.analysis_id == "mem-act" for a in result)


# ── Lines 227, 231-236: get_active_analyses Redis returns unexpected type ──


@pytest.mark.asyncio
async def test_get_active_analyses_unexpected_data_type(service):
    """Cover lines 227 & 231-236: Redis data of unexpected type in get_active_analyses."""
    svc, rmock = service
    rmock.async_scan_keys.return_value = ["analysis_status:weird"]
    rmock.get.return_value = 99999  # not dict, not str
    result = await svc.get_active_analyses()
    assert isinstance(result, list)


# ── Lines 259-270: get_active_analyses exception path ──


@pytest.mark.asyncio
async def test_get_active_analyses_json_decode_error(service):
    """Cover lines 259-270: JSON decode error falls back to memory store."""
    svc, rmock = service
    rmock.async_scan_keys.return_value = ["analysis_status:bad"]
    rmock.get.return_value = "not-valid-json{{"
    active = AnalysisStatusInfo(
        analysis_id="mem-active",
        status=AnalysisStatus.RUNNING,
        progress=30.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=None,
        metadata=None,
    )
    svc._memory_store["mem-active"] = active
    result = await svc.get_active_analyses()
    assert any(a.analysis_id == "mem-active" for a in result)


# ── Lines 284-285: cleanup scan_keys exception ──


@pytest.mark.asyncio
async def test_cleanup_scan_keys_exception(service):
    """Cover lines 284-285: scan_keys raises during cleanup."""
    svc, rmock = service
    rmock.async_scan_keys = AsyncMock(side_effect=RuntimeError("scan broken"))
    result = await svc.cleanup_completed_analyses()
    assert result == 0  # cleanup proceeds with empty keys list


# ── Lines 295, 299-304: cleanup unexpected data type ──


@pytest.mark.asyncio
async def test_cleanup_unexpected_data_type(service):
    """Cover lines 299-304: unexpected data type during cleanup."""
    svc, rmock = service
    rmock.async_scan_keys.return_value = ["analysis_status:u"]
    rmock.get.return_value = 42  # not dict, not str
    result = await svc.cleanup_completed_analyses()
    assert result == 0


@pytest.mark.asyncio
async def test_cleanup_dict_data(service):
    """Cover line 295: data is dict type during cleanup."""
    svc, rmock = service
    old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    status_dict = AnalysisStatusInfo(
        analysis_id="old",
        status=AnalysisStatus.COMPLETED,
        progress=100.0,
        started_at=old_time,
        completed_at=old_time,
        estimated_completion=old_time,
        metadata=None,
    ).model_dump()
    rmock.async_scan_keys.return_value = ["analysis_status:old"]
    rmock.get.return_value = status_dict  # dict, not JSON string
    result = await svc.cleanup_completed_analyses(older_than_hours=24)
    assert result == 1


# ── Lines 325-336: cleanup exception path ──


@pytest.mark.asyncio
async def test_cleanup_json_decode_error(service):
    """Cover lines 325-336: JSON parse error during cleanup returns 0."""
    svc, rmock = service
    rmock.async_scan_keys.return_value = ["analysis_status:bad"]
    rmock.get.return_value = "{bad json"
    result = await svc.cleanup_completed_analyses()
    assert result == 0


# ── Lines 369-371, 375: singleton holder ──


def test_service_holder_get_instance():
    """Cover lines 369-371: _AnalysisStatusServiceHolder.get_instance."""
    holder = _AnalysisStatusServiceHolder()
    instance = holder.get_instance()
    assert isinstance(instance, AnalysisStatusService)
    # Second call returns same instance
    assert holder.get_instance() is instance


def test_service_holder_reset():
    """Cover line 375: _AnalysisStatusServiceHolder.reset_instance."""
    holder = _AnalysisStatusServiceHolder()
    holder.get_instance()
    holder.reset_instance()
    assert holder._instance is None


# ── Line 383: get_analysis_status_service ──


def test_get_analysis_status_service():
    """Cover line 383: module-level get_analysis_status_service function."""
    svc = get_analysis_status_service()
    assert isinstance(svc, AnalysisStatusService)


# ── update_status with details kwarg ──


@pytest.mark.asyncio
async def test_update_status_with_details(service):
    """Cover line 138-139: update_status with details kwarg."""
    svc, rmock = service
    await svc.create_analysis_session("det-001", "basic")
    result = await svc.update_status(
        "det-001",
        AnalysisStatus.RUNNING,
        progress=40.0,
        details={"files_analyzed": 10},
    )
    assert result.metadata["files_analyzed"] == 10


# ── Store status Redis failure path ──


@pytest.mark.asyncio
async def test_store_status_redis_failure(service):
    """Cover line 350-352: Redis unavailable during store falls back to memory."""
    svc, rmock = service
    rmock.set = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    # create_analysis_session calls _store_status internally
    result = await svc.create_analysis_session("store-fail", "basic")
    assert result.analysis_id == "store-fail"
    assert "store-fail" in svc._memory_store


# ── cleanup with failed status ──


@pytest.mark.asyncio
async def test_cleanup_removes_failed_old_analyses(service):
    """Cover line 310: cleanup includes FAILED status."""
    svc, rmock = service
    old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    info = AnalysisStatusInfo(
        analysis_id="old-fail",
        status=AnalysisStatus.FAILED,
        progress=30.0,
        started_at=old_time,
        completed_at=old_time,
        estimated_completion=old_time,
        metadata=None,
    )
    rmock.async_scan_keys.return_value = ["analysis_status:old-fail"]
    rmock.get.return_value = json.dumps(info.model_dump())
    count = await svc.cleanup_completed_analyses(older_than_hours=24)
    assert count == 1


# ── get_active_analyses dedup ──


@pytest.mark.asyncio
async def test_get_active_analyses_dedup(service):
    """Cover lines 253-255: dedup between Redis and memory store."""
    svc, rmock = service
    info = AnalysisStatusInfo(
        analysis_id="dup-001",
        status=AnalysisStatus.RUNNING,
        progress=50.0,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        estimated_completion=None,
        metadata=None,
    )
    # Present in both Redis and memory
    rmock.async_scan_keys.return_value = ["analysis_status:dup-001"]
    rmock.get.return_value = json.dumps(info.model_dump())
    svc._memory_store["dup-001"] = info
    result = await svc.get_active_analyses()
    ids = [a.analysis_id for a in result]
    assert ids.count("dup-001") == 1
