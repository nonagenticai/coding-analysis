"""Unit tests for batch analysis endpoints."""

from __future__ import annotations

import pathlib
import sys
import types as _types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# sys.modules surgery
# ---------------------------------------------------------------------------

_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"

for _key in list(sys.modules):
    if _key in ("app.api.endpoints.analysis_modules.batch_analysis",):
        del sys.modules[_key]

# Ensure app packages exist with real __path__
for _pkg, _subpath in (
    ("app", "app"),
    ("app.api", "app/api"),
    ("app.api.endpoints", "app/api/endpoints"),
    ("app.api.endpoints.analysis_modules", "app/api/endpoints/analysis_modules"),
    ("app.api.models", "app/api/models"),
    ("app.core", "app/core"),
):
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = [str(_src_dir / _subpath)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m
    elif not getattr(sys.modules[_pkg], "__path__", None):
        sys.modules[_pkg].__path__ = [str(_src_dir / _subpath)]

# Stub services
if "services" not in sys.modules:
    _svc = _types.ModuleType("services")
    _svc.__path__ = [str(_src_dir / "services")]
    sys.modules["services"] = _svc
_svc_stubs = {
    "services.code_graph_metrics": {"CodeGraphMetricsService": MagicMock},
    "services.github_provisioning_service": {"get_github_provisioning_service": MagicMock()},
}
for _mod_name, _attrs in _svc_stubs.items():
    if _mod_name not in sys.modules:
        _s = _types.ModuleType(_mod_name)
        for _a, _v in _attrs.items():
            setattr(_s, _a, _v)
        sys.modules[_mod_name] = _s


# Stub uuid_v7 — uuid7() must return an object with .hex attribute
class _FakeUUID:
    hex = "deadbeef12345678"


if "uuid_v7" not in sys.modules:
    _uv7 = _types.ModuleType("uuid_v7")
    _uv7.__path__ = ["__uuid_v7_stub__"]
    sys.modules["uuid_v7"] = _uv7
if "uuid_v7.base" not in sys.modules:
    _uv7b = _types.ModuleType("uuid_v7.base")
    _uv7b.uuid7 = lambda: _FakeUUID()
    sys.modules["uuid_v7.base"] = _uv7b
else:
    sys.modules["uuid_v7.base"].uuid7 = lambda: _FakeUUID()

# Stub processing.multi_language_factory
if "processing.multi_language_factory" not in sys.modules:
    _mlf_stub = _types.ModuleType("processing.multi_language_factory")
    _mlf_stub.get_processor_for_language = AsyncMock(return_value=MagicMock())
    sys.modules["processing.multi_language_factory"] = _mlf_stub

# Stub app.core.dependencies
if "app.core.dependencies" not in sys.modules:
    _dep_stub = _types.ModuleType("app.core.dependencies")
    _dep_stub.get_multi_language_factory = MagicMock(return_value=MagicMock())
    sys.modules["app.core.dependencies"] = _dep_stub

# Stub analysis_utils
if "app.api.endpoints.analysis_modules.analysis_utils" not in sys.modules:
    _au_stub = _types.ModuleType("app.api.endpoints.analysis_modules.analysis_utils")
    _au_stub.resolve_container_files = AsyncMock(return_value=["/app/data/repo/main.py"])
    sys.modules["app.api.endpoints.analysis_modules.analysis_utils"] = _au_stub
else:
    _au_stub = sys.modules["app.api.endpoints.analysis_modules.analysis_utils"]
    if not hasattr(_au_stub, "resolve_container_files"):
        _au_stub.resolve_container_files = AsyncMock(return_value=["/app/data/repo/main.py"])

# Now import
from app.api.endpoints.analysis_modules.batch_analysis import (  # noqa: E402
    analyze_files_batch,
)
from app.api.models.analysis_api_models import BatchAnalysisRequest  # noqa: E402

_RESOLVE_FILES = "app.api.endpoints.analysis_modules.batch_analysis.resolve_container_files"
_PROCESS_CONCURRENT = "app.api.endpoints.analysis_modules.batch_analysis.process_files_concurrent"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchAnalysisEndpoint:
    """Tests for the batch analysis endpoint."""

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_all_files_batch_success(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/a.py", "/app/data/repo/b.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/app/data/repo/a.py",
                "success": True,
                "execution_time": 0.1,
                "entities_count": 5,
                "relationships_count": 2,
                "language": "python",
            },
            {
                "index": 1,
                "path": "/app/data/repo/b.py",
                "success": True,
                "execution_time": 0.2,
                "entities_count": 3,
                "relationships_count": 1,
                "language": "python",
            },
        ]

        request = BatchAnalysisRequest(container_name="my-project", analysis_type="augmented")
        response = await analyze_files_batch(request=request, tenant_id="t")

        assert response["batch_id"]
        assert response["container_name"] == "my-project"
        assert response["batch_summary"]["total_files"] == 2
        assert response["batch_summary"]["successful_files"] == 2
        assert response["batch_summary"]["failed_files"] == 0

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_explicit_file_list_success(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/x.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/app/data/repo/x.py",
                "success": True,
                "execution_time": 0.05,
                "entities_count": 2,
                "relationships_count": 0,
                "language": "python",
            },
        ]

        request = BatchAnalysisRequest(
            container_name="proj",
            file_paths=["/app/data/repo/x.py"],
            analysis_type="basic",
        )
        response = await analyze_files_batch(request=request, tenant_id="t")
        assert response["batch_summary"]["total_files"] == 1

    @patch(_RESOLVE_FILES)
    async def test_no_code_files_found_returns_404(self, mock_resolve):
        mock_resolve.return_value = []

        request = BatchAnalysisRequest(container_name="empty-project", analysis_type="basic")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_files_batch(request=request, tenant_id="t")
        assert exc_info.value.status_code == 404

    async def test_invalid_analysis_request_returns_400(self):
        """A ValueError raised during processing should yield 400."""
        with patch(_RESOLVE_FILES, side_effect=ValueError("missing code")):
            request = BatchAnalysisRequest(container_name="bad", analysis_type="x")
            with pytest.raises(HTTPException) as exc_info:
                await analyze_files_batch(request=request, tenant_id="t")
            assert exc_info.value.status_code == 400

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_processor_timeout_branch(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/slow.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/app/data/repo/slow.py",
                "success": False,
                "error": "timeout",
                "execution_time": 30,
            },
        ]

        request = BatchAnalysisRequest(container_name="proj", analysis_type="augmented")
        response = await analyze_files_batch(request=request, tenant_id="t")
        assert response["batch_summary"]["failed_files"] == 1
        assert response["batch_summary"]["successful_files"] == 0

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_aggregation_enabled_vs_disabled(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/a.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/app/data/repo/a.py",
                "success": True,
                "execution_time": 0.1,
                "entities_count": 3,
                "relationships_count": 1,
                "language": "python",
                "lines_of_code": 100,
                "complexity_score": 2.0,
                "parser_used": "treesitter",
            },
        ]

        # Aggregation enabled (default)
        request = BatchAnalysisRequest(
            container_name="proj",
            analysis_type="augmented",
            batch_options={"aggregate_results": True},
        )
        response = await analyze_files_batch(request=request, tenant_id="t")
        assert response["aggregated_results"] is not None

        # Aggregation disabled
        request2 = BatchAnalysisRequest(
            container_name="proj",
            analysis_type="augmented",
            batch_options={"aggregate_results": False},
        )
        response2 = await analyze_files_batch(request=request2, tenant_id="t")
        assert response2["aggregated_results"] is None

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_per_file_mixed_success_failure_shape(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/a.py", "/app/data/repo/b.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/app/data/repo/a.py",
                "success": True,
                "execution_time": 0.1,
                "entities_count": 5,
                "relationships_count": 2,
                "language": "python",
            },
            {
                "index": 1,
                "path": "/app/data/repo/b.py",
                "success": False,
                "error": "parse error",
                "execution_time": 0,
            },
        ]

        request = BatchAnalysisRequest(container_name="proj", analysis_type="augmented")
        response = await analyze_files_batch(request=request, tenant_id="t")
        assert response["batch_summary"]["successful_files"] == 1
        assert response["batch_summary"]["failed_files"] == 1
        assert response["batch_summary"]["success_rate"] == 0.5

    @patch(_PROCESS_CONCURRENT)
    @patch(_RESOLVE_FILES)
    async def test_performance_metrics_populated(self, mock_resolve, mock_process):
        mock_resolve.return_value = ["/app/data/repo/a.py", "/app/data/repo/b.py"]
        mock_process.return_value = [
            {
                "index": 0,
                "path": "/a.py",
                "success": True,
                "execution_time": 0.1,
                "entities_count": 5,
                "relationships_count": 2,
                "language": "python",
            },
            {
                "index": 1,
                "path": "/b.py",
                "success": True,
                "execution_time": 0.3,
                "entities_count": 3,
                "relationships_count": 1,
                "language": "java",
            },
        ]

        request = BatchAnalysisRequest(container_name="proj", analysis_type="augmented")
        response = await analyze_files_batch(request=request, tenant_id="t")
        pm = response["performance_metrics"]
        assert pm["fastest_file"] == 0.1
        assert pm["slowest_file"] == 0.3
        assert pm["total_entities"] == 8
        assert pm["total_relationships"] == 3
        assert pm["languages_analyzed"] == 2


# ---------------------------------------------------------------------------
# Tests for process_files_concurrent and analyze_single_file_batch
# (lines 248-249, 261-294, 300-370)
# ---------------------------------------------------------------------------

import asyncio
import tempfile

from app.api.endpoints.analysis_modules.batch_analysis import (
    analyze_single_file_batch,
    process_files_concurrent,
)


class TestProcessFilesConcurrent:
    """Tests for process_files_concurrent function."""

    async def test_handles_exception_results(self):
        """Exception results from gather are converted to error dicts (lines 272-280)."""
        tasks = [
            {"index": 0, "path": "/fake/a.py", "timeout": 1},
            {"index": 1, "path": "/fake/b.py", "timeout": 1},
        ]
        semaphore = asyncio.Semaphore(2)

        with patch(
            "app.api.endpoints.analysis_modules.batch_analysis.analyze_single_file_batch",
            side_effect=RuntimeError("analysis failed"),
        ):
            results = await process_files_concurrent(tasks, semaphore)
            assert len(results) == 2
            assert results[0]["success"] is False
            assert "analysis failed" in results[0]["error"]
            assert results[1]["success"] is False

    async def test_handles_non_dict_results(self):
        """Non-dict results are converted to error dicts (lines 284-292)."""
        tasks = [{"index": 0, "path": "/fake/a.py", "timeout": 1}]
        semaphore = asyncio.Semaphore(2)

        with patch(
            "app.api.endpoints.analysis_modules.batch_analysis.analyze_single_file_batch",
            return_value="not a dict",
        ):
            results = await process_files_concurrent(tasks, semaphore)
            assert len(results) == 1
            assert results[0]["success"] is False
            assert "Unexpected result type" in results[0]["error"]


class TestAnalyzeSingleFileBatch:
    """Tests for analyze_single_file_batch function."""

    async def test_file_not_found(self):
        """FileNotFoundError for missing file returns error dict (lines 304-305, 368-377)."""
        task = {"index": 0, "path": "/nonexistent/file.py", "timeout": 10}
        result = await analyze_single_file_batch(task)
        assert result["success"] is False
        assert "not found" in result["error"].lower() or "No such file" in result["error"]

    async def test_successful_analysis(self):
        """Full successful analysis flow (lines 308-366)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write("def hello():\n    return 1\n")
            f.flush()
            temp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.language = "python"
            mock_result.success = True
            mock_result.entities = [MagicMock()]
            mock_result.relationships = []
            mock_result.lines_of_code = 2
            mock_result.complexity_score = 1.0
            mock_result.parser_used = MagicMock(value="treesitter")
            mock_result.errors = []
            mock_result.warnings = []

            mock_processor = MagicMock()
            mock_processor.parse_file = AsyncMock(return_value=mock_result)
            mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_result)

            task = {
                "index": 0,
                "path": temp_path,
                "language": "python",
                "timeout": 10,
                "priority": "high",
            }
            with patch(
                "app.api.endpoints.analysis_modules.batch_analysis.get_processor_for_language",
                new_callable=AsyncMock,
                return_value=mock_processor,
            ):
                result = await analyze_single_file_batch(task)
                assert result["success"] is True
                assert result["language"] == "python"
                assert result["entities_count"] == 1
                assert result["priority"] == "high"
        finally:
            import os

            os.unlink(temp_path)

    async def test_timeout_raises(self):
        """Timeout during analysis raises AsyncTimeoutError (lines 338-341)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write("x = 1\n")
            f.flush()
            temp_path = f.name

        try:
            mock_processor = MagicMock()

            async def slow_parse(*args, **kwargs):
                await asyncio.sleep(100)

            mock_processor.parse_file = slow_parse

            task = {"index": 0, "path": temp_path, "language": "python", "timeout": 0.001}
            with patch(
                "app.api.endpoints.analysis_modules.batch_analysis.get_processor_for_language",
                return_value=mock_processor,
            ):
                with pytest.raises(Exception):
                    await analyze_single_file_batch(task)
        finally:
            import os

            os.unlink(temp_path)
