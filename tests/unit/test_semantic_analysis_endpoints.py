"""Tests for semantic_analysis.py uncovered paths.

Covers lines 57-146, 151-157, 475, 575-577, 696-721, 732-736.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_VALIDATE_CONTAINER = "app.api.endpoints.semantic_analysis.validate_and_get_container_for_tenant"
_RESOLVE_FILES = "app.api.endpoints.semantic_analysis._resolve_container_files"
_GET_SEM_SERVICE = "app.api.endpoints.semantic_analysis.get_semantic_analysis_service"


# ---------------------------------------------------------------------------
# _validate_direct_file_paths
# ---------------------------------------------------------------------------


class TestValidateDirectFilePaths:
    def test_file_not_found_raises_400(self):
        from app.api.endpoints.semantic_analysis import _validate_direct_file_paths

        with pytest.raises(HTTPException) as exc_info:
            _validate_direct_file_paths(["/nonexistent/file.py"])
        assert exc_info.value.status_code == 400

    def test_valid_file_paths(self, tmp_path):
        from app.api.endpoints.semantic_analysis import _validate_direct_file_paths

        f = tmp_path / "test.py"
        f.write_text("x = 1")
        result = _validate_direct_file_paths([str(f)])
        assert len(result) == 1
        assert result[0] == f


# ---------------------------------------------------------------------------
# _resolve_container_files
# ---------------------------------------------------------------------------


_MATERIALIZE = "app.api.endpoints.semantic_analysis.materialize_tenant_workspace"


class TestResolveContainerFiles:
    """The workspace lives in the tenant's workspace pod, never on this pod.

    Replaced: test_codebase_path_not_exists_raises_404 / test_codebase_is_file_raises_400
    (they pinned the os.path checks on THIS pod's disk that 404'd every real
    workspace, prod 2026-10-11) and the list_containers-based not-found test.
    """

    @pytest.mark.asyncio
    async def test_container_not_found_raises_404_exact_name(self):
        from app.api.endpoints.semantic_analysis import _resolve_container_files

        with (
            patch(
                _VALIDATE_CONTAINER,
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=404, detail="Container 'cv-ems' not found"),
            ) as mock_validate,
            patch(_MATERIALIZE, new_callable=AsyncMock) as mock_mat,
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("cv-ems", "tenant-1")
        assert exc_info.value.status_code == 404
        mock_validate.assert_awaited_once_with("cv-ems", "tenant-1")
        mock_mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unreadable_workspace_is_error_not_fallback(self):
        from app.api.endpoints.semantic_analysis import _resolve_container_files

        info = MagicMock(codebase_path="/workspace")
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=info),
            patch(
                _MATERIALIZE,
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=422, detail="WORKSPACE_UNRESOLVABLE"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("cv-ems-p4", "tenant-1")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_resolves_specific_files(self, tmp_path):
        from app.api.endpoints.semantic_analysis import _resolve_container_files

        (tmp_path / "main.py").write_text("x = 1")
        info = MagicMock(codebase_path="/workspace")
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=info),
            patch(_MATERIALIZE, new_callable=AsyncMock, return_value=tmp_path) as mock_mat,
        ):
            result = await _resolve_container_files("cv-ems-p4", "tenant-1", ["main.py"])
        assert result == [tmp_path / "main.py"]
        assert result.materialized_root == tmp_path
        args, kwargs = mock_mat.call_args
        assert args == ("cv-ems-p4", "tenant-1")
        assert kwargs["rel_paths"] == ["main.py"]
        assert kwargs["container_info"] is info

    @pytest.mark.asyncio
    async def test_auto_discovers_materialised_files(self, tmp_path):
        from app.api.endpoints.semantic_analysis import _resolve_container_files

        (tmp_path / "main.py").write_text("x = 1")
        info = MagicMock(codebase_path="/workspace")
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=info),
            patch(_MATERIALIZE, new_callable=AsyncMock, return_value=tmp_path) as mock_mat,
        ):
            result = await _resolve_container_files("cv-ems-p4", "tenant-1")
        assert result == [tmp_path / "main.py"]
        assert "*.py" in mock_mat.call_args.kwargs["name_patterns"]
        result.release()
        assert not tmp_path.exists()


# ---------------------------------------------------------------------------
# analyze_codebase_semantics
# ---------------------------------------------------------------------------


class TestAnalyzeCodebaseSemantics:
    @pytest.mark.asyncio
    @patch(_GET_SEM_SERVICE)
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_no_files_returns_empty(self, mock_validate, mock_resolve, mock_svc):
        from app.api.endpoints.semantic_analysis import (
            SemanticAnalysisRequest,
            analyze_codebase_semantics,
        )

        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = []

        req = SemanticAnalysisRequest(container_name="test")
        result = await analyze_codebase_semantics(request=req, tenant_id="test-tenant")
        assert result.result.entities_count == 0

    @pytest.mark.asyncio
    @patch(_GET_SEM_SERVICE)
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_success_with_insights(self, mock_validate, mock_resolve, mock_svc):
        from pathlib import Path

        from app.api.endpoints.semantic_analysis import (
            SemanticAnalysisRequest,
            analyze_codebase_semantics,
        )

        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/tmp/test.py")]

        mock_service = MagicMock()
        mock_service.analyze_codebase_semantics = AsyncMock(
            return_value={
                "analysis_summary": {"status": "ok"},
                "entities": [{"name": "Foo"}],
                "basic_relationships": [{"type": "calls"}],
                "semantic_relationships": {"ffi": [{"type": "ffi"}]},
            }
        )
        mock_service.get_semantic_insights = AsyncMock(
            return_value={
                "language_distribution": {"python": {"count": 1}},
                "cross_language_interactions": [],
                "architectural_patterns": [],
                "complexity_hotspots": [],
                "coupling_analysis": {"pairs": []},
            }
        )
        mock_svc.return_value = mock_service

        req = SemanticAnalysisRequest(
            container_name="test", include_insights=True, analysis_type="full"
        )
        result = await analyze_codebase_semantics(request=req, tenant_id="test-tenant")
        assert result.result.entities_count == 1
        assert result.result.semantic_insights is not None

    @pytest.mark.asyncio
    @patch(_GET_SEM_SERVICE)
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_basic_analysis_skips_insights(self, mock_validate, mock_resolve, mock_svc):
        from pathlib import Path

        from app.api.endpoints.semantic_analysis import (
            SemanticAnalysisRequest,
            analyze_codebase_semantics,
        )

        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/tmp/test.py")]

        mock_service = MagicMock()
        mock_service.analyze_codebase_semantics = AsyncMock(
            return_value={
                "analysis_summary": {"status": "ok"},
                "entities": [],
                "basic_relationships": [],
                "semantic_relationships": {},
            }
        )
        mock_svc.return_value = mock_service

        req = SemanticAnalysisRequest(
            container_name="test", analysis_type="basic", include_insights=True
        )
        result = await analyze_codebase_semantics(request=req, tenant_id="test-tenant")
        # basic analysis skips insights
        assert result.result.semantic_insights is None

    @pytest.mark.asyncio
    @patch(_VALIDATE_CONTAINER, new_callable=AsyncMock)
    async def test_error_raises_500(self, mock_validate):
        from app.api.endpoints.semantic_analysis import (
            SemanticAnalysisRequest,
            analyze_codebase_semantics,
        )

        mock_validate.side_effect = RuntimeError("boom")

        req = SemanticAnalysisRequest(container_name="test")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_codebase_semantics(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_semantic_analysis_capabilities
# ---------------------------------------------------------------------------


class TestGetSemanticCapabilities:
    @pytest.mark.asyncio
    async def test_success(self):
        from app.api.endpoints.semantic_analysis import get_semantic_analysis_capabilities

        result = await get_semantic_analysis_capabilities(tenant_id="test-tenant")
        assert len(result.supported_languages) > 0
        assert len(result.cross_language_patterns) > 0

    @pytest.mark.asyncio
    async def test_error_raises_500(self):
        from app.api.endpoints.semantic_analysis import get_semantic_analysis_capabilities

        # Patch _get_relationship_description to cause an error during iteration
        with patch(
            "app.api.endpoints.semantic_analysis._get_relationship_description",
            side_effect=RuntimeError("broken"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_semantic_analysis_capabilities(tenant_id="test-tenant")
            assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_semantic_patterns
# ---------------------------------------------------------------------------


class TestGetSemanticPatterns:
    @pytest.mark.asyncio
    async def test_all_patterns(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        result = await get_semantic_patterns(pattern_type=None, tenant_id="test-tenant")
        assert len(result.cross_language_patterns) > 0
        assert len(result.architectural_patterns) > 0
        assert len(result.type_system_patterns) > 0

    @pytest.mark.asyncio
    async def test_cross_language_filter(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        result = await get_semantic_patterns(pattern_type="cross_language", tenant_id="test-tenant")
        assert len(result.cross_language_patterns) > 0
        assert len(result.architectural_patterns) == 0

    @pytest.mark.asyncio
    async def test_architectural_filter(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        result = await get_semantic_patterns(pattern_type="architectural", tenant_id="test-tenant")
        assert len(result.architectural_patterns) > 0
        assert len(result.cross_language_patterns) == 0

    @pytest.mark.asyncio
    async def test_type_system_filter(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        result = await get_semantic_patterns(pattern_type="type_system", tenant_id="test-tenant")
        assert len(result.type_system_patterns) > 0
        assert len(result.cross_language_patterns) == 0

    @pytest.mark.asyncio
    async def test_unknown_pattern_type_raises_400(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        with pytest.raises(HTTPException) as exc_info:
            await get_semantic_patterns(pattern_type="unknown", tenant_id="test-tenant")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_pattern_type_raises_400(self):
        from app.api.endpoints.semantic_analysis import get_semantic_patterns

        with pytest.raises(HTTPException) as exc_info:
            await get_semantic_patterns(pattern_type="  ", tenant_id="test-tenant")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _get_relationship_description
# ---------------------------------------------------------------------------


class TestGetRelationshipDescription:
    def test_known_types(self):
        from app.api.endpoints.semantic_analysis import _get_relationship_description

        mock_type = MagicMock()
        mock_type.value = "ffi_call"
        desc = _get_relationship_description(mock_type)
        assert "Foreign function" in desc

    def test_unknown_type(self):
        from app.api.endpoints.semantic_analysis import _get_relationship_description

        mock_type = MagicMock()
        mock_type.value = "some_new_type"
        desc = _get_relationship_description(mock_type)
        assert "Advanced relationship" in desc


# ---------------------------------------------------------------------------
# _build_semantic_insights
# ---------------------------------------------------------------------------


class TestBuildSemanticInsights:
    def test_builds_correctly(self):
        from app.api.endpoints.semantic_analysis import _build_semantic_insights

        data = {
            "language_distribution": {"python": {"count": 5}},
            "cross_language_interactions": [
                {
                    "source_language": "python",
                    "target_language": "c",
                    "relationship_type": "ffi",
                    "source_entity": "call_c",
                    "target_entity": "c_func",
                }
            ],
            "architectural_patterns": [{"pattern": "microservice", "count": 3}],
            "complexity_hotspots": [
                {
                    "entity_name": "BigClass",
                    "entity_type": "class",
                    "language": "python",
                    "relationship_count": 15,
                    "file_path": "big.py",
                }
            ],
            "coupling_analysis": {"pairs": []},
        }
        result = _build_semantic_insights(data)
        assert len(result.cross_language_interactions) == 1
        assert len(result.architectural_patterns) == 1
        assert len(result.complexity_hotspots) == 1

    def test_handles_none_interactions(self):
        from app.api.endpoints.semantic_analysis import _build_semantic_insights

        data = {
            "language_distribution": {},
            "cross_language_interactions": None,
            "architectural_patterns": None,
            "complexity_hotspots": None,
            "coupling_analysis": {},
        }
        result = _build_semantic_insights(data)
        assert result.cross_language_interactions == []
