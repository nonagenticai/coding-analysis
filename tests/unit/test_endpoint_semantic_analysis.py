"""Unit tests for semantic analysis endpoints.

No sys.modules surgery needed — the endpoint module imports cleanly
with ``pythonpath = ["src"]`` (set in pyproject.toml) and the conftest
pre-loading of ``services`` and ``services.github_provisioning_service``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.semantic_analysis import (
    SemanticAnalysisRequest,
    SemanticAnalysisResponse,
    analyze_codebase_semantics,
    get_semantic_analysis_capabilities,
    get_semantic_patterns,
)

_VALIDATE = "app.api.endpoints.semantic_analysis.validate_and_get_container_for_tenant"
_RESOLVE_FILES = "app.api.endpoints.semantic_analysis._resolve_container_files"
_GET_SAS = "app.api.endpoints.semantic_analysis.get_semantic_analysis_service"


class TestSemanticAnalysisEndpoints:
    """Tests for semantic analysis endpoints."""

    @patch(_GET_SAS)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE)
    async def test_semantic_analysis_success(self, mock_validate, mock_resolve, mock_get_sas):
        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/app/data/repo/main.py")]

        mock_svc = AsyncMock()
        mock_svc.analyze_codebase_semantics = AsyncMock(
            return_value={
                "analysis_summary": {"status": "completed", "files_analyzed": 1},
                "entities": [{"name": "MyClass", "type": "class", "language": "python"}],
                "basic_relationships": [{"source": "a", "target": "b", "type": "calls"}],
                "semantic_relationships": {
                    "ffi_call": [{"source": "rust_fn", "target": "c_fn", "confidence": 0.95}]
                },
            }
        )
        mock_svc.get_semantic_insights = AsyncMock(
            return_value={
                "language_distribution": {"python": {"entities": 1}},
                "cross_language_interactions": [],
                "architectural_patterns": [],
                "complexity_hotspots": [],
                "coupling_analysis": {},
            }
        )
        mock_get_sas.return_value = mock_svc

        request = SemanticAnalysisRequest(container_name="proj")
        response = await analyze_codebase_semantics(request=request, tenant_id="t")

        # Validate response type
        assert isinstance(response, SemanticAnalysisResponse)

        # Validate entity/relationship counts
        assert response.result.entities_count == 1
        assert response.result.basic_relationships_count == 1
        assert response.result.semantic_relationships_count == 1

        # Validate analysis_summary structure
        assert response.result.analysis_summary["status"] == "completed"
        assert response.result.analysis_summary["files_analyzed"] == 1

        # Validate entities list carried through
        assert len(response.result.entities) == 1
        assert response.result.entities[0]["name"] == "MyClass"

        # Validate basic_relationships carried through
        assert len(response.result.basic_relationships) == 1
        assert response.result.basic_relationships[0]["source"] == "a"

        # Validate semantic_relationships dict carried through
        assert "ffi_call" in response.result.semantic_relationships
        assert len(response.result.semantic_relationships["ffi_call"]) == 1

        # Validate semantic insights were built
        assert response.result.semantic_insights is not None
        assert response.result.semantic_insights.language_distribution == {
            "python": {"entities": 1}
        }

        # Validate execution_time is a positive float
        assert response.result.execution_time > 0

        # Validate service was called with correct args
        mock_svc.analyze_codebase_semantics.assert_awaited_once()
        call_args = mock_svc.analyze_codebase_semantics.call_args
        assert call_args[0][0] == [Path("/app/data/repo/main.py")]

    async def test_semantic_capabilities_success(self):
        response = await get_semantic_analysis_capabilities(tenant_id="t")

        # Validate relationship types are populated from the real enum
        assert len(response.semantic_relationship_types) > 0
        type_values = [rt["type"] for rt in response.semantic_relationship_types]
        # Verify specific enum members show up
        assert "ffi_call" in type_values
        assert "api_endpoint" in type_values
        assert "service_call" in type_values
        assert "database_access" in type_values
        # Each entry must have type and description
        for rt in response.semantic_relationship_types:
            assert "type" in rt
            assert "description" in rt
            assert len(rt["description"]) > 0

        # Validate supported languages include expected ones
        assert len(response.supported_languages) > 0
        assert "python" in response.supported_languages
        assert "java" in response.supported_languages
        assert "rust" in response.supported_languages
        assert "go" in response.supported_languages

        # Validate cross-language patterns
        assert "rust_c_ffi" in response.cross_language_patterns
        assert "java_jni" in response.cross_language_patterns

        # Validate analysis features are all boolean True
        assert response.analysis_features["ffi_detection"] is True
        assert response.analysis_features["api_endpoint_mapping"] is True

        # Validate performance characteristics are present
        assert "typical_analysis_time" in response.performance_characteristics

    async def test_semantic_patterns_success(self):
        response = await get_semantic_patterns(pattern_type=None, tenant_id="t")

        # All three pattern categories must be populated
        assert len(response.cross_language_patterns) > 0
        assert len(response.architectural_patterns) > 0
        assert len(response.type_system_patterns) > 0

        # Verify specific cross-language pattern names
        cl_names = [p["name"] for p in response.cross_language_patterns]
        assert "rust_c_ffi" in cl_names
        assert "java_jni" in cl_names
        assert "go_cgo" in cl_names
        assert "rest_api_calls" in cl_names

        # Verify cross-language pattern structure
        for p in response.cross_language_patterns:
            assert "name" in p
            assert "description" in p
            assert "source_languages" in p
            assert "target_languages" in p
            assert "detection_method" in p

        # Verify specific architectural pattern names
        arch_names = [p["name"] for p in response.architectural_patterns]
        assert "service_oriented" in arch_names
        assert "event_driven" in arch_names
        assert "data_access" in arch_names

        # Verify specific type system pattern names
        ts_names = [p["name"] for p in response.type_system_patterns]
        assert "generic_constraints" in ts_names
        assert "polymorphic_calls" in ts_names
        assert "type_conversions" in ts_names

    async def test_semantic_patterns_filter_cross_language(self):
        """Filtering by pattern_type='cross_language' returns only cross-language patterns."""
        response = await get_semantic_patterns(pattern_type="cross_language", tenant_id="t")
        assert len(response.cross_language_patterns) > 0
        assert response.architectural_patterns == []
        assert response.type_system_patterns == []

    async def test_semantic_patterns_filter_architectural(self):
        """Filtering by pattern_type='architectural' returns only architectural patterns."""
        response = await get_semantic_patterns(pattern_type="architectural", tenant_id="t")
        assert response.cross_language_patterns == []
        assert len(response.architectural_patterns) > 0
        assert response.type_system_patterns == []

    async def test_semantic_patterns_filter_type_system(self):
        """Filtering by pattern_type='type_system' returns only type system patterns."""
        response = await get_semantic_patterns(pattern_type="type_system", tenant_id="t")
        assert response.cross_language_patterns == []
        assert response.architectural_patterns == []
        assert len(response.type_system_patterns) > 0

    async def test_semantic_patterns_invalid_filter_raises(self):
        """Unknown pattern_type raises HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
            await get_semantic_patterns(pattern_type="bogus", tenant_id="t")
        assert exc_info.value.status_code == 400

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE)
    async def test_parser_failure_mapping(self, mock_validate, mock_resolve):
        """When _resolve_container_files returns empty, response says no supported files."""
        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = []

        request = SemanticAnalysisRequest(container_name="empty-proj")
        response = await analyze_codebase_semantics(request=request, tenant_id="t")

        assert response.result.entities_count == 0
        assert response.result.basic_relationships_count == 0
        assert response.result.semantic_relationships_count == 0
        assert response.result.analysis_summary["status"] == "no_supported_files"
        assert response.result.analysis_summary["container"] == "empty-proj"
        assert response.result.execution_time > 0

    @patch(_GET_SAS)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE)
    async def test_analyzer_failure_mapping(self, mock_validate, mock_resolve, mock_get_sas):
        """When the service raises, the endpoint returns 500."""
        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/app/data/repo/main.py")]
        mock_svc = AsyncMock()
        mock_svc.analyze_codebase_semantics = AsyncMock(side_effect=RuntimeError("boom"))
        mock_get_sas.return_value = mock_svc

        request = SemanticAnalysisRequest(container_name="proj")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_codebase_semantics(request=request, tenant_id="t")
        assert exc_info.value.status_code == 500
        assert "boom" in str(exc_info.value.detail)

    @patch(_GET_SAS)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE)
    async def test_analysis_without_insights(self, mock_validate, mock_resolve, mock_get_sas):
        """When include_insights=False, semantic_insights should be None."""
        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/app/data/repo/main.py")]

        mock_svc = AsyncMock()
        mock_svc.analyze_codebase_semantics = AsyncMock(
            return_value={
                "analysis_summary": {"status": "completed"},
                "entities": [],
                "basic_relationships": [],
                "semantic_relationships": {},
            }
        )
        mock_get_sas.return_value = mock_svc

        request = SemanticAnalysisRequest(container_name="proj", include_insights=False)
        response = await analyze_codebase_semantics(request=request, tenant_id="t")

        assert response.result.semantic_insights is None
        # get_semantic_insights should NOT have been called
        mock_svc.get_semantic_insights.assert_not_awaited()

    @patch(_GET_SAS)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE)
    async def test_basic_analysis_skips_insights(self, mock_validate, mock_resolve, mock_get_sas):
        """When analysis_type='basic', insights are skipped even with include_insights=True."""
        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = [Path("/app/data/repo/main.py")]

        mock_svc = AsyncMock()
        mock_svc.analyze_codebase_semantics = AsyncMock(
            return_value={
                "analysis_summary": {"status": "completed"},
                "entities": [],
                "basic_relationships": [],
                "semantic_relationships": {},
            }
        )
        mock_get_sas.return_value = mock_svc

        request = SemanticAnalysisRequest(
            container_name="proj",
            analysis_type="basic",
            include_insights=True,
        )
        response = await analyze_codebase_semantics(request=request, tenant_id="t")

        assert response.result.semantic_insights is None
        mock_svc.get_semantic_insights.assert_not_awaited()
