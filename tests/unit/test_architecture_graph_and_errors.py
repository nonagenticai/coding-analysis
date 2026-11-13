"""Tests for architecture.py uncovered paths: graph endpoint, error branches, trigger_analysis.

Covers lines 85-86, 147-155, 198-208, 249-259, 300-310, 347, 351-361,
414-424, 460-466, 483-493, 535-638.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_RESOLVE = "app.api.endpoints.architecture._resolve_container_to_codebase"
_PROC = "app.api.endpoints.architecture._try_get_architecture_processor"
_GET_CM = "app.api.endpoints.architecture.get_container_manager"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(id_="comp-1", name="MyService", layer="service", deps=None):
    comp = MagicMock()
    comp.id = id_
    comp.name = name
    comp.layer = layer
    comp.dependencies = deps or []
    return comp


def _make_layer(name="service"):
    layer = MagicMock()
    layer.name = name
    return layer


def _make_metric(component_id="comp-1", value=5.0, instability=0.3, abstractness=0.4, distance=0.2):
    m = MagicMock()
    m.component_id = component_id
    m.value = value
    m.instability = instability
    m.abstractness = abstractness
    m.distance = distance
    return m


def _make_analysis_result(components=None, layers=None, metrics=None, patterns=None):
    result = MagicMock()
    result.codebase_id = "test-codebase"
    result.components = components or [_make_component()]
    result.layers = layers or [_make_layer()]
    result.metrics = metrics or [_make_metric()]
    result.patterns = patterns or []
    result.summary = {"status": 1}
    result.execution_time = 1.23
    return result


# ---------------------------------------------------------------------------
# _resolve_container_to_codebase: internal error path (line 85-86)
# ---------------------------------------------------------------------------


class TestResolveContainerErrors:
    @pytest.mark.asyncio
    async def test_generic_exception_becomes_500(self):
        from app.api.endpoints.architecture import _resolve_container_to_codebase

        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(side_effect=RuntimeError("db timeout"))
        with (
            patch(_GET_CM, return_value=mock_mgr),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_to_codebase("my-container", "t1")
        assert exc_info.value.status_code == 500
        assert "Failed to resolve container" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# list_components: FileNotFoundError -> 404 (line 147-148),
#                  generic "not found" string -> 404 (152-153),
#                  generic exception -> 500 (154-155)
# ---------------------------------------------------------------------------


class TestListComponentsErrors:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import list_components

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("repo gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await list_components(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_not_found_string_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import list_components

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("Resource not found in DB"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await list_components(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import list_components

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("db crashed"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await list_components(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_dependencies error paths (198-208)
# ---------------------------------------------------------------------------


class TestGetDependenciesErrors:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_dependencies

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_dependencies(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_dependencies

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("crash"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_dependencies(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_layers error paths (249-259)
# ---------------------------------------------------------------------------


class TestGetLayersErrors:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_layers

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_layers(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_does_not_exist_error_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_layers

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("path does not exist"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_layers(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# get_metrics error paths (300-310)
# ---------------------------------------------------------------------------


class TestGetMetricsErrors:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_metrics

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_metrics(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_metrics

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("crash"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_metrics(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_patterns: processor None (347), error paths (351-361)
# ---------------------------------------------------------------------------


class TestGetPatternsErrorPaths:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_processor_none_returns_empty_list(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_patterns

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        mock_proc.return_value = None

        result = await get_patterns(container_name="c", tenant_id="t")
        assert result == []

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_patterns

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_patterns(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# get_full_analysis error paths (414-424)
# ---------------------------------------------------------------------------


class TestGetFullAnalysisErrors:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_full_analysis

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_full_analysis(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_full_analysis

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("internal"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_full_analysis(container_name="c", tenant_id="t")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# trigger_analysis: empty name (460), processor None (464-466), errors (483-493)
# ---------------------------------------------------------------------------


class TestTriggerAnalysisEdgeCases:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_empty_container_name_raises_400(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import trigger_analysis

        with pytest.raises(HTTPException) as exc_info:
            await trigger_analysis(container_name="  ", force_rebuild=False, tenant_id="t")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_processor_none_returns_fallback(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import trigger_analysis

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        mock_proc.return_value = None

        result = await trigger_analysis(container_name="c", force_rebuild=True, tenant_id="t")
        assert result.components == []
        assert result.layers == []
        assert result.execution_time == 0.0

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import trigger_analysis

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await trigger_analysis(container_name="c", force_rebuild=False, tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import trigger_analysis

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("boom"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await trigger_analysis(container_name="c", force_rebuild=False, tenant_id="t")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_architecture_graph: full success path with components (535-638)
# ---------------------------------------------------------------------------


class TestGetArchitectureGraphSuccess:
    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_with_service_component(self, mock_resolve, mock_proc):
        """Test graph endpoint with a 'service' named component builds correct graph."""
        from app.api.endpoints.architecture import get_architecture_graph

        comp = _make_component(id_="c1", name="UserService", layer="service", deps=["c2"])
        comp2 = _make_component(id_="c2", name="DataModule", layer="data", deps=[])
        metric = _make_metric(component_id="c1", value=15.0, distance=0.6)
        metric2 = _make_metric(component_id="c2", value=3.0, distance=0.1)
        layer1 = _make_layer("service")
        layer2 = _make_layer("data")
        result = _make_analysis_result(
            components=[comp, comp2],
            layers=[layer1, layer2],
            metrics=[metric, metric2],
        )

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("c", tenant_id="t")

        assert len(response.nodes) == 2
        assert len(response.edges) == 1
        assert response.edges[0].source == "c1"
        assert response.edges[0].target == "c2"
        assert response.metrics.total_components == 2
        assert response.metrics.total_dependencies == 1

        # Check node types: 'service' in name -> type 'service'
        service_node = [n for n in response.nodes if n.id == "c1"][0]
        assert service_node.type == "service"

        # Check module type detection
        module_node = [n for n in response.nodes if n.id == "c2"][0]
        assert module_node.type == "module"

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_with_package_component(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        comp = _make_component(id_="p1", name="CorePackage", layer="core", deps=[])
        metric = _make_metric(component_id="p1", value=2.0, distance=0.0)
        result = _make_analysis_result(components=[comp], metrics=[metric])

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("c", tenant_id="t")
        node = response.nodes[0]
        assert node.type == "package"
        # Low complexity, low distance -> no issues
        assert node.issues == 0

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_with_high_complexity_hotspot(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        comp = _make_component(id_="h1", name="BigComp", layer="core", deps=[])
        metric = _make_metric(component_id="h1", value=20.0, distance=0.8)
        result = _make_analysis_result(components=[comp], metrics=[metric])

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("c", tenant_id="t")
        node = response.nodes[0]
        # High complexity -> hotspot score should be > 0
        assert node.hotspotScore is not None and node.hotspotScore > 0
        # complexity > 10 -> estimated_issues > 0
        assert node.issues is not None and node.issues > 0
        # hotspot score > 0.7 -> counted in hotspot_count
        assert response.metrics.hotspot_count >= 1

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_no_metrics_for_component(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        comp = _make_component(id_="nm1", name="PlainComp", layer="core", deps=[])
        # No metric matching this component
        metric = _make_metric(component_id="other", value=5.0)
        result = _make_analysis_result(components=[comp], metrics=[metric])

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("c", tenant_id="t")
        node = response.nodes[0]
        # No metrics -> metrics is None, hotspot is None, issues is None
        assert node.metrics is None
        assert node.hotspotScore is None
        assert node.issues is None

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_file_not_found_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=FileNotFoundError("gone"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_architecture_graph("c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_generic_error_raises_500(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("crash"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_architecture_graph("c", tenant_id="t")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_not_found_string_raises_404(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("Container not found"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_architecture_graph("c", tenant_id="t")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_empty_metrics_avg_complexity_zero(self, mock_resolve, mock_proc):
        from app.api.endpoints.architecture import get_architecture_graph

        comp = _make_component(id_="e1", name="EmptyComp", deps=[])
        result = MagicMock()
        result.codebase_id = "test"
        result.components = [comp]
        result.layers = [_make_layer()]
        result.metrics = []  # truly empty
        result.patterns = []

        mock_resolve.return_value = ("cb", Path("/app/data/repo"))
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("c", tenant_id="t")
        assert response.metrics.avg_complexity == 0.0
