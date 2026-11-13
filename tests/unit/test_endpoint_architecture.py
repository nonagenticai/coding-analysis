"""Unit tests for architecture analysis endpoints."""

from __future__ import annotations

import pathlib
import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# sys.modules surgery — must run before importing the real endpoint module.
#
# architecture.py has a deep import chain:
#   architecture.py -> processing.architecture.architecture_processor
#                   -> services.docker_service
#                   -> services/__init__.py -> db.facade (postgres)
#
# We must stub the broken leaves before importing.
# ---------------------------------------------------------------------------

_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"

# 1. Evict conftest stubs for modules we need to load for real.
for _key in list(sys.modules):
    if _key in (
        "app",
        "app.api",
        "app.api.endpoints",
        "app.api.endpoints.architecture",
        "app.core",
        "app.core.file_service",
        "app.core.security_config",
    ):
        del sys.modules[_key]

# 2. Ensure app packages exist with real __path__
for _pkg, _subpath in (
    ("app", "app"),
    ("app.api", "app/api"),
    ("app.api.endpoints", "app/api/endpoints"),
    ("app.api.models", "app/api/models"),
    ("app.api.utils", "app/api/utils"),
    ("app.core", "app/core"),
):
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = [str(_src_dir / _subpath)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m
    elif not getattr(sys.modules[_pkg], "__path__", None):
        sys.modules[_pkg].__path__ = [str(_src_dir / _subpath)]

# 3. Stub services as a package so services/__init__.py doesn't run
if "services" not in sys.modules:
    _svc = _types.ModuleType("services")
    _svc.__path__ = [str(_src_dir / "services")]
    _svc.__package__ = "services"
    sys.modules["services"] = _svc
elif not getattr(sys.modules["services"], "__path__", None):
    sys.modules["services"].__path__ = [str(_src_dir / "services")]

# Stub specific service modules
_ds_stub = _types.ModuleType("services.docker_service")
_ds_stub.DockerService = MagicMock
_ds_stub.get_docker_service = MagicMock(return_value=MagicMock())
sys.modules.setdefault("services.docker_service", _ds_stub)

_cgm_stub = _types.ModuleType("services.code_graph_metrics")
_cgm_stub.CodeGraphMetricsService = MagicMock
sys.modules.setdefault("services.code_graph_metrics", _cgm_stub)

_mlgs_stub = _types.ModuleType("services.code_graph.multi_language_graph_service")
_mlgs_stub.MultiLanguageGraphServiceFactory = MagicMock
sys.modules.setdefault("services.code_graph.multi_language_graph_service", _mlgs_stub)

# Ensure services.code_graph package
if "services.code_graph" not in sys.modules:
    _scg = _types.ModuleType("services.code_graph")
    _scg.__path__ = [str(_src_dir / "services" / "code_graph")]
    _scg.__package__ = "services.code_graph"
    sys.modules["services.code_graph"] = _scg

# 5. Stub processing.architecture as a package
if "processing" not in sys.modules:
    _proc = _types.ModuleType("processing")
    _proc.__path__ = [str(_src_dir / "processing")]
    _proc.__package__ = "processing"
    sys.modules["processing"] = _proc

if "processing.architecture" not in sys.modules:
    _arch_pkg = _types.ModuleType("processing.architecture")
    _arch_pkg.__path__ = [str(_src_dir / "processing" / "architecture")]
    _arch_pkg.__package__ = "processing.architecture"
    sys.modules["processing.architecture"] = _arch_pkg

if "processing.architecture.architecture_processor" not in sys.modules:
    _ap_stub = _types.ModuleType("processing.architecture.architecture_processor")
    _ap_stub.ArchitectureProcessor = MagicMock
    sys.modules["processing.architecture.architecture_processor"] = _ap_stub

if "processing.architecture.graph_adapter" not in sys.modules:
    _ga_stub = _types.ModuleType("processing.architecture.graph_adapter")
    _ga_stub.GraphAdapter = MagicMock
    sys.modules["processing.architecture.graph_adapter"] = _ga_stub

# 6. Stub app.core.container_manager
if "app.core.container_manager" not in sys.modules:
    _cm_stub = _types.ModuleType("app.core.container_manager")
    _cm_stub.get_container_manager = MagicMock(return_value=MagicMock())
    sys.modules["app.core.container_manager"] = _cm_stub

# 7. Stub app.api.utils.container_validation
if "app.api.utils.container_validation" not in sys.modules:
    _cv_stub = _types.ModuleType("app.api.utils.container_validation")
    _cv_stub.validate_and_get_container_for_tenant = AsyncMock()
    sys.modules["app.api.utils.container_validation"] = _cv_stub

# 8. Now import the real module
from app.api.endpoints.architecture import (  # noqa: E402
    CycleCheckRequest,
    _find_cycle_signatures,
    cycle_check,
    get_architecture_graph,
    get_dependencies,
    get_full_analysis,
    get_layers,
    get_metrics,
    get_patterns,
    list_components,
    trigger_analysis,
)
from models.architecture_models import (  # noqa: E402
    ArchitectureAnalysisResult,
    ArchitectureGraphResponse,
)

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
    result.components = [_make_component()] if components is None else components
    result.layers = [_make_layer()] if layers is None else layers
    result.metrics = [_make_metric()] if metrics is None else metrics
    result.patterns = [] if patterns is None else patterns
    result.summary = {"status": 1}
    result.execution_time = 1.23
    return result


_RESOLVE = "app.api.endpoints.architecture._resolve_container_to_codebase"
_PROC = "app.api.endpoints.architecture._try_get_architecture_processor"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArchitectureEndpoints:
    """Tests for architecture analysis endpoints."""

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_components_endpoint_success(self, mock_resolve, mock_proc):
        comp1 = _make_component(id_="svc-auth", name="AuthService", layer="service", deps=["db-1"])
        comp2 = _make_component(id_="db-1", name="DatabaseModule", layer="infrastructure", deps=[])
        mock_resolve.return_value = ("test-codebase", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp1, comp2])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await list_components(container_name="my-container", tenant_id="test-tenant")

        assert len(response) == 2
        assert response[0].id == "svc-auth"
        assert response[0].name == "AuthService"
        assert response[0].layer == "service"
        assert response[0].dependencies == ["db-1"]
        assert response[1].id == "db-1"
        assert response[1].dependencies == []
        mock_resolve.assert_awaited_once_with("my-container", "test-tenant")

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_dependencies_endpoint_success_shape(self, mock_resolve, mock_proc):
        comp1 = _make_component(id_="a", name="A", deps=["b", "c"])
        comp2 = _make_component(id_="b", name="B", deps=["c"])
        comp3 = _make_component(id_="c", name="C", deps=[])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp1, comp2, comp3])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_dependencies(container_name="c", tenant_id="t")

        assert isinstance(response, list)
        assert len(response) == 3
        # Validate each entry has exactly the right keys
        for entry in response:
            assert set(entry.keys()) == {"component", "dependencies"}
        # Validate content transformation from components to dep mappings
        assert response[0]["component"] == "a"
        assert response[0]["dependencies"] == ["b", "c"]
        assert response[1]["component"] == "b"
        assert response[1]["dependencies"] == ["c"]
        assert response[2]["component"] == "c"
        assert response[2]["dependencies"] == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_dependencies_empty_components(self, mock_resolve, mock_proc):
        """An analysis with no components returns an empty dependency list."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_dependencies(container_name="c", tenant_id="t")

        assert response == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_dependencies_response_model_accepts_shape(self, mock_resolve, mock_proc):
        """Regression for ResponseValidationError (4 errors).

        The endpoint returns ``{"component": <str>, "dependencies": [<str>...]}``.
        The old ``response_model=list[dict[str, list[str]]]`` rejected the str
        ``component`` value (1 error per component). The fixed model
        ``list[dict[str, str | list[str]]]`` must accept it.
        """
        from pydantic import TypeAdapter, ValidationError

        comps = [
            _make_component(id_="a", name="A", deps=["b"]),
            _make_component(id_="b", name="B", deps=[]),
        ]
        mock_resolve.return_value = ("cb", "/workspace")
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=_make_analysis_result(components=comps))
        mock_proc.return_value = processor

        response = await get_dependencies(container_name="c", tenant_id="t")

        # Fixed model accepts the mixed str / list[str] values.
        TypeAdapter(list[dict[str, str | list[str]]]).validate_python(response)
        # The old model would reject the str "component" value -> the bug.
        with pytest.raises(ValidationError):
            TypeAdapter(list[dict[str, list[str]]]).validate_python(response)

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_layers_endpoint_success(self, mock_resolve, mock_proc):
        layer1 = _make_layer(name="presentation")
        layer2 = _make_layer(name="domain")
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(layers=[layer1, layer2])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_layers(container_name="c", tenant_id="t")

        assert isinstance(response, dict)
        assert list(response.keys()) == ["layers"]
        assert len(response["layers"]) == 2
        assert response["layers"][0].name == "presentation"
        assert response["layers"][1].name == "domain"

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_layers_empty_processor_fallback(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = None

        response = await get_layers(container_name="c", tenant_id="t")
        assert response == {"layers": []}

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_metrics_endpoint_success(self, mock_resolve, mock_proc):
        metric1 = _make_metric(component_id="svc-a", value=8.5, instability=0.6)
        metric2 = _make_metric(component_id="svc-b", value=2.1, instability=0.1)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(metrics=[metric1, metric2])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_metrics(container_name="c", tenant_id="t")

        assert isinstance(response, dict)
        assert list(response.keys()) == ["metrics"]
        assert len(response["metrics"]) == 2
        assert response["metrics"][0].component_id == "svc-a"
        assert response["metrics"][0].value == 8.5
        assert response["metrics"][1].component_id == "svc-b"
        assert response["metrics"][1].value == 2.1

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_metrics_empty_processor_fallback(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = None

        response = await get_metrics(container_name="c", tenant_id="t")
        assert response == {"metrics": []}

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_patterns_endpoint_success(self, mock_resolve, mock_proc):
        pattern1 = MagicMock()
        pattern1.pattern_type = "singleton"
        pattern1.confidence = 0.9
        pattern2 = MagicMock()
        pattern2.pattern_type = "observer"
        pattern2.confidence = 0.75
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(patterns=[pattern1, pattern2])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_patterns(container_name="c", tenant_id="t")

        assert len(response) == 2
        assert response[0].pattern_type == "singleton"
        assert response[0].confidence == 0.9
        assert response[1].pattern_type == "observer"

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_patterns_empty_when_none_detected(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(patterns=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_patterns(container_name="c", tenant_id="t")
        assert response == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_full_analysis_success_validates_all_fields(self, mock_resolve, mock_proc):
        comp = _make_component(id_="web-ui", name="WebUI", layer="presentation")
        layer = _make_layer(name="presentation")
        metric = _make_metric(component_id="web-ui", value=7.0)
        mock_resolve.return_value = ("my-codebase", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(
            components=[comp], layers=[layer], metrics=[metric], patterns=[]
        )
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_full_analysis(container_name="c", tenant_id="t")

        assert response.codebase_id == "test-codebase"
        assert response.components == [comp]
        assert response.layers == [layer]
        assert response.metrics == [metric]
        assert response.patterns == []
        assert response.summary == {"status": 1}
        assert response.execution_time == 1.23

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_success_validates_result(self, mock_resolve, mock_proc):
        comp = _make_component(id_="api-gw", name="ApiGateway")
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await trigger_analysis(container_name="c", force_rebuild=False, tenant_id="t")

        assert response.codebase_id == "test-codebase"
        assert response.components[0].id == "api-gw"
        assert response.execution_time == 1.23

    @patch("services.code_graph.graph_builder.build_code_graph")
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_passes_force_rebuild_to_job(
        self, mock_resolve, mock_proc, mock_build
    ):
        """force_rebuild=True must re-extract the graph (build_code_graph) AND flow into the job."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result()
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor
        mock_build.return_value = {"nodes": 5, "edges": 4, "skipped": False}

        await trigger_analysis(container_name="c", force_rebuild=True, tenant_id="t")

        # force_rebuild re-extracts the post-refactor graph before the processor loads it
        mock_build.assert_awaited_once_with("cb", "t", force_rebuild=True)

        # Inspect the job argument passed to processor.process
        call_args = processor.process.call_args
        job = call_args[0][0]
        assert job.force_rebuild is True
        assert job.codebase_id == "cb"
        assert job.repo_path == Path("/app/data/repo")

    @patch("services.code_graph.graph_builder.build_code_graph")
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_skips_rebuild_when_not_forced(
        self, mock_resolve, mock_proc, mock_build
    ):
        """force_rebuild=False must NOT call build_code_graph (read-path stays cheap)."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=_make_analysis_result())
        mock_proc.return_value = processor

        await trigger_analysis(container_name="c", force_rebuild=False, tenant_id="t")

        mock_build.assert_not_awaited()

    @patch(_RESOLVE)
    async def test_container_not_found_returns_404(self, mock_resolve):
        mock_resolve.side_effect = HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "Container 'missing' not found",
                "error_code": "CONTAINER_NOT_FOUND",
                "details": {},
            },
        )

        with pytest.raises(HTTPException) as exc_info:
            await list_components(container_name="missing", tenant_id="t")

        assert exc_info.value.status_code == 404
        assert "missing" in str(exc_info.value.detail)

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_processor_unavailable_returns_fallback_analysis(self, mock_resolve, mock_proc):
        """When processor is None, full analysis returns a valid empty ArchitectureAnalysisResult."""
        mock_resolve.return_value = ("my-repo", "/app/data/repo")
        mock_proc.return_value = None

        response = await get_full_analysis(container_name="c", tenant_id="t")

        assert isinstance(response, ArchitectureAnalysisResult)
        assert response.codebase_id == "my-repo"
        assert response.components == []
        assert response.layers == []
        assert response.metrics == []
        assert response.patterns == []
        assert response.summary == {"status": 0}
        assert response.execution_time == 0.0
        # generated_at should be an ISO timestamp string
        assert len(response.generated_at) > 10

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_processor_unavailable_returns_fallback(
        self, mock_resolve, mock_proc
    ):
        """trigger_analysis also returns fallback when processor is None."""
        mock_resolve.return_value = ("my-repo", "/app/data/repo")
        mock_proc.return_value = None

        response = await trigger_analysis(container_name="c", force_rebuild=True, tenant_id="t")

        assert isinstance(response, ArchitectureAnalysisResult)
        assert response.codebase_id == "my-repo"
        assert response.components == []
        assert response.execution_time == 0.0

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_file_not_found_raises_404(self, mock_resolve, mock_proc):
        """FileNotFoundError from processor is converted to 404."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            side_effect=FileNotFoundError("Repository path /app/data/repo does not exist")
        )
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await list_components(container_name="c", tenant_id="t")

        assert exc_info.value.status_code == 404
        assert "does not exist" in exc_info.value.detail

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_generic_not_found_string_raises_404(self, mock_resolve, mock_proc):
        """Generic exception containing 'not found' in message becomes 404."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("Container X not found in cluster"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_layers(container_name="c", tenant_id="t")

        assert exc_info.value.status_code == 404

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_unexpected_error_raises_500(self, mock_resolve, mock_proc):
        """Unexpected errors without 'not found' become 500."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(side_effect=RuntimeError("database connection timed out"))
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await get_metrics(container_name="c", tenant_id="t")

        assert exc_info.value.status_code == 500
        assert "database connection timed out" in str(exc_info.value.detail)

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_empty_name_raises_400(self, mock_resolve, mock_proc):
        """Empty container name triggers bad_request."""
        with pytest.raises(HTTPException) as exc_info:
            await trigger_analysis(container_name="", force_rebuild=False, tenant_id="t")

        assert exc_info.value.status_code == 400

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_trigger_analysis_whitespace_name_raises_400(self, mock_resolve, mock_proc):
        """Whitespace-only container name triggers bad_request."""
        with pytest.raises(HTTPException) as exc_info:
            await trigger_analysis(container_name="   ", force_rebuild=False, tenant_id="t")

        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Tests merged from test_endpoint_architecture_coverage.py
# ---------------------------------------------------------------------------


class TestArchitectureRouter:
    """Verify the architecture router is correctly defined."""

    def test_router_has_routes(self):
        from app.api.endpoints.architecture import router

        assert len(router.routes) > 0

    def test_has_components_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("components" in p for p in paths)

    def test_has_dependencies_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("dependencies" in p for p in paths)

    def test_has_layers_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("layers" in p for p in paths)

    def test_has_metrics_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("metrics" in p for p in paths)

    def test_has_patterns_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("patterns" in p for p in paths)

    def test_has_analysis_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("analysis" in p for p in paths)

    def test_has_graph_route(self):
        from app.api.endpoints.architecture import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("graph" in p for p in paths)

    def test_has_analyze_post_route(self):
        from app.api.endpoints.architecture import router

        post_routes = [
            r.path
            for r in router.routes
            if hasattr(r, "path") and hasattr(r, "methods") and "POST" in r.methods
        ]
        assert any("analyze" in p for p in post_routes)


class TestResolveContainerToCodebase:
    """Test the _resolve_container_to_codebase helper."""

    @pytest.mark.asyncio
    async def test_returns_codebase_id_and_path_on_success(self):
        from app.api.endpoints.architecture import _resolve_container_to_codebase

        mock_container = MagicMock()
        mock_container.container_name = "my-repo"
        # Repos are cloned to the container's codebase_path (e.g. /workspace), not
        # the old hardcoded /app/data/repo.
        mock_container.codebase_path = "/workspace"

        mock_manager = MagicMock()
        mock_manager.get_container = AsyncMock(return_value=mock_container)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_manager,
            ),
            # Self-heal is best-effort and tested separately; stub it here so the
            # resolver test does not touch the database.
            patch(
                "app.api.endpoints.architecture._ensure_graph_populated",
                new=AsyncMock(),
            ),
        ):
            codebase_id, repo_path = await _resolve_container_to_codebase("my-repo", "tenant-1")

        assert codebase_id == "my-repo"
        assert repo_path == Path("/workspace")

    @pytest.mark.asyncio
    async def test_raises_404_when_container_not_found(self):
        from app.api.endpoints.architecture import _resolve_container_to_codebase

        mock_manager = MagicMock()
        mock_manager.get_container = AsyncMock(side_effect=ValueError("not found"))

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_manager,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_to_codebase("missing", "tenant-1")

        assert exc_info.value.status_code == 404
        assert "missing" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_raises_404_when_container_is_none(self):
        from app.api.endpoints.architecture import _resolve_container_to_codebase

        mock_manager = MagicMock()
        mock_manager.get_container = AsyncMock(return_value=None)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_manager,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_to_codebase("gone", "tenant-1")

        assert exc_info.value.status_code == 404
        assert "gone" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_500(self):
        """Non-HTTP exceptions from container manager become 500."""
        from app.api.endpoints.architecture import _resolve_container_to_codebase

        mock_manager = MagicMock()
        mock_manager.get_container = AsyncMock(side_effect=RuntimeError("DB down"))

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_manager,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_to_codebase("repo", "tenant-1")

        assert exc_info.value.status_code == 500
        assert "DB down" in str(exc_info.value.detail)


class TestEnsureGraphPopulated:
    """Test the best-effort self-heal that builds the graph before analysis."""

    @pytest.mark.asyncio
    async def test_calls_build_code_graph(self):
        from app.api.endpoints.architecture import _ensure_graph_populated

        mock_build = AsyncMock(return_value={"nodes": 5, "edges": 3, "skipped": False})
        with patch(
            "services.code_graph.graph_builder.build_code_graph",
            mock_build,
        ):
            await _ensure_graph_populated("my-repo", "tenant-1")

        mock_build.assert_awaited_once_with("my-repo", "tenant-1")

    @pytest.mark.asyncio
    async def test_swallows_build_errors(self):
        """A build failure must never propagate (read endpoints must not 500)."""
        from app.api.endpoints.architecture import _ensure_graph_populated

        mock_build = AsyncMock(side_effect=RuntimeError("analysis upstream down"))
        with patch(
            "services.code_graph.graph_builder.build_code_graph",
            mock_build,
        ):
            # Must not raise.
            await _ensure_graph_populated("my-repo", "tenant-1")


class TestTryGetArchitectureProcessor:
    """Test the _try_get_architecture_processor helper."""

    @pytest.mark.asyncio
    async def test_returns_none_on_service_failure(self):
        from app.api.endpoints.architecture import _try_get_architecture_processor

        # Replace the entire factory object so we control get_service
        mock_factory = MagicMock()
        mock_factory.get_service = AsyncMock(side_effect=RuntimeError("no DB"))

        with patch(
            "app.api.endpoints.architecture.MultiLanguageGraphServiceFactory",
            mock_factory,
        ):
            result = await _try_get_architecture_processor()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_processor_on_success(self):
        from app.api.endpoints.architecture import _try_get_architecture_processor

        mock_service = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_service = AsyncMock(return_value=mock_service)

        with (
            patch(
                "app.api.endpoints.architecture.MultiLanguageGraphServiceFactory",
                mock_factory,
            ),
            patch("app.api.endpoints.architecture.GraphAdapter") as mock_adapter_cls,
            patch("app.api.endpoints.architecture.ArchitectureProcessor") as mock_proc_cls,
        ):
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            result = await _try_get_architecture_processor()

        # Verify the processor was constructed with the right dependencies
        mock_adapter_cls.assert_called_once_with(mock_service)
        mock_proc_cls.assert_called_once_with(mock_adapter)
        assert result == mock_proc_cls.return_value


class TestListComponentsEndpoint:
    """Test the list_components endpoint handler."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_processor_unavailable(self):
        from app.api.endpoints.architecture import list_components

        mock_container = MagicMock()
        mock_container.container_name = "repo"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_mgr,
            ),
            patch(
                "app.api.endpoints.architecture._try_get_architecture_processor",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await list_components("repo", tenant_id="t1")

        assert result == []


class TestGetDependenciesEndpoint:
    """Test the get_dependencies endpoint handler."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_processor_unavailable(self):
        from app.api.endpoints.architecture import get_dependencies

        mock_container = MagicMock()
        mock_container.container_name = "repo"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_mgr,
            ),
            patch(
                "app.api.endpoints.architecture._try_get_architecture_processor",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await get_dependencies("repo", tenant_id="t1")

        assert result == []


class TestGetLayersEndpoint:
    """Test the get_layers endpoint handler."""

    @pytest.mark.asyncio
    async def test_returns_empty_layers_when_processor_unavailable(self):
        from app.api.endpoints.architecture import get_layers

        mock_container = MagicMock()
        mock_container.container_name = "repo"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_mgr,
            ),
            patch(
                "app.api.endpoints.architecture._try_get_architecture_processor",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await get_layers("repo", tenant_id="t1")

        assert result == {"layers": []}


class TestGetArchitectureGraphEndpoint:
    """Test the get_architecture_graph endpoint handler."""

    @pytest.mark.asyncio
    async def test_returns_empty_graph_when_processor_unavailable(self):
        mock_container = MagicMock()
        mock_container.container_name = "repo"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)

        with (
            patch(
                "app.api.endpoints.architecture.get_container_manager",
                return_value=mock_mgr,
            ),
            patch(
                "app.api.endpoints.architecture._try_get_architecture_processor",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await get_architecture_graph("repo", tenant_id="t1")

        assert isinstance(result, ArchitectureGraphResponse)
        assert result.nodes == []
        assert result.edges == []
        assert result.layers == []
        assert result.metrics.total_components == 0
        assert result.metrics.total_dependencies == 0
        assert result.metrics.avg_complexity == 0.0
        assert result.metrics.hotspot_count == 0

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_component_type_classification(self, mock_resolve, mock_proc):
        """Components are classified by name: service, module, package, or generic component."""
        comps = [
            _make_component(id_="c1", name="AuthService", layer="svc"),
            _make_component(id_="c2", name="UserModule", layer="mod"),
            _make_component(id_="c3", name="CorePackage", layer="pkg"),
            _make_component(id_="c4", name="Utils", layer="util"),
            _make_component(id_="c5", name="pkg-helpers", layer="util"),
        ]
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=comps, metrics=[], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        node_types = {n.id: n.type for n in response.nodes}
        assert node_types["c1"] == "service"
        assert node_types["c2"] == "module"
        assert node_types["c3"] == "package"
        assert node_types["c4"] == "component"  # no keyword match
        assert node_types["c5"] == "package"  # "pkg" in name

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_edges_from_dependencies(self, mock_resolve, mock_proc):
        """Edges are built from component dependency lists."""
        comp1 = _make_component(id_="a", name="A", deps=["b", "c"])
        comp2 = _make_component(id_="b", name="B", deps=["c"])
        comp3 = _make_component(id_="c", name="C", deps=[])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp1, comp2, comp3], metrics=[], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        edge_pairs = [(e.source, e.target) for e in response.edges]
        assert ("a", "b") in edge_pairs
        assert ("a", "c") in edge_pairs
        assert ("b", "c") in edge_pairs
        assert len(response.edges) == 3
        # All edges should be type "dependency" with weight 1
        for edge in response.edges:
            assert edge.type == "dependency"
            assert edge.weight == 1

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_metrics_computation(self, mock_resolve, mock_proc):
        """Verify the summary metrics: total_components, total_dependencies, avg_complexity."""
        comp1 = _make_component(id_="x", name="X", deps=["y"])
        comp2 = _make_component(id_="y", name="Y", deps=[])
        metric1 = _make_metric(component_id="x", value=10.0, distance=0.2)
        metric2 = _make_metric(component_id="y", value=4.0, distance=0.1)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(
            components=[comp1, comp2],
            metrics=[metric1, metric2],
            layers=[_make_layer("svc")],
        )
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        assert response.metrics.total_components == 2
        assert response.metrics.total_dependencies == 1  # only x->y
        assert response.metrics.avg_complexity == 7.0  # (10 + 4) / 2
        assert response.layers == ["svc"]

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_hotspot_score_calculation(self, mock_resolve, mock_proc):
        """Hotspot score = min(1.0, (value/10)*0.5 + distance*0.5)."""
        # value=20 -> (20/10)*0.5=1.0, distance=0.8 -> 0.8*0.5=0.4 => min(1.0, 1.4) = 1.0
        comp_hot = _make_component(id_="hot", name="HotComponent")
        metric_hot = _make_metric(component_id="hot", value=20.0, distance=0.8)
        # value=2 -> (2/10)*0.5=0.1, distance=0.1 -> 0.1*0.5=0.05 => 0.15
        comp_cold = _make_component(id_="cold", name="ColdComponent")
        metric_cold = _make_metric(component_id="cold", value=2.0, distance=0.1)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(
            components=[comp_hot, comp_cold],
            metrics=[metric_hot, metric_cold],
            layers=[],
        )
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        nodes_by_id = {n.id: n for n in response.nodes}
        assert nodes_by_id["hot"].hotspotScore == 1.0
        assert nodes_by_id["cold"].hotspotScore == pytest.approx(0.15)
        # hotspot_count: only nodes > 0.7 qualify
        assert response.metrics.hotspot_count == 1

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_estimated_issues_high_complexity(self, mock_resolve, mock_proc):
        """Components with complexity>10 or distance>0.5 get estimated issues."""
        comp = _make_component(id_="messy", name="MessyComponent")
        # complexity=15, distance=0.6 -> int(15/5) + int(0.6*10) = 3 + 6 = 9
        metric = _make_metric(component_id="messy", value=15.0, distance=0.6)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp], metrics=[metric], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        node = response.nodes[0]
        assert node.issues == 9

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_estimated_issues_low_complexity(self, mock_resolve, mock_proc):
        """Components with low complexity and low distance get 0 issues."""
        comp = _make_component(id_="clean", name="CleanComponent")
        metric = _make_metric(component_id="clean", value=3.0, distance=0.1)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp], metrics=[metric], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        assert response.nodes[0].issues == 0

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_no_metrics_for_component(self, mock_resolve, mock_proc):
        """Component with no matching metric gets None for metrics, hotspot, issues."""
        comp = _make_component(id_="orphan", name="OrphanComponent")
        # Metric for a different component
        metric = _make_metric(component_id="other", value=5.0)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp], metrics=[metric], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        node = response.nodes[0]
        assert node.metrics is None
        assert node.hotspotScore is None
        assert node.issues is None

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_node_preserves_layer(self, mock_resolve, mock_proc):
        """Graph nodes carry the layer assignment from their component."""
        comp = _make_component(id_="x", name="X", layer="infrastructure")
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp], metrics=[], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        assert response.nodes[0].layer == "infrastructure"
        assert response.nodes[0].name == "X"

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_metrics_with_none_distance(self, mock_resolve, mock_proc):
        """Metrics with distance=None use 0.0 fallback in hotspot calculation."""
        comp = _make_component(id_="nd", name="NoDistance")
        metric = _make_metric(component_id="nd", value=6.0, distance=None)
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        result = _make_analysis_result(components=[comp], metrics=[metric], layers=[])
        processor.process = AsyncMock(return_value=result)
        mock_proc.return_value = processor

        response = await get_architecture_graph("repo", tenant_id="t1")

        node = response.nodes[0]
        # (6/10)*0.5 + 0.0*0.5 = 0.3
        assert node.hotspotScore == pytest.approx(0.3)
        # complexity=6 < 10 and distance=None (treated as 0) < 0.5
        # BUT the code checks comp_metrics.get("distance", 0) which is None (from metric.distance)
        # not 0.0 — the "or 0.0" in the hotspot calc doesn't affect estimated_issues
        # Since value=6 < 10 and distance=None... let's check actual behavior
        # In comp_metrics dict: "distance": metric.distance or 0.0 => 0.0
        # Then: complexity=6 < 10 and distance=0 < 0.5 => estimated_issues=0
        assert node.issues == 0


class TestFindCycleSignatures:
    """Unit tests for the pure SCC cycle enumerator."""

    def test_acyclic_graph_has_no_cycles(self):
        adj = {"a": ["b", "c"], "b": ["c"], "c": []}
        assert _find_cycle_signatures(adj) == []

    def test_two_node_cycle(self):
        adj = {"a": ["b"], "b": ["a"]}
        sigs = _find_cycle_signatures(adj)
        assert sigs == [["a", "b"]]

    def test_self_loop_is_a_cycle(self):
        adj = {"a": ["a"], "b": []}
        assert _find_cycle_signatures(adj) == [["a"]]

    def test_three_node_cycle_plus_acyclic_tail(self):
        adj = {"a": ["b"], "b": ["c"], "c": ["a"], "d": ["a"]}
        sigs = _find_cycle_signatures(adj)
        assert sigs == [["a", "b", "c"]]

    def test_edges_to_unknown_nodes_ignored(self):
        # 'a' -> 'ext' where 'ext' has no entry: cannot close a cycle.
        adj = {"a": ["ext"], "b": []}
        assert _find_cycle_signatures(adj) == []

    def test_two_independent_cycles_both_returned(self):
        adj = {"a": ["b"], "b": ["a"], "x": ["y"], "y": ["x"], "z": []}
        sigs = {tuple(s) for s in _find_cycle_signatures(adj)}
        assert sigs == {("a", "b"), ("x", "y")}


class TestCycleCheckEndpoint:
    """Tests for the DOING architecture-delta gate endpoint."""

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_baseline_mode_reports_current_cycles(self, mock_resolve, mock_proc):
        # a <-> b is a cycle; c is acyclic.
        comp_a = _make_component(id_="a", deps=["b"])
        comp_b = _make_component(id_="b", deps=["a"])
        comp_c = _make_component(id_="c", deps=[])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            return_value=_make_analysis_result(components=[comp_a, comp_b, comp_c])
        )
        mock_proc.return_value = processor

        resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["success"] is True
        assert resp["mode"] == "baseline"
        assert resp["cycles"] == [["a", "b"]]
        assert resp["current_cycles"] == [["a", "b"]]

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_gate_blocks_on_new_cycle(self, mock_resolve, mock_proc):
        # Baseline had no cycle; post-refactor introduces a <-> b.
        comp_a = _make_component(id_="a", deps=["b"])
        comp_b = _make_component(id_="b", deps=["a"])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            return_value=_make_analysis_result(components=[comp_a, comp_b])
        )
        mock_proc.return_value = processor

        resp = await cycle_check(
            container_name="c",
            body=CycleCheckRequest(baseline_cycles=[]),
            tenant_id="t",
        )

        assert resp["success"] is False
        assert resp["mode"] == "gate"
        assert resp["new_cycles"] == [["a", "b"]]
        assert "new dependency cycle" in resp["error"]

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_gate_passes_when_no_new_cycle(self, mock_resolve, mock_proc):
        # The a<->b cycle was already in the baseline -> not a NEW cycle.
        comp_a = _make_component(id_="a", deps=["b"])
        comp_b = _make_component(id_="b", deps=["a"])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            return_value=_make_analysis_result(components=[comp_a, comp_b])
        )
        mock_proc.return_value = processor

        resp = await cycle_check(
            container_name="c",
            body=CycleCheckRequest(baseline_cycles=[["a", "b"]]),
            tenant_id="t",
        )

        assert resp["success"] is True
        assert resp["mode"] == "gate"
        assert resp["new_cycles"] == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_gate_passes_when_cycle_removed(self, mock_resolve, mock_proc):
        # Baseline had a<->b; the refactor removed it -> no new cycle, pass.
        comp_a = _make_component(id_="a", deps=[])
        comp_b = _make_component(id_="b", deps=[])
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            return_value=_make_analysis_result(components=[comp_a, comp_b])
        )
        mock_proc.return_value = processor

        resp = await cycle_check(
            container_name="c",
            body=CycleCheckRequest(baseline_cycles=[["a", "b"]]),
            tenant_id="t",
        )

        assert resp["success"] is True
        assert resp["new_cycles"] == []
        assert resp["current_cycles"] == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_processor_unavailable_is_vacuous_pass(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = None

        resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["success"] is True
        assert resp["cycles"] == []

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_missing_container_raises_404(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", "/app/data/repo")
        processor = AsyncMock()
        processor.process = AsyncMock(
            side_effect=FileNotFoundError("Repository path /app/data/repo does not exist")
        )
        mock_proc.return_value = processor

        with pytest.raises(HTTPException) as exc_info:
            await cycle_check(
                container_name="c",
                body=CycleCheckRequest(baseline_cycles=[]),
                tenant_id="t",
            )
        assert exc_info.value.status_code == 404


class TestCycleCheckBaselineFreshness:
    """The baseline must describe the tree as it is NOW, not a stale extraction.

    THE DEFECT THIS PINS (measured live on AA drive 6, 2026-09-01). ``build_code_graph``
    opens with ``if not force_rebuild and await graph_is_populated(...): return
    {"skipped": True}``, and the graph is keyed on ``container_name``, which the AA
    drivers reuse across drives. Baseline mode reached it only through the best-effort
    self-heal, which passes no ``force_rebuild`` — so the "baseline" described whatever
    earlier state built the graph, typically the PREVIOUS drive's POST-refactor tree.
    The gate then compared post-refactor to post-refactor and could never fail: a
    container holding a real module-level import ring reported ``cycles: []``.
    """

    _BUILD = "services.code_graph.graph_builder.build_code_graph"

    @staticmethod
    def _ring():
        """alpha -> beta -> gamma -> alpha, across three components."""
        return [
            _make_component(id_="alpha", deps=["beta"]),
            _make_component(id_="beta", deps=["gamma"]),
            _make_component(id_="gamma", deps=["alpha"]),
        ]

    @staticmethod
    def _acyclic():
        return [
            _make_component(id_="alpha", deps=["beta"]),
            _make_component(id_="beta", deps=[]),
        ]

    def _processor(self, components):
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=_make_analysis_result(components=components))
        return processor

    # --- RED against origin/main -------------------------------------------------
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_baseline_forces_a_fresh_graph_build(self, mock_resolve, mock_proc):
        """FAILS on origin/main: baseline mode never asked for a re-extraction."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = self._processor(self._ring())
        mock_build = AsyncMock(return_value={"nodes": 12, "edges": 9, "skipped": False})

        with patch(self._BUILD, mock_build):
            resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        forced = [
            call for call in mock_build.await_args_list if call.kwargs.get("force_rebuild") is True
        ]
        assert forced, (
            "baseline mode must FORCE a graph rebuild; without it the baseline "
            "describes a stale tree and the gate can never fail. calls="
            f"{mock_build.await_args_list}"
        )
        assert resp["cycles"] == [["alpha", "beta", "gamma"]]
        assert resp["graph_built"] is True

    # --- negative control: the fix must not become "rebuild always" --------------
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_gate_mode_does_not_force_a_rebuild(self, mock_resolve, mock_proc):
        """The procedure's own `rebuild_code_graph` step already forced one.

        Without this control the cheap "fix" is to rebuild unconditionally, which
        doubles a slow extraction on every gate for no added signal.
        """
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = self._processor(self._acyclic())
        mock_build = AsyncMock(return_value={"nodes": 4, "edges": 1, "skipped": True})

        with patch(self._BUILD, mock_build):
            resp = await cycle_check(
                container_name="c",
                body=CycleCheckRequest(baseline_cycles=[["alpha", "beta", "gamma"]]),
                tenant_id="t",
            )

        forced = [
            call for call in mock_build.await_args_list if call.kwargs.get("force_rebuild") is True
        ]
        assert not forced, f"gate mode must not re-extract; calls={mock_build.await_args_list}"
        assert resp["success"] is True
        assert resp["baseline_cycle_count"] == 1

    # --- positive control: a genuinely acyclic tree still reports no cycle -------
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_acyclic_tree_still_reports_no_cycle_and_is_grounded(
        self, mock_resolve, mock_proc
    ):
        """Guards against "make everything report a cycle"."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = self._processor(self._acyclic())
        mock_build = AsyncMock(return_value={"nodes": 4, "edges": 1, "skipped": False})

        with patch(self._BUILD, mock_build):
            resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["cycles"] == []
        assert resp["grounded"] is True
        assert resp["ungrounded_reason"] is None
        assert resp["component_count"] == 2

    # --- provenance: too small a graph is not a finding --------------------------
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_single_component_is_not_grounded(self, mock_resolve, mock_proc):
        """A cycle needs >= 2 components; below that "no cycle" is absence of evidence."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = self._processor([_make_component(id_="solo", deps=[])])
        mock_build = AsyncMock(return_value={"nodes": 2, "edges": 0, "skipped": False})

        with patch(self._BUILD, mock_build):
            resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["cycles"] == []
        assert resp["grounded"] is False
        assert "1 component" in resp["ungrounded_reason"]

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_failed_baseline_build_is_ungrounded(self, mock_resolve, mock_proc):
        """A build that raised means the baseline may describe a stale tree."""
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = self._processor(self._ring())
        mock_build = AsyncMock(side_effect=RuntimeError("graph service down"))

        with patch(self._BUILD, mock_build):
            resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["grounded"] is False
        assert "stale" in resp["ungrounded_reason"]
        assert resp["graph_built"] is False

    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_missing_backend_is_ungrounded_not_a_clean_pass(self, mock_resolve, mock_proc):
        """`success` stays True (never hard-block DOING on infra) but nothing was measured.

        Before this, a missing graph backend returned a bare `success: True` with
        `cycles: []` — indistinguishable from a measured clean bill of health.
        """
        mock_resolve.return_value = ("cb", "/app/data/repo")
        mock_proc.return_value = None

        resp = await cycle_check(container_name="c", body=None, tenant_id="t")

        assert resp["success"] is True
        assert resp["grounded"] is False
        assert "absence of evidence" in resp["ungrounded_reason"]


class TestEnsureGraphPopulatedForce:
    """`force=True` is opt-in; the default call shape must stay byte-identical."""

    @pytest.mark.asyncio
    async def test_force_passes_force_rebuild(self):
        from app.api.endpoints.architecture import _ensure_graph_populated

        mock_build = AsyncMock(return_value={"nodes": 5, "edges": 3, "skipped": False})
        with patch("services.code_graph.graph_builder.build_code_graph", mock_build):
            result = await _ensure_graph_populated("my-repo", "tenant-1", force=True)

        mock_build.assert_awaited_once_with("my-repo", "tenant-1", force_rebuild=True)
        assert result == {"nodes": 5, "edges": 3, "skipped": False}

    @pytest.mark.asyncio
    async def test_failed_build_returns_none_and_does_not_raise(self):
        from app.api.endpoints.architecture import _ensure_graph_populated

        mock_build = AsyncMock(side_effect=RuntimeError("down"))
        with patch("services.code_graph.graph_builder.build_code_graph", mock_build):
            assert await _ensure_graph_populated("my-repo", "t", force=True) is None


class TestArchitectureOpsStageSelection:
    """D7: which pipeline stages each of the three timed-out ops asks for.

    The processor no longer has a quality stage and its dependency stage only
    parses manifests (see test_architecture_processor.TestPipelineWorkIsBounded).
    The graph op does not report the package count, so it skips even that.
    """

    @staticmethod
    def _processor():
        processor = AsyncMock()
        processor.process = AsyncMock(return_value=_make_analysis_result())
        return processor

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_graph_op_skips_the_dependency_stage(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", Path("/workspace"))
        mock_proc.return_value = processor = self._processor()

        await get_architecture_graph("repo", tenant_id="t1")

        processor.process.assert_awaited_once()
        assert processor.process.await_args.kwargs == {"include_dependency_analysis": False}

    @pytest.mark.asyncio
    @patch(_PROC)
    @patch(_RESOLVE)
    async def test_full_analysis_and_trigger_keep_the_package_count(self, mock_resolve, mock_proc):
        mock_resolve.return_value = ("cb", Path("/workspace"))
        mock_proc.return_value = processor = self._processor()

        await get_full_analysis("repo", tenant_id="t1")
        await trigger_analysis("repo", force_rebuild=False, tenant_id="t1")

        assert processor.process.await_count == 2
        for call in processor.process.await_args_list:
            job = call.args[0]
            assert job.tenant_id == "t1"
            assert call.kwargs == {}
