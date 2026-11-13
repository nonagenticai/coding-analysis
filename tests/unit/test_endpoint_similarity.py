"""Unit tests for similarity detection endpoints (similarity_endpoints.py — the v2/refactored version).

Tests verify actual request->response transformation, error handling, and input validation
by calling the endpoint functions directly.
"""

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
    if _key in (
        "app.api.endpoints.similarity",
        "app.api.endpoints.similarity_modules",
        "app.api.endpoints.similarity_modules.similarity_endpoints",
        "app.api.endpoints.similarity_modules.similarity_config",
        "app.api.endpoints.similarity_modules.similarity_processing",
    ):
        if _key in sys.modules:
            del sys.modules[_key]

for _pkg, _subpath in (
    ("app", "app"),
    ("app.api", "app/api"),
    ("app.api.endpoints", "app/api/endpoints"),
    ("app.api.endpoints.similarity_modules", "app/api/endpoints/similarity_modules"),
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

# services stubs
if "services" not in sys.modules:
    _svc = _types.ModuleType("services")
    _svc.__path__ = [str(_src_dir / "services")]
    sys.modules["services"] = _svc
if "services.code_graph_metrics" not in sys.modules:
    _scgm = _types.ModuleType("services.code_graph_metrics")
    _scgm.CodeGraphMetricsService = MagicMock
    sys.modules["services.code_graph_metrics"] = _scgm

# Stub similarity_detector service
from enum import StrEnum  # noqa: E402


class _FakeSimilarityScope(StrEnum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    BLOCK = "block"


class _FakeSimilarityType(StrEnum):
    EXACT = "exact"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TOKEN = "token"


if "services.similarity_detector" not in sys.modules:
    try:
        import services.similarity_detector  # noqa: F401 — side-effect import
    except ImportError:
        _sd_stub = _types.ModuleType("services.similarity_detector")
        _sd_stub.SimilarityScope = _FakeSimilarityScope
        _sd_stub.SimilarityType = _FakeSimilarityType
        _sd_stub.SimilarityMatch = MagicMock
        _sd_stub.SimilarityAnalysisResult = MagicMock
        _sd_stub.SimilarityDetector = MagicMock
        _mock_detector = MagicMock()
        _sd_stub.get_similarity_detector = MagicMock(return_value=_mock_detector)
        sys.modules["services.similarity_detector"] = _sd_stub

# app.core.container_metadata_registry — ensure get_container_registry exists
if "app.core.container_metadata_registry" not in sys.modules:
    _cmr_stub = _types.ModuleType("app.core.container_metadata_registry")
    _cmr_stub.ContainerMetadataRegistry = MagicMock
    _cmr_stub.get_container_registry = MagicMock(return_value=MagicMock())
    sys.modules["app.core.container_metadata_registry"] = _cmr_stub
else:
    if not hasattr(sys.modules["app.core.container_metadata_registry"], "get_container_registry"):
        sys.modules["app.core.container_metadata_registry"].get_container_registry = MagicMock(
            return_value=MagicMock()
        )


# uuid_v7
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

# Now import
from app.api.endpoints.similarity_modules.similarity_endpoints import (  # noqa: E402
    analyze_similarity,
    detect_clones,
    find_cross_language_patterns,
    get_similarity_capabilities,
)

_GET_DETECTOR = "app.api.endpoints.similarity_modules.similarity_endpoints.get_similarity_detector"
_VALIDATE_REQ = "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityConfig.validate_analysis_request"
_RESOLVE_FILES = (
    "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityConfig.resolve_file_paths"
)
_VALIDATE_FILES = (
    "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityConfig.validate_file_paths"
)
_GET_CAPS = (
    "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityConfig.get_capabilities"
)
_PROCESS_RESULT = "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityProcessor.convert_result_to_model"
_PROCESS_CLONES = "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityProcessor.process_clone_results"
_PROCESS_CROSS = "app.api.endpoints.similarity_modules.similarity_endpoints.SimilarityProcessor.process_cross_language_results"


# ---------------------------------------------------------------------------
# Tests for v2 endpoint functions (called directly, not via TestClient)
# ---------------------------------------------------------------------------


class TestSimilarityEndpoints:
    """Tests for similarity detection endpoints."""

    @patch(_PROCESS_RESULT)
    @patch(_GET_DETECTOR)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_similarity_analysis_success_message_and_result(
        self, mock_val, mock_resolve, mock_detector, mock_proc
    ):
        """Verify the response message includes file/match count and result fields are set."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest
        from app.api.endpoints.similarity_modules.similarity_processing import (
            SimilarityAnalysisResponseModel,
        )

        mock_val.return_value = None
        mock_resolve.return_value = ["/a.py", "/b.py"]

        mock_result = MagicMock()
        mock_result.total_files_analyzed = 2
        mock_result.total_matches_found = 1
        mock_det = AsyncMock()
        mock_det.analyze_similarity = AsyncMock(return_value=mock_result)
        mock_detector.return_value = mock_det

        response_model = SimilarityAnalysisResponseModel(
            total_files_analyzed=2,
            total_entities_compared=10,
            total_matches_found=1,
            analysis_time_seconds=0.5,
            matches_by_type={},
            matches_by_score={},
            cross_language_matches=[],
            language_distribution={},
            average_similarity_score=0.8,
            highest_similarity_match=None,
            refactoring_opportunities=[],
            duplicate_code_ratio=0.1,
            cross_language_patterns=[],
        )
        mock_proc.return_value = response_model

        request = SimilarityAnalysisRequest(
            container_name="proj",
            scope="function",
        )
        result = await analyze_similarity(request=request, tenant_id="t")

        # Verify the message format: "Analyzed X files, found Y similarity matches"
        assert result.message == "Analyzed 2 files, found 1 similarity matches"
        assert result.result.total_files_analyzed == 2
        assert result.result.total_matches_found == 1
        assert result.result.average_similarity_score == 0.8
        # BaseResponse fields must be populated
        assert result.status == "success"
        assert result.timestamp is not None
        assert result.request_id is not None
        # Verify the result model sub-fields
        assert result.result.total_entities_compared == 10
        assert result.result.analysis_time_seconds == 0.5
        assert result.result.duplicate_code_ratio == 0.1
        assert result.result.matches_by_type == {}
        assert result.result.cross_language_matches == []

    @patch(_PROCESS_CLONES)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_clone_detection_returns_structured_response(
        self, mock_detector, mock_validate, mock_proc
    ):
        """Verify clone detection returns a CloneDetectionResponse with correct fields."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/a.py", "/b.py"]
        mock_det = AsyncMock()
        mock_det.detect_code_clones = AsyncMock(return_value=[{"clone": "data"}])
        mock_detector.return_value = mock_det
        mock_proc.return_value = {
            "total_clones": 1,
            "files_analyzed": 2,
            "clones": [],
            "summary": {"avg_similarity": 0.9},
        }

        request = CloneDetectionRequest(file_paths=["/a.py", "/b.py"])
        result = await detect_clones(request=request, tenant_id="t")

        assert result.total_clones == 1
        assert result.files_analyzed == 2
        assert "1 clones found" in result.message
        assert result.clone_threshold == 0.85  # default from model
        # Verify inherited BaseResponse fields
        assert result.status == "success"
        assert result.timestamp is not None
        assert result.request_id is not None
        # Verify list/dict fields
        assert result.clones == []
        assert result.summary == {"avg_similarity": 0.9}

    @patch(_PROCESS_CROSS)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_cross_language_patterns_returns_structured_response(
        self, mock_detector, mock_validate, mock_proc
    ):
        """Verify cross-language detection returns CrossLanguagePatternResponse with correct fields."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_validate.return_value = ["/a.py", "/b.rs"]
        mock_det = AsyncMock()
        mock_det.find_cross_language_patterns = AsyncMock(return_value=[{"pattern": "data"}])
        mock_detector.return_value = mock_det
        mock_proc.return_value = {
            "total_patterns": 3,
            "files_analyzed": 2,
            "patterns": [],
            "language_pairs": {"python-rust": 3},
            "summary": {},
        }

        request = CrossLanguagePatternRequest(file_paths=["/a.py", "/b.rs"])
        result = await find_cross_language_patterns(request=request, tenant_id="t")

        assert result.total_patterns == 3
        assert result.files_analyzed == 2
        assert "3 patterns found" in result.message
        assert result.language_pairs == {"python-rust": 3}
        assert result.pattern_threshold == 0.60  # default from model
        # Verify inherited BaseResponse fields
        assert result.status == "success"
        assert result.timestamp is not None
        assert result.patterns == []
        assert result.summary == {}

    @patch(_GET_CAPS)
    async def test_capabilities_returns_structured_response_with_algorithms(self, mock_caps):
        """Verify capabilities returns SimilarityCapabilitiesResponse with parsed algorithms."""
        mock_caps.return_value = {
            "similarity_types": ["exact", "near", "structural"],
            "analysis_scopes": ["file", "function"],
            # Source code has algorithms as dict[str, str], endpoint converts to list[str]
            "algorithms": {
                "cosine": "Vector-based comparison",
                "jaccard": "Set-based matching",
            },
            "supported_languages": ["python", "javascript"],
            "default_thresholds": {"exact": 0.95, "near": 0.85},
            "limitations": {"max_files": 100},
        }

        result = await get_similarity_capabilities(tenant_id="t")

        assert result.similarity_types == ["exact", "near", "structural"]
        assert result.analysis_scopes == ["file", "function"]
        # algorithms dict keys become the list
        assert set(result.algorithms) == {"cosine", "jaccard"}
        assert "python" in result.supported_languages
        assert "javascript" in result.supported_languages
        assert result.message == "Similarity detection capabilities retrieved successfully"
        # Verify thresholds and performance_limits come through
        assert result.thresholds == {"exact": 0.95, "near": 0.85}
        assert result.performance_limits == {"max_files": 100}

    @patch(_GET_CAPS)
    async def test_capabilities_algorithms_already_list_passes_through(self, mock_caps):
        """When algorithms is already a list (not dict), it passes through unchanged."""
        mock_caps.return_value = {
            "similarity_types": [],
            "analysis_scopes": [],
            "algorithms": ["cosine", "jaccard", "lsh"],
            "supported_languages": [],
            "default_thresholds": {},
            "limitations": {},
        }

        result = await get_similarity_capabilities(tenant_id="t")

        assert result.algorithms == ["cosine", "jaccard", "lsh"]

    @patch(_GET_CAPS)
    async def test_capabilities_missing_keys_default_to_empty(self, mock_caps):
        """When get_capabilities returns a dict missing some keys, defaults apply."""
        mock_caps.return_value = {}

        result = await get_similarity_capabilities(tenant_id="t")

        assert result.similarity_types == []
        assert result.analysis_scopes == []
        assert result.algorithms == []
        assert result.supported_languages == []
        assert result.thresholds == {}
        assert result.performance_limits == {}

    async def test_clone_detection_no_file_paths_and_no_container_raises_400(self):
        """When neither file_paths nor container_name given for clone detection, raises 400."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        request = CloneDetectionRequest()
        with pytest.raises(HTTPException) as exc_info:
            await detect_clones(request=request, tenant_id="t")
        assert exc_info.value.status_code == 400
        assert (
            "file_paths" in exc_info.value.detail.lower()
            or "container" in exc_info.value.detail.lower()
        )

    async def test_cross_language_no_file_paths_and_no_container_raises_400(self):
        """When neither file_paths nor container_name given for cross-language, raises 400."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        request = CrossLanguagePatternRequest()
        with pytest.raises(HTTPException) as exc_info:
            await find_cross_language_patterns(request=request, tenant_id="t")
        assert exc_info.value.status_code == 400
        assert (
            "file_paths" in exc_info.value.detail.lower()
            or "container" in exc_info.value.detail.lower()
        )

    @patch(_GET_DETECTOR)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_similarity_runtime_error_raises_500(
        self, mock_val, mock_resolve, mock_detector
    ):
        """A RuntimeError during analysis raises HTTPException with status 500 and the error message."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.return_value = ["/a.py", "/b.py"]
        mock_det = AsyncMock()
        mock_det.analyze_similarity = AsyncMock(side_effect=RuntimeError("analysis engine crash"))
        mock_detector.return_value = mock_det

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 500
        assert "analysis engine crash" in exc_info.value.detail
        assert "Similarity analysis failed" in exc_info.value.detail

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_too_few_files_raises_400(self, mock_val, mock_resolve):
        """Fewer than 2 files raises 400."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.return_value = ["/a.py"]

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 400
        assert "at least 2" in exc_info.value.detail.lower()

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_zero_files_raises_400(self, mock_val, mock_resolve):
        """Zero resolved files raises 400."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.return_value = []

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 400

    @patch(_GET_DETECTOR)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_invalid_scope_raises_400(self, mock_val, mock_resolve, mock_det):
        """Invalid scope raises 400 with valid scopes listed."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.return_value = ["/a.py", "/b.py"]

        request = SimilarityAnalysisRequest(container_name="proj", scope="invalid_scope_value")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 400
        assert "Invalid scope" in exc_info.value.detail

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_file_not_found_raises_404(self, mock_val, mock_resolve):
        """FileNotFoundError raises 404."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.side_effect = FileNotFoundError("file missing")

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 404
        assert "file missing" in exc_info.value.detail

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_not_found_in_message_raises_404(self, mock_val, mock_resolve):
        """Exception with 'not found' in message raises 404."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.side_effect = ValueError("container not found in registry")

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 404

    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_does_not_exist_in_message_raises_404(self, mock_val, mock_resolve):
        """Exception with 'does not exist' in message also raises 404."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        mock_val.return_value = None
        mock_resolve.side_effect = ValueError("path does not exist on disk")

        request = SimilarityAnalysisRequest(container_name="proj", scope="function")
        with pytest.raises(HTTPException) as exc_info:
            await analyze_similarity(request=request, tenant_id="t")
        assert exc_info.value.status_code == 404
        assert "does not exist" in exc_info.value.detail

    @patch(_PROCESS_CLONES)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_clone_detection_with_container_name(
        self, mock_det, mock_validate, mock_resolve, mock_proc
    ):
        """Clone detection using container_name instead of file_paths."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_resolve.return_value = ["/a.py", "/b.py"]
        mock_validate.return_value = ["/a.py", "/b.py"]
        mock_d = AsyncMock()
        mock_d.detect_code_clones = AsyncMock(return_value=[])
        mock_det.return_value = mock_d
        mock_proc.return_value = {
            "total_clones": 0,
            "files_analyzed": 2,
            "clones": [],
            "summary": {},
        }

        request = CloneDetectionRequest(container_name="proj")
        result = await detect_clones(request=request, tenant_id="t")
        assert result.total_clones == 0
        assert result.files_analyzed == 2
        assert result.message == "Clone detection completed: 0 clones found"

    @patch(_VALIDATE_FILES)
    async def test_clone_detection_insufficient_files(self, mock_validate):
        """Clone detection with fewer than 2 valid files returns short response."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/a.py"]  # only 1

        request = CloneDetectionRequest(file_paths=["/a.py"])
        result = await detect_clones(request=request, tenant_id="t")
        assert result.total_clones == 0
        assert "Insufficient" in result.message
        assert result.files_analyzed == 1
        assert result.clone_threshold == 0.85
        assert result.clones == []
        assert result.summary == {}

    @patch(_VALIDATE_FILES)
    async def test_clone_detection_zero_valid_files(self, mock_validate):
        """Clone detection with zero valid files returns short response."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = []

        request = CloneDetectionRequest(file_paths=["/nonexistent.py"])
        result = await detect_clones(request=request, tenant_id="t")
        assert result.total_clones == 0
        assert result.files_analyzed == 0

    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_clone_detection_exception_raises_500(self, mock_det, mock_validate):
        """Unexpected error in clone detection raises 500."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/a.py", "/b.py"]
        mock_d = AsyncMock()
        mock_d.detect_code_clones = AsyncMock(side_effect=RuntimeError("crash"))
        mock_det.return_value = mock_d

        request = CloneDetectionRequest(file_paths=["/a.py", "/b.py"])
        with pytest.raises(HTTPException) as exc_info:
            await detect_clones(request=request, tenant_id="t")
        assert exc_info.value.status_code == 500
        assert "Clone detection failed" in exc_info.value.detail
        assert "crash" in exc_info.value.detail

    @patch(_PROCESS_CROSS)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_cross_lang_with_container_name(
        self, mock_det, mock_validate, mock_resolve, mock_proc
    ):
        """Cross-language detection using container_name."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_resolve.return_value = ["/a.py", "/b.rs"]
        mock_validate.return_value = ["/a.py", "/b.rs"]
        mock_d = AsyncMock()
        mock_d.find_cross_language_patterns = AsyncMock(return_value=[])
        mock_det.return_value = mock_d
        mock_proc.return_value = {
            "total_patterns": 0,
            "files_analyzed": 2,
            "patterns": [],
            "language_pairs": {},
            "summary": {},
        }

        request = CrossLanguagePatternRequest(container_name="proj")
        result = await find_cross_language_patterns(request=request, tenant_id="t")
        assert result.total_patterns == 0
        assert result.files_analyzed == 2
        assert result.language_pairs == {}

    @patch(_VALIDATE_FILES)
    async def test_cross_lang_insufficient_files(self, mock_validate):
        """Cross-language with fewer than 2 valid files returns short response."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_validate.return_value = ["/a.py"]

        request = CrossLanguagePatternRequest(file_paths=["/a.py"])
        result = await find_cross_language_patterns(request=request, tenant_id="t")
        assert result.total_patterns == 0
        assert "Insufficient" in result.message
        assert result.files_analyzed == 1
        assert result.pattern_threshold == 0.60
        assert result.patterns == []
        assert result.language_pairs == {}
        assert result.summary == {}

    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_cross_lang_exception_raises_500(self, mock_det, mock_validate):
        """Unexpected error in cross-language raises 500."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_validate.return_value = ["/a.py", "/b.rs"]
        mock_d = AsyncMock()
        mock_d.find_cross_language_patterns = AsyncMock(side_effect=RuntimeError("crash"))
        mock_det.return_value = mock_d

        request = CrossLanguagePatternRequest(file_paths=["/a.py", "/b.rs"])
        with pytest.raises(HTTPException) as exc_info:
            await find_cross_language_patterns(request=request, tenant_id="t")
        assert exc_info.value.status_code == 500
        assert "Cross-language pattern detection failed" in exc_info.value.detail
        assert "crash" in exc_info.value.detail

    @patch(_GET_CAPS)
    async def test_capabilities_error_raises_500(self, mock_caps):
        """Error in get_capabilities raises 500."""
        mock_caps.side_effect = RuntimeError("config error")
        with pytest.raises(HTTPException) as exc_info:
            await get_similarity_capabilities(tenant_id="t")
        assert exc_info.value.status_code == 500
        assert "Error retrieving capabilities" in exc_info.value.detail
        assert "config error" in exc_info.value.detail

    @patch(_PROCESS_CLONES)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_clone_detection_custom_threshold(self, mock_det, mock_validate, mock_proc):
        """Clone detection uses custom threshold from request."""
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/a.py", "/b.py"]
        mock_d = AsyncMock()
        mock_d.detect_code_clones = AsyncMock(return_value=[])
        mock_det.return_value = mock_d
        mock_proc.return_value = {
            "total_clones": 0,
            "files_analyzed": 2,
            "clones": [],
            "summary": {},
        }

        request = CloneDetectionRequest(file_paths=["/a.py", "/b.py"], clone_threshold=0.95)
        result = await detect_clones(request=request, tenant_id="t")
        assert result.clone_threshold == 0.95
        # Verify the detector was called with the custom threshold
        mock_d.detect_code_clones.assert_called_once_with(
            file_paths=["/a.py", "/b.py"], clone_threshold=0.95
        )

    @patch(_PROCESS_CROSS)
    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_cross_lang_custom_threshold(self, mock_det, mock_validate, mock_proc):
        """Cross-language detection uses custom threshold from request."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_validate.return_value = ["/a.py", "/b.rs"]
        mock_d = AsyncMock()
        mock_d.find_cross_language_patterns = AsyncMock(return_value=[])
        mock_det.return_value = mock_d
        mock_proc.return_value = {
            "total_patterns": 0,
            "files_analyzed": 2,
            "patterns": [],
            "language_pairs": {},
            "summary": {},
        }

        request = CrossLanguagePatternRequest(file_paths=["/a.py", "/b.rs"], pattern_threshold=0.80)
        result = await find_cross_language_patterns(request=request, tenant_id="t")
        assert result.pattern_threshold == 0.80
        mock_d.find_cross_language_patterns.assert_called_once_with(
            file_paths=["/a.py", "/b.rs"], pattern_threshold=0.80
        )

    @patch(_PROCESS_RESULT)
    @patch(_GET_DETECTOR)
    @patch(_RESOLVE_FILES)
    @patch(_VALIDATE_REQ)
    async def test_analyze_passes_all_params_to_detector(
        self, mock_val, mock_resolve, mock_detector, mock_proc
    ):
        """Verify all request params are forwarded to the detector."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest
        from app.api.endpoints.similarity_modules.similarity_processing import (
            SimilarityAnalysisResponseModel,
        )

        mock_val.return_value = None
        mock_resolve.return_value = ["/a.py", "/b.py", "/c.py"]

        mock_result = MagicMock()
        mock_result.total_files_analyzed = 3
        mock_result.total_matches_found = 0
        mock_det = AsyncMock()
        mock_det.analyze_similarity = AsyncMock(return_value=mock_result)
        mock_detector.return_value = mock_det
        mock_proc.return_value = SimilarityAnalysisResponseModel(
            total_files_analyzed=3,
            total_entities_compared=0,
            total_matches_found=0,
            analysis_time_seconds=0.1,
            matches_by_type={},
            matches_by_score={},
            cross_language_matches=[],
            language_distribution={},
            average_similarity_score=0.0,
            highest_similarity_match=None,
            refactoring_opportunities=[],
            duplicate_code_ratio=0.0,
            cross_language_patterns=[],
        )

        request = SimilarityAnalysisRequest(
            container_name="proj",
            scope="file",
            min_similarity_score=0.7,
            detect_cross_language=False,
            max_files=50,
        )
        await analyze_similarity(request=request, tenant_id="t")

        call_kwargs = mock_det.analyze_similarity.call_args.kwargs
        assert call_kwargs["file_paths"] == ["/a.py", "/b.py", "/c.py"]
        assert call_kwargs["min_similarity_score"] == 0.7
        assert call_kwargs["detect_cross_language"] is False
        assert call_kwargs["max_files"] == 50


# ---------------------------------------------------------------------------
# Tests for SimilarityConfig helper methods (no endpoint mocking needed)
# ---------------------------------------------------------------------------


class TestSimilarityConfigHelpers:
    """Direct tests for SimilarityConfig static methods."""

    def test_get_file_language_python(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("/some/path/main.py") == "python"

    def test_get_file_language_java(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("App.java") == "java"

    def test_get_file_language_typescript(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("component.tsx") == "typescript"
        assert SimilarityConfig.get_file_language("utils.ts") == "typescript"

    def test_get_file_language_javascript_jsx(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("app.jsx") == "javascript"
        assert SimilarityConfig.get_file_language("index.js") == "javascript"

    def test_get_file_language_cpp_variants(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("main.cpp") == "cpp"
        assert SimilarityConfig.get_file_language("main.cc") == "cpp"
        assert SimilarityConfig.get_file_language("main.cxx") == "cpp"
        assert SimilarityConfig.get_file_language("header.hpp") == "cpp"

    def test_get_file_language_c_and_header(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("main.c") == "c"
        assert SimilarityConfig.get_file_language("header.h") == "c"

    def test_get_file_language_unknown_extension(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("data.txt") == "unknown"
        assert SimilarityConfig.get_file_language("Makefile") == "unknown"

    def test_get_file_language_case_insensitive(self):
        """Extension matching is case-insensitive via .lower()."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.get_file_language("Main.PY") == "python"
        assert SimilarityConfig.get_file_language("App.JAVA") == "java"

    def test_is_supported_file_true(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.is_supported_file("main.py") is True
        assert SimilarityConfig.is_supported_file("app.java") is True
        assert SimilarityConfig.is_supported_file("lib.rs") is True

    def test_is_supported_file_false(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        assert SimilarityConfig.is_supported_file("readme.md") is False
        assert SimilarityConfig.is_supported_file("data.json") is False
        assert SimilarityConfig.is_supported_file("image.png") is False

    def test_get_default_thresholds_has_expected_keys(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        thresholds = SimilarityConfig.get_default_thresholds()

        assert "exact_duplicates" in thresholds
        assert "near_duplicates" in thresholds
        assert "code_clones" in thresholds
        assert "cross_language_patterns" in thresholds
        # All values are between 0 and 1
        for key, value in thresholds.items():
            assert 0.0 <= value <= 1.0, f"Threshold {key}={value} out of range"
        # exact_duplicates should be the highest threshold
        assert thresholds["exact_duplicates"] >= thresholds["near_duplicates"]
        assert thresholds["near_duplicates"] >= thresholds["structural_similarity"]

    def test_get_performance_recommendations_small(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(10)
        assert rec["max_files_suggestion"] is None
        assert rec["memory_usage"] == "Low"
        assert "Full analysis recommended" in rec["recommendations"]

    def test_get_performance_recommendations_medium(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(100)
        assert rec["memory_usage"] == "Medium"
        assert any("Function-level" in r for r in rec["recommendations"])

    def test_get_performance_recommendations_large(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(500)
        assert rec["max_files_suggestion"] == 500
        assert rec["memory_usage"] == "High"

    def test_get_performance_recommendations_very_large(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(2000)
        assert rec["max_files_suggestion"] == 200
        assert rec["memory_usage"] == "Very High"
        assert rec["recommended_scope"] == "file_level"

    def test_get_performance_recommendations_boundary_49(self):
        """Boundary: 49 files should be 'small' tier."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(49)
        assert rec["memory_usage"] == "Low"

    def test_get_performance_recommendations_boundary_50(self):
        """Boundary: 50 files should be 'medium' tier."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(50)
        assert rec["memory_usage"] == "Medium"

    def test_get_performance_recommendations_boundary_200(self):
        """Boundary: 200 files should be 'large' tier."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(200)
        assert rec["memory_usage"] == "High"

    def test_get_performance_recommendations_boundary_1000(self):
        """Boundary: 1000 files should be 'very large' tier."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        rec = SimilarityConfig.get_performance_recommendations(1000)
        assert rec["memory_usage"] == "Very High"

    def test_validate_analysis_request_missing_both_raises_400(self):
        """validate_analysis_request raises 400 when neither container_name nor file_paths given."""
        from app.api.endpoints.similarity_modules.similarity_config import (
            SimilarityAnalysisRequest,
            SimilarityConfig,
        )

        request = SimilarityAnalysisRequest()
        with pytest.raises(HTTPException) as exc_info:
            SimilarityConfig.validate_analysis_request(request)
        assert exc_info.value.status_code == 400
        assert "container_name" in exc_info.value.detail.lower() or (
            "file_paths" in exc_info.value.detail.lower()
        )

    def test_validate_analysis_request_with_container_passes(self):
        from app.api.endpoints.similarity_modules.similarity_config import (
            SimilarityAnalysisRequest,
            SimilarityConfig,
        )

        request = SimilarityAnalysisRequest(container_name="proj")
        # Should not raise
        SimilarityConfig.validate_analysis_request(request)

    def test_validate_analysis_request_with_file_paths_passes(self):
        from app.api.endpoints.similarity_modules.similarity_config import (
            SimilarityAnalysisRequest,
            SimilarityConfig,
        )

        request = SimilarityAnalysisRequest(file_paths=["/a.py", "/b.py"])
        # Should not raise
        SimilarityConfig.validate_analysis_request(request)

    def test_validate_file_paths_filters_nonexistent(self, tmp_path):
        """validate_file_paths only returns paths that exist on disk."""
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        real_file = tmp_path / "exists.py"
        real_file.write_text("pass")

        result = SimilarityConfig.validate_file_paths([str(real_file), "/this/does/not/exist.py"])
        assert result == [str(real_file)]

    def test_validate_file_paths_all_nonexistent_returns_empty(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        result = SimilarityConfig.validate_file_paths(["/no/such/a.py", "/no/such/b.py"])
        assert result == []

    def test_validate_file_paths_empty_input(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        result = SimilarityConfig.validate_file_paths([])
        assert result == []

    def test_supported_extensions_contains_common_languages(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        exts = SimilarityConfig.SUPPORTED_EXTENSIONS
        assert ".py" in exts
        assert ".java" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".go" in exts
        assert ".rs" in exts

    def test_skip_directories_contains_common_excludes(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityConfig

        skip = SimilarityConfig.SKIP_DIRECTORIES
        assert "node_modules" in skip
        assert "__pycache__" in skip
        assert ".git" in skip
        assert "venv" in skip


# ---------------------------------------------------------------------------
# Tests for SimilarityMetrics utility class
# ---------------------------------------------------------------------------


class TestSimilarityMetrics:
    """Direct tests for SimilarityMetrics math functions."""

    def test_cosine_similarity_identical_vectors(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(result - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_vectors(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-9

    def test_cosine_similarity_empty_vectors(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        assert SimilarityMetrics.cosine_similarity([], []) == 0.0

    def test_cosine_similarity_different_lengths(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        assert SimilarityMetrics.cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        assert SimilarityMetrics.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_jaccard_similarity_identical_sets(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"})
        assert abs(result - 1.0) < 1e-9

    def test_jaccard_similarity_disjoint_sets(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.jaccard_similarity({"a", "b"}, {"c", "d"})
        assert result == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 -> 0.5
        assert abs(result - 0.5) < 1e-9

    def test_jaccard_similarity_both_empty(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        assert SimilarityMetrics.jaccard_similarity(set(), set()) == 1.0

    def test_edit_distance_similarity_identical(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.edit_distance_similarity("hello", "hello")
        assert abs(result - 1.0) < 1e-9

    def test_edit_distance_similarity_completely_different(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.edit_distance_similarity("abc", "xyz")
        # All 3 chars different -> distance=3, max_len=3 -> 1-3/3=0.0
        assert abs(result) < 1e-9

    def test_edit_distance_similarity_empty_strings(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        assert SimilarityMetrics.edit_distance_similarity("", "") == 1.0

    def test_weighted_similarity_default_weights(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        # Default: structural=0.4, semantic=0.4, syntactic=0.2
        result = SimilarityMetrics.weighted_similarity_score(1.0, 1.0, 1.0)
        assert abs(result - 1.0) < 1e-9

    def test_weighted_similarity_custom_weights(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.weighted_similarity_score(
            0.5, 0.0, 0.0, weights={"structural": 1.0, "semantic": 0.0, "syntactic": 0.0}
        )
        assert abs(result - 0.5) < 1e-9

    def test_weighted_similarity_zero_weights(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.weighted_similarity_score(
            0.5, 0.5, 0.5, weights={"structural": 0.0, "semantic": 0.0, "syntactic": 0.0}
        )
        assert result == 0.0

    def test_confidence_score_high_similarity_many_features(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.confidence_score(0.9, 10, 1.0)
        # 0.9*0.6 + 1.0*0.25 + 1.0*0.15 = 0.54 + 0.25 + 0.15 = 0.94
        assert abs(result - 0.94) < 1e-9

    def test_confidence_score_low_features(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityMetrics

        result = SimilarityMetrics.confidence_score(0.9, 1, 1.0)
        # 0.9*0.6 + 0.1*0.25 + 1.0*0.15 = 0.54 + 0.025 + 0.15 = 0.715
        assert abs(result - 0.715) < 1e-9


# ---------------------------------------------------------------------------
# Tests for SimilarityThresholds
# ---------------------------------------------------------------------------


class TestSimilarityThresholds:
    """Tests for threshold categorization and priority logic."""

    def test_categorize_exact_duplicate(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.categorize_by_score(0.95) == "exact_duplicate"
        assert SimilarityThresholds.categorize_by_score(1.0) == "exact_duplicate"

    def test_categorize_near_duplicate(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.categorize_by_score(0.90) == "near_duplicate"
        assert SimilarityThresholds.categorize_by_score(0.85) == "near_duplicate"

    def test_categorize_structural_similar(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.categorize_by_score(0.78) == "structural_similar"

    def test_categorize_low_similarity(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.categorize_by_score(0.3) == "low_similarity"
        assert SimilarityThresholds.categorize_by_score(0.0) == "low_similarity"

    def test_categorize_confidence_levels(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.categorize_confidence(0.9) == "high"
        assert SimilarityThresholds.categorize_confidence(0.7) == "medium"
        assert SimilarityThresholds.categorize_confidence(0.5) == "low"
        assert SimilarityThresholds.categorize_confidence(0.2) == "very_low"

    def test_get_refactoring_priority_critical(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_refactoring_priority(0.95, 0.9, 20) == "critical"

    def test_get_refactoring_priority_high(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_refactoring_priority(0.88, 0.75, 8) == "high"

    def test_get_refactoring_priority_medium(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_refactoring_priority(0.82, 0.65, 3) == "medium"

    def test_get_refactoring_priority_low(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_refactoring_priority(0.72, 0.4, 2) == "low"

    def test_get_refactoring_priority_none(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_refactoring_priority(0.5, 0.3, 1) == "none"

    def test_adjust_threshold_for_scope_function_level(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        # function_level has 0.0 adjustment
        assert SimilarityThresholds.adjust_threshold_for_scope(0.8, "function_level") == 0.8

    def test_adjust_threshold_for_scope_file_level(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        # file_level has +0.05 adjustment (more lenient)
        assert abs(SimilarityThresholds.adjust_threshold_for_scope(0.8, "file_level") - 0.85) < 1e-9

    def test_adjust_threshold_for_scope_block_level(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        # block_level has -0.05 adjustment (more strict)
        assert (
            abs(SimilarityThresholds.adjust_threshold_for_scope(0.8, "block_level") - 0.75) < 1e-9
        )

    def test_adjust_threshold_clamped_to_0_1(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        # Should not go below 0
        assert SimilarityThresholds.adjust_threshold_for_scope(0.02, "block_level") == 0.0
        # Should not go above 1
        assert SimilarityThresholds.adjust_threshold_for_scope(0.98, "file_level") == 1.0

    def test_get_threshold_known_type(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_threshold("exact_duplicate") == 0.95
        assert SimilarityThresholds.get_threshold("code_clone") == 0.85

    def test_get_threshold_unknown_type_returns_default(self):
        from app.api.endpoints.similarity_modules.similarity_metrics import SimilarityThresholds

        assert SimilarityThresholds.get_threshold("nonexistent") == 0.5


# ---------------------------------------------------------------------------
# Tests for Pydantic request model validation
# ---------------------------------------------------------------------------


class TestRequestModelValidation:
    """Test Pydantic model validation on request objects."""

    def test_similarity_analysis_request_defaults(self):
        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        req = SimilarityAnalysisRequest()
        assert req.container_name is None
        assert req.file_paths is None
        assert req.scope == "function"
        assert req.min_similarity_score == 0.5
        assert req.detect_cross_language is True
        assert req.max_files is None

    def test_similarity_analysis_request_score_bounds(self):
        from pydantic import ValidationError

        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        with pytest.raises(ValidationError):
            SimilarityAnalysisRequest(min_similarity_score=1.5)
        with pytest.raises(ValidationError):
            SimilarityAnalysisRequest(min_similarity_score=-0.1)

    def test_similarity_analysis_request_max_files_bounds(self):
        from pydantic import ValidationError

        from app.api.endpoints.similarity_modules.similarity_config import SimilarityAnalysisRequest

        with pytest.raises(ValidationError):
            SimilarityAnalysisRequest(max_files=0)
        with pytest.raises(ValidationError):
            SimilarityAnalysisRequest(max_files=1001)

    def test_clone_detection_request_defaults(self):
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        req = CloneDetectionRequest()
        assert req.container_name is None
        assert req.file_paths is None
        assert req.clone_threshold == 0.85

    def test_clone_detection_request_threshold_bounds(self):
        from pydantic import ValidationError

        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        with pytest.raises(ValidationError):
            CloneDetectionRequest(clone_threshold=0.4)  # below ge=0.5
        with pytest.raises(ValidationError):
            CloneDetectionRequest(clone_threshold=1.1)  # above le=1.0

    def test_cross_language_request_defaults(self):
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        req = CrossLanguagePatternRequest()
        assert req.container_name is None
        assert req.file_paths is None
        assert req.pattern_threshold == 0.60

    def test_cross_language_request_threshold_bounds(self):
        from pydantic import ValidationError

        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        with pytest.raises(ValidationError):
            CrossLanguagePatternRequest(pattern_threshold=0.2)  # below ge=0.3
        with pytest.raises(ValidationError):
            CrossLanguagePatternRequest(pattern_threshold=1.5)  # above le=1.0


def _real_match(
    source_file: str,
    target_file: str,
    *,
    source_language: str = "python",
    target_language: str = "python",
) -> MagicMock:
    """A detector match shaped like services.similarity_detector.SimilarityMatch."""
    m = MagicMock()
    for key, value in dict(
        id="m-1",
        similarity_type=MagicMock(value="exact"),
        scope=MagicMock(value="function"),
        score=0.99,
        confidence=0.95,
        source_file=source_file,
        source_entity="summarize_items",
        source_language=source_language,
        source_lines=(1, 22),
        target_file=target_file,
        target_entity="tally_entries",
        target_language=target_language,
        target_lines=(1, 22),
        matching_features=["structure"],
        differences=[],
        structural_similarity=1.0,
        semantic_similarity=0.9,
        syntactic_similarity=1.0,
        is_cross_language=source_language != target_language,
        summary="identical bodies",
    ).items():
        setattr(m, key, value)
    return m


class TestFoundResultsSerialise:
    """Regression: a run that FOUND something 500'd (models in list[dict] fields).

    Live 2026-10-12, planted clone in cv-m2-hello: "Clone detection failed: 1 validation
    error for CloneDetectionResponse clones.0 Input should be a valid dictionary". The
    processors are deliberately NOT mocked here; mocking them is how this stayed green.
    """

    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_detect_clones_with_a_clone_returns_dicts(self, mock_detector, mock_validate):
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/w/pkg/clone_a.py", "/w/pkg/clone_b.py"]
        det = AsyncMock()
        det.detect_code_clones = AsyncMock(
            return_value=[_real_match("/w/pkg/clone_a.py", "/w/pkg/clone_b.py")]
        )
        mock_detector.return_value = det

        result = await detect_clones(
            request=CloneDetectionRequest(file_paths=["/w/pkg/clone_a.py", "/w/pkg/clone_b.py"]),
            tenant_id="t",
        )

        assert result.total_clones == 1
        assert isinstance(result.clones[0], dict)
        assert {result.clones[0]["source_file"], result.clones[0]["target_file"]} == {
            "/w/pkg/clone_a.py",
            "/w/pkg/clone_b.py",
        }
        assert result.summary["clone_categories"]["exact_clones"] == 1
        result.model_dump_json()  # the wire form must serialise too

    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_detect_clones_with_none_found_stays_empty(self, mock_detector, mock_validate):
        from app.api.endpoints.similarity_modules.similarity_config import CloneDetectionRequest

        mock_validate.return_value = ["/w/a.py", "/w/b.py"]
        det = AsyncMock()
        det.detect_code_clones = AsyncMock(return_value=[])
        mock_detector.return_value = det

        result = await detect_clones(
            request=CloneDetectionRequest(file_paths=["/w/a.py", "/w/b.py"]), tenant_id="t"
        )

        assert result.total_clones == 0
        assert result.clones == []

    @patch(_VALIDATE_FILES)
    @patch(_GET_DETECTOR)
    async def test_cross_language_pattern_found_returns_dicts(self, mock_detector, mock_validate):
        from app.api.endpoints.similarity_modules.similarity_config import (
            CrossLanguagePatternRequest,
        )

        mock_validate.return_value = ["/w/a.py", "/w/b.rs"]
        det = AsyncMock()
        det.find_cross_language_patterns = AsyncMock(
            return_value=[_real_match("/w/a.py", "/w/b.rs", target_language="rust")]
        )
        mock_detector.return_value = det

        result = await find_cross_language_patterns(
            request=CrossLanguagePatternRequest(file_paths=["/w/a.py", "/w/b.rs"]), tenant_id="t"
        )

        assert result.total_patterns == 1
        assert isinstance(result.patterns[0], dict)
        assert result.patterns[0]["target_language"] == "rust"
        result.model_dump_json()
