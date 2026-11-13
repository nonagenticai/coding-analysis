"""Unit tests for static analysis endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api.endpoints.static_analysis import (
    _analyze_files,
    _generate_recommendations,
    _identify_blocking_issues,
    analyze_signatures,
    generate_call_graph,
    generate_documentation,
    validate_code_changes,
)
from models.static_analysis_models import (
    AdvancedValidationRequest,
    AdvancedValidationResponse,
    CallGraphRequest,
    CallGraphResponse,
    CodeChange,
    DocumentationRequest,
    SignatureAnalysisRequest,
    SignatureAnalysisResponse,
)

_VALIDATE = "app.api.endpoints.static_analysis.validate_and_get_container_for_tenant"
_GET_REPO_PATH = "app.api.endpoints.static_analysis._materialize_workspace_for"
_GET_REDIS = "app.api.endpoints.static_analysis.get_redis"
_GET_REDIS_SVC = "app.api.endpoints.static_analysis.get_redis_service"
_ANALYZE_FILES = "app.api.endpoints.static_analysis._analyze_files"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestIdentifyBlockingIssues:
    """Tests for _identify_blocking_issues helper."""

    def test_no_issues_returns_empty(self):
        """Validation results with no taint analysis produce no blocking issues."""
        result = _identify_blocking_issues(
            [
                {"file_path": "a.py", "safety_score": 0.9},
            ]
        )
        assert result == []

    def test_high_severity_taint_issue_is_blocking(self):
        """A high-severity taint issue should be reported as blocking."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "handler.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "high", "type": "SQL injection"},
                        ],
                    },
                },
            ]
        )
        assert len(result) == 1
        assert "SQL injection" in result[0]
        assert "handler.py" in result[0]

    def test_critical_severity_taint_issue_is_blocking(self):
        """A critical-severity taint issue should be reported as blocking."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "auth.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "critical", "type": "Command injection"},
                        ],
                    },
                },
            ]
        )
        assert len(result) == 1
        assert "Command injection" in result[0]
        assert "auth.py" in result[0]

    def test_low_severity_not_blocking(self):
        """Low-severity issues should NOT be blocking."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "a.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "low", "type": "info leak"},
                        ],
                    },
                },
            ]
        )
        assert result == []

    def test_medium_severity_not_blocking(self):
        """Medium-severity issues should NOT be blocking."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "a.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "medium", "type": "SSRF"},
                        ],
                    },
                },
            ]
        )
        assert result == []

    def test_multiple_files_multiple_issues(self):
        """Multiple high/critical issues across files should all be reported."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "views.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "high", "type": "XSS"},
                            {"severity": "low", "type": "info exposure"},
                        ],
                    },
                },
                {
                    "file_path": "db.py",
                    "taint_analysis": {
                        "security_issues": [
                            {"severity": "critical", "type": "SQL injection"},
                        ],
                    },
                },
            ]
        )
        assert len(result) == 2
        # First issue from views.py (high severity XSS)
        assert any("XSS" in issue and "views.py" in issue for issue in result)
        # Second issue from db.py (critical SQL injection)
        assert any("SQL injection" in issue and "db.py" in issue for issue in result)

    def test_empty_list_returns_empty(self):
        """Empty validation results produce no blocking issues."""
        assert _identify_blocking_issues([]) == []

    def test_taint_analysis_not_a_dict_ignored(self):
        """If taint_analysis is not a dict (e.g. None or string), it is safely skipped."""
        result = _identify_blocking_issues(
            [
                {"file_path": "a.py", "taint_analysis": None},
                {"file_path": "b.py", "taint_analysis": "not a dict"},
            ]
        )
        assert result == []

    def test_missing_type_field_uses_unknown(self):
        """If the issue dict has no 'type' key, falls back to 'Unknown issue'."""
        result = _identify_blocking_issues(
            [
                {
                    "file_path": "x.py",
                    "taint_analysis": {
                        "security_issues": [{"severity": "high"}],
                    },
                },
            ]
        )
        assert len(result) == 1
        assert "Unknown issue" in result[0]


class TestGenerateRecommendations:
    """Tests for _generate_recommendations helper."""

    def test_safe_score_no_issues(self):
        """High safety score and no blocking issues -> safe for deployment."""
        result = _generate_recommendations(0.9, [])
        assert len(result) == 1
        assert "safe for deployment" in result[0].lower()

    def test_low_score_no_issues(self):
        """Low safety score triggers test coverage recommendation."""
        result = _generate_recommendations(0.5, [])
        assert any("test coverage" in r.lower() for r in result)

    def test_blocking_issues_present(self):
        """Blocking issues trigger security recommendation."""
        result = _generate_recommendations(0.9, ["auth.py: SQL injection"])
        assert any("security" in r.lower() for r in result)

    def test_low_score_and_blocking_issues(self):
        """Both low score and blocking issues produce both recommendations."""
        result = _generate_recommendations(0.3, ["file.py: XSS"])
        assert len(result) == 2
        assert any("test coverage" in r.lower() for r in result)
        assert any("security" in r.lower() for r in result)

    def test_boundary_score_exactly_0_7(self):
        """Score of exactly 0.7 should NOT trigger low-score recommendation."""
        result = _generate_recommendations(0.7, [])
        assert len(result) == 1
        assert "safe for deployment" in result[0].lower()

    def test_boundary_score_just_below_0_7(self):
        """Score of 0.69 triggers low-score recommendation."""
        result = _generate_recommendations(0.69, [])
        assert any("test coverage" in r.lower() for r in result)


class TestAnalyzeFiles:
    """Tests for _analyze_files helper."""

    def test_empty_file_list(self):
        """No files to analyze returns zeros."""
        extractor = MagicMock()
        result = _analyze_files([], extractor, "/repo")
        assert result["functions"] == []
        assert result["dependencies"] == set()
        assert result["total_functions"] == 0
        assert result["typed_functions"] == 0
        extractor.extract_comprehensive_signatures.assert_not_called()

    def test_single_file_with_typed_function(self):
        """A file with one typed function is counted correctly."""
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.return_value = {
            "functions": [
                {
                    "name": "greet",
                    "module": "hello",
                    "return_type": "str",
                    "parameters": [{"name": "name", "type": "str"}],
                },
            ],
            "external_dependencies": ["os"],
        }
        files = [Path("/repo/hello.py")]
        result = _analyze_files(files, extractor, "/repo")
        assert result["total_functions"] == 1
        assert result["typed_functions"] == 1
        assert result["functions"][0]["name"] == "greet"
        assert "os" in result["dependencies"]

    def test_single_file_with_untyped_function(self):
        """A function with no return_type and no typed params is not counted as typed."""
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.return_value = {
            "functions": [
                {
                    "name": "do_stuff",
                    "module": "main",
                    "return_type": None,
                    "parameters": [{"name": "x"}],
                },
            ],
        }
        files = [Path("/repo/main.py")]
        result = _analyze_files(files, extractor, "/repo")
        assert result["total_functions"] == 1
        assert result["typed_functions"] == 0

    def test_multiple_files_aggregation(self):
        """Functions and dependencies from multiple files are aggregated."""
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.side_effect = [
            {
                "functions": [
                    {"name": "a", "module": "m1", "return_type": "int", "parameters": []},
                ],
                "external_dependencies": ["numpy"],
            },
            {
                "functions": [
                    {"name": "b", "module": "m2", "return_type": None, "parameters": []},
                    {"name": "c", "module": "m2", "return_type": "str", "parameters": []},
                ],
                "external_dependencies": ["pandas", "numpy"],
            },
        ]
        files = [Path("/repo/m1.py"), Path("/repo/m2.py")]
        result = _analyze_files(files, extractor, "/repo")
        assert result["total_functions"] == 3
        assert result["typed_functions"] == 2  # a (return_type=int), c (return_type=str)
        assert result["dependencies"] == {"numpy", "pandas"}

    def test_file_error_is_skipped(self):
        """If a file raises an error during extraction, it is skipped."""
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.side_effect = [
            OSError("permission denied"),
            {
                "functions": [
                    {"name": "ok", "module": "good", "return_type": "bool", "parameters": []},
                ],
            },
        ]
        files = [Path("/repo/bad.py"), Path("/repo/good.py")]
        result = _analyze_files(files, extractor, "/repo")
        assert result["total_functions"] == 1
        assert result["functions"][0]["name"] == "ok"


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestStaticAnalysisEndpoints:
    """Tests for static analysis endpoints."""

    # ---- Call graph success + error ----

    @patch("app.api.endpoints.static_analysis.CallGraphGenerator")
    @patch(_GET_REPO_PATH)
    @patch(_GET_REDIS)
    @patch(_VALIDATE)
    async def test_call_graph_generation_success(
        self, mock_validate, mock_redis, mock_repo, mock_cgg
    ):
        mock_validate.return_value = MagicMock()
        mock_redis.return_value = None  # no cache
        mock_repo.return_value = Path("/tmp/test-repo")

        mock_result = MagicMock(spec=CallGraphResponse)
        mock_result.analysis_id = "cg_123"
        mock_result.call_graph = {"main": ["helper"]}
        mock_result.critical_paths = ["main -> helper"]
        mock_result.metrics = MagicMock(total_functions=3)
        mock_result.model_dump_json.return_value = "{}"
        mock_analyzer = AsyncMock()
        mock_analyzer.generate_comprehensive_call_graph = AsyncMock(return_value=mock_result)
        mock_cgg.return_value = mock_analyzer

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        result = await generate_call_graph(
            request=request, background_tasks=bg, redis_client=None, tenant_id="t"
        )
        assert result.analysis_id == "cg_123"
        assert result.call_graph == {"main": ["helper"]}
        assert result.critical_paths == ["main -> helper"]
        assert result.metrics.total_functions == 3

    @patch(_VALIDATE)
    async def test_invalid_code_returns_error(self, mock_validate):
        """When validate_and_get_container_for_tenant raises 404, it propagates."""
        mock_validate.side_effect = HTTPException(status_code=404, detail="not found")

        request = CallGraphRequest(container_name="missing")
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await generate_call_graph(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 404

    @patch("app.api.endpoints.static_analysis.CallGraphGenerator")
    @patch(_GET_REPO_PATH)
    @patch(_GET_REDIS)
    @patch(_VALIDATE)
    async def test_analyzer_failure_returns_error(
        self, mock_validate, mock_redis, mock_repo, mock_cgg
    ):
        mock_validate.return_value = MagicMock()
        mock_redis.return_value = None
        mock_repo.return_value = Path("/tmp/test-repo")

        mock_analyzer = AsyncMock()
        mock_analyzer.generate_comprehensive_call_graph = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        mock_cgg.return_value = mock_analyzer

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await generate_call_graph(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 500

    # ---- Call graph: Redis cache hit paths ----

    @patch(_GET_REDIS_SVC)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_call_graph_redis_cache_hit_dict(
        self, mock_validate, mock_repo, mock_get_redis_svc
    ):
        """When redis has a cached dict, real model validation deserializes it."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        cached_dict = {
            "analysis_id": "cg_cached",
            "call_graph": {"main": ["helper", "utils"]},
            "critical_paths": ["main -> helper"],
            "orphaned_functions": ["unused_func"],
            "metrics": {
                "total_functions": 5,
                "max_call_depth": 3,
                "cyclic_dependencies": [],
            },
        }
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_dict)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        result = await generate_call_graph(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, CallGraphResponse)
        assert result.analysis_id == "cg_cached"
        assert result.call_graph == {"main": ["helper", "utils"]}
        assert result.critical_paths == ["main -> helper"]
        assert result.orphaned_functions == ["unused_func"]
        assert result.metrics.total_functions == 5
        assert result.metrics.max_call_depth == 3
        assert result.metrics.cyclic_dependencies == []

    @patch(_GET_REDIS_SVC)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_call_graph_redis_cache_hit_json_string(
        self, mock_validate, mock_repo, mock_get_redis_svc
    ):
        """When redis has a cached JSON string, real model_validate_json deserializes it."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        cached_data = {
            "analysis_id": "cg_json_cached",
            "call_graph": {"entry": ["a", "b"]},
            "visual_output": None,
            "critical_paths": ["entry -> a -> b"],
            "orphaned_functions": [],
            "metrics": {
                "total_functions": 10,
                "max_call_depth": 4,
                "cyclic_dependencies": ["a <-> b"],
            },
        }
        cached_json = json.dumps(cached_data)
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_json)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        result = await generate_call_graph(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, CallGraphResponse)
        assert result.analysis_id == "cg_json_cached"
        assert result.call_graph == {"entry": ["a", "b"]}
        assert result.critical_paths == ["entry -> a -> b"]
        assert result.metrics.total_functions == 10
        assert result.metrics.max_call_depth == 4
        assert result.metrics.cyclic_dependencies == ["a <-> b"]

    @patch(_GET_REDIS_SVC)
    @patch("app.api.endpoints.static_analysis.CallGraphGenerator")
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_call_graph_redis_cache_error_continues(
        self, mock_validate, mock_repo, mock_cgg, mock_get_redis_svc
    ):
        """When redis cache lookup fails, call graph generation continues normally."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(side_effect=ConnectionError("redis down"))
        mock_get_redis_svc.return_value = mock_redis_svc

        mock_result = MagicMock(spec=CallGraphResponse)
        mock_result.analysis_id = "cg_fresh"
        mock_result.model_dump_json.return_value = "{}"
        mock_analyzer = AsyncMock()
        mock_analyzer.generate_comprehensive_call_graph = AsyncMock(return_value=mock_result)
        mock_cgg.return_value = mock_analyzer

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        result = await generate_call_graph(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )
        assert result.analysis_id == "cg_fresh"

    @patch(_GET_REDIS_SVC)
    @patch("app.api.endpoints.static_analysis.CallGraphGenerator")
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_call_graph_with_redis_caches_result(
        self, mock_validate, mock_repo, mock_cgg, mock_get_redis_svc
    ):
        """When redis_client is provided and cache misses, result is cached via background task."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        # Redis cache miss
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=None)
        mock_get_redis_svc.return_value = mock_redis_svc

        mock_result = MagicMock(spec=CallGraphResponse)
        mock_result.analysis_id = "cg_new"
        mock_result.model_dump_json.return_value = '{"analysis_id": "cg_new"}'
        mock_analyzer = AsyncMock()
        mock_analyzer.generate_comprehensive_call_graph = AsyncMock(return_value=mock_result)
        mock_cgg.return_value = mock_analyzer

        request = CallGraphRequest(container_name="proj")
        bg = BackgroundTasks()
        result = await generate_call_graph(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )
        assert result.analysis_id == "cg_new"
        # BackgroundTasks should have tasks added (cache + store + cleanup)
        assert len(bg.tasks) >= 2

    # ---- Signature endpoint: success + redis cache + error paths ----

    @patch(_ANALYZE_FILES)
    @patch(_GET_REPO_PATH)
    @patch(_GET_REDIS)
    @patch(_VALIDATE)
    async def test_signature_analysis_success(self, mock_validate, mock_redis, mock_repo, mock_af):
        mock_validate.return_value = MagicMock()
        mock_redis.return_value = None
        mock_repo.return_value = Path("/tmp/test-repo")
        mock_af.return_value = {
            "functions": [
                {
                    "name": "foo",
                    "module": "main",
                    "signature": "def foo() -> int",
                    "return_type": "int",
                    "parameters": [],
                    "complexity_metrics": {"cyclomatic": 1, "cognitive": 0},
                }
            ],
            "dependencies": set(),
            "total_functions": 1,
            "typed_functions": 1,
        }

        request = SignatureAnalysisRequest(container_name="proj", language="python")
        bg = BackgroundTasks()
        result = await analyze_signatures(
            request=request, background_tasks=bg, redis_client=None, tenant_id="t"
        )
        assert isinstance(result, SignatureAnalysisResponse)
        assert result.total_functions == 1
        assert result.typed_functions == 1
        assert result.type_coverage == 1.0
        assert len(result.functions) == 1
        assert result.functions[0].name == "foo"
        assert result.functions[0].return_type == "int"

    @patch(_GET_REDIS_SVC)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_signature_redis_cache_hit_dict(
        self, mock_validate, mock_repo, mock_get_redis_svc
    ):
        """When redis has a cached dict for signatures, real model_validate deserializes it."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        cached_dict = {
            "analysis_id": "sig_cached",
            "functions": [
                {
                    "name": "bar",
                    "module": "utils",
                    "signature": "def bar(x: int) -> str",
                    "return_type": "str",
                    "parameters": [
                        {"name": "x", "type": "int", "required": True},
                    ],
                    "complexity_metrics": {"cyclomatic": 2, "cognitive": 1},
                },
            ],
            "type_coverage": 0.85,
            "external_dependencies": ["requests"],
            "total_functions": 10,
            "typed_functions": 8,
        }
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_dict)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = SignatureAnalysisRequest(container_name="proj", language="python")
        bg = BackgroundTasks()
        result = await analyze_signatures(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, SignatureAnalysisResponse)
        assert result.analysis_id == "sig_cached"
        assert result.total_functions == 10
        assert result.typed_functions == 8
        assert result.type_coverage == 0.85
        assert len(result.functions) == 1
        assert result.functions[0].name == "bar"
        assert result.functions[0].return_type == "str"
        assert result.external_dependencies == ["requests"]

    @patch(_GET_REDIS_SVC)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_signature_redis_cache_hit_json_string(
        self, mock_validate, mock_repo, mock_get_redis_svc
    ):
        """When redis has a cached JSON string for signatures, real model_validate_json works."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        cached_data = {
            "analysis_id": "sig_json_cached",
            "functions": [],
            "type_coverage": 0.0,
            "external_dependencies": [],
            "total_functions": 0,
            "typed_functions": 0,
        }
        cached_json = json.dumps(cached_data)
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_json)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = SignatureAnalysisRequest(container_name="proj", language="python")
        bg = BackgroundTasks()
        result = await analyze_signatures(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, SignatureAnalysisResponse)
        assert result.analysis_id == "sig_json_cached"
        assert result.total_functions == 0
        assert result.functions == []

    @patch(_ANALYZE_FILES)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_signature_analysis_exception_returns_500(
        self, mock_validate, mock_repo, mock_af
    ):
        """When signature analysis raises an unexpected exception, returns 500."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")
        mock_af.side_effect = RuntimeError("analysis boom")

        request = SignatureAnalysisRequest(container_name="proj", language="python")
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await analyze_signatures(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 500

    @patch(_ANALYZE_FILES)
    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_signature_analysis_js_files(self, mock_validate, mock_repo, mock_af):
        """JavaScript is refused: the extractor is Python-only, and listing .js/.ts
        files here answered "0 functions" -- a vacuous result, not an analysis."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        request = SignatureAnalysisRequest(container_name="proj", language="javascript")
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await analyze_signatures(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 400
        mock_af.assert_not_called()

    # ---- Advanced validation: success + redis cache + error paths ----

    @patch("app.api.endpoints.static_analysis._run_validation_analyses")
    @patch(_GET_REDIS)
    @patch(_VALIDATE)
    async def test_advanced_validation_success(self, mock_validate, mock_redis, mock_run):
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")
        mock_redis.return_value = None
        # The per-file score is no longer supplied by the analyzer stage; it is
        # derived from the analyzers that reported a result.
        mock_run.return_value = {
            "file_path": "main.py",
            "analyzers_ran": ["taint_analysis"],
            "analyzers_not_computed": [],
            "symbolic_execution": None,
            "taint_analysis": {"status": "completed", "security_issues": []},
            "formal_verification": None,
        }

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(
                    file_path="main.py", modified_code="print('hello')", change_type="MODIFIED"
                )
            ],
            validation_types=["taint_analysis"],
            security_level="medium",
        )
        bg = BackgroundTasks()
        result = await validate_code_changes(
            request=request, background_tasks=bg, redis_client=None, tenant_id="t"
        )
        assert isinstance(result, AdvancedValidationResponse)
        assert result.verdict == "pass"
        assert result.overall_safety_score == 1.0
        assert len(result.validation_results) == 1
        assert result.validation_results[0].file_path == "main.py"
        assert result.validation_results[0].safety_score == 1.0
        assert result.analyzers_ran == ["taint_analysis"]

    @patch(_GET_REDIS_SVC)
    @patch(_VALIDATE)
    async def test_validation_redis_cache_hit_dict(self, mock_validate, mock_get_redis_svc):
        """When redis has a cached dict for validation, real model_validate deserializes it."""
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")

        cached_dict = {
            "analysis_id": "adv_cached",
            "validation_results": [
                {
                    "file_path": "main.py",
                    "symbolic_execution": None,
                    "taint_analysis": None,
                    "formal_verification": None,
                    "safety_score": 0.88,
                },
            ],
            "overall_safety_score": 0.88,
            "blocking_issues": [],
            "recommendations": ["Code changes appear safe for deployment"],
        }
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_dict)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(file_path="main.py", modified_code="print('hi')", change_type="MODIFIED")
            ],
            validation_types=[],
            security_level="medium",
        )
        bg = BackgroundTasks()
        result = await validate_code_changes(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, AdvancedValidationResponse)
        assert result.analysis_id == "adv_cached"
        assert result.overall_safety_score == 0.88
        assert len(result.validation_results) == 1
        assert result.validation_results[0].file_path == "main.py"
        assert result.validation_results[0].safety_score == 0.88
        assert result.blocking_issues == []
        assert "safe for deployment" in result.recommendations[0].lower()

    @patch(_GET_REDIS_SVC)
    @patch(_VALIDATE)
    async def test_validation_redis_cache_hit_json_string(self, mock_validate, mock_get_redis_svc):
        """When redis has a cached JSON string for validation, real model_validate_json works."""
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")

        cached_data = {
            "analysis_id": "adv_json_cached",
            "validation_results": [
                {
                    "file_path": "handler.py",
                    "symbolic_execution": {
                        "status": "completed",
                        "edge_cases_found": ["division by zero"],
                        "assertion_violations": [],
                    },
                    "taint_analysis": None,
                    "formal_verification": None,
                    "safety_score": 0.75,
                },
            ],
            "overall_safety_score": 0.75,
            "blocking_issues": [],
            "recommendations": ["Increase test coverage for modified functions"],
        }
        cached_json = json.dumps(cached_data)
        mock_redis_svc = MagicMock()
        mock_redis_svc.get = AsyncMock(return_value=cached_json)
        mock_get_redis_svc.return_value = mock_redis_svc

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(
                    file_path="handler.py", modified_code="print('hi')", change_type="MODIFIED"
                )
            ],
            validation_types=[],
            security_level="medium",
        )
        bg = BackgroundTasks()
        result = await validate_code_changes(
            request=request, background_tasks=bg, redis_client=MagicMock(), tenant_id="t"
        )

        assert isinstance(result, AdvancedValidationResponse)
        assert result.analysis_id == "adv_json_cached"
        assert result.overall_safety_score == 0.75
        assert result.validation_results[0].file_path == "handler.py"
        assert result.validation_results[0].symbolic_execution.status == "completed"
        assert result.validation_results[0].symbolic_execution.edge_cases_found == [
            "division by zero"
        ]

    # ---- Security tests ----

    @patch(_VALIDATE)
    async def test_validation_absolute_path_rejected(self, mock_validate):
        """When code_changes has an absolute file path, returns 400."""
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(
                    file_path="/absolute/path.py",
                    modified_code="print('hi')",
                    change_type="MODIFIED",
                )
            ],
            validation_types=[],
            security_level="medium",
        )
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await validate_code_changes(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 400

    @patch(_VALIDATE)
    async def test_validation_path_outside_bounds_rejected(self, mock_validate):
        """When code_changes file path traverses outside container, returns 400."""
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(
                    file_path="../../etc/passwd",
                    modified_code="print('hi')",
                    change_type="MODIFIED",
                )
            ],
            validation_types=[],
            security_level="medium",
        )
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await validate_code_changes(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 400

    # ---- Validation error handling ----

    @patch("app.api.endpoints.static_analysis._run_validation_analyses")
    @patch(_VALIDATE)
    async def test_validation_per_change_error_handling(self, mock_validate, mock_run):
        """A change that could not be validated abstains — it is neither 0 nor a pass.

        Scoring it 0.0 read as "measured, and dangerous"; scoring it anything
        else read as "measured, and fine". Nothing was measured, so the score is
        None and the verdict is an abstention.
        """
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")
        mock_run.side_effect = OSError("file error")

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(file_path="main.py", modified_code="print('hi')", change_type="MODIFIED")
            ],
            validation_types=["taint_analysis"],
            security_level="medium",
        )
        bg = BackgroundTasks()
        result = await validate_code_changes(
            request=request, background_tasks=bg, redis_client=None, tenant_id="t"
        )
        assert result.overall_safety_score is None
        assert result.verdict == "abstain"
        assert [a.reason for a in result.analyzers_not_computed] == ["file error"]

    @patch(_VALIDATE)
    async def test_validation_generic_exception_returns_500(self, mock_validate):
        """When validate_code_changes raises unexpected error, returns 500."""
        mock_validate.side_effect = ValueError("unexpected")

        request = AdvancedValidationRequest(
            container_name="proj",
            code_changes=[
                CodeChange(file_path="main.py", modified_code="print('hi')", change_type="MODIFIED")
            ],
            validation_types=[],
            security_level="medium",
        )
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await validate_code_changes(
                request=request, background_tasks=bg, redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 500

    # ---- Documentation endpoint ----

    @patch(_GET_REPO_PATH)
    @patch(_GET_REDIS)
    @patch(_VALIDATE)
    async def test_documentation_generation_success(self, mock_validate, mock_redis, mock_repo):
        mock_validate.return_value = MagicMock()
        mock_redis.return_value = None
        mock_repo.return_value = Path("/tmp/test-repo")

        request = DocumentationRequest(container_name="proj", doc_types=["architecture"])
        bg = BackgroundTasks()
        result = await generate_documentation(
            request=request, background_tasks=bg, _redis_client=None, tenant_id="t"
        )
        assert result.analysis_id
        assert result.quality_score >= 0
        assert "architecture" in result.documentation
        assert result.documentation["architecture"].content

    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_documentation_api_reference_and_change_summary(self, mock_validate, mock_repo):
        """Test documentation generation with api_reference and change_summary doc types."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        request = DocumentationRequest(
            container_name="proj",
            doc_types=["architecture", "api_reference", "change_summary"],
        )
        bg = BackgroundTasks()
        result = await generate_documentation(
            request=request, background_tasks=bg, _redis_client=None, tenant_id="t"
        )
        assert "architecture" in result.documentation
        assert "api_reference" in result.documentation
        assert "change_summary" in result.documentation
        assert result.quality_score > 0
        assert len(result.generated_files) >= 3

    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_documentation_doc_type_error_handling(self, mock_validate, mock_repo):
        """When a doc type raises, it is captured as error section."""
        mock_validate.return_value = MagicMock()
        mock_repo.return_value = Path("/tmp/test-repo")

        with patch(
            "app.api.endpoints.static_analysis._generate_architecture_docs",
            side_effect=RuntimeError("doc gen failed"),
        ):
            request = DocumentationRequest(container_name="proj", doc_types=["architecture"])
            bg = BackgroundTasks()
            result = await generate_documentation(
                request=request, background_tasks=bg, _redis_client=None, tenant_id="t"
            )
        assert "architecture" in result.documentation
        doc = result.documentation["architecture"]
        assert "Error" in doc.content

    @patch(_GET_REPO_PATH)
    @patch(_VALIDATE)
    async def test_documentation_generic_exception_returns_500(self, mock_validate, mock_repo):
        """When generate_documentation raises unexpected error, returns 500."""
        mock_repo.side_effect = ValueError("unexpected doc error")

        request = DocumentationRequest(container_name="proj", doc_types=["architecture"])
        bg = BackgroundTasks()
        with pytest.raises(HTTPException) as exc_info:
            await generate_documentation(
                request=request, background_tasks=bg, _redis_client=None, tenant_id="t"
            )
        assert exc_info.value.status_code == 500
