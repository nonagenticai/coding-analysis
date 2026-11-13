"""Tests for static_analysis.py uncovered paths.

Covers: get_redis, _cache_result,
_store_* functions, _run_validation_analyses, _analyze_files,
_identify_blocking_issues, _generate_recommendations, generate docs endpoints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_VALIDATE = "app.api.endpoints.static_analysis.validate_and_get_container_for_tenant"
_GET_REDIS_SVC = "app.api.endpoints.static_analysis.get_redis_service"


# ---------------------------------------------------------------------------
# get_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    @pytest.mark.asyncio
    async def test_returns_redis_client(self):
        from app.api.endpoints.static_analysis import get_redis

        mock_redis_svc = MagicMock()
        mock_redis_svc.client = MagicMock()
        with patch(_GET_REDIS_SVC, return_value=mock_redis_svc):
            result = await get_redis()
        assert result is mock_redis_svc.client

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        from app.api.endpoints.static_analysis import get_redis

        with patch(_GET_REDIS_SVC, side_effect=ConnectionError("no redis")):
            result = await get_redis()
        assert result is None

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self):
        from app.api.endpoints.static_analysis import get_redis

        with patch(_GET_REDIS_SVC, side_effect=RuntimeError("redis error")):
            result = await get_redis()
        assert result is None


# ---------------------------------------------------------------------------
# _store_call_graph_analysis / _store_signature_analysis / _store_validation_analysis
# ---------------------------------------------------------------------------


class TestStoreAnalyses:
    @pytest.mark.asyncio
    async def test_store_call_graph(self):
        from app.api.endpoints.static_analysis import _store_call_graph_analysis

        result = MagicMock()
        result.analysis_id = "cg-1"
        await _store_call_graph_analysis(result, {})

    @pytest.mark.asyncio
    async def test_store_signature(self):
        from app.api.endpoints.static_analysis import _store_signature_analysis

        result = MagicMock()
        result.analysis_id = "sig-1"
        await _store_signature_analysis(result, {})

    @pytest.mark.asyncio
    async def test_store_validation(self):
        from app.api.endpoints.static_analysis import _store_validation_analysis

        result = MagicMock()
        result.analysis_id = "val-1"
        await _store_validation_analysis(result, {})

    @pytest.mark.asyncio
    async def test_store_call_graph_exception(self):
        """When logging throws OSError, exception is caught silently."""
        from app.api.endpoints.static_analysis import _store_call_graph_analysis

        result = MagicMock()
        result.analysis_id = property(lambda self: (_ for _ in ()).throw(OSError("disk full")))
        # Accessing analysis_id raises OSError
        type(result).analysis_id = property(lambda s: (_ for _ in ()).throw(OSError("disk")))
        # Should not raise
        await _store_call_graph_analysis(result, {})

    @pytest.mark.asyncio
    async def test_store_signature_exception(self):
        """When logging throws RuntimeError, exception is caught silently."""
        from app.api.endpoints.static_analysis import _store_signature_analysis

        result = MagicMock()
        type(result).analysis_id = property(lambda s: (_ for _ in ()).throw(RuntimeError("err")))
        await _store_signature_analysis(result, {})

    @pytest.mark.asyncio
    async def test_store_validation_exception(self):
        """When logging throws OSError, exception is caught silently."""
        from app.api.endpoints.static_analysis import _store_validation_analysis

        result = MagicMock()
        type(result).analysis_id = property(lambda s: (_ for _ in ()).throw(OSError("err")))
        await _store_validation_analysis(result, {})


# ---------------------------------------------------------------------------
# _cache_result
# ---------------------------------------------------------------------------


class TestCacheResult:
    @pytest.mark.asyncio
    async def test_success(self):
        from app.api.endpoints.static_analysis import _cache_result

        mock_redis_svc = MagicMock()
        mock_redis_svc.set = AsyncMock()
        with patch(_GET_REDIS_SVC, return_value=mock_redis_svc):
            await _cache_result(MagicMock(), "key", "value", 3600)
        mock_redis_svc.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_logs_warning(self):
        from app.api.endpoints.static_analysis import _cache_result

        with patch(_GET_REDIS_SVC, side_effect=ConnectionError("no redis")):
            # Should not raise
            await _cache_result(MagicMock(), "key", "value", 3600)


# ---------------------------------------------------------------------------
# _identify_blocking_issues
# ---------------------------------------------------------------------------


class TestIdentifyBlockingIssues:
    def test_no_issues(self):
        from app.api.endpoints.static_analysis import _identify_blocking_issues

        result = _identify_blocking_issues([{"file_path": "a.py"}])
        assert result == []

    def test_with_critical_taint_issue(self):
        from app.api.endpoints.static_analysis import _identify_blocking_issues

        results = [
            {
                "file_path": "a.py",
                "taint_analysis": {
                    "security_issues": [{"severity": "critical", "type": "sql_injection"}]
                },
            }
        ]
        blocking = _identify_blocking_issues(results)
        assert len(blocking) == 1
        assert "sql_injection" in blocking[0]

    def test_low_severity_not_blocking(self):
        from app.api.endpoints.static_analysis import _identify_blocking_issues

        results = [
            {
                "file_path": "a.py",
                "taint_analysis": {"security_issues": [{"severity": "low", "type": "info_leak"}]},
            }
        ]
        blocking = _identify_blocking_issues(results)
        assert len(blocking) == 0


# ---------------------------------------------------------------------------
# _generate_recommendations (static_analysis version)
# ---------------------------------------------------------------------------


class TestStaticGenerateRecommendations:
    def test_low_score_gets_coverage_recommendation(self):
        from app.api.endpoints.static_analysis import _generate_recommendations

        result = _generate_recommendations(0.5, [])
        assert any("coverage" in r.lower() for r in result)

    def test_blocking_issues_gets_security_recommendation(self):
        from app.api.endpoints.static_analysis import _generate_recommendations

        result = _generate_recommendations(0.9, ["critical issue"])
        assert any("security" in r.lower() for r in result)

    def test_safe_code(self):
        from app.api.endpoints.static_analysis import _generate_recommendations

        result = _generate_recommendations(0.9, [])
        assert any("safe for deployment" in r.lower() for r in result)


# ---------------------------------------------------------------------------
# _analyze_files
# ---------------------------------------------------------------------------


class TestAnalyzeFiles:
    def test_empty_files(self):
        from app.api.endpoints.static_analysis import _analyze_files

        result = _analyze_files([], MagicMock(), "/tmp/repo")
        assert result["total_functions"] == 0

    def test_with_functions(self):
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        extractor.extract_comprehensive_signatures = MagicMock(
            return_value={
                "functions": [
                    {"name": "foo", "return_type": "int", "parameters": []},
                    {"name": "bar", "return_type": None, "parameters": [{"type": None}]},
                ],
                "external_dependencies": ["os", "sys"],
            }
        )
        result = _analyze_files([Path("/tmp/test.py")], extractor, "/tmp/repo")
        assert result["total_functions"] == 2
        assert result["typed_functions"] == 1  # foo has return_type
        assert "os" in result["dependencies"]

    def test_file_error_continues(self):
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        extractor.extract_comprehensive_signatures = MagicMock(
            side_effect=RuntimeError("parse error")
        )
        result = _analyze_files([Path("/tmp/bad.py")], extractor, "/tmp/repo")
        assert result["total_functions"] == 0

    def test_typed_function_by_parameter(self):
        """Function is counted as typed if any parameter has a type annotation."""
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        extractor.extract_comprehensive_signatures = MagicMock(
            return_value={
                "functions": [
                    {
                        "name": "baz",
                        "return_type": None,
                        "parameters": [{"name": "x", "type": "int"}],
                    },
                ],
                "external_dependencies": [],
            }
        )
        result = _analyze_files([Path("/tmp/test.py")], extractor, "/tmp/repo")
        assert result["typed_functions"] == 1

    def test_untyped_function(self):
        """Function with no return_type and no param types is not counted as typed."""
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        extractor.extract_comprehensive_signatures = MagicMock(
            return_value={
                "functions": [
                    {
                        "name": "baz",
                        "return_type": None,
                        "parameters": [{"name": "x", "type": None}],
                    },
                ],
            }
        )
        result = _analyze_files([Path("/tmp/test.py")], extractor, "/tmp/repo")
        assert result["typed_functions"] == 0

    def test_multiple_files_accumulate(self):
        """Results accumulate across multiple files."""
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        call_count = [0]

        def mock_extract(file_path, repo_path):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "functions": [{"name": "a", "return_type": "str", "parameters": []}],
                    "external_dependencies": ["os"],
                }
            return {
                "functions": [{"name": "b", "return_type": None, "parameters": []}],
                "external_dependencies": ["sys"],
            }

        extractor.extract_comprehensive_signatures = mock_extract
        result = _analyze_files([Path("/tmp/a.py"), Path("/tmp/b.py")], extractor, "/tmp/repo")
        assert result["total_functions"] == 2
        assert result["typed_functions"] == 1
        assert "os" in result["dependencies"]
        assert "sys" in result["dependencies"]

    def test_value_error_continues(self):
        """ValueError in file analysis is caught and processing continues."""
        from app.api.endpoints.static_analysis import _analyze_files

        extractor = MagicMock()
        extractor.extract_comprehensive_signatures = MagicMock(side_effect=ValueError("bad value"))
        result = _analyze_files([Path("/tmp/bad.py")], extractor, "/tmp/repo")
        assert result["total_functions"] == 0


# ---------------------------------------------------------------------------
# _run_validation_analyses
# ---------------------------------------------------------------------------


class TestRunValidationAnalyses:
    @pytest.mark.asyncio
    async def test_symbolic_execution(self, tmp_path):
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"
        mock_change.modified_code = "x = 1"

        mock_vt = MagicMock()
        mock_vt.value = "symbolic_execution"

        mock_sl = MagicMock()
        mock_sl.value = "medium"

        with patch("app.api.endpoints.static_analysis.SymbolicValidator") as mock_validator_cls:
            mock_instance = MagicMock()
            mock_instance.run_symbolic_analysis_on_file = AsyncMock(
                return_value={
                    "symbolic_execution": {
                        "status": "passed",
                        "edge_cases": ["case1"],
                        "violations": [],
                    }
                }
            )
            mock_validator_cls.return_value = mock_instance

            result = await _run_validation_analyses(str(f), mock_change, [mock_vt], mock_sl)

        assert "symbolic_execution" in result
        assert result["symbolic_execution"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_taint_analysis(self, tmp_path):
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"

        mock_vt = MagicMock()
        mock_vt.value = "taint_analysis"

        mock_sl = MagicMock()
        mock_sl.value = "medium"

        with patch("app.api.endpoints.static_analysis.SymbolicValidator"):
            result = await _run_validation_analyses(str(f), mock_change, [mock_vt], mock_sl)

        assert "taint_analysis" in result
        assert result["taint_analysis"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_formal_verification(self, tmp_path):
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"

        mock_vt = MagicMock()
        mock_vt.value = "formal_verification"

        mock_sl = MagicMock()
        mock_sl.value = "medium"

        with patch("app.api.endpoints.static_analysis.SymbolicValidator"):
            result = await _run_validation_analyses(str(f), mock_change, [mock_vt], mock_sl)

        assert "formal_verification" in result
        # Earned by a real run of FormalVerifier, not hard-coded.
        assert result["formal_verification"]["status"] == "passed"
        assert result["formal_verification"]["coverage"] != 0.92

    @pytest.mark.asyncio
    async def test_symbolic_execution_failure(self, tmp_path):
        """When symbolic analysis fails, result contains 'failed' status."""
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"

        mock_vt = MagicMock()
        mock_vt.value = "symbolic_execution"

        mock_sl = MagicMock()
        mock_sl.value = "medium"

        with patch("app.api.endpoints.static_analysis.SymbolicValidator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.run_symbolic_analysis_on_file = AsyncMock(
                side_effect=RuntimeError("symbolic fail")
            )
            mock_cls.return_value = mock_instance

            result = await _run_validation_analyses(str(f), mock_change, [mock_vt], mock_sl)

        assert result["symbolic_execution"]["status"] == "failed"
        assert "symbolic fail" in result["symbolic_execution"]["reason"]
        # A failed analyzer did not run, so it cannot contribute to a pass.
        assert result["analyzers_ran"] == []

    @pytest.mark.asyncio
    async def test_security_level_without_value_attr(self, tmp_path):
        """When security_level doesn't have .value, defaults to MEDIUM."""
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"

        # security_level without .value attr
        mock_sl = "medium"  # plain string, no .value

        with patch("app.api.endpoints.static_analysis.SymbolicValidator"):
            result = await _run_validation_analyses(str(f), mock_change, [], mock_sl)

        # No analyzers requested means nothing was checked — not a 0.8 baseline.
        assert "safety_score" not in result
        assert result["analyzers_ran"] == []

    @pytest.mark.asyncio
    async def test_validation_config_error(self, tmp_path):
        """When ValidationConfig construction fails, error is caught."""
        from app.api.endpoints.static_analysis import _run_validation_analyses

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        mock_change = MagicMock()
        mock_change.file_path = "test.py"

        mock_vt = MagicMock()
        mock_vt.value = "symbolic_execution"

        mock_sl = MagicMock()
        mock_sl.value = "medium"

        with patch(
            "app.api.endpoints.static_analysis.ValidationConfig",
            side_effect=ValueError("bad config"),
        ):
            result = await _run_validation_analyses(str(f), mock_change, [mock_vt], mock_sl)

        assert result["symbolic_execution"]["status"] == "failed"
        assert "bad config" in result["symbolic_execution"]["reason"]
        assert result["analyzers_ran"] == []


# ---------------------------------------------------------------------------
# _generate_architecture_docs / _generate_api_reference / _generate_change_summary
# ---------------------------------------------------------------------------


class TestGenerateDocFunctions:
    @pytest.mark.asyncio
    async def test_architecture_docs_with_diagrams(self):
        from app.api.endpoints.static_analysis import _generate_architecture_docs

        result = await _generate_architecture_docs("/tmp/repo", True, "markdown")
        assert "Architecture" in result["content"]
        assert len(result["diagrams"]) > 0

    @pytest.mark.asyncio
    async def test_architecture_docs_without_diagrams(self):
        from app.api.endpoints.static_analysis import _generate_architecture_docs

        result = await _generate_architecture_docs("/tmp/repo", False, "markdown")
        assert result["diagrams"] == []

    @pytest.mark.asyncio
    async def test_api_reference(self):
        from app.api.endpoints.static_analysis import _generate_api_reference

        result = await _generate_api_reference("/tmp/repo", "markdown")
        assert "API Reference" in result["content"]

    @pytest.mark.asyncio
    async def test_change_summary(self):
        from app.api.endpoints.static_analysis import _generate_change_summary

        result = await _generate_change_summary("/tmp/repo", "markdown")
        assert "Change Summary" in result["content"]
