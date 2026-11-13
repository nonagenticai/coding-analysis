"""Tests for static_analysis.py helper functions — covers _analyze_files, _identify_blocking_issues,
_generate_recommendations, _cache_result, _store_* functions and
_run_validation_analyses."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class _FakeSecurityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _FakeValidationType(StrEnum):
    SYMBOLIC_EXECUTION = "symbolic_execution"
    TAINT_ANALYSIS = "taint_analysis"
    FORMAL_VERIFICATION = "formal_verification"


# Import functions under test
from app.api.endpoints.static_analysis import (
    _analyze_files,
    _cache_result,
    _generate_api_reference,
    _generate_architecture_docs,
    _generate_change_summary,
    _generate_recommendations,
    _identify_blocking_issues,
    _run_validation_analyses,
    _store_call_graph_analysis,
    _store_signature_analysis,
    _store_validation_analysis,
)

# ---------------------------------------------------------------------------
# Tests for _analyze_files
# ---------------------------------------------------------------------------


class TestAnalyzeFiles:
    """Tests for _analyze_files helper."""

    def test_empty_files(self):
        result = _analyze_files([], MagicMock(), "/tmp")
        assert result["functions"] == []
        assert result["total_functions"] == 0
        assert result["typed_functions"] == 0

    def test_with_files(self):
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.return_value = {
            "functions": [
                {"name": "foo", "return_type": "int", "parameters": []},
                {"name": "bar", "return_type": None, "parameters": [{"type": None}]},
            ],
            "external_dependencies": ["os", "sys"],
        }
        files = [Path("/tmp/a.py")]
        result = _analyze_files(files, extractor, "/tmp")
        assert result["total_functions"] == 2
        assert result["typed_functions"] == 1
        assert "os" in result["dependencies"]

    def test_file_error_skipped(self):
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.side_effect = RuntimeError("parse error")
        files = [Path("/tmp/a.py")]
        result = _analyze_files(files, extractor, "/tmp")
        assert result["total_functions"] == 0

    def test_typed_function_with_param_type(self):
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.return_value = {
            "functions": [
                {"name": "foo", "return_type": None, "parameters": [{"type": "str"}]},
            ],
        }
        result = _analyze_files([Path("/tmp/x.py")], extractor, "/tmp")
        assert result["typed_functions"] == 1

    def test_untyped_function(self):
        extractor = MagicMock()
        extractor.extract_comprehensive_signatures.return_value = {
            "functions": [
                {"name": "foo", "return_type": None, "parameters": []},
            ],
        }
        result = _analyze_files([Path("/tmp/x.py")], extractor, "/tmp")
        assert result["typed_functions"] == 0


# ---------------------------------------------------------------------------
# Tests for _identify_blocking_issues
# ---------------------------------------------------------------------------


class TestIdentifyBlockingIssues:
    """Tests for _identify_blocking_issues."""

    def test_no_issues(self):
        results = [{"taint_analysis": {"security_issues": []}}]
        assert _identify_blocking_issues(results) == []

    def test_high_severity_issue(self):
        results = [
            {
                "file_path": "main.py",
                "taint_analysis": {
                    "security_issues": [{"severity": "high", "type": "SQL Injection"}]
                },
            }
        ]
        issues = _identify_blocking_issues(results)
        assert len(issues) == 1
        assert "SQL Injection" in issues[0]

    def test_critical_severity_issue(self):
        results = [
            {
                "file_path": "auth.py",
                "taint_analysis": {"security_issues": [{"severity": "critical", "type": "RCE"}]},
            }
        ]
        issues = _identify_blocking_issues(results)
        assert len(issues) == 1

    def test_low_severity_ignored(self):
        results = [
            {
                "file_path": "utils.py",
                "taint_analysis": {"security_issues": [{"severity": "low", "type": "Info leak"}]},
            }
        ]
        assert _identify_blocking_issues(results) == []

    def test_no_taint_analysis(self):
        results = [{"file_path": "main.py"}]
        assert _identify_blocking_issues(results) == []

    def test_taint_analysis_not_dict(self):
        results = [{"taint_analysis": "error"}]
        assert _identify_blocking_issues(results) == []


# ---------------------------------------------------------------------------
# Tests for _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    """Tests for _generate_recommendations."""

    def test_low_safety_score(self):
        recs = _generate_recommendations(0.5, [])
        assert any("test coverage" in r.lower() for r in recs)

    def test_blocking_issues(self):
        recs = _generate_recommendations(0.9, ["Critical issue"])
        assert any("security" in r.lower() for r in recs)

    def test_no_issues_high_score(self):
        recs = _generate_recommendations(0.95, [])
        assert any("safe" in r.lower() for r in recs)

    def test_both_low_score_and_blocking(self):
        recs = _generate_recommendations(0.3, ["Issue"])
        assert len(recs) == 2


# ---------------------------------------------------------------------------
# Tests for _cache_result and _store_* functions
# ---------------------------------------------------------------------------


class TestCacheAndStoreResults:
    """Tests for caching and storage functions."""

    async def test_cache_result_success(self):
        mock_redis_svc = MagicMock()
        mock_redis_svc.set = AsyncMock()
        with patch(
            "app.api.endpoints.static_analysis.get_redis_service", return_value=mock_redis_svc
        ):
            await _cache_result(MagicMock(), "key", "value", 3600)
            mock_redis_svc.set.assert_called_once()

    async def test_cache_result_connection_error(self):
        mock_redis_svc = MagicMock()
        mock_redis_svc.set = AsyncMock(side_effect=ConnectionError("fail"))
        with patch(
            "app.api.endpoints.static_analysis.get_redis_service", return_value=mock_redis_svc
        ):
            await _cache_result(MagicMock(), "key", "value", 3600)

    async def test_store_call_graph(self):
        result = MagicMock(analysis_id="cg-123")
        await _store_call_graph_analysis(result, {})

    async def test_store_signature(self):
        result = MagicMock(analysis_id="sig-123")
        await _store_signature_analysis(result, {})

    async def test_store_validation(self):
        result = MagicMock(analysis_id="val-123")
        await _store_validation_analysis(result, {})


# ---------------------------------------------------------------------------
# Tests for _run_validation_analyses
# ---------------------------------------------------------------------------


class TestRunValidationAnalyses:
    """Tests for _run_validation_analyses.

    These used to assert the fabricated constants (safety_score 0.8, taint
    'completed' with no issues, formal verification 'passed'). They now assert
    the opposite property: nothing is defaulted, and an analyzer that did not
    produce a result is filed as an abstention instead of a pass.
    """

    async def test_no_analyzers_requested_produces_no_score(self):
        change = MagicMock(file_path="main.py")
        result = await _run_validation_analyses(
            "/tmp/test.py", change, [], _FakeSecurityLevel.MEDIUM
        )
        assert result["file_path"] == "main.py"
        # There is no baseline safety score any more; an unchecked change is
        # not 0.8 safe, it is unchecked.
        assert "safety_score" not in result
        assert result["analyzers_ran"] == []
        assert result["analyzers_not_computed"][0]["analyzer"] == "none"

    async def test_with_symbolic_execution(self):
        vt = _FakeValidationType.SYMBOLIC_EXECUTION
        change = MagicMock(file_path="main.py")
        mock_analyzer = AsyncMock()
        # run_symbolic_analysis_on_file nests its findings under the analyzer name.
        mock_analyzer.run_symbolic_analysis_on_file = AsyncMock(
            return_value={
                "symbolic_execution": {
                    "status": "passed",
                    "edge_cases": ["case1"],
                    "violations": [],
                }
            }
        )
        with (
            patch("app.api.endpoints.static_analysis.ValidationConfig", lambda **kw: kw),
            patch("app.api.endpoints.static_analysis.ValidationType", _FakeValidationType),
            patch("app.api.endpoints.static_analysis.SecurityLevel", _FakeSecurityLevel),
            patch(
                "app.api.endpoints.static_analysis.SymbolicValidator", return_value=mock_analyzer
            ),
        ):
            result = await _run_validation_analyses(
                "/tmp/test.py", change, [vt], _FakeSecurityLevel.MEDIUM
            )
        assert result["symbolic_execution"]["status"] == "completed"
        assert result["symbolic_execution"]["edge_cases_found"] == ["case1"]
        assert result["analyzers_ran"] == ["symbolic_execution"]

    async def test_symbolic_execution_reports_violations_it_is_given(self):
        """The old code read `.get("violations")` off the OUTER dict — always empty."""
        vt = _FakeValidationType.SYMBOLIC_EXECUTION
        change = MagicMock(file_path="main.py")
        mock_analyzer = AsyncMock()
        mock_analyzer.run_symbolic_analysis_on_file = AsyncMock(
            return_value={
                "symbolic_execution": {
                    "status": "failed",
                    "edge_cases": [],
                    "violations": ["assert x > 0 fails for x=-1"],
                }
            }
        )
        with (
            patch("app.api.endpoints.static_analysis.ValidationConfig", lambda **kw: kw),
            patch("app.api.endpoints.static_analysis.ValidationType", _FakeValidationType),
            patch("app.api.endpoints.static_analysis.SecurityLevel", _FakeSecurityLevel),
            patch(
                "app.api.endpoints.static_analysis.SymbolicValidator", return_value=mock_analyzer
            ),
        ):
            result = await _run_validation_analyses(
                "/tmp/test.py", change, [vt], _FakeSecurityLevel.MEDIUM
            )
        assert result["symbolic_execution"]["assertion_violations"] == [
            "assert x > 0 fails for x=-1"
        ]

    async def test_symbolic_execution_tool_missing_is_not_computed(self):
        vt = _FakeValidationType.SYMBOLIC_EXECUTION
        change = MagicMock(file_path="main.py")
        mock_analyzer = AsyncMock()
        mock_analyzer.run_symbolic_analysis_on_file = AsyncMock(
            return_value={
                "symbolic_execution": {"status": "tool_not_found", "error": "CrossHair missing"}
            }
        )
        with (
            patch("app.api.endpoints.static_analysis.ValidationConfig", lambda **kw: kw),
            patch("app.api.endpoints.static_analysis.ValidationType", _FakeValidationType),
            patch("app.api.endpoints.static_analysis.SecurityLevel", _FakeSecurityLevel),
            patch(
                "app.api.endpoints.static_analysis.SymbolicValidator", return_value=mock_analyzer
            ),
        ):
            result = await _run_validation_analyses(
                "/tmp/test.py", change, [vt], _FakeSecurityLevel.MEDIUM
            )
        assert result["symbolic_execution"]["status"] == "not_computed"
        assert result["analyzers_ran"] == []
        assert result["analyzers_not_computed"][0]["reason"] == "CrossHair missing"

    async def test_symbolic_execution_failure(self):
        vt = _FakeValidationType.SYMBOLIC_EXECUTION
        change = MagicMock(file_path="main.py")
        mock_analyzer = AsyncMock()
        mock_analyzer.run_symbolic_analysis_on_file = AsyncMock(side_effect=RuntimeError("boom"))
        with (
            patch("app.api.endpoints.static_analysis.ValidationConfig", lambda **kw: kw),
            patch("app.api.endpoints.static_analysis.ValidationType", _FakeValidationType),
            patch("app.api.endpoints.static_analysis.SecurityLevel", _FakeSecurityLevel),
            patch(
                "app.api.endpoints.static_analysis.SymbolicValidator", return_value=mock_analyzer
            ),
        ):
            result = await _run_validation_analyses(
                "/tmp/test.py", change, [vt], _FakeSecurityLevel.MEDIUM
            )
        assert result["symbolic_execution"]["status"] == "failed"
        # A failed analyzer did not run — it must not count towards a pass.
        assert result["analyzers_ran"] == []

    async def test_with_taint_analysis(self, tmp_path):
        """The taint result is now produced by a real analyzer, not a literal."""
        vt = _FakeValidationType.TAINT_ANALYSIS
        change = MagicMock(file_path="main.py")
        target = tmp_path / "t.py"
        target.write_text("import os\n\n\ndef go():\n    os.system(input())\n")

        result = await _run_validation_analyses(
            str(target), change, [vt], _FakeSecurityLevel.MEDIUM
        )
        assert result["taint_analysis"]["status"] == "completed"
        assert result["taint_analysis"]["security_issues"], "a real defect must be reported"
        assert result["analyzers_ran"] == ["taint_analysis"]

    async def test_taint_analysis_on_clean_code_finds_nothing(self, tmp_path):
        vt = _FakeValidationType.TAINT_ANALYSIS
        change = MagicMock(file_path="main.py")
        target = tmp_path / "t.py"
        target.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        result = await _run_validation_analyses(
            str(target), change, [vt], _FakeSecurityLevel.MEDIUM
        )
        assert result["taint_analysis"]["security_issues"] == []

    async def test_with_formal_verification(self, tmp_path):
        """Formal verification is now run, not hard-coded to 'passed'/0.92."""
        vt = _FakeValidationType.FORMAL_VERIFICATION
        change = MagicMock(file_path="main.py")
        target = tmp_path / "t.py"
        target.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        result = await _run_validation_analyses(
            str(target), change, [vt], _FakeSecurityLevel.MEDIUM
        )
        formal = result["formal_verification"]
        assert formal["status"] == "passed"
        assert formal["coverage"] != 0.92
        assert formal["properties_verified"] != ["no_null_pointer", "bounds_checking"]
        assert result["analyzers_ran"] == ["formal_verification"]

    async def test_formal_verification_reports_violations(self, tmp_path):
        vt = _FakeValidationType.FORMAL_VERIFICATION
        change = MagicMock(file_path="main.py")
        target = tmp_path / "t.py"
        target.write_text("def run(x):\n    return x.value\n")

        result = await _run_validation_analyses(
            str(target), change, [vt], _FakeSecurityLevel.MEDIUM
        )
        assert result["formal_verification"]["status"] == "violations_found"
        assert result["formal_verification"]["violations"]

    async def test_security_level_without_value(self):
        change = MagicMock(file_path="main.py")
        result = await _run_validation_analyses("/tmp/test.py", change, [], "medium")
        assert result["analyzers_ran"] == []


# ---------------------------------------------------------------------------
# Tests for doc generation helpers
# ---------------------------------------------------------------------------


class TestDocGenerationHelpers:
    """Tests for _generate_architecture_docs, _generate_api_reference, _generate_change_summary."""

    async def test_architecture_docs_with_diagrams(self):
        result = await _generate_architecture_docs("/tmp/repo", True, "markdown")
        assert "architecture_diagram.svg" in result["diagrams"]
        assert result["quality_score"] == 0.8

    async def test_architecture_docs_without_diagrams(self):
        result = await _generate_architecture_docs("/tmp/repo", False, "markdown")
        assert result["diagrams"] == []

    async def test_api_reference(self):
        result = await _generate_api_reference("/tmp/repo", "markdown")
        assert "API Reference" in result["content"]
        assert result["quality_score"] == 0.9

    async def test_change_summary(self):
        result = await _generate_change_summary("/tmp/repo", "markdown")
        assert "Change Summary" in result["content"]
        assert result["quality_score"] == 0.85


# ---------------------------------------------------------------------------
