"""Capability endpoints report only what actually runs (#134).

``/api/quality/languages`` (+ ``/extensions``, ``/{language}/tools``, ``/info``) and
``/api/analysis/capabilities`` are computed from the real dispatch tables. These
tests fail if a listed language/tool has no runner, and EXECUTE every listed
(analysis type, language) pair to prove it has language-specific logic.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from processing.architecture.architecture_pattern_analysis import enabled_tool_details_by_language
from processing.architecture.tool_execution import ToolExecution
from processing.architecture.tool_execution._base import TOOL_DISPATCH

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class TestQualityCapabilities:
    def test_every_listed_tool_has_a_coroutine_runner(self):
        listed = enabled_tool_details_by_language()
        assert any(listed.values()), "nothing listed"
        for lang, tools in listed.items():
            for tool in tools:
                method = getattr(ToolExecution, TOOL_DISPATCH[tool["name"]][0], None)
                assert inspect.iscoroutinefunction(method), (lang, tool["name"])

    def test_every_dispatched_tool_has_a_runner(self):
        for name, (method, _) in TOOL_DISPATCH.items():
            assert inspect.iscoroutinefunction(getattr(ToolExecution, method, None)), name

    @pytest.mark.asyncio
    async def test_unknown_tool_reports_no_runner(self):
        issues, result = await ToolExecution().run_tool("flake8", [], Path("/repo"), "python")
        assert issues == []
        assert result["success"] is False
        assert result["error"] == "no runner for tool 'flake8'"

    def test_endpoints_over_http(self):
        from app.api.endpoints.quality import router

        app = FastAPI()
        app.include_router(router, prefix="/q")
        c = TestClient(app)
        langs = {entry["name"]: entry for entry in c.get("/q/languages").json()["languages"]}
        every_tool = {t for entry in langs.values() for t in entry["tools"]}
        assert not {"flake8", "staticcheck", "go vet", "mypy", "pylint"} & every_tool
        assert langs["rust"]["tools"] == ["clippy", "rustfmt", "cargo-audit"]
        assert langs["rust"]["extensions"] == [".rs"]
        assert langs["python"]["extensions"] == [".py"]
        assert langs["javascript"]["extensions"] == [".js", ".jsx"]  # registry order
        assert ".rs" in c.get("/q/extensions").json()["extensions"]
        body = c.get("/q/rs/tools").json()
        assert body["language"] == "rust"  # alias resolved to canonical
        rust = {t["name"]: t for t in body["tools"]}
        assert rust["clippy"]["path_scoping"] == "whole_project_findings_filtered"
        assert rust["cargo-audit"]["path_scoping"] == "skipped"
        py = {t["name"]: t for t in c.get("/q/py/tools").json()["tools"]}
        assert py["ruff"]["path_scoping"] == "paths"
        assert c.get("/q/C%23/tools").json()["language"] == "csharp"
        assert c.get("/q/cobol/tools").status_code == 404
        info = c.get("/q/info").json()["available_tools"]
        assert info == {name: entry["tools"] for name, entry in langs.items()}

    def test_openapi_declares_the_shapes(self):
        from app.api.endpoints.quality import router

        app = FastAPI()
        app.include_router(router, prefix="/q")
        schemas = app.openapi()["components"]["schemas"]
        assert "path_scoping" in schemas["QualityTool"]["properties"]
        assert schemas["QualityTool"]["properties"]["path_scoping"]["enum"] == [
            "paths",
            "whole_project_findings_filtered",
            "skipped",
        ]
        for name in (
            "QualityLanguagesResponse",
            "QualityLanguageToolsResponse",
            "QualityInfoResponse",
        ):
            assert name in schemas


# ---------------------------------------------------------------------------
# /api/analysis/capabilities
# ---------------------------------------------------------------------------

_BASIC_SAMPLES = {
    "python": ('print("x"\n', "syntax-check"),
    "javascript": ("console.log(1)\n", "no-console"),
    "typescript": ("console.log(1)\n", "no-console"),
    "java": ('System.out.println("x");\n', "no-println"),
}
_PY = "import os\n\ndef add(a, b):\n    return a + b\n\nclass K:\n    def m(self):\n        return 1\n"


def _request(kind: str, code: str, language: str):
    from app.api.models.analysis_api_models import AnalysisRequest, PydanticAIAnalysisType

    return AnalysisRequest(
        analysis_type=PydanticAIAnalysisType(kind), code=code, language=language, container_name="w"
    )


class TestAnalysisCapabilities:
    def test_shape_is_derived(self):
        from app.api.endpoints.analysis_modules.capabilities import compute_analysis_capabilities

        caps = compute_analysis_capabilities()
        assert caps.languages_by_type == {
            "basic": ["java", "javascript", "python", "typescript"],
            "augmented": ["python"],
            "ast": ["python"],
            "deep": ["python"],
            "security": [],
            "performance": [],
        }
        assert caps.supported_languages == ["java", "javascript", "python", "typescript"]
        assert caps.language_agnostic_types == ["security", "performance"]
        # No runner produces these: the old list claimed both, plus 8 languages.
        assert caps.features["test_coverage"] is False
        assert caps.features["technical_debt"] is False
        assert caps.features["cross_file_dependencies"] is False

    @pytest.mark.asyncio
    async def test_runner_outputs_table_matches_what_runners_produce(self):
        from app.api.endpoints.analysis_modules.core_analysis import (
            ANALYSIS_OUTPUTS,
            ANALYSIS_RUNNERS,
        )

        assert set(ANALYSIS_OUTPUTS) == set(ANALYSIS_RUNNERS)
        for kind, run in ANALYSIS_RUNNERS.items():
            result = await run(_request(kind.value, _PY, "python"), "a", 0.0)
            produced = {
                part
                for part, present in (
                    ("complexity_metrics", (result.metrics.complexity or 0) > 0),
                    ("suggestions", bool(result.suggestions)),
                )
                if present
            }
            assert produced == ANALYSIS_OUTPUTS[kind], kind

    def test_features_follow_the_outputs_table(self, monkeypatch):
        """Negative control: with no runner filling a part, its feature flips false."""
        from app.api.endpoints.analysis_modules import capabilities as caps_mod

        monkeypatch.setattr(
            caps_mod, "ANALYSIS_OUTPUTS", {k: frozenset() for k in caps_mod.ANALYSIS_RUNNERS}
        )
        caps = caps_mod.compute_analysis_capabilities()
        assert caps.features["complexity_metrics"] is False
        assert caps.features["resolve_suggestions"] is False
        monkeypatch.undo()
        caps = caps_mod.compute_analysis_capabilities()
        assert caps.features["complexity_metrics"] is True
        assert caps.features["resolve_suggestions"] is True

    @pytest.mark.asyncio
    async def test_every_basic_language_has_its_rule_and_others_do_not(self):
        from app.api.endpoints.analysis_modules.core_analysis import (
            ANALYSIS_LANGUAGES,
            basic_analysis,
        )
        from app.api.models.analysis_api_models import PydanticAIAnalysisType

        listed = ANALYSIS_LANGUAGES[PydanticAIAnalysisType.BASIC]
        assert set(listed) == set(_BASIC_SAMPLES), "a listed language has no sample/rule"
        all_samples = "".join(code for code, _ in _BASIC_SAMPLES.values())
        for lang in listed:
            code, rule = _BASIC_SAMPLES[lang]
            result = await basic_analysis(_request("basic", code, lang), "a", 0.0)
            assert rule in {i.rule for i in result.issues}, lang
        # Negative control: an unlisted language gets none of the language rules.
        result = await basic_analysis(_request("basic", all_samples, "rust"), "a", 0.0)
        assert not {r for _, r in _BASIC_SAMPLES.values()} & {i.rule for i in result.issues}

    @pytest.mark.asyncio
    async def test_ast_languages_parse_and_others_are_refused(self):
        from services.code_analysis.ast_validation_service import (
            AST_LANGUAGES,
            ASTValidationService,
        )

        svc = ASTValidationService()
        for lang in AST_LANGUAGES:
            res = await svc.analyze_proposed_code_ast(_PY, "f", lang)
            assert res.is_valid and res.functions, lang
        res = await svc.analyze_proposed_code_ast("fn main() {}", "f", "rust")
        assert not res.is_valid and "not supported" in res.syntax_errors[0]

    @pytest.mark.asyncio
    async def test_deep_and_augmented_languages_produce_entities(self, tmp_path):
        from app.api.endpoints.analysis_modules.core_analysis import AUGMENTED_LANGUAGES
        from processing.analyzers.deep import DeepCodeAnalyzer
        from processing.analyzers.deep.analyzer import SNIPPET_LANGUAGES
        from processing.analyzers.python_analyzer import PythonCodeProcessor

        for lang in SNIPPET_LANGUAGES:
            res = await DeepCodeAnalyzer(str(tmp_path), language=lang).analyze_code_snippet(
                _PY, lang
            )
            assert res.entities, lang
        res = await DeepCodeAnalyzer(str(tmp_path), language="rust").analyze_code_snippet(
            "fn main() {}", "rust"
        )
        assert not res.entities  # negative control: unlisted -> nothing understood
        assert AUGMENTED_LANGUAGES == ("python",)
        result = await PythonCodeProcessor().analyze_snippet(_PY, "s.py")
        assert result.entities or getattr(result, "functions", None)

    def test_route_over_http(self):
        from app.api.endpoints.analysis_modules.capabilities import router
        from auth.tenant import get_current_user_with_tenant

        app = FastAPI()
        app.include_router(router, prefix="/api/analysis")
        app.dependency_overrides[get_current_user_with_tenant] = lambda: SimpleNamespace(
            preferred_username="u"
        )
        caps = TestClient(app).get("/api/analysis/capabilities").json()["capabilities"]
        assert "rust" not in caps["supported_languages"]
        assert caps["languages_by_type"]["ast"] == ["python"]
        doc = app.openapi()["paths"]["/api/analysis/capabilities"]["get"]["description"]
        assert "Cross-file dependency tracking" not in doc
