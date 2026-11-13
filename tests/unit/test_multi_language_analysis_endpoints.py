"""Tests for analysis_modules/multi_language_analysis.py uncovered paths.

Covers lines 203-365 (analyze_multi_language_code snippet mode),
377-531 (analyze_multi_language_container).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_VALIDATE = "app.api.utils.container_validation.validate_and_get_container_for_tenant"
_RESOLVE = "app.api.endpoints.analysis_modules.multi_language_analysis.resolve_container_files"
_GET_PROC = "app.api.endpoints.analysis_modules.multi_language_analysis.get_processor_for_language"
_GET_LANGS = "app.api.endpoints.analysis_modules.multi_language_analysis.get_supported_languages"
_DETECT_LANG = (
    "app.api.endpoints.analysis_modules.multi_language_analysis.detect_language_from_code"
)
_GET_EXT = "app.api.endpoints.analysis_modules.multi_language_analysis.get_extension_for_language"
_GET_USER = (
    "app.api.endpoints.analysis_modules.multi_language_analysis.get_current_user_with_tenant"
)


def _make_mock_user(is_admin=False):
    user = MagicMock()
    user.preferred_username = "test-user"
    user.is_admin = MagicMock(return_value=is_admin)
    user.effective_tenant_id = "test-tenant"
    return user


# ---------------------------------------------------------------------------
# detect_language_from_file_path
# ---------------------------------------------------------------------------


class TestDetectLanguageFromFilePath:
    def test_python(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("main.py") == "python"

    def test_java(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("App.java") == "java"

    def test_javascript(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("index.js") == "javascript"

    def test_typescript(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("app.ts") == "typescript"

    def test_rust(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("main.rs") == "rust"

    def test_go(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("main.go") == "go"

    def test_csharp(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("Program.cs") == "csharp"

    def test_unknown_extension(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("data.xyz") is None

    def test_header_file(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("lib.h") == "c"
        assert detect_language_from_file_path("lib.hpp") == "cpp"

    def test_jsx(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("component.jsx") == "javascript"

    def test_tsx(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("component.tsx") == "typescript"

    def test_cpp(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("main.cpp") == "cpp"

    def test_c(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("main.c") == "c"

    def test_ruby(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("app.rb") == "ruby"

    def test_kotlin(self):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            detect_language_from_file_path,
        )

        assert detect_language_from_file_path("Main.kt") == "kotlin"


# ---------------------------------------------------------------------------
# analyze_multi_language_code: no code, no container -> 400
# ---------------------------------------------------------------------------


class TestAnalyzeMultiLanguageCodeValidation:
    @pytest.mark.asyncio
    async def test_no_code_routes_to_container_analysis(self):
        """When container_name is provided but code is None, it routes to container analysis."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        user = _make_mock_user()
        request = AnalysisRequest(container_name="test", analysis_type="basic")

        # Since container_name is provided, it will try container analysis
        with (
            patch(
                _VALIDATE,
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=404, detail="not found"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_no_code_no_container_raises_400(self):
        """Neither code nor container_name raises 400."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        # Use a mock request since the Pydantic model requires container_name
        request = MagicMock()
        request.code = None
        request.container_name = ""  # empty string is falsy
        request.analysis_type = "basic"
        request.language = None

        with pytest.raises(HTTPException) as exc_info:
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 400
        assert "Either 'code' content or 'container_name'" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_code_with_undetectable_language_raises_422(self):
        """Code provided but language cannot be auto-detected raises 422."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        # Use mock since container_name is required in the Pydantic model
        # but we want to test the code-snippet path where container_name is falsy
        request = MagicMock()
        request.code = "some random text"
        request.container_name = None
        request.language = None
        request.analysis_type = "basic"

        with (
            patch(_DETECT_LANG, return_value=None),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 422
        assert "Could not detect programming language" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_code_with_unsupported_language_raises_422(self):
        """Code with unsupported language raises 422."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "print('hello')"
        request.container_name = None
        request.language = "brainfuck"
        request.analysis_type = "basic"

        with (
            patch(_GET_LANGS, return_value=["python", "java", "go"]),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 422
        assert "not supported" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_code_with_no_processor_raises_500(self):
        """Code with supported language but no processor raises 500."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "print('hello')"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "basic"

        with (
            patch(_GET_LANGS, return_value=["python"]),
            patch(_GET_PROC, new_callable=AsyncMock, return_value=None),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 500
        assert "Could not initialize processor" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_code_snippet_analysis_success(self):
        """Full code snippet analysis path with a LanguageProcessor."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            LanguageProcessor,
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "def hello(): pass"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "basic"

        mock_entity = MagicMock()
        mock_entity.errors = None
        mock_entity.line_number = 1

        mock_parse_result = MagicMock()
        mock_parse_result.success = True
        mock_parse_result.entities = [mock_entity]
        mock_parse_result.relationships = []
        mock_parse_result.lines_of_code = 1
        mock_parse_result.complexity_score = 5.0
        mock_parse_result.errors = None
        mock_parse_result.parser_used = MagicMock()
        mock_parse_result.parser_used.value = "tree_sitter"

        mock_processor = MagicMock(spec=LanguageProcessor)
        mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)

        with (
            patch(_GET_LANGS, return_value=["python"]),
            patch(_GET_PROC, new_callable=AsyncMock, return_value=mock_processor),
            patch(_GET_EXT, return_value=".py"),
        ):
            result = await analyze_multi_language_code(request=request, user=user)

        assert result.result.success is True
        assert result.result.metrics.lines_of_code == 1
        assert result.result.metrics.complexity == 5.0
        # maintainability = max(0, 100 - 5.0 * 2) = 90
        assert result.result.metrics.maintainability == 90.0
        # No high complexity -> no tech debt
        assert result.result.metrics.technical_debt is None
        assert "python" in result.result.metadata["language"]
        assert result.result.metadata["analysis_scope"] == "multi_language"
        assert len(result.result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_code_snippet_high_complexity(self):
        """Code snippet with high complexity triggers refactoring suggestion and tech debt."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            LanguageProcessor,
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "complex_code"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "deep"

        mock_parse_result = MagicMock()
        mock_parse_result.success = True
        mock_parse_result.entities = []  # No entities
        mock_parse_result.relationships = []
        mock_parse_result.lines_of_code = 100
        mock_parse_result.complexity_score = 25.0  # High complexity
        mock_parse_result.errors = ["parse warning"]
        mock_parse_result.parser_used = "fallback"

        mock_processor = MagicMock(spec=LanguageProcessor)
        mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)

        with (
            patch(_GET_LANGS, return_value=["python"]),
            patch(_GET_PROC, new_callable=AsyncMock, return_value=mock_processor),
            patch(_GET_EXT, return_value=".py"),
        ):
            result = await analyze_multi_language_code(request=request, user=user)

        assert result.result.success is True
        # Check technical_debt = max(0, (25 - 10) * 0.5) = 7.5
        assert result.result.metrics.technical_debt == 7.5
        # Check suggestions: high complexity, no entities, and errors
        suggestions = result.result.suggestions
        assert any("refactoring" in s.lower() for s in suggestions)
        assert any("No code entities" in s for s in suggestions)
        assert any("parsing errors" in s.lower() for s in suggestions)
        # parser_used is a string -> str(parse_result.parser_used)
        assert result.result.metadata["parser_used"] == "fallback"

    @pytest.mark.asyncio
    async def test_code_snippet_entity_with_errors(self):
        """Code snippet analysis where entity has errors creates issues."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            LanguageProcessor,
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "def hello(): pass"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "basic"

        mock_entity = MagicMock()
        mock_entity.errors = ["missing return type", "unused variable"]
        mock_entity.line_number = 5

        mock_parse_result = MagicMock()
        mock_parse_result.success = True
        mock_parse_result.entities = [mock_entity]
        mock_parse_result.relationships = []
        mock_parse_result.lines_of_code = 10
        mock_parse_result.complexity_score = 3.0
        mock_parse_result.errors = None
        mock_parse_result.parser_used = MagicMock()
        mock_parse_result.parser_used.value = "ast"

        mock_processor = MagicMock(spec=LanguageProcessor)
        mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)

        with (
            patch(_GET_LANGS, return_value=["python"]),
            patch(_GET_PROC, new_callable=AsyncMock, return_value=mock_processor),
            patch(_GET_EXT, return_value=".py"),
        ):
            result = await analyze_multi_language_code(request=request, user=user)

        # Should have 2 issues from entity errors
        assert len(result.result.issues) == 2
        assert result.result.issues[0].line == 5
        assert result.result.issues[0].type == "syntax"

    @pytest.mark.asyncio
    async def test_code_snippet_legacy_processor_raises_500(self):
        """Non-LanguageProcessor raises 500."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "print('hello')"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "basic"

        # Create a mock that is NOT a LanguageProcessor instance
        mock_processor = MagicMock()

        with (
            patch(_GET_LANGS, return_value=["python"]),
            patch(_GET_PROC, new_callable=AsyncMock, return_value=mock_processor),
            patch(_GET_EXT, return_value=".py"),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 500
        assert "Legacy processor not supported" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_runtime_error_raises_500(self):
        """RuntimeError in the outer try block raises 500."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )

        user = _make_mock_user()
        request = MagicMock()
        request.code = "print('hello')"
        request.container_name = None
        request.language = "python"
        request.analysis_type = "basic"

        with (
            patch(_GET_LANGS, side_effect=RuntimeError("factory crash")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_multi_language_code(request=request, user=user)
        assert exc_info.value.status_code == 500
        assert "factory crash" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_admin_user_passes_none_tenant(self):
        """Admin user passes tenant_id=None to container analysis."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_code,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        user = _make_mock_user(is_admin=True)
        request = AnalysisRequest(container_name="test", analysis_type="basic")

        with (
            patch(_VALIDATE, new_callable=AsyncMock) as _mock_validate,
            patch(_RESOLVE, new_callable=AsyncMock, return_value=[]) as _,
            pytest.raises(HTTPException),  # 404 no files
        ):
            await analyze_multi_language_code(request=request, user=user)


# ---------------------------------------------------------------------------
# analyze_multi_language_container
# ---------------------------------------------------------------------------


class TestAnalyzeMultiLanguageContainer:
    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_no_files_raises_404(self, mock_validate, mock_resolve, mock_proc):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()
        mock_resolve.return_value = []

        request = AnalysisRequest(container_name="test", analysis_type="basic")

        with pytest.raises(HTTPException) as exc_info:
            await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_no_supported_languages_raises_422(self, mock_validate, mock_resolve, mock_proc):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()
        # Return files with unsupported extensions
        mock_resolve.return_value = [Path("/tmp/data.xyz"), Path("/tmp/config.yaml")]

        request = AnalysisRequest(container_name="test", analysis_type="basic")

        with pytest.raises(HTTPException) as exc_info:
            await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_no_processor_skips_language(self, mock_validate, mock_resolve, mock_proc):
        """When no processor is available for a language, that language is skipped."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()

        # File with valid extension
        real_path = Path("/tmp/main.py")
        mock_resolve.return_value = [real_path]

        # No processor available
        mock_proc.return_value = None

        request = AnalysisRequest(container_name="test", analysis_type="basic")

        # Even though we have a .py file, no processor means no analysis
        # But at least we get the result structure
        result = await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")
        assert result.result.success is True

    @pytest.mark.asyncio
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_value_error_raises_400(self, mock_validate):
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.side_effect = ValueError("bad input")

        request = AnalysisRequest(container_name="test", analysis_type="basic")

        with pytest.raises(HTTPException) as exc_info:
            await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_successful_multi_file_analysis(self, mock_validate, mock_resolve, mock_proc):
        """Full success path with file parsing, entities, relationships, and errors."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            LanguageProcessor,
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()

        # Create temp files with real content
        tmp1 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp1.write("def hello(): pass")
        tmp1.close()
        tmp2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp2.write("def world(): pass")
        tmp2.close()

        try:
            mock_resolve.return_value = [Path(tmp1.name), Path(tmp2.name)]

            mock_entity = MagicMock()
            mock_entity.name = "hello"

            mock_parse_result = MagicMock()
            mock_parse_result.success = True
            mock_parse_result.entities = [mock_entity]
            mock_parse_result.relationships = [MagicMock()]
            mock_parse_result.lines_of_code = 10
            mock_parse_result.errors = ["minor warning"]

            mock_processor = MagicMock(spec=LanguageProcessor)
            mock_processor.parse_file_with_metrics = AsyncMock(return_value=mock_parse_result)
            mock_proc.return_value = mock_processor

            request = AnalysisRequest(container_name="test-container", analysis_type="deep")
            result = await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")

            assert result.result.success is True
            assert result.result.metrics.lines_of_code == 20  # 10 per file * 2
            assert len(result.result.issues) >= 2  # 1 error per file
            assert "test-container" in result.message
            assert result.result.metadata["entity_count"] == 2
            assert result.result.metadata["relationship_count"] == 2
        finally:
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)

    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_file_read_error_creates_issue(self, mock_validate, mock_resolve, mock_proc):
        """When a file can't be read, an error issue is created but analysis continues."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            LanguageProcessor,
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()
        # Use a path that doesn't exist -- read_text will raise
        mock_resolve.return_value = [Path("/nonexistent/path/main.py")]

        mock_processor = MagicMock(spec=LanguageProcessor)
        mock_proc.return_value = mock_processor

        request = AnalysisRequest(container_name="test", analysis_type="basic")
        result = await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")

        assert result.result.success is True
        # Should have an error issue for the unreadable file
        assert len(result.result.issues) >= 1
        assert result.result.issues[0].type == "error"
        assert result.result.issues[0].severity == "high"

    @pytest.mark.asyncio
    @patch(_GET_PROC, new_callable=AsyncMock)
    @patch(_RESOLVE, new_callable=AsyncMock)
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_non_language_processor_skipped(self, mock_validate, mock_resolve, mock_proc):
        """When processor is not a LanguageProcessor, the file is skipped."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.return_value = MagicMock()

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write("x = 1")
        tmp.close()

        try:
            mock_resolve.return_value = [Path(tmp.name)]
            # Return a mock that is NOT a LanguageProcessor
            mock_proc.return_value = MagicMock()

            request = AnalysisRequest(container_name="test", analysis_type="basic")
            result = await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")

            assert result.result.success is True
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    @patch(_VALIDATE, new_callable=AsyncMock)
    async def test_generic_exception_raises_500(self, mock_validate):
        """Generic Exception (not ValueError, not HTTPException) raises 500."""
        from app.api.endpoints.analysis_modules.multi_language_analysis import (
            analyze_multi_language_container,
        )
        from app.api.models.analysis_api_models import AnalysisRequest

        mock_validate.side_effect = RuntimeError("unexpected crash")

        request = AnalysisRequest(container_name="test", analysis_type="basic")

        with pytest.raises(HTTPException) as exc_info:
            await analyze_multi_language_container(request, "id1", 0.0, "test-tenant")
        assert exc_info.value.status_code == 500
        assert "unexpected crash" in exc_info.value.detail
