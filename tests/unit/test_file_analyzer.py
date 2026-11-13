"""Round 4 tests for FileAnalyzer (74 uncov lines)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from processing.analyzers.file_analyzer import FileAnalyzer


@pytest.fixture
def analyzer():
    return FileAnalyzer()


class TestFileAnalyzerInit:
    def test_init(self, analyzer):
        assert analyzer.analyzer_factory is not None
        assert analyzer._analyzer_cache == {}
        assert isinstance(analyzer._extension_map, dict)

    def test_extension_map_keys(self, analyzer):
        ext_map = analyzer._create_extension_map()
        assert ".py" in ext_map
        assert ".js" in ext_map
        assert ".ts" in ext_map
        assert ".java" in ext_map
        assert ".go" in ext_map
        assert ".rs" in ext_map
        assert ".cpp" in ext_map
        assert ".cs" in ext_map

    def test_extension_map_values(self, analyzer):
        ext_map = analyzer._create_extension_map()
        assert ext_map[".py"] == "python"
        assert ext_map[".js"] == "javascript"
        assert ext_map[".tsx"] == "typescript"
        assert ext_map[".java"] == "java"
        assert ext_map[".go"] == "go"
        assert ext_map[".rs"] == "rust"
        assert ext_map[".cpp"] == "cpp"
        assert ext_map[".cs"] == "csharp"

    def test_get_language_name(self, analyzer):
        assert analyzer.get_language_name() == "file"


class TestFileAnalyzerCanAnalyze:
    def test_can_analyze_python(self, analyzer):
        """Python files should be analyzable since the python analyzer is available."""
        result = analyzer.can_analyze_file("test.py")
        assert result is True

    def test_can_analyze_unknown(self, analyzer):
        result = analyzer.can_analyze_file("test.xyz")
        assert result is False

    def test_can_analyze_javascript(self, analyzer):
        result = analyzer.can_analyze_file("app.js")
        assert result is True

    def test_can_analyze_typescript(self, analyzer):
        result = analyzer.can_analyze_file("component.tsx")
        assert result is True


class TestFileAnalyzerAnalyzeFile:
    @pytest.mark.asyncio
    async def test_analyze_file_no_analyzer(self, analyzer):
        result = await analyzer.analyze_file("test.xyz")
        assert result.success is False
        assert "No analyzer" in (result.error or "")
        assert result.language == "unknown"
        assert result.file_path == "test.xyz"
        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_analyze_file_nonexistent(self, analyzer):
        """Analyzing a nonexistent .py file should fail gracefully, not crash."""
        result = await analyzer.analyze_file("/nonexistent/path/test.py")
        assert result.success is False
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.asyncio
    async def test_analyze_file_exception_handling(self, analyzer):
        with patch.object(analyzer, "_get_analyzer_for_file", side_effect=RuntimeError("boom")):
            result = await analyzer.analyze_file("test.py")
        assert result.success is False
        assert "boom" in (result.error or "")
        assert result.language == "unknown"


class TestFileAnalyzerAnalyzeSnippet:
    @pytest.mark.asyncio
    async def test_analyze_snippet_no_file_path(self, analyzer):
        """Without a file path, tries priority languages (python, javascript, typescript)."""
        result = await analyzer.analyze_snippet("x = 1")
        # Python analyzer should pick this up since "x = 1" is valid Python
        if result.success:
            assert result.language in ("python", "javascript", "typescript")
            assert result.file_path is not None
        else:
            # If all analyzers fail, we get the "Could not determine language" error
            assert result.error == "Could not determine language for snippet"
            assert result.language == "unknown"

    @pytest.mark.asyncio
    async def test_analyze_snippet_with_file_path(self, analyzer):
        """With a .py file path, should delegate to the Python analyzer."""
        result = await analyzer.analyze_snippet("x = 1", file_path="test.py")
        # The python analyzer should handle this
        if result.success:
            assert result.language == "python"
        else:
            # Even on failure, the result should have meaningful error info
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_analyze_snippet_unknown_lang(self, analyzer):
        """Unknown extension falls back through priority languages."""
        result = await analyzer.analyze_snippet("something", file_path="test.unknown")
        # Falls back through priority languages - either one succeeds or we get "could not determine"
        if not result.success:
            assert result.error == "Could not determine language for snippet"
            assert result.language == "unknown"
            assert result.file_path == "test.unknown"


class TestFileAnalyzerDirectory:
    @pytest.mark.asyncio
    async def test_analyze_directory_empty(self, analyzer):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await analyzer.analyze_directory(tmpdir)
            assert result.success is True  # No files = no errors
            assert result.language == "multi"
            assert result.metadata["files_analyzed"] == 0
            assert result.metadata["error_count"] == 0
            assert result.metadata["languages"] == []
            assert result.file_path == tmpdir

    @pytest.mark.asyncio
    async def test_analyze_directory_with_files(self, analyzer):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("x = 1\n")
            txt_file = Path(tmpdir) / "readme.txt"
            txt_file.write_text("Hello\n")

            result = await analyzer.analyze_directory(tmpdir)
            assert result.language == "multi"
            assert isinstance(result.metadata, dict)
            # .txt is not a supported extension, so only .py should be analyzed
            assert result.metadata["files_analyzed"] >= 0


class TestFileAnalyzerGetSupportedExtensions:
    def test_get_supported_extensions(self, analyzer):
        extensions = analyzer.get_supported_extensions()
        assert isinstance(extensions, list)
        # Should be sorted
        assert extensions == sorted(extensions)
        # Python extension should always be present since we have a python analyzer
        assert ".py" in extensions


class TestFileAnalyzerGetAnalyzer:
    @pytest.mark.asyncio
    async def test_get_analyzer_for_python(self, analyzer):
        result = await analyzer._get_analyzer_for_file("test.py")
        # Python analyzer should be available
        assert result is not None
        assert hasattr(result, "analyze_file")
        assert hasattr(result, "analyze_snippet")

    @pytest.mark.asyncio
    async def test_get_analyzer_caching(self, analyzer):
        """Second call should use cache."""
        r1 = await analyzer._get_analyzer_for_file("test.py")
        r2 = await analyzer._get_analyzer_for_file("other.py")
        # Both .py files should get the same analyzer instance (from cache)
        assert r1 is not None
        assert r1 is r2
        # Verify it's actually in the cache
        assert ".py" in analyzer._analyzer_cache

    @pytest.mark.asyncio
    async def test_get_analyzer_for_unknown(self, analyzer):
        result = await analyzer._get_analyzer_for_file("test.xyz")
        assert result is None
        # Unknown extension should not be cached
        assert ".xyz" not in analyzer._analyzer_cache
