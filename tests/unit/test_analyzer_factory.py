"""Tests for processing.analyzers.analyzer_factory module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from processing.analyzers.analyzer_factory import (
    AnalyzerFactory,
    get_analyzer,
    get_analyzer_factory,
    get_analyzer_for_file,
)
from processing.analyzers.base_analyzer import BaseCodeProcessor

# Cache the module reference at import time (before sys.modules["processing"]
# gets replaced by stubs in other test files).
_factory_mod = sys.modules["processing.analyzers.analyzer_factory"]


class TestAnalyzerFactoryInit:
    def test_default_initialization(self):
        factory = AnalyzerFactory()
        assert factory.file_service is None
        assert "python" in factory._language_mappings
        assert "javascript" in factory._language_mappings
        assert "typescript" in factory._language_mappings
        assert ".py" in factory._extension_mappings
        assert ".js" in factory._extension_mappings
        assert ".ts" in factory._extension_mappings

    def test_with_file_service(self):
        fs = MagicMock()
        factory = AnalyzerFactory(file_service=fs)
        assert factory.file_service is fs


class TestAnalyzerFactoryGetAnalyzer:
    def test_get_python_analyzer(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer("python")
        assert analyzer is not None
        assert "python" in factory._analyzers

    def test_get_same_analyzer_cached(self):
        factory = AnalyzerFactory()
        a1 = factory.get_analyzer("python")
        a2 = factory.get_analyzer("python")
        assert a1 is a2

    def test_unsupported_language_returns_none(self):
        factory = AnalyzerFactory()
        result = factory.get_analyzer("cobol")
        assert result is None

    def test_error_creating_analyzer_returns_none(self):
        factory = AnalyzerFactory()
        # Register a broken analyzer class
        factory._language_mappings["broken"] = type(
            "Bad", (), {"__init__": lambda s: (_ for _ in ()).throw(TypeError("boom"))}
        )
        result = factory.get_analyzer("broken")
        assert result is None


class TestAnalyzerFactoryGetAnalyzerForFile:
    def test_python_file(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer_for_file("src/main.py")
        assert analyzer is not None

    def test_pyw_file(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer_for_file("gui.pyw")
        assert analyzer is not None

    def test_javascript_file(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer_for_file("index.js")
        assert analyzer is not None

    def test_tsx_file(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer_for_file("component.tsx")
        assert analyzer is not None

    def test_d_ts_file(self):
        factory = AnalyzerFactory()
        analyzer = factory.get_analyzer_for_file("types.d.ts")
        assert analyzer is not None

    def test_unsupported_file(self):
        factory = AnalyzerFactory()
        result = factory.get_analyzer_for_file("data.csv")
        assert result is None

    def test_error_returns_none(self):
        factory = AnalyzerFactory()
        # Passing something that causes Path to fail won't easily error
        # but we can still test the function returns None for bad extensions
        result = factory.get_analyzer_for_file("file.unknown")
        assert result is None


class TestAnalyzerFactorySupportedQueries:
    def test_get_supported_languages(self):
        factory = AnalyzerFactory()
        langs = factory.get_supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs

    def test_get_supported_extensions(self):
        factory = AnalyzerFactory()
        exts = factory.get_supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".d.ts" in exts

    def test_is_supported_file_true(self):
        factory = AnalyzerFactory()
        assert factory.is_supported_file("test.py") is True
        assert factory.is_supported_file("app.js") is True

    def test_is_supported_file_false(self):
        factory = AnalyzerFactory()
        assert factory.is_supported_file("data.csv") is False

    def test_is_supported_file_d_ts(self):
        factory = AnalyzerFactory()
        assert factory.is_supported_file("types.d.ts") is True

    def test_get_language_for_file(self):
        factory = AnalyzerFactory()
        assert factory.get_language_for_file("test.py") == "python"
        assert factory.get_language_for_file("app.js") == "javascript"
        assert factory.get_language_for_file("types.d.ts") == "typescript"
        assert factory.get_language_for_file("data.csv") is None


class TestAnalyzerFactoryRegister:
    def test_register_analyzer(self):
        factory = AnalyzerFactory()

        class DummyAnalyzer(BaseCodeProcessor):
            def get_supported_extensions(self):
                return [".rb"]

            def get_language_name(self):
                return "ruby"

        factory.register_analyzer("ruby", DummyAnalyzer, [".rb", ".rake"])
        assert "ruby" in factory._language_mappings
        assert factory._extension_mappings[".rb"] == "ruby"
        assert factory._extension_mappings[".rake"] == "ruby"

    def test_register_clears_cache(self):
        factory = AnalyzerFactory()
        # Get an analyzer to cache it
        factory.get_analyzer("python")
        assert "python" in factory._analyzers

        # Re-register python
        old_class = factory._language_mappings["python"]
        factory.register_analyzer("python", old_class, [".py"])
        assert "python" not in factory._analyzers


class TestAnalyzerFactoryInfo:
    def test_get_analyzer_info(self):
        factory = AnalyzerFactory()
        info = factory.get_analyzer_info()
        assert "python" in info
        assert "class" in info["python"]
        assert "extensions" in info["python"]
        assert "cached" in info["python"]
        assert info["python"]["cached"] is False

    def test_get_analyzer_info_with_cached(self):
        factory = AnalyzerFactory()
        factory.get_analyzer("python")
        info = factory.get_analyzer_info()
        assert info["python"]["cached"] is True


class TestAnalyzerFactoryInitializeAll:
    @pytest.mark.asyncio
    async def test_initialize_all_analyzers(self):
        factory = AnalyzerFactory()
        # Should not raise
        await factory.initialize_all_analyzers()
        # All analyzers should be cached after initialization
        assert len(factory._analyzers) >= 1


class TestAnalyzerFactoryClearCache:
    def test_clear_cache(self):
        factory = AnalyzerFactory()
        factory.get_analyzer("python")
        assert len(factory._analyzers) > 0
        factory.clear_cache()
        assert len(factory._analyzers) == 0


class TestConvenienceFunctions:
    def test_get_analyzer_factory_creates_singleton(self):
        # Reset global state
        _factory_mod._default_factory = None

        factory1 = get_analyzer_factory()
        factory2 = get_analyzer_factory()
        assert factory1 is factory2

        # Clean up
        _factory_mod._default_factory = None

    def test_get_analyzer_factory_with_different_file_service(self):
        _factory_mod._default_factory = None

        fs1 = MagicMock()
        factory1 = get_analyzer_factory(fs1)
        assert factory1.file_service is fs1

        fs2 = MagicMock()
        factory2 = get_analyzer_factory(fs2)
        # Should create new factory since file_service differs
        assert factory2.file_service is fs2

        # Clean up
        _factory_mod._default_factory = None

    def test_get_analyzer_for_file_convenience(self):
        _factory_mod._default_factory = None

        analyzer = get_analyzer_for_file("test.py")
        assert analyzer is not None

        # Clean up
        _factory_mod._default_factory = None

    def test_get_analyzer_convenience(self):
        _factory_mod._default_factory = None

        analyzer = get_analyzer("python")
        assert analyzer is not None

        # Clean up
        _factory_mod._default_factory = None
