"""Deep unit tests for processing/multi_language_factory.py

Covers:
- MultiLanguageProcessorFactory: init, get_processor, get_processor_for_file,
  get_supported_languages, get_supported_extensions, is_supported_file,
  get_language_info, create_processor_set
- CodebaseProcessorSet: analyze_codebase_async, create_minimal_graph
- Module-level helpers: get_multi_language_factory, get_processor_for_file,
  get_processor_for_language, get_supported_languages, MultiLanguageFactory alias
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from processing.languages.base import (
    LanguageProcessor,
)
from processing.languages.registry import LanguageRegistry
from processing.multi_language_factory import (
    CodebaseProcessorSet,
    MultiLanguageFactory,
    MultiLanguageProcessorFactory,
    get_multi_language_factory,
)
from processing.multi_language_factory import (
    get_processor_for_file as mod_get_processor_for_file,
)
from processing.multi_language_factory import (
    get_processor_for_language as mod_get_processor_for_language,
)
from processing.multi_language_factory import (
    get_supported_languages as mod_get_supported_languages,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_factory():
    """Reset the module-level _multi_language_factory before each test."""
    mlf_mod = sys.modules.get("processing.multi_language_factory")
    if mlf_mod is None:
        import processing.multi_language_factory as mlf_mod
    mlf_mod._multi_language_factory = None
    yield
    mlf_mod._multi_language_factory = None


@pytest.fixture
def mock_registry():
    """Create a fresh mock LanguageRegistry."""
    reg = MagicMock(spec=LanguageRegistry)
    reg.get_supported_languages.return_value = ["java", "go"]
    reg.get_supported_extensions.return_value = {".java", ".go"}
    reg.is_language_supported.return_value = False
    reg.detect_language.return_value = None
    return reg


@pytest.fixture
def factory():
    """Create a MultiLanguageProcessorFactory with mocked language initialization."""
    with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
        f = MultiLanguageProcessorFactory(file_service=MagicMock())
    return f


# ===========================================================================
# MultiLanguageProcessorFactory tests
# ===========================================================================


class TestMultiLanguageProcessorFactoryInit:
    """Tests for factory initialization."""

    def test_init_with_file_service(self):
        """Factory stores provided file_service."""
        fs = MagicMock()
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = MultiLanguageProcessorFactory(file_service=fs)
        assert f.file_service is fs

    def test_init_without_file_service_creates_default(self):
        """Factory creates a default FileService when None."""
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = MultiLanguageProcessorFactory(file_service=None)
        assert f.file_service is not None

    def test_legacy_processors_registered(self):
        """Legacy python/javascript/typescript processors are registered."""
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = MultiLanguageProcessorFactory()
        assert "python" in f._legacy_processors
        assert "javascript" in f._legacy_processors
        assert "typescript" in f._legacy_processors

    def test_initialize_language_processors_import_error(self):
        """_initialize_language_processors logs warning on ImportError."""
        with patch("processing.multi_language_factory.language_registry"):
            # Force an ImportError by patching the import mechanism
            with patch.dict(
                sys.modules,
                {
                    "processing.languages.cpp.cpp_processor": None,
                },
            ):
                # Should not raise
                f = MultiLanguageProcessorFactory()
                # Factory should still be created
                assert f is not None


class TestGetProcessor:
    """Tests for get_processor method."""

    async def test_get_processor_from_registry(self, factory):
        """get_processor returns processor from registry when language supported."""
        mock_proc = MagicMock(spec=LanguageProcessor)
        mock_proc.__class__.__name__ = "MockProcessor"
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_processor = AsyncMock(return_value=mock_proc)
            result = await factory.get_processor("java")
        assert result is mock_proc

    async def test_get_processor_case_insensitive(self, factory):
        """get_processor converts language to lowercase."""
        mock_proc = MagicMock(spec=LanguageProcessor)
        mock_proc.__class__.__name__ = "MockProcessor"
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_processor = AsyncMock(return_value=mock_proc)
            await factory.get_processor("JAVA")
        mock_reg.is_language_supported.assert_called_with("java")

    async def test_get_processor_registry_failure_falls_back_to_legacy(self, factory):
        """If registry processor creation fails, falls back to legacy."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_processor = AsyncMock(side_effect=RuntimeError("boom"))
            result = await factory.get_processor("python")
        # Should fall back to legacy python processor
        assert result is not None

    async def test_get_processor_legacy_fallback(self, factory):
        """get_processor falls back to legacy when not in registry."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor("python")
        assert result is not None

    async def test_get_processor_legacy_creation_failure(self, factory):
        """get_processor returns None when legacy creation fails."""
        factory._legacy_processors["badlang"] = MagicMock(side_effect=TypeError("no"))
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor("badlang")
        assert result is None

    async def test_get_processor_unsupported_language(self, factory):
        """get_processor returns None for unknown language."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor("cobol")
        assert result is None


class TestGetProcessorForFile:
    """Tests for get_processor_for_file method."""

    async def test_get_processor_for_file_detected_by_registry(self, factory):
        """Uses registry detect_language when it recognizes the file."""
        mock_proc = MagicMock()
        mock_proc.__class__.__name__ = "MockProc"
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = "java"
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_processor = AsyncMock(return_value=mock_proc)
            result = await factory.get_processor_for_file("/foo/Bar.java")
        assert result is mock_proc

    async def test_get_processor_for_file_legacy_extension(self, factory):
        """Falls back to legacy extension map for .py files."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor_for_file("/foo/bar.py")
        assert result is not None

    async def test_get_processor_for_file_legacy_js(self, factory):
        """Falls back to legacy extension map for .js files."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor_for_file("/foo/bar.js")
        assert result is not None

    async def test_get_processor_for_file_legacy_tsx(self, factory):
        """Falls back to legacy extension map for .tsx files."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor_for_file("/foo/bar.tsx")
        assert result is not None

    async def test_get_processor_for_file_unsupported(self, factory):
        """Returns None for an unsupported file extension."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor_for_file("/foo/bar.xyz")
        assert result is None

    async def test_get_processor_for_file_path_object(self, factory):
        """Accepts a Path object."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await factory.get_processor_for_file(Path("/foo/bar.py"))
        assert result is not None


class TestSupportedLanguages:
    """Tests for get_supported_languages method."""

    def test_combines_registry_and_legacy(self, factory):
        """Supported languages combine new registry + legacy."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_languages.return_value = ["java", "go"]
            langs = factory.get_supported_languages()
        assert "java" in langs
        assert "go" in langs
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs

    def test_no_duplicates(self, factory):
        """No duplicate languages when registry and legacy overlap."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_languages.return_value = ["python", "java"]
            langs = factory.get_supported_languages()
        assert langs.count("python") == 1


class TestSupportedExtensions:
    """Tests for get_supported_extensions method."""

    def test_combines_registry_and_legacy_extensions(self, factory):
        """Supported extensions combine registry + legacy."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_extensions.return_value = {".java", ".go"}
            exts = factory.get_supported_extensions()
        assert ".java" in exts
        assert ".go" in exts
        assert ".py" in exts
        assert ".js" in exts
        assert ".jsx" in exts
        assert ".ts" in exts
        assert ".tsx" in exts


class TestIsSupportedFile:
    """Tests for is_supported_file method."""

    def test_supported_by_extension(self, factory):
        """Returns True for a file with known extension."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_extensions.return_value = set()
            mock_reg.detect_language.return_value = None
            # .py is in legacy extensions
            result = factory.is_supported_file("/foo/bar.py")
        assert result is True

    def test_supported_by_registry_detection(self, factory):
        """Returns True when registry detects the language."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_extensions.return_value = set()
            mock_reg.detect_language.return_value = "java"
            result = factory.is_supported_file("/foo/Bar.java")
        assert result is True

    def test_unsupported_file(self, factory):
        """Returns False for unrecognized extension."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_extensions.return_value = set()
            mock_reg.detect_language.return_value = None
            result = factory.is_supported_file("/foo/bar.xyz")
        assert result is False

    def test_accepts_path_object(self, factory):
        """Accepts Path objects."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.get_supported_extensions.return_value = set()
            mock_reg.detect_language.return_value = None
            result = factory.is_supported_file(Path("/foo/bar.py"))
        assert result is True


class TestGetLanguageInfo:
    """Tests for get_language_info method."""

    def test_info_from_registry(self, factory):
        """Returns language info from the new registry."""
        mock_features = MagicMock()
        mock_features.model_dump.return_value = {"syntax_parsing": True}
        mock_config = MagicMock()
        mock_config.name = "java"
        mock_config.display_name = "Java"
        mock_config.extensions = [".java"]
        mock_config.parser_type.value = "tree_sitter"
        mock_config.features = mock_features

        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_language_config.return_value = mock_config
            info = factory.get_language_info("Java")
        assert info["name"] == "java"
        assert info["display_name"] == "Java"
        assert info["extensions"] == [".java"]
        assert info["parser_type"] == "tree_sitter"
        assert info["features"] == {"syntax_parsing": True}

    def test_info_from_registry_error_falls_back_legacy(self, factory):
        """Falls back to legacy info when registry raises."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = True
            mock_reg.get_language_config.side_effect = RuntimeError("boom")
            info = factory.get_language_info("python")
        assert info is not None
        assert info["name"] == "python"

    def test_info_legacy_python(self, factory):
        """Returns legacy info for python."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            info = factory.get_language_info("python")
        assert info["name"] == "python"
        assert info["display_name"] == "Python"
        assert ".py" in info["extensions"]

    def test_info_legacy_javascript(self, factory):
        """Returns legacy info for javascript."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            info = factory.get_language_info("javascript")
        assert info["name"] == "javascript"
        assert ".js" in info["extensions"]

    def test_info_legacy_typescript(self, factory):
        """Returns legacy info for typescript."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            info = factory.get_language_info("typescript")
        assert info["name"] == "typescript"
        assert ".ts" in info["extensions"]

    def test_info_unsupported_returns_none(self, factory):
        """Returns None for unsupported language."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            info = factory.get_language_info("cobol")
        assert info is None

    def test_info_case_insensitive(self, factory):
        """Language name is lowercased."""
        with patch("processing.multi_language_factory.language_registry") as mock_reg:
            mock_reg.is_language_supported.return_value = False
            info = factory.get_language_info("PYTHON")
        assert info["name"] == "python"


class TestCreateProcessorSet:
    """Tests for create_processor_set method."""

    def test_returns_codebase_processor_set(self, factory):
        """create_processor_set returns a CodebaseProcessorSet."""
        ps = factory.create_processor_set()
        assert isinstance(ps, CodebaseProcessorSet)
        assert ps.factory is factory


# ===========================================================================
# CodebaseProcessorSet tests
# ===========================================================================


class TestCodebaseProcessorSetAnalyze:
    """Tests for CodebaseProcessorSet.analyze_codebase_async."""

    @staticmethod
    def _analyze_patches(gdc, service_cls):
        """Patch the deferred imports in analyze_codebase_async.

        Source imports ``from db.facade import GraphDatabaseConnection`` and
        ``from services.code_graph.multi_language_graph_service import
        MultiLanguageGraphService`` — both bound in the consuming module's
        namespace at call time, so we patch those exact targets.
        """
        _facade = types.ModuleType("db.facade")
        _facade.GraphDatabaseConnection = gdc
        _svc = types.ModuleType("services.code_graph.multi_language_graph_service")
        _svc.MultiLanguageGraphService = service_cls
        return patch.dict(
            sys.modules,
            {
                "db.facade": _facade,
                "services.code_graph.multi_language_graph_service": _svc,
            },
        )

    async def test_analyze_nonexistent_path(self, factory):
        """Returns failed status for non-existent codebase path."""
        ps = CodebaseProcessorSet(factory)

        mock_db_conn = AsyncMock()
        mock_db_conn.close = AsyncMock()
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        _gdc = MagicMock(return_value=mock_db_conn)
        _service_cls = MagicMock(return_value=mock_service)

        with self._analyze_patches(_gdc, _service_cls):
            result = await ps.analyze_codebase_async(
                codebase_id="test",
                codebase_path="/nonexistent/path/that/does/not/exist",
            )
        assert result["status"] == "failed"
        assert "does not exist" in result["error"]
        assert result["nodes_created"] == 0

    async def test_analyze_success(self, factory, tmp_path):
        """Returns success with result data on happy path."""
        ps = CodebaseProcessorSet(factory)

        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_service.analyze_directory = AsyncMock(
            return_value={
                "total_entities": 5,
                "total_relationships": 3,
                "files_processed": 2,
                "languages_detected": ["python"],
                "session_id": "sess123",
            }
        )
        _service_cls = MagicMock(return_value=mock_service)
        mock_db_conn = AsyncMock()
        mock_db_conn.close = AsyncMock()
        _gdc = MagicMock(return_value=mock_db_conn)

        with self._analyze_patches(_gdc, _service_cls):
            # Create the codebase path so it "exists"
            codebase = tmp_path / "codebase"
            codebase.mkdir()
            result = await ps.analyze_codebase_async(
                codebase_id="test",
                codebase_path=str(codebase),
            )
        assert result["status"] == "success"
        assert result["nodes_created"] == 5
        assert result["edges_created"] == 3
        assert result["files_processed"] == 2
        assert result["languages_detected"] == ["python"]
        assert result["session_id"] == "sess123"

    async def test_analyze_exception(self, factory):
        """Returns failed status on general exception (service init blows up)."""
        ps = CodebaseProcessorSet(factory)

        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock(side_effect=RuntimeError("db error"))
        _service_cls = MagicMock(return_value=mock_service)
        _gdc = MagicMock(return_value=AsyncMock())

        with self._analyze_patches(_gdc, _service_cls):
            result = await ps.analyze_codebase_async(
                codebase_id="test",
                codebase_path="/some/path",
            )
        assert result["status"] == "failed"
        assert "db error" in result["error"]
        assert result["nodes_created"] == 0


class TestCodebaseProcessorSetMinimalGraph:
    """Tests for CodebaseProcessorSet.create_minimal_graph."""

    @staticmethod
    def _make_minimal_patches(mock_db_conn, mock_mapper_cls):
        """Build a patch.dict for the deferred imports in create_minimal_graph.

        Source imports the postgres facade + entity mapper from ``db.*``
        (``coding_client.db`` was deleted in the cutover) and the core models
        from ``models.core_models``; we patch those exact module targets.
        """
        _gdc = MagicMock(return_value=mock_db_conn)

        class _FakeEntityType:
            FILE = "file"

        class _FakeCoreCodeEntity:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        _fake_facade = types.ModuleType("db.facade")
        _fake_facade.GraphDatabaseConnection = _gdc
        _fake_mapper = types.ModuleType("db.entity_mapper")
        _fake_mapper.MultiLanguageEntityMapper = mock_mapper_cls
        _fake_core = types.ModuleType("models.core_models")
        _fake_core.CoreCodeEntity = _FakeCoreCodeEntity
        _fake_core.EntityType = _FakeEntityType

        return patch.dict(
            sys.modules,
            {
                "db.facade": _fake_facade,
                "db.entity_mapper": _fake_mapper,
                "models.core_models": _fake_core,
            },
        )

    async def test_minimal_graph_nonexistent_path(self, factory):
        """Returns failed status for non-existent path."""
        ps = CodebaseProcessorSet(factory)
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock()
        mock_db_conn.get_connection = MagicMock(return_value=MagicMock())
        mock_db_conn.close = AsyncMock()

        mock_mapper_cls = MagicMock(return_value=MagicMock())

        with self._make_minimal_patches(mock_db_conn, mock_mapper_cls):
            result = await ps.create_minimal_graph(
                codebase_id="test",
                codebase_path="/nonexistent/path/that/does/not/exist",
            )
        assert result["status"] == "failed"
        assert "does not exist" in result["error"]
        assert result["nodes_created"] == 0

    async def test_minimal_graph_success(self, factory, tmp_path):
        """Creates minimal graph for supported files."""
        ps = CodebaseProcessorSet(factory)
        with patch.object(factory, "get_supported_extensions", return_value=[".py", ".js"]):
            codebase = tmp_path / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("print('hi')")
            (codebase / "app.js").write_text("console.log('hi')")
            (codebase / "readme.txt").write_text("not code")

            mock_conn = MagicMock()
            mock_conn.commit = AsyncMock()
            mock_db_conn = MagicMock()
            mock_db_conn.connect = AsyncMock()
            mock_db_conn.get_connection = MagicMock(return_value=mock_conn)
            mock_db_conn.close = AsyncMock()

            mock_mapper = AsyncMock()
            mock_mapper_cls = MagicMock(return_value=mock_mapper)

            with self._make_minimal_patches(mock_db_conn, mock_mapper_cls):
                result = await ps.create_minimal_graph(
                    codebase_id="test",
                    codebase_path=str(codebase),
                )
        assert result["status"] == "success"
        assert result["nodes_created"] == 2  # .py and .js, not .txt
        assert result["edges_created"] == 0

    async def test_minimal_graph_exception(self, factory):
        """Returns failed status on exception (connect() blows up)."""
        ps = CodebaseProcessorSet(factory)

        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock(side_effect=RuntimeError("db fail"))
        mock_db_conn.close = AsyncMock()
        mock_mapper_cls = MagicMock()

        with self._make_minimal_patches(mock_db_conn, mock_mapper_cls):
            result = await ps.create_minimal_graph(
                codebase_id="test",
                codebase_path="/some/path",
            )
        assert result["status"] == "failed"
        assert "db fail" in result["error"]
        assert result["nodes_created"] == 0

    async def test_minimal_graph_exception_with_db_connection(self, factory, tmp_path):
        """Closes db_connection in exception handler when it exists."""
        ps = CodebaseProcessorSet(factory)
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock()
        mock_db_conn.get_connection = MagicMock(side_effect=RuntimeError("conn error"))
        mock_db_conn.close = AsyncMock()

        mock_mapper_cls = MagicMock()

        with self._make_minimal_patches(mock_db_conn, mock_mapper_cls):
            result = await ps.create_minimal_graph(
                codebase_id="test",
                codebase_path=str(tmp_path),
            )
        assert result["status"] == "failed"
        mock_db_conn.close.assert_awaited()

    async def test_minimal_graph_exception_close_fails(self, factory, tmp_path):
        """Handles failure when closing db_connection in exception handler."""
        ps = CodebaseProcessorSet(factory)
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock()
        mock_db_conn.get_connection = MagicMock(side_effect=RuntimeError("conn error"))
        mock_db_conn.close = AsyncMock(side_effect=RuntimeError("close fail"))

        mock_mapper_cls = MagicMock()

        with self._make_minimal_patches(mock_db_conn, mock_mapper_cls):
            result = await ps.create_minimal_graph(
                codebase_id="test",
                codebase_path=str(tmp_path),
            )
        assert result["status"] == "failed"
        assert "conn error" in result["error"]


# ===========================================================================
# Module-level helper tests
# ===========================================================================


class TestGetMultiLanguageFactory:
    """Tests for get_multi_language_factory()."""

    async def test_creates_factory_on_first_call(self):
        """Creates and caches factory on first call."""
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = await get_multi_language_factory()
        assert isinstance(f, MultiLanguageProcessorFactory)

    async def test_returns_same_instance(self):
        """Returns cached factory on subsequent calls."""
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f1 = await get_multi_language_factory()
            f2 = await get_multi_language_factory()
        assert f1 is f2

    async def test_accepts_custom_file_service(self):
        """Uses provided file_service."""
        fs = MagicMock()
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = await get_multi_language_factory(file_service=fs)
        assert f.file_service is fs

    async def test_creates_default_file_service_when_none(self):
        """Creates default FileService when file_service is None."""
        with patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"):
            f = await get_multi_language_factory(file_service=None)
        assert f.file_service is not None


class TestModuleLevelHelpers:
    """Tests for module-level convenience functions."""

    async def test_get_processor_for_file_delegates(self):
        """mod-level get_processor_for_file delegates to factory."""
        with (
            patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"),
            patch("processing.multi_language_factory.language_registry") as mock_reg,
        ):
            mock_reg.detect_language.return_value = None
            mock_reg.is_language_supported.return_value = False
            result = await mod_get_processor_for_file("/foo/bar.py")
        # Should return a processor for .py
        assert result is not None

    async def test_get_processor_for_language_delegates(self):
        """mod-level get_processor_for_language delegates to factory."""
        with (
            patch.object(MultiLanguageProcessorFactory, "_initialize_language_processors"),
            patch("processing.multi_language_factory.language_registry") as mock_reg,
        ):
            mock_reg.is_language_supported.return_value = False
            result = await mod_get_processor_for_language("python")
        assert result is not None

    def test_get_supported_languages_sync(self):
        """Synchronous get_supported_languages returns known languages."""
        langs = mod_get_supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs
        assert "java" in langs
        assert "rust" in langs
        assert "go" in langs
        assert "csharp" in langs
        assert "cpp" in langs
        assert "c" in langs


class TestBackwardCompatibilityAlias:
    """Test MultiLanguageFactory alias."""

    def test_alias_is_same_class(self):
        """MultiLanguageFactory is an alias for MultiLanguageProcessorFactory."""
        assert MultiLanguageFactory is MultiLanguageProcessorFactory
