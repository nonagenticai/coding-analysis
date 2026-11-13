"""Unit tests for processing.multi_language_processor module."""

from __future__ import annotations

from pathlib import Path

import pytest

# coding_client.* stubs are installed by conftest.py
from models.core_models import EntityType
from processing.languages.base import (
    LanguageConfig,
    LanguageFeatures,
    ParseResult,
    ParserType,
)
from processing.multi_language_processor import (
    FallbackManager,
    MultiLanguageProcessor,
    VersionDetectorRegistry,
    detect_java_version,
    detect_python_version,
    detect_rust_version,
    ensure_parse_result,
    regex_fallback_strategy,
    structure_only_fallback,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _make_config(name: str = "python") -> LanguageConfig:
    return LanguageConfig(
        name=name,
        display_name=name.capitalize(),
        extensions=[".py"],
        parser_type=ParserType.REGEX_FALLBACK,
        features=LanguageFeatures(),
    )


@pytest.fixture
def version_registry():
    return VersionDetectorRegistry()


@pytest.fixture
def fallback_mgr():
    return FallbackManager()


@pytest.fixture
def processor(version_registry, fallback_mgr):
    return MultiLanguageProcessor(
        config=_make_config("python"),
        version_detector=version_registry,
        fallback_mgr=fallback_mgr,
    )


# ===========================================================================
# ensure_parse_result
# ===========================================================================


class TestEnsureParseResult:
    """Tests for ensure_parse_result utility."""

    def test_returns_existing_parse_result_unchanged(self):
        pr = ParseResult(
            success=True,
            file_path="a.py",
            language="python",
            parser_used=ParserType.REGEX_FALLBACK,
        )
        result = ensure_parse_result(pr, Path("a.py"), "python")
        assert result is pr

    def test_converts_dict_with_success_true(self):
        d = {"success": True, "entities": [], "relationships": [], "lines_of_code": 42}
        result = ensure_parse_result(d, Path("b.py"), "python", "test ctx")
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.lines_of_code == 42
        assert result.language == "python"
        assert result.parser_used == ParserType.REGEX_FALLBACK

    def test_converts_dict_with_success_false_default(self):
        d = {}
        result = ensure_parse_result(d, Path("c.py"), "java")
        assert result.success is False
        assert result.language == "java"

    def test_converts_dict_preserves_optional_fields(self):
        d = {
            "success": True,
            "ast": {"type": "Module"},
            "complexity_score": 3.5,
            "parse_time_ms": 12.0,
            "warnings": ["w1"],
            "properties": {"x": 1},
        }
        result = ensure_parse_result(d, Path("d.py"), "go")
        assert result.ast == {"type": "Module"}
        assert result.complexity_score == 3.5
        assert result.parse_time_ms == 12.0
        assert result.warnings == ["w1"]
        assert result.properties == {"x": 1}

    def test_converts_unknown_type_to_failure(self):
        result = ensure_parse_result(12345, Path("e.py"), "rust", "bad")
        assert isinstance(result, ParseResult)
        assert result.success is False
        assert "Invalid result type" in result.errors[0]

    def test_converts_none_to_failure(self):
        result = ensure_parse_result(None, Path("f.py"), "csharp")
        assert result.success is False


# ===========================================================================
# VersionDetectorRegistry
# ===========================================================================


class TestVersionDetectorRegistry:
    """Tests for VersionDetectorRegistry."""

    async def test_detect_version_no_detector_registered(self, version_registry):
        result = await version_registry.detect_version("unknown_lang", Path("x.py"), "code")
        assert result is None

    async def test_detect_version_returns_string(self, version_registry):
        async def detector(fp, content):
            return "3.12"

        version_registry.register_detector("python", detector)
        result = await version_registry.detect_version("python", Path("a.py"), "pass")
        assert result == "3.12"

    async def test_detect_version_returns_none_for_non_string(self, version_registry):
        async def detector(fp, content):
            return 42  # not a string

        version_registry.register_detector("python", detector)
        result = await version_registry.detect_version("python", Path("a.py"), "pass")
        assert result is None

    async def test_detect_version_handles_runtime_error(self, version_registry):
        async def bad_detector(fp, content):
            raise RuntimeError("boom")

        version_registry.register_detector("python", bad_detector)
        result = await version_registry.detect_version("python", Path("a.py"), "pass")
        assert result is None

    async def test_detect_version_handles_value_error(self, version_registry):
        async def bad_detector(fp, content):
            raise ValueError("bad value")

        version_registry.register_detector("python", bad_detector)
        result = await version_registry.detect_version("python", Path("a.py"), "pass")
        assert result is None

    async def test_detect_version_handles_os_error(self, version_registry):
        async def bad_detector(fp, content):
            raise OSError("io fail")

        version_registry.register_detector("python", bad_detector)
        result = await version_registry.detect_version("python", Path("a.py"), "pass")
        assert result is None

    def test_register_detector_overwrites(self, version_registry):
        async def d1(fp, content):
            return "v1"

        async def d2(fp, content):
            return "v2"

        version_registry.register_detector("go", d1)
        version_registry.register_detector("go", d2)
        assert version_registry._detectors["go"] is d2


# ===========================================================================
# FallbackManager
# ===========================================================================


class TestFallbackManager:
    """Tests for FallbackManager."""

    async def test_handle_failure_no_strategies(self, fallback_mgr):
        result = await fallback_mgr.handle_failure(
            "unknown", Path("x.py"), "code", RuntimeError("fail")
        )
        assert result.success is False
        assert "Primary parser failed" in result.errors[0]
        assert "All fallback strategies failed" in result.errors[1]

    async def test_handle_failure_first_strategy_succeeds(self, fallback_mgr):
        async def good_strategy(fp, content, error, language="unknown"):
            return ParseResult(
                success=True,
                file_path=str(fp),
                language=language,
                parser_used=ParserType.REGEX_FALLBACK,
            )

        fallback_mgr.register_fallback("python", good_strategy)
        result = await fallback_mgr.handle_failure(
            "python", Path("a.py"), "code", RuntimeError("fail")
        )
        assert result.success is True

    async def test_handle_failure_skips_unsuccessful_strategy(self, fallback_mgr):
        async def bad_strategy(fp, content, error, language="unknown"):
            return ParseResult(
                success=False,
                file_path=str(fp),
                language=language,
                parser_used=ParserType.REGEX_FALLBACK,
                errors=["nope"],
            )

        async def good_strategy(fp, content, error, language="unknown"):
            return ParseResult(
                success=True,
                file_path=str(fp),
                language=language,
                parser_used=ParserType.REGEX_FALLBACK,
            )

        fallback_mgr.register_fallback("python", bad_strategy)
        fallback_mgr.register_fallback("python", good_strategy)
        result = await fallback_mgr.handle_failure(
            "python", Path("a.py"), "code", RuntimeError("fail")
        )
        assert result.success is True

    async def test_handle_failure_catches_strategy_exception(self, fallback_mgr):
        async def exploding_strategy(fp, content, error, language="unknown"):
            raise RuntimeError("strategy boom")

        fallback_mgr.register_fallback("python", exploding_strategy)
        result = await fallback_mgr.handle_failure(
            "python", Path("a.py"), "code", RuntimeError("fail")
        )
        assert result.success is False

    async def test_handle_failure_strategy_without_language_param(self, fallback_mgr):
        """Strategy that doesn't accept language param triggers TypeError fallback."""

        async def old_strategy(fp, content, error):
            return ParseResult(
                success=True,
                file_path=str(fp),
                language="python",
                parser_used=ParserType.REGEX_FALLBACK,
            )

        fallback_mgr.register_fallback("python", old_strategy)
        result = await fallback_mgr.handle_failure(
            "python", Path("a.py"), "code", RuntimeError("fail")
        )
        assert result.success is True

    async def test_handle_failure_converts_dict_result(self, fallback_mgr):
        async def dict_strategy(fp, content, error, language="unknown"):
            return {"success": True, "entities": [], "relationships": []}

        fallback_mgr.register_fallback("python", dict_strategy)
        result = await fallback_mgr.handle_failure(
            "python", Path("a.py"), "code", RuntimeError("fail")
        )
        assert isinstance(result, ParseResult)
        assert result.success is True

    def test_register_fallback_appends(self, fallback_mgr):
        async def s1(fp, content, error, language="unknown"):
            pass

        async def s2(fp, content, error, language="unknown"):
            pass

        fallback_mgr.register_fallback("go", s1)
        fallback_mgr.register_fallback("go", s2)
        assert len(fallback_mgr._fallback_strategies["go"]) == 2


# ===========================================================================
# MultiLanguageProcessor
# ===========================================================================


class TestMultiLanguageProcessor:
    """Tests for MultiLanguageProcessor."""

    async def test_initialize_impl_no_error(self, processor):
        # _initialize_impl is a no-op, should not raise
        await processor._initialize_impl()

    async def test_parse_file_delegates_to_primary_parse(self):
        vr = VersionDetectorRegistry()
        fm = FallbackManager()

        class TestProcessor(MultiLanguageProcessor):
            async def _primary_parse(self, file_path, content, version):
                return ParseResult(
                    success=True,
                    file_path=str(file_path),
                    language="python",
                    parser_used=ParserType.REGEX_FALLBACK,
                    properties={"detected_version": None},
                )

            async def extract_entities(self, ast, file_path):
                return []

            async def extract_relationships(self, ast, entities, file_path):
                return []

        proc = TestProcessor(config=_make_config(), version_detector=vr, fallback_mgr=fm)
        result = await proc.parse_file(Path("test.py"), "pass")
        assert result.success is True

    async def test_parse_file_uses_fallback_on_primary_failure(self):
        vr = VersionDetectorRegistry()
        fm = FallbackManager()

        async def good_fallback(fp, content, error, language="unknown"):
            return ParseResult(
                success=True,
                file_path=str(fp),
                language=language,
                parser_used=ParserType.REGEX_FALLBACK,
                properties={},
            )

        fm.register_fallback("python", good_fallback)

        class FailProcessor(MultiLanguageProcessor):
            async def _primary_parse(self, file_path, content, version):
                raise RuntimeError("primary fail")

            async def extract_entities(self, ast, file_path):
                return []

            async def extract_relationships(self, ast, entities, file_path):
                return []

        proc = FailProcessor(config=_make_config(), version_detector=vr, fallback_mgr=fm)
        result = await proc.parse_file(Path("test.py"), "pass")
        assert result.success is True

    async def test_parse_file_sets_detected_version(self):
        vr = VersionDetectorRegistry()

        async def py_detector(fp, content):
            return "3.12"

        vr.register_detector("python", py_detector)
        fm = FallbackManager()

        class TestProcessor(MultiLanguageProcessor):
            async def _primary_parse(self, file_path, content, version):
                return ParseResult(
                    success=True,
                    file_path=str(file_path),
                    language="python",
                    parser_used=ParserType.REGEX_FALLBACK,
                    properties={},
                )

            async def extract_entities(self, ast, file_path):
                return []

            async def extract_relationships(self, ast, entities, file_path):
                return []

        proc = TestProcessor(config=_make_config(), version_detector=vr, fallback_mgr=fm)
        result = await proc.parse_file(Path("test.py"), "pass")
        assert result.properties["detected_version"] == "3.12"

    async def test_primary_parse_raises_not_implemented(self, processor):
        with pytest.raises(NotImplementedError):
            await processor._primary_parse(Path("x.py"), "code", None)

    async def test_extract_entities_raises_not_implemented(self, processor):
        with pytest.raises(NotImplementedError):
            await processor.extract_entities(None, Path("x.py"))

    async def test_extract_relationships_raises_not_implemented(self, processor):
        with pytest.raises(NotImplementedError):
            await processor.extract_relationships(None, [], Path("x.py"))


# ===========================================================================
# detect_python_version
# ===========================================================================


class TestDetectPythonVersion:
    """Tests for the default Python version detector."""

    async def test_python3_shebang(self):
        assert await detect_python_version(Path("a.py"), "#!/usr/bin/env python3\npass") == "3.x"

    async def test_python2_shebang(self):
        assert await detect_python_version(Path("a.py"), "#!/usr/bin/env python2\npass") == "2.x"

    async def test_python3_version_in_shebang(self):
        assert await detect_python_version(Path("a.py"), "#!/usr/bin/python3.10\npass") == "3.x"

    async def test_python2_version_in_shebang(self):
        assert await detect_python_version(Path("a.py"), "# python 2.7\npass") == "2.x"

    async def test_fstring_detection(self):
        result = await detect_python_version(Path("a.py"), 'x = f"hello {name}"')
        # f-strings contain '"' which triggers "3.6+"
        assert result == "3.6+"

    async def test_async_def_detection(self):
        # Content must NOT contain '"' (would match f-string first)
        result = await detect_python_version(Path("a.py"), "async def foo():\n    pass")
        assert result == "3.5+"

    async def test_await_detection(self):
        result = await detect_python_version(Path("a.py"), "result = await coro()")
        assert result == "3.5+"

    async def test_yield_from_detection(self):
        result = await detect_python_version(Path("a.py"), "yield from gen()")
        assert result == "3.3+"

    async def test_no_features_returns_none(self):
        result = await detect_python_version(Path("a.py"), "x = 1 + 2")
        assert result is None


# ===========================================================================
# detect_java_version
# ===========================================================================


class TestDetectJavaVersion:
    """Tests for the default Java version detector."""

    async def test_var_keyword(self):
        assert await detect_java_version(Path("A.java"), "var x = 10;") == "10+"

    async def test_record_keyword(self):
        assert await detect_java_version(Path("A.java"), "record Point(int x, int y) {}") == "14+"

    async def test_sealed_keyword(self):
        assert await detect_java_version(Path("A.java"), "sealed class Shape {}") == "17+"

    async def test_switch_expression(self):
        code = "switch (x) { case 1 -> 10; }"
        assert await detect_java_version(Path("A.java"), code) == "14+"

    async def test_module_keyword(self):
        assert await detect_java_version(Path("m.java"), "module com.example {}") == "9+"

    async def test_default_java8(self):
        assert await detect_java_version(Path("A.java"), "public class A {}") == "8+"


# ===========================================================================
# detect_rust_version
# ===========================================================================


class TestDetectRustVersion:
    """Tests for the default Rust version detector."""

    async def test_async_fn_and_async_move(self):
        # Requires BOTH "async fn" AND "async move" to trigger 1.39+
        code = "async fn foo() { let f = async move { 42 }; }"
        assert await detect_rust_version(Path("main.rs"), code) == "1.39+"

    async def test_async_fn_alone_not_enough(self):
        # Only "async fn" without "async move" does not trigger 1.39+
        assert await detect_rust_version(Path("main.rs"), "async fn foo() {}") == "1.0+"

    async def test_const_fn(self):
        assert (
            await detect_rust_version(Path("lib.rs"), "const fn compute() -> u32 { 42 }") == "1.31+"
        )

    async def test_dyn_keyword(self):
        assert await detect_rust_version(Path("lib.rs"), "fn f(x: &dyn Trait) {}") == "1.27+"

    async def test_default_rust(self):
        assert await detect_rust_version(Path("lib.rs"), "fn main() {}") == "1.0+"

    async def test_cargo_toml_2021_edition(self, tmp_path):
        main_rs = tmp_path / "main.rs"
        main_rs.write_text("fn main() {}")
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_toml.write_text('[package]\nedition = "2021"\n')
        result = await detect_rust_version(main_rs, "fn main() {}")
        assert result == "1.56+"

    async def test_cargo_toml_2018_edition(self, tmp_path):
        main_rs = tmp_path / "main.rs"
        main_rs.write_text("fn main() {}")
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_toml.write_text('[package]\nedition = "2018"\n')
        result = await detect_rust_version(main_rs, "fn main() {}")
        assert result == "1.31+"

    async def test_cargo_toml_not_found(self, tmp_path):
        main_rs = tmp_path / "main.rs"
        main_rs.write_text("fn main() {}")
        # no Cargo.toml
        result = await detect_rust_version(main_rs, "fn main() {}")
        assert result == "1.0+"


# ===========================================================================
# regex_fallback_strategy
# ===========================================================================


class TestRegexFallbackStrategy:
    """Tests for regex_fallback_strategy."""

    async def test_extracts_functions(self):
        code = "def hello():\n    pass\ndef world():\n    pass"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"), "python")
        assert result.success is True
        func_names = [e.name for e in result.entities if e.type == EntityType.FUNCTION]
        assert "hello" in func_names
        assert "world" in func_names

    async def test_extracts_classes(self):
        code = "class Foo:\n    pass\nclass Bar:\n    pass"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"), "python")
        class_names = [e.name for e in result.entities if e.type == EntityType.CLASS]
        assert "Foo" in class_names
        assert "Bar" in class_names

    async def test_counts_lines(self):
        code = "line1\nline2\nline3"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"))
        assert result.lines_of_code == 3

    async def test_empty_content(self):
        result = await regex_fallback_strategy(Path("a.py"), "", RuntimeError("err"))
        assert result.success is True
        assert result.entities == []

    async def test_mixed_entities(self):
        code = "class MyClass:\n    def my_method(self):\n        pass"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"), "python")
        names = [e.name for e in result.entities]
        assert "MyClass" in names
        assert "my_method" in names

    async def test_language_parameter_used(self):
        code = "def f(): pass"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"), "go")
        for entity in result.entities:
            assert entity.language == "go"

    async def test_line_numbers_computed(self):
        code = "x = 1\ny = 2\ndef foo():\n    pass"
        result = await regex_fallback_strategy(Path("a.py"), code, RuntimeError("err"), "python")
        funcs = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].line_number == 3


# ===========================================================================
# structure_only_fallback
# ===========================================================================


class TestStructureOnlyFallback:
    """Tests for structure_only_fallback."""

    async def test_creates_file_entity(self):
        code = "some code here\nline2"
        result = await structure_only_fallback(Path("mod.py"), code, RuntimeError("err"), "python")
        assert result.success is True
        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity.type == EntityType.FILE
        assert entity.name == "mod.py"

    async def test_captures_metadata(self):
        code = "a\nb\nc"
        err = RuntimeError("parse failed")
        result = await structure_only_fallback(Path("x.py"), code, err, "python")
        entity = result.entities[0]
        assert entity.properties["file_size"] == len(code)
        assert entity.properties["line_count"] == 3
        assert "parse failed" in entity.properties["parse_error"]

    async def test_lines_of_code(self):
        code = "line1\nline2"
        result = await structure_only_fallback(Path("x.py"), code, RuntimeError("err"))
        assert result.lines_of_code == 2

    async def test_language_parameter(self):
        result = await structure_only_fallback(
            Path("x.rs"), "fn main() {}", RuntimeError("err"), "rust"
        )
        assert result.language == "rust"
        assert result.entities[0].language == "rust"


# ===========================================================================
# MultiLanguageAnalysisProcessor tests (multi_language_analysis_processor.py)
# Covering uncovered lines: 66-97, 109-139, 161-171, 193-223
# ===========================================================================

from unittest.mock import AsyncMock, MagicMock, patch


class TestMultiLanguageAnalysisProcessor:
    """Tests for processing.multi_language_analysis_processor.MultiLanguageAnalysisProcessor."""

    @pytest.mark.asyncio
    async def test_mlap_process_exception_handling(self):
        """process() exception path (lines 93-97)."""
        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.settings", mock_settings),
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler") as mock_teh,
            patch(
                "processing.multi_language_analysis_processor.GraphDatabaseConnection"
            ) as mock_gdb,
            patch(
                "processing.multi_language_analysis_processor.TaskParameterValidator"
            ) as mock_tpv,
            patch("processing.multi_language_analysis_processor.FileSystemValidator") as mock_fsv,
            patch("processing.multi_language_analysis_processor.PerformanceUtils") as mock_perf,
            patch(
                "processing.multi_language_analysis_processor.LanguageDetectionUtils"
            ) as mock_ldu,
        ):
            mock_gs.return_value = mock_settings
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_tpv.validate_multi_language_analysis_params.return_value = {
                "is_valid": True,
                "errors": [],
                "validated_params": {
                    "code_graph_id": "test",
                    "codebase_path": "/tmp/test",
                },
            }
            mock_fsv.validate_codebase_path.return_value = {"is_valid": True}
            mock_ldu.initialize_multi_language_factory = AsyncMock(return_value=MagicMock())
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.get_connection.return_value = AsyncMock()
            mock_db.close = AsyncMock()
            mock_gdb.return_value = mock_db
            mock_perf.create_timer.side_effect = RuntimeError("timer error")
            error_handler = MagicMock()
            error_handler.create_error_response.return_value = {
                "success": False,
                "error": "timer error",
            }
            mock_teh.return_value = error_handler

            proc = MultiLanguageAnalysisProcessor()
            proc.error_handler = error_handler
            result = await proc.process({"codebase_path": "/tmp/test", "code_graph_id": "test"})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_mlap_process_languages_no_processor(self):
        """_process_languages when processor is None returns error dict."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler") as mock_teh,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_factory = AsyncMock()
            mock_factory.get_processor = AsyncMock(return_value=None)
            error_handler = MagicMock()
            mock_teh.return_value = error_handler

            proc = MultiLanguageAnalysisProcessor()
            proc.error_handler = error_handler

            ctx = {
                "language_stats": {"python": 5},
                "supported_files": [],
                "factory": mock_factory,
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_languages(ctx)
            assert "python" in result
            assert result["python"]["error"] == "No processor available"

    @pytest.mark.asyncio
    async def test_mlap_process_languages_exception(self):
        """_process_languages when processor raises returns error dict."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler") as mock_teh,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_factory = AsyncMock()
            mock_factory.get_processor = AsyncMock(side_effect=RuntimeError("processor error"))
            error_handler = MagicMock()
            mock_teh.return_value = error_handler

            proc = MultiLanguageAnalysisProcessor()
            proc.error_handler = error_handler

            ctx = {
                "language_stats": {"python": 5},
                "supported_files": [],
                "factory": mock_factory,
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_languages(ctx)
            assert "python" in result
            assert "error" in result["python"]

    @pytest.mark.asyncio
    async def test_mlap_process_language_sample(self):
        """_process_language_sample processes sample files."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch(
                "processing.multi_language_analysis_processor.AnalysisMetadataManager"
            ) as mock_amm,
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_factory = MagicMock()
            mock_factory.get_extension_language_map.return_value = {".py": "python"}
            mock_factory.get_language_info.return_value = {"name": "python"}
            mock_processor = MagicMock()
            mock_processor.__class__.__name__ = "PythonProcessor"

            amm_instance = MagicMock()
            amm_instance.store_language_analysis = AsyncMock()
            mock_amm.return_value = amm_instance

            proc = MultiLanguageAnalysisProcessor()
            proc.metadata_manager = amm_instance

            ctx = {
                "language": "python",
                "file_count": 5,
                "supported_files": [Path("/tmp/test.py"), Path("/tmp/other.js")],
                "factory": mock_factory,
                "processor": mock_processor,
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_language_sample(ctx)
            assert result["total_files"] == 5
            assert result["processor_type"] == "PythonProcessor"

    @pytest.mark.asyncio
    async def test_mlap_process_validation_failure(self):
        """process() returns errors on validation failure (lines 49-54)."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch(
                "processing.multi_language_analysis_processor.TaskParameterValidator"
            ) as mock_tpv,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_tpv.validate_multi_language_analysis_params.return_value = {
                "is_valid": False,
                "errors": ["missing code_graph_id"],
            }

            proc = MultiLanguageAnalysisProcessor()
            result = await proc.process({})
            assert result["success"] is False
            assert "missing code_graph_id" in result["errors"]

    @pytest.mark.asyncio
    async def test_mlap_process_fs_validation_failure(self):
        """process() returns error on filesystem validation failure (lines 62-63)."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch(
                "processing.multi_language_analysis_processor.TaskParameterValidator"
            ) as mock_tpv,
            patch("processing.multi_language_analysis_processor.FileSystemValidator") as mock_fsv,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            mock_tpv.validate_multi_language_analysis_params.return_value = {
                "is_valid": True,
                "errors": [],
                "validated_params": {
                    "code_graph_id": "test",
                    "codebase_path": "/nonexistent",
                },
            }
            mock_fsv.validate_codebase_path.return_value = {
                "is_valid": False,
                "error": "path does not exist",
            }

            proc = MultiLanguageAnalysisProcessor()
            result = await proc.process({"codebase_path": "/nonexistent", "code_graph_id": "test"})
            assert result["success"] is False
            assert result["error"] == "path does not exist"

    def test_mlap_get_supported_languages(self):
        """get_supported_languages delegates correctly (line 41)."""
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import (
                MultiLanguageAnalysisProcessor,
            )

            proc = MultiLanguageAnalysisProcessor()
            langs = proc.get_supported_languages()
            assert isinstance(langs, list)
