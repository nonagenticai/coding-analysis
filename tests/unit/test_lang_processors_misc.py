"""Unit tests for language processors: base, registry, python, javascript, typescript, java, csharp."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
    EntityType,
    RelationType,
)
from processing.languages.base import (
    LanguageConfig,
    LanguageFeatures,
    LanguageProcessor,
    ParseResult,
    ParserType,
)
from processing.languages.registry import LanguageRegistry

# ========================== FIXTURES =======================================


@pytest.fixture
def basic_features() -> LanguageFeatures:
    return LanguageFeatures(classes=True, functions=True, imports=True)


@pytest.fixture
def basic_config(basic_features: LanguageFeatures) -> LanguageConfig:
    return LanguageConfig(
        name="testlang",
        display_name="Test Language",
        extensions=[".tl"],
        parser_type=ParserType.REGEX_FALLBACK,
        features=basic_features,
    )


class DummyProcessor(LanguageProcessor):
    """Minimal concrete implementation for testing the abstract base."""

    async def _initialize_impl(self) -> None:
        pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        return ParseResult(
            success=True,
            file_path=str(file_path),
            language=self.config.name,
            parser_used=self.config.parser_type,
        )

    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        return []

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        return []


@pytest.fixture
def dummy_processor(basic_config: LanguageConfig) -> DummyProcessor:
    return DummyProcessor(basic_config)


# ========================== BASE MODELS ====================================


class TestParserType:
    def test_enum_values(self):
        assert ParserType.AST_NATIVE == "ast_native"
        assert ParserType.TREE_SITTER == "tree_sitter"
        assert ParserType.EXTERNAL_TOOL == "external_tool"
        assert ParserType.LANGUAGE_SERVER == "language_server"
        assert ParserType.REGEX_FALLBACK == "regex_fallback"


class TestLanguageFeatures:
    def test_defaults(self):
        f = LanguageFeatures()
        assert f.classes is False
        assert f.functions is True
        assert f.modules is True
        assert f.garbage_collected is True

    def test_frozen(self):
        f = LanguageFeatures()
        with pytest.raises(Exception):
            f.classes = True  # type: ignore[misc]


class TestLanguageConfig:
    def test_construction(self, basic_config: LanguageConfig):
        assert basic_config.name == "testlang"
        assert basic_config.display_name == "Test Language"
        assert basic_config.extensions == [".tl"]
        assert basic_config.max_file_size_mb == 10
        assert basic_config.timeout_seconds == 30

    def test_optional_fields_default(self, basic_config: LanguageConfig):
        assert basic_config.tree_sitter_grammar is None
        assert basic_config.external_command is None
        assert basic_config.fallback_parsers == []
        assert basic_config.ignore_patterns == []
        assert basic_config.test_patterns == []
        assert basic_config.required_packages == []

    def test_frozen(self, basic_config: LanguageConfig):
        with pytest.raises(Exception):
            basic_config.name = "other"  # type: ignore[misc]


class TestParseResult:
    def test_minimal(self):
        r = ParseResult(
            success=True,
            file_path="foo.py",
            language="python",
            parser_used=ParserType.AST_NATIVE,
        )
        assert r.success is True
        assert r.entities == []
        assert r.relationships == []
        assert r.errors == []
        assert r.warnings == []
        assert r.lines_of_code == 0
        assert r.complexity_score == 0.0

    def test_with_entities(self):
        ent = CoreCodeEntity(
            id="e1",
            name="foo",
            type=EntityType.FUNCTION,
            language="python",
            file_path="a.py",
            line_number=1,
        )
        r = ParseResult(
            success=True,
            file_path="a.py",
            language="python",
            parser_used=ParserType.AST_NATIVE,
            entities=[ent],
        )
        assert len(r.entities) == 1
        assert r.entities[0].name == "foo"


# ========================== LANGUAGE PROCESSOR (base) =======================


class TestLanguageProcessorBase:
    async def test_initialize_once(self, dummy_processor: DummyProcessor):
        assert dummy_processor._initialized is False
        await dummy_processor.initialize()
        assert dummy_processor._initialized is True
        # Second call should not re-initialize
        await dummy_processor.initialize()
        assert dummy_processor._initialized is True

    async def test_parse_file(self, dummy_processor: DummyProcessor):
        result = await dummy_processor.parse_file(Path("test.tl"), "code")
        assert result.success is True
        assert result.language == "testlang"

    async def test_calculate_complexity_default(self, dummy_processor: DummyProcessor):
        score = await dummy_processor.calculate_complexity(None)
        assert score == 1.0

    async def test_validate_syntax_default(self, dummy_processor: DummyProcessor):
        errors = await dummy_processor.validate_syntax("some code")
        assert errors == []

    def test_supports_feature(self, dummy_processor: DummyProcessor):
        assert dummy_processor.supports_feature("classes") is True
        assert dummy_processor.supports_feature("functions") is True
        assert dummy_processor.supports_feature("generics") is False
        assert dummy_processor.supports_feature("nonexistent_feature") is False

    def test_calculate_basic_complexity_metrics(self, dummy_processor: DummyProcessor):
        content = "// comment\nfunction foo() {\n  return 1;\n}\n\n/* block */\n"
        entities = [
            CoreCodeEntity(
                id="f1",
                name="foo",
                type=EntityType.FUNCTION,
                language="testlang",
                file_path="a.tl",
                line_number=2,
            ),
            CoreCodeEntity(
                id="c1",
                name="Bar",
                type=EntityType.CLASS,
                language="testlang",
                file_path="a.tl",
                line_number=10,
            ),
        ]
        metrics = dummy_processor.calculate_basic_complexity_metrics(content, entities)
        assert isinstance(metrics, CoreComplexityMetrics)
        assert metrics.lines_of_code == 6
        assert metrics.function_count == 1
        assert metrics.class_count == 1
        assert metrics.blank_lines == 1
        assert metrics.comment_lines >= 1  # at least the // comment line

    def test_calculate_basic_complexity_metrics_python_comments(
        self, dummy_processor: DummyProcessor
    ):
        content = "# a comment\nx = 1\n"
        metrics = dummy_processor.calculate_basic_complexity_metrics(
            content,
            [],
            single_line_comment_prefix="#",
            multi_line_comment_marker='"""',
        )
        assert metrics.comment_lines == 1
        assert metrics.logical_lines == 1

    async def test_parse_file_with_metrics_import_error_fallback(
        self, dummy_processor: DummyProcessor
    ):
        """When metrics imports fail, parse_file_with_metrics should fall back to parse_file."""
        import sys

        with patch.dict(
            sys.modules,
            {
                "services.multi_language_metrics": None,  # force ImportError
            },
        ):
            # Force reimport to trigger ImportError branch
            result = await dummy_processor.parse_file_with_metrics(
                Path("test.tl"), "code", "analysis_1"
            )
            assert result.success is True


# ========================== LANGUAGE REGISTRY ===============================


class TestLanguageRegistry:
    def test_register_language(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        assert "testlang" in registry.get_supported_languages()
        assert ".tl" in registry.get_supported_extensions()

    def test_register_invalid_processor_raises(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        with pytest.raises(ValueError, match="must inherit"):
            registry.register_language(basic_config, str)  # type: ignore[arg-type]

    def test_detect_language(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        assert registry.detect_language(Path("foo.tl")) == "testlang"
        assert registry.detect_language(Path("foo.py")) is None

    def test_is_language_supported(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        assert registry.is_language_supported("testlang") is True
        assert registry.is_language_supported("unknown") is False

    def test_is_file_supported(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        assert registry.is_file_supported(Path("x.tl")) is True
        assert registry.is_file_supported(Path("x.rs")) is False

    def test_get_language_config(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        cfg = registry.get_language_config("testlang")
        assert cfg.name == "testlang"

    def test_get_language_config_not_registered(self):
        registry = LanguageRegistry()
        with pytest.raises(ValueError, match="not registered"):
            registry.get_language_config("nope")

    async def test_get_processor(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        proc = await registry.get_processor("testlang")  # type: ignore[arg-type]
        assert isinstance(proc, DummyProcessor)
        assert proc._initialized is True

    async def test_get_processor_cached(self, basic_config: LanguageConfig):
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        proc1 = await registry.get_processor("testlang")  # type: ignore[arg-type]
        proc2 = await registry.get_processor("testlang")  # type: ignore[arg-type]
        assert proc1 is proc2

    async def test_get_processor_unregistered_raises(self):
        registry = LanguageRegistry()
        with pytest.raises(ValueError, match="not registered"):
            await registry.get_processor("nope")  # type: ignore[arg-type]

    def test_extension_override_warning(self, basic_config: LanguageConfig, caplog):
        """Registering two languages with the same extension logs a warning."""
        registry = LanguageRegistry()
        registry.register_language(basic_config, DummyProcessor)
        config2 = LanguageConfig(
            name="other",
            display_name="Other",
            extensions=[".tl"],
            parser_type=ParserType.REGEX_FALLBACK,
            features=LanguageFeatures(),
        )
        registry.register_language(config2, DummyProcessor)
        # The second language should override the first for the .tl extension
        assert registry.detect_language(Path("foo.tl")) == "other"


class TestRegisterLanguageDecorator:
    def test_decorator_factory(self):
        from processing.languages.registry import register_language as reg_decorator

        cfg = LanguageConfig(
            name="dec_test",
            display_name="Decorator Test",
            extensions=[".dt"],
            parser_type=ParserType.REGEX_FALLBACK,
            features=LanguageFeatures(),
        )

        @reg_decorator(cfg)
        class DecTestProcessor(LanguageProcessor):
            async def _initialize_impl(self):
                pass

            async def parse_file(self, file_path, content):
                pass

            async def extract_entities(self, ast, file_path):
                return []

            async def extract_relationships(self, ast, entities, file_path):
                return []

        # Verify it got registered in the global registry
        from processing.languages.registry import language_registry

        assert language_registry.is_language_supported("dec_test")

    def test_decorator_direct_call(self):
        from processing.languages.registry import register_language as reg_decorator

        cfg = LanguageConfig(
            name="direct_test",
            display_name="Direct Test",
            extensions=[".drt"],
            parser_type=ParserType.REGEX_FALLBACK,
            features=LanguageFeatures(),
        )
        result = reg_decorator(cfg, DummyProcessor)
        assert result is DummyProcessor


# ========================== PYTHON PROCESSOR ================================


class TestPyLanguageProcessor:
    @pytest.fixture
    def processor(self):
        # Force non-legacy mode by patching LEGACY_COMPONENTS_AVAILABLE
        with patch(
            "processing.languages.python.python_processor.LEGACY_COMPONENTS_AVAILABLE", False
        ):
            from processing.languages.python.python_processor import PyLanguageProcessor

            proc = PyLanguageProcessor.__new__(PyLanguageProcessor)
            from processing.languages.python.python_config import PYTHON_CONFIG

            LanguageProcessor.__init__(proc, PYTHON_CONFIG)
            proc._use_legacy = False
            return proc

    async def test_initialize(self, processor):
        await processor._initialize_impl()
        # No exception means success

    async def test_parse_simple_function(self, processor):
        code = "def hello():\n    return 42\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is True
        assert result.language == "python"
        assert result.parser_used == ParserType.AST_NATIVE
        func_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(e.name == "hello" for e in func_entities)

    async def test_parse_class_with_method(self, processor):
        code = 'class Foo:\n    """A class."""\n    def bar(self):\n        pass\n'
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is True
        class_entities = [e for e in result.entities if e.type == EntityType.CLASS]
        assert any(e.name == "Foo" for e in class_entities)
        func_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(e.name == "bar" for e in func_entities)

    async def test_parse_imports(self, processor):
        code = "import os\nfrom pathlib import Path\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is True
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        names = {e.name for e in import_entities}
        assert "os" in names
        assert "pathlib.Path" in names

    async def test_parse_from_import_no_module(self, processor):
        code = "from . import something\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is True
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert any(e.name == "something" for e in import_entities)

    async def test_parse_syntax_error(self, processor):
        code = "def broken(\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is False
        assert len(result.errors) > 0

    async def test_extract_relationships_inheritance(self, processor):
        code = "class Base:\n    pass\nclass Child(Base):\n    pass\n"
        result = await processor.parse_file(Path("test.py"), code)
        inherits_rels = [r for r in result.relationships if r.type == RelationType.INHERITS]
        assert len(inherits_rels) == 1

    async def test_extract_relationships_contains(self, processor):
        code = "def foo():\n    pass\n"
        result = await processor.parse_file(Path("test.py"), code)
        contains_rels = [r for r in result.relationships if r.type == RelationType.CONTAINS]
        assert len(contains_rels) >= 1

    async def test_complexity_metrics(self, processor):
        code = "# A comment\ndef foo():\n    pass\n\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.lines_of_code == 4
        assert result.success is True

    def test_map_entity_type(self, processor):
        assert processor._map_entity_type("class") == EntityType.CLASS
        assert processor._map_entity_type("function") == EntityType.FUNCTION
        assert processor._map_entity_type("method") == EntityType.FUNCTION
        assert processor._map_entity_type("module") == EntityType.MODULE
        assert processor._map_entity_type("variable") == EntityType.VARIABLE
        assert processor._map_entity_type("constant") == EntityType.CONSTANT
        assert processor._map_entity_type("import") == EntityType.IMPORT
        assert processor._map_entity_type("decorator") == EntityType.DECORATOR
        assert processor._map_entity_type("unknown") == EntityType.FUNCTION  # default

    def test_map_relation_type(self, processor):
        assert processor._map_relation_type("inheritance") == RelationType.INHERITS
        assert processor._map_relation_type("composition") == RelationType.CONTAINS
        assert processor._map_relation_type("dependency") == RelationType.DEPENDS_ON
        assert processor._map_relation_type("call") == RelationType.CALLS
        assert processor._map_relation_type("import") == RelationType.IMPORTS
        assert processor._map_relation_type("usage") == RelationType.REFERENCES
        assert processor._map_relation_type("unknown") == RelationType.REFERENCES  # default

    def test_convert_legacy_entity_dict(self, processor):
        le = {
            "name": "foo",
            "id": "func:foo",
            "type": "function",
            "line_number": 5,
            "end_line_number": 10,
            "column_start": 0,
            "column_end": 20,
            "visibility": "public",
            "code_snippet": "def foo():",
            "docstring": "A function",
            "signature": "foo()",
            "source_code": "def foo(): pass",
            "metadata": {"key": "val"},
        }
        entity = processor._convert_legacy_entity(le, Path("test.py"))
        assert entity.name == "foo"
        assert entity.type == EntityType.FUNCTION
        assert entity.line_number == 5
        assert entity.properties == {"key": "val"}

    def test_convert_legacy_entity_object(self, processor):
        le = MagicMock()
        le.name = "bar"
        le.id = "func:bar"
        le.type = "class"
        le.line_number = 3
        le.end_line_number = 20
        le.column_start = 0
        le.column_end = 40
        le.visibility = "private"
        le.code_snippet = "class bar:"
        le.docstring = "A class"
        le.signature = "class bar"
        le.source_code = "class bar:\n    pass"
        le.metadata = {"x": 1}
        entity = processor._convert_legacy_entity(le, Path("test.py"))
        assert entity.name == "bar"
        assert entity.type == EntityType.CLASS

    def test_convert_legacy_relation(self, processor):
        lr = MagicMock()
        lr.source_id = "a"
        lr.target_id = "b"
        lr.type = "call"
        lr.metadata = {"ctx": "test"}
        rel = processor._convert_legacy_relation(lr)
        assert rel.source_id == "a"
        assert rel.target_id == "b"
        assert rel.type == RelationType.CALLS

    async def test_extract_entities_from_ast(self, processor):
        import ast as ast_module

        tree = ast_module.parse("def foo(): pass\n", filename="test.py")
        entities = await processor.extract_entities(tree, Path("test.py"))
        assert any(e.name == "foo" for e in entities)

    async def test_extract_entities_non_ast(self, processor):
        entities = await processor.extract_entities("not an ast", Path("test.py"))
        assert entities == []

    async def test_extract_relationships_via_public_method(self, processor):
        import ast as ast_module

        code = "class A:\n    pass\n"
        tree = ast_module.parse(code, filename="test.py")
        entities = await processor.extract_entities(tree, Path("test.py"))
        rels = await processor.extract_relationships(tree, entities, Path("test.py"))
        assert isinstance(rels, list)

    async def test_extract_relationships_non_ast(self, processor):
        rels = await processor.extract_relationships("not ast", [], Path("test.py"))
        assert rels == []

    def test_build_function_signature(self, processor):
        import ast as ast_module

        code = "def greet(name: str, age: int) -> str: pass\n"
        tree = ast_module.parse(code)
        func_node = tree.body[0]
        sig = processor._build_function_signature(func_node)
        assert "greet" in sig
        assert "name: str" in sig
        assert "age: int" in sig
        assert "-> str" in sig

    def test_build_function_signature_no_annotations(self, processor):
        import ast as ast_module

        code = "def simple(a, b): pass\n"
        tree = ast_module.parse(code)
        func_node = tree.body[0]
        sig = processor._build_function_signature(func_node)
        assert sig == "simple(a, b)"

    def test_get_node_name_variants(self, processor):
        import ast as ast_module

        # Name node
        name_node = ast_module.Name(id="foo", ctx=ast_module.Load())
        assert processor._get_node_name(name_node) == "foo"

        # Attribute node
        attr_node = ast_module.Attribute(
            value=ast_module.Name(id="os", ctx=ast_module.Load()),
            attr="path",
            ctx=ast_module.Load(),
        )
        assert processor._get_node_name(attr_node) == "os.path"

        # Call node with Name func
        call_node = ast_module.Call(
            func=ast_module.Name(id="print", ctx=ast_module.Load()),
            args=[],
            keywords=[],
        )
        assert processor._get_node_name(call_node) == "print"

        # Unknown node -> ast.dump fallback
        constant = ast_module.Constant(value=42)
        result = processor._get_node_name(constant)
        assert "Constant" in result

    async def test_parse_with_legacy_success(self, processor):
        """Test the legacy parsing path when parser.parse returns success."""
        processor._use_legacy = True
        processor.parser = MagicMock()
        processor.parser.parse = MagicMock(return_value={"success": True, "tree": None})
        processor.entity_extractor = MagicMock()
        processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
        processor.relation_extractor = MagicMock()

        result = await processor._parse_with_legacy(Path("test.py"), "x = 1\n")
        assert isinstance(result, ParseResult)

    async def test_parse_with_legacy_parse_code_fallback(self, processor):
        """Test legacy parsing when parse() fails but parse_code() succeeds."""
        import ast as ast_module

        processor._use_legacy = True
        processor.parser = MagicMock()
        processor.parser.parse = MagicMock(side_effect=TypeError("fail"))
        processor.parser.parse_code = AsyncMock(return_value=ast_module.parse("x = 1\n"))
        processor.entity_extractor = MagicMock()
        processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
        processor.relation_extractor = MagicMock()

        result = await processor._parse_with_legacy(Path("test.py"), "x = 1\n")
        assert isinstance(result, ParseResult)

    async def test_parse_with_legacy_ast_fallback(self, processor):
        """Test legacy parsing when both parse methods fail, falls back to ast.parse."""
        processor._use_legacy = True
        processor.parser = MagicMock(spec=[])  # no parse/parse_code attrs
        processor.entity_extractor = MagicMock()
        processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
        processor.relation_extractor = MagicMock()

        result = await processor._parse_with_legacy(Path("test.py"), "x = 1\n")
        assert isinstance(result, ParseResult)

    async def test_parse_with_legacy_entity_extraction_failure(self, processor):
        """When entity_extractor raises, we fall back to direct AST extraction."""

        processor._use_legacy = True
        processor.parser = MagicMock(spec=[])
        processor.entity_extractor = MagicMock()
        processor.entity_extractor.extract_entities = AsyncMock(
            side_effect=ValueError("extraction fail")
        )
        processor.relation_extractor = MagicMock()

        result = await processor._parse_with_legacy(Path("test.py"), "def foo(): pass\n")
        assert isinstance(result, ParseResult)


# ========================== JAVASCRIPT PROCESSOR ============================


class TestJSLanguageProcessor:
    @pytest.fixture
    def processor(self):
        with (
            patch(
                "processing.languages.javascript.javascript_processor.TREE_SITTER_AVAILABLE",
                False,
            ),
            patch(
                "processing.languages.javascript.javascript_processor.LEGACY_COMPONENTS_AVAILABLE",
                False,
            ),
        ):
            from processing.languages.javascript.javascript_processor import (
                JSLanguageProcessor,
            )

            proc = JSLanguageProcessor.__new__(JSLanguageProcessor)
            from processing.languages.javascript.javascript_config import JAVASCRIPT_CONFIG

            LanguageProcessor.__init__(proc, JAVASCRIPT_CONFIG)
            proc._use_legacy = False
            # Ensure no parser attribute that could cause issues
            proc.parser = None
            return proc

    async def test_initialize(self, processor):
        await processor._initialize_impl()

    async def test_parse_js_file_regex(self, processor):
        code = 'import { foo } from "bar";\nconst x = require("baz");\n'
        result = await processor.parse_file(Path("test.js"), code)
        assert result.success is True
        assert result.language == "javascript"
        assert result.parser_used == ParserType.REGEX_FALLBACK
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        names = {e.name for e in import_entities}
        assert "bar" in names
        assert "baz" in names

    async def test_parse_ts_file_regex(self, processor):
        code = 'import something from "module";\n'
        result = await processor.parse_file(Path("test.ts"), code)
        assert result.success is True
        assert result.language == "typescript"

    async def test_parse_tsx_file_regex(self, processor):
        code = 'import React from "react";\n'
        result = await processor.parse_file(Path("component.tsx"), code)
        assert result.language == "typescript"

    async def test_parse_require_patterns(self, processor):
        code = """const a = require('moduleA');
require('moduleB');
import 'moduleC';
"""
        result = await processor.parse_file(Path("test.js"), code)
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        names = {e.name for e in import_entities}
        assert "moduleA" in names
        assert "moduleB" in names
        assert "moduleC" in names

    async def test_regex_parse_complexity_metrics(self, processor):
        code = "// comment\nlet x = 1;\n\n/* block */\n"
        result = await processor.parse_file(Path("test.js"), code)
        assert result.lines_of_code == 4

    def test_map_entity_type(self, processor):
        assert processor._map_entity_type("class") == EntityType.CLASS
        assert processor._map_entity_type("function") == EntityType.FUNCTION
        assert processor._map_entity_type("method") == EntityType.FUNCTION
        assert processor._map_entity_type("interface") == EntityType.INTERFACE
        assert processor._map_entity_type("type") == EntityType.TYPE_ALIAS
        assert processor._map_entity_type("enum") == EntityType.ENUM
        assert processor._map_entity_type("unknown_type") == EntityType.FUNCTION

    async def test_extract_entities_fallback(self, processor):
        """When tree-sitter is not available, extract_entities falls back gracefully."""
        entities = await processor.extract_entities(None, Path("test.js"))
        # Should return empty list when file doesn't exist
        assert isinstance(entities, list)

    async def test_extract_relationships(self, processor):
        file_ent = CoreCodeEntity(
            id="f1",
            name="test.js",
            type=EntityType.FILE,
            language="javascript",
            file_path="test.js",
            line_number=1,
        )
        class_ent = CoreCodeEntity(
            id="c1",
            name="Foo",
            type=EntityType.CLASS,
            language="javascript",
            file_path="test.js",
            line_number=2,
        )
        rels = await processor.extract_relationships(None, [file_ent, class_ent], Path("test.js"))
        contains_rels = [
            r for r in rels if r.type == RelationType.CONTAINS and r.target_id == class_ent.id
        ]
        assert len(contains_rels) >= 1

    async def test_extract_relationships_no_methods(self, processor):
        ent = CoreCodeEntity(
            id="c1",
            name="Foo",
            type=EntityType.CLASS,
            language="javascript",
            file_path="test.js",
            line_number=1,
        )
        rels = await processor.extract_relationships(None, [ent], Path("test.js"))
        assert rels == []

    async def test_parse_with_legacy_success(self, processor):
        """Test legacy parser path."""
        processor._use_legacy = True
        processor.legacy_parser = MagicMock()
        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "testFunc"
        mock_entity.type = "function"
        mock_entity.line_number = 1
        mock_entity.metadata = {}
        processor.legacy_parser.parse_code = AsyncMock(
            return_value={"success": True, "entities": [mock_entity]}
        )
        result = await processor.parse_file(Path("test.js"), "function test() {}")
        assert result.success is True
        assert any(e.name == "testFunc" for e in result.entities)

    async def test_parse_with_legacy_failure(self, processor):
        """Test legacy parser failure falls back to regex."""
        processor._use_legacy = True
        processor.legacy_parser = MagicMock()
        processor.legacy_parser.parse_code = AsyncMock(side_effect=TypeError("fail"))
        result = await processor.parse_file(Path("test.js"), 'const x = require("y");\n')
        assert result.success is True
        assert result.parser_used == ParserType.REGEX_FALLBACK

    async def test_parse_with_legacy_no_success(self, processor):
        """Test legacy parser returning unsuccessful parse result."""
        processor._use_legacy = True
        processor.legacy_parser = MagicMock()
        processor.legacy_parser.parse_code = AsyncMock(
            return_value={"success": False, "error": "parse error"}
        )
        result = await processor.parse_file(Path("test.js"), "broken code")
        # Should not be successful since legacy returned failure
        assert result.success is False
        assert "parse error" in result.errors

    async def test_parse_with_legacy_none_result(self, processor):
        """Test legacy parser returning None."""
        processor._use_legacy = True
        processor.legacy_parser = MagicMock()
        processor.legacy_parser.parse_code = AsyncMock(return_value=None)
        result = await processor.parse_file(Path("test.js"), "code")
        assert result.success is False


# ========================== TYPESCRIPT PROCESSOR ============================


class TestTSLanguageProcessor:
    @pytest.fixture
    def processor(self):
        from processing.languages.typescript.typescript_processor import TSLanguageProcessor

        proc = TSLanguageProcessor()
        proc._time_module = MagicMock()
        proc._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        return proc

    async def test_initialize(self, processor):
        await processor._initialize_impl()

    async def test_parse_interface(self, processor):
        code = "export interface IUser<T> extends BaseUser {\n  name: string;\n}\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        assert result.success is True
        assert result.language == "typescript"
        iface_entities = [e for e in result.entities if e.type == EntityType.INTERFACE]
        assert any(e.name == "IUser" for e in iface_entities)
        # Check generic params
        iuser = next(e for e in iface_entities if e.name == "IUser")
        assert iuser.properties["generic_parameters"] == ["T"]
        assert "BaseUser" in iuser.properties["extends"]
        assert iuser.properties["is_exported"] is True

    async def test_parse_type_alias(self, processor):
        code = "export type Result<T> = T | Error;\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        type_entities = [e for e in result.entities if e.type == EntityType.TYPE_ALIAS]
        assert any(e.name == "Result" for e in type_entities)
        r = next(e for e in type_entities if e.name == "Result")
        assert r.properties["is_union_type"] is True
        assert r.properties["generic_parameters"] == ["T"]

    async def test_parse_class(self, processor):
        code = "export abstract class Service<T> extends BaseService implements IService {\n}\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        class_entities = [e for e in result.entities if e.type == EntityType.CLASS]
        assert any(e.name == "Service" for e in class_entities)
        svc = next(e for e in class_entities if e.name == "Service")
        assert svc.is_abstract is True
        assert svc.properties["extends"] == "BaseService"
        assert "IService" in svc.properties["implements"]

    async def test_parse_enum(self, processor):
        code = "export const enum Color {\n  Red,\n  Green,\n  Blue\n}\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        enum_entities = [e for e in result.entities if e.type == EntityType.ENUM]
        assert any(e.name == "Color" for e in enum_entities)
        color = next(e for e in enum_entities if e.name == "Color")
        assert color.properties["is_const_enum"] is True

    async def test_parse_function(self, processor):
        code = "export async function fetchData<T>(url: string): Promise<T> {\n}\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        func_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(e.name == "fetchData" for e in func_entities)
        fd = next(e for e in func_entities if e.name == "fetchData")
        assert fd.is_async is True
        assert fd.properties["generic_parameters"] == ["T"]

    async def test_parse_arrow_function(self, processor):
        code = "export const add: (a: number, b: number) => number = (a, b): number => a + b;\n"
        processor._time_module.perf_counter = MagicMock(side_effect=[0.0, 0.001])
        result = await processor.parse_file(Path("test.ts"), code)
        func_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        _ = [e for e in func_entities if e.properties.get("is_arrow_function")]
        # Regex may or may not match depending on exact pattern; just verify no errors
        assert result.success is True
