"""Coverage boost tests for JavaProcessor — targets uncovered tree-sitter code paths.

Focuses on:
- parse_file dispatch (tree-sitter available vs not)
- _regex_parse edge cases (interfaces, enums, generics, constructors)
- extract_entities/extract_relationships with None AST
- TREE_SITTER_AVAILABLE flag influence
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import CoreCodeEntity, EntityType
from processing.languages.base import ParseResult, ParserType
from processing.languages.java.java_processor import JavaProcessor


@pytest.fixture
def processor():
    return JavaProcessor()


# ── parse_file dispatch ────────────────────────────────────────────────


class TestJavaParseFileDispatch:
    """Test parse_file routing between tree-sitter and regex fallback."""

    @pytest.mark.asyncio
    async def test_parse_file_uses_regex_when_no_parser(self, processor):
        """When self.parser is None, parse_file falls back to regex."""
        processor.parser = None
        fp = Path("Test.java")
        code = "package com.example; public class Test { }"
        result = await processor.parse_file(fp, code)
        assert result.success is True
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_parse_file_handles_parent_result_without_success_attr(self, processor):
        """When tree-sitter returns object lacking `success`, fallback fires."""
        processor.parser = MagicMock()
        # Simulate super().parse_file returning something without 'success'
        mock_result = MagicMock(spec=[])  # no attributes at all
        with patch(
            "processing.parsers.tree_sitter_base.BaseTreeSitterProcessor.parse_file",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await processor.parse_file(Path("T.java"), "class T {}")
            # Should return a ParseResult with success=False since the parent result
            # has no 'success' attribute
            assert isinstance(result, ParseResult)
            assert result.success is False
            assert result.language == "java"
            assert "Invalid result from parent parser" in result.errors


# ── _regex_parse edge cases ────────────────────────────────────────────


JAVA_MULTI_INTERFACE = """\
package com.example;

import java.io.Serializable;

public interface Identifiable {
    String getId();
}

public interface Nameable {
    String getName();
}
"""

JAVA_GENERICS_COMPLEX = """\
package com.example.generics;

import java.util.Optional;
import java.util.Map;

public class GenericService<T extends Comparable<T>> {
    public Optional<T> findFirst(Map<String, T> items) {
        return Optional.empty();
    }

    protected <R> R transform(T input) {
        return null;
    }
}
"""

JAVA_CONSTRUCTOR_ONLY = """\
package com.example;

public class Builder {
    public Builder() {
        // no-op
    }

    public Builder(String name) {
        // name constructor
    }
}
"""


class TestJavaRegexEdgeCases:
    @pytest.mark.asyncio
    async def test_multiple_interfaces(self, processor):
        result = await processor._regex_parse(Path("Interfaces.java"), JAVA_MULTI_INTERFACE)
        assert result.success is True
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert len(packages) == 1
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 1
        assert imports[0].name == "java.io.Serializable"

    @pytest.mark.asyncio
    async def test_generics_complex(self, processor):
        result = await processor._regex_parse(Path("GenericService.java"), JAVA_GENERICS_COMPLEX)
        assert result.success is True
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        assert "findFirst" in method_names
        assert "transform" in method_names

    @pytest.mark.asyncio
    async def test_constructor_only_class(self, processor):
        result = await processor._regex_parse(Path("Builder.java"), JAVA_CONSTRUCTOR_ONLY)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Builder"

    @pytest.mark.asyncio
    async def test_non_static_import(self, processor):
        code = "import java.util.ArrayList;\npublic class Foo {}"
        result = await processor._regex_parse(Path("Foo.java"), code)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 1
        assert imports[0].properties["is_static"] is False

    @pytest.mark.asyncio
    async def test_method_with_multiple_params(self, processor):
        code = """\
public class Svc {
    public void process(String a, int b, boolean c) {
    }
}
"""
        result = await processor._regex_parse(Path("Svc.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(m.name == "process" for m in methods)

    @pytest.mark.asyncio
    async def test_static_method(self, processor):
        code = """\
public class Util {
    public static int max(int a, int b) {
        return Math.max(a, b);
    }
}
"""
        result = await processor._regex_parse(Path("Util.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(m.name == "max" for m in methods)

    @pytest.mark.asyncio
    async def test_method_with_throws_multiple(self, processor):
        code = """\
public class Ex {
    public void risky() throws IOException, SQLException {
    }
}
"""
        result = await processor._regex_parse(Path("Ex.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(m.name == "risky" for m in methods)

    @pytest.mark.asyncio
    async def test_relationships_empty_for_regex(self, processor):
        code = "public class Foo {}"
        result = await processor._regex_parse(Path("Foo.java"), code)
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_all_entity_ids_contain_file_path(self, processor):
        code = """\
package org.test;
import java.util.List;
public class App {
    public void run() {}
}
"""
        result = await processor._regex_parse(Path("App.java"), code)
        for entity in result.entities:
            if entity.type != EntityType.PACKAGE:
                assert "App.java" in entity.id


# ── extract_entities / extract_relationships fallback paths ────────────


class TestJavaFallbackMethods:
    @pytest.mark.asyncio
    async def test_extract_entities_with_none_ast(self, processor):
        entities = await processor.extract_entities(None, Path("test.java"))
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_relationships_with_none_ast(self, processor):
        rels = await processor.extract_relationships(None, [], Path("test.java"))
        assert rels == []

    @pytest.mark.asyncio
    async def test_extract_entities_with_empty_entities(self, processor):
        entities = await processor.extract_entities(None, Path("empty.java"))
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_relationships_with_entities_but_no_ast(self, processor):
        entity = CoreCodeEntity(
            id="class:Foo:test.java",
            name="Foo",
            type=EntityType.CLASS,
            language="java",
            file_path="test.java",
            line_number=1,
        )
        rels = await processor.extract_relationships(None, [entity], Path("test.java"))
        assert rels == []


# ── Config and initialization ──────────────────────────────────────────


class TestJavaProcessorConfig:
    def test_default_config_has_java_name(self):
        p = JavaProcessor()
        assert p.config.name == "java"

    def test_custom_config(self):
        from processing.languages.java.java_config import JAVA_CONFIG

        p = JavaProcessor(config=JAVA_CONFIG)
        assert p.config is JAVA_CONFIG

    def test_field_node_types(self):
        p = JavaProcessor()
        assert "field_declaration" in p.FIELD_NODE_TYPES
        assert "variable_declarator" in p.FIELD_NODE_TYPES

    def test_import_node_types(self):
        p = JavaProcessor()
        assert "import_declaration" in p.IMPORT_NODE_TYPES

    @pytest.mark.asyncio
    async def test_initialize_impl_when_no_tree_sitter(self):
        p = JavaProcessor()
        # Should not raise even without tree-sitter
        await p._initialize_impl()


# ── Additional coverage tests ─────────────────────────────────────────


class TestJavaRegexParseAdditional:
    """Additional tests for Java regex parsing to cover more lines."""

    @pytest.fixture
    def processor(self):
        return JavaProcessor()

    @pytest.mark.asyncio
    async def test_empty_code(self, processor):
        result = await processor._regex_parse(Path("Empty.java"), "")
        assert result.success is True
        assert result.entities == []
        assert result.lines_of_code == 0

    @pytest.mark.asyncio
    async def test_package_line_number(self, processor):
        code = "// comment\npackage org.example;\npublic class Foo {}"
        result = await processor._regex_parse(Path("Foo.java"), code)
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert len(packages) == 1
        assert packages[0].name == "org.example"
        assert packages[0].line_number == 2
        assert packages[0].id == "package:org.example"

    @pytest.mark.asyncio
    async def test_import_line_numbers(self, processor):
        code = "// header\nimport java.util.List;\nimport java.util.Map;"
        result = await processor._regex_parse(Path("Foo.java"), code)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 2
        list_import = next(i for i in imports if i.name == "java.util.List")
        map_import = next(i for i in imports if i.name == "java.util.Map")
        assert list_import.line_number == 2
        assert map_import.line_number == 3

    @pytest.mark.asyncio
    async def test_class_line_number(self, processor):
        code = "package org;\n\npublic class MyClass {\n}"
        result = await processor._regex_parse(Path("MyClass.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].line_number == 3

    @pytest.mark.asyncio
    async def test_method_line_number(self, processor):
        code = "public class Svc {\n    public void run() {\n    }\n}"
        result = await processor._regex_parse(Path("Svc.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        run = next(m for m in methods if m.name == "run")
        assert run.line_number == 2

    @pytest.mark.asyncio
    async def test_entity_language_is_java(self, processor):
        code = "package org;\nimport java.util.List;\npublic class A { public void m() {} }"
        result = await processor._regex_parse(Path("A.java"), code)
        for entity in result.entities:
            assert entity.language == "java"

    @pytest.mark.asyncio
    async def test_entity_file_path_set(self, processor):
        code = "public class B { public void n() {} }"
        result = await processor._regex_parse(Path("B.java"), code)
        for entity in result.entities:
            assert entity.file_path == "B.java"

    @pytest.mark.asyncio
    async def test_static_import_property_true(self, processor):
        code = "import static java.lang.Math.PI;\npublic class X {}"
        result = await processor._regex_parse(Path("X.java"), code)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 1
        assert imports[0].name == "java.lang.Math.PI"
        assert imports[0].properties["is_static"] is True
        assert imports[0].properties["is_wildcard"] is False

    @pytest.mark.asyncio
    async def test_wildcard_import(self, processor):
        code = "import java.util.*;\npublic class W {}"
        result = await processor._regex_parse(Path("W.java"), code)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 1
        assert imports[0].properties["is_wildcard"] is True

    @pytest.mark.asyncio
    async def test_parser_used_is_regex(self, processor):
        result = await processor._regex_parse(Path("T.java"), "class T {}")
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_lines_of_code_count(self, processor):
        code = "line1\nline2\nline3\n"
        result = await processor._regex_parse(Path("T.java"), code)
        assert result.lines_of_code == 3

    @pytest.mark.asyncio
    async def test_abstract_class_detected(self, processor):
        code = "public abstract class Base { }"
        result = await processor._regex_parse(Path("Base.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Base"

    @pytest.mark.asyncio
    async def test_final_class_detected(self, processor):
        code = "public final class Immutable { }"
        result = await processor._regex_parse(Path("Immutable.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Immutable"

    @pytest.mark.asyncio
    async def test_method_keyword_skipping(self, processor):
        """Verify that i, for, while are skipped as method names."""
        code = """\
public class Looper {
    public void loop() {
        for (int i = 0; i < 10; i++) {}
        while (true) { break; }
    }
}
"""
        result = await processor._regex_parse(Path("Looper.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        assert "i" not in method_names
        assert "for" not in method_names
        assert "while" not in method_names
        assert "loop" in method_names

    @pytest.mark.asyncio
    async def test_multiple_methods(self, processor):
        code = """\
public class Multi {
    public String getName() { return ""; }
    private int getAge() { return 0; }
    protected void doWork() {}
}
"""
        result = await processor._regex_parse(Path("Multi.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        names = {m.name for m in methods}
        assert "getName" in names
        assert "getAge" in names
        assert "doWork" in names
        assert len(methods) == 3
