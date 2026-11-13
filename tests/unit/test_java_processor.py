"""Tests for Java language processor regex-based parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType
from processing.languages.java.java_processor import JavaProcessor


@pytest.fixture
def processor():
    return JavaProcessor()


SAMPLE_JAVA_CODE = """\
package com.example.myapp;

import java.util.List;
import java.util.Map;
import static java.lang.Math.abs;

public abstract class UserService {

    private final String name;

    public void processUsers(List<String> users) throws IllegalArgumentException {
        for (String user : users) {
            validateUser(user);
        }
    }

    private boolean validateUser(String user) {
        return user != null && !user.isEmpty();
    }

    public static int calculate(int a, int b) {
        return a + b;
    }
}
"""


@pytest.mark.asyncio
async def test_regex_parse_success(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    assert result.success is True
    assert result.language == "java"
    assert result.file_path == "UserService.java"
    assert result.lines_of_code == len(SAMPLE_JAVA_CODE.splitlines())


@pytest.mark.asyncio
async def test_regex_parse_extracts_package(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    packages = [e for e in result.entities if e.type.value == "package"]
    assert len(packages) == 1
    assert packages[0].name == "com.example.myapp"
    assert packages[0].language == "java"


@pytest.mark.asyncio
async def test_regex_parse_extracts_imports(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    imports = [e for e in result.entities if e.type.value == "import"]
    assert len(imports) == 3
    import_names = {e.name for e in imports}
    assert "java.util.List" in import_names
    assert "java.util.Map" in import_names
    assert "java.lang.Math.abs" in import_names


@pytest.mark.asyncio
async def test_regex_parse_detects_static_import(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    imports = [e for e in result.entities if e.type.value == "import"]
    static_imports = [e for e in imports if e.properties and e.properties.get("is_static")]
    assert len(static_imports) == 1
    assert static_imports[0].name == "java.lang.Math.abs"


@pytest.mark.asyncio
async def test_regex_parse_extracts_class(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    classes = [e for e in result.entities if e.type.value == "class"]
    assert len(classes) == 1
    assert classes[0].name == "UserService"
    assert classes[0].language == "java"


@pytest.mark.asyncio
async def test_regex_parse_extracts_methods(processor):
    fp = Path("UserService.java")
    result = await processor._regex_parse(fp, SAMPLE_JAVA_CODE)
    methods = [e for e in result.entities if e.type.value == "function"]
    method_names = {e.name for e in methods}
    assert "processUsers" in method_names
    assert "validateUser" in method_names
    assert "calculate" in method_names


@pytest.mark.asyncio
async def test_regex_parse_skips_keywords_in_methods(processor):
    code = "public class Foo {\n  for (int i = 0; i < 10; i++) {\n  }\n}\n"
    fp = Path("Foo.java")
    result = await processor._regex_parse(fp, code)
    methods = [e for e in result.entities if e.type.value == "function"]
    method_names = {e.name for e in methods}
    assert "for" not in method_names
    assert "while" not in method_names


@pytest.mark.asyncio
async def test_regex_parse_empty_code(processor):
    fp = Path("Empty.java")
    result = await processor._regex_parse(fp, "")
    assert result.success is True
    assert result.entities == []
    assert result.lines_of_code == 0


@pytest.mark.asyncio
async def test_regex_parse_no_package(processor):
    code = "public class Standalone { public void run() {} }"
    fp = Path("Standalone.java")
    result = await processor._regex_parse(fp, code)
    packages = [e for e in result.entities if e.type.value == "package"]
    assert len(packages) == 0


@pytest.mark.asyncio
async def test_regex_parse_wildcard_import(processor):
    code = "import java.util.*;\npublic class Foo {}\n"
    fp = Path("Foo.java")
    result = await processor._regex_parse(fp, code)
    imports = [e for e in result.entities if e.type.value == "import"]
    assert len(imports) == 1
    assert imports[0].properties["is_wildcard"] is True


@pytest.mark.asyncio
async def test_parse_file_without_tree_sitter(processor):
    """parse_file falls back to regex when tree-sitter is not available."""
    fp = Path("Test.java")
    result = await processor.parse_file(fp, SAMPLE_JAVA_CODE)
    assert result.success is True
    assert len(result.entities) > 0


@pytest.mark.asyncio
async def test_initialize_impl(processor):
    """_initialize_impl should not raise."""
    await processor._initialize_impl()


@pytest.mark.asyncio
async def test_extract_entities_fallback(processor):
    """extract_entities returns empty list for fallback."""
    entities = await processor.extract_entities(None, Path("test.java"))
    assert entities == []


@pytest.mark.asyncio
async def test_extract_relationships_fallback(processor):
    """extract_relationships returns empty list for fallback."""
    rels = await processor.extract_relationships(None, [], Path("test.java"))
    assert rels == []


def test_class_node_types(processor):
    assert "class_declaration" in processor.CLASS_NODE_TYPES
    assert "interface_declaration" in processor.CLASS_NODE_TYPES
    assert "enum_declaration" in processor.CLASS_NODE_TYPES


def test_method_node_types(processor):
    assert "method_declaration" in processor.METHOD_NODE_TYPES
    assert "constructor_declaration" in processor.METHOD_NODE_TYPES


# ---------------------------------------------------------------------------
# Tests merged from test_round4_java_processor.py
# ---------------------------------------------------------------------------

SAMPLE_JAVA_R4 = """\
package com.example.service;

import java.util.List;
import java.util.Map;
import static java.lang.Math.PI;

public abstract class UserService {
    private final String name;
    public static int counter;

    public UserService(String name) {
        this.name = name;
    }

    public List<User> getUsers() {
        return List.of();
    }

    private void processData(Map<String, Object> data) throws Exception {
        // do stuff
    }

    protected static boolean isValid(String input) {
        return input != null;
    }
}
"""

JAVA_INTERFACE_R4 = """\
package com.example.api;

import java.util.concurrent.CompletableFuture;

public interface DataRepository {
    CompletableFuture<Void> save(Object entity);
    Object findById(String id);
}
"""

JAVA_ENUM_R4 = """\
package com.example.model;

public enum Status {
    ACTIVE,
    INACTIVE,
    PENDING
}
"""


class TestJavaRegexParse:
    """Test the _regex_parse fallback path which covers the bulk of uncovered lines."""

    @pytest.mark.asyncio
    async def test_regex_parse_extracts_package(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        assert result.success is True
        assert result.language == "java"
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert len(packages) == 1
        assert packages[0].name == "com.example.service"
        assert packages[0].id == "package:com.example.service"

    @pytest.mark.asyncio
    async def test_regex_parse_extracts_imports(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 3
        import_names = {e.name for e in imports}
        assert "java.util.List" in import_names
        assert "java.util.Map" in import_names
        assert "java.lang.Math.PI" in import_names

    @pytest.mark.asyncio
    async def test_regex_parse_static_import(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        static_import = [e for e in imports if e.name == "java.lang.Math.PI"][0]
        assert static_import.properties["is_static"] is True
        assert static_import.properties["is_wildcard"] is False

    @pytest.mark.asyncio
    async def test_regex_parse_extracts_class(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"

    @pytest.mark.asyncio
    async def test_regex_parse_extracts_methods(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {e.name for e in methods}
        assert "getUsers" in method_names
        assert "processData" in method_names
        assert "isValid" in method_names
        assert "class" not in method_names
        assert "interface" not in method_names

    @pytest.mark.asyncio
    async def test_regex_parse_skips_keywords(self, processor):
        code = """\
public class Test {
    public void doSomething() {
        for (int i = 0; i < 10; i++) {
            while (true) { break; }
        }
    }
}
"""
        result = await processor._regex_parse(Path("Test.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {e.name for e in methods}
        assert "i" not in method_names
        assert "for" not in method_names
        assert "while" not in method_names

    @pytest.mark.asyncio
    async def test_regex_parse_line_numbers(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert packages[0].line_number == 1

    @pytest.mark.asyncio
    async def test_regex_parse_returns_correct_parser_type(self, processor):
        from processing.languages.base import ParserType

        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_regex_parse_lines_of_code(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        assert result.lines_of_code == len(SAMPLE_JAVA_R4.splitlines())

    @pytest.mark.asyncio
    async def test_regex_parse_methods_with_throws(self, processor):
        code = """\
public class Ex {
    public void risky() throws IOException {
        // body
    }
}
"""
        result = await processor._regex_parse(Path("Ex.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(m.name == "risky" for m in methods)

    @pytest.mark.asyncio
    async def test_regex_parse_methods_with_generics(self, processor):
        code = """\
public class Ex {
    public List<String> getNames() {
        return null;
    }
}
"""
        result = await processor._regex_parse(Path("Ex.java"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert any(m.name == "getNames" for m in methods)

    @pytest.mark.asyncio
    async def test_regex_parse_interface(self, processor):
        result = await processor._regex_parse(Path("Repo.java"), JAVA_INTERFACE_R4)
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert packages[0].name == "com.example.api"
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        assert len(imports) == 1

    @pytest.mark.asyncio
    async def test_regex_parse_enum(self, processor):
        result = await processor._regex_parse(Path("Status.java"), JAVA_ENUM_R4)
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert packages[0].name == "com.example.model"

    @pytest.mark.asyncio
    async def test_entity_ids_unique(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        ids = [e.id for e in result.entities]
        assert len(ids) == len(set(ids)), "Entity IDs should be unique"

    @pytest.mark.asyncio
    async def test_multiple_classes(self, processor):
        code = """\
class Foo { }
class Bar { }
class Baz { }
"""
        result = await processor._regex_parse(Path("Multi.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 3

    @pytest.mark.asyncio
    async def test_private_class(self, processor):
        code = "private class InnerHelper { }"
        result = await processor._regex_parse(Path("Inner.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "InnerHelper"

    @pytest.mark.asyncio
    async def test_final_class(self, processor):
        code = "public final class Config { }"
        result = await processor._regex_parse(Path("Config.java"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Config"

    @pytest.mark.asyncio
    async def test_import_entity_file_path(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        for imp in imports:
            assert "Test.java" in imp.id
            assert imp.file_path == "Test.java"

    @pytest.mark.asyncio
    async def test_class_entity_attributes(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        cls = classes[0]
        assert cls.language == "java"
        assert cls.file_path == "Test.java"
        assert cls.line_number > 0

    @pytest.mark.asyncio
    async def test_method_entity_attributes(self, processor):
        result = await processor._regex_parse(Path("Test.java"), SAMPLE_JAVA_R4)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        for m in methods:
            assert m.language == "java"
            assert m.file_path == "Test.java"
            assert m.line_number > 0


class TestJavaProcessorInit:
    def test_constructor_default_config(self):
        p = JavaProcessor()
        assert p.config is not None

    def test_class_node_types(self):
        p = JavaProcessor()
        assert "class_declaration" in p.CLASS_NODE_TYPES
        assert "interface_declaration" in p.CLASS_NODE_TYPES

    def test_method_node_types(self):
        p = JavaProcessor()
        assert "method_declaration" in p.METHOD_NODE_TYPES
        assert "constructor_declaration" in p.METHOD_NODE_TYPES

    @pytest.mark.asyncio
    async def test_initialize_impl(self):
        p = JavaProcessor()
        await p._initialize_impl()


# ---------------------------------------------------------------------------
# Tree-sitter parsing tests
# ---------------------------------------------------------------------------

from processing.languages.java.java_processor import TREE_SITTER_AVAILABLE

SAMPLE_JAVA_TS = """\
package com.example.service;

import java.util.List;
import java.util.Map;
import static java.lang.Math.PI;

public abstract class UserService {
    private final String name;
    public static int counter;

    public UserService(String name) {
        this.name = name;
    }

    public List<User> getUsers() {
        return List.of();
    }

    private void processData(Map<String, Object> data) throws Exception {
    }

    protected static boolean isValid(String input) {
        return input != null;
    }
}

interface Validator {
    boolean validate(String input);
}

enum Status {
    ACTIVE,
    INACTIVE,
    PENDING
}
"""


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestJavaTreeSitterParsing:
    """Tests exercising tree-sitter parsing (not regex fallback)."""

    @pytest.fixture
    async def ts_processor(self):
        p = JavaProcessor()
        await p._initialize_impl()
        return p

    @pytest.fixture
    def java_file(self, tmp_path):
        fp = tmp_path / "Test.java"
        fp.write_text(SAMPLE_JAVA_TS)
        return fp

    @pytest.mark.asyncio
    async def test_tree_sitter_parse_success(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        assert result.success is True
        assert result.parser_used.value == "tree_sitter"
        assert len(result.entities) > 0

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_package(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        packages = [e for e in result.entities if e.type == EntityType.PACKAGE]
        assert len(packages) == 1
        assert packages[0].name == "com.example.service"

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_imports(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        import_names = {e.name for e in imports}
        assert "java.util.List" in import_names
        assert "java.util.Map" in import_names

    @pytest.mark.asyncio
    async def test_tree_sitter_static_import(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        static_imports = [e for e in imports if e.properties.get("is_static")]
        assert len(static_imports) >= 1

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_class(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "UserService" in class_names

    @pytest.mark.asyncio
    async def test_tree_sitter_class_modifiers(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        user_svc = next((c for c in classes if c.name == "UserService"), None)
        assert user_svc is not None
        assert user_svc.visibility == "public"
        assert user_svc.is_abstract is True

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_interface(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        interfaces = [e for e in result.entities if e.type == EntityType.INTERFACE]
        iface_names = {i.name for i in interfaces}
        assert "Validator" in iface_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_enum(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        enums = [e for e in result.entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "Status" in enum_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_methods(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        funcs = [e for e in result.entities if e.type == EntityType.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "getUsers" in func_names
        assert "processData" in func_names
        assert "isValid" in func_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_constructor(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        funcs = [e for e in result.entities if e.type == EntityType.FUNCTION]
        constructors = [f for f in funcs if f.properties.get("is_constructor")]
        assert len(constructors) >= 1

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_fields(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        var_names = {v.name for v in variables}
        assert "name" in var_names or "counter" in var_names

    @pytest.mark.asyncio
    async def test_tree_sitter_method_modifiers(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        funcs = [e for e in result.entities if e.type == EntityType.FUNCTION]
        is_valid = next((f for f in funcs if f.name == "isValid"), None)
        if is_valid:
            assert is_valid.is_static is True

    @pytest.mark.asyncio
    async def test_tree_sitter_entity_line_numbers(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        for entity in result.entities:
            assert entity.line_number >= 1

    @pytest.mark.asyncio
    async def test_tree_sitter_entity_language(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        for entity in result.entities:
            assert entity.language == "java"

    @pytest.mark.asyncio
    async def test_tree_sitter_relationships(self, ts_processor, java_file):
        result = await ts_processor.parse_file(java_file, SAMPLE_JAVA_TS)
        assert isinstance(result.relationships, list)
        # Should have import relationships at minimum
        assert len(result.relationships) > 0

    @pytest.mark.asyncio
    async def test_tree_sitter_empty_code(self, ts_processor, tmp_path):
        fp = tmp_path / "Empty.java"
        fp.write_text("")
        result = await ts_processor.parse_file(fp, "")
        assert result.success is True
        assert len(result.entities) == 0
