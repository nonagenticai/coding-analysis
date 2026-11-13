"""Tests for C++ syntax parser regex-based parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType
from processing.languages.cpp.cpp_syntax_parser import CppSyntaxParser


@pytest.fixture
def parser():
    return CppSyntaxParser()


SAMPLE_CPP_CODE = """\
#include <iostream>
#include <vector>
#include "myheader.h"

#define MAX_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace MyApp {

class Base {
public:
    virtual void process() = 0;
};

class UserService : public Base {
private:
    std::string name_;
    int count_;

public:
    void process() override {
        std::cout << name_ << std::endl;
    }

    static int calculate(int a, int b) {
        return a + b;
    }
};

struct Point {
    int x;
    int y;
};

enum class Color {
    Red,
    Green,
    Blue
};

typedef unsigned long ulong;
typedef int (*FuncPtr)(int, int);

template <typename T>
class Container {
    T value;
};

int globalVar = 42;

}  // namespace MyApp
"""

SAMPLE_C_CODE = """\
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

enum Status {
    OK,
    ERROR
};

typedef unsigned int uint;

void printMessage(const char* msg) {
    printf("%s\\n", msg);
}

int add(int a, int b) {
    return a + b;
}
"""


def test_is_cpp_file_by_extension(parser):
    assert parser.is_cpp_file(Path("main.cpp"), "") is True
    assert parser.is_cpp_file(Path("main.cc"), "") is True
    assert parser.is_cpp_file(Path("main.cxx"), "") is True
    assert parser.is_cpp_file(Path("main.hpp"), "") is True
    assert parser.is_cpp_file(Path("main.hxx"), "") is True


def test_is_cpp_file_by_content(parser):
    assert parser.is_cpp_file(Path("main.h"), "namespace MyApp {}") is True
    assert parser.is_cpp_file(Path("main.h"), "class Foo {};") is True
    assert parser.is_cpp_file(Path("main.h"), "#include <iostream>") is True


def test_is_not_cpp_file(parser):
    assert parser.is_cpp_file(Path("main.c"), "int x = 0;") is False


@pytest.mark.asyncio
async def test_parse_cpp_includes(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    includes = [e for e in entities if e.type.value == "import"]
    include_names = {e.name for e in includes}
    assert "iostream" in include_names
    assert "vector" in include_names
    assert "myheader.h" in include_names


@pytest.mark.asyncio
async def test_parse_cpp_namespaces(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    namespaces = [e for e in entities if e.type.value == "namespace"]
    ns_names = {e.name for e in namespaces}
    assert "MyApp" in ns_names


@pytest.mark.asyncio
async def test_parse_cpp_classes(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    classes = [e for e in entities if e.type.value == "class"]
    class_names = {e.name for e in classes}
    assert "Base" in class_names
    assert "UserService" in class_names


@pytest.mark.asyncio
async def test_parse_cpp_class_inheritance(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    classes = [e for e in entities if e.type.value == "class"]
    user_svc = next((c for c in classes if c.name == "UserService"), None)
    assert user_svc is not None
    inheritance = user_svc.properties.get("inheritance", [])
    assert len(inheritance) > 0
    assert any("Base" in i for i in inheritance)


@pytest.mark.asyncio
async def test_parse_cpp_structs(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    structs = [e for e in entities if e.type.value == "struct"]
    struct_names = {e.name for e in structs}
    assert "Point" in struct_names


@pytest.mark.asyncio
async def test_parse_cpp_enums(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    enums = [e for e in entities if e.type.value == "enum"]
    enum_names = {e.name for e in enums}
    assert "Color" in enum_names


@pytest.mark.asyncio
async def test_parse_cpp_functions(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    funcs = [e for e in entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    assert "calculate" in func_names


@pytest.mark.asyncio
async def test_parse_cpp_function_skips_keywords(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    funcs = [e for e in entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    for kw in ["for", "while", "switch", "catch", "else", "namespace"]:
        assert kw not in func_names


@pytest.mark.asyncio
async def test_parse_cpp_templates(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    templates = [e for e in entities if e.type.value == "generic"]
    template_names = {e.name for e in templates}
    assert "Container" in template_names


@pytest.mark.asyncio
async def test_parse_cpp_typedefs(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    typedefs = [e for e in entities if e.type.value == "type_alias"]
    typedef_names = {e.name for e in typedefs}
    assert "ulong" in typedef_names


@pytest.mark.asyncio
async def test_parse_cpp_macros(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    macros = [e for e in entities if e.type.value == "macro"]
    macro_names = {e.name for e in macros}
    assert "MAX_SIZE" in macro_names
    assert "MIN" in macro_names


@pytest.mark.asyncio
async def test_parse_cpp_variables(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    variables = [e for e in entities if e.type.value == "variable"]
    var_names = {e.name for e in variables}
    # Global variables should be detected
    assert "globalVar" in var_names


@pytest.mark.asyncio
async def test_parse_c_code(parser):
    """C code should be parsed as C, not C++."""
    fp = Path("main.c")
    entities = await parser.parse_with_regex(fp, SAMPLE_C_CODE)
    # Should not extract namespaces, classes, templates from C code
    c_entities = [e for e in entities if e.language == "c"]
    assert len(c_entities) > 0
    namespaces = [e for e in entities if e.type.value == "namespace"]
    assert len(namespaces) == 0


@pytest.mark.asyncio
async def test_parse_c_structs(parser):
    fp = Path("main.c")
    entities = await parser.parse_with_regex(fp, SAMPLE_C_CODE)
    structs = [e for e in entities if e.type.value == "struct"]
    struct_names = {e.name for e in structs}
    assert "Node" in struct_names


@pytest.mark.asyncio
async def test_parse_c_enums(parser):
    fp = Path("main.c")
    entities = await parser.parse_with_regex(fp, SAMPLE_C_CODE)
    enums = [e for e in entities if e.type.value == "enum"]
    enum_names = {e.name for e in enums}
    assert "Status" in enum_names


@pytest.mark.asyncio
async def test_parse_c_functions(parser):
    fp = Path("main.c")
    entities = await parser.parse_with_regex(fp, SAMPLE_C_CODE)
    funcs = [e for e in entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    assert "add" in func_names


@pytest.mark.asyncio
async def test_parse_empty_code(parser):
    fp = Path("empty.cpp")
    entities = await parser.parse_with_regex(fp, "")
    assert entities == []


@pytest.mark.asyncio
async def test_line_numbers_correct(parser):
    fp = Path("main.cpp")
    entities = await parser.parse_with_regex(fp, SAMPLE_CPP_CODE)
    for entity in entities:
        assert entity.line_number >= 1


# ---------------------------------------------------------------------------
# Tests merged from test_round4_cpp_syntax_parser.py
# ---------------------------------------------------------------------------

SAMPLE_CPP_R4 = """\
#include <iostream>
#include <vector>
#include "myheader.h"

namespace MyApp {

class Animal : public Base {
public:
    virtual void speak() const;
    int getAge() const { return age_; }
private:
    int age_;
};

struct Point {
    double x;
    double y;
};

enum class Color {
    Red,
    Green,
    Blue
};

template<typename T>
T add(T a, T b) {
    return a + b;
}

typedef unsigned int uint32;

void globalFunction(int x) {
    // body
}

static int helperFunction(double val) {
    return static_cast<int>(val);
}

#define MAX_SIZE 100
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int globalVar = 42;
std::string name = "test";

} // namespace MyApp
"""

SAMPLE_C_R4 = """\
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int value;
    struct Node* next;
};

enum Status {
    OK,
    ERROR
};

void process(int* data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d\\n", data[i]);
    }
}

typedef struct Node NodeType;
"""


class TestCppDetection:
    def test_is_cpp_file_by_extension(self, parser):
        assert parser.is_cpp_file(Path("test.cpp"), "") is True
        assert parser.is_cpp_file(Path("test.cc"), "") is True
        assert parser.is_cpp_file(Path("test.hpp"), "") is True
        assert parser.is_cpp_file(Path("test.cxx"), "") is True
        assert parser.is_cpp_file(Path("test.h++"), "") is True

    def test_is_cpp_by_content(self, parser):
        assert parser.is_cpp_file(Path("test.h"), "namespace Foo {") is True
        assert parser.is_cpp_file(Path("test.h"), "class Foo {") is True
        assert parser.is_cpp_file(Path("test.h"), "#include <iostream>") is True

    def test_is_c_file(self, parser):
        assert parser.is_cpp_file(Path("test.c"), "int main() {}") is False


class TestCppRegexParsing:
    @pytest.mark.asyncio
    async def test_extracts_includes(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "iostream" in include_names
        assert "vector" in include_names
        assert "myheader.h" in include_names

    @pytest.mark.asyncio
    async def test_extracts_namespace(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        assert any(ns.name == "MyApp" for ns in namespaces)

    @pytest.mark.asyncio
    async def test_extracts_class(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        assert any(c.name == "Animal" for c in classes)
        animal = [c for c in classes if c.name == "Animal"][0]
        assert animal.properties["is_cpp_class"] is True

    @pytest.mark.asyncio
    async def test_class_inheritance(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        animal = [c for c in classes if c.name == "Animal"][0]
        assert len(animal.properties["inheritance"]) > 0

    @pytest.mark.asyncio
    async def test_extracts_struct(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        assert any(s.name == "Point" for s in structs)

    @pytest.mark.asyncio
    async def test_extracts_enum(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        enums = [e for e in entities if e.type == EntityType.ENUM]
        assert any(e.name == "Color" for e in enums)

    @pytest.mark.asyncio
    async def test_extracts_functions(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in functions}
        assert "globalFunction" in fn_names
        assert "helperFunction" in fn_names

    @pytest.mark.asyncio
    async def test_skips_keywords(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in functions}
        assert "for" not in fn_names
        assert "while" not in fn_names
        assert "switch" not in fn_names

    @pytest.mark.asyncio
    async def test_extracts_templates(self, parser):
        """Template 'add' should be extracted -- either as a GENERIC template entity or as a FUNCTION."""
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in functions}
        assert "add" in fn_names

    @pytest.mark.asyncio
    async def test_extracts_typedefs(self, parser):
        """Typedef 'uint32' should be extracted as TYPE_ALIAS."""
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        typedefs = [e for e in entities if e.type == EntityType.TYPE_ALIAS]
        typedef_names = {t.name for t in typedefs}
        assert "uint32" in typedef_names
        uint32_entity = [t for t in typedefs if t.name == "uint32"][0]
        assert "unsigned" in uint32_entity.properties["base_type"]

    @pytest.mark.asyncio
    async def test_extracts_macros(self, parser):
        """Macros MAX_SIZE and MIN should be found."""
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        macros = [e for e in entities if e.type == EntityType.MACRO]
        macro_names = {m.name for m in macros}
        assert "MAX_SIZE" in macro_names
        assert "MIN" in macro_names
        for macro in macros:
            assert macro.properties["is_macro"] is True

    @pytest.mark.asyncio
    async def test_extracts_variables(self, parser):
        """Global variables globalVar and name should be found."""
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        var_names = {v.name for v in variables}
        assert "globalVar" in var_names or "name" in var_names
        for var in variables:
            assert var.properties["is_global"] is True
            assert "variable_type" in var.properties

    @pytest.mark.asyncio
    async def test_empty_code(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), "")
        assert entities == []


class TestCCodeRegexParsing:
    @pytest.mark.asyncio
    async def test_c_file_includes(self, parser):
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "stdio.h" in include_names
        assert "stdlib.h" in include_names

    @pytest.mark.asyncio
    async def test_c_file_struct(self, parser):
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        assert any(s.name == "Node" for s in structs)

    @pytest.mark.asyncio
    async def test_c_file_enum(self, parser):
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        enums = [e for e in entities if e.type == EntityType.ENUM]
        assert any(e.name == "Status" for e in enums)

    @pytest.mark.asyncio
    async def test_c_file_function(self, parser):
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in functions}
        assert "process" in fn_names

    @pytest.mark.asyncio
    async def test_c_file_no_namespace(self, parser):
        """C files should not extract namespaces."""
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        assert namespaces == []

    @pytest.mark.asyncio
    async def test_c_file_no_class(self, parser):
        """C files should not extract classes."""
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        assert classes == []

    @pytest.mark.asyncio
    async def test_c_file_language_label(self, parser):
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        for e in entities:
            assert e.language == "c"

    @pytest.mark.asyncio
    async def test_c_file_typedef(self, parser):
        """C sample has 'typedef struct Node NodeType' which should be extracted."""
        entities = await parser.parse_with_regex(Path("test.c"), SAMPLE_C_R4)
        typedefs = [e for e in entities if e.type == EntityType.TYPE_ALIAS]
        typedef_names = {t.name for t in typedefs}
        assert "NodeType" in typedef_names


class TestCppEntityAttributes:
    @pytest.mark.asyncio
    async def test_line_numbers(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        for e in entities:
            assert e.line_number >= 1

    @pytest.mark.asyncio
    async def test_file_paths(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        for e in entities:
            assert e.file_path == "test.cpp"

    @pytest.mark.asyncio
    async def test_language_cpp(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        for e in entities:
            assert e.language == "cpp"

    @pytest.mark.asyncio
    async def test_entity_ids_unique(self, parser):
        entities = await parser.parse_with_regex(Path("test.cpp"), SAMPLE_CPP_R4)
        ids = [e.id for e in entities]
        assert len(ids) == len(set(ids))


class TestCppInit:
    def test_constructor(self):
        p = CppSyntaxParser()
        assert "namespace_definition" in p.NAMESPACE_NODE_TYPES
        assert "class_specifier" in p.CLASS_NODE_TYPES


# ---------------------------------------------------------------------------
# Tree-sitter parsing tests
# ---------------------------------------------------------------------------

from processing.languages.cpp.cpp_syntax_parser import TREE_SITTER_AVAILABLE

SAMPLE_CPP_TS = """\
#include <iostream>
#include <vector>
#include "myheader.h"

#define MAX_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace MyApp {

class Base {
public:
    virtual void process() = 0;
};

class UserService : public Base {
private:
    std::string name_;

public:
    void process() override {
        std::cout << name_ << std::endl;
    }

    static int calculate(int a, int b) {
        return a + b;
    }
};

struct Point {
    int x;
    int y;
};

enum Color {
    Red,
    Green,
    Blue
};

typedef unsigned long ulong;

template<typename T>
class Container {
    T value;
};

int globalVar = 42;

}
"""


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestCppTreeSitterParsing:
    """Tests exercising tree-sitter parsing for C++."""

    @pytest.fixture
    def ts_parser(self):
        return CppSyntaxParser()

    @pytest.fixture
    def cpp_tree(self):
        import tree_sitter_cpp
        from tree_sitter import Language, Parser

        lang = Language(tree_sitter_cpp.language())
        parser = Parser(lang)
        tree = parser.parse(SAMPLE_CPP_TS.encode("utf-8"))
        return tree

    @pytest.fixture
    def content_bytes(self):
        return SAMPLE_CPP_TS.encode("utf-8")

    @pytest.mark.asyncio
    async def test_parse_with_tree_sitter_returns_entities(
        self, ts_parser, cpp_tree, content_bytes
    ):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_ts_extracts_includes(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "iostream" in include_names
        assert "vector" in include_names
        assert "myheader.h" in include_names

    @pytest.mark.asyncio
    async def test_ts_extracts_namespaces(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        ns_names = {e.name for e in namespaces}
        assert "MyApp" in ns_names

    @pytest.mark.asyncio
    async def test_ts_extracts_classes(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "Base" in class_names
        assert "UserService" in class_names

    @pytest.mark.asyncio
    async def test_ts_extracts_structs(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        struct_names = {s.name for s in structs}
        assert "Point" in struct_names

    @pytest.mark.asyncio
    async def test_ts_extracts_enums(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        enums = [e for e in entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "Color" in enum_names

    @pytest.mark.asyncio
    async def test_ts_extracts_functions(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in functions}
        assert len(fn_names) > 0

    @pytest.mark.asyncio
    async def test_ts_extracts_typedefs(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        typedefs = [e for e in entities if e.type == EntityType.TYPE_ALIAS]
        typedef_names = {t.name for t in typedefs}
        assert "ulong" in typedef_names

    @pytest.mark.asyncio
    async def test_ts_extracts_templates(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        templates = [e for e in entities if e.type == EntityType.GENERIC]
        template_names = {t.name for t in templates}
        assert "Container" in template_names

    @pytest.mark.asyncio
    async def test_ts_entity_line_numbers(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        for entity in entities:
            assert entity.line_number >= 1

    @pytest.mark.asyncio
    async def test_ts_entity_language(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        for entity in entities:
            assert entity.language == "cpp"

    @pytest.mark.asyncio
    async def test_ts_entity_file_path(self, ts_parser, cpp_tree, content_bytes):
        entities = await ts_parser.parse_with_tree_sitter(cpp_tree, Path("test.cpp"), content_bytes)
        for entity in entities:
            assert entity.file_path == "test.cpp"

    @pytest.mark.asyncio
    async def test_ts_empty_code(self, ts_parser):
        import tree_sitter_cpp
        from tree_sitter import Language, Parser

        lang = Language(tree_sitter_cpp.language())
        p = Parser(lang)
        tree = p.parse(b"")
        entities = await ts_parser.parse_with_tree_sitter(tree, Path("empty.cpp"), b"")
        assert entities == []

    def test_find_nodes_by_type(self, ts_parser, cpp_tree):
        nodes = ts_parser._find_nodes_by_type(cpp_tree, ts_parser.CLASS_NODE_TYPES)
        assert len(nodes) >= 2  # Base and UserService

    def test_get_node_text(self, ts_parser, cpp_tree, content_bytes):
        nodes = ts_parser._find_nodes_by_type(cpp_tree, ts_parser.NAMESPACE_NODE_TYPES)
        assert len(nodes) >= 1
        text = ts_parser._get_node_text(nodes[0], content_bytes)
        assert "MyApp" in text
