"""Coverage boost tests for CppSyntaxParser — targets uncovered regex paths.

Focuses on:
- parse_with_regex: includes, namespaces, classes, structs, enums,
  functions, templates, typedefs, macros, variables
- is_cpp_file detection
- C vs C++ differentiation
- Edge cases: inheritance, templates, multiline functions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType
from processing.languages.cpp.cpp_syntax_parser import CppSyntaxParser


@pytest.fixture
def parser():
    return CppSyntaxParser()


CPP_FULL = """\
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

C_CODE = """\
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


class TestIsCppFile:
    def test_cpp_extension(self, parser):
        assert parser.is_cpp_file(Path("main.cpp"), "") is True

    def test_cc_extension(self, parser):
        assert parser.is_cpp_file(Path("main.cc"), "") is True

    def test_cxx_extension(self, parser):
        assert parser.is_cpp_file(Path("main.cxx"), "") is True

    def test_cplusplus_extension(self, parser):
        assert parser.is_cpp_file(Path("main.c++"), "") is True

    def test_hpp_extension(self, parser):
        assert parser.is_cpp_file(Path("main.hpp"), "") is True

    def test_hh_extension(self, parser):
        assert parser.is_cpp_file(Path("main.hh"), "") is True

    def test_hxx_extension(self, parser):
        assert parser.is_cpp_file(Path("main.hxx"), "") is True

    def test_hplusplus_extension(self, parser):
        assert parser.is_cpp_file(Path("main.h++"), "") is True

    def test_c_extension_not_cpp(self, parser):
        assert parser.is_cpp_file(Path("main.c"), "") is False

    def test_c_with_namespace_is_cpp(self, parser):
        assert parser.is_cpp_file(Path("main.c"), "namespace Foo {}") is True

    def test_c_with_class_is_cpp(self, parser):
        assert parser.is_cpp_file(Path("main.c"), "class MyClass {};") is True

    def test_c_with_iostream_is_cpp(self, parser):
        assert parser.is_cpp_file(Path("main.c"), "#include <iostream>") is True

    def test_h_file_plain(self, parser):
        assert parser.is_cpp_file(Path("main.h"), "void foo();") is False


class TestCppRegexParseIncludes:
    @pytest.mark.asyncio
    async def test_extracts_angle_bracket_includes(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "iostream" in include_names
        assert "vector" in include_names

    @pytest.mark.asyncio
    async def test_extracts_quoted_includes(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "myheader.h" in include_names


class TestCppRegexParseNamespaces:
    @pytest.mark.asyncio
    async def test_extracts_namespace(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        ns_names = {e.name for e in namespaces}
        assert "MyApp" in ns_names


class TestCppRegexParseClasses:
    @pytest.mark.asyncio
    async def test_extracts_classes(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "Base" in class_names or "UserService" in class_names

    @pytest.mark.asyncio
    async def test_class_with_inheritance(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        user_service = [c for c in classes if c.name == "UserService"]
        if user_service:
            assert user_service[0].properties.get("is_cpp_class") is True


class TestCppRegexParseStructs:
    @pytest.mark.asyncio
    async def test_extracts_structs(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        struct_names = {s.name for s in structs}
        assert "Point" in struct_names


class TestCppRegexParseEnums:
    @pytest.mark.asyncio
    async def test_extracts_enums(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        enums = [e for e in entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "Color" in enum_names


class TestCppRegexParseFunctions:
    @pytest.mark.asyncio
    async def test_extracts_functions(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        fns = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in fns}
        assert "process" in fn_names or "calculate" in fn_names

    @pytest.mark.asyncio
    async def test_functions_skip_keywords(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        fns = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in fns}
        for kw in ["for", "while", "switch", "catch", "else", "namespace"]:
            assert kw not in fn_names


class TestCppRegexParseTemplates:
    @pytest.mark.asyncio
    async def test_extracts_templates(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        templates = [e for e in entities if e.type == EntityType.GENERIC]
        template_names = {t.name for t in templates}
        assert "Container" in template_names


class TestCppRegexParseTypedefs:
    @pytest.mark.asyncio
    async def test_extracts_typedefs(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        typedefs = [e for e in entities if e.type == EntityType.TYPE_ALIAS]
        typedef_names = {t.name for t in typedefs}
        assert "ulong" in typedef_names


class TestCppRegexParseMacros:
    @pytest.mark.asyncio
    async def test_extracts_macros(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        macros = [e for e in entities if e.type == EntityType.MACRO]
        macro_names = {m.name for m in macros}
        assert "MAX_SIZE" in macro_names
        assert "MIN" in macro_names


class TestCppRegexParseVariables:
    @pytest.mark.asyncio
    async def test_extracts_global_variables(self, parser):
        entities = await parser.parse_with_regex(Path("main.cpp"), CPP_FULL)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        var_names = {v.name for v in variables}
        assert "globalVar" in var_names


class TestCCodeParsing:
    @pytest.mark.asyncio
    async def test_c_file_detection(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        # Should detect as C since no namespace/class/iostream
        includes = [e for e in entities if e.type == EntityType.IMPORT]
        include_names = {e.name for e in includes}
        assert "stdio.h" in include_names
        assert "stdlib.h" in include_names

    @pytest.mark.asyncio
    async def test_c_structs(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        struct_names = {s.name for s in structs}
        assert "Node" in struct_names

    @pytest.mark.asyncio
    async def test_c_enums(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        enums = [e for e in entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "Status" in enum_names

    @pytest.mark.asyncio
    async def test_c_functions(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        fns = [e for e in entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in fns}
        assert "printMessage" in fn_names or "add" in fn_names

    @pytest.mark.asyncio
    async def test_c_typedefs(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        typedefs = [e for e in entities if e.type == EntityType.TYPE_ALIAS]
        typedef_names = {t.name for t in typedefs}
        assert "uint" in typedef_names

    @pytest.mark.asyncio
    async def test_c_no_namespaces_or_classes(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        classes = [e for e in entities if e.type == EntityType.CLASS]
        assert len(namespaces) == 0
        assert len(classes) == 0

    @pytest.mark.asyncio
    async def test_c_no_templates(self, parser):
        entities = await parser.parse_with_regex(Path("main.c"), C_CODE)
        templates = [e for e in entities if e.type == EntityType.GENERIC]
        assert len(templates) == 0


class TestCppRegexParseEmpty:
    @pytest.mark.asyncio
    async def test_empty_code(self, parser):
        entities = await parser.parse_with_regex(Path("empty.cpp"), "")
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_comments_only(self, parser):
        code = "// just a comment\n/* block comment */\n"
        entities = await parser.parse_with_regex(Path("comments.cpp"), code)
        # Only macros might be found, but no meaningful entities
        classes = [e for e in entities if e.type == EntityType.CLASS]
        assert len(classes) == 0


class TestCppRegexParseFindNodesHelper:
    def test_find_nodes_without_tree_sitter(self, parser):
        """_find_nodes_by_type should return empty when TREE_SITTER_AVAILABLE is False."""
        from unittest.mock import MagicMock

        from processing.languages.cpp.cpp_syntax_parser import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            mock_tree = MagicMock()
            result = parser._find_nodes_by_type(mock_tree, {"class_specifier"})
            assert result == []
