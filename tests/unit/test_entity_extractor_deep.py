"""Unit tests for processing.extractors.entity_extractor.EntityExtractor."""

from __future__ import annotations

import ast

import pytest

from models.core_models import CoreCodeEntity, EntityType
from processing.extractors.entity_extractor import EntityExtractor

FILE_PATH = "/src/example.py"


@pytest.fixture
def extractor() -> EntityExtractor:
    """Return a fresh EntityExtractor instance."""
    return EntityExtractor()


def _parse(code: str) -> ast.Module:
    """Helper: parse code string into an AST Module."""
    return ast.parse(code)


def _lines(code: str) -> list[str]:
    """Helper: split code into lines for file_lines param."""
    return code.splitlines()


# ---------------------------------------------------------------------------
# _create_entity_with_defaults
# ---------------------------------------------------------------------------


class TestCreateEntityWithDefaults:
    """Tests for the _create_entity_with_defaults helper."""

    def test_minimal_entity(self, extractor: EntityExtractor) -> None:
        entity = extractor._create_entity_with_defaults(
            entity_id="test-id",
            name="test_func",
            entity_type=EntityType.FUNCTION,
            file_path=FILE_PATH,
            line_number=10,
        )
        assert isinstance(entity, CoreCodeEntity)
        assert entity.id == "test-id"
        assert entity.name == "test_func"
        assert entity.type == EntityType.FUNCTION
        assert entity.file_path == FILE_PATH
        assert entity.line_number == 10
        assert entity.language == "python"
        assert entity.visibility == "public"
        assert entity.column_start == 0
        assert entity.column_end == 0

    def test_custom_language(self, extractor: EntityExtractor) -> None:
        entity = extractor._create_entity_with_defaults(
            entity_id="js-id",
            name="myFunc",
            entity_type=EntityType.FUNCTION,
            file_path="/src/app.js",
            line_number=1,
            language="javascript",
        )
        assert entity.language == "javascript"

    def test_kwargs_forwarded(self, extractor: EntityExtractor) -> None:
        entity = extractor._create_entity_with_defaults(
            entity_id="id",
            name="x",
            entity_type=EntityType.VARIABLE,
            file_path=FILE_PATH,
            line_number=5,
            column_start=4,
            column_end=10,
            visibility="private",
            signature="x: int = 5",
            source_code="x: int = 5",
            end_line_number=5,
            code_snippet="x: int = 5",
            docstring="A variable",
            metadata={"custom": True},
        )
        assert entity.column_start == 4
        assert entity.column_end == 10
        assert entity.visibility == "private"
        assert entity.signature == "x: int = 5"
        assert entity.source_code == "x: int = 5"
        assert entity.end_line_number == 5
        assert entity.code_snippet == "x: int = 5"
        assert entity.docstring == "A variable"
        assert entity.metadata["custom"] is True


# ---------------------------------------------------------------------------
# extract_python_entities
# ---------------------------------------------------------------------------


class TestExtractPythonEntities:
    """Tests for extract_python_entities (async)."""

    async def test_empty_module(self, extractor: EntityExtractor) -> None:
        code = ""
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        # Should at minimum have a module entity
        assert len(entities) >= 1
        modules = [e for e in entities if e.type == EntityType.MODULE]
        assert len(modules) == 1

    async def test_module_entity_metadata(self, extractor: EntityExtractor) -> None:
        code = '"""Module docstring."""\nx = 1\n'
        tree = _parse(code)
        lines = _lines(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH, lines)
        module = [e for e in entities if e.type == EntityType.MODULE][0]
        assert module.metadata["total_lines"] == len(lines)
        assert module.metadata["is_package"] is False
        assert module.docstring == "Module docstring."

    async def test_package_init_module(self, extractor: EntityExtractor) -> None:
        code = ""
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, "/src/mypackage/__init__.py")
        module = [e for e in entities if e.type == EntityType.MODULE][0]
        assert module.metadata["is_package"] is True

    async def test_class_extraction(self, extractor: EntityExtractor) -> None:
        code = "class MyClass:\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        classes = [e for e in entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"
        assert classes[0].line_number == 1

    async def test_class_with_base_classes(self, extractor: EntityExtractor) -> None:
        code = "class Child(Base, Mixin):\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert "Base" in cls.metadata["base_classes"]
        assert "Mixin" in cls.metadata["base_classes"]

    async def test_class_with_decorators(self, extractor: EntityExtractor) -> None:
        code = "@dataclass\nclass Data:\n    x: int = 0\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert "dataclass" in cls.metadata["decorators"]

    async def test_class_with_docstring(self, extractor: EntityExtractor) -> None:
        code = 'class Foo:\n    """Foo docs."""\n    pass\n'
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert cls.metadata["docstring"] == "Foo docs."
        assert cls.docstring == "Foo docs."

    async def test_class_methods_extraction(self, extractor: EntityExtractor) -> None:
        code = (
            "class Service:\n"
            "    def sync_method(self):\n"
            "        pass\n"
            "    async def async_method(self):\n"
            "        pass\n"
            "    @property\n"
            "    def prop(self):\n"
            "        return 1\n"
            "    @classmethod\n"
            "    def from_config(cls):\n"
            "        pass\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        pass\n"
        )
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        methods = cls.metadata["methods"]
        assert len(methods) == 5

        sync = [m for m in methods if m["name"] == "sync_method"][0]
        assert sync["is_async"] is False
        assert sync["is_property"] is False

        am = [m for m in methods if m["name"] == "async_method"][0]
        assert am["is_async"] is True

        prop = [m for m in methods if m["name"] == "prop"][0]
        assert prop["is_property"] is True

        cm = [m for m in methods if m["name"] == "from_config"][0]
        assert cm["is_classmethod"] is True

        sm = [m for m in methods if m["name"] == "helper"][0]
        assert sm["is_staticmethod"] is True

    async def test_class_variables_extraction(self, extractor: EntityExtractor) -> None:
        code = "class Config:\n    DEBUG = True\n    name = 'app'\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert "DEBUG" in cls.metadata["class_variables"]
        assert "name" in cls.metadata["class_variables"]

    async def test_class_code_snippet(self, extractor: EntityExtractor) -> None:
        code = "class Foo:\n    pass\n"
        tree = _parse(code)
        lines = _lines(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH, lines)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert cls.code_snippet is not None
        assert "class Foo" in cls.code_snippet

    async def test_function_extraction(self, extractor: EntityExtractor) -> None:
        code = "def hello(name: str) -> str:\n    return f'Hello {name}'\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        funcs = [e for e in entities if e.type == EntityType.FUNCTION]
        assert len(funcs) == 1
        fn = funcs[0]
        assert fn.name == "hello"
        assert fn.metadata["is_async"] is False
        assert fn.metadata["return_annotation"] == "str"
        assert fn.metadata["parameters"][0]["name"] == "name"
        assert fn.metadata["parameters"][0]["annotation"] == "str"

    async def test_function_with_decorators(self, extractor: EntityExtractor) -> None:
        code = "@staticmethod\ndef helper():\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert "staticmethod" in fn.metadata["decorators"]
        assert fn.metadata["is_staticmethod"] is True

    async def test_function_complexity(self, extractor: EntityExtractor) -> None:
        code = (
            "def complex_fn(x):\n"
            "    if x > 0:\n"
            "        for i in range(x):\n"
            "            if i % 2 == 0:\n"
            "                return i\n"
            "    return -1\n"
        )
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.metadata["cyclomatic_complexity"] >= 3
        assert fn.metadata["cognitive_complexity"] >= 1

    async def test_function_with_self_is_method(self, extractor: EntityExtractor) -> None:
        # When extracted standalone, a function with 'sel' as first param is a method
        # Note: the code checks for "sel" not "self" — testing what exists
        code = "def method(sel, x):\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.metadata["is_method"] is True

    async def test_function_with_cls_is_method(self, extractor: EntityExtractor) -> None:
        code = "def method(cls):\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.metadata["is_method"] is True

    async def test_function_no_params_not_method(self, extractor: EntityExtractor) -> None:
        code = "def standalone():\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.metadata["is_method"] is False

    async def test_function_code_snippet(self, extractor: EntityExtractor) -> None:
        code = "def foo():\n    return 42\n"
        tree = _parse(code)
        lines = _lines(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH, lines)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.code_snippet is not None
        assert "def foo" in fn.code_snippet

    async def test_async_function_extraction(self, extractor: EntityExtractor) -> None:
        code = "async def fetch(url: str) -> bytes:\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        funcs = [e for e in entities if e.type == EntityType.FUNCTION]
        assert len(funcs) == 1
        fn = funcs[0]
        assert fn.name == "fetch"
        assert fn.metadata["is_async"] is True
        assert fn.metadata["return_annotation"] == "bytes"

    async def test_variable_assignment(self, extractor: EntityExtractor) -> None:
        code = "MAX_SIZE = 100\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        assert len(variables) == 1
        v = variables[0]
        assert v.name == "MAX_SIZE"
        assert v.metadata["is_constant"] is True
        assert v.metadata["value_type"] == "int"

    async def test_non_constant_variable(self, extractor: EntityExtractor) -> None:
        code = "count = 0\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        v = variables[0]
        assert v.name == "count"
        assert v.metadata["is_constant"] is False

    async def test_multiple_assignment_targets(self, extractor: EntityExtractor) -> None:
        # Multiple targets like x = y = 1 produces Assign with two targets
        code = "x = y = 1\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        names = {v.name for v in variables}
        assert "x" in names
        assert "y" in names

    async def test_annotated_variable(self, extractor: EntityExtractor) -> None:
        code = "name: str = 'hello'\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        assert len(variables) == 1
        v = variables[0]
        assert v.name == "name"
        assert v.metadata["annotation"] == "str"
        assert v.metadata["is_annotated"] is True
        assert v.metadata["has_value"] is True
        assert v.metadata["value_type"] == "str"

    async def test_annotated_variable_no_value(self, extractor: EntityExtractor) -> None:
        code = "count: int\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        variables = [e for e in entities if e.type == EntityType.VARIABLE]
        assert len(variables) == 1
        v = variables[0]
        assert v.name == "count"
        assert v.metadata["has_value"] is False
        assert v.metadata["value_type"] is None

    async def test_mixed_entities(self, extractor: EntityExtractor) -> None:
        code = (
            "MAX = 10\n"
            "class Foo:\n"
            "    pass\n"
            "def bar():\n"
            "    pass\n"
            "async def baz():\n"
            "    pass\n"
            "x: int = 5\n"
        )
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        types_found = {e.type for e in entities}
        assert EntityType.MODULE in types_found
        assert EntityType.CLASS in types_found
        assert EntityType.FUNCTION in types_found
        assert EntityType.VARIABLE in types_found

    async def test_no_file_lines_no_code_snippet(self, extractor: EntityExtractor) -> None:
        code = "class Foo:\n    pass\n"
        tree = _parse(code)
        # No file_lines passed
        entities = await extractor.extract_python_entities(tree, FILE_PATH, None)
        cls = [e for e in entities if e.type == EntityType.CLASS][0]
        assert cls.code_snippet is None


# ---------------------------------------------------------------------------
# extract_import_entities
# ---------------------------------------------------------------------------


class TestExtractImportEntities:
    """Tests for extract_import_entities (async)."""

    async def test_import_statement(self, extractor: EntityExtractor) -> None:
        code = "import os\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        assert len(entities) == 1
        imp = entities[0]
        assert imp.name == "os"
        assert imp.type == EntityType.IMPORT
        assert imp.metadata["import_type"] == "import"
        assert imp.metadata["module"] == "os"
        assert imp.metadata["alias"] is None

    async def test_import_with_alias(self, extractor: EntityExtractor) -> None:
        code = "import numpy as np\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        imp = entities[0]
        assert imp.name == "numpy"
        assert imp.metadata["alias"] == "np"

    async def test_import_multiple(self, extractor: EntityExtractor) -> None:
        code = "import os, sys\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "os" in names
        assert "sys" in names

    async def test_from_import(self, extractor: EntityExtractor) -> None:
        code = "from os.path import join\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        assert len(entities) == 1
        imp = entities[0]
        assert imp.name == "join"
        assert imp.metadata["import_type"] == "from_import"
        assert imp.metadata["module"] == "os.path"
        assert imp.metadata["full_name"] == "os.path.join"

    async def test_from_import_with_alias(self, extractor: EntityExtractor) -> None:
        code = "from collections import OrderedDict as OD\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        imp = entities[0]
        assert imp.name == "OrderedDict"
        assert imp.metadata["alias"] == "OD"

    async def test_from_import_multiple(self, extractor: EntityExtractor) -> None:
        code = "from typing import List, Dict, Optional\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        assert len(entities) == 3
        names = {e.name for e in entities}
        assert "List" in names
        assert "Dict" in names
        assert "Optional" in names

    async def test_relative_import(self, extractor: EntityExtractor) -> None:
        code = "from . import utils\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        imp = entities[0]
        assert imp.metadata["level"] == 1
        # Relative imports are not external
        assert imp.metadata["is_external"] is False

    async def test_relative_import_from_parent(self, extractor: EntityExtractor) -> None:
        code = "from ..models import User\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        imp = entities[0]
        assert imp.metadata["level"] == 2

    async def test_external_import_detection(self, extractor: EntityExtractor) -> None:
        code = "from fastapi import FastAPI\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        imp = entities[0]
        assert imp.metadata["is_external"] is True

    async def test_no_imports(self, extractor: EntityExtractor) -> None:
        code = "x = 1\n"
        tree = _parse(code)
        entities = await extractor.extract_import_entities(tree, FILE_PATH)
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# _is_external_import
# ---------------------------------------------------------------------------


class TestIsExternalImport:
    """Tests for the _is_external_import helper."""

    def test_empty_module_name(self, extractor: EntityExtractor) -> None:
        assert extractor._is_external_import("") is False

    def test_stdlib_modules(self, extractor: EntityExtractor) -> None:
        for mod in ["os", "sys", "json", "datetime", "typing", "logging", "asyncio"]:
            assert extractor._is_external_import(mod) is True, f"{mod} should be external"

    def test_stdlib_submodule(self, extractor: EntityExtractor) -> None:
        assert extractor._is_external_import("os.path") is True

    def test_third_party_modules(self, extractor: EntityExtractor) -> None:
        for mod in ["numpy", "pandas", "requests", "flask", "django", "fastapi"]:
            assert extractor._is_external_import(mod) is True, f"{mod} should be external"

    def test_third_party_submodule(self, extractor: EntityExtractor) -> None:
        assert extractor._is_external_import("numpy.linalg") is True

    def test_relative_import(self, extractor: EntityExtractor) -> None:
        assert extractor._is_external_import(".utils") is False

    def test_dotdot_relative_import(self, extractor: EntityExtractor) -> None:
        assert extractor._is_external_import("..models") is False

    def test_unknown_module_defaults_external(self, extractor: EntityExtractor) -> None:
        # Unknown module names default to external=True
        assert extractor._is_external_import("my_custom_module") is True


# ---------------------------------------------------------------------------
# _get_value_type
# ---------------------------------------------------------------------------


class TestGetValueType:
    """Tests for _get_value_type."""

    def test_int_constant(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = 42").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "int"

    def test_str_constant(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = 'hello'").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "str"

    def test_float_constant(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = 3.14").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "float"

    def test_bool_constant(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = True").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "bool"

    def test_none_constant(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = None").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "NoneType"

    def test_list_value(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = [1, 2, 3]").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "list"

    def test_dict_value(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = {'a': 1}").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "dict"

    def test_set_value(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = {1, 2, 3}").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "set"

    def test_tuple_value(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = (1, 2)").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "tuple"

    def test_call_value(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = MyClass()").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "MyClass"

    def test_unknown_value(self, extractor: EntityExtractor) -> None:
        # Attribute access isn't covered, returns "unknown"
        node = ast.parse("x = obj.attr").body[0]
        assert isinstance(node, ast.Assign)
        assert extractor._get_value_type(node.value) == "unknown"


# ---------------------------------------------------------------------------
# _get_target_annotation
# ---------------------------------------------------------------------------


class TestGetTargetAnnotation:
    """Tests for _get_target_annotation."""

    def test_no_annotation(self, extractor: EntityExtractor) -> None:
        node = ast.parse("x = 1").body[0]
        assert isinstance(node, ast.Assign)
        target = node.targets[0]
        result = extractor._get_target_annotation(target)
        assert result is None

    def test_with_annotation_attr(self, extractor: EntityExtractor) -> None:
        # Manually attach an annotation attribute to test the branch
        node = ast.parse("x = 1").body[0]
        assert isinstance(node, ast.Assign)
        target = node.targets[0]
        target.annotation = ast.Name(id="int")  # type: ignore[attr-defined]
        result = extractor._get_target_annotation(target)
        assert result == "int"


# ---------------------------------------------------------------------------
# _extract_annotation_string
# ---------------------------------------------------------------------------


class TestExtractAnnotationString:
    """Tests for _extract_annotation_string."""

    def test_name_annotation(self, extractor: EntityExtractor) -> None:
        node = ast.Name(id="int")
        assert extractor._extract_annotation_string(node) == "int"

    def test_constant_annotation(self, extractor: EntityExtractor) -> None:
        node = ast.Constant(value="List[str]")
        assert extractor._extract_annotation_string(node) == "List[str]"

    def test_attribute_annotation(self, extractor: EntityExtractor) -> None:
        # typing.List
        node = ast.Attribute(
            value=ast.Name(id="typing"),
            attr="List",
        )
        assert extractor._extract_annotation_string(node) == "typing.List"

    def test_subscript_annotation(self, extractor: EntityExtractor) -> None:
        # List[str]
        node = ast.Subscript(
            value=ast.Name(id="List"),
            slice=ast.Name(id="str"),
        )
        assert extractor._extract_annotation_string(node) == "List[str]"

    def test_tuple_annotation(self, extractor: EntityExtractor) -> None:
        # str, int (as part of Dict[str, int])
        node = ast.Tuple(
            elts=[ast.Name(id="str"), ast.Name(id="int")],
        )
        assert extractor._extract_annotation_string(node) == "str, int"

    def test_bitor_union_annotation(self, extractor: EntityExtractor) -> None:
        # str | int
        node = ast.BinOp(
            left=ast.Name(id="str"),
            op=ast.BitOr(),
            right=ast.Name(id="int"),
        )
        assert extractor._extract_annotation_string(node) == "str | int"

    def test_non_bitor_binop_falls_through(self, extractor: EntityExtractor) -> None:
        # A BinOp that isn't BitOr should fall through to unparse
        node = ast.BinOp(
            left=ast.Name(id="a"),
            op=ast.Add(),
            right=ast.Name(id="b"),
        )
        result = extractor._extract_annotation_string(node)
        # Should go through unparse path
        assert result is not None
        assert isinstance(result, str)

    def test_complex_subscript_annotation(self, extractor: EntityExtractor) -> None:
        # Dict[str, int]
        node = ast.Subscript(
            value=ast.Name(id="Dict"),
            slice=ast.Tuple(
                elts=[ast.Name(id="str"), ast.Name(id="int")],
            ),
        )
        assert extractor._extract_annotation_string(node) == "Dict[str, int]"

    def test_fallback_with_id(self, extractor: EntityExtractor) -> None:
        # Create a node that has an 'id' attribute but isn't ast.Name
        # to test the hasattr(annotation_node, "id") fallback
        class FakeNode:
            id = "fallback_name"

        result = extractor._extract_annotation_string(FakeNode())  # type: ignore[arg-type]
        assert result == "fallback_name"


# ---------------------------------------------------------------------------
# _create_module_entity
# ---------------------------------------------------------------------------


class TestCreateModuleEntity:
    """Tests for _create_module_entity."""

    def test_module_from_regular_file(self, extractor: EntityExtractor) -> None:
        code = '"""Doc."""\n'
        tree = _parse(code)
        entity = extractor._create_module_entity(tree, FILE_PATH, _lines(code))
        assert entity.type == EntityType.MODULE
        assert entity.name == "example"
        assert entity.metadata["total_lines"] == 1
        assert entity.metadata["is_package"] is False

    def test_module_from_init_file(self, extractor: EntityExtractor) -> None:
        tree = _parse("")
        entity = extractor._create_module_entity(tree, "/src/mypackage/__init__.py", [])
        assert entity.name == "mypackage"
        assert entity.metadata["is_package"] is True

    def test_module_no_file_lines(self, extractor: EntityExtractor) -> None:
        tree = _parse("")
        entity = extractor._create_module_entity(tree, FILE_PATH, None)
        assert entity.metadata["total_lines"] == 0

    def test_module_docstring_only_for_ast_module(self, extractor: EntityExtractor) -> None:
        # A non-Module AST should not extract docstring
        code = "def foo():\n    pass\n"
        tree = _parse(code)
        func_node = tree.body[0]  # FunctionDef, not Module
        entity = extractor._create_module_entity(func_node, FILE_PATH)  # type: ignore[arg-type]
        assert entity.metadata["docstring"] is None


# ---------------------------------------------------------------------------
# _create_function_entity
# ---------------------------------------------------------------------------


class TestCreateFunctionEntity:
    """Tests for _create_function_entity."""

    def test_function_with_return_annotation(self, extractor: EntityExtractor) -> None:
        code = "def greet(name: str) -> str:\n    return name\n"
        tree = _parse(code)
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        entity = extractor._create_function_entity(node, FILE_PATH)
        assert entity.metadata["return_annotation"] == "str"

    def test_function_no_return_annotation(self, extractor: EntityExtractor) -> None:
        code = "def greet(name):\n    return name\n"
        tree = _parse(code)
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        entity = extractor._create_function_entity(node, FILE_PATH)
        assert entity.metadata["return_annotation"] is None

    def test_property_decorator_detection(self, extractor: EntityExtractor) -> None:
        code = "@property\ndef x(self):\n    return 1\n"
        tree = _parse(code)
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        entity = extractor._create_function_entity(node, FILE_PATH)
        assert entity.metadata["is_property"] is True


# ---------------------------------------------------------------------------
# _create_async_function_entity
# ---------------------------------------------------------------------------


class TestCreateAsyncFunctionEntity:
    """Tests for _create_async_function_entity."""

    def test_async_flag_set(self, extractor: EntityExtractor) -> None:
        code = "async def fetch():\n    pass\n"
        tree = _parse(code)
        node = tree.body[0]
        assert isinstance(node, ast.AsyncFunctionDef)
        entity = extractor._create_async_function_entity(node, FILE_PATH)
        assert entity.metadata["is_async"] is True
        assert entity.type == EntityType.FUNCTION

    def test_async_inherits_function_metadata(self, extractor: EntityExtractor) -> None:
        code = "async def process(data: list) -> dict:\n    pass\n"
        tree = _parse(code)
        node = tree.body[0]
        assert isinstance(node, ast.AsyncFunctionDef)
        entity = extractor._create_async_function_entity(node, FILE_PATH)
        assert entity.metadata["return_annotation"] == "dict"
        assert entity.metadata["parameters"][0]["name"] == "data"


# ---------------------------------------------------------------------------
# _create_import_entities
# ---------------------------------------------------------------------------


class TestCreateImportEntities:
    """Tests for _create_import_entities."""

    def test_single_import(self, extractor: EntityExtractor) -> None:
        tree = _parse("import os")
        node = tree.body[0]
        assert isinstance(node, ast.Import)
        entities = extractor._create_import_entities(node, FILE_PATH)
        assert len(entities) == 1
        assert entities[0].metadata["is_external"] is True

    def test_multi_import(self, extractor: EntityExtractor) -> None:
        tree = _parse("import os, sys, json")
        node = tree.body[0]
        assert isinstance(node, ast.Import)
        entities = extractor._create_import_entities(node, FILE_PATH)
        assert len(entities) == 3


# ---------------------------------------------------------------------------
# _create_import_from_entities
# ---------------------------------------------------------------------------


class TestCreateImportFromEntities:
    """Tests for _create_import_from_entities."""

    def test_from_import_no_module(self, extractor: EntityExtractor) -> None:
        # "from . import foo" has module=None
        tree = _parse("from . import foo")
        node = tree.body[0]
        assert isinstance(node, ast.ImportFrom)
        entities = extractor._create_import_from_entities(node, FILE_PATH)
        assert len(entities) == 1
        imp = entities[0]
        assert imp.name == "foo"
        assert imp.metadata["module"] == ""
        assert imp.metadata["full_name"] == "foo"

    def test_from_import_with_module(self, extractor: EntityExtractor) -> None:
        tree = _parse("from os.path import join, exists")
        node = tree.body[0]
        assert isinstance(node, ast.ImportFrom)
        entities = extractor._create_import_from_entities(node, FILE_PATH)
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "join" in names
        assert "exists" in names
        for e in entities:
            assert e.metadata["module"] == "os.path"
            assert e.metadata["is_external"] is True


# ---------------------------------------------------------------------------
# _create_variable_entities
# ---------------------------------------------------------------------------


class TestCreateVariableEntities:
    """Tests for _create_variable_entities."""

    def test_simple_assign(self, extractor: EntityExtractor) -> None:
        tree = _parse("x = 10")
        node = tree.body[0]
        assert isinstance(node, ast.Assign)
        entities = extractor._create_variable_entities(node, FILE_PATH)
        assert len(entities) == 1
        assert entities[0].name == "x"
        assert entities[0].metadata["value_type"] == "int"
        assert entities[0].metadata["is_constant"] is False

    def test_constant_naming_convention(self, extractor: EntityExtractor) -> None:
        tree = _parse("MAX_RETRIES = 3")
        node = tree.body[0]
        assert isinstance(node, ast.Assign)
        entities = extractor._create_variable_entities(node, FILE_PATH)
        assert entities[0].metadata["is_constant"] is True

    def test_tuple_target_skipped(self, extractor: EntityExtractor) -> None:
        # Tuple unpacking targets are not ast.Name so they're skipped
        tree = _parse("a, b = 1, 2")
        node = tree.body[0]
        assert isinstance(node, ast.Assign)
        entities = extractor._create_variable_entities(node, FILE_PATH)
        # Tuple target is not ast.Name, so no entities
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# _create_annotated_variable_entities
# ---------------------------------------------------------------------------


class TestCreateAnnotatedVariableEntities:
    """Tests for _create_annotated_variable_entities."""

    def test_annotated_with_value(self, extractor: EntityExtractor) -> None:
        tree = _parse("name: str = 'hello'")
        node = tree.body[0]
        assert isinstance(node, ast.AnnAssign)
        entities = extractor._create_annotated_variable_entities(node, FILE_PATH)
        assert len(entities) == 1
        e = entities[0]
        assert e.metadata["annotation"] == "str"
        assert e.metadata["is_annotated"] is True
        assert e.metadata["has_value"] is True

    def test_annotated_without_value(self, extractor: EntityExtractor) -> None:
        tree = _parse("count: int")
        node = tree.body[0]
        assert isinstance(node, ast.AnnAssign)
        entities = extractor._create_annotated_variable_entities(node, FILE_PATH)
        assert len(entities) == 1
        e = entities[0]
        assert e.metadata["has_value"] is False
        assert e.metadata["value_type"] is None

    def test_annotated_non_name_target_skipped(self, extractor: EntityExtractor) -> None:
        # self.x: int = 0  => target is ast.Attribute, not ast.Name
        tree = _parse("class C:\n    def __init__(self):\n        self.x: int = 0\n")
        # Walk to find the AnnAssign node
        ann_assigns = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)]
        assert len(ann_assigns) == 1
        entities = extractor._create_annotated_variable_entities(ann_assigns[0], FILE_PATH)
        # self.x target is Attribute, not Name, so no entities
        assert len(entities) == 0

    def test_annotated_constant(self, extractor: EntityExtractor) -> None:
        tree = _parse("TIMEOUT: int = 30")
        node = tree.body[0]
        assert isinstance(node, ast.AnnAssign)
        entities = extractor._create_annotated_variable_entities(node, FILE_PATH)
        assert entities[0].metadata["is_constant"] is True


# ---------------------------------------------------------------------------
# End-to-end scenarios
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests combining multiple entity types."""

    async def test_full_module_analysis(self, extractor: EntityExtractor) -> None:
        code = (
            '"""Module doc."""\n'
            "import os\n"
            "from typing import List\n"
            "\n"
            "MAX_SIZE: int = 100\n"
            "\n"
            "class MyService:\n"
            '    """Service class."""\n'
            "    name = 'default'\n"
            "\n"
            "    def __init__(self, config: dict) -> None:\n"
            "        self.config = config\n"
            "\n"
            "    async def process(self, data: List[str]) -> dict:\n"
            "        return {}\n"
            "\n"
            "def helper(x: int) -> int:\n"
            "    return x + 1\n"
        )
        tree = _parse(code)
        lines = _lines(code)

        entities = await extractor.extract_python_entities(tree, FILE_PATH, lines)
        imports = await extractor.extract_import_entities(tree, FILE_PATH)

        # Check entity counts
        modules = [e for e in entities if e.type == EntityType.MODULE]
        classes = [e for e in entities if e.type == EntityType.CLASS]
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        variables = [e for e in entities if e.type == EntityType.VARIABLE]

        assert len(modules) == 1
        assert len(classes) == 1
        assert len(functions) >= 3  # __init__, process, helper
        assert len(variables) >= 1  # MAX_SIZE + class var name

        # Check imports separately
        assert len(imports) >= 2  # os and List

    async def test_function_no_return_no_params(self, extractor: EntityExtractor) -> None:
        code = "def noop():\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_python_entities(tree, FILE_PATH)
        fn = [e for e in entities if e.type == EntityType.FUNCTION][0]
        assert fn.metadata["return_annotation"] is None
        assert fn.metadata["parameters"] == []
        assert fn.metadata["is_method"] is False
