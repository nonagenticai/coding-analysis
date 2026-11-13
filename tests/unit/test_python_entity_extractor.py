"""Unit tests for processing.python.python_entity_extractor.PythonEntityExtractor."""

from __future__ import annotations

import ast

import pytest

from processing.python.python_entity_extractor import PythonEntityExtractor

FILE_PATH = "/src/example.py"


@pytest.fixture
def extractor():
    """Return a fresh PythonEntityExtractor instance."""
    return PythonEntityExtractor()


def _parse(code: str) -> ast.Module:
    """Helper: parse code string into an AST Module."""
    return ast.parse(code)


# ---------------------------------------------------------------------------
# Class entities
# ---------------------------------------------------------------------------


class TestClassExtraction:
    """Tests for extracting class entities."""

    async def test_simple_class(self, extractor):
        tree = _parse("class Foo:\n    pass\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert len(classes) == 1
        cls = classes[0]
        assert cls["name"] == "Foo"
        assert cls["id"] == f"{FILE_PATH}::Foo"
        assert cls["file_path"] == FILE_PATH
        assert cls["start_line"] == 1

    async def test_class_with_bases(self, extractor):
        tree = _parse("class Child(Base):\n    pass\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert len(classes) == 1
        assert "Base" in classes[0]["properties"]["bases"]

    async def test_class_with_multiple_bases(self, extractor):
        tree = _parse("class Multi(Alpha, Beta):\n    pass\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert len(classes) == 1
        bases = classes[0]["properties"]["bases"]
        assert "Alpha" in bases
        assert "Beta" in bases

    async def test_class_with_docstring(self, extractor):
        code = 'class Documented:\n    """This is a docstring."""\n    pass\n'
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["docstring"] == "This is a docstring."

    async def test_class_without_docstring(self, extractor):
        tree = _parse("class NoDocs:\n    x = 1\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert classes[0]["docstring"] is None

    async def test_class_with_decorators(self, extractor):
        code = "@dataclass\nclass Config:\n    x: int = 0\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        assert len(classes) == 1
        assert "dataclass" in classes[0]["properties"]["decorators"]

    async def test_multiple_classes(self, extractor):
        code = "class A:\n    pass\n\nclass B:\n    pass\n\nclass C:\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        classes = [e for e in entities if e["type"] == "class"]
        names = {c["name"] for c in classes}
        assert names == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Function entities
# ---------------------------------------------------------------------------


class TestFunctionExtraction:
    """Tests for extracting standalone function entities."""

    async def test_standalone_function(self, extractor):
        tree = _parse("def greet(name):\n    return f'Hello {name}'\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        functions = [e for e in entities if e["type"] == "function"]
        assert len(functions) == 1
        fn = functions[0]
        assert fn["name"] == "greet"
        assert fn["id"] == f"{FILE_PATH}::greet"
        assert fn["type"] == "function"
        assert "name" in fn["properties"]["parameters"]

    async def test_function_with_docstring(self, extractor):
        code = 'def work():\n    """Does work."""\n    pass\n'
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        functions = [e for e in entities if e["type"] == "function"]
        assert functions[0]["docstring"] == "Does work."

    async def test_function_with_multiple_params(self, extractor):
        tree = _parse("def add(a, b, c=0):\n    return a + b + c\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        functions = [e for e in entities if e["type"] == "function"]
        params = functions[0]["properties"]["parameters"]
        assert "a" in params
        assert "b" in params
        assert "c" in params

    async def test_function_no_parent_class(self, extractor):
        tree = _parse("def standalone():\n    pass\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        functions = [e for e in entities if e["type"] == "function"]
        assert functions[0]["properties"]["parent_class"] is None


# ---------------------------------------------------------------------------
# Method entities
# ---------------------------------------------------------------------------


class TestMethodExtraction:
    """Tests for extracting method entities inside classes."""

    async def test_method_type_is_method(self, extractor):
        code = "class Svc:\n    def run(self):\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        assert len(methods) == 1
        assert methods[0]["name"] == "run"
        assert methods[0]["type"] == "method"

    async def test_method_id_includes_class(self, extractor):
        code = "class Svc:\n    def run(self):\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        assert methods[0]["id"] == f"{FILE_PATH}::Svc.run"

    async def test_method_parent_class_set(self, extractor):
        code = "class Svc:\n    def run(self):\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        assert methods[0]["properties"]["parent_class"] == "Svc"

    async def test_method_parameters_include_self(self, extractor):
        code = "class Svc:\n    def run(self, x, y):\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        params = methods[0]["properties"]["parameters"]
        assert "self" in params
        assert "x" in params
        assert "y" in params

    async def test_multiple_methods_in_class(self, extractor):
        code = "class Svc:\n    def start(self):\n        pass\n    def stop(self):\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        method_names = {m["name"] for m in methods}
        assert method_names == {"start", "stop"}

    async def test_method_with_decorator(self, extractor):
        code = "class Svc:\n    @staticmethod\n    def helper():\n        pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        assert "staticmethod" in methods[0]["properties"]["decorators"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    async def test_empty_ast_returns_empty_list(self, extractor):
        tree = _parse("")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")
        assert entities == []

    async def test_none_ast_returns_empty_list(self, extractor):
        entities = await extractor.extract_entities(None, FILE_PATH, "example")
        assert entities == []

    async def test_mixed_classes_and_functions(self, extractor):
        code = "class A:\n    def m(self):\n        pass\n\ndef standalone():\n    pass\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        types = {e["type"] for e in entities}
        assert "class" in types
        assert "method" in types
        assert "function" in types

    async def test_entity_has_required_keys(self, extractor):
        tree = _parse("class X:\n    pass\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        required_keys = {
            "id",
            "name",
            "type",
            "file_path",
            "start_line",
            "end_line",
            "docstring",
            "properties",
        }
        for entity in entities:
            assert required_keys.issubset(entity.keys()), (
                f"Entity {entity.get('name')} missing keys: {required_keys - entity.keys()}"
            )


# ---------------------------------------------------------------------------
# Integration with sample_python_code fixture
# ---------------------------------------------------------------------------


class TestWithSampleCode:
    """Tests using the shared sample_python_code fixture."""

    async def test_extracts_all_classes(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        class_names = {e["name"] for e in entities if e["type"] == "class"}
        assert "BaseClass" in class_names
        assert "ChildClass" in class_names
        assert "AnotherClass" in class_names

    async def test_extracts_standalone_functions(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        fn_names = {e["name"] for e in entities if e["type"] == "function"}
        assert "standalone_function" in fn_names
        assert "async_standalone" in fn_names  # AsyncFunctionDef is now handled

    async def test_extracts_methods(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        method_names = {e["name"] for e in entities if e["type"] == "method"}
        assert "__init__" in method_names
        assert "do_work" in method_names
        assert "async_method" in method_names  # AsyncFunctionDef is now handled
        assert "run" in method_names

    async def test_extracts_async_standalone_function(self, extractor):
        """AsyncFunctionDef at module level should be extracted as type 'function'."""
        tree = _parse("async def fetch_data():\n    await something()\n")
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        functions = [e for e in entities if e["type"] == "function"]
        assert len(functions) == 1
        assert functions[0]["name"] == "fetch_data"
        assert functions[0]["id"] == f"{FILE_PATH}::fetch_data"

    async def test_extracts_async_method(self, extractor):
        """AsyncFunctionDef inside a class should be extracted as type 'method'."""
        code = "class Svc:\n    async def run(self):\n        await self.process()\n"
        tree = _parse(code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        methods = [e for e in entities if e["type"] == "method"]
        assert len(methods) == 1
        assert methods[0]["name"] == "run"
        assert methods[0]["id"] == f"{FILE_PATH}::Svc.run"
        assert methods[0]["properties"]["parent_class"] == "Svc"

    async def test_childclass_inherits_baseclass(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        entities = await extractor.extract_entities(tree, FILE_PATH, "example")

        child = next(e for e in entities if e["name"] == "ChildClass")
        assert "BaseClass" in child["properties"]["bases"]
