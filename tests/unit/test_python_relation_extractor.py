"""Unit tests for processing.python.python_relation_extractor.PythonRelationExtractor."""

from __future__ import annotations

import ast

import pytest

from processing.python.python_relation_extractor import PythonRelationExtractor

FILE_PATH = "/src/example.py"
MODULE_NAME = "example"


@pytest.fixture
def extractor():
    """Return a fresh PythonRelationExtractor instance."""
    return PythonRelationExtractor()


def _parse(code: str) -> ast.Module:
    """Helper: parse code string into an AST Module."""
    return ast.parse(code)


def _rels_of_type(rels: list[dict], rel_type: str) -> list[dict]:
    """Filter relationships by type."""
    return [r for r in rels if r["type"] == rel_type]


# ---------------------------------------------------------------------------
# extract_all_relations
# ---------------------------------------------------------------------------


class TestExtractAllRelations:
    """Tests for the top-level extract_all_relations method."""

    async def test_returns_list(self, extractor):
        tree = _parse("x = 1\n")
        result = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)
        assert isinstance(result, list)

    async def test_empty_ast_returns_empty(self, extractor):
        tree = _parse("")
        result = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)
        assert result == []

    async def test_none_ast_returns_empty(self, extractor):
        result = await extractor.extract_all_relations(None, FILE_PATH, MODULE_NAME)
        assert result == []

    async def test_combines_all_relation_types(self, extractor):
        code = (
            "import os\n\nclass Child(Base):\n    def work(self):\n        os.path.join('a', 'b')\n"
        )
        tree = _parse(code)
        result = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        types_found = {r["type"] for r in result}
        assert "INHERITS" in types_found
        assert "CONTAINS" in types_found
        assert "IMPORTS" in types_found
        assert "CALLS" in types_found


# ---------------------------------------------------------------------------
# _extract_basic_relationships: INHERITS
# ---------------------------------------------------------------------------


class TestInheritance:
    """Tests for INHERITS relationship extraction."""

    async def test_simple_inheritance(self, extractor):
        tree = _parse("class Child(Base):\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) == 1
        assert inherits[0]["source"] == f"{FILE_PATH}::Child"
        assert inherits[0]["target"] == "Base"

    async def test_multiple_bases(self, extractor):
        tree = _parse("class Multi(Alpha, Beta):\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) == 2
        targets = {r["target"] for r in inherits}
        assert targets == {"Alpha", "Beta"}

    async def test_qualified_inheritance(self, extractor):
        tree = _parse("class Child(module.BaseClass):\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) == 1
        assert inherits[0]["target"] == "module.BaseClass"

    async def test_deeply_qualified_inheritance(self, extractor):
        tree = _parse("class Child(pkg.sub.Base):\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) == 1
        assert inherits[0]["target"] == "pkg.sub.Base"

    async def test_no_base_no_inherits(self, extractor):
        tree = _parse("class Standalone:\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) == 0

    async def test_inherits_has_tentative_flag(self, extractor):
        tree = _parse("class Child(Base):\n    pass\n")
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        inherits = _rels_of_type(rels, "INHERITS")
        assert inherits[0].get("tentative") is True


# ---------------------------------------------------------------------------
# _extract_basic_relationships: CONTAINS
# ---------------------------------------------------------------------------


class TestContains:
    """Tests for CONTAINS relationship extraction (class -> method)."""

    async def test_class_contains_method(self, extractor):
        code = "class Svc:\n    def run(self):\n        pass\n"
        tree = _parse(code)
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        contains = _rels_of_type(rels, "CONTAINS")
        assert len(contains) == 1
        assert contains[0]["source"] == f"{FILE_PATH}::Svc"
        assert contains[0]["target"] == f"{FILE_PATH}::Svc.run"

    async def test_class_contains_multiple_methods(self, extractor):
        code = "class Svc:\n    def start(self):\n        pass\n    def stop(self):\n        pass\n"
        tree = _parse(code)
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        contains = _rels_of_type(rels, "CONTAINS")
        assert len(contains) == 2
        targets = {r["target"] for r in contains}
        assert f"{FILE_PATH}::Svc.start" in targets
        assert f"{FILE_PATH}::Svc.stop" in targets


# ---------------------------------------------------------------------------
# _extract_basic_relationships: MODULE_CONTAINS
# ---------------------------------------------------------------------------


class TestModuleContains:
    """Tests for MODULE_CONTAINS relationship extraction."""

    async def test_standalone_function_module_contains(self, extractor):
        code = "def helper():\n    pass\n"
        tree = _parse(code)
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        module_contains = _rels_of_type(rels, "MODULE_CONTAINS")
        assert len(module_contains) == 1
        assert module_contains[0]["source"] == FILE_PATH
        assert module_contains[0]["target"] == f"{FILE_PATH}::helper"

    async def test_method_not_in_module_contains(self, extractor):
        code = "class Svc:\n    def method(self):\n        pass\n"
        tree = _parse(code)
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        module_contains = _rels_of_type(rels, "MODULE_CONTAINS")
        # Methods inside a class should NOT appear as MODULE_CONTAINS
        targets = {r["target"] for r in module_contains}
        assert f"{FILE_PATH}::method" not in targets

    async def test_multiple_standalone_functions(self, extractor):
        code = "def a():\n    pass\n\ndef b():\n    pass\n"
        tree = _parse(code)
        rels = extractor._extract_basic_relationships(tree, FILE_PATH)

        module_contains = _rels_of_type(rels, "MODULE_CONTAINS")
        assert len(module_contains) == 2
        targets = {r["target"] for r in module_contains}
        assert f"{FILE_PATH}::a" in targets
        assert f"{FILE_PATH}::b" in targets


# ---------------------------------------------------------------------------
# _extract_function_calls: CALLS
# ---------------------------------------------------------------------------


class TestFunctionCalls:
    """Tests for CALLS relationship extraction."""

    async def test_simple_function_call(self, extractor):
        code = "def caller():\n    helper()\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        assert len(calls) >= 1
        assert any(r["source"] == f"{FILE_PATH}::caller" and r["target"] == "helper" for r in calls)

    async def test_method_calling_function(self, extractor):
        code = "class Svc:\n    def run(self):\n        do_work()\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        assert any(
            r["source"] == f"{FILE_PATH}::Svc.run" and r["target"] == "do_work" for r in calls
        )

    async def test_method_calling_attribute(self, extractor):
        code = "class Svc:\n    def run(self):\n        self.helper.process()\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        assert any(
            r["source"] == f"{FILE_PATH}::Svc.run" and "process" in r["target"] for r in calls
        )

    async def test_call_outside_function_not_recorded(self, extractor):
        """Calls at module level (not inside a function) should not be recorded."""
        code = "print('hello')\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        # Module-level calls have no enclosing function, so current_function is None
        # and the visitor skips them
        assert len(calls) == 0

    async def test_calls_has_tentative_flag(self, extractor):
        code = "def f():\n    g()\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        assert len(calls) >= 1
        assert calls[0].get("tentative") is True

    async def test_nested_class_method_call(self, extractor):
        code = "class Outer:\n    def method(self):\n        result = some_func(1, 2)\n"
        tree = _parse(code)
        rels = extractor._extract_function_calls(tree, FILE_PATH)

        calls = _rels_of_type(rels, "CALLS")
        matching = [r for r in calls if r["target"] == "some_func"]
        assert len(matching) == 1
        assert matching[0]["source"] == f"{FILE_PATH}::Outer.method"


# ---------------------------------------------------------------------------
# _extract_imports: IMPORTS
# ---------------------------------------------------------------------------


class TestImports:
    """Tests for IMPORTS relationship extraction."""

    async def test_import_statement(self, extractor):
        code = "import os\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert imports[0]["source"] == FILE_PATH
        assert imports[0]["target"] == "os"
        assert imports[0]["alias"] == "os"

    async def test_import_with_alias(self, extractor):
        code = "import numpy as np\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert imports[0]["target"] == "numpy"
        assert imports[0]["alias"] == "np"

    async def test_multiple_imports(self, extractor):
        code = "import os\nimport sys\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 2
        targets = {r["target"] for r in imports}
        assert targets == {"os", "sys"}

    async def test_from_import(self, extractor):
        code = "from os.path import join\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert "join" in imports[0]["target"]
        assert imports[0]["alias"] == "join"

    async def test_from_import_with_alias(self, extractor):
        code = "from os.path import join as pjoin\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert imports[0]["alias"] == "pjoin"

    async def test_from_import_multiple_names(self, extractor):
        code = "from typing import Any, Optional, List\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 3
        aliases = {r["alias"] for r in imports}
        assert aliases == {"Any", "Optional", "List"}

    async def test_bare_relative_import_skipped(self, extractor):
        """``from . import sibling`` has node.module=None, so the extractor skips it."""
        code = "from . import sibling\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        # The extractor guards on `if node.module:` so bare relative imports are skipped
        assert len(imports) == 0

    async def test_relative_import_with_module(self, extractor):
        """``from ..parent import helper`` has node.module='parent', so it is processed."""
        code = "from ..parent import helper\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert imports[0]["level"] == 2

    async def test_dotted_relative_import(self, extractor):
        """``from .utils import tool`` has node.module='utils', level=1."""
        code = "from .utils import tool\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        assert imports[0]["level"] == 1
        assert "tool" in imports[0]["target"]

    async def test_star_import(self, extractor):
        code = "from os import *\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 1
        # Star imports: target is the module path (module_path when name=="*")
        assert imports[0]["target"] == "os"
        # alias for star import is "*" (name.asname is None, so alias=name.name="*")
        assert imports[0]["alias"] == "*"
        assert imports[0]["level"] == 0

    async def test_no_imports_returns_empty(self, extractor):
        code = "x = 1\n"
        tree = _parse(code)
        rels = extractor._extract_imports(tree, FILE_PATH)

        imports = _rels_of_type(rels, "IMPORTS")
        assert len(imports) == 0


# ---------------------------------------------------------------------------
# Integration with sample_python_code fixture
# ---------------------------------------------------------------------------


class TestWithSampleCode:
    """Tests using the shared sample_python_code fixture."""

    async def test_inherits_detected(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        inherits = _rels_of_type(rels, "INHERITS")
        assert len(inherits) >= 1
        # ChildClass inherits BaseClass
        child_inherits = [r for r in inherits if "ChildClass" in r["source"]]
        assert len(child_inherits) == 1
        assert child_inherits[0]["target"] == "BaseClass"

    async def test_contains_detected(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        contains = _rels_of_type(rels, "CONTAINS")
        # ChildClass contains __init__, do_work, and async_method
        child_contains = [r for r in contains if "ChildClass" in r["source"]]
        assert len(child_contains) >= 3  # __init__, do_work, async_method

    async def test_module_contains_standalone_functions(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        module_contains = _rels_of_type(rels, "MODULE_CONTAINS")
        targets = {r["target"] for r in module_contains}
        assert f"{FILE_PATH}::standalone_function" in targets
        # AsyncFunctionDef is now handled
        assert f"{FILE_PATH}::async_standalone" in targets

    async def test_imports_detected(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        imports = _rels_of_type(rels, "IMPORTS")
        # import os, import sys = 2
        # from pathlib import Path = 1
        # from typing import Any, Optional = 2
        # from collections.abc import Callable = 1
        # from ..parent import helper = 1
        # from . import sibling_module is SKIPPED (node.module is None)
        # Total = 7
        assert len(imports) >= 6

    async def test_async_method_contains_detected(self, extractor):
        """AsyncFunctionDef methods should produce CONTAINS relationships."""
        code = "class Svc:\n    async def handler(self):\n        await self.process()\n"
        tree = _parse(code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        contains = _rels_of_type(rels, "CONTAINS")
        assert len(contains) == 1
        assert contains[0]["source"] == f"{FILE_PATH}::Svc"
        assert contains[0]["target"] == f"{FILE_PATH}::Svc.handler"

    async def test_async_standalone_module_contains(self, extractor):
        """Async standalone functions should produce MODULE_CONTAINS relationships."""
        code = "async def fetch():\n    pass\n"
        tree = _parse(code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        module_contains = _rels_of_type(rels, "MODULE_CONTAINS")
        assert len(module_contains) == 1
        assert module_contains[0]["target"] == f"{FILE_PATH}::fetch"

    async def test_async_method_calls_tracked(self, extractor):
        """Calls inside async methods should be tracked."""
        code = "class Svc:\n    async def run(self):\n        await self.db.query()\n"
        tree = _parse(code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        calls = _rels_of_type(rels, "CALLS")
        assert len(calls) >= 1
        assert any(r["source"] == f"{FILE_PATH}::Svc.run" and "query" in r["target"] for r in calls)

    async def test_calls_detected(self, extractor, sample_python_code):
        tree = _parse(sample_python_code)
        rels = await extractor.extract_all_relations(tree, FILE_PATH, MODULE_NAME)

        calls = _rels_of_type(rels, "CALLS")
        # ChildClass.__init__ calls Helper() -> 1
        # ChildClass.do_work calls os.path.join, str, self.helper.process -> 3
        # ChildClass.async_method calls self.helper.async_process -> 1
        # AnotherClass.run calls ChildClass(), child.do_work -> 2
        assert len(calls) >= 4  # async_method calls are now tracked
