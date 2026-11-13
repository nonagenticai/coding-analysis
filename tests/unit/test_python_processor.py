"""Unit tests for processing.python_processor.CorePythonProcessor.

Tests the real CorePythonProcessor using direct imports.
The module depends on app.core.file_service which is available at src/app/core/.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from app.core.file_service import FileService
from processing.python_processor import CorePythonProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def file_service():
    """Return a mock FileService with basic stubs."""
    svc = MagicMock(spec=FileService)
    svc.read_file = MagicMock(return_value="x = 1\n")
    svc.file_exists = MagicMock(return_value=True)
    svc.list_files = MagicMock(return_value=[])
    return svc


@pytest.fixture
def processor(file_service):
    """Return a CorePythonProcessor with a mock FileService."""
    return CorePythonProcessor(file_service)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for CorePythonProcessor construction."""

    def test_none_file_service_raises_value_error(self):
        with pytest.raises(ValueError, match="FileService is required"):
            CorePythonProcessor(None)  # type: ignore[arg-type]

    def test_valid_file_service_accepted(self, file_service):
        proc = CorePythonProcessor(file_service)
        assert proc.file_service is file_service
        assert proc.language == "python"

    def test_processed_files_starts_empty(self, processor):
        assert processor.processed_files == set()

    def test_components_initialized(self, processor):
        from processing.python.python_entity_extractor import PythonEntityExtractor
        from processing.python.python_parser import PythonParser
        from processing.python.python_relation_extractor import PythonRelationExtractor

        assert isinstance(processor.parser, PythonParser)
        assert isinstance(processor.entity_extractor, PythonEntityExtractor)
        assert isinstance(processor.relation_extractor, PythonRelationExtractor)


# ---------------------------------------------------------------------------
# get_supported_extensions
# ---------------------------------------------------------------------------


class TestGetSupportedExtensions:
    """Tests for get_supported_extensions."""

    def test_returns_py_and_pyi(self, processor):
        extensions = processor.get_supported_extensions()
        assert ".py" in extensions
        assert ".pyi" in extensions

    def test_returns_list(self, processor):
        assert isinstance(processor.get_supported_extensions(), list)

    def test_returns_exactly_two(self, processor):
        assert len(processor.get_supported_extensions()) == 2


# ---------------------------------------------------------------------------
# process_document
# ---------------------------------------------------------------------------


class TestProcessDocument:
    """Tests for process_document."""

    @pytest.mark.asyncio
    async def test_valid_py_file_returns_entities_and_relationships(self, processor, tmp_path):
        code = "class Foo:\n    def bar(self):\n        pass\n\ndef standalone():\n    pass\n"
        py_file = tmp_path / "sample.py"
        py_file.write_text(code)

        result = await processor.process_document(str(py_file), code)

        assert result["file_path"] == str(py_file)
        assert result["language"] == "python"
        assert isinstance(result["entities"], list)
        assert isinstance(result["relationships"], list)
        assert len(result["entities"]) > 0

    @pytest.mark.asyncio
    async def test_non_py_file_returns_empty(self, processor, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("const x = 1;")

        result = await processor.process_document(str(js_file))

        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_empty(self, processor):
        result = await processor.process_document("/nonexistent/path/file.py")

        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_syntax_error_returns_empty(self, processor, tmp_path):
        py_file = tmp_path / "broken.py"
        py_file.write_text("def broken(\n")

        result = await processor.process_document(str(py_file), "def broken(\n")

        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_file_added_to_processed_files(self, processor, tmp_path):
        py_file = tmp_path / "tracked.py"
        py_file.write_text("x = 1\n")

        await processor.process_document(str(py_file), "x = 1\n")

        assert str(py_file) in processor.processed_files

    @pytest.mark.asyncio
    async def test_reads_file_when_code_is_none(self, processor, tmp_path):
        code = "y = 2\n"
        py_file = tmp_path / "read_me.py"
        py_file.write_text(code)

        processor.file_service.read_file.return_value = code
        await processor.process_document(str(py_file))

        processor.file_service.read_file.assert_called_once_with(str(py_file))

    @pytest.mark.asyncio
    async def test_extracts_class_entities(self, processor, tmp_path):
        code = "class MyClass:\n    '''Doc.'''\n    pass\n"
        py_file = tmp_path / "cls.py"
        py_file.write_text(code)

        result = await processor.process_document(str(py_file), code)

        class_entities = [e for e in result["entities"] if e["type"] == "class"]
        assert len(class_entities) == 1
        assert class_entities[0]["name"] == "MyClass"

    @pytest.mark.asyncio
    async def test_extracts_function_entities(self, processor, tmp_path):
        code = "def helper(x):\n    return x + 1\n"
        py_file = tmp_path / "fn.py"
        py_file.write_text(code)

        result = await processor.process_document(str(py_file), code)

        fn_entities = [e for e in result["entities"] if e["type"] == "function"]
        assert len(fn_entities) == 1
        assert fn_entities[0]["name"] == "helper"

    @pytest.mark.asyncio
    async def test_extracts_inheritance_relationships(self, processor, tmp_path):
        code = "class Parent:\n    pass\n\nclass Child(Parent):\n    pass\n"
        py_file = tmp_path / "inh.py"
        py_file.write_text(code)

        result = await processor.process_document(str(py_file), code)

        inherits = [r for r in result["relationships"] if r["type"] == "INHERITS"]
        assert len(inherits) == 1
        assert inherits[0]["target"] == "Parent"


# ---------------------------------------------------------------------------
# _extract_basic_relationships
# ---------------------------------------------------------------------------


class TestExtractBasicRelationships:
    """Tests for the _extract_basic_relationships private method."""

    @pytest.mark.asyncio
    async def test_inheritance_detection(self, processor):
        code = "class Child(Base):\n    pass\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        inheritance = [r for r in rels if r["type"] == "inheritance"]
        assert len(inheritance) == 1
        assert inheritance[0]["source"] == "Child"
        assert inheritance[0]["target"] == "Base"

    @pytest.mark.asyncio
    async def test_qualified_inheritance(self, processor):
        code = "class Child(mod.Base):\n    pass\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        inheritance = [r for r in rels if r["type"] == "inheritance"]
        assert len(inheritance) == 1
        assert inheritance[0]["target"] == "mod.Base"

    @pytest.mark.asyncio
    async def test_composition_class_level_assign(self, processor):
        code = "class Svc:\n    helper = Helper()\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        composition = [r for r in rels if r["type"] == "composition"]
        assert len(composition) == 1
        assert composition[0]["source"] == "Svc"
        assert composition[0]["target"] == "Helper"
        assert composition[0]["properties"]["attribute"] == "helper"

    @pytest.mark.asyncio
    async def test_composition_annotated_assign(self, processor):
        code = "class Svc:\n    helper: Helper = Helper()\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        composition = [r for r in rels if r["type"] == "composition"]
        assert len(composition) == 1
        assert composition[0]["target"] == "Helper"

    @pytest.mark.asyncio
    async def test_uses_attribute_access(self, processor):
        code = "class Svc:\n    def run(self):\n        os.path.join('a', 'b')\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        uses = [r for r in rels if r["type"] == "uses"]
        assert len(uses) >= 1
        targets = {r["target"] for r in uses}
        assert "os" in targets or "os.path" in targets

    @pytest.mark.asyncio
    async def test_self_references_excluded_from_uses(self, processor):
        code = "class Svc:\n    def run(self):\n        self.x = 1\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        uses = [r for r in rels if r["type"] == "uses"]
        targets = {r["target"] for r in uses}
        assert "self" not in targets

    @pytest.mark.asyncio
    async def test_empty_class_no_relationships(self, processor):
        code = "class Empty:\n    pass\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_no_classes_returns_empty(self, processor):
        code = "x = 1\ny = 2\n"
        tree = ast.parse(code)
        rels = await processor._extract_basic_relationships(tree, "/test.py")

        assert rels == []


# ---------------------------------------------------------------------------
# _extract_function_calls
# ---------------------------------------------------------------------------


class TestExtractFunctionCalls:
    """Tests for the _extract_function_calls private method."""

    @pytest.mark.asyncio
    async def test_simple_call(self, processor):
        code = "def caller():\n    helper()\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        assert len(calls) >= 1
        assert any(c["callee"] == "helper" for c in calls)

    @pytest.mark.asyncio
    async def test_caller_resolved(self, processor):
        code = "def caller():\n    helper()\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        matching = [c for c in calls if c["callee"] == "helper"]
        assert matching[0]["caller"] == "caller"

    @pytest.mark.asyncio
    async def test_method_call_with_class_context(self, processor):
        code = "class Svc:\n    def run(self):\n        do_work()\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        matching = [c for c in calls if c["callee"] == "do_work"]
        assert len(matching) == 1
        assert matching[0]["caller"] == "Svc.run"

    @pytest.mark.asyncio
    async def test_attribute_call_resolved(self, processor):
        code = "def f():\n    os.path.join('a', 'b')\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        matching = [c for c in calls if "join" in c["callee"]]
        assert len(matching) == 1
        assert matching[0]["callee"] == "os.path.join"

    @pytest.mark.asyncio
    async def test_module_level_call(self, processor):
        code = "print('hello')\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        # Module-level calls still get recorded with caller="<module>"
        matching = [c for c in calls if c["callee"] == "print"]
        assert len(matching) == 1
        assert matching[0]["caller"] == "<module>"

    @pytest.mark.asyncio
    async def test_call_arg_count(self, processor):
        code = "def f():\n    func(1, 2, key='val')\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        matching = [c for c in calls if c["callee"] == "func"]
        assert matching[0]["properties"]["arg_count"] == 3

    @pytest.mark.asyncio
    async def test_call_location_recorded(self, processor):
        code = "def f():\n    helper()\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        matching = [c for c in calls if c["callee"] == "helper"]
        assert "location" in matching[0]
        assert matching[0]["location"]["file"] == "/test.py"
        assert matching[0]["location"]["line"] == 2

    @pytest.mark.asyncio
    async def test_no_calls_returns_empty(self, processor):
        code = "x = 1\n"
        tree = ast.parse(code)
        calls = await processor._extract_function_calls(tree, "/test.py")

        assert calls == []


# ---------------------------------------------------------------------------
# _extract_imports
# ---------------------------------------------------------------------------


class TestExtractImports:
    """Tests for the _extract_imports private method."""

    @pytest.mark.asyncio
    async def test_import_statement(self, processor):
        code = "import os\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 1
        assert imports[0]["source"] == "/test.py"
        assert imports[0]["target"] == "os"
        assert imports[0]["type"] == "import"
        assert imports[0]["properties"]["import_type"] == "absolute"

    @pytest.mark.asyncio
    async def test_import_with_alias(self, processor):
        code = "import numpy as np\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert imports[0]["properties"]["alias"] == "np"

    @pytest.mark.asyncio
    async def test_from_import(self, processor):
        code = "from os.path import join\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 1
        assert imports[0]["type"] == "import_from"
        assert "join" in imports[0]["target"]
        assert imports[0]["properties"]["module"] == "os.path"
        assert imports[0]["properties"]["import_type"] == "absolute"

    @pytest.mark.asyncio
    async def test_from_import_multiple_names(self, processor):
        code = "from typing import Any, Optional\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 2
        targets = {i["target"] for i in imports}
        assert any("Any" in t for t in targets)
        assert any("Optional" in t for t in targets)

    @pytest.mark.asyncio
    async def test_relative_import_dot(self, processor):
        code = "from . import sibling\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 1
        assert imports[0]["properties"]["import_type"] == "relative"
        assert imports[0]["properties"]["level"] == 1

    @pytest.mark.asyncio
    async def test_relative_import_dotdot(self, processor):
        code = "from ..parent import helper\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 1
        assert imports[0]["properties"]["import_type"] == "relative"
        assert imports[0]["properties"]["level"] == 2

    @pytest.mark.asyncio
    async def test_star_import(self, processor):
        code = "from os import *\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 1
        assert "*" in imports[0]["target"]

    @pytest.mark.asyncio
    async def test_no_imports_returns_empty(self, processor):
        code = "x = 1\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert imports == []

    @pytest.mark.asyncio
    async def test_import_line_recorded(self, processor):
        code = "import os\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert imports[0]["properties"]["line"] == 1

    @pytest.mark.asyncio
    async def test_multiple_import_types(self, processor):
        code = "import os\nfrom sys import argv\n"
        tree = ast.parse(code)
        imports = await processor._extract_imports(tree, "/test.py")

        assert len(imports) == 2
        types = {i["type"] for i in imports}
        assert "import" in types
        assert "import_from" in types


# ---------------------------------------------------------------------------
# analyze_directory
# ---------------------------------------------------------------------------


class TestAnalyzeDirectory:
    """Tests for analyze_directory to cover uncovered lines."""

    @pytest.mark.asyncio
    async def test_invalid_directory_returns_empty(self, processor):
        result = await processor.analyze_directory("/nonexistent/dir/abc123")
        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_valid_directory_with_python_files(self, processor, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("def greet():\n    pass\n")

        result = await processor.analyze_directory(str(tmp_path))
        assert "entities" in result
        assert "directory" in result
        assert result["directory"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_skips_already_processed(self, processor, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("x = 1\n")
        # Pre-mark as processed
        processor.processed_files.add(str(py_file))

        result = await processor.analyze_directory(str(tmp_path))
        # Should produce no entities since file is skipped
        assert result["entities"] == []

    @pytest.mark.asyncio
    async def test_non_recursive_skips_subdirs(self, processor, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "mod.py").write_text("x = 1\n")
        (tmp_path / "top.py").write_text("y = 2\n")

        result = await processor.analyze_directory(str(tmp_path), recursive=False)
        # Only top.py should be processed
        assert result["directory"] == str(tmp_path)


# ---------------------------------------------------------------------------
# _get_module_name
# ---------------------------------------------------------------------------


class TestGetModuleName:
    """Tests for _get_module_name helper."""

    def test_simple_path(self, processor):
        name = processor._get_module_name("/project/src/utils.py")
        assert name == "utils"

    def test_init_file(self, processor):
        name = processor._get_module_name("/project/src/package/__init__.py")
        assert name == "__init__"
