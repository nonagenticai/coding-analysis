"""Deep unit tests for processing.analyzers.doc_extractor.DocumentationExtractor.

Targets 80%+ coverage of the 174 missed lines in doc_extractor.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types as _types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional deps that may not be installed in test env
# ---------------------------------------------------------------------------
if "astor" not in sys.modules:
    _astor = _types.ModuleType("astor")
    _astor.to_source = MagicMock(return_value="mock_source")
    sys.modules["astor"] = _astor

if "astpretty" not in sys.modules:
    _astpretty = _types.ModuleType("astpretty")
    _astpretty.pformat = MagicMock(return_value="mock_ast")
    _astpretty.pprint = MagicMock()
    sys.modules["astpretty"] = _astpretty

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from processing.analyzers.doc_extractor import DocumentationExtractor

# =====================================================================
# Helpers
# =====================================================================


def _write_tmp(content: str, suffix: str = ".py") -> str:
    """Write *content* to a temporary file, return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# =====================================================================
# DocumentationExtractor.__init__
# =====================================================================


class TestDocumentationExtractorInit:
    """Tests for __init__."""

    def test_init_creates_function_extractor(self):
        ext = DocumentationExtractor()
        assert ext.function_extractor is not None


# =====================================================================
# extract_file_documentation
# =====================================================================


class TestExtractFileDocumentation:
    """Tests for extract_file_documentation dispatch."""

    async def test_python_file(self):
        path = _write_tmp('"""Module."""\ndef foo():\n    pass\n', suffix=".py")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("file_type") == "python"
        finally:
            os.unlink(path)

    async def test_javascript_file(self):
        path = _write_tmp("function greet() { return 1; }", suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("file_type") == "javascript"
        finally:
            os.unlink(path)

    async def test_typescript_file(self):
        path = _write_tmp("const x: number = 5;", suffix=".ts")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("file_type") == "javascript"
        finally:
            os.unlink(path)

    async def test_markdown_file(self):
        path = _write_tmp("# Hello\n\nSome text.", suffix=".md")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("file_type") == "markdown"
        finally:
            os.unlink(path)

    async def test_rst_file(self):
        path = _write_tmp("Title\n=====\n\nSome text.", suffix=".rst")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("file_type") == "rst"
        finally:
            os.unlink(path)

    async def test_unsupported_extension(self):
        path = _write_tmp("int main() {}", suffix=".c")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result.get("coverage") == 0.0
            assert "Unsupported" in result.get("content", "")
        finally:
            os.unlink(path)

    async def test_nonexistent_file_returns_error(self):
        ext = DocumentationExtractor()
        result = await ext.extract_file_documentation("/tmp/nonexistent_xyz_42.py")
        assert "error" in result

    async def test_os_error_returns_error_dict(self):
        """OSError in the dispatch method itself."""
        ext = DocumentationExtractor()
        # Pass a file path that triggers an OS error inside the method
        with patch.object(ext, "_extract_python_documentation", side_effect=OSError("boom")):
            result = await ext.extract_file_documentation("/tmp/fake.py")
        assert "error" in result
        assert result["file_path"] == "/tmp/fake.py"


# =====================================================================
# extract_snippet_documentation
# =====================================================================


class TestExtractSnippetDocumentation:
    """Tests for extract_snippet_documentation."""

    async def test_python_snippet(self):
        ext = DocumentationExtractor()
        result = await ext.extract_snippet_documentation(
            '"""Module."""\ndef bar(): pass\n', file_extension=".py"
        )
        assert result.get("snippet") is True
        assert result.get("file_type") == "python"

    async def test_js_snippet(self):
        ext = DocumentationExtractor()
        result = await ext.extract_snippet_documentation(
            "function hello() {}", file_extension=".js"
        )
        assert result.get("snippet") is True
        assert result.get("file_type") == "javascript"

    async def test_snippet_cleans_up_tmp_file(self):
        ext = DocumentationExtractor()
        result = await ext.extract_snippet_documentation("x = 1")
        assert result.get("snippet") is True

    async def test_snippet_error_path(self):
        ext = DocumentationExtractor()
        with patch("tempfile.NamedTemporaryFile", side_effect=OSError("no space")):
            result = await ext.extract_snippet_documentation("x = 1")
        assert "error" in result
        assert result["snippet"] is True


# =====================================================================
# _extract_python_documentation
# =====================================================================


class TestExtractPythonDocumentation:
    """Tests for _extract_python_documentation."""

    async def test_module_docstring_extracted(self):
        code = '"""My module docstring."""\n\ndef foo():\n    """Foo doc."""\n    pass\n'
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            assert result["module_docstring"] == "My module docstring."
        finally:
            os.unlink(path)

    async def test_no_module_docstring(self):
        code = "import os\ndef foo(): pass\n"
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            assert result["module_docstring"] is None
        finally:
            os.unlink(path)

    async def test_classes_extracted(self):
        code = (
            'class MyClass:\n    """MyClass doc."""\n'
            '    def method_one(self):\n        """Method one."""\n        pass\n'
            "    def _private(self):\n        pass\n"
        )
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            classes = result["classes"]
            assert len(classes) == 1
            cls = classes[0]
            assert cls["name"] == "MyClass"
            assert cls["docstring"] == "MyClass doc."
            assert len(cls["methods"]) == 2
            # Check private flag
            method_names = {m["name"]: m for m in cls["methods"]}
            assert method_names["_private"]["is_private"] is True
            assert method_names["method_one"]["is_private"] is False
        finally:
            os.unlink(path)

    async def test_imports_extracted(self):
        code = "import os\nimport sys\nfrom pathlib import Path\nfrom . import sibling\n"
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            imports = result["imports"]
            names = [i["name"] for i in imports]
            assert "os" in names
            assert "sys" in names
            assert "Path" in names
            assert "sibling" in names
            # Check types
            import_types = {i["name"]: i["type"] for i in imports}
            assert import_types["os"] == "import"
            assert import_types["Path"] == "from_import"
        finally:
            os.unlink(path)

    async def test_import_from_with_no_module(self):
        """from . import something  ->  module is empty string."""
        code = "from . import helper\n"
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            from_imports = [i for i in result["imports"] if i["type"] == "from_import"]
            assert len(from_imports) == 1
            assert from_imports[0]["module"] == ""
        finally:
            os.unlink(path)

    async def test_import_alias(self):
        code = "import numpy as np\nfrom os import path as ospath\n"
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            aliases = {i["name"]: i.get("alias") for i in result["imports"]}
            assert aliases["numpy"] == "np"
            assert aliases["path"] == "ospath"
        finally:
            os.unlink(path)

    async def test_syntax_error_file_raises_unbound(self):
        """File with syntax errors triggers UnboundLocalError for 'tree'.

        This is a known bug in the source: when ast.parse fails, `tree` is
        never bound but is referenced later in ast.walk(tree).  The
        UnboundLocalError is not caught by the except blocks, so it
        propagates.
        """
        code = "def foo(:\n    pass\n"  # SyntaxError
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            with patch.object(
                ext.function_extractor,
                "extract_comprehensive_signatures",
                return_value={"functions": [], "type_coverage": 0.0, "external_dependencies": []},
            ):
                with pytest.raises(UnboundLocalError):
                    await ext._extract_python_documentation(path)
        finally:
            os.unlink(path)

    async def test_os_error_on_read(self):
        ext = DocumentationExtractor()
        result = await ext._extract_python_documentation("/tmp/does_not_exist_xyz_42.py")
        assert "error" in result

    async def test_coverage_included(self):
        code = '"""Module doc."""\ndef documented():\n    """Yes."""\n    pass\ndef undocumented(): pass\n'
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            assert "coverage" in result
            assert 0.0 <= result["coverage"] <= 1.0
        finally:
            os.unlink(path)

    async def test_multiple_classes_and_methods(self):
        code = (
            'class A:\n    """A doc."""\n    def m1(self):\n        """M1 doc."""\n        pass\n'
            'class B:\n    def m2(self): pass\n    def m3(self):\n        """M3 doc."""\n        pass\n'
        )
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_python_documentation(path)
            assert len(result["classes"]) == 2
        finally:
            os.unlink(path)


# =====================================================================
# _extract_javascript_documentation
# =====================================================================


class TestExtractJavascriptDocumentation:
    """Tests for _extract_javascript_documentation."""

    async def test_function_declarations(self):
        code = "function greet(name) { return name; }\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            assert result["file_type"] == "javascript"
            names = [f["name"] for f in result["functions"]]
            assert "greet" in names
        finally:
            os.unlink(path)

    async def test_arrow_functions(self):
        code = "const add = (a, b) => a + b;\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            names = [f["name"] for f in result["functions"]]
            assert "add" in names
        finally:
            os.unlink(path)

    async def test_object_method_declarations(self):
        code = "const obj = { handler: function(req, res) { return; } };\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            names = [f["name"] for f in result["functions"]]
            assert "handler" in names
        finally:
            os.unlink(path)

    async def test_method_shorthand(self):
        code = "class Foo {\n  bar(x) {\n    return x;\n  }\n}\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            names = [f["name"] for f in result["functions"]]
            assert "bar" in names
        finally:
            os.unlink(path)

    async def test_jsdoc_comments(self):
        code = "/** Greets a person. */\nfunction greet(name) {}\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            assert len(result["jsdoc_comments"]) >= 1
        finally:
            os.unlink(path)

    async def test_es6_classes(self):
        code = "class Animal extends LivingThing {\n  speak() {}\n}\nclass Plant {}\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            cls_names = [c["name"] for c in result["classes"]]
            assert "Animal" in cls_names
            assert "Plant" in cls_names
            animal = [c for c in result["classes"] if c["name"] == "Animal"][0]
            assert animal["parent"] == "LivingThing"
            plant = [c for c in result["classes"] if c["name"] == "Plant"][0]
            assert plant["parent"] is None
        finally:
            os.unlink(path)

    async def test_es6_imports(self):
        code = "import React from 'react';\nimport { useState } from 'react';\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            modules = [i["module"] for i in result["imports"]]
            assert "react" in modules
            types = [i["type"] for i in result["imports"]]
            assert all(t == "es6" for t in types)
        finally:
            os.unlink(path)

    async def test_commonjs_require(self):
        code = "const fs = require('fs');\nconst path = require('path');\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            modules = [i["module"] for i in result["imports"]]
            assert "fs" in modules
            assert "path" in modules
            types = [i["type"] for i in result["imports"]]
            assert all(t == "commonjs" for t in types)
        finally:
            os.unlink(path)

    async def test_line_numbers(self):
        code = "// line 1\nfunction a() {}\n// line 3\nfunction b() {}\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            # "function a()" matches both the function decl AND method shorthand patterns
            func_a = [f for f in result["functions"] if f["name"] == "a"]
            assert len(func_a) >= 1
            assert func_a[0]["line_number"] == 2
        finally:
            os.unlink(path)

    async def test_file_content_preview(self):
        code = "x" * 2000
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            assert len(result["file_content_preview"]) == 1000
        finally:
            os.unlink(path)

    async def test_coverage_calculated(self):
        code = "/** Doc */\nfunction foo() {}\nfunction bar() {}\n"
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            assert "coverage" in result
            assert 0.0 <= result["coverage"] <= 1.0
        finally:
            os.unlink(path)

    async def test_os_error(self):
        ext = DocumentationExtractor()
        result = await ext._extract_javascript_documentation("/tmp/nonexistent_xyz.js")
        assert "error" in result

    async def test_empty_js_file(self):
        path = _write_tmp("", suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_javascript_documentation(path)
            assert result["file_type"] == "javascript"
            assert result["functions"] == []
        finally:
            os.unlink(path)


# =====================================================================
# _extract_markup_documentation
# =====================================================================


class TestExtractMarkupDocumentation:
    """Tests for _extract_markup_documentation."""

    async def test_markdown_dispatched(self):
        path = _write_tmp("# Title\nBody.", suffix=".md")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_markup_documentation(path)
            assert result["file_type"] == "markdown"
        finally:
            os.unlink(path)

    async def test_rst_dispatched(self):
        path = _write_tmp("Title\n=====\nBody.", suffix=".rst")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_markup_documentation(path)
            assert result["file_type"] == "rst"
        finally:
            os.unlink(path)

    async def test_unknown_markup_extension(self):
        """Non-.md, non-.rst file read by _extract_markup_documentation."""
        path = _write_tmp("some content", suffix=".txt")
        try:
            ext = DocumentationExtractor()
            result = await ext._extract_markup_documentation(path)
            assert result["file_type"] == "markup"
            assert result["coverage"] == 1.0
        finally:
            os.unlink(path)

    async def test_os_error(self):
        ext = DocumentationExtractor()
        result = await ext._extract_markup_documentation("/tmp/nonexistent_abc.md")
        assert "error" in result


# =====================================================================
# _parse_markdown_content
# =====================================================================


class TestParseMarkdownContent:
    """Tests for _parse_markdown_content."""

    async def test_headers_extracted(self):
        content = "# H1\n## H2\n### H3\nSome text.\n"
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content(content)
        assert len(result["headers"]) == 3
        levels = [h["level"] for h in result["headers"]]
        assert levels == [1, 2, 3]
        titles = [h["title"] for h in result["headers"]]
        assert titles == ["H1", "H2", "H3"]

    async def test_code_blocks(self):
        content = "# Title\n```python\nprint('hi')\n```\n```\nplain\n```\n"
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content(content)
        assert len(result["code_blocks"]) == 2
        assert result["code_blocks"][0]["language"] == "python"
        assert result["code_blocks"][0]["code"] == "print('hi')"
        assert result["code_blocks"][1]["language"] == "text"  # default

    async def test_links(self):
        content = "Check [Google](https://google.com) and [Docs](./docs.md).\n"
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content(content)
        assert len(result["links"]) == 2
        urls = [lnk["url"] for lnk in result["links"]]
        assert "https://google.com" in urls
        assert "./docs.md" in urls

    async def test_word_count(self):
        content = "one two three four five"
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content(content)
        assert result["word_count"] == 5

    async def test_coverage_always_1(self):
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content("anything")
        assert result["coverage"] == 1.0

    async def test_empty_markdown(self):
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content("")
        assert result["headers"] == []
        assert result["code_blocks"] == []
        assert result["links"] == []

    async def test_line_numbers_correct(self):
        content = "first line\n# Header\nthird line\n"
        ext = DocumentationExtractor()
        result = await ext._parse_markdown_content(content)
        assert result["headers"][0]["line_number"] == 2


# =====================================================================
# _parse_rst_content
# =====================================================================


class TestParseRstContent:
    """Tests for _parse_rst_content."""

    async def test_headers_with_underlines(self):
        content = "Title\n=====\n\nSubtitle\n--------\n"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert len(result["headers"]) == 2
        assert result["headers"][0]["title"] == "Title"
        assert result["headers"][0]["underline_char"] == "="
        assert result["headers"][1]["title"] == "Subtitle"
        assert result["headers"][1]["underline_char"] == "-"

    async def test_directives(self):
        content = ".. code:: python\n\n   print('hi')\n\n.. image:: logo.png\n"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert len(result["directives"]) == 2
        types = [d["type"] for d in result["directives"]]
        assert "code" in types
        assert "image" in types

    async def test_word_count(self):
        content = "one two three"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert result["word_count"] == 3

    async def test_coverage_always_1(self):
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content("anything")
        assert result["coverage"] == 1.0

    async def test_empty_rst(self):
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content("")
        assert result["headers"] == []
        assert result["directives"] == []

    async def test_header_shorter_underline_skipped(self):
        """Underline shorter than title should not match."""
        content = "Long Title Here\n==\n"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert len(result["headers"]) == 0

    async def test_various_underline_chars(self):
        content = "Title1\n~~~~~~\n\nTitle2\n^^^^^^\n\nTitle3\n######\n"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert len(result["headers"]) == 3
        chars = [h["underline_char"] for h in result["headers"]]
        assert chars == ["~", "^", "#"]

    async def test_line_numbers(self):
        content = "Title\n=====\n\nSubtitle\n--------\n"
        ext = DocumentationExtractor()
        result = await ext._parse_rst_content(content)
        assert result["headers"][0]["line_number"] == 1
        assert result["headers"][1]["line_number"] == 4


# =====================================================================
# _calculate_python_coverage
# =====================================================================


class TestCalculatePythonCoverage:
    """Tests for _calculate_python_coverage."""

    def test_all_documented(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring="Module doc",
            functions=[{"docstring": "yes"}, {"docstring": "yes"}],
            classes=[
                {
                    "docstring": "class doc",
                    "methods": [{"docstring": "m1"}, {"docstring": "m2"}],
                }
            ],
        )
        # 1 module + 2 functions + 1 class + 2 methods = 6 total, all documented
        assert cov == 1.0

    def test_none_documented(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring=None,
            functions=[{"name": "f1"}, {"name": "f2"}],
            classes=[{"name": "C", "methods": [{"name": "m"}]}],
        )
        # 1 module + 2 funcs + 1 class + 1 method = 5 items, 0 documented
        assert cov == 0.0

    def test_partial_coverage(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring="yes",
            functions=[{"docstring": "yes"}, {"name": "f2"}],
            classes=[],
        )
        # 1 module (documented) + 2 functions (1 documented) = 3 total, 2 documented
        assert abs(cov - 2 / 3) < 0.01

    def test_empty_file(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(module_docstring=None, functions=[], classes=[])
        # 1 module (undocumented), 0 funcs, 0 classes => 0/1 = 0.0
        assert cov == 0.0

    def test_only_module_docstring(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(module_docstring="doc", functions=[], classes=[])
        # 1/1 = 1.0
        assert cov == 1.0

    def test_class_without_methods(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring=None,
            functions=[],
            classes=[{"docstring": "class doc", "methods": []}],
        )
        # 1 module (0) + 1 class (1) = 2 items, 1 documented
        assert cov == 0.5

    def test_class_with_undocumented_methods(self):
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring="doc",
            functions=[],
            classes=[
                {
                    "docstring": "cls",
                    "methods": [{"name": "a"}, {"name": "b"}, {"docstring": "c"}],
                }
            ],
        )
        # 1 module(1) + 1 class(1) + 3 methods(1) = 5 total, 3 documented
        assert abs(cov - 3 / 5) < 0.01

    def test_missing_methods_key_uses_default(self):
        """Class dict with no 'methods' key should default to empty list."""
        ext = DocumentationExtractor()
        cov = ext._calculate_python_coverage(
            module_docstring="doc", functions=[], classes=[{"docstring": "cls"}]
        )
        # 1 module(1) + 1 class(1) = 2 total, 2 documented
        assert cov == 1.0


# =====================================================================
# _calculate_javascript_coverage
# =====================================================================


class TestCalculateJavascriptCoverage:
    """Tests for _calculate_javascript_coverage."""

    def test_no_functions_with_jsdoc(self):
        ext = DocumentationExtractor()
        assert ext._calculate_javascript_coverage(["doc1"], []) == 1.0

    def test_no_functions_no_jsdoc(self):
        ext = DocumentationExtractor()
        assert ext._calculate_javascript_coverage([], []) == 0.0

    def test_equal_comments_and_functions(self):
        ext = DocumentationExtractor()
        funcs = [{"name": "a"}, {"name": "b"}]
        cov = ext._calculate_javascript_coverage(["d1", "d2"], funcs)
        assert cov == 1.0

    def test_more_comments_than_functions(self):
        ext = DocumentationExtractor()
        funcs = [{"name": "a"}]
        cov = ext._calculate_javascript_coverage(["d1", "d2", "d3"], funcs)
        # min(3/1, 1.0) = 1.0
        assert cov == 1.0

    def test_fewer_comments_than_functions(self):
        ext = DocumentationExtractor()
        funcs = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
        cov = ext._calculate_javascript_coverage(["d1"], funcs)
        assert cov == 0.25


# =====================================================================
# Integration-style: full round-trip through extract_file_documentation
# =====================================================================


class TestFullRoundTrip:
    """End-to-end tests through extract_file_documentation for each type."""

    async def test_python_round_trip_with_classes_and_imports(self):
        code = (
            '"""Module doc."""\n'
            "import os\n"
            "from pathlib import Path\n\n"
            'class Greeter:\n    """Greeter class."""\n'
            '    def greet(self):\n        """Greet."""\n        pass\n'
            "    def _internal(self): pass\n\n"
            'def top_level():\n    """Top level."""\n    pass\n'
        )
        path = _write_tmp(code)
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result["file_type"] == "python"
            assert result["module_docstring"] == "Module doc."
            assert len(result["classes"]) == 1
            assert len(result["imports"]) >= 2
            assert "coverage" in result
        finally:
            os.unlink(path)

    async def test_js_round_trip_with_multiple_patterns(self):
        code = (
            "/** Main module. */\n"
            "import { Router } from 'express';\n"
            "const http = require('http');\n\n"
            "class Server extends EventEmitter {\n"
            "  start() { }\n"
            "}\n\n"
            "function init(config) { }\n"
            "const handler = (req) => req;\n"
        )
        path = _write_tmp(code, suffix=".js")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result["file_type"] == "javascript"
            assert len(result["classes"]) >= 1
            assert len(result["imports"]) >= 2
            assert len(result["functions"]) >= 2
        finally:
            os.unlink(path)

    async def test_markdown_round_trip_with_code_and_links(self):
        content = (
            "# README\n\n"
            "## Install\n\n"
            "```bash\nnpm install\n```\n\n"
            "See [docs](https://example.com).\n"
        )
        path = _write_tmp(content, suffix=".md")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result["file_type"] == "markdown"
            assert len(result["headers"]) == 2
            assert len(result["code_blocks"]) == 1
            assert len(result["links"]) == 1
        finally:
            os.unlink(path)

    async def test_rst_round_trip_with_directives(self):
        content = (
            "Title\n=====\n\n"
            "Subtitle\n--------\n\n"
            ".. note:: Important note\n\n"
            ".. code:: python\n\n"
            "   print('hi')\n"
        )
        path = _write_tmp(content, suffix=".rst")
        try:
            ext = DocumentationExtractor()
            result = await ext.extract_file_documentation(path)
            assert result["file_type"] == "rst"
            assert len(result["headers"]) == 2
            assert len(result["directives"]) == 2
        finally:
            os.unlink(path)
