"""Unit tests for processing.python.python_parser.PythonParser."""

from __future__ import annotations

import ast

import pytest

from processing.python.python_parser import PythonParser


@pytest.fixture
def parser():
    """Return a fresh PythonParser instance."""
    return PythonParser()


# ---------------------------------------------------------------------------
# parse_code
# ---------------------------------------------------------------------------


class TestParseCode:
    """Tests for PythonParser.parse_code."""

    async def test_valid_python_returns_ast_module(self, parser):
        code = "x = 1\ny = 2\n"
        result = await parser.parse_code(code, "test.py")
        assert isinstance(result, ast.Module)

    async def test_valid_python_preserves_structure(self, parser):
        code = "def greet(name):\n    return f'Hello, {name}'\n"
        result = await parser.parse_code(code, "greet.py")
        assert isinstance(result, ast.Module)
        assert len(result.body) == 1
        assert isinstance(result.body[0], ast.FunctionDef)
        assert result.body[0].name == "greet"

    async def test_syntax_error_returns_none(self, parser):
        code = "def broken(\n"
        result = await parser.parse_code(code, "broken.py")
        assert result is None

    async def test_indentation_error_returns_none(self, parser):
        code = "if True:\nprint('bad')\n"
        result = await parser.parse_code(code, "indent.py")
        assert result is None

    async def test_empty_string_returns_ast_module(self, parser):
        result = await parser.parse_code("", "empty.py")
        assert isinstance(result, ast.Module)
        assert result.body == []

    async def test_whitespace_only_returns_ast_module(self, parser):
        result = await parser.parse_code("   \n\n  \n", "spaces.py")
        assert isinstance(result, ast.Module)
        assert result.body == []

    async def test_class_definition_parsed(self, parser):
        code = "class Foo:\n    x = 1\n"
        result = await parser.parse_code(code, "foo.py")
        assert isinstance(result, ast.Module)
        assert isinstance(result.body[0], ast.ClassDef)
        assert result.body[0].name == "Foo"

    async def test_multiple_statements(self, parser):
        code = "import os\nimport sys\nx = 1\n"
        result = await parser.parse_code(code, "multi.py")
        assert isinstance(result, ast.Module)
        assert len(result.body) == 3

    async def test_options_parameter_accepted(self, parser):
        """Options parameter is accepted but unused for Python parsing."""
        code = "x = 1\n"
        result = await parser.parse_code(code, "test.py", options={"some": "option"})
        assert isinstance(result, ast.Module)


# ---------------------------------------------------------------------------
# validate_syntax
# ---------------------------------------------------------------------------


class TestValidateSyntax:
    """Tests for PythonParser.validate_syntax."""

    async def test_valid_code_returns_true(self, parser):
        result = await parser.validate_syntax("x = 1\n", "test.py")
        assert result is True

    async def test_invalid_code_returns_false(self, parser):
        result = await parser.validate_syntax("def broken(\n", "bad.py")
        assert result is False

    async def test_empty_string_is_valid(self, parser):
        result = await parser.validate_syntax("", "empty.py")
        assert result is True

    async def test_complex_valid_code(self, parser):
        code = "class MyClass:\n    def method(self, x: int) -> str:\n        return str(x)\n"
        result = await parser.validate_syntax(code, "cls.py")
        assert result is True

    async def test_incomplete_expression_is_invalid(self, parser):
        result = await parser.validate_syntax("x = (1 +", "expr.py")
        assert result is False

    async def test_valid_async_code(self, parser):
        code = "async def run():\n    await something()\n"
        result = await parser.validate_syntax(code, "async.py")
        assert result is True
