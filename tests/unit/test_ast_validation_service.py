"""Unit tests for services/code_analysis/ast_validation_service.py.

Covers:
- ASTValidationService.analyze_proposed_code_ast
- ASTValidationService._analyze_python_ast
- ASTValidationService._extract_import_info
- ASTValidationService._extract_variable_info
- ASTValidationService._get_attribute_name
- ASTValidationService._calculate_complexity_score
- ASTIssue and ASTAnalysisResult models
"""

from __future__ import annotations

import ast
from unittest.mock import AsyncMock, MagicMock

from services.code_analysis.ast_validation_service import (
    ASTAnalysisResult,
    ASTIssue,
    ASTValidationService,
)

# ---------------------------------------------------------------------------
# Tests for models
# ---------------------------------------------------------------------------


class TestASTModels:
    """Tests for ASTIssue and ASTAnalysisResult models."""

    def test_ast_issue_defaults(self):
        issue = ASTIssue(description="test issue")
        assert issue.description == "test issue"
        assert issue.line_number == 0

    def test_ast_issue_with_line(self):
        issue = ASTIssue(description="line issue", line_number=42)
        assert issue.line_number == 42

    def test_ast_analysis_result_valid(self):
        result = ASTAnalysisResult(
            is_valid=True,
            syntax_errors=[],
            imports=[],
            functions=[],
            classes=[],
            variables=[],
            complexity_score=1.5,
            line_count=10,
        )
        assert result.is_valid is True
        assert result.complexity_score == 1.5

    def test_ast_analysis_result_with_issues(self):
        result = ASTAnalysisResult(
            is_valid=False,
            syntax_errors=["error1"],
            imports=[],
            functions=[],
            classes=[],
            variables=[],
            complexity_score=0.0,
            line_count=5,
            issues=[ASTIssue(description="issue1")],
        )
        assert len(result.issues) == 1
        assert result.syntax_errors == ["error1"]


# ---------------------------------------------------------------------------
# Tests for ASTValidationService
# ---------------------------------------------------------------------------


class TestASTValidationService:
    """Tests for ASTValidationService."""

    def test_init_default_extractor(self):
        """Should create with default PythonEntityExtractor."""
        svc = ASTValidationService()
        assert svc.entity_extractor is not None

    def test_init_custom_extractor(self):
        """Should accept a custom entity extractor."""
        mock_extractor = MagicMock()
        svc = ASTValidationService(entity_extractor=mock_extractor)
        assert svc.entity_extractor is mock_extractor


class TestAnalyzeProposedCodeAst:
    """Tests for analyze_proposed_code_ast."""

    async def test_non_python_language(self):
        """Non-python language should return not valid with error."""
        svc = ASTValidationService()
        result = await svc.analyze_proposed_code_ast(
            "const x = 1;", "test.js", language="javascript"
        )
        assert result.is_valid is False
        assert any("javascript" in e.lower() for e in result.syntax_errors)
        assert result.line_count == 1

    async def test_python_valid_code(self):
        """Valid Python should succeed."""
        svc = ASTValidationService()
        code = "x = 1\ny = 2\n"
        result = await svc.analyze_proposed_code_ast(code, "test.py", language="python")
        assert result.is_valid is True
        assert result.line_count == 2

    async def test_python_syntax_error(self):
        """Invalid Python syntax should be caught."""
        svc = ASTValidationService()
        code = "def foo(\n"
        result = await svc.analyze_proposed_code_ast(code, "test.py", language="python")
        assert result.is_valid is False
        assert len(result.syntax_errors) > 0

    async def test_exception_in_analysis(self):
        """An unexpected exception should return not valid."""
        mock_extractor = MagicMock()
        # Make extract_entities raise an exception
        mock_extractor.extract_entities = AsyncMock(side_effect=RuntimeError("boom"))
        svc = ASTValidationService(entity_extractor=mock_extractor)
        # This code parses fine but entity extraction will fail
        code = "x = 1"
        result = await svc.analyze_proposed_code_ast(code, "test.py", language="python")
        # The exception is caught in the outer try
        assert result.is_valid is False

    async def test_empty_code(self):
        """Empty code should parse successfully."""
        svc = ASTValidationService()
        result = await svc.analyze_proposed_code_ast("", "empty.py", language="python")
        assert result.is_valid is True
        assert result.line_count == 0


class TestAnalyzePythonAst:
    """Tests for _analyze_python_ast internals."""

    async def test_extracts_imports(self):
        """Should extract import and from-import statements."""
        svc = ASTValidationService()
        code = "import os\nfrom pathlib import Path\n"
        result = await svc._analyze_python_ast(code)
        assert result.is_valid is True
        import_names = [imp["module"] for imp in result.imports]
        assert "os" in import_names
        assert "pathlib.Path" in import_names

    async def test_extracts_variables(self):
        """Should extract variable assignments."""
        svc = ASTValidationService()
        code = "x = 1\nname = 'test'\n"
        result = await svc._analyze_python_ast(code)
        assert result.is_valid is True
        var_names = [v["name"] for v in result.variables]
        assert "x" in var_names
        assert "name" in var_names

    async def test_extracts_functions_and_classes(self):
        """Should extract function and class definitions."""
        svc = ASTValidationService()
        code = '''
class MyClass:
    """A class."""
    def method(self):
        pass

def standalone():
    """A function."""
    pass
'''
        result = await svc._analyze_python_ast(code)
        assert result.is_valid is True
        class_names = [c["name"] for c in result.classes]
        assert "MyClass" in class_names

    async def test_complexity_score_calculated(self):
        """Should have a complexity score for code with control flow."""
        svc = ASTValidationService()
        code = """
def complex_func(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                try:
                    print(i)
                except Exception:
                    pass
    while x > 10:
        x -= 1
"""
        result = await svc._analyze_python_ast(code)
        assert result.is_valid is True
        assert result.complexity_score > 0

    async def test_syntax_error_caught(self):
        """Should catch and report syntax errors."""
        svc = ASTValidationService()
        code = "def foo(:\n    pass"
        result = await svc._analyze_python_ast(code)
        assert result.is_valid is False
        assert len(result.syntax_errors) > 0


class TestExtractImportInfo:
    """Tests for _extract_import_info."""

    def test_import_statement(self):
        svc = ASTValidationService()
        node = ast.parse("import os").body[0]
        result = svc._extract_import_info(node)
        assert len(result) == 1
        assert result[0]["module"] == "os"
        assert result[0]["type"] == "import"

    def test_import_with_alias(self):
        svc = ASTValidationService()
        node = ast.parse("import numpy as np").body[0]
        result = svc._extract_import_info(node)
        assert result[0]["module"] == "numpy"
        assert result[0]["alias"] == "np"

    def test_from_import(self):
        svc = ASTValidationService()
        node = ast.parse("from os.path import join").body[0]
        result = svc._extract_import_info(node)
        assert result[0]["module"] == "os.path.join"
        assert result[0]["type"] == "from_import"

    def test_from_import_multiple(self):
        svc = ASTValidationService()
        node = ast.parse("from typing import Any, Optional").body[0]
        result = svc._extract_import_info(node)
        assert len(result) == 2

    def test_relative_import(self):
        svc = ASTValidationService()
        code = "from . import sibling"
        tree = ast.parse(code)
        node = tree.body[0]
        result = svc._extract_import_info(node)
        assert result[0]["level"] > 0


class TestExtractVariableInfo:
    """Tests for _extract_variable_info."""

    def test_simple_assignment(self):
        svc = ASTValidationService()
        node = ast.parse("x = 1").body[0]
        result = svc._extract_variable_info(node)
        assert len(result) == 1
        assert result[0]["name"] == "x"
        assert result[0]["type"] == "assignment"

    def test_tuple_unpack_not_captured(self):
        """Tuple unpacking targets are not Name nodes, so not captured."""
        svc = ASTValidationService()
        node = ast.parse("a, b = 1, 2").body[0]
        result = svc._extract_variable_info(node)
        # Tuple target is ast.Tuple, not ast.Name at top level
        assert len(result) == 0


class TestGetAttributeName:
    """Tests for _get_attribute_name."""

    def test_simple_name(self):
        svc = ASTValidationService()
        node = ast.parse("x", mode="eval").body
        assert svc._get_attribute_name(node) == "x"

    def test_attribute(self):
        svc = ASTValidationService()
        node = ast.parse("os.path", mode="eval").body
        result = svc._get_attribute_name(node)
        assert result == "os.path"

    def test_nested_attribute(self):
        svc = ASTValidationService()
        node = ast.parse("os.path.join", mode="eval").body
        result = svc._get_attribute_name(node)
        assert result == "os.path.join"

    def test_unknown_node_type(self):
        svc = ASTValidationService()
        node = ast.parse("1 + 2", mode="eval").body
        result = svc._get_attribute_name(node)
        assert result == "unknown"


class TestCalculateComplexityScore:
    """Tests for _calculate_complexity_score."""

    def test_empty_code(self):
        svc = ASTValidationService()
        tree = ast.parse("")
        score = svc._calculate_complexity_score(tree)
        assert score == 0.0

    def test_simple_code(self):
        svc = ASTValidationService()
        tree = ast.parse("x = 1\ny = 2\n")
        score = svc._calculate_complexity_score(tree)
        assert score >= 0.0

    def test_complex_code(self):
        svc = ASTValidationService()
        code = """
if True:
    for i in range(10):
        while True:
            try:
                pass
            except Exception:
                pass
"""
        tree = ast.parse(code)
        score = svc._calculate_complexity_score(tree)
        assert score > 0.0
        assert score <= 10.0

    def test_function_adds_complexity(self):
        svc = ASTValidationService()
        code = """
def a():
    pass
def b():
    pass
async def c():
    pass
"""
        tree = ast.parse(code)
        score = svc._calculate_complexity_score(tree)
        assert score > 0.0
