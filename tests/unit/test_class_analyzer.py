"""Unit tests for processing/ast_analysis/class_analyzer.py.

Covers:
- ClassAnalyzer.extract_class_structure
"""

from __future__ import annotations

from processing.ast_analysis.class_analyzer import ClassAnalyzer


class TestClassAnalyzer:
    """Tests for ClassAnalyzer."""

    def test_extract_simple_class(self, tmp_path):
        """Should extract a simple class with methods and attributes."""
        code = '''
class MyClass:
    """My class docstring."""
    class_attr = 42

    def method_a(self):
        pass

    def method_b(self, x):
        return x
'''
        f = tmp_path / "test.py"
        f.write_text(code)

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert "MyClass" in result
        cls = result["MyClass"]
        assert "method_a" in cls["methods"]
        assert "method_b" in cls["methods"]
        assert "class_attr" in cls["attributes"]
        assert cls["docstring"] == "My class docstring."
        assert cls["file"] == str(f)

    def test_extract_inherited_class(self, tmp_path):
        """Should extract base classes."""
        code = """
class Parent:
    pass

class Child(Parent):
    pass
"""
        f = tmp_path / "test.py"
        f.write_text(code)

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert "Child" in result
        assert "Parent" in result["Child"]["bases"]

    def test_extract_empty_file(self, tmp_path):
        """Should return empty dict for file with no classes."""
        f = tmp_path / "test.py"
        f.write_text("x = 1\ny = 2\n")

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert result == {}

    def test_syntax_error_file(self, tmp_path):
        """Should return error dict for files with syntax errors."""
        f = tmp_path / "bad.py"
        f.write_text("def foo(:\n    pass")

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert "__error__" in result
        assert result["__error__"]["type"] == "SyntaxError"

    def test_multiple_classes(self, tmp_path):
        """Should extract multiple classes from one file."""
        code = """
class A:
    pass

class B:
    pass

class C:
    pass
"""
        f = tmp_path / "test.py"
        f.write_text(code)

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_class_with_no_methods(self, tmp_path):
        """Should handle class with only pass."""
        code = '''
class Empty:
    """Empty class."""
    pass
'''
        f = tmp_path / "test.py"
        f.write_text(code)

        analyzer = ClassAnalyzer()
        result = analyzer.extract_class_structure(str(f))

        assert "Empty" in result
        assert result["Empty"]["methods"] == []
        assert result["Empty"]["docstring"] == "Empty class."
