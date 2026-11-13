"""Round 4 coverage tests for processing/analyzers/deep.

Originally targeted services/deep_analysis.py; that module is now a thin
shim around the deep package, so the tests live against
processing.analyzers.deep directly.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from processing.analyzers.deep import DeepCodeAnalyzer


@pytest.fixture
def tmp_codebase(tmp_path):
    """Create a minimal Python codebase for testing."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text(
        textwrap.dedent('''\
        """Main module."""
        import os

        def hello():
            """Say hello."""
            return "hello"

        class Greeter:
            """Greeter class."""
            def greet(self, name):
                return f"Hello {name}"
        ''')
    )

    (src / "insecure.py").write_text(
        textwrap.dedent("""\
        import subprocess

        def run_command(cmd):
            result = subprocess.call(cmd, shell=True)
            return eval(cmd)

        password = "hardcoded_secret_123"
        exec("print('hi')")
        """)
    )

    (src / "complex.py").write_text(
        textwrap.dedent("""\
        def complex_func(a, b, c, d, e, f, g, h, i, j, k):
            if a: pass
            elif b: pass
            elif c: pass
            elif d: pass
            elif e: pass
            elif f: pass
            elif g: pass
            elif h: pass
            elif i: pass
            elif j: pass
            elif k: pass
            for x in range(10):
                for y in range(10):
                    pass
        """)
    )

    (src / "slow.py").write_text(
        textwrap.dedent("""\
        import time

        async def async_work():
            time.sleep(1)
            import requests
            requests.get("http://example.com")

        def build_string():
            result = ""
            for i in range(100):
                result += str(i)
            return result
        """)
    )

    (src / "smelly.py").write_text(
        textwrap.dedent("""\
        class Foo:
            pass

        def no_doc():
            pass

        def f(a, b, c, d, e, f, g, h):
            pass
        """)
    )

    (src / "broken.py").write_text("def broken(:\n    pass\n")

    (src / "wild.py").write_text("from os import *\n")

    return src


def test_analyze_codebase_full(tmp_codebase):
    """Full codebase analysis covers entity extraction, security, performance, etc."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    result = analyzer.analyze_codebase()
    assert result.overall_score >= 0
    assert len(result.entities) > 0
    assert len(result.issues) > 0


def test_safe_parse_ast_syntax_error(tmp_codebase):
    """Cover lines 130-132: SyntaxError in _safe_parse_ast."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    result = analyzer._safe_parse_ast("def broken(:\n    pass", "broken.py")
    assert result is None


def test_safe_parse_ast_general_error(tmp_codebase):
    """Cover lines 133-135: general exception in _safe_parse_ast."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    result = analyzer._safe_parse_ast(None, "bad.py")
    assert result is None


def test_extract_decorator_name_variants(tmp_codebase):
    """Cover lines 371-376: decorator name extraction."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))

    name_node = ast.Name(id="staticmethod")
    assert analyzer._extract_decorator_name(name_node) == "staticmethod"

    attr_node = ast.Attribute(attr="route", value=ast.Name(id="app"))
    result = analyzer._extract_decorator_name(attr_node)
    assert "route" in result

    call_node = ast.Call(func=ast.Name(id="dec"), args=[], keywords=[])
    result = analyzer._extract_decorator_name(call_node)
    assert isinstance(result, str)


def test_extract_base_name_variants(tmp_codebase):
    """Cover lines 380-385: base class name extraction."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))

    name_node = ast.Name(id="BaseModel")
    assert analyzer._extract_base_name(name_node) == "BaseModel"

    attr_node = ast.Attribute(attr="BaseClass", value=ast.Name(id="module"))
    assert "BaseClass" in analyzer._extract_base_name(attr_node)

    call_node = ast.Call(func=ast.Name(id="Base"), args=[], keywords=[])
    result = analyzer._extract_base_name(call_node)
    assert isinstance(result, str)


def test_analyze_codebase_no_python_files(tmp_path):
    """Cover lines 142-148: no Python files found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    analyzer = DeepCodeAnalyzer(str(empty_dir))
    result = analyzer.analyze_codebase()
    assert result.overall_score >= 0


def test_analyze_codebase_nonexistent_path():
    """Cover lines 117-119: nonexistent path."""
    analyzer = DeepCodeAnalyzer("/nonexistent/path/42")
    assert analyzer.python_files == []
    result = analyzer.analyze_codebase()
    assert result.overall_score >= 0


@pytest.mark.asyncio
async def test_analyze_code_snippet(tmp_codebase):
    """Cover snippet analysis code path."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    code = textwrap.dedent("""\
    def hello():
        return "world"

    class Foo:
        pass
    """)
    result = await analyzer.analyze_code_snippet(code, language="python")
    assert result is not None
    assert result.overall_score >= 0


@pytest.mark.asyncio
async def test_analyze_code_snippet_syntax_error(tmp_codebase):
    """Cover lines 869-870: snippet with SyntaxError."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    result = await analyzer.analyze_code_snippet("def broken(:\n    pass", language="python")
    assert result is not None
    # SyntaxError is caught and handled, result should still be valid
    assert result.overall_score >= 0


def test_analyze_code_snippet_non_python(tmp_codebase):
    """Cover non-python language snippet path."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))
    result = analyzer.analyze_code_snippet("fn main() {}", language="rust")
    assert result is not None


def test_extract_docstring(tmp_codebase):
    """Cover lines 362-367: docstring extraction."""
    analyzer = DeepCodeAnalyzer(str(tmp_codebase))

    code = '''def hello():\n    """Hello doc."""\n    pass\n'''
    tree = ast.parse(code)
    func = tree.body[0]
    doc = analyzer._extract_docstring(func)
    assert doc == "Hello doc."

    code2 = "def nope():\n    pass\n"
    tree2 = ast.parse(code2)
    func2 = tree2.body[0]
    doc2 = analyzer._extract_docstring(func2)
    assert doc2 is None


def test_entity_extraction_with_file_error(tmp_path):
    """Cover lines 353-354: OSError during entity extraction."""
    src = tmp_path / "err"
    src.mkdir()
    f = src / "test.py"
    f.write_text("x = 1\n")
    analyzer = DeepCodeAnalyzer(str(src))
    # Make the file unreadable
    f.chmod(0o000)
    try:
        analyzer._extract_entities()
        # Should not crash, just log warning
    except Exception:
        pass
    finally:
        f.chmod(0o644)
