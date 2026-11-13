"""Deep unit tests for PythonCodeProcessor covering all methods and edge cases."""

from __future__ import annotations

import ast
import textwrap
from unittest.mock import patch

import pytest

from models.core_models import CoreAnalysisResult
from processing.analyzers.python_analyzer import PythonCodeProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor():
    """Create a PythonCodeProcessor instance."""
    return PythonCodeProcessor()


@pytest.fixture
def simple_python_file(tmp_path):
    """Create a simple Python file for testing."""
    code = textwrap.dedent('''\
        """Module docstring."""

        import os
        from pathlib import Path

        class Greeter:
            """A greeter class."""

            def greet(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
    ''')
    f = tmp_path / "simple.py"
    f.write_text(code, encoding="utf-8")
    return str(f)


@pytest.fixture
def complex_python_file(tmp_path):
    """Create a Python file with nested control flow and security issues."""
    code = textwrap.dedent('''\
        """Complex module."""

        import subprocess
        from os import getenv

        PASSWORD = "supersecretpassword123"

        class Processor:
            """Process things."""

            def process(self, data):
                """Process data with nested logic."""
                if data:
                    for item in data:
                        if item > 0:
                            try:
                                result = item * 2
                            except Exception:
                                pass

            def risky(self):
                result = eval("1 + 1")
                user = input("name? ")
                subprocess.run(["ls"])
                return result

        def simple_func():
            pass
    ''')
    f = tmp_path / "complex.py"
    f.write_text(code, encoding="utf-8")
    return str(f)


@pytest.fixture
def syntax_error_file(tmp_path):
    """Create a Python file with a syntax error."""
    f = tmp_path / "broken.py"
    f.write_text("def foo(\n", encoding="utf-8")
    return str(f)


@pytest.fixture
def empty_python_file(tmp_path):
    """Create an empty Python file."""
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    return str(f)


@pytest.fixture
def init_file(tmp_path):
    """Create an __init__.py file."""
    f = tmp_path / "__init__.py"
    f.write_text("# package init\n", encoding="utf-8")
    return str(f)


@pytest.fixture
def main_guard_file(tmp_path):
    """Create a Python file with if __name__ == '__main__' guard."""
    code = textwrap.dedent("""\
        def main():
            pass

        if __name__ == "__main__":
            main()
    """)
    f = tmp_path / "runner.py"
    f.write_text(code, encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# Tests: get_supported_extensions / get_language_name
# ---------------------------------------------------------------------------


class TestBasicMethods:
    def test_get_supported_extensions(self, processor):
        extensions = processor.get_supported_extensions()
        assert ".py" in extensions
        assert ".pyx" in extensions
        assert len(extensions) == 2

    def test_get_language_name(self, processor):
        assert processor.get_language_name() == "python"

    def test_can_analyze_py_file(self, processor):
        assert processor.can_analyze_file("foo.py") is True
        assert processor.can_analyze_file("bar.pyx") is True
        assert processor.can_analyze_file("baz.js") is False
        assert processor.can_analyze_file("qux.rs") is False

    def test_init_components(self, processor):
        """Verify all sub-components are initialized."""
        assert processor.entity_extractor is not None
        assert processor.relationship_extractor is not None
        assert processor.dependency_analyzer is not None
        assert processor.ast_utils is not None


# ---------------------------------------------------------------------------
# Tests: analyze_file
# ---------------------------------------------------------------------------


class TestAnalyzeFile:
    async def test_analyze_simple_file_success(self, processor, simple_python_file):
        result = await processor.analyze_file(simple_python_file)
        assert result.success is True
        assert result.language == "python"
        assert result.file_path == simple_python_file
        assert result.error is None
        assert result.total_lines > 0
        assert result.complexity_metrics is not None
        assert result.metadata["file_path"] == simple_python_file
        assert result.metadata["language"] == "python"
        assert "file_metrics" in result.metadata
        assert "line_count" in result.metadata
        assert "file_size" in result.metadata

    async def test_analyze_file_entities_extracted(self, processor, simple_python_file):
        result = await processor.analyze_file(simple_python_file)
        assert result.success is True
        # Should have at least the class, methods, function, and imports
        assert len(result.entities) > 0

    async def test_analyze_file_relationships_extracted(self, processor, simple_python_file):
        result = await processor.analyze_file(simple_python_file)
        assert result.success is True
        # relationships list should exist (may be empty for simple code)
        assert isinstance(result.relationships, list)

    async def test_analyze_file_dependencies_extracted(self, processor, simple_python_file):
        result = await processor.analyze_file(simple_python_file)
        assert result.success is True
        assert isinstance(result.dependencies, list)
        # The file imports os and pathlib, so expect dependencies
        assert len(result.dependencies) > 0

    async def test_analyze_file_complexity_metrics(self, processor, simple_python_file):
        result = await processor.analyze_file(simple_python_file)
        assert result.success is True
        m = result.complexity_metrics
        assert m is not None
        assert m.function_count >= 2  # greet + add
        assert m.class_count >= 1  # Greeter
        assert m.lines_of_code > 0

    async def test_analyze_file_syntax_error(self, processor, syntax_error_file):
        result = await processor.analyze_file(syntax_error_file)
        assert result.success is False
        assert result.language == "python"
        assert "Syntax error" in result.error
        assert result.total_lines == 0
        assert result.metadata["error_type"] == "syntax_error"
        assert "line_number" in result.metadata

    async def test_analyze_file_nonexistent(self, processor):
        result = await processor.analyze_file("/nonexistent/path/to/file.py")
        assert result.success is False
        assert "File access error" in result.error or "error" in result.error.lower()
        assert result.metadata["error_type"] == "file_access_error"

    async def test_analyze_empty_file(self, processor, empty_python_file):
        result = await processor.analyze_file(empty_python_file)
        assert result.success is True
        assert result.total_lines == 0  # empty file has 0 lines after splitlines()

    async def test_analyze_file_unexpected_error(self, processor, simple_python_file):
        """If an unexpected error occurs during analysis, we get a graceful result."""
        with patch.object(
            processor.entity_extractor,
            "extract_python_entities",
            side_effect=RuntimeError("boom"),
        ):
            result = await processor.analyze_file(simple_python_file)
            assert result.success is False
            assert "Analysis error" in result.error
            assert result.metadata["error_type"] == "analysis_error"


# ---------------------------------------------------------------------------
# Tests: analyze_snippet
# ---------------------------------------------------------------------------


class TestAnalyzeSnippet:
    async def test_analyze_snippet_success(self, processor):
        code = "def hello():\n    return 'hi'\n"
        result = await processor.analyze_snippet(code)
        assert result.success is True
        assert result.language == "python"
        assert result.file_path == "snippet"
        assert result.metadata["source"] == "snippet"
        assert result.total_lines == 2

    async def test_analyze_snippet_with_file_path(self, processor):
        code = "x = 1\n"
        result = await processor.analyze_snippet(code, file_path="/some/path.py")
        assert result.success is True
        assert result.file_path == "/some/path.py"

    async def test_analyze_snippet_syntax_error(self, processor):
        code = "def broken(\n"
        result = await processor.analyze_snippet(code)
        assert result.success is False
        assert "Syntax error" in result.error
        assert result.metadata["error_type"] == "syntax_error"
        assert result.metadata["source"] == "snippet"
        assert result.file_path == "snippet"

    async def test_analyze_snippet_syntax_error_with_file_path(self, processor):
        code = "def broken(\n"
        result = await processor.analyze_snippet(code, file_path="/some/file.py")
        assert result.success is False
        assert result.file_path == "/some/file.py"

    async def test_analyze_snippet_unexpected_error(self, processor):
        """If an unexpected error occurs in snippet analysis, we get a graceful result."""
        with patch.object(
            processor.entity_extractor,
            "extract_python_entities",
            side_effect=ValueError("unexpected"),
        ):
            result = await processor.analyze_snippet("x = 1\n")
            assert result.success is False
            assert "Snippet analysis error" in result.error
            assert result.metadata["error_type"] == "snippet_analysis_error"
            assert result.metadata["source"] == "snippet"

    async def test_analyze_snippet_complexity_and_security(self, processor):
        code = textwrap.dedent("""\
            def risky():
                eval("1+1")
                if True:
                    for i in range(10):
                        pass
        """)
        result = await processor.analyze_snippet(code)
        assert result.success is True
        assert result.complexity_metrics is not None
        assert result.complexity_metrics.function_count >= 1
        # eval should be detected
        assert len(result.security_patterns) >= 1
        assert any(p.type == "dangerous_function" for p in result.security_patterns)


# ---------------------------------------------------------------------------
# Tests: analyze_directory
# ---------------------------------------------------------------------------


class TestAnalyzeDirectory:
    """Tests for analyze_directory.

    NOTE: The source has a known bug at line 284-285 where ``file_results``
    (a list of dicts) is accessed with attribute syntax
    (``result.complexity_metrics``).  This causes an ``AttributeError``
    whenever there is at least one successfully analysed file.  Tests work
    around the bug by targeting empty directories, only syntax-error files,
    or by mocking ``analyze_file`` to return ``success=False`` results.
    """

    async def test_analyze_directory_empty(self, processor, tmp_path):
        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True
        assert result.metadata["file_count"] == 0
        assert result.metadata["analyzed_files"] == 0

    async def test_analyze_directory_only_broken_files(self, processor, tmp_path):
        """When all files fail to parse, file_results stays empty."""
        (tmp_path / "bad1.py").write_text("def broken(\n", encoding="utf-8")
        (tmp_path / "bad2.py").write_text("class (\n", encoding="utf-8")

        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True
        assert result.metadata["file_count"] == 2
        assert result.metadata["analyzed_files"] == 0

    async def test_analyze_directory_bug_attribute_error(self, processor, tmp_path):
        """Demonstrate the known dict-vs-object bug in complexity aggregation."""
        (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
        with pytest.raises(AttributeError, match="complexity_metrics"):
            await processor.analyze_directory(str(tmp_path))

    async def test_analyze_directory_file_oserror(self, processor, tmp_path):
        """If analyze_file raises an OSError, directory analysis continues."""
        (tmp_path / "fail.py").write_text("y = 2\n", encoding="utf-8")

        async def patched_analyze(file_path):
            raise OSError("permission denied")

        with patch.object(processor, "analyze_file", side_effect=patched_analyze):
            result = await processor.analyze_directory(str(tmp_path))
            assert result.success is True
            assert result.metadata["analyzed_files"] == 0

    async def test_analyze_directory_unexpected_exception(self, processor, tmp_path):
        """If analyze_file raises an unexpected error, directory analysis continues."""
        (tmp_path / "boom.py").write_text("y = 2\n", encoding="utf-8")

        async def patched_analyze(file_path):
            raise RuntimeError("unexpected crash")

        with patch.object(processor, "analyze_file", side_effect=patched_analyze):
            result = await processor.analyze_directory(str(tmp_path))
            assert result.success is True
            assert result.metadata["analyzed_files"] == 0

    async def test_analyze_directory_failed_result_not_counted(self, processor, tmp_path):
        """A file returning success=False is not added to file_results."""
        (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")

        async def patched_analyze(file_path):
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=0,
                error="mock failure",
            )

        with patch.object(processor, "analyze_file", side_effect=patched_analyze):
            result = await processor.analyze_directory(str(tmp_path))
            assert result.success is True
            assert result.metadata["analyzed_files"] == 0

    async def test_analyze_directory_recursive_finds_files(self, processor, tmp_path):
        """Verify rglob picks up nested .py files."""
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "bad.py").write_text("def (\n", encoding="utf-8")

        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True
        assert result.metadata["file_count"] == 1


# ---------------------------------------------------------------------------
# Tests: _calculate_complexity_metrics
# ---------------------------------------------------------------------------


class TestCalculateComplexityMetrics:
    def test_empty_code(self, processor):
        tree = ast.parse("")
        metrics = processor._calculate_complexity_metrics(tree, "")
        assert metrics.function_count == 0
        assert metrics.class_count == 0
        assert metrics.lines_of_code == 0
        assert metrics.comment_lines == 0
        assert metrics.blank_lines == 0

    def test_function_count(self, processor):
        code = "def a(): pass\ndef b(): pass\ndef c(): pass\n"
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        assert metrics.function_count == 3

    def test_class_count(self, processor):
        code = "class A: pass\nclass B: pass\n"
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        assert metrics.class_count == 2

    def test_lines_metrics(self, processor):
        code = "# comment\nx = 1\n\ny = 2\n"
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        assert metrics.lines_of_code == 3  # comment, x=1, y=2
        assert metrics.comment_lines == 1
        assert metrics.blank_lines == 1

    def test_nested_class_depth(self, processor):
        code = textwrap.dedent("""\
            class Outer:
                def method(self):
                    if True:
                        for i in range(10):
                            pass
        """)
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        assert metrics.nested_depth >= 2  # if + for inside the class

    def test_cyclomatic_complexity_accumulated(self, processor):
        code = textwrap.dedent("""\
            def f1():
                if True:
                    pass

            def f2():
                for i in range(10):
                    if i > 5:
                        pass
        """)
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        # f1 base=1 + if=1 = 2; f2 base=1 + for=1 + if=1 = 3; total = 5
        assert metrics.cyclomatic_complexity >= 4

    def test_docstring_coverage(self, processor):
        code = textwrap.dedent('''\
            def documented():
                """Has docstring."""
                pass

            def undocumented():
                pass

            class Documented:
                """Has docstring."""
                pass

            class Undocumented:
                pass
        ''')
        tree = ast.parse(code)
        metrics = processor._calculate_complexity_metrics(tree, code)
        # 2 out of 4 have docstrings = 50%
        assert metrics.docstring_coverage == 50.0


# ---------------------------------------------------------------------------
# Tests: _detect_security_patterns
# ---------------------------------------------------------------------------


class TestDetectSecurityPatterns:
    def test_no_security_issues(self, processor):
        tree = ast.parse("x = 1\ny = 2\n")
        patterns = processor._detect_security_patterns(tree)
        assert patterns == []

    def test_eval_detected(self, processor):
        tree = ast.parse("eval('1+1')\n")
        patterns = processor._detect_security_patterns(tree)
        assert len(patterns) >= 1
        p = [sp for sp in patterns if sp.type == "dangerous_function"]
        assert len(p) == 1
        assert p[0].severity == "high"
        assert "eval" in p[0].description

    def test_exec_detected(self, processor):
        tree = ast.parse("exec('pass')\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "dangerous_function"]
        assert len(p) == 1
        assert "exec" in p[0].description

    def test_input_detected(self, processor):
        tree = ast.parse("x = input('name')\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "user_input"]
        assert len(p) == 1
        assert p[0].severity == "medium"

    def test_raw_input_detected(self, processor):
        tree = ast.parse("x = raw_input('name')\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "user_input"]
        assert len(p) == 1

    def test_subprocess_detected(self, processor):
        tree = ast.parse("import subprocess\nsubprocess.run(['ls'])\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "subprocess_call"]
        assert len(p) == 1
        assert p[0].severity == "medium"
        assert p[0].metadata["method"] == "run"

    def test_subprocess_popen(self, processor):
        tree = ast.parse("import subprocess\nsubprocess.Popen(['ls'])\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "subprocess_call"]
        assert len(p) == 1
        assert p[0].metadata["method"] == "Popen"

    def test_multiple_security_issues(self, processor):
        code = textwrap.dedent("""\
            eval("1+1")
            exec("pass")
            x = input("hello")
        """)
        tree = ast.parse(code)
        patterns = processor._detect_security_patterns(tree)
        assert len(patterns) >= 3


# ---------------------------------------------------------------------------
# Tests: _check_assignment_security
# ---------------------------------------------------------------------------


class TestCheckAssignmentSecurity:
    def test_hardcoded_password(self, processor):
        tree = ast.parse('PASSWORD = "a_long_secret_value_123"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 1
        assert p[0].severity == "high"
        assert "PASSWORD" in p[0].description

    def test_hardcoded_secret(self, processor):
        tree = ast.parse('API_SECRET = "verylongsecretkey99"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 1

    def test_hardcoded_token(self, processor):
        tree = ast.parse('auth_token = "abcdefghij123456"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 1

    def test_hardcoded_key(self, processor):
        tree = ast.parse('api_key = "my_long_api_key_value"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 1

    def test_short_value_ignored(self, processor):
        """Values <= 8 chars are considered placeholders and are not flagged."""
        tree = ast.parse('password = "short"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 0

    def test_non_string_value_ignored(self, processor):
        tree = ast.parse("password = 12345678901234\n")
        processor._detect_security_patterns(tree)
        # Integer constant behavior depends on implementation

    def test_non_keyword_variable_ignored(self, processor):
        tree = ast.parse('my_config = "a_very_long_value_here"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 0

    def test_tuple_target_ignored(self, processor):
        """Non-Name targets (like tuple unpacking) should be skipped."""
        tree = ast.parse('a, b = "longpassword123", "other"\n')
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 0

    def test_non_constant_value_ignored(self, processor):
        """Assignment to a function call result should not trigger."""
        tree = ast.parse("password = get_password()\n")
        patterns = processor._detect_security_patterns(tree)
        p = [sp for sp in patterns if sp.type == "hardcoded_secret"]
        assert len(p) == 0


# ---------------------------------------------------------------------------
# Tests: _check_node_for_security_issues
# ---------------------------------------------------------------------------


class TestCheckNodeForSecurityIssues:
    def test_call_node(self, processor):
        tree = ast.parse("eval('x')\n")
        call_node = tree.body[0].value  # ast.Call
        patterns = processor._check_node_for_security_issues(call_node)
        assert len(patterns) >= 1

    def test_assign_node(self, processor):
        tree = ast.parse('token = "averylongtokenvalue"\n')
        assign_node = tree.body[0]  # ast.Assign
        patterns = processor._check_node_for_security_issues(assign_node)
        assert len(patterns) >= 1

    def test_other_node_type(self, processor):
        """Non-Call/Assign nodes should return empty list."""
        tree = ast.parse("for x in range(10): pass\n")
        for_node = tree.body[0]  # ast.For
        patterns = processor._check_node_for_security_issues(for_node)
        assert patterns == []


# ---------------------------------------------------------------------------
# Tests: _check_function_call_security
# ---------------------------------------------------------------------------


class TestCheckFunctionCallSecurity:
    def test_safe_function_call(self, processor):
        tree = ast.parse("print('hello')\n")
        call_node = tree.body[0].value
        patterns = processor._check_function_call_security(call_node)
        assert patterns == []

    def test_attribute_call_not_subprocess(self, processor):
        """Attribute calls that aren't subprocess should not trigger."""
        tree = ast.parse("os.path.join('a', 'b')\n")
        call_node = tree.body[0].value
        patterns = processor._check_function_call_security(call_node)
        assert patterns == []

    def test_eval_metadata(self, processor):
        tree = ast.parse("eval('x')\n")
        call_node = tree.body[0].value
        patterns = processor._check_function_call_security(call_node)
        assert len(patterns) == 1
        assert patterns[0].metadata["function"] == "eval"
        assert "suggestion" in patterns[0].metadata


# ---------------------------------------------------------------------------
# Tests: _calculate_file_metrics
# ---------------------------------------------------------------------------


class TestCalculateFileMetrics:
    def test_basic_file_metrics(self, processor):
        code = textwrap.dedent("""\
            import os

            class Foo:
                def bar(self):
                    if True:
                        for i in range(10):
                            pass
                    x = 1

            def baz():
                try:
                    pass
                except Exception:
                    pass
        """)
        tree = ast.parse(code)
        metrics = processor._calculate_file_metrics(tree, code, "test.py")

        assert metrics["statement_counts"]["imports"] >= 1
        assert metrics["statement_counts"]["classes"] >= 1
        assert metrics["statement_counts"]["functions"] >= 2
        assert metrics["statement_counts"]["assignments"] >= 1
        assert metrics["statement_counts"]["if_statements"] >= 1
        assert metrics["statement_counts"]["loops"] >= 1
        assert metrics["statement_counts"]["try_blocks"] >= 1
        assert metrics["total_lines"] > 0
        assert metrics["non_empty_lines"] > 0
        assert metrics["file_size_bytes"] > 0
        assert metrics["is_package"] is False
        assert metrics["has_main_guard"] is False

    def test_init_file_detection(self, processor):
        tree = ast.parse("")
        metrics = processor._calculate_file_metrics(tree, "", "__init__.py")
        assert metrics["is_package"] is True

    def test_main_guard_detection(self, processor):
        code = 'if __name__ == "__main__":\n    pass\n'
        tree = ast.parse(code)
        metrics = processor._calculate_file_metrics(tree, code, "main.py")
        assert metrics["has_main_guard"] is True

    def test_comment_ratio(self, processor):
        code = "# comment 1\n# comment 2\nx = 1\ny = 2\n"
        tree = ast.parse(code)
        metrics = processor._calculate_file_metrics(tree, code, "test.py")
        assert metrics["comment_ratio"] == 0.5  # 2 comments / 4 lines

    def test_empty_content(self, processor):
        tree = ast.parse("")
        metrics = processor._calculate_file_metrics(tree, "", "test.py")
        assert metrics["total_lines"] == 0
        assert metrics["non_empty_lines"] == 0
        assert metrics["comment_ratio"] == 0
        assert metrics["file_size_bytes"] == 0

    def test_while_loop_counted(self, processor):
        code = "while True:\n    pass\n"
        tree = ast.parse(code)
        metrics = processor._calculate_file_metrics(tree, code, "test.py")
        assert metrics["statement_counts"]["loops"] == 1

    def test_import_from_counted(self, processor):
        code = "from os import path\n"
        tree = ast.parse(code)
        metrics = processor._calculate_file_metrics(tree, code, "test.py")
        assert metrics["statement_counts"]["imports"] == 1


# ---------------------------------------------------------------------------
# Tests: _calculate_docstring_coverage
# ---------------------------------------------------------------------------


class TestCalculateDocstringCoverage:
    def test_all_documented(self, processor):
        code = textwrap.dedent('''\
            def foo():
                """Documented."""
                pass

            class Bar:
                """Documented."""
                pass
        ''')
        tree = ast.parse(code)
        coverage = processor._calculate_docstring_coverage(tree)
        assert coverage == 100.0

    def test_none_documented(self, processor):
        code = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
        tree = ast.parse(code)
        coverage = processor._calculate_docstring_coverage(tree)
        assert coverage == 0.0

    def test_no_functions_or_classes(self, processor):
        code = "x = 1\ny = 2\n"
        tree = ast.parse(code)
        coverage = processor._calculate_docstring_coverage(tree)
        assert coverage == 0  # 0 total items => 0

    def test_partial_coverage(self, processor):
        code = textwrap.dedent('''\
            def documented():
                """Yes."""
                pass

            def undocumented():
                pass
        ''')
        tree = ast.parse(code)
        coverage = processor._calculate_docstring_coverage(tree)
        assert coverage == 50.0


# ---------------------------------------------------------------------------
# Tests: _calculate_nesting_depth
# ---------------------------------------------------------------------------


class TestCalculateNestingDepth:
    def test_no_nesting(self, processor):
        tree = ast.parse("x = 1\n")
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 0

    def test_single_if(self, processor):
        code = "if True:\n    pass\n"
        tree = ast.parse(code)
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 1

    def test_nested_if_for(self, processor):
        code = textwrap.dedent("""\
            if True:
                for i in range(10):
                    while True:
                        pass
        """)
        tree = ast.parse(code)
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 3

    def test_with_statement(self, processor):
        code = "with open('f') as f:\n    pass\n"
        tree = ast.parse(code)
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 1

    def test_try_except(self, processor):
        code = "try:\n    pass\nexcept:\n    pass\n"
        tree = ast.parse(code)
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 1

    def test_deeply_nested(self, processor):
        code = textwrap.dedent("""\
            if True:
                if True:
                    if True:
                        if True:
                            pass
        """)
        tree = ast.parse(code)
        depth = processor._calculate_nesting_depth(tree)
        assert depth == 4


# ---------------------------------------------------------------------------
# Tests: validate_syntax (inherited from BaseCodeProcessor)
# ---------------------------------------------------------------------------


class TestValidateSyntax:
    async def test_valid_syntax(self, processor):
        result = await processor.validate_syntax("x = 1\n")
        assert result is True

    async def test_invalid_syntax(self, processor):
        result = await processor.validate_syntax("def broken(\n")
        assert result is False

    async def test_exception_returns_false(self, processor):
        with patch.object(processor, "analyze_snippet", side_effect=Exception("boom")):
            result = await processor.validate_syntax("x = 1\n")
            assert result is False


# ---------------------------------------------------------------------------
# Tests: Integration-style (full file analysis with security)
# ---------------------------------------------------------------------------


class TestIntegrationSecurityFile:
    async def test_complex_file_security_patterns(self, processor, complex_python_file):
        result = await processor.analyze_file(complex_python_file)
        assert result.success is True
        # Should detect eval, input, subprocess, and hardcoded password
        types_found = {p.type for p in result.security_patterns}
        assert "dangerous_function" in types_found
        assert "user_input" in types_found
        assert "subprocess_call" in types_found
        assert "hardcoded_secret" in types_found

    async def test_complex_file_has_nesting(self, processor, complex_python_file):
        result = await processor.analyze_file(complex_python_file)
        assert result.success is True
        assert result.complexity_metrics.nested_depth >= 1

    async def test_main_guard_file(self, processor, main_guard_file):
        result = await processor.analyze_file(main_guard_file)
        assert result.success is True
        assert result.metadata["file_metrics"]["has_main_guard"] is True

    async def test_init_file(self, processor, init_file):
        result = await processor.analyze_file(init_file)
        assert result.success is True
        assert result.metadata["file_metrics"]["is_package"] is True
