"""Unit tests for FunctionExtractor."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from processing.ast_analysis.function_extractor import FunctionExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_temp_py(code: str) -> str:
    """Write Python code to a temp file, return its path."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(code))
    return path


@pytest.fixture
def extractor():
    fe = FunctionExtractor()
    # The ASTDebugMixin stub in conftest doesn't provide real methods.
    # Patch the mixin methods that may be called at runtime.
    fe.syntax_error_handling = MagicMock(return_value={"context": "mock debug context"})
    fe.debug_ast_node = MagicMock()
    return fe


@pytest.fixture
def extractor_no_inference():
    fe = FunctionExtractor(enable_type_inference=False, analyze_usage_patterns=False)
    fe.syntax_error_handling = MagicMock(return_value={"context": "mock debug context"})
    fe.debug_ast_node = MagicMock()
    return fe


# ---------------------------------------------------------------------------
# extract_function_signatures
# ---------------------------------------------------------------------------


class TestExtractFunctionSignatures:
    """Tests for extract_function_signatures."""

    def test_simple_function(self, extractor):
        path = _write_temp_py("""
def hello():
    pass
""")
        try:
            result = extractor.extract_function_signatures(path)
            assert "hello" in result
            assert result["hello"]["inputs"] == []
            assert result["hello"]["output"] is None
            assert result["hello"]["file"] == path
            assert result["hello"]["line"] == 2
        finally:
            os.unlink(path)

    def test_function_with_annotations(self, extractor):
        path = _write_temp_py("""
def add(a: int, b: int) -> int:
    return a + b
""")
        try:
            result = extractor.extract_function_signatures(path)
            assert "add" in result
            assert any("int" in p for p in result["add"]["inputs"])
            assert result["add"]["output"] is not None
            assert "int" in result["add"]["output"]
        finally:
            os.unlink(path)

    def test_function_with_docstring(self, extractor):
        path = _write_temp_py('''
def greet(name):
    """Say hello to name."""
    return f"Hello {name}"
''')
        try:
            result = extractor.extract_function_signatures(path)
            assert result["greet"]["docstring"] == "Say hello to name."
        finally:
            os.unlink(path)

    def test_multiple_functions(self, extractor):
        path = _write_temp_py("""
def foo():
    pass

def bar():
    pass

def baz():
    pass
""")
        try:
            result = extractor.extract_function_signatures(path)
            assert "foo" in result
            assert "bar" in result
            assert "baz" in result
        finally:
            os.unlink(path)

    def test_syntax_error_returns_error(self, extractor):
        path = _write_temp_py("def broken(:\n    pass\n")
        try:
            result = extractor.extract_function_signatures(path)
            assert "error" in result
            assert "Syntax error" in result["error"]
            assert "details" in result
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self, extractor):
        # open() is outside the try block, so FileNotFoundError propagates
        with pytest.raises(FileNotFoundError):
            extractor.extract_function_signatures("/nonexistent/file.py")

    def test_no_functions(self, extractor):
        path = _write_temp_py("x = 42\n")
        try:
            result = extractor.extract_function_signatures(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_unannotated_params(self, extractor):
        path = _write_temp_py("""
def foo(a, b, c):
    pass
""")
        try:
            result = extractor.extract_function_signatures(path)
            assert result["foo"]["inputs"] == ["a", "b", "c"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _extract_parameters
# ---------------------------------------------------------------------------


class TestExtractParameters:
    """Tests for _extract_parameters."""

    def test_params_with_types(self, extractor):
        tree = ast.parse("def f(x: int, y: str): pass")
        func_node = tree.body[0]
        params = extractor._extract_parameters(func_node)
        assert len(params) == 2
        assert "int" in params[0]
        assert "str" in params[1]

    def test_params_without_types(self, extractor):
        tree = ast.parse("def f(a, b): pass")
        func_node = tree.body[0]
        params = extractor._extract_parameters(func_node)
        assert params == ["a", "b"]

    def test_mixed_params(self, extractor):
        tree = ast.parse("def f(a, b: int): pass")
        func_node = tree.body[0]
        params = extractor._extract_parameters(func_node)
        assert params[0] == "a"
        assert "int" in params[1]


# ---------------------------------------------------------------------------
# _extract_return_type
# ---------------------------------------------------------------------------


class TestExtractReturnType:
    """Tests for _extract_return_type."""

    def test_with_return_type(self, extractor):
        tree = ast.parse("def f() -> str: pass")
        func_node = tree.body[0]
        ret = extractor._extract_return_type(func_node)
        assert ret is not None
        assert "str" in ret

    def test_no_return_type(self, extractor):
        tree = ast.parse("def f(): pass")
        func_node = tree.body[0]
        ret = extractor._extract_return_type(func_node)
        assert ret is None


# ---------------------------------------------------------------------------
# extract_with_ast_visualization
# ---------------------------------------------------------------------------


class TestExtractWithAstVisualization:
    """Tests for extract_with_ast_visualization."""

    def test_returns_signatures_and_debug(self, extractor):
        path = _write_temp_py("""
def compute(x: int) -> int:
    return x * 2
""")
        try:
            result = extractor.extract_with_ast_visualization(path)
            assert "signatures" in result
            assert "function_count" in result
            assert result["function_count"] == 1
            assert "compute" in result["signatures"]
            assert "ast_debug" in result
            assert result["ast_debug"]["parse_successful"] is True
        finally:
            os.unlink(path)

    def test_ast_info_present(self, extractor):
        path = _write_temp_py("""
def foo():
    pass
""")
        try:
            result = extractor.extract_with_ast_visualization(path)
            sig = result["signatures"]["foo"]
            assert "ast_info" in sig
            assert sig["ast_info"]["node_type"] == "FunctionDef"
        finally:
            os.unlink(path)

    def test_syntax_error(self, extractor):
        path = _write_temp_py("def bad(:\n    pass\n")
        try:
            result = extractor.extract_with_ast_visualization(path)
            assert result["function_count"] == 0
            assert "error" in result
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# extract_comprehensive_signatures
# ---------------------------------------------------------------------------


class TestExtractComprehensiveSignatures:
    """Tests for extract_comprehensive_signatures."""

    def test_non_python_file(self, extractor):
        result = extractor.extract_comprehensive_signatures("file.js")
        assert result["total_functions"] == 0
        assert "error" in result

    def test_basic_extraction(self, extractor):
        path = _write_temp_py("""
def greet(name: str) -> str:
    \"\"\"Greet someone.\"\"\"
    return f"Hello {name}"
""")
        try:
            result = extractor.extract_comprehensive_signatures(path)
            assert "functions" in result
            assert len(result["functions"]) == 1
            func = result["functions"][0]
            assert func["name"] == "greet"
            assert func["return_type"] is not None
            assert func["docstring"] == "Greet someone."
            assert "metadata" in result
            assert result["metadata"]["total_functions"] == 1
        finally:
            os.unlink(path)

    def test_type_coverage(self, extractor):
        path = _write_temp_py("""
def typed(x: int) -> int:
    return x

def untyped(x):
    return x
""")
        try:
            result = extractor.extract_comprehensive_signatures(path)
            assert result["type_coverage"] == 0.5
            assert result["metadata"]["typed_functions"] == 1
            assert result["metadata"]["total_functions"] == 2
        finally:
            os.unlink(path)

    def test_with_repository_path(self, extractor):
        path = _write_temp_py("""
def foo():
    pass
""")
        try:
            repo_path = os.path.dirname(path)
            result = extractor.extract_comprehensive_signatures(path, repo_path)
            assert len(result["functions"]) == 1
            func = result["functions"][0]
            assert func["module"] is not None
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self, extractor):
        # open() is outside the try block, so FileNotFoundError propagates
        with pytest.raises(FileNotFoundError):
            extractor.extract_comprehensive_signatures("/nonexistent/file.py")

    def test_analysis_id_generated(self, extractor):
        path = _write_temp_py("def f(): pass\n")
        try:
            result = extractor.extract_comprehensive_signatures(path)
            assert result["analysis_id"].startswith("sig_")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _extract_detailed_parameters
# ---------------------------------------------------------------------------


class TestExtractDetailedParameters:
    """Tests for _extract_detailed_parameters."""

    def test_params_with_defaults(self, extractor):
        tree = ast.parse("def f(a, b=10, c='x'): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        assert len(params) == 3
        assert params[0]["required"] is True
        assert params[1]["required"] is False
        assert params[1]["name"] == "b"
        assert params[2]["required"] is False

    def test_varargs(self, extractor):
        tree = ast.parse("def f(*args): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        assert any(p["name"] == "*args" for p in params)

    def test_kwargs(self, extractor):
        tree = ast.parse("def f(**kwargs): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        assert any(p["name"] == "**kwargs" for p in params)

    def test_varargs_with_annotation(self, extractor):
        tree = ast.parse("def f(*args: int): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        vararg = [p for p in params if p["name"] == "*args"][0]
        assert "int" in vararg["type"]

    def test_kwargs_with_annotation(self, extractor):
        tree = ast.parse("def f(**kwargs: str): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        kwarg = [p for p in params if p["name"] == "**kwargs"][0]
        assert "str" in kwarg["type"]

    def test_no_params(self, extractor):
        tree = ast.parse("def f(): pass")
        func_node = tree.body[0]
        params = extractor._extract_detailed_parameters(func_node)
        assert params == []


# ---------------------------------------------------------------------------
# _build_function_relationships
# ---------------------------------------------------------------------------


class TestBuildFunctionRelationships:
    """Tests for _build_function_relationships."""

    def test_builds_calls_and_called_by(self, extractor):
        code = textwrap.dedent("""
def caller():
    callee()

def callee():
    pass
""")
        tree = ast.parse(code)
        extractor._build_function_relationships(tree)
        assert "callee" in extractor.function_calls_map.get("caller", set())
        assert "caller" in extractor.called_by_map.get("callee", set())

    def test_attribute_calls(self, extractor):
        code = textwrap.dedent("""
def worker():
    os.path.join("a", "b")
""")
        tree = ast.parse(code)
        extractor._build_function_relationships(tree)
        calls = extractor.function_calls_map.get("worker", set())
        assert len(calls) > 0

    def test_clears_previous_maps(self, extractor):
        extractor.function_calls_map["old"] = {"stale"}
        extractor.called_by_map["stale"] = {"old"}
        tree = ast.parse("def new_func(): pass")
        extractor._build_function_relationships(tree)
        assert "old" not in extractor.function_calls_map
        assert "stale" not in extractor.called_by_map


# ---------------------------------------------------------------------------
# _calculate_complexity_metrics
# ---------------------------------------------------------------------------


class TestCalculateComplexityMetrics:
    """Tests for _calculate_complexity_metrics."""

    def test_simple_function(self, extractor):
        tree = ast.parse("def f(): pass")
        metrics = extractor._calculate_complexity_metrics(tree.body[0])
        assert metrics["cyclomatic"] == 1
        assert metrics["cognitive"] == 0

    def test_if_increases_complexity(self, extractor):
        code = textwrap.dedent("""
def f():
    if True:
        pass
""")
        tree = ast.parse(code)
        metrics = extractor._calculate_complexity_metrics(tree.body[0])
        assert metrics["cyclomatic"] >= 2
        assert metrics["cognitive"] >= 1

    def test_loop_increases_complexity(self, extractor):
        code = textwrap.dedent("""
def f():
    for i in range(10):
        pass
    while True:
        break
""")
        tree = ast.parse(code)
        metrics = extractor._calculate_complexity_metrics(tree.body[0])
        assert metrics["cyclomatic"] >= 3
        assert metrics["cognitive"] >= 2

    def test_try_except_complexity(self, extractor):
        code = textwrap.dedent("""
def f():
    try:
        pass
    except ValueError:
        pass
""")
        tree = ast.parse(code)
        metrics = extractor._calculate_complexity_metrics(tree.body[0])
        assert metrics["cyclomatic"] >= 2
        assert metrics["cognitive"] >= 1

    def test_bool_op_complexity(self, extractor):
        code = textwrap.dedent("""
def f():
    if a and b and c:
        pass
""")
        tree = ast.parse(code)
        metrics = extractor._calculate_complexity_metrics(tree.body[0])
        # BoolOp with 3 values adds 2, plus the if adds 1
        assert metrics["cyclomatic"] >= 4


# ---------------------------------------------------------------------------
# _extract_external_dependencies
# ---------------------------------------------------------------------------


class TestExtractExternalDependencies:
    """Tests for _extract_external_dependencies."""

    def test_no_dependencies(self, extractor):
        tree = ast.parse("def f(): pass")
        deps = extractor._extract_external_dependencies(tree.body[0])
        assert deps == []

    def test_attribute_dependencies(self, extractor):
        code = textwrap.dedent("""
import os
import json as j
def f(item):
    os.path.join("a", "b")
    j.dumps({})
    item.strip()
""")
        tree = ast.parse(code)
        extractor.import_bindings = extractor._import_bindings(tree)
        deps = extractor._extract_external_dependencies(tree.body[2])
        assert deps == ["json", "os"]  # imports only; never item.strip


# ---------------------------------------------------------------------------
# _build_function_signature
# ---------------------------------------------------------------------------


class TestBuildFunctionSignature:
    """Tests for _build_function_signature."""

    def test_no_params_no_return(self, extractor):
        sig = extractor._build_function_signature("foo", [], None)
        assert sig == "def foo()"

    def test_with_return_type(self, extractor):
        sig = extractor._build_function_signature("foo", [], "int")
        assert sig == "def foo() -> int"

    def test_typed_params(self, extractor):
        params = [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "str"},
        ]
        sig = extractor._build_function_signature("bar", params, None)
        assert sig == "def bar(x: int, y: str)"

    def test_params_with_defaults(self, extractor):
        params = [
            {"name": "x", "type": "int", "default": "0"},
            {"name": "y", "type": None, "default": "'hello'"},
        ]
        sig = extractor._build_function_signature("baz", params, "bool")
        assert sig == "def baz(x: int = 0, y='hello') -> bool"

    def test_untyped_params(self, extractor):
        params = [{"name": "a"}, {"name": "b"}]
        sig = extractor._build_function_signature("fn", params, None)
        assert sig == "def fn(a, b)"


# ---------------------------------------------------------------------------
# _extract_param_description
# ---------------------------------------------------------------------------


class TestExtractParamDescription:
    """Tests for _extract_param_description."""

    def test_no_docstring(self, extractor):
        tree = ast.parse("def f(x): pass")
        result = extractor._extract_param_description(tree.body[0], "x")
        assert result is None

    def test_param_in_docstring(self, extractor):
        code = textwrap.dedent('''
def f(x):
    """Do something.

    x: The input value
    """
    pass
''')
        tree = ast.parse(code)
        result = extractor._extract_param_description(tree.body[0], "x")
        assert result is not None
        assert "input value" in result

    def test_param_not_in_docstring(self, extractor):
        code = textwrap.dedent('''
def f(x):
    """Do something."""
    pass
''')
        tree = ast.parse(code)
        result = extractor._extract_param_description(tree.body[0], "x")
        assert result is None


# ---------------------------------------------------------------------------
# _get_module_name
# ---------------------------------------------------------------------------


class TestGetModuleName:
    """Tests for _get_module_name."""

    def test_without_repo_path(self, extractor):
        result = extractor._get_module_name("/some/path/module.py")
        # Should convert file path to module-like string
        assert "module" in result

    def test_with_repo_path(self, extractor):
        result = extractor._get_module_name("/repo/src/pkg/module.py", "/repo")
        # Should strip 'src' prefix
        assert "pkg" in result
        assert "module" in result
        assert "src" not in result

    def test_src_prefix_removed(self, extractor):
        result = extractor._get_module_name("/repo/src/mypackage/file.py", "/repo")
        parts = result.split(".")
        assert "src" not in parts


# ---------------------------------------------------------------------------
# _ast_to_string
# ---------------------------------------------------------------------------


class TestAstToString:
    """Tests for _ast_to_string."""

    def test_simple_name_node(self, extractor):
        node = ast.parse("int", mode="eval").body
        result = extractor._ast_to_string(node)
        assert "int" in result

    def test_invalid_node_fallback(self, extractor):
        # AST Module node can't be converted by astor.to_source easily
        node = ast.Module(body=[], type_ignores=[])
        result = extractor._ast_to_string(node)
        # Should return something, even if it's the type name
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _get_timestamp
# ---------------------------------------------------------------------------


class TestGetTimestamp:
    """Tests for _get_timestamp."""

    def test_returns_iso_string(self, extractor):
        ts = extractor._get_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format contains T separator


# ---------------------------------------------------------------------------
# run_type_inference
# ---------------------------------------------------------------------------


class TestRunTypeInference:
    """Tests for run_type_inference."""

    def test_disabled(self, extractor_no_inference):
        result = extractor_no_inference.run_type_inference("some_file.py")
        assert result == {"type_inference": "disabled"}

    @patch("processing.ast_analysis.function_extractor.subprocess.run")
    def test_with_json_output(self, mock_run, extractor):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"errors": []}),
            returncode=0,
        )
        result = extractor.run_type_inference("file.py")
        assert result["type_inference"] == {"errors": []}

    @patch("processing.ast_analysis.function_extractor.subprocess.run")
    def test_with_non_json_output(self, mock_run, extractor):
        mock_run.return_value = MagicMock(
            stdout="some plain text output",
            returncode=0,
        )
        result = extractor.run_type_inference("file.py")
        assert result["type_inference"] == "some plain text output"

    @patch("processing.ast_analysis.function_extractor.subprocess.run")
    def test_no_output(self, mock_run, extractor):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = extractor.run_type_inference("file.py")
        assert result["type_inference"] == "no_output"

    @patch("processing.ast_analysis.function_extractor.subprocess.run")
    def test_timeout(self, mock_run, extractor):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ty", timeout=30)
        result = extractor.run_type_inference("file.py")
        assert "error" in result["type_inference"]

    @patch("processing.ast_analysis.function_extractor.subprocess.run")
    def test_file_not_found(self, mock_run, extractor):
        mock_run.side_effect = FileNotFoundError("ty not found")
        result = extractor.run_type_inference("file.py")
        assert "error" in result["type_inference"]


# ---------------------------------------------------------------------------
# _extract_usage_examples
# ---------------------------------------------------------------------------


class TestExtractUsageExamples:
    """Tests for _extract_usage_examples."""

    def test_never_synthesises_an_example(self, extractor):
        """No call site and no docstring example -> no example (was "f(x, y)")."""
        tree = ast.parse("def f(x, y): pass")
        extractor.call_sites = {}
        assert extractor._extract_usage_examples(tree.body[0], "mod.f") == []

    def test_real_call_site_is_the_example(self, extractor):
        code = "def f(x):\n    return x\n\n\ndef g():\n    return f(41)\n"
        tree = ast.parse(code)
        extractor._source = code
        extractor._build_function_relationships(tree)
        assert extractor._extract_usage_examples(tree.body[0], "mod.f") == ["f(41)"]

    def test_docstring_example(self, extractor):
        code = textwrap.dedent('''
def greet(name):
    """Greet someone.

    Example usage:
        greet(name)
    """
    pass
''')
        tree = ast.parse(code)
        func_node = tree.body[0]
        examples = extractor._extract_usage_examples(func_node, "mod.greet")
        assert len(examples) >= 1

    def test_limits_to_three(self, extractor):
        tree = ast.parse("def f(a, b, c, d, e): pass")
        func_node = tree.body[0]
        examples = extractor._extract_usage_examples(func_node, "mod.f")
        assert len(examples) <= 3


# ---------------------------------------------------------------------------
# _extract_function_relationships
# ---------------------------------------------------------------------------


class TestExtractFunctionRelationships:
    """Tests for _extract_function_relationships."""

    def test_returns_expected_keys(self, extractor):
        tree = ast.parse("def f(): pass")
        func_node = tree.body[0]
        result = extractor._extract_function_relationships(func_node, "mod.f")
        assert "usage_examples" in result
        assert "called_by" in result
        assert "calls" in result


# ---------------------------------------------------------------------------
# analyze_with_directory_context
# ---------------------------------------------------------------------------


class TestAnalyzeWithDirectoryContext:
    """Tests for analyze_with_directory_context."""

    def test_without_project_root(self, extractor):
        path = _write_temp_py("def f(): pass\n")
        try:
            result = extractor.analyze_with_directory_context(path)
            assert "functions" in result
        finally:
            os.unlink(path)

    @patch("processing.ast_analysis.function_extractor.DirectoryTreeAnalyzer")
    def test_with_project_root(self, mock_analyzer_cls, extractor):
        path = _write_temp_py("def f(): pass\n")
        project_root = os.path.dirname(path)
        mock_instance = MagicMock()
        mock_instance.analyze_directory.return_value = {
            "tree": {"type": "directory", "name": "root", "children": []},
            "statistics": {"total_files": 1},
            "language_distribution": {"python": 1},
        }
        mock_analyzer_cls.return_value = mock_instance
        try:
            result = extractor.analyze_with_directory_context(path, project_root)
            assert "directory_context" in result
            assert result["directory_context"]["project_statistics"]["total_files"] == 1
        finally:
            os.unlink(path)

    def test_file_not_found(self, extractor):
        # analyze_with_directory_context calls extract_comprehensive_signatures
        # which has open() outside try, so FileNotFoundError propagates
        # through the generic Exception catch in analyze_with_directory_context
        result = extractor.analyze_with_directory_context("/nonexistent/file.py")
        assert "error" in result

    @patch(
        "processing.ast_analysis.function_extractor.DirectoryTreeAnalyzer",
        new=None,
    )
    def test_directory_analyzer_not_available(self, extractor):
        path = _write_temp_py("def f(): pass\n")
        try:
            result = extractor.analyze_with_directory_context(path, "/some/root")
            assert "error" in result
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _find_file_in_tree
# ---------------------------------------------------------------------------


class TestFindFileInTree:
    """Tests for _find_file_in_tree."""

    def test_finds_file(self, extractor):
        tree = {
            "type": "directory",
            "name": "root",
            "children": [
                {
                    "type": "directory",
                    "name": "src",
                    "children": [
                        {"type": "file", "name": "main.py"},
                    ],
                }
            ],
        }
        result = extractor._find_file_in_tree(tree, "src/main.py")
        assert result is not None
        assert result["name"] == "main.py"

    def test_file_not_found(self, extractor):
        tree = {
            "type": "directory",
            "name": "root",
            "children": [],
        }
        result = extractor._find_file_in_tree(tree, "src/missing.py")
        assert result is None

    def test_directory_not_found_in_path(self, extractor):
        tree = {
            "type": "directory",
            "name": "root",
            "children": [
                {"type": "file", "name": "other.py"},
            ],
        }
        result = extractor._find_file_in_tree(tree, "nonexistent/file.py")
        assert result is None

    def test_non_directory_in_path(self, extractor):
        tree = {
            "type": "file",
            "name": "root.py",
        }
        result = extractor._find_file_in_tree(tree, "sub/file.py")
        assert result is None


# ---------------------------------------------------------------------------
# _extract_comprehensive_function_data
# ---------------------------------------------------------------------------


class TestExtractComprehensiveFunctionData:
    """Tests for _extract_comprehensive_function_data."""

    def test_basic_data(self, extractor):
        code = textwrap.dedent('''
def worker(x: int) -> str:
    """Do work."""
    return str(x)
''')
        tree = ast.parse(code)
        extractor._build_function_relationships(tree)
        func_node = tree.body[0]
        result = extractor._extract_comprehensive_function_data(func_node, "/tmp/test.py")
        assert result["name"] == "worker"
        assert result["docstring"] == "Do work."
        assert result["line_number"] == 2
        assert "complexity_metrics" in result
        assert "parameters" in result
        assert result["is_async"] is False

    def test_decorators(self, extractor):
        code = textwrap.dedent("""
@staticmethod
def decorated():
    pass
""")
        tree = ast.parse(code)
        extractor._build_function_relationships(tree)
        func_node = tree.body[0]
        result = extractor._extract_comprehensive_function_data(func_node, "/tmp/test.py")
        assert len(result["decorators"]) == 1

    def test_with_repository_path(self, extractor):
        code = "def f(): pass\n"
        tree = ast.parse(code)
        extractor._build_function_relationships(tree)
        func_node = tree.body[0]
        result = extractor._extract_comprehensive_function_data(
            func_node, "/repo/src/pkg/mod.py", "/repo"
        )
        assert "pkg" in result["module"]


# ---------------------------------------------------------------------------
# Constructor / init
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for FunctionExtractor.__init__."""

    def test_default_init(self):
        fe = FunctionExtractor()
        assert fe.enable_type_inference is True
        assert fe.analyze_usage_patterns is True
        assert fe.function_calls_map == {}
        assert fe.called_by_map == {}

    def test_custom_init(self):
        fe = FunctionExtractor(enable_type_inference=False, analyze_usage_patterns=False)
        assert fe.enable_type_inference is False
        assert fe.analyze_usage_patterns is False


class TestSignaturesAreGroundedNotInvented:
    """Live 2026-10-12 (cv-m2-hello, planted pkg/): ``calls``/``called_by`` were empty
    for every function (maps keyed by short name, read by module-qualified name),
    ``usage_examples`` were synthesised from parameter names, async defs were skipped
    and ``external_dependencies`` listed method calls such as ``item.strip``."""

    CODE = textwrap.dedent("""
        import os
        from json import dumps


        def add(a: int, b: int) -> int:
            return a + b


        def caller(items):
            for item in items:
                item.strip()
            os.path.join("a", "b")
            dumps({})
            return add(1, 2)


        async def handler():
            return add(3, 4)


        def unused(x):
            return x
    """)

    def _analyse(self, tmp_path, extractor):
        path = tmp_path / "pkg_sig.py"
        path.write_text(self.CODE)
        result = extractor.extract_comprehensive_signatures(str(path), str(tmp_path))
        return {f["name"]: f for f in result["functions"]}, result

    def test_calls_and_called_by_are_populated(self, tmp_path, extractor):
        funcs, _ = self._analyse(tmp_path, extractor)
        assert "add" in funcs["caller"]["calls"]
        assert set(funcs["add"]["called_by"]) == {"caller", "handler"}
        assert funcs["unused"]["called_by"] == []

    def test_async_functions_are_extracted(self, tmp_path, extractor):
        funcs, result = self._analyse(tmp_path, extractor)
        assert funcs["handler"]["is_async"] is True
        assert result["metadata"]["total_functions"] == 4

    def test_usage_examples_are_real_call_sites_only(self, tmp_path, extractor):
        funcs, _ = self._analyse(tmp_path, extractor)
        assert set(funcs["add"]["usage_examples"]) == {"add(1, 2)", "add(3, 4)"}
        assert funcs["unused"]["usage_examples"] == []  # never "unused(x)"

    def test_external_dependencies_are_imports_not_method_calls(self, tmp_path, extractor):
        funcs, result = self._analyse(tmp_path, extractor)
        assert funcs["caller"]["external_dependencies"] == ["json.dumps", "os"]
        assert all("strip" not in d for d in result["external_dependencies"])
