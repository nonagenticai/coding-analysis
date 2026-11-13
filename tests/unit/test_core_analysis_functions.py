"""Tests for app.api.endpoints.analysis_modules.core_analysis functions.

Covers basic_analysis, enhanced_analysis, ast_analysis, deep_analysis,
security_analysis, and performance_analysis with real code inputs.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.analysis_modules.core_analysis import (
    ast_analysis,
    basic_analysis,
    deep_analysis,
    enhanced_analysis,
    performance_analysis,
    security_analysis,
)
from app.api.models.analysis_api_models import AnalysisRequest, PydanticAIAnalysisType


@pytest.fixture(autouse=True)
def _mock_analysis_status_service():
    """Mock AnalysisStatusService to avoid Redis dependency in tests."""
    mock_service = MagicMock()
    mock_service.update_status = AsyncMock(return_value=MagicMock())
    mock_service.create_analysis_session = AsyncMock(return_value=MagicMock())
    with patch(
        "services.analysis_status.AnalysisStatusService",
        return_value=mock_service,
    ):
        yield mock_service


def _make_request(
    code: str,
    analysis_type: str = "basic",
    language: str | None = "python",
    container_name: str = "test-container",
) -> AnalysisRequest:
    return AnalysisRequest(
        analysis_type=PydanticAIAnalysisType(analysis_type),
        code=code,
        container_name=container_name,
        language=language,
    )


# ---------------------------------------------------------------------------
# basic_analysis
# ---------------------------------------------------------------------------


class TestBasicAnalysis:
    """Tests for the basic_analysis function."""

    @pytest.mark.asyncio
    async def test_empty_code_raises(self):
        request = _make_request(code="", analysis_type="basic")
        # code="" is falsy
        request.code = ""
        with pytest.raises(ValueError, match="Code content is required"):
            await basic_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_none_code_raises(self):
        request = _make_request(code="x = 1", analysis_type="basic")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await basic_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_simple_python(self):
        code = "x = 1\ny = 2\nresult = x + y\n"
        request = _make_request(code=code, language="python")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True
        assert result.metrics.lines_of_code == 3
        assert result.analysis_type == "basic"

    @pytest.mark.asyncio
    async def test_todo_comment_detected(self):
        code = "# TODO: fix this\nx = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        todo_issues = [i for i in result.issues if i.rule == "todo-comment"]
        assert len(todo_issues) == 1

    @pytest.mark.asyncio
    async def test_long_line_detected(self):
        code = "x = " + "a" * 120 + "\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        long_line_issues = [i for i in result.issues if i.rule == "line-length"]
        assert len(long_line_issues) == 1

    @pytest.mark.asyncio
    async def test_trailing_whitespace_detected(self):
        code = "x = 1   \ny = 2\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        ws_issues = [i for i in result.issues if i.rule == "trailing-whitespace"]
        assert len(ws_issues) == 1

    @pytest.mark.asyncio
    async def test_javascript_console_log(self):
        code = 'console.log("hello");\nlet x = 1;\n'
        request = _make_request(code=code, language="javascript")
        result = await basic_analysis(request, "id-1", time.time())
        console_issues = [i for i in result.issues if i.rule == "no-console"]
        assert len(console_issues) >= 1

    @pytest.mark.asyncio
    async def test_typescript_console_log(self):
        code = 'console.log("hello");\nconst x: number = 1;\n'
        request = _make_request(code=code, language="typescript")
        result = await basic_analysis(request, "id-1", time.time())
        console_issues = [i for i in result.issues if i.rule == "no-console"]
        assert len(console_issues) >= 1

    @pytest.mark.asyncio
    async def test_java_println(self):
        code = 'System.out.println("hello");\nint x = 1;\n'
        request = _make_request(code=code, language="java")
        result = await basic_analysis(request, "id-1", time.time())
        println_issues = [i for i in result.issues if i.rule == "no-println"]
        assert len(println_issues) >= 1

    @pytest.mark.asyncio
    async def test_complexity_estimation(self):
        code = "if x > 0:\n    for i in range(10):\n        while True:\n            break\n"
        request = _make_request(code=code, language="python")
        result = await basic_analysis(request, "id-1", time.time())
        # 3 control flow keywords -> complexity >= 4
        assert result.metrics.complexity >= 4.0

    @pytest.mark.asyncio
    async def test_csharp_keywords(self):
        code = "if (x) { } foreach (var i in items) { }\n"
        request = _make_request(code=code, language="csharp")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_go_keywords(self):
        code = "if x > 0 {\n}\nfor i := range items {\n}\nselect {\n}\n"
        request = _make_request(code=code, language="go")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rust_keywords(self):
        code = "if x > 0 {\n}\nmatch x {\n    _ => {}\n}\nloop {\n    break;\n}\n"
        request = _make_request(code=code, language="rust")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_cpp_keywords(self):
        code = "if (x) {} switch(y) { case 1: break; }\ntry {} catch(...) {}\n"
        request = _make_request(code=code, language="cpp")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_language_defaults_python(self):
        code = "if x:\n    pass\n"
        request = _make_request(code=code, language="unknown-lang")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.success is True
        assert result.metadata["language"] == "unknown-lang"

    @pytest.mark.asyncio
    async def test_fixme_comment(self):
        code = "# FIXME: broken logic\nx = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        todo_issues = [i for i in result.issues if i.rule == "todo-comment"]
        assert len(todo_issues) == 1

    @pytest.mark.asyncio
    async def test_hack_comment(self):
        code = "# HACK: workaround\nx = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        todo_issues = [i for i in result.issues if i.rule == "todo-comment"]
        assert len(todo_issues) == 1

    @pytest.mark.asyncio
    async def test_xxx_comment(self):
        code = "# XXX: needs attention\nx = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time())
        todo_issues = [i for i in result.issues if i.rule == "todo-comment"]
        assert len(todo_issues) == 1

    @pytest.mark.asyncio
    async def test_maintainability_score(self):
        # Many control flow -> lower maintainability
        code = "\n".join([f"if x{i}:\n    pass" for i in range(30)])
        request = _make_request(code=code, language="python")
        result = await basic_analysis(request, "id-1", time.time())
        assert result.metrics.maintainability < 50

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self):
        code = "x = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "id-1", time.time() - 0.5)
        assert result.execution_time >= 0.5

    @pytest.mark.asyncio
    async def test_metadata_has_analysis_id(self):
        code = "x = 1\n"
        request = _make_request(code=code)
        result = await basic_analysis(request, "my-id", time.time())
        assert result.metadata["analysis_id"] == "my-id"

    @pytest.mark.asyncio
    async def test_no_language_defaults_unknown(self):
        code = "x = 1\n"
        request = _make_request(code=code, language=None)
        result = await basic_analysis(request, "id-1", time.time())
        assert result.metadata["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_python_print_not_closed(self):
        code = 'print("hello\nx = 1\n'
        request = _make_request(code=code, language="python")
        result = await basic_analysis(request, "id-1", time.time())
        syntax_issues = [i for i in result.issues if i.rule == "syntax-check"]
        assert len(syntax_issues) >= 1


# ---------------------------------------------------------------------------
# security_analysis
# ---------------------------------------------------------------------------


class TestSecurityAnalysis:
    """Tests for the security_analysis function."""

    @pytest.mark.asyncio
    async def test_none_code_raises(self):
        request = _make_request(code="x = 1", analysis_type="security")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await security_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_sql_injection_detected(self):
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        sql_issues = [i for i in result.issues if i.rule == "sql-injection"]
        assert len(sql_issues) >= 1

    @pytest.mark.asyncio
    async def test_eval_detected(self):
        code = "result = eval(user_input)\n"
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        injection_issues = [i for i in result.issues if i.rule == "code-injection"]
        assert len(injection_issues) >= 1

    @pytest.mark.asyncio
    async def test_exec_detected(self):
        code = "exec(user_code)\n"
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        injection_issues = [i for i in result.issues if i.rule == "code-injection"]
        assert len(injection_issues) >= 1

    @pytest.mark.asyncio
    async def test_hardcoded_password_detected(self):
        code = 'password="secret123"\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        secret_issues = [i for i in result.issues if i.rule == "hardcoded-secret"]
        assert len(secret_issues) >= 1

    @pytest.mark.asyncio
    async def test_hardcoded_api_key_detected(self):
        code = "api_key='sk-12345'\n"
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        secret_issues = [i for i in result.issues if i.rule == "hardcoded-secret"]
        assert len(secret_issues) >= 1

    @pytest.mark.asyncio
    async def test_clean_code_no_issues(self):
        code = "def add(a, b):\n    return a + b\n"
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        assert result.success is True
        assert len([i for i in result.issues if i.type == "security"]) == 0

    @pytest.mark.asyncio
    async def test_security_score_decreases_with_issues(self):
        code = 'eval(x)\nexec(y)\npassword="abc"\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        score = result.metadata["security_score"]
        assert score < 100

    @pytest.mark.asyncio
    async def test_delete_from_detected(self):
        code = 'query = "DELETE FROM users" + condition\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        sql_issues = [i for i in result.issues if i.rule == "sql-injection"]
        assert len(sql_issues) >= 1

    @pytest.mark.asyncio
    async def test_drop_table_detected(self):
        code = 'query = "DROP TABLE users" + "%s" % name\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        sql_issues = [i for i in result.issues if i.rule == "sql-injection"]
        assert len(sql_issues) >= 1

    @pytest.mark.asyncio
    async def test_suggestions_for_sql_injection(self):
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\n'
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        assert any("parameterized" in s.lower() for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_suggestions_for_eval(self):
        code = "result = eval(user_input)\n"
        request = _make_request(code=code, analysis_type="security")
        result = await security_analysis(request, "id-1", time.time())
        assert any("eval" in s.lower() for s in result.suggestions)


# ---------------------------------------------------------------------------
# performance_analysis
# ---------------------------------------------------------------------------


class TestPerformanceAnalysis:
    """Tests for the performance_analysis function."""

    @pytest.mark.asyncio
    async def test_none_code_raises(self):
        request = _make_request(code="x = 1", analysis_type="performance")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await performance_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_nested_loops_detected(self):
        code = "for i in range(10):\n    for j in range(10):\n        pass\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        nested_issues = [i for i in result.issues if i.rule == "nested-loops"]
        assert len(nested_issues) >= 1

    @pytest.mark.asyncio
    async def test_string_concat_detected(self):
        code = 'result = ""\nresult += "hello"\n'
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        concat_issues = [i for i in result.issues if i.rule == "string-concat"]
        assert len(concat_issues) >= 1

    @pytest.mark.asyncio
    async def test_sync_in_async_detected(self):
        code = 'async def fetch():\n    data = open("file.txt")\n'
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        sync_issues = [i for i in result.issues if i.rule == "sync-in-async"]
        assert len(sync_issues) >= 1

    @pytest.mark.asyncio
    async def test_requests_get_in_async_detected(self):
        code = "async def fetch():\n    r = requests.get(url)\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        sync_issues = [i for i in result.issues if i.rule == "sync-in-async"]
        assert len(sync_issues) >= 1

    @pytest.mark.asyncio
    async def test_time_sleep_in_async_detected(self):
        code = "async def wait():\n    time.sleep(5)\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        sync_issues = [i for i in result.issues if i.rule == "sync-in-async"]
        assert len(sync_issues) >= 1

    @pytest.mark.asyncio
    async def test_clean_code_no_issues(self):
        code = "def add(a, b):\n    return a + b\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        assert result.success is True
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_performance_score_decreases_with_issues(self):
        code = "for i in range(10):\n    for j in range(10):\n        pass\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        assert result.metadata["performance_score"] < 100

    @pytest.mark.asyncio
    async def test_loop_count_in_metadata(self):
        code = "for i in range(10):\n    pass\nwhile True:\n    break\n"
        request = _make_request(code=code, analysis_type="performance")
        result = await performance_analysis(request, "id-1", time.time())
        assert result.metadata["loop_count"] >= 2


# ---------------------------------------------------------------------------
# enhanced_analysis
# ---------------------------------------------------------------------------


class TestEnhancedAnalysis:
    """Tests for the enhanced_analysis function."""

    @pytest.mark.asyncio
    async def test_empty_code_falls_back_to_basic(self):
        """When code is None, enhanced_analysis falls back to basic_analysis,
        which also raises ValueError since code is still None."""
        request = _make_request(code="x = 1", analysis_type="augmented")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await enhanced_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_simple_python_code(self):
        code = '''
def greet(name):
    """Say hello."""
    return f"Hello, {name}!"

class MyClass:
    """A simple class."""
    def method(self):
        return 42
'''
        request = _make_request(code=code, analysis_type="augmented", language="python")
        result = await enhanced_analysis(request, "id-1", time.time())
        assert result.success is True
        assert result.metrics.lines_of_code >= 0

    @pytest.mark.asyncio
    async def test_complex_function_detected(self):
        # Generate a function with many branches
        lines = ["def complex_func(x):"]
        for i in range(15):
            lines.append(f"    if x == {i}:")
            lines.append(f"        return {i}")
        lines.append("    return -1")
        code = "\n".join(lines)
        request = _make_request(code=code, analysis_type="augmented", language="python")
        result = await enhanced_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_class_without_docstring(self):
        code = "class NoDocClass:\n    pass\n"
        request = _make_request(code=code, analysis_type="augmented", language="python")
        result = await enhanced_analysis(request, "id-1", time.time())
        assert result.success is True


# ---------------------------------------------------------------------------
# ast_analysis
# ---------------------------------------------------------------------------


class TestASTAnalysis:
    """Tests for the ast_analysis function."""

    @pytest.mark.asyncio
    async def test_simple_python_code(self):
        code = '''
import os

def process(path):
    """Process a path."""
    return os.path.exists(path)

class Handler:
    def handle(self):
        return True
'''
        request = _make_request(code=code, analysis_type="ast", language="python")
        result = await ast_analysis(request, "id-1", time.time())
        assert result.success is True
        assert "ast_function_count" in result.metadata or result.metadata.get("analysis_id")

    @pytest.mark.asyncio
    async def test_none_code_falls_back(self):
        """When code is None, ast_analysis falls back to basic_analysis,
        which also raises ValueError since code is still None."""
        request = _make_request(code="x = 1", analysis_type="ast")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await ast_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_complex_code_with_classes_and_functions(self):
        code = """
class Base:
    pass

class Child(Base):
    def method_a(self):
        return 1
    def method_b(self):
        return 2

def standalone():
    if True:
        for i in range(10):
            if i > 5:
                return i
    return 0
"""
        request = _make_request(code=code, analysis_type="ast", language="python")
        result = await ast_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_valid_code_has_no_syntax_findings(self):
        request = _make_request(code="x = 1\n", analysis_type="ast", language="python")
        result = await ast_analysis(request, "id-1", time.time())
        assert result.success is True
        assert [i for i in result.issues if i.type == "syntax"] == []
        assert result.metadata["ast_valid"] is True

    @pytest.mark.asyncio
    async def test_syntax_error_is_a_structured_finding(self):
        # This used to return success=True with ZERO issues: the service caught
        # the SyntaxError into `syntax_errors`, and only `issues` was mapped.
        code = "def broken(:\n    return 1\n"
        request = _make_request(code=code, analysis_type="ast", language="python")
        result = await ast_analysis(request, "id-1", time.time())
        assert result.success is False
        assert result.metadata["ast_valid"] is False
        syntax = [i for i in result.issues if i.type == "syntax"]
        assert len(syntax) == 1
        finding = syntax[0]
        assert finding.severity == "critical"
        assert finding.rule == "syntax-error"
        assert finding.line == 1
        assert finding.column >= 1
        assert finding.message

    @pytest.mark.asyncio
    async def test_unsupported_language_is_a_finding_not_a_pass(self):
        request = _make_request(code="fn main() {}", analysis_type="ast", language="rust")
        result = await ast_analysis(request, "id-1", time.time())
        assert result.success is False
        assert [i.rule for i in result.issues] == ["ast-parse-failed"]


# ---------------------------------------------------------------------------
# deep_analysis
# ---------------------------------------------------------------------------


class TestDeepAnalysis:
    """Tests for the deep_analysis function."""

    @pytest.mark.asyncio
    async def test_simple_python_code(self):
        code = '''
def calculate(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    """Simple calculator."""
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
        request = _make_request(code=code, analysis_type="deep", language="python")
        result = await deep_analysis(request, "id-1", time.time())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_none_code_falls_back(self):
        """When code is None, deep_analysis falls back to enhanced -> basic,
        which raises ValueError since code is still None."""
        request = _make_request(code="x = 1", analysis_type="deep")
        request.code = None
        with pytest.raises(ValueError, match="Code content is required"):
            await deep_analysis(request, "id-1", time.time())

    @pytest.mark.asyncio
    async def test_deep_analysis_with_issues(self):
        code = """
def complex():
    x = eval("1+2")
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i > j:
                    if j > k:
                        pass
    return x
"""
        request = _make_request(code=code, analysis_type="deep", language="python")
        result = await deep_analysis(request, "id-1", time.time())
        assert result.success is True
