"""Unit tests for ErrorParser service."""

from __future__ import annotations

import pytest

from models.bug_models import ErrorType, ParsedError
from services.error_parser import ErrorParser


@pytest.fixture
def parser() -> ErrorParser:
    return ErrorParser(codebase_path="/tmp/test-project")


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string(self, parser: ErrorParser):
        assert parser.parse_error_logs("") == []

    def test_whitespace_only(self, parser: ErrorParser):
        assert parser.parse_error_logs("   \n\t\n  ") == []

    def test_none_like_empty(self, parser: ErrorParser):
        # The method checks `not error_logs` first
        assert parser.parse_error_logs("") == []


# ---------------------------------------------------------------------------
# Python tracebacks
# ---------------------------------------------------------------------------


class TestPythonTracebacks:
    def test_import_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "app/main.py", line 3, in <module>\n'
            "    import requests\n"
            "ModuleNotFoundError: No module named 'requests'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == ErrorType.IMPORT_ERROR
        assert err.file_path == "app/main.py"
        assert err.line_number == 3
        assert err.function_name == "<module>"
        assert "requests" in err.message
        assert err.confidence > 0.5

    def test_key_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "data.py", line 10, in process\n'
            "    val = d['missing']\n"
            "KeyError: 'missing'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.KEY_ERROR
        assert errors[0].file_path == "data.py"
        assert errors[0].line_number == 10
        assert errors[0].function_name == "process"

    def test_type_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "calc.py", line 5, in add\n'
            "    return a + b\n"
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.TYPE_ERROR

    def test_name_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "script.py", line 2, in <module>\n'
            "    print(undefined_var)\n"
            "NameError: name 'undefined_var' is not defined\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.NAME_ERROR

    def test_attribute_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "app.py", line 7, in run\n'
            "    s.spllit(',')\n"
            "AttributeError: 'str' object has no attribute 'spllit'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.ATTRIBUTE_ERROR

    def test_index_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "util.py", line 15, in get_item\n'
            "    return items[99]\n"
            "IndexError: list index out of range\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.INDEX_ERROR

    def test_value_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "parse.py", line 8, in convert\n'
            "    return int('abc')\n"
            "ValueError: invalid literal for int() with base 10: 'abc'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.VALUE_ERROR

    def test_syntax_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "bad.py", line 1\n'
            "    def foo(\n"
            "          ^\n"
            "SyntaxError: unexpected EOF while parsing\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.SYNTAX_ERROR

    def test_runtime_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 20, in execute\n'
            "    raise RuntimeError('limit exceeded')\n"
            "RuntimeError: limit exceeded\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.RUNTIME_ERROR

    def test_multi_frame_stack_trace(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "main.py", line 10, in <module>\n'
            "    result = process(data)\n"
            '  File "process.py", line 5, in process\n'
            "    return transform(data)\n"
            '  File "transform.py", line 3, in transform\n'
            "    return data['key']\n"
            "KeyError: 'key'\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        err = errors[0]
        assert len(err.stack_trace) == 3
        # Last frame should be the error location
        assert err.file_path == "transform.py"
        assert err.line_number == 3
        assert err.function_name == "transform"

    def test_multiple_tracebacks(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1, in f\n'
            "    pass\n"
            "ValueError: bad value\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "b.py", line 2, in g\n'
            "    pass\n"
            "TypeError: wrong type\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 2
        assert errors[0].error_type == ErrorType.VALUE_ERROR
        assert errors[1].error_type == ErrorType.TYPE_ERROR


# ---------------------------------------------------------------------------
# JavaScript errors
# ---------------------------------------------------------------------------


class TestJavaScriptErrors:
    def test_type_error_with_stack(self, parser: ErrorParser):
        logs = (
            "TypeError: Cannot read properties of undefined (reading 'map')\n"
            "    at processData (/app/src/index.js:42:10)\n"
            "    at main (/app/src/index.js:10:5)\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == ErrorType.TYPE_ERROR
        assert "map" in err.message
        assert err.file_path == "/app/src/index.js"
        assert err.line_number == 42
        assert err.function_name == "processData"
        assert len(err.stack_trace) == 2

    def test_reference_error(self, parser: ErrorParser):
        logs = (
            "ReferenceError: myVar is not defined\n    at Object.<anonymous> (/app/test.js:3:1)\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.NAME_ERROR

    def test_syntax_error(self, parser: ErrorParser):
        logs = "SyntaxError: Unexpected token }\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.SYNTAX_ERROR

    def test_range_error(self, parser: ErrorParser):
        logs = "RangeError: Maximum call stack size exceeded\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.INDEX_ERROR


# ---------------------------------------------------------------------------
# Go panics
# ---------------------------------------------------------------------------


class TestGoPanics:
    def test_basic_panic(self, parser: ErrorParser):
        logs = (
            "panic: runtime error: index out of range [5] with length 3\n"
            "\n"
            "goroutine 1 [running]:\n"
            "main.process(0x1234)\n"
            "\t/app/main.go:42 +0x80\n"
            "main.main()\n"
            "\t/app/main.go:10 +0x40\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == ErrorType.RUNTIME_ERROR
        assert "index out of range" in err.message
        assert err.file_path == "/app/main.go"
        assert err.line_number == 42
        assert err.function_name == "main.process"

    def test_panic_without_stack(self, parser: ErrorParser):
        logs = "panic: something went wrong\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].message == "something went wrong"
        assert errors[0].error_type == ErrorType.RUNTIME_ERROR


# ---------------------------------------------------------------------------
# Generic errors
# ---------------------------------------------------------------------------


class TestGenericErrors:
    def test_generic_error_line(self, parser: ErrorParser):
        logs = "ERROR: Failed to connect to database\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.RUNTIME_ERROR
        assert "Failed to connect to database" in errors[0].message
        assert errors[0].confidence < 0.5

    def test_error_colon_pattern(self, parser: ErrorParser):
        logs = "Error: timeout waiting for response\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        assert "timeout" in errors[0].message

    def test_no_errors_in_normal_log(self, parser: ErrorParser):
        logs = "INFO: Application started successfully\nDEBUG: Processing request\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 0

    def test_dedup_generic_errors(self, parser: ErrorParser):
        logs = "ERROR: connection failed\nERROR: connection failed\nERROR: connection failed\n"
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Return type validation
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_returns_list_of_parsed_error(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in f\n'
            "    pass\n"
            "ValueError: bad\n"
        )
        result = parser.parse_error_logs(logs)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, ParsedError)

    def test_stack_trace_is_list_of_dicts(self, parser: ErrorParser):
        logs = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in f\n'
            "    pass\n"
            "ValueError: bad\n"
        )
        errors = parser.parse_error_logs(logs)
        assert len(errors) == 1
        for frame in errors[0].stack_trace:
            assert isinstance(frame, dict)
            assert "file" in frame
            assert "line" in frame
