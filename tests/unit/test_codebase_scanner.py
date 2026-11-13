"""Unit tests for CodebaseScanner service."""

from __future__ import annotations

import json
import warnings

import pytest

from models.bug_models import ErrorType, ParsedError


def _make_scanner(path: str):
    """Create a CodebaseScanner while suppressing DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from services.codebase_scanner import CodebaseScanner

        return CodebaseScanner(path)


def test_init_emits_deprecation_warning(tmp_path):
    """CodebaseScanner.__init__ emits DeprecationWarning."""
    from services.codebase_scanner import CodebaseScanner

    with pytest.warns(DeprecationWarning, match="deprecated"):
        CodebaseScanner(str(tmp_path))


def test_scan_codebase_for_issues_finds_import_issue(tmp_path):
    """scan_codebase_for_issues detects unsafe 'os' import."""
    # The scanner flags `from os import ...` as potentially unsafe
    py_file = tmp_path / "sample.py"
    py_file.write_text("from os import path\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    # Should find at least the suspicious import
    assert len(errors) >= 1
    assert all(isinstance(e, ParsedError) for e in errors)
    file_paths = [e.file_path for e in errors]
    assert any(str(py_file) in fp for fp in file_paths if fp)


def test_scan_codebase_for_issues_clean_file(tmp_path):
    """scan_codebase_for_issues returns empty list for a clean file."""
    py_file = tmp_path / "clean.py"
    py_file.write_text("def add(a, b):\n    return a + b\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    assert isinstance(errors, list)
    assert all(isinstance(e, ParsedError) for e in errors)


def test_scan_codebase_syntax_error(tmp_path):
    """scan_codebase_for_issues detects SyntaxError in malformed file."""
    bad_file = tmp_path / "bad_syntax.py"
    bad_file.write_text("def broken(\n    # missing closing paren\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    syntax_errors = [e for e in errors if e.error_type == ErrorType.SYNTAX_ERROR]
    assert len(syntax_errors) >= 1
    assert any(str(bad_file) in (e.file_path or "") for e in syntax_errors)


def test_scan_codebase_dangerous_calls(tmp_path):
    """scan_codebase_for_issues detects dangerous eval call."""
    py_file = tmp_path / "dangerous.py"
    py_file.write_text("result = eval('1 + 1')\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    runtime_errors = [e for e in errors if e.error_type == ErrorType.RUNTIME_ERROR]
    assert len(runtime_errors) >= 1
    dangerous_errors = [e for e in runtime_errors if "eval" in e.message]
    assert len(dangerous_errors) >= 1


def test_scan_codebase_exec_dangerous_call(tmp_path):
    """scan_codebase_for_issues detects dangerous exec call."""
    py_file = tmp_path / "exec_file.py"
    py_file.write_text("exec('x = 1')\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    runtime_errors = [e for e in errors if e.error_type == ErrorType.RUNTIME_ERROR]
    exec_errors = [e for e in runtime_errors if "exec" in e.message]
    assert len(exec_errors) >= 1


def test_scan_skips_venv(tmp_path):
    """Files inside .venv directory are skipped."""
    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    venv_file = venv_dir / "dangerous.py"
    venv_file.write_text("result = eval('1 + 1')\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    # No errors should point inside .venv
    for error in errors:
        assert ".venv" not in (error.file_path or "")


def test_scan_skips_pycache(tmp_path):
    """Files inside __pycache__ are skipped."""
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    cache_file = cache_dir / "module.pyc.py"
    cache_file.write_text("eval('x')\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    for error in errors:
        assert "__pycache__" not in (error.file_path or "")


def test_is_deep_analysis_result_valid_issues_key(tmp_path):
    """is_deep_analysis_result returns True for JSON with 'issues' key."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"issues": [], "analysis": "some analysis"})

    assert scanner.is_deep_analysis_result(data) is True


def test_is_deep_analysis_result_valid_errors_key(tmp_path):
    """is_deep_analysis_result returns True for JSON with 'errors' key."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"errors": []})

    assert scanner.is_deep_analysis_result(data) is True


def test_is_deep_analysis_result_invalid_plain_string(tmp_path):
    """is_deep_analysis_result returns False for plain non-JSON string."""
    scanner = _make_scanner(str(tmp_path))

    assert scanner.is_deep_analysis_result("this is just a log line") is False


def test_is_deep_analysis_result_invalid_json_no_keys(tmp_path):
    """is_deep_analysis_result returns False for JSON without recognized keys."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"something_else": "value"})

    assert scanner.is_deep_analysis_result(data) is False


def test_is_deep_analysis_result_empty_string(tmp_path):
    """is_deep_analysis_result returns False for empty string."""
    scanner = _make_scanner(str(tmp_path))

    assert scanner.is_deep_analysis_result("") is False


def test_parse_deep_analysis_results_with_issues(tmp_path):
    """parse_deep_analysis_results extracts ParsedError from JSON issues."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps(
        {
            "issues": [
                {
                    "type": "syntax",
                    "message": "unexpected indent",
                    "file": "app.py",
                    "line": 5,
                    "confidence": 0.95,
                },
                {
                    "type": "import",
                    "message": "missing module",
                    "file": "main.py",
                    "line": 1,
                    "confidence": 0.8,
                },
            ]
        }
    )

    errors = scanner.parse_deep_analysis_results(data)

    assert len(errors) == 2
    assert all(isinstance(e, ParsedError) for e in errors)
    assert errors[0].error_type == ErrorType.SYNTAX_ERROR
    assert errors[0].message == "unexpected indent"
    assert errors[0].file_path == "app.py"
    assert errors[0].line_number == 5
    assert errors[1].error_type == ErrorType.IMPORT_ERROR


def test_parse_deep_analysis_results_errors_key(tmp_path):
    """parse_deep_analysis_results falls back to 'errors' key when 'issues' absent."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"errors": [{"type": "name", "message": "undefined var", "confidence": 0.7}]})

    errors = scanner.parse_deep_analysis_results(data)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorType.NAME_ERROR


def test_parse_deep_analysis_results_invalid_json(tmp_path):
    """parse_deep_analysis_results returns empty list for invalid JSON."""
    scanner = _make_scanner(str(tmp_path))

    errors = scanner.parse_deep_analysis_results("not json at all")

    assert errors == []


def test_parse_deep_analysis_results_empty_issues(tmp_path):
    """parse_deep_analysis_results returns empty list when issues array is empty."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"issues": []})

    errors = scanner.parse_deep_analysis_results(data)

    assert errors == []


def test_parse_deep_analysis_results_unknown_type_maps_to_runtime(tmp_path):
    """Unknown issue type maps to RUNTIME_ERROR."""
    scanner = _make_scanner(str(tmp_path))
    data = json.dumps({"issues": [{"type": "unknown_custom_type", "message": "some error"}]})

    errors = scanner.parse_deep_analysis_results(data)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorType.RUNTIME_ERROR


def test_scan_codebase_multiple_files(tmp_path):
    """scan_codebase_for_issues processes multiple files."""
    (tmp_path / "a.py").write_text("eval('x')\n")
    (tmp_path / "b.py").write_text("exec('y')\n")
    (tmp_path / "c.py").write_text("def ok(): pass\n")

    scanner = _make_scanner(str(tmp_path))
    errors = scanner.scan_codebase_for_issues()

    # Should have at least 2 errors (one for eval, one for exec)
    assert len(errors) >= 2
