"""Unit tests for FixGenerator service."""

from __future__ import annotations

import pytest

from models.bug_models import CodeFix, ErrorType, ParsedError
from services.fix_generator import FixGenerator


@pytest.fixture
def generator() -> FixGenerator:
    return FixGenerator(codebase_path="/tmp/test-project")


def _make_error(
    error_type: ErrorType,
    message: str = "",
    file_path: str | None = "src/app.py",
    line_number: int | None = 10,
    function_name: str | None = "run",
) -> ParsedError:
    """Helper to create a ParsedError with sensible defaults."""
    return ParsedError(
        error_type=error_type,
        message=message,
        file_path=file_path,
        line_number=line_number,
        function_name=function_name,
        stack_trace=[],
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Return type validation
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_returns_list_of_codefix(self, generator: FixGenerator):
        error = _make_error(ErrorType.IMPORT_ERROR, "No module named 'requests'")
        fixes = generator.generate_fixes_for_error(error)
        assert isinstance(fixes, list)
        assert len(fixes) > 0
        for fix in fixes:
            assert isinstance(fix, CodeFix)

    def test_fix_has_required_fields(self, generator: FixGenerator):
        error = _make_error(ErrorType.KEY_ERROR, "KeyError: 'missing'")
        fixes = generator.generate_fixes_for_error(error)
        fix = fixes[0]
        assert fix.file_path
        assert fix.explanation
        assert isinstance(fix.confidence, float)
        assert 0.0 <= fix.confidence <= 1.0


# ---------------------------------------------------------------------------
# IMPORT_ERROR
# ---------------------------------------------------------------------------


class TestImportError:
    def test_generates_requirements_fix(self, generator: FixGenerator):
        error = _make_error(ErrorType.IMPORT_ERROR, "No module named 'requests'")
        fixes = generator.generate_fixes_for_error(error)
        # Should have at least a requirements.txt fix
        req_fixes = [f for f in fixes if "requirements" in f.file_path]
        assert len(req_fixes) >= 1
        assert "requests" in req_fixes[0].fixed_code

    def test_extracts_module_from_message(self, generator: FixGenerator):
        error = _make_error(ErrorType.IMPORT_ERROR, "No module named 'pandas.core'")
        fixes = generator.generate_fixes_for_error(error)
        # Should extract top-level 'pandas'
        req_fixes = [f for f in fixes if "requirements" in f.file_path]
        assert any("pandas" in f.fixed_code for f in req_fixes)

    def test_has_test_suggestion(self, generator: FixGenerator):
        error = _make_error(ErrorType.IMPORT_ERROR, "No module named 'flask'")
        fixes = generator.generate_fixes_for_error(error)
        assert any(f.test_suggestion for f in fixes)

    def test_confidence_range(self, generator: FixGenerator):
        error = _make_error(ErrorType.IMPORT_ERROR, "No module named 'xyz'")
        fixes = generator.generate_fixes_for_error(error)
        for fix in fixes:
            assert 0.5 <= fix.confidence <= 0.9


# ---------------------------------------------------------------------------
# SYNTAX_ERROR
# ---------------------------------------------------------------------------


class TestSyntaxError:
    def test_missing_colon(self, generator: FixGenerator):
        error = _make_error(ErrorType.SYNTAX_ERROR, "expected ':'")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any("colon" in f.explanation.lower() for f in fixes)

    def test_unexpected_eof(self, generator: FixGenerator):
        error = _make_error(ErrorType.SYNTAX_ERROR, "unexpected EOF while parsing")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any(
            "bracket" in f.explanation.lower() or "unclosed" in f.explanation.lower() for f in fixes
        )

    def test_invalid_syntax(self, generator: FixGenerator):
        error = _make_error(ErrorType.SYNTAX_ERROR, "invalid syntax")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1

    def test_generic_syntax(self, generator: FixGenerator):
        error = _make_error(ErrorType.SYNTAX_ERROR, "some other syntax issue")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# ATTRIBUTE_ERROR
# ---------------------------------------------------------------------------


class TestAttributeError:
    def test_suggests_similar_attribute(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.ATTRIBUTE_ERROR,
            "'str' object has no attribute 'spllit'",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        # Should suggest 'split' as the closest match
        assert any("split" in f.fixed_code for f in fixes)

    def test_unknown_type_fallback(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.ATTRIBUTE_ERROR,
            "'MyClass' object has no attribute 'foo'",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1

    def test_no_attribute_in_message(self, generator: FixGenerator):
        error = _make_error(ErrorType.ATTRIBUTE_ERROR, "weird attribute error message")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# NAME_ERROR
# ---------------------------------------------------------------------------


class TestNameError:
    def test_suggests_import_for_known_module(self, generator: FixGenerator):
        error = _make_error(ErrorType.NAME_ERROR, "name 'os' is not defined")
        fixes = generator.generate_fixes_for_error(error)
        import_fixes = [f for f in fixes if "import os" in f.fixed_code]
        assert len(import_fixes) >= 1

    def test_suggests_define_for_unknown_name(self, generator: FixGenerator):
        error = _make_error(ErrorType.NAME_ERROR, "name 'my_var' is not defined")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any("my_var" in f.explanation for f in fixes)

    def test_generic_name_error(self, generator: FixGenerator):
        error = _make_error(ErrorType.NAME_ERROR, "something is not defined somehow")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# TYPE_ERROR
# ---------------------------------------------------------------------------


class TestTypeError:
    def test_unsupported_operand(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.TYPE_ERROR,
            "unsupported operand type(s) for +: 'int' and 'str'",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any(
            "convert" in f.explanation.lower() or "type" in f.explanation.lower() for f in fixes
        )

    def test_wrong_arguments(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.TYPE_ERROR,
            "foo() takes 2 positional arguments but 3 were given",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any("argument" in f.explanation.lower() for f in fixes)

    def test_not_callable(self, generator: FixGenerator):
        error = _make_error(ErrorType.TYPE_ERROR, "'int' object is not callable")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert any("callable" in f.explanation.lower() for f in fixes)

    def test_generic_type_error(self, generator: FixGenerator):
        error = _make_error(ErrorType.TYPE_ERROR, "something went wrong with types")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# KEY_ERROR
# ---------------------------------------------------------------------------


class TestKeyError:
    def test_suggests_get_method(self, generator: FixGenerator):
        error = _make_error(ErrorType.KEY_ERROR, "'missing_key'")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        fix = fixes[0]
        assert ".get(" in fix.fixed_code
        assert fix.confidence >= 0.7

    def test_key_in_explanation(self, generator: FixGenerator):
        error = _make_error(ErrorType.KEY_ERROR, "'user_id'")
        fixes = generator.generate_fixes_for_error(error)
        assert any("user_id" in f.explanation for f in fixes)


# ---------------------------------------------------------------------------
# INDEX_ERROR
# ---------------------------------------------------------------------------


class TestIndexError:
    def test_suggests_bounds_check(self, generator: FixGenerator):
        error = _make_error(ErrorType.INDEX_ERROR, "list index out of range")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        fix = fixes[0]
        assert "len(" in fix.fixed_code or "bounds" in fix.explanation.lower()
        assert fix.confidence >= 0.5

    def test_has_test_suggestion(self, generator: FixGenerator):
        error = _make_error(ErrorType.INDEX_ERROR, "list index out of range")
        fixes = generator.generate_fixes_for_error(error)
        assert fixes[0].test_suggestion


# ---------------------------------------------------------------------------
# VALUE_ERROR
# ---------------------------------------------------------------------------


class TestValueError:
    def test_suggests_validation(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.VALUE_ERROR,
            "invalid literal for int() with base 10: 'abc'",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        assert "valid" in fixes[0].explanation.lower()


# ---------------------------------------------------------------------------
# Generic / fallback
# ---------------------------------------------------------------------------


class TestGenericFix:
    def test_runtime_error(self, generator: FixGenerator):
        error = _make_error(ErrorType.RUNTIME_ERROR, "limit exceeded")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert fixes[0].confidence >= 0.4

    def test_logic_error(self, generator: FixGenerator):
        error = _make_error(ErrorType.LOGIC_ERROR, "assertion failed")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1

    def test_performance_error(self, generator: FixGenerator):
        error = _make_error(ErrorType.PERFORMANCE_ERROR, "slow query detected")
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_file_path(self, generator: FixGenerator):
        error = _make_error(ErrorType.KEY_ERROR, "'x'", file_path=None)
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert fixes[0].file_path  # Should fall back to "<unknown>"

    def test_no_line_number(self, generator: FixGenerator):
        error = _make_error(ErrorType.VALUE_ERROR, "bad value", line_number=None)
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# Multi-language dispatch (JavaScript/TypeScript, Go)
# ---------------------------------------------------------------------------


class TestMultiLanguageDispatch:
    def test_js_import_error_suggests_package_json(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.IMPORT_ERROR,
            "Cannot find module 'express'",
            file_path="src/server.js",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        pkg_fixes = [f for f in fixes if "package.json" in f.file_path]
        assert len(pkg_fixes) >= 1
        assert "express" in pkg_fixes[0].fixed_code
        assert "npm install express" in pkg_fixes[0].test_suggestion

    def test_ts_import_error_resolves_typescript(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.IMPORT_ERROR,
            "Cannot find module 'react' or its corresponding type declarations.",
            file_path="src/App.tsx",
        )
        fixes = generator.generate_fixes_for_error(error)
        pkg_fixes = [f for f in fixes if "package.json" in f.file_path]
        assert len(pkg_fixes) >= 1
        assert "react" in pkg_fixes[0].fixed_code

    def test_js_syntax_error_unclosed(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.SYNTAX_ERROR,
            "Unexpected end of input",
            file_path="src/app.js",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        assert "unclosed" in fixes[0].explanation.lower()

    def test_go_import_error_suggests_go_get(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.IMPORT_ERROR,
            'cannot find package "github.com/gin-gonic/gin" in any of:',
            file_path="main.go",
        )
        fixes = generator.generate_fixes_for_error(error)
        gomod_fixes = [f for f in fixes if "go.mod" in f.file_path]
        assert len(gomod_fixes) >= 1
        assert "github.com/gin-gonic/gin" in gomod_fixes[0].fixed_code
        assert "go get github.com/gin-gonic/gin" in gomod_fixes[0].test_suggestion

    def test_go_syntax_error_eof(self, generator: FixGenerator):
        error = _make_error(
            ErrorType.SYNTAX_ERROR,
            "unexpected EOF",
            file_path="handler.go",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        assert "gofmt" in fixes[0].test_suggestion

    def test_go_untemplated_type_error_degrades_gracefully(self, generator: FixGenerator):
        """A Go error type with no specific template still reports the issue."""
        error = _make_error(
            ErrorType.TYPE_ERROR,
            "mismatched types int and string",
            file_path="calc.go",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert "mismatched types" in fixes[0].explanation

    def test_rust_error_degrades_gracefully(self, generator: FixGenerator):
        """Rust has no templates yet — issue is detected but not patched."""
        error = _make_error(
            ErrorType.SYNTAX_ERROR,
            "expected one of `,` or `}`",
            file_path="src/lib.rs",
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) >= 1
        assert isinstance(fixes[0], CodeFix)

    def test_python_path_unchanged_for_py_files(self, generator: FixGenerator):
        """A .py file still goes through the full Python template set."""
        error = _make_error(
            ErrorType.IMPORT_ERROR,
            "No module named 'requests'",
            file_path="src/app.py",
        )
        fixes = generator.generate_fixes_for_error(error)
        req_fixes = [f for f in fixes if "requirements" in f.file_path]
        assert len(req_fixes) >= 1
        assert "requests" in req_fixes[0].fixed_code

    def test_no_file_path_defaults_to_python(self, generator: FixGenerator):
        """Errors with no file path keep the Python behaviour (bare tracebacks)."""
        error = _make_error(
            ErrorType.KEY_ERROR,
            "'missing'",
            file_path=None,
        )
        fixes = generator.generate_fixes_for_error(error)
        assert len(fixes) == 1
        assert ".get(" in fixes[0].fixed_code
