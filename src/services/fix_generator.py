"""Fix generator service for producing code fixes based on parsed errors.

Pattern-matches on error types to generate targeted CodeFix suggestions
with confidence scores and test recommendations.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from pathlib import Path

from models.bug_models import CodeFix, ErrorType, ParsedError
from processing.languages import language_registry

logger = logging.getLogger(__name__)


def _resolve_language(error: ParsedError) -> str:
    """Resolve the source language for *error* from its ``file_path`` extension.

    Falls back to ``"python"`` when there is no file path (errors parsed from
    bare tracebacks, the ``requirements.txt`` fallback path) or when the
    extension is not recognised — this keeps the Python fix path byte-identical
    to its pre-multi-language behaviour and never sends a real Python error
    down a degraded path.
    """
    if not error.file_path:
        return "python"
    return language_registry.detect_language(Path(error.file_path)) or "python"


class FixGenerator:
    """Generates code fixes for identified errors."""

    def __init__(self, codebase_path: str) -> None:
        self.codebase_path = codebase_path

    def generate_fixes_for_error(self, error: ParsedError) -> list[CodeFix]:
        """Generate fix suggestions for a given error.

        Dispatches first on source language (derived from ``error.file_path``),
        then on error type. Python uses the full per-error-type template set;
        JavaScript/TypeScript and Go have targeted import/syntax templates;
        every other language and every un-templated (language, error-type) pair
        gracefully degrades to a generic, detected-but-unpatched suggestion.
        Returns a list of CodeFix objects, never empty.
        """
        language = _resolve_language(error)
        try:
            if language == "python":
                handler = _FIX_HANDLERS.get(error.error_type, _fix_generic)
                fixes = handler(self, error)
            else:
                lang_handlers = _LANG_FIX_HANDLERS.get(language, {})
                handler = lang_handlers.get(error.error_type)
                fixes = handler(self, error) if handler is not None else _fix_generic(self, error)
        except Exception:
            logger.exception("Fix generation failed for %s", error.error_type)
            fixes = [_fix_generic(self, error)]
        return fixes  # type: ignore[invalid-return-type]  # ty: ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Per-error-type fix generators
# ---------------------------------------------------------------------------


def _fix_import_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for ImportError / ModuleNotFoundError."""
    module_name = _extract_module_name(error.message)
    file_path = error.file_path or "requirements.txt"

    fixes: list[CodeFix] = []

    # Suggest adding to requirements.txt
    req_path = os.path.join(gen.codebase_path, "requirements.txt")
    if module_name:
        fixes.append(
            CodeFix(
                file_path=req_path,
                line_number=None,
                original_code="",
                fixed_code=f"{module_name}>=0.1.0\n",
                explanation=f"Add missing dependency '{module_name}' to requirements.txt",
                confidence=0.7,
                test_suggestion=f"Run 'pip install {module_name}' and verify the import succeeds",
            )
        )

    # Suggest fixing the import line itself
    if error.line_number and error.file_path:
        fixes.append(
            CodeFix(
                file_path=error.file_path,
                line_number=error.line_number,
                original_code=f"import {module_name}" if module_name else "",
                fixed_code=(
                    f"try:\n    import {module_name}\nexcept ImportError:\n"
                    f"    {module_name} = None  # Optional dependency"
                    if module_name
                    else "# Fix missing import"
                ),
                explanation="Wrap import in try/except to handle missing dependency gracefully",
                confidence=0.6,
                test_suggestion="Verify the module handles the missing import gracefully",
            )
        )

    if not fixes:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Install missing dependency",
                explanation=f"Import error: {error.message}. Install the missing package.",
                confidence=0.5,
                test_suggestion="Install the missing package and re-run",
            )
        )

    return fixes


def _fix_syntax_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for SyntaxError."""
    file_path = error.file_path or "<unknown>"
    msg = error.message.lower()
    fixes: list[CodeFix] = []

    if "expected ':'" in msg or "expected :" in msg or "missing :" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Add missing colon at end of statement",
                explanation="Missing colon after if/for/def/class statement",
                confidence=0.8,
                test_suggestion="Add ':' at the end of the statement and re-run",
            )
        )
    elif "unexpected eof" in msg or "eof" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Close unclosed bracket/parenthesis/string",
                explanation="Unexpected end of file - likely unclosed bracket, parenthesis, or string",
                confidence=0.7,
                test_suggestion="Check for unclosed brackets, parentheses, or strings",
            )
        )
    elif "invalid syntax" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Review syntax at this line",
                explanation=(
                    "Invalid syntax detected. Common causes: missing comma, "
                    "wrong operator, mismatched brackets."
                ),
                confidence=0.6,
                test_suggestion="Review the line for typos, missing commas, or wrong operators",
            )
        )
    else:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Fix syntax error",
                explanation=f"Syntax error: {error.message}",
                confidence=0.5,
                test_suggestion="Fix the syntax issue and re-run the script",
            )
        )

    return fixes


def _fix_attribute_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for AttributeError - suggest similar attribute names."""
    file_path = error.file_path or "<unknown>"

    # Try to extract the attribute name from the message
    # Typical: "'str' object has no attribute 'spllit'"
    attr_match = re.search(r"has no attribute '(\w+)'", error.message)
    type_match = re.search(r"'(\w+)' object", error.message)

    attr_name = attr_match.group(1) if attr_match else None
    type_name = type_match.group(1) if type_match else None

    fixes: list[CodeFix] = []

    if attr_name and type_name:
        # Suggest similar attribute names from known builtins
        known_attrs = _get_common_attrs(type_name)
        similar = difflib.get_close_matches(attr_name, known_attrs, n=3, cutoff=0.6)

        if similar:
            best = similar[0]
            fixes.append(
                CodeFix(
                    file_path=file_path,
                    line_number=error.line_number,
                    original_code=f".{attr_name}",
                    fixed_code=f".{best}",
                    explanation=f"'{attr_name}' is not an attribute of '{type_name}'. Did you mean '{best}'?",
                    confidence=0.8,
                    test_suggestion=f"Replace '.{attr_name}' with '.{best}' and verify behavior",
                )
            )

    if not fixes:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code=f"# Check attribute name: {error.message}",
                explanation=f"Attribute error: {error.message}. Verify the attribute exists on the object.",
                confidence=0.5,
                test_suggestion="Use dir(obj) to list available attributes and correct the name",
            )
        )

    return fixes


def _fix_name_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for NameError - suggest similar variable names or missing imports."""
    file_path = error.file_path or "<unknown>"

    name_match = re.search(r"name '(\w+)' is not defined", error.message)
    name = name_match.group(1) if name_match else None

    fixes: list[CodeFix] = []

    if name:
        # Check if it looks like a common module name
        common_modules = [
            "os",
            "sys",
            "re",
            "json",
            "math",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "typing",
            "logging",
            "dataclasses",
            "enum",
            "abc",
            "copy",
            "io",
            "csv",
            "requests",
            "numpy",
            "pandas",
            "pytest",
        ]
        similar_modules = difflib.get_close_matches(name, common_modules, n=1, cutoff=0.7)
        if name in common_modules or similar_modules:
            mod = name if name in common_modules else similar_modules[0]
            fixes.append(
                CodeFix(
                    file_path=file_path,
                    line_number=1,
                    original_code="",
                    fixed_code=f"import {mod}",
                    explanation=f"'{name}' is not defined. Add 'import {mod}' at the top of the file.",
                    confidence=0.75,
                    test_suggestion=f"Add 'import {mod}' and verify the code runs",
                )
            )

        # Generic suggestion for any undefined name
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code=name,
                fixed_code=f"# TODO: define or import '{name}'",
                explanation=f"Name '{name}' is not defined. Ensure it is imported or defined before use.",
                confidence=0.6,
                test_suggestion=f"Define '{name}' or import it before this line",
            )
        )
    else:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Fix undefined name reference",
                explanation=f"Name error: {error.message}",
                confidence=0.5,
                test_suggestion="Ensure all names are defined or imported before use",
            )
        )

    return fixes


def _fix_type_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for TypeError - suggest type conversions."""
    file_path = error.file_path or "<unknown>"
    msg = error.message.lower()
    fixes: list[CodeFix] = []

    if "can't multiply" in msg or "unsupported operand" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Convert operands to compatible types (e.g., int(), float(), str())",
                explanation="Type mismatch in operation. Convert operands to compatible types.",
                confidence=0.7,
                test_suggestion="Wrap operands with int(), float(), or str() as appropriate",
            )
        )
    elif "argument" in msg or "takes" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Check function signature and adjust arguments",
                explanation=(
                    "Wrong number or type of arguments passed to function. "
                    "Check the function signature."
                ),
                confidence=0.7,
                test_suggestion="Check the function signature and pass the correct arguments",
            )
        )
    elif "not callable" in msg:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Remove () or check variable is callable",
                explanation="Attempted to call a non-callable object. Remove '()' or check the variable.",
                confidence=0.7,
                test_suggestion="Verify the object is callable before calling it",
            )
        )
    else:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="# Add explicit type conversion",
                explanation=f"Type error: {error.message}. Add explicit type conversion.",
                confidence=0.6,
                test_suggestion="Add explicit type conversion (int(), str(), float()) where needed",
            )
        )

    return fixes


def _fix_key_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for KeyError - suggest .get() with default."""
    file_path = error.file_path or "<unknown>"

    # Extract the key from the message, e.g. KeyError: 'missing_key'
    key_match = re.search(r"['\"](\w+)['\"]", error.message)
    key_name = key_match.group(1) if key_match else "key"

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code=f"['{key_name}']",
            fixed_code=f".get('{key_name}', None)",
            explanation=(
                f"Key '{key_name}' not found in dictionary. "
                f"Use .get('{key_name}', default) to avoid KeyError."
            ),
            confidence=0.8,
            test_suggestion=(
                f"Replace dict['{key_name}'] with dict.get('{key_name}', default_value) "
                "and add a test for the missing key scenario"
            ),
        ),
    ]


def _fix_index_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for IndexError - suggest bounds checking."""
    file_path = error.file_path or "<unknown>"

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code="",
            fixed_code="if index < len(sequence):\n    value = sequence[index]",
            explanation="Index out of range. Add bounds checking before accessing by index.",
            confidence=0.75,
            test_suggestion="Add a bounds check (if idx < len(lst)) and test with empty and short lists",
        ),
    ]


def _fix_value_error(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for ValueError - suggest input validation."""
    file_path = error.file_path or "<unknown>"

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code="",
            fixed_code="# Add input validation before this operation",
            explanation=f"Value error: {error.message}. Validate inputs before the operation.",
            confidence=0.65,
            test_suggestion="Add input validation and test with edge-case values (empty, None, wrong type)",
        ),
    ]


def _fix_generic(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Generic fix suggestion for unhandled error types."""
    file_path = error.file_path or "<unknown>"

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code="",
            fixed_code="# Review and fix this error",
            explanation=f"Error ({error.error_type.value}): {error.message}. Review the code at this location.",
            confidence=0.5,
            test_suggestion="Review the error, apply the appropriate fix, and add a regression test",
        ),
    ]


# Dispatch table (Python — full per-error-type template set)
_FIX_HANDLERS: dict = {
    ErrorType.IMPORT_ERROR: _fix_import_error,
    ErrorType.SYNTAX_ERROR: _fix_syntax_error,
    ErrorType.ATTRIBUTE_ERROR: _fix_attribute_error,
    ErrorType.NAME_ERROR: _fix_name_error,
    ErrorType.TYPE_ERROR: _fix_type_error,
    ErrorType.KEY_ERROR: _fix_key_error,
    ErrorType.INDEX_ERROR: _fix_index_error,
    ErrorType.VALUE_ERROR: _fix_value_error,
}


# ---------------------------------------------------------------------------
# Per-language fix generators (non-Python)
#
# JavaScript/TypeScript and Go get targeted import/syntax templates — the two
# highest-value, language-specific error classes the multi-language scanner and
# ErrorParser actually produce. Other error types (and Rust/Java/C#/C++) fall
# through to ``_fix_generic`` — detected and reported, but not auto-patched.
# TODO(lang): per-language templates for Rust/Java/C#/C++ and for the remaining
# error types (attribute/name/type/...) are a follow-up; they need real
# per-ecosystem fix knowledge rather than half-built placeholders.
# ---------------------------------------------------------------------------


def _fix_import_error_js(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for a JavaScript/TypeScript missing-module error."""
    module_name = _extract_js_module_name(error.message)
    file_path = error.file_path or "package.json"
    fixes: list[CodeFix] = []

    if module_name:
        pkg_path = os.path.join(gen.codebase_path, "package.json")
        fixes.append(
            CodeFix(
                file_path=pkg_path,
                line_number=None,
                original_code="",
                fixed_code=f'"{module_name}": "^1.0.0"',
                explanation=(
                    f"Add missing dependency '{module_name}' to package.json "
                    f'"dependencies" (or run "npm install {module_name}")'
                ),
                confidence=0.7,
                test_suggestion=f"Run 'npm install {module_name}' and verify the import resolves",
            )
        )
        if error.line_number and error.file_path:
            fixes.append(
                CodeFix(
                    file_path=error.file_path,
                    line_number=error.line_number,
                    original_code="",
                    fixed_code=f"// Verify the import path/specifier for '{module_name}'",
                    explanation=(
                        "Module not found — confirm the package is installed and the "
                        "import specifier (path, casing, extension) is correct"
                    ),
                    confidence=0.6,
                    test_suggestion="Verify the import resolves after installing the dependency",
                )
            )

    if not fixes:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="// Install the missing module",
                explanation=f"Import error: {error.message}. Install the missing package.",
                confidence=0.5,
                test_suggestion="Install the missing package with npm/yarn/pnpm and re-run",
            )
        )

    return fixes


def _fix_syntax_error_js(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for a JavaScript/TypeScript syntax error."""
    file_path = error.file_path or "<unknown>"
    msg = error.message.lower()

    if "unexpected end" in msg or "eof" in msg:
        explanation = "Unexpected end of input — likely an unclosed brace, bracket, paren, or string/template literal"
        fixed_code = "// Close the unclosed brace/bracket/parenthesis/string"
        test_suggestion = "Check for unclosed { } [ ] ( ) or ` ' \" and re-run the linter"
        confidence = 0.7
    elif "unexpected token" in msg:
        explanation = (
            "Unexpected token — common causes: missing semicolon or comma, "
            "stray bracket, or a reserved word used as an identifier"
        )
        fixed_code = "// Review the token at this position (missing ; , or stray bracket)"
        test_suggestion = "Review the line for a missing ';'/',' or a misplaced bracket"
        confidence = 0.6
    else:
        explanation = f"Syntax error: {error.message}"
        fixed_code = "// Fix syntax error"
        test_suggestion = "Fix the syntax issue and re-run the linter/build"
        confidence = 0.5

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code="",
            fixed_code=fixed_code,
            explanation=explanation,
            confidence=confidence,
            test_suggestion=test_suggestion,
        ),
    ]


def _fix_import_error_go(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for a Go missing-package error."""
    module_name = _extract_go_module_name(error.message)
    file_path = error.file_path or "go.mod"
    fixes: list[CodeFix] = []

    if module_name:
        fixes.append(
            CodeFix(
                file_path=os.path.join(gen.codebase_path, "go.mod"),
                line_number=None,
                original_code="",
                fixed_code=f"require {module_name} latest",
                explanation=(
                    f"Add missing dependency '{module_name}' — run "
                    f"'go get {module_name}' to add it to go.mod and go.sum"
                ),
                confidence=0.7,
                test_suggestion=f"Run 'go get {module_name}' then 'go build ./...' to verify",
            )
        )
        if error.line_number and error.file_path:
            fixes.append(
                CodeFix(
                    file_path=error.file_path,
                    line_number=error.line_number,
                    original_code="",
                    fixed_code=f'// Verify the import path "{module_name}" is correct',
                    explanation=(
                        "Package not found — confirm the import path is spelled correctly "
                        "and the module is required in go.mod"
                    ),
                    confidence=0.6,
                    test_suggestion="Run 'go mod tidy' and verify the package resolves",
                )
            )

    if not fixes:
        fixes.append(
            CodeFix(
                file_path=file_path,
                line_number=error.line_number,
                original_code="",
                fixed_code="// Add the missing package",
                explanation=f"Import error: {error.message}. Add the missing package.",
                confidence=0.5,
                test_suggestion="Run 'go get <package>' and 'go mod tidy', then re-build",
            )
        )

    return fixes


def _fix_syntax_error_go(gen: FixGenerator, error: ParsedError) -> list[CodeFix]:
    """Fix for a Go syntax error."""
    file_path = error.file_path or "<unknown>"
    msg = error.message.lower()

    if "unexpected eof" in msg or "eof" in msg:
        explanation = "Unexpected end of file — likely an unclosed brace, bracket, or parenthesis"
        fixed_code = "// Close the unclosed brace/bracket/parenthesis"
        test_suggestion = "Check for unclosed { } [ ] ( ), then run 'gofmt' and 'go build'"
        confidence = 0.7
    elif "expected" in msg:
        explanation = (
            f"Parser expected a token that was missing ({error.message}) — "
            "common causes: missing brace/paren, missing ';', or a malformed declaration"
        )
        fixed_code = "// Add the expected token (brace/paren/semicolon) at this position"
        test_suggestion = "Run 'gofmt -e' on the file to pinpoint the missing token"
        confidence = 0.6
    else:
        explanation = f"Syntax error: {error.message}"
        fixed_code = "// Fix syntax error"
        test_suggestion = "Run 'gofmt -e' and 'go build ./...' to surface the issue"
        confidence = 0.5

    return [
        CodeFix(
            file_path=file_path,
            line_number=error.line_number,
            original_code="",
            fixed_code=fixed_code,
            explanation=explanation,
            confidence=confidence,
            test_suggestion=test_suggestion,
        ),
    ]


# Per-language dispatch table. A missing (language, error-type) entry falls
# through to ``_fix_generic`` — the issue is still reported, just not patched.
_LANG_FIX_HANDLERS: dict[str, dict] = {
    "javascript": {
        ErrorType.IMPORT_ERROR: _fix_import_error_js,
        ErrorType.SYNTAX_ERROR: _fix_syntax_error_js,
    },
    "typescript": {
        ErrorType.IMPORT_ERROR: _fix_import_error_js,
        ErrorType.SYNTAX_ERROR: _fix_syntax_error_js,
    },
    "go": {
        ErrorType.IMPORT_ERROR: _fix_import_error_go,
        ErrorType.SYNTAX_ERROR: _fix_syntax_error_go,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_module_name(message: str) -> str | None:
    """Extract module name from import error messages."""
    # "No module named 'foo'" or "No module named 'foo.bar'"
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
    if m:
        # Return the top-level package name
        return m.group(1).split(".")[0]

    # "cannot import name 'X' from 'Y'"
    m = re.search(r"from ['\"]([^'\"]+)['\"]", message)
    if m:
        return m.group(1).split(".")[0]

    return None


def _extract_js_module_name(message: str) -> str | None:
    """Extract a module specifier from a JavaScript/TypeScript import error."""
    # Node: "Cannot find module 'express'"
    # webpack: "Module not found: Error: Can't resolve 'lodash'"
    # TS: "Cannot find module 'react' or its corresponding type declarations."
    m = re.search(
        r"(?:Cannot find module|[Cc]an't resolve|Module not found)\D*['\"]([^'\"]+)['\"]", message
    )
    if m:
        return m.group(1)
    # Fallback: any single-quoted/double-quoted specifier in the message.
    m = re.search(r"['\"]([^'\"]+)['\"]", message)
    return m.group(1) if m else None


def _extract_go_module_name(message: str) -> str | None:
    """Extract a package import path from a Go missing-package error."""
    # "cannot find package "github.com/foo/bar" in any of:"
    # 'import "github.com/foo/bar": cannot find module'
    m = re.search(r'package "([^"]+)"', message)
    if m:
        return m.group(1)
    m = re.search(r'import "([^"]+)"', message)
    if m:
        return m.group(1)
    # "no required module provides package github.com/foo/bar"
    # "package github.com/foo/bar is not in std"
    m = re.search(r"package ([^\s\"]+/[^\s\"]+)", message)
    return m.group(1) if m else None


def _get_common_attrs(type_name: str) -> list[str]:
    """Return common attribute names for builtin types (for fuzzy matching)."""
    attrs: dict[str, list[str]] = {
        "str": [
            "capitalize",
            "casefold",
            "center",
            "count",
            "encode",
            "endswith",
            "expandtabs",
            "find",
            "format",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdecimal",
            "isdigit",
            "isidentifier",
            "islower",
            "isnumeric",
            "isprintable",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
        ],
        "list": [
            "append",
            "clear",
            "copy",
            "count",
            "extend",
            "index",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
        ],
        "dict": [
            "clear",
            "copy",
            "fromkeys",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
        ],
        "set": [
            "add",
            "clear",
            "copy",
            "difference",
            "discard",
            "intersection",
            "isdisjoint",
            "issubset",
            "issuperset",
            "pop",
            "remove",
            "symmetric_difference",
            "union",
            "update",
        ],
    }
    return attrs.get(type_name, [])
