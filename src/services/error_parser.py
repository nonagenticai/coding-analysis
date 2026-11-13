"""Error parser service for extracting structured errors from log output.

Supports Python tracebacks, JavaScript errors, Go panics, and generic error patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from models.bug_models import ErrorType, ParsedError

logger = logging.getLogger(__name__)

# Mapping from Python exception class names to ErrorType
_PYTHON_EXCEPTION_MAP: dict[str, ErrorType] = {
    "SyntaxError": ErrorType.SYNTAX_ERROR,
    "IndentationError": ErrorType.SYNTAX_ERROR,
    "TabError": ErrorType.SYNTAX_ERROR,
    "ImportError": ErrorType.IMPORT_ERROR,
    "ModuleNotFoundError": ErrorType.IMPORT_ERROR,
    "NameError": ErrorType.NAME_ERROR,
    "TypeError": ErrorType.TYPE_ERROR,
    "AttributeError": ErrorType.ATTRIBUTE_ERROR,
    "IndexError": ErrorType.INDEX_ERROR,
    "KeyError": ErrorType.KEY_ERROR,
    "ValueError": ErrorType.VALUE_ERROR,
    "RuntimeError": ErrorType.RUNTIME_ERROR,
    "RecursionError": ErrorType.RUNTIME_ERROR,
    "StopIteration": ErrorType.RUNTIME_ERROR,
    "OSError": ErrorType.RUNTIME_ERROR,
    "IOError": ErrorType.RUNTIME_ERROR,
    "FileNotFoundError": ErrorType.RUNTIME_ERROR,
    "PermissionError": ErrorType.RUNTIME_ERROR,
    "ZeroDivisionError": ErrorType.RUNTIME_ERROR,
    "OverflowError": ErrorType.RUNTIME_ERROR,
    "AssertionError": ErrorType.LOGIC_ERROR,
}

# Mapping from JavaScript error names to ErrorType
_JS_ERROR_MAP: dict[str, ErrorType] = {
    "SyntaxError": ErrorType.SYNTAX_ERROR,
    "TypeError": ErrorType.TYPE_ERROR,
    "ReferenceError": ErrorType.NAME_ERROR,
    "RangeError": ErrorType.INDEX_ERROR,
    "URIError": ErrorType.VALUE_ERROR,
    "EvalError": ErrorType.RUNTIME_ERROR,
    "Error": ErrorType.RUNTIME_ERROR,
}

# Python traceback frame pattern: File "path", line N, in func
_PY_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+))?',
)

# Python exception line: ExceptionType: message
_PY_EXCEPTION_RE = re.compile(
    r"^(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit))\s*:\s*(?P<msg>.*)",
)

# Bare exception (no message), e.g. "KeyboardInterrupt"
_PY_BARE_EXCEPTION_RE = re.compile(
    r"^(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*$",
)

# JS stack frame: at functionName (file:line:col) or at file:line:col
_JS_FRAME_RE = re.compile(
    r"^\s*at\s+(?:(?P<func>[^\s(]+)\s+)?\(?(?P<file>[^:)]+):(?P<line>\d+):(?P<col>\d+)\)?",
)

# JS error line: ErrorType: message
_JS_ERROR_RE = re.compile(
    r"^(?P<type>(?:Syntax|Type|Reference|Range|URI|Eval|)Error)\s*:\s*(?P<msg>.*)",
)

# Go panic pattern
_GO_PANIC_RE = re.compile(r"^panic:\s*(?P<msg>.*)", re.MULTILINE)

# Go goroutine header
_GO_GOROUTINE_RE = re.compile(r"^goroutine\s+\d+\s+\[(?P<state>[^\]]+)\]:", re.MULTILINE)

# Go stack frame: package.Function(args)\n\tfile.go:line +offset
_GO_FRAME_RE = re.compile(
    r"^(?P<func>[^\s(]+)\((?P<args>[^)]*)\)\s*\n\t(?P<file>[^\s:]+):(?P<line>\d+)",
    re.MULTILINE,
)

# Generic error pattern
_GENERIC_ERROR_RE = re.compile(
    r"^.*(?:ERROR|Error)\s*[:\-]\s*(?P<msg>.+)",
    re.MULTILINE,
)


class ErrorParser:
    """Parses error logs into structured error objects."""

    def __init__(self, codebase_path: str) -> None:
        self.codebase_path = codebase_path

    def parse_error_logs(self, error_logs: str) -> list[ParsedError]:
        """Parse error logs into structured errors.

        Detects Python tracebacks, JavaScript errors, Go panics, and generic errors.
        Returns an empty list for empty or whitespace-only input.
        """
        if not error_logs or not error_logs.strip():
            return []

        errors: list[ParsedError] = []

        # Try each parser in order of specificity
        py_errors = self._parse_python_tracebacks(error_logs)
        errors.extend(py_errors)

        js_errors = self._parse_javascript_errors(error_logs)
        errors.extend(js_errors)

        go_errors = self._parse_go_panics(error_logs)
        errors.extend(go_errors)

        # Only parse generic errors from lines not already covered
        if not errors:
            generic_errors = self._parse_generic_errors(error_logs)
            errors.extend(generic_errors)

        logger.info("Parsed %d errors from logs", len(errors))
        return errors

    def _parse_python_tracebacks(self, text: str) -> list[ParsedError]:
        """Extract Python traceback blocks from text."""
        errors: list[ParsedError] = []

        # Split on traceback headers
        parts = re.split(r"(Traceback \(most recent call last\):)", text)

        # parts will be [before, header, body, header, body, ...]
        i = 1
        while i < len(parts) - 1:
            if parts[i].strip() == "Traceback (most recent call last):":
                block = parts[i + 1]
                parsed = self._parse_single_python_traceback(block)
                if parsed:
                    errors.append(parsed)
                i += 2
            else:
                i += 1

        return errors

    def _parse_single_python_traceback(self, block: str) -> ParsedError | None:
        """Parse a single Python traceback block (everything after the header)."""
        lines = block.split("\n")
        stack_trace: list[dict[str, Any]] = []
        file_path: str | None = None
        line_number: int | None = None
        function_name: str | None = None
        error_type = ErrorType.RUNTIME_ERROR
        message = ""

        for line in lines:
            frame_match = _PY_FRAME_RE.match(line)
            if frame_match:
                frame: dict[str, Any] = {
                    "file": frame_match.group("file"),
                    "line": int(frame_match.group("line")),
                }
                if frame_match.group("func"):
                    frame["function"] = frame_match.group("func")
                stack_trace.append(frame)
                # Keep the last frame as the error location
                file_path = frame_match.group("file")
                line_number = int(frame_match.group("line"))
                function_name = frame_match.group("func")
                continue

            exc_match = _PY_EXCEPTION_RE.match(line.strip())
            if exc_match:
                exc_name = exc_match.group("exc").split(".")[-1]
                message = exc_match.group("msg").strip()
                error_type = _PYTHON_EXCEPTION_MAP.get(exc_name, ErrorType.RUNTIME_ERROR)
                break

            bare_match = _PY_BARE_EXCEPTION_RE.match(line.strip())
            if bare_match:
                exc_name = bare_match.group("exc").split(".")[-1]
                message = exc_name
                error_type = _PYTHON_EXCEPTION_MAP.get(exc_name, ErrorType.RUNTIME_ERROR)
                break

        if not message and not stack_trace:
            return None

        return ParsedError(
            error_type=error_type,
            message=message,
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            stack_trace=stack_trace,
            confidence=0.9 if stack_trace else 0.7,
        )

    def _parse_javascript_errors(self, text: str) -> list[ParsedError]:
        """Extract JavaScript-style errors from text."""
        errors: list[ParsedError] = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            js_match = _JS_ERROR_RE.match(line)
            if js_match:
                error_name = js_match.group("type")
                message = js_match.group("msg").strip()
                error_type = _JS_ERROR_MAP.get(error_name, ErrorType.RUNTIME_ERROR)

                # Collect stack frames following this error line
                stack_trace: list[dict[str, Any]] = []
                file_path: str | None = None
                line_number: int | None = None
                function_name: str | None = None

                j = i + 1
                while j < len(lines):
                    frame_match = _JS_FRAME_RE.match(lines[j])
                    if frame_match:
                        frame: dict[str, Any] = {
                            "file": frame_match.group("file"),
                            "line": int(frame_match.group("line")),
                            "column": int(frame_match.group("col")),
                        }
                        if frame_match.group("func"):
                            frame["function"] = frame_match.group("func")
                        stack_trace.append(frame)
                        # First frame is the error location
                        if file_path is None:
                            file_path = frame_match.group("file")
                            line_number = int(frame_match.group("line"))
                            function_name = frame_match.group("func")
                        j += 1
                    else:
                        break

                # Only add if this doesn't look like a Python traceback exception line
                # (Python tracebacks also have "TypeError:" etc.)
                is_in_python_traceback = False
                if i > 0:
                    for k in range(max(0, i - 20), i):
                        if "Traceback (most recent call last):" in lines[k]:
                            is_in_python_traceback = True
                            break

                if not is_in_python_traceback:
                    errors.append(
                        ParsedError(
                            error_type=error_type,
                            message=message,
                            file_path=file_path,
                            line_number=line_number,
                            function_name=function_name,
                            stack_trace=stack_trace,
                            confidence=0.85 if stack_trace else 0.6,
                        )
                    )

                i = j
            else:
                i += 1

        return errors

    def _parse_go_panics(self, text: str) -> list[ParsedError]:
        """Extract Go panic blocks from text."""
        errors: list[ParsedError] = []

        panic_matches = list(_GO_PANIC_RE.finditer(text))
        for panic_match in panic_matches:
            message = panic_match.group("msg").strip()

            # Find stack frames after the panic
            after_panic = text[panic_match.end() :]
            stack_trace: list[dict[str, Any]] = []
            file_path: str | None = None
            line_number: int | None = None
            function_name: str | None = None

            for frame_match in _GO_FRAME_RE.finditer(after_panic):
                frame: dict[str, Any] = {
                    "function": frame_match.group("func"),
                    "file": frame_match.group("file"),
                    "line": int(frame_match.group("line")),
                }
                stack_trace.append(frame)
                if file_path is None:
                    file_path = frame_match.group("file")
                    line_number = int(frame_match.group("line"))
                    function_name = frame_match.group("func")

            errors.append(
                ParsedError(
                    error_type=ErrorType.RUNTIME_ERROR,
                    message=message,
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    stack_trace=stack_trace,
                    confidence=0.85 if stack_trace else 0.7,
                )
            )

        return errors

    def _parse_generic_errors(self, text: str) -> list[ParsedError]:
        """Extract generic error patterns from text."""
        errors: list[ParsedError] = []
        seen_messages: set[str] = set()

        for match in _GENERIC_ERROR_RE.finditer(text):
            message = match.group("msg").strip()
            if not message or message in seen_messages:
                continue
            seen_messages.add(message)

            errors.append(
                ParsedError(
                    error_type=ErrorType.RUNTIME_ERROR,
                    message=message,
                    file_path=None,
                    line_number=None,
                    function_name=None,
                    stack_trace=[],
                    confidence=0.4,
                )
            )

        return errors
