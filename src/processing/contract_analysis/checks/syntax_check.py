"""Python 3 syntax validation using ast.parse()."""

from __future__ import annotations

import ast
import re

from .types import Violation, ViolationSeverity

_PEP758_MSG = "multiple exception types must be parenthesized"


def _fix_pep758(content: str) -> str:
    """Auto-parenthesize bare except clauses for PEP 758 tolerance.

    Transforms: except A, B, C:
    Into:        except (A, B, C):  # fmt: skip
    """
    return re.sub(
        r"^(\s*except\s+)(\w[\w.,\s]+\w)(\s*:)",
        lambda m: (
            m.group(1) + "(" + m.group(2) + ")" + m.group(3)
            if "," in m.group(2) and "(" not in m.group(2)
            else m.group(0)
        ),
        content,
        flags=re.MULTILINE,
    )


def check_syntax(
    sources: dict[str, str],
) -> list[Violation]:
    """Validate that all source files are valid Python 3.

    Tolerates PEP 758 (Python 3.14) unparenthesized multi-except syntax.
    Uses Python's own ast.parse() -- the authoritative parser.
    """
    violations = []
    for filename, content in sources.items():
        try:
            ast.parse(content, filename=filename)
        except SyntaxError as e:
            if e.msg and _PEP758_MSG in e.msg:
                # PEP 758: valid in Python 3.14+, try with parenthesized version
                try:
                    ast.parse(_fix_pep758(content), filename=filename)
                    continue  # PEP 758 only -- skip
                except SyntaxError as e2:
                    e = e2  # Different error after fix -- report that instead
            violations.append(
                Violation(
                    severity=ViolationSeverity.FAIL,
                    check="python3_syntax",
                    message=f"SyntaxError in {filename} line {e.lineno}: {e.msg}",
                    service_location=f"{filename}:{e.lineno}",
                )
            )
    return violations
