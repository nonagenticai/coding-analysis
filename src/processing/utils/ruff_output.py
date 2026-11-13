"""ruff diagnostic parsing shared by every quality path that runs ruff.

``ruff check --output-format json`` and ``ruff format --check --output-format
json`` both emit a JSON array of::

    {"code": "F401", "message": "`sys` imported but unused", "name": "unused-import",
     "filename": "/abs/path.py", "location": {"row": 1, "column": 8},
     "end_location": {...}, "fix": {...} | null, "url": "...", "severity": "error"}

(``ruff format --check`` rows carry ``code == "unformatted"``.) The rule_id is
ruff's own code. ruff keeps the flake8 ``E``/``W``/``F`` codes, pylint maps to
``PL*`` (``PLC``/``PLE``/``PLR``/``PLW``), isort is ``I``.

Exit contract (ruff 0.17.3, fixtures in ``tests/fixtures/ruff``): ``0`` clean,
``1`` violations found / files would be reformatted, ``2`` abnormal
termination (bad config, bad argv, ``ruff format`` on a file it cannot parse).
Only 0, and 1 with at least one parsed row, measured the code.
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

FORMAT_CODE = "unformatted"

# ``--output-format concise`` row (ruff 0.17.3, ``ruff check`` and
# ``ruff format --check`` alike):
#   ``path:row:col: F401 [*] `os` imported but unused``   (lint code, ``[*]`` = fixable)
#   ``path:row:col: invalid-syntax: Expected an expression``  (named diagnostic)
#   ``path:row:col: unformatted: File would be reformatted``  (ruff format --check)
# Summary lines ("Found 3 errors.", "[*] 2 fixable ...", "All checks passed!",
# "1 file would be reformatted") have no ``:row:col:`` and are skipped. The
# path may contain spaces; it is the shortest prefix followed by ``:row:col: ``.
_CONCISE_ROW = re.compile(
    r"^(?P<file>.+?):(?P<row>\d+):(?P<col>\d+): "
    r"(?:(?P<code>[A-Z]+[0-9]+) (?P<fix>\[\*\] )?|(?P<name>[a-z][a-z0-9-]*): )"
    r"(?P<message>.*)$"
)


class RuffDiagnostic(NamedTuple):
    code: str
    message: str
    name: str
    file: str
    line: int
    column: int | None
    fixable: bool
    url: str


def ruff_severity(code: str) -> str:
    """Standard severity for a ruff code (flake8 + pylint conventions)."""
    if code.startswith("PLE") or code.startswith("E9") or code == "invalid-syntax":
        return "critical"
    if code.startswith("F"):  # pyflakes: undefined names, unused imports ...
        return "critical"
    if code.startswith("E"):
        return "high"
    if code.startswith(("W", "PLW")):
        return "medium"
    return "low"  # I, PLC, PLR, formatting, everything else


def parse_ruff_json(stdout: str | None) -> list[RuffDiagnostic] | None:
    """Parse ruff's JSON array. ``None`` when stdout is not that report."""
    try:
        data = json.loads(stdout or "")
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    out: list[RuffDiagnostic] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or row.get("name") or "").strip()
        if not code:
            continue
        location: dict[str, Any] = row.get("location") or {}
        line = location.get("row")
        column = location.get("column")
        out.append(
            RuffDiagnostic(
                code=code,
                message=str(row.get("message") or ""),
                name=str(row.get("name") or code),
                file=str(row.get("filename") or ""),
                line=line if isinstance(line, int) and line > 0 else 1,
                column=column if isinstance(column, int) and column > 0 else None,
                fixable=row.get("fix") is not None,
                url=str(row.get("url") or ""),
            )
        )
    return out


def parse_ruff_concise(stdout: str | None) -> list[RuffDiagnostic]:
    """Parse ruff's ``--output-format concise`` rows (summary lines skipped).

    Concise output is used on the container-exec path because the JSON format
    embeds every fix edit (and, for ``ruff format``, the reformatted source):
    1.36 MB for ``ruff check`` on a 481-file repo, over coding-containers'
    1 MiB-per-stream exec cap. Concise is one short line per diagnostic.
    """
    out: list[RuffDiagnostic] = []
    for raw in (stdout or "").splitlines():
        match = _CONCISE_ROW.match(raw.rstrip())
        if match is None:
            continue
        code = match.group("code") or match.group("name") or ""
        out.append(
            RuffDiagnostic(
                code=code,
                message=match.group("message"),
                name=code,
                file=match.group("file"),
                line=int(match.group("row")) or 1,
                column=int(match.group("col")) or None,
                fixable=match.group("fix") is not None,
                url="",
            )
        )
    return out


def ruff_failure(
    exit_code: int | None,
    diagnostics: list[RuffDiagnostic] | None,
    stdout: str | None = "",
    stderr: str | None = "",
    tool: str = "ruff",
) -> str | None:
    """Why this ruff run did not measure the code, or ``None`` if it did."""
    output = "\n".join(filter(None, (stderr, stdout))).strip()
    if exit_code == 0 and diagnostics is not None:
        return None
    if exit_code == 1 and diagnostics:
        return None
    if exit_code in (0, 1):
        reason = f"{tool} exited {exit_code} but its report could not be parsed"
    else:
        reason = f"{tool} exited {exit_code}: abnormal termination (config/usage/parse error)"
    return f"{reason}: {output[-2000:]}" if output else reason
