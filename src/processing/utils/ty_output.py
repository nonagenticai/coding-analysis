"""ty diagnostic parsing shared by every quality path that runs ty.

Every runner invokes ``ty check --output-format gitlab <paths>``. That format is
a JSON array (GitLab Code Quality report) of::

    {"check_name": "invalid-return-type",
     "description": "invalid-return-type: Return type does not match ...",
     "severity": "major",            # info | minor | major | critical | blocker
     "fingerprint": "<16 hex chars>",
     "location": {"path": "pkg/a.py",
                  "positions": {"begin": {"line": 4, "column": 12},
                                "end":   {"line": 4, "column": 13}}}}

A finding's identity downstream is keyed on ``tool + rule_id + file + unit``,
so the rule_id MUST be ty's own rule name (``check_name``: ``invalid-argument-type``,
``unresolved-import`` ...). A constant rule_id would collapse every ty
diagnostic in a unit onto one identity.

ty's exit contract (measured on ty 0.0.92, fixtures in ``tests/fixtures/ty``):

* ``0`` -- checked, clean (stdout ``[]``);
* ``1`` -- checked, diagnostics found (a syntax error is an ``invalid-syntax``
  diagnostic, exit 1 -- ty keeps checking the other files);
* ``2`` -- it did NOT check: an unparseable ``pyproject.toml`` / ``ty.toml``, an
  argparse rejection, or a target that does not exist (which still prints an
  ``io`` row on stdout -- that row is not a finding about the code);
* anything else (``101`` = a Rust panic) -- crashed.

Only 0, and 1 with at least one parsed diagnostic, measured the code. Every
other outcome is a tool FAILURE, never a clean run (see :func:`ty_failure`).
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

# ty's GitLab severities -> this service's standard severity scale.
_SEVERITY = {
    "blocker": "critical",
    "critical": "critical",
    "major": "high",  # ty `error`
    "minor": "medium",  # ty `warn`
    "info": "low",  # ty `info`
}


class TyDiagnostic(NamedTuple):
    rule: str
    message: str
    severity: str  # standard scale: critical/high/medium/low
    raw_severity: str  # ty's gitlab severity, verbatim
    file: str
    line: int
    column: int | None
    fingerprint: str


def ty_severity(raw: Any) -> str:
    """Map a ty GitLab severity onto the standard scale (unknown -> medium)."""
    return _SEVERITY.get(str(raw or "").strip().lower(), "medium")


def _message(rule: str, description: str) -> str:
    """Strip ty's ``"<rule>: "`` prefix from the description."""
    prefix = f"{rule}: "
    return description[len(prefix) :] if rule and description.startswith(prefix) else description


def _position(row: dict[str, Any]) -> tuple[int, int | None]:
    begin = ((row.get("location") or {}).get("positions") or {}).get("begin") or {}
    line = begin.get("line")
    column = begin.get("column")
    return (
        line if isinstance(line, int) and line > 0 else 1,
        column if isinstance(column, int) and column > 0 else None,
    )


def parse_ty_gitlab(stdout: str | None) -> list[TyDiagnostic] | None:
    """Parse ty's GitLab JSON. ``None`` when stdout is not that report at all.

    An empty list is a parsed, clean report (``[]``); ``None`` means nothing
    parseable came back, which a caller must never read as "clean".
    """
    try:
        data = json.loads(stdout or "")
    except (json.JSONDecodeError, ValueError):
        return None
    return parse_ty_rows(data)


def parse_ty_rows(data: Any) -> list[TyDiagnostic] | None:
    """Parse an already-decoded ty GitLab report (``None`` if not a list)."""
    if not isinstance(data, list):
        return None
    out: list[TyDiagnostic] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rule = str(row.get("check_name") or "").strip()
        description = str(row.get("description") or "")
        if not rule:
            # ty always names the rule; a row without one is not a diagnostic
            # this service can give an identity to.
            continue
        line, column = _position(row)
        path = (row.get("location") or {}).get("path")
        out.append(
            TyDiagnostic(
                rule=rule,
                message=_message(rule, description),
                severity=ty_severity(row.get("severity")),
                raw_severity=str(row.get("severity") or ""),
                file=str(path or ""),
                line=line,
                column=column,
                fingerprint=str(row.get("fingerprint") or ""),
            )
        )
    return out


def ty_failure(
    exit_code: int | None,
    diagnostics: list[TyDiagnostic] | None,
    stdout: str | None = "",
    stderr: str | None = "",
) -> str | None:
    """Why this ty run did not measure the code, or ``None`` if it did.

    ``0`` with a parseable report is clean; ``1`` needs at least one parsed
    diagnostic. Everything else -- exit 2 (bad config, bad argv, missing
    target), a panic, unparseable output -- is a failure carrying ty's own
    output so the operator can see why.
    """
    output = "\n".join(filter(None, (stderr, stdout))).strip()
    if exit_code == 0 and diagnostics is not None:
        return None
    if exit_code == 1 and diagnostics:
        return None
    if exit_code == 0:
        reason = "ty exited 0 but its report could not be parsed"
    elif exit_code == 1:
        reason = "ty exited 1 (diagnostics found) but no diagnostic could be parsed"
    else:
        reason = f"ty exited {exit_code}: it did not finish checking (crash/config/usage error)"
    return f"{reason}: {output[-2000:]}" if output else reason
