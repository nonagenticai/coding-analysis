"""Dependency-free security analyzer for code-change validation.

Why this module exists
----------------------
``PysaAnalyzer`` shells out to ``pyre analyze``. ``pyre-check`` is not a
dependency of this service (see ``pyproject.toml``), so on every deployment that
analyzer either fails to start or returns ``_create_empty_result()`` — an empty
``security_issues`` list that is indistinguishable from "this code is clean".
That is the defect this module exists to close: the advanced-validation endpoint
needs a taint/security analyzer that genuinely runs wherever the service runs,
and an analyzer that *cannot* run must say ``not_computed`` rather than hand back
an empty result that reads as a pass.

What it reuses
--------------
No detection rules are re-implemented here. Both rule sets already ship in this
repo and already run without extra tooling:

* ``processing.analyzers.deep.security.analyze`` — the AST rules behind the deep
  analysis endpoints: ``eval()``/``exec()`` calls (critical), dangerous imports
  (``pickle``/``subprocess``/…, high) and hardcoded credentials (high). It is
  invoked through the same duck-typed collector protocol the deep analyzer uses
  (an object carrying ``python_files`` and ``issues``).
* ``python_test_suite.src.security_manager._DANGEROUS_PATTERNS`` — the shell and
  code-execution sink list this repo already treats as always dangerous
  "regardless of config". The AST pass above deliberately does not cover
  ``os.system(...)``-style attribute sinks; this list does.

Only the severity/recommendation mapping for that sink list, and the glue that
merges and de-duplicates the two passes, are new.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from processing.analyzers.deep import security as deep_security
from processing.analyzers.deep.base import CodeIssue

# Private in its home module, but it is the repo's single source of truth for
# "always dangerous" Python sinks — reused rather than copied so that a future
# addition there reaches this analyzer too.
from python_test_suite.src.security_manager import _DANGEROUS_PATTERNS

logger = logging.getLogger(__name__)

#: Classification for each entry of the reused ``_DANGEROUS_PATTERNS`` list:
#: ``pattern -> (issue type, severity, recommendation)``. A pattern that is added
#: upstream without an entry here still reports, at ``high``, via the fallback in
#: ``_classify_sink``.
_SINK_CLASSIFICATION: dict[str, tuple[str, str, str]] = {
    "os.system(": (
        "Command Injection",
        "high",
        "Use subprocess with shell=False and validate the argument list.",
    ),
    "subprocess.call(": (
        "Command Injection",
        "high",
        "Pass an argument list with shell=False and validate every element.",
    ),
    "subprocess.run(": (
        "Command Injection",
        "high",
        "Pass an argument list with shell=False and validate every element.",
    ),
    "subprocess.Popen(": (
        "Command Injection",
        "high",
        "Pass an argument list with shell=False and validate every element.",
    ),
    "eval(": (
        "Code Injection",
        "critical",
        "Never evaluate untrusted input; use ast.literal_eval or an explicit parser.",
    ),
    "exec(": (
        "Code Injection",
        "critical",
        "Never execute untrusted input; dispatch on an explicit allowlist instead.",
    ),
    "__import__(": (
        "Code Injection",
        "critical",
        "Import from an explicit allowlist rather than a caller-supplied name.",
    ),
}

_FALLBACK_SINK = (
    "Dangerous Call",
    "high",
    "Review this call against the service's security guidance.",
)


@dataclass
class _SingleFileCollector:
    """A minimal `IssueCollector` covering exactly one file.

    `deep.security.analyze` reads `python_files` and appends to `issues`; that
    surface is declared as `processing.analyzers.deep.base.IssueCollector`, and
    this satisfies it structurally. It exists so the deep rules can be reused
    over one proposed change without building a whole `DeepCodeAnalyzer`.
    """

    python_files: list[Path]
    issues: list[CodeIssue] = field(default_factory=list)


def _classify_sink(pattern: str) -> tuple[str, str, str]:
    """Classify one reused dangerous pattern."""
    return _SINK_CLASSIFICATION.get(pattern, _FALLBACK_SINK)


def _line_snippet(content: str, line_number: int) -> str:
    """Return the source line at ``line_number`` (1-based), trimmed."""
    lines = content.splitlines()
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1].strip()
    return ""


def analyze_file_security(file_path: str | Path, display_path: str | None = None) -> dict[str, Any]:
    """Run this repo's own security rule sets over one Python file.

    Args:
        file_path: Path to the Python file to analyze. The caller usually writes
            the proposed content to a temp file, so this is not the name the
            reader of the report cares about.
        display_path: The path to report issue locations against — the path in
            the code change. Defaults to ``file_path``'s name.

    Returns:
        A dict shaped like ``TaintAnalysisResult``:

        * ``{"status": "completed", "security_issues": [...]}`` when the rules ran.
        * ``{"status": "failed", "reason": ..., "security_issues": []}`` when the
          file could not be read or could not be parsed. A file that does not
          parse has NOT been checked, so it must never report as clean.
    """
    path = Path(file_path)
    reported_path = display_path or path.name

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Security analysis could not read %s: %s", path, e)
        return {
            "status": "failed",
            "reason": f"could not read file: {e}",
            "security_issues": [],
        }

    # A syntax error means neither pass can see the code. Report it instead of
    # returning an empty — i.e. apparently clean — result.
    try:
        ast.parse(content, filename=str(path))
    except SyntaxError as e:
        return {
            "status": "failed",
            "reason": f"file does not parse as Python: {e}",
            "security_issues": [],
        }

    issues: list[dict[str, Any]] = []
    # Lines already reported by the AST pass; the sink scan below is line-based
    # and would otherwise double-report e.g. an `eval()` call.
    ast_reported_lines: set[int] = set()

    # ── Pass 1: the deep analyzer's AST security rules ──────────────────────
    collector = _SingleFileCollector(python_files=[path])
    try:
        deep_security.analyze(collector)
    except (OSError, RuntimeError, ValueError) as e:  # pragma: no cover - defensive
        logger.warning("Deep security rules failed for %s: %s", path, e)
        return {
            "status": "failed",
            "reason": f"AST security rules failed: {e}",
            "security_issues": [],
        }

    for issue in collector.issues:
        line_number = issue.line_number or 0
        ast_reported_lines.add(line_number)
        severity = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
        issues.append(
            {
                "severity": severity,
                "type": issue.title,
                "location": f"{reported_path}:{line_number}",
                "line": line_number,
                "flow_path": [],
                "recommendation": issue.suggestion,
                "message": issue.description,
                "detector": "deep.security",
                "code_snippet": issue.code_snippet or "",
            }
        )

    # ── Pass 2: the reused always-dangerous sink list ───────────────────────
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line_number in ast_reported_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in _DANGEROUS_PATTERNS:
            if pattern not in line:
                continue
            issue_type, severity, recommendation = _classify_sink(pattern)
            issues.append(
                {
                    "severity": severity,
                    "type": issue_type,
                    "location": f"{reported_path}:{line_number}",
                    "line": line_number,
                    "flow_path": [],
                    "recommendation": recommendation,
                    "message": f"Dangerous call `{pattern.rstrip('(')}` reached by this change.",
                    "detector": "dangerous-patterns",
                    "code_snippet": _line_snippet(content, line_number),
                }
            )
            # One finding per line is enough to block it; keep the report readable.
            break

    return {"status": "completed", "security_issues": issues}
