"""Rust tool runners (clippy, rustfmt, cargo-audit)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any

from models.architecture_models import ArchitectureQualityIssue
from processing.architecture.rust_gate import (
    CARGO_AUDIT_COMMAND,
    CLIPPY_COMMAND,
    RUSTFMT_COMMAND,
)

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\(B")
#: ``Diff in /abs/file.rs:12:`` (rustfmt >= 1.6), ``Diff in /abs/file.rs at line 12:``
#: or, oldest, ``Diff in file.rs:`` (no line: reported at line 1).
_RUSTFMT_DIFF_RE = re.compile(r"^Diff in (?P<file>.+?)(?::(?P<l1>\d+)| at line (?P<l2>\d+))?:\s*$")


_ANY_RUSTFMT_DIFF = re.compile(r"^Diff in ", re.MULTILINE)


def _repo_relative(path: str, base: str, repo_root: str) -> str:
    """``path`` (absolute, or relative to ``base``) relative to ``repo_root``.

    A path outside the repo (a registry crate) is returned absolute.
    """
    p = PurePosixPath(path)
    if not p.is_absolute():
        p = PurePosixPath(base) / p
    parts: list[str] = []
    for part in p.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    norm = PurePosixPath(*parts) if parts else PurePosixPath("/")
    try:
        return norm.relative_to(PurePosixPath(repo_root)).as_posix()
    except ValueError:
        return norm.as_posix()


def _clippy_severity(level: str, rule: str) -> str:
    if level in ("note", "help"):
        return "low"
    if rule.startswith("clippy::"):
        return "medium"
    # A rustc diagnostic: a hard compile error (E-code / no code) is high; a
    # rustc lint (unused_variables, ...) is medium.
    if not rule or re.fullmatch(r"E\d{4}", rule):
        return "high"
    return "medium"


def _in_repo_site(
    span: dict[str, Any], workspace_root: str, repo_root: str
) -> tuple[str, int, int | None] | None:
    """(repo-relative file, line, column) of the outermost in-repo site of *span*:
    the span itself, or the macro call site its ``expansion`` chain leads to."""
    site = None
    cur: dict[str, Any] | None = span
    while cur:
        span = cur
        file_path = _repo_relative(span.get("file_name") or "", workspace_root, repo_root)
        if not file_path.startswith("/"):
            site = (file_path, span.get("line_start") or 1, span.get("column_start"))
        cur = (span.get("expansion") or {}).get("span")
    return site


def parse_clippy_json(
    stdout: str, workspace_root: str, repo_root: str
) -> list[ArchitectureQualityIssue]:
    """Clippy ``--message-format=json`` -> findings (file, line, rule, severity, message).

    A lint in code shared by several targets (lib + lib test) is reported once
    per target by cargo; duplicates are folded. Span-less summaries
    ("aborting due to ...") are not findings.

    A diagnostic raised inside a macro expansion points at the macro's body (for
    a dependency's macro, a file in the cargo registry). It is reported at the
    outermost in-repo call site (``span.expansion``), so each invocation is its
    own finding; one with no location in the repo is not reported.
    """
    issues: list[ArchitectureQualityIssue] = []
    seen: set[tuple[str, int, int | None, str, str]] = set()
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        inner = msg.get("message") or {}
        spans = inner.get("spans") or []
        if not spans:
            continue
        primary = next((s for s in spans if s.get("is_primary")), spans[0])
        code = inner.get("code")
        rule = (code.get("code") or "") if isinstance(code, dict) else ""
        level = inner.get("level") or "warning"
        located = _in_repo_site(primary, workspace_root, repo_root)
        if located is None:
            continue
        file_path, line, col = located
        message = inner.get("message") or ""
        key = (file_path, line, col, rule, message)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            ArchitectureQualityIssue(
                id=f"clippy-{rule or 'rustc'}-{len(issues)}",
                severity=_clippy_severity(level, rule),
                message=message,
                rule_id=rule or "rustc",
                rule_name="Clippy lint" if rule.startswith("clippy::") else "rustc diagnostic",
                file_path=file_path,
                line_number=line,
                column_number=col,
                tool="clippy",
                category="style" if rule.startswith("clippy::") else "correctness",
                description=inner.get("rendered") or message,
            )
        )
    return issues


def parse_rustfmt_check(
    output: str, crate_dir: str, repo_root: str
) -> list[ArchitectureQualityIssue]:
    """``cargo fmt --all -- --check`` output -> one finding per diff hunk in the repo."""
    issues: list[ArchitectureQualityIssue] = []
    for raw in output.splitlines():
        match = _RUSTFMT_DIFF_RE.match(_ANSI_RE.sub("", raw).strip())
        if not match:
            continue
        file_path = _repo_relative(match.group("file"), crate_dir, repo_root)
        if file_path.startswith("/"):
            continue  # a path dependency outside the repo (``--all`` formats it too)
        line = int(match.group("l1") or match.group("l2") or 1)
        issues.append(
            ArchitectureQualityIssue(
                id=f"rustfmt-format-{len(issues)}",
                severity="low",
                message=f"Not rustfmt-formatted (diff hunk at line {line})",
                rule_id="rustfmt",
                rule_name="Rust formatting",
                file_path=file_path,
                line_number=line,
                column_number=None,
                tool="rustfmt",
                category="formatting",
                description=f"cargo fmt --check reports a diff in {file_path} at line {line}",
            )
        )
    return issues


def _mark_no_verdict(
    result: dict[str, Any], issues: list[ArchitectureQualityIssue], tool: str
) -> None:
    """A non-zero exit with no finding is a failed run, not a pass."""
    exit_code = result.get("exit_code", 0)
    if exit_code not in (0, None) and not issues:
        result["success"] = False
        result["error"] = (
            f"{tool} exited {exit_code} without a parseable finding: "
            f"{str(result.get('stderr') or '').strip()[-500:]}"
        )


class RustToolsMixin:
    """Mixin providing Rust static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    def _crate_dir(self, repo_path: Path) -> Path:
        """The crate/workspace dir cargo runs in: ``repo_path`` + the run's workdir."""
        workdir = getattr(self, "_workdir", None)
        return repo_path / workdir if workdir else repo_path

    async def _cargo_workspace_root(self, crate_dir: Path) -> str:
        """Absolute cargo workspace root of *crate_dir*.

        Clippy reports span paths relative to the WORKSPACE root, not the crate
        dir, so ``src/lib.rs`` of ``crates/core`` comes back as
        ``crates/core/src/lib.rs``. ``cargo metadata --no-deps`` resolves it
        without compiling; if it fails the crate dir is the best guess.
        """
        _, result = await self._execute_tool_in_container(
            "cargo-metadata",
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            crate_dir,
            timeout=60,
        )
        if result.get("success") and result.get("exit_code") == 0:
            try:
                root = json.loads(result.get("stdout") or "").get("workspace_root")
            except ValueError:
                root = None
            if root:
                return root
        return str(crate_dir)

    async def _run_clippy(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """``cargo clippy --workspace --all-targets --message-format=json`` in the
        run's crate dir.

        Every lint is a finding. A non-zero exit with no parseable finding (no
        Cargo.toml, resolver/network error) is a failed run, never a pass.
        """
        crate_dir = self._crate_dir(repo_path)
        ws_root = await self._cargo_workspace_root(crate_dir)
        issues, result = await self._execute_tool_in_container(
            "clippy", list(CLIPPY_COMMAND), crate_dir, timeout=300
        )
        if not result.get("success"):
            return issues, result
        issues = parse_clippy_json(result.get("stdout") or "", ws_root, str(repo_path))
        _mark_no_verdict(result, issues, "clippy")
        return issues, result

    async def _run_rustfmt(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """``cargo fmt --all -- --check`` in the run's crate dir.

        One finding per ``Diff in <file>:<line>:`` hunk in the repo. Exit 0 is
        clean; exit 1 with hunks is a formatting failure (the verdict), even when
        every hunk is in a path dependency outside the repo (the repo is then
        clean); exit non-zero with no hunk at all is a tool failure.
        """
        crate_dir = self._crate_dir(repo_path)
        issues, result = await self._execute_tool_in_container(
            "rustfmt", list(RUSTFMT_COMMAND), crate_dir, timeout=120
        )
        if not result.get("success"):
            return issues, result
        output = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
        issues = parse_rustfmt_check(output, str(crate_dir), str(repo_path))
        if not _ANY_RUSTFMT_DIFF.search(_ANSI_RE.sub("", output)):
            _mark_no_verdict(result, issues, "rustfmt")
        return issues, result

    async def _run_cargo_audit(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """``cargo audit --json`` in the run's crate dir; findings point at that
        crate's ``Cargo.lock`` (repo-relative)."""
        crate_dir = self._crate_dir(repo_path)
        issues, result = await self._execute_tool_in_container(
            "cargo-audit", list(CARGO_AUDIT_COMMAND), crate_dir, timeout=120
        )
        if not result.get("success"):
            return issues, result
        lock = _repo_relative("Cargo.lock", str(crate_dir), str(repo_path))
        issues = parse_cargo_audit_json(result.get("stdout") or "", lock)
        _mark_no_verdict(result, issues, "cargo-audit")
        return issues, result


#: Every cargo-audit finding gets this one severity: the report carries no
#: severity word (only a CVSS vector), and this is developer feedback.
CARGO_AUDIT_SEVERITY = "medium"


def parse_cargo_audit_json(stdout: str, lock_path: str) -> list[ArchitectureQualityIssue]:
    """``cargo audit --json`` -> one finding per vulnerable (advisory, package version).

    Unmaintained/yanked *warnings* are not vulnerabilities and are not reported.
    """
    try:
        data = json.loads(stdout)
    except ValueError:
        return []
    issues: list[ArchitectureQualityIssue] = []
    for vuln in data.get("vulnerabilities", {}).get("list", []):
        advisory = vuln.get("advisory", {})
        package = vuln.get("package", {})
        adv_id = advisory.get("id") or "unknown"
        title = advisory.get("title") or adv_id
        issues.append(
            ArchitectureQualityIssue(
                id=f"cargo-audit-{adv_id}-{len(issues)}",
                severity=CARGO_AUDIT_SEVERITY,
                message=f"{package.get('name', '?')} {package.get('version', '?')}: {title}",
                rule_id=adv_id,
                rule_name="Security Advisory",
                file_path=lock_path,
                line_number=1,
                column_number=None,
                tool="cargo-audit",
                category="security",
                description=advisory.get("description") or title,
                external_url=advisory.get("url"),
            )
        )
    return issues
