"""Base class for the architecture tool-execution dispatcher.

Holds the shared container-execution helper, the line-based output parser,
and the ``run_tool`` dispatcher that fans tool names out to language-specific
mixin methods.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from clients.coding_containers import get_coding_containers_client
from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)

# How a tool takes several scoped targets in place of its ``str(repo_path)``
# argument. Declared per tool; a tool absent from SCOPE_MODES is unscopable.
SEPARATE = "separate"  # each path its own positional argv element
COMMA_AFTER_FLAG = "comma_after_flag"  # one flag value, comma-joined (pmd -d a,b)
UNSCOPABLE = "unscopable"  # project-wide by construction; skipped when scoped
#: Runs project-wide even when scoped (it cannot take paths), and the analyzer
#: then keeps only the findings on the scoped paths. For tools whose whole-crate
#: run is the only meaningful one (cargo) and whose findings carry a file path.
FILTER_FINDINGS = "filter_findings"

SCOPE_MODES: dict[str, str] = {
    # Python
    "ruff": SEPARATE,
    "ruff-format": SEPARATE,
    "ty": SEPARATE,
    "bandit": SEPARATE,  # ``-r`` is a boolean flag; targets are positional
    "radon": SEPARATE,
    # JavaScript / TypeScript
    "eslint": SEPARATE,
    "prettier": SEPARATE,
    "jshint": SEPARATE,
    "tsc": UNSCOPABLE,  # tsconfig project, no path argument
    # Java
    "checkstyle": SEPARATE,
    "pmd": COMMA_AFTER_FLAG,  # ``-d`` takes a comma-separated list
    "spotbugs": UNSCOPABLE,  # analyses compiled classes, not source paths
    # C / C++
    "cppcheck": SEPARATE,
    "cpplint": SEPARATE,
    "clang-tidy": UNSCOPABLE,  # needs a compile database, not a directory
    # C#
    "dotnet-format": UNSCOPABLE,  # takes a project/solution, not files
    "roslynator": UNSCOPABLE,
    # Multi-language
    "semgrep": SEPARATE,
    # Go / Rust: ``./...`` and cargo are whole-module by construction.
    "golint": UNSCOPABLE,
    "go-vet": UNSCOPABLE,
    "golangci-lint": UNSCOPABLE,
    "gosec": UNSCOPABLE,
    "clippy": FILTER_FINDINGS,
    "rustfmt": FILTER_FINDINGS,
    "cargo-metadata": FILTER_FINDINGS,  # helper for clippy's span paths
    "cargo-audit": UNSCOPABLE,
}

#: THE tool dispatch table: tool name -> (runner method on ``ToolExecution``,
#: whether the runner takes the language). ``run_tool`` runs exactly these, and
#: the quality capability endpoints list only tools present here.
TOOL_DISPATCH: dict[str, tuple[str, bool]] = {
    # Python
    "ruff": ("_run_ruff", False),
    "ruff-format": ("_run_ruff_format", False),
    "ty": ("_run_ty", False),
    "bandit": ("_run_bandit", False),
    "radon": ("_run_radon", False),
    # JavaScript / TypeScript
    "eslint": ("_run_eslint", True),
    "tsc": ("_run_typescript_compiler", False),
    "prettier": ("_run_prettier", True),
    "jshint": ("_run_jshint", False),
    # Java
    "checkstyle": ("_run_checkstyle", False),
    "pmd": ("_run_pmd", False),
    "spotbugs": ("_run_spotbugs", False),
    # Go
    "golint": ("_run_golint", False),
    "go-vet": ("_run_go_vet", False),
    "golangci-lint": ("_run_golangci_lint", False),
    "gosec": ("_run_gosec", False),
    # Rust
    "clippy": ("_run_clippy", False),
    "rustfmt": ("_run_rustfmt", False),
    "cargo-audit": ("_run_cargo_audit", False),
    # C / C++
    "cppcheck": ("_run_cppcheck", False),
    "clang-tidy": ("_run_clang_tidy", False),
    "cpplint": ("_run_cpplint", False),
    # C#
    "dotnet-format": ("_run_dotnet_format", False),
    "roslynator": ("_run_roslynator", False),
    # Multi-language security
    "semgrep": ("_run_semgrep", True),
}

# Output a tool emits when a target does not exist or it analysed no files.
_MISSING_RE = re.compile(
    r"no such file or directory"
    r"|does not exist"
    r"|no files matching"
    r"|could not find or open"
    r"|can'?t open"
    r"|no python files"
    r"|there are no \.pyi?\b"
    r"|invalid scanning root"
    r"|no files (?:found|to (?:check|lint|analy[sz]e))",
    re.IGNORECASE,
)

# Tools whose stdout is plain text that never echoes source, so a free-text
# missing-target phrase there is the tool speaking, not the analysed code.
# (eslint's JSON carries ``source``, jshint's ``evidence``: those are matched on
# stderr only.) No count-based phrase such as "0 files" is matched -- a
# formatter's normal summary says "0 files would be left unchanged".
_TEXT_STDOUT_TOOLS = frozenset({"prettier", "cppcheck", "cpplint", "checkstyle"})


# ruff concise output for a target it could not read.
_RUFF_UNREADABLE_RE = re.compile(r"^.+?(?::\d+:\d+: E902 |: io: )", re.MULTILINE)


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


class BaseToolExecution:
    """Base for ``ToolExecution`` — shared helpers and the dispatcher."""

    def __init__(self, container_name: str = "", tenant_id: str | None = None) -> None:
        """Initialize the tool execution system.

        Args:
            container_name: Per-repo container to exec tools in. Normally left
                empty and set per-run by ``run_tool`` (each request analyses a
                different codebase, so the target container is request-scoped).
            tenant_id: Tenant that OWNS that container. Request-scoped for the
                same reason and normally set per-run by ``run_tool``.
        """
        self._container_name = container_name
        self._tenant_id = tenant_id
        # Path scope for the current run (repo-relative files/dirs), or None
        # for a whole-repo run. Request-scoped, rebound by every ``run_tool``.
        self._scope_paths: list[str] | None = None
        # Repo-relative directory a project tool (cargo) runs in, or None for
        # the repo root. Request-scoped, rebound by every ``run_tool``.
        self._workdir: str | None = None

    def _scope_command(
        self, tool_name: str, command: list[str], repo_path: Path
    ) -> list[str] | None:
        """Rewrite a tool argv so it targets only the scoped paths.

        Every path-taking tool names its target as the literal ``str(repo_path)``
        argument. How that one argument becomes several scoped paths is declared
        per tool in ``SCOPE_MODES`` -- never inferred from the argv, because the
        argv shape cannot tell ``bandit -r <dir>`` (``-r`` is a boolean, the
        paths are positional) from ``pmd -d <a,b>`` (a comma-list flag value).
        A tool with no declaration, a tool declared ``unscopable`` and a tool
        whose argv has no target argument (``go vet ./...``) all return
        ``None``: the caller skips it rather than analyse files nobody asked
        about, or point it at a joined path that does not exist.
        """
        if not self._scope_paths:
            return command
        mode = SCOPE_MODES.get(tool_name, UNSCOPABLE)
        if mode == FILTER_FINDINGS:
            return command
        target = str(repo_path)
        if mode == UNSCOPABLE or target not in command:
            return None
        scoped = [str(repo_path / p) for p in self._scope_paths]
        out: list[str] = []
        for arg in command:
            if arg != target:
                out.append(arg)
            elif mode == COMMA_AFTER_FLAG:
                out.append(",".join(scoped))
            else:
                out.extend(scoped)
        return out

    @staticmethod
    def _missing_target_signal(tool_name: str, stdout: str, stderr: str) -> str | None:
        """Return why a tool's output shows it did not analyse its targets.

        A tool handed a path that does not exist usually still exits 0/1 with
        an empty report -- a no-verdict that would otherwise be scored as a
        clean pass. Structured reports are inspected field-wise (their issue
        text and code snippets could quote any phrase); free-text patterns are
        matched against stderr, and against stdout only for tools whose stdout
        is a report and never echoes source.
        """
        if tool_name in ("ruff", "ruff-format"):
            # concise rows: ``path:1:1: E902 No such file ...`` (check) /
            # ``path: io: ...`` (format --check).
            if _RUFF_UNREADABLE_RE.search(stdout):
                return f"{tool_name} E902/io: a target could not be read"
        elif tool_name == "bandit":
            data = _json_or_none(stdout)
            if isinstance(data, dict):
                for err in data.get("errors") or []:
                    reason = str(err.get("reason", "")) if isinstance(err, dict) else str(err)
                    if _MISSING_RE.search(reason):
                        return f"bandit: {reason}"
        elif tool_name == "ty":
            data = _json_or_none(stdout)
            if isinstance(data, list) and any(
                isinstance(v, dict) and v.get("check_name") == "io" for v in data
            ):
                return "ty io: a target could not be read"
        elif tool_name == "radon":
            data = _json_or_none(stdout)
            if isinstance(data, dict):
                for fname, val in data.items():
                    if isinstance(val, dict) and val.get("error"):
                        return f"radon: {fname}: {val['error']}"
        match = _MISSING_RE.search(stderr)
        if match is None and tool_name in _TEXT_STDOUT_TOOLS:
            match = _MISSING_RE.search(stdout)
        if match is not None:
            return f"{tool_name}: {match.group(0)}"
        return None

    async def _execute_tool_in_container(
        self,
        tool_name: str,
        command: list[str],
        repo_path: Path,
        timeout: int = 60,
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Execute a tool via the coding-containers service.

        Args:
            tool_name: Human-readable tool name (for logging).
            command: Command list to execute.
            repo_path: Repository path inside the container.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple of (issues list, result dict).
        """
        issues: list[ArchitectureQualityIssue] = []
        result: dict[str, Any] = {"success": False}

        scoped_command = self._scope_command(tool_name, command, repo_path)
        if scoped_command is None:
            # Path-scoped run and this tool cannot be pointed at paths: it was
            # NOT run. ``skipped`` keeps it out of tools_attempted entirely.
            result["skipped"] = True
            result["error"] = f"{tool_name} cannot be path-scoped; skipped"
            return issues, result
        command = scoped_command

        client = get_coding_containers_client()
        # The exec is made AS the tenant that owns the container. Without
        # ``X-Tenant-Id``, coding-containers resolves the caller as this
        # service's own account and its BOLA guard answers 404 "Container not
        # found" in ~1.2 ms — so every tool "failed" and the report came back
        # inconclusive for a workspace that plainly existed. The tenant is
        # threaded from the authenticated request, never defaulted.
        resp = await client.execute(
            container_name=self._container_name,
            command=command,
            workdir=str(repo_path),
            timeout=timeout,
            tenant_id=self._tenant_id,
        )

        # exit_code < 0  → the exec call itself failed (httpx error, 400 from a
        #   non-whitelisted command, 404 for a missing container).
        # 126 / 127      → the shell ran but the tool is not executable / not
        #   found inside the container. Treat both as "the tool did not run" so
        #   the quality analyzer abstains instead of scoring a vacuous pass.
        exit_code = resp.get("exit_code", -1)
        if exit_code < 0 or exit_code in (126, 127):
            result["error"] = resp.get("stderr", "Container execution failed")
            return issues, result

        if resp.get("truncated"):
            # coding-containers caps each exec stream (1 MiB). A truncated
            # report is partial: parsing it as complete would silently drop
            # findings (or, for a JSON report, fail to parse). It is a tool
            # FAILURE, never a partial-as-complete success.
            result["error"] = (
                f"{tool_name} output truncated by coding-containers "
                f"({resp.get('bytes_omitted', 'unknown')} bytes omitted); report is incomplete"
            )
            result["exit_code"] = exit_code
            result["truncated"] = True
            return issues, result

        if self._scope_paths and SCOPE_MODES.get(tool_name) != FILTER_FINDINGS:
            missing = self._missing_target_signal(
                tool_name, resp.get("stdout", "") or "", resp.get("stderr", "") or ""
            )
            if missing is not None:
                # A scoped tool that could not find (or analysed none of) its
                # targets produced no verdict; reporting it as a 0-issue
                # success would be a vacuous pass.
                result["error"] = f"scoped target not analysed: {missing}"
                return issues, result

        result["success"] = True
        result["exit_code"] = resp.get("exit_code", -1)
        result["stdout"] = resp.get("stdout", "")
        result["stderr"] = resp.get("stderr", "")
        result["duration"] = resp.get("duration", 0.0)

        return issues, result

    @staticmethod
    def _safe_json(stdout: str, expected: type | tuple[type, ...]) -> Any | None:
        """Parse tool stdout as JSON, returning it only if its top-level type
        matches ``expected`` (dict/list).

        Tools sometimes emit a bare string/number that is valid JSON but not the
        expected shape (e.g. an error message), or a list where a dict was
        expected. Returning None lets callers skip cleanly instead of crashing
        on ``.get()`` / iteration (``'str' object has no attribute 'get'``).
        """
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, expected):
            logger.warning(
                "Unexpected tool output shape: expected %s, got %s",
                expected,
                type(data).__name__,
            )
            return None
        return data

    @staticmethod
    def _parse_line_issues(
        stdout: str,
        tool: str,
        category: str,
        severity: str = "medium",
    ) -> list[ArchitectureQualityIssue]:
        """Parse simple line-based tool output into issues.

        Expects lines in the form ``file:line: message`` or
        ``file:line:col: message``.
        """
        issues: list[ArchitectureQualityIssue] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 3)
            if len(parts) >= 3:
                file_path = parts[0].strip()
                line_no_str = parts[1].strip()
                message = parts[-1].strip()
                try:
                    line_number = int(line_no_str)
                except ValueError:
                    line_number = 1
                issues.append(
                    ArchitectureQualityIssue(
                        id=f"{tool}-line-{len(issues)}",
                        severity=severity,
                        message=message,
                        rule_id=f"{tool}-issue",
                        rule_name=f"{tool} issue",
                        file_path=file_path,
                        line_number=line_number,
                        column_number=None,
                        tool=tool,
                        category=category,
                        description=message,
                    )
                )
        return issues

    async def run_tool(
        self,
        tool_name: str,
        _command: list[str],
        repo_path: Path,
        language: str,
        container_name: str = "",
        tenant_id: str | None = None,
        scope_paths: list[str] | None = None,
        workdir: str | None = None,
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run a specific quality analysis tool.

        ``scope_paths`` (repo-relative files/dirs) narrows the tool to those
        paths; it is rebound on EVERY call (``None`` = whole repo) because this
        instance is reused across requests, so a scope can never leak into the
        next, unscoped run. A tool that cannot be scoped returns
        ``{"skipped": True}`` without executing.

        Every tool executes via ``coding-containers`` exec into the per-repo
        ``container_name`` — the repo is mounted only inside that container
        (``/app/data/repo``), never in this analysis pod. ``container_name`` is
        set request-scoped here so the per-language mixins (which use the shared
        ``_execute_tool_in_container``) target the right container.

        ``tenant_id`` is set the same way and for the same reason: the container
        is tenant-scoped, so the exec has to present the tenant that owns it or
        coding-containers' BOLA guard 404s it.
        """
        issues: list[ArchitectureQualityIssue] = []
        result: dict[str, Any] = {"success": False, "error": None}

        if container_name:
            # Container and owning tenant are rebound as a PAIR. Updating one
            # without the other would exec against tenant A's container while
            # asserting tenant B's identity — this instance is reused for every
            # tool in a run, so a half-update would outlive the request that
            # made it.
            self._container_name = container_name
            self._tenant_id = tenant_id
        elif tenant_id is not None:
            self._tenant_id = tenant_id
        self._scope_paths = list(scope_paths) if scope_paths else None
        self._workdir = workdir or None

        try:
            logger.info("Running %s on %s in %s", tool_name, repo_path, self._container_name)

            entry = TOOL_DISPATCH.get(tool_name)
            if entry is None:
                logger.warning("Unknown tool: %s", tool_name)
                result["error"] = f"no runner for tool {tool_name!r}"
            else:
                method_name, takes_language = entry
                runner = getattr(self, method_name)
                if takes_language:
                    issues, result = await runner(repo_path, language)
                else:
                    issues, result = await runner(repo_path)

        except (
            subprocess.SubprocessError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            # Defense-in-depth: a malformed tool output that slips past the
            # per-parser type guards must degrade *this* tool (success=False) and
            # let the others run — never abort the whole quality analysis.
            AttributeError,
            TypeError,
            KeyError,
        ) as e:
            logger.error("Error running %s: %s", tool_name, e)
            result["error"] = str(e)

        return issues, result
