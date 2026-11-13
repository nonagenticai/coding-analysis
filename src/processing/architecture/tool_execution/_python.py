"""Python language tool runners (ruff check, ruff format, ty, bandit, radon)."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue
from processing.architecture.tool_execution import _base
from processing.utils.ruff_output import (
    FORMAT_CODE,
    parse_ruff_concise,
    ruff_failure,
    ruff_severity,
)
from processing.utils.ty_output import parse_ty_gitlab, ty_failure

logger = logging.getLogger(__name__)

# ruff replaces flake8 (E/W/F, ruff's default set), isort (I) and pylint (PL*).
# --extend-select keeps the analysed repo's own ruff config and adds the
# isort + pylint families those removed runners used to cover.
RUFF_EXTEND_SELECT = "I,PL"

# Both ruff runs use ``--output-format concise`` (one line per diagnostic), NOT
# json: the JSON rows embed every fix edit / the reformatted source and blow
# past coding-containers' 1 MiB-per-stream exec cap on mid-size repos (1.36 MB
# measured for 481 files), which truncates the report.


def _failed(result: dict[str, Any], reason: str, tool: str) -> dict[str, Any]:
    failed = dict(result)
    failed["success"] = False
    failed["error"] = reason
    logger.warning("%s did not produce a usable result: %s", tool, reason)
    return failed


class PythonToolsMixin:
    """Mixin providing Python static-analysis runners."""

    # The mixin calls these on the composed class.
    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_ruff(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run ``ruff check`` (flake8 + isort + pylint rules) via the per-repo container."""
        command = [
            "ruff",
            "check",
            str(repo_path),
            "--output-format=concise",
            "--extend-select",
            RUFF_EXTEND_SELECT,
        ]
        issues, result = await self._execute_tool_in_container(
            "ruff", command, repo_path, timeout=60
        )
        if not result.get("success"):
            return issues, result

        diagnostics = parse_ruff_concise(result.get("stdout"))
        failure = ruff_failure(
            result.get("exit_code"), diagnostics, result.get("stdout"), result.get("stderr")
        )
        if failure is not None:
            return [], _failed(result, failure, "ruff")

        for diag in diagnostics or []:
            issues.append(
                ArchitectureQualityIssue(
                    id=f"ruff-{diag.code}-{len(issues)}",
                    severity=self._ruff_severity_to_standard(diag.code),
                    message=diag.message,
                    rule_id=diag.code,
                    rule_name=diag.name,
                    file_path=diag.file,
                    line_number=diag.line,
                    column_number=diag.column or 1,
                    tool="ruff",
                    category="style",
                    description=diag.message,
                    resolve_suggestions=(
                        f"Auto-fixable: ruff check --fix --select {diag.code}"
                        if diag.fixable
                        else None
                    ),
                )
            )
        return issues, result

    def _ruff_severity_to_standard(self, code: str) -> str:
        """Convert Ruff rule codes to standard severity."""
        return ruff_severity(code)

    async def _run_ruff_format(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run ``ruff format --check`` via the per-repo container (replaces black)."""
        command = ["ruff", "format", "--check", "--output-format=concise", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "ruff-format", command, repo_path, timeout=60
        )
        if not result.get("success"):
            return issues, result

        diagnostics = parse_ruff_concise(result.get("stdout"))
        failure = ruff_failure(
            result.get("exit_code"),
            diagnostics,
            result.get("stdout"),
            result.get("stderr"),
            tool="ruff format",
        )
        if failure is not None:
            return [], _failed(result, failure, "ruff-format")

        for diag in diagnostics or []:
            if diag.code != FORMAT_CODE:
                continue
            issues.append(
                ArchitectureQualityIssue(
                    id=f"ruff-format-{len(issues)}",
                    severity="low",
                    message=f"File needs formatting: {diag.file}",
                    rule_id=FORMAT_CODE,
                    rule_name="ruff format",
                    file_path=diag.file,
                    line_number=diag.line,
                    column_number=None,
                    tool="ruff-format",
                    category="formatting",
                    description=f"File would be reformatted by ruff format: {diag.file}",
                    resolve_suggestions="ruff format <file>",
                )
            )
        return issues, result

    async def _run_ty(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run the ty type checker via the per-repo container.

        ``ty check --output-format gitlab`` emits one JSON array; rule_id is
        ty's rule name (``check_name``). A run that did not check the code
        (exit 2: bad config / argv / missing target; a panic; unparseable
        output) is a tool FAILURE (``success=False`` + ``error``, no findings),
        never a clean run -- see processing.utils.ty_output.
        """
        command = ["ty", "check", "--output-format", "gitlab", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "ty", command, repo_path, timeout=120
        )
        if not result.get("success"):
            return issues, result

        diagnostics = parse_ty_gitlab(result.get("stdout"))
        failure = ty_failure(
            result.get("exit_code"), diagnostics, result.get("stdout"), result.get("stderr")
        )
        if failure is not None:
            return [], _failed(result, failure, "ty")

        for diag in diagnostics or []:
            issues.append(
                ArchitectureQualityIssue(
                    id=f"ty-{diag.rule}-{len(issues)}",
                    severity=diag.severity,
                    message=diag.message,
                    rule_id=diag.rule,
                    rule_name=diag.rule,
                    file_path=diag.file,
                    line_number=diag.line,
                    column_number=diag.column or 1,
                    tool="ty",
                    category="type-safety",
                    description=diag.message,
                    external_url=f"https://docs.astral.sh/ty/reference/rules/#{diag.rule}",
                )
            )
        return issues, result

    async def _read_workspace_file(self, path: Path) -> str | None:
        """Return a workspace file's text via ``cat`` in the pod, None if unreadable."""
        try:
            resp = await _base.get_coding_containers_client().execute(
                container_name=self._container_name,
                command=["cat", str(path)],
                workdir=str(path.parent),
                timeout=15,
                tenant_id=self._tenant_id,
            )
        except Exception:  # probe only; fall back to bandit defaults
            logger.warning("bandit config probe failed for %s", path, exc_info=True)
            return None
        if resp.get("exit_code") != 0 or resp.get("truncated"):
            return None
        stdout = resp.get("stdout")
        return stdout if isinstance(stdout, str) else None

    async def _bandit_config_args(self, repo_path: Path) -> list[str]:
        """Bandit flags selecting the repo's own config, or [] when it has none.

        ``-c pyproject.toml`` is passed only when ``[tool.bandit]`` exists: bandit
        errors on a pyproject without that section. The runtime image is
        Python 3.14, where bandit reads TOML via stdlib ``tomllib`` (no extra).
        """
        pyproject = repo_path / "pyproject.toml"
        text = await self._read_workspace_file(pyproject)
        if text is not None:
            try:
                has_section = "bandit" in tomllib.loads(text).get("tool", {})
            except tomllib.TOMLDecodeError:
                has_section = False
            if has_section:
                return ["-c", str(pyproject)]
        ini = repo_path / ".bandit"
        if await self._read_workspace_file(ini) is not None:
            return ["--ini", str(ini)]
        return []

    async def _run_bandit(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Bandit security scanner via the per-repo container.

        The analysed repo's own bandit config is honoured (``[tool.bandit]`` in
        pyproject.toml, else a ``.bandit`` INI), so its ``exclude_dirs`` /
        ``skips`` decide what counts -- the same as the fleet CI archetype.
        """
        command = ["bandit", "-r", str(repo_path), "-f", "json"]
        command[1:1] = await self._bandit_config_args(repo_path)
        issues, result = await self._execute_tool_in_container(
            "bandit", command, repo_path, timeout=60
        )

        if result.get("success") and result.get("stdout"):
            report = self._safe_json(result["stdout"], dict)
            if report is not None:
                for issue_data in report.get("results", []):
                    if not isinstance(issue_data, dict):
                        continue
                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"bandit-{issue_data.get('test_id', 'unknown')}-{len(issues)}",
                            severity=issue_data.get("issue_severity", "medium").lower(),
                            message=issue_data.get("issue_text", ""),
                            rule_id=issue_data.get("test_id", ""),
                            rule_name=issue_data.get("test_name", ""),
                            file_path=issue_data.get("filename", ""),
                            line_number=issue_data.get("line_number", 1),
                            column_number=issue_data.get("col_offset", 1),
                            tool="bandit",
                            category="security",
                            description=issue_data.get("issue_text", ""),
                            external_url=issue_data.get("more_info", ""),
                        )
                    )

        return issues, result

    async def _run_radon(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Radon cyclomatic-complexity analyzer via the per-repo container.

        Only complexity >= C is reported (`-n C`); issues are raised for
        functions with cyclomatic complexity >= 10.
        """
        command = ["radon", "cc", str(repo_path), "-j", "-n", "C"]
        issues, result = await self._execute_tool_in_container(
            "radon", command, repo_path, timeout=60
        )

        if result.get("success") and result.get("stdout"):
            report = self._safe_json(result["stdout"], dict)
            if report is not None:
                for file_path, functions in report.items():
                    if not isinstance(functions, list):
                        continue
                    for func_data in functions:
                        if not isinstance(func_data, dict):
                            continue
                        if func_data.get("complexity", 0) >= 10:
                            issues.append(
                                ArchitectureQualityIssue(
                                    id=f"radon-cc-{len(issues)}",
                                    severity=(
                                        "high" if func_data["complexity"] >= 20 else "medium"
                                    ),
                                    message=(
                                        f"Function '{func_data['name']}' has complexity "
                                        f"{func_data['complexity']}"
                                    ),
                                    rule_id="complex-function",
                                    rule_name="High Complexity",
                                    file_path=file_path,
                                    line_number=func_data.get("lineno", 1),
                                    column_number=func_data.get("col_offset", 1),
                                    tool="radon",
                                    category="maintainability",
                                    description=(
                                        f"Cyclomatic complexity of {func_data['complexity']} "
                                        "is too high"
                                    ),
                                    resolve_suggestions=(
                                        "Consider breaking this function into smaller, "
                                        "more focused functions"
                                    ),
                                )
                            )

        return issues, result
