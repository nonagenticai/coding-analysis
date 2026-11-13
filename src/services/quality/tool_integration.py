#!/usr/bin/env python3
"""Quality tool integration module.

Handles execution of quality analysis tools (ruff, ty, bandit, radon, etc.)
and parsing their output into structured results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from processing.utils.ruff_output import FORMAT_CODE, parse_ruff_json, ruff_failure
from processing.utils.ty_output import parse_ty_gitlab, ty_failure

from .quality_config import TOOL_TIMEOUTS
from .quality_models import AppToolResult, QualityServiceIssue

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes quality analysis tools and parses their output."""

    async def run_quality_tool(
        self, tool_name: str, tool_config: dict[str, Any], file_path: str, language: str
    ) -> AppToolResult:
        """Run a single quality tool and return structured results."""
        start_time = time.time()

        try:
            # Find project root for tools that need it
            current_dir = Path(file_path).parent
            project_root = current_dir
            for _ in range(5):
                if any(
                    (project_root / marker).exists()
                    for marker in [
                        ".git",
                        "setup.py",
                        "pyproject.toml",
                        "requirements.txt",
                        "package.json",
                    ]
                ):
                    break
                if project_root.parent == project_root:
                    break
                project_root = project_root.parent

            command = self._build_command(tool_name, tool_config, file_path, project_root)
            work_dir = self._get_work_dir(tool_name, file_path, project_root)
            env = self._build_env(file_path, project_root)
            tool_timeout = TOOL_TIMEOUTS.get(tool_name, TOOL_TIMEOUTS["default"])

            logger.debug(f"Running {tool_name}: {' '.join(command)} (timeout: {tool_timeout}s)")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=tool_timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"{tool_name} execution timed out after {tool_timeout} seconds")

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            exit_code = process.returncode or 0

            issues = await self._parse_tool_output(tool_name, stdout_str, stderr_str, language)
            execution_time = time.time() - start_time
            success = self._check_success(tool_name, exit_code, stdout_str, stderr_str)
            if not success and tool_name in ("ruff", "ruff-format", "ty"):
                # A run that did not measure the code (exit 2 / panic): its
                # rows (ty's ``io`` row, a config error) are not findings.
                issues = []

            if not success:
                logger.warning(
                    f"{tool_name} failed - exit_code: {exit_code}, "
                    f"stdout: {stdout_str[:200]}, stderr: {stderr_str[:200]}"
                )
            else:
                logger.info(
                    f"{tool_name} completed successfully in {execution_time:.2f}s "
                    f"with {len(issues)} issues"
                )

            return AppToolResult(
                tool=tool_name,
                success=success,
                issues=issues,
                execution_time=execution_time,
                output=stdout_str,
                error=stderr_str,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error running {tool_name}: {e}")
            return AppToolResult(
                tool=tool_name,
                success=False,
                issues=[],
                execution_time=execution_time,
                output="",
                error=str(e),
            )

    def _build_command(
        self,
        tool_name: str,
        tool_config: dict[str, Any],
        file_path: str,
        project_root: Path,
    ) -> list[str]:
        """Build the command line for a specific tool."""
        base_command = tool_config["command"].copy()

        if tool_name == "ty":
            # Isolated single-file analysis: the analysed project's dependencies
            # are not installed here, so unresolvable imports/attributes are
            # noise (the mypy runner this replaced disabled import/attr-defined
            # for the same reason). Every other ty rule is reported.
            command = base_command + [
                "--ignore",
                "unresolved-import",
                "--ignore",
                "unresolved-attribute",
            ]
            try:
                relative_path = Path(file_path).relative_to(project_root)
                command.append(str(relative_path))
            except ValueError:
                command.append(str(file_path))
            return command
        elif tool_name == "vulture":
            return base_command + ["--min-confidence=80", str(file_path)]
        elif tool_name == "bandit":
            return base_command + ["-q", str(file_path)]
        elif tool_name == "radon":
            return base_command + [str(Path(file_path).parent)]
        elif tool_name == "eslint":
            return base_command + [
                "--config",
                str(Path(file_path).parent / ".eslintrc.json"),
                "--no-ignore",
                str(file_path),
            ]
        else:
            return base_command + [str(file_path)]

    def _get_work_dir(self, tool_name: str, file_path: str, project_root: Path) -> str:
        """Determine working directory for tool execution."""
        if tool_name == "ty":
            return str(project_root)
        return str(Path(file_path).parent)

    def _build_env(self, file_path: str, project_root: Path) -> dict[str, str]:
        """Build environment variables for tool execution."""
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        new_paths = [str(project_root)]

        current = Path(file_path).parent
        while current != project_root and current != current.parent:
            new_paths.append(str(current))
            current = current.parent

        if pythonpath:
            new_paths.append(pythonpath)

        env["PYTHONPATH"] = ":".join(new_paths)
        return env

    def _check_success(
        self, tool_name: str, exit_code: int, stdout_str: str, stderr_str: str
    ) -> bool:
        """Determine if tool execution was successful based on exit code and output."""
        if tool_name == "ty":
            # exit 2 (config/argv/missing target) or a panic is a FAILURE,
            # never a clean run.
            return (
                ty_failure(exit_code, parse_ty_gitlab(stdout_str), stdout_str, stderr_str) is None
            )
        elif tool_name in ("ruff", "ruff-format"):
            diagnostics = parse_ruff_json(stdout_str)
            return ruff_failure(exit_code, diagnostics, stdout_str, stderr_str) is None
        elif tool_name == "prettier":
            syntax_error = "SyntaxError: Unexpected token" in stdout_str
            return exit_code in [0, 1] or (exit_code == 2 and syntax_error)
        elif tool_name == "bandit":
            return exit_code in [0, 1] and not stderr_str.strip()
        elif tool_name == "vulture":
            return exit_code in [0, 1]
        elif tool_name == "eslint":
            config_error = "couldn't find a configuration file" in stderr_str.lower()
            return exit_code in [0, 1] and not config_error
        else:
            return exit_code == 0

    async def _parse_tool_output(
        self, tool_name: str, stdout: str, stderr: str, language: str
    ) -> list[QualityServiceIssue]:
        """Parse tool output to extract quality issues."""
        issues: list[QualityServiceIssue] = []

        try:
            if tool_name in ("ruff", "ruff-format") and stdout:
                issues = self._parse_ruff(tool_name, stdout)
            elif tool_name == "ty":
                issues = self._parse_ty(stdout)
            elif tool_name == "bandit" and stdout:
                issues = self._parse_bandit(stdout)
            elif tool_name == "eslint" and stdout:
                issues = self._parse_eslint(stdout)
            elif tool_name == "prettier" and (stdout or stderr):
                issues = self._parse_formatter(tool_name, stdout, stderr)
            elif tool_name == "vulture":
                issues = self._parse_vulture(stdout, stderr)
            elif tool_name == "radon" and stdout:
                issues = self._parse_radon(stdout)
        except Exception as e:
            logger.warning(f"Error parsing {tool_name} output: {e}")

        return issues

    def _parse_ruff(self, tool_name: str, stdout: str) -> list[QualityServiceIssue]:
        """Parse ``ruff check`` / ``ruff format --check`` JSON output."""
        issues: list[QualityServiceIssue] = []
        for diag in parse_ruff_json(stdout) or []:
            if diag.code == FORMAT_CODE:
                severity, message = "info", f"{diag.file}: file would be reformatted"
            elif diag.code.startswith(("F", "E9", "PLE")) or diag.code == "invalid-syntax":
                severity, message = "error", diag.message
            elif diag.code.startswith(("E", "W", "PLW")):
                severity, message = "warning", diag.message
            else:
                severity, message = "info", diag.message
            issues.append(
                QualityServiceIssue(
                    tool=tool_name,
                    severity=severity,
                    message=message,
                    line=diag.line,
                    column=diag.column,
                    rule_id=diag.code,
                    fixable=diag.fixable or diag.code == FORMAT_CODE,
                )
            )
        return issues

    def _parse_ty(self, stdout: str) -> list[QualityServiceIssue]:
        """Parse ``ty check --output-format gitlab`` output; rule_id = ty rule name."""
        return [
            QualityServiceIssue(
                tool="ty",
                severity="warning" if diag.raw_severity == "minor" else "error",
                message=diag.message,
                line=diag.line,
                column=diag.column,
                rule_id=diag.rule,
                fixable=False,
            )
            for diag in parse_ty_gitlab(stdout) or []
        ]

    def _parse_bandit(self, stdout: str) -> list[QualityServiceIssue]:
        """Parse bandit JSON output."""
        issues: list[QualityServiceIssue] = []
        try:
            bandit_data = json.loads(stdout.strip())
            for result in bandit_data.get("results", []):
                issues.append(
                    QualityServiceIssue(
                        tool="bandit",
                        severity=result.get("issue_severity", "medium").lower(),
                        message=result.get("issue_text", ""),
                        line=result.get("line_number"),
                        rule_id=result.get("test_id"),
                        fixable=False,
                    )
                )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse bandit JSON output: {e}")
            issues.append(
                QualityServiceIssue(
                    tool="bandit",
                    severity="error",
                    message=f"Bandit failed to produce valid JSON output. Parse error: {e}",
                    fixable=False,
                )
            )
        return issues

    def _parse_eslint(self, stdout: str) -> list[QualityServiceIssue]:
        """Parse ESLint JSON output."""
        issues: list[QualityServiceIssue] = []
        try:
            eslint_data = json.loads(stdout)
            for file_result in eslint_data:
                for message in file_result.get("messages", []):
                    issues.append(
                        QualityServiceIssue(
                            tool="eslint",
                            severity="error" if message.get("severity") == 2 else "warning",
                            message=message.get("message", ""),
                            line=message.get("line"),
                            column=message.get("column"),
                            rule_id=message.get("ruleId"),
                            fixable=message.get("fix") is not None,
                        )
                    )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse eslint JSON output: {e}")
        return issues

    def _parse_formatter(
        self, tool_name: str, stdout: str, stderr: str
    ) -> list[QualityServiceIssue]:
        """Parse output from text formatters (prettier)."""
        issues: list[QualityServiceIssue] = []
        output_text = stdout + "\n" + stderr

        if tool_name == "prettier":
            if "SyntaxError: Unexpected token" in output_text:
                line_match = re.search(r"line (\d+)", output_text)
                line_num = int(line_match.group(1)) if line_match else None
                issues.append(
                    QualityServiceIssue(
                        tool=tool_name,
                        severity="error",
                        message="Prettier syntax error - file contains invalid syntax",
                        line=line_num,
                        fixable=False,
                    )
                )
            elif "[warn]" in output_text and "Code style issues found" in output_text:
                for line in output_text.split("\n"):
                    if line.startswith("[warn]") and "Code style issues found" not in line:
                        file_name = line.replace("[warn]", "").strip()
                        if file_name:
                            issues.append(
                                QualityServiceIssue(
                                    tool=tool_name,
                                    severity="info",
                                    message=f"File '{file_name}' has formatting issues",
                                    fixable=True,
                                )
                            )
            elif output_text.strip():
                issues.append(
                    QualityServiceIssue(
                        tool=tool_name,
                        severity="info",
                        message="Prettier found formatting issues.",
                        fixable=True,
                    )
                )
        return issues

    def _parse_vulture(self, stdout: str, stderr: str) -> list[QualityServiceIssue]:
        """Parse vulture dead code output."""
        issues: list[QualityServiceIssue] = []
        for line in (stdout + "\n" + stderr).split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                parts = line.split(":")
                if len(parts) >= 3:
                    try:
                        line_num = int(parts[1]) if parts[1].isdigit() else None
                        message = ":".join(parts[2:]).strip()
                        issues.append(
                            QualityServiceIssue(
                                tool="vulture",
                                severity="warning",
                                message=f"Dead code: {message}",
                                line=line_num,
                                fixable=False,
                            )
                        )
                    except (ValueError, IndexError):
                        pass
        return issues

    def _parse_radon(self, stdout: str) -> list[QualityServiceIssue]:
        """Parse radon complexity output."""
        issues: list[QualityServiceIssue] = []
        try:
            radon_data = json.loads(stdout)
            for _file_path, functions in radon_data.items():
                for func in functions:
                    complexity = func.get("complexity", 0)
                    if complexity > 10:
                        issues.append(
                            QualityServiceIssue(
                                tool="radon",
                                severity="warning" if complexity > 15 else "info",
                                message=f"High complexity: {func.get('name', 'unknown')} (complexity: {complexity})",
                                line=func.get("lineno"),
                                fixable=False,
                            )
                        )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse radon JSON output: {e}")
        return issues


# Global tool executor instance
tool_executor = ToolExecutor()
