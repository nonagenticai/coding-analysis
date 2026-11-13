"""Security configuration for command execution."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected."""


@dataclass
class CommandConfig:
    """Configuration for secure command execution."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    timeout: int = 30
    allowed_commands: list[str] = field(default_factory=list)
    working_directory: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    max_output_size: int = 1024 * 1024  # 1MB


class SecureRunner:
    """Callable secure command runner that supports both call and run_command patterns."""

    def __call__(
        self,
        config: CommandConfig | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a command with security constraints.

        Args:
            config: Command configuration.
            command: Command to execute (used when config is None).
            args: Command arguments (used when config is None).
            timeout: Execution timeout in seconds.

        Returns:
            CompletedProcess with stdout/stderr.

        Raises:
            SecurityError: If command is not allowed or execution fails.
        """
        if config is not None:
            cmd = [config.command, *config.args] if config.command else []
            if not cmd and command is not None:
                cmd = [command, *(args or [])]
            eff_timeout = config.timeout
        elif command is not None:
            cmd = [command, *(args or [])]
            eff_timeout = timeout
        else:
            raise SecurityError("No command specified")

        return self._run(cmd, config, eff_timeout)

    def run_command(
        self,
        cmd: list[str],
        config: CommandConfig | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Execute cmd list with optional security config.

        Args:
            cmd: Full command and arguments as a list.
            config: Optional CommandConfig for security constraints.

        Returns:
            CompletedProcess with stdout/stderr.

        Raises:
            SecurityError: If command is not allowed or execution fails.
        """
        if config is not None and config.allowed_commands:
            if cmd[0] not in config.allowed_commands:
                raise SecurityError(
                    f"Command '{cmd[0]}' not in allowed commands: {config.allowed_commands}"
                )

        timeout = config.timeout if config is not None else 30
        return self._run(cmd, config, timeout)

    def _run(
        self,
        cmd: list[str],
        config: CommandConfig | None,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        """Internal execution helper."""
        if not cmd:
            raise SecurityError("Empty command")

        cwd: str | None = None
        env: dict[str, str] | None = None
        max_output = 1024 * 1024

        if config is not None:
            cwd = config.working_directory
            max_output = config.max_output_size
            if config.environment:
                env = {**os.environ, **config.environment}

        logger.debug("secure_runner: %s (timeout=%d)", cmd, timeout)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
                env=env,
            )

            # Truncate oversized output
            if len(result.stdout) > max_output:
                result = subprocess.CompletedProcess(
                    args=result.args,
                    returncode=result.returncode,
                    stdout=result.stdout[:max_output],
                    stderr=result.stderr,
                )
            if len(result.stderr) > max_output:
                result = subprocess.CompletedProcess(
                    args=result.args,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr[:max_output],
                )

            return result

        except subprocess.TimeoutExpired as exc:
            raise SecurityError(f"Command timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise SecurityError(f"Command not found: {cmd[0]}") from exc


# Module-level singleton — supports both secure_runner(...) and secure_runner.run_command(...)
secure_runner = SecureRunner()
