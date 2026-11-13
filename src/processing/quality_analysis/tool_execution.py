"""
Tool execution management for quality analysis tools.
Handles command execution, timeout management, and result processing.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from .tool_definitions import QualityTool

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    def __init__(self, tool_name: str, exit_code: int, stderr: str):
        self.tool_name = tool_name
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"Tool {tool_name} failed with exit code {exit_code}: {stderr}")


class ToolExecutor:
    """Manages execution of quality analysis tools."""

    def __init__(self):
        self.default_timeout = 120

    def execute_tool(
        self, tool: QualityTool, target_path: str, timeout: int | None = None
    ) -> tuple[bool, str, str]:
        """Execute a quality tool on the target path.

        Args:
            tool: The QualityTool instance to execute
            target_path: Path to the code to analyze
            timeout: Optional timeout override

        Returns:
            Tuple of (success, stdout, stderr)

        Raises:
            ToolExecutionError: If the tool execution fails
        """
        command = self._prepare_command(tool, target_path)
        execution_timeout = timeout or tool.timeout or self.default_timeout

        logger.info(f"Executing tool {tool.name} with command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=execution_timeout,
                cwd=target_path if target_path else None,
            )

            success = result.returncode == 0
            if not success:
                logger.warning(f"Tool {tool.name} returned non-zero exit code: {result.returncode}")

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Tool {tool.name} timed out after {execution_timeout} seconds"
            logger.error(error_msg)
            raise ToolExecutionError(tool.name, -1, error_msg)

        except FileNotFoundError:
            error_msg = f"Tool {tool.name} not found. Please ensure it's installed."
            logger.error(error_msg)
            raise ToolExecutionError(tool.name, -2, error_msg)

        except Exception as e:
            error_msg = f"Unexpected error executing tool {tool.name}: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(tool.name, -3, error_msg)

    def execute_tool_safe(
        self, tool: QualityTool, target_path: str, timeout: int | None = None
    ) -> dict[str, Any]:
        """Execute a tool safely, returning a structured result.

        Args:
            tool: The QualityTool instance to execute
            target_path: Path to the code to analyze
            timeout: Optional timeout override

        Returns:
            Dict containing execution results and metadata
        """
        try:
            success, stdout, stderr = self.execute_tool(tool, target_path, timeout)

            return {
                "tool_name": tool.name,
                "success": success,
                "output": stdout,
                "error": stderr,
                "output_format": tool.output_format,
                "execution_time": None,  # Could be enhanced to track time
                "exit_code": 0 if success else 1,
            }

        except ToolExecutionError as e:
            return {
                "tool_name": tool.name,
                "success": False,
                "output": "",
                "error": e.stderr,
                "output_format": tool.output_format,
                "execution_time": None,
                "exit_code": e.exit_code,
            }

    def _prepare_command(self, tool: QualityTool, target_path: str) -> list[str]:
        """Prepare the command for execution by adding target path if needed.

        Args:
            tool: The QualityTool instance
            target_path: Path to the code to analyze

        Returns:
            List of command parts ready for execution
        """
        command = tool.command.copy()

        # Some tools need the target path appended
        if self._tool_needs_target_path(tool, target_path):
            command.append(target_path)

        return command

    def _tool_needs_target_path(self, tool: QualityTool, target_path: str) -> bool:
        """Determine if a tool needs the target path appended to its command.

        Args:
            tool: The QualityTool instance
            target_path: Path to the code to analyze

        Returns:
            True if the tool needs target path appended
        """
        # Tools that typically don't need explicit path (they scan current directory)
        no_path_tools = {
            "ruff",
            "ty",
            "bandit",
            "eslint",
            "prettier",
            "tsc",
            "cargo",
            "go",
            "dotnet",
        }

        # Check if tool command starts with any of these
        if tool.command and any(tool.command[0].startswith(prefix) for prefix in no_path_tools):
            return False

        # Check if path is already in command
        if target_path in tool.command or "." in tool.command:
            return False

        return True

    def validate_tool_availability(self, tool: QualityTool) -> bool:
        """Check if a tool is available for execution.

        Args:
            tool: The QualityTool instance to check

        Returns:
            True if the tool is available
        """
        if not tool.command:
            return False

        try:
            # Try to run the tool with --version or --help
            test_commands = ["--version", "--help", "-h"]
            base_command = tool.command[0]

            for test_arg in test_commands:
                try:
                    subprocess.run(
                        [base_command, test_arg],
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            return False

        except Exception:
            return False

    def get_tool_version(self, tool: QualityTool) -> str | None:
        """Get the version of an installed tool.

        Args:
            tool: The QualityTool instance

        Returns:
            Version string if available, None otherwise
        """
        if not tool.command:
            return None

        try:
            result = subprocess.run(
                [tool.command[0], "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except Exception:
            pass

        return None
