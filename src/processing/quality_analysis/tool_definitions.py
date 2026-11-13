"""
Tool definitions for quality analysis tools.
Defines the basic data structures and categories for quality tools.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCategory(StrEnum):
    """Categories of quality tools."""

    LINTER = "linter"
    FORMATTER = "formatter"
    TYPE_CHECKER = "type_checker"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    # New categories for AI Agent Static Analysis Augmentation
    CALL_GRAPH = "call_graph"  # Call graph generation tools
    SIGNATURE_ANALYSIS = "signature"  # Comprehensive signature extraction
    SYMBOLIC_ANALYSIS = "symbolic"  # Symbolic execution tools
    ARCHITECTURE = "architecture"  # Architecture documentation


class QualityTool:
    """Quality tool configuration.

    Attributes:
        name: Instance attribute name.
        category: Instance attribute category.
        command: Instance attribute command.
        output_format: Instance attribute output_format.
        description: Instance attribute description.
        installation_cmd: Instance attribute installation_cmd.
        timeout: Instance attribute timeout.
        config_files: Instance attribute config_files.

    Example:
        >>> obj = QualityTool("name_value", "category_value", "command_value", "output_format_value", "description_value")
    """

    def __init__(
        self,
        name: str,
        category: ToolCategory,
        command: list[str],
        output_format: str,
        description: str,
        *,
        installation_cmd: list[str] | None = None,
        timeout: int = 30,
        config_files: list[str] | None = None,
    ):
        self.name = name
        self.category = category
        self.command = command
        self.output_format = output_format
        self.description = description
        self.installation_cmd = installation_cmd or []
        self.timeout = timeout
        self.config_files = config_files or []

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary representation.

        Returns:
            Dict[str, Any]: The dict[str, any] result."""
        return {
            "name": self.name,
            "category": self.category.value,
            "command": self.command,
            "output_format": self.output_format,
            "description": self.description,
            "installation_cmd": self.installation_cmd,
            "timeout": self.timeout,
            "config_files": self.config_files,
        }

    def get_command_string(self) -> str:
        """Get the command as a single string for execution.

        Returns:
            str: The str result."""
        return " ".join(self.command)
