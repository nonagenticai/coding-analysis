"""
Quality tools registry for multi-language code analysis.
Main registry interface that coordinates the modular components.
"""

from __future__ import annotations

import logging
from typing import Any

from .tool_configuration import ToolConfiguration
from .tool_definitions import QualityTool, ToolCategory
from .tool_discovery import ToolDiscovery
from .tool_execution import ToolExecutor
from .tool_result_processing import ToolResultProcessor

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of quality analysis tools by language.

    Coordinates multiple modular components to provide a unified interface
    for tool discovery, configuration, execution, and result processing.

    Attributes:
        discovery: Tool discovery and registration component.
        configuration: Tool configuration and installation component.
        executor: Tool execution management component.
        result_processor: Tool result processing component."""

    def __init__(self):
        self.discovery = ToolDiscovery()
        self.configuration = ToolConfiguration()
        self.executor = ToolExecutor()
        self.result_processor = ToolResultProcessor()

    def get_tools_for_language(self, language: str) -> dict[str, QualityTool]:
        """Get quality tools for specified language.

        Args:
            language (str): The language parameter.

        Returns:
            Dict[str, QualityTool]: The dict[str, qualitytool] result."""
        return self.discovery.get_tools_for_language(language)

    def get_tool(self, language: str, tool_name: str) -> QualityTool | None:
        """Get specific tool for language.

        Args:
            language (str): The language parameter.
            tool_name (str): The tool_name parameter.

        Returns:
            Optional[QualityTool]: The optional[qualitytool] result."""
        return self.discovery.get_tool(language, tool_name)

    def get_tools_by_category(
        self, language: str, category: ToolCategory
    ) -> dict[str, QualityTool]:
        """Get tools by category for a language.

        Args:
            language (str): The language parameter.
            category (ToolCategory): The category parameter.

        Returns:
            Dict[str, QualityTool]: The dict[str, qualitytool] result."""
        return self.discovery.get_tools_by_category(language, category)

    def get_installation_commands(self, language: str) -> list[str]:
        """Get commands to install tools in container.

        Args:
            language (str): The language parameter.

        Returns:
            List[str]: The list[str] result."""
        return self.configuration.get_installation_commands(language)

    def list_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            List[str]: The list[str] result."""
        return self.discovery.list_supported_languages()

    def list_available_tools(self, language: str) -> list[str]:
        """Get list of available tool names for a language.

        Args:
            language (str): The language parameter.

        Returns:
            List[str]: The list[str] result."""
        return self.discovery.list_available_tools(language)

    def get_tool_info(self, language: str, tool_name: str) -> dict[str, Any] | None:
        """Get detailed information about a tool.

        Args:
            language (str): The language parameter.
            tool_name (str): The tool_name parameter.

        Returns:
            Optional[Dict[str, Any]]: The optional[dict[str, any]] result."""
        tool = self.get_tool(language, tool_name)
        return tool.to_dict() if tool else None

    # Extended functionality using the modular components

    def execute_tool(
        self,
        language: str,
        tool_name: str,
        target_path: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a quality tool and return processed results.

        Args:
            language: Programming language
            tool_name: Name of the tool to execute
            target_path: Path to analyze
            timeout: Optional timeout override

        Returns:
            Processed tool results
        """
        tool = self.get_tool(language, tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found for language {language}"}

        # Execute the tool
        execution_result = self.executor.execute_tool_safe(tool, target_path, timeout)

        # Process the results if execution was successful
        if execution_result["success"] and execution_result["output"]:
            processed_result = self.result_processor.process_tool_result(
                tool, execution_result["output"], execution_result["exit_code"]
            )
            # Merge execution metadata with processed results
            processed_result.update(
                {
                    "execution_time": execution_result.get("execution_time"),
                    "command_used": tool.get_command_string(),
                }
            )
            return processed_result
        else:
            return execution_result

    def validate_tool_availability(self, language: str, tool_name: str) -> bool:
        """Check if a tool is available for execution.

        Args:
            language: Programming language
            tool_name: Name of the tool

        Returns:
            True if tool is available
        """
        tool = self.get_tool(language, tool_name)
        if not tool:
            return False
        return self.executor.validate_tool_availability(tool)

    def get_configuration_templates(self, language: str) -> dict[str, str]:
        """Get configuration file templates for a language.

        Args:
            language: Programming language

        Returns:
            Dict mapping config file names to template content
        """
        return self.configuration.get_language_specific_config_templates(language)

    def run_analysis_suite(
        self,
        language: str,
        target_path: str,
        categories: list[ToolCategory] | None = None,
    ) -> dict[str, Any]:
        """Run a complete analysis suite for a language.

        Args:
            language: Programming language
            target_path: Path to analyze
            categories: Optional list of tool categories to run

        Returns:
            Comprehensive analysis results
        """
        results = []
        tools = self.get_tools_for_language(language)

        for tool_name, tool in tools.items():
            # Filter by categories if specified
            if categories and tool.category not in categories:
                continue

            # Skip if tool is not available
            if not self.executor.validate_tool_availability(tool):
                logger.warning(f"Tool {tool_name} is not available, skipping")
                continue

            try:
                result = self.execute_tool(language, tool_name, target_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute {tool_name}: {str(e)}")
                results.append({"tool_name": tool_name, "success": False, "error": str(e)})

        # Normalize and summarize results
        normalized_results = self.result_processor.normalize_results(results)
        normalized_results["language"] = language
        normalized_results["target_path"] = target_path

        return normalized_results


class _ToolRegistrySingleton:
    """Singleton pattern for tool registry.

    Attributes:
        _instance (Optional[ToolRegistry]): Class attribute _instance."""

    _instance: ToolRegistry | None = None

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """Get the singleton instance of ToolRegistry.

        Returns:
            ToolRegistry: The toolregistry result."""
        if cls._instance is None:
            cls._instance = ToolRegistry()
        return cls._instance


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance.

    Returns:
        ToolRegistry: The toolregistry result."""
    return _ToolRegistrySingleton.get_instance()


def get_tools_for_language(language: str) -> dict[str, QualityTool]:
    """Convenience function to get tools for a language.

    Args:
        language (str): The language parameter.

    Returns:
        Dict[str, QualityTool]: The dict[str, qualitytool] result."""
    return get_tool_registry().get_tools_for_language(language)


def get_installation_commands(language: str) -> list[str]:
    """Convenience function to get installation commands.

    Args:
        language (str): The language parameter.

    Returns:
        List[str]: The list[str] result."""
    return get_tool_registry().get_installation_commands(language)


def list_supported_languages() -> list[str]:
    """Convenience function to list supported languages.

    Returns:
        List[str]: The list[str] result."""
    return get_tool_registry().list_supported_languages()
