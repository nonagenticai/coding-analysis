"""
Quality Analysis Module - Multi-language code quality analysis and reporting.

This module provides tools for analyzing code quality across multiple programming languages,
generating comprehensive reports, and managing quality tool execution.

Key Components:
- QualityReportGenerator: Main interface for generating quality reports
- ToolRegistry: Registry of quality analysis tools by language
- QualityToolRunner: Orchestrates quality tool execution in containers
- Various analyzers for specific languages and protocols
"""

from .fix_formatters import FixFormatter
from .fix_models import QualityAnalysisResolveSuggestions as FixModel
from .fix_rules import FixRule

# Fix and Improvement Tools
from .fix_suggester import FixSuggester

# HTTP Quality Analysis
from .http_quality_metrics import HTTPQualityMetrics
from .http_quality_models import HTTPQualityModel
from .http_quality_runner import HTTPQualityRunner

# Language-specific Analyzers
from .java_quality_analyzer import JavaQualityAnalyzer

# Quality Analysis Runners
from .quality_runner import QualityToolRunner
from .report_chart_generator import ReportChartGenerator, create_chart_generator

# Report Components (for advanced usage)
from .report_configuration import (
    ReportConfigurationManager,
    ReportContext,
    ReportSettings,
    create_configuration_manager,
    create_default_settings,
)
from .report_data_processor import ReportDataProcessor, create_data_processor

# Report Generation
from .report_generator import QualityReportGenerator
from .report_output_handler import ReportOutputHandler, create_output_handler
from .report_template_manager import ReportTemplateManager, create_template_manager
from .tool_configuration import ToolConfiguration

# Tool Definitions and Configuration
from .tool_definitions import QualityTool, ToolCategory
from .tool_discovery import ToolDiscovery
from .tool_execution import ToolExecutor

# Tool Registry and Management
from .tool_registry import (
    ToolRegistry,
    get_installation_commands,
    get_tool_registry,
    get_tools_for_language,
    list_supported_languages,
)
from .tool_result_processing import ToolResultProcessor

# Trend Analysis
from .trend_analyzer import TrendAnalyzer

__all__ = [
    # Main interfaces
    "QualityReportGenerator",
    "ToolRegistry",
    "QualityToolRunner",
    "HTTPQualityRunner",
    # Tool registry functions
    "get_tool_registry",
    "get_tools_for_language",
    "get_installation_commands",
    "list_supported_languages",
    # Tool components
    "QualityTool",
    "ToolCategory",
    "ToolDiscovery",
    "ToolConfiguration",
    "ToolExecutor",
    "ToolResultProcessor",
    # Analyzers
    "JavaQualityAnalyzer",
    "TrendAnalyzer",
    # Fix tools
    "FixSuggester",
    "FixFormatter",
    "FixModel",
    "FixRule",
    # HTTP quality
    "HTTPQualityMetrics",
    "HTTPQualityModel",
    # Report components
    "ReportConfigurationManager",
    "ReportContext",
    "ReportSettings",
    "create_configuration_manager",
    "create_default_settings",
    "ReportDataProcessor",
    "create_data_processor",
    "ReportChartGenerator",
    "create_chart_generator",
    "ReportTemplateManager",
    "create_template_manager",
    "ReportOutputHandler",
    "create_output_handler",
]
