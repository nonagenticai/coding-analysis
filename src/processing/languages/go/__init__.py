"""
Go language processing module.

This module provides comprehensive Go language analysis capabilities including:
- AST parsing using Tree-sitter with regex fallback
- Module and package management
- Import dependency analysis
- Code quality and complexity metrics
- Test execution and analysis

The main entry point is GoProcessor, which coordinates all Go-specific analysis modules.
"""

from .go_config import GO_CONFIG
from .go_dependency_analyzer import GoDependencyAnalyzer
from .go_module_manager import GoModuleManager
from .go_processor import GoProcessor
from .go_quality_analyzer import GoQualityAnalyzer
from .go_syntax_parser import GoSyntaxParser
from .go_test_executor import GoTestExecutor

__all__ = [
    "GoProcessor",
    "GO_CONFIG",
    "GoSyntaxParser",
    "GoModuleManager",
    "GoDependencyAnalyzer",
    "GoQualityAnalyzer",
    "GoTestExecutor",
]
