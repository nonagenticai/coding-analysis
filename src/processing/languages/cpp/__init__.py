"""
C++ language processing module.

This module provides comprehensive C++ and C language processing capabilities
including syntax parsing, dependency analysis, quality metrics, CMake support,
and test execution.
"""

from __future__ import annotations

from .cpp_cmake_manager import CppCMakeManager

# Configuration
from .cpp_config import CPP_CONFIG
from .cpp_dependency_analyzer import CppDependencyAnalyzer

# Main processor
from .cpp_processor import CppProcessor
from .cpp_quality_analyzer import CppQualityAnalyzer

# Specialized analyzers and components
from .cpp_syntax_parser import CppSyntaxParser
from .cpp_test_executor import CppTestExecutor

__all__ = [
    # Main processor
    "CppProcessor",
    # Configuration
    "CPP_CONFIG",
    # Specialized components (for advanced usage)
    "CppSyntaxParser",
    "CppDependencyAnalyzer",
    "CppQualityAnalyzer",
    "CppCMakeManager",
    "CppTestExecutor",
]
