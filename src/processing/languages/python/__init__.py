"""
Python language processing module.

This module provides language-specific processing for Python code,
including parsing, entity extraction, and relationship analysis.
"""

from .python_config import PYTHON_CONFIG
from .python_processor import PyLanguageProcessor

__all__ = [
    "PYTHON_CONFIG",
    "PyLanguageProcessor",
]
