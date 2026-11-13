"""
JavaScript language processing module.

Provides JavaScript and TypeScript language processing capabilities using Tree-sitter
with fallback support for regex-based parsing.
"""

from __future__ import annotations

from .javascript_config import JAVASCRIPT_CONFIG, TYPESCRIPT_CONFIG
from .javascript_processor import JSLanguageProcessor

__all__ = [
    "JSLanguageProcessor",
    "JAVASCRIPT_CONFIG",
    "TYPESCRIPT_CONFIG",
]
