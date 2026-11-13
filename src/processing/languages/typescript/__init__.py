"""
TypeScript Language Processing Module

This module provides TypeScript language processing capabilities including:
- TypeScript file parsing and analysis
- Type system analysis (interfaces, types, generics)
- Decorator extraction and analysis
- Module and namespace handling
- Advanced TypeScript features support
"""

from __future__ import annotations

from .typescript_processor import TYPESCRIPT_CONFIG, TSLanguageProcessor

__all__ = [
    "TSLanguageProcessor",
    "TYPESCRIPT_CONFIG",
]
