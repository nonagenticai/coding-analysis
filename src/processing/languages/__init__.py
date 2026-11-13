"""Language processing modules for multi-language code analysis.

This package provides language-specific processors for parsing and analyzing
source code across multiple programming languages. It includes both legacy
processors and new tree-sitter based processors with a unified interface.

Key Components:
- Base classes and interfaces for language processors
- Language-specific processor implementations
- Language configuration and registry system
- Parser strategies (AST, Tree-sitter, external tools)

Supported Languages:
- Python (native AST)
- JavaScript/TypeScript (Tree-sitter)
- Java (Tree-sitter)
- C/C++ (Tree-sitter)
- C# (Tree-sitter)
- Go (Tree-sitter)
- Rust (Tree-sitter)
"""

from __future__ import annotations

import logging

# Base classes and types
from .base import (
    LanguageConfig,
    LanguageFeatures,
    LanguageProcessor,
    ParseResult,
    ParserType,
)

# Language registry
from .registry import language_registry, register_language

logger = logging.getLogger(__name__)

# Language processors - imported to trigger registration
# Import errors are logged but not raised to allow partial functionality
_processor_imports = [
    ("cpp.cpp_processor", "CppProcessor"),
    ("csharp.csharp_processor", "CSharpProcessor"),
    ("go.go_processor", "GoProcessor"),
    ("java.java_processor", "JavaProcessor"),
    ("javascript.javascript_processor", "JSLanguageProcessor"),
    ("python.python_processor", "PyLanguageProcessor"),
    ("rust.rust_processor", "RustProcessor"),
    ("typescript.typescript_processor", "TSLanguageProcessor"),
]

# Dynamically import processors and collect successfully imported ones
_imported_processors = []
for module_path, class_name in _processor_imports:
    try:
        # Use importlib for more reliable imports
        import importlib

        full_module_path = f"processing.languages.{module_path}"
        module = importlib.import_module(full_module_path)
        processor_class = getattr(module, class_name)
        globals()[class_name] = processor_class
        _imported_processors.append(class_name)
        logger.debug(f"Successfully imported {class_name} from {module_path}")
    except ImportError as e:
        logger.warning(f"Could not import {class_name} from {module_path}: {e}")
    except Exception as e:
        logger.error(f"Error importing {class_name} from {module_path}: {e}")

__all__ = [
    # Base classes (always available)
    "LanguageConfig",
    "LanguageFeatures",
    "LanguageProcessor",
    "ParseResult",
    "ParserType",
    # Registry (always available)
    "language_registry",
    "register_language",
] + _imported_processors  # Add successfully imported processors
