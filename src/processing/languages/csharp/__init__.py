"""C# language processing module.

This module provides Tree-sitter based parsing and analysis for C# code,
with a regex-based fallback parser for environments where Tree-sitter
is not available.

The module supports extraction of:
- Namespaces and using directives
- Classes, interfaces, structs, and enums
- Methods, properties, fields, and events
- Delegates and type relationships
- Access modifiers and language-specific features
"""

from .csharp_config import CSHARP_CONFIG
from .csharp_processor import CSharpProcessor

__all__ = ["CSharpProcessor", "CSHARP_CONFIG"]
