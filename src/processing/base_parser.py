"""Base parser module for language-specific code parsing.

This module provides the abstract base class for implementing language-specific
code parsers that can convert source code into structured representations like ASTs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class CodeParser(ABC):
    """Abstract base class for language-specific code parsers."""

    @abstractmethod
    async def parse_code(
        self,
        code: str,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Parse code content into an AST or other structured representation.

        Args:
            code: Source code content to parse
            file_path: Path to the file (for error reporting)
            options: Optional parser-specific options

        Returns:
            Parsed AST or structured representation

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """

    @abstractmethod
    async def validate_syntax(
        self,
        code: str,
        file_path: str,
    ) -> bool:
        """Validate if the code has valid syntax without full parsing.

        Args:
            code: Source code content to validate
            file_path: Path to the file (for error reporting)

        Returns:
            True if syntax is valid, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
