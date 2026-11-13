"""Module for parsing Python code."""

from __future__ import annotations

import ast
import logging
from typing import Any

from ..base_parser import CodeParser

logger = logging.getLogger(__name__)


class PythonParser(CodeParser):
    """Parser for Python code using the built-in ast module."""

    async def parse_code(
        self,
        code: str,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Parse Python code into an AST.

        Args:
            code: Python source code content
            file_path: Path to the file (for error reporting)
            options: Optional parser-specific options (unused for Python)

        Returns:
            Parsed AST tree
        """
        try:
            return ast.parse(code, filename=file_path)
        except SyntaxError as e:
            logger.error(
                "Syntax error parsing Python file %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return None
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(
                "Error parsing Python file %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return None

    async def validate_syntax(
        self,
        code: str,
        file_path: str,
    ) -> bool:
        """Validate if the Python code has valid syntax without full parsing.

        Args:
            code: Python source code content to validate
            file_path: Path to the file (for error reporting)

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code, filename=file_path)
            return True
        except SyntaxError:
            logger.debug(
                "Syntax error in Python file %s during validation",
                file_path,
            )
            return False
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(
                "Unexpected error validating Python file %s: %s",
                file_path,
                str(e),
            )
            return False
