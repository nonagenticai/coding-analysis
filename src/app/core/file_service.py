"""File service for reading/writing code files in analysis context."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileService:
    """Service for file operations needed by analysis endpoints."""

    def __init__(self, base_path: str | None = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def read_file(self, path: str) -> str:
        """Read a file and return its contents."""
        file_path = self.base_path / path if not Path(path).is_absolute() else Path(path)
        return file_path.read_text(encoding="utf-8")

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        file_path = self.base_path / path if not Path(path).is_absolute() else Path(path)
        return file_path.exists()

    def list_files(self, directory: str, pattern: str = "**/*") -> list[str]:
        """List files matching a pattern in a directory."""
        dir_path = (
            self.base_path / directory if not Path(directory).is_absolute() else Path(directory)
        )
        return [str(p.relative_to(dir_path)) for p in dir_path.glob(pattern) if p.is_file()]
