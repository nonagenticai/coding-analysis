"""Initialization module for dependency plugins in architecture processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from models.architecture_models import ExternalPackage

logger = logging.getLogger(__name__)


class BaseDepLoader(Protocol):
    """Protocol for dependency loader plugins."""

    def detect(self, repo: Path) -> bool:
        """Return True if this loader can handle the repository."""
        ...

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse dependencies from the repository."""
        ...


class AbstractDepLoader(ABC):
    """Abstract base class for dependency loaders."""

    @abstractmethod
    def detect(self, repo: Path) -> bool:
        """Return True if this loader can handle the repository."""

    @abstractmethod
    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse dependencies from the repository."""

    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read a file, returning empty string on error.

        An empty string is indistinguishable from an empty manifest downstream,
        and this helper is used at 19 call sites across all six loaders -- so an
        unreadable manifest used to surface as "this repo declares no
        dependencies" and a 0.0 risk score. It still degrades to "", because a
        single unreadable manifest should not abort the whole extraction, but it
        is now loud in the logs rather than silent.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Loaders probe for optional manifests; absence is expected and not
            # a failure, so it stays at debug.
            logger.debug("Manifest %s not present", file_path)
            return ""
        except Exception as exc:
            logger.warning(
                "Could not read dependency manifest %s (%s: %s); its dependencies "
                "will be missing from the report",
                file_path,
                type(exc).__name__,
                exc,
            )
            return ""
