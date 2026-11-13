"""Dependency parser — scans project dependency files and resolves package info."""

from __future__ import annotations

import difflib
import logging
import re
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQ_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*([><=!~]+)?\s*([^\s,;]+)?")


class DependencyParser:
    """Parses dependency files and resolves package information."""

    def __init__(self, project_root: str) -> None:
        self._root = Path(project_root)
        self._packages: dict[str, dict[str, Any]] = {}
        self._scanned = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_requirements(self, file_path: str) -> list[dict[str, Any]]:
        """Parse a requirements.txt-style file.

        Returns list of {"name": str, "version": str, "extras": list[str]}.
        """
        results: list[dict[str, Any]] = []
        path = Path(file_path)
        if not path.is_file():
            return results

        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r", "-e", "--editable", "--index", "--extra", "--find")):
                continue
            # Strip inline comments
            if " #" in line:
                line = line[: line.index(" #")].strip()

            match = _REQ_PATTERN.match(line)
            if not match:
                continue

            raw_name = match.group(1)
            operator = match.group(2) or ""
            version_str = match.group(3) or ""

            extras: list[str] = []
            name = raw_name
            if "[" in raw_name:
                name = raw_name.split("[")[0]
                extras_part = raw_name.split("[")[1].rstrip("]")
                extras = [e.strip() for e in extras_part.split(",") if e.strip()]

            version = f"{operator}{version_str}" if operator else version_str
            results.append({"name": name, "version": version, "extras": extras})

        return results

    def parse_pyproject(self, file_path: str) -> list[dict[str, Any]]:
        """Parse dependencies from a pyproject.toml [project.dependencies] section."""
        results: list[dict[str, Any]] = []
        path = Path(file_path)
        if not path.is_file():
            return results

        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse pyproject.toml at %s", file_path)
            return results

        deps: list[str] = data.get("project", {}).get("dependencies", [])
        for spec in deps:
            spec = spec.strip()
            if not spec:
                continue
            match = _REQ_PATTERN.match(spec)
            if not match:
                continue

            raw_name = match.group(1)
            operator = match.group(2) or ""
            version_str = match.group(3) or ""

            extras: list[str] = []
            name = raw_name
            if "[" in raw_name:
                name = raw_name.split("[")[0]
                extras_part = raw_name.split("[")[1].rstrip("]")
                extras = [e.strip() for e in extras_part.split(",") if e.strip()]

            version = f"{operator}{version_str}" if operator else version_str
            results.append({"name": name, "version": version, "extras": extras})

        return results

    async def get_available_packages(self) -> dict[str, dict[str, Any]]:
        """Scan project root for dependency files and return {name: info} dict."""
        if self._scanned:
            return self._packages

        self._packages = {}

        # requirements*.txt files
        for req in self._root.glob("requirements*.txt"):
            for pkg in self.parse_requirements(str(req)):
                self._packages[pkg["name"].lower()] = pkg

        # Also look inside a requirements/ directory
        req_dir = self._root / "requirements"
        if req_dir.is_dir():
            for req in req_dir.glob("*.txt"):
                for pkg in self.parse_requirements(str(req)):
                    self._packages.setdefault(pkg["name"].lower(), pkg)

        # pyproject.toml
        pyproject = self._root / "pyproject.toml"
        if pyproject.is_file():
            for pkg in self.parse_pyproject(str(pyproject)):
                self._packages.setdefault(pkg["name"].lower(), pkg)

        self._scanned = True
        return self._packages

    def get_package_info(self, name: str) -> dict[str, Any] | None:
        """Return parsed info for *name* if present in scanned packages."""
        return self._packages.get(name.lower())

    def get_similar_packages(self, name: str, n: int = 5, cutoff: float = 0.6) -> list[str]:
        """Return up to *n* package names similar to *name* from scanned packages."""
        return difflib.get_close_matches(
            name.lower(), list(self._packages.keys()), n=n, cutoff=cutoff
        )
