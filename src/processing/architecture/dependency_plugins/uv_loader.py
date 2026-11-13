"""Loader for UV dependencies in architecture processing."""

from __future__ import annotations

from pathlib import Path

from models.architecture_models import ExternalPackage

from . import AbstractDepLoader


class UvLoader(AbstractDepLoader):
    """Loader for uv lock files."""

    def detect(self, repo: Path) -> bool:
        """Detect if this is a uv project."""
        return (repo / "uv.lock").is_file()

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse uv dependencies from uv.lock."""
        packages: list[ExternalPackage] = []
        lock_path = repo / "uv.lock"

        if not lock_path.is_file():
            return packages

        try:
            content = lock_path.read_text(encoding="utf-8")
            # uv.lock is typically in TOML format
            data = self._parse_uv_lock_content(content)

            for package_data in data.get("package", []):
                name = package_data.get("name", "")
                version = package_data.get("version", "")

                if name and version:
                    packages.append(
                        ExternalPackage(
                            name=name,
                            version=version,
                            license="Unknown",
                            ecosystem="pypi",
                        )
                    )

        except Exception:
            # Fallback to basic text parsing
            packages.extend(self._parse_simple_lock(repo))

        return packages

    def _parse_uv_lock_content(self, content: str) -> dict:
        """Parse uv.lock content (TOML format)."""
        try:
            import tomli

            return tomli.loads(content)
        except ImportError:
            # Fallback to manual parsing
            return self._parse_toml_simple(content)

    def _parse_toml_simple(self, content: str) -> dict:
        """Basic TOML parsing for uv.lock when tomli is not available."""
        result: dict = {"package": []}
        lines = content.splitlines()

        current_package: dict = {}
        in_package = False

        for line in lines:
            line = line.strip()

            if line == "[[package]]":
                if current_package:
                    result["package"].append(current_package)
                current_package = {}
                in_package = True
                continue

            if in_package and "=" in line:
                try:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    current_package[key] = value
                except Exception:
                    continue

        # Add the last package
        if current_package:
            result["package"].append(current_package)

        return result

    def _parse_simple_lock(self, repo: Path) -> list[ExternalPackage]:
        """Basic text parsing of uv.lock."""
        packages: list[ExternalPackage] = []
        lock_path = repo / "uv.lock"

        if not lock_path.is_file():
            return packages

        content = self._read_file_safely(lock_path)
        lines = content.splitlines()

        current_name = None
        current_version = None

        for line in lines:
            line = line.strip()

            if line.startswith('name = "'):
                current_name = line.split('"')[1]
            elif line.startswith('version = "'):
                current_version = line.split('"')[1]

                if current_name and current_version:
                    packages.append(
                        ExternalPackage(
                            name=current_name,
                            version=current_version,
                            license="Unknown",
                            ecosystem="pypi",
                        )
                    )
                    current_name = None
                    current_version = None

        return packages
