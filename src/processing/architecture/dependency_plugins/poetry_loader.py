"""Loader for managing dependencies using Poetry in architecture processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.architecture_models import ExternalPackage

from . import AbstractDepLoader

try:
    import tomllib as tomli  # Python 3.11+
except ImportError:
    try:
        import tomli
    except ImportError:
        tomli = None  # type: ignore


class PoetryLoader(AbstractDepLoader):
    """Loader for Poetry pyproject.toml files."""

    def detect(self, repo: Path) -> bool:
        """Detect if this is a Poetry project."""
        pyproject_path = repo / "pyproject.toml"
        if not pyproject_path.is_file():
            return False

        # Check if pyproject.toml contains Poetry configuration
        content = self._read_file_safely(pyproject_path)
        return "[tool.poetry]" in content

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse Poetry dependencies from pyproject.toml and poetry.lock."""
        if tomli is None:
            # Fallback to basic text parsing if tomli not available
            return self._parse_simple(repo)

        packages = []

        # Parse pyproject.toml
        pyproject_packages = self._parse_pyproject_toml(repo)
        packages.extend(pyproject_packages)

        # If poetry.lock exists, get exact versions from there
        lock_packages = self._parse_poetry_lock(repo)
        if lock_packages:
            # Update versions with locked versions by creating new package objects
            lock_by_name = {pkg.name: pkg for pkg in lock_packages}
            updated_packages = []
            for package in packages:
                if package.name in lock_by_name:
                    locked_package = lock_by_name[package.name]
                    updated_package = ExternalPackage(
                        name=package.name,
                        version=locked_package.version,
                        license=package.license,
                        ecosystem=package.ecosystem,
                        description=package.description,
                        current_version=locked_package.version,
                    )
                    updated_packages.append(updated_package)
                else:
                    updated_packages.append(package)
            packages = updated_packages

        return packages

    def _parse_pyproject_toml(self, repo: Path) -> list[ExternalPackage]:
        """Parse dependencies from pyproject.toml."""
        packages: list[ExternalPackage] = []
        pyproject_path = repo / "pyproject.toml"

        if not pyproject_path.is_file():
            return packages

        try:
            content = pyproject_path.read_text(encoding="utf-8")
            assert tomli is not None, "tomllib/tomli is required for pyproject.toml parsing"
            data = tomli.loads(content)

            tool_poetry = data.get("tool", {}).get("poetry", {})

            # Parse main dependencies from both old and new formats
            # Old format: [tool.poetry.dependencies]
            dependencies = tool_poetry.get("dependencies", {})
            packages.extend(self._parse_dependencies_section(dependencies))

            # New format: [project.dependencies] (Poetry 2.0+)
            project_dependencies = data.get("project", {}).get("dependencies", [])
            packages.extend(self._parse_project_dependencies(project_dependencies))

            # Parse dev dependencies - old format: [tool.poetry.dev-dependencies]
            dev_dependencies = tool_poetry.get("dev-dependencies", {})
            packages.extend(self._parse_dependencies_section(dev_dependencies))

            # Parse dev dependencies - new format: [tool.poetry.group.dev.dependencies]
            groups = tool_poetry.get("group", {})
            for group_name, group_data in groups.items():
                # Parse dependencies from each group
                group_deps = group_data.get("dependencies", {})
                packages.extend(self._parse_dependencies_section(group_deps))

        except Exception:
            # Fallback to basic parsing
            packages.extend(self._parse_simple_pyproject(repo))

        return packages

    def _parse_dependencies_section(self, dependencies: dict[str, Any]) -> list[ExternalPackage]:
        """Parse a dependencies section from pyproject.toml."""
        packages = []

        for name, spec in dependencies.items():
            # Skip Python itself
            if name.lower() == "python":
                continue

            if isinstance(spec, str):
                # Basic version specification
                packages.append(
                    ExternalPackage(
                        name=name,
                        version=spec,
                        license="Unknown",
                        ecosystem="pypi",
                    )
                )
            elif isinstance(spec, dict):
                # Complex specification with version, extras, etc.
                version = spec.get("version", "*")
                packages.append(
                    ExternalPackage(
                        name=name,
                        version=version,
                        license="Unknown",
                        ecosystem="pypi",
                    )
                )

        return packages

    def _parse_project_dependencies(self, dependencies: list[str]) -> list[ExternalPackage]:
        """Parse dependencies from project.dependencies list format."""
        packages = []

        for dep_spec in dependencies:
            # Parse dependency specifications like "requests>=2.28.0"
            # Split on common version operators
            for op in [">=", "<=", "==", "~=", "!=", ">", "<"]:
                if op in dep_spec:
                    name, version = dep_spec.split(op, 1)
                    name = name.strip()
                    version = op + version.strip()
                    break
            else:
                # No version specifier, just package name
                name = dep_spec.strip()
                version = "*"

            packages.append(
                ExternalPackage(
                    name=name,
                    version=version,
                    license="Unknown",
                    ecosystem="pypi",
                )
            )

        return packages

    def _parse_poetry_lock(self, repo: Path) -> list[ExternalPackage]:
        """Parse exact versions from poetry.lock."""
        packages: list[ExternalPackage] = []
        lock_path = repo / "poetry.lock"

        if not lock_path.is_file():
            return packages

        try:
            content = lock_path.read_text(encoding="utf-8")
            assert tomli is not None, "tomllib/tomli is required for poetry.lock parsing"
            data = tomli.loads(content)

            # Poetry lock format has packages in [[package]] sections
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
            pass

        return packages

    def _parse_simple(self, repo: Path) -> list[ExternalPackage]:
        """Basic text-based parsing when tomli is not available."""
        packages = []

        # Parse pyproject.toml
        packages.extend(self._parse_simple_pyproject(repo))

        # Parse poetry.lock
        packages.extend(self._parse_simple_lock(repo))

        return packages

    def _parse_simple_pyproject(self, repo: Path) -> list[ExternalPackage]:
        """Basic text parsing of pyproject.toml."""
        packages: list[ExternalPackage] = []
        pyproject_path = repo / "pyproject.toml"

        if not pyproject_path.is_file():
            return packages

        content = self._read_file_safely(pyproject_path)
        lines = content.splitlines()

        in_dependencies = False
        dependency_sections = {
            "[tool.poetry.dependencies]",
            "[tool.poetry.dev-dependencies]",
            "[tool.poetry.group.dev.dependencies]",
            "[tool.poetry.group.test.dependencies]",
            "[tool.poetry.group.docs.dependencies]",
        }

        # Also check for other group patterns
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[tool.poetry.group.") and stripped.endswith(".dependencies]"):
                dependency_sections.add(stripped)

        for line in lines:
            line = line.strip()

            if line in dependency_sections:
                in_dependencies = True
                continue
            elif line.startswith("[") and in_dependencies:
                in_dependencies = False
                continue

            if in_dependencies and "=" in line and not line.startswith("#"):
                try:
                    name, spec = line.split("=", 1)
                    name = name.strip()
                    spec = spec.strip().strip("\"'")

                    if name.lower() != "python":
                        packages.append(
                            ExternalPackage(
                                name=name,
                                version=spec,
                                license="Unknown",
                                ecosystem="pypi",
                            )
                        )
                except Exception:
                    continue

        return packages

    def _parse_simple_lock(self, repo: Path) -> list[ExternalPackage]:
        """Basic text parsing of poetry.lock."""
        packages: list[ExternalPackage] = []
        lock_path = repo / "poetry.lock"

        if not lock_path.is_file():
            return packages

        content = self._read_file_safely(lock_path)
        lines = content.splitlines()

        current_package: str | None = None
        for line in lines:
            line = line.strip()

            if line.startswith('name = "'):
                current_package = line.split('"')[1]
            elif line.startswith('version = "') and current_package:
                version = line.split('"')[1]
                packages.append(
                    ExternalPackage(
                        name=current_package,
                        version=version,
                        license="Unknown",
                        ecosystem="pypi",
                    )
                )
                current_package = None

        return packages
