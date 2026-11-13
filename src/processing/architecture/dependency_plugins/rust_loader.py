"""Rust dependency loader for parsing Cargo.toml and Cargo.lock files.

This module provides functionality to detect and parse Rust project dependencies
from Cargo.toml manifest files and Cargo.lock lockfiles, including support for
workspaces, target-specific dependencies, and various dependency specifications.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+

    TOMLLIB_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib  # Backport for older Python versions

        TOMLLIB_AVAILABLE = True
    except ImportError:
        TOMLLIB_AVAILABLE = False
        tomllib = None  # type: ignore

from models.architecture_models import ExternalPackage

from . import AbstractDepLoader

logger = logging.getLogger(__name__)


class RustDependencyLoader(AbstractDepLoader):
    """Loader for Rust dependencies from Cargo.toml and Cargo.lock files."""

    def __init__(self):
        """Initialize the Rust dependency loader."""
        self.dependency_sections = [
            "dependencies",
            "dev-dependencies",
            "build-dependencies",
        ]

    def detect(self, repo: Path) -> bool:
        """Detect if this is a Rust project with Cargo package management."""
        # Primary detection: Cargo.toml file
        if (repo / "Cargo.toml").is_file():
            return True

        # Secondary detection: Cargo.lock without Cargo.toml (edge case)
        if (repo / "Cargo.lock").is_file():
            logger.warning("Found Cargo.lock without Cargo.toml in %s", repo)
            return True

        # Workspace detection: look for nested Cargo.toml files
        for cargo_toml in repo.rglob("Cargo.toml"):
            if cargo_toml.is_file():
                return True

        return False

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse Rust dependencies from Cargo.toml and Cargo.lock."""
        if not TOMLLIB_AVAILABLE or tomllib is None:
            logger.error("TOML parsing library not available. Install tomli or use Python 3.11+")
            return []

        packages: list[ExternalPackage] = []

        try:
            # Parse main Cargo.toml
            main_cargo_toml = repo / "Cargo.toml"
            if main_cargo_toml.is_file():
                packages.extend(self._parse_cargo_toml(main_cargo_toml))

            # Parse Cargo.lock for exact versions if available
            cargo_lock = repo / "Cargo.lock"
            if cargo_lock.is_file():
                lock_packages = self._parse_cargo_lock(cargo_lock)
                # Merge lockfile data with Cargo.toml data
                packages = self._merge_package_data(packages, lock_packages)

            # Handle workspaces
            workspace_packages = self._parse_workspace_members(repo, main_cargo_toml)
            packages.extend(workspace_packages)

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error parsing Rust dependencies: %s", e)

        # Remove duplicates and return
        return self._deduplicate_packages(packages)

    def _parse_cargo_toml(self, cargo_toml_path: Path) -> list[ExternalPackage]:
        """Parse dependencies from a Cargo.toml file."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(cargo_toml_path)
            if not content:
                return packages

            assert tomllib is not None, "tomllib/tomli is required for Cargo.toml parsing"
            data = tomllib.loads(content)

            # Parse different dependency sections
            packages.extend(self._parse_standard_dependencies(data))

            # Parse target-specific dependencies
            packages.extend(self._parse_target_dependencies(data))

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error parsing %s: %s", cargo_toml_path, e)

        return packages

    def _parse_standard_dependencies(self, data: dict[str, Any]) -> list[ExternalPackage]:
        """Parse standard dependency sections from Cargo.toml data."""
        packages: list[ExternalPackage] = []

        for section in self.dependency_sections:
            if section in data and isinstance(data[section], dict):
                section_packages = self._parse_dependency_section(data[section], section)
                packages.extend(section_packages)

        return packages

    def _parse_target_dependencies(self, data: dict[str, Any]) -> list[ExternalPackage]:
        """Parse target-specific dependencies from Cargo.toml data."""
        packages: list[ExternalPackage] = []

        if "target" not in data:
            return packages

        for target_name, target_data in data["target"].items():
            if not isinstance(target_data, dict):
                continue

            for section in self.dependency_sections:
                if section in target_data:
                    target_packages = self._parse_dependency_section(
                        target_data[section], f"{section} (target: {target_name})"
                    )
                    packages.extend(target_packages)

        return packages

    def _parse_dependency_section(
        self, dependencies: dict[str, Any], section_name: str
    ) -> list[ExternalPackage]:
        """Parse a dependency section from Cargo.toml."""
        packages: list[ExternalPackage] = []

        for name, spec in dependencies.items():
            package = self._create_package_from_spec(name, spec, section_name)
            if package:
                packages.append(package)

        return packages

    def _create_package_from_spec(
        self, name: str, spec: Any, dep_type: str
    ) -> ExternalPackage | None:
        """Create an ExternalPackage from a name and dependency specification."""
        # Validate crate name
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
            logger.warning("Invalid crate name: %s", name)
            return None

        version_spec, description = self._parse_dependency_spec(spec, dep_type)
        if version_spec is None:
            logger.warning("Unknown dependency specification format for %s: %s", name, spec)
            return None

        # Normalize version specification
        normalized_version = self.normalize_version(version_spec)

        return ExternalPackage(
            name=name,
            version=normalized_version,
            license="Unknown",
            ecosystem="crates.io",
            description=description,
        )

    def _parse_dependency_spec(self, spec: Any, dep_type: str) -> tuple[str | None, str]:
        """Parse dependency specification and return version and description."""
        description = f"Rust crate ({dep_type})"

        if isinstance(spec, str):
            return spec, description

        if isinstance(spec, dict):
            version_spec = self._extract_version_from_dict_spec(spec)
            description = self._build_description_from_dict_spec(spec, description)
            return version_spec, description

        return None, description

    def _extract_version_from_dict_spec(self, spec: dict[str, Any]) -> str:
        """Extract version specification from dictionary spec."""
        if "version" in spec:
            return spec["version"]
        if "git" in spec:
            return self._build_git_version_spec(spec)
        if "path" in spec:
            return f"path:{spec['path']}"
        # Registry dependency without explicit version
        return "*"

    def _build_git_version_spec(self, spec: dict[str, Any]) -> str:
        """Build version specification for git dependencies."""
        version_spec = f"git:{spec['git']}"
        if "branch" in spec:
            version_spec += f"#{spec['branch']}"
        elif "tag" in spec:
            version_spec += f"@{spec['tag']}"
        elif "rev" in spec:
            version_spec += f"@{spec['rev']}"
        return version_spec

    def _build_description_from_dict_spec(self, spec: dict[str, Any], base_description: str) -> str:
        """Build description from dictionary specification."""
        description = base_description

        # Check for optional dependencies
        if spec.get("optional", False):
            description += " (optional)"

        # Check for features
        if "features" in spec:
            features = ", ".join(spec["features"])
            description += f" [features: {features}]"

        return description

    def normalize_version(self, version_spec: str) -> str:
        """Normalize a version specification to a standard format."""
        if not version_spec:
            return "*"

        version_spec = version_spec.strip()

        # Git dependencies
        if version_spec.startswith("git:"):
            return version_spec

        # Path dependencies
        if version_spec.startswith("path:"):
            return version_spec

        # Registry dependencies with version requirements
        # Cargo supports: "1.0", "=1.0.0", ">=1.0", "~1.0", "^1.0", "*"
        return version_spec

    def _parse_cargo_lock(self, cargo_lock_path: Path) -> list[ExternalPackage]:
        """Parse exact versions from Cargo.lock file."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(cargo_lock_path)
            if not content:
                return packages

            assert tomllib is not None, "tomllib/tomli is required for Cargo.lock parsing"
            data = tomllib.loads(content)

            # Cargo.lock has a [[package]] array with all dependencies
            if "package" in data and isinstance(data["package"], list):
                for pkg_data in data["package"]:
                    if isinstance(pkg_data, dict) and "name" in pkg_data and "version" in pkg_data:
                        # Skip the root package (current project)
                        if self._is_root_package(pkg_data):
                            continue

                        source = pkg_data.get(
                            "source",
                            "registry+https://github.com/rust-lang/crates.io-index",
                        )
                        ecosystem = self._determine_ecosystem_from_source(source)

                        package = ExternalPackage(
                            name=pkg_data["name"],
                            version=pkg_data["version"],
                            license="Unknown",
                            ecosystem=ecosystem,
                            description="Rust crate (locked version)",
                        )
                        packages.append(package)

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error parsing Cargo.lock %s: %s", cargo_lock_path, e)

        return packages

    def _is_root_package(self, pkg_data: dict[str, Any]) -> bool:
        """Check if a package entry is the root package (current project)."""
        # Root packages typically don't have a 'source' field
        return "source" not in pkg_data

    def _determine_ecosystem_from_source(self, source: str) -> str:
        """Determine the ecosystem based on the source field in Cargo.lock."""
        if "crates.io" in source:
            return "crates.io"
        if "git+" in source:
            return "git"
        if source.startswith("registry+"):
            return "crates.io"  # Alternative registry, but still crates ecosystem
        return "unknown"

    def _parse_workspace_members(self, repo: Path, main_cargo_toml: Path) -> list[ExternalPackage]:
        """Parse dependencies from workspace member packages."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(main_cargo_toml)
            if not content:
                return packages

            assert tomllib is not None, "tomllib/tomli is required for workspace parsing"
            data = tomllib.loads(content)

            # Check if this is a workspace
            if "workspace" not in data:
                return packages

            workspace = data["workspace"]

            # Get workspace members
            members = workspace.get("members", [])

            # Also check for exclude patterns (negative members)
            exclude = workspace.get("exclude", [])

            # Find workspace member packages
            for member_pattern in members:
                member_packages = self._find_workspace_member_packages(
                    repo, member_pattern, exclude
                )
                packages.extend(member_packages)

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error parsing workspace members: %s", e)

        return packages

    def _find_workspace_member_packages(
        self, repo: Path, pattern: str, exclude: list[str]
    ) -> list[ExternalPackage]:
        """Find packages matching a workspace member pattern."""
        packages: list[ExternalPackage] = []

        try:
            # Handle glob patterns
            if "*" in pattern:
                for member_dir in repo.glob(pattern):
                    if member_dir.is_dir() and not self._is_excluded(member_dir, repo, exclude):
                        cargo_toml = member_dir / "Cargo.toml"
                        if cargo_toml.is_file():
                            member_packages = self._parse_cargo_toml(cargo_toml)
                            packages.extend(member_packages)
            else:
                # Direct path
                member_dir = repo / pattern
                if member_dir.is_dir() and not self._is_excluded(member_dir, repo, exclude):
                    cargo_toml = member_dir / "Cargo.toml"
                    if cargo_toml.is_file():
                        member_packages = self._parse_cargo_toml(cargo_toml)
                        packages.extend(member_packages)

        except (OSError, ValueError) as e:
            logger.error("Error finding workspace member packages for pattern %s: %s", pattern, e)

        return packages

    def _is_excluded(self, member_dir: Path, repo: Path, exclude: list[str]) -> bool:
        """Check if a directory is excluded from the workspace."""
        relative_path = member_dir.relative_to(repo)

        for exclude_pattern in exclude:
            if relative_path.match(exclude_pattern):
                return True

        return False

    def _merge_package_data(
        self,
        manifest_packages: list[ExternalPackage],
        lock_packages: list[ExternalPackage],
    ) -> list[ExternalPackage]:
        """Merge package data from manifest and lockfile."""
        # Create lookup for lockfile packages
        lock_lookup = {pkg.name: pkg for pkg in lock_packages}

        merged_packages = []

        # Update manifest packages with exact versions from lockfile
        for pkg in manifest_packages:
            if pkg.name in lock_lookup:
                lock_pkg = lock_lookup[pkg.name]
                # Use exact version from lockfile but keep other data from manifest
                merged_pkg = ExternalPackage(
                    name=pkg.name,
                    version=lock_pkg.version,  # Exact version from lockfile
                    license=(lock_pkg.license if lock_pkg.license != "Unknown" else pkg.license),
                    ecosystem=lock_pkg.ecosystem,
                    description=pkg.description,  # Keep description from manifest
                )
                merged_packages.append(merged_pkg)
            else:
                # Keep original if not found in lockfile
                merged_packages.append(pkg)

        # Add packages that are only in lockfile (transitive dependencies)
        manifest_names = {pkg.name for pkg in manifest_packages}
        for lock_pkg in lock_packages:
            if lock_pkg.name not in manifest_names:
                # This is a transitive dependency
                transitive_pkg = ExternalPackage(
                    name=lock_pkg.name,
                    version=lock_pkg.version,
                    license=lock_pkg.license,
                    ecosystem=lock_pkg.ecosystem,
                    description="Rust crate (transitive dependency)",
                )
                merged_packages.append(transitive_pkg)

        return merged_packages

    def _deduplicate_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """Remove duplicate packages while preserving order."""
        seen: set[str] = set()
        unique_packages: list[ExternalPackage] = []

        for pkg in packages:
            if pkg.name not in seen:
                seen.add(pkg.name)
                unique_packages.append(pkg)

        return unique_packages
