"""JavaScript/TypeScript dependency loader for npm, yarn, and pnpm package managers.

This module provides functionality to detect and parse JavaScript/TypeScript
projects and extract their dependency information from package.json files
and associated lockfiles (package-lock.json, yarn.lock, pnpm-lock.yaml).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from models.architecture_models import ExternalPackage

from . import AbstractDepLoader

logger = logging.getLogger(__name__)


class JSDependencyLoader(AbstractDepLoader):
    """Loader for JavaScript/TypeScript dependencies from npm, yarn, and pnpm."""

    def __init__(self):
        """Initialize the JS dependency loader."""
        self.package_managers = {
            "npm": {
                "manifest": "package.json",
                "lockfile": "package-lock.json",
                "detect_files": ["package.json", "package-lock.json"],
            },
            "yarn": {
                "manifest": "package.json",
                "lockfile": "yarn.lock",
                "detect_files": ["package.json", "yarn.lock"],
            },
            "pnpm": {
                "manifest": "package.json",
                "lockfile": "pnpm-lock.yaml",
                "detect_files": ["package.json", "pnpm-lock.yaml"],
            },
        }

    def detect(self, repo: Path) -> bool:
        """Detect if this is a JavaScript/TypeScript project with package management."""
        # Primary detection: package.json file
        if (repo / "package.json").is_file():
            return True

        # Secondary detection: lockfiles without package.json (edge case)
        lockfiles = ["package-lock.json", "yarn.lock", "pnpm-lock.yaml"]
        for lockfile in lockfiles:
            if (repo / lockfile).is_file():
                logger.warning("Found %s without package.json in %s", lockfile, repo)
                return True

        # Workspace detection: look for nested package.json files
        for package_json in repo.rglob("package.json"):
            if package_json.is_file():
                return True

        return False

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse JavaScript dependencies from package.json and lockfiles."""
        packages: list[ExternalPackage] = []

        try:
            # Determine which package manager is being used
            package_manager = self._detect_package_manager(repo)
            logger.info("Detected package manager: %s", package_manager)

            # Parse main package.json
            main_package_json = repo / "package.json"
            if main_package_json.is_file():
                packages.extend(self._parse_package_json(main_package_json))

            # Parse lockfile for exact versions if available
            if package_manager in self.package_managers:
                lockfile_name = self.package_managers[package_manager]["lockfile"]
                lockfile_path = repo / str(lockfile_name)
                if lockfile_path.is_file():
                    lock_packages = self._parse_lockfile(lockfile_path, package_manager)
                    # Merge lockfile data with package.json data
                    packages = self._merge_package_data(packages, lock_packages)

            # Handle workspaces/monorepos
            workspace_packages = self._parse_workspaces(repo)
            packages.extend(workspace_packages)

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("File system error parsing JavaScript dependencies: %s", e)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error("Parse error in JavaScript dependencies: %s", e)
        except (TypeError, ValueError, AttributeError) as e:
            logger.error("Unexpected error parsing JavaScript dependencies: %s", e)

        # Remove duplicates and return
        return self._deduplicate_packages(packages)

    def _detect_package_manager(self, repo: Path) -> str:
        """Detect which package manager is being used."""
        # Check for lockfiles in order of preference
        if (repo / "pnpm-lock.yaml").is_file():
            return "pnpm"
        if (repo / "yarn.lock").is_file():
            return "yarn"
        if (repo / "package-lock.json").is_file():
            return "npm"
        # Default to npm if only package.json exists
        return "npm"

    def _parse_package_json(self, package_json_path: Path) -> list[ExternalPackage]:
        """Parse dependencies from a package.json file."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(package_json_path)
            if not content:
                return packages

            data = json.loads(content)

            # Extract different types of dependencies
            dep_types = [
                "dependencies",
                "devDependencies",
                "peerDependencies",
                "optionalDependencies",
            ]

            for dep_type in dep_types:
                if dep_type in data and isinstance(data[dep_type], dict):
                    for name, version_spec in data[dep_type].items():
                        package = self._create_package_from_spec(name, version_spec, dep_type)
                        if package:
                            packages.append(package)

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", package_json_path, e)
        except (TypeError, ValueError, AttributeError) as e:
            logger.error("Error parsing %s: %s", package_json_path, e)

        return packages

    def _create_package_from_spec(
        self, name: str, version_spec: str, dep_type: str
    ) -> ExternalPackage | None:
        """Create an ExternalPackage from a name and version specification."""
        # Handle scoped packages (@organization/package-name)
        if name.startswith("@") and "/" in name:
            # Scoped package is valid
            pass
        elif not re.match(r"^[a-zA-Z0-9._-]+$", name.replace("@", "").replace("/", "")):
            logger.warning("Invalid package name: %s", name)
            return None

        # Normalize version specification
        normalized_version = self.normalize_version(version_spec)

        return ExternalPackage(
            name=name,
            version=normalized_version,
            license="Unknown",
            ecosystem="npm",
            description=f"JavaScript package ({dep_type})",
        )

    def normalize_version(self, version_spec: str) -> str:
        """Normalize a version specification to a standard format."""
        if not version_spec:
            return "*"

        # Handle different version specification types
        version_spec = version_spec.strip()

        # Determine the appropriate prefix based on version specification type
        if version_spec.startswith(("git+", "git://", "https://github.com", "github:")):
            result = f"git:{version_spec}"
        elif version_spec.startswith(("file:", "./", "../")):
            result = f"file:{version_spec}"
        elif version_spec.startswith(("http://", "https://")):
            result = f"url:{version_spec}"
        elif version_spec == "workspace:*" or version_spec.startswith("workspace:"):
            result = version_spec
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9-]*$", version_spec):
            result = f"tag:{version_spec}"
        else:
            # Standard semver ranges
            result = version_spec

        return result

    def _parse_lockfile(self, lockfile_path: Path, package_manager: str) -> list[ExternalPackage]:
        """Parse exact versions from lockfiles."""
        packages: list[ExternalPackage] = []

        try:
            if package_manager == "npm":
                packages = self._parse_npm_lockfile(lockfile_path)
            elif package_manager == "yarn":
                packages = self._parse_yarn_lockfile(lockfile_path)
            elif package_manager == "pnpm":
                packages = self._parse_pnpm_lockfile(lockfile_path)
        except (TypeError, ValueError, AttributeError, UnicodeDecodeError) as e:
            logger.error("Error parsing %s: %s", lockfile_path, e)

        return packages

    def _parse_npm_lockfile(self, lockfile_path: Path) -> list[ExternalPackage]:
        """Parse npm package-lock.json file."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(lockfile_path)
            if not content:
                return packages

            data = json.loads(content)

            # Handle both lockfile v1 and v2/v3 formats
            if "packages" in data:
                # Lockfile v2/v3 format
                for path, info in data["packages"].items():
                    if path == "":  # Root package
                        continue

                    # Extract package name from path
                    name = self._extract_package_name_from_path(path)
                    if name and "version" in info:
                        package = ExternalPackage(
                            name=name,
                            version=info["version"],
                            license=info.get("license", "Unknown"),
                            ecosystem="npm",
                            description="JavaScript package (npm lockfile)",
                        )
                        packages.append(package)

            elif "dependencies" in data:
                # Lockfile v1 format
                packages.extend(self._parse_npm_v1_dependencies(data["dependencies"]))

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in npm lockfile %s: %s", lockfile_path, e)

        return packages

    def _parse_npm_v1_dependencies(self, dependencies: dict[str, Any]) -> list[ExternalPackage]:
        """Parse dependencies from npm lockfile v1 format."""
        packages: list[ExternalPackage] = []

        for name, info in dependencies.items():
            if isinstance(info, dict) and "version" in info:
                package = ExternalPackage(
                    name=name,
                    version=info["version"],
                    license=info.get("license", "Unknown"),
                    ecosystem="npm",
                )
                packages.append(package)

                # Recursively parse nested dependencies
                if "dependencies" in info:
                    packages.extend(self._parse_npm_v1_dependencies(info["dependencies"]))

        return packages

    def _parse_yarn_lockfile(self, lockfile_path: Path) -> list[ExternalPackage]:
        """Parse yarn.lock file (yarn v1 format)."""
        packages: list[ExternalPackage] = []

        try:
            content = self._read_file_safely(lockfile_path)
            if not content:
                return packages

            # Parse yarn.lock format (not JSON, custom format)
            current_package = None
            current_version = None

            for line in content.splitlines():
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Package declaration line
                if line.endswith(":") and not line.startswith(" "):
                    # Extract package name from declaration
                    package_spec = line.rstrip(":")
                    current_package = self._extract_yarn_package_name(package_spec)
                    current_version = None

                # Version line
                elif line.startswith("version ") and current_package:
                    current_version = line.split("version ")[1].strip().strip('"')

                    package = ExternalPackage(
                        name=current_package,
                        version=current_version,
                        license="Unknown",
                        ecosystem="npm",
                        description="JavaScript package (yarn lockfile)",
                    )
                    packages.append(package)

        except (TypeError, ValueError, AttributeError, UnicodeDecodeError) as e:
            logger.error("Error parsing yarn lockfile: %s", e)

        return packages

    def _parse_pnpm_lockfile(self, lockfile_path: Path) -> list[ExternalPackage]:
        """Parse pnpm-lock.yaml file."""
        packages: list[ExternalPackage] = []

        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot parse pnpm-lock.yaml")
            return packages

        try:
            content = self._read_file_safely(lockfile_path)
            if not content:
                return packages

            data = yaml.safe_load(content)

            # Parse packages section
            if "packages" in data:
                for package_spec, info in data["packages"].items():
                    name, version = self._parse_pnpm_package_spec(package_spec)
                    if name and version:
                        package = ExternalPackage(
                            name=name,
                            version=version,
                            license=info.get("license", "Unknown"),
                            ecosystem="npm",
                            description="JavaScript package (pnpm lockfile)",
                        )
                        packages.append(package)

        except (
            yaml.YAMLError,
            TypeError,
            ValueError,
            AttributeError,
            UnicodeDecodeError,
        ) as e:
            logger.error("Error parsing pnpm lockfile: %s", e)

        return packages

    def _extract_package_name_from_path(self, path: str) -> str | None:
        """Extract package name from npm lockfile path."""
        # Remove "node_modules/" prefix and handle scoped packages
        if path.startswith("node_modules/"):
            path = path[13:]  # Remove "node_modules/"

        # Handle scoped packages
        if path.startswith("@"):
            parts = path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        else:
            # Regular package
            return path.split("/")[0]

        return None

    def _extract_yarn_package_name(self, package_spec: str) -> str | None:
        """Extract package name from yarn lockfile package specification."""
        # Format: "package@version" or "@scope/package@version"
        if "@" in package_spec:
            if package_spec.startswith("@"):
                # Scoped package: @scope/package@version
                parts = package_spec.split("@")
                if len(parts) >= 3:
                    return f"@{parts[1]}"
            else:
                # Regular package: package@version
                return package_spec.split("@")[0]

        return package_spec

    def _parse_pnpm_package_spec(self, package_spec: str) -> tuple[str | None, str | None]:
        """Parse pnpm package specification to extract name and version."""
        # Format: "/package/version" or "/@scope/package/version"
        if package_spec.startswith("/"):
            package_spec = package_spec[1:]  # Remove leading slash

        parts = package_spec.split("/")

        if package_spec.startswith("@"):
            # Scoped package: @scope/package/version
            if len(parts) >= 3:
                name = f"@{parts[0]}/{parts[1]}"
                version = parts[2]
                return name, version
        else:
            # Regular package: package/version
            if len(parts) >= 2:
                name = parts[0]
                version = parts[1]
                return name, version

        return None, None

    def _parse_workspaces(self, repo: Path) -> list[ExternalPackage]:
        """Parse dependencies from workspace/monorepo packages."""
        packages: list[ExternalPackage] = []

        try:
            # Check main package.json for workspace configuration
            main_package_json = repo / "package.json"
            if not main_package_json.is_file():
                return packages

            content = self._read_file_safely(main_package_json)
            if not content:
                return packages

            data = json.loads(content)

            # Get workspace patterns
            workspace_patterns = []
            if "workspaces" in data:
                if isinstance(data["workspaces"], list):
                    workspace_patterns = data["workspaces"]
                elif isinstance(data["workspaces"], dict) and "packages" in data["workspaces"]:
                    workspace_patterns = data["workspaces"]["packages"]

            # Find workspace packages
            for pattern in workspace_patterns:
                workspace_packages = self._find_workspace_packages(repo, pattern)
                packages.extend(workspace_packages)

        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.error("Error parsing workspaces: %s", e)

        return packages

    def _find_workspace_packages(self, repo: Path, pattern: str) -> list[ExternalPackage]:
        """Find packages matching a workspace pattern."""
        packages: list[ExternalPackage] = []

        try:
            # Find all package.json files in matching directories
            for workspace_dir in repo.glob(pattern):
                if workspace_dir.is_dir():
                    package_json = workspace_dir / "package.json"
                    if package_json.is_file():
                        workspace_packages = self._parse_package_json(package_json)
                        packages.extend(workspace_packages)

        except (OSError, ValueError, AttributeError) as e:
            logger.error("Error finding workspace packages for pattern %s: %s", pattern, e)

        return packages

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
                    ecosystem=pkg.ecosystem,
                    description=pkg.description,
                )
                merged_packages.append(merged_pkg)
            else:
                # Keep original if not found in lockfile
                merged_packages.append(pkg)

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
