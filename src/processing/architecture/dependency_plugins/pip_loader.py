"""Plugin for loading Python dependencies using pip."""

from __future__ import annotations

import re
from pathlib import Path

from models.architecture_models import ExternalPackage

from . import AbstractDepLoader


class PipLoader(AbstractDepLoader):
    """Loader for pip requirements.txt files."""

    def detect(self, repo: Path) -> bool:
        """Detect if this is a pip-based Python project."""
        # Look for requirements.txt files
        requirements_files = [
            "requirements.txt",
            "requirements/base.txt",
            "requirements/dev.txt",
            "requirements/prod.txt",
            "requirements/production.txt",
        ]

        for req_file in requirements_files:
            if (repo / req_file).is_file():
                return True

        # Also check for any requirements*.txt files
        for file_path in repo.rglob("requirements*.txt"):
            if file_path.is_file():
                return True

        return False

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse pip requirements files."""
        packages: list[ExternalPackage] = []

        # Find all requirements files
        requirements_files = []

        # Standard locations
        standard_files = [
            "requirements.txt",
            "requirements/base.txt",
            "requirements/dev.txt",
            "requirements/prod.txt",
            "requirements/production.txt",
        ]

        for req_filename in standard_files:
            file_path = repo / req_filename
            if file_path.is_file():
                requirements_files.append(file_path)

        # Search for any other requirements*.txt files
        for file_path in repo.rglob("requirements*.txt"):
            if file_path.is_file() and file_path not in requirements_files:
                requirements_files.append(file_path)

        # Parse each requirements file
        for req_file in requirements_files:
            packages.extend(self._parse_requirements_file(req_file))

        # Remove duplicates while preserving order
        seen = set()
        unique_packages = []
        for pkg in packages:
            if pkg.name not in seen:
                seen.add(pkg.name)
                unique_packages.append(pkg)

        return unique_packages

    def _parse_requirements_file(self, file_path: Path) -> list[ExternalPackage]:
        """Parse a single requirements.txt file."""
        packages: list[ExternalPackage] = []
        content = self._read_file_safely(file_path)

        if not content:
            return packages

        for line in content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Skip -r/-e/--editable directives for now
            if line.startswith(("-r", "-e", "--editable")):
                continue

            # Parse package specification
            package = self._parse_package_spec(line)
            if package:
                packages.append(package)

        return packages

    def _parse_package_spec(self, spec: str) -> ExternalPackage | None:
        """Parse a single package specification line."""
        # Remove inline comments
        if "#" in spec:
            spec = spec.split("#")[0].strip()

        if not spec:
            return None

        # Handle different version specifiers
        # Examples: package==1.0.0, package>=1.0, package~=1.0, package[extra]==1.0

        # Extract package name and version constraint
        # This regex handles: name[extras]==version, name>=version, etc.
        pattern = r"^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*([><=!~]+)?\s*([^\s,;]+)?"
        match = re.match(pattern, spec)

        if not match:
            # Try more straightforward pattern for just package name
            simple_pattern = r"^([a-zA-Z0-9_-]+)"
            simple_match = re.match(simple_pattern, spec)
            if simple_match:
                name = simple_match.group(1)
                return ExternalPackage(name=name, version="*", license="Unknown", ecosystem="pypi")
            return None

        name = match.group(1)
        operator = match.group(2) or ""
        version = match.group(3) or "*"

        # Clean up extras from name if present
        if "[" in name:
            name = name.split("[")[0]

        # For exact versions (==), use just the version
        if operator == "==":
            current_version = version
        else:
            # For other operators, store the full constraint
            current_version = f"{operator}{version}" if operator else version

        return ExternalPackage(
            name=name,
            version=current_version,
            license="Unknown",
            ecosystem="pypi",
        )
