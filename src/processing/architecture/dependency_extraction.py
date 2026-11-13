"""
Dependency Extraction and Parsing Module

Handles dependency discovery, parsing, and basic processing including
deduplication and version normalization.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from packaging.version import InvalidVersion, Version

from models.architecture_models import ExternalPackage

from .dependency_plugins import BaseDepLoader
from .dependency_plugins.js_loader import JSDependencyLoader
from .dependency_plugins.pip_loader import PipLoader
from .dependency_plugins.poetry_loader import PoetryLoader
from .dependency_plugins.rust_loader import RustDependencyLoader
from .dependency_plugins.system_package_loader import SystemPackageLoader
from .dependency_plugins.uv_loader import UvLoader

logger = logging.getLogger(__name__)


class DependencyExtractionError(RuntimeError):
    """Every loader that found a manifest failed to parse it.

    Distinct from "this repository genuinely declares no dependencies". Both used
    to return an empty list, so a repo whose manifests could not be read scored
    ``risk_score = 0.0`` — reported to the user as a clean bill of health for a
    dependency set that was never examined. The caller turns this into an
    ``inconclusive`` report instead (``risk_score = -1.0``), mirroring how an OSV
    outage is already handled.
    """

    def __init__(self, repo: Path, failures: list[str]) -> None:
        self.repo = repo
        self.failures = failures
        super().__init__(
            f"all {len(failures)} dependency loader(s) that detected a manifest in {repo} "
            f"failed: {'; '.join(failures)}"
        )


class DependencyExtractor:
    """
    Handles dependency extraction and parsing from various package managers.

    Supports multiple package managers and provides package deduplication
    and version normalization functionality.
    """

    def __init__(self) -> None:
        """Initialize the dependency extractor with all available loaders."""
        self.loaders: list[BaseDepLoader] = [
            PipLoader(),
            PoetryLoader(),
            UvLoader(),
            JSDependencyLoader(),
            RustDependencyLoader(),
            SystemPackageLoader(),
        ]

        logger.info(
            "DependencyExtractor initialized with %d loaders",
            len(self.loaders),
        )

    def extract_dependencies(self, repo: Path) -> list[ExternalPackage]:
        """
        Extract dependencies from the repository using all available loaders.

        Args:
            repo: Path to the repository to analyze

        Returns:
            List of extracted packages from all detected package managers

        Raises:
            ValueError: If repo path is invalid
            RuntimeError: If extraction fails for all loaders
        """
        if not repo.exists():
            raise ValueError(f"Repository path does not exist: {repo}")

        logger.info("Starting dependency extraction for repository at: %s", repo)

        packages: list[ExternalPackage] = []
        # Loaders that detected a manifest and then failed to parse it. Distinguishes
        # a genuinely dependency-free repo from one we simply could not read.
        failures: list[str] = []

        # Try each loader
        for loader in self.loaders:
            logger.debug(
                "Checking loader %s for repository at %s",
                loader.__class__.__name__,
                repo,
            )
            try:
                if loader.detect(repo):
                    logger.info(
                        "Using %s to parse dependencies",
                        loader.__class__.__name__,
                    )
                    loader_packages = loader.parse(repo)
                    packages.extend(loader_packages)
                    logger.debug(
                        "Found %d packages with %s",
                        len(loader_packages),
                        loader.__class__.__name__,
                    )
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Loader %s failed: %s", loader.__class__.__name__, e)
                failures.append(f"{loader.__class__.__name__}: {e}")
                continue

        if not packages:
            # "No manifest exists" and "every manifest we found failed to parse"
            # are NOT the same answer. Returning [] for both made the caller
            # report `risk_score = 0.0` — a clean bill of health — for a repo
            # whose dependencies were never actually read. Only the first case
            # is legitimately empty; the second is a failed extraction, which is
            # what this method's own docstring has always promised to raise on.
            if failures:
                raise DependencyExtractionError(repo, failures)
            logger.warning("No dependencies found in repository at %s", repo)
            return []

        # Remove duplicates
        unique_packages = self.deduplicate_packages(packages)
        logger.info(
            "Found %d unique packages after deduplication",
            len(unique_packages),
        )

        return unique_packages

    def deduplicate_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """
        Remove duplicate packages, keeping the most specific version.

        Args:
            packages: List of packages that may contain duplicates

        Returns:
            List of unique packages with most specific versions
        """
        package_map = {}

        for package in packages:
            name = package.name.lower()

            if name not in package_map:
                package_map[name] = package
            else:
                # Keep the package with the most specific version
                existing = package_map[name]
                if self._is_more_specific_version(
                    package.version or package.current_version or "*",
                    existing.version or existing.current_version or "*",
                ):
                    package_map[name] = package

        return list(package_map.values())

    def _is_more_specific_version(self, version1: str, version2: str) -> bool:
        """
        Check if version1 is more specific than version2.

        Args:
            version1: First version string to compare
            version2: Second version string to compare

        Returns:
            True if version1 is more specific than version2
        """
        # Exact versions are more specific than constraints
        if "==" in version2 and "==" not in version1:
            return False
        if "==" in version1 and "==" not in version2:
            return True

        # Specific versions are more specific than wildcards
        if version2 == "*":
            return True
        if version1 == "*":
            return False

        return False

    def enrich_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """
        Enrich packages with latest version and license information.

        Args:
            packages: List of packages to enrich

        Returns:
            List of enriched packages with additional metadata
        """
        enriched = []

        for package in packages:
            enriched_package = ExternalPackage(
                name=package.name,
                version=package.version or package.current_version or "*",
                license=self._get_license_info(package.name) or "Unknown",
                ecosystem=package.ecosystem,
                description=package.description,
                # Legacy fields for compatibility
                current_version=package.current_version,
                latest_version=self._get_latest_version(package.name),
                severity=package.severity,
            )
            enriched.append(enriched_package)

        return enriched

    def _get_latest_version(self, package_name: str) -> str | None:
        """
        Get the latest version of a package from PyPI.

        Args:
            package_name: Name of the package to check

        Returns:
            Latest version string or None if not found
        """
        try:
            response = httpx.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)

            if response.status_code == 200:
                data = response.json()
                version = data.get("info", {}).get("version")
                return version if isinstance(version, str) else None

        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return None

    def _get_license_info(self, package_name: str) -> str | None:
        """
        Get license information for a package.

        Args:
            package_name: Name of the package to check

        Returns:
            License string or None if not found
        """
        try:
            response = httpx.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)

            if response.status_code == 200:
                data = response.json()
                info = data.get("info", {})

                # Try different license fields
                license_info = info.get("license") or info.get("license_expression") or ""

                # Also check classifiers for license info
                classifiers = info.get("classifiers", [])
                for classifier in classifiers:
                    if classifier.startswith("License ::"):
                        license_from_classifier = classifier.split("::")[-1].strip()
                        if license_from_classifier and license_from_classifier != "OSI Approved":
                            return str(license_from_classifier)

                return license_info if license_info else None

        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return None

    def parse_pypi_requirement(self, requirement: str) -> tuple[str | None, str | None]:
        """
        Parse a PyPI requirement string to extract name and version.

        Args:
            requirement: PyPI requirement string (e.g., 'package>=1.0')

        Returns:
            Tuple of (package_name, version) or (None, None) if parsing fails
        """
        # Handle requirements with extras: package[extra]==version
        requirement = re.sub(r"\[.*?\]", "", requirement)

        # Handle various version specifiers
        patterns = [
            r"^([a-zA-Z0-9_.-]+)\s*([><=!~]+)\s*([^\s;,]+)",  # package>=1.0
            r"^([a-zA-Z0-9_.-]+)\s*$",  # just package name
        ]

        for pattern in patterns:
            match = re.match(pattern, requirement.strip())
            if match:
                name = match.group(1)
                version = match.group(3) if len(match.groups()) >= 3 else None
                return name, version

        return None, None

    def clean_version_for_osv(self, version: str) -> str | None:
        """
        Clean version string for OSV API compatibility.

        Args:
            version: Version string to clean

        Returns:
            Cleaned version string or None if invalid
        """
        # Remove operators and constraints
        version = version.replace("==", "").replace(">=", "").replace("<=", "")
        version = version.replace("~=", "").replace("!=", "").replace(">", "").replace("<", "")
        version = version.strip()

        if version == "*" or not version:
            return None

        try:
            # Validate version format
            Version(version)
            return version
        except InvalidVersion:
            return None

    def get_outdated_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """
        Get packages that have newer versions available.

        Args:
            packages: List of packages to check

        Returns:
            List of packages that are outdated
        """
        outdated = []

        for package in packages:
            if package.latest_version and package.current_version != package.latest_version:
                try:
                    current = self.clean_version_for_osv(
                        package.current_version or package.version or "*"
                    )
                    latest = package.latest_version

                    if not current:
                        outdated.append(package)
                    elif Version(current) < Version(latest):
                        outdated.append(package)
                except (InvalidVersion, ValueError, TypeError):  # fmt: skip
                    # If version comparison fails, consider it outdated
                    outdated.append(package)

        return outdated

    def get_vulnerable_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """
        Get packages with known vulnerabilities.

        Args:
            packages: List of packages to check

        Returns:
            List of packages with vulnerabilities
        """
        return [pkg for pkg in packages if pkg.severity and pkg.severity != "low"]
