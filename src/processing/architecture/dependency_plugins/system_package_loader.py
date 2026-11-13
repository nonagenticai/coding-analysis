"""System package dependency loader for detecting and managing OS-level dependencies.

This module provides functionality to detect system package dependencies required by
Python packages, parse setup.py/pyproject.toml for system requirements, and integrate
with package managers like apt, yum, dnf, brew, etc.

The loader can:
- Parse Python package metadata for system library requirements
- Use apt-file to find packages that provide missing libraries
- Detect common patterns in error messages for missing system dependencies
- Generate installation commands for various package managers
- Integrate with container environments for automatic installation
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from models.architecture_models import ExternalPackage
from models.system_models import PackageManager

from . import AbstractDepLoader


@dataclass(frozen=True)
class DetectedSystemPackage:
    """Represents a detected system-level package requirement.

    Note: This class is frozen (immutable) to allow it to be used in sets for deduplication.
    All collection fields use tuples instead of lists for hashability.
    """

    name: str
    package_manager: PackageManager
    library_name: str | None = None
    header_files: tuple = field(default_factory=tuple)  # Immutable for hashability
    description: str | None = None
    required_by: tuple = field(default_factory=tuple)  # Immutable for hashability


class SystemPackageLoader(AbstractDepLoader):
    """Loader for detecting and managing system package dependencies."""

    # Common Python packages that require system dependencies
    KNOWN_DEPENDENCIES = {
        # Scientific computing
        "numpy": ["libblas-dev", "liblapack-dev", "gfortran"],
        "scipy": ["libblas-dev", "liblapack-dev", "gfortran", "libatlas-base-dev"],
        "pandas": ["python3-dev"],
        # Database drivers
        "psycopg2": ["libpq-dev", "postgresql-client"],
        "psycopg2-binary": ["libpq-dev"],
        "mysqlclient": ["default-libmysqlclient-dev", "build-essential"],
        "cx_Oracle": ["libaio1"],
        # Image processing
        "pillow": ["libjpeg-dev", "libpng-dev", "libtiff-dev", "libwebp-dev"],
        "opencv-python": ["libopencv-dev", "python3-opencv"],
        # XML/HTML parsing
        "lxml": ["libxml2-dev", "libxslt1-dev"],
        # Cryptography
        "cryptography": ["libssl-dev", "libffi-dev", "python3-dev"],
        # Other common packages
        "pycairo": ["libcairo2-dev"],
        "pygraphviz": ["graphviz", "graphviz-dev"],
        "pyaudio": ["portaudio19-dev"],
        "python-ldap": ["libldap2-dev", "libsasl2-dev"],
        "python-magic": ["libmagic1"],
    }

    # Mapping of error patterns to system packages
    ERROR_PATTERNS = {
        r"error: Microsoft Visual C\+\+ \d+\.\d+ is required": {
            "packages": ["build-essential", "gcc", "g++"],
            "description": "C/C++ compiler required",
        },
        r"fatal error: Python\.h: No such file or directory": {
            "packages": ["python3-dev"],
            "description": "Python development headers required",
        },
        r"fatal error: openssl/.*\.h: No such file or directory": {
            "packages": ["libssl-dev"],
            "description": "OpenSSL development headers required",
        },
        r"pg_config.*not found": {
            "packages": ["libpq-dev", "postgresql-client"],
            "description": "PostgreSQL development libraries required",
        },
        r"mysql_config.*not found": {
            "packages": ["default-libmysqlclient-dev"],
            "description": "MySQL development libraries required",
        },
        r"xml2-config.*not found": {
            "packages": ["libxml2-dev"],
            "description": "libxml2 development libraries required",
        },
        r"fatal error: ffi\.h: No such file or directory": {
            "packages": ["libffi-dev"],
            "description": "Foreign Function Interface libraries required",
        },
    }

    def __init__(self):
        """Initialize the system package loader."""
        self._apt_file_cache: dict[str, list[str]] = {}
        self._detected_os: str | None = None
        self._package_manager: PackageManager = PackageManager.APT

    def detect(self, repo: Path) -> bool:
        """Detect if this repository might need system dependencies."""
        # Always return True as any Python project might need system deps
        # The actual detection happens when parsing
        return True

    def parse(self, repo: Path) -> list[ExternalPackage]:
        """Parse and detect system package dependencies."""
        system_deps: set[DetectedSystemPackage] = set()

        # Detect package manager for the current system
        self._detect_package_manager()

        # 1. Parse Python dependency files
        python_packages = self._extract_python_packages(repo)

        # 2. Check known dependencies
        for pkg in python_packages:
            if pkg in self.KNOWN_DEPENDENCIES:
                for sys_pkg in self.KNOWN_DEPENDENCIES[pkg]:
                    dep = DetectedSystemPackage(
                        name=sys_pkg,
                        package_manager=self._package_manager,
                        required_by=(pkg,),  # Use tuple for immutable hashable dataclass
                    )
                    system_deps.add(dep)

        # 3. Parse setup.py for system_requirements or comments
        setup_deps = self._parse_setup_py(repo)
        system_deps.update(setup_deps)

        # 4. Parse pyproject.toml for build requirements
        pyproject_deps = self._parse_pyproject_toml(repo)
        system_deps.update(pyproject_deps)

        # 5. Check for error logs or build failures
        error_deps = self._check_error_logs(repo)
        system_deps.update(error_deps)

        # Convert to ExternalPackage format
        packages = []
        for dep in system_deps:
            packages.append(
                ExternalPackage(
                    name=dep.name,
                    version="*",  # System packages don't usually have version constraints
                    license="System",
                    ecosystem=f"system-{dep.package_manager.value}",
                    description=dep.description or f"Required by: {', '.join(dep.required_by)}",
                )
            )

        return packages

    def _detect_package_manager(self) -> None:
        """Detect the system's package manager."""
        # Check for various package managers
        if self._command_exists("apt-get"):
            self._package_manager = PackageManager.APT
            self._detected_os = "debian/ubuntu"
        elif self._command_exists("yum"):
            self._package_manager = PackageManager.YUM
            self._detected_os = "redhat/centos"
        elif self._command_exists("dnf"):
            self._package_manager = PackageManager.DNF
            self._detected_os = "fedora"
        elif self._command_exists("brew"):
            self._package_manager = PackageManager.BREW
            self._detected_os = "macos"
        elif self._command_exists("pacman"):
            self._package_manager = PackageManager.PACMAN
            self._detected_os = "arch"
        elif self._command_exists("apk"):
            self._package_manager = PackageManager.APK
            self._detected_os = "alpine"
        else:
            # Default to APT as it's most common in containers
            self._package_manager = PackageManager.APT
            self._detected_os = "unknown"

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the system."""
        try:
            subprocess.run(["which", command], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _extract_python_packages(self, repo: Path) -> set[str]:
        """Extract Python package names from various dependency files."""
        packages = set()

        # Check requirements.txt files
        for req_file in repo.rglob("requirements*.txt"):
            content = self._read_file_safely(req_file)
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name
                    pkg_name = re.split(r"[<>=!~]", line)[0].strip()
                    if pkg_name:
                        packages.add(pkg_name.lower())

        # Check setup.py
        setup_py = repo / "setup.py"
        if setup_py.exists():
            content = self._read_file_safely(setup_py)
            # Look for install_requires
            install_requires_match = re.search(
                r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
            )
            if install_requires_match:
                requires_content = install_requires_match.group(1)
                for match in re.finditer(r'["\']([^"\']+)["\']', requires_content):
                    pkg_spec = match.group(1)
                    pkg_name = re.split(r"[<>=!~]", pkg_spec)[0].strip()
                    if pkg_name:
                        packages.add(pkg_name.lower())

        # Check pyproject.toml
        pyproject = repo / "pyproject.toml"
        if pyproject.exists():
            content = self._read_file_safely(pyproject)
            # Simple regex-based extraction
            for match in re.finditer(r'^\s*"?([a-zA-Z0-9_-]+)"?\s*[=>]', content, re.MULTILINE):
                pkg_name = match.group(1)
                if pkg_name and pkg_name.lower() != "python":
                    packages.add(pkg_name.lower())

        return packages

    def _parse_setup_py(self, repo: Path) -> set[DetectedSystemPackage]:
        """Parse setup.py for system requirements."""
        deps = set()
        setup_py = repo / "setup.py"

        if not setup_py.exists():
            return deps

        content = self._read_file_safely(setup_py)

        # Look for system_requirements or similar patterns
        # Common patterns in setup.py comments or documentation
        patterns = [
            r"#\s*system[- ]?requirements?:\s*(.+)",
            r"#\s*apt[- ]?get\s+install\s+(.+)",
            r"#\s*requires?:\s*lib(\w+)",
            r'""".*?system.*?requirements?.*?:\s*(.+?)"""',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                requirement_text = match.group(1)
                # Extract package names
                for pkg in re.findall(r"[\w-]+(?:-dev)?", requirement_text):
                    if pkg and not pkg.startswith("-"):
                        deps.add(
                            DetectedSystemPackage(
                                name=pkg,
                                package_manager=self._package_manager,
                                description="Found in setup.py",
                            )
                        )

        return deps

    def _parse_pyproject_toml(self, repo: Path) -> set[DetectedSystemPackage]:
        """Parse pyproject.toml for build system requirements."""
        deps = set()
        pyproject = repo / "pyproject.toml"

        if not pyproject.exists():
            return deps

        content = self._read_file_safely(pyproject)

        # Look for build-system requirements that might indicate system deps
        # Also check for comments about system requirements
        if "manylinux" in content or "wheel" in content:
            # Projects building wheels often need build tools
            deps.add(
                DetectedSystemPackage(
                    name="build-essential",
                    package_manager=self._package_manager,
                    description="Build tools for Python wheels",
                )
            )

        # Check for specific build backends that need system deps
        if "maturin" in content:  # Rust/Python projects
            deps.add(
                DetectedSystemPackage(
                    name="cargo",
                    package_manager=self._package_manager,
                    description="Rust toolchain for maturin",
                )
            )
            deps.add(
                DetectedSystemPackage(
                    name="rustc",
                    package_manager=self._package_manager,
                    description="Rust compiler for maturin",
                )
            )

        return deps

    def _check_error_logs(self, repo: Path) -> set[DetectedSystemPackage]:
        """Check for error logs that indicate missing system dependencies."""
        deps = set()

        # Look for common build error log files
        log_patterns = ["*.log", "build/*.log", "pip-log.txt", "install.log"]

        for pattern in log_patterns:
            for log_file in repo.glob(pattern):
                if log_file.is_file():
                    content = self._read_file_safely(log_file)

                    # Check against known error patterns
                    for error_pattern, dep_info in self.ERROR_PATTERNS.items():
                        if re.search(error_pattern, content, re.IGNORECASE):
                            for pkg in dep_info["packages"]:
                                deps.add(
                                    DetectedSystemPackage(
                                        name=pkg,
                                        package_manager=self._package_manager,
                                        description=str(dep_info["description"]),
                                    )
                                )

        return deps

    def find_package_for_library(self, library_name: str) -> str | None:
        """Use apt-file or similar tools to find which package provides a library."""
        if self._package_manager != PackageManager.APT:
            # apt-file is Debian/Ubuntu specific
            return None

        # Check cache first
        if library_name in self._apt_file_cache:
            packages = self._apt_file_cache[library_name]
            return packages[0] if packages else None

        try:
            # Use apt-file to search for the library
            result = subprocess.run(
                ["apt-file", "search", library_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the output
            packages = []
            for line in result.stdout.splitlines():
                if line.strip():
                    # apt-file output format: "package: /path/to/file"
                    package = line.split(":")[0].strip()
                    if package:
                        packages.append(package)

            # Cache the result
            self._apt_file_cache[library_name] = packages

            # Prefer -dev packages for development headers
            dev_packages = [p for p in packages if p.endswith("-dev")]
            if dev_packages:
                return dev_packages[0]

            return packages[0] if packages else None

        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def generate_install_command(self, packages: list[str]) -> str:
        """Generate the appropriate install command for the detected package manager."""
        if not packages:
            return ""

        commands = {
            PackageManager.APT: f"apt-get update && apt-get install -y {' '.join(packages)}",
            PackageManager.YUM: f"yum install -y {' '.join(packages)}",
            PackageManager.DNF: f"dnf install -y {' '.join(packages)}",
            PackageManager.BREW: f"brew install {' '.join(packages)}",
            PackageManager.PACMAN: f"pacman -S --noconfirm {' '.join(packages)}",
            PackageManager.APK: f"apk add --no-cache {' '.join(packages)}",
        }

        return commands.get(self._package_manager, "")
