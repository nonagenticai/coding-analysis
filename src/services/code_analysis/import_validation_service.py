"""Import validation service module."""

from __future__ import annotations

from typing import Any

"""Import Validation Service

Handles comprehensive validation of imports and dependencies in code.
Provides detailed analysis of import statements, dependency resolution,
and compatibility checking across different environments.

This service validates imports against standard library modules, third-party
packages, and local project dependencies to ensure code reliability and
proper dependency management.
"""
import logging
import os

import psycopg
from pydantic import BaseModel

from db.session import get_raw_connection
from services.dependency_parser import DependencyParser

logger = logging.getLogger(__name__)


class ImportValidationResult(BaseModel):
    """Model for import validation results.

    This model captures the comprehensive results of import validation
    including categorized imports, validation status, and recommendations.
    """

    valid_imports: list[dict[str, Any]]
    invalid_imports: list[dict[str, Any]]
    missing_dependencies: list[str]
    stdlib_imports: list[str]
    third_party_imports: list[str]
    local_imports: list[str]
    suggestions: list[str]


class ImportValidationService:
    """Service for comprehensive import and dependency validation.

    This service provides advanced import validation capabilities including:
    - Standard library import validation
    - Third-party package verification
    - Local dependency resolution
    - Missing dependency detection
    - Import categorization and analysis

    The service integrates with dependency parsers to provide detailed
    insights into import structure and dependency management.
    """

    def __init__(self, project_root: str | None = None):
        """Initialize the ImportValidationService service.

        This initializes the ImportValidationService with required dependencies
        and configuration for service operations.

        Args:
            project_root (Optional[str]): Project root
        """
        self.stdlib_modules = self._get_stdlib_modules()
        self.project_root = project_root or os.getcwd()
        self.dependency_parser = DependencyParser(self.project_root)

    async def validate_imports(
        self, code_graph_id: str, imports: list[dict[str, Any]]
    ) -> ImportValidationResult:
        """
        Validate imports against the codebase graph and standard libraries.

        Args:
            code_graph_id: ID of the code graph
            imports: List of import information from AST analysis
            graph_manager: Optional graph manager instance

        Returns:
            ImportValidationResult with validation details
        """
        valid_imports: list[Any] = []
        invalid_imports: list[Any] = []
        missing_dependencies: list[Any] = []
        stdlib_imports: list[Any] = []
        third_party_imports: list[Any] = []
        local_imports: list[Any] = []
        suggestions: list[Any] = []
        for import_info in imports:
            try:
                module_name = import_info.get("module", "")
                import_info.get("type", "")
                if not module_name:
                    continue
                if self.is_stdlib_module(module_name):
                    stdlib_imports.append(module_name)
                    valid_imports.append(import_info)
                    continue
                is_in_graph = await self._check_import_in_graph(code_graph_id, module_name)
                if is_in_graph:
                    local_imports.append(module_name)
                    valid_imports.append(import_info)
                    continue
                available_packages = await self.dependency_parser.get_available_packages()
                if await self._is_third_party_available(module_name, available_packages):
                    third_party_imports.append(module_name)
                    valid_imports.append(import_info)
                elif self._is_likely_third_party(module_name):
                    invalid_imports.append(import_info)
                    missing_dependencies.append(module_name)
                    package_info = self.dependency_parser.get_package_info(module_name)
                    if package_info:
                        suggestions.append(
                            f"Module '{module_name}' found in dependencies but may not be installed."
                        )
                    else:
                        similar_packages = self.dependency_parser.get_similar_packages(module_name)
                        if similar_packages:
                            suggestions.append(
                                f"Module '{module_name}' not found. Did you mean: {', '.join(similar_packages)}?"
                            )
                        else:
                            suggestions.append(
                                f"Module '{module_name}' not found. Add it to dependencies or check spelling."
                            )
                else:
                    invalid_imports.append(import_info)
                    missing_dependencies.append(module_name)
                    similar_packages = self.dependency_parser.get_similar_packages(module_name)
                    if similar_packages:
                        suggestions.append(
                            f"Module '{module_name}' not found. Did you mean: {', '.join(similar_packages)}?"
                        )
                    else:
                        suggestions.append(
                            f"Module '{module_name}' not found. Consider adding it to dependencies."
                        )
            except Exception as e:
                logger.warning(f"Error validating import {import_info}: {e}")
                invalid_imports.append(import_info)
        return ImportValidationResult(
            valid_imports=valid_imports,
            invalid_imports=invalid_imports,
            missing_dependencies=missing_dependencies,
            stdlib_imports=stdlib_imports,
            third_party_imports=third_party_imports,
            local_imports=local_imports,
            suggestions=suggestions,
        )

    async def _check_import_in_graph(self, code_graph_id: str, module_name: str) -> bool:
        """Check if an import exists in the code graph using direct database access."""
        try:
            async with get_raw_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM nodes
                    WHERE code_graph_id = %s
                    AND type IN ('module', 'file')
                    AND properties->>'name' = %s
                    """,
                    (code_graph_id, module_name),
                )
                result = await cursor.fetchone()
                count = int(result[0]) if result else 0
                return count > 0
        except psycopg.Error:
            return False
        except Exception as e:
            logger.error(f"Error checking import in graph: {e}")
            return False

    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the Python standard library."""
        if module_name.startswith("."):
            return False
        top_level = module_name.split(".")[0]
        return top_level in self.stdlib_modules

    async def _is_third_party_available(
        self, module_name: str, available_packages: dict[str, Any]
    ) -> bool:
        """Check if a third-party module is available in project dependencies."""
        top_level = module_name.split(".")[0]
        if top_level.lower() in available_packages:
            return True
        import_to_package_map = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "bs4": "beautifulsoup4",
            "jwt": "pyjwt",
            "dotenv": "python-dotenv",
            "jose": "python-jose",
            "multipart": "python-multipart",
            "redis": "redis",
            "pytest": "pytest",
        }
        mapped_package = import_to_package_map.get(top_level.lower())
        if mapped_package and mapped_package.lower() in available_packages:
            return True
        for package_name in available_packages.keys():
            if package_name.startswith(top_level.lower()) or top_level.lower().startswith(
                package_name
            ):
                return True
        return False

    def _is_likely_third_party(self, module_name: str) -> bool:
        """Heuristic to determine if a module is likely third-party."""
        common_third_party = {
            "requests",
            "numpy",
            "pandas",
            "flask",
            "django",
            "fastapi",
            "sqlalchemy",
            "pydantic",
            "click",
            "pytest",
            "matplotlib",
            "scipy",
            "sklearn",
            "tensorflow",
            "torch",
            "redis",
            "celery",
            "aiohttp",
            "asyncio",
            "uvloop",
            "gunicorn",
            "uvicorn",
            "pyastgrep",
            "uvicorn",
            "sqlite_utils",
            "python_multipart",
            "python_jose",
            "passlib",
            "openpyxl",
            "timm",
            "prometheus_client",
            "starlette_exporter",
            "networkx",
            "astpretty",
            "astor",
            "jinja2",
            "directory_tree",
            "docker",
            "schemathesis",
            "github",
            "aiofiles",
            "psutil",
        }
        top_level = module_name.split(".")[0]
        return top_level in common_third_party

    def _get_stdlib_modules(self) -> set[Any]:
        """Get set of standard library module names."""
        stdlib_modules = {
            "__future__",
            "_thread",
            "abc",
            "aifc",
            "argparse",
            "array",
            "ast",
            "asynchat",
            "asyncio",
            "asyncore",
            "atexit",
            "audioop",
            "base64",
            "bdb",
            "binascii",
            "binhex",
            "bisect",
            "builtins",
            "bz2",
            "calendar",
            "cgi",
            "cgitb",
            "chunk",
            "cmath",
            "cmd",
            "code",
            "codecs",
            "codeop",
            "collections",
            "colorsys",
            "compileall",
            "concurrent",
            "configparser",
            "contextlib",
            "copy",
            "copyreg",
            "cProfile",
            "crypt",
            "csv",
            "ctypes",
            "curses",
            "dataclasses",
            "datetime",
            "dbm",
            "decimal",
            "difflib",
            "dis",
            "distutils",
            "doctest",
            "email",
            "encodings",
            "ensurepip",
            "enum",
            "errno",
            "faulthandler",
            "fcntl",
            "filecmp",
            "fileinput",
            "fnmatch",
            "formatter",
            "fractions",
            "ftplib",
            "functools",
            "gc",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "grp",
            "gzip",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "http",
            "imaplib",
            "imghdr",
            "imp",
            "importlib",
            "inspect",
            "io",
            "ipaddress",
            "itertools",
            "json",
            "keyword",
            "lib2to3",
            "linecache",
            "locale",
            "logging",
            "lzma",
            "mailbox",
            "mailcap",
            "marshal",
            "math",
            "mimetypes",
            "mmap",
            "modulefinder",
            "msilib",
            "msvcrt",
            "multiprocessing",
            "netrc",
            "nis",
            "nntplib",
            "numbers",
            "operator",
            "optparse",
            "os",
            "ossaudiodev",
            "parser",
            "pathlib",
            "pdb",
            "pickle",
            "pickletools",
            "pipes",
            "pkgutil",
            "platform",
            "plistlib",
            "poplib",
            "posix",
            "pprint",
            "profile",
            "pstats",
            "pty",
            "pwd",
            "py_compile",
            "pyclbr",
            "pydoc",
            "queue",
            "quopri",
            "random",
            "re",
            "readline",
            "reprlib",
            "resource",
            "rlcompleter",
            "runpy",
            "sched",
            "secrets",
            "select",
            "selectors",
            "shelve",
            "shlex",
            "shutil",
            "signal",
            "site",
            "smtpd",
            "smtplib",
            "sndhdr",
            "socket",
            "socketserver",
            "spwd",
            "sqlite3",
            "ssl",
            "stat",
            "statistics",
            "string",
            "stringprep",
            "struct",
            "subprocess",
            "sunau",
            "symbol",
            "symtable",
            "sys",
            "sysconfig",
            "syslog",
            "tabnanny",
            "tarfile",
            "telnetlib",
            "tempfile",
            "termios",
            "test",
            "textwrap",
            "threading",
            "time",
            "timeit",
            "tkinter",
            "token",
            "tokenize",
            "trace",
            "traceback",
            "tracemalloc",
            "tty",
            "turtle",
            "turtledemo",
            "types",
            "typing",
            "unicodedata",
            "unittest",
            "urllib",
            "uu",
            "uuid",
            "venv",
            "warnings",
            "wave",
            "weakre",
            "webbrowser",
            "winreg",
            "winsound",
            "wsgire",
            "xdrlib",
            "xml",
            "xmlrpc",
            "zipapp",
            "zipfile",
            "zipimport",
            "zlib",
            "zoneinfo",
        }
        return stdlib_modules
