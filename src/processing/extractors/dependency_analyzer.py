"""Module for analyzing code dependencies."""

from __future__ import annotations

import ast
import logging
from typing import Any

from models.core_models import DependencyInfoModel

from ..utils.ast_utils import ASTUtils

logger = logging.getLogger(__name__)


class CodeDependencyExtractor:
    """Analyzes dependencies and imports in Python code

    Attributes:
        ast_utils (ASTUtils): Instance attribute ast_utils.
        stdlib_modules (Set): Instance attribute stdlib_modules.
        common_third_party (Set): Instance attribute common_third_party."""

    def __init__(self) -> None:
        self.ast_utils = ASTUtils()

        # Standard library modules (Python 3.8+)
        self.stdlib_modules = {
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
            "multiprocessing",
            "netrc",
            "nntplib",
            "numbers",
            "operator",
            "optparse",
            "os",
            "ossaudiodev",
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
        }

        # Common third-party packages
        self.common_third_party = {
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "sklearn",
            "tensorflow",
            "torch",
            "requests",
            "flask",
            "django",
            "fastapi",
            "sqlalchemy",
            "pytest",
            "click",
            "pydantic",
            "redis",
            "celery",
            "gunicorn",
            "uvicorn",
            "aiohttp",
            "httpx",
            "pillow",
            "opencv",
            "beautifulsoup4",
            "lxml",
            "jinja2",
            "boto3",
            "psycopg2",
        }

    async def analyze_imports(self, tree: ast.AST, file_path: str) -> list[DependencyInfoModel]:
        """Analyze all imports in the AST

        Args:
            tree (ast.AST): The tree parameter.
            file_path (str): The file_path parameter.

        Returns:
            List[DependencyInfoModel]: The list[dependencyinfomodel] result."""
        dependencies: list[Any] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend(self._analyze_import_statement(node, file_path))
            elif isinstance(node, ast.ImportFrom):
                dependencies.extend(self._analyze_import_from_statement(node, file_path))

        return dependencies

    def _analyze_import_statement(
        self, node: ast.Import, file_path: str
    ) -> list[DependencyInfoModel]:
        """Analyze 'import module' statements"""
        dependencies: list[Any] = []
        for alias in node.names:
            module_name = alias.name
            dependency_type = self._classify_dependency(module_name, file_path)

            dependencies.append(
                DependencyInfoModel(
                    module_name=module_name,
                    import_type="import",
                    alias=alias.asname,
                    line_number=node.lineno,
                    is_external=dependency_type != "local",
                    metadata={
                        "dependency_type": dependency_type,
                        "root_module": module_name.split(".")[0],
                        "is_aliased": alias.asname is not None,
                        "file_path": file_path,
                    },
                )
            )

        return dependencies

    def _analyze_import_from_statement(
        self, node: ast.ImportFrom, file_path: str
    ) -> list[DependencyInfoModel]:
        """Analyze 'from module import item' statements"""
        dependencies: list[Any] = []
        module_name = node.module or ""
        dependency_type = self._classify_dependency(module_name)

        for alias in node.names:
            # Handle star imports
            if alias.name == "*":
                dependencies.append(
                    DependencyInfoModel(
                        module_name=module_name,
                        import_type="from_import_star",
                        alias=None,
                        line_number=node.lineno,
                        is_external=dependency_type != "local",
                        metadata={
                            "dependency_type": dependency_type,
                            "root_module": (module_name.split(".")[0] if module_name else ""),
                            "is_star_import": True,
                            "relative_level": node.level,
                            "file_path": file_path,
                            "warning": "Star imports can pollute namespace",
                        },
                    )
                )
            else:
                full_name = f"{module_name}.{alias.name}" if module_name else alias.name

                dependencies.append(
                    DependencyInfoModel(
                        module_name=module_name,
                        import_type="from_import",
                        alias=alias.asname,
                        line_number=node.lineno,
                        is_external=dependency_type != "local",
                        metadata={
                            "dependency_type": dependency_type,
                            "root_module": (module_name.split(".")[0] if module_name else ""),
                            "imported_name": alias.name,
                            "full_name": full_name,
                            "is_aliased": alias.asname is not None,
                            "relative_level": node.level,
                            "file_path": file_path,
                        },
                    )
                )

        return dependencies

    def _classify_dependency(self, module_name: str, file_path: str = "") -> str:
        """Classify a dependency as stdlib, third-party, or local"""
        if not module_name:
            return "local"

        # Handle relative imports
        if module_name.startswith("."):
            return "local"

        root_module = module_name.split(".")[0]

        # Check if it's a standard library module
        if root_module in self.stdlib_modules:
            return "stdlib"

        # Check if it's a known third-party package
        if root_module in self.common_third_party:
            return "third_party"

        # Check for common patterns
        if any(root_module.startswith(prefix) for prefix in ["_", "__"]):
            return "stdlib"  # Private/dunder modules are usually stdlib

        # Check if module name matches local project structure
        # This is a heuristic - modules without dots that aren't recognized
        # as stdlib or third-party are likely local modules
        if "." not in module_name and len(module_name) < 20:
            # Short module names without dots are often local
            return "local"

        # If we can't determine, assume it's third-party
        return "third_party"

    async def analyze_dependency_graph(
        self, dependencies: list[DependencyInfoModel], tree: ast.AST | None = None
    ) -> dict[str, Any]:
        """Analyze the dependency graph and extract insights

        Args:
            dependencies (List[DependencyInfoModel]): The dependencies parameter.
            tree (Optional[ast.AST], optional): The tree parameter. Defaults to None.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await analyze_dependency_graph("dependencies_value", "tree_value")"""
        analysis: dict[str, Any] = {
            "total_dependencies": len(dependencies),
            "by_type": {},
            "by_category": {},
            "external_dependencies": [],
            "local_dependencies": [],
            "potential_issues": [],
            "unused_imports": [],
            "circular_dependencies": [],
        }

        # Group by import type
        for dep in dependencies:
            import_type = dep.import_type
            if import_type not in analysis["by_type"]:
                analysis["by_type"][import_type] = 0
            analysis["by_type"][import_type] += 1

        # Group by dependency category
        for dep in dependencies:
            dep_type = dep.metadata.get("dependency_type", "unknown")
            if dep_type not in analysis["by_category"]:
                analysis["by_category"][dep_type] = 0
            analysis["by_category"][dep_type] += 1

            if dep.is_external:
                analysis["external_dependencies"].append(
                    {
                        "module": dep.module_name,
                        "type": dep_type,
                        "line": dep.line_number,
                    }
                )
            else:
                analysis["local_dependencies"].append(
                    {"module": dep.module_name, "line": dep.line_number}
                )

        # Detect potential issues
        analysis["potential_issues"] = self._detect_dependency_issues(dependencies)

        # Detect unused imports if AST is provided
        if tree is not None:
            unused_imports = await self.detect_unused_imports(tree, dependencies)
            analysis["unused_imports"] = unused_imports
            analysis["potential_issues"].extend(unused_imports)

        return analysis

    def _detect_dependency_issues(
        self, dependencies: list[DependencyInfoModel]
    ) -> list[dict[str, Any]]:
        """Detect potential issues in dependencies"""
        issues: list[Any] = []
        # Check for star imports
        star_imports = [dep for dep in dependencies if dep.import_type == "from_import_star"]
        for dep in star_imports:
            issues.append(
                {
                    "type": "star_import",
                    "severity": "medium",
                    "message": f"Star import from '{dep.module_name}' can pollute namespace",
                    "line": dep.line_number,
                    "suggestion": "Import specific items instead of using '*'",
                }
            )

        # Check for deprecated modules
        deprecated_modules = {"imp", "formatter", "asynchat", "asyncore"}
        for dep in dependencies:
            root_module = dep.module_name.split(".")[0] if dep.module_name else ""
            if root_module in deprecated_modules:
                issues.append(
                    {
                        "type": "deprecated_module",
                        "severity": "high",
                        "message": f"Module '{root_module}' is deprecated",
                        "line": dep.line_number,
                        "suggestion": "Consider using alternative modules",
                    }
                )

        # Check for potential security issues
        security_sensitive = {"eval", "exec", "subprocess", "os"}
        for dep in dependencies:
            if dep.metadata.get("imported_name") in security_sensitive:
                issues.append(
                    {
                        "type": "security_sensitive",
                        "severity": "medium",
                        "message": f"Import of '{dep.metadata.get('imported_name')}' requires careful security review",
                        "line": dep.line_number,
                        "suggestion": "Ensure proper input validation and sanitization",
                    }
                )

        return issues

    async def detect_unused_imports(
        self, tree: ast.AST, dependencies: list[DependencyInfoModel]
    ) -> list[dict[str, Any]]:
        """Detect unused imports by tracking symbol usage in the AST.

        Args:
            tree (ast.AST): The tree parameter.
            dependencies (List[DependencyInfoModel]): The dependencies parameter.

        Returns:
            List[Dict[str, Any]]:     List of issues for unused imports.

        Example:
            >>> await detect_unused_imports("tree_value", "dependencies_value")"""
        # Track all imported symbols
        imported_symbols: dict[str, DependencyInfoModel] = {}

        for dep in dependencies:
            if dep.import_type == "import":
                # For 'import module' statements
                symbol = dep.alias or dep.module_name
                imported_symbols[symbol] = dep
            elif dep.import_type == "from_import":
                # For 'from module import item' statements
                imported_name = dep.metadata.get("imported_name", "")
                symbol = dep.alias or imported_name
                if symbol:
                    imported_symbols[symbol] = dep
            elif dep.import_type == "from_import_star":
                # Star imports can't be reliably tracked
                logger.debug(
                    "Skipping star import from %s for unused import detection",
                    dep.module_name,
                )

        # Track symbol usage
        used_symbols = self._find_used_symbols(tree)

        # Find unused imports
        unused_issues: list[dict[str, Any]] = []
        for symbol, dep in imported_symbols.items():
            if symbol not in used_symbols:
                unused_issues.append(
                    {
                        "type": "unused_import",
                        "severity": "low",
                        "message": f"Import '{symbol}' is unused",
                        "line": dep.line_number,
                        "suggestion": f"Remove unused import '{symbol}'",
                        "module": dep.module_name,
                        "import_type": dep.import_type,
                    }
                )

        return unused_issues

    def _find_used_symbols(self, tree: ast.AST) -> set[str]:
        """Find all symbols used in the AST.

        Args:
            tree: AST to analyze

        Returns:
            Set of symbol names that are used in the code
        """
        used_symbols: set[str] = set()

        class SymbolUsageFinder(ast.NodeVisitor):
            """SymbolUsageFinder class."""

            def visit_Name(self, node: ast.Name) -> None:
                """Function visit_Name.

                Args:
                    node (ast.Name): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_Name.

                Args:
                    node (ast.Name): The node parameter.

                Returns:
                    None: The none result."""
                # Track name usage (excluding function/class definitions)
                if isinstance(node.ctx, (ast.Load, ast.Del)):
                    used_symbols.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                """Function visit_Attribute.

                Args:
                    node (ast.Attribute): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_Attribute.

                Args:
                    node (ast.Attribute): The node parameter.

                Returns:
                    None: The none result."""
                # Track attribute access (e.g., module.function)
                if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_symbols.add(node.value.id)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Function visit_FunctionDef.

                Args:
                    node (ast.FunctionDef): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_FunctionDef.

                Args:
                    node (ast.FunctionDef): The node parameter.

                Returns:
                    None: The none result."""
                # Skip function name itself but visit body
                for stmt in node.body:
                    self.visit(stmt)
                for decorator in node.decorator_list:
                    self.visit(decorator)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                """Function visit_AsyncFunctionDef.

                Args:
                    node (ast.AsyncFunctionDef): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_AsyncFunctionDef.

                Args:
                    node (ast.AsyncFunctionDef): The node parameter.

                Returns:
                    None: The none result."""
                # Skip function name itself but visit body
                for stmt in node.body:
                    self.visit(stmt)
                for decorator in node.decorator_list:
                    self.visit(decorator)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                """Function visit_ClassDef.

                Args:
                    node (ast.ClassDef): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_ClassDef.

                Args:
                    node (ast.ClassDef): The node parameter.

                Returns:
                    None: The none result."""
                # Visit base classes and body
                for base in node.bases:
                    self.visit(base)
                for stmt in node.body:
                    self.visit(stmt)
                for decorator in node.decorator_list:
                    self.visit(decorator)

            def visit_Import(self, node: ast.Import) -> None:
                """Function visit_Import.

                Args:
                    node (ast.Import): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_Import.

                Args:
                    node (ast.Import): The node parameter.

                Returns:
                    None: The none result."""
                # Skip import statements themselves
                pass

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                """Function visit_ImportFrom.

                Args:
                    node (ast.ImportFrom): The node parameter.

                Returns:
                    None: The none result."""
                """Method visit_ImportFrom.

                Args:
                    node (ast.ImportFrom): The node parameter.

                Returns:
                    None: The none result."""
                # Skip import statements themselves
                pass

        finder = SymbolUsageFinder()
        finder.visit(tree)

        return used_symbols

    async def detect_circular_dependencies(
        self, module_path: str, project_dependencies: dict[str, list[str]]
    ) -> list[dict[str, Any]]:
        """Detect circular dependencies in the project.

        Args:
            module_path (str): The module_path parameter.
            project_dependencies (Dict[str, List[str]]): The project_dependencies parameter.

        Returns:
            List[Dict[str, Any]]:     List of circular dependency issues.

        Example:
            >>> await detect_circular_dependencies("module_path_value", "project_dependencies_value")
        """
        circular_issues: list[dict[str, Any]] = []
        visited: set[str] = set()

        def find_cycles(module: str, path: list[str]) -> list[str] | None:
            """DFS to find circular dependencies.

            Args:
                module (str): The module parameter.
                path (List[str]): The path parameter.

            Returns:
                Optional[List[str]]: The optional[list[str]] result.

            Example:
                >>> find_cycles("module_value", "path_value")"""
            if module in path:
                # Found a cycle
                cycle_start = path.index(module)
                return path[cycle_start:] + [module]

            if module in visited:
                return None

            visited.add(module)
            path.append(module)

            # Check dependencies of this module
            if module in project_dependencies:
                for dep in project_dependencies[module]:
                    if cycle := find_cycles(dep, path.copy()):
                        return cycle

            path.pop()
            return None

        # Start DFS from the current module
        if cycle := find_cycles(module_path, []):
            circular_issues.append(
                {
                    "type": "circular_dependency",
                    "severity": "high",
                    "message": f"Circular dependency detected: {' -> '.join(cycle)}",
                    "modules": cycle,
                    "suggestion": "Refactor to break the circular dependency",
                }
            )

        # Also check for circular dependencies starting from other modules
        # that might involve this module
        for module in project_dependencies:
            if module not in visited:
                if cycle := find_cycles(module, []):
                    if module_path in cycle:
                        circular_issues.append(
                            {
                                "type": "circular_dependency",
                                "severity": "high",
                                "message": f"Circular dependency detected: {' -> '.join(cycle)}",
                                "modules": cycle,
                                "suggestion": "Refactor to break the circular dependency",
                            }
                        )

        # Remove duplicates
        unique_issues = []
        seen_cycles = set()

        for issue in circular_issues:
            cycle_key = frozenset(issue["modules"])
            if cycle_key not in seen_cycles:
                seen_cycles.add(cycle_key)
                unique_issues.append(issue)

        return unique_issues

    async def build_project_dependency_graph(self, project_root: str) -> dict[str, list[str]]:
        """Build a dependency graph for the entire project.

        Args:
            project_root (str): The project_root parameter.

        Returns:
            Dict[str, List[str]]:     Dictionary mapping module paths to their dependencies.

        Example:
            >>> await build_project_dependency_graph("project_root_value")"""
        import os
        from pathlib import Path

        project_dependencies: dict[str, list[str]] = {}

        # Walk through all Python files in the project
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_root)

                    try:
                        with open(file_path, encoding="utf-8") as f:
                            code = f.read()

                        tree = ast.parse(code)
                        dependencies = await self.analyze_imports(tree, relative_path)

                        # Convert dependencies to module paths
                        module_deps = []
                        for dep in dependencies:
                            # Skip external dependencies
                            if dep.metadata.get("dependency_type") != "local":
                                continue

                            # Convert module name to file path
                            module_name = dep.module_name
                            if module_name:
                                # Handle relative imports
                                if dep.metadata.get("relative_level", 0) > 0:
                                    # Calculate relative import path
                                    current_dir = Path(relative_path).parent
                                    for _ in range(dep.metadata.get("relative_level", 0) - 1):
                                        current_dir = current_dir.parent

                                    if module_name:
                                        module_path = current_dir / module_name.replace(".", "/")
                                    else:
                                        module_path = current_dir
                                else:
                                    # For absolute imports in the same directory
                                    # Check if it's a local module
                                    module_path = Path(module_name.replace(".", "/"))

                                    # Also check in the same directory as the current file
                                    same_dir_path = Path(relative_path).parent / module_name

                                    # Try different file extensions and locations
                                    for base_path in [module_path, same_dir_path]:
                                        for ext in [".py", "/__init__.py"]:
                                            potential_path = str(base_path) + ext
                                            full_path = os.path.join(project_root, potential_path)
                                            if os.path.exists(full_path):
                                                module_deps.append(potential_path)
                                                break
                                        else:
                                            continue
                                        break

                        project_dependencies[relative_path] = module_deps

                    except (SyntaxError, UnicodeDecodeError) as e:
                        logger.warning("Error parsing %s: %s", file_path, str(e))

        return project_dependencies

    async def get_dependency_summary(
        self, dependencies: list[DependencyInfoModel]
    ) -> dict[str, Any]:
        """Get a summary of dependencies for reporting

        Args:
            dependencies (List[DependencyInfoModel]): The dependencies parameter.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await get_dependency_summary("dependencies_value")"""
        stdlib_deps = [
            dep for dep in dependencies if dep.metadata.get("dependency_type") == "stdlib"
        ]
        third_party_deps = [
            dep for dep in dependencies if dep.metadata.get("dependency_type") == "third_party"
        ]
        local_deps = [dep for dep in dependencies if dep.metadata.get("dependency_type") == "local"]

        return {
            "total_count": len(dependencies),
            "stdlib_count": len(stdlib_deps),
            "third_party_count": len(third_party_deps),
            "local_count": len(local_deps),
            "unique_modules": len(set(dep.module_name for dep in dependencies if dep.module_name)),
            "has_star_imports": any(dep.import_type == "from_import_star" for dep in dependencies),
            "external_modules": list(
                set(dep.module_name for dep in dependencies if dep.is_external and dep.module_name)
            ),
            "local_modules": list(
                set(
                    dep.module_name
                    for dep in dependencies
                    if not dep.is_external and dep.module_name
                )
            ),
        }
