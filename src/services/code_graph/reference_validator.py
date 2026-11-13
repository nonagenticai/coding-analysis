"""Reference validation service for code graph analysis.

This module provides tools to validate code references and prevent hallucinations
in AI-generated responses by verifying that references to files, functions, classes,
and variables actually exist in the codebase.
"""

from __future__ import annotations

import ast
import logging
import os
import signal
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.package_discovery_service import PackageDiscoveryService
logger = logging.getLogger(__name__)


class ValidationTimeoutError(Exception):
    """Custom timeout exception for validation operations."""


@contextmanager
def timeout(seconds: int) -> Generator[None]:
    """Context manager for timeouts using signal alarm.

    Args:
        seconds (int): The seconds parameter.

    Returns:
        None: The none result.

    Raises:
        ValidationTimeoutError: Raised under certain conditions."""

    def timeout_handler(signum, _frame) -> None:
        """Function timeout_handler.

        Args:
            signum: The signum parameter.
            _frame: The _frame parameter.

        Returns:
            None: The none result.

        Raises:
            ValidationTimeoutError: Raised under certain conditions."""
        raise ValidationTimeoutError(f"Operation timed out after {seconds} seconds")

    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


class ReferenceValidator:
    """Validates code references to prevent hallucinations in AI-generated responses.

    This class verifies that references to files, functions, classes, and variables
    actually exist in the codebase, preventing hallucinations where AI might invent
    non-existent code elements.

    Attributes:
        codebase_path: Instance attribute codebase_path."""

    def __init__(self, codebase_path: str):
        """Initialize with the path to the codebase root."""
        self.codebase_path = codebase_path
        self.file_index: dict[str, Any] = {}
        self.entity_index: dict[str, dict[str, Any]] = {
            "functions": {},
            "classes": {},
            "methods": {},
            "variables": {},
            "imports": {},
        }
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build indexes of files and code entities for quick validation."""
        for root, _dirs, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.codebase_path)
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            code = f.read()
                        self.file_index[rel_path] = {
                            "content": code,
                            "absolute_path": file_path,
                        }
                        tree = ast.parse(code)
                        self._index_entities(tree, rel_path)
                    except (SyntaxError, UnicodeDecodeError) as e:
                        logger.warning("Error parsing %s: %s", file_path, str(e))

    def _index_entities(self, tree: ast.Module, file_path: str) -> None:
        """Index all code entities from an AST."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and node in parent.body:
                        is_method = True
                        method_key = f"{parent.name}.{node.name}"
                        self.entity_index["methods"][method_key] = {
                            "file": file_path,
                            "line": node.lineno,
                            "class": parent.name,
                        }
                        break
                if not is_method:
                    self.entity_index["functions"][node.name] = {
                        "file": file_path,
                        "line": node.lineno,
                    }
            elif isinstance(node, ast.ClassDef):
                self.entity_index["classes"][node.name] = {
                    "file": file_path,
                    "line": node.lineno,
                    "methods": [
                        m.name
                        for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
            elif isinstance(node, ast.Import):
                for name in node.names:
                    self.entity_index["imports"][name.name] = {
                        "file": file_path,
                        "line": node.lineno,
                        "alias": name.asname,
                    }
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        import_name = f"{node.module}.{name.name}"
                        self.entity_index["imports"][import_name] = {
                            "file": file_path,
                            "line": node.lineno,
                            "alias": name.asname,
                        }

    def validate_file_reference(self, file_path: str) -> tuple[bool, str | None]:
        """Validate that a file reference exists in the codebase.

        Args:
            file_path (str): The file_path parameter.

        Returns:
            Tuple[bool, Optional[str]]:     (exists, correct_path)
            - exists: Boolean indicating if file exists
            - correct_path: The correct path if found, or suggested alternatives.

        Example:
            >>> validate_file_reference("file_path_value")"""
        if os.path.isabs(file_path):
            if os.path.isfile(file_path):
                rel_path = os.path.relpath(file_path, self.codebase_path)
                return (True, rel_path)
            return (False, None)
        if file_path in self.file_index:
            return (True, file_path)
        similar_paths: list[Any] = []
        for indexed_path in self.file_index:
            if os.path.basename(file_path) == os.path.basename(indexed_path):
                similar_paths.append(indexed_path)
            elif file_path in indexed_path:
                similar_paths.append(indexed_path)
        if similar_paths:
            return (
                False,
                f"Not found, did you mean one of these? {', '.join(similar_paths)}",
            )
        return (False, None)

    def validate_function_reference(self, function_name: str) -> tuple[bool, dict | None]:
        """Validate that a function reference exists in the codebase.

        Args:
            function_name (str): The function_name parameter.

        Returns:
            Tuple[bool, Optional[Dict]]:     (exists, details)
            - exists: Boolean indicating if function exists
            - details: Details about the function if found.

        Example:
            >>> validate_function_reference("example")"""
        if function_name in self.entity_index["functions"]:
            return (True, self.entity_index["functions"][function_name])
        if "." in function_name and function_name in self.entity_index["methods"]:
            return (True, self.entity_index["methods"][function_name])
        similar_functions: list[Any] = []
        for indexed_function in self.entity_index["functions"]:
            if function_name.lower() in indexed_function.lower():
                similar_functions.append(indexed_function)
        if similar_functions:
            return (False, {"similar_functions": similar_functions})
        return (False, None)

    def validate_class_reference(self, class_name: str) -> tuple[bool, dict | None]:
        """Validate that a class reference exists in the codebase.

        Args:
            class_name (str): The class_name parameter.

        Returns:
            Tuple[bool, Optional[Dict]]:     (exists, details)
            - exists: Boolean indicating if class exists
            - details: Details about the class if found.

        Example:
            >>> validate_class_reference("example")"""
        if class_name in self.entity_index["classes"]:
            return (True, self.entity_index["classes"][class_name])
        similar_classes: list[Any] = []
        for indexed_class in self.entity_index["classes"]:
            if class_name.lower() in indexed_class.lower():
                similar_classes.append(indexed_class)
        if similar_classes:
            return (False, {"similar_classes": similar_classes})
        return (False, None)

    def _create_error_response(
        self, error_type: str, error_name: str, error_message: str, suggestion: str
    ) -> dict[str, Any]:
        """Create a standardized error response."""
        return {
            "valid": False,
            "issues": [
                {
                    "type": error_type,
                    "name": error_name,
                    "valid": False,
                    "suggestion": error_message,
                }
            ],
            "overall_score": 0.0,
            "suggestion": suggestion,
        }

    def _validate_import_node(self, node: ast.Import, result: dict[str, Any]) -> tuple[int, int]:
        """Validate import nodes and return (total_checked, issues_found)."""
        total_checked = 0
        issues_found = 0
        for name in node.names:
            total_checked += 1
            exists = name.name in self.entity_index["imports"]
            if not exists:
                issues_found += 1
                result["valid"] = False
                result["issues"].append(
                    {
                        "type": "import",
                        "name": name.name,
                        "valid": False,
                        "suggestion": self._suggest_similar_import(name.name),
                    }
                )
        return (total_checked, issues_found)

    def _validate_import_from_node(
        self, node: ast.ImportFrom, result: dict[str, Any]
    ) -> tuple[int, int]:
        """Validate ImportFrom nodes and return (total_checked, issues_found)."""
        total_checked = 0
        issues_found = 0
        if node.module:
            for name in node.names:
                total_checked += 1
                import_name = f"{node.module}.{name.name}"
                exists = import_name in self.entity_index["imports"]
                if not exists:
                    issues_found += 1
                    result["valid"] = False
                    result["issues"].append(
                        {
                            "type": "import",
                            "name": import_name,
                            "valid": False,
                            "suggestion": self._suggest_similar_import(import_name),
                        }
                    )
        return (total_checked, issues_found)

    def _validate_call_node(self, node: ast.Call, result: dict[str, Any]) -> tuple[int, int]:
        """Validate Call nodes and return (total_checked, issues_found)."""
        total_checked = 0
        issues_found = 0
        if hasattr(node.func, "id"):
            total_checked += 1
            func_name = str(node.func.id)
            exists, _ = self.validate_function_reference(func_name)
            if not exists:
                issues_found += 1
                result["valid"] = False
                result["issues"].append(
                    {
                        "type": "function",
                        "name": func_name,
                        "valid": False,
                        "suggestion": self._suggest_similar_function(func_name),
                    }
                )
        elif hasattr(node.func, "attr") and hasattr(node.func, "value"):
            if hasattr(node.func.value, "id"):
                total_checked += 1
                method_name = f"{node.func.value.id}.{node.func.attr}"
                exists, _ = self.validate_function_reference(method_name)
                if not exists:
                    issues_found += 1
                    result["valid"] = False
                    result["issues"].append(
                        {
                            "type": "method",
                            "name": method_name,
                            "valid": False,
                            "suggestion": self._suggest_similar_method(method_name),
                        }
                    )
        return (total_checked, issues_found)

    def validate_code_snippet(self, code_snippet: str) -> dict[str, Any]:
        """Validate that a code snippet references real entities in the codebase.

        Analyzes imports, function calls, class references, etc. to verify they exist.

        Args:
            code_snippet (str): The code_snippet parameter.

        Returns:
            Dict[str, Any]:     {
                "valid": bool,
                "issues": [
                    {
                        "type": "import|function|class|method",
                        "name": "entity_name",
                        "valid": bool,
                        "suggestion": "Suggested alternative if any"
                    }
                ],
                "overall_score": float,
                "suggestion": str|None
            }.

        Example:
            >>> validate_code_snippet("code_snippet_value")"""
        result: dict[str, Any] = {"valid": True, "issues": []}
        try:
            if not code_snippet or not isinstance(code_snippet, str):
                return self._create_error_response(
                    "input",
                    "Code Snippet",
                    "Code snippet must be a non-empty string",
                    "Provide a valid Python code snippet",
                )
            try:
                with timeout(5):
                    tree = ast.parse(code_snippet)
            except ValidationTimeoutError:
                return self._create_error_response(
                    "timeout",
                    "Parse Timeout",
                    "Code snippet parsing timed out. The snippet may be too complex or contain malformed syntax.",
                    "Simplify the code snippet or check for syntax errors",
                )
            total_checked = 0
            issues_found = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    checked, found = self._validate_import_node(node, result)
                    total_checked += checked
                    issues_found += found
                elif isinstance(node, ast.ImportFrom):
                    checked, found = self._validate_import_from_node(node, result)
                    total_checked += checked
                    issues_found += found
                elif isinstance(node, ast.Call):
                    checked, found = self._validate_call_node(node, result)
                    total_checked += checked
                    issues_found += found
            if total_checked > 0:
                result["overall_score"] = (total_checked - issues_found) / total_checked
            else:
                result["overall_score"] = 1.0
            if result["issues"]:
                result["suggestion"] = (
                    f"Resolve {len(result['issues'])} validation issue(s) found in the code snippet"
                )
            else:
                result["suggestion"] = None
            return result
        except SyntaxError:
            return {
                "valid": False,
                "issues": [
                    {
                        "type": "syntax",
                        "name": "Syntax Error",
                        "valid": False,
                        "suggestion": "The code snippet has syntax errors and couldn't be analyzed.",
                    }
                ],
                "overall_score": 0.0,
                "suggestion": "Resolve syntax errors in the code snippet",
            }

    def _suggest_similar_import(self, import_name: str) -> str | None:
        """Suggest similar imports."""
        similar_imports: list[Any] = []
        for indexed_import in self.entity_index["imports"]:
            if import_name.lower() in indexed_import.lower():
                similar_imports.append(indexed_import)
        if similar_imports:
            return f"Did you mean one of these? {', '.join(similar_imports)}"
        return None

    def _suggest_similar_function(self, function_name: str) -> str | None:
        """Suggest similar functions."""
        similar_functions: list[Any] = []
        for indexed_function in self.entity_index["functions"]:
            if function_name.lower() in indexed_function.lower():
                similar_functions.append(indexed_function)
        if similar_functions:
            return f"Did you mean one of these? {', '.join(similar_functions)}"
        return None

    def _suggest_similar_method(self, method_name: str) -> str | None:
        """Suggest similar methods."""
        similar_methods: list[Any] = []
        for indexed_method in self.entity_index["methods"]:
            if method_name.lower() in indexed_method.lower():
                similar_methods.append(indexed_method)
        if similar_methods:
            return f"Did you mean one of these? {', '.join(similar_methods)}"
        return None


class PackageReferenceValidator:
    """Validates code references against both source code and container packages.

    Attributes:
        base_validator (ReferenceValidator): Instance attribute base_validator.
        package_discovery: Instance attribute package_discovery."""

    def __init__(
        self,
        codebase_path: str,
        package_discovery: PackageDiscoveryService | None = None,
    ):
        self.base_validator = ReferenceValidator(codebase_path)
        self.package_discovery = package_discovery
        self.container_packages: dict[str, Any] = {}

    async def initialize_for_container(self, container_name: str) -> None:
        """Initialize package information for a specific container.

        Args:
            container_name (str): The container_name parameter.

        Returns:
            None: The none result."""
        if self.package_discovery:
            self.container_packages = await self.package_discovery.discover_container_packages(
                container_name
            )

    def validate_import_statement(self, import_name: str) -> tuple[bool, dict[str, Any]]:
        """Validate import against source code, installed packages, AND standard library.

        Args:
            import_name (str): The import_name parameter.

        Returns:
            Tuple[bool, Dict[str, Any]]: The tuple[bool, dict[str, any]] result."""
        source_info_raw = self.base_validator.entity_index["imports"].get(import_name)
        source_valid = source_info_raw is not None
        source_info: dict[str, Any] = source_info_raw if source_info_raw else {}
        is_stdlib = False
        if self.package_discovery:
            is_stdlib = self.package_discovery.is_standard_library_module(import_name)
        package_valid = self._is_package_importable(import_name)
        package_info = self._get_package_info(import_name) if package_valid else {}
        is_valid = bool(source_valid) or is_stdlib or package_valid
        return (
            is_valid,
            {
                "import_name": import_name,
                "valid": is_valid,
                "found_in_source": bool(source_valid),
                "found_in_stdlib": is_stdlib,
                "found_in_packages": package_valid,
                "source_details": source_info,
                "package_details": package_info,
                "resolution": (
                    "source"
                    if source_valid
                    else (
                        "standard_library"
                        if is_stdlib
                        else "package"
                        if package_valid
                        else "not_found"
                    )
                ),
            },
        )

    def _is_package_importable(self, import_name: str) -> bool:
        """Check if import is available in container packages."""
        if not self.container_packages:
            return False
        modules_index = self.container_packages.get("modules_index", {})
        if import_name in modules_index:
            return True
        for _package_name, package_info in modules_index.items():
            modules = package_info.get("modules", [])
            if import_name in modules:
                return True
            if any(import_name.startswith(f"{module}.") for module in modules):
                return True
        return False

    def _get_package_info(self, import_name: str) -> dict[str, Any]:
        """Get detailed information about a package/module."""
        if not self.container_packages:
            return {}
        modules_index = self.container_packages.get("modules_index", {})
        if import_name in modules_index:
            return dict(modules_index[import_name])
        for package_name, package_info in modules_index.items():
            modules = package_info.get("modules", [])
            if import_name in modules or any(
                import_name.startswith(f"{module}.") for module in modules
            ):
                return {
                    "providing_package": package_name,
                    "package_version": package_info.get("version", "unknown"),
                    "package_type": package_info.get("type", "unknown"),
                    "available_modules": modules,
                }
        return {}

    async def validate_code_snippet_comprehensive(
        self, code_snippet: str, container_name: str | None = None
    ) -> dict[str, Any]:
        """Comprehensive validation of a code snippet with augmented package awareness.

        Args:
            code_snippet (str): The code_snippet parameter.
            container_name (Optional[str], optional): The container_name parameter. Defaults to None.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await validate_code_snippet_comprehensive("code_snippet_value", "example")
        """
        if container_name and self.package_discovery:
            await self.initialize_for_container(container_name)
        try:
            tree = ast.parse(code_snippet)
        except SyntaxError as e:
            return {
                "valid": False,
                "issues": [{"type": "syntax_error", "message": str(e), "valid": False}],
                "overall_score": 0.0,
            }
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    is_valid, details = self.validate_import_statement(alias.name)
                    issues.append(
                        {
                            "type": "import",
                            "name": alias.name,
                            "valid": is_valid,
                            "details": details,
                            "line": node.lineno,
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    is_valid, details = self.validate_import_statement(node.module)
                    issues.append(
                        {
                            "type": "from_import",
                            "module": node.module,
                            "valid": is_valid,
                            "details": details,
                            "line": node.lineno,
                        }
                    )
                    for alias in node.names:
                        if is_valid:
                            issues.append(
                                {
                                    "type": "from_import_item",
                                    "module": node.module,
                                    "name": alias.name,
                                    "valid": True,
                                    "details": {"inherited_from_module": True},
                                    "line": node.lineno,
                                }
                            )
        total_checks = len(issues)
        passed_checks = sum(1 for issue in issues if issue.get("valid", False))
        overall_score = passed_checks / total_checks if total_checks > 0 else 1.0
        overall_valid = overall_score >= 0.5
        return {
            "valid": overall_valid,
            "issues": issues,
            "overall_score": overall_score,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "container_packages_info": (
                {
                    "total_packages": self.container_packages.get("total_packages", 0),
                    "standard_library_modules": self.container_packages.get(
                        "standard_library_modules", 0
                    ),
                    "total_importable_modules": self.container_packages.get(
                        "total_importable_modules", 0
                    ),
                }
                if self.container_packages
                else None
            ),
        }
