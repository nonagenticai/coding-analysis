"""Codebase scanning service for automatic issue detection."""

from __future__ import annotations

import ast
import json
import logging
import warnings
from pathlib import Path
from typing import Any

from models.bug_models import ErrorType, ParsedError

logger = logging.getLogger(__name__)


class CodebaseScanner:
    """Scans codebase for potential issues and errors."""

    def __init__(self, codebase_path: str):
        """Initialize the CodebaseScanner service."""
        # DEPRECATION WARNING - only shown when actually used
        warnings.warn(
            "CodebaseScanner is deprecated and will be removed in a future version. "
            "Use container-based scanning services via GitHub provisioning instead. "
            "Container environments provide better isolation and more comprehensive analysis.",
            DeprecationWarning,
            stacklevel=2,
        )

        # This initializes the CodebaseScanner with required dependencies
        # and configuration for service operations.
        # Args: codebase_path (str): Codebase path
        self.codebase_path = Path(codebase_path)

    def scan_codebase_for_issues(self) -> list[ParsedError]:
        """Scan the entire codebase for potential issues."""
        logger.info(f"Scanning codebase at {self.codebase_path}")

        errors: list[Any] = []
        python_files = list(self.codebase_path.rglob("*.py"))

        for file_path in python_files:
            # Skip certain directories
            if any(
                skip in str(file_path)
                for skip in [
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "venv",
                    ".venv",
                ]
            ):
                continue

            try:
                file_errors = self._scan_python_file(file_path)
                errors.extend(file_errors)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to scan {file_path}: {e}")

        logger.info(f"Found {len(errors)} potential issues in codebase")
        return errors

    def _scan_python_file(self, file_path: Path) -> list[ParsedError]:
        """Scan a single Python file for issues."""
        errors: list[Any] = []
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                # Syntax error found
                error = ParsedError(
                    error_type=ErrorType.SYNTAX_ERROR,
                    message=str(e),
                    file_path=str(file_path),
                    line_number=e.lineno,
                    function_name=None,
                    stack_trace=[],
                    confidence=0.9,
                )
                errors.append(error)
                return errors  # Can't analyze further if syntax is broken

            # Analyze the AST for potential issues
            errors.extend(self._analyze_ast_for_issues(tree, file_path, content))

        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Error scanning {file_path}: {e}")

        return errors

    def _analyze_ast_for_issues(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> list[ParsedError]:
        """Analyze AST for potential issues."""
        errors: list[Any] = []
        lines = content.split("\n")

        for node in ast.walk(tree):
            # Check for potential import issues
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_errors = self._check_import_node(node, file_path, lines)
                errors.extend(import_errors)

            # Check for potential name issues
            elif isinstance(node, ast.Name):
                name_errors = self._check_name_node(node, file_path, lines)
                errors.extend(name_errors)

            # Check for potential attribute issues
            elif isinstance(node, ast.Attribute):
                attr_errors = self._check_attribute_node(node, file_path, lines)
                errors.extend(attr_errors)

            # Check for potential function call issues
            elif isinstance(node, ast.Call):
                call_errors = self._check_call_node(node, file_path, lines)
                errors.extend(call_errors)

        return errors

    def _check_import_node(
        self, node: ast.AST, file_path: Path, _lines: list[str]
    ) -> list[ParsedError]:
        """Check import nodes for potential issues."""
        errors: list[Any] = []
        if isinstance(node, ast.ImportFrom):
            if node.module:
                # Check for common problematic imports
                problematic_modules = ["os", "subprocess", "eval", "exec"]
                if any(prob in node.module for prob in problematic_modules):
                    error = ParsedError(
                        error_type=ErrorType.IMPORT_ERROR,
                        message=f"Potentially unsafe import: {node.module}",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        function_name=None,
                        stack_trace=[],
                        confidence=0.6,
                    )
                    errors.append(error)

        return errors

    def _check_name_node(
        self, node: ast.Name, file_path: Path, _lines: list[str]
    ) -> list[ParsedError]:
        """Check name nodes for potential issues."""
        errors: list[Any] = []
        # Check for common undefined variable patterns
        if isinstance(node.ctx, ast.Load):
            # This is a streamlined check - in reality, you'd need scope analysis
            if node.id in ["undefined_var", "missing_import"]:
                error = ParsedError(
                    error_type=ErrorType.NAME_ERROR,
                    message=f"Potentially undefined variable: {node.id}",
                    file_path=str(file_path),
                    line_number=node.lineno,
                    function_name=None,
                    stack_trace=[],
                    confidence=0.5,
                )
                errors.append(error)

        return errors

    def _check_attribute_node(
        self, node: ast.Attribute, file_path: Path, _lines: list[str]
    ) -> list[ParsedError]:
        """Check attribute nodes for potential issues."""
        errors: list[Any] = []
        # Check for common attribute errors
        common_issues = {
            "append": "Calling append on non-list object",
            "keys": "Calling keys on non-dict object",
            "split": "Calling split on non-string object",
        }

        if node.attr in common_issues:
            # This is a streamlined check
            error = ParsedError(
                error_type=ErrorType.ATTRIBUTE_ERROR,
                message=f"Potential attribute error: {common_issues[node.attr]}",
                file_path=str(file_path),
                line_number=node.lineno,
                function_name=None,
                stack_trace=[],
                confidence=0.4,
            )
            errors.append(error)

        return errors

    def _check_call_node(
        self, node: ast.Call, file_path: Path, _lines: list[str]
    ) -> list[ParsedError]:
        """Check function call nodes for potential issues."""
        errors: list[Any] = []
        # Check for potentially dangerous function calls
        if isinstance(node.func, ast.Name):
            dangerous_functions = ["eval", "exec", "compile"]
            if node.func.id in dangerous_functions:
                error = ParsedError(
                    error_type=ErrorType.RUNTIME_ERROR,
                    message=f"Potentially dangerous function call: {node.func.id}",
                    file_path=str(file_path),
                    line_number=node.lineno,
                    function_name=None,
                    stack_trace=[],
                    confidence=0.8,
                )
                errors.append(error)

        return errors

    def is_deep_analysis_result(self, text: str) -> bool:
        """Check if text contains deep analysis results in JSON format."""
        try:
            data = json.loads(text)
            # Check if it has the structure of a deep analysis result
            return isinstance(data, dict) and any(
                key in data for key in ["issues", "analysis", "recommendations", "errors"]
            )
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_deep_analysis_results(self, analysis_json: str) -> list[ParsedError]:
        """Parse deep analysis results from JSON format."""
        try:
            data = json.loads(analysis_json)
            errors: list[Any] = []
            # Extract issues from various possible structures
            issues = data.get("issues", [])
            if not issues:
                issues = data.get("errors", [])
            if not issues:
                issues = data.get("problems", [])

            for issue in issues:
                if isinstance(issue, dict):
                    error_type = self._map_issue_type_to_error_type(
                        issue.get("type", "runtime_error")
                    )

                    error = ParsedError(
                        error_type=error_type,
                        message=issue.get("message", "Unknown issue"),
                        file_path=issue.get("file", None),
                        line_number=issue.get("line", None),
                        function_name=issue.get("function", None),
                        stack_trace=issue.get("stack_trace", []),
                        confidence=issue.get("confidence", 0.7),
                    )
                    errors.append(error)

            return errors

        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Failed to parse deep analysis results: {e}")
            return []

    def _map_issue_type_to_error_type(self, issue_type: str) -> ErrorType:
        """Map issue type string to ErrorType enum."""
        mapping = {
            "syntax": ErrorType.SYNTAX_ERROR,
            "import": ErrorType.IMPORT_ERROR,
            "name": ErrorType.NAME_ERROR,
            "type": ErrorType.TYPE_ERROR,
            "attribute": ErrorType.ATTRIBUTE_ERROR,
            "index": ErrorType.INDEX_ERROR,
            "key": ErrorType.KEY_ERROR,
            "value": ErrorType.VALUE_ERROR,
            "logic": ErrorType.LOGIC_ERROR,
            "performance": ErrorType.PERFORMANCE_ERROR,
        }

        return mapping.get(issue_type.lower(), ErrorType.RUNTIME_ERROR)
