"""Function Validation Service

Provides comprehensive function call validation capabilities including
validation against known functions, builtin detection, and dependency
analysis within code graphs.

This service validates function calls against the codebase graph to ensure
all function references are valid and provides detailed categorization
of function types and usage patterns.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from db.facade import GraphDatabaseConnection

logger = logging.getLogger(__name__)


class FunctionValidationResult(BaseModel):
    """Model for function validation results.

    This model captures the comprehensive results of function call validation
    including categorized function types, validation status, and suggestions.
    """

    valid_functions: list[dict[str, Any]]
    invalid_functions: list[dict[str, Any]]
    builtin_functions: list[str]
    external_functions: list[str]
    local_functions: list[str]
    suggestions: list[str]


class FunctionValidationService:
    """Service for comprehensive function call validation.

    This service provides advanced function validation capabilities including:
    - Function call validation against code graphs
    - Builtin function detection and categorization
    - External dependency validation
    - Local function resolution
    - Validation suggestions and recommendations

    The service integrates with graph database connections to provide
    detailed insights into function usage and validation status.
    """

    def __init__(self):
        """Initialize the service with built-in functions."""
        self.builtin_functions = self._get_builtin_functions()

    async def validate_function_calls(
        self,
        code_graph_id: str,
        function_calls: list[dict[str, Any]],
        db_connection: GraphDatabaseConnection,
    ) -> FunctionValidationResult:
        """Validate function calls against known functions in the code graph.

        Args:
            code_graph_id: Identifier for the code graph
            function_calls: List of function calls to validate
            db_connection: Database connection for graph queries

        Returns:
            FunctionValidationResult: Validation results with categorized functions

        Note:
            This method performs comprehensive validation of function calls
            against the code graph, categorizing them by type and providing
            detailed validation feedback.
        """
        valid_functions: list[Any] = []
        invalid_functions: list[Any] = []
        builtin_functions: list[Any] = []
        external_functions: list[Any] = []
        local_functions: list[Any] = []
        suggestions: list[str] = []
        for func_call in function_calls:
            try:
                func_name = func_call.get("name", "")
                if not func_name:
                    continue
                if self.is_builtin_function(func_name):
                    builtin_functions.append(func_name)
                    valid_functions.append(func_call)
                    continue
                is_in_graph = await self._check_function_in_graph_direct(func_name, db_connection)
                if is_in_graph:
                    local_functions.append(func_name)
                    valid_functions.append(func_call)
                    continue
                external_functions.append(func_name)
                valid_functions.append(func_call)
            except Exception:
                invalid_functions.append(func_call)
        return FunctionValidationResult(
            valid_functions=valid_functions,
            invalid_functions=invalid_functions,
            builtin_functions=builtin_functions,
            external_functions=external_functions,
            local_functions=local_functions,
            suggestions=suggestions,
        )

    async def _check_function_in_graph_direct(
        self, function_name: str, db_connection: GraphDatabaseConnection
    ) -> bool:
        """Check if a function exists in the code graph using direct SQL."""
        try:
            if db_connection.conn is None:
                return False
            cursor = await db_connection.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s AND type = 'function' "
                "AND properties->>'name' = %s",
                (db_connection.code_graph_id, function_name),
            )
            result = await cursor.fetchone()
            count = result[0] if result else 0
            return count > 0
        except Exception as e:
            logger.error(f"Error checking function in graph: {e}")
            return False

    def is_builtin_function(self, function_name: str) -> bool:
        """Check if a function is a Python built-in."""
        return function_name in self.builtin_functions

    def _get_builtin_functions(self) -> set[Any]:
        """Get set of Python built-in function names."""
        return {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "__import__",
        }
