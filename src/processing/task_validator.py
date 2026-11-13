"""
Task validation and monitoring functionality.
Handles error validation, task parameter validation, and processing monitoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)


class TaskParameterValidator:
    """Validates task parameters and input data."""

    @staticmethod
    def validate_codebase_analysis_params(task_data: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters for codebase analysis task."""
        errors = []

        # Check required parameters
        if not task_data.get("codebase_path"):
            errors.append("Missing required parameter: codebase_path")

        # Validate optional parameters
        scan_depth = task_data.get("scan_depth", 1)
        if not isinstance(scan_depth, int) or scan_depth < 1:
            errors.append("scan_depth must be a positive integer")

        # No "default" fallback: a missing code_graph_id must fail validation
        # rather than silently funnel untagged tasks into one shared graph
        # (cross-tenant bleed). The non-empty-string check below rejects None.
        code_graph_id = task_data.get("code_graph_id")
        if not isinstance(code_graph_id, str) or not code_graph_id.strip():
            errors.append("code_graph_id must be a non-empty string")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "validated_params": {
                "codebase_path": task_data.get("codebase_path"),
                "scan_depth": scan_depth,
                "code_graph_id": code_graph_id,
            },
        }

    @staticmethod
    def validate_centrality_calculation_params(
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate parameters for centrality calculation task."""
        errors = []

        # No "default" fallback: a missing code_graph_id must fail validation
        # rather than silently funnel untagged tasks into one shared graph
        # (cross-tenant bleed). The non-empty-string check below rejects None.
        code_graph_id = task_data.get("code_graph_id")
        if not isinstance(code_graph_id, str) or not code_graph_id.strip():
            errors.append("code_graph_id must be a non-empty string")

        node_type = task_data.get("node_type")
        if node_type is not None and not isinstance(node_type, str):
            errors.append("node_type must be a string or None")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "validated_params": {
                "code_graph_id": code_graph_id,
                "node_type": node_type,
            },
        }

    @staticmethod
    def validate_multi_language_analysis_params(
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate parameters for multi-language analysis task."""
        errors = []

        if not task_data.get("codebase_path"):
            errors.append("Missing required parameter: codebase_path")

        # No "default" fallback: a missing code_graph_id must fail validation
        # rather than silently funnel untagged tasks into one shared graph
        # (cross-tenant bleed). The non-empty-string check below rejects None.
        code_graph_id = task_data.get("code_graph_id")
        if not isinstance(code_graph_id, str) or not code_graph_id.strip():
            errors.append("code_graph_id must be a non-empty string")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "validated_params": {
                "codebase_path": task_data.get("codebase_path"),
                "code_graph_id": code_graph_id,
            },
        }


class TaskErrorHandler:
    """Handles and categorizes task execution errors."""

    ERROR_CATEGORIES = {
        "FILE_SYSTEM": [FileNotFoundError, OSError, PermissionError],
        "DATABASE": [psycopg.Error],
        "DATA_PROCESSING": [ValueError, json.JSONDecodeError, KeyError, TypeError],
        "IMPORT_MODULE": [ImportError, AttributeError, ModuleNotFoundError],
        "RUNTIME": [RuntimeError, MemoryError, RecursionError],
        "UNICODE": [UnicodeDecodeError, UnicodeError],
    }

    @classmethod
    def categorize_error(cls, error: Exception) -> str:
        """Categorize an error based on its type."""
        error_type = type(error)

        for category, error_types in cls.ERROR_CATEGORIES.items():
            if error_type in error_types:
                return category

        return "UNKNOWN"

    @classmethod
    def create_error_response(cls, error: Exception, context: str = "") -> dict[str, Any]:
        """Create a standardized error response."""
        category = cls.categorize_error(error)

        return {
            "success": False,
            "error": str(error),
            "error_category": category,
            "error_type": type(error).__name__,
            "context": context,
        }

    @staticmethod
    def log_error(error: Exception, context: str, task_type: str = "unknown"):
        """Log an error with appropriate level and context."""
        error_category = TaskErrorHandler.categorize_error(error)

        if error_category in ["FILE_SYSTEM", "DATABASE"]:
            logger.error(
                "%s error in %s task (%s): %s",
                error_category.lower(),
                task_type,
                context,
                error,
            )
        elif error_category in ["RUNTIME", "IMPORT_MODULE"]:
            logger.error(
                "%s error in %s task (%s): %s",
                error_category.lower(),
                task_type,
                context,
                error,
                exc_info=True,
            )
        else:
            logger.warning(
                "%s error in %s task (%s): %s",
                error_category.lower(),
                task_type,
                context,
                error,
            )


class TaskMonitor:
    """Monitors task execution and provides insights."""

    def __init__(self):
        self._total_tasks: int = 0
        self._successful_tasks: int = 0
        self._failed_tasks: int = 0
        self._error_categories: dict[str, int] = {}

    def record_task_start(self, task_type: str, task_data: dict[str, Any]):
        """Record the start of a task."""
        self._total_tasks += 1
        logger.info("Starting %s task (total: %d)", task_type, self._total_tasks)

    def record_task_success(self, task_type: str, duration: float):
        """Record a successful task completion."""
        self._successful_tasks += 1
        logger.info(
            "Completed %s task successfully in %.2fs (success rate: %.1f%%)",
            task_type,
            duration,
            (self._successful_tasks / self._total_tasks) * 100,
        )

    def record_task_failure(self, task_type: str, error: Exception, duration: float):
        """Record a failed task."""
        self._failed_tasks += 1
        error_category = TaskErrorHandler.categorize_error(error)

        if error_category not in self._error_categories:
            self._error_categories[error_category] = 0
        self._error_categories[error_category] += 1

        logger.error(
            "Failed %s task after %.2fs (failure rate: %.1f%%, error: %s)",
            task_type,
            duration,
            (self._failed_tasks / self._total_tasks) * 100,
            error_category,
        )

    @property
    def task_stats(self) -> dict[str, Any]:
        """Get task stats as a dict for backward compatibility."""
        return {
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "error_categories": self._error_categories,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get current task execution statistics."""
        total = self._total_tasks
        if total == 0:
            return {"message": "No tasks executed yet"}

        return {
            "total_tasks": total,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": (self._successful_tasks / total) * 100,
            "failure_rate": (self._failed_tasks / total) * 100,
            "error_categories": self._error_categories,
        }

    def reset_statistics(self):
        """Reset all task statistics."""
        self._total_tasks = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._error_categories = {}
        logger.info("Task statistics reset")


class FileSystemValidator:
    """Validates file system operations and paths."""

    @staticmethod
    def validate_codebase_path(codebase_path: str) -> dict[str, Any]:
        """Validate that a codebase path exists and is accessible."""
        import os
        from pathlib import Path

        try:
            path_obj = Path(codebase_path)

            if not path_obj.exists():
                return {
                    "is_valid": False,
                    "error": f"Codebase path does not exist: {codebase_path}",
                }

            if not path_obj.is_dir():
                return {
                    "is_valid": False,
                    "error": f"Codebase path is not a directory: {codebase_path}",
                }

            # Check if directory is readable
            if not os.access(codebase_path, os.R_OK):
                return {
                    "is_valid": False,
                    "error": f"Codebase path is not readable: {codebase_path}",
                }

            return {
                "is_valid": True,
                "absolute_path": str(path_obj.absolute()),
                "size_info": FileSystemValidator._get_directory_info(path_obj),
            }

        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Error validating codebase path: {str(e)}",
            }

    @staticmethod
    def _get_directory_info(path_obj) -> dict[str, Any]:
        """Get basic information about a directory."""
        try:
            files = list(path_obj.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])

            return {
                "total_files": file_count,
                "total_directories": dir_count,
                "total_items": len(files),
            }
        except Exception:
            return {"error": "Could not gather directory information"}
