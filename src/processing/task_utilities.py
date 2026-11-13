"""
Task utilities and helper functions.
Shared utilities for file operations, language detection, and common task operations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.core.file_service import FileService
from processing.multi_language_factory import get_multi_language_factory
from services.code_graph.multi_language_graph_service import (
    get_multi_language_factory as get_graph_factory,
)

logger = logging.getLogger(__name__)


class FileOperationUtils:
    """Utilities for file system operations."""

    @staticmethod
    def get_supported_file_types() -> list[str]:
        """Get list of supported file extensions."""
        return [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".rs",
            ".go",
            ".cs",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
        ]

    @staticmethod
    def get_language_extensions() -> dict[str, str]:
        """Get supported file extensions and their languages."""
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".rs": "rust",
            ".go": "go",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
        }

    @staticmethod
    def filter_directories(dirs: list[str]) -> None:
        """Filter out hidden and build directories in-place."""
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in [
                "__pycache__",
                "node_modules",
                "target",
                "build",
                "dist",
            ]
        ]

    @staticmethod
    def should_process_file(file_path: str) -> bool:
        """Check if a file should be processed based on its characteristics."""
        # Skip hidden files
        if os.path.basename(file_path).startswith("."):
            return False

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_extensions = FileOperationUtils.get_language_extensions()

        return file_ext in supported_extensions

    @staticmethod
    def get_file_language(file_path: str) -> str | None:
        """Get the programming language for a file based on its extension."""
        file_ext = os.path.splitext(file_path)[1].lower()
        language_map = FileOperationUtils.get_language_extensions()
        return language_map.get(file_ext)

    @staticmethod
    def get_file_info(file_path: str) -> dict[str, Any]:
        """Get comprehensive information about a file."""
        try:
            stat_info = os.stat(file_path)
            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": stat_info.st_size,
                "modified_time": stat_info.st_mtime,
                "extension": os.path.splitext(file_path)[1].lower(),
                "language": FileOperationUtils.get_file_language(file_path),
                "is_supported": FileOperationUtils.should_process_file(file_path),
            }
        except Exception as e:
            logger.warning("Error getting file info for %s: %s", file_path, e)
            return {
                "path": file_path,
                "error": str(e),
                "is_supported": False,
            }


class LanguageDetectionUtils:
    """Utilities for language detection and processing."""

    @staticmethod
    async def initialize_multi_language_factory():
        """Initialize and return a multi-language factory."""
        file_service = FileService()
        return await get_multi_language_factory(file_service)

    @staticmethod
    async def initialize_graph_factory():
        """Initialize and return a graph factory."""
        file_service = FileService()
        return await get_graph_factory(file_service)

    @staticmethod
    async def scan_supported_files(
        codebase_path: Path, multi_language_factory
    ) -> tuple[list[Path], dict[str, int]]:
        """Scan for supported files and collect language statistics."""
        supported_files = []
        language_stats = {}

        try:
            for file_path in codebase_path.rglob("*"):
                if file_path.is_file() and multi_language_factory.is_supported_file(file_path):
                    supported_files.append(file_path)

                    # Get language for file
                    processor = await multi_language_factory.get_processor_for_file(file_path)
                    if processor:
                        language = getattr(processor, "language", "unknown")
                        if hasattr(processor, "get_language"):
                            language = processor.get_language()

                        if language not in language_stats:
                            language_stats[language] = 0
                        language_stats[language] += 1

        except Exception as e:
            logger.error("Error scanning supported files: %s", e)

        return supported_files, language_stats

    @staticmethod
    def get_supported_languages() -> list[str]:
        """Get list of supported programming languages."""
        return [
            "python",
            "javascript",
            "typescript",
            "java",
            "rust",
            "go",
            "csharp",
            "cpp",
            "c",
        ]


class DatabaseUtils:
    """Utilities for database operations with tenant isolation support."""

    @staticmethod
    async def get_node_count(
        conn,
        node_type: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Get the count of nodes, optionally filtered by type and tenant.

        Args:
            conn: Database connection
            node_type: Optional node type filter
            tenant_id: Optional tenant ID for isolation (required for multi-tenant safety)

        Returns:
            Number of matching nodes
        """
        try:
            conditions = []
            params: list[Any] = []

            if tenant_id:
                conditions.append("tenant_id = %s")
                params.append(tenant_id)

            if node_type:
                conditions.append("type = %s")
                params.append(node_type)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT COUNT(*) FROM nodes WHERE {where_clause}"

            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error("Error getting node count: %s", e)
            return 0

    @staticmethod
    async def get_edge_count(
        conn,
        relationship_type: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Get the count of edges, optionally filtered by relationship type and tenant.

        Args:
            conn: Database connection
            relationship_type: Optional relationship type filter
            tenant_id: Optional tenant ID for isolation (required for multi-tenant safety)

        Returns:
            Number of matching edges
        """
        try:
            conditions = []
            params: list[Any] = []

            if tenant_id:
                conditions.append("tenant_id = %s")
                params.append(tenant_id)

            if relationship_type:
                conditions.append("relationship = %s")
                params.append(relationship_type)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT COUNT(*) FROM edges WHERE {where_clause}"

            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error("Error getting edge count: %s", e)
            return 0

    @staticmethod
    async def get_node_types(conn, tenant_id: str | None = None) -> list[str]:
        """Get all distinct node types in the database.

        Args:
            conn: Database connection
            tenant_id: Optional tenant ID for isolation

        Returns:
            List of distinct node types
        """
        try:
            if tenant_id:
                cursor = await conn.execute(
                    "SELECT DISTINCT type FROM nodes WHERE tenant_id = %s",
                    (tenant_id,),
                )
            else:
                cursor = await conn.execute("SELECT DISTINCT type FROM nodes")

            rows = await cursor.fetchall()
            return [row[0] for row in rows if row[0]]
        except Exception as e:
            logger.error("Error getting node types: %s", e)
            return []

    @staticmethod
    async def get_relationship_types(conn, tenant_id: str | None = None) -> list[str]:
        """Get all distinct relationship types in the database.

        Args:
            conn: Database connection
            tenant_id: Optional tenant ID for isolation

        Returns:
            List of distinct relationship types
        """
        try:
            if tenant_id:
                cursor = await conn.execute(
                    "SELECT DISTINCT relationship FROM edges WHERE tenant_id = %s",
                    (tenant_id,),
                )
            else:
                cursor = await conn.execute("SELECT DISTINCT relationship FROM edges")

            rows = await cursor.fetchall()
            return [row[0] for row in rows if row[0]]
        except Exception as e:
            logger.error("Error getting relationship types: %s", e)
            return []

    @staticmethod
    async def get_nodes(
        conn,
        node_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get nodes with tenant isolation.

        Args:
            conn: Database connection
            node_type: Optional node type filter
            tenant_id: Tenant ID for isolation (required for multi-tenant safety)
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of node dictionaries
        """
        try:
            conditions = []
            params: list[Any] = []

            if tenant_id:
                conditions.append("tenant_id = %s")
                params.append(tenant_id)

            if node_type:
                conditions.append("type = %s")
                params.append(node_type)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT id, type, properties, created_at, updated_at, tenant_id::text
                FROM nodes WHERE {where_clause}
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "properties": row[2] or {},
                    "created_at": row[3],
                    "updated_at": row[4],
                    "tenant_id": row[5],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Error getting nodes: %s", e)
            return []

    @staticmethod
    async def get_edges(
        conn,
        source_id: str | None = None,
        target_id: str | None = None,
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get edges with tenant isolation.

        Args:
            conn: Database connection
            source_id: Optional source node ID filter
            target_id: Optional target node ID filter
            relationship_type: Optional relationship type filter
            tenant_id: Tenant ID for isolation (required for multi-tenant safety)
            limit: Maximum number of edges to return
            offset: Number of edges to skip

        Returns:
            List of edge dictionaries
        """
        try:
            conditions = []
            params: list[Any] = []

            if tenant_id:
                conditions.append("tenant_id = %s")
                params.append(tenant_id)

            if source_id:
                conditions.append("source = %s")
                params.append(source_id)

            if target_id:
                conditions.append("target = %s")
                params.append(target_id)

            if relationship_type:
                conditions.append("relationship = %s")
                params.append(relationship_type)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT id, source, target, relationship, properties, weight,
                       created_at, tenant_id::text
                FROM edges WHERE {where_clause}
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "source": row[1],
                    "target": row[2],
                    "relationship": row[3],
                    "properties": row[4] or {},
                    "weight": row[5],
                    "created_at": row[6],
                    "tenant_id": row[7],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Error getting edges: %s", e)
            return []


class ProcessingUtils:
    """Utilities for task processing operations."""

    @staticmethod
    def create_processing_context(factory, conn, code_graph_id: str, **kwargs) -> dict[str, Any]:
        """Create a standardized processing context."""
        context = {
            "factory": factory,
            "conn": conn,
            "code_graph_id": code_graph_id,
            "timestamp": time.time(),
        }
        context.update(kwargs)
        return context

    @staticmethod
    def create_storage_context(
        conn, code_graph_id: str, file_path: str, language: str
    ) -> dict[str, Any]:
        """Create a standardized storage context."""
        return {
            "conn": conn,
            "code_graph_id": code_graph_id,
            "file_path": file_path,
            "language": language,
        }

    @staticmethod
    async def process_file_safely(
        file_path: str, processor, progress_callback=None
    ) -> dict[str, Any]:
        """Safely process a file with error handling."""
        try:
            # Read file content
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Process file and extract structure
            result = await processor.process_file(file_path, content)

            if progress_callback:
                progress_callback(file_path, True)

            return {
                "success": True,
                "file_path": file_path,
                "result": result,
            }

        except (
            UnicodeDecodeError,
            PermissionError,
            FileNotFoundError,
            ValueError,
            OSError,
            ImportError,
            AttributeError,
            RuntimeError,
            MemoryError,
        ) as e:
            logger.warning("Error processing file %s: %s", file_path, e)

            if progress_callback:
                progress_callback(file_path, False)

            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    @staticmethod
    def log_progress(processed: int, total: int, interval: int = 100):
        """Log processing progress at specified intervals."""
        if processed % interval == 0 and processed > 0:
            percentage = (processed / total) * 100 if total > 0 else 0
            logger.info("Processed %d/%d files (%.1f%%)", processed, total, percentage)


class PerformanceUtils:
    """Utilities for performance monitoring and optimization."""

    @staticmethod
    def create_timer():
        """Create a simple timer context manager."""

        class Timer:
            def __init__(self):
                self.start_time: float = 0.0
                self.end_time: float = 0.0
                self.duration: float = 0.0

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_time = time.time()
                self.duration = self.end_time - self.start_time

            def elapsed(self):
                if self.start_time:
                    return time.time() - self.start_time
                return 0

        return Timer()

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in a human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"

    @staticmethod
    def estimate_remaining_time(processed: int, total: int, elapsed_time: float) -> str | None:
        """Estimate remaining processing time."""
        if processed == 0 or total == 0:
            return None

        rate = processed / elapsed_time  # items per second
        remaining_items = total - processed
        remaining_seconds = remaining_items / rate

        return PerformanceUtils.format_duration(remaining_seconds)


class JsonUtils:
    """Utilities for JSON operations."""

    @staticmethod
    def safe_json_loads(json_string: str, default=None):
        """Safely load JSON with fallback."""
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return default if default is not None else {}

    @staticmethod
    def safe_json_dumps(data: Any, default=None) -> str:
        """Safely dump JSON with fallback."""
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):  # fmt: skip
            if default is not None:
                return json.dumps(default)
            return "{}"

    @staticmethod
    def prepare_node_properties(properties: dict[str, Any]) -> str:
        """Prepare node properties for database storage."""
        # Clean up properties that can't be serialized
        cleaned_properties = {}
        for key, value in properties.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                cleaned_properties[key] = value
            else:
                cleaned_properties[key] = str(value)

        return JsonUtils.safe_json_dumps(cleaned_properties)


class PathUtils:
    """Utilities for path operations."""

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize a file path."""
        return os.path.normpath(os.path.abspath(path))

    @staticmethod
    def get_relative_path(file_path: str, base_path: str) -> str:
        """Get relative path from base path."""
        try:
            return os.path.relpath(file_path, base_path)
        except ValueError:
            # Can't create relative path (different drives on Windows)
            return file_path

    @staticmethod
    def create_unique_id(file_path: str, element_name: str) -> str:
        """Create a unique ID for a code element."""
        normalized_path = PathUtils.normalize_path(file_path)
        return f"{normalized_path}::{element_name}"

    @staticmethod
    def is_subpath(path: str, parent: str) -> bool:
        """Check if path is a subpath of parent."""
        try:
            Path(parent).resolve().relative_to(Path(path).resolve())
            return True
        except ValueError:
            return False
