"""
Task result processing and metadata management.
Handles analysis results, metadata storage, and result building.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

# Shared upsert SQL (analysis-owned PostgreSQL schema; scoped by code_graph_id).
#
# These write with a NULL tenant_id: there is no tenant in scope on the task path
# (analyze_codebase / multi_language_analysis in processing/task_processors.py),
# because the task payload carries none. The conflict targets still name tenant_id
# because the tenant-scoped unique indexes (migration 0002) are now the only ones —
# NULL tenants collapse via NULLS NOT DISTINCT.
#
# ⚠️ This is NOT limited to "internal metadata", as an earlier version of this
# comment claimed. FileProcessingResultHandler routes real `file`, `function`,
# `class`, `method` and dependency entities through the same statements — 6,543 of
# the untenanted `nodes` rows migration 0006 had to attribute came from here. An
# unattributed row is NOT hidden: the architecture reads in
# processing/architecture/graph_adapter.py and services/code_graph/graph_builder.py
# match `tenant_id = %s OR tenant_id IS NULL`, so it is served to EVERY tenant.
# Pinned in tests/unit/test_tenant_writer_ratchet.py so no further writer joins it.
_UPSERT_NODE_SQL = """
    INSERT INTO nodes (code_graph_id, id, type, properties)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (code_graph_id, tenant_id, id) DO UPDATE SET
        type = EXCLUDED.type,
        properties = EXCLUDED.properties,
        updated_at = now()
"""

_UPSERT_EDGE_SQL = """
    INSERT INTO edges (code_graph_id, source, target, relationship)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO NOTHING
"""


class TaskResultProcessor:
    """Handles processing and storage of task results."""

    async def build_analysis_result(
        self,
        conn,
        processed_files: int,
        scan_depth: int,
        start_time: float,
        code_graph_id: str,
    ) -> dict[str, Any]:
        """Build the analysis result with statistics."""
        from .task_execution_engine import calculate_pagerank

        # Calculate statistics with direct async SQL
        node_count = await self._get_node_count(conn, code_graph_id)
        edge_count = await self._get_edge_count(conn, code_graph_id)

        # Calculate centrality metrics if deep scan
        pagerank: dict[str, Any] = {}
        if scan_depth > 1:
            pagerank = await calculate_pagerank(conn, code_graph_id)

        duration = time.time() - start_time

        return {
            "success": True,
            "processed_files": processed_files,
            "node_count": node_count,
            "edge_count": edge_count,
            "duration_seconds": duration,
            "important_nodes": (
                await self._get_important_nodes(conn, code_graph_id, pagerank)
                if scan_depth > 1
                else []
            ),
        }

    async def _get_node_count(self, conn, code_graph_id: str) -> int:
        """Get the total number of nodes in the database."""
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def _get_edge_count(self, conn, code_graph_id: str) -> int:
        """Get the total number of edges in the database."""
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (code_graph_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def _get_important_nodes(
        self, conn, code_graph_id: str, pagerank: dict[str, float]
    ) -> list:
        """Get the most important nodes based on PageRank using direct async SQL."""
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

        important_nodes: list[Any] = []
        for node_id, score in top_nodes:
            # Direct async SQL query instead of graph_manager.get_node()
            cursor = await conn.execute(
                "SELECT id, type, properties FROM nodes WHERE code_graph_id = %s AND id = %s",
                (code_graph_id, node_id),
            )
            row = await cursor.fetchone()

            if row:
                properties = row[2] or {}
                important_nodes.append(
                    {
                        "id": node_id,
                        "type": row[1],
                        "name": properties.get("name", ""),
                        "pagerank": score,
                    }
                )

        return important_nodes

    def build_multi_language_result(
        self, analysis_data: dict[str, Any], start_time: float
    ) -> dict[str, Any]:
        """Build the final multi-language analysis result."""
        duration = time.time() - start_time
        logger.info("✅ Multi-language analysis completed in %.2fs", duration)

        return {
            "success": True,
            "duration_seconds": duration,
            "total_files_analyzed": len(analysis_data["supported_files"]),
            "languages_detected": list(analysis_data["language_stats"].keys()),
            "language_stats": analysis_data["language_stats"],
            "language_analysis_results": analysis_data["language_analysis_results"],
            "supported_languages": analysis_data[
                "multi_language_factory"
            ].get_supported_languages(),
        }


class AnalysisMetadataManager:
    """Manages storage and retrieval of analysis metadata."""

    async def store_language_analysis(
        self, conn, code_graph_id: str, language: str, language_results: dict
    ) -> None:
        """Store language-specific analysis results."""
        await conn.execute(
            _UPSERT_NODE_SQL,
            (
                code_graph_id,
                f"lang_analysis_{language}_{code_graph_id}",
                "language_analysis",
                Jsonb(
                    {
                        "code_graph_id": code_graph_id,
                        "language": language,
                        "analysis_results": language_results,
                        "analysis_timestamp": time.time(),
                    }
                ),
            ),
        )

    async def store_analysis_metadata(self, metadata_context: dict[str, Any]) -> None:
        """Store overall multi-language analysis metadata."""
        conn = metadata_context["conn"]
        code_graph_id = metadata_context["code_graph_id"]
        supported_files = metadata_context["supported_files"]
        language_stats = metadata_context["language_stats"]
        multi_language_factory = metadata_context["multi_language_factory"]

        await conn.execute(
            _UPSERT_NODE_SQL,
            (
                code_graph_id,
                f"multi_lang_analysis_{code_graph_id}",
                "multi_language_metadata",
                Jsonb(
                    {
                        "code_graph_id": code_graph_id,
                        "total_files": len(supported_files),
                        "languages_detected": list(language_stats.keys()),
                        "language_stats": language_stats,
                        "supported_extensions": multi_language_factory.get_supported_extensions(),
                        "analysis_timestamp": time.time(),
                        "processor_factory_info": {
                            "supported_languages": multi_language_factory.get_supported_languages(),
                            "total_processors": len(
                                multi_language_factory.get_supported_languages()
                            ),
                        },
                    }
                ),
            ),
        )
        await conn.commit()

    async def get_analysis_metadata(self, conn, code_graph_id: str) -> dict[str, Any]:
        """Retrieve analysis metadata for a code graph."""
        cursor = await conn.execute(
            "SELECT properties FROM nodes WHERE code_graph_id = %s AND id = %s AND type = %s",
            (code_graph_id, f"multi_lang_analysis_{code_graph_id}", "multi_language_metadata"),
        )
        row = await cursor.fetchone()

        if row and row[0]:
            return row[0]

        return {}

    async def get_language_analysis(
        self, conn, code_graph_id: str, language: str
    ) -> dict[str, Any]:
        """Retrieve language-specific analysis results."""
        cursor = await conn.execute(
            "SELECT properties FROM nodes WHERE code_graph_id = %s AND id = %s AND type = %s",
            (code_graph_id, f"lang_analysis_{language}_{code_graph_id}", "language_analysis"),
        )
        row = await cursor.fetchone()

        if row and row[0]:
            return row[0]

        return {}


class FileProcessingResultHandler:
    """Handles results from file processing operations."""

    async def store_processing_results(
        self,
        conn,
        code_graph_id: str,
        processing_data: dict[str, Any],
    ) -> None:
        """Store processing results in the database.

        Args:
            conn: Database connection
            code_graph_id: Graph ID for storage
            processing_data: Dictionary containing file_path, language, and result
        """
        file_path = processing_data["file_path"]
        language = processing_data["language"]
        result = processing_data["result"]

        try:
            # Store file node
            await self._store_file_node(conn, code_graph_id, file_path, language)

            # Create storage context
            storage_ctx = {
                "conn": conn,
                "code_graph_id": code_graph_id,
                "file_path": file_path,
                "language": language,
            }

            # Store functions
            await self._store_functions(storage_ctx, result.get("functions", []))

            # Store classes and their methods
            await self._store_classes_and_methods(storage_ctx, result.get("classes", []))

            # Store dependencies
            await self._store_dependencies(storage_ctx, result.get("dependencies", []))

            await conn.commit()

        except Exception as e:
            logger.error("Error storing processing results for %s: %s", file_path, e)
            await conn.rollback()
            raise

    async def _store_file_node(
        self, conn, code_graph_id: str, file_path: str, language: str
    ) -> None:
        """Store file node in database."""
        import os

        file_properties = {
            "name": os.path.basename(file_path),
            "path": file_path,
            "language": language,
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        }

        await conn.execute(
            _UPSERT_NODE_SQL,
            (code_graph_id, file_path, "file", Jsonb(file_properties)),
        )

    async def _store_functions(self, storage_ctx: dict[str, Any], functions: list) -> None:
        """Store function nodes and edges."""
        conn = storage_ctx["conn"]
        code_graph_id = storage_ctx["code_graph_id"]
        file_path = storage_ctx["file_path"]
        language = storage_ctx["language"]

        for func in functions:
            func_id = f"{file_path}::{func.get('name', 'unknown')}"
            func_properties = {
                "name": func.get("name"),
                "line_number": func.get("line_number"),
                "file_path": file_path,
                "language": language,
                "signature": func.get("signature", ""),
                "complexity": func.get("complexity", 0),
            }

            await conn.execute(
                _UPSERT_NODE_SQL,
                (code_graph_id, func_id, "function", Jsonb(func_properties)),
            )

            # Create edge from file to function
            await conn.execute(
                _UPSERT_EDGE_SQL,
                (code_graph_id, file_path, func_id, "contains"),
            )

    async def _store_classes_and_methods(self, storage_ctx: dict[str, Any], classes: list) -> None:
        """Store class nodes, method nodes, and their relationships."""
        conn = storage_ctx["conn"]
        code_graph_id = storage_ctx["code_graph_id"]
        file_path = storage_ctx["file_path"]
        language = storage_ctx["language"]

        for cls in classes:
            cls_id = f"{file_path}::{cls.get('name', 'unknown')}"
            cls_properties = {
                "name": cls.get("name"),
                "line_number": cls.get("line_number"),
                "file_path": file_path,
                "language": language,
                "methods": len(cls.get("methods", [])),
            }

            await conn.execute(
                _UPSERT_NODE_SQL,
                (code_graph_id, cls_id, "class", Jsonb(cls_properties)),
            )

            # Create edge from file to class
            await conn.execute(
                _UPSERT_EDGE_SQL,
                (code_graph_id, file_path, cls_id, "contains"),
            )

            # Store class methods
            await self._store_class_methods(storage_ctx, cls_id, cls)

    async def _store_class_methods(
        self, storage_ctx: dict[str, Any], cls_id: str, cls: dict
    ) -> None:
        """Store method nodes for a class."""
        conn = storage_ctx["conn"]
        code_graph_id = storage_ctx["code_graph_id"]
        file_path = storage_ctx["file_path"]
        language = storage_ctx["language"]

        for method in cls.get("methods", []):
            method_id = f"{cls_id}::{method.get('name', 'unknown')}"
            method_properties = {
                "name": method.get("name"),
                "line_number": method.get("line_number"),
                "class_name": cls.get("name"),
                "file_path": file_path,
                "language": language,
                "visibility": method.get("visibility", "public"),
            }

            await conn.execute(
                _UPSERT_NODE_SQL,
                (code_graph_id, method_id, "method", Jsonb(method_properties)),
            )

            # Create edge from class to method
            await conn.execute(
                _UPSERT_EDGE_SQL,
                (code_graph_id, cls_id, method_id, "contains"),
            )

    async def _store_dependencies(self, storage_ctx: dict[str, Any], dependencies: list) -> None:
        """Store dependency nodes and edges."""
        conn = storage_ctx["conn"]
        code_graph_id = storage_ctx["code_graph_id"]
        file_path = storage_ctx["file_path"]
        language = storage_ctx["language"]

        for dep in dependencies:
            dep_id = f"dep::{dep.get('target', 'unknown')}"
            dep_properties = {
                "name": dep.get("target"),
                "type": dep.get("type", "import"),
                "language": language,
            }

            await conn.execute(
                _UPSERT_NODE_SQL,
                (code_graph_id, dep_id, "dependency", Jsonb(dep_properties)),
            )

            # Create dependency edge
            await conn.execute(
                _UPSERT_EDGE_SQL,
                (code_graph_id, file_path, dep_id, "depends_on"),
            )
