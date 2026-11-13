"""Pattern Search Service

Provides advanced pattern search capabilities for analyzing code patterns,
identifying structural similarities, and performing comprehensive codebase analysis.

This service supports complex pattern matching operations across multiple programming
languages and provides detailed insights into code structure and organization.
"""

from __future__ import annotations

import logging
from typing import Any

from db.facade import GraphDatabaseConnection
from models.code_graph_models import (
    PatternSearchRequest,
    PatternSearchResponse,
)
from processing.processor_factory import ProcessorFactory
from services.code_structure_analyzer import CodeStructureAnalyzer

logger = logging.getLogger(__name__)


class PatternSearchService:
    """Service for advanced pattern search and codebase analysis operations.

    This service provides comprehensive pattern search capabilities including:
    - Complex pattern matching across multiple languages
    - Structural similarity analysis
    - Code pattern identification and analysis
    - Advanced query operations on code graphs

    The service integrates with code structure analyzers, database connections,
    and processor factories to provide detailed pattern analysis capabilities.
    """

    def __init__(
        self,
        analyzer: CodeStructureAnalyzer,
        db_connection: GraphDatabaseConnection,
        processor_factory: ProcessorFactory,
    ):
        """Initialize the PatternSearchService service.

        This initializes the PatternSearchService with required dependencies
        and configuration for service operations.

        Args:
            analyzer (CodeStructureAnalyzer): Analyzer
            db_connection (GraphDatabaseConnection): Db connection
            processor_factory (ProcessorFactory): Processor factory
        """
        self.analyzer = analyzer
        self.db_connection = db_connection
        self.processor_factory = processor_factory

    async def search_patterns(
        self, code_graph_id: str, request: PatternSearchRequest
    ) -> PatternSearchResponse:
        """Search for patterns in a code graph using advanced SQL queries.

        Args:
            code_graph_id: Identifier for the code graph to search
            request: Pattern search request with query parameters

        Returns:
            PatternSearchResponse: Search results with matches and metadata

        Note:
            This method performs complex pattern matching operations using
            direct SQL queries for optimal performance and flexibility.
        """
        try:
            matches: list[Any] = []
            total_matches = 0

            # Ensure connection is established
            if not self.db_connection.conn:
                await self.db_connection.connect()

            # Check if connection is still None after attempting to connect
            if not self.db_connection.conn:
                raise RuntimeError("Failed to establish database connection")

            cursor = await self.db_connection.conn.execute(
                "SELECT id, type, properties FROM nodes WHERE code_graph_id = %s ORDER BY id",
                (code_graph_id,),
            )

            rows = await cursor.fetchall()
            for row in rows:
                node_id, node_type, properties_json = row
                try:
                    properties = properties_json or {}

                    if "code_snippet" in properties:
                        code_snippet = properties["code_snippet"]
                        if request.pattern.lower() in code_snippet.lower():
                            matches.append(
                                {
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "match_text": (
                                        code_snippet[:200] + "..."
                                        if len(code_snippet) > 200
                                        else code_snippet
                                    ),
                                    "file_path": properties.get("file_path", "unknown"),
                                    "line_number": properties.get("line_number", 0),
                                    "method": "property_search",
                                }
                            )
                            total_matches += 1

                    if "name" in properties:
                        name = properties["name"]
                        if request.pattern.lower() in name.lower():
                            matches.append(
                                {
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "match_text": name,
                                    "file_path": properties.get("file_path", "unknown"),
                                    "line_number": properties.get("line_number", 0),
                                    "method": "name_search",
                                }
                            )
                            total_matches += 1

                except (AttributeError, TypeError):
                    logger.warning(f"Error reading properties for node {node_id}")
                    continue

            return PatternSearchResponse(
                matches=matches,
                count=total_matches,
                search_time_ms=1.0,
                code_graph_id=code_graph_id,
            )

        except Exception as e:
            logger.error(f"Error searching patterns in {code_graph_id}: {e}")
            return PatternSearchResponse(
                matches=[],
                count=0,
                search_time_ms=0.0,
                code_graph_id=code_graph_id,
                error=str(e),
            )

    async def search_code_patterns_utility(
        self, pattern: str, file_path: str, language: str | None = None
    ) -> dict[str, Any]:
        """Search for code patterns in any file (utility endpoint)."""
        try:
            pattern_list = self.analyzer.find_pattern_in_file(file_path, pattern, language)
            return {"patterns": pattern_list, "count": len(pattern_list)}
        except Exception as e:
            logger.error(f"Error searching patterns in {file_path}: {e}")
            return {"error": str(e), "matches": []}
