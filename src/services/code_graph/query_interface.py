"""Module for querying code graph databases and interfaces."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from db.facade import GraphDatabaseConnection

logger = logging.getLogger(__name__)


class CodeQueryInterface:
    """Interface for querying code knowledge from CodeGraph using natural language."""

    def __init__(self, db_connection: GraphDatabaseConnection):
        """Initialize with direct database connection."""
        self.db = db_connection
        self.conn = db_connection.get_connection()

    async def query_code_knowledge(self, query: str) -> dict:
        """Query code knowledge using natural language queries converted to SQL."""
        # Pattern matching for different query types
        query_patterns: list[dict[str, Any]] = [
            {
                "pattern": r"(?:what|which) (?:classes|class) (?:use|uses) ([A-Za-z0-9_]+)",
                "handler": self._find_classes_using_entity,
            },
            {
                "pattern": r"how is (?:the )?([A-Za-z0-9_]+) function implemented",
                "handler": self._find_function_implementation,
            },
        ]

        # Try to match query to pattern
        for pattern_obj in query_patterns:
            pattern_str: str = pattern_obj["pattern"]
            handler: Callable[[str], Awaitable[list[dict[str, Any]]]] = pattern_obj["handler"]
            match = re.search(pattern_str, query, re.IGNORECASE)
            if match:
                target = match.group(1) if len(match.groups()) > 0 else ""
                try:
                    results = await handler(target)
                    return {"query": query, "results": results}
                except Exception as e:
                    return {"query": query, "error": str(e)}

        # Fallback to keyword search
        keywords = self._extract_keywords(query)
        if keywords:
            results = await self._search_by_keywords(keywords)
            return {
                "query": query,
                "search_method": "keyword",
                "results": results,
            }

        return {"query": query, "error": "No matching pattern found"}

    async def _find_classes_using_entity(self, entity_name: str) -> list[dict[str, Any]]:
        """Find classes that use/depend on a specific entity using direct SQL."""
        try:
            # Find edges where target matches the entity name
            cursor = await self.conn.execute(
                """
                SELECT DISTINCT n.id, n.type, n.properties->>'name' as name,
                       n.properties->>'file_path' as file_path
                FROM nodes n
                JOIN edges e ON n.code_graph_id = e.code_graph_id AND n.id = e.source
                JOIN nodes target
                    ON e.code_graph_id = target.code_graph_id AND e.target = target.id
                WHERE n.code_graph_id = %s
                AND n.type = 'Class'
                AND (target.id = %s OR target.properties->>'name' = %s)
                AND e.relationship IN ('IMPORTS', 'CALLS', 'REFERENCES')
            """,
                (self.db.code_graph_id, entity_name, entity_name),
            )

            results: list[Any] = []
            rows = await cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "class_name": row[2] or row[0],
                        "file": row[3] or "unknown",
                        "type": row[1],
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error finding classes using {entity_name}: {e}")
            raise

    async def _find_function_implementation(self, function_name: str) -> list[dict[str, Any]]:
        """Find function implementation details using direct SQL."""
        try:
            cursor = await self.conn.execute(
                """
                SELECT id, properties->>'name' as name,
                       properties->>'file_path' as file_path,
                       properties->>'line_number' as line_number,
                       properties->>'docstring' as docstring
                FROM nodes
                WHERE code_graph_id = %s
                AND type = 'Function'
                AND (id = %s OR properties->>'name' = %s)
            """,
                (self.db.code_graph_id, function_name, function_name),
            )

            results: list[Any] = []
            rows = await cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "function_name": row[1] or row[0],
                        "file": row[2] or "unknown",
                        "line": row[3] or 0,
                        "description": row[4] or "",
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error finding function implementation {function_name}: {e}")
            raise

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract key technical terms from the query."""
        stop_words = [
            "what",
            "which",
            "how",
            "is",
            "are",
            "the",
            "in",
            "on",
            "at",
            "by",
            "with",
            "about",
        ]
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Prioritize code-like terms
        code_like_terms = [
            word for word in keywords if "_" in word or any(c.isupper() for c in word)
        ]
        return code_like_terms if code_like_terms else keywords[:3]

    async def _search_by_keywords(self, keywords: list[str]) -> list[dict]:
        """Search code elements by keywords using direct SQL."""
        results: list[Any] = []
        for keyword in keywords[:2]:  # Limit to first 2 keywords
            try:
                # Search in nodes using FTS if available
                cursor = await self.conn.execute(
                    """
                    SELECT type, properties->>'name' as name,
                           properties->>'file_path' as path,
                           properties->>'line_number' as line
                    FROM nodes
                    WHERE code_graph_id = %s
                      AND (properties->>'name' LIKE %s
                           OR properties->>'docstring' LIKE %s)
                    LIMIT 5
                """,
                    (self.db.code_graph_id, f"%{keyword}%", f"%{keyword}%"),
                )

                rows = await cursor.fetchall()
                for row in rows:
                    results.append(
                        {
                            "name": row[1] or "unnamed",
                            "type": row[0],
                            "path": row[2] or "unknown",
                            "line": row[3] or 0,
                        }
                    )
            except Exception as e:
                logger.error(f"Error searching for keyword {keyword}: {e}")

        return results
