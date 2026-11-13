"""
Multi-language graph database service that extends the existing code graph functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid7

from psycopg.types.json import Jsonb

from db.entity_mapper import MultiLanguageEntityMapper
from db.facade import GraphDatabaseConnection
from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
    SourceLocation,
)
from processing.multi_language_factory import get_multi_language_factory

logger = logging.getLogger(__name__)


class MultiLanguageGraphService:
    """Service for managing multi-language code graphs."""

    def __init__(self, db_connection: GraphDatabaseConnection):
        self.db = db_connection
        self.entity_mapper: MultiLanguageEntityMapper | None = None
        self._known_entities: set[str] = set()

    async def initialize(self) -> None:
        """Initialize the service."""
        await self.db.connect()
        conn = self.db.get_connection()
        self.entity_mapper = MultiLanguageEntityMapper(
            conn, self.db.code_graph_id, self.db.tenant_id
        )

    async def analyze_file(self, file_path: Path, content: str | None = None) -> CoreAnalysisResult:
        """Analyze a single file and store results in the database."""
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        if content is None:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        factory = await get_multi_language_factory()
        processor = await factory.get_processor_for_file(file_path)
        if not processor:
            logger.warning(f"No processor available for {file_path}")
            return CoreAnalysisResult(
                language="unknown",
                file_path=str(file_path),
                success=False,
                entities=[],
                relationships=[],
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=f"No processor available for file type: {file_path.suffix}",
                parse_time_ms=0.0,
                total_lines=0,
                metadata={},
            )
        try:
            if hasattr(processor, "parse_file"):
                parse_result = await processor.parse_file(file_path, content)  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
            else:
                logger.warning(f"Processor for {file_path} doesn't have parse_file method")
                return CoreAnalysisResult(
                    language="unknown",
                    file_path=str(file_path),
                    success=False,
                    entities=[],
                    relationships=[],
                    dependencies=[],
                    complexity_metrics=None,
                    security_patterns=[],
                    error="Processor doesn't support parse_file method",
                    parse_time_ms=0.0,
                    total_lines=0,
                    metadata={},
                )
            if not parse_result.success:
                return CoreAnalysisResult(
                    language=parse_result.language,
                    file_path=str(file_path),
                    success=False,
                    entities=[],
                    relationships=[],
                    dependencies=[],
                    complexity_metrics=None,
                    security_patterns=[],
                    error=(
                        "; ".join(parse_result.errors) if parse_result.errors else "Parse failed"
                    ),
                    parse_time_ms=getattr(parse_result, "parse_time_ms", 0.0),
                    total_lines=getattr(parse_result, "total_lines", 0),
                    metadata={"warnings": getattr(parse_result, "warnings", [])},
                )
            for entity in parse_result.entities:
                normalized_entity = self._normalize_entity(
                    entity, parse_result.language, str(file_path)
                )
                if normalized_entity:
                    await self.entity_mapper.store_entity(normalized_entity)
                    self._known_entities.add(normalized_entity.id)
            for relationship in parse_result.relationships:
                normalized_relationship = self._normalize_relationship(
                    relationship, parse_result.language, str(file_path)
                )
                if normalized_relationship:
                    if self._can_store_relationship(normalized_relationship):
                        await self.entity_mapper.store_relationship(normalized_relationship)
                    else:
                        logger.debug(
                            "Skipping relationship %s -> %s (missing nodes)",
                            normalized_relationship.source_id,
                            normalized_relationship.target_id,
                        )
            await self.db.get_connection().commit()
            return CoreAnalysisResult(
                language=parse_result.language,
                file_path=str(file_path),
                success=True,
                entities=parse_result.entities,
                relationships=parse_result.relationships,
                dependencies=getattr(parse_result, "dependencies", []),
                complexity_metrics=getattr(parse_result, "complexity_metrics", None),
                security_patterns=getattr(parse_result, "security_patterns", []),
                error=None,
                parse_time_ms=parse_result.parse_time_ms,
                total_lines=parse_result.lines_of_code,
                metadata=getattr(parse_result, "metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            await self.db.get_connection().rollback()
            return CoreAnalysisResult(
                language=(
                    getattr(processor, "language", "unknown")
                    if hasattr(processor, "language")
                    else "unknown"
                ),
                file_path=str(file_path),
                success=False,
                entities=[],
                relationships=[],
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={},
            )

    async def analyze_directory(
        self, directory: Path, languages: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze all supported files in a directory.

        PERFORMANCE OPTIMIZATION: Batch language detection by using extension-based
        lookup instead of calling _detect_file_language() in a loop (which creates
        a factory and processor for each file). Only falls back to async detection
        for files with unknown extensions.
        """
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        session_id = str(uuid7())
        try:
            self._known_entities = set()
            await self._start_analysis_session(session_id, languages or [])
            factory = await get_multi_language_factory()

            # Fast extension-to-language mapping for batch lookup
            extension_language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".jsx": "javascript",
                ".java": "java",
                ".go": "go",
                ".rs": "rust",
                ".cs": "csharp",
                ".cpp": "cpp",
                ".c": "c",
                ".h": "c",
                ".hpp": "cpp",
                ".rb": "ruby",
            }

            supported_files = []
            languages_set = set(languages) if languages else None

            for file_path in directory.rglob("*"):
                if file_path.is_file() and factory.is_supported_file(file_path):
                    if languages_set:
                        # OPTIMIZATION: Use fast extension lookup instead of async call
                        ext = file_path.suffix.lower()
                        detected_lang = extension_language_map.get(ext)
                        # Only call async detection for unknown extensions
                        if detected_lang is None:
                            detected_lang = await self._detect_file_language(file_path)
                        if detected_lang and detected_lang in languages_set:
                            supported_files.append(file_path)
                    else:
                        supported_files.append(file_path)
            results = []
            successful_files = 0
            failed_files = 0
            total_entities = 0
            total_relationships = 0
            for file_path in supported_files:
                try:
                    result = await self.analyze_file(file_path)
                    results.append(result)
                    if result.success:
                        successful_files += 1
                        total_entities += len(result.entities)
                        total_relationships += len(result.relationships)
                    else:
                        failed_files += 1
                except Exception as e:
                    logger.error(f"Failed to analyze {file_path}: {e}")
                    failed_files += 1
            await self._complete_analysis_session(
                session_id,
                successful_files,
                failed_files,
                total_entities,
                total_relationships,
            )
            return {
                "session_id": session_id,
                "directory": str(directory),
                "files_processed": successful_files,
                "files_failed": failed_files,
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "results": results,
                "languages_detected": list(set(r.language for r in results if r.success)),
            }
        except Exception as e:
            logger.error(f"Directory analysis failed: {e}")
            await self._fail_analysis_session(session_id, str(e))
            raise

    def _normalize_entity(
        self, entity: Any, language: str | None, default_file_path: str
    ) -> CoreCodeEntity | None:
        """Convert legacy dict-based entities into CoreCodeEntity models."""
        if isinstance(entity, CoreCodeEntity):
            return entity

        if not isinstance(entity, dict):
            logger.warning("Unsupported entity payload type: %s", type(entity))
            return None

        entity_id = entity.get("id") or f"{default_file_path}::unknown"
        name = entity.get("name") or entity_id.split("::")[-1]
        entity_type_value = entity.get("type") or "function"
        entity_type = self._map_entity_type(entity_type_value)

        file_path = entity.get("file_path") or default_file_path
        line_start = (
            entity.get("line_number") or entity.get("start_line") or entity.get("line") or 1
        )
        end_line = entity.get("end_line_number") or entity.get("end_line")
        column_start = entity.get("column_start") or 0
        column_end = entity.get("column_end") or 0

        properties = entity.get("properties") or {}
        metadata = entity.get("metadata") or {}

        try:
            return CoreCodeEntity(
                id=str(entity_id),
                name=str(name),
                type=entity_type,
                language=(entity.get("language") or language or "unknown"),
                file_path=str(file_path),
                line_number=int(line_start),
                end_line_number=int(end_line) if end_line is not None else None,
                column_start=int(column_start) if column_start is not None else None,
                column_end=int(column_end) if column_end is not None else None,
                visibility=entity.get("visibility"),
                is_static=bool(entity.get("is_static", False)),
                is_abstract=bool(entity.get("is_abstract", False)),
                is_final=bool(entity.get("is_final", False)),
                is_async=bool(entity.get("is_async", False)),
                docstring=entity.get("docstring"),
                signature=entity.get("signature"),
                properties=properties,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Failed to normalize entity %s: %s", entity_id, exc)
            return None

    def _normalize_relationship(
        self, relationship: Any, language: str | None, default_file_path: str
    ) -> EntityRelation | None:
        """Convert legacy dict-based relationships into EntityRelation models."""
        if isinstance(relationship, EntityRelation):
            return relationship

        if not isinstance(relationship, dict):
            logger.warning("Unsupported relationship payload type: %s", type(relationship))
            return None

        source = relationship.get("source")
        target = relationship.get("target")
        if not source or not target:
            logger.debug("Skipping relationship with missing endpoints: %s", relationship)
            return None

        relation_type_value = relationship.get("type") or "references"
        relation_type = self._map_relation_type(relation_type_value)

        properties = relationship.get("properties") or {}
        if relationship.get("tentative"):
            properties["tentative"] = True

        location = self._build_location(relationship.get("location"), default_file_path)

        try:
            return EntityRelation(
                source_id=str(source),
                target_id=str(target),
                type=relation_type,
                properties=properties,
                context=relationship.get("context"),
                location=location,
            )
        except Exception as exc:
            logger.error("Failed to normalize relationship %s -> %s: %s", source, target, exc)
            return None

    @staticmethod
    def _map_entity_type(entity_type: Any) -> EntityType:
        """Map string-based entity types to EntityType enum."""
        value = str(entity_type).lower()
        alias_map = {
            "method": EntityType.FUNCTION,
            "classmethod": EntityType.FUNCTION,
            "staticmethod": EntityType.FUNCTION,
            "module": EntityType.MODULE,
            "file": EntityType.FILE,
            "attribute": EntityType.VARIABLE,
        }
        if value in alias_map:
            return alias_map[value]
        try:
            return EntityType(value)
        except ValueError:
            logger.debug("Unknown entity type '%s', defaulting to FUNCTION", value)
            return EntityType.FUNCTION

    @staticmethod
    def _map_relation_type(relation_type: Any) -> RelationType:
        """Map string-based relationship types to RelationType enum."""
        value = str(relation_type).lower()
        alias_map = {
            "module_contains": RelationType.CONTAINS,
            "contains": RelationType.CONTAINS,
            "inherits": RelationType.INHERITS,
            "extends": RelationType.INHERITS,
            "implements": RelationType.IMPLEMENTS,
            "calls": RelationType.CALLS,
            "call": RelationType.CALLS,
            "references": RelationType.REFERENCES,
            "usage": RelationType.REFERENCES,
            "imports": RelationType.IMPORTS,
            "import": RelationType.IMPORTS,
            "depends_on": RelationType.DEPENDS_ON,
        }
        mapped = alias_map.get(value)
        if mapped:
            return mapped
        try:
            return RelationType(value)
        except ValueError:
            logger.debug("Unknown relationship type '%s', defaulting to REFERENCES", value)
            return RelationType.REFERENCES

    @staticmethod
    def _build_location(
        location_payload: dict[str, Any] | None, default_file_path: str
    ) -> SourceLocation | None:
        """Create SourceLocation from dictionary payload."""
        if not isinstance(location_payload, dict):
            return None

        file_path = location_payload.get("file_path") or default_file_path
        line_start = (
            location_payload.get("line_start")
            or location_payload.get("line")
            or location_payload.get("line_number")
        )
        line_end = (
            location_payload.get("line_end")
            or location_payload.get("line")
            or location_payload.get("line_number")
        )
        column_start = location_payload.get("column_start") or 0
        column_end = location_payload.get("column_end") or column_start

        if line_start is None or line_end is None:
            return None

        try:
            return SourceLocation(
                file_path=str(file_path),
                line_start=int(line_start),
                line_end=int(line_end),
                column_start=int(column_start),
                column_end=int(column_end),
            )
        except Exception:
            return None

    def _can_store_relationship(self, relationship: EntityRelation) -> bool:
        """Validate that both relationship endpoints have known nodes."""
        if not relationship.source_id or not relationship.target_id:
            return False
        if relationship.source_id not in self._known_entities:
            return False
        if relationship.target_id not in self._known_entities:
            return False
        return True

    async def get_language_summary(self) -> dict[str, Any]:
        """Get summary statistics by language."""
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        cursor = await self.db.get_connection().execute(
            "SELECT language, COUNT(*) FROM nodes WHERE code_graph_id = %s GROUP BY language",
            (self.db.code_graph_id,),
        )
        summary: dict[str, Any] = {}
        async for row in cursor:
            summary[row[0]] = row[1]
        return summary

    async def get_entities_by_language(
        self, language: str, entity_type: str | None = None
    ) -> list[dict]:
        """Get entities filtered by language and optionally by type."""
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        sql_query = (
            "SELECT id, type, language, file_path, line_number, properties "
            "FROM nodes WHERE code_graph_id = %s AND language = %s"
        )
        params: list[Any] = [self.db.code_graph_id, language]
        if entity_type:
            sql_query += " AND type = %s"
            params.append(entity_type)
        cursor = await self.db.get_connection().execute(sql_query, params)
        results = []
        async for row in cursor:
            results.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "language": row[2],
                    "file_path": row[3],
                    "line_number": row[4],
                    "properties": row[5] or {},
                }
            )
        return results

    async def search_entities(self, query: str, language: str | None = None) -> list[dict]:
        """Search entities by name or properties."""
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        sql_query = (
            "\n            SELECT id, type, language, file_path, line_number, properties"
            "\n            FROM nodes"
            "\n            WHERE code_graph_id = %s"
            "\n            AND ("
            "\n                properties->>'name' LIKE %s OR"
            "\n                id LIKE %s OR"
            "\n                docstring LIKE %s"
            "\n            )\n        "
        )
        params = [self.db.code_graph_id, f"%{query}%", f"%{query}%", f"%{query}%"]
        if language:
            sql_query += " AND language = %s"
            params.append(language)
        sql_query += " ORDER BY language, type, line_number LIMIT 100"
        cursor = await self.db.get_connection().execute(sql_query, params)
        results = []
        async for row in cursor:
            results.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "language": row[2],
                    "file_path": row[3],
                    "line_number": row[4],
                    "properties": row[5] or {},
                }
            )
        return results

    async def get_cross_language_relationships(self) -> list[dict]:
        """Find relationships that cross language boundaries."""
        if not self.entity_mapper:
            raise RuntimeError("Service not initialized")
        cursor = await self.db.get_connection().execute(
            "\n            SELECT"
            "\n                e.source, e.target, e.relationship,"
            "\n                n1.language as source_language,"
            "\n                n2.language as target_language,"
            "\n                n1.type as source_type,"
            "\n                n2.type as target_type"
            "\n            FROM edges e"
            "\n            JOIN nodes n1 ON e.code_graph_id = n1.code_graph_id AND e.source = n1.id"
            "\n            JOIN nodes n2 ON e.code_graph_id = n2.code_graph_id AND e.target = n2.id"
            "\n            WHERE e.code_graph_id = %s AND n1.language != n2.language\n        ",
            (self.db.code_graph_id,),
        )
        relationships = []
        async for row in cursor:
            relationships.append(
                {
                    "source": row[0],
                    "target": row[1],
                    "relationship": row[2],
                    "source_language": row[3],
                    "target_language": row[4],
                    "source_type": row[5],
                    "target_type": row[6],
                }
            )
        return relationships

    async def _detect_file_language(self, file_path: Path) -> str | None:
        """Detect the language of a file."""
        factory = await get_multi_language_factory()
        processor = await factory.get_processor_for_file(file_path)
        if processor:
            if hasattr(processor, "config") and hasattr(processor.config, "name"):
                return str(processor.config.name)
            elif hasattr(processor, "language"):
                return str(processor.language)
            else:
                extension = file_path.suffix.lower()
                language_map = {
                    ".py": "python",
                    ".js": "javascript",
                    ".ts": "typescript",
                    ".java": "java",
                    ".go": "go",
                    ".rs": "rust",
                    ".cs": "csharp",
                    ".cpp": "cpp",
                    ".c": "c",
                    ".rb": "ruby",
                }
                return language_map.get(extension)
        return None

    async def _start_analysis_session(self, session_id: str, languages: list[str]) -> None:
        """Start a new analysis session."""
        await self.db.get_connection().execute(
            "\n            INSERT INTO analysis_sessions (code_graph_id, id, languages, status)"
            "\n            VALUES (%s, %s, %s, 'running')\n        ",
            (self.db.code_graph_id, session_id, Jsonb(languages)),
        )

    async def _complete_analysis_session(
        self,
        session_id: str,
        files_processed: int,
        files_failed: int,
        total_entities: int,
        total_relationships: int,
    ) -> None:
        """Complete an analysis session."""
        await self.db.get_connection().execute(
            "\n            UPDATE analysis_sessions\n            SET completed_at = now(),\n                files_processed = %s,\n                files_failed = %s,\n                total_entities = %s,\n                total_relationships = %s,\n                status = 'completed'\n            WHERE code_graph_id = %s AND id = %s\n        ",
            (
                files_processed,
                files_failed,
                total_entities,
                total_relationships,
                self.db.code_graph_id,
                session_id,
            ),
        )

    async def _fail_analysis_session(self, session_id: str, error_message: str) -> None:
        """Mark an analysis session as failed."""
        await self.db.get_connection().execute(
            "\n            UPDATE analysis_sessions\n            SET completed_at = now(),\n                error_message = %s,\n                status = 'failed'\n            WHERE code_graph_id = %s AND id = %s\n        ",
            (error_message, self.db.code_graph_id, session_id),
        )


class MultiLanguageGraphServiceFactory:
    """Factory for creating multi-language graph services."""

    _instance: MultiLanguageGraphService | None = None

    @classmethod
    async def get_service(cls, code_graph_id: str | None = None) -> MultiLanguageGraphService:
        """Get or create a multi-language graph service instance.

        ``code_graph_id`` defaults to a shared sentinel: the singleton backs the
        read-only architecture endpoint, which scopes every query by the
        per-request codebase id, so the connection's bound graph id is unused on
        that path.
        """
        if cls._instance is None:
            db_connection = GraphDatabaseConnection(code_graph_id or "__shared__")
            cls._instance = MultiLanguageGraphService(db_connection)
            await cls._instance.initialize()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
