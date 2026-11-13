"""Maps multi-language entities/relationships into the PostgreSQL code graph.

PostgreSQL reimplementation of the legacy
``coding_client.db.migrations.MultiLanguageEntityMapper``. Only the methods the
analysis service actually calls are provided: ``store_entity`` and
``store_relationship``. Both are scoped to a ``code_graph_id`` and use
``INSERT ... ON CONFLICT DO UPDATE`` covering every non-key column to preserve
the legacy ``INSERT OR REPLACE`` (full-row overwrite) semantics — a partial
``DO UPDATE`` would silently drop columns on re-ingest.
"""

from __future__ import annotations

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

# Non-key columns updated on conflict (everything except the
# (code_graph_id, tenant_id, id) identity — migration 0002).
_STORE_ENTITY_SQL = """
    INSERT INTO nodes (
        code_graph_id, id, type, language, file_path, line_number,
        column_start, column_end, visibility, is_static, is_abstract,
        is_final, is_async, signature, docstring, properties, content_hash,
        created_at, updated_at, tenant_id
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s
    )
    ON CONFLICT (code_graph_id, tenant_id, id) DO UPDATE SET
        type = EXCLUDED.type,
        language = EXCLUDED.language,
        file_path = EXCLUDED.file_path,
        line_number = EXCLUDED.line_number,
        column_start = EXCLUDED.column_start,
        column_end = EXCLUDED.column_end,
        visibility = EXCLUDED.visibility,
        is_static = EXCLUDED.is_static,
        is_abstract = EXCLUDED.is_abstract,
        is_final = EXCLUDED.is_final,
        is_async = EXCLUDED.is_async,
        signature = EXCLUDED.signature,
        docstring = EXCLUDED.docstring,
        properties = EXCLUDED.properties,
        content_hash = EXCLUDED.content_hash,
        created_at = EXCLUDED.created_at,
        updated_at = EXCLUDED.updated_at
"""

_STORE_RELATIONSHIP_SQL = """
    INSERT INTO edges (
        code_graph_id, source, target, relationship, weight, context,
        location_file, location_line, location_column, properties,
        created_at, tenant_id
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) DO UPDATE SET
        weight = EXCLUDED.weight,
        context = EXCLUDED.context,
        location_file = EXCLUDED.location_file,
        location_line = EXCLUDED.location_line,
        location_column = EXCLUDED.location_column,
        properties = EXCLUDED.properties,
        created_at = EXCLUDED.created_at
"""


class MultiLanguageEntityMapper:
    """Stores code entities and relationships for one code graph."""

    def __init__(
        self,
        connection: AsyncConnection,
        code_graph_id: str,
        tenant_id: str | None = None,
    ) -> None:
        self.conn = connection
        self.code_graph_id = code_graph_id
        self.tenant_id = tenant_id

    async def store_entity(self, entity) -> None:
        """Insert or fully overwrite a CoreCodeEntity in the ``nodes`` table."""
        properties = {
            "return_type": (entity.return_type.model_dump() if entity.return_type else None),
            "parameter_types": [pt.model_dump() for pt in entity.parameter_types],
            "properties": entity.properties,
            "metadata": entity.metadata,
        }

        await self.conn.execute(
            _STORE_ENTITY_SQL,
            (
                self.code_graph_id,
                entity.id,
                entity.type.value,
                entity.language,
                entity.file_path,
                entity.line_number,
                entity.column_start,
                entity.column_end,
                entity.visibility,
                1 if entity.is_static else 0,
                1 if entity.is_abstract else 0,
                1 if entity.is_final else 0,
                1 if entity.is_async else 0,
                entity.signature,
                entity.docstring,
                Jsonb(properties),
                entity.properties.get("content_hash"),
                entity.created_at,
                entity.updated_at,
                self.tenant_id,
            ),
        )

    async def store_relationship(self, relationship) -> None:
        """Insert or fully overwrite an EntityRelation in the ``edges`` table."""
        location = relationship.location
        await self.conn.execute(
            _STORE_RELATIONSHIP_SQL,
            (
                self.code_graph_id,
                relationship.source_id,
                relationship.target_id,
                relationship.type.value,
                relationship.weight,
                relationship.context,
                location.file_path if location else None,
                location.line_start if location else None,
                location.column_start if location else None,
                (Jsonb(relationship.properties) if relationship.properties else None),
                relationship.created_at,
                self.tenant_id,
            ),
        )
