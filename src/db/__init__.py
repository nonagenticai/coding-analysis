"""Analysis-owned PostgreSQL code-graph database layer.

Replaces the removed ``coding_client.db`` aiosqlite subtree. The runtime uses
a single psycopg ``AsyncConnectionPool`` (``session``); ``facade`` and
``entity_mapper`` provide the legacy-compatible write surface; ``models``
defines the schema for Alembic.
"""

from __future__ import annotations

from .entity_mapper import MultiLanguageEntityMapper
from .facade import CodeGraphDatabase, GraphDatabaseConnection
from .session import close_pool, get_pool, get_raw_connection

__all__ = [
    "CodeGraphDatabase",
    "GraphDatabaseConnection",
    "MultiLanguageEntityMapper",
    "close_pool",
    "get_pool",
    "get_raw_connection",
]
