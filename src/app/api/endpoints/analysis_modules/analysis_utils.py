"""
Analysis Utility Functions

Shared utility functions for the analysis endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path as PathLib
from typing import Any

import psycopg
from fastapi import HTTPException
from psycopg.types.json import Jsonb

from app.api.utils.workspace_files import (
    MaterializedFiles,
    materialize_tenant_workspace,
    materialized_files,
    remove_materialized,
)
from db.session import get_raw_connection

logger = logging.getLogger(__name__)

# Supported file extensions for different languages
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",  # Python
        ".js",  # JavaScript
        ".jsx",  # React JavaScript
        ".ts",  # TypeScript
        ".tsx",  # React TypeScript
        ".java",  # Java
        ".kt",  # Kotlin
        ".scala",  # Scala
        ".cpp",  # C++
        ".cc",  # C++
        ".cxx",  # C++
        ".c",  # C
        ".h",  # C/C++ headers
        ".hpp",  # C++ headers
        ".cs",  # C#
        ".fs",  # F#
        ".vb",  # Visual Basic
        ".go",  # Go
        ".rs",  # Rust
        ".rb",  # Ruby
        ".php",  # PHP
        ".swift",  # Swift
        ".m",  # Objective-C
        ".mm",  # Objective-C++
        ".r",  # R
        ".R",  # R (alternative)
        ".pl",  # Perl
        ".pm",  # Perl module
        ".sh",  # Shell script
        ".bash",  # Bash script
        ".ps1",  # PowerShell
        ".lua",  # Lua
        ".dart",  # Dart
        ".elm",  # Elm
        ".ex",  # Elixir
        ".exs",  # Elixir script
        ".erl",  # Erlang
        ".hrl",  # Erlang header
        ".clj",  # Clojure
        ".cljs",  # ClojureScript
        ".hs",  # Haskell
        ".lhs",  # Literate Haskell
        ".ml",  # OCaml
        ".mli",  # OCaml interface
        ".vim",  # Vim script
        ".asm",  # Assembly
        ".s",  # Assembly
        ".sql",  # SQL
        ".jl",  # Julia
        ".nim",  # Nim
        ".cr",  # Crystal
        ".d",  # D
        ".pas",  # Pascal
        ".pp",  # Pascal
        ".ada",  # Ada
        ".adb",  # Ada body
        ".ads",  # Ada spec
        ".f90",  # Fortran
        ".f95",  # Fortran
        ".f03",  # Fortran
        ".f08",  # Fortran
        ".cob",  # COBOL
        ".cbl",  # COBOL
        ".groovy",  # Groovy
        ".gradle",  # Gradle
        ".sbt",  # SBT
    }
)

# Directories to skip (common non-code directories)
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        ".bzr",  # Version control
        "node_modules",
        "__pycache__",
        ".tox",  # Package managers/cache
        "build",
        "dist",
        "target",
        "bin",
        "obj",  # Build outputs
        ".idea",
        ".vscode",
        ".vs",  # IDEs
        "vendor",
        "third_party",
        "external",  # Dependencies
        "docs",
        "documentation",  # Documentation
        "test_results",
        "coverage",  # Test outputs
        ".pytest_cache",
        ".coverage",  # Test cache
        "venv",
        "env",
        ".env",
        ".venv",  # Virtual environments
        "temp",
        "tmp",
        "cache",  # Temporary files
    }
)


async def populate_graph_database(
    container_name: str, analysis_data: dict[str, Any], tenant_id: str | None = None
) -> bool:
    """
    Graph database population with intelligent path resolution.
    This is the critical resolve for database population during analysis sessions.

    ``tenant_id`` is persisted on every node/edge row so the graph is attributed
    to its owning tenant (multi-tenant isolation). Leaving it NULL — as this
    writer historically did — does NOT hide the rows; it exposes them. Three read
    sites are deliberately NULL-tolerant (``tenant_id = %s OR tenant_id IS NULL``
    in ``processing/architecture/graph_adapter.py`` and
    ``services/code_graph/graph_builder.py``, added by PR #54 so cycle detection
    did not regress on legacy graphs), and every ``code_graph_id``-only read in
    ``graphs_reader.py`` does not filter on tenant at all. So an unattributed row
    is served to EVERY tenant that clears the container-name ownership check —
    only ``list_graphs`` and ``_clear_parsed_nodes`` are strict. Callers must
    always pass a tenant.
    """
    code_graph_id = container_name
    try:
        logger.info("🚀 Starting graph database population for code graph: %s", code_graph_id)
        logger.info(
            "📊 Processing %d nodes, %d edges",
            len(analysis_data.get("nodes", [])),
            len(analysis_data.get("edges", [])),
        )

        # Shared PostgreSQL: schema is owned by Alembic; rows are scoped to one
        # ``code_graph_id``. No per-container file, file lock, or DDL here.
        async with get_raw_connection() as conn:
            # Get existing node IDs for edge validation (scoped to this graph)
            cursor = await conn.execute(
                "SELECT id FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
            )
            rows = await cursor.fetchall()
            existing_node_ids: set[Any] = {row[0] for row in rows if row is not None}
            logger.debug("Found %d existing nodes in database", len(existing_node_ids))

            # Insert analysis data with error handling and deduplication
            nodes_inserted = 0
            edges_inserted = 0
            nodes_skipped = 0
            edges_skipped = 0

            # Process nodes (name/file_path stored inside properties JSON)
            seen_node_ids: set[Any] = set()
            for node in analysis_data.get("nodes", []):
                try:
                    node_id = node["id"]

                    # Skip duplicates to prevent unnecessary replacements
                    if node_id in seen_node_ids:
                        nodes_skipped += 1
                        logger.debug("Skipping duplicate node: %s", node_id)
                        continue
                    seen_node_ids.add(node_id)

                    # Store name and file_path in properties JSON
                    properties = node.get("properties", {})
                    if "name" not in properties and "name" in node:
                        properties["name"] = node["name"]
                    if "file_path" not in properties and "file_path" in node:
                        properties["file_path"] = node.get("file_path", "")

                    # ON CONFLICT DO NOTHING preserves the legacy INSERT OR IGNORE
                    cursor = await conn.execute(
                        "INSERT INTO nodes (code_graph_id, id, type, properties, tenant_id) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (code_graph_id, tenant_id, id) DO NOTHING",
                        (code_graph_id, node_id, node["type"], Jsonb(properties), tenant_id),
                    )

                    # Only count if row was actually inserted
                    if cursor.rowcount > 0:
                        nodes_inserted += 1
                    else:
                        nodes_skipped += 1
                        logger.debug("Node already exists: %s", node_id)

                except (psycopg.Error, ValueError, KeyError) as node_error:
                    logger.warning(
                        "⚠️ Failed to insert node %s: %s",
                        node.get("id", "unknown"),
                        node_error,
                    )

            # Process edges
            seen_edge_combinations: set[Any] = set()
            orphaned_edges_skipped = 0
            for edge in analysis_data.get("edges", []):
                try:
                    # Create a unique identifier for the edge
                    edge_key = (
                        edge["source_id"],
                        edge["target_id"],
                        edge["type"],
                    )

                    # Skip duplicates
                    if edge_key in seen_edge_combinations:
                        edges_skipped += 1
                        logger.debug("Skipping duplicate edge: %s", edge_key)
                        continue
                    seen_edge_combinations.add(edge_key)

                    # Validate that both source and target nodes exist
                    # (including newly inserted ones)
                    all_node_ids = existing_node_ids | seen_node_ids
                    if (
                        edge["source_id"] not in all_node_ids
                        or edge["target_id"] not in all_node_ids
                    ):
                        orphaned_edges_skipped += 1
                        edges_skipped += 1
                        logger.debug("Skipping orphaned edge: %s (missing nodes)", edge_key)
                        continue

                    cursor = await conn.execute(
                        "INSERT INTO edges (code_graph_id, source, target, relationship, "
                        "properties, tenant_id) VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (code_graph_id, tenant_id, source, target, relationship) "
                        "DO NOTHING",
                        (
                            code_graph_id,
                            edge["source_id"],
                            edge["target_id"],
                            edge["type"],
                            Jsonb(edge.get("properties", {})),
                            tenant_id,
                        ),
                    )

                    # Only count if row was actually inserted
                    if cursor.rowcount > 0:
                        edges_inserted += 1
                    else:
                        edges_skipped += 1
                        logger.debug("Edge already exists: %s", edge_key)

                except (psycopg.Error, ValueError, KeyError) as edge_error:
                    logger.warning(
                        "⚠️ Failed to insert edge %s: %s",
                        edge.get("id", "unknown"),
                        edge_error,
                    )

            await conn.commit()

            # Verify the data was inserted (scoped to this graph)
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (code_graph_id,)
            )
            node_row = await cursor.fetchone()
            final_node_count: int = int(node_row[0]) if node_row is not None else 0

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (code_graph_id,)
            )
            edge_row = await cursor.fetchone()
            final_edge_count: int = int(edge_row[0]) if edge_row is not None else 0

            logger.info("✅ Graph database populated successfully!")
            logger.info("   📊 Inserted %s nodes, %s edges", nodes_inserted, edges_inserted)
            logger.info(
                "   📊 Skipped %s duplicate nodes, %s duplicate edges",
                nodes_skipped,
                edges_skipped,
            )
            if orphaned_edges_skipped > 0:
                logger.info(
                    "   🔗 Prevented %s orphaned edges (referencing non-existent nodes)",
                    orphaned_edges_skipped,
                )
            logger.info(
                "   📊 Final counts: %s nodes, %s edges",
                final_node_count,
                final_edge_count,
            )
            logger.info("   🗃️ Code graph: %s", code_graph_id)

            # Perform database integrity check (scoped to this graph)
            try:
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM edges e
                    WHERE e.code_graph_id = %s
                      AND (e.source NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s)
                           OR e.target NOT IN (SELECT id FROM nodes WHERE code_graph_id = %s))
                """,
                    (code_graph_id, code_graph_id, code_graph_id),
                )
                orphan_row = await cursor.fetchone()
                orphaned_count: int = int(orphan_row[0]) if orphan_row is not None else 0

                if orphaned_count > 0:
                    logger.warning("   ⚠️ Found %s orphaned edges in database", orphaned_count)
                else:
                    logger.info("   ✅ Database integrity verified: no orphaned edges found")

                # Check node types distribution
                cursor = await conn.execute(
                    "SELECT type, COUNT(*) as count FROM nodes WHERE code_graph_id = %s "
                    "GROUP BY type ORDER BY count DESC LIMIT 5",
                    (code_graph_id,),
                )
                node_types = await cursor.fetchall()
                if node_types:
                    type_counts: dict[Any, Any] = {}
                    for row in node_types:
                        if row is not None and len(row) >= 2:
                            type_counts[row[0]] = row[1]
                    logger.info("   📈 Top node types: %s", type_counts)

            except psycopg.Error as integrity_error:
                logger.warning("   ⚠️ Database integrity check failed: %s", integrity_error)

            # Log discrepancy analysis if needed
            total_attempted_nodes = nodes_inserted + nodes_skipped
            total_attempted_edges = edges_inserted + edges_skipped
            if final_node_count != total_attempted_nodes:
                node_difference = total_attempted_nodes - final_node_count
                logger.info(
                    "   📝 Node count analysis: %s attempted, %s final",
                    total_attempted_nodes,
                    final_node_count,
                )
                logger.info(
                    "   🔍 Duplicate node filtering: %s duplicate nodes skipped",
                    node_difference,
                )
            if final_edge_count != total_attempted_edges:
                edge_difference = total_attempted_edges - final_edge_count
                logger.info(
                    "   📝 Edge count analysis: %s attempted, %s final",
                    total_attempted_edges,
                    final_edge_count,
                )
                logger.info(
                    "   🔍 Duplicate edge filtering: %s duplicate/invalid edges skipped",
                    edge_difference,
                )

        return True

    except (psycopg.Error, OSError, RuntimeError) as e:
        logger.error("❌ Database population failed: %s", e)
        logger.exception("Database population error details:")
        return False


async def resolve_container_files(
    container_name: str,
    file_paths: list[str] | None = None,
    tenant_id: str | None = None,
    *,
    repo: str | None = None,
) -> MaterializedFiles:
    """Resolve a container's code files to readable LOCAL paths.

    EXACT, tenant-scoped container lookup plus a materialised copy of the live
    workspace tree (read through coding-containers). The previous version
    matched containers by substring/prefix/suffix ("cv-ems" resolved to
    "cv-ems-p4") and, when ``codebase_path`` was not on THIS pod, fell back to
    ``/tmp/shared_repos``, ``/app/shared_data``, ``/shared`` or ``/app/data`` --
    answering confidently about a DIFFERENT tree. Neither exists any more: a
    missing container is 404 (403 if another tenant's), an unreadable workspace
    is 422 ``WORKSPACE_UNRESOLVABLE``, an unconfined path is 400.

    The caller must ``.release()`` the result when done.
    """
    if file_paths:
        root = await materialize_tenant_workspace(
            container_name, tenant_id, rel_paths=file_paths, repo=repo
        )
        files = [PathLib(p) for p in materialized_files(root)]
    else:
        root = await materialize_tenant_workspace(
            container_name,
            tenant_id,
            name_patterns=tuple(f"*{ext}" for ext in sorted(_CODE_EXTENSIONS)),
            repo=repo,
        )
        try:
            files = find_code_files(root)
        except BaseException:
            remove_materialized(root)
            raise
    return MaterializedFiles(files, materialized_root=root)


def find_code_files(codebase_path: PathLib) -> list[PathLib]:
    """
    Find all code files in the given codebase path.
    Returns a list of Path objects for supported file types.
    """
    logger.info("🔍 Scanning for code files in: %s", codebase_path)

    code_files = []
    files_scanned = 0
    dirs_skipped = 0

    try:
        for item in codebase_path.rglob("*"):
            # Skip if any RELATIVE path component is in skip list
            # (use relative_to so absolute path components like /tmp don't match)
            rel_parts = item.relative_to(codebase_path).parts
            if any(part in _SKIP_DIRS for part in rel_parts):
                if item.is_dir():
                    dirs_skipped += 1
                continue

            # Only process files
            if not item.is_file():
                continue

            files_scanned += 1

            # Check if file has a supported extension
            if item.suffix.lower() in _CODE_EXTENSIONS:
                # Additional checks to avoid false positives
                try:
                    # Skip very large files (> 10MB)
                    if item.stat().st_size > 10 * 1024 * 1024:
                        logger.debug("Skipping large file: %s", item)
                        continue

                    # Skip binary files (basic check)
                    try:
                        with open(item, encoding="utf-8") as f:
                            f.read(1024)  # Try to read first 1KB as text
                        code_files.append(item)
                    except (UnicodeDecodeError, UnicodeError):  # fmt: skip
                        logger.debug("Skipping binary file: %s", item)
                        continue

                except (OSError, PermissionError) as e:
                    logger.debug("Skipping inaccessible file %s: %s", item, e)
                    continue

    except (OSError, PermissionError) as e:
        logger.error("Error scanning directory %s: %s", codebase_path, e)
        raise HTTPException(status_code=500, detail=f"Failed to scan directory: {str(e)}")

    logger.info(
        "📊 Scan complete: found %d code files from %d total files (skipped %d directories)",
        len(code_files),
        files_scanned,
        dirs_skipped,
    )

    # Sort files for consistent ordering
    code_files.sort()

    # Log sample of found files for debugging
    if code_files:
        sample_size = min(5, len(code_files))
        sample_files = [str(f.relative_to(codebase_path)) for f in code_files[:sample_size]]
        logger.debug("Sample files found: %s", sample_files)
        if len(code_files) > sample_size:
            logger.debug("... and %d more files", len(code_files) - sample_size)

    return code_files


def detect_language_from_code(code: str) -> str | None:
    """
    Detect programming language from code content.
    Returns the detected language or None if detection fails.
    """
    # Simple heuristics for language detection
    code = code.strip()

    # Python indicators
    if any(keyword in code for keyword in ["def ", "import ", "from ", "class ", "if __name__"]):
        return "python"

    # JavaScript/TypeScript indicators
    has_js_keywords = any(
        keyword in code for keyword in ["function ", "const ", "let ", "var ", "=>", "console.log"]
    )
    has_ts_indicators = any(
        keyword in code for keyword in ["interface ", ": string", ": number", ": boolean"]
    )
    if has_ts_indicators:
        return "typescript"
    if has_js_keywords:
        return "javascript"

    # Java indicators
    if any(
        keyword in code
        for keyword in [
            "public class",
            "private ",
            "public static void main",
            "System.out.println",
        ]
    ):
        return "java"

    # C/C++ indicators
    if any(keyword in code for keyword in ["#include", "int main(", "printf(", "std::"]):
        if "std::" in code or "class " in code:
            return "cpp"
        return "c"

    # Go indicators
    if any(keyword in code for keyword in ["package ", "func ", "import (", "fmt.Println"]):
        return "go"

    # Rust indicators
    if any(keyword in code for keyword in ["fn ", "let mut", "println!", "use std::", "struct "]):
        return "rust"

    # Default to None if can't detect
    return None


def get_extension_for_language(language: str) -> str:
    """Get file extension for a programming language."""
    extensions = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "cpp": ".cpp",
        "c": ".c",
        "go": ".go",
        "rust": ".rs",
        "csharp": ".cs",
        "php": ".php",
        "ruby": ".rb",
        "swift": ".swift",
        "kotlin": ".kt",
        "scala": ".scala",
        "dart": ".dart",
        "r": ".r",
        "lua": ".lua",
        "perl": ".pl",
        "shell": ".sh",
        "powershell": ".ps1",
    }
    return extensions.get(language.lower(), ".txt")


def detect_language_from_file_path(file_path: str) -> str | None:
    """Detect language from file extension."""
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".m": "objectivec",
        ".r": "r",
        ".R": "r",
        ".pl": "perl",
        ".sh": "shell",
        ".bash": "shell",
        ".ps1": "powershell",
        ".lua": "lua",
        ".dart": "dart",
        ".elm": "elm",
        ".ex": "elixir",
        ".erl": "erlang",
        ".clj": "clojure",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".vim": "vim",
        ".sql": "sql",
        ".jl": "julia",
        ".nim": "nim",
        ".cr": "crystal",
        ".d": "d",
        ".pas": "pascal",
        ".ada": "ada",
        ".f90": "fortran",
        ".cob": "cobol",
        ".groovy": "groovy",
    }

    path = PathLib(file_path)
    return extension_map.get(path.suffix.lower())
