"""Multi-language code-graph extraction for the deep analyzer.

Drives the per-language ``LanguageProcessor`` framework
(``processing.languages``) over a polyglot source tree and merges every
language's entities/relationships into a single graph. Used by
``DeepCodeAnalyzer.analyze_graph_multilang`` so the architecture graph can be
built for any language the platform supports — not just Python.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast

# EXCLUDED_DIRS (directories never worth walking into) is ALSO the bug hunt's
# declared scope policy. Its ONE definition lives in the self-contained hunt
# agent, which runs inside the workspace; this walk uses it from there so the
# two cannot disagree.
from hunt_agent.hunt import EXCLUDED_DIRS, is_in_excluded_dir
from models.core_models import CoreCodeEntity, EntityRelation
from processing.languages import language_registry
from processing.languages.registry import SupportedLanguage

logger = logging.getLogger(__name__)

# Bound how many files are parsed concurrently so a large monorepo does not
# open thousands of file handles / parser instances at once.
_MAX_CONCURRENT_PARSES = 16


def is_supported_source_file(path: Path) -> bool:
    """True when ``path``'s extension maps to a registered, detectable language.

    The definition of "a source file" for the walk below. The bug hunt uses the
    agent's own table (``hunt_agent.hunt.SOURCE_EXTENSIONS``), kept equal to
    this registry by ``tests/unit/test_hunt_agent.py``. Directory scope is a
    separate predicate (:func:`is_in_excluded_dir`).
    """
    if path.suffix.lower() not in language_registry.get_supported_extensions():
        return False
    return language_registry.detect_language(path) is not None


def collect_source_files(root: Path, excluded_dirs: frozenset[str] | None = None) -> list[Path]:
    """Collect every source file under ``root`` in a supported language.

    Walks ``root`` once, keeping files whose extension maps to a registered
    language. Skips VCS / dependency / build directories and files larger than
    the language's configured ``max_file_size_mb``.

    Args:
        root: Repository root to scan.
        excluded_dirs: Directory names to skip; ``None`` means :data:`EXCLUDED_DIRS`.

    Returns:
        Sorted list of source file paths in a supported language.
    """
    if not root.exists():
        return []
    excluded = EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_in_excluded_dir(path.relative_to(root), excluded):
            continue
        if not is_supported_source_file(path):
            continue
        language = language_registry.detect_language(path)
        if language is None:
            continue
        try:
            max_mb = language_registry.get_language_config(language).max_file_size_mb
            if path.stat().st_size > max_mb * 1024 * 1024:
                logger.debug("Skipping oversized file %s", path)
                continue
        except (OSError, ValueError):
            continue
        files.append(path)
    return sorted(files)


async def extract_graph_multilang(
    codebase_path: Path,
    source_files: list[Path],
) -> tuple[list[CoreCodeEntity], list[EntityRelation]]:
    """Extract a merged code graph across all supported languages.

    Groups ``source_files`` by detected language, parses each with its
    registered ``LanguageProcessor``, and merges every language's entities and
    relationships into a single ``(entities, relationships)`` pair so polyglot
    repositories produce one architecture graph. A failure on a single file is
    logged and skipped — it never aborts the repository.

    Args:
        codebase_path: Repository root; entity ``file_path`` values are made
            relative to it for parity with the Python analysis path.
        source_files: Files to parse (typically from ``collect_source_files``).

    Returns:
        A tuple of ``(entities, relationships)`` merged across all languages.
    """
    by_language: dict[str, list[Path]] = {}
    for path in source_files:
        language = language_registry.detect_language(path)
        if language is None:
            continue
        by_language.setdefault(language, []).append(path)

    entities: list[CoreCodeEntity] = []
    relationships: list[EntityRelation] = []
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PARSES)

    def _relativize(value: str) -> str:
        """Make an entity file path relative to the codebase root, if possible."""
        try:
            return str(Path(value).relative_to(codebase_path))
        except ValueError:
            return value

    async def _parse_one(
        processor: object, path: Path
    ) -> tuple[list[CoreCodeEntity], list[EntityRelation]]:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("⚠️  Could not read %s: %s", path, exc)
            return [], []
        async with semaphore:
            try:
                # The absolute path is passed through: some processors re-read
                # the file from disk rather than using ``content``.
                # ``processor`` is a LanguageProcessor; typed loosely here only
                # to keep the inner helper signature simple.
                result = await processor.parse_file(path, content)  # type: ignore[attr-defined]
            except (OSError, UnicodeDecodeError, RuntimeError, ValueError) as exc:
                logger.warning("⚠️  Failed to parse %s: %s", path, exc)
                return [], []
        if not result.success:
            logger.debug("Parser reported failure for %s: %s", path, result.errors)
        # Normalise entity file paths to be relative to the codebase root, for
        # parity with the Python analysis path.
        for entity in result.entities:
            entity.file_path = _relativize(entity.file_path)
        return result.entities, result.relationships

    for language, paths in by_language.items():
        try:
            processor = await language_registry.get_processor(cast("SupportedLanguage", language))
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "⚠️  No usable processor for %s (%d files skipped): %s",
                language,
                len(paths),
                exc,
            )
            continue
        results = await asyncio.gather(*(_parse_one(processor, path) for path in paths))
        for file_entities, file_relationships in results:
            entities.extend(file_entities)
            relationships.extend(file_relationships)
        logger.info("Multi-language graph: %s → %d files parsed", language, len(paths))

    logger.info(
        "✅ Multi-language extraction complete: %d entities, %d relationships across %d languages",
        len(entities),
        len(relationships),
        len(by_language),
    )
    return entities, relationships
