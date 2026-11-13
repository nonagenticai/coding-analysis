"""Shared multi-language source-parsing helpers.

Thin wrappers over the ``processing.languages`` framework, used by the
repo-scanner, bug-hunter and similarity analyzers so none of them has to
reach for the stdlib ``ast`` module or ``*.py`` globs directly:

- :func:`parse_source` — entities + relationships for a single file in any
  supported language, with per-file failure isolation.
- :func:`node_type_sequence` — a flat depth-first syntax-node-type sequence
  for structural similarity comparison, uniform across every language.

Both return an "empty" result (``None`` / ``[]``) rather than raising, so a
single unparseable file never aborts a batch.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import cast

from hunt_agent.hunt import get_parser
from processing.languages import language_registry
from processing.languages.base import LanguageProcessor, ParseResult
from processing.languages.registry import SupportedLanguage

logger = logging.getLogger(__name__)


def _get_ts_parser(grammar: str) -> object | None:
    """Return a cached tree-sitter ``Parser`` for *grammar*, or ``None``.

    The grammar table and parser cache are the hunt agent's
    (``hunt_agent.hunt.GRAMMARS`` / ``get_parser``): the bug hunt parses inside
    the workspace with that code, so the service parses with the same code.

    Args:
        grammar: Tree-sitter grammar name (``LanguageConfig.tree_sitter_grammar``).

    Returns:
        A ready ``tree_sitter.Parser``, or ``None`` if the grammar module or
        the ``tree_sitter`` package is unavailable.
    """
    parser = get_parser(grammar)
    if parser is None:
        logger.debug("tree-sitter parser unavailable for grammar %s", grammar)
    return parser


def parse_tree(path: Path, content: str) -> object | None:
    """Parse *content* into a tree-sitter ``Tree``, detecting language by extension.

    A lightweight, synchronous alternative to :func:`parse_source` for callers
    that want the raw syntax tree (structural fingerprints, syntax-error
    detection, node-span analysis) rather than extracted entities.

    Args:
        path: Source file path — only its extension is used for detection.
        content: Source text to parse.

    Returns:
        A ``tree_sitter.Tree``, or ``None`` when the language is unknown, the
        grammar is unavailable, or parsing fails.
    """
    language = language_registry.detect_language(path)
    if language is None:
        return None
    try:
        grammar = language_registry.get_language_config(language).tree_sitter_grammar
    except ValueError:
        return None
    if not grammar:
        return None

    parser = _get_ts_parser(grammar)
    if parser is None:
        return None

    try:
        return parser.parse(content.encode("utf-8"))  # type: ignore[attr-defined]
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.warning("⚠️  tree-sitter parse failed for %s: %s", path, exc)
        return None


def node_type_sequence(path: Path, content: str) -> list[str]:
    """Return a flat depth-first sequence of syntax-node types for *content*.

    The language is detected from *path*'s extension and parsed with its
    tree-sitter grammar, giving a structural fingerprint that is directly
    comparable across languages (e.g. via ``difflib.SequenceMatcher``).

    Args:
        path: Source file path — only its extension is used for detection.
        content: Source text to parse.

    Returns:
        Pre-order list of tree-sitter node-type names. Empty when the language
        is unknown, the grammar is unavailable, or parsing fails — callers
        treat ``[]`` as "no structural signal", never as an error.
    """
    tree = parse_tree(path, content)
    if tree is None:
        return []

    types: list[str] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        types.append(node.type)
        # Push children reversed so they are visited left-to-right (pre-order).
        stack.extend(reversed(node.children))
    return types


async def _run_processor(
    processor: LanguageProcessor,
    path: Path,
    content: str,
    rewrite_to: str | None,
) -> ParseResult | None:
    """Run ``processor.parse_file`` with failure isolation.

    Args:
        processor: Language processor to drive.
        path: On-disk path handed to the processor (some re-read it from disk).
        content: Source text.
        rewrite_to: If set, every entity's ``file_path`` (and the result's) is
            rewritten to this value — used when *path* is a temporary file.

    Returns:
        The :class:`ParseResult`, or ``None`` if parsing raised.
    """
    try:
        result = await processor.parse_file(path, content)
    except (OSError, UnicodeDecodeError, RuntimeError, ValueError) as exc:
        logger.warning("⚠️  Failed to parse %s: %s", path, exc)
        return None
    if rewrite_to is not None:
        for entity in result.entities:
            entity.file_path = rewrite_to
        result.file_path = rewrite_to
    return result


async def parse_source(path: Path, content: str) -> ParseResult | None:
    """Parse a single source file into entities + relationships, any language.

    Detects the language from *path*'s extension and dispatches to the
    registered ``LanguageProcessor``. Some processors (Go, C++, C#) re-read
    the file from disk rather than using *content*; when *path* does not exist
    on disk the content is materialised into a temporary file so those
    processors still work, and entity paths are rewritten back to *path*.

    Args:
        path: Source file path — extension drives language detection.
        content: Source text to parse.

    Returns:
        A :class:`ParseResult`, or ``None`` for an unknown extension, a missing
        processor, or a parse failure. Never raises for a single bad file.
    """
    language = language_registry.detect_language(path)
    if language is None:
        return None
    try:
        processor = await language_registry.get_processor(cast("SupportedLanguage", language))
    except (ValueError, RuntimeError) as exc:
        logger.warning("⚠️  No usable processor for %s: %s", language, exc)
        return None

    if path.is_file():
        return await _run_processor(processor, path, content, rewrite_to=None)

    # Path is not on disk (e.g. content pulled out of a container). Materialise
    # it so processors that re-read from disk can find the file.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=path.suffix or ".txt", encoding="utf-8", delete=False
        ) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        return await _run_processor(processor, tmp_path, content, rewrite_to=str(path))
    except OSError as exc:
        logger.warning("⚠️  Could not materialise %s for parsing: %s", path, exc)
        return None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
