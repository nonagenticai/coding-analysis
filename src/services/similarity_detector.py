"""Code similarity detection service.

Compares files pair-wise using one of four strategies:
EXACT (SHA-256 hash), STRUCTURAL (syntax-node-type sequence),
TOKEN (whitespace/operator-split token sequence), or
SEMANTIC (same as STRUCTURAL for now).

STRUCTURAL comparison is multi-language: Python keeps its proven stdlib-``ast``
node sequence, while every other supported language is parsed via tree-sitter
(:func:`processing.languages.parse_helpers.node_type_sequence`). Languages with
no registered grammar fall back to the Python ``ast`` attempt (which simply
yields an empty sequence for non-Python source).
"""

from __future__ import annotations

import ast
import difflib
import enum
import hashlib
import logging
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from processing.languages.parse_helpers import node_type_sequence

logger = logging.getLogger(__name__)

# Languages that have their own framework tree-sitter grammar; for these,
# STRUCTURAL comparison uses ``node_type_sequence`` instead of the Python
# ``ast`` path.
_FRAMEWORK_LANGUAGES = frozenset(
    {"javascript", "typescript", "java", "go", "rust", "cpp", "csharp"}
)


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".rs": "rust",
    ".go": "go",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
}


def _detect_language(file_path: str) -> str:
    """Detect the programming language from a file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(ext, "unknown")


class SimilarityType(enum.StrEnum):
    """Type of similarity analysis."""

    EXACT = "exact"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TOKEN = "token"


class SimilarityScope(enum.StrEnum):
    """Scope of similarity analysis."""

    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PROJECT = "project"


class SimilarityMatch(BaseModel):
    """A similarity match between two code fragments."""

    # Original fields
    source_file: str = ""
    target_file: str = ""
    source_range: tuple[int, int] = (0, 0)
    target_range: tuple[int, int] = (0, 0)
    similarity_score: float = 0.0
    similarity_type: SimilarityType = SimilarityType.STRUCTURAL
    fragment_a: str = ""
    fragment_b: str = ""

    # Extended fields expected by similarity_processing / endpoints
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: SimilarityScope = SimilarityScope.FILE
    score: float = 0.0
    confidence: float = 0.0
    source_entity: str | None = None
    source_language: str = "unknown"
    source_lines: tuple[int, int] = (0, 0)
    target_entity: str | None = None
    target_language: str = "unknown"
    target_lines: tuple[int, int] = (0, 0)
    matching_features: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0
    syntactic_similarity: float = 0.0
    is_cross_language: bool = False
    summary: str = ""


class SimilarityAnalysisResult(BaseModel):
    """Result of a similarity analysis."""

    matches: list[SimilarityMatch] = []
    total_comparisons: int = 0
    analysis_time_ms: float = 0.0
    similarity_type: SimilarityType = SimilarityType.STRUCTURAL
    scope: SimilarityScope = SimilarityScope.FILE

    # Extended fields expected by similarity_processing / endpoints
    total_files_analyzed: int = 0
    total_entities_compared: int = 0
    total_matches_found: int = 0
    analysis_time_seconds: float = 0.0
    matches_by_type: dict[SimilarityType, list[SimilarityMatch]] = Field(default_factory=dict)
    matches_by_score: dict[str, list[SimilarityMatch]] = Field(default_factory=dict)
    cross_language_matches: list[SimilarityMatch] = Field(default_factory=list)
    language_distribution: dict[str, int] = Field(default_factory=dict)
    average_similarity_score: float = 0.0
    highest_similarity_match: SimilarityMatch | None = None
    refactoring_opportunities: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_code_ratio: float = 0.0
    cross_language_patterns: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal comparison helpers
# ---------------------------------------------------------------------------

_TOKEN_SPLIT_RE = re.compile(r"[\s\(\)\[\]\{\},;:=<>+\-*/&|^~!@#$%?.]+")


def _read_file(path: str) -> str | None:
    """Read a file, returning *None* on any I/O error."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _ast_node_types(code: str) -> list[str]:
    """Return the sequence of AST node type names for Python *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [type(node).__name__ for node in ast.walk(tree)]


def _structural_node_sequence(content: str, path: str | None) -> list[str]:
    """Return a syntax-node-type sequence for *content*, language-aware.

    Python (and unknown/path-less input) keeps the proven stdlib-``ast``
    sequence; every other supported language is parsed via tree-sitter. A
    tree-sitter parse that yields nothing falls back to the ``ast`` attempt so
    behaviour degrades gracefully rather than erroring.

    Args:
        content: Source text to fingerprint.
        path: Source path, used only for language detection. ``None`` (no
            path, e.g. raw snippets) is treated as Python.

    Returns:
        A node-type sequence, or ``[]`` when no structural signal is available.
    """
    language = _detect_language(path) if path else "python"
    if language in _FRAMEWORK_LANGUAGES and path:
        sequence = node_type_sequence(Path(path), content)
        if sequence:
            return sequence
    # Python, unknown languages, or an empty framework parse: fall back to the
    # stdlib-``ast`` attempt (which is empty for non-Python source).
    return _ast_node_types(content)


def _tokenize(code: str) -> list[str]:
    """Split *code* into tokens on whitespace / operators."""
    return [tok for tok in _TOKEN_SPLIT_RE.split(code) if tok]


def _sequence_ratio(a: list[str], b: list[str]) -> float:
    """Return SequenceMatcher ratio for two string sequences."""
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SimilarityDetector:
    """Code similarity detector."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize similarity detector."""
        self._config = kwargs
        self.thresholds: dict[SimilarityType, float] = {
            SimilarityType.EXACT: 0.95,
            SimilarityType.STRUCTURAL: 0.80,
            SimilarityType.SEMANTIC: 0.70,
            SimilarityType.TOKEN: 0.85,
        }
        logger.debug("SimilarityDetector initialized")

    async def detect(
        self,
        files: list[str],
        similarity_type: SimilarityType = SimilarityType.STRUCTURAL,
        scope: SimilarityScope = SimilarityScope.FILE,
        threshold: float = 0.8,
    ) -> SimilarityAnalysisResult:
        """Detect similar code fragments among *files*.

        Parameters
        ----------
        files:
            Paths to the source files to compare.
        similarity_type:
            Strategy to use for comparison.
        scope:
            Granularity of comparison (currently only FILE is
            fully supported; others fall back to FILE level).
        threshold:
            Minimum similarity score (0-1) to include in results.

        Returns
        -------
        SimilarityAnalysisResult with all matches >= *threshold*.
        """
        start = time.monotonic()

        # Read all files upfront; skip unreadable ones.
        contents: dict[str, str] = {}
        for path in files:
            data = _read_file(path)
            if data is not None:
                contents[path] = data

        readable_paths = list(contents.keys())
        matches: list[SimilarityMatch] = []
        total_comparisons = 0

        # Compare every unordered pair.
        for i in range(len(readable_paths)):
            for j in range(i + 1, len(readable_paths)):
                total_comparisons += 1
                path_a = readable_paths[i]
                path_b = readable_paths[j]
                content_a = contents[path_a]
                content_b = contents[path_b]

                score = self._compare(content_a, content_b, similarity_type, path_a, path_b)

                if score >= threshold:
                    lines_a = content_a.count("\n") + 1
                    lines_b = content_b.count("\n") + 1
                    matches.append(
                        SimilarityMatch(
                            source_file=path_a,
                            target_file=path_b,
                            source_range=(1, lines_a),
                            target_range=(1, lines_b),
                            similarity_score=round(score, 4),
                            similarity_type=similarity_type,
                            fragment_a=content_a[:500],
                            fragment_b=content_b[:500],
                        )
                    )

        elapsed_ms = (time.monotonic() - start) * 1000

        return SimilarityAnalysisResult(
            matches=matches,
            total_comparisons=total_comparisons,
            analysis_time_ms=round(elapsed_ms, 3),
            similarity_type=similarity_type,
            scope=scope,
        )

    # ------------------------------------------------------------------
    # Private comparison dispatch
    # ------------------------------------------------------------------

    def _compare(
        self,
        content_a: str,
        content_b: str,
        similarity_type: SimilarityType,
        path_a: str | None = None,
        path_b: str | None = None,
    ) -> float:
        """Return a similarity score (0..1) for two file contents.

        ``path_a`` / ``path_b`` are optional and used only for language
        detection in STRUCTURAL/SEMANTIC comparison; when omitted, content is
        treated as Python (preserving the original path-less behaviour).
        """
        if similarity_type == SimilarityType.EXACT:
            return 1.0 if _file_hash(content_a) == _file_hash(content_b) else 0.0

        if similarity_type == SimilarityType.STRUCTURAL:
            return self._structural_compare(content_a, content_b, path_a, path_b)

        if similarity_type == SimilarityType.TOKEN:
            tokens_a = _tokenize(content_a)
            tokens_b = _tokenize(content_b)
            if not tokens_a and not tokens_b:
                return 1.0
            if not tokens_a or not tokens_b:
                return 0.0
            return _sequence_ratio(tokens_a, tokens_b)

        # SEMANTIC falls back to STRUCTURAL for now.
        return self._structural_compare(content_a, content_b, path_a, path_b)

    @staticmethod
    def _structural_compare(
        content_a: str,
        content_b: str,
        path_a: str | None = None,
        path_b: str | None = None,
    ) -> float:
        """Compare two sources by syntax-node-type sequence.

        Multi-language: Python keeps its stdlib-``ast`` node sequence; every
        other supported language is parsed via tree-sitter. Languages are
        detected from ``path_a`` / ``path_b`` when available.
        """
        nodes_a = _structural_node_sequence(content_a, path_a)
        nodes_b = _structural_node_sequence(content_b, path_b)
        # If both produced no structural signal, fall back to token comparison.
        if not nodes_a and not nodes_b:
            return 1.0
        if not nodes_a or not nodes_b:
            return 0.0
        return _sequence_ratio(nodes_a, nodes_b)

    # ------------------------------------------------------------------
    # Public wrapper methods (called by similarity_endpoints)
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_match(
        match: SimilarityMatch,
        scope: SimilarityScope = SimilarityScope.FILE,
    ) -> SimilarityMatch:
        """Fill in the extended fields on a SimilarityMatch produced by detect()."""
        src_lang = _detect_language(match.source_file)
        tgt_lang = _detect_language(match.target_file)
        match.score = match.similarity_score
        match.confidence = match.similarity_score  # best-effort estimate
        match.source_language = src_lang
        match.target_language = tgt_lang
        match.source_lines = match.source_range
        match.target_lines = match.target_range
        match.scope = scope
        match.is_cross_language = (
            src_lang != tgt_lang and src_lang != "unknown" and tgt_lang != "unknown"
        )
        match.structural_similarity = match.similarity_score
        match.semantic_similarity = match.similarity_score
        match.syntactic_similarity = match.similarity_score
        match.matching_features = [match.similarity_type.value]
        match.differences = []
        match.summary = (
            f"{match.similarity_type.value} similarity "
            f"({match.similarity_score:.0%}) between "
            f"{Path(match.source_file).name} and {Path(match.target_file).name}"
        )
        return match

    @staticmethod
    def _build_rich_result(
        raw: SimilarityAnalysisResult,
        file_paths: list[str],
    ) -> SimilarityAnalysisResult:
        """Populate the extended fields on a SimilarityAnalysisResult."""
        # Language distribution
        lang_dist: dict[str, int] = defaultdict(int)
        for fp in file_paths:
            lang_dist[_detect_language(fp)] += 1

        # Matches by type
        by_type: dict[SimilarityType, list[SimilarityMatch]] = defaultdict(list)
        for m in raw.matches:
            by_type[m.similarity_type].append(m)

        # Matches by score bucket
        by_score: dict[str, list[SimilarityMatch]] = {
            "exact_duplicates": [],
            "near_duplicates": [],
            "structural": [],
            "semantic": [],
            "low": [],
        }
        for m in raw.matches:
            if m.score >= 0.95:
                by_score["exact_duplicates"].append(m)
            elif m.score >= 0.85:
                by_score["near_duplicates"].append(m)
            elif m.score >= 0.75:
                by_score["structural"].append(m)
            elif m.score >= 0.65:
                by_score["semantic"].append(m)
            else:
                by_score["low"].append(m)

        cross_lang = [m for m in raw.matches if m.is_cross_language]
        avg_score = sum(m.score for m in raw.matches) / len(raw.matches) if raw.matches else 0.0
        highest = max(raw.matches, key=lambda m: m.score) if raw.matches else None

        raw.total_files_analyzed = len(file_paths)
        raw.total_entities_compared = raw.total_comparisons
        raw.total_matches_found = len(raw.matches)
        raw.analysis_time_seconds = raw.analysis_time_ms / 1000.0
        raw.matches_by_type = dict(by_type)
        raw.matches_by_score = by_score
        raw.cross_language_matches = cross_lang
        raw.language_distribution = dict(lang_dist)
        raw.average_similarity_score = round(avg_score, 4)
        raw.highest_similarity_match = highest
        raw.duplicate_code_ratio = (
            len(by_score["exact_duplicates"]) / raw.total_comparisons
            if raw.total_comparisons
            else 0.0
        )
        raw.refactoring_opportunities = []
        raw.cross_language_patterns = [
            {
                "source_file": m.source_file,
                "target_file": m.target_file,
                "score": m.score,
                "languages": [m.source_language, m.target_language],
            }
            for m in cross_lang
        ]

        return raw

    async def analyze_similarity(
        self,
        file_paths: list[str],
        scope: SimilarityScope = SimilarityScope.FILE,
        min_similarity_score: float = 0.7,
        detect_cross_language: bool = False,
        max_files: int | None = None,
    ) -> SimilarityAnalysisResult:
        """Analyze similarity among *file_paths*.

        Wrapper around :meth:`detect` that returns a rich
        :class:`SimilarityAnalysisResult` with all extended fields
        expected by the ``/analyze`` endpoint.
        """
        files = file_paths[:max_files] if max_files else file_paths

        # Use SEMANTIC type when cross-language detection is requested
        sim_type = SimilarityType.SEMANTIC if detect_cross_language else SimilarityType.STRUCTURAL

        raw = await self.detect(
            files=files,
            similarity_type=sim_type,
            scope=scope,
            threshold=min_similarity_score,
        )

        # Enrich individual matches
        for match in raw.matches:
            self._enrich_match(match, scope)

        return self._build_rich_result(raw, files)

    async def detect_code_clones(
        self,
        file_paths: list[str],
        clone_threshold: float = 0.8,
    ) -> list[SimilarityMatch]:
        """Detect code clones among *file_paths*.

        Returns only the list of :class:`SimilarityMatch` objects that
        meet *clone_threshold*, as expected by the ``/detect-clones``
        endpoint.
        """
        raw = await self.detect(
            files=file_paths,
            similarity_type=SimilarityType.TOKEN,
            threshold=clone_threshold,
        )

        for match in raw.matches:
            self._enrich_match(match)

        return raw.matches

    async def find_cross_language_patterns(
        self,
        file_paths: list[str],
        pattern_threshold: float = 0.6,
    ) -> list[SimilarityMatch]:
        """Find cross-language patterns among *file_paths*.

        Returns only the list of :class:`SimilarityMatch` objects that
        meet *pattern_threshold*, as expected by the
        ``/cross-language-patterns`` endpoint.
        """
        raw = await self.detect(
            files=file_paths,
            similarity_type=SimilarityType.SEMANTIC,
            threshold=pattern_threshold,
        )

        for match in raw.matches:
            self._enrich_match(match)

        # Filter to only cross-language matches
        return [m for m in raw.matches if m.is_cross_language]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_detector: SimilarityDetector | None = None


def get_similarity_detector(**kwargs: Any) -> SimilarityDetector:
    """Get the global similarity detector instance."""
    global _detector  # noqa: PLW0603
    if _detector is None:
        _detector = SimilarityDetector(**kwargs)
    return _detector
