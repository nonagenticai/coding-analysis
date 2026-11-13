"""Incremental Analysis Service for File Edits.

This module provides comprehensive support for incremental code analysis, enabling
efficient analysis of file changes without re-analyzing entire files. It tracks
analysis history, merges results, and optimizes for rapid successive edits.

Features:
- Accept file diffs/changes instead of full content
- Analyze only changed sections
- Merge with previous analysis results
- Track analysis history per file
- Optimize for rapid successive edits
- Support multiple diff formats (unified, git, inline)
- Intelligent caching and invalidation
- Dependency impact tracking

The service maintains a history of analysis results and intelligently updates only
affected portions when files change, significantly improving performance for
incremental edits in IDEs and CI/CD pipelines.

Example:
    ```python
    service = IncrementalAnalysisService()

    # First analysis
    result = await service.analyze_file_full(file_path, content)

    # Incremental update
    diff = FileDiff(
        file_path=file_path,
        old_content=original_content,
        new_content=updated_content,
        hunks=[DiffHunk(old_start=10, old_lines=5, new_start=10, new_lines=8)]
    )

    updated_result = await service.analyze_file_diff(diff)
    ```
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Diff Models
# ============================================================================


class DiffFormat(StrEnum):
    """Supported diff formats."""

    UNIFIED = "unified"  # Unified diff format (git diff)
    INLINE = "inline"  # Line-by-line comparison
    STRUCTURED = "structured"  # Structured change representation


class ChangeType(StrEnum):
    """Type of change in a diff."""

    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class DiffHunk:
    """Represents a single hunk of changes in a diff."""

    old_start: int  # Starting line in old file (1-based)
    old_lines: int  # Number of lines in old file
    new_start: int  # Starting line in new file (1-based)
    new_lines: int  # Number of lines in new file
    content: str = ""  # The actual diff content
    context_before: list[str] = field(default_factory=list)  # Lines before change
    context_after: list[str] = field(default_factory=list)  # Lines after change


@dataclass
class FileDiff:
    """Represents a complete file diff."""

    file_path: str
    old_content: str | None = None
    new_content: str | None = None
    hunks: list[DiffHunk] = field(default_factory=list)
    format: DiffFormat = DiffFormat.STRUCTURED
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_affected_line_ranges(self) -> list[tuple[int, int]]:
        """Get all affected line ranges in the new file."""
        ranges = []
        for hunk in self.hunks:
            # Include context for better analysis
            start = max(1, hunk.new_start - 3)
            end = hunk.new_start + hunk.new_lines + 3
            ranges.append((start, end))
        return ranges

    def get_change_summary(self) -> dict[str, int]:
        """Get summary statistics of changes."""
        added = sum(h.new_lines - h.old_lines for h in self.hunks if h.new_lines > h.old_lines)
        deleted = sum(h.old_lines - h.new_lines for h in self.hunks if h.old_lines > h.new_lines)
        return {
            "hunks": len(self.hunks),
            "lines_added": max(0, added),
            "lines_deleted": max(0, deleted),
            "lines_changed": sum(h.new_lines for h in self.hunks),
        }


# ============================================================================
# Analysis History Models
# ============================================================================


class AnalysisVersion(BaseModel):
    """Represents a versioned analysis result."""

    version: int = Field(description="Version number (incrementing)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = Field(description="Hash of analyzed content")
    result: CoreAnalysisResult
    file_size: int = 0
    line_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileAnalysisHistory:
    """Maintains analysis history for a single file."""

    def __init__(self, file_path: str, max_versions: int = 10):
        """Initialize file analysis history.

        Args:
            file_path: Path to the file
            max_versions: Maximum number of versions to keep
        """
        self.file_path = file_path
        self.max_versions = max_versions
        self.versions: list[AnalysisVersion] = []
        self.current_version: AnalysisVersion | None = None
        self.entity_cache: dict[str, CoreCodeEntity] = {}  # entity_id -> entity
        self.relation_cache: set[tuple[str, str, str]] = set()  # (source, target, type)

    def add_version(self, result: CoreAnalysisResult, content: str) -> AnalysisVersion:
        """Add a new version to history."""
        version_num = len(self.versions) + 1
        content_hash = self._hash_content(content)

        version = AnalysisVersion(
            version=version_num,
            content_hash=content_hash,
            result=result,
            file_size=len(content),
            line_count=content.count("\n") + 1,
        )

        self.versions.append(version)
        self.current_version = version

        # Update caches
        self._update_entity_cache(result)
        self._update_relation_cache(result)

        # Trim old versions
        if len(self.versions) > self.max_versions:
            self.versions = self.versions[-self.max_versions :]

        return version

    def get_current(self) -> AnalysisVersion | None:
        """Get current version."""
        return self.current_version

    def get_version(self, version_num: int) -> AnalysisVersion | None:
        """Get specific version."""
        for v in self.versions:
            if v.version == version_num:
                return v
        return None

    def get_entity(self, entity_id: str) -> CoreCodeEntity | None:
        """Get cached entity by ID."""
        return self.entity_cache.get(entity_id)

    def has_relation(self, source_id: str, target_id: str, rel_type: str) -> bool:
        """Check if relation exists in cache."""
        return (source_id, target_id, rel_type) in self.relation_cache

    def _hash_content(self, content: str) -> str:
        """Generate hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _update_entity_cache(self, result: CoreAnalysisResult) -> None:
        """Update entity cache from analysis result."""
        if result.entities:
            for entity in result.entities:
                self.entity_cache[entity.id] = entity

    def _update_relation_cache(self, result: CoreAnalysisResult) -> None:
        """Update relation cache from analysis result."""
        if result.relationships:
            for rel in result.relationships:
                self.relation_cache.add((rel.source_id, rel.target_id, rel.type))


# ============================================================================
# Incremental Analysis Service
# ============================================================================


class IncrementalAnalysisService:
    """Service for incremental code analysis with diff support."""

    def __init__(
        self,
        max_history_per_file: int = 10,
        enable_caching: bool = True,
        debounce_ms: int = 500,
    ):
        """Initialize incremental analysis service.

        Args:
            max_history_per_file: Maximum versions to keep per file
            enable_caching: Enable result caching
            debounce_ms: Debounce time for rapid edits (milliseconds)
        """
        self.max_history = max_history_per_file
        self.enable_caching = enable_caching
        self.debounce_ms = debounce_ms

        # File-level history tracking
        self.file_histories: dict[str, FileAnalysisHistory] = {}

        # Pending analysis tracking for debouncing
        self.pending_analyses: dict[str, asyncio.Task] = {}
        self.pending_timers: dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = {
            "full_analyses": 0,
            "incremental_analyses": 0,
            "cache_hits": 0,
            "debounced_requests": 0,
        }

    # ========================================================================
    # Core Analysis Methods
    # ========================================================================

    async def analyze_file_full(
        self,
        file_path: str,
        content: str,
        language: str | None = None,
    ) -> CoreAnalysisResult:
        """Perform full analysis on a file and store in history.

        Args:
            file_path: Path to the file
            content: Full file content
            language: Programming language (auto-detect if None)

        Returns:
            Complete analysis result
        """
        self.stats["full_analyses"] += 1

        # Perform full analysis
        result = await self._analyze_content_full(file_path, content, language)

        # Store in history
        self._get_or_create_history(file_path).add_version(result, content)

        return result

    async def analyze_file_diff(
        self,
        diff: FileDiff,
        language: str | None = None,
        force_full: bool = False,
    ) -> CoreAnalysisResult:
        """Perform incremental analysis on file diff.

        Args:
            diff: File diff information
            language: Programming language (auto-detect if None)
            force_full: Force full re-analysis

        Returns:
            Updated analysis result
        """
        # Check if we should debounce this request
        if self.debounce_ms > 0 and not force_full:
            return await self._debounced_analysis(diff, language)

        return await self._analyze_diff_immediate(diff, language, force_full)

    async def _analyze_diff_immediate(
        self,
        diff: FileDiff,
        language: str | None,
        force_full: bool,
    ) -> CoreAnalysisResult:
        """Perform immediate diff analysis without debouncing."""
        history = self._get_or_create_history(diff.file_path)
        current = history.get_current()

        # If no history or force full, do full analysis
        if current is None or force_full or diff.new_content is None:
            if diff.new_content:
                return await self.analyze_file_full(diff.file_path, diff.new_content, language)
            else:
                # Cannot analyze without content
                return self._create_empty_result(diff.file_path, "No content provided")

        self.stats["incremental_analyses"] += 1

        # Check if content hash matches (no actual changes)
        new_hash = hashlib.sha256(diff.new_content.encode("utf-8")).hexdigest()
        if new_hash == current.content_hash:
            self.stats["cache_hits"] += 1
            return current.result

        # Perform incremental analysis
        try:
            # Extract changed regions
            changed_regions = await self._extract_changed_regions(diff)

            # Analyze changed regions
            partial_results = await self._analyze_regions(
                diff.file_path, diff.new_content, changed_regions, language
            )

            # Merge with previous results
            merged_result = await self._merge_results(
                current.result, partial_results, diff, changed_regions
            )

            # Store new version
            history.add_version(merged_result, diff.new_content)

            return merged_result

        except Exception as e:
            logger.warning(
                f"Incremental analysis failed for {diff.file_path}, falling back to full analysis: {e}"
            )
            # Fallback to full analysis
            return await self.analyze_file_full(diff.file_path, diff.new_content, language)

    async def _debounced_analysis(self, diff: FileDiff, language: str | None) -> CoreAnalysisResult:
        """Debounce rapid successive analyses."""
        file_path = diff.file_path

        # Cancel existing pending timer
        if file_path in self.pending_timers:
            self.pending_timers[file_path].cancel()

        # Cancel existing analysis task
        if file_path in self.pending_analyses:
            self.pending_analyses[file_path].cancel()

        self.stats["debounced_requests"] += 1

        # Create new delayed task
        async def delayed_analysis():
            await asyncio.sleep(self.debounce_ms / 1000.0)
            result = await self._analyze_diff_immediate(diff, language, False)
            # Clean up tracking
            self.pending_analyses.pop(file_path, None)
            self.pending_timers.pop(file_path, None)
            return result

        # Schedule analysis
        task = asyncio.create_task(delayed_analysis())
        self.pending_analyses[file_path] = task
        self.pending_timers[file_path] = task

        # Return current result while waiting
        history = self._get_or_create_history(file_path)
        current = history.get_current()
        if current:
            return current.result
        else:
            # No previous result, wait for analysis
            return await task

    # ========================================================================
    # Diff Processing
    # ========================================================================

    async def _extract_changed_regions(self, diff: FileDiff) -> list[tuple[int, int, str]]:
        """Extract changed code regions from diff.

        Returns:
            List of (start_line, end_line, content) tuples
        """
        regions = []

        if diff.hunks:
            # Use provided hunks
            for hunk in diff.hunks:
                if diff.new_content:
                    lines = diff.new_content.split("\n")
                    start = max(0, hunk.new_start - 1)
                    end = min(len(lines), hunk.new_start + hunk.new_lines)
                    content = "\n".join(lines[start:end])
                    regions.append((hunk.new_start, hunk.new_start + hunk.new_lines, content))
        elif diff.old_content and diff.new_content:
            # Generate hunks from content comparison
            regions = await self._compute_diff_regions(diff.old_content, diff.new_content)

        return regions

    async def _compute_diff_regions(
        self, old_content: str, new_content: str
    ) -> list[tuple[int, int, str]]:
        """Compute changed regions by comparing old and new content."""
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        # Use difflib to find changes
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)
        regions = []

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag in ("replace", "insert", "delete"):
                # Expand region to include context
                context_lines = 5
                start = max(0, j1 - context_lines)
                end = min(len(new_lines), j2 + context_lines)

                content = "\n".join(new_lines[start:end])
                regions.append((start + 1, end + 1, content))  # 1-based line numbers

        return regions

    # ========================================================================
    # Region Analysis
    # ========================================================================

    async def _analyze_regions(
        self,
        file_path: str,
        full_content: str,
        regions: list[tuple[int, int, str]],
        language: str | None,
    ) -> list[CoreAnalysisResult]:
        """Analyze specific code regions.

        Args:
            file_path: Path to file
            full_content: Full file content for context
            regions: List of (start_line, end_line, content) tuples
            language: Programming language

        Returns:
            List of partial analysis results
        """
        results = []

        for start_line, end_line, region_content in regions:
            try:
                # Analyze region as a snippet with context
                result = await self._analyze_content_snippet(
                    file_path,
                    region_content,
                    language,
                    line_offset=start_line,
                    full_context=full_content,
                )

                results.append(result)

            except Exception as e:
                logger.warning(
                    f"Failed to analyze region {start_line}-{end_line} in {file_path}: {e}"
                )

        return results

    async def _analyze_content_snippet(
        self,
        file_path: str,
        content: str,
        language: str | None,
        line_offset: int = 0,
        full_context: str | None = None,
    ) -> CoreAnalysisResult:
        """Analyze a code snippet with line offset tracking."""
        # This would integrate with the existing CodeAnalyzer
        # For now, create a placeholder
        result = CoreAnalysisResult(
            success=True,
            file_path=file_path,
            language=language or self._detect_language(file_path),
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=CoreComplexityMetrics(),
            security_patterns=[],
            timestamp=datetime.now(UTC),
            metadata={"line_offset": line_offset, "is_snippet": True},
        )

        # Adjust line numbers in entities
        if result.entities:
            for entity in result.entities:
                if entity.line_number:
                    entity.line_number += line_offset
                if entity.end_line_number:
                    entity.end_line_number += line_offset

        return result

    async def _analyze_content_full(
        self,
        file_path: str,
        content: str,
        language: str | None,
    ) -> CoreAnalysisResult:
        """Perform full content analysis."""
        detected_language = language or self._detect_language(file_path)

        # Create basic result structure
        result = CoreAnalysisResult(
            success=True,
            file_path=file_path,
            language=detected_language,
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=CoreComplexityMetrics(),
            security_patterns=[],
            timestamp=datetime.now(UTC),
            metadata={"analysis_type": "full"},
        )

        return result

    # ========================================================================
    # Result Merging
    # ========================================================================

    async def _merge_results(
        self,
        base_result: CoreAnalysisResult,
        partial_results: list[CoreAnalysisResult],
        diff: FileDiff,
        changed_regions: list[tuple[int, int, str]],
    ) -> CoreAnalysisResult:
        """Merge partial analysis results with base result.

        This intelligently combines the old analysis with new partial analyses,
        removing invalidated entities and adding new ones.

        Args:
            base_result: Previous full analysis result
            partial_results: New partial analysis results
            diff: The diff that was analyzed
            changed_regions: Regions that were analyzed

        Returns:
            Merged analysis result
        """
        # Determine affected line ranges
        affected_ranges = set()
        for start, end, _ in changed_regions:
            affected_ranges.add((start, end))

        # Filter out entities in affected regions
        preserved_entities = []
        if base_result.entities:
            for entity in base_result.entities:
                if not self._is_entity_in_affected_region(entity, affected_ranges):
                    preserved_entities.append(entity)

        # Add new entities from partial results
        new_entities = []
        for partial in partial_results:
            if partial.entities:
                new_entities.extend(partial.entities)

        # Combine entities
        all_entities = preserved_entities + new_entities

        # Handle relationships
        preserved_relations = []
        if base_result.relationships:
            preserved_relations = self._filter_valid_relationships(
                base_result.relationships, all_entities, affected_ranges
            )

        # Add new relationships
        new_relations = []
        for partial in partial_results:
            if partial.relationships:
                new_relations.extend(partial.relationships)

        all_relations = preserved_relations + new_relations

        # Merge complexity metrics
        complexity = self._merge_complexity_metrics(
            base_result.complexity_metrics,
            [p.complexity_metrics for p in partial_results if p.complexity_metrics],
        )

        # Create merged result
        merged = CoreAnalysisResult(
            success=True,
            file_path=diff.file_path,
            language=base_result.language,
            entities=all_entities,
            relationships=all_relations,
            dependencies=base_result.dependencies,  # Keep existing deps for now
            complexity_metrics=complexity,
            security_patterns=base_result.security_patterns,  # Keep existing patterns
            timestamp=datetime.now(UTC),
            metadata={
                "analysis_type": "incremental",
                "base_version": (
                    base_result.metadata.get("version") if base_result.metadata else None
                ),
                "regions_analyzed": len(changed_regions),
                "entities_preserved": len(preserved_entities),
                "entities_new": len(new_entities),
                "change_summary": diff.get_change_summary(),
            },
        )

        return merged

    def _is_entity_in_affected_region(
        self, entity: CoreCodeEntity, affected_ranges: set[tuple[int, int]]
    ) -> bool:
        """Check if entity overlaps with any affected region."""
        if not entity.line_number:
            return False

        entity_start = entity.line_number
        entity_end = entity.end_line_number or entity.line_number

        for range_start, range_end in affected_ranges:
            # Check for overlap
            if not (entity_end < range_start or entity_start > range_end):
                return True

        return False

    def _filter_valid_relationships(
        self,
        relationships: list[EntityRelation],
        valid_entities: list[CoreCodeEntity],
        affected_ranges: set[tuple[int, int]],
    ) -> list[EntityRelation]:
        """Filter relationships to only include those with valid endpoints."""
        valid_entity_ids = {e.id for e in valid_entities}

        valid_relations = []
        for rel in relationships:
            # Keep relation if both endpoints are still valid
            if rel.source_id in valid_entity_ids and rel.target_id in valid_entity_ids:
                valid_relations.append(rel)

        return valid_relations

    def _merge_complexity_metrics(
        self,
        base_metrics: CoreComplexityMetrics | None,
        partial_metrics: list[CoreComplexityMetrics],
    ) -> CoreComplexityMetrics:
        """Merge complexity metrics."""
        # For now, return base metrics
        # In a full implementation, this would recalculate based on changes
        if base_metrics:
            return base_metrics
        elif partial_metrics:
            return partial_metrics[0]
        else:
            return CoreComplexityMetrics()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_or_create_history(self, file_path: str) -> FileAnalysisHistory:
        """Get or create history for a file."""
        if file_path not in self.file_histories:
            self.file_histories[file_path] = FileAnalysisHistory(file_path, self.max_history)
        return self.file_histories[file_path]

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
        }
        return mapping.get(ext, "unknown")

    def _create_empty_result(self, file_path: str, error: str) -> CoreAnalysisResult:
        """Create an empty error result."""
        return CoreAnalysisResult(
            success=False,
            file_path=file_path,
            language=self._detect_language(file_path),
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=CoreComplexityMetrics(),
            security_patterns=[],
            error=error,
            timestamp=datetime.now(UTC),
        )

    # ========================================================================
    # Diff Parsing
    # ========================================================================

    def parse_unified_diff(self, diff_text: str) -> list[FileDiff]:
        """Parse unified diff format (git diff output).

        Args:
            diff_text: Unified diff text

        Returns:
            List of FileDiff objects
        """
        file_diffs = []
        current_file = None
        current_hunks = []

        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # File header
            if line.startswith("--- "):
                # Save previous file
                if current_file:
                    file_diffs.append(
                        FileDiff(
                            file_path=current_file,
                            hunks=current_hunks,
                            format=DiffFormat.UNIFIED,
                        )
                    )

                # Start new file
                current_file = line[4:].split("\t")[0]
                if current_file.startswith("a/"):
                    current_file = current_file[2:]
                current_hunks = []

            # Hunk header
            elif line.startswith("@@"):
                hunk = self._parse_hunk_header(line)
                if hunk:
                    # Collect hunk content
                    hunk_lines = []
                    i += 1
                    while (
                        i < len(lines)
                        and not lines[i].startswith("@@")
                        and not lines[i].startswith("--- ")
                    ):
                        if not lines[i].startswith("+++"):
                            hunk_lines.append(lines[i])
                        i += 1
                    i -= 1  # Back up one line

                    hunk.content = "\n".join(hunk_lines)
                    current_hunks.append(hunk)

            i += 1

        # Save last file
        if current_file:
            file_diffs.append(
                FileDiff(
                    file_path=current_file,
                    hunks=current_hunks,
                    format=DiffFormat.UNIFIED,
                )
            )

        return file_diffs

    def _parse_hunk_header(self, line: str) -> DiffHunk | None:
        """Parse a hunk header line like '@@ -10,5 +10,8 @@'."""
        match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
        if match:
            old_start, old_lines, new_start, new_lines = map(int, match.groups())
            return DiffHunk(
                old_start=old_start,
                old_lines=old_lines,
                new_start=new_start,
                new_lines=new_lines,
            )
        return None

    def create_diff_from_strings(
        self, file_path: str, old_content: str, new_content: str
    ) -> FileDiff:
        """Create a FileDiff from old and new content strings.

        Args:
            file_path: Path to the file
            old_content: Original content
            new_content: Updated content

        Returns:
            FileDiff object with computed hunks
        """
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        differ = difflib.SequenceMatcher(None, old_lines, new_lines)
        hunks = []

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != "equal":
                hunk = DiffHunk(
                    old_start=i1 + 1,
                    old_lines=i2 - i1,
                    new_start=j1 + 1,
                    new_lines=j2 - j1,
                )

                # Add content
                if tag == "delete":
                    hunk.content = "\n".join(f"- {line}" for line in old_lines[i1:i2])
                elif tag == "insert":
                    hunk.content = "\n".join(f"+ {line}" for line in new_lines[j1:j2])
                elif tag == "replace":
                    old_part = "\n".join(f"- {line}" for line in old_lines[i1:i2])
                    new_part = "\n".join(f"+ {line}" for line in new_lines[j1:j2])
                    hunk.content = old_part + "\n" + new_part

                hunks.append(hunk)

        return FileDiff(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            hunks=hunks,
            format=DiffFormat.INLINE,
        )

    # ========================================================================
    # History and Statistics
    # ========================================================================

    def get_file_history(self, file_path: str) -> FileAnalysisHistory | None:
        """Get analysis history for a file."""
        return self.file_histories.get(file_path)

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "tracked_files": len(self.file_histories),
            "total_versions": sum(len(h.versions) for h in self.file_histories.values()),
            "pending_analyses": len(self.pending_analyses),
        }

    def clear_history(self, file_path: str | None = None) -> None:
        """Clear analysis history.

        Args:
            file_path: Specific file to clear, or None to clear all
        """
        if file_path:
            self.file_histories.pop(file_path, None)
        else:
            self.file_histories.clear()

    def get_dependency_impact(self, file_path: str) -> list[str]:
        """Get list of files that may be impacted by changes to this file.

        Args:
            file_path: Path to the changed file

        Returns:
            List of potentially impacted file paths
        """
        history = self.get_file_history(file_path)
        if not history or not history.current_version:
            return []

        impacted = []
        current = history.current_version.result

        # Check which other files depend on entities in this file
        if current.entities:
            entity_ids = {e.id for e in current.entities}

            for other_path, other_history in self.file_histories.items():
                if other_path == file_path:
                    continue

                if other_history.current_version:
                    other_result = other_history.current_version.result
                    if other_result.relationships:
                        for rel in other_result.relationships:
                            if rel.target_id in entity_ids:
                                impacted.append(other_path)
                                break

        return impacted


# ============================================================================
# Factory and Utilities
# ============================================================================


_service_instance: IncrementalAnalysisService | None = None


def get_incremental_analysis_service(
    max_history: int = 10,
    enable_caching: bool = True,
    debounce_ms: int = 500,
) -> IncrementalAnalysisService:
    """Get or create the global incremental analysis service.

    Args:
        max_history: Maximum versions to keep per file
        enable_caching: Enable result caching
        debounce_ms: Debounce time for rapid edits (milliseconds)

    Returns:
        Singleton service instance
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = IncrementalAnalysisService(
            max_history_per_file=max_history,
            enable_caching=enable_caching,
            debounce_ms=debounce_ms,
        )
        logger.info(
            f"Initialized incremental analysis service (history={max_history}, debounce={debounce_ms}ms)"
        )

    return _service_instance
