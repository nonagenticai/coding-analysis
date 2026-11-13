"""Multi-language processing metrics and parser strategy tracking.

Provides metrics classes for tracking language parser performance,
cache hit rates, and processing times across the analysis pipeline.
"""

from __future__ import annotations

import enum
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ParserStrategy(enum.StrEnum):
    """Strategy used to parse a file."""

    TREE_SITTER = "tree_sitter"
    REGEX_FALLBACK = "regex_fallback"
    EXTERNAL_TOOL = "external_tool"
    LEGACY = "legacy"
    CACHED = "cached"


@dataclass
class LanguageOperationMetrics:
    """Metrics collected during a single language operation."""

    language: str = ""
    file_path: Path | None = None
    analysis_id: str | None = None
    parse_time_ms: float = 0.0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    lines_of_code: int = 0
    errors_count: int = 0
    parser_used: ParserStrategy = ParserStrategy.LEGACY
    cache_hit: bool = False


@asynccontextmanager
async def track_language_operation(language: str, file_path: Path, analysis_id: str | None = None):
    """Context manager that yields metrics for a language parsing operation."""
    metrics = LanguageOperationMetrics(
        language=language,
        file_path=file_path,
        analysis_id=analysis_id,
    )
    yield metrics


class MultiLanguageMetricsServiceProcessingMetrics:
    """Processing metrics for parallel file analysis."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.total_files: int = 0
        self.processed_files: int = 0
        self.failed_files: int = 0
        self.processing_times: list[tuple[str, float, bool]] = []

    def start(self) -> None:
        """Record processing start time."""
        self.start_time = time.time()

    def end(self) -> None:
        """Record processing end time."""
        self.end_time = time.time()

    def add_file_time(self, file_path: str, duration: float, success: bool) -> None:
        """Record processing time for a single file."""
        self.processed_files += 1
        if not success:
            self.failed_files += 1
        self.processing_times.append((file_path, duration, success))

    def get_report(self) -> dict[str, Any]:
        """Return a summary report of all processing metrics."""
        total_duration = 0.0
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time

        avg_file_time = 0.0
        if self.processed_files > 0:
            total_file_time = sum(d for _, d, _ in self.processing_times)
            avg_file_time = total_file_time / self.processed_files

        success_rate = 0.0
        if self.processed_files > 0:
            success_rate = (self.processed_files - self.failed_files) / self.processed_files * 100

        slowest = sorted(self.processing_times, key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "avg_file_time": avg_file_time,
            "slowest_files": [{"file": f, "duration": d, "success": s} for f, d, s in slowest],
        }
