"""Parallel file processing for multi-language code analysis.

This module provides high-performance parallel processing capabilities for analyzing
multiple source code files concurrently. It supports multi-language processing,
automatic language detection, and comprehensive performance metrics tracking.

The parallel processor uses asyncio for concurrent processing, automatically
balancing workload based on available CPU cores while providing detailed metrics
on processing performance, including file processing times and success rates.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import (
    Any,
    TypeVar,
)

from app.core.config.core_config_settings import get_settings
from app.core.file_service import FileService
from processing.processor_factory import ProcessorFactory
from services.multi_language_metrics import (
    MultiLanguageMetricsServiceProcessingMetrics,
)

logger = logging.getLogger(__name__)

# Type variable for result type
T = TypeVar("T")


class ProcessingMetrics:
    """Metrics for processing performance monitoring."""

    def __init__(self) -> None:
        """Initialize processing metrics."""
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.processing_times: list[tuple[str, float, bool]] = []

    def start(self) -> None:
        """Start processing and record start time."""
        self.start_time = time.time()

    def end(self) -> None:
        """End processing and record end time."""
        self.end_time = time.time()

    def add_file_time(self, file_path: str, duration: float, success: bool) -> None:
        """Add processing time for a file.

        Args:
            file_path: Path to the processed file
            duration: Processing duration in seconds
            success: Whether processing was successful
        """
        self.processed_files += 1
        if not success:
            self.failed_files += 1

        self.processing_times.append((file_path, duration, success))

    def get_report(self) -> dict[str, Any]:
        """Get a report of processing metrics.

        Returns:
            Dictionary with processing metrics
        """
        total_duration: float = 0.0
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time

        avg_file_time: float = 0.0
        if self.processed_files > 0:
            total_file_time = sum(duration for _, duration, _ in self.processing_times)
            avg_file_time = total_file_time / self.processed_files

        success_rate = 0.0
        if self.processed_files > 0:
            success_rate = (self.processed_files - self.failed_files) / self.processed_files * 100

        # Find slowest files
        slowest_files = sorted(
            self.processing_times,
            key=lambda x: x[1],
            reverse=True,  # Sort by duration
        )[:5]  # Top 5 slowest

        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "avg_file_time": avg_file_time,
            "slowest_files": [
                {"file": f, "duration": d, "success": s} for f, d, s in slowest_files
            ],
        }


class ParallelProcessor:
    """Processor for parallel processing of files."""

    def __init__(self, max_workers: int | None = None):
        """Initialize the parallel processor.

        Args:
            max_workers: Maximum number of worker processes/threads
        """
        get_settings()

        # Use provided max_workers or calculate based on CPU cores
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)  # Limit to 8 by default

        # Create processor factory for file processing
        file_service = FileService()
        self.processor_factory = ProcessorFactory(file_service)

        # Performance metrics
        self.metrics = MultiLanguageMetricsServiceProcessingMetrics()

    async def process_directory(
        self, directory: str, languages: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], MultiLanguageMetricsServiceProcessingMetrics]:
        """Process all files in a directory in parallel.

        Args:
            directory: Directory to process
            languages: Optional list of languages to filter by

        Returns:
            Tuple containing list of processing results and processing metrics
        """
        # Start metrics
        self.metrics = MultiLanguageMetricsServiceProcessingMetrics()
        self.metrics.start()

        # Find all files in directory
        file_paths = await self._find_files(directory, languages)
        self.metrics.total_files = len(file_paths)

        logger.info(f"Found {len(file_paths)} files to process in {directory}")

        # Process files in parallel using ThreadPoolExecutor
        # We use threads rather than processes because most of the work is I/O bound
        results: list[Any] = []
        # Create tasks for each file
        tasks: list[Any] = []
        for file_path in file_paths:
            task = asyncio.create_task(self._process_file_with_metrics(file_path))
            tasks.append(task)

        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks)
        results.extend([r for r in completed_results if r])

        # End metrics
        self.metrics.end()

        return results, self.metrics

    async def _find_files(self, directory: str, languages: list[str] | None = None) -> list[str]:
        """Find files to process in a directory.

        Args:
            directory: Directory to search in
            languages: Optional list of languages to filter by

        Returns:
            List of file paths to process
        """
        supported_extensions: list[Any] = []
        # Get supported extensions from processor factory
        if languages:
            # Filter by specified languages
            all_extensions = self.processor_factory.get_extension_language_map()
            for ext, lang in all_extensions.items():
                if lang in languages:
                    supported_extensions.append(ext)
        else:
            # Use all supported extensions
            supported_extensions = list(self.processor_factory.get_extension_language_map().keys())

        # Walk directory to find matching files
        file_paths: list[Any] = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext in supported_extensions:
                    file_paths.append(file_path)

        return file_paths

    async def _process_file_with_metrics(self, file_path: str) -> dict[str, Any] | None:
        """Process a single file and record metrics.

        Args:
            file_path: Path to the file to process

        Returns:
            Processing result or None if processing failed
        """
        start_time = time.time()
        success = False
        result = None

        try:
            processor = self.processor_factory.get_processor_for_file(file_path)
            if processor:
                result = await processor.process_document(file_path)
                success = True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result = {
                "file_path": file_path,
                "error": str(e),
                "entities": [],
                "relations": [],
            }

        # Record metrics
        duration = time.time() - start_time
        self.metrics.add_file_time(file_path, duration, success)

        return result
