"""Parallel processing helpers for component extraction.

This module contains helper classes and functions for parallel processing
of component extraction tasks to improve performance on large codebases.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx: Any = None

from models.architecture_models import Component

# Configure logger
logger = logging.getLogger(__name__)


class ComponentExtractorParallel:
    """Helper class for parallel component extraction operations."""

    def __init__(
        self,
        max_workers: int,
        use_processes: bool,
        min_parallel_threshold: int,
    ):
        """Initialize the parallel processor.

        Args:
            max_workers: Maximum number of parallel workers
            use_processes: Whether to use processes instead of threads
            min_parallel_threshold: Minimum graph size to trigger parallel processing
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.min_parallel_threshold = min_parallel_threshold

    def create_node_chunks(self, graph: Any) -> list[list[tuple[Any, dict[str, Any]]]]:
        """Create chunks of nodes for parallel processing."""
        all_nodes = list(graph.nodes(data=True))
        chunk_size = max(1, len(all_nodes) // self.max_workers)

        node_chunks = [all_nodes[i : i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

        logger.info(
            "Processing %d nodes in %d chunks (workers: %d)",
            len(all_nodes),
            len(node_chunks),
            self.max_workers,
        )

        return node_chunks

    def execute_parallel_processing(
        self,
        node_chunks: list[list[tuple[Any, dict[str, Any]]]],
        process_func: Any,
    ) -> list[Any]:
        """Execute parallel processing of node chunks with generic function."""
        executor_cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        results = []

        with executor_cls(max_workers=min(self.max_workers, len(node_chunks))) as executor:
            futures = {
                executor.submit(process_func, chunk): i for i, chunk in enumerate(node_chunks)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    OSError,
                    MemoryError,
                ) as e:
                    logger.error("Error in parallel processing: %s", str(e))

        return results

    def merge_file_mappings(self, file_component_maps: list[dict[str, str]]) -> dict[str, str]:
        """Merge file-to-component mappings from all workers."""
        file_to_component: dict[str, str] = {}
        for mapping in file_component_maps:
            file_to_component.update(mapping)
        return file_to_component

    def group_files_by_component(self, file_to_component: dict[str, str]) -> dict[str, list[str]]:
        """Group files by their component ID."""
        component_files: dict[str, list[str]] = {}
        for file_path, component_id in file_to_component.items():
            if component_id not in component_files:
                component_files[component_id] = []
            component_files[component_id].append(file_path)
        return component_files

    def create_components_from_groups(
        self,
        component_files: dict[str, list[str]],
        name_generator: Callable[[str], str],
    ) -> list[Component]:
        """Create component objects from grouped files."""
        components: list[Component] = []
        for component_id, files in component_files.items():
            component = Component(
                id=component_id,
                name=name_generator(component_id),
                files=sorted(files),
                dependencies=[],  # Will be resolved later
                layer=None,
            )
            components.append(component)
        return components

    def classify_files_for_build_dirs(
        self, nodes: list[tuple[Any, dict[str, Any]]], build_dirs: list[str]
    ) -> dict[str, set[str]]:
        """Classify files into build directories.

        Args:
            nodes: List of (node_id, node_data) tuples
            build_dirs: List of build file directories

        Returns:
            Mapping from build directories to sets of file paths
        """
        result: dict[str, set[str]] = {build_dir: set() for build_dir in build_dirs}

        for _, node_data in nodes:
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            for build_dir in build_dirs:
                if file_path.startswith(build_dir):
                    result[build_dir].add(file_path)
                    break

        return result

    def process_node_chunk_for_folders(
        self,
        nodes: list[tuple[Any, dict[str, Any]]],
        folder_roots: Sequence[str],
        pattern_matcher: Callable[[str, str], bool],
        component_id_generator: Callable[[str, str], str],
    ) -> dict[str, str]:
        """Process a chunk of nodes for folder-based component detection.

        Args:
            nodes: List of (node_id, node_data) tuples
            folder_roots: Folder glob patterns for component detection
            pattern_matcher: Function to match file paths against patterns
            component_id_generator: Function to generate component IDs

        Returns:
            Mapping from file paths to component IDs
        """
        file_to_component: dict[str, str] = {}

        for _, node_data in nodes:
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Find the first matching folder pattern
            for pattern in folder_roots:
                if pattern_matcher(file_path, pattern):
                    component_id = component_id_generator(file_path, pattern)
                    file_to_component[file_path] = component_id
                    break

        return file_to_component

    def process_chunk_annotations(
        self,
        nodes: list[tuple[Any, dict[str, Any]]],
        patterns: list[tuple[str, str]],
        code_extractor: Callable[[dict[str, Any]], str],
        language_matcher: Callable[[str, str], bool],
    ) -> dict[str, set[str]]:
        """Process a chunk of nodes for annotation-based component detection.

        Args:
            nodes: List of (node_id, node_data) tuples
            patterns: List of (regex_pattern, language) tuples for matching
            code_extractor: Function to extract code from node data
            language_matcher: Function to match file paths to languages

        Returns:
            Mapping from component names to sets of file paths
        """
        import re

        component_files: dict[str, set[str]] = {}

        for _, node_data in nodes:
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Get node content (code)
            node_code = code_extractor(node_data)
            if not node_code:
                continue

            # Check for component annotations
            for pattern, lang in patterns:
                # Skip language-specific patterns if they don't match the file
                if lang != "any" and not language_matcher(file_path, lang):
                    continue

                matches = re.findall(pattern, node_code, re.MULTILINE)
                for component_name in matches:
                    if component_name not in component_files:
                        component_files[component_name] = set()
                    component_files[component_name].add(file_path)

        return component_files
