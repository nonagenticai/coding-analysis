"""Component extraction module for identifying architectural components from code graphs.

This module provides functionality to extract coarse-grained architectural components
from a dependency graph using various detection strategies including folder-based,
build file-based, and annotation-based detection. It supports parallel processing
for performance optimization on large codebases.
"""

from __future__ import annotations

import fnmatch
import logging
import multiprocessing
import os
import re
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx
else:
    # Try to import networkx, but make it optional
    try:
        import networkx as nx

        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False
        # NetworkX integration is optional - application will work without graph analysis
        nx = None

from coding_client.utils.profiling import ProgressReporter, profile, progressive_task

from models.architecture_models import (
    Component,
    ComponentExtractionStats,
    RepoConfig,
)
from processing.architecture.component_extractor_helpers import (
    ComponentExtractionHelpers,
)
from processing.architecture.component_extractor_parallel import (
    ComponentExtractorParallel,
)

# Configure logger
logger = logging.getLogger(__name__)


class ComponentExtractor:
    """Extract coarse-grained components from a code graph using various detection strategies.

    Features:
    - Memory and CPU profiling for large-scale processing
    - Adaptive parallelism based on workload size and system resources
    - Progressive reporting for long-running extraction tasks
    - Performance metrics collection and visualization data generation
    """

    def __init__(
        self,
        max_workers: int | None = None,
        use_processes: bool = False,
        min_parallel_threshold: int = 1000,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ):
        """Initialize the component extractor with performance options.

        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            use_processes: Whether to use processes instead of threads (default: False)
            min_parallel_threshold: Minimum graph size to trigger parallel processing
            progress_callback: Optional callback for progress reporting
        """
        # Set up parallelism configuration
        self.max_workers = max_workers or self._get_default_worker_count()
        self.use_processes = use_processes
        self.min_parallel_threshold = min_parallel_threshold
        self.progress_callback = progress_callback

        # Performance metrics
        self.extraction_metrics: dict[str, Any] = {}

        # Initialize parallel processing helper
        self.parallel_processor = ComponentExtractorParallel(
            max_workers=self.max_workers,
            use_processes=self.use_processes,
            min_parallel_threshold=self.min_parallel_threshold,
        )

    def _get_default_worker_count(self) -> int:
        """Determine the default number of workers based on environment and system."""
        # Check for environment variable override
        if "ARCH_PROCESS_POOL_SIZE" in os.environ:
            try:
                return max(1, int(os.environ["ARCH_PROCESS_POOL_SIZE"]))
            except (ValueError, TypeError):  # fmt: skip
                pass

        # Otherwise use CPU count with some headroom
        cpu_count = multiprocessing.cpu_count()
        return max(1, cpu_count - 1) if cpu_count > 2 else 1

    @profile(log_level=logging.INFO)
    @progressive_task(name="Component Extraction")
    def extract(
        self,
        graph: nx.DiGraph,
        config: RepoConfig,
        progress_reporter: ProgressReporter | None = None,
    ) -> list[Component]:
        """Extract components using configured detection strategies.

        Args:
            graph: Dependency graph to analyze
            config: Repository configuration with detection strategies
            progress_reporter: Progress reporter (injected by decorator)

        Returns:
            List of extracted components with dependencies resolved
        """
        # Initialize extraction context
        extraction_info = self._initialize_extraction(graph, progress_reporter)

        # Run detection strategies
        components, used_strategies = self._run_detection_strategies(
            graph, config, progress_reporter
        )

        # Process and finalize components
        resolved_components = self._process_components(components, graph, progress_reporter)

        # Store extraction metrics
        self._store_extraction_metrics(
            extraction_info, used_strategies, len(components), resolved_components
        )

        logger.info(
            "Component extraction complete, found %d components",
            len(resolved_components),
        )
        return resolved_components

    def _initialize_extraction(
        self,
        graph: nx.DiGraph,
        progress_reporter: ProgressReporter | None,
    ) -> dict[str, Any]:
        """Initialize extraction process and return context information."""
        total_nodes = len(list(graph.nodes()))
        graph_size_mb = self._estimate_graph_size(graph)

        logger.info(
            "Starting component extraction on graph with %d nodes (est. size: %.2f MB)",
            total_nodes,
            graph_size_mb,
        )

        if progress_reporter:
            progress_reporter.total_steps = 5  # Major extraction phases
            self._setup_progress_callback(progress_reporter)

        return {"total_nodes": total_nodes, "graph_size_mb": graph_size_mb}

    def _setup_progress_callback(self, progress_reporter: ProgressReporter) -> None:
        """Set up combined progress callback if needed."""
        if self.progress_callback:
            original_callback = progress_reporter.callback

            def combined_callback(current: int, total: int, metrics: dict[str, Any]) -> None:
                if original_callback:
                    original_callback(current, total, metrics)
                if self.progress_callback:
                    self.progress_callback(current, total, metrics)

            progress_reporter.callback = combined_callback

    def _run_detection_strategies(
        self,
        graph: nx.DiGraph,
        config: RepoConfig,
        progress_reporter: ProgressReporter | None,
    ) -> tuple[list[Component], list[str]]:
        """Run all configured detection strategies."""
        components: list[Component] = []
        used_strategies: list[str] = []

        # Strategy 1: Folder-based detection
        if "folder" in config.detect_by:
            strategy_components = self._run_folder_strategy(graph, config, progress_reporter)
            components.extend(strategy_components)
            used_strategies.append("folder")

        # Strategy 2: Build file detection
        if "build_file" in config.detect_by:
            strategy_components = self._run_build_file_strategy(graph, config, progress_reporter)
            components.extend(strategy_components)
            used_strategies.append("build_file")

        # Strategy 3: Annotation-based detection
        if "annotation" in config.detect_by:
            strategy_components = self._run_annotation_strategy(graph, progress_reporter)
            components.extend(strategy_components)
            used_strategies.append("annotation")

        return components, used_strategies

    def _run_folder_strategy(
        self,
        graph: nx.DiGraph,
        config: RepoConfig,
        progress_reporter: ProgressReporter | None,
    ) -> list[Component]:
        """Run folder-based component detection strategy."""
        logger.info("Extracting components using folder strategy")
        folder_components = self._extract_by_folder(graph, config.folder_roots)

        if progress_reporter:
            progress_reporter.update(
                1,
                strategy="folder",
                components_found=len(folder_components),
            )

        return folder_components

    def _run_build_file_strategy(
        self,
        graph: nx.DiGraph,
        config: RepoConfig,
        progress_reporter: ProgressReporter | None,
    ) -> list[Component]:
        """Run build file-based component detection strategy."""
        logger.info("Extracting components using build file strategy")
        build_components = self._extract_by_build_files(graph, config.build_files)

        if progress_reporter:
            progress_reporter.update(
                1,
                strategy="build_file",
                components_found=len(build_components),
            )

        return build_components

    def _run_annotation_strategy(
        self,
        graph: nx.DiGraph,
        progress_reporter: ProgressReporter | None,
    ) -> list[Component]:
        """Run annotation-based component detection strategy."""
        logger.info("Extracting components using annotation strategy")
        annotation_components = self._extract_by_annotations(graph)

        if progress_reporter:
            progress_reporter.update(
                1,
                strategy="annotation",
                components_found=len(annotation_components),
            )

        return annotation_components

    def _process_components(
        self,
        components: list[Component],
        graph: nx.DiGraph,
        progress_reporter: ProgressReporter | None,
    ) -> list[Component]:
        """Process raw components: deduplicate and resolve dependencies."""
        initial_count = len(components)
        logger.info("Initial component extraction found %d components", initial_count)

        # Remove duplicates
        logger.info("Deduplicating components")
        unique_components = self._deduplicate_components(components)

        if progress_reporter:
            progress_reporter.update(
                1,
                phase="deduplicate",
                before_count=initial_count,
                after_count=len(unique_components),
            )

        # Resolve dependencies
        logger.info("Resolving component dependencies")
        resolved_components = self._resolve_dependencies(unique_components, graph)

        if progress_reporter:
            progress_reporter.update(
                1,
                phase="resolve_dependencies",
                components_count=len(resolved_components),
            )

        return resolved_components

    def _store_extraction_metrics(
        self,
        extraction_info: dict[str, Any],
        used_strategies: list[str],
        initial_count: int,
        resolved_components: list[Component],
    ) -> None:
        """Store metrics about the extraction process."""
        self.extraction_metrics = {
            "strategies_used": used_strategies,
            "initial_components": initial_count,
            "deduplicated_components": len(resolved_components),
            "final_components": len(resolved_components),
            "graph_nodes": extraction_info["total_nodes"],
            "graph_size_mb": extraction_info["graph_size_mb"],
        }

    def _estimate_graph_size(self, graph: nx.DiGraph) -> float:
        """Estimate the memory size of the graph in MB."""
        # Sample a subset of nodes to estimate average node size
        sample_size = min(100, len(list(graph.nodes())))
        if sample_size == 0:
            return 0.0

        total_size = 0.0
        node_data_items = list(graph.nodes(data=True))
        for i, (_, data) in enumerate(node_data_items):
            if i >= sample_size:
                break

            # Estimate size of node data dictionary
            node_size = sum(len(str(k)) + len(str(v)) for k, v in data.items())
            total_size += float(node_size)

        # Estimate average node size
        avg_node_size = total_size / float(sample_size) if sample_size > 0 else 0.0

        # Estimate total graph size including edges
        node_count = len(list(graph.nodes()))
        edge_count = len(list(graph.edges()))
        estimated_size = (avg_node_size * float(node_count)) + (20.0 * float(edge_count))

        # Convert to MB
        return estimated_size / (1024.0 * 1024.0)

    @profile
    def _extract_by_folder(self, graph: nx.DiGraph, folder_roots: Sequence[str]) -> list[Component]:
        """Extract components based on folder structure.

        Uses adaptive parallelism for large graphs to update performance.

        Args:
            graph: Dependency graph to process
            folder_roots: Folder glob patterns for component detection

        Returns:
            List of extracted components (without dependencies)
        """
        # When a repo ships no .codegraph.yml, ``folder_roots`` is empty and no
        # folder grouping would happen — leaving a fully-parsed graph with zero
        # components. Fall back to grouping by top-level directory ("*") so
        # architecture analysis yields meaningful components out of the box.
        if not folder_roots:
            folder_roots = ["*"]

        # Determine if we should use parallel processing
        node_count = len(list(graph.nodes()))
        use_parallel = node_count >= self.min_parallel_threshold and self.max_workers > 1

        if use_parallel:
            logger.info(
                "Using parallel processing for folder extraction (%d nodes)",
                node_count,
            )
            return self._extract_by_folder_parallel(graph, folder_roots)

        logger.info(
            "Using sequential processing for folder extraction (%d nodes)",
            node_count,
        )
        return self._extract_by_folder_sequential(graph, folder_roots)

    def _extract_by_folder_sequential(
        self,
        graph: nx.DiGraph,
        folder_roots: Sequence[str],
    ) -> list[Component]:
        """Sequential implementation of folder-based component extraction."""
        components: list[Component] = []
        file_to_component: dict[str, str] = {}

        # Group files by matching folder patterns
        for _, node_data in graph.nodes(data=True):
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Find the first matching folder pattern
            for pattern in folder_roots:
                if self._matches_pattern(file_path, pattern):
                    component_id = self._get_component_id_from_pattern(file_path, pattern)
                    file_to_component[file_path] = component_id
                    break

        # Create components from grouped files
        component_files: dict[str, list[str]] = {}
        for file_path, component_id in file_to_component.items():
            if component_id not in component_files:
                component_files[component_id] = []
            component_files[component_id].append(file_path)

        # Build component objects
        for component_id, files in component_files.items():
            component = Component(
                id=component_id,
                name=self._get_component_name(component_id),
                files=sorted(files),
                dependencies=[],  # Will be resolved later
                layer=None,
            )
            components.append(component)

        return components

    def _extract_by_folder_parallel(
        self,
        graph: nx.DiGraph,
        folder_roots: Sequence[str],
    ) -> list[Component]:
        """Parallel implementation of folder-based component extraction."""
        # Create and process node chunks
        node_chunks = self.parallel_processor.create_node_chunks(graph)

        # Execute parallel processing
        process_func = partial(
            self.parallel_processor.process_node_chunk_for_folders,
            folder_roots=folder_roots,
            pattern_matcher=self._matches_pattern,
            component_id_generator=self._get_component_id_from_pattern,
        )
        file_component_maps = self.parallel_processor.execute_parallel_processing(
            node_chunks, process_func
        )

        # Build components from results
        return self._build_components_from_maps(file_component_maps)

    def _build_components_from_maps(
        self, file_component_maps: list[dict[str, str]]
    ) -> list[Component]:
        """Build component objects from file-to-component mappings."""
        # Merge results
        file_to_component = self.parallel_processor.merge_file_mappings(file_component_maps)

        # Group by component
        component_files = self.parallel_processor.group_files_by_component(file_to_component)

        # Create components
        return self.parallel_processor.create_components_from_groups(
            component_files, self._get_component_name
        )

    @profile
    def _extract_by_build_files(
        self,
        graph: nx.DiGraph,
        build_files: Sequence[str],
    ) -> list[Component]:
        """Extract components based on build file locations.

        This method identifies build files in the graph and creates components
        from files in the same directory structure.

        Args:
            graph: Dependency graph to analyze
            build_files: List of build file names to detect (e.g., "package.json")

        Returns:
            List of extracted components
        """
        # Determine if we should use parallel processing
        node_count = len(list(graph.nodes()))
        use_parallel = node_count >= self.min_parallel_threshold and self.max_workers > 1

        if use_parallel:
            logger.info(
                "Using parallel processing for build file detection (%d nodes)",
                node_count,
            )
            return self._extract_by_build_files_parallel(graph, build_files)

        logger.info(
            "Using sequential processing for build file detection (%d nodes)",
            node_count,
        )
        return self._extract_by_build_files_sequential(graph, build_files)

    def _extract_by_build_files_sequential(
        self,
        graph: nx.DiGraph,
        build_files: Sequence[str],
    ) -> list[Component]:
        """Sequential implementation of build file-based component extraction."""
        components: list[Component] = []

        # Find all build files in the graph
        build_file_nodes = []
        for node_data in graph.nodes.values():
            # file_path may be None (synthetic nodes, or rows whose path lives
            # only in properties before the adapter coalesces it) — guard against
            # Path(None) which raises TypeError.
            file_path = node_data.get("file_path") or ""
            if not file_path:
                continue
            if any(Path(file_path).name == build_file for build_file in build_files):
                build_file_nodes.append(file_path)

        logger.info(
            "Found %d build files for component detection",
            len(build_file_nodes),
        )

        # For each build file, create a component containing all files in its directory
        for build_file_path in build_file_nodes:
            build_dir = str(Path(build_file_path).parent)
            component_files = []

            # Collect all files in the same directory and subdirectories
            for _, node_data in graph.nodes(data=True):
                file_path = node_data.get("file_path", "")
                if file_path and file_path.startswith(build_dir):
                    component_files.append(file_path)

            # Create a component
            if component_files:
                component_id = f"component_{Path(build_dir).name}"
                component = Component(
                    id=component_id,
                    name=self._get_component_name(component_id),
                    files=sorted(component_files),
                    dependencies=[],  # Will be resolved later
                    layer=None,
                )
                components.append(component)

        return components

    def _extract_by_build_files_parallel(
        self,
        graph: nx.DiGraph,
        build_files: Sequence[str],
    ) -> list[Component]:
        """Parallel implementation of build file-based component extraction."""
        # Find build files and create directories
        build_dirs = self._find_build_directories(graph, build_files)
        if not build_dirs:
            return []

        # Process files in parallel
        dir_to_files = self._classify_files_parallel(graph, build_dirs)

        # Create components from classified files
        return self._create_build_components(dir_to_files)

    def _find_build_directories(self, graph: nx.DiGraph, build_files: Sequence[str]) -> list[str]:
        """Find all build file directories in the graph."""
        build_file_paths = []
        for node_data in graph.nodes.values():
            # file_path may be None (synthetic nodes, or edge-created nodes that
            # never received attributes) — guard against Path(None) which raises
            # TypeError. Mirrors the sequential path's guard.
            file_path = node_data.get("file_path") or ""
            if not file_path:
                continue
            if any(Path(file_path).name == build_file for build_file in build_files):
                build_file_paths.append(file_path)

        logger.info(
            "Found %d build files for component detection",
            len(build_file_paths),
        )

        return [str(Path(path).parent) for path in build_file_paths]

    def _classify_files_parallel(
        self,
        graph: nx.DiGraph,
        build_dirs: list[str],
    ) -> dict[str, set[str]]:
        """Classify files into build directories using parallel processing."""
        # Create node chunks
        node_chunks = self.parallel_processor.create_node_chunks(graph)

        # Execute parallel classification
        process_func = partial(
            self.parallel_processor.classify_files_for_build_dirs, build_dirs=build_dirs
        )
        results = self.parallel_processor.execute_parallel_processing(node_chunks, process_func)

        # Initialize result dictionary and merge results
        dir_to_files: dict[str, set[str]] = {build_dir: set() for build_dir in build_dirs}

        for result in results:
            for build_dir, files in result.items():
                dir_to_files[build_dir].update(files)

        return dir_to_files

    def _create_build_components(self, dir_to_files: dict[str, set[str]]) -> list[Component]:
        """Create components from build directory file mappings."""
        return ComponentExtractionHelpers.create_build_components(
            dir_to_files, self._get_component_name
        )

    @profile
    def _extract_by_annotations(self, graph: nx.DiGraph) -> list[Component]:
        """Extract components based on code annotations and decorators.

        This method identifies component annotations in code files
        and groups them into logical components. It supports multiple
        annotation formats across different programming languages.

        Args:
            graph: Dependency graph to analyze

        Returns:
            List of extracted components
        """
        # Determine if we should use parallel processing
        node_count = len(list(graph.nodes()))
        use_parallel = node_count >= self.min_parallel_threshold and self.max_workers > 1

        if use_parallel:
            logger.info(
                "Using parallel processing for annotation detection (%d nodes)",
                node_count,
            )
            return self._extract_by_annotations_parallel(graph)

        logger.info(
            "Using sequential processing for annotation detection (%d nodes)",
            node_count,
        )
        return self._extract_by_annotations_sequential(graph)

    def _extract_by_annotations_sequential(self, graph: nx.DiGraph) -> list[Component]:
        """Sequential implementation of annotation-based component extraction."""
        components: list[Component] = []
        component_files: dict[str, set[str]] = {}

        # Get annotation patterns
        annotation_patterns = self._get_annotation_patterns()

        # Scan nodes for annotations
        for _, node_data in graph.nodes(data=True):
            file_path = node_data.get("file_path", "")
            if not file_path:
                continue

            # Get node content (code)
            node_code = node_data.get("code", "") or node_data.get("content", "")
            if not node_code:
                continue

            # Check for component annotations
            for pattern, lang in annotation_patterns:
                # Skip language-specific patterns if they don't match the file
                if lang != "any":
                    ext_map = {
                        "python": ".py",
                        "javascript": ".js",
                        "typescript": ".ts",
                        "java": ".java",
                    }
                    if lang in ext_map and not file_path.endswith(ext_map[lang]):
                        continue

                matches = re.findall(pattern, node_code, re.MULTILINE)
                for component_name in matches:
                    if component_name not in component_files:
                        component_files[component_name] = set()
                    component_files[component_name].add(file_path)

        # Also check for module-level annotations in files
        file_annotations = ComponentExtractionHelpers.scan_file_annotations(
            graph,
            lambda path: node_data.get("code", "") or node_data.get("content", ""),
        )
        for component_name, files in file_annotations.items():
            if component_name not in component_files:
                component_files[component_name] = set()
            component_files[component_name].update(files)

        logger.info("Found %d annotated components", len(component_files))

        # Create components from collected files
        for component_name, files in component_files.items():
            if files:  # Only create component if it has files
                component = Component(
                    id=f"annotation-{component_name}",
                    name=component_name,
                    files=sorted(files),
                    dependencies=[],
                    layer=None,
                )
                components.append(component)

        return components

    def _extract_by_annotations_parallel(self, graph: nx.DiGraph) -> list[Component]:
        """Parallel implementation of annotation-based component extraction."""
        # Process chunks in parallel to find component annotations
        worker_results = self._process_annotation_chunks(graph)

        # Merge results and add file-level annotations
        component_files = self._merge_annotation_results(worker_results, graph)

        # Create components from collected files
        return self._create_annotation_components(component_files)

    def _process_annotation_chunks(self, graph: nx.DiGraph) -> list[dict[str, set[str]]]:
        """Process graph chunks in parallel to find annotations."""
        node_chunks = self.parallel_processor.create_node_chunks(graph)
        annotation_patterns = self._get_annotation_patterns()

        process_func = partial(
            self.parallel_processor.process_chunk_annotations,
            patterns=annotation_patterns,
            code_extractor=ComponentExtractionHelpers.get_node_code,
            language_matcher=ComponentExtractionHelpers.matches_language,
        )

        return self.parallel_processor.execute_parallel_processing(node_chunks, process_func)

    def _merge_annotation_results(
        self,
        worker_results: list[dict[str, set[str]]],
        graph: nx.DiGraph,
    ) -> dict[str, set[str]]:
        """Merge annotation results from all workers and add file-level annotations."""
        component_files: dict[str, set[str]] = {}

        # Merge worker results
        for worker_result in worker_results:
            for component_name, files in worker_result.items():
                if component_name not in component_files:
                    component_files[component_name] = set()
                component_files[component_name].update(files)

        # Add file-level annotations (sequential processing)
        file_annotations = ComponentExtractionHelpers.scan_file_annotations(
            graph, ComponentExtractionHelpers.get_node_code
        )
        for component_name, files in file_annotations.items():
            if component_name not in component_files:
                component_files[component_name] = set()
            component_files[component_name].update(files)

        return component_files

    def _create_annotation_components(
        self, component_files: dict[str, set[str]]
    ) -> list[Component]:
        """Create components from annotation file mappings."""
        logger.info("Found %d annotated components", len(component_files))
        return ComponentExtractionHelpers.create_annotation_components(component_files)

    def _get_annotation_patterns(self) -> list[tuple[str, str]]:
        """Get the list of annotation patterns for component detection."""
        return ComponentExtractionHelpers.get_annotation_patterns()

    @profile
    def _deduplicate_components(self, components: list[Component]) -> list[Component]:
        """Remove duplicate components, merging files when necessary.

        Args:
            components: List of components that may contain duplicates

        Returns:
            Deduplicated list of components with merged file sets
        """
        logger.info("Deduplicating %d components", len(components))
        component_map: dict[str, Component] = {}
        duplicates_found = 0

        for component in components:
            if component.id in component_map:
                # Merge files from duplicate components
                duplicates_found += 1
                existing = component_map[component.id]
                all_files = set(existing.files) | set(component.files)
                existing.files = sorted(all_files)
            else:
                component_map[component.id] = component

        logger.info("Found and merged %d duplicate components", duplicates_found)
        return list(component_map.values())

    @profile
    def _resolve_dependencies(
        self,
        components: list[Component],
        graph: nx.DiGraph,
    ) -> list[Component]:
        """Resolve dependencies between components.

        This method analyzes the dependency graph to determine component-level
        dependencies based on file-level dependencies in the graph.

        Args:
            components: List of extracted components
            graph: Dependency graph to analyze for relationships

        Returns:
            List of components with resolved dependencies
        """
        logger.info("Resolving dependencies between %d components", len(components))

        # Create mapping from file to component
        file_to_component: dict[str, str] = {}
        for component in components:
            for file_path in component.files:
                file_to_component[file_path] = component.id

        # Find dependencies between components
        for component in components:
            dependencies = self._find_component_dependencies(component, graph, file_to_component)
            component.dependencies = sorted(dependencies)

        return components

    def _find_component_dependencies(
        self,
        component: Component,
        graph: nx.DiGraph,
        file_to_component: dict[str, str],
    ) -> set[str]:
        """Find dependencies for a single component.

        Args:
            component: Component to analyze
            graph: Dependency graph
            file_to_component: Mapping from file paths to component IDs

        Returns:
            Set of component IDs that this component depends on
        """
        dependencies: set[str] = set()

        # Check all files in this component
        for file_path in component.files:
            # Find nodes in this file
            file_nodes = [
                node_id
                for node_id, data in graph.nodes(data=True)
                if data.get("file_path") == file_path
            ]

            # Check outgoing edges from nodes in this file
            for node_id in file_nodes:
                self._add_node_dependencies(
                    node_id, graph, file_to_component, component.id, dependencies
                )

        return dependencies

    def _add_node_dependencies(
        self,
        node_id: Any,
        graph: nx.DiGraph,
        file_to_component: dict[str, str],
        current_component_id: str,
        dependencies: set[str],
    ) -> None:
        """Add dependencies from a single node to the dependency set.

        Args:
            node_id: Node to check for dependencies
            graph: Dependency graph
            file_to_component: Mapping from file paths to component IDs
            current_component_id: ID of the current component
            dependencies: Set to add dependencies to
        """
        for _, target_id, edge_data in graph.out_edges(node_id, data=True):
            # ``contains`` is a STRUCTURAL edge (a file contains its functions/
            # classes), never a dependency. Counting it as one turns a
            # same-named-entity node collision into a bogus cross-component edge
            # (the live-found alpha->gamma false dependency). Real dependencies are
            # imports / calls / inherits / references.
            rel = str(edge_data.get("relationship") or edge_data.get("type") or "").lower()
            if rel == "contains":
                continue

            # A function-local (deferred) import does not run at module import
            # time, so it cannot participate in an import-time dependency cycle.
            # Exclude it from architecture dependency edges (consistent with the AA
            # refactor worker, which breaks a cycle by relocating a top-level import
            # into a function body). The ``deferred`` flag is stamped at extraction
            # time in processing/analyzers/deep/entities.py.
            props = edge_data.get("properties") or {}
            if rel in ("imports", "imports_from") and props.get("deferred"):
                continue

            target_data = graph.nodes[target_id]
            target_file = target_data.get("file_path", "")

            if target_file and target_file in file_to_component:
                target_component = file_to_component[target_file]
                if target_component != current_component_id:
                    dependencies.add(target_component)

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern."""
        return fnmatch.fnmatch(file_path, pattern)

    def _get_component_id_from_pattern(self, file_path: str, pattern: str) -> str:
        """Generate a component ID from a file path and pattern."""
        # Extract the relevant part of the path based on the pattern
        if "**" in pattern:
            # For patterns like "src/apps/**", extract the app name
            pattern_parts = pattern.split("**")
            if len(pattern_parts) >= 2:
                prefix = pattern_parts[0].rstrip("/")
                remaining_path = file_path[len(prefix) :].lstrip("/")
                component_part = (
                    remaining_path.split("/")[0] if "/" in remaining_path else remaining_path
                )
                return f"component_{prefix.replace('/', '_')}_{component_part}"

        # Fallback: use the directory name
        return f"component_{Path(file_path).parent.name}"

    def _get_component_name(self, component_id: str) -> str:
        """Generate a human-readable name from a component ID."""
        # Remove "component_" prefix and convert underscores to spaces
        if component_id.startswith("component_"):
            name = component_id[10:]  # Remove "component_" prefix
            return name.replace("_", " ").title()
        return component_id

    def get_extraction_stats(self, components: list[Component]) -> ComponentExtractionStats:
        """Get statistics about the extraction process."""
        total_files = sum(len(comp.files) for comp in components)

        return ComponentExtractionStats(
            components_found=len(components),
            edges_scanned=total_files,  # Approximation
        )
