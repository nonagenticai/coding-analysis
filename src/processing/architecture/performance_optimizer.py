"""Performance optimization module for architecture processing.

This module provides comprehensive performance optimization capabilities for
code graph architecture analysis, including memory-efficient graph processing,
adaptive parallelism management, and performance metrics tracking.

Key components:
- MemoryEfficientGraphProcessor: Streaming graph processing with memory managemen
- AdaptiveParallelismManager: Dynamic worker allocation based on system resources
- ArchitecturePerformanceOptimizer: Main coordinator for optimized architecture processing
- Performance monitoring decorators and metrics collection
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
import time
from collections.abc import Callable, Iterator
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import (
    Any,
    TypeVar,
)

import networkx as nx

# Make psutil optional
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil: Any = None

from models.architecture_models import (
    ArchitectureAnalysisResult,
    Component,
    Layer,
    RepoConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization settings."""

    chunk_size: int = 1000
    max_memory_mb: int = 512
    min_workers: int = 2
    max_workers: int | None = None
    enable_profiling: bool = False


@dataclass
class ProcessingCounts:
    """Processing count metrics."""

    components_processed: int
    nodes_analyzed: int
    edges_processed: int


@dataclass
class ArchitectureCacheMetrics:
    """Cache performance metrics."""

    cache_hits: int
    cache_misses: int


@dataclass
class ArchitecturePerformanceMetrics:
    """Performance metrics for architecture processing."""

    total_processing_time: float
    memory_usage_mb: float
    cpu_utilization: float
    processing_counts: ProcessingCounts
    cache_metrics: ArchitectureCacheMetrics
    parallel_workers_used: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_processing_time": self.total_processing_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization": self.cpu_utilization,
            "components_processed": self.processing_counts.components_processed,
            "nodes_analyzed": self.processing_counts.nodes_analyzed,
            "edges_processed": self.processing_counts.edges_processed,
            "cache_hits": self.cache_metrics.cache_hits,
            "cache_misses": self.cache_metrics.cache_misses,
            "parallel_workers_used": self.parallel_workers_used,
        }


class MemoryEfficientGraphProcessor[T]:
    """Memory-efficient graph processor using streaming and chunking."""

    def __init__(self, chunk_size: int = 1000, max_memory_mb: int = 512):
        """Initialize the memory-efficient processor.

        Args:
            chunk_size: Number of nodes to process in each chunk
            max_memory_mb: Maximum memory usage threshold in MB
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0.0
        self._memory_lock = Lock()

    def process_graph_streaming(
        self,
        graph: nx.DiGraph,
        processor_func: Callable[[list[str]], T],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[T]:
        """Process graph nodes in memory-efficient streaming fashion.

        Args:
            graph: The graph to process
            processor_func: Function to process each chunk of nodes
            progress_callback: Optional callback for progress updates

        Yields:
            Results from processing each chunk
        """
        total_nodes = graph.number_of_nodes()
        processed_nodes = 0

        # Process nodes in chunks to manage memory
        for chunk in self._chunk_nodes(list(graph.nodes()), self.chunk_size):
            # Check memory usage before processing
            self._check_memory_usage()

            # Process chunk
            try:
                result = processor_func(chunk)
                yield result

                processed_nodes += len(chunk)

                # Update progress
                if progress_callback:
                    progress_callback(processed_nodes, total_nodes)

            except (RuntimeError, ValueError, TypeError, MemoryError) as e:
                logger.error("Error processing chunk: %s", e)
                continue

            # Force garbage collection if memory is high
            if self._get_memory_usage_mb() > self.max_memory_mb * 0.8:
                gc.collect()

    def _chunk_nodes(self, nodes: list[str], chunk_size: int) -> Iterator[list[str]]:
        """Split nodes into chunks for processing."""
        for i in range(0, len(nodes), chunk_size):
            yield nodes[i : i + chunk_size]

    def _check_memory_usage(self) -> None:
        """Check current memory usage and log warnings if high."""
        current_mb = self._get_memory_usage_mb()

        with self._memory_lock:
            self.current_memory_mb = float(current_mb)

        if current_mb > self.max_memory_mb:
            logger.warning(
                "Memory usage (%.1fMB) exceeds threshold (%sMB)",
                current_mb,
                self.max_memory_mb,
            )

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))

    def get_memory_status(self) -> dict[str, float]:
        """Get current memory status information.

        Returns:
            Dictionary containing memory usage metrics
        """
        current_usage = self._get_memory_usage_mb()
        return {
            "current_memory_mb": current_usage,
            "max_memory_mb": float(self.max_memory_mb),
            "memory_utilization_percent": (current_usage / self.max_memory_mb) * 100,
            "available_memory_mb": float(self.max_memory_mb - current_usage),
        }


class AdaptiveParallelismManager:
    """Manages adaptive parallelism based on system resources and workload."""

    def __init__(self, min_workers: int = 2, max_workers: int | None = None):
        """Initialize the adaptive parallelism manager.

        Args:
            min_workers: Minimum number of workers to maintain
            max_workers: Maximum number of workers (defaults to CPU count)
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.current_workers = min_workers
        self.performance_history: list[float] = []
        self.load_history: list[float] = []

        # System monitoring
        self.cpu_threshold = 0.8
        self.memory_threshold = 0.8

        logger.info(
            "Adaptive parallelism initialized: %s-%s workers",
            min_workers,
            self.max_workers,
        )

    def get_optimal_worker_count(self, task_count: int, task_complexity: float = 1.0) -> int:
        """Calculate optimal number of workers based on current conditions.

        Args:
            task_count: Number of tasks to process
            task_complexity: Complexity factor (1.0 = normal, >1.0 = complex)

        Returns:
            Optimal number of workers
        """
        # Base calculation on task count and complexity
        base_workers = min(task_count, self.max_workers)

        # Adjust based on system resources
        if HAS_PSUTIL:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent / 100
        else:
            cpu_usage = 50.0  # Default moderate usage
            memory_usage = 0.5  # Default 50% memory

        # Reduce workers if system is under stress
        if cpu_usage > self.cpu_threshold * 100:
            base_workers = max(self.min_workers, int(base_workers * 0.7))

        if memory_usage > self.memory_threshold:
            base_workers = max(self.min_workers, int(base_workers * 0.5))

        # Adjust based on task complexity
        if task_complexity > 1.5:
            base_workers = max(self.min_workers, int(base_workers * 0.8))
        elif task_complexity < 0.5:
            base_workers = min(self.max_workers, int(base_workers * 1.2))

        # Store history for future optimization
        self.load_history.append(cpu_usage)
        if len(self.load_history) > 10:
            self.load_history.pop(0)

        optimal_workers = max(self.min_workers, min(self.max_workers, base_workers))

        logger.debug(
            "Calculated optimal workers: %s (CPU: %.1f%%, Memory: %.1f%%)",
            optimal_workers,
            cpu_usage,
            memory_usage,
        )

        return optimal_workers

    def execute_parallel_tasks(
        self,
        tasks: list[Callable],
        use_processes: bool = False,
        task_complexity: float = 1.0,
    ) -> list[Any]:
        """Execute tasks in parallel with adaptive worker management.

        Args:
            tasks: List of callable tasks to execute
            use_processes: Whether to use processes instead of threads
            task_complexity: Complexity factor for worker calculation

        Returns:
            List of results from task execution
        """
        if not tasks:
            return []

        start_time = time.time()
        optimal_workers = self.get_optimal_worker_count(len(tasks), task_complexity)

        try:
            final_results = self._execute_tasks_with_executor(tasks, use_processes, optimal_workers)
        except (RuntimeError, OSError, MemoryError) as e:
            logger.error("Parallel execution failed: %s", e)
            final_results = [task() for task in tasks]

        self._record_performance_metrics(time.time() - start_time)

        logger.info(
            "Parallel execution completed: %s tasks in %.2fs with %s workers",
            len(tasks),
            time.time() - start_time,
            optimal_workers,
        )

        return final_results

    def _execute_tasks_with_executor(
        self, tasks: list[Callable], use_processes: bool, optimal_workers: int
    ) -> list[Any]:
        """Execute tasks using thread or process pool executor."""
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with executor_class(max_workers=optimal_workers) as executor:
            future_to_task = {executor.submit(task): i for i, task in enumerate(tasks)}

            results = self._collect_results(future_to_task)
            results.sort(key=lambda x: x[0])
            return [result for _, result in results]

    def _collect_results(self, future_to_task: dict) -> list[tuple]:
        """Collect results from completed futures."""
        results = []
        for future in as_completed(future_to_task):
            task_index = future_to_task[future]
            try:
                result = future.result()
                results.append((task_index, result))
            except (RuntimeError, ValueError, TypeError, TimeoutError) as e:
                logger.error("Task %s failed: %s", task_index, e)
                results.append((task_index, None))
        return results

    def _record_performance_metrics(self, execution_time: float) -> None:
        """Record performance metrics for the execution."""
        self.performance_history.append(execution_time)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)


class ArchitecturePerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(
        self,
        config: PerformanceConfig | None = None,
    ):
        """Initialize the performance optimizer.

        Args:
            config: Performance configuration settings
        """
        self.config = config or PerformanceConfig()
        self.chunk_size = self.config.chunk_size
        self.max_memory_mb = self.config.max_memory_mb
        self.enable_profiling = self.config.enable_profiling

        # Initialize components
        self.memory_processor = MemoryEfficientGraphProcessor[Any](
            self.config.chunk_size, self.config.max_memory_mb
        )
        self.parallelism_manager = AdaptiveParallelismManager(
            self.config.min_workers, self.config.max_workers
        )

        # Metrics tracking
        self.metrics = ArchitecturePerformanceMetrics(
            total_processing_time=0,
            memory_usage_mb=0,
            cpu_utilization=0,
            processing_counts=ProcessingCounts(
                components_processed=0,
                nodes_analyzed=0,
                edges_processed=0,
            ),
            cache_metrics=ArchitectureCacheMetrics(
                cache_hits=0,
                cache_misses=0,
            ),
            parallel_workers_used=0,
        )

        # Performance profiling
        self.profiling_data: dict[str, list[float]] = {}

        logger.info("Performance optimizer initialized")

    def optimize_architecture_processing(
        self,
        graph: nx.DiGraph,
        components: list[Component],
        processing_functions: dict[str, Callable],
    ) -> ArchitectureAnalysisResult:
        """Optimize the complete architecture processing pipeline.

        Args:
            graph: The code graph to analyze
            components: List of components to process
            processing_functions: Dictionary of processing functions to execute

        Returns:
            Architecture analysis result with performance metrics
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()

        logger.info(
            "Starting optimized architecture processing for %s components",
            len(components),
        )

        try:
            # Determine optimal processing strategy
            processing_strategy = self._determine_processing_strategy(graph, components)

            # Execute processing based on strategy
            if processing_strategy == "streaming":
                result = self._process_streaming(graph, components, processing_functions)
            elif processing_strategy == "parallel":
                result = self._process_parallel(graph, components, processing_functions)
            else:
                result = self._process_sequential(graph, components, processing_functions)

            # Update metrics
            self.metrics.total_processing_time = time.time() - start_time
            self.metrics.memory_usage_mb = self._get_memory_usage_mb() - initial_memory
            self.metrics.cpu_utilization = psutil.cpu_percent() if HAS_PSUTIL else 0.0
            self.metrics.processing_counts.components_processed = len(components)
            self.metrics.processing_counts.nodes_analyzed = graph.number_of_nodes()
            self.metrics.processing_counts.edges_processed = graph.number_of_edges()

            logger.info(
                "Architecture processing completed in %.2fs",
                self.metrics.total_processing_time,
            )

            return result

        except Exception as e:
            logger.error("Architecture processing failed: %s", e)
            raise

    def _determine_processing_strategy(self, graph: nx.DiGraph, components: list[Component]) -> str:
        """Determine the optimal processing strategy based on data characteristics.

        Args:
            graph: The code graph
            components: List of components

        Returns:
            Processing strategy: 'streaming', 'parallel', or 'sequential'
        """
        node_count = graph.number_of_nodes()
        component_count = len(components)
        memory_usage = self._get_memory_usage_mb()

        # Use streaming for large graphs or low memory
        if node_count > 10000 or memory_usage > self.max_memory_mb * 0.6:
            return "streaming"

        # Use parallel for moderate workloads with sufficient resources
        cpu_count = (psutil.cpu_count() or 2) if HAS_PSUTIL else 2
        if component_count > 10 and cpu_count > 2:
            return "parallel"

        # Use sequential for small workloads
        return "sequential"

    def _process_streaming(
        self,
        graph: nx.DiGraph,
        components: list[Component],
        processing_functions: dict[str, Callable],
    ) -> ArchitectureAnalysisResult:
        """Process using streaming strategy for large datasets."""
        logger.info("Using streaming processing strategy")

        # Implement streaming processing logic
        # This would involve processing components in chunks
        # For now, fallback to sequential processing
        return self._process_sequential(graph, components, processing_functions)

    def _process_parallel(
        self,
        graph: nx.DiGraph,
        components: list[Component],
        processing_functions: dict[str, Callable],
    ) -> ArchitectureAnalysisResult:
        """Process using parallel strategy for moderate workloads.

        Executes multiple processing functions in parallel and aggregates their results
        into a comprehensive ArchitectureAnalysisResult. This method handles the proper
        collection and combination of metrics, patterns, and layer analysis results.

        Args:
            graph: The code graph to analyze
            components: List of components to process
            processing_functions: Dictionary of processing functions to execute

        Returns:
            ArchitectureAnalysisResult with aggregated analysis results
        """
        logger.info("Using parallel processing strategy")
        processing_start_time = time.time()

        # Execute tasks in parallel
        parallel_results = self._execute_parallel_processing_tasks(
            graph, components, processing_functions
        )

        # Aggregate results from parallel execution
        aggregated_data = self._aggregate_parallel_results(parallel_results, components)

        # Build final result
        execution_time = time.time() - processing_start_time
        result = self._build_architecture_result(
            components, aggregated_data, execution_time, processing_functions
        )

        logger.info(
            "Parallel processing completed: %s metrics, %s patterns, %s layers in %.2fs",
            len(aggregated_data["metrics"]),
            len(aggregated_data["patterns"]),
            len(aggregated_data["layers"]),
            execution_time,
        )

        return result

    def _execute_parallel_processing_tasks(
        self,
        graph: nx.DiGraph,
        components: list[Component],
        processing_functions: dict[str, Callable],
    ) -> list[Any]:
        """Execute processing functions in parallel."""

        def create_task(func_name: str, func: Callable) -> Callable:
            """Create a task function for parallel execution."""

            def task():
                return {
                    "name": func_name,
                    "result": func(graph, components),
                }

            return task

        tasks = [create_task(func_name, func) for func_name, func in processing_functions.items()]
        return self.parallelism_manager.execute_parallel_tasks(tasks, use_processes=False)

    def _aggregate_parallel_results(
        self, parallel_results: list[Any], components: list[Component]
    ) -> dict[str, Any]:
        """Aggregate results from parallel processing tasks."""
        aggregated = {
            "metrics": [],
            "patterns": [],
            "layer_mappings": {},
            "summary": {"components": len(components)},
        }

        for task_result in parallel_results:
            if task_result is None:
                continue

            func_name = task_result.get("name", "unknown")
            result = task_result.get("result")

            if result is None:
                logger.warning("Processing function '%s' returned None result", func_name)
                continue

            self._categorize_result(func_name, result, aggregated)

        return aggregated

    def _categorize_result(self, func_name: str, result: Any, aggregated: dict[str, Any]) -> None:
        """Categorize result based on function name and type."""
        if "metric" in func_name.lower():
            self._handle_metrics_result(result, aggregated["metrics"], func_name)
        elif "pattern" in func_name.lower():
            self._handle_patterns_result(result, aggregated["patterns"], func_name)
        elif "layer" in func_name.lower():
            self._handle_layer_result(result, aggregated["layer_mappings"], func_name)
        elif "summary" in func_name.lower():
            self._handle_summary_result(result, aggregated["summary"], func_name)
        else:
            logger.debug("Unhandled result type from function '%s': %s", func_name, type(result))

    def _handle_metrics_result(self, result: Any, metrics_list: list, func_name: str) -> None:
        """Handle metrics result aggregation."""
        if isinstance(result, list):
            metrics_list.extend(result)
        elif hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
            metrics_list.extend(list(result))
        else:
            logger.warning("Unexpected metrics result type from '%s': %s", func_name, type(result))

    def _handle_patterns_result(self, result: Any, patterns_list: list, func_name: str) -> None:
        """Handle patterns result aggregation."""
        if isinstance(result, list):
            patterns_list.extend(result)
        elif hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
            patterns_list.extend(list(result))
        else:
            logger.warning("Unexpected pattern result type from '%s': %s", func_name, type(result))

    def _handle_layer_result(self, result: Any, layer_mappings: dict, func_name: str) -> None:
        """Handle layer result aggregation."""
        if isinstance(result, dict):
            layer_mappings.update(result)
        else:
            logger.warning("Unexpected layer result type from '%s': %s", func_name, type(result))

    def _handle_summary_result(self, result: Any, summary_data: dict, func_name: str) -> None:
        """Handle summary result aggregation."""
        if isinstance(result, dict):
            summary_data.update(result)
        else:
            logger.warning("Unexpected summary result type from '%s': %s", func_name, type(result))

    def _build_architecture_result(
        self,
        components: list[Component],
        aggregated_data: dict[str, Any],
        execution_time: float,
        processing_functions: dict[str, Callable],
    ) -> ArchitectureAnalysisResult:
        """Build the final ArchitectureAnalysisResult."""
        # Convert layer mappings to Layer objects
        layers_list = [
            Layer(
                name=layer_name,
                components=component_ids,
                component_count=len(component_ids),
                dependencies=[],
            )
            for layer_name, component_ids in aggregated_data["layer_mappings"].items()
        ]

        # Update summary with aggregated data
        summary_data = aggregated_data["summary"]
        summary_data.update(
            {
                "total_components": len(components),
                "total_layers": len(layers_list),
                "total_patterns": len(aggregated_data["patterns"]),
                "total_metrics": len(aggregated_data["metrics"]),
                "parallel_tasks_executed": len(processing_functions),
            }
        )

        # Update performance metrics
        self.metrics.parallel_workers_used = self.parallelism_manager.current_workers

        return ArchitectureAnalysisResult(
            codebase_id=getattr(self, "codebase_id", "unknown"),
            generated_at=datetime.now(UTC).isoformat(),
            components=components,
            layers=layers_list,
            metrics=aggregated_data["metrics"],
            patterns=aggregated_data["patterns"],
            summary=summary_data,
            execution_time=execution_time,
            config_used=RepoConfig(),
        )

    def _process_sequential(
        self,
        graph: nx.DiGraph,
        components: list[Component],
        processing_functions: dict[str, Callable],
    ) -> ArchitectureAnalysisResult:
        """Process using sequential strategy for small workloads."""
        logger.info("Using sequential processing strategy")

        # Execute processing functions sequentially
        results = {}
        for func_name, func in processing_functions.items():
            results[func_name] = func(graph, components)

        # Combine results into ArchitectureAnalysisResul
        return ArchitectureAnalysisResult(
            codebase_id="unknown",
            generated_at=datetime.now(UTC).isoformat(),
            components=components,
            metrics=[],  # Would be populated by actual metrics calculation
            layers=[],
            patterns=[],
            summary={"components": len(components)},
            execution_time=0.0,
            config_used=RepoConfig(),  # Minimal default config
        )

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = ArchitecturePerformanceMetrics(
            total_processing_time=0,
            memory_usage_mb=0,
            cpu_utilization=0,
            processing_counts=ProcessingCounts(
                components_processed=0,
                nodes_analyzed=0,
                edges_processed=0,
            ),
            cache_metrics=ArchitectureCacheMetrics(
                cache_hits=0,
                cache_misses=0,
            ),
            parallel_workers_used=0,
        )

        self.profiling_data.clear()

        logger.info("Performance metrics reset")


# Decorator for performance monitoring
def performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to monitor function performance."""
    func_name = getattr(func, "__name__", repr(func))

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0

        try:
            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.info(
                "%s executed in %.3fs, memory delta: %.1fMB",
                func_name,
                execution_time,
                memory_delta,
            )

            return result

        except Exception as e:
            logger.error("%s failed: %s", func_name, e)
            raise

    return wrapper


# Alias for backward compatibility
PerformanceOptimizer = ArchitecturePerformanceOptimizer
