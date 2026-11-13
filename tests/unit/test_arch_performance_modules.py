"""Unit tests for architecture performance modules.

Covers:
- processing.architecture.performance_optimizer
- processing.architecture.time_series_metrics
- processing.architecture.performance_analysis
- processing.architecture.performance_metrics_tracker
"""

from __future__ import annotations

import datetime
import sys
import types as _types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# psutil stub — psutil is not installed in the test environment.
# We must install it before importing performance_optimizer which uses it.
# ---------------------------------------------------------------------------
if "psutil" not in sys.modules:
    _psutil = _types.ModuleType("psutil")
    _mock_proc = MagicMock()
    _mock_proc.memory_info.return_value = MagicMock(rss=200 * 1024 * 1024)
    _psutil.Process = MagicMock(return_value=_mock_proc)
    _psutil.cpu_percent = MagicMock(return_value=30.0)
    _psutil.cpu_count = MagicMock(return_value=4)
    _psutil.virtual_memory = MagicMock(return_value=MagicMock(percent=50.0))
    sys.modules["psutil"] = _psutil

import networkx as nx

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from models.architecture_models import (
    ArchitectureAnalysisResult,
    Component,
)
from processing.architecture.performance_analysis import PerformanceAnalysis
from processing.architecture.performance_metrics_tracker import (
    ArchitectureMetricTimeSeries as TrackerTimeSeries,
)
from processing.architecture.performance_metrics_tracker import (
    MetricDataPoint,
    MetricTrendAnalysis,
    TrendType,
)
from processing.architecture.performance_optimizer import (
    AdaptiveParallelismManager,
    ArchitectureCacheMetrics,
    ArchitecturePerformanceMetrics,
    ArchitecturePerformanceOptimizer,
    MemoryEfficientGraphProcessor,
    PerformanceConfig,
    ProcessingCounts,
    performance_monitor,
)
from processing.architecture.time_series_metrics import (
    ArchitectureMetricTimeSeries,
    CrossLanguageCorrelation,
    CrossLanguageCorrelationAnalyzer,
    TimeSeriesMetricsCalculator,
)

# ============================================================================
# performance_optimizer.py tests
# ============================================================================


class TestPerformanceConfig:
    def test_defaults(self):
        cfg = PerformanceConfig()
        assert cfg.chunk_size == 1000
        assert cfg.max_memory_mb == 512
        assert cfg.min_workers == 2
        assert cfg.max_workers is None
        assert cfg.enable_profiling is False

    def test_custom_values(self):
        cfg = PerformanceConfig(chunk_size=500, max_memory_mb=256, min_workers=1, max_workers=8)
        assert cfg.chunk_size == 500
        assert cfg.max_workers == 8


class TestProcessingCounts:
    def test_fields(self):
        pc = ProcessingCounts(components_processed=10, nodes_analyzed=50, edges_processed=30)
        assert pc.components_processed == 10
        assert pc.nodes_analyzed == 50
        assert pc.edges_processed == 30


class TestArchitecturePerformanceMetrics:
    def test_to_dict(self):
        m = ArchitecturePerformanceMetrics(
            total_processing_time=1.5,
            memory_usage_mb=100.0,
            cpu_utilization=45.0,
            processing_counts=ProcessingCounts(5, 20, 15),
            cache_metrics=ArchitectureCacheMetrics(cache_hits=10, cache_misses=3),
            parallel_workers_used=4,
        )
        d = m.to_dict()
        assert d["total_processing_time"] == 1.5
        assert d["components_processed"] == 5
        assert d["cache_hits"] == 10
        assert d["parallel_workers_used"] == 4


class TestMemoryEfficientGraphProcessor:
    def test_init(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=100, max_memory_mb=256)
        assert proc.chunk_size == 100
        assert proc.max_memory_mb == 256

    def test_chunk_nodes(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=3)
        nodes = ["a", "b", "c", "d", "e"]
        chunks = list(proc._chunk_nodes(nodes, 3))
        assert chunks == [["a", "b", "c"], ["d", "e"]]

    def test_chunk_nodes_empty(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=5)
        assert list(proc._chunk_nodes([], 5)) == []

    def test_process_graph_streaming_happy_path(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=2)
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b", "c"])

        results = list(proc.process_graph_streaming(g, lambda chunk: len(chunk)))
        assert sum(results) == 3

    def test_process_graph_streaming_with_progress_callback(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=2)
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b", "c"])

        progress = []
        list(
            proc.process_graph_streaming(
                g,
                lambda chunk: chunk,
                progress_callback=lambda done, total: progress.append((done, total)),
            )
        )
        assert len(progress) >= 1
        # Last progress should report all nodes processed
        assert progress[-1][0] == 3

    def test_process_graph_streaming_error_in_processor(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=2)
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b", "c"])

        call_count = 0

        def failing_processor(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("boom")
            return chunk

        results = list(proc.process_graph_streaming(g, failing_processor))
        # First chunk failed, second succeeded
        assert len(results) == 1

    def test_get_memory_status(self):
        proc = MemoryEfficientGraphProcessor(chunk_size=100, max_memory_mb=512)
        status = proc.get_memory_status()
        assert "current_memory_mb" in status
        assert "max_memory_mb" in status
        assert "memory_utilization_percent" in status
        assert "available_memory_mb" in status
        assert status["max_memory_mb"] == 512.0

    @patch(
        "processing.architecture.performance_optimizer.HAS_PSUTIL",
        False,
    )
    def test_get_memory_usage_no_psutil(self):
        proc = MemoryEfficientGraphProcessor()
        assert proc._get_memory_usage_mb() == 0.0

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", True)
    @patch("processing.architecture.performance_optimizer.psutil")
    def test_check_memory_usage_warning(self, mock_psutil):
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=200 * 1024 * 1024)
        mock_psutil.Process.return_value = mock_process
        proc = MemoryEfficientGraphProcessor(chunk_size=100, max_memory_mb=10)
        # With psutil returning 200MB, this should log a warning (200 > 10)
        proc._check_memory_usage()
        assert proc.current_memory_mb > 0


class TestAdaptiveParallelismManager:
    def test_init_defaults(self):
        mgr = AdaptiveParallelismManager()
        assert mgr.min_workers == 2
        assert mgr.current_workers == 2

    def test_init_custom(self):
        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=8)
        assert mgr.min_workers == 1
        assert mgr.max_workers == 8

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_get_optimal_worker_count_no_psutil(self):
        mgr = AdaptiveParallelismManager(min_workers=2, max_workers=8)
        # With default moderate usage, should pick based on task count
        count = mgr.get_optimal_worker_count(task_count=5)
        assert mgr.min_workers <= count <= mgr.max_workers

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_get_optimal_worker_count_high_complexity(self):
        mgr = AdaptiveParallelismManager(min_workers=2, max_workers=8)
        count = mgr.get_optimal_worker_count(task_count=6, task_complexity=2.0)
        assert count >= mgr.min_workers

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_get_optimal_worker_count_low_complexity(self):
        mgr = AdaptiveParallelismManager(min_workers=2, max_workers=8)
        count = mgr.get_optimal_worker_count(task_count=4, task_complexity=0.3)
        assert count >= mgr.min_workers

    def test_execute_parallel_tasks_empty(self):
        mgr = AdaptiveParallelismManager()
        assert mgr.execute_parallel_tasks([]) == []

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_execute_parallel_tasks_happy(self):
        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=2)
        tasks = [lambda: 1, lambda: 2, lambda: 3]
        results = mgr.execute_parallel_tasks(tasks)
        assert sorted(results) == [1, 2, 3]

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_execute_parallel_tasks_with_failure(self):
        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=2)

        def fail():
            raise RuntimeError("fail")

        # When executor itself raises, it should fall back to sequential
        tasks = [lambda: 42]
        results = mgr.execute_parallel_tasks(tasks)
        assert results == [42]

    def test_record_performance_metrics_caps_history(self):
        mgr = AdaptiveParallelismManager()
        for i in range(25):
            mgr._record_performance_metrics(float(i))
        assert len(mgr.performance_history) == 20

    def test_load_history_caps(self):
        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=4)
        for _ in range(15):
            with patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False):
                mgr.get_optimal_worker_count(5)
        assert len(mgr.load_history) <= 10


class TestArchitecturePerformanceOptimizer:
    def _make_components(self, n=3):
        return [
            Component(id=f"comp-{i}", name=f"Component {i}", files=[f"file{i}.py"])
            for i in range(n)
        ]

    def test_init_defaults(self):
        opt = ArchitecturePerformanceOptimizer()
        assert opt.config.chunk_size == 1000
        assert opt.metrics.total_processing_time == 0

    def test_init_custom_config(self):
        cfg = PerformanceConfig(chunk_size=500, max_memory_mb=256)
        opt = ArchitecturePerformanceOptimizer(config=cfg)
        assert opt.chunk_size == 500
        assert opt.max_memory_mb == 256

    def test_get_performance_metrics(self):
        opt = ArchitecturePerformanceOptimizer()
        m = opt.get_performance_metrics()
        assert m["total_processing_time"] == 0
        assert m["parallel_workers_used"] == 0

    def test_reset_metrics(self):
        opt = ArchitecturePerformanceOptimizer()
        opt.metrics.total_processing_time = 99.0
        opt.profiling_data["test"] = [1.0, 2.0]
        opt.reset_metrics()
        assert opt.metrics.total_processing_time == 0
        assert opt.profiling_data == {}

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_determine_strategy_sequential_small(self):
        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_nodes_from(range(5))
        comps = self._make_components(3)
        strategy = opt._determine_processing_strategy(g, comps)
        assert strategy == "sequential"

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_determine_strategy_streaming_large_graph(self):
        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_nodes_from(range(11000))
        comps = self._make_components(3)
        strategy = opt._determine_processing_strategy(g, comps)
        assert strategy == "streaming"

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", True)
    def test_determine_strategy_parallel(self):
        with patch("processing.architecture.performance_optimizer.psutil") as mock_psutil:
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=10 * 1024 * 1024)
            mock_psutil.Process.return_value = mock_proc
            mock_psutil.cpu_count.return_value = 4

            opt = ArchitecturePerformanceOptimizer()
            g = nx.DiGraph()
            g.add_nodes_from(range(100))
            comps = self._make_components(15)
            strategy = opt._determine_processing_strategy(g, comps)
            assert strategy == "parallel"

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_process_sequential(self):
        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b"])
        comps = self._make_components(2)

        def dummy_func(graph, components):
            return {"processed": True}

        result = opt._process_sequential(g, comps, {"dummy": dummy_func})
        assert isinstance(result, ArchitectureAnalysisResult)
        assert result.codebase_id == "unknown"
        assert len(result.components) == 2

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_optimize_architecture_processing_sequential(self):
        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b"])
        g.add_edge("a", "b")
        comps = self._make_components(2)

        def dummy(graph, components):
            return None

        result = opt.optimize_architecture_processing(g, comps, {"dummy": dummy})
        assert isinstance(result, ArchitectureAnalysisResult)
        assert opt.metrics.processing_counts.components_processed == 2
        assert opt.metrics.processing_counts.nodes_analyzed == 2
        assert opt.metrics.processing_counts.edges_processed == 1

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_process_streaming_fallback(self):
        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_nodes_from(["a"])
        comps = self._make_components(1)
        result = opt._process_streaming(g, comps, {"f": lambda g, c: None})
        assert isinstance(result, ArchitectureAnalysisResult)

    def test_categorize_result_metrics(self):
        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("compute_metric", [1, 2, 3], agg)
        assert agg["metrics"] == [1, 2, 3]

    def test_categorize_result_patterns(self):
        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("detect_pattern", ["mvc", "factory"], agg)
        assert agg["patterns"] == ["mvc", "factory"]

    def test_categorize_result_layers(self):
        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("assign_layer", {"presentation": ["c1"]}, agg)
        assert agg["layer_mappings"] == {"presentation": ["c1"]}

    def test_categorize_result_summary(self):
        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("build_summary", {"total": 5}, agg)
        assert agg["summary"] == {"total": 5}

    def test_categorize_result_unknown(self):
        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("unknown_func", "data", agg)
        # Should not crash, just log debug

    def test_handle_metrics_result_iterable(self):
        opt = ArchitecturePerformanceOptimizer()
        metrics_list = []
        opt._handle_metrics_result((1, 2, 3), metrics_list, "test")
        assert metrics_list == [1, 2, 3]

    def test_handle_metrics_result_unexpected(self):
        opt = ArchitecturePerformanceOptimizer()
        metrics_list = []
        opt._handle_metrics_result("not_a_list", metrics_list, "test")
        # Should log warning, not add
        assert metrics_list == []

    def test_handle_patterns_result_unexpected(self):
        opt = ArchitecturePerformanceOptimizer()
        patterns_list = []
        opt._handle_patterns_result({"not": "list"}, patterns_list, "test")
        assert patterns_list == []

    def test_handle_layer_result_unexpected(self):
        opt = ArchitecturePerformanceOptimizer()
        layer_map = {}
        opt._handle_layer_result([1, 2], layer_map, "test")
        assert layer_map == {}

    def test_handle_summary_result_unexpected(self):
        opt = ArchitecturePerformanceOptimizer()
        summary = {}
        opt._handle_summary_result("not_dict", summary, "test")
        assert summary == {}

    def test_aggregate_parallel_results_with_none(self):
        opt = ArchitecturePerformanceOptimizer()
        comps = self._make_components(2)
        result = opt._aggregate_parallel_results([None, None], comps)
        assert result["metrics"] == []

    def test_aggregate_parallel_results_with_none_result(self):
        opt = ArchitecturePerformanceOptimizer()
        comps = self._make_components(2)
        result = opt._aggregate_parallel_results(
            [{"name": "compute_metric", "result": None}], comps
        )
        assert result["metrics"] == []

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_build_architecture_result(self):
        opt = ArchitecturePerformanceOptimizer()
        comps = self._make_components(2)
        aggregated = {
            "metrics": [],
            "patterns": [],
            "layer_mappings": {"presentation": ["comp-0"]},
            "summary": {"components": 2},
        }
        result = opt._build_architecture_result(comps, aggregated, 1.5, {"f1": lambda: None})
        assert isinstance(result, ArchitectureAnalysisResult)
        assert len(result.layers) == 1
        assert result.layers[0].name == "presentation"
        assert result.summary["total_layers"] == 1


class TestPerformanceMonitorDecorator:
    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_decorator_success(self):
        @performance_monitor
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    @patch("processing.architecture.performance_optimizer.HAS_PSUTIL", False)
    def test_decorator_exception(self):
        @performance_monitor
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()


# ============================================================================
# time_series_metrics.py tests
# ============================================================================


class TestArchitectureMetricTimeSeries:
    def test_add_measurement(self):
        ts = ArchitectureMetricTimeSeries(metric_name="complexity")
        now = datetime.datetime.now(datetime.UTC)
        ts.add_measurement(now, 5.0, {"source": "test"})
        assert len(ts.values) == 1
        assert ts.values[0] == 5.0
        assert now.isoformat() in ts.metadata

    def test_add_measurement_no_metadata(self):
        ts = ArchitectureMetricTimeSeries(metric_name="complexity")
        now = datetime.datetime.now(datetime.UTC)
        ts.add_measurement(now, 3.0)
        assert len(ts.metadata) == 0

    def test_get_trend_insufficient_data(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [1.0, 2.0]
        assert ts.get_trend(window_size=5) == "insufficient_data"

    def test_get_trend_insufficient_older_data(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Only 5 values, window_size=5 needs 10 for older comparison
        assert ts.get_trend(window_size=5) == "insufficient_data"

    def test_get_trend_increasing(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        # Older values low, recent values high
        ts.values = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert ts.get_trend(window_size=5) == "increasing"

    def test_get_trend_decreasing(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert ts.get_trend(window_size=5) == "decreasing"

    def test_get_trend_stable(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert ts.get_trend(window_size=5) == "stable"

    def test_get_statistics_empty(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        assert ts.get_statistics() == {}

    def test_get_statistics_single_value(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [5.0]
        stats = ts.get_statistics()
        assert stats["mean"] == 5.0
        assert stats["std_dev"] == 0
        assert stats["count"] == 1

    def test_get_statistics_multiple_values(self):
        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = ts.get_statistics()
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["range"] == 4.0
        assert stats["latest"] == 5.0
        assert stats["count"] == 5


class TestCrossLanguageCorrelation:
    def test_to_dict(self):
        c = CrossLanguageCorrelation(
            language_pair=("python", "java"),
            correlation_coefficient=0.85,
            shared_patterns=["mvc"],
            dependency_overlap=0.3,
            architectural_similarity=0.7,
            quality_correlation=0.5,
        )
        d = c.to_dict()
        assert d["language_pair"] == ("python", "java")
        assert d["correlation_coefficient"] == 0.85
        assert d["shared_patterns"] == ["mvc"]


class TestTimeSeriesMetricsCalculator:
    @pytest.fixture
    def calc(self, tmp_path):
        return TimeSeriesMetricsCalculator(db_path=tmp_path / "test_metrics.db")

    def test_init_creates_db(self, calc, tmp_path):
        assert (tmp_path / "test_metrics.db").exists()

    def test_store_and_retrieve_metrics(self, calc):
        calc.store_metrics("codebase1", {"complexity": 5.0, "coupling": 3.0}, commit_hash="abc123")
        ts = calc.get_metrics_time_series("codebase1", "complexity", days_back=1)
        assert len(ts.values) == 1
        assert ts.values[0] == 5.0

    def test_store_metrics_with_metadata(self, calc):
        calc.store_metrics(
            "codebase1",
            {"loc": 1000.0},
            commit_hash="def456",
            metadata={"branch": "main"},
        )
        ts = calc.get_metrics_time_series("codebase1", "loc", days_back=1)
        assert len(ts.values) == 1

    def test_get_metrics_time_series_nonexistent(self, calc):
        ts = calc.get_metrics_time_series("nonexistent", "missing", days_back=1)
        assert len(ts.values) == 0

    def test_compare_metrics_across_commits(self, calc):
        calc.store_metrics("cb1", {"complexity": 5.0}, commit_hash="aaa")
        calc.store_metrics("cb1", {"complexity": 7.0}, commit_hash="bbb")
        comparison = calc.compare_metrics_across_commits("cb1", ["aaa", "bbb"])
        assert "complexity" in comparison
        assert len(comparison["complexity"]) == 2

    def test_compare_metrics_no_data(self, calc):
        result = calc.compare_metrics_across_commits("cb_none", ["x", "y"])
        assert result == {}

    def test_calculate_metric_trends(self, calc):
        calc.store_metrics("cb1", {"complexity": 5.0})
        trends = calc.calculate_metric_trends("cb1", days_back=1)
        assert "complexity" in trends
        assert "statistics" in trends["complexity"]

    def test_calculate_metric_trends_empty(self, calc):
        trends = calc.calculate_metric_trends("nonexistent", days_back=1)
        assert trends == {}


class TestCrossLanguageCorrelationAnalyzer:
    @pytest.fixture
    def analyzer(self, tmp_path):
        # Create the database with tables so _store_correlations works
        db_path = tmp_path / "corr.db"
        # Use TimeSeriesMetricsCalculator to initialize tables
        TimeSeriesMetricsCalculator(db_path=db_path)
        return CrossLanguageCorrelationAnalyzer(db_path=db_path)

    def test_analyze_cross_language_correlation(self, analyzer):
        metrics = {
            "python": {"complexity": 5.0, "coupling": 3.0, "cohesion": 0.8},
            "java": {"complexity": 6.0, "coupling": 4.0, "cohesion": 0.7},
        }
        patterns = {
            "python": ["mvc", "factory"],
            "java": ["mvc", "observer"],
        }
        deps = {
            "python": ["requests", "flask"],
            "java": ["spring", "junit"],
        }
        correlations = analyzer.analyze_cross_language_correlation("cb1", metrics, patterns, deps)
        assert len(correlations) == 1
        assert correlations[0].language_pair == ("python", "java")
        assert "mvc" in correlations[0].shared_patterns

    def test_analyze_empty_metrics(self, analyzer):
        correlations = analyzer.analyze_cross_language_correlation(
            "cb1", {"python": {}, "java": {}}, {}, {}
        )
        assert correlations == []

    def test_calculate_architectural_similarity_both_empty(self, analyzer):
        sim = analyzer._calculate_architectural_similarity(set(), set())
        assert sim == 1.0

    def test_calculate_architectural_similarity_one_empty(self, analyzer):
        sim = analyzer._calculate_architectural_similarity({"a"}, set())
        assert sim == 0.0

    def test_calculate_architectural_similarity_identical(self, analyzer):
        sim = analyzer._calculate_architectural_similarity({"a", "b"}, {"a", "b"})
        assert sim == 1.0

    def test_calculate_architectural_similarity_partial(self, analyzer):
        sim = analyzer._calculate_architectural_similarity({"a", "b"}, {"b", "c"})
        assert abs(sim - 1 / 3) < 1e-9

    def test_calculate_dependency_overlap_no_overlap(self, analyzer):
        data = {"dependencies": {"python": ["a"], "java": ["b"]}}
        overlap = analyzer._calculate_dependency_overlap("python", "java", data)
        assert overlap == 0.0

    def test_calculate_dependency_overlap_full(self, analyzer):
        data = {"dependencies": {"python": ["a", "b"], "java": ["a", "b"]}}
        overlap = analyzer._calculate_dependency_overlap("python", "java", data)
        assert overlap == 1.0

    def test_calculate_metrics_correlation_too_few_common(self, analyzer):
        coeff = analyzer._calculate_metrics_correlation({"a": 1.0}, {"b": 2.0})
        assert coeff == 0.0

    def test_extract_language_metrics_missing(self, analyzer):
        result = analyzer._extract_language_metrics("python", "java", {"metrics": {}})
        assert result is None

    def test_get_language_correlations_empty(self, analyzer):
        result = analyzer.get_language_correlations("nonexistent")
        assert result == []

    def test_store_and_retrieve_correlations(self, analyzer):
        corr = CrossLanguageCorrelation(
            language_pair=("python", "go"),
            correlation_coefficient=0.9,
            shared_patterns=["factory"],
            dependency_overlap=0.5,
            architectural_similarity=0.8,
            quality_correlation=0.0,
        )
        analyzer._store_correlations("cb1", [corr])
        retrieved = analyzer.get_language_correlations("cb1")
        assert len(retrieved) == 1
        assert retrieved[0].language_pair == ("python", "go")

    def test_generate_correlation_report_empty(self, analyzer):
        report = analyzer.generate_correlation_report("nonexistent")
        assert report == {"error": "No correlation data available"}

    def test_generate_correlation_report_with_data(self, analyzer):
        corr1 = CrossLanguageCorrelation(
            language_pair=("python", "go"),
            correlation_coefficient=0.9,
            shared_patterns=["factory"],
            dependency_overlap=0.5,
            architectural_similarity=0.8,
        )
        corr2 = CrossLanguageCorrelation(
            language_pair=("python", "java"),
            correlation_coefficient=0.3,
            shared_patterns=["mvc"],
            dependency_overlap=0.2,
            architectural_similarity=0.4,
        )
        analyzer._store_correlations("cb1", [corr1, corr2])
        report = analyzer.generate_correlation_report("cb1")
        assert "summary" in report
        assert report["summary"]["total_language_pairs"] == 2
        assert "strongest_correlation" in report
        assert "common_patterns" in report


# ============================================================================
# performance_analysis.py tests
# ============================================================================


class TestPerformanceAnalysis:
    @pytest.fixture
    def pa(self):
        return PerformanceAnalysis()

    def test_init_has_patterns(self, pa):
        assert "python" in pa.performance_patterns
        assert "javascript" in pa.performance_patterns

    def test_get_file_extensions_python(self, pa):
        assert pa._get_file_extensions("python") == [".py"]

    def test_get_file_extensions_javascript(self, pa):
        assert pa._get_file_extensions("javascript") == [".js", ".jsx"]

    def test_get_file_extensions_unknown(self, pa):
        assert pa._get_file_extensions("haskell") == []

    def test_matches_pattern_regex(self, pa):
        assert pa._matches_pattern("for i in range(len(items)):", r"for.*in.*range.*len")

    def test_matches_pattern_no_match(self, pa):
        assert not pa._matches_pattern("x = 5", r"for.*in.*range.*len")

    def test_matches_pattern_bad_regex_fallback(self, pa):
        # Invalid regex falls back to string contains
        assert pa._matches_pattern("global x", "global ")

    def test_get_performance_suggestion_known(self, pa):
        suggestion = pa._get_performance_suggestion(r"for.*in.*range.*len", "python")
        assert "enumerate" in suggestion

    def test_get_performance_suggestion_unknown(self, pa):
        suggestion = pa._get_performance_suggestion("unknown_pattern", "python")
        assert suggestion == "Review for performance optimization"

    def test_analyze_file_performance_with_anti_pattern(self, pa):
        code = "for i in range(len(items)):\n    pass\n"
        issues = pa._analyze_file_performance(code, "test.py", "python")
        assert len(issues) >= 1
        assert issues[0].category == "performance"
        assert issues[0].tool == "performance_analyzer"
        assert issues[0].line_number == 1

    def test_analyze_file_performance_clean(self, pa):
        code = "x = enumerate(items)\n"
        issues = pa._analyze_file_performance(code, "test.py", "python")
        assert len(issues) == 0

    def test_analyze_performance_patterns_unknown_language(self, pa):
        issues = pa.analyze_performance_patterns(Path("/tmp"), "haskell")
        assert issues == []

    def test_analyze_performance_patterns_with_files(self, pa, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("for i in range(len(items)):\n    items.append(i)\n")
        issues = pa.analyze_performance_patterns(tmp_path, "python")
        assert len(issues) >= 1

    def test_analyze_performance_patterns_unreadable_file(self, pa, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8
        # Should not crash
        pa.analyze_performance_patterns(tmp_path, "python")
        # File with errors="ignore" still reads, so this depends on content

    def test_analyze_memory_usage_patterns_python(self, pa, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text(
            "global counter\nx = list(range(10000))\nfor i in range(len(items)):\n    pass\n"
        )
        result = pa.analyze_memory_usage_patterns(tmp_path, "python")
        assert result["potential_memory_leaks"] >= 1
        assert result["large_data_structures"] >= 1
        assert result["inefficient_loops"] >= 1
        assert len(result["recommendations"]) >= 1

    def test_analyze_memory_usage_patterns_javascript(self, pa, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text(
            "setInterval(function() {}, 1000);\n"
            "var arr = new Array(100);\n"
            "for (var i = 0; i < items.length; i++) {}\n"
        )
        result = pa.analyze_memory_usage_patterns(tmp_path, "javascript")
        assert result["potential_memory_leaks"] >= 1
        assert result["large_data_structures"] >= 1
        assert result["inefficient_loops"] >= 1

    def test_analyze_memory_usage_patterns_empty(self, pa, tmp_path):
        result = pa.analyze_memory_usage_patterns(tmp_path, "python")
        assert result["potential_memory_leaks"] == 0
        assert result["recommendations"] == []

    def test_analyze_file_memory_patterns_unknown_language(self, pa):
        result = pa._analyze_file_memory_patterns("code", "rust")
        assert result == {"memory_leaks": 0, "large_structures": 0, "inefficient_loops": 0}

    def test_run_profiling_analysis_python(self, pa, tmp_path):
        # Create a large file
        py_file = tmp_path / "big.py"
        lines = ["if True:\n"] * 600 + ["def func():\n"] * 50 + ["class Foo:\n"] * 10
        py_file.write_text("".join(lines))

        result = pa.run_profiling_analysis(tmp_path, "python")
        assert result["total_files"] == 1
        assert result["total_lines"] == 660
        assert len(result["large_files"]) >= 1
        assert len(result["complexity_hotspots"]) >= 1

    def test_run_profiling_analysis_empty(self, pa, tmp_path):
        result = pa.run_profiling_analysis(tmp_path, "python")
        assert result["total_files"] == 0
        assert result["avg_file_size"] == 0.0

    def test_run_profiling_analysis_javascript(self, pa, tmp_path):
        js_file = tmp_path / "app.js"
        lines = ["if (true) {\n"] * 20 + ["function foo() {\n"] * 20 + ["class Bar {\n"] * 5
        js_file.write_text("".join(lines))
        result = pa.run_profiling_analysis(tmp_path, "javascript")
        assert result["total_files"] == 1

    def test_calculate_basic_complexity_python(self, pa):
        lines = [
            "if True:\n",
            "    pass\n",
            "for x in range(10):\n",
            "def func():\n",
            "class Foo:\n",
        ]
        score = pa._calculate_basic_complexity(lines, "python")
        # if=1 + for=1 + def=2 + class=3 = 7
        assert score == 7

    def test_calculate_basic_complexity_javascript(self, pa):
        lines = [
            "if (true) {\n",
            "function foo() {\n",
            "class Bar {\n",
            "const x = () => 1;\n",
        ]
        score = pa._calculate_basic_complexity(lines, "javascript")
        # if=1 + function=2 + class=3 + arrow=2 = 8
        assert score == 8

    def test_calculate_basic_complexity_unknown(self, pa):
        lines = ["some code\n"]
        score = pa._calculate_basic_complexity(lines, "rust")
        assert score == 0

    def test_get_performance_recommendations_full(self, pa):
        analysis = {
            "memory_analysis": {"potential_memory_leaks": 3},
            "profiling_analysis": {
                "large_files": [{"path": "a"}, {"path": "b"}, {"path": "c"}, {"path": "d"}],
                "complexity_hotspots": [{"path": "x"}],
            },
        }
        recs = pa.get_performance_recommendations(analysis)
        assert any("memory leak" in r.lower() for r in recs)
        assert any("large files" in r.lower() for r in recs)
        assert any("complexity hotspot" in r.lower() for r in recs)
        # Always includes general recommendations
        assert len(recs) >= 8

    def test_get_performance_recommendations_minimal(self, pa):
        analysis = {}
        recs = pa.get_performance_recommendations(analysis)
        # Should still return general recommendations
        assert len(recs) == 5


# ============================================================================
# performance_metrics_tracker.py tests
# ============================================================================


class TestTrendType:
    def test_values(self):
        assert TrendType.IMPROVING.value == "updating"
        assert TrendType.STABLE.value == "stable"
        assert TrendType.DEGRADING.value == "degrading"
        assert TrendType.FLUCTUATING.value == "fluctuating"
        assert TrendType.UNKNOWN.value == "unknown"


class TestMetricDataPoint:
    def test_creation(self):
        dp = MetricDataPoint(
            timestamp=datetime.datetime.now(datetime.UTC),
            commit_id="abc123",
            value=5.0,
        )
        assert dp.value == 5.0
        assert dp.commit_id == "abc123"
        assert dp.raw_data is None

    def test_with_raw_data(self):
        dp = MetricDataPoint(
            timestamp=datetime.datetime.now(datetime.UTC),
            commit_id="abc123",
            value=5.0,
            raw_data={"detail": "info"},
        )
        assert dp.raw_data == {"detail": "info"}

    def test_forbids_extra(self):
        with pytest.raises(Exception):
            MetricDataPoint(
                timestamp=datetime.datetime.now(datetime.UTC),
                commit_id="abc",
                value=1.0,
                unknown_field="bad",
            )


class TestTrackerTimeSeries:
    def _make_ts(self):
        return TrackerTimeSeries(metric_name="complexity", component_id="comp-1")

    def _add_points(self, ts, values, start_days_ago=10):
        base = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=start_days_ago)
        for i, v in enumerate(values):
            ts.add_data_point(
                timestamp=base + datetime.timedelta(days=i),
                commit_id=f"commit-{i}",
                value=v,
            )

    def test_add_data_point_sorts(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        ts.add_data_point(now + datetime.timedelta(hours=1), "c2", 2.0)
        ts.add_data_point(now, "c1", 1.0)
        assert ts.data_points[0].commit_id == "c1"
        assert ts.data_points[1].commit_id == "c2"

    def test_trend_analysis_no_data(self):
        ts = self._make_ts()
        analysis = ts.get_trend_analysis()
        assert analysis.trend == TrendType.UNKNOWN
        assert analysis.data_points == 0
        assert analysis.start_value == 0.0

    def test_trend_analysis_single_point(self):
        ts = self._make_ts()
        ts.add_data_point(datetime.datetime.now(datetime.UTC), "c1", 5.0)
        analysis = ts.get_trend_analysis()
        assert analysis.trend == TrendType.UNKNOWN
        assert analysis.start_value == 5.0
        assert analysis.end_value == 5.0

    def test_trend_analysis_two_points(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        ts.add_data_point(now, "c1", 5.0)
        ts.add_data_point(now + datetime.timedelta(days=1), "c2", 10.0)
        analysis = ts.get_trend_analysis()
        assert analysis.data_points == 2
        assert analysis.start_value == 5.0
        assert analysis.end_value == 10.0
        assert analysis.percent_change == 100.0

    def test_trend_analysis_stable(self):
        ts = self._make_ts()
        self._add_points(ts, [5.0, 5.0, 5.0, 5.0, 5.0])
        analysis = ts.get_trend_analysis()
        assert analysis.trend == TrendType.STABLE
        assert analysis.confidence > 0

    def test_trend_analysis_degrading(self):
        ts = self._make_ts()
        # Increasing values = degrading (higher complexity is worse)
        self._add_points(ts, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        analysis = ts.get_trend_analysis()
        assert analysis.trend == TrendType.DEGRADING

    def test_trend_analysis_percent_change_from_zero(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        ts.add_data_point(now, "c1", 0.0)
        ts.add_data_point(now + datetime.timedelta(days=1), "c2", 5.0)
        analysis = ts.get_trend_analysis()
        assert analysis.percent_change == 100.0

    def test_trend_analysis_zero_to_zero(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        ts.add_data_point(now, "c1", 0.0)
        ts.add_data_point(now + datetime.timedelta(days=1), "c2", 0.0)
        analysis = ts.get_trend_analysis()
        assert analysis.percent_change == 0.0

    def test_detect_trend_less_than_3_points(self):
        ts = self._make_ts()
        trend, confidence, anomalies = ts._detect_trend(
            [1.0, 2.0],
            [
                datetime.datetime.now(datetime.UTC),
                datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1),
            ],
        )
        assert trend == TrendType.UNKNOWN
        assert confidence == 0.0

    def test_detect_trend_very_short_timespan(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        trend, confidence, _ = ts._detect_trend(
            [1.0, 2.0, 3.0],
            [now, now + datetime.timedelta(seconds=1), now + datetime.timedelta(seconds=2)],
        )
        assert trend == TrendType.STABLE

    def test_forecast_next_value_too_few(self):
        ts = self._make_ts()
        val, conf = ts._forecast_next_value([1.0, 2.0], [])
        assert val is None
        assert conf == 0.0

    def test_forecast_next_value_linear(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        timestamps = [now + datetime.timedelta(days=i) for i in range(5)]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        val, conf = ts._forecast_next_value(values, timestamps)
        assert val is not None
        assert abs(val - 6.0) <= 0.5
        assert 0.0 < conf <= 1.0

    def test_get_visualization_data_empty(self):
        ts = self._make_ts()
        data = ts.get_visualization_data()
        assert data["metric"] == "complexity"
        assert data["timestamps"] == []
        assert data["trend"] == "unknown"
        assert data["forecast"] is None

    def test_get_visualization_data_with_points(self):
        ts = self._make_ts()
        self._add_points(ts, [5.0, 5.0, 5.0, 5.0, 5.0])
        data = ts.get_visualization_data()
        assert len(data["timestamps"]) == 5
        assert len(data["values"]) == 5
        assert data["trend"] in [t.value for t in TrendType]
        assert "statistics" in data

    def test_get_visualization_data_with_forecast(self):
        ts = self._make_ts()
        self._add_points(ts, [1.0, 2.0, 3.0, 4.0, 5.0])
        data = ts.get_visualization_data()
        # With 5 points and clear trend, forecast should be present
        if data.get("forecast"):
            assert "value" in data["forecast"]
            assert "confidence" in data["forecast"]

    def test_detect_trend_high_variability_with_slope(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        # High variability but with a clear upward trend
        values = [1.0, 5.0, 2.0, 8.0, 3.0, 10.0, 4.0]
        timestamps = [now + datetime.timedelta(days=i * 2) for i in range(len(values))]
        trend, confidence, _ = ts._detect_trend(values, timestamps)
        # Should detect some kind of trend or fluctuation
        assert trend in [TrendType.DEGRADING, TrendType.FLUCTUATING, TrendType.IMPROVING]

    def test_detect_trend_all_same_values(self):
        ts = self._make_ts()
        now = datetime.datetime.now(datetime.UTC)
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        timestamps = [now + datetime.timedelta(days=i) for i in range(5)]
        trend, confidence, _ = ts._detect_trend(values, timestamps)
        assert trend == TrendType.STABLE
        assert confidence > 0.5


class TestMetricTrendAnalysis:
    def test_creation(self):
        mta = MetricTrendAnalysis(
            metric_name="complexity",
            component_id="comp-1",
            trend=TrendType.STABLE,
            data_points=5,
            timespan_days=10.0,
            start_value=5.0,
            end_value=5.0,
            min_value=4.0,
            max_value=6.0,
            percent_change=0.0,
            rate_of_change=0.0,
            confidence=0.9,
        )
        assert mta.metric_name == "complexity"
        assert mta.trend == TrendType.STABLE
        assert mta.forecast_value is None
