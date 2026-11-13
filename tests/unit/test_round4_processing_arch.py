"""Round 4 tests for processing/architecture modules.

Covers: component_extractor, component_extractor_helpers, dependency_graph,
dependency_visualization, performance_optimizer, architecture_processor,
graph_adapter, time_series_metrics.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from models.architecture_models import (
    Component,
    ExternalPackage,
    RepoConfig,
    Vulnerability,
)

# ---------------------------------------------------------------------------
# ComponentExtractor tests
# ---------------------------------------------------------------------------


class TestComponentExtractor:
    """Tests for ComponentExtractor."""

    def _make_extractor(self):
        from processing.architecture.component_extractor import ComponentExtractor

        return ComponentExtractor(max_workers=2, min_parallel_threshold=5000)

    def _make_graph(self, nodes: list[dict]):
        g = nx.DiGraph()
        for n in nodes:
            g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
        return g

    def test_get_default_worker_count(self):
        ext = self._make_extractor()
        assert ext.max_workers >= 1

    def test_get_default_worker_count_from_env(self):
        with patch.dict("os.environ", {"ARCH_PROCESS_POOL_SIZE": "4"}):
            from processing.architecture.component_extractor import ComponentExtractor

            ext = ComponentExtractor()
            assert ext.max_workers == 4

    def test_get_default_worker_count_invalid_env(self):
        with patch.dict("os.environ", {"ARCH_PROCESS_POOL_SIZE": "bad"}):
            from processing.architecture.component_extractor import ComponentExtractor

            ext = ComponentExtractor()
            assert ext.max_workers >= 1

    def test_matches_pattern(self):
        ext = self._make_extractor()
        assert ext._matches_pattern("src/apps/auth/main.py", "src/apps/**") is True
        assert ext._matches_pattern("tests/test_main.py", "src/**") is False

    def test_get_component_id_from_pattern_with_glob(self):
        ext = self._make_extractor()
        cid = ext._get_component_id_from_pattern("src/apps/auth/main.py", "src/apps/**")
        assert "auth" in cid

    def test_get_component_id_from_pattern_fallback(self):
        ext = self._make_extractor()
        cid = ext._get_component_id_from_pattern("src/module/file.py", "*.py")
        assert "component_" in cid

    def test_get_component_name(self):
        ext = self._make_extractor()
        assert ext._get_component_name("component_src_apps_auth") == "Src Apps Auth"
        assert ext._get_component_name("custom_name") == "custom_name"

    def test_estimate_graph_size_empty(self):
        ext = self._make_extractor()
        g = nx.DiGraph()
        assert ext._estimate_graph_size(g) == 0.0

    def test_estimate_graph_size_with_data(self):
        ext = self._make_extractor()
        g = nx.DiGraph()
        for i in range(10):
            g.add_node(f"node_{i}", file_path=f"src/file_{i}.py", code="x" * 100)
        for i in range(9):
            g.add_edge(f"node_{i}", f"node_{i + 1}")
        size = ext._estimate_graph_size(g)
        assert size > 0

    def test_deduplicate_components(self):
        ext = self._make_extractor()
        c1 = Component(id="c1", name="C1", files=["a.py", "b.py"], dependencies=[])
        c2 = Component(id="c1", name="C1", files=["b.py", "c.py"], dependencies=[])
        result = ext._deduplicate_components([c1, c2])
        assert len(result) == 1
        assert "c.py" in result[0].files
        assert "a.py" in result[0].files

    def test_extract_by_folder_sequential(self):
        ext = self._make_extractor()
        g = self._make_graph(
            [
                {"id": "n1", "file_path": "src/auth/login.py"},
                {"id": "n2", "file_path": "src/auth/register.py"},
                {"id": "n3", "file_path": "src/api/handler.py"},
            ]
        )
        components = ext._extract_by_folder_sequential(g, ["src/auth/*", "src/api/*"])
        assert len(components) >= 1

    def test_extract_by_build_files_sequential(self):
        ext = self._make_extractor()
        g = self._make_graph(
            [
                {"id": "n1", "file_path": "frontend/package.json"},
                {"id": "n2", "file_path": "frontend/src/app.js"},
            ]
        )
        components = ext._extract_by_build_files_sequential(g, ["package.json"])
        assert len(components) >= 1

    def test_extract_by_annotations_sequential(self):
        ext = self._make_extractor()
        g = self._make_graph(
            [
                {
                    "id": "n1",
                    "file_path": "src/auth.py",
                    "code": "# component: auth_service\nclass Auth: pass",
                },
            ]
        )
        components = ext._extract_by_annotations_sequential(g)
        # Should find component from comment annotation
        assert isinstance(components, list)

    def test_resolve_dependencies(self):
        ext = self._make_extractor()
        g = nx.DiGraph()
        g.add_node("n1", file_path="a.py")
        g.add_node("n2", file_path="b.py")
        g.add_edge("n1", "n2")
        c1 = Component(id="c1", name="C1", files=["a.py"], dependencies=[])
        c2 = Component(id="c2", name="C2", files=["b.py"], dependencies=[])
        resolved = ext._resolve_dependencies([c1, c2], g)
        assert "c2" in resolved[0].dependencies

    def test_get_extraction_stats(self):
        ext = self._make_extractor()
        c1 = Component(id="c1", name="C1", files=["a.py", "b.py"], dependencies=[])
        stats = ext.get_extraction_stats([c1])
        assert stats.components_found == 1
        assert stats.edges_scanned == 2


# ---------------------------------------------------------------------------
# ComponentExtractionHelpers tests
# ---------------------------------------------------------------------------


class TestComponentExtractionHelpers:
    def test_get_annotation_patterns(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        patterns = ComponentExtractionHelpers.get_annotation_patterns()
        assert len(patterns) > 0
        assert all(len(p) == 2 for p in patterns)

    def test_get_node_code_direct(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        assert ComponentExtractionHelpers.get_node_code({"code": "hello"}) == "hello"

    def test_get_node_code_nested(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        assert (
            ComponentExtractionHelpers.get_node_code({"properties": {"code": "nested"}}) == "nested"
        )

    def test_get_node_code_empty(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        assert ComponentExtractionHelpers.get_node_code({}) == ""

    def test_matches_language(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        assert ComponentExtractionHelpers.matches_language("test.py", "python") is True
        assert ComponentExtractionHelpers.matches_language("test.js", "javascript") is True
        assert ComponentExtractionHelpers.matches_language("test.py", "java") is False
        assert ComponentExtractionHelpers.matches_language("test.rs", "rust") is True
        assert ComponentExtractionHelpers.matches_language("test.go", "go") is True

    def test_extract_component_annotation_python(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        code = '@component("auth_service")\nclass Auth: pass'
        name = ComponentExtractionHelpers.extract_component_annotation(code)
        assert name == "auth_service"

    def test_extract_component_annotation_comment(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        code = "# Component: my-component\nclass X: pass"
        name = ComponentExtractionHelpers.extract_component_annotation(code)
        assert name == "my-component"

    def test_extract_component_annotation_empty(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        assert ComponentExtractionHelpers.extract_component_annotation("") is None
        assert ComponentExtractionHelpers.extract_component_annotation(None) is None

    def test_create_build_components(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        dir_to_files = {"/app/frontend": {"a.js", "b.js"}, "/app/backend": set()}
        components = ComponentExtractionHelpers.create_build_components(
            dir_to_files, lambda cid: cid.replace("_", " ").title()
        )
        assert len(components) == 1  # Only the one with files
        assert components[0].id.startswith("component_")

    def test_create_annotation_components(self):
        from processing.architecture.component_extractor_helpers import ComponentExtractionHelpers

        cf = {"auth": {"a.py", "b.py"}, "empty": set()}
        result = ComponentExtractionHelpers.create_annotation_components(cf)
        assert len(result) == 1
        assert result[0].name == "auth"


# ---------------------------------------------------------------------------
# DependencyGraphBuilder tests
# ---------------------------------------------------------------------------


class TestDependencyGraphBuilder:
    """Tests for DependencyGraphBuilder."""

    def _make_builder(self):
        from processing.architecture.dependency_graph import DependencyGraphBuilder

        return DependencyGraphBuilder(timeout=5)

    def test_parse_pypi_requirement_simple(self):
        builder = self._make_builder()
        name, version = builder._parse_pypi_requirement("requests>=2.28.0")
        assert name == "requests"
        assert version == "2.28.0"

    def test_parse_pypi_requirement_name_only(self):
        builder = self._make_builder()
        name, version = builder._parse_pypi_requirement("flask")
        assert name == "flask"
        assert version is None

    def test_parse_pypi_requirement_with_extras(self):
        builder = self._make_builder()
        name, version = builder._parse_pypi_requirement("package[extra1,extra2]>=1.0")
        assert name == "package"

    def test_parse_pypi_requirement_unparseable(self):
        builder = self._make_builder()
        name, version = builder._parse_pypi_requirement("!!!invalid")
        assert name is None

    def test_build_dependency_tree(self):
        builder = self._make_builder()
        packages = [
            ExternalPackage(name="flask", version="2.0", license="BSD", ecosystem="pypi"),
            ExternalPackage(name="requests", version="2.28", license="Apache", ecosystem="pypi"),
        ]
        tree = builder.build_dependency_tree(packages)
        assert "flask" in tree
        assert tree["flask"]["version"] == "2.0"

    def test_find_dependency_paths(self):
        builder = self._make_builder()
        packages = [
            ExternalPackage(name="a", version="1.0", license="MIT", ecosystem="pypi"),
            ExternalPackage(name="b", version="1.0", license="MIT", ecosystem="pypi"),
        ]
        paths = builder.find_dependency_paths(packages, "b")
        assert isinstance(paths, list)

    @pytest.mark.asyncio
    async def test_analyze_transitive_dependencies_max_depth(self):
        builder = self._make_builder()
        pkg = ExternalPackage(name="test_pkg", version="1.0", license="MIT", ecosystem="unknown")
        result = await builder.analyze_transitive_dependencies([pkg], max_depth=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_build_dependency_graph(self):
        builder = self._make_builder()
        pkg = ExternalPackage(name="test_pkg", version="1.0", license="MIT", ecosystem="unknown")
        result = await builder.build_dependency_graph([pkg], max_depth=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# DependencyVulnerabilityScanner tests
# ---------------------------------------------------------------------------


class TestDependencyVulnerabilityScanner:
    def _make_scanner(self):
        from processing.architecture.dependency_visualization import DependencyVulnerabilityScanner

        return DependencyVulnerabilityScanner(enable_cache=True, batch_size=5)

    def test_get_cache_key(self):
        scanner = self._make_scanner()
        pkg = ExternalPackage(name="flask", version="2.0", license="BSD", ecosystem="pypi")
        key = scanner._get_cache_key(pkg)
        assert len(key) == 16

    def _make_vuln(self, **kwargs):
        defaults = dict(
            id="V1",
            severity="HIGH",
            summary="x",
            details="d",
            affected_versions=[],
            fixed_versions=[],
            references=[],
            published_date="2024-01-01",
            modified_date="2024-01-02",
        )
        defaults.update(kwargs)
        return Vulnerability(**defaults)

    def test_deduplicate_vulnerabilities(self):
        scanner = self._make_scanner()
        v1 = self._make_vuln(id="V1")
        v2 = self._make_vuln(id="V1")
        v3 = self._make_vuln(id="V2", severity="LOW", summary="y")
        result = scanner._deduplicate_vulnerabilities([v1, v2, v3])
        assert len(result) == 2

    def test_cvss_to_severity(self):
        scanner = self._make_scanner()
        assert scanner._cvss_to_severity(9.5) == "CRITICAL"
        assert scanner._cvss_to_severity(7.5) == "HIGH"
        assert scanner._cvss_to_severity(5.0) == "MODERATE"
        assert scanner._cvss_to_severity(2.0) == "LOW"

    def test_extract_severity_from_osv_cvss(self):
        scanner = self._make_scanner()
        vuln_data = {"severity": [{"type": "CVSS_V3", "score": "9.1"}]}
        assert scanner._extract_severity_from_osv(vuln_data) == "CRITICAL"

    def test_extract_severity_from_osv_database_specific(self):
        scanner = self._make_scanner()
        vuln_data = {"database_specific": {"severity": "high"}}
        assert scanner._extract_severity_from_osv(vuln_data) == "HIGH"

    def test_extract_severity_from_osv_unknown(self):
        scanner = self._make_scanner()
        assert scanner._extract_severity_from_osv({}) == "UNKNOWN"

    def test_extract_version_ranges(self):
        scanner = self._make_scanner()
        vuln_data = {
            "affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.5.0"}]}]}]
        }
        affected, fixed = scanner._extract_version_ranges(vuln_data)
        assert "0" in affected
        assert "1.5.0" in fixed

    def test_parse_osv_vulnerability(self):
        scanner = self._make_scanner()
        vuln_data = {
            "id": "GHSA-1234",
            "summary": "Test vuln",
            "details": "Details",
            "severity": [{"type": "CVSS_V3", "score": "7.5"}],
            "affected": [],
            "references": [{"url": "https://example.com"}],
            "published": "2024-01-01",
            "modified": "2024-02-01",
        }
        vuln = scanner._parse_osv_vulnerability(vuln_data)
        assert vuln is not None
        assert vuln.id == "GHSA-1234"
        assert vuln.severity == "HIGH"

    def test_parse_osv_vulnerability_empty_id(self):
        scanner = self._make_scanner()
        assert scanner._parse_osv_vulnerability({}) is None

    def test_cache_and_separate(self):
        scanner = self._make_scanner()
        pkg = ExternalPackage(name="cached_pkg", version="1.0", license="MIT", ecosystem="pypi")
        scanner._cache_package_vulnerabilities(pkg, [])
        cached, uncached = scanner._separate_cached_packages([pkg])
        assert len(cached) == 1
        assert len(uncached) == 0

    def test_cache_disabled(self):
        from processing.architecture.dependency_visualization import DependencyVulnerabilityScanner

        scanner = DependencyVulnerabilityScanner(enable_cache=False)
        pkg = ExternalPackage(name="pkg", version="1.0", license="MIT", ecosystem="pypi")
        scanner._cache_package_vulnerabilities(pkg, [])
        assert len(scanner.vulnerability_cache) == 0

    def test_clear_cache(self):
        scanner = self._make_scanner()
        scanner.vulnerability_cache["key"] = {"data": [], "timestamp": datetime.now(UTC)}
        scanner.clear_cache()
        assert len(scanner.vulnerability_cache) == 0

    def test_get_cache_stats(self):
        scanner = self._make_scanner()
        scanner.vulnerability_cache["key"] = {"data": [], "timestamp": datetime.now(UTC)}
        stats = scanner.get_cache_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities_empty(self):
        scanner = self._make_scanner()
        result = await scanner.scan_vulnerabilities([])
        assert result.vulnerabilities == []


# ---------------------------------------------------------------------------
# DependencyVisualizer tests
# ---------------------------------------------------------------------------


class TestDependencyVisualizer:
    def _make_visualizer(self):
        from processing.architecture.dependency_visualization import DependencyVisualizer

        return DependencyVisualizer()

    def test_generate_dependency_tree_text_empty(self):
        viz = self._make_visualizer()
        assert viz.generate_dependency_tree_text([]) == "No dependencies found."

    def test_generate_dependency_tree_text_with_packages(self):
        viz = self._make_visualizer()
        packages = [
            ExternalPackage(name="flask", version="2.0", license="BSD", ecosystem="pypi"),
        ]
        text = viz.generate_dependency_tree_text(packages)
        assert "flask" in text
        assert "2.0" in text

    def test_generate_vulnerability_summary_empty(self):
        viz = self._make_visualizer()
        assert viz.generate_vulnerability_summary_text([]) == "No vulnerabilities found."

    def test_generate_vulnerability_summary_with_vulns(self):
        viz = self._make_visualizer()
        vulns = [
            Vulnerability(
                id="V1",
                severity="CRITICAL",
                summary="Critical bug",
                details="d",
                affected_versions=[],
                fixed_versions=[],
                references=[],
                published_date="2024-01-01",
                modified_date="2024-01-02",
            ),
            Vulnerability(
                id="V2",
                severity="HIGH",
                summary="High bug",
                details="d",
                affected_versions=[],
                fixed_versions=[],
                references=[],
                published_date="2024-01-01",
                modified_date="2024-01-02",
            ),
        ]
        text = viz.generate_vulnerability_summary_text(vulns)
        assert "Critical: 1" in text
        assert "High: 1" in text


# ---------------------------------------------------------------------------
# PerformanceOptimizer tests
# ---------------------------------------------------------------------------


class TestPerformanceOptimizer:
    def test_performance_config_defaults(self):
        from processing.architecture.performance_optimizer import PerformanceConfig

        config = PerformanceConfig()
        assert config.chunk_size == 1000
        assert config.max_memory_mb == 512

    def test_architecture_performance_metrics_to_dict(self):
        from processing.architecture.performance_optimizer import (
            ArchitectureCacheMetrics,
            ArchitecturePerformanceMetrics,
            ProcessingCounts,
        )

        metrics = ArchitecturePerformanceMetrics(
            total_processing_time=1.5,
            memory_usage_mb=256.0,
            cpu_utilization=50.0,
            processing_counts=ProcessingCounts(10, 100, 200),
            cache_metrics=ArchitectureCacheMetrics(5, 3),
            parallel_workers_used=4,
        )
        d = metrics.to_dict()
        assert d["total_processing_time"] == 1.5
        assert d["cache_hits"] == 5

    def test_memory_efficient_graph_processor_chunk_nodes(self):
        from processing.architecture.performance_optimizer import MemoryEfficientGraphProcessor

        proc = MemoryEfficientGraphProcessor(chunk_size=3)
        chunks = list(proc._chunk_nodes(["a", "b", "c", "d", "e"], 3))
        assert len(chunks) == 2
        assert chunks[0] == ["a", "b", "c"]

    def test_memory_efficient_graph_processor_get_memory_status(self):
        from processing.architecture.performance_optimizer import MemoryEfficientGraphProcessor

        proc = MemoryEfficientGraphProcessor()
        status = proc.get_memory_status()
        assert "current_memory_mb" in status
        assert "max_memory_mb" in status

    def test_adaptive_parallelism_manager_init(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=4)
        assert mgr.min_workers == 1
        assert mgr.max_workers == 4

    def test_adaptive_parallelism_manager_get_optimal_workers(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=8)
        workers = mgr.get_optimal_worker_count(10, task_complexity=1.0)
        assert 1 <= workers <= 8

    def test_adaptive_parallelism_execute_empty(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        mgr = AdaptiveParallelismManager()
        result = mgr.execute_parallel_tasks([])
        assert result == []

    def test_adaptive_parallelism_execute_tasks(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        mgr = AdaptiveParallelismManager(min_workers=1, max_workers=2)
        tasks = [lambda: 1, lambda: 2, lambda: 3]
        results = mgr.execute_parallel_tasks(tasks)
        assert sorted(results) == [1, 2, 3]

    def test_optimizer_init_and_reset(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        opt.metrics.total_processing_time = 5.0
        opt.reset_metrics()
        assert opt.metrics.total_processing_time == 0

    def test_optimizer_get_performance_metrics(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        m = opt.get_performance_metrics()
        assert "total_processing_time" in m

    def test_determine_processing_strategy_small(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        for i in range(5):
            g.add_node(f"n{i}")
        strategy = opt._determine_processing_strategy(g, [MagicMock()] * 3)
        assert strategy == "sequential"

    def test_process_sequential(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        g = nx.DiGraph()
        g.add_node("n1")
        comp = Component(id="c1", name="C1", files=["a.py"], dependencies=[])
        result = opt._process_sequential(g, [comp], {"noop": lambda g, c: None})
        assert result.codebase_id == "unknown"

    def test_categorize_result_metrics(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("calculate_metrics", [1, 2, 3], agg)
        assert agg["metrics"] == [1, 2, 3]

    def test_categorize_result_patterns(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("detect_patterns", ["p1"], agg)
        assert agg["patterns"] == ["p1"]

    def test_categorize_result_layers(self):
        from processing.architecture.performance_optimizer import ArchitecturePerformanceOptimizer

        opt = ArchitecturePerformanceOptimizer()
        agg = {"metrics": [], "patterns": [], "layer_mappings": {}, "summary": {}}
        opt._categorize_result("assign_layers", {"api": ["c1"]}, agg)
        assert agg["layer_mappings"] == {"api": ["c1"]}


# ---------------------------------------------------------------------------
# ArchitectureProcessor tests
# ---------------------------------------------------------------------------


class TestArchitectureProcessor:
    def test_validate_configuration_valid(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        config = RepoConfig(detect_by=["folder", "build_file"])
        assert proc.validate_configuration(config) is True

    def test_validate_configuration_invalid_strategy(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        config = RepoConfig(detect_by=["invalid_strategy"])
        assert proc.validate_configuration(config) is False

    def test_validate_configuration_invalid_layers(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        # Pydantic validates layers, so we use a mock config to bypass validation
        config = MagicMock()
        config.detect_by = ["folder"]
        config.layers = {"bad": "not_a_list"}
        assert proc.validate_configuration(config) is False

    def test_get_component_fingerprint_minimal(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        fp = proc.get_component_fingerprint("test_codebase")
        assert len(fp) == 64  # SHA256 hex

    def test_get_component_fingerprint_with_components(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        g = nx.DiGraph()
        g.add_node(
            "n1",
            file_path="a.py",
            type="class",
            name="MyClass",
            properties={"visibility": "public", "modifiers": ["static"]},
        )
        comp = Component(id="c1", name="C1", files=["a.py"], dependencies=["c2"])
        fp = proc.get_component_fingerprint("test_codebase", components=[comp], graph=g)
        assert len(fp) == 64

    def test_convert_quality_result_empty(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        assert proc._convert_quality_result(None) == {}

    async def test_load_repository_config_fallback(self):
        from unittest.mock import AsyncMock, patch

        from processing.architecture.architecture_processor import ArchitectureProcessor

        mock_adapter = MagicMock()
        proc = ArchitectureProcessor(graph_adapter=mock_adapter)
        # No .codegraph.yml in the container -> fall back to defaults. The config
        # is now read via the coding-containers exec API, not local fs.
        client = MagicMock()
        client.read_file = AsyncMock(return_value=None)
        with patch("clients.coding_containers.get_coding_containers_client", return_value=client):
            config = await proc._load_repository_config(Path("/workspace"), "cn-1")
        assert isinstance(config, RepoConfig)


# ---------------------------------------------------------------------------
# TimeSeriesMetrics tests
# ---------------------------------------------------------------------------


class TestTimeSeriesMetrics:
    def test_metric_time_series_add_and_stats(self):
        from processing.architecture.time_series_metrics import ArchitectureMetricTimeSeries

        ts = ArchitectureMetricTimeSeries(metric_name="complexity")
        ts.add_measurement(datetime.now(UTC), 10.0)
        ts.add_measurement(datetime.now(UTC), 20.0)
        stats = ts.get_statistics()
        assert stats["mean"] == 15.0
        assert stats["count"] == 2

    def test_metric_time_series_empty_stats(self):
        from processing.architecture.time_series_metrics import ArchitectureMetricTimeSeries

        ts = ArchitectureMetricTimeSeries(metric_name="x")
        assert ts.get_statistics() == {}

    def test_metric_time_series_trend_insufficient(self):
        from processing.architecture.time_series_metrics import ArchitectureMetricTimeSeries

        ts = ArchitectureMetricTimeSeries(metric_name="x")
        ts.add_measurement(datetime.now(UTC), 1.0)
        assert ts.get_trend() == "insufficient_data"

    def test_metric_time_series_trend_increasing(self):
        from processing.architecture.time_series_metrics import ArchitectureMetricTimeSeries

        ts = ArchitectureMetricTimeSeries(metric_name="x")
        base = datetime.now(UTC)
        for i in range(10):
            ts.add_measurement(base + timedelta(hours=i), float(i * 10))
        trend = ts.get_trend(window_size=5)
        assert trend == "increasing"

    def test_metric_time_series_trend_stable(self):
        from processing.architecture.time_series_metrics import ArchitectureMetricTimeSeries

        ts = ArchitectureMetricTimeSeries(metric_name="x")
        base = datetime.now(UTC)
        for i in range(10):
            ts.add_measurement(base + timedelta(hours=i), 50.0)
        assert ts.get_trend(window_size=5) == "stable"

    def test_cross_language_correlation_to_dict(self):
        from processing.architecture.time_series_metrics import CrossLanguageCorrelation

        c = CrossLanguageCorrelation(
            language_pair=("python", "javascript"),
            correlation_coefficient=0.85,
            shared_patterns=["MVC"],
        )
        d = c.to_dict()
        assert d["language_pair"] == ("python", "javascript")
        assert d["shared_patterns"] == ["MVC"]

    def test_time_series_calculator_store_and_retrieve(self):
        from processing.architecture.time_series_metrics import TimeSeriesMetricsCalculator

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            calc = TimeSeriesMetricsCalculator(db_path=db_path)
            calc.store_metrics(
                "codebase1", {"complexity": 10.0, "coupling": 5.0}, commit_hash="abc123"
            )
            ts = calc.get_metrics_time_series("codebase1", "complexity", days_back=1)
            assert len(ts.values) == 1
            assert ts.values[0] == 10.0
        finally:
            db_path.unlink(missing_ok=True)

    def test_time_series_calculator_compare_commits(self):
        from processing.architecture.time_series_metrics import TimeSeriesMetricsCalculator

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            calc = TimeSeriesMetricsCalculator(db_path=db_path)
            calc.store_metrics("cb1", {"m1": 10.0}, commit_hash="aaa")
            calc.store_metrics("cb1", {"m1": 20.0}, commit_hash="bbb")
            comparison = calc.compare_metrics_across_commits("cb1", ["aaa", "bbb"])
            assert "m1" in comparison
        finally:
            db_path.unlink(missing_ok=True)

    def test_cross_language_correlation_analyzer(self):
        from processing.architecture.time_series_metrics import CrossLanguageCorrelationAnalyzer

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            # Initialize DB with required tables
            from processing.architecture.time_series_metrics import TimeSeriesMetricsCalculator

            TimeSeriesMetricsCalculator(db_path=db_path)

            analyzer = CrossLanguageCorrelationAnalyzer(db_path=db_path)
            metrics = {
                "python": {"complexity": 10.0, "coupling": 5.0},
                "javascript": {"complexity": 12.0, "coupling": 4.0},
            }
            patterns = {"python": ["MVC", "Singleton"], "javascript": ["MVC"]}
            deps = {"python": ["requests"], "javascript": ["axios"]}
            correlations = analyzer.analyze_cross_language_correlation(
                "cb1", metrics, patterns, deps
            )
            assert len(correlations) == 1
            assert correlations[0].language_pair == ("python", "javascript")
        finally:
            db_path.unlink(missing_ok=True)

    def test_architectural_similarity(self):
        from processing.architecture.time_series_metrics import CrossLanguageCorrelationAnalyzer

        analyzer = CrossLanguageCorrelationAnalyzer()
        assert analyzer._calculate_architectural_similarity(set(), set()) == 1.0
        assert analyzer._calculate_architectural_similarity({"a"}, set()) == 0.0
        assert analyzer._calculate_architectural_similarity(
            {"a", "b"}, {"a", "c"}
        ) == pytest.approx(1 / 3)
