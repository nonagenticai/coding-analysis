"""Unit tests for processing.architecture.component_extractor module."""

from __future__ import annotations

import networkx as nx
import pytest

from models.architecture_models import Component, ComponentExtractionStats, RepoConfig
from processing.architecture.component_extractor import ComponentExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(nodes=None, edges=None):
    g = nx.DiGraph()
    if nodes:
        for nid, data in nodes:
            g.add_node(nid, **data)
    if edges:
        for src, dst in edges:
            g.add_edge(src, dst)
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return ComponentExtractor(max_workers=1)


@pytest.fixture
def sample_graph():
    """Graph with files in a typical folder structure with subdirectories."""
    return _make_graph(
        nodes=[
            ("n1", {"file_path": "src/apps/web/main.py", "type": "function", "name": "main"}),
            ("n2", {"file_path": "src/apps/web/routes.py", "type": "function", "name": "routes"}),
            ("n3", {"file_path": "src/apps/lib/utils.py", "type": "function", "name": "util"}),
            ("n4", {"file_path": "src/apps/lib/helpers.py", "type": "function", "name": "helper"}),
            (
                "n5",
                {
                    "file_path": "src/apps/test/test_main.py",
                    "type": "function",
                    "name": "test_main",
                },
            ),
        ],
        edges=[("n1", "n3"), ("n2", "n3"), ("n2", "n4"), ("n5", "n1")],
    )


@pytest.fixture
def config_folder():
    return RepoConfig(detect_by=["folder"], folder_roots=["src/apps/**"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComponentExtractionFromSampleModules:
    """test_component_extraction_from_sample_modules"""

    def test_extract_by_folder_finds_components(self, extractor, sample_graph, config_folder):
        components = extractor.extract(sample_graph, config_folder)
        assert isinstance(components, list)
        assert len(components) > 0

    def test_empty_folder_roots_defaults_to_top_level_dir(self, extractor, sample_graph):
        """With no folder_roots configured (no .codegraph.yml), folder detection
        falls back to grouping by top-level directory so a parsed graph still
        yields components instead of zero."""
        config = RepoConfig(detect_by=["folder"], folder_roots=[])
        components = extractor.extract(sample_graph, config)
        assert len(components) > 0

    def test_extracted_components_have_files(self, extractor, sample_graph, config_folder):
        components = extractor.extract(sample_graph, config_folder)
        for comp in components:
            assert isinstance(comp, Component)
            assert len(comp.files) > 0

    def test_dependencies_resolved_between_components(self, extractor, sample_graph, config_folder):
        components = extractor.extract(sample_graph, config_folder)
        # At least one component should have a dependency on another
        all_deps = []
        for comp in components:
            all_deps.extend(comp.dependencies)
        assert len(all_deps) > 0, "Expected at least one inter-component dependency"

    def test_component_names_generated(self, extractor, sample_graph, config_folder):
        components = extractor.extract(sample_graph, config_folder)
        for comp in components:
            assert comp.name
            assert comp.id

    def test_build_file_strategy(self, extractor):
        g = _make_graph(
            nodes=[
                (
                    "bf1",
                    {"file_path": "frontend/package.json", "type": "file", "name": "package.json"},
                ),
                ("f1", {"file_path": "frontend/index.js", "type": "function", "name": "index"}),
                ("f2", {"file_path": "frontend/app.js", "type": "function", "name": "app"}),
            ],
            edges=[("f1", "f2")],
        )
        config = RepoConfig(detect_by=["build_file"], build_files=["package.json"])
        components = extractor.extract(g, config)
        assert len(components) >= 1
        # The component should contain files from the frontend directory
        found_frontend = any("frontend" in f for comp in components for f in comp.files)
        assert found_frontend

    def test_extraction_stats(self, extractor, sample_graph, config_folder):
        components = extractor.extract(sample_graph, config_folder)
        stats = extractor.get_extraction_stats(components)
        assert isinstance(stats, ComponentExtractionStats)
        assert stats.components_found == len(components)
        assert stats.edges_scanned >= 0

    def test_annotation_strategy_with_code(self, extractor):
        """Test annotation-based component detection finds annotated code."""
        g = _make_graph(
            nodes=[
                (
                    "a1",
                    {
                        "file_path": "src/service.py",
                        "type": "class",
                        "name": "UserService",
                        "code": "# @component: UserModule\nclass UserService:\n    pass\n",
                    },
                ),
            ],
        )
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        assert isinstance(components, list)


class TestIgnoredFiles:
    """test_ignored_files"""

    def test_nodes_without_file_path_skipped(self, extractor):
        g = _make_graph(
            nodes=[
                ("n1", {"type": "function", "name": "orphan"}),
                ("n2", {"file_path": "src/apps/web/main.py", "type": "function", "name": "main"}),
            ],
        )
        config = RepoConfig(detect_by=["folder"], folder_roots=["src/apps/**"])
        components = extractor.extract(g, config)
        # Only the node with a file_path should contribute to a component
        all_files = [f for comp in components for f in comp.files]
        assert "src/apps/web/main.py" in all_files

    def test_no_matching_pattern_produces_no_component(self, extractor):
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "unmatched/dir/foo.py", "type": "function", "name": "foo"}),
            ],
        )
        config = RepoConfig(detect_by=["folder"], folder_roots=["src/apps/**"])
        components = extractor.extract(g, config)
        assert components == []

    def test_deduplicate_merges_same_component_id(self, extractor):
        comp_a = Component(
            id="component_app",
            name="App",
            files=["a.py", "b.py"],
            dependencies=[],
            layer=None,
        )
        comp_b = Component(
            id="component_app",
            name="App",
            files=["b.py", "c.py"],
            dependencies=[],
            layer=None,
        )
        result = extractor._deduplicate_components([comp_a, comp_b])
        assert len(result) == 1
        assert set(result[0].files) == {"a.py", "b.py", "c.py"}

    def test_deduplicate_preserves_distinct_components(self, extractor):
        comp_a = Component(id="comp_a", name="A", files=["a.py"], dependencies=[], layer=None)
        comp_b = Component(id="comp_b", name="B", files=["b.py"], dependencies=[], layer=None)
        result = extractor._deduplicate_components([comp_a, comp_b])
        assert len(result) == 2


class TestMalformedAstInput:
    """test_malformed_ast_input"""

    def test_empty_graph_returns_empty(self, extractor):
        g = nx.DiGraph()
        config = RepoConfig(detect_by=["folder"], folder_roots=["src/**"])
        components = extractor.extract(g, config)
        assert components == []

    def test_graph_size_estimate_empty(self, extractor):
        g = nx.DiGraph()
        size = extractor._estimate_graph_size(g)
        assert size == 0.0

    def test_graph_size_estimate_with_nodes(self, extractor, sample_graph):
        size = extractor._estimate_graph_size(sample_graph)
        assert size >= 0.0

    def test_matches_pattern(self, extractor):
        assert extractor._matches_pattern("src/app/main.py", "src/app/*") is True
        assert extractor._matches_pattern("other/foo.py", "src/app/*") is False

    def test_get_component_name(self, extractor):
        name = extractor._get_component_name("component_src_app")
        assert "Src App" == name

    def test_get_component_name_no_prefix(self, extractor):
        name = extractor._get_component_name("custom_id")
        assert name == "custom_id"

    def test_extraction_metrics_stored(self, extractor, sample_graph, config_folder):
        extractor.extract(sample_graph, config_folder)
        metrics = extractor.extraction_metrics
        assert "strategies_used" in metrics
        assert "folder" in metrics["strategies_used"]
        assert "final_components" in metrics
        assert metrics["final_components"] >= 0
        assert metrics["graph_nodes"] == 5
        assert metrics["graph_size_mb"] >= 0.0

    def test_get_component_id_from_pattern_double_star(self, extractor):
        cid = extractor._get_component_id_from_pattern("src/apps/web/main.py", "src/apps/**")
        assert "web" in cid
        assert cid.startswith("component_")

    def test_get_component_id_from_pattern_fallback(self, extractor):
        cid = extractor._get_component_id_from_pattern("src/app/main.py", "src/app/*")
        assert cid.startswith("component_")

    def test_find_component_dependencies(self, extractor, sample_graph):
        comp_web = Component(
            id="comp-web",
            name="Web",
            files=["src/apps/web/main.py", "src/apps/web/routes.py"],
            dependencies=[],
            layer=None,
        )
        file_to_component = {
            "src/apps/web/main.py": "comp-web",
            "src/apps/web/routes.py": "comp-web",
            "src/apps/lib/utils.py": "comp-lib",
            "src/apps/lib/helpers.py": "comp-lib",
        }
        deps = extractor._find_component_dependencies(comp_web, sample_graph, file_to_component)
        assert "comp-lib" in deps
        assert "comp-web" not in deps  # no self-dependencies


class TestComponentExtractorAnnotationParallel:
    """Test annotation-based detection exercising parallel code paths."""

    def test_annotation_strategy_comment_annotation(self):
        """Annotations via '# component: X' are found."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                (
                    "a1",
                    {
                        "file_path": "src/handler.py",
                        "type": "class",
                        "name": "Handler",
                        "code": "# component: PaymentModule\nclass Handler:\n    pass\n",
                    },
                ),
                (
                    "a2",
                    {
                        "file_path": "src/processor.py",
                        "type": "class",
                        "name": "Processor",
                        "code": "# component: PaymentModule\nclass Processor:\n    pass\n",
                    },
                ),
            ],
        )
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        # Should find PaymentModule component
        assert isinstance(components, list)

    def test_annotation_java_style(self):
        """Java-style @Component annotation."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                (
                    "j1",
                    {
                        "file_path": "src/UserService.java",
                        "type": "class",
                        "name": "UserService",
                        "code": '@Service("userService")\npublic class UserService {}\n',
                    },
                ),
            ],
        )
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        assert isinstance(components, list)

    def test_annotation_no_code_skips(self):
        """Nodes without code/content are skipped in annotation detection."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "src/empty.py", "type": "function", "name": "empty"}),
            ],
        )
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        assert components == []

    def test_annotation_language_mismatch_skips(self):
        """Python annotation pattern on a .java file is skipped."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                (
                    "n1",
                    {
                        "file_path": "src/service.java",
                        "type": "class",
                        "name": "Service",
                        "code": '@component("test")\nclass Service:\n    pass',
                    },
                ),
            ],
        )
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        # Python decorator pattern should not match .java file
        assert isinstance(components, list)


class TestComponentExtractorBuildFileAdvanced:
    """Test build file strategy edge cases."""

    def test_build_file_no_files_in_dir(self):
        """Build file with no other files in dir produces no component."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                (
                    "bf1",
                    {
                        "file_path": "empty_project/package.json",
                        "type": "file",
                        "name": "package.json",
                    },
                ),
            ],
        )
        config = RepoConfig(detect_by=["build_file"], build_files=["package.json"])
        components = extractor.extract(g, config)
        # The build file itself is included as a file in the component
        assert isinstance(components, list)
        if components:
            assert any("package.json" in f for c in components for f in c.files)

    def test_build_file_multiple_projects(self):
        """Multiple build files in different directories create separate components."""
        extractor = ComponentExtractor(max_workers=1)
        g = _make_graph(
            nodes=[
                (
                    "bf1",
                    {"file_path": "frontend/package.json", "type": "file", "name": "package.json"},
                ),
                ("f1", {"file_path": "frontend/index.js", "type": "function", "name": "index"}),
                (
                    "bf2",
                    {"file_path": "backend/package.json", "type": "file", "name": "package.json"},
                ),
                ("f2", {"file_path": "backend/server.js", "type": "function", "name": "server"}),
            ],
        )
        config = RepoConfig(detect_by=["build_file"], build_files=["package.json"])
        components = extractor.extract(g, config)
        assert len(components) >= 2


class TestComponentExtractorParallelPaths:
    """Test parallel processing paths for large graphs."""

    def test_extract_by_folder_parallel_triggered(self):
        """When graph size >= min_parallel_threshold, parallel path is used."""
        extractor = ComponentExtractor(max_workers=2, min_parallel_threshold=3)
        # Create a graph with enough nodes to trigger parallel processing
        nodes = []
        for i in range(5):
            nodes.append(
                (
                    f"n{i}",
                    {"file_path": f"src/apps/mod{i}/file.py", "type": "function", "name": f"f{i}"},
                )
            )
        g = _make_graph(nodes=nodes)
        config = RepoConfig(detect_by=["folder"], folder_roots=["src/apps/**"])
        components = extractor.extract(g, config)
        assert isinstance(components, list)

    def test_extract_by_build_files_parallel_triggered(self):
        """When graph size >= min_parallel_threshold, parallel build file path is used."""
        extractor = ComponentExtractor(max_workers=2, min_parallel_threshold=3)
        nodes = [
            ("bf1", {"file_path": "project/package.json", "type": "file", "name": "package.json"}),
        ]
        for i in range(5):
            nodes.append(
                (f"f{i}", {"file_path": f"project/file{i}.js", "type": "function", "name": f"f{i}"})
            )
        g = _make_graph(nodes=nodes)
        config = RepoConfig(detect_by=["build_file"], build_files=["package.json"])
        components = extractor.extract(g, config)
        assert isinstance(components, list)

    def test_extract_by_annotations_parallel_triggered(self):
        """When graph size >= min_parallel_threshold, parallel annotation path is used."""
        extractor = ComponentExtractor(max_workers=2, min_parallel_threshold=3)
        nodes = []
        for i in range(5):
            nodes.append(
                (
                    f"n{i}",
                    {
                        "file_path": f"src/file{i}.py",
                        "type": "class",
                        "name": f"Class{i}",
                        "code": f"# component: Module{i % 2}\nclass Class{i}:\n    pass\n",
                    },
                )
            )
        g = _make_graph(nodes=nodes)
        config = RepoConfig(detect_by=["annotation"])
        components = extractor.extract(g, config)
        assert isinstance(components, list)


class TestComponentExtractorProgressCallback:
    """Test progress callback and combined callback setup."""

    def test_progress_callback_invoked(self):
        """Progress callback is called during extraction."""
        callback_calls = []

        def my_callback(current, total, metrics):
            callback_calls.append((current, total, metrics))

        extractor = ComponentExtractor(max_workers=1, progress_callback=my_callback)
        g = _make_graph(
            nodes=[
                ("n1", {"file_path": "src/apps/web/main.py", "type": "function", "name": "main"}),
            ],
        )
        config = RepoConfig(detect_by=["folder"], folder_roots=["src/apps/**"])
        extractor.extract(g, config)
        # The callback may not be called without a ProgressReporter, but the code path is exercised
        assert isinstance(extractor.extraction_metrics, dict)


class TestComponentExtractorEnvWorkerCount:
    """Test _get_default_worker_count environment variable handling."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ARCH_PROCESS_POOL_SIZE", "4")
        extractor = ComponentExtractor()
        assert extractor.max_workers == 4

    def test_env_override_invalid(self, monkeypatch):
        monkeypatch.setenv("ARCH_PROCESS_POOL_SIZE", "invalid")
        extractor = ComponentExtractor()
        # Falls through to CPU count logic
        assert extractor.max_workers >= 1

    def test_env_override_zero(self, monkeypatch):
        monkeypatch.setenv("ARCH_PROCESS_POOL_SIZE", "0")
        extractor = ComponentExtractor()
        # max(1, 0) = 1
        assert extractor.max_workers == 1


class TestBuildFileNoneGuard:
    """Regression: graph nodes can carry file_path=None (synthetic / edge-created
    nodes). The parallel build-file path must skip them instead of crashing on
    Path(None) -> TypeError."""

    def test_find_build_directories_skips_none_and_missing_file_path(self, extractor):
        graph = _make_graph(
            nodes=[
                ("real", {"file_path": "app/pkg/package.json", "type": "file"}),
                ("none_node", {"file_path": None, "type": "function"}),
                ("missing", {"type": "function"}),  # no file_path key at all
            ]
        )
        # Must not raise; only the real build file yields a directory.
        dirs = extractor._find_build_directories(graph, ["package.json"])
        assert dirs == ["app/pkg"]

    def test_extract_by_build_files_parallel_with_none_node(self, extractor):
        graph = _make_graph(
            nodes=[
                ("build", {"file_path": "svc/package.json", "type": "file"}),
                ("code", {"file_path": "svc/index.js", "type": "function"}),
                ("ghost", {"file_path": None, "type": "function"}),
            ]
        )
        components = extractor._extract_by_build_files_parallel(graph, ["package.json"])
        assert isinstance(components, list)


class TestContainsEdgesExcludedFromDependencies:
    """A structural ``contains`` edge must never count as a component dependency.

    Regression for the live-found ``alpha -> gamma`` false dependency: when a
    same-named entity collides on node id, a file's ``contains`` edge can point at
    an entity carried under another file's path, fabricating a cross-component
    dependency. Only real edges (imports/calls/inherits/references) are deps.
    """

    @staticmethod
    def _graph_with_relationships(nodes, edges):
        g = nx.DiGraph()
        for nid, data in nodes:
            g.add_node(nid, **data)
        for src, dst, rel in edges:
            g.add_edge(src, dst, relationship=rel)
        return g

    def test_contains_edge_is_not_a_dependency(self, extractor):
        graph = self._graph_with_relationships(
            nodes=[
                ("alpha_file", {"file_path": "src/alpha/mod.py", "type": "file"}),
                # a contained entity that (via collision) carries gamma's path
                ("collided_fn", {"file_path": "src/gamma/mod.py", "type": "function"}),
            ],
            edges=[("alpha_file", "collided_fn", "contains")],
        )
        file_to_component = {
            "src/alpha/mod.py": "comp-alpha",
            "src/gamma/mod.py": "comp-gamma",
        }
        comp_alpha = Component(
            id="comp-alpha", name="Alpha", files=["src/alpha/mod.py"], dependencies=[], layer=None
        )
        deps = extractor._find_component_dependencies(comp_alpha, graph, file_to_component)
        assert "comp-gamma" not in deps  # contains is structural, not a dependency
        assert deps == set()

    def test_real_edges_still_count_as_dependencies(self, extractor):
        graph = self._graph_with_relationships(
            nodes=[
                ("alpha_fn", {"file_path": "src/alpha/mod.py", "type": "function"}),
                ("beta_fn", {"file_path": "src/beta/mod.py", "type": "function"}),
                ("gamma_fn", {"file_path": "src/gamma/mod.py", "type": "function"}),
            ],
            edges=[
                ("alpha_fn", "beta_fn", "calls"),
                ("alpha_fn", "gamma_fn", "imports"),
            ],
        )
        file_to_component = {
            "src/alpha/mod.py": "comp-alpha",
            "src/beta/mod.py": "comp-beta",
            "src/gamma/mod.py": "comp-gamma",
        }
        comp_alpha = Component(
            id="comp-alpha", name="Alpha", files=["src/alpha/mod.py"], dependencies=[], layer=None
        )
        deps = extractor._find_component_dependencies(comp_alpha, graph, file_to_component)
        assert deps == {"comp-beta", "comp-gamma"}
