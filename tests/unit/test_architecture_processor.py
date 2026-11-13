"""Unit tests for processing.architecture.architecture_processor module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from models.architecture_models import (
    ArchitectureJobRequest,
    Component,
    RepoConfig,
)
from processing.architecture.architecture_processor import ArchitectureProcessor


@pytest.fixture(autouse=True)
def _mock_containers_client():
    """Keep ``process()`` tests hermetic + fast.

    ``process()`` now reaches the coding-containers exec API (config read via
    ``read_file``, dependency manifests via ``execute``). Without a mock these
    tests make real network calls that hang on connect-timeout. Tests that need
    specific dependency/quality behaviour override those analyzer methods
    directly, so a benign default client here does not interfere.
    """
    client = MagicMock()
    client.read_file = AsyncMock(return_value=None)
    client.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    # _load_repository_config imports the factory at call time (clients.*), while
    # dependency_analyzer binds it at module load — patch both sites.
    with (
        patch("clients.coding_containers.get_coding_containers_client", return_value=client),
        patch(
            "processing.architecture.dependency_analyzer.get_coding_containers_client",
            return_value=client,
        ),
    ):
        yield client


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
def graph_adapter():
    adapter = AsyncMock()
    g = _make_graph(
        nodes=[
            ("n1", {"file_path": "src/app/main.py", "type": "function", "name": "main"}),
            ("n2", {"file_path": "src/lib/utils.py", "type": "function", "name": "util"}),
        ],
        edges=[("n1", "n2")],
    )
    adapter.load_graph = AsyncMock(return_value=g)
    return adapter


@pytest.fixture
def processor(graph_adapter):
    with patch("processing.architecture.architecture_processor.YamlConfigLoader") as mock_loader:
        mock_loader.return_value.load.side_effect = FileNotFoundError("no config")
        proc = ArchitectureProcessor(graph_adapter=graph_adapter)
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndProcessorOrchestration:
    """test_end_to_end_processor_orchestration_with_mocked_collaborators"""

    async def test_process_returns_analysis_result(self, processor, tmp_path):
        job = ArchitectureJobRequest(
            codebase_id="test-codebase",
            repo_path=tmp_path,
            tenant_id="tenant-a",
        )
        result = await processor.process(job)
        assert result.codebase_id == "test-codebase"
        assert isinstance(result.components, list)
        assert isinstance(result.layers, list)
        assert isinstance(result.patterns, list)
        assert result.execution_time >= 0

    async def test_process_contains_summary(self, processor, tmp_path):
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")
        result = await processor.process(job)
        assert "total_components" in result.summary
        assert "total_layers" in result.summary
        assert result.summary["total_components"] >= 0
        assert result.summary["total_layers"] >= 0
        assert result.summary["total_patterns"] >= 0
        assert "dependencies_analyzed" in result.summary

    def test_validate_configuration_valid(self, processor):
        config = RepoConfig(detect_by=["folder", "build_file"])
        assert processor.validate_configuration(config) is True

    def test_validate_configuration_invalid_strategy(self, processor):
        config = RepoConfig(detect_by=["invalid_strategy"])
        assert processor.validate_configuration(config) is False

    def test_validate_configuration_annotation_strategy(self, processor):
        config = RepoConfig(detect_by=["annotation"])
        assert processor.validate_configuration(config) is True

    def test_validate_configuration_invalid_layers_type(self, processor):
        # Pydantic validates layers at construction, so test with a mock
        config = MagicMock(spec=RepoConfig)
        config.detect_by = ["folder"]
        config.layers = {"bad": "not_a_list"}
        assert processor.validate_configuration(config) is False

    def test_validate_configuration_valid_layers(self, processor):
        config = RepoConfig(
            detect_by=["folder"],
            layers={"presentation": ["src/ui/*"], "domain": ["src/domain/*"]},
        )
        assert processor.validate_configuration(config) is True


class TestEmptyCodebase:
    """test_empty_codebase"""

    async def test_empty_graph_produces_empty_result(self, tmp_path):
        adapter = AsyncMock()
        adapter.load_graph = AsyncMock(return_value=nx.DiGraph())

        with patch("processing.architecture.architecture_processor.YamlConfigLoader") as ml:
            ml.return_value.load.side_effect = FileNotFoundError()
            proc = ArchitectureProcessor(graph_adapter=adapter)

        job = ArchitectureJobRequest(codebase_id="empty", repo_path=tmp_path, tenant_id="tenant-a")
        result = await proc.process(job)
        assert result.summary["total_components"] == 0
        assert result.summary["total_patterns"] == 0
        assert result.components == []
        assert result.patterns == []

    def test_fingerprint_without_components(self):
        adapter = AsyncMock()
        with patch("processing.architecture.architecture_processor.YamlConfigLoader") as ml:
            ml.return_value.load.side_effect = FileNotFoundError()
            proc = ArchitectureProcessor(graph_adapter=adapter)

        fp = proc.get_component_fingerprint("cid")
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA256 hex

    def test_fingerprint_with_components_and_graph(self):
        adapter = AsyncMock()
        with patch("processing.architecture.architecture_processor.YamlConfigLoader") as ml:
            ml.return_value.load.side_effect = FileNotFoundError()
            proc = ArchitectureProcessor(graph_adapter=adapter)

        g = _make_graph(
            nodes=[
                (
                    "n1",
                    {
                        "file_path": "src/app.py",
                        "type": "class",
                        "name": "App",
                        "properties": {"visibility": "public", "modifiers": ["static"]},
                    },
                ),
            ],
        )
        comps = [
            Component(
                id="comp-app",
                name="App",
                files=["src/app.py"],
                dependencies=["comp-lib"],
                layer=None,
            )
        ]
        fp = proc.get_component_fingerprint("cid", components=comps, graph=g)
        assert len(fp) == 64
        # Fingerprint should be deterministic
        fp2 = proc.get_component_fingerprint("cid", components=comps, graph=g)
        assert fp == fp2

    def test_fingerprint_different_for_different_codebases(self):
        adapter = AsyncMock()
        with patch("processing.architecture.architecture_processor.YamlConfigLoader") as ml:
            ml.return_value.load.side_effect = FileNotFoundError()
            proc = ArchitectureProcessor(graph_adapter=adapter)

        fp1 = proc.get_component_fingerprint("cid-1")
        fp2 = proc.get_component_fingerprint("cid-2")
        assert fp1 != fp2


class TestTenantThreading:
    """The job's tenant reaches the two container-backed stages.

    Both mirror files out of the tenant's coding-containers workspace; a call
    with no ``X-Tenant-Id`` is resolved as this service's own account and the
    BOLA guard answers 404, so the stage silently measures nothing.
    """

    async def test_dependency_stage_runs_as_the_jobs_tenant(self, processor, tmp_path):
        processor.dependency_analyzer.count_declared_packages = AsyncMock(return_value=0)
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")
        await processor.process(job)
        count = processor.dependency_analyzer.count_declared_packages
        assert count.await_args.kwargs.get("tenant_id") == "tenant-a", (
            "the dependency stage dropped the job's tenant before the container read"
        )


class TestCollaboratorFailureHandling:
    """test_collaborator_failure_handling"""

    async def test_dependency_analysis_failure_is_non_fatal(self, processor, tmp_path):
        """Dependency analysis failure should not prevent overall result."""
        processor.dependency_analyzer.count_declared_packages = AsyncMock(
            side_effect=FileNotFoundError("no deps")
        )
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")
        result = await processor.process(job)
        assert result is not None
        assert result.summary["dependencies_analyzed"] == 0

    async def test_graph_analysis_exception_degrades_to_empty(self, processor, tmp_path):
        """An unexpected exception from the graph extractors must not 500.

        The extractors handle empty graphs safely, but a malformed graph can
        raise types beyond FileNotFoundError/ImportError/ValueError. Those must
        degrade to an empty (well-formed) result so the verify-architecture
        oracle never receives an error.
        """
        processor.component_extractor.extract = MagicMock(
            side_effect=KeyError("missing node attribute")
        )
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")

        result = await processor.process(job)

        assert result is not None
        assert result.codebase_id == "cid"
        assert result.components == []
        assert result.layers == []
        assert result.patterns == []
        assert result.summary["total_components"] == 0

    async def test_graph_load_failure_raises(self, processor, tmp_path):
        processor.graph_adapter.load_graph = AsyncMock(side_effect=RuntimeError("graph fail"))
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")
        with pytest.raises(RuntimeError, match="graph fail"):
            await processor.process(job)


class TestConvertQualityResult:
    """Tests for _convert_quality_result."""

    def test_convert_none_returns_empty(self, processor):
        assert processor._convert_quality_result(None) == {}

    def test_convert_quality_result_with_data(self, processor):
        quality_result = MagicMock()
        quality_result.overall_score = 85.0
        issue = MagicMock()
        issue.severity = "warning"
        issue.message = "test issue"
        issue.file_path = "src/a.py"
        issue.line_number = 10
        quality_result.issues = [issue]
        quality_result.summary = {"total": 1}

        result = processor._convert_quality_result(quality_result)
        assert result["overall_score"] == 85.0
        assert result["issues_count"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "warning"
        assert result["summary"] == {"total": 1}


class TestLoadRepositoryConfig:
    """Tests for _load_repository_config."""

    async def test_load_config_no_file_uses_defaults(self, tmp_path):
        """No .codegraph.yml in the container -> default config."""
        proc = ArchitectureProcessor(graph_adapter=AsyncMock())
        client = MagicMock()
        client.read_file = AsyncMock(return_value=None)
        with patch("clients.coding_containers.get_coding_containers_client", return_value=client):
            config = await proc._load_repository_config(tmp_path, "cn-1")
        assert isinstance(config, RepoConfig)

    async def test_load_config_from_container_yaml(self, tmp_path):
        """.codegraph.yml is read from the container via exec and parsed."""
        proc = ArchitectureProcessor(graph_adapter=AsyncMock())
        client = MagicMock()
        client.read_file = AsyncMock(
            return_value="detect_by:\n  - folder\nfolder_roots:\n  - 'src/**'\n"
        )
        with patch("clients.coding_containers.get_coding_containers_client", return_value=client):
            config = await proc._load_repository_config(tmp_path, "cn-1")
        assert "folder" in config.detect_by
        client.read_file.assert_awaited_once()

    async def test_load_config_exec_error_uses_defaults(self, tmp_path):
        """An exec/network error must not break analysis — defaults are returned."""
        proc = ArchitectureProcessor(graph_adapter=AsyncMock())
        client = MagicMock()
        client.read_file = AsyncMock(side_effect=RuntimeError("containers unreachable"))
        with patch("clients.coding_containers.get_coding_containers_client", return_value=client):
            config = await proc._load_repository_config(tmp_path, "cn-1")
        assert isinstance(config, RepoConfig)


class TestPipelineWorkIsBounded:
    """D7: the architecture ops timed out at the 120 s edge on a 37-file repo.

    Measured live on cv-sweep-ems (2026-10-12, 130 s request): graph stages 1 s,
    then a per-package api.osv.dev vulnerability scan (96 s, egress reset, 125
    packages retried one at a time) whose result was discarded except for
    ``len(packages)``, then six container-exec quality tools (33 s) whose result
    was only logged. These tests count the calls to those two expensive steps
    through the REAL processor and the REAL dependency analyzer, so they bound
    the work without a wall clock: the count is zero, whatever the repo size.
    """

    _REQUIREMENTS = "requests==2.31.0\nflask==3.0.0\nclick==8.1.7\n"

    @pytest.fixture
    def counted(self):
        from processing.architecture.dependency_visualization import (
            DependencyVulnerabilityScanner,
        )
        from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer

        scan = AsyncMock(side_effect=AssertionError("vulnerability scan must not run"))
        quality = AsyncMock(side_effect=AssertionError("quality tools must not run"))
        with (
            patch.object(DependencyVulnerabilityScanner, "scan_vulnerabilities", scan),
            patch.object(ArchitectureQualityAnalyzer, "analyze", quality),
        ):
            yield scan, quality

    async def test_full_analysis_counts_packages_without_scanning(
        self, processor, tmp_path, counted
    ):
        scan, quality = counted
        (tmp_path / "requirements.txt").write_text(self._REQUIREMENTS)
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")

        result = await processor.process(job)

        # The summary still reports the declared packages (same value as before) ...
        assert result.summary["dependencies_analyzed"] == 3
        # ... without a single per-package network scan or container-exec linter.
        assert scan.await_count == 0
        assert quality.await_count == 0

    async def test_graph_only_read_skips_the_manifest_read(self, processor, tmp_path, counted):
        scan, quality = counted
        (tmp_path / "requirements.txt").write_text(self._REQUIREMENTS)
        extract = MagicMock(wraps=processor.dependency_analyzer.extractor.extract_dependencies)
        processor.dependency_analyzer.extractor.extract_dependencies = extract
        job = ArchitectureJobRequest(codebase_id="cid", repo_path=tmp_path, tenant_id="tenant-a")

        result = await processor.process(job, include_dependency_analysis=False)

        assert extract.call_count == 0
        assert result.summary["dependencies_analyzed"] == 0
        assert scan.await_count == 0
        assert quality.await_count == 0

    async def test_count_declared_packages_is_manifest_only(self, tmp_path, counted):
        from processing.architecture.dependency_analyzer import ArchitectureDependencyAnalyzer

        scan, _ = counted
        (tmp_path / "requirements.txt").write_text(self._REQUIREMENTS)

        n = await ArchitectureDependencyAnalyzer().count_declared_packages(
            tmp_path, "cid", tenant_id="tenant-a"
        )

        assert n == 3
        assert scan.await_count == 0

    async def test_count_declared_packages_without_manifests_is_zero(self, tmp_path):
        from processing.architecture.dependency_analyzer import ArchitectureDependencyAnalyzer

        missing = tmp_path / "absent"
        assert await ArchitectureDependencyAnalyzer().count_declared_packages(missing, "cid") == 0
