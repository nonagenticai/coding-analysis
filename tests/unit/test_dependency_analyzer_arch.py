"""Unit tests for processing.architecture.dependency_analyzer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.architecture_models import (
    DependencyRiskReport,
    ExternalPackage,
    Vulnerability,
)
from processing.architecture.dependency_analyzer import ArchitectureDependencyAnalyzer
from processing.architecture.dependency_visualization import VulnerabilityScanResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_package(
    name: str = "requests", version: str = "2.28.0", ecosystem: str = "pypi"
) -> ExternalPackage:
    return ExternalPackage(name=name, version=version, ecosystem=ecosystem)


def _make_vulnerability(
    vuln_id: str = "GHSA-123",
    severity: str = "high",
    pkg_name: str = "requests",
) -> Vulnerability:
    return Vulnerability(
        id=vuln_id,
        severity=severity,
        summary=f"Vuln in {pkg_name}",
        details="Details",
        affected_versions=["<2.29.0"],
        fixed_versions=["2.29.0"],
        references=["https://example.com"],
        published_date="2024-01-01",
        modified_date="2024-01-02",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return ArchitectureDependencyAnalyzer(timeout=5, enable_cache=False)


@pytest.fixture(autouse=True)
def _stub_container_client(monkeypatch):
    """By default report no manifests in the container, so ``analyze`` falls
    back to the local repo path and the existing local-fs test semantics hold
    (and no real network call is made to coding-containers)."""
    client = MagicMock()
    client.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    client.read_file = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "processing.architecture.dependency_analyzer.get_coding_containers_client",
        lambda: client,
    )
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDependencyEdgesExtraction:
    """test_dependency_edges_extraction"""

    async def test_analyze_with_packages_found(self, analyzer, tmp_path):
        """Full pipeline returns DependencyRiskReport when packages exist."""
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\n")

        packages = [_make_package()]
        vulns = [_make_vulnerability()]

        analyzer.extractor.extract_dependencies = MagicMock(return_value=packages)
        analyzer.vulnerability_scanner.scan_vulnerabilities = AsyncMock(
            return_value=VulnerabilityScanResult(vulns, packages_attempted=1, packages_scanned=1)
        )

        report = await analyzer.analyze(tmp_path, "cid-1")
        assert isinstance(report, DependencyRiskReport)
        assert report.codebase_id == "cid-1"
        assert len(report.packages) == 1
        assert len(report.vulnerabilities) == 1

    async def test_analyze_no_packages_returns_empty_report(self, analyzer, tmp_path):
        (tmp_path / "dummy.txt").write_text("nothing")
        analyzer.extractor.extract_dependencies = MagicMock(return_value=[])

        report = await analyzer.analyze(tmp_path, "cid-empty")
        assert report.codebase_id == "cid-empty"
        assert len(report.packages) == 0
        assert report.risk_score == 0.0

    async def test_analyze_with_transitive_dependencies(self, analyzer, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask==2.0.0\n")

        packages = [_make_package("flask", "2.0.0")]
        expanded = packages + [_make_package("werkzeug", "2.0.1")]

        analyzer.extractor.extract_dependencies = MagicMock(return_value=packages)
        analyzer.graph_builder.build_dependency_graph = AsyncMock(return_value=expanded)
        analyzer.vulnerability_scanner.scan_vulnerabilities = AsyncMock(
            return_value=VulnerabilityScanResult([], packages_attempted=1, packages_scanned=1)
        )

        report = await analyzer.analyze(tmp_path, "cid-trans", include_transitive=True)
        assert len(report.packages) == 2

    def test_create_empty_report(self, analyzer):
        report = analyzer._create_empty_report("test-cid")
        assert report.codebase_id == "test-cid"
        assert report.risk_score == 0.0
        assert report.summary.total_packages == 0
        assert report.packages == []
        assert report.vulnerabilities == []


class TestCycleHandling:
    """test_cycle_handling"""

    def test_detect_circular_dependencies_no_cycles(self, analyzer):
        packages = [_make_package("a"), _make_package("b")]
        dep_map = {"a": ["b"], "b": []}
        cycles = analyzer.detect_circular_dependencies(packages, dep_map)
        assert isinstance(cycles, list)

    def test_detect_circular_dependencies_with_cycles(self, analyzer):
        packages = [_make_package("a"), _make_package("b")]
        dep_map = {"a": ["b"], "b": ["a"]}
        cycles = analyzer.detect_circular_dependencies(packages, dep_map)
        assert isinstance(cycles, list)

    def test_get_ecosystem_breakdown(self, analyzer):
        packages = [
            _make_package("requests", ecosystem="pypi"),
            _make_package("flask", ecosystem="pypi"),
            _make_package("lodash", ecosystem="npm"),
        ]
        breakdown = analyzer.get_ecosystem_breakdown(packages)
        assert breakdown.get("pypi", 0) == 2
        assert breakdown.get("npm", 0) == 1

    def test_get_ecosystem_breakdown_empty(self, analyzer):
        breakdown = analyzer.get_ecosystem_breakdown([])
        assert breakdown == {}


class TestMissingImports:
    """test_missing_imports"""

    async def test_analyze_nonexistent_repo_returns_empty_report(self, analyzer):
        """The repo is read from the container, not the analyzer pod's local fs.
        With no local path and no container manifests, analyze returns an empty
        report instead of raising on a missing local path."""
        report = await analyzer.analyze(Path("/nonexistent/repo"), "cid")
        assert isinstance(report, DependencyRiskReport)
        assert report.packages == []

    async def test_analyze_empty_codebase_id_raises(self, analyzer, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            await analyzer.analyze(tmp_path, "   ")

    async def test_extraction_failure_raises_runtime_error(self, analyzer, tmp_path):
        (tmp_path / "requirements.txt").write_text("broken")
        analyzer.extractor.extract_dependencies = MagicMock(
            side_effect=RuntimeError("parse failure")
        )
        with pytest.raises(RuntimeError, match="Dependency analysis failed"):
            await analyzer.analyze(tmp_path, "cid-fail")

    def test_legacy_deduplicate_packages(self, analyzer):
        packages = [
            _make_package("requests", "2.28.0"),
            _make_package("requests", "2.28.0"),
            _make_package("flask", "2.0.0"),
        ]
        deduped = analyzer._deduplicate_packages(packages)
        assert isinstance(deduped, list)
        # Should deduplicate the repeated 'requests'
        names = [p.name for p in deduped]
        assert names.count("requests") == 1
        assert "flask" in names

    def test_get_cache_stats(self, analyzer):
        stats = analyzer.get_cache_stats()
        assert isinstance(stats, dict)


class TestMaterializeFromContainer:
    """The repo lives inside the analysis container, not on the analyzer pod's
    local fs — manifests must be read via coding-containers exec/read_file."""

    async def test_materialize_reads_manifests_preserving_paths(self, analyzer, monkeypatch):
        client = MagicMock()
        client.execute = AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": "/workspace/requirements.txt\n/workspace/sub/pyproject.toml\n",
            }
        )

        async def _read(container_name, file_path, tenant_id=None):
            return {
                "/workspace/requirements.txt": "requests==2.28.0\n",
                "/workspace/sub/pyproject.toml": "[project]\nname = 'x'\n",
            }.get(file_path)

        client.read_file = AsyncMock(side_effect=_read)
        monkeypatch.setattr(
            "processing.architecture.dependency_analyzer.get_coding_containers_client",
            lambda: client,
        )

        tmp = await analyzer._materialize_repo_from_container("cid", "/workspace")
        try:
            assert tmp is not None
            assert (tmp / "requirements.txt").read_text() == "requests==2.28.0\n"
            assert (tmp / "sub" / "pyproject.toml").exists()
            # find targeted the right container with the manifest names
            assert client.execute.call_args.args[0] == "cid"
            find_cmd = client.execute.call_args.args[1]
            assert find_cmd[0] == "find"
            assert "requirements*.txt" in find_cmd
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    async def test_manifest_reads_carry_the_owning_tenant(self, analyzer, monkeypatch):
        """The container is tenant-scoped; an untenanted read is 404'd.

        coding-containers resolves a call with no ``X-Tenant-Id`` as this
        service's own account, and its BOLA guard then answers 404 for the
        customer's workspace.
        """
        tenant = "0195f48b-d000-7000-8000-000000000003"
        client = MagicMock()
        client.execute = AsyncMock(
            return_value={"exit_code": 0, "stdout": "/workspace/requirements.txt\n"}
        )

        async def _read(container_name, file_path, tenant_id=None):
            return "requests==2.28.0\n"

        client.read_file = AsyncMock(side_effect=_read)
        monkeypatch.setattr(
            "processing.architecture.dependency_analyzer.get_coding_containers_client",
            lambda: client,
        )

        tmp = await analyzer._materialize_repo_from_container("cid", "/workspace", tenant_id=tenant)
        try:
            assert client.execute.call_args.kwargs.get("tenant_id") == tenant, (
                "the manifest `find` exec carried NO tenant -- the BOLA guard 404s it"
            )
            assert client.read_file.call_args.kwargs.get("tenant_id") == tenant
        finally:
            import shutil

            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    async def test_materialize_returns_none_when_no_manifests(self, analyzer, monkeypatch):
        client = MagicMock()
        client.execute = AsyncMock(return_value={"exit_code": 0, "stdout": ""})
        client.read_file = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "processing.architecture.dependency_analyzer.get_coding_containers_client",
            lambda: client,
        )
        assert await analyzer._materialize_repo_from_container("cid") is None

    async def test_analyze_uses_container_manifests_with_real_loaders(self, analyzer, monkeypatch):
        """End-to-end: manifests come from the container; the real pip loader
        parses them into packages even though /workspace is not on local disk."""
        client = MagicMock()
        client.execute = AsyncMock(
            return_value={"exit_code": 0, "stdout": "/workspace/requirements.txt\n"}
        )
        client.read_file = AsyncMock(return_value="requests==2.28.0\nflask>=2.0\n")
        monkeypatch.setattr(
            "processing.architecture.dependency_analyzer.get_coding_containers_client",
            lambda: client,
        )
        analyzer.vulnerability_scanner.scan_vulnerabilities = AsyncMock(
            return_value=VulnerabilityScanResult([], packages_attempted=1, packages_scanned=1)
        )

        report = await analyzer.analyze(Path("/workspace"), "cid-x")
        names = {p.name for p in report.packages}
        assert "requests" in names
        assert "flask" in names
