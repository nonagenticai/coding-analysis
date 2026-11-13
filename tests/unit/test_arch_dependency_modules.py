"""Unit tests for architecture dependency modules.

Covers:
- dependency_visualization.py (DependencyVulnerabilityScanner, DependencyVisualizer)
- dependency_graph.py (DependencyGraphBuilder)
- dependency_extraction.py (DependencyExtractor)
- dependency_analysis.py (DependencyAnalyzer)
- circular_detection.py (CircularDependencyDetector)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.architecture_models import (
    DependencyRiskReport,
    DependencyRiskSummary,
    ExternalPackage,
    Vulnerability,
)
from processing.architecture.circular_detection import CircularDependencyDetector
from processing.architecture.dependency_analysis import DependencyAnalyzer
from processing.architecture.dependency_extraction import (
    DependencyExtractionError,
    DependencyExtractor,
)
from processing.architecture.dependency_graph import DependencyGraphBuilder
from processing.architecture.dependency_visualization import (
    DependencyVisualizer,
    DependencyVulnerabilityScanner,
    VulnerabilityScanResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pkg(
    name: str = "requests",
    version: str = "2.28.0",
    ecosystem: str = "pypi",
    **kwargs,
) -> ExternalPackage:
    return ExternalPackage(name=name, version=version, ecosystem=ecosystem, **kwargs)


def _vuln(
    vuln_id: str = "GHSA-123",
    severity: str = "HIGH",
    pkg_name: str = "requests",
) -> Vulnerability:
    return Vulnerability(
        id=vuln_id,
        severity=severity,
        summary=f"Vuln in {pkg_name}",
        details="Details here",
        affected_versions=["<2.29.0"],
        fixed_versions=["2.29.0"],
        references=["https://example.com"],
        published_date="2024-01-01",
        modified_date="2024-01-02",
    )


# ============================================================================
# DependencyVulnerabilityScanner tests
# ============================================================================
class TestDependencyVulnerabilityScanner:
    """Tests for DependencyVulnerabilityScanner."""

    def test_init_defaults(self):
        scanner = DependencyVulnerabilityScanner()
        assert scanner.timeout == 30
        assert scanner.enable_cache is True
        assert scanner.batch_size == 10
        assert scanner.max_concurrent_batches == 5

    def test_init_custom(self):
        scanner = DependencyVulnerabilityScanner(
            timeout=60,
            enable_cache=False,
            cache_ttl_hours=12,
            batch_size=5,
            max_concurrent_batches=2,
        )
        assert scanner.timeout == 60
        assert scanner.enable_cache is False
        assert scanner.batch_size == 5
        assert scanner.max_concurrent_batches == 2

    # -- scan_vulnerabilities --

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities_empty(self):
        scanner = DependencyVulnerabilityScanner()
        result = await scanner.scan_vulnerabilities([])
        assert result.vulnerabilities == []
        assert result.packages_attempted == 0
        assert not result.inconclusive

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities_uses_cache(self):
        scanner = DependencyVulnerabilityScanner()
        pkg = _pkg()
        v = _vuln()
        # Pre-populate cache
        cache_key = scanner._get_cache_key(pkg)
        scanner.vulnerability_cache[cache_key] = {
            "data": [v],
            "timestamp": datetime.now(UTC),
        }
        result = await scanner.scan_vulnerabilities([pkg])
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].id == "GHSA-123"
        assert result.packages_scanned == 1
        assert not result.inconclusive

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities_queries_uncached(self):
        scanner = DependencyVulnerabilityScanner()
        pkg = _pkg()

        async def _fake(pkgs, res):
            res.vulnerabilities.append(_vuln())
            res.packages_scanned += len(pkgs)

        scanner._query_osv_batched = AsyncMock(side_effect=_fake)
        result = await scanner.scan_vulnerabilities([pkg])
        assert len(result.vulnerabilities) == 1
        scanner._query_osv_batched.assert_awaited_once()

    # -- _separate_cached_packages --

    def test_separate_cached_packages_cache_disabled(self):
        scanner = DependencyVulnerabilityScanner(enable_cache=False)
        pkgs = [_pkg()]
        cached, uncached = scanner._separate_cached_packages(pkgs)
        assert cached == []
        assert uncached == pkgs

    def test_separate_cached_packages_expired(self):
        scanner = DependencyVulnerabilityScanner(cache_ttl_hours=1)
        pkg = _pkg()
        cache_key = scanner._get_cache_key(pkg)
        scanner.vulnerability_cache[cache_key] = {
            "data": [],
            "timestamp": datetime.now(UTC) - timedelta(hours=2),
        }
        cached, uncached = scanner._separate_cached_packages([pkg])
        assert cached == []
        assert len(uncached) == 1
        assert cache_key not in scanner.vulnerability_cache

    def test_separate_cached_packages_valid(self):
        scanner = DependencyVulnerabilityScanner()
        pkg = _pkg()
        cache_key = scanner._get_cache_key(pkg)
        scanner.vulnerability_cache[cache_key] = {
            "data": [_vuln()],
            "timestamp": datetime.now(UTC),
        }
        cached, uncached = scanner._separate_cached_packages([pkg])
        assert len(cached) == 1
        assert uncached == []

    def test_separate_cached_packages_not_cached(self):
        scanner = DependencyVulnerabilityScanner()
        cached, uncached = scanner._separate_cached_packages([_pkg()])
        assert cached == []
        assert len(uncached) == 1

    # -- _get_cache_key --

    def test_get_cache_key_deterministic(self):
        scanner = DependencyVulnerabilityScanner()
        pkg = _pkg(name="flask", version="2.0", ecosystem="pypi")
        k1 = scanner._get_cache_key(pkg)
        k2 = scanner._get_cache_key(pkg)
        assert k1 == k2
        assert len(k1) == 16

    def test_get_cache_key_different_packages(self):
        scanner = DependencyVulnerabilityScanner()
        k1 = scanner._get_cache_key(_pkg(name="a"))
        k2 = scanner._get_cache_key(_pkg(name="b"))
        assert k1 != k2

    # -- _query_osv_batched --

    @pytest.mark.asyncio
    async def test_query_osv_batched_splits_batches(self):
        scanner = DependencyVulnerabilityScanner(batch_size=2)
        pkgs = [_pkg(name=f"pkg{i}") for i in range(5)]
        scanner._query_single_batch = AsyncMock(return_value=VulnerabilityScanResult())
        await scanner._query_osv_batched(pkgs, VulnerabilityScanResult())
        assert scanner._query_single_batch.await_count == 3  # ceil(5/2)

    @pytest.mark.asyncio
    async def test_query_osv_batched_handles_exceptions(self):
        scanner = DependencyVulnerabilityScanner(batch_size=2)
        pkgs = [_pkg(name=f"pkg{i}") for i in range(2)]
        scanner._query_single_batch = AsyncMock(side_effect=RuntimeError("fail"))
        result = VulnerabilityScanResult(packages_attempted=len(pkgs))
        await scanner._query_osv_batched(pkgs, result)
        assert result.vulnerabilities == []
        # A batch that raised must be recorded, not silently counted as clean.
        assert result.packages_scanned == 0
        assert result.errors

    # -- _query_single_batch --

    @pytest.mark.asyncio
    async def test_query_single_batch_no_aiohttp(self):
        scanner = DependencyVulnerabilityScanner()
        with patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", False):
            result = await scanner._query_single_batch([_pkg()])
        assert result.vulnerabilities == []
        assert result.packages_scanned == 0
        assert result.errors

    @pytest.mark.asyncio
    async def test_query_single_batch_success(self):
        scanner = DependencyVulnerabilityScanner()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {
                        "vulns": [
                            {
                                "id": "CVE-1",
                                "summary": "test",
                                "details": "d",
                                "affected": [],
                                "references": [],
                            }
                        ]
                    }
                ]
            }
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_single_batch([_pkg()])
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].id == "CVE-1"

    @pytest.mark.asyncio
    async def test_query_single_batch_timeout(self):
        scanner = DependencyVulnerabilityScanner()
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = AsyncMock(
                __aenter__=AsyncMock(side_effect=TimeoutError("timeout")),
                __aexit__=AsyncMock(return_value=False),
            )
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_single_batch([_pkg()])
        assert result.vulnerabilities == []
        # A timed-out batch is unscanned, never "clean".
        assert result.packages_scanned == 0
        assert result.errors

    # -- _query_individual_fallback --

    @pytest.mark.asyncio
    async def test_query_individual_fallback_no_aiohttp(self):
        scanner = DependencyVulnerabilityScanner()
        with patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", False):
            result = await scanner._query_individual_fallback([_pkg()])
        assert result.vulnerabilities == []
        assert result.packages_scanned == 0
        assert result.errors

    @pytest.mark.asyncio
    async def test_query_individual_fallback_success(self):
        scanner = DependencyVulnerabilityScanner()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "vulns": [
                    {
                        "id": "CVE-2",
                        "summary": "fallback vuln",
                        "details": "d",
                        "affected": [],
                        "references": [],
                    }
                ]
            }
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_individual_fallback([_pkg()])
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].id == "CVE-2"
        assert result.packages_scanned == 1

    @pytest.mark.asyncio
    async def test_query_individual_fallback_rate_limited(self):
        scanner = DependencyVulnerabilityScanner()
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_individual_fallback([_pkg()])
        assert result.vulnerabilities == []
        # Nothing was checked -> unscanned + a recorded reason, not clean.
        assert result.packages_scanned == 0
        assert result.errors

    @pytest.mark.asyncio
    async def test_query_individual_fallback_exception(self):
        scanner = DependencyVulnerabilityScanner()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=RuntimeError("conn error")),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_individual_fallback([_pkg()])
        assert result.vulnerabilities == []
        # Nothing was checked -> unscanned + a recorded reason, not clean.
        assert result.packages_scanned == 0
        assert result.errors

    # -- _parse_osv_vulnerability error path --

    def test_parse_osv_vulnerability_key_error(self):
        scanner = DependencyVulnerabilityScanner()
        # Cause a KeyError by having affected with bad structure
        data = {
            "id": "CVE-ERR",
            "summary": "test",
            "details": "test",
            "affected": [{"ranges": [{"events": [{"introduced": "0"}]}]}],
            "references": [],
            "published": "2024-01-01",
            "modified": "2024-01-01",
        }
        result = scanner._parse_osv_vulnerability(data)
        assert result is not None
        assert result.id == "CVE-ERR"

    # -- _query_single_batch rate limited / fallback --

    @pytest.mark.asyncio
    async def test_query_single_batch_rate_limited(self):
        scanner = DependencyVulnerabilityScanner()
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            # A 429 falls back to individual requests; the fallback's own
            # provenance must be propagated, not flattened into "clean".
            scanner._query_individual_fallback = AsyncMock(
                return_value=VulnerabilityScanResult(
                    packages_attempted=1, errors=["Rate limited by OSV.dev API"]
                )
            )
            result = await scanner._query_single_batch([_pkg()])
        assert result.vulnerabilities == []
        assert result.packages_scanned == 0
        assert result.errors == ["Rate limited by OSV.dev API"]

    @pytest.mark.asyncio
    async def test_query_single_batch_other_status(self):
        scanner = DependencyVulnerabilityScanner()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_visualization.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_visualization.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await scanner._query_single_batch([_pkg()])
        assert result.vulnerabilities == []
        # Nothing was checked -> unscanned + a recorded reason, not clean.
        assert result.packages_scanned == 0
        assert result.errors

    # -- _cache_package_vulnerabilities --

    def test_cache_package_vulnerabilities_stores(self):
        scanner = DependencyVulnerabilityScanner()
        pkg = _pkg()
        vulns = [_vuln()]
        scanner._cache_package_vulnerabilities(pkg, vulns)
        key = scanner._get_cache_key(pkg)
        assert key in scanner.vulnerability_cache
        assert scanner.vulnerability_cache[key]["data"] == vulns

    def test_cache_package_vulnerabilities_disabled(self):
        scanner = DependencyVulnerabilityScanner(enable_cache=False)
        scanner._cache_package_vulnerabilities(_pkg(), [_vuln()])
        assert len(scanner.vulnerability_cache) == 0

    # -- _deduplicate_vulnerabilities --

    def test_deduplicate_vulnerabilities(self):
        scanner = DependencyVulnerabilityScanner()
        v1 = _vuln(vuln_id="A")
        v2 = _vuln(vuln_id="B")
        v3 = _vuln(vuln_id="A")
        result = scanner._deduplicate_vulnerabilities([v1, v2, v3])
        assert len(result) == 2
        assert {v.id for v in result} == {"A", "B"}

    def test_deduplicate_vulnerabilities_empty(self):
        scanner = DependencyVulnerabilityScanner()
        assert scanner._deduplicate_vulnerabilities([]) == []

    # -- _parse_osv_vulnerability --

    def test_parse_osv_vulnerability_valid(self):
        scanner = DependencyVulnerabilityScanner()
        data = {
            "id": "CVE-2024-001",
            "summary": "A bug",
            "details": "Details",
            "affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.2.3"}]}]}],
            "references": [{"url": "https://cve.example.com"}],
            "published": "2024-01-01",
            "modified": "2024-06-01",
            "severity": [{"type": "CVSS_V3", "score": "9.5"}],
        }
        v = scanner._parse_osv_vulnerability(data)
        assert v is not None
        assert v.id == "CVE-2024-001"
        assert v.severity == "CRITICAL"
        assert v.affected_versions == ["0"]
        assert v.fixed_versions == ["1.2.3"]
        assert "https://cve.example.com" in v.references

    def test_parse_osv_vulnerability_no_id(self):
        scanner = DependencyVulnerabilityScanner()
        assert scanner._parse_osv_vulnerability({}) is None
        assert scanner._parse_osv_vulnerability({"id": ""}) is None

    def test_parse_osv_vulnerability_bad_data(self):
        scanner = DependencyVulnerabilityScanner()
        # Force a parsing error via bad severity data that triggers exception
        result = scanner._parse_osv_vulnerability({"id": "X", "severity": "not-a-list"})
        assert result is not None  # severity falls through to UNKNOWN
        assert result.severity == "UNKNOWN"

    # -- _extract_version_ranges --

    def test_extract_version_ranges(self):
        scanner = DependencyVulnerabilityScanner()
        data = {
            "affected": [
                {"ranges": [{"events": [{"introduced": "1.0"}, {"fixed": "2.0"}]}]},
                {"ranges": [{"events": [{"introduced": "3.0"}]}]},
            ]
        }
        affected, fixed = scanner._extract_version_ranges(data)
        assert affected == ["1.0", "3.0"]
        assert fixed == ["2.0"]

    def test_extract_version_ranges_empty(self):
        scanner = DependencyVulnerabilityScanner()
        affected, fixed = scanner._extract_version_ranges({})
        assert affected == []
        assert fixed == []

    # -- _extract_severity_from_osv --

    def test_extract_severity_cvss(self):
        scanner = DependencyVulnerabilityScanner()
        assert (
            scanner._extract_severity_from_osv({"severity": [{"type": "CVSS_V3", "score": "9.5"}]})
            == "CRITICAL"
        )

    def test_extract_severity_database_specific(self):
        scanner = DependencyVulnerabilityScanner()
        assert (
            scanner._extract_severity_from_osv({"database_specific": {"severity": "high"}})
            == "HIGH"
        )

    def test_extract_severity_unknown(self):
        scanner = DependencyVulnerabilityScanner()
        assert scanner._extract_severity_from_osv({}) == "UNKNOWN"

    def test_extract_severity_database_none(self):
        scanner = DependencyVulnerabilityScanner()
        assert (
            scanner._extract_severity_from_osv({"database_specific": {"severity": None}})
            == "UNKNOWN"
        )

    # -- _cvss_to_severity --

    def test_cvss_to_severity(self):
        scanner = DependencyVulnerabilityScanner()
        assert scanner._cvss_to_severity(9.5) == "CRITICAL"
        assert scanner._cvss_to_severity(9.0) == "CRITICAL"
        assert scanner._cvss_to_severity(7.5) == "HIGH"
        assert scanner._cvss_to_severity(7.0) == "HIGH"
        assert scanner._cvss_to_severity(5.0) == "MODERATE"
        assert scanner._cvss_to_severity(4.0) == "MODERATE"
        assert scanner._cvss_to_severity(3.9) == "LOW"
        assert scanner._cvss_to_severity(0.0) == "LOW"

    # -- clear_cache / get_cache_stats --

    def test_clear_cache(self):
        scanner = DependencyVulnerabilityScanner()
        scanner.vulnerability_cache["key"] = {"data": [], "timestamp": datetime.now(UTC)}
        scanner.clear_cache()
        assert len(scanner.vulnerability_cache) == 0

    def test_get_cache_stats_empty(self):
        scanner = DependencyVulnerabilityScanner()
        stats = scanner.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["expired_entries"] == 0

    def test_get_cache_stats_mixed(self):
        scanner = DependencyVulnerabilityScanner(cache_ttl_hours=1)
        scanner.vulnerability_cache["valid"] = {
            "data": [],
            "timestamp": datetime.now(UTC),
        }
        scanner.vulnerability_cache["expired"] = {
            "data": [],
            "timestamp": datetime.now(UTC) - timedelta(hours=2),
        }
        stats = scanner.get_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1


# ============================================================================
# DependencyVisualizer tests
# ============================================================================
class TestDependencyVisualizer:
    """Tests for DependencyVisualizer."""

    def test_generate_dependency_tree_empty(self):
        viz = DependencyVisualizer()
        assert viz.generate_dependency_tree_text([]) == "No dependencies found."

    def test_generate_dependency_tree_basic(self):
        viz = DependencyVisualizer()
        pkgs = [_pkg(name="flask", version="2.0.0", license="MIT")]
        text = viz.generate_dependency_tree_text(pkgs)
        assert "flask" in text
        assert "2.0.0" in text
        assert "MIT" in text

    def test_generate_dependency_tree_with_severity(self):
        viz = DependencyVisualizer()
        pkgs = [_pkg(name="vulnerable-pkg", severity="HIGH")]
        text = viz.generate_dependency_tree_text(pkgs)
        assert "Security: HIGH" in text

    def test_generate_dependency_tree_truncation(self):
        viz = DependencyVisualizer()
        pkgs = [_pkg(name=f"pkg{i}") for i in range(25)]
        text = viz.generate_dependency_tree_text(pkgs)
        assert "and 5 more packages" in text

    def test_generate_vulnerability_summary_empty(self):
        viz = DependencyVisualizer()
        assert viz.generate_vulnerability_summary_text([]) == "No vulnerabilities found."

    def test_generate_vulnerability_summary_basic(self):
        viz = DependencyVisualizer()
        vulns = [
            _vuln(vuln_id="V1", severity="CRITICAL"),
            _vuln(vuln_id="V2", severity="HIGH"),
            _vuln(vuln_id="V3", severity="MODERATE"),
            _vuln(vuln_id="V4", severity="LOW"),
        ]
        text = viz.generate_vulnerability_summary_text(vulns)
        assert "Total Vulnerabilities: 4" in text
        assert "Critical: 1" in text
        assert "High: 1" in text
        assert "Moderate: 1" in text
        assert "Low: 1" in text

    def test_generate_vulnerability_summary_truncation(self):
        viz = DependencyVisualizer()
        vulns = [_vuln(vuln_id=f"V{i}") for i in range(15)]
        text = viz.generate_vulnerability_summary_text(vulns)
        # Only top 10 should appear in details
        lines = text.split("\n")
        numbered = [line for line in lines if line and line[0].isdigit()]
        assert len(numbered) == 10


# ============================================================================
# DependencyGraphBuilder tests
# ============================================================================
class TestDependencyGraphBuilder:
    """Tests for DependencyGraphBuilder."""

    def test_init(self):
        builder = DependencyGraphBuilder(timeout=60)
        assert builder.timeout == 60

    # -- build_dependency_graph --

    @pytest.mark.asyncio
    async def test_build_dependency_graph_success(self):
        builder = DependencyGraphBuilder()
        pkgs = [_pkg()]
        builder.analyze_transitive_dependencies = AsyncMock(return_value=pkgs)
        result = await builder.build_dependency_graph(pkgs, max_depth=2)
        assert result == pkgs

    @pytest.mark.asyncio
    async def test_build_dependency_graph_error(self):
        builder = DependencyGraphBuilder()
        builder.analyze_transitive_dependencies = AsyncMock(side_effect=ValueError("boom"))
        with pytest.raises(RuntimeError, match="Dependency graph building failed"):
            await builder.build_dependency_graph([_pkg()])

    # -- analyze_transitive_dependencies --

    @pytest.mark.asyncio
    async def test_analyze_transitive_no_deps(self):
        builder = DependencyGraphBuilder()
        builder._resolve_package_dependencies = AsyncMock(return_value=[])
        result = await builder.analyze_transitive_dependencies([_pkg()], max_depth=2)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_analyze_transitive_with_deps(self):
        builder = DependencyGraphBuilder()
        child = _pkg(name="child")
        builder._resolve_package_dependencies = AsyncMock(
            side_effect=lambda pkg: [child] if pkg.name == "requests" else []
        )
        result = await builder.analyze_transitive_dependencies([_pkg()], max_depth=3)
        names = {p.name for p in result}
        assert "requests" in names
        assert "child" in names

    @pytest.mark.asyncio
    async def test_analyze_transitive_max_depth(self):
        builder = DependencyGraphBuilder()
        call_count = 0

        async def mock_resolve(pkg):
            nonlocal call_count
            call_count += 1
            return [_pkg(name=f"dep{call_count}")]

        builder._resolve_package_dependencies = mock_resolve
        result = await builder.analyze_transitive_dependencies([_pkg()], max_depth=1)
        # Depth 0 processes requests, depth 1 would be max => stops
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_analyze_transitive_skips_processed(self):
        builder = DependencyGraphBuilder()
        pkg = _pkg()
        resolve_count = 0

        async def mock_resolve(p):
            nonlocal resolve_count
            resolve_count += 1
            return []

        builder._resolve_package_dependencies = mock_resolve
        # Pass same package twice
        await builder.analyze_transitive_dependencies([pkg, pkg], max_depth=3)
        assert resolve_count == 1

    @pytest.mark.asyncio
    async def test_analyze_transitive_handles_resolve_error(self):
        builder = DependencyGraphBuilder()
        builder._resolve_package_dependencies = AsyncMock(side_effect=RuntimeError("API down"))
        result = await builder.analyze_transitive_dependencies([_pkg()], max_depth=3)
        assert len(result) == 1  # original package still included

    # -- _resolve_package_dependencies --

    @pytest.mark.asyncio
    async def test_resolve_package_pypi(self):
        builder = DependencyGraphBuilder()
        builder._resolve_pypi_dependencies = AsyncMock(return_value=[_pkg(name="dep1")])
        result = await builder._resolve_package_dependencies(_pkg(ecosystem="pypi"))
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_resolve_package_npm(self):
        builder = DependencyGraphBuilder()
        builder._resolve_npm_dependencies = AsyncMock(return_value=[])
        result = await builder._resolve_package_dependencies(_pkg(ecosystem="npm"))
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_package_crates(self):
        builder = DependencyGraphBuilder()
        builder._resolve_crates_dependencies = AsyncMock(return_value=[])
        result = await builder._resolve_package_dependencies(_pkg(ecosystem="crates.io"))
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_package_unknown_ecosystem(self):
        builder = DependencyGraphBuilder()
        result = await builder._resolve_package_dependencies(_pkg(ecosystem="unknown"))
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_package_error(self):
        builder = DependencyGraphBuilder()
        builder._resolve_pypi_dependencies = AsyncMock(side_effect=Exception("err"))
        result = await builder._resolve_package_dependencies(_pkg(ecosystem="pypi"))
        assert result == []

    # -- _resolve_pypi_dependencies --

    @pytest.mark.asyncio
    async def test_resolve_pypi_no_aiohttp(self):
        builder = DependencyGraphBuilder()
        with patch("processing.architecture.dependency_graph.HAS_AIOHTTP", False):
            result = await builder._resolve_pypi_dependencies(_pkg())
        assert result == []

    # -- _resolve_npm_dependencies --

    @pytest.mark.asyncio
    async def test_resolve_npm_no_aiohttp(self):
        builder = DependencyGraphBuilder()
        with patch("processing.architecture.dependency_graph.HAS_AIOHTTP", False):
            result = await builder._resolve_npm_dependencies(_pkg(ecosystem="npm"))
        assert result == []

    # -- _resolve_crates_dependencies --

    @pytest.mark.asyncio
    async def test_resolve_crates_no_aiohttp(self):
        builder = DependencyGraphBuilder()
        with patch("processing.architecture.dependency_graph.HAS_AIOHTTP", False):
            result = await builder._resolve_crates_dependencies(_pkg(ecosystem="crates.io"))
        assert result == []

    # -- _parse_pypi_requirement --

    def test_parse_pypi_requirement_with_version(self):
        builder = DependencyGraphBuilder()
        name, ver = builder._parse_pypi_requirement("flask>=2.0.0")
        assert name == "flask"
        assert ver == "2.0.0"

    def test_parse_pypi_requirement_name_only(self):
        builder = DependencyGraphBuilder()
        name, ver = builder._parse_pypi_requirement("flask")
        assert name == "flask"
        assert ver is None

    def test_parse_pypi_requirement_with_extras(self):
        builder = DependencyGraphBuilder()
        name, ver = builder._parse_pypi_requirement("pkg[extra]==1.0")
        assert name == "pkg"
        assert ver == "1.0"

    def test_parse_pypi_requirement_invalid(self):
        builder = DependencyGraphBuilder()
        name, ver = builder._parse_pypi_requirement("???!!!")
        assert name is None
        assert ver is None

    # -- build_dependency_tree --

    def test_build_dependency_tree(self):
        builder = DependencyGraphBuilder()
        pkgs = [_pkg(name="flask", version="2.0"), _pkg(name="click", version="8.0")]
        tree = builder.build_dependency_tree(pkgs)
        assert "flask" in tree
        assert tree["flask"]["version"] == "2.0"
        assert "click" in tree
        assert tree["click"]["ecosystem"] == "pypi"

    def test_build_dependency_tree_empty(self):
        builder = DependencyGraphBuilder()
        assert builder.build_dependency_tree([]) == {}

    def test_build_dependency_tree_uses_current_version(self):
        builder = DependencyGraphBuilder()
        pkg = _pkg(name="mylib", version="")
        pkg.current_version = "3.0"
        tree = builder.build_dependency_tree([pkg])
        assert tree["mylib"]["version"] == "3.0"

    # -- find_dependency_paths --

    def test_find_dependency_paths_simple(self):
        builder = DependencyGraphBuilder()
        pkgs = [_pkg(name="target")]
        paths = builder.find_dependency_paths(pkgs, "target")
        assert any("target" in path for path in paths)

    def test_find_dependency_paths_not_found(self):
        builder = DependencyGraphBuilder()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        paths = builder.find_dependency_paths(pkgs, "nonexistent")
        assert paths == []

    def test_find_dependency_paths_cycle_protection(self):
        """find_dependency_paths should not recurse infinitely with duplicate names."""
        builder = DependencyGraphBuilder()
        pkgs = [_pkg(name="a"), _pkg(name="b"), _pkg(name="target")]
        paths = builder.find_dependency_paths(pkgs, "target")
        # At least one path should contain "target"
        assert any("target" in path for path in paths)

    # -- _resolve_pypi_dependencies with aiohttp (mocked) --

    @pytest.mark.asyncio
    async def test_resolve_pypi_dependencies_success(self):
        builder = DependencyGraphBuilder()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"info": {"requires_dist": ["urllib3>=1.21.1", "certifi>=2017.4.17"]}}
        )
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_pypi_dependencies(_pkg())
        assert len(result) == 2
        names = {r.name for r in result}
        assert "urllib3" in names
        assert "certifi" in names

    @pytest.mark.asyncio
    async def test_resolve_pypi_dependencies_exception(self):
        builder = DependencyGraphBuilder()
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = AsyncMock(
                __aenter__=AsyncMock(side_effect=RuntimeError("connection error")),
                __aexit__=AsyncMock(return_value=False),
            )
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_pypi_dependencies(_pkg())
        assert result == []

    # -- _resolve_npm_dependencies with aiohttp (mocked) --

    @pytest.mark.asyncio
    async def test_resolve_npm_dependencies_success(self):
        builder = DependencyGraphBuilder()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "dist-tags": {"latest": "1.0.0"},
                "versions": {
                    "1.0.0": {
                        "dependencies": {
                            "lodash": "^4.17.0",
                            "chalk": "^5.0.0",
                        }
                    }
                },
            }
        )
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_npm_dependencies(
                _pkg(name="express", version="*", ecosystem="npm")
            )
        assert len(result) == 2
        names = {r.name for r in result}
        assert "lodash" in names
        assert "chalk" in names

    @pytest.mark.asyncio
    async def test_resolve_npm_dependencies_exception(self):
        builder = DependencyGraphBuilder()
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = AsyncMock(
                __aenter__=AsyncMock(side_effect=RuntimeError("connection error")),
                __aexit__=AsyncMock(return_value=False),
            )
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_npm_dependencies(_pkg(name="express", ecosystem="npm"))
        assert result == []

    # -- _resolve_crates_dependencies with aiohttp (mocked) --

    @pytest.mark.asyncio
    async def test_resolve_crates_dependencies_success(self):
        builder = DependencyGraphBuilder()
        mock_crate_resp = AsyncMock()
        mock_crate_resp.status = 200
        mock_crate_resp.json = AsyncMock(return_value={"crate": {"newest_version": "1.0.0"}})
        mock_deps_resp = AsyncMock()
        mock_deps_resp.status = 200
        mock_deps_resp.json = AsyncMock(
            return_value={
                "dependencies": [
                    {"crate_id": "serde", "req": "^1.0", "kind": "normal"},
                    {"crate_id": "rand", "req": "^0.8", "kind": "dev"},
                ]
            }
        )
        # Build mocked session with two sequential GET calls
        get_call_count = 0

        def make_get_context(url):
            nonlocal get_call_count
            get_call_count += 1
            if get_call_count == 1:
                return AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_crate_resp),
                    __aexit__=AsyncMock(return_value=False),
                )
            return AsyncMock(
                __aenter__=AsyncMock(return_value=mock_deps_resp),
                __aexit__=AsyncMock(return_value=False),
            )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=make_get_context)
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_crates_dependencies(
                _pkg(name="tokio", version="*", ecosystem="crates.io")
            )
        # Only normal deps, not dev deps
        assert len(result) == 1
        assert result[0].name == "serde"

    @pytest.mark.asyncio
    async def test_resolve_crates_dependencies_exception(self):
        builder = DependencyGraphBuilder()
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = AsyncMock(
                __aenter__=AsyncMock(side_effect=RuntimeError("connection error")),
                __aexit__=AsyncMock(return_value=False),
            )
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_crates_dependencies(
                _pkg(name="tokio", ecosystem="crates.io")
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_crates_dependencies_not_found(self):
        builder = DependencyGraphBuilder()
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        mock_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
        with (
            patch("processing.architecture.dependency_graph.HAS_AIOHTTP", True),
            patch("processing.architecture.dependency_graph.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession.return_value = mock_cm
            mock_aiohttp.ClientTimeout = MagicMock
            result = await builder._resolve_crates_dependencies(
                _pkg(name="nonexistent", version="*", ecosystem="crates.io")
            )
        assert result == []


# ============================================================================
# DependencyExtractor tests
# ============================================================================
class TestDependencyExtractor:
    """Tests for DependencyExtractor."""

    def test_init(self):
        extractor = DependencyExtractor()
        assert len(extractor.loaders) == 6

    # -- extract_dependencies --

    def test_extract_dependencies_path_not_exists(self, tmp_path):
        extractor = DependencyExtractor()
        bad_path = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            extractor.extract_dependencies(bad_path)

    def test_extract_dependencies_no_loaders_match(self, tmp_path):
        extractor = DependencyExtractor()
        for loader in extractor.loaders:
            loader.detect = MagicMock(return_value=False)
        result = extractor.extract_dependencies(tmp_path)
        assert result == []

    def test_extract_dependencies_loader_detects(self, tmp_path):
        extractor = DependencyExtractor()
        pkg = _pkg(name="flask")
        for loader in extractor.loaders:
            loader.detect = MagicMock(return_value=False)
        # Make first loader detect and return packages
        extractor.loaders[0].detect = MagicMock(return_value=True)
        extractor.loaders[0].parse = MagicMock(return_value=[pkg])
        result = extractor.extract_dependencies(tmp_path)
        assert len(result) == 1
        assert result[0].name == "flask"

    def test_extract_dependencies_loader_error(self, tmp_path):
        """Every loader erroring is INCONCLUSIVE, not "no dependencies".

        This used to assert `result == []` — the same value a genuinely
        dependency-free repo returns. The caller scored that 0.0, reporting a clean
        bill of health for a dependency set that was never read. A loader that
        cannot even run `detect()` leaves us unable to say whether manifests exist,
        so the extraction now fails loud and the analyzer abstains.
        """
        extractor = DependencyExtractor()
        for loader in extractor.loaders:
            loader.detect = MagicMock(side_effect=RuntimeError("broke"))

        with pytest.raises(DependencyExtractionError) as exc:
            extractor.extract_dependencies(tmp_path)

        assert len(exc.value.failures) == len(extractor.loaders)

    def test_extract_dependencies_deduplicates(self, tmp_path):
        extractor = DependencyExtractor()
        pkg1 = _pkg(name="flask", version="*")
        pkg2 = _pkg(name="Flask", version="==2.0")
        for loader in extractor.loaders:
            loader.detect = MagicMock(return_value=False)
        extractor.loaders[0].detect = MagicMock(return_value=True)
        extractor.loaders[0].parse = MagicMock(return_value=[pkg1, pkg2])
        result = extractor.extract_dependencies(tmp_path)
        assert len(result) == 1

    # -- deduplicate_packages --

    def test_deduplicate_packages_keeps_specific(self):
        extractor = DependencyExtractor()
        p1 = _pkg(name="flask", version="*")
        p2 = _pkg(name="Flask", version="2.0")
        result = extractor.deduplicate_packages([p1, p2])
        assert len(result) == 1
        # p2 is more specific (non-wildcard)
        assert result[0].version == "2.0"

    def test_deduplicate_packages_empty(self):
        extractor = DependencyExtractor()
        assert extractor.deduplicate_packages([]) == []

    def test_deduplicate_packages_unique(self):
        extractor = DependencyExtractor()
        result = extractor.deduplicate_packages([_pkg(name="a"), _pkg(name="b")])
        assert len(result) == 2

    # -- _is_more_specific_version --

    def test_is_more_specific_exact_vs_constraint(self):
        extractor = DependencyExtractor()
        assert extractor._is_more_specific_version("==1.0", ">=1.0") is True
        assert extractor._is_more_specific_version(">=1.0", "==1.0") is False

    def test_is_more_specific_wildcard(self):
        extractor = DependencyExtractor()
        assert extractor._is_more_specific_version("1.0", "*") is True
        assert extractor._is_more_specific_version("*", "1.0") is False

    def test_is_more_specific_both_same(self):
        extractor = DependencyExtractor()
        assert extractor._is_more_specific_version("1.0", "1.0") is False

    # -- enrich_packages --

    def test_enrich_packages(self):
        extractor = DependencyExtractor()
        extractor._get_license_info = MagicMock(return_value="MIT")
        extractor._get_latest_version = MagicMock(return_value="3.0.0")
        pkgs = [_pkg(name="flask", version="2.0")]
        result = extractor.enrich_packages(pkgs)
        assert len(result) == 1
        assert result[0].license == "MIT"
        assert result[0].latest_version == "3.0.0"

    # -- _get_latest_version --

    def test_get_latest_version_success(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"info": {"version": "3.0.0"}}
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_latest_version("flask")
        assert result == "3.0.0"

    def test_get_latest_version_not_found(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_latest_version("nonexistent")
        assert result is None

    def test_get_latest_version_exception(self):
        extractor = DependencyExtractor()
        import httpx

        with patch(
            "processing.architecture.dependency_extraction.httpx.get",
            side_effect=httpx.HTTPError("err"),
        ):
            result = extractor._get_latest_version("flask")
        assert result is None

    # -- _get_license_info --

    def test_get_license_info_from_license_field(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"info": {"license": "MIT", "classifiers": []}}
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_license_info("flask")
        assert result == "MIT"

    def test_get_license_info_from_classifier(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "info": {
                "license": "",
                "classifiers": ["License :: OSI Approved :: BSD License"],
            }
        }
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_license_info("flask")
        assert result == "BSD License"

    def test_get_license_info_none(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"info": {"license": "", "classifiers": []}}
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_license_info("flask")
        assert result is None

    def test_get_license_info_osi_only(self):
        extractor = DependencyExtractor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "info": {
                "license": "",
                "classifiers": ["License :: OSI Approved"],
            }
        }
        with patch(
            "processing.architecture.dependency_extraction.httpx.get", return_value=mock_resp
        ):
            result = extractor._get_license_info("flask")
        # "OSI Approved" is skipped, so falls through to empty license -> None
        assert result is None

    # -- parse_pypi_requirement --

    def test_parse_pypi_requirement_version(self):
        extractor = DependencyExtractor()
        name, ver = extractor.parse_pypi_requirement("flask>=2.0")
        assert name == "flask"
        assert ver == "2.0"

    def test_parse_pypi_requirement_name_only(self):
        extractor = DependencyExtractor()
        name, ver = extractor.parse_pypi_requirement("flask")
        assert name == "flask"
        assert ver is None

    def test_parse_pypi_requirement_extras(self):
        extractor = DependencyExtractor()
        name, ver = extractor.parse_pypi_requirement("pkg[extra]==1.0")
        assert name == "pkg"
        assert ver == "1.0"

    def test_parse_pypi_requirement_invalid(self):
        extractor = DependencyExtractor()
        name, ver = extractor.parse_pypi_requirement("!!!")
        assert name is None
        assert ver is None

    # -- clean_version_for_osv --

    def test_clean_version_valid(self):
        extractor = DependencyExtractor()
        assert extractor.clean_version_for_osv("==1.2.3") == "1.2.3"
        assert extractor.clean_version_for_osv(">=2.0.0") == "2.0.0"

    def test_clean_version_wildcard(self):
        extractor = DependencyExtractor()
        assert extractor.clean_version_for_osv("*") is None
        assert extractor.clean_version_for_osv("") is None

    def test_clean_version_invalid(self):
        extractor = DependencyExtractor()
        assert extractor.clean_version_for_osv("not-a-version") is None

    # -- get_outdated_packages --

    def test_get_outdated_packages(self):
        extractor = DependencyExtractor()
        pkg = _pkg(name="flask", version="1.0.0")
        pkg = ExternalPackage(
            name="flask",
            version="1.0.0",
            current_version="1.0.0",
            latest_version="2.0.0",
            ecosystem="pypi",
        )
        result = extractor.get_outdated_packages([pkg])
        assert len(result) == 1

    def test_get_outdated_packages_up_to_date(self):
        extractor = DependencyExtractor()
        pkg = ExternalPackage(
            name="flask",
            version="2.0.0",
            current_version="2.0.0",
            latest_version="2.0.0",
            ecosystem="pypi",
        )
        result = extractor.get_outdated_packages([pkg])
        assert len(result) == 0

    def test_get_outdated_packages_no_latest(self):
        extractor = DependencyExtractor()
        result = extractor.get_outdated_packages([_pkg()])
        assert result == []

    def test_get_outdated_packages_invalid_version(self):
        extractor = DependencyExtractor()
        pkg = ExternalPackage(
            name="x",
            version="bad",
            current_version="bad",
            latest_version="2.0.0",
            ecosystem="pypi",
        )
        result = extractor.get_outdated_packages([pkg])
        # Invalid version comparison → considered outdated
        assert len(result) == 1

    # -- get_vulnerable_packages --

    def test_get_vulnerable_packages(self):
        extractor = DependencyExtractor()
        p1 = _pkg(name="a", severity="high")
        p2 = _pkg(name="b", severity="low")
        p3 = _pkg(name="c", severity=None)
        p4 = _pkg(name="d", severity="critical")
        result = extractor.get_vulnerable_packages([p1, p2, p3, p4])
        names = {p.name for p in result}
        assert "a" in names
        assert "d" in names
        assert "b" not in names
        assert "c" not in names


# ============================================================================
# DependencyAnalyzer tests
# ============================================================================
class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""

    def test_init(self):
        analyzer = DependencyAnalyzer()
        assert analyzer is not None

    # -- generate_risk_report --

    def test_generate_risk_report(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg()]
        vulns = [_vuln(severity="CRITICAL")]
        report = analyzer.generate_risk_report("codebase-1", pkgs, vulns)
        assert isinstance(report, DependencyRiskReport)
        assert report.codebase_id == "codebase-1"
        assert len(report.packages) == 1
        assert len(report.vulnerabilities) == 1
        assert report.risk_score > 0.0

    def test_generate_risk_report_empty(self):
        analyzer = DependencyAnalyzer()
        report = analyzer.generate_risk_report("cb-1", [], [])
        assert report.risk_score == 0.0
        assert report.summary.total_packages == 0

    def test_generate_risk_report_empty_id(self):
        analyzer = DependencyAnalyzer()
        with pytest.raises(ValueError, match="cannot be empty"):
            analyzer.generate_risk_report("  ", [], [])

    # -- calculate_risk_score --

    def test_calculate_risk_score_empty(self):
        analyzer = DependencyAnalyzer()
        assert analyzer.calculate_risk_score([]) == 0.0

    def test_calculate_risk_score_critical(self):
        analyzer = DependencyAnalyzer()
        vulns = [_vuln(severity="CRITICAL")]
        score = analyzer.calculate_risk_score(vulns)
        assert score == 10.0

    def test_calculate_risk_score_mixed(self):
        analyzer = DependencyAnalyzer()
        vulns = [
            _vuln(vuln_id="1", severity="CRITICAL"),
            _vuln(vuln_id="2", severity="LOW"),
        ]
        score = analyzer.calculate_risk_score(vulns)
        assert 0.0 < score <= 10.0

    def test_calculate_risk_score_many_vulns(self):
        analyzer = DependencyAnalyzer()
        vulns = [_vuln(vuln_id=f"V{i}", severity="MODERATE") for i in range(20)]
        score = analyzer.calculate_risk_score(vulns)
        assert score <= 10.0  # capped

    def test_calculate_risk_score_unknown_severity(self):
        """An ungraded finding is unknown risk, not zero risk.

        This asserted 0.0, which is what made a real advisory we could not
        grade indistinguishable from a clean package -- and let one CRITICAL
        among four UNKNOWNs average out to a reassuring 2.8.
        """
        analyzer = DependencyAnalyzer()
        vulns = [_vuln(severity="UNKNOWN")]
        score = analyzer.calculate_risk_score(vulns)
        assert score > 0.0

    # -- generate_summary --

    def test_generate_summary(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(), _pkg(name="click")]
        vulns = [
            _vuln(vuln_id="V1", severity="HIGH"),
            _vuln(vuln_id="V2", severity="LOW"),
        ]
        summary = analyzer.generate_summary(pkgs, vulns)
        assert isinstance(summary, DependencyRiskSummary)
        assert summary.total_packages == 2
        assert summary.total_vulnerabilities == 2
        assert summary.high_vulnerabilities == 1
        assert summary.low_vulnerabilities == 1

    def test_generate_summary_empty(self):
        analyzer = DependencyAnalyzer()
        summary = analyzer.generate_summary([], [])
        assert summary.total_packages == 0
        assert summary.total_vulnerabilities == 0

    # -- _get_vulnerable_packages_count --

    def test_get_vulnerable_packages_count(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="requests")]
        vulns = [_vuln(vuln_id="GHSA-requests-123", severity="HIGH", pkg_name="requests")]
        count = analyzer._get_vulnerable_packages_count(pkgs, vulns)
        assert count == 1

    def test_get_vulnerable_packages_count_no_match(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="flask")]
        vulns = [_vuln(vuln_id="CVE-2024-0001", severity="HIGH", pkg_name="django")]
        count = analyzer._get_vulnerable_packages_count(pkgs, vulns)
        assert count == 0

    # -- categorize_vulnerabilities_by_severity --

    def test_categorize_vulnerabilities(self):
        analyzer = DependencyAnalyzer()
        vulns = [
            _vuln(vuln_id="1", severity="CRITICAL"),
            _vuln(vuln_id="2", severity="HIGH"),
            _vuln(vuln_id="3", severity="HIGH"),
            _vuln(vuln_id="4", severity="MODERATE"),
            _vuln(vuln_id="5", severity="LOW"),
        ]
        cats = analyzer.categorize_vulnerabilities_by_severity(vulns)
        assert cats["critical_vulnerabilities"] == 1
        assert cats["high_vulnerabilities"] == 2
        assert cats["moderate_vulnerabilities"] == 1
        assert cats["low_vulnerabilities"] == 1

    def test_categorize_vulnerabilities_empty(self):
        analyzer = DependencyAnalyzer()
        cats = analyzer.categorize_vulnerabilities_by_severity([])
        assert all(v == 0 for v in cats.values())

    # -- analyze_dependency_health --

    def test_analyze_dependency_health_empty(self):
        analyzer = DependencyAnalyzer()
        health = analyzer.analyze_dependency_health([])
        assert health["total_packages"] == 0
        assert health["health_score"] == 10.0

    def test_analyze_dependency_health_perfect(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="flask", license="MIT")]
        health = analyzer.analyze_dependency_health(pkgs)
        assert health["total_packages"] == 1
        assert health["health_score"] == 10.0

    def test_analyze_dependency_health_degraded(self):
        analyzer = DependencyAnalyzer()
        pkgs = [
            ExternalPackage(
                name="vuln",
                version="1.0",
                ecosystem="pypi",
                severity="HIGH",
                license="Unknown",
                current_version="1.0",
                latest_version="2.0",
            ),
        ]
        health = analyzer.analyze_dependency_health(pkgs)
        assert health["health_score"] < 10.0
        assert health["outdated_packages"] == 1
        assert health["packages_with_vulnerabilities"] == 1
        assert health["unlicensed_packages"] == 1

    # -- get_ecosystem_breakdown --

    def test_get_ecosystem_breakdown(self):
        analyzer = DependencyAnalyzer()
        pkgs = [
            _pkg(ecosystem="pypi"),
            _pkg(name="b", ecosystem="pypi"),
            _pkg(name="c", ecosystem="npm"),
        ]
        breakdown = analyzer.get_ecosystem_breakdown(pkgs)
        assert breakdown["pypi"] == 2
        assert breakdown["npm"] == 1

    def test_get_ecosystem_breakdown_empty(self):
        analyzer = DependencyAnalyzer()
        assert analyzer.get_ecosystem_breakdown([]) == {}

    # -- identify_high_risk_packages --

    def test_identify_high_risk_packages_with_vulns(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="requests")]
        vulns = [
            _vuln(vuln_id="requests-CVE1", severity="CRITICAL"),
            _vuln(vuln_id="requests-CVE2", severity="HIGH"),
        ]
        result = analyzer.identify_high_risk_packages(pkgs, vulns)
        assert len(result) >= 1
        assert result[0]["name"] == "requests"
        assert result[0]["risk_score"] >= 2.0

    def test_identify_high_risk_packages_below_threshold(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="safe", license="MIT")]
        vulns = []
        result = analyzer.identify_high_risk_packages(pkgs, vulns)
        assert result == []

    def test_identify_high_risk_packages_outdated_plus_no_license(self):
        analyzer = DependencyAnalyzer()
        pkg = ExternalPackage(
            name="old-pkg",
            version="1.0",
            ecosystem="pypi",
            current_version="1.0",
            latest_version="5.0",
            license="Unknown",
        )
        result = analyzer.identify_high_risk_packages([pkg], [])
        # outdated = 1.0, no license = 0.5 → total 1.5 < 2.0 threshold
        assert result == []

    def test_identify_high_risk_sorts_by_score(self):
        analyzer = DependencyAnalyzer()
        pkgs = [_pkg(name="low"), _pkg(name="high")]
        vulns = [
            _vuln(vuln_id="low-v", severity="HIGH"),
            _vuln(vuln_id="high-v1", severity="CRITICAL"),
            _vuln(vuln_id="high-v2", severity="CRITICAL"),
        ]
        result = analyzer.identify_high_risk_packages(pkgs, vulns)
        if len(result) >= 2:
            assert result[0]["risk_score"] >= result[1]["risk_score"]


# ============================================================================
# CircularDependencyDetector tests
# ============================================================================
class TestCircularDependencyDetector:
    """Tests for CircularDependencyDetector."""

    def test_init(self):
        detector = CircularDependencyDetector()
        assert detector is not None

    # -- detect_circular_dependencies --

    def test_detect_no_cycles(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b"), _pkg(name="c")]
        dep_map = {"a": ["b"], "b": ["c"], "c": []}
        cycles = detector.detect_circular_dependencies(pkgs, dep_map)
        assert cycles == []

    def test_detect_simple_cycle(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = {"a": ["b"], "b": ["a"]}
        cycles = detector.detect_circular_dependencies(pkgs, dep_map)
        assert len(cycles) >= 1
        # The cycle should contain both a and b
        all_nodes = set()
        for c in cycles:
            all_nodes.update(c)
        assert "a" in all_nodes
        assert "b" in all_nodes

    def test_detect_three_node_cycle(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b"), _pkg(name="c")]
        dep_map = {"a": ["b"], "b": ["c"], "c": ["a"]}
        cycles = detector.detect_circular_dependencies(pkgs, dep_map)
        assert len(cycles) >= 1

    def test_detect_empty_dep_map(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        cycles = detector.detect_circular_dependencies(pkgs, {})
        assert cycles == []

    def test_detect_self_loop(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a")]
        dep_map = {"a": ["a"]}
        cycles = detector.detect_circular_dependencies(pkgs, dep_map)
        assert len(cycles) >= 1

    # -- _build_dependency_map --

    def test_build_dependency_map(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = detector._build_dependency_map(pkgs)
        assert "a" in dep_map
        assert "b" in dep_map
        assert dep_map["a"] == []
        assert dep_map["b"] == []

    # -- _deduplicate_cycles --

    def test_deduplicate_cycles_identical(self):
        detector = CircularDependencyDetector()
        cycles = [["a", "b", "a"], ["a", "b", "a"]]
        result = detector._deduplicate_cycles(cycles)
        assert len(result) == 1

    def test_deduplicate_cycles_rotated(self):
        detector = CircularDependencyDetector()
        # Same cycle, different starting point
        cycles = [["a", "b", "c", "a"], ["b", "c", "a", "b"]]
        result = detector._deduplicate_cycles(cycles)
        assert len(result) == 1

    def test_deduplicate_cycles_empty(self):
        detector = CircularDependencyDetector()
        assert detector._deduplicate_cycles([]) == []
        assert detector._deduplicate_cycles([[]]) == []

    def test_deduplicate_cycles_different(self):
        detector = CircularDependencyDetector()
        cycles = [["a", "b", "a"], ["c", "d", "c"]]
        result = detector._deduplicate_cycles(cycles)
        assert len(result) == 2

    # -- analyze_circular_dependency_impact --

    def test_analyze_impact_no_cycles(self):
        detector = CircularDependencyDetector()
        result = detector.analyze_circular_dependency_impact([], [])
        assert result["total_cycles"] == 0
        assert result["impact_score"] == 0.0

    def test_analyze_impact_with_cycles(self):
        detector = CircularDependencyDetector()
        cycles = [["a", "b", "a"], ["c", "d", "e", "c"]]
        pkgs = [_pkg(name=n) for n in ["a", "b", "c", "d", "e"]]
        result = detector.analyze_circular_dependency_impact(cycles, pkgs)
        assert result["total_cycles"] == 2
        assert result["packages_involved"] == 5
        assert result["largest_cycle_size"] == 4
        assert result["impact_score"] > 0.0
        assert len(result["cycle_details"]) == 2

    def test_analyze_impact_score_capped(self):
        detector = CircularDependencyDetector()
        # Many large cycles to push impact_score beyond 10
        cycles = [[f"p{i}" for i in range(10)] + ["p0"] for _ in range(10)]
        pkgs = [_pkg(name=f"p{i}") for i in range(10)]
        result = detector.analyze_circular_dependency_impact(cycles, pkgs)
        assert result["impact_score"] <= 10.0

    # -- get_breaking_recommendations --

    def test_get_breaking_recommendations_empty(self):
        detector = CircularDependencyDetector()
        assert detector.get_breaking_recommendations([]) == []

    def test_get_breaking_recommendations_two_node(self):
        detector = CircularDependencyDetector()
        cycles = [["a", "b", "a"]]
        recs = detector.get_breaking_recommendations(cycles)
        assert len(recs) == 1
        assert recs[0]["cycle_size"] == 3
        assert recs[0]["priority"] == "high"
        assert len(recs[0]["breaking_points"]) >= 1

    def test_get_breaking_recommendations_three_plus(self):
        detector = CircularDependencyDetector()
        cycles = [["a", "b", "c", "d", "a"]]
        recs = detector.get_breaking_recommendations(cycles)
        assert len(recs) == 1
        # Should have direct removal strategies + abstraction layer
        bp = recs[0]["breaking_points"]
        strategies = [b["strategy"] for b in bp]
        assert "Remove direct dependency" in strategies
        assert "Introduce abstraction layer" in strategies

    def test_get_breaking_recommendations_single_node(self):
        detector = CircularDependencyDetector()
        cycles = [["a"]]
        recs = detector.get_breaking_recommendations(cycles)
        assert recs == []

    def test_get_breaking_recommendations_priority(self):
        detector = CircularDependencyDetector()
        small = ["a", "b", "c", "a"]  # size 4 -> medium? no, <= 3 check is on len(cycle)
        big = ["a", "b", "c", "d", "e", "a"]  # size 6
        recs = detector.get_breaking_recommendations([small, big])
        # small has len 4 > 3 → medium, big has len 6 > 3 → medium
        for r in recs:
            if r["cycle_size"] <= 3:
                assert r["priority"] == "high"
            else:
                assert r["priority"] == "medium"

    # -- validate_acyclic_graph --

    def test_validate_acyclic_true(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b"), _pkg(name="c")]
        dep_map = {"a": ["b"], "b": ["c"], "c": []}
        assert detector.validate_acyclic_graph(pkgs, dep_map) is True

    def test_validate_acyclic_false(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = {"a": ["b"], "b": ["a"]}
        assert detector.validate_acyclic_graph(pkgs, dep_map) is False

    def test_validate_acyclic_empty(self):
        detector = CircularDependencyDetector()
        assert detector.validate_acyclic_graph([], {}) is True

    def test_validate_acyclic_isolated_nodes(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = {"a": [], "b": []}
        assert detector.validate_acyclic_graph(pkgs, dep_map) is True

    # -- find_strongly_connected_components --

    def test_find_scc_no_cycles(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = {"a": ["b"], "b": []}
        sccs = detector.find_strongly_connected_components(pkgs, dep_map)
        assert sccs == []  # No SCC with >1 node

    def test_find_scc_simple_cycle(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b")]
        dep_map = {"a": ["b"], "b": ["a"]}
        sccs = detector.find_strongly_connected_components(pkgs, dep_map)
        assert len(sccs) == 1
        assert set(sccs[0]) == {"a", "b"}

    def test_find_scc_multiple(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name=n) for n in ["a", "b", "c", "d"]]
        dep_map = {
            "a": ["b"],
            "b": ["a"],
            "c": ["d"],
            "d": ["c"],
        }
        sccs = detector.find_strongly_connected_components(pkgs, dep_map)
        assert len(sccs) == 2

    def test_find_scc_three_node(self):
        detector = CircularDependencyDetector()
        pkgs = [_pkg(name="a"), _pkg(name="b"), _pkg(name="c")]
        dep_map = {"a": ["b"], "b": ["c"], "c": ["a"]}
        sccs = detector.find_strongly_connected_components(pkgs, dep_map)
        assert len(sccs) == 1
        assert set(sccs[0]) == {"a", "b", "c"}

    def test_find_scc_empty(self):
        detector = CircularDependencyDetector()
        sccs = detector.find_strongly_connected_components([], {})
        assert sccs == []
