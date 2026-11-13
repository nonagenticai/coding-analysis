"""
Dependency Analysis Module

Main orchestrator for comprehensive dependency analysis including vulnerability
scanning, risk assessment, and security reporting using multiple package managers.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clients.coding_containers import get_coding_containers_client
from models.architecture_models import (
    DependencyRiskReport,
    DependencyRiskSummary,
    ExternalPackage,
    Vulnerability,
)
from processing.architecture.circular_detection import CircularDependencyDetector
from processing.architecture.container_fs import materialize_from_container
from processing.architecture.dependency_analysis import DependencyAnalyzer
from processing.architecture.dependency_extraction import (
    DependencyExtractionError,
    DependencyExtractor,
)
from processing.architecture.dependency_graph import DependencyGraphBuilder
from processing.architecture.dependency_visualization import (
    DependencyVulnerabilityScanner,
)

logger = logging.getLogger(__name__)


class ArchitectureDependencyAnalyzer:
    """
    Main orchestrator for dependency analysis and risk assessment.

    Coordinates between extraction, graph building, vulnerability scanning,
    and analysis modules to provide comprehensive dependency insights.
    """

    def __init__(
        self,
        timeout: int = 30,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24,
    ) -> None:
        """
        Initialize the dependency analyzer orchestrator.

        Args:
            timeout: HTTP timeout for vulnerability scanning in seconds
            enable_cache: Whether to enable vulnerability result caching
            cache_ttl_hours: Time-to-live for cached vulnerability results in hours
        """
        # Initialize component modules
        self.extractor = DependencyExtractor()
        self.graph_builder = DependencyGraphBuilder(timeout=timeout)
        self.analyzer = DependencyAnalyzer()
        self.circular_detector = CircularDependencyDetector()
        self.vulnerability_scanner = DependencyVulnerabilityScanner(
            timeout=timeout,
            enable_cache=enable_cache,
            cache_ttl_hours=cache_ttl_hours,
        )

        logger.info(
            "ArchitectureDependencyAnalyzer initialized with timeout: %ds, cache: %s",
            timeout,
            enable_cache,
        )

    async def analyze(
        self,
        repo: Path,
        codebase_id: str,
        include_transitive: bool = False,
        max_depth: int = 3,
        tenant_id: str | None = None,
    ) -> DependencyRiskReport:
        """
        Analyze dependencies in the repository and assess security risks.

        Args:
            repo: Path to the repository to analyze
            codebase_id: Unique identifier for the codebase
            include_transitive: Whether to include transitive dependencies in analysis
            max_depth: Maximum depth for transitive dependency resolution
            tenant_id: Tenant that OWNS the container the manifests are mirrored
                out of, threaded from the authenticated caller. Without it
                coding-containers resolves this service as its own account and
                its BOLA guard 404s the workspace.

        Returns:
            Comprehensive dependency risk report

        Raises:
            ValueError: If repo path is invalid or codebase_id is empty
            RuntimeError: If dependency analysis fails
        """
        if not codebase_id.strip():
            raise ValueError("Codebase ID cannot be empty")

        logger.info(
            "Starting dependency analysis for codebase '%s' (container: %s, path: %s)",
            codebase_id,
            codebase_id,
            repo,
        )

        # The repo lives inside the analysis container at `repo` (e.g. /workspace),
        # NOT on the analyzer pod's local filesystem. Mirror the dependency
        # manifests out of the container into a temp dir, then run the (sync,
        # Path-based) loaders against real local files — the same container-exec
        # model the quality tools use. Fall back to a local `repo` path if one
        # happens to exist (local-fs callers / tests).
        materialized = await self._materialize_repo_from_container(
            codebase_id, str(repo), tenant_id=tenant_id
        )
        if materialized is not None:
            effective_repo = materialized
        elif repo.exists():
            effective_repo = repo
        else:
            logger.warning(
                "No dependency manifests found in container '%s' at %s",
                codebase_id,
                repo,
            )
            return self._create_empty_report(codebase_id)

        try:
            # Step 1: Extract dependencies.
            # A repo that genuinely declares no dependencies IS a 0.0-risk answer.
            # A repo whose manifests all failed to parse is NOT — nothing was read,
            # so there is no risk score to report. The extractor now separates the
            # two and this path abstains rather than inventing a clean result.
            try:
                packages = self.extractor.extract_dependencies(effective_repo)
            except DependencyExtractionError as e:
                logger.warning(
                    "Dependency extraction for '%s' was inconclusive: %s", codebase_id, e
                )
                return self._create_inconclusive_report(codebase_id, e.failures)

            if not packages:
                logger.warning("No dependencies found in repository at %s", effective_repo)
                return self._create_empty_report(codebase_id)

            # Step 2: Optionally build dependency graph with transitive dependencies
            if include_transitive:
                logger.info(
                    "Building dependency graph with transitive dependencies (max depth: %d)",
                    max_depth,
                )
                packages = await self.graph_builder.build_dependency_graph(packages, max_depth)

            # Step 3: Scan for vulnerabilities
            scan = await self.vulnerability_scanner.scan_vulnerabilities(packages)
            logger.info(
                "Found %d vulnerabilities across %d of %d packages",
                len(scan.vulnerabilities),
                scan.packages_scanned,
                scan.packages_attempted,
            )

            # Step 4: Generate risk report
            report = self.analyzer.generate_risk_report(codebase_id, packages, scan.vulnerabilities)

            # Step 5: Carry the scan provenance onto the report. Without this an
            # OSV outage and a clean codebase are the same 0.0 risk score.
            report.packages_attempted = scan.packages_attempted
            report.packages_scanned = scan.packages_scanned
            report.scan_errors = list(scan.errors)
            report.inconclusive = scan.inconclusive
            report.partial = scan.partial
            if report.inconclusive:
                # Honest abstain, mirroring QualityReport's overall_score = -1.0:
                # nothing was checked, so there is no risk score to report. 0.0
                # here would read as "no risk found".
                report.risk_score = -1.0
                logger.warning(
                    "Dependency scan for '%s' was inconclusive: 0 of %d packages checked (%s)",
                    codebase_id,
                    scan.packages_attempted,
                    "; ".join(scan.errors) or "no reason recorded",
                )

            logger.info(
                "Dependency analysis completed. Found %d packages, %d vulnerabilities, "
                "risk score: %.1f (inconclusive=%s, partial=%s)",
                len(packages),
                len(scan.vulnerabilities),
                report.risk_score,
                report.inconclusive,
                report.partial,
            )

            return report

        except Exception as e:
            logger.error(
                "Dependency analysis failed for codebase '%s': %s",
                codebase_id,
                e,
                exc_info=True,
            )
            raise RuntimeError(f"Dependency analysis failed: {e}") from e
        finally:
            if materialized is not None:
                shutil.rmtree(materialized, ignore_errors=True)

    async def count_declared_packages(
        self,
        repo: Path,
        codebase_id: str,
        tenant_id: str | None = None,
    ) -> int:
        """Count the packages the repo's manifests declare. No network, no scan.

        This is ``analyze()`` minus the vulnerability scan: materialize the
        manifests out of the tenant's container, parse them, count. The
        architecture pipeline only ever reported ``len(packages)`` from the
        dependency stage, yet it ran the full per-package ``api.osv.dev`` scan to
        get it — 96 s of the 130 s request on a 37-file repo (2026-10-12, egress
        to osv.dev reset, 125 packages retried one by one), which is what pushed
        the architecture ops past the 120 s edge timeout.

        Returns 0 when there are no manifests, none declare anything, or they
        could not be parsed — the same count ``analyze()`` yields in those cases.
        """
        if not codebase_id.strip():
            raise ValueError("Codebase ID cannot be empty")

        materialized = await self._materialize_repo_from_container(
            codebase_id, str(repo), tenant_id=tenant_id
        )
        if materialized is not None:
            effective_repo = materialized
        elif repo.exists():
            effective_repo = repo
        else:
            return 0

        try:
            return len(self.extractor.extract_dependencies(effective_repo))
        except DependencyExtractionError as e:
            logger.warning("Dependency extraction for '%s' was inconclusive: %s", codebase_id, e)
            return 0
        finally:
            if materialized is not None:
                shutil.rmtree(materialized, ignore_errors=True)

    # Manifest files every dependency loader looks for (union of each loader's
    # detect()/parse() targets under dependency_plugins/). Keep in sync with the
    # loaders so the container `find` discovers everything they can parse.
    _MANIFEST_NAMES: tuple[str, ...] = (
        "requirements*.txt",
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.toml",
        "Cargo.lock",
        "setup.py",
    )

    async def _materialize_repo_from_container(
        self,
        container_name: str,
        repo_root: str = "/workspace",
        tenant_id: str | None = None,
    ) -> Path | None:
        """Copy dependency manifests out of the analysis container into a local
        temp dir, preserving relative paths, so the sync Path-based loaders can
        read them.

        Returns the temp dir (caller must clean it up), or None if no manifests
        were found / fetched.
        """
        # Resolve the client here (keeps this module's symbol patchable in tests)
        # and delegate the find/copy mechanics to the shared helper. Pin the
        # historical manifest-only excludes so behaviour is unchanged.
        client = get_coding_containers_client()
        return await materialize_from_container(
            client,
            container_name,
            repo_root,
            self._MANIFEST_NAMES,
            ("*/node_modules/*", "*/.git/*", "*/target/*", "*/.venv/*"),
            tmp_prefix="depscan-",
            tenant_id=tenant_id,
        )

    def _create_inconclusive_report(
        self, codebase_id: str, failures: list[str]
    ) -> DependencyRiskReport:
        """Abstain: the manifests could not be read, so nothing was scanned.

        `risk_score = -1.0` + `inconclusive` mirrors the OSV-outage path in `analyze`
        (and QualityReport's `overall_score = -1.0`). Returning 0.0 here would read as
        "no risk found" for a dependency set that was never examined.
        """
        return DependencyRiskReport(
            codebase_id=codebase_id,
            generated_at=datetime.now(UTC).isoformat(),
            packages=[],
            vulnerabilities=[],
            risk_score=-1.0,
            inconclusive=True,
            scan_errors=list(failures),
            summary=DependencyRiskSummary(
                total_packages=0,
                vulnerable_packages=0,
                total_vulnerabilities=0,
                critical_vulnerabilities=0,
                high_vulnerabilities=0,
                moderate_vulnerabilities=0,
                low_vulnerabilities=0,
            ),
        )

    def _create_empty_report(self, codebase_id: str) -> DependencyRiskReport:
        """Create an empty report when no dependencies are found.

        Genuinely dependency-free: 0.0 is the honest answer here. The
        could-not-read case goes to `_create_inconclusive_report` instead.
        """
        return DependencyRiskReport(
            codebase_id=codebase_id,
            generated_at=datetime.now(UTC).isoformat(),
            packages=[],
            vulnerabilities=[],
            risk_score=0.0,
            summary=DependencyRiskSummary(
                total_packages=0,
                vulnerable_packages=0,
                total_vulnerabilities=0,
                critical_vulnerabilities=0,
                high_vulnerabilities=0,
                moderate_vulnerabilities=0,
                low_vulnerabilities=0,
            ),
        )

    # Convenience methods that delegate to component modules

    def get_outdated_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """Get packages that have newer versions available."""
        return self.extractor.get_outdated_packages(packages)

    def get_vulnerable_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """Get packages with known vulnerabilities."""
        return self.extractor.get_vulnerable_packages(packages)

    async def analyze_transitive_dependencies(
        self, packages: list[ExternalPackage], max_depth: int = 3
    ) -> list[ExternalPackage]:
        """Recursively resolve and analyze transitive dependencies."""
        return await self.graph_builder.analyze_transitive_dependencies(packages, max_depth)

    def detect_circular_dependencies(
        self,
        packages: list[ExternalPackage],
        dependency_map: dict[str, list[str]] | None = None,
    ) -> list[list[str]]:
        """Detect circular dependencies in the package graph."""
        return self.circular_detector.detect_circular_dependencies(packages, dependency_map or {})

    def analyze_dependency_health(self, packages: list[ExternalPackage]) -> dict[str, Any]:
        """Analyze overall dependency health metrics."""
        return self.analyzer.analyze_dependency_health(packages)

    def get_ecosystem_breakdown(self, packages: list[ExternalPackage]) -> dict[str, int]:
        """Get breakdown of packages by ecosystem."""
        return self.analyzer.get_ecosystem_breakdown(packages)

    def identify_high_risk_packages(
        self, packages: list[ExternalPackage], vulnerabilities: list[Vulnerability]
    ) -> list[dict[str, Any]]:
        """Identify packages that pose the highest risk."""
        return self.analyzer.identify_high_risk_packages(packages, vulnerabilities)

    def clear_vulnerability_cache(self) -> None:
        """Clear the vulnerability cache."""
        self.vulnerability_scanner.clear_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get vulnerability cache statistics."""
        return self.vulnerability_scanner.get_cache_stats()

    # Legacy method aliases for backward compatibility
    def _deduplicate_packages(self, packages: list[ExternalPackage]) -> list[ExternalPackage]:
        """Legacy method - use extractor.deduplicate_packages instead."""
        return self.extractor.deduplicate_packages(packages)

    def _calculate_risk_score(self, vulnerabilities: list[Vulnerability]) -> float:
        """Legacy method - use analyzer.calculate_risk_score instead."""
        return self.analyzer.calculate_risk_score(vulnerabilities)

    def _generate_summary(
        self, packages: list[ExternalPackage], vulnerabilities: list[Vulnerability]
    ) -> DependencyRiskSummary:
        """Legacy method - use analyzer.generate_summary instead."""
        return self.analyzer.generate_summary(packages, vulnerabilities)

    async def _query_osv_vulnerabilities(
        self, packages: list[ExternalPackage]
    ) -> list[Vulnerability]:
        """Legacy method - use vulnerability_scanner.scan_vulnerabilities instead.

        Note this drops the scan provenance; callers that need to tell an outage
        from a clean result must use ``scan_vulnerabilities`` directly.
        """
        return (await self.vulnerability_scanner.scan_vulnerabilities(packages)).vulnerabilities
