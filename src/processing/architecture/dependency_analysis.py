"""
Dependency Analysis and Metrics Module

Handles risk scoring, summary generation, vulnerability categorization,
and overall dependency health metrics.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from models.architecture_models import (
    DependencyRiskReport,
    DependencyRiskSummary,
    ExternalPackage,
    Vulnerability,
)

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """
    Analyzes dependencies and generates risk assessments and metrics.

    Provides comprehensive analysis including risk scoring, vulnerability
    categorization, and dependency health summaries.
    """

    def __init__(self) -> None:
        """Initialize the dependency analyzer."""
        logger.info("DependencyAnalyzer initialized")

    def generate_risk_report(
        self,
        codebase_id: str,
        packages: list[ExternalPackage],
        vulnerabilities: list[Vulnerability],
    ) -> DependencyRiskReport:
        """
        Generate a comprehensive dependency risk report.

        Args:
            codebase_id: Unique identifier for the codebase
            packages: List of all packages in the dependency graph
            vulnerabilities: List of identified vulnerabilities

        Returns:
            Complete dependency risk report

        Raises:
            ValueError: If codebase_id is empty
        """
        if not codebase_id.strip():
            raise ValueError("Codebase ID cannot be empty")

        logger.info(
            "Generating risk report for codebase '%s' with %d packages and %d vulnerabilities",
            codebase_id,
            len(packages),
            len(vulnerabilities),
        )

        # Calculate risk score
        risk_score = self.calculate_risk_score(vulnerabilities)

        # Generate summary
        summary = self.generate_summary(packages, vulnerabilities)

        report = DependencyRiskReport(
            codebase_id=codebase_id,
            generated_at=datetime.now(UTC).isoformat(),
            packages=packages,
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            summary=summary,
        )

        logger.info(
            "Risk report generated: %d packages, %d vulnerabilities, risk score: %.1f",
            len(packages),
            len(vulnerabilities),
            risk_score,
        )

        return report

    def calculate_risk_score(self, vulnerabilities: list[Vulnerability]) -> float:
        """
        Calculate overall risk score based on vulnerabilities.

        Args:
            vulnerabilities: List of vulnerabilities

        Returns:
            Risk score from 0.0 to 10.0
        """
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            "CRITICAL": 10.0,
            "HIGH": 7.0,
            "MODERATE": 4.0,
            "LOW": 2.0,
            "NONE": 0.0,
        }

        # A vulnerability whose severity we could not determine is *unknown*, not
        # harmless. Scoring it 0.0 and averaging it in dragged the mean toward
        # "safe": one CRITICAL alongside four UNKNOWNs used to score 2.8, which
        # reads as a low-risk codebase. Score over what we actually graded, and
        # let the unknowns raise the floor rather than lower the average.
        graded = [v for v in vulnerabilities if v.severity in severity_weights]
        ungraded = len(vulnerabilities) - len(graded)

        if graded:
            average_score = sum(severity_weights[v.severity] for v in graded) / len(graded)
        else:
            # Nothing could be graded, but findings exist. Report a mid-scale
            # score so an all-UNKNOWN result is never mistaken for a clean one.
            average_score = 5.0

        # Apply scaling factor for multiple vulnerabilities
        scaling_factor = min(1.0 + (len(vulnerabilities) - 1) * 0.1, 2.0)

        risk_score = min(10.0, average_score * scaling_factor)

        logger.debug(
            "Calculated risk score: %.2f (%d graded, %d ungraded of %d vulnerabilities)",
            risk_score,
            len(graded),
            ungraded,
            len(vulnerabilities),
        )

        return risk_score

    def generate_summary(
        self,
        packages: list[ExternalPackage],
        vulnerabilities: list[Vulnerability],
    ) -> DependencyRiskSummary:
        """
        Generate summary statistics for the dependency analysis.

        Args:
            packages: List of all packages
            vulnerabilities: List of vulnerabilities

        Returns:
            Dependency risk summary with key metrics
        """
        # Count vulnerabilities by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        for vuln in vulnerabilities:
            if vuln.severity in severity_counts:
                severity_counts[vuln.severity] += 1

        # Count vulnerable packages
        vulnerable_package_count = self._get_vulnerable_packages_count(packages, vulnerabilities)

        summary = DependencyRiskSummary(
            total_packages=len(packages),
            vulnerable_packages=vulnerable_package_count,
            total_vulnerabilities=len(vulnerabilities),
            critical_vulnerabilities=severity_counts["CRITICAL"],
            high_vulnerabilities=severity_counts["HIGH"],
            moderate_vulnerabilities=severity_counts["MODERATE"],
            low_vulnerabilities=severity_counts["LOW"],
        )

        logger.debug(
            "Generated summary: %d total packages, %d vulnerable, %d total vulnerabilities",
            summary.total_packages,
            summary.vulnerable_packages,
            summary.total_vulnerabilities,
        )

        return summary

    def _get_vulnerable_packages_count(
        self,
        packages: list[ExternalPackage],
        vulnerabilities: list[Vulnerability],
    ) -> int:
        """
        Get count of packages that have vulnerabilities.

        Args:
            packages: List of packages to check
            vulnerabilities: List of vulnerabilities to match against

        Returns:
            Number of packages with at least one vulnerability
        """
        vulnerable_packages = set()

        for vuln in vulnerabilities:
            # This is a streamlined approach - in reality we'd need to match
            # packages to vulnerabilities more precisely
            for package in packages:
                if (
                    package.name.lower() in vuln.id.lower()
                    or package.name.lower() in vuln.summary.lower()
                ):
                    vulnerable_packages.add(package.name)

        return len(vulnerable_packages)

    def categorize_vulnerabilities_by_severity(
        self, vulnerabilities: list[Vulnerability]
    ) -> dict[str, int]:
        """
        Categorize vulnerabilities by severity level.

        Args:
            vulnerabilities: List of vulnerabilities to categorize

        Returns:
            Dictionary with counts for each severity level
        """
        categories = {
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "moderate_vulnerabilities": 0,
            "low_vulnerabilities": 0,
        }

        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity == "critical":
                categories["critical_vulnerabilities"] += 1
            elif severity == "high":
                categories["high_vulnerabilities"] += 1
            elif severity == "moderate":
                categories["moderate_vulnerabilities"] += 1
            elif severity == "low":
                categories["low_vulnerabilities"] += 1

        return categories

    def analyze_dependency_health(self, packages: list[ExternalPackage]) -> dict[str, Any]:
        """
        Analyze overall dependency health metrics.

        Args:
            packages: List of packages to analyze

        Returns:
            Dictionary containing health metrics
        """
        total_packages = len(packages)
        if total_packages == 0:
            return {
                "total_packages": 0,
                "outdated_packages": 0,
                "packages_with_vulnerabilities": 0,
                "unlicensed_packages": 0,
                "health_score": 10.0,  # Perfect score for empty set
            }

        # Count various health metrics
        outdated_count = sum(
            1
            for pkg in packages
            if pkg.latest_version and pkg.current_version != pkg.latest_version
        )

        vulnerable_count = sum(
            1
            for pkg in packages
            if pkg.severity and pkg.severity.upper() in ["CRITICAL", "HIGH", "MODERATE"]
        )

        unlicensed_count = sum(
            1 for pkg in packages if not pkg.license or pkg.license.lower() in ["unknown", ""]
        )

        # Calculate health score (0-10, where 10 is perfect health)
        health_score = 10.0
        health_score -= (outdated_count / total_packages) * 3.0  # Up to -3 for outdated
        health_score -= (vulnerable_count / total_packages) * 5.0  # Up to -5 for vulnerable
        health_score -= (unlicensed_count / total_packages) * 2.0  # Up to -2 for unlicensed
        health_score = max(0.0, health_score)

        return {
            "total_packages": total_packages,
            "outdated_packages": outdated_count,
            "packages_with_vulnerabilities": vulnerable_count,
            "unlicensed_packages": unlicensed_count,
            "health_score": round(health_score, 1),
            "outdated_percentage": round((outdated_count / total_packages) * 100, 1),
            "vulnerable_percentage": round((vulnerable_count / total_packages) * 100, 1),
            "unlicensed_percentage": round((unlicensed_count / total_packages) * 100, 1),
        }

    def get_ecosystem_breakdown(self, packages: list[ExternalPackage]) -> dict[str, int]:
        """
        Get breakdown of packages by ecosystem.

        Args:
            packages: List of packages to analyze

        Returns:
            Dictionary mapping ecosystem names to package counts
        """
        ecosystem_counts = {}

        for package in packages:
            ecosystem = package.ecosystem or "unknown"
            ecosystem_counts[ecosystem] = ecosystem_counts.get(ecosystem, 0) + 1

        return ecosystem_counts

    def identify_high_risk_packages(
        self, packages: list[ExternalPackage], vulnerabilities: list[Vulnerability]
    ) -> list[dict[str, Any]]:
        """
        Identify packages that pose the highest risk.

        Args:
            packages: List of packages to analyze
            vulnerabilities: List of vulnerabilities

        Returns:
            List of high-risk package information
        """
        high_risk_packages = []

        # Create a mapping of package names to their vulnerabilities
        package_vulns = {}
        for vuln in vulnerabilities:
            # Simple matching - in reality this would be more sophisticated
            for package in packages:
                if package.name.lower() in vuln.id.lower():
                    if package.name not in package_vulns:
                        package_vulns[package.name] = []
                    package_vulns[package.name].append(vuln)

        # Score packages based on their risk factors
        for package in packages:
            risk_factors = []
            risk_score = 0.0

            # Check for vulnerabilities
            if package.name in package_vulns:
                vulns = package_vulns[package.name]
                critical_vulns = sum(1 for v in vulns if v.severity == "CRITICAL")
                high_vulns = sum(1 for v in vulns if v.severity == "HIGH")

                if critical_vulns > 0:
                    risk_factors.append(f"{critical_vulns} critical vulnerabilities")
                    risk_score += critical_vulns * 3.0

                if high_vulns > 0:
                    risk_factors.append(f"{high_vulns} high vulnerabilities")
                    risk_score += high_vulns * 2.0

            # Check if package is outdated
            if package.latest_version and package.current_version != package.latest_version:
                risk_factors.append("outdated version")
                risk_score += 1.0

            # Check license issues
            if not package.license or package.license.lower() in ["unknown", ""]:
                risk_factors.append("no license information")
                risk_score += 0.5

            # Only include packages with significant risk
            if risk_score >= 2.0:
                high_risk_packages.append(
                    {
                        "name": package.name,
                        "version": package.version or package.current_version,
                        "ecosystem": package.ecosystem,
                        "risk_score": round(risk_score, 1),
                        "risk_factors": risk_factors,
                        "vulnerability_count": len(package_vulns.get(package.name, [])),
                    }
                )

        # Sort by risk score (highest first)
        high_risk_packages.sort(key=lambda x: x["risk_score"], reverse=True)

        return high_risk_packages[:20]  # Return top 20 high-risk packages
