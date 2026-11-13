"""Regression tests for the OSV dependency scan reporting a fabricated clean bill of health.

Every test here was written against a reproduction on the live OSV API and fails
on the pre-fix code. The defect class is a security report that answers
"0 vulnerabilities, risk 0.0" for a codebase that has real, published CVEs --
the single most dangerous kind of fabricated result in this service, because
"clean" is the answer a consumer acts on.

Four independent mechanisms produced that same false-clean answer:

1. Loaders emit lowercase ``"pypi"``; OSV only accepts ``"PyPI"`` and rejects the
   *whole batch* with HTTP 400 on one bad ecosystem.
2. A non-200 response returned a bare ``[]``, indistinguishable from clean.
3. ``severity[].score`` is a CVSS *vector string*, never a number, so
   ``float(score)`` raised and the advisory was dropped entirely.
4. ``/v1/querybatch`` answers with ``{"id", "modified"}`` stubs only, so parsed
   findings carried severity ``UNKNOWN``, which the risk score weighed as 0.0.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from models.architecture_models import ExternalPackage, Vulnerability
from processing.architecture.dependency_analysis import DependencyAnalyzer
from processing.architecture.dependency_visualization import (
    DependencyVulnerabilityScanner,
    cvss_v3_vector_to_severity,
    normalize_osv_ecosystem,
)


def _pkg(name: str = "jinja2", ecosystem: str = "pypi", version: str = "3.1.2") -> ExternalPackage:
    return ExternalPackage(name=name, version=version, license="BSD", ecosystem=ecosystem)


def _vuln(vuln_id: str, severity: str) -> Vulnerability:
    return Vulnerability(
        id=vuln_id,
        severity=severity,
        summary="",
        details="",
        affected_versions=[],
        fixed_versions=[],
        references=[],
        published_date="",
        modified_date="",
    )


# --- 1. ecosystem normalisation -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pypi", "PyPI"),  # what every Python loader actually emits
        ("PyPI", "PyPI"),
        ("npm", "npm"),
        ("crates.io", "crates.io"),
        ("go", "Go"),
        ("system-apt", None),  # OSV cannot answer; must not be called "clean"
        ("git", None),
        (None, None),
    ],
)
def test_ecosystem_is_normalised_to_osv_spelling(raw, expected):
    assert normalize_osv_ecosystem(raw) == expected


@pytest.mark.asyncio
async def test_unqueryable_ecosystems_are_reported_unscanned_not_clean():
    """A package OSV cannot be asked about must never count as scanned."""
    scanner = DependencyVulnerabilityScanner()
    result = await scanner.scan_vulnerabilities([_pkg("libssl", ecosystem="system-apt")])

    assert result.vulnerabilities == []
    assert result.packages_attempted == 1
    assert result.packages_scanned == 0
    assert result.inconclusive is True
    assert result.errors, "an unscannable package must record why it was not scanned"


@pytest.mark.asyncio
async def test_unqueryable_package_does_not_poison_the_batch():
    """One bad ecosystem used to 400 the whole request, losing every good package."""
    scanner = DependencyVulnerabilityScanner()
    captured: list[list[ExternalPackage]] = []

    async def _fake(packages, result):
        captured.append(list(packages))
        result.packages_scanned += len(packages)

    scanner._query_osv_batched = AsyncMock(side_effect=_fake)
    await scanner.scan_vulnerabilities([_pkg("jinja2"), _pkg("libssl", ecosystem="system-apt")])

    assert captured, "the queryable package must still be sent"
    assert [p.name for p in captured[0]] == ["jinja2"]


# --- 2. outage is not "clean" ---------------------------------------------


@pytest.mark.asyncio
async def test_scan_result_distinguishes_outage_from_genuinely_clean():
    scanner = DependencyVulnerabilityScanner()

    async def _failed(packages, result):
        result.errors.append("OSV API returned status 500")

    scanner._query_osv_batched = AsyncMock(side_effect=_failed)
    outage = await scanner.scan_vulnerabilities([_pkg()])

    async def _clean(packages, result):
        result.packages_scanned += len(packages)

    scanner.vulnerability_cache.clear()
    scanner._query_osv_batched = AsyncMock(side_effect=_clean)
    clean = await scanner.scan_vulnerabilities([_pkg()])

    # Both have an empty vulnerability list -- that is exactly why the old
    # `list[Vulnerability]` return type was a fabrication.
    assert outage.vulnerabilities == clean.vulnerabilities == []
    assert outage.inconclusive is True
    assert clean.inconclusive is False


# --- 3. CVSS vector parsing (advisories were dropped entirely) -------------


@pytest.mark.parametrize(
    ("vector", "band"),
    [
        # Published CVSS v3.1 reference vectors and their scores.
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "CRITICAL"),  # 9.8
        ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", "HIGH"),  # 7.8
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", "MODERATE"),  # 6.1
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N", "MODERATE"),  # 5.4
        ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:L", "LOW"),  # 3.7
        ("CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N", "NONE"),  # 0.0
        ("not-a-vector", None),
        ("", None),
    ],
)
def test_cvss_vector_scores_to_the_published_band(vector, band):
    assert cvss_v3_vector_to_severity(vector) == band


def test_real_osv_record_is_not_dropped_by_its_cvss_vector():
    """OSV always sends a vector string; ``float()`` on it used to drop the advisory."""
    scanner = DependencyVulnerabilityScanner()
    record = {
        "id": "GHSA-h5c8-rqwp-cp95",
        "summary": "Jinja vulnerable to HTML attribute injection",
        "details": "...",
        "modified": "2024-01-11T15:20:48Z",
        "published": "2024-01-11T03:15:11Z",
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N"}],
        "database_specific": {"severity": "MODERATE"},
    }

    parsed = scanner._parse_osv_vulnerability(record)

    assert parsed is not None, "a real OSV advisory must never be silently discarded"
    assert parsed.id == "GHSA-h5c8-rqwp-cp95"
    assert parsed.severity == "MODERATE"


def test_vector_only_advisory_is_graded_from_the_vector():
    """No ``database_specific`` to fall back on: the vector must be scored.

    On the pre-fix code ``float("CVSS:3.1/...")`` raised before any fallback was
    reached, so this advisory was dropped and the package looked clean.
    """
    scanner = DependencyVulnerabilityScanner()
    parsed = scanner._parse_osv_vulnerability(
        {
            "id": "PYSEC-vector-only",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
            ],
        }
    )

    assert parsed is not None
    assert parsed.severity == "CRITICAL"  # 9.8


def test_vector_takes_precedence_over_the_database_band():
    """Proves the band is computed, not read off ``database_specific``."""
    scanner = DependencyVulnerabilityScanner()
    parsed = scanner._parse_osv_vulnerability(
        {
            "id": "GHSA-disagreeing",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
            ],
            "database_specific": {"severity": "LOW"},
        }
    )

    assert parsed is not None
    assert parsed.severity == "CRITICAL"


def test_cvss_v4_only_record_falls_back_to_the_database_band():
    scanner = DependencyVulnerabilityScanner()
    parsed = scanner._parse_osv_vulnerability(
        {
            "id": "GHSA-cpwx-vrp4-4pq7",
            "severity": [{"type": "CVSS_V4", "score": "CVSS:4.0/AV:L/AC:L/AT:P/PR:L"}],
            "database_specific": {"severity": "MODERATE"},
        }
    )
    assert parsed is not None
    assert parsed.severity == "MODERATE"


def test_unparseable_advisory_is_reported_not_dropped():
    scanner = DependencyVulnerabilityScanner()
    parsed = scanner._parse_osv_vulnerability({"id": "GHSA-broken", "affected": "not-a-list"})

    assert parsed is not None, "a known advisory id must survive a parse failure"
    assert parsed.id == "GHSA-broken"
    assert parsed.severity == "UNKNOWN"


# --- 4. risk score must not read UNKNOWN as harmless ----------------------


def test_unknown_severity_does_not_dilute_the_risk_score():
    """One CRITICAL among four UNKNOWNs used to score 2.8 -- i.e. "low risk"."""
    analyzer = DependencyAnalyzer()
    vulns = [_vuln("a", "CRITICAL")] + [_vuln(f"u{i}", "UNKNOWN") for i in range(4)]

    score = analyzer.calculate_risk_score(vulns)

    assert score >= 7.0, f"a CRITICAL finding must not be averaged away (got {score})"


def test_all_unknown_findings_never_score_as_clean():
    analyzer = DependencyAnalyzer()
    score = analyzer.calculate_risk_score([_vuln(f"u{i}", "UNKNOWN") for i in range(3)])

    assert score > 0.0, "findings we could not grade are unknown risk, not zero risk"


def test_no_vulnerabilities_still_scores_zero():
    """Guard the fix against overcorrecting: genuinely clean must stay 0.0."""
    assert DependencyAnalyzer().calculate_risk_score([]) == 0.0


# --- versioned queries -----------------------------------------------------


def test_osv_query_pins_the_installed_version():
    """A version-less OSV query returns every advisory ever filed for the package."""
    scanner = DependencyVulnerabilityScanner()
    query = scanner._osv_query(_pkg(version="3.1.6"))

    assert query["package"]["ecosystem"] == "PyPI"
    assert query["version"] == "3.1.6"


@pytest.mark.parametrize("version", ["*", "unknown", ""])
def test_osv_query_omits_a_meaningless_version(version):
    scanner = DependencyVulnerabilityScanner()
    assert "version" not in scanner._osv_query(_pkg(version=version))
