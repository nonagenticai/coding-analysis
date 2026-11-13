"""
Dependency Visualization Utilities Module

Handles OSV vulnerability scanning, caching, batch processing,
and vulnerability data management for dependency analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp: Any = None

from models.architecture_models import ExternalPackage, Vulnerability

logger = logging.getLogger(__name__)

# OSV rejects an unknown ecosystem with HTTP 400 `code = InvalidArgument desc =
# invalid ecosystem` -- and it rejects the WHOLE batch, not the offending entry.
# The loaders emit lowercase "pypi" (9 sites), plus "system-<mgr>" and "git",
# none of which OSV accepts, so any batch carrying one of those returned no
# results at all. Map what we can onto OSV's canonical spelling and drop the
# rest before the request, recording them as unscanned.
_OSV_ECOSYSTEM_ALIASES = {
    "pypi": "PyPI",
    "python": "PyPI",
    "npm": "npm",
    "node": "npm",
    "javascript": "npm",
    "crates.io": "crates.io",
    "crates": "crates.io",
    "rust": "crates.io",
    "cargo": "crates.io",
    "go": "Go",
    "golang": "Go",
    "maven": "Maven",
    "java": "Maven",
    "nuget": "NuGet",
    "packagist": "Packagist",
    "rubygems": "RubyGems",
    "hex": "Hex",
    "pub": "Pub",
}


# --- CVSS v3.1 base score (spec section 7.1) ------------------------------
# OSV reports severity as a vector string, so a band can only be derived by
# actually scoring the vector. Metric weights are from the CVSS v3.1
# specification and are fixed constants, not tuning knobs.
_CVSS_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_CVSS_AC = {"L": 0.77, "H": 0.44}
_CVSS_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_CVSS_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
_CVSS_UI = {"N": 0.85, "R": 0.62}
_CVSS_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}


def _cvss_roundup(value: float) -> float:
    """CVSS v3.1 Roundup: round *up* to one decimal place."""
    scaled = int(round(value * 100_000))
    if scaled % 10_000 == 0:
        return scaled / 100_000.0
    return (math.floor(scaled / 10_000) + 1) / 10.0


def cvss_v3_vector_to_severity(vector: str) -> str | None:
    """Score a CVSS v3.x vector string and map it onto an OSV severity band.

    Returns ``None`` when the vector cannot be parsed, so the caller can fall
    back rather than discard the advisory. Validated against the published
    reference scores for the CVSS v3.1 examples.
    """
    try:
        metrics = dict(part.split(":", 1) for part in vector.strip().split("/") if ":" in part)
        scope_changed = metrics["S"] == "C"
        iss = 1 - (
            (1 - _CVSS_CIA[metrics["C"]])
            * (1 - _CVSS_CIA[metrics["I"]])
            * (1 - _CVSS_CIA[metrics["A"]])
        )
        if scope_changed:
            impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
        else:
            impact = 6.42 * iss

        privileges = (_CVSS_PR_CHANGED if scope_changed else _CVSS_PR_UNCHANGED)[metrics["PR"]]
        exploitability = (
            8.22
            * _CVSS_AV[metrics["AV"]]
            * _CVSS_AC[metrics["AC"]]
            * privileges
            * _CVSS_UI[metrics["UI"]]
        )
    except (KeyError, ValueError, TypeError):
        return None

    if impact <= 0:
        base = 0.0
    else:
        multiplier = 1.08 if scope_changed else 1.0
        base = _cvss_roundup(min(multiplier * (impact + exploitability), 10.0))

    if base >= 9.0:
        return "CRITICAL"
    if base >= 7.0:
        return "HIGH"
    if base >= 4.0:
        return "MODERATE"
    if base > 0.0:
        return "LOW"
    return "NONE"


def normalize_osv_ecosystem(ecosystem: str | None) -> str | None:
    """Map a loader ecosystem onto OSV's canonical spelling.

    Returns ``None`` when OSV cannot be queried for this ecosystem at all
    (``system-apt``, ``git``, ...), so the caller can report those packages as
    unscanned instead of silently reporting them as clean.
    """
    if not ecosystem:
        return None
    return _OSV_ECOSYSTEM_ALIASES.get(ecosystem.strip().lower())


class VulnerabilityScanResult:
    """Vulnerabilities plus the provenance needed to judge how much to trust them.

    ``scan_vulnerabilities`` used to return a bare ``list[Vulnerability]``, which
    made an OSV outage, an unqueryable ecosystem and a genuinely clean codebase
    all look identical (``[]``). Callers now get the counts and the reasons.
    """

    __slots__ = ("errors", "packages_attempted", "packages_scanned", "vulnerabilities")

    def __init__(
        self,
        vulnerabilities: list[Vulnerability] | None = None,
        packages_attempted: int = 0,
        packages_scanned: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        self.vulnerabilities = vulnerabilities if vulnerabilities is not None else []
        self.packages_attempted = packages_attempted
        self.packages_scanned = packages_scanned
        self.errors = errors if errors is not None else []

    @property
    def inconclusive(self) -> bool:
        """True when packages were submitted but none could be checked."""
        return self.packages_attempted > 0 and self.packages_scanned == 0

    @property
    def partial(self) -> bool:
        """True when some, but not all, submitted packages were checked."""
        return 0 < self.packages_scanned < self.packages_attempted


class DependencyVulnerabilityScanner:
    """
    Handles vulnerability scanning using OSV.dev API with caching and batching.

    Provides efficient vulnerability detection with rate limiting,
    caching, and batch processing capabilities.
    """

    def __init__(
        self,
        timeout: int = 30,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24,
        batch_size: int = 10,
        max_concurrent_batches: int = 5,
    ) -> None:
        """
        Initialize the vulnerability scanner.

        Args:
            timeout: HTTP timeout for vulnerability scanning in seconds
            enable_cache: Whether to enable vulnerability result caching
            cache_ttl_hours: Time-to-live for cached vulnerability results in hours
            batch_size: Number of packages to batch together
            max_concurrent_batches: Maximum concurrent batch requests
        """
        self.timeout = timeout
        self.osv_api_url = "https://api.osv.dev/v1/query"
        self.osv_batch_url = "https://api.osv.dev/v1/querybatch"
        # /v1/querybatch answers with id-only stubs; full records come from here.
        self.osv_vuln_url = "https://api.osv.dev/v1/vulns"

        # Cache configuration
        self.enable_cache = enable_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.vulnerability_cache: dict[str, dict[str, Any]] = {}

        # Batching configuration
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

        logger.info(
            "DependencyVulnerabilityScanner initialized (cache: %s, batch_size: %d)",
            enable_cache,
            batch_size,
        )

    async def scan_vulnerabilities(
        self, packages: list[ExternalPackage]
    ) -> VulnerabilityScanResult:
        """
        Scan packages for vulnerabilities using OSV.dev API with batching and caching.

        Args:
            packages: List of packages to check for vulnerabilities

        Returns:
            A :class:`VulnerabilityScanResult` carrying the vulnerabilities found
            *and* how many of the submitted packages were actually checked. An
            empty ``vulnerabilities`` list is only a clean result when
            ``packages_scanned == packages_attempted``.
        """
        result = VulnerabilityScanResult(packages_attempted=len(packages))

        if not packages:
            return result

        # Drop packages OSV cannot be asked about at all. Leaving them in poisons
        # the entire batch (OSV rejects the whole request on one bad ecosystem),
        # which is how every Python package silently went unscanned.
        queryable: list[ExternalPackage] = []
        unsupported: dict[str, int] = {}
        for package in packages:
            if normalize_osv_ecosystem(package.ecosystem):
                queryable.append(package)
            else:
                unsupported[package.ecosystem or "<unset>"] = (
                    unsupported.get(package.ecosystem or "<unset>", 0) + 1
                )

        for ecosystem, count in sorted(unsupported.items()):
            message = f"{count} package(s) with ecosystem {ecosystem!r} are not queryable via OSV"
            logger.warning("%s - reported as unscanned, not as clean", message)
            result.errors.append(message)

        if not queryable:
            return result

        # Separate cached and uncached packages
        cached_packages, uncached_packages = self._separate_cached_packages(queryable)

        # Get vulnerabilities from cache
        for package in cached_packages:
            cache_key = self._get_cache_key(package)
            cached_vulns = self.vulnerability_cache[cache_key]["data"]
            result.vulnerabilities.extend(cached_vulns)
        result.packages_scanned += len(cached_packages)

        logger.info(
            "Retrieved %d cached vulnerability results, querying %d packages",
            len(cached_packages),
            len(uncached_packages),
        )

        # Query uncached packages in batches
        if uncached_packages:
            await self._query_osv_batched(uncached_packages, result)

        # Remove duplicate vulnerabilities based on ID
        result.vulnerabilities = self._deduplicate_vulnerabilities(result.vulnerabilities)

        logger.info(
            "Found %d vulnerabilities across %d of %d packages (%d error(s))",
            len(result.vulnerabilities),
            result.packages_scanned,
            result.packages_attempted,
            len(result.errors),
        )
        return result

    async def _hydrate_vulns(
        self, session: Any, osv_results: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Fetch full advisory records for the id-only stubs ``querybatch`` returns.

        Returns a mapping of advisory id -> full OSV record. Ids that could not
        be fetched are simply absent, and the caller falls back to the stub --
        the finding is still reported, just with UNKNOWN severity, which is
        honest about what we know rather than silently scoring it 0.0.
        """
        ids = {
            vuln["id"]
            for osv_result in osv_results
            for vuln in osv_result.get("vulns", [])
            if isinstance(vuln, dict) and vuln.get("id")
        }
        if not ids:
            return {}

        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def fetch(vuln_id: str) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                try:
                    async with session.get(f"{self.osv_vuln_url}/{vuln_id}") as response:
                        if response.status == 200:
                            return vuln_id, await response.json()
                        logger.warning(
                            "OSV returned status %d hydrating advisory %s; it will be "
                            "reported with UNKNOWN severity",
                            response.status,
                            vuln_id,
                        )
                except Exception as exc:  # noqa: BLE001 - degrade, never drop
                    logger.warning(
                        "Could not hydrate OSV advisory %s (%s: %s); it will be "
                        "reported with UNKNOWN severity",
                        vuln_id,
                        type(exc).__name__,
                        exc,
                    )
                return vuln_id, None

        fetched = await asyncio.gather(*(fetch(i) for i in sorted(ids)))
        return {vuln_id: data for vuln_id, data in fetched if data is not None}

    def _osv_query(self, package: ExternalPackage) -> dict[str, Any]:
        """Build a single OSV query for a package.

        ``version`` is mandatory for a meaningful answer: OSV interprets a
        version-less query as "every advisory ever filed against this package
        name", so a fully patched dependency came back with every historical CVE
        attached. Verified against api.osv.dev: jinja2 with no version returns 20
        advisories, jinja2 at 3.1.6 returns 0.
        """
        query: dict[str, Any] = {
            "package": {
                "name": package.name,
                "ecosystem": normalize_osv_ecosystem(package.ecosystem),
            }
        }
        if package.version and package.version not in {"*", "unknown", ""}:
            query["version"] = package.version
        return query

    def _separate_cached_packages(
        self, packages: list[ExternalPackage]
    ) -> tuple[list[ExternalPackage], list[ExternalPackage]]:
        """Separate packages into cached and uncached lists."""
        cached: list[ExternalPackage] = []
        uncached: list[ExternalPackage] = []

        if not self.enable_cache:
            return cached, packages

        current_time = datetime.now(UTC)

        for package in packages:
            cache_key = self._get_cache_key(package)

            if cache_key in self.vulnerability_cache:
                cached_data = self.vulnerability_cache[cache_key]
                cache_time = cached_data["timestamp"]

                # Check if cache is still valid
                if current_time - cache_time < self.cache_ttl:
                    cached.append(package)
                else:
                    # Cache expired, remove and query again
                    del self.vulnerability_cache[cache_key]
                    uncached.append(package)
            else:
                uncached.append(package)

        return cached, uncached

    def _get_cache_key(self, package: ExternalPackage) -> str:
        """Generate a cache key for a package."""
        key_data = f"{package.name}:{package.ecosystem}:{package.version}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def _query_osv_batched(
        self, packages: list[ExternalPackage], result: VulnerabilityScanResult
    ) -> None:
        """Query OSV API using batched requests with concurrency control.

        Accumulates into ``result`` so a batch that never completed is counted as
        unscanned rather than silently contributing zero vulnerabilities.
        """
        # Split packages into batches
        batches = [
            packages[i : i + self.batch_size] for i in range(0, len(packages), self.batch_size)
        ]

        # Use semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_batch(
            batch: list[ExternalPackage],
        ) -> VulnerabilityScanResult:
            async with semaphore:
                return await self._query_single_batch(batch)

        # Process all batches concurrently
        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=True,
        )

        # Collect results and handle exceptions
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, BaseException):
                message = f"OSV batch {i} raised {type(batch_result).__name__}: {batch_result}"
                logger.error("%s", message)
                result.errors.append(message)
            else:
                result.vulnerabilities.extend(batch_result.vulnerabilities)
                result.packages_scanned += batch_result.packages_scanned
                result.errors.extend(batch_result.errors)

    async def _query_single_batch(self, packages: list[ExternalPackage]) -> VulnerabilityScanResult:
        """Query a single batch of packages."""
        result = VulnerabilityScanResult(packages_attempted=len(packages))

        if not HAS_AIOHTTP:
            message = "aiohttp not available - cannot query OSV database"
            logger.warning("%s", message)
            result.errors.append(message)
            return result

        try:
            # Prepare batch payload
            batch_payload = {"queries": [self._osv_query(pkg) for pkg in packages]}

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(self.osv_batch_url, json=batch_payload) as response:
                    if response.status == 200:
                        # Process each result and cache it
                        response_data = await response.json()
                        osv_results = response_data.get("results", [])

                        # /v1/querybatch answers with `{"id", "modified"}` stubs
                        # only -- no severity, summary, details or references
                        # (verified against api.osv.dev). Parsing those stubs
                        # directly yields severity="UNKNOWN" for every finding,
                        # which the risk score then reads as 0.0. Hydrate the
                        # full advisories before parsing.
                        hydrated = await self._hydrate_vulns(session, osv_results)

                        for i, osv_result in enumerate(osv_results):
                            if i >= len(packages):
                                continue

                            package_vulnerabilities = []
                            for stub in osv_result.get("vulns", []):
                                vuln_data = hydrated.get(stub.get("id"), stub)
                                if vulnerability := self._parse_osv_vulnerability(vuln_data):
                                    package_vulnerabilities.append(vulnerability)
                                    result.vulnerabilities.append(vulnerability)

                            # Cache the result if package is valid
                            if self.enable_cache:
                                self._cache_package_vulnerabilities(
                                    packages[i], package_vulnerabilities
                                )

                        # Only the packages OSV actually answered for count as
                        # scanned. A short results array means the tail was not
                        # checked, and must not be reported as clean.
                        result.packages_scanned = min(len(osv_results), len(packages))
                        if result.packages_scanned < len(packages):
                            message = (
                                f"OSV returned {len(osv_results)} result(s) for "
                                f"{len(packages)} queried package(s)"
                            )
                            logger.warning("%s", message)
                            result.errors.append(message)

                    elif response.status == 429:
                        logger.warning("Rate limited by OSV.dev API, implementing backoff")
                        # Longer backoff for batch requests
                        await asyncio.sleep(2)
                        # Retry with individual requests as fallback
                        return await self._query_individual_fallback(packages)
                    else:
                        body = (await response.text())[:200]
                        message = (
                            f"OSV API returned status {response.status} for batch request: {body}"
                        )
                        logger.warning("%s", message)
                        result.errors.append(message)

        except TimeoutError:
            message = "Timeout during batch vulnerability query"
            logger.warning("%s", message)
            result.errors.append(message)
        except Exception as e:
            logger.warning("Error during batch vulnerability query: %s", e)
            # Fallback to individual requests
            fallback = await self._query_individual_fallback(packages)
            fallback.errors.insert(0, f"Batch query failed ({type(e).__name__}: {e})")
            return fallback

        return result

    async def _query_individual_fallback(
        self, packages: list[ExternalPackage]
    ) -> VulnerabilityScanResult:
        """Fallback to individual requests when batch request fails."""
        result = VulnerabilityScanResult(packages_attempted=len(packages))

        if not HAS_AIOHTTP:
            message = "aiohttp not available - cannot query OSV database"
            logger.warning("%s", message)
            result.errors.append(message)
            return result

        logger.info(
            "Falling back to individual requests for %d packages",
            len(packages),
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            for package in packages:
                try:
                    async with session.post(
                        self.osv_api_url, json=self._osv_query(package)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            vulns = data.get("vulns", [])
                            package_vulnerabilities = []

                            for vuln_data in vulns:
                                vulnerability = self._parse_osv_vulnerability(vuln_data)
                                if vulnerability:
                                    package_vulnerabilities.append(vulnerability)
                                    result.vulnerabilities.append(vulnerability)

                            # Cache the result
                            if self.enable_cache:
                                self._cache_package_vulnerabilities(
                                    package, package_vulnerabilities
                                )
                            result.packages_scanned += 1

                        elif response.status == 429:
                            message = f"Rate limited by OSV.dev API, skipped package {package.name}"
                            logger.warning("%s", message)
                            result.errors.append(message)
                            await asyncio.sleep(1)

                        else:
                            message = (
                                f"OSV API returned status {response.status} for {package.name}"
                            )
                            logger.warning("%s", message)
                            result.errors.append(message)

                        # Add small delay between individual requests to be respectful
                        await asyncio.sleep(0.1)

                except Exception as e:
                    message = f"Error querying vulnerabilities for package {package.name}: {e}"
                    logger.warning("%s", message)
                    result.errors.append(message)

        return result

    def _cache_package_vulnerabilities(
        self, package: ExternalPackage, vulnerabilities: list[Vulnerability]
    ) -> None:
        """Cache vulnerability results for a package."""
        if not self.enable_cache:
            return

        cache_key = self._get_cache_key(package)
        self.vulnerability_cache[cache_key] = {
            "data": vulnerabilities,
            "timestamp": datetime.now(UTC),
        }

    def _deduplicate_vulnerabilities(
        self, vulnerabilities: list[Vulnerability]
    ) -> list[Vulnerability]:
        """Remove duplicate vulnerabilities based on ID."""
        seen_vuln_ids = set()
        unique_vulnerabilities = []

        for vuln in vulnerabilities:
            if vuln.id not in seen_vuln_ids:
                seen_vuln_ids.add(vuln.id)
                unique_vulnerabilities.append(vuln)

        return unique_vulnerabilities

    def _parse_osv_vulnerability(self, vuln_data: dict[str, Any]) -> Vulnerability | None:
        """
        Parse OSV vulnerability data into our Vulnerability model.

        Args:
            vuln_data: Raw vulnerability data from OSV.dev API

        Returns:
            Parsed Vulnerability object or None if parsing fails
        """
        try:
            if not (vuln_id := vuln_data.get("id", "")):
                return None

            # Extract version ranges
            affected_versions, fixed_versions = self._extract_version_ranges(vuln_data)

            return Vulnerability(
                id=vuln_id,
                severity=self._extract_severity_from_osv(vuln_data),
                summary=vuln_data.get("summary", ""),
                details=vuln_data.get("details", ""),
                affected_versions=affected_versions,
                fixed_versions=fixed_versions,
                references=[
                    ref.get("url", "") for ref in vuln_data.get("references", []) if ref.get("url")
                ],
                published_date=vuln_data.get("published", ""),
                modified_date=vuln_data.get("modified", ""),
            )

        except Exception as e:  # noqa: BLE001 - degrade to UNKNOWN, never drop or crash
            # Never let a parsing hiccup erase a known advisory: a dropped
            # vulnerability is indistinguishable from a clean package. Emit the
            # identifier we do have, flagged UNKNOWN, so it is still counted and
            # still visible to whoever reads the report.
            logger.warning(
                "Failed to fully parse OSV vulnerability %s (%s: %s); reporting it "
                "with UNKNOWN severity rather than dropping it",
                vuln_data.get("id", "<no id>"),
                type(e).__name__,
                e,
            )
            if vuln_id := vuln_data.get("id", ""):
                return Vulnerability(
                    id=str(vuln_id),
                    severity="UNKNOWN",
                    summary=str(vuln_data.get("summary", "")),
                    details="Advisory could not be fully parsed from the OSV record.",
                    affected_versions=[],
                    fixed_versions=[],
                    references=[],
                    published_date=str(vuln_data.get("published", "")),
                    modified_date=str(vuln_data.get("modified", "")),
                )
            return None

    def _extract_version_ranges(self, vuln_data: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Extract affected and fixed versions from vulnerability data."""
        affected_versions: list[str] = []
        fixed_versions: list[str] = []

        for affected in vuln_data.get("affected", []):
            for range_info in affected.get("ranges", []):
                for event in range_info.get("events", []):
                    if "introduced" in event:
                        affected_versions.append(event["introduced"])
                    elif "fixed" in event:
                        fixed_versions.append(event["fixed"])

        return affected_versions, fixed_versions

    def _extract_severity_from_osv(self, vuln_data: dict[str, Any]) -> str:
        """Extract severity from OSV vulnerability data.

        OSV always reports ``severity[].score`` as a CVSS *vector string*
        (``CVSS:3.1/AV:N/...``), never as a number -- verified against
        api.osv.dev for GHSA and PYSEC records alike. This used to call
        ``float(score)`` on it, which raised ``ValueError`` for every
        severity-carrying advisory; the caller caught it and returned ``None``,
        so **every** vulnerability that had a CVSS severity was silently dropped
        from the report. Parse the vector properly, and never raise: an
        unparseable severity must still yield a counted vulnerability.
        """
        for sev in vuln_data.get("severity") or []:
            if not isinstance(sev, dict):
                continue
            score = sev.get("score")
            if not score:
                continue
            if str(sev.get("type", "")).upper().startswith("CVSS_V3"):
                if (band := cvss_v3_vector_to_severity(str(score))) is not None:
                    return band
            # A bare number is not what OSV emits, but accept it defensively
            # rather than discarding the advisory.
            try:
                return self._cvss_to_severity(float(score))
            except (TypeError, ValueError):
                pass

        # CVSS_V4 vectors use different arithmetic; GitHub-sourced records carry
        # a plain band here, which is a better answer than UNKNOWN.
        db_specific = vuln_data.get("database_specific") or {}
        if severity := db_specific.get("severity"):
            return str(severity).upper()

        # Genuinely no severity data (typical of PYSEC records). UNKNOWN is
        # honest; it is *not* treated as "no risk" by the risk score.
        return "UNKNOWN"

    def _cvss_to_severity(self, score: float) -> str:
        """Convert CVSS score to severity level."""
        if score >= 9.0:
            return "CRITICAL"
        if score >= 7.0:
            return "HIGH"
        if score >= 4.0:
            return "MODERATE"
        return "LOW"

    def clear_cache(self) -> None:
        """Clear the vulnerability cache."""
        self.vulnerability_cache.clear()
        logger.info("Vulnerability cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        current_time = datetime.now(UTC)
        valid_entries = 0
        expired_entries = 0

        for cache_data in self.vulnerability_cache.values():
            cache_time = cache_data["timestamp"]
            if current_time - cache_time < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1

        return {
            "total_entries": len(self.vulnerability_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_hit_potential": f"{(valid_entries / max(1, len(self.vulnerability_cache))) * 100:.1f}%",
        }


class DependencyVisualizer:
    """
    Provides visualization utilities for dependency graphs and analysis results.

    Generates textual representations and data structures suitable for
    various visualization tools and reporting.
    """

    def __init__(self) -> None:
        """Initialize the dependency visualizer."""
        logger.info("DependencyVisualizer initialized")

    def generate_dependency_tree_text(
        self, packages: list[ExternalPackage], max_depth: int = 3
    ) -> str:
        """
        Generate a text-based tree representation of dependencies.

        Args:
            packages: List of packages to visualize
            max_depth: Maximum depth to display

        Returns:
            Text representation of the dependency tree
        """
        if not packages:
            return "No dependencies found."

        tree_lines = ["Dependency Tree:", "=" * 50]

        for package in packages[:20]:  # Limit to first 20 packages
            tree_lines.append(f"├── {package.name} ({package.version})")
            tree_lines.append(f"│   ├── Ecosystem: {package.ecosystem}")
            tree_lines.append(f"│   ├── License: {package.license or 'Unknown'}")
            if package.severity:
                tree_lines.append(f"│   └── Security: {package.severity}")
            tree_lines.append("│")

        if len(packages) > 20:
            tree_lines.append(f"... and {len(packages) - 20} more packages")

        return "\n".join(tree_lines)

    def generate_vulnerability_summary_text(self, vulnerabilities: list[Vulnerability]) -> str:
        """
        Generate a text summary of vulnerabilities.

        Args:
            vulnerabilities: List of vulnerabilities

        Returns:
            Text summary of vulnerabilities
        """
        if not vulnerabilities:
            return "No vulnerabilities found."

        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        for vuln in vulnerabilities:
            if vuln.severity in severity_counts:
                severity_counts[vuln.severity] += 1

        summary_lines = [
            "Vulnerability Summary:",
            "=" * 50,
            f"Total Vulnerabilities: {len(vulnerabilities)}",
            f"Critical: {severity_counts['CRITICAL']}",
            f"High: {severity_counts['HIGH']}",
            f"Moderate: {severity_counts['MODERATE']}",
            f"Low: {severity_counts['LOW']}",
            "",
            "Top Vulnerabilities:",
            "-" * 30,
        ]

        # Show top 10 vulnerabilities
        for i, vuln in enumerate(vulnerabilities[:10]):
            summary_lines.append(f"{i + 1}. {vuln.id} ({vuln.severity}) - {vuln.summary[:80]}...")

        return "\n".join(summary_lines)
