"""A repo whose manifests could not be read must not score as dependency-free.

`extract_dependencies` returned `[]` both when a repository genuinely declares no
dependencies AND when every loader that found a manifest failed to parse it. The
caller turned that empty list into `risk_score = 0.0` — a clean bill of health for a
dependency set that was never examined. That is the same class as the OSV-outage bug
fixed in #71, one layer higher: #71 made the *scan* honest, this makes the
*extraction* honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.architecture_models import ExternalPackage
from processing.architecture.dependency_analyzer import ArchitectureDependencyAnalyzer
from processing.architecture.dependency_extraction import (
    DependencyExtractionError,
    DependencyExtractor,
)


class _Loader:
    """Stand-in loader: detects or not, parses or raises."""

    def __init__(self, *, detects: bool, raises: Exception | None = None, packages=None):
        self._detects = detects
        self._raises = raises
        self._packages = packages or []

    def detect(self, repo: Path) -> bool:
        return self._detects

    def parse(self, repo: Path):
        if self._raises is not None:
            raise self._raises
        return self._packages


def _extractor(*loaders) -> DependencyExtractor:
    ex = DependencyExtractor()
    ex.loaders = list(loaders)
    return ex


def _pkg() -> ExternalPackage:
    return ExternalPackage(name="jinja2", version="3.1.2", ecosystem="PyPI")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_all_detecting_loaders_failing_raises_rather_than_returning_empty(tmp_path):
    """The case that used to read as 'no dependencies'."""
    ex = _extractor(
        _Loader(detects=True, raises=ValueError("requirements.txt is malformed")),
        _Loader(detects=True, raises=OSError("package.json unreadable")),
    )

    with pytest.raises(DependencyExtractionError) as exc:
        ex.extract_dependencies(tmp_path)

    assert len(exc.value.failures) == 2, "every failing loader must be named"
    assert any("malformed" in f for f in exc.value.failures)


def test_no_loader_detecting_anything_is_legitimately_empty(tmp_path):
    """A genuinely dependency-free repo must still return [] — not an error."""
    ex = _extractor(_Loader(detects=False), _Loader(detects=False))

    assert ex.extract_dependencies(tmp_path) == []


def test_a_partial_failure_that_still_found_packages_does_not_raise(tmp_path):
    """One broken manifest alongside a readable one is not an inconclusive scan."""
    ex = _extractor(
        _Loader(detects=True, raises=ValueError("Cargo.toml is malformed")),
        _Loader(detects=True, packages=[_pkg()]),
    )

    packages = ex.extract_dependencies(tmp_path)

    assert [p.name for p in packages] == ["jinja2"]


# --------------------------------------------------------------------------
# the user-visible consequence
# --------------------------------------------------------------------------


class _RaisingExtractor:
    def extract_dependencies(self, repo: Path):
        raise DependencyExtractionError(repo, ["PipLoader: requirements.txt is malformed"])


class _EmptyExtractor:
    def extract_dependencies(self, repo: Path):
        return []


async def test_unreadable_manifests_abstain_instead_of_scoring_zero(tmp_path):
    """THE bug: an unread dependency set must never be reported as 0.0 risk."""
    analyzer = ArchitectureDependencyAnalyzer()
    analyzer.extractor = _RaisingExtractor()

    report = await analyzer.analyze(tmp_path, "cb-1")

    assert report.inconclusive is True
    assert report.risk_score == -1.0, "0.0 would read as 'no risk found'"
    assert report.scan_errors, "the reason must be carried to the caller, not dropped"


async def test_a_genuinely_dependency_free_repo_still_scores_zero(tmp_path):
    """Control: the honest 0.0 case must NOT be turned into an abstain."""
    analyzer = ArchitectureDependencyAnalyzer()
    analyzer.extractor = _EmptyExtractor()

    report = await analyzer.analyze(tmp_path, "cb-2")

    assert report.inconclusive is False
    assert report.risk_score == 0.0
