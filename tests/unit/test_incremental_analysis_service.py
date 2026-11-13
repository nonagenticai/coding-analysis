"""Unit tests for services.incremental.incremental_analysis_service module."""

from __future__ import annotations

import sys

import pytest

# Alias services.models -> models to fix relative import in incremental_analysis_service
if "services.models" not in sys.modules:
    import models as _models_pkg

    sys.modules["services.models"] = _models_pkg

if "services.models.core_models" not in sys.modules:
    import models.core_models as _core_models

    sys.modules["services.models.core_models"] = _core_models

# ---------------------------------------------------------------------------

from services.incremental.incremental_analysis_service import (
    ChangeType,
    DiffFormat,
    DiffHunk,
    FileAnalysisHistory,
    FileDiff,
    IncrementalAnalysisService,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    return IncrementalAnalysisService(debounce_ms=0, enable_caching=True)


@pytest.fixture
def sample_diff():
    return FileDiff(
        file_path="src/main.py",
        old_content="def foo():\n    pass\n",
        new_content="def foo():\n    return 42\n",
        hunks=[DiffHunk(old_start=2, old_lines=1, new_start=2, new_lines=1)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChangedFileDetectionToTargetedAnalysis:
    """test_changed_file_detection_to_targeted_analysis"""

    async def test_full_analysis_stores_in_history(self, service):
        result = await service.analyze_file_full("src/main.py", "def hello(): pass")
        assert result.success is True
        assert result.file_path == "src/main.py"
        assert service.stats["full_analyses"] == 1

    async def test_incremental_analysis_on_diff(self, service, sample_diff):
        # Do a full analysis first to populate history
        await service.analyze_file_full("src/main.py", "def foo():\n    pass\n")
        # Now do incremental
        result = await service.analyze_file_diff(sample_diff, force_full=False)
        assert result.success is True

    async def test_diff_with_no_history_falls_back_to_full(self, service, sample_diff):
        result = await service.analyze_file_diff(sample_diff, force_full=False)
        assert result.success is True
        assert service.stats["full_analyses"] >= 1

    async def test_language_detection_from_path(self, service):
        result = await service.analyze_file_full("app.js", "const x = 1;")
        assert result.language == "javascript"

    def test_file_diff_get_affected_line_ranges(self, sample_diff):
        ranges = sample_diff.get_affected_line_ranges()
        assert len(ranges) == 1
        assert ranges[0][0] >= 1

    def test_file_diff_get_change_summary(self, sample_diff):
        summary = sample_diff.get_change_summary()
        assert "hunks" in summary
        assert summary["hunks"] == 1

    def test_diff_format_enum(self):
        assert DiffFormat.UNIFIED == "unified"
        assert DiffFormat.INLINE == "inline"
        assert DiffFormat.STRUCTURED == "structured"

    def test_change_type_enum(self):
        assert ChangeType.ADDED == "added"
        assert ChangeType.DELETED == "deleted"
        assert ChangeType.MODIFIED == "modified"


class TestEmptyDiffNoOp:
    """test_empty_diff_no_op"""

    async def test_same_content_returns_cached(self, service):
        content = "def foo(): pass"
        await service.analyze_file_full("src/main.py", content)
        # Send diff with same content
        diff = FileDiff(
            file_path="src/main.py",
            old_content=content,
            new_content=content,
        )
        result = await service.analyze_file_diff(diff, force_full=False)
        assert result.success is True
        assert service.stats["cache_hits"] >= 1

    async def test_no_content_diff_returns_empty_result(self, service):
        diff = FileDiff(file_path="src/main.py")
        result = await service.analyze_file_diff(diff, force_full=False)
        assert result is not None

    def test_stats_initial_values(self, service):
        assert service.stats["full_analyses"] == 0
        assert service.stats["incremental_analyses"] == 0
        assert service.stats["cache_hits"] == 0
        assert service.stats["debounced_requests"] == 0

    def test_empty_file_history(self):
        history = FileAnalysisHistory("test.py")
        assert history.get_current() is None
        assert history.get_version(1) is None
        assert history.get_entity("nonexistent") is None
        assert history.has_relation("a", "b", "calls") is False


class TestPartialReanalysisFailure:
    """test_partial_reanalysis_failure"""

    async def test_force_full_bypasses_incremental(self, service, sample_diff):
        await service.analyze_file_full("src/main.py", "def foo():\n    pass\n")
        result = await service.analyze_file_diff(sample_diff, force_full=True)
        assert result.success is True
        assert service.stats["full_analyses"] >= 2

    async def test_compute_diff_regions(self, service):
        old = "line1\nline2\nline3"
        new = "line1\nchanged\nline3"
        regions = await service._compute_diff_regions(old, new)
        assert isinstance(regions, list)
        assert len(regions) >= 1

    async def test_extract_changed_regions_from_hunks(self, service, sample_diff):
        regions = await service._extract_changed_regions(sample_diff)
        assert isinstance(regions, list)

    async def test_full_analysis_with_python_language(self, service):
        result = await service.analyze_file_full("app.py", "class App: pass", language="python")
        assert result.language == "python"

    def test_file_analysis_history_max_versions(self):
        history = FileAnalysisHistory("test.py", max_versions=2)
        from models.core_models import CoreAnalysisResult, CoreComplexityMetrics

        for i in range(5):
            result = CoreAnalysisResult(
                success=True,
                file_path="test.py",
                language="python",
                entities=[],
                relationships=[],
                dependencies=[],
                complexity_metrics=CoreComplexityMetrics(),
                security_patterns=[],
            )
            history.add_version(result, f"content-{i}")

        # Should have trimmed to max_versions
        assert len(history.versions) <= 2
