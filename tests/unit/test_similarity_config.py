"""Tests for similarity_config.py — covers SimilarityConfig static methods,
model classes, and get_similarity_config function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.similarity_modules.similarity_config import (
    CloneDetectionRequest,
    CrossLanguagePatternRequest,
    SimilarityAnalysisRequest,
    SimilarityConfig,
    get_similarity_config,
)

# ---------------------------------------------------------------------------
# Tests for SimilarityConfig
# ---------------------------------------------------------------------------


def _patch_materialize(monkeypatch, mock) -> None:
    # Patch the globals the method ACTUALLY resolves against: another test module
    # re-imports similarity_config, so importlib could hand back a different copy.
    monkeypatch.setitem(
        SimilarityConfig.resolve_container_files.__globals__, "materialize_tenant_workspace", mock
    )


class TestSimilarityConfig:
    """Tests for SimilarityConfig static methods."""

    def test_validate_analysis_request_no_container_no_files(self):
        req = SimilarityAnalysisRequest()
        with pytest.raises(HTTPException) as exc:
            SimilarityConfig.validate_analysis_request(req)
        assert exc.value.status_code == 400

    def test_validate_analysis_request_with_container(self):
        req = SimilarityAnalysisRequest(container_name="test")
        SimilarityConfig.validate_analysis_request(req)  # No raise

    def test_validate_analysis_request_with_files(self):
        req = SimilarityAnalysisRequest(file_paths=["/a.py"])
        SimilarityConfig.validate_analysis_request(req)  # No raise

    def test_validate_file_paths_existing(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("pass")
        result = SimilarityConfig.validate_file_paths([str(f)])
        assert len(result) == 1

    def test_validate_file_paths_nonexistent(self):
        result = SimilarityConfig.validate_file_paths(["/nonexistent/file.py"])
        assert result == []

    def test_validate_file_paths_mixed(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("pass")
        result = SimilarityConfig.validate_file_paths([str(f), "/nonexistent.py"])
        assert len(result) == 1

    def test_get_file_language_python(self):
        assert SimilarityConfig.get_file_language("test.py") == "python"

    def test_get_file_language_java(self):
        assert SimilarityConfig.get_file_language("App.java") == "java"

    def test_get_file_language_javascript(self):
        assert SimilarityConfig.get_file_language("index.js") == "javascript"

    def test_get_file_language_typescript(self):
        assert SimilarityConfig.get_file_language("app.ts") == "typescript"

    def test_get_file_language_tsx(self):
        assert SimilarityConfig.get_file_language("Component.tsx") == "typescript"

    def test_get_file_language_go(self):
        assert SimilarityConfig.get_file_language("main.go") == "go"

    def test_get_file_language_rust(self):
        assert SimilarityConfig.get_file_language("lib.rs") == "rust"

    def test_get_file_language_csharp(self):
        assert SimilarityConfig.get_file_language("Program.cs") == "csharp"

    def test_get_file_language_cpp(self):
        assert SimilarityConfig.get_file_language("main.cpp") == "cpp"

    def test_get_file_language_c_header(self):
        assert SimilarityConfig.get_file_language("util.h") == "c"

    def test_get_file_language_kotlin(self):
        assert SimilarityConfig.get_file_language("Main.kt") == "kotlin"

    def test_get_file_language_ruby(self):
        assert SimilarityConfig.get_file_language("app.rb") == "ruby"

    def test_get_file_language_unknown(self):
        assert SimilarityConfig.get_file_language("data.csv") == "unknown"

    def test_is_supported_file_py(self):
        assert SimilarityConfig.is_supported_file("test.py") is True

    def test_is_supported_file_txt(self):
        assert SimilarityConfig.is_supported_file("readme.txt") is False

    def test_is_supported_file_java(self):
        assert SimilarityConfig.is_supported_file("App.java") is True

    def test_is_supported_file_rs(self):
        assert SimilarityConfig.is_supported_file("lib.rs") is True

    def test_get_default_thresholds(self):
        thresholds = SimilarityConfig.get_default_thresholds()
        assert "exact_duplicates" in thresholds
        assert thresholds["exact_duplicates"] == 0.95
        assert "code_clones" in thresholds

    def test_get_performance_recommendations_small(self):
        recs = SimilarityConfig.get_performance_recommendations(20)
        assert recs["analysis_time_estimate"] == "< 1 minute"
        assert recs["max_files_suggestion"] is None

    def test_get_performance_recommendations_medium(self):
        recs = SimilarityConfig.get_performance_recommendations(100)
        assert "1-5 minutes" in recs["analysis_time_estimate"]

    def test_get_performance_recommendations_large(self):
        recs = SimilarityConfig.get_performance_recommendations(500)
        assert recs["max_files_suggestion"] == 500

    def test_get_performance_recommendations_very_large(self):
        recs = SimilarityConfig.get_performance_recommendations(2000)
        assert recs["max_files_suggestion"] == 200
        assert recs["recommended_scope"] == "file_level"

    def test_get_capabilities(self):
        caps = SimilarityConfig.get_capabilities()
        assert "similarity_types" in caps
        assert "analysis_scopes" in caps
        assert "supported_languages" in caps
        assert caps["features"]["cross_language_detection"] is True

    async def test_resolve_file_paths_with_container(self):
        with patch.object(
            SimilarityConfig, "resolve_container_files", new_callable=AsyncMock
        ) as mock:
            mock.return_value = ["/a.py"]
            result = await SimilarityConfig.resolve_file_paths("container", None, None)
            assert result == ["/a.py"]

    async def test_resolve_file_paths_without_container(self):
        result = await SimilarityConfig.resolve_file_paths(None, ["/a.py", "/b.py"], None)
        assert result == ["/a.py", "/b.py"]

    async def test_resolve_file_paths_no_container_no_files(self):
        result = await SimilarityConfig.resolve_file_paths(None, None, None)
        assert result == []

    # The container path used to look the name up in a LOCAL registry (404 for
    # every real tenant workspace, prod 2026-10-11: cv-ems-p4) and then walk
    # ``codebase_path`` on THIS pod's disk. The four registry/os.walk tests that
    # pinned that behaviour were replaced by the ones below: resolution is now an
    # exact, tenant-scoped lookup plus a materialised copy of the live workspace.

    async def test_resolve_container_files_not_found_is_404(self, monkeypatch):
        from services import docker_service as ds

        lookup = AsyncMock(return_value=None)
        monkeypatch.setattr(
            ds, "get_docker_service", lambda: MagicMock(find_container_by_name=lookup)
        )
        with pytest.raises(HTTPException) as exc:
            await SimilarityConfig.resolve_container_files(
                "cv-ems", None, None, tenant_id="tenant-a"
            )
        assert exc.value.status_code == 404
        # exact name + caller tenant: 'cv-ems' is never widened to 'cv-ems-p4'
        lookup.assert_awaited_once_with("cv-ems", "tenant-a")

    async def test_resolve_container_files_materialises_tenant_workspace(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "b.py").write_text("pass")
        mat = AsyncMock(return_value=tmp_path)
        _patch_materialize(monkeypatch, mat)

        result = await SimilarityConfig.resolve_container_files(
            "cv-ems-p4", None, None, tenant_id="tenant-a"
        )

        assert sorted(result) == sorted([str(tmp_path / "a.py"), str(tmp_path / "pkg" / "b.py")])
        assert result.materialized_root == tmp_path
        args, kwargs = mat.call_args
        assert args == ("cv-ems-p4", "tenant-a")
        assert "*.py" in kwargs["name_patterns"]

    async def test_resolve_container_files_specific_paths_are_rel_paths(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("pass")
        mat = AsyncMock(return_value=tmp_path)
        _patch_materialize(monkeypatch, mat)

        result = await SimilarityConfig.resolve_container_files(
            "c1", ["src/a.py"], None, tenant_id="t"
        )

        assert list(result) == [str(tmp_path / "src" / "a.py")]
        assert mat.call_args.kwargs["rel_paths"] == ["src/a.py"]

    async def test_resolve_container_files_all_with_limit(self, tmp_path, monkeypatch):
        for name in ("a.py", "b.py", "c.py"):
            (tmp_path / name).write_text("pass")
        _patch_materialize(monkeypatch, AsyncMock(return_value=tmp_path))
        result = await SimilarityConfig.resolve_container_files("c1", None, 2, tenant_id="t")
        assert len(result) == 2

    async def test_release_files_removes_materialised_root(self, tmp_path):
        from app.api.endpoints.similarity_modules.similarity_config import ResolvedFiles

        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.py").write_text("x")
        SimilarityConfig.release_files(ResolvedFiles([str(root / "a.py")], materialized_root=root))
        assert not root.exists()
        SimilarityConfig.release_files(["/plain/list.py"])  # no root: no-op
        SimilarityConfig.release_files(None)


# ---------------------------------------------------------------------------
# Tests for model classes
# ---------------------------------------------------------------------------


class TestSimilarityConfigModels:
    """Tests for model classes."""

    def test_similarity_analysis_request_defaults(self):
        req = SimilarityAnalysisRequest()
        assert req.scope == "function"
        assert req.min_similarity_score == 0.5
        assert req.detect_cross_language is True
        assert req.max_files is None

    def test_clone_detection_request_defaults(self):
        req = CloneDetectionRequest()
        assert req.clone_threshold == 0.85

    def test_cross_language_pattern_request_defaults(self):
        req = CrossLanguagePatternRequest()
        assert req.pattern_threshold == 0.60


# ---------------------------------------------------------------------------
# Tests for get_similarity_config
# ---------------------------------------------------------------------------


class TestGetSimilarityConfig:
    """Tests for get_similarity_config function."""

    def test_returns_config(self):
        config = get_similarity_config()
        assert "supported_extensions" in config
        assert "skip_directories" in config
        assert "default_thresholds" in config
        assert "algorithms" in config
        assert "analysis_types" in config
        assert "performance" in config

    def test_algorithms_have_details(self):
        config = get_similarity_config()
        for algo_name, algo_info in config["algorithms"].items():
            assert "description" in algo_info
            assert "suitable_for" in algo_info
            assert "weight" in algo_info

    def test_analysis_types(self):
        config = get_similarity_config()
        assert "general" in config["analysis_types"]
        assert "refactoring" in config["analysis_types"]
        assert "clone_detection" in config["analysis_types"]
        assert "cross_language" in config["analysis_types"]
