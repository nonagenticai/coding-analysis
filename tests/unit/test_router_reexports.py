"""Tests for router re-export modules.

These modules (analysis.py, similarity.py, similarity_modules/__init__.py, repo_scanner.py)
are thin wrappers that aggregate sub-module routers. Testing them ensures imports work
and the public API surface is correct.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi import APIRouter


def _force_real_import(dotted_name: str):
    """Force-load the real module from disk, cleaning any stubs.

    Many test modules inject bare ``types.ModuleType`` stubs into sys.modules.
    We detect stubs by checking for __spec__ (real modules always have it).
    If a stub is found, we remove it (and children) and re-import.
    """
    mod = sys.modules.get(dotted_name)
    if mod is not None:
        # Real module check: has __spec__ from importlib
        if getattr(mod, "__spec__", None) is not None:
            # Verify it actually has meaningful content
            if hasattr(mod, "router") or hasattr(mod, "__all__") or hasattr(mod, "__version__"):
                return mod

    # Clean the module and all children from sys.modules
    to_delete = sorted(
        (k for k in sys.modules if k == dotted_name or k.startswith(dotted_name + ".")),
        key=len,
        reverse=True,  # delete children first
    )
    for key in to_delete:
        del sys.modules[key]

    return importlib.import_module(dotted_name)


@pytest.fixture(autouse=True)
def _ensure_real_modules():
    """Ensure real modules are available by cleaning any stubs from
    other test modules' sys.modules surgery before each test.
    """
    # List of module names that may have been stubbed by other test files.
    # We clean and reimport them in dependency order.
    modules_to_check = [
        "services.similarity_detector",
        "processing.analyzers.deep",
        "services.github_provisioning_service",
        "services.code_graph_metrics",
        "services.analysis_status",
        "app.api.endpoints.similarity_modules",
        "app.api.endpoints.similarity",
        "app.api.endpoints.analysis_modules",
        "app.api.endpoints.analysis",
        "app.api.endpoints.repo_scanner_modules",
        "app.api.endpoints.repo_scanner",
    ]
    for mod_name in modules_to_check:
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "__spec__", None) is None:
            # It's a stub. Clean it and children.
            to_delete = [k for k in sys.modules if k == mod_name or k.startswith(mod_name + ".")]
            for key in to_delete:
                del sys.modules[key]

    # Now import the modules we need fresh
    for mod_name in modules_to_check:
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
            except Exception:
                pass  # Some modules may have optional deps

    yield


# ===================================================================
# analysis.py
# ===================================================================


class TestAnalysisRouterModule:
    """Tests for the top-level analysis.py router aggregator."""

    def test_analysis_module_has_router(self):
        from app.api.endpoints.analysis import router

        assert isinstance(router, APIRouter)

    def test_analysis_router_includes_sub_routers(self):
        from app.api.endpoints.analysis import router

        assert len(router.routes) > 0

    def test_analysis_module_exports_router_in_all(self):
        import app.api.endpoints.analysis as mod

        assert "router" in mod.__all__

    def test_analysis_module_has_logger(self):
        import app.api.endpoints.analysis as mod

        assert hasattr(mod, "logger")


# ===================================================================
# similarity.py
# ===================================================================


class TestSimilarityRouterModule:
    """Tests for the top-level similarity.py router aggregator."""

    def test_similarity_module_has_router(self):
        from app.api.endpoints.similarity import router

        assert isinstance(router, APIRouter)

    def test_similarity_router_includes_sub_routers(self):
        from app.api.endpoints.similarity import router

        assert len(router.routes) > 0

    def test_similarity_module_exports_router_in_all(self):
        import app.api.endpoints.similarity as mod

        assert "router" in mod.__all__

    def test_similarity_module_exports_classes(self):
        import app.api.endpoints.similarity as mod

        for name in [
            "SimilarityConfig",
            "SimilarityProcessor",
            "SimilarityMetrics",
            "SimilarityScoring",
            "SimilarityThresholds",
            "CodeFeatureExtractor",
            "SimilarityAlgorithms",
        ]:
            assert name in mod.__all__, f"{name} not in __all__"

    def test_similarity_module_has_logger(self):
        import app.api.endpoints.similarity as mod

        assert hasattr(mod, "logger")


# ===================================================================
# similarity_modules/__init__.py
# ===================================================================


class TestSimilarityModulesInit:
    """Tests for the similarity_modules package __init__.py."""

    def test_init_exports_router(self):
        from app.api.endpoints.similarity_modules import router

        assert isinstance(router, APIRouter)

    def test_init_exports_similarity_router_alias(self):
        from app.api.endpoints.similarity_modules import similarity_router

        assert isinstance(similarity_router, APIRouter)

    def test_init_exports_algorithm_functions(self):
        from app.api.endpoints.similarity_modules import (
            calculate_similarity,
            detect_code_clones,
            find_similar_functions,
        )

        assert callable(calculate_similarity)
        assert callable(detect_code_clones)
        assert callable(find_similar_functions)

    def test_init_exports_metrics(self):
        from app.api.endpoints.similarity_modules import (
            SimilarityMetric,
            calculate_metrics,
            get_metric_weights,
        )

        assert SimilarityMetric is not None
        assert callable(calculate_metrics)
        assert callable(get_metric_weights)

    def test_init_exports_processing(self):
        from app.api.endpoints.similarity_modules import (
            SimilarityProcessor,
            batch_process_similarities,
            process_similarity_request,
        )

        assert SimilarityProcessor is not None
        assert callable(batch_process_similarities)
        assert callable(process_similarity_request)

    def test_init_exports_config(self):
        from app.api.endpoints.similarity_modules import (
            SimilarityConfig,
            get_similarity_config,
        )

        assert SimilarityConfig is not None
        assert callable(get_similarity_config)

    def test_all_list_is_comprehensive(self):
        import app.api.endpoints.similarity_modules as mod

        assert "router" in mod.__all__
        assert "SimilarityConfig" in mod.__all__
        assert "SimilarityProcessor" in mod.__all__
        assert "CodeFeatureExtractor" in mod.__all__
        assert "SimilarityAlgorithms" in mod.__all__


# ===================================================================
# repo_scanner.py
# ===================================================================


class TestRepoScannerModule:
    """Tests for the top-level repo_scanner.py re-export module."""

    def test_repo_scanner_has_router(self):
        from app.api.endpoints.repo_scanner import router

        assert isinstance(router, APIRouter)

    def test_repo_scanner_exports_router_in_all(self):
        import app.api.endpoints.repo_scanner as mod

        assert "router" in mod.__all__

    def test_repo_scanner_has_version(self):
        import app.api.endpoints.repo_scanner as mod

        assert hasattr(mod, "__version__")
        assert mod.__version__ == "2.1.0"

    def test_repo_scanner_has_metadata(self):
        import app.api.endpoints.repo_scanner as mod

        assert hasattr(mod, "__author__")
        assert hasattr(mod, "__description__")

    def test_repo_scanner_all_includes_function_names(self):
        import app.api.endpoints.repo_scanner as mod

        for name in [
            "scan_dependencies",
            "scan_imports",
            "validate_code",
            "scan_dependencies_v2",
            "validate_code_v2",
            "validate_scanned_imports",
            "execute_in_container",
            "process_scan_results",
            "get_container_manager_v2",
        ]:
            assert name in mod.__all__, f"{name} not in __all__"
