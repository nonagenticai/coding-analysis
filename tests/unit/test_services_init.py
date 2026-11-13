"""Unit tests for services/__init__.py module.

Verifies that the services package correctly re-exports all expected classes.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock


def _load_real_services():
    """Get the real services module, reloading if it was stubbed."""
    mod = sys.modules.get("services")
    if mod is None:
        return importlib.import_module("services")
    # If it's a real module with __all__ defined, it's fine
    if hasattr(mod, "__all__") and isinstance(getattr(mod, "__all__", None), list):
        return mod
    # Stub — force reimport
    if getattr(mod, "__file__", None) or getattr(mod, "__spec__", None):
        return importlib.reload(mod)
    del sys.modules["services"]
    return importlib.import_module("services")


class TestServicesPackageExports:
    """Verify that services/__init__.py exports all expected classes."""

    def test_init_defines_all(self):
        """The module should define __all__ with the right names."""
        mod = _load_real_services()
        assert hasattr(mod, "__all__"), "services/__init__.py must define __all__"
        assert isinstance(mod.__all__, list)

    def test_exports_contain_all_expected_names(self):
        """__all__ should contain exactly the expected 11 service classes."""
        mod = _load_real_services()
        expected = {
            "AnalysisCache",
            "AnalysisStatusService",
            "BugAnalyzer",
            "BugHunter",
            "CodebaseScanner",
            "CodeGraphMetricsService",
            "CodeStructureAnalyzer",
            "ComplexityAnalysisService",
            "DeepAnalysisServiceCodeEntity",
            "QualityAnalysisDelegate",
            "QualityMetricsAnalyzer",
        }
        actual = set(mod.__all__)
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_export_count(self):
        """Should export exactly 11 service classes."""
        mod = _load_real_services()
        assert len(mod.__all__) == 11

    def test_all_exports_are_classes(self):
        """Every name in __all__ should be a class (type), not a MagicMock stub."""
        mod = _load_real_services()
        for name in mod.__all__:
            obj = getattr(mod, name, None)
            assert obj is not None, f"{name} not found on services module"
            assert isinstance(obj, type), f"{name} should be a class, got {type(obj).__name__}"
            assert not isinstance(obj, MagicMock), f"{name} is a MagicMock, not a real class"

    def test_each_export_has_module_attr(self):
        """Each exported class should have a __module__ attribute pointing to a
        real source module (services.* for the legacy services or
        processing.analyzers.* for classes that have since been moved into
        per-domain analyzer packages while still re-exported from
        services/__init__.py)."""
        mod = _load_real_services()
        allowed_prefixes = ("services.", "processing.analyzers.")
        for name in mod.__all__:
            cls = getattr(mod, name, None)
            if cls is None or isinstance(cls, MagicMock):
                continue
            assert hasattr(cls, "__module__"), f"{name} has no __module__"
            assert cls.__module__.startswith(allowed_prefixes), (
                f"{name}.__module__ is {cls.__module__!r}, expected one of {allowed_prefixes}"
            )

    def test_package_has_docstring(self):
        """services/__init__.py should have a module docstring."""
        mod = _load_real_services()
        assert mod.__doc__ is not None, "services/__init__.py should have a docstring"
        assert "Services layer" in mod.__doc__

    def test_no_duplicate_exports(self):
        """__all__ should not contain duplicates."""
        mod = _load_real_services()
        assert len(mod.__all__) == len(set(mod.__all__))

    def test_imports_match_all(self):
        """Every name in __all__ should be accessible as an attribute on the module."""
        mod = _load_real_services()
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} listed in __all__ but not importable"
