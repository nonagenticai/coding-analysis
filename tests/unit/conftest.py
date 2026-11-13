"""Unit-test conftest — prevent sys.modules pollution between test modules.

Many test files inject fake ``types.ModuleType`` or ``MagicMock`` stubs into
``sys.modules`` at import time so they can test individual sub-modules in
isolation.  Those stubs must not leak into later test files that need the
*real* modules.

Strategy
--------
Before each test module is collected (imported) and before each test module's
first test runs, scan sys.modules for entries that were NOT present at baseline
time and look like hand-built stubs.  Remove those stubs so that subsequent
import statements use the real import machinery.

A module is considered a "stub" if it was added after baseline AND:
  1. Is a MagicMock (used by some test files as module stand-ins), OR
  2. Is a types.ModuleType with no __file__, no __spec__, and no __loader__
     (real modules always have at least one of these set by importlib)

We explicitly avoid removing:
  - C extensions (they can't be re-loaded)
  - Anything with __file__ or __spec__ (properly imported)
  - Baseline modules (were there before any test ran)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, NonCallableMagicMock

# Keys present in sys.modules before any unit-test file is collected.
_BASELINE_KEYS: frozenset[str] = frozenset(sys.modules.keys())

# Track the last test module whose stubs are currently active.
_active_module: str | None = None


def _is_stub(mod: object) -> bool:
    """Return True if *mod* looks like a hand-built stub, not a real import."""
    if mod is None:
        return True

    # MagicMock / NonCallableMagicMock used as module stand-ins.
    if isinstance(mod, (MagicMock, NonCallableMagicMock)):
        return True

    # Only consider actual module objects from here on.
    if not isinstance(mod, types.ModuleType):
        return False

    # Real modules always get __file__ set (except namespace packages).
    if getattr(mod, "__file__", None) is not None:
        return False

    # Real modules always get __spec__ set by the import machinery.
    if getattr(mod, "__spec__", None) is not None:
        return False

    # Real modules get __loader__ set.
    if getattr(mod, "__loader__", None) is not None:
        return False

    # At this point: no __file__, no __spec__, no __loader__.
    # This is almost certainly a hand-built types.ModuleType("name") stub.
    return True


def _purge_stubs() -> None:
    """Remove stub entries from sys.modules that were added after baseline."""
    to_remove = []
    for key in sys.modules:
        if key in _BASELINE_KEYS:
            continue
        if _is_stub(sys.modules[key]):
            to_remove.append(key)

    for key in to_remove:
        del sys.modules[key]


# ---------------------------------------------------------------------------
# Collection-time hook
# ---------------------------------------------------------------------------


def pytest_collectstart(collector) -> None:  # noqa: ANN001
    """Purge stubs before each Module is imported."""
    from _pytest.python import Module

    if isinstance(collector, Module):
        _purge_stubs()


# ---------------------------------------------------------------------------
# Execution-time hook
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item) -> None:  # noqa: ANN001
    """Purge stubs on module transition during test execution."""
    global _active_module  # noqa: PLW0603
    filepath = str(item.fspath)
    if filepath != _active_module:
        _purge_stubs()
        _active_module = filepath
