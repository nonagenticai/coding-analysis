"""Regression tests: real Python import cycles must be detected.

Before the fix, the deep analyzer emitted import edges with a synthetic
``module_<dotted>`` target that no node ever had, so every import edge was
dropped as an orphan (the graph-adapter JOIN keeps only edges whose both
endpoints are real nodes). Component dependencies were therefore empty and
``verify-architecture``'s cycle check returned ``[]`` even for a textbook import
cycle — a *vacuous* pass. These tests pin the resolution-by-full-module-path fix
in ``processing.analyzers.deep.entities``.

They drive the LIVE path (``DeepCodeAnalyzer._extract_entities``), rebuild the
graph exactly as ``GraphAdapter`` does (edge kept only when both endpoints are
real nodes), then run the real ``ComponentExtractor`` and the architecture
endpoint's Tarjan cycle finder — no database required.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from app.api.endpoints.architecture import _find_cycle_signatures
from models.architecture_models import RepoConfig
from processing.analyzers.deep.analyzer import DeepCodeAnalyzer
from processing.analyzers.deep.entities import (
    _module_keys,
    _resolve_import_target,
    _source_roots,
)
from processing.architecture.component_extractor import ComponentExtractor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_repo(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _build_graph(entities, relationships) -> nx.DiGraph:
    """Rebuild the architecture graph the way ``GraphAdapter.load_graph`` does:
    only relationships whose source AND target are real nodes survive (mirrors
    the SQL JOIN + orphan-edge cleanup)."""
    graph: nx.DiGraph = nx.DiGraph()
    node_ids: set[str] = set()
    for e in entities:
        etype = e.type.value if hasattr(e.type, "value") else str(e.type)
        graph.add_node(
            e.id,
            type=etype,
            name=getattr(e, "name", None),
            file_path=getattr(e, "file_path", None) or "",
            properties=getattr(e, "metadata", {}) or {},
        )
        node_ids.add(e.id)
    for r in relationships:
        s = getattr(r, "source_id", None)
        t = getattr(r, "target_id", None)
        if s in node_ids and t in node_ids:
            rel = getattr(r, "type", None)
            rel = rel.value if hasattr(rel, "value") else str(rel)
            # Carry edge metadata into ``properties`` exactly as GraphAdapter does
            # (graph_adapter.py loads edges.properties, which is populated from the
            # CodeRelationship.metadata by analysis_endpoints). ComponentExtractor
            # reads ``properties`` to drop deferred (function-local) import edges.
            props = getattr(r, "metadata", {}) or {}
            graph.add_edge(s, t, relationship=rel, properties=props)
    return graph


def _component_cycles(root: Path) -> list[list[str]]:
    analyzer = DeepCodeAnalyzer(str(root))
    analyzer._extract_entities()  # noqa: SLF001 - drives the live extraction path
    graph = _build_graph(analyzer.entities, analyzer.relationships)
    components = ComponentExtractor().extract(graph, RepoConfig())
    adjacency = {c.id: list(c.dependencies or []) for c in components}
    return _find_cycle_signatures(adjacency)


# --------------------------------------------------------------------------- #
# Unit tests for the resolution helpers
# --------------------------------------------------------------------------- #
def test_module_keys():
    assert _module_keys("pkg/core.py") == ["pkg.core"]
    assert _module_keys("a/b/c.py") == ["a.b.c"]
    assert _module_keys("pkg/__init__.py") == ["pkg"]
    assert _module_keys("a/b/__init__.py") == ["a.b"]
    assert _module_keys("README.md") == []


def test_resolve_full_path_not_basename():
    # Two modules share the basename 'core' but live in different packages.
    index = {
        "pkg_a.core": "FID_A",
        "pkg_b.core": "FID_B",
    }
    # `from pkg_b.core import f` resolves to pkg_b/core.py, never collides on 'core'.
    assert _resolve_import_target(index, "pkg_b.core", ["f"], 0, "pkg_a/core.py") == "FID_B"
    # `from pkg_a import core` (submodule import) resolves to pkg_a/core.py.
    assert _resolve_import_target(index, "pkg_a", ["core"], 0, "pkg_b/core.py") == "FID_A"


def test_resolve_relative_import():
    index = {"pkg.core": "FID_CORE", "pkg.util": "FID_UTIL"}
    # `from .util import f` inside pkg/core.py -> pkg.util
    assert _resolve_import_target(index, "util", ["f"], 1, "pkg/core.py") == "FID_UTIL"
    # `from . import util` inside pkg/core.py -> pkg.util
    assert _resolve_import_target(index, None, ["util"], 1, "pkg/core.py") == "FID_UTIL"


def test_resolve_external_import_is_unresolved():
    index = {"pkg.core": "FID_CORE"}
    # stdlib / third-party imports are not local -> no intra-repo edge.
    assert _resolve_import_target(index, "os", ["path"], 0, "pkg/core.py") is None
    assert _resolve_import_target(index, "numpy", ["array"], 0, "pkg/core.py") is None


# --------------------------------------------------------------------------- #
# End-to-end cycle detection
# --------------------------------------------------------------------------- #
def test_same_basename_three_package_cycle_detected(tmp_path):
    """The regression: 3 packages each with a `core.py`, mutually importing.

    Pre-fix this produced no edges (synthetic targets dropped) and no cycle.
    """
    _write_repo(
        tmp_path,
        {
            "pkg_alpha/__init__.py": "",
            "pkg_alpha/core.py": (
                "from pkg_beta import core as beta_core\n\n"
                "def alpha_fn():\n    return beta_core.beta_fn()\n"
            ),
            "pkg_beta/__init__.py": "",
            "pkg_beta/core.py": (
                "from pkg_gamma import core as gamma_core\n\n"
                "def beta_fn():\n    return gamma_core.gamma_fn()\n"
            ),
            "pkg_gamma/__init__.py": "",
            "pkg_gamma/core.py": (
                "from pkg_alpha import core as alpha_core\n\n"
                "def gamma_fn():\n    return alpha_core.alpha_fn()\n"
            ),
        },
    )
    cycles = _component_cycles(tmp_path)
    assert len(cycles) == 1
    assert set(cycles[0]) == {
        "component_pkg_alpha",
        "component_pkg_beta",
        "component_pkg_gamma",
    }


def test_distinct_module_names_two_package_cycle_detected(tmp_path):
    """Locks the previously-working distinct-basename path (full-path imports)."""
    _write_repo(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/alpha.py": ("from pkg_b.beta import f_b\n\ndef f_a():\n    return f_b()\n"),
            "pkg_b/__init__.py": "",
            "pkg_b/beta.py": ("from pkg_a.alpha import f_a\n\ndef f_b():\n    return f_a()\n"),
        },
    )
    cycles = _component_cycles(tmp_path)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"component_pkg_a", "component_pkg_b"}


def test_acyclic_repo_has_no_cycle(tmp_path):
    """A clean DAG (a -> b -> c) must NOT be reported as a cycle."""
    _write_repo(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/core.py": "from pkg_b import core as b\n\ndef fa():\n    return b.fb()\n",
            "pkg_b/__init__.py": "",
            "pkg_b/core.py": "from pkg_c import core as c\n\ndef fb():\n    return c.fc()\n",
            "pkg_c/__init__.py": "",
            "pkg_c/core.py": "def fc():\n    return 1\n",
        },
    )
    assert _component_cycles(tmp_path) == []


def test_external_imports_do_not_create_local_edges(tmp_path):
    """stdlib/third-party imports stay unresolved (no false intra-repo edges)."""
    _write_repo(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/core.py": "import os\nfrom typing import Any\n\ndef fa(x: Any):\n    return os.getcwd()\n",
        },
    )
    analyzer = DeepCodeAnalyzer(str(tmp_path))
    analyzer._extract_entities()  # noqa: SLF001
    import_targets = [
        r.target_id
        for r in analyzer.relationships
        if getattr(r, "type", None) in ("imports", "imports_from")
    ]
    # external imports keep the synthetic, unresolved target (dropped downstream)
    assert all(t.startswith("module_") for t in import_targets)
    assert _component_cycles(tmp_path) == []


# --------------------------------------------------------------------------- #
# Deferred (function-local) imports do not participate in an import-time cycle
# --------------------------------------------------------------------------- #
def test_deferred_function_local_import_breaks_cycle(tmp_path):
    """A function-local (deferred) import is the standard Python idiom for breaking
    an import-time cycle and is exactly what the AA refactor worker emits. The
    deep analyzer must mark it ``deferred`` and ComponentExtractor must exclude it
    from architecture dependency edges, so the cycle is NO LONGER reported.

    Same 3-package cycle as ``test_same_basename_three_package_cycle_detected``,
    except gamma's import of alpha is moved inside a function body.
    """
    _write_repo(
        tmp_path,
        {
            "aacyc_alpha/__init__.py": "",
            "aacyc_alpha/core.py": (
                "from aacyc_beta import core as _b\n\ndef a():\n    return 'a'\n"
            ),
            "aacyc_beta/__init__.py": "",
            "aacyc_beta/core.py": (
                "from aacyc_gamma import core as _c\n\ndef b():\n    return 'b'\n"
            ),
            "aacyc_gamma/__init__.py": "",
            # gamma -> alpha edge is DEFERRED (relocated into the function body)
            "aacyc_gamma/core.py": (
                "def c():\n    from aacyc_alpha import core as _a  # deferred\n    return 'c'\n"
            ),
        },
    )
    # The deferred import is still extracted as an edge, but marked deferred.
    analyzer = DeepCodeAnalyzer(str(tmp_path))
    analyzer._extract_entities()  # noqa: SLF001
    gamma_alpha = [
        r
        for r in analyzer.relationships
        if getattr(r, "type", None) == "imports_from"
        and (r.metadata or {}).get("module") == "aacyc_alpha"
    ]
    assert gamma_alpha, "expected the gamma->alpha imports_from edge to be extracted"
    assert all(r.metadata.get("deferred") is True for r in gamma_alpha)
    # …and because it is deferred, the import-time cycle is broken.
    assert _component_cycles(tmp_path) == []


def test_module_level_import_is_not_marked_deferred(tmp_path):
    """A top-level import must keep ``deferred=False`` so the module-level cycle
    (the textbook ImportError case) is still detected — no regression."""
    _write_repo(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/core.py": "from pkg_b import core as b\n\ndef fa():\n    return b.fb()\n",
            "pkg_b/__init__.py": "",
            "pkg_b/core.py": "def fb():\n    return 1\n",
        },
    )
    analyzer = DeepCodeAnalyzer(str(tmp_path))
    analyzer._extract_entities()  # noqa: SLF001
    a_to_b = [
        r
        for r in analyzer.relationships
        if getattr(r, "type", None) == "imports_from"
        and (r.metadata or {}).get("module") == "pkg_b"
    ]
    assert a_to_b and all(r.metadata.get("deferred") is False for r in a_to_b)


# --------------------------------------------------------------------------- #
# src-layout: ``src/`` is a source root, imports drop the ``src.`` prefix
# --------------------------------------------------------------------------- #
def test_source_roots_detects_src_layout(tmp_path):
    """A top-level ``src/`` with no ``__init__.py`` is a source root."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
    assert _source_roots(tmp_path) == {"src"}


def test_source_roots_ignores_real_src_package(tmp_path):
    """A genuine package literally named ``src`` (``src/__init__.py`` present) is
    imported as ``src.foo`` and must NOT be treated as a strippable source root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    assert _source_roots(tmp_path) == set()


def test_src_layout_cycle_detected(tmp_path):
    """The core bug: a 3-package cycle living under ``src/`` (src-layout) was
    INVISIBLE before the fix — ``from aacyc_beta import core`` resolved to nothing
    because the index was keyed ``src.aacyc_beta.core``. Now the cycle is detected.
    """
    _write_repo(
        tmp_path,
        {
            "src/aacyc_alpha/__init__.py": "",
            "src/aacyc_alpha/core.py": (
                "from aacyc_beta import core as _b\n\ndef a():\n    return 'a'\n"
            ),
            "src/aacyc_beta/__init__.py": "",
            "src/aacyc_beta/core.py": (
                "from aacyc_gamma import core as _c\n\ndef b():\n    return 'b'\n"
            ),
            "src/aacyc_gamma/__init__.py": "",
            "src/aacyc_gamma/core.py": (
                "from aacyc_alpha import core as _a\n\ndef c():\n    return 'c'\n"
            ),
        },
    )
    cycles = _component_cycles(tmp_path)
    assert len(cycles) == 1
    assert set(cycles[0]) == {
        "component_aacyc_alpha",
        "component_aacyc_beta",
        "component_aacyc_gamma",
    }


def test_src_layout_deferred_import_breaks_cycle(tmp_path):
    """The live arch_refactoring idiom under a realistic src-layout: the worker
    relocates gamma's import of alpha into a function body (deferred). The analyzer
    must resolve the src-layout imports AND exclude the deferred edge, so the
    import-time cycle is broken — a non-vacuous structural PASS."""
    _write_repo(
        tmp_path,
        {
            "src/aacyc_alpha/__init__.py": "",
            "src/aacyc_alpha/core.py": (
                "from aacyc_beta import core as _b\n\ndef a():\n    return 'a'\n"
            ),
            "src/aacyc_beta/__init__.py": "",
            "src/aacyc_beta/core.py": (
                "from aacyc_gamma import core as _c\n\ndef b():\n    return 'b'\n"
            ),
            "src/aacyc_gamma/__init__.py": "",
            # gamma -> alpha edge is DEFERRED (relocated into the function body)
            "src/aacyc_gamma/core.py": (
                "def c():\n    from aacyc_alpha import core as _a  # deferred\n    return 'c'\n"
            ),
        },
    )
    # The deferred src-layout import is still extracted (and resolved), but marked.
    analyzer = DeepCodeAnalyzer(str(tmp_path))
    analyzer._extract_entities()  # noqa: SLF001
    gamma_alpha = [
        r
        for r in analyzer.relationships
        if getattr(r, "type", None) == "imports_from"
        and (r.metadata or {}).get("module") == "aacyc_alpha"
    ]
    assert gamma_alpha, "expected the gamma->alpha imports_from edge to be extracted"
    assert all(r.metadata.get("deferred") is True for r in gamma_alpha)
    # …and because it is deferred, the import-time cycle is broken.
    assert _component_cycles(tmp_path) == []


def test_src_layout_acyclic_no_false_cycle(tmp_path):
    """A clean src-layout DAG (a -> b -> c) must NOT be reported as a cycle —
    the source-root stripping introduces no spurious edges."""
    _write_repo(
        tmp_path,
        {
            "src/pkg_a/__init__.py": "",
            "src/pkg_a/core.py": "from pkg_b import core as b\n\ndef fa():\n    return b.fb()\n",
            "src/pkg_b/__init__.py": "",
            "src/pkg_b/core.py": "from pkg_c import core as c\n\ndef fb():\n    return c.fc()\n",
            "src/pkg_c/__init__.py": "",
            "src/pkg_c/core.py": "def fc():\n    return 1\n",
        },
    )
    assert _component_cycles(tmp_path) == []
