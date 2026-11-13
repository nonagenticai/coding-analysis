"""Python import resolution in the module graph: source roots, stdlib shadowing, guards."""

from __future__ import annotations

import pytest

from processing.structure.module_graph import python_graph


def _edges(files: dict[str, str], root: str = "") -> list[tuple[str, str, str]]:
    return sorted((e.source, e.target, e.kind) for e in python_graph(files, root).edges)


# --------------------------------------------------------------------------- source roots


def test_stdlib_import_never_matches_a_same_named_internal_module():
    files = {
        "app/__init__.py": "",
        "app/logging.py": "import app.main\n",
        "app/main.py": "import logging\nimport json\n",
        "app/json.py": "X = 1\n",
    }
    g = python_graph(files)
    # logging.py -> main is real; main's `import logging` is the stdlib, not app/logging.py.
    assert _edges(files) == [("app.logging", "app.main", "import")]
    assert g.sccs == []


def test_third_party_import_never_matches_by_suffix():
    files = {
        "app/__init__.py": "",
        "app/requests.py": "X = 1\n",
        "app/main.py": "import requests\n",
    }
    assert _edges(files) == []


def test_src_layout_without_root_resolves_through_the_source_root():
    files = {
        "src/app/__init__.py": "",
        "src/app/main.py": "from app.db import engine\nimport app.util\n",
        "src/app/db.py": "engine = 1\n",
        "src/app/util.py": "",
    }
    # src/ is a source root, not a package: modules are named as Python imports them.
    assert _edges(files) == [
        ("app.main", "app.db", "from_import"),
        ("app.main", "app.util", "import"),
    ]


def test_inner_package_suffix_is_not_a_source_root():
    # `import db` from inside package `app` is NOT `app.db` in Python 3.
    files = {
        "src/app/__init__.py": "",
        "src/app/main.py": "import db\n",
        "src/app/db.py": "",
    }
    assert _edges(files) == []


def test_script_directory_is_on_the_import_path():
    files = {"scripts/run.py": "import helper\n", "scripts/helper.py": "X = 1\n"}
    assert _edges(files) == [("scripts.run", "scripts.helper", "import")]


def test_script_directory_still_never_shadows_the_stdlib():
    files = {"scripts/run.py": "import logging\n", "scripts/logging.py": "X = 1\n"}
    assert _edges(files) == []


def test_top_level_module_can_shadow_stdlib_like_the_interpreter():
    files = {"main.py": "import logging\n", "logging.py": "X = 1\n"}
    assert _edges(files) == [("main", "logging", "import")]


def test_import_ambiguous_across_source_roots_is_unresolved_with_candidates():
    files = {
        "a/pkg/__init__.py": "",
        "a/pkg/x.py": "",
        "b/pkg/__init__.py": "",
        "b/pkg/x.py": "",
        "tool.py": "import pkg.x\n",
    }
    g = python_graph(files)
    assert g.edges == []
    (u,) = [u for u in g.unresolved if u.name == "pkg.x"]
    assert u.candidates == ["a.pkg.x", "b.pkg.x"]


# --------------------------------------------------------------------------- guards / relative


def test_if_not_type_checking_is_a_runtime_import():
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": (
            "from typing import TYPE_CHECKING\n"
            "if not TYPE_CHECKING:\n    from pkg import b\n"
            "else:\n    from pkg import c\n"
        ),
        "pkg/b.py": "",
        "pkg/c.py": "",
    }
    assert _edges(files) == [
        ("pkg.a", "pkg.b", "from_import"),
        ("pkg.a", "pkg.c", "type_only"),
    ]


def test_if_type_checking_is_type_only():
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": "import typing\nif typing.TYPE_CHECKING:\n    from pkg import b\n",
        "pkg/b.py": "",
    }
    assert _edges(files) == [("pkg.a", "pkg.b", "type_only")]


def test_relative_import_climbing_past_the_package_root_is_unresolved():
    files = {"pkg/__init__.py": "", "pkg/a.py": "from .. import other\n"}
    g = python_graph(files)
    assert g.edges == []
    assert [(u.file, u.name, u.reason) for u in g.unresolved] == [
        ("pkg/a.py", "..", "relative import climbs past the top-level package")
    ]


def test_top_level_from_dot_import_links_the_sibling_module():
    files = {"a.py": "from . import b\n", "b.py": "X = 1\n"}
    assert _edges(files) == [("a", "b", "from_import")]
    files = {"src/a.py": "from . import b\n", "src/b.py": "X = 1\n"}
    assert _edges(files, root="src") == [("a", "b", "from_import")]


def test_relative_import_is_exact_not_root_searched():
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": "from .missing import x\n",
        "missing.py": "",
    }
    # Never the top-level missing.py; only the importing package itself.
    assert _edges(files) == [("pkg.a", "pkg", "from_import")]


# --------------------------------------------------------------------------- declared source roots


def _modules(files: dict[str, str], root: str = "") -> list[str]:
    return sorted(n.module for n in python_graph(files, root).nodes)


def test_pyproject_setuptools_package_dir_names_modules_from_that_root():
    files = {
        "pyproject.toml": '[tool.setuptools]\npackage-dir = {"" = "lib"}\n',
        "lib/pkg/__init__.py": "",
        "lib/pkg/a.py": "from pkg import b\n",
        "lib/pkg/b.py": "X = 1\n",
    }
    assert _modules(files) == ["pkg", "pkg.a", "pkg.b"]
    assert _edges(files) == [("pkg.a", "pkg.b", "from_import")]


def test_pyproject_in_a_subdir_declares_roots_relative_to_itself():
    files = {
        "svc/pyproject.toml": '[tool.setuptools.packages.find]\nwhere = ["code"]\n',
        "svc/code/pkg/__init__.py": "",
        "svc/code/pkg/a.py": "import pkg.b\n",
        "svc/code/pkg/b.py": "",
        "tools/run.py": "",
    }
    # Outside every source root a file keeps its path-derived name.
    assert _modules(files) == ["pkg", "pkg.a", "pkg.b", "tools.run"]
    assert _edges(files) == [("pkg.a", "pkg.b", "import")]


@pytest.mark.parametrize(
    "manifest, text",
    [
        ("pyproject.toml", '[[tool.poetry.packages]]\ninclude = "pkg"\nfrom = "py"\n'),
        ("pyproject.toml", '[tool.hatch.build.targets.wheel]\npackages = ["py/pkg"]\n'),
        ("pyproject.toml", '[tool.pdm.build]\npackage-dir = "py"\n'),
        ("setup.cfg", "[options]\npackage_dir =\n    =py\n"),
        ("setup.cfg", "[options.packages.find]\nwhere = py\n"),
    ],
)
def test_every_declared_root_form_is_honoured(manifest, text):
    files = {manifest: text, "py/pkg/__init__.py": "", "py/pkg/a.py": ""}
    assert _modules(files) == ["pkg", "pkg.a"]


def test_declared_root_that_leaves_the_repo_is_ignored():
    files = {
        "pyproject.toml": '[tool.setuptools]\npackage-dir = {"" = "../elsewhere"}\n',
        "pkg/__init__.py": "",
        "pkg/a.py": "",
    }
    assert _modules(files) == ["pkg", "pkg.a"]


def test_src_that_is_itself_a_package_is_not_a_root():
    # Negative control: src/__init__.py makes `src` a real package, named as such.
    files = {"src/__init__.py": "", "src/pkg/__init__.py": "", "src/pkg/a.py": ""}
    assert _modules(files) == ["src", "src.pkg", "src.pkg.a"]
