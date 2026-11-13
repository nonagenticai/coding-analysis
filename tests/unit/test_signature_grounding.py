"""Signature analysis relates functions ACROSS files and checks call arity.

Live 2026-10-12 on cv-m2-hello (planted ``pkg/sig.py``: ``add(a, b)`` called as
``add(1)``): the op returned the real signature but nothing that could catch the
bad call -- calls/called_by were empty and usage_examples were invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.endpoints.static_analysis import _analyze_files
from processing.ast_analysis.function_extractor import FunctionExtractor


def _repo(tmp_path: Path, files: dict[str, str]) -> list[Path]:
    paths = []
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        paths.append(p)
    return sorted(paths)


FILES = {
    "pkg/__init__.py": "",
    "pkg/sig.py": (
        "def add(a: int, b: int) -> int:\n    return a + b\n\n\n"
        "def caller():\n    return add(1)\n\n\n"
        "def fine():\n    return add(1, b=2)\n"
    ),
    "pkg/other.py": (
        "import requests\nfrom pkg.sig import add\n\n\n"
        "def remote():\n    requests.get('x')\n    return add(1, 2, 3)\n\n\n"
        "def kw():\n    return add(a=1, c=2)\n"
    ),
    "pkg/shadow.py": (
        "def add(x):\n    return x\n"  # a second `add`: binding no longer certain
    ),
    "pkg/broken.py": "def oops(:\n",
}


def test_mismatches_found_and_clean_calls_pass(tmp_path):
    files = {k: v for k, v in FILES.items() if k != "pkg/shadow.py"}
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    found = {(m["caller"], m["line"]): m["reason"] for m in result["call_arity_mismatches"]}
    assert ("pkg.sig.caller", 6) in found and "missing required argument(s) ['b']" in found[
        ("pkg.sig.caller", 6)
    ]
    assert ("pkg.other.remote", 7) in found and "3 positional" in found[("pkg.other.remote", 7)]
    assert ("pkg.other.kw", 11) in found and "unknown keyword(s) ['c']" in found[
        ("pkg.other.kw", 11)
    ]
    assert not any(c == "pkg.sig.fine" for c, _ in found)  # add(1, b=2) binds


def test_ambiguous_callee_is_not_judged(tmp_path):
    result = _analyze_files(_repo(tmp_path, FILES), FunctionExtractor(), str(tmp_path))
    assert result["call_arity_mismatches"] == []


def test_cross_file_callers_externals_and_unparsed_files(tmp_path):
    files = {k: v for k, v in FILES.items() if k != "pkg/shadow.py"}
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    add = next(f for f in result["functions"] if f["name"] == "add")
    assert set(add["called_by"]) == {
        "pkg.sig.caller",
        "pkg.sig.fine",
        "pkg.other.remote",
        "pkg.other.kw",
    }
    assert "add(1)" in add["usage_examples"]
    assert result["dependencies"] == {"requests"}  # pkg.sig is in-repo, not external
    assert [f["file"] for f in result["files_not_computed"]] == ["pkg/broken.py"]


# --- binding like Python does (validator false positives, 2026-10-12) ---------

BINDING_DEFS = (
    "def kw(a, *, b):\n    return a + b\n\n\n"
    "def po(a, /, b):\n    return a + b\n\n\n"
    "def pod(a, /, b=2):\n    return a + b\n\n\n"
    "def add(a, b):\n    return a + b\n"
)


def _mismatches(tmp_path: Path, files: dict[str, str]) -> dict[str, str]:
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    return {m["call"]: m["reason"] for m in result["call_arity_mismatches"]}


def test_kw_only_pos_only_and_defaults_bind_cleanly(tmp_path):
    found = _mismatches(
        tmp_path,
        {
            "pkg/defs.py": BINDING_DEFS,
            "pkg/use.py": (
                "from pkg.defs import kw, po, pod, add\n\n\n"
                "def ok():\n"
                "    kw(1, b=2)\n    po(1, 2)\n    po(1, b=2)\n    pod(1)\n    add(1, b=2)\n"
            ),
        },
    )
    assert found == {}


def test_real_binding_errors_still_fire(tmp_path):
    found = _mismatches(
        tmp_path,
        {
            "pkg/defs.py": BINDING_DEFS,
            "pkg/use.py": (
                "from pkg.defs import kw, po, add\n\n\n"
                "def bad():\n"
                "    kw(1)\n    kw(1, 2)\n    po(a=1, b=2)\n    add(1, a=2)\n"
            ),
        },
    )
    assert "missing required argument(s) ['b']" in found["kw(1)"]
    assert "2 positional arguments, at most 1" in found["kw(1, 2)"]
    assert "positional-only argument(s) passed by keyword ['a']" in found["po(a=1, b=2)"]
    assert "multiple values for argument(s) ['a']" in found["add(1, a=2)"]


def test_signature_renders_slash_and_star(tmp_path):
    result = _analyze_files(
        _repo(tmp_path, {"pkg/defs.py": BINDING_DEFS}), FunctionExtractor(), str(tmp_path)
    )
    sigs = {f["name"]: f["signature"] for f in result["functions"]}
    assert sigs["kw"] == "def kw(a, *, b)"
    assert sigs["po"] == "def po(a, /, b)"
    assert sigs["pod"].startswith("def pod(a, /, b=")  # default text is astor's
    po = next(f for f in result["functions"] if f["name"] == "po")
    assert [p["kind"] for p in po["parameters"]] == ["positional_only", "positional_or_keyword"]
    kw = next(f for f in result["functions"] if f["name"] == "kw")
    assert [(p["name"], p["required"], p["kind"]) for p in kw["parameters"]] == [
        ("a", True, "positional_or_keyword"),
        ("b", True, "keyword_only"),
    ]


def test_decorated_callee_is_not_judged(tmp_path):
    found = _mismatches(
        tmp_path,
        {
            "pkg/cli.py": (
                "import click\n\n\n"
                "@click.command()\n@click.option('--x')\ndef main(x):\n    print(x)\n\n\n"
                "def run():\n    main()\n"
            )
        },
    )
    assert found == {}


SHADOWS = {
    "lambda": (
        "def add(a, b):\n    return a + b\n\n\n"
        "def caller():\n    add = lambda a, b, c: a\n    return add(1, 2, 3)\n"
    ),
    "nested_def": (
        "def add(a, b):\n    return a + b\n\n\n"
        "def caller():\n    def add(a, b, c):\n        return a\n    return add(1, 2, 3)\n"
    ),
    "parameter": (
        "def add(a, b):\n    return a + b\n\n\ndef caller(add):\n    return add(1, 2, 3)\n"
    ),
    "import_as": (
        "from operator import mul as add\n\n\n"
        "def add(a, b):\n    return a + b\n\n\n"
        "def caller():\n    return add(1, 2, 3)\n"
    ),
}


def test_rebound_name_is_not_judged(tmp_path):
    for case, text in SHADOWS.items():
        found = _mismatches(tmp_path / case, {"pkg/m.py": text})
        assert found == {}, case


def test_import_as_shadow_in_other_module(tmp_path):
    found = _mismatches(
        tmp_path,
        {
            "pkg/defs.py": BINDING_DEFS,
            "pkg/use.py": "import pkg.defs as add\n\n\ndef caller():\n    return add(1, 2, 3)\n",
        },
    )
    assert found == {}


# The CALLEE's own module rebinds the name, or defines it conditionally: the
# call may not reach the def, so it is never judged (validator repros, b3a4dd8).
CALLEE_REBINDS = {
    "partial": (
        "import functools\n\n\ndef f(a, b):\n    return a + b\n\n\nf = functools.partial(f, 1)\n",
        "from pkg.a import f\n\n\ndef use():\n    return f(2)\n",
    ),
    "wrap_by_assignment": (
        "def wrap(fn):\n    return fn\n\n\ndef f(a, b):\n    return a\n\n\nf = wrap(f)\n",
        "from pkg.a import f\n\n\ndef use():\n    return f(1, 2, 3)\n",
    ),
    "class_after_def": (
        "def f(a, b):\n    return a\n\n\nclass f:\n    def __init__(self):\n        pass\n",
        "from pkg.a import f\n\n\ndef use():\n    return f()\n",
    ),
    "import_after_def": (
        "def f(a, b):\n    return a\n\n\nfrom os.path import basename as f\n",
        "from pkg.a import f\n\n\ndef use():\n    return f('x')\n",
    ),
    "global_rebind": (
        "def f(a, b):\n    return a\n\n\ndef setup():\n    global f\n    f = lambda a: a\n",
        "from pkg.a import f\n\n\ndef use():\n    return f(1)\n",
    ),
    "try_import_fallback": (
        "try:\n    from _speedups import f\nexcept ImportError:\n\n"
        "    def f(a, b):\n        return a\n",
        "from pkg.a import f\n\n\ndef use():\n    return f(1)\n",
    ),
    "globals_write": (
        "def f(a, b):\n    return a\n\n\nglobals()['f'] = lambda a: a\n",
        "from pkg.a import f\n\n\ndef use():\n    return f(1)\n",
    ),
}


def test_callee_module_rebinding_is_not_judged(tmp_path):
    for case, (callee, caller) in CALLEE_REBINDS.items():
        found = _mismatches(
            tmp_path / case,
            {"pkg/__init__.py": "", "pkg/a.py": callee, "pkg/b.py": caller},
        )
        assert found == {}, case


def test_conditional_def_is_not_top_level(tmp_path):
    result = _analyze_files(
        _repo(
            tmp_path,
            {
                "pkg/a.py": "import sys\n\nif sys:\n\n    def f(a):\n        return a\n\n\n"
                "def g(a):\n    return a\n"
            },
        ),
        FunctionExtractor(),
        str(tmp_path),
    )
    top = {f["name"]: f["is_top_level"] for f in result["functions"]}
    assert top == {"f": False, "g": True}


def test_untouched_callee_still_judged_across_modules(tmp_path):
    found = _mismatches(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "import os\n\n\ndef f(a, b):\n    return os.sep\n",
            "pkg/b.py": "from pkg.a import f\n\n\ndef use():\n    return f(1)\n",
        },
    )
    assert found == {"f(1)": "missing required argument(s) ['b']"}


def test_var_args_have_no_invented_type_and_defaults_unparse(tmp_path):
    result = _analyze_files(
        _repo(tmp_path, {"pkg/a.py": "def f(a=(1, 2), *args, **kw):\n    return a\n"}),
        FunctionExtractor(),
        str(tmp_path),
    )
    (func,) = result["functions"]
    assert [p["type"] for p in func["parameters"]] == [None, None, None]
    assert func["signature"] == "def f(a=(1, 2), *args, **kw)"


_A2 = "def f(a, b):\n    return a\n"
_USE_F1 = "from pkg.a import f\n\n\ndef use():\n    return f(1)\n"
_DYNAMIC_CASES = {
    "sys_modules_attr": {
        "pkg/a.py": "import sys\n\n\n" + _A2 + "sys.modules[__name__].f = lambda a: a\n"
    },
    "setattr_module": {
        "pkg/a.py": "import sys\n\n\n" + _A2 + "setattr(sys.modules[__name__], 'f', lambda a: a)\n"
    },
    "globals_update": {"pkg/a.py": _A2 + "globals().update(f=lambda a: a)\n"},
    "monkeypatch_other_module": {
        "pkg/a.py": _A2,
        "pkg/patch.py": "import pkg.a as A\n\nA.f = lambda x: x\n",
    },
    "defaults_attr": {"pkg/a.py": _A2 + "f.__defaults__ = (0,)\n"},
    "exec_def": {"pkg/a.py": _A2 + "exec('def f(a):\\n    return a')\n"},
}


@pytest.mark.parametrize("case", sorted(_DYNAMIC_CASES))
def test_dynamic_rebinding_is_not_judged(tmp_path, case):
    files = _DYNAMIC_CASES[case]
    found = _mismatches(tmp_path, {"pkg/__init__.py": "", "pkg/b.py": _USE_F1, **files})
    assert found == {}, case


def test_plain_wrong_arity_still_fires_without_dynamic_constructs(tmp_path):
    found = _mismatches(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": _A2, "pkg/b.py": _USE_F1})
    assert found == {"f(1)": "missing required argument(s) ['b']"}


# --- a THIRD module rewrites the callee (round-4 validator, 2026-10-12) -------

_CROSS_MODULE_PATCHES = {
    "defaults_via_from_import": "from pkg.a import f\n\nf.__defaults__ = (0,)\n",
    "defaults_dotted": "import pkg.a\n\npkg.a.f.__defaults__ = (0,)\n",
    "code_attr": "from pkg.a import f\n\nf.__code__ = (lambda a: a).__code__\n",
    "setattr_const": "import pkg.a\n\nsetattr(pkg.a, 'f', lambda a: a)\n",
    "dunder_dict_key": "import pkg.a\n\npkg.a.__dict__['f'] = lambda a: a\n",
    "vars_key": "import pkg.a\n\nvars(pkg.a)['f'] = lambda a: a\n",
    "dunder_dict_update": "import pkg.a as A\n\nA.__dict__.update(f=lambda a: a)\n",
    "mock_patch": "from unittest import mock\n\nmock.patch('pkg.b.f', lambda a: a).start()\n",
    "monkeypatch_dotted": "def test_x(monkeypatch):\n    monkeypatch.setattr('pkg.a.f', len)\n",
}


@pytest.mark.parametrize("case", sorted(_CROSS_MODULE_PATCHES))
def test_third_module_rewrite_is_not_judged(tmp_path, case):
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": _A2,
        "pkg/b.py": _USE_F1,
        "pkg/p.py": _CROSS_MODULE_PATCHES[case],
    }
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    assert result["call_arity_mismatches"] == [], case
    assert result["call_arity_status"] == "computed", case


def test_non_constant_setattr_on_module_disables_arity(tmp_path):
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": _A2,
        "pkg/b.py": _USE_F1,
        "pkg/p.py": "import pkg.a\n\n\ndef patch(name, fn):\n    setattr(pkg.a, name, fn)\n",
    }
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    assert result["call_arity_mismatches"] == []
    assert result["call_arity_status"] == "not_computed"
    assert "pkg.p" in result["call_arity_reason"]


def test_non_constant_setattr_on_self_keeps_arity(tmp_path):
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": _A2,
        "pkg/b.py": _USE_F1,
        "pkg/p.py": (
            "class C:\n    def __init__(self, **kw):\n"
            "        for k, v in kw.items():\n            setattr(self, k, v)\n"
        ),
    }
    result = _analyze_files(_repo(tmp_path, files), FunctionExtractor(), str(tmp_path))
    assert result["call_arity_status"] == "computed"
    assert [m["call"] for m in result["call_arity_mismatches"]] == ["f(1)"]
