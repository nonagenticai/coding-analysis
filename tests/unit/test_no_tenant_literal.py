"""The ``"n8n-dev"`` tenant literal must not come back into ``src/``.

WHAT WAS WRONG
--------------
This service carried the literal in **two** independent settings classes::

    src/config/settings.py:39                    DEFAULT_TENANT_ID: str = "n8n-dev"
    src/app/core/config/core_config_settings.py  default_tenant_id: str = "n8n-dev"

Both had **zero readers** — ``src/auth/tenant.py`` imports the name from
``coding_client.auth.tenant``, where it is ``None`` when unset. A dead field is
not inert, and the lowercase one was worse than dead: its ``model_config`` is
``case_sensitive: False``, so it *was* being populated from the
``DEFAULT_TENANT_ID`` env var. A live-looking, unread tenant.

``"n8n-dev"`` is not junk and it is not a real tenant_id either, which is exactly
what made it dangerous. It **was** a real tenant, as a text tenant_id, until the
2026-04-29 Keycloak migration moved tenant identity to uuid7 and demoted the
label to a ``tenant_name`` attribute. So the fallback pointed at the *stale label
form* of a tenant that really exists: an unset ``DEFAULT_TENANT_ID`` would not
have 401'd and would not have crashed — coding-realtime would have **vouched,
with a valid service credential, for a tenant nobody chose**.

coding-client removed the identical literal from the SDK in 1.9.0 and says so in
``coding_client.auth.outbound``'s module docstring. That fix reached this repo's
*imports* and not its *settings*, because this repo defined its own copies.

An AST sweep over 1143 files found the literal at **six** sites in five repos.
The two LIVE ones were ``coding-realtime`` and ``coding-gateway``, which read it
into outbound ``X-Tenant-Id`` headers sent alongside their own service token —
i.e. a vouched call for a tenant nobody chose. This repo's two are dead, and are
removed here so the next reader cannot revive them.

WHY THE SUITE DID NOT CATCH IT
------------------------------
``tests/unit/test_settings.py`` asserted ``s.DEFAULT_TENANT_ID == "n8n-dev"``.
It did not merely fail to catch the defect — it **pinned the defect as correct**,
so it would have gone red on the fix and stayed green on the bug. That assertion
is inverted, not deleted, so a revert is caught. This file adds the rule those
tests cannot express: the literal must not return under any name, in any module.

MUTANTS — each was run against this file
----------------------------------------
    M1  restore either settings field with the literal               -> FAILS
    M2  reintroduce it under another name, same literal              -> FAILS
    M3  put the literal in a docstring/comment only                  -> passes
        (deliberate: prose explaining the removal must stay legal)
    M4  point the scan root at an empty directory                    -> the
        non-vacuity test FAILS, not the rules
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

# Resolve the scan root through an IMPORTED module, never a path relative to
# this file. A relative path is the classic way a guard ends up scanning an
# empty directory and passing vacuously (M4).
_MODULE_FILE = importlib.import_module("config.settings").__file__
assert _MODULE_FILE is not None, "config.settings has no __file__"
SRC = pathlib.Path(_MODULE_FILE).parent.parent

BANNED_TENANT_LITERAL = "n8n-dev"


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """``id()`` of every docstring Constant — module, class and function.

    Docstrings are prose, not values the code can hand out, so they are exempt
    from the literal rule. They have to be excluded by IDENTITY rather than by
    "is the first statement", because ``ast.walk`` yields them flattened and a
    positional test would also exempt a real assignment that happens to be
    first in its block.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and (doc := ast.get_docstring(node, clean=False)) is not None
        ):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                assert first.value.value == doc  # the node we exempt IS the docstring
                out.add(id(first.value))
    return out


@pytest.fixture(scope="module")
def parsed() -> list[tuple[pathlib.Path, ast.Module]]:
    """Every ``src/`` module, parsed once."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return out


def test_the_scan_is_not_vacuous(parsed):
    """Prove the walk found this service's real source before trusting a pass.

    Without this, deleting or mistyping ``SRC`` turns every rule below into a
    green that measured nothing.
    """
    files = {p for p, _ in parsed}
    assert len(files) >= 300, f"only {len(files)} source files found under {SRC}"
    names = {p.name for p in files}
    assert "settings.py" in names, f"settings.py not in the scanned set under {SRC}"
    assert "core_config_settings.py" in names, (
        f"core_config_settings.py not in the scanned set under {SRC} — the second "
        "settings class is exactly the one a shallow scan misses"
    )

    # Positive control that the AST walk actually parses string constants: a
    # string we MEASURED to be in the tree (7 occurrences) must be found by the
    # same traversal the rule uses. Measured, not guessed: the first draft of
    # this control named a string that the fix itself had just removed, so the
    # control could never hold. It failed loudly, which is the point of it.
    found_known = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "X-Tenant-Id" in node.value
        for _, tree in parsed
        for node in ast.walk(tree)
    )
    assert found_known, "the AST walk found no known string constant — it is not seeing code"


def test_the_n8n_dev_tenant_literal_is_absent_from_src(parsed):
    """No ``src/`` module may carry ``"n8n-dev"`` as a string constant.

    Keyed on the LITERAL, not on the variable name, so reintroducing it under
    any other name (``FALLBACK_TENANT``, ``TENANT_DEFAULT``, a pydantic field
    default) is still caught. Comments and docstrings are invisible to the AST
    walk by design — prose explaining why the literal was removed must remain
    legal, and this docstring is itself the standing false-positive control.
    """
    offenders = []
    for path, tree in parsed:
        exempt = _docstring_nodes(tree)
        offenders += [
            f"{path.relative_to(SRC)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and BANNED_TENANT_LITERAL in node.value
            and id(node) not in exempt
        ]
    assert not offenders, (
        f"the {BANNED_TENANT_LITERAL!r} tenant literal is back at: {offenders}. "
        "It is the stale pre-uuid7 label of a real tenant, so it fails SILENTLY: "
        "calls carry a valid service token while asserting a tenant nobody chose. "
        "Read the tenant from coding_client.auth.tenant.DEFAULT_TENANT_ID (None "
        "when unset) and refuse via outbound.tenant_headers instead."
    )


def test_neither_settings_class_has_a_default_tenant_field():
    """Neither settings class may re-grow a tenant default.

    Complements the literal rule: this catches the field coming back with a
    *different* (e.g. uuid-shaped) default, which the literal rule cannot see
    and which is the same defect — a tenant chosen by the code rather than by
    configuration. Both classes are checked because having two was how one of
    them stayed unnoticed.
    """
    from app.core.config.core_config_settings import Settings as CoreSettings
    from config.settings import Settings

    assert "DEFAULT_TENANT_ID" not in Settings.model_fields, (
        "Settings.DEFAULT_TENANT_ID is back. The tenant must come from "
        "coding_client.auth.tenant.DEFAULT_TENANT_ID, which is None when unset, "
        "so a misconfigured deploy refuses instead of guessing."
    )
    assert "default_tenant_id" not in CoreSettings.model_fields, (
        "core_config_settings.Settings.default_tenant_id is back. This class is "
        "case-insensitive, so the field is populated from DEFAULT_TENANT_ID "
        "whether or not anything reads it."
    )
