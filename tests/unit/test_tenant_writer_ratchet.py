"""No NEW writer may insert a ``nodes``/``edges`` row it cannot attribute.

WHAT WAS WRONG
--------------
``tenant_id`` became a real ``uuid`` in migration 0005, and 0006 attributed every
untenanted row whose graph names an owner. Neither closes the source: rows with no
tenant keep arriving, because some writers do not put ``tenant_id`` in the INSERT
column list at all. Measured on prod ``coding_analysis`` @ worker4 after 0005:
**8,901 rows** across ``nodes`` (7,759), ``edges`` (1,139) and
``dependency_report_snapshots`` (3) held no tenant.

That is not a cosmetic gap. Reads in this service are NULL-TOLERANT on purpose —
``graph_adapter._get_all_entities`` / ``._get_all_relationships`` and
``graph_builder.graph_is_populated`` all match ``tenant_id = %s OR tenant_id IS
NULL`` (PR #54, so the ``architecture_refactoring`` cycle-detection arc did not
regress on legacy graphs), and every ``code_graph_id``-only read in
``graphs_reader.py`` does not mention ``tenant_id`` at all. So an unattributed row
is not hidden from tenants, it is served to **all** of them. NULL is acting as a
wildcard, which is the "all tenants living in the identity column" shape this
estate has been burned by before.

WHY THE OFFENDERS ARE NOT SIMPLY FIXED
--------------------------------------
Because they have no tenant to write. Both live on the task path — the processors
registered as ``analyze_codebase`` and ``multi_language_analysis`` in
``src/processing/task_processors.py`` — and the task payload carries no tenant
anywhere: ``grep -c tenant src/processing/task_processors.py`` is **0**, likewise
``src/processing/codebase_analysis_processor.py``. Threading one through is an
interface change to the task system, i.e. an owner decision, not a side effect of
attributing existing rows.

So this file does what CAN be done today: it FLOORS the damage. The four
tenant-blind statements below are pinned with their reason. A fifth cannot be added
without this test going red, and a fixed one cannot be left in the list.

WHY A RATCHET AND NOT A BAN
---------------------------
A plain "every INSERT must carry tenant_id" rule would be red on ``main`` from the
moment it lands, so it would be skipped, xfailed or deleted within a week. A
ratchet that floors on the offenders actually measured is the version that survives
— provided it cannot pass vacuously, which is what
:func:`test_the_sweep_is_not_vacuous` and the exact-match assertion below are for.
``assert len(blind) <= 4`` would be satisfied by a sweep that found nothing at all.

MUTANTS — each was run against this file
----------------------------------------
    M1  drop ``tenant_id`` from a currently-OK INSERT column list   -> FAILS
    M2  add a new tenant-blind ``INSERT INTO nodes`` in a new file  -> FAILS
    M3  fix an allowlisted writer, leave it in the allowlist        -> FAILS
        (the ratchet must tighten, not just hold)
    M4  point ``SRC`` at an empty directory                         -> the
        non-vacuity test FAILS, not the rules
    M5  break the column-list regex so nothing matches              -> the
        non-vacuity test FAILS
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re
from typing import NamedTuple

import pytest

# Resolve the scan root through an IMPORTED module, never a path relative to this
# file. A relative path is the classic way a guard ends up scanning an empty
# directory and passing vacuously (M4).
_MODULE_FILE = importlib.import_module("config.settings").__file__
assert _MODULE_FILE is not None, "config.settings has no __file__"
SRC = pathlib.Path(_MODULE_FILE).parent.parent

#: ``INSERT INTO <table> (<columns>)`` over the two tenant-scoped graph tables.
#: Matched against string CONSTANTS from the AST rather than raw file text, so a
#: table name inside a comment or a docstring cannot register as a writer.
INSERT_RE = re.compile(r"INSERT\s+INTO\s+(nodes|edges)\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)

#: Total ``INSERT INTO nodes|edges`` statements in ``src/``, MEASURED on the commit
#: that added this file. A floor, not an equality: adding a properly attributed
#: writer is fine and must not fail. Its job is to catch a sweep that stopped
#: seeing code (M5).
KNOWN_INSERT_SITES = 14


class BlindWriter(NamedTuple):
    """One INSERT statement that omits ``tenant_id`` from its column list."""

    module: str
    table: str
    #: Why it cannot be fixed here. Not decoration — an entry whose reason has
    #: expired is an entry that should have been removed.
    reason: str


#: The COMPLETE set of tenant-blind graph writers, measured on prod's ``origin/main``
#: at revision 0005. Exact-match, checked in BOTH directions:
#:
#:   * a writer here that no longer exists       -> remove it (the ratchet tightens)
#:   * a tenant-blind writer not here            -> fix it, do not extend this list
#:
#: Extending this list is a deliberate act that has to survive review. That is the
#: entire mechanism.
KNOWN_TENANT_BLIND_WRITERS: frozenset[BlindWriter] = frozenset(
    {
        BlindWriter(
            "processing/task_result_processor.py",
            "nodes",
            "Task path (analyze_codebase / multi_language_analysis). Writes real "
            "file/function/class/method entities, not only the 'internal metadata' "
            "its module comment claims. No tenant exists in the task payload.",
        ),
        BlindWriter(
            "processing/task_result_processor.py",
            "edges",
            "Same statement pair as the nodes upsert above; same missing tenant.",
        ),
        BlindWriter(
            "services/code_graph/graph_database_populator.py",
            "nodes",
            "Legacy twin of analysis_utils.populate_graph_database, which does the "
            "same job WITH a tenant. Imported only by examples/ and tests on "
            "origin/main — dead in production, but still exported and callable.",
        ),
        BlindWriter(
            "services/code_graph/graph_database_populator.py",
            "edges",
            "Same module as above; same missing tenant.",
        ),
    }
)


@pytest.fixture(scope="module")
def insert_sites() -> list[tuple[str, str, tuple[str, ...]]]:
    """``(module, table, columns)`` for every graph INSERT reachable from ``src/``."""
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.relative_to(SRC).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            for match in INSERT_RE.finditer(node.value):
                columns = tuple(col.strip() for col in match.group(2).replace("\n", " ").split(","))
                out.append((module, match.group(1).lower(), columns))
    return out


def test_the_sweep_is_not_vacuous(insert_sites):
    """Prove the sweep found this service's real writers before trusting a pass.

    Every rule below is an assertion about a SET. A sweep that silently found
    nothing satisfies all of them, so the sweep is measured first (M4, M5).
    """
    assert len(insert_sites) >= KNOWN_INSERT_SITES, (
        f"found only {len(insert_sites)} graph INSERT statements under {SRC}, "
        f"expected at least {KNOWN_INSERT_SITES} — the sweep is not seeing code"
    )

    # Positive control on the PARSING, not just the file walk: the ratchet's whole
    # signal is the column list, so a known-good column list must come back intact.
    # Without this, a regex that matched the statement but captured an empty group
    # would report every writer as tenant-blind and every allowlist entry as stale.
    reader = [
        cols
        for module, table, cols in insert_sites
        if module == "app/api/endpoints/graphs_reader.py" and table == "nodes"
    ]
    assert reader, "graphs_reader.py's repository-anchor INSERT was not found"
    assert "tenant_id" in reader[0], (
        "the one writer that CANNOT produce a NULL (initialize_graph, behind "
        "Depends(require_tenant)) parsed without tenant_id — the column list is "
        "not being captured"
    )

    tables = {table for _, table, _ in insert_sites}
    assert tables == {"nodes", "edges"}, f"unexpected tables in the sweep: {tables}"


def test_no_new_tenant_blind_graph_writer(insert_sites):
    """The set of writers that cannot attribute their rows must not grow.

    An unattributed row is readable by every tenant on this service's NULL-tolerant
    read paths, so a new tenant-blind writer is a cross-tenant exposure, not a
    style question.
    """
    blind = {
        (module, table) for module, table, columns in insert_sites if "tenant_id" not in columns
    }
    allowed = {(w.module, w.table) for w in KNOWN_TENANT_BLIND_WRITERS}

    new = blind - allowed
    assert not new, (
        "new graph writer(s) omit tenant_id from the INSERT column list, so every "
        f"row they write is readable by every tenant: {sorted(new)}. Put the tenant "
        "in the column list; do not add it to KNOWN_TENANT_BLIND_WRITERS."
    )


def test_the_ratchet_tightens_when_a_writer_is_fixed(insert_sites):
    """A fixed writer must be REMOVED from the allowlist, not left in it.

    Without this direction the list only ever grows stale, and a future reader
    inherits four documented "known offenders" that were repaired long ago — an
    exclusion whose reason has expired is where the next real one hides (M3).
    """
    blind = {
        (module, table) for module, table, columns in insert_sites if "tenant_id" not in columns
    }
    allowed = {(w.module, w.table) for w in KNOWN_TENANT_BLIND_WRITERS}

    stale = allowed - blind
    assert not stale, (
        f"{sorted(stale)} now attribute their rows — delete them from "
        "KNOWN_TENANT_BLIND_WRITERS so the floor drops with the fix."
    )


def test_every_allowlisted_writer_carries_a_reason():
    """An exclusion with no recorded reason is an exclusion nobody can retire."""
    for writer in KNOWN_TENANT_BLIND_WRITERS:
        assert writer.reason.strip(), f"{writer.module}:{writer.table} has no reason"
        assert len(writer.reason) > 40, (
            f"{writer.module}:{writer.table} has a reason too short to act on: {writer.reason!r}"
        )
