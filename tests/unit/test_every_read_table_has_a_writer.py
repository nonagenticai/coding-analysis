"""Guard: every table this service READS must have a WRITER (coding-analysis#106).

Why this exists: the ``/quality/dashboard`` router read ``quality_analysis_results``,
``quality_issues``, ``tool_configurations``, ``languages`` and
``project_languages`` — tables that no code wrote and that did not even exist in
the ``coding_analysis`` database. Every read therefore degraded to an empty (or
500) response that looked like "no data yet" forever, and nothing noticed. A
reader without a writer is dead code that answers.

The scan is static and deliberately simple: every string literal under ``src/``
that looks like SQL (an upper-case ``SELECT``/``INSERT``/``UPDATE``/``DELETE``/
``WITH`` keyword) contributes

* READS  — identifiers after ``FROM`` / ``JOIN``;
* WRITES — identifiers after ``INSERT INTO`` / ``UPDATE`` / ``MERGE INTO`` /
  ``COPY … FROM`` (an ``… ON CONFLICT`` upsert is an ``INSERT INTO``).

``DELETE FROM x`` is neither: deleting rows creates none. CTE names, function
calls (``FROM unnest(…)``), sub-selects and system catalogs are excluded.

A read table with no writer fails the test unless it is in ``ALLOWLIST`` with a
one-line reason (written by an alembic data migration, or by another service).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: Read-without-writer tables that are legitimate, each with its reason.
ALLOWLIST: dict[str, str] = {
    "users": (
        "not a query this service runs: it only appears inside sample source text "
        "that processing/analyzers/{test_discovery,integration_tests}.py emit as "
        "SQL-injection fixtures"
    ),
}

_SQL_HINT = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b")
_NOISE = [
    # EXTRACT(EPOCH FROM ts), SUBSTRING(x FROM 2), TRIM(BOTH ' ' FROM x)
    re.compile(r"\b(EXTRACT|SUBSTRING|TRIM|OVERLAY|POSITION)\s*\([^()]*\)", re.IGNORECASE),
    # a IS [NOT] DISTINCT FROM b
    re.compile(r"\bDISTINCT\s+FROM\b"),
]
_IDENT = r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
_READ = re.compile(r"\b(?:FROM|JOIN)\s+" + _IDENT + r"(\s*\()?")
_DELETE = re.compile(r"\bDELETE\s+FROM\s+" + _IDENT)
_WRITE = [
    # INSERT INTO, and SQLite's INSERT OR REPLACE / OR IGNORE INTO
    re.compile(r"\bINSERT\s+(?:OR\s+[A-Z]+\s+)?INTO\s+" + _IDENT),
    re.compile(r"\bUPDATE\s+" + _IDENT + r"\s+(?:AS\s+\w+\s+)?SET\b"),
    re.compile(r"\bUPDATE\s+" + _IDENT + r"\s+\w+\s+SET\b"),
    re.compile(r"\bMERGE\s+INTO\s+" + _IDENT),
    re.compile(r"\bCOPY\s+" + _IDENT + r"\s*(?:\([^)]*\))?\s+FROM\b"),
]
_CTE = re.compile(r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(")
_SYSTEM = re.compile(r"^(pg_|information_schema\b)")
_KEYWORDS = frozenset({"SELECT", "LATERAL", "ONLY", "VALUES"})


def _string_literals(source: str) -> list[str]:
    """Every string literal in a module; f-strings contribute their constant parts."""
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append(
                "".join(
                    v.value if isinstance(v, ast.Constant) else " __expr__ " for v in node.values
                )
            )
    return out


def scan_sql(source: str) -> tuple[set[str], set[str]]:
    """Return ``(reads, writes)`` table names referenced by SQL literals in ``source``."""
    reads: set[str] = set()
    writes: set[str] = set()
    for literal in _string_literals(source):
        if not _SQL_HINT.search(literal):
            continue
        text = literal
        for pattern in _NOISE:
            text = pattern.sub(" ", text)
        ctes = {m.group(1) for m in _CTE.finditer(text)}
        deleted = {m.start(1) for m in _DELETE.finditer(text)}
        for pattern in _WRITE:
            for m in pattern.finditer(text):
                writes.add(m.group(1).lower())
        for m in _READ.finditer(text):
            name = m.group(1)
            if m.start(1) in deleted or m.group(2):  # DELETE FROM x / FROM fn(...)
                continue
            if name.upper() in _KEYWORDS or name in ctes or _SYSTEM.match(name):
                continue
            if name == "__expr__":
                continue
            reads.add(name.lower())
    return reads, writes


def scan_tree(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """``(table -> files that read it, tables written anywhere)`` for ``root/**/*.py``."""
    readers: dict[str, set[str]] = {}
    writes: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        r, w = scan_sql(path.read_text(encoding="utf-8"))
        for table in r:
            readers.setdefault(table, set()).add(str(path.relative_to(root)))
        writes |= w
    return readers, writes


def unwritten_reads(root: Path, allowlist: dict[str, str]) -> dict[str, set[str]]:
    readers, writes = scan_tree(root)
    return {t: files for t, files in readers.items() if t not in writes and t not in allowlist}


# ---------------------------------------------------------------------------
# negative controls — the guard must be able to fail
# ---------------------------------------------------------------------------


def test_negative_control_flags_a_read_with_no_writer(tmp_path):
    (tmp_path / "reader.py").write_text(
        'Q = "SELECT a FROM ghost_table g JOIN real_table r ON r.id = g.id"\n'
        'W = "INSERT INTO real_table (id) VALUES (%s) ON CONFLICT (id) DO NOTHING"\n',
        encoding="utf-8",
    )
    assert unwritten_reads(tmp_path, {}) == {"ghost_table": {"reader.py"}}


def test_negative_control_delete_is_not_a_writer(tmp_path):
    (tmp_path / "m.py").write_text(
        'A = "SELECT * FROM doomed"\nB = "DELETE FROM doomed WHERE id = %s"\n', encoding="utf-8"
    )
    assert set(unwritten_reads(tmp_path, {})) == {"doomed"}


def test_scanner_ignores_ctes_functions_catalogs_and_prose():
    source = '''
DOC = """This reads from the repo and joins from nothing."""
Q = """
WITH candidate AS (SELECT id FROM jobs WHERE status = 'queued')
UPDATE jobs AS j SET status = 'running' FROM candidate WHERE j.id = candidate.id
"""
R = "SELECT EXTRACT(EPOCH FROM created_at), x FROM unnest(%s) AS u(x)"
S = "SELECT 1 FROM pg_catalog.pg_class WHERE a IS DISTINCT FROM b"
T = f"SELECT count(*) FROM {table}"
'''
    reads, writes = scan_sql(source)
    assert reads == {"jobs"}
    assert writes == {"jobs"}


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_every_read_table_has_a_writer():
    orphans = unwritten_reads(SRC, ALLOWLIST)
    assert not orphans, "tables read by src/ with no writer anywhere in src/:\n" + "\n".join(
        f"  {table}: read in {sorted(files)}" for table, files in sorted(orphans.items())
    )


def test_quality_results_is_both_read_and_written():
    readers, writes = scan_tree(SRC)
    assert "quality_results" in readers, "nothing reads quality_results"
    assert "quality_results" in writes, "nothing writes quality_results"


@pytest.mark.parametrize("table", sorted(ALLOWLIST))
def test_allowlist_entries_are_still_read(table):
    """A stale allowlist entry is a licence to not look — drop it once unread."""
    readers, _ = scan_tree(SRC)
    assert table in readers, f"{table} is allowlisted but no longer read; remove it"
