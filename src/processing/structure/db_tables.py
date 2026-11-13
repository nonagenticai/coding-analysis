"""Database table inventory: which file defines, alters, reads and writes which table.

Python (``ast``)
    * SQLAlchemy: ``__tablename__ = "x"`` / ``__tablename__: str = "x"`` (``orm_model``), ``Table("x", ...)``
      (``sqlalchemy_table``); ORM use of a mapped class: ``select(M)``,
      ``session.query(M)``, ``session.get(M, ..)`` read; ``insert(M)``,
      ``update(M)``, ``delete(M)``, ``session.add(M(...))``/``add_all``/``merge``
      write (the class -> table map is built across all files first).
    * Alembic ``op.*``: ``create_table`` defines; ``add_column``, ``drop_column``,
      ``alter_column``, ``drop_table``, ``rename_table``, ``create_index``,
      ``create_foreign_key``, ``create_unique_constraint``, ``batch_alter_table``,
      ... alter.
    * Raw SQL: the string argument (literal or f-string) of ``execute`` /
      ``executemany`` / ``exec_driver_sql`` / ``text`` calls (``conn.execute``,
      ``op.execute``, ``session.execute(text(...))``).
Rust (tree-sitter)
    * The string literals inside ``sqlx::query!``/``query_as!``/``query_scalar!``
      or ``sqlx::query(..)`` (tagged ``sqlx``) and inside ``.execute(..)`` calls.
SQL files
    * Every ``*.sql`` file (``migration`` when under a ``migrations`` directory).

Only those strings are classified: a string elsewhere (a log message, a
docstring) is never read as SQL. Classification (linear regexes): ``CREATE
TABLE`` defines; ``ALTER TABLE``, ``DROP TABLE``, ``CREATE INDEX .. ON`` alter;
``INSERT INTO``, ``UPDATE .. SET``, ``DELETE FROM``, ``TRUNCATE``, ``MERGE INTO``
write; ``FROM``/``JOIN`` read. Table names are lowercased and unquoted; a schema
qualifier (``billing.invoices``, ``public.users``) is split off into ``schema``
on the reference (and ``schemas`` on the table), so a table is one entry whatever
schema it is named with. A table name built from an interpolation
(``f"SELECT * FROM {t}"``) is unresolved, never guessed.

System catalogs are not tables of the repo and are dropped (see
:func:`is_system_catalog`): introspection code (``FROM pg_catalog.pg_class c JOIN
pg_attribute a``, ``FROM information_schema.tables``) would otherwise report
``pg_class``/``pg_attribute`` and -- because the schema is split off before the
table is keyed -- a bogus table named ``tables``.
"""

from __future__ import annotations

import ast
import bisect
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .text import clip

_NAME = r'((?:"[^"\n]{1,128}"|`[^`\n]{1,128}`|\[[^\]\n]{1,128}\]|[A-Za-z_][\w$]{0,127}|\x00)(?:\.(?:"[^"\n]{1,128}"|[A-Za-z_][\w$]{0,127}|\x00))?)'
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "defined_in",
        re.compile(
            r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:GLOBAL|LOCAL)\s+)?(?:UNLOGGED\s+|TEMP(?:ORARY)?\s+)?"
            rf"TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_NAME}",
            re.I,
        ),
    ),
    ("altered_in", re.compile(rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_NAME}", re.I)),
    ("altered_in", re.compile(rf"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?{_NAME}", re.I)),
    (
        "altered_in",
        re.compile(
            r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
            rf"(?:[\w\"]+\s+)?ON\s+(?:ONLY\s+)?{_NAME}",
            re.I,
        ),
    ),
    ("written_by", re.compile(rf"\bINSERT\s+INTO\s+{_NAME}", re.I)),
    (
        "written_by",
        re.compile(rf"\bUPDATE\s+(?:ONLY\s+)?{_NAME}\s+(?:AS\s+\w+\s+|\w+\s+)?SET\b", re.I),
    ),
    ("written_by", re.compile(rf"\bDELETE\s+FROM\s+(?:ONLY\s+)?{_NAME}", re.I)),
    ("written_by", re.compile(rf"\bTRUNCATE\s+(?:TABLE\s+)?(?:ONLY\s+)?{_NAME}", re.I)),
    ("written_by", re.compile(rf"\bMERGE\s+INTO\s+{_NAME}", re.I)),
    ("read_by", re.compile(rf"(?<!DELETE )\bFROM\s+(?:ONLY\s+)?{_NAME}(?!\s*\()", re.I)),
    ("read_by", re.compile(rf"\bJOIN\s+(?:LATERAL\s+)?{_NAME}(?!\s*\()", re.I)),
]
_KEYWORDS = {
    "select",
    "where",
    "set",
    "values",
    "lateral",
    "only",
    "unnest",
    "jsonb_to_recordset",
    "json_to_recordset",
    "generate_series",
}
_HOLE = "\x00"
#: Schemas that hold only catalogs/views of the database itself. Postgres reserves
#: every ``pg_`` *schema* name (``pg_catalog``, ``pg_toast``, ``pg_temp_N``), so the
#: prefix covers them all; ``information_schema`` is the SQL-standard catalog
#: (Postgres, MySQL, SQL Server).
_SYSTEM_SCHEMAS = frozenset({"information_schema"})
_SQL_COMMENT = re.compile(r"--[^\n]*")
_ALEMBIC_DEFINE = {"create_table"}
_ALEMBIC_ALTER_FIRST = {
    "add_column",
    "drop_column",
    "alter_column",
    "drop_table",
    "rename_table",
    "batch_alter_table",
    "drop_constraint",
    "create_check_constraint",
    "create_primary_key",
    "bulk_insert",
}
_ALEMBIC_ALTER_SECOND = {
    "create_index",
    "drop_index",
    "create_unique_constraint",
    "create_foreign_key",
}
_ORM_READ = {"select", "query", "get", "scalars", "get_one"}
_ORM_WRITE = {"insert", "update", "delete"}
_SESSION_WRITE = {"add", "add_all", "merge"}
#: Python calls whose first (string) argument is SQL.
_SQL_CALLS = {"execute", "executemany", "exec_driver_sql", "text"}
_SQLX = {"query", "query_as", "query_scalar", "query_unchecked", "query_as_unchecked"}
#: Longest call target text read when looking for an sqlx query function.
_MAX_CALLEE = 300


@dataclass
class TableRef:
    file: str
    line: int
    via: str
    statement: str | None = None
    #: The schema the reference names (``billing`` for ``billing.invoices``), None
    #: when unqualified.
    schema: str | None = None


@dataclass
class TableFacts:
    table: str  # lowercased, unquoted, WITHOUT its schema
    #: Every schema the table is referenced under (refs carry their own).
    schemas: list[str] = field(default_factory=list)
    defined_in: list[TableRef] = field(default_factory=list)
    altered_in: list[TableRef] = field(default_factory=list)
    read_by: list[TableRef] = field(default_factory=list)
    written_by: list[TableRef] = field(default_factory=list)


@dataclass
class UnresolvedTable:
    file: str
    line: int
    reason: str
    expression: str


@dataclass
class TableInventory:
    tables: list[TableFacts]
    unresolved: list[UnresolvedTable]
    files_unparsed: dict[str, str]


class _Collector:
    def __init__(self) -> None:
        self.tables: dict[str, TableFacts] = {}
        self.unresolved: list[UnresolvedTable] = []
        self.unparsed: dict[str, str] = {}
        self._seen: set[tuple[str, str, str, int, str]] = set()

    def add(self, table: str, role: str, file: str, line: int, via: str, stmt: str | None = None):
        schema, name = split_table(table)
        if not name or name in _KEYWORDS or is_system_catalog(schema, name):
            return
        key = (name, role, file, line, via)
        if key in self._seen:
            return
        self._seen.add(key)
        facts = self.tables.setdefault(name, TableFacts(name))
        if schema and schema not in facts.schemas:
            facts.schemas.append(schema)
        getattr(facts, role).append(TableRef(file, line, via, stmt, schema))

    def sql(self, text: str, file: str, line: int, via: str):
        text = _SQL_COMMENT.sub("", text)
        # Computed once per text: counting newlines from the start per match was
        # O(matches x text).
        newlines = [i for i, ch in enumerate(text) if ch == "\n"]
        for role, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1)
                at = line + bisect.bisect_left(newlines, m.start())
                if _HOLE in name:
                    self.unresolved.append(
                        UnresolvedTable(
                            file, at, "table name is interpolated", _clip(text[m.start() : m.end()])
                        )
                    )
                    continue
                self.add(name, role, file, at, via, _clip(text[m.start() : m.end()]))


def _clip(text: str) -> str:
    return clip(text.replace(_HOLE, "{…}"), 120)


def split_table(name: str) -> tuple[str | None, str]:
    """``(schema, table)`` of a written name, lowercased and unquoted.

    ``billing.invoices`` -> (billing, invoices); ``invoices`` -> (None, invoices);
    ``"Public"."T"`` -> (public, t). The table is keyed by its bare name, so
    ``billing.invoices`` and ``invoices`` are one table whose refs keep their schema.
    """
    parts = [p.strip('"`[]').lower() for p in name.split(".")]
    parts = [p for p in parts if p]
    if not parts:
        return None, ""
    return (parts[-2] if len(parts) > 1 else None), parts[-1]


def is_system_catalog(schema: str | None, name: str) -> bool:
    """Whether ``schema.name`` (as split by :func:`split_table`) is a system catalog.

    * Qualified: the schema is ``information_schema`` or any ``pg_*`` schema
      (Postgres reserves the prefix for schemas, so no user schema can have it).
      ``information_schema.tables`` is therefore never the table ``tables``.
    * Unqualified: the name starts with ``pg_``. Postgres always searches
      ``pg_catalog`` first, so an unqualified catalog name (``pg_class``) resolves to
      the catalog whatever the search_path; the prefix rather than a list of
      catalog names also covers the ``pg_stat_*`` views and extension views
      (``pg_stat_statements``) and does not rot across Postgres versions. Postgres
      does NOT reserve ``pg_`` for table names, so an unqualified user table named
      ``pg_*`` is missed -- a deliberate trade (Postgres' own docs advise against
      such names); qualify it (``public.pg_x``) and it is kept.
    """
    if schema is not None:
        return schema in _SYSTEM_SCHEMAS or schema.startswith("pg_")
    return name.startswith("pg_")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _str_value(node: ast.AST | None) -> str | None:
    """A string literal, or an f-string with every hole replaced by a marker."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append(_HOLE)
        return "".join(out)
    return None


def _call_name(call: ast.Call) -> tuple[str, str | None]:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        recv = fn.value
        recv_name = (
            recv.id
            if isinstance(recv, ast.Name)
            else recv.attr
            if isinstance(recv, ast.Attribute)
            else None
        )
        return fn.attr, recv_name
    if isinstance(fn, ast.Name):
        return fn.id, None
    return "", None


def _model_tables(trees: dict[str, ast.Module]) -> dict[str, str]:
    """Mapped class name -> table, across files (unique names only)."""
    found: dict[str, list[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (
                    _is_tablename(stmt)
                    and (value := _str_value(_assigned(stmt))) is not None
                    and _HOLE not in value
                ):
                    found.setdefault(node.name, []).append(value)
    return {k: v[0] for k, v in found.items() if len(set(v)) == 1}


def _is_tablename(stmt: ast.stmt) -> bool:
    """``__tablename__ = ...`` or the annotated ``__tablename__: str = ...``."""
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in stmt.targets)
    return (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == "__tablename__"
        and stmt.value is not None
    )


def _assigned(stmt: ast.stmt) -> ast.expr | None:
    return stmt.value if isinstance(stmt, ast.Assign | ast.AnnAssign) else None


def _python(path: str, tree: ast.Module, models: dict[str, str], out: _Collector) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if _is_tablename(stmt):
                    value = _str_value(_assigned(stmt))
                    if value is None or _HOLE in value:
                        out.unresolved.append(
                            UnresolvedTable(
                                path,
                                stmt.lineno,
                                "__tablename__ is computed",
                                ast.unparse(stmt)[:120],
                            )
                        )
                    else:
                        out.add(value, "defined_in", path, stmt.lineno, "orm_model", node.name)
        if not isinstance(node, ast.Call):
            continue
        name, recv = _call_name(node)
        args = node.args
        first = _str_value(args[0]) if args else None
        if name == "Table" and args:
            if first is None or _HOLE in first:
                out.unresolved.append(
                    UnresolvedTable(
                        path, node.lineno, "Table() name is computed", ast.unparse(node)[:120]
                    )
                )
            else:
                out.add(first, "defined_in", path, node.lineno, "sqlalchemy_table")
            continue
        if recv == "op":
            idx = (
                0
                if name in _ALEMBIC_DEFINE | _ALEMBIC_ALTER_FIRST
                else 1
                if name in _ALEMBIC_ALTER_SECOND
                else None
            )
            if name == "create_foreign_key" and len(args) > 2:
                ref = _str_value(args[2])
                if ref is not None and _HOLE not in ref:
                    out.add(
                        ref, "altered_in", path, node.lineno, "alembic:create_foreign_key(referent)"
                    )
            if idx is not None:
                table_arg = args[idx] if len(args) > idx else None
                if table_arg is None:
                    kw = {k.arg: k.value for k in node.keywords}
                    table_arg = kw.get("table_name") or kw.get("source_table") or kw.get("table")
                value = _str_value(table_arg)
                if value is None or _HOLE in value:
                    out.unresolved.append(
                        UnresolvedTable(
                            path,
                            node.lineno,
                            f"op.{name} table is computed",
                            ast.unparse(node)[:120],
                        )
                    )
                else:
                    role = "defined_in" if name in _ALEMBIC_DEFINE else "altered_in"
                    out.add(value, role, path, node.lineno, f"alembic:{name}")
                    if name == "rename_table" and len(args) > 1 and (new := _str_value(args[1])):
                        out.add(new, "defined_in", path, node.lineno, "alembic:rename_table")
                continue
        mapped = [models[a.id] for a in args if isinstance(a, ast.Name) and a.id in models]
        if name in _ORM_READ and mapped:
            for table in mapped:
                out.add(table, "read_by", path, node.lineno, f"orm:{name}")
        elif name in _ORM_WRITE and mapped and recv in (None, "sa", "sqlalchemy"):
            for table in mapped:
                out.add(table, "written_by", path, node.lineno, f"orm:{name}")
        elif name in _SESSION_WRITE and args:
            inner = args[0].elts if isinstance(args[0], ast.List | ast.Tuple) else [args[0]]
            for item in inner:
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id in models
                ):
                    out.add(
                        models[item.func.id], "written_by", path, node.lineno, f"orm:session.{name}"
                    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and _call_name(node)[0] in _SQL_CALLS:
            text = _str_value(node.args[0])
            if text is not None:
                out.sql(text, path, node.args[0].lineno, "sql")


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def _rust(path: str, source: str, out: _Collector) -> None:
    from processing.contract_analysis.rust_http import string_value

    tree = _parse("rust", source.encode("utf-8"))
    if tree is None:
        out.unparsed[path] = "grammar not available"
        return
    stack: list[tuple[Any, str | None]] = [(tree.root_node, None)]
    while stack:
        node, via = stack.pop()
        if node.type == "macro_invocation":
            name = node.child_by_field_name("macro")
            text = name.text.decode() if name is not None and name.text else ""
            if text.rsplit("::", 1)[-1] in _SQLX:
                via = "sqlx"
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            # Only a short callee can name sqlx (``sqlx::query::<T>``); a method
            # call's function text is its whole receiver chain, and reading it for
            # every call of a long chain was O(chain^2).
            short = fn is not None and fn.end_byte - fn.start_byte <= _MAX_CALLEE
            text = fn.text.decode() if short and fn.text else ""
            base = text.split("::<", 1)[0]
            if base.rsplit("::", 1)[-1] in _SQLX and ("sqlx" in base or "::" not in base):
                via = "sqlx"
            elif fn is not None and fn.type == "field_expression":
                field_name = fn.child_by_field_name("field")
                if field_name is not None and field_name.text == b"execute":
                    via = via or "sql"
        value = string_value(node)
        if value is not None and via is not None:
            out.sql(value, path, node.start_point[0] + 1, via)
            continue
        stack.extend((c, via) for c in node.named_children)


def table_inventory(files: dict[str, str]) -> TableInventory:
    """Tables of *files* (``{path: source}``: ``.py``, ``.rs`` and ``.sql``)."""
    out = _Collector()
    trees: dict[str, ast.Module] = {}
    for path in sorted(files):
        if path.endswith(".py"):
            try:
                trees[path] = ast.parse(files[path], filename=path)
            except (SyntaxError, ValueError, RecursionError) as exc:
                out.unparsed[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    models = _model_tables(trees)
    for path, tree in trees.items():
        _python(path, tree, models, out)
    for path in sorted(files):
        if path.endswith(".rs"):
            _rust(path, files[path], out)
        elif path.endswith(".sql"):
            parts = PurePosixPath(path).parts
            via = (
                "migration" if any(p in ("migrations", "migration") for p in parts) else "sql_file"
            )
            out.sql(files[path], path, 1, via)
    for facts in out.tables.values():
        for role in ("defined_in", "altered_in", "read_by", "written_by"):
            getattr(facts, role).sort(key=lambda r: (r.file, r.line, r.via))
    return TableInventory(
        tables=[out.tables[k] for k in sorted(out.tables)],
        unresolved=sorted(out.unresolved, key=lambda u: (u.file, u.line)),
        files_unparsed=out.unparsed,
    )
