"""Python per-file facts (stdlib ``ast``): imports, dynamic imports, public API surface.

Visibility follows Python's own conventions: a literal ``__all__`` is the public
surface; without one every top-level name not starting with ``_`` is public.
Methods of a public class are public unless they start with ``_`` (``__init__``
and ``__call__`` are kept: they are the constructor and call signatures).
Anything that cannot be decided statically (a computed ``__all__``, a module
``__getattr__``, ``importlib.import_module(name)``) is reported as unresolved.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .text import unparse
from .types import ApiSymbol, Param, Signature, Unresolved

_KEEP_DUNDERS = {"__init__", "__call__"}
_DYNAMIC_IMPORTERS = {"import_module", "__import__"}
_ATTR_LIMIT = 120


@dataclass
class PyImport:
    candidates: list[str]  # absolute dotted names, most specific first
    kind: str  # import | from_import | dynamic | type_only
    line: int
    # Relative imports name an exact module; absolute ones are resolved against
    # the repository's source roots (see ``module_graph``).
    relative: bool = False


@dataclass
class PyFile:
    path: str
    module: str
    loc: int
    imports: list[PyImport] = field(default_factory=list)
    symbols: list[ApiSymbol] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)
    is_package: bool = False


def module_name(path: str, root: str = "") -> str:
    """``src/pkg/mod.py`` (root ``src``) -> ``pkg.mod``; ``pkg/__init__.py`` -> ``pkg``."""
    rel = path
    prefix = root.strip("/")
    if prefix and (rel == prefix or rel.startswith(prefix + "/")):
        rel = rel[len(prefix) + 1 :]
    parts = list(PurePosixPath(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def count_loc(source: str) -> int:
    """Non-blank lines that are not only a comment."""
    return sum(
        1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )


def _unparse(node: ast.AST | None) -> str | None:
    return unparse(node, _ATTR_LIMIT)


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Signature:
    a = fn.args
    params: list[Param] = []
    positional = [*a.posonlyargs, *a.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults)) + list(
        a.defaults
    )
    for arg, default in zip(positional, defaults, strict=True):
        params.append(Param(arg.arg, _unparse(arg.annotation), _unparse(default)))
    if a.vararg is not None:
        params.append(Param("*" + a.vararg.arg, _unparse(a.vararg.annotation)))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        params.append(Param(arg.arg, _unparse(arg.annotation), _unparse(default)))
    if a.kwarg is not None:
        params.append(Param("**" + a.kwarg.arg, _unparse(a.kwarg.annotation)))
    return Signature(
        params=params,
        returns=_unparse(fn.returns),
        is_async=isinstance(fn, ast.AsyncFunctionDef),
    )


def _literal_names(node: ast.expr) -> list[str] | None:
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        out = []
        for elt in node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return None
            out.append(elt.value)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_names(node.left), _literal_names(node.right)
        return None if left is None or right is None else left + right
    return None


def _is_type_checking(node: ast.expr) -> bool:
    """``TYPE_CHECKING`` or ``typing.TYPE_CHECKING`` (any qualifier), nothing else."""
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    return isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"


def _type_checking_guard(test: ast.expr) -> tuple[bool, bool]:
    """(body is type-only, else-branch is type-only) for an ``if`` test.

    ``if TYPE_CHECKING:`` makes the body type-only; ``if not TYPE_CHECKING:`` makes
    the ``else`` branch type-only and the body a real runtime import.
    """
    if _is_type_checking(test):
        return True, False
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _is_type_checking(test.operand)
    ):
        return False, True
    return False, False


def _top_level(body: list[ast.stmt], type_only: bool = False):
    """Module-level statements, descending into ``if``/``try`` blocks (not defs)."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            body_only, else_only = _type_checking_guard(stmt.test)
            yield from _top_level(stmt.body, type_only or body_only)
            yield from _top_level(stmt.orelse, type_only or else_only)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _top_level(block, type_only)
            for handler in stmt.handlers:
                yield from _top_level(handler.body, type_only)
        else:
            yield stmt, type_only


def _package_of(module: str, is_package: bool) -> str:
    return module if is_package else module.rpartition(".")[0]


def _absolute(base_pkg: str, level: int, module: str | None) -> str | None:
    """Absolute dotted name of a relative import, ``None`` when it climbs past the root.

    Level 1 is the importing module's own package (``""`` for a top-level module,
    whose siblings are top-level modules). Each further level strips one package;
    stripping the last one is Python's "attempted relative import beyond
    top-level package".
    """
    parts = base_pkg.split(".") if base_pkg else []
    if level > 1:
        if level - 1 >= len(parts):
            return None
        parts = parts[: len(parts) - (level - 1)]
    if module:
        parts.append(module)
    return ".".join(p for p in parts if p)


def parse_python_file(path: str, source: str, root: str = "") -> PyFile:
    """Facts of one Python file. Raises ``SyntaxError``/``ValueError`` when unparsable."""
    tree = ast.parse(source, filename=path)
    module = module_name(path, root)
    is_package = PurePosixPath(path).name == "__init__.py"
    facts = PyFile(path=path, module=module, loc=count_loc(source), is_package=is_package)
    base_pkg = _package_of(module, is_package)
    _imports(tree, facts, base_pkg)
    _surface(tree, facts)
    return facts


def _imports(tree: ast.Module, facts: PyFile, base_pkg: str) -> None:
    type_only_lines: set[int] = set()
    for stmt, type_only in _top_level(tree.body):
        if type_only and isinstance(stmt, ast.Import | ast.ImportFrom):
            type_only_lines.add(stmt.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            kind = "type_only" if node.lineno in type_only_lines else "import"
            for alias in node.names:
                facts.imports.append(PyImport(_prefixes(alias.name), kind, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            kind = "type_only" if node.lineno in type_only_lines else "from_import"
            if not node.level:
                target = node.module or ""
                for alias in node.names:
                    subs = [f"{target}.{alias.name}"] if alias.name != "*" and target else []
                    facts.imports.append(PyImport([*subs, *_prefixes(target)], kind, node.lineno))
                continue
            resolved = _absolute(base_pkg, node.level, node.module)
            if resolved is None:
                facts.unresolved.append(
                    Unresolved(
                        file=facts.path,
                        line=node.lineno,
                        name="." * node.level + (node.module or ""),
                        reason="relative import climbs past the top-level package",
                    )
                )
                continue
            for alias in node.names:
                subs = [f"{resolved}.{alias.name}" if resolved else alias.name]
                if alias.name == "*":
                    subs = []
                facts.imports.append(
                    PyImport([*subs, *_prefixes(resolved)], kind, node.lineno, relative=True)
                )
        elif isinstance(node, ast.Call):
            _dynamic_import(node, facts)
        elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__" and node in tree.body:
            facts.unresolved.append(
                Unresolved(
                    file=facts.path,
                    line=node.lineno,
                    name="__getattr__",
                    reason="module __getattr__: attributes (and lazy imports) are computed at runtime",
                )
            )


def _prefixes(dotted: str) -> list[str]:
    parts = [p for p in dotted.split(".") if p]
    return [".".join(parts[:k]) for k in range(len(parts), 0, -1)]


def _dynamic_import(node: ast.Call, facts: PyFile) -> None:
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
    if name not in _DYNAMIC_IMPORTERS:
        return
    if isinstance(fn, ast.Attribute) and _unparse(fn.value) not in ("importlib", "builtins"):
        return
    arg = node.args[0] if node.args else None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        facts.imports.append(PyImport(_prefixes(arg.value), "dynamic", node.lineno))
        return
    facts.unresolved.append(
        Unresolved(
            file=facts.path,
            line=node.lineno,
            name=name,
            reason="dynamic import with a computed module name",
            expression=_unparse(node),
        )
    )


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    return [d for d in (_unparse(dec) for dec in node.decorator_list) if d]


def _surface(tree: ast.Module, facts: PyFile) -> None:
    exported: list[str] | None = None
    defined: dict[str, tuple[str, int]] = {}  # name -> (kind, line) of top-level bindings
    imported: dict[str, int] = {}
    nodes: list[tuple[ast.stmt, bool]] = list(_top_level(tree.body))
    for stmt, _ in nodes:
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign | ast.AugAssign):
            targets = [stmt.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = stmt.value if not isinstance(stmt, ast.AnnAssign) else stmt.value
                names = _literal_names(value) if value is not None else None
                if names is None:
                    facts.unresolved.append(
                        Unresolved(
                            file=facts.path,
                            line=stmt.lineno,
                            name="__all__",
                            reason="__all__ is computed: the public surface cannot be read statically",
                        )
                    )
                    exported = exported or []
                elif isinstance(stmt, ast.AugAssign):
                    exported = (exported or []) + names
                else:
                    exported = names
            elif isinstance(target, ast.Name):
                defined.setdefault(target.id, ("const", stmt.lineno))
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            defined.setdefault(stmt.name, ("function", stmt.lineno))
        elif isinstance(stmt, ast.ClassDef):
            defined.setdefault(stmt.name, ("class", stmt.lineno))
        elif isinstance(stmt, ast.Import | ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    imported[alias.asname or alias.name.split(".")[0]] = stmt.lineno

    def public(name: str) -> bool:
        if exported is not None:
            return name in exported
        return not name.startswith("_")

    module = facts.module
    for stmt, _ in nodes:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef) and public(stmt.name):
            facts.symbols.append(
                ApiSymbol(
                    id=f"{module}.{stmt.name}",
                    module_path=module,
                    name=stmt.name,
                    kind="function",
                    visibility="public",
                    file=facts.path,
                    line=stmt.lineno,
                    signature=_signature(stmt),
                    attributes=_decorators(stmt),
                )
            )
        elif isinstance(stmt, ast.ClassDef) and public(stmt.name):
            facts.symbols.append(
                ApiSymbol(
                    id=f"{module}.{stmt.name}",
                    module_path=module,
                    name=stmt.name,
                    kind="class",
                    visibility="public",
                    file=facts.path,
                    line=stmt.lineno,
                    attributes=[
                        *(f"base:{b}" for b in (_unparse(x) for x in stmt.bases) if b),
                        *_decorators(stmt),
                    ],
                )
            )
            for item in stmt.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if item.name.startswith("_") and item.name not in _KEEP_DUNDERS:
                    continue
                facts.symbols.append(
                    ApiSymbol(
                        id=f"{module}.{stmt.name}.{item.name}",
                        module_path=module,
                        name=item.name,
                        kind="method",
                        visibility="public",
                        file=facts.path,
                        line=item.lineno,
                        signature=_signature(item),
                        attributes=_decorators(item),
                        owner=stmt.name,
                    )
                )
    for name, (kind, line) in sorted(defined.items(), key=lambda kv: kv[1][1]):
        if kind == "const" and name != "__all__" and public(name):
            facts.symbols.append(
                ApiSymbol(
                    id=f"{module}.{name}",
                    module_path=module,
                    name=name,
                    kind="const",
                    visibility="public",
                    file=facts.path,
                    line=line,
                )
            )
    for name in exported or []:
        if name in defined:
            continue
        if name in imported:
            facts.symbols.append(
                ApiSymbol(
                    id=f"{module}.{name}",
                    module_path=module,
                    name=name,
                    kind="reexport",
                    visibility="public",
                    file=facts.path,
                    line=imported[name],
                )
            )
        else:
            facts.unresolved.append(
                Unresolved(
                    file=facts.path,
                    line=1,
                    name=name,
                    reason="name listed in __all__ is not bound in the module",
                )
            )
