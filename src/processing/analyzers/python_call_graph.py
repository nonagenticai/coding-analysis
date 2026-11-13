"""
In-process Python call graph built with the stdlib ``ast`` module.

No external tool is needed. Nodes are every function and method, qualified as
``package.module.Class.method``; edges are call sites that resolve, in order,
to:

1. a name bound in an enclosing function scope or the same module,
2. ``self.m`` / ``cls.m`` inside a class (including bases defined in the repo),
3. a name or module imported with ``import x`` / ``from x import y`` that
   points at a repo module.

Anything else (``obj.method()`` on an unknown receiver, third-party calls) is
counted in ``unresolved_calls`` and never turned into an edge. Calls to
builtins are neither edges nor unresolved.
"""

from __future__ import annotations

import ast
import builtins
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_BUILTINS = frozenset(dir(builtins))
_SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", ".mypy_cache", "build"}
)

_EDGE = "edge"
_IGNORE = "ignore"
_UNRESOLVED = "unresolved"


@dataclass
class _Module:
    name: str
    rel_path: str
    tree: ast.Module
    # top-level name -> qualified target (function/class qualname or module name)
    bindings: dict[str, str] = field(default_factory=dict)


@dataclass
class _Func:
    qualname: str
    module: str
    local_name: str  # qualname relative to the module, e.g. "C.m"
    rel_path: str
    line: int
    kind: str  # "function" | "method"
    node: ast.FunctionDef | ast.AsyncFunctionDef
    class_qualname: str | None
    # enclosing function scopes, innermost last: {name -> qualname}
    scopes: list[dict[str, str]]


@dataclass
class PythonCallGraph:
    """Result of the AST backend."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    unresolved_calls: int
    external_nodes: list[dict[str, Any]]
    files_analyzed: int
    parse_errors: list[str]


def module_name_for(rel_path: str) -> str:
    """``pkg/sub/mod.py`` -> ``pkg.sub.mod``; ``pkg/__init__.py`` -> ``pkg``."""
    parts = rel_path.replace(os.sep, "/")[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p) or "__init__"


def find_python_files(roots: list[str]) -> list[tuple[str, str]]:
    """``(file, import root)`` for every ``.py`` file under *roots*, sorted."""
    found: dict[str, str] = {}
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            if root.endswith(".py"):
                found.setdefault(root, os.path.dirname(root))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".py"):
                    found.setdefault(os.path.abspath(os.path.join(dirpath, filename)), root)
    return sorted(found.items())


class PythonCallGraphBuilder:
    """Build a call graph for the ``.py`` files under a repository."""

    def __init__(self, repository_path: str, include_external: bool = False) -> None:
        self.repo = os.path.abspath(repository_path)
        self.include_external = include_external
        self.modules: dict[str, _Module] = {}
        self.funcs: dict[str, _Func] = {}
        self.classes: dict[str, list[str]] = {}  # class qualname -> base qualnames
        self.parse_errors: list[str] = []

    # ------------------------------------------------------------------ load
    def build(self, files: list[tuple[str, str]]) -> PythonCallGraph:
        for path, root in files:
            self._load(path, root)
        for module in self.modules.values():
            self._collect_bindings(module)
        for module in self.modules.values():
            self._collect_defs(module.tree.body, module, None, [], prefix="")
        # bases need every class known first
        for module in self.modules.values():
            self._collect_bases(module.tree.body, module, "")

        edges: dict[tuple[str, str], dict[str, Any]] = {}
        external: dict[str, dict[str, Any]] = {}
        unresolved = 0
        for func in self.funcs.values():
            for call in self._calls_in(func.node):
                outcome, target = self._resolve(call.func, func)
                if outcome == _IGNORE:
                    continue
                if outcome == _UNRESOLVED:
                    unresolved += 1
                    if not self.include_external:
                        continue
                    external.setdefault(
                        target, {"id": target, "function": target, "module": "", "external": True}
                    )
                    call_type = "external"
                else:
                    call_type = "direct"
                edge = edges.setdefault(
                    (func.qualname, target),
                    {
                        "from": func.qualname,
                        "to": target,
                        "line": call.lineno,
                        "call_type": call_type,
                        "count": 0,
                    },
                )
                edge["count"] += 1

        nodes = [
            {
                "id": f.qualname,
                "function": f.local_name,
                "module": f.module,
                "file": f.rel_path,
                "line_number": f.line,
                "type": f.kind,
            }
            for f in sorted(self.funcs.values(), key=lambda f: (f.rel_path, f.line))
        ]
        return PythonCallGraph(
            nodes=nodes,
            edges=list(edges.values()),
            unresolved_calls=unresolved,
            external_nodes=list(external.values()),
            files_analyzed=len(self.modules),
            parse_errors=self.parse_errors,
        )

    def _load(self, path: str, root: str) -> None:
        rel = os.path.relpath(path, self.repo)
        try:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source, filename=rel)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
            self.parse_errors.append(f"{rel}: {exc}")
            return
        import_rel = os.path.relpath(path, root).replace(os.sep, "/")
        if import_rel.startswith("src/"):
            import_rel = import_rel[len("src/") :]
        name = module_name_for(import_rel)
        if name in self.modules:
            self.parse_errors.append(f"{rel}: duplicate module name {name}, skipped")
            return
        self.modules[name] = _Module(name=name, rel_path=rel, tree=tree)

    def _is_package(self, module: _Module) -> bool:
        return module.rel_path.endswith("__init__.py")

    def _absolute(self, module: _Module, level: int, target: str | None) -> str:
        if level == 0:
            return target or ""
        parts = module.name.split(".")
        if not self._is_package(module):
            parts = parts[:-1]
        if level > 1:
            parts = parts[: len(parts) - (level - 1)]
        base = ".".join(parts)
        if target:
            return f"{base}.{target}" if base else target
        return base

    def _collect_bindings(self, module: _Module) -> None:
        for stmt in ast.walk(module.tree):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.asname:
                        module.bindings[alias.asname] = alias.name
                    else:
                        top = alias.name.split(".")[0]
                        module.bindings.setdefault(top, top)
            elif isinstance(stmt, ast.ImportFrom):
                source = self._absolute(module, stmt.level, stmt.module)
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    module.bindings[bound] = f"{source}.{alias.name}" if source else alias.name

    def _collect_defs(
        self,
        body: list[ast.stmt],
        module: _Module,
        class_qual: str | None,
        scopes: list[dict[str, str]],
        prefix: str,
    ) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                local = f"{prefix}{stmt.name}"
                qual = f"{module.name}.{local}"
                self.classes.setdefault(qual, [])
                if not scopes and not class_qual:
                    module.bindings[stmt.name] = qual
                self._collect_defs(stmt.body, module, qual, scopes, prefix=f"{local}.")
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = f"{prefix}{stmt.name}"
                qual = f"{module.name}.{local}"
                if scopes:
                    scopes[-1][stmt.name] = qual
                elif not class_qual:
                    module.bindings[stmt.name] = qual
                inner: dict[str, str] = {}
                self.funcs[qual] = _Func(
                    qualname=qual,
                    module=module.name,
                    local_name=local,
                    rel_path=module.rel_path,
                    line=stmt.lineno,
                    kind="method" if class_qual and not scopes else "function",
                    node=stmt,
                    class_qualname=class_qual,
                    scopes=[*scopes, inner],
                )
                # nested defs: register into `inner` before calls are resolved
                self._collect_defs(stmt.body, module, class_qual, [*scopes, inner], f"{local}.")
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
                for block in _sub_blocks(stmt):
                    self._collect_defs(block, module, class_qual, scopes, prefix)

    def _collect_bases(self, body: list[ast.stmt], module: _Module, prefix: str) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                local = f"{prefix}{stmt.name}"
                qual = f"{module.name}.{local}"
                for base in stmt.bases:
                    dotted = _dotted(base)
                    if dotted is None:
                        continue
                    resolved = self._resolve_dotted(dotted, module, [])
                    if resolved in self.classes:
                        self.classes[qual].append(resolved)
                self._collect_bases(stmt.body, module, f"{local}.")

    # --------------------------------------------------------------- resolve
    def _calls_in(self, func_node: ast.AST) -> list[ast.Call]:
        """Calls in *func_node*'s own body (not inside nested defs/classes)."""
        calls: list[ast.Call] = []
        stack: list[ast.AST] = list(ast.iter_child_nodes(func_node))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # decorators/defaults of nested defs run in this scope
                stack.extend(getattr(node, "decorator_list", []))
                continue
            if isinstance(node, ast.Call):
                calls.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return sorted(calls, key=lambda c: (c.lineno, c.col_offset))

    def _method_in_class(self, class_qual: str, name: str, depth: int = 0) -> str | None:
        candidate = f"{class_qual}.{name}"
        if candidate in self.funcs:
            return candidate
        if depth > 10:
            return None
        for base in self.classes.get(class_qual, []):
            found = self._method_in_class(base, name, depth + 1)
            if found:
                return found
        return None

    def _resolve_dotted(self, dotted: str, module: _Module, scopes: list[dict[str, str]]) -> str:
        head, _, rest = dotted.partition(".")
        bound = None
        for scope in reversed(scopes):
            if head in scope:
                bound = scope[head]
                break
        if bound is None:
            bound = module.bindings.get(head)
        if bound is None:
            return dotted
        return f"{bound}.{rest}" if rest else bound

    def _resolve(self, func_expr: ast.expr, caller: _Func) -> tuple[str, str]:
        """(outcome, target): an edge target, an ignored call, or an unresolved name."""
        module = self.modules[caller.module]
        dotted = _dotted(func_expr)
        if dotted is None:
            return _UNRESOLVED, "<dynamic>"
        head, _, rest = dotted.partition(".")

        # self.m() / cls.m() inside a class
        if head in {"self", "cls"} and caller.class_qualname and rest and "." not in rest:
            target = self._method_in_class(caller.class_qualname, rest)
            return (_EDGE, target) if target else (_UNRESOLVED, dotted)

        is_bound = any(head in scope for scope in caller.scopes) or head in module.bindings
        if not is_bound and not rest and head in _BUILTINS:
            return _IGNORE, dotted

        qual = self._resolve_dotted(dotted, module, caller.scopes)
        if qual in self.funcs:
            return _EDGE, qual
        if qual in self.classes:
            # Instantiation: an edge to __init__ when the repo defines one.
            init = self._method_in_class(qual, "__init__")
            return (_EDGE, init) if init else (_IGNORE, qual)
        owner, _, attr = qual.rpartition(".")
        if owner in self.classes and attr:
            method = self._method_in_class(owner, attr)
            if method:
                return _EDGE, method
        return _UNRESOLVED, qual


def _sub_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    blocks: list[list[ast.stmt]] = []
    for name in ("body", "orelse", "finalbody"):
        block = getattr(stmt, name, None)
        if block:
            blocks.append(block)
    for handler in getattr(stmt, "handlers", []) or []:
        blocks.append(handler.body)
    return blocks


def _dotted(expr: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def build_python_call_graph(
    repository_path: str, roots: list[str], include_external: bool = False
) -> PythonCallGraph:
    """Build the call graph for all ``.py`` files under *roots*."""
    builder = PythonCallGraphBuilder(repository_path, include_external)
    return builder.build(find_python_files(roots))


__all__ = [
    "PythonCallGraph",
    "PythonCallGraphBuilder",
    "build_python_call_graph",
    "find_python_files",
    "module_name_for",
]
