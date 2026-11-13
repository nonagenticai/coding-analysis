"""Internal module graph of one repository (Python or Rust), with SCCs and porting layers.

* ``nodes``: every internal module (Python: every parsed ``.py`` file; Rust: every
  file-backed module reachable from a crate root), with non-blank, non-comment
  LOC and its public-symbol count (the same extractor as the API surface).
* ``edges``: internal dependencies only. Python ``import``/``from ... import``
  (``type_only`` under ``if TYPE_CHECKING``) and ``importlib.import_module`` /
  ``__import__`` with a literal name (``dynamic``). Rust ``use`` declarations
  (``use``) and ``crate::``/``super::``/``self::`` qualified paths in code
  (``path``). Absolute Python imports resolve only under the repository's
  source roots (see ``_ModuleIndex``): standard-library and third-party names
  never match an internal module by suffix.
* ``sccs``: strongly connected components with more than one module, or a module
  that depends on itself (Tarjan, ``call_graph_metrics``).
* ``order``: layers of the SCC-condensed DAG, LEAVES FIRST. Layer 0 depends on no
  other internal module; layer k depends only on layers < k. Members of one SCC
  are always in the same layer. Each layer is sorted.
* ``slices``: the porting units. One per SCC of the condensed DAG (an import cycle
  is ONE slice), topologically ordered LEAVES FIRST, ids ``s0..``, with the slice
  ids each one depends on and the summed LOC of its modules.
* ``unresolved``: dynamic imports with a computed name, relative imports that
  climb past the top-level package, absolute imports that match under several
  source roots (with ``candidates``), Rust ``mod`` files that
  do not exist, ``.rs`` files no crate root reaches, ``use`` paths that resolve to
  nothing known, ``include!``.
"""

from __future__ import annotations

import posixpath
import sys
import tomllib
from collections import defaultdict
from typing import Any

from processing.graph_sccs import strongly_connected_components

from . import python_facts as pf
from . import rust_facts as rf
from .text import as_dict, as_list, as_str, get_path
from .types import ModuleEdge, ModuleGraph, ModuleNode, Slice, Unresolved


def condense(
    nodes: list[str], edges: list[tuple[str, str]], loc: dict[str, int] | None = None
) -> tuple[list[list[str]], list[list[str]], list[Slice]]:
    """(sccs, order, slices) for a dependency graph where ``(a, b)`` means *a* depends on *b*.

    ``slices``: one per SCC of the condensed DAG (a cycle is ONE slice), sorted by
    ``(layer, first module)`` so every slice comes after the slices it depends on;
    ids are ``s0``, ``s1``, ... in that order and ``depends_on`` lists slice ids.
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in sorted(set(edges)):
        if a in adj and b in adj:
            adj[a].append(b)
    components = strongly_connected_components(sorted(nodes), adj)
    comp_of = {member: i for i, comp in enumerate(components) for member in comp}
    level: dict[int, int] = {}
    deps_of: dict[int, set[int]] = {}
    for i, comp in enumerate(components):  # reverse topological: dependencies first
        deps_of[i] = {comp_of[b] for a in comp for b in adj[a]} - {i}
        level[i] = 1 + max((level[d] for d in deps_of[i]), default=-1)
    order: dict[int, list[str]] = defaultdict(list)
    for i, comp in enumerate(components):
        order[level[i]].extend(comp)
    self_loop = [len(comp) > 1 or comp[0] in adj.get(comp[0], []) for comp in components]
    cyclic = [sorted(comp) for i, comp in enumerate(components) if self_loop[i]]
    ranked = sorted(range(len(components)), key=lambda i: (level[i], min(components[i])))
    slice_id = {comp: f"s{k}" for k, comp in enumerate(ranked)}
    loc = loc or {}
    slices = [
        Slice(
            id=slice_id[i],
            layer=level[i],
            modules=sorted(components[i]),
            cyclic=self_loop[i],
            depends_on=sorted((slice_id[d] for d in deps_of[i]), key=lambda sid: int(sid[1:])),
            loc=sum(loc.get(m, 0) for m in components[i]),
        )
        for i in ranked
    ]
    return sorted(cyclic), [sorted(order[k]) for k in sorted(order)], slices


def layers(
    nodes: list[str], edges: list[tuple[str, str]]
) -> tuple[list[list[str]], list[list[str]]]:
    """(sccs, order) for a dependency graph where ``(a, b)`` means *a* depends on *b*."""
    sccs, order, _ = condense(nodes, edges)
    return sccs, order


def _finish(
    language: str,
    nodes: list[ModuleNode],
    edges: list[ModuleEdge],
    unresolved: list[Unresolved],
    unparsed: dict[str, str],
) -> ModuleGraph:
    unique: dict[tuple[str, str, str], ModuleEdge] = {}
    for edge in sorted(edges, key=lambda e: (e.source, e.target, e.kind, e.file, e.line)):
        unique.setdefault((edge.source, edge.target, edge.kind), edge)
    kept = list(unique.values())
    sccs, order, slices = condense(
        [n.module for n in nodes],
        [(e.source, e.target) for e in kept],
        {n.module: n.loc for n in nodes},
    )
    return ModuleGraph(
        language=language,
        nodes=sorted(nodes, key=lambda n: n.module),
        edges=kept,
        sccs=sccs,
        order=order,
        slices=slices,
        unresolved=sorted(unresolved, key=lambda u: (u.file, u.line, u.name or "")),
        files_unparsed=dict(sorted(unparsed.items())),
    )


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _join(base: str, rel: Any) -> str | None:
    """``base/rel`` normalised, or None when it is the base itself or leaves it."""
    if not isinstance(rel, str):
        return None
    joined = posixpath.normpath(posixpath.join(base, rel.strip().strip("/")))
    if joined in (".", "") or joined.startswith(("../", "/")) or joined == "..":
        return None
    return joined


def _declared_roots(path: str, text: str) -> list[str]:
    """Package roots a ``pyproject.toml`` / ``setup.cfg`` at *path* declares."""
    base = posixpath.dirname(path)
    found: list[str | None] = []
    if path.endswith("pyproject.toml"):
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return []
        found.append(
            _join(base, as_dict(get_path(data, "tool", "setuptools", "package-dir")).get(""))
        )
        for where in as_list(get_path(data, "tool", "setuptools", "packages", "find", "where")):
            found.append(_join(base, where))
        for pkg in as_list(get_path(data, "tool", "poetry", "packages")):
            src = as_str(as_dict(pkg).get("from"))
            found.append(_join(base, src) if src else None)
        for pkg in as_list(
            get_path(data, "tool", "hatch", "build", "targets", "wheel", "packages")
        ):
            parent = posixpath.dirname(pkg) if isinstance(pkg, str) else ""
            found.append(_join(base, parent) if parent else None)
        found.append(
            _join(base, as_str(get_path(data, "tool", "pdm", "build", "package-dir")) or "")
        )
    else:  # setup.cfg: ``package_dir = =src`` and ``[options.packages.find] where = src``
        section = ""
        for raw in text.splitlines()[:5000]:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
            elif section == "options" and line.startswith("="):
                found.append(_join(base, line[1:]))
            elif section == "options" and line.startswith("package_dir") and "==" not in line:
                value = line.partition("=")[2].strip()
                if value.startswith("="):
                    found.append(_join(base, value[1:]))
            elif section == "options.packages.find" and line.startswith("where"):
                found.append(_join(base, line.partition("=")[2]))
    return [r for r in found if r]


def python_source_roots(files: dict[str, str]) -> list[str]:
    """Directories whose children are top-level packages (``src/`` of a src layout).

    Declared in ``pyproject.toml`` (setuptools ``package-dir``/``packages.find.where``,
    poetry ``packages.from``, hatch wheel ``packages``, pdm ``package-dir``) or
    ``setup.cfg``; or by convention a ``src`` directory that is not itself a package
    and directly holds one (``src/pkg/__init__.py``). Longest first.
    """
    roots: set[str] = set()
    packages = {posixpath.dirname(p) for p in files if p.endswith("/__init__.py")}
    for path, text in files.items():
        name = posixpath.basename(path)
        if name in ("pyproject.toml", "setup.cfg"):
            roots.update(_declared_roots(path, text))
    for pkg in packages:
        parent = posixpath.dirname(pkg)
        if posixpath.basename(parent) == "src" and parent not in packages:
            roots.add(parent)
    return sorted(roots, key=lambda r: (-len(r), r))


def _root_of(path: str, roots: list[str], default: str) -> str:
    for r in roots:  # longest first
        if path.startswith(r + "/"):
            return r
    return default


def parse_python_files(
    files: dict[str, str], root: str = ""
) -> tuple[list[pf.PyFile], dict[str, str]]:
    """Facts of every ``.py`` file; module names are relative to the file's source root
    (``python_source_roots``: ``src/pkg/a.py`` is ``pkg.a``), else to *root*."""
    parsed: list[pf.PyFile] = []
    unparsed: dict[str, str] = {}
    prefix = root.strip("/")
    roots = [
        r
        for r in python_source_roots(files)
        if not prefix or r == prefix or r.startswith(prefix + "/")
    ]
    for path in sorted(files):
        if not path.endswith(".py"):
            continue
        try:
            parsed.append(pf.parse_python_file(path, files[path], _root_of(path, roots, root)))
        except (SyntaxError, ValueError, RecursionError) as exc:
            unparsed[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return parsed, unparsed


_STDLIB = frozenset(sys.stdlib_module_names)
_GRAPH_REASONS = ("dynamic import", "relative import")


class _ModuleIndex:
    """Resolve imports to internal dotted module names, the way the interpreter would.

    Relative imports are exact names. Absolute imports are looked up under the
    repository's *source roots* only, never by bare suffix, so ``import logging``
    can never land on ``app/logging.py``:

    * ``""`` (the analysed root, i.e. ``files``' own top level), and
    * every directory that is not itself inside a package and directly holds a
      package (``src/`` in a src-layout with no ``root`` given), and
    * for one importer, its own directory when that is not a package (a script's
      directory is ``sys.path[0]``).

    A standard-library top-level name only ever matches at ``""`` (a real
    top-level shadowing module); third-party names match nothing because they
    are not internal modules under any root.
    """

    def __init__(self, modules: dict[str, pf.PyFile]) -> None:
        self.exact = set(modules)
        self.packages = {m for m, f in modules.items() if f.is_package}
        roots = {""}
        for pkg in self.packages:
            parent = pkg.rpartition(".")[0]
            if parent and not self._inside_package(parent) and parent not in self.packages:
                roots.add(parent)
        self.roots = sorted(roots)

    def _inside_package(self, dotted: str) -> bool:
        parts = dotted.split(".")
        return any(".".join(parts[:k]) in self.packages for k in range(1, len(parts) + 1))

    def _under(self, root: str, name: str) -> str | None:
        full = f"{root}.{name}" if root else name
        return full if full in self.exact else None

    def find(self, name: str, importer: str, relative: bool) -> tuple[str | None, list[str]]:
        """(target, ambiguous candidates). Both empty means "not internal"."""
        if not name:
            return None, []
        if relative:
            return (name if name in self.exact else None), []
        if name.split(".", 1)[0] in _STDLIB:
            return self._under("", name), []
        roots = list(self.roots)
        own_dir = importer.rpartition(".")[0]
        if own_dir and own_dir not in self.packages and own_dir not in roots:
            roots.append(own_dir)
        hits = sorted({h for h in (self._under(r, name) for r in roots) if h})
        if len(hits) > 1:
            return None, hits
        return (hits[0] if hits else None), []


def python_graph(files: dict[str, str], root: str = "") -> ModuleGraph:
    parsed, unparsed = parse_python_files(files, root)
    by_module: dict[str, pf.PyFile] = {}
    unresolved: list[Unresolved] = []
    for facts in parsed:
        if facts.module in by_module:
            unresolved.append(
                Unresolved(
                    file=facts.path,
                    line=1,
                    name=facts.module,
                    reason=f"module name also produced by {by_module[facts.module].path}",
                )
            )
            continue
        by_module[facts.module] = facts
    index = _ModuleIndex(by_module)
    nodes = [ModuleNode(m, f.path, f.loc, len(f.symbols)) for m, f in by_module.items() if m]
    edges: list[ModuleEdge] = []
    for module, facts in by_module.items():
        unresolved.extend(u for u in facts.unresolved if u.reason.startswith(_GRAPH_REASONS))
        for imp in facts.imports:
            target: str | None = None
            for candidate in imp.candidates:
                target, tied = index.find(candidate, module, imp.relative)
                if tied:
                    unresolved.append(
                        Unresolved(
                            file=facts.path,
                            line=imp.line,
                            name=candidate,
                            reason="absolute import matches modules under several source roots",
                            candidates=tied,
                        )
                    )
                    break
                if target:
                    break
            if target is None or target == module:
                continue
            edges.append(ModuleEdge(module, target, imp.kind, facts.path, imp.line))
    return _finish("python", nodes, edges, unresolved, unparsed)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def _module_nodes_in(container: Any):
    """Descendants of a module body, not descending into nested ``mod`` items."""
    stack = list(container.named_children)
    while stack:
        cur = stack.pop()
        if cur.type == "mod_item":
            continue
        yield cur
        stack.extend(cur.named_children)


def rust_graph(files: dict[str, str]) -> ModuleGraph:
    ws = rf.load_workspace(files)
    symbols, _ = rf.surface(ws)
    crates = {c.name: c for c in ws.crates}
    lib_crates = {c.name: c.name for c in ws.crates if c.is_lib}
    public_count: dict[str, int] = defaultdict(int)
    for sym in symbols:
        if sym.visibility.startswith("pub") and sym.kind != "impl_fn":
            public_count[ws.modules[sym.module_path].file_module] += 1
    nodes = [
        ModuleNode(m.path, m.file, m.loc, public_count[m.path])
        for m in ws.modules.values()
        if m.file_module == m.path
    ]
    edges: list[ModuleEdge] = []
    unresolved = list(ws.unresolved)
    for module in ws.modules.values():
        crate = crates[module.crate]
        local = rf.local_item_names(module.node)
        source = module.file_module
        for node in _module_nodes_in(module.node):
            if node.type == "use_declaration":
                arg = node.child_by_field_name("argument")
                for segs in rf.use_paths(arg):
                    res = rf.resolve_path(segs, module.path, ws, crate, lib_crates, local)
                    if res.external:
                        continue
                    if res.target is None:
                        unresolved.append(
                            Unresolved(
                                file=module.file,
                                line=node.start_point[0] + 1,
                                name="::".join(segs),
                                reason="use path root is not a module, workspace crate or declared dependency",
                            )
                        )
                        continue
                    target = ws.modules[res.target].file_module
                    if target != source:
                        edges.append(
                            ModuleEdge(source, target, "use", module.file, node.start_point[0] + 1)
                        )
        for segs, line in rf.scoped_roots(module.node):
            res = rf.resolve_path(segs, module.path, ws, crate, lib_crates, local)
            if res.target is None:
                continue
            target = ws.modules[res.target].file_module
            if target != source:
                edges.append(ModuleEdge(source, target, "path", module.file, line))
    return _finish("rust", nodes, edges, unresolved, ws.unparsed)


def module_graph(files: dict[str, str], language: str, root: str = "") -> ModuleGraph:
    if language == "rust":
        return rust_graph(files)
    return python_graph(files, root)
