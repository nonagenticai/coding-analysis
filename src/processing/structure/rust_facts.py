"""Rust facts (tree-sitter): crates, the module tree, ``use`` edges and the API surface.

Modules are found the way rustc finds them: from each crate root (``[lib]`` /
``[[bin]]`` paths in ``Cargo.toml``, else ``src/lib.rs``, ``src/main.rs`` and
``src/bin/*.rs``) by following ``mod x;`` declarations (``x.rs`` or
``x/mod.rs``, or the ``#[path = "..."]`` file). A ``.rs`` file no crate root
reaches is reported, never silently counted as a module.

``use`` paths resolve through ``crate::``/``self::``/``super::``, workspace crate
names and child modules; a path rooted at ``std``/``core``/``alloc`` or a declared
dependency is external (not an edge). Anything else is unresolved.

Macro-generated items are never silently missing: ``macro_rules!`` definitions,
item-position macro invocations, ``#[derive(...)]`` and attribute macros
(``#[get("/x")]``, ``#[tokio::main]``, ``#[async_trait]``) are listed under
``unresolved`` with the macro, while the items they decorate are still listed.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from processing.symbols import _parse

from .text import as_dict, as_list, as_str, clip, get_path
from .text import node_text as _t
from .types import ApiSymbol, Member, Param, Signature, Unresolved

_EXTERNAL_ROOTS = {"std", "core", "alloc", "proc_macro", "test"}
#: Attributes that do not generate or rewrite items (built-ins and derive helpers).
_INERT_ATTRIBUTES = {
    "cfg",
    "cfg_attr",
    "test",
    "allow",
    "warn",
    "deny",
    "forbid",
    "expect",
    "doc",
    "inline",
    "cold",
    "must_use",
    "non_exhaustive",
    "repr",
    "path",
    "deprecated",
    "automatically_derived",
    "macro_export",
    "macro_use",
    "track_caller",
    "no_mangle",
    "export_name",
    "link",
    "link_name",
    "used",
    "ignore",
    "should_panic",
    "global_allocator",
    "rustfmt::skip",
    "serde",
    "sqlx",
    "validate",
    "schemars",
    "arg",
    "command",
    "clap",
    "error",
    "from",
    "source",
    "strum",
    "default",
    "builder",
    "diesel",
    "sea_orm",
    "utoipa",
    "schema",
    "garde",
}
_ITEM_KINDS = {
    "function_item": "function",
    "struct_item": "struct",
    "enum_item": "enum",
    "trait_item": "trait",
    "const_item": "const",
    "static_item": "static",
    "type_item": "type_alias",
    "union_item": "union",
}
_LIMIT = 200


def _line(node: Any) -> int:
    return node.start_point[0] + 1


def _clip(text: str) -> str:
    return clip(text, _LIMIT)


@dataclass
class Crate:
    name: str  # the name code uses (``-`` -> ``_``); bins: ``<bin>`` or ``<bin>(bin)``
    root_file: str
    package_dir: str
    is_lib: bool
    dependencies: set[str] = field(default_factory=set)


@dataclass
class RustModule:
    path: str  # ``crate_name::a::b``
    file: str
    crate: str
    node: Any  # declaration_list / source_file holding the module's items
    file_module: str  # the file-backed module this (maybe inline) module lives in
    loc: int = 0


@dataclass
class RustWorkspace:
    crates: list[Crate]
    modules: dict[str, RustModule]
    unresolved: list[Unresolved]
    unparsed: dict[str, str]
    trees: dict[str, Any] = field(default_factory=dict)


def _cargo_deps(data: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    tables = [data.get(k) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
    for target in as_dict(data.get("target")).values():
        tables += [get_path(target, k) for k in ("dependencies", "dev-dependencies")]
    for table in tables:
        names.update(str(k).replace("-", "_") for k in as_dict(table))
    return names


def default_bin_path(name: str, files: set[str] | dict[str, str], pkg_dir: str) -> str:
    """Cargo's default for a ``[[bin]]`` without ``path``: ``src/bin/<n>.rs``, else
    ``src/bin/<n>/main.rs`` when only that exists (relative to the package)."""
    flat = f"src/bin/{name}.rs"
    nested = f"src/bin/{name}/main.rs"

    def full(rel: str) -> str:
        return f"{pkg_dir}/{rel}" if pkg_dir else rel

    return nested if full(flat) not in files and full(nested) in files else flat


def discover_crates(files: dict[str, str]) -> tuple[list[Crate], list[Unresolved]]:
    crates: list[Crate] = []
    problems: list[Unresolved] = []
    rs_files = {p for p in files if p.endswith(".rs")}
    manifests = sorted(p for p in files if PurePosixPath(p).name == "Cargo.toml")
    for manifest in manifests:
        try:
            data = tomllib.loads(files[manifest])
        except tomllib.TOMLDecodeError as exc:
            problems.append(Unresolved(file=manifest, line=1, reason=str(exc)))
            continue
        package = as_dict(data.get("package"))
        package_name = as_str(package.get("name"))
        if not package_name:
            continue  # a virtual workspace manifest
        pkg_dir = str(PurePosixPath(manifest).parent)
        pkg_dir = "" if pkg_dir == "." else pkg_dir

        def rel(p: str, base: str = pkg_dir) -> str:
            return str(PurePosixPath(base) / p) if base else p

        deps = _cargo_deps(data)
        pkg_name = package_name.replace("-", "_")
        lib = as_dict(data.get("lib"))
        lib_path = rel(as_str(lib.get("path")) or "src/lib.rs")
        lib_name = (as_str(lib.get("name")) or pkg_name).replace("-", "_")
        has_lib = lib_path in rs_files
        if has_lib:
            crates.append(Crate(lib_name, lib_path, pkg_dir, True, deps))
        bins: list[tuple[str, str]] = []
        for entry in as_list(data.get("bin")):
            bin_name = as_str(get_path(entry, "name"))
            if not bin_name:
                continue
            path = as_str(entry.get("path"))
            if path is None:
                path = (
                    "src/main.rs"
                    if bin_name == package_name
                    else default_bin_path(bin_name, rs_files, pkg_dir)
                )
            bins.append((bin_name.replace("-", "_"), rel(path)))
        declared = {p for _, p in bins}
        if rel("src/main.rs") in rs_files and rel("src/main.rs") not in declared:
            bins.append((pkg_name, rel("src/main.rs")))
        for p in sorted(rs_files):
            parent = PurePosixPath(p).parent
            if p in declared:
                continue
            if str(parent) == rel("src/bin"):
                bins.append((PurePosixPath(p).stem.replace("-", "_"), p))
            elif PurePosixPath(p).name == "main.rs" and str(parent.parent) == rel("src/bin"):
                bins.append((parent.name.replace("-", "_"), p))
        for name, path in bins:
            if path not in rs_files:
                problems.append(
                    Unresolved(file=manifest, line=1, name=name, reason="[[bin]] path not found")
                )
                continue
            label = f"{name}(bin)" if has_lib and name == lib_name else name
            crates.append(
                Crate(label, path, pkg_dir, False, deps | ({lib_name} if has_lib else set()))
            )
    if not manifests:
        for p in sorted(rs_files):
            if PurePosixPath(p).name in ("lib.rs", "main.rs"):
                crates.append(Crate("crate", p, str(PurePosixPath(p).parent), p.endswith("lib.rs")))
    return crates, problems


_COMMENTS = ("line_comment", "block_comment")


def outer_attributes(container: Any) -> dict[int, list[Any]]:
    """``{item node id: its outer attribute items, in source order}`` for *container*.

    One forward pass over the container's children: attributes accumulate until
    the next item, comments between them and the item are skipped. Walking back
    from every item instead (``prev_named_sibling``, itself linear in the
    sibling index) made a file of N comment lines cost O(N^3).
    """
    out: dict[int, list[Any]] = {}
    pending: list[Any] = []
    for child in container.named_children:
        kind = child.type
        if kind == "attribute_item":
            pending.append(child)
        elif kind not in _COMMENTS:
            if pending:
                out[child.id] = pending
                pending = []
    return out


def _attr_name(attr_item: Any) -> tuple[str, Any]:
    attr = next((c for c in attr_item.named_children if c.type == "attribute"), None)
    if attr is None:
        return "", None
    head = attr.named_children[0] if attr.named_children else None
    return _t(head), attr


def _path_attr(attrs: list[Any]) -> str | None:
    for item in attrs:
        name, attr = _attr_name(item)
        if name == "path" and attr is not None:
            value = attr.child_by_field_name("value")
            if value is not None:
                inner = next((c for c in value.named_children if c.type == "string_content"), None)
                return _t(inner) if inner is not None else None
    return None


def load_workspace(files: dict[str, str]) -> RustWorkspace:
    """Parse every reachable module of every crate in *files*."""
    crates, problems = discover_crates(files)
    ws = RustWorkspace(crates=crates, modules={}, unresolved=problems, unparsed={})
    reached: set[str] = set()

    def tree_of(path: str) -> Any | None:
        if path in ws.trees:
            return ws.trees[path]
        tree = _parse("rust", files[path].encode("utf-8"))
        if tree is None:
            ws.unparsed[path] = "grammar not available"
            return None
        ws.trees[path] = tree
        return tree

    def visit_file(path: str, mod_path: str, crate: str, depth: int) -> None:
        if path in reached or depth > 64:
            return
        tree = tree_of(path)
        if tree is None:
            return
        reached.add(path)
        name = PurePosixPath(path).name
        mod_rs_like = name in ("mod.rs", "lib.rs", "main.rs") or depth == 0
        here = PurePosixPath(path).parent
        child_dir = here if mod_rs_like else here / PurePosixPath(path).stem
        ws.modules[mod_path] = RustModule(
            mod_path, path, crate, tree.root_node, mod_path, loc=_loc(files[path])
        )
        visit_items(tree.root_node, path, mod_path, crate, child_dir, here, depth, mod_path)

    def visit_items(
        container: Any,
        path: str,
        mod_path: str,
        crate: str,
        child_dir: PurePosixPath,
        file_dir: PurePosixPath,
        depth: int,
        file_module: str,
    ) -> None:
        attrs_of = outer_attributes(container)
        for item in container.named_children:
            if item.type != "mod_item":
                if item.type == "macro_invocation" and _t(item.child_by_field_name("macro")) in (
                    "include",
                    "std::include",
                ):
                    ws.unresolved.append(
                        Unresolved(
                            file=path,
                            line=_line(item),
                            macro="include!",
                            reason="include! splices another file: its items are not resolved",
                            expression=_clip(_t(item)),
                        )
                    )
                continue
            name = _t(item.child_by_field_name("name"))
            sub = f"{mod_path}::{name}"
            body = item.child_by_field_name("body")
            if body is not None:  # inline `mod x { ... }`
                ws.modules[sub] = RustModule(sub, path, crate, body, file_module)
                visit_items(
                    body, path, sub, crate, child_dir / name, file_dir, depth + 1, file_module
                )
                continue
            explicit = _path_attr(attrs_of.get(item.id, []))
            if explicit is not None:
                candidates = [str(file_dir / explicit)]
            else:
                candidates = [str(child_dir / f"{name}.rs"), str(child_dir / name / "mod.rs")]
            candidates = [str(PurePosixPath(c)) for c in candidates]
            found = [c for c in candidates if c in files]
            if not found:
                ws.unresolved.append(
                    Unresolved(
                        file=path,
                        line=_line(item),
                        name=sub,
                        reason=f"module file not found (looked for {', '.join(candidates)})",
                    )
                )
                continue
            visit_file(found[0], sub, crate, depth + 1)

    for crate in crates:
        visit_file(crate.root_file, crate.name, crate.name, 0)
    for path in sorted(p for p in files if p.endswith(".rs") and p not in reached):
        if path in ws.unparsed:
            continue
        ws.unresolved.append(
            Unresolved(
                file=path,
                line=1,
                reason="file is not reachable from any crate root (no `mod` declaration)",
            )
        )
    return ws


def _loc(source: str) -> int:
    return sum(
        1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("//")
    )


# ---------------------------------------------------------------------------
# use paths
# ---------------------------------------------------------------------------


def use_paths(arg: Any, prefix: list[str] | None = None) -> list[list[str]]:
    """Expand a use tree into full paths (segments), ``*`` for a glob."""
    prefix = prefix or []
    if arg is None:
        return []
    kind = arg.type
    if kind in ("identifier", "crate", "self", "super", "metavariable"):
        return [[*prefix, _t(arg)]]
    if kind == "scoped_identifier":
        return [[*prefix, *_t(arg).replace(" ", "").split("::")]]
    if kind == "use_as_clause":
        return use_paths(arg.child_by_field_name("path"), prefix)
    if kind == "use_wildcard":
        inner = arg.named_children[0] if arg.named_children else None
        base = use_paths(inner, prefix)[0] if inner is not None else prefix
        return [[*base, "*"]]
    if kind == "scoped_use_list":
        path = arg.child_by_field_name("path")
        base = use_paths(path, prefix)[0] if path is not None else prefix
        return use_paths(arg.child_by_field_name("list"), base)
    if kind == "use_list":
        out: list[list[str]] = []
        for child in arg.named_children:
            for p in use_paths(child, prefix):
                out.append(p[:-1] if p and p[-1] == "self" else p)
        return out
    return [[*prefix, _t(arg)]]


@dataclass
class UseResolution:
    target: str | None  # module path (file-backed or inline) the use lands in
    external: bool = False


def resolve_path(
    segments: list[str],
    module: str,
    ws: RustWorkspace,
    crate: Crate,
    lib_crates: dict[str, str],
    local_items: set[str],
) -> UseResolution:
    """Resolve a path used in *module* to the deepest known module, or external/None."""
    if not segments:
        return UseResolution(None)
    first = segments[0]
    rest = segments[1:]
    if first == "crate":
        base = crate.name
    elif first == "self":
        base = module
    elif first == "super":
        base = module
        while segments and segments[0] == "super":
            if "::" not in base:
                return UseResolution(None)
            base = base.rsplit("::", 1)[0]
            segments = segments[1:]
        rest = segments
    elif f"{module}::{first}" in ws.modules:
        base, rest = f"{module}::{first}", rest
    elif first in lib_crates:
        base = lib_crates[first]
    elif first in _EXTERNAL_ROOTS or first in crate.dependencies:
        return UseResolution(None, external=True)
    elif first in local_items or first == "Self" or first[:1].isupper():
        return UseResolution(module)
    else:
        return UseResolution(None)
    current = base
    for seg in rest:
        if seg in ("*", "self"):
            break
        candidate = f"{current}::{seg}"
        if candidate in ws.modules:
            current = candidate
        else:
            break
    return UseResolution(current)


def local_item_names(container: Any) -> set[str]:
    names = set()
    for item in container.named_children:
        name = item.child_by_field_name("name")
        if name is not None:
            names.add(_t(name))
    return names


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


#: The ONE Rust visibility vocabulary, for items and their members alike: the
#: modifier as written (``pub``, ``pub(crate)``, ``pub(super)``, ``pub(in a::b)``)
#: or ``private`` when there is none. Enum variants are ``pub`` (as public as
#: their enum); trait-impl methods are ``pub`` (as public as the trait) and are
#: told apart by ``kind == "impl_fn"``.
PRIVATE = "private"


def _visibility(node: Any) -> str | None:
    """The written visibility modifier of *node*, or None when it has none."""
    for child in node.named_children:
        if child.type == "visibility_modifier":
            return " ".join(_t(child).split())
    return None


def _signature(fn: Any) -> Signature:
    params: list[Param] = []
    plist = fn.child_by_field_name("parameters")
    for p in plist.named_children if plist is not None else []:
        if p.type == "self_parameter":
            params.append(Param("self", _clip(_t(p))))
        elif p.type == "parameter":
            params.append(
                Param(
                    _clip(_t(p.child_by_field_name("pattern"))) or "_",
                    _clip(_t(p.child_by_field_name("type"))),
                )
            )
        elif p.type == "variadic_parameter":
            params.append(Param("...", None))
    generics = _t(fn.child_by_field_name("type_parameters"))
    where = next((c for c in fn.named_children if c.type == "where_clause"), None)
    if where is not None:
        generics = f"{generics} {_t(where)}".strip()
    modifiers = next((c for c in fn.named_children if c.type == "function_modifiers"), None)
    returns = fn.child_by_field_name("return_type")
    return Signature(
        params=params,
        returns=_clip(_t(returns)) if returns is not None else None,
        generics=_clip(generics) or None,
        is_async=modifiers is not None and "async" in _t(modifiers).split(),
    )


def _members(node: Any) -> list[Member]:
    body = node.child_by_field_name("body")
    out: list[Member] = []
    if body is None:
        return out
    if node.type == "enum_item":
        for v in body.named_children:
            if v.type == "enum_variant":
                vbody = v.child_by_field_name("body")
                out.append(
                    Member(_t(v.child_by_field_name("name")), _clip(_t(vbody)) or None, "pub")
                )
        return out
    if body.type == "field_declaration_list":
        for f in body.named_children:
            if f.type == "field_declaration":
                out.append(
                    Member(
                        _t(f.child_by_field_name("name")),
                        _clip(_t(f.child_by_field_name("type"))),
                        _visibility(f) or PRIVATE,
                    )
                )
    elif body.type == "ordered_field_declaration_list":
        idx = 0
        pending: str | None = None  # ``pub`` of ``struct A(pub u8, u16)`` precedes its type
        for f in body.named_children:
            if f.type == "visibility_modifier":
                pending = " ".join(_t(f).split())
                continue
            out.append(Member(str(idx), _clip(_t(f)), pending or PRIVATE))
            pending = None
            idx += 1
    return out


def _macro_problems(
    attr_items: list[Any], path: str, item_name: str | None
) -> tuple[list[str], list[Unresolved]]:
    attrs: list[str] = []
    problems: list[Unresolved] = []
    for attr_item in attr_items:
        name, attr = _attr_name(attr_item)
        if attr is None:
            continue
        text = _clip(_t(attr))
        attrs.append(text)
        root = name.split("::")[0]
        if name == "derive":
            args = attr.child_by_field_name("arguments")
            problems.append(
                Unresolved(
                    file=path,
                    line=_line(attr_item),
                    name=item_name,
                    macro=f"derive{_clip(_t(args))}" if args is not None else "derive",
                    reason="derive-generated impls are not listed",
                )
            )
        elif name not in _INERT_ATTRIBUTES and root not in ("clippy", "rustfmt"):
            problems.append(
                Unresolved(
                    file=path,
                    line=_line(attr_item),
                    name=item_name,
                    macro=f"#[{name}]",
                    reason="attribute macro may generate or rewrite this item",
                )
            )
    return attrs, problems


def surface(ws: RustWorkspace) -> tuple[list[ApiSymbol], list[Unresolved]]:
    symbols: list[ApiSymbol] = []
    problems: list[Unresolved] = []
    for mod_path in sorted(ws.modules):
        module = ws.modules[mod_path]
        _surface_items(module.node, module.file, mod_path, symbols, problems)
    return symbols, problems


def _surface_items(
    container: Any,
    path: str,
    mod_path: str,
    symbols: list[ApiSymbol],
    problems: list[Unresolved],
) -> None:
    attrs_of = outer_attributes(container)
    for item in container.named_children:
        kind = item.type
        attrs_here = attrs_of.get(item.id, [])
        if kind == "macro_definition":
            name = _t(item.child_by_field_name("name"))
            exported = any(_attr_name(a)[0] == "macro_export" for a in attrs_here)
            if exported:
                symbols.append(
                    ApiSymbol(
                        id=f"{mod_path}::{name}!",
                        module_path=mod_path,
                        name=f"{name}!",
                        kind="macro",
                        visibility="pub",
                        file=path,
                        line=_line(item),
                    )
                )
            problems.append(
                Unresolved(
                    file=path,
                    line=_line(item),
                    name=name,
                    macro=f"macro_rules! {name}",
                    reason="items generated by invocations of this macro are not listed",
                )
            )
            continue
        invocation = item
        if kind == "expression_statement" and item.named_children:
            invocation = item.named_children[0]
        if invocation.type == "macro_invocation":
            mac = _t(invocation.child_by_field_name("macro"))
            if mac not in ("include", "std::include"):
                problems.append(
                    Unresolved(
                        file=path,
                        line=_line(invocation),
                        macro=f"{mac}!",
                        reason="macro invocation in item position may generate items",
                        expression=_clip(_t(invocation)),
                    )
                )
            continue
        if kind == "impl_item":
            _impl(item, attrs_here, path, mod_path, symbols, problems)
            continue
        if kind not in _ITEM_KINDS:
            continue
        vis = _visibility(item)
        name = _t(item.child_by_field_name("name"))
        attrs, macro_problems = _macro_problems(attrs_here, path, f"{mod_path}::{name}")
        problems.extend(macro_problems)
        if vis is None:
            continue
        symbol = ApiSymbol(
            id=f"{mod_path}::{name}",
            module_path=mod_path,
            name=name,
            kind=_ITEM_KINDS[kind],
            visibility=vis,
            file=path,
            line=_line(item),
            attributes=attrs,
        )
        if kind == "function_item":
            symbol.signature = _signature(item)
        elif kind in ("struct_item", "enum_item", "union_item"):
            symbol.members = _members(item)
            generics = _t(item.child_by_field_name("type_parameters"))
            if generics:
                symbol.signature = Signature(generics=_clip(generics))
        elif kind in ("const_item", "static_item", "type_item"):
            symbol.signature = Signature(returns=_clip(_t(item.child_by_field_name("type"))))
        symbols.append(symbol)
        if kind == "trait_item":
            body = item.child_by_field_name("body")
            for fn in body.named_children if body is not None else []:
                if fn.type in ("function_signature_item", "function_item"):
                    fname = _t(fn.child_by_field_name("name"))
                    symbols.append(
                        ApiSymbol(
                            id=f"{mod_path}::{name}::{fname}",
                            module_path=mod_path,
                            name=fname,
                            kind="method",
                            visibility=vis,
                            file=path,
                            line=_line(fn),
                            signature=_signature(fn),
                            owner=name,
                        )
                    )


def _impl(
    item: Any,
    attrs: list[Any],
    path: str,
    mod_path: str,
    symbols: list[ApiSymbol],
    problems: list[Unresolved],
) -> None:
    type_name = _clip(_t(item.child_by_field_name("type")))
    trait = item.child_by_field_name("trait")
    trait_name = _clip(_t(trait)) if trait is not None else None
    _, macro_problems = _macro_problems(attrs, path, f"impl {type_name}")
    problems.extend(macro_problems)
    body = item.child_by_field_name("body")
    fn_attrs = outer_attributes(body) if body is not None else {}
    for fn in body.named_children if body is not None else []:
        if fn.type == "macro_invocation":
            problems.append(
                Unresolved(
                    file=path,
                    line=_line(fn),
                    macro=f"{_t(fn.child_by_field_name('macro'))}!",
                    reason="macro invocation in an impl block may generate methods",
                    expression=_clip(_t(fn)),
                )
            )
            continue
        if fn.type != "function_item":
            continue
        fname = _t(fn.child_by_field_name("name"))
        attrs_fn, fn_problems = _macro_problems(
            fn_attrs.get(fn.id, []), path, f"{type_name}::{fname}"
        )
        problems.extend(fn_problems)
        if trait_name is not None:
            kind, vis = "impl_fn", "pub"
        else:
            declared = _visibility(fn)
            if declared is None:
                continue
            kind, vis = "method", declared
        symbols.append(
            ApiSymbol(
                id=f"{mod_path}::{type_name}::{fname}"
                if trait_name is None
                else f"{mod_path}::<{type_name} as {trait_name}>::{fname}",
                module_path=mod_path,
                name=fname,
                kind=kind,
                visibility=vis,
                file=path,
                line=_line(fn),
                signature=_signature(fn),
                attributes=attrs_fn,
                owner=type_name,
                trait=trait_name,
            )
        )


def scoped_roots(node: Any) -> list[tuple[list[str], int]]:
    """Outermost qualified paths in code (``crate::x::f``, ``db::connect``), not ``use``."""
    out: list[tuple[list[str], int]] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in ("use_declaration", "mod_item"):
            continue  # use: edges of their own; mod: the nested module's own paths
        if cur.type in ("scoped_identifier", "scoped_type_identifier"):
            parent = cur.parent
            if parent is None or parent.type not in ("scoped_identifier", "scoped_type_identifier"):
                text = re.sub(r"<.*", "", _t(cur)).replace(" ", "")
                segs = [s for s in text.split("::") if s]
                if len(segs) > 1:
                    out.append((segs, _line(cur)))
            continue
        stack.extend(cur.named_children)
    return out
