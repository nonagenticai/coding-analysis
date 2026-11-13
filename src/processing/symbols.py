"""What one source file defines and imports, read from its tree-sitter tree.

coding-gateway validates the function/class references an agent proposes
against the code the caller actually has. The multi-language endpoint cannot
answer that: it returns entity COUNTS, and its TypeScript processor is
regex-only (it misses ``export const f = () => ...`` and class methods). This
module walks the parse tree with an explicit node-type table PER GRAMMAR, so a
name is reported only when the grammar produced a defining node for it, and a
name that is not defined is absent rather than guessed.

Kinds beyond the obvious ones: ``const_function`` is a name bound to a
function value, including one wrapped in calls or type assertions
(``memo(function Row() {})``, ``(() => 1) as Fn``, Go ``var f = func() {}``,
Python ``f = lambda: ...``, C++ ``auto f = [](...) {...};``); ``variable`` is
any other module/namespace-level binding (JS/TS top-level, exported or
``declare`` declarators, Python module assignments, Go top-level var/const,
Rust ``const``/``static``, C++ namespace-scope variables and variable
templates); ``field`` is a class/struct data member with ``parent`` set
(Python class-body assignments incl. annotated ones, JS/TS class fields with
a non-function value, Java/C#/C++ fields, Go named struct fields, Rust
associated consts in ``impl``/``trait``). Callers decide whether a variable or
field satisfies a lookup. Nested namespaces (``a::b``, ``A.B``) are reported
whole AND per segment. A named function expression (``f(function g() {})``)
reports ``g`` as a function wherever it appears. ``concept`` is a C++20
``concept_definition`` (a ``requires`` clause or a concept-constrained template
parameter defines nothing).

Inheritance and re-export facts, so a caller can tell "inherited / re-exported"
from "absent" without guessing:

- ``bases`` (only on class-like symbols, only when non-empty): the base names
  as written, type arguments stripped, qualification kept (``pkg.Base``,
  ``ns::Base``). Python superclasses (keyword args, splats and calls skipped),
  JS/TS ``extends`` + TS ``implements`` and interface ``extends``, Java
  ``extends``/``implements``, C# ``base_list``, C++ ``base_class_clause``
  (access specifiers and ``virtual`` dropped), Go embedded struct fields and
  embedded interface types (``*log.Logger`` -> ``log.Logger``). A base that is
  an expression (``extends Mixin(B)``) is not listed.
- ``impls`` (file level, Rust): ``{type, trait, line}`` per ``impl Trait for
  Type`` anywhere in the file; negative impls are skipped. Rust structs carry
  no ``bases``.
- ``reexports`` (file level): ``{name, from_module, alias, line}``. JS/TS
  top-level ``export {a, b as c} from 'm'``, ``export * [as ns] from 'm'``
  (name ``*``), and ``export {x}`` whose ``from_module`` is where this file
  imported ``x`` from (None when it did not import it); Python module-level
  ``from m import x [as y]`` in a package ``__init__.py`` (needs *file_path*);
  Rust top-level plain ``pub use`` (``pub(crate)`` excluded). Every other
  language returns an empty list.

Grammar quirks handled on purpose:

- C++ ``Widget w(config);`` and ``void p(Foo);`` produce the same tree. A
  body-less declaration counts as a function unless every parameter is a bare
  type name AND the return type is not built-in (see ``_cpp_is_prototype``);
  otherwise it is reported as a variable.
- C++ partial/explicit specialisations (``struct is_ring<Ring<T>> {}``) are
  reported under the template's name, without the argument list.
- C++ ``class WIDGET_API Widget { ... };`` (an export macro the grammar cannot
  expand) parses as a function_definition whose type is the body-less
  ``class WIDGET_API``; it is reported as class ``Widget`` with its members
  (see ``_cpp_macro_class``), including a destructor error recovery broke out.

Known gaps (a name here may be reported absent although it is defined):

- Go embedded struct fields (``*log.Logger``): the name is a type defined
  elsewhere, so it is deliberately not reported as a symbol (it is listed in
  the struct's ``bases``, which is how a caller resolves promoted members);
  re-exports inside nested modules/namespaces (only top-level statements are
  read); fields of anonymous classes
  (JS ``class { x = 1 }`` passed as a value, Java anonymous class bodies);
- names bound only inside function bodies, other than JS ``const f = ...``
  function values and named function expressions, which are reported at any
  depth;
- enum members (Java/C#/C++/TS/Rust variants), C# auto-property backing and
  JS ``this.x = ...`` / Python ``self.x = ...`` instance attributes;
- Python names bound by ``for``/``with``/``global``/walrus or under a
  module-level ``if``/``try`` block; ``require()`` bindings (reported as
  imports instead);
- macro-generated definitions in C++ and Rust, and C++ export-macro classes
  whose head error recovery mangles beyond ``final`` or a base list.

The caller passes the file CONTENT, never a container name: container mode
fuzzy-matches names and can answer about the wrong tree.

Parsing reuses the hunt agent's grammar table and parser cache
(``hunt_agent.hunt``) so the service has one grammar loader. The only addition
is TSX: hunt parses ``.tsx`` with the plain TypeScript grammar, which turns JSX
into ERROR nodes, so ``.tsx`` files here use the same package's
``language_tsx`` factory instead.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import PurePath
from typing import Any

from hunt_agent import hunt

TSX = "tsx"
_TSX_GRAMMAR = ("tree_sitter_typescript", "language_tsx")
SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(sorted({*hunt.GRAMMARS, TSX}))

_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "golang": "go",
    "rs": "rust",
    "c": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "csharp": "c_sharp",
    "c#": "c_sharp",
    "cs": "c_sharp",
}

Symbols = list[dict[str, Any]]
Imports = list[dict[str, Any]]
#: (node, enclosing type name, symbols out, imports out) -> type name for children.
Handler = Callable[[Any, str | None, Symbols, Imports], str | None]


def _unsupported(what: str) -> ValueError:
    return ValueError(f"{what}; supported languages: {', '.join(SUPPORTED_LANGUAGES)}")


def resolve_language(language: str | None, file_path: str | None) -> str:
    """The grammar name for an explicit *language* or a *file_path* extension."""
    is_tsx = bool(file_path) and str(file_path).lower().endswith(".tsx")
    if language:
        key = _ALIASES.get(language.strip().lower(), language.strip().lower())
        if key not in SUPPORTED_LANGUAGES:
            raise _unsupported(f"unsupported language: {language}")
        return TSX if key == "typescript" and is_tsx else key
    if file_path:
        if is_tsx:
            return TSX
        found = hunt.source_language(PurePath(file_path))
        if found is None:
            raise _unsupported(f"unsupported file extension: {file_path}")
        return found[0]
    raise _unsupported("language or file_path is required")


_TSX_PARSER: dict[str, Any] = {}


def _tsx_parser() -> Any | None:
    # Same distribution as hunt's "typescript" row: if hunt cannot load it,
    # neither can we (and a test that disables hunt's parsers disables this).
    if hunt.get_parser("typescript") is None:
        return None
    if TSX not in _TSX_PARSER:
        try:
            from tree_sitter import Language, Parser

            module = importlib.import_module(_TSX_GRAMMAR[0])
            _TSX_PARSER[TSX] = Parser(Language(getattr(module, _TSX_GRAMMAR[1])()))
        except (ImportError, AttributeError, ValueError, TypeError):
            _TSX_PARSER[TSX] = None
    return _TSX_PARSER[TSX]


def _parse(grammar: str, content: bytes) -> Any | None:
    if grammar in hunt.GRAMMARS:
        return hunt.parse_bytes(grammar, content)
    parser = _tsx_parser()
    if parser is None:
        return None
    try:
        return parser.parse(content)
    except (ValueError, RuntimeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _text(node: Any) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", "replace")


def _field(node: Any, name: str) -> str | None:
    return _text(node.child_by_field_name(name)) or None


def _unquote(value: str) -> str:
    return value.strip().strip("\"'`<>")


def _sym(
    out: Symbols, name: str | None, kind: str, node: Any, parent: str | None
) -> dict[str, Any] | None:
    if not name:
        return None
    entry = {
        "name": name,
        "kind": kind,
        "line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "parent": parent,
    }
    out.append(entry)
    return entry


def _strip_generics(text: str) -> str:
    """``ns::Base<int>`` -> ``ns::Base``: whitespace and ``<...>`` removed."""
    out: list[str] = []
    depth = 0
    for ch in "".join(text.split()):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _set_bases(entry: dict[str, Any] | None, bases: list[str | None]) -> None:
    """Attach the written base names to a class-like symbol; only when any."""
    names = [b for b in bases if b]
    if entry is not None and names:
        entry["bases"] = names


def _descendants(node: Any, node_type: str) -> list[Any]:
    found: list[Any] = []
    stack = [node] if node is not None else []
    while stack:
        current = stack.pop()
        if current.type == node_type:
            found.append(current)
        stack.extend(reversed(current.children))
    return found


def _imp(out: Imports, module: str, names: list[str], node: Any) -> None:
    out.append(
        {"module": module, "names": [n for n in names if n], "line": node.start_point[0] + 1}
    )


def _first_descendant(node: Any, types: frozenset[str]) -> Any | None:
    queue = [node] if node is not None else []
    while queue:
        current = queue.pop(0)
        if current.type in types:
            return current
        queue.extend(current.children)
    return None


def _namespace_segments(out: Symbols, full: str, separator: str, node: Any) -> None:
    """``a::b`` / ``A.B``: each segment is a namespace too, so any one is found."""
    segments = [s.strip() for s in full.split(separator)]
    if len(segments) < 2:
        return
    for segment in segments:
        _sym(out, segment.removeprefix("inline").strip(), "namespace", node, None)


def _has_ancestor(node: Any, types: frozenset[str]) -> bool:
    current = node.parent
    while current is not None:
        if current.type in types:
            return True
        current = current.parent
    return False


# ---------------------------------------------------------------------------
# Per-grammar handlers. A function/method resets the enclosing type to None,
# so a function nested in a method is a plain function, not a method.
# ---------------------------------------------------------------------------


def _python_targets(target: Any) -> list[Any]:
    """Identifier nodes an assignment target binds (``a``, ``a, b``, ``(a, b)``)."""
    if target is None:
        return []
    if target.type == "identifier":
        return [target]
    if target.type in ("pattern_list", "tuple_pattern", "list_pattern"):
        return [n for c in target.named_children for n in _python_targets(c)]
    return []  # attribute / subscript targets bind no new name


def _python_assignment(node: Any, parent: str | None, syms: Symbols) -> None:
    """Module-level ``NAME = value`` -> variable (``const_function`` for a
    lambda); class-level ``NAME = value`` / ``NAME: T = value`` / ``NAME: T``
    -> field of the class (``const_function`` for a lambda). Chained
    ``a = b = 1`` binds every target."""
    statement = node.parent
    while statement is not None and statement.type == "assignment":
        statement = statement.parent  # inner link of a chain
    if statement is None or statement.type != "expression_statement":
        return
    scope = statement.parent
    at_module = scope is not None and scope.type == "module"
    in_class = (
        scope is not None
        and scope.type == "block"
        and scope.parent is not None
        and scope.parent.type == "class_definition"
    )
    if not (at_module or in_class):
        return
    value = node.child_by_field_name("right")
    while value is not None and value.type == "assignment":
        value = value.child_by_field_name("right")
    is_lambda = value is not None and value.type == "lambda"
    kind = "const_function" if is_lambda else ("field" if in_class else "variable")
    for name in _python_targets(node.child_by_field_name("left")):
        _sym(syms, _text(name), kind, node, parent if in_class else None)


def _python_bases(superclasses: Any) -> list[str | None]:
    """``class A(pkg.Base, Generic[T], metaclass=M)`` -> ``pkg.Base``, ``Generic``.
    Keyword arguments, splats and calls name no base and are skipped."""
    bases: list[str | None] = []
    for child in superclasses.named_children if superclasses is not None else []:
        if child.type == "subscript":
            child = child.child_by_field_name("value")
        if child is not None and child.type in ("identifier", "attribute"):
            bases.append("".join(_text(child).split()))
    return bases


def _python(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t == "assignment":
        _python_assignment(node, parent, syms)
        return parent
    if t == "type_alias_statement":
        left = _first_descendant(node.child_by_field_name("left"), frozenset({"identifier"}))
        _sym(syms, _text(left) or None, "type", node, parent)
        return parent
    if t == "function_definition":
        _sym(syms, _field(node, "name"), "method" if parent else "function", node, parent)
        return None
    if t == "class_definition":
        name = _field(node, "name")
        entry = _sym(syms, name, "class", node, parent)
        _set_bases(entry, _python_bases(node.child_by_field_name("superclasses")))
        return name or parent
    if t == "import_statement":
        for child in node.children_by_field_name("name"):
            if child.type == "aliased_import":
                _imp(imps, _field(child, "name") or "", [_field(child, "alias") or ""], node)
            else:
                _imp(imps, _text(child), [], node)
    elif t == "import_from_statement":
        names = [
            (_field(c, "name") if c.type == "aliased_import" else _text(c)) or ""
            for c in node.children_by_field_name("name")
        ]
        if any(c.type == "wildcard_import" for c in node.children):
            names.append("*")
        _imp(imps, _field(node, "module_name") or "", names, node)
    return parent


_JS_TYPES = {
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "class": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "internal_module": "namespace",
    "module": "namespace",
}
_JS_CONTAINERS = frozenset({"class_declaration", "abstract_class_declaration", "class"})
_JS_FUNCTIONS = frozenset(
    {"function_declaration", "generator_function_declaration", "function_signature"}
)
_JS_METHODS = frozenset({"method_definition", "method_signature", "abstract_method_signature"})
_JS_FUNCTION_VALUES = frozenset(
    {"arrow_function", "function_expression", "function", "generator_function"}
)


_JS_VALUE_WRAPPERS = frozenset(
    {"satisfies_expression", "as_expression", "parenthesized_expression", "non_null_expression"}
)


def _js_binding_names(target: Any) -> list[str]:
    if target is None:
        return []
    if target.type in ("identifier", "shorthand_property_identifier_pattern"):
        return [_text(target)]
    names: list[str] = []
    if target.type in ("object_pattern", "array_pattern"):
        for child in target.named_children:
            if child.type == "pair_pattern":
                names.extend(_js_binding_names(child.child_by_field_name("value")))
            elif child.type in ("assignment_pattern", "object_assignment_pattern"):
                names.extend(_js_binding_names(child.child_by_field_name("left")))
            elif child.type == "rest_pattern":
                names.extend(_js_binding_names(next(iter(child.named_children), None)))
            else:
                names.extend(_js_binding_names(child))
    return names


def _js_unwrap(value: Any) -> Any:
    """*value* without ``(...)``, ``as T``, ``satisfies T`` and ``!`` around it."""
    while value is not None and value.type in _JS_VALUE_WRAPPERS:
        value = next(iter(value.named_children), None)
    return value


def _js_function_values(value: Any) -> list[Any]:
    """The function/class nodes *value* evaluates to or wraps.

    ``() => 1`` is itself one; ``memo(function Row() {})``,
    ``forwardRef((p, ref) => ...)`` and ``React.memo(withX(() => 1))`` wrap one
    in a call argument, at any call depth. Anything else yields nothing.
    """
    value = _js_unwrap(value)
    if value is None:
        return []
    if value.type in _JS_FUNCTION_VALUES or value.type == "class":
        return [value]
    if value.type != "call_expression":
        return []
    args = value.child_by_field_name("arguments")
    found: list[Any] = []
    for arg in args.named_children if args is not None else []:
        found.extend(_js_function_values(arg))
    return found


def _js_is_require(value: Any) -> bool:
    return (
        value is not None
        and value.type == "call_expression"
        and _field(value, "function") == "require"
    )


def _js_top_level(declarator: Any) -> bool:
    """Whether a ``variable_declarator`` is at module (or namespace) scope."""
    scope = declarator.parent.parent if declarator.parent is not None else None
    while scope is not None and scope.type in ("export_statement", "ambient_declaration"):
        scope = scope.parent  # export const x / declare const x
    if scope is None:
        return False
    if scope.type == "program":
        return True
    return (
        scope.type == "statement_block"
        and scope.parent is not None
        and scope.parent.type in ("internal_module", "module")
    )


def _js_key(node: Any) -> str | None:
    """A property/method key as written: quotes stripped, computed keys only
    when they are a plain string literal (``['x']``); anything else is None."""
    if node is None:
        return None
    if node.type == "string":
        return _unquote(_text(node)) or None
    if node.type == "computed_property_name":
        inner = node.named_children
        if len(inner) == 1 and inner[0].type == "string":
            return _unquote(_text(inner[0])) or None
        return None
    return _text(node) or None


def _js_declarator_name(node: Any) -> str | None:
    """The identifier a ``variable_declarator`` binds, when *node* is its value."""
    owner = node.parent
    if owner is None or owner.type != "variable_declarator":
        return None
    target = owner.child_by_field_name("name")
    return _text(target) if target is not None and target.type == "identifier" else None


_JS_BASE_NAMES = frozenset(
    {"identifier", "type_identifier", "member_expression", "nested_type_identifier"}
)


def _js_base_name(node: Any) -> str | None:
    """A heritage entry as written (``ns.Base``, ``I``); ``Base<T>`` loses its
    arguments; a call (``Mixin(B)``) or other expression names no base."""
    if node is not None and node.type == "generic_type":
        node = node.child_by_field_name("name")
    if node is None or node.type not in _JS_BASE_NAMES:
        return None
    return "".join(_text(node).split()) or None


def _js_bases(node: Any) -> list[str | None]:
    """JS ``class_heritage`` holds the expression; TS splits it into
    ``extends_clause`` (field ``value``) and ``implements_clause``; a TS
    interface lists its bases in ``extends_type_clause`` (field ``type``)."""
    bases: list[str | None] = []
    for child in node.named_children:
        if child.type == "class_heritage":
            for clause in child.named_children:
                if clause.type == "extends_clause":
                    values = clause.children_by_field_name("value")
                elif clause.type == "implements_clause":
                    values = clause.named_children
                else:
                    values = [clause]
                bases.extend(_js_base_name(v) for v in values)
        elif child.type == "extends_type_clause":
            bases.extend(_js_base_name(v) for v in child.children_by_field_name("type"))
    return bases


def _js(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t in _JS_TYPES:
        name = _field(node, "name")
        if name is None and t == "class":
            name = _js_declarator_name(node)  # const C = class { ... }
        if name is None and node.parent is not None and node.parent.type == "export_statement":
            name = "default"
        kind = _JS_TYPES[t]
        bases = _js_bases(node) if kind in ("class", "interface") else []
        _set_bases(_sym(syms, _unquote(name) if name else None, kind, node, parent), bases)
        if kind == "namespace" and name:
            _namespace_segments(syms, _unquote(name), ".", node)
        if t == "class" and _field(node, "name"):
            _set_bases(_sym(syms, _js_declarator_name(node), "class", node, parent), bases)
        if t in _JS_CONTAINERS or kind == "interface":
            return name
        return parent
    if t in _JS_FUNCTIONS:
        _sym(syms, _field(node, "name"), "function", node, None)
        return None
    if t in _JS_METHODS:
        owner = node.parent
        if owner is not None and owner.type == "object":
            # ``{ formatItem() {} }``: same parent as a pair-valued function.
            parent = _js_declarator_name(owner)
        _sym(syms, _js_key(node.child_by_field_name("name")), "method", node, parent)
        return None
    if t in ("public_field_definition", "field_definition"):
        value = _js_unwrap(node.child_by_field_name("value"))
        key = node.child_by_field_name("name") or node.child_by_field_name("property")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            _sym(syms, _js_key(key), "method", node, parent)
        elif parent:
            _sym(syms, _js_key(key), "field", node, parent)
    elif t in _JS_FUNCTION_VALUES and t != "arrow_function":
        # A named function expression is a definition wherever it appears:
        # asyncHandler(async function createUser() {}), let f = function g() {}.
        _sym(syms, _field(node, "name"), "function", node, None)
    elif t == "pair":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            owner = node.parent
            obj_parent = _js_declarator_name(owner) if owner is not None else None
            _sym(syms, _js_key(node.child_by_field_name("key")), "method", node, obj_parent)
    elif t == "assignment_expression":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is not None and left.type == "member_expression" and right is not None:
            if right.type in _JS_FUNCTION_VALUES:
                if _text(left) != "module.exports":  # its named value reports itself
                    _sym(syms, _field(left, "property"), "function", node, None)
    elif t == "variable_declarator":
        target = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        functions = _js_function_values(value)
        if target is not None and target.type == "identifier" and functions:
            # memo(function UserRow() {}): the inner name reports itself.
            _sym(syms, _text(target), "const_function", node, None)
        elif _js_top_level(node) and not _js_is_require(_js_unwrap(value)):
            for name in _js_binding_names(target):
                _sym(syms, name, "variable", node, None)
    elif t == "export_statement":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            _sym(syms, "default", "function", node, None)
    elif t == "import_statement":
        names: list[str] = []
        source = _field(node, "source") or ""
        for clause in node.named_children:
            if clause.type == "import_require_clause":
                source = _field(clause, "source") or source
                names.extend(_text(c) for c in clause.named_children if c.type == "identifier")
            if clause.type != "import_clause":
                continue
            for child in clause.named_children:
                if child.type == "identifier":
                    names.append(_text(child))
                elif child.type == "namespace_import":
                    names.append("*")
                elif child.type == "named_imports":
                    names.extend(
                        _field(spec, "name") or ""
                        for spec in child.named_children
                        if spec.type == "import_specifier"
                    )
        _imp(imps, _unquote(source), names, node)
    elif t == "call_expression" and _field(node, "function") == "require":
        args = node.child_by_field_name("arguments")
        if args is not None and len(args.named_children) == 1:
            arg = args.named_children[0]
            if arg.type in ("string", "template_string"):
                decl = node.parent
                bound = (
                    _js_binding_names(decl.child_by_field_name("name"))
                    if decl is not None and decl.type == "variable_declarator"
                    else []
                )
                _imp(imps, _unquote(_text(arg)), bound, node)
    return parent


def _go_type_name(node: Any) -> str | None:
    """``log.Logger`` / ``Base`` / ``G[int]`` -> the type name as written, no
    type arguments; the ``*`` of ``*log.Logger`` is an anonymous token."""
    if node is not None and node.type == "pointer_type":
        node = next(iter(node.named_children), None)
    if node is not None and node.type == "generic_type":
        node = node.child_by_field_name("type")
    if node is None or node.type not in ("type_identifier", "qualified_type"):
        return None
    return "".join(_text(node).split()) or None


def _go_bases(body: Any) -> list[str | None]:
    """Embedded struct fields (a field_declaration with no ``name``) and
    embedded interface types (a ``type_elem`` holding one type, not a union
    or ``~T`` approximation)."""
    if body is None:
        return []
    if body.type == "struct_type":
        fields = next((c for c in body.named_children if c.type == "field_declaration_list"), None)
        return [
            _go_type_name(f.child_by_field_name("type"))
            for f in (fields.named_children if fields is not None else [])
            if f.type == "field_declaration" and f.child_by_field_name("name") is None
        ]
    if body.type == "interface_type":
        return [
            _go_type_name(e.named_children[0])
            for e in body.named_children
            if e.type == "type_elem" and len(e.named_children) == 1
        ]
    return []


def _go(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t == "function_declaration":
        _sym(syms, _field(node, "name"), "function", node, None)
        return None
    if t == "method_declaration":
        receiver = _first_descendant(
            node.child_by_field_name("receiver"), frozenset({"type_identifier"})
        )
        _sym(syms, _field(node, "name"), "method", node, _text(receiver) or None)
        return None
    if t in ("type_spec", "type_alias"):
        name = _field(node, "name")
        body = node.child_by_field_name("type")
        body_type = body.type if body is not None else ""
        kind = {"struct_type": "struct", "interface_type": "interface"}.get(body_type, "type")
        _set_bases(_sym(syms, name, kind, node, parent), _go_bases(body))
        return name
    if t in ("var_spec", "const_spec"):
        if not _has_ancestor(node, frozenset({"block", "func_literal"})):
            value = node.child_by_field_name("value")
            values = value.named_children if value is not None else []
            for index, name in enumerate(node.children_by_field_name("name")):
                func = index < len(values) and values[index].type == "func_literal"
                _sym(syms, _text(name), "const_function" if func else "variable", node, None)
    elif t == "field_declaration" and parent:
        # Embedded fields (``*log.Logger``) are skipped: the name is a type
        # defined elsewhere, and reporting it would vouch for that type here.
        # They are listed in the struct's ``bases`` instead (``_go_bases``).
        for name in node.children_by_field_name("name"):
            _sym(syms, _text(name), "field", node, parent)
    elif t in ("method_elem", "method_spec") and parent:
        _sym(syms, _field(node, "name"), "method", node, parent)
    elif t == "import_spec":
        _imp(imps, _unquote(_field(node, "path") or ""), [_field(node, "name") or ""], node)
    return parent


_JAVA_TYPES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "class",
    "annotation_type_declaration": "interface",
}
_JAVA_METHODS = frozenset(
    {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
)


_JAVA_BASE_NAMES = frozenset({"type_identifier", "scoped_type_identifier", "generic_type"})


def _java_bases(node: Any) -> list[str | None]:
    """``extends`` (field ``superclass``), ``implements`` (``super_interfaces``)
    and an interface's ``extends`` (``extends_interfaces``), each a type_list."""
    types: list[Any] = []
    superclass = node.child_by_field_name("superclass")
    if superclass is not None:
        types.extend(superclass.named_children)
    for child in node.named_children:
        if child.type in ("super_interfaces", "extends_interfaces"):
            for type_list in child.named_children:
                if type_list.type == "type_list":
                    types.extend(type_list.named_children)
    return [_strip_generics(_text(n)) for n in types if n.type in _JAVA_BASE_NAMES]


def _java(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t in _JAVA_TYPES:
        name = _field(node, "name")
        _set_bases(_sym(syms, name, _JAVA_TYPES[t], node, parent), _java_bases(node))
        if t == "record_declaration":
            # record P(int id): each component has an implicit accessor id().
            params = node.child_by_field_name("parameters")
            for param in params.named_children if params is not None else []:
                _sym(syms, _field(param, "name"), "method", param, name)
        return name
    if t in _JAVA_METHODS:
        _sym(syms, _field(node, "name"), "method", node, parent)
        return None
    if t == "annotation_type_element_declaration":
        _sym(syms, _field(node, "name"), "method", node, parent)
    elif t in ("field_declaration", "constant_declaration") and parent:
        for declarator in node.children_by_field_name("declarator"):
            _sym(syms, _field(declarator, "name"), "field", declarator, parent)
    elif t == "import_declaration":
        path = next(
            (c for c in node.named_children if c.type in ("scoped_identifier", "identifier")),
            None,
        )
        dotted = _text(path)
        if any(c.type == "asterisk" for c in node.named_children):
            _imp(imps, dotted, ["*"], node)
        else:
            _imp(imps, dotted, [dotted.rsplit(".", 1)[-1]], node)
    return parent


_RUST_TYPES = {
    "struct_item": "struct",
    "union_item": "struct",
    "enum_item": "enum",
    "type_item": "type",
    "mod_item": "module",
}


def _rust_type_name(node: Any) -> str | None:
    while node is not None and node.type in (
        "generic_type",
        "reference_type",
        "scoped_type_identifier",
    ):
        node = node.child_by_field_name("name" if node.type == "scoped_type_identifier" else "type")
    return _text(node) or None


def _rust_use(arg: Any) -> tuple[str, list[str]]:
    t = arg.type
    if t == "scoped_identifier":
        return _field(arg, "path") or "", [_field(arg, "name") or ""]
    if t in ("scoped_use_list", "use_list"):
        items = arg.child_by_field_name("list") if t == "scoped_use_list" else arg
        names: list[str] = []
        for child in items.named_children if items is not None else []:
            if child.type in ("identifier", "self"):
                names.append(_text(child))
            else:
                module, sub = _rust_use(child)
                names.extend(sub or [module])
        return (_field(arg, "path") or "") if t == "scoped_use_list" else "", names
    if t == "use_as_clause":
        module, _ = _rust_use(arg.child_by_field_name("path"))
        return module, [_field(arg, "alias") or ""]
    if t == "use_wildcard":
        return _text(arg).removesuffix("*").removesuffix("::"), ["*"]
    return _text(arg), []


def _rust(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t in ("function_item", "function_signature_item"):
        _sym(syms, _field(node, "name"), "method" if parent else "function", node, parent)
        return None
    if t == "impl_item":
        return _rust_type_name(node.child_by_field_name("type"))
    if t == "trait_item":
        name = _field(node, "name")
        _sym(syms, name, "trait", node, parent)
        return name
    if t in _RUST_TYPES:
        _sym(syms, _field(node, "name"), _RUST_TYPES[t], node, parent)
        return None if t == "mod_item" else parent
    if t in ("const_item", "static_item"):
        if parent:  # impl / trait body: an associated constant
            _sym(syms, _field(node, "name"), "field", node, parent)
        elif not _has_ancestor(node, frozenset({"function_item", "block"})):
            _sym(syms, _field(node, "name"), "variable", node, None)
    elif t == "associated_type":
        _sym(syms, _field(node, "name"), "type", node, parent)
    elif t == "macro_definition":
        _sym(syms, _field(node, "name"), "macro", node, None)
    elif t == "use_declaration":
        arg = node.child_by_field_name("argument")
        if arg is not None:
            module, names = _rust_use(arg)
            _imp(imps, module, names, node)
    return parent


_CPP_TYPES = {
    "class_specifier": "class",
    "struct_specifier": "struct",
    "union_specifier": "struct",
    "enum_specifier": "enum",
}
_CPP_DECLARATOR_WRAPPERS = frozenset(
    {"pointer_declarator", "reference_declarator", "parenthesized_declarator"}
)
_CPP_TYPEDEF_WRAPPERS = _CPP_DECLARATOR_WRAPPERS | {"array_declarator", "function_declarator"}
_CPP_VARIABLE_WRAPPERS = _CPP_DECLARATOR_WRAPPERS | {"array_declarator", "init_declarator"}
_CPP_VARIABLE_NAMES = frozenset({"identifier", "field_identifier", "qualified_identifier"})
#: Return types that cannot be the type of a constructed object, so a
#: declaration ``int f(Foo);`` is a prototype, never ``Widget w(config);``.
_CPP_BUILTIN_TYPES = frozenset({"primitive_type", "sized_type_specifier"})


def _cpp_unwrap(node: Any, wrappers: frozenset[str]) -> Any | None:
    while node is not None and node.type in wrappers:
        node = node.child_by_field_name("declarator") or (
            node.named_children[-1] if node.named_children else None
        )
    return node


def _cpp_callable(declarator: Any) -> tuple[Any, Any, bool] | None:
    """``(function_declarator, name node, is_function)`` for *declarator*.

    ``void (*getfp())(int)`` is a function returning a function pointer: the
    outer function_declarator wraps a parenthesized pointer to the real one.
    ``void (*cb)(int)`` wraps a pointer to a plain name: a pointer variable,
    so ``is_function`` is False.
    """
    func = _cpp_unwrap(declarator, _CPP_DECLARATOR_WRAPPERS)
    if func is None or func.type != "function_declarator":
        return None
    inner = func.child_by_field_name("declarator")
    if inner is None or inner.type not in _CPP_DECLARATOR_WRAPPERS:
        return func, inner, True
    unwrapped = _cpp_unwrap(inner, _CPP_DECLARATOR_WRAPPERS)
    if unwrapped is not None and unwrapped.type == "function_declarator":
        return _cpp_callable(unwrapped)
    return func, unwrapped, False


def _cpp_is_prototype(node: Any, func: Any) -> bool:
    """Whether a body-less ``declaration`` declares a function, not an object.

    ``Widget w(config);`` parses as a function_declarator whose parameter is a
    bare type_identifier ``config``; so does ``void p(Foo);``. A parameter
    with a declarator, qualifier or built-in type is unambiguous; when every
    parameter is a bare type name, only a built-in return type (``void``,
    ``int``) settles it, since no object has that type and a constructor call.
    """
    param_list = func.child_by_field_name("parameters")
    if param_list is None:
        return False
    params = [c for c in param_list.named_children if c.type != "comment"]
    bare = 0
    for param in params:
        if param.type == "variadic_parameter_declaration":
            continue
        if param.type not in ("parameter_declaration", "optional_parameter_declaration"):
            return False  # a literal or expression: a constructor call
        if (
            param.child_by_field_name("declarator") is None
            and len(param.named_children) == 1
            and param.named_children[0].type == "type_identifier"
        ):
            bare += 1
    if params and bare == len(params):
        type_node = node.child_by_field_name("type")
        return type_node is not None and type_node.type in _CPP_BUILTIN_TYPES
    return True


def _cpp_name(node: Any) -> tuple[str | None, str | None]:
    owner = None
    while node is not None and node.type == "qualified_identifier":
        scope = node.child_by_field_name("scope")
        if scope is not None:
            owner = _field(scope, "name") if scope.type == "template_type" else _text(scope)
        node = node.child_by_field_name("name")
    if node is not None and node.type == "template_function":
        node = node.child_by_field_name("name")
    return (_text(node) or None), owner


def _cpp_variable(
    node: Any, declarator: Any, parent: str | None, syms: Symbols, is_member: bool
) -> None:
    """``int g = 0;`` / ``int *p, a[3];`` / ``int Foo::count = 0;``: a
    variable at namespace scope, a field of *parent* (or of the qualifying
    class) at class scope; ``auto f = [](int a) {...};`` is a const_function."""
    value = (
        declarator.child_by_field_name("value") if declarator.type == "init_declarator" else None
    )
    if value is None:
        value = node.child_by_field_name("default_value")
    target = _cpp_unwrap(declarator, _CPP_VARIABLE_WRAPPERS)
    if target is None or target.type not in _CPP_VARIABLE_NAMES:
        return
    name, owner = _cpp_name(target)
    owner = owner or parent
    if value is not None and value.type == "lambda_expression":
        _sym(syms, name, "const_function", node, owner)
    else:
        _sym(syms, name, "field" if owner or is_member else "variable", node, owner)


def _cpp_macro_class(node: Any) -> tuple[str, str] | None:
    """``(name, kind)`` when a ``function_definition`` is really an export-macro
    class: ``class WIDGET_API Widget { ... };`` parses as a function whose
    return type is a body-less ``class WIDGET_API``, whose declarator is the
    identifier ``Widget`` and whose "body" holds the members. With ``final``
    the grammar makes ``final`` the declarator and buries the name in ERROR.
    """
    if node is None or node.type != "function_definition":
        return None
    spec = node.child_by_field_name("type")
    if spec is None or spec.type not in _CPP_TYPES or spec.type == "enum_specifier":
        return None
    if spec.child_by_field_name("body") is not None:
        return None
    declarator = node.child_by_field_name("declarator")
    if declarator is None or declarator.type != "identifier":
        return None
    name = _text(declarator)
    if name == "final":
        error = next((c for c in node.children if c.type == "ERROR"), None)
        name = _text(_first_descendant(error, frozenset({"identifier"})))
    return (name, _CPP_TYPES[spec.type]) if name else None


def _cpp_is_local(node: Any) -> bool:
    """Whether *node* sits in a function body (a compound_statement that is not
    an export-macro class body)."""
    current = node.parent
    while current is not None:
        if current.type == "compound_statement":
            return _cpp_macro_class(current.parent) is None
        current = current.parent
    return False


def _cpp_recovered_destructor(node: Any, owner: str) -> bool:
    """Whether *node* is ``~Owner()`` that error recovery broke out of an
    export-macro class body: ``virtual ~Widget();`` arrives either as an
    ERROR holding a function_declarator over a destructor_name, or as the
    expression statement ``~Widget();``."""
    if node.type == "ERROR":
        return any(
            child.type == "function_declarator"
            and _text(child.child_by_field_name("declarator")) == f"~{owner}"
            for child in node.named_children
        )
    expr = next(iter(node.named_children), None)
    if expr is None or expr.type != "unary_expression" or _field(expr, "operator") != "~":
        return False
    call = expr.child_by_field_name("argument")
    return call is not None and call.type == "call_expression" and _field(call, "function") == owner


_CPP_BASE_NAMES = frozenset({"type_identifier", "qualified_identifier", "template_type"})


def _cpp_bases(node: Any) -> list[str | None]:
    """``: public virtual ns::Base<int>, Other`` -> ``ns::Base``, ``Other``.
    Access specifiers are named children of their own; ``virtual`` is anonymous."""
    clause = next((c for c in node.named_children if c.type == "base_class_clause"), None)
    return [
        _strip_generics(_text(c))
        for c in (clause.named_children if clause is not None else [])
        if c.type in _CPP_BASE_NAMES
    ]


def _cpp(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    macro_class = _cpp_macro_class(node)
    if macro_class is not None:
        _sym(syms, macro_class[0], macro_class[1], node, parent)
        return macro_class[0]
    if t in ("function_definition", "declaration", "field_declaration"):
        if t == "declaration" and _cpp_is_local(node):
            return parent  # a local declaration, never a prototype
        for declarator in node.children_by_field_name("declarator"):
            whole = declarator
            if declarator.type == "init_declarator":
                # virtual void render() const = 0;  /  void (*cb)() = nullptr;
                declarator = declarator.child_by_field_name("declarator")
            found = _cpp_callable(declarator)
            if found is None:
                if t != "function_definition":
                    _cpp_variable(node, whole, parent, syms, t == "field_declaration")
                continue
            func, name_node, is_function = found
            name, owner = _cpp_name(name_node)
            owner = owner or parent
            if not is_function:
                _sym(syms, name, "field" if t == "field_declaration" else "variable", node, owner)
            elif t != "declaration" or _cpp_is_prototype(node, func):
                _sym(syms, name, "method" if owner else "function", node, owner)
            else:  # Widget w(config);  an object constructed, not a prototype
                _sym(syms, name, "field" if owner else "variable", node, owner)
        return None if t == "function_definition" else parent
    if t in _CPP_TYPES and node.child_by_field_name("body") is not None:
        name_node = node.child_by_field_name("name")
        if name_node is not None and name_node.type == "template_type":
            name_node = name_node.child_by_field_name("name")  # is_ring<Ring<T>>
        name = _text(name_node) or None
        entry = _sym(syms, name, _CPP_TYPES[t], node, parent)
        if t != "enum_specifier":
            _set_bases(entry, _cpp_bases(node))
        return name if t != "enum_specifier" else parent
    if t == "concept_definition":
        # template<typename T> concept Addable = ...;  A requires clause or a
        # concept-constrained parameter is a different node and defines nothing.
        _sym(syms, _field(node, "name"), "concept", node, parent)
        return parent
    if t in ("expression_statement", "ERROR") and parent and not _cpp_is_local(node):
        if _cpp_recovered_destructor(node, parent):
            _sym(syms, f"~{parent}", "method", node, parent)
    elif t == "namespace_definition":
        name = _field(node, "name")
        _sym(syms, name, "namespace", node, None)
        if name:
            _namespace_segments(syms, name, "::", node)
    elif t == "alias_declaration":
        _sym(syms, _field(node, "name"), "type", node, parent)
    elif t == "type_definition":
        for declarator in node.children_by_field_name("declarator"):
            _sym(syms, _text(_cpp_unwrap(declarator, _CPP_TYPEDEF_WRAPPERS)), "type", node, parent)
    elif t in ("preproc_def", "preproc_function_def"):
        _sym(syms, _field(node, "name"), "macro", node, None)
    elif t == "preproc_include":
        _imp(imps, _unquote(_field(node, "path") or ""), [], node)
    return parent


_CS_TYPES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "struct_declaration": "struct",
    "enum_declaration": "enum",
    "record_declaration": "class",
    "record_struct_declaration": "struct",
    "delegate_declaration": "type",
}
_CS_METHODS = frozenset({"method_declaration", "constructor_declaration", "destructor_declaration"})
_CS_NAMES = frozenset({"identifier", "qualified_name", "alias_qualified_name", "generic_name"})


def _cs_bases(node: Any) -> list[str | None]:
    """``: Ns.Base<T>, IFoo`` -> ``Ns.Base``, ``IFoo``; a record's
    ``B(X)`` primary-constructor base contributes its type ``B``."""
    base_list = next((c for c in node.named_children if c.type == "base_list"), None)
    bases: list[str | None] = []
    for child in base_list.named_children if base_list is not None else []:
        if child.type == "primary_constructor_base_type":
            child = child.child_by_field_name("type")
        if child is not None and child.type in _CS_NAMES:
            bases.append(_strip_generics(_text(child)))
    return bases


def _c_sharp(node: Any, parent: str | None, syms: Symbols, imps: Imports) -> str | None:
    t = node.type
    if t in _CS_TYPES:
        name = _field(node, "name")
        entry = _sym(syms, name, _CS_TYPES[t], node, parent)
        if t not in ("enum_declaration", "delegate_declaration"):
            _set_bases(entry, _cs_bases(node))
        if t in ("record_declaration", "record_struct_declaration"):
            # record ProductDto(int Id, string Name): positional properties.
            params = next((c for c in node.named_children if c.type == "parameter_list"), None)
            for param in params.named_children if params is not None else []:
                if param.type == "parameter":
                    _sym(syms, _field(param, "name"), "property", param, name)
        return name if t != "enum_declaration" else parent
    if t in _CS_METHODS:
        name = _field(node, "name")
        if t == "destructor_declaration" and name:
            name = f"~{name}"
        _sym(syms, name, "method", node, parent)
        return None
    if t == "local_function_statement":
        _sym(syms, _field(node, "name"), "function", node, None)
        return None
    if t in ("property_declaration", "event_declaration"):
        kind = "property" if t == "property_declaration" else "event"
        _sym(syms, _field(node, "name"), kind, node, parent)
        return None
    if t == "indexer_declaration":
        _sym(syms, "this", "indexer", node, parent)
        return None
    if t == "operator_declaration":
        _sym(syms, f"operator {_field(node, 'operator') or ''}".strip(), "operator", node, parent)
        return None
    if t == "conversion_operator_declaration":
        _sym(syms, f"operator {_field(node, 'type') or ''}".strip(), "operator", node, parent)
        return None
    if t == "field_declaration" and parent:
        for declaration in node.named_children:
            if declaration.type != "variable_declaration":
                continue
            for var in declaration.named_children:
                if var.type == "variable_declarator":
                    _sym(syms, _field(var, "name"), "field", var, parent)
        return parent
    if t == "event_field_declaration":
        for declaration in node.named_children:
            if declaration.type != "variable_declaration":
                continue
            for var in declaration.named_children:
                if var.type == "variable_declarator":
                    _sym(syms, _field(var, "name"), "event", var, parent)
        return parent
    if t in ("namespace_declaration", "file_scoped_namespace_declaration"):
        name = _field(node, "name")
        _sym(syms, name, "namespace", node, None)
        if name:
            _namespace_segments(syms, name, ".", node)
    elif t == "using_directive":
        parts = [_text(c) for c in node.named_children if c.type in _CS_NAMES]
        if parts:
            _imp(imps, parts[-1], parts[:1] if len(parts) > 1 else [], node)
    return parent


_HANDLERS: dict[str, Handler] = {
    "python": _python,
    "javascript": _js,
    "typescript": _js,
    TSX: _js,
    "go": _go,
    "java": _java,
    "rust": _rust,
    "cpp": _cpp,
    "c_sharp": _c_sharp,
}


# ---------------------------------------------------------------------------
# File-level facts: re-exports (JS/TS, Python package __init__, Rust pub use)
# and Rust ``impl Trait for Type``. Read from top-level nodes only.
# ---------------------------------------------------------------------------

Reexports = list[dict[str, Any]]


def _reexport(out: Reexports, name: str, module: str | None, alias: str | None, node: Any) -> None:
    if name:
        out.append(
            {
                "name": name,
                "from_module": module,
                "alias": alias or None,
                "line": node.start_point[0] + 1,
            }
        )


def _js_import_bindings(root: Any) -> dict[str, str]:
    """Local name -> module, for every top-level ``import`` in the file."""
    bindings: dict[str, str] = {}
    for statement in root.named_children:
        if statement.type != "import_statement":
            continue
        source = _unquote(_field(statement, "source") or "")
        for clause in statement.named_children:
            if clause.type != "import_clause":
                continue
            for child in clause.named_children:
                if child.type == "identifier":
                    bindings[_text(child)] = source
                elif child.type == "namespace_import":
                    for ident in child.named_children:
                        if ident.type == "identifier":
                            bindings[_text(ident)] = source
                elif child.type == "named_imports":
                    for spec in child.named_children:
                        if spec.type == "import_specifier":
                            local = _field(spec, "alias") or _field(spec, "name")
                            if local:
                                bindings[_unquote(local)] = source
    return bindings


def _js_reexports(root: Any) -> Reexports:
    """``export { a, b as c } from './m'``, ``export * from './m'``,
    ``export * as ns from './m'``, and ``export { x }`` (module = where this
    file imported ``x`` from, None when it did not import it)."""
    out: Reexports = []
    bindings: dict[str, str] | None = None
    for statement in root.named_children:
        if statement.type != "export_statement":
            continue
        source_node = statement.child_by_field_name("source")
        source = _unquote(_text(source_node)) if source_node is not None else None
        clause = next((c for c in statement.named_children if c.type == "export_clause"), None)
        star = next((c for c in statement.named_children if c.type == "namespace_export"), None)
        if clause is not None:
            if source is None and bindings is None:
                bindings = _js_import_bindings(root)
            for spec in clause.named_children:
                if spec.type != "export_specifier":
                    continue
                name = _unquote(_field(spec, "name") or "")
                alias = _unquote(_field(spec, "alias") or "")
                module = source if source is not None else (bindings or {}).get(name)
                _reexport(out, name, module, alias, statement)
        elif star is not None and source is not None:
            alias = next((_text(c) for c in star.named_children), None)
            _reexport(out, "*", source, _unquote(alias or ""), statement)
        elif source is not None and any(c.type == "*" for c in statement.children):
            _reexport(out, "*", source, None, statement)
    return out


def _python_reexports(root: Any, file_path: str | None) -> Reexports:
    """In a package ``__init__``, every module-level ``from m import x``."""
    if not file_path or PurePath(file_path).name not in ("__init__.py", "__init__.pyi"):
        return []
    out: Reexports = []
    for statement in root.named_children:
        if statement.type != "import_from_statement":
            continue
        module = _field(statement, "module_name") or ""
        for child in statement.children_by_field_name("name"):
            if child.type == "aliased_import":
                name = _field(child, "name") or ""
                _reexport(out, name, module, _field(child, "alias"), statement)
            else:
                _reexport(out, _text(child), module, None, statement)
        if any(c.type == "wildcard_import" for c in statement.children):
            _reexport(out, "*", module, None, statement)
    return out


def _rust_join(*parts: str) -> str:
    return "::".join(p for p in parts if p)


def _rust_use_items(arg: Any, prefix: str) -> list[tuple[str, str | None, str | None]]:
    """``(name, module, alias)`` for each item one ``use`` argument brings in."""
    join = _rust_join
    t = arg.type
    if t == "scoped_identifier":
        path = join(prefix, _field(arg, "path") or "")
        return [(_field(arg, "name") or "", path or None, None)]
    if t == "use_as_clause":
        path = arg.child_by_field_name("path")
        items = _rust_use_items(path, prefix) if path is not None else []
        return [(name, module, _field(arg, "alias")) for name, module, _ in items[:1]]
    if t == "use_wildcard":
        module = join(prefix, _text(arg).removesuffix("*").removesuffix("::"))
        return [("*", module or None, None)]
    if t in ("scoped_use_list", "use_list"):
        if t == "scoped_use_list":
            prefix = join(prefix, _field(arg, "path") or "")
            arg = arg.child_by_field_name("list")
        found: list[tuple[str, str | None, str | None]] = []
        for child in arg.named_children if arg is not None else []:
            found.extend(_rust_use_items(child, prefix))
        return found
    return [(_text(arg), prefix or None, None)]


def _rust_reexports(root: Any) -> Reexports:
    """Top-level ``pub use`` (plain ``pub`` only: ``pub(crate)`` is not part of
    the crate's public surface)."""
    out: Reexports = []
    for statement in root.named_children:
        if statement.type != "use_declaration":
            continue
        visibility = next(
            (c for c in statement.named_children if c.type == "visibility_modifier"), None
        )
        arg = statement.child_by_field_name("argument")
        if visibility is None or _text(visibility) != "pub" or arg is None:
            continue
        for name, module, alias in _rust_use_items(arg, ""):
            _reexport(out, name, module, alias, statement)
    return out


def _rust_impls(root: Any) -> list[dict[str, Any]]:
    """``impl Trait for Type`` anywhere in the file; a negative impl
    (``impl !Send for T``) asserts the opposite and is skipped."""
    out: list[dict[str, Any]] = []
    for node in _descendants(root, "impl_item"):
        trait = node.child_by_field_name("trait")
        target = node.child_by_field_name("type")
        if trait is None or target is None or any(c.type == "!" for c in node.children):
            continue
        while target is not None and target.type == "reference_type":
            target = target.child_by_field_name("type")  # impl T for &'a mut Foo
        type_name = _strip_generics(_text(target))
        trait_name = _strip_generics(_text(trait))
        if type_name and trait_name:
            out.append({"type": type_name, "trait": trait_name, "line": node.start_point[0] + 1})
    return out


def _file_level(grammar: str, root: Any, file_path: str | None) -> tuple[Reexports, list[Any]]:
    if grammar in ("javascript", "typescript", TSX):
        return _js_reexports(root), []
    if grammar == "python":
        return _python_reexports(root, file_path), []
    if grammar == "rust":
        return _rust_reexports(root), _rust_impls(root)
    return [], []


def extract_symbols(
    content: bytes, *, language: str | None = None, file_path: str | None = None
) -> dict[str, Any]:
    """Symbols and imports defined by one file.

    Returns ``{language, parsed, parse_error_line, symbols, imports, reexports,
    impls}``. ``parsed`` is False only when no grammar or parse is available; a
    file with a syntax error still reports what the tree recovered, plus the
    first error line. *file_path* also decides whether a Python file is a
    package ``__init__`` (whose from-imports are re-exports). Raises
    ``ValueError`` for an unsupported or unresolvable language.
    """
    grammar = resolve_language(language, file_path)
    result: dict[str, Any] = {
        "language": grammar,
        "parsed": False,
        "parse_error_line": None,
        "symbols": [],
        "imports": [],
        "reexports": [],
        "impls": [],
    }
    tree = _parse(grammar, content)
    if tree is None:
        return result
    handler = _HANDLERS[grammar]
    symbols: Symbols = []
    imports: Imports = []
    stack: list[tuple[Any, str | None]] = [(tree.root_node, None)]
    while stack:
        node, parent = stack.pop()
        child_parent = handler(node, parent, symbols, imports) if node.is_named else parent
        stack.extend((child, child_parent) for child in reversed(node.children))
    reexports, impls = _file_level(grammar, tree.root_node, file_path)
    result.update(
        parsed=True,
        parse_error_line=hunt.first_error_line(tree.root_node),
        symbols=symbols,
        imports=imports,
        reexports=reexports,
        impls=impls,
    )
    return result
