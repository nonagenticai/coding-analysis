"""Result types shared by the structure extractors (all picklable dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Param:
    name: str
    type: str | None = None
    default: str | None = None


@dataclass
class Signature:
    params: list[Param] = field(default_factory=list)
    returns: str | None = None
    generics: str | None = None  # Rust: ``<T: Clone>`` plus any ``where`` clause
    is_async: bool = False


@dataclass
class Member:
    """A struct field or enum variant (Rust), listed on the owning symbol."""

    name: str
    type: str | None = None
    visibility: str | None = None


@dataclass
class ApiSymbol:
    id: str
    module_path: str
    name: str
    #: function | class | method | const | struct | enum | trait | impl_fn | static |
    #: type_alias | union | macro | reexport
    kind: str
    visibility: str
    file: str
    line: int
    signature: Signature | None = None
    members: list[Member] = field(default_factory=list)
    #: Attribute macros / derives / decorators on the item, as written (``derive(Debug)``).
    attributes: list[str] = field(default_factory=list)
    owner: str | None = None  # class / impl type / trait the symbol belongs to
    trait: str | None = None  # impl_fn: the trait implemented


@dataclass
class Unresolved:
    file: str
    line: int
    reason: str
    name: str | None = None
    #: The macro responsible (``macro_rules! mk``, ``derive(Serialize)``, ``#[get]``) when
    #: the unresolved item is macro-generated.
    macro: str | None = None
    expression: str | None = None
    #: Equally plausible targets when the item is ambiguous rather than unknown.
    candidates: list[str] = field(default_factory=list)


@dataclass
class ModuleNode:
    module: str
    file: str
    loc: int
    public_symbols: int


@dataclass
class ModuleEdge:
    source: str
    target: str
    kind: str  # import | from_import | dynamic | type_only | use | path
    file: str
    line: int


@dataclass
class Slice:
    """One porting unit: an SCC of the module graph (a cycle is ONE slice)."""

    id: str
    layer: int
    modules: list[str]
    cyclic: bool
    depends_on: list[str]
    loc: int


@dataclass
class ModuleGraph:
    language: str
    nodes: list[ModuleNode]
    edges: list[ModuleEdge]
    sccs: list[list[str]]
    order: list[list[str]]
    unresolved: list[Unresolved]
    files_unparsed: dict[str, str] = field(default_factory=dict)
    slices: list[Slice] = field(default_factory=list)


@dataclass
class ApiSurface:
    language: str
    symbols: list[ApiSymbol]
    unresolved: list[Unresolved]
    files_unparsed: dict[str, str] = field(default_factory=dict)
