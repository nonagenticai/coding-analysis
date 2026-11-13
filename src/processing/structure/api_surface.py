"""Public API surface with signatures, Python or Rust (one extractor per language).

Python (``python_facts``): ``__all__`` when literal, else names without a leading
``_``; functions, classes, public methods (plus ``__init__``/``__call__``),
module-level bindings (``const``) and ``__all__`` re-exports (``reexport``).

Rust (``rust_facts``): every item with a ``pub`` visibility (``pub``,
``pub(crate)``, ... as written) in every module reachable from a crate root:
functions, structs/enums/unions with their fields/variants (``members``), traits
and their methods, consts/statics/type aliases, inherent ``pub`` methods
(``method``) and every trait-impl method (``impl_fn``, visibility ``pub``: as
public as the trait). One vocabulary for items and members: the modifier as
written, or ``private`` (a non-``pub`` field). Macro-generated items are listed under ``unresolved`` with the
macro responsible, never silently missing.
"""

from __future__ import annotations

from . import module_graph as mg
from . import rust_facts as rf
from .types import ApiSurface


def api_surface(files: dict[str, str], language: str, root: str = "") -> ApiSurface:
    if language == "rust":
        ws = rf.load_workspace(files)
        symbols, problems = rf.surface(ws)
        unresolved = [
            u
            for u in ws.unresolved
            if u.reason.startswith(("module file not found", "file is not reachable", "include!"))
        ]
        return ApiSurface(
            language="rust",
            symbols=sorted(symbols, key=lambda s: (s.file, s.line, s.id)),
            unresolved=sorted([*unresolved, *problems], key=lambda u: (u.file, u.line)),
            files_unparsed=dict(sorted(ws.unparsed.items())),
        )
    parsed, unparsed = mg.parse_python_files(files, root)
    symbols = [s for f in parsed for s in f.symbols]
    problems = [
        u for f in parsed for u in f.unresolved if not u.reason.startswith("dynamic import")
    ]
    return ApiSurface(
        language="python",
        symbols=sorted(symbols, key=lambda s: (s.file, s.line, s.id)),
        unresolved=sorted(problems, key=lambda u: (u.file, u.line)),
        files_unparsed=unparsed,
    )
