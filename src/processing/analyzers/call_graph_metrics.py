"""
Graph metrics for a call graph: cycles (SCCs), depth, critical paths, orphans.

Cycles are strongly connected components of size > 1 plus self-recursive
functions. Depth is the longest caller->callee chain in the SCC-condensed DAG
(so recursion cannot make it infinite), capped at ``max_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass

from processing.graph_sccs import strongly_connected_components as _sccs

#: Functions that are expected to have no callers inside the repository.
ENTRY_POINT_NAMES = frozenset({"main", "cli", "run", "app", "setUp", "tearDown"})


@dataclass
class GraphAnalysis:
    cycles: list[str]
    max_depth: int
    critical_paths: list[str]
    orphaned: list[str]
    internal_edges: int


def is_entry_point(node_id: str) -> bool:
    """main/cli, dunder methods and tests are called from outside the graph."""
    last = node_id.rsplit(".", 1)[-1]
    return (
        last in ENTRY_POINT_NAMES
        or (last.startswith("__") and last.endswith("__"))
        or last.startswith("test_")
    )


def analyse_graph(
    node_ids: list[str], edges: list[tuple[str, str]], max_depth: int = 10
) -> GraphAnalysis:
    """Cycles, capped depth, top critical paths and orphaned functions."""
    nodes = list(dict.fromkeys([*node_ids, *(n for e in edges for n in e if n)]))
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    callers: dict[str, set[str]] = {n: set() for n in nodes}
    unique = {(a, b) for a, b in edges if a and b}
    for a, b in sorted(unique):
        adj[a].append(b)
        if a != b:
            callers[b].add(a)

    components = _sccs(nodes, adj)
    comp_of = {member: i for i, comp in enumerate(components) for member in comp}
    cycles = [
        " -> ".join([*comp, comp[0]])
        for comp in components
        if len(comp) > 1 or comp[0] in adj.get(comp[0], [])
    ]

    # Condensed DAG: Tarjan emits components in reverse topological order,
    # so every successor component is finished before its predecessors.
    dag: dict[int, set[int]] = {i: set() for i in range(len(components))}
    has_pred: set[int] = set()
    for a, b in unique:
        ca, cb = comp_of[a], comp_of[b]
        if ca != cb:
            dag[ca].add(cb)
            has_pred.add(cb)
    longest: dict[int, int] = {}
    best_next: dict[int, int | None] = {}
    for i in range(len(components)):  # reverse topological order
        best, nxt = 0, None
        for j in sorted(dag[i]):
            if longest[j] + 1 > best:
                best, nxt = longest[j] + 1, j
        longest[i], best_next[i] = best, nxt

    depth = max(longest.values(), default=0)
    sources = sorted(
        (i for i in range(len(components)) if i not in has_pred and longest[i] > 0),
        key=lambda i: (-longest[i], components[i][0]),
    )
    critical: list[str] = []
    for i in sources[:10]:
        path = [components[i][0]]
        cur: int | None = best_next[i]
        while cur is not None and len(path) <= max_depth:
            path.append(components[cur][0])
            cur = best_next[cur]
        critical.append(" -> ".join(path))

    orphaned = sorted(n for n in node_ids if not callers.get(n) and not is_entry_point(n))
    return GraphAnalysis(
        cycles=cycles,
        max_depth=min(depth, max_depth),
        critical_paths=critical,
        orphaned=orphaned,
        internal_edges=len(unique),
    )


__all__ = ["ENTRY_POINT_NAMES", "GraphAnalysis", "analyse_graph", "is_entry_point"]
