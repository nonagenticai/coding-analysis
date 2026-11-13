"""Strongly connected components (Tarjan), shared by the call graph and module graph.

A leaf module with no heavy imports, so isolated extraction workers can use it.
"""

from __future__ import annotations


def strongly_connected_components(nodes: list[str], adj: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative (deep call chains must not hit recursion limits).

    Components come out in reverse topological order: every component is emitted
    after all components reachable from it (``processing.structure.module_graph``
    relies on this to layer the condensed DAG in one pass).
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0
    for start in nodes:
        if start in index:
            continue
        work: list[tuple[str, int]] = [(start, 0)]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, i = work[-1]
            succ = adj.get(node, [])
            if i < len(succ):
                work[-1] = (node, i + 1)
                nxt = succ[i]
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                result.append(sorted(component))
    return result
