"""Constrained Shortest Path Problem (CSPP) instances.

Paper Sec. 2.2: find the minimum-cost s->t path in a directed graph such that the
total resource consumption stays below ``r_limit`` (single resource, M = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Edge = tuple[int, int]


@dataclass
class CSPPInstance:
    n_nodes: int
    source: int
    target: int
    edges: list[Edge]
    cost: dict[Edge, float]
    resource: dict[Edge, int]
    r_limit: int
    meta: dict = field(default_factory=dict)

    def path_cost(self, path: list[int]) -> float:
        return sum(self.cost[(u, v)] for u, v in zip(path, path[1:]))

    def path_resource(self, path: list[int]) -> int:
        return sum(self.resource[(u, v)] for u, v in zip(path, path[1:]))


def simple_paths(inst: CSPPInstance) -> list[list[int]]:
    """All simple source->target paths (DFS; fine for the small graphs used here)."""
    adj: dict[int, list[int]] = {v: [] for v in range(inst.n_nodes)}
    for u, v in inst.edges:
        adj[u].append(v)
    out, stack = [], [(inst.source, [inst.source])]
    while stack:
        node, path = stack.pop()
        if node == inst.target:
            out.append(path)
            continue
        for nxt in adj[node]:
            if nxt not in path:
                stack.append((nxt, path + [nxt]))
    return out


def solve_classical(inst: CSPPInstance) -> tuple[list[int] | None, float]:
    """Exact solution by enumeration. Returns (path, cost); (None, inf) if infeasible."""
    best, best_cost = None, float("inf")
    for p in simple_paths(inst):
        if inst.path_resource(p) <= inst.r_limit and inst.path_cost(p) < best_cost:
            best, best_cost = p, inst.path_cost(p)
    return best, best_cost


def generate_instance(
    n_nodes: int = 4,
    seed: int | None = None,
    edge_prob: float = 1.0,
    prune: bool = True,
    cost_range: tuple[int, int] = (1, 10),
    resource_range: tuple[int, int] = (1, 6),
    r_limit: int | None = None,
) -> CSPPInstance:
    """Random CSPP instance on a (possibly complete) bi-directed graph.

    prune: drop edges entering the source / leaving the target. Constraints (7)-(8)
        force those variables to 0 anyway, so removing them saves qubits.
    r_limit: if None, chosen so the resource constraint is *active* (the
        unconstrained shortest path is infeasible) whenever possible.
    """
    rng = np.random.default_rng(seed)
    s, t = 0, n_nodes - 1
    while True:
        edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j or (prune and (j == s or i == t)):
                    continue
                if rng.random() <= edge_prob:
                    edges.append((i, j))
        cost = {e: int(rng.integers(cost_range[0], cost_range[1] + 1)) for e in edges}
        res = {e: int(rng.integers(resource_range[0], resource_range[1] + 1)) for e in edges}
        inst = CSPPInstance(n_nodes, s, t, edges, cost, res, r_limit=10**9)

        paths = simple_paths(inst)
        if not paths:
            continue
        if r_limit is not None:
            inst.r_limit = r_limit
        else:
            r_min = min(inst.path_resource(p) for p in paths)
            r_free = inst.path_resource(min(paths, key=inst.path_cost))
            inst.r_limit = int(rng.integers(r_min, r_free)) if r_free > r_min else r_free
        if solve_classical(inst)[0] is not None:
            inst.meta["seed"] = seed
            return inst
