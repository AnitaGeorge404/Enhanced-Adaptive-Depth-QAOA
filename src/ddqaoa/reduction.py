"""Optimality-preserving instance reduction for CSPP (Feature 1).

Shrinks the QUBO formulation by eliminating qubits and ZZ interaction terms
with mathematical guarantees of optimality preservation.

Rules applied to a fixed point:
- R0: Structural fixing: edges entering s or leaving t are fixed to 0.
- R1: Resource reachability: Dijkstra forward from s (rho_s) and backward to t (rho_t).
      Remove edge (u, v) if rho_s(u) + r_uv + rho_t(v) > r_limit.
      Preserves ALL feasible paths.
- R2: Cost-bound pruning: with an upper bound UB from a feasible path, remove
      edge (u, v) if gamma_s(u) + c_uv + gamma_t(v) > UB (strict inequality preserves
      all optimal paths and ties).
- R3: Lagrangian reduced-cost fixing: for optimal multiplier lambda >= 0,
      bound = d_s^lambda(u) + (c_uv + lambda * r_uv) + d_t^lambda(v) - lambda * r_limit.
      Fix to 0 if bound > UB.
- R4: Forced-edge fixing: an edge on every resource-feasible s-t path is fixed to 1
      and its qubit eliminated by substitution in the QUBO.
- Slack tightening: effective slack is r_limit - rho_s(t), reducing required slack qubits.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cspp import CSPPInstance, Edge, simple_paths, solve_classical


def dijkstra_distances(
    n_nodes: int,
    edges: list[Edge],
    weights: dict[Edge, float | int],
    source: int,
    reverse: bool = False,
) -> dict[int, float]:
    """Compute single-source shortest path distances using Dijkstra's algorithm.

    If reverse=True, searches backward to find minimum distances from all nodes to source.
    """
    adj: dict[int, list[tuple[int, float]]] = {v: [] for v in range(n_nodes)}
    for (u, v) in edges:
        w = float(weights.get((u, v), float("inf")))
        if reverse:
            adj[v].append((u, w))
        else:
            adj[u].append((v, w))

    dist: dict[int, float] = {v: float("inf") for v in range(n_nodes)}
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            cand = d + w
            if cand < dist[v]:
                dist[v] = cand
                heapq.heappush(pq, (cand, v))

    return dist


@dataclass
class ReductionResult:
    """Result of optimality-preserving instance reduction."""

    original_instance: CSPPInstance
    reduced_instance: CSPPInstance
    edge_map: dict[Edge, Edge]  # reduced_edge -> original_edge
    fixed_edges: dict[Edge, int]  # original_edge -> 0 or 1
    slack_bits_before: int
    slack_bits_after: int
    preserves_all_feasible: bool
    is_feasible: bool
    stats: dict[str, Any] = field(default_factory=dict)

    def lift(self, reduced_bits: np.ndarray | list[int] | dict[Edge, int]) -> dict[Edge, int]:
        """Lift a solution on the reduced instance to original edge assignments {edge: 0|1}."""
        full_assignment: dict[Edge, int] = dict(self.fixed_edges)

        if isinstance(reduced_bits, dict):
            for red_edge, val in reduced_bits.items():
                orig_edge = self.edge_map.get(red_edge, red_edge)
                full_assignment[orig_edge] = int(val)
        else:
            bits_arr = np.asarray(reduced_bits, dtype=int)
            # Map active edges (edges not fixed to 0 or 1) by index
            active_edges = [e for e in self.reduced_instance.edges if self.fixed_edges.get(e) not in (0, 1)]
            for k, red_edge in enumerate(active_edges):
                if k < len(bits_arr):
                    orig_edge = self.edge_map.get(red_edge, red_edge)
                    full_assignment[orig_edge] = int(bits_arr[k])

        # Ensure all original edges are defined (default 0)
        for e in self.original_instance.edges:
            if e not in full_assignment:
                full_assignment[e] = 0
        return full_assignment

    def lift_path(self, reduced_path: list[int] | None) -> list[int] | None:
        """Map a node path from the reduced instance to the original graph."""
        return reduced_path


def _find_bridge_edges(inst: CSPPInstance, candidate_edges: list[Edge]) -> list[Edge]:
    """Find edges whose removal completely disconnects resource-feasible s-t paths."""
    s, t = inst.source, inst.target
    rho_s = dijkstra_distances(inst.n_nodes, candidate_edges, inst.resource, s)
    if rho_s[t] > inst.r_limit:
        return []

    bridges: list[Edge] = []
    for e in candidate_edges:
        sub_edges = [edge for edge in candidate_edges if edge != e]
        dist_s = dijkstra_distances(inst.n_nodes, sub_edges, inst.resource, s)
        if dist_s[t] > inst.r_limit:
            # Removing e eliminates all resource-feasible paths from s to t
            bridges.append(e)
    return bridges


def _lagrangian_subgradient(
    inst: CSPPInstance,
    candidate_edges: list[Edge],
    upper_bound: float,
    iters: int = 25,
) -> float:
    """Find a Lagrangian multiplier lambda that maximizes the lower bound L(lambda)."""
    s, t = inst.source, inst.target
    best_lambda = 0.0
    best_lb = -float("inf")

    lambdas = np.linspace(0.0, 5.0, iters)
    for lam in lambdas:
        mod_weights = {e: float(inst.cost[e]) + lam * float(inst.resource[e]) for e in candidate_edges}
        dist = dijkstra_distances(inst.n_nodes, candidate_edges, mod_weights, s)
        lb = dist[t] - lam * inst.r_limit
        if lb > best_lb:
            best_lb = lb
            best_lambda = float(lam)

    return best_lambda


def reduce_instance(
    inst: CSPPInstance,
    *,
    upper_bound: float | None = None,
    lagrangian: bool = False,
    max_iters: int = 50,
) -> ReductionResult:
    """Apply optimality-preserving reduction rules R0-R4 to a fixed point."""
    s, t = inst.source, inst.target
    n_nodes = inst.n_nodes
    orig_slack_bits = max(1, math.ceil(math.log2(inst.r_limit + 1)))

    fixed_edges: dict[Edge, int] = {}
    candidate_edges = list(inst.edges)
    preserves_all_feasible = True

    # If upper_bound not provided, solve classical
    if upper_bound is None:
        best_p, best_c = solve_classical(inst)
        if best_p is not None:
            upper_bound = best_c
        else:
            return ReductionResult(
                original_instance=inst,
                reduced_instance=CSPPInstance(n_nodes, s, t, [], {}, {}, inst.r_limit),
                edge_map={},
                fixed_edges={e: 0 for e in inst.edges},
                slack_bits_before=orig_slack_bits,
                slack_bits_after=1,
                preserves_all_feasible=True,
                is_feasible=False,
                stats={"reason": "infeasible_original", "iterations": 0},
            )

    iteration = 0
    while iteration < max_iters:
        iteration += 1
        changed = False

        # R0: Structural fixing (edges entering s or leaving t)
        for e in list(candidate_edges):
            u, v = e
            if v == s or u == t:
                fixed_edges[e] = 0
                candidate_edges.remove(e)
                changed = True

        # R1: Resource reachability Dijkstra
        rho_s = dijkstra_distances(n_nodes, candidate_edges, inst.resource, s, reverse=False)
        rho_t = dijkstra_distances(n_nodes, candidate_edges, inst.resource, t, reverse=True)

        if rho_s[t] > inst.r_limit or math.isinf(rho_s[t]):
            return ReductionResult(
                original_instance=inst,
                reduced_instance=CSPPInstance(n_nodes, s, t, [], {}, {}, inst.r_limit),
                edge_map={},
                fixed_edges={e: 0 for e in inst.edges},
                slack_bits_before=orig_slack_bits,
                slack_bits_after=1,
                preserves_all_feasible=preserves_all_feasible,
                is_feasible=False,
                stats={"reason": "no_resource_feasible_path", "iterations": iteration},
            )

        for e in list(candidate_edges):
            u, v = e
            r_uv = float(inst.resource[e])
            if rho_s[u] + r_uv + rho_t[v] > inst.r_limit:
                fixed_edges[e] = 0
                candidate_edges.remove(e)
                changed = True

        # R2: Cost-bound pruning
        if upper_bound is not None and math.isfinite(upper_bound):
            gamma_s = dijkstra_distances(n_nodes, candidate_edges, inst.cost, s, reverse=False)
            gamma_t = dijkstra_distances(n_nodes, candidate_edges, inst.cost, t, reverse=True)
            for e in list(candidate_edges):
                u, v = e
                c_uv = float(inst.cost[e])
                if gamma_s[u] + c_uv + gamma_t[v] > upper_bound + 1e-12:
                    fixed_edges[e] = 0
                    candidate_edges.remove(e)
                    preserves_all_feasible = False
                    changed = True

        # R3: Lagrangian reduced-cost fixing
        if lagrangian and upper_bound is not None and candidate_edges:
            lam = _lagrangian_subgradient(inst, candidate_edges, upper_bound)
            if lam > 0.0:
                mod_w = {e: float(inst.cost[e]) + lam * float(inst.resource[e]) for e in candidate_edges}
                d_s = dijkstra_distances(n_nodes, candidate_edges, mod_w, s, reverse=False)
                d_t = dijkstra_distances(n_nodes, candidate_edges, mod_w, t, reverse=True)
                for e in list(candidate_edges):
                    u, v = e
                    w_uv = mod_w[e]
                    lb_e = d_s[u] + w_uv + d_t[v] - lam * inst.r_limit
                    if lb_e > upper_bound + 1e-12:
                        fixed_edges[e] = 0
                        candidate_edges.remove(e)
                        preserves_all_feasible = False
                        changed = True

        # R4: Forced-edge fixing
        bridges = _find_bridge_edges(inst, candidate_edges)
        for b in bridges:
            if b in candidate_edges and fixed_edges.get(b) != 1:
                fixed_edges[b] = 1
                # Forced edge remains in candidate_edges for graph connectivity/path search,
                # but is marked fixed_edges[b]=1 so its QUBO variable is eliminated.
                changed = True

        if not changed:
            break

    # Slack register tightening
    rho_s_final = dijkstra_distances(n_nodes, candidate_edges, inst.resource, s)
    min_res = int(rho_s_final[t]) if math.isfinite(rho_s_final[t]) else 0
    effective_slack = max(0, inst.r_limit - min_res)
    slack_bits_after = max(1, math.ceil(math.log2(effective_slack + 1)))

    reduced_cost = {e: inst.cost[e] for e in candidate_edges}
    reduced_resource = {e: inst.resource[e] for e in candidate_edges}
    reduced_inst = CSPPInstance(
        n_nodes=n_nodes,
        source=s,
        target=t,
        edges=candidate_edges,
        cost=reduced_cost,
        resource=reduced_resource,
        r_limit=inst.r_limit,
        meta={"reduced": True, "effective_slack": effective_slack},
    )

    edge_map = {e: e for e in candidate_edges}

    stats = {
        "iterations": iteration,
        "original_edges": len(inst.edges),
        "reduced_edges": len(candidate_edges),
        "fixed_0_count": sum(1 for v in fixed_edges.values() if v == 0),
        "fixed_1_count": sum(1 for v in fixed_edges.values() if v == 1),
        "active_qubo_edges": len(candidate_edges) - sum(1 for v in fixed_edges.values() if v == 1),
        "slack_reduction": orig_slack_bits - slack_bits_after,
        "upper_bound_used": upper_bound,
    }

    return ReductionResult(
        original_instance=inst,
        reduced_instance=reduced_inst,
        edge_map=edge_map,
        fixed_edges=fixed_edges,
        slack_bits_before=orig_slack_bits,
        slack_bits_after=slack_bits_after,
        preserves_all_feasible=preserves_all_feasible,
        is_feasible=True,
        stats=stats,
    )
