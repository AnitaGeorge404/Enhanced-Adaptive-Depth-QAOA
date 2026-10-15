"""CSPP -> QUBO (paper Eqs. 12-16) -> Ising (Eq. 17).

Variables: one binary x_ij per edge, followed by a binary slack register that turns
the inequality  sum r_ij x_ij <= r_limit  into the equality
sum r_ij x_ij + sum_k 2^k s_k = r_limit, which is then penalised quadratically.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .cspp import CSPPInstance, Edge


@dataclass
class QUBO:
    n_vars: int
    linear: dict[int, float]                 # g_i
    quadratic: dict[tuple[int, int], float]  # Q_ij, i < j
    offset: float                            # c_0
    edge_index: dict[Edge, int]
    n_slack: int

    def energy(self, x: np.ndarray) -> float:
        e = self.offset + sum(g * x[i] for i, g in self.linear.items())
        return e + sum(q * x[i] * x[j] for (i, j), q in self.quadratic.items())


@dataclass
class Ising:
    n_qubits: int
    h: dict[int, float]
    J: dict[tuple[int, int], float]
    offset: float


class _QuboBuilder:
    def __init__(self) -> None:
        self.lin: dict[int, float] = defaultdict(float)
        self.quad: dict[tuple[int, int], float] = defaultdict(float)
        self.offset = 0.0

    def add_linear(self, terms: dict[int, float]) -> None:
        for i, a in terms.items():
            self.lin[i] += a

    def add_squared(self, terms: dict[int, float], const: float, weight: float) -> None:
        """weight * (sum_i a_i x_i + const)^2, using x_i^2 = x_i."""
        items = list(terms.items())
        for n, (i, a) in enumerate(items):
            self.lin[i] += weight * (a * a + 2 * const * a)
            for j, b in items[n + 1:]:
                key = (min(i, j), max(i, j))
                self.quad[key] += weight * 2 * a * b
        self.offset += weight * const * const


def cspp_to_qubo(
    inst: CSPPInstance,
    lam: float | None = None,
    rho: float | None = None,
    n_slack: int | None = None,
    fixed_edges: dict[Edge, int] | None = None,
) -> QUBO:
    """Build H_CSPP = H_cost + H_resource + H_flow as a QUBO.

    lam, rho: penalty weights (default: 1 + total edge cost, so any violation
        outweighs every feasible path cost).
    n_slack: slack bits (default: ceil(log2(r_limit + 1))).
    fixed_edges: dictionary mapping edges to 0 or 1. Variables fixed to 0 or 1
        are eliminated from the QUBO, updating linear terms and constant offset.
    """
    big = 1.0 + sum(inst.cost.values())
    lam = big if lam is None else lam
    rho = big if rho is None else rho
    n_slack = max(1, math.ceil(math.log2(inst.r_limit + 1))) if n_slack is None else n_slack

    fixed = fixed_edges or {}
    fixed_0 = {e for e, v in fixed.items() if v == 0}
    fixed_1 = {e for e, v in fixed.items() if v == 1}

    # Active edges allocated a binary variable
    active_edges = [e for e in inst.edges if e not in fixed_0 and e not in fixed_1]
    idx = {e: k for k, e in enumerate(active_edges)}
    slack = [len(idx) + k for k in range(n_slack)]
    s, t = inst.source, inst.target
    b = _QuboBuilder()

    # H_cost (Eq. 13)
    b.add_linear({idx[e]: inst.cost[e] for e in active_edges})
    b.offset += sum(inst.cost[e] for e in fixed_1 if e in inst.cost)

    # H_resource (Eq. 14) with slack register
    res_terms = {idx[e]: float(inst.resource[e]) for e in active_edges}
    res_terms.update({q: float(2**k) for k, q in enumerate(slack)})
    fixed_res = sum(float(inst.resource[e]) for e in fixed_1 if e in inst.resource)
    b.add_squared(res_terms, fixed_res - inst.r_limit, rho)

    # H_flow (Eq. 15)
    out_active = lambda v: {idx[e]: 1.0 for e in active_edges if e[0] == v}
    in_active = lambda v: {idx[e]: 1.0 for e in active_edges if e[1] == v}
    out_fixed1 = lambda v: sum(1.0 for e in fixed_1 if e[0] == v and e in inst.edges)
    in_fixed1 = lambda v: sum(1.0 for e in fixed_1 if e[1] == v and e in inst.edges)

    # Source: sum_{e out of s} x_e = 1
    b.add_squared(out_active(s), out_fixed1(s) - 1.0, lam)
    # Target: sum_{e into t} x_e = 1
    b.add_squared(in_active(t), in_fixed1(t) - 1.0, lam)
    # No flow into source or out of target
    b.add_squared(in_active(s), in_fixed1(s), lam)
    b.add_squared(out_active(t), out_fixed1(t), lam)

    # Intermediate flow balance: in(v) - out(v) = 0
    for v in range(inst.n_nodes):
        if v in (s, t):
            continue
        balance = dict(in_active(v))
        for i, a in out_active(v).items():
            balance[i] = balance.get(i, 0.0) - a
        const_v = in_fixed1(v) - out_fixed1(v)
        b.add_squared(balance, const_v, lam)

    quad = {k: v for k, v in b.quad.items() if v != 0.0}
    return QUBO(len(idx) + n_slack, dict(b.lin), quad, b.offset, idx, n_slack)


def qubo_to_ising(q: QUBO) -> Ising:
    """Substitute x_i = (1 - z_i) / 2 (z_i = eigenvalue of Pauli Z_i)."""
    h: dict[int, float] = defaultdict(float)
    J: dict[tuple[int, int], float] = defaultdict(float)
    offset = q.offset
    for i, g in q.linear.items():
        offset += g / 2
        h[i] -= g / 2
    for (i, j), w in q.quadratic.items():
        offset += w / 4
        h[i] -= w / 4
        h[j] -= w / 4
        J[(i, j)] += w / 4
    h = {i: v for i, v in h.items() if abs(v) > 1e-12}
    J = {k: v for k, v in J.items() if abs(v) > 1e-12}
    return Ising(q.n_vars, h, J, offset)


def decode(bits: np.ndarray | list[int], q: QUBO) -> list[Edge]:
    """Edges selected by a bitstring (wire i <-> variable i)."""
    return [e for e, k in q.edge_index.items() if bits[k] == 1]
