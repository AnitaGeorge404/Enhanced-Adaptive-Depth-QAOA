r"""Linear Programming (LP) and Continuous Relaxation Warm-Starting (Feature 2).

Instead of starting QAOA from an uninformative uniform superposition |+>^{\otimes N},
warm-starting seeds single-qubit state preparations using solutions from a polynomial-time
classical continuous relaxation (e.g. LP for CSPP or box-constrained relaxation for QUBO).

Theory (Egger et al., Quantum 2021):
Given fractional probabilities c_i in [0, 1] from the continuous relaxation:
1. Regularize c_i in [eps, 1 - eps] to prevent vanishing gradients on deterministic qubits.
2. Map to rotation angle: theta_i = 2 * arcsin(sqrt(c_i)).
3. Prepare product state: |psi_0> = (x)_{i=0}^{N-1} (cos(theta_i/2)|0> + sin(theta_i/2)|1>).
   Note that for c_i = 0.5 (unbiased), theta_i = pi/2, exactly recovering |+> = Ry(pi/2)|0>.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any
import numpy as np
from scipy.optimize import linprog, minimize

from .cspp import CSPPInstance, Edge
from .qubo import QUBO, cspp_to_qubo


def probabilities_to_angles(
    probs: np.ndarray | list[float],
    eps: float = 0.05,
) -> np.ndarray:
    """Map continuous probabilities c_i in [0, 1] to single-qubit Ry angles theta_i.

    theta_i = 2 * arcsin(sqrt(clip(c_i, eps, 1 - eps))).
    """
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return 2.0 * np.arcsin(np.sqrt(p))


def solve_cspp_lp_relaxation(
    inst: CSPPInstance,
    n_slack: int | None = None,
    fixed_edges: dict[Edge, int] | None = None,
) -> np.ndarray:
    """Solve the continuous LP relaxation of CSPP.

    Returns:
        Fractional variable assignments x* in [0, 1]^N matching the QUBO variable indexing
        (active edges followed by slack register bits).
    """
    fixed = fixed_edges or {}
    fixed_0 = {e for e, v in fixed.items() if v == 0}
    fixed_1 = {e for e, v in fixed.items() if v == 1}

    active_edges = [e for e in inst.edges if e not in fixed_0 and e not in fixed_1]
    n_edges = len(active_edges)
    edge_to_idx = {e: k for k, e in enumerate(active_edges)}

    n_slack = max(1, math.ceil(math.log2(inst.r_limit + 1))) if n_slack is None else n_slack
    n_vars = n_edges + n_slack

    # Objective: min sum c_e * x_e
    c = np.zeros(n_vars, dtype=float)
    for e in active_edges:
        c[edge_to_idx[e]] = float(inst.cost[e])

    # Equalities: Flow conservation equations
    # in(v) - out(v) = b_v
    s, t = inst.source, inst.target
    eq_rows = []
    eq_b = []

    # Source node: sum out - sum in = 1
    row_s = np.zeros(n_vars, dtype=float)
    for e in active_edges:
        if e[0] == s:
            row_s[edge_to_idx[e]] += 1.0
        if e[1] == s:
            row_s[edge_to_idx[e]] -= 1.0
    fixed_out_s = sum(1.0 for e in fixed_1 if e[0] == s)
    fixed_in_s = sum(1.0 for e in fixed_1 if e[1] == s)
    eq_rows.append(row_s)
    eq_b.append(1.0 - (fixed_out_s - fixed_in_s))

    # Target node: sum in - sum out = 1
    row_t = np.zeros(n_vars, dtype=float)
    for e in active_edges:
        if e[1] == t:
            row_t[edge_to_idx[e]] += 1.0
        if e[0] == t:
            row_t[edge_to_idx[e]] -= 1.0
    fixed_in_t = sum(1.0 for e in fixed_1 if e[1] == t)
    fixed_out_t = sum(1.0 for e in fixed_1 if e[0] == t)
    eq_rows.append(row_t)
    eq_b.append(1.0 - (fixed_in_t - fixed_out_t))

    # Intermediate nodes: sum in - sum out = 0
    for v in range(inst.n_nodes):
        if v in (s, t):
            continue
        row_v = np.zeros(n_vars, dtype=float)
        for e in active_edges:
            if e[1] == v:
                row_v[edge_to_idx[e]] += 1.0
            if e[0] == v:
                row_v[edge_to_idx[e]] -= 1.0
        fixed_in_v = sum(1.0 for e in fixed_1 if e[1] == v)
        fixed_out_v = sum(1.0 for e in fixed_1 if e[0] == v)
        eq_rows.append(row_v)
        eq_b.append(0.0 - (fixed_in_v - fixed_out_v))

    # Resource equality constraint with slack register:
    # sum r_e * x_e + sum 2^k * s_k = R_limit - fixed_resource
    res_row = np.zeros(n_vars, dtype=float)
    for e in active_edges:
        res_row[edge_to_idx[e]] = float(inst.resource[e])
    for k in range(n_slack):
        res_row[n_edges + k] = float(1 << k)

    fixed_res = sum(float(inst.resource[e]) for e in fixed_1 if e in inst.resource)
    eq_rows.append(res_row)
    eq_b.append(float(inst.r_limit) - fixed_res)

    A_eq = np.array(eq_rows, dtype=float)
    b_eq = np.array(eq_b, dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n_vars)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success and res.x is not None:
        return np.clip(res.x, 0.0, 1.0)

    # Fallback to inequality LP if exact slack partition is tight
    res_ub_row = np.zeros((1, n_vars), dtype=float)
    for e in active_edges:
        res_ub_row[0, edge_to_idx[e]] = float(inst.resource[e])
    b_ub = np.array([float(inst.r_limit) - fixed_res], dtype=float)

    # Omit slack register in equality for fallback
    res_fallback = linprog(c, A_ub=res_ub_row, b_ub=b_ub, A_eq=A_eq[:-1], b_eq=b_eq[:-1], bounds=bounds, method="highs")
    if res_fallback.success and res_fallback.x is not None:
        sol = np.clip(res_fallback.x, 0.0, 1.0)
        # Compute inferred slack
        active_res = sum(float(inst.resource[e]) * sol[edge_to_idx[e]] for e in active_edges)
        rem_slack = max(0.0, float(inst.r_limit) - fixed_res - active_res)
        for k in range(n_slack):
            sol[n_edges + k] = min(1.0, max(0.0, (rem_slack % (2 << k)) / (1 << k)))
        return sol

    # Unbiased 0.5 fallback
    return np.full(n_vars, 0.5, dtype=float)


def solve_qubo_continuous_relaxation(qubo: QUBO) -> np.ndarray:
    """Continuous box-constrained relaxation min_{x in [0, 1]^N} x^T Q x + c^T x."""
    n = qubo.n_vars
    c = np.zeros(n, dtype=float)
    for i, g in qubo.linear.items():
        c[i] = g

    Q = np.zeros((n, n), dtype=float)
    for (i, j), w in qubo.quadratic.items():
        Q[i, j] = w / 2.0
        Q[j, i] = w / 2.0

    def objective(x):
        return float(np.dot(c, x) + np.dot(x, np.dot(Q, x)) + qubo.offset)

    def gradient(x):
        return c + 2.0 * np.dot(Q, x)

    x0 = np.full(n, 0.5, dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]

    res = minimize(objective, x0, jac=gradient, bounds=bounds, method="L-BFGS-B")
    if res.success and res.x is not None:
        return np.clip(res.x, 0.0, 1.0)
    return x0


def get_warm_start_angles(
    instance_or_qubo: Any,
    eps: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """Compute warm-start initial state preparation angles theta in [0, pi]^N.

    Accepts either a CSPPInstance, MaxCutInstance, or a QUBO instance.
    """
    if isinstance(instance_or_qubo, CSPPInstance):
        probs = solve_cspp_lp_relaxation(instance_or_qubo, **kwargs)
    elif isinstance(instance_or_qubo, QUBO):
        probs = solve_qubo_continuous_relaxation(instance_or_qubo)
    elif hasattr(instance_or_qubo, "cut_value") and hasattr(instance_or_qubo, "n_nodes"):
        n = instance_or_qubo.n_nodes
        lin: dict[int, float] = defaultdict(float)
        quad: dict[tuple[int, int], float] = defaultdict(float)
        for (u, v) in instance_or_qubo.edges:
            w = instance_or_qubo.weights.get((u, v), 1.0)
            lin[u] -= w
            lin[v] -= w
            quad[(min(u, v), max(u, v))] += 2.0 * w
        q = QUBO(n, dict(lin), dict(quad), 0.0, {}, 0)
        probs = solve_qubo_continuous_relaxation(q)
    else:
        raise TypeError(f"Unsupported problem type for warm-starting: {type(instance_or_qubo)}")

    return probabilities_to_angles(probs, eps=eps)
