"""Evaluation metrics, feasibility checking, and end-to-end quantum cost (Feature 4).

Features:
1. Multi-level feasibility checking:
   - "structural": node flow conservation equations (Eqs. 7-8).
   - "path": connected simple s-t path, no isolated loops, resource <= r_limit (slack marginalized).
   - "qubo": all penalty terms zero, including exact binary slack equality.
2. decode_path: maps bitstrings (with reduction lift if applicable) to simple node paths.
3. Post-selected objective: E_ps = sum_{x in F} P(x) E(x) / sum_{x in F} P(x) with y_min fallback.
4. Time-to-solution: TTS_99 = shots_per_run * ceil(ln(0.01) / ln(1 - P_optimal)).
5. Feasible-conditioned metrics and best-of-S gap.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pennylane as qml

from .circuits import QAOAProblem, index_to_bits
from .cspp import CSPPInstance, Edge
from .qubo import QUBO

FeasibilityLevel = Literal["structural", "path", "qubo"]


def approximation_ratio(energy: float, problem: QAOAProblem) -> float:
    """r = (E_max - <H>) / (E_max - E_min), in [0, 1] with r = 1 at the ground state."""
    denom = problem.e_max - problem.e_min
    if abs(denom) < 1e-12:
        return 1.0
    return float((problem.e_max - energy) / denom)


def success_probability(probs: np.ndarray, problem: QAOAProblem) -> float:
    """Probability of sampling an optimal bitstring."""
    return float(np.sum(np.asarray(probs)[problem.ground_states]))


def cumulative_cnots(depth_per_step: list[int], cnots_per_layer: int) -> int:
    """Sum over optimiser steps of the CNOT count of the circuit used at that step."""
    return int(sum(depth_per_step) * cnots_per_layer)


def decode_path(
    bitstring: np.ndarray | list[int],
    instance: CSPPInstance,
    reduction_result: Any | None = None,
) -> list[int] | None:
    """Decode a bitstring into a simple directed s->t path in the instance graph.

    Returns None if bitstring does not form a valid, connected simple s->t path
    or if it contains disconnected cycles / branchings.
    """
    if reduction_result is not None:
        edge_assignment = reduction_result.lift(bitstring)
    else:
        bits = np.asarray(bitstring, dtype=int)
        edge_assignment = {
            e: int(bits[k]) for k, e in enumerate(instance.edges) if k < len(bits)
        }

    active_edges = [e for e, v in edge_assignment.items() if v == 1]
    if not active_edges:
        return None

    s, t = instance.source, instance.target
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    adj: dict[int, list[int]] = defaultdict(list)

    for (u, v) in active_edges:
        out_deg[u] += 1
        in_deg[v] += 1
        adj[u].append(v)

    if out_deg[s] != 1 or in_deg[s] != 0:
        return None
    if in_deg[t] != 1 or out_deg[t] != 0:
        return None

    # Follow path from s to t
    curr = s
    path = [s]
    visited = {s}

    while curr != t:
        nbrs = adj.get(curr, [])
        if len(nbrs) != 1:
            return None  # Dead end or branching
        nxt = nbrs[0]
        if nxt in visited:
            return None  # Cycle
        visited.add(nxt)
        path.append(nxt)
        curr = nxt

    # Ensure no disconnected components or isolated cycles exist
    if len(active_edges) != len(path) - 1:
        return None

    return path


def is_feasible(
    bitstring: np.ndarray | list[int],
    instance: CSPPInstance,
    level: FeasibilityLevel = "path",
    reduction_result: Any | None = None,
    qubo: QUBO | None = None,
) -> bool:
    """Multi-level feasibility validation.

    Levels:
    - "structural": Satisfies flow conservation (source, target, intermediate node balance).
    - "path": Forms a simple s->t path (no cycles/subtours) and resource <= r_limit (slack ignored).
    - "qubo": All penalty quadratics equal zero, including exact slack equality match.
    """
    if reduction_result is not None:
        edge_assignment = reduction_result.lift(bitstring)
    else:
        bits = np.asarray(bitstring, dtype=int)
        edge_assignment = {
            e: int(bits[k]) for k, e in enumerate(instance.edges) if k < len(bits)
        }

    s, t = instance.source, instance.target
    active_edges = [e for e, v in edge_assignment.items() if v == 1]

    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for (u, v) in active_edges:
        out_deg[u] += 1
        in_deg[v] += 1

    # Structural check
    structural_ok = (
        out_deg[s] == 1
        and in_deg[s] == 0
        and in_deg[t] == 1
        and out_deg[t] == 0
        and all(in_deg[v] == out_deg[v] for v in range(instance.n_nodes) if v not in (s, t))
    )

    if level == "structural":
        return bool(structural_ok)

    # Path check
    path = decode_path(bitstring, instance, reduction_result)
    if path is None:
        return False

    res_used = instance.path_resource(path)
    if res_used > instance.r_limit:
        return False

    if level == "path":
        return True

    # QUBO check: must also verify slack bits equal r_limit - res_used
    bits_arr = np.asarray(bitstring, dtype=int)
    n_edges = len(instance.edges) if reduction_result is None else len(reduction_result.reduced_instance.edges)
    n_slack = len(bits_arr) - n_edges
    if n_slack <= 0 and qubo is not None:
        n_slack = qubo.n_slack

    if n_slack > 0 and len(bits_arr) >= n_edges + n_slack:
        slack_bits = bits_arr[n_edges: n_edges + n_slack]
        slack_val = sum(int(b) * (2**k) for k, b in enumerate(slack_bits))
        req_slack = instance.r_limit - res_used
        return slack_val == req_slack

    return True


def build_feasible_mask(
    problem: QAOAProblem,
    instance: CSPPInstance,
    level: FeasibilityLevel = "path",
    reduction_result: Any | None = None,
    qubo: QUBO | None = None,
) -> np.ndarray:
    """Precompute a boolean mask indicating feasibility for all 2^N basis states."""
    n = problem.n_qubits
    mask = np.zeros(2**n, dtype=bool)
    for idx in range(2**n):
        bits = index_to_bits(idx, n)
        if is_feasible(bits, instance, level=level, reduction_result=reduction_result, qubo=qubo):
            mask[idx] = True
    return mask


def postselected_energy(
    probs: np.ndarray,
    diag: np.ndarray,
    feasible_mask: np.ndarray,
    y_min: float = 1e-4,
) -> tuple[float, float]:
    """Compute post-selected energy E_ps = sum_{x in F} P(x) E(x) / sum_{x in F} P(x).

    Returns (E_ps, yield_y). Falls back to unmitigated energy if yield_y < y_min.
    """
    probs_arr = np.asarray(probs, dtype=float)
    diag_arr = np.asarray(diag, dtype=float)
    y = float(np.sum(probs_arr[feasible_mask]))

    if y < y_min:
        # Fall back to unmitigated energy
        return float(np.dot(probs_arr, diag_arr)), y

    e_ps = float(np.dot(probs_arr[feasible_mask], diag_arr[feasible_mask]) / y)
    return e_ps, y


def time_to_solution_99(
    p_optimal: float,
    shots_per_batch: int = 1024,
) -> float:
    """Compute expected quantum shots to observe an optimal solution with 99% confidence.

    TTS_99 = shots_per_batch * ceil( ln(0.01) / ln(1 - P_optimal) ).
    """
    p_opt = float(p_optimal)
    if p_opt <= 0.0 or not math.isfinite(p_opt):
        return float("inf")
    if p_opt >= 1.0:
        return float(shots_per_batch)

    # 1 - P_optimal in (0, 1)
    ln_conf = math.log(0.01)  # ~ -4.60517
    ln_fail = math.log(1.0 - p_opt)
    batches = math.ceil(ln_conf / ln_fail)
    return float(batches * shots_per_batch)


def feasible_conditioned_ratio(
    probs: np.ndarray,
    problem: QAOAProblem,
    feasible_mask: np.ndarray,
    y_min: float = 1e-4,
) -> float:
    """Approximation ratio evaluated strictly over feasible sampled states."""
    e_ps, y = postselected_energy(probs, problem.diag, feasible_mask, y_min=y_min)
    return approximation_ratio(e_ps, problem)


def best_of_s_gap(
    probs: np.ndarray,
    problem: QAOAProblem,
    feasible_mask: np.ndarray,
    s_samples: int = 1000,
    seed: int | None = None,
) -> float:
    """Best-of-S feasible cost gap: difference between minimum feasible sample and ground state."""
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=float)
    p = p / np.sum(p)

    samples = rng.choice(len(p), size=s_samples, p=p)
    feas_samples = [idx for idx in samples if feasible_mask[idx]]

    if not feas_samples:
        return float("inf")

    best_sample_energy = float(np.min(problem.diag[feas_samples]))
    return float(best_sample_energy - problem.e_min)
