"""Property-based tests for optimality-preserving instance reduction (Feature 1).

Tests (property-based, seeded, >= 200 random instances, n=4..7):
1. Exact optimal cost of reduced == original.
2. With R1 only, the SET of feasible paths is identical.
3. lift(optimal reduced solution) is an optimal original solution.
4. QUBO ground state of the reduced model decodes to an optimal original path.
5. Idempotence: reduce_instance(reduced_instance) == reduced_instance.
6. Infeasible instance handled gracefully without crashing.
7. Tie handling keeps all optimal paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from ddqaoa.circuits import QAOAProblem, index_to_bits, ising_diagonal
from ddqaoa.cspp import CSPPInstance, generate_instance, simple_paths, solve_classical
from ddqaoa.qubo import cspp_to_qubo, decode, qubo_to_ising
from ddqaoa.reduction import reduce_instance, dijkstra_distances


def test_reduction_optimal_cost_preservation_200_instances():
    """Property 1: exact optimal cost of reduced == original across 200 instances."""
    rng = np.random.default_rng(12345)
    seeds = rng.integers(1, 1_000_000, size=200)

    for i, s in enumerate(seeds):
        n = int(rng.integers(4, 8))
        prob = float(rng.uniform(0.6, 1.0))
        inst = generate_instance(n_nodes=n, seed=int(s), edge_prob=prob, prune=False)
        orig_path, orig_cost = solve_classical(inst)
        if orig_path is None:
            continue

        res = reduce_instance(inst)
        assert res.is_feasible is True
        red_path, red_cost = solve_classical(res.reduced_instance)
        # Optimal cost must match exactly
        assert red_path is not None, f"Instance {i} (seed {s}) failed to find path in reduced instance"
        assert np.isclose(red_cost, orig_cost), f"Mismatch in cost: red={red_cost} != orig={orig_cost}"


def test_reduction_r1_only_feasible_paths_preserved():
    """Property 2: with R1 only (no cost bound), the set of feasible paths is identical."""
    rng = np.random.default_rng(54321)
    for s in rng.integers(1, 1_000_000, size=50):
        n = int(rng.integers(4, 7))
        inst = generate_instance(n_nodes=n, seed=int(s), edge_prob=0.8, prune=False)
        orig_paths = [p for p in simple_paths(inst) if inst.path_resource(p) <= inst.r_limit]
        if not orig_paths:
            continue

        # Run with upper_bound=inf so R2 does not prune any feasible paths
        res = reduce_instance(inst, upper_bound=float("inf"), lagrangian=False)
        assert res.preserves_all_feasible is True
        red_paths = [p for p in simple_paths(res.reduced_instance)
                     if res.reduced_instance.path_resource(p) <= res.reduced_instance.r_limit]

        # Sets of paths must be identical
        orig_set = {tuple(p) for p in orig_paths}
        red_set = {tuple(p) for p in red_paths}
        assert orig_set == red_set, f"Mismatch in feasible paths: orig={orig_set} != red={red_set}"


def test_reduction_lift_operator():
    """Property 3: lift(optimal reduced solution) produces an optimal original solution."""
    rng = np.random.default_rng(99999)
    for s in rng.integers(1, 1_000_000, size=50):
        n = int(rng.integers(4, 7))
        inst = generate_instance(n_nodes=n, seed=int(s), prune=False)
        _, orig_cost = solve_classical(inst)
        if not np.isfinite(orig_cost):
            continue

        res = reduce_instance(inst)
        red_path, red_cost = solve_classical(res.reduced_instance)
        assert red_path is not None

        # Build reduced bit assignment for active edges
        red_edges_in_path = set(zip(red_path, red_path[1:]))
        red_bits = {e: 1 if e in red_edges_in_path else 0 for e in res.reduced_instance.edges}

        lifted_assignment = res.lift(red_bits)
        # Check cost of lifted assignment on original instance
        lifted_edges = [e for e, v in lifted_assignment.items() if v == 1]
        lifted_cost = sum(inst.cost[e] for e in lifted_edges)
        lifted_res = sum(inst.resource[e] for e in lifted_edges)

        assert np.isclose(lifted_cost, orig_cost)
        assert lifted_res <= inst.r_limit


def test_reduction_qubo_ground_state_decoding():
    """Property 4: QUBO ground state of reduced model decodes to an optimal original path."""
    rng = np.random.default_rng(77777)
    for s in rng.integers(1, 1_000_000, size=30):
        inst = generate_instance(n_nodes=4, seed=int(s), prune=True)
        orig_path, orig_cost = solve_classical(inst)
        if orig_path is None:
            continue

        res = reduce_instance(inst)
        # Build reduced QUBO with fixed edges handled
        q_red = cspp_to_qubo(
            inst,
            n_slack=res.slack_bits_after,
            fixed_edges=res.fixed_edges,
        )
        ising_red = qubo_to_ising(q_red)
        diag_red = ising_diagonal(ising_red)

        # Ground state index
        min_idx = int(np.argmin(diag_red))
        bits = index_to_bits(min_idx, q_red.n_vars)
        decoded_edges = decode(bits, q_red)

        # Add fixed 1 edges
        all_active_edges = decoded_edges + [e for e, v in res.fixed_edges.items() if v == 1]
        active_cost = sum(inst.cost[e] for e in all_active_edges)
        assert np.isclose(active_cost, orig_cost)


def test_reduction_idempotence():
    """Property 5: reduce_instance(reduced_instance) == reduced_instance."""
    rng = np.random.default_rng(88888)
    for s in rng.integers(1, 1_000_000, size=30):
        inst = generate_instance(n_nodes=5, seed=int(s), prune=False)
        res1 = reduce_instance(inst)
        if not res1.is_feasible:
            continue
        res2 = reduce_instance(res1.reduced_instance)
        assert set(res1.reduced_instance.edges) == set(res2.reduced_instance.edges)
        assert res1.slack_bits_after == res2.slack_bits_after


def test_reduction_infeasible_instance_handling():
    """Property 6: Infeasible instance handled cleanly without crashing."""
    # Graph with no path or r_limit = 0 with positive edge resources
    edges = [(0, 1), (1, 2)]
    cost = {(0, 1): 2, (1, 2): 3}
    res = {(0, 1): 5, (1, 2): 5}
    inst = CSPPInstance(n_nodes=3, source=0, target=2, edges=edges, cost=cost, resource=res, r_limit=4)

    result = reduce_instance(inst)
    assert result.is_feasible is False
    assert len(result.reduced_instance.edges) == 0


def test_reduction_tie_handling_preserves_all_optimal_paths():
    """Property 7: Strict inequality in R2 preserves all optimal paths when ties exist."""
    # Build diamond graph with two paths of identical minimal cost & resource
    # Path A: 0 -> 1 -> 3 (cost 5, res 4)
    # Path B: 0 -> 2 -> 3 (cost 5, res 4)
    # Path C: 0 -> 3 (cost 10, res 4 - suboptimal)
    edges = [(0, 1), (1, 3), (0, 2), (2, 3), (0, 3)]
    cost = {(0, 1): 2, (1, 3): 3, (0, 2): 2, (2, 3): 3, (0, 3): 10}
    resource = {(0, 1): 2, (1, 3): 2, (0, 2): 2, (2, 3): 2, (0, 3): 2}
    inst = CSPPInstance(n_nodes=4, source=0, target=3, edges=edges, cost=cost, resource=resource, r_limit=5)

    res = reduce_instance(inst, upper_bound=5.0)
    # Suboptimal edge (0, 3) must be pruned (cost 10 > 5)
    assert (0, 3) not in res.reduced_instance.edges
    # Both optimal paths must be present
    paths = simple_paths(res.reduced_instance)
    assert [0, 1, 3] in paths
    assert [0, 2, 3] in paths
