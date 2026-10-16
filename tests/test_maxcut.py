"""Unit and integration tests for the Max-Cut Multi-Problem Adapter (Feature 8)."""
from __future__ import annotations

import numpy as np
import pytest

from ddqaoa import (
    MaxCutInstance,
    build_maxcut_problem,
    generate_maxcut_instance,
    maxcut_to_ising,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_fixed_qaoa,
)


def test_maxcut_instance_generation():
    # 1. Complete graph K_4 has 4 * 3 / 2 = 6 edges
    k4 = generate_maxcut_instance(n_nodes=4, graph_type="complete")
    assert len(k4.edges) == 6

    # 2. Cycle graph C_5 has 5 edges
    c5 = generate_maxcut_instance(n_nodes=5, graph_type="cycle")
    assert len(c5.edges) == 5

    # 3. 3-regular graph on 6 nodes has 6 * 3 / 2 = 9 edges
    reg6 = generate_maxcut_instance(n_nodes=6, graph_type="regular", d=3, seed=42)
    assert len(reg6.edges) == 9


def test_maxcut_ising_mapping_matches_classical_optimum():
    """Verify that min energy of Ising matches -max_cut."""
    inst = generate_maxcut_instance(n_nodes=6, graph_type="erdos_renyi", edge_prob=0.7, seed=123)
    best_bits, exact_cut = inst.solve_classical()

    inst, problem = build_maxcut_problem(inst, normalize=False)

    # In unnormalized Ising: E_min = -exact_cut
    assert np.isclose(problem.e_min, -exact_cut)

    # Ground state bitstring must achieve maximum cut
    for gs_idx in problem.ground_states:
        n = inst.n_nodes
        bits = np.array([(gs_idx >> (n - 1 - k)) & 1 for k in range(n)])
        cut = inst.cut_value(bits)
        assert np.isclose(cut, exact_cut)


def test_run_enhanced_ddqaoa_on_maxcut():
    """Run Enhanced DDQAOA on a 3-regular Max-Cut benchmark instance with analytical simulator."""
    inst = generate_maxcut_instance(n_nodes=6, graph_type="regular", d=3, seed=888)
    _, problem = build_maxcut_problem(inst, normalize=True)

    res = run_enhanced_ddqaoa(
        problem,
        p_max=4,
        max_steps=50,
        stepsize=0.04,
        k=15,
        min_steps_per_depth=10,
        device="analytical",
    )

    assert res.approx_ratio > 0.85
    assert res.final_depth >= 1
    assert len(res.energies) > 0
    assert res.cumulative_cnots > 0


def test_run_fixed_qaoa_on_maxcut():
    """Run fixed p=2 QAOA baseline on Max-Cut."""
    inst = generate_maxcut_instance(n_nodes=6, graph_type="cycle", seed=42)
    _, problem = build_maxcut_problem(inst, normalize=True)

    res = run_fixed_qaoa(
        problem,
        p=2,
        steps=50,
        stepsize=0.04,
        device="analytical",
    )

    assert res.approx_ratio > 0.75
    assert res.final_depth == 2
