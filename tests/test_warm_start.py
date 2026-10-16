"""Unit and integration tests for LP Relaxation Warm-Starting (Feature 2)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from ddqaoa import (
    AnalyticalSimulator,
    build_problem,
    generate_instance,
    get_warm_start_angles,
    make_analytical_qnodes,
    make_qnodes,
    probabilities_to_angles,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    solve_cspp_lp_relaxation,
)


def test_probabilities_to_angles_properties():
    # 1. Neutral 0.5 probability should map exactly to pi/2 (Hadamard state |+>)
    angles = probabilities_to_angles([0.5])
    assert np.isclose(angles[0], math.pi / 2.0)

    # 2. Clamping checks with eps=0.05
    angles_ext = probabilities_to_angles([0.0, 1.0], eps=0.05)
    assert angles_ext[0] > 0.0
    assert angles_ext[1] < math.pi
    assert np.isclose(angles_ext[0], 2.0 * math.asin(math.sqrt(0.05)))
    assert np.isclose(angles_ext[1], 2.0 * math.asin(math.sqrt(0.95)))


def test_cspp_lp_relaxation_feasibility():
    inst = generate_instance(n_nodes=4, seed=42, prune=True)
    sol = solve_cspp_lp_relaxation(inst)

    assert len(sol) > 0
    assert np.all(sol >= 0.0)
    assert np.all(sol <= 1.0)

    # Flow out of source must equal 1
    out_s = sum(sol[k] for k, e in enumerate(inst.edges) if e[0] == inst.source and k < len(sol))
    assert np.isclose(out_s, 1.0, atol=1e-3)


def test_warm_start_initial_energy_superiority():
    """Verify that warm-started ansatz starts at higher initial approximation ratio than uniform superposition."""
    inst = generate_instance(n_nodes=4, seed=123, prune=True)
    qubo, problem = build_problem(inst)

    # 1. Standard uniform initialization |+>^{\otimes N}
    sim_uniform = AnalyticalSimulator(problem)
    # Energy at gamma=0, beta=0 (initial state expectation)
    e_uniform = sim_uniform.energy(np.array([0.0]), np.array([0.0]))

    # 2. Warm-started initialization
    theta_ws = get_warm_start_angles(inst, eps=0.05)
    sim_warm = AnalyticalSimulator(problem, warm_start_angles=theta_ws)
    e_warm = sim_warm.energy(np.array([0.0]), np.array([0.0]))

    # Minimizing H_C: warm-started state should achieve lower energy
    assert e_warm < e_uniform


def test_run_enhanced_ddqaoa_with_warm_starting():
    """Run Enhanced DDQAOA seeded by LP relaxation."""
    inst = generate_instance(n_nodes=4, seed=77, prune=True)
    qubo, problem = build_problem(inst)
    theta_ws = get_warm_start_angles(inst, eps=0.05)

    res = run_enhanced_ddqaoa(
        problem,
        p_max=3,
        max_steps=40,
        stepsize=0.04,
        k=12,
        min_steps_per_depth=10,
        device="analytical",
        warm_start_angles=theta_ws,
    )

    assert res.approx_ratio > 0.85
    assert res.final_depth >= 1
    assert math.isfinite(res.best_energy)
