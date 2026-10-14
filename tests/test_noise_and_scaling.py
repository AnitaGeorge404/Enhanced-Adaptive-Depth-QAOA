"""Unit and integration tests for NISQ noise modeling and large graph scaling."""
from __future__ import annotations

import numpy as np
import pytest

from ddqaoa import (
    NoiseConfig,
    NoiseEvaluationResult,
    build_problem,
    evaluate_bitstring_energy,
    evaluate_bitstrings_batch,
    evaluate_under_noise,
    find_sparse_ground_state,
    generate_instance,
    make_noisy_qnodes,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    solve_classical,
    solve_classical_heuristic,
    stream_ising_diagonal,
)


def test_noisy_qnodes_density_matrix():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)

    noise = NoiseConfig(p_single=0.01, p_two=0.02, shots=None)
    probs_fn, cost_fn = make_noisy_qnodes(problem, noise)

    gammas = np.array([0.2, 0.3])
    betas = np.array([0.1, 0.4])

    p = probs_fn(gammas, betas)
    assert np.isclose(np.sum(p), 1.0)
    assert np.all(p >= 0.0)

    cost = cost_fn(gammas, betas)
    assert np.isfinite(cost)


def test_noisy_qnodes_with_shots():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)

    shots = 500
    noise = NoiseConfig(p_single=0.005, p_two=0.01, shots=shots, seed=123)
    probs_fn, cost_fn = make_noisy_qnodes(problem, noise)

    gammas = np.array([0.2])
    betas = np.array([0.1])

    p = probs_fn(gammas, betas)
    assert np.isclose(np.sum(p), 1.0)
    # Counts divided by shots must be discrete multiples of 1/shots
    assert np.all(np.isclose((p * shots) % 1.0, 0.0) | np.isclose((p * shots) % 1.0, 1.0))


def test_evaluate_under_noise_retention():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)

    res = run_enhanced_ddqaoa(problem, p_max=3, max_steps=40, stepsize=0.04, k=10, min_steps_per_depth=10)
    noise = NoiseConfig(p_single=0.005, p_two=0.02, shots=1024, seed=77)

    eval_res = evaluate_under_noise(res.gammas, res.betas, problem, noise, method_name=res.method)
    assert isinstance(eval_res, NoiseEvaluationResult)
    assert eval_res.ideal_approx_ratio > 0.0
    assert eval_res.noisy_approx_ratio > 0.0
    assert 0.0 < eval_res.approx_ratio_retention <= 1.2
    assert 0.0 <= eval_res.noisy_success_prob <= 1.0


def test_sparse_bitstring_energy_consistency():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)

    ising = problem.ising
    n = ising.n_qubits

    # Check 50 random bitstrings against problem.diag
    for idx in range(min(50, 2**n)):
        bits = np.array([(idx >> (n - 1 - k)) & 1 for k in range(n)])
        single_e = evaluate_bitstring_energy(bits, ising)
        expected_e = float(problem.diag[idx])
        assert np.isclose(single_e, expected_e, atol=1e-6)


def test_stream_and_find_sparse_ground_state():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)
    ising = problem.ising

    e_min, e_max, ground_indices = find_sparse_ground_state(ising, chunk_size=32)

    exact_min = float(problem.diag.min())
    exact_max = float(problem.diag.max())

    assert np.isclose(e_min, exact_min, atol=1e-6)
    assert np.isclose(e_max, exact_max, atol=1e-6)
    assert len(ground_indices) == len(problem.ground_states)
    assert set(ground_indices) == set(problem.ground_states)


def test_solve_classical_heuristic_large_graph():
    # Generate larger graph with n_nodes = 7
    inst = generate_instance(n_nodes=7, seed=99, edge_prob=0.8)
    opt_path, opt_cost = solve_classical(inst)
    heur_path, heur_cost = solve_classical_heuristic(inst, max_paths=20_000)

    assert heur_path is not None
    assert heur_cost < float("inf")
    # Heuristic with sufficient search budget should reach the exact path or feasible low cost
    assert inst.path_resource(heur_path) <= inst.r_limit
    assert np.isclose(heur_cost, opt_cost)
