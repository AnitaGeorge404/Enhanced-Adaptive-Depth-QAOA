"""Unit and integration tests for Readout Error Mitigation (REM) (Feature 5)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from ddqaoa import (
    NoiseConfig,
    NoiseEvaluationResult,
    apply_readout_mitigation,
    apply_readout_noise,
    build_feasible_mask,
    build_problem,
    evaluate_under_noise,
    generate_instance,
    postselected_energy,
    project_to_simplex,
    run_enhanced_ddqaoa,
    success_probability,
)


def test_simplex_projection():
    """Verify projection onto the probability simplex."""
    # 1. Already valid probability vector
    p = np.array([0.2, 0.5, 0.3])
    proj = project_to_simplex(p)
    assert np.allclose(p, proj)

    # 2. Vector with negative components and sum != 1
    v = np.array([-0.1, 0.8, 0.5, -0.2])
    proj_v = project_to_simplex(v)
    assert np.all(proj_v >= 0.0)
    assert np.isclose(np.sum(proj_v), 1.0)

    # 3. Uniform target
    ones = np.ones(4)
    proj_ones = project_to_simplex(ones)
    assert np.allclose(proj_ones, 0.25)


def test_readout_noise_and_exact_inversion():
    """Verify that in the absence of finite-shot sampling, M^{-1} M p = p exactly."""
    n = 4
    rng = np.random.default_rng(42)
    p_true = rng.dirichlet(np.ones(1 << n))

    p_01 = 0.03
    p_10 = 0.05

    noisy_p = apply_readout_noise(p_true, p_01=p_01, p_10=p_10, n_qubits=n)
    assert np.isclose(np.sum(noisy_p), 1.0)
    assert not np.allclose(noisy_p, p_true)

    mitigated_p = apply_readout_mitigation(noisy_p, p_01=p_01, p_10=p_10, n_qubits=n, project=False)
    assert np.allclose(mitigated_p, p_true, atol=1e-12)


def test_rem_recovers_success_probability_under_shot_noise():
    """Verify that REM improves ground state probability recovery under physical readout errors and shots."""
    inst = generate_instance(n_nodes=4, seed=123, prune=True)
    _, problem = build_problem(inst)
    n = problem.n_qubits

    # Ground state distribution concentrated at ground states
    p_ideal = np.zeros(1 << n)
    p_ideal[problem.ground_states[0]] = 0.8
    p_ideal[problem.ground_states[0] ^ 1] = 0.2

    p_01 = 0.04
    p_10 = 0.04

    # Apply physical readout error channel
    noisy_phys = apply_readout_noise(p_ideal, p_01=p_01, p_10=p_10, n_qubits=n)

    # Simulate finite measurement sampling
    shots = 5000
    rng = np.random.default_rng(999)
    counts = rng.multinomial(shots, noisy_phys)
    p_sampled = counts / float(shots)

    # Apply Readout Error Mitigation
    p_mitigated = apply_readout_mitigation(p_sampled, p_01=p_01, p_10=p_10, n_qubits=n, project=True)

    succ_ideal = success_probability(p_ideal, problem)
    succ_noisy = success_probability(p_sampled, problem)
    succ_mit = success_probability(p_mitigated, problem)

    assert succ_noisy < succ_ideal
    # REM must substantially restore success probability towards ideal
    assert succ_mit > succ_noisy
    assert math.isclose(succ_mit, succ_ideal, abs_tol=0.08)


def test_rem_postselection_fidelity_recovery():
    """Verify that REM restores valid path post-selection yield degraded by measurement noise."""
    inst = generate_instance(n_nodes=4, seed=42, prune=True)
    qubo, problem = build_problem(inst)
    n = problem.n_qubits
    f_mask = build_feasible_mask(problem, inst, level="path", qubo=qubo)

    # Synthetic distribution concentrated on feasible states
    p_ideal = np.zeros(1 << n)
    feas_indices = np.flatnonzero(f_mask)
    p_ideal[feas_indices[0]] = 0.6
    p_ideal[feas_indices[1]] = 0.4

    # Postselected yield under ideal conditions is 1.0
    _, y_ideal = postselected_energy(p_ideal, problem.diag, f_mask)
    assert np.isclose(y_ideal, 1.0)

    # Under 5% bit-flip readout noise, bits flip and feasible yield drops
    p_noisy = apply_readout_noise(p_ideal, p_01=0.05, p_10=0.05, n_qubits=n)
    _, y_noisy = postselected_energy(p_noisy, problem.diag, f_mask)
    assert y_noisy < 0.90

    # REM restores the true feasible yield
    p_mit = apply_readout_mitigation(p_noisy, p_01=0.05, p_10=0.05, n_qubits=n, project=True)
    _, y_mit = postselected_energy(p_mit, problem.diag, f_mask)
    assert y_mit > y_noisy
    assert y_mit > 0.95


def test_evaluate_under_noise_with_rem_integration():
    """Verify evaluate_under_noise reports REM mitigation metrics."""
    inst = generate_instance(n_nodes=4, seed=42, prune=True)
    _, problem = build_problem(inst)

    gammas = np.array([0.2, 0.3])
    betas = np.array([0.1, 0.2])

    noise = NoiseConfig(
        p_single=0.001,
        p_two=0.01,
        p_readout_01=0.03,
        p_readout_10=0.03,
        shots=2000,
        seed=42,
    )

    eval_res = evaluate_under_noise(
        gammas=gammas,
        betas=betas,
        problem=problem,
        noise=noise,
        apply_rem=True,
    )

    assert eval_res.mitigated_approx_ratio is not None
    assert eval_res.mitigated_success_prob is not None
    assert eval_res.rem_recovery_ratio is not None
    assert eval_res.mitigated_approx_ratio > 0.0

    summary = eval_res.summary()
    assert "mitigated_approx_ratio" in summary
    assert "p_readout_01" in summary["noise_config"]
