"""Unit tests for noise-aware stopping, fidelity surrogate, and rollback (Feature 3).

Tests:
1. FidelityModel limits: eps=0 -> E_ideal; huge eps -> E_mm.
2. Rollback returns best-noisy-depth parameters.
3. Predictive stopping halts when predicted gain <= 0.
4. Determinism of noise-aware DDQAOA given seed.
5. Depolarizing simulation on default.mixed produces valid probabilities.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from ddqaoa import (
    FidelityModel,
    NoiseAwareStopper,
    build_problem,
    generate_instance,
    make_noisy_qnodes,
    run_noise_aware_ddqaoa,
)


def test_fidelity_model_limits():
    """Property 1: FidelityModel limits: eps=0 -> E_ideal; huge eps -> E_mm."""
    model_noiseless = FidelityModel(n_qubits=4, n_zz=6, n_z=2, eps2=0.0, eps1=0.0, e_mm=5.0)
    assert model_noiseless.fidelity(p=1) == 1.0
    assert model_noiseless.fidelity(p=5) == 1.0
    assert model_noiseless.predict_noisy_energy(p=3, e_ideal=2.0) == 2.0

    # Huge noise limit
    model_noisy = FidelityModel(n_qubits=4, n_zz=6, n_z=2, eps2=0.99, eps1=0.99, e_mm=5.0)
    assert model_noisy.fidelity(p=2) < 1e-10
    pred_mm = model_noisy.predict_noisy_energy(p=2, e_ideal=1.0)
    assert math.isclose(pred_mm, 5.0, abs_tol=1e-5)


def test_fidelity_model_fitting_and_mae():
    """Property 1b: Fidelity calibration fitting and MAE evaluation."""
    model = FidelityModel(n_qubits=4, n_zz=5, n_z=0, eps2=0.001, e_mm=10.0)
    # Synthetic calibration points generated with true eps2 = 0.005
    calib = [
        (1, 2.0, (1 - 0.005)**5 * 2.0 + (1 - (1 - 0.005)**5) * 10.0),
        (2, 1.5, (1 - 0.005)**10 * 1.5 + (1 - (1 - 0.005)**10) * 10.0),
    ]
    fitted_eps = model.fit(calib)
    assert 0.001 < fitted_eps < 0.01

    mae = model.prediction_mae(calib)
    assert mae < 0.1


def test_noise_aware_stopper_reactive_rollback():
    """Property 2: Reactive stopper triggers rollback to the best noisy depth."""
    stopper = NoiseAwareStopper(rule="reactive", m=2, tau=0.05, min_depth=1)

    # Depth 1: energy 4.0
    g1, b1 = np.array([0.1]), np.array([0.2])
    d1 = stopper.evaluate_depth_transition(1, g1, b1, noisy_energy=4.0)
    assert not d1.stop

    # Depth 2: energy 3.0 (improved! becomes best)
    g2, b2 = np.array([0.1, 0.2]), np.array([0.2, 0.3])
    d2 = stopper.evaluate_depth_transition(2, g2, b2, noisy_energy=3.0)
    assert not d2.stop
    assert stopper.best_depth == 2

    # Depth 3: energy 3.1 (worse, stagnant count 1)
    g3, b3 = np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])
    d3 = stopper.evaluate_depth_transition(3, g3, b3, noisy_energy=3.1)
    assert not d3.stop

    # Depth 4: energy 3.2 (worse again, stagnant count 2 -> STOP and ROLLBACK)
    g4, b4 = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4, 0.5])
    d4 = stopper.evaluate_depth_transition(4, g4, b4, noisy_energy=3.2)
    assert d4.stop is True
    assert d4.rollback is True
    assert d4.best_depth == 2
    assert np.allclose(d4.best_gammas, g2)
    assert np.allclose(d4.best_betas, b2)


def test_noise_aware_stopper_predictive():
    """Property 3: Predictive stopper stops when surrogate predicted gain <= 0."""
    # Model with high noise where next depth is predicted to worsen
    model = FidelityModel(n_qubits=4, n_zz=6, eps2=0.05, e_mm=10.0)
    stopper = NoiseAwareStopper(rule="predictive", fidelity_model=model, min_depth=1)

    g1, b1 = np.array([0.2]), np.array([0.3])
    # At depth 1, ideal energy is 2.0; estimated next gain is tiny 1e-4
    # The heavy noise decay will outweigh the tiny ideal gain
    dec = stopper.evaluate_depth_transition(
        current_depth=1,
        gammas=g1,
        betas=b1,
        noisy_energy=model.predict_noisy_energy(1, 2.0),
        ideal_energy=2.0,
        estimated_ideal_gain=1e-4,
    )
    # Predictive stopper should halt expansion
    assert dec.stop is True
    assert dec.predicted_next_gain <= 0.0


def test_run_noise_aware_ddqaoa_deterministic():
    """Property 4: run_noise_aware_ddqaoa is deterministic given the instance and parameters."""
    inst = generate_instance(n_nodes=4, seed=333, prune=True)
    _, problem = build_problem(inst)

    res1 = run_noise_aware_ddqaoa(
        problem,
        eps2=0.005,
        rule="reactive",
        objective_mode="ideal",
        p_max=3,
        max_steps=50,
        stepsize=0.03,
        k=15,
        m=1,
    )

    res2 = run_noise_aware_ddqaoa(
        problem,
        eps2=0.005,
        rule="reactive",
        objective_mode="ideal",
        p_max=3,
        max_steps=50,
        stepsize=0.03,
        k=15,
        m=1,
    )

    assert res1.final_depth == res2.final_depth
    assert math.isclose(res1.best_energy, res2.best_energy, rel_tol=1e-10)
    assert np.allclose(res1.gammas, res2.gammas)


def test_noisy_qnodes_channel_execution():
    """Property 5: Depolarizing simulation on default.mixed yields valid probabilities."""
    inst = generate_instance(n_nodes=4, seed=444, prune=True)
    _, problem = build_problem(inst)

    probs_fn, cost_fn = make_noisy_qnodes(problem, eps2=0.01, eps1=0.001)
    g = np.array([0.2])
    b = np.array([0.3])

    probs = probs_fn(g, b)
    assert len(probs) == 2**problem.n_qubits
    assert math.isclose(float(np.sum(probs)), 1.0, abs_tol=1e-5)
    assert np.all(probs >= -1e-7)

    cost = cost_fn(g, b)
    assert math.isfinite(float(cost))
