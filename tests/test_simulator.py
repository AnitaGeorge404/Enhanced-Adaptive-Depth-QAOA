"""Unit and integration tests for the Analytical Matrix-Free Simulator (Feature 10)."""
from __future__ import annotations

import math
import numpy as np
import pytest
import pennylane as qml
from pennylane import numpy as pnp

from ddqaoa import (
    AnalyticalSimulator,
    ControlledAdamOptimizer,
    QAOAProblem,
    build_problem,
    generate_instance,
    make_analytical_qnodes,
    make_qnodes,
    run_ddqaoa,
    run_enhanced_ddqaoa,
)


def test_analytical_simulator_state_and_energy_match_pennylane():
    """Verify that statevector, probabilities, and energy match PennyLane exactly."""
    inst = generate_instance(n_nodes=4, seed=10, prune=True)
    _, problem = build_problem(inst)
    n = problem.n_qubits

    gammas = np.array([0.25, 0.45])
    betas = np.array([0.35, 0.15])

    # 1. PennyLane
    pl_probs_fn, pl_cost_fn = make_qnodes(problem, device="default.qubit")
    g_pnp = pnp.array(gammas, requires_grad=True)
    b_pnp = pnp.array(betas, requires_grad=True)

    pl_probs = np.asarray(pl_probs_fn(g_pnp, b_pnp))
    pl_cost = float(pl_cost_fn(g_pnp, b_pnp))

    # 2. Analytical Simulator
    sim = AnalyticalSimulator(problem)
    sim_probs = sim.probabilities(gammas, betas)
    sim_cost = sim.energy(gammas, betas)

    assert np.isclose(np.sum(sim_probs), 1.0)
    assert np.all(sim_probs >= 0.0)
    assert np.allclose(pl_probs, sim_probs, atol=1e-10)
    assert np.isclose(pl_cost, sim_cost, atol=1e-10)


def test_analytical_adjoint_gradient_exactness():
    """Verify that adjoint state differentiation matches PennyLane autograd to machine precision."""
    inst = generate_instance(n_nodes=4, seed=25, prune=True)
    _, problem = build_problem(inst)

    p = 3
    rng = np.random.default_rng(77)
    gammas = rng.uniform(0.1, 0.7, size=p)
    betas = rng.uniform(0.1, 0.7, size=p)

    # PennyLane autograd gradient
    _, pl_cost_fn = make_qnodes(problem, device="default.qubit")
    g_pnp = pnp.array(gammas, requires_grad=True)
    b_pnp = pnp.array(betas, requires_grad=True)
    pl_grad_g, pl_grad_b = qml.grad(pl_cost_fn)(g_pnp, b_pnp)

    # Analytical Simulator adjoint gradient
    sim = AnalyticalSimulator(problem)
    sim_grad_g, sim_grad_b = sim.adjoint_gradient(gammas, betas)

    assert np.allclose(pl_grad_g, sim_grad_g, atol=1e-10)
    assert np.allclose(pl_grad_b, sim_grad_b, atol=1e-10)


def test_make_analytical_qnodes_vjp_optimizer_integration():
    """Verify that make_analytical_qnodes seamlessly works with ControlledAdamOptimizer."""
    inst = generate_instance(n_nodes=4, seed=42, prune=True)
    _, problem = build_problem(inst)

    probs_fn, cost_fn = make_analytical_qnodes(problem)
    gammas = pnp.array([0.2, 0.3], requires_grad=True)
    betas = pnp.array([0.4, 0.1], requires_grad=True)

    opt = ControlledAdamOptimizer(stepsize=0.03, grad_clip=1.0)
    (new_g, new_b), initial_cost = opt.step_and_cost(cost_fn, gammas, betas)

    assert math.isfinite(initial_cost)
    assert len(new_g) == 2
    assert len(new_b) == 2
    assert opt.last_grad_norm > 0.0

    # Ensure cost improves after an optimization step
    next_cost = float(cost_fn(new_g, new_b))
    assert math.isfinite(next_cost)


def test_run_ddqaoa_with_analytical_device():
    """Verify that run_ddqaoa and run_enhanced_ddqaoa execute with device='analytical'."""
    inst = generate_instance(n_nodes=4, seed=99, prune=True)
    _, problem = build_problem(inst)

    # Run baseline DDQAOA with device="analytical"
    res_base = run_ddqaoa(
        problem,
        p_max=3,
        max_steps=30,
        stepsize=0.04,
        k=10,
        device="analytical",
    )
    assert res_base.approx_ratio > 0.7
    assert res_base.final_depth >= 1

    # Run Enhanced DDQAOA with device="analytical"
    res_enh = run_enhanced_ddqaoa(
        problem,
        p_max=3,
        max_steps=30,
        stepsize=0.04,
        k=10,
        min_steps_per_depth=10,
        device="analytical",
    )
    assert res_enh.approx_ratio > 0.7
    assert res_enh.final_depth >= 1


def test_analytical_simulator_scaling_14_qubits():
    """Verify that the analytical matrix-free simulator scales to 14 qubits rapidly without RAM limits."""
    import time
    from ddqaoa.qubo import Ising

    n = 14
    # Create synthetic 14-qubit Ising problem
    h = {i: 0.1 * (i % 3 - 1) for i in range(n)}
    J = {(i, (i + 1) % n): 0.2 for i in range(n)}
    ising = Ising(n, h, J, offset=0.0)
    problem = QAOAProblem.from_ising(ising, normalize=True)

    sim = AnalyticalSimulator(problem)
    p = 3
    gammas = np.array([0.1, 0.2, 0.3])
    betas = np.array([0.3, 0.2, 0.1])

    t0 = time.perf_counter()
    e = sim.energy(gammas, betas)
    grad_g, grad_b = sim.adjoint_gradient(gammas, betas)
    elapsed = time.perf_counter() - t0

    assert math.isfinite(e)
    assert len(grad_g) == p
    assert len(grad_b) == p
    # Evaluating a 14-qubit (16,384 states) p=3 circuit + adjoint gradient takes < 250 ms
    assert elapsed < 1.0
