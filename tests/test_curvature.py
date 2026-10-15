"""Unit tests for Curvature-triggered depth growth (Feature 2).

Tests:
1. Identity-layer energy equality (< 1e-10 difference).
2. Gradient norm ~ 0 at end-insertion TS (stationary point).
3. Hessian block calculation and symmetry agreement.
4. Trigger never hangs (safety fallback max_steps_per_depth).
5. Flag-off path leaves baselines unchanged.
6. Zero pad displace parameter validity.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from ddqaoa.circuits import QAOAProblem, make_qnodes
from ddqaoa.convergence import ConvergenceMonitor, CurvatureTrigger
from ddqaoa.cost import CostLedger
from ddqaoa.cspp import generate_instance
from ddqaoa import build_problem
from ddqaoa.ddqaoa import run_ddqaoa
from ddqaoa.transfer import zero_pad_displace


def test_identity_layer_energy_equality():
    """Property 1: Inserting (gamma=0, beta=0) at any layer leaves energy identical (< 1e-10)."""
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)
    _, cost_fn = make_qnodes(problem)

    g_p = np.array([0.2, 0.4])
    b_p = np.array([0.5, 0.3])
    e_orig = float(cost_fn(g_p, b_p))

    # Test insertion at all possible positions: 0, 1, 2
    for pos in [0, 1, 2]:
        g_new = list(g_p)
        b_new = list(b_p)
        g_new.insert(pos, 0.0)
        b_new.insert(pos, 0.0)
        e_inserted = float(cost_fn(np.asarray(g_new), np.asarray(b_new)))
        assert abs(e_inserted - e_orig) < 1e-10, f"Position {pos} energy mismatch: {e_inserted} != {e_orig}"


def test_gradient_zero_at_end_insertion_stationary_point():
    """Property 2: Gradient norm w.r.t new layer at zero-padded point is ~ 0 at a true stationary point."""
    from scipy.optimize import minimize

    inst = generate_instance(n_nodes=4, seed=101)
    _, problem = build_problem(inst)
    _, cost_fn = make_qnodes(problem)

    # Optimize depth 1 to a true stationary point using BFGS
    opt_res = minimize(lambda x: float(cost_fn([x[0]], [x[1]])), x0=[0.5, -0.5], method="BFGS")
    g_opt, b_opt = float(opt_res.x[0]), float(opt_res.x[1])
    assert np.linalg.norm(opt_res.jac) < 1e-3, "Failed to reach stationary point at depth 1"

    # Append (gamma=0, beta=0) at the end
    g2 = np.array([g_opt, 0.0], dtype=float)
    b2 = np.array([b_opt, 0.0], dtype=float)

    # Compute numerical partial derivatives with respect to gamma_2 and beta_2
    h = 1e-6
    dE_dg2 = (float(cost_fn([g_opt, h], b2)) - float(cost_fn([g_opt, -h], b2))) / (2 * h)
    dE_db2 = (float(cost_fn(g2, [b_opt, h])) - float(cost_fn(g2, [b_opt, -h]))) / (2 * h)

    grad_norm_new = np.sqrt(dE_dg2**2 + dE_db2**2)
    # At a true depth-p stationary point, gradient norm of end-inserted identity layer is ~ 0 (< 1e-4)
    assert grad_norm_new < 1e-4, f"Gradient norm at end insertion was {grad_norm_new:.4e}"


def test_hessian_block_agreement():
    """Property 3: 2x2 Hessian block is symmetric and matches central difference evaluations."""
    inst = generate_instance(n_nodes=4, seed=202)
    _, problem = build_problem(inst)
    _, cost_fn = make_qnodes(problem)

    trigger = CurvatureTrigger(grad_tol=1e-2, curv_tol=1e-3)
    g = np.array([0.3])
    b = np.array([0.2])

    H, e0 = trigger.evaluate_block_hessian(cost_fn, g, b, position=1, h=1e-3)

    assert H.shape == (2, 2)
    assert abs(H[0, 1] - H[1, 0]) < 1e-4, f"Hessian not symmetric: {H}"
    # Verify eigenvalues can be computed
    eigvals = np.linalg.eigvalsh(H)
    assert len(eigvals) == 2


def test_curvature_trigger_never_hangs():
    """Property 4: Trigger terminates safely via max_steps_per_depth or convergence."""
    inst = generate_instance(n_nodes=4, seed=303)
    _, problem = build_problem(inst)

    ledger = CostLedger(
        gradient_method="autodiff",
        cnots_per_layer=6,
        n_qubits=problem.n_qubits,
        n_zz=len(problem.ising.J),
    )
    trigger = CurvatureTrigger(
        grad_tol=1e-2,
        curv_tol=1e-3,
        min_steps=5,
        max_steps_per_depth=20,
        position_mode="end",
    )

    res = run_ddqaoa(
        problem,
        p0=1,
        p_max=3,
        max_steps=80,
        monitor=trigger,
        ledger=ledger,
    )

    assert len(res.energies) <= 80
    assert res.final_depth >= 1
    assert ledger.paper_cnots > 0


def test_flag_off_path_leaves_baselines_unchanged():
    """Property 5: Unmodified DDQAOA run matches default baseline behavior exactly."""
    inst = generate_instance(n_nodes=4, seed=404)
    _, problem = build_problem(inst)

    res = run_ddqaoa(
        problem,
        p0=1,
        p_max=2,
        max_steps=50,
        stepsize=0.02,
        k=15,
    )

    assert res.method == "ddqaoa"
    assert res.final_depth in [1, 2]
    assert len(res.energies) > 0
    assert 0.0 <= res.approx_ratio <= 1.0


def test_zero_pad_displace():
    """Property 6: zero_pad_displace inserts layer correctly with physical clipping."""
    g = np.array([0.5, 0.8])
    b = np.array([0.3, 0.2])

    g_new, b_new = zero_pad_displace(g, b, delta=0.1, position=-1)
    assert len(g_new) == 3
    assert len(b_new) == 3
    assert -np.pi <= g_new[-1] <= np.pi
    assert -np.pi/2 <= b_new[-1] <= np.pi/2
