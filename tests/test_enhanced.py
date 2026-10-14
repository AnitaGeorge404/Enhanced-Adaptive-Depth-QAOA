"""Tests for Enhanced DDQAOA research improvements (Improvements 1-5)."""
from __future__ import annotations

import json
import math
import numpy as np
import pytest

from ddqaoa import (
    AdaptiveTransfer,
    ControlledAdamOptimizer,
    ConvergenceMonitor,
    EnhancedConvergenceMonitor,
    OptimizerConfig,
    ResourceBudget,
    ResourceDecision,
    adaptive_transfer,
    aggregate_benchmark_results,
    build_problem,
    fourier_transfer,
    generate_instance,
    interp_transfer,
    paired_bootstrap_ci,
    paired_statistical_tests,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_fixed_qaoa,
    solve_classical,
)


# ==============================================================================
# Improvement 1: Enhanced Convergence Monitor
# ==============================================================================

def test_enhanced_monitor_synthetic_plateau():
    m = EnhancedConvergenceMonitor(eps_abs=1e-4, eps_rel=1e-4, k=10, min_steps=15)
    # 25 flat steps
    fired = [m.update(5.0) for _ in range(25)]
    # Must not fire before min_steps (steps 1 through 14)
    assert not any(fired[:14])
    # Must fire once min_steps (step 15) is reached
    assert fired[14] is True



def test_enhanced_monitor_no_premature_jump_before_min_steps():
    m = EnhancedConvergenceMonitor(eps_abs=1e-3, eps_rel=1e-3, sigma_rel=1e-2, k=5, min_steps=20)
    # Energy with tiny variance from step 0
    for i in range(19):
        assert not m.update(3.0 + 1e-6 * (i % 2))
    # At or after step 20, convergence can be triggered
    assert m.update(3.0) is True


def test_enhanced_monitor_descending_slope_prevents_premature_stopping():
    m = EnhancedConvergenceMonitor(
        eps_abs=1e-4,
        eps_rel=1e-4,
        sigma_rel=1.0,  # high variance threshold so variance alone would trigger
        k=5,
        min_steps=10,
        slope_tol=1e-4,
    )
    # Steadily descending sequence
    steps = [10.0 - 0.2 * i for i in range(25)]
    fired = [m.update(e) for e in steps]
    # Because slope is strongly negative (-0.2), monitor must not fire
    assert not any(fired)


def test_enhanced_monitor_scale_invariance():
    # Scale by 10^6: relative monitor should still detect plateau
    m = EnhancedConvergenceMonitor(eps_abs=1e-4, eps_rel=1e-4, sigma_rel=1e-5, k=10, min_steps=15)
    base = 1e6
    fired = [m.update(base + 0.01 * (i % 2)) for i in range(30)]
    assert any(fired[15:])


def test_enhanced_monitor_diagnostics():
    m = EnhancedConvergenceMonitor(k=10, min_steps=10)
    for e in [5.0, 4.0, 3.5, 3.5]:
        m.update(e)
    d = m.diagnostics()
    assert d["depth_step"] == 4
    assert np.isclose(d["depth_best"], 3.5)
    assert np.isclose(d["global_best"], 3.5)


# ==============================================================================
# Improvement 2: Adaptive Parameter Transfer
# ==============================================================================

def test_adaptive_transfer_p1():
    g, b = adaptive_transfer([0.5], [0.4], delta=0.15)
    assert len(g) == 2 and len(b) == 2
    assert np.isclose(g[0], 0.5 * 0.85) and np.isclose(g[1], 0.5 * 1.15)
    assert np.isclose(b[0], 0.4 * 1.15) and np.isclose(b[1], 0.4 * 0.85)


def test_adaptive_transfer_higher_depths_shape_and_bounds():
    for p in (2, 3, 5, 8):
        g0 = np.linspace(0.1, 0.8, p)
        b0 = np.linspace(0.8, 0.1, p)
        g_new, b_new = adaptive_transfer(g0, b0)
        assert len(g_new) == p + 1
        assert len(b_new) == p + 1
        assert np.all(np.isfinite(g_new))
        assert np.all(np.isfinite(b_new))
        assert np.all(g_new >= -np.pi) and np.all(g_new <= np.pi)
        assert np.all(b_new >= -np.pi / 2) and np.all(b_new <= np.pi / 2)


def test_fourier_transfer_shapes_and_values():
    for p in (1, 2, 4, 6):
        g0 = np.linspace(0.2, 0.9, p)
        b0 = np.linspace(0.9, 0.2, p)
        g_new, b_new = fourier_transfer(g0, b0)
        assert len(g_new) == p + 1
        assert len(b_new) == p + 1
        assert np.all(np.isfinite(g_new))
        assert np.all(np.isfinite(b_new))


def test_adaptive_transfer_dispatcher():
    dispatcher_adapt = AdaptiveTransfer(strategy="adaptive")
    dispatcher_four = AdaptiveTransfer(strategy="fourier")
    dispatcher_interp = AdaptiveTransfer(strategy="interp")

    g0, b0 = [0.3, 0.6], [0.7, 0.2]
    ga, ba = dispatcher_adapt(g0, b0)
    gf, bf = dispatcher_four(g0, b0)
    gi, bi = dispatcher_interp(g0, b0)

    assert len(ga) == len(gf) == len(gi) == 3
    assert len(ba) == len(bf) == len(bi) == 3


# ==============================================================================
# Improvement 3: Resource-Aware Stopping
# ==============================================================================

def test_resource_budget_cnot_exhaustion():
    budget = ResourceBudget(max_cnots=1000)
    # Below budget
    d1 = budget.evaluate_transition(
        current_depth=2,
        approx_ratio_current=0.90,
        approx_ratio_prev=0.85,
        success_prob_current=0.1,
        cumulative_cnots_used=800,
        cnots_per_layer=50,
    )
    assert not d1.stop

    # Exceeding budget
    d2 = budget.evaluate_transition(
        current_depth=3,
        approx_ratio_current=0.92,
        approx_ratio_prev=0.90,
        success_prob_current=0.12,
        cumulative_cnots_used=1200,
        cnots_per_layer=50,
    )
    assert d2.stop
    assert "Cumulative CNOT budget" in d2.reason


def test_resource_budget_marginal_gain_stopping():
    budget = ResourceBudget(min_marginal_gain=0.01, patience_layers=1, min_depth=2)
    # First transition p=1 -> 2: marginal gain = 0.05 (good)
    d1 = budget.evaluate_transition(
        current_depth=2,
        approx_ratio_current=0.85,
        approx_ratio_prev=0.80,
        success_prob_current=0.1,
        cumulative_cnots_used=500,
        cnots_per_layer=20,
    )
    assert not d1.stop

    # Second transition p=2 -> 3: marginal gain = 0.002 < 0.01 (diminishing return)
    d2 = budget.evaluate_transition(
        current_depth=3,
        approx_ratio_current=0.852,
        approx_ratio_prev=0.85,
        success_prob_current=0.11,
        cumulative_cnots_used=1000,
        cnots_per_layer=20,
    )
    assert d2.stop
    assert "Diminishing returns" in d2.reason


def test_resource_budget_target_ratio_early_stopping():
    budget = ResourceBudget(target_approx_ratio=0.95)
    d = budget.evaluate_transition(
        current_depth=2,
        approx_ratio_current=0.96,
        approx_ratio_prev=0.80,
        success_prob_current=0.2,
        cumulative_cnots_used=500,
        cnots_per_layer=20,
    )
    assert d.stop
    assert "Target approximation ratio" in d.reason


# ==============================================================================
# Improvement 4: Optimization Control & Parameter Stability
# ==============================================================================

def test_controlled_adam_gradient_clipping():
    opt = ControlledAdamOptimizer(stepsize=0.01, grad_clip=1.0)
    # Define a steep objective function with large gradient
    def steep_func(x):
        return 100.0 * x[0] ** 2

    import pennylane as qml
    from pennylane import numpy as pnp
    x = pnp.array([10.0], requires_grad=True)
    g, forward = opt.compute_grad(steep_func, (x,), {})
    norm = float(np.linalg.norm(g[0]))
    assert np.isclose(norm, 1.0, atol=1e-5)
    assert opt.last_grad_norm >= 1.0


def test_controlled_adam_depth_decay():
    opt = ControlledAdamOptimizer(stepsize=0.1, depth_decay=0.2)
    assert np.isclose(opt.stepsize, 0.1)

    opt.set_depth(4)
    expected = 0.1 / math.sqrt(1.0 + 0.2 * 3)
    assert np.isclose(opt.stepsize, expected)


def test_controlled_adam_nan_recovery():
    opt = ControlledAdamOptimizer(stepsize=0.01, recover_nan=True)
    def bad_func(x):
        return float("nan")

    from pennylane import numpy as pnp
    x = pnp.array([1.0], requires_grad=True)
    g, _ = opt.compute_grad(bad_func, (x,), {})
    # Should safely return zero grad to avoid destroying parameters
    assert np.all(np.asarray(g[0]) == 0.0)


# ==============================================================================
# Improvement 5: Statistical Validation Module
# ==============================================================================

def test_paired_bootstrap_ci_exact_difference():
    # Construct exact difference +0.1
    b = [0.8, 0.85, 0.9, 0.75, 0.88]
    e = [x + 0.1 for x in b]
    res = paired_bootstrap_ci(b, e, n_bootstraps=500, seed=123)
    assert np.isclose(res["mean_diff"], 0.1)
    assert np.isclose(res["ci_lower"], 0.1)
    assert np.isclose(res["ci_upper"], 0.1)


def test_paired_statistical_tests():
    b = [0.80, 0.82, 0.85, 0.79, 0.83, 0.84, 0.81]
    e = [x + 0.05 for x in b]
    res = paired_statistical_tests(b, e)
    assert res["n"] == 7
    assert res["mean_diff"] > 0.04
    assert res["t_pvalue"] < 0.001
    assert res["wilcoxon_pvalue"] < 0.05


def test_aggregate_benchmark_results():
    runs = [
        {"instance": 0, "method": "ddqaoa", "approx_ratio": 0.92, "success_prob": 0.15, "cumulative_cnots": 50000, "final_depth": 5, "status": "SUCCESS"},
        {"instance": 0, "method": "ddqaoa_enhanced", "approx_ratio": 0.96, "success_prob": 0.22, "cumulative_cnots": 40000, "final_depth": 4, "status": "SUCCESS"},
        {"instance": 1, "method": "ddqaoa", "approx_ratio": 0.90, "success_prob": 0.10, "cumulative_cnots": 60000, "final_depth": 6, "status": "SUCCESS"},
        {"instance": 1, "method": "ddqaoa_enhanced", "approx_ratio": 0.95, "success_prob": 0.18, "cumulative_cnots": 45000, "final_depth": 4, "status": "SUCCESS"},
    ]
    summary = aggregate_benchmark_results(runs, baseline_method="ddqaoa")
    assert "ddqaoa" in summary["methods"]
    assert "ddqaoa_enhanced" in summary["methods"]
    assert "ddqaoa_enhanced" in summary["paired_against_baseline"]
    paired = summary["paired_against_baseline"]["ddqaoa_enhanced"]
    assert paired["approx_ratio"]["ci"]["mean_diff"] > 0


# ==============================================================================
# End-to-End Integration & Reproducibility Tests
# ==============================================================================

def test_run_enhanced_ddqaoa_smoke():
    inst = generate_instance(n_nodes=4, seed=7)
    _, problem = build_problem(inst)
    res = run_enhanced_ddqaoa(
        problem,
        p_max=3,
        max_steps=120,
        stepsize=0.05,
        k=15,
        min_steps_per_depth=10,
    )
    assert 1 <= res.final_depth <= 3
    assert 0.0 <= res.approx_ratio <= 1.0
    assert res.cumulative_cnots > 0
    assert res.stopping_reason != ""
    assert len(res.grad_norms) > 0
    # Serialization test
    summary = res.summary()
    serialized = json.dumps(summary)
    assert isinstance(serialized, str)


def test_reproducibility_with_fixed_seed():
    inst = generate_instance(n_nodes=4, seed=42)
    _, problem = build_problem(inst)
    res1 = run_fixed_qaoa(problem, p=2, steps=30, stepsize=0.02, seed=123)
    res2 = run_fixed_qaoa(problem, p=2, steps=30, stepsize=0.02, seed=123)
    assert np.allclose(res1.energies, res2.energies)
    assert np.isclose(res1.approx_ratio, res2.approx_ratio)
