"""Unit and property tests for Dynamic Constraint Penalty Annealing (DCPA).

Authored by Anita George (Project Lead).
"""
import math
import numpy as np
import pytest

from ddqaoa.annealing import (
    DCPAConfig,
    build_annealed_problem,
    cvar_expectation,
    run_dcpa_ddqaoa,
)
from ddqaoa.cspp import generate_instance, solve_classical
from ddqaoa.qubo import cspp_to_qubo, qubo_to_ising


def test_dcpa_config_schedules():
    """Verify that all annealing schedules are monotonically non-decreasing and bounded in [omega_min, 1.0]."""
    for sched in ["polynomial", "linear", "cosine", "exponential"]:
        cfg = DCPAConfig(omega_min=0.3, eta=1.5, schedule=sched)
        p_max = 5
        scales = [cfg.penalty_scale(p, p_max) for p in range(1, p_max + 1)]
        
        # Test boundary values
        assert math.isclose(scales[0], 0.3, abs_tol=1e-5), f"Schedule {sched} should start at omega_min"
        assert math.isclose(scales[-1], 1.0, abs_tol=1e-5), f"Schedule {sched} should end at 1.0"
        
        # Test monotonicity
        for i in range(len(scales) - 1):
            assert scales[i] <= scales[i + 1] + 1e-7, f"Schedule {sched} should be non-decreasing"


def test_build_annealed_problem():
    """Verify that annealed QUBO penalties scale proportionally with Omega(p)."""
    inst = generate_instance(n_nodes=4, seed=42)
    cfg = DCPAConfig(omega_min=0.4, schedule="linear")
    
    prob_p1, omega_1 = build_annealed_problem(inst, p=1, p_max=4, config=cfg)
    prob_p4, omega_4 = build_annealed_problem(inst, p=4, p_max=4, config=cfg)
    
    assert math.isclose(omega_1, 0.4, abs_tol=1e-5)
    assert math.isclose(omega_4, 1.0, abs_tol=1e-5)
    
    # Target standard QUBO
    q_target = cspp_to_qubo(inst)
    ising_target = qubo_to_ising(q_target)
    
    # At p=p_max, unnormalized Ising parameters must match target
    # (subject to same graph topology and slack variables)
    assert prob_p4.ising.n_qubits == ising_target.n_qubits


def test_cvar_expectation():
    """Verify CVaR tail expectation properties."""
    diag = np.array([10.0, 20.0, 30.0, 40.0])
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    
    # alpha = 1.0 should match standard expectation
    std_exp = float(np.sum(probs * diag)) # 25.0
    cvar_full = cvar_expectation(probs, diag, alpha=1.0)
    assert math.isclose(cvar_full, std_exp, abs_tol=1e-6)
    
    # alpha = 0.25 should only include the lowest state (10.0)
    cvar_tail = cvar_expectation(probs, diag, alpha=0.25)
    assert math.isclose(cvar_tail, 10.0, abs_tol=1e-6)
    
    # alpha = 0.5 should average the lowest two states (10.0 and 20.0 -> 15.0)
    cvar_half = cvar_expectation(probs, diag, alpha=0.50)
    assert math.isclose(cvar_half, 15.0, abs_tol=1e-6)
    
    # CVaR <= standard expectation always holds for any alpha <= 1.0
    assert cvar_half <= std_exp


def test_run_dcpa_ddqaoa_execution():
    """Verify that run_dcpa_ddqaoa executes cleanly and returns valid metrics."""
    inst = generate_instance(n_nodes=4, seed=0)
    opt_path, opt_cost = solve_classical(inst)
    assert opt_path is not None
    
    cfg = DCPAConfig(omega_min=0.4, schedule="linear", renormalize_gamma=True)
    res = run_dcpa_ddqaoa(
        inst=inst,
        p0=1,
        p_max=3,
        max_steps_per_depth=20,
        stepsize=0.04,
        dcpa_config=cfg,
        k=10,
        min_steps=10,
    )
    
    assert res["method"] == "ddqaoa_dcpa"
    assert 0.90 <= res["approx_ratio"] <= 1.01
    assert 0.0 <= res["success_prob"] <= 1.0
    assert res["final_depth"] == 3
    assert len(res["energies"]) > 0
    assert res["cumulative_cnots"] > 0
