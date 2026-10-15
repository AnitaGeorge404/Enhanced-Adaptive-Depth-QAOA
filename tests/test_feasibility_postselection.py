"""Unit tests for feasibility post-selection, decoding, and end-to-end costs (Feature 4).

Tests:
1. Feasibility checker vs exact enumerator:
   - every enumerated feasible path passes "path".
   - every violated-constraint construction fails.
2. Levels check: structural vs path vs qubo.
3. Post-selected energy matches standard energy when all states are feasible.
4. Fallback to unmitigated energy when yield < y_min.
5. TTS_99 formula and edge cases (P_optimal=0 -> inf, P_optimal=1 -> 1 batch).
6. Postselected objective optimization in DDQAOA.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from ddqaoa import (
    build_feasible_mask,
    build_problem,
    cspp_to_qubo,
    decode_path,
    generate_instance,
    is_feasible,
    postselected_energy,
    reduce_instance,
    run_ddqaoa,
    time_to_solution_99,
)
from ddqaoa.cspp import simple_paths
from ddqaoa.circuits import index_to_bits, make_qnodes


def test_feasibility_checker_vs_exact_enumerator():
    """Property 1: Every enumerated feasible path passes 'path' check; invalid ones fail."""
    inst = generate_instance(n_nodes=4, seed=123, prune=True)
    q = cspp_to_qubo(inst)
    all_simple = simple_paths(inst)

    for p in all_simple:
        res = inst.path_resource(p)
        # Construct bitstring corresponding to path p
        p_edges = set(zip(p, p[1:]))
        bits = [1 if e in p_edges else 0 for e in inst.edges]
        # Append dummy slack bits (e.g. 0s)
        bits += [0] * q.n_slack

        feas = is_feasible(bits, inst, level="path")
        if res <= inst.r_limit:
            assert feas is True, f"Valid path {p} failed feasibility check"
            assert decode_path(bits, inst) == p
        else:
            assert feas is False, f"Infeasible path {p} (res {res} > {inst.r_limit}) passed check"


def test_feasibility_violated_constraint_constructions():
    """Property 1b: Broken paths, disconnected loops, and dead-ends fail feasibility check."""
    inst = generate_instance(n_nodes=5, seed=456, prune=True)
    n_vars = len(inst.edges) + 2

    # 1. Empty graph (no edges selected)
    empty_bits = [0] * n_vars
    assert not is_feasible(empty_bits, inst, level="path")
    assert decode_path(empty_bits, inst) is None

    # 2. Disconnected cycle without source-target connection
    # Pick intermediate edges if available
    cycle_edges = [(1, 2), (2, 3), (3, 1)]
    bits = [1 if e in cycle_edges else 0 for e in inst.edges] + [0, 0]
    assert not is_feasible(bits, inst, level="path")
    assert decode_path(bits, inst) is None

    # 3. Path plus disconnected cycle
    valid_paths = [p for p in simple_paths(inst) if inst.path_resource(p) <= inst.r_limit]
    if valid_paths:
        p0 = valid_paths[0]
        p_edges = set(zip(p0, p0[1:]))
        # Add a disconnected edge not in path
        other_edges = [e for e in inst.edges if e not in p_edges]
        if other_edges:
            corrupted = set(p_edges) | {other_edges[0]}
            c_bits = [1 if e in corrupted else 0 for e in inst.edges] + [0, 0]
            assert not is_feasible(c_bits, inst, level="path")


def test_feasibility_levels_differentiation():
    """Property 2: Test differentiation between structural, path, and qubo levels."""
    inst = generate_instance(n_nodes=4, seed=789, prune=True)
    q = cspp_to_qubo(inst)

    paths = [p for p in simple_paths(inst) if inst.path_resource(p) <= inst.r_limit]
    assert len(paths) > 0
    p = paths[0]
    p_edges = set(zip(p, p[1:]))
    r_used = inst.path_resource(p)
    req_slack = inst.r_limit - r_used

    # Build bitstring with exact slack bits
    bits_edge = [1 if e in p_edges else 0 for e in inst.edges]
    correct_slack = [(req_slack >> k) & 1 for k in range(q.n_slack)]
    bitstring_exact = bits_edge + correct_slack

    # Must pass all three levels
    assert is_feasible(bitstring_exact, inst, level="structural", qubo=q) is True
    assert is_feasible(bitstring_exact, inst, level="path", qubo=q) is True
    assert is_feasible(bitstring_exact, inst, level="qubo", qubo=q) is True

    # Mutate slack bits to wrong value
    wrong_slack = [(1 - b) for b in correct_slack]
    bitstring_wrong_slack = bits_edge + wrong_slack

    # Must pass structural and path, but FAIL qubo level
    assert is_feasible(bitstring_wrong_slack, inst, level="structural", qubo=q) is True
    assert is_feasible(bitstring_wrong_slack, inst, level="path", qubo=q) is True
    if req_slack != sum(b * (2**k) for k, b in enumerate(wrong_slack)):
        assert is_feasible(bitstring_wrong_slack, inst, level="qubo", qubo=q) is False


def test_postselected_energy_equals_plain_energy_when_all_feasible():
    """Property 3: E_ps equals standard energy when all basis states are feasible."""
    probs = np.array([0.1, 0.4, 0.3, 0.2])
    diag = np.array([1.0, 2.0, 3.0, 4.0])
    all_feas_mask = np.array([True, True, True, True])

    e_ps, y = postselected_energy(probs, diag, all_feas_mask)
    e_std = float(np.dot(probs, diag))

    assert math.isclose(y, 1.0)
    assert math.isclose(e_ps, e_std, rel_tol=1e-12)


def test_postselected_energy_fallback_when_yield_low():
    """Property 4: Post-selected energy falls back to unmitigated energy when yield < y_min."""
    probs = np.array([0.99999, 0.00001])
    diag = np.array([10.0, 2.0])
    mask = np.array([False, True])  # yield is only 1e-5

    e_ps, y = postselected_energy(probs, diag, mask, y_min=1e-4)
    e_unmitigated = float(np.dot(probs, diag))

    assert y < 1e-4
    assert math.isclose(e_ps, e_unmitigated, rel_tol=1e-12)


def test_time_to_solution_99_edge_cases():
    """Property 5: TTS_99 edge cases: P_opt=0 -> inf, P_opt=1 -> 1 batch."""
    # Zero probability -> infinite shots
    assert time_to_solution_99(0.0) == float("inf")
    assert time_to_solution_99(-0.5) == float("inf")

    # Certain probability -> exactly 1 batch of shots
    assert time_to_solution_99(1.0, shots_per_batch=1024) == 1024.0

    # P_opt = 0.5: ln(0.01)/ln(0.5) = -4.60517 / -0.69315 = 6.6438 -> ceil = 7 batches
    tts = time_to_solution_99(0.5, shots_per_batch=1024)
    assert tts == 7 * 1024.0


def test_postselected_objective_in_ddqaoa():
    """Property 6: DDQAOA runs with objective='postselected' and records postselected metrics."""
    inst = generate_instance(n_nodes=4, seed=555, prune=True)
    q, problem = build_problem(inst)
    mask = build_feasible_mask(problem, inst, level="path", qubo=q)

    assert np.sum(mask) > 0, "At least one feasible basis state must exist"

    res = run_ddqaoa(
        problem,
        p0=1,
        p_max=2,
        max_steps=40,
        stepsize=0.03,
        k=15,
        objective="postselected",
        feasible_mask=mask,
    )

    assert "p_feasible" in res.metadata
    assert "postselected_energy" in res.metadata
    assert "postselected_ratio" in res.metadata
    assert "tts_99" in res.metadata
    assert 0.0 <= res.metadata["p_feasible"] <= 1.0
