"""Unit tests for CostLedger and resource accounting."""
import json
import pytest
from pathlib import Path

from ddqaoa.cost import CostLedger, estimate_hessian_cost


def test_cost_ledger_paper_cnots_synthetic_trajectory():
    # Suppose n_qubits=4, cnots_per_layer=10
    ledger = CostLedger(
        gradient_method="autodiff",
        cnots_per_layer=10,
        n_qubits=4,
        n_zz=5,
    )

    depth_trajectory = [1, 1, 2, 2, 3]
    for step, p in enumerate(depth_trajectory):
        ledger.record_step(step=step, depth=p, energy=-1.0 * step)

    # paper_cnots = sum(p_t * cnots_per_layer) = (1 + 1 + 2 + 2 + 3) * 10 = 90
    assert ledger.paper_cnots == 90
    assert len(ledger.history) == 5
    assert ledger.history[-1].paper_cnots_cum == 90


def test_cost_ledger_monotonicity_and_additivity():
    ledger = CostLedger(
        gradient_method="parameter_shift",
        shots=100,
        cnots_per_layer=6,
        n_qubits=4,
        n_zz=3,
        n_z=2,
    )

    prev_cnots = 0
    prev_execs = 0
    prev_shots = 0

    for step in range(5):
        ledger.record_step(step=step, depth=2, energy=-0.5 * step, grad_norm=0.1)
        assert ledger.paper_cnots >= prev_cnots
        assert ledger.circuit_executions >= prev_execs
        assert ledger.shots_total is not None and ledger.shots_total >= prev_shots
        prev_cnots = ledger.paper_cnots
        prev_execs = ledger.circuit_executions
        prev_shots = ledger.shots_total

    # Add hessian evaluations
    prev_hessian = ledger.hessian_cost
    ledger.record_hessian_evaluations(40)
    assert ledger.hessian_cost == prev_hessian + 40
    assert ledger.circuit_executions == prev_execs + 40


def test_gradient_methods_accounting():
    # 1. autodiff
    ad_ledger = CostLedger(gradient_method="autodiff", n_qubits=3, n_zz=2)
    ad_ledger.record_step(step=0, depth=1, energy=-1.0)
    assert ad_ledger.is_hardware_faithful is False
    assert ad_ledger.circuit_executions == 1  # 1 forward eval, 0 grad execs

    # 2. spsa
    spsa_ledger = CostLedger(gradient_method="spsa", n_qubits=3, n_zz=2)
    spsa_ledger.record_step(step=0, depth=1, energy=-1.0)
    assert spsa_ledger.is_hardware_faithful is True
    assert spsa_ledger.circuit_executions == 3  # 1 forward + 2 spsa shifts

    # 3. parameter_shift: occurrences per layer = (n_z + n_zz) + n_qubits = (1 + 2) + 3 = 6
    # For depth 2: total occurrences = 2 * 6 = 12. 2 shifts per occurrence -> 24 grad execs.
    ps_ledger = CostLedger(gradient_method="parameter_shift", n_qubits=3, n_zz=2, n_z=1)
    ps_ledger.record_step(step=0, depth=2, energy=-1.0)
    assert ps_ledger.is_hardware_faithful is True
    assert ps_ledger.circuit_executions == 1 + (2 * 12)  # 25


def test_estimate_hessian_cost():
    # n_qubits=4, n_zz=5, p=2, n_z=1
    # occurrences per layer = 1 + 5 + 4 = 10
    # block mode: 4 * (10)^2 = 400
    cost_block = estimate_hessian_cost(n_qubits=4, n_zz=5, p=2, mode="block", n_z=1)
    assert cost_block == 4 * (10**2)

    # full mode: occurrences = 2 * 10 = 20 -> 4 * (20)^2 = 1600
    cost_full = estimate_hessian_cost(n_qubits=4, n_zz=5, p=2, mode="full", n_z=1)
    assert cost_full == 4 * (20**2)


def test_jsonl_logging(tmp_path: Path):
    out_file = tmp_path / "trajectory.jsonl"
    ledger = CostLedger(gradient_method="autodiff", cnots_per_layer=4, n_qubits=2, n_zz=2)
    ledger.record_step(0, 1, -1.0, grad_norm=0.2, curvature_info={"min_eig": -0.5})
    ledger.record_step(1, 2, -1.5, grad_norm=0.05)
    ledger.to_jsonl(out_file)

    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text().strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["step"] == 0
    assert lines[0]["depth"] == 1
    assert lines[0]["curvature_info"]["min_eig"] == -0.5
    assert lines[1]["paper_cnots_cum"] == 4 + 8
