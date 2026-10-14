"""NISQ Noise modeling and robustness evaluation for QAOA circuits.

Supports:
- Density matrix mixed-state simulation via ``default.mixed``.
- Single-qubit depolarizing noise channels after single-qubit rotations (RX, RZ).
- Two-qubit depolarizing noise channels after two-qubit entangling gates (IsingZZ / CNOT).
- Finite measurement shot noise sampling (e.g. S = 1024) via ``qml.set_shots`` or sample counting.
- Noise robustness benchmarking comparing approximation ratio and success probability retention:
      Robustness Retention = Metric(Noisy) / Metric(Ideal)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pennylane as qml

from .circuits import QAOAProblem, qaoa_layer
from .metrics import approximation_ratio, success_probability


@dataclass
class NoiseConfig:
    """Configuration for physical noise channels on NISQ hardware."""
    p_single: float = 0.001     # 1-qubit gate depolarizing error probability
    p_two: float = 0.01         # 2-qubit gate depolarizing error probability
    shots: int | None = None     # None for exact density matrix expectation; int (e.g. 1024) for shot sampling
    seed: int | None = None      # Random seed for shot sampling


def qaoa_noisy_layer(gamma, beta, problem: QAOAProblem, noise: NoiseConfig) -> None:
    """QAOA layer with realistic gate-level depolarizing noise."""
    ising = problem.ising
    n = ising.n_qubits

    # 1. RZ and IsingZZ cost operations with noise
    for i, hi in ising.h.items():
        qml.RZ(2 * gamma * hi, wires=i)
        if noise.p_single > 0.0:
            qml.DepolarizingChannel(noise.p_single, wires=i)

    for (i, j), Jij in ising.J.items():
        qml.IsingZZ(2 * gamma * Jij, wires=[i, j])
        if noise.p_two > 0.0:
            # Apply depolarizing noise to both participating qubits in the 2-qubit gate
            qml.DepolarizingChannel(noise.p_two, wires=i)
            qml.DepolarizingChannel(noise.p_two, wires=j)

    # 2. RX mixer operations with noise
    for i in range(n):
        qml.RX(2 * beta, wires=i)
        if noise.p_single > 0.0:
            qml.DepolarizingChannel(noise.p_single, wires=i)


def make_noisy_qnodes(
    problem: QAOAProblem,
    noise: NoiseConfig | None = None,
) -> tuple[Callable, Callable]:
    """Create (probs_fn, cost_fn) simulating noisy quantum hardware on ``default.mixed``.

    Returns:
        probs_fn(gammas, betas): returns probability vector of shape (2^n,).
        cost_fn(gammas, betas): returns scalar expected energy.
    """
    noise = noise if noise is not None else NoiseConfig()
    n = problem.n_qubits
    dev = qml.device("default.mixed", wires=n)

    @qml.qnode(dev)
    def _noisy_circuit(gammas, betas):
        # Initial state: |+>^n
        for w in range(n):
            qml.Hadamard(wires=w)
            if noise.p_single > 0.0:
                qml.DepolarizingChannel(noise.p_single, wires=w)

        for g, b in zip(gammas, betas):
            qaoa_noisy_layer(g, b, problem, noise)

        return qml.probs(wires=range(n))

    diag = problem.diag

    def probs_fn(gammas, betas) -> np.ndarray:
        raw_probs = np.asarray(_noisy_circuit(gammas, betas))
        if noise.shots is not None and noise.shots > 0:
            rng = np.random.default_rng(noise.seed)
            # Sample from the multinomial distribution given the physical density matrix probabilities
            counts = rng.multinomial(noise.shots, raw_probs)
            return counts / float(noise.shots)
        return raw_probs

    def cost_fn(gammas, betas) -> float:
        p = probs_fn(gammas, betas)
        return float(np.dot(p, diag))

    return probs_fn, cost_fn


@dataclass
class NoiseEvaluationResult:
    """Evaluation of trained QAOA parameters under ideal vs noisy hardware conditions."""
    method: str
    depth: int
    ideal_energy: float
    ideal_approx_ratio: float
    ideal_success_prob: float
    noisy_energy: float
    noisy_approx_ratio: float
    noisy_success_prob: float
    approx_ratio_retention: float   # noisy_r / ideal_r
    success_prob_retention: float   # noisy_ps / ideal_ps
    noise_config: NoiseConfig

    def summary(self) -> dict:
        return {
            "method": self.method,
            "depth": self.depth,
            "ideal_approx_ratio": self.ideal_approx_ratio,
            "ideal_success_prob": self.ideal_success_prob,
            "noisy_approx_ratio": self.noisy_approx_ratio,
            "noisy_success_prob": self.noisy_success_prob,
            "approx_ratio_retention": self.approx_ratio_retention,
            "success_prob_retention": self.success_prob_retention,
            "noise_config": {
                "p_single": self.noise_config.p_single,
                "p_two": self.noise_config.p_two,
                "shots": self.noise_config.shots,
            }
        }


def evaluate_under_noise(
    gammas: np.ndarray,
    betas: np.ndarray,
    problem: QAOAProblem,
    noise: NoiseConfig,
    method_name: str = "qaoa",
    ideal_probs_fn: Callable | None = None,
) -> NoiseEvaluationResult:
    """Evaluate variational angles under both ideal statevector and noisy mixed-state models."""
    # Ideal evaluation
    if ideal_probs_fn is None:
        from .circuits import make_qnodes
        ideal_probs_fn, _ = make_qnodes(problem)

    ideal_p = np.asarray(ideal_probs_fn(gammas, betas))
    ideal_e = float(np.dot(ideal_p, problem.diag))
    ideal_r = approximation_ratio(ideal_e, problem)
    ideal_ps = success_probability(ideal_p, problem)

    # Noisy evaluation
    noisy_probs_fn, _ = make_noisy_qnodes(problem, noise)
    noisy_p = noisy_probs_fn(gammas, betas)
    noisy_e = float(np.dot(noisy_p, problem.diag))
    noisy_r = approximation_ratio(noisy_e, problem)
    noisy_ps = success_probability(noisy_p, problem)

    r_retention = noisy_r / ideal_r if ideal_r > 0 else 0.0
    ps_retention = noisy_ps / ideal_ps if ideal_ps > 0 else 0.0

    return NoiseEvaluationResult(
        method=method_name,
        depth=len(gammas),
        ideal_energy=ideal_e,
        ideal_approx_ratio=ideal_r,
        ideal_success_prob=ideal_ps,
        noisy_energy=noisy_e,
        noisy_approx_ratio=noisy_r,
        noisy_success_prob=noisy_ps,
        approx_ratio_retention=r_retention,
        success_prob_retention=ps_retention,
        noise_config=noise,
    )
