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
    p_readout_01: float = 0.0    # Readout bit-flip probability 0 -> 1
    p_readout_10: float = 0.0    # Readout bit-flip probability 1 -> 0


def apply_readout_noise(
    probs: np.ndarray,
    p_01: float,
    p_10: float,
    n_qubits: int,
) -> np.ndarray:
    """Apply tensor-factored bit-flip readout noise channels: P_noisy = M * P_true."""
    if p_01 <= 0.0 and p_10 <= 0.0:
        return probs

    res = np.array(probs, dtype=float, copy=True)
    for w in range(n_qubits):
        lead = 1 << w
        trail = 1 << (n_qubits - 1 - w)
        res_r = res.reshape(lead, 2, trail)
        p0 = res_r[:, 0, :].copy()
        p1 = res_r[:, 1, :].copy()
        # Assignment:
        # P(meas 0) = (1 - p_01) * p0 + p_10 * p1
        # P(meas 1) = p_01 * p0 + (1 - p_10) * p1
        res_r[:, 0, :] = (1.0 - p_01) * p0 + p_10 * p1
        res_r[:, 1, :] = p_01 * p0 + (1.0 - p_10) * p1
    return res


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto probability simplex {p >= 0, sum(p) = 1} (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def apply_readout_mitigation(
    noisy_probs: np.ndarray,
    p_01: float,
    p_10: float,
    n_qubits: int,
    project: bool = True,
) -> np.ndarray:
    """Invert the readout confusion matrix M^{-1} = (x)_i A_i^{-1} to recover true solution frequencies.

    Single-qubit assignment matrix:
        A_i = [[1 - p_01, p_10],
               [p_01,     1 - p_10]]
        det = 1 - p_01 - p_10
        A_i^{-1} = (1/det) * [[1 - p_10, -p_10],
                              [-p_01,     1 - p_01]]

    Simplex projection ensures physical non-negative probabilities summing to 1.
    """
    if p_01 <= 0.0 and p_10 <= 0.0:
        return noisy_probs

    det = 1.0 - p_01 - p_10
    if abs(det) < 1e-6:
        raise ValueError(f"Readout noise too high for inversion: p_01={p_01}, p_10={p_10}")

    inv_a00 = (1.0 - p_10) / det
    inv_a01 = -p_10 / det
    inv_a10 = -p_01 / det
    inv_a11 = (1.0 - p_01) / det

    res = np.array(noisy_probs, dtype=float, copy=True)
    for w in range(n_qubits):
        lead = 1 << w
        trail = 1 << (n_qubits - 1 - w)
        res_r = res.reshape(lead, 2, trail)
        q0 = res_r[:, 0, :].copy()
        q1 = res_r[:, 1, :].copy()
        res_r[:, 0, :] = inv_a00 * q0 + inv_a01 * q1
        res_r[:, 1, :] = inv_a10 * q0 + inv_a11 * q1

    if project:
        return project_to_simplex(res)
    return res


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
    eps2: float | None = None,
    eps1: float | None = None,
    device: str = "default.mixed",
    warm_start_angles: np.ndarray | None = None,
) -> tuple[Callable, Callable]:
    """Create (probs_fn, cost_fn) simulating noisy quantum hardware on ``default.mixed``.

    Supports passing either a NoiseConfig instance or eps2, eps1 directly, as well as
    warm-start Ry state preparation angles.
    """
    if eps2 is not None:
        p_single = eps1 if eps1 is not None else eps2 / 10.0
        noise = NoiseConfig(p_single=p_single, p_two=eps2)
    elif noise is None:
        noise = NoiseConfig()
    n = problem.n_qubits
    dev = qml.device(device, wires=n)

    @qml.qnode(dev)
    def _noisy_circuit(gammas, betas):
        # Initial state: |+>^n or warm-start Ry(theta)
        for w in range(n):
            if warm_start_angles is not None:
                qml.RY(warm_start_angles[w], wires=w)
            else:
                qml.Hadamard(wires=w)
            if noise.p_single > 0.0:
                qml.DepolarizingChannel(noise.p_single, wires=w)

        for g, b in zip(gammas, betas):
            qaoa_noisy_layer(g, b, problem, noise)

        return qml.probs(wires=range(n))

    diag = problem.diag

    def probs_fn(gammas, betas) -> np.ndarray:
        raw_probs = np.asarray(_noisy_circuit(gammas, betas))
        # Apply physical readout error channel
        if noise.p_readout_01 > 0.0 or noise.p_readout_10 > 0.0:
            raw_probs = apply_readout_noise(raw_probs, noise.p_readout_01, noise.p_readout_10, n)
        if noise.shots is not None and noise.shots > 0:
            rng = np.random.default_rng(noise.seed)
            # Sample from the multinomial distribution given the physical probabilities
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
    mitigated_energy: float | None = None
    mitigated_approx_ratio: float | None = None
    mitigated_success_prob: float | None = None
    rem_recovery_ratio: float | None = None
    rem_recovery_succ_p: float | None = None

    @property
    def unmitigated_success_prob(self) -> float:
        return self.noisy_success_prob

    @property
    def unmitigated_approx_ratio(self) -> float:
        return self.noisy_approx_ratio

    def summary(self) -> dict:
        d = {
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
                "p_readout_01": self.noise_config.p_readout_01,
                "p_readout_10": self.noise_config.p_readout_10,
            }
        }
        if self.mitigated_approx_ratio is not None:
            d["mitigated_approx_ratio"] = self.mitigated_approx_ratio
            d["mitigated_success_prob"] = self.mitigated_success_prob
            d["rem_recovery_ratio"] = self.rem_recovery_ratio
            d["rem_recovery_succ_p"] = self.rem_recovery_succ_p
        return d


def evaluate_under_noise(
    gammas: np.ndarray,
    betas: np.ndarray,
    problem: QAOAProblem,
    noise: NoiseConfig,
    method_name: str = "qaoa",
    ideal_probs_fn: Callable | None = None,
    apply_rem: bool = True,
    warm_start_angles: np.ndarray | None = None,
) -> NoiseEvaluationResult:
    """Evaluate variational angles under both ideal statevector and noisy mixed-state models."""
    # Ideal evaluation
    if ideal_probs_fn is None:
        from .circuits import make_qnodes
        ideal_probs_fn, _ = make_qnodes(problem, warm_start_angles=warm_start_angles)

    ideal_p = np.asarray(ideal_probs_fn(gammas, betas))
    ideal_e = float(np.dot(ideal_p, problem.diag))
    ideal_r = approximation_ratio(ideal_e, problem)
    ideal_ps = success_probability(ideal_p, problem)

    # Noisy evaluation
    noisy_probs_fn, _ = make_noisy_qnodes(problem, noise, warm_start_angles=warm_start_angles)
    noisy_p = noisy_probs_fn(gammas, betas)
    noisy_e = float(np.dot(noisy_p, problem.diag))
    noisy_r = approximation_ratio(noisy_e, problem)
    noisy_ps = success_probability(noisy_p, problem)

    r_retention = noisy_r / ideal_r if ideal_r > 0 else 0.0
    ps_retention = noisy_ps / ideal_ps if ideal_ps > 0 else 0.0

    mit_e, mit_r, mit_ps = None, None, None
    rem_rec_r, rem_rec_ps = None, None

    # Readout Error Mitigation
    if apply_rem and (noise.p_readout_01 > 0.0 or noise.p_readout_10 > 0.0):
        mit_p = apply_readout_mitigation(
            noisy_p,
            noise.p_readout_01,
            noise.p_readout_10,
            problem.n_qubits,
            project=True,
        )
        mit_e = float(np.dot(mit_p, problem.diag))
        mit_r = approximation_ratio(mit_e, problem)
        mit_ps = success_probability(mit_p, problem)

        denom_r = ideal_r - noisy_r
        rem_rec_r = (mit_r - noisy_r) / denom_r if abs(denom_r) > 1e-12 else 1.0

        denom_ps = ideal_ps - noisy_ps
        rem_rec_ps = (mit_ps - noisy_ps) / denom_ps if abs(denom_ps) > 1e-12 else 1.0

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
        mitigated_energy=mit_e,
        mitigated_approx_ratio=mit_r,
        mitigated_success_prob=mit_ps,
        rem_recovery_ratio=rem_rec_r,
        rem_recovery_succ_p=rem_rec_ps,
    )
