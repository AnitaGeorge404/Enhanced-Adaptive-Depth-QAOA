"""Noise-aware stopping rules, fidelity surrogate modeling, and crossover analysis (Feature 3).

Under quantum gate noise (e.g. depolarizing channel), deeper QAOA circuits suffer
from exponential fidelity decay F(p) = (1 - eps1)^n1(p) * (1 - eps2)^n2(p).
Standard plateau-based DDQAOA continues adding layers that degrade actual noisy performance.

This module provides:
1. FidelityModel: global-depolarizing surrogate E_noisy(p) ~ F(p) * E_ideal(p) + (1 - F(p)) * E_mm.
   Includes calibration fitting from shallow depths p in {1, 2} and prediction error evaluation.
2. NoiseAwareStopper:
   - Reactive rule: monitors noisy energy across depths, triggers early stopping after m
     consecutive non-improving depths, and ROLLS BACK parameters to the best noisy checkpoint.
   - Predictive rule: stops when the surrogate-predicted noisy gain of depth p+1 is <= 0.
3. run_noise_aware_ddqaoa: executes DDQAOA with noise-aware depth control under either
   ideal training with noisy evaluation or noisy training on default.mixed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pennylane as qml

from .circuits import QAOAProblem, cnots_per_layer, make_qnodes, qaoa_layer
from .convergence import ConvergenceMonitor
from .metrics import approximation_ratio, success_probability
from .transfer import AdaptiveTransfer, interp_transfer


def make_noisy_qnodes(
    problem: QAOAProblem,
    eps2: float,
    eps1: float | None = None,
    device: str = "default.mixed",
):
    """Create noisy PennyLane QNodes using depolarizing noise channels on default.mixed."""
    n = problem.n_qubits
    eps1_val = eps1 if eps1 is not None else eps2 / 10.0
    dev = qml.device(device, wires=n)

    @qml.qnode(dev)
    def noisy_probs_fn(gammas, betas):
        for w in range(n):
            qml.Hadamard(wires=w)
            if eps1_val > 0.0:
                qml.DepolarizingChannel(eps1_val, wires=w)

        for g, b in zip(gammas, betas):
            # Phase separator with noise
            for i, hi in problem.ising.h.items():
                qml.RZ(2 * g * hi, wires=i)
                if eps1_val > 0.0:
                    qml.DepolarizingChannel(eps1_val, wires=i)

            for (i, j), Jij in problem.ising.J.items():
                qml.IsingZZ(2 * g * Jij, wires=[i, j])
                if eps2 > 0.0:
                    qml.DepolarizingChannel(eps2, wires=i)
                    qml.DepolarizingChannel(eps2, wires=j)

            # Mixer with noise
            for i in range(problem.ising.n_qubits):
                qml.RX(2 * b, wires=i)
                if eps1_val > 0.0:
                    qml.DepolarizingChannel(eps1_val, wires=i)

        return qml.probs(wires=range(n))

    diag = problem.diag

    def noisy_cost_fn(gammas, betas):
        return qml.math.dot(noisy_probs_fn(gammas, betas), diag)

    return noisy_probs_fn, noisy_cost_fn


@dataclass
class FidelityModel:
    """Surrogate model for noisy expectation values under global depolarizing noise."""

    n_qubits: int
    n_zz: int
    n_z: int = 0
    eps2: float = 0.0
    eps1: float | None = None
    e_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.eps1 is None:
            self.eps1 = self.eps2 / 10.0

    def n1(self, p: int) -> int:
        """Total 1-qubit gate occurrences through depth p."""
        return p * (self.n_qubits + self.n_z)

    def n2(self, p: int) -> int:
        """Total 2-qubit gate occurrences through depth p."""
        return p * self.n_zz

    def fidelity(self, p: int) -> float:
        """Circuit fidelity F(p) = (1 - eps1)^n1(p) * (1 - eps2)^n2(p)."""
        if self.eps2 <= 0.0 and (self.eps1 is None or self.eps1 <= 0.0):
            return 1.0
        f1 = max(0.0, 1.0 - float(self.eps1 or 0.0)) ** self.n1(p)
        f2 = max(0.0, 1.0 - float(self.eps2)) ** self.n2(p)
        return float(f1 * f2)

    def predict_noisy_energy(self, p: int, e_ideal: float) -> float:
        """Predict noisy energy: E_noisy(p) ~ F(p) * E_ideal(p) + (1 - F(p)) * E_mm."""
        f = self.fidelity(p)
        return float(f * e_ideal + (1.0 - f) * self.e_mm)

    def fit(self, calibration_points: list[tuple[int, float, float]]) -> float:
        """Fit effective eps2 from shallow depth observations (p, E_ideal, E_noisy).

        Returns fitted effective eps2.
        """
        if not calibration_points:
            return self.eps2

        log_ratios = []
        gate_counts = []
        for p, e_ideal, e_noisy in calibration_points:
            denom = e_ideal - self.e_mm
            numer = e_noisy - self.e_mm
            if abs(denom) > 1e-6 and numer / denom > 0.0:
                f_obs = min(1.0, max(1e-6, numer / denom))
                log_ratios.append(math.log(f_obs))
                # Effective 2-qubit gate equivalent: n2(p) + 0.1 * n1(p)
                gate_counts.append(self.n2(p) + 0.1 * self.n1(p))

        if log_ratios and gate_counts and sum(gate_counts) > 0:
            slope = sum(log_ratios) / sum(gate_counts)
            eff_eps = 1.0 - math.exp(slope)
            self.eps2 = max(0.0, float(eff_eps))
            self.eps1 = self.eps2 / 10.0

        return self.eps2

    def prediction_mae(self, test_points: list[tuple[int, float, float]]) -> float:
        """Mean absolute error (MAE) of surrogate predictions vs actual noisy evaluations."""
        if not test_points:
            return 0.0
        errors = [
            abs(self.predict_noisy_energy(p, e_id) - e_ny)
            for p, e_id, e_ny in test_points
        ]
        return float(np.mean(errors))


@dataclass
class NoiseStopDecision:
    """Decision output from NoiseAwareStopper."""
    stop: bool
    reason: str
    rollback: bool = False
    best_depth: int = 1
    best_noisy_energy: float = float("inf")
    best_gammas: np.ndarray | None = None
    best_betas: np.ndarray | None = None
    predicted_next_gain: float = 0.0


class NoiseAwareStopper:
    """Controls depth progression under noise via reactive and predictive stopping rules."""

    def __init__(
        self,
        rule: Literal["reactive", "predictive", "both"] = "reactive",
        m: int = 2,
        tau: float = 1e-3,
        fidelity_model: FidelityModel | None = None,
        min_depth: int = 1,
    ) -> None:
        self.rule = rule
        self.m = int(m)
        self.tau = float(tau)
        self.fidelity_model = fidelity_model
        self.min_depth = int(min_depth)

        self.best_noisy_energy = float("inf")
        self.best_depth = 1
        self.best_gammas: np.ndarray | None = None
        self.best_betas: np.ndarray | None = None
        self.stagnant_count = 0
        self.history: list[dict[str, Any]] = []

    def evaluate_depth_transition(
        self,
        current_depth: int,
        gammas: np.ndarray,
        betas: np.ndarray,
        noisy_energy: float,
        ideal_energy: float | None = None,
        estimated_ideal_gain: float | None = None,
    ) -> NoiseStopDecision:
        """Evaluate noisy performance at depth completion and decide whether to stop/rollback."""
        noisy_e = float(noisy_energy)
        improved = noisy_e < (self.best_noisy_energy - self.tau)

        if improved or self.best_gammas is None:
            self.best_noisy_energy = noisy_e
            self.best_depth = current_depth
            self.best_gammas = np.array(gammas, copy=True)
            self.best_betas = np.array(betas, copy=True)
            self.stagnant_count = 0
        else:
            self.stagnant_count += 1

        self.history.append({
            "depth": current_depth,
            "noisy_energy": noisy_e,
            "best_noisy_energy": self.best_noisy_energy,
            "stagnant_count": self.stagnant_count,
        })

        if current_depth < self.min_depth:
            return NoiseStopDecision(
                stop=False,
                reason="Below min_depth",
                best_depth=self.best_depth,
                best_noisy_energy=self.best_noisy_energy,
            )

        # 1. Reactive stopping rule: m consecutive non-improving depths
        if self.rule in ("reactive", "both"):
            if self.stagnant_count >= self.m:
                return NoiseStopDecision(
                    stop=True,
                    reason=(
                        f"Reactive noise stop: noisy energy failed to improve by > {self.tau} "
                        f"for {self.m} consecutive depths (best depth was p={self.best_depth})"
                    ),
                    rollback=True,
                    best_depth=self.best_depth,
                    best_noisy_energy=self.best_noisy_energy,
                    best_gammas=self.best_gammas,
                    best_betas=self.best_betas,
                )

        # 2. Predictive stopping rule: predicted noisy gain of next depth <= 0
        if self.rule in ("predictive", "both") and self.fidelity_model is not None:
            e_id = ideal_energy if ideal_energy is not None else noisy_e
            gain_id = estimated_ideal_gain if estimated_ideal_gain is not None else 1e-2
            e_id_next = e_id - abs(gain_id)

            pred_curr = self.fidelity_model.predict_noisy_energy(current_depth, e_id)
            pred_next = self.fidelity_model.predict_noisy_energy(current_depth + 1, e_id_next)
            # Energy minimization: gain is pred_curr - pred_next
            pred_gain = pred_curr - pred_next

            if pred_gain <= 0.0:
                return NoiseStopDecision(
                    stop=True,
                    reason=(
                        f"Predictive noise stop: surrogate predicts non-positive noisy gain "
                        f"({pred_gain:+.4e}) for depth {current_depth + 1}"
                    ),
                    rollback=(self.best_depth < current_depth),
                    best_depth=self.best_depth,
                    best_noisy_energy=self.best_noisy_energy,
                    best_gammas=self.best_gammas,
                    best_betas=self.best_betas,
                    predicted_next_gain=pred_gain,
                )

        return NoiseStopDecision(
            stop=False,
            reason="Noise-aware criteria satisfied to advance depth",
            best_depth=self.best_depth,
            best_noisy_energy=self.best_noisy_energy,
            best_gammas=self.best_gammas,
            best_betas=self.best_betas,
        )


def run_noise_aware_ddqaoa(
    problem: QAOAProblem,
    eps2: float,
    eps1: float | None = None,
    rule: Literal["reactive", "predictive", "both"] = "reactive",
    objective_mode: Literal["ideal", "noisy"] = "ideal",
    p0: int = 1,
    p_max: int = 6,
    max_steps: int = 400,
    stepsize: float = 0.03,
    k: int = 25,
    m: int = 2,
    tau: float = 1e-3,
    transfer_strategy: str = "adaptive",
    device: str = "default.qubit",
    noisy_device: str = "default.mixed",
    noisy_backend: Literal["auto", "mixed", "surrogate"] = "auto",
    seed: int | None = None,
) -> Any:
    """Execute DDQAOA with noise-aware stopping and automatic rollback.

    objective_mode:
        - "ideal": optimization proceeds on ideal statevector; noisy energy is evaluated
          at each depth boundary to drive the stopping and rollback decision.
        - "noisy": optimization proceeds directly on noisy default.mixed device.
    noisy_backend:
        - "auto": uses exact mixed-state density matrix for small systems (n <= 7) or
          when training under noise; uses calibrated FidelityModel surrogate for n > 7.
        - "mixed": forces exact default.mixed simulation.
        - "surrogate": uses global-depolarizing FidelityModel surrogate.
    """
    e_mm = float(np.mean(problem.diag))
    fid_model = FidelityModel(
        n_qubits=problem.n_qubits,
        n_zz=len(problem.ising.J),
        n_z=len(problem.ising.h),
        eps2=eps2,
        eps1=eps1,
        e_mm=e_mm,
    )

    stopper = NoiseAwareStopper(
        rule=rule,
        m=m,
        tau=tau,
        fidelity_model=fid_model,
        min_depth=1,
    )

    # Set up ideal and noisy evaluation functions
    ideal_probs_fn, ideal_cost_fn = make_qnodes(problem, device=device)

    # Backend selection
    if noisy_backend == "mixed":
        use_exact_mixed = (eps2 > 0.0)
    elif noisy_backend == "surrogate":
        use_exact_mixed = False
    else:  # "auto"
        use_exact_mixed = (problem.n_qubits <= 7 or objective_mode == "noisy") and (eps2 > 0.0)

    if use_exact_mixed:
        noisy_probs_fn, noisy_cost_fn = make_noisy_qnodes(problem, eps2=eps2, eps1=eps1, device=noisy_device)
    else:
        noisy_probs_fn = ideal_probs_fn
        def noisy_cost_fn(g, b):
            e_id = float(ideal_cost_fn(g, b))
            return fid_model.predict_noisy_energy(len(g), e_id)

    train_cost_fn = noisy_cost_fn if objective_mode == "noisy" else ideal_cost_fn
    transfer = AdaptiveTransfer(strategy=transfer_strategy)
    monitor = ConvergenceMonitor(k=k)

    gammas = np.array([0.1] * p0, dtype=float)
    betas = np.array([0.1] * p0, dtype=float)
    from pennylane import numpy as pnp
    opt = qml.AdamOptimizer(stepsize)

    energies = []
    depths = []
    switches = []
    stopping_reason = f"Reached p_max={p_max}"
    final_gammas = gammas
    final_betas = betas

    for step in range(max_steps):
        g_var = pnp.array(gammas, requires_grad=True)
        b_var = pnp.array(betas, requires_grad=True)
        (params, energy) = opt.step_and_cost(train_cost_fn, g_var, b_var)
        energy_val = float(energy)
        energies.append(energy_val)
        depths.append(len(gammas))
        gammas, betas = np.asarray(params[0]), np.asarray(params[1])

        if monitor.update(energy_val):
            # Depth converged: evaluate noisy energy
            curr_noisy_e = float(noisy_cost_fn(gammas, betas))
            curr_ideal_e = float(ideal_cost_fn(gammas, betas))

            decision = stopper.evaluate_depth_transition(
                current_depth=len(gammas),
                gammas=gammas,
                betas=betas,
                noisy_energy=curr_noisy_e,
                ideal_energy=curr_ideal_e,
            )

            if decision.stop:
                stopping_reason = decision.reason
                if decision.rollback and decision.best_gammas is not None:
                    final_gammas = decision.best_gammas
                    final_betas = decision.best_betas
                else:
                    final_gammas = gammas
                    final_betas = betas
                break

            if len(gammas) >= p_max:
                stopping_reason = f"Reached p_max={p_max}"
                final_gammas = gammas
                final_betas = betas
                break

            g_new, b_new = transfer(gammas, betas)
            gammas = np.asarray(g_new, dtype=float)
            betas = np.asarray(b_new, dtype=float)
            opt = qml.AdamOptimizer(stepsize)
            monitor.reset()
            switches.append(step + 1)

    if stopping_reason == f"Reached p_max={p_max}":
        final_gammas = gammas
        final_betas = betas

    # Compute final metrics at chosen / rolled back parameters
    final_noisy_energy = float(noisy_cost_fn(final_gammas, final_betas))
    final_ideal_energy = float(ideal_cost_fn(final_gammas, final_betas))
    noisy_probs = np.asarray(noisy_probs_fn(final_gammas, final_betas))
    ideal_probs = np.asarray(ideal_probs_fn(final_gammas, final_betas))

    r_noisy = approximation_ratio(final_noisy_energy, problem)
    r_ideal = approximation_ratio(final_ideal_energy, problem)
    retention = (r_noisy / r_ideal) if r_ideal > 0 else 0.0

    from .ddqaoa import RunResult
    return RunResult(
        method=f"ddqaoa_noise_aware_{rule}",
        gammas=np.asarray(final_gammas),
        betas=np.asarray(final_betas),
        best_energy=final_noisy_energy,
        approx_ratio=r_noisy,
        success_prob=success_probability(noisy_probs, problem),
        final_depth=len(final_gammas),
        cumulative_cnots=int(sum(depths) * cnots_per_layer(problem.ising)),
        energies=energies,
        depths=depths,
        depth_switch_steps=switches,
        stopping_reason=stopping_reason,
        metadata={
            "eps2": eps2,
            "eps1": eps1,
            "rule": rule,
            "objective_mode": objective_mode,
            "noisy_energy": final_noisy_energy,
            "ideal_energy": final_ideal_energy,
            "noisy_approx_ratio": r_noisy,
            "ideal_approx_ratio": r_ideal,
            "retention": retention,
            "best_depth": stopper.best_depth,
            "rollback_applied": (len(final_gammas) < len(gammas)),
        },
    )
