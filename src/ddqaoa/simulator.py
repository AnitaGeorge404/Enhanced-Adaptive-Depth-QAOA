"""Analytical Matrix-Free Simulator for QAOA (Feature 10).

Exploits the strictly diagonal structure of the Ising cost Hamiltonian H_C in the
computational basis:
    U_C(gamma) |psi> = exp(-i gamma H_C) |psi> = exp(-i gamma C(z)) psi(z)
and the factored single-qubit rotations of the transverse-field mixer H_M:
    U_M(beta) |psi> = (x)_{j=0}^{n-1} exp(-i beta X_j) |psi>

Key Features:
1. Matrix-Free: Never allocates 2^n x 2^n dense operator matrices.
2. Exact Adjoint Gradient: Evaluates dE/d(gamma) and dE/d(beta) for all 2p parameters
   in a single backward sweep with O(p * n * 2^n) time and O(p * 2^n) memory.
3. Autograd & PennyLane Integration: Exposes primitive VJP hooks so ControlledAdamOptimizer
   and qml.grad work seamlessly without changes.
4. Warm-Starting Ready: Supports arbitrary initial product states |psi_0> = (x)_j (cos(theta_j/2)|0> + sin(theta_j/2)|1>).
"""
from __future__ import annotations

import math
from typing import Callable
import numpy as np
import autograd.extend as extend
import pennylane as qml
from pennylane import numpy as pnp

from .circuits import QAOAProblem


def apply_cost_layer(psi: np.ndarray, gamma: float, diag: np.ndarray) -> np.ndarray:
    """Element-wise diagonal phase multiplication: exp(-i * gamma * diag) * psi."""
    phase = np.exp(-1j * float(gamma) * diag)
    return phase * psi


def apply_mixer_layer(psi: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    """Apply factored single-qubit RX(2*beta) rotations across all n qubits.

    RX(2*beta) = exp(-i * beta * X) = cos(beta) * I - i * sin(beta) * X.
    Contraction is performed in-place along axes without full matrix construction.
    """
    c = math.cos(float(beta))
    s = -1j * math.sin(float(beta))
    res = psi.copy()

    for w in range(n_qubits):
        lead = 1 << w
        trail = 1 << (n_qubits - 1 - w)
        res_r = res.reshape(lead, 2, trail)
        v0 = res_r[:, 0, :].copy()
        v1 = res_r[:, 1, :].copy()
        res_r[:, 0, :] = c * v0 + s * v1
        res_r[:, 1, :] = s * v0 + c * v1

    return res


def compute_H_M_psi(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute H_M |psi> = sum_{w=0}^{n-1} X_w |psi>."""
    total = np.zeros_like(psi)
    for w in range(n_qubits):
        lead = 1 << w
        trail = 1 << (n_qubits - 1 - w)
        psi_r = psi.reshape(lead, 2, trail)
        flipped = np.empty_like(psi_r)
        flipped[:, 0, :] = psi_r[:, 1, :]
        flipped[:, 1, :] = psi_r[:, 0, :]
        total += flipped.reshape(-1)
    return total


class AnalyticalSimulator:
    """High-performance matrix-free statevector simulator for QAOA."""

    def __init__(
        self,
        problem: QAOAProblem,
        warm_start_angles: np.ndarray | None = None,
    ) -> None:
        self.problem = problem
        self.n_qubits = problem.n_qubits
        self.diag = np.asarray(problem.diag, dtype=float)
        self.warm_start_angles = (
            np.asarray(warm_start_angles, dtype=float)
            if warm_start_angles is not None
            else None
        )
        self._psi0 = self._prepare_initial_state()

    def _prepare_initial_state(self) -> np.ndarray:
        n = self.n_qubits
        if self.warm_start_angles is None:
            # Standard uniform superposition: |+>^{\otimes n}
            return np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=np.complex128)

        # Warm-started separable state: (x)_j (cos(theta_j/2)|0> + sin(theta_j/2)|1>)
        thetas = self.warm_start_angles
        psi = np.array([1.0], dtype=np.complex128)
        for w in range(n):
            th = thetas[w]
            q_state = np.array([math.cos(th / 2.0), math.sin(th / 2.0)], dtype=np.complex128)
            psi = np.kron(psi, q_state)
        return psi

    def forward_trajectory(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Simulate forward trajectory, storing intermediate states for adjoint differentiation.

        Returns:
            final_psi: state after all p layers.
            trajectory: list of (psi_c, psi_m) for each layer l = 0 ... p-1.
        """
        p = len(gammas)
        psi = self._psi0.copy()
        trajectory = []

        for l in range(p):
            psi_c = apply_cost_layer(psi, gammas[l], self.diag)
            psi_m = apply_mixer_layer(psi_c, betas[l], self.n_qubits)
            trajectory.append((psi_c, psi_m))
            psi = psi_m

        return psi, trajectory

    def simulate(self, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """Compute final statevector without saving intermediate trajectory."""
        p = len(gammas)
        psi = self._psi0.copy()
        for l in range(p):
            psi = apply_cost_layer(psi, gammas[l], self.diag)
            psi = apply_mixer_layer(psi, betas[l], self.n_qubits)
        return psi

    def probabilities(self, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """Exact probability distribution P(z) = |psi(z)|^2 of length 2^n."""
        psi = self.simulate(gammas, betas)
        probs = np.abs(psi) ** 2
        return np.real(probs)

    def energy(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
        feasible_mask: np.ndarray | None = None,
        objective: str = "energy",
        y_min: float = 1e-4,
    ) -> float:
        """Scalar cost expectation value."""
        probs = self.probabilities(gammas, betas)
        if objective == "postselected" and feasible_mask is not None:
            f_mask = np.asarray(feasible_mask, dtype=float)
            p_feas = probs * f_mask
            y = float(np.sum(p_feas))
            if y < y_min:
                return float(np.dot(probs, self.diag))
            return float(np.dot(p_feas, self.diag) / (y + 1e-15))
        return float(np.dot(probs, self.diag))

    def adjoint_gradient(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
        feasible_mask: np.ndarray | None = None,
        objective: str = "energy",
        y_min: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact analytical gradients w.r.t gammas and betas via Adjoint State Method.

        Mathematical Derivation:
        Let E = <psi_p | H_C | psi_p>.
        Back-propagating lambda_p = H_C |psi_p>:
            dE/d(beta_l)  = 2 * Im[ <lambda_l | H_M | psi_{l, m}> ]
            lambda_{l, c} = exp(i * beta_l * H_M) |lambda_l>
            dE/d(gamma_l) = 2 * Im[ <lambda_{l, c} | H_C | psi_{l, c}> ]
            lambda_{l-1}  = exp(i * gamma_l * H_C) |lambda_{l, c}>

        Complexity: O(p * n * 2^n) time, exactly 1 backward sweep.
        """
        g_arr = np.asarray(gammas, dtype=float)
        b_arr = np.asarray(betas, dtype=float)
        p = len(g_arr)
        if p == 0:
            return np.array([]), np.array([])

        final_psi, trajectory = self.forward_trajectory(g_arr, b_arr)
        probs = np.abs(final_psi) ** 2

        # Handle post-selected objective gradient weight if applicable
        if objective == "postselected" and feasible_mask is not None:
            f_mask = np.asarray(feasible_mask, dtype=float)
            y = float(np.sum(probs * f_mask))
            if y >= y_min:
                # E_ps = (<H_C>_F) / y
                e_ps = float(np.dot(probs * f_mask, self.diag) / (y + 1e-15))
                eff_diag = (self.diag - e_ps) * f_mask / (y + 1e-15)
                lambda_state = eff_diag * final_psi
            else:
                lambda_state = self.diag * final_psi
        else:
            lambda_state = self.diag * final_psi

        grad_g = np.zeros(p, dtype=float)
        grad_b = np.zeros(p, dtype=float)

        for l in reversed(range(p)):
            psi_c, psi_m = trajectory[l]

            # 1. Gradient w.r.t beta_l
            h_m_psi = compute_H_M_psi(psi_m, self.n_qubits)
            grad_b[l] = 2.0 * float(np.imag(np.vdot(lambda_state, h_m_psi)))

            # Pull back lambda through mixer: lambda <- U_M^\dagger(beta_l) |lambda>
            lambda_state = apply_mixer_layer(lambda_state, -b_arr[l], self.n_qubits)

            # 2. Gradient w.r.t gamma_l
            h_c_psi = self.diag * psi_c
            grad_g[l] = 2.0 * float(np.imag(np.vdot(lambda_state, h_c_psi)))

            # Pull back lambda through cost: lambda <- U_C^\dagger(gamma_l) |lambda>
            lambda_state = apply_cost_layer(lambda_state, -g_arr[l], self.diag)

        return grad_g, grad_b


def make_analytical_qnodes(
    problem: QAOAProblem,
    feasible_mask: np.ndarray | None = None,
    objective: str = "energy",
    y_min: float = 1e-4,
    warm_start_angles: np.ndarray | None = None,
) -> tuple[Callable, Callable]:
    """Create (probs_fn, cost_fn) backed by the Analytical Matrix-Free Simulator.

    Drop-in replacement for circuits.make_qnodes. Integrates with Pennylane/Autograd
    optimizers via custom vector-Jacobian products (VJPs).
    """
    sim = AnalyticalSimulator(problem, warm_start_angles=warm_start_angles)

    def probs_fn(gammas, betas) -> np.ndarray:
        g = np.asarray(gammas, dtype=float)
        b = np.asarray(betas, dtype=float)
        return sim.probabilities(g, b)

    @extend.primitive
    def cost_fn(gammas, betas):
        g = np.asarray(gammas, dtype=float)
        b = np.asarray(betas, dtype=float)
        return sim.energy(g, b, feasible_mask=feasible_mask, objective=objective, y_min=y_min)

    def vjp_gammas(ans, gammas, betas):
        def _vjp(v):
            g = np.asarray(gammas, dtype=float)
            b = np.asarray(betas, dtype=float)
            grad_g, _ = sim.adjoint_gradient(
                g, b, feasible_mask=feasible_mask, objective=objective, y_min=y_min
            )
            return v * grad_g
        return _vjp

    def vjp_betas(ans, gammas, betas):
        def _vjp(v):
            g = np.asarray(gammas, dtype=float)
            b = np.asarray(betas, dtype=float)
            _, grad_b = sim.adjoint_gradient(
                g, b, feasible_mask=feasible_mask, objective=objective, y_min=y_min
            )
            return v * grad_b
        return _vjp

    extend.defvjp(cost_fn, vjp_gammas, vjp_betas)

    return probs_fn, cost_fn
