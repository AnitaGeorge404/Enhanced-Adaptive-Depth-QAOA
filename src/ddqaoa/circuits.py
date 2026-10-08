"""QAOA ansatz (paper Eq. 1) for a diagonal Ising cost Hamiltonian, in PennyLane.

Because H_C is diagonal, <H_C> = sum_x p(x) E(x). We precompute the energy of every
basis state once (``ising_diagonal``) and evaluate the cost as probs @ diag, which is
exact, differentiable by backprop, and gives success probabilities for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

from .qubo import Ising


def ising_diagonal(ising: Ising) -> np.ndarray:
    """E(z) for all 2^n basis states. Wire 0 is the most significant bit (PennyLane order)."""
    n = ising.n_qubits
    idx = np.arange(2**n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    z = 1 - 2 * bits
    e = np.full(2**n, ising.offset, dtype=float)
    for i, hi in ising.h.items():
        e += hi * z[:, i]
    for (i, j), Jij in ising.J.items():
        e += Jij * z[:, i] * z[:, j]
    return e


def index_to_bits(index: int, n: int) -> np.ndarray:
    return np.array([(index >> (n - 1 - k)) & 1 for k in range(n)])


def to_pennylane_hamiltonian(ising: Ising) -> qml.Hamiltonian:
    coeffs, ops = [], []
    for i, hi in ising.h.items():
        coeffs.append(hi)
        ops.append(qml.Z(i))
    for (i, j), Jij in ising.J.items():
        coeffs.append(Jij)
        ops.append(qml.Z(i) @ qml.Z(j))
    return qml.Hamiltonian(coeffs, ops)


def cnots_per_layer(ising: Ising) -> int:
    """Each ZZ term compiles to CNOT-RZ-CNOT (paper Sec. 5.2)."""
    return 2 * len(ising.J)


@dataclass
class QAOAProblem:
    """Everything the optimisers need for one instance."""
    ising: Ising
    diag: np.ndarray      # energies of all basis states (scaled)
    scale: float          # diag = raw_energy / scale
    e_min: float
    e_max: float
    ground_states: np.ndarray  # indices achieving e_min

    @classmethod
    def from_ising(cls, ising: Ising, normalize: bool = True) -> "QAOAProblem":
        """normalize: divide H by max |coefficient| so gamma ~ O(1) works for any instance."""
        scale = 1.0
        if normalize:
            scale = max([abs(v) for v in ising.h.values()] + [abs(v) for v in ising.J.values()])
            ising = Ising(ising.n_qubits,
                          {k: v / scale for k, v in ising.h.items()},
                          {k: v / scale for k, v in ising.J.items()},
                          ising.offset / scale)
        diag = ising_diagonal(ising)
        e_min = float(diag.min())
        ground = np.flatnonzero(np.isclose(diag, e_min))
        return cls(ising, diag, scale, e_min, float(diag.max()), ground)

    @property
    def n_qubits(self) -> int:
        return self.ising.n_qubits


def qaoa_layer(gamma, beta, ising: Ising) -> None:
    """exp(-i beta H_M) exp(-i gamma H_C)."""
    for i, hi in ising.h.items():
        qml.RZ(2 * gamma * hi, wires=i)
    for (i, j), Jij in ising.J.items():
        qml.IsingZZ(2 * gamma * Jij, wires=[i, j])
    for i in range(ising.n_qubits):
        qml.RX(2 * beta, wires=i)


def make_qnodes(problem: QAOAProblem, device: str = "default.qubit", device_kwargs=None):
    """Return (probs_fn, cost_fn); both take (gammas, betas) of any length p."""
    n = problem.n_qubits
    dev = qml.device(device, wires=n, **(device_kwargs or {}))

    @qml.qnode(dev, diff_method="backprop")
    def probs_fn(gammas, betas):
        for w in range(n):
            qml.Hadamard(wires=w)
        for g, b in zip(gammas, betas):
            qaoa_layer(g, b, problem.ising)
        return qml.probs(wires=range(n))

    diag = problem.diag

    def cost_fn(gammas, betas):
        return qml.math.dot(probs_fn(gammas, betas), diag)

    return probs_fn, cost_fn
