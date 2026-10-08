"""Convergence detection that triggers a depth increase (paper Eqs. 18-19, Alg. 1)."""
from __future__ import annotations

import math

import numpy as np


class ConvergenceMonitor:
    """Plateau (no improvement > eps for k steps) OR low variance over the last k/2 steps.

    Call ``update(E_t)`` once per optimiser step; it returns True when the current
    depth is considered converged. Call ``reset()`` after growing the circuit.
    """

    def __init__(self, eps: float = 1e-4, sigma: float = 1e-6, k: int = 50) -> None:
        self.eps, self.sigma, self.k = eps, sigma, k
        self.best = math.inf
        self.reset()

    def reset(self) -> None:
        # E_best is intentionally kept across depths (Alg. 1); history/counter are per depth.
        self.history: list[float] = []
        self.counter = 0

    def update(self, energy: float) -> bool:
        self.history.append(energy)
        if energy < self.best - self.eps:
            self.best, self.counter = energy, 0
        else:
            self.counter += 1
        if len(self.history) < self.k:  # wait for the patience window
            return False
        plateau = self.counter >= self.k
        window = self.history[-math.ceil(self.k / 2):]
        return plateau or float(np.var(window)) < self.sigma
