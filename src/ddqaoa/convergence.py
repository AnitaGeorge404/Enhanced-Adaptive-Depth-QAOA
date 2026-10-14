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


class EnhancedConvergenceMonitor:
    """Robust adaptive convergence monitor for DDQAOA (Improvement 1).

    Extends plateau-and-variance detection with:
    1. Minimum step enforcement (min_steps) to prevent premature depth increases.
    2. Dual absolute (eps_abs) and relative (eps_rel) improvement criteria,
       making convergence scale-invariant.
    3. Per-depth reference tracking (depth_best) so initial parameter relaxation
       at a new depth does not falsely increment the plateau counter.
    4. Rolling window linear trend slope check to ensure depth is not increased
       while energy is still actively descending.
    5. Scale-normalized variance test (var / scale^2 < sigma_rel).
    """

    def __init__(
        self,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        sigma_rel: float = 1e-6,
        k: int = 40,
        min_steps: int = 20,
        slope_tol: float = 1e-5,
        window: int | None = None,
    ) -> None:
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.sigma_rel = float(sigma_rel)
        self.k = int(k)
        self.min_steps = int(min_steps)
        self.slope_tol = float(slope_tol)
        self.w_len = int(window) if window is not None else max(5, self.k // 2)

        self.global_best = math.inf
        self.reset()

    def reset(self) -> None:
        """Reset state for a new circuit depth."""
        self.history: list[float] = []
        self.counter = 0
        self.depth_step = 0
        self.depth_best = math.inf

    def update(self, energy: float) -> bool:
        """Record energy step and evaluate convergence."""
        energy = float(energy)
        self.history.append(energy)
        self.depth_step += 1

        # Check local per-depth improvement
        delta_abs = self.depth_best - energy
        scale = max(abs(self.depth_best), 1.0) if math.isfinite(self.depth_best) else 1.0
        delta_rel = delta_abs / scale

        if delta_abs > self.eps_abs and delta_rel > self.eps_rel:
            self.depth_best = energy
            self.counter = 0
        else:
            self.counter += 1

        if energy < self.global_best:
            self.global_best = energy

        # Guard against premature transitions before min_steps
        if self.depth_step < self.min_steps or self.depth_step < self.k:
            return False

        plateau = self.counter >= self.k

        # Rolling window evaluation
        w = np.asarray(self.history[-self.w_len:], dtype=float)
        mean_scale = max(float(np.mean(np.abs(w))), 1.0)
        norm_var = float(np.var(w)) / (mean_scale ** 2)
        low_var = norm_var < self.sigma_rel

        # Trend slope check: linear regression slope of energy over window
        # E_t = beta_0 + beta_1 * t
        n = len(w)
        t = np.arange(n, dtype=float)
        t_var = np.var(t)
        slope = float(np.cov(t, w, bias=True)[0, 1] / t_var) if t_var > 1e-12 else 0.0

        # If energy is still descending meaningfully, do not declare convergence
        descending = slope < -self.slope_tol * mean_scale
        if descending:
            return False

        return plateau or low_var

    def diagnostics(self) -> dict[str, float | int | bool]:
        """Return diagnostic metrics for monitoring and debugging."""
        w = np.asarray(self.history[-self.w_len:], dtype=float) if len(self.history) >= 2 else np.array([0.0])
        mean_scale = max(float(np.mean(np.abs(w))), 1.0)
        norm_var = float(np.var(w)) / (mean_scale ** 2) if len(w) > 1 else 0.0
        return {
            "depth_step": self.depth_step,
            "counter": self.counter,
            "depth_best": self.depth_best,
            "global_best": self.global_best,
            "norm_var": norm_var,
        }

