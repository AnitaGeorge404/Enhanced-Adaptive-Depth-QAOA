"""Parameter transfer p -> p+1 (paper Eqs. 20-22, INTERP of Zhou et al. 2020).

A transfer strategy is any callable (gammas, betas) -> (gammas', betas') returning
p+1 parameters, so alternatives (e.g. FOURIER, random, ...) can be plugged into
``run_ddqaoa`` as extensions.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _interp(params: np.ndarray, kind: str) -> np.ndarray:
    p = len(params)
    old = np.linspace(0.0, 1.0, p)      # s_k = (k-1)/(p-1)
    new = np.linspace(0.0, 1.0, p + 1)  # s'_j = (j-1)/p
    if kind == "cubic":
        return CubicSpline(old, params)(new)
    return np.interp(new, old, params)


def interp_transfer(gammas, betas, gamma_coef: float = 1.2, beta_coef: float = 0.8):
    """Paper's adaptive interpolation transfer."""
    gammas, betas = np.asarray(gammas, float), np.asarray(betas, float)
    p = len(gammas)
    if p == 1:
        return (np.array([gammas[0], gamma_coef * gammas[0]]),
                np.array([betas[0], beta_coef * betas[0]]))
    kind = "cubic" if p >= 4 else "linear"
    return _interp(gammas, kind), _interp(betas, kind)
