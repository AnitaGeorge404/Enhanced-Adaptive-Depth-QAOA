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


def fourier_transfer(gammas, betas, q_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FOURIER parameter transfer based on Zhou et al. (PRX 2020).

    Expresses gamma and beta in discrete sine and cosine bases:
        gamma_i = sum_k u_k * sin((k - 0.5) * (i - 0.5) * pi / p)
        beta_i  = sum_k v_k * cos((k - 0.5) * (i - 0.5) * pi / p)

    Parameters at depth p are inverted to find continuous spectral weights (u, v),
    which are then evaluated on the denser grid at depth p + 1.
    """
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    p = len(gammas)
    if p == 1:
        # For p=1, expand to p=2 using adiabatic schedule direction
        return (np.array([0.85 * gammas[0], 1.15 * gammas[0]]),
                np.array([1.15 * betas[0], 0.85 * betas[0]]))

    q = min(p, q_max) if q_max is not None else p
    k = np.arange(1, q + 1, dtype=float)  # shape (q,)

    # Evaluation grid at depth p: i in 1..p
    i_old = np.arange(1, p + 1, dtype=float)[:, None]  # shape (p, 1)
    # Basis matrices at depth p
    S_old = np.sin((k - 0.5) * (i_old - 0.5) * np.pi / p)
    C_old = np.cos((k - 0.5) * (i_old - 0.5) * np.pi / p)

    # Invert via least-squares
    u, _, _, _ = np.linalg.lstsq(S_old, gammas, rcond=None)
    v, _, _, _ = np.linalg.lstsq(C_old, betas, rcond=None)

    # Evaluation grid at depth p + 1: j in 1..(p+1)
    j_new = np.arange(1, p + 2, dtype=float)[:, None]  # shape (p+1, 1)
    S_new = np.sin((k - 0.5) * (j_new - 0.5) * np.pi / (p + 1))
    C_new = np.cos((k - 0.5) * (j_new - 0.5) * np.pi / (p + 1))

    g_new = S_new @ u
    b_new = C_new @ v

    # Bound clipping for numerical and quantum stability
    g_new = np.clip(g_new, -np.pi, np.pi)
    b_new = np.clip(b_new, -np.pi / 2, np.pi / 2)
    return g_new, b_new


def adaptive_transfer(
    gammas,
    betas,
    delta: float = 0.15,
    method: str = "pchip",
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive shape-preserving parameter transfer (Improvement 2).

    1. p=1 -> 2: Uses adiabatic action-conserving split:
       gamma -> [gamma*(1-delta), gamma*(1+delta)]
       beta  -> [beta*(1+delta),  beta*(1-delta)]
    2. p >= 2: Uses monotonicity-preserving Hermite interpolation (PCHIP)
       which avoids the overshoot artifacts of cubic splines near boundaries.
    3. Clamps angles into physical QAOA ranges to prevent drift.
    """
    from scipy.interpolate import PchipInterpolator

    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    p = len(gammas)

    if p == 1:
        g0, b0 = gammas[0], betas[0]
        return (np.array([g0 * (1.0 - delta), g0 * (1.0 + delta)]),
                np.array([b0 * (1.0 + delta), b0 * (1.0 - delta)]))

    old_s = np.linspace(0.0, 1.0, p)
    new_s = np.linspace(0.0, 1.0, p + 1)

    if method == "pchip" and p >= 2:
        try:
            g_interp = PchipInterpolator(old_s, gammas)(new_s)
            b_interp = PchipInterpolator(old_s, betas)(new_s)
        except Exception:
            g_interp = np.interp(new_s, old_s, gammas)
            b_interp = np.interp(new_s, old_s, betas)
    else:
        kind = "cubic" if p >= 4 else "linear"
        g_interp = _interp(gammas, kind)
        b_interp = _interp(betas, kind)

    # Physical angle bounds clipping
    g_interp = np.clip(g_interp, -np.pi, np.pi)
    b_interp = np.clip(b_interp, -np.pi / 2, np.pi / 2)
    return np.asarray(g_interp, dtype=float), np.asarray(b_interp, dtype=float)


def zero_pad_displace(
    gammas,
    betas,
    delta: float = 0.1,
    position: int = -1,
    v_min: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert an identity layer (gamma=0, beta=0) and displace along descent eigenvector v_min.

    theta_0 = theta_TS + delta * v_min.
    Displacement is required because Adam optimizer does not move from a point with zero gradient.
    """
    g = list(np.asarray(gammas, dtype=float))
    b = list(np.asarray(betas, dtype=float))
    p = len(g)

    if position == -1 or position >= p:
        insert_idx = p
    elif position < 0:
        insert_idx = max(0, p + 1 + position)
    else:
        insert_idx = min(p, position)

    if v_min is not None:
        v = np.asarray(v_min, dtype=float).flatten()
        if len(v) >= 2:
            dg, db = float(delta * v[0]), float(delta * v[1])
        else:
            dg, db = float(delta), float(delta)
    else:
        dg = float(delta / np.sqrt(2))
        db = float(-delta / np.sqrt(2))

    g.insert(insert_idx, dg)
    b.insert(insert_idx, db)

    g_new = np.clip(np.asarray(g, dtype=float), -np.pi, np.pi)
    b_new = np.clip(np.asarray(b, dtype=float), -np.pi / 2, np.pi / 2)
    return g_new, b_new


class AdaptiveTransfer:
    """Configurable parameter transfer dispatcher supporting multiple strategies."""

    def __init__(self, strategy: str = "adaptive", **kwargs) -> None:
        self.strategy = strategy
        self.kwargs = kwargs

    def __call__(self, gammas, betas) -> tuple[np.ndarray, np.ndarray]:
        if self.strategy == "adaptive":
            return adaptive_transfer(gammas, betas, **self.kwargs)
        elif self.strategy == "fourier":
            return fourier_transfer(gammas, betas, **self.kwargs)
        elif self.strategy == "interp":
            return interp_transfer(gammas, betas, **self.kwargs)
        elif self.strategy == "zero_pad_displace":
            return zero_pad_displace(gammas, betas, **self.kwargs)
        elif self.strategy == "ramp":
            p = len(gammas) + 1
            frac = (np.arange(p) + 0.5) / p
            return 0.8 * frac, 0.8 * (1.0 - frac)
        else:
            raise ValueError(f"Unknown transfer strategy: {self.strategy}")


