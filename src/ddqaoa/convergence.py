"""Convergence detection that triggers a depth increase (paper Eqs. 18-19, Alg. 1)."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass
class CurvatureDecision:
    """Decision emitted by CurvatureTrigger."""
    converged: bool
    should_grow: bool
    should_stop: bool
    reason: str
    position: int = -1
    displaced_gammas: np.ndarray | None = None
    displaced_betas: np.ndarray | None = None
    lambda_min: float = 0.0
    v_min: np.ndarray | None = None
    predicted_gain: float = 0.0
    hessian_evals: int = 0


class CurvatureTrigger:
    """Curvature-triggered depth growth based on saddle-point escape (Feature 2).

    Inserting an identity layer (gamma=0, beta=0) at depth p yields a depth-(p+1)
    stationary saddle point (gradient ~ 0). If the Hessian has a negative eigenvalue
    lambda_min < -curv_tol, its eigenvector points along a descent direction.
    """

    def __init__(
        self,
        grad_tol: float = 1e-2,
        curv_tol: float = 1e-3,
        min_steps: int = 20,
        max_steps_per_depth: int = 150,
        position_mode: str = "end",  # "end" or "best"
        hessian_mode: str = "block", # "block" or "full"
        delta: float = 0.1,
    ) -> None:
        self.grad_tol = float(grad_tol)
        self.curv_tol = float(curv_tol)
        self.min_steps = int(min_steps)
        self.max_steps_per_depth = int(max_steps_per_depth)
        self.position_mode = position_mode
        self.hessian_mode = hessian_mode
        self.delta = float(delta)

        self.reset()

    def reset(self) -> None:
        """Reset step counters for a new depth."""
        self.steps_at_depth = 0
        self.history: list[float] = []

    def update(self, energy: float) -> bool:
        """Compatibility shim for ConvergenceMonitor interface."""
        self.steps_at_depth += 1
        self.history.append(float(energy))
        return False

    def evaluate_block_hessian(
        self,
        cost_fn,
        gammas: np.ndarray,
        betas: np.ndarray,
        position: int,
        h: float = 1e-3,
    ) -> tuple[np.ndarray, float]:
        """Compute 2x2 Hessian of inserted identity layer (gamma=0, beta=0) at position.

        Returns (Hessian_matrix, energy_at_zero).
        """
        g_base = list(np.asarray(gammas, dtype=float))
        b_base = list(np.asarray(betas, dtype=float))
        p = len(g_base)
        pos = p if position == -1 or position >= p else max(0, position)

        def eval_inserted(dg: float, db: float) -> float:
            g = list(g_base)
            b = list(b_base)
            g.insert(pos, dg)
            b.insert(pos, db)
            return float(cost_fn(np.asarray(g), np.asarray(b)))

        e00 = eval_inserted(0.0, 0.0)
        e_p0 = eval_inserted(h, 0.0)
        e_m0 = eval_inserted(-h, 0.0)
        e_0p = eval_inserted(0.0, h)
        e_0m = eval_inserted(0.0, -h)
        e_pp = eval_inserted(h, h)
        e_pm = eval_inserted(h, -h)
        e_mp = eval_inserted(-h, h)
        e_mm = eval_inserted(-h, -h)

        h_gg = (e_p0 - 2 * e00 + e_m0) / (h ** 2)
        h_bb = (e_0p - 2 * e00 + e_0m) / (h ** 2)
        h_gb = (e_pp - e_pm - e_mp + e_mm) / (4 * (h ** 2))

        H = np.array([[h_gg, h_gb], [h_gb, h_bb]], dtype=float)
        return H, e00

    def evaluate_full_hessian(
        self,
        cost_fn,
        gammas: np.ndarray,
        betas: np.ndarray,
        position: int,
        h: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute full 2(p+1) x 2(p+1) Hessian at inserted identity point."""
        g_base = list(np.asarray(gammas, dtype=float))
        b_base = list(np.asarray(betas, dtype=float))
        p = len(g_base)
        pos = p if position == -1 or position >= p else max(0, position)

        g_init = list(g_base)
        b_init = list(b_base)
        g_init.insert(pos, 0.0)
        b_init.insert(pos, 0.0)

        theta0 = np.concatenate([g_init, b_init])
        dim = len(theta0)
        p_new = len(g_init)

        def eval_theta(th: np.ndarray) -> float:
            return float(cost_fn(th[:p_new], th[p_new:]))

        e0 = eval_theta(theta0)
        H = np.zeros((dim, dim), dtype=float)

        # Diagonal entries
        for i in range(dim):
            th_p = np.array(theta0)
            th_m = np.array(theta0)
            th_p[i] += h
            th_m[i] -= h
            H[i, i] = (eval_theta(th_p) - 2 * e0 + eval_theta(th_m)) / (h ** 2)

        # Off-diagonal entries
        for i in range(dim):
            for j in range(i + 1, dim):
                th_pp = np.array(theta0)
                th_pm = np.array(theta0)
                th_mp = np.array(theta0)
                th_mm = np.array(theta0)
                th_pp[i] += h; th_pp[j] += h
                th_pm[i] += h; th_pm[j] -= h
                th_mp[i] -= h; th_mp[j] += h
                th_mm[i] -= h; th_mm[j] -= h
                val = (eval_theta(th_pp) - eval_theta(th_pm) - eval_theta(th_mp) + eval_theta(th_mm)) / (4 * (h ** 2))
                H[i, j] = val
                H[j, i] = val

        return H, np.array(g_init), np.array(b_init)

    def check_step(
        self,
        step: int,
        energy: float,
        grad_norm: float | None,
        gammas: np.ndarray,
        betas: np.ndarray,
        problem: Any,
        cost_fn: Any,
        ledger: Any | None = None,
    ) -> CurvatureDecision:
        """Evaluate whether to grow circuit depth based on gradient and Hessian curvature."""
        self.steps_at_depth += 1
        self.history.append(float(energy))
        p = len(gammas)

        # Safety fallback: force transition or stopping if max_steps_per_depth reached
        if self.steps_at_depth >= self.max_steps_per_depth:
            # Fall back to displacement along standard schedule
            from .transfer import zero_pad_displace
            g_new, b_new = zero_pad_displace(gammas, betas, delta=self.delta, position=-1)
            return CurvatureDecision(
                converged=True,
                should_grow=True,
                should_stop=False,
                reason=f"Safety fallback: reached max_steps_per_depth ({self.max_steps_per_depth})",
                position=p,
                displaced_gammas=g_new,
                displaced_betas=b_new,
            )

        # Wait until min_steps satisfied
        if self.steps_at_depth < self.min_steps:
            return CurvatureDecision(
                converged=False, should_grow=False, should_stop=False, reason="Below min_steps"
            )

        # Evaluate curvature only when gradient is near stationary point
        gnorm = grad_norm if grad_norm is not None else 0.0
        if gnorm > self.grad_tol:
            return CurvatureDecision(
                converged=False, should_grow=False, should_stop=False,
                reason=f"Gradient norm {gnorm:.4e} > grad_tol {self.grad_tol:.4e}"
            )

        # We are stationary! Evaluate curvature
        candidate_positions = [p] if self.position_mode == "end" else list(range(p + 1))
        best_lambda = 0.0
        best_v = None
        best_pos = -1
        best_g_new = None
        best_b_new = None
        best_gain = 0.0

        from .cost import estimate_hessian_cost
        n_qubits = problem.n_qubits
        n_zz = len(problem.ising.J)
        n_z = len(problem.ising.h)
        total_evals_spent = 0

        for pos in candidate_positions:
            eval_cost = estimate_hessian_cost(n_qubits, n_zz, p, mode=self.hessian_mode, n_z=n_z)
            total_evals_spent += eval_cost

            if self.hessian_mode == "block":
                H, e0 = self.evaluate_block_hessian(cost_fn, gammas, betas, pos)
                eigvals, eigvecs = np.linalg.eigh(H)
                lam_min = float(eigvals[0])
                v_min = eigvecs[:, 0]

                if lam_min < best_lambda:
                    best_lambda = lam_min
                    best_v = v_min
                    best_pos = pos

                    # Pick displacement direction (+v or -v) that lowers energy
                    g_pos = list(gammas)
                    b_pos = list(betas)
                    g_pos.insert(pos, float(self.delta * v_min[0]))
                    b_pos.insert(pos, float(self.delta * v_min[1]))
                    e_plus = float(cost_fn(np.asarray(g_pos), np.asarray(b_pos)))

                    g_neg = list(gammas)
                    b_neg = list(betas)
                    g_neg.insert(pos, float(-self.delta * v_min[0]))
                    b_neg.insert(pos, float(-self.delta * v_min[1]))
                    e_minus = float(cost_fn(np.asarray(g_neg), np.asarray(b_neg)))

                    if e_plus <= e_minus:
                        best_g_new = np.asarray(g_pos, dtype=float)
                        best_b_new = np.asarray(b_pos, dtype=float)
                    else:
                        best_g_new = np.asarray(g_neg, dtype=float)
                        best_b_new = np.asarray(b_neg, dtype=float)

                    best_gain = 0.5 * abs(lam_min) * (self.delta ** 2)

            elif self.hessian_mode == "full":
                H, g_init, b_init = self.evaluate_full_hessian(cost_fn, gammas, betas, pos)
                eigvals, eigvecs = np.linalg.eigh(H)
                lam_min = float(eigvals[0])
                v_min = eigvecs[:, 0]

                if lam_min < best_lambda:
                    best_lambda = lam_min
                    best_v = v_min
                    best_pos = pos

                    th0 = np.concatenate([g_init, b_init])
                    p_new = len(g_init)
                    th_plus = th0 + self.delta * v_min
                    th_minus = th0 - self.delta * v_min
                    if cost_fn(th_plus[:p_new], th_plus[p_new:]) <= cost_fn(th_minus[:p_new], th_minus[p_new:]):
                        best_g_new = th_plus[:p_new]
                        best_b_new = th_plus[p_new:]
                    else:
                        best_g_new = th_minus[:p_new]
                        best_b_new = th_minus[p_new:]

                    best_gain = 0.5 * abs(lam_min) * (self.delta ** 2)

        # Charge ledger
        if ledger is not None and total_evals_spent > 0:
            ledger.record_hessian_evaluations(total_evals_spent)

        # Trigger decision
        if best_lambda < -self.curv_tol and best_g_new is not None:
            return CurvatureDecision(
                converged=True,
                should_grow=True,
                should_stop=False,
                reason=f"Curvature trigger: lambda_min={best_lambda:.4e} < -{self.curv_tol:.4e} at pos {best_pos}",
                position=best_pos,
                displaced_gammas=best_g_new,
                displaced_betas=best_b_new,
                lambda_min=best_lambda,
                v_min=best_v,
                predicted_gain=best_gain,
                hessian_evals=total_evals_spent,
            )
        else:
            # Stationary point with no negative curvature below threshold
            return CurvatureDecision(
                converged=True,
                should_grow=False,
                should_stop=True,
                reason="no_descent_direction",
                position=best_pos,
                lambda_min=best_lambda,
                hessian_evals=total_evals_spent,
            )


