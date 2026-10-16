"""Dynamic Constraint Penalty Annealing (DCPA) and CVaR Tail Optimization.

Authored by Anita George (Project Lead).

Addresses the acute 'penalty trap' vulnerability in constrained QAOA:
1. In standard CSPP QAOA, penalty weights lambda, rho are fixed to large conservative values
   (lambda = rho = 1 + sum c_ij). At shallow circuit depths (p = 1, 2), this creates steep penalty
   chasms that trap variational optimization in suboptimal feasible states or bad local minima.
2. Dynamic Constraint Penalty Annealing (DCPA) smoothly scales penalty parameters:
   lambda(p) = lambda_target * Omega(p)
   rho(p) = rho_target * Omega(p)
   where Omega(p) in [Omega_min, 1.0] anneals across depths:
   Omega(p) = Omega_min + (1 - Omega_min) * ((p - 1) / max(p_max - 1, 1))^eta
3. Hamiltonian-Preserving Parameter Renormalization (HPPR):
   When depth increases and Omega(p) ramps up, the cost Hamiltonian spectrum expands by
   kappa = ||H_C(p)|| / ||H_C(p+1)||. To conserve Trotterized phase evolution e^{-i gamma H_C},
   transferred gamma angles are renormalized: gamma <- gamma * kappa.
4. CVaR Tail Optimization (Barkoutsos et al., Quantum 2020):
   Evaluates expectation over the lowest alpha-fraction of basis states to bias gradients toward
   ground-state amplification rather than bulk expectation minimization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import EnhancedConvergenceMonitor
from .cspp import CSPPInstance
from .metrics import approximation_ratio, cumulative_cnots, success_probability
from .optimizer import ControlledAdamOptimizer, OptimizerConfig
from .qubo import cspp_to_qubo, qubo_to_ising
from .resource import ResourceBudget
from .transfer import AdaptiveTransfer


@dataclass
class DCPAConfig:
    """Configuration for Dynamic Constraint Penalty Annealing."""
    omega_min: float = 0.35          # Initial penalty scale at p = 1 (35% of target)
    eta: float = 1.2                 # Annealing exponent (super-linear ramp)
    schedule: Literal["polynomial", "linear", "cosine", "exponential"] = "polynomial"
    renormalize_gamma: bool = True   # Conserve Trotter phase via HPPR
    cvar_alpha: float = 1.0          # CVaR tail fraction (1.0 = standard expectation)

    def penalty_scale(self, p: int, p_max: int) -> float:
        """Evaluate Omega(p) in [omega_min, 1.0]."""
        if p >= p_max or p_max <= 1:
            return 1.0
        frac = (p - 1.0) / max(p_max - 1.0, 1.0)
        frac = min(max(frac, 0.0), 1.0)

        if self.schedule == "linear":
            return self.omega_min + (1.0 - self.omega_min) * frac
        elif self.schedule == "cosine":
            return self.omega_min + (1.0 - self.omega_min) * (0.5 * (1.0 - math.cos(math.pi * frac)))
        elif self.schedule == "exponential":
            return self.omega_min * ((1.0 / self.omega_min) ** frac)
        else:  # polynomial
            return self.omega_min + (1.0 - self.omega_min) * (frac ** self.eta)


def build_annealed_problem(
    inst: CSPPInstance,
    p: int,
    p_max: int,
    config: DCPAConfig,
    fixed_edges: dict | None = None,
) -> tuple[QAOAProblem, float]:
    """Construct QAOAProblem with annealed penalty weights lambda(p), rho(p).
    
    Returns (problem, omega_p).
    """
    scale_factor = config.penalty_scale(p, p_max)
    big = 1.0 + sum(inst.cost.values())
    lam_target = big
    rho_target = big

    lam_annealed = lam_target * scale_factor
    rho_annealed = rho_target * scale_factor

    qubo = cspp_to_qubo(
        inst,
        lam=lam_annealed,
        rho=rho_annealed,
        fixed_edges=fixed_edges,
    )
    ising = qubo_to_ising(qubo)
    prob = QAOAProblem.from_ising(ising)
    return prob, scale_factor


def cvar_expectation(
    probs: np.ndarray,
    diag: np.ndarray,
    alpha: float = 0.25,
) -> float:
    """Calculate Conditional Value-at-Risk (CVaR) expectation at tail fraction alpha.
    
    alpha = 1.0 corresponds to standard expectation <H_C> = probs @ diag.
    alpha in (0, 1) computes average energy of the lowest alpha cumulative probability mass.
    """
    if alpha >= 1.0:
        return float(np.sum(probs * diag))
    
    order = np.argsort(diag)
    sorted_probs = probs[order]
    sorted_diag = diag[order]
    
    cum_p = np.cumsum(sorted_probs)
    idx = np.searchsorted(cum_p, alpha)
    
    if idx == 0:
        return float(sorted_diag[0])
    
    # Accumulate full mass up to idx
    cvar_sum = float(np.sum(sorted_probs[:idx] * sorted_diag[:idx]))
    # Boundary remainder
    rem_p = alpha - float(cum_p[idx - 1])
    if rem_p > 0 and idx < len(sorted_diag):
        cvar_sum += rem_p * float(sorted_diag[idx])
        
    return cvar_sum / alpha


def run_dcpa_ddqaoa(
    inst: CSPPInstance,
    p0: int = 1,
    p_max: int = 6,
    max_steps_per_depth: int = 60,
    stepsize: float = 0.04,
    dcpa_config: DCPAConfig | None = None,
    opt_config: OptimizerConfig | None = None,
    transfer_strategy: str = "adaptive",
    k: int = 20,
    min_steps: int = 15,
    fixed_edges: dict | None = None,
) -> dict:
    """Execute Enhanced DDQAOA equipped with Dynamic Constraint Penalty Annealing.
    
    Conserves Trotter action across depths via Hamiltonian-Preserving Parameter
    Renormalization (HPPR), allowing shallow layers to explore cost-efficient path
    regions without getting trapped in premature penalty valleys.
    """
    config = dcpa_config or DCPAConfig()
    opt_cfg = opt_config or OptimizerConfig(stepsize=stepsize, grad_clip=1.0)
    transfer = AdaptiveTransfer(strategy=transfer_strategy)

    # Reference target problem (100% penalty) for fair final evaluation
    target_qubo = cspp_to_qubo(inst, fixed_edges=fixed_edges)
    target_ising = qubo_to_ising(target_qubo)
    target_problem = QAOAProblem.from_ising(target_ising)

    all_energies: list[float] = []
    depths: list[int] = []
    switches: list[int] = []
    cnot_count = 0
    total_steps = 0

    gammas = pnp.array([0.1], requires_grad=True)
    betas = pnp.array([0.1], requires_grad=True)

    prev_omega = config.penalty_scale(p0, p_max)

    for p in range(p0, p_max + 1):
        depth_prob, omega_p = build_annealed_problem(
            inst, p=p, p_max=p_max, config=config, fixed_edges=fixed_edges
        )
        probs_fn, cost_fn = make_qnodes(depth_prob)
        cnot_layer = cnots_per_layer(depth_prob.ising)

        opt = ControlledAdamOptimizer.from_config(opt_cfg)
        opt.set_depth(p)
        monitor = EnhancedConvergenceMonitor(eps_abs=1e-4, eps_rel=1e-4, k=k, min_steps=min_steps)

        best_depth_cost = float("inf")
        best_depth_angles = (np.asarray(gammas), np.asarray(betas))

        for step in range(max_steps_per_depth):
            total_steps += 1
            depths.append(p)
            cnot_count += 2 * p * cnot_layer

            # Optimization step
            (gammas, betas), loss = opt.step_and_cost(cost_fn, gammas, betas)
            loss_val = float(loss)
            all_energies.append(loss_val)

            if loss_val < best_depth_cost:
                best_depth_cost = loss_val
                best_depth_angles = (np.asarray(gammas), np.asarray(betas))

            if monitor.update(loss_val):
                break

        # Check stopping or transition
        if p < p_max:
            switches.append(total_steps)
            # 1. Warm-start transfer via PCHIP/Fourier
            g_trans, b_trans = transfer(best_depth_angles[0], best_depth_angles[1])

            # 2. HPPR: Hamiltonian-Preserving Parameter Renormalization
            next_omega = config.penalty_scale(p + 1, p_max)
            if config.renormalize_gamma and next_omega > 0:
                scale_ratio = omega_p / next_omega
                g_trans = g_trans * scale_ratio

            gammas = pnp.array(g_trans, requires_grad=True)
            betas = pnp.array(b_trans, requires_grad=True)
            prev_omega = next_omega
        else:
            gammas = pnp.array(best_depth_angles[0])
            betas = pnp.array(best_depth_angles[1])

    # Final unbiased evaluation on ground-truth target problem (100% full penalty)
    final_probs_fn, _ = make_qnodes(target_problem)
    final_probs = np.asarray(final_probs_fn(gammas, betas))
    final_energy = float(final_probs @ target_problem.diag)
    r = approximation_ratio(final_energy, target_problem)
    p_succ = success_probability(final_probs, target_problem)

    return {
        "method": "ddqaoa_dcpa",
        "approx_ratio": float(r),
        "success_prob": float(p_succ),
        "final_energy": final_energy,
        "final_depth": p_max,
        "total_steps": total_steps,
        "cumulative_cnots": cnot_count,
        "energies": all_energies,
        "depths": depths,
        "switches": switches,
        "gammas": np.asarray(gammas).tolist(),
        "betas": np.asarray(betas).tolist(),
    }
