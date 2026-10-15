"""Dynamic Depth QAOA (paper Algorithm 1) and the fixed-depth QAOA baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import ConvergenceMonitor, CurvatureTrigger, EnhancedConvergenceMonitor
from .cost import CostLedger, GradientMethod
from .metrics import approximation_ratio, cumulative_cnots, success_probability
from .optimizer import ControlledAdamOptimizer, OptimizerConfig
from .resource import ResourceBudget, ResourceDecision
from .transfer import AdaptiveTransfer, adaptive_transfer, fourier_transfer, interp_transfer

TransferFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class RunResult:
    method: str
    gammas: np.ndarray
    betas: np.ndarray
    best_energy: float
    approx_ratio: float
    success_prob: float
    final_depth: int
    cumulative_cnots: int
    energies: list[float] = field(default_factory=list)    # per optimiser step
    depths: list[int] = field(default_factory=list)        # circuit depth used at each step
    depth_switch_steps: list[int] = field(default_factory=list)
    stopping_reason: str = ""
    grad_norms: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ledger: CostLedger | None = None

    def summary(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.__dict__.items() if k != "ledger"}
        if self.ledger is not None:
            d["ledger"] = self.ledger.summary()
        return d


def _finish(
    method: str,
    problem: QAOAProblem,
    probs_fn,
    gammas,
    betas,
    energies: list[float],
    depths: list[int],
    switches: list[int],
    stopping_reason: str = "",
    grad_norms: list[float] | None = None,
    metadata: dict | None = None,
    ledger: CostLedger | None = None,
) -> RunResult:
    probs = np.asarray(probs_fn(gammas, betas))
    energy = float(probs @ problem.diag)
    meta = dict(metadata or {})
    if ledger is not None:
        meta["ledger"] = ledger.summary()
    return RunResult(
        method=method,
        gammas=np.asarray(gammas),
        betas=np.asarray(betas),
        best_energy=energy,
        approx_ratio=approximation_ratio(energy, problem),
        success_prob=success_probability(probs, problem),
        final_depth=len(gammas),
        cumulative_cnots=cumulative_cnots(depths, cnots_per_layer(problem.ising)),
        energies=energies,
        depths=depths,
        depth_switch_steps=switches,
        stopping_reason=stopping_reason,
        grad_norms=grad_norms or [],
        metadata=meta,
        ledger=ledger,
    )


def run_ddqaoa(
    problem: QAOAProblem,
    p0: int = 1,
    p_max: int = 10,
    max_steps: int = 2000,
    stepsize: float = 0.01,
    eps: float = 1e-4,
    sigma: float = 1e-6,
    k: int = 50,
    init: tuple[float, float] = (0.1, 0.1),
    transfer: TransferFn = interp_transfer,
    monitor: ConvergenceMonitor | None = None,
    device: str = "default.qubit",
    callback: Callable[[int, int, float], None] | None = None,
    resource_budget: ResourceBudget | None = None,
    optimizer_config: OptimizerConfig | None = None,
    method_name: str = "ddqaoa",
    ledger: CostLedger | None = None,
    gradient_method: GradientMethod = "autodiff",
    shots: int | None = None,
) -> RunResult:
    """Grow the circuit from p0 to p_max, adding a layer whenever the energy converges.

    ``transfer`` and ``monitor`` are the extension points (new parameter-transfer
    rules / new convergence criteria). ``resource_budget`` and ``optimizer_config``
    provide resource-aware stopping and controlled optimization.
    """
    probs_fn, cost_fn = make_qnodes(problem, device)
    monitor = monitor if monitor is not None else ConvergenceMonitor(eps, sigma, k)

    if ledger is None:
        ledger = CostLedger(
            gradient_method=gradient_method,
            shots=shots,
            cnots_per_layer=cnots_per_layer(problem.ising),
            n_qubits=problem.n_qubits,
            n_zz=len(problem.ising.J),
            n_z=len(problem.ising.h),
        )

    gammas = pnp.array([init[0]] * p0, requires_grad=True)
    betas = pnp.array([init[1]] * p0, requires_grad=True)

    if optimizer_config is not None:
        opt = ControlledAdamOptimizer.from_config(optimizer_config)
        opt.set_depth(p0)
    else:
        opt = qml.AdamOptimizer(stepsize)

    best = (np.inf, gammas, betas)  # best parameters at the *current* depth
    energies, depths, switches = [], [], []
    grad_norms: list[float] = []
    prev_depth_best_ratio: float | None = None
    stopping_reason = f"Max steps ({max_steps}) reached"

    for step in range(max_steps):
        params, energy = opt.step_and_cost(cost_fn, gammas, betas)
        energy = float(energy)  # energy of (gammas, betas), i.e. before this step
        energies.append(energy)
        depths.append(len(gammas))

        g_norm = opt.last_grad_norm if isinstance(opt, ControlledAdamOptimizer) else None
        if g_norm is not None:
            grad_norms.append(g_norm)
        ledger.record_step(step=step, depth=len(gammas), energy=energy, grad_norm=g_norm)

        if energy < best[0]:
            best = (energy, gammas, betas)
        if callback:
            callback(step, len(gammas), energy)
        gammas, betas = params

        # Check transition condition
        if isinstance(monitor, CurvatureTrigger):
            curv_decision = monitor.check_step(
                step=step,
                energy=energy,
                grad_norm=g_norm,
                gammas=np.asarray(best[1]),
                betas=np.asarray(best[2]),
                problem=problem,
                cost_fn=cost_fn,
                ledger=ledger,
            )
            if curv_decision.should_stop:
                stopping_reason = curv_decision.reason
                break
            should_grow = curv_decision.should_grow
            displaced_g = curv_decision.displaced_gammas
            displaced_b = curv_decision.displaced_betas
        else:
            should_grow = monitor.update(energy)
            displaced_g, displaced_b = None, None

        if should_grow:
            curr_r = approximation_ratio(best[0], problem)

            # Evaluate resource-aware stopping if budget configured
            if resource_budget is not None:
                current_cnots = cumulative_cnots(depths, cnots_per_layer(problem.ising))
                best_probs = np.asarray(probs_fn(best[1], best[2]))
                curr_ps = success_probability(best_probs, problem)

                decision = resource_budget.evaluate_transition(
                    current_depth=len(gammas),
                    approx_ratio_current=curr_r,
                    approx_ratio_prev=prev_depth_best_ratio,
                    success_prob_current=curr_ps,
                    cumulative_cnots_used=current_cnots,
                    cnots_per_layer=cnots_per_layer(problem.ising),
                )
                if decision.stop:
                    stopping_reason = decision.reason
                    break

            prev_depth_best_ratio = curr_r

            if len(gammas) >= p_max:
                stopping_reason = f"Reached p_max={p_max}"
                break

            if displaced_g is not None and displaced_b is not None:
                g_new, b_new = displaced_g, displaced_b
            else:
                g_new, b_new = transfer(np.asarray(best[1]), np.asarray(best[2]))

            gammas = pnp.array(g_new, requires_grad=True)
            betas = pnp.array(b_new, requires_grad=True)

            if optimizer_config is not None:
                opt = ControlledAdamOptimizer.from_config(optimizer_config)
                opt.set_depth(len(gammas))
            else:
                opt = qml.AdamOptimizer(stepsize)  # re-initialise Adam moments

            monitor.reset()
            best = (np.inf, gammas, betas)
            switches.append(step + 1)

    # Final evaluation at the deepest circuit reached (Alg. 1, line 20).
    if float(cost_fn(gammas, betas)) > best[0]:
        gammas, betas = best[1], best[2]

    return _finish(
        method_name,
        problem,
        probs_fn,
        gammas,
        betas,
        energies,
        depths,
        switches,
        stopping_reason=stopping_reason,
        grad_norms=grad_norms,
        ledger=ledger,
    )


def run_enhanced_ddqaoa(
    problem: QAOAProblem,
    p0: int = 1,
    p_max: int = 10,
    max_steps: int = 2000,
    stepsize: float = 0.01,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    sigma_rel: float = 1e-6,
    k: int = 40,
    min_steps_per_depth: int = 20,
    slope_tol: float = 1e-5,
    transfer_strategy: str = "adaptive",
    max_cnots: int | None = None,
    min_marginal_gain: float = 1e-3,
    patience_layers: int = 1,
    target_approx_ratio: float | None = None,
    grad_clip: float | None = 1.0,
    depth_decay: float = 0.1,
    init: tuple[float, float] = (0.1, 0.1),
    device: str = "default.qubit",
    callback: Callable[[int, int, float], None] | None = None,
    method_name: str = "ddqaoa_enhanced",
    ledger: CostLedger | None = None,
    gradient_method: GradientMethod = "autodiff",
    shots: int | None = None,
) -> RunResult:
    """Enhanced DDQAOA combining all four algorithmic improvements:

    1. EnhancedConvergenceMonitor: scale-invariant dual thresholds, min-step guard, slope test.
    2. Adaptive parameter transfer: adiabatic PCHIP interpolation avoiding overshoot artifacts.
    3. Resource-aware stopping: checks marginal ratio improvement vs CNOT gate consumption.
    4. Controlled Adam optimizer: gradient norm clipping, depth-adaptive learning rate, NaN recovery.
    """
    monitor = EnhancedConvergenceMonitor(
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        sigma_rel=sigma_rel,
        k=k,
        min_steps=min_steps_per_depth,
        slope_tol=slope_tol,
    )

    transfer = AdaptiveTransfer(strategy=transfer_strategy)

    budget = ResourceBudget(
        max_cnots=max_cnots,
        min_marginal_gain=min_marginal_gain,
        patience_layers=patience_layers,
        target_approx_ratio=target_approx_ratio,
        min_depth=2,
    )

    opt_config = OptimizerConfig(
        stepsize=stepsize,
        grad_clip=grad_clip,
        depth_decay=depth_decay,
        recover_nan=True,
    )

    return run_ddqaoa(
        problem=problem,
        p0=p0,
        p_max=p_max,
        max_steps=max_steps,
        stepsize=stepsize,
        init=init,
        transfer=transfer,
        monitor=monitor,
        device=device,
        callback=callback,
        resource_budget=budget,
        optimizer_config=opt_config,
        method_name=method_name,
        ledger=ledger,
        gradient_method=gradient_method,
        shots=shots,
    )


def run_fixed_qaoa(
    problem: QAOAProblem,
    p: int,
    steps: int = 1200,
    stepsize: float = 0.01,
    init: str = "random",
    seed: int | None = None,
    device: str = "default.qubit",
    optimizer_config: OptimizerConfig | None = None,
    ledger: CostLedger | None = None,
    gradient_method: GradientMethod = "autodiff",
    shots: int | None = None,
) -> RunResult:
    """Standard QAOA at fixed depth p, optimised with Adam for a fixed number of steps.

    init: "random" (U[0, 1)) or "ramp" (linear-ramp / annealing-like schedule).
    """
    probs_fn, cost_fn = make_qnodes(problem, device)
    if init == "ramp":
        frac = (np.arange(p) + 0.5) / p
        g0, b0 = 0.8 * frac, 0.8 * (1 - frac)
    else:
        rng = np.random.default_rng(seed)
        g0, b0 = rng.random(p), rng.random(p)
    gammas = pnp.array(g0, requires_grad=True)
    betas = pnp.array(b0, requires_grad=True)

    if ledger is None:
        ledger = CostLedger(
            gradient_method=gradient_method,
            shots=shots,
            cnots_per_layer=cnots_per_layer(problem.ising),
            n_qubits=problem.n_qubits,
            n_zz=len(problem.ising.J),
            n_z=len(problem.ising.h),
        )

    if optimizer_config is not None:
        opt = ControlledAdamOptimizer.from_config(optimizer_config)
        opt.set_depth(p)
    else:
        opt = qml.AdamOptimizer(stepsize)

    energies = []
    grad_norms: list[float] = []

    for step in range(steps):
        (gammas, betas), energy = opt.step_and_cost(cost_fn, gammas, betas)
        energy_val = float(energy)
        energies.append(energy_val)
        g_norm = opt.last_grad_norm if isinstance(opt, ControlledAdamOptimizer) else None
        if g_norm is not None:
            grad_norms.append(g_norm)
        ledger.record_step(step=step, depth=p, energy=energy_val, grad_norm=g_norm)

    return _finish(
        f"p={p}",
        problem,
        probs_fn,
        gammas,
        betas,
        energies,
        [p] * steps,
        [],
        grad_norms=grad_norms,
        ledger=ledger,
    )

