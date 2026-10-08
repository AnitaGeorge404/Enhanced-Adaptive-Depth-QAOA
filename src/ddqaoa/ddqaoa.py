"""Dynamic Depth QAOA (paper Algorithm 1) and the fixed-depth QAOA baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import ConvergenceMonitor
from .metrics import approximation_ratio, cumulative_cnots, success_probability
from .transfer import interp_transfer

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

    def summary(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}


def _finish(method, problem, probs_fn, gammas, betas, energies, depths, switches) -> RunResult:
    probs = np.asarray(probs_fn(gammas, betas))
    energy = float(probs @ problem.diag)
    return RunResult(
        method=method,
        gammas=np.asarray(gammas), betas=np.asarray(betas),
        best_energy=energy,
        approx_ratio=approximation_ratio(energy, problem),
        success_prob=success_probability(probs, problem),
        final_depth=len(gammas),
        cumulative_cnots=cumulative_cnots(depths, cnots_per_layer(problem.ising)),
        energies=energies, depths=depths, depth_switch_steps=switches,
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
) -> RunResult:
    """Grow the circuit from p0 to p_max, adding a layer whenever the energy converges.

    ``transfer`` and ``monitor`` are the two extension points (new parameter-transfer
    rules / new convergence criteria). ``callback(step, depth, energy)`` is called
    after every optimiser step (e.g. for live progress output).
    """
    probs_fn, cost_fn = make_qnodes(problem, device)
    monitor = monitor or ConvergenceMonitor(eps, sigma, k)

    gammas = pnp.array([init[0]] * p0, requires_grad=True)
    betas = pnp.array([init[1]] * p0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize)
    best = (np.inf, gammas, betas)  # best parameters at the *current* depth
    energies, depths, switches = [], [], []

    for step in range(max_steps):
        params, energy = opt.step_and_cost(cost_fn, gammas, betas)
        energy = float(energy)  # energy of (gammas, betas), i.e. before this step
        energies.append(energy)
        depths.append(len(gammas))
        if energy < best[0]:
            best = (energy, gammas, betas)
        if callback:
            callback(step, len(gammas), energy)
        gammas, betas = params

        if monitor.update(energy):
            if len(gammas) >= p_max:
                break
            g_new, b_new = transfer(np.asarray(best[1]), np.asarray(best[2]))
            gammas = pnp.array(g_new, requires_grad=True)
            betas = pnp.array(b_new, requires_grad=True)
            opt = qml.AdamOptimizer(stepsize)  # re-initialise Adam moments
            monitor.reset()
            best = (np.inf, gammas, betas)
            switches.append(step + 1)

    # Final evaluation at the deepest circuit reached (Alg. 1, line 20).
    if float(cost_fn(gammas, betas)) > best[0]:
        gammas, betas = best[1], best[2]
    return _finish("ddqaoa", problem, probs_fn, gammas, betas, energies, depths, switches)


def run_fixed_qaoa(
    problem: QAOAProblem,
    p: int,
    steps: int = 1200,
    stepsize: float = 0.01,
    init: str = "random",
    seed: int | None = None,
    device: str = "default.qubit",
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
    opt = qml.AdamOptimizer(stepsize)
    energies = []
    for _ in range(steps):
        (gammas, betas), energy = opt.step_and_cost(cost_fn, gammas, betas)
        energies.append(float(energy))
    return _finish(f"p={p}", problem, probs_fn, gammas, betas, energies, [p] * steps, [])
