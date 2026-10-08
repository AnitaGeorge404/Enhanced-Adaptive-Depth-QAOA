"""DDQAOA for the Constrained Shortest Path Problem.

Reproduction of Saini et al., "Dynamic depth quantum approximate optimization
algorithm for solving constrained shortest path problem", Quantum Mach. Intell. 8:104 (2026).
"""
from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import ConvergenceMonitor
from .cspp import CSPPInstance, generate_instance, solve_classical
from .ddqaoa import RunResult, run_ddqaoa, run_fixed_qaoa
from .qubo import cspp_to_qubo, decode, qubo_to_ising
from .transfer import interp_transfer


def build_problem(inst: CSPPInstance, normalize: bool = True, **qubo_kwargs):
    """CSPP instance -> (QUBO, QAOAProblem)."""
    q = cspp_to_qubo(inst, **qubo_kwargs)
    return q, QAOAProblem.from_ising(qubo_to_ising(q), normalize=normalize)


__all__ = [
    "CSPPInstance", "generate_instance", "solve_classical",
    "cspp_to_qubo", "qubo_to_ising", "decode", "build_problem",
    "QAOAProblem", "make_qnodes", "cnots_per_layer",
    "ConvergenceMonitor", "interp_transfer",
    "RunResult", "run_ddqaoa", "run_fixed_qaoa",
]
