"""DDQAOA for the Constrained Shortest Path Problem.

Reproduction of Saini et al., "Dynamic depth quantum approximate optimization
algorithm for solving constrained shortest path problem", Quantum Mach. Intell. 8:104 (2026).
"""
from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import ConvergenceMonitor, CurvatureDecision, CurvatureTrigger, EnhancedConvergenceMonitor
from .cost import CostLedger, estimate_hessian_cost
from .cspp import CSPPInstance, generate_instance, solve_classical
from .ddqaoa import RunResult, run_ddqaoa, run_enhanced_ddqaoa, run_fixed_qaoa
from .optimizer import ControlledAdamOptimizer, OptimizerConfig
from .qubo import cspp_to_qubo, decode, qubo_to_ising
from .reduction import ReductionResult, reduce_instance
from .resource import ResourceBudget, ResourceDecision
from .statistics import (
    aggregate_benchmark_results,
    paired_bootstrap_ci,
    paired_statistical_tests,
)
from .transfer import (
    AdaptiveTransfer,
    adaptive_transfer,
    fourier_transfer,
    interp_transfer,
    zero_pad_displace,
)


def build_problem(inst: CSPPInstance, normalize: bool = True, **qubo_kwargs):
    """CSPP instance -> (QUBO, QAOAProblem)."""
    q = cspp_to_qubo(inst, **qubo_kwargs)
    return q, QAOAProblem.from_ising(qubo_to_ising(q), normalize=normalize)


__all__ = [
    "CSPPInstance", "generate_instance", "solve_classical",
    "cspp_to_qubo", "qubo_to_ising", "decode", "build_problem",
    "reduce_instance", "ReductionResult",
    "QAOAProblem", "make_qnodes", "cnots_per_layer",
    "CostLedger", "estimate_hessian_cost",
    "ConvergenceMonitor", "EnhancedConvergenceMonitor",
    "CurvatureTrigger", "CurvatureDecision",
    "interp_transfer", "adaptive_transfer", "fourier_transfer", "zero_pad_displace", "AdaptiveTransfer",
    "ResourceBudget", "ResourceDecision",
    "ControlledAdamOptimizer", "OptimizerConfig",
    "RunResult", "run_ddqaoa", "run_enhanced_ddqaoa", "run_fixed_qaoa",
    "paired_bootstrap_ci", "paired_statistical_tests", "aggregate_benchmark_results",
]

