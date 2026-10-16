"""DDQAOA for the Constrained Shortest Path Problem.

Reproduction of Saini et al., "Dynamic depth quantum approximate optimization
algorithm for solving constrained shortest path problem", Quantum Mach. Intell. 8:104 (2026).
"""
from .circuits import QAOAProblem, cnots_per_layer, make_qnodes
from .convergence import ConvergenceMonitor, CurvatureDecision, CurvatureTrigger, EnhancedConvergenceMonitor
from .cost import CostLedger, estimate_hessian_cost
from .cspp import CSPPInstance, generate_instance, solve_classical
from .metrics import (
    approximation_ratio,
    best_of_s_gap,
    build_feasible_mask,
    cumulative_cnots,
    decode_path,
    feasible_conditioned_ratio,
    is_feasible,
    postselected_energy,
    success_probability,
    time_to_solution_99,
)
from .noise_stopping import FidelityModel, NoiseAwareStopper, run_noise_aware_ddqaoa
from .ddqaoa import RunResult, run_ddqaoa, run_enhanced_ddqaoa, run_fixed_qaoa
from .maxcut import MaxCutInstance, build_maxcut_problem, generate_maxcut_instance, maxcut_to_ising
from .optimizer import ControlledAdamOptimizer, OptimizerConfig
from .qubo import cspp_to_qubo, decode, qubo_to_ising
from .reduction import ReductionResult, reduce_instance
from .resource import ResourceBudget, ResourceDecision
from .simulator import AnalyticalSimulator, make_analytical_qnodes
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
from .warm_start import (
    get_warm_start_angles,
    probabilities_to_angles,
    solve_cspp_lp_relaxation,
    solve_qubo_continuous_relaxation,
)


from .noise import (
    NoiseConfig,
    NoiseEvaluationResult,
    apply_readout_mitigation,
    apply_readout_noise,
    evaluate_under_noise,
    make_noisy_qnodes,
    project_to_simplex,
)
from .scaling import (
    evaluate_bitstring_energy,
    evaluate_bitstrings_batch,
    find_sparse_ground_state,
    solve_classical_heuristic,
    stream_ising_diagonal,
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
    "MaxCutInstance", "generate_maxcut_instance", "maxcut_to_ising", "build_maxcut_problem",
    "CostLedger", "estimate_hessian_cost",
    "approximation_ratio", "success_probability", "cumulative_cnots",
    "decode_path", "is_feasible", "postselected_energy", "time_to_solution_99",
    "feasible_conditioned_ratio", "build_feasible_mask", "best_of_s_gap",
    "ConvergenceMonitor", "EnhancedConvergenceMonitor",
    "CurvatureTrigger", "CurvatureDecision",
    "interp_transfer", "adaptive_transfer", "fourier_transfer", "zero_pad_displace", "AdaptiveTransfer",
    "solve_cspp_lp_relaxation", "solve_qubo_continuous_relaxation", "probabilities_to_angles", "get_warm_start_angles",
    "ResourceBudget", "ResourceDecision",
    "ControlledAdamOptimizer", "OptimizerConfig",
    "RunResult", "run_ddqaoa", "run_enhanced_ddqaoa", "run_fixed_qaoa",
    "FidelityModel", "NoiseAwareStopper", "make_noisy_qnodes", "run_noise_aware_ddqaoa",
    "paired_bootstrap_ci", "paired_statistical_tests", "aggregate_benchmark_results",
    "NoiseConfig", "NoiseEvaluationResult", "evaluate_under_noise", "make_noisy_qnodes",
    "apply_readout_noise", "apply_readout_mitigation", "project_to_simplex",
    "AnalyticalSimulator", "make_analytical_qnodes",
    "evaluate_bitstring_energy", "evaluate_bitstrings_batch",
    "stream_ising_diagonal", "find_sparse_ground_state", "solve_classical_heuristic",
]

