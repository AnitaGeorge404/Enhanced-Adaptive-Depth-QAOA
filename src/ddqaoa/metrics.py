"""Evaluation metrics (paper Sec. 4.2 and 5.2)."""
from __future__ import annotations

import numpy as np

from .circuits import QAOAProblem


def approximation_ratio(energy: float, problem: QAOAProblem) -> float:
    """r = (E_max - <H>) / (E_max - E_min), in [0, 1] with r = 1 at the ground state.

    The paper writes r = <H>/E_min "normalised to [0, 1]"; this min-max form is the
    standard normalisation and does not depend on the Hamiltonian's constant offset.
    """
    return (problem.e_max - energy) / (problem.e_max - problem.e_min)


def success_probability(probs: np.ndarray, problem: QAOAProblem) -> float:
    """Probability of sampling an optimal bitstring."""
    return float(np.sum(np.asarray(probs)[problem.ground_states]))


def cumulative_cnots(depth_per_step: list[int], cnots_per_layer: int) -> int:
    """Sum over optimiser steps of the CNOT count of the circuit used at that step."""
    return int(sum(depth_per_step) * cnots_per_layer)
