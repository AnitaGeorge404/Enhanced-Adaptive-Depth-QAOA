"""Resource-aware adaptive depth selection for DDQAOA (Improvement 3).

Guides depth growth by expected solution-quality improvement relative to
quantum circuit resources (CNOT gate counts and decoherence risk).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResourceDecision:
    """Outcome of resource-aware evaluation at a depth transition."""
    stop: bool
    reason: str
    marginal_ratio_gain: float = 0.0
    marginal_cnots: int = 0
    efficiency: float = 0.0


@dataclass
class ResourceBudget:
    """Resource-aware depth controller.

    Attributes:
        max_cnots: Hard upper limit on cumulative CNOT gates across all optimization steps.
        min_marginal_gain: Minimum required improvement in approximation ratio (delta r)
            when stepping to a new depth layer.
        patience_layers: Number of consecutive depth transitions with marginal gain < min_marginal_gain
            before terminating depth expansion.
        target_approx_ratio: Target approximation ratio (e.g. 0.99); once achieved, further
            depth expansion stops to save resources.
        target_success_prob: Target success probability; stops further depth expansion when met.
        min_depth: Minimum depth layer to reach before resource early-stopping can be triggered.
    """
    max_cnots: int | None = None
    min_marginal_gain: float = 1e-3
    patience_layers: int = 1
    target_approx_ratio: float | None = None
    target_success_prob: float | None = None
    min_depth: int = 1

    stagnant_layers: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False)

    def evaluate_transition(
        self,
        current_depth: int,
        approx_ratio_current: float,
        approx_ratio_prev: float | None,
        success_prob_current: float,
        cumulative_cnots_used: int,
        cnots_per_layer: int,
    ) -> ResourceDecision:
        """Evaluate whether adding another depth layer is justified."""
        # 1. Target quality reached
        if self.target_approx_ratio is not None and approx_ratio_current >= self.target_approx_ratio:
            return ResourceDecision(
                stop=True,
                reason=f"Target approximation ratio ({self.target_approx_ratio:.4f}) reached at depth {current_depth}",
                marginal_ratio_gain=0.0,
            )

        if self.target_success_prob is not None and success_prob_current >= self.target_success_prob:
            return ResourceDecision(
                stop=True,
                reason=f"Target success probability ({self.target_success_prob:.4f}) reached at depth {current_depth}",
                marginal_ratio_gain=0.0,
            )

        # 2. Hard CNOT budget check: if adding next layer would exceed max_cnots
        if self.max_cnots is not None:
            if cumulative_cnots_used >= self.max_cnots:
                return ResourceDecision(
                    stop=True,
                    reason=f"Cumulative CNOT budget ({self.max_cnots:,}) reached ({cumulative_cnots_used:,} CNOTs)",
                )

        # Guard against stopping before min_depth
        if current_depth < self.min_depth:
            return ResourceDecision(stop=False, reason="Below minimum depth")

        # 3. Marginal gain check relative to previous depth
        if approx_ratio_prev is not None:
            marginal_gain = approx_ratio_current - approx_ratio_prev
            marginal_cnot_step = (current_depth + 1) * cnots_per_layer
            efficiency = marginal_gain / max(marginal_cnot_step, 1)

            self.history.append({
                "depth": current_depth,
                "approx_ratio": approx_ratio_current,
                "marginal_gain": marginal_gain,
                "cumulative_cnots": cumulative_cnots_used,
            })

            if marginal_gain < self.min_marginal_gain:
                self.stagnant_layers += 1
                if self.stagnant_layers >= self.patience_layers:
                    return ResourceDecision(
                        stop=True,
                        reason=(
                            f"Diminishing returns: marginal gain ({marginal_gain:+.4e}) "
                            f"< threshold ({self.min_marginal_gain:.4e}) for {self.stagnant_layers} layer(s)"
                        ),
                        marginal_ratio_gain=marginal_gain,
                        marginal_cnots=marginal_cnot_step,
                        efficiency=efficiency,
                    )
            else:
                self.stagnant_layers = 0

            return ResourceDecision(
                stop=False,
                reason="Adequate marginal gain",
                marginal_ratio_gain=marginal_gain,
                marginal_cnots=marginal_cnot_step,
                efficiency=efficiency,
            )

        return ResourceDecision(stop=False, reason="First evaluated depth")
