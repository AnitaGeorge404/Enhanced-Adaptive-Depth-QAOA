"""Stronger optimization control and parameter stability (Improvement 4).

Features:
1. Gradient norm clipping to prevent parameter divergence and instability on steep cost landscapes.
2. Depth-aware learning rate scheduling (lr_p = lr_0 / sqrt(1 + depth_decay * (p - 1))).
3. Safe recovery from non-finite (NaN / Inf) gradients and energy evaluations.
4. Tracking of gradient norms for convergence and landscape diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml


@dataclass
class OptimizerConfig:
    """Configuration for enhanced classical optimization in QAOA."""
    stepsize: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    depth_decay: float = 0.1
    recover_nan: bool = True


class ControlledAdamOptimizer(qml.AdamOptimizer):
    """Adam optimizer with gradient clipping, depth-adaptive stepsize, and numerical safety."""

    def __init__(
        self,
        stepsize: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float | None = 1.0,
        depth_decay: float = 0.1,
        recover_nan: bool = True,
    ) -> None:
        super().__init__(stepsize=stepsize, beta1=beta1, beta2=beta2, eps=eps)
        self.base_stepsize = float(stepsize)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.depth_decay = float(depth_decay)
        self.recover_nan = bool(recover_nan)
        self.current_depth = 1
        self.grad_norm_history: list[float] = []
        self.last_grad_norm: float = 0.0

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "ControlledAdamOptimizer":
        return cls(
            stepsize=config.stepsize,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            grad_clip=config.grad_clip,
            depth_decay=config.depth_decay,
            recover_nan=config.recover_nan,
        )

    def set_depth(self, p: int) -> None:
        """Scale stepsize adaptively based on circuit depth p."""
        self.current_depth = int(p)
        if self.depth_decay > 0.0 and self.current_depth >= 1:
            scale = 1.0 / math.sqrt(1.0 + self.depth_decay * (self.current_depth - 1))
            self.stepsize = self.base_stepsize * scale
        else:
            self.stepsize = self.base_stepsize

    def compute_grad(self, objective_fn, args, kwargs, grad_fn=None):
        """Compute gradients with optional norm clipping and NaN recovery."""
        grad, forward = super().compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)

        # Check for non-finite values in forward pass or gradient
        if self.recover_nan:
            has_nan = False
            if forward is not None and not math.isfinite(float(forward)):
                has_nan = True
            for g in grad:
                if not np.all(np.isfinite(np.asarray(g))):
                    has_nan = True
                    break
            if has_nan:
                # Return zero gradients to preserve current parameters safely
                zero_grad = tuple(np.zeros_like(np.asarray(g)) for g in grad)
                self.last_grad_norm = 0.0
                self.grad_norm_history.append(0.0)
                return zero_grad, forward

        # Compute gradient norm across all parameter arrays
        sq_sum = sum(float(np.sum(np.asarray(g, dtype=float) ** 2)) for g in grad)
        total_norm = math.sqrt(sq_sum)
        self.last_grad_norm = total_norm
        self.grad_norm_history.append(total_norm)

        # Apply gradient norm clipping if configured
        if self.grad_clip is not None and total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-12)
            clipped_grad = tuple(g * scale for g in grad)
            return clipped_grad, forward

        return grad, forward
