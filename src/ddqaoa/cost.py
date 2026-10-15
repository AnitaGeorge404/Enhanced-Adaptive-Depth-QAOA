"""Cost accounting and gate ledger for QAOA optimization loops.

Tracks:
- paper_cnots: sum over optimizer steps of CNOTs(p_t) (replicates Saini et al. exactly).
- circuit_executions: energy evaluations + gradient evaluations based on gradient_method:
    - "autodiff": simulation-only, hardware cost = 0 (flagged is_hardware_faithful=False).
    - "parameter_shift": 2 shifts per gate occurrence of shared parameters.
    - "spsa": 2 circuit evaluations per step.
- shots_total & cnot_shot_weighted: execution counts scaled by shots and depth CNOTs.
- hessian_cost: circuit evaluations spent on curvature testing (Feature 2).
- Per-step JSONL logging of optimization trajectories.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

GradientMethod = Literal["autodiff", "parameter_shift", "spsa"]
HessianMode = Literal["block", "full"]


def estimate_hessian_cost(
    n_qubits: int,
    n_zz: int,
    p: int,
    mode: HessianMode = "block",
    n_z: int = 0,
) -> int:
    """Estimate circuit executions required for parameter-shift Hessian evaluation.

    In QAOA, gamma and beta are shared across multiple gates:
    - gamma in layer l appears in (n_z + n_zz) gates (RZ and IsingZZ).
    - beta in layer l appears in n_qubits gates (RX).

    Using the bivariate parameter-shift rule (Mari et al. / Banchi & Crooks):
    d^2 E / (d theta_i d theta_j) requires 4 shifted circuit evaluations for each
    pair of gate occurrences. For diagonal elements d^2 E / d theta_i^2, 4 shifted
    evaluations per occurrence pair (or 3 evaluations with unshifted central point).

    As an upper bound:
    - mode="block": 2x2 Hessian of the newly added layer (gamma_{p+1}, beta_{p+1}).
      Occurrences: N_gamma = n_z + n_zz, N_beta = n_qubits.
      d^2 E / d gamma^2: <= 4 * N_gamma * N_gamma
      d^2 E / d beta^2:  <= 4 * N_beta * N_beta
      d^2 E / (d gamma d beta): <= 4 * N_gamma * N_beta
      Total block cost = 4 * (N_gamma + N_beta)^2.
    - mode="full": full 2p x 2p Hessian across all layers.
      Total occurrences across all p layers: M = p * (N_gamma + N_beta).
      Total full cost = 4 * M^2.
    """
    n_gamma = n_z + n_zz
    n_beta = n_qubits
    if mode == "block":
        total_occurrences = n_gamma + n_beta
        return 4 * (total_occurrences**2)
    elif mode == "full":
        total_occurrences = p * (n_gamma + n_beta)
        return 4 * (total_occurrences**2)
    else:
        raise ValueError(f"Unknown Hessian mode: {mode}")


@dataclass
class StepSnapshot:
    """Snapshot of a single optimization step."""
    step: int
    depth: int
    energy: float
    grad_norm: float | None = None
    curvature_info: dict[str, Any] | None = None
    paper_cnots_cum: int = 0
    circuit_executions_cum: int = 0
    shots_cum: int | None = None
    cnot_shot_weighted_cum: float | None = None
    hessian_cost_cum: int = 0


@dataclass
class CostLedger:
    """Threaded counter tracking quantum and classical resource consumption."""

    gradient_method: GradientMethod = "autodiff"
    shots: int | None = None  # None indicates exact statevector simulation
    cnots_per_layer: int = 0
    n_qubits: int = 0
    n_zz: int = 0
    n_z: int = 0

    # Primary accounting metrics
    paper_cnots: int = 0
    circuit_executions: int = 0
    shots_total: int | None = None
    cnot_shot_weighted: float | None = None
    hessian_cost: int = 0
    is_hardware_faithful: bool = True

    # Detailed step logs
    history: list[StepSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.gradient_method == "autodiff":
            self.is_hardware_faithful = False
        if self.shots is not None:
            self.shots_total = 0
            self.cnot_shot_weighted = 0.0

    def record_step(
        self,
        step: int,
        depth: int,
        energy: float,
        grad_norm: float | None = None,
        curvature_info: dict[str, Any] | None = None,
        evaluations_forward: int = 1,
    ) -> None:
        """Record an optimization step and advance all monotonic resource counters."""
        step_cnots = 2 * depth * self.n_zz if self.cnots_per_layer == 0 else depth * self.cnots_per_layer
        self.paper_cnots += step_cnots

        # Gradient executions based on method
        if self.gradient_method == "autodiff":
            grad_execs = 0
        elif self.gradient_method == "spsa":
            grad_execs = 2
        elif self.gradient_method == "parameter_shift":
            # 2 shifts per gate occurrence of shared parameters in depth p
            n_gamma = self.n_z + self.n_zz
            n_beta = self.n_qubits
            occurrences_per_layer = n_gamma + n_beta
            total_occurrences = depth * occurrences_per_layer
            grad_execs = 2 * total_occurrences
        else:
            raise ValueError(f"Unknown gradient method: {self.gradient_method}")

        step_execs = evaluations_forward + grad_execs
        self.circuit_executions += step_execs

        if self.shots is not None:
            step_shots = step_execs * self.shots
            assert self.shots_total is not None
            assert self.cnot_shot_weighted is not None
            self.shots_total += step_shots
            self.cnot_shot_weighted += float(step_shots * step_cnots)

        snapshot = StepSnapshot(
            step=step,
            depth=depth,
            energy=energy,
            grad_norm=grad_norm,
            curvature_info=curvature_info,
            paper_cnots_cum=self.paper_cnots,
            circuit_executions_cum=self.circuit_executions,
            shots_cum=self.shots_total,
            cnot_shot_weighted_cum=self.cnot_shot_weighted,
            hessian_cost_cum=self.hessian_cost,
        )
        self.history.append(snapshot)

    def record_hessian_evaluations(self, count: int) -> None:
        """Record circuit evaluations expended on Hessian / curvature analysis."""
        if count < 0:
            raise ValueError("Hessian evaluation count must be non-negative.")
        self.hessian_cost += count
        self.circuit_executions += count
        if self.shots is not None:
            assert self.shots_total is not None
            self.shots_total += count * self.shots

    def to_jsonl(self, filepath: str | Path) -> None:
        """Write step trajectory to a JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for snap in self.history:
                f.write(json.dumps(asdict(snap)) + "\n")

    def summary(self) -> dict[str, Any]:
        """Summary dictionary of ledger state."""
        return {
            "paper_cnots": self.paper_cnots,
            "circuit_executions": self.circuit_executions,
            "gradient_method": self.gradient_method,
            "is_hardware_faithful": self.is_hardware_faithful,
            "shots": self.shots,
            "shots_total": self.shots_total,
            "cnot_shot_weighted": self.cnot_shot_weighted,
            "hessian_cost": self.hessian_cost,
            "total_steps": len(self.history),
        }
