"""Multi-Problem Adapter Suite: Max-Cut Problem (Feature 8).

Generalizes the Dynamic Depth QAOA engine beyond the Constrained Shortest Path Problem (CSPP)
to standard graph combinatorial optimization benchmarks, demonstrating algorithmic generality.

Problem Formulation:
Given an undirected graph G = (V, E) with weights w_{uv} >= 0:
    Max-Cut(x) = sum_{(u, v) in E} w_{uv} (x_u (1 - x_v) + x_v (1 - x_u))
Substituting x_i = (1 - z_i) / 2 into the objective:
    Cut(z) = sum_{(u, v) in E} w_{uv} (1 - z_u z_v) / 2

To minimize H_C = -Cut(z) in standard QAOA:
    H_C = (1/2) * sum_{(u, v) in E} w_{uv} Z_u Z_v - (1/2) * sum_{(u, v) in E} w_{uv}
with zero linear fields (h_i = 0) and purely quadratic Ising interactions J_{uv} = w_{uv} / 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from .circuits import QAOAProblem
from .qubo import Ising

GraphType = Literal["erdos_renyi", "regular", "cycle", "complete"]


@dataclass
class MaxCutInstance:
    """Graph instance for the Maximum Cut problem."""
    n_nodes: int
    edges: list[tuple[int, int]]
    weights: dict[tuple[int, int], float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        # Normalize edge ordering: (min(u, v), max(u, v))
        norm_edges = []
        norm_weights = {}
        for (u, v) in self.edges:
            edge = (min(u, v), max(u, v))
            norm_edges.append(edge)
            norm_weights[edge] = float(self.weights.get((u, v), self.weights.get((v, u), 1.0)))
        self.edges = sorted(list(set(norm_edges)))
        self.weights = norm_weights

    def cut_value(self, bits: np.ndarray | list[int]) -> float:
        """Evaluate cut size for bitstring x in {0, 1}^n."""
        b = np.asarray(bits, dtype=int)
        val = 0.0
        for (u, v) in self.edges:
            if b[u] != b[v]:
                val += self.weights.get((u, v), 1.0)
        return val

    def solve_classical(self) -> tuple[np.ndarray, float]:
        """Exact maximum cut by exhaustive state enumeration (up to n ~ 20)."""
        n = self.n_nodes
        best_cut = -1.0
        best_bits = np.zeros(n, dtype=int)

        # Symmetry: fix vertex 0 to 0
        for idx in range(1 << (n - 1)):
            bits = np.array([(idx >> (n - 2 - k)) & 1 if k >= 0 else 0 for k in range(-1, n - 1)])
            bits[0] = 0
            val = self.cut_value(bits)
            if val > best_cut:
                best_cut = val
                best_bits = bits.copy()

        return best_bits, best_cut


def generate_maxcut_instance(
    n_nodes: int = 6,
    edge_prob: float = 0.5,
    seed: int | None = None,
    graph_type: GraphType = "erdos_renyi",
    d: int = 3,
    weighted: bool = False,
    weight_range: tuple[float, float] = (1.0, 5.0),
) -> MaxCutInstance:
    """Generate random graph benchmark instances for Max-Cut.

    graph_type:
        - "erdos_renyi": G(n, p) random graph.
        - "regular": d-regular graph (standard QAOA benchmark, e.g. 3-regular).
        - "cycle": ring graph C_n.
        - "complete": complete graph K_n.
    """
    rng = np.random.default_rng(seed)
    edges = []

    if graph_type == "complete":
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                edges.append((u, v))

    elif graph_type == "cycle":
        for u in range(n_nodes):
            edges.append((u, (u + 1) % n_nodes))

    elif graph_type == "regular":
        # Generate random d-regular graph
        import networkx as nx
        G = nx.random_regular_graph(d=min(d, n_nodes - 1), n=n_nodes, seed=seed)
        edges = list(G.edges())

    else:  # "erdos_renyi"
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if rng.random() <= edge_prob:
                    edges.append((u, v))

        # Ensure graph has at least one connected component
        if not edges and n_nodes > 1:
            edges.append((0, 1))

    weights = {}
    for e in edges:
        if weighted:
            w = float(rng.uniform(weight_range[0], weight_range[1]))
        else:
            w = 1.0
        weights[e] = w

    return MaxCutInstance(
        n_nodes=n_nodes,
        edges=edges,
        weights=weights,
        meta={"graph_type": graph_type, "seed": seed, "d": d, "weighted": weighted},
    )


def maxcut_to_ising(inst: MaxCutInstance) -> Ising:
    """Map MaxCutInstance to an Ising Hamiltonian H_C = (1/2) * sum J_{uv} Z_u Z_v - offset.

    Minimizing H_C maximizes the cut value:
        Min Energy E = -Max Cut.
    """
    n = inst.n_nodes
    h = {i: 0.0 for i in range(n)}
    J = {}
    total_w = 0.0

    for (u, v) in inst.edges:
        w = inst.weights.get((u, v), 1.0)
        total_w += w
        edge = (min(u, v), max(u, v))
        # H_C = sum w/2 Z_u Z_v - sum w/2
        J[edge] = 0.5 * w

    offset = -0.5 * total_w
    return Ising(n_qubits=n, h={k: v for k, v in h.items() if abs(v) > 1e-12}, J=J, offset=offset)


def build_maxcut_problem(
    inst: MaxCutInstance,
    normalize: bool = True,
) -> tuple[MaxCutInstance, QAOAProblem]:
    """Build a QAOAProblem ready for DDQAOA from a MaxCutInstance."""
    ising = maxcut_to_ising(inst)
    problem = QAOAProblem.from_ising(ising, normalize=normalize)
    return inst, problem
