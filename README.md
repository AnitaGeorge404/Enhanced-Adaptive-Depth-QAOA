# DDQAOA for the Constrained Shortest Path Problem

Course project for **Quantum Computing for Engineers**. This repo reproduces and extends:

> R. Saini, N. Mohamed, S. Al-Kuwari, A. Farouk, *Dynamic depth quantum approximate
> optimization algorithm for solving constrained shortest path problem*,
> Quantum Machine Intelligence 8:104 (2026). [doi:10.1007/s42484-026-00442-0](https://doi.org/10.1007/s42484-026-00442-0)
> (PDF: [`s42484-026-00442-0.pdf`](s42484-026-00442-0.pdf))

**Idea of the paper.** Standard QAOA needs the circuit depth `p` chosen in advance. DDQAOA
starts at `p = 1`, optimizes until the energy converges, then adds a layer and
warm-starts the new parameters by interpolating the old ones. It stops at `p_max`.
On CSPP instances this gives the same or better approximation ratio than fixed `p = 10/15`
while using 2–4× fewer cumulative CNOTs.

## Pipeline

```
CSPPInstance ──► QUBO (Eqs. 12–16) ──► Ising (Eq. 17) ──► QAOA circuit (Eq. 1)
  cspp.py           qubo.py               qubo.py           circuits.py
                                                               │
         ┌──────────────── ddqaoa.py (Algorithm 1) ────────────┤
         │  Adam step ─► ConvergenceMonitor (Eqs. 18–19) ──yes─► interp_transfer (Eqs. 20–22), p ← p+1
         └──────────────── run_fixed_qaoa (baselines p = 3, 5, 10, 15)
                                                               │
                                  metrics.py: approx. ratio, success prob., cumulative CNOTs
```

| File | Paper section | Contents |
|---|---|---|
| [src/ddqaoa/cspp.py](src/ddqaoa/cspp.py) | 2.2 | Instance dataclass, random generator, exact classical solver (enumeration) |
| [src/ddqaoa/qubo.py](src/ddqaoa/qubo.py) | 2.2.1–2.2.2 | CSPP → QUBO with flow and resource penalties + slack register; QUBO → Ising |
| [src/ddqaoa/circuits.py](src/ddqaoa/circuits.py) | 2.1 | PennyLane QAOA ansatz, energy diagonal, CNOT count per layer |
| [src/ddqaoa/convergence.py](src/ddqaoa/convergence.py) | 3, Eqs. 18–19 | Plateau + variance convergence test |
| [src/ddqaoa/transfer.py](src/ddqaoa/transfer.py) | 3, Eqs. 20–22 | `p=1→2` (×1.2 / ×0.8) and linear/cubic interpolation |
| [src/ddqaoa/ddqaoa.py](src/ddqaoa/ddqaoa.py) | Alg. 1, 4.1 | `run_ddqaoa` and the `run_fixed_qaoa` baseline |
| [src/ddqaoa/metrics.py](src/ddqaoa/metrics.py) | 4.2, 5.2 | Approximation ratio, success probability, cumulative CNOTs |
| [experiments/run_benchmark.py](experiments/run_benchmark.py) | 4–5 | DDQAOA vs fixed-depth benchmark → JSON |
| [experiments/plot_results.py](experiments/plot_results.py) | Figs. 2–5, Tables 1–2 | Box plots, trajectories, summary table |
| [docs/EXTENSIONS.md](docs/EXTENSIONS.md) | 6 | Candidate extensions + where they plug in |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

```python
from ddqaoa import generate_instance, build_problem, solve_classical, run_ddqaoa, run_fixed_qaoa

inst = generate_instance(n_nodes=4, seed=0)
qubo, problem = build_problem(inst)            # problem.n_qubits, problem.e_min, ...
print(solve_classical(inst))                   # exact answer for reference

dd = run_ddqaoa(problem, p_max=10, max_steps=1200)
fx = run_fixed_qaoa(problem, p=10, steps=1200)
print(dd.approx_ratio, dd.success_prob, dd.final_depth, dd.cumulative_cnots)
```

Benchmark (quick smoke test, then a paper-like run):

```bash
python experiments/run_benchmark.py --n-instances 2 --steps 400 --p-list 3 5 --p-max 6 --stepsize 0.05 --k 30 --out results/smoke.json
python experiments/plot_results.py results/smoke.json
python experiments/run_benchmark.py --n-nodes 4 --n-slack 3 --n-instances 100 --steps 1200
```

## Implementation choices (things the paper leaves open)

- **Slack register.** Eq. 14 is written as an equality penalty. We add binary slack bits
  `Σ r_ij x_ij + Σ 2^k s_k = r_limit` so it encodes the inequality of Eq. 10. The paper
  mentions a slack register in Sec. 4.1.
- **Pruning.** By default, edges into the source and out of the target are removed, since
  Eqs. 7–8 force them to 0. With this, a complete 4-node graph has 7 edge qubits plus
  slack: `--n-slack 3` gives the paper's 10 qubits. Use `generate_instance(prune=False)` to keep all edges.
- **Penalty weights.** By default `λ = ρ = 1 + Σ c_ij`. The Hamiltonian is then divided by
  its largest coefficient (`normalize=True`), so the same `γ` initialisation and learning rate work on every instance.
- **Approximation ratio.** We use `r = (E_max − ⟨H⟩)/(E_max − E_min)`, which lies in [0, 1]
  and doesn't depend on the constant offset.
- **Cost evaluation.** `H_C` is diagonal, so `⟨H_C⟩ = probs · diag(H_C)` is exact and gives
  the success probability from the same simulation.
- **Convergence history** is reset at every depth increase, so the variance window only
  covers the current depth. `E_best` carries over, as in Alg. 1.

## Known limitations / TODO

- **Speed.** PennyLane `default.qubit` with backprop takes roughly 0.1–0.3 s per step at 10 qubits.
  The full 100-instance benchmark will take a long time. See the simulator speed-up in `docs/EXTENSIONS.md`.
- With complete graphs and costs in [1, 10], the direct `s→t` edge is often optimal. Use
  `edge_prob < 1` or other cost ranges for harder instances.
- The paper doesn't give the initial parameters for fixed-depth QAOA. We use U[0,1) (`init="random"`), and `"ramp"` is also available.
