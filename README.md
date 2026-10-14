# Enhanced Adaptive-Depth QAOA for the Constrained Shortest Path Problem

Course and research project for **Quantum Computing for Engineers**. This repository reproduces and substantially improves:

> R. Saini, N. Mohamed, S. Al-Kuwari, A. Farouk, *Dynamic depth quantum approximate
> optimization algorithm for solving constrained shortest path problem*,
> Quantum Machine Intelligence 8:104 (2026). [doi:10.1007/s42484-026-00442-0](https://doi.org/10.1007/s42484-026-00442-0)
> (PDF: [`s42484-026-00442-0.pdf`](s42484-026-00442-0.pdf))

---

## 5 Substantial Research Improvements

Building on Saini et al. (2026), this repository implements five complementary algorithmic and scientific improvements:

1. **Robust Multi-Criteria Convergence Detection (`EnhancedConvergenceMonitor`):**
   - Enforces a minimum exploration budget per depth (`min_steps`) to prevent premature layer growth.
   - Dual absolute ($\varepsilon_{\text{abs}}$) and relative ($\varepsilon_{\text{rel}}$) improvement thresholds for scale-invariance.
   - Per-depth reference tracking (`depth_best`) avoiding false plateau counts upon layer transition.
   - Rolling regression slope guard ($\beta_1 < -\tau_{\text{slope}}$) to inhibit stopping while energy is actively descending.

2. **Adaptive Parameter Transfer with Boundary Preservation (`AdaptiveTransfer`):**
   - Action-preserving adiabatic split for $p=1 \to 2$ avoiding arbitrary hardcoded factors.
   - Monotonic PCHIP interpolation eliminating boundary overshoot artifacts common to cubic splines.
   - Continuous FOURIER-basis parameter transfer (Zhou et al., PRX 2020) projecting discrete angles onto smooth spectral modes.
   - Physical angle domain clamping ($\gamma \in [-\pi, \pi]$, $\beta \in [-\pi/2, \pi/2]$).

3. **Resource-Aware Adaptive Depth Selection (`ResourceBudget`):**
   - Evaluates marginal solution quality gain ($\Delta r_p / \Delta \text{CNOT}_p$) at each depth transition.
   - Early stopping on diminishing returns when $\Delta r_p < \tau_{\text{gain}}$ over a patience window.
   - Target approximation ratio thresholding and cumulative CNOT hard budget limit.

4. **Stronger Optimization Control and Parameter Stability (`ControlledAdamOptimizer`):**
   - Euclidean gradient norm clipping preventing parameter divergence on steep cost landscapes.
   - Depth-adaptive step-size scaling ($\alpha(p) = \alpha_0 / \sqrt{1 + \kappa(p-1)}$).
   - Safe numerical recovery from non-finite (NaN / Inf) gradients.

5. **Rigorous Benchmarking & Statistical Validation Pipeline (`statistics.py`):**
   - Paired Student's t-test and Wilcoxon signed-rank test.
   - 2,000-resample paired bootstrap 95% confidence intervals for mean and median differences.
   - Decoupled instance generation seeds from optimization seeds.
   - Automated ablation suite evaluating individual component contributions.
   - Quality-versus-resource Pareto frontier curves.

Detailed mathematical derivations and design rationale are available in [`docs/RESEARCH_IMPROVEMENTS.md`](docs/RESEARCH_IMPROVEMENTS.md).

---

## Pipeline

```
CSPPInstance ──► QUBO (Eqs. 12–16) ──► Ising (Eq. 17) ──► QAOA circuit (Eq. 1)
  cspp.py           qubo.py               qubo.py           circuits.py
                                                               │
         ┌──────────────── ddqaoa.py ──────────────────────────┤
         │  Controlled Adam step (optimizer.py)                │
         │       │                                             │
         │  EnhancedConvergenceMonitor (convergence.py)        │
         │       │ (converged)                                 │
         │  ResourceBudget: marginal gain vs CNOT? ──no───────► Final Result
         │       │ (yes)                                       │
         │  AdaptiveTransfer (PCHIP / FOURIER, transfer.py)    │
         │       │ p ◄─ p + 1                                  │
         │       └─────────────────────────────────────────────┘
         │
         ├──────────────── run_ddqaoa (original paper baseline)
         └──────────────── run_fixed_qaoa (baselines p = 3, 5, 10, 15)
                                                               │
                                  statistics.py: paired bootstrap CIs, Wilcoxon test, Pareto curves
```

| File | Purpose |
|---|---|
| [src/ddqaoa/cspp.py](src/ddqaoa/cspp.py) | CSPP instance dataclass, random generator, exact classical solver |
| [src/ddqaoa/qubo.py](src/ddqaoa/qubo.py) | CSPP → QUBO with flow/resource penalties + slack register; QUBO → Ising |
| [src/ddqaoa/circuits.py](src/ddqaoa/circuits.py) | PennyLane QAOA ansatz, diagonal cost evaluation, CNOT counting |
| [src/ddqaoa/convergence.py](src/ddqaoa/convergence.py) | Original `ConvergenceMonitor` + `EnhancedConvergenceMonitor` (Improvement 1) |
| [src/ddqaoa/transfer.py](src/ddqaoa/transfer.py) | Original `interp_transfer` + `adaptive_transfer`, `fourier_transfer`, `AdaptiveTransfer` (Improvement 2) |
| [src/ddqaoa/resource.py](src/ddqaoa/resource.py) | `ResourceBudget` and `ResourceDecision` (Improvement 3) |
| [src/ddqaoa/optimizer.py](src/ddqaoa/optimizer.py) | `ControlledAdamOptimizer` and `OptimizerConfig` (Improvement 4) |
| [src/ddqaoa/statistics.py](src/ddqaoa/statistics.py) | Paired t-tests, Wilcoxon tests, bootstrap CIs, summary aggregation (Improvement 5) |
| [src/ddqaoa/ddqaoa.py](src/ddqaoa/ddqaoa.py) | Baseline `run_ddqaoa`, `run_enhanced_ddqaoa`, and `run_fixed_qaoa` |
| [src/ddqaoa/metrics.py](src/ddqaoa/metrics.py) | Approximation ratio, success probability, cumulative CNOTs |
| [experiments/run_benchmark.py](experiments/run_benchmark.py) | Paired benchmarking with ablations and statistical output |
| [experiments/plot_results.py](experiments/plot_results.py) | Pareto curves, boxplots, trajectories, ablation charts, markdown reports |
| [docs/RESEARCH_IMPROVEMENTS.md](docs/RESEARCH_IMPROVEMENTS.md) | Full theoretical and empirical research documentation |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

---

## Usage

```python
from ddqaoa import (
    generate_instance,
    build_problem,
    solve_classical,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_fixed_qaoa,
)

inst = generate_instance(n_nodes=4, seed=0)
qubo, problem = build_problem(inst)
print("Exact classical path:", solve_classical(inst))

# 1. Original Paper Baseline DDQAOA
baseline = run_ddqaoa(problem, p_max=6, max_steps=400, stepsize=0.03, k=30)
print(f"Baseline: r={baseline.approx_ratio:.4f}, cumCNOT={baseline.cumulative_cnots:,}")

# 2. Enhanced DDQAOA (Improvements 1-4)
enhanced = run_enhanced_ddqaoa(problem, p_max=6, max_steps=400, stepsize=0.03, k=30)
print(f"Enhanced: r={enhanced.approx_ratio:.4f}, cumCNOT={enhanced.cumulative_cnots:,}")

# 3. Fixed-Depth Baseline
fixed = run_fixed_qaoa(problem, p=3, steps=400, stepsize=0.03, seed=0)
print(f"Fixed p=3: r={fixed.approx_ratio:.4f}, cumCNOT={fixed.cumulative_cnots:,}")
```

---

## Benchmarks & Reproduction

### Quick Smoke Test
```bash
python experiments/run_benchmark.py --smoke --ablations --out results/smoke_research.json
python experiments/plot_results.py results/smoke_research.json
```

### Full Research Benchmark with Ablations
```bash
python experiments/run_benchmark.py --n-instances 5 --steps 300 --p-list 3 5 --p-max 6 --ablations --out results/benchmark_research.json
python experiments/plot_results.py results/benchmark_research.json
```

### Interactive Step-by-Step Demo
```bash
python demo.py
```

