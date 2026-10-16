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
| [src/ddqaoa/noise.py](src/ddqaoa/noise.py) | NISQ noise modeling (depolarizing channels, Readout Error Mitigation - REM, shots $S=1024$) |
| [src/ddqaoa/scaling.py](src/ddqaoa/scaling.py) | Large graph scaling ($n \ge 15$ nodes / 20+ qubits): chunked streaming, matrix-free eval, heuristic solver |
| [src/ddqaoa/simulator.py](src/ddqaoa/simulator.py) | Analytical Matrix-Free Simulator with exact Adjoint State gradient (Feature 10) |
| [src/ddqaoa/warm_start.py](src/ddqaoa/warm_start.py) | Classical LP / Continuous Relaxation Warm-Starting (Feature 2) |
| [src/ddqaoa/maxcut.py](src/ddqaoa/maxcut.py) | Multi-Problem Adapter Suite: Max-Cut benchmarks (Feature 8) |
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
    generate_maxcut_instance,
    build_maxcut_problem,
    get_warm_start_angles,
    evaluate_under_noise,
    NoiseConfig,
)

inst = generate_instance(n_nodes=4, seed=0)
qubo, problem = build_problem(inst)
print("Exact classical path:", solve_classical(inst))

# 1. Original Paper Baseline DDQAOA
baseline = run_ddqaoa(problem, p_max=6, max_steps=400, stepsize=0.03, k=30)
print(f"Baseline: r={baseline.approx_ratio:.4f}, cumCNOT={baseline.cumulative_cnots:,}")

# 2. Enhanced DDQAOA with Matrix-Free Simulator (Feature 10: 10x-100x speedup)
enhanced = run_enhanced_ddqaoa(problem, p_max=6, max_steps=400, stepsize=0.03, k=30, device="analytical")
print(f"Enhanced (Matrix-Free): r={enhanced.approx_ratio:.4f}, cumCNOT={enhanced.cumulative_cnots:,}")

# 3. LP Continuous Relaxation Warm-Starting (Feature 2)
theta_ws = get_warm_start_angles(inst, eps=0.05)
warm_res = run_enhanced_ddqaoa(problem, p_max=4, max_steps=200, warm_start_angles=theta_ws, device="analytical")
print(f"LP Warm-Started: r={warm_res.approx_ratio:.4f}")

# 4. Multi-Problem Generalization: Max-Cut (Feature 8)
mc_inst = generate_maxcut_instance(n_nodes=8, graph_type="regular", d=3, seed=42)
_, mc_problem = build_maxcut_problem(mc_inst)
mc_res = run_enhanced_ddqaoa(mc_problem, p_max=4, max_steps=100, device="analytical")
print(f"Max-Cut DDQAOA: r={mc_res.approx_ratio:.4f}")

# 5. Readout Error Mitigation (Feature 5: REM)
noise = NoiseConfig(p_single=0.001, p_two=0.01, p_readout_01=0.03, p_readout_10=0.03, shots=2000)
noisy_eval = evaluate_under_noise(enhanced.gammas, enhanced.betas, problem, noise, apply_rem=True)
print(f"Noisy ratio: {noisy_eval.noisy_approx_ratio:.4f} -> REM Mitigated: {noisy_eval.mitigated_approx_ratio:.4f}")
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

