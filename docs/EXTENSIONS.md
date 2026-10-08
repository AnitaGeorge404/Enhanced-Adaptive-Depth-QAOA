# Extension ideas

The core code has two pluggable hooks in `run_ddqaoa`:

- `transfer=`: any `(gammas, betas) -> (gammas', betas')` returning p+1 parameters
- `monitor=`: any object with `update(energy) -> bool` and `reset()`

Most of the ideas below only need a new function or class in `src/ddqaoa/extensions/`
plus a flag in `experiments/run_benchmark.py`. Pick one or two. They are roughly ordered from easiest to hardest.

## 1. Alternative parameter-transfer strategies (Sec. 6: "more sophisticated transfer")
- **FOURIER** (Zhou et al. 2020): optimise `u_k, v_k` with
  `γ_i = Σ u_k sin((k−½)(i−½)π/p)`. Growing p then only means appending zero modes.
- **Linear-ramp re-init** at every depth, used as an ablation to show how much the transfer helps.
- Sweep the 1.2 / 0.8 coefficients of Eqs. 20–21.

## 2. Alternative convergence criteria (Sec. 6)
- A gradient-norm threshold `‖∇E‖ < τ` in place of plateau/variance.
- Relative improvement `ΔE / |E|`, which makes the test scale-free.
- A sensitivity study over `ε, σ, k`.

## 3. Noise resilience (Sec. 6: "assess noise resilience")
- Run on `default.mixed` with depolarizing noise after each CNOT/IsingZZ (`qml.DepolarizingChannel`).
- Hypothesis: DDQAOA's shallower average circuits help more as noise increases.
  Plot the approximation ratio against the noise rate for DDQAOA and fixed p.
- Add finite-shot sampling in place of exact expectations (`shots=1024`).

## 4. Hardware constraints
- Transpile to a linear or heavy-hex coupling map (Qiskit or `qml.transforms`) and count
  CNOTs after SWAP insertion. The paper's CNOT counts assume all-to-all connectivity.

## 5. Other problems (Sec. 6: "generalization to other combinatorial problems")
- Max-Cut, TSP or a knapsack problem with the same DDQAOA loop. Only a new `*_to_qubo` is needed.

## 6. Engineering: faster simulator
- A NumPy statevector backend that exploits the diagonal `H_C` (cost layer = elementwise
  phase, mixer = per-qubit RX) with adjoint gradients. Alternatively, `lightning.qubit` with `diff_method="adjoint"`.
  This should give a large speed-up and makes the 100-instance (and 16/22-qubit) runs practical.

## 7. Penalty-weight study
- Examine how `λ, ρ` affect the landscape and the success probability. Large penalties
  compress the cost signal. Try `λ = ρ = α · max path cost` for several values of α.
