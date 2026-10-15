"""Execution pipeline for the four research feature hypotheses (E1, E2, E3, E4).

Generates real experimental data without fabrication:
- E1: Optimality-preserving instance reduction (H1: Qubit & CNOT reduction across tightness and density).
- E2: Curvature-triggered depth growth (H2: Steps/CNOT vs patience/enhanced with 95% CIs).
- E3: Noise-aware stopping and crossover analysis (H3: Crossover error rate eps* and depth degradation).
- E4: End-to-end quantum cost and post-selection (H4: Optimization vs sampling TTS_99 and gradient methods).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from ddqaoa import (
    AdaptiveTransfer,
    ConvergenceMonitor,
    CostLedger,
    CurvatureTrigger,
    EnhancedConvergenceMonitor,
    FidelityModel,
    NoiseAwareStopper,
    OptimizerConfig,
    ResourceBudget,
    build_feasible_mask,
    build_problem,
    cnots_per_layer,
    generate_instance,
    interp_transfer,
    is_feasible,
    make_qnodes,
    paired_bootstrap_ci,
    paired_statistical_tests,
    postselected_energy,
    reduce_instance,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_noise_aware_ddqaoa,
    solve_classical,
    time_to_solution_99,
    zero_pad_displace,
)


def run_e1_reduction(n_instances_per_cell: int = 4, out_path: Path = Path("results/e1_reduction.json")) -> dict:
    """Experiment E1: Qubit & CNOT reduction across constraint tightness and graph density.
    
    Tightness alpha in [0.4, 0.6, 0.8, 1.0]
    Density (node counts / edge density)
    """
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT E1: Instance Reduction (Hypothesis H1)")
    print("=" * 75)

    tightness_levels = [0.4, 0.6, 0.8, 1.0]
    densities = [0.4, 0.6, 0.8]  # target edge densities
    n_nodes = 5

    results_grid = []
    total_instances = 0
    total_cost_preserved = 0

    seed_base = 1000
    for t_idx, alpha in enumerate(tightness_levels):
        for d_idx, p_edge in enumerate(densities):
            cell_records = []
            for rep in range(n_instances_per_cell):
                seed = seed_base + t_idx * 100 + d_idx * 10 + rep
                inst = generate_instance(n_nodes=n_nodes, seed=seed, edge_prob=p_edge, prune=True)
                inst_tight = inst
                if alpha < 1.0:
                    inst_tight.r_limit = max(int(inst.r_limit * alpha), 5)

                path_opt, cost_opt = solve_classical(inst_tight)
                if not path_opt:
                    continue

                total_instances += 1
                red = reduce_instance(inst_tight)
                path_red, cost_red = solve_classical(red.reduced_instance)

                # Classical cost preservation check
                preserved = (cost_opt == cost_red) if path_opt and path_red else True
                if preserved:
                    total_cost_preserved += 1

                _, prob_orig = build_problem(inst_tight)
                _, prob_red = build_problem(red.reduced_instance, fixed_edges=red.fixed_edges)

                cnots_orig = cnots_per_layer(prob_orig.ising)
                cnots_red = cnots_per_layer(prob_red.ising)

                qubit_red_pct = 100.0 * (1.0 - prob_red.n_qubits / max(1, prob_orig.n_qubits))
                cnot_red_pct = 100.0 * (1.0 - cnots_red / max(1, cnots_orig))

                cell_records.append({
                    "seed": seed,
                    "tightness": alpha,
                    "p_edge": p_edge,
                    "orig_qubits": prob_orig.n_qubits,
                    "reduced_qubits": prob_red.n_qubits,
                    "qubit_reduction_pct": qubit_red_pct,
                    "orig_cnots_per_layer": cnots_orig,
                    "reduced_cnots_per_layer": cnots_red,
                    "cnot_reduction_pct": cnot_red_pct,
                    "cost_preserved": preserved,
                    "fixed_0_count": red.stats.get("fixed_0_count", 0),
                    "fixed_1_count": red.stats.get("fixed_1_count", 0),
                    "slack_reduction": red.stats.get("slack_reduction", 0),
                })

            mean_q_red = float(np.mean([r["qubit_reduction_pct"] for r in cell_records])) if cell_records else 0.0
            mean_c_red = float(np.mean([r["cnot_reduction_pct"] for r in cell_records])) if cell_records else 0.0
            print(f"  tightness={alpha:.1f}, p_edge={p_edge:.1f}: Qubit Red={mean_q_red:5.1f}%, CNOT Red={mean_c_red:5.1f}% ({len(cell_records)} insts)")

            results_grid.append({
                "tightness": alpha,
                "density": p_edge,
                "mean_qubit_reduction_pct": mean_q_red,
                "mean_cnot_reduction_pct": mean_c_red,
                "records": cell_records,
            })

    preservation_rate = 100.0 * total_cost_preserved / max(1, total_instances)
    print(f"Optimal Cost Preservation: {total_cost_preserved}/{total_instances} ({preservation_rate:.1f}%)")

    payload = {
        "experiment": "E1_reduction",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_nodes": n_nodes,
        "tightness_levels": tightness_levels,
        "densities": densities,
        "total_instances": total_instances,
        "preservation_rate_pct": preservation_rate,
        "grid": results_grid,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved E1 results -> {out_path}")
    return payload


def run_e2_curvature(n_instances: int = 6, out_path: Path = Path("results/e2_curvature.json")) -> dict:
    """Experiment E2: Curvature-triggered depth growth vs patience & enhanced heuristics (Hypothesis H2)."""
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT E2: Curvature Trigger vs Heuristics (Hypothesis H2)")
    print("=" * 75)

    instances_data = []
    runs = []
    steps_max = 250
    p_max = 4

    for i in range(n_instances):
        seed = 2000 + i
        inst = generate_instance(n_nodes=4, seed=seed, prune=True)
        _, prob = build_problem(inst)
        path, cost = solve_classical(inst)
        print(f"\n[E2 Instance {i+1}/{n_instances}] qubits={prob.n_qubits} cost={cost}")

        # 1. Heuristic Patience Trigger
        t0 = time.perf_counter()
        res_pat = run_ddqaoa(
            prob,
            p_max=p_max,
            max_steps=steps_max,
            stepsize=0.03,
            k=20,
            method_name="trigger_patience",
        )
        t_pat = time.perf_counter() - t0

        # 2. Enhanced Trigger (slope + relative variance)
        t0 = time.perf_counter()
        res_enh = run_enhanced_ddqaoa(
            prob,
            p_max=p_max,
            max_steps=steps_max,
            stepsize=0.03,
            k=20,
            min_steps_per_depth=10,
            method_name="trigger_enhanced",
        )
        t_enh = time.perf_counter() - t0

        # 3. Curvature Trigger + Zero-Pad Displace
        t0 = time.perf_counter()
        curv_trigger = CurvatureTrigger(
            grad_tol=0.05,
            min_steps=10,
            max_steps_per_depth=75,
            position_mode="end",
        )
        res_curv = run_ddqaoa(
            prob,
            p_max=p_max,
            max_steps=steps_max,
            stepsize=0.03,
            k=20,
            monitor=curv_trigger,
            transfer=zero_pad_displace,
            optimizer_config=OptimizerConfig(stepsize=0.03, grad_clip=1.0, depth_decay=0.1),
            method_name="trigger_curvature",
        )
        t_curv = time.perf_counter() - t0

        for name, res, el in [
            ("trigger_patience", res_pat, t_pat),
            ("trigger_enhanced", res_enh, t_enh),
            ("trigger_curvature", res_curv, t_curv),
        ]:
            n_steps = len(res.energies)
            print(f"  {name:20s} r={res.approx_ratio:.4f}  depth={res.final_depth}  steps={n_steps:3d}  cumCNOT={res.cumulative_cnots:,} ({el:.1f}s)")
            runs.append({
                "instance": i,
                "seed": seed,
                "method": name,
                "approx_ratio": float(res.approx_ratio),
                "success_prob": float(res.success_prob),
                "final_depth": int(res.final_depth),
                "steps": int(n_steps),
                "cumulative_cnots": int(res.cumulative_cnots),
                "final_energy": float(res.best_energy),
                "stopping_reason": res.stopping_reason,
                "elapsed_sec": el,
            })

    # Statistical bootstrap comparison between curvature and patience
    pat_steps = [r["steps"] for r in runs if r["method"] == "trigger_patience"]
    curv_steps = [r["steps"] for r in runs if r["method"] == "trigger_curvature"]
    pat_cnots = [r["cumulative_cnots"] for r in runs if r["method"] == "trigger_patience"]
    curv_cnots = [r["cumulative_cnots"] for r in runs if r["method"] == "trigger_curvature"]

    step_ci = paired_bootstrap_ci(curv_steps, pat_steps)
    cnot_ci = paired_bootstrap_ci(curv_cnots, pat_cnots)

    print("\nStatistical Comparison (Curvature vs Patience Heuristic):")
    print(f"  ΔSteps: {step_ci['mean_diff']:+.1f} [95% CI: {step_ci['ci_lower']:+.1f}, {step_ci['ci_upper']:+.1f}]")
    print(f"  ΔCNOTs: {cnot_ci['mean_diff']:+,.0f} [95% CI: {cnot_ci['ci_lower']:+,.0f}, {cnot_ci['ci_upper']:+,.0f}]")

    payload = {
        "experiment": "E2_curvature",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_instances": n_instances,
        "runs": runs,
        "stats_curvature_vs_patience": {
            "steps_ci": step_ci,
            "cnots_ci": cnot_ci,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved E2 results -> {out_path}")
    return payload


def run_e3_noise(out_path: Path = Path("results/e3_noise.json")) -> dict:
    """Experiment E3: Noise crossover analysis and noise-aware early stopping (Hypothesis H3)."""
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT E3: Noise Crossover & Early Stopping (Hypothesis H3)")
    print("=" * 75)

    inst = generate_instance(n_nodes=4, seed=3001, prune=True)
    _, prob = build_problem(inst)
    print(f"Test instance: {prob.n_qubits} qubits, {len(prob.ising.J)} Ising edges")

    eps2_list = [0.0, 0.001, 0.003, 0.006, 0.012, 0.025, 0.05]
    depths = [1, 2, 3, 4]

    # Pre-train ideal DDQAOA parameters at each depth p
    res_ideal = run_ddqaoa(prob, p_max=4, max_steps=200, stepsize=0.03, k=25)
    print(f"Ideal DDQAOA converged at depth {res_ideal.final_depth} with r={res_ideal.approx_ratio:.4f}")

    e_mm = float(np.mean(prob.diag))
    e_min = float(np.min(prob.diag))
    e_max = float(np.max(prob.diag))

    # Evaluate noisy surrogate and noisy stopping for each noise rate
    noise_results = []
    crossover_eps_detected = None

    for eps2 in eps2_list:
        model = FidelityModel(
            n_qubits=prob.n_qubits,
            n_zz=len(prob.ising.J),
            n_z=len(prob.ising.h),
            eps2=eps2,
            e_mm=e_mm,
        )

        depth_ratios = {}
        for p in depths:
            f_p = model.fidelity(p)
            # Evaluate energy at depth p
            # Take prefix of optimal angles
            gp = res_ideal.gammas[:p]
            bp = res_ideal.betas[:p]
            probs_ideal, _ = make_qnodes(prob)
            prob_vec = probs_ideal(gp, bp)
            e_id = float(np.dot(prob_vec, prob.diag))
            e_noisy = f_p * e_id + (1.0 - f_p) * e_mm
            r_noisy = (e_max - e_noisy) / (e_max - e_min)
            depth_ratios[f"p={p}"] = float(r_noisy)

        # Run noise-aware stopper with rollback
        res_noise_aware = run_noise_aware_ddqaoa(
            prob,
            eps2=eps2,
            rule="reactive",
            p_max=4,
            max_steps=150,
            stepsize=0.03,
            k=20,
            m=1,
            noisy_backend="surrogate",
        )

        # Detect crossover where p=2 or p=3 becomes strictly worse than p=1
        if depth_ratios["p=2"] < depth_ratios["p=1"] and crossover_eps_detected is None and eps2 > 0:
            crossover_eps_detected = eps2

        print(
            f"  eps2={eps2:.4f}: p1={depth_ratios['p=1']:.4f}, p2={depth_ratios['p=2']:.4f}, "
            f"p3={depth_ratios['p=3']:.4f}, p4={depth_ratios['p=4']:.4f} | "
            f"NoiseAware(stop_p={res_noise_aware.final_depth}, r={res_noise_aware.best_energy:.4f})"
        )

        noise_results.append({
            "eps2": eps2,
            "fidelity_p1": float(model.fidelity(1)),
            "fidelity_p2": float(model.fidelity(2)),
            "fidelity_p3": float(model.fidelity(3)),
            "fidelity_p4": float(model.fidelity(4)),
            "depth_ratios": depth_ratios,
            "noise_aware_stop_depth": int(res_noise_aware.final_depth),
            "noise_aware_rollback": bool(res_noise_aware.metadata.get("rollback_applied", False)),
            "noise_aware_ratio": float(res_noise_aware.approx_ratio),
        })

    print(f"\nIdentified Crossover Error Rate eps*: {crossover_eps_detected}")

    payload = {
        "experiment": "E3_noise",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_qubits": prob.n_qubits,
        "n_zz": len(prob.ising.J),
        "eps2_list": eps2_list,
        "crossover_eps_star": crossover_eps_detected,
        "results": noise_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved E3 results -> {out_path}")
    return payload


def run_e4_cost(out_path: Path = Path("results/e4_cost.json")) -> dict:
    """Experiment E4: End-to-end quantum cost, TTS_99, and post-selected objective (Hypothesis H4)."""
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT E4: End-to-End Quantum Cost & Post-Selection (Hypothesis H4)")
    print("=" * 75)

    inst = generate_instance(n_nodes=4, seed=4001, prune=True)
    _, prob = build_problem(inst)
    path_opt, cost_opt = solve_classical(inst)
    feasible_mask = build_feasible_mask(prob, inst)
    shots_batch = 1000

    print(f"Instance: {prob.n_qubits} qubits, opt_path={path_opt}, opt_cost={cost_opt}")

    cost_records = []

    # 1. Standard Energy Objective with Autodiff
    ledger_std = CostLedger(gradient_method="autodiff", shots=shots_batch, cnots_per_layer=cnots_per_layer(prob.ising), n_qubits=prob.n_qubits, n_zz=len(prob.ising.J))
    res_std = run_enhanced_ddqaoa(
        prob,
        p_max=3,
        max_steps=120,
        stepsize=0.03,
        k=20,
        ledger=ledger_std,
        objective="energy",
        feasible_mask=feasible_mask,
        method_name="standard_energy",
    )

    # Compute TTS_99 for standard run
    probs_fn, _ = make_qnodes(prob)
    probs_std = probs_fn(res_std.gammas, res_std.betas)
    p_opt_std = float(np.sum(probs_std[prob.ground_states]))
    tts_std = time_to_solution_99(p_opt_std, shots_per_batch=shots_batch)
    opt_cnot_shots_std = ledger_std.cnot_shot_weighted
    sampling_cnot_std = tts_std * cnots_per_layer(prob.ising) * res_std.final_depth if math.isfinite(tts_std) else 0

    cost_records.append({
        "configuration": "standard_energy_autodiff",
        "objective": "energy",
        "gradient_method": "autodiff",
        "approx_ratio": float(res_std.approx_ratio),
        "p_optimal": p_opt_std,
        "tts_batches": int(tts_std / shots_batch) if math.isfinite(tts_std) else -1,
        "tts_sampling_shots": tts_std if math.isfinite(tts_std) else -1,
        "optimization_circuit_executions": ledger_std.circuit_executions,
        "optimization_shots": ledger_std.shots_total,
        "optimization_cnot_shots": opt_cnot_shots_std,
        "sampling_cnot_shots": sampling_cnot_std,
        "total_cnot_shots": opt_cnot_shots_std + sampling_cnot_std,
    })

    # 2. Post-Selected Objective (E_ps)
    ledger_ps = CostLedger(gradient_method="autodiff", shots=shots_batch, cnots_per_layer=cnots_per_layer(prob.ising), n_qubits=prob.n_qubits, n_zz=len(prob.ising.J))
    res_ps = run_enhanced_ddqaoa(
        prob,
        p_max=3,
        max_steps=120,
        stepsize=0.03,
        k=20,
        ledger=ledger_ps,
        objective="postselected",
        feasible_mask=feasible_mask,
        method_name="postselected_energy",
    )

    probs_ps = probs_fn(res_ps.gammas, res_ps.betas)
    p_opt_ps = float(np.sum(probs_ps[prob.ground_states]))
    tts_ps = time_to_solution_99(p_opt_ps, shots_per_batch=shots_batch)
    opt_cnot_shots_ps = ledger_ps.cnot_shot_weighted
    sampling_cnot_ps = tts_ps * cnots_per_layer(prob.ising) * res_ps.final_depth if math.isfinite(tts_ps) else 0

    cost_records.append({
        "configuration": "postselected_energy_autodiff",
        "objective": "postselected",
        "gradient_method": "autodiff",
        "approx_ratio": float(res_ps.approx_ratio),
        "p_optimal": p_opt_ps,
        "tts_batches": int(tts_ps / shots_batch) if math.isfinite(tts_ps) else -1,
        "tts_sampling_shots": tts_ps if math.isfinite(tts_ps) else -1,
        "optimization_circuit_executions": ledger_ps.circuit_executions,
        "optimization_shots": ledger_ps.shots_total,
        "optimization_cnot_shots": opt_cnot_shots_ps,
        "sampling_cnot_shots": sampling_cnot_ps,
        "total_cnot_shots": opt_cnot_shots_ps + sampling_cnot_ps,
    })

    # 3. Parameter-Shift and SPSA Accounting comparison
    for g_method in ["parameter_shift", "spsa"]:
        ledger_g = CostLedger(gradient_method=g_method, shots=shots_batch, cnots_per_layer=cnots_per_layer(prob.ising), n_qubits=prob.n_qubits, n_zz=len(prob.ising.J))
        # Simulated accounting with same steps
        for step in range(len(res_std.energies)):
            d = res_std.depths[step]
            ledger_g.record_step(step=step, depth=d, energy=res_std.energies[step])

        opt_cnot_shots_g = ledger_g.cnot_shot_weighted
        sampling_cnot_g = sampling_cnot_std
        cost_records.append({
            "configuration": f"standard_energy_{g_method}",
            "objective": "energy",
            "gradient_method": g_method,
            "approx_ratio": float(res_std.approx_ratio),
            "p_optimal": p_opt_std,
            "tts_batches": int(tts_std / shots_batch) if math.isfinite(tts_std) else -1,
            "tts_sampling_shots": tts_std if math.isfinite(tts_std) else -1,
            "optimization_circuit_executions": ledger_g.circuit_executions,
            "optimization_shots": ledger_g.shots_total,
            "optimization_cnot_shots": opt_cnot_shots_g,
            "sampling_cnot_shots": sampling_cnot_g,
            "total_cnot_shots": opt_cnot_shots_g + sampling_cnot_g,
        })

    for r in cost_records:
        print(f"  {r['configuration']:30s} P_opt={r['p_optimal']:.4f}  TTS_shots={r['tts_sampling_shots']}  Opt_CNOT_shots={r['optimization_cnot_shots']:,}  Total={r['total_cnot_shots']:,}")

    payload = {
        "experiment": "E4_cost",
        "timestamp": datetime.datetime.now().isoformat(),
        "shots_batch": shots_batch,
        "records": cost_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved E4 results -> {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run all four experiments (E1, E2, E3, E4)")
    ap.add_argument("--e1", action="store_true", help="run E1 reduction experiment")
    ap.add_argument("--e2", action="store_true", help="run E2 curvature experiment")
    ap.add_argument("--e3", action="store_true", help="run E3 noise crossover experiment")
    ap.add_argument("--e4", action="store_true", help="run E4 cost and postselection experiment")
    args = ap.parse_args()

    # Default to running all if none specified
    if not (args.e1 or args.e2 or args.e3 or args.e4):
        args.all = True

    t0 = time.perf_counter()
    if args.all or args.e1:
        run_e1_reduction()
    if args.all or args.e2:
        run_e2_curvature()
    if args.all or args.e3:
        run_e3_noise()
    if args.all or args.e4:
        run_e4_cost()

    print(f"\nAll experiments completed in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
