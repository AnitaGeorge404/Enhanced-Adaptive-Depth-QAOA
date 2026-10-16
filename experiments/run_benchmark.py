"""Benchmark DDQAOA vs Enhanced DDQAOA vs fixed-depth QAOA and ablations.

Supports reproducible paired benchmarking, separate instance/optimization seeds,
graceful failure tracking, ablation analysis, and statistical evaluation.

Examples:
    # Quick smoke test:
    python experiments/run_benchmark.py --smoke --out results/smoke.json

    # Full controlled comparison with ablations:
    python experiments/run_benchmark.py --n-instances 5 --steps 400 --p-list 3 5 --ablations --out results/benchmark_research.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
import traceback
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddqaoa import (
    AdaptiveTransfer,
    ConvergenceMonitor,
    CurvatureTrigger,
    EnhancedConvergenceMonitor,
    FidelityModel,
    NoiseAwareStopper,
    NoiseConfig,
    OptimizerConfig,
    ResourceBudget,
    aggregate_benchmark_results,
    build_feasible_mask,
    build_maxcut_problem,
    build_problem,
    evaluate_under_noise,
    generate_instance,
    generate_maxcut_instance,
    get_warm_start_angles,
    interp_transfer,
    reduce_instance,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_fixed_qaoa,
    run_noise_aware_ddqaoa,
    solve_classical,
    zero_pad_displace,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-nodes", type=int, default=4)
    ap.add_argument("--n-slack", type=int, default=None, help="slack qubits (default: minimal)")
    ap.add_argument("--n-instances", type=int, default=5)
    ap.add_argument("--instance-seed", type=int, default=0, help="seed for graph generation")
    ap.add_argument("--opt-seed", type=int, default=42, help="seed for optimizer initialization")
    ap.add_argument("--p-list", type=int, nargs="*", default=[3, 5], help="fixed depth QAOA baselines")
    ap.add_argument("--p-max", type=int, default=6)
    ap.add_argument("--steps", type=int, default=400, help="optimiser steps per run")
    ap.add_argument("--stepsize", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=30, help="patience window")
    ap.add_argument("--ablations", action="store_true", help="run individual ablation configurations")
    ap.add_argument("--smoke", action="store_true", help="run ultra-fast smoke test")
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.json"))

    # Research flags (Features 1-4)
    ap.add_argument("--reduce", action="store_true", help="enable optimality-preserving instance reduction (R0-R4)")
    ap.add_argument("--trigger", choices=["patience", "enhanced", "curvature"], default="enhanced", help="depth growth trigger")
    ap.add_argument("--transfer", choices=["interp", "adaptive", "fourier", "zero_pad_displace"], default="adaptive", help="parameter transfer strategy")
    ap.add_argument("--noise-aware", action="store_true", help="run noise-aware stopping and crossover analysis")
    ap.add_argument("--noise-eps2", type=float, default=0.005, help="two-qubit gate depolarizing error rate")
    ap.add_argument("--noise-rule", choices=["reactive", "predictive"], default="reactive", help="noise stopping rule")
    ap.add_argument("--noise-backend", choices=["auto", "surrogate", "mixed"], default="auto", help="backend for noisy evaluation")
    ap.add_argument("--objective", choices=["energy", "postselected"], default="energy", help="training objective")
    ap.add_argument("--gradient-method", choices=["autodiff", "parameter_shift", "spsa"], default="autodiff", help="gradient estimation method")
    ap.add_argument("--shots-per-batch", type=int, default=None, help="shots per circuit execution (None for exact statevector)")

    # Extended Research Suite (Features 2, 5, 8, 10)
    ap.add_argument("--device", choices=["default.qubit", "analytical"], default="default.qubit", help="simulation engine ('analytical' is 10-100x faster matrix-free simulator)")
    ap.add_argument("--problem", choices=["cspp", "maxcut"], default="cspp", help="optimization benchmark problem (cspp or maxcut)")
    ap.add_argument("--graph-type", choices=["regular", "erdos_renyi", "cycle", "complete"], default="regular", help="graph structure for MaxCut")
    ap.add_argument("--warm-start", action="store_true", help="enable LP / continuous relaxation warm-starting")
    ap.add_argument("--rem", action="store_true", help="evaluate final states with readout error mitigation")
    ap.add_argument("--rem-p01", type=float, default=0.03, help="bitflip 0->1 error rate for REM evaluation")
    ap.add_argument("--rem-p10", type=float, default=0.03, help="bitflip 1->0 error rate for REM evaluation")
    args = ap.parse_args()

    if args.smoke:
        args.n_instances = 2
        args.steps = 150
        args.p_list = [3]
        args.p_max = 4
        args.k = 15

    records = []
    print("=" * 78)
    print(f"BENCHMARK PIPELINE: {args.n_instances} instances, steps={args.steps}, p_max={args.p_max}")
    print(f"Instance seed: {args.instance_seed} | Opt seed: {args.opt_seed}")
    print("=" * 78)

    for i in range(args.n_instances):
        inst_seed = args.instance_seed + i
        if args.problem == "maxcut":
            inst = generate_maxcut_instance(
                n_nodes=args.n_nodes,
                seed=inst_seed,
                graph_type=args.graph_type,
            )
            inst, problem = build_maxcut_problem(inst)
            _, best_cost = inst.solve_classical()
            feasible_mask = None
            print(f"\n[instance {i:02d} (seed={inst_seed})] MaxCut ({args.graph_type}) qubits={problem.n_qubits} optimal cut={best_cost:.1f}")
        else:
            inst = generate_instance(n_nodes=args.n_nodes, seed=inst_seed)
            _, problem = build_problem(inst, n_slack=args.n_slack)
            path, cost = solve_classical(inst)
            feasible_mask = build_feasible_mask(problem, inst)
            print(f"\n[instance {i:02d} (seed={inst_seed})] CSPP qubits={problem.n_qubits} optimal path={path} cost={cost}")

        warm_angles = None
        if args.warm_start:
            try:
                warm_angles = get_warm_start_angles(inst)
            except Exception as err:
                print(f"   [Warm-Start Warning] Failed to compute warm start angles: {err}")

        # Method suite
        runs: list[tuple[str, any]] = []

        # 1. Baseline DDQAOA (original paper)
        runs.append((
            "ddqaoa",
            lambda: run_ddqaoa(
                problem,
                p_max=args.p_max,
                max_steps=args.steps,
                stepsize=args.stepsize,
                k=args.k,
                device=args.device,
            ),
        ))

        # 2. Enhanced DDQAOA
        runs.append((
            "ddqaoa_enhanced",
            lambda: run_enhanced_ddqaoa(
                problem,
                p_max=args.p_max,
                max_steps=args.steps,
                stepsize=args.stepsize,
                k=args.k,
                min_steps_per_depth=max(10, args.k // 2),
                min_marginal_gain=5e-4,
                patience_layers=1,
                grad_clip=1.0,
                depth_decay=0.1,
                gradient_method=args.gradient_method,
                shots=args.shots_per_batch,
                objective=args.objective,
                feasible_mask=feasible_mask,
                device=args.device,
            ),
        ))

        # 3. Warm-Started DDQAOA (Feature 2)
        if args.warm_start and warm_angles is not None:
            runs.append((
                "ddqaoa_warm_start",
                lambda: run_enhanced_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    min_steps_per_depth=max(10, args.k // 2),
                    min_marginal_gain=5e-4,
                    patience_layers=1,
                    grad_clip=1.0,
                    depth_decay=0.1,
                    gradient_method=args.gradient_method,
                    shots=args.shots_per_batch,
                    objective=args.objective,
                    feasible_mask=feasible_mask,
                    device=args.device,
                    warm_start_angles=warm_angles,
                    method_name="ddqaoa_warm_start",
                ),
            ))

        # 4. Research Features (if flags enabled or requested)
        if args.reduce and args.problem == "cspp":
            red_res = reduce_instance(inst)
            _, problem_reduced = build_problem(
                red_res.reduced_instance,
                n_slack=args.n_slack,
                fixed_edges=red_res.fixed_edges,
            )
            feasible_mask_red = build_feasible_mask(problem_reduced, red_res.reduced_instance)
            runs.append((
                "ddqaoa_reduced",
                lambda: run_enhanced_ddqaoa(
                    problem_reduced,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    min_steps_per_depth=max(10, args.k // 2),
                    min_marginal_gain=5e-4,
                    patience_layers=1,
                    grad_clip=1.0,
                    depth_decay=0.1,
                    gradient_method=args.gradient_method,
                    shots=args.shots_per_batch,
                    objective=args.objective,
                    feasible_mask=feasible_mask_red,
                    device=args.device,
                    method_name="ddqaoa_reduced",
                ),
            ))

        if args.trigger == "curvature" or args.transfer == "zero_pad_displace":
            transfer_fn = zero_pad_displace if args.transfer == "zero_pad_displace" else AdaptiveTransfer(strategy=args.transfer)
            runs.append((
                "ddqaoa_curvature",
                lambda: run_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    monitor=CurvatureTrigger(min_steps=max(5, args.k // 2), max_steps_per_depth=max(10, args.steps // 2)),
                    transfer=transfer_fn,
                    optimizer_config=OptimizerConfig(stepsize=args.stepsize, grad_clip=1.0, depth_decay=0.1),
                    resource_budget=ResourceBudget(min_marginal_gain=5e-4, patience_layers=1, min_depth=2),
                    gradient_method=args.gradient_method,
                    shots=args.shots_per_batch,
                    objective=args.objective,
                    feasible_mask=feasible_mask,
                    device=args.device,
                    method_name="ddqaoa_curvature",
                ),
            ))

        if args.noise_aware:
            runs.append((
                "ddqaoa_noise_aware",
                lambda: run_noise_aware_ddqaoa(
                    problem,
                    eps2=args.noise_eps2,
                    rule=args.noise_rule,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    noisy_backend=args.noise_backend,
                ),
            ))

        if args.objective == "postselected" and not any(r[0] == "ddqaoa_postselected" for r in runs):
            runs.append((
                "ddqaoa_postselected",
                lambda: run_enhanced_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    objective="postselected",
                    feasible_mask=feasible_mask,
                    gradient_method=args.gradient_method,
                    shots=args.shots_per_batch,
                    device=args.device,
                    method_name="ddqaoa_postselected",
                ),
            ))

        # 3. Ablations
        if args.ablations:
            # Ablation 1: No enhanced monitor (use original ConvergenceMonitor)
            runs.append((
                "abl_orig_monitor",
                lambda: run_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    monitor=ConvergenceMonitor(k=args.k),
                    transfer=AdaptiveTransfer(strategy="adaptive"),
                    resource_budget=ResourceBudget(min_marginal_gain=5e-4, patience_layers=1, min_depth=2),
                    optimizer_config=OptimizerConfig(stepsize=args.stepsize, grad_clip=1.0, depth_decay=0.1),
                    device=args.device,
                    method_name="abl_orig_monitor",
                ),
            ))

            # Ablation 2: No adaptive transfer (use original interp_transfer)
            runs.append((
                "abl_orig_transfer",
                lambda: run_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    monitor=EnhancedConvergenceMonitor(k=args.k, min_steps=max(10, args.k // 2)),
                    transfer=interp_transfer,
                    resource_budget=ResourceBudget(min_marginal_gain=5e-4, patience_layers=1, min_depth=2),
                    optimizer_config=OptimizerConfig(stepsize=args.stepsize, grad_clip=1.0, depth_decay=0.1),
                    device=args.device,
                    method_name="abl_orig_transfer",
                ),
            ))

            # Ablation 3: No resource-aware stopping
            runs.append((
                "abl_no_resource_budget",
                lambda: run_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    monitor=EnhancedConvergenceMonitor(k=args.k, min_steps=max(10, args.k // 2)),
                    transfer=AdaptiveTransfer(strategy="adaptive"),
                    resource_budget=None,
                    optimizer_config=OptimizerConfig(stepsize=args.stepsize, grad_clip=1.0, depth_decay=0.1),
                    device=args.device,
                    method_name="abl_no_resource_budget",
                ),
            ))

            # Ablation 4: No controlled optimizer (vanilla Adam)
            runs.append((
                "abl_vanilla_adam",
                lambda: run_ddqaoa(
                    problem,
                    p_max=args.p_max,
                    max_steps=args.steps,
                    stepsize=args.stepsize,
                    k=args.k,
                    monitor=EnhancedConvergenceMonitor(k=args.k, min_steps=max(10, args.k // 2)),
                    transfer=AdaptiveTransfer(strategy="adaptive"),
                    resource_budget=ResourceBudget(min_marginal_gain=5e-4, patience_layers=1, min_depth=2),
                    optimizer_config=None,
                    device=args.device,
                    method_name="abl_vanilla_adam",
                ),
            ))

        # 4. Fixed-depth QAOA baselines
        for p in args.p_list:
            runs.append((
                f"p={p}",
                lambda p=p: run_fixed_qaoa(
                    problem,
                    p,
                    steps=args.steps,
                    stepsize=args.stepsize,
                    seed=args.opt_seed + i,
                    device=args.device,
                ),
            ))

        # Execute runs for current instance
        for name, fn in runs:
            t0 = time.perf_counter()
            try:
                res = fn()
                elapsed = time.perf_counter() - t0
                print(
                    f"   {name:22s} r={res.approx_ratio:.4f}  P_succ={res.success_prob:.4f}  "
                    f"depth={res.final_depth:2d}  cumCNOT={res.cumulative_cnots:,}  "
                    f"({elapsed:.1f}s)"
                )
                summary_data = res.summary()

                if args.rem:
                    try:
                        noise_cfg = NoiseConfig(
                            p_single=0.001,
                            p_two=args.noise_eps2,
                            p_readout_01=args.rem_p01,
                            p_readout_10=args.rem_p10,
                            shots=args.shots_per_batch or 2000,
                            seed=args.opt_seed + i,
                        )
                        eval_res = evaluate_under_noise(
                            gammas=res.gammas,
                            betas=res.betas,
                            problem=problem,
                            noise=noise_cfg,
                            apply_rem=True,
                            warm_start_angles=warm_angles if name == "ddqaoa_warm_start" else None,
                        )
                        summary_data["rem"] = eval_res.summary()
                        print(
                            f"      [REM] P_raw={eval_res.unmitigated_success_prob:.4f} -> "
                            f"P_mit={eval_res.mitigated_success_prob:.4f} "
                            f"(gain={eval_res.rem_recovery_ratio:.2f}x)"
                        )
                    except Exception as rem_err:
                        print(f"      [REM Warning] {rem_err}")

                records.append({
                    "instance": i,
                    "instance_seed": inst_seed,
                    "opt_seed": args.opt_seed + i,
                    "n_qubits": problem.n_qubits,
                    "status": "SUCCESS",
                    "elapsed_sec": elapsed,
                    **summary_data,
                })
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"   {name:22s} FAILED: {exc} ({elapsed:.1f}s)")
                records.append({
                    "instance": i,
                    "instance_seed": inst_seed,
                    "opt_seed": args.opt_seed + i,
                    "method": name,
                    "n_qubits": problem.n_qubits,
                    "status": "FAILED",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_sec": elapsed,
                })

    # Aggregate statistical validation
    analysis = aggregate_benchmark_results(records, baseline_method="ddqaoa")

    payload = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "0.2.0",
        },
        "args": {k: str(v) for k, v in vars(args).items()},
        "analysis": analysis,
        "runs": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(records)} runs and statistical analysis -> {args.out}")


if __name__ == "__main__":
    main()
