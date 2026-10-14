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
from pathlib import Path

from ddqaoa import (
    AdaptiveTransfer,
    ConvergenceMonitor,
    EnhancedConvergenceMonitor,
    OptimizerConfig,
    ResourceBudget,
    aggregate_benchmark_results,
    build_problem,
    generate_instance,
    interp_transfer,
    run_ddqaoa,
    run_enhanced_ddqaoa,
    run_fixed_qaoa,
    solve_classical,
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
        inst = generate_instance(n_nodes=args.n_nodes, seed=inst_seed)
        _, problem = build_problem(inst, n_slack=args.n_slack)
        path, cost = solve_classical(inst)
        print(f"\n[instance {i:02d} (seed={inst_seed})] qubits={problem.n_qubits} optimal path={path} cost={cost}")

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
            ),
        ))

        # 2. Enhanced DDQAOA (all 4 improvements combined)
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
