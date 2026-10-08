"""Benchmark DDQAOA vs fixed-depth QAOA on random CSPP instances (paper Sec. 4-5).

Example (quick):
    python experiments/run_benchmark.py --n-instances 3 --steps 300 --p-list 3 5
Paper-like 10-qubit setting:
    python experiments/run_benchmark.py --n-nodes 4 --n-slack 3 --n-instances 100 --steps 1200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ddqaoa import build_problem, generate_instance, run_ddqaoa, run_fixed_qaoa, solve_classical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=4)
    ap.add_argument("--n-slack", type=int, default=None, help="slack qubits (default: minimal)")
    ap.add_argument("--n-instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-list", type=int, nargs="*", default=[3, 5, 10, 15])
    ap.add_argument("--p-max", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1200, help="optimiser steps per run")
    ap.add_argument("--stepsize", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=50, help="DDQAOA patience window")
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    args = ap.parse_args()

    records = []
    for i in range(args.n_instances):
        inst = generate_instance(n_nodes=args.n_nodes, seed=args.seed + i)
        _, problem = build_problem(inst, n_slack=args.n_slack)
        path, cost = solve_classical(inst)
        print(f"[instance {i}] qubits={problem.n_qubits} optimal path={path} cost={cost}")

        runs = [("ddqaoa", lambda: run_ddqaoa(problem, p_max=args.p_max, max_steps=args.steps,
                                               stepsize=args.stepsize, k=args.k))]
        runs += [(f"p={p}", lambda p=p: run_fixed_qaoa(problem, p, steps=args.steps,
                                                       stepsize=args.stepsize, seed=args.seed + i))
                 for p in args.p_list]
        for name, fn in runs:
            t0 = time.perf_counter()
            res = fn()
            print(f"   {name:8s} r={res.approx_ratio:.4f}  P_succ={res.success_prob:.4f}  "
                  f"depth={res.final_depth:2d}  cumCNOT={res.cumulative_cnots:,}  "
                  f"({time.perf_counter() - t0:.1f}s)")
            records.append({"instance": i, "n_qubits": problem.n_qubits, **res.summary()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "runs": records}, indent=1))
    print(f"saved {len(records)} runs -> {args.out}")


if __name__ == "__main__":
    main()
