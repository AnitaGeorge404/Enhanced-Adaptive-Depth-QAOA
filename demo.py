"""Step-by-step demo of the project.

    python demo.py            # runs straight through (~1-2 min)
    python demo.py --pause    # waits for Enter between steps (for presenting)
    python demo.py --seed 25 --compare 3 5

Steps: build a route problem -> solve it classically -> turn it into a quantum energy
formula -> run DDQAOA (watch it add layers) -> run normal fixed-depth QAOA ->
look at what the quantum circuit measures -> compare and save a plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddqaoa import build_problem, cnots_per_layer, decode, generate_instance, make_qnodes
from ddqaoa import run_ddqaoa, run_fixed_qaoa, solve_classical
from ddqaoa.circuits import index_to_bits
from ddqaoa.cspp import simple_paths
from ddqaoa.metrics import approximation_ratio


def header(n: int, title: str, pause: bool) -> None:
    if pause and n > 1:
        input("\n[press Enter to continue]")
    print(f"\n{'=' * 70}\nSTEP {n}: {title}\n{'=' * 70}")


def route_str(path) -> str:
    return " -> ".join(map(str, path))


def edges_to_route(edges, s: int, t: int) -> str | None:
    """Follow the chosen roads from s; return the route if they form one clean s->t path."""
    nxt = dict(edges)
    if len(nxt) != len(edges):
        return None
    path, node = [s], s
    while node in nxt and len(path) <= len(edges) + 1:
        node = nxt[node]
        path.append(node)
    return route_str(path) if node == t and len(path) == len(edges) + 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=11, help="which random map to use")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300, help="optimiser steps per method")
    ap.add_argument("--p-max", type=int, default=5, help="max layers for DDQAOA")
    ap.add_argument("--compare", type=int, nargs="*", default=[3], help="fixed depths to compare")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    args = ap.parse_args()
    lr, k = 0.05, 30

    # ------------------------------------------------------------------ 1
    header(1, "The problem - a small map with roads", args.pause)
    inst = generate_instance(n_nodes=args.nodes, seed=args.seed)
    s, t = inst.source, inst.target
    print(f"Go from city {s} to city {t}. Each road has a COST and uses some FUEL.\n")
    print(f"   {'road':10s} {'cost':>5s} {'fuel':>5s}")
    for u, v in inst.edges:
        print(f"   {u} -> {v:<5d} {inst.cost[(u, v)]:5d} {inst.resource[(u, v)]:5d}")
    print(f"\nFuel limit: {inst.r_limit}")

    # ------------------------------------------------------------------ 2
    header(2, "Classical answer - check every route (only possible for tiny maps)", args.pause)
    best_path, best_cost = solve_classical(inst)
    for p in sorted(simple_paths(inst), key=inst.path_cost):
        ok = inst.path_resource(p) <= inst.r_limit
        mark = "  <-- BEST VALID ROUTE" if p == best_path else ""
        print(f"   {route_str(p):16s} cost {inst.path_cost(p):3d}   fuel {inst.path_resource(p):3d}"
              f"   {'ok' if ok else 'TOO MUCH FUEL'}{mark}")
    print("\nThe number of routes explodes as maps grow - that's why we try a quantum method.")

    # ------------------------------------------------------------------ 3
    header(3, "Turn the problem into a quantum energy formula (QUBO -> Ising)", args.pause)
    qubo, problem = build_problem(inst)
    n_roads = len(inst.edges)
    n_cnot = cnots_per_layer(problem.ising)
    print(f"   1 qubit per road          : {n_roads}")
    print(f"   slack qubits (fuel limit) : {qubo.n_slack}")
    print(f"   total qubits              : {problem.n_qubits}")
    print(f"   single-qubit terms (h Z)  : {len(problem.ising.h)}")
    print(f"   qubit-pair terms (J ZZ)   : {len(problem.ising.J)}  ->  {n_cnot} CNOT gates per layer")
    g = index_to_bits(int(problem.ground_states[0]), problem.n_qubits)
    bits = "".join(map(str, g))
    print(f"\nLowest-energy bitstring: {bits[:n_roads]} | {bits[n_roads:]}   (roads | slack)")
    print(f"It decodes to route   : {edges_to_route(decode(g, qubo), s, t)}")
    print("Same as the classical answer -> the formula is correct.")
    print(f"Search space: 2^{problem.n_qubits} = {2 ** problem.n_qubits} bitstrings.")

    # ------------------------------------------------------------------ 4
    header(4, "DDQAOA - start with 1 layer, add a layer whenever progress stalls", args.pause)
    state = {"depth": 0}

    def progress(step: int, depth: int, energy: float) -> None:
        if depth != state["depth"]:
            state["depth"] = depth
            print(f"   step {step:4d}: now using {depth} layer(s) "
                  f"({depth * n_cnot} CNOTs)   quality r = {approximation_ratio(energy, problem):.4f}")

    dd = run_ddqaoa(problem, p_max=args.p_max, max_steps=args.steps, stepsize=lr, k=k, callback=progress)
    print(f"   finished after {len(dd.energies)} steps at {dd.final_depth} layers, "
          f"quality r = {dd.approx_ratio:.4f}")

    # ------------------------------------------------------------------ 5
    header(5, f"Normal QAOA - fixed number of layers {args.compare}, chosen in advance", args.pause)
    fixed = []
    for p in args.compare:
        res = run_fixed_qaoa(problem, p, steps=args.steps, stepsize=lr, seed=args.seed)
        fixed.append(res)
        print(f"   p = {p:2d}: quality r = {res.approx_ratio:.4f}   ({p * n_cnot} CNOTs every step)")

    # ------------------------------------------------------------------ 6
    header(6, "What does the DDQAOA circuit actually measure?", args.pause)
    probs_fn, _ = make_qnodes(problem)
    probs = np.asarray(probs_fn(dd.gammas, dd.betas))
    print(f"   {'chance':>7s}   {'bitstring':{problem.n_qubits + 3}s} route")
    for idx in np.argsort(probs)[::-1][:6]:
        b = index_to_bits(int(idx), problem.n_qubits)
        bs = "".join(map(str, b))
        route = edges_to_route(decode(b, qubo), s, t) or "not a valid route"
        star = "  <-- optimal" if idx in problem.ground_states else ""
        print(f"   {probs[idx]:7.2%}   {bs[:n_roads]}|{bs[n_roads:]}   {route}{star}")
    print(f"\nRandom guessing would find the optimum with chance 1/{2 ** problem.n_qubits}"
          f" = {1 / 2 ** problem.n_qubits:.2%}; DDQAOA: {dd.success_prob:.2%}")
    print("The other top results are 'near misses' (low energy, but break a rule). QAOA optimises the\n"
          "AVERAGE energy, not the chance of the exact answer - the paper sees the same effect, and\n"
          "it is what our CVaR extension tries to fix.")

    # ------------------------------------------------------------------ 7
    header(7, "Comparison", args.pause)
    print(f"   {'method':8s} {'quality r':>10s} {'P(optimal)':>11s} {'layers':>7s} {'total CNOTs used':>17s}")
    for res in [dd] + fixed:
        print(f"   {res.method:8s} {res.approx_ratio:10.4f} {res.success_prob:11.2%} "
              f"{res.final_depth:7d} {res.cumulative_cnots:17,d}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for res in [dd] + fixed:
        r = [approximation_ratio(e, problem) for e in res.energies]
        ax1.plot(r, label=res.method, lw=2 if res is dd else 1.2)
        ax2.plot(np.array(res.depths) * n_cnot, label=res.method, lw=2 if res is dd else 1.2)
    for x in dd.depth_switch_steps:
        ax1.axvline(x, ls=":", color="gray", lw=0.8)
    ax1.set(title="Answer quality while training", xlabel="optimiser step", ylabel="approximation ratio r")
    ax2.set(title="Circuit size (CNOT gates) while training", xlabel="optimiser step", ylabel="CNOTs per circuit")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    out = Path("results/demo.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    print("Dotted lines on the left = moments DDQAOA added a layer.")


if __name__ == "__main__":
    main()
