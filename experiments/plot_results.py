"""Plots and tables from a benchmark JSON (paper Figs. 2-5, Tables 1-2).

    python experiments/plot_results.py results/benchmark.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(path: str) -> None:
    runs = json.loads(Path(path).read_text())["runs"]
    out_dir = Path(path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)
    methods = sorted(by_method, key=lambda m: (m != "ddqaoa", len(m), m))

    # Table 1 / Table 2 style summary
    base = np.mean([r["cumulative_cnots"] for r in by_method["ddqaoa"]]) if "ddqaoa" in by_method else None
    print(f"{'method':8s} {'mean r':>8s} {'med r':>8s} {'mean Ps':>8s} {'med Ps':>8s} {'cumCNOT':>12s} {'vs DD':>6s}")
    for m in methods:
        rs = np.array([r["approx_ratio"] for r in by_method[m]])
        ps = np.array([r["success_prob"] for r in by_method[m]])
        cn = np.mean([r["cumulative_cnots"] for r in by_method[m]])
        rel = f"{cn / base:.2f}x" if base else "-"
        print(f"{m:8s} {rs.mean():8.4f} {np.median(rs):8.4f} {ps.mean():8.4f} {np.median(ps):8.4f} {cn:12,.0f} {rel:>6s}")

    # Fig. 2 / Fig. 4: distributions
    for key, label in [("approx_ratio", "Approximation ratio"), ("success_prob", "Success probability")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([[r[key] for r in by_method[m]] for m in methods], tick_labels=methods)
        ax.set_ylabel(label)
        ax.set_xlabel("Method")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}_box.png", dpi=150)
        plt.close(fig)

    # Fig. 3 / Fig. 5: trajectories for the first instance
    first = [r for r in runs if r["instance"] == runs[0]["instance"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in first:
        ax1.plot(r["energies"], label=r["method"])
        ax2.plot(r["depths"], label=r["method"])
        for s in r["depth_switch_steps"]:
            ax1.axvline(s, ls=":", lw=0.6, color="gray")
    ax1.set(xlabel="Optimiser step", ylabel="<H_C> (scaled)")
    ax2.set(xlabel="Optimiser step", ylabel="Circuit depth p")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.json")
