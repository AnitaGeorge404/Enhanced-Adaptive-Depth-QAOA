"""Publication-quality plots and statistical reports from benchmark JSON.

Generates:
1. Boxplots of approximation ratio and success probability across instances.
2. Quality-versus-resource Pareto curves (Approximation Ratio vs Cumulative CNOTs).
3. Circuit depth and energy convergence trajectories.
4. Ablation contribution comparisons.
5. Markdown statistical validation report with paired hypothesis tests and bootstrap CIs.

Usage:
    python experiments/plot_results.py results/benchmark.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(path: str) -> None:
    data = json.loads(Path(path).read_text())
    runs = data.get("runs", data if isinstance(data, list) else [])
    out_dir = Path(path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_method = defaultdict(list)
    for r in runs:
        if r.get("status", "SUCCESS") == "SUCCESS":
            by_method[r["method"]].append(r)

    if not by_method:
        print("No successful runs found.")
        return

    # Prioritize standard baseline and enhanced at top of ordering
    priority = ["ddqaoa", "ddqaoa_enhanced", "p=3", "p=5", "p=10", "p=15"]
    methods = sorted(
        by_method.keys(),
        key=lambda m: (priority.index(m) if m in priority else 100, m)
    )

    # 1. Summary Table to console
    base = np.mean([r["cumulative_cnots"] for r in by_method["ddqaoa"]]) if "ddqaoa" in by_method else None
    print(f"{'method':22s} {'mean r':>8s} {'med r':>8s} {'mean Ps':>8s} {'med Ps':>8s} {'cumCNOT':>12s} {'vs DD':>7s}")
    print("-" * 75)

    summary_rows = []
    for m in methods:
        rs = np.array([r["approx_ratio"] for r in by_method[m]])
        ps = np.array([r["success_prob"] for r in by_method[m]])
        cn = np.mean([r["cumulative_cnots"] for r in by_method[m]])
        depths = np.mean([r["final_depth"] for r in by_method[m]])
        rel = f"{cn / base:.2f}x" if base else "-"
        print(f"{m:22s} {rs.mean():8.4f} {np.median(rs):8.4f} {ps.mean():8.4f} {np.median(ps):8.4f} {cn:12,.0f} {rel:>7s}")
        summary_rows.append({
            "method": m,
            "mean_r": float(rs.mean()),
            "std_r": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
            "median_r": float(np.median(rs)),
            "mean_ps": float(ps.mean()),
            "std_ps": float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0,
            "mean_cnot": float(cn),
            "mean_depth": float(depths),
            "rel_cnot": rel,
        })

    # 2. Boxplots: Approximation Ratio and Success Probability
    for key, label in [("approx_ratio", "Approximation Ratio r"), ("success_prob", "Success Probability P_succ")]:
        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 4.5))
        box_data = [[r[key] for r in by_method[m]] for m in methods]
        bp = ax.boxplot(box_data, tick_labels=methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel("Method", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}_box.png", dpi=200)
        plt.close(fig)

    # 3. Pareto Frontier: Approximation Ratio vs Cumulative CNOTs
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for idx, m in enumerate(methods):
        mean_cnots = np.mean([r["cumulative_cnots"] for r in by_method[m]])
        mean_r = np.mean([r["approx_ratio"] for r in by_method[m]])
        std_r = np.std([r["approx_ratio"] for r in by_method[m]], ddof=1) if len(by_method[m]) > 1 else 0.0
        ax.errorbar(
            mean_cnots,
            mean_r,
            yerr=std_r,
            fmt='o',
            color=colors[idx],
            label=m,
            markersize=8,
            capsize=4,
            elinewidth=1.5,
        )
    ax.set_xlabel("Cumulative CNOT Gate Count (lower is better)", fontsize=11)
    ax.set_ylabel("Mean Approximation Ratio r (higher is better)", fontsize=11)
    ax.set_title("Quality-versus-Resource Trade-off (Pareto Evaluation)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", dpi=200)
    plt.close(fig)

    # 4. Trajectories for the first instance
    first_inst = runs[0].get("instance", 0)
    first_runs = [r for r in runs if r.get("instance") == first_inst and r.get("status", "SUCCESS") == "SUCCESS"]
    if first_runs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in first_runs:
            m = r["method"]
            energies = r.get("energies", [])
            depths = r.get("depths", [])
            lw = 2.0 if "enhanced" in m else (1.5 if m == "ddqaoa" else 1.0)
            if energies:
                ax1.plot(energies, label=m, lw=lw)
            if depths:
                ax2.plot(depths, label=m, lw=lw)
            for s in r.get("depth_switch_steps", []):
                if m in ("ddqaoa", "ddqaoa_enhanced"):
                    ax1.axvline(s, ls=":", lw=0.6, color="gray", alpha=0.7)
        ax1.set(xlabel="Optimiser Step", ylabel="Energy <H_C> (scaled)", title="Energy Convergence")
        ax2.set(xlabel="Optimiser Step", ylabel="Circuit Depth p", title="Circuit Depth Progression")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectories.png", dpi=200)
        plt.close(fig)

    # 5. Ablation Bar Chart (if ablation runs present)
    ablations = [m for m in methods if m.startswith("abl_")]
    if ablations and "ddqaoa_enhanced" in by_method:
        fig, ax = plt.subplots(figsize=(8, 4))
        comp_methods = ["ddqaoa", "ddqaoa_enhanced"] + ablations
        r_means = [np.mean([r["approx_ratio"] for r in by_method[m]]) for m in comp_methods]
        r_stds = [np.std([r["approx_ratio"] for r in by_method[m]], ddof=1) if len(by_method[m]) > 1 else 0.0 for m in comp_methods]
        y_pos = np.arange(len(comp_methods))
        ax.barh(y_pos, r_means, xerr=r_stds, align='center', alpha=0.8, color=['#888888', '#2ca02c'] + ['#1f77b4'] * len(ablations), capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comp_methods)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Approximation Ratio r", fontsize=11)
        ax.set_title("Ablation Study: Contribution of Each Improvement", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_comparison.png", dpi=200)
        plt.close(fig)

    # 6. Markdown Statistical Report
    report_lines = [
        "# Experimental Benchmark & Statistical Analysis Report",
        "",
        f"**Source Data:** `{path}`  ",
        f"**Evaluated Methods:** {', '.join(methods)}  ",
        f"**Total Run Records:** {len(runs)}  ",
        "",
        "## Summary Results Table",
        "",
        "| Method | Mean Approximation Ratio | Median Ratio | Mean Success Prob | Cumulative CNOTs | Relative CNOTs vs DDQAOA | Mean Depth |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| `{row['method']}` | {row['mean_r']:.4f} ± {row['std_r']:.4f} | {row['median_r']:.4f} | "
            f"{row['mean_ps']:.4f} ± {row['std_ps']:.4f} | {row['mean_cnot']:,.0f} | {row['rel_cnot']} | {row['mean_depth']:.1f} |"
        )

    # Attach paired analysis if available
    analysis = data.get("analysis", {})
    paired = analysis.get("paired_against_baseline", {})
    if paired:
        report_lines.extend([
            "",
            "## Paired Hypothesis Tests & Bootstrap Confidence Intervals (vs `ddqaoa`)",
            "",
            "| Comparison (Method vs `ddqaoa`) | Mean Δr (95% Bootstrap CI) | Paired t-test p-value | Wilcoxon p-value | Cohen's d | Mean ΔCNOT |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for m, pdata in paired.items():
            r_ci = pdata["approx_ratio"]["ci"]
            r_test = pdata["approx_ratio"]["test"]
            c_ci = pdata["cumulative_cnots"]["ci"]
            ci_str = f"{r_ci['mean_diff']:+.4f} [{r_ci['ci_lower']:+.4f}, {r_ci['ci_upper']:+.4f}]"
            t_p = f"{r_test['t_pvalue']:.4f}" if not np.isnan(r_test['t_pvalue']) else "N/A"
            w_p = f"{r_test['wilcoxon_pvalue']:.4f}" if not np.isnan(r_test['wilcoxon_pvalue']) else "N/A"
            d_val = f"{r_test['cohens_d']:+.2f}" if not np.isnan(r_test['cohens_d']) else "N/A"
            c_str = f"{c_ci['mean_diff']:+,.0f}"
            report_lines.append(f"| `{m}` | {ci_str} | {t_p} | {w_p} | {d_val} | {c_str} |")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved figures and statistical report -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.json")
