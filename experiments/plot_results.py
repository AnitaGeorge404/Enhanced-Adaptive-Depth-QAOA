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
import math
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


def plot_reduction_heatmap(e1_path: Path | str, out_dir: Path | str | None = None) -> None:
    """Plot reduction heatmap: tightness x density -> mean qubit reduction percentage."""
    path = Path(e1_path)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tightness_levels = data.get("tightness_levels", [0.4, 0.6, 0.8, 1.0])
    densities = data.get("densities", [0.4, 0.6, 0.8])
    grid = data.get("grid", [])

    matrix = np.zeros((len(tightness_levels), len(densities)))
    for item in grid:
        t_val = item["tightness"]
        d_val = item["density"]
        if t_val in tightness_levels and d_val in densities:
            t_idx = tightness_levels.index(t_val)
            d_idx = densities.index(d_val)
            matrix[t_idx, d_idx] = item["mean_qubit_reduction_pct"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    cax = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(50, np.max(matrix) + 5))
    fig.colorbar(cax, label="Mean Qubit Reduction (%)")

    ax.set_xticks(range(len(densities)))
    ax.set_xticklabels([f"{d:.1f}" for d in densities])
    ax.set_yticks(range(len(tightness_levels)))
    ax.set_yticklabels([f"{t:.1f}" for t in tightness_levels])

    ax.set_xlabel("Graph Edge Density ($p_{edge}$)", fontsize=11)
    ax.set_ylabel("Resource Constraint Tightness Factor ($\\alpha$)", fontsize=11)
    ax.set_title("Instance Reduction: Mean Qubit Reduction (%)\n(Rules R0-R4 Optimality-Preserving)", fontsize=12)

    for i in range(len(tightness_levels)):
        for j in range(len(densities)):
            val = matrix[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    fig.tight_layout()
    out_file = out_dir / "reduction_heatmap.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved reduction heatmap -> {out_file}")


def plot_curvature_comparison(e2_path: Path | str, out_dir: Path | str | None = None) -> None:
    """Plot E2 comparison: steps and cumulative CNOTs across triggers with bootstrap 95% CIs."""
    path = Path(e2_path)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = data.get("runs", [])
    methods = ["trigger_patience", "trigger_enhanced", "trigger_curvature"]
    method_labels = ["Patience Trigger", "Enhanced Slope Trigger", "Curvature Trigger"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    steps_means, steps_errs = [], []
    cnot_means, cnot_errs = [], []

    for m in methods:
        m_runs = [r for r in runs if r["method"] == m]
        st = [r["steps"] for r in m_runs]
        cn = [r["cumulative_cnots"] for r in m_runs]

        steps_means.append(np.mean(st) if st else 0)
        cnot_means.append(np.mean(cn) if cn else 0)

        # Standard error / 95% CI approximation
        steps_errs.append(1.96 * np.std(st, ddof=1) / math.sqrt(len(st)) if len(st) > 1 else 0)
        cnot_errs.append(1.96 * np.std(cn, ddof=1) / math.sqrt(len(cn)) if len(cn) > 1 else 0)

    x = np.arange(len(methods))
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4"]

    ax1.bar(x, steps_means, yerr=steps_errs, capsize=5, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=15, ha="right")
    ax1.set_ylabel("Total Optimization Steps to Convergence", fontsize=11)
    ax1.set_title("Optimization Steps (Mean ± 95% CI)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, cnot_means, yerr=cnot_errs, capsize=5, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=15, ha="right")
    ax2.set_ylabel("Cumulative CNOT Gate Count", fontsize=11)
    ax2.set_title("Circuit CNOT Gate Count (Mean ± 95% CI)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    out_file = out_dir / "curvature_steps_cnot.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved curvature comparison -> {out_file}")


def plot_noise_crossover(e3_path: Path | str, out_dir: Path | str | None = None) -> None:
    """Plot E3 crossover analysis: noisy approximation ratio vs error rate eps2 with crossover eps*."""
    path = Path(e3_path)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = data.get("results", [])
    eps_star = data.get("crossover_eps_star")

    eps_vals = [r["eps2"] for r in results]
    r_p1 = [r["depth_ratios"]["p=1"] for r in results]
    r_p2 = [r["depth_ratios"]["p=2"] for r in results]
    r_p3 = [r["depth_ratios"]["p=3"] for r in results]
    r_p4 = [r["depth_ratios"]["p=4"] for r in results]
    r_na = [r["noise_aware_ratio"] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(eps_vals, r_p1, "o--", label="Fixed Depth p=1", color="#333333", lw=1.5)
    ax.plot(eps_vals, r_p2, "s--", label="Fixed Depth p=2", color="#1f77b4", lw=1.5)
    ax.plot(eps_vals, r_p3, "^--", label="Fixed Depth p=3", color="#ff7f0e", lw=1.5)
    ax.plot(eps_vals, r_p4, "d--", label="Fixed Depth p=4", color="#d62728", lw=1.5)
    ax.plot(eps_vals, r_na, "*-", label="Noise-Aware DDQAOA (Ours)", color="#2ca02c", lw=2.5, markersize=9)

    if eps_star is not None:
        ax.axvline(eps_star, color="purple", linestyle=":", lw=2, label=f"Crossover $\\varepsilon^* \\approx {eps_star:.4f}$")
        ax.text(eps_star * 1.05, min(r_p4) + 0.05, f"Crossover $\\varepsilon^*$\n(p=2 degrades below p=1)", color="purple", fontsize=9)

    ax.set_xlabel("Two-Qubit Gate Depolarizing Error Rate ($\\varepsilon_2$)", fontsize=11)
    ax.set_ylabel("Noisy Approximation Ratio ($r_{\\text{noisy}}$)", fontsize=11)
    ax.set_title("Noise Crossover Analysis & Noise-Aware Stopping (Feature 3)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out_file = out_dir / "noise_crossover.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved noise crossover plot -> {out_file}")


def plot_cost_breakdown(e4_path: Path | str, out_dir: Path | str | None = None) -> None:
    """Plot E4 cost breakdown: stacked bar chart of Optimization CNOT-shots vs Sampling TTS_99."""
    path = Path(e4_path)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = data.get("records", [])
    configs = [r["configuration"] for r in records]
    labels = [
        "Std Energy\n(Autodiff)",
        "Post-Selected\n(Autodiff)",
        "Std Energy\n(Param-Shift)",
        "Std Energy\n(SPSA)",
    ][:len(configs)]

    opt_costs = [r["optimization_cnot_shots"] for r in records]
    samp_costs = [r["sampling_cnot_shots"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    width = 0.55

    p1 = ax.bar(x, opt_costs, width, label="Optimization CNOT-Shots", color="#4C72B0", alpha=0.85)
    p2 = ax.bar(x, samp_costs, width, bottom=opt_costs, label="Sampling CNOT-Shots ($TTS_{99}$)", color="#E15759", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Total Quantum CNOT-Shots ($log_{10}$ scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("End-to-End Quantum Resource Cost Breakdown (Feature 4)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(fontsize=10)

    for i in range(len(configs)):
        tot = opt_costs[i] + samp_costs[i]
        p_opt = records[i]["p_optimal"]
        ax.text(i, tot * 1.25, f"$P_{{opt}}$={p_opt*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out_file = out_dir / "cost_breakdown.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved cost breakdown plot -> {out_file}")


def plot_all_research_figures(results_dir: Path | str = Path("results")) -> None:
    """Check for research experiment JSON files in results_dir and generate all 4 plots."""
    rdir = Path(results_dir)
    e1_file = rdir / "e1_reduction.json"
    e2_file = rdir / "e2_curvature.json"
    e3_file = rdir / "e3_noise.json"
    e4_file = rdir / "e4_cost.json"

    if e1_file.exists():
        plot_reduction_heatmap(e1_file, rdir)
    if e2_file.exists():
        plot_curvature_comparison(e2_file, rdir)
    if e3_file.exists():
        plot_noise_crossover(e3_file, rdir)
    if e4_file.exists():
        plot_cost_breakdown(e4_file, rdir)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.json"
    p = Path(target)
    if p.name == "e1_reduction.json":
        plot_reduction_heatmap(p)
    elif p.name == "e2_curvature.json":
        plot_curvature_comparison(p)
    elif p.name == "e3_noise.json":
        plot_noise_crossover(p)
    elif p.name == "e4_cost.json":
        plot_cost_breakdown(p)
    elif p.is_dir():
        plot_all_research_figures(p)
    else:
        main(target)
        plot_all_research_figures(p.parent)

