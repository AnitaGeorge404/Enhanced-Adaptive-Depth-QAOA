"""Statistical analysis and rigorous experimental validation (Improvement 5).

Provides paired hypothesis tests, bootstrap confidence intervals, and
scientific evaluation metrics across instances and random seeds.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def paired_bootstrap_ci(
    baseline: np.ndarray | list[float],
    enhanced: np.ndarray | list[float],
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate non-parametric paired bootstrap confidence intervals for the mean difference.

    Difference is defined as enhanced - baseline.
    """
    b = np.asarray(baseline, dtype=float)
    e = np.asarray(enhanced, dtype=float)
    if len(b) != len(e):
        raise ValueError(f"Length mismatch: {len(b)} vs {len(e)}")
    if len(b) == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    diffs = e - b
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_bootstraps, dtype=float)
    boot_medians = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
        boot_medians[i] = np.median(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    med_ci_lower = float(np.percentile(boot_medians, 100.0 * alpha))
    med_ci_upper = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha)))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median_diff": median_diff,
        "median_ci_lower": med_ci_lower,
        "median_ci_upper": med_ci_upper,
    }


def paired_statistical_tests(
    baseline: np.ndarray | list[float],
    enhanced: np.ndarray | list[float],
) -> dict[str, float | str]:
    """Perform paired Student's t-test and Wilcoxon signed-rank test.

    Computes effect size (Cohen's d).
    """
    b = np.asarray(baseline, dtype=float)
    e = np.asarray(enhanced, dtype=float)
    diffs = e - b
    n = len(diffs)

    if n < 2:
        return {
            "n": n,
            "mean_diff": float(np.mean(diffs)) if n == 1 else 0.0,
            "t_stat": float("nan"),
            "t_pvalue": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_pvalue": float("nan"),
            "cohens_d": float("nan"),
            "interpretation": "Insufficient samples",
        }

    # Cohen's d for paired samples
    std_diff = float(np.std(diffs, ddof=1))
    cohens_d = float(np.mean(diffs) / std_diff) if std_diff > 1e-12 else 0.0

    # Paired Student's t-test
    try:
        t_res = stats.ttest_rel(e, b)
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t_stat, t_pval = float("nan"), float("nan")

    # Wilcoxon signed-rank test
    try:
        if np.allclose(diffs, 0.0):
            w_stat, w_pval = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    return {
        "n": n,
        "mean_diff": float(np.mean(diffs)),
        "std_diff": std_diff,
        "t_stat": t_stat,
        "t_pvalue": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval,
        "cohens_d": cohens_d,
    }


def aggregate_benchmark_results(runs: list[dict], baseline_method: str = "ddqaoa") -> dict[str, Any]:
    """Group benchmark runs by method and compute summary statistics + paired tests."""
    from collections import defaultdict

    by_method = defaultdict(list)
    for r in runs:
        if r.get("status", "SUCCESS") == "SUCCESS":
            by_method[r["method"]].append(r)

    methods = sorted(by_method.keys(), key=lambda m: (m != baseline_method, len(m), m))
    summary: dict[str, Any] = {"methods": {}, "paired_against_baseline": {}}

    for m in methods:
        mr = by_method[m]
        r_vals = np.array([x["approx_ratio"] for x in mr])
        p_vals = np.array([x["success_prob"] for x in mr])
        c_vals = np.array([x["cumulative_cnots"] for x in mr])
        d_vals = np.array([x["final_depth"] for x in mr])
        step_vals = np.array([len(x.get("energies", [])) for x in mr])

        summary["methods"][m] = {
            "count": len(mr),
            "approx_ratio": {
                "mean": float(np.mean(r_vals)),
                "std": float(np.std(r_vals, ddof=1)) if len(r_vals) > 1 else 0.0,
                "median": float(np.median(r_vals)),
                "iqr": float(np.percentile(r_vals, 75) - np.percentile(r_vals, 25)),
            },
            "success_prob": {
                "mean": float(np.mean(p_vals)),
                "std": float(np.std(p_vals, ddof=1)) if len(p_vals) > 1 else 0.0,
                "median": float(np.median(p_vals)),
                "iqr": float(np.percentile(p_vals, 75) - np.percentile(p_vals, 25)),
            },
            "cumulative_cnots": {
                "mean": float(np.mean(c_vals)),
                "std": float(np.std(c_vals, ddof=1)) if len(c_vals) > 1 else 0.0,
                "median": float(np.median(c_vals)),
            },
            "final_depth": {
                "mean": float(np.mean(d_vals)),
                "median": float(np.median(d_vals)),
            },
            "steps": {
                "mean": float(np.mean(step_vals)),
                "median": float(np.median(step_vals)),
            },
        }

    # Paired comparisons against baseline_method
    if baseline_method in by_method:
        base_runs = {r["instance"]: r for r in by_method[baseline_method]}
        for m in methods:
            if m == baseline_method:
                continue
            curr_runs = {r["instance"]: r for r in by_method[m]}
            common_instances = sorted(set(base_runs.keys()) & set(curr_runs.keys()))
            if not common_instances:
                continue

            b_r = [base_runs[i]["approx_ratio"] for i in common_instances]
            e_r = [curr_runs[i]["approx_ratio"] for i in common_instances]
            b_p = [base_runs[i]["success_prob"] for i in common_instances]
            e_p = [curr_runs[i]["success_prob"] for i in common_instances]
            b_c = [base_runs[i]["cumulative_cnots"] for i in common_instances]
            e_c = [curr_runs[i]["cumulative_cnots"] for i in common_instances]

            r_ci = paired_bootstrap_ci(b_r, e_r)
            r_test = paired_statistical_tests(b_r, e_r)

            p_ci = paired_bootstrap_ci(b_p, e_p)
            p_test = paired_statistical_tests(b_p, e_p)

            c_ci = paired_bootstrap_ci(b_c, e_c)
            c_test = paired_statistical_tests(b_c, e_c)

            summary["paired_against_baseline"][m] = {
                "n_instances": len(common_instances),
                "approx_ratio": {"ci": r_ci, "test": r_test},
                "success_prob": {"ci": p_ci, "test": p_test},
                "cumulative_cnots": {"ci": c_ci, "test": c_test},
            }

    return summary
