"""Phase 9: uncertainty reporting and the ranking-versus-value gap.

(a) Replicate-based confidence intervals: for each explicand, R independent
    runs of an estimator give a mean and a standard error per feature; the
    question is how often the exact value lies inside the resulting 95%
    interval. Permutation sampling is unbiased, KernelSHAP variants are not.
(b) The draft says rankings settle before values do. Using the Phase 1 raw
    runs, count how often a run has the exact top-3 and is still more than 5%
    off in relative error.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.data_model import ExperimentSetup
from src.estimators import ESTIMATOR_REGISTRY


def run_phase9(
    setup: ExperimentSetup,
    tabulated_cache: list,
    budgets: list[int] = (64, 128, 256, 512, 1024, 2048),
    replicates: int = 10,
    estimators: tuple[str, ...] = ("Permutation", "Antithetic Permutation", "Paired KernelSHAP"),
) -> dict:
    t0 = time.time()
    d = setup.d
    n_exp = setup.n_explicands
    print(f"\n--- Running Phase 9 (uncertainty reporting, R={replicates} replicates) ---")

    tcrit = stats.t.ppf(0.975, df=replicates - 1)
    rows = []
    for i in range(n_exp):
        v_table, phi_exact, _ = tabulated_cache[i]
        for est in estimators:
            fn = ESTIMATOR_REGISTRY[est]
            for m in budgets:
                runs = np.array([
                    fn(v_table=v_table, d=d, m=m, seed=100_000 + 1000 * i + r)
                    for r in range(replicates)
                ])
                if np.isnan(runs).any():
                    continue
                mean = runs.mean(axis=0)
                se = runs.std(axis=0, ddof=1) / np.sqrt(replicates)
                lo, hi = mean - tcrit * se, mean + tcrit * se
                covered = (phi_exact >= lo) & (phi_exact <= hi)
                z = (mean - phi_exact) / np.where(se > 0, se, np.nan)
                rows.append({
                    "explicand": i, "estimator": est, "budget": m,
                    "coverage": covered.mean(),
                    "mean_halfwidth_rel": float(np.mean(tcrit * se) / np.abs(phi_exact).mean()),
                    "mean_abs_z": float(np.nanmean(np.abs(z))),
                    "replicate_mean_rel_err": float(np.linalg.norm(mean - phi_exact)
                                                    / np.linalg.norm(phi_exact)),
                })
    df_ci = pd.DataFrame(rows)
    df_ci.to_csv("results/raw/phase9_ci_raw.csv", index=False)
    ci_agg = df_ci.groupby(["estimator", "budget"]).agg(
        coverage_mean=("coverage", "mean"),
        coverage_se=("coverage", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        halfwidth_rel=("mean_halfwidth_rel", "mean"),
        mean_abs_z=("mean_abs_z", "mean"),
        replicate_mean_rel_err=("replicate_mean_rel_err", "mean"),
    ).reset_index()
    ci_agg.to_csv("results/tables/phase9_ci_coverage.csv", index=False)

    # (b) ranking vs value gap from the Phase 1 raw runs
    raw1 = pd.read_csv("results/raw/phase1_raw.csv").dropna(subset=["rel_err"])
    gap_rows = []
    for (est, m), g in raw1.groupby(["estimator", "budget"]):
        perfect = g[g.top3 >= 1.0 - 1e-12]
        gap_rows.append({
            "estimator": est, "budget": m,
            "p_top3_perfect": len(perfect) / len(g),
            "p_err_over_5pct_given_perfect": (float((perfect.rel_err > 0.05).mean())
                                              if len(perfect) else np.nan),
            "median_err_given_perfect": (float(perfect.rel_err.median())
                                         if len(perfect) else np.nan),
            "p_err_over_5pct_all": float((g.rel_err > 0.05).mean()),
        })
    df_gap = pd.DataFrame(gap_rows)
    df_gap.to_csv("results/tables/phase9_ranking_vs_value_gap.csv", index=False)

    # Figure 8
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"Permutation": "#1f77b4", "Antithetic Permutation": "#ff7f0e",
              "KernelSHAP": "#2ca02c", "Paired KernelSHAP": "#d62728",
              "Multilinear Extension": "#9467bd"}
    markers = {"Permutation": "o", "Antithetic Permutation": "s", "KernelSHAP": "^",
               "Paired KernelSHAP": "D", "Multilinear Extension": "v"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    for est in estimators:
        s = ci_agg[ci_agg.estimator == est].sort_values("budget")
        ax1.errorbar(s.budget, s.coverage_mean, yerr=s.coverage_se, color=colors[est],
                     marker=markers[est], capsize=3, linewidth=1.8, label=est)
    ax1.axhline(0.95, color="gray", linestyle="--", label="Nominal 95%")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(list(budgets))
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(f"Budget per run $m$ ({replicates} replicates)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Coverage of exact $\\phi_i$ by the 95% interval", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Replicate-based confidence intervals", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True, loc="lower left")
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    for est in ESTIMATOR_REGISTRY:
        s = df_gap[df_gap.estimator == est].sort_values("budget")
        ax2.plot(s.budget, s.p_err_over_5pct_given_perfect, color=colors[est],
                 marker=markers[est], linewidth=1.8, label=est)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(list(budgets))
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.set_xlabel("Evaluation budget $m$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("P(relative error > 5% | exact top-3 recovered)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) A correct ranking does not imply accurate values", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, frameon=True)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/figure8_uncertainty_and_ranking_gap.png", dpi=300)
    plt.savefig("results/figures/figure8_uncertainty_and_ranking_gap.pdf")
    plt.close()

    print(ci_agg.to_string(index=False))
    print(df_gap.to_string(index=False))
    print(f"Phase 9 complete in {time.time() - t0:.2f} s")
    return {"ci": ci_agg, "gap": df_gap}
