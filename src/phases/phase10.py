"""Phase 10: is the multilinear efficiency gap quadrature bias or sampling noise?

The draft attributes the multilinear estimator's efficiency gap to coarse
quadrature (bias that more samples cannot remove). Phases 1 and 4 show the gap
falling with the budget at roughly the Monte Carlo rate, which points to noise.
Here each estimator is run R times per explicand; the mean over runs estimates
the expected output, so that

    squared error of one run = squared bias + variance

can be separated, and the efficiency gap of the *averaged* estimate (pure bias)
can be compared with the average gap of single runs (bias plus noise).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.estimators import multilinear_extension, paired_kernel_shap


def run_phase10(
    setup: ExperimentSetup,
    tabulated_cache: list,
    budgets: list[int] = (512, 1024, 2048, 4096, 8192),
    K_list: list[int] = (4, 8, 32),
    replicates: int = 100,
) -> dict:
    t0 = time.time()
    d = setup.d
    n_exp = setup.n_explicands
    print(f"\n--- Running Phase 10 (multilinear bias/variance, R={replicates}, K={list(K_list)}) ---")

    configs = [(f"Multilinear K={K}", multilinear_extension, {"K": K}) for K in K_list]
    configs.append(("Paired KernelSHAP", paired_kernel_shap, {}))

    rows = []
    for i in range(n_exp):
        v_table, phi, eff_target = tabulated_cache[i]
        norm_phi = np.linalg.norm(phi)
        for name, fn, kw in configs:
            for m in budgets:
                runs = np.array([
                    fn(v_table=v_table, d=d, m=m, seed=200_000 + 1000 * i + r, **kw)
                    for r in range(replicates)
                ])
                if np.isnan(runs).any():
                    continue
                mean = runs.mean(axis=0)
                single_err = np.linalg.norm(runs - phi, axis=1) / norm_phi
                noise = np.sqrt(np.mean(np.sum((runs - mean) ** 2, axis=1))) / norm_phi
                bias = np.linalg.norm(mean - phi) / norm_phi
                rows.append({
                    "explicand": i, "estimator": name, "budget": m,
                    "single_run_rel_err": single_err.mean(),
                    "bias_rel": bias,
                    "noise_rel": noise,
                    "bias_noise_floor": noise / np.sqrt(replicates),
                    "eff_gap_single_run": np.abs(runs.sum(axis=1) - eff_target).mean(),
                    "eff_gap_of_mean": abs(mean.sum() - eff_target),
                })
        print(f"  explicand {i + 1}/{n_exp} done ({time.time() - t0:.0f} s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/phase10_raw.csv", index=False)

    def se(s):
        return s.std(ddof=1) / np.sqrt(len(s))

    cols = ["single_run_rel_err", "bias_rel", "noise_rel", "bias_noise_floor",
            "eff_gap_single_run", "eff_gap_of_mean"]
    agg = df.groupby(["estimator", "budget"])[cols].agg(["mean", se])
    agg.columns = [f"{a}_{'se' if b == 'se' else 'mean'}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv("results/tables/phase10_multilinear_bias_variance.csv", index=False)

    # log-log slopes of the single-run efficiency gap and of the averaged-estimate gap
    slope_rows = []
    for name, g in agg.groupby("estimator"):
        g = g.sort_values("budget")
        lm = np.log10(g.budget.values.astype(float))
        for col in ("eff_gap_single_run_mean", "eff_gap_of_mean_mean", "noise_rel_mean", "bias_rel_mean"):
            y = np.log10(np.maximum(g[col].values, 1e-12))
            slope_rows.append({"estimator": name, "quantity": col.replace("_mean", ""),
                               "loglog_slope": np.polyfit(lm, y, 1)[0]})
    df_slopes = pd.DataFrame(slope_rows)
    df_slopes.to_csv("results/tables/phase10_slopes.csv", index=False)

    # Figure 9
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    palette = {"Multilinear K=4": "#9467bd", "Multilinear K=8": "#8c564b",
               "Multilinear K=32": "#e377c2", "Paired KernelSHAP": "#d62728"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    for name in palette:
        g = agg[agg.estimator == name].sort_values("budget")
        if g.empty:
            continue
        ax1.errorbar(g.budget, g.eff_gap_single_run_mean, yerr=g.eff_gap_single_run_se,
                     color=palette[name], marker="o", capsize=3, linewidth=1.6,
                     label=f"{name}, single run")
        if name.startswith("Multilinear"):
            ax1.plot(g.budget, g.eff_gap_of_mean_mean, color=palette[name], marker="x",
                     linestyle="--", linewidth=1.2, label=f"{name}, average of {replicates} runs")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Evaluation budget $m$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Efficiency gap (log-odds)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) The gap is mostly run-to-run noise", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7.5, frameon=True, ncol=1)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    for name in palette:
        g = agg[agg.estimator == name].sort_values("budget")
        if g.empty:
            continue
        ax2.plot(g.budget, g.noise_rel_mean, color=palette[name], marker="o", linewidth=1.6,
                 label=f"{name}: noise")
        ax2.plot(g.budget, g.bias_rel_mean, color=palette[name], marker="s", linestyle="--",
                 linewidth=1.2, label=f"{name}: bias")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Evaluation budget $m$", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Relative $\ell_2$ magnitude", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Bias and noise of the estimates", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7, frameon=True, ncol=2)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/figure9_multilinear_bias_variance.png", dpi=300)
    plt.savefig("results/figures/figure9_multilinear_bias_variance.pdf")
    plt.close()

    print(agg.round(5).to_string(index=False))
    print(df_slopes.round(3).to_string(index=False))
    print(f"Phase 10 complete in {time.time() - t0:.2f} s")
    return {"agg": agg, "slopes": df_slopes}
