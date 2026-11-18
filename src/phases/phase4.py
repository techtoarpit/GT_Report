"""Phase 4: Multilinear Extension grid resolution ablation."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley
from src.estimators import multilinear_extension
from src.metrics import compute_relative_error, compute_efficiency_gap


def run_phase4(
    setup: ExperimentSetup,
    K_list: list[int] = [4, 8, 16, 32],
    budgets: list[int] = [64, 128, 256, 512, 1024, 2048],
    n_seeds: int = 20,
    tabulated_cache: list | None = None,
    quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute Phase 4:
    - Ablate K in {4, 8, 16, 32} for Multilinear Extension at d=12
    - Evaluate relative error and efficiency gap across budgets
    - Returns NaN where T_interior < K - 2
    - Saves results/raw/phase4_raw.csv, results/tables/phase4_multilinear_ablation.csv,
      and results/figures/figure4_multilinear_ablation.{png,pdf}
    """
    t0_phase = time.time()
    d = setup.d
    n_explicands = setup.n_explicands
    if quick:
        K_list = [4, 8]
        n_seeds = 5
        
    print(f"\n--- Running Phase 4: Multilinear Grid Ablation (K={K_list}, budgets={budgets}) ---")
    
    # Ensure we have tabulated cache
    if tabulated_cache is None:
        print("Tabulating value functions for Phase 4...")
        tabulated_cache = []
        for i in range(n_explicands):
            x = setup.explicands[i]
            v_table, _, _ = tabulate_value_function(setup.model, x, setup.background)
            phi_exact, _, _ = compute_exact_shapley(v_table, d)
            tabulated_cache.append((v_table, phi_exact, v_table[-1] - v_table[0]))
            
    records = []
    for i in range(n_explicands):
        v_table, phi_exact, eff_target = tabulated_cache[i]
        
        for s in range(n_seeds):
            seed = 1000 * i + s
            for K in K_list:
                for m in budgets:
                    phi_hat = multilinear_extension(v_table, d, m, seed, K=K)
                    rel_err = compute_relative_error(phi_hat, phi_exact)
                    eff_gap = compute_efficiency_gap(phi_hat, eff_target)
                    
                    records.append({
                        "explicand": i,
                        "seed": seed,
                        "K": K,
                        "budget": m,
                        "rel_err": rel_err,
                        "eff_gap": eff_gap
                    })
                    
    df_raw = pd.DataFrame(records)
    df_raw.to_csv("results/raw/phase4_raw.csv", index=False)
    print(f"Phase 4 raw data saved to results/raw/phase4_raw.csv ({len(df_raw)} rows)")
    
    # Aggregation
    df_by_explicand = df_raw.groupby(["K", "budget", "explicand"]).agg(
        rel_err=("rel_err", "mean"),
        eff_gap=("eff_gap", "mean")
    ).reset_index()
    
    def calc_se(series):
        if series.dropna().empty or len(series.dropna()) <= 1:
            return 0.0
        return float(series.dropna().std(ddof=1) / np.sqrt(len(series.dropna())))

    df_agg = df_by_explicand.groupby(["K", "budget"]).agg(
        rel_err_mean=("rel_err", "mean"),
        rel_err_se=("rel_err", calc_se),
        eff_gap_mean=("eff_gap", "mean"),
        eff_gap_se=("eff_gap", calc_se),
    ).reset_index()
    
    # Save formatted table
    table4_rows = []
    for K in K_list:
        sub = df_agg[df_agg["K"] == K]
        for b in budgets:
            match = sub[sub["budget"] == b]
            if not match.empty and not np.isnan(match["rel_err_mean"].values[0]):
                m_err = match["rel_err_mean"].values[0]
                se_err = match["rel_err_se"].values[0]
                m_eff = match["eff_gap_mean"].values[0]
                se_eff = match["eff_gap_se"].values[0]
                table4_rows.append({
                    "K": K,
                    "budget": b,
                    "rel_err": f"{m_err:.4f} +/- {se_err:.4f}",
                    "eff_gap": f"{m_eff:.4f} +/- {se_eff:.4f}",
                    "rel_err_numeric": m_err,
                    "eff_gap_numeric": m_eff
                })
            else:
                table4_rows.append({
                    "K": K,
                    "budget": b,
                    "rel_err": "NaN (budget too small)",
                    "eff_gap": "NaN (budget too small)",
                    "rel_err_numeric": np.nan,
                    "eff_gap_numeric": np.nan
                })
                
    df_table4 = pd.DataFrame(table4_rows)
    df_table4.to_csv("results/tables/phase4_multilinear_ablation.csv", index=False)
    print("Saved Phase 4 table to results/tables/phase4_multilinear_ablation.csv")
    
    # --- Figure 4: Multilinear Ablation Plots ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    colors_K = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c", 32: "#d62728"}
    markers_K = {4: "o", 8: "s", 16: "^", 32: "D"}
    
    for K in K_list:
        sub = df_agg[df_agg["K"] == K].dropna(subset=["rel_err_mean"]).sort_values("budget")
        if not sub.empty:
            ax1.errorbar(
                sub["budget"],
                sub["rel_err_mean"],
                yerr=sub["rel_err_se"],
                label=f"K = {K}",
                color=colors_K.get(K, "black"),
                marker=markers_K.get(K, "o"),
                capsize=3,
                linewidth=1.8,
                markersize=6
            )
            ax2.errorbar(
                sub["budget"],
                sub["eff_gap_mean"],
                yerr=sub["eff_gap_se"],
                label=f"K = {K}",
                color=colors_K.get(K, "black"),
                marker=markers_K.get(K, "o"),
                capsize=3,
                linewidth=1.8,
                markersize=6
            )
            
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10)
    ax1.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Relative Error (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Relative Error vs Grid Size $K$", fontsize=12, fontweight="bold")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.legend(frameon=True, fontsize=9.5)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=10)
    ax2.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Efficiency Gap (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Efficiency Gap vs Grid Size $K$", fontsize=12, fontweight="bold")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure4_multilinear_ablation.png", dpi=300)
    plt.savefig("results/figures/figure4_multilinear_ablation.pdf")
    plt.close()
    print("Saved Figure 4 to results/figures/figure4_multilinear_ablation.{png,pdf}")
    
    elapsed_total = time.time() - t0_phase
    print(f"Phase 4 complete in {elapsed_total:.2f} s")
    
    return df_raw, df_agg, {
        "runtime": elapsed_total
    }
