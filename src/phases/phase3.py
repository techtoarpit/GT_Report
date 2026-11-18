"""Phase 3: Scale up to d=16 (65,536 coalitions)."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley
from src.estimators import ESTIMATOR_REGISTRY
from src.metrics import (
    compute_relative_error,
    compute_top3_agreement,
    compute_efficiency_gap,
)


def run_phase3(
    budgets: list[int] = [128, 256, 512, 1024, 2048, 4096, 8192],
    n_explicands: int = 5,
    n_seeds: int = 10,
    quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute Phase 3 benchmark at d=16:
    - 65,536 coalitions exact per explicand
    - 5 explicands, 10 seeds per explicand (seed = 1000 * i + s)
    - Budgets [128, 256, 512, 1024, 2048, 4096, 8192]
    - Saves results/raw/phase3_raw.csv, results/tables/phase3_table.csv, results/figures/figure3_scale_d16.{png,pdf}
    """
    t0_phase = time.time()
    d = 16
    if quick:
        n_explicands = 2
        n_seeds = 3
        budgets = [128, 256, 512, 1024]
        
    print(f"\n--- Running Phase 3: Scale up to d=16 ({n_explicands} explicands, {n_seeds} seeds, budgets={budgets}) ---")
    setup = ExperimentSetup(d=d, n_explicands=n_explicands)
    
    records = []
    estimators_order = [
        "Permutation",
        "Antithetic Permutation",
        "KernelSHAP",
        "Paired KernelSHAP",
        "Multilinear Extension"
    ]
    
    for i in range(n_explicands):
        x = setup.explicands[i]
        print(f"Explicand {i+1}/{n_explicands} (d=16): Tabulating 65,536 coalitions...")
        v_table, tab_time, fwd_passes = tabulate_value_function(
            setup.model, x, setup.background, chunk_size=1024
        )
        phi_exact, eff_gap_exact, exact_time = compute_exact_shapley(v_table, d)
        print(f"  Tabulated in {tab_time:.2f}s ({fwd_passes} passes), exact Shapley in {exact_time*1000:.1f}ms (eff gap={eff_gap_exact:.2e})")
        
        v_empty = v_table[0]
        v_N = v_table[-1]
        eff_target = v_N - v_empty
        
        for s in range(n_seeds):
            seed = 1000 * i + s
            for est_name, est_func in ESTIMATOR_REGISTRY.items():
                for m in budgets:
                    phi_hat = est_func(v_table=v_table, d=d, m=m, seed=seed)
                    rel_err = compute_relative_error(phi_hat, phi_exact)
                    top3 = compute_top3_agreement(phi_hat, phi_exact)
                    eff_gap = compute_efficiency_gap(phi_hat, eff_target)
                    
                    records.append({
                        "explicand": i,
                        "seed": seed,
                        "estimator": est_name,
                        "budget": m,
                        "rel_err": rel_err,
                        "top3": top3,
                        "eff_gap": eff_gap
                    })
                    
    df_raw = pd.DataFrame(records)
    df_raw.to_csv("results/raw/phase3_raw.csv", index=False)
    print(f"Phase 3 raw data saved to results/raw/phase3_raw.csv ({len(df_raw)} rows)")
    
    # Aggregation
    df_by_explicand = df_raw.groupby(["estimator", "budget", "explicand"]).agg(
        rel_err=("rel_err", "mean"),
        top3=("top3", "mean"),
        eff_gap=("eff_gap", "mean")
    ).reset_index()
    
    def calc_se(series):
        if series.dropna().empty or len(series.dropna()) <= 1:
            return 0.0
        return float(series.dropna().std(ddof=1) / np.sqrt(len(series.dropna())))

    df_agg = df_by_explicand.groupby(["estimator", "budget"]).agg(
        rel_err_mean=("rel_err", "mean"),
        rel_err_se=("rel_err", calc_se),
        top3_mean=("top3", "mean"),
        top3_se=("top3", calc_se),
        eff_gap_mean=("eff_gap", "mean"),
        eff_gap_se=("eff_gap", calc_se),
    ).reset_index()
    
    # Table format
    phase3_rows = []
    for est in estimators_order:
        sub = df_agg[df_agg["estimator"] == est]
        row_dict = {"Estimator": est}
        for b in budgets:
            match = sub[sub["budget"] == b]
            if not match.empty and not np.isnan(match["rel_err_mean"].values[0]):
                m_val = match["rel_err_mean"].values[0]
                se_val = match["rel_err_se"].values[0]
                row_dict[f"m={b}"] = f"{m_val:.4f} +/- {se_val:.4f}"
            else:
                row_dict[f"m={b}"] = "NaN"
        phase3_rows.append(row_dict)
        
    df_table3 = pd.DataFrame(phase3_rows)
    df_table3.to_csv("results/tables/phase3_table.csv", index=False)
    print("Saved Phase 3 table to results/tables/phase3_table.csv")
    
    # Check ordering at max budget
    b_max = max(budgets)
    sub_max = df_agg[df_agg["budget"] == b_max].sort_values("rel_err_mean")
    ordering_persists = (sub_max["estimator"].iloc[0] == "Paired KernelSHAP")
    print(f"At m={b_max}, lowest error method: {sub_max['estimator'].iloc[0]} (Ordering persists: {ordering_persists})")
    
    # --- Figure 3: d=16 scaling ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    colors = {
        "Permutation": "#1f77b4",
        "Antithetic Permutation": "#ff7f0e",
        "KernelSHAP": "#2ca02c",
        "Paired KernelSHAP": "#d62728",
        "Multilinear Extension": "#9467bd"
    }
    markers = {
        "Permutation": "o",
        "Antithetic Permutation": "s",
        "KernelSHAP": "^",
        "Paired KernelSHAP": "D",
        "Multilinear Extension": "v"
    }
    
    for est in estimators_order:
        sub = df_agg[df_agg["estimator"] == est].dropna(subset=["rel_err_mean"]).sort_values("budget")
        if not sub.empty:
            ax1.errorbar(
                sub["budget"],
                sub["rel_err_mean"],
                yerr=sub["rel_err_se"],
                label=est,
                color=colors[est],
                marker=markers[est],
                capsize=3,
                linewidth=1.8,
                markersize=6
            )
            
    min_b = min(budgets)
    perm_anchor = df_agg[(df_agg["estimator"] == "Permutation") & (df_agg["budget"] == min_b)]
    if not perm_anchor.empty and not np.isnan(perm_anchor["rel_err_mean"].values[0]):
        anchor_val = perm_anchor["rel_err_mean"].values[0]
        m_ref = np.array(budgets, dtype=float)
        ref_line = anchor_val * np.sqrt(min_b / m_ref)
        ax1.plot(m_ref, ref_line, "k--", alpha=0.7, label=r"$O(m^{-1/2})$ reference")
        
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10)
    ax1.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Relative Error $\Vert\hat{\phi} - \phi\Vert_2 / \Vert\phi\Vert_2$", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 3 (d=16): Convergence vs Budget", fontsize=12, fontweight="bold")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    
    for est in estimators_order:
        sub = df_agg[df_agg["estimator"] == est].dropna(subset=["top3_mean"]).sort_values("budget")
        if not sub.empty:
            ax2.errorbar(
                sub["budget"],
                sub["top3_mean"],
                yerr=sub["top3_se"],
                label=est,
                color=colors[est],
                marker=markers[est],
                capsize=3,
                linewidth=1.8,
                markersize=6
            )
            
    ax2.axhline(0.99, color="gray", linestyle="--", alpha=0.7, label="99% Agreement Target")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Top-3 Feature Overlap Fraction", fontsize=11, fontweight="bold")
    ax2.set_title("Phase 3 (d=16): Top-3 Ranking Fidelity", fontsize=12, fontweight="bold")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure3_scale_d16.png", dpi=300)
    plt.savefig("results/figures/figure3_scale_d16.pdf")
    plt.close()
    print("Saved Figure 3 to results/figures/figure3_scale_d16.{png,pdf}")
    
    elapsed_total = time.time() - t0_phase
    print(f"Phase 3 complete in {elapsed_total:.2f} s")
    
    return df_raw, df_agg, {
        "ordering_persists": bool(ordering_persists),
        "runtime": elapsed_total
    }
