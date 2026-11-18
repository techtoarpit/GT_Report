"""Phase 1: Main benchmark study."""
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


def run_phase1(
    setup: ExperimentSetup,
    budgets: list[int] = [64, 128, 256, 512, 1024, 2048],
    n_seeds: int = 20,
    quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute Phase 1 benchmark:
    - Tabulates 2^d coalitions for each explicand
    - Computes exact Shapley values
    - Evaluates 5 estimators across budgets and random seeds (seed = 1000 * i + s)
    - Aggregates by averaging seeds within explicand, then computing mean +/- se across explicands
    - Generates Table II, Top-3 agreement table, Figure 1 (PNG & PDF), and derived metrics
    """
    t0_phase = time.time()
    d = setup.d
    n_explicands = setup.n_explicands
    
    records = []
    tabulation_times = []
    exact_times = []
    
    # Store tabulated tables and exact values so downstream phases can reuse them
    tabulated_cache = []
    
    print(f"\n--- Running Phase 1 (d={d}, {n_explicands} explicands, {n_seeds} seeds, budgets={budgets}) ---")
    
    for i in range(n_explicands):
        x = setup.explicands[i]
        v_table, tab_time, fwd_passes = tabulate_value_function(setup.model, x, setup.background)
        phi_exact, eff_gap_exact, exact_time = compute_exact_shapley(v_table, d)
        
        tabulation_times.append(tab_time)
        exact_times.append(exact_time)
        tabulated_cache.append((v_table, phi_exact, v_table[-1] - v_table[0]))
        
        v_empty = v_table[0]
        v_N = v_table[-1]
        eff_target = v_N - v_empty
        
        print(f"Explicand {i+1}/{n_explicands}: Tabulated in {tab_time:.3f}s ({fwd_passes} passes), exact Shapley in {exact_time*1000:.2f}ms")
        
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
    df_raw.to_csv("results/raw/phase1_raw.csv", index=False)
    print(f"Phase 1 raw data saved to results/raw/phase1_raw.csv ({len(df_raw)} rows)")
    
    # --- Aggregation Protocol ---
    # 1. Average over seeds within each explicand
    df_by_explicand = df_raw.groupby(["estimator", "budget", "explicand"]).agg(
        rel_err=("rel_err", "mean"),
        top3=("top3", "mean"),
        eff_gap=("eff_gap", "mean")
    ).reset_index()
    
    # 2. Mean and standard error across explicands
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
    
    # --- Table II: relative error vs budget (mean +/- s.e.) and efficiency gap at m=2048 ---
    table2_rows = []
    estimators_order = [
        "Permutation",
        "Antithetic Permutation",
        "KernelSHAP",
        "Paired KernelSHAP",
        "Multilinear Extension"
    ]
    
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
                
        # Efficiency gap at max budget (m=2048 or max in budgets)
        b_max = max(budgets)
        match_max = sub[sub["budget"] == b_max]
        if not match_max.empty and not np.isnan(match_max["eff_gap_mean"].values[0]):
            eff_mean = match_max["eff_gap_mean"].values[0]
            eff_se = match_max["eff_gap_se"].values[0]
            row_dict[f"Eff_Gap_m={b_max}"] = f"{eff_mean:.4f} +/- {eff_se:.4f}"
        else:
            row_dict[f"Eff_Gap_m={b_max}"] = "NaN"
        table2_rows.append(row_dict)
        
    df_table2 = pd.DataFrame(table2_rows)
    df_table2.to_csv("results/tables/table2_main_benchmark.csv", index=False)
    df_table2.to_csv("results/tables/table_ii.csv", index=False)
    print("Saved Table II to results/tables/table2_main_benchmark.csv")
    
    # --- Top-3 Agreement Table ---
    top3_rows = []
    for est in estimators_order:
        sub = df_agg[df_agg["estimator"] == est]
        row_dict = {"Estimator": est}
        for b in budgets:
            match = sub[sub["budget"] == b]
            if not match.empty and not np.isnan(match["top3_mean"].values[0]):
                m_val = match["top3_mean"].values[0]
                se_val = match["top3_se"].values[0]
                row_dict[f"m={b}"] = f"{m_val:.4f} +/- {se_val:.4f}"
            else:
                row_dict[f"m={b}"] = "NaN"
        top3_rows.append(row_dict)
    df_top3 = pd.DataFrame(top3_rows)
    df_top3.to_csv("results/tables/top3_agreement.csv", index=False)
    print("Saved Top-3 agreement table to results/tables/top3_agreement.csv")
    
    # --- Derived Numbers ---
    derived = {}
    
    # 1. Log-log fitted slope per estimator: log(rel_err) ~ slope * log(m) + c
    slopes = {}
    for est in estimators_order:
        sub = df_agg[(df_agg["estimator"] == est) & (~df_agg["rel_err_mean"].isna())]
        if len(sub) >= 2:
            log_m = np.log10(sub["budget"].values)
            log_err = np.log10(sub["rel_err_mean"].values)
            slope, _ = np.polyfit(log_m, log_err, 1)
            slopes[est] = float(slope)
        else:
            slopes[est] = float("nan")
    derived["log_log_slopes"] = slopes
    
    # 2. % reduction of paired KernelSHAP vs KernelSHAP and vs Permutation at m=2048
    b_target = 2048 if 2048 in budgets else max(budgets)
    err_paired = df_agg[(df_agg["estimator"] == "Paired KernelSHAP") & (df_agg["budget"] == b_target)]["rel_err_mean"].values[0]
    err_ks = df_agg[(df_agg["estimator"] == "KernelSHAP") & (df_agg["budget"] == b_target)]["rel_err_mean"].values[0]
    err_perm = df_agg[(df_agg["estimator"] == "Permutation") & (df_agg["budget"] == b_target)]["rel_err_mean"].values[0]
    
    pct_red_vs_ks = float((err_ks - err_paired) / err_ks * 100.0)
    pct_red_vs_perm = float((err_perm - err_paired) / err_perm * 100.0)
    derived[f"paired_kernelshap_reduction_vs_kernelshap_at_m_{b_target}_pct"] = pct_red_vs_ks
    derived[f"paired_kernelshap_reduction_vs_permutation_at_m_{b_target}_pct"] = pct_red_vs_perm
    derived[f"paired_kernelshap_rel_err_at_m_{b_target}"] = float(err_paired)
    
    # 3. Budget at which paired KernelSHAP matches permutation's error at m=2048 (log-log interpolation)
    sub_paired = df_agg[(df_agg["estimator"] == "Paired KernelSHAP") & (~df_agg["rel_err_mean"].isna())].sort_values("budget")
    log_m_p = np.log10(sub_paired["budget"].values)
    log_err_p = np.log10(sub_paired["rel_err_mean"].values)
    target_log_err = np.log10(err_perm)
    
    # Linear interpolation in log-log space
    # log_m = interp(target_log_err, log_err_p, log_m_p)
    # Note: log_err_p is decreasing with log_m_p
    sort_idx = np.argsort(log_err_p)
    matched_log_m = float(np.interp(target_log_err, log_err_p[sort_idx], log_m_p[sort_idx]))
    matched_budget = float(10 ** matched_log_m)
    speedup = float(b_target / matched_budget)
    derived["paired_kernelshap_matching_budget_vs_perm2048"] = matched_budget
    derived["speedup_factor_vs_perm2048"] = speedup
    
    # 4. Smallest budget reaching top-3 agreement >= 0.99
    reach_099 = {}
    for est in estimators_order:
        sub = df_agg[(df_agg["estimator"] == est) & (~df_agg["top3_mean"].isna())].sort_values("budget")
        qualified = sub[sub["top3_mean"] >= 0.99]
        if not qualified.empty:
            reach_099[est] = int(qualified["budget"].iloc[0])
        else:
            reach_099[est] = "not reached"
    derived["smallest_budget_top3_ge_099"] = reach_099
    
    # 5. Exact cost
    avg_tab_time = float(np.mean(tabulation_times))
    avg_exact_time = float(np.mean(exact_times))
    exact_cost = {
        "coalitions": 1 << d,
        "forward_passes": (1 << d) * len(setup.background),
        "mean_tabulation_seconds": avg_tab_time,
        "mean_exact_calc_seconds": avg_exact_time,
        "total_exact_seconds_per_instance": avg_tab_time + avg_exact_time
    }
    derived["exact_cost"] = exact_cost
    
    # Save derived metrics
    with open("results/tables/phase1_derived_metrics.json", "w") as f:
        json.dump(derived, f, indent=2)
        
    # Also save as tabular CSV for easy reading
    derived_rows = [
        {"Metric": "Paired KernelSHAP relative error at m=2048", "Value": f"{err_paired:.5f}"},
        {"Metric": "Paired KernelSHAP % reduction vs KernelSHAP (m=2048)", "Value": f"{pct_red_vs_ks:.2f}%"},
        {"Metric": "Paired KernelSHAP % reduction vs Permutation (m=2048)", "Value": f"{pct_red_vs_perm:.2f}%"},
        {"Metric": "Matching budget where Paired KS equals Permutation at m=2048", "Value": f"{matched_budget:.1f}"},
        {"Metric": "Speedup factor of Paired KS over Permutation (m=2048)", "Value": f"{speedup:.2f}x"},
        {"Metric": "Exact evaluations (coalitions)", "Value": str(1 << d)},
        {"Metric": "Exact forward passes (coalitions x 100 bg)", "Value": str((1 << d) * len(setup.background))},
        {"Metric": "Measured total exact seconds per instance", "Value": f"{avg_tab_time + avg_exact_time:.3f} s"}
    ]
    for est, s_val in slopes.items():
        derived_rows.append({"Metric": f"Log-log slope: {est}", "Value": f"{s_val:.3f}" if not np.isnan(s_val) else "NaN"})
    for est, b_val in reach_099.items():
        derived_rows.append({"Metric": f"Smallest budget for top3 >= 0.99: {est}", "Value": str(b_val)})
    pd.DataFrame(derived_rows).to_csv("results/tables/phase1_derived_metrics.csv", index=False)
    print("Saved Phase 1 derived metrics to results/tables/phase1_derived_metrics.csv and .json")
    
    # --- Figure 1: Main Benchmark Plots ---
    # (a) Log-log relative error vs m with O(m^-1/2) reference line anchored at permutation m=64
    # (b) Top-3 agreement vs m
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
            
    # Reference line O(m^-1/2) anchored at permutation m=min_budget point
    min_b = min(budgets)
    perm_anchor_match = df_agg[(df_agg["estimator"] == "Permutation") & (df_agg["budget"] == min_b)]
    if not perm_anchor_match.empty and not np.isnan(perm_anchor_match["rel_err_mean"].values[0]):
        anchor_val = perm_anchor_match["rel_err_mean"].values[0]
        m_ref = np.array(budgets, dtype=float)
        ref_line = anchor_val * np.sqrt(min_b / m_ref)
        ax1.plot(m_ref, ref_line, "k--", alpha=0.7, label=r"$O(m^{-1/2})$ reference")
        
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10)
    ax1.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Relative Error $\Vert\hat{\phi} - \phi\Vert_2 / \Vert\phi\Vert_2$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Convergence vs Budget (Log-Log)", fontsize=12, fontweight="bold")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    
    # (b) Top-3 agreement vs m
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
    ax2.set_title("(b) Top-3 Ranking Fidelity", fontsize=12, fontweight="bold")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure1_main_benchmark.png", dpi=300)
    plt.savefig("results/figures/figure1_main_benchmark.pdf")
    plt.close()
    print("Saved Figure 1 to results/figures/figure1_main_benchmark.{png,pdf}")
    
    elapsed_total = time.time() - t0_phase
    print(f"Phase 1 complete in {elapsed_total:.2f} s")
    
    return df_raw, df_agg, {
        "tabulated_cache": tabulated_cache,
        "derived": derived,
        "runtime": elapsed_total
    }
