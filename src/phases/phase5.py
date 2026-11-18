"""Phase 5: Convergence-check diagnostic demo."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley
from src.estimators import paired_kernel_shap
from src.metrics import compute_relative_error, compute_top3_agreement


def run_phase5(
    setup: ExperimentSetup,
    budgets: list[int] = [64, 128, 256, 512, 1024, 2048],
    tabulated_cache: list | None = None,
    quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute Phase 5:
    - Runs two independent seeds of Paired KernelSHAP on the same explicand
    - Measures inter-run agreement: Top-3 feature overlap and Spearman rank correlation of |phi|
    - Compares against true relative error vs exact Shapley
    - Saves results/raw/phase5_raw.csv, results/tables/phase5_convergence_check.csv,
      and results/figures/figure5_convergence_check.{png,pdf}
    """
    t0_phase = time.time()
    d = setup.d
    n_explicands = setup.n_explicands
    
    print(f"\n--- Running Phase 5: Convergence Check Diagnostic (budgets={budgets}) ---")
    
    if tabulated_cache is None:
        tabulated_cache = []
        for i in range(n_explicands):
            x = setup.explicands[i]
            v_table, _, _ = tabulate_value_function(setup.model, x, setup.background)
            phi_exact, _, _ = compute_exact_shapley(v_table, d)
            tabulated_cache.append((v_table, phi_exact, v_table[-1] - v_table[0]))
            
    records = []
    
    for i in range(n_explicands):
        v_table, phi_exact, _ = tabulated_cache[i]
        
        for m in budgets:
            seed_a = 50000 + 2 * i
            seed_b = 50000 + 2 * i + 1
            
            phi_a = paired_kernel_shap(v_table, d, m, seed=seed_a)
            phi_b = paired_kernel_shap(v_table, d, m, seed=seed_b)
            
            # 1. Top-3 agreement between the two runs
            top3_a = set(np.argsort(np.abs(phi_a))[-3:])
            top3_b = set(np.argsort(np.abs(phi_b))[-3:])
            two_run_top3_overlap = len(top3_a.intersection(top3_b)) / 3.0
            
            # 2. Spearman rank correlation of |phi|
            spearman_res = stats.spearmanr(np.abs(phi_a), np.abs(phi_b))
            spearman_corr = float(spearman_res.statistic if hasattr(spearman_res, 'statistic') else spearman_res[0])
            
            # 3. True relative error vs exact
            err_a = compute_relative_error(phi_a, phi_exact)
            err_b = compute_relative_error(phi_b, phi_exact)
            avg_true_rel_err = (err_a + err_b) / 2.0
            
            # 4. True top-3 agreement vs exact
            top3_acc_a = compute_top3_agreement(phi_a, phi_exact)
            top3_acc_b = compute_top3_agreement(phi_b, phi_exact)
            avg_true_top3 = (top3_acc_a + top3_acc_b) / 2.0
            
            records.append({
                "explicand": i,
                "budget": m,
                "two_run_top3_overlap": two_run_top3_overlap,
                "two_run_spearman": spearman_corr,
                "true_rel_err": avg_true_rel_err,
                "true_top3_accuracy": avg_true_top3
            })
            
    df_raw = pd.DataFrame(records)
    df_raw.to_csv("results/raw/phase5_raw.csv", index=False)
    
    # Aggregate across explicands
    df_summary = df_raw.groupby("budget").agg(
        two_run_top3_mean=("two_run_top3_overlap", "mean"),
        two_run_top3_se=("two_run_top3_overlap", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        spearman_mean=("two_run_spearman", "mean"),
        spearman_se=("two_run_spearman", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        true_rel_err_mean=("true_rel_err", "mean"),
        true_rel_err_se=("true_rel_err", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        true_top3_mean=("true_top3_accuracy", "mean"),
        true_top3_se=("true_top3_accuracy", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
    ).reset_index()
    
    df_summary.to_csv("results/tables/phase5_convergence_check.csv", index=False)
    print("Saved Phase 5 table to results/tables/phase5_convergence_check.csv")
    
    # --- Figure 5: Convergence Check Demo ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # (a) Agreement between two runs vs budget
    ax1.errorbar(
        df_summary["budget"],
        df_summary["spearman_mean"],
        yerr=df_summary["spearman_se"],
        marker="o",
        color="#2ca02c",
        linewidth=2,
        capsize=3,
        label="2-Seed Spearman Rank Corr."
    )
    ax1.errorbar(
        df_summary["budget"],
        df_summary["two_run_top3_mean"],
        yerr=df_summary["two_run_top3_se"],
        marker="s",
        color="#1f77b4",
        linewidth=2,
        capsize=3,
        label="2-Seed Top-3 Overlap"
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Diagnostic Metric Between Two Runs", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Cheap Two-Seed Diagnostic Agreement", fontsize=12, fontweight="bold")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(frameon=True, fontsize=9.5)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    
    # (b) Diagnostic vs True Relative Error
    ax2.plot(
        df_summary["budget"],
        df_summary["true_rel_err_mean"],
        marker="D",
        color="#d62728",
        linewidth=2,
        label="True Relative Error vs Exact"
    )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=10)
    ax2.set_xlabel("Evaluation Budget $m$ (Coalitions)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Relative Error (Log Scale)", fontsize=11, fontweight="bold", color="#d62728")
    ax2.set_title("(b) True Convergence vs Exact Values", fontsize=12, fontweight="bold")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure5_convergence_check.png", dpi=300)
    plt.savefig("results/figures/figure5_convergence_check.pdf")
    plt.close()
    print("Saved Figure 5 to results/figures/figure5_convergence_check.{png,pdf}")
    
    elapsed_total = time.time() - t0_phase
    print(f"Phase 5 complete in {elapsed_total:.2f} s")
    
    return df_raw, df_summary, {
        "runtime": elapsed_total
    }
