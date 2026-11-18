"""Phase 7: feature-dependence audit.

The draft report describes the twelve selected features as "moderately
correlated but not pathologically so". This phase measures that directly and
also records which features carry the exact Shapley mass.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup


def run_phase7(setup: ExperimentSetup, tabulated_cache: list) -> dict:
    t0 = time.time()
    d = setup.d
    print(f"\n--- Running Phase 7 (feature dependence audit, d={d}) ---")

    names = [str(n) for n in setup.feature_names]
    corr = np.corrcoef(setup.X_train.T)
    iu = np.triu_indices(d, 1)
    abs_r = np.abs(corr[iu])

    # Variance inflation factors from the inverse correlation matrix
    vif = np.diag(np.linalg.inv(corr))
    cond = float(np.linalg.cond(corr))

    summary = pd.DataFrame([
        {"metric": "number of feature pairs", "value": len(abs_r)},
        {"metric": "mean |r|", "value": abs_r.mean()},
        {"metric": "median |r|", "value": float(np.median(abs_r))},
        {"metric": "max |r|", "value": abs_r.max()},
        {"metric": "fraction of pairs with |r| > 0.7", "value": float((abs_r > 0.7).mean())},
        {"metric": "fraction of pairs with |r| > 0.9", "value": float((abs_r > 0.9).mean())},
        {"metric": "max VIF", "value": float(vif.max())},
        {"metric": "median VIF", "value": float(np.median(vif))},
        {"metric": "condition number of correlation matrix", "value": cond},
    ])
    summary.to_csv("results/tables/phase7_dependence_summary.csv", index=False)
    pd.DataFrame(corr, index=names, columns=names).to_csv(
        "results/tables/phase7_correlation_matrix.csv")

    # Mean |phi| per feature across explicands (exact values)
    phis = np.array([c[1] for c in tabulated_cache])
    mean_abs_phi = np.abs(phis).mean(axis=0)
    share = mean_abs_phi / mean_abs_phi.sum()
    df_feat = pd.DataFrame({
        "feature": names,
        "vif": vif,
        "mean_abs_exact_shapley": mean_abs_phi,
        "share_of_total_abs": share,
    })
    df_feat.to_csv("results/tables/phase7_feature_table.csv", index=False)

    # Figure 6
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=300,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax1.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xticks(range(d))
    ax1.set_yticks(range(d))
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.grid(False)
    for i in range(d):
        for j in range(d):
            ax1.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(corr[i, j]) > 0.75 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03, label="Pearson correlation")
    ax1.set_title("(a) Correlation of the selected features", fontsize=12, fontweight="bold")

    order = np.argsort(mean_abs_phi)
    ax2.barh(np.array(names)[order], mean_abs_phi[order], color="#1f77b4")
    ax2.set_xlabel("Mean |exact Shapley value| (log-odds)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Where the attribution mass sits", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/figure6_feature_dependence.png", dpi=300)
    plt.savefig("results/figures/figure6_feature_dependence.pdf")
    plt.close()

    print(summary.to_string(index=False))
    print(f"Phase 7 complete in {time.time() - t0:.2f} s")
    return {"summary": summary, "features": df_feat}
