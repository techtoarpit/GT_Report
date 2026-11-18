"""Phase 8: sensitivity of the exact Shapley values to the value function.

The draft argues that changing how a missing feature is replaced moves the
target by more than the estimators' approximation error. Here the exact
values are recomputed under different background sets (size and draw) and the
shift is compared with the error of the sampling estimators from Phase 1.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley
from src.metrics import compute_relative_error, compute_top3_agreement


def run_phase8(
    setup: ExperimentSetup,
    tabulated_cache: list,
    bg_sizes: list[int] = (10, 25, 50, 100, 200),
    bg_draws: int = 5,
    n_explicands: int | None = None,
) -> dict:
    t0 = time.time()
    d = setup.d
    n_exp = n_explicands or setup.n_explicands
    print(f"\n--- Running Phase 8 (value-function sensitivity, {n_exp} explicands, "
          f"B in {list(bg_sizes)}, {bg_draws} draws each) ---")

    n_train = len(setup.X_train)
    rows = []

    # Mean-baseline game: a single reference point (training mean)
    mean_bg = setup.X_train.mean(axis=0, keepdims=True)

    for i in range(n_exp):
        x = setup.explicands[i]
        phi_ref = tabulated_cache[i][1]

        v, _, _ = tabulate_value_function(setup.model, x, mean_bg)
        phi_alt, _, _ = compute_exact_shapley(v, d)
        rows.append({"explicand": i, "setting": "mean baseline", "B": 1, "draw": 0,
                     "shift": compute_relative_error(phi_alt, phi_ref),
                     "top3": compute_top3_agreement(phi_alt, phi_ref)})

        for B in bg_sizes:
            for draw in range(1, bg_draws + 1):
                if B == 100 and draw == 0:
                    continue
                rng = np.random.default_rng(10_000 * B + draw)
                idx = rng.choice(n_train, size=B, replace=False)
                v, _, _ = tabulate_value_function(setup.model, x, setup.X_train[idx])
                phi_alt, _, _ = compute_exact_shapley(v, d)
                rows.append({"explicand": i, "setting": f"B={B}", "B": B, "draw": draw,
                             "shift": compute_relative_error(phi_alt, phi_ref),
                             "top3": compute_top3_agreement(phi_alt, phi_ref)})
        print(f"  explicand {i + 1}/{n_exp} done ({time.time() - t0:.0f} s elapsed)")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv("results/raw/phase8_raw.csv", index=False)

    # Average draws within explicand, then mean and standard error across explicands
    per_exp = df_raw.groupby(["setting", "B", "explicand"])[["shift", "top3"]].mean().reset_index()
    agg = per_exp.groupby(["setting", "B"]).agg(
        shift_mean=("shift", "mean"),
        shift_se=("shift", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        top3_mean=("top3", "mean"),
        top3_se=("top3", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reset_index().sort_values("B")
    agg.to_csv("results/tables/phase8_value_function_sensitivity.csv", index=False)

    # Estimator error from the main benchmark, for the comparison lines
    raw1 = pd.read_csv("results/raw/phase1_raw.csv")
    ref_lines = {}
    for est in ("Permutation", "Paired KernelSHAP"):
        for m in (512, 2048):
            sub = raw1[(raw1.estimator == est) & (raw1.budget == m) & (raw1.explicand < n_exp)]
            ref_lines[(est, m)] = sub.groupby("explicand").rel_err.mean().mean()
    pd.DataFrame(
        [{"estimator": k[0], "budget": k[1], "mean_rel_err": v} for k, v in ref_lines.items()]
    ).to_csv("results/tables/phase8_estimator_reference.csv", index=False)

    # Figure 7
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    bgs = agg[agg.setting != "mean baseline"]
    ax1.errorbar(bgs.B, bgs.shift_mean, yerr=bgs.shift_se, marker="o", color="#1f77b4",
                 capsize=3, linewidth=1.8, label="Exact values under a resampled background")
    mb = agg[agg.setting == "mean baseline"].iloc[0]
    ax1.axhline(mb.shift_mean, color="#9467bd", linestyle="-.", linewidth=1.5,
                label="Exact values, single mean baseline")
    styles = {("Paired KernelSHAP", 512): ("#d62728", "--"),
              ("Paired KernelSHAP", 2048): ("#d62728", ":"),
              ("Permutation", 512): ("#2ca02c", "--"),
              ("Permutation", 2048): ("#2ca02c", ":")}
    for (est, m), (c, ls) in styles.items():
        ax1.axhline(ref_lines[(est, m)], color=c, linestyle=ls, linewidth=1.4,
                    label=f"{est} estimator error, m={m}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Background sample size $B$", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Shift of exact $\phi$ relative to the $B{=}100$ reference", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Target movement vs approximation error", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, frameon=True)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    ax2.errorbar(bgs.B, bgs.top3_mean, yerr=bgs.top3_se, marker="s", color="#ff7f0e",
                 capsize=3, linewidth=1.8, label="Resampled background")
    ax2.axhline(mb.top3_mean, color="#9467bd", linestyle="-.", linewidth=1.5, label="Mean baseline")
    ax2.set_xscale("log")
    ax2.set_ylim(0.5, 1.03)
    ax2.set_xlabel("Background sample size $B$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Top-3 overlap with the reference", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Ranking sensitivity to the value function", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, frameon=True)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/figure7_value_function_sensitivity.png", dpi=300)
    plt.savefig("results/figures/figure7_value_function_sensitivity.pdf")
    plt.close()

    print(agg.to_string(index=False))
    print(f"Phase 8 complete in {time.time() - t0:.2f} s")
    return {"agg": agg, "ref_lines": ref_lines}
