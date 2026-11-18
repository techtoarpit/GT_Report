"""Phase 2: The Cost Wall - Exponential scaling of exact Shapley computation."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley


def format_seconds(s: float) -> str:
    """Format seconds into human-readable duration."""
    if s < 60:
        return f"{s:.2f} s"
    elif s < 3600:
        return f"{s / 60:.1f} min"
    elif s < 86400:
        return f"{s / 3600:.1f} hours"
    elif s < 31536000:
        return f"{s / 86400:.1f} days"
    elif s < 31536000 * 1000:
        return f"{s / 31536000:.1e} years"
    else:
        return f"{s / 31536000:.1e} years"


def run_phase2(
    d_list: list[int] = [6, 8, 10, 12, 14, 16],
    quick: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Execute Phase 2:
    - Measure real time to tabulate 2^d coalitions and compute exact Shapley for d in d_list
    - Fit exponential trend: log10(seconds) ~ a * d + b
    - Extrapolate for d in {20, 30, 50}
    - Save table to results/tables/phase2_cost_wall.csv
    - Save plot to results/figures/figure2_cost_wall.{png, pdf}
    """
    t0_phase = time.time()
    if quick:
        d_list = [6, 8, 10, 12]
        
    print(f"\n--- Running Phase 2: The Cost Wall (d={d_list}) ---")
    
    rows = []
    for d in d_list:
        print(f"Evaluating exact cost for d={d} (2^{d} = {1 << d} coalitions)...")
        setup = ExperimentSetup(d=d, n_explicands=1)
        x0 = setup.explicands[0]
        
        # Measure tabulation time
        v_table, tab_time, fwd_passes = tabulate_value_function(
            setup.model, x0, setup.background, chunk_size=1024
        )
        
        # Measure exact Shapley time
        _, _, exact_time = compute_exact_shapley(v_table, d)
        
        total_sec = tab_time + exact_time
        print(f"  d={d}: tab={tab_time:.3f}s, exact={exact_time*1000:.2f}ms, total={total_sec:.3f}s")
        
        rows.append({
            "d": d,
            "coalitions": 1 << d,
            "forward_passes": (1 << d) * len(setup.background),
            "seconds": total_sec,
            "human_time": format_seconds(total_sec),
            "type": "measured"
        })
        
    df_measured = pd.DataFrame(rows)
    
    # Fit log10(seconds) vs d
    d_vals = df_measured["d"].values
    log_sec = np.log10(df_measured["seconds"].values)
    slope, intercept = np.polyfit(d_vals, log_sec, 1)
    
    # Extrapolations for d in {20, 30, 50}
    extrap_d = [20, 30, 50]
    extrap_rows = []
    for d in extrap_d:
        sec_extrap = float(10 ** (slope * d + intercept))
        extrap_rows.append({
            "d": d,
            "coalitions": 1 << d,
            "forward_passes": (1 << d) * 100,
            "seconds": sec_extrap,
            "human_time": format_seconds(sec_extrap),
            "type": "extrapolated"
        })
        print(f"  Extrapolated d={d}: {sec_extrap:.2e} s ({format_seconds(sec_extrap)})")
        
    df_all = pd.concat([df_measured, pd.DataFrame(extrap_rows)], ignore_index=True)
    df_all.to_csv("results/tables/phase2_cost_wall.csv", index=False)
    print("Saved Phase 2 table to results/tables/phase2_cost_wall.csv")
    
    # --- Figure 2: The Cost Wall Plot ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    # Plot measured points
    ax.scatter(df_measured["d"], df_measured["seconds"], color="#1f77b4", s=70, zorder=4, label="Measured Real Time")
    ax.plot(df_measured["d"], df_measured["seconds"], color="#1f77b4", linewidth=2.0, zorder=3)
    
    # Plot extrapolation curve
    d_fit_grid = np.linspace(6, 50, 100)
    sec_fit_grid = 10 ** (slope * d_fit_grid + intercept)
    ax.plot(d_fit_grid, sec_fit_grid, "r--", linewidth=1.8, alpha=0.8, label=f"Fitted Trend: $10^{{{slope:.3f}d {intercept:+.2f}}}$ s")
    
    # Plot extrapolated points
    df_extrap = df_all[df_all["type"] == "extrapolated"]
    ax.scatter(df_extrap["d"], df_extrap["seconds"], color="#d62728", marker="x", s=90, linewidth=2, zorder=4, label="Extrapolated (d=20, 30, 50)")
    
    for _, r in df_extrap.iterrows():
        ax.annotate(
            f"d={int(r['d'])}\n{r['human_time']}",
            xy=(r["d"], r["seconds"]),
            xytext=(r["d"] - 3.5, r["seconds"] * 4.0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
            fontsize=8.5,
            fontweight="bold",
            color="#900C3F",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", edgecolor="#d62728", alpha=0.9)
        )
        
    ax.set_yscale("log")
    ax.set_xlabel("Number of Features $d$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time per Explicand (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 2: The Cost Wall - Exponential Scaling of Exact Shapley", fontsize=12, fontweight="bold")
    ax.set_xticks([6, 8, 10, 12, 14, 16, 20, 30, 40, 50])
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure2_cost_wall.png", dpi=300)
    plt.savefig("results/figures/figure2_cost_wall.pdf")
    plt.close()
    print("Saved Figure 2 to results/figures/figure2_cost_wall.{png,pdf}")
    
    elapsed_total = time.time() - t0_phase
    print(f"Phase 2 complete in {elapsed_total:.2f} s")
    
    return df_all, {
        "slope": float(slope),
        "intercept": float(intercept),
        "runtime": elapsed_total
    }
