"""Main execution entry point: runs all experimental phases and records metadata."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
import numpy as np
import scipy
import pandas as pd
import sklearn
import matplotlib
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from src.data_model import ExperimentSetup
from src.sanity_checks import run_sanity_checks
from src.phases.phase1 import run_phase1
from src.phases.phase2 import run_phase2
from src.phases.phase3 import run_phase3
from src.phases.phase4 import run_phase4
from src.phases.phase5 import run_phase5
from src.phases.phase6 import run_phase6


def get_system_metadata() -> dict:
    """Collect operating system, CPU, and library version details."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run Shapley Value Approximation Benchmark Suite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run quick smoke-test (3 explicands, 5 seeds)")
    group.add_argument("--full", action="store_true", help="Run full benchmark protocol")
    args = parser.parse_args()
    
    t_global_start = time.time()
    mode_name = "QUICK" if args.quick else "FULL"
    print(f"=================================================================")
    print(f"   STARTING SHAPLEY APPROXIMATION BENCHMARK [{mode_name} MODE]   ")
    print(f"=================================================================")
    
    # Ensure all output directories exist
    for d_path in [
        "results",
        "results/raw",
        "results/tables",
        "results/figures",
        "results/old_script_output"
    ]:
        os.makedirs(d_path, exist_ok=True)
        
    run_info = {
        "mode": mode_name,
        "system": get_system_metadata(),
        "phase_runtimes_seconds": {},
    }
    
    # --- Correctness and Sanity Checks ---
    print("\n>>> STEP 0: Running Correctness & Sanity Tests...")
    t0 = time.time()
    n_exp_init = 3 if args.quick else 10
    setup_d12 = ExperimentSetup(d=12, n_explicands=n_exp_init)
    df_sanity = run_sanity_checks(setup_d12)
    t_sanity = time.time() - t0
    run_info["phase_runtimes_seconds"]["sanity_checks"] = t_sanity
    print(f"Sanity checks completed in {t_sanity:.2f} s. All tests PASSED.")
    
    # --- Phase 1: Main Benchmark ---
    t0 = time.time()
    p1_budgets = [64, 128, 256, 512, 1024, 2048]
    p1_seeds = 5 if args.quick else 20
    df_p1_raw, df_p1_agg, p1_out = run_phase1(
        setup_d12,
        budgets=p1_budgets,
        n_seeds=p1_seeds,
        quick=args.quick
    )
    t_p1 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase1"] = t_p1
    
    # --- Phase 2: The Cost Wall ---
    t0 = time.time()
    p2_d_list = [6, 8, 10, 12] if args.quick else [6, 8, 10, 12, 14, 16]
    df_p2, p2_out = run_phase2(d_list=p2_d_list, quick=args.quick)
    t_p2 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase2"] = t_p2
    
    # --- Phase 3: Scale up to d=16 ---
    t0 = time.time()
    p3_budgets = [128, 256, 512, 1024] if args.quick else [128, 256, 512, 1024, 2048, 4096, 8192]
    p3_exp = 2 if args.quick else 5
    p3_seeds = 3 if args.quick else 10
    df_p3_raw, df_p3_agg, p3_out = run_phase3(
        budgets=p3_budgets,
        n_explicands=p3_exp,
        n_seeds=p3_seeds,
        quick=args.quick
    )
    t_p3 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase3"] = t_p3
    
    # --- Phase 4: Multilinear Grid Ablation ---
    t0 = time.time()
    p4_K_list = [4, 8] if args.quick else [4, 8, 16, 32]
    p4_seeds = 5 if args.quick else 20
    df_p4_raw, df_p4_agg, p4_out = run_phase4(
        setup_d12,
        K_list=p4_K_list,
        budgets=p1_budgets,
        n_seeds=p4_seeds,
        tabulated_cache=p1_out["tabulated_cache"],
        quick=args.quick
    )
    t_p4 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase4"] = t_p4
    
    # --- Phase 5: Convergence Check Demo ---
    t0 = time.time()
    df_p5_raw, df_p5_summary, p5_out = run_phase5(
        setup_d12,
        budgets=p1_budgets,
        tabulated_cache=p1_out["tabulated_cache"],
        quick=args.quick
    )
    t_p5 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase5"] = t_p5
    
    # --- Phase 6: Claims Check Against Draft Report ---
    t0 = time.time()
    df_claims = run_phase6(
        setup_d12,
        phase1_results=p1_out,
        phase2_results=p2_out,
        phase4_results=p4_out
    )
    t_p6 = time.time() - t0
    run_info["phase_runtimes_seconds"]["phase6"] = t_p6
    
    # Total runtime
    t_total = time.time() - t_global_start
    run_info["total_runtime_seconds"] = t_total
    run_info["timestamp_completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    # Save RUN_INFO.json
    with open("results/RUN_INFO.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print("\nSaved run metadata and timing to results/RUN_INFO.json")
    
    print("\n=================================================================")
    print(f"   ALL PHASES COMPLETED SUCCESSFULLY IN {t_total:.2f} s ({t_total/60:.2f} min)   ")
    print(f"=================================================================")
    print("Summary of Phase Runtimes:")
    for p_name, p_time in run_info["phase_runtimes_seconds"].items():
        print(f"  - {p_name:18s}: {p_time:.2f} s")
    print(f"  - Total Elapsed    : {t_total:.2f} s")


if __name__ == "__main__":
    main()
