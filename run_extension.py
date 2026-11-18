"""Runs the extension phases (7-9) on top of the finished Phase 1-6 results.

Phases 1-6 are not recomputed; their raw CSVs in results/raw are reused.
    python run_extension.py --quick
    python run_extension.py --full
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import numpy as np
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function
from src.exact import compute_exact_shapley
from src.phases.phase7 import run_phase7
from src.phases.phase8 import run_phase8
from src.phases.phase9 import run_phase9
from src.phases.phase10 import run_phase10

REF_ACCURACY = 0.93567      # recorded in results/tables/sanity_checks.csv
REF_V_N_EXPLICAND0 = -2.025251


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--phases", default="7,8,9,10",
                    help="comma-separated subset of extension phases to run")
    args = ap.parse_args()
    wanted = {int(p) for p in args.phases.split(",")}

    t_start = time.time()
    for p in ("results/raw", "results/tables", "results/figures"):
        os.makedirs(p, exist_ok=True)

    setup = ExperimentSetup(d=12, n_explicands=10)

    # The new phases compare against the stored Phase 1 raw runs, so make sure this
    # environment rebuilds the same model and explicands as the original run.
    cache = []
    for i in range(setup.n_explicands):
        v, _, _ = tabulate_value_function(setup.model, setup.explicands[i], setup.background)
        phi, _, _ = compute_exact_shapley(v, setup.d)
        cache.append((v, phi, v[-1] - v[0]))
    ok_acc = abs(setup.test_accuracy - REF_ACCURACY) < 1e-4
    ok_v = abs(cache[0][0][-1] - REF_V_N_EXPLICAND0) < 1e-5
    print(f"Reproducibility check: accuracy={setup.test_accuracy:.5f} (ref {REF_ACCURACY}), "
          f"v(N)[0]={cache[0][0][-1]:.6f} (ref {REF_V_N_EXPLICAND0})")
    if not (ok_acc and ok_v):
        raise SystemExit("Model/explicands differ from the original run; refusing to mix results.")

    info = {"mode": "QUICK" if args.quick else "FULL", "runtimes_seconds": {}}

    if 7 in wanted:
        t0 = time.time()
        run_phase7(setup, cache)
        info["runtimes_seconds"]["phase7"] = time.time() - t0

    if 8 in wanted:
        t0 = time.time()
        if args.quick:
            run_phase8(setup, cache, bg_sizes=(10, 100), bg_draws=2, n_explicands=2)
        else:
            run_phase8(setup, cache)
        info["runtimes_seconds"]["phase8"] = time.time() - t0

    if 9 in wanted:
        t0 = time.time()
        if args.quick:
            run_phase9(setup, cache, budgets=(128, 512), replicates=5)
        else:
            run_phase9(setup, cache)
        info["runtimes_seconds"]["phase9"] = time.time() - t0

    if 10 in wanted:
        t0 = time.time()
        if args.quick:
            run_phase10(setup, cache, budgets=(512, 1024), K_list=(4, 32), replicates=10)
        else:
            run_phase10(setup, cache)
        info["runtimes_seconds"]["phase10"] = time.time() - t0

    info["total_runtime_seconds"] = time.time() - t_start
    info["timestamp_completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    import sys, sklearn, scipy, pandas, matplotlib
    info["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "scipy": scipy.__version__, "pandas": pandas.__version__,
                           "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    tag = "" if args.phases == "7,8,9,10" else "_phases" + args.phases.replace(",", "-")
    out = f"results/RUN_INFO_extension{tag}.json" if args.full else f"results/RUN_INFO_extension{tag}_quick.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nExtension phases finished in {info['total_runtime_seconds']:.1f} s -> {out}")


if __name__ == "__main__":
    main()
