# Changelog

All notable changes, bug discoveries, and fixes during the experimental study are logged here.

## Full Experimental Benchmark Execution - 2026-10-14
- Successfully executed the complete benchmark protocol (`python run_all.py --full`) in 237.47 s (3.96 min), well under the 20-minute threshold.
- System environment logged in `results/RUN_INFO.json`: AMD64, 16 logical cores, Windows 11, Python 3.12.7, NumPy 1.26.4, SciPy 1.16.2, Pandas 2.3.3, Scikit-Learn 1.7.2, Matplotlib 3.9.2.
- Memory safety: Chunks of 1024 coalitions maintained heap consumption below 300 MB throughout the entire run, easily staying under the 2 GB RAM budget.
- All 10 sanity checks passed cleanly:
  - Exact Shapley efficiency gap: $< 1.8 \times 10^{-15}$
  - Toy additive game recovery: all 5 estimators matched within $< 10^{-15}$
  - 3-player non-additive game: matched hand-calculated values $[7.0, 1.0, 4.0]$ exactly.
- All 6 phases executed and deliverables produced:
  - Phase 1: 10 explicands, 20 seeds, 6 budgets, 5 estimators (6,000 runs) -> Table II, Top-3 agreement, Figure 1 (PNG & PDF), derived metrics.
  - Phase 2: The Cost Wall -> Profiling $d \in \{6, 8, 10, 12, 14, 16\}$, exponential fit, extrapolations for $d \in \{20, 30, 50\}$, Figure 2 (PNG & PDF).
  - Phase 3: Scale-up to $d=16$ (65,536 coalitions) -> 5 explicands, 10 seeds, 7 budgets, Figure 3 (PNG & PDF). Confirmed Paired KernelSHAP ranking persistence in tiny budget fraction regime.
  - Phase 4: Multilinear grid ablation ($K \in \{4, 8, 16, 32\}$) -> Table and Figure 4 (PNG & PDF). Confirmed quadrature efficiency gap persists even at $K=32$.
  - Phase 5: Cheap two-seed convergence diagnostic -> Table and Figure 5 (PNG & PDF). Confirmed Spearman correlation $\ge 0.995$ and top-3 overlap $= 1.0$ effectively diagnose true error $< 1.5\%$.
  - Phase 6: Systematic audit against draft report recorded in `results/tables/claims_check.csv`.

## Initial Setup & Bug Fixes - 2026-10-14
- Fixed array bounds indexing in exact Shapley weight precomputation: subset size array includes $0$ to $d$; ensured indexing for subsets of $N \setminus \{i\}$ bounds properly without index errors.
- Handled loky physical core detection warning on Windows environment by setting `LOKY_MAX_CPU_COUNT` and filtering extraneous warnings.
- Suppressed standard output buffering during long-running background tasks.
