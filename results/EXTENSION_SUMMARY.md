# Extension Study (Phases 7-10)

Run on 2026-11-16 with `python run_extension.py --full` (phases 7-9: 315 s) and `--phases 10` (422 s), Python 3.12.7 / NumPy 1.26.4 / scikit-learn 1.7.2. The script verified before running that the rebuilt model reproduces the stored mid-sem run (test accuracy 0.93567, v(N) of explicand 0 = -2.025251). All numbers are mean +/- standard error over the 10 explicands unless stated.

## Why these phases

The draft report makes statements that the six original phases never tested. Phases 7-10 test them.

| Draft statement | Phase | Verdict |
| :--- | :---: | :--- |
| "The features are moderately correlated but not pathologically so." | 7 | **Not supported.** Mean abs. pairwise correlation 0.78, max 0.998, 74% of pairs above 0.7, 27% above 0.9, median VIF 135, max VIF 1846, condition number 3.5e4. |
| "A different removal strategy moves the target by more than the approximation error." | 8 | **Supported.** Redrawing the 100-row background shifts exact values by 0.13-0.18 (relative L2); a single mean baseline shifts them by 0.78 and keeps only 70% of the top-3. Paired KernelSHAP error at m=2048 is 0.0076, Permutation 0.0204. |
| "Rankings stabilise before values; a plot can look converged while numbers are 5% off." | 9 | **Supported for unpaired estimators, not for paired.** Given an exact top-3, P(rel. error > 5%) is 0.60 (Permutation) / 0.61 (KernelSHAP) at m=256 and 0.22 / 0.20 at m=512. Antithetic: 0 from m=512. Paired KernelSHAP: 0 from m=256 (0.45 at m=64). |
| "Replicate-based uncertainty estimates are cheap and rarely reported." | 9 | **Works, at 10x the budget.** 95% t-intervals from 10 runs cover the exact value 0.92-0.98 of the time (Permutation 0.940, Antithetic 0.958, Paired KS 0.931 averaged over budgets). Paired KS sits consistently a little below 0.95. |
| "The multilinear estimator's efficiency gap is quadrature bias that does not shrink." | 10 | **Partly.** True for K=4: gap of the averaged estimate is flat at 0.075-0.078 for m = 512..8192, and bias (0.0089) overtakes noise at m of about 4096. **Not true for K=8 and K=32:** bias is 0.0017 / 0.0008 at m=8192, the single-run gap falls with slope -0.50 / -0.52 (Monte Carlo rate), i.e. it is mostly noise. |

## Phase 10 detail (100 replicates x 10 explicands, d=12)

| Estimator | m | single-run rel. err | bias | noise | eff. gap (single run) | eff. gap (mean of 100 runs) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multilinear K=4 | 512 | 0.0270 | 0.0092 | 0.0265 | 0.183 | 0.075 |
| Multilinear K=4 | 8192 | 0.0108 | 0.0089 | 0.0062 | 0.084 | 0.078 |
| Multilinear K=8 | 512 | 0.0303 | 0.0036 | 0.0314 | 0.206 | 0.023 |
| Multilinear K=8 | 8192 | 0.0075 | 0.0017 | 0.0076 | 0.051 | 0.011 |
| Multilinear K=32 | 512 | 0.0340 | 0.0033 | 0.0353 | 0.227 | 0.017 |
| Multilinear K=32 | 8192 | 0.0079 | 0.0008 | 0.0082 | 0.052 | 0.006 |
| Paired KernelSHAP | 512 | 0.0153 | 0.0015 | 0.0156 | 0 | 0 |
| Paired KernelSHAP | 8192 | 0.0038 | 0.0004 | 0.0039 | 0 | 0 |

The resolvable bias floor with 100 replicates is noise/sqrt(100), about 0.0008 at m=8192 (column `bias_noise_floor` in the CSV), so the K=32 bias is at the detection limit while the K=4 bias is clearly real.

## Corrections made to earlier files

1. `RESULTS_SUMMARY.md`, headline table, Top-3 rows: the Permutation / Antithetic / KernelSHAP / Multilinear entries were mis-transcribed and did not match `top3_agreement.csv` (which matches the raw runs). Corrected values at m=512: 95.0 / 97.2 / 95.5 / 98.0 / 95.5 % and at m=2048: 97.7 / 99.7 / 98.8 / 100.0 / 98.0 % (Perm / Anti / KS / Paired KS / Multilinear). The Paired KernelSHAP m=512 value (98.0%) was already right.
2. `RESULTS_SUMMARY.md` Finding 3: Paired KernelSHAP at m=1024 is 99.5% (was 99.3%).
3. `MID_SEM_TALKING_POINTS.md` point 4: Paired KernelSHAP reaches 100% (was 99.7%) at m=2048.
4. `RESULTS_SUMMARY.md` Findings 6 and 9: correction notes added (see Phase 10 above).

## Other things worth knowing

- Phase 8's shift does not decay with B beyond B=25. This is expected, not a finding about convergence: the reference is itself one 100-row draw with its own sampling noise, so the plateau of 0.13-0.18 reflects the noise of two background draws. It supports only the comparison of scales stated above.
- Phase 9's confidence-interval study spends 10x the budget per explanation; no single-run interval was tried.
- In Figure 9a the Paired KernelSHAP single-run line is exactly zero and therefore not drawn on the log axis (it still appears in the legend).
- The Phase 5 diagnostic uses one seed pair per explicand (n=10 pairs), so it is a plausibility check, not a calibrated stopping rule.
