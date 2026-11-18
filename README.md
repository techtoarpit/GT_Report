# Efficient Shapley Value Approximation Benchmark Suite

Companion code, results and paper for **"How Cheaply Can Shapley Attributions Be Estimated? Five Model-Agnostic Estimators Measured Against Exact Values"**.

Authors: Ayushman Bhatt and Arpit Mishra, Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India.

The repository contains a from-scratch implementation of five model-agnostic Shapley value estimators, a benchmark that measures them against exact values, three audits of the assumptions behind the benchmark, and the IEEE conference paper that reports the results. No external SHAP library is used.

---

## 1. Summary of Findings

All numbers below are from `results/` and appear in the paper.

| Question | Result |
| :--- | :--- |
| Most accurate estimator | Paired KernelSHAP, at every budget and at both $d=12$ and $d=16$ |
| Accuracy at half the exact cost ($d=12$, $m=2048$ of 4096) | Relative error 0.76%, about 63% lower than permutation sampling |
| Evaluations needed to match permutation sampling at $m=2048$ | Paired KernelSHAP about 321 (6.4x fewer); antithetic permutation 643 (3.2x); multilinear 1123 (1.8x); KernelSHAP 2023 |
| Convergence rate | All estimators follow roughly $m^{-1/2}$; pairing changes the constant, not the rate |
| Cost of exact computation | Time grows by about 1.95x per added feature; 1.0 s at $d=12$, 17.8 s at $d=16$ |
| Multilinear estimator, $K=4$ | Bias floor of about 0.009 (relative $\ell_2$) that more samples do not remove |
| Multilinear estimator, $K=8,32$ | Small bias; the efficiency gap in a single run is mostly sampling noise |
| Feature dependence | Mean absolute correlation 0.78, median VIF 135 |
| Background sensitivity | Redrawing the background moves the exact values by 0.13-0.18, about 20x the error of Paired KernelSHAP at $m=2048$ |
| Interval coverage | 95% $t$-intervals from 10 runs cover the exact values at 0.92-0.98 |
| Ranking versus values | For unpaired estimators, runs with the correct top three can still have relative error above 5% (0.60 at $m=256$ for permutation sampling) |

---

## 2. System Architecture

The study is a four-stage pipeline. The same diagram appears as Fig. 1 of the paper.

```
 1. Data and model          2. Game and ground truth      3. Estimators (budget m)     4. Evaluation
 ----------------------     --------------------------    ------------------------     ---------------------------
 WDBC data (569 x 30)       Background B (100 rows)       Permutation sampling         Metrics: relative error,
        |                   10 explained instances        Antithetic permutation       top-3 agreement,
 Stratified 70/30 split            |                      KernelSHAP                   efficiency gap
        |                   Interventional value v(S)     Paired KernelSHAP                   |
 Top-d features by                 |                      Multilinear extension        Benchmark d=12, d=16;
 mutual information         Tabulate all 2^d coalitions          ^                     cost of exactness
        |                          |  ------ m lookups ---------+                      Multilinear study
 Gradient boosting model    Exact Shapley values  ------ exact values (reference) ---> Audits
 (log-odds output)
```

1. **Data and model.** The Wisconsin Diagnostic Breast Cancer data are split 70/30 (stratified). The $d$ features with the highest mutual information with the label are selected on the training part only, and a histogram gradient boosting classifier (200 iterations, maximum depth 4) is fitted. Test accuracy is 93.57%. The explained output is the log-odds.
2. **Game and ground truth.** An interventional value function fills missing features from a fixed background of 100 training rows. The value function is tabulated once for all $2^d$ coalitions of each explained instance, and exact Shapley values are computed from that table.
3. **Estimators.** Each estimator receives a budget of $m$ coalition lookups from the same table, so no estimator benefits from implementation details.
4. **Evaluation.** Estimates are compared with the exact values using three metrics. The results feed the benchmark (accuracy, scaling, multilinear study) and the audits (feature dependence, background sensitivity, interval coverage, convergence check).

---

## 3. Estimators

1. **Permutation sampling.** Draws $\lfloor m/(d+1) \rfloor$ random orderings. Each ordering costs $d+1$ lookups and gives one marginal contribution per feature.
2. **Antithetic permutation sampling.** Pairs each ordering with its reverse, giving $\lfloor m/(2(d+1)) \rfloor$ pairs.
3. **KernelSHAP.** Draws $m$ coalitions with size $s \in \{1,\dots,d-1\}$ chosen proportional to $(d-1)/(s(d-s))$ and a uniformly random subset of that size. Solves the constrained weighted least-squares problem, with the efficiency constraint enforced exactly by eliminating one coefficient.
4. **Paired KernelSHAP.** Draws $m/2$ coalitions as above and adds the complement $N \setminus S$ of each, for $m$ lookups in total.
5. **Multilinear extension.** Integrates over the inclusion probability $q$ with the trapezoid rule on $K$ grid points ($K=4$ by default). The endpoints $q=0$ and $q=1$ are deterministic. The estimator returns NaN when the budget cannot give every interior grid point a sample.

---

## 4. Experimental Phases

| Phase | Description | Outputs |
| :--- | :--- | :--- |
| Sanity checks | Efficiency to within $10^{-11}$, exact recovery on an additive toy game for all five estimators, hand-checked three-player game | `results/tables/sanity_checks.csv` |
| 1. Main benchmark | $d=12$, 10 explicands, 20 seeds, budgets 64 to 2048 | Table II, top-3 table, Figure 1 |
| 2. Cost of exactness | Wall-clock time for $d \in \{6,\dots,16\}$, exponential fit, extrapolation to $d=20,30,50$ | Figure 2 |
| 3. Scale to $d=16$ | 65,536 coalitions, 5 explicands, 10 seeds, budgets up to 8192 | Figure 3 |
| 4. Multilinear grid ablation | $K \in \{4,8,16,32\}$ | Figure 4 |
| 5. Convergence check | Agreement of two independent runs compared with the true error | Figure 5 |
| 6. Claims audit | Row-by-row verification of the claims of the draft report | `results/tables/claims_check.csv` |
| 7. Feature dependence | Correlation matrix, VIF, condition number, mean absolute Shapley value per feature | Figure 6 |
| 8. Value-function sensitivity | Exact values recomputed with backgrounds of 10 to 200 rows and with a single mean baseline | Figure 7 |
| 9. Uncertainty | Coverage of 95% $t$-intervals from 10 runs; error of runs that recover the exact top three | Figure 8 |
| 10. Multilinear bias and variance | 100 replicates per explicand and budget for $K \in \{4,8,32\}$ and Paired KernelSHAP | Figure 9 |

---

## 5. Reproduction

### Requirements

- Python 3.10 or newer
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`

```bash
pip install numpy scipy pandas scikit-learn matplotlib jupyter
```

The stored results were produced with Python 3.12.7, NumPy 1.26.4, SciPy 1.16.2 and scikit-learn 1.7.2. A different scikit-learn version can change the trained model slightly.

### Running the experiments

```bash
python run_all.py --quick                    # smoke test, about 60 seconds
python run_all.py --full                     # sanity checks and phases 1-6, about 4 minutes

python run_extension.py --quick              # smoke test of phases 7-10
python run_extension.py --full               # phases 7-10, about 12 minutes
python run_extension.py --full --phases 10   # a subset of the extension phases
```

`run_extension.py` rebuilds the model first and checks that the test accuracy (0.93567) and $v(N)$ of explicand 0 (-2.025251) match the stored run. It stops if they do not. It reuses the Phase 1 raw runs and does not recompute phases 1-6.

### Demonstration notebook

`notebooks/Shapley_Mid_Sem_Demo.ipynb`, built by `make_notebook.py`, loads the stored tables and figures of the sanity checks and phases 1-6 without recomputation.

```bash
jupyter notebook notebooks/Shapley_Mid_Sem_Demo.ipynb
```

### Building the paper

The paper uses the IEEEtran conference class. Either of the following produces `paper/main.pdf` (11 pages):

```bash
cd paper
tectonic main.tex
# or
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The paper can also be built by uploading the `paper/` folder to Overleaf.

---

## 6. Scope

The study uses one dataset (Wisconsin Diagnostic Breast Cancer), one model class (gradient boosting) and one interventional value function. The $d=16$ benchmark uses 5 explicands, and the convergence check uses one seed pair per explicand. The paper states these limits and lists the following as outside its scope: other datasets and model classes, conditional value functions, $d>16$, stratified-sampling and leverage-score estimators, a TreeSHAP cross-check, and unbiased KernelSHAP.

---

## 7. Repository Structure

```
GT_Report/
├── README.md
├── run_all.py                  # Sanity checks and phases 1-6 (--quick, --full)
├── run_extension.py            # Phases 7-10 (--quick, --full, --phases)
├── make_notebook.py            # Builds the demonstration notebook
├── paper/
│   ├── main.tex                # IEEEtran conference paper
│   ├── main.pdf                # Compiled paper (11 pages)
│   ├── references.bib          # Bibliography
│   └── figures/                # The 9 result figures (PDF)
├── src/
│   ├── data_model.py           # Data loading, feature selection, model training
│   ├── value_function.py       # Chunked coalition tabulation
│   ├── exact.py                # Exact Shapley values and efficiency check
│   ├── estimators.py           # The five estimators
│   ├── metrics.py              # Relative error, top-3 agreement, efficiency gap
│   ├── sanity_checks.py        # Correctness tests
│   └── phases/                 # phase1.py ... phase10.py
├── notebooks/
│   └── Shapley_Mid_Sem_Demo.ipynb
├── results/
│   ├── ASSUMPTIONS.md          # Methodological choices
│   ├── RUN_INFO.json           # Machine, library versions, phase runtimes
│   ├── CHANGELOG.md            # Corrections and experiment log
│   ├── RESULTS_SUMMARY.md      # Summary of phases 1-6
│   ├── MID_SEM_TALKING_POINTS.md
│   ├── EXTENSION_SUMMARY.md    # Findings of phases 7-10
│   ├── raw/                    # Raw per-run CSV files
│   ├── tables/                 # All tables (CSV, JSON)
│   └── figures/                # Figures (PNG and PDF)
├── shapley_results/            # Archived snapshot of the phase 1-6 outputs
└── shapley_results.zip         # Archive of the same snapshot
```

`results/` holds the final outputs of all phases. `shapley_results/` and `shapley_results.zip` are an earlier snapshot of the phase 1-6 outputs, kept for reference.
