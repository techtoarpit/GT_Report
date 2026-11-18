# Methodological Assumptions & Specification Choices

This document details all implementation choices, interpretations, and assumptions adopted in the experimental study of efficient Shapley value approximation, in accordance with the project specification.

## 1. Dataset & Preprocessing
- **Source**: `sklearn.datasets.load_breast_cancer` (569 samples, 30 features).
- **Split**: 70% train (398 samples), 30% test (171 samples), stratified on binary labels $y$, with `random_state=0`.
- **Feature Selection**: Top-$d$ features selected strictly using `mutual_info_classif(X_train, y_train, random_state=0)` on the training split only. Indices are ordered descending by mutual information score.
- **Model**: `HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)`. The model is retrained on the selected top-$d$ features for each dimension $d$ evaluated.
- **Explanation Target**: Log-odds output via `model.decision_function(X)`.

## 2. Background and Explicands
- **Background Set**: 100 points selected from the training split without replacement using `numpy.random.default_rng(0).choice(len(X_train), size=100, replace=False)`.
- **Explicands**: Test instances selected without replacement using `numpy.random.default_rng(0).choice(len(X_test), size=n, replace=False)`.
  - Phase 1 & 4 & 5: $n=10$ explicands.
  - Phase 3 ($d=16$): $n=5$ explicands.
  - Phase 2: $n=1$ explicand (the first explicand, index 0).

## 3. Value Function & Exact Shapley
- **Interventional Value Function**: For a coalition $S \subseteq \{0, \dots, d-1\}$, $v(S) = \frac{1}{|B|} \sum_{b \in B} f(x_S, b_{N \setminus S})$, where features in $S$ are taken from explicand $x$ and features outside $S$ are drawn from background point $b$.
- **Tabulation**: Precomputed for all $2^d$ coalitions once per explicand and stored in an indexed array for $O(1)$ coalition lookups. Evaluated in batches over coalitions to ensure RAM consumption remains well below 2 GB.
- **Exact Shapley**: Evaluated using the classical formulation:
  $$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (d - |S| - 1)!}{d!} (v(S \cup \{i\}) - v(S))$$
  Verified to satisfy efficiency $\sum_{i=1}^d \phi_i = v(N) - v(\emptyset)$ to $< 10^{-11}$.

## 4. Estimator Implementations
- **Budget $m$**: Represents coalition evaluations (table lookups). $v(\emptyset)$ and $v(N)$ are free.
- **Permutation Sampling**: Draws $P = \lfloor m / (d+1) \rfloor$ random permutations. Each permutation evaluates the chain of $d+1$ nested prefixes, averaging the marginal contributions.
- **Antithetic Permutation**: Draws $P = \lfloor m / (2(d+1)) \rfloor$ random permutations and pairs each with its exact reverse. Each pair evaluates $2(d+1)$ prefixes/complements, averaging marginal contributions.
- **KernelSHAP**:
  - Samples $m$ coalitions. Coalition size $s \in \{1, \dots, d-1\}$ is drawn with probability proportional to $\frac{d-1}{s(d-s)}$, followed by a uniformly random subset of size $s$.
  - Constrained least squares with efficiency constraint $\sum_{i=1}^d \beta_i = v(N) - v(\emptyset)$ solved by eliminating feature $d-1$: $\beta_{d-1} = (v(N) - v(\emptyset)) - \sum_{i=1}^{d-2} \beta_i$.
- **Paired KernelSHAP**:
  - Samples $m/2$ coalitions identically to KernelSHAP, and includes both each sampled coalition $S$ and its exact complement $N \setminus S$, totaling $m$ evaluations.
  - Solves the same constrained least squares problem.
- **Multilinear Extension**:
  - Discretizes $[0, 1]$ into $K$ uniform grid points $q_k = k / (K-1)$ for $k \in \{0, \dots, K-1\}$.
  - Endpoints $q=0$ and $q=1$ are evaluated with 1 deterministic sample each ($d+1$ evaluations each).
  - The remaining $T_{\text{interior}} = \lfloor m / (d+1) \rfloor - 2$ samples are allocated across the $K-2$ interior points using round-robin with random remainder.
  - If $T_{\text{interior}} < K - 2$, the method cannot cover all interior grid points and returns `NaN`.
  - At each grid point, a sample draws $z \sim \text{Bernoulli}(q)^{\otimes d}$ and evaluates $z$ plus its $d$ single-feature flips ($d+1$ evaluations). Feature $i$'s marginal is $v(z \text{ with } i) - v(z \text{ without } i)$.
  - Integration uses trapezoid weights: $0.5 / (K-1)$ at endpoints, $1.0 / (K-1)$ at interior points.

## 5. Random Seeds & Aggregation
- **Seeding**: For explicand $i$ (0-indexed) and repeat $s$ (0-indexed), the estimator seed is strictly $1000 \times i + s$.
- **Aggregation Protocol**: For any metric and budget, values are first averaged across the 20 seeds for each explicand, and then the mean and sample standard error ($\text{std} / \sqrt{n}$) are reported across the $n$ explicands.
- **Log-log Fit & Interpolation**:
  - Log-log slope is computed by ordinary least squares linear regression on $(\log_{10}(m), \log_{10}(\text{mean rel\_err}))$.
  - Interpolation to find matching budget $m^*$ where Paired KernelSHAP reaches Permutation's $m=2048$ error is performed linearly in log-log space.
