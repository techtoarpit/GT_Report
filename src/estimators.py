"""Five Shapley value approximation estimators implemented from scratch."""
from __future__ import annotations
import numpy as np


def permutation_sampling(
    v_table: np.ndarray,
    d: int,
    m: int,
    seed: int,
    **kwargs
) -> np.ndarray:
    """
    1. Permutation sampling estimator.
    Draws floor(m / (d + 1)) random permutations. Each permutation costs d + 1 evaluations.
    Averages marginal contributions across permutations.
    """
    P = m // (d + 1)
    if P == 0:
        return np.full(d, np.nan, dtype=np.float64)
        
    rng = np.random.default_rng(seed)
    marginal_sums = np.zeros(d, dtype=np.float64)
    v_empty = v_table[0]
    
    for _ in range(P):
        perm = rng.permutation(d)
        mask = 0
        v_curr = v_empty
        for feat in perm:
            next_mask = mask | (1 << feat)
            v_next = v_table[next_mask]
            marginal_sums[feat] += (v_next - v_curr)
            mask = next_mask
            v_curr = v_next
            
    return marginal_sums / P


def antithetic_permutation(
    v_table: np.ndarray,
    d: int,
    m: int,
    seed: int,
    **kwargs
) -> np.ndarray:
    """
    2. Antithetic permutation estimator.
    Each random permutation is used together with its reverse.
    A pair costs 2(d + 1) evaluations; floor(m / (2(d + 1))) pairs.
    """
    pairs = m // (2 * (d + 1))
    if pairs == 0:
        return np.full(d, np.nan, dtype=np.float64)
        
    rng = np.random.default_rng(seed)
    marginal_sums = np.zeros(d, dtype=np.float64)
    v_empty = v_table[0]
    
    for _ in range(pairs):
        perm = rng.permutation(d)
        # Evaluate forward and reverse permutations
        for p in (perm, perm[::-1]):
            mask = 0
            v_curr = v_empty
            for feat in p:
                next_mask = mask | (1 << feat)
                v_next = v_table[next_mask]
                marginal_sums[feat] += (v_next - v_curr)
                mask = next_mask
                v_curr = v_next
                
    return marginal_sums / (2 * pairs)


def kernel_shap(
    v_table: np.ndarray,
    d: int,
    m: int,
    seed: int,
    **kwargs
) -> np.ndarray:
    """
    3. KernelSHAP estimator.
    Sample m coalitions with size s in 1..d-1 drawn with probability proportional
    to (d-1)/(s(d-s)), then a uniformly random subset of that size.
    Solve least squares with efficiency constraint sum(beta) = v(N) - v(empty)
    imposed exactly by eliminating one coefficient. y = v(S) - v(empty).
    """
    if m < 1:
        return np.full(d, np.nan, dtype=np.float64)
        
    rng = np.random.default_rng(seed)
    v_empty = v_table[0]
    v_N = v_table[-1]
    eff_target = v_N - v_empty
    
    sizes = np.arange(1, d)
    weights = (d - 1.0) / (sizes * (d - sizes))
    probs = weights / weights.sum()
    
    sampled_sizes = rng.choice(sizes, size=m, p=probs)
    Z = np.zeros((m, d), dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)
    
    for j in range(m):
        sz = sampled_sizes[j]
        subset = rng.choice(d, size=sz, replace=False)
        mask = int(np.bitwise_or.reduce(1 << subset))
        Z[j, subset] = 1.0
        y[j] = v_table[mask] - v_empty
        
    # Solve constrained least squares: sum(beta) = eff_target
    # Eliminate beta_{d-1}: beta_{d-1} = eff_target - sum_{i=0}^{d-2} beta_i
    # Z_tilde = Z[:, :d-1] - Z[:, d-1, np.newaxis]
    # y_tilde = y - Z[:, d-1] * eff_target
    Z_tilde = Z[:, :d-1] - Z[:, d-1, np.newaxis]
    y_tilde = y - Z[:, d-1] * eff_target
    
    beta_reduced, _, _, _ = np.linalg.lstsq(Z_tilde, y_tilde, rcond=None)
    beta = np.zeros(d, dtype=np.float64)
    beta[:d-1] = beta_reduced
    beta[d-1] = eff_target - np.sum(beta_reduced)
    
    return beta


def paired_kernel_shap(
    v_table: np.ndarray,
    d: int,
    m: int,
    seed: int,
    **kwargs
) -> np.ndarray:
    """
    4. Paired KernelSHAP estimator.
    Sample m/2 coalitions with size s in 1..d-1 drawn with probability proportional
    to (d-1)/(s(d-s)), and also evaluate each one's complement (m evaluations total).
    Same constrained least squares solve.
    """
    m_half = m // 2
    if m_half < 1:
        return np.full(d, np.nan, dtype=np.float64)
        
    rng = np.random.default_rng(seed)
    v_empty = v_table[0]
    v_N = v_table[-1]
    eff_target = v_N - v_empty
    full_mask = (1 << d) - 1
    
    sizes = np.arange(1, d)
    weights = (d - 1.0) / (sizes * (d - sizes))
    probs = weights / weights.sum()
    
    sampled_sizes = rng.choice(sizes, size=m_half, p=probs)
    m_total = 2 * m_half
    Z = np.zeros((m_total, d), dtype=np.float64)
    y = np.zeros(m_total, dtype=np.float64)
    
    for j in range(m_half):
        sz = sampled_sizes[j]
        subset = rng.choice(d, size=sz, replace=False)
        mask = int(np.bitwise_or.reduce(1 << subset))
        
        # Primary coalition
        Z[2 * j, subset] = 1.0
        y[2 * j] = v_table[mask] - v_empty
        
        # Complement coalition
        comp_mask = full_mask ^ mask
        Z[2 * j + 1] = 1.0 - Z[2 * j]
        y[2 * j + 1] = v_table[comp_mask] - v_empty
        
    Z_tilde = Z[:, :d-1] - Z[:, d-1, np.newaxis]
    y_tilde = y - Z[:, d-1] * eff_target
    
    beta_reduced, _, _, _ = np.linalg.lstsq(Z_tilde, y_tilde, rcond=None)
    beta = np.zeros(d, dtype=np.float64)
    beta[:d-1] = beta_reduced
    beta[d-1] = eff_target - np.sum(beta_reduced)
    
    return beta


def multilinear_extension(
    v_table: np.ndarray,
    d: int,
    m: int,
    seed: int,
    K: int = 4,
    **kwargs
) -> np.ndarray:
    """
    5. Multilinear extension estimator.
    Trapezoid rule over K grid points q in [0, 1] (default K=4).
    One sample draws z with each feature present with prob q and evaluates z plus
    its d single-feature flips (d+1 evaluations); marginal of feature i is v(z with i) - v(z without i).
    q=0 and q=1 are deterministic, so use one sample each; spread remaining floor(m/(d+1))-2
    samples over the K-2 interior points (round-robin, random remainder).
    If fewer interior samples than interior points, return NaN.
    Trapezoid weights are 0.5/(K-1) at endpoints and 1/(K-1) inside.
    """
    T = m // (d + 1)
    num_interior = K - 2
    T_interior = T - 2
    
    # If there are fewer interior samples than interior points, return NaN
    if T_interior < num_interior:
        return np.full(d, np.nan, dtype=np.float64)
        
    rng = np.random.default_rng(seed)
    
    # Allocate samples across interior points: round-robin, random remainder
    base = T_interior // num_interior
    rem = T_interior % num_interior
    counts = np.full(num_interior, base, dtype=int)
    if rem > 0:
        extra_idx = rng.choice(num_interior, size=rem, replace=False)
        counts[extra_idx] += 1
        
    grid_points = np.linspace(0.0, 1.0, K)
    marginals = np.zeros((K, d), dtype=np.float64)
    
    # q = 0 (deterministic): z = empty
    v_empty = v_table[0]
    for i in range(d):
        marginals[0, i] = v_table[1 << i] - v_empty
        
    # q = 1 (deterministic): z = full
    full_mask = (1 << d) - 1
    v_full = v_table[full_mask]
    for i in range(d):
        marginals[K - 1, i] = v_full - v_table[full_mask ^ (1 << i)]
        
    # Interior points: q in grid_points[1 : K-1]
    for idx_int, q in enumerate(grid_points[1 : K - 1]):
        n_samples = counts[idx_int]
        acc_marginals = np.zeros(d, dtype=np.float64)
        for _ in range(n_samples):
            # Draw z with each feature present with prob q
            z_bits = (rng.random(d) < q).astype(int)
            mask = int(np.bitwise_or.reduce(1 << np.where(z_bits == 1)[0])) if np.any(z_bits) else 0
            v_z = v_table[mask]
            for i in range(d):
                flip_mask = mask ^ (1 << i)
                v_flip = v_table[flip_mask]
                if z_bits[i] == 1:
                    # z has i, flip without i
                    acc_marginals[i] += (v_z - v_flip)
                else:
                    # z without i, flip with i
                    acc_marginals[i] += (v_flip - v_z)
        marginals[1 + idx_int] = acc_marginals / n_samples
        
    # Trapezoid integration weights
    weights = np.ones(K, dtype=np.float64) / (K - 1)
    weights[0] = 0.5 / (K - 1)
    weights[-1] = 0.5 / (K - 1)
    
    phi = np.sum(marginals * weights[:, np.newaxis], axis=0)
    return phi


ESTIMATOR_REGISTRY = {
    "Permutation": permutation_sampling,
    "Antithetic Permutation": antithetic_permutation,
    "KernelSHAP": kernel_shap,
    "Paired KernelSHAP": paired_kernel_shap,
    "Multilinear Extension": multilinear_extension,
}
