"""Exact Shapley value calculation from precomputed coalition value table."""
from __future__ import annotations
import math
import time
import numpy as np


def compute_exact_shapley(v_table: np.ndarray, d: int) -> tuple[np.ndarray, float, float]:
    """
    Compute exact Shapley values from the tabulated coalition value function v.
    phi_i = sum_{S subseteq N \\ {i}} [ |S|! (d - |S| - 1)! / d! ] * [ v(S + {i}) - v(S) ]
    
    Parameters
    ----------
    v_table : 1D array of shape (2^d,)
    d : int, number of features
    
    Returns
    -------
    phi : 1D array of shape (d,), exact Shapley values
    eff_gap : float, absolute difference between sum(phi) and v(N) - v(empty)
    elapsed_time : float, computation time in seconds
    """
    t0 = time.perf_counter()
    num_coalitions = 1 << d
    assert len(v_table) == num_coalitions, f"v_table length {len(v_table)} != 2^{d}"
    
    # Precompute weights for subset sizes s in {0, ..., d - 1}
    weights = np.array([
        math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
        for s in range(d)
    ], dtype=np.float64)
    
    masks = np.arange(num_coalitions, dtype=np.int64)
    # Subset sizes (number of set bits) for each mask
    sizes = np.array([bin(m).count('1') for m in range(num_coalitions)], dtype=np.int32)
    
    phi = np.zeros(d, dtype=np.float64)
    for i in range(d):
        bit = 1 << i
        # Subsets S not containing feature i
        no_i_masks = masks[(masks & bit) == 0]
        with_i_masks = no_i_masks | bit
        marginals = v_table[with_i_masks] - v_table[no_i_masks]
        phi[i] = np.sum(weights[sizes[no_i_masks]] * marginals)
        
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    
    # Efficiency check
    eff_target = v_table[-1] - v_table[0]
    eff_gap = abs(float(np.sum(phi) - eff_target))
    assert eff_gap < 1e-9, f"Exact Shapley efficiency violated: gap={eff_gap}"
    
    return phi, eff_gap, elapsed_time
