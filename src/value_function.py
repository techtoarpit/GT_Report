"""Interventional value function computation and coalition tabulation."""
from __future__ import annotations
import time
import numpy as np


def tabulate_value_function(
    model,
    explicand: np.ndarray,
    background: np.ndarray,
    chunk_size: int = 1024
) -> tuple[np.ndarray, float, int]:
    """
    Tabulate interventional value function v(S) for all 2^d coalitions.
    v(S) = mean over background b of f(x_S, b_{N \\ S}), where f is decision_function.
    
    Parameters
    ----------
    model : fitted classifier with decision_function
    explicand : 1D array of shape (d,)
    background : 2D array of shape (n_bg, d), typically (100, d)
    chunk_size : int, number of coalitions processed per batch to bound memory
    
    Returns
    -------
    v_table : 1D array of shape (2^d,) containing v(S) for mask S in [0, 2^d - 1]
    elapsed_time : float, tabulation time in seconds
    forward_passes : int, number of model prediction calls (2^d * len(background))
    """
    d = len(explicand)
    n_bg = len(background)
    num_coalitions = 1 << d
    v_table = np.zeros(num_coalitions, dtype=np.float64)
    
    t0 = time.perf_counter()
    
    # Process coalitions in chunks to guarantee memory stays well under 2 GB
    for c_start in range(0, num_coalitions, chunk_size):
        c_end = min(c_start + chunk_size, num_coalitions)
        batch_size = c_end - c_start
        masks = np.arange(c_start, c_end)
        
        # Boolean mask tensor of shape (batch_size, 1, d)
        # Bit j of masks indicates if feature j is in coalition
        Z = ((masks[:, np.newaxis] & (1 << np.arange(d))) != 0)[:, np.newaxis, :]
        
        # Replace background features with explicand features when Z is True
        # Shape: (batch_size, n_bg, d) -> (batch_size * n_bg, d)
        batch_X = np.where(Z, explicand, background[np.newaxis, :, :]).reshape(batch_size * n_bg, d)
        
        preds = model.decision_function(batch_X)
        v_table[c_start:c_end] = preds.reshape(batch_size, n_bg).mean(axis=1)
        
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    forward_passes = num_coalitions * n_bg
    
    # Verify correctness: v(empty) == mean f(background), v(N) == f(explicand)
    f_bg_mean = model.decision_function(background).mean()
    f_x = model.decision_function(explicand[np.newaxis, :])[0]
    
    assert np.isclose(v_table[0], f_bg_mean, atol=1e-7), f"v(empty) {v_table[0]} != f(B) {f_bg_mean}"
    assert np.isclose(v_table[-1], f_x, atol=1e-7), f"v(N) {v_table[-1]} != f(x) {f_x}"
    
    return v_table, elapsed_time, forward_passes
