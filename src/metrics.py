"""Evaluation metrics for Shapley value approximations."""
from __future__ import annotations
import numpy as np


def compute_relative_error(phi_hat: np.ndarray, phi_exact: np.ndarray) -> float:
    """
    Relative error: ||phi_hat - phi_exact||_2 / ||phi_exact||_2
    Returns NaN if phi_hat contains NaNs.
    """
    if np.isnan(phi_hat).any():
        return np.nan
    norm_exact = np.linalg.norm(phi_exact)
    if norm_exact == 0.0:
        return 0.0
    return float(np.linalg.norm(phi_hat - phi_exact) / norm_exact)


def compute_top3_agreement(phi_hat: np.ndarray, phi_exact: np.ndarray) -> float:
    """
    Top-3 agreement: fraction of the 3 largest-|phi| exact features also in the estimate's top 3.
    Returns NaN if phi_hat contains NaNs.
    """
    if np.isnan(phi_hat).any():
        return np.nan
    # Indices of top-3 largest absolute values
    top3_exact = set(np.argsort(np.abs(phi_exact))[-3:])
    top3_hat = set(np.argsort(np.abs(phi_hat))[-3:])
    overlap = len(top3_exact.intersection(top3_hat))
    return float(overlap / 3.0)


def compute_efficiency_gap(phi_hat: np.ndarray, eff_target: float) -> float:
    """
    Efficiency gap: |sum(phi_hat) - (v(N) - v(empty))|
    Returns NaN if phi_hat contains NaNs.
    """
    if np.isnan(phi_hat).any():
        return np.nan
    return float(abs(np.sum(phi_hat) - eff_target))
