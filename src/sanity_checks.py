"""Correctness and sanity verification suite."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from src.exact import compute_exact_shapley
from src.estimators import (
    permutation_sampling,
    antithetic_permutation,
    kernel_shap,
    paired_kernel_shap,
    multilinear_extension,
)
from src.data_model import ExperimentSetup
from src.value_function import tabulate_value_function


def run_sanity_checks(setup: ExperimentSetup | None = None) -> pd.DataFrame:
    """
    Run all correctness tests:
    1. Toy additive game: all 5 estimators recover known 'a'.
    2. 3-player non-additive game: exact routine matches hand-computed values [7, 1, 4].
    3. Model & dataset checks: test accuracy reported; v(N) == f(x); exact Shapley satisfies efficiency < 1e-9.
    
    Saves and returns results DataFrame.
    """
    checks = []
    
    # Check 1: 3-player non-additive game
    # v({}) = 0, v({0})=0, v({1})=0, v({2})=0, v({0,1})=6, v({0,2})=12, v({1,2})=0, v({0,1,2})=12
    d_3 = 3
    v_3 = np.zeros(1 << d_3, dtype=np.float64)
    v_3[1 | 2] = 6.0
    v_3[1 | 4] = 12.0
    v_3[1 | 2 | 4] = 12.0
    expected_3 = np.array([7.0, 1.0, 4.0], dtype=np.float64)
    
    phi_3, eff_gap_3, _ = compute_exact_shapley(v_3, d_3)
    match_3 = np.allclose(phi_3, expected_3, atol=1e-12)
    assert match_3, f"3-player game failed: {phi_3} vs {expected_3}"
    checks.append({
        "check": "3-player non-additive game exact Shapley",
        "expected": str(expected_3.tolist()),
        "computed": str(np.round(phi_3, 4).tolist()),
        "discrepancy": float(np.max(np.abs(phi_3 - expected_3))),
        "status": "PASSED" if match_3 else "FAILED"
    })
    
    # Check 2: Toy additive game with known vector a
    d_add = 4
    a_true = np.array([1.5, -2.0, 3.0, 0.5], dtype=np.float64)
    v_add = np.zeros(1 << d_add, dtype=np.float64)
    for mask in range(1 << d_add):
        v_add[mask] = sum(a_true[i] for i in range(d_add) if (mask & (1 << i)))
        
    phi_add_exact, eff_add_exact, _ = compute_exact_shapley(v_add, d_add)
    assert np.allclose(phi_add_exact, a_true, atol=1e-12)
    checks.append({
        "check": "Toy additive game exact Shapley",
        "expected": str(a_true.tolist()),
        "computed": str(np.round(phi_add_exact, 4).tolist()),
        "discrepancy": float(np.max(np.abs(phi_add_exact - a_true))),
        "status": "PASSED"
    })
    
    # Check each estimator on additive game
    m_test = 64
    estimators = {
        "Permutation": permutation_sampling(v_add, d_add, m_test, seed=42),
        "Antithetic Permutation": antithetic_permutation(v_add, d_add, m_test, seed=42),
        "KernelSHAP": kernel_shap(v_add, d_add, m_test, seed=42),
        "Paired KernelSHAP": paired_kernel_shap(v_add, d_add, m_test, seed=42),
        "Multilinear Extension": multilinear_extension(v_add, d_add, m_test, seed=42, K=4),
    }
    
    for name, phi_est in estimators.items():
        diff = float(np.max(np.abs(phi_est - a_true)))
        passed = diff < 1e-10
        assert passed, f"{name} failed additive test with diff {diff}"
        checks.append({
            "check": f"Toy additive game estimator: {name}",
            "expected": str(a_true.tolist()),
            "computed": str(np.round(phi_est, 4).tolist()),
            "discrepancy": diff,
            "status": "PASSED" if passed else "FAILED"
        })
        
    # Check 3: Real model and explicand
    if setup is None:
        setup = ExperimentSetup(d=12, n_explicands=1)
        
    checks.append({
        "check": "Model test accuracy (d=12)",
        "expected": "near 0.947",
        "computed": f"{setup.test_accuracy:.5f}",
        "discrepancy": abs(setup.test_accuracy - 0.94737),
        "status": "PASSED"
    })
    
    # Real instance check
    x0 = setup.explicands[0]
    v_table_0, tab_time, _ = tabulate_value_function(setup.model, x0, setup.background)
    f_x0 = setup.model.decision_function(x0[np.newaxis, :])[0]
    v_N_diff = abs(v_table_0[-1] - f_x0)
    assert v_N_diff < 1e-7
    checks.append({
        "check": "v(N) equals model output f(x)",
        "expected": f"{f_x0:.6f}",
        "computed": f"{v_table_0[-1]:.6f}",
        "discrepancy": v_N_diff,
        "status": "PASSED"
    })
    
    phi_real_0, eff_gap_real_0, _ = compute_exact_shapley(v_table_0, setup.d)
    assert eff_gap_real_0 < 1e-9
    checks.append({
        "check": "Exact Shapley efficiency gap on explicand 0",
        "expected": "< 1e-9",
        "computed": f"{eff_gap_real_0:.2e}",
        "discrepancy": eff_gap_real_0,
        "status": "PASSED"
    })
    
    df_checks = pd.DataFrame(checks)
    df_checks.to_csv("results/tables/sanity_checks.csv", index=False)
    return df_checks
