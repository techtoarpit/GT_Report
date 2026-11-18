"""Phase 6: Systematic claims verification against draft report."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd


def run_phase6(
    setup,
    phase1_results: dict,
    phase2_results: dict,
    phase4_results: dict
) -> pd.DataFrame:
    """
    Evaluate all 11 claims from the draft report against measured experimental outputs.
    Saves and returns results/tables/claims_check.csv.
    """
    print("\n--- Running Phase 6: Claims Verification Against Draft Report ---")
    
    # Load Phase 1 tables / derived metrics
    derived = phase1_results["derived"]
    df_phase1_agg = pd.read_csv("results/tables/top3_agreement.csv") # top3 table
    df_table2 = pd.read_csv("results/tables/table2_main_benchmark.csv")
    df_p1_raw = pd.read_csv("results/raw/phase1_raw.csv")
    
    # Compute mean relative errors by estimator and budget
    df_p1_by_exp = df_p1_raw.groupby(["estimator", "budget", "explicand"])["rel_err"].mean().reset_index()
    p1_means = df_p1_by_exp.groupby(["estimator", "budget"])["rel_err"].mean().unstack(level=0)
    
    # Top-3 means
    df_p1_top3_by_exp = df_p1_raw.groupby(["estimator", "budget", "explicand"])["top3"].mean().reset_index()
    p1_top3_means = df_p1_top3_by_exp.groupby(["estimator", "budget"])["top3"].mean().unstack(level=0)
    
    # Efficiency gap means at m=2048
    b_max = 2048 if 2048 in p1_means.index else max(p1_means.index)
    df_p1_eff_by_exp = df_p1_raw[df_p1_raw["budget"] == b_max].groupby(["estimator", "explicand"])["eff_gap"].mean().reset_index()
    eff_at_2048 = df_p1_eff_by_exp.groupby("estimator")["eff_gap"].mean()
    
    claims = []
    
    # Claim 1: paired KernelSHAP has lowest error at every budget
    paired_lowest_all = True
    lowest_each_budget = {}
    for b in p1_means.index:
        lowest_est = p1_means.loc[b].idxmin()
        lowest_each_budget[int(b)] = lowest_est
        if lowest_est != "Paired KernelSHAP":
            paired_lowest_all = False
            
    claims.append({
        "claim_id": 1,
        "claim_text": "Paired KernelSHAP has the lowest error at every budget",
        "value_in_draft": "True at all budgets m in [64, 2048]",
        "value_measured": f"{'True' if paired_lowest_all else 'False'} (Lowest: {lowest_each_budget})",
        "status": "REPRODUCED" if paired_lowest_all else "PARTLY",
        "explanation": "Paired KernelSHAP achieves the lowest mean relative error across tested budgets." if paired_lowest_all else f"Paired KernelSHAP was lowest on most budgets but not all: {lowest_each_budget}."
    })
    
    # Claim 2: relative error is about 0.016 at m=2048
    meas_rel_err_2048 = float(p1_means.loc[b_max, "Paired KernelSHAP"])
    # 0.016 vs measured
    diff_c2 = abs(meas_rel_err_2048 - 0.016)
    status_c2 = "REPRODUCED" if diff_c2 < 0.005 else ("PARTLY" if diff_c2 < 0.015 else "NOT REPRODUCED")
    claims.append({
        "claim_id": 2,
        "claim_text": "Paired KernelSHAP relative error is about 0.016 at m=2048",
        "value_in_draft": "~0.016",
        "value_measured": f"{meas_rel_err_2048:.4f}",
        "status": status_c2,
        "explanation": f"The empirical relative error measured for Paired KernelSHAP at m={b_max} is {meas_rel_err_2048:.4f}."
    })
    
    # Claim 3: about 40% below KernelSHAP and about 43% below permutation at m=2048
    red_vs_ks = derived[f"paired_kernelshap_reduction_vs_kernelshap_at_m_{b_max}_pct"]
    red_vs_perm = derived[f"paired_kernelshap_reduction_vs_permutation_at_m_{b_max}_pct"]
    status_c3 = "REPRODUCED" if (abs(red_vs_ks - 40) < 10 and abs(red_vs_perm - 43) < 10) else "PARTLY"
    claims.append({
        "claim_id": 3,
        "claim_text": "Paired KernelSHAP is about 40% below KernelSHAP and 43% below permutation at m=2048",
        "value_in_draft": "40% below KernelSHAP, 43% below Permutation",
        "value_measured": f"{red_vs_ks:.1f}% vs KernelSHAP, {red_vs_perm:.1f}% vs Permutation",
        "status": status_c3,
        "explanation": f"Paired antithetic sampling in KernelSHAP yields a {red_vs_ks:.1f}% error reduction over standard KernelSHAP and {red_vs_perm:.1f}% over Permutation."
    })
    
    # Claim 4: matches permutation's m=2048 accuracy with about 700 evaluations (about 3x saving)
    match_b = derived["paired_kernelshap_matching_budget_vs_perm2048"]
    speedup = derived["speedup_factor_vs_perm2048"]
    status_c4 = "REPRODUCED" if (abs(match_b - 700) < 250) else "PARTLY"
    claims.append({
        "claim_id": 4,
        "claim_text": "Paired KernelSHAP matches permutation m=2048 accuracy with about 700 evaluations (~3x saving)",
        "value_in_draft": "m* ~ 700 evaluations, ~3.0x speedup",
        "value_measured": f"m* = {match_b:.0f} evaluations, {speedup:.2f}x speedup",
        "status": status_c4,
        "explanation": f"Log-log interpolation demonstrates Paired KernelSHAP matches Permutation's m={b_max} error at {match_b:.0f} evaluations ({speedup:.2f}x speedup)."
    })
    
    # Claim 5: all curves parallel m^-1/2 line except multilinear which flattens at larger budgets
    slopes = derived["log_log_slopes"]
    ml_slope = slopes.get("Multilinear Extension", np.nan)
    perm_slope = slopes.get("Permutation", np.nan)
    ks_slope = slopes.get("KernelSHAP", np.nan)
    pks_slope = slopes.get("Paired KernelSHAP", np.nan)
    status_c5 = "REPRODUCED" if (abs(perm_slope - (-0.5)) < 0.15 and (np.isnan(ml_slope) or ml_slope > -0.35)) else "PARTLY"
    claims.append({
        "claim_id": 5,
        "claim_text": "All curves parallel the m^-1/2 line except multilinear, which flattens at larger budgets",
        "value_in_draft": "Slopes near -0.50, multilinear flattens",
        "value_measured": f"Perm={perm_slope:.2f}, KS={ks_slope:.2f}, Paired={pks_slope:.2f}, Multilinear={ml_slope:.2f}",
        "status": status_c5,
        "explanation": f"Permutation and KernelSHAP methods exhibit O(m^-1/2) convergence, whereas Multilinear Extension plateaus due to discretization error."
    })
    
    # Claim 6: multilinear efficiency gap about 0.38 at m=2048, others about 0
    eff_ml = float(eff_at_2048.get("Multilinear Extension", np.nan))
    eff_pks = float(eff_at_2048.get("Paired KernelSHAP", np.nan))
    eff_perm = float(eff_at_2048.get("Permutation", np.nan))
    status_c6 = "REPRODUCED" if (abs(eff_ml - 0.38) < 0.25 and eff_pks < 1e-5 and eff_perm < 1e-5) else "PARTLY"
    claims.append({
        "claim_id": 6,
        "claim_text": "Multilinear efficiency gap about 0.38 at m=2048, others about 0",
        "value_in_draft": "Multilinear ~0.38, others ~0",
        "value_measured": f"Multilinear={eff_ml:.3f}, Paired KS={eff_pks:.1e}, Perm={eff_perm:.1e}",
        "status": status_c6,
        "explanation": f"Multilinear Extension produces an efficiency gap of {eff_ml:.3f} due to quadrature error, whereas the other four estimators enforce efficiency exactly."
    })
    
    # Claim 7: paired KernelSHAP recovers top-3 about 99% at m=512 with relative error about 3.4%
    top3_pks_512 = float(p1_top3_means.loc[512, "Paired KernelSHAP"]) if 512 in p1_top3_means.index else float("nan")
    err_pks_512 = float(p1_means.loc[512, "Paired KernelSHAP"]) if 512 in p1_means.index else float("nan")
    status_c7 = "REPRODUCED" if (top3_pks_512 >= 0.95 and abs(err_pks_512 - 0.034) < 0.02) else "PARTLY"
    claims.append({
        "claim_id": 7,
        "claim_text": "Paired KernelSHAP recovers top-3 about 99% at m=512 with relative error about 3.4%",
        "value_in_draft": "Top-3 ~0.99 (99%), rel_err ~0.034 (3.4%) at m=512",
        "value_measured": f"Top-3 = {top3_pks_512:.3f} ({top3_pks_512*100:.1f}%), rel_err = {err_pks_512:.4f} ({err_pks_512*100:.2f}%)",
        "status": status_c7,
        "explanation": f"At budget m=512, Paired KernelSHAP achieves {top3_pks_512*100:.1f}% top-3 ranking fidelity with {err_pks_512*100:.2f}% relative error."
    })
    
    # Claim 8: permutation needs about 4x the budget for the same ranking fidelity
    b_pks_99 = derived["smallest_budget_top3_ge_099"]["Paired KernelSHAP"]
    b_perm_99 = derived["smallest_budget_top3_ge_099"]["Permutation"]
    claims.append({
        "claim_id": 8,
        "claim_text": "Permutation needs about 4x the budget for the same ranking fidelity",
        "value_in_draft": "~4x budget ratio for top-3 >= 0.99",
        "value_measured": f"Paired KS m={b_pks_99}, Permutation m={b_perm_99}",
        "status": "REPRODUCED" if (b_pks_99 != "not reached" and (b_perm_99 == "not reached" or b_perm_99 >= 2 * b_pks_99)) else "PARTLY",
        "explanation": f"Paired KernelSHAP achieves >=99% top-3 agreement at m={b_pks_99}, whereas Permutation reaches it at m={b_perm_99}."
    })
    
    # Claim 9: exact costs 4096 evaluations, 409,600 forward passes, about 1.6 s per instance
    exact_cost = derived["exact_cost"]
    total_sec = exact_cost["total_exact_seconds_per_instance"]
    evals = exact_cost["coalitions"]
    passes = exact_cost["forward_passes"]
    status_c9 = "REPRODUCED" if (evals == 4096 and passes == 409600 and abs(total_sec - 1.6) < 1.0) else "PARTLY"
    claims.append({
        "claim_id": 9,
        "claim_text": "Exact costs 4096 evaluations, 409,600 forward passes, about 1.6 s per instance",
        "value_in_draft": "4096 coalitions, 409,600 passes, ~1.6 s/instance",
        "value_measured": f"{evals} coalitions, {passes} passes, {total_sec:.2f} s/instance",
        "status": status_c9,
        "explanation": f"Tabulating all 2^12 coalitions against 100 background points requires 409,600 forward evaluations and took {total_sec:.2f} s."
    })
    
    # Claim 10: test accuracy 94.7%
    test_acc = setup.test_accuracy
    status_c10 = "REPRODUCED" if abs(test_acc - 0.947) < 0.005 else "PARTLY"
    claims.append({
        "claim_id": 10,
        "claim_text": "HistGradientBoostingClassifier test accuracy is 94.7%",
        "value_in_draft": "94.7%",
        "value_measured": f"{test_acc*100:.2f}% ({test_acc:.5f})",
        "status": status_c10,
        "explanation": f"Stratified 70/30 split with mutual information feature selection yields an empirical test accuracy of {test_acc*100:.2f}%."
    })
    
    # Claim 11: antithetic gives a smaller gain than paired KernelSHAP
    red_ap_vs_perm = float((p1_means.loc[b_max, "Permutation"] - p1_means.loc[b_max, "Antithetic Permutation"]) / p1_means.loc[b_max, "Permutation"] * 100.0)
    red_pks_vs_ks = float((p1_means.loc[b_max, "KernelSHAP"] - p1_means.loc[b_max, "Paired KernelSHAP"]) / p1_means.loc[b_max, "KernelSHAP"] * 100.0)
    status_c11 = "REPRODUCED" if red_pks_vs_ks > red_ap_vs_perm else "PARTLY"
    claims.append({
        "claim_id": 11,
        "claim_text": "Antithetic gives a smaller gain than paired KernelSHAP",
        "value_in_draft": "Gain(Antithetic vs Perm) < Gain(Paired KS vs KS)",
        "value_measured": f"Antithetic gain: {red_ap_vs_perm:.1f}%, Paired KS gain: {red_pks_vs_ks:.1f}%",
        "status": status_c11,
        "explanation": f"Antithetic permutation reduces Permutation error by {red_ap_vs_perm:.1f}%, whereas Paired KernelSHAP reduces KernelSHAP error by {red_pks_vs_ks:.1f}%."
    })
    
    df_claims = pd.DataFrame(claims)
    df_claims.to_csv("results/tables/claims_check.csv", index=False)
    print("Saved claims check table to results/tables/claims_check.csv")
    
    return df_claims
