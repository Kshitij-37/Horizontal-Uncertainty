"""
Proper apples-to-apples LMG variance decomposition comparison:
  - Thesis-final model (z-scored formula A, forest-out, n=38, with bias)
  - exp-M2c (z-scored non-rough + per-formula z-scored + min for roughness,
    forest-in, n=45, no bias)

For each model we compute:
  - naive per-feature share (beta_k^2 * Var(z_k), sums to naive-total)
  - LMG per-feature share (Shapley over all K! orderings, sums to actual variance)

For the LMG comparison, only the 5 sigma-driving features are considered
(bias/mu doesn't enter log(sigma) variance).

Reads the thesis-final JSON directly. Rebuilds pair-level features from the
input Excel using the same aggregation code paths as production.
"""
import os
import sys
import json
import math
import itertools
import importlib.machinery
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_RFC = os.path.normpath(os.path.join(HERE, "..", "roughness_formula_comparison.py"))
_loader = importlib.machinery.SourceFileLoader("rfc", _RFC)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _RFC, loader=_loader))
_loader.exec_module(rfc)
pmod = rfc.pmod

THESIS_JSON  = os.path.normpath(os.path.join(
    HERE, "..", "..", "Bayesian_approach", "Final model", "Results",
    "ws_uncertainty_pairlevel_results.json"))
EXPM2C_JSON  = os.path.join(HERE, "Results", "model_results.json")
EXPM2C_CONTRIB = os.path.join(HERE, "Results", "feature_contributions_per_pair.csv")

FOREST_MASTS = ["2011WM011", "2014WM011", "2012WM006"]


# ─────────────────────────────────────────────────────────────────
# LMG / decomposition core
# ─────────────────────────────────────────────────────────────────
def decompose(betas, cov_matrix, labels):
    """Compute naive, full-covariance (= LMG for fixed betas), and pairwise cross-terms.
    Returns dict with keys naive, lmg, total, cross_terms.
    """
    K = len(betas)
    variances = np.diag(cov_matrix)
    naive = betas ** 2 * variances
    total = float(betas @ cov_matrix @ betas)

    # LMG = Shapley average over K! orderings (equals beta * (Cov beta) for fixed betas)
    # Compute properly via subset enumeration to verify.
    subset_var = {}
    for r in range(K + 1):
        for combo in itertools.combinations(range(K), r):
            idx = list(combo)
            if idx:
                subset_var[frozenset(combo)] = float(
                    betas[idx] @ cov_matrix[np.ix_(idx, idx)] @ betas[idx])
            else:
                subset_var[frozenset(combo)] = 0.0

    lmg = np.zeros(K)
    for k in range(K):
        for r in range(K):
            weight = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
            for combo in itertools.combinations([j for j in range(K) if j != k], r):
                S = frozenset(combo); Sk = S | {k}
                lmg[k] += weight * (subset_var[Sk] - subset_var[S])

    # Pairwise cross-terms
    cross = []
    for i in range(K):
        for j in range(i+1, K):
            term = 2 * betas[i] * betas[j] * cov_matrix[i, j]
            cross.append((labels[i], labels[j], float(cov_matrix[i, j] / np.sqrt(variances[i]*variances[j])),
                           float(term)))
    return {"naive": naive, "lmg": lmg, "total_var": total,
            "naive_sum": float(naive.sum()), "cross_terms": cross,
            "correlations": cov_matrix / np.sqrt(np.outer(variances, variances))}


# ─────────────────────────────────────────────────────────────────
# LOAD & DECOMPOSE — thesis-final
# ─────────────────────────────────────────────────────────────────
def decompose_thesis_final():
    print("=" * 80)
    print("THESIS-FINAL MODEL  (z-scored formula A, forest-out, WITH bias)")
    print("=" * 80)

    with open(THESIS_JSON) as f:
        meta = json.load(f)
    mp = meta["model_params"]
    scalers = meta["scalers"]

    # Feature order — matches thesis
    feature_order = [
        ("gamma_dist",       "dist_sat",           "Saturating distance"),
        ("gamma_turning",    "turning_sat",        "Saturating turning"),
        ("gamma_speedup",    "wm_abs_log_speedup", "WM |log speedup ratio|"),
        ("gamma_roughness",  "roughness_sat",      "Saturating roughness (formula A)"),
        ("gamma_dz",         "dz_sat",             "Saturating |dz|"),
    ]
    betas = np.array([mp[gn] for gn, _, _ in feature_order])
    labels = [disp for _, _, disp in feature_order]
    raw_cols = [rc for _, rc, _ in feature_order]

    # Rebuild pair_df using rfc with formula A and thesis-final exclusions
    _saved = list(pmod.EXCLUDED_MASTS)
    try:
        pair_df, _ = rfc.build_pair_data_with_formula(rfc.load_sector_data(), "A")
    finally:
        pmod.EXCLUDED_MASTS = _saved

    # Z-score using thesis-final stored scalers (so the covariance matrix matches
    # what the model saw at fit time)
    Z_cols = {}
    for gn, raw_col, _ in feature_order:
        mean = scalers[f"{raw_col}_mean"]; std = scalers[f"{raw_col}_std"]
        Z_cols[raw_col] = (pair_df[raw_col].values - mean) / std
    Z = np.column_stack([Z_cols[rc] for rc in raw_cols])

    cov = np.cov(Z, rowvar=False, ddof=1)
    n = Z.shape[0]
    print(f"n_pairs = {n}   (thesis-final training set)")
    print(f"Fitted betas: {dict(zip([gn for gn, _, _ in feature_order], betas.round(4)))}")

    d = decompose(betas, cov, labels)
    return d, labels


# ─────────────────────────────────────────────────────────────────
# LOAD & DECOMPOSE — exp-M2c
# ─────────────────────────────────────────────────────────────────
def decompose_expM2c():
    print("\n" + "=" * 80)
    print("exp-M2c MODEL  (adaptive roughness, forest-in, NO bias)")
    print("=" * 80)

    with open(EXPM2C_JSON) as f:
        meta = json.load(f)
    mp = meta["model_params"]

    # 5 features: 4 non-rough (z-scored) + rough_M2c (min of centered z-scores)
    feature_specs = [
        ("gamma_dist",       "Saturating distance",     "z_dist"),
        ("gamma_turning",    "Saturating turning",      "z_turning"),
        ("gamma_speedup",    "WM |log speedup ratio|",  "z_speedup"),
        ("gamma_dz",         "Saturating |dz|",         "z_dz"),
        ("gamma_roughness",  "Adaptive roughness M2c",  "x_rough"),
    ]
    betas = np.array([mp[gn] for gn, _, _ in feature_specs])
    labels = [disp for _, disp, _ in feature_specs]

    # Reuse the per-pair z-scored / x_rough values from feature_contributions.csv
    # (they were saved by the production model run)
    cdf = pd.read_csv(EXPM2C_CONTRIB)
    z_dist    = cdf["contrib_gamma_dist"]     / mp["gamma_dist"]
    z_turning = cdf["contrib_gamma_turning"]  / mp["gamma_turning"]
    z_speedup = cdf["contrib_gamma_speedup"]  / mp["gamma_speedup"]
    z_dz      = cdf["contrib_gamma_dz"]       / mp["gamma_dz"]
    x_rough   = cdf["x_rough"]
    Z = np.column_stack([z_dist, z_turning, z_speedup, z_dz, x_rough])
    n = Z.shape[0]
    print(f"n_pairs = {n}   (exp-M2c training set, forest-in)")
    print(f"Fitted betas: {dict(zip([gn for gn, _, _ in feature_specs], betas.round(4)))}")

    cov = np.cov(Z, rowvar=False, ddof=1)
    d = decompose(betas, cov, labels)
    return d, labels


# ─────────────────────────────────────────────────────────────────
# PRINT COMPARISON
# ─────────────────────────────────────────────────────────────────
def print_decomp(label, d, labels):
    print(f"\n--- {label} ---")
    print(f"  Total Var(log_sigma): {d['total_var']:.5f}   "
          f"Naive sum: {d['naive_sum']:.5f}   Ratio: {d['total_var']/d['naive_sum']:.2f}x")
    print(f"  {'Feature':<38}{'Naive %':>10}{'LMG %':>10}")
    print("  " + "-" * 58)
    for i in np.argsort(-d["lmg"]):
        print(f"  {labels[i]:<38}"
              f"{100*d['naive'][i]/d['naive_sum']:>9.1f}%"
              f"{100*d['lmg'][i]/d['total_var']:>9.1f}%")


def main():
    thesis, thesis_labels = decompose_thesis_final()
    expM2c, expM2c_labels = decompose_expM2c()

    print("\n" + "=" * 80)
    print("APPLES-TO-APPLES LMG COMPARISON")
    print("=" * 80)
    print_decomp("THESIS-FINAL (formula A, forest-out, with bias)", thesis, thesis_labels)
    print_decomp("exp-M2c (adaptive roughness, forest-in, no bias)", expM2c, expM2c_labels)

    # Common-name comparison table (matching by display order)
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE LMG (aligned by feature name)")
    print("=" * 80)
    print(f"  {'Feature':<32}{'thesis LMG %':>15}{'exp-M2c LMG %':>17}{'shift':>10}")
    print("  " + "-" * 74)
    common_map = {
        "Saturating distance":              "Saturating distance",
        "Saturating turning":               "Saturating turning",
        "WM |log speedup ratio|":           "WM |log speedup ratio|",
        "Saturating |dz|":                  "Saturating |dz|",
        "Saturating roughness (formula A)": "Adaptive roughness M2c",
    }
    t_pct = {thesis_labels[i]: 100*thesis["lmg"][i]/thesis["total_var"] for i in range(len(thesis_labels))}
    e_pct = {expM2c_labels[i]: 100*expM2c["lmg"][i]/expM2c["total_var"] for i in range(len(expM2c_labels))}
    for t_name, e_name in common_map.items():
        tv = t_pct.get(t_name, 0); ev = e_pct.get(e_name, 0)
        shift = ev - tv
        marker = "*" if abs(shift) >= 5.0 else " "
        print(f"  {t_name:<32}{tv:>14.1f}%{ev:>16.1f}%{shift:>+8.1f}pp {marker}")

    # Also compare naive-to-naive for context
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE NAIVE (for context)")
    print("=" * 80)
    print(f"  {'Feature':<32}{'thesis naive %':>17}{'exp-M2c naive %':>19}{'shift':>10}")
    print("  " + "-" * 78)
    t_naive = {thesis_labels[i]: 100*thesis["naive"][i]/thesis["naive_sum"]
               for i in range(len(thesis_labels))}
    e_naive = {expM2c_labels[i]: 100*expM2c["naive"][i]/expM2c["naive_sum"]
               for i in range(len(expM2c_labels))}
    for t_name, e_name in common_map.items():
        tv = t_naive.get(t_name, 0); ev = e_naive.get(e_name, 0)
        shift = ev - tv
        marker = "*" if abs(shift) >= 5.0 else " "
        print(f"  {t_name:<32}{tv:>16.1f}%{ev:>18.1f}%{shift:>+8.1f}pp {marker}")


if __name__ == "__main__":
    main()
