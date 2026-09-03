"""
Proper variance decomposition for exp-M2c, accounting for feature correlations.

The naive `beta_k**2` decomposition assumes features are independent. In reality,
the 5 features are correlated (e.g., log_speedup vs dz at 0.90), which means the
total variance of log(sigma) is NOT simply the sum of squared coefficients — it
has covariance cross-terms.

This script computes three decompositions:
  1. NAIVE          — beta_k^2 * Var(feature_k), assumes independence
                      (what the model script currently prints)
  2. FULL COVARIANCE — each feature k's share = beta_k * (sum_j beta_j * Cov(k,j)) / total_var
                      Attributes the full covariance to the row-feature. Sums to 100%.
  3. LMG / SHAPLEY  — averages marginal contribution over all K! orderings.
                      Fair symmetric attribution. Sums to total variance.

Also prints the pairwise covariance cross-terms so we can see WHERE the shared
variance actually lives.

Reads from:
  - Results/model_results.json                    (fitted coefficients)
  - Results/feature_contributions_per_pair.csv    (per-pair z-scores)
"""
import os
import json
import itertools
import math

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH    = os.path.join(HERE, "Results", "model_results.json")
CONTRIB_PATH = os.path.join(HERE, "Results", "feature_contributions_per_pair.csv")

# Feature order (matches the model script)
NON_ROUGH_FEATS = [
    ("gamma_dist",    "dist_sat",            "Saturating distance"),
    ("gamma_turning", "turning_sat",         "Saturating turning"),
    ("gamma_speedup", "wm_abs_log_speedup",  "WM |log speedup ratio|"),
    ("gamma_dz",      "dz_sat",              "Saturating |dz|"),
]
ROUGH_GAMMA = "gamma_roughness"
ROUGH_LABEL = "Adaptive roughness M2c"

# All 5 features in order (display, gamma_name, "column key")
FEATURES = (
    [(disp, gn, gn.replace("gamma_", "z_")) for gn, _, disp in NON_ROUGH_FEATS]
    + [(ROUGH_LABEL, ROUGH_GAMMA, "x_rough")]
)

# ── load coefficients ────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    meta = json.load(f)
mp = meta["model_params"]
betas = np.array([mp[gn] for _, gn, _ in FEATURES])
labels = [disp for disp, _, _ in FEATURES]
keys   = [k for _, _, k in FEATURES]
K      = len(FEATURES)

# ── recover per-pair z-scored features from the contributions CSV ────────────
# contrib_gamma_X = gamma_X * z_X, so z_X = contrib / gamma
cdf = pd.read_csv(CONTRIB_PATH)
z_data = {}
for disp, gn, key in FEATURES:
    if key == "x_rough":
        z_data[key] = cdf["x_rough"].values      # already stored directly
    else:
        contrib_col = f"contrib_{gn}"
        z_data[key] = cdf[contrib_col].values / mp[gn]

Z = np.column_stack([z_data[k] for k in keys])   # shape (n_pairs, 5)
n = Z.shape[0]

# ── covariance and correlation matrices ──────────────────────────────────────
cov_matrix  = np.cov(Z, rowvar=False, ddof=1)     # 5x5 covariance
corr_matrix = np.corrcoef(Z, rowvar=False)         # 5x5 correlation
variances   = np.diag(cov_matrix)                  # per-feature variance

print("=" * 80)
print("exp-M2c PROPER VARIANCE DECOMPOSITION")
print("=" * 80)
print(f"\nn_pairs = {n}")
print(f"log_sigma0 posterior mean = {mp['log_sigma0']:.4f}\n")

# ── feature variances (would be 1 for z-scored features; x_rough differs) ────
print("Per-feature variance (across training pairs):")
print(f"  {'Feature':<32}{'Var(feature)':>15}{'beta':>10}")
print("  " + "-" * 57)
for i, disp in enumerate(labels):
    print(f"  {disp:<32}{variances[i]:>15.4f}{betas[i]:>10.4f}")

# ── correlation matrix ───────────────────────────────────────────────────────
print("\nCorrelation matrix (rho_ij):")
print("  " + "".join(f"{disp[:12]:>13}" for disp in labels))
for i, disp in enumerate(labels):
    print(f"  {disp[:12]:<13} " + "  ".join(f"{corr_matrix[i,j]:>+.3f}" for j in range(K)))

# ── total variance of log_sigma ──────────────────────────────────────────────
total_var = float(betas @ cov_matrix @ betas)
naive_sum = float(np.sum(betas**2 * variances))

print(f"\nTotal Var(log_sigma) predictions: {total_var:.5f}")
print(f"Naive sum (ignoring covariance):  {naive_sum:.5f}")
print(f"Ratio actual/naive:               {total_var / naive_sum:.3f}  "
      f"({'features cooperate on average' if total_var > naive_sum else 'features cancel on average'})")

# ── DECOMPOSITION 1: naive (beta^2 * Var) ────────────────────────────────────
print("\n" + "=" * 80)
print("DECOMPOSITION 1 — NAIVE  (beta_k^2 * Var(feature_k), assumes independence)")
print("=" * 80)
naive_contribs = betas**2 * variances
print(f"  {'Feature':<32}{'contrib':>12}{'% of naive':>12}")
print("  " + "-" * 56)
for i in np.argsort(-naive_contribs):
    print(f"  {labels[i]:<32}{naive_contribs[i]:>12.5f}"
          f"{100*naive_contribs[i]/naive_sum:>11.1f}%")

# ── DECOMPOSITION 2: FULL COVARIANCE (row-attributed) ────────────────────────
# For each feature k, its share = beta_k * (Cov beta)_k / total_var
# Sums to 100% by construction (since sum_k beta_k * (Cov beta)_k = beta^T Cov beta)
print("\n" + "=" * 80)
print("DECOMPOSITION 2 — FULL COVARIANCE (attributes cross-terms to row-feature)")
print("=" * 80)
row_contribs = betas * (cov_matrix @ betas)   # per-feature "row" contribution
print(f"  {'Feature':<32}{'contrib':>12}{'% of total':>12}")
print("  " + "-" * 56)
for i in np.argsort(-row_contribs):
    print(f"  {labels[i]:<32}{row_contribs[i]:>12.5f}"
          f"{100*row_contribs[i]/total_var:>11.1f}%")
print(f"  {'TOTAL':<32}{sum(row_contribs):>12.5f}{'100.0%':>12}")

# ── DECOMPOSITION 3: LMG / SHAPLEY (average over orderings) ──────────────────
# For each feature k and each ordering pi of the K features, the marginal
# contribution of k is: var(features in pi up to and including k) - var(up to just before k).
# LMG(k) = average over all K! orderings.
# Since our "variance explained by subset S" is the fixed-beta version:
#   Var_S = beta_S^T * Cov_SS * beta_S
# We enumerate all 2^K subsets and cache their variances, then compute LMG.
print("\n" + "=" * 80)
print("DECOMPOSITION 3 — LMG / SHAPLEY (average marginal contribution over K! orderings)")
print("=" * 80)

def var_of_subset(S):
    """Var(sum_{k in S} beta_k * feature_k) using the fitted betas."""
    if not S:
        return 0.0
    idx = list(S)
    return float(betas[idx] @ cov_matrix[np.ix_(idx, idx)] @ betas[idx])

# Cache variance of each subset
subset_var = {}
for r in range(K + 1):
    for combo in itertools.combinations(range(K), r):
        subset_var[frozenset(combo)] = var_of_subset(combo)

# LMG for each feature
lmg = np.zeros(K)
# For each feature k, average (var(S ∪ {k}) - var(S)) over all subsets S not containing k,
# weighted such that the average is over all K! orderings. The formula:
#   LMG(k) = (1/K!) * sum_{orderings} marginal_at_k
#          = sum_{S subset of features, k not in S}
#              |S|! * (K-|S|-1)! / K!   *   (var(S ∪ {k}) - var(S))
for k in range(K):
    for r in range(K):   # size of S (subsets not containing k)
        weight = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
        for combo in itertools.combinations([j for j in range(K) if j != k], r):
            S = frozenset(combo)
            Sk = S | {k}
            lmg[k] += weight * (subset_var[Sk] - subset_var[S])

print(f"  {'Feature':<32}{'LMG contrib':>13}{'% of total':>12}")
print("  " + "-" * 57)
for i in np.argsort(-lmg):
    print(f"  {labels[i]:<32}{lmg[i]:>13.5f}"
          f"{100*lmg[i]/total_var:>11.1f}%")
print(f"  {'TOTAL':<32}{lmg.sum():>13.5f}{'100.0%':>12}")

# ── PAIRWISE CROSS-TERMS: where does the shared variance live? ───────────────
print("\n" + "=" * 80)
print("PAIRWISE COVARIANCE CROSS-TERMS  (2 * beta_i * beta_j * Cov(i, j))")
print("=" * 80)
print("These represent SHARED variance between feature pairs. Positive = features")
print("reinforce each other; negative = features partially cancel.")
print()
cross_terms = []
for i in range(K):
    for j in range(i+1, K):
        term = 2 * betas[i] * betas[j] * cov_matrix[i, j]
        cross_terms.append((abs(term), term, i, j))
cross_terms.sort(reverse=True)
print(f"  {'Feature 1':<26}{'Feature 2':<26}{'cross-term':>13}{'% of total':>12}")
print("  " + "-" * 76)
for abs_term, term, i, j in cross_terms:
    sign = "+" if term >= 0 else "-"
    print(f"  {labels[i][:24]:<26}{labels[j][:24]:<26}"
          f"{term:>+13.5f}{100*term/total_var:>+11.1f}%")

sum_cross = sum(t for _, t, _, _ in cross_terms)
sum_diag  = float(np.sum(betas**2 * variances))
print(f"\n  Sum of diagonal terms (naive): {sum_diag:>+.5f} ({100*sum_diag/total_var:>+.1f}%)")
print(f"  Sum of cross-terms:            {sum_cross:>+.5f} ({100*sum_cross/total_var:>+.1f}%)")
print(f"  Total:                         {sum_diag + sum_cross:>+.5f}  (should match total_var)")

# ── SIDE-BY-SIDE COMPARISON ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 80)
naive_pct = 100 * naive_contribs / naive_sum
row_pct   = 100 * row_contribs   / total_var
lmg_pct   = 100 * lmg            / total_var
print(f"  {'Feature':<32}{'Naive %':>10}{'Full-cov %':>13}{'LMG %':>10}")
print("  " + "-" * 65)
order = np.argsort(-lmg_pct)
for i in order:
    print(f"  {labels[i]:<32}{naive_pct[i]:>9.1f}%{row_pct[i]:>12.1f}%{lmg_pct[i]:>9.1f}%")

# ── INTERPRETATION FOR DISTANCE SPECIFICALLY ─────────────────────────────────
dist_idx = 0  # first feature is distance
print("\n" + "=" * 80)
print("FOCUS: DISTANCE — is its 26% share 'real' or shared?")
print("=" * 80)
print(f"  Naive share (beta_dist^2 * Var / naive_sum):   {naive_pct[dist_idx]:>5.1f}%")
print(f"  Full-covariance share (beta * Cov beta):       {row_pct[dist_idx]:>5.1f}%")
print(f"  LMG (average marginal over orderings):         {lmg_pct[dist_idx]:>5.1f}%")

# Show distance's cross-terms with each other feature
print(f"\n  Distance's covariance cross-terms with other features:")
for j in range(1, K):
    term = 2 * betas[dist_idx] * betas[j] * cov_matrix[dist_idx, j]
    print(f"    with {labels[j]:<28}: {term:>+.5f}  ({100*term/total_var:>+5.1f}% of total)")

print("\n  If LMG ≈ Naive: distance genuinely carries independent signal.")
print("  If LMG << Naive: distance's naive share is inflated by shared variance.")
print("  If LMG > Naive: distance is a genuine suppressor helping OTHER features fit.")
