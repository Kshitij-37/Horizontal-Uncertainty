"""
For each Herzhausen KEPT pair, break down the predicted sigma into feature
contributions under exp-A and exp-M. Shows how much of the prediction comes
from roughness vs the complexity/geometry features (distance, turning, speedup, dz).

log(sigma) = log_sigma0 + sum_k [ beta_k * feature_k / std_k ]

So each feature's "contribution to log(sigma)" is beta_k * feature_k / std_k.
We convert to a multiplicative factor: exp(contribution) is how much that feature
multiplies the baseline exp(log_sigma0).
"""
import os, json, importlib.machinery, importlib.util
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RFC = os.path.normpath(os.path.join(_HERE, "roughness_formula_comparison.py"))
_loader = importlib.machinery.SourceFileLoader("rfc", _RFC)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _RFC, loader=_loader))
_loader.exec_module(rfc)
pmod = rfc.pmod

EXP_A_JSON = os.path.join(_HERE, "sigma_only_refit", "Results", "exp-A__forest-out.json")
EXP_M_JSON = os.path.join(_HERE, "sigma_only_refit", "Results", "exp-M__forest-in.json")

HERZHAUSEN_MASTS = {"2019PA023", "2019PA024", "2020PA011"}

def is_herzhausen(pair_id):
    parts = pair_id.split("__")
    return parts[0] in HERZHAUSEN_MASTS and parts[1] in HERZHAUSEN_MASTS


def load_pair_data():
    """Load pair data with formulas A and M using the production exclusion list
    (Hultema and Malarberget stay excluded; matches exp-A forest-out training set)."""
    sector_df = rfc.load_sector_data()
    pair_A, fc = rfc.build_pair_data_with_formula(sector_df, "A")
    pair_B, _  = rfc.build_pair_data_with_formula(sector_df, "B")
    # Build pair_M (same as in refit_sigma_only.py)
    mag_by = pair_A.set_index("pair_id")["wm_abs_roughness"]
    mm_by  = pair_B.set_index("pair_id")["wm_abs_roughness"]
    combined = pd.concat([mag_by, mm_by], axis=1).min(axis=1)
    pair_M = pair_A.copy()
    pair_M["wm_abs_roughness"] = pair_M["pair_id"].map(combined).values
    pair_M["roughness_sat"] = 1.0 - np.exp(-pair_M["wm_abs_roughness"].values / rfc.ROUGH_SAT_SCALE)
    return pair_A, pair_M, fc, sector_df


def compute_e_overall(sector_df, pair_id):
    """Frequency-weighted signed overall error for one pair, matching production."""
    grp = sector_df[sector_df["pair_id"] == pair_id]
    w_p = pd.to_numeric(grp["Sample_count_pred"], errors="coerce").fillna(0).values
    w_s = pd.to_numeric(grp["Sample_count_self"], errors="coerce").fillna(0).values
    ws_p = pd.to_numeric(grp["Mean_windspeed_predicted"], errors="coerce").values
    ws_s = pd.to_numeric(grp["Mean_windspeed_self"],      errors="coerce").values
    if w_p.sum() == 0 or w_s.sum() == 0:
        return np.nan
    WS_p = float(np.sum((w_p/w_p.sum()) * ws_p))
    WS_s = float(np.sum((w_s/w_s.sum()) * ws_s))
    return (WS_p - WS_s) / WS_s


def decompose(row, coeffs, stds, feats):
    """Return per-feature contribution to log(sigma) and the total sigma."""
    log_sigma0 = coeffs["log_sigma0"]
    beta_map = coeffs.get("betas", coeffs.get("gammas"))
    contribs = {}
    log_sig = log_sigma0
    for f in feats:
        c = beta_map[f] * row[f] / stds[f]
        contribs[f] = c
        log_sig += c
    return contribs, float(np.exp(log_sig))


# ── load fitted models ────────────────────────────────────────────────────────
with open(EXP_A_JSON) as f: A_meta = json.load(f)
with open(EXP_M_JSON) as f: M_meta = json.load(f)

feats = A_meta["feature_order"]
A_coeffs = {"log_sigma0": A_meta["coeffs"]["log_sigma0"], "betas": A_meta["coeffs"]["betas"]}
A_stds   = A_meta["feature_stds"]
M_coeffs = {"log_sigma0": M_meta["coeffs"]["log_sigma0"], "betas": M_meta["coeffs"]["betas"]}
M_stds   = M_meta["feature_stds"]

# ── load pair data ───────────────────────────────────────────────────────────
pair_A, pair_M, fc, sector_df = load_pair_data()

# ── restrict to Herzhausen KEPT pairs ─────────────────────────────────────────
herz_pairs = [p for p in pair_A["pair_id"].unique() if is_herzhausen(p)]

print("=" * 108)
print("HERZHAUSEN PAIRS — PREDICTED sigma DECOMPOSITION UNDER exp-A vs exp-M")
print("=" * 108)

for pid in sorted(herz_pairs):
    row_A = pair_A[pair_A["pair_id"] == pid].iloc[0]
    row_M = pair_M[pair_M["pair_id"] == pid].iloc[0]
    e_actual = compute_e_overall(sector_df, pid)
    abs_e = abs(e_actual)

    contribs_A, sigma_A = decompose(row_A, A_coeffs, A_stds, feats)
    contribs_M, sigma_M = decompose(row_M, M_coeffs, M_stds, feats)

    print(f"\n--- {pid} ---")
    print(f"  Actual |e| = {abs_e:.4f}  ({100*abs_e:.2f}%)")
    print(f"  Predicted sigma  exp-A: {sigma_A:.4f} ({100*sigma_A:.2f}%)   "
          f"exp-M: {sigma_M:.4f} ({100*sigma_M:.2f}%)")
    print()
    print(f"  {'feature':<26}{'exp-A raw':>13}{'exp-A contrib':>16}"
          f"{'exp-M raw':>13}{'exp-M contrib':>16}")
    print(f"  {'log_sigma0':<26}{'-':>13}{A_coeffs['log_sigma0']:>+16.4f}"
          f"{'-':>13}{M_coeffs['log_sigma0']:>+16.4f}")
    for f in feats:
        rawA = float(row_A[f])
        rawM = float(row_M[f])
        cA = contribs_A[f]
        cM = contribs_M[f]
        print(f"  {f:<26}{rawA:>13.5f}{cA:>+16.4f}"
              f"{rawM:>13.5f}{cM:>+16.4f}")
    total_c_A = sum(contribs_A.values())
    total_c_M = sum(contribs_M.values())
    print(f"  {'sum of contributions':<26}{'':<13}{total_c_A:>+16.4f}"
          f"{'':<13}{total_c_M:>+16.4f}")
    print(f"  {'log(sigma) total':<26}{'':<13}"
          f"{A_coeffs['log_sigma0']+total_c_A:>+16.4f}"
          f"{'':<13}{M_coeffs['log_sigma0']+total_c_M:>+16.4f}")
    print(f"  {'sigma = exp(above)':<26}{'':<13}{sigma_A:>16.5f}"
          f"{'':<13}{sigma_M:>16.5f}")

print("\n" + "=" * 108)
print("QUICK RANKING — % of total (positive) contribution attributable to each feature")
print("=" * 108)
print("Note: negative contributions can occur; we take fraction of |sum of positive contributions|.")
for pid in sorted(herz_pairs):
    row_A = pair_A[pair_A["pair_id"] == pid].iloc[0]
    row_M = pair_M[pair_M["pair_id"] == pid].iloc[0]
    contribs_A, _ = decompose(row_A, A_coeffs, A_stds, feats)
    contribs_M, _ = decompose(row_M, M_coeffs, M_stds, feats)
    total_pos_A = sum(max(c, 0) for c in contribs_A.values()) or 1.0
    total_pos_M = sum(max(c, 0) for c in contribs_M.values()) or 1.0
    print(f"\n  {pid}")
    print(f"    {'feature':<26}{'% exp-A':>10}{'% exp-M':>10}")
    for f in feats:
        pctA = 100 * max(contribs_A[f], 0) / total_pos_A
        pctM = 100 * max(contribs_M[f], 0) / total_pos_M
        print(f"    {f:<26}{pctA:>9.1f}%{pctM:>9.1f}%")
