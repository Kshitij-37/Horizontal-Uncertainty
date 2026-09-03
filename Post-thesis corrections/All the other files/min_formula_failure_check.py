"""
Diagnostic: where does the min(magnitude, mismatch) roughness formula break down?

The concern: when rs_WTG and rs_MM have opposite signs with similar magnitudes,
A = |(rs_W + rs_M)/2| collapses toward zero due to sign cancellation, while
B = |rs_W - rs_M| stays large. min(A, B) picks A -> feature says "no roughness
uncertainty" -> under-predicts.

This script identifies:
  1. Sector-level: how often does opposite-sign happen? What fraction of sectors?
  2. Pair-level: for which pairs does A drop far below B? (cancellation-driven)
  3. For those pairs: what does the model over/under-predict vs actual |e|?
  4. Are the "risky" pairs kept in the model or excluded?

Output: CSV of per-pair diagnostics + console summary.
"""
import os
import numpy as np
import pandas as pd

INPUT_XLSX = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "Results", "min_formula_failure_check.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

FOREST_MASTS = {"2011WM011", "2014WM011", "2012WM006"}
OTHER_EXCLUDED = {"2015WM018","2021PA004","2022PA008","2022PA018","2019HE001",
                   "2019HE002","2019HE003","2022PA021","2023PA085","2024PA014","2024PA107"}

# ── load and prep ────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_XLSX)
for old, new in [("rough_speedup_WTG_frac", "rough_speedup_WTG_frac_new"),
                 ("rough_speedup_MM_frac",  "rough_speedup_MM_frac_new")]:
    if new in df.columns and old in df.columns:
        mask = df[new].notna()
        df.loc[mask, old] = df.loc[mask, new]

def classify(mm, wtg):
    if mm in FOREST_MASTS or wtg in FOREST_MASTS:
        return "FOREST"
    if mm in OTHER_EXCLUDED or wtg in OTHER_EXCLUDED:
        return "OTHER_EXCL"
    return "KEPT"

# ── sector-level sign analysis ───────────────────────────────────────────────
rs_W = pd.to_numeric(df["rough_speedup_WTG_frac"], errors="coerce")
rs_M = pd.to_numeric(df["rough_speedup_MM_frac"],  errors="coerce")
valid = ~(np.isnan(rs_W) | np.isnan(rs_M))
df_v = df[valid].copy()
rs_W_v = rs_W[valid]; rs_M_v = rs_M[valid]

opp_sign = (rs_W_v * rs_M_v < 0)
zero_either = (rs_W_v == 0) | (rs_M_v == 0)
same_sign = (rs_W_v * rs_M_v > 0)

print("=" * 90)
print("SECTOR-LEVEL SIGN ANALYSIS  (on the corrected input data)")
print("=" * 90)
print(f"  Total valid sector rows: {valid.sum()}")
print(f"  Opposite sign (rs_W * rs_M < 0): {opp_sign.sum()} ({100*opp_sign.mean():.1f}%)")
print(f"  Same sign (rs_W * rs_M > 0):     {same_sign.sum()} ({100*same_sign.mean():.1f}%)")
print(f"  One or both zero:                 {zero_either.sum()} ({100*zero_either.mean():.1f}%)")

# ── per-pair aggregation with min-formula diagnostics ─────────────────────────
rows = []
for pair_id, grp in df_v.groupby("pair_id"):
    mm  = pair_id.split("__")[1]
    wtg = pair_id.split("__")[0]

    w = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
    if w.sum() == 0:
        continue
    w = w / w.sum()

    rs_W_p = grp["rough_speedup_WTG_frac"].values
    rs_M_p = grp["rough_speedup_MM_frac"].values

    # Per-sector A and B contributions
    sec_A = np.abs((rs_W_p + rs_M_p) / 2)
    sec_B = np.abs(rs_W_p - rs_M_p)

    # Pair-level (freq-weighted)
    A_pair = float(np.sum(w * sec_A))
    B_pair = float(np.sum(w * sec_B))

    # How many opposite-sign sectors in this pair, and their energy weight?
    opp_mask = (rs_W_p * rs_M_p < 0)
    opp_n = int(opp_mask.sum())
    opp_weight = float(np.sum(w[opp_mask])) if opp_n > 0 else 0.0

    # Cancellation index: (B - A) / max(B, eps) — how much A undershoots B
    cancel_idx = (B_pair - A_pair) / max(B_pair, 1e-9)

    # Observed |e|
    ws_p = pd.to_numeric(grp["Mean_windspeed_predicted"], errors="coerce").values
    ws_s = pd.to_numeric(grp["Mean_windspeed_self"], errors="coerce").values
    w_pred = pd.to_numeric(grp["Sample_count_pred"], errors="coerce").fillna(0).values
    w_self = pd.to_numeric(grp["Sample_count_self"], errors="coerce").fillna(0).values
    if w_pred.sum() > 0 and w_self.sum() > 0:
        WS_p = np.sum((w_pred/w_pred.sum()) * ws_p)
        WS_s = np.sum((w_self/w_self.sum()) * ws_s)
        e_ovr = (WS_p - WS_s) / WS_s
        abs_e = float(abs(e_ovr))
    else:
        abs_e = np.nan

    rows.append({
        "pair_id": pair_id,
        "group": classify(mm, wtg),
        "A_magnitude": A_pair,
        "B_mismatch":  B_pair,
        "min_AB":      min(A_pair, B_pair),
        "which_wins":  "A" if A_pair < B_pair else "B",
        "cancellation_idx": cancel_idx,       # 0 = A and B agree; 1 = A collapsed to 0 vs large B
        "opp_sign_sectors": opp_n,
        "opp_sign_weight": opp_weight,
        "abs_e_overall": abs_e,
    })

out = pd.DataFrame(rows).sort_values("cancellation_idx", ascending=False).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

# ── console: pairs where min-formula is most at risk of under-predicting ─────
print("\n" + "=" * 108)
print("TOP 20 PAIRS BY 'CANCELLATION INDEX'  (how much A undershoots B due to sign cancellation)")
print("=" * 108)
print(f"  {'pair_id':<32}{'group':<11}{'A':>9}{'B':>9}{'min':>9}{'cancel':>8}{'opp_sec':>8}"
      f"{'opp_wt':>8}{'|e|':>8}")
for _, r in out.head(20).iterrows():
    eA = f"{r['abs_e_overall']:.4f}" if pd.notna(r['abs_e_overall']) else "  N/A "
    print(f"  {r['pair_id']:<32}{r['group']:<11}"
          f"{r['A_magnitude']:>9.5f}{r['B_mismatch']:>9.5f}{r['min_AB']:>9.5f}"
          f"{r['cancellation_idx']:>8.3f}{r['opp_sign_sectors']:>8d}"
          f"{r['opp_sign_weight']:>8.3f}{eA:>8}")

# ── which of these top-cancellation pairs are in the model? ───────────────────
top10 = out.head(10)
kept_top = (top10["group"] == "KEPT").sum()
print(f"\nOf the 10 most cancellation-affected pairs: {kept_top} are KEPT in the model")

# ── group-level summary ──────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("CANCELLATION INDEX BY GROUP")
print("=" * 108)
print(f"  {'group':<12}{'n':>4}{'cancel_med':>12}{'cancel_max':>12}"
      f"{'opp_sec_med':>13}{'|e|_med':>10}{'|e|_max':>10}")
for grp_name in ["KEPT", "FOREST", "OTHER_EXCL"]:
    sub = out[out["group"] == grp_name]
    if len(sub) == 0:
        continue
    print(f"  {grp_name:<12}{len(sub):>4}{sub['cancellation_idx'].median():>12.3f}"
          f"{sub['cancellation_idx'].max():>12.3f}"
          f"{sub['opp_sign_sectors'].median():>13.1f}"
          f"{sub['abs_e_overall'].median():>10.4f}{sub['abs_e_overall'].max():>10.4f}")

# ── which formula "wins" the min, across the dataset ─────────────────────────
kept_only = out[out["group"] == "KEPT"]
print("\n" + "=" * 108)
print("WHICH FORMULA WINS THE MIN?  (KEPT pairs only)")
print("=" * 108)
print(f"  A wins (A < B): {(kept_only['which_wins'] == 'A').sum()} of {len(kept_only)} kept pairs")
print(f"  B wins (B < A): {(kept_only['which_wins'] == 'B').sum()} of {len(kept_only)} kept pairs")
print("\n  When A wins -> model uses magnitude (potentially cancelled). This is the risky case.")
print("  When B wins -> model uses mismatch (proper dissimilarity). No cancellation concern.")

# ── specific check: any pair where A is essentially zero but observed |e| is large? ────
print("\n" + "=" * 108)
print("THE FAILURE CASE:  A ~ 0 (heavy cancellation) but observed |e| is large")
print("=" * 108)
risky = out[(out["A_magnitude"] < 0.001) & (out["abs_e_overall"] > 0.05)]
if len(risky) == 0:
    print("  No pairs found with A < 0.001 AND |e| > 0.05.  Good sign — the failure mode")
    print("  doesn't actually manifest in our data at scale.")
else:
    print(f"  Found {len(risky)} such pairs — the min formula would under-predict these:")
    for _, r in risky.iterrows():
        print(f"    {r['pair_id']:<32} A={r['A_magnitude']:.5f}  B={r['B_mismatch']:.5f}  "
              f"|e|={r['abs_e_overall']:.4f}  group={r['group']}")

print(f"\nSaved: {OUT_CSV}")
