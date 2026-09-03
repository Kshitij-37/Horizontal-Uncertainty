"""
Per-pair comparison of exp-A__forest-out vs exp-M__forest-in.

Loads the LOO prediction CSVs from both configs, joins by pair_id, computes
per-pair prediction differences, and identifies:
  - which pairs the two models disagree most about
  - which model's prediction is closer to actual |e_overall| per pair
  - forest pairs (only in exp-M) shown separately
  - overall "who wins how often" summary
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_A = os.path.join(HERE, "Results", "exp-A__forest-out_loo.csv")
CSV_M = os.path.join(HERE, "Results", "exp-M__forest-in_loo.csv")
OUT_CSV = os.path.join(HERE, "Results", "compare_expA_expM_per_pair.csv")

# Same forest identifiers we've been using
FOREST_MASTS = {"2011WM011", "2014WM011", "2012WM006"}
OTHER_EXCLUDED = {"2015WM018","2021PA004","2022PA008","2022PA018","2019HE001",
                   "2019HE002","2019HE003","2022PA021","2023PA085","2024PA014","2024PA107"}

def classify(pair_id):
    mm  = pair_id.split("__")[1]
    wtg = pair_id.split("__")[0]
    if mm in FOREST_MASTS or wtg in FOREST_MASTS:
        return "FOREST"
    if mm in OTHER_EXCLUDED or wtg in OTHER_EXCLUDED:
        return "OTHER_EXCL"
    return "KEPT"


# ── load and join ─────────────────────────────────────────────────────────────
dfA = pd.read_csv(CSV_A).rename(columns={"predicted_sigma": "pred_A"})
dfM = pd.read_csv(CSV_M).rename(columns={"predicted_sigma": "pred_M"})

# actual_abs_error should agree between the two files for shared pairs; use A's version
merged = dfM.merge(dfA[["pair_id", "pred_A"]], on="pair_id", how="outer")
merged["group"] = merged["pair_id"].apply(classify)
merged["abs_e"] = merged["actual_abs_error"]

# per-pair prediction diff and per-model errors
merged["diff_M_minus_A"] = merged["pred_M"] - merged["pred_A"]
merged["err_A"] = (merged["pred_A"] - merged["abs_e"]).abs()
merged["err_M"] = (merged["pred_M"] - merged["abs_e"]).abs()
merged["closer"] = np.where(merged["err_A"] < merged["err_M"], "A",
                    np.where(merged["err_M"] < merged["err_A"], "M", "tie"))

merged = merged.sort_values("diff_M_minus_A", key=lambda s: s.abs(), ascending=False)
merged.to_csv(OUT_CSV, index=False)

# ── console: top disagreements ─────────────────────────────────────────────────
print("=" * 108)
print("TOP 15 PAIRS BY |exp-M - exp-A| PREDICTION DIFFERENCE")
print("=" * 108)
print(f"  {'pair_id':<32}{'group':<11}{'|e|':>9}{'pred_A':>9}{'pred_M':>9}{'M-A':>10}{'err_A':>9}{'err_M':>9}  closer")
for _, r in merged.head(15).iterrows():
    pA = f"{r['pred_A']:.4f}" if pd.notna(r['pred_A']) else "  N/A "
    dA = f"{r['diff_M_minus_A']:+.4f}" if pd.notna(r['pred_A']) else "  N/A  "
    eA = f"{r['err_A']:.4f}" if pd.notna(r['err_A']) else "  N/A "
    closer = r["closer"] if pd.notna(r["closer"]) else "onlyM"
    print(f"  {r['pair_id']:<32}{r['group']:<11}{r['abs_e']:>9.4f}{pA:>9}"
          f"{r['pred_M']:>9.4f}{dA:>10}{eA:>9}{r['err_M']:>9.4f}  {closer}")

# ── forest pairs (present only in exp-M) ──────────────────────────────────────
forest_only = merged[merged["group"] == "FOREST"]
if len(forest_only) > 0:
    print("\n" + "=" * 108)
    print(f"FOREST PAIRS (n={len(forest_only)}; only exp-M forest-in has predictions for these)")
    print("=" * 108)
    print(f"  {'pair_id':<32}{'|e|':>9}{'pred_M':>9}{'err_M':>9}")
    for _, r in forest_only.iterrows():
        print(f"  {r['pair_id']:<32}{r['abs_e']:>9.4f}{r['pred_M']:>9.4f}{r['err_M']:>9.4f}")

# ── who wins by group ─────────────────────────────────────────────────────────
shared = merged.dropna(subset=["pred_A", "pred_M"]).copy()   # pairs in both
print("\n" + "=" * 108)
print(f"WHO IS CLOSER TO |actual e|, BY GROUP  (shared pairs only, n={len(shared)})")
print("=" * 108)
print(f"  {'group':<12}{'n':>4}{'A_wins':>9}{'M_wins':>9}{'ties':>7}"
      f"{'mean|errA|':>13}{'mean|errM|':>13}")
for grp_name in ["KEPT", "OTHER_EXCL"]:
    sub = shared[shared["group"] == grp_name]
    if len(sub) == 0:
        continue
    aw = (sub["closer"] == "A").sum()
    mw = (sub["closer"] == "M").sum()
    tie = (sub["closer"] == "tie").sum()
    print(f"  {grp_name:<12}{len(sub):>4}{aw:>9}{mw:>9}{tie:>7}"
          f"{sub['err_A'].mean():>13.4f}{sub['err_M'].mean():>13.4f}")

# ── overall summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("SUMMARY STATS")
print("=" * 108)
print(f"  shared pairs (in both):    {len(shared)}")
print(f"    A wins:                  {(shared['closer'] == 'A').sum()}")
print(f"    M wins:                  {(shared['closer'] == 'M').sum()}")
print(f"    ties:                    {(shared['closer'] == 'tie').sum()}")
print(f"    mean |err_A|:            {shared['err_A'].mean():.4f}")
print(f"    mean |err_M|:            {shared['err_M'].mean():.4f}")
print(f"    median |M-A diff|:       {shared['diff_M_minus_A'].abs().median():.4f}")
print(f"    max |M-A diff|:          {shared['diff_M_minus_A'].abs().max():.4f}")
print(f"  extra pairs only in exp-M: {(merged['pred_A'].isna()).sum()}  (forest)")
print(f"\nSaved: {OUT_CSV}")
