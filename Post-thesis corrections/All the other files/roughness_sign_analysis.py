"""
Roughness Sign Distribution Analysis
-------------------------------------
Checks whether rough_speedup_WTG_frac and rough_speedup_MM_frac
have opposite signs in the training data, and how that affects
different formula choices.

Run this script to get empirical evidence before choosing
a roughness formula for the model.
"""

import pandas as pd
import numpy as np

# ---- Config ----
INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

EXCLUDED_MASTS = [
    "2015WM018", "2021PA004", "2022PA008",
    "2022PA018",
    "2011WM011", "2014WM011",
    "2019HE001", "2019HE002", "2019HE003",
    "2022PA021",
    "2023PA085",
    "2024PA014",
    "2012WM006",
    "2024PA107",
]


def main():
    df = pd.read_excel(INPUT_PATH)

    # Filter to valid sectors only
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    # Apply exclusions (same as final model)
    df["mast_A"] = df["pair_id"].apply(lambda x: x.split("__")[0])
    df["mast_B"] = df["pair_id"].apply(lambda x: x.split("__")[1])
    mask = (~df["mast_A"].isin(EXCLUDED_MASTS)) & (~df["mast_B"].isin(EXCLUDED_MASTS))
    df = df[mask].copy()

    # Use windPRO detailed values where available (same logic as final model)
    if "rough_speedup_WTG_frac_new" in df.columns:
        upgrade_mask = df["rough_speedup_WTG_frac_new"].notna()
        df.loc[upgrade_mask, "rough_speedup_WTG_frac"] = df.loc[upgrade_mask, "rough_speedup_WTG_frac_new"]
    if "rough_speedup_MM_frac_new" in df.columns:
        upgrade_mask = df["rough_speedup_MM_frac_new"].notna()
        df.loc[upgrade_mask, "rough_speedup_MM_frac"] = df.loc[upgrade_mask, "rough_speedup_MM_frac_new"]

    rs_WTG = pd.to_numeric(df["rough_speedup_WTG_frac"], errors="coerce")
    rs_MM = pd.to_numeric(df["rough_speedup_MM_frac"], errors="coerce")

    valid = rs_WTG.notna() & rs_MM.notna()
    wtg = rs_WTG[valid].values
    mm = rs_MM[valid].values

    print("=" * 70)
    print("ROUGHNESS SPEEDUP SIGN ANALYSIS")
    print("=" * 70)
    print(f"Total sector-level rows (after exclusions): {len(df)}")
    print(f"Rows with valid rs_WTG and rs_MM: {valid.sum()}")
    print(f"Pairs in dataset: {df[valid]['pair_id'].nunique()}")

    # --- Basic statistics ---
    print("\n--- rs_WTG (rough_speedup_WTG_frac) ---")
    print(f"  Min:      {wtg.min():.6f}")
    print(f"  Max:      {wtg.max():.6f}")
    print(f"  Mean:     {wtg.mean():.6f}")
    print(f"  Std:      {wtg.std():.6f}")
    print(f"  Positive: {(wtg > 0).sum()} ({(wtg > 0).mean()*100:.1f}%)")
    print(f"  Negative: {(wtg < 0).sum()} ({(wtg < 0).mean()*100:.1f}%)")
    print(f"  Zero:     {(wtg == 0).sum()}")

    print("\n--- rs_MM (rough_speedup_MM_frac) ---")
    print(f"  Min:      {mm.min():.6f}")
    print(f"  Max:      {mm.max():.6f}")
    print(f"  Mean:     {mm.mean():.6f}")
    print(f"  Std:      {mm.std():.6f}")
    print(f"  Positive: {(mm > 0).sum()} ({(mm > 0).mean()*100:.1f}%)")
    print(f"  Negative: {(mm < 0).sum()} ({(mm < 0).mean()*100:.1f}%)")
    print(f"  Zero:     {(mm == 0).sum()}")

    # --- Opposite sign analysis ---
    opp = (wtg * mm) < 0
    print("\n" + "=" * 70)
    print("OPPOSITE SIGN ANALYSIS (WTG * MM < 0)")
    print("=" * 70)
    print(f"  Sectors with opposite sign: {opp.sum()} ({opp.mean()*100:.1f}%)")

    if opp.sum() > 0:
        print(f"\n  In those sectors:")
        print(f"    WTG range: [{wtg[opp].min():.5f}, {wtg[opp].max():.5f}]")
        print(f"    MM range:  [{mm[opp].min():.5f}, {mm[opp].max():.5f}]")

        # How many pairs are affected?
        valid_df = df[valid].copy()
        valid_df["opp_sign"] = opp
        opp_pairs = valid_df[valid_df["opp_sign"]]["pair_id"].unique()
        total_pairs = valid_df["pair_id"].nunique()
        print(f"\n  Pairs affected: {len(opp_pairs)} / {total_pairs}")

        # Per-pair breakdown
        print(f"\n  Per-pair breakdown (opposite-sign sectors / total sectors):")
        for pid in sorted(opp_pairs):
            sub = valid_df[valid_df["pair_id"] == pid]
            n_opp = sub["opp_sign"].sum()
            n_total = len(sub)
            loc = sub["location"].iloc[0] if "location" in sub.columns else ""
            print(f"    {pid:<35} {n_opp:>2}/{n_total:<2} sectors  ({loc})")

    # --- Same sign analysis ---
    same = (wtg * mm) > 0
    print(f"\n--- SAME SIGN BREAKDOWN ---")
    print(f"  Sectors with same sign: {same.sum()} ({same.mean()*100:.1f}%)")
    both_pos = (wtg > 0) & (mm > 0)
    both_neg = (wtg < 0) & (mm < 0)
    print(f"    Both positive (acceleration): {both_pos.sum()} ({both_pos.mean()*100:.1f}%)")
    print(f"    Both negative (deceleration): {both_neg.sum()} ({both_neg.mean()*100:.1f}%)")

    # --- Formula comparison (sector-level) ---
    print("\n" + "=" * 70)
    print("FORMULA COMPARISON (sector-level raw values before energy weighting)")
    print("=" * 70)

    formulas = {
        "A: |(WTG+MM)/2|  (current model)": np.abs((wtg + mm) / 2),
        "B: |WTG - MM|    (difference)": np.abs(wtg - mm),
        "C: |(WTG+MM)/2 * (WTG-MM)| (product)": np.abs((wtg + mm) / 2 * (wtg - mm)),
        "D: (|WTG|+|MM|)/2 (avg of absolutes)": (np.abs(wtg) + np.abs(mm)) / 2,
        "E: max(|WTG|, |MM|) (maximum)": np.maximum(np.abs(wtg), np.abs(mm)),
    }

    for name, vals in formulas.items():
        print(f"\n  {name}")
        print(f"    Mean: {vals.mean():.6f}  Std: {vals.std():.6f}")
        print(f"    Range: [{vals.min():.6f}, {vals.max():.6f}]")
        print(f"    Zeros (< 1e-8): {(vals < 1e-8).sum()}")
        # Check if ROUGH_SAT_SCALE=0.01 is appropriate
        sat_vals = 1.0 - np.exp(-vals / 0.01)
        print(f"    After saturation (scale=0.01): mean={sat_vals.mean():.4f}  "
              f"range=[{sat_vals.min():.4f}, {sat_vals.max():.4f}]")

    # --- Where formulas disagree most ---
    print("\n" + "=" * 70)
    print("CANCELLATION ANALYSIS: |(WTG+MM)/2| vs (|WTG|+|MM|)/2")
    print("=" * 70)
    current = np.abs((wtg + mm) / 2)
    avg_abs = (np.abs(wtg) + np.abs(mm)) / 2
    gap = avg_abs - current  # always >= 0 by triangle inequality

    print(f"\n  Mean gap (avg_abs - current): {gap.mean():.6f}")
    print(f"  Max gap:  {gap.max():.6f}")
    print(f"  Sectors where gap > 0.001: {(gap > 0.001).sum()} ({(gap > 0.001).mean()*100:.1f}%)")
    print(f"  Sectors where gap > 0.005: {(gap > 0.005).sum()} ({(gap > 0.005).mean()*100:.1f}%)")

    # --- Product formula edge case ---
    print("\n" + "=" * 70)
    print("PRODUCT FORMULA EDGE CASE: avg~0 but diff!=0")
    print("=" * 70)
    avg_near_zero = np.abs((wtg + mm) / 2) < 0.005
    diff_nonzero = np.abs(wtg - mm) > 0.005
    edge_case = avg_near_zero & diff_nonzero
    print(f"  Cases where |avg| < 0.005 but |diff| > 0.005: {edge_case.sum()} ({edge_case.mean()*100:.1f}%)")
    if edge_case.sum() > 0:
        print(f"  These sectors have large mismatch but product formula gives ~0")
        print(f"    WTG in edge cases: [{wtg[edge_case].min():.5f}, {wtg[edge_case].max():.5f}]")
        print(f"    MM in edge cases:  [{mm[edge_case].min():.5f}, {mm[edge_case].max():.5f}]")

    # --- Correlation with absolute error (sector-level) ---
    print("\n" + "=" * 70)
    print("CORRELATION WITH |SECTOR ERROR|  (quick sanity check)")
    print("=" * 70)

    # Compute sector error where possible
    if "Mean_windspeed_predicted" in df.columns and "Mean_windspeed_self" in df.columns:
        e_sector = (
            (df.loc[valid, "Mean_windspeed_predicted"] - df.loc[valid, "Mean_windspeed_self"]) /
            df.loc[valid, "Mean_windspeed_self"]
        ).values
        abs_e = np.abs(e_sector)

        print(f"\n  {'Formula':<45} {'Pearson r':>10} {'Spearman rho':>12}")
        print("  " + "-" * 70)
        for name, vals in formulas.items():
            from scipy.stats import spearmanr, pearsonr
            r_p, _ = pearsonr(vals, abs_e)
            r_s, _ = spearmanr(vals, abs_e)
            short_name = name.split("(")[0].strip()
            print(f"  {short_name:<45} {r_p:>10.3f} {r_s:>12.3f}")
    else:
        print("  (skipped — windspeed columns not found)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Opposite-sign sectors:  {opp.sum()} / {len(wtg)} ({opp.mean()*100:.1f}%)
  Same-sign sectors:      {same.sum()} / {len(wtg)} ({same.mean()*100:.1f}%)
  One or both = 0:        {((wtg == 0) | (mm == 0)).sum()}

  If opposite-sign is common (>10%), the current formula |(WTG+MM)/2|
  underestimates roughness complexity for those sectors.

  The avg-of-absolutes formula (|WTG|+|MM|)/2 never has sign cancellation
  and matches what was used in the original exploration (Script_v3.py).
    """)


if __name__ == "__main__":
    main()
