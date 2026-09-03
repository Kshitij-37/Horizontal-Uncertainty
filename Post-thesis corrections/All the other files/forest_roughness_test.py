"""
Forest roughness test — does a mismatch feature explain the excluded forest pairs?

Part 1 (this file, no pymc): un-exclude the Hultema + Malarberget forest masts and, for
each forest pair, compare the magnitude feature |(rs_WTG+rs_MM)/2| vs the mismatch feature
|rs_WTG - rs_MM| against the ACTUAL error. Decides whether the forest pairs are a
feature-blind-spot (low error but high magnitude -> mismatch fixes it) or genuinely hard sites.

Part 2 (LOO, pymc): run `python forest_roughness_test.py loo` in pymc-env to fit formulas
A/B/E/Q on the forest-inclusive set via the existing roughness_formula_comparison harness.
"""
import os
import sys
import numpy as np
import pandas as pd

INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# Run-button mode (the ▶ button passes no arguments, so it reads this):
#   "loo"        -> Part 2: A/B/E/Q LOO comparison  (pymc, ~15-20 min)
#   "diagnostic" -> Part 1: fast feature diagnostic (no pymc)
MODE = "loo"

# Excluded pairs the MAGNITUDE roughness feature over-flags (high roughness, low mismatch, low error):
FOREST_MASTS = [
    "2022PA018",               # Kayislar   (predicted 27% vs actual ~5%)
    "2011WM011", "2014WM011",  # Hultema
    "2012WM006",               # Malarberget
]
# Full production exclusions MINUS the over-flagged masts (i.e. keep those pairs IN for the test):
EXCLUDED_FOR_TEST = [
    "2015WM018", "2021PA004", "2022PA008",
    "2019HE001", "2019HE002", "2019HE003",
    "2022PA021", "2023PA085", "2024PA014", "2024PA107",
]


def build_pairs(df, excluded):
    df = df.drop_duplicates().copy()
    df["mast_A"] = df["pair_id"].str.split("__").str[0]
    df["mast_B"] = df["pair_id"].str.split("__").str[1]
    df = df[(~df["mast_A"].isin(excluded)) & (~df["mast_B"].isin(excluded))].copy()

    for old, new in {
        "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac": "rough_speedup_MM_frac_new",
    }.items():
        if new in df.columns:
            m = df[new].notna(); df.loc[m, old] = df.loc[m, new]

    rows = []
    for pid, g in df.groupby("pair_id"):
        wp = pd.to_numeric(g["Sample_count_pred"], errors="coerce").fillna(0).values.astype(float)
        ws = pd.to_numeric(g["Sample_count_self"], errors="coerce").fillna(0).values.astype(float)
        if wp.sum() == 0 or ws.sum() == 0:
            wp = np.ones(len(g)); ws = np.ones(len(g))
        wp = wp / wp.sum(); ws = ws / ws.sum()
        WSs = float(np.sum(ws * g["Mean_windspeed_self"].values))
        e = (float(np.sum(wp * g["Mean_windspeed_predicted"].values)) - WSs) / WSs

        wf = pd.to_numeric(g["freq_MM"], errors="coerce").fillna(0).values
        wf = wf / wf.sum() if wf.sum() > 0 else wp

        rW = pd.to_numeric(g["rough_speedup_WTG_frac"], errors="coerce").values
        rM = pd.to_numeric(g["rough_speedup_MM_frac"], errors="coerce").values
        valid = ~(np.isnan(rW) | np.isnan(rM))
        if valid.sum() == 0:
            continue
        wfv = wf[valid] / wf[valid].sum()
        rW, rM = rW[valid], rM[valid]

        featA = float(np.sum(wfv * np.abs((rW + rM) / 2)))       # magnitude of average (current)
        featB = float(np.sum(wfv * np.abs(rW - rM)))             # mismatch
        wm_rW = float(np.sum(wfv * rW))                          # signed roughness at WTG site
        wm_rM = float(np.sum(wfv * rM))                          # signed roughness at MM site

        rows.append({
            "pair_id": pid,
            "mast_A": g["mast_A"].iloc[0], "mast_B": g["mast_B"].iloc[0],
            "wm_rs_WTG": wm_rW, "wm_rs_MM": wm_rM,
            "featA_magnitude": featA, "featB_mismatch": featB,
            "abs_e_pct": abs(e) * 100.0,
            "distance_m": float(g["distance_m"].iloc[0]),
            "is_forest": (g["mast_A"].iloc[0] in FOREST_MASTS) or (g["mast_B"].iloc[0] in FOREST_MASTS),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(INPUT_PATH)
    pf = build_pairs(df, EXCLUDED_FOR_TEST)
    forest = pf[pf["is_forest"]].copy()
    normal = pf[~pf["is_forest"]].copy()
    print(f"Total pairs (forest-inclusive): {len(pf)}   forest pairs: {len(forest)}   other: {len(normal)}\n")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("=== FOREST PAIRS ===")
    show = forest[["pair_id", "wm_rs_WTG", "wm_rs_MM", "featA_magnitude", "featB_mismatch",
                   "abs_e_pct", "distance_m"]].copy()
    for c in ["wm_rs_WTG", "wm_rs_MM", "featA_magnitude", "featB_mismatch"]:
        show[c] = show[c].round(4)
    show["abs_e_pct"] = show["abs_e_pct"].round(2); show["distance_m"] = show["distance_m"].round(0)
    print(show.to_string(index=False))

    print("\n=== MEANS: forest vs other ===")
    print(f"  {'group':<8}{'featA(mag)':>12}{'featB(mism)':>13}{'mismatch/mag':>14}{'mean|e|%':>10}")
    for name, s in [("forest", forest), ("other", normal)]:
        ratio = (s["featB_mismatch"] / s["featA_magnitude"].replace(0, np.nan)).mean()
        print(f"  {name:<8}{s['featA_magnitude'].mean():>12.4f}{s['featB_mismatch'].mean():>13.4f}"
              f"{ratio:>14.2f}{s['abs_e_pct'].mean():>10.2f}")

    print("\n=== Which feature tracks actual |error| BETTER on the forest-inclusive set? ===")
    for lab, s in [("ALL (forest-incl)", pf), ("forest pairs only", forest)]:
        if len(s) < 3:
            print(f"  {lab:<20}: n={len(s)} too few"); continue
        pa = s["abs_e_pct"].corr(s["featA_magnitude"]); ra = s["abs_e_pct"].corr(s["featA_magnitude"], method="spearman")
        pb = s["abs_e_pct"].corr(s["featB_mismatch"]);  rb = s["abs_e_pct"].corr(s["featB_mismatch"], method="spearman")
        print(f"  {lab:<20}: magnitude r={pa:+.3f}/rho={ra:+.3f}   mismatch r={pb:+.3f}/rho={rb:+.3f}")

    print("\nRead: if forest pairs show HIGH featA (magnitude) + LOW featB (mismatch) + LOW |e|,")
    print("the magnitude feature over-flags them and mismatch would fix it (critique validated).")
    print("If forest pairs have HIGH |e| that neither feature tracks, they are just hard sites.")


def run_loo():
    """Part 2: forest-inclusive LOO of formulas A/B/E/Q. Needs pymc (run in pymc-env)."""
    import importlib.machinery, importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    loader = importlib.machinery.SourceFileLoader("rfc", os.path.join(here, "roughness_formula_comparison.py"))
    spec = importlib.util.spec_from_file_location("rfc", os.path.join(here, "roughness_formula_comparison.py"), loader=loader)
    rfc = importlib.util.module_from_spec(spec); loader.exec_module(rfc)
    # keep the forest masts in by overriding the exclusion list the harness reads
    rfc.pmod.EXCLUDED_MASTS = EXCLUDED_FOR_TEST
    sector_df = rfc.load_sector_data()
    print(f"Forest-inclusive sector rows: {len(sector_df)}  pairs: {sector_df['pair_id'].nunique()}")
    for key, desc in [("A", "magnitude |(W+M)/2|"), ("B", "mismatch |W-M|"),
                      ("E", "both: avg_abs + diff"), ("Q", "gated magnitude (500)")]:
        pair_df, fc = rfc.build_pair_data_with_formula(sector_df, key)
        _, r, rho, bias = rfc.run_loo_cv_custom(pair_df, fc)
        print(f"  {key} ({desc:<24}) n={len(pair_df):2d}  Pearson={r:.4f}  Spearman={rho:.4f}  bias={bias:+.4f}")


if __name__ == "__main__":
    _mode = sys.argv[1].lower() if len(sys.argv) > 1 else MODE
    if _mode == "loo":
        run_loo()
    else:
        main()
