"""
Distance feature investigation
------------------------------
Tests the hypothesis that dist_sat = 1 - exp(-distance_m / distance_A) is INVERTED
relative to terrain difficulty, because of mast-siting practice:

  flat terrain    -> masts far apart (large distance_m), large distance_A  -> ratio d/dA large -> dist_sat HIGH
  complex terrain -> masts close     (small distance_m), small distance_A  -> ratio d/dA small -> dist_sat LOW

If true, dist_sat would be large where error is small (flat) and small where error is
large (complex) -> weak/inverted correlation with error.

Also tests the user's claim that distance explains RESIDUAL variance after the other
four features, via partial correlation.

pandas/numpy only.
"""
import numpy as np
import pandas as pd
import os

INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
OUT_DIR = os.path.dirname(__file__)

DZ_SAT_SCALE = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE = 3.0
PROXIMITY_GATE_SCALE = 500.0
EXCLUDED_MASTS = [
    "2015WM018", "2021PA004", "2022PA008", "2022PA018",
    "2011WM011", "2014WM011", "2019HE001", "2019HE002", "2019HE003",
    "2022PA021", "2023PA085", "2024PA014", "2012WM006", "2024PA107",
]


def build_pairs(df):
    df = df.drop_duplicates().copy()
    df["mast_A"] = df["pair_id"].str.split("__").str[0]
    df["mast_B"] = df["pair_id"].str.split("__").str[1]
    df = df[(~df["mast_A"].isin(EXCLUDED_MASTS)) & (~df["mast_B"].isin(EXCLUDED_MASTS))].copy()

    for old, new in {
        "d_turning_deg": "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor": "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac": "rough_speedup_MM_frac_new",
    }.items():
        if new in df.columns:
            m = df[new].notna(); df.loc[m, old] = df.loc[m, new]

    m = (df["overall_speedup_WTG_factor"] > 0) & (df["overall_speedup_MM_factor"] > 0)
    df["abs_log_speedup_ratio"] = np.nan
    df.loc[m, "abs_log_speedup_ratio"] = np.abs(
        np.log(df.loc[m, "overall_speedup_WTG_factor"] / df.loc[m, "overall_speedup_MM_factor"]))

    rows = []
    for pid, g in df.groupby("pair_id"):
        wp = pd.to_numeric(g["Sample_count_pred"], errors="coerce").fillna(0).values.astype(float)
        ws = pd.to_numeric(g["Sample_count_self"], errors="coerce").fillna(0).values.astype(float)
        if wp.sum() == 0 or ws.sum() == 0:
            wp = np.ones(len(g)); ws = np.ones(len(g))
        wp = wp / wp.sum(); ws = ws / ws.sum()
        e = (float(np.sum(wp * g["Mean_windspeed_predicted"].values)) -
             float(np.sum(ws * g["Mean_windspeed_self"].values))) / float(np.sum(ws * g["Mean_windspeed_self"].values))

        dm = g["distance_m"].iloc[0]; dA = g["distance_A"].iloc[0]
        if pd.isna(dA) or dA <= 0:
            continue
        dist_sat = 1 - np.exp(-dm / dA)

        wf = pd.to_numeric(g["freq_MM"], errors="coerce").fillna(0).values
        wf = wf / wf.sum() if wf.sum() > 0 else wp
        turning_sat = 1 - np.exp(-float(np.sum(wf * g["d_turning_deg"].abs().values)) / TURN_SAT_SCALE)
        wm_spd = float(np.sum(wf * g["abs_log_speedup_ratio"].fillna(0).values))

        rs_W = pd.to_numeric(g["rough_speedup_WTG_frac"], errors="coerce").values
        rs_M = pd.to_numeric(g["rough_speedup_MM_frac"], errors="coerce").values
        abs_rough = np.abs((rs_W + rs_M) / 2); v = ~np.isnan(abs_rough)
        if v.sum() == 0:
            continue
        wm_rough = float(np.sum(wf[v] * abs_rough[v]) / wf[v].sum()) * (1 - np.exp(-dm / PROXIMITY_GATE_SCALE))
        roughness_sat = 1 - np.exp(-wm_rough / ROUGH_SAT_SCALE)

        dz = float(g["dz"].iloc[0]); dz_sat = 1 - np.exp(-abs(dz) / DZ_SAT_SCALE)

        rows.append({
            "pair_id": pid, "abs_e": abs(e),
            "distance_m": float(dm), "distance_A": float(dA), "ratio_d_dA": float(dm / dA),
            "dist_sat": dist_sat, "turning_sat": turning_sat,
            "wm_abs_log_speedup": wm_spd, "roughness_sat": roughness_sat, "dz_sat": dz_sat,
            "RIX": pd.to_numeric(g.get("RIX_avg_0.3_overall_pair"), errors="coerce").iloc[0]
                   if "RIX_avg_0.3_overall_pair" in g.columns else np.nan,
            "meas_height": float(g["meas_height"].iloc[0]) if not pd.isna(g["meas_height"].iloc[0]) else np.nan,
        })
    return pd.DataFrame(rows)


def partial_corr(df, x, y, controls):
    def resid(t):
        A = np.column_stack([np.ones(len(df))] + [df[c].values for c in controls])
        b = df[t].values
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return b - A @ coef
    return np.corrcoef(resid(x), resid(y))[0, 1]


def main():
    df = pd.read_excel(INPUT_PATH)
    pf = build_pairs(df).dropna(subset=["abs_e", "dist_sat", "RIX"])
    print(f"Pairs: {len(pf)}\n")

    print("Distances (m):   ", f"min={pf['distance_m'].min():.0f}  median={pf['distance_m'].median():.0f}  max={pf['distance_m'].max():.0f}")
    print("distance_A (m):  ", f"min={pf['distance_A'].min():.0f}  median={pf['distance_A'].median():.0f}  max={pf['distance_A'].max():.0f}")
    print("ratio d/dA:      ", f"min={pf['ratio_d_dA'].min():.2f}  median={pf['ratio_d_dA'].median():.2f}  max={pf['ratio_d_dA'].max():.2f}")
    print("dist_sat:        ", f"min={pf['dist_sat'].min():.3f}  median={pf['dist_sat'].median():.3f}  max={pf['dist_sat'].max():.3f}\n")

    print("HYPOTHESIS 1 — is distance_A complexity-derived? (complex terrain -> small dA)")
    print(f"  corr(distance_A, RIX)          = {pf['distance_A'].corr(pf['RIX']):+.3f}   (expect negative)")
    print(f"  corr(distance_A, roughness_sat)= {pf['distance_A'].corr(pf['roughness_sat']):+.3f}")
    print(f"  corr(distance_m, RIX)          = {pf['distance_m'].corr(pf['RIX']):+.3f}   (siting: flat->far?)")

    print("\nHYPOTHESIS 2 — is dist_sat INVERTED vs difficulty? (complex -> LOW dist_sat)")
    print(f"  corr(dist_sat, RIX)            = {pf['dist_sat'].corr(pf['RIX']):+.3f}   (user predicts negative)")
    print(f"  corr(dist_sat, roughness_sat)  = {pf['dist_sat'].corr(pf['roughness_sat']):+.3f}")
    print(f"  corr(dist_sat, turning_sat)    = {pf['dist_sat'].corr(pf['turning_sat']):+.3f}")
    print(f"  corr(ratio_d_dA, RIX)          = {pf['ratio_d_dA'].corr(pf['RIX']):+.3f}")

    print("\nCorrelation with actual |error|:")
    for f in ["distance_m", "distance_A", "ratio_d_dA", "dist_sat"]:
        print(f"  {f:<14} Pearson={pf['abs_e'].corr(pf[f]):+.3f}  Spearman={pf['abs_e'].corr(pf[f], method='spearman'):+.3f}")

    print("\nHYPOTHESIS 3 — does dist_sat explain RESIDUAL variance (user's claim)?")
    others = ["turning_sat", "wm_abs_log_speedup", "roughness_sat", "dz_sat"]
    print(f"  partial corr(dist_sat, |e| | 4 other features) = {partial_corr(pf,'dist_sat','abs_e',others):+.3f}")
    print(f"  partial corr(distance_m, |e| | 4 other feats)  = {partial_corr(pf,'distance_m','abs_e',others):+.3f}")
    print(f"  (compare raw corr(dist_sat,|e|) = {pf['abs_e'].corr(pf['dist_sat']):+.3f})")

    print("\nTerrain split by RIX (flat vs complex):")
    med = pf["RIX"].median()
    flat = pf[pf["RIX"] <= med]; cplx = pf[pf["RIX"] > med]
    for name, s in [("FLAT  (RIX<=med)", flat), ("COMPLEX(RIX>med)", cplx)]:
        print(f"  {name} n={len(s)}: distance_m={s['distance_m'].median():.0f}m  distance_A={s['distance_A'].median():.0f}m  "
              f"dist_sat={s['dist_sat'].mean():.3f}  |e|={s['abs_e'].mean():.2%}")

    pf.to_csv(os.path.join(OUT_DIR, "distance_feature_table.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'distance_feature_table.csv')}")


if __name__ == "__main__":
    main()
