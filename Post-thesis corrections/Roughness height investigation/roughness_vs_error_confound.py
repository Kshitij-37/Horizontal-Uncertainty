"""
Roughness feature vs actual error — is it a genuine signal or a height artifact?
--------------------------------------------------------------------------------
Investigates, on the EXISTING pair-level training data (no WAsP re-runs), whether
the roughness feature (formula A: |(rs_WTG+rs_MM)/2|) predicts actual cross-prediction
error independently of measurement height and distance.

Key questions:
  1. Does roughness_sat correlate with actual |error|?
  2. Is roughness_sat confounded with meas_height (lower masts -> bigger roughness)?
  3. Does roughness survive when we control for height (and distance)?  -> genuine signal
  4. Does meas_height add anything to error beyond roughness?          -> height signal

Also prints the raw speedup value corresponding to +N SD (Point 2 sanity check).

Uses only pandas/numpy (no pymc) so it runs in the default interpreter.
"""
import numpy as np
import pandas as pd
import os

INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
OUT_DIR = os.path.dirname(__file__)

# Match the pair-level model (ws_uncertainty_model_roughness_fix)
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

    # windPRO detailed upgrades
    upgrade = {
        "d_turning_deg": "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor": "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac": "rough_speedup_MM_frac_new",
    }
    for old, new in upgrade.items():
        if new in df.columns:
            m = df[new].notna()
            df.loc[m, old] = df.loc[m, new]

    # sector |log speedup ratio|
    m = (df["overall_speedup_WTG_factor"] > 0) & (df["overall_speedup_MM_factor"] > 0)
    df["abs_log_speedup_ratio"] = np.nan
    df.loc[m, "abs_log_speedup_ratio"] = np.abs(
        np.log(df.loc[m, "overall_speedup_WTG_factor"] / df.loc[m, "overall_speedup_MM_factor"])
    )

    rows = []
    for pid, g in df.groupby("pair_id"):
        wp = pd.to_numeric(g["Sample_count_pred"], errors="coerce").fillna(0).values.astype(float)
        ws = pd.to_numeric(g["Sample_count_self"], errors="coerce").fillna(0).values.astype(float)
        if wp.sum() == 0 or ws.sum() == 0:
            wp = np.ones(len(g)); ws = np.ones(len(g))
        wp = wp / wp.sum(); ws = ws / ws.sum()
        WS_pred = float(np.sum(wp * g["Mean_windspeed_predicted"].values))
        WS_self = float(np.sum(ws * g["Mean_windspeed_self"].values))
        e = (WS_pred - WS_self) / WS_self

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
        abs_rough = np.abs((rs_W + rs_M) / 2)          # formula A
        v = ~np.isnan(abs_rough)
        if v.sum() == 0:
            continue
        wm_rough = float(np.sum(wf[v] * abs_rough[v]) / wf[v].sum())
        wm_rough *= (1 - np.exp(-dm / PROXIMITY_GATE_SCALE))     # proximity gate
        roughness_sat = 1 - np.exp(-wm_rough / ROUGH_SAT_SCALE)

        dz = float(g["dz"].iloc[0])
        dz_sat = 1 - np.exp(-abs(dz) / DZ_SAT_SCALE)

        rows.append({
            "pair_id": pid,
            "abs_e": abs(e),
            "dist_sat": dist_sat,
            "turning_sat": turning_sat,
            "wm_abs_log_speedup": wm_spd,
            "roughness_sat": roughness_sat,
            "dz_sat": dz_sat,
            "meas_height": float(g["meas_height"].iloc[0]) if not pd.isna(g["meas_height"].iloc[0]) else np.nan,
            "distance_m": dm,
            "RIX": pd.to_numeric(g.get("RIX_avg_0.3_overall_pair"), errors="coerce").iloc[0]
                   if "RIX_avg_0.3_overall_pair" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def partial_corr(df, x, y, controls):
    """corr(x, y | controls) via residuals of OLS."""
    def resid(target):
        A = np.column_stack([np.ones(len(df))] + [df[c].values for c in controls])
        b = df[target].values
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return b - A @ coef
    return np.corrcoef(resid(x), resid(y))[0, 1]


def main():
    df = pd.read_excel(INPUT_PATH)
    pf = build_pairs(df).dropna(subset=["abs_e", "roughness_sat", "meas_height", "dist_sat"])
    print(f"Pairs analysed: {len(pf)}")
    print(f"meas_height range: {pf['meas_height'].min():.0f}-{pf['meas_height'].max():.0f} m "
          f"(median {pf['meas_height'].median():.0f})\n")

    feats = ["dist_sat", "turning_sat", "wm_abs_log_speedup", "roughness_sat", "dz_sat", "meas_height", "RIX"]
    print("Correlation with actual |error|:")
    print(f"  {'feature':<22}{'Pearson':>9}{'Spearman':>10}")
    for f in feats:
        sub = pf.dropna(subset=[f])
        pr = sub["abs_e"].corr(sub[f]); sr = sub["abs_e"].corr(sub[f], method="spearman")
        print(f"  {f:<22}{pr:>9.3f}{sr:>10.3f}")

    print("\nConfound check:")
    print(f"  corr(roughness_sat, meas_height) = {pf['roughness_sat'].corr(pf['meas_height']):+.3f}")
    print(f"  corr(roughness_sat, dist_sat)    = {pf['roughness_sat'].corr(pf['dist_sat']):+.3f}")

    print("\nPartial correlations of roughness_sat with |error|:")
    print(f"  raw                              = {pf['abs_e'].corr(pf['roughness_sat']):+.3f}")
    print(f"  controlling for meas_height      = {partial_corr(pf,'roughness_sat','abs_e',['meas_height']):+.3f}")
    print(f"  controlling for dist_sat         = {partial_corr(pf,'roughness_sat','abs_e',['dist_sat']):+.3f}")
    print(f"  controlling for height + dist    = {partial_corr(pf,'roughness_sat','abs_e',['meas_height','dist_sat']):+.3f}")

    print("\nDoes height add anything beyond roughness?")
    print(f"  corr(meas_height, |e|) raw       = {pf['abs_e'].corr(pf['meas_height']):+.3f}")
    print(f"  partial (control roughness_sat)  = {partial_corr(pf,'meas_height','abs_e',['roughness_sat']):+.3f}")

    print("\nLow vs high measurement height:")
    med = pf["meas_height"].median()
    lo = pf[pf["meas_height"] <= med]; hi = pf[pf["meas_height"] > med]
    print(f"  <= {med:.0f} m (n={len(lo)}): mean |e|={lo['abs_e'].mean():.2%}  mean roughness_sat={lo['roughness_sat'].mean():.3f}")
    print(f"  >  {med:.0f} m (n={len(hi)}): mean |e|={hi['abs_e'].mean():.2%}  mean roughness_sat={hi['roughness_sat'].mean():.3f}")

    # ---- Point 2: what does +N SD of speedup look like physically? ----
    print("\n" + "=" * 60)
    print("POINT 2: raw speedup value at +N SD (physical sanity check)")
    print("=" * 60)
    m, s = pf["wm_abs_log_speedup"].mean(), pf["wm_abs_log_speedup"].std()
    print(f"  wm_abs_log_speedup: mean={m:.4f}  std={s:.4f}  observed max={pf['wm_abs_log_speedup'].max():.4f}")
    print(f"  {'z':>4}{'raw |log ratio|':>16}{'implied speedup ratio (exp)':>30}")
    for z in [1, 3, 5, 10, 15]:
        raw = m + z * s
        print(f"  {z:>4}{raw:>16.3f}{np.exp(raw):>30.2f}x")

    pf.to_csv(os.path.join(OUT_DIR, "pair_features_with_height.csv"), index=False)
    print(f"\nSaved pair table: {os.path.join(OUT_DIR, 'pair_features_with_height.csv')}")


if __name__ == "__main__":
    main()
