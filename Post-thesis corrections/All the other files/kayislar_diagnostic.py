"""
Kayislar (2022PA018) diagnostic — what does the model expect vs reality?
Includes the excluded Kayislar mast, computes all 5 features + the current model's
predicted sigma/mu for its pair(s), and compares to the actual error. Also prints its
sector energy concentration ("only 2 directions dominate energy").
No pymc — uses the saved JSON params applied as a held-out prediction.
"""
import json, numpy as np, pandas as pd

INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
JSON_PATH = r"Post-thesis corrections\Results\ws_uncertainty_pairlevel_results.json"
KAYISLAR = "2022PA018"

DZ_SAT_SCALE = 40; ROUGH_SAT_SCALE = 0.01; TURN_SAT_SCALE = 3.0; PROXIMITY_GATE_SCALE = 500.0
# production exclusions MINUS Kayislar (so its pairs are kept in for inspection)
EXCLUDED = ["2015WM018", "2021PA004", "2022PA008", "2011WM011", "2014WM011",
            "2019HE001", "2019HE002", "2019HE003", "2022PA021", "2023PA085",
            "2024PA014", "2012WM006", "2024PA107"]

J = json.load(open(JSON_PATH)); mp = J["model_params"]; sc = J["scalers"]


def z(col, v): return (v - sc[f"{col}_mean"]) / sc[f"{col}_std"]


def build(df):
    df = df.drop_duplicates().copy()
    df["mA"] = df["pair_id"].str.split("__").str[0]; df["mB"] = df["pair_id"].str.split("__").str[1]
    df = df[(~df["mA"].isin(EXCLUDED)) & (~df["mB"].isin(EXCLUDED))].copy()
    for o, n in {"d_turning_deg": "d_turning_deg_new", "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
                 "overall_speedup_MM_factor": "overall_speedup_MM_factor_new", "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
                 "rough_speedup_MM_frac": "rough_speedup_MM_frac_new"}.items():
        if n in df: m = df[n].notna(); df.loc[m, o] = df.loc[m, n]
    m = (df["overall_speedup_WTG_factor"] > 0) & (df["overall_speedup_MM_factor"] > 0)
    df["alsr"] = np.nan
    df.loc[m, "alsr"] = np.abs(np.log(df.loc[m, "overall_speedup_WTG_factor"] / df.loc[m, "overall_speedup_MM_factor"]))
    rows = []
    for pid, g in df.groupby("pair_id"):
        wp = pd.to_numeric(g["Sample_count_pred"], errors="coerce").fillna(0).values.astype(float)
        ws = pd.to_numeric(g["Sample_count_self"], errors="coerce").fillna(0).values.astype(float)
        if wp.sum() == 0 or ws.sum() == 0: wp = np.ones(len(g)); ws = np.ones(len(g))
        wp /= wp.sum(); ws /= ws.sum()
        WSs = float(np.sum(ws * g["Mean_windspeed_self"].values))
        e = (float(np.sum(wp * g["Mean_windspeed_predicted"].values)) - WSs) / WSs
        dm = float(g["distance_m"].iloc[0]); dA = g["distance_A"].iloc[0]
        if pd.isna(dA) or dA <= 0: continue
        wf = pd.to_numeric(g["freq_MM"], errors="coerce").fillna(0).values
        wf = wf / wf.sum() if wf.sum() > 0 else wp
        dist_sat = 1 - np.exp(-dm / dA)
        turning_sat = 1 - np.exp(-float(np.sum(wf * g["d_turning_deg"].abs().values)) / TURN_SAT_SCALE)
        spd = float(np.sum(wf * g["alsr"].fillna(0).values))
        rW = pd.to_numeric(g["rough_speedup_WTG_frac"], errors="coerce").values
        rM = pd.to_numeric(g["rough_speedup_MM_frac"], errors="coerce").values
        ar = np.abs((rW + rM) / 2); v = ~np.isnan(ar)
        wmr = float(np.sum(wf[v] * ar[v]) / wf[v].sum()) * (1 - np.exp(-dm / PROXIMITY_GATE_SCALE)) if v.sum() else np.nan
        roughness_sat = 1 - np.exp(-wmr / ROUGH_SAT_SCALE)
        dz = float(g["dz"].iloc[0]); dz_sat = 1 - np.exp(-abs(dz) / DZ_SAT_SCALE)
        # predicted sigma/mu (current model, applied as held-out)
        ls = (mp["log_sigma0"] + mp["gamma_dist"] * z("dist_sat", dist_sat)
              + mp["gamma_turning"] * z("turning_sat", turning_sat)
              + mp["gamma_speedup"] * z("wm_abs_log_speedup", spd)
              + mp["gamma_roughness"] * z("roughness_sat", roughness_sat)
              + mp["gamma_dz"] * z("dz_sat", dz_sat))
        sigma = np.exp(ls); mu = mp["beta_dz"] * z("dz", dz)
        rows.append(dict(pair_id=pid, dist_sat=dist_sat, turning_sat=turning_sat, spd=spd,
                         roughness_sat=roughness_sat, dz_sat=dz_sat, dz=dz, distance_m=dm,
                         pred_sigma=sigma * 100, pred_mu=mu * 100, actual=abs(e) * 100, signed=e * 100,
                         is_kay=(KAYISLAR in pid)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(INPUT_PATH)
    pf = build(df)
    kay = pf[pf["is_kay"]]
    print(f"Pairs: {len(pf)}   Kayislar pairs: {len(kay)}\n")
    print("=== KAYISLAR pair(s): model expectation vs reality ===")
    cols = ["pair_id", "dist_sat", "turning_sat", "spd", "roughness_sat", "dz_sat",
            "distance_m", "pred_sigma", "pred_mu", "actual", "signed"]
    show = kay[cols].copy()
    for c in ["dist_sat", "turning_sat", "spd", "roughness_sat", "dz_sat"]: show[c] = show[c].round(3)
    for c in ["pred_sigma", "pred_mu", "actual", "signed"]: show[c] = show[c].round(2)
    show["distance_m"] = show["distance_m"].round(0)
    print(show.to_string(index=False))

    print("\n=== z-scores of Kayislar features (how extreme is each vs training) ===")
    for _, r in kay.iterrows():
        zs = {f: round(z(col, r[col]), 2) for f, col in
              [("dist", "dist_sat"), ("turn", "turning_sat"), ("spd", "wm_abs_log_speedup"),
               ("rough", "roughness_sat"), ("dz_sat", "dz_sat")]}
        print(f"  {r['pair_id']}: {zs}")

    print("\n=== context: normal pairs |error| range ===")
    other = pf[~pf["is_kay"]]
    print(f"  other pairs: actual |e| mean={other['actual'].mean():.2f}%  max={other['actual'].max():.2f}%")
    print(f"  Kayislar   : actual |e| = {kay['actual'].values.round(2)}   predicted sigma = {kay['pred_sigma'].values.round(2)}%")

    # sector energy concentration for Kayislar
    print("\n=== Kayislar sector energy concentration (freq_MM) ===")
    raw = df[(df['pair_id'].str.contains(KAYISLAR))].drop_duplicates('pair_id')
    for pid in kay['pair_id']:
        g = df[df['pair_id'] == pid]
        fr = pd.to_numeric(g['freq_MM'], errors='coerce').fillna(0).values
        fr = fr / fr.sum() if fr.sum() else fr
        top = np.sort(fr)[::-1]
        print(f"  {pid}: top-2 sectors = {top[0]*100:.0f}% + {top[1]*100:.0f}% = {(top[0]+top[1])*100:.0f}% of energy")


if __name__ == "__main__":
    main()
