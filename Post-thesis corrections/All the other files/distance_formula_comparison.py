"""
Distance Formula Comparison via LOO CV
--------------------------------------
Tests whether the current distance feature  dist_sat = 1 - exp(-distance_m / distance_A)
is the best parameterisation, or whether the dA-normalisation / saturation is costing
out-of-sample accuracy.

Only the DISTANCE feature changes between variants. The other four features
(turning_sat, wm_abs_log_speedup, roughness_sat=formula A, dz_sat) and the dz bias
term are held identical to the current model.

Variants:
  A_current : 1 - exp(-distance_m / distance_A)   (current model; dA = complexity-derived)
  B_raw     : distance_m                           (raw linear, no normalisation/saturation)
  C_fix5k   : 1 - exp(-distance_m / 5000)          (fixed saturation scale)
  D_fix8k   : 1 - exp(-distance_m / 8000)          (fixed saturation scale)
  E_log     : log(distance_m)                      (log distance)
  F_drop    : (no distance feature)                (4-feature model)

For each variant it reports pair-level LOO CV Pearson r, Spearman rho, bias, mean
predicted sigma, and the same-location floor (all features at their raw minimum).

RUN IN pymc-env:  python distance_formula_comparison.py
(Requires pymc / arviz. Self-contained: uses only pandas/numpy for feature building.)
"""
import os
import sys
import time
import contextlib
import numpy as np
import pandas as pd
import pymc as pm

# ─── CONFIG (matches the current pair-level model) ──────────────────────────
INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DZ_SAT_SCALE = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE = 3.0
PROXIMITY_GATE_SCALE = 500.0

EXCLUDED_MASTS = [
    "2015WM018", "2021PA004", "2022PA008", "2022PA018",
    "2011WM011", "2014WM011", "2019HE001", "2019HE002", "2019HE003",
    "2022PA021", "2023PA085", "2024PA014", "2012WM006", "2024PA107",
]

# Fast LOO settings — bump up for a final run
FAST_DRAWS = 500
FAST_TUNE = 500
FAST_CHAINS = 2
# cores=1 -> sequential sampling. REQUIRED on Windows: parallel sampling throws
# BrokenPipeError (the worker pipe closes, made worse by stdout redirection).
FAST_CORES = 1

# The four non-distance features, held constant across variants
OTHER_FEATURES = [
    ("gamma_turning",   "turning_sat",        "Saturating |turning|"),
    ("gamma_speedup",   "wm_abs_log_speedup", "WM |log speedup ratio|"),
    ("gamma_roughness", "roughness_sat",      "Saturating |roughness| (formula A)"),
    ("gamma_dz",        "dz_sat",             "Saturating |dz|"),
]

# Distance variants: key -> (raw_col or None for drop, description)
DISTANCE_VARIANTS = {
    "A_current": ("dist_sat",     "1 - exp(-d / distance_A)  (current)"),
    "B_raw":     ("dist_raw",     "distance_m  (raw linear)"),
    "C_fix5k":   ("dist_fix5000", "1 - exp(-d / 5000)"),
    "D_fix8k":   ("dist_fix8000", "1 - exp(-d / 8000)"),
    "E_log":     ("dist_log",     "log(distance_m)"),
    "F_drop":    (None,           "no distance feature (4-feature)"),
}


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_o, old_e = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull; sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_o; sys.stderr = old_e


# ─── PAIR-LEVEL FEATURE BUILD ───────────────────────────────────────────────
def build_pairs(df):
    """One row per directional pair, with all candidate distance columns."""
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
        WS_self = float(np.sum(ws * g["Mean_windspeed_self"].values))
        e = (float(np.sum(wp * g["Mean_windspeed_predicted"].values)) - WS_self) / WS_self

        dm = float(g["distance_m"].iloc[0]); dA = g["distance_A"].iloc[0]
        if pd.isna(dA) or dA <= 0:
            continue

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
            "pair_id": pid, "e_overall": e,
            "location": g["location"].iloc[0] if "location" in g.columns else "",
            "distance_m": dm,
            # distance variants
            "dist_sat":     1 - np.exp(-dm / dA),
            "dist_raw":     dm,
            "dist_fix5000": 1 - np.exp(-dm / 5000.0),
            "dist_fix8000": 1 - np.exp(-dm / 8000.0),
            "dist_log":     np.log(max(dm, 1.0)),
            # held-constant features
            "turning_sat": turning_sat, "wm_abs_log_speedup": wm_spd,
            "roughness_sat": roughness_sat, "dz_sat": dz_sat, "dz": dz,
        })
    return pd.DataFrame(rows)


def feature_config_for(variant_key):
    dist_col, _ = DISTANCE_VARIANTS[variant_key]
    if dist_col is None:
        return list(OTHER_FEATURES)
    return [("gamma_dist", dist_col, f"distance ({dist_col})")] + list(OTHER_FEATURES)


# ─── Z-SCORING + MODEL + LOO (generic over feature_config) ──────────────────
def rebuild_data(pair_df, feature_config):
    pair_df = pair_df.copy()
    cols = [rc for _, rc, _ in feature_config] + ["dz", "e_overall"]
    pair_df = pair_df[~pair_df[cols].isna().any(axis=1)].copy()

    scalers = {}
    for _, rc, _ in feature_config:
        mean = pair_df[rc].mean(); std = pair_df[rc].std() or 1.0
        pair_df[f"{rc}_z"] = (pair_df[rc] - mean) / std
        scalers[f"{rc}_mean"] = mean; scalers[f"{rc}_std"] = std
    dz_mean = pair_df["dz"].mean(); dz_std = pair_df["dz"].std() or 1.0
    pair_df["dz_z"] = (pair_df["dz"] - dz_mean) / dz_std
    scalers["dz_mean"] = dz_mean; scalers["dz_std"] = dz_std

    data = {"e": pair_df["e_overall"].values, "dz_z": pair_df["dz_z"].values,
            "scalers": scalers, "df": pair_df}
    for _, rc, _ in feature_config:
        data[f"{rc}_z"] = pair_df[f"{rc}_z"].values
    return data


def fit_model(data, feature_config, draws=FAST_DRAWS, tune=FAST_TUNE):
    with pm.Model() as model:
        e_data = pm.Data("e", data["e"])
        dz_z = pm.Data("dz_z", data["dz_z"])
        fdata = {gn: pm.Data(f"{gn}_z", data[f"{rc}_z"]) for gn, rc, _ in feature_config}

        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=-3.9, sigma=0.5)
        beta_dz = pm.Normal("beta_dz", mu=0, sigma=0.05)
        gammas = {gn: pm.HalfNormal(gn, sigma=0.3) for gn, _, _ in feature_config}

        log_sigma = log_sigma0
        for gn in gammas:
            log_sigma = log_sigma + gammas[gn] * fdata[gn]
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        mu = pm.Deterministic("mu", beta_dz * dz_z)
        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=e_data)

        idata = pm.sample(draws=draws, tune=tune, target_accept=0.95,
                          chains=FAST_CHAINS, cores=FAST_CORES,
                          progressbar=False, random_seed=42)
    return idata


def run_loo(pair_df, feature_config, draws=FAST_DRAWS, tune=FAST_TUNE):
    pair_df = pair_df.copy()
    pair_df["_grp"] = pair_df["pair_id"].apply(lambda x: "__".join(sorted(x.split("__"))))
    folds = pair_df.groupby("_grp")["pair_id"].unique().to_dict()

    results = []
    for fold_key, held in folds.items():
        train = pair_df[~pair_df["pair_id"].isin(held)].copy()
        test = pair_df[pair_df["pair_id"].isin(held)].copy()
        if len(train) < 5:
            continue
        tdata = rebuild_data(train, feature_config)
        try:
            with suppress_output():
                idata = fit_model(tdata, feature_config, draws=draws, tune=tune)
        except Exception as ex:
            print(f"    fold {fold_key} failed: {ex}")
            continue

        ls0 = float(idata.posterior["log_sigma0"].values.mean())
        bdz = float(idata.posterior["beta_dz"].values.mean())
        gam = {gn: float(idata.posterior[gn].values.mean()) for gn, _, _ in feature_config}
        sc = tdata["scalers"]

        def z(col, val):
            return (val - sc.get(f"{col}_mean", 0)) / sc.get(f"{col}_std", 1)

        for _, r in test.iterrows():
            ls = ls0 + sum(gam[gn] * z(rc, r[rc]) for gn, rc, _ in feature_config)
            results.append({
                "pair_id": r["pair_id"],
                "predicted_sigma": float(np.exp(ls)),
                "actual_error": abs(r["e_overall"]),
            })

    rdf = pd.DataFrame(results)
    if len(rdf) < 2:
        return rdf, np.nan, np.nan, np.nan
    r_p = rdf["predicted_sigma"].corr(rdf["actual_error"])
    r_s = rdf["predicted_sigma"].corr(rdf["actual_error"], method="spearman")
    bias = (rdf["predicted_sigma"] - rdf["actual_error"]).mean()
    return rdf, r_p, r_s, bias


def full_fit_floor(pair_df, feature_config):
    """Fit on all data; return sigma0 (z=0) and same-location floor (all raw features minimal)."""
    data = rebuild_data(pair_df, feature_config)
    with suppress_output():
        idata = fit_model(data, feature_config)
    ls0 = float(idata.posterior["log_sigma0"].values.mean())
    gam = {gn: float(idata.posterior[gn].values.mean()) for gn, _, _ in feature_config}
    sc = data["scalers"]
    # same-location: every feature at its raw minimum in the data
    ls_floor = ls0
    for gn, rc, _ in feature_config:
        raw_min = pair_df[rc].min()
        ls_floor += gam[gn] * (raw_min - sc[f"{rc}_mean"]) / sc[f"{rc}_std"]
    return float(np.exp(ls0)), float(np.exp(ls_floor))


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DISTANCE FORMULA COMPARISON via LOO CV")
    print("=" * 70)
    df = pd.read_excel(INPUT_PATH)
    pair_df = build_pairs(df)
    print(f"Pairs: {len(pair_df)}\n")

    out = []
    for key, (dist_col, desc) in DISTANCE_VARIANTS.items():
        print(f"{'=' * 70}\nVARIANT {key}: {desc}\n{'=' * 70}")
        fc = feature_config_for(key)
        t0 = time.time()
        _, r_p, r_s, bias = run_loo(pair_df, fc)
        try:
            sigma0, floor = full_fit_floor(pair_df, fc)
        except Exception as ex:
            print(f"    floor fit failed: {ex}"); sigma0, floor = np.nan, np.nan
        dt = (time.time() - t0) / 60
        print(f"  Pearson r={r_p:.4f}  Spearman={r_s:.4f}  bias={bias:+.4f} "
              f"({bias*100:+.2f}pp)  sigma0={sigma0*100:.2f}%  floor={floor*100:.3f}%  [{dt:.1f}m]")
        out.append({"variant": key, "description": desc, "pearson_r": r_p,
                    "spearman_rho": r_s, "bias": bias, "sigma0_pct": sigma0 * 100,
                    "floor_pct": floor * 100, "minutes": dt})

    summ = pd.DataFrame(out).sort_values("pearson_r", ascending=False)
    print("\n" + "=" * 70 + "\nSUMMARY (sorted by Pearson r)\n" + "=" * 70)
    print(f"  {'variant':<10}{'Pearson':>9}{'Spearman':>10}{'bias(pp)':>10}{'sigma0%':>9}{'floor%':>9}")
    print("  " + "-" * 62)
    for _, r in summ.iterrows():
        print(f"  {r['variant']:<10}{r['pearson_r']:>9.4f}{r['spearman_rho']:>10.4f}"
              f"{r['bias']*100:>+10.2f}{r['sigma0_pct']:>9.2f}{r['floor_pct']:>9.3f}")

    csv = os.path.join(RESULTS_DIR, "distance_formula_comparison.csv")
    summ.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")
    best = summ.iloc[0]
    print(f"\nBEST BY PEARSON: {best['variant']} — {best['description']}  (r={best['pearson_r']:.4f})")
    print("Note: with n≈40 and collinear features, small Pearson gaps are within noise —"
          " prefer a variant only if it wins on BOTH Pearson and Spearman with sensible bias.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
