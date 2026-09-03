"""
Self-prediction sigma per mast — what does the current production model predict
for each mast's uncertainty when the "new turbine" is at the same location as the
mast itself (dist=0, dz=0, turning=0, log_speedup=0, roughness=mast's own).

Uses production fitted coefficients from ws_uncertainty_pairlevel_results.json.
No PyMC required — pure arithmetic on the posterior means.

At self-pair: rs_WTG = rs_MM = mast's own rs per sector, so
   roughness_pair = weighted_mean( |rs_own_sector| )
All other features -> 0. Then sigma = exp( log_sigma0 + sum_k gamma_k * z_k ).
"""
import os, json
import numpy as np
import pandas as pd

INPUT_XLSX = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
MODEL_JSON = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Bayesian_approach\Final model\Results\ws_uncertainty_pairlevel_results.json"
OUT_CSV    = os.path.join(os.path.dirname(__file__), "Results", "self_prediction_sigma.csv")

ROUGH_SAT_SCALE = 0.01
EXCLUDED = {"2015WM018","2021PA004","2022PA008","2022PA018","2011WM011","2014WM011",
            "2019HE001","2019HE002","2019HE003","2022PA021","2023PA085","2024PA014",
            "2012WM006","2024PA107"}

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# ── load fitted model ────────────────────────────────────────────────────────
with open(MODEL_JSON) as f:
    mj = json.load(f)
p = mj["model_params"]
s = mj["scalers"]
log_sigma0 = p["log_sigma0"]
gamma = {"dist_sat":       p["gamma_dist"],
         "turning_sat":    p["gamma_turning"],
         "wm_abs_log_speedup": p["gamma_speedup"],
         "roughness_sat":  p["gamma_roughness"],
         "dz_sat":         p["gamma_dz"]}

def z(feature_name, raw_value):
    return (raw_value - s[f"{feature_name}_mean"]) / s[f"{feature_name}_std"]

# ── load sector-level input ──────────────────────────────────────────────────
df = pd.read_excel(INPUT_XLSX)

# apply the same WindPRO upgrades production applies (so rs values match what
# was in the training set):
for old, new in [("rough_speedup_WTG_frac", "rough_speedup_WTG_frac_new"),
                 ("rough_speedup_MM_frac",  "rough_speedup_MM_frac_new")]:
    if new in df.columns and old in df.columns:
        mask = df[new].notna()
        df.loc[mask, old] = df.loc[mask, new]

# ── extract per-mast (mast-owned) sector data ────────────────────────────────
# Every mast appears as the "MM" in some pairs. In those rows freq_MM is the
# mast's own sector freq and rough_speedup_MM_frac is its own roughness speedup.
# Take one representative pair per mast (deterministic: first pair_id it appears
# in as MM). Sector data is a physical property of the mast, same across pairs.

def mast_of_pair(pair_id, role):
    parts = pair_id.split("__")
    return parts[0] if role == "WTG" else parts[1]

# Every row corresponds to one (pair, sector). We need to know which mast is MM
# in that row.  If the input has explicit mast columns, use them; otherwise
# split pair_id.
if "MM_mast" in df.columns:
    df["_mm"] = df["MM_mast"]
else:
    df["_mm"] = df["pair_id"].apply(lambda x: mast_of_pair(x, "MM"))

records = []
for mast, mrows in df.groupby("_mm"):
    if mast in EXCLUDED:
        continue
    # pick the first pair_id this mast is MM in (any is fine; sector values are
    # the mast's property, not the pair's).
    first_pair = mrows["pair_id"].iloc[0]
    grp = mrows[mrows["pair_id"] == first_pair]
    freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
    if freq.sum() == 0:
        continue
    w = freq / freq.sum()
    rs = pd.to_numeric(grp["rough_speedup_MM_frac"], errors="coerce").values
    valid = ~np.isnan(rs)
    if valid.sum() == 0:
        rs_self = np.nan
        roughness_sat = np.nan
    else:
        rs_self = float(np.sum(w[valid] * np.abs(rs[valid])) / w[valid].sum())
        roughness_sat = float(1.0 - np.exp(-rs_self / ROUGH_SAT_SCALE))

    # apply model: all features 0 except roughness_sat
    log_s = log_sigma0
    for feat in ["dist_sat","turning_sat","wm_abs_log_speedup","dz_sat"]:
        log_s += gamma[feat] * z(feat, 0.0)
    if not np.isnan(roughness_sat):
        log_s += gamma["roughness_sat"] * z("roughness_sat", roughness_sat)
    sigma_self = float(np.exp(log_s))
    records.append({"mast": mast,
                    "rs_own_wmean_abs": rs_self,
                    "roughness_sat_self": roughness_sat,
                    "sigma_self_pct": sigma_self * 100})

out = pd.DataFrame(records).sort_values("sigma_self_pct").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

print(f"\n{'mast':<15}{'|rs|_own':>10}{'roughness_sat':>16}{'sigma_self_%':>15}")
print("-" * 56)
for _, r in out.iterrows():
    rs_str = f"{r['rs_own_wmean_abs']:.4f}" if pd.notna(r['rs_own_wmean_abs']) else "  N/A "
    rsat_str = f"{r['roughness_sat_self']:.4f}" if pd.notna(r['roughness_sat_self']) else "  N/A "
    print(f"  {r['mast']:<13}{rs_str:>10}{rsat_str:>16}{r['sigma_self_pct']:>14.3f}%")

print(f"\nN = {len(out)}   min = {out['sigma_self_pct'].min():.3f}%   "
      f"median = {out['sigma_self_pct'].median():.3f}%   "
      f"max = {out['sigma_self_pct'].max():.3f}%")
print(f"Saved: {OUT_CSV}")
