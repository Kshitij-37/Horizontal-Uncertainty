"""
Worst-case self-prediction sigma under exp-A and exp-Q.

Reads the current corrected input Excel to find the max sector-level |rs|,
then computes self-prediction sigma for a hypothetical mast whose every sector
carries that maximum absolute roughness speedup.

Compares against the numbers computed from the prior sign-analysis value (0.0256).

Runs in seconds; only needs pandas + numpy + openpyxl (no PyMC).
"""
import json, os
import numpy as np
import pandas as pd

INPUT_XLSX  = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
EXP_A_JSON  = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Post-thesis corrections\sigma_only_refit\Results\exp-A__forest-out.json"
EXP_Q_JSON  = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Post-thesis corrections\sigma_only_refit\Results\exp-Q__forest-out.json"
ROUGH_SAT_SCALE = 0.01

# ── read the current corrected data ───────────────────────────────────────────
df = pd.read_excel(INPUT_XLSX)
# apply the same WindPRO upgrades used in the model
for old, new in [("rough_speedup_MM_frac", "rough_speedup_MM_frac_new"),
                 ("rough_speedup_WTG_frac", "rough_speedup_WTG_frac_new")]:
    if new in df.columns and old in df.columns:
        mask = df[new].notna()
        df.loc[mask, old] = df.loc[mask, new]

rs_MM  = pd.to_numeric(df["rough_speedup_MM_frac"],  errors="coerce")
rs_WTG = pd.to_numeric(df["rough_speedup_WTG_frac"], errors="coerce")

max_rs_MM_abs  = float(rs_MM.abs().max())
max_rs_WTG_abs = float(rs_WTG.abs().max())
max_rs = max(max_rs_MM_abs, max_rs_WTG_abs)

# also report signed extremes for reference
print("=" * 72)
print("CORRECTED-DATA ROUGHNESS EXTREMES")
print("=" * 72)
print(f"  rs_MM  range: [{rs_MM.min():+.5f}, {rs_MM.max():+.5f}]  max|rs_MM|  = {max_rs_MM_abs:.5f}")
print(f"  rs_WTG range: [{rs_WTG.min():+.5f}, {rs_WTG.max():+.5f}]  max|rs_WTG| = {max_rs_WTG_abs:.5f}")
print(f"  Overall max |rs| = {max_rs:.5f}")
print(f"  Prior sign-analysis value (from new 1.txt): 0.02560")
print(f"  Delta: {max_rs - 0.0256:+.5f}")

# ── load fitted coeffs ────────────────────────────────────────────────────────
with open(EXP_A_JSON) as f: A = json.load(f)
with open(EXP_Q_JSON) as f: Q = json.load(f)

def self_sigma(coeffs_json, form, max_rs_val):
    ls0   = coeffs_json["coeffs"]["log_sigma0"]
    beta  = coeffs_json["coeffs"]["betas"]["roughness_sat"]
    std_r = coeffs_json["feature_stds"]["roughness_sat"]
    rsat_A = 1.0 - np.exp(-max_rs_val / ROUGH_SAT_SCALE)   # formula A saturated value
    if form == "exp-A":
        rsat = rsat_A               # no gate
    elif form == "exp-Q":
        rsat = rsat_A * (1.0 - np.exp(-0.0 / 500.0))   # gate at d=0 → 0
    else:
        raise ValueError(form)
    log_s = ls0 + beta * rsat / std_r    # other 4 features are 0 at same-site
    return float(np.exp(log_s)), rsat_A, rsat

def report(max_rs_val, label):
    print(f"\n---- worst-case self-sigma at max|rs| = {max_rs_val:.5f}  ({label}) ----")
    for form, cj in [("exp-A", A), ("exp-Q", Q)]:
        sig, rsat_A_val, rsat_used = self_sigma(cj, form, max_rs_val)
        print(f"  {form}: roughness_sat_A = {rsat_A_val:.4f}, roughness_after_gate = {rsat_used:.4f}, "
              f"sigma_self = {sig*100:.3f}%")

report(0.02560, "assumed from prior sign analysis")
report(max_rs,   "actual, from current corrected data")

print("\n" + "=" * 72)
