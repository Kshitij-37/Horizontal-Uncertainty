"""
WS Horizontal Uncertainty — Pair-Level Bayesian Model (exp-M2c FINAL CANDIDATE)

===============================================================================
WHAT'S DIFFERENT FROM THE THESIS-FINAL MODEL
===============================================================================

Three deliberate changes from `ws_uncertainty_model_pairlevel_roughsat`:

  1. BIAS TERM REMOVED (mu = 0)
     The old model had mu = beta_dz * dz_z as a signed-error mean predictor.
     This model drops that entirely. Rationale: supervisor was skeptical of the
     bias/uncertainty separation, and beta_dz was contributing < 2% of the total
     explanation. Model is now purely a sigma predictor.

  2. TARGET CHANGED FROM SIGNED e TO |e|, LIKELIHOOD → HalfStudentT
     Old model: e_overall ~ StudentT(nu, mu, sigma)
     New model: |e_overall| ~ HalfStudentT(nu, sigma)
     Sigma retains the "scale of the underlying two-sided distribution"
     interpretation, so downstream P75/P90/P99 multipliers (0.67 sigma, 1.28 sigma,
     2.33 sigma) remain valid.

  3. ROUGHNESS FEATURE IS ADAPTIVE (formula M2c = "std-first, min-later, centered")
     The old model used formula A (magnitude): |(rs_W + rs_M)/2| — a magnitude
     that does not vanish at same-site, causing the terrain-dependent floor
     problem (0.28% flat → 1.21% rough).

     exp-M2c uses BOTH formula A (magnitude) and formula B (mismatch |rs_W - rs_M|),
     z-scores each separately on the training set, then takes min per pair:
         z_A = (sat_A - mean_A_train) / std_A_train
         z_B = (sat_B - mean_B_train) / std_B_train
         rough_M2c = min(z_A, z_B)
     Rationale: at same-site sat_B = 0 -> z_B = -mean_B/std_B (constant negative)
     -> min picks z_B for all rough sites -> nearly-constant floor across masts.
     Per-formula standardization is mathematically principled (each formula
     scaled by its own natural spread, not a mixed distribution).

===============================================================================
MODEL SPECIFICATION
===============================================================================

  |e_overall_i| ~ HalfStudentT(nu, sigma_i)

  log(sigma_i) = log_sigma0
               + gamma_dist    * z(dist_sat)         # centered z-score
               + gamma_turning * z(turning_sat)      # centered z-score
               + gamma_speedup * z(wm_abs_log_speedup)  # centered z-score
               + gamma_dz      * z(dz_sat)           # centered z-score
               + gamma_rough   * x_rough             # min(z_A, z_B) — adaptive roughness

  where:
    z(f)    = (f - mean_train(f)) / std_train(f)
    z_A     = (roughness_sat_A - mean_train(sat_A)) / std_train(sat_A)
    z_B     = (roughness_sat_B - mean_train(sat_B)) / std_train(sat_B)
    x_rough = min(z_A, z_B)

Features (5 sigma, all pair-level):
  1. dist_sat              — 1 - exp(-d / d_A), saturating distance
  2. turning_sat           — 1 - exp(-wm_abs_turning / TURN_SAT_SCALE)
  3. wm_abs_log_speedup    — Weighted mean |log(speedup_WTG / speedup_MM)|
  4. roughness (M2c)       — min(z(sat_A), z(sat_B)) where
                                 sat_A = 1 - exp(-wm_abs_roughness_A / ROUGH_SAT_SCALE)
                                 sat_B = 1 - exp(-wm_abs_roughness_B / ROUGH_SAT_SCALE)
  5. dz_sat                — 1 - exp(-|dz| / 40), saturating height difference

  Bias/mu term: NONE (removed).

Sector aggregation weights: MM Weibull frequency distribution (freq_MM).

===============================================================================
OUTPUTS
===============================================================================

Written to `Results/` next to this script:
  - model_results.json                    Fitted params, LOO metrics, self-pred summary
  - mcmc_trace.nc                         Full MCMC trace (ArviZ InferenceData)
  - loo_pair_predictions.csv              Per-pair LOO predictions (pair holdout)
  - loo_site_predictions.csv              Per-pair LOO predictions (site holdout)
  - loo_pair_scatter.html                 Interactive scatter plot (pair holdout)
  - loo_site_scatter.html                 Interactive scatter plot (site holdout)
  - self_prediction_per_mast.csv          Per-mast same-site sigma (42 rows)
  - feature_contributions_per_pair.csv    Per-pair log(sigma) breakdown by feature

===============================================================================
"""
import os
import sys
import json
import math
import itertools
import contextlib
import multiprocessing as mp
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
from scipy.special import gamma as gamma_func
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

# feature_config for non-roughness features (roughness handled separately in M2c)
NON_ROUGH_FEATURE_CONFIG = [
    # (gamma_name,      raw_col,               display_name)
    ("gamma_dist",     "dist_sat",             "Saturating distance (1-exp(-d/dA))"),
    ("gamma_turning",  "turning_sat",          "Saturating |turning| (1-exp(-t/s))"),
    ("gamma_speedup",  "wm_abs_log_speedup",   "WM |log speedup ratio|"),
    ("gamma_dz",       "dz_sat",               "Saturating |dz| (1-exp(-|dz|/40))"),
]
# roughness (M2c) added as a separate 5th feature with its own handling
ROUGH_GAMMA_NAME    = "gamma_roughness"
ROUGH_DISPLAY_NAME  = "Adaptive roughness M2c: min(z(sat_A), z(sat_B))"

DZ_SAT_SCALE    = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE  = 3.0

EXCLUDED_MASTS = [
    "2015WM018", "2021PA004", "2022PA008",   # Sallachy — hills and valleys
    "2022PA018",                             # Kayislar — only 2 dominant directions
    "2019HE001", "2019HE002", "2019HE003",   # Herzhausen CFD — not comparable to WAsP
    "2022PA021",                             # Taaibos — 3-mast, dropped for fit
    "2023PA085",                             # Ukhanda — 2 mast + LiDAR
    "2024PA014",                             # Balver Wald — uncertain site
    #"2024PA107",                             # Slovenska East — missing displacement
    #"2012WM006",                             # Malarberget
    #"2011WM011", "2014WM011",                # Hultema
    
    # FOREST masts INCLUDED for exp-M2c (adaptive formula handles them):
    #   2011WM011, 2014WM011 (Hultema), 2012WM006 (Malarberget)
]

# Sampler settings
FULL_FIT_DRAWS = 2000
FULL_FIT_TUNE  = 2000
FULL_FIT_CHAINS = 4
FULL_FIT_CORES  = 1        # Windows requires cores=1 for stability
LOO_DRAWS      = 1000
LOO_TUNE       = 1000
LOO_CHAINS     = 2
LOO_CORES      = 1

# Priors (production-style, matches thesis except that the roughness prior is now
# for a min-of-z feature that can be negative)
NU_ALPHA, NU_BETA        = 2.0, 0.2
LOG_SIGMA0_MU, LOG_SIGMA0_SD = -3.9, 0.5
GAMMA_SD                 = 0.3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results")
# Explicit output file paths (no more prefix pattern — renames replaced the m2c prefix)
OUTPUT_LOO_PAIR_CSV     = os.path.join(RESULTS_DIR, "loo_pair_predictions.csv")
OUTPUT_LOO_SITE_CSV     = os.path.join(RESULTS_DIR, "loo_site_predictions.csv")
OUTPUT_LOO_PAIR_HTML    = os.path.join(RESULTS_DIR, "loo_pair_scatter.html")
OUTPUT_LOO_SITE_HTML    = os.path.join(RESULTS_DIR, "loo_site_scatter.html")
OUTPUT_SELF_PRED_CSV    = os.path.join(RESULTS_DIR, "self_prediction_per_mast.csv")
OUTPUT_CONTRIB_CSV      = os.path.join(RESULTS_DIR, "feature_contributions_per_pair.csv")
OUTPUT_MODEL_JSON       = os.path.join(RESULTS_DIR, "model_results.json")
OUTPUT_IDATA_NC         = os.path.join(RESULTS_DIR, "mcmc_trace.nc")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ─────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────
def build_pair_training_data(df: pd.DataFrame) -> dict:
    """Aggregate sector-level data to one row per directional pair, computing
    BOTH formula A (magnitude) and formula B (mismatch) roughness values so that
    the M2c fit can standardize each separately.
    """
    d = df.copy()

    required = [
        "pair_id", "sector_name", "location",
        "Mean_windspeed_predicted", "Mean_windspeed_self",
        "d_turning_deg",
        "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
        "weight_energy_predicted",
        "Sample_count_pred", "Sample_count_self",
        "distance_m", "distance_A",
        "dz",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Use windPRO detailed values where available (matches production behavior)
    upgrade_map = {
        "d_turning_deg":              "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor":  "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac":     "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac":      "rough_speedup_MM_frac_new",
    }
    n_upgraded = 0
    for old_col, new_col in upgrade_map.items():
        if new_col in d.columns:
            mask = d[new_col].notna()
            n_rows = mask.sum()
            if n_rows > 0:
                d.loc[mask, old_col] = d.loc[mask, new_col]
                n_upgraded = max(n_upgraded, n_rows)
    if n_upgraded > 0:
        print(f"  WindPRO detailed values: {n_upgraded}/{len(d)} rows upgraded")

    # Sector-level |log speedup ratio|
    mask_spd = (
        d["overall_speedup_WTG_factor"].notna() &
        d["overall_speedup_MM_factor"].notna() &
        (d["overall_speedup_WTG_factor"] > 0) &
        (d["overall_speedup_MM_factor"] > 0)
    )
    d["abs_log_speedup_ratio"] = np.nan
    d.loc[mask_spd, "abs_log_speedup_ratio"] = np.abs(
        np.log(d.loc[mask_spd, "overall_speedup_WTG_factor"] /
               d.loc[mask_spd, "overall_speedup_MM_factor"])
    )

    # ── Aggregate to pair level ──────────────────────────────────
    pair_rows = []
    for pair_id, grp in d.groupby("pair_id"):
        w_energy = grp["weight_energy_predicted"].values
        if w_energy.sum() == 0:
            continue

        # Sample count weights for computing overall WS
        w_pred = pd.to_numeric(grp["Sample_count_pred"], errors="coerce").fillna(0).values
        w_self = pd.to_numeric(grp["Sample_count_self"], errors="coerce").fillna(0).values
        if w_pred.sum() == 0 or w_self.sum() == 0:
            print(f"  WARNING: zero sample counts for {pair_id}, falling back to equal weights")
            w_pred = np.ones(len(grp))
            w_self = np.ones(len(grp))
        w_pred_norm = w_pred / w_pred.sum()
        w_self_norm = w_self / w_self.sum()

        # Target: frequency-weighted overall error
        WS_pred_overall = float(np.sum(w_pred_norm * grp["Mean_windspeed_predicted"].values))
        WS_self_overall = float(np.sum(w_self_norm * grp["Mean_windspeed_self"].values))
        e_overall = (WS_pred_overall - WS_self_overall) / WS_self_overall

        # Feature 1: saturating distance
        distance_m = grp["distance_m"].iloc[0]
        distance_A = grp["distance_A"].iloc[0]
        if pd.isna(distance_A) or distance_A <= 0:
            print(f"  WARNING: distance_A NaN/zero for {pair_id}, skipping")
            continue
        dist_sat = float(1.0 - np.exp(-distance_m / distance_A))

        # Sector weights: MM Weibull frequency
        w_freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
        w_freq_norm = w_freq / w_freq.sum() if w_freq.sum() > 0 else w_pred_norm

        # Feature 2: weighted mean |turning| → saturated
        abs_turn = grp["d_turning_deg"].abs().values
        wm_abs_turning = float(np.sum(w_freq_norm * abs_turn))
        turning_sat = float(1.0 - np.exp(-wm_abs_turning / TURN_SAT_SCALE))

        # Feature 3: weighted mean |log speedup ratio|
        abs_log_spd = grp["abs_log_speedup_ratio"].fillna(0).values
        wm_abs_log_speedup = float(np.sum(w_freq_norm * abs_log_spd))

        # Roughness — BOTH FORMULAS (needed for M2c)
        rs_WTG = pd.to_numeric(grp["rough_speedup_WTG_frac"], errors="coerce").values \
            if "rough_speedup_WTG_frac" in grp.columns else np.full(len(grp), np.nan)
        rs_MM = pd.to_numeric(grp["rough_speedup_MM_frac"], errors="coerce").values \
            if "rough_speedup_MM_frac" in grp.columns else np.full(len(grp), np.nan)
        valid_r = ~(np.isnan(rs_WTG) | np.isnan(rs_MM))

        # Formula A: magnitude of the signed mean
        abs_rough_A = np.abs((rs_WTG + rs_MM) / 2)
        if valid_r.sum() > 0:
            wm_abs_rough_A = float(
                np.sum(w_freq_norm[valid_r] * abs_rough_A[valid_r]) /
                w_freq_norm[valid_r].sum()
            )
        else:
            wm_abs_rough_A = np.nan

        # Formula B: mismatch |rs_W - rs_M|
        abs_rough_B = np.abs(rs_WTG - rs_MM)
        if valid_r.sum() > 0:
            wm_abs_rough_B = float(
                np.sum(w_freq_norm[valid_r] * abs_rough_B[valid_r]) /
                w_freq_norm[valid_r].sum()
            )
        else:
            wm_abs_rough_B = np.nan

        # Saturate each
        rough_sat_A = (1.0 - np.exp(-wm_abs_rough_A / ROUGH_SAT_SCALE)
                       if not np.isnan(wm_abs_rough_A) else np.nan)
        rough_sat_B = (1.0 - np.exp(-wm_abs_rough_B / ROUGH_SAT_SCALE)
                       if not np.isnan(wm_abs_rough_B) else np.nan)

        # Feature 5: saturating |dz|
        dz = float(grp["dz"].iloc[0]) if "dz" in grp.columns else np.nan
        dz_sat = float(1.0 - np.exp(-abs(dz) / DZ_SAT_SCALE)) if not pd.isna(dz) else np.nan

        pair_rows.append({
            "pair_id":              pair_id,
            "e_overall":            e_overall,
            "abs_e_overall":        abs(e_overall),
            "WS_pred_overall":      WS_pred_overall,
            "WS_self_overall":      WS_self_overall,
            "dist_sat":             dist_sat,
            "wm_abs_turning":       wm_abs_turning,
            "turning_sat":          turning_sat,
            "wm_abs_log_speedup":   wm_abs_log_speedup,
            "wm_abs_roughness_A":   wm_abs_rough_A,
            "wm_abs_roughness_B":   wm_abs_rough_B,
            "rough_sat_A":          rough_sat_A,
            "rough_sat_B":          rough_sat_B,
            "dz":                   dz,
            "dz_sat":               dz_sat,
            "abs_dz":               abs(dz) if not pd.isna(dz) else np.nan,
            "location":             grp["location"].iloc[0] if "location" in grp.columns else "",
            "distance_m":           distance_m,
        })

    pair_df = pd.DataFrame(pair_rows)

    # Drop pairs with NaN in any feature or target
    required_cols = ([rc for _, rc, _ in NON_ROUGH_FEATURE_CONFIG] +
                     ["rough_sat_A", "rough_sat_B", "e_overall"])
    nan_mask = pair_df[required_cols].isna().any(axis=1)
    if nan_mask.any():
        dropped = pair_df.loc[nan_mask, "pair_id"].tolist()
        print(f"  WARNING: dropping {nan_mask.sum()} pairs with NaN features: {dropped}")
        pair_df = pair_df[~nan_mask].copy()

    print(f"  Dataset: {len(pair_df)} directional pairs")
    print(f"  Target |e_overall|: mean={pair_df['abs_e_overall'].mean()*100:.2f}%  "
          f"std={pair_df['abs_e_overall'].std()*100:.2f}%  "
          f"max={pair_df['abs_e_overall'].max()*100:.1f}%")

    print(f"\n  {'pair_id':<35} {'WS_pred':>8} {'WS_self':>8} {'e_overall':>10}")
    print(f"  {'-' * 65}")
    for _, row in pair_df.iterrows():
        print(f"  {row['pair_id']:<35} {row['WS_pred_overall']:>8.3f} "
              f"{row['WS_self_overall']:>8.3f} {row['e_overall']*100:>+9.2f}%")

    # Standardize non-rough features (centered z-score, production-style)
    scalers = {}
    for _, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
        mean = pair_df[raw_col].mean()
        std  = pair_df[raw_col].std()
        std  = std if std > 0 else 1.0
        pair_df[f"{raw_col}_z"] = (pair_df[raw_col] - mean) / std
        scalers[f"{raw_col}_mean"] = mean
        scalers[f"{raw_col}_std"]  = std

    # Standardize roughness_A and roughness_B (per-formula, centered)
    for r in ["rough_sat_A", "rough_sat_B"]:
        mean = pair_df[r].mean()
        std  = pair_df[r].std()
        std  = std if std > 0 else 1.0
        pair_df[f"{r}_z"] = (pair_df[r] - mean) / std
        scalers[f"{r}_mean"] = mean
        scalers[f"{r}_std"]  = std

    # Combined roughness feature = min(z_A, z_B)
    pair_df["rough_M2c"] = np.minimum(pair_df["rough_sat_A_z"], pair_df["rough_sat_B_z"])

    print(f"\n  Feature ranges (raw, before standardization):")
    for _, raw_col, display in NON_ROUGH_FEATURE_CONFIG:
        print(f"    {display:<45} [{pair_df[raw_col].min():.4f}, {pair_df[raw_col].max():.4f}]")
    print(f"    {'roughness sat_A (magnitude)':<45} "
          f"[{pair_df['rough_sat_A'].min():.4f}, {pair_df['rough_sat_A'].max():.4f}]")
    print(f"    {'roughness sat_B (mismatch)':<45} "
          f"[{pair_df['rough_sat_B'].min():.4f}, {pair_df['rough_sat_B'].max():.4f}]")
    print(f"    {'rough_M2c = min(z_A, z_B)':<45} "
          f"[{pair_df['rough_M2c'].min():.4f}, {pair_df['rough_M2c'].max():.4f}]")

    result = {
        "e_abs":     pair_df["abs_e_overall"].values,
        "e_signed":  pair_df["e_overall"].values,
        "pair_ids":  pair_df["pair_id"].values,
        "scalers":   scalers,
        "df":        pair_df,
    }
    for _, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
        result[f"{raw_col}_z"] = pair_df[f"{raw_col}_z"].values
    result["rough_M2c"] = pair_df["rough_M2c"].values

    return result


def rebuild_data_dict(pair_df: pd.DataFrame) -> dict:
    """Rebuild data dict from a subset (for LOO CV — computes new scalers on the
    training fold, avoiding data leakage)."""
    pair_df = pair_df.copy()

    required_cols = ([rc for _, rc, _ in NON_ROUGH_FEATURE_CONFIG] +
                     ["rough_sat_A", "rough_sat_B", "e_overall"])
    nan_mask = pair_df[required_cols].isna().any(axis=1)
    if nan_mask.any():
        pair_df = pair_df[~nan_mask].copy()

    scalers = {}
    for _, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
        mean = pair_df[raw_col].mean()
        std  = pair_df[raw_col].std()
        std  = std if std > 0 else 1.0
        pair_df[f"{raw_col}_z"] = (pair_df[raw_col] - mean) / std
        scalers[f"{raw_col}_mean"] = mean
        scalers[f"{raw_col}_std"]  = std

    for r in ["rough_sat_A", "rough_sat_B"]:
        mean = pair_df[r].mean()
        std  = pair_df[r].std()
        std  = std if std > 0 else 1.0
        pair_df[f"{r}_z"] = (pair_df[r] - mean) / std
        scalers[f"{r}_mean"] = mean
        scalers[f"{r}_std"]  = std

    pair_df["rough_M2c"] = np.minimum(pair_df["rough_sat_A_z"], pair_df["rough_sat_B_z"])

    result = {
        "e_abs":    pair_df["abs_e_overall"].values,
        "e_signed": pair_df["e_overall"].values,
        "pair_ids": pair_df["pair_id"].values,
        "scalers":  scalers,
        "df":       pair_df,
    }
    for _, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
        result[f"{raw_col}_z"] = pair_df[f"{raw_col}_z"].values
    result["rough_M2c"] = pair_df["rough_M2c"].values
    return result


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
def fit_model(data: dict, draws=FULL_FIT_DRAWS, tune=FULL_FIT_TUNE,
              chains=FULL_FIT_CHAINS, cores=FULL_FIT_CORES):
    """Sigma-only exp-M2c model with HalfStudentT likelihood on |e|.

    log(sigma) = log_sigma0 + sum_k gamma_k * z_k + gamma_rough * rough_M2c
    """
    e_abs = data["e_abs"]
    n = len(e_abs)

    print(f"\n  Pair-level exp-M2c model: {n} pairs, sigma-only, HalfStudentT likelihood")

    with pm.Model() as model:
        e_data      = pm.Data("e_abs",  e_abs)
        dist_z      = pm.Data("dist_z",     data["dist_sat_z"])
        turning_z   = pm.Data("turning_z",  data["turning_sat_z"])
        speedup_z   = pm.Data("speedup_z",  data["wm_abs_log_speedup_z"])
        dz_sat_z    = pm.Data("dz_sat_z",   data["dz_sat_z"])
        rough_M2c   = pm.Data("rough_M2c",  data["rough_M2c"])

        # Priors
        nu         = pm.Gamma("nu", alpha=NU_ALPHA, beta=NU_BETA)
        log_sigma0 = pm.Normal("log_sigma0", mu=LOG_SIGMA0_MU, sigma=LOG_SIGMA0_SD)

        gamma_dist      = pm.HalfNormal("gamma_dist",      sigma=GAMMA_SD)
        gamma_turning   = pm.HalfNormal("gamma_turning",   sigma=GAMMA_SD)
        gamma_speedup   = pm.HalfNormal("gamma_speedup",   sigma=GAMMA_SD)
        gamma_dz        = pm.HalfNormal("gamma_dz",        sigma=GAMMA_SD)
        gamma_roughness = pm.HalfNormal("gamma_roughness", sigma=GAMMA_SD)

        log_sigma = (
            log_sigma0
            + gamma_dist      * dist_z
            + gamma_turning   * turning_z
            + gamma_speedup   * speedup_z
            + gamma_dz        * dz_sat_z
            + gamma_roughness * rough_M2c   # min(z_A, z_B), can be negative
        )
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        # HalfStudentT: |e| ~ HalfStudentT(nu, sigma). Sigma is the scale of the
        # underlying two-sided distribution, so downstream P75/P90 multipliers
        # (0.67, 1.28, 2.33 * sigma) remain valid.
        pm.HalfStudentT("obs", nu=nu, sigma=sigma, observed=e_data)

        print(f"  Sampling posterior ({chains} chains x {draws} draws + {tune} tune)...")
        idata = pm.sample(
            draws=draws, tune=tune,
            target_accept=0.95,
            chains=chains, cores=cores,
            progressbar=False,
            random_seed=42,
        )

    return model, idata, data["scalers"]


# ─────────────────────────────────────────────────────────────────
# VARIANCE DECOMPOSITION HELPERS
# (LMG / Shapley — accounts for feature correlations, unlike naive beta^2)
# ─────────────────────────────────────────────────────────────────
def _assemble_features_for_decomp(idata, data):
    """Pack the 5 fitted betas and the standardized feature matrix used by the
    model, so LMG can be computed. Returns (labels, betas, Z) where Z is the
    per-pair matrix of standardized feature values used by the model."""
    labels = [display for _, _, display in NON_ROUGH_FEATURE_CONFIG] + [ROUGH_DISPLAY_NAME]
    betas = np.array([
        float(idata.posterior[gn].values.mean())
        for gn, _, _ in NON_ROUGH_FEATURE_CONFIG
    ] + [float(idata.posterior["gamma_roughness"].values.mean())])
    Z_cols = [data[f"{raw_col}_z"] for _, raw_col, _ in NON_ROUGH_FEATURE_CONFIG]
    Z_cols.append(data["rough_M2c"])
    Z = np.column_stack(Z_cols)
    return labels, betas, Z


def _compute_lmg(betas, Z_matrix):
    """Compute both naive and LMG (Shapley) variance decompositions.
    Returns (naive_arr, lmg_arr, total_var).

    LMG is the average marginal contribution over all K! feature orderings.
    For fixed betas, this equals beta_k * (Cov @ beta)_k, so it sums to
    beta^T Cov beta = actual Var(log_sigma) predictions.
    """
    K = len(betas)
    cov = np.cov(Z_matrix, rowvar=False, ddof=1)
    variances = np.diag(cov)
    naive = betas ** 2 * variances
    total = float(betas @ cov @ betas)

    # Enumerate all 2^K subsets and cache their variance
    subset_var = {}
    for r in range(K + 1):
        for combo in itertools.combinations(range(K), r):
            idx = list(combo)
            subset_var[frozenset(combo)] = (
                float(betas[idx] @ cov[np.ix_(idx, idx)] @ betas[idx])
                if idx else 0.0)

    # Shapley: weighted average of marginal contributions across orderings
    lmg = np.zeros(K)
    for k in range(K):
        for r in range(K):
            w = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
            for combo in itertools.combinations([j for j in range(K) if j != k], r):
                S = frozenset(combo); Sk = S | {k}
                lmg[k] += w * (subset_var[Sk] - subset_var[S])
    return naive, lmg, total


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────
def print_diagnostics(idata, scalers, data):
    print("\n" + "=" * 70)
    print("PAIR-LEVEL exp-M2c MODEL — DIAGNOSTICS (sigma-only, adaptive roughness)")
    print("=" * 70)

    var_names = (["nu", "log_sigma0"] +
                 [gn for gn, _, _ in NON_ROUGH_FEATURE_CONFIG] +
                 ["gamma_roughness"])
    print(az.summary(idata, var_names=var_names))

    nu_samples = idata.posterior["nu"].values.flatten()
    print(f"\n  Degrees of freedom: nu = {nu_samples.mean():.2f} +/- {nu_samples.std():.2f}")
    print(f"    95% CI: [{np.percentile(nu_samples, 2.5):.2f}, "
          f"{np.percentile(nu_samples, 97.5):.2f}]")

    log_sigma0_s = idata.posterior["log_sigma0"].values.flatten()
    sigma0 = np.exp(log_sigma0_s)
    print(f"\n  Baseline sigma at training-mean pair (all z=0):")
    print(f"    sigma0 = exp(log_sigma0) = {sigma0.mean():.4f} ({sigma0.mean()*100:.2f}%)")
    print(f"    Note: this is NOT the same-site floor. Floor is computed via per-mast self-pred.")

    nu_mean = nu_samples.mean()
    if nu_mean > 2:
        ratio = np.sqrt(nu_mean / np.pi) * gamma_func((nu_mean - 1) / 2) / gamma_func(nu_mean / 2)
        print(f"\n  For HalfStudentT with nu={nu_mean:.1f}: E[|e|] = {ratio:.3f} * sigma")

    print("\n" + "=" * 70)
    print("SIGMA DRIVER EFFECTS")
    print("=" * 70)
    print(f"  {'Driver':<45} {'gamma mean':>10} {'Multiplier @ z=+1':>19}")
    print("  " + "-" * 76)
    for gamma_name, _, display in NON_ROUGH_FEATURE_CONFIG:
        g = idata.posterior[gamma_name].values.flatten()
        print(f"  {display:<45} {g.mean():>10.3f} {np.exp(g.mean()):>18.3f}x")
    g = idata.posterior["gamma_roughness"].values.flatten()
    print(f"  {ROUGH_DISPLAY_NAME:<45} {g.mean():>10.3f} {np.exp(g.mean()):>18.3f}x")

    # ── Variance decomposition — BOTH naive AND proper (LMG/Shapley) ────
    # Naive treats features as independent; LMG properly attributes shared variance
    # via Shapley averaging over all K! feature orderings. Features are correlated
    # (turning-speedup up to 0.8), so naive can be MISLEADING. Always cite LMG.
    labels_full, betas_full, Z_matrix = _assemble_features_for_decomp(idata, data)
    naive_arr, lmg_arr, total_var_lmg = _compute_lmg(betas_full, Z_matrix)

    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION — NAIVE  (beta^2 * Var, assumes independence)")
    print("=" * 70)
    naive_sum = float(naive_arr.sum())
    order_n = np.argsort(-naive_arr)
    print(f"  {'Feature':<48} {'Var contrib':>11} {'% of naive':>11}")
    print("  " + "-" * 71)
    for i in order_n:
        print(f"  {labels_full[i]:<48} {naive_arr[i]:>11.5f} "
              f"{100 * naive_arr[i] / naive_sum:>10.1f}%")

    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION — LMG / SHAPLEY  (proper, accounts for correlations)")
    print("=" * 70)
    print(f"  {'Feature':<48} {'Var contrib':>11} {'% of total':>11}")
    print("  " + "-" * 71)
    order_l = np.argsort(-lmg_arr)
    for i in order_l:
        print(f"  {labels_full[i]:<48} {lmg_arr[i]:>11.5f} "
              f"{100 * lmg_arr[i] / total_var_lmg:>10.1f}%")
    print("  " + "-" * 71)
    print(f"  {'TOTAL':<48} {total_var_lmg:>11.5f} {'100.0%':>11}")

    complexity_labels = {"Saturating distance (1-exp(-d/dA))",
                          "Saturating |turning| (1-exp(-t/s))",
                          "WM |log speedup ratio|",
                          "Saturating |dz| (1-exp(-|dz|/40))"}
    rough_idx = labels_full.index(ROUGH_DISPLAY_NAME)
    complexity_share = sum(lmg_arr[i] for i, lab in enumerate(labels_full)
                            if lab in complexity_labels) / total_var_lmg
    print(f"\n  Complexity features (dist+turning+speedup+dz) LMG share: {100*complexity_share:.1f}%")
    print(f"  Roughness feature LMG share:                              "
          f"{100*lmg_arr[rough_idx]/total_var_lmg:.1f}%")
    print(f"  Ratio total_var/naive_sum: {total_var_lmg/naive_sum:.2f}x "
          f"({'features cooperate on average' if total_var_lmg > naive_sum else 'features cancel'})")

    # (kept the old baseline-var report for continuity with the reference script)
    baseline_var = float(idata.posterior["log_sigma0"].values.flatten().var())
    print(f"\n  Baseline (log_sigma0 posterior variance): {baseline_var:.5f}  "
          f"(NOT included in the LMG total — reflects posterior uncertainty, "
          f"not feature-driven variance)")


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPER (used by LOO and self-prediction)
# ─────────────────────────────────────────────────────────────────
def predict_sigma_from_coeffs(row, coeffs, scalers):
    """Compute sigma for a single pair (row) given fitted coeffs and scalers.

    row must contain the raw feature columns:
      dist_sat, turning_sat, wm_abs_log_speedup, dz_sat, rough_sat_A, rough_sat_B
    """
    log_sigma0  = coeffs["log_sigma0"]
    gammas      = coeffs["gammas"]

    log_s = log_sigma0
    for gamma_name, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
        m = scalers[f"{raw_col}_mean"]
        s = scalers[f"{raw_col}_std"]
        z = (row[raw_col] - m) / s
        log_s += gammas[gamma_name] * z

    # Roughness: per-formula z-score then min
    mA = scalers["rough_sat_A_mean"]; sA = scalers["rough_sat_A_std"]
    mB = scalers["rough_sat_B_mean"]; sB = scalers["rough_sat_B_std"]
    z_A = (row["rough_sat_A"] - mA) / sA
    z_B = (row["rough_sat_B"] - mB) / sB
    x_rough = min(z_A, z_B)
    log_s += gammas["gamma_roughness"] * x_rough

    return float(np.exp(log_s)), float(x_rough)


def extract_coeffs(idata):
    return {
        "log_sigma0": float(idata.posterior["log_sigma0"].values.mean()),
        "gammas": {
            gn: float(idata.posterior[gn].values.mean())
            for gn in [gn for gn, _, _ in NON_ROUGH_FEATURE_CONFIG] + ["gamma_roughness"]
        },
    }


# ─────────────────────────────────────────────────────────────────
# LOO CV
# ─────────────────────────────────────────────────────────────────
def leave_one_pair_out_cv(pair_df: pd.DataFrame, draws=LOO_DRAWS, tune=LOO_TUNE,
                          holdout_by: str = "pair"):
    """LOO CV with two holdout strategies:
      "pair" — hold out both A->B and B->A of each physical pair together
      "site" — hold out ALL pairs from the same location
    Per-fold standardization avoids data leakage.
    """
    assert holdout_by in ("pair", "site"), "holdout_by must be 'pair' or 'site'"
    pair_df = pair_df.copy()

    if holdout_by == "pair":
        pair_df["_group_key"] = pair_df["pair_id"].apply(
            lambda x: "__".join(sorted(x.split("__")))
        )
        fold_label = "physical pairs"
    else:
        if "location" not in pair_df.columns or pair_df["location"].isna().all():
            raise ValueError("holdout_by='site' requires a non-empty 'location' column.")
        pair_df["_group_key"] = pair_df["location"]
        fold_label = "sites"

    fold_groups = pair_df.groupby("_group_key")["pair_id"].unique().to_dict()
    all_folds   = list(fold_groups.keys())

    print("=" * 70)
    print(f"LEAVE-ONE-OUT CV  ({len(all_folds)} {fold_label})  "
          f"[holdout_by='{holdout_by}']")
    print("=" * 70)

    results = []
    for i, fold_key in enumerate(all_folds):
        held_out_ids = fold_groups[fold_key]
        print(f"\n[{i+1}/{len(all_folds)}] Holding out: {fold_key} "
              f"({len(held_out_ids)} directions)")

        train_df = pair_df[~pair_df["pair_id"].isin(held_out_ids)].copy()
        test_df  = pair_df[ pair_df["pair_id"].isin(held_out_ids)].copy()
        if len(train_df) < 5:
            print("  Too few training pairs, skipping.")
            continue

        train_data = rebuild_data_dict(train_df)
        try:
            with suppress_output():
                _, idata, _ = fit_model(train_data, draws=draws, tune=tune,
                                         chains=LOO_CHAINS, cores=LOO_CORES)
        except Exception as ex:
            print(f"  Model fitting failed: {ex}")
            continue

        coeffs = extract_coeffs(idata)
        train_scalers = train_data["scalers"]

        for _, row in test_df.iterrows():
            sigma_pred, x_rough = predict_sigma_from_coeffs(row, coeffs, train_scalers)
            actual = abs(row["e_overall"])
            results.append({
                "pair_id":          row["pair_id"],
                "predicted_sigma":  sigma_pred,
                "actual_error":     actual,
                "e_overall_signed": row["e_overall"],
                "location":         row.get("location", ""),
                "abs_dz":           row.get("abs_dz", np.nan),
                "distance_m":       row["distance_m"],
                "rough_M2c_test":   x_rough,
            })
            print(f"  {row['pair_id']}: sigma={sigma_pred:.2%}  "
                  f"actual={actual:.2%}  (signed={row['e_overall']*100:+.1f}%)")

    results_df = pd.DataFrame(results)
    if len(results_df) < 2:
        print("Not enough folds completed.")
        return results_df, np.nan, np.nan, np.nan

    csv_path = OUTPUT_LOO_PAIR_CSV if holdout_by == "pair" else OUTPUT_LOO_SITE_CSV
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Metrics
    r_pearson  = results_df["predicted_sigma"].corr(results_df["actual_error"])
    r_spearman = results_df["predicted_sigma"].corr(results_df["actual_error"], method="spearman")
    bias       = (results_df["predicted_sigma"] - results_df["actual_error"]).mean()

    print("\n" + "=" * 70)
    print(f"LOO RESULTS  [{holdout_by}-level holdout, sigma-only]")
    print("=" * 70)
    print(f"  Pairs tested:              {len(results_df)}")
    print(f"  Pearson r:                 {r_pearson:.3f}")
    print(f"  Spearman rho:              {r_spearman:.3f}")
    print(f"  Bias:                      {bias:+.4f} ({bias*100:+.2f} pp)")
    print(f"  Mean predicted sigma:      {results_df['predicted_sigma'].mean():.2%}")
    print(f"  Mean actual |e|:           {results_df['actual_error'].mean():.2%}")

    # Interactive scatter plot
    hover_data = list(zip(
        results_df["pair_id"], results_df["location"],
        results_df["e_overall_signed"], results_df["predicted_sigma"],
    ))
    hover_tmpl = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
        "Signed e: %{customdata[2]:.1%}<br>"
        "sigma_pred: %{customdata[3]:.2%}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df["predicted_sigma"], y=results_df["actual_error"],
        mode="markers",
        marker=dict(size=10, color=results_df["abs_dz"], colorscale="Viridis",
                    opacity=0.8, colorbar=dict(title="|dz| (m)")),
        customdata=hover_data, hovertemplate=hover_tmpl, showlegend=False,
    ))
    max_val = max(results_df["predicted_sigma"].max(), results_df["actual_error"].max()) * 1.15
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color="red", dash="dash", width=2))
    fig.update_layout(
        title=f"LOO CV — exp-M2c model  [{holdout_by}-level holdout]  "
              f"r={r_pearson:.3f}, rho={r_spearman:.3f}",
        xaxis_title="Predicted sigma", yaxis_title="Actual |overall error|",
        template="plotly_white", height=600, width=900,
    )
    html_path = OUTPUT_LOO_PAIR_HTML if holdout_by == "pair" else OUTPUT_LOO_SITE_HTML
    fig.write_html(html_path)
    print(f"Saved: {html_path}")

    return results_df, r_pearson, r_spearman, bias


# ─────────────────────────────────────────────────────────────────
# PER-MAST SELF-PREDICTION SIGMA
# ─────────────────────────────────────────────────────────────────
def per_mast_self_prediction(pair_df, sector_df, coeffs, scalers):
    """For each mast that appears in the training set, compute the model's
    self-prediction sigma (mast-vs-itself). At self-pair:
      - dist_sat, turning_sat, wm_abs_log_speedup, dz_sat = 0
      - rough_sat_B = 0 (mismatch is trivially zero)
      - rough_sat_A = mast's own weighted-mean absolute roughness (nonzero for rough sites)
    So x_rough = min(z_A_self, z_B_self) where:
      z_A_self = (rough_sat_A_self - mean_A) / std_A   (varies by mast)
      z_B_self = (0 - mean_B) / std_B                    (constant negative)
    For rough masts z_A_self > z_B_self, so min picks z_B_self (constant) → same-value floor.
    For very flat masts z_A_self may be close to z_B_self → floor varies slightly.
    """
    print("\n" + "=" * 70)
    print("PER-MAST SELF-PREDICTION SIGMA  (mast-against-itself)")
    print("=" * 70)

    rows = []
    for mast, mrows in sector_df.groupby(sector_df["pair_id"].apply(lambda x: x.split("__")[1])):
        first_pair = mrows["pair_id"].iloc[0]
        grp = mrows[mrows["pair_id"] == first_pair]
        freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
        if freq.sum() == 0:
            continue
        w = freq / freq.sum()
        rs = pd.to_numeric(grp["rough_speedup_MM_frac"], errors="coerce").values
        valid = ~np.isnan(rs)
        if valid.sum() == 0:
            continue
        rs_self_wmean = float(np.sum(w[valid] * np.abs(rs[valid])) / w[valid].sum())
        rough_sat_A_self = 1.0 - np.exp(-rs_self_wmean / ROUGH_SAT_SCALE)

        # Simulated self-pair row: all non-roughness raw features are 0
        synth_row = {rc: 0.0 for _, rc, _ in NON_ROUGH_FEATURE_CONFIG}
        synth_row["rough_sat_A"] = rough_sat_A_self
        synth_row["rough_sat_B"] = 0.0

        sigma_self, x_rough_self = predict_sigma_from_coeffs(synth_row, coeffs, scalers)
        rows.append({
            "mast":               mast,
            "rs_own_wmean_abs":   rs_self_wmean,
            "rough_sat_A_self":   rough_sat_A_self,
            "x_rough_self":       x_rough_self,
            "sigma_self":         sigma_self,
            "sigma_self_pct":     sigma_self * 100,
        })

    self_df = pd.DataFrame(rows).sort_values("sigma_self_pct").reset_index(drop=True)
    csv_path = OUTPUT_SELF_PRED_CSV
    self_df.to_csv(csv_path, index=False)

    print(f"\n  {'mast':<14}{'|rs|_own':>10}{'rough_sat_A':>13}"
          f"{'x_rough':>10}{'sigma_self':>12}")
    print("  " + "-" * 59)
    for _, r in self_df.iterrows():
        print(f"  {r['mast']:<14}{r['rs_own_wmean_abs']:>10.4f}"
              f"{r['rough_sat_A_self']:>13.4f}{r['x_rough_self']:>+10.4f}"
              f"{r['sigma_self_pct']:>11.3f}%")

    print(f"\n  Summary: N={len(self_df)}  "
          f"min={self_df['sigma_self_pct'].min():.3f}%  "
          f"median={self_df['sigma_self_pct'].median():.3f}%  "
          f"max={self_df['sigma_self_pct'].max():.3f}%")
    print(f"  Spread (max/min): {self_df['sigma_self_pct'].max() / self_df['sigma_self_pct'].min():.2f}x")
    print(f"  Saved: {csv_path}")

    return self_df


# ─────────────────────────────────────────────────────────────────
# PER-PAIR FEATURE CONTRIBUTION DECOMPOSITION
# ─────────────────────────────────────────────────────────────────
def per_pair_feature_contributions(pair_df, coeffs, scalers):
    """For each training pair, break down log(sigma) into per-feature contributions.
    Useful for understanding which features drive predictions on specific pairs
    (e.g., Herzhausen, forest, etc.)."""
    print("\n" + "=" * 70)
    print("PER-PAIR FEATURE CONTRIBUTIONS  (log(sigma) breakdown)")
    print("=" * 70)

    rows = []
    for _, row in pair_df.iterrows():
        contrib = {"pair_id": row["pair_id"], "location": row.get("location", ""),
                   "actual_abs_e": abs(row["e_overall"])}
        log_sigma0 = coeffs["log_sigma0"]
        contrib["log_sigma0"] = log_sigma0
        log_s = log_sigma0

        for gamma_name, raw_col, _ in NON_ROUGH_FEATURE_CONFIG:
            m = scalers[f"{raw_col}_mean"]
            s = scalers[f"{raw_col}_std"]
            z = (row[raw_col] - m) / s
            c = coeffs["gammas"][gamma_name] * z
            contrib[f"contrib_{gamma_name}"] = c
            log_s += c

        mA = scalers["rough_sat_A_mean"]; sA = scalers["rough_sat_A_std"]
        mB = scalers["rough_sat_B_mean"]; sB = scalers["rough_sat_B_std"]
        z_A = (row["rough_sat_A"] - mA) / sA
        z_B = (row["rough_sat_B"] - mB) / sB
        x_rough = min(z_A, z_B)
        c_rough = coeffs["gammas"]["gamma_roughness"] * x_rough
        contrib["z_A"] = z_A
        contrib["z_B"] = z_B
        contrib["x_rough"] = x_rough
        contrib["contrib_gamma_roughness"] = c_rough
        contrib["which_min_wins"] = "z_A" if z_A < z_B else "z_B"
        log_s += c_rough

        contrib["log_sigma_total"] = log_s
        contrib["sigma_predicted"] = float(np.exp(log_s))
        rows.append(contrib)

    contrib_df = pd.DataFrame(rows)
    csv_path = OUTPUT_CONTRIB_CSV
    contrib_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Which formula 'wins' the min per pair:")
    win_counts = contrib_df["which_min_wins"].value_counts()
    for k, v in win_counts.items():
        print(f"    {k}: {v} pairs")

    return contrib_df


# ─────────────────────────────────────────────────────────────────
# CALIBRATION COVERAGE (HalfStudentT version)
# ─────────────────────────────────────────────────────────────────
def calibration_coverage_test(results_df, nu):
    """For a HalfStudentT model, check the empirical coverage of confidence
    intervals derived from sigma. Assumes symmetric error distribution centered
    at zero (which is the bias-dropped assumption)."""
    print("\n" + "=" * 70)
    print("CALIBRATION COVERAGE TEST  (|e| < t_crit * sigma_pred)")
    print("=" * 70)
    print(f"\n  {'Nominal':>10} {'Empirical':>11} {'Assessment'}")
    print(f"  {'-' * 45}")
    for level in [0.50, 0.68, 0.80, 0.90, 0.95]:
        t_crit = stats.t.ppf((1 + level) / 2, df=nu)
        within = (results_df["actual_error"] < t_crit * results_df["predicted_sigma"]).mean()
        if abs(within - level) < 0.05:
            assessment = "Good"
        elif within > level:
            assessment = "Conservative (safe)"
        else:
            assessment = "Under-coverage (risky)"
        print(f"  {level:>9.0%} {within:>10.0%}  {assessment}")


# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
def save_results(idata, scalers, data, loo_pair=None, loo_site=None, self_df=None):
    nu_samples   = idata.posterior["nu"].values.flatten()
    log_sigma0_s = idata.posterior["log_sigma0"].values.flatten()

    model_params = {
        "nu":         float(nu_samples.mean()),
        "nu_std":     float(nu_samples.std()),
        "nu_95ci":    [float(np.percentile(nu_samples, 2.5)),
                        float(np.percentile(nu_samples, 97.5))],
        "log_sigma0": float(log_sigma0_s.mean()),
        "sigma0":     float(np.exp(log_sigma0_s).mean()),
    }

    gamma_posteriors = {}
    variance_decomposition = {}
    for gamma_name, raw_col, display in NON_ROUGH_FEATURE_CONFIG:
        g = idata.posterior[gamma_name].values.flatten()
        model_params[gamma_name] = float(g.mean())
        gamma_posteriors[gamma_name] = {
            "median":           float(np.median(g)),
            "mean":             float(g.mean()),
            "std":              float(g.std()),
            "multiplier_at_z1": float(np.exp(g.mean())),
            "feature":          raw_col,
            "display_name":     display,
        }
        variance_decomposition[display] = float(g.mean() ** 2)

    # Roughness — variance uses empirical var of rough_M2c (min of z-scores)
    g_rough = idata.posterior["gamma_roughness"].values.flatten()
    model_params["gamma_roughness"] = float(g_rough.mean())
    rough_M2c_var = float(np.var(data["rough_M2c"]))
    gamma_posteriors["gamma_roughness"] = {
        "median":                    float(np.median(g_rough)),
        "mean":                      float(g_rough.mean()),
        "std":                       float(g_rough.std()),
        "feature":                   "rough_M2c",
        "display_name":              ROUGH_DISPLAY_NAME,
        "rough_M2c_variance_train":  rough_M2c_var,
    }
    variance_decomposition[ROUGH_DISPLAY_NAME] = float(g_rough.mean() ** 2 * rough_M2c_var)

    baseline_var = float(log_sigma0_s.var())
    total_var = baseline_var + sum(variance_decomposition.values())
    variance_decomposition["_baseline"] = baseline_var
    variance_decomposition["_total"] = total_var
    for k in list(variance_decomposition.keys()):
        if not k.startswith("_"):
            variance_decomposition[f"{k}_pct"] = float(variance_decomposition[k] / total_var * 100)
    variance_decomposition["_baseline_pct"] = float(baseline_var / total_var * 100)

    # ── LMG / Shapley variance decomposition (proper, accounts for correlations) ──
    # Naive attribution above is misleading because features are correlated
    # (turning-speedup ~0.8, speedup-dz ~0.9). LMG averages marginal contributions
    # over all K! orderings — the number to cite when comparing across variants.
    labels_lmg, betas_lmg, Z_lmg = _assemble_features_for_decomp(idata, data)
    naive_arr, lmg_arr, total_var_lmg = _compute_lmg(betas_lmg, Z_lmg)
    variance_decomposition_lmg = {
        "total_var_predictions": float(total_var_lmg),
        "naive_sum":             float(naive_arr.sum()),
        "ratio_actual_over_naive": float(total_var_lmg / max(float(naive_arr.sum()), 1e-12)),
        "per_feature": {
            labels_lmg[i]: {
                "naive_contrib":  float(naive_arr[i]),
                "naive_pct":      float(100 * naive_arr[i] / max(float(naive_arr.sum()), 1e-12)),
                "lmg_contrib":    float(lmg_arr[i]),
                "lmg_pct":        float(100 * lmg_arr[i] / total_var_lmg),
            } for i in range(len(labels_lmg))
        },
    }
    # convenience aggregates for defense
    complexity_labels = {"Saturating distance (1-exp(-d/dA))",
                          "Saturating |turning| (1-exp(-t/s))",
                          "WM |log speedup ratio|",
                          "Saturating |dz| (1-exp(-|dz|/40))"}
    variance_decomposition_lmg["complexity_features_lmg_pct"] = float(sum(
        variance_decomposition_lmg["per_feature"][lab]["lmg_pct"]
        for lab in labels_lmg if lab in complexity_labels))
    variance_decomposition_lmg["roughness_lmg_pct"] = float(
        variance_decomposition_lmg["per_feature"][ROUGH_DISPLAY_NAME]["lmg_pct"])

    def loo_summary(results_df):
        if results_df is None or len(results_df) < 2:
            return None
        r_p = results_df["predicted_sigma"].corr(results_df["actual_error"])
        r_s = results_df["predicted_sigma"].corr(results_df["actual_error"], method="spearman")
        b   = (results_df["predicted_sigma"] - results_df["actual_error"]).mean()
        return {
            "n_pairs":            len(results_df),
            "pearson":            float(r_p),
            "spearman":           float(r_s),
            "bias":               float(b),
            "mean_pred_sigma":    float(results_df["predicted_sigma"].mean()),
            "mean_actual_error":  float(results_df["actual_error"].mean()),
        }

    self_summary = None
    if self_df is not None and len(self_df) > 0:
        self_summary = {
            "n_masts": int(len(self_df)),
            "min_pct":    float(self_df["sigma_self_pct"].min()),
            "median_pct": float(self_df["sigma_self_pct"].median()),
            "max_pct":    float(self_df["sigma_self_pct"].max()),
            "spread":     float(self_df["sigma_self_pct"].max() / self_df["sigma_self_pct"].min()),
        }

    results = {
        "model_type":             "WS_pair_level",
        "variant":                "exp-M2c (adaptive roughness, sigma-only, HalfStudentT)",
        "target":                 "|e_overall| (absolute frequency-weighted signed overall error)",
        "distribution":           "HalfStudentT(nu, sigma)",
        "n_features":             len(NON_ROUGH_FEATURE_CONFIG) + 1,
        "bias_term":              "NONE (mu = 0 — supervisor-driven decision)",
        "roughness_formula":      "M2c: min(z(sat_A), z(sat_B)) with per-formula training z-score",
        "dz_sat_scale":           DZ_SAT_SCALE,
        "rough_sat_scale":        ROUGH_SAT_SCALE,
        "turn_sat_scale":         TURN_SAT_SCALE,
        "features":               [display for _, _, display in NON_ROUGH_FEATURE_CONFIG]
                                     + [ROUGH_DISPLAY_NAME],
        "model_params":           model_params,
        "gamma_posteriors":       gamma_posteriors,
        "variance_decomposition":     variance_decomposition,       # naive (kept for legacy)
        "variance_decomposition_lmg": variance_decomposition_lmg,   # LMG/Shapley (defense-quality)
        "scalers":                {k: float(v) for k, v in scalers.items()},
        "data_summary": {
            "n_pairs":  int(len(data["e_abs"])),
            "mean_abs_e": float(data["e_abs"].mean()),
            "std_abs_e":  float(data["e_abs"].std()),
            "max_abs_e":  float(data["e_abs"].max()),
        },
        "loo_pair":               loo_summary(loo_pair),
        "loo_site":               loo_summary(loo_site),
        "self_prediction":        self_summary,
        "excluded_masts":         EXCLUDED_MASTS,
        "forest_in":              True,
    }

    json_path = OUTPUT_MODEL_JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    idata_path = OUTPUT_IDATA_NC
    idata.to_netcdf(idata_path)
    print(f"Saved: {idata_path}")

    return results


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("WS UNCERTAINTY MODEL — exp-M2c (adaptive roughness, sigma-only)")
    print("=" * 70)
    print("\nModel: |e_overall| ~ HalfStudentT(nu, sigma)")
    print("  log(sigma) = log_sigma0 + sum_k gamma_k * z_k + gamma_rough * min(z_A, z_B)")
    print("  NO BIAS TERM (mu = 0)")
    print("\nFeatures:")
    for i, (_, _, display) in enumerate(NON_ROUGH_FEATURE_CONFIG, 1):
        print(f"  {i}. {display}")
    print(f"  5. {ROUGH_DISPLAY_NAME}")
    print(f"\nSaturation scales: turning={TURN_SAT_SCALE}, roughness={ROUGH_SAT_SCALE}, dz={DZ_SAT_SCALE}")
    print(f"\nExcluded masts ({len(EXCLUDED_MASTS)}): {EXCLUDED_MASTS}")
    print("  Note: forest masts (Hultema, Malarberget) INCLUDED under exp-M2c.")

    print("\nLoading data...")
    df = pd.read_excel(INPUT_PATH)
    print(f"  Raw: {len(df)} rows, {df['pair_id'].nunique()} pairs")

    df = df.drop_duplicates()
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()
    df["mast_A"] = df["pair_id"].apply(lambda x: x.split("__")[0])
    df["mast_B"] = df["pair_id"].apply(lambda x: x.split("__")[1])
    mask = (~df["mast_A"].isin(EXCLUDED_MASTS)) & (~df["mast_B"].isin(EXCLUDED_MASTS))
    df = df[mask].copy()
    print(f"  After filtering: {len(df)} rows, {df['pair_id'].nunique()} pairs")

    print("\nAggregating to pair level (computing formula A and formula B roughness)...")
    data = build_pair_training_data(df)
    pair_df = data["df"]

    print("\nFitting full model...")
    model, idata, scalers = fit_model(data)

    print_diagnostics(idata, scalers, data)

    coeffs = extract_coeffs(idata)

    print("\nComputing per-pair feature contributions...")
    contrib_df = per_pair_feature_contributions(pair_df, coeffs, scalers)

    print("\nComputing per-mast self-prediction sigma...")
    self_df = per_mast_self_prediction(pair_df, df, coeffs, scalers)

    print("\nRunning LOO CV (pair-level holdout)...")
    loo_pair_df, r_pearson, r_spearman, bias = leave_one_pair_out_cv(
        pair_df, holdout_by="pair"
    )
    if len(loo_pair_df) >= 2:
        nu_mean = float(idata.posterior["nu"].values.mean())
        calibration_coverage_test(loo_pair_df, nu_mean)

    print("\nRunning LOO CV (site-level holdout)...")
    loo_site_df, *_ = leave_one_pair_out_cv(pair_df, holdout_by="site")
    if len(loo_site_df) >= 2:
        nu_mean = float(idata.posterior["nu"].values.mean())
        calibration_coverage_test(loo_site_df, nu_mean)

    print("\nSaving results...")
    save_results(idata, scalers, data,
                  loo_pair=loo_pair_df, loo_site=loo_site_df, self_df=self_df)

    print("\n" + "=" * 70)
    print("DONE. See Results/ for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
