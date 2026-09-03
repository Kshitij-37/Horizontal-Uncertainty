"""
Roughness Formula Comparison via LOO CV
----------------------------------------
Tests 7 roughness formula variants via pair-level leave-one-out cross-validation,
comparing Pearson r, Spearman rho, and bias side-by-side.

This tells us which roughness formula best balances physical correctness
with predictive power.

Options tested:
  N: Averaging the value of absolutes, and gating it with saturated distance
  0: log (rough_WTG/rough_MM)  — original log ratio (not used in final model)
  A: |(WTG+MM)/2|          — current model (absolute of average)
  B: |WTG - MM|            — mismatch (difference)
  C: |(WTG+MM)/2*(WTG-MM)| — product (avg * diff)
  D: no roughness feature  — 4-feature model
  E: two features          — avg_abs + difference (both, let model decide)
  F: (|WTG|+|MM|)/2        — average of absolutes (no sign cancellation)
  G: max(|WTG|, |MM|)      — maximum absolute value
  H: "|WTG-MM| + sign-conflict overlap",
  I: "|WTG-MM| * 2 if sign flip",
  J: "|WTG-MM| * (1 + common_abs / ROUGH_SAT_SCALE)",
  K: "Normalized mismatch = |WTG-MM| / (avg_abs + eps)",
  L: "Two features: common_abs + diff_abs/2",
  M: "Three features: common_abs + diff_abs + sign_conflict",
  T: "Magnitude-mismatch: ||rs_WTG| - |rs_MM|| — sign-independent complement to log_speedup",


Usage: python roughness_formula_comparison.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import importlib.util
import importlib.machinery
import pymc as pm
import arviz as az
import contextlib
from scipy import stats

# ─── IMPORT FINAL MODEL MODULE ─────────────────────────────────
_base_dir = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..",
    "Bayesian_approach", "Final model"))
# Accept either the no-extension file or the .py-extension version
_candidates = [
    os.path.join(_base_dir, "ws_uncertainty_model_pairlevel_final"),
    os.path.join(_base_dir, "ws_uncertainty_model_pairlevel_final.py"),
]
_final_model_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])

_loader = importlib.machinery.SourceFileLoader("pmod", _final_model_path)
_spec = importlib.util.spec_from_file_location("pmod", _final_model_path, loader=_loader)
pmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmod)

# ─── CONFIG ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FAST_DRAWS = 500
FAST_TUNE = 500
FAST_CHAINS = 2

ROUGH_SAT_SCALE = pmod.ROUGH_SAT_SCALE  # 0.01


# ─── HELPERS ────────────────────────────────────────────────────
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout/stderr during MCMC sampling."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def load_sector_data():
    """Load and filter sector-level data (same as final model)."""
    df = pd.read_excel(pmod.INPUT_PATH)
    df = df.drop_duplicates()
    df = df[df["sector_name"].isin(pmod.SECTOR_LABELS)].copy()

    # Apply exclusions
    df["mast_A"] = df["pair_id"].apply(lambda x: x.split("__")[0])
    df["mast_B"] = df["pair_id"].apply(lambda x: x.split("__")[1])
    mask = (~df["mast_A"].isin(pmod.EXCLUDED_MASTS)) & (~df["mast_B"].isin(pmod.EXCLUDED_MASTS))
    df = df[mask].copy()

    # Use windPRO detailed values where available
    upgrade_map = {
        "d_turning_deg":              "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor":  "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac":     "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac":      "rough_speedup_MM_frac_new",
    }
    for old_col, new_col in upgrade_map.items():
        if new_col in df.columns:
            upgrade_mask = df[new_col].notna()
            if upgrade_mask.sum() > 0:
                df.loc[upgrade_mask, old_col] = df.loc[upgrade_mask, new_col]

    return df


def build_pair_data_with_formula(sector_df, formula_key, sat_scale=ROUGH_SAT_SCALE):
    """
    Build pair-level DataFrame with a specific roughness formula.

    Special cases:
      D: no roughness feature
      E: two roughness features (common_abs + diff_abs)
      L: two roughness features (common_abs + diff_abs/2)
      M: three roughness features (common_abs + diff_abs + sign_conflict)

    New options:
      N: distance-gated A
      H: sign-aware dissimilarity
      I: difference with binary sign-flip penalty
      J: severity-weighted mismatch
      K: normalized mismatch
      L: common/differential decomposition
      M: common/differential/sign-conflict decomposition
    """
    d = sector_df.copy()
    eps = 1e-6

    # Sector-level signed error
    d["e_sector"] = (
        (d["Mean_windspeed_predicted"] - d["Mean_windspeed_self"]) /
        d["Mean_windspeed_self"]
    )

    # Sector-level |log speedup ratio|
    mask_spd = (
        d["overall_speedup_WTG_factor"].notna() &
        d["overall_speedup_MM_factor"].notna() &
        (d["overall_speedup_WTG_factor"] > 0) &
        (d["overall_speedup_MM_factor"] > 0)
    )
    d["abs_log_speedup_ratio"] = np.nan
    d.loc[mask_spd, "abs_log_speedup_ratio"] = np.abs(
        np.log(
            d.loc[mask_spd, "overall_speedup_WTG_factor"] /
            d.loc[mask_spd, "overall_speedup_MM_factor"]
        )
    )

    # Aggregate to pair level
    pair_rows = []
    for pair_id, grp in d.groupby("pair_id"):
        w_energy = grp["weight_energy_predicted"].values
        if w_energy.sum() == 0:
            continue

        # Sample count weights
        w_pred = pd.to_numeric(grp["Sample_count_pred"], errors="coerce").fillna(0).values
        w_self = pd.to_numeric(grp["Sample_count_self"], errors="coerce").fillna(0).values
        if w_pred.sum() == 0 or w_self.sum() == 0:
            w_pred = np.ones(len(grp))
            w_self = np.ones(len(grp))
        w_pred_norm = w_pred / w_pred.sum()
        w_self_norm = w_self / w_self.sum()

        # Target
        WS_pred_overall = float(np.sum(w_pred_norm * grp["Mean_windspeed_predicted"].values))
        WS_self_overall = float(np.sum(w_self_norm * grp["Mean_windspeed_self"].values))
        e_overall = (WS_pred_overall - WS_self_overall) / WS_self_overall

        # Feature 1: saturating distance
        distance_m = grp["distance_m"].iloc[0]
        distance_A = grp["distance_A"].iloc[0]
        if pd.isna(distance_A) or distance_A <= 0:
            continue
        dist_sat = float(1.0 - np.exp(-distance_m / distance_A))

        # Sector weights: MM Weibull frequency
        w_freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
        w_freq_norm = w_freq / w_freq.sum() if w_freq.sum() > 0 else w_pred_norm

        # Feature 2: weighted mean |turning| → saturated
        abs_turn = grp["d_turning_deg"].abs().values
        wm_abs_turning = float(np.sum(w_freq_norm * abs_turn))
        turning_sat = float(1.0 - np.exp(-wm_abs_turning / pmod.TURN_SAT_SCALE))

        # Feature 3: weighted mean |log speedup ratio|
        abs_log_spd = grp["abs_log_speedup_ratio"].fillna(0).values
        wm_abs_log_speedup = float(np.sum(w_freq_norm * abs_log_spd))

        # Roughness sector arrays
        rs_WTG = pd.to_numeric(grp["rough_speedup_WTG_frac"], errors="coerce").values \
            if "rough_speedup_WTG_frac" in grp.columns else np.full(len(grp), np.nan)
        rs_MM = pd.to_numeric(grp["rough_speedup_MM_frac"], errors="coerce").values \
            if "rough_speedup_MM_frac" in grp.columns else np.full(len(grp), np.nan)

        # Common helper terms
        common_signed = (rs_WTG + rs_MM) / 2
        common_abs = (np.abs(rs_WTG) + np.abs(rs_MM)) / 2
        diff_abs = np.abs(rs_WTG - rs_MM)
        diff_abs_half = diff_abs / 2
        sign_flip = (rs_WTG * rs_MM < 0).astype(float)
        sign_conflict = np.where(
            rs_WTG * rs_MM < 0,
            np.minimum(np.abs(rs_WTG), np.abs(rs_MM)),
            0.0
        )

        # Formula-specific sector-level roughness quantity
        abs_rough = None
        rough_diff = None
        rough_sign_conflict = None
        use_distance_gate = False

        if formula_key == "0":
            ratio = (np.abs(rs_WTG) + eps) / (np.abs(rs_MM) + eps)
            abs_rough = np.abs(np.log(ratio))

        elif formula_key == "A":
            abs_rough = np.abs(common_signed)

        elif formula_key == "B":
            abs_rough = diff_abs

        elif formula_key == "C":
            abs_rough = np.abs(common_signed * (rs_WTG - rs_MM))

        elif formula_key == "D":
            abs_rough = None  # dropped later

        elif formula_key == "E":
            # Existing two-feature option:
            #   primary = avg of absolutes
            #   secondary = absolute difference
            abs_rough = common_abs
            rough_diff = diff_abs

        elif formula_key == "F":
            abs_rough = common_abs

        elif formula_key == "G":
            abs_rough = np.maximum(np.abs(rs_WTG), np.abs(rs_MM))

        elif formula_key == "H":
            # Sign-aware dissimilarity:
            # mismatch + overlapping opposite-sign magnitude
            abs_rough = diff_abs + sign_conflict

        elif formula_key == "I":
            # Difference doubled when signs disagree
            abs_rough = diff_abs * (1.0 + sign_flip)

        elif formula_key == "J":
            # Severity-weighted mismatch
            abs_rough = diff_abs * (1.0 + common_abs / sat_scale)

        elif formula_key == "K":
            # Relative / normalized mismatch
            abs_rough = diff_abs / (common_abs + eps)
            abs_rough = np.minimum(abs_rough, 3.0)  # clip to avoid near-zero blow-up

        elif formula_key == "L":
            # Two features: common severity + differential magnitude
            abs_rough = common_abs
            rough_diff = diff_abs_half

        elif formula_key == "M":
            # Three features: common severity + mismatch + sign-conflict
            abs_rough = common_abs
            rough_diff = diff_abs
            rough_sign_conflict = sign_conflict
        
        elif formula_key == "N":
                    # Distance-gated A
                    abs_rough = np.abs(common_signed)
                    use_distance_gate = True

        elif formula_key == "P":
            # Proximity-gated A: gate with fixed 300m exponential scale
            abs_rough = np.abs(common_signed)
            # Gate applied after aggregation (pair-level, see below)

        elif formula_key == "Q":
            # Proximity-gated A: gate with fixed 500m exponential scale
            abs_rough = np.abs(common_signed)
            # Gate applied after aggregation (pair-level, see below)

        elif formula_key == "R":
            # Proximity-gated A: gate with fixed 600m exponential scale
            abs_rough = np.abs(common_signed)
            # Gate applied after aggregation (pair-level, see below)

        elif formula_key == "S":
                    # Proximity-gated A: gate with fixed 700m exponential scale
                    abs_rough = np.abs(common_signed)
                    # Gate applied after aggregation (pair-level, see below)

        elif formula_key == "T":
            # Magnitude-mismatch: ||rs_WTG| - |rs_MM||
            # Measures "how differently BIG the roughness corrections are"
            # (ignoring direction, since log_speedup already captures directional info
            # via the overall speedup ratio which includes roughness effects).
            # Zero for same-magnitude pairs (regardless of sign), so vanishes at self-pair.
            abs_rough = np.abs(np.abs(rs_WTG) - np.abs(rs_MM))

        else:
            raise ValueError(f"Unknown formula_key: {formula_key}")

        # Compute weighted primary roughness feature
        if formula_key == "D":
            wm_abs_roughness = np.nan
            roughness_sat = np.nan
        else:
            valid_rough = np.isfinite(abs_rough)
            if valid_rough.sum() > 0:
                wm_abs_roughness_base = float(
                    np.sum(w_freq_norm[valid_rough] * abs_rough[valid_rough]) /
                    w_freq_norm[valid_rough].sum()
                )
            else:
                wm_abs_roughness_base = np.nan

            if np.isfinite(wm_abs_roughness_base):
                if use_distance_gate:
                    wm_abs_roughness = wm_abs_roughness_base * dist_sat
                elif formula_key == "P":
                    proximity_gate = 1.0 - np.exp(-distance_m / 300.0)
                    wm_abs_roughness = wm_abs_roughness_base * proximity_gate
                elif formula_key == "Q":
                    proximity_gate = 1.0 - np.exp(-distance_m / 500.0)
                    wm_abs_roughness = wm_abs_roughness_base * proximity_gate
                elif formula_key == "R":
                    proximity_gate = 1.0 - np.exp(-distance_m / 600.0)
                    wm_abs_roughness = wm_abs_roughness_base * proximity_gate
                elif formula_key == "S":
                    proximity_gate = 1.0 - np.exp(-distance_m / 700.0)
                    wm_abs_roughness = wm_abs_roughness_base * proximity_gate

                else:
                    wm_abs_roughness = wm_abs_roughness_base
            else:
                wm_abs_roughness = np.nan

            if np.isfinite(wm_abs_roughness):
                roughness_sat = float(1.0 - np.exp(-wm_abs_roughness / sat_scale))
            else:
                roughness_sat = np.nan

        # Optional extra roughness features
        wm_rough_diff = np.nan
        rough_diff_sat = np.nan
        wm_rough_sign_conflict = np.nan
        rough_sign_conflict_sat = np.nan

        if formula_key in {"E", "L", "M"} and rough_diff is not None:
            valid_diff = np.isfinite(rough_diff)
            if valid_diff.sum() > 0:
                wm_rough_diff = float(
                    np.sum(w_freq_norm[valid_diff] * rough_diff[valid_diff]) /
                    w_freq_norm[valid_diff].sum()
                )
            if np.isfinite(wm_rough_diff):
                rough_diff_sat = float(1.0 - np.exp(-wm_rough_diff / sat_scale))

        if formula_key == "M" and rough_sign_conflict is not None:
            valid_sc = np.isfinite(rough_sign_conflict)
            if valid_sc.sum() > 0:
                wm_rough_sign_conflict = float(
                    np.sum(w_freq_norm[valid_sc] * rough_sign_conflict[valid_sc]) /
                    w_freq_norm[valid_sc].sum()
                )
            if np.isfinite(wm_rough_sign_conflict):
                rough_sign_conflict_sat = float(
                    1.0 - np.exp(-wm_rough_sign_conflict / sat_scale)
                )

        # Feature 5: saturating |dz|
        dz = float(grp["dz"].iloc[0]) if "dz" in grp.columns else np.nan
        dz_sat = float(1.0 - np.exp(-abs(dz) / pmod.DZ_SAT_SCALE)) if not pd.isna(dz) else np.nan

        row = {
            "pair_id":             pair_id,
            "e_overall":           e_overall,
            "WS_pred_overall":     WS_pred_overall,
            "WS_self_overall":     WS_self_overall,
            "dist_sat":            dist_sat,
            "wm_abs_turning":      wm_abs_turning,
            "turning_sat":         turning_sat,
            "wm_abs_log_speedup":  wm_abs_log_speedup,
            "wm_abs_roughness":    wm_abs_roughness,
            "roughness_sat":       roughness_sat,
            "dz":                  dz,
            "dz_sat":              dz_sat,
            "abs_dz":              abs(dz) if not pd.isna(dz) else np.nan,
            "location":            grp["location"].iloc[0] if "location" in grp.columns else "",
            "distance_m":          distance_m,
        }

        if formula_key in {"E", "L", "M"}:
            row["wm_rough_diff"] = wm_rough_diff
            row["rough_diff_sat"] = rough_diff_sat

        if formula_key == "M":
            row["wm_rough_sign_conflict"] = wm_rough_sign_conflict
            row["rough_sign_conflict_sat"] = rough_sign_conflict_sat

        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    # Define feature config for this formula
    if formula_key == "D":
        feature_config = [fc for fc in pmod.FEATURE_CONFIG if fc[0] != "gamma_roughness"]

    elif formula_key in {"E", "L"}:
        feature_config = list(pmod.FEATURE_CONFIG) + [
            ("gamma_rough_diff", "rough_diff_sat", "Saturating roughness differential"),
        ]

    elif formula_key == "M":
        feature_config = list(pmod.FEATURE_CONFIG) + [
            ("gamma_rough_diff", "rough_diff_sat", "Saturating roughness differential"),
            ("gamma_rough_sign_conflict", "rough_sign_conflict_sat", "Saturating sign-conflict overlap"),
        ]

    else:
        feature_config = list(pmod.FEATURE_CONFIG)

    feature_cols = [raw_col for _, raw_col, _ in feature_config] + ["dz", "e_overall"]

    # Drop NaN rows
    nan_mask = pair_df[feature_cols].isna().any(axis=1)
    if nan_mask.any():
        pair_df = pair_df[~nan_mask].copy()

    return pair_df, feature_config


def rebuild_data_dict_custom(pair_df, feature_config):
    """Rebuild data dict (z-scoring) for a custom feature config."""
    pair_df = pair_df.copy()
    feature_cols = [raw_col for _, raw_col, _ in feature_config] + ["dz", "e_overall"]
    nan_mask = pair_df[feature_cols].isna().any(axis=1)
    if nan_mask.any():
        pair_df = pair_df[~nan_mask].copy()

    scalers = {}
    for _, raw_col, _ in feature_config:
        mean = pair_df[raw_col].mean()
        std = pair_df[raw_col].std()
        std = std if std > 0 else 1.0
        pair_df[f"{raw_col}_z"] = (pair_df[raw_col] - mean) / std
        scalers[f"{raw_col}_mean"] = mean
        scalers[f"{raw_col}_std"] = std

    dz_mean = pair_df["dz"].mean()
    dz_std = pair_df["dz"].std()
    dz_std = dz_std if dz_std > 0 else 1.0
    pair_df["dz_z"] = (pair_df["dz"] - dz_mean) / dz_std
    scalers["dz_mean"] = dz_mean
    scalers["dz_std"] = dz_std

    result = {
        "e": pair_df["e_overall"].values,
        "dz_z": pair_df["dz_z"].values,
        "pair_ids": pair_df["pair_id"].values,
        "scalers": scalers,
        "df": pair_df,
    }
    for _, raw_col, _ in feature_config:
        result[f"{raw_col}_z"] = pair_df[f"{raw_col}_z"].values

    return result


def fit_model_custom(data, feature_config, draws=FAST_DRAWS, tune=FAST_TUNE):
    """Fit a model with a custom feature config."""
    e = data["e"]
    n = len(e)

    with pm.Model() as model:
        e_data = pm.Data("e", e)
        dz_z = pm.Data("dz_z", data["dz_z"])

        # Feature data
        feature_data = {}
        for gamma_name, raw_col, _ in feature_config:
            feature_data[gamma_name] = pm.Data(f"{gamma_name}_z", data[f"{raw_col}_z"])

        # Priors
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=-3.9, sigma=0.5)
        beta_dz = pm.Normal("beta_dz", mu=0, sigma=0.05)

        gamma_vars = {}
        for gamma_name, _, _ in feature_config:
            gamma_vars[gamma_name] = pm.HalfNormal(gamma_name, sigma=0.3)

        # log(sigma) linear model
        log_sigma = log_sigma0
        for gamma_name in gamma_vars:
            log_sigma = log_sigma + gamma_vars[gamma_name] * feature_data[gamma_name]

        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        mu = pm.Deterministic("mu", beta_dz * dz_z)

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=e_data)

        idata = pm.sample(
            draws=draws, tune=tune,
            target_accept=0.95,
            chains=FAST_CHAINS, cores=FAST_CHAINS,
            random_seed=42,
        )

    return model, idata, data["scalers"]


def run_loo_cv_custom(pair_df, feature_config, draws=FAST_DRAWS, tune=FAST_TUNE):
    """
    Run pair-level LOO CV with a custom feature config.
    Returns (results_df, pearson_r, spearman_rho, bias).
    """
    pair_df = pair_df.copy()
    pair_df["_group_key"] = pair_df["pair_id"].apply(
        lambda x: "__".join(sorted(x.split("__")))
    )

    fold_groups = pair_df.groupby("_group_key")["pair_id"].unique().to_dict()
    all_folds = list(fold_groups.keys())

    results = []
    for i, fold_key in enumerate(all_folds):
        held_out_ids = fold_groups[fold_key]

        train_df = pair_df[~pair_df["pair_id"].isin(held_out_ids)].copy()
        test_df = pair_df[pair_df["pair_id"].isin(held_out_ids)].copy()

        if len(train_df) < 5:
            continue

        train_data = rebuild_data_dict_custom(train_df, feature_config)

        try:
            with suppress_output():
                _, idata, _ = fit_model_custom(train_data, feature_config, draws=draws, tune=tune)
        except Exception as ex:
            print(f"  Fold {i+1} failed: {ex}")
            continue

        # Extract posterior means
        log_sigma0_mean = float(idata.posterior["log_sigma0"].values.mean())
        beta_dz_mean = float(idata.posterior["beta_dz"].values.mean())
        gammas = {}
        for gn, _, _ in feature_config:
            if gn in idata.posterior:
                gammas[gn] = float(idata.posterior[gn].values.mean())

        train_scalers = train_data["scalers"]

        def standardize(col_name, value):
            m = train_scalers.get(f"{col_name}_mean", 0)
            s = train_scalers.get(f"{col_name}_std", 1)
            return (value - m) / s

        # Predict for held-out pairs
        for _, row in test_df.iterrows():
            log_sigma_pred = log_sigma0_mean
            for gn, raw_col, _ in feature_config:
                if gn in gammas:
                    z_val = standardize(raw_col, row[raw_col])
                    log_sigma_pred += gammas[gn] * z_val

            sigma_pred = float(np.exp(log_sigma_pred))
            dz_z_pred = standardize("dz", row["dz"])
            mu_pred = float(beta_dz_mean * dz_z_pred)

            results.append({
                "pair_id": row["pair_id"],
                "predicted_sigma": sigma_pred,
                "predicted_mu": mu_pred,
                "actual_error": abs(row["e_overall"]),
                "e_overall_signed": row["e_overall"],
            })

    results_df = pd.DataFrame(results)
    if len(results_df) < 2:
        return results_df, np.nan, np.nan, np.nan

    r_pearson = results_df["predicted_sigma"].corr(results_df["actual_error"])
    r_spearman = results_df["predicted_sigma"].corr(results_df["actual_error"], method="spearman")
    bias = (results_df["predicted_sigma"] - results_df["actual_error"]).mean()

    return results_df, r_pearson, r_spearman, bias


# ─── MAIN ───────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ROUGHNESS FORMULA COMPARISON VIA LOO CV")
    print("=" * 70)

    print("\nLoading sector-level data...")
    sector_df = load_sector_data()
    print(f"  Sector-level rows: {len(sector_df)}")
    print(f"  Pairs: {sector_df['pair_id'].nunique()}")

    # Define formula options (P and Q first for quick comparison)
    formula_options = {
        "P": "Proximity-gated A: |(WTG+MM)/2| * (1-exp(-d/300))",
        "Q": "Proximity-gated A: |(WTG+MM)/2| * (1-exp(-d/500))",
        "R": "Proximity-gated A: |(WTG+MM)/2| * (1-exp(-d/600))",
        "S": "Proximity-gated A: |(WTG+MM)/2| * (1-exp(-d/700))",
        "A": "|(WTG+MM)/2|  (current model)",
        # --- Uncomment below to re-run all options ---
        # "N": "Distance-gated A: |(WTG+MM)/2| * dist_sat",
        # "0": "np.abs(np.log(abs(rs_WTG)/abs(rs_MM)))",
        # "B": "|WTG - MM|  (difference/mismatch)",
        # "C": "|(WTG+MM)/2 * (WTG-MM)|  (product)",
        # "D": "No roughness feature (4-feature model)",
        # "E": "Two features: avg_abs + difference",
        # "F": "(|WTG|+|MM|)/2  (avg of absolutes)",
        # "G": "max(|WTG|, |MM|)  (maximum)",
        # "H": "|WTG-MM| + sign-conflict overlap",
        # "I": "|WTG-MM| * 2 if sign flip",
        # "J": "|WTG-MM| * (1 + common_abs / ROUGH_SAT_SCALE)",
        # "K": "Normalized mismatch = |WTG-MM| / (avg_abs + eps)",
        # "L": "Two features: common_abs + diff_abs/2",
        # "M": "Three features: common_abs + diff_abs + sign_conflict",
    }


    all_results = []

    for key, description in formula_options.items():
        print(f"\n{'=' * 70}")
        print(f"OPTION {key}: {description}")
        print(f"{'=' * 70}")

        t0 = time.time()

        # Build pair data with this formula
        pair_df, feature_config = build_pair_data_with_formula(sector_df, key)
        n_pairs = len(pair_df)
        print(f"  Pairs: {n_pairs}")

        # Print raw roughness feature range (before z-scoring)
        if key != "D":
            rough_vals = pair_df["roughness_sat"].values
            raw_rough = pair_df["wm_abs_roughness"].values
            print(f"  Raw roughness range (before sat): [{np.nanmin(raw_rough):.6f}, {np.nanmax(raw_rough):.6f}]")
            print(f"  Saturated roughness range: [{np.nanmin(rough_vals):.4f}, {np.nanmax(rough_vals):.4f}]")
            print(f"  Saturation scale used: {ROUGH_SAT_SCALE}")
        if key == "E":
            diff_vals = pair_df["rough_diff_sat"].values
            raw_diff = pair_df["wm_rough_diff"].values
            print(f"  Raw rough_diff range: [{np.nanmin(raw_diff):.6f}, {np.nanmax(raw_diff):.6f}]")
            print(f"  Saturated rough_diff range: [{np.nanmin(diff_vals):.4f}, {np.nanmax(diff_vals):.4f}]")

        # Run LOO CV
        print(f"\n  Running LOO CV ({FAST_DRAWS} draws, {FAST_TUNE} tune, {FAST_CHAINS} chains)...")
        results_df, r_pearson, r_spearman, bias = run_loo_cv_custom(
            pair_df, feature_config, draws=FAST_DRAWS, tune=FAST_TUNE
        )

        elapsed = time.time() - t0
        print(f"\n  RESULTS ({elapsed/60:.1f} min):")
        print(f"    Pearson r:   {r_pearson:.4f}")
        print(f"    Spearman rho: {r_spearman:.4f}")
        print(f"    Bias:        {bias:+.4f} ({bias*100:+.2f} pp)")
        print(f"    N predictions: {len(results_df)}")

        all_results.append({
            "option": key,
            "description": description,
            "n_pairs": n_pairs,
            "pearson_r": r_pearson,
            "spearman_rho": r_spearman,
            "bias": bias,
            "elapsed_min": elapsed / 60,
        })

    # ─── SUMMARY TABLE ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values("pearson_r", ascending=False)

    print(f"\n  {'Opt':<4} {'Description':<42} {'Pearson':>8} {'Spearman':>9} {'Bias':>8} {'Time':>6}")
    print("  " + "-" * 80)
    for _, row in summary_df.iterrows():
        print(f"  {row['option']:<4} {row['description']:<42} "
              f"{row['pearson_r']:>8.4f} {row['spearman_rho']:>9.4f} "
              f"{row['bias']:>+8.4f} {row['elapsed_min']:>5.1f}m")

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "roughness_formula_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Best option
    best = summary_df.iloc[0]
    print(f"\n  BEST BY PEARSON: Option {best['option']} — {best['description']}")
    print(f"    Pearson={best['pearson_r']:.4f}  Spearman={best['spearman_rho']:.4f}  Bias={best['bias']:+.4f}")


if __name__ == "__main__":
    main()
