"""
Uncertainty Calculator for EY v6 and WS v2 Models

This script:
1. Loads input data (12 rows per pair, one per sector)
2. Computes all derived features
3. Calculates uncertainty for both EY and WS models
4. Provides driver attribution breakdown
5. Exports results to Excel

Usage:
    python uncertainty_calculator.py

    Modify INPUT_FILE path to point to your data.

Input format:
    Excel file with 12 rows per pair (see uncertainty_input_template.xlsx)

Output:
    - Console summary with driver attribution
    - Excel file with detailed results
"""

import numpy as np
import pandas as pd
import os

# -----------------------------
# MODEL PARAMETERS (from training)
# -----------------------------

EY_MODEL = {
    "model_version": "v6",
    "model_params": {
        "log_sigma0": -2.46108833203344,
        "gamma_dist": 0.15902994576749785,
        "gamma_dz_abs": 0.3415109150990276,
        "gamma_speedup_mag": 0.24118973742163344,
        "gamma_RIX_avg": 0.19169276553714668,
        "gamma_sev_frac": 0.20350918884202468,
        "gamma_flip_grad": 0.19489512956287333,
        "gamma_sample_short": 0.21351479579342564,
        "gamma_conc": 0.13060780572340724,
        "gamma_turning_std": 0.14157633591992463,
        "gamma_speedup_diff_std": 0.13265615751552606,
    },
    "scalers": {
        "y_std": 0.11979373378899379,
        "dist_mean": 0.15279633013031305,
        "dist_std": 0.3341765103083774,
        "dz_abs_mean": 36.93134328358209,
        "dz_abs_std": 31.140544234334026,
        "log_speedup_mag_mean": 0.03136620162237417,
        "log_speedup_mag_std": 0.031156197903347557,
        "RIX_avg_mean": 0.8012558968685363,
        "RIX_avg_std": 1.5077635390535946,
        "sev_frac_mean": 0.07767248844406058,
        "sev_frac_std": 0.13328646192153168,
        "flip_grad_mean": 0.0011899104315394727,
        "flip_grad_std": 0.0023904389688447496,
        "sample_short_mean": 1.3502056972940035,
        "sample_short_std": 0.7065034040817063,
        "conc_mean": 0.4143355872327952,
        "conc_std": 0.14150702843987353,
        "turning_std_mean": 0.5298251772958649,
        "turning_std_std": 0.6438294941132616,
        "speedup_diff_std_mean": 0.023578222440941262,
        "speedup_diff_std_std": 0.027228083355914468,
    },
}

WS_MODEL = {
    "model_version": "v2",
    "model_params": {
        "log_sigma0": -1.5487820161160644,
        "gamma_dist": 0.41411818699879305,
        "gamma_dz_abs": 0.10745336280046718,
        "gamma_speedup_mag": 0.10777859909309483,
        "gamma_RIX_avg": 0.10590639211860972,
        "gamma_sev_frac": 0.1286538085192583,
        "gamma_flip_grad": 0.21244008423241428,
        "gamma_sample_short": 0.1467377562934069,
        "gamma_conc": 0.13923250419281688,
        "gamma_turning_std": 0.13089398957875062,
        "gamma_speedup_diff_std": 0.15414944295147257,
    },
    "scalers": {
        "y_std": 0.06432326735278521,
        "dist_mean": 0.15279633013031305,
        "dist_std": 0.3341765103083774,
        "dz_abs_mean": 36.93134328358209,
        "dz_abs_std": 31.140544234334026,
        "log_speedup_mag_mean": 0.03136620162237417,
        "log_speedup_mag_std": 0.031156197903347557,
        "RIX_avg_mean": 0.8012558968685363,
        "RIX_avg_std": 1.5077635390535946,
        "sev_frac_mean": 0.07767248844406058,
        "sev_frac_std": 0.13328646192153168,
        "flip_grad_mean": 0.0011899104315394727,
        "flip_grad_std": 0.0023904389688447496,
        "sample_short_mean": 1.3502056972940035,
        "sample_short_std": 0.7065034040817063,
        "conc_mean": 0.4143355872327952,
        "conc_std": 0.14150702843987353,
        "turning_std_mean": 0.5298251772958649,
        "turning_std_std": 0.6438294941132616,
        "speedup_diff_std_mean": 0.023578222440941262,
        "speedup_diff_std_std": 0.027228083355914468,
    },
}

# After line ~95 (after WS_MODEL definition), add:

# -----------------------------
# DIAGNOSTIC: Check baseline σ
# -----------------------------
print("=" * 50)
print("MODEL BASELINE CHECK")
print("=" * 50)

ey_base_sigma = np.exp(EY_MODEL["model_params"]["log_sigma0"]) * EY_MODEL["scalers"]["y_std"]
ws_base_sigma = np.exp(WS_MODEL["model_params"]["log_sigma0"]) * WS_MODEL["scalers"]["y_std"]

print(f"EY baseline σ (all z=0): {ey_base_sigma*100:.2f}%")
print(f"WS baseline σ (all z=0): {ws_base_sigma*100:.2f}%")
print(f"EY training data std (y_std): {EY_MODEL['scalers']['y_std']*100:.2f}%")
print(f"WS training data std (y_std): {WS_MODEL['scalers']['y_std']*100:.2f}%")
print("=" * 50 + "\n")



# Reference sample count for sample_shortfall calculation
# This is log(max_samples) from training data
LOG_MAX_SAMPLES = np.log(112470 + 1)  # From training data

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}


# Driver display names
DRIVER_NAMES = {
    "dist": "Distance",
    "dz_abs": "|dz| (Height Diff)",
    "speedup_mag": "|log_speedup|",
    "RIX_avg": "RIX_avg (Terrain)",
    "sev_frac": "Severity Fraction",
    "flip_grad": "Flip × Gradient",
    "sample_short": "Sample Shortfall",
    "conc": "Concentration Ratio",
    "turning_std": "Turning Std",
    "speedup_diff_std": "Speedup Diff Std",
}


# -----------------------------
# FEATURE COMPUTATION
# -----------------------------

def compute_features_for_pair(pair_data: pd.DataFrame) -> dict:
    """
    Compute all model features from 12-sector input data for one pair.

    Parameters:
    -----------
    pair_data : pd.DataFrame
        DataFrame with 12 rows (one per sector) for a single pair

    Returns:
    --------
    dict with all computed features
    """

    # Validate input
    if len(pair_data) != 12:
        raise ValueError(f"Expected 12 rows (sectors), got {len(pair_data)}")

    # Sort by sector
    pair_data = pair_data.copy()
    pair_data["sector_idx"] = pair_data["sector_name"].map(SECTOR_TO_IDX)
    pair_data = pair_data.sort_values("sector_idx")

    # Extract arrays
    weights = pair_data["weight_energy_predicted"].values
    weights = weights / weights.sum()  # Normalize

    speedup_WTG = pair_data["overall_speedup_WTG_factor"].values
    speedup_MM = pair_data["overall_speedup_MM_factor"].values
    RIX_avg = pair_data["RIX_avg_0.3_sector"].values
    dRIX_03 = pair_data["dRIX_0.3_sector"].values
    dRIX_0501 = pair_data["dRIX_0.0501_sector"].values
    turning_deg = pair_data["d_turning_deg"].values
    flip_fraction = pair_data["flip_fraction_weighted_by_predicted_power"].values
    sample_count = pair_data["Sample_count_pred"].values

    # Pair-level values (same across all sectors)
    dz = pair_data["dz"].iloc[0]
    distance_m = pair_data["distance_m"].iloc[0]
    distance_A = pair_data["distance_A"].iloc[0]
    distance_B = pair_data["distance_B"].iloc[0]

    # =========================================
    # FEATURE 1: Distance Score
    # =========================================
    denom = max(distance_B - distance_A, 1e-6)
    dist_excess = max(distance_m - distance_A, 0.0)
    dist_score = dist_excess / denom

    # =========================================
    # FEATURE 2: |dz|
    # =========================================
    dz_abs = abs(dz)

    # =========================================
    # FEATURE 3: |log_speedup| (energy-weighted)
    # =========================================
    log_speedup_sector = np.log(speedup_WTG / np.clip(speedup_MM, 1e-6, None))
    log_speedup_mag = np.sum(weights * np.abs(log_speedup_sector))

    # =========================================
    # FEATURE 4: RIX_avg (energy-weighted)
    # =========================================
    RIX_avg_pair = np.sum(weights * RIX_avg)

    # =========================================
    # FEATURE 5: Severity Fraction (energy-weighted)
    # =========================================
    dRIX_mild = dRIX_0501 - dRIX_03
    abs_severe = np.abs(dRIX_03)
    abs_mild = np.abs(dRIX_mild)

    severity_fraction_sector = abs_severe / (abs_severe + abs_mild + 1e-6)
    both_zero = (abs_severe < 1e-6) & (abs_mild < 1e-6)
    severity_fraction_sector[both_zero] = 0.0

    sev_frac_pair = np.sum(weights * severity_fraction_sector)

    # =========================================
    # FEATURE 6: Flip × Gradient (energy-weighted)
    # =========================================
    speedup_gradient_sector = np.zeros(12)

    for s in range(12):
        turn = turning_deg[s]
        current_speedup = speedup_WTG[s]

        if turn > 0:
            adjacent_idx = (s + 1) % 12
        elif turn < 0:
            adjacent_idx = (s - 1) % 12
        else:
            speedup_gradient_sector[s] = 0.0
            continue

        adjacent_speedup = speedup_WTG[adjacent_idx]

        log_current = np.log(max(current_speedup, 1e-6))
        log_adjacent = np.log(max(adjacent_speedup, 1e-6))
        speedup_gradient_sector[s] = abs(log_current - log_adjacent)

    flip_x_gradient_sector = flip_fraction * speedup_gradient_sector
    flip_grad_pair = np.sum(weights * flip_x_gradient_sector)

    # =========================================
    # FEATURE 7: Sample Shortfall
    # =========================================
    total_samples = np.sum(sample_count)
    log_total_samples = np.log(total_samples + 1)
    sample_shortfall = LOG_MAX_SAMPLES - log_total_samples
    sample_shortfall = max(sample_shortfall, 0.0)  # Can't be negative

    # =========================================
    # FEATURE 8: Concentration Ratio
    # =========================================
    hhi = np.sum(weights ** 2)
    effective_sectors = 1.0 / hhi
    concentration_ratio = (12 - effective_sectors) / 11

    # =========================================
    # FEATURE 9: Turning Std
    # =========================================
    turning_abs = np.abs(turning_deg)
    turning_std = np.std(turning_abs)

    # =========================================
    # FEATURE 10: Speedup Diff Std
    # =========================================
    speedup_diff = speedup_MM - speedup_WTG
    speedup_diff_std = np.std(speedup_diff)

    return {
        "pair_id": pair_data["pair_id"].iloc[0],
        "dz": dz,
        "distance_m": distance_m,
        # Raw features
        "dist_score": dist_score,
        "dz_abs": dz_abs,
        "log_speedup_mag": log_speedup_mag,
        "RIX_avg": RIX_avg_pair,
        "sev_frac": sev_frac_pair,
        "flip_grad": flip_grad_pair,
        "sample_short": sample_shortfall,
        "conc": concentration_ratio,
        "turning_std": turning_std,
        "speedup_diff_std": speedup_diff_std,
        # Additional info
        "total_samples": total_samples,
        "effective_sectors": effective_sectors,
    }


def standardize_features(features: dict, scalers: dict) -> dict:
    """
    Standardize features using training scalers (z-scoring).
    """
    z_scores = {}

    feature_map = {
        "dist_score": ("dist_mean", "dist_std"),
        "dz_abs": ("dz_abs_mean", "dz_abs_std"),
        "log_speedup_mag": ("log_speedup_mag_mean", "log_speedup_mag_std"),
        "RIX_avg": ("RIX_avg_mean", "RIX_avg_std"),
        "sev_frac": ("sev_frac_mean", "sev_frac_std"),
        "flip_grad": ("flip_grad_mean", "flip_grad_std"),
        "sample_short": ("sample_short_mean", "sample_short_std"),
        "conc": ("conc_mean", "conc_std"),
        "turning_std": ("turning_std_mean", "turning_std_std"),
        "speedup_diff_std": ("speedup_diff_std_mean", "speedup_diff_std_std"),
    }

    for feature_name, (mean_key, std_key) in feature_map.items():
        raw_value = features[feature_name]
        mean = scalers[mean_key]
        std = scalers[std_key]
        z_scores[feature_name] = (raw_value - mean) / std

    return z_scores


# -----------------------------
# UNCERTAINTY CALCULATION
# -----------------------------

def calculate_uncertainty(features: dict, model: dict, verbose: bool = False) -> dict:
    """
    Calculate uncertainty using the specified model.

    Returns dict with:
    - sigma_standardized: σ in standardized units
    - sigma_original: σ in original units (EY deviation or WS deviation)
    - range_68: 68% confidence interval
    - range_95: 95% confidence interval
    - driver_contributions: breakdown by driver
    """

    params = model["model_params"]
    scalers = model["scalers"]

    # Standardize features
    z_scores = standardize_features(features, scalers)

    if verbose:
        print(f"\n  Z-scores for {features.get('pair_id', 'unknown')}:")
        for name, z in z_scores.items():
            print(f"    {name:<20}: {z:+.2f}")

    # Calculate log-sigma components
    log_sigma0 = params["log_sigma0"]

    driver_keys = [
        "dist", "dz_abs", "speedup_mag", "RIX_avg", "sev_frac",
        "flip_grad", "sample_short", "conc", "turning_std", "speedup_diff_std"
    ]

    # Map feature names to gamma keys
    feature_to_gamma = {
        "dist_score": "gamma_dist",
        "dz_abs": "gamma_dz_abs",
        "log_speedup_mag": "gamma_speedup_mag",
        "RIX_avg": "gamma_RIX_avg",
        "sev_frac": "gamma_sev_frac",
        "flip_grad": "gamma_flip_grad",
        "sample_short": "gamma_sample_short",
        "conc": "gamma_conc",
        "turning_std": "gamma_turning_std",
        "speedup_diff_std": "gamma_speedup_diff_std",
    }

    # Calculate contributions
    contributions = {}
    log_sigma_total = log_sigma0

    for feature_name, gamma_key in feature_to_gamma.items():
        gamma = params[gamma_key]
        z = z_scores[feature_name]
        contribution = gamma * z

        # Short name for display
        short_name = feature_name.replace("_score", "").replace("log_speedup_mag", "speedup_mag")
        contributions[short_name] = {
            "gamma": gamma,
            "z_score": z,
            "log_sigma_contribution": contribution,
            "raw_value": features[feature_name],
        }

        log_sigma_total += contribution

    # Calculate sigma
    sigma_standardized = np.exp(log_sigma_total)
    sigma_original = sigma_standardized * scalers["y_std"]

    # Confidence intervals
    range_68 = sigma_original
    range_95 = 1.96 * sigma_original

    # Calculate percentage contribution of each driver
    # (relative to deviation from baseline)
    total_deviation = log_sigma_total - log_sigma0

    for name, contrib in contributions.items():
        if abs(total_deviation) > 1e-6:
            contrib["pct_contribution"] = (contrib["log_sigma_contribution"] / total_deviation) * 100
        else:
            contrib["pct_contribution"] = 0.0

    return {
        "log_sigma0": log_sigma0,
        "log_sigma_total": log_sigma_total,
        "sigma_standardized": sigma_standardized,
        "sigma_original": sigma_original,
        "sigma_pct": sigma_original * 100,
        "range_68_pct": range_68 * 100,
        "range_95_pct": range_95 * 100,
        "driver_contributions": contributions,

    }


# -----------------------------
# OUTPUT AND DISPLAY
# -----------------------------

def print_pair_results(pair_id: str, features: dict, ey_result: dict, ws_result: dict):
    """Print detailed results for one pair."""

    print("\n" + "=" * 80)
    print(f"PAIR: {pair_id}")
    print("=" * 80)

    # =========================================
    # RAW FEATURES
    # =========================================
    print("\n" + "-" * 40)
    print("INPUT FEATURES")
    print("-" * 40)

    print(f"\n  Pair-level:")
    print(f"    dz (height diff):     {features['dz']:+.1f} m")
    print(f"    |dz|:                 {features['dz_abs']:.1f} m")
    print(f"    Distance:             {features['distance_m']:.0f} m")
    print(f"    Distance score:       {features['dist_score']:.3f}")

    print(f"\n  Computed features:")
    print(f"    |log_speedup|:        {features['log_speedup_mag']:.4f}")
    print(f"    RIX_avg:              {features['RIX_avg']:.2f}")
    print(f"    Severity fraction:    {features['sev_frac']:.3f}")
    print(f"    Flip × Gradient:      {features['flip_grad']:.5f}")
    print(f"    Sample shortfall:     {features['sample_short']:.2f}")
    print(f"    Concentration ratio:  {features['conc']:.3f}")
    print(f"    Turning std:          {features['turning_std']:.3f}")
    print(f"    Speedup diff std:     {features['speedup_diff_std']:.4f}")

    print(f"\n  Additional info:")
    print(f"    Total samples:        {features['total_samples']:.0f}")
    print(f"    Effective sectors:    {features['effective_sectors']:.1f}")

    # =========================================
    # EY UNCERTAINTY
    # =========================================
    print("\n" + "-" * 40)
    print("ENERGY YIELD UNCERTAINTY (EY v6)")
    print("-" * 40)

    print(f"\n  σ (EY deviation):       {ey_result['sigma_pct']:.2f}%")
    print(f"  68% CI:                 ±{ey_result['range_68_pct']:.2f}%")
    print(f"  95% CI:                 ±{ey_result['range_95_pct']:.2f}%")

    print(f"\n  Driver Attribution (sorted by contribution):")
    print(f"  {'Driver':<25} {'γ':<8} {'z-score':<10} {'Contrib':<12} {'%':<8}")
    print("  " + "-" * 65)

    # Sort by absolute contribution
    sorted_drivers = sorted(
        ey_result["driver_contributions"].items(),
        key=lambda x: abs(x[1]["log_sigma_contribution"]),
        reverse=True
    )

    for name, contrib in sorted_drivers:
        display_name = DRIVER_NAMES.get(name, name)
        sign = "+" if contrib["log_sigma_contribution"] >= 0 else ""
        print(f"  {display_name:<25} {contrib['gamma']:.3f}   {contrib['z_score']:+.2f}      "
              f"{sign}{contrib['log_sigma_contribution']:.3f}       {contrib['pct_contribution']:+.1f}%")

    # =========================================
    # WS UNCERTAINTY
    # =========================================
    print("\n" + "-" * 40)
    print("WINDSPEED UNCERTAINTY (WS v2)")
    print("-" * 40)

    print(f"\n  σ (WS deviation):       {ws_result['sigma_pct']:.2f}%")
    print(f"  68% CI:                 ±{ws_result['range_68_pct']:.2f}%")
    print(f"  95% CI:                 ±{ws_result['range_95_pct']:.2f}%")

    print(f"\n  Driver Attribution (sorted by contribution):")
    print(f"  {'Driver':<25} {'γ':<8} {'z-score':<10} {'Contrib':<12} {'%':<8}")
    print("  " + "-" * 65)

    sorted_drivers = sorted(
        ws_result["driver_contributions"].items(),
        key=lambda x: abs(x[1]["log_sigma_contribution"]),
        reverse=True
    )

    for name, contrib in sorted_drivers:
        display_name = DRIVER_NAMES.get(name, name)
        sign = "+" if contrib["log_sigma_contribution"] >= 0 else ""
        print(f"  {display_name:<25} {contrib['gamma']:.3f}   {contrib['z_score']:+.2f}      "
              f"{sign}{contrib['log_sigma_contribution']:.3f}       {contrib['pct_contribution']:+.1f}%")

    # =========================================
    # COMPARISON
    # =========================================
    print("\n" + "-" * 40)
    print("COMPARISON: EY vs WS")
    print("-" * 40)

    print(f"\n  {'Metric':<25} {'EY':<15} {'WS':<15}")
    print("  " + "-" * 55)
    print(f"  {'σ (deviation)':<25} {ey_result['sigma_pct']:.2f}%{'':<10} {ws_result['sigma_pct']:.2f}%")
    print(f"  {'95% CI':<25} ±{ey_result['range_95_pct']:.2f}%{'':<8} ±{ws_result['range_95_pct']:.2f}%")

    # Top drivers comparison
    ey_top = max(ey_result["driver_contributions"].items(),
                 key=lambda x: abs(x[1]["log_sigma_contribution"]))
    ws_top = max(ws_result["driver_contributions"].items(),
                 key=lambda x: abs(x[1]["log_sigma_contribution"]))

    print(f"\n  Top driver (EY):        {DRIVER_NAMES.get(ey_top[0], ey_top[0])}")
    print(f"  Top driver (WS):        {DRIVER_NAMES.get(ws_top[0], ws_top[0])}")


def generate_recommendations(features: dict, ey_result: dict, ws_result: dict) -> list:
    """Generate actionable recommendations based on high-contributing drivers."""

    recommendations = []

    # Check sample shortfall
    if features["sample_short"] > 1.0:
        recommendations.append(
            f"📊 CAMPAIGN LENGTH: Sample shortfall is high ({features['sample_short']:.2f}). "
            f"Consider extending the measurement campaign to reduce epistemic uncertainty."
        )

    # Check height difference
    if features["dz_abs"] > 50:
        recommendations.append(
            f"📏 HEIGHT DIFFERENCE: Large height difference ({features['dz_abs']:.0f}m). "
            f"This is the main driver of EY uncertainty. Consider mast placement closer to hub height."
        )

    # Check distance
    if features["dist_score"] > 0.5:
        recommendations.append(
            f"📍 DISTANCE: Sites are relatively far apart (score: {features['dist_score']:.2f}). "
            f"This is the main driver of WS uncertainty. Consider using a closer reference mast."
        )

    # Check terrain severity
    if features["RIX_avg"] > 5.0:
        recommendations.append(
            f"⛰️ TERRAIN: High terrain severity (RIX_avg: {features['RIX_avg']:.1f}%). "
            f"Complex terrain increases model uncertainty. Consider CFD analysis."
        )

    # Check concentration
    if features["conc"] > 0.6:
        recommendations.append(
            f"🎯 ENERGY CONCENTRATION: Energy is concentrated in few sectors (ratio: {features['conc']:.2f}). "
            f"Prediction errors have less chance to average out. Check dominant sectors carefully."
        )

    return recommendations


def export_results_to_excel(all_results: list, output_path: str):
    """Export detailed results to Excel."""

    # Summary sheet
    summary_data = []

    for result in all_results:
        summary_data.append({
            "pair_id": result["pair_id"],
            "dz_m": result["features"]["dz"],
            "distance_m": result["features"]["distance_m"],
            "EY_sigma_pct": result["ey"]["sigma_pct"],
            "EY_95CI_pct": result["ey"]["range_95_pct"],
            "WS_sigma_pct": result["ws"]["sigma_pct"],
            "WS_95CI_pct": result["ws"]["range_95_pct"],
            "EY_top_driver": max(result["ey"]["driver_contributions"].items(),
                                 key=lambda x: abs(x[1]["log_sigma_contribution"]))[0],
            "WS_top_driver": max(result["ws"]["driver_contributions"].items(),
                                 key=lambda x: abs(x[1]["log_sigma_contribution"]))[0],
        })

    summary_df = pd.DataFrame(summary_data)

    # Features sheet
    features_data = []
    for result in all_results:
        row = {"pair_id": result["pair_id"]}
        row.update(result["features"])
        features_data.append(row)

    features_df = pd.DataFrame(features_data)

    # EY driver contributions sheet
    ey_contrib_data = []
    for result in all_results:
        row = {"pair_id": result["pair_id"]}
        for driver, contrib in result["ey"]["driver_contributions"].items():
            row[f"{driver}_z"] = contrib["z_score"]
            row[f"{driver}_contrib"] = contrib["log_sigma_contribution"]
        ey_contrib_data.append(row)

    ey_contrib_df = pd.DataFrame(ey_contrib_data)

    # WS driver contributions sheet
    ws_contrib_data = []
    for result in all_results:
        row = {"pair_id": result["pair_id"]}
        for driver, contrib in result["ws"]["driver_contributions"].items():
            row[f"{driver}_z"] = contrib["z_score"]
            row[f"{driver}_contrib"] = contrib["log_sigma_contribution"]
        ws_contrib_data.append(row)

    ws_contrib_df = pd.DataFrame(ws_contrib_data)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        features_df.to_excel(writer, sheet_name='Features', index=False)
        ey_contrib_df.to_excel(writer, sheet_name='EY_Contributions', index=False)
        ws_contrib_df.to_excel(writer, sheet_name='WS_Contributions', index=False)

    print(f"\nExported results to: {output_path}")


# -----------------------------
# MAIN FUNCTION
# -----------------------------

def process_input_file(input_path: str, output_path: str = None):
    """
    Process input file and calculate uncertainties for all pairs.

    Parameters:
    -----------
    input_path : str
        Path to input Excel file (12 rows per pair)
    output_path : str, optional
        Path for output Excel file. If None, creates one in same directory.
    """

    print("=" * 80)
    print("UNCERTAINTY CALCULATOR - EY v6 / WS v2")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_excel(input_path)

    # Validate required columns
    required_columns = [
        "pair_id", "sector_name", "dz", "distance_m", "distance_A", "distance_B",
        "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
        "RIX_avg_0.3_sector", "dRIX_0.3_sector", "dRIX_0.0501_sector",
        "d_turning_deg", "flip_fraction_weighted_by_predicted_power",
        "weight_energy_predicted", "Sample_count_pred",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Process each pair
    pairs = df.groupby("pair_id")
    n_pairs = len(pairs)

    print(f"Found {n_pairs} pairs")

    all_results = []

    for pair_id, pair_data in pairs:
        # Validate sector count
        if len(pair_data) != 12:
            print(f"  WARNING: {pair_id} has {len(pair_data)} sectors (expected 12), skipping")
            continue

        # Compute features
        features = compute_features_for_pair(pair_data)

        # Calculate uncertainties
        ey_result = calculate_uncertainty(features, EY_MODEL, verbose=True)
        ws_result = calculate_uncertainty(features, WS_MODEL, verbose=False)

        # Print results
        print_pair_results(pair_id, features, ey_result, ws_result)

        # Generate recommendations
        recommendations = generate_recommendations(features, ey_result, ws_result)
        if recommendations:
            print("\n  RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"    {rec}")

        # Store for export
        all_results.append({
            "pair_id": pair_id,
            "features": features,
            "ey": ey_result,
            "ws": ws_result,
            "recommendations": recommendations,
        })

    # Export to Excel
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_uncertainty_results.xlsx"

    export_results_to_excel(all_results, output_path)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL PAIRS")
    print("=" * 80)

    ey_sigmas = [r["ey"]["sigma_pct"] for r in all_results]
    ws_sigmas = [r["ws"]["sigma_pct"] for r in all_results]

    print(f"\n  {'Metric':<30} {'EY':<15} {'WS':<15}")
    print("  " + "-" * 60)
    print(f"  {'Mean σ':<30} {np.mean(ey_sigmas):.2f}%{'':<10} {np.mean(ws_sigmas):.2f}%")
    print(f"  {'Min σ':<30} {np.min(ey_sigmas):.2f}%{'':<10} {np.min(ws_sigmas):.2f}%")
    print(f"  {'Max σ':<30} {np.max(ey_sigmas):.2f}%{'':<10} {np.max(ws_sigmas):.2f}%")
    print(f"  {'Median σ':<30} {np.median(ey_sigmas):.2f}%{'':<10} {np.median(ws_sigmas):.2f}%")

    return all_results


# -----------------------------
# COMMAND LINE INTERFACE
# -----------------------------

if __name__ == "__main__":
    # =========================================
    # CONFIGURATION - MODIFY THIS
    # =========================================

    INPUT_FILE = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Uncertainty calculator\CEA\CEA_uncertainty_input 2.xlsx"  #todo: Input file goes here.
    OUTPUT_FILE = None  # Set to None to auto-generate, or specify path #todo: Output path to be added here.

    # =========================================
    # RUN
    # =========================================

    try:
        results = process_input_file(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Please update the INPUT_FILE path in the script.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise