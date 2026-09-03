"""
Uncertainty Calculator: Translate Real-World Inputs to EY Uncertainty

This script:
1. Loads your data to compute the scaling parameters
2. Takes real-world inputs (dz, distance, speedup, RIX, severity)
3. Outputs the predicted uncertainty range

Usage:
    python uncertainty_calculator.py

Then modify the EXAMPLE_INPUTS section for your specific case.
"""

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# MODEL PARAMETERS (from v2 results)
# These are the posterior means from your model run
MODEL_PARAMS = {
    "log_sigma0": -2.529,  # Baseline log-sigma
    "gamma_dist": 0.172,  # Distance effect
    "gamma_dz_abs": 0.576,  # |dz| effect (DOMINANT)
    "gamma_speedup_mag": 0.440,  # |speedup| effect
    "gamma_RIX_avg": 0.164,  # RIX_avg effect
    "gamma_sev_frac": 0.212,  # Severity fraction effect
    "y_std": 0.121,  # Will be computed from data
}


def compute_scalers_from_data(df):
    """
    Compute the scaling parameters (mean, std) for each input variable.
    These are needed to convert real-world values to z-scores.
    """

    print("=" * 70)
    print("COMPUTING SCALERS FROM TRAINING DATA")
    print("=" * 70)

    # Filter to valid rows (same as model training)
    needed = [
        "pair_id", "EY_deviation_sector_frac", "weight_energy",
        "weight_energy_predicted", "distance_m", "distance_A", "distance_B",
        "dz", "dRIX_0.3_sector", "dRIX_0.0501_sector", "RIX_avg_0.3_sector",
        "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
    ]

    d = df.dropna(subset=needed).copy()

    # Aggregate to pair level (same as model)

    # 1. EY deviation (target)
    d["contrib"] = d["weight_energy"] * d["EY_deviation_sector_frac"]
    y_by_pair = d.groupby("pair_id")["contrib"].sum()

    # 2. Distance score
    pair_dist = d.groupby("pair_id").agg(
        distance_m=("distance_m", "first"),
        distance_A=("distance_A", "first"),
        distance_B=("distance_B", "first"),
    )
    denom = np.clip(pair_dist["distance_B"] - pair_dist["distance_A"], 1e-6, None)
    dist_excess = np.maximum(pair_dist["distance_m"] - pair_dist["distance_A"], 0.0)
    dist_score = dist_excess / denom

    # 3. dz
    dz_by_pair = d.groupby("pair_id")["dz"].first()
    dz_abs = np.abs(dz_by_pair)

    # 4. Speedup (energy-weighted)
    d["log_speedup"] = np.log(d["overall_speedup_WTG_factor"] /
                              np.clip(d["overall_speedup_MM_factor"], 1e-6, None))
    d["w_log_speedup_mag"] = d["weight_energy_predicted"] * np.abs(d["log_speedup"])
    speedup_mag_by_pair = d.groupby("pair_id")["w_log_speedup_mag"].sum()

    # 5. RIX_avg (energy-weighted)
    d["w_RIX_avg"] = d["weight_energy_predicted"] * d["RIX_avg_0.3_sector"]
    RIX_avg_by_pair = d.groupby("pair_id")["w_RIX_avg"].sum()

    # 6. Severity fraction (energy-weighted)
    d["dRIX_mild"] = d["dRIX_0.0501_sector"] - d["dRIX_0.3_sector"]
    abs_severe = np.abs(d["dRIX_0.3_sector"])
    abs_mild = np.abs(d["dRIX_mild"])
    d["severity_fraction"] = abs_severe / (abs_severe + abs_mild + 1e-6)
    d.loc[(abs_severe < 1e-6) & (abs_mild < 1e-6), "severity_fraction"] = 0.0
    d["w_sev_frac"] = d["weight_energy_predicted"] * d["severity_fraction"]
    sev_frac_by_pair = d.groupby("pair_id")["w_sev_frac"].sum()

    # Compute scalers (mean and std for each variable)
    scalers = {
        "y_mean": y_by_pair.mean(),
        "y_std": y_by_pair.std(),

        "dist_mean": dist_score.mean(),
        "dist_std": dist_score.std(),

        "dz_abs_mean": dz_abs.mean(),
        "dz_abs_std": dz_abs.std(),

        "speedup_mag_mean": speedup_mag_by_pair.mean(),
        "speedup_mag_std": speedup_mag_by_pair.std(),

        "RIX_avg_mean": RIX_avg_by_pair.mean(),
        "RIX_avg_std": RIX_avg_by_pair.std(),

        "sev_frac_mean": sev_frac_by_pair.mean(),
        "sev_frac_std": sev_frac_by_pair.std(),

        # Also store raw distance stats for reference
        "dist_m_mean": pair_dist["distance_m"].mean(),
        "dist_A_mean": pair_dist["distance_A"].mean(),
        "dist_B_mean": pair_dist["distance_B"].mean(),
    }

    print("\nScaling parameters (from training data):")
    print("-" * 50)
    print(f"\nTarget (EY deviation):")
    print(f"  mean = {scalers['y_mean']:.4f}, std = {scalers['y_std']:.4f}")

    print(f"\nDistance score:")
    print(f"  mean = {scalers['dist_mean']:.4f}, std = {scalers['dist_std']:.4f}")
    print(f"  (Raw distance: mean = {scalers['dist_m_mean']:.0f}m)")
    print(f"  (distance_A mean = {scalers['dist_A_mean']:.0f}m, distance_B mean = {scalers['dist_B_mean']:.0f}m)")

    print(f"\n|dz| (height difference):")
    print(f"  mean = {scalers['dz_abs_mean']:.1f}m, std = {scalers['dz_abs_std']:.1f}m")

    print(f"\n|log(speedup ratio)|:")
    print(f"  mean = {scalers['speedup_mag_mean']:.4f}, std = {scalers['speedup_mag_std']:.4f}")

    print(f"\nRIX_avg_0.3:")
    print(f"  mean = {scalers['RIX_avg_mean']:.2f}, std = {scalers['RIX_avg_std']:.2f}")

    print(f"\nSeverity fraction:")
    print(f"  mean = {scalers['sev_frac_mean']:.3f}, std = {scalers['sev_frac_std']:.3f}")

    return scalers


def calculate_uncertainty(
        dz_m: float,
        distance_m: float,
        distance_A: float,
        distance_B: float,
        log_speedup_mag: float,
        RIX_avg: float,
        severity_fraction: float,
        scalers: dict,
        params: dict = MODEL_PARAMS,
        verbose: bool = True
):
    """
    Calculate predicted uncertainty given real-world inputs.

    Parameters:
    -----------
    dz_m : float
        Height difference in meters (can be positive or negative, we use |dz|)
    distance_m : float
        Distance between locations in meters
    distance_A : float
        Lower bound of acceptable distance (terrain-adjusted)
    distance_B : float
        Upper bound of acceptable distance (terrain-adjusted)
    log_speedup_mag : float
        |log(WTG_speedup / MM_speedup)|
    RIX_avg : float
        Average RIX_0.3 value (average of WTG and MM)
    severity_fraction : float
        Fraction of complexity difference in severe terrain (0-1)
    scalers : dict
        Scaling parameters from training data
    params : dict
        Model parameters (gammas)
    verbose : bool
        Print detailed calculation steps

    Returns:
    --------
    dict with sigma and uncertainty range
    """

    if verbose:
        print("\n" + "=" * 70)
        print("UNCERTAINTY CALCULATION")
        print("=" * 70)
        print("\nStep 1: Raw inputs")
        print("-" * 50)
        print(f"  |dz| = {abs(dz_m):.1f} m")
        print(f"  distance = {distance_m:.0f} m")
        print(f"  distance_A = {distance_A:.0f} m, distance_B = {distance_B:.0f} m")
        print(f"  |log(speedup ratio)| = {log_speedup_mag:.4f}")
        print(f"  RIX_avg = {RIX_avg:.2f}")
        print(f"  severity_fraction = {severity_fraction:.3f}")

    # Step 2: Convert to model units

    # Distance score
    denom = max(distance_B - distance_A, 1e-6)
    dist_excess = max(distance_m - distance_A, 0.0)
    dist_score = dist_excess / denom

    # |dz|
    dz_abs = abs(dz_m)

    if verbose:
        print(f"\nStep 2: Convert to model units")
        print("-" * 50)
        print(f"  distance_score = max({distance_m} - {distance_A}, 0) / ({distance_B} - {distance_A})")
        print(f"                 = {dist_score:.4f}")

    # Step 3: Standardize (convert to z-scores)

    z_dist = (dist_score - scalers["dist_mean"]) / scalers["dist_std"]
    z_dz_abs = (dz_abs - scalers["dz_abs_mean"]) / scalers["dz_abs_std"]
    z_speedup = (log_speedup_mag - scalers["speedup_mag_mean"]) / scalers["speedup_mag_std"]
    z_RIX = (RIX_avg - scalers["RIX_avg_mean"]) / scalers["RIX_avg_std"]
    z_sev = (severity_fraction - scalers["sev_frac_mean"]) / scalers["sev_frac_std"]

    if verbose:
        print(f"\nStep 3: Standardize to z-scores")
        print("-" * 50)
        print(
            f"  z_dist     = ({dist_score:.4f} - {scalers['dist_mean']:.4f}) / {scalers['dist_std']:.4f} = {z_dist:+.2f}")
        print(
            f"  z_|dz|     = ({dz_abs:.1f} - {scalers['dz_abs_mean']:.1f}) / {scalers['dz_abs_std']:.1f} = {z_dz_abs:+.2f}")
        print(
            f"  z_speedup  = ({log_speedup_mag:.4f} - {scalers['speedup_mag_mean']:.4f}) / {scalers['speedup_mag_std']:.4f} = {z_speedup:+.2f}")
        print(
            f"  z_RIX      = ({RIX_avg:.2f} - {scalers['RIX_avg_mean']:.2f}) / {scalers['RIX_avg_std']:.2f} = {z_RIX:+.2f}")
        print(
            f"  z_sev      = ({severity_fraction:.3f} - {scalers['sev_frac_mean']:.3f}) / {scalers['sev_frac_std']:.3f} = {z_sev:+.2f}")

    # Step 4: Apply sigma model

    log_sigma0 = params["log_sigma0"]
    gamma_dist = params["gamma_dist"]
    gamma_dz_abs = params["gamma_dz_abs"]
    gamma_speedup = params["gamma_speedup_mag"]
    gamma_RIX = params["gamma_RIX_avg"]
    gamma_sev = params["gamma_sev_frac"]

    log_sigma = (log_sigma0
                 + gamma_dist * z_dist
                 + gamma_dz_abs * z_dz_abs
                 + gamma_speedup * z_speedup
                 + gamma_RIX * z_RIX
                 + gamma_sev * z_sev)

    sigma_standardized = np.exp(log_sigma)

    if verbose:
        print(f"\nStep 4: Apply sigma model")
        print("-" * 50)
        print(f"  log(σ) = log_σ₀ + γ_dist×z_dist + γ_|dz|×z_|dz| + γ_speedup×z_speedup + γ_RIX×z_RIX + γ_sev×z_sev")
        print(
            f"         = {log_sigma0:.3f} + {gamma_dist:.3f}×({z_dist:+.2f}) + {gamma_dz_abs:.3f}×({z_dz_abs:+.2f}) + {gamma_speedup:.3f}×({z_speedup:+.2f}) + {gamma_RIX:.3f}×({z_RIX:+.2f}) + {gamma_sev:.3f}×({z_sev:+.2f})")
        print(
            f"         = {log_sigma0:.3f} + ({gamma_dist * z_dist:+.3f}) + ({gamma_dz_abs * z_dz_abs:+.3f}) + ({gamma_speedup * z_speedup:+.3f}) + ({gamma_RIX * z_RIX:+.3f}) + ({gamma_sev * z_sev:+.3f})")
        print(f"         = {log_sigma:.3f}")
        print(f"  σ (standardized) = exp({log_sigma:.3f}) = {sigma_standardized:.4f}")

    # Step 5: Convert back to original units

    y_std = scalers["y_std"]
    sigma_original = sigma_standardized * y_std

    # 95% confidence interval (approximately ±1.96σ)
    range_95 = 1.96 * sigma_original

    # 68% confidence interval (approximately ±1σ)
    range_68 = sigma_original

    if verbose:
        print(f"\nStep 5: Convert to original units (EY deviation)")
        print("-" * 50)
        print(f"  σ (original) = σ (standardized) × y_std")
        print(f"               = {sigma_standardized:.4f} × {y_std:.4f}")
        print(f"               = {sigma_original:.4f}")

        print(f"\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n  σ (EY deviation) = {sigma_original:.4f} = {sigma_original * 100:.2f}%")
        print(f"\n  68% confidence interval: ±{range_68 * 100:.2f}%")
        print(f"  95% confidence interval: ±{range_95 * 100:.2f}%")
        print(f"\n  Interpretation: The true EY is likely within ±{range_95 * 100:.1f}% of the")
        print(f"                  cross-predicted value (95% confidence)")

    return {
        "sigma_standardized": sigma_standardized,
        "sigma_original": sigma_original,
        "range_68_pct": range_68 * 100,
        "range_95_pct": range_95 * 100,
        "z_scores": {
            "z_dist": z_dist,
            "z_dz_abs": z_dz_abs,
            "z_speedup": z_speedup,
            "z_RIX": z_RIX,
            "z_sev": z_sev,
        }
    }


def calculate_uncertainty_simple(
        dz_m: float,
        log_speedup_mag: float,
        RIX_avg: float,
        severity_fraction: float,
        distance_score: float = None,
        scalers: dict = None,
        params: dict = MODEL_PARAMS,
):
    """
    Simplified version that takes distance_score directly.

    If distance_score is None, assumes average distance conditions.
    """

    if distance_score is None:
        distance_score = scalers["dist_mean"] if scalers else 0.15  # Approximate average

    if scalers is None:
        # Use approximate scalers from model output
        scalers = {
            "dist_mean": 0.15, "dist_std": 0.25,
            "dz_abs_mean": 35.0, "dz_abs_std": 30.0,
            "speedup_mag_mean": 0.03, "speedup_mag_std": 0.02,
            "RIX_avg_mean": 0.80, "RIX_avg_std": 1.20,
            "sev_frac_mean": 0.08, "sev_frac_std": 0.12,
            "y_std": 0.121,
        }

    # Standardize
    z_dist = (distance_score - scalers["dist_mean"]) / scalers["dist_std"]
    z_dz_abs = (abs(dz_m) - scalers["dz_abs_mean"]) / scalers["dz_abs_std"]
    z_speedup = (log_speedup_mag - scalers["speedup_mag_mean"]) / scalers["speedup_mag_std"]
    z_RIX = (RIX_avg - scalers["RIX_avg_mean"]) / scalers["RIX_avg_std"]
    z_sev = (severity_fraction - scalers["sev_frac_mean"]) / scalers["sev_frac_std"]

    # Apply model
    log_sigma = (params["log_sigma0"]
                 + params["gamma_dist"] * z_dist
                 + params["gamma_dz_abs"] * z_dz_abs
                 + params["gamma_speedup_mag"] * z_speedup
                 + params["gamma_RIX_avg"] * z_RIX
                 + params["gamma_sev_frac"] * z_sev)

    sigma = np.exp(log_sigma) * scalers["y_std"]

    return {
        "sigma_pct": sigma * 100,
        "range_95_pct": 1.96 * sigma * 100,
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 70)
    print("UNCERTAINTY CALCULATOR")
    print("=" * 70)

    # Load data and compute scalers
    print("\nLoading data...")
    df = pd.read_excel(sector_model_path)
    scalers = compute_scalers_from_data(df)

    # =========================================
    # EXAMPLE 1: Your specific case
    # =========================================

    print("\n\n" + "=" * 70)
    print("EXAMPLE 1: Your specified inputs")
    print("=" * 70)

    result1 = calculate_uncertainty(
        dz_m=20,  # 20m height difference
        distance_m=4000,  # 4 km
        distance_A=1000,  # Assumed lower bound (adjust based on your data)
        distance_B=10000,  # Assumed upper bound (adjust based on your data)
        log_speedup_mag=0.05,  # |log(WTG_speedup/MM_speedup)| ≈ 5% difference
        RIX_avg=5.0,  # High terrain severity
        severity_fraction=0.2,  # 20% of complexity diff is in severe terrain
        scalers=scalers,
        verbose=True
    )

    # =========================================
    # EXAMPLE 2: Best case scenario
    # =========================================

    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Best case (flat terrain, close distance)")
    print("=" * 70)

    result2 = calculate_uncertainty(
        dz_m=5,  # Small height difference
        distance_m=500,  # Close distance
        distance_A=1000,  # Within lower bound
        distance_B=10000,
        log_speedup_mag=0.01,  # Small speedup difference
        RIX_avg=0.5,  # Low terrain severity
        severity_fraction=0.0,  # No severe terrain
        scalers=scalers,
        verbose=True
    )

    # =========================================
    # EXAMPLE 3: Worst case scenario
    # =========================================

    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Worst case (complex terrain, large differences)")
    print("=" * 70)

    result3 = calculate_uncertainty(
        dz_m=100,  # Large height difference
        distance_m=8000,  # Far distance
        distance_A=1000,
        distance_B=10000,
        log_speedup_mag=0.10,  # Large speedup difference
        RIX_avg=8.0,  # Very complex terrain
        severity_fraction=0.5,  # Half of complexity is severe
        scalers=scalers,
        verbose=True
    )

    # =========================================
    # SUMMARY TABLE
    # =========================================

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<40} {'σ':<10} {'95% range'}")
    print("-" * 65)
    print(
        f"{'Your case (dz=20m, RIX=5)':<40} {result1['sigma_original'] * 100:.2f}%     ±{result1['range_95_pct']:.1f}%")
    print(f"{'Best case (flat, close)':<40} {result2['sigma_original'] * 100:.2f}%     ±{result2['range_95_pct']:.1f}%")
    print(
        f"{'Worst case (complex, far)':<40} {result3['sigma_original'] * 100:.2f}%     ±{result3['range_95_pct']:.1f}%")

    return scalers


if __name__ == "__main__":
    scalers = main()