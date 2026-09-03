"""
Uncertainty Calculator v2.0 — WS Sector-Level (Student-t, 6 Features)

Uses the v4 model: e ~ StudentT(nu, 0, sigma)
    log(sigma) = log(sigma_0) + sum(gamma_j * z_j)

6 features (all MM-only, deployment safe):
  1. dist_norm         — log(distance / distance_A)
  2. turning_gradient   — max neighbour turning difference per sector
  3. speedup_diff_std   — std of speedup difference across sectors
  4. k_MM_deviation     — |k_MM - 2.0| per sector
  5. sample_shortfall   — log deficit in sample count
  6. TI_MM              — turbulence intensity at predictor mast per sector

Confidence intervals use Student-t quantiles (not Normal 1.96*sigma).

Usage:
    1. Run the v4 model (WS_uncertainty_6feature.py) to generate results JSON
    2. Update RESULTS_JSON_PATH below
    3. Set INPUT_FILE to your site data Excel
    4. Run: python Uncertainty_calculator_v2.py

Input format:
    Excel with 12 rows per pair (one per sector). Required columns listed in
    REQUIRED_COLUMNS below.

Output:
    - Console summary with sector-level and pair-level results
    - Excel file with 4 sheets: Summary, Sector_Detail, Driver_Attribution, Recommendations
    - Visualisation PNGs (driver bar chart, sector polar plot, waterfall)
"""

import numpy as np
import pandas as pd
import json
import os
from scipy import stats as sp_stats
from scipy.special import gamma as gamma_func

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

# Path to results JSON from v4 model training
RESULTS_JSON_PATH = (
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling"
    r"\Bayesian_approach\WS_Bayesian_approach\WS Uncertainty_v3"
    r"\ws_model_v4_6feat_results.json"
)

# Input file — 12 rows per pair
INPUT_FILE = (
    r"C:\Kshitij stuff\Horizontal Uncertainty Check"
    r"\Uncertainty calculator\CEA\CEA_uncertainty_input 2.xlsx"
)

OUTPUT_FILE = None  # Set to None to auto-generate from INPUT_FILE

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}

REQUIRED_COLUMNS = [
    "pair_id", "sector_name",
    "d_turning_deg",
    "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
    "weight_energy_predicted",
    "distance_m", "distance_A",
    "Sample_count_pred",
    "k_MM",           # Weibull k at MM — per sector
    "TI_MM_clean",    # Turbulence intensity at MM — per sector (optional)
]

# Feature display names
FEATURE_DISPLAY = {
    "dist_norm":        "Normalised distance",
    "turning_gradient": "Turning gradient",
    "speedup_diff_std": "Speedup diff std",
    "k_MM_deviation":   "Weibull k deviation",
    "sample_shortfall": "Sample shortfall",
    "TI_MM":            "TI at MM",
}

# Confidence levels to report
CI_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────

def load_model_params(json_path: str) -> dict:
    """
    Load trained model parameters from v4 results JSON.

    Expected JSON structure (from WS_uncertainty_6feature.py save_results()):
    {
        "model": "Student-t sector-level v4 (6 features)",
        "nu": {"median": ..., "mean": ...},
        "log_sigma0": {"median": ..., "mean": ...},
        "gammas": {
            "gamma_dist_norm": {"median": ..., "mean": ..., "std": ..., "multiplier": ...},
            ...
        },
        "scalers": {
            "log_dist_norm_mean": ..., "log_dist_norm_std": ...,
            ...
        },
        ...
    }
    """
    if not os.path.exists(json_path):
        print(f"\nWARNING: Results JSON not found: {json_path}")
        print("Using PLACEHOLDER parameters. Run the v4 model first to get real values.\n")
        return _placeholder_params()

    with open(json_path, "r") as f:
        raw = json.load(f)

    # Extract what we need
    params = {
        "nu": raw["nu"]["median"],
        "log_sigma0": raw["log_sigma0"]["median"],
    }

    # Gamma posteriors
    params["gammas"] = {}
    gamma_keys = [
        "gamma_dist_norm", "gamma_turning_grad", "gamma_speedup_diff_std",
        "gamma_k_dev", "gamma_sample", "gamma_TI_MM",
    ]
    for gk in gamma_keys:
        if gk in raw.get("gammas", {}):
            params["gammas"][gk] = raw["gammas"][gk]["median"]
        else:
            params["gammas"][gk] = 0.0
            print(f"  WARNING: {gk} not found in results JSON, using 0.0")

    # Scalers
    params["scalers"] = raw.get("scalers", {})

    # y_std for converting standardised sigma to original units
    params["y_std"] = params["scalers"].get("e_std", params["scalers"].get("y_std", 0.064))

    print(f"Loaded model parameters from: {json_path}")
    print(f"  nu = {params['nu']:.1f}")
    print(f"  log_sigma0 = {params['log_sigma0']:.4f}")
    for gk, gv in params["gammas"].items():
        print(f"  {gk} = {gv:.4f}")

    return params


def _placeholder_params() -> dict:
    """Placeholder parameters — replace by running the v4 model."""
    return {
        "nu": 9.0,
        "log_sigma0": -1.55,
        "gammas": {
            "gamma_dist_norm": 0.30,
            "gamma_turning_grad": 0.20,
            "gamma_speedup_diff_std": 0.15,
            "gamma_k_dev": 0.15,
            "gamma_sample": 0.20,
            "gamma_TI_MM": 0.10,
        },
        "scalers": {
            "log_dist_norm_mean": 0.0, "log_dist_norm_std": 1.0,
            "turning_gradient_mean": 0.0, "turning_gradient_std": 1.0,
            "speedup_diff_std_mean": 0.0, "speedup_diff_std_std": 1.0,
            "k_MM_deviation_mean": 0.0, "k_MM_deviation_std": 1.0,
            "sample_shortfall_mean": 0.0, "sample_shortfall_std": 1.0,
            "TI_MM_clean_mean": 0.0, "TI_MM_clean_std": 1.0,
        },
        "y_std": 0.064,
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION (sector-level)
# ─────────────────────────────────────────────────────────────────

def compute_sector_features(pair_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 6 features for one pair (12 sector rows).
    Returns the same dataframe with feature columns added.
    """
    d = pair_data.copy()

    if len(d) != 12:
        raise ValueError(f"Expected 12 sectors, got {len(d)}")

    d["sector_idx"] = d["sector_name"].map(SECTOR_TO_IDX)
    d = d.sort_values("sector_idx")

    # --- Feature 1: dist_norm (pair-level, same for all sectors) ---
    distance_m = d["distance_m"].iloc[0]
    distance_A = max(d["distance_A"].iloc[0], 1.0)  # avoid log(0)
    d["log_dist_norm"] = np.log(distance_m / distance_A)

    # --- Feature 2: turning_gradient (sector-level) ---
    turning = d["d_turning_deg"].values
    n = len(turning)
    gradients = np.zeros(n)
    for i in range(n):
        left = turning[(i - 1) % n]
        right = turning[(i + 1) % n]
        current = turning[i]
        gradients[i] = max(abs(current - left), abs(current - right))
    d["turning_gradient"] = gradients

    # --- Feature 3: speedup_diff_std (pair-level) ---
    speedup_diff = d["overall_speedup_MM_factor"].values - d["overall_speedup_WTG_factor"].values
    d["speedup_diff_std"] = np.std(speedup_diff)

    # --- Feature 4: k_MM_deviation (sector-level) ---
    if "k_MM" in d.columns and d["k_MM"].notna().any():
        d["k_MM_deviation"] = np.abs(d["k_MM"] - 2.0)
        # Fill NaN with 0 (neutral contribution after z-scoring to mean)
        if d["k_MM_deviation"].isna().any():
            d["k_MM_deviation"] = d["k_MM_deviation"].fillna(d["k_MM_deviation"].mean())
    else:
        d["k_MM_deviation"] = 0.0

    # --- Feature 5: sample_shortfall (pair-level) ---
    total_samples = d["Sample_count_pred"].sum()
    # Use training-data max as reference (from model training)
    LOG_MAX_SAMPLES = np.log(112470 + 1)
    d["sample_shortfall"] = max(LOG_MAX_SAMPLES - np.log(total_samples + 1), 0.0)

    # --- Feature 6: TI_MM (sector-level) ---
    has_ti = "TI_MM_clean" in d.columns and d["TI_MM_clean"].notna().any()
    if has_ti:
        ti_median = d["TI_MM_clean"].median()
        d["TI_MM_clean"] = d["TI_MM_clean"].fillna(ti_median)
    else:
        d["TI_MM_clean"] = 0.0

    d["has_ti"] = has_ti

    return d


def standardise_features(d: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Z-score features using training-set scalers."""
    feature_cols = {
        "log_dist_norm":    ("log_dist_norm_mean",    "log_dist_norm_std"),
        "turning_gradient": ("turning_gradient_mean",  "turning_gradient_std"),
        "speedup_diff_std": ("speedup_diff_std_mean",  "speedup_diff_std_std"),
        "k_MM_deviation":   ("k_MM_deviation_mean",    "k_MM_deviation_std"),
        "sample_shortfall": ("sample_shortfall_mean",  "sample_shortfall_std"),
        "TI_MM_clean":      ("TI_MM_clean_mean",       "TI_MM_clean_std"),
    }

    for col, (mean_key, std_key) in feature_cols.items():
        if col not in d.columns:
            d[f"{col}_z"] = 0.0
            continue
        mean = scalers.get(mean_key, 0.0)
        std = scalers.get(std_key, 1.0)
        if std == 0:
            std = 1.0
        d[f"{col}_z"] = (d[col] - mean) / std

    return d


# ─────────────────────────────────────────────────────────────────
# UNCERTAINTY CALCULATION
# ─────────────────────────────────────────────────────────────────

def calculate_sector_uncertainty(d: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate sigma and confidence intervals for each sector row.
    Returns dataframe with sigma, CI columns, and driver contributions.
    """
    log_sigma0 = params["log_sigma0"]
    nu = params["nu"]
    gammas = params["gammas"]
    y_std = params["y_std"]

    # Map gamma keys to z-score column names
    gamma_to_zcol = {
        "gamma_dist_norm":        "log_dist_norm_z",
        "gamma_turning_grad":     "turning_gradient_z",
        "gamma_speedup_diff_std": "speedup_diff_std_z",
        "gamma_k_dev":            "k_MM_deviation_z",
        "gamma_sample":           "sample_shortfall_z",
        "gamma_TI_MM":            "TI_MM_clean_z",
    }

    # Short names for display
    gamma_to_feature = {
        "gamma_dist_norm":        "dist_norm",
        "gamma_turning_grad":     "turning_gradient",
        "gamma_speedup_diff_std": "speedup_diff_std",
        "gamma_k_dev":            "k_MM_deviation",
        "gamma_sample":           "sample_shortfall",
        "gamma_TI_MM":            "TI_MM",
    }

    # Calculate log(sigma) per sector
    log_sigma = np.full(len(d), log_sigma0)

    # Store per-driver contributions
    for gk, gamma_val in gammas.items():
        zcol = gamma_to_zcol[gk]
        feat_name = gamma_to_feature[gk]
        contribution = gamma_val * d[zcol].values
        log_sigma += contribution
        d[f"contrib_{feat_name}"] = contribution

    # Sigma in standardised and original units
    d["sigma_std"] = np.exp(log_sigma)
    d["sigma_pct"] = d["sigma_std"] * y_std * 100  # percentage

    # E[|e|] for Student-t: 0.87*sigma when nu≈9
    e_abs_factor = np.sqrt(nu / (nu - 2)) * (
        2 * np.sqrt(nu) * gamma_func((nu + 1) / 2)
        / ((nu - 1) * np.sqrt(np.pi) * gamma_func(nu / 2))
    ) if nu > 2 else 0.87
    d["expected_abs_error_pct"] = d["sigma_std"] * y_std * e_abs_factor * 100

    # Confidence intervals using Student-t quantiles
    for level in CI_LEVELS:
        alpha = 1 - level
        t_val = sp_stats.t.ppf(1 - alpha / 2, df=nu)
        d[f"CI_{int(level*100)}_lower_pct"] = -t_val * d["sigma_std"] * y_std * 100
        d[f"CI_{int(level*100)}_upper_pct"] = +t_val * d["sigma_std"] * y_std * 100

    # Driver attribution: percentage of deviation from baseline
    contrib_cols = [c for c in d.columns if c.startswith("contrib_")]
    total_deviation = log_sigma - log_sigma0
    for cc in contrib_cols:
        feat = cc.replace("contrib_", "")
        pct_col = f"pct_{feat}"
        d[pct_col] = np.where(
            np.abs(total_deviation) > 1e-6,
            (d[cc] / total_deviation) * 100,
            0.0
        )

    return d


def aggregate_pair_results(d: pd.DataFrame, params: dict) -> dict:
    """
    Aggregate sector-level results to pair-level using energy weights.
    """
    weights = d["weight_energy_predicted"].values
    weights = weights / weights.sum()

    sigma_pct = d["sigma_pct"].values
    nu = params["nu"]

    # Energy-weighted average sigma
    sigma_pair = np.sum(weights * sigma_pct)

    # Energy-weighted CIs
    ci_results = {}
    for level in CI_LEVELS:
        key = int(level * 100)
        ci_results[f"CI_{key}_lower"] = np.sum(weights * d[f"CI_{key}_lower_pct"].values)
        ci_results[f"CI_{key}_upper"] = np.sum(weights * d[f"CI_{key}_upper_pct"].values)

    # Energy-weighted driver attribution
    contrib_cols = [c for c in d.columns if c.startswith("contrib_")]
    driver_contribs = {}
    for cc in contrib_cols:
        feat = cc.replace("contrib_", "")
        driver_contribs[feat] = {
            "weighted_contribution": np.sum(weights * d[cc].values),
            "display_name": FEATURE_DISPLAY.get(feat, feat),
        }

    # Total log-sigma deviation
    total_contrib = sum(dc["weighted_contribution"] for dc in driver_contribs.values())
    for feat, dc in driver_contribs.items():
        if abs(total_contrib) > 1e-6:
            dc["pct_of_total"] = (dc["weighted_contribution"] / total_contrib) * 100
        else:
            dc["pct_of_total"] = 0.0
        dc["sigma_multiplier"] = np.exp(dc["weighted_contribution"])

    return {
        "pair_id": d["pair_id"].iloc[0],
        "distance_m": d["distance_m"].iloc[0],
        "sigma_pct": sigma_pair,
        "expected_abs_error_pct": np.sum(weights * d["expected_abs_error_pct"].values),
        **ci_results,
        "driver_contributions": driver_contribs,
        "sector_sigmas": dict(zip(d["sector_name"].values, sigma_pct)),
        "sector_weights": dict(zip(d["sector_name"].values, weights)),
    }


# ─────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────

def generate_recommendations(pair_result: dict, features_raw: pd.DataFrame) -> list:
    """Generate actionable recommendations based on high-contributing drivers."""
    recs = []
    drivers = pair_result["driver_contributions"]

    # Sort drivers by absolute contribution
    ranked = sorted(drivers.items(), key=lambda x: abs(x[1]["weighted_contribution"]), reverse=True)
    top_driver = ranked[0][0] if ranked else None

    # Distance
    if "dist_norm" in drivers and drivers["dist_norm"]["sigma_multiplier"] > 1.15:
        mult = drivers["dist_norm"]["sigma_multiplier"]
        recs.append(
            f"DISTANCE: Sites are far apart relative to TR6 guideline "
            f"(sigma multiplier: {mult:.2f}x). Consider using a closer reference mast."
        )

    # Sample shortfall
    if "sample_shortfall" in drivers and drivers["sample_shortfall"]["sigma_multiplier"] > 1.10:
        mult = drivers["sample_shortfall"]["sigma_multiplier"]
        recs.append(
            f"CAMPAIGN LENGTH: Sample shortfall is high (sigma multiplier: {mult:.2f}x). "
            f"Extending the measurement campaign would reduce epistemic uncertainty."
        )

    # Turning gradient
    if "turning_gradient" in drivers and drivers["turning_gradient"]["sigma_multiplier"] > 1.10:
        mult = drivers["turning_gradient"]["sigma_multiplier"]
        # Find the sectors with highest turning gradient
        top_sectors = features_raw.nlargest(3, "turning_gradient")[["sector_name", "turning_gradient"]]
        sector_str = ", ".join(f"{r['sector_name']} ({r['turning_gradient']:.1f} deg)"
                               for _, r in top_sectors.iterrows())
        recs.append(
            f"FLOW COMPLEXITY: Large turning gradients in sectors {sector_str} "
            f"(sigma multiplier: {mult:.2f}x). These sectors have rapid directional "
            f"changes that increase prediction uncertainty."
        )

    # Speedup diff
    if "speedup_diff_std" in drivers and drivers["speedup_diff_std"]["sigma_multiplier"] > 1.10:
        mult = drivers["speedup_diff_std"]["sigma_multiplier"]
        recs.append(
            f"TERRAIN MISMATCH: High variability in speedup differences across sectors "
            f"(sigma multiplier: {mult:.2f}x). The terrain at predictor and target sites "
            f"responds very differently by direction."
        )

    # Weibull k
    if "k_MM_deviation" in drivers and drivers["k_MM_deviation"]["sigma_multiplier"] > 1.10:
        mult = drivers["k_MM_deviation"]["sigma_multiplier"]
        recs.append(
            f"WIND REGIME: Weibull k deviates from typical value of 2.0 in some sectors "
            f"(sigma multiplier: {mult:.2f}x). Unusual wind speed distributions increase "
            f"transfer uncertainty."
        )

    # TI
    if "TI_MM" in drivers and drivers["TI_MM"]["sigma_multiplier"] > 1.10:
        mult = drivers["TI_MM"]["sigma_multiplier"]
        recs.append(
            f"TURBULENCE: High turbulence intensity at MM "
            f"(sigma multiplier: {mult:.2f}x). Turbulent conditions increase "
            f"wind speed transfer uncertainty."
        )

    if not recs:
        recs.append("All drivers are within normal ranges. Prediction uncertainty is near baseline.")

    return recs


# ─────────────────────────────────────────────────────────────────
# CONSOLE OUTPUT
# ─────────────────────────────────────────────────────────────────

def print_pair_results(pair_result: dict, recs: list):
    """Print detailed results for one pair."""
    pr = pair_result
    print("\n" + "=" * 80)
    print(f"PAIR: {pr['pair_id']}")
    print("=" * 80)

    # --- Overall uncertainty ---
    print(f"\n  Wind Speed Transfer Uncertainty (Student-t, v4 6-feature model)")
    print(f"  {'sigma (WS deviation):':<30} {pr['sigma_pct']:.2f}%")
    print(f"  {'E[|error|]:':<30} {pr['expected_abs_error_pct']:.2f}%")

    print(f"\n  Confidence Intervals:")
    for level in CI_LEVELS:
        key = int(level * 100)
        lo = pr[f"CI_{key}_lower"]
        hi = pr[f"CI_{key}_upper"]
        print(f"    {level*100:.0f}% CI:  [{lo:+.2f}%, {hi:+.2f}%]")

    # --- Driver attribution ---
    print(f"\n  Driver Attribution (sorted by impact):")
    print(f"  {'Driver':<25} {'gamma*z':<10} {'Multiplier':<12} {'% of total':<10}")
    print("  " + "-" * 57)

    ranked = sorted(
        pr["driver_contributions"].items(),
        key=lambda x: abs(x[1]["weighted_contribution"]),
        reverse=True
    )

    for feat, dc in ranked:
        display = dc["display_name"]
        print(f"  {display:<25} {dc['weighted_contribution']:+.4f}    "
              f"{dc['sigma_multiplier']:.3f}x       {dc['pct_of_total']:+.1f}%")

    # --- Sector breakdown ---
    print(f"\n  Sector-level sigma (%):")
    sigmas = pr["sector_sigmas"]
    weights = pr["sector_weights"]
    for sector in SECTOR_LABELS:
        if sector in sigmas:
            bar = "#" * int(sigmas[sector] / max(sigmas.values()) * 20)
            print(f"    {sector:<5} {sigmas[sector]:5.2f}%  (w={weights[sector]:.3f})  {bar}")

    # --- Recommendations ---
    if recs:
        print(f"\n  RECOMMENDATIONS:")
        for i, rec in enumerate(recs, 1):
            print(f"    {i}. {rec}")


# ─────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────

def export_results_to_excel(all_results: list, sector_dfs: list, output_path: str):
    """Export results to Excel with 4 sheets."""

    # Sheet 1: Summary (one row per pair)
    summary_rows = []
    for pr in all_results:
        row = {
            "pair_id": pr["pair_id"],
            "distance_m": pr["distance_m"],
            "sigma_pct": pr["sigma_pct"],
            "E_abs_error_pct": pr["expected_abs_error_pct"],
        }
        for level in CI_LEVELS:
            key = int(level * 100)
            row[f"CI_{key}_lower"] = pr[f"CI_{key}_lower"]
            row[f"CI_{key}_upper"] = pr[f"CI_{key}_upper"]

        # Top driver
        ranked = sorted(pr["driver_contributions"].items(),
                        key=lambda x: abs(x[1]["weighted_contribution"]), reverse=True)
        row["top_driver"] = ranked[0][1]["display_name"] if ranked else ""
        row["top_driver_multiplier"] = ranked[0][1]["sigma_multiplier"] if ranked else 1.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sheet 2: Sector detail (12 rows per pair)
    sector_detail = pd.concat(sector_dfs, ignore_index=True)
    keep_cols = [
        "pair_id", "sector_name", "sigma_pct", "expected_abs_error_pct",
    ]
    for level in CI_LEVELS:
        key = int(level * 100)
        keep_cols += [f"CI_{key}_lower_pct", f"CI_{key}_upper_pct"]
    # Add raw feature columns
    keep_cols += ["log_dist_norm", "turning_gradient", "speedup_diff_std",
                  "k_MM_deviation", "sample_shortfall", "TI_MM_clean"]
    available = [c for c in keep_cols if c in sector_detail.columns]
    sector_out = sector_detail[available]

    # Sheet 3: Driver attribution (one row per pair, columns per driver)
    driver_rows = []
    for pr in all_results:
        row = {"pair_id": pr["pair_id"]}
        for feat, dc in pr["driver_contributions"].items():
            display = dc["display_name"]
            row[f"{display}_contribution"] = dc["weighted_contribution"]
            row[f"{display}_multiplier"] = dc["sigma_multiplier"]
            row[f"{display}_pct"] = dc["pct_of_total"]
        driver_rows.append(row)
    driver_df = pd.DataFrame(driver_rows)

    # Sheet 4: Recommendations
    rec_rows = []
    for pr, recs in zip(all_results, [pr.get("recommendations", []) for pr in all_results]):
        for i, rec in enumerate(recs, 1):
            rec_rows.append({"pair_id": pr["pair_id"], "priority": i, "recommendation": rec})
    rec_df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(columns=["pair_id", "priority", "recommendation"])

    # Write
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        sector_out.to_excel(writer, sheet_name="Sector_Detail", index=False)
        driver_df.to_excel(writer, sheet_name="Driver_Attribution", index=False)
        rec_df.to_excel(writer, sheet_name="Recommendations", index=False)

    print(f"\nExported results to: {output_path}")


# ─────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────

def create_visualisations(all_results: list, sector_dfs: list, output_dir: str):
    """
    Generate publication-quality visualisation PNGs.

    1. Driver attribution bar chart (horizontal, per pair)
    2. Sector-level sigma polar plot (one per pair)
    3. Waterfall chart showing cumulative sigma build-up
    4. Multi-pair comparison dashboard (if >1 pair)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not available — skipping visualisations.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    for pr in all_results:
        pair_id = pr["pair_id"]
        safe_id = pair_id.replace("/", "_").replace("\\", "_")

        # ── 1. Driver attribution bar chart ──────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ranked = sorted(
            pr["driver_contributions"].items(),
            key=lambda x: x[1]["weighted_contribution"],
            reverse=True
        )
        names = [dc["display_name"] for _, dc in ranked]
        contribs = [dc["weighted_contribution"] for _, dc in ranked]
        colors = ["#e74c3c" if c > 0 else "#27ae60" for c in contribs]

        bars = ax.barh(names, contribs, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Contribution to log(sigma)")
        ax.set_title(f"Driver Attribution — {pair_id}", fontweight="bold")
        ax.invert_yaxis()

        # Add multiplier labels
        for bar, (_, dc) in zip(bars, ranked):
            x = bar.get_width()
            ax.text(x + 0.005 * np.sign(x), bar.get_y() + bar.get_height() / 2,
                    f"{dc['sigma_multiplier']:.2f}x",
                    va="center", ha="left" if x >= 0 else "right", fontsize=8, color="#555")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_id}_driver_attribution.png"))
        plt.close(fig)

        # ── 2. Sector polar plot ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        sector_sigmas = pr["sector_sigmas"]
        sector_weights = pr["sector_weights"]

        angles = [np.deg2rad(i * 30) for i in range(12)]
        angles.append(angles[0])  # close the loop

        sigmas = [sector_sigmas.get(s, 0) for s in SECTOR_LABELS]
        sigmas.append(sigmas[0])

        weights_arr = [sector_weights.get(s, 0) for s in SECTOR_LABELS]
        weights_arr.append(weights_arr[0])

        # Sigma line
        ax.plot(angles, sigmas, "o-", color="#e74c3c", linewidth=2, label="sigma (%)", markersize=5)
        ax.fill(angles, sigmas, alpha=0.15, color="#e74c3c")

        # Weight line (scaled to be visible)
        max_sigma = max(sigmas) if max(sigmas) > 0 else 1
        scaled_weights = [w * max_sigma / max(weights_arr) * 0.6 for w in weights_arr]
        ax.plot(angles, scaled_weights, "s--", color="#3498db", linewidth=1.5,
                label="energy weight (scaled)", markersize=4, alpha=0.7)

        ax.set_thetagrids([i * 30 for i in range(12)], SECTOR_LABELS)
        ax.set_title(f"Sector Uncertainty — {pair_id}", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_id}_sector_polar.png"))
        plt.close(fig)

        # ── 3. Waterfall chart ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))

        log_sigma0 = np.log(pr["sigma_pct"] / 100 / all_results[0].get("y_std", 0.064))
        # Use the actual baseline
        baseline = float(pr.get("log_sigma0", -1.55))

        steps = [("Baseline", baseline, "#95a5a6")]
        cumulative = baseline
        for feat, dc in ranked:
            c = dc["weighted_contribution"]
            steps.append((dc["display_name"], c, "#e74c3c" if c > 0 else "#27ae60"))
            cumulative += c
        steps.append(("Final log(sigma)", cumulative, "#2c3e50"))

        x_pos = list(range(len(steps)))
        running = 0
        for i, (label, val, color) in enumerate(steps):
            if i == 0:
                ax.bar(i, val, color=color, edgecolor="white", width=0.6)
                running = val
            elif i == len(steps) - 1:
                ax.bar(i, cumulative, color=color, edgecolor="white", width=0.6)
            else:
                ax.bar(i, val, bottom=running, color=color, edgecolor="white", width=0.6)
                running += val

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in steps], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("log(sigma)")
        ax.set_title(f"Sigma Build-up Waterfall — {pair_id}", fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_id}_waterfall.png"))
        plt.close(fig)

    # ── 4. Multi-pair comparison (if >1 pair) ────────────────────
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        pair_ids = [pr["pair_id"] for pr in all_results]
        sigmas = [pr["sigma_pct"] for pr in all_results]

        # Bar chart of sigmas
        ax = axes[0]
        colors = cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sigmas)))
        ax.barh(pair_ids, sigmas, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel("sigma (%)")
        ax.set_title("Pair-Level Uncertainty Comparison", fontweight="bold")
        ax.invert_yaxis()

        for i, (pid, s) in enumerate(zip(pair_ids, sigmas)):
            ax.text(s + 0.05, i, f"{s:.2f}%", va="center", fontsize=8)

        # 95% CI comparison
        ax = axes[1]
        for i, pr in enumerate(all_results):
            lo = pr["CI_95_lower"]
            hi = pr["CI_95_upper"]
            ax.plot([lo, hi], [i, i], "o-", color=colors[i], linewidth=2, markersize=6)
            ax.plot(0, i, "|", color="black", markersize=10)

        ax.set_yticks(range(len(pair_ids)))
        ax.set_yticklabels(pair_ids)
        ax.set_xlabel("WS deviation (%)")
        ax.set_title("95% Confidence Intervals", fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "multi_pair_comparison.png"))
        plt.close(fig)

    print(f"Saved visualisations to: {output_dir}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def process_input_file(input_path: str, params: dict, output_path: str = None):
    """
    Process input file and calculate sector-level uncertainties for all pairs.
    """
    print("=" * 80)
    print("UNCERTAINTY CALCULATOR v2.0 — WS Student-t (6 features)")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_excel(input_path)

    # Check required columns (TI_MM_clean is optional)
    core_required = [c for c in REQUIRED_COLUMNS if c not in ("TI_MM_clean",)]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_ti = "TI_MM_clean" in df.columns and df["TI_MM_clean"].notna().sum() > 0
    print(f"TI data: {'available' if has_ti else 'NOT available (TI_MM contribution will be zero)'}")

    # Process each pair
    pairs = df.groupby("pair_id")
    n_pairs = len(pairs)
    print(f"Found {n_pairs} pair(s)\n")

    all_results = []
    sector_dfs = []

    for pair_id, pair_data in pairs:
        if len(pair_data) != 12:
            print(f"  WARNING: {pair_id} has {len(pair_data)} sectors (expected 12), skipping")
            continue

        # Compute features
        d = compute_sector_features(pair_data)

        # Standardise
        d = standardise_features(d, params["scalers"])

        # Calculate uncertainty
        d = calculate_sector_uncertainty(d, params)

        # Aggregate to pair level
        pair_result = aggregate_pair_results(d, params)
        pair_result["y_std"] = params["y_std"]
        pair_result["log_sigma0"] = params["log_sigma0"]

        # Recommendations
        recs = generate_recommendations(pair_result, d)
        pair_result["recommendations"] = recs

        # Print
        print_pair_results(pair_result, recs)

        all_results.append(pair_result)
        sector_dfs.append(d)

    if not all_results:
        print("\nNo valid pairs found.")
        return []

    # Output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_uncertainty_v2_results.xlsx"

    # Export Excel
    export_results_to_excel(all_results, sector_dfs, output_path)

    # Visualisations
    output_dir = os.path.dirname(output_path) or os.path.dirname(input_path)
    create_visualisations(all_results, sector_dfs, output_dir)

    # Final summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL PAIRS")
        print("=" * 80)

        sigmas = [r["sigma_pct"] for r in all_results]
        print(f"\n  {'Metric':<25} {'Value':<15}")
        print("  " + "-" * 40)
        print(f"  {'Mean sigma':<25} {np.mean(sigmas):.2f}%")
        print(f"  {'Min sigma':<25} {np.min(sigmas):.2f}%")
        print(f"  {'Max sigma':<25} {np.max(sigmas):.2f}%")
        print(f"  {'Median sigma':<25} {np.median(sigmas):.2f}%")

    return all_results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model parameters
    params = load_model_params(RESULTS_JSON_PATH)

    # Run
    try:
        results = process_input_file(INPUT_FILE, params, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Please update INPUT_FILE in the script.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
