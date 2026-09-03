"""
Distance-Terrain Interaction Exploration

Hypothesis: In flat terrain, closer masts have lower-than-expected uncertainty
            in a way that doesn't happen in complex terrain. distance_A (from FGW)
            captures terrain-dependent "acceptable" distance.

Plots:
1. (predicted_sigma_fixed - actual_|error|) vs distance_m, colored by terrain category
2. Same residual vs distance_m / distance_A (terrain-normalised distance)
3. Actual |error| vs distance_m by terrain category
4. Actual |error| vs distance_m / distance_A by terrain category

Terrain categories (based on RIX_avg_0.0501_overall_pair):
  - Flat:         < 40
  - Semi-complex: 40 - 50
  - Complex:      > 50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
SECTOR_MODEL_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
MODEL_SCRIPT_DIR = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Bayesian_approach\WS_Bayesian_approach\WS Uncertainty_v3"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}

# Terrain bins
TERRAIN_BINS = {
    "Flat": (0, 40),
    "Semi-complex": (40, 50),
    "Complex": (50, 999),
}
TERRAIN_COLORS = {
    "Flat": "#2ecc71",
    "Semi-complex": "#f39c12",
    "Complex": "#e74c3c",
}


def load_data():
    """Load data, compute pair-level features, return pair-level DataFrame."""
    import sys
    sys.path.insert(0, MODEL_SCRIPT_DIR)

    df = pd.read_excel(SECTOR_MODEL_PATH)

    # Basic filters (match main script)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    # Signed relative WS error per sector
    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # Pair-level RIX: mean of RIX_avg_0.0501_sector across sectors
    if "RIX_avg_0.0501_overall_pair" in df.columns:
        print("  Using existing RIX_avg_0.0501_overall_pair column")
    else:
        print("  Computing RIX_avg_0.0501_overall_pair from sector means")
        df["RIX_avg_0.0501_overall_pair"] = df.groupby("pair_id")["RIX_avg_0.0501_sector"].transform("mean")

    # Terrain category
    def classify_terrain(rix):
        for name, (lo, hi) in TERRAIN_BINS.items():
            if lo <= rix < hi:
                return name
        return "Complex"

    df["terrain_cat"] = df["RIX_avg_0.0501_overall_pair"].apply(classify_terrain)

    # Normalised distance
    df["dist_norm"] = df["distance_m"] / np.clip(df["distance_A"], 1, None)

    # Energy-weighted aggregation to pair level
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "distance_m": g["distance_m"].iloc[0],
            "distance_A": g["distance_A"].iloc[0],
            "dist_norm": g["dist_norm"].iloc[0],
            "RIX_pair": g["RIX_avg_0.0501_overall_pair"].iloc[0],
            "terrain_cat": g["terrain_cat"].iloc[0],
            "location": g["location"].iloc[0],
            "abs_dz": np.abs(g["dz"].iloc[0]),
        })
    ).reset_index()

    # Ensure numeric
    for col in ["actual_abs_error", "distance_m", "distance_A", "dist_norm", "RIX_pair", "abs_dz"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    print(f"\n  Loaded {len(pair_agg)} pairs")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        print(f"    {cat}: {n} pairs")

    return df, pair_agg


def try_load_fixed_effects_sigma(pair_agg):
    """
    Try to load the fitted model and compute fixed-effects-only sigma.
    Falls back to None if model artifacts aren't available.
    """
    try:
        import sys
        sys.path.insert(0, MODEL_SCRIPT_DIR)
        import arviz as az
        from plot_pair_level_fixed_effects import compute_sigma_fixed_only
        from pathlib import Path

        # Try loading saved idata
        idata_path = Path(MODEL_SCRIPT_DIR) / "ws_sector_model_v3_studentt_idata.nc"
        if not idata_path.exists():
            print("  No saved idata found - skipping fixed-effects sigma")
            return None

        # We need the full data dict to compute fixed-effects sigma
        # This requires running the main script's build function
        print("  Loading model artifacts for fixed-effects sigma...")
        idata = az.from_netcdf(str(idata_path))
        return idata
    except Exception as ex:
        print(f"  Could not load model: {ex}")
        return None


def plot_exploration(pair_agg, sigma_fixed_pairs=None):
    """
    Create 4-panel figure:
      Top row: actual |error| vs distance_m and vs dist_norm
      Bottom row: residual (predicted - actual) vs distance_m and vs dist_norm
                  (only if sigma_fixed available)
    """

    has_sigma = sigma_fixed_pairs is not None
    nrows = 2 if has_sigma else 1

    fig, axes = plt.subplots(nrows, 2, figsize=(16, 7 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # ---- Row 1: Actual |error| vs distance ----
    for col_idx, (x_col, x_label) in enumerate([
        ("distance_m", "Distance (m)"),
        ("dist_norm", "Distance / Distance_A (normalised)"),
    ]):
        ax = axes[0, col_idx]
        for cat in ["Flat", "Semi-complex", "Complex"]:
            mask = pair_agg["terrain_cat"] == cat
            subset = pair_agg[mask]
            if len(subset) == 0:
                continue

            ax.scatter(
                subset[x_col], subset["actual_abs_error"],
                color=TERRAIN_COLORS[cat], label=cat,
                alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
                zorder=3,
            )

            # Trend line
            if len(subset) > 3:
                slope, intercept, r, p, _ = stats.linregress(subset[x_col], subset["actual_abs_error"])
                x_range = np.linspace(subset[x_col].min(), subset[x_col].max(), 50)
                ax.plot(x_range, slope * x_range + intercept,
                        color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
                # Annotate r value
                ax.text(
                    x_range[-1], slope * x_range[-1] + intercept,
                    f" r={r:.2f}", color=TERRAIN_COLORS[cat],
                    fontsize=9, fontweight="bold", va="center",
                )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
        ax.set_title(f"Actual |Error| vs {x_label.split('(')[0].strip()}", fontsize=13, fontweight="bold")
        ax.legend(title="Terrain", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # ---- Row 2: Residual (predicted sigma - actual) vs distance ----
    if has_sigma:
        pair_agg = pair_agg.copy()
        pair_agg["residual"] = sigma_fixed_pairs - pair_agg["actual_abs_error"]

        for col_idx, (x_col, x_label) in enumerate([
            ("distance_m", "Distance (m)"),
            ("dist_norm", "Distance / Distance_A (normalised)"),
        ]):
            ax = axes[1, col_idx]

            ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5, zorder=1)

            for cat in ["Flat", "Semi-complex", "Complex"]:
                mask = pair_agg["terrain_cat"] == cat
                subset = pair_agg[mask]
                if len(subset) == 0:
                    continue

                ax.scatter(
                    subset[x_col], subset["residual"],
                    color=TERRAIN_COLORS[cat], label=cat,
                    alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
                    zorder=3,
                )

                if len(subset) > 3:
                    slope, intercept, r, p, _ = stats.linregress(subset[x_col], subset["residual"])
                    x_range = np.linspace(subset[x_col].min(), subset[x_col].max(), 50)
                    ax.plot(x_range, slope * x_range + intercept,
                            color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
                    ax.text(
                        x_range[-1], slope * x_range[-1] + intercept,
                        f" r={r:.2f}", color=TERRAIN_COLORS[cat],
                        fontsize=9, fontweight="bold", va="center",
                    )

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("Residual (Predicted σ_fixed - Actual |Error|)", fontsize=12)
            ax.set_title(f"Model Residual vs {x_label.split('(')[0].strip()}", fontsize=13, fontweight="bold")
            ax.legend(title="Terrain", fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Distance-Terrain Exploration: Does close distance in flat terrain reduce uncertainty?",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("distance_terrain_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: distance_terrain_exploration.png")


def print_summary_stats(pair_agg):
    """Print summary statistics by terrain category."""

    print("\n" + "=" * 70)
    print("SUMMARY: Actual |Error| by Terrain and Distance")
    print("=" * 70)

    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) == 0:
            continue

        print(f"\n  {cat} (n={len(subset)}):")
        print(f"    Distance range:     {subset['distance_m'].min():.0f} - {subset['distance_m'].max():.0f} m")
        print(f"    Distance_A range:   {subset['distance_A'].min():.0f} - {subset['distance_A'].max():.0f} m")
        print(f"    dist_norm range:    {subset['dist_norm'].min():.2f} - {subset['dist_norm'].max():.2f}")
        print(f"    Mean |error|:       {subset['actual_abs_error'].mean():.2%}")
        print(f"    RIX_pair range:     {subset['RIX_pair'].min():.1f} - {subset['RIX_pair'].max():.1f}")

        # Split by close vs far (median distance)
        median_dist = subset["distance_m"].median()
        close = subset[subset["distance_m"] <= median_dist]
        far = subset[subset["distance_m"] > median_dist]
        if len(close) > 1 and len(far) > 1:
            print(f"    Close (d <= {median_dist:.0f}m): mean |e| = {close['actual_abs_error'].mean():.2%}  (n={len(close)})")
            print(f"    Far   (d >  {median_dist:.0f}m): mean |e| = {far['actual_abs_error'].mean():.2%}  (n={len(far)})")

        # Correlation: distance vs error
        if len(subset) > 3:
            r_raw, p_raw = stats.pearsonr(subset["distance_m"], subset["actual_abs_error"])
            r_norm, p_norm = stats.pearsonr(subset["dist_norm"], subset["actual_abs_error"])
            print(f"    Corr(distance, |e|):    r={r_raw:.3f}  p={p_raw:.3f}")
            print(f"    Corr(dist_norm, |e|):   r={r_norm:.3f}  p={p_norm:.3f}")

    # Overall comparison: does normalising help?
    print(f"\n  --- Overall correlations ---")
    r_raw, _ = stats.pearsonr(pair_agg["distance_m"], pair_agg["actual_abs_error"])
    r_norm, _ = stats.pearsonr(pair_agg["dist_norm"], pair_agg["actual_abs_error"])
    print(f"    Corr(distance_m, |e|):  r={r_raw:.3f}")
    print(f"    Corr(dist_norm, |e|):   r={r_norm:.3f}")
    if abs(r_norm) > abs(r_raw):
        print(f"    -> Normalising by distance_A IMPROVES correlation ({abs(r_norm):.3f} > {abs(r_raw):.3f})")
    else:
        print(f"    -> Normalising by distance_A does NOT improve correlation")


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    df, pair_agg = load_data()
    print_summary_stats(pair_agg)
    plot_exploration(pair_agg, sigma_fixed_pairs=None)
