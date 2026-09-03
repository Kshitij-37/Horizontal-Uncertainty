"""
Exploration: Windspeed Sensitivity (Power Curve Slope) Feature

Theory:
- Wind turbine power production is nonlinear with wind speed
- The slope of the power curve (dP/dv) varies with wind speed:
  - Low slope near cut-in (~3 m/s) and rated power (~12-16 m/s)
  - High slope in the middle range (~7-10 m/s)
- When mean windspeed is in a HIGH SLOPE region:
  - Small uncertainty in windspeed → Large uncertainty in power
- When mean windspeed is in a LOW SLOPE region:
  - Small uncertainty in windspeed → Small uncertainty in power

Feature: |slope| at the Mean_windspeed_predicted value
- Higher |slope| = more sensitive to windspeed errors = more uncertainty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# Power curve data (user-provided)
# Columns: Windspeed (m/s), Power (kW), Slope (kW per 0.5 m/s step)
POWER_CURVE_DATA = """
Windspeed,Power,Slope
3,34,232
3.5,150,284
4,292,350
4.5,467,418
5,676,502
5.5,927,604
6,1229,710
6.5,1584,832
7,2000,952
7.5,2476,1082
8,3017,1218
8.5,3626,1316
9,4284,1266
9.5,4917,1132
10,5483,798
10.5,5882,464
11,6114,124
11.5,6176,42
12,6197,6
12.5,6200,0
13,6200,0
13.5,6200,0
14,6200,0
14.5,6200,0
15,6200,0
15.5,6200,0
16,6200,0
16.5,6200,-28
17,6186,-218
17.5,6077,-448
18,5853,-526
18.5,5590,-484
19,5348,-506
19.5,5095,-540
20,4825,-574
20.5,4538,-574
21,4251,-594
21.5,3954,-580
22,3664,-594
22.5,3367,-606
23,3064,-602
23.5,2763,-624
24,2451,102.125
"""


def load_power_curve():
    """Load and process the power curve data."""
    from io import StringIO

    df_pc = pd.read_csv(StringIO(POWER_CURVE_DATA.strip()))

    # Convert slope from "per 0.5 m/s step" to "per 1 m/s" for easier interpretation
    # Original slope = (P2 - P1) / 0.5 m/s
    # We want slope per m/s, so multiply by 2? No wait, let's keep it as-is for now
    # Actually the slope column is already dP/dv where dv = 0.5, so slope_per_ms = slope / 0.5 = slope * 2
    df_pc["slope_per_ms"] = df_pc["Slope"] * 2  # kW per m/s

    print("=" * 70)
    print("POWER CURVE DATA")
    print("=" * 70)

    print(f"\nWindspeed range: {df_pc['Windspeed'].min():.1f} - {df_pc['Windspeed'].max():.1f} m/s")
    print(f"Power range: {df_pc['Power'].min():.0f} - {df_pc['Power'].max():.0f} kW")
    print(f"Slope range: {df_pc['Slope'].min():.0f} - {df_pc['Slope'].max():.0f} kW/(0.5 m/s)")
    print(f"         or: {df_pc['slope_per_ms'].min():.0f} - {df_pc['slope_per_ms'].max():.0f} kW/(m/s)")

    # Find key wind speeds
    max_slope_idx = df_pc["Slope"].abs().idxmax()
    print(f"\nMax |slope| at windspeed: {df_pc.loc[max_slope_idx, 'Windspeed']:.1f} m/s")
    print(f"  Slope = {df_pc.loc[max_slope_idx, 'Slope']:.0f} kW/(0.5 m/s)")

    # Find rated power region (where slope ≈ 0)
    rated_mask = df_pc["Slope"].abs() < 10
    if rated_mask.any():
        rated_speeds = df_pc.loc[rated_mask, "Windspeed"]
        print(f"Rated power region (|slope| < 10): {rated_speeds.min():.1f} - {rated_speeds.max():.1f} m/s")

    return df_pc


def create_slope_lookup(df_pc):
    """Create interpolation function for slope lookup."""

    # Create interpolation function
    # Use absolute slope for uncertainty (both positive and negative slopes indicate sensitivity)
    slope_interp = interp1d(
        df_pc["Windspeed"].values,
        df_pc["Slope"].values,  # Keep signed for potential directional analysis
        kind="linear",
        bounds_error=False,
        fill_value=(df_pc["Slope"].iloc[0], df_pc["Slope"].iloc[-1])  # Extrapolate with edge values
    )

    abs_slope_interp = interp1d(
        df_pc["Windspeed"].values,
        np.abs(df_pc["Slope"].values),  # Absolute slope for sigma model
        kind="linear",
        bounds_error=False,
        fill_value=(np.abs(df_pc["Slope"].iloc[0]), np.abs(df_pc["Slope"].iloc[-1]))
    )

    return slope_interp, abs_slope_interp


def visualize_power_curve(df_pc):
    """Visualize the power curve and its slope."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Power curve
    ax = axes[0]
    ax.plot(df_pc["Windspeed"], df_pc["Power"], 'b-', linewidth=2)
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power Curve")
    ax.grid(True, alpha=0.3)
    ax.axhline(6200, color='red', linestyle='--', alpha=0.5, label='Rated Power')
    ax.legend()

    # 2. Slope (signed)
    ax = axes[1]
    ax.plot(df_pc["Windspeed"], df_pc["Slope"], 'g-', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Slope (kW per 0.5 m/s)")
    ax.set_title("Power Curve Slope (Signed)")
    ax.grid(True, alpha=0.3)

    # Highlight high sensitivity region
    high_sens = df_pc[df_pc["Slope"].abs() > 800]
    if len(high_sens) > 0:
        ax.axvspan(high_sens["Windspeed"].min(), high_sens["Windspeed"].max(),
                   alpha=0.2, color='red', label='High sensitivity')
        ax.legend()

    # 3. Absolute slope (for sigma model)
    ax = axes[2]
    ax.plot(df_pc["Windspeed"], np.abs(df_pc["Slope"]), 'r-', linewidth=2)
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("|Slope| (kW per 0.5 m/s)")
    ax.set_title("|Slope| - Sensitivity for Sigma Model")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("power_curve_analysis.png", dpi=150)
    plt.close()
    print("\nSaved: power_curve_analysis.png")


def load_and_analyze_data(sector_model_path, abs_slope_interp):
    """Load sector data and compute windspeed sensitivity."""

    print("\n" + "=" * 70)
    print("LOADING SECTOR DATA")
    print("=" * 70)

    df = pd.read_excel(sector_model_path)

    # Check for Mean_windspeed_predicted column
    ws_cols = [c for c in df.columns if 'windspeed' in c.lower() or 'wind_speed' in c.lower()]
    print(f"\nWindspeed-related columns found:")
    for col in ws_cols:
        print(f"  {col}")

    if "Mean_windspeed_predicted" not in df.columns:
        # Try alternative names
        alt_names = ["mean_windspeed_predicted", "Mean_Windspeed_Predicted",
                     "meanwindspeed_predicted", "MeanWindspeed_predicted"]
        for alt in alt_names:
            if alt in df.columns:
                df["Mean_windspeed_predicted"] = df[alt]
                break
        else:
            print("\n⚠ Mean_windspeed_predicted column not found!")
            print("Please check column names and update the script.")
            return None

    # Compute windspeed sensitivity (absolute slope at mean windspeed)
    df["windspeed_sensitivity"] = abs_slope_interp(df["Mean_windspeed_predicted"])

    print(f"\nMean_windspeed_predicted statistics:")
    ws = df["Mean_windspeed_predicted"].dropna()
    print(f"  Range: [{ws.min():.1f}, {ws.max():.1f}] m/s")
    print(f"  Mean: {ws.mean():.1f} m/s")
    print(f"  Median: {ws.median():.1f} m/s")

    print(f"\nWindspeed sensitivity (|slope|) statistics:")
    sens = df["windspeed_sensitivity"].dropna()
    print(f"  Range: [{sens.min():.0f}, {sens.max():.0f}]")
    print(f"  Mean: {sens.mean():.0f}")
    print(f"  Median: {sens.median():.0f}")

    return df


def analyze_sector_level(df):
    """Test windspeed sensitivity at sector level."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    required = ["EY_deviation_sector_frac", "windspeed_sensitivity", "weight_energy_predicted"]
    df_valid = df.dropna(subset=required).copy()

    print(f"\nValid sector observations: {len(df_valid)}")

    y = df_valid["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)
    sensitivity = df_valid["windspeed_sensitivity"].values

    # =========================================
    # CORRELATION WITH |EY DEVIATION|
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATION WITH |EY DEVIATION|")
    print("-" * 50)

    r, p = stats.pearsonr(sensitivity, abs_y)
    rho, _ = stats.spearmanr(sensitivity, abs_y)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print(f"\nWindspeed sensitivity vs |EY deviation|:")
    print(f"  Pearson r  = {r:+.4f} {sig}")
    print(f"  Spearman ρ = {rho:+.4f}")
    print(f"  p-value    = {p:.4f}")

    if r > 0 and p < 0.05:
        print("\n  ✓ SIGNIFICANT: Higher sensitivity → larger errors")
        print("  → ADD windspeed_sensitivity to sigma model")
    else:
        print("\n  → No significant relationship at sector level")
        print("  → May still show effect at pair level")

    # =========================================
    # STRATIFIED ANALYSIS
    # =========================================

    print("\n" + "-" * 50)
    print("STRATIFIED ANALYSIS BY SENSITIVITY")
    print("-" * 50)

    # Split into low/medium/high sensitivity
    terciles = np.percentile(sensitivity, [33, 67])

    low_mask = sensitivity <= terciles[0]
    med_mask = (sensitivity > terciles[0]) & (sensitivity <= terciles[1])
    high_mask = sensitivity > terciles[1]

    print(f"\n  Low sensitivity (≤{terciles[0]:.0f}):")
    print(f"    n = {low_mask.sum()}, |EY| mean = {abs_y[low_mask].mean():.4f}, std = {abs_y[low_mask].std():.4f}")

    print(f"\n  Medium sensitivity ({terciles[0]:.0f} - {terciles[1]:.0f}):")
    print(f"    n = {med_mask.sum()}, |EY| mean = {abs_y[med_mask].mean():.4f}, std = {abs_y[med_mask].std():.4f}")

    print(f"\n  High sensitivity (>{terciles[1]:.0f}):")
    print(f"    n = {high_mask.sum()}, |EY| mean = {abs_y[high_mask].mean():.4f}, std = {abs_y[high_mask].std():.4f}")

    # Levene's test
    stat, p_levene = stats.levene(abs_y[low_mask], abs_y[med_mask], abs_y[high_mask])
    print(f"\n  Levene's test for equal variances: p = {p_levene:.4f}")

    if p_levene < 0.05:
        print("  → SIGNIFICANT: Error spread differs by sensitivity level")
    else:
        print("  → No significant difference in spread")

    return df_valid


def analyze_pair_level(df):
    """Aggregate to pair level and analyze."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted)")
    print("=" * 70)

    required = ["pair_id", "EY_deviation_sector_frac", "weight_energy",
                "weight_energy_predicted", "windspeed_sensitivity"]

    df_valid = df.dropna(subset=required).copy()

    # Energy-weighted aggregation
    df_valid["w_ey_dev"] = df_valid["weight_energy"] * df_valid["EY_deviation_sector_frac"]
    df_valid["w_sensitivity"] = df_valid["weight_energy_predicted"] * df_valid["windspeed_sensitivity"]

    pair_agg = df_valid.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
        windspeed_sensitivity=("w_sensitivity", "sum"),
    ).reset_index()

    print(f"\nPairs: {len(pair_agg)}")

    y = pair_agg["ey_deviation"].values
    abs_y = np.abs(y)
    sensitivity = pair_agg["windspeed_sensitivity"].values

    print("\n" + "-" * 50)
    print("PAIR-LEVEL CORRELATION")
    print("-" * 50)

    r, p = stats.pearsonr(sensitivity, abs_y)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print(f"\nWindspeed sensitivity vs |EY deviation|:")
    print(f"  Pearson r = {r:+.4f} {sig}")
    print(f"  p-value   = {p:.4f}")

    return pair_agg


def create_visualizations(df_sector, pair_agg):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Distribution of windspeed sensitivity
    ax = axes[0]
    sensitivity = df_sector["windspeed_sensitivity"].values
    ax.hist(sensitivity, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Windspeed Sensitivity (|slope|)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Windspeed Sensitivity")

    # 2. Sector level: sensitivity vs |EY|
    ax = axes[1]
    x = df_sector["windspeed_sensitivity"].values
    y = np.abs(df_sector["EY_deviation_sector_frac"].values)

    ax.scatter(x, y, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(y)
    r, _ = stats.pearsonr(x[valid], y[valid])

    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("Windspeed Sensitivity (|slope|)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: Sensitivity vs |EY|\nr = {r:.3f}")

    # 3. Pair level: sensitivity vs |EY|
    ax = axes[2]
    x = pair_agg["windspeed_sensitivity"].values
    y = np.abs(pair_agg["ey_deviation"].values)

    ax.scatter(x, y, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(y)
    r, _ = stats.pearsonr(x[valid], y[valid])

    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("Windspeed Sensitivity (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: Sensitivity vs |EY|\nr = {r:.3f}")

    plt.tight_layout()
    plt.savefig("windspeed_sensitivity_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: windspeed_sensitivity_exploration.png")


def print_recommendations(sector_r, pair_r):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
WINDSPEED SENSITIVITY FEATURE SUMMARY:

  Sector-level correlation: r = {sector_r:+.4f}
  Pair-level correlation:   r = {pair_r:+.4f}

RECOMMENDATION:
""")

    if pair_r > 0.1:
        print("  ✓ ADD windspeed_sensitivity to sigma model")
        print("  Formula: γ_ws_sens × |slope_at_mean_windspeed|")
        print("  Physical interpretation: Sectors with high power curve slope")
        print("  are more sensitive to windspeed errors → more uncertainty")
    elif pair_r > 0:
        print("  ? Weak positive effect - consider adding, but may not be significant")
    else:
        print("  ✗ No positive relationship found - do not add")


def main():
    print("=" * 70)
    print("WINDSPEED SENSITIVITY EXPLORATION")
    print("(Power Curve Slope as Uncertainty Driver)")
    print("=" * 70)

    # Load and analyze power curve
    df_pc = load_power_curve()
    slope_interp, abs_slope_interp = create_slope_lookup(df_pc)
    visualize_power_curve(df_pc)

    # Load sector data
    df = load_and_analyze_data(sector_model_path, abs_slope_interp)

    if df is None:
        return

    # Sector-level analysis
    df_sector = analyze_sector_level(df)

    # Get sector-level correlation for summary
    y_sector = np.abs(df_sector["EY_deviation_sector_frac"].values)
    sens_sector = df_sector["windspeed_sensitivity"].values
    r_sector, _ = stats.pearsonr(sens_sector, y_sector)

    # Pair-level analysis
    pair_agg = analyze_pair_level(df)

    # Get pair-level correlation for summary
    y_pair = np.abs(pair_agg["ey_deviation"].values)
    sens_pair = pair_agg["windspeed_sensitivity"].values
    r_pair, _ = stats.pearsonr(sens_pair, y_pair)

    # Visualizations
    create_visualizations(df_sector, pair_agg)

    # Recommendations
    print_recommendations(r_sector, r_pair)

    return df, df_sector, pair_agg, abs_slope_interp


if __name__ == "__main__":
    result = main()
    if result:
        df, df_sector, pair_agg, abs_slope_interp = result