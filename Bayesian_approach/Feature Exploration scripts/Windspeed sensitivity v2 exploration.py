"""
Windspeed Sensitivity Exploration v2: Amplification Effect

REVISED THEORY:
- The power curve slope doesn't directly cause uncertainty
- Instead, it AMPLIFIES windspeed errors into power errors
- |Power error| ≈ slope × |Windspeed error|

TEST:
- Compute windspeed_error = |Mean_windspeed_predicted - Mean_windspeed_self|
- Compute amplified_error = slope × windspeed_error
- Test if amplified_error correlates with |EY_deviation|

ALTERNATIVE FRAMINGS:
1. Does slope amplify windspeed errors? (slope × ws_error vs |EY|)
2. Does the slope moderate the ws_error → EY_error relationship?
3. Normalize EY deviation by energy weight


Questions i need to answer:
Is this script interpolating between the windspeed values?
How representative is the power curve i have fed the script?
The ws_error is the biggest indicator of EY uncertainty, which is a no-brainer.
But this error won't be available to me in later stages, how can i make this information useful?
Crux of the matter: is there a way to include the magnitude of the windspeed parametric itself as an indicator of uncertainty?
Secondly, what Claude said about my theory being completely wrong with most stability around 8-9 m/s. Talk to MRR about this, might get some idea.  


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from io import StringIO

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# Power curve data
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
    """Load power curve and create interpolation functions."""
    df_pc = pd.read_csv(StringIO(POWER_CURVE_DATA.strip()))

    abs_slope_interp = interp1d(
        df_pc["Windspeed"].values,
        np.abs(df_pc["Slope"].values),
        kind="linear",
        bounds_error=False,
        fill_value=(np.abs(df_pc["Slope"].iloc[0]), np.abs(df_pc["Slope"].iloc[-1]))
    )

    return df_pc, abs_slope_interp


def load_and_prepare_data(path, abs_slope_interp):
    """Load data and compute derived features."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    # Check required columns
    required = ["Mean_windspeed_predicted", "Mean_windspeed_self",
                "EY_deviation_sector_frac", "weight_energy", "weight_energy_predicted"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠ Missing columns: {missing}")
        return None

    df = df.dropna(subset=required).copy()
    print(f"\nValid observations: {len(df)}")

    # =========================================
    # COMPUTE DERIVED FEATURES
    # =========================================

    # 1. Windspeed error (absolute)
    df["ws_error"] = np.abs(df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])

    # 2. Windspeed error (signed) - for checking directionality
    df["ws_error_signed"] = df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"]

    # 3. Slope at predicted windspeed
    df["slope_at_ws"] = abs_slope_interp(df["Mean_windspeed_predicted"])

    # 4. Slope at true windspeed (alternative)
    df["slope_at_ws_true"] = abs_slope_interp(df["Mean_windspeed_self"])

    # 5. Average slope (between predicted and true)
    df["slope_avg"] = (df["slope_at_ws"] + df["slope_at_ws_true"]) / 2

    # 6. AMPLIFIED ERROR = slope × windspeed_error
    df["amplified_error"] = df["slope_at_ws"] * df["ws_error"]
    df["amplified_error_avg"] = df["slope_avg"] * df["ws_error"]

    # 7. Normalized amplified error (by max power for scale)
    df["amplified_error_norm"] = df["amplified_error"] / 6200  # Divide by rated power

    # 8. EY deviation per unit energy weight (normalized)
    df["ey_dev_per_weight"] = df["EY_deviation_sector_frac"] / np.clip(df["weight_energy"], 1e-6, None)

    print("\nDerived features computed:")
    print(f"  ws_error: mean={df['ws_error'].mean():.3f} m/s, max={df['ws_error'].max():.3f} m/s")
    print(f"  slope_at_ws: mean={df['slope_at_ws'].mean():.0f}")
    print(f"  amplified_error: mean={df['amplified_error'].mean():.1f}, max={df['amplified_error'].max():.1f}")

    return df


def analyze_amplification_effect(df):
    """Test if slope amplifies windspeed errors."""

    print("\n" + "=" * 70)
    print("TEST 1: AMPLIFICATION EFFECT")
    print("Does slope × ws_error predict |EY_deviation| better than ws_error alone?")
    print("=" * 70)

    y = df["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)

    ws_error = df["ws_error"].values
    slope = df["slope_at_ws"].values
    amplified = df["amplified_error"].values

    # =========================================
    # CORRELATION COMPARISON
    # =========================================

    print("\n" + "-" * 50)
    print("SECTOR-LEVEL CORRELATIONS WITH |EY DEVIATION|")
    print("-" * 50)

    metrics = [
        ("ws_error", ws_error, "Windspeed error alone"),
        ("slope", slope, "Slope alone"),
        ("amplified", amplified, "Slope × ws_error (AMPLIFIED)"),
    ]

    for name, x, label in metrics:
        valid = np.isfinite(x) & np.isfinite(abs_y)
        r, p = stats.pearsonr(x[valid], abs_y[valid])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    # =========================================
    # STRATIFIED ANALYSIS
    # =========================================

    print("\n" + "-" * 50)
    print("STRATIFIED: Does ws_error → |EY| relationship depend on slope?")
    print("-" * 50)

    slope_median = np.median(slope)

    low_slope = slope <= slope_median
    high_slope = slope > slope_median

    r_low, _ = stats.pearsonr(ws_error[low_slope], abs_y[low_slope])
    r_high, _ = stats.pearsonr(ws_error[high_slope], abs_y[high_slope])

    print(f"\n  When slope is LOW (≤{slope_median:.0f}):")
    print(f"    ws_error vs |EY|: r = {r_low:+.4f} (n={low_slope.sum()})")

    print(f"\n  When slope is HIGH (>{slope_median:.0f}):")
    print(f"    ws_error vs |EY|: r = {r_high:+.4f} (n={high_slope.sum()})")

    if abs(r_high) > abs(r_low) and r_high > 0:
        print(f"\n  ✓ SUPPORTS THEORY: ws_error matters MORE when slope is high")
    else:
        print(f"\n  ✗ Does NOT support amplification theory")

    return metrics


def analyze_pair_level(df):
    """Pair-level analysis with different aggregation approaches."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS")
    print("=" * 70)

    required = ["pair_id", "EY_deviation_sector_frac", "weight_energy",
                "weight_energy_predicted", "ws_error", "slope_at_ws", "amplified_error"]

    df_valid = df.dropna(subset=required).copy()

    # =========================================
    # APPROACH A: Energy-weighted aggregation (standard)
    # =========================================

    print("\n" + "-" * 50)
    print("APPROACH A: Energy-weighted aggregation")
    print("-" * 50)

    df_valid["w_ey_dev"] = df_valid["weight_energy"] * df_valid["EY_deviation_sector_frac"]
    df_valid["w_ws_error"] = df_valid["weight_energy_predicted"] * df_valid["ws_error"]
    df_valid["w_slope"] = df_valid["weight_energy_predicted"] * df_valid["slope_at_ws"]
    df_valid["w_amplified"] = df_valid["weight_energy_predicted"] * df_valid["amplified_error"]

    pair_agg_A = df_valid.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
        ws_error=("w_ws_error", "sum"),
        slope=("w_slope", "sum"),
        amplified_error=("w_amplified", "sum"),
    ).reset_index()

    y = np.abs(pair_agg_A["ey_deviation"].values)

    for col in ["ws_error", "slope", "amplified_error"]:
        x = pair_agg_A[col].values
        r, p = stats.pearsonr(x, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col}: r = {r:+.4f} {sig}")

    # =========================================
    # APPROACH B: Weight by TRUE energy (weight_energy)
    # =========================================

    print("\n" + "-" * 50)
    print("APPROACH B: Weight by TRUE energy (weight_energy)")
    print("-" * 50)

    df_valid["w_ws_error_true"] = df_valid["weight_energy"] * df_valid["ws_error"]
    df_valid["w_slope_true"] = df_valid["weight_energy"] * df_valid["slope_at_ws"]
    df_valid["w_amplified_true"] = df_valid["weight_energy"] * df_valid["amplified_error"]

    pair_agg_B = df_valid.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
        ws_error=("w_ws_error_true", "sum"),
        slope=("w_slope_true", "sum"),
        amplified_error=("w_amplified_true", "sum"),
    ).reset_index()

    y = np.abs(pair_agg_B["ey_deviation"].values)

    for col in ["ws_error", "slope", "amplified_error"]:
        x = pair_agg_B[col].values
        r, p = stats.pearsonr(x, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col}: r = {r:+.4f} {sig}")

    # =========================================
    # APPROACH C: Simple mean (unweighted)
    # =========================================

    print("\n" + "-" * 50)
    print("APPROACH C: Simple mean (unweighted)")
    print("-" * 50)

    pair_agg_C = df_valid.groupby("pair_id").agg(
        ey_deviation=("EY_deviation_sector_frac", "mean"),
        ws_error=("ws_error", "mean"),
        slope=("slope_at_ws", "mean"),
        amplified_error=("amplified_error", "mean"),
    ).reset_index()

    y = np.abs(pair_agg_C["ey_deviation"].values)

    for col in ["ws_error", "slope", "amplified_error"]:
        x = pair_agg_C[col].values
        r, p = stats.pearsonr(x, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col}: r = {r:+.4f} {sig}")

    return pair_agg_A, pair_agg_B, pair_agg_C


def analyze_low_wind_effect(df):
    """
    Test alternative theory: Low-wind sectors have more uncertainty
    due to cut-in threshold effects, not slope amplification.
    """

    print("\n" + "=" * 70)
    print("TEST 2: LOW-WIND THRESHOLD EFFECT")
    print("Do sectors near cut-in (3-5 m/s) have more uncertainty?")
    print("=" * 70)

    abs_y = np.abs(df["EY_deviation_sector_frac"].values)
    ws = df["Mean_windspeed_predicted"].values

    # Define wind speed regions
    near_cutin = ws <= 5  # Near cut-in
    mid_range = (ws > 5) & (ws <= 9)  # High slope region
    near_rated = ws > 9  # Near rated/beyond

    print(f"\n  Near cut-in (≤5 m/s): n={near_cutin.sum()}")
    print(f"    |EY| mean = {abs_y[near_cutin].mean():.4f}")
    print(f"    |EY| std  = {abs_y[near_cutin].std():.4f}")

    print(f"\n  Mid-range (5-9 m/s): n={mid_range.sum()}")
    print(f"    |EY| mean = {abs_y[mid_range].mean():.4f}")
    print(f"    |EY| std  = {abs_y[mid_range].std():.4f}")

    print(f"\n  Near rated (>9 m/s): n={near_rated.sum()}")
    print(f"    |EY| mean = {abs_y[near_rated].mean():.4f}")
    print(f"    |EY| std  = {abs_y[near_rated].std():.4f}")

    # Create a "distance from optimal" feature
    # Optimal = 7-9 m/s where model is likely best calibrated
    optimal_ws = 8.0
    df["ws_distance_from_optimal"] = np.abs(ws - optimal_ws)

    dist_optimal = df["ws_distance_from_optimal"].values
    r, p = stats.pearsonr(dist_optimal, abs_y)

    print(f"\n  Distance from optimal (8 m/s) vs |EY|:")
    print(f"    Pearson r = {r:+.4f} (p = {p:.4f})")

    if r > 0 and p < 0.05:
        print("  ✓ SUPPORTS: Sectors far from optimal windspeed have more uncertainty")

    return df


def analyze_normalized_by_energy(df):
    """
    Test: Normalize EY deviation by energy contribution.
    High-energy sectors naturally have smaller relative errors?
    """

    print("\n" + "=" * 70)
    print("TEST 3: ENERGY-NORMALIZED ANALYSIS")
    print("Does high energy contribution mask uncertainty?")
    print("=" * 70)

    # EY deviation normalized by energy weight
    ey_dev_norm = np.abs(df["EY_deviation_sector_frac"]) / np.clip(df["weight_energy"], 1e-6, None)

    slope = df["slope_at_ws"].values
    ws = df["Mean_windspeed_predicted"].values

    # Correlation with slope
    valid = np.isfinite(ey_dev_norm) & np.isfinite(slope) & (ey_dev_norm < np.percentile(ey_dev_norm, 99))
    r, p = stats.pearsonr(slope[valid], ey_dev_norm[valid])

    print(f"\n  Slope vs |EY_dev| / energy_weight:")
    print(f"    Pearson r = {r:+.4f} (p = {p:.4f})")

    # Correlation with windspeed
    r2, p2 = stats.pearsonr(ws[valid], ey_dev_norm[valid])
    print(f"\n  Windspeed vs |EY_dev| / energy_weight:")
    print(f"    Pearson r = {r2:+.4f} (p = {p2:.4f})")


def create_visualizations(df, pair_agg):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_y = np.abs(df["EY_deviation_sector_frac"].values)

    # 1. ws_error vs |EY| (sector)
    ax = axes[0, 0]
    x = df["ws_error"].values
    ax.scatter(x, abs_y, alpha=0.3, s=10)
    r, _ = stats.pearsonr(x, abs_y)
    ax.set_xlabel("Windspeed error (m/s)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: ws_error vs |EY|\nr = {r:.3f}")

    # 2. amplified_error vs |EY| (sector)
    ax = axes[0, 1]
    x = df["amplified_error"].values
    ax.scatter(x, abs_y, alpha=0.3, s=10)
    r, _ = stats.pearsonr(x, abs_y)
    ax.set_xlabel("Amplified error (slope × ws_error)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: amplified vs |EY|\nr = {r:.3f}")

    # 3. Windspeed vs |EY| (sector)
    ax = axes[0, 2]
    x = df["Mean_windspeed_predicted"].values
    ax.scatter(x, abs_y, alpha=0.3, s=10)
    r, _ = stats.pearsonr(x, abs_y)
    ax.set_xlabel("Mean windspeed (m/s)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: windspeed vs |EY|\nr = {r:.3f}")

    # 4. Pair-level: ws_error
    ax = axes[1, 0]
    y_pair = np.abs(pair_agg["ey_deviation"].values)
    x = pair_agg["ws_error"].values
    ax.scatter(x, y_pair, alpha=0.6)
    r, _ = stats.pearsonr(x, y_pair)
    slope, intercept = np.polyfit(x, y_pair, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Windspeed error (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: ws_error vs |EY|\nr = {r:.3f}")

    # 5. Pair-level: amplified_error
    ax = axes[1, 1]
    x = pair_agg["amplified_error"].values
    ax.scatter(x, y_pair, alpha=0.6)
    r, _ = stats.pearsonr(x, y_pair)
    slope, intercept = np.polyfit(x, y_pair, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Amplified error (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: amplified vs |EY|\nr = {r:.3f}")

    # 6. Pair-level: slope
    ax = axes[1, 2]
    x = pair_agg["slope"].values
    ax.scatter(x, y_pair, alpha=0.6)
    r, _ = stats.pearsonr(x, y_pair)
    slope_fit, intercept = np.polyfit(x, y_pair, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope_fit * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Slope (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: slope vs |EY|\nr = {r:.3f}")

    plt.tight_layout()
    plt.savefig("windspeed_sensitivity_v2_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: windspeed_sensitivity_v2_exploration.png")


def print_summary():
    """Print final summary and recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    print("""
KEY FINDINGS:

1. SLOPE ALONE shows NEGATIVE correlation with |EY|
   → High-slope sectors (7-10 m/s) actually have SMALLER errors
   → Likely because the model is well-calibrated for core wind speeds

2. AMPLIFICATION EFFECT (slope × ws_error):
   → Test if this captures uncertainty better than slope or ws_error alone
   → If amplified_error has positive correlation, the theory is validated

3. ALTERNATIVE HYPOTHESIS:
   → Sectors far from "optimal" windspeed (8 m/s) may have more uncertainty
   → This would capture both low-wind (cut-in effects) and high-wind (derating) issues

RECOMMENDATION:
   Based on results, consider:
   - If amplified_error works: Add slope × ws_error to sigma model
   - If distance_from_optimal works: Add |windspeed - 8| to sigma model
   - If nothing works: The power curve slope may not be a useful uncertainty driver
""")


def main():
    print("=" * 70)
    print("WINDSPEED SENSITIVITY EXPLORATION v2")
    print("Testing Amplification Effect and Alternative Framings")
    print("=" * 70)

    # Load power curve
    df_pc, abs_slope_interp = load_power_curve()

    # Load and prepare data
    df = load_and_prepare_data(sector_model_path, abs_slope_interp)

    if df is None:
        return

    # Test amplification effect
    analyze_amplification_effect(df)

    # Pair-level with different aggregation approaches
    pair_agg_A, pair_agg_B, pair_agg_C = analyze_pair_level(df)

    # Test low-wind effect
    df = analyze_low_wind_effect(df)

    # Test energy-normalized analysis
    analyze_normalized_by_energy(df)

    # Visualizations
    create_visualizations(df, pair_agg_A)

    # Summary
    print_summary()

    return df, pair_agg_A


if __name__ == "__main__":
    result = main()