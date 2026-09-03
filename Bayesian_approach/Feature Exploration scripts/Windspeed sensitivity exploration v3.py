"""
Windspeed Sensitivity Exploration

Hypothesis:
    When mean windspeed is in the steep power curve region (7-9 m/s),
    small windspeed errors get amplified into larger energy yield errors.

Approach:
    - Use Mean_windspeed_predicted (deployment-ready)
    - 7-9 m/s identified as steep slope region across multiple turbines
    - Calculate energy-weighted exposure to this region

Metrics to test:
    1. steep_fraction: Fraction of energy from sectors with WS in 7-9 m/s
    2. steep_energy_weight: Sum of energy weights for sectors in steep region
    3. distance_from_steep: Energy-weighted distance from 8 m/s (center of steep)
    4. in_steep_binary: Does the pair have >50% energy in steep region?

Also test sensitivity to range definition (6-10 m/s as alternative).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

# Steep region definition
STEEP_LOW = 7.0
STEEP_HIGH = 9.0
STEEP_CENTER = 8.0

# Alternative range for sensitivity check
ALT_STEEP_LOW = 6.0
ALT_STEEP_HIGH = 10.0


def load_and_prepare_data(path):
    """Load data and compute deviations."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("WINDSPEED SENSITIVITY EXPLORATION")
    print("=" * 70)
    print(f"\nSteep region definition: {STEEP_LOW}-{STEEP_HIGH} m/s")
    print(f"Alternative range: {ALT_STEEP_LOW}-{ALT_STEEP_HIGH} m/s")

    required = [
        "pair_id", "sector_name",
        "EY_deviation_sector_frac",
        "weight_energy_predicted",
        "Mean_windspeed_predicted",
        "Mean_windspeed_self",
    ]

    df = df.dropna(subset=required).copy()

    # Compute WS deviation for reference
    ws_actual = df["Mean_windspeed_self"].values
    ws_pred = df["Mean_windspeed_predicted"].values
    ws_denom = np.maximum(ws_actual, 3.0)
    df["WS_deviation_rel"] = np.abs(ws_pred - ws_actual) / ws_denom

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    # Check windspeed distribution
    print(f"\n  Mean_windspeed_predicted distribution:")
    print(f"    Range: [{df['Mean_windspeed_predicted'].min():.1f}, {df['Mean_windspeed_predicted'].max():.1f}] m/s")
    print(f"    Mean: {df['Mean_windspeed_predicted'].mean():.1f} m/s")
    print(f"    Median: {df['Mean_windspeed_predicted'].median():.1f} m/s")

    # Count sectors in steep region
    in_steep = (df['Mean_windspeed_predicted'] >= STEEP_LOW) & (df['Mean_windspeed_predicted'] <= STEEP_HIGH)
    print(
        f"\n  Sectors in steep region ({STEEP_LOW}-{STEEP_HIGH} m/s): {in_steep.sum()} / {len(df)} ({in_steep.mean() * 100:.1f}%)")

    return df


def compute_pair_features(df):
    """Compute windspeed sensitivity features at pair level."""

    print("\n" + "=" * 70)
    print("COMPUTING PAIR-LEVEL FEATURES")
    print("=" * 70)

    results = []

    for pair_id, group in df.groupby("pair_id"):
        # Weights
        weights = group["weight_energy_predicted"].values
        weights = weights / weights.sum()

        # Windspeeds
        ws_pred = group["Mean_windspeed_predicted"].values

        # =========================================
        # METRIC 1: Steep fraction (7-9 m/s)
        # =========================================
        in_steep = (ws_pred >= STEEP_LOW) & (ws_pred <= STEEP_HIGH)
        steep_fraction = np.sum(weights[in_steep])

        # =========================================
        # METRIC 2: Alternative steep fraction (6-10 m/s)
        # =========================================
        in_steep_alt = (ws_pred >= ALT_STEEP_LOW) & (ws_pred <= ALT_STEEP_HIGH)
        steep_fraction_alt = np.sum(weights[in_steep_alt])

        # =========================================
        # METRIC 3: Distance from steep center (8 m/s)
        # =========================================
        # Energy-weighted distance from 8 m/s
        distance_from_steep = np.sum(weights * np.abs(ws_pred - STEEP_CENTER))

        # =========================================
        # METRIC 4: Energy-weighted mean WS
        # =========================================
        ws_mean_weighted = np.sum(weights * ws_pred)

        # =========================================
        # METRIC 5: "Closeness" to steep (inverse distance)
        # =========================================
        # Higher = closer to steep region center
        closeness_to_steep = 1.0 / (distance_from_steep + 0.5)

        # =========================================
        # METRIC 6: Proximity score
        # =========================================
        # How close is each sector to the steep region?
        # 0 if outside, 1 if at center (8 m/s)
        def proximity_score(ws):
            if ws < STEEP_LOW:
                return max(0, 1 - (STEEP_LOW - ws) / 2)  # Ramp up from 5 m/s
            elif ws > STEEP_HIGH:
                return max(0, 1 - (ws - STEEP_HIGH) / 2)  # Ramp down to 11 m/s
            else:
                # In steep region: closer to center = higher score
                return 1 - abs(ws - STEEP_CENTER) / (STEEP_HIGH - STEEP_CENTER)

        proximity_scores = np.array([proximity_score(w) for w in ws_pred])
        weighted_proximity = np.sum(weights * proximity_scores)

        # =========================================
        # TARGETS: Aggregate deviations
        # =========================================
        ey_dev = np.sum(weights * group["EY_deviation_sector_frac"].values)
        ws_dev = np.sum(weights * group["WS_deviation_rel"].values)

        results.append({
            "pair_id": pair_id,
            # Metrics
            "steep_fraction": steep_fraction,
            "steep_fraction_alt": steep_fraction_alt,
            "distance_from_steep": distance_from_steep,
            "ws_mean_weighted": ws_mean_weighted,
            "closeness_to_steep": closeness_to_steep,
            "weighted_proximity": weighted_proximity,
            # Targets
            "ey_deviation": ey_dev,
            "ws_deviation": ws_dev,
        })

    pair_df = pd.DataFrame(results)

    # =========================================
    # PRINT STATISTICS
    # =========================================

    print("\n" + "-" * 50)
    print("FEATURE STATISTICS")
    print("-" * 50)

    features = [
        ("steep_fraction", f"Steep fraction ({STEEP_LOW}-{STEEP_HIGH} m/s)"),
        ("steep_fraction_alt", f"Steep fraction alt ({ALT_STEEP_LOW}-{ALT_STEEP_HIGH} m/s)"),
        ("distance_from_steep", "Distance from 8 m/s"),
        ("ws_mean_weighted", "Energy-weighted mean WS"),
        ("closeness_to_steep", "Closeness to steep"),
        ("weighted_proximity", "Weighted proximity score"),
    ]

    for col, label in features:
        vals = pair_df[col]
        print(f"\n  {label}:")
        print(f"    Range: [{vals.min():.3f}, {vals.max():.3f}]")
        print(f"    Mean: {vals.mean():.3f}")
        print(f"    Median: {vals.median():.3f}")

    # Categorize pairs
    high_steep = pair_df["steep_fraction"] > 0.5
    print(f"\n  Pairs with >50% energy in steep region: {high_steep.sum()} / {len(pair_df)}")

    return pair_df


def analyze_correlations(pair_df):
    """Test correlations with |EY| and |WS| deviation."""

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    features = [
        ("steep_fraction", f"Steep fraction ({STEEP_LOW}-{STEEP_HIGH} m/s)"),
        ("steep_fraction_alt", f"Steep fraction alt ({ALT_STEEP_LOW}-{ALT_STEEP_HIGH} m/s)"),
        ("distance_from_steep", "Distance from 8 m/s"),
        ("ws_mean_weighted", "Energy-weighted mean WS"),
        ("closeness_to_steep", "Closeness to steep"),
        ("weighted_proximity", "Weighted proximity score"),
    ]

    print(f"\n  {'Feature':<45} {'vs |EY|':<15} {'vs |WS|':<15}")
    print("  " + "-" * 75)

    results = []

    for col, label in features:
        x = pair_df[col].values

        r_ey, p_ey = stats.pearsonr(x, abs_ey)
        r_ws, p_ws = stats.pearsonr(x, abs_ws)

        sig_ey = "**" if p_ey < 0.01 else "*" if p_ey < 0.05 else ""
        sig_ws = "**" if p_ws < 0.01 else "*" if p_ws < 0.05 else ""

        results.append({
            "Feature": label,
            "Column": col,
            "r_EY": r_ey,
            "p_EY": p_ey,
            "r_WS": r_ws,
            "p_WS": p_ws,
        })

        print(f"  {label:<45} r={r_ey:+.3f} {sig_ey:<3} r={r_ws:+.3f} {sig_ws:<3}")

    return pd.DataFrame(results)


def analyze_by_category(pair_df):
    """Stratified analysis by steep exposure."""

    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # Split by steep fraction median
    median_steep = pair_df["steep_fraction"].median()
    low_steep = pair_df["steep_fraction"] <= median_steep
    high_steep = pair_df["steep_fraction"] > median_steep

    print(f"\n  Median steep fraction: {median_steep:.3f}")
    print(f"\n  Low steep exposure (≤{median_steep:.3f}): n={low_steep.sum()}")
    print(f"    Mean |EY|: {abs_ey[low_steep].mean():.4f} ({abs_ey[low_steep].mean() * 100:.2f}%)")
    print(f"    Mean |WS|: {abs_ws[low_steep].mean():.4f} ({abs_ws[low_steep].mean() * 100:.2f}%)")

    print(f"\n  High steep exposure (>{median_steep:.3f}): n={high_steep.sum()}")
    print(f"    Mean |EY|: {abs_ey[high_steep].mean():.4f} ({abs_ey[high_steep].mean() * 100:.2f}%)")
    print(f"    Mean |WS|: {abs_ws[high_steep].mean():.4f} ({abs_ws[high_steep].mean() * 100:.2f}%)")

    # T-test
    if high_steep.sum() > 5 and low_steep.sum() > 5:
        t_ey, p_ey = stats.ttest_ind(abs_ey[high_steep], abs_ey[low_steep])
        t_ws, p_ws = stats.ttest_ind(abs_ws[high_steep], abs_ws[low_steep])

        print(f"\n  T-test |EY|: t={t_ey:.2f}, p={p_ey:.4f} {'*' if p_ey < 0.05 else ''}")
        print(f"  T-test |WS|: t={t_ws:.2f}, p={p_ws:.4f} {'*' if p_ws < 0.05 else ''}")

        ratio_ey = abs_ey[high_steep].mean() / abs_ey[low_steep].mean()
        print(f"\n  Ratio (high/low): {ratio_ey:.2f}× for |EY|")


def check_redundancy(pair_df):
    """Check correlation with existing sigma drivers we'd need to load."""

    print("\n" + "=" * 70)
    print("RELATIONSHIP WITH WINDSPEED")
    print("=" * 70)

    # Check relationship between metrics and mean windspeed
    ws_mean = pair_df["ws_mean_weighted"].values
    steep_frac = pair_df["steep_fraction"].values

    r, p = stats.pearsonr(ws_mean, steep_frac)
    print(f"\n  ws_mean_weighted vs steep_fraction: r={r:+.3f}")
    print(f"  → {'Strong' if abs(r) > 0.5 else 'Moderate' if abs(r) > 0.3 else 'Weak'} relationship")

    # Check if steep fraction is just a proxy for windspeed being ~8 m/s
    print(f"\n  Interpretation:")
    print(f"    steep_fraction is high when most energy comes from 7-9 m/s winds")
    print(f"    This happens when mean WS is in that range")
    print(f"    But also depends on WS distribution (could have bimodal wind)")


def create_visualizations(pair_df, save_plots=False):
    """Create diagnostic plots (optional - disabled by default to save IDE resources)."""

    if not save_plots:
        print("\n  [Plots disabled - set save_plots=True in main() to enable]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # 1. Steep fraction vs |EY|
    ax = axes[0, 0]
    ax.scatter(pair_df["steep_fraction"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["steep_fraction"], abs_ey)
    ax.set_xlabel(f"Steep Fraction ({STEEP_LOW}-{STEEP_HIGH} m/s)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Steep Fraction vs |EY|\nr = {r:.3f}")

    # Add trend line
    if len(pair_df) > 2:
        z = np.polyfit(pair_df["steep_fraction"], abs_ey, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pair_df["steep_fraction"].min(), pair_df["steep_fraction"].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2)

    # 2. Weighted proximity vs |EY|
    ax = axes[0, 1]
    ax.scatter(pair_df["weighted_proximity"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["weighted_proximity"], abs_ey)
    ax.set_xlabel("Weighted Proximity to Steep")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Proximity Score vs |EY|\nr = {r:.3f}")

    # 3. Distance from steep vs |EY|
    ax = axes[0, 2]
    ax.scatter(pair_df["distance_from_steep"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["distance_from_steep"], abs_ey)
    ax.set_xlabel("Distance from 8 m/s")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Distance from Steep vs |EY|\nr = {r:.3f}")

    # 4. Energy-weighted WS vs |EY|
    ax = axes[1, 0]
    ax.scatter(pair_df["ws_mean_weighted"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["ws_mean_weighted"], abs_ey)
    ax.axvline(STEEP_LOW, color='red', linestyle='--', alpha=0.5, label='Steep region')
    ax.axvline(STEEP_HIGH, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(STEEP_LOW, STEEP_HIGH, alpha=0.1, color='red')
    ax.set_xlabel("Energy-Weighted Mean WS (m/s)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Mean WS vs |EY|\nr = {r:.3f}")
    ax.legend()

    # 5. Windspeed histogram
    ax = axes[1, 1]
    ax.hist(pair_df["ws_mean_weighted"], bins=15, alpha=0.7, edgecolor='black')
    ax.axvline(STEEP_LOW, color='red', linestyle='--', label=f'Steep region ({STEEP_LOW}-{STEEP_HIGH} m/s)')
    ax.axvline(STEEP_HIGH, color='red', linestyle='--')
    ax.axvspan(STEEP_LOW, STEEP_HIGH, alpha=0.1, color='red')
    ax.set_xlabel("Energy-Weighted Mean WS (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mean Windspeed")
    ax.legend()

    # 6. Steep fraction histogram
    ax = axes[1, 2]
    ax.hist(pair_df["steep_fraction"], bins=15, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f"Steep Fraction ({STEEP_LOW}-{STEEP_HIGH} m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Steep Exposure")

    plt.tight_layout()
    plt.savefig("windspeed_sensitivity_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: windspeed_sensitivity_exploration.png")


def print_recommendations(corr_results):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Find best metric
    best_ey = corr_results.loc[corr_results["r_EY"].abs().idxmax()]

    print(f"\n  Best metric for |EY|: {best_ey['Feature']}")
    print(f"    Correlation: r = {best_ey['r_EY']:+.3f} (p = {best_ey['p_EY']:.4f})")

    if best_ey['r_EY'] > 0.15 and best_ey['p_EY'] < 0.1:
        print(f"\n  ✓ POSITIVE CORRELATION FOUND")
        print(f"    Higher exposure to steep region → higher |EY| deviation")
        print(f"    Consider adding to model")
    elif best_ey['r_EY'] < -0.15 and best_ey['p_EY'] < 0.1:
        print(f"\n  ✗ NEGATIVE CORRELATION (opposite to hypothesis)")
        print(f"    Higher exposure to steep region → LOWER |EY| deviation")
        print(f"    This contradicts the amplification theory")
    else:
        print(f"\n  ? WEAK/NO CORRELATION")
        print(f"    Windspeed sensitivity does not appear to drive EY uncertainty")
        print(f"    in this dataset")

    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    print("""
    If correlation is POSITIVE:
        → Power curve amplification is real
        → Add steep_fraction or proximity to model

    If correlation is NEGATIVE or ZERO:
        → Other factors dominate (terrain, distance, height)
        → Power curve effects may be implicit in existing drivers
        → Do NOT add to model
    """)


def main():
    print("=" * 70)
    print("WINDSPEED SENSITIVITY EXPLORATION")
    print(f"Testing: Does operating in steep power curve region (7-9 m/s)")
    print(f"         increase energy yield uncertainty?")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    # Compute features
    pair_df = compute_pair_features(df)

    # Correlation analysis
    corr_results = analyze_correlations(pair_df)

    # Stratified analysis
    analyze_by_category(pair_df)

    # Relationship check
    check_redundancy(pair_df)

    # Visualizations (disabled by default to save IDE resources)
    create_visualizations(pair_df, save_plots=False)

    # Recommendations
    print_recommendations(corr_results)

    return df, pair_df, corr_results


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_df, corr_results = result