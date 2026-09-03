"""
Energy Concentration Exploration

Hypothesis: If energy production is concentrated in few sectors,
prediction errors have less chance to "average out" → higher uncertainty

Metrics to test:
1. max_weight: Maximum sector weight (does one sector dominate?)
2. top2_weight: Sum of 2 largest weights (do 2 sectors carry most energy?)
3. effective_sectors: 1 / Σ(weight²) — ranges from 1 (concentrated) to 12 (even)
4. concentration_ratio: 1 - effective_sectors/12 — higher = more concentrated

All metrics computed from weight_energy_predicted (deployment-ready)
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
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}


def load_and_prepare_data(path):
    """Load data and compute WS deviation for comparison."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("ENERGY CONCENTRATION EXPLORATION")
    print("=" * 70)

    required = [
        "pair_id", "sector_name",
        "EY_deviation_sector_frac",
        "weight_energy_predicted",
        "Mean_windspeed_predicted", "Mean_windspeed_self",
    ]

    df = df.dropna(subset=required).copy()

    # Compute WS deviation for comparison
    ws_actual = df["Mean_windspeed_self"].values
    ws_pred = df["Mean_windspeed_predicted"].values
    ws_denom = np.maximum(ws_actual, 3.0)
    df["WS_deviation_rel"] = np.abs(ws_pred - ws_actual) / ws_denom

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    return df


def compute_concentration_metrics(df):
    """Compute energy concentration metrics for each pair."""

    print("\n" + "=" * 70)
    print("COMPUTING CONCENTRATION METRICS")
    print("=" * 70)

    results = []

    for pair_id, group in df.groupby("pair_id"):
        # Get weights and normalize
        weights = group["weight_energy_predicted"].values
        weights = weights / weights.sum()  # Ensure normalized

        # Sort weights descending
        weights_sorted = np.sort(weights)[::-1]

        # =========================================
        # CONCENTRATION METRICS
        # =========================================

        # 1. Max weight (does one sector dominate?)
        max_weight = weights.max()

        # 2. Top-2 concentration (do 2 sectors carry most energy?)
        top2_weight = weights_sorted[0] + weights_sorted[1]

        # 3. Top-3 concentration
        top3_weight = weights_sorted[0] + weights_sorted[1] + weights_sorted[2]

        # 4. Herfindahl-Hirschman Index (sum of squared weights)
        hhi = np.sum(weights ** 2)

        # 5. Effective number of sectors (inverse HHI)
        # Ranges from 1 (all in one sector) to 12 (perfectly even)
        effective_sectors = 1.0 / hhi

        # 6. Concentration ratio: 0 = even, 1 = concentrated
        # (12 - effective_sectors) / 11 normalizes to [0, 1]
        concentration_ratio = (12 - effective_sectors) / 11

        # =========================================
        # AGGREGATE DEVIATIONS TO PAIR LEVEL
        # =========================================

        # Energy-weighted EY deviation
        ey_dev = np.sum(weights * group["EY_deviation_sector_frac"].values)

        # Energy-weighted WS deviation
        ws_dev = np.sum(weights * group["WS_deviation_rel"].values)

        # Location if available
        location = group["location"].iloc[0] if "location" in group.columns else pair_id

        results.append({
            "pair_id": pair_id,
            "location": location,
            "max_weight": max_weight,
            "top2_weight": top2_weight,
            "top3_weight": top3_weight,
            "hhi": hhi,
            "effective_sectors": effective_sectors,
            "concentration_ratio": concentration_ratio,
            "ey_deviation": ey_dev,
            "ws_deviation": ws_dev,
        })

    pair_df = pd.DataFrame(results)

    # =========================================
    # BASIC STATISTICS
    # =========================================

    print("\n" + "-" * 50)
    print("CONCENTRATION METRIC STATISTICS")
    print("-" * 50)

    metrics = ["max_weight", "top2_weight", "effective_sectors", "concentration_ratio"]

    for metric in metrics:
        vals = pair_df[metric]
        print(f"\n  {metric}:")
        print(f"    Range: [{vals.min():.3f}, {vals.max():.3f}]")
        print(f"    Mean: {vals.mean():.3f}")
        print(f"    Median: {vals.median():.3f}")

    # Interpretation guide
    print("\n" + "-" * 50)
    print("INTERPRETATION GUIDE")
    print("-" * 50)
    print("""
  max_weight:
    - Even distribution: ~0.083 (1/12)
    - Concentrated: >0.25 (one sector has >25% of energy)

  effective_sectors:
    - Even distribution: ~12
    - Concentrated: <6 (effectively only a few sectors matter)

  concentration_ratio:
    - Even distribution: ~0
    - Concentrated: >0.5
""")

    return pair_df


def analyze_correlations(pair_df):
    """Test correlations between concentration and deviation."""

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values  # Already absolute

    metrics = [
        ("max_weight", "Max weight (single sector dominance)"),
        ("top2_weight", "Top-2 weight (2 sectors)"),
        ("top3_weight", "Top-3 weight (3 sectors)"),
        ("concentration_ratio", "Concentration ratio (0=even, 1=concentrated)"),
        ("effective_sectors", "Effective sectors (12=even, 1=concentrated)"),
    ]

    print(f"\n  {'Metric':<45} {'vs |EY|':<15} {'vs |WS|':<15}")
    print("  " + "-" * 75)

    results = []

    for col, label in metrics:
        x = pair_df[col].values

        r_ey, p_ey = stats.pearsonr(x, abs_ey)
        r_ws, p_ws = stats.pearsonr(x, abs_ws)

        sig_ey = "**" if p_ey < 0.01 else "*" if p_ey < 0.05 else ""
        sig_ws = "**" if p_ws < 0.01 else "*" if p_ws < 0.05 else ""

        results.append({
            "Metric": label,
            "Column": col,
            "r_EY": r_ey,
            "p_EY": p_ey,
            "r_WS": r_ws,
            "p_WS": p_ws,
        })

        print(f"  {label:<45} r={r_ey:+.3f} {sig_ey:<3} r={r_ws:+.3f} {sig_ws:<3}")

    # Note about expected direction
    print("\n" + "-" * 50)
    print("EXPECTED DIRECTIONS")
    print("-" * 50)
    print("""
  If hypothesis is correct:
    - max_weight, top2_weight, concentration_ratio: POSITIVE correlation
      (more concentrated → more uncertainty)
    - effective_sectors: NEGATIVE correlation
      (fewer effective sectors → more uncertainty)
""")

    return pd.DataFrame(results)


def analyze_stratified(pair_df):
    """Stratified analysis: compare high vs low concentration."""

    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # Use concentration_ratio for stratification
    conc = pair_df["concentration_ratio"].values
    median_conc = np.median(conc)

    low_conc = conc <= median_conc
    high_conc = conc > median_conc

    print(f"\n  Split by concentration_ratio (median = {median_conc:.3f}):")

    print(f"\n  LOW concentration (more even):")
    print(f"    n = {low_conc.sum()}")
    print(f"    Mean |EY|: {abs_ey[low_conc].mean():.4f}")
    print(f"    Mean |WS|: {abs_ws[low_conc].mean():.4f}")

    print(f"\n  HIGH concentration (more concentrated):")
    print(f"    n = {high_conc.sum()}")
    print(f"    Mean |EY|: {abs_ey[high_conc].mean():.4f}")
    print(f"    Mean |WS|: {abs_ws[high_conc].mean():.4f}")

    # T-tests
    t_ey, p_ey = stats.ttest_ind(abs_ey[low_conc], abs_ey[high_conc])
    t_ws, p_ws = stats.ttest_ind(abs_ws[low_conc], abs_ws[high_conc])

    print(f"\n  T-test (high vs low concentration):")
    print(f"    |EY|: t = {t_ey:.2f}, p = {p_ey:.4f} {'*' if p_ey < 0.05 else ''}")
    print(f"    |WS|: t = {t_ws:.2f}, p = {p_ws:.4f} {'*' if p_ws < 0.05 else ''}")

    if abs_ey[high_conc].mean() > abs_ey[low_conc].mean() and p_ey < 0.1:
        print("\n  ✓ SUPPORTS HYPOTHESIS: High concentration → higher |EY|")
    elif abs_ey[high_conc].mean() < abs_ey[low_conc].mean():
        print("\n  ✗ OPPOSITE: High concentration → LOWER |EY|")
    else:
        print("\n  ? No significant difference")


def check_confounds(pair_df, df_sector):
    """Check if concentration is confounded with other features."""

    print("\n" + "=" * 70)
    print("CONFOUND CHECK")
    print("=" * 70)

    # Compute other pair-level features for comparison
    pair_features = df_sector.groupby("pair_id").agg(
        dz=("dz", "first") if "dz" in df_sector.columns else ("pair_id", "count"),
        distance_m=("distance_m", "first") if "distance_m" in df_sector.columns else ("pair_id", "count"),
    ).reset_index()

    # Merge with concentration metrics
    merged = pair_df.merge(pair_features, on="pair_id", how="left")

    conc = merged["concentration_ratio"].values

    features_to_check = []

    if "dz" in merged.columns and merged["dz"].notna().any():
        features_to_check.append(("dz", "|dz|", np.abs(merged["dz"].values)))

    if "distance_m" in merged.columns and merged["distance_m"].notna().any():
        features_to_check.append(("distance_m", "Distance", merged["distance_m"].values))

    if len(features_to_check) == 0:
        print("\n  No confound features available to check.")
        return

    print(f"\n  Correlation of concentration_ratio with other features:")
    print(f"\n  {'Feature':<20} {'Correlation'}")
    print("  " + "-" * 35)

    for col, label, values in features_to_check:
        valid = np.isfinite(values) & np.isfinite(conc)
        if valid.sum() < 10:
            continue

        r, p = stats.pearsonr(conc[valid], values[valid])
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

        flag = "⚠ HIGH" if abs(r) > 0.5 else "? moderate" if abs(r) > 0.3 else ""
        print(f"  {label:<20} r = {r:+.3f} {sig:<3} {flag}")


def create_visualizations(pair_df):
    """Create diagnostic visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # 1. Distribution of effective sectors
    ax = axes[0, 0]
    ax.hist(pair_df["effective_sectors"], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(pair_df["effective_sectors"].mean(), color='red', linestyle='--',
               label=f'Mean: {pair_df["effective_sectors"].mean():.1f}')
    ax.axvline(12, color='green', linestyle=':', label='Max (even): 12')
    ax.set_xlabel("Effective Sectors")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Energy Concentration\n(Higher = More Even)")
    ax.legend()

    # 2. Concentration ratio vs |EY|
    ax = axes[0, 1]
    ax.scatter(pair_df["concentration_ratio"], abs_ey, alpha=0.6)

    r, _ = stats.pearsonr(pair_df["concentration_ratio"], abs_ey)

    # Trend line
    slope, intercept = np.polyfit(pair_df["concentration_ratio"], abs_ey, 1)
    x_line = np.linspace(pair_df["concentration_ratio"].min(),
                         pair_df["concentration_ratio"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("Concentration Ratio (0=even, 1=concentrated)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Concentration vs |EY|\nr = {r:.3f}")

    # 3. Concentration ratio vs |WS|
    ax = axes[1, 0]
    ax.scatter(pair_df["concentration_ratio"], abs_ws, alpha=0.6)

    r, _ = stats.pearsonr(pair_df["concentration_ratio"], abs_ws)

    slope, intercept = np.polyfit(pair_df["concentration_ratio"], abs_ws, 1)
    x_line = np.linspace(pair_df["concentration_ratio"].min(),
                         pair_df["concentration_ratio"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("Concentration Ratio (0=even, 1=concentrated)")
    ax.set_ylabel("|WS Deviation|")
    ax.set_title(f"Concentration vs |WS|\nr = {r:.3f}")

    # 4. Max weight vs |EY|
    ax = axes[1, 1]
    ax.scatter(pair_df["max_weight"], abs_ey, alpha=0.6, label="EY", c='blue')
    ax.scatter(pair_df["max_weight"], abs_ws, alpha=0.6, label="WS", c='orange')

    ax.axvline(1 / 12, color='green', linestyle=':', label='Even (1/12)')
    ax.set_xlabel("Max Sector Weight")
    ax.set_ylabel("Deviation")
    ax.set_title("Max Weight vs Deviation")
    ax.legend()

    plt.tight_layout()
    plt.savefig("energy_concentration_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: energy_concentration_exploration.png")


def print_recommendations(corr_results):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Find best metric
    best_ey = corr_results.loc[corr_results["r_EY"].abs().idxmax()]
    best_ws = corr_results.loc[corr_results["r_WS"].abs().idxmax()]

    print(f"\nBest metric for |EY|: {best_ey['Metric']}")
    print(f"  r = {best_ey['r_EY']:+.3f} (p = {best_ey['p_EY']:.4f})")

    print(f"\nBest metric for |WS|: {best_ws['Metric']}")
    print(f"  r = {best_ws['r_WS']:+.3f} (p = {best_ws['p_WS']:.4f})")

    # Recommendation
    max_r = max(abs(best_ey['r_EY']), abs(best_ws['r_WS']))

    print("\n" + "-" * 50)
    print("RECOMMENDATION")
    print("-" * 50)

    if max_r > 0.20:
        print(f"""
  ✓ ADD energy concentration to sigma model

  Best metric: {best_ey['Column'] if abs(best_ey['r_EY']) > abs(best_ws['r_WS']) else best_ws['Column']}

  Add as:
    log(σ) = ... + γ_conc × concentration_metric
""")
    elif max_r > 0.10:
        print(f"""
  ? CONSIDER adding energy concentration

  Correlation is weak but present (r ≈ {max_r:.2f})
  Could be combined with other minor features
""")
    else:
        print(f"""
  ✗ DO NOT add energy concentration

  Correlation is too weak (r < 0.10)
  Hypothesis not supported by data
""")


def main():
    print("=" * 70)
    print("ENERGY CONCENTRATION EXPLORATION")
    print("Testing: Concentrated energy → less averaging → more uncertainty?")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    # Compute concentration metrics
    pair_df = compute_concentration_metrics(df)

    # Correlation analysis
    corr_results = analyze_correlations(pair_df)

    # Stratified analysis
    analyze_stratified(pair_df)

    # Confound check
    check_confounds(pair_df, df)

    # Visualizations
    create_visualizations(pair_df)

    # Recommendations
    print_recommendations(corr_results)

    return df, pair_df, corr_results


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_df, corr_results = result