"""
Roughness Changes Exploration

Feature: roughness_ch_avg
- Number of land cover transitions (e.g., forest→water) within 10km radius
- Averaged between WTG and MM locations
- Range: 0-10 (capped at 10)
- Sector-specific

Hypothesis: More roughness changes → internal boundary layers →
            harder to model → more uncertainty

Note: This is different from RIX (orography/hills).
      Roughness is about land cover (topography).
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
    """Load data and verify roughness column exists."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("ROUGHNESS CHANGES EXPLORATION")
    print("=" * 70)

    # Check for roughness column
    if "roughness_ch_avg" not in df.columns:
        roughness_cols = [c for c in df.columns if 'rough' in c.lower()]
        print(f"⚠ 'roughness_ch_avg' not found!")
        print(f"  Available roughness columns: {roughness_cols}")
        return None

    required = [
        "pair_id", "sector_name", "EY_deviation_sector_frac",
        "weight_energy", "weight_energy_predicted",
        "roughness_ch_avg"
    ]

    df = df.dropna(subset=required).copy()

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    return df


def analyze_basic_statistics(df):
    """Basic statistics of roughness_ch_avg."""

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    rc = df["roughness_ch_avg"]

    print(f"\n  Range: [{rc.min():.2f}, {rc.max():.2f}]")
    print(f"  Mean: {rc.mean():.2f}")
    print(f"  Median: {rc.median():.2f}")
    print(f"  Std: {rc.std():.2f}")

    print(f"\n  Percentiles:")
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(rc, p)
        print(f"    {p}th: {val:.2f}")

    # Distribution of values
    print(f"\n  Value distribution:")
    value_counts = rc.round(0).value_counts().sort_index()
    for val, count in value_counts.items():
        pct = count / len(rc) * 100
        bar = "█" * int(pct / 2)
        print(f"    {val:>4.0f}: {count:>4} ({pct:>5.1f}%) {bar}")


def analyze_sector_level(df):
    """Test roughness_ch_avg at sector level."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    y = df["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)
    roughness = df["roughness_ch_avg"].values

    print(f"\nValid sector observations: {len(df)}")

    # =========================================
    # CORRELATION WITH |EY DEVIATION|
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATION WITH |EY DEVIATION|")
    print("-" * 50)

    r, p = stats.pearsonr(roughness, abs_y)
    rho, p_rho = stats.spearmanr(roughness, abs_y)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print(f"\n  roughness_ch_avg vs |EY deviation|:")
    print(f"    Pearson r  = {r:+.4f} {sig}")
    print(f"    Spearman ρ = {rho:+.4f}")
    print(f"    p-value    = {p:.4f}")

    if r > 0 and p < 0.05:
        print(f"\n  ✓ SIGNIFICANT: More roughness changes → more error")
        print(f"  → Consider adding to sigma model")
    elif r > 0:
        print(f"\n  ? Positive but not significant at sector level")
        print(f"  → Check pair-level analysis")
    else:
        print(f"\n  ✗ No positive relationship found")

    # =========================================
    # STRATIFIED ANALYSIS
    # =========================================

    print("\n" + "-" * 50)
    print("STRATIFIED ANALYSIS BY ROUGHNESS LEVEL")
    print("-" * 50)

    # Split into terciles
    terciles = np.percentile(roughness, [33, 67])

    low_mask = roughness <= terciles[0]
    med_mask = (roughness > terciles[0]) & (roughness <= terciles[1])
    high_mask = roughness > terciles[1]

    print(f"\n  Low roughness (≤{terciles[0]:.1f}):")
    print(f"    n = {low_mask.sum()}, |EY| mean = {abs_y[low_mask].mean():.4f}, std = {abs_y[low_mask].std():.4f}")

    print(f"\n  Medium roughness ({terciles[0]:.1f} - {terciles[1]:.1f}):")
    print(f"    n = {med_mask.sum()}, |EY| mean = {abs_y[med_mask].mean():.4f}, std = {abs_y[med_mask].std():.4f}")

    print(f"\n  High roughness (>{terciles[1]:.1f}):")
    print(f"    n = {high_mask.sum()}, |EY| mean = {abs_y[high_mask].mean():.4f}, std = {abs_y[high_mask].std():.4f}")

    # Levene's test for variance differences
    stat, p_levene = stats.levene(abs_y[low_mask], abs_y[med_mask], abs_y[high_mask])
    print(f"\n  Levene's test for equal variances: p = {p_levene:.4f}")

    if p_levene < 0.05:
        print("  → SIGNIFICANT: Error spread differs by roughness level")
    else:
        print("  → No significant difference in spread")

    return r, p


def analyze_pair_level(df):
    """Aggregate to pair level and analyze."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted)")
    print("=" * 70)

    # Energy-weighted aggregation
    df = df.copy()
    df["w_ey_dev"] = df["weight_energy"] * df["EY_deviation_sector_frac"]
    df["w_roughness"] = df["weight_energy_predicted"] * df["roughness_ch_avg"]

    pair_agg = df.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
        roughness_ch_avg=("w_roughness", "sum"),
        location=("location", "first") if "location" in df.columns else ("pair_id", "first"),
    ).reset_index()

    print(f"\nPairs: {len(pair_agg)}")

    y = pair_agg["ey_deviation"].values
    abs_y = np.abs(y)
    roughness = pair_agg["roughness_ch_avg"].values

    print("\n" + "-" * 50)
    print("PAIR-LEVEL CORRELATION")
    print("-" * 50)

    r, p = stats.pearsonr(roughness, abs_y)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print(f"\n  roughness_ch_avg vs |EY deviation|:")
    print(f"    Pearson r = {r:+.4f} {sig}")
    print(f"    p-value   = {p:.4f}")

    return pair_agg, r, p


def check_redundancy(df):
    """Check if roughness_ch_avg is correlated with existing features."""

    print("\n" + "=" * 70)
    print("REDUNDANCY CHECK: Correlation with Existing Features")
    print("=" * 70)

    roughness = df["roughness_ch_avg"].values

    # Features to check
    features_to_check = [
        ("RIX_avg_0.3_sector", "RIX_avg (orography)"),
        ("distance_m", "Distance"),
        ("dz", "dz (height diff)"),
        ("overall_speedup_WTG_factor", "Speedup factor"),
        ("Sample_count_pred", "Sample count"),
    ]

    print(f"\n  {'Feature':<30} {'Correlation with roughness_ch_avg'}")
    print("  " + "-" * 55)

    for col, label in features_to_check:
        if col not in df.columns:
            print(f"  {label:<30} (column not found)")
            continue

        x = df[col].values
        valid = np.isfinite(x) & np.isfinite(roughness)

        if valid.sum() < 10:
            continue

        r, p = stats.pearsonr(roughness[valid], x[valid])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        flag = "⚠ HIGH" if abs(r) > 0.5 else "? moderate" if abs(r) > 0.3 else ""
        print(f"  {label:<30} r = {r:+.3f} {sig:<4} {flag}")

    print(f"\n  Interpretation:")
    print(f"    High correlation (|r| > 0.5) → Feature may be redundant")
    print(f"    Low correlation → Feature captures unique information")


def create_visualizations(df, pair_agg):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    abs_y_sector = np.abs(df["EY_deviation_sector_frac"].values)
    abs_y_pair = np.abs(pair_agg["ey_deviation"].values)

    # 1. Distribution of roughness_ch_avg
    ax = axes[0, 0]
    ax.hist(df["roughness_ch_avg"], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df["roughness_ch_avg"].mean(), color='red', linestyle='--',
               label=f'Mean: {df["roughness_ch_avg"].mean():.2f}')
    ax.set_xlabel("roughness_ch_avg")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Roughness Changes")
    ax.legend()

    # 2. Sector-level: roughness vs |EY|
    ax = axes[0, 1]
    ax.scatter(df["roughness_ch_avg"], abs_y_sector, alpha=0.3, s=10)

    r, _ = stats.pearsonr(df["roughness_ch_avg"], abs_y_sector)

    # Trend line
    slope, intercept = np.polyfit(df["roughness_ch_avg"], abs_y_sector, 1)
    x_line = np.linspace(df["roughness_ch_avg"].min(), df["roughness_ch_avg"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("roughness_ch_avg")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: Roughness vs |EY|\nr = {r:.3f}")

    # 3. Pair-level: roughness vs |EY|
    ax = axes[1, 0]
    ax.scatter(pair_agg["roughness_ch_avg"], abs_y_pair, alpha=0.6)

    r, _ = stats.pearsonr(pair_agg["roughness_ch_avg"], abs_y_pair)

    slope, intercept = np.polyfit(pair_agg["roughness_ch_avg"], abs_y_pair, 1)
    x_line = np.linspace(pair_agg["roughness_ch_avg"].min(), pair_agg["roughness_ch_avg"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("roughness_ch_avg (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: Roughness vs |EY|\nr = {r:.3f}")

    # 4. Roughness vs RIX (redundancy check)
    ax = axes[1, 1]
    if "RIX_avg_0.3_sector" in df.columns:
        ax.scatter(df["roughness_ch_avg"], df["RIX_avg_0.3_sector"], alpha=0.3, s=10)

        valid = df["RIX_avg_0.3_sector"].notna()
        r, _ = stats.pearsonr(df.loc[valid, "roughness_ch_avg"], df.loc[valid, "RIX_avg_0.3_sector"])

        ax.set_xlabel("roughness_ch_avg")
        ax.set_ylabel("RIX_avg_0.3")
        ax.set_title(f"Roughness vs RIX (redundancy check)\nr = {r:.3f}")
    else:
        ax.text(0.5, 0.5, "RIX_avg_0.3_sector\nnot found", ha='center', va='center')
        ax.set_title("Roughness vs RIX (redundancy check)")

    plt.tight_layout()
    plt.savefig("roughness_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: roughness_exploration.png")


def print_recommendations(sector_r, sector_p, pair_r, pair_p):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
ROUGHNESS_CH_AVG SUMMARY:

  Sector-level: r = {sector_r:+.4f} (p = {sector_p:.4f})
  Pair-level:   r = {pair_r:+.4f} (p = {pair_p:.4f})
""")

    # Recommendation logic
    if pair_r > 0.15 and pair_p < 0.1:
        print("""
RECOMMENDATION: ✓ ADD to sigma model

  The feature shows a meaningful positive correlation with |EY|.

  Add as:
    log(σ) = ... + γ_roughness × roughness_ch_avg

  Expected effect: ~{:.0f}% increase in σ per SD increase in roughness
""".format((pair_r / 0.3) * 20))  # rough estimate based on typical gamma values

    elif pair_r > 0 and pair_p < 0.2:
        print("""
RECOMMENDATION: ? CONSIDER adding to sigma model

  The feature shows a weak positive correlation.
  Could be combined with other minor features into a "complexity" composite.

  Options:
    A) Add alone (may not be significant)
    B) Combine with other minor features
    C) Skip for now, focus on stronger drivers
""")

    else:
        print("""
RECOMMENDATION: ✗ DO NOT add to sigma model

  The feature does not show a meaningful positive correlation with |EY|.
  It may not capture uncertainty beyond what existing features already explain.
""")


def main():
    print("=" * 70)
    print("ROUGHNESS CHANGES EXPLORATION")
    print("Testing: More land cover transitions → more uncertainty?")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    if df is None:
        return

    # Basic statistics
    analyze_basic_statistics(df)

    # Sector-level analysis
    sector_r, sector_p = analyze_sector_level(df)

    # Pair-level analysis
    pair_agg, pair_r, pair_p = analyze_pair_level(df)

    # Redundancy check
    check_redundancy(df)

    # Visualizations
    create_visualizations(df, pair_agg)

    # Recommendations
    print_recommendations(sector_r, sector_p, pair_r, pair_p)

    return df, pair_agg


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_agg = result