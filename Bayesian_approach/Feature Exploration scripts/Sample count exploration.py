"""
Sample Count Exploration

Hypothesis: More samples → better statistical estimate → less uncertainty

Metrics to test:
1. Sector-level:
   - sample_count_pred (raw)
   - log(sample_count_pred) (diminishing returns)
   - 1/√sample_count_pred (standard error proxy)

2. Pair-level:
   - Total samples (campaign length proxy)
   - Min samples across sectors (weakest link)
   - Mean/Median samples per sector
   - log versions

Reference: 144 samples = 1 day (10-min data)
   1 week = ~1,008 samples
   1 month = ~4,320 samples
   1 year = ~52,560 samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SAMPLES_PER_DAY = 144


def load_and_prepare_data(path):
    """Load data and compute sample count metrics."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("SAMPLE COUNT EXPLORATION")
    print("=" * 70)

    # Check for sample_count_pred column
    if "Sample_count_pred" not in df.columns:
        sample_cols = [c for c in df.columns if 'sample' in c.lower()]
        print(f"⚠ 'Sample_count_pred' not found. Available sample columns: {sample_cols}")
        return None

    required = ["pair_id", "sector_name", "EY_deviation_sector_frac",
                "weight_energy", "weight_energy_predicted", "Sample_count_pred"]

    df = df.dropna(subset=required).copy()

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    # =========================================
    # BASIC STATISTICS
    # =========================================

    print("\n" + "-" * 50)
    print("SAMPLE COUNT STATISTICS (sector-level)")
    print("-" * 50)

    sc = df["Sample_count_pred"]

    print(f"\n  Range: [{sc.min():.0f}, {sc.max():.0f}] samples")
    print(f"  Mean: {sc.mean():.0f} samples")
    print(f"  Median: {sc.median():.0f} samples")
    print(f"  Std: {sc.std():.0f} samples")

    # Convert to days for interpretation
    print(f"\n  In days (144 samples/day):")
    print(f"    Range: [{sc.min() / SAMPLES_PER_DAY:.1f}, {sc.max() / SAMPLES_PER_DAY:.1f}] days")
    print(f"    Mean: {sc.mean() / SAMPLES_PER_DAY:.1f} days")
    print(f"    Median: {sc.median() / SAMPLES_PER_DAY:.1f} days")

    # Percentiles
    print(f"\n  Percentiles:")
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(sc, p)
        print(f"    {p}th: {val:.0f} samples ({val / SAMPLES_PER_DAY:.1f} days)")

    # =========================================
    # COMPUTE SECTOR-LEVEL METRICS
    # =========================================

    print("\n" + "-" * 50)
    print("COMPUTING SECTOR-LEVEL METRICS")
    print("-" * 50)

    # Raw sample count
    df["sample_count"] = df["Sample_count_pred"]

    # Log sample count (add small epsilon to avoid log(0))
    df["log_sample_count"] = np.log(df["sample_count"] + 1)

    # Inverse sqrt (standard error proxy) - LOWER is better, so we'll flip for correlation
    # Actually, for sigma model, higher 1/sqrt means MORE uncertainty
    df["inv_sqrt_sample"] = 1.0 / np.sqrt(df["sample_count"] + 1)

    # Days worth of data
    df["sample_days"] = df["sample_count"] / SAMPLES_PER_DAY

    print("\n  Metrics computed:")
    print(f"    sample_count: raw count")
    print(f"    log_sample_count: log(count + 1)")
    print(f"    inv_sqrt_sample: 1/√(count + 1) — higher = more uncertainty")
    print(f"    sample_days: count / 144")

    return df


def compute_pair_level_metrics(df):
    """Aggregate sample counts to pair level."""

    print("\n" + "-" * 50)
    print("COMPUTING PAIR-LEVEL METRICS")
    print("-" * 50)

    # Aggregate sample counts by pair
    pair_samples = df.groupby("pair_id").agg(
        total_samples=("sample_count", "sum"),
        min_samples=("sample_count", "min"),
        max_samples=("sample_count", "max"),
        mean_samples=("sample_count", "mean"),
        median_samples=("sample_count", "median"),
        std_samples=("sample_count", "std"),
    ).reset_index()

    # Log versions
    pair_samples["log_total_samples"] = np.log(pair_samples["total_samples"] + 1)
    pair_samples["log_min_samples"] = np.log(pair_samples["min_samples"] + 1)

    # Inverse sqrt versions
    pair_samples["inv_sqrt_total"] = 1.0 / np.sqrt(pair_samples["total_samples"] + 1)
    pair_samples["inv_sqrt_min"] = 1.0 / np.sqrt(pair_samples["min_samples"] + 1)

    # Days
    pair_samples["total_days"] = pair_samples["total_samples"] / SAMPLES_PER_DAY
    pair_samples["min_days"] = pair_samples["min_samples"] / SAMPLES_PER_DAY

    print(f"\n  Pair-level statistics:")
    print(
        f"    Total samples per pair: [{pair_samples['total_samples'].min():.0f}, {pair_samples['total_samples'].max():.0f}]")
    print(f"    Total days per pair: [{pair_samples['total_days'].min():.0f}, {pair_samples['total_days'].max():.0f}]")
    print(
        f"    Min samples in any sector: [{pair_samples['min_samples'].min():.0f}, {pair_samples['min_samples'].max():.0f}]")

    return pair_samples


def analyze_sector_level(df):
    """Test sector-level sample count metrics."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    abs_y = np.abs(df["EY_deviation_sector_frac"].values)

    # Metrics to test (name, column, expected_sign, description)
    # expected_sign: "+" means we expect positive correlation with |EY|
    #                "-" means we expect negative correlation with |EY|
    metrics = [
        ("sample_count", "sample_count", "-", "More samples → less error"),
        ("log_sample_count", "log_sample_count", "-", "Diminishing returns"),
        ("inv_sqrt_sample", "inv_sqrt_sample", "+", "Standard error proxy"),
        ("sample_days", "sample_days", "-", "Days of data"),
    ]

    print("\n" + "-" * 50)
    print("CORRELATIONS WITH |EY DEVIATION|")
    print("-" * 50)

    results = []

    for name, col, expected, desc in metrics:
        x = df[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        r, p = stats.pearsonr(x[valid], abs_y[valid])
        rho, _ = stats.spearmanr(x[valid], abs_y[valid])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        match = "✓" if (expected == "+" and r > 0) or (expected == "-" and r < 0) else "✗"

        results.append({
            "Metric": name,
            "Pearson_r": r,
            "Spearman_rho": rho,
            "p_value": p,
            "Expected": expected,
            "Match": match,
        })

        print(f"\n{name} ({desc}):")
        print(f"  Pearson r  = {r:+.4f} {sig}")
        print(f"  Spearman ρ = {rho:+.4f}")
        print(f"  Expected: {expected}, Actual: {'+' if r > 0 else '-'} {match}")

    return pd.DataFrame(results)


def analyze_pair_level(df, pair_samples):
    """Test pair-level sample count metrics."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted)")
    print("=" * 70)

    # Aggregate EY deviation to pair level
    df["w_ey_dev"] = df["weight_energy"] * df["EY_deviation_sector_frac"]

    pair_ey = df.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
    ).reset_index()

    # Merge with sample metrics
    pair_data = pair_ey.merge(pair_samples, on="pair_id")

    abs_y = np.abs(pair_data["ey_deviation"].values)

    print(f"\nPairs: {len(pair_data)}")

    # Metrics to test
    metrics = [
        ("total_samples", "-", "Campaign length"),
        ("log_total_samples", "-", "Campaign length (log)"),
        ("min_samples", "-", "Weakest sector"),
        ("log_min_samples", "-", "Weakest sector (log)"),
        ("mean_samples", "-", "Average per sector"),
        ("inv_sqrt_total", "+", "SE proxy (total)"),
        ("inv_sqrt_min", "+", "SE proxy (min)"),
    ]

    print("\n" + "-" * 50)
    print("CORRELATIONS WITH |EY DEVIATION|")
    print("-" * 50)

    pair_results = []

    for col, expected, desc in metrics:
        x = pair_data[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        r, p = stats.pearsonr(x[valid], abs_y[valid])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        match = "✓" if (expected == "+" and r > 0) or (expected == "-" and r < 0) else "✗"

        pair_results.append({
            "Metric": col,
            "Description": desc,
            "Pearson_r": r,
            "p_value": p,
            "Expected": expected,
            "Match": match,
        })

        print(f"\n{col} ({desc}):")
        print(f"  Pearson r = {r:+.4f} {sig}")
        print(f"  Expected: {expected}, Actual: {'+' if r > 0 else '-'} {match}")

    return pair_data, pd.DataFrame(pair_results)


def check_confounds(df):
    """Check for confounds between sample count and energy weight."""

    print("\n" + "=" * 70)
    print("CONFOUND CHECK: Sample Count vs Energy Weight")
    print("=" * 70)

    sc = df["sample_count"].values
    ew = df["weight_energy_predicted"].values

    r, p = stats.pearsonr(sc, ew)

    print(f"\n  sample_count vs weight_energy_predicted:")
    print(f"    Pearson r = {r:+.4f} (p = {p:.4f})")

    if abs(r) > 0.5:
        print(f"\n  ⚠ HIGH CORRELATION: Sample count is confounded with energy weight")
        print(f"    This means high-energy sectors naturally have more samples")
        print(f"    The effect may already be captured by energy weighting")
    elif abs(r) > 0.3:
        print(f"\n  ? MODERATE CORRELATION: Some confounding present")
    else:
        print(f"\n  ✓ LOW CORRELATION: Sample count is independent of energy weight")
        print(f"    This is good — it captures different information")


def analyze_threshold_effect(df):
    """Test if there's a threshold below which uncertainty increases."""

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS: Is there a minimum reliable sample count?")
    print("=" * 70)

    abs_y = np.abs(df["EY_deviation_sector_frac"].values)
    sc = df["sample_count"].values

    # Test different thresholds (in days)
    thresholds_days = [7, 14, 30, 60, 90]

    print(f"\n  Testing if |EY| differs above/below sample thresholds:")
    print(f"\n  {'Threshold':<20} {'Below: n, mean |EY|':<25} {'Above: n, mean |EY|':<25} {'T-test p'}")
    print("  " + "-" * 80)

    for days in thresholds_days:
        threshold = days * SAMPLES_PER_DAY

        below = sc < threshold
        above = sc >= threshold

        n_below = below.sum()
        n_above = above.sum()

        if n_below < 10 or n_above < 10:
            continue

        mean_below = abs_y[below].mean()
        mean_above = abs_y[above].mean()

        t_stat, p_val = stats.ttest_ind(abs_y[below], abs_y[above])

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {days} days ({threshold:.0f} samples){'':<3} "
              f"n={n_below:<4} μ={mean_below:.4f}{'':>5} "
              f"n={n_above:<4} μ={mean_above:.4f}{'':>5} "
              f"p={p_val:.4f} {sig}")

    # Find optimal threshold
    print("\n" + "-" * 50)
    print("FINDING OPTIMAL THRESHOLD")
    print("-" * 50)

    best_diff = 0
    best_threshold = None

    for threshold in range(100, 5000, 100):
        below = sc < threshold
        above = sc >= threshold

        if below.sum() < 20 or above.sum() < 20:
            continue

        diff = abs_y[below].mean() - abs_y[above].mean()

        if diff > best_diff:
            best_diff = diff
            best_threshold = threshold

    if best_threshold:
        print(f"\n  Optimal threshold: {best_threshold} samples ({best_threshold / SAMPLES_PER_DAY:.1f} days)")
        print(f"  Mean |EY| difference: {best_diff:.4f}")

        # Create binary feature suggestion
        below = sc < best_threshold
        print(f"\n  Suggested binary feature: sample_count < {best_threshold}")
        print(f"    Below: n={below.sum()}, mean |EY|={abs_y[below].mean():.4f}")
        print(f"    Above: n=(~below).sum(), mean |EY|={abs_y[~below].mean():.4f}")


def create_visualizations(df, pair_data):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_y_sector = np.abs(df["EY_deviation_sector_frac"].values)
    abs_y_pair = np.abs(pair_data["ey_deviation"].values)

    # 1. Distribution of sample counts
    ax = axes[0, 0]
    ax.hist(df["sample_count"], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df["sample_count"].median(), color='red', linestyle='--',
               label=f'Median: {df["sample_count"].median():.0f}')
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Sample Counts (Sector)")
    ax.legend()

    # 2. Sector: sample_count vs |EY|
    ax = axes[0, 1]
    ax.scatter(df["sample_count"], abs_y_sector, alpha=0.3, s=10)
    r, _ = stats.pearsonr(df["sample_count"], abs_y_sector)
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Sector: Sample Count vs |EY|\nr = {r:.3f}")

    # 3. Sector: log_sample_count vs |EY|
    ax = axes[0, 2]
    ax.scatter(df["log_sample_count"], abs_y_sector, alpha=0.3, s=10)
    r, _ = stats.pearsonr(df["log_sample_count"], abs_y_sector)
    ax.set_xlabel("log(Sample Count)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Sector: log(Sample Count) vs |EY|\nr = {r:.3f}")

    # 4. Pair: total_samples vs |EY|
    ax = axes[1, 0]
    ax.scatter(pair_data["total_samples"], abs_y_pair, alpha=0.6)
    r, _ = stats.pearsonr(pair_data["total_samples"], abs_y_pair)
    # Add trend line
    slope, intercept = np.polyfit(pair_data["total_samples"], abs_y_pair, 1)
    x_line = np.linspace(pair_data["total_samples"].min(), pair_data["total_samples"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Total Samples (Pair)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Pair: Total Samples vs |EY|\nr = {r:.3f}")

    # 5. Pair: min_samples vs |EY|
    ax = axes[1, 1]
    ax.scatter(pair_data["min_samples"], abs_y_pair, alpha=0.6)
    r, _ = stats.pearsonr(pair_data["min_samples"], abs_y_pair)
    slope, intercept = np.polyfit(pair_data["min_samples"], abs_y_pair, 1)
    x_line = np.linspace(pair_data["min_samples"].min(), pair_data["min_samples"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Min Samples in Any Sector (Pair)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Pair: Min Samples vs |EY|\nr = {r:.3f}")

    # 6. Pair: inv_sqrt_min vs |EY|
    ax = axes[1, 2]
    ax.scatter(pair_data["inv_sqrt_min"], abs_y_pair, alpha=0.6)
    r, _ = stats.pearsonr(pair_data["inv_sqrt_min"], abs_y_pair)
    slope, intercept = np.polyfit(pair_data["inv_sqrt_min"], abs_y_pair, 1)
    x_line = np.linspace(pair_data["inv_sqrt_min"].min(), pair_data["inv_sqrt_min"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("1/√(Min Samples) — SE Proxy")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Pair: SE Proxy (min) vs |EY|\nr = {r:.3f}")

    plt.tight_layout()
    plt.savefig("sample_count_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: sample_count_exploration.png")


def print_recommendations(sector_results, pair_results):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Find best sector-level metric
    sector_best = sector_results.loc[sector_results["Pearson_r"].abs().idxmax()]

    # Find best pair-level metric
    pair_best = pair_results.loc[pair_results["Pearson_r"].abs().idxmax()]

    print(f"\nBEST SECTOR-LEVEL METRIC:")
    print(f"  {sector_best['Metric']}: r = {sector_best['Pearson_r']:+.4f}")

    print(f"\nBEST PAIR-LEVEL METRIC:")
    print(f"  {pair_best['Metric']} ({pair_best['Description']}): r = {pair_best['Pearson_r']:+.4f}")

    # Recommendation
    print("\n" + "-" * 50)
    print("RECOMMENDATION FOR SIGMA MODEL")
    print("-" * 50)

    if abs(pair_best["Pearson_r"]) > 0.15:
        print(f"""
  ✓ ADD sample count to sigma model

  Best option: {pair_best['Metric']}

  If using inv_sqrt metric (positive gamma expected):
    log(σ) = ... + γ_sample × (1/√sample_count)

  If using raw/log metric (negative gamma expected):
    log(σ) = ... - γ_sample × log(sample_count)
    Or use inverse: γ × (1/sample_count)
""")
    else:
        print(f"""
  ? WEAK EFFECT: Sample count shows r = {pair_best['Pearson_r']:+.4f}

  Consider adding if:
    - Domain knowledge supports it
    - Want completeness in uncertainty model

  Skip if:
    - Want to keep model parsimonious
    - Effect is confounded with energy weighting
""")


def main():
    print("=" * 70)
    print("SAMPLE COUNT EXPLORATION")
    print("Testing: More samples → less uncertainty?")
    print("=" * 70)

    # Load and prepare data
    df = load_and_prepare_data(sector_model_path)

    if df is None:
        return

    # Compute pair-level metrics
    pair_samples = compute_pair_level_metrics(df)

    # Sector-level analysis
    sector_results = analyze_sector_level(df)

    # Pair-level analysis
    pair_data, pair_results = analyze_pair_level(df, pair_samples)

    # Check confounds
    check_confounds(df)

    # Threshold analysis
    analyze_threshold_effect(df)

    # Visualizations
    create_visualizations(df, pair_data)

    # Recommendations
    print_recommendations(sector_results, pair_results)

    return df, pair_data, sector_results, pair_results


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_data, sector_results, pair_results = result