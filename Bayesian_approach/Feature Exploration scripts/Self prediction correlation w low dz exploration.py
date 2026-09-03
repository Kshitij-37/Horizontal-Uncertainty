"""
Exploratory Analysis: self_prediction_MM Impact on Overall Energy Yield Deviation

Testing the relationship between:
1. self_prediction_MM (how well MM predicts itself at the measurement site)
2. Overall energy yield deviation (how well extrapolation works)

Research Question:
When sites have similar elevations (|dz| ≤ 5m), does the sign and magnitude
of MM's self-prediction error predict the sign and magnitude of extrapolation error?

Hypothesis:
- If MM over-predicts at site A (self_pred_MM > 0), does it also over-predict
  when extrapolating from A to B?
- Or is there an inverse relationship (local bias doesn't transfer)?

Goal: Understand if self_prediction_MM should go in:
- Mean model (μ) - if there's a directional effect
- Sigma model (σ) - if magnitude affects uncertainty spread
- Both
- Neither


Results:
 - Same sign (match):     33.3% (3/9)
Opposite sign (anti):  66.7% (6/9)
Binomial test p-value: 0.5078

⚠ NO SIGNIFICANT RELATIONSHIP
  → Sign of self_pred_MM does not predict sign of EY deviation
  → May be due to small sample size (n=9)
  → Consider testing magnitude effects instead

Detailed breakdown:
  Self+ & EY+:   3 (33.3%)  ← Same sign (direct)
  Self- & EY-:   0 ( 0.0%)  ← Same sign (direct)
  Self+ & EY-:   6 (66.7%)  ← Opposite (inverse)
  Self- & EY+:   0 ( 0.0%)  ← Opposite (inverse)

For comparison - |dz| > 5.0m:
  Same sign rate: 46.4% (n=56)
  → Relationship STRONGER at larger |dz|!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# Threshold for "similar elevations"
DZ_THRESHOLD = 5.0  # meters

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}


def load_and_prepare_data(path):
    """Load data and prepare for analysis."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    # Show available columns with "self" or "prediction" in them
    print("\nColumns related to self-prediction:")
    self_cols = [c for c in df.columns if 'self' in c.lower() or 'prediction' in c.lower()]
    if self_cols:
        for col in self_cols:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
    else:
        print("  ⚠ No columns found with 'self' or 'prediction' in name!")

    needed = [
        "pair_id",
        "measurement_id",
        "prediction_id",
        "sector_name",
        "EY_deviation_sector_frac",
        "weight_energy",
        "dz",
        "self_prediction_MM",
    ]

    print(f"\nTotal rows in file: {len(df)}")

    # Check which needed columns are missing
    missing = [col for col in needed if col not in df.columns]
    if missing:
        print(f"\n⚠ MISSING COLUMNS: {missing}")
        print("\nAvailable columns in file:")
        for col in sorted(df.columns):
            print(f"  {col}")
        raise ValueError(f"Required columns missing: {missing}")

    df_valid = df.dropna(subset=needed).copy()
    print(f"Rows with all needed columns: {len(df_valid)}")

    # Normalize IDs
    df_valid["measurement_id"] = df_valid["measurement_id"].astype(str).str.strip().str.upper()
    df_valid["prediction_id"] = df_valid["prediction_id"].astype(str).str.strip().str.upper()

    # Sector index
    df_valid["sector_idx"] = df_valid["sector_name"].map(SECTOR_TO_IDX)
    df_valid = df_valid.dropna(subset=["sector_idx"]).copy()

    # Calculate overall EY deviation per pair (energy-weighted)
    df_valid["weighted_ey_dev"] = (df_valid["weight_energy"].astype(float) *
                                   df_valid["EY_deviation_sector_frac"].astype(float))

    pair_level = df_valid.groupby("pair_id").agg(
        overall_ey_deviation=("weighted_ey_dev", "sum"),
        dz=("dz", "first"),
        self_prediction_MM=("self_prediction_MM", "first"),
        n_sectors=("sector_idx", "count"),
    ).reset_index()

    # Note: For exploration, we don't require all 12 sectors
    # Just use whatever sectors are available
    print(f"\nTotal pairs: {len(pair_level)}")
    print(f"Sectors per pair - min: {pair_level['n_sectors'].min()}, max: {pair_level['n_sectors'].max()}")
    print(f"Pairs with all 12 sectors: {(pair_level['n_sectors'] == 12).sum()}")

    return df_valid, pair_level


def analyze_raw_distributions(pair_level):
    """Analyze raw distributions of key variables."""

    print("\n" + "=" * 70)
    print("RAW DATA DISTRIBUTIONS")
    print("=" * 70)

    # Overall stats
    print(f"\nTotal pairs: {len(pair_level)}")

    # self_prediction_MM distribution
    self_pred = pair_level["self_prediction_MM"].values

    # Check if we have any valid data
    if len(self_pred) == 0:
        print("\n⚠ ERROR: No valid self_prediction_MM values found!")
        print("Check your Excel file - the 'self_prediction_MM' column may be:")
        print("  - Missing")
        print("  - All NaN values")
        print("  - Named differently")
        return

    print(f"\nself_prediction_MM statistics:")
    print(f"  Min:    {self_pred.min():.4f} ({self_pred.min() * 100:.2f}%)")
    print(f"  Mean:   {self_pred.mean():.4f} ({self_pred.mean() * 100:.2f}%)")
    print(f"  Median: {np.median(self_pred):.4f} ({np.median(self_pred) * 100:.2f}%)")
    print(f"  Max:    {self_pred.max():.4f} ({self_pred.max() * 100:.2f}%)")
    print(f"  Std:    {self_pred.std():.4f} ({self_pred.std() * 100:.2f}%)")

    n_pos = (self_pred > 0).sum()
    n_neg = (self_pred < 0).sum()
    n_zero = (self_pred == 0).sum()

    print(f"\nself_prediction_MM sign distribution:")
    print(f"  Positive: {n_pos} ({n_pos / len(self_pred) * 100:.1f}%)")
    print(f"  Negative: {n_neg} ({n_neg / len(self_pred) * 100:.1f}%)")
    print(f"  Zero:     {n_zero} ({n_zero / len(self_pred) * 100:.1f}%)")

    # dz distribution
    dz = pair_level["dz"].values
    print(f"\ndz (elevation difference) statistics:")
    print(f"  Min:    {dz.min():.1f}m")
    print(f"  Mean:   {dz.mean():.1f}m")
    print(f"  Median: {np.median(dz):.1f}m")
    print(f"  Max:    {dz.max():.1f}m")
    print(f"  Std:    {dz.std():.1f}m")

    # Overall EY deviation distribution
    ey_dev = pair_level["overall_ey_deviation"].values
    print(f"\nOverall EY deviation statistics:")
    print(f"  Min:    {ey_dev.min():.4f} ({ey_dev.min() * 100:.2f}%)")
    print(f"  Mean:   {ey_dev.mean():.4f} ({ey_dev.mean() * 100:.2f}%)")
    print(f"  Median: {np.median(ey_dev):.4f} ({np.median(ey_dev) * 100:.2f}%)")
    print(f"  Max:    {ey_dev.max():.4f} ({ey_dev.max() * 100:.2f}%)")
    print(f"  Std:    {ey_dev.std():.4f} ({ey_dev.std() * 100:.2f}%)")


def analyze_by_dz_threshold(pair_level, threshold=5.0):
    """Analyze patterns stratified by elevation difference threshold."""

    print("\n" + "=" * 70)
    print(f"ANALYSIS BY |dz| THRESHOLD = {threshold}m")
    print("=" * 70)

    abs_dz = np.abs(pair_level["dz"].values)
    mask_small = abs_dz <= threshold
    mask_large = abs_dz > threshold

    n_total = len(pair_level)
    n_small = mask_small.sum()
    n_large = mask_large.sum()

    print(f"\nSample split:")
    if n_total > 0:
        print(f"  |dz| ≤ {threshold}m: {n_small} pairs ({n_small / n_total * 100:.1f}%)")
        print(f"  |dz| > {threshold}m: {n_large} pairs ({n_large / n_total * 100:.1f}%)")
    else:
        print(f"  ⚠ No pairs in dataset!")
        return

    if n_small == 0:
        print(f"\n⚠ NO pairs with |dz| ≤ {threshold}m")
        print(f"   Minimum |dz| in dataset: {abs_dz.min():.1f}m")
        print(f"   → Try a larger threshold!")
        return

    # Stats for each group
    for label, mask in [("Small |dz|", mask_small), ("Large |dz|", mask_large)]:
        subset = pair_level[mask]

        if len(subset) == 0:
            print(f"\n{label}: No data")
            continue

        print(f"\n{label} (|dz| {'≤' if label == 'Small |dz|' else '>'} {threshold}m):")
        print(
            f"  self_pred_MM range: [{subset['self_prediction_MM'].min():.4f}, {subset['self_prediction_MM'].max():.4f}]")
        print(f"  self_pred_MM mean:  {subset['self_prediction_MM'].mean():.4f}")

        n_pos = (subset['self_prediction_MM'] > 0).sum()
        n_neg = (subset['self_prediction_MM'] < 0).sum()
        n_total = len(subset)

        print(f"  self_pred_MM sign: {n_pos} positive, {n_neg} negative")
        print(
            f"  EY deviation range: [{subset['overall_ey_deviation'].min():.4f}, {subset['overall_ey_deviation'].max():.4f}]")
        print(f"  EY deviation mean:  {subset['overall_ey_deviation'].mean():.4f}")


def sign_matching_analysis(pair_level, threshold=5.0):
    """
    Core analysis: Does sign of self_pred_MM match sign of overall EY deviation?
    """

    print("\n" + "=" * 70)
    print(f"SIGN MATCHING ANALYSIS (|dz| ≤ {threshold}m)")
    print("=" * 70)
    print(f"\nQuestion: When sites have similar elevations (|dz| ≤ {threshold}m),")
    print(f"does sign(self_pred_MM) match sign(overall_ey_deviation)?")

    abs_dz = np.abs(pair_level["dz"].values)
    mask_small_dz = abs_dz <= threshold

    # Safety check for empty data
    if len(abs_dz) == 0:
        print("\n⚠ No pair-level data available!")
        return None

    self_pred = pair_level["self_prediction_MM"].values
    ey_dev = pair_level["overall_ey_deviation"].values

    n_small = mask_small_dz.sum()
    print(f"\nTotal pairs with |dz| ≤ {threshold}m: {n_small}")

    # Early exit if no pairs meet threshold
    if n_small == 0:
        print(f"\n⚠ NO PAIRS found with |dz| ≤ {threshold}m!")
        print(f"\nYour data has these |dz| values:")
        print(f"  Min |dz|: {abs_dz.min():.1f}m")
        print(f"  25th percentile: {np.percentile(abs_dz, 25):.1f}m")
        print(f"  Median |dz|: {np.median(abs_dz):.1f}m")
        print(f"  75th percentile: {np.percentile(abs_dz, 75):.1f}m")
        print(f"  Max |dz|: {abs_dz.max():.1f}m")
        print(f"\n💡 SUGGESTIONS:")
        print(f"  - Try threshold = {np.percentile(abs_dz, 25):.0f}m (25th percentile)")
        print(f"  - Or threshold = {np.median(abs_dz):.0f}m (median)")
        print(f"  - Change DZ_THRESHOLD at top of script")
        return None

    print(f"self_pred_MM range: [{self_pred[mask_small_dz].min():.4f}, {self_pred[mask_small_dz].max():.4f}]")
    print(f"EY deviation range: [{ey_dev[mask_small_dz].min():.4f}, {ey_dev[mask_small_dz].max():.4f}]")

    # Calculate signs
    sign_self_pred = np.sign(self_pred)
    sign_ey_dev = np.sign(ey_dev)

    print(f"\nSign distribution for |dz| ≤ {threshold}m:")
    print(f"  self_pred_MM: +1={((sign_self_pred == 1) & mask_small_dz).sum()}, "
          f"0={((sign_self_pred == 0) & mask_small_dz).sum()}, "
          f"-1={((sign_self_pred == -1) & mask_small_dz).sum()}")
    print(f"  EY deviation: +1={((sign_ey_dev == 1) & mask_small_dz).sum()}, "
          f"0={((sign_ey_dev == 0) & mask_small_dz).sum()}, "
          f"-1={((sign_ey_dev == -1) & mask_small_dz).sum()}")

    # Filter out zeros
    valid = mask_small_dz & (sign_self_pred != 0) & (sign_ey_dev != 0)
    print(f"\nAfter filtering zeros: {valid.sum()} valid pairs")

    if valid.sum() == 0:
        print("\n⚠ No valid pairs for sign matching analysis!")
        return None

    # Calculate matches
    matches = (sign_self_pred[valid] == sign_ey_dev[valid]).sum()
    anti_matches = (sign_self_pred[valid] == -sign_ey_dev[valid]).sum()
    n_valid = valid.sum()
    match_rate = matches / n_valid
    anti_match_rate = anti_matches / n_valid

    # Statistical test
    result = binomtest(int(matches), int(n_valid), 0.5, alternative='two-sided')

    print(f"\n{'-' * 70}")
    print("RESULTS:")
    print(f"{'-' * 70}")
    print(f"Same sign (match):     {match_rate:.1%} ({matches}/{n_valid})")
    print(f"Opposite sign (anti):  {anti_match_rate:.1%} ({anti_matches}/{n_valid})")
    print(f"Binomial test p-value: {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        if match_rate > 0.5:
            print(f"\n✓ DIRECT RELATIONSHIP: Signs significantly MATCH!")
            print(f"  → When self_pred_MM > 0 → EY deviation tends to be > 0")
            print(f"  → When self_pred_MM < 0 → EY deviation tends to be < 0")
            print(f"\n  INTERPRETATION:")
            print(f"    MM's local bias TRANSFERS to extrapolation")
            print(f"    → self_pred_MM is predictive of extrapolation direction")
            print(f"    → RECOMMEND: Add to MEAN model with POSITIVE beta")
        else:
            print(f"\n✓ INVERSE RELATIONSHIP: Signs significantly OPPOSITE!")
            print(f"  → When self_pred_MM > 0 → EY deviation tends to be < 0")
            print(f"  → When self_pred_MM < 0 → EY deviation tends to be > 0")
            print(f"\n  INTERPRETATION:")
            print(f"    MM's local bias does NOT transfer - it reverses!")
            print(f"    → self_pred_MM is anti-predictive")
            print(f"    → RECOMMEND: Add to MEAN model with NEGATIVE beta")
    else:
        print(f"\n⚠ NO SIGNIFICANT RELATIONSHIP")
        print(f"  → Sign of self_pred_MM does not predict sign of EY deviation")
        print(f"  → May be due to small sample size (n={n_valid})")
        print(f"  → Consider testing magnitude effects instead")

    # Detailed breakdown
    both_pos = ((sign_self_pred == 1) & (sign_ey_dev == 1) & valid).sum()
    both_neg = ((sign_self_pred == -1) & (sign_ey_dev == -1) & valid).sum()
    self_pos_ey_neg = ((sign_self_pred == 1) & (sign_ey_dev == -1) & valid).sum()
    self_neg_ey_pos = ((sign_self_pred == -1) & (sign_ey_dev == 1) & valid).sum()

    print(f"\nDetailed breakdown:")
    print(f"  Self+ & EY+:  {both_pos:2d} ({both_pos / n_valid:5.1%})  ← Same sign (direct)")
    print(f"  Self- & EY-:  {both_neg:2d} ({both_neg / n_valid:5.1%})  ← Same sign (direct)")
    print(f"  Self+ & EY-:  {self_pos_ey_neg:2d} ({self_pos_ey_neg / n_valid:5.1%})  ← Opposite (inverse)")
    print(f"  Self- & EY+:  {self_neg_ey_pos:2d} ({self_neg_ey_pos / n_valid:5.1%})  ← Opposite (inverse)")

    # Comparison with large |dz|
    mask_large_dz = abs_dz > threshold
    valid_large = mask_large_dz & (sign_self_pred != 0) & (sign_ey_dev != 0)

    if valid_large.sum() > 0:
        matches_large = (sign_self_pred[valid_large] == sign_ey_dev[valid_large]).sum()
        match_rate_large = matches_large / valid_large.sum()

        print(f"\nFor comparison - |dz| > {threshold}m:")
        print(f"  Same sign rate: {match_rate_large:.1%} (n={valid_large.sum()})")

        if match_rate_large > match_rate:
            print(f"  → Relationship STRONGER at larger |dz|!")
        elif match_rate_large < match_rate:
            print(f"  → Relationship WEAKER at larger |dz|")
        else:
            print(f"  → Similar relationship at both scales")

    return {
        "match_rate": match_rate,
        "anti_match_rate": anti_match_rate,
        "p_value": result.pvalue,
        "n_valid": n_valid,
        "both_pos": both_pos,
        "both_neg": both_neg,
        "self_pos_ey_neg": self_pos_ey_neg,
        "self_neg_ey_pos": self_neg_ey_pos,
    }


def correlation_analysis(pair_level, threshold=5.0):
    """Analyze correlation between self_pred_MM and EY deviation."""

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    abs_dz = np.abs(pair_level["dz"].values)

    for label, mask in [
        (f"|dz| ≤ {threshold}m", abs_dz <= threshold),
        (f"|dz| > {threshold}m", abs_dz > threshold),
        ("All pairs", np.ones(len(pair_level), dtype=bool))
    ]:
        subset = pair_level[mask]

        if len(subset) < 3:
            print(f"\n{label}: Too few samples (n={len(subset)})")
            continue

        x = subset["self_prediction_MM"].values
        y = subset["overall_ey_deviation"].values

        # Pearson correlation (linear)
        r_pearson, p_pearson = stats.pearsonr(x, y)

        # Spearman correlation (rank, more robust)
        r_spearman, p_spearman = stats.spearmanr(x, y)

        sig_pearson = "***" if p_pearson < 0.001 else "**" if p_pearson < 0.01 else "*" if p_pearson < 0.05 else ""
        sig_spearman = "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else ""

        print(f"\n{label} (n={len(subset)}):")
        print(f"  Pearson  r = {r_pearson:+.3f} {sig_pearson:5s} (p={p_pearson:.4f})")
        print(f"  Spearman ρ = {r_spearman:+.3f} {sig_spearman:5s} (p={p_spearman:.4f})")

        if p_pearson < 0.05:
            if r_pearson > 0:
                print(f"  → POSITIVE correlation: self_pred_MM ↑ → EY deviation ↑")
            else:
                print(f"  → NEGATIVE correlation: self_pred_MM ↑ → EY deviation ↓")


def magnitude_analysis(pair_level, threshold=5.0):
    """Does magnitude of self_pred_MM affect spread of EY deviation?"""

    print("\n" + "=" * 70)
    print("MAGNITUDE EFFECT ON SPREAD (for σ model)")
    print("=" * 70)

    abs_dz = np.abs(pair_level["dz"].values)
    mask_small = abs_dz <= threshold

    subset = pair_level[mask_small].copy()

    if len(subset) < 10:
        print(f"\n⚠ Too few samples for magnitude analysis (n={len(subset)})")
        return

    print(f"\nQuestion: Does |self_pred_MM| affect spread of EY deviation?")
    print(f"(for |dz| ≤ {threshold}m, n={len(subset)})")

    # Calculate magnitudes
    abs_self_pred = np.abs(subset["self_prediction_MM"].values)
    abs_ey_dev = np.abs(subset["overall_ey_deviation"].values)

    # Correlation between magnitudes
    r, p = stats.pearsonr(abs_self_pred, abs_ey_dev)

    print(f"\nCorrelation |self_pred_MM| vs |EY deviation|:")
    print(f"  r = {r:+.3f}, p = {p:.4f}")

    if p < 0.05 and r > 0:
        print(f"  → SIGNIFICANT: Larger |self_pred_MM| → Larger |EY deviation|")
        print(f"  → RECOMMEND: Add |self_pred_MM| to SIGMA model")
    else:
        print(f"  → Not significant for sigma model")

    # Split by magnitude tertiles
    if len(subset) >= 9:
        q33 = np.percentile(abs_self_pred, 33)
        q66 = np.percentile(abs_self_pred, 66)

        low_mag = abs_self_pred <= q33
        med_mag = (abs_self_pred > q33) & (abs_self_pred <= q66)
        high_mag = abs_self_pred > q66

        print(f"\nEY deviation spread by |self_pred_MM| tertile:")

        for tlabel, tmask in [("Low", low_mag), ("Medium", med_mag), ("High", high_mag)]:
            ey_subset = subset.loc[subset.index[tmask], "overall_ey_deviation"].values
            if len(ey_subset) > 0:
                print(f"\n  {tlabel} |self_pred_MM| (n={tmask.sum()}):")
                print(f"    Std dev of EY deviation: {ey_subset.std():.4f}")
                print(f"    IQR of EY deviation:     {np.percentile(ey_subset, 75) - np.percentile(ey_subset, 25):.4f}")


def create_visualizations(pair_level, threshold=5.0):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_dz = np.abs(pair_level["dz"].values)
    mask_small = abs_dz <= threshold
    mask_large = abs_dz > threshold

    # =========================================
    # ROW 1: OVERALL PATTERNS
    # =========================================

    # 1. Scatter: self_pred_MM vs EY deviation (all pairs)
    ax = axes[0, 0]
    x_all = pair_level["self_prediction_MM"].values
    y_all = pair_level["overall_ey_deviation"].values

    ax.scatter(x_all, y_all, alpha=0.5, s=30, label="All pairs")

    r, p = stats.pearsonr(x_all, y_all)
    slope, intercept, _, _, _ = stats.linregress(x_all, y_all)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("self_prediction_MM")
    ax.set_ylabel("Overall EY deviation")
    ax.set_title(f"All pairs: r={r:.3f}, p={p:.4f}")
    ax.grid(True, alpha=0.3)

    # 2. Scatter: Stratified by |dz|
    ax = axes[0, 1]

    x_small = pair_level.loc[mask_small, "self_prediction_MM"].values
    y_small = pair_level.loc[mask_small, "overall_ey_deviation"].values

    x_large = pair_level.loc[mask_large, "self_prediction_MM"].values
    y_large = pair_level.loc[mask_large, "overall_ey_deviation"].values

    ax.scatter(x_small, y_small, alpha=0.6, s=40, label=f"|dz| ≤ {threshold}m", color='blue')
    ax.scatter(x_large, y_large, alpha=0.4, s=20, label=f"|dz| > {threshold}m", color='red')

    if len(x_small) > 2:
        r_small, _ = stats.pearsonr(x_small, y_small)
        slope_s, intercept_s, _, _, _ = stats.linregress(x_small, y_small)
        x_line_s = np.linspace(x_small.min(), x_small.max(), 100)
        ax.plot(x_line_s, slope_s * x_line_s + intercept_s, 'b-', linewidth=2,
                label=f"Small |dz|: r={r_small:.3f}")

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("self_prediction_MM")
    ax.set_ylabel("Overall EY deviation")
    ax.set_title(f"Stratified by |dz| threshold = {threshold}m")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Histogram: self_pred_MM distribution
    ax = axes[0, 2]

    ax.hist(x_small, bins=15, alpha=0.6, label=f"|dz| ≤ {threshold}m", color='blue', density=True)
    ax.hist(x_large, bins=15, alpha=0.4, label=f"|dz| > {threshold}m", color='red', density=True)

    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("self_prediction_MM")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of self_pred_MM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================
    # ROW 2: SMALL |dz| FOCUS
    # =========================================

    # 4. Sign matching visualization
    ax = axes[1, 0]

    if len(x_small) > 0:
        sign_x = np.sign(x_small)
        sign_y = np.sign(y_small)

        # Create contingency table
        both_pos = ((sign_x == 1) & (sign_y == 1)).sum()
        both_neg = ((sign_x == -1) & (sign_y == -1)).sum()
        x_pos_y_neg = ((sign_x == 1) & (sign_y == -1)).sum()
        x_neg_y_pos = ((sign_x == -1) & (sign_y == 1)).sum()

        categories = ['Self+\nEY+', 'Self-\nEY-', 'Self+\nEY-', 'Self-\nEY+']
        counts = [both_pos, both_neg, x_pos_y_neg, x_neg_y_pos]
        colors = ['green', 'green', 'red', 'red']

        bars = ax.bar(categories, counts, color=colors, alpha=0.6, edgecolor='black')

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel("Count")
        ax.set_title(f"Sign Matching (|dz| ≤ {threshold}m)\nGreen=Match, Red=Opposite")
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # 5. Magnitude scatter
    ax = axes[1, 1]

    if len(x_small) > 0:
        abs_x_small = np.abs(x_small)
        abs_y_small = np.abs(y_small)

        ax.scatter(abs_x_small, abs_y_small, alpha=0.6, s=40, color='purple')

        r_mag, p_mag = stats.pearsonr(abs_x_small, abs_y_small)
        slope_m, intercept_m, _, _, _ = stats.linregress(abs_x_small, abs_y_small)
        x_line_m = np.linspace(0, abs_x_small.max(), 100)
        ax.plot(x_line_m, slope_m * x_line_m + intercept_m, 'r-', linewidth=2)

        ax.set_xlabel("|self_prediction_MM|")
        ax.set_ylabel("|Overall EY deviation|")
        ax.set_title(f"Magnitude correlation (|dz| ≤ {threshold}m)\nr={r_mag:.3f}, p={p_mag:.4f}")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # 6. Box plot: EY dev by self_pred_MM tertiles
    ax = axes[1, 2]

    if len(x_small) >= 9:
        q33 = np.percentile(np.abs(x_small), 33)
        q66 = np.percentile(np.abs(x_small), 66)

        categories = pd.cut(np.abs(x_small), bins=[-np.inf, q33, q66, np.inf],
                            labels=["Low", "Medium", "High"])

        plot_df = pd.DataFrame({
            "EY deviation": y_small,
            "|self_pred_MM|": categories
        })

        sns.boxplot(data=plot_df, x="|self_pred_MM|", y="EY deviation", ax=ax)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"EY dev spread by |self_pred_MM| (|dz| ≤ {threshold}m)")
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Too few samples', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f"self_pred_MM_exploration_dz{threshold}m.png", dpi=150)
    plt.close()
    print(f"\nSaved: self_pred_MM_exploration_dz{threshold}m.png")


def print_recommendations(pair_level, sign_results, threshold=5.0):
    """Print recommendations for Bayesian model."""

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR BAYESIAN MODEL")
    print("=" * 70)

    if sign_results is None:
        print("\n⚠ Insufficient data for recommendations")
        return

    abs_dz = np.abs(pair_level["dz"].values)
    mask_small = abs_dz <= threshold

    x_small = pair_level.loc[mask_small, "self_prediction_MM"].values
    y_small = pair_level.loc[mask_small, "overall_ey_deviation"].values

    # Overall correlation
    if len(x_small) >= 3:
        r, p = stats.pearsonr(x_small, y_small)

        print(f"\n1. MEAN MODEL (μ) - Directional effect:")
        print(f"   Correlation: r = {r:+.3f}, p = {p:.4f}")
        print(f"   Sign matching: {sign_results['match_rate']:.1%} match rate (p={sign_results['p_value']:.4f})")

        if sign_results['p_value'] < 0.05:
            if sign_results['match_rate'] > 0.5:
                print(f"\n   ✓ ADD self_pred_MM to mean model")
                print(f"   ✓ Expected beta: POSITIVE")
                print(f"   ✓ Interpretation: MM's local bias transfers to extrapolation")
            else:
                print(f"\n   ✓ ADD self_pred_MM to mean model")
                print(f"   ✓ Expected beta: NEGATIVE")
                print(f"   ✓ Interpretation: MM's local bias reverses in extrapolation")
        else:
            print(f"\n   ✗ Effect not significant (n={sign_results['n_valid']} may be too small)")
            print(f"   → Consider collecting more data at similar elevations")
            print(f"   → Or test at pair level without |dz| restriction")

    # Magnitude effect
    if len(x_small) >= 3:
        abs_x = np.abs(x_small)
        abs_y = np.abs(y_small)
        r_mag, p_mag = stats.pearsonr(abs_x, abs_y)

        print(f"\n2. SIGMA MODEL (σ) - Spread effect:")
        print(f"   Correlation |self_pred_MM| vs |EY dev|: r = {r_mag:+.3f}, p = {p_mag:.4f}")

        if p_mag < 0.05 and r_mag > 0:
            print(f"\n   ✓ ADD |self_pred_MM| to sigma model")
            print(f"   ✓ Expected gamma: POSITIVE")
            print(f"   ✓ Interpretation: Poor self-prediction → more uncertainty")
        else:
            print(f"\n   ✗ No significant effect on spread")

    print(f"\n3. SAMPLE SIZE CONSIDERATION:")
    print(f"   n = {sign_results['n_valid']} pairs with |dz| ≤ {threshold}m")
    if sign_results['n_valid'] < 30:
        print(f"   ⚠ Small sample - results may not be robust")
        print(f"   → Consider using larger |dz| threshold")
        print(f"   → Or include in model but with skeptical prior")

    print(f"\n4. FEATURE SPECIFICATION:")
    print(f"   Use: self_prediction_MM (raw percentage values)")
    print(f"   Don't forget to STANDARDIZE in the model!")
    print(f"   - For μ: use signed self_pred_MM")
    print(f"   - For σ: use |self_pred_MM| (magnitude)")


def main():
    print("=" * 70)
    print("SELF_PREDICTION_MM EXPLORATION")
    print(f"Testing with |dz| threshold = {DZ_THRESHOLD}m")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_sector, pair_level = load_and_prepare_data(sector_model_path)

    # Basic distributions
    analyze_raw_distributions(pair_level)

    # Stratify by dz threshold
    analyze_by_dz_threshold(pair_level, threshold=DZ_THRESHOLD)

    # Core sign matching analysis
    sign_results = sign_matching_analysis(pair_level, threshold=DZ_THRESHOLD)

    # Correlation analysis
    correlation_analysis(pair_level, threshold=DZ_THRESHOLD)

    # Magnitude effects
    magnitude_analysis(pair_level, threshold=DZ_THRESHOLD)

    # Visualizations
    create_visualizations(pair_level, threshold=DZ_THRESHOLD)

    # Recommendations
    print_recommendations(pair_level, sign_results, threshold=DZ_THRESHOLD)

    return df_sector, pair_level, sign_results


if __name__ == "__main__":
    df_sector, pair_level, sign_results = main()