"""
Exploratory Analysis: Speedup Factor Impact on Sector-wise Energy Yield Deviation

Testing the relationship between:
1. d_overall_speedup_factor (ratio: WTG/MM)
2. overall_speedup_WTG_factor - overall_speedup_MM_factor (difference)

And sector-wise EY deviation.

Goal: Understand if speedup factors should go in:
- Mean model (μ) - if there's a directional effect
- Sigma model (σ) - if magnitude affects uncertainty spread
- Both
- Neither
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]


def load_and_prepare_data(path):
    """Load data and create speedup-related features."""

    df = pd.read_excel(path)

    # Check which columns exist
    print("=" * 70)
    print("AVAILABLE SPEEDUP COLUMNS")
    print("=" * 70)

    speedup_cols = [c for c in df.columns if 'speedup' in c.lower()]
    for col in speedup_cols:
        print(f"  {col}")

    # Create derived features
    if "overall_speedup_WTG_factor" in df.columns and "overall_speedup_MM_factor" in df.columns:
        # Difference (additive)
        df["speedup_diff"] = df["overall_speedup_WTG_factor"] - df["overall_speedup_MM_factor"]

        # Log ratio (multiplicative) - more natural for ratios
        df["log_speedup_ratio"] = np.log(
            df["overall_speedup_WTG_factor"] /
            np.clip(df["overall_speedup_MM_factor"], 1e-6, None)
        )

        # If d_overall_speedup_factor exists, check if it matches our calculation
        if "d_overall_speedup_factor" in df.columns:
            df["speedup_ratio_check"] = df["overall_speedup_WTG_factor"] / df["overall_speedup_MM_factor"]
            match = np.allclose(df["d_overall_speedup_factor"], df["speedup_ratio_check"], rtol=0.01, equal_nan=True)
            print(f"\nd_overall_speedup_factor ≈ WTG/MM ratio: {match}")

    return df


def analyze_sector_level(df):
    """Analyze speedup effects at sector level."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    # Filter to rows with valid data
    needed = ["EY_deviation_sector_frac", "overall_speedup_WTG_factor",
              "overall_speedup_MM_factor", "weight_energy_predicted"]
    df_valid = df.dropna(subset=needed).copy()

    print(f"\nValid sector observations: {len(df_valid)}")

    # Create features
    df_valid["speedup_diff"] = (df_valid["overall_speedup_WTG_factor"] -
                                df_valid["overall_speedup_MM_factor"])

    df_valid["log_speedup_ratio"] = np.log(
        df_valid["overall_speedup_WTG_factor"] /
        np.clip(df_valid["overall_speedup_MM_factor"], 1e-6, None)
    )

    # Check correlation between true and predicted weights
    corr_weights = df_valid["weight_energy"].corr(df_valid["weight_energy_predicted"])
    print(f"Correlation between weight_energy and weight_energy_predicted: {corr_weights:.3f}")

    # Also test absolute values for sigma model
    df_valid["abs_speedup_diff"] = np.abs(df_valid["speedup_diff"])
    df_valid["abs_log_speedup_ratio"] = np.abs(df_valid["log_speedup_ratio"])

    # Target
    y = df_valid["EY_deviation_sector_frac"].values

    # =========================================
    # CORRELATION ANALYSIS
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATIONS WITH EY DEVIATION (sector-level)")
    print("-" * 50)

    features_to_test = [
        ("speedup_diff", "WTG - MM (difference)"),
        ("log_speedup_ratio", "log(WTG/MM) (log ratio)"),
        ("abs_speedup_diff", "|WTG - MM| (for sigma)"),
        ("abs_log_speedup_ratio", "|log(WTG/MM)| (for sigma)"),
    ]

    results = []
    for col, label in features_to_test:
        x = df_valid[col].values

        # Pearson correlation
        r, p = stats.pearsonr(x, y)

        # Spearman (rank) correlation - more robust to outliers
        rho, p_spearman = stats.spearmanr(x, y)

        results.append({
            "Feature": label,
            "Pearson r": r,
            "Pearson p": p,
            "Spearman ρ": rho,
            "Spearman p": p_spearman,
        })

        sig_pearson = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        sig_spearman = "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else ""

        print(f"\n{label}:")
        print(f"  Pearson r  = {r:+.3f} {sig_pearson}")
        print(f"  Spearman ρ = {rho:+.3f} {sig_spearman}")

    print("\n  (* p<0.05, ** p<0.01, *** p<0.001)")

    # =========================================
    # DIRECTIONAL EFFECT CHECK
    # =========================================

    print("\n" + "-" * 50)
    print("DIRECTIONAL EFFECT ANALYSIS")
    print("-" * 50)
    print("Question: Does speedup ratio predict the DIRECTION of error?")

    # If speedup_ratio > 1 (WTG has more speedup than MM), what happens?
    # Hypothesis: Higher speedup at WTG means we might overpredict there

    high_speedup_wtg = df_valid["log_speedup_ratio"] > 0  # WTG > MM
    low_speedup_wtg = df_valid["log_speedup_ratio"] < 0  # WTG < MM

    ey_high = y[high_speedup_wtg]
    ey_low = y[low_speedup_wtg]

    print(f"\nWhen WTG speedup > MM speedup (n={high_speedup_wtg.sum()}):")
    print(f"  Mean EY deviation: {ey_high.mean():+.4f}")
    print(f"  Median EY deviation: {np.median(ey_high):+.4f}")

    print(f"\nWhen WTG speedup < MM speedup (n={low_speedup_wtg.sum()}):")
    print(f"  Mean EY deviation: {ey_low.mean():+.4f}")
    print(f"  Median EY deviation: {np.median(ey_low):+.4f}")

    # T-test
    t_stat, p_val = stats.ttest_ind(ey_high, ey_low)
    print(f"\nT-test for difference: t={t_stat:.2f}, p={p_val:.4f}")

    if p_val < 0.05:
        if ey_high.mean() > ey_low.mean():
            print("→ SIGNIFICANT: Higher WTG speedup → Higher (more positive) EY deviation")
            print("→ Suggests: speedup ratio should go in MEAN model with POSITIVE beta")
        else:
            print("→ SIGNIFICANT: Higher WTG speedup → Lower (more negative) EY deviation")
            print("→ Suggests: speedup ratio should go in MEAN model with NEGATIVE beta")
    else:
        print("→ NOT SIGNIFICANT: No clear directional effect")
        print("→ Suggests: speedup ratio may only affect SIGMA (spread), not mean")

    # =========================================
    # SPREAD EFFECT CHECK
    # =========================================

    print("\n" + "-" * 50)
    print("SPREAD EFFECT ANALYSIS")
    print("-" * 50)
    print("Question: Does speedup MAGNITUDE affect error SPREAD?")

    # Bin by absolute speedup ratio magnitude
    abs_log_ratio = np.abs(df_valid["log_speedup_ratio"])

    # Tertiles
    q33 = np.percentile(abs_log_ratio, 33)
    q66 = np.percentile(abs_log_ratio, 66)

    low_mag = abs_log_ratio <= q33
    med_mag = (abs_log_ratio > q33) & (abs_log_ratio <= q66)
    high_mag = abs_log_ratio > q66

    print(f"\nEY deviation spread by |log(speedup ratio)| tertile:")

    for label, mask in [("Low", low_mag), ("Medium", med_mag), ("High", high_mag)]:
        ey_subset = y[mask]
        print(f"\n  {label} magnitude (n={mask.sum()}):")
        print(f"    Std dev of EY deviation: {ey_subset.std():.4f}")
        print(f"    IQR of EY deviation: {np.percentile(ey_subset, 75) - np.percentile(ey_subset, 25):.4f}")

    # Levene's test for equality of variances
    stat, p_levene = stats.levene(y[low_mag], y[med_mag], y[high_mag])
    print(f"\nLevene's test for equal variances: stat={stat:.2f}, p={p_levene:.4f}")

    if p_levene < 0.05:
        print("→ SIGNIFICANT: Speedup magnitude affects error SPREAD")
        print("→ Suggests: |speedup ratio| should go in SIGMA model")
    else:
        print("→ NOT SIGNIFICANT: No clear effect on spread")

    return df_valid


def analyze_pair_level(df):
    """Aggregate to pair level and analyze."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted aggregation)")
    print("=" * 70)

    # Filter valid rows
    needed = ["pair_id", "EY_deviation_sector_frac", "weight_energy",
              "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
              "weight_energy_predicted"]
    df_valid = df.dropna(subset=needed).copy()

    # Create sector-level features
    df_valid["speedup_diff"] = (df_valid["overall_speedup_WTG_factor"] -
                                df_valid["overall_speedup_MM_factor"])
    df_valid["log_speedup_ratio"] = np.log(
        df_valid["overall_speedup_WTG_factor"] /
        np.clip(df_valid["overall_speedup_MM_factor"], 1e-6, None)
    )

    # Aggregate to pair level
    # Energy-weighted EY deviation
    df_valid["weighted_ey_dev"] = df_valid["weight_energy_predicted"] * df_valid["EY_deviation_sector_frac"]

    # Energy-weighted speedup features
    df_valid["weighted_speedup_diff"] = df_valid["weight_energy_predicted"] * df_valid["speedup_diff"]
    df_valid["weighted_log_ratio"] = df_valid["weight_energy"] * df_valid["log_speedup_ratio"]
    df_valid["weighted_abs_log_ratio"] = df_valid["weight_energy_predicted"] * np.abs(df_valid["log_speedup_ratio"])

    pair_agg = df_valid.groupby("pair_id").agg(
        ey_deviation=("weighted_ey_dev", "sum"),
        speedup_diff_signed=("weighted_speedup_diff", "sum"),
        log_ratio_signed=("weighted_log_ratio", "sum"),
        log_ratio_magnitude=("weighted_abs_log_ratio", "sum"),
    ).reset_index()

    print(f"\nPairs: {len(pair_agg)}")

    # Correlations at pair level
    print("\n" + "-" * 50)
    print("CORRELATIONS WITH OVERALL EY DEVIATION (pair-level)")
    print("-" * 50)

    y = pair_agg["ey_deviation"].values

    for col, label in [
        ("speedup_diff_signed", "Signed speedup diff (for μ)"),
        ("log_ratio_signed", "Signed log ratio (for μ)"),
        ("log_ratio_magnitude", "|log ratio| magnitude (for σ)"),
    ]:
        x = pair_agg[col].values
        r, p = stats.pearsonr(x, y)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.3f} {sig}")

    return pair_agg


def create_visualizations(df_sector, df_pair):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # =========================================
    # ROW 1: SECTOR-LEVEL PLOTS
    # =========================================

    # 1. Scatter: log speedup ratio vs EY deviation (sector)
    ax = axes[0, 0]
    x = df_sector["log_speedup_ratio"]
    y = df_sector["EY_deviation_sector_frac"]
    ax.scatter(x, y, alpha=0.3, s=10)

    # Add regression line
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("log(WTG speedup / MM speedup)")
    ax.set_ylabel("EY deviation (sector)")
    ax.set_title(f"Sector-level: r={r:.3f}, p={p:.4f}")

    # 2. Scatter: speedup difference vs EY deviation (sector)
    ax = axes[0, 1]
    x = df_sector["speedup_diff"]
    y = df_sector["EY_deviation_sector_frac"]
    ax.scatter(x, y, alpha=0.3, s=10)

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("WTG speedup - MM speedup")
    ax.set_ylabel("EY deviation (sector)")
    ax.set_title(f"Sector-level: r={r:.3f}, p={p:.4f}")

    # 3. Box plot: EY deviation by speedup tertile
    ax = axes[0, 2]
    abs_log = np.abs(df_sector["log_speedup_ratio"])
    q33, q66 = np.percentile(abs_log, [33, 66])

    categories = pd.cut(abs_log, bins=[-np.inf, q33, q66, np.inf],
                        labels=["Low", "Medium", "High"])

    plot_df = pd.DataFrame({
        "EY deviation": df_sector["EY_deviation_sector_frac"],
        "|log speedup ratio|": categories
    })

    sns.boxplot(data=plot_df, x="|log speedup ratio|", y="EY deviation", ax=ax)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Spread by speedup magnitude")

    # =========================================
    # ROW 2: PAIR-LEVEL PLOTS
    # =========================================

    # 4. Scatter: signed log ratio vs EY deviation (pair)
    ax = axes[1, 0]
    x = df_pair["log_ratio_signed"]
    y = df_pair["ey_deviation"]
    ax.scatter(x, y, alpha=0.6)

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Energy-weighted log(WTG/MM)")
    ax.set_ylabel("Overall EY deviation")
    ax.set_title(f"Pair-level (for μ): r={r:.3f}, p={p:.4f}")

    # 5. Scatter: magnitude vs |EY deviation| (pair)
    ax = axes[1, 1]
    x = df_pair["log_ratio_magnitude"]
    y = np.abs(df_pair["ey_deviation"])
    ax.scatter(x, y, alpha=0.6)

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("Energy-weighted |log(WTG/MM)|")
    ax.set_ylabel("|Overall EY deviation|")
    ax.set_title(f"Pair-level (for σ): r={r:.3f}, p={p:.4f}")

    # 6. Distribution comparison
    ax = axes[1, 2]

    # Split pairs by speedup magnitude
    median_mag = df_pair["log_ratio_magnitude"].median()
    low_mag = df_pair["log_ratio_magnitude"] <= median_mag
    high_mag = df_pair["log_ratio_magnitude"] > median_mag

    ax.hist(df_pair.loc[low_mag, "ey_deviation"], bins=20, alpha=0.5,
            label=f"Low |speedup| (n={low_mag.sum()})", density=True)
    ax.hist(df_pair.loc[high_mag, "ey_deviation"], bins=20, alpha=0.5,
            label=f"High |speedup| (n={high_mag.sum()})", density=True)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Overall EY deviation")
    ax.set_ylabel("Density")
    ax.set_title("EY deviation distribution by speedup magnitude")
    ax.legend()

    plt.tight_layout()
    plt.savefig("speedup_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: speedup_exploration.png")


def print_recommendations(df_sector, df_pair):
    """Print recommendations based on analysis."""

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR BAYESIAN MODEL")
    print("=" * 70)

    # Get correlations
    y_sector = df_sector["EY_deviation_sector_frac"].values
    log_ratio = df_sector["log_speedup_ratio"].values
    r_signed, p_signed = stats.pearsonr(log_ratio, y_sector)

    abs_log_ratio = np.abs(log_ratio)
    abs_ey = np.abs(y_sector)
    r_magnitude, p_magnitude = stats.pearsonr(abs_log_ratio, abs_ey)

    print("\n1. MEAN MODEL (μ) - Directional effect:")
    print(f"   Correlation (signed): r = {r_signed:+.3f}, p = {p_signed:.4f}")

    if p_signed < 0.05:
        if r_signed > 0:
            print("   → ADD log_speedup_ratio to mean model")
            print("   → Expected beta: POSITIVE")
            print("   → Interpretation: Higher WTG speedup → overprediction")
        else:
            print("   → ADD log_speedup_ratio to mean model")
            print("   → Expected beta: NEGATIVE")
            print("   → Interpretation: Higher WTG speedup → underprediction")
    else:
        print("   → Effect not significant at sector level")
        print("   → Consider testing at pair level before adding to μ")

    print("\n2. SIGMA MODEL (σ) - Spread effect:")
    print(f"   Correlation (|speedup| vs |error|): r = {r_magnitude:+.3f}, p = {p_magnitude:.4f}")

    if p_magnitude < 0.05 and r_magnitude > 0:
        print("   → ADD |log_speedup_ratio| to sigma model")
        print("   → Expected gamma: POSITIVE")
        print("   → Interpretation: Larger speedup difference → more uncertainty")
    else:
        print("   → Effect on spread not clearly significant")
        print("   → May still be worth testing in full model")

    print("\n3. FEATURE RECOMMENDATION:")
    print("   Use: log(WTG_speedup / MM_speedup)")
    print("   Reason: Ratios are better handled on log scale")
    print("   - For μ: use signed log ratio")
    print("   - For σ: use |log ratio| (magnitude)")


def main():
    print("=" * 70)
    print("SPEEDUP FACTOR EXPLORATION")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    # Sector-level analysis
    df_sector = analyze_sector_level(df)

    # Pair-level analysis
    df_pair = analyze_pair_level(df)

    # Visualizations
    create_visualizations(df_sector, df_pair)

    # Recommendations
    print_recommendations(df_sector, df_pair)

    return df, df_sector, df_pair


if __name__ == "__main__":
    df, df_sector, df_pair = main()

