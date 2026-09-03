"""
dRIX Exploration: Testing Option 1 and Option 5

Option 1: Severity Fraction
- What fraction of the complexity DIFFERENCE is in flow-separating terrain (>17°)?
- severity_fraction = |dRIX_0.3| / (|dRIX_0.3| + |dRIX_mild| + ε)
- where dRIX_mild = dRIX_0.0501 - dRIX_0.3

Option 5: Keep Features Separate
- For μ: dRIX_0.3 (signed) - direction of complexity difference
- For σ: |dRIX_0.3| - magnitude of complexity difference
- For σ: RIX_avg_0.3 - absolute severity of environment

Theory being tested:
- Flow separation starts at ~17° slope
- dRIX_0.3 captures the "flow separation" terrain difference
- When complexity differences are concentrated in severe terrain (high severity_fraction),
  the directional effect should be stronger
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
    """Load data and check available columns."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("AVAILABLE RIX/dRIX COLUMNS")
    print("=" * 70)

    rix_cols = [c for c in df.columns if 'rix' in c.lower() or 'RIX' in c]
    for col in sorted(rix_cols):
        print(f"  {col}")

    return df


def create_derived_features(df):
    """Create Option 1 and Option 5 features."""

    print("\n" + "=" * 70)
    print("CREATING DERIVED FEATURES")
    print("=" * 70)

    d = df.copy()

    # =========================================
    # OPTION 1: SEVERITY FRACTION
    # =========================================

    # dRIX_mild = difference in "gentle slopes only" (between 2° and 17°)
    d["dRIX_mild"] = d["dRIX_0.0501_sector"] - d["dRIX_0.3_sector"]

    # Severity fraction: what fraction of total |difference| is in severe terrain?
    abs_severe = np.abs(d["dRIX_0.3_sector"])
    abs_mild = np.abs(d["dRIX_mild"])

    d["severity_fraction"] = abs_severe / (abs_severe + abs_mild + 1e-6)

    # For cases where both are 0 (no complexity difference at all), set to 0
    both_zero = (abs_severe < 1e-6) & (abs_mild < 1e-6)
    d.loc[both_zero, "severity_fraction"] = 0.0

    print("\nOption 1: Severity Fraction")
    print(f"  Formula: |dRIX_0.3| / (|dRIX_0.3| + |dRIX_mild|)")
    print(f"  Range: [{d['severity_fraction'].min():.3f}, {d['severity_fraction'].max():.3f}]")
    print(f"  Mean: {d['severity_fraction'].mean():.3f}")
    print(f"  Cases where both dRIX are ~0: {both_zero.sum()}")

    # Interaction: signed dRIX_0.3 weighted by severity fraction
    # This amplifies dRIX_0.3 when the difference is concentrated in severe terrain
    d["dRIX_severity_weighted"] = d["dRIX_0.3_sector"] * d["severity_fraction"]

    # =========================================
    # OPTION 5: SEPARATE FEATURES
    # =========================================

    print("\nOption 5: Separate Features")

    # For μ (mean model): signed dRIX_0.3
    d["dRIX_03_signed"] = d["dRIX_0.3_sector"]
    print(f"  dRIX_0.3 (signed) for μ: range [{d['dRIX_03_signed'].min():.2f}, {d['dRIX_03_signed'].max():.2f}]")

    # For σ (sigma model): magnitude of dRIX_0.3
    d["dRIX_03_mag"] = np.abs(d["dRIX_0.3_sector"])
    print(f"  |dRIX_0.3| for σ: range [{d['dRIX_03_mag'].min():.2f}, {d['dRIX_03_mag'].max():.2f}]")

    # For σ (sigma model): absolute severity of environment (RIX_avg_0.3)
    if "RIX_avg_0.3_sector" in d.columns:
        d["RIX_avg_03"] = d["RIX_avg_0.3_sector"]
        print(f"  RIX_avg_0.3 for σ: range [{d['RIX_avg_03'].min():.2f}, {d['RIX_avg_03'].max():.2f}]")
    else:
        print("  WARNING: RIX_avg_0.3_sector not found!")
        d["RIX_avg_03"] = np.nan

    return d


def analyze_sector_level(df):
    """Analyze at sector level."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    required = ["EY_deviation_sector_frac", "dRIX_0.3_sector", "dRIX_0.0501_sector",
                "severity_fraction", "weight_energy_predicted"]

    df_valid = df.dropna(subset=required).copy()
    print(f"\nValid sector observations: {len(df_valid)}")

    y = df_valid["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)

    # =========================================
    # CORRELATIONS FOR MEAN MODEL (μ)
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATIONS FOR MEAN MODEL (μ) - Directional effects")
    print("-" * 50)
    print("(Looking for correlation with SIGNED EY deviation)")

    mu_features = [
        ("dRIX_03_signed", "dRIX_0.3 (signed)"),
        ("dRIX_severity_weighted", "dRIX_0.3 × severity_fraction"),
    ]

    mu_results = []
    for col, label in mu_features:
        if col not in df_valid.columns:
            continue
        x = df_valid[col].values
        valid = np.isfinite(x) & np.isfinite(y)

        r, p = stats.pearsonr(x[valid], y[valid])
        mu_results.append((col, label, r, p))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    # =========================================
    # CORRELATIONS FOR SIGMA MODEL (σ)
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATIONS FOR SIGMA MODEL (σ) - Spread effects")
    print("-" * 50)
    print("(Looking for correlation with |EY deviation|)")

    sigma_features = [
        ("dRIX_03_mag", "|dRIX_0.3| (magnitude)"),
        ("RIX_avg_03", "RIX_avg_0.3 (absolute severity)"),
        ("severity_fraction", "Severity fraction"),
    ]

    sigma_results = []
    for col, label in sigma_features:
        if col not in df_valid.columns:
            continue
        x = df_valid[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        if valid.sum() < 10:
            continue

        r, p = stats.pearsonr(x[valid], abs_y[valid])
        sigma_results.append((col, label, r, p))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    # =========================================
    # SEVERITY FRACTION DEEP DIVE
    # =========================================

    print("\n" + "-" * 50)
    print("SEVERITY FRACTION DEEP DIVE")
    print("-" * 50)

    sev_frac = df_valid["severity_fraction"].values

    # Distribution
    print(f"\nDistribution of severity_fraction:")
    print(f"  0.0 (all mild): {(sev_frac == 0).sum()} cases ({(sev_frac == 0).mean() * 100:.1f}%)")
    print(f"  0.0 - 0.5: {((sev_frac > 0) & (sev_frac <= 0.5)).sum()} cases")
    print(f"  0.5 - 1.0: {((sev_frac > 0.5) & (sev_frac < 1.0)).sum()} cases")
    print(f"  1.0 (all severe): {(sev_frac == 1.0).sum()} cases ({(sev_frac == 1.0).mean() * 100:.1f}%)")

    # Does high severity fraction amplify the dRIX effect?
    print("\n  Checking if severity_fraction modulates dRIX effect:")

    # Split into low vs high severity fraction
    low_sev = sev_frac <= 0.5
    high_sev = sev_frac > 0.5

    dRIX = df_valid["dRIX_03_signed"].values

    if low_sev.sum() > 10 and high_sev.sum() > 10:
        r_low, p_low = stats.pearsonr(dRIX[low_sev], y[low_sev])
        r_high, p_high = stats.pearsonr(dRIX[high_sev], y[high_sev])

        print(f"\n  When severity_fraction ≤ 0.5 (mild differences dominate):")
        print(f"    dRIX vs EY: r = {r_low:+.3f} (n={low_sev.sum()})")

        print(f"\n  When severity_fraction > 0.5 (severe differences dominate):")
        print(f"    dRIX vs EY: r = {r_high:+.3f} (n={high_sev.sum()})")

        if abs(r_high) > abs(r_low):
            print(f"\n  → Supports theory: dRIX effect is STRONGER when differences are concentrated in severe terrain")
        else:
            print(f"\n  → Does NOT support theory: dRIX effect is similar or weaker in severe terrain")

    return df_valid, mu_results, sigma_results


def analyze_pair_level(df):
    """Aggregate to pair level and analyze."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted aggregation)")
    print("=" * 70)

    required = ["pair_id", "EY_deviation_sector_frac", "weight_energy",
                "weight_energy_predicted", "dRIX_0.3_sector", "severity_fraction"]

    df_valid = df.dropna(subset=required).copy()

    # Energy-weighted aggregation
    df_valid["w_ey_dev"] = df_valid["weight_energy"] * df_valid["EY_deviation_sector_frac"]
    df_valid["w_dRIX_03"] = df_valid["weight_energy_predicted"] * df_valid["dRIX_03_signed"]
    df_valid["w_dRIX_03_mag"] = df_valid["weight_energy_predicted"] * df_valid["dRIX_03_mag"]
    df_valid["w_severity_frac"] = df_valid["weight_energy_predicted"] * df_valid["severity_fraction"]
    df_valid["w_dRIX_sev_weighted"] = df_valid["weight_energy_predicted"] * df_valid["dRIX_severity_weighted"]

    if "RIX_avg_03" in df_valid.columns:
        df_valid["w_RIX_avg_03"] = df_valid["weight_energy_predicted"] * df_valid["RIX_avg_03"]

    agg_dict = {
        "ey_deviation": ("w_ey_dev", "sum"),
        "dRIX_03_signed": ("w_dRIX_03", "sum"),
        "dRIX_03_mag": ("w_dRIX_03_mag", "sum"),
        "severity_fraction": ("w_severity_frac", "sum"),
        "dRIX_severity_weighted": ("w_dRIX_sev_weighted", "sum"),
    }

    if "w_RIX_avg_03" in df_valid.columns:
        agg_dict["RIX_avg_03"] = ("w_RIX_avg_03", "sum")

    pair_agg = df_valid.groupby("pair_id").agg(**agg_dict).reset_index()

    print(f"\nPairs: {len(pair_agg)}")

    y = pair_agg["ey_deviation"].values
    abs_y = np.abs(y)

    # =========================================
    # MEAN MODEL (μ) CORRELATIONS
    # =========================================

    print("\n" + "-" * 50)
    print("PAIR-LEVEL: MEAN MODEL (μ) - Directional effects")
    print("-" * 50)

    mu_features = [
        ("dRIX_03_signed", "dRIX_0.3 (signed)"),
        ("dRIX_severity_weighted", "dRIX_0.3 × severity_fraction"),
    ]

    pair_mu_results = []
    for col, label in mu_features:
        if col not in pair_agg.columns:
            continue
        x = pair_agg[col].values
        valid = np.isfinite(x) & np.isfinite(y)

        r, p = stats.pearsonr(x[valid], y[valid])
        pair_mu_results.append((col, label, r, p))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    # =========================================
    # SIGMA MODEL (σ) CORRELATIONS
    # =========================================

    print("\n" + "-" * 50)
    print("PAIR-LEVEL: SIGMA MODEL (σ) - Spread effects")
    print("-" * 50)

    sigma_features = [
        ("dRIX_03_mag", "|dRIX_0.3| (magnitude)"),
        ("RIX_avg_03", "RIX_avg_0.3 (absolute severity)"),
        ("severity_fraction", "Severity fraction"),
    ]

    pair_sigma_results = []
    for col, label in sigma_features:
        if col not in pair_agg.columns:
            continue
        x = pair_agg[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        if valid.sum() < 10:
            continue

        r, p = stats.pearsonr(x[valid], abs_y[valid])
        pair_sigma_results.append((col, label, r, p))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    return pair_agg, pair_mu_results, pair_sigma_results


def analyze_rix_avg_additional_value(df):
    """Check if RIX_avg adds information beyond dRIX for sigma model."""

    print("\n" + "=" * 70)
    print("DOES RIX_avg ADD VALUE BEYOND |dRIX| FOR SIGMA?")
    print("=" * 70)

    if "RIX_avg_0.3_sector" not in df.columns:
        print("RIX_avg_0.3_sector not available")
        return

    required = ["EY_deviation_sector_frac", "dRIX_0.3_sector", "RIX_avg_0.3_sector"]
    df_valid = df.dropna(subset=required).copy()

    abs_y = np.abs(df_valid["EY_deviation_sector_frac"].values)
    abs_dRIX = np.abs(df_valid["dRIX_0.3_sector"].values)
    RIX_avg = df_valid["RIX_avg_0.3_sector"].values

    # Simple correlation
    r1, p1 = stats.pearsonr(abs_dRIX, abs_y)
    r2, p2 = stats.pearsonr(RIX_avg, abs_y)

    print(f"\nSimple correlations with |EY deviation|:")
    print(f"  |dRIX_0.3|: r = {r1:+.3f} (p = {p1:.4f})")
    print(f"  RIX_avg_0.3: r = {r2:+.3f} (p = {p2:.4f})")

    # Partial correlation: does RIX_avg add info after controlling for |dRIX|?
    # Regress |EY| on |dRIX|, check if residuals correlate with RIX_avg

    slope, intercept = np.polyfit(abs_dRIX, abs_y, 1)
    residuals = abs_y - (slope * abs_dRIX + intercept)

    r_partial, p_partial = stats.pearsonr(RIX_avg, residuals)

    print(f"\nPartial correlation (RIX_avg vs |EY| residuals after controlling for |dRIX|):")
    print(f"  r = {r_partial:+.3f} (p = {p_partial:.4f})")

    if r_partial > 0 and p_partial < 0.05:
        print("\n→ YES: RIX_avg adds information beyond |dRIX|")
        print("→ Interpretation: Even when complexity DIFFERENCE is small,")
        print("   high absolute complexity increases uncertainty")
    else:
        print("\n→ NO: RIX_avg doesn't add significant information beyond |dRIX|")

    # Also check correlation between |dRIX| and RIX_avg
    r_collinear, _ = stats.pearsonr(abs_dRIX, RIX_avg)
    print(f"\nCorrelation between |dRIX_0.3| and RIX_avg_0.3: r = {r_collinear:.3f}")

    if abs(r_collinear) > 0.7:
        print("→ WARNING: High collinearity - they capture similar information")
    else:
        print("→ Low collinearity - they capture different aspects")


def create_visualizations(df_sector, pair_agg):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    y_sector = df_sector["EY_deviation_sector_frac"].values
    y_pair = pair_agg["ey_deviation"].values

    # =========================================
    # ROW 1: SECTOR LEVEL
    # =========================================

    # 1. Severity fraction distribution
    ax = axes[0, 0]
    sev_frac = df_sector["severity_fraction"].values
    ax.hist(sev_frac, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', label='0.5 threshold')
    ax.set_xlabel("Severity Fraction")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Severity Fraction\n(0 = all mild, 1 = all severe)")
    ax.legend()

    # 2. dRIX_0.3 vs EY deviation (sector)
    ax = axes[0, 1]
    x = df_sector["dRIX_03_signed"].values
    ax.scatter(x, y_sector, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(y_sector)
    slope, intercept, r, p, se = stats.linregress(x[valid], y_sector[valid])
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("dRIX_0.3 (signed)")
    ax.set_ylabel("EY deviation")
    ax.set_title(f"Sector: dRIX_0.3 vs EY\nr = {r:.3f}")

    # 3. dRIX × severity_fraction vs EY (sector)
    ax = axes[0, 2]
    x = df_sector["dRIX_severity_weighted"].values
    ax.scatter(x, y_sector, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(y_sector)
    if valid.sum() > 2:
        slope, intercept, r, p, se = stats.linregress(x[valid], y_sector[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        title = f"Sector: dRIX × severity vs EY\nr = {r:.3f}"
    else:
        title = "Sector: dRIX × severity vs EY"

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("dRIX_0.3 × severity_fraction")
    ax.set_ylabel("EY deviation")
    ax.set_title(title)

    # =========================================
    # ROW 2: PAIR LEVEL
    # =========================================

    # 4. dRIX_0.3 vs EY (pair)
    ax = axes[1, 0]
    x = pair_agg["dRIX_03_signed"].values
    ax.scatter(x, y_pair, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(y_pair)
    slope, intercept, r, p, se = stats.linregress(x[valid], y_pair[valid])
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("dRIX_0.3 (energy-weighted)")
    ax.set_ylabel("Overall EY deviation")
    ax.set_title(f"Pair: dRIX_0.3 vs EY\nr = {r:.3f}")

    # 5. dRIX × severity vs EY (pair)
    ax = axes[1, 1]
    x = pair_agg["dRIX_severity_weighted"].values
    ax.scatter(x, y_pair, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(y_pair)
    if valid.sum() > 2:
        slope, intercept, r, p, se = stats.linregress(x[valid], y_pair[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        title = f"Pair: dRIX × severity vs EY\nr = {r:.3f}"
    else:
        title = "Pair: dRIX × severity vs EY"

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("dRIX_0.3 × severity_fraction (energy-weighted)")
    ax.set_ylabel("Overall EY deviation")
    ax.set_title(title)

    # 6. |dRIX_0.3| vs |EY| and RIX_avg vs |EY| comparison (pair)
    ax = axes[1, 2]
    abs_y_pair = np.abs(y_pair)

    x1 = pair_agg["dRIX_03_mag"].values
    valid1 = np.isfinite(x1) & np.isfinite(abs_y_pair)
    r1, _ = stats.pearsonr(x1[valid1], abs_y_pair[valid1])
    ax.scatter(x1, abs_y_pair, alpha=0.6, label=f"|dRIX_0.3| (r={r1:.3f})", color='blue')

    if "RIX_avg_03" in pair_agg.columns:
        x2 = pair_agg["RIX_avg_03"].values
        valid2 = np.isfinite(x2) & np.isfinite(abs_y_pair)
        if valid2.sum() > 10:
            r2, _ = stats.pearsonr(x2[valid2], abs_y_pair[valid2])
            # Normalize for comparison
            x2_norm = (x2 - x2.mean()) / x2.std() * x1.std() + x1.mean()
            ax.scatter(x2_norm, abs_y_pair, alpha=0.6, label=f"RIX_avg (r={r2:.3f})", color='green', marker='x')

    ax.set_xlabel("Magnitude (normalized)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title("Sigma drivers: |dRIX| vs RIX_avg")
    ax.legend()

    plt.tight_layout()
    plt.savefig("dRIX_option1_option5_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: dRIX_option1_option5_exploration.png")


def print_recommendations(sector_mu, sector_sigma, pair_mu, pair_sigma):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    # =========================================
    # OPTION 1 EVALUATION
    # =========================================

    print("\n" + "-" * 50)
    print("OPTION 1: Severity Fraction")
    print("-" * 50)

    # Compare dRIX_0.3 alone vs dRIX × severity
    dRIX_only = [r for r in pair_mu if r[0] == "dRIX_03_signed"]
    dRIX_weighted = [r for r in pair_mu if r[0] == "dRIX_severity_weighted"]

    if dRIX_only and dRIX_weighted:
        r_only = abs(dRIX_only[0][2])
        r_weighted = abs(dRIX_weighted[0][2])

        print(f"\nPair-level correlations:")
        print(f"  dRIX_0.3 alone:            r = {dRIX_only[0][2]:+.4f}")
        print(f"  dRIX_0.3 × severity_frac:  r = {dRIX_weighted[0][2]:+.4f}")

        if r_weighted > r_only:
            print(f"\n→ ✓ Severity weighting IMPROVES prediction by {((r_weighted / r_only) - 1) * 100:.1f}%")
            print("→ Supports theory: concentrated severe terrain differences matter more")
        else:
            print(f"\n→ ✗ Severity weighting does NOT improve prediction")
            print("→ Simple dRIX_0.3 is sufficient")

    # =========================================
    # OPTION 5 EVALUATION
    # =========================================

    print("\n" + "-" * 50)
    print("OPTION 5: Separate Features")
    print("-" * 50)

    print("\nFor MEAN model (μ):")
    for col, label, r, p in pair_mu:
        if col == "dRIX_03_signed":
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label}: r = {r:+.4f} {sig}")
            if abs(r) > 0.1 and p < 0.05:
                print(f"  → USE in μ model")
            else:
                print(f"  → Effect is weak, consider removing from μ")

    print("\nFor SIGMA model (σ):")
    for col, label, r, p in pair_sigma:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label}: r = {r:+.4f} {sig}")
        if r > 0.1 and p < 0.05:
            print(f"  → USE in σ model")
        elif r > 0:
            print(f"  → Weak positive effect, optional")
        else:
            print(f"  → No effect, skip")

    # =========================================
    # OVERALL RECOMMENDATION
    # =========================================

    print("\n" + "-" * 50)
    print("SUMMARY RECOMMENDATION")
    print("-" * 50)

    print("""
Based on this analysis, for your Bayesian model:

MEAN MODEL (μ):
  - Keep: dRIX_0.3 (signed) if correlation is meaningful
  - Test: dRIX × severity_fraction if Option 1 shows improvement

SIGMA MODEL (σ):
  - Keep: |dRIX_0.3| for magnitude of complexity difference
  - Add: RIX_avg_0.3 if it shows independent contribution
  - Test: severity_fraction if it correlates with |EY|
""")


def main():
    print("=" * 70)
    print("dRIX EXPLORATION: Option 1 and Option 5")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    # Create derived features
    df = create_derived_features(df)

    # Sector-level analysis
    df_sector, sector_mu, sector_sigma = analyze_sector_level(df)

    # Pair-level analysis
    pair_agg, pair_mu, pair_sigma = analyze_pair_level(df)

    # Check if RIX_avg adds value
    analyze_rix_avg_additional_value(df)

    # Visualizations
    create_visualizations(df_sector, pair_agg)

    # Recommendations
    print_recommendations(sector_mu, sector_sigma, pair_mu, pair_sigma)

    return df, df_sector, pair_agg


if __name__ == "__main__":
    df, df_sector, pair_agg = main()