"""
Exploration: Deflection Metrics and Speedup Gradient Interactions

Metrics to test:
1. flip_fraction_weighted_by_predicted_power - samples that actually flipped
2. edge_fraction_weighted_by_predicted_power - samples at risk of flipping
3. speedup_gradient_directed - gradient to the relevant adjacent sector (based on turn direction)
4. flip_fraction × speedup_gradient - MAIN HYPOTHESIS
5. edge_fraction × speedup_gradient - alternative

Theory:
- When samples flip to adjacent sector, they get "wrong" speedup applied
- The error magnitude depends on:
  a) How many samples flip (flip_fraction)
  b) How different the speedups are (speedup_gradient)
- So the interaction term should be the best predictor
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


def load_data(path):
    """Load data and verify required columns exist."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("VERIFYING REQUIRED COLUMNS")
    print("=" * 70)

    required_cols = [
        "pair_id", "sector_name",
        "EY_deviation_sector_frac",
        "weight_energy_predicted",
        "d_turning_deg",
        "abs_d_turning_deg",
        "flip_fraction_weighted_by_predicted_power",
        "edge_fraction_weighted_by_predicted_power",
        "overall_speedup_WTG_factor",
        "overall_speedup_MM_factor",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"\n⚠ MISSING COLUMNS: {missing}")
        print("\nAvailable columns containing 'flip', 'edge', 'turn':")
        for c in df.columns:
            if any(kw in c.lower() for kw in ['flip', 'edge', 'turn']):
                print(f"  {c}")
    else:
        print("\n✓ All required columns found!")

    return df


def compute_speedup_gradient_directed(df):
    """
    Compute speedup gradient to the RELEVANT adjacent sector based on turn direction.

    If turn > 0: samples move to NEXT sector → gradient to next
    If turn < 0: samples move to PREVIOUS sector → gradient to previous

    Using log difference for symmetry: |log(current) - log(adjacent)|
    """

    print("\n" + "=" * 70)
    print("COMPUTING DIRECTED SPEEDUP GRADIENT")
    print("=" * 70)

    df = df.copy()
    df["sector_idx"] = df["sector_name"].map(SECTOR_TO_IDX)

    # Get all pairs
    pairs = df["pair_id"].unique()

    results = []

    for pair_id in pairs:
        pair_data = df[df["pair_id"] == pair_id].copy()

        if len(pair_data) != 12:
            continue

        # Sort by sector
        pair_data = pair_data.sort_values("sector_idx")

        # Get arrays
        speedups = pair_data["overall_speedup_WTG_factor"].values
        turns = pair_data["d_turning_deg"].values
        sector_names = pair_data["sector_name"].values
        sector_indices = pair_data["sector_idx"].values

        for i in range(12):
            current_speedup = speedups[i]
            turn = turns[i]

            # Determine which adjacent sector is relevant based on turn direction
            if turn > 0:
                # Samples move to NEXT sector (clockwise in compass terms)
                adjacent_idx = (i + 1) % 12
                adjacent_label = "next"
            elif turn < 0:
                # Samples move to PREVIOUS sector (counter-clockwise)
                adjacent_idx = (i - 1) % 12
                adjacent_label = "prev"
            else:
                # No turning, no gradient matters
                adjacent_idx = i
                adjacent_label = "none"

            adjacent_speedup = speedups[adjacent_idx]

            # Log difference (symmetric)
            log_current = np.log(np.clip(current_speedup, 1e-6, None))
            log_adjacent = np.log(np.clip(adjacent_speedup, 1e-6, None))

            speedup_gradient_log = abs(log_current - log_adjacent)

            # Also compute simple difference for comparison
            speedup_gradient_simple = abs(current_speedup - adjacent_speedup)

            results.append({
                "pair_id": pair_id,
                "sector_name": sector_names[i],
                "sector_idx": sector_indices[i],
                "d_turning_deg": turn,
                "adjacent_sector": adjacent_label,
                "current_speedup": current_speedup,
                "adjacent_speedup": adjacent_speedup,
                "speedup_gradient_log": speedup_gradient_log,
                "speedup_gradient_simple": speedup_gradient_simple,
            })

    grad_df = pd.DataFrame(results)

    print(f"\nComputed gradients for {len(grad_df)} sector-pair combinations")
    print(f"\nSpeedup gradient (log) statistics:")
    print(f"  Mean: {grad_df['speedup_gradient_log'].mean():.4f}")
    print(f"  Std: {grad_df['speedup_gradient_log'].std():.4f}")
    print(f"  Range: [{grad_df['speedup_gradient_log'].min():.4f}, {grad_df['speedup_gradient_log'].max():.4f}]")

    # Merge back
    df_merged = df.merge(
        grad_df[["pair_id", "sector_name", "speedup_gradient_log", "speedup_gradient_simple",
                 "adjacent_speedup", "adjacent_sector"]],
        on=["pair_id", "sector_name"],
        how="left"
    )

    return df_merged


def create_interaction_terms(df):
    """Create interaction terms between flip/edge fractions and speedup gradient."""

    print("\n" + "=" * 70)
    print("CREATING INTERACTION TERMS")
    print("=" * 70)

    df = df.copy()

    # Interaction: flip_fraction × speedup_gradient
    df["flip_x_gradient_log"] = (
            df["flip_fraction_weighted_by_predicted_power"] * df["speedup_gradient_log"]
    )

    df["flip_x_gradient_simple"] = (
            df["flip_fraction_weighted_by_predicted_power"] * df["speedup_gradient_simple"]
    )

    # Interaction: edge_fraction × speedup_gradient
    df["edge_x_gradient_log"] = (
            df["edge_fraction_weighted_by_predicted_power"] * df["speedup_gradient_log"]
    )

    df["edge_x_gradient_simple"] = (
            df["edge_fraction_weighted_by_predicted_power"] * df["speedup_gradient_simple"]
    )

    print("\nInteraction terms created:")
    print(f"  flip_x_gradient_log:    mean={df['flip_x_gradient_log'].mean():.4f}")
    print(f"  edge_x_gradient_log:    mean={df['edge_x_gradient_log'].mean():.4f}")

    return df


def analyze_sector_level(df):
    """Test all metrics at sector level."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    required = [
        "EY_deviation_sector_frac",
        "flip_fraction_weighted_by_predicted_power",
        "edge_fraction_weighted_by_predicted_power",
        "abs_d_turning_deg",
        "speedup_gradient_log",
        "flip_x_gradient_log",
        "edge_x_gradient_log",
    ]

    df_valid = df.dropna(subset=required).copy()
    print(f"\nValid sector observations: {len(df_valid)}")

    y = df_valid["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)

    # =========================================
    # BASIC STATISTICS
    # =========================================

    print("\n" + "-" * 50)
    print("BASIC STATISTICS")
    print("-" * 50)

    for col in ["flip_fraction_weighted_by_predicted_power",
                "edge_fraction_weighted_by_predicted_power",
                "abs_d_turning_deg",
                "speedup_gradient_log"]:
        data = df_valid[col].values
        print(f"\n{col}:")
        print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")

    # =========================================
    # CORRELATIONS WITH |EY DEVIATION|
    # =========================================

    print("\n" + "-" * 50)
    print("CORRELATIONS WITH |EY DEVIATION| (for SIGMA model)")
    print("-" * 50)

    metrics_to_test = [
        ("abs_d_turning_deg", "|turning angle|"),
        ("flip_fraction_weighted_by_predicted_power", "flip_fraction (power-weighted)"),
        ("edge_fraction_weighted_by_predicted_power", "edge_fraction (power-weighted)"),
        ("speedup_gradient_log", "speedup_gradient (log)"),
        ("flip_x_gradient_log", "flip_fraction × speedup_gradient [MAIN]"),
        ("edge_x_gradient_log", "edge_fraction × speedup_gradient"),
    ]

    results = []

    for col, label in metrics_to_test:
        x = df_valid[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        r, p = stats.pearsonr(x[valid], abs_y[valid])
        rho, _ = stats.spearmanr(x[valid], abs_y[valid])

        results.append({
            "Metric": label,
            "Column": col,
            "Pearson_r": r,
            "Spearman_rho": rho,
            "p_value": p,
        })

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r  = {r:+.4f} {sig}")
        print(f"  Spearman ρ = {rho:+.4f}")

    print("\n  (* p<0.05, ** p<0.01, *** p<0.001)")

    # =========================================
    # COMPARE FLIP vs EDGE
    # =========================================

    print("\n" + "-" * 50)
    print("FLIP FRACTION vs EDGE FRACTION COMPARISON")
    print("-" * 50)

    flip = df_valid["flip_fraction_weighted_by_predicted_power"].values
    edge = df_valid["edge_fraction_weighted_by_predicted_power"].values

    print(f"\nCorrelation between flip and edge: r = {np.corrcoef(flip, edge)[0, 1]:.3f}")
    print(f"\nCases where flip > edge: {(flip > edge).sum()} ({(flip > edge).mean() * 100:.1f}%)")
    print(f"Cases where flip ≤ edge: {(flip <= edge).sum()} ({(flip <= edge).mean() * 100:.1f}%)")

    # Which is more predictive?
    r_flip, _ = stats.pearsonr(flip, abs_y)
    r_edge, _ = stats.pearsonr(edge, abs_y)

    print(f"\nPredictive power comparison:")
    print(f"  flip_fraction vs |EY|: r = {r_flip:+.4f}")
    print(f"  edge_fraction vs |EY|: r = {r_edge:+.4f}")

    if abs(r_flip) > abs(r_edge):
        print("  → flip_fraction is MORE predictive")
    else:
        print("  → edge_fraction is MORE predictive")

    return df_valid, results


def analyze_pair_level(df):
    """Aggregate to pair level and test."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted)")
    print("=" * 70)

    required = [
        "pair_id",
        "EY_deviation_sector_frac",
        "weight_energy",
        "weight_energy_predicted",
        "flip_fraction_weighted_by_predicted_power",
        "edge_fraction_weighted_by_predicted_power",
        "abs_d_turning_deg",
        "speedup_gradient_log",
        "flip_x_gradient_log",
        "edge_x_gradient_log",
    ]

    df_valid = df.dropna(subset=required).copy()

    # Energy-weighted aggregation
    df_valid["w_ey_dev"] = df_valid["weight_energy"] * df_valid["EY_deviation_sector_frac"]
    df_valid["w_flip"] = df_valid["weight_energy_predicted"] * df_valid["flip_fraction_weighted_by_predicted_power"]
    df_valid["w_edge"] = df_valid["weight_energy_predicted"] * df_valid["edge_fraction_weighted_by_predicted_power"]
    df_valid["w_abs_turn"] = df_valid["weight_energy_predicted"] * df_valid["abs_d_turning_deg"]
    df_valid["w_gradient"] = df_valid["weight_energy_predicted"] * df_valid["speedup_gradient_log"]
    df_valid["w_flip_x_grad"] = df_valid["weight_energy_predicted"] * df_valid["flip_x_gradient_log"]
    df_valid["w_edge_x_grad"] = df_valid["weight_energy_predicted"] * df_valid["edge_x_gradient_log"]

    pair_agg = df_valid.groupby("pair_id").agg(
        ey_deviation=("w_ey_dev", "sum"),
        flip_fraction=("w_flip", "sum"),
        edge_fraction=("w_edge", "sum"),
        abs_turning=("w_abs_turn", "sum"),
        speedup_gradient=("w_gradient", "sum"),
        flip_x_gradient=("w_flip_x_grad", "sum"),
        edge_x_gradient=("w_edge_x_grad", "sum"),
    ).reset_index()

    print(f"\nPairs: {len(pair_agg)}")

    y = pair_agg["ey_deviation"].values
    abs_y = np.abs(y)

    # =========================================
    # CORRELATIONS
    # =========================================

    print("\n" + "-" * 50)
    print("PAIR-LEVEL CORRELATIONS WITH |EY DEVIATION|")
    print("-" * 50)

    metrics = [
        ("abs_turning", "|turning angle|"),
        ("flip_fraction", "flip_fraction"),
        ("edge_fraction", "edge_fraction"),
        ("speedup_gradient", "speedup_gradient"),
        ("flip_x_gradient", "flip × gradient [MAIN]"),
        ("edge_x_gradient", "edge × gradient"),
    ]

    pair_results = []

    for col, label in metrics:
        x = pair_agg[col].values
        valid = np.isfinite(x) & np.isfinite(abs_y)

        r, p = stats.pearsonr(x[valid], abs_y[valid])
        pair_results.append((col, label, r, p))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\n{label}:")
        print(f"  Pearson r = {r:+.4f} {sig}")

    return pair_agg, pair_results


def analyze_interaction_effect(df):
    """
    Test if the interaction (flip × gradient) is better than individual terms.
    """

    print("\n" + "=" * 70)
    print("INTERACTION EFFECT ANALYSIS")
    print("=" * 70)

    required = [
        "EY_deviation_sector_frac",
        "flip_fraction_weighted_by_predicted_power",
        "speedup_gradient_log",
        "flip_x_gradient_log",
    ]

    df_valid = df.dropna(subset=required).copy()

    abs_y = np.abs(df_valid["EY_deviation_sector_frac"].values)
    flip = df_valid["flip_fraction_weighted_by_predicted_power"].values
    gradient = df_valid["speedup_gradient_log"].values
    interaction = df_valid["flip_x_gradient_log"].values

    # Individual correlations
    r_flip, _ = stats.pearsonr(flip, abs_y)
    r_grad, _ = stats.pearsonr(gradient, abs_y)
    r_int, _ = stats.pearsonr(interaction, abs_y)

    print(f"\nIndividual vs Interaction:")
    print(f"  flip_fraction alone:        r = {r_flip:+.4f}")
    print(f"  speedup_gradient alone:     r = {r_grad:+.4f}")
    print(f"  flip × gradient:            r = {r_int:+.4f}")

    # Is interaction better than the sum of parts?
    print(f"\n  Analysis:")
    if abs(r_int) > abs(r_flip) and abs(r_int) > abs(r_grad):
        print("  ✓ Interaction is BETTER than either individual term")
        print("  → Supports theory: flips only matter when speedups differ")
    elif abs(r_int) > max(abs(r_flip), abs(r_grad)):
        print("  ✓ Interaction is better than the best individual term")
    else:
        best = "flip_fraction" if abs(r_flip) > abs(r_grad) else "speedup_gradient"
        print(f"  ✗ Interaction is NOT better than {best}")
        print(f"  → May want to use {best} alone instead")

    # Stratified analysis
    print("\n" + "-" * 50)
    print("STRATIFIED ANALYSIS: High vs Low Gradient")
    print("-" * 50)

    grad_median = np.median(gradient)

    low_grad = gradient <= grad_median
    high_grad = gradient > grad_median

    r_flip_low, _ = stats.pearsonr(flip[low_grad], abs_y[low_grad])
    r_flip_high, _ = stats.pearsonr(flip[high_grad], abs_y[high_grad])

    print(f"\n  When speedup_gradient is LOW (≤{grad_median:.4f}):")
    print(f"    flip_fraction vs |EY|: r = {r_flip_low:+.4f} (n={low_grad.sum()})")

    print(f"\n  When speedup_gradient is HIGH (>{grad_median:.4f}):")
    print(f"    flip_fraction vs |EY|: r = {r_flip_high:+.4f} (n={high_grad.sum()})")

    if abs(r_flip_high) > abs(r_flip_low):
        print(f"\n  ✓ Supports theory: flip_fraction matters MORE when gradient is high")
    else:
        print(f"\n  ✗ Does NOT support theory")


def create_visualizations(df_sector, pair_agg):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_y_sector = np.abs(df_sector["EY_deviation_sector_frac"].values)
    abs_y_pair = np.abs(pair_agg["ey_deviation"].values)

    # =========================================
    # ROW 1: SECTOR LEVEL
    # =========================================

    # 1. flip_fraction vs |EY|
    ax = axes[0, 0]
    x = df_sector["flip_fraction_weighted_by_predicted_power"].values
    ax.scatter(x, abs_y_sector, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(abs_y_sector)
    r, _ = stats.pearsonr(x[valid], abs_y_sector[valid])

    ax.set_xlabel("flip_fraction (power-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: flip_fraction vs |EY|\nr = {r:.3f}")

    # 2. speedup_gradient vs |EY|
    ax = axes[0, 1]
    x = df_sector["speedup_gradient_log"].values
    ax.scatter(x, abs_y_sector, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(abs_y_sector)
    r, _ = stats.pearsonr(x[valid], abs_y_sector[valid])

    ax.set_xlabel("speedup_gradient (log)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: speedup_gradient vs |EY|\nr = {r:.3f}")

    # 3. flip × gradient vs |EY| (MAIN)
    ax = axes[0, 2]
    x = df_sector["flip_x_gradient_log"].values
    ax.scatter(x, abs_y_sector, alpha=0.3, s=10)

    valid = np.isfinite(x) & np.isfinite(abs_y_sector)
    r, _ = stats.pearsonr(x[valid], abs_y_sector[valid])

    slope, intercept = np.polyfit(x[valid], abs_y_sector[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("flip_fraction × speedup_gradient")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Sector: INTERACTION vs |EY|\nr = {r:.3f}")

    # =========================================
    # ROW 2: PAIR LEVEL
    # =========================================

    # 4. flip_fraction vs |EY| (pair)
    ax = axes[1, 0]
    x = pair_agg["flip_fraction"].values
    ax.scatter(x, abs_y_pair, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(abs_y_pair)
    r, _ = stats.pearsonr(x[valid], abs_y_pair[valid])

    slope, intercept = np.polyfit(x[valid], abs_y_pair[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("flip_fraction (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: flip_fraction vs |EY|\nr = {r:.3f}")

    # 5. speedup_gradient vs |EY| (pair)
    ax = axes[1, 1]
    x = pair_agg["speedup_gradient"].values
    ax.scatter(x, abs_y_pair, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(abs_y_pair)
    r, _ = stats.pearsonr(x[valid], abs_y_pair[valid])

    slope, intercept = np.polyfit(x[valid], abs_y_pair[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("speedup_gradient (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: speedup_gradient vs |EY|\nr = {r:.3f}")

    # 6. flip × gradient vs |EY| (pair) - MAIN
    ax = axes[1, 2]
    x = pair_agg["flip_x_gradient"].values
    ax.scatter(x, abs_y_pair, alpha=0.6)

    valid = np.isfinite(x) & np.isfinite(abs_y_pair)
    r, _ = stats.pearsonr(x[valid], abs_y_pair[valid])

    slope, intercept = np.polyfit(x[valid], abs_y_pair[valid], 1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel("flip × gradient (energy-weighted)")
    ax.set_ylabel("|EY deviation|")
    ax.set_title(f"Pair: INTERACTION vs |EY|\nr = {r:.3f}")

    plt.tight_layout()
    plt.savefig("deflection_flip_gradient_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: deflection_flip_gradient_exploration.png")


def print_recommendations(sector_results, pair_results):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    # Sort sector results by |correlation|
    sector_sorted = sorted(sector_results, key=lambda x: abs(x["Pearson_r"]), reverse=True)

    print("\n" + "-" * 50)
    print("SECTOR-LEVEL RANKING (by |correlation|)")
    print("-" * 50)

    print(f"\n{'Rank':<6} {'Metric':<45} {'r':<10} {'Sig'}")
    print("-" * 70)
    for i, row in enumerate(sector_sorted, 1):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row[
                                                                                                 "p_value"] < 0.05 else ""
        print(f"{i:<6} {row['Metric']:<45} {row['Pearson_r']:+.4f}    {sig}")

    # Sort pair results
    pair_sorted = sorted(pair_results, key=lambda x: abs(x[2]), reverse=True)

    print("\n" + "-" * 50)
    print("PAIR-LEVEL RANKING (by |correlation|)")
    print("-" * 50)

    print(f"\n{'Rank':<6} {'Metric':<45} {'r':<10} {'Sig'}")
    print("-" * 70)
    for i, (col, label, r, p) in enumerate(pair_sorted, 1):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{i:<6} {label:<45} {r:+.4f}    {sig}")

    # Recommendation
    print("\n" + "-" * 50)
    print("RECOMMENDATION FOR BAYESIAN MODEL")
    print("-" * 50)

    best_sector = sector_sorted[0]
    best_pair = pair_sorted[0]

    print(f"""
Based on correlation analysis:

BEST METRIC AT SECTOR LEVEL: {best_sector['Metric']}
  r = {best_sector['Pearson_r']:+.4f}

BEST METRIC AT PAIR LEVEL: {best_pair[1]}
  r = {best_pair[2]:+.4f}

RECOMMENDED ACTION:
  Add the best-performing metric to your sigma model as:

  log(σ) = ... + γ_flip_gradient × flip_fraction × speedup_gradient

  OR if individual terms perform better:

  log(σ) = ... + γ_flip × flip_fraction
""")


def main():
    print("=" * 70)
    print("DEFLECTION METRICS EXPLORATION")
    print("flip_fraction, edge_fraction, speedup_gradient, and interactions")
    print("=" * 70)

    # Load data
    df = load_data(sector_model_path)

    # Compute directed speedup gradient
    df = compute_speedup_gradient_directed(df)

    # Create interaction terms
    df = create_interaction_terms(df)

    # Sector-level analysis
    df_sector, sector_results = analyze_sector_level(df)

    # Pair-level analysis
    pair_agg, pair_results = analyze_pair_level(df)

    # Interaction effect analysis
    analyze_interaction_effect(df)

    # Visualizations
    create_visualizations(df_sector, pair_agg)

    # Recommendations
    print_recommendations(sector_results, pair_results)

    return df, df_sector, pair_agg


if __name__ == "__main__":
    df, df_sector, pair_agg = main()