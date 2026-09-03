"""
Additional Sigma Driver Exploration

Features to test:
1. Speedup variability (two versions):
   - std(MM - WTG) across sectors (difference variability)
   - std(WTG/MM) across sectors (ratio variability)

2. dz × distance interaction:
   - |dz| × distance_score

3. Turning angle variability:
   - std(|d_turning_deg|) across sectors

All tested against both |EY| and |WS| deviation.
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
    """Load data and compute deviations."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("ADDITIONAL SIGMA DRIVER EXPLORATION")
    print("=" * 70)

    required = [
        "pair_id", "sector_name",
        "EY_deviation_sector_frac",
        "weight_energy_predicted",
        "Mean_windspeed_predicted", "Mean_windspeed_self",
        "overall_speedup_WTG_factor", "overall_speedup_MM_factor",
        "d_overall_speedup_factor",
        "d_turning_deg",
        "dz", "distance_m", "distance_A", "distance_B",
    ]

    df = df.dropna(subset=required).copy()

    # Compute WS deviation
    ws_actual = df["Mean_windspeed_self"].values
    ws_pred = df["Mean_windspeed_predicted"].values
    ws_denom = np.maximum(ws_actual, 3.0)
    df["WS_deviation_rel"] = np.abs(ws_pred - ws_actual) / ws_denom

    # Compute speedup difference (MM - WTG)
    df["speedup_diff"] = df["overall_speedup_MM_factor"] - df["overall_speedup_WTG_factor"]

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    return df


def compute_pair_features(df):
    """Compute all new features at pair level."""

    print("\n" + "=" * 70)
    print("COMPUTING PAIR-LEVEL FEATURES")
    print("=" * 70)

    results = []

    for pair_id, group in df.groupby("pair_id"):
        # Weights
        weights = group["weight_energy_predicted"].values
        weights = weights / weights.sum()

        # =========================================
        # 1. SPEEDUP VARIABILITY
        # =========================================

        # Version A: Std of difference (MM - WTG)
        speedup_diff = group["speedup_diff"].values
        speedup_diff_std = np.std(speedup_diff)
        speedup_diff_range = np.max(speedup_diff) - np.min(speedup_diff)

        # Does sign flip? (some positive, some negative)
        has_sign_flip = (speedup_diff.min() < 0) and (speedup_diff.max() > 0)

        # Version B: Std of ratio (WTG/MM = d_overall_speedup_factor)
        speedup_ratio = group["d_overall_speedup_factor"].values
        speedup_ratio_std = np.std(speedup_ratio)
        speedup_ratio_range = np.max(speedup_ratio) - np.min(speedup_ratio)

        # Version C: Std of log ratio (what we already use in model)
        log_ratio = np.log(np.clip(speedup_ratio, 1e-6, None))
        log_ratio_std = np.std(log_ratio)

        # =========================================
        # 2. dz × DISTANCE INTERACTION
        # =========================================

        dz = group["dz"].iloc[0]
        dz_abs = np.abs(dz)

        dist_m = group["distance_m"].iloc[0]
        dist_A = group["distance_A"].iloc[0]
        dist_B = group["distance_B"].iloc[0]
        denom = max(dist_B - dist_A, 1e-6)
        dist_score = max(dist_m - dist_A, 0.0) / denom

        dz_x_dist = dz_abs * dist_score
        dz_x_dist_raw = dz_abs * dist_m  # Also test raw distance

        # =========================================
        # 3. TURNING ANGLE VARIABILITY
        # =========================================

        turning = group["d_turning_deg"].values
        turning_abs = np.abs(turning)
        turning_std = np.std(turning_abs)
        turning_range = np.max(turning_abs) - np.min(turning_abs)
        turning_mean = np.mean(turning_abs)

        # Does turning flip direction? (some positive, some negative)
        has_turning_flip = (turning.min() < 0) and (turning.max() > 0)

        # =========================================
        # 4. AGGREGATE DEVIATIONS
        # =========================================

        ey_dev = np.sum(weights * group["EY_deviation_sector_frac"].values)
        ws_dev = np.sum(weights * group["WS_deviation_rel"].values)

        location = group["location"].iloc[0] if "location" in group.columns else pair_id

        results.append({
            "pair_id": pair_id,
            "location": location,
            # Speedup variability
            "speedup_diff_std": speedup_diff_std,
            "speedup_diff_range": speedup_diff_range,
            "speedup_ratio_std": speedup_ratio_std,
            "speedup_ratio_range": speedup_ratio_range,
            "log_ratio_std": log_ratio_std,
            "has_sign_flip": int(has_sign_flip),
            # dz × distance
            "dz_abs": dz_abs,
            "dist_score": dist_score,
            "dist_m": dist_m,
            "dz_x_dist": dz_x_dist,
            "dz_x_dist_raw": dz_x_dist_raw,
            # Turning variability
            "turning_std": turning_std,
            "turning_range": turning_range,
            "turning_mean": turning_mean,
            "has_turning_flip": int(has_turning_flip),
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
        ("speedup_diff_std", "Speedup diff std (MM-WTG)"),
        ("speedup_ratio_std", "Speedup ratio std (WTG/MM)"),
        ("log_ratio_std", "Log ratio std"),
        ("dz_x_dist", "dz × distance"),
        ("turning_std", "Turning angle std"),
    ]

    for col, label in features:
        vals = pair_df[col]
        print(f"\n  {label}:")
        print(f"    Range: [{vals.min():.4f}, {vals.max():.4f}]")
        print(f"    Mean: {vals.mean():.4f}")
        print(f"    Median: {vals.median():.4f}")

    # Sign flip statistics
    print(f"\n  Pairs with speedup sign flip: {pair_df['has_sign_flip'].sum()} / {len(pair_df)}")
    print(f"  Pairs with turning sign flip: {pair_df['has_turning_flip'].sum()} / {len(pair_df)}")

    return pair_df


def analyze_correlations(pair_df):
    """Test correlations with |EY| and |WS| deviation."""

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    features = [
        # Speedup variability
        ("speedup_diff_std", "Speedup diff std (MM-WTG)"),
        ("speedup_diff_range", "Speedup diff range"),
        ("speedup_ratio_std", "Speedup ratio std"),
        ("speedup_ratio_range", "Speedup ratio range"),
        ("log_ratio_std", "Log ratio std"),
        ("has_sign_flip", "Has speedup sign flip"),
        # dz × distance
        ("dz_x_dist", "dz × distance (score)"),
        ("dz_x_dist_raw", "dz × distance (raw m)"),
        # Turning variability
        ("turning_std", "Turning angle std"),
        ("turning_range", "Turning angle range"),
        ("turning_mean", "Turning angle mean"),
        ("has_turning_flip", "Has turning sign flip"),
    ]

    print(f"\n  {'Feature':<35} {'vs |EY|':<15} {'vs |WS|':<15}")
    print("  " + "-" * 65)

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

        print(f"  {label:<35} r={r_ey:+.3f} {sig_ey:<3} r={r_ws:+.3f} {sig_ws:<3}")

    return pd.DataFrame(results)


def analyze_by_category(pair_df):
    """Analyze grouped comparisons."""

    print("\n" + "=" * 70)
    print("CATEGORICAL ANALYSIS")
    print("=" * 70)

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # =========================================
    # Sign flip analysis
    # =========================================

    print("\n" + "-" * 50)
    print("SPEEDUP SIGN FLIP (MM-WTG changes sign across sectors)")
    print("-" * 50)

    flip = pair_df["has_sign_flip"] == 1
    no_flip = pair_df["has_sign_flip"] == 0

    print(f"\n  No flip (consistent relationship): n={no_flip.sum()}")
    print(f"    Mean |EY|: {abs_ey[no_flip].mean():.4f}")
    print(f"    Mean |WS|: {abs_ws[no_flip].mean():.4f}")

    print(f"\n  Has flip (inconsistent relationship): n={flip.sum()}")
    print(f"    Mean |EY|: {abs_ey[flip].mean():.4f}")
    print(f"    Mean |WS|: {abs_ws[flip].mean():.4f}")

    if flip.sum() > 5 and no_flip.sum() > 5:
        t_ey, p_ey = stats.ttest_ind(abs_ey[flip], abs_ey[no_flip])
        print(f"\n  T-test |EY|: t={t_ey:.2f}, p={p_ey:.4f} {'*' if p_ey < 0.05 else ''}")

    # =========================================
    # Turning flip analysis
    # =========================================

    print("\n" + "-" * 50)
    print("TURNING SIGN FLIP (deflection changes direction)")
    print("-" * 50)

    turn_flip = pair_df["has_turning_flip"] == 1
    turn_no_flip = pair_df["has_turning_flip"] == 0

    print(f"\n  No flip (consistent deflection): n={turn_no_flip.sum()}")
    print(f"    Mean |EY|: {abs_ey[turn_no_flip].mean():.4f}")

    print(f"\n  Has flip (mixed deflection): n={turn_flip.sum()}")
    print(f"    Mean |EY|: {abs_ey[turn_flip].mean():.4f}")


def check_redundancy(pair_df):
    """Check correlation with existing sigma drivers."""

    print("\n" + "=" * 70)
    print("REDUNDANCY CHECK: Correlation with Existing Drivers")
    print("=" * 70)

    new_features = ["speedup_diff_std", "dz_x_dist", "turning_std"]
    existing_features = ["dz_abs", "dist_score"]

    print(f"\n  {'New Feature':<25} {'vs |dz|':<15} {'vs dist_score':<15}")
    print("  " + "-" * 55)

    for new_col in new_features:
        new_vals = pair_df[new_col].values

        r_dz, _ = stats.pearsonr(new_vals, pair_df["dz_abs"].values)
        r_dist, _ = stats.pearsonr(new_vals, pair_df["dist_score"].values)

        flag_dz = "⚠" if abs(r_dz) > 0.5 else ""
        flag_dist = "⚠" if abs(r_dist) > 0.5 else ""

        print(f"  {new_col:<25} r={r_dz:+.3f} {flag_dz:<3} r={r_dist:+.3f} {flag_dist:<3}")

    print("\n  ⚠ = High correlation (>0.5) — feature may be redundant")


def create_visualizations(pair_df):
    """Create diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abs_ey = np.abs(pair_df["ey_deviation"].values)
    abs_ws = pair_df["ws_deviation"].values

    # 1. Speedup diff std vs |EY|
    ax = axes[0, 0]
    ax.scatter(pair_df["speedup_diff_std"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["speedup_diff_std"], abs_ey)
    slope, intercept = np.polyfit(pair_df["speedup_diff_std"], abs_ey, 1)
    x_line = np.linspace(pair_df["speedup_diff_std"].min(), pair_df["speedup_diff_std"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Speedup Diff Std (MM-WTG)")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Speedup Variability vs |EY|\nr = {r:.3f}")

    # 2. dz × distance vs |EY|
    ax = axes[0, 1]
    ax.scatter(pair_df["dz_x_dist"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["dz_x_dist"], abs_ey)
    slope, intercept = np.polyfit(pair_df["dz_x_dist"], abs_ey, 1)
    x_line = np.linspace(pair_df["dz_x_dist"].min(), pair_df["dz_x_dist"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("|dz| × Distance Score")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"dz × Distance Interaction vs |EY|\nr = {r:.3f}")

    # 3. Turning std vs |EY|
    ax = axes[0, 2]
    ax.scatter(pair_df["turning_std"], abs_ey, alpha=0.6)
    r, _ = stats.pearsonr(pair_df["turning_std"], abs_ey)
    slope, intercept = np.polyfit(pair_df["turning_std"], abs_ey, 1)
    x_line = np.linspace(pair_df["turning_std"].min(), pair_df["turning_std"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Turning Angle Std")
    ax.set_ylabel("|EY Deviation|")
    ax.set_title(f"Turning Variability vs |EY|\nr = {r:.3f}")

    # 4. Speedup diff std vs |WS|
    ax = axes[1, 0]
    ax.scatter(pair_df["speedup_diff_std"], abs_ws, alpha=0.6, c='orange')
    r, _ = stats.pearsonr(pair_df["speedup_diff_std"], abs_ws)
    ax.set_xlabel("Speedup Diff Std (MM-WTG)")
    ax.set_ylabel("|WS Deviation|")
    ax.set_title(f"Speedup Variability vs |WS|\nr = {r:.3f}")

    # 5. Comparison: speedup_diff_std vs speedup_ratio_std
    ax = axes[1, 1]
    ax.scatter(pair_df["speedup_diff_std"], pair_df["speedup_ratio_std"], alpha=0.6)
    r, _ = stats.pearsonr(pair_df["speedup_diff_std"], pair_df["speedup_ratio_std"])
    ax.set_xlabel("Speedup Diff Std (MM-WTG)")
    ax.set_ylabel("Speedup Ratio Std (WTG/MM)")
    ax.set_title(f"Diff vs Ratio Variability\nr = {r:.3f}")

    # 6. Sign flip boxplot
    ax = axes[1, 2]
    flip_data = [
        abs_ey[pair_df["has_sign_flip"] == 0],
        abs_ey[pair_df["has_sign_flip"] == 1]
    ]
    bp = ax.boxplot(flip_data, labels=["No Flip", "Has Flip"])
    ax.set_ylabel("|EY Deviation|")
    ax.set_title("Speedup Sign Flip Effect")

    plt.tight_layout()
    plt.savefig("additional_features_exploration.png", dpi=150)
    plt.close()
    print("\nSaved: additional_features_exploration.png")


def print_recommendations(corr_results):
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Group features
    speedup_features = corr_results[corr_results["Column"].str.contains("speedup|ratio|sign")]
    dz_dist_features = corr_results[corr_results["Column"].str.contains("dz_x")]
    turning_features = corr_results[corr_results["Column"].str.contains("turning")]

    print("\n" + "-" * 50)
    print("1. SPEEDUP VARIABILITY")
    print("-" * 50)

    best_speedup = speedup_features.loc[speedup_features["r_EY"].abs().idxmax()]
    print(f"\n  Best metric: {best_speedup['Feature']}")
    print(f"    vs |EY|: r = {best_speedup['r_EY']:+.3f}")
    print(f"    vs |WS|: r = {best_speedup['r_WS']:+.3f}")

    if abs(best_speedup['r_EY']) > 0.15:
        print(f"\n  ✓ RECOMMEND adding to model")
    else:
        print(f"\n  ? WEAK effect — consider skipping")

    print("\n" + "-" * 50)
    print("2. dz × DISTANCE INTERACTION")
    print("-" * 50)

    best_dz_dist = dz_dist_features.loc[dz_dist_features["r_EY"].abs().idxmax()]
    print(f"\n  Best metric: {best_dz_dist['Feature']}")
    print(f"    vs |EY|: r = {best_dz_dist['r_EY']:+.3f}")
    print(f"    vs |WS|: r = {best_dz_dist['r_WS']:+.3f}")

    if abs(best_dz_dist['r_EY']) > 0.15:
        print(f"\n  ✓ RECOMMEND adding to model")
    else:
        print(f"\n  ? WEAK effect — consider skipping")

    print("\n" + "-" * 50)
    print("3. TURNING ANGLE VARIABILITY")
    print("-" * 50)

    best_turning = turning_features.loc[turning_features["r_EY"].abs().idxmax()]
    print(f"\n  Best metric: {best_turning['Feature']}")
    print(f"    vs |EY|: r = {best_turning['r_EY']:+.3f}")
    print(f"    vs |WS|: r = {best_turning['r_WS']:+.3f}")

    if abs(best_turning['r_EY']) > 0.15:
        print(f"\n  ✓ RECOMMEND adding to model")
    else:
        print(f"\n  ? WEAK effect — consider skipping")


def main():
    print("=" * 70)
    print("ADDITIONAL SIGMA DRIVER EXPLORATION")
    print("Testing: Speedup variability, dz×distance, Turning variability")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)

    # Compute features
    pair_df = compute_pair_features(df)

    # Correlation analysis
    corr_results = analyze_correlations(pair_df)

    # Categorical analysis
    analyze_by_category(pair_df)

    # Redundancy check
    check_redundancy(pair_df)

    # Visualizations
    create_visualizations(pair_df)

    # Recommendations
    print_recommendations(corr_results)

    return df, pair_df, corr_results


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_df, corr_results = result