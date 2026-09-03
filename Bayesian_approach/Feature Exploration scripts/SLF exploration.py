"""
Surface Land Fraction (SLF) Exploration

Features tested:
  1. dSLF          = SLF_WTG - SLF_MM  (signed difference)
  2. abs_dSLF      = |dSLF|            (magnitude of land/water mismatch)
  3. min_SLF       = min(SLF_WTG, SLF_MM)  (most marine site in pair)
  4. SLF_avg       = (SLF_WTG + SLF_MM) / 2  (average marine exposure)
  5. water_flag_WTG = SLF_WTG < 0.5    (binary: WTG sector mostly water)
  6. water_flag_MM  = SLF_MM < 0.5     (binary: MM sector mostly water)
  7. water_flag_any = min_SLF < 0.5    (binary: either site mostly water)
  8. coastal_transition = one site > 0.8 AND other < 0.3
  9. SLF_x_roughness_ch_avg  = min_SLF * roughness_ch_avg  (interaction)
 10. SLF_x_ref_length_diff   = min_SLF * abs_reference_length_diff (interaction)

SLF ranges from 0 to 1: 1 = 100% land, 0 = 100% water.
Water roughness is set to 0, so sectors with low SLF may cause prediction
issues due to degenerate roughness modelling.

Data note: SLF is a fractional value (not percentage).
           ~83/732 pairs have SLF_MM + SLF_WTG < 1.9
           ~11 pairs have |SLF_WTG - SLF_MM| > 0.1
           Most values are very close to 1.0 (predominantly land).
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


# ==============================================================
# DATA LOADING & FEATURE ENGINEERING
# ==============================================================

def load_and_prepare_data(path):
    """Load data, verify SLF columns exist, compute derived features."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("SURFACE LAND FRACTION (SLF) EXPLORATION")
    print("=" * 70)

    # Check SLF columns exist
    for col in ["SLF_WTG", "SLF_MM"]:
        if col not in df.columns:
            slf_cols = [c for c in df.columns if 'SLF' in c.upper()]
            print(f"  '{col}' not found!")
            print(f"  Available SLF columns: {slf_cols}")
            return None

    required = [
        "pair_id", "sector_name", "EY_deviation_sector_frac",
        "weight_energy", "weight_energy_predicted",
        "SLF_WTG", "SLF_MM"
    ]

    df = df.dropna(subset=required).copy()

    # --- Derived SLF features ---
    df["dSLF"] = df["SLF_WTG"] - df["SLF_MM"]
    df["abs_dSLF"] = np.abs(df["dSLF"])
    df["min_SLF"] = np.minimum(df["SLF_WTG"], df["SLF_MM"])
    df["SLF_avg"] = (df["SLF_WTG"] + df["SLF_MM"]) / 2.0

    # Binary flags
    df["water_flag_WTG"] = (df["SLF_WTG"] < 0.5).astype(int)
    df["water_flag_MM"] = (df["SLF_MM"] < 0.5).astype(int)
    df["water_flag_any"] = (df["min_SLF"] < 0.5).astype(int)
    df["coastal_transition"] = (
        ((df["SLF_WTG"] > 0.8) & (df["SLF_MM"] < 0.3)) |
        ((df["SLF_MM"] > 0.8) & (df["SLF_WTG"] < 0.3))
    ).astype(int)

    # Interaction features (only if roughness columns exist)
    if "roughness_ch_avg" in df.columns:
        df["SLF_x_roughness_ch"] = df["min_SLF"] * df["roughness_ch_avg"]
    if "abs_reference_length_diff" in df.columns:
        df["SLF_x_ref_length_diff"] = df["min_SLF"] * df["abs_reference_length_diff"]

    print(f"\nValid observations: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique()}")

    return df


# ==============================================================
# BASIC STATISTICS
# ==============================================================

def analyze_basic_statistics(df):
    """Distribution summary of raw SLF and derived features."""

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    for col, label in [("SLF_WTG", "SLF at WTG"), ("SLF_MM", "SLF at MM")]:
        vals = df[col]
        print(f"\n  {label}:")
        print(f"    Range: [{vals.min():.4f}, {vals.max():.4f}]")
        print(f"    Mean:  {vals.mean():.4f}")
        print(f"    Median: {vals.median():.4f}")
        for p in [5, 25, 50, 75, 95]:
            print(f"    {p}th pctl: {np.percentile(vals, p):.4f}")

    # Key distribution facts
    slf_sum = df["SLF_WTG"] + df["SLF_MM"]
    n_total = len(df)

    print(f"\n  Distribution summary (sector-level observations):")
    print(f"    Total observations: {n_total}")
    print(f"    SLF_WTG + SLF_MM < 1.9 (some water): {(slf_sum < 1.9).sum()} ({(slf_sum < 1.9).mean()*100:.1f}%)")
    print(f"    SLF_WTG + SLF_MM < 1.5 (substantial water): {(slf_sum < 1.5).sum()} ({(slf_sum < 1.5).mean()*100:.1f}%)")
    print(f"    min(SLF) < 0.5 (at least one site mostly water): {(df['min_SLF'] < 0.5).sum()} ({(df['min_SLF'] < 0.5).mean()*100:.1f}%)")
    print(f"    |dSLF| > 0.1 (meaningful mismatch): {(df['abs_dSLF'] > 0.1).sum()} ({(df['abs_dSLF'] > 0.1).mean()*100:.1f}%)")
    print(f"    Coastal transition flag: {df['coastal_transition'].sum()} ({df['coastal_transition'].mean()*100:.1f}%)")

    # Unique pairs with water exposure
    water_pairs = df.loc[df["min_SLF"] < 0.5, "pair_id"].nunique()
    total_pairs = df["pair_id"].nunique()
    print(f"\n    Unique pairs with min_SLF < 0.5: {water_pairs}/{total_pairs}")


# ==============================================================
# SECTOR-LEVEL ANALYSIS
# ==============================================================

def analyze_sector_level(df):
    """Test all SLF features at sector level against EY deviation."""

    print("\n" + "=" * 70)
    print("SECTOR-LEVEL ANALYSIS")
    print("=" * 70)

    y = df["EY_deviation_sector_frac"].values
    abs_y = np.abs(y)

    # List of features to test
    continuous_features = [
        ("dSLF", "dSLF (signed)"),
        ("abs_dSLF", "|dSLF| (magnitude)"),
        ("min_SLF", "min(SLF_WTG, SLF_MM)"),
        ("SLF_avg", "SLF average"),
        ("SLF_WTG", "SLF_WTG (raw)"),
        ("SLF_MM", "SLF_MM (raw)"),
    ]

    # Add interaction features if available
    if "SLF_x_roughness_ch" in df.columns:
        continuous_features.append(("SLF_x_roughness_ch", "min_SLF x roughness_ch"))
    if "SLF_x_ref_length_diff" in df.columns:
        continuous_features.append(("SLF_x_ref_length_diff", "min_SLF x ref_length_diff"))

    binary_features = [
        ("water_flag_WTG", "Water flag WTG (SLF<0.5)"),
        ("water_flag_MM", "Water flag MM (SLF<0.5)"),
        ("water_flag_any", "Water flag any (min SLF<0.5)"),
        ("coastal_transition", "Coastal transition"),
    ]

    results = {}

    # --- CONTINUOUS FEATURES ---
    print(f"\n{'CONTINUOUS FEATURES':^70}")
    print("-" * 70)
    print(f"  {'Feature':<35} {'vs signed EY':>15} {'vs |EY|':>15}")
    print(f"  {'':35} {'r (p)':>15} {'r (p)':>15}")
    print("  " + "-" * 65)

    for col, label in continuous_features:
        if col not in df.columns:
            continue
        x = df[col].values
        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 10:
            print(f"  {label:<35} {'n/a':>15} {'n/a':>15}")
            continue

        # vs signed EY (mean-model candidate)
        r_signed, p_signed = stats.pearsonr(x[valid], y[valid])
        sig_s = "***" if p_signed < 0.001 else "**" if p_signed < 0.01 else "*" if p_signed < 0.05 else ""

        # vs |EY| (sigma-model candidate)
        r_abs, p_abs = stats.pearsonr(x[valid], abs_y[valid])
        sig_a = "***" if p_abs < 0.001 else "**" if p_abs < 0.01 else "*" if p_abs < 0.05 else ""

        print(f"  {label:<35} {r_signed:+.4f}{sig_s:<4} ({p_signed:.3f}) {r_abs:+.4f}{sig_a:<4} ({p_abs:.3f})")

        results[col] = {
            "label": label,
            "r_signed": r_signed, "p_signed": p_signed,
            "r_abs": r_abs, "p_abs": p_abs,
            "type": "continuous",
        }

    # --- BINARY FEATURES ---
    print(f"\n{'BINARY FEATURES':^70}")
    print("-" * 70)
    print(f"  {'Feature':<35} {'n=1':>6} {'|EY| if 0':>12} {'|EY| if 1':>12} {'t-stat':>8} {'p':>8}")
    print("  " + "-" * 65)

    for col, label in binary_features:
        if col not in df.columns:
            continue

        flag = df[col].values
        n_flagged = flag.sum()

        if n_flagged < 3 or (len(flag) - n_flagged) < 3:
            print(f"  {label:<35} {n_flagged:>6} {'too few observations':>44}")
            results[col] = {
                "label": label, "n_flagged": n_flagged,
                "r_abs": np.nan, "p_abs": np.nan,
                "type": "binary", "skip": True,
            }
            continue

        group0 = abs_y[flag == 0]
        group1 = abs_y[flag == 1]

        t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {label:<35} {n_flagged:>6} {group0.mean():>12.4f} {group1.mean():>12.4f} {t_stat:>+8.2f} {p_val:>7.4f} {sig}")

        # Also compute point-biserial correlation (equivalent to Pearson for binary)
        r_pb, p_pb = stats.pearsonr(flag, abs_y)

        results[col] = {
            "label": label, "n_flagged": n_flagged,
            "mean_0": group0.mean(), "mean_1": group1.mean(),
            "t_stat": t_stat, "p_ttest": p_val,
            "r_abs": r_pb, "p_abs": p_pb,
            "type": "binary", "skip": False,
        }

    # --- STRATIFIED ANALYSIS for min_SLF ---
    print(f"\n{'STRATIFIED ANALYSIS: min_SLF':^70}")
    print("-" * 70)

    min_slf = df["min_SLF"].values

    # Use meaningful thresholds instead of terciles (since data is skewed toward 1)
    thresholds = [
        ("Water-dominated (< 0.5)", min_slf < 0.5),
        ("Mixed (0.5 - 0.9)", (min_slf >= 0.5) & (min_slf < 0.9)),
        ("Land-dominated (>= 0.9)", min_slf >= 0.9),
    ]

    for label, mask in thresholds:
        n = mask.sum()
        if n > 0:
            mean_ey = abs_y[mask].mean()
            std_ey = abs_y[mask].std()
            print(f"  {label:<35} n={n:<6} |EY| mean={mean_ey:.4f}  std={std_ey:.4f}")
        else:
            print(f"  {label:<35} n=0")

    # Levene's test between groups (only if all groups have data)
    groups = [abs_y[mask] for _, mask in thresholds if mask.sum() > 2]
    if len(groups) >= 2:
        stat_lev, p_lev = stats.levene(*groups)
        print(f"\n  Levene's test for variance equality: stat={stat_lev:.3f}, p={p_lev:.4f}")
        if p_lev < 0.05:
            print("  -> SIGNIFICANT: Error spread differs by water exposure level")
        else:
            print("  -> No significant difference in spread")

    return results


# ==============================================================
# PAIR-LEVEL ANALYSIS
# ==============================================================

def analyze_pair_level(df):
    """Aggregate to pair level (energy-weighted) and test."""

    print("\n" + "=" * 70)
    print("PAIR-LEVEL ANALYSIS (energy-weighted)")
    print("=" * 70)

    df = df.copy()

    # Energy-weighted EY deviation
    df["w_ey_dev"] = df["weight_energy"] * df["EY_deviation_sector_frac"]

    # Energy-weighted feature aggregation
    features_to_agg = {
        "dSLF": "dSLF",
        "abs_dSLF": "abs_dSLF",
        "min_SLF": "min_SLF",
        "SLF_avg": "SLF_avg",
    }
    if "SLF_x_roughness_ch" in df.columns:
        features_to_agg["SLF_x_roughness_ch"] = "SLF_x_roughness_ch"
    if "SLF_x_ref_length_diff" in df.columns:
        features_to_agg["SLF_x_ref_length_diff"] = "SLF_x_ref_length_diff"

    for feat_col in features_to_agg:
        df[f"w_{feat_col}"] = df["weight_energy_predicted"] * df[feat_col]

    # Build aggregation dict
    agg_dict = {
        "w_ey_dev": "sum",
        "location": "first",
    }
    for feat_col in features_to_agg:
        agg_dict[f"w_{feat_col}"] = "sum"

    # Also aggregate binary flags: pair has water if ANY sector has water
    for flag in ["water_flag_any", "coastal_transition"]:
        if flag in df.columns:
            agg_dict[flag] = "max"

    pair_agg = df.groupby("pair_id").agg(agg_dict).reset_index()

    # Rename weighted columns back
    for feat_col in features_to_agg:
        pair_agg = pair_agg.rename(columns={f"w_{feat_col}": feat_col})

    pair_agg = pair_agg.rename(columns={"w_ey_dev": "ey_deviation"})

    print(f"\nPairs: {len(pair_agg)}")

    y = pair_agg["ey_deviation"].values
    abs_y = np.abs(y)

    pair_results = {}

    # --- CONTINUOUS ---
    print(f"\n  {'Feature':<35} {'vs signed EY':>15} {'vs |EY|':>15}")
    print(f"  {'':35} {'r (p)':>15} {'r (p)':>15}")
    print("  " + "-" * 65)

    for feat_col, label in [
        ("dSLF", "dSLF (signed)"),
        ("abs_dSLF", "|dSLF| (magnitude)"),
        ("min_SLF", "min(SLF)"),
        ("SLF_avg", "SLF average"),
        ("SLF_x_roughness_ch", "min_SLF x roughness_ch"),
        ("SLF_x_ref_length_diff", "min_SLF x ref_length_diff"),
    ]:
        if feat_col not in pair_agg.columns:
            continue

        x = pair_agg[feat_col].values
        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 10:
            continue

        r_signed, p_signed = stats.pearsonr(x[valid], y[valid])
        r_abs, p_abs = stats.pearsonr(x[valid], abs_y[valid])

        sig_s = "***" if p_signed < 0.001 else "**" if p_signed < 0.01 else "*" if p_signed < 0.05 else ""
        sig_a = "***" if p_abs < 0.001 else "**" if p_abs < 0.01 else "*" if p_abs < 0.05 else ""

        print(f"  {label:<35} {r_signed:+.4f}{sig_s:<4} ({p_signed:.3f}) {r_abs:+.4f}{sig_a:<4} ({p_abs:.3f})")

        pair_results[feat_col] = {
            "r_signed": r_signed, "p_signed": p_signed,
            "r_abs": r_abs, "p_abs": p_abs,
        }

    # --- BINARY (pair-level) ---
    print(f"\n  {'Binary feature':<35} {'n=1':>6} {'|EY| if 0':>12} {'|EY| if 1':>12} {'t-stat':>8} {'p':>8}")
    print("  " + "-" * 65)

    for flag_col, label in [
        ("water_flag_any", "Any sector with water"),
        ("coastal_transition", "Coastal transition"),
    ]:
        if flag_col not in pair_agg.columns:
            continue

        flag = pair_agg[flag_col].values
        n1 = int(flag.sum())

        if n1 < 3 or (len(flag) - n1) < 3:
            print(f"  {label:<35} {n1:>6} {'too few observations':>44}")
            continue

        g0 = abs_y[flag == 0]
        g1 = abs_y[flag == 1]

        t, p_val = stats.ttest_ind(g0, g1, equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {label:<35} {n1:>6} {g0.mean():>12.4f} {g1.mean():>12.4f} {t:>+8.2f} {p_val:>7.4f} {sig}")

        pair_results[flag_col] = {"r_abs": np.nan, "p_abs": p_val, "t_stat": t}

    return pair_agg, pair_results


# ==============================================================
# REDUNDANCY CHECK
# ==============================================================

def check_redundancy(df):
    """Check if SLF features are collinear with existing model features."""

    print("\n" + "=" * 70)
    print("REDUNDANCY CHECK: Correlation with Existing Features")
    print("=" * 70)

    slf_features = ["min_SLF", "abs_dSLF", "SLF_avg"]
    existing = [
        ("RIX_avg_0.3_sector", "RIX_avg (orography)"),
        ("distance_m", "Distance"),
        ("dz", "dz (height diff)"),
        ("overall_speedup_WTG_factor", "Speedup WTG"),
        ("roughness_ch_avg", "Roughness changes"),
        ("reference_length_WTG", "Ref. length WTG"),
        ("reference_length_MM", "Ref. length MM"),
        ("abs_reference_length_diff", "Ref. length diff"),
        ("Sample_count_pred", "Sample count"),
        ("TI_MM_clean", "TI (clean)"),
    ]

    for slf_col in slf_features:
        if slf_col not in df.columns:
            continue

        print(f"\n  {slf_col} vs:")
        print(f"  {'Existing feature':<30} {'r':>8} {'flag':>12}")
        print("  " + "-" * 50)

        slf_vals = df[slf_col].values

        for col, label in existing:
            if col not in df.columns:
                continue

            x = df[col].values
            valid = np.isfinite(x) & np.isfinite(slf_vals)

            if valid.sum() < 10:
                continue

            r, p = stats.pearsonr(slf_vals[valid], x[valid])
            flag = " HIGH" if abs(r) > 0.5 else " moderate" if abs(r) > 0.3 else ""
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label:<30} {r:+.3f} {sig:<4}{flag}")


# ==============================================================
# VISUALIZATIONS
# ==============================================================

def create_visualizations(df, pair_agg):
    """Create diagnostic plots for SLF features."""

    abs_y_sector = np.abs(df["EY_deviation_sector_frac"].values)
    abs_y_pair = np.abs(pair_agg["ey_deviation"].values)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Surface Land Fraction (SLF) Feature Exploration", fontsize=14, fontweight="bold")

    # ---- Row 1: Distributions ----

    # 1a. Distribution of SLF_WTG and SLF_MM
    ax = axes[0, 0]
    ax.hist(df["SLF_WTG"], bins=30, alpha=0.6, label="SLF_WTG", edgecolor="black")
    ax.hist(df["SLF_MM"], bins=30, alpha=0.6, label="SLF_MM", edgecolor="black")
    ax.set_xlabel("SLF")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of SLF")
    ax.legend()

    # 1b. Distribution of min_SLF
    ax = axes[0, 1]
    ax.hist(df["min_SLF"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", label="Water threshold (0.5)")
    ax.set_xlabel("min(SLF_WTG, SLF_MM)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of min_SLF")
    ax.legend()

    # 1c. Distribution of abs_dSLF
    ax = axes[0, 2]
    ax.hist(df["abs_dSLF"], bins=30, edgecolor="black", alpha=0.7, color="darkorange")
    ax.axvline(0.1, color="red", linestyle="--", label="Threshold (0.1)")
    ax.set_xlabel("|SLF_WTG - SLF_MM|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of |dSLF|")
    ax.legend()

    # ---- Row 2: Sector-level scatter ----

    # 2a. min_SLF vs |EY|
    ax = axes[1, 0]
    ax.scatter(df["min_SLF"], abs_y_sector, alpha=0.2, s=8)
    valid = np.isfinite(df["min_SLF"].values) & np.isfinite(abs_y_sector)
    if valid.sum() > 2:
        x_v = df["min_SLF"].values[valid]
        y_v = abs_y_sector[valid]
        slope, intercept = np.polyfit(x_v, y_v, 1)
        x_line = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        r, _ = stats.pearsonr(x_v, y_v)
        ax.set_title(f"Sector: min_SLF vs |EY|\nr = {r:.4f}")
    ax.set_xlabel("min(SLF_WTG, SLF_MM)")
    ax.set_ylabel("|EY deviation|")

    # 2b. abs_dSLF vs |EY|
    ax = axes[1, 1]
    ax.scatter(df["abs_dSLF"], abs_y_sector, alpha=0.2, s=8)
    valid = np.isfinite(df["abs_dSLF"].values) & np.isfinite(abs_y_sector)
    if valid.sum() > 2:
        x_v = df["abs_dSLF"].values[valid]
        y_v = abs_y_sector[valid]
        slope, intercept = np.polyfit(x_v, y_v, 1)
        x_line = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        r, _ = stats.pearsonr(x_v, y_v)
        ax.set_title(f"Sector: |dSLF| vs |EY|\nr = {r:.4f}")
    ax.set_xlabel("|SLF_WTG - SLF_MM|")
    ax.set_ylabel("|EY deviation|")

    # 2c. SLF_avg vs |EY|
    ax = axes[1, 2]
    ax.scatter(df["SLF_avg"], abs_y_sector, alpha=0.2, s=8)
    valid = np.isfinite(df["SLF_avg"].values) & np.isfinite(abs_y_sector)
    if valid.sum() > 2:
        x_v = df["SLF_avg"].values[valid]
        y_v = abs_y_sector[valid]
        slope, intercept = np.polyfit(x_v, y_v, 1)
        x_line = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
        r, _ = stats.pearsonr(x_v, y_v)
        ax.set_title(f"Sector: SLF_avg vs |EY|\nr = {r:.4f}")
    ax.set_xlabel("(SLF_WTG + SLF_MM) / 2")
    ax.set_ylabel("|EY deviation|")

    # ---- Row 3: Pair-level + box plots ----

    # 3a. Pair-level: min_SLF vs |EY|
    ax = axes[2, 0]
    if "min_SLF" in pair_agg.columns:
        ax.scatter(pair_agg["min_SLF"], abs_y_pair, alpha=0.5, s=20)
        valid = np.isfinite(pair_agg["min_SLF"].values) & np.isfinite(abs_y_pair)
        if valid.sum() > 2:
            x_v = pair_agg["min_SLF"].values[valid]
            y_v = abs_y_pair[valid]
            slope, intercept = np.polyfit(x_v, y_v, 1)
            x_line = np.linspace(x_v.min(), x_v.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
            r, _ = stats.pearsonr(x_v, y_v)
            ax.set_title(f"Pair: min_SLF vs |EY|\nr = {r:.4f}")
    ax.set_xlabel("min_SLF (energy-weighted)")
    ax.set_ylabel("|EY deviation|")

    # 3b. Box plot: water_flag_any vs |EY| at sector level
    ax = axes[2, 1]
    if "water_flag_any" in df.columns:
        groups = [abs_y_sector[df["water_flag_any"] == 0], abs_y_sector[df["water_flag_any"] == 1]]
        labels_bp = [f"Land\n(n={len(groups[0])})", f"Water\n(n={len(groups[1])})"]

        bp = ax.boxplot(groups, labels=labels_bp, patch_artist=True, widths=0.5)
        colors = ["#4CAF50", "#2196F3"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("|EY deviation|")
        ax.set_title("Sector: Water exposure vs |EY|")
    else:
        ax.text(0.5, 0.5, "water_flag_any not available", ha='center', va='center')

    # 3c. Redundancy: min_SLF vs roughness_ch_avg
    ax = axes[2, 2]
    if "roughness_ch_avg" in df.columns:
        ax.scatter(df["min_SLF"], df["roughness_ch_avg"], alpha=0.2, s=8)
        valid = df["roughness_ch_avg"].notna() & df["min_SLF"].notna()
        if valid.sum() > 2:
            r, _ = stats.pearsonr(df.loc[valid, "min_SLF"], df.loc[valid, "roughness_ch_avg"])
            ax.set_title(f"min_SLF vs roughness_ch (redundancy)\nr = {r:.4f}")
        ax.set_xlabel("min_SLF")
        ax.set_ylabel("roughness_ch_avg")
    else:
        ax.text(0.5, 0.5, "roughness_ch_avg\nnot found", ha='center', va='center')
        ax.set_title("Redundancy check")

    plt.tight_layout()
    plt.savefig("SLF_exploration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: SLF_exploration.png")


# ==============================================================
# RECOMMENDATIONS
# ==============================================================

def print_recommendations(sector_results, pair_results):
    """Print final feature-by-feature recommendation."""

    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
DATA CAVEAT: Most SLF values are close to 1.0 (predominantly land).
Only a small fraction of sector-observations have meaningful water exposure.
Features based on SLF will have limited variance and may lack statistical
power even if the physical effect is real.
""")

    all_features = [
        ("min_SLF", "min(SLF) - most marine exposure in pair"),
        ("abs_dSLF", "|dSLF| - land/water mismatch magnitude"),
        ("dSLF", "dSLF (signed) - directional bias from water"),
        ("SLF_avg", "SLF average - overall marine exposure"),
        ("water_flag_any", "Water flag (binary, min_SLF < 0.5)"),
        ("coastal_transition", "Coastal transition (binary)"),
        ("SLF_x_roughness_ch", "min_SLF x roughness_ch (interaction)"),
        ("SLF_x_ref_length_diff", "min_SLF x ref_length_diff (interaction)"),
    ]

    print(f"  {'Feature':<45} {'Verdict':<12} {'Evidence'}")
    print("  " + "-" * 80)

    for col, label in all_features:

        # Check sector-level results
        s_res = sector_results.get(col, {})
        p_res = pair_results.get(col, {})

        # Get best p-value from either level
        p_sector = s_res.get("p_abs", 1.0)
        p_pair = p_res.get("p_abs", 1.0)
        r_sector = s_res.get("r_abs", 0.0)
        r_pair = p_res.get("r_abs", 0.0)

        # Handle NaN
        if pd.isna(p_sector):
            p_sector = 1.0
        if pd.isna(p_pair):
            p_pair = 1.0
        if pd.isna(r_sector):
            r_sector = 0.0
        if pd.isna(r_pair):
            r_pair = 0.0

        # Binary features use t-test p-value
        if s_res.get("type") == "binary":
            p_sector = s_res.get("p_abs", 1.0) if not s_res.get("skip", False) else 1.0

        # Decision
        best_p = min(p_sector, p_pair)
        best_r = max(abs(r_sector), abs(r_pair))

        if best_p < 0.05 and best_r > 0.1:
            verdict = "ADD"
            evidence = f"r={best_r:+.3f}, p={best_p:.4f}"
        elif best_p < 0.1 and best_r > 0.05:
            verdict = "CONSIDER"
            evidence = f"r={best_r:+.3f}, p={best_p:.4f}"
        elif s_res.get("skip", False):
            verdict = "SKIP"
            evidence = "too few observations"
        else:
            verdict = "SKIP"
            evidence = f"r={best_r:+.3f}, p={best_p:.4f}"

        print(f"  {label:<45} {verdict:<12} {evidence}")

    print(f"""
NOTE: Given the extreme skew (most SLF ~ 1.0), consider:
  - Even non-significant features may matter physically for coastal sites
  - A binary water_flag may capture the effect better than continuous SLF
  - Interaction terms (SLF x roughness) test whether existing features
    break down for marine sectors (roughness = 0 problem)
  - If adding to the Bayesian model, consider using a binary indicator
    rather than continuous SLF, since the continuous version has almost
    no variance for the majority of observations.
""")


# ==============================================================
# TERRAIN-STRATIFIED ANALYSIS (v2)
# ==============================================================

TERRAIN_BINS = {
    "Flat": (0, 40),
    "Semi-complex": (40, 50),
    "Complex": (50, 999),
}
TERRAIN_COLORS = {
    "Flat": "#2ecc71",
    "Semi-complex": "#f39c12",
    "Complex": "#e74c3c",
}


def terrain_stratified_analysis(df):
    """
    Key addition: test water_exposure = 2 - (SLF_WTG + SLF_MM) stratified
    by terrain, following the same approach that unlocked roughness mismatch.
    """
    print("\n" + "=" * 70)
    print("TERRAIN-STRATIFIED SLF ANALYSIS (v2)")
    print("=" * 70)

    df = df.copy()

    # --- Compute water_exposure at sector level ---
    df["water_exposure"] = 2.0 - (df["SLF_WTG"] + df["SLF_MM"])
    # Also: |dSLF| at sector level (already computed, but ensure it exists)
    if "abs_dSLF" not in df.columns:
        df["abs_dSLF"] = np.abs(df["SLF_WTG"] - df["SLF_MM"])

    # WS error (for energy-weighted aggregation)
    if "e" not in df.columns:
        df["e"] = (
            (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
            / df["Mean_windspeed_self"]
        )
    df["abs_error"] = np.abs(df["e"])

    # Pair-level RIX
    if "RIX_avg_0.0501_overall_pair" not in df.columns:
        df["RIX_avg_0.0501_overall_pair"] = (
            df.groupby("pair_id")["RIX_avg_0.0501_sector"].transform("mean")
        )

    def classify_terrain(rix):
        for name, (lo, hi) in TERRAIN_BINS.items():
            if lo <= rix < hi:
                return name
        return "Complex"

    df["terrain_cat"] = df["RIX_avg_0.0501_overall_pair"].apply(classify_terrain)

    # --- Energy-weighted pair-level aggregation ---
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "water_exposure": (
                (g["water_exposure"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "abs_dSLF": (
                (g["abs_dSLF"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "min_SLF": (
                (g[["SLF_WTG", "SLF_MM"]].min(axis=1) * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "RIX_pair": g["RIX_avg_0.0501_overall_pair"].iloc[0],
            "terrain_cat": g["terrain_cat"].iloc[0],
            "location": g["location"].iloc[0],
            "distance_m": g["distance_m"].iloc[0],
            "log_dist_norm": np.log(g["distance_m"].iloc[0] / max(g["distance_A"].iloc[0], 1)),
        }),
        include_groups=False,
    ).reset_index()

    for col in ["actual_abs_error", "water_exposure", "abs_dSLF", "min_SLF",
                 "RIX_pair", "distance_m", "log_dist_norm"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    pair_agg = pair_agg.dropna(subset=["water_exposure"])

    print(f"\n  Loaded {len(pair_agg)} pairs")
    print(f"  water_exposure range: [{pair_agg['water_exposure'].min():.4f}, {pair_agg['water_exposure'].max():.4f}]")
    print(f"  Pairs with water_exposure > 0.02: {(pair_agg['water_exposure'] > 0.02).sum()}")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        if n > 0:
            n_water = ((pair_agg["terrain_cat"] == cat) & (pair_agg["water_exposure"] > 0.02)).sum()
            print(f"    {cat}: {n} pairs ({n_water} with water)")

    # --- Features to test ---
    features = [
        ("water_exposure", "Water exposure (2 - SLF_sum)"),
        ("abs_dSLF", "|dSLF| (water mismatch)"),
        ("min_SLF", "min(SLF) (most exposed site)"),
    ]

    # --- Overall correlations ---
    print(f"\n  --- Overall correlations with |error| ---")
    for feat_col, feat_name in features:
        r, p = stats.pearsonr(pair_agg[feat_col], pair_agg["actual_abs_error"])
        print(f"    {feat_name:35s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # --- Stratified by terrain ---
    print(f"\n  --- Stratified by terrain ---")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 4:
            continue
        print(f"\n  {cat} (n={len(subset)}):")
        for feat_col, feat_name in features:
            # Check variance
            if subset[feat_col].std() < 1e-8:
                print(f"    {feat_name:35s}: no variance")
                continue
            r, p = stats.pearsonr(subset[feat_col], subset["actual_abs_error"])
            print(f"    {feat_name:35s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # --- Partial correlations controlling for dist_norm + RIX ---
    print(f"\n  --- Partial correlations (controlling for log_dist_norm + RIX_pair) ---")
    try:
        from sklearn.linear_model import LinearRegression
        controls = pair_agg[["log_dist_norm", "RIX_pair"]].values

        reg_err = LinearRegression().fit(controls, pair_agg["actual_abs_error"])
        resid_error = pair_agg["actual_abs_error"] - reg_err.predict(controls)

        for feat_col, feat_name in features:
            reg_feat = LinearRegression().fit(controls, pair_agg[feat_col])
            resid_feat = pair_agg[feat_col] - reg_feat.predict(controls)
            r_partial, p_partial = stats.pearsonr(resid_feat, resid_error)
            print(f"    {feat_name:35s}: r={r_partial:+.3f}  p={p_partial:.3f}  {'*' if p_partial < 0.05 else ''}")
    except ImportError:
        print("    (sklearn not available)")

    # --- Partial controlling for roughness_mismatch too ---
    print(f"\n  --- Partial correlations (controlling for dist_norm + RIX + roughness_mismatch) ---")
    try:
        from sklearn.linear_model import LinearRegression

        # Compute roughness_mismatch at pair level if not already present
        if "roughness_mismatch" not in pair_agg.columns:
            if "reference_length_WTG" in df.columns and "reference_length_MM" in df.columns:
                df["_rough_mm"] = np.abs(
                    np.log(df["reference_length_WTG"].clip(lower=1e-6)
                           / df["reference_length_MM"].clip(lower=1e-6))
                )
                rough_pair = df.groupby("pair_id").apply(
                    lambda g: (g["_rough_mm"] * g["weight_energy_predicted"]).sum()
                    / g["weight_energy_predicted"].sum(),
                    include_groups=False,
                )
                pair_agg["roughness_mismatch"] = pair_agg["pair_id"].map(rough_pair)

        if "roughness_mismatch" in pair_agg.columns:
            controls_ext = pair_agg[["log_dist_norm", "RIX_pair", "roughness_mismatch"]].values

            reg_err2 = LinearRegression().fit(controls_ext, pair_agg["actual_abs_error"])
            resid_error2 = pair_agg["actual_abs_error"] - reg_err2.predict(controls_ext)

            for feat_col, feat_name in features:
                reg_feat2 = LinearRegression().fit(controls_ext, pair_agg[feat_col])
                resid_feat2 = pair_agg[feat_col] - reg_feat2.predict(controls_ext)
                r_partial2, p_partial2 = stats.pearsonr(resid_feat2, resid_error2)
                print(f"    {feat_name:35s}: r={r_partial2:+.3f}  p={p_partial2:.3f}  {'*' if p_partial2 < 0.05 else ''}")
        else:
            print("    (roughness_mismatch not available)")
    except ImportError:
        print("    (sklearn not available)")

    # --- Correlation with roughness_mismatch (redundancy) ---
    if "roughness_mismatch" in pair_agg.columns:
        print(f"\n  --- Redundancy with roughness_mismatch ---")
        for feat_col, feat_name in features:
            r, p = stats.pearsonr(pair_agg[feat_col], pair_agg["roughness_mismatch"])
            print(f"    {feat_name:35s} vs roughness_mm: r={r:+.3f}  {'HIGH' if abs(r) > 0.5 else ''}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: water_exposure vs error by terrain
    ax = axes[0, 0]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) < 3:
            continue
        ax.scatter(subset["water_exposure"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if len(subset) > 3 and subset["water_exposure"].std() > 1e-8:
            r_cat, _ = stats.pearsonr(subset["water_exposure"], subset["actual_abs_error"])
            sl, ic, _, _, _ = stats.linregress(subset["water_exposure"], subset["actual_abs_error"])
            x_r = np.linspace(subset["water_exposure"].min(), subset["water_exposure"].max(), 50)
            ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
            ax.text(x_r[-1], sl * x_r[-1] + ic, f" r={r_cat:.2f}",
                    color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")
    r_all, p_all = stats.pearsonr(pair_agg["water_exposure"], pair_agg["actual_abs_error"])
    ax.set_xlabel("Water Exposure (2 - SLF_sum)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title(f"Water Exposure vs |Error| by Terrain (overall r={r_all:.2f}, p={p_all:.3f})", fontsize=11, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: colored by RIX
    ax = axes[0, 1]
    sc = ax.scatter(pair_agg["water_exposure"], pair_agg["actual_abs_error"],
                    c=pair_agg["RIX_pair"], cmap="RdYlGn_r", alpha=0.7, s=60,
                    edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="RIX_avg_0.0501_pair")
    ax.set_xlabel("Water Exposure (2 - SLF_sum)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title(f"Water Exposure colored by RIX", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 3: |dSLF| vs error by terrain
    ax = axes[1, 0]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) < 3:
            continue
        ax.scatter(subset["abs_dSLF"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if len(subset) > 3 and subset["abs_dSLF"].std() > 1e-8:
            r_cat, _ = stats.pearsonr(subset["abs_dSLF"], subset["actual_abs_error"])
            sl, ic, _, _, _ = stats.linregress(subset["abs_dSLF"], subset["actual_abs_error"])
            x_r = np.linspace(subset["abs_dSLF"].min(), subset["abs_dSLF"].max(), 50)
            ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
            ax.text(x_r[-1], sl * x_r[-1] + ic, f" r={r_cat:.2f}",
                    color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")
    ax.set_xlabel("|dSLF| (Water Mismatch)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title("|dSLF| vs |Error| by Terrain", fontsize=11, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 4: water_exposure vs roughness_mismatch (redundancy check)
    ax = axes[1, 1]
    if "roughness_mismatch" in pair_agg.columns:
        sc = ax.scatter(pair_agg["water_exposure"], pair_agg["roughness_mismatch"],
                        c=pair_agg["actual_abs_error"], cmap="YlOrRd", alpha=0.7, s=60,
                        edgecolors="white", linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="Actual |Error|")
        r_red, _ = stats.pearsonr(pair_agg["water_exposure"], pair_agg["roughness_mismatch"])
        ax.set_xlabel("Water Exposure", fontsize=12)
        ax.set_ylabel("Roughness Mismatch", fontsize=12)
        ax.set_title(f"Water Exposure vs Roughness Mismatch (r={r_red:.2f})\ncolor = |Error|",
                     fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "roughness_mismatch not available", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    fig.suptitle("SLF v2: Terrain-Stratified Water Exposure Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("SLF_terrain_stratified.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: SLF_terrain_stratified.png")

    return pair_agg


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 70)
    print("SURFACE LAND FRACTION (SLF) EXPLORATION")
    print("Testing: Does water exposure drive prediction uncertainty?")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(sector_model_path)
    if df is None:
        return

    # Basic statistics
    analyze_basic_statistics(df)

    # Sector-level analysis
    sector_results = analyze_sector_level(df)

    # Pair-level analysis
    pair_agg, pair_results = analyze_pair_level(df)

    # Redundancy check
    check_redundancy(df)

    # Visualizations
    create_visualizations(df, pair_agg)

    # Recommendations (original)
    print_recommendations(sector_results, pair_results)

    # NEW: Terrain-stratified analysis
    pair_agg_v2 = terrain_stratified_analysis(df)

    return df, pair_agg, pair_agg_v2


if __name__ == "__main__":
    result = main()
    if result:
        df, pair_agg, pair_agg_v2 = result
