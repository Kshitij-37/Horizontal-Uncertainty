"""
dRIX Mismatch Exploration

Hypothesis: Prediction error is driven by terrain ASYMMETRY between MM and WTG,
            not absolute terrain complexity. When both sites are equally complex,
            errors cancel. dRIX captures this mismatch.

Problem: dRIX_0.3 (slope > 30%) is near-zero for most pairs — no variance.
         dRIX_0.0501 (slope > 5%) should have more variance.

Variants tested:
  1. |dRIX_0.0501|                           — total terrain mismatch
  2. |dRIX_0.3|                              — severe-only mismatch (current model)
  3. |dRIX_0.0501| × severity_fraction       — mismatch weighted by severity
  4. severity_fraction × |dRIX_0.0501|       — same as 3 (commutative), just to be explicit

Also checks: non-linearity (LOWESS), terrain stratification, partial correlations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# CONFIG
# =============================================================================
SECTOR_MODEL_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

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

EPS = 1e-6


def load_data():
    """Load data, compute dRIX variants, aggregate to pair level."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    # Check columns
    required = ["dRIX_0.0501_sector", "dRIX_0.3_sector"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found")
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{len(df)} non-null")

    # ---- Sector-level features ----

    # Absolute dRIX values
    df["abs_dRIX_0.0501"] = np.abs(df["dRIX_0.0501_sector"])
    df["abs_dRIX_0.3"] = np.abs(df["dRIX_0.3_sector"])

    # Severity fraction
    df["dRIX_mild"] = df["dRIX_0.0501_sector"] - df["dRIX_0.3_sector"]
    df["severity_fraction"] = (
        np.abs(df["dRIX_0.3_sector"])
        / (np.abs(df["dRIX_0.3_sector"]) + np.abs(df["dRIX_mild"]) + EPS)
    )

    # Variant 3: dRIX_0.0501 × severity_fraction
    df["dRIX_severity"] = df["abs_dRIX_0.0501"] * df["severity_fraction"]

    # WS error
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

    # Print sector-level ranges
    print(f"\n  Sector-level ranges:")
    print(f"    |dRIX_0.0501|:     [{df['abs_dRIX_0.0501'].min():.3f}, {df['abs_dRIX_0.0501'].max():.3f}]  "
          f"mean={df['abs_dRIX_0.0501'].mean():.3f}  non-zero: {(df['abs_dRIX_0.0501'] > 0.01).sum()}/{len(df)}")
    print(f"    |dRIX_0.3|:        [{df['abs_dRIX_0.3'].min():.3f}, {df['abs_dRIX_0.3'].max():.3f}]  "
          f"mean={df['abs_dRIX_0.3'].mean():.3f}  non-zero: {(df['abs_dRIX_0.3'] > 0.01).sum()}/{len(df)}")
    print(f"    severity_fraction: [{df['severity_fraction'].min():.3f}, {df['severity_fraction'].max():.3f}]")
    print(f"    dRIX_severity:     [{df['dRIX_severity'].min():.3f}, {df['dRIX_severity'].max():.3f}]")

    # ---- Pair-level aggregation (energy-weighted) ----
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "abs_dRIX_0.0501": (
                (g["abs_dRIX_0.0501"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "abs_dRIX_0.3": (
                (g["abs_dRIX_0.3"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "dRIX_severity": (
                (g["dRIX_severity"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "severity_fraction": (
                (g["severity_fraction"] * g["weight_energy_predicted"]).sum()
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

    for col in ["actual_abs_error", "abs_dRIX_0.0501", "abs_dRIX_0.3",
                 "dRIX_severity", "severity_fraction", "RIX_pair",
                 "distance_m", "log_dist_norm"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    pair_agg = pair_agg.dropna(subset=["abs_dRIX_0.0501"])

    print(f"\n  Loaded {len(pair_agg)} pairs")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        if n > 0:
            print(f"    {cat}: {n} pairs")

    return df, pair_agg


def print_summary(pair_agg):
    """Correlation summary for all variants."""

    features = [
        ("abs_dRIX_0.0501", "|dRIX_0.0501| (total mismatch)"),
        ("abs_dRIX_0.3", "|dRIX_0.3| (severe mismatch, current)"),
        ("dRIX_severity", "|dRIX_0.0501| × severity_fraction"),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY: dRIX Mismatch Variants")
    print("=" * 70)

    # Overall
    print(f"\n  --- Overall correlations with |error| ---")
    for feat_col, feat_name in features:
        if pair_agg[feat_col].std() < EPS:
            print(f"    {feat_name:45s}: no variance")
            continue
        r, p = stats.pearsonr(pair_agg[feat_col], pair_agg["actual_abs_error"])
        print(f"    {feat_name:45s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # Variance comparison
    print(f"\n  --- Feature variance (pair-level) ---")
    for feat_col, feat_name in features:
        vals = pair_agg[feat_col]
        print(f"    {feat_name:45s}: std={vals.std():.4f}  IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]")

    # Stratified by terrain
    print(f"\n  --- Stratified by terrain ---")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 4:
            continue
        print(f"\n  {cat} (n={len(subset)}):")
        for feat_col, feat_name in features:
            if subset[feat_col].std() < EPS:
                print(f"    {feat_name:45s}: no variance")
                continue
            r, p = stats.pearsonr(subset[feat_col], subset["actual_abs_error"])
            print(f"    {feat_name:45s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # Partial correlations (controlling for log_dist_norm + RIX_pair)
    print(f"\n  --- Partial correlations (controlling for log_dist_norm + RIX_pair) ---")
    try:
        from sklearn.linear_model import LinearRegression
        controls = pair_agg[["log_dist_norm", "RIX_pair"]].values

        reg_err = LinearRegression().fit(controls, pair_agg["actual_abs_error"])
        resid_error = pair_agg["actual_abs_error"] - reg_err.predict(controls)

        for feat_col, feat_name in features:
            if pair_agg[feat_col].std() < EPS:
                print(f"    {feat_name:45s}: no variance")
                continue
            reg_feat = LinearRegression().fit(controls, pair_agg[feat_col])
            resid_feat = pair_agg[feat_col] - reg_feat.predict(controls)
            r_partial, p_partial = stats.pearsonr(resid_feat, resid_error)
            print(f"    {feat_name:45s}: r={r_partial:+.3f}  p={p_partial:.3f}  {'*' if p_partial < 0.05 else ''}")
    except ImportError:
        print("    (sklearn not available)")

    # Correlation between variants
    print(f"\n  --- Correlation between variants ---")
    for i, (c1, n1) in enumerate(features):
        for c2, n2 in features[i+1:]:
            if pair_agg[c1].std() > EPS and pair_agg[c2].std() > EPS:
                r, _ = stats.pearsonr(pair_agg[c1], pair_agg[c2])
                print(f"    {n1[:30]:30s} vs {n2[:30]:30s}: r={r:.3f}")

    # Comparison with current model's dRIX_0.3
    print(f"\n  --- Does dRIX_0.0501 improve over dRIX_0.3? ---")
    r_03, p_03 = stats.pearsonr(pair_agg["abs_dRIX_0.3"], pair_agg["actual_abs_error"])
    r_005, p_005 = stats.pearsonr(pair_agg["abs_dRIX_0.0501"], pair_agg["actual_abs_error"])
    print(f"    |dRIX_0.3|   : r={r_03:+.3f}  p={p_03:.3f}")
    print(f"    |dRIX_0.0501|: r={r_005:+.3f}  p={p_005:.3f}")
    if abs(r_005) > abs(r_03):
        print(f"    → dRIX_0.0501 is BETTER ({abs(r_005):.3f} > {abs(r_03):.3f})")
    else:
        print(f"    → dRIX_0.3 is better or equal")


def plot_exploration(pair_agg):
    """Create exploration plots."""

    features = [
        ("abs_dRIX_0.0501", "|dRIX_0.0501| (total mismatch)"),
        ("abs_dRIX_0.3", "|dRIX_0.3| (severe, current)"),
        ("dRIX_severity", "|dRIX_0.0501| × severity_frac"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- Row 1: Each variant vs error, by terrain ----
    for col_idx, (feat_col, feat_label) in enumerate(features):
        ax = axes[0, col_idx]
        for cat in ["Flat", "Semi-complex", "Complex"]:
            mask = pair_agg["terrain_cat"] == cat
            subset = pair_agg[mask]
            if len(subset) < 3:
                continue
            ax.scatter(subset[feat_col], subset["actual_abs_error"],
                       color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                       edgecolors="white", linewidths=0.5, zorder=3)
            if len(subset) > 3 and subset[feat_col].std() > EPS:
                r_cat, _ = stats.pearsonr(subset[feat_col], subset["actual_abs_error"])
                sl, ic, _, _, _ = stats.linregress(subset[feat_col], subset["actual_abs_error"])
                x_r = np.linspace(subset[feat_col].min(), subset[feat_col].max(), 50)
                ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat],
                        linestyle="--", alpha=0.8, linewidth=2)
                ax.text(x_r[-1], sl * x_r[-1] + ic, f" r={r_cat:.2f}",
                        color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")

        if pair_agg[feat_col].std() > EPS:
            r_all, p_all = stats.pearsonr(pair_agg[feat_col], pair_agg["actual_abs_error"])
            ax.set_title(f"{feat_label}\n(overall r={r_all:.2f}, p={p_all:.3f})",
                         fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{feat_label}\n(no variance)", fontsize=11, fontweight="bold")

        ax.set_xlabel(feat_label, fontsize=10)
        ax.set_ylabel("Actual |Error|", fontsize=10)
        ax.legend(title="Terrain", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # ---- Row 2: Non-linearity checks ----

    # Plot 4: LOWESS for dRIX_0.0501
    ax = axes[1, 0]
    x_vals = pair_agg["abs_dRIX_0.0501"].values
    y_vals = pair_agg["actual_abs_error"].values
    ax.scatter(x_vals, y_vals, alpha=0.7, s=60, edgecolors="white",
               linewidths=0.5, c="#2c3e50", zorder=3)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y_vals, x_vals, frac=0.5, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], "r-", linewidth=2.5, label="LOWESS")
    except ImportError:
        pass

    if x_vals.std() > EPS:
        sl, ic, r_lin, _, _ = stats.linregress(x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        ax.plot(x_line, sl * x_line + ic, "b--", linewidth=1.5, alpha=0.5,
                label=f"Linear (r={r_lin:.2f})")

    ax.set_xlabel("|dRIX_0.0501|", fontsize=11)
    ax.set_ylabel("Actual |Error|", fontsize=11)
    ax.set_title("Non-linearity check: |dRIX_0.0501|", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 5: dRIX_0.0501 vs dRIX_0.3 (how much info does 0.3 miss?)
    ax = axes[1, 1]
    sc = ax.scatter(pair_agg["abs_dRIX_0.0501"], pair_agg["abs_dRIX_0.3"],
                    c=pair_agg["actual_abs_error"], cmap="YlOrRd", alpha=0.7, s=60,
                    edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Actual |Error|")
    ax.plot([0, pair_agg["abs_dRIX_0.0501"].max()],
            [0, pair_agg["abs_dRIX_0.0501"].max()], "k--", alpha=0.3)
    ax.set_xlabel("|dRIX_0.0501| (all terrain)", fontsize=11)
    ax.set_ylabel("|dRIX_0.3| (severe only)", fontsize=11)
    ax.set_title("dRIX_0.0501 vs dRIX_0.3\n(color = |Error|)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 6: Histogram of each variant's values
    ax = axes[1, 2]
    for feat_col, feat_label, color in [
        ("abs_dRIX_0.0501", "|dRIX_0.0501|", "#3498db"),
        ("abs_dRIX_0.3", "|dRIX_0.3|", "#e74c3c"),
        ("dRIX_severity", "dRIX × severity", "#8e44ad"),
    ]:
        vals = pair_agg[feat_col].values
        ax.hist(vals, bins=20, alpha=0.5, label=feat_label, color=color, edgecolor="white")

    ax.set_xlabel("Feature value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of dRIX variants\n(more spread = more signal potential)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("dRIX Mismatch: Does terrain asymmetry drive prediction error?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("dRIX_mismatch_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: dRIX_mismatch_exploration.png")


if __name__ == "__main__":
    df, pair_agg = load_data()
    print_summary(pair_agg)
    plot_exploration(pair_agg)
