"""
Roughness Mismatch Exploration (v2 — sector-level + terrain-dampened)

Feature: abs_log_ref_length_ratio = |log(ref_length_WTG / ref_length_MM)|
         Computed at SECTOR level from reference_length_WTG and reference_length_MM.

Previous finding: roughness mismatch correlates with error in flat terrain (r=0.33)
                  but not complex terrain. This makes physical sense — in flat terrain,
                  roughness is the primary flow driver.

Two terrain-dampened variants tested:
  A) roughness_mismatch / (1 + RIX_avg_scaled)
     - RIX_avg_scaled = RIX_avg_0.0501_sector / max(RIX_avg_0.0501_sector)
     - Dampens smoothly as terrain gets more complex

  B) roughness_mismatch × (1 - severity_fraction)
     - severity_fraction = |dRIX_0.3| / (|dRIX_0.3| + |dRIX_mild| + ε)
     - Dampens specifically based on fraction of severe terrain

Both are computed at SECTOR level, then energy-weighted to pair level for plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
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
    """Load data, compute sector-level roughness features, aggregate to pair level."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    # Check required columns
    required = ["reference_length_WTG", "reference_length_MM",
                 "RIX_avg_0.0501_sector", "dRIX_0.3_sector", "dRIX_0.0501_sector"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found")
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{len(df)} non-null")

    # ---- Sector-level features ----

    # Raw roughness mismatch (sector-level)
    df["roughness_mismatch"] = np.abs(
        np.log(df["reference_length_WTG"].clip(lower=EPS)
               / df["reference_length_MM"].clip(lower=EPS))
    )

    # Variant A: dampened by RIX_avg_scaled
    rix_max = df["RIX_avg_0.0501_sector"].max()
    rix_max = max(rix_max, 1.0)  # avoid division by zero
    df["RIX_avg_scaled"] = df["RIX_avg_0.0501_sector"] / rix_max
    df["rough_dampened_rix"] = df["roughness_mismatch"] / (1 + df["RIX_avg_scaled"])

    # Variant B: dampened by severity_fraction
    df["dRIX_mild"] = df["dRIX_0.0501_sector"] - df["dRIX_0.3_sector"]
    df["severity_fraction"] = (
        np.abs(df["dRIX_0.3_sector"])
        / (np.abs(df["dRIX_0.3_sector"]) + np.abs(df["dRIX_mild"]) + EPS)
    )
    df["rough_dampened_severity"] = df["roughness_mismatch"] * (1 - df["severity_fraction"])

    # Sector-level error
    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # Pair-level RIX
    if "RIX_avg_0.0501_overall_pair" not in df.columns:
        df["RIX_avg_0.0501_overall_pair"] = df.groupby("pair_id")["RIX_avg_0.0501_sector"].transform("mean")

    def classify_terrain(rix):
        for name, (lo, hi) in TERRAIN_BINS.items():
            if lo <= rix < hi:
                return name
        return "Complex"

    df["terrain_cat"] = df["RIX_avg_0.0501_overall_pair"].apply(classify_terrain)

    # Print sector-level ranges
    print(f"\n  Sector-level feature ranges:")
    print(f"    roughness_mismatch:       [{df['roughness_mismatch'].min():.3f}, {df['roughness_mismatch'].max():.3f}]")
    print(f"    rough_dampened_rix:        [{df['rough_dampened_rix'].min():.3f}, {df['rough_dampened_rix'].max():.3f}]")
    print(f"    rough_dampened_severity:   [{df['rough_dampened_severity'].min():.3f}, {df['rough_dampened_severity'].max():.3f}]")
    print(f"    severity_fraction:         [{df['severity_fraction'].min():.3f}, {df['severity_fraction'].max():.3f}]")

    # ---- Pair-level aggregation (energy-weighted) ----
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            # Energy-weighted sector-level features
            "roughness_mismatch": (
                (g["roughness_mismatch"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "rough_dampened_rix": (
                (g["rough_dampened_rix"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "rough_dampened_severity": (
                (g["rough_dampened_severity"] * g["weight_energy_predicted"]).sum()
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

    for col in ["actual_abs_error", "roughness_mismatch", "rough_dampened_rix",
                 "rough_dampened_severity", "RIX_pair", "distance_m", "log_dist_norm"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    pair_agg = pair_agg.dropna(subset=["roughness_mismatch"])

    print(f"\n  Loaded {len(pair_agg)} pairs")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        if n > 0:
            print(f"    {cat}: {n} pairs")

    return df, pair_agg


def plot_exploration(pair_agg):
    """Create 3x2 figure comparing raw, RIX-dampened, and severity-dampened."""

    features = [
        ("roughness_mismatch", "Raw Roughness Mismatch\n|log(z₀_WTG / z₀_MM)|"),
        ("rough_dampened_rix", "Variant A: Dampened by RIX\nroughness / (1 + RIX_scaled)"),
        ("rough_dampened_severity", "Variant B: Dampened by Severity\nroughness × (1 - severity_fraction)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

    for row_idx, (feat_col, feat_label) in enumerate(features):

        # Left: scatter by terrain with trend lines per category
        ax = axes[row_idx, 0]
        for cat in ["Flat", "Semi-complex", "Complex"]:
            mask = pair_agg["terrain_cat"] == cat
            subset = pair_agg[mask]
            if len(subset) < 3:
                continue
            ax.scatter(subset[feat_col], subset["actual_abs_error"],
                       color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                       edgecolors="white", linewidths=0.5, zorder=3)
            r_cat, p_cat = stats.pearsonr(subset[feat_col], subset["actual_abs_error"])
            sl, ic, _, _, _ = stats.linregress(subset[feat_col], subset["actual_abs_error"])
            x_r = np.linspace(subset[feat_col].min(), subset[feat_col].max(), 50)
            ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat],
                    linestyle="--", alpha=0.8, linewidth=2)
            ax.text(x_r[-1], sl * x_r[-1] + ic,
                    f" r={r_cat:.2f} {'*' if p_cat < 0.05 else ''}",
                    color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")

        ax.set_xlabel(feat_label, fontsize=11)
        ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=11)
        r_all, p_all = stats.pearsonr(pair_agg[feat_col], pair_agg["actual_abs_error"])
        ax.set_title(f"By Terrain (overall r={r_all:.2f}, p={p_all:.3f})", fontsize=12, fontweight="bold")
        ax.legend(title="Terrain", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Right: overall scatter colored by RIX
        ax = axes[row_idx, 1]
        sc = ax.scatter(pair_agg[feat_col], pair_agg["actual_abs_error"],
                        c=pair_agg["RIX_pair"], cmap="RdYlGn_r", alpha=0.7, s=60,
                        edgecolors="white", linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="RIX_avg_0.0501_pair")

        # Overall trend
        sl, ic, _, _, _ = stats.linregress(pair_agg[feat_col], pair_agg["actual_abs_error"])
        x_r = np.linspace(pair_agg[feat_col].min(), pair_agg[feat_col].max(), 50)
        ax.plot(x_r, sl * x_r + ic, "k--", alpha=0.6, linewidth=2)
        ax.set_xlabel(feat_label, fontsize=11)
        ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=11)
        ax.set_title(f"Colored by RIX (r={r_all:.2f})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Roughness Mismatch: Raw vs Terrain-Dampened Variants (sector-level, energy-weighted)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("roughness_mismatch_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: roughness_mismatch_exploration.png")


def print_summary(pair_agg):
    """Compare all three variants with correlations and partial correlations."""

    features = [
        ("roughness_mismatch", "Raw roughness mismatch"),
        ("rough_dampened_rix", "Variant A (RIX-dampened)"),
        ("rough_dampened_severity", "Variant B (severity-dampened)"),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY: Roughness Mismatch — Raw vs Dampened Variants")
    print("=" * 70)

    # Overall correlations
    print(f"\n  --- Overall correlations with |error| ---")
    for feat_col, feat_name in features:
        r, p = stats.pearsonr(pair_agg[feat_col], pair_agg["actual_abs_error"])
        print(f"    {feat_name:35s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # Stratified by terrain
    print(f"\n  --- Stratified by terrain ---")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 4:
            continue
        print(f"\n  {cat} (n={len(subset)}):")
        for feat_col, feat_name in features:
            r, p = stats.pearsonr(subset[feat_col], subset["actual_abs_error"])
            print(f"    {feat_name:35s}: r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # Partial correlations (controlling for log_dist_norm + RIX_pair)
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

    # Correlation between variants
    print(f"\n  --- Correlation between variants ---")
    r_ab, _ = stats.pearsonr(pair_agg["rough_dampened_rix"], pair_agg["rough_dampened_severity"])
    r_raw_a, _ = stats.pearsonr(pair_agg["roughness_mismatch"], pair_agg["rough_dampened_rix"])
    r_raw_b, _ = stats.pearsonr(pair_agg["roughness_mismatch"], pair_agg["rough_dampened_severity"])
    print(f"    Raw vs A (RIX):         r={r_raw_a:.3f}")
    print(f"    Raw vs B (severity):    r={r_raw_b:.3f}")
    print(f"    A vs B:                 r={r_ab:.3f}")

    # Winner summary
    print(f"\n  --- Recommendation ---")
    results = {}
    for feat_col, feat_name in features:
        # Flat-terrain r (where we expect signal)
        flat = pair_agg[pair_agg["terrain_cat"] == "Flat"]
        if len(flat) > 3:
            r_flat, _ = stats.pearsonr(flat[feat_col], flat["actual_abs_error"])
        else:
            r_flat = 0
        # Complex-terrain r (should be low/near-zero)
        comp = pair_agg[pair_agg["terrain_cat"] == "Complex"]
        if len(comp) > 3:
            r_comp, _ = stats.pearsonr(comp[feat_col], comp["actual_abs_error"])
        else:
            r_comp = 0
        results[feat_name] = {"r_flat": r_flat, "r_complex": r_comp,
                               "score": abs(r_flat) - abs(r_comp)}
        print(f"    {feat_name:35s}: flat r={r_flat:+.3f}, complex r={r_comp:+.3f}, "
              f"score (|r_flat| - |r_complex|) = {results[feat_name]['score']:+.3f}")

    best = max(results, key=lambda k: results[k]["score"])
    print(f"\n    → Best variant: {best}")
    print(f"      (highest flat signal with lowest complex noise)")


if __name__ == "__main__":
    df, pair_agg = load_data()
    print_summary(pair_agg)
    plot_exploration(pair_agg)
