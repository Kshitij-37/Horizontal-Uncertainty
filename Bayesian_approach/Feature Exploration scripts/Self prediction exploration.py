"""
Self-Prediction (MM) Exploration

Hypothesis: self_prediction_MM measures how well the CFD model predicts wind speed
            at the measurement mast itself (% over/underprediction). A high |self_prediction_MM|
            indicates the model struggles even at the MM location, so extrapolation to
            WTG will likely be worse.

self_prediction_MM is a single pair-level value (not sector-level).

Plots:
1. Actual |error| vs |self_prediction_MM|, colored by terrain
2. Actual |error| vs self_prediction_MM (signed), colored by terrain
3. Residual analysis: does self_pred explain variance beyond existing features?
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


def load_data():
    """Load data, compute pair-level features, return pair-level DataFrame."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    # Check self_prediction_MM availability
    if "self_prediction_MM" not in df.columns:
        raise ValueError("self_prediction_MM column not found in data")

    n_valid = df["self_prediction_MM"].notna().sum()
    n_total = len(df)
    print(f"  self_prediction_MM: {n_valid}/{n_total} non-null sector rows")

    # Signed relative WS error per sector
    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # Pair-level RIX
    if "RIX_avg_0.0501_overall_pair" in df.columns:
        pass
    else:
        df["RIX_avg_0.0501_overall_pair"] = df.groupby("pair_id")["RIX_avg_0.0501_sector"].transform("mean")

    # Terrain category
    def classify_terrain(rix):
        for name, (lo, hi) in TERRAIN_BINS.items():
            if lo <= rix < hi:
                return name
        return "Complex"

    df["terrain_cat"] = df["RIX_avg_0.0501_overall_pair"].apply(classify_terrain)

    # Pair-level aggregation (energy-weighted)
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "self_pred_MM": g["self_prediction_MM"].iloc[0],  # pair-level value
            "abs_self_pred_MM": np.abs(g["self_prediction_MM"].iloc[0]),
            "distance_m": g["distance_m"].iloc[0],
            "abs_dz": np.abs(g["dz"].iloc[0]),
            "RIX_pair": g["RIX_avg_0.0501_overall_pair"].iloc[0],
            "terrain_cat": g["terrain_cat"].iloc[0],
            "location": g["location"].iloc[0],
            # Existing features for partial correlation
            "log_dist_norm": np.log(g["distance_m"].iloc[0] / max(g["distance_A"].iloc[0], 1)),
            "slope": np.abs(g["dz"].iloc[0]) / max(g["distance_m"].iloc[0], 1),
        }),
        include_groups=False,
    ).reset_index()

    # Ensure numeric
    for col in ["actual_abs_error", "self_pred_MM", "abs_self_pred_MM", "distance_m",
                 "abs_dz", "RIX_pair", "log_dist_norm", "slope"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    # Drop pairs missing self_prediction
    n_before = len(pair_agg)
    pair_agg = pair_agg.dropna(subset=["self_pred_MM"])
    n_after = len(pair_agg)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} pairs with missing self_prediction_MM")

    print(f"\n  Loaded {len(pair_agg)} pairs with self_prediction_MM")
    print(f"  self_pred_MM range: [{pair_agg['self_pred_MM'].min():.2%}, {pair_agg['self_pred_MM'].max():.2%}]")
    print(f"  |self_pred_MM| range: [{pair_agg['abs_self_pred_MM'].min():.2%}, {pair_agg['abs_self_pred_MM'].max():.2%}]")

    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        if n > 0:
            print(f"    {cat}: {n} pairs")

    return df, pair_agg


def plot_exploration(pair_agg):
    """Create 2x2 exploration figure."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ---- Plot 1: |error| vs |self_pred| by terrain ----
    ax = axes[0, 0]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset["abs_self_pred_MM"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Overall trend
    r, p = stats.pearsonr(pair_agg["abs_self_pred_MM"], pair_agg["actual_abs_error"])
    slope, intercept, _, _, _ = stats.linregress(pair_agg["abs_self_pred_MM"], pair_agg["actual_abs_error"])
    x_range = np.linspace(pair_agg["abs_self_pred_MM"].min(), pair_agg["abs_self_pred_MM"].max(), 50)
    ax.plot(x_range, slope * x_range + intercept, "k--", alpha=0.6, linewidth=2)
    ax.set_xlabel("|self_prediction_MM| (%)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title(f"|Error| vs |Self Prediction MM| (r={r:.2f}, p={p:.3f})", fontsize=13, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Plot 2: |error| vs signed self_pred by terrain ----
    ax = axes[0, 1]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset["self_pred_MM"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)

    r_signed, p_signed = stats.pearsonr(pair_agg["self_pred_MM"], pair_agg["actual_abs_error"])
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("self_prediction_MM (signed %)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title(f"|Error| vs Self Prediction MM signed (r={r_signed:.2f}, p={p_signed:.3f})", fontsize=13, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Plot 3: By terrain category — trend lines ----
    ax = axes[1, 0]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) < 3:
            continue
        ax.scatter(subset["abs_self_pred_MM"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)
        r_cat, p_cat = stats.pearsonr(subset["abs_self_pred_MM"], subset["actual_abs_error"])
        sl, ic, _, _, _ = stats.linregress(subset["abs_self_pred_MM"], subset["actual_abs_error"])
        x_r = np.linspace(subset["abs_self_pred_MM"].min(), subset["abs_self_pred_MM"].max(), 50)
        ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
        ax.text(x_r[-1], sl * x_r[-1] + ic, f" r={r_cat:.2f}",
                color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")

    ax.set_xlabel("|self_prediction_MM| (%)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title("By Terrain: |Error| vs |Self Prediction MM|", fontsize=13, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Plot 4: |self_pred| vs dist_norm and dz (confounders check) ----
    ax = axes[1, 1]
    sc = ax.scatter(pair_agg["abs_self_pred_MM"], pair_agg["actual_abs_error"],
                    c=pair_agg["log_dist_norm"], cmap="viridis", alpha=0.7, s=60,
                    edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="log(distance/distance_A)")
    ax.set_xlabel("|self_prediction_MM| (%)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title("|Error| vs |Self Pred| colored by dist_norm", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Self-Prediction (MM) Exploration: Does CFD self-accuracy predict extrapolation error?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("self_prediction_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: self_prediction_exploration.png")


def print_summary(pair_agg):
    """Summary statistics and partial correlations."""
    print("\n" + "=" * 70)
    print("SUMMARY: Self-Prediction MM Analysis")
    print("=" * 70)

    # Overall correlations
    r_abs, p_abs = stats.pearsonr(pair_agg["abs_self_pred_MM"], pair_agg["actual_abs_error"])
    r_signed, p_signed = stats.pearsonr(pair_agg["self_pred_MM"], pair_agg["actual_abs_error"])
    print(f"\n  Overall correlations with actual |error|:")
    print(f"    |self_pred_MM|  : r={r_abs:+.3f}  p={p_abs:.3f}  {'*' if p_abs < 0.05 else ''}")
    print(f"    self_pred_MM    : r={r_signed:+.3f}  p={p_signed:.3f}  {'*' if p_signed < 0.05 else ''}")

    # By terrain
    print(f"\n  --- Stratified by terrain ---")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 4:
            continue
        r, p = stats.pearsonr(subset["abs_self_pred_MM"], subset["actual_abs_error"])
        print(f"  {cat} (n={len(subset)}): r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")
        print(f"    mean |self_pred|: {subset['abs_self_pred_MM'].mean():.2%}")
        print(f"    mean |error|:     {subset['actual_abs_error'].mean():.2%}")

    # Partial correlation: controlling for log_dist_norm and RIX
    print(f"\n  --- Partial correlations (controlling for log_dist_norm + RIX_pair) ---")
    try:
        from sklearn.linear_model import LinearRegression
        controls = pair_agg[["log_dist_norm", "RIX_pair"]].values
        # Residualize abs_self_pred
        reg1 = LinearRegression().fit(controls, pair_agg["abs_self_pred_MM"])
        resid_self = pair_agg["abs_self_pred_MM"] - reg1.predict(controls)
        # Residualize actual_abs_error
        reg2 = LinearRegression().fit(controls, pair_agg["actual_abs_error"])
        resid_error = pair_agg["actual_abs_error"] - reg2.predict(controls)

        r_partial, p_partial = stats.pearsonr(resid_self, resid_error)
        print(f"    |self_pred_MM| partial r={r_partial:+.3f}  p={p_partial:.3f}  {'*' if p_partial < 0.05 else ''}")
    except ImportError:
        print("    (sklearn not available for partial correlations)")

    # Correlation with other features (confounders)
    print(f"\n  --- self_pred_MM correlations with existing features ---")
    for feat_name, feat_col in [("log_dist_norm", "log_dist_norm"), ("RIX_pair", "RIX_pair"),
                                 ("|dz|", "abs_dz"), ("slope", "slope")]:
        if feat_col in pair_agg.columns:
            r, p = stats.pearsonr(pair_agg["abs_self_pred_MM"], pair_agg[feat_col])
            print(f"    {feat_name:20s}: r={r:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    df, pair_agg = load_data()
    print_summary(pair_agg)
    plot_exploration(pair_agg)
