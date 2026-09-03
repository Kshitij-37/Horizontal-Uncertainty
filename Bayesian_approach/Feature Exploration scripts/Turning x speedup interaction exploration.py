"""
Turning × Speedup Gradient Interaction Exploration

Hypothesis: When BOTH turning angle and speedup change rapidly between neighbouring
            sectors, the flow is particularly hard to model. Either alone is manageable;
            together they compound prediction error.

Features:
  - abs_turning: |d_turning_deg| per sector
  - speedup_gradient: max neighbouring-sector difference in log(speedup) per pair
  - turning_gradient: max neighbouring-sector difference in turning per pair
  - interaction: turning_gradient × speedup_gradient (pair-level)

Plots:
1. Actual |error| vs interaction, by terrain
2. 2D heatmap: turning_gradient vs speedup_gradient, color = error
3. Terrain-stratified correlations
4. Partial correlations controlling for existing features
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
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}

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


def max_neighbour_diff(group, col):
    """Max absolute difference between neighbouring sectors (circular)."""
    ordered = group.sort_values("sector_idx")[col].values
    diffs = np.abs(np.diff(ordered))
    wrap_diff = np.abs(ordered[-1] - ordered[0])
    all_diffs = np.append(diffs, wrap_diff)
    return np.max(all_diffs) if len(all_diffs) > 0 else 0.0


def load_data():
    """Load data, compute features, return pair-level DataFrame."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()
    df["sector_idx"] = df["sector_name"].map(SECTOR_TO_IDX)

    # Signed relative WS error per sector
    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # Compute sector-level features needed for gradients
    df["log_speedup_WTG"] = np.log(df["overall_speedup_WTG_factor"])
    df["log_speedup_MM"] = np.log(df["overall_speedup_MM_factor"])
    df["abs_turning"] = np.abs(df["d_turning_deg"])

    # Pair-level gradients
    speedup_grad_WTG = df.groupby("pair_id", group_keys=False).apply(
        lambda g: pd.Series(max_neighbour_diff(g, "log_speedup_WTG"), index=g.index))
    speedup_grad_MM = df.groupby("pair_id", group_keys=False).apply(
        lambda g: pd.Series(max_neighbour_diff(g, "log_speedup_MM"), index=g.index))
    df["speedup_gradient"] = np.abs(speedup_grad_WTG - speedup_grad_MM)

    df["turning_gradient"] = df.groupby("pair_id", group_keys=False).apply(
        lambda g: pd.Series(max_neighbour_diff(g, "d_turning_deg"), index=g.index))

    # Pair-level RIX
    if "RIX_avg_0.0501_overall_pair" not in df.columns:
        df["RIX_avg_0.0501_overall_pair"] = df.groupby("pair_id")["RIX_avg_0.0501_sector"].transform("mean")

    def classify_terrain(rix):
        for name, (lo, hi) in TERRAIN_BINS.items():
            if lo <= rix < hi:
                return name
        return "Complex"

    df["terrain_cat"] = df["RIX_avg_0.0501_overall_pair"].apply(classify_terrain)

    # Pair-level aggregation
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "turning_gradient": g["turning_gradient"].iloc[0],
            "speedup_gradient": g["speedup_gradient"].iloc[0],
            "turning_x_speedup": g["turning_gradient"].iloc[0] * g["speedup_gradient"].iloc[0],
            "distance_m": g["distance_m"].iloc[0],
            "abs_dz": np.abs(g["dz"].iloc[0]),
            "RIX_pair": g["RIX_avg_0.0501_overall_pair"].iloc[0],
            "terrain_cat": g["terrain_cat"].iloc[0],
            "location": g["location"].iloc[0],
            "log_dist_norm": np.log(g["distance_m"].iloc[0] / max(g["distance_A"].iloc[0], 1)),
        }),
        include_groups=False,
    ).reset_index()

    for col in ["actual_abs_error", "turning_gradient", "speedup_gradient",
                 "turning_x_speedup", "distance_m", "abs_dz", "RIX_pair", "log_dist_norm"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    print(f"\n  Loaded {len(pair_agg)} pairs")
    print(f"  turning_gradient range:  [{pair_agg['turning_gradient'].min():.2f}, {pair_agg['turning_gradient'].max():.2f}]")
    print(f"  speedup_gradient range:  [{pair_agg['speedup_gradient'].min():.4f}, {pair_agg['speedup_gradient'].max():.4f}]")
    print(f"  interaction range:       [{pair_agg['turning_x_speedup'].min():.4f}, {pair_agg['turning_x_speedup'].max():.4f}]")

    for cat in ["Flat", "Semi-complex", "Complex"]:
        n = (pair_agg["terrain_cat"] == cat).sum()
        if n > 0:
            print(f"    {cat}: {n} pairs")

    return df, pair_agg


def plot_exploration(pair_agg):
    """Create 2x2 exploration figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ---- Plot 1: |error| vs interaction by terrain ----
    ax = axes[0, 0]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset["turning_x_speedup"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)

    r, p = stats.pearsonr(pair_agg["turning_x_speedup"], pair_agg["actual_abs_error"])
    sl, ic, _, _, _ = stats.linregress(pair_agg["turning_x_speedup"], pair_agg["actual_abs_error"])
    x_r = np.linspace(pair_agg["turning_x_speedup"].min(), pair_agg["turning_x_speedup"].max(), 50)
    ax.plot(x_r, sl * x_r + ic, "k--", alpha=0.6, linewidth=2)
    ax.set_xlabel("Turning Gradient × Speedup Gradient", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title(f"|Error| vs Turning×Speedup Interaction (r={r:.2f}, p={p:.3f})", fontsize=13, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Plot 2: 2D scatter — turning_grad vs speedup_grad, color = error ----
    ax = axes[0, 1]
    sc = ax.scatter(pair_agg["turning_gradient"], pair_agg["speedup_gradient"],
                    c=pair_agg["actual_abs_error"], cmap="YlOrRd", alpha=0.8, s=70,
                    edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Actual |Error|")
    ax.set_xlabel("Turning Gradient (deg)", fontsize=12)
    ax.set_ylabel("Speedup Gradient (log ratio)", fontsize=12)
    ax.set_title("2D: Turning vs Speedup Gradient (color = |Error|)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Individual components vs error ----
    ax = axes[1, 0]
    r_turn, p_turn = stats.pearsonr(pair_agg["turning_gradient"], pair_agg["actual_abs_error"])
    r_speed, p_speed = stats.pearsonr(pair_agg["speedup_gradient"], pair_agg["actual_abs_error"])
    r_inter, p_inter = stats.pearsonr(pair_agg["turning_x_speedup"], pair_agg["actual_abs_error"])

    labels = ["Turning\ngradient", "Speedup\ngradient", "Turning ×\nSpeedup"]
    r_values = [r_turn, r_speed, r_inter]
    p_values = [p_turn, p_speed, p_inter]
    colors = ["#3498db", "#e67e22", "#8e44ad"]

    bars = ax.bar(labels, r_values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    for bar, r_val, p_val in zip(bars, r_values, p_values):
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"r={r_val:.2f}{sig}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Pearson r with |Error|", fontsize=12)
    ax.set_title("Component vs Interaction Correlation", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Plot 4: By terrain — interaction correlation ----
    ax = axes[1, 1]
    for cat in ["Flat", "Semi-complex", "Complex"]:
        mask = pair_agg["terrain_cat"] == cat
        subset = pair_agg[mask]
        if len(subset) < 4:
            continue
        ax.scatter(subset["turning_x_speedup"], subset["actual_abs_error"],
                   color=TERRAIN_COLORS[cat], label=cat, alpha=0.7, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)
        r_cat, p_cat = stats.pearsonr(subset["turning_x_speedup"], subset["actual_abs_error"])
        sl, ic, _, _, _ = stats.linregress(subset["turning_x_speedup"], subset["actual_abs_error"])
        x_r = np.linspace(subset["turning_x_speedup"].min(), subset["turning_x_speedup"].max(), 50)
        ax.plot(x_r, sl * x_r + ic, color=TERRAIN_COLORS[cat], linestyle="--", alpha=0.8, linewidth=2)
        ax.text(x_r[-1], sl * x_r[-1] + ic, f" r={r_cat:.2f}",
                color=TERRAIN_COLORS[cat], fontsize=9, fontweight="bold", va="center")

    ax.set_xlabel("Turning Gradient × Speedup Gradient", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.set_title("By Terrain: |Error| vs Interaction", fontsize=13, fontweight="bold")
    ax.legend(title="Terrain", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Turning × Speedup Gradient Interaction: Does compound flow complexity drive error?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("turning_x_speedup_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: turning_x_speedup_exploration.png")


def print_summary(pair_agg):
    """Summary statistics and partial correlations."""
    print("\n" + "=" * 70)
    print("SUMMARY: Turning × Speedup Gradient Interaction")
    print("=" * 70)

    # Overall
    r_turn, p_turn = stats.pearsonr(pair_agg["turning_gradient"], pair_agg["actual_abs_error"])
    r_speed, p_speed = stats.pearsonr(pair_agg["speedup_gradient"], pair_agg["actual_abs_error"])
    r_inter, p_inter = stats.pearsonr(pair_agg["turning_x_speedup"], pair_agg["actual_abs_error"])

    print(f"\n  Overall correlations with |error|:")
    print(f"    turning_gradient     : r={r_turn:+.3f}  p={p_turn:.3f}  {'*' if p_turn < 0.05 else ''}")
    print(f"    speedup_gradient     : r={r_speed:+.3f}  p={p_speed:.3f}  {'*' if p_speed < 0.05 else ''}")
    print(f"    turning × speedup    : r={r_inter:+.3f}  p={p_inter:.3f}  {'*' if p_inter < 0.05 else ''}")

    improvement = abs(r_inter) - max(abs(r_turn), abs(r_speed))
    if improvement > 0:
        print(f"\n  → Interaction IMPROVES over best component by {improvement:.3f}")
    else:
        print(f"\n  → Interaction does NOT improve over best component (Δ={improvement:.3f})")

    # By terrain
    print(f"\n  --- Stratified by terrain ---")
    for cat in ["Flat", "Semi-complex", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 4:
            continue
        r, p = stats.pearsonr(subset["turning_x_speedup"], subset["actual_abs_error"])
        print(f"  {cat} (n={len(subset)}): r={r:+.3f}  p={p:.3f}  {'*' if p < 0.05 else ''}")

    # Partial correlation controlling for dist_norm + RIX
    print(f"\n  --- Partial correlations (controlling for log_dist_norm + RIX_pair) ---")
    try:
        from sklearn.linear_model import LinearRegression
        controls = pair_agg[["log_dist_norm", "RIX_pair"]].values

        reg1 = LinearRegression().fit(controls, pair_agg["turning_x_speedup"])
        resid_inter = pair_agg["turning_x_speedup"] - reg1.predict(controls)

        reg2 = LinearRegression().fit(controls, pair_agg["actual_abs_error"])
        resid_error = pair_agg["actual_abs_error"] - reg2.predict(controls)

        r_partial, p_partial = stats.pearsonr(resid_inter, resid_error)
        print(f"    turning × speedup partial: r={r_partial:+.3f}  p={p_partial:.3f}  {'*' if p_partial < 0.05 else ''}")
    except ImportError:
        print("    (sklearn not available)")

    # Correlation between components
    r_comp, _ = stats.pearsonr(pair_agg["turning_gradient"], pair_agg["speedup_gradient"])
    print(f"\n  Correlation between components: r={r_comp:.3f}")
    if abs(r_comp) > 0.6:
        print(f"  ⚠ High collinearity — interaction may not add much beyond components")


if __name__ == "__main__":
    df, pair_agg = load_data()
    print_summary(pair_agg)
    plot_exploration(pair_agg)
