"""
Power Curve Sensitivity Exploration

Hypothesis: Projects at mean wind speeds in the steep part of the power curve
            (~5-8 m/s) have higher energy-weighted errors because small WS errors
            get amplified by the power curve slope.

Uses a generic normalised power curve to compute sensitivity at each pair's
energy-weighted mean wind speed.

Generic power curve:
    P(v) = 0                    for v < v_cutin (3.5 m/s)
    P(v) = ((v - v_cutin) / (v_rated - v_cutin))^3   for v_cutin <= v <= v_rated
    P(v) = 1                    for v > v_rated (12 m/s)

Sensitivity = dP/dv (normalised slope at that wind speed)
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

# Generic power curve parameters
V_CUTIN = 3.5
V_RATED = 12.0


def generic_power(v):
    """Normalised generic power curve (0-1)."""
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    cubic_mask = (v >= V_CUTIN) & (v <= V_RATED)
    above_rated = v > V_RATED
    p[cubic_mask] = ((v[cubic_mask] - V_CUTIN) / (V_RATED - V_CUTIN)) ** 3
    p[above_rated] = 1.0
    return p


def generic_sensitivity(v):
    """
    dP/dv of generic power curve — how much power changes per m/s.
    Higher = more sensitive to wind speed errors.
    """
    v = np.asarray(v, dtype=float)
    s = np.zeros_like(v)
    cubic_mask = (v >= V_CUTIN) & (v <= V_RATED)
    s[cubic_mask] = 3 * ((v[cubic_mask] - V_CUTIN) / (V_RATED - V_CUTIN)) ** 2 / (V_RATED - V_CUTIN)
    return s


def generic_elasticity(v):
    """
    Relative sensitivity: (dP/P) / (dv/v) = (v/P) * dP/dv
    How much a 1% WS error amplifies into energy error.
    For pure cubic: elasticity = 3. Near rated: drops to 0.
    """
    v = np.asarray(v, dtype=float)
    p = generic_power(v)
    s = generic_sensitivity(v)
    # Avoid division by zero
    safe = p > 1e-6
    e = np.zeros_like(v)
    e[safe] = (v[safe] / p[safe]) * s[safe]
    return e


def load_data():
    """Load and aggregate to pair level with energy-weighted mean WS."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # Energy-weighted pair-level aggregation
    pair_agg = df.groupby("pair_id").apply(
        lambda g: pd.Series({
            "actual_abs_error": (
                (g["abs_error"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "mean_ws_self": (
                (g["Mean_windspeed_self"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "mean_ws_pred": (
                (g["Mean_windspeed_predicted"] * g["weight_energy_predicted"]).sum()
                / g["weight_energy_predicted"].sum()
            ),
            "location": g["location"].iloc[0],
            "distance_m": g["distance_m"].iloc[0],
        })
    ).reset_index()

    for col in ["actual_abs_error", "mean_ws_self", "mean_ws_pred", "distance_m"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    # Compute sensitivity and elasticity at pair mean WS
    pair_agg["sensitivity"] = generic_sensitivity(pair_agg["mean_ws_self"].values)
    pair_agg["elasticity"] = generic_elasticity(pair_agg["mean_ws_self"].values)
    pair_agg["generic_power"] = generic_power(pair_agg["mean_ws_self"].values)

    print(f"  Loaded {len(pair_agg)} pairs")
    print(f"  Mean WS range: {pair_agg['mean_ws_self'].min():.1f} - {pair_agg['mean_ws_self'].max():.1f} m/s")
    print(f"  Sensitivity range: {pair_agg['sensitivity'].min():.4f} - {pair_agg['sensitivity'].max():.4f}")
    print(f"  Elasticity range: {pair_agg['elasticity'].min():.2f} - {pair_agg['elasticity'].max():.2f}")

    return pair_agg


def plot_exploration(pair_agg):
    """4-panel exploration."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ---- Panel 1: Generic power curve + sensitivity ----
    ax = axes[0, 0]
    v_range = np.linspace(2, 16, 200)
    ax.plot(v_range, generic_power(v_range), "b-", linewidth=2, label="Power (normalised)")
    ax2 = ax.twinx()
    ax2.plot(v_range, generic_sensitivity(v_range), "r--", linewidth=2, label="Sensitivity (dP/dv)")
    ax2.plot(v_range, generic_elasticity(v_range), "g:", linewidth=2, label="Elasticity (% amplification)")

    # Mark where pairs fall
    for _, row in pair_agg.iterrows():
        ax.axvline(row["mean_ws_self"], color="gray", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("Normalised Power", fontsize=12, color="blue")
    ax2.set_ylabel("Sensitivity / Elasticity", fontsize=12, color="red")
    ax.set_title("Generic Power Curve + Sensitivity", fontsize=13, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Actual |error| vs mean WS ----
    ax = axes[0, 1]
    sc = ax.scatter(
        pair_agg["mean_ws_self"], pair_agg["actual_abs_error"],
        c=pair_agg["distance_m"], cmap="viridis",
        s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Distance (m)")

    if len(pair_agg) > 3:
        r, p = stats.pearsonr(pair_agg["mean_ws_self"], pair_agg["actual_abs_error"])
        slope, intercept, _, _, _ = stats.linregress(pair_agg["mean_ws_self"], pair_agg["actual_abs_error"])
        x_fit = np.linspace(pair_agg["mean_ws_self"].min(), pair_agg["mean_ws_self"].max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=2)
        ax.set_title(f"Actual |Error| vs Mean WS (r={r:.2f}, p={p:.3f})", fontsize=13, fontweight="bold")
    else:
        ax.set_title("Actual |Error| vs Mean WS", fontsize=13, fontweight="bold")

    ax.set_xlabel("Energy-weighted Mean WS at self (m/s)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Actual |error| vs sensitivity ----
    ax = axes[1, 0]
    sc = ax.scatter(
        pair_agg["sensitivity"], pair_agg["actual_abs_error"],
        c=pair_agg["mean_ws_self"], cmap="coolwarm",
        s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Mean WS (m/s)")

    if len(pair_agg) > 3:
        r, p = stats.pearsonr(pair_agg["sensitivity"], pair_agg["actual_abs_error"])
        slope, intercept, _, _, _ = stats.linregress(pair_agg["sensitivity"], pair_agg["actual_abs_error"])
        x_fit = np.linspace(pair_agg["sensitivity"].min(), pair_agg["sensitivity"].max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=2)
        ax.set_title(f"Actual |Error| vs Sensitivity (r={r:.2f}, p={p:.3f})", fontsize=13, fontweight="bold")
    else:
        ax.set_title("Actual |Error| vs Sensitivity", fontsize=13, fontweight="bold")

    ax.set_xlabel("Generic Power Curve Sensitivity (dP/dv)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Actual |error| vs elasticity ----
    ax = axes[1, 1]
    sc = ax.scatter(
        pair_agg["elasticity"], pair_agg["actual_abs_error"],
        c=pair_agg["mean_ws_self"], cmap="coolwarm",
        s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Mean WS (m/s)")

    if len(pair_agg) > 3:
        r, p = stats.pearsonr(pair_agg["elasticity"], pair_agg["actual_abs_error"])
        slope, intercept, _, _, _ = stats.linregress(pair_agg["elasticity"], pair_agg["actual_abs_error"])
        x_fit = np.linspace(pair_agg["elasticity"].min(), pair_agg["elasticity"].max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=2)
        ax.set_title(f"Actual |Error| vs Elasticity (r={r:.2f}, p={p:.3f})", fontsize=13, fontweight="bold")
    else:
        ax.set_title("Actual |Error| vs Elasticity", fontsize=13, fontweight="bold")

    ax.set_xlabel("Elasticity (WS error amplification factor)", fontsize=12)
    ax.set_ylabel("Actual |Error| (energy-weighted)", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Power Curve Sensitivity: Does mean WS affect energy-weighted prediction error?",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("power_curve_sensitivity_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: power_curve_sensitivity_exploration.png")


def print_summary(pair_agg):
    """Bin pairs by mean WS and show error stats."""

    print("\n" + "=" * 70)
    print("SUMMARY: |Error| by Mean Wind Speed Bin")
    print("=" * 70)

    bins = [(3, 5), (5, 6), (6, 7), (7, 9), (9, 15)]
    print(f"\n  {'WS bin':>12}  {'n':>4}  {'mean |e|':>10}  {'median |e|':>12}  {'sensitivity':>12}  {'elasticity':>11}")
    print(f"  {'-' * 72}")

    for lo, hi in bins:
        mask = (pair_agg["mean_ws_self"] >= lo) & (pair_agg["mean_ws_self"] < hi)
        subset = pair_agg[mask]
        if len(subset) == 0:
            continue
        print(
            f"  {lo:>4}-{hi:<4} m/s  {len(subset):>4}  "
            f"{subset['actual_abs_error'].mean():>9.2%}  "
            f"{subset['actual_abs_error'].median():>11.2%}  "
            f"{subset['sensitivity'].mean():>11.4f}  "
            f"{subset['elasticity'].mean():>10.2f}"
        )

    # Key correlations
    print(f"\n  Correlations with actual |error|:")
    for col, label in [
        ("mean_ws_self", "Mean WS"),
        ("sensitivity", "Sensitivity"),
        ("elasticity", "Elasticity"),
    ]:
        r, p = stats.pearsonr(pair_agg[col], pair_agg["actual_abs_error"])
        print(f"    {label:<15}: r={r:.3f}  p={p:.3f}")


if __name__ == "__main__":
    pair_agg = load_data()
    print_summary(pair_agg)
    plot_exploration(pair_agg)
