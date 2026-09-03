"""
Power Curve Sensitivity Exploration

Hypothesis: Projects at mean wind speeds in the steep part of the power curve
            (~5-8 m/s) have higher energy-weighted errors because small WS errors
            get amplified by the power curve slope.

Now supports custom power curve input from CSV/Excel or manual data entry.

Custom power curve format (CSV or Excel):
    windspeed,power
    0,0
    3,0
    4,50
    5,200
    ...
    12,1500
    25,1500

Sensitivity = dP/dv (slope at that wind speed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

# -----------------------------
# CONFIG
# -----------------------------
SECTOR_MODEL_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

# -----------------------------
# POWER CURVE CLASS
# -----------------------------
class PowerCurve:
    """
    Power curve with sensitivity and elasticity calculations.
    
    Can be initialized from:
    1. CSV/Excel file: PowerCurve.from_file(path)
    2. Arrays: PowerCurve.from_arrays(windspeeds, powers)
    3. Generic curve: PowerCurve.generic()
    """
    
    def __init__(self, windspeeds, powers, normalize=True):
        """
        Initialize power curve from windspeed and power arrays.
        
        Parameters:
        -----------
        windspeeds : array-like
            Wind speeds in m/s (must be sorted ascending)
        powers : array-like
            Corresponding power values (kW or arbitrary units)
        normalize : bool
            If True, normalize power to [0, 1]
        """
        self.ws_raw = np.asarray(windspeeds, dtype=float)
        self.p_raw = np.asarray(powers, dtype=float)
        
        # Sort by windspeed
        sort_idx = np.argsort(self.ws_raw)
        self.ws_raw = self.ws_raw[sort_idx]
        self.p_raw = self.p_raw[sort_idx]
        
        # Normalize if requested
        if normalize:
            p_max = self.p_raw.max()
            if p_max > 0:
                self.p_norm = self.p_raw / p_max
                self.normalized = True
            else:
                raise ValueError("Cannot normalize: max power is zero")
        else:
            self.p_norm = self.p_raw.copy()
            self.normalized = False
        
        # Create interpolator
        self.power_interp = interp1d(
            self.ws_raw, self.p_norm, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(0, self.p_norm[-1])
        )
        
        # Pre-compute sensitivity (numerical derivative)
        self._compute_sensitivity()
        
        print(f"  Power curve loaded:")
        print(f"    WS range: {self.ws_raw.min():.1f} - {self.ws_raw.max():.1f} m/s")
        print(f"    Power range: {self.p_raw.min():.1f} - {self.p_raw.max():.1f} {'(normalized)' if normalize else ''}")
        print(f"    Cut-in: ~{self.ws_raw[self.p_norm > 0.01][0]:.1f} m/s")
        print(f"    Rated: ~{self.ws_raw[self.p_norm > 0.99][0]:.1f} m/s" if any(self.p_norm > 0.99) else "    Rated: Not reached")
    
    def _compute_sensitivity(self):
        """Compute dP/dv numerically using central differences."""
        dv = np.diff(self.ws_raw)
        dp = np.diff(self.p_norm)
        
        # Sensitivity at midpoints
        sensitivity_mid = dp / dv
        ws_mid = (self.ws_raw[:-1] + self.ws_raw[1:]) / 2
        
        # Interpolate back to original grid
        self.sensitivity_interp = interp1d(
            ws_mid, sensitivity_mid,
            kind='linear',
            bounds_error=False,
            fill_value=(0, 0)  # Zero sensitivity outside range
        )
    
    def power(self, v):
        """Get power at wind speed(s) v."""
        return self.power_interp(v)
    
    def sensitivity(self, v):
        """
        Get dP/dv at wind speed(s) v.
        Higher = more sensitive to wind speed errors.
        """
        return self.sensitivity_interp(v)
    
    def elasticity(self, v):
        """
        Get relative sensitivity: (dP/P) / (dv/v) = (v/P) * dP/dv
        How much a 1% WS error amplifies into energy error.
        """
        v = np.asarray(v, dtype=float)
        p = self.power(v)
        s = self.sensitivity(v)
        
        # Avoid division by zero
        safe = p > 1e-6
        e = np.zeros_like(v)
        e[safe] = (v[safe] / p[safe]) * s[safe]
        return e
    
    @classmethod
    def from_file(cls, filepath, ws_col='windspeed', power_col='power', normalize=True):
        """
        Load power curve from CSV or Excel file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV or Excel file
        ws_col : str
            Column name for wind speed
        power_col : str
            Column name for power
        normalize : bool
            Normalize power to [0, 1]
        
        Example CSV format:
            windspeed,power
            0,0
            3,0
            4,50
            5,200
            ...
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("File must be .csv, .xlsx, or .xls")
        
        # Try to find columns (case-insensitive)
        cols_lower = {col.lower(): col for col in df.columns}
        
        ws_col_actual = cols_lower.get(ws_col.lower())
        power_col_actual = cols_lower.get(power_col.lower())
        
        if ws_col_actual is None or power_col_actual is None:
            raise ValueError(
                f"Could not find columns '{ws_col}' and '{power_col}'\n"
                f"Available columns: {list(df.columns)}"
            )
        
        return cls(df[ws_col_actual].values, df[power_col_actual].values, normalize=normalize)
    
    @classmethod
    def from_arrays(cls, windspeeds, powers, normalize=True):
        """Create from numpy arrays or lists."""
        return cls(windspeeds, powers, normalize=normalize)
    
    @classmethod
    def generic(cls, v_cutin=3.5, v_rated=12.0):
        """
        Create generic cubic power curve.
        
        P(v) = 0                    for v < v_cutin
        P(v) = ((v - v_cutin) / (v_rated - v_cutin))^3   for v_cutin <= v <= v_rated
        P(v) = 1                    for v > v_rated
        """
        v = np.linspace(0, 25, 100)
        p = np.zeros_like(v)
        
        cubic_mask = (v >= v_cutin) & (v <= v_rated)
        above_rated = v > v_rated
        
        p[cubic_mask] = ((v[cubic_mask] - v_cutin) / (v_rated - v_cutin)) ** 2
        p[above_rated] = 1.0
        
        return cls(v, p, normalize=False)


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(power_curve):
    """Load and aggregate to pair level with energy-weighted mean WS."""
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()

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
            "RIX_pair": g["RIX_avg_0.0501_overall_pair"].iloc[0],
        })
    ).reset_index()

    for col in ["actual_abs_error", "mean_ws_self", "mean_ws_pred", "distance_m", "RIX_pair"]:
        pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    # Terrain category (same thresholds as distance exploration)
    pair_agg["terrain_cat"] = pd.cut(
        pair_agg["RIX_pair"],
        bins=[0, 40, 50, 999],
        labels=["Flat", "Semi-complex", "Complex"],
    )

    # Compute sensitivity and elasticity at pair mean WS using provided power curve
    pair_agg["sensitivity"] = power_curve.sensitivity(pair_agg["mean_ws_self"].values)
    pair_agg["elasticity"] = power_curve.elasticity(pair_agg["mean_ws_self"].values)
    pair_agg["generic_power"] = power_curve.power(pair_agg["mean_ws_self"].values)

    print(f"\n  Loaded {len(pair_agg)} pairs")
    print(f"  Mean WS range: {pair_agg['mean_ws_self'].min():.1f} - {pair_agg['mean_ws_self'].max():.1f} m/s")
    print(f"  Sensitivity range: {pair_agg['sensitivity'].min():.4f} - {pair_agg['sensitivity'].max():.4f}")
    print(f"  Elasticity range: {pair_agg['elasticity'].min():.2f} - {pair_agg['elasticity'].max():.2f}")

    return pair_agg


def plot_exploration(pair_agg, power_curve):
    """4-panel exploration."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ---- Panel 1: Power curve + sensitivity ----
    ax = axes[0, 0]
    v_range = np.linspace(
        max(0, power_curve.ws_raw.min() - 2), 
        min(25, power_curve.ws_raw.max() + 2), 
        200
    )
    
    ax.plot(v_range, power_curve.power(v_range), "b-", linewidth=2, label="Power (normalized)")
    ax2 = ax.twinx()
    ax2.plot(v_range, power_curve.sensitivity(v_range), "r--", linewidth=2, label="Sensitivity (dP/dv)")
    ax2.plot(v_range, power_curve.elasticity(v_range), "g:", linewidth=2, label="Elasticity (% amplification)")

    # Mark where pairs fall
    for _, row in pair_agg.iterrows():
        ax.axvline(row["mean_ws_self"], color="gray", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("Normalized Power", fontsize=12, color="blue")
    ax2.set_ylabel("Sensitivity / Elasticity", fontsize=12, color="red")
    ax.set_title("Power Curve + Sensitivity", fontsize=13, fontweight="bold")

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

    ax.set_xlabel("Power Curve Sensitivity (dP/dv)", fontsize=12)
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


def terrain_controlled_analysis(pair_agg):
    """
    Check if WS/sensitivity/elasticity correlations survive after controlling
    for terrain. Two approaches:
    1. Stratified: compute correlations within flat-only and complex-only groups
    2. Partial correlation: correlate residuals after regressing out RIX_pair
    """

    TERRAIN_COLORS = {"Flat": "#2ecc71", "Semi-complex": "#f39c12", "Complex": "#e74c3c"}

    print("\n" + "=" * 70)
    print("TERRAIN-CONTROLLED ANALYSIS")
    print("=" * 70)

    # --- 1. Stratified correlations ---
    print("\n  --- Stratified by terrain category ---")
    for cat in ["Flat", "Complex"]:
        subset = pair_agg[pair_agg["terrain_cat"] == cat]
        if len(subset) < 5:
            print(f"\n  {cat} (n={len(subset)}): too few pairs for meaningful correlation")
            continue

        print(f"\n  {cat} (n={len(subset)}):")
        for col, label in [
            ("mean_ws_self", "Mean WS"),
            ("sensitivity", "Sensitivity"),
            ("elasticity", "Elasticity"),
        ]:
            r, p = stats.pearsonr(subset[col], subset["actual_abs_error"])
            sig = "*" if p < 0.05 else ""
            print(f"    {label:<15}: r={r:+.3f}  p={p:.3f} {sig}")

    # --- 2. Partial correlations (regress out RIX_pair) ---
    print(f"\n  --- Partial correlations (controlling for RIX_pair) ---")
    valid = pair_agg.dropna(subset=["RIX_pair", "actual_abs_error"])

    for col, label in [
        ("mean_ws_self", "Mean WS"),
        ("sensitivity", "Sensitivity"),
        ("elasticity", "Elasticity"),
    ]:
        # Regress out RIX from both the feature and the error
        slope_x, intercept_x, _, _, _ = stats.linregress(valid["RIX_pair"], valid[col])
        resid_x = valid[col] - (slope_x * valid["RIX_pair"] + intercept_x)

        slope_y, intercept_y, _, _, _ = stats.linregress(valid["RIX_pair"], valid["actual_abs_error"])
        resid_y = valid["actual_abs_error"] - (slope_y * valid["RIX_pair"] + intercept_y)

        r_partial, p_partial = stats.pearsonr(resid_x, resid_y)
        sig = "*" if p_partial < 0.05 else ""
        print(f"    {label:<15}: r_partial={r_partial:+.3f}  p={p_partial:.3f} {sig}")

    # --- 3. Terrain-stratified plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (col, label) in enumerate([
        ("mean_ws_self", "Mean WS (m/s)"),
        ("sensitivity", "Sensitivity (dP/dv)"),
        ("elasticity", "Elasticity"),
    ]):
        ax = axes[ax_idx]
        for cat in ["Flat", "Complex"]:
            subset = pair_agg[pair_agg["terrain_cat"] == cat]
            if len(subset) == 0:
                continue

            ax.scatter(
                subset[col], subset["actual_abs_error"],
                color=TERRAIN_COLORS[cat], label=cat,
                alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
            )

            if len(subset) > 3:
                slope, intercept, r, p, _ = stats.linregress(subset[col], subset["actual_abs_error"])
                x_fit = np.linspace(subset[col].min(), subset[col].max(), 50)
                ax.plot(x_fit, slope * x_fit + intercept,
                        color=TERRAIN_COLORS[cat], linestyle="--", linewidth=2)
                ax.text(
                    x_fit[-1], slope * x_fit[-1] + intercept,
                    f" r={r:.2f}", color=TERRAIN_COLORS[cat],
                    fontsize=9, fontweight="bold", va="center",
                )

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Actual |Error|", fontsize=12)
        ax.set_title(f"|Error| vs {label.split('(')[0].strip()} by Terrain", fontsize=12, fontweight="bold")
        ax.legend(title="Terrain", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Terrain-Controlled: Does WS/sensitivity signal survive within terrain groups?",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("power_curve_terrain_controlled.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: power_curve_terrain_controlled.png")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    
    # ===================================================
    # OPTION 1: Load from file
    # ===================================================
    power_curve = PowerCurve.from_file(
        r"C:\Users\K_Trivedi\Desktop\Power curve.xlsx",
        ws_col='Wind speed [m/s]',  # adjust column names as needed
        power_col='Power [kW]',
        normalize=True
    )
    
    # ===================================================
    # OPTION 2: Manual input (example Vestas V90-2MW-like curve)
    # ===================================================
    # windspeeds = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25]
    # powers =     [0, 0, 66, 154, 282, 460, 687, 1005, 1407, 1775, 2000, 2000, 2000]  # kW
    # power_curve = PowerCurve.from_arrays(windspeeds, powers, normalize=True)
    
    # ===================================================
    # OPTION 3: Use generic cubic curve (default)
    # ===================================================
    power_curve = PowerCurve.generic(v_cutin=3.5, v_rated=12.0)
    
    # Run analysis
    pair_agg = load_data(power_curve)
    print_summary(pair_agg)
    plot_exploration(pair_agg, power_curve)
    terrain_controlled_analysis(pair_agg)