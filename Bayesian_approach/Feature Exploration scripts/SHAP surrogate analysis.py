"""
SHAP Surrogate Model Analysis

Uses a GradientBoosting surrogate to provide an independent "second opinion"
on the Bayesian model's feature set. This script:

1. Incremental R² analysis — adds features one at a time, ordered by marginal
   improvement, to show the "natural" priority ordering.
2. SHAP feature importance — compares to Bayesian γ rankings.
3. SHAP interaction values — flags feature pairs worth testing as interactions.
4. SHAP dependence plots — reveals non-linearities the linear-in-log model misses.
5. Mutual information — catches non-linear dependencies that Pearson r misses.

NOTE: This does NOT replace the Bayesian model. It's a diagnostic tool to
      validate the feature set and catch things the Bayesian model might miss.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
SECTOR_MODEL_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

SECTOR_LABELS = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
SECTOR_TO_IDX = {lab: i for i, lab in enumerate(SECTOR_LABELS)}

EPS = 1e-6

# Feature definitions: (column_name, display_name, level)
# "level" = "sector" or "pair" — determines aggregation method
FEATURE_DEFS = [
    # Sector-level features (energy-weighted to pair)
    ("abs_turning", "|Turning angle|", "sector"),
    ("abs_log_speedup", "|log(Speedup)|", "sector"),
    ("RIX_avg_0.3_sector", "RIX avg (terrain)", "sector"),
    ("abs_dRIX_0.3_sector", "|dRIX| (terrain mismatch)", "sector"),
    ("flip_fraction", "Flip fraction", "sector"),
    ("speedup_gradient", "Speedup gradient", "sector"),
    ("turning_gradient", "Turning gradient", "sector"),
    ("RIX_relative", "Relative RIX", "sector"),
    ("log_energy_weight", "Log energy weight", "sector"),
    ("k_MM_deviation", "Weibull k deviation", "sector"),
    ("k_std", "Inter-sector k variability", "sector"),
    ("severity_fraction", "Severity fraction", "sector"),
    ("severity_weighted_RIX", "Severity-weighted RIX", "sector"),
    ("roughness_mismatch", "Roughness mismatch", "sector"),
    ("abs_speedup_mismatch", "|Speedup mismatch|", "sector"),
    ("abs_deflection_mismatch", "|Deflection mismatch|", "sector"),
    ("TI_MM_clean", "TI at MM", "sector"),
    ("abs_dTI", "|dTI| mismatch", "sector"),
    # Pair-level features (take first per pair)
    ("log_dist_norm", "Normalised distance", "pair"),
    ("abs_dz", "|dz| (height diff)", "pair"),
    ("slope", "Slope (dz/distance)", "pair"),
    ("sample_shortfall", "Sample shortfall", "pair"),
    ("speedup_diff_std", "Speedup diff std", "pair"),
    ("concentration_ratio", "Concentration ratio", "pair"),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_gradient(group, value_col):
    """Max neighbouring-sector gradient (circular)."""
    group = group.sort_values("sector_idx")
    values = group[value_col].values
    n = len(values)
    gradients = np.zeros(n)
    for i in range(n):
        left = values[(i - 1) % n]
        right = values[(i + 1) % n]
        current = values[i]
        gradients[i] = max(abs(current - left), abs(current - right))
    return pd.Series(gradients, index=group.index)


def concentration_ratio(weights):
    effective_sectors = 1 / (weights ** 2).sum()
    return (12 - effective_sectors) / 11


def load_and_prepare():
    """Load data, compute all features, aggregate to pair level."""
    print("Loading data...")
    df = pd.read_excel(SECTOR_MODEL_PATH)
    df = df[df["sector_name"].isin(SECTOR_LABELS)].copy()
    df["sector_idx"] = df["sector_name"].map(SECTOR_TO_IDX)

    # Target: signed relative WS error
    df["e"] = (
        (df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"])
        / df["Mean_windspeed_self"]
    )
    df["abs_error"] = np.abs(df["e"])

    # ---- Compute sector-level features ----
    df["abs_turning"] = np.abs(df["d_turning_deg"])
    df["log_speedup_WTG"] = np.log(np.clip(df["overall_speedup_WTG_factor"], EPS, None))
    df["log_speedup_MM"] = np.log(np.clip(df["overall_speedup_MM_factor"], EPS, None))
    df["abs_log_speedup"] = np.abs(df["log_speedup_WTG"] - df["log_speedup_MM"])

    df["abs_dRIX_0.3_sector"] = np.abs(df["dRIX_0.3_sector"])

    # Flip fraction
    df["flip_sign"] = np.sign(df["log_speedup_WTG"]) != np.sign(df["log_speedup_MM"])
    df["flip_fraction"] = df.groupby("pair_id")["flip_sign"].transform("mean")

    # Speedup gradient
    df["speedup_gradient_WTG"] = df.groupby("pair_id", group_keys=False).apply(
        lambda g: compute_gradient(g, "log_speedup_WTG"), include_groups=False)
    df["speedup_gradient_MM"] = df.groupby("pair_id", group_keys=False).apply(
        lambda g: compute_gradient(g, "log_speedup_MM"), include_groups=False)
    df["speedup_gradient"] = np.abs(df["speedup_gradient_WTG"] - df["speedup_gradient_MM"])

    # Turning gradient
    df["turning_gradient"] = df.groupby("pair_id", group_keys=False).apply(
        lambda g: compute_gradient(g, "d_turning_deg"), include_groups=False)

    # Relative RIX
    pair_mean_RIX = df.groupby("pair_id")["RIX_avg_0.3_sector"].transform("mean")
    df["RIX_relative"] = df["RIX_avg_0.3_sector"] / (pair_mean_RIX + 0.1)

    # Log energy weight
    df["log_energy_weight"] = np.log(df["weight_energy_predicted"] + EPS)

    # Weibull k deviation
    if "k_MM" in df.columns:
        pair_mean_k = df.groupby("pair_id")["k_MM"].transform("mean")
        df["k_MM_deviation"] = np.abs(df["k_MM"] - pair_mean_k)
        df["k_std"] = df.groupby("pair_id")["k_MM"].transform("std")
    else:
        df["k_MM_deviation"] = 0.0
        df["k_std"] = 0.0

    # Severity fraction
    df["dRIX_mild"] = df["dRIX_0.0501_sector"] - df["dRIX_0.3_sector"]
    df["severity_fraction"] = (
        np.abs(df["dRIX_0.3_sector"])
        / (np.abs(df["dRIX_0.3_sector"]) + np.abs(df["dRIX_mild"]) + EPS)
    )
    df["severity_weighted_RIX"] = df["RIX_avg_0.0501_sector"] * df["severity_fraction"]

    # Roughness mismatch
    df["roughness_mismatch"] = np.abs(
        np.log(df["reference_length_WTG"].clip(lower=EPS)
               / df["reference_length_MM"].clip(lower=EPS))
    )

    # Speedup/deflection mismatch
    df["abs_speedup_mismatch"] = np.abs(
        df["overall_speedup_WTG_factor"] - df["overall_speedup_MM_factor"]
    )
    df["abs_deflection_mismatch"] = np.abs(df["d_turning_deg"])

    # TI features
    if "TI_MM_clean" not in df.columns and "TI_MM" in df.columns:
        df["TI_MM_clean"] = df["TI_MM"]
    if "abs_dTI" not in df.columns:
        if "TI_WTG" in df.columns and "TI_MM_clean" in df.columns:
            df["abs_dTI"] = np.abs(df["TI_WTG"] - df["TI_MM_clean"])
        else:
            df["abs_dTI"] = 0.0

    # Speedup diff std
    df["speedup_diff"] = df["overall_speedup_MM_factor"] - df["overall_speedup_WTG_factor"]
    df["speedup_diff_std"] = df.groupby("pair_id")["speedup_diff"].transform("std")

    # Concentration ratio
    df["concentration_ratio"] = df.groupby("pair_id")["weight_energy_predicted"].transform(
        concentration_ratio)

    # ---- Pair-level features ----
    df["log_dist_norm"] = np.log(df["distance_m"] / np.clip(df["distance_A"], 1, None))
    df["abs_dz"] = np.abs(df["dz"])
    df["slope"] = df["abs_dz"] / np.clip(df["distance_m"], EPS, None)

    pair_samples = df.groupby("pair_id")["Sample_count_pred"].transform("sum")
    max_samples = pair_samples.max()
    df["sample_shortfall"] = np.log(max_samples + 1) - np.log(pair_samples + 1)

    # ---- Aggregate to pair level (energy-weighted) ----
    sector_features = [f[0] for f in FEATURE_DEFS if f[2] == "sector"]
    pair_features = [f[0] for f in FEATURE_DEFS if f[2] == "pair"]

    def agg_pair(g):
        result = {}
        # Target
        w = g["weight_energy_predicted"].values
        w_sum = w.sum()
        result["actual_abs_error"] = (g["abs_error"].values * w).sum() / w_sum

        # Sector features: energy-weighted
        for feat in sector_features:
            if feat in g.columns:
                vals = g[feat].values
                valid = np.isfinite(vals)
                if valid.all():
                    result[feat] = (vals * w).sum() / w_sum
                else:
                    result[feat] = np.nan

        # Pair features: take first
        for feat in pair_features:
            if feat in g.columns:
                result[feat] = g[feat].iloc[0]

        result["location"] = g["location"].iloc[0]
        return pd.Series(result)

    pair_agg = df.groupby("pair_id", group_keys=False).apply(
        agg_pair, include_groups=False).reset_index()

    # Ensure numeric
    feature_cols = [f[0] for f in FEATURE_DEFS]
    for col in feature_cols + ["actual_abs_error"]:
        if col in pair_agg.columns:
            pair_agg[col] = pd.to_numeric(pair_agg[col], errors="coerce")

    # Drop rows with NaN in any feature
    available_features = [f for f in feature_cols if f in pair_agg.columns]
    n_before = len(pair_agg)
    pair_agg = pair_agg.dropna(subset=available_features + ["actual_abs_error"])
    n_after = len(pair_agg)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} pairs with missing data")

    print(f"  {len(pair_agg)} pairs, {len(available_features)} features")

    return pair_agg, available_features


# =============================================================================
# 1. INCREMENTAL R² ANALYSIS
# =============================================================================

def incremental_r2(pair_agg, features):
    """Forward stepwise: add one feature at a time, track R² improvement."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    print("\n" + "=" * 70)
    print("1. INCREMENTAL R² ANALYSIS (Forward Stepwise)")
    print("=" * 70)

    X = pair_agg[features].values
    y = pair_agg["actual_abs_error"].values

    feature_names = [f[1] for f in FEATURE_DEFS if f[0] in features]
    feat_map = {f[0]: f[1] for f in FEATURE_DEFS}

    remaining = list(range(len(features)))
    selected = []
    selected_names = []
    r2_history = []

    print(f"\n  {'Step':<5} {'Feature added':<35} {'CV R²':>8} {'ΔR²':>8}")
    print(f"  {'-'*60}")

    prev_r2 = 0.0

    for step in range(min(len(features), 15)):  # Stop at 15 to avoid overfitting
        best_r2 = -np.inf
        best_idx = None

        for idx in remaining:
            trial = selected + [idx]
            X_trial = X[:, trial]

            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            scores = cross_val_score(model, X_trial, y, cv=5,
                                      scoring="r2")
            mean_r2 = scores.mean()

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        delta = best_r2 - prev_r2

        feat_name = feat_map.get(features[best_idx], features[best_idx])
        selected_names.append(feat_name)
        r2_history.append(best_r2)

        print(f"  {step+1:<5} {feat_name:<35} {best_r2:>8.3f} {delta:>+8.3f}")

        prev_r2 = best_r2

        # Stop if marginal gain is tiny
        if step > 3 and delta < 0.005:
            print(f"\n  Stopped: marginal ΔR² < 0.005")
            break

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(r2_history) + 1), r2_history, "o-", color="#2c3e50",
            linewidth=2, markersize=8)
    for i, (name, r2) in enumerate(zip(selected_names, r2_history)):
        ax.annotate(name, (i + 1, r2), textcoords="offset points",
                    xytext=(5, 10), fontsize=8, rotation=30, ha="left")
    ax.set_xlabel("Number of features", fontsize=12)
    ax.set_ylabel("5-fold CV R²", fontsize=12)
    ax.set_title("Incremental R²: Forward Feature Addition", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("incremental_r2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: incremental_r2.png")

    return selected, selected_names, r2_history


# =============================================================================
# 2. SHAP ANALYSIS
# =============================================================================

def shap_analysis(pair_agg, features):
    """Fit GBM, compute SHAP values, plot importance + interactions."""
    try:
        import shap
    except ImportError:
        print("\n  SHAP not installed. Run: pip install shap")
        return None

    from sklearn.ensemble import GradientBoostingRegressor

    print("\n" + "=" * 70)
    print("2. SHAP FEATURE IMPORTANCE & INTERACTIONS")
    print("=" * 70)

    feat_map = {f[0]: f[1] for f in FEATURE_DEFS}
    display_names = [feat_map.get(f, f) for f in features]

    X = pair_agg[features].copy()
    X.columns = display_names
    y = pair_agg["actual_abs_error"].values

    # Fit model
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X, y)

    train_r2 = model.score(X, y)
    print(f"\n  GBM train R² = {train_r2:.3f}")

    # SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---- 2a. SHAP importance bar plot ----
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (GBM Surrogate)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: shap_importance.png")

    # ---- 2b. SHAP beeswarm plot ----
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("SHAP Beeswarm: Feature Value vs Impact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: shap_beeswarm.png")

    # ---- 2c. SHAP interaction values (top pairs) ----
    print("  Computing SHAP interaction values (this may take a moment)...")
    try:
        shap_interaction = explainer.shap_interaction_values(X)

        # Sum absolute interaction values for each feature pair
        n_features = len(features)
        interaction_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    interaction_matrix[i, j] = np.abs(shap_interaction[:, i, j]).mean()

        # Top 10 interactions
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((
                    display_names[i], display_names[j],
                    interaction_matrix[i, j] + interaction_matrix[j, i]
                ))
        interactions.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  Top 10 Feature Interactions:")
        print(f"  {'Feature A':<30} {'Feature B':<30} {'Interaction':>12}")
        print(f"  {'-'*75}")
        for a, b, val in interactions[:10]:
            print(f"  {a:<30} {b:<30} {val:>12.5f}")

        # Heatmap of interactions
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(interaction_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_names, fontsize=8)
        plt.colorbar(im, ax=ax, label="Mean |SHAP interaction|")
        ax.set_title("SHAP Interaction Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("shap_interactions.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved: shap_interactions.png")

    except Exception as ex:
        print(f"  Interaction computation failed: {ex}")

    # ---- 2d. Dependence plots for top features ----
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax_idx, feat_idx in enumerate(top_indices):
        ax = axes[ax_idx // 3, ax_idx % 3]
        shap.dependence_plot(
            feat_idx, shap_values, X, ax=ax, show=False,
            interaction_index="auto",
        )
        ax.set_title(display_names[feat_idx], fontsize=11, fontweight="bold")

    fig.suptitle("SHAP Dependence Plots (top 6 features — color = strongest interaction)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: shap_dependence.png")

    # ---- Print ranking comparison ----
    print(f"\n  SHAP vs Bayesian Ranking Comparison:")
    print(f"  {'Rank':<5} {'SHAP (GBM)':<35} {'mean |SHAP|':>12}")
    print(f"  {'-'*55}")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for rank, idx in enumerate(sorted_idx):
        print(f"  {rank+1:<5} {display_names[idx]:<35} {mean_abs_shap[idx]:>12.5f}")

    return shap_values, model


# =============================================================================
# 3. MUTUAL INFORMATION
# =============================================================================

def mutual_information_analysis(pair_agg, features):
    """Non-linear dependency check using mutual information."""
    from sklearn.feature_selection import mutual_info_regression

    print("\n" + "=" * 70)
    print("3. MUTUAL INFORMATION (Non-linear dependency check)")
    print("=" * 70)

    feat_map = {f[0]: f[1] for f in FEATURE_DEFS}

    X = pair_agg[features].values
    y = pair_agg["actual_abs_error"].values

    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    # Also compute Pearson r for comparison
    pearson_r = np.array([
        abs(stats.pearsonr(X[:, i], y)[0]) for i in range(len(features))
    ])

    # Sort by MI
    sorted_idx = np.argsort(mi)[::-1]

    print(f"\n  {'Rank':<5} {'Feature':<35} {'MI':>8} {'|r|':>8} {'MI-r gap':>10}")
    print(f"  {'-'*70}")
    for rank, idx in enumerate(sorted_idx):
        feat_name = feat_map.get(features[idx], features[idx])
        gap = mi[idx] - pearson_r[idx]
        flag = " ← non-linear?" if gap > 0.05 else ""
        print(f"  {rank+1:<5} {feat_name:<35} {mi[idx]:>8.3f} {pearson_r[idx]:>8.3f} {gap:>+10.3f}{flag}")

    # Plot MI vs Pearson r
    fig, ax = plt.subplots(figsize=(10, 8))
    display_names = [feat_map.get(f, f) for f in features]
    ax.scatter(pearson_r, mi, s=60, alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3)

    # Label points
    for i, name in enumerate(display_names):
        ax.annotate(name, (pearson_r[i], mi[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)

    # Diagonal
    max_val = max(pearson_r.max(), mi.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="MI = |r|")
    ax.set_xlabel("|Pearson r| with |error|", fontsize=12)
    ax.set_ylabel("Mutual Information with |error|", fontsize=12)
    ax.set_title("MI vs Pearson r: Points above diagonal have non-linear signal",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mi_vs_pearson.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: mi_vs_pearson.png")

    return mi, pearson_r


# =============================================================================
# 4. RIX NON-LINEARITY INVESTIGATION
# =============================================================================

def rix_nonlinearity_analysis(pair_agg, features, shap_values):
    """Deep dive into RIX_avg's non-linear relationship with error."""
    try:
        import shap
    except ImportError:
        print("  SHAP not installed")
        return

    from sklearn.ensemble import GradientBoostingRegressor

    print("\n" + "=" * 70)
    print("4. RIX NON-LINEARITY INVESTIGATION")
    print("=" * 70)

    feat_map = {f[0]: f[1] for f in FEATURE_DEFS}
    display_names = [feat_map.get(f, f) for f in features]

    X = pair_agg[features].copy()
    X.columns = display_names
    y = pair_agg["actual_abs_error"].values

    # Find RIX column index
    rix_col = "RIX avg (terrain)"
    if rix_col not in X.columns:
        print(f"  {rix_col} not found in features")
        return

    rix_idx = list(X.columns).index(rix_col)
    rix_values = X[rix_col].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- Plot 1: SHAP dependence for RIX ----
    ax = axes[0, 0]
    shap.dependence_plot(
        rix_idx, shap_values, X, ax=ax, show=False,
        interaction_index="auto",
    )
    ax.set_title("SHAP Dependence: RIX avg", fontsize=12, fontweight="bold")

    # ---- Plot 2: Raw RIX vs |error| with LOWESS ----
    ax = axes[0, 1]
    ax.scatter(rix_values, y, alpha=0.7, s=60, edgecolors="white",
               linewidths=0.5, c="#2c3e50", zorder=3)

    # LOWESS smooth
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y, rix_values, frac=0.5, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], "r-", linewidth=2.5, label="LOWESS")
    except ImportError:
        pass

    # Linear trend for comparison
    slope, intercept, r_lin, _, _ = stats.linregress(rix_values, y)
    x_line = np.linspace(rix_values.min(), rix_values.max(), 50)
    ax.plot(x_line, slope * x_line + intercept, "b--", linewidth=1.5,
            alpha=0.5, label=f"Linear (r={r_lin:.2f})")

    ax.set_xlabel("RIX avg (terrain)", fontsize=11)
    ax.set_ylabel("Actual |Error|", fontsize=11)
    ax.set_title("RIX vs |Error|: Linear vs LOWESS", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Binned RIX — mean error per bin ----
    ax = axes[0, 2]
    bins = [0, 5, 15, 25, 35, 50, 70]
    bin_labels = ["0-5", "5-15", "15-25", "25-35", "35-50", "50+"]
    pair_agg_copy = pair_agg.copy()
    pair_agg_copy["rix_bin"] = pd.cut(
        pair_agg_copy[features[rix_idx]], bins=bins, labels=bin_labels,
        include_lowest=True
    )

    bin_stats = pair_agg_copy.groupby("rix_bin", observed=False).agg(
        mean_error=("actual_abs_error", "mean"),
        std_error=("actual_abs_error", "std"),
        n=("actual_abs_error", "count"),
    )

    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(bin_stats)))
    bars = ax.bar(range(len(bin_stats)), bin_stats["mean_error"],
                  yerr=bin_stats["std_error"] / np.sqrt(bin_stats["n"].clip(lower=1)),
                  color=colors, edgecolor="white", linewidth=1.5, alpha=0.9)

    # Add n labels
    for i, (idx, row) in enumerate(bin_stats.iterrows()):
        if row["n"] > 0:
            ax.text(i, row["mean_error"] + 0.003, f"n={int(row['n'])}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("RIX avg bin", fontsize=11)
    ax.set_ylabel("Mean |Error| ± SE", fontsize=11)
    ax.set_title("Binned RIX: Is there a threshold?", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    print(f"\n  Binned RIX statistics:")
    print(f"  {'Bin':<10} {'n':>5} {'mean |e|':>10} {'std |e|':>10}")
    print(f"  {'-'*40}")
    for idx, row in bin_stats.iterrows():
        if row["n"] > 0:
            print(f"  {idx:<10} {int(row['n']):>5} {row['mean_error']:>10.2%} {row['std_error']:>10.2%}")

    # ---- Plot 4: RIX vs |error| colored by distance ----
    ax = axes[1, 0]
    dist_values = X["Normalised distance"].values
    sc = ax.scatter(rix_values, y, c=dist_values, cmap="viridis",
                    alpha=0.7, s=60, edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Normalised distance")
    ax.set_xlabel("RIX avg", fontsize=11)
    ax.set_ylabel("Actual |Error|", fontsize=11)
    ax.set_title("RIX vs |Error| colored by dist_norm", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Plot 5: RIX vs |error| colored by roughness_mismatch ----
    ax = axes[1, 1]
    if "Roughness mismatch" in X.columns:
        rough_values = X["Roughness mismatch"].values
        sc = ax.scatter(rix_values, y, c=rough_values, cmap="YlOrRd",
                        alpha=0.7, s=60, edgecolors="white", linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="Roughness mismatch")
    ax.set_xlabel("RIX avg", fontsize=11)
    ax.set_ylabel("Actual |Error|", fontsize=11)
    ax.set_title("RIX vs |Error| colored by roughness", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Plot 6: Threshold test — split at different RIX cutoffs ----
    ax = axes[1, 2]
    cutoffs = np.arange(5, 50, 5)
    r_below = []
    r_above = []
    mean_below = []
    mean_above = []

    for cut in cutoffs:
        below = y[rix_values <= cut]
        above = y[rix_values > cut]
        mean_below.append(below.mean() if len(below) > 0 else np.nan)
        mean_above.append(above.mean() if len(above) > 0 else np.nan)

    ax.plot(cutoffs, mean_below, "o-", color="#2ecc71", linewidth=2,
            markersize=8, label="Mean |error| below cutoff")
    ax.plot(cutoffs, mean_above, "s-", color="#e74c3c", linewidth=2,
            markersize=8, label="Mean |error| above cutoff")

    # t-test at each cutoff
    for i, cut in enumerate(cutoffs):
        below = y[rix_values <= cut]
        above = y[rix_values > cut]
        if len(below) > 3 and len(above) > 3:
            _, p = stats.ttest_ind(below, above, equal_var=False)
            if p < 0.05:
                ax.annotate(f"p={p:.3f}*", (cut, (mean_below[i] + mean_above[i]) / 2),
                           fontsize=7, ha="center", color="black", alpha=0.7)

    ax.set_xlabel("RIX cutoff", fontsize=11)
    ax.set_ylabel("Mean |Error|", fontsize=11)
    ax.set_title("Threshold scan: mean error above/below RIX cutoff",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("RIX Non-Linearity: Does terrain complexity have a threshold effect?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("rix_nonlinearity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: rix_nonlinearity.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pair_agg, features = load_and_prepare()

    # 1. Incremental R²
    selected, selected_names, r2_history = incremental_r2(pair_agg, features)

    # 2. SHAP
    shap_vals, gbm_model = shap_analysis(pair_agg, features)

    # 3. Mutual Information
    mi, pearson_r = mutual_information_analysis(pair_agg, features)

    # 4. RIX non-linearity
    rix_nonlinearity_analysis(pair_agg, features, shap_vals)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
  Generated files:
    incremental_r2.png    — Feature addition order + R² curve
    shap_importance.png   — SHAP bar chart
    shap_beeswarm.png     — SHAP beeswarm (value vs impact direction)
    shap_interactions.png — Interaction heatmap
    shap_dependence.png   — Top 6 dependence plots
    mi_vs_pearson.png     — Mutual info vs linear correlation

  Key things to check:
    1. Does SHAP ranking match your Bayesian γ ranking?
       If not, investigate features that rank differently.
    2. Do interaction values flag feature pairs you haven't tested?
    3. Do dependence plots show non-linearities (curves, thresholds)?
    4. Are there features with high MI but low r? (non-linear signal)
    """)
