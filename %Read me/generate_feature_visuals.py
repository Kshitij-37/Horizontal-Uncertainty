"""
Generate visual diagrams for Feature Documentation v3.
Run this script to produce PNG figures summarising the model's feature set.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── house style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2196F3"
BLUE_LIGHT = "#90CAF9"
GREEN = "#4CAF50"
GREEN_LIGHT = "#A5D6A7"
RED = "#F44336"
RED_LIGHT = "#EF9A9A"
ORANGE = "#FF9800"
GREY = "#9E9E9E"
DARK = "#333333"


# =========================================================================
# FIGURE 1: Feature tier overview (horizontal bar chart)
# =========================================================================
def plot_feature_tiers():
    fig, ax = plt.subplots(figsize=(14, 10))

    # All 21 features ranked by gamma (dTI removed: WTG TI unavailable at deployment)
    features = [
        ("Normalised distance", 0.365, "Pair", True),
        ("Turning gradient", 0.095, "Sector", True),
        ("Speedup diff std", 0.067, "Pair", True),
        ("Weibull k deviation", 0.066, "Sector", True),
        ("Sample shortfall", 0.062, "Pair", True),
        ("TI at MM", 0.062, "Sector", True),
        ("Flip fraction", 0.059, "Sector", False),
        ("k sector std", 0.055, "Pair", False),
        ("Speedup gradient", 0.055, "Sector", False),
        ("Slope (dz/dist)", 0.048, "Pair", False),
        ("Concentration ratio", 0.039, "Pair", False),
        ("|dz| height diff", 0.034, "Pair", False),
        ("Roughness mismatch", 0.030, "Sector", False),
        ("Severity-weighted RIX", 0.029, "Pair", False),
        ("Terrain complexity", 0.028, "Sector", False),
        ("Speedup mismatch", 0.027, "Sector", False),
        ("Deflection mismatch", 0.024, "Sector", False),
        ("Relative RIX", 0.023, "Sector", False),
        ("Severity fraction", 0.018, "Pair", False),
        ("Terrain mismatch", 0.017, "Sector", False),
        ("Log energy weight", 0.009, "Sector", False),
    ]

    features = features[::-1]  # reverse for bottom-up plotting

    names = [f[0] for f in features]
    gammas = [f[1] for f in features]
    levels = [f[2] for f in features]
    in_top = [f[3] for f in features]

    y_pos = np.arange(len(features))
    colors = []
    for inc, lvl in zip(in_top, levels):
        if inc:
            colors.append(BLUE if lvl == "Pair" else GREEN)
        else:
            colors.append(BLUE_LIGHT if lvl == "Pair" else GREEN_LIGHT)

    bars = ax.barh(y_pos, gammas, color=colors, height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Posterior mean gamma (effect size)", fontsize=12)
    ax.set_title("Feature Ranking by Effect Size (gamma)\nDark = in recommended 6-feature model | Light = passenger (removed)", fontsize=13, pad=15)

    # Add gamma values at end of bars
    for bar, gamma in zip(bars, gammas):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{gamma:.3f}", va="center", fontsize=9, color=DARK)

    # Add multiplier axis on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    mult_ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
    ax2.set_xticks(mult_ticks)
    ax2.set_xticklabels([f"{np.exp(g):.2f}x" for g in mult_ticks], fontsize=9)
    ax2.set_xlabel("sigma multiplier (per +1 SD)", fontsize=10)
    ax2.spines["top"].set_visible(True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=BLUE, label="Pair-level (recommended)"),
        mpatches.Patch(facecolor=GREEN, label="Sector-level (recommended)"),
        mpatches.Patch(facecolor=BLUE_LIGHT, label="Pair-level (passenger)"),
        mpatches.Patch(facecolor=GREEN_LIGHT, label="Sector-level (passenger)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    # Tier divider lines
    tier1_cutoff = 1  # after dist_norm (index from top)
    tier2_cutoff = 6  # after TI_MM (dTI removed)
    ax.axhline(y=len(features) - tier1_cutoff - 0.5, color=RED, linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=len(features) - tier2_cutoff - 0.5, color=ORANGE, linestyle="--", alpha=0.5, linewidth=1)

    ax.text(0.30, len(features) - 0.3, "TIER 1", fontsize=9, color=RED, fontweight="bold", alpha=0.7)
    ax.text(0.30, len(features) - tier1_cutoff - 1.0, "TIER 2 (recommended set)", fontsize=9, color=ORANGE, fontweight="bold", alpha=0.7)
    ax.text(0.30, len(features) - tier2_cutoff - 1.0, "TIER 3 (passengers)", fontsize=9, color=GREY, fontweight="bold", alpha=0.7)

    plt.tight_layout()
    plt.savefig(r"%Read me/feature_tier_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: feature_tier_ranking.png")


# =========================================================================
# FIGURE 2: Ablation results comparison
# =========================================================================
def plot_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    configs = [
        ("Dist only\n(1 feat)", 0.253, 0.415, 0.37),
        ("All Tier 2\n(9 feat)", 0.712, 0.746, 1.35),
        ("Tier 2 no TI\n(8 feat)", 0.650, 0.734, 1.29),
        ("Top 5\n(5 feat)", 0.771, 0.837, 0.96),
        ("Top 5 + TI\n(6 feat)", 0.786, 0.846, 1.03),
        ("Full model\n(21 feat)", 0.819, 0.749, 2.06),
    ]

    names = [c[0] for c in configs]
    pearson = [c[1] for c in configs]
    spearman = [c[2] for c in configs]
    bias = [c[3] for c in configs]
    x = np.arange(len(configs))

    # Highlight the recommended config
    colors_p = [BLUE_LIGHT] * len(configs)
    colors_s = [GREEN_LIGHT] * len(configs)
    colors_b = [RED_LIGHT] * len(configs)
    colors_p[4] = BLUE
    colors_s[4] = GREEN
    colors_b[4] = RED

    # Pearson r
    axes[0].bar(x, pearson, color=colors_p, edgecolor="white", width=0.7)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title("LOO Pearson r\n(linear correlation)", fontsize=12)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(pearson):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold" if i == 4 else "normal")

    # Spearman rho
    axes[1].bar(x, spearman, color=colors_s, edgecolor="white", width=0.7)
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("LOO Spearman rho\n(ranking quality)", fontsize=12)
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(spearman):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold" if i == 4 else "normal")

    # Bias
    axes[2].bar(x, bias, color=colors_b, edgecolor="white", width=0.7)
    axes[2].set_ylabel("Bias (pp): sigma - |error|")
    axes[2].set_title("Prediction Bias\n(lower = better)", fontsize=12)
    for i, v in enumerate(bias):
        axes[2].text(i, v + 0.05, f"+{v:.1f}pp", ha="center", fontsize=9, fontweight="bold" if i == 4 else "normal")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)

    fig.suptitle("Feature Ablation Study: Top 5 + TI is the recommended configuration",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(r"%Read me/feature_ablation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: feature_ablation_comparison.png")


# =========================================================================
# FIGURE 3: Feature concept diagrams (what each feature measures)
# =========================================================================
def plot_feature_concepts():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- 1. Normalised distance ---
    ax = axes[0, 0]
    distances = np.linspace(500, 20000, 100)
    dist_A_flat = 8000
    dist_A_complex = 3000
    ax.plot(distances, np.log(distances / dist_A_flat), color=GREEN, linewidth=2, label="Flat terrain (dist_A=8km)")
    ax.plot(distances, np.log(distances / dist_A_complex), color=RED, linewidth=2, label="Complex terrain (dist_A=3km)")
    ax.axhline(y=0, color=GREY, linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(distances, -2, 0, alpha=0.05, color=GREEN)
    ax.fill_between(distances, 0, 3, alpha=0.05, color=RED)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("log(distance / distance_A)")
    ax.set_title("1. Normalised Distance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(15000, -0.3, "Within\nbounds", fontsize=9, color=GREEN, ha="center", fontstyle="italic")
    ax.text(15000, 0.5, "Exceeds\nbounds", fontsize=9, color=RED, ha="center", fontstyle="italic")
    ax.set_ylim(-2, 3)

    # --- 2. Turning gradient ---
    ax = axes[0, 1]
    sectors = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
    turning_easy = [2, 2.5, 3, 3.5, 3, 2.5, 2, 2.5, 3, 3.5, 3, 2.5]
    turning_hard = [1, 2, 7, 8, 2, 1, 1, 3, 6, 8, 3, 1]
    x_sec = np.arange(12)
    ax.plot(x_sec, turning_easy, "o-", color=GREEN, linewidth=2, label="Low gradient (easy)")
    ax.plot(x_sec, turning_hard, "s-", color=RED, linewidth=2, label="High gradient (hard)")
    # Highlight gradient
    ax.annotate("", xy=(3, 8), xytext=(2, 7),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.annotate("gradient = 6 deg", xy=(3.2, 7.5), fontsize=9, color=RED, fontstyle="italic")
    ax.set_xticks(x_sec)
    ax.set_xticklabels(sectors, fontsize=8, rotation=45)
    ax.set_ylabel("Deflection (deg)")
    ax.set_title("2. Turning Gradient", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # --- 3. Speedup diff std ---
    ax = axes[0, 2]
    su_diff_consistent = [0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.02, 0.01]
    su_diff_variable = [0.08, -0.05, 0.10, -0.03, 0.07, 0.02, -0.06, 0.09, -0.04, 0.11, -0.02, 0.05]
    ax.bar(x_sec - 0.2, su_diff_consistent, 0.35, color=GREEN, alpha=0.8, label=f"Consistent (std={np.std(su_diff_consistent):.3f})")
    ax.bar(x_sec + 0.2, su_diff_variable, 0.35, color=RED, alpha=0.8, label=f"Variable (std={np.std(su_diff_variable):.3f})")
    ax.axhline(y=0, color=DARK, linewidth=0.5)
    ax.set_xticks(x_sec)
    ax.set_xticklabels(sectors, fontsize=8, rotation=45)
    ax.set_ylabel("Speedup_MM - Speedup_WTG")
    ax.set_title("3. Speedup Diff Std", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # --- 4. Weibull k deviation ---
    ax = axes[1, 0]
    from scipy.stats import weibull_min
    k_values = [2.0, 2.0, 3.5, 2.0, 2.0]
    k_mean = 2.0
    ws = np.linspace(0, 20, 200)
    colors_k = [GREY, GREY, RED, GREY, GREY]
    labels_k = ["Normal sector (k=2.0)", None, "Unusual sector (k=3.5)", None, None]
    for k, c, lbl in zip(k_values, colors_k, labels_k):
        A = 7  # scale
        pdf = weibull_min.pdf(ws, k, scale=A)
        ax.plot(ws, pdf, color=c, linewidth=2 if c == RED else 1, alpha=1 if c == RED else 0.4, label=lbl)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("4. Weibull k Deviation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.annotate("k_deviation = |3.5 - 2.0| = 1.5", xy=(12, 0.12), fontsize=9, color=RED, fontstyle="italic")

    # --- 5. Sample shortfall ---
    ax = axes[1, 1]
    pair_samples = [12000, 8000, 5000, 3000, 1500]
    pair_labels = ["12 months", "8 months", "5 months", "3 months", "1.5 months"]
    max_s = 12000
    shortfalls = [np.log(max_s) - np.log(s) for s in pair_samples]
    bars = ax.barh(np.arange(5), shortfalls, color=[GREEN, GREEN_LIGHT, ORANGE, RED_LIGHT, RED], height=0.6)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("Sample shortfall = log(max) - log(samples)")
    ax.set_title("5. Sample Shortfall", fontsize=12, fontweight="bold")
    for bar, sf in zip(bars, shortfalls):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{sf:.2f}", va="center", fontsize=10)

    # --- 6. TI concept ---
    ax = axes[1, 2]
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    ws_low_ti = 8 + 0.5 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.3, 500)
    ws_high_ti = 8 + 0.5 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 1.5, 500)
    ax.plot(t, ws_low_ti, color=GREEN, linewidth=0.8, alpha=0.8, label=f"Low TI ({np.std(ws_low_ti)/np.mean(ws_low_ti):.1%})")
    ax.plot(t, ws_high_ti, color=RED, linewidth=0.8, alpha=0.8, label=f"High TI ({np.std(ws_high_ti)/np.mean(ws_high_ti):.1%})")
    ax.axhline(y=8, color=DARK, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title("6. Turbulence Intensity (TI)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(2, 14)

    fig.suptitle("What Each Feature Measures", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(r"%Read me/feature_concept_diagrams.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: feature_concept_diagrams.png")


# =========================================================================
# FIGURE 4: Features explored and rejected (visual summary)
# =========================================================================
def plot_rejected_features():
    fig, ax = plt.subplots(figsize=(14, 8))

    rejected = [
        ("SLF / Water exposure", 0.44, 0.00, "Redundant with\nroughness_mismatch"),
        ("Power curve sensitivity", -0.37, 0.05, "Terrain-confounded\n(flat r=0.05)"),
        ("Self-prediction |MM|", -0.31, -0.12, "Confounded with\nterrain & WS"),
        ("Standalone log_distance", 0.30, None, "Confounded; replaced\nby dist_norm"),
        ("|dRIX_0.0501|", None, -0.28, "NEGATIVE partial r\n(confounded)"),
        ("Turning x Speedup\ninteraction", None, None, "Worse than\ncomponents alone"),
        ("Roughness change count", -0.02, None, "No signal\n(r ~ 0)"),
        ("dz x Distance\ninteraction", 0.16, None, "Redundant with\ndistance (r=0.97)"),
        ("Steep region exposure", -0.10, None, "Opposite to\nhypothesis"),
    ]

    y_pos = np.arange(len(rejected))[::-1]

    for i, (name, raw_r, partial_r, reason) in enumerate(rejected):
        y = y_pos[i]

        # Raw correlation bar
        if raw_r is not None:
            color = RED if raw_r < 0 else ORANGE
            ax.barh(y + 0.15, raw_r, height=0.25, color=color, alpha=0.6, label="Raw r" if i == 0 else None)
            ax.text(raw_r + (0.02 if raw_r >= 0 else -0.02), y + 0.15,
                    f"raw: {raw_r:+.2f}", va="center", fontsize=8,
                    ha="left" if raw_r >= 0 else "right")

        # Partial correlation bar
        if partial_r is not None:
            ax.barh(y - 0.15, partial_r, height=0.25, color=RED, alpha=0.9, label="Partial r" if i == 0 else None)
            ax.text(partial_r + (0.02 if partial_r >= 0 else -0.02), y - 0.15,
                    f"partial: {partial_r:+.2f}", va="center", fontsize=8,
                    ha="left" if partial_r >= 0 else "right")

        # Reason box
        ax.text(0.55, y, reason, fontsize=9, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=RED_LIGHT, alpha=0.3))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rejected], fontsize=10)
    ax.set_xlabel("Correlation with |error|", fontsize=11)
    ax.axvline(x=0, color=DARK, linewidth=0.5)
    ax.set_xlim(-0.5, 0.9)
    ax.set_title("Features Explored and Rejected\n(raw correlation can be misleading -- always check partial r and terrain stratification)",
                 fontsize=13, pad=15)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(r"%Read me/features_rejected_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: features_rejected_summary.png")


# =========================================================================
# FIGURE 5: Model structure diagram
# =========================================================================
def plot_model_structure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(5, 7.5, "Model Structure: Windspeed Sector-Level Uncertainty",
            fontsize=16, fontweight="bold", ha="center", va="center")

    # Data box
    data_box = mpatches.FancyBboxPatch((0.3, 5.5), 3.5, 1.5, boxstyle="round,pad=0.2",
                                        facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=2)
    ax.add_patch(data_box)
    ax.text(2.05, 6.8, "INPUT DATA", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.05, 6.3, "492 sectors from 41 pairs\n12 sectors per pair\nSigned WS error per sector", fontsize=9, ha="center")

    # Feature box
    feat_box = mpatches.FancyBboxPatch((0.3, 3.2), 3.5, 2.0, boxstyle="round,pad=0.2",
                                       facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=2)
    ax.add_patch(feat_box)
    ax.text(2.05, 4.9, "6 FEATURES", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.05, 4.4, "Pair: dist_norm, speedup_diff_std,\n"
                        "      sample_shortfall\n"
                        "Sector: turning_grad, k_deviation,\n"
                        "        TI_MM",
            fontsize=9, ha="center", family="monospace")

    # Arrow from data to features
    ax.annotate("", xy=(2.05, 5.2), xytext=(2.05, 5.5),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Model box
    model_box = mpatches.FancyBboxPatch((5.0, 3.5), 4.5, 3.5, boxstyle="round,pad=0.2",
                                         facecolor="#FFF9C4", edgecolor=ORANGE, linewidth=2)
    ax.add_patch(model_box)
    ax.text(7.25, 6.7, "BAYESIAN MODEL", fontsize=11, fontweight="bold", ha="center")
    ax.text(7.25, 6.1, r"$e \sim \mathrm{StudentT}(\nu,\ \mu=0,\ \sigma)$", fontsize=13, ha="center")
    ax.text(7.25, 5.4, r"$\log(\sigma) = \log\sigma_0 + \sum \gamma_i \cdot z_i + \epsilon_{pair}$",
            fontsize=13, ha="center")
    ax.text(7.25, 4.7, r"$\gamma_i \sim \mathrm{HalfNormal}(0.3)$    (always $\geq 0$)", fontsize=11, ha="center")
    ax.text(7.25, 4.2, r"$\epsilon_{pair} \sim \mathrm{Normal}(0, \sigma_{pair})$", fontsize=11, ha="center")
    ax.text(7.25, 3.7, r"$\nu \approx 9$    (moderately heavy tails)", fontsize=10, ha="center", color=GREY)

    # Arrow from features to model
    ax.annotate("", xy=(5.0, 4.5), xytext=(3.8, 4.2),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Output box
    out_box = mpatches.FancyBboxPatch((5.0, 0.5), 4.5, 2.5, boxstyle="round,pad=0.2",
                                       facecolor=RED_LIGHT, edgecolor=RED, linewidth=2, alpha=0.5)
    ax.add_patch(out_box)
    ax.text(7.25, 2.7, "OUTPUT", fontsize=11, fontweight="bold", ha="center")
    ax.text(7.25, 2.1, r"Predicted $\sigma$ per sector", fontsize=11, ha="center")
    ax.text(7.25, 1.5, "Confidence intervals:\n"
                        "68% CI:  +/- 1.0 * sigma\n"
                        "90% CI:  +/- 1.8 * sigma\n"
                        "95% CI:  +/- 2.3 * sigma",
            fontsize=9, ha="center", family="monospace")

    # Arrow from model to output
    ax.annotate("", xy=(7.25, 3.0), xytext=(7.25, 3.5),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Validation box
    val_box = mpatches.FancyBboxPatch((0.3, 0.5), 3.5, 2.3, boxstyle="round,pad=0.2",
                                       facecolor="#E8EAF6", edgecolor="#5C6BC0", linewidth=2)
    ax.add_patch(val_box)
    ax.text(2.05, 2.5, "VALIDATION (LOO CV)", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.05, 1.8, "Hold out physical pair\n(both directions)\n"
                        "Pearson r = 0.82\n"
                        "Spearman = 0.75\n"
                        "90% coverage = 100%",
            fontsize=9, ha="center")

    plt.savefig(r"%Read me/model_structure_diagram.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: model_structure_diagram.png")


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

    plot_feature_tiers()
    plot_ablation()
    plot_feature_concepts()
    plot_rejected_features()
    plot_model_structure()

    print("\nAll visuals generated in '%Read me/' folder.")
