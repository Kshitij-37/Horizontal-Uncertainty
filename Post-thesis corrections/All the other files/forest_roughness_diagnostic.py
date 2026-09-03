"""
Diagnostic: are forest pairs fundamentally different from other pairs in the
magnitude vs mismatch structure of their roughness features?

For each pair compute, from the corrected input Excel:
  - roughness magnitude (freq-weighted mean of |(rs_W + rs_M)/2|)  — formula A input
  - roughness mismatch  (freq-weighted mean of |rs_W - rs_M|)      — formula B input
  - ratio               (mismatch / magnitude)
  - observed |e_overall| (freq-weighted signed error, magnitude of)
  - distance, dz

Group pairs by:
  * KEPT       — used in production Bayesian model
  * FOREST     — excluded specifically for being forest (Hultema, Malarberget)
  * OTHER_EXCL — excluded for other reasons (Sallachy, Kayislar, Herzhausen, etc.)

Output:
  - CSV per pair with all above
  - Console summary comparing group medians
  - Scatter plot: magnitude vs mismatch, colored by group, marker sized by |e_overall|
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_XLSX = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "Results", "forest_roughness_diagnostic.csv")
OUT_PNG = os.path.join(OUT_DIR, "Results", "forest_roughness_diagnostic.png")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

FOREST_MASTS = {"2011WM011", "2014WM011", "2012WM006"}
OTHER_EXCLUDED = {"2015WM018","2021PA004","2022PA008","2022PA018","2019HE001",
                   "2019HE002","2019HE003","2022PA021","2023PA085","2024PA014","2024PA107"}

# ── load and prep ─────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_XLSX)
for old, new in [("d_turning_deg", "d_turning_deg_new"),
                 ("overall_speedup_WTG_factor", "overall_speedup_WTG_factor_new"),
                 ("overall_speedup_MM_factor", "overall_speedup_MM_factor_new"),
                 ("rough_speedup_WTG_frac", "rough_speedup_WTG_frac_new"),
                 ("rough_speedup_MM_frac", "rough_speedup_MM_frac_new")]:
    if new in df.columns and old in df.columns:
        mask = df[new].notna()
        df.loc[mask, old] = df.loc[mask, new]

# ── aggregate per pair ────────────────────────────────────────────────────────
def classify(mm, wtg):
    if mm in FOREST_MASTS or wtg in FOREST_MASTS:
        return "FOREST"
    if mm in OTHER_EXCLUDED or wtg in OTHER_EXCLUDED:
        return "OTHER_EXCL"
    return "KEPT"

rows = []
for pair_id, grp in df.groupby("pair_id"):
    mm  = pair_id.split("__")[1]
    wtg = pair_id.split("__")[0]

    rs_W = pd.to_numeric(grp["rough_speedup_WTG_frac"], errors="coerce").values
    rs_M = pd.to_numeric(grp["rough_speedup_MM_frac"],  errors="coerce").values
    valid = ~(np.isnan(rs_W) | np.isnan(rs_M))
    if valid.sum() == 0:
        continue
    rs_W, rs_M = rs_W[valid], rs_M[valid]

    # freq weights (MM Weibull frequency, same aggregation the production model uses)
    freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values[valid]
    if freq.sum() == 0:
        continue
    w = freq / freq.sum()

    magnitude = float(np.sum(w * np.abs((rs_W + rs_M) / 2.0)))   # formula A input
    mismatch  = float(np.sum(w * np.abs(rs_W - rs_M)))            # formula B input

    # observed |e_overall|: sample-count-weighted overall wind speeds
    w_pred = pd.to_numeric(grp["Sample_count_pred"], errors="coerce").fillna(0).values[valid]
    w_self = pd.to_numeric(grp["Sample_count_self"], errors="coerce").fillna(0).values[valid]
    ws_p = pd.to_numeric(grp["Mean_windspeed_predicted"], errors="coerce").values[valid]
    ws_s = pd.to_numeric(grp["Mean_windspeed_self"], errors="coerce").values[valid]
    if w_pred.sum() > 0 and w_self.sum() > 0:
        WS_pred_overall = float(np.sum((w_pred / w_pred.sum()) * ws_p))
        WS_self_overall = float(np.sum((w_self / w_self.sum()) * ws_s))
        e_overall = (WS_pred_overall - WS_self_overall) / WS_self_overall
        abs_e = float(abs(e_overall))
    else:
        e_overall = np.nan; abs_e = np.nan

    distance_m = float(grp["distance_m"].iloc[0])
    dz         = float(grp["dz"].iloc[0]) if "dz" in grp.columns else np.nan

    rows.append({
        "pair_id": pair_id, "MM": mm, "WTG": wtg,
        "group": classify(mm, wtg),
        "magnitude": magnitude, "mismatch": mismatch,
        "ratio_mismatch_over_magnitude": (mismatch / magnitude) if magnitude > 0 else np.nan,
        "abs_e_overall": abs_e, "distance_m": distance_m, "dz": dz,
    })

out = pd.DataFrame(rows).sort_values(["group", "magnitude"]).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

# ── console summary ───────────────────────────────────────────────────────────
print("=" * 100)
print(f"{'group':<12}{'n':>5}{'mag_med':>11}{'mag_max':>11}{'mm_med':>10}{'mm_max':>10}"
      f"{'ratio_med':>11}{'abs_e_med':>11}{'abs_e_max':>11}")
print("=" * 100)
for grp_name in ["KEPT", "FOREST", "OTHER_EXCL"]:
    sub = out[out["group"] == grp_name]
    if len(sub) == 0:
        continue
    print(f"{grp_name:<12}{len(sub):>5}"
          f"{sub['magnitude'].median():>11.5f}{sub['magnitude'].max():>11.5f}"
          f"{sub['mismatch'].median():>10.5f}{sub['mismatch'].max():>10.5f}"
          f"{sub['ratio_mismatch_over_magnitude'].median():>11.3f}"
          f"{sub['abs_e_overall'].median():>11.4f}{sub['abs_e_overall'].max():>11.4f}")

# ── per-forest-pair table ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("FOREST PAIRS (detail)")
print("=" * 100)
fdf = out[out["group"] == "FOREST"].sort_values("pair_id")
for _, r in fdf.iterrows():
    print(f"  {r['pair_id']:<30}  magnitude={r['magnitude']:.5f}  mismatch={r['mismatch']:.5f}  "
          f"ratio={r['ratio_mismatch_over_magnitude']:.3f}  |e|={r['abs_e_overall']:.4f}  d={r['distance_m']:.0f}m")

# ── scatter plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
colors = {"KEPT": "#4a90d9", "FOREST": "#d94a4a", "OTHER_EXCL": "#a0a0a0"}
for grp_name in ["KEPT", "OTHER_EXCL", "FOREST"]:   # forest on top
    sub = out[out["group"] == grp_name]
    sizes = 40 + 800 * sub["abs_e_overall"].fillna(0)   # bigger dot = larger observed error
    ax.scatter(sub["magnitude"], sub["mismatch"],
               c=colors[grp_name], s=sizes, alpha=0.65, edgecolor="black",
               linewidth=0.5, label=f"{grp_name} (n={len(sub)})")

# diagonal ratio=1 reference
mmax = max(out["magnitude"].max(), out["mismatch"].max()) * 1.05
ax.plot([0, mmax], [0, mmax], color="#888", linestyle="--", linewidth=0.8, label="ratio = 1")

ax.set_xlabel("Roughness MAGNITUDE  =  wm( |(rs_W + rs_M) / 2| )   (formula A input)")
ax.set_ylabel("Roughness MISMATCH   =  wm( |rs_W - rs_M| )         (formula B input)")
ax.set_title("Per-pair roughness structure by group\n(marker size = |observed e_overall|)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG)
plt.close(fig)

print(f"\nWrote: {OUT_CSV}")
print(f"Wrote: {OUT_PNG}")
