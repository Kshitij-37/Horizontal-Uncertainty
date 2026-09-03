"""
Per-pair table of Windspeed Uncertainty vs Windspeed Bias, from the most
recent Collected output Excel produced by Timeseries_toggle_with_Samplestatus.py.

Both metrics are already computed there:
  Windspeed uncertainty = std(predicted - measured) / mean(measured)      (scatter)
  Windspeed Bias        = mean(predicted - measured) / mean(measured)     (systematic offset)

We just aggregate and mark which pairs involve masts that are EXCLUDED from the
production Bayesian model, so we can eyeball whether bias/uncertainty patterns
distinguish the problematic ones.
"""
import os, glob
import pandas as pd

FOLDER = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output without graphs"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "Results", "uncertainty_vs_bias_per_pair.csv")

# Excluded masts and the reason group they belong to (from production model)
EXCLUDED = {
    "2015WM018": "Sallachy (complex terrain)",
    "2021PA004": "Sallachy (complex terrain)",
    "2022PA008": "Sallachy (complex terrain)",
    "2022PA018": "Kayislar (only 2 dominant directions)",
    "2011WM011": "Hultema (forest, persistent outlier)",
    "2014WM011": "Hultema (forest, persistent outlier)",
    "2019HE001": "Herzhausen (CFD, not comparable)",
    "2019HE002": "Herzhausen (CFD, not comparable)",
    "2019HE003": "Herzhausen (CFD, not comparable)",
    "2022PA021": "Taaibos (3-mast, dropped for fit)",
    "2023PA085": "Ukhanda (2 mast + LiDAR)",
    "2024PA014": "Balver Wald (uncertain site)",
    "2012WM006": "Malarberget (forest, persistent outlier)",
    "2024PA107": "Slovenska East (missing displacement)",
}

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
files = sorted(glob.glob(os.path.join(FOLDER, "Collected output_*.xlsx")))
if not files:
    raise SystemExit(f"No Collected output_*.xlsx in {FOLDER}")
newest = files[-1]
print(f"Reading: {os.path.basename(newest)}")

df = pd.read_excel(newest)

# Cast the two metric columns to numeric (in case of formatting glitches)
df["Windspeed uncertainty"] = pd.to_numeric(df["Windspeed uncertainty"], errors="coerce")
df["Windspeed Bias"] = pd.to_numeric(df["Windspeed Bias"], errors="coerce")

# Cast mast IDs to str for lookup
df["MM"] = df["Measurement Number"].astype(str)
df["WTG"] = df["Prediction Number"].astype(str)

def flag_pair(mm, wtg):
    """Return which of the two masts (if any) is on the exclusion list, and why."""
    tags = []
    if mm in EXCLUDED:  tags.append(f"MM={mm}:{EXCLUDED[mm]}")
    if wtg in EXCLUDED: tags.append(f"WTG={wtg}:{EXCLUDED[wtg]}")
    return "; ".join(tags) if tags else ""

df["exclusion_flag"] = df.apply(lambda r: flag_pair(r["MM"], r["WTG"]), axis=1)
df["excluded"] = df["exclusion_flag"] != ""

# Sort by uncertainty desc so worst pairs are up top
out = df[["Location", "MM", "WTG", "Windspeed uncertainty", "Windspeed Bias",
          "Distance between measurements in meters", "excluded", "exclusion_flag"]].copy()
out = out.rename(columns={"Distance between measurements in meters": "distance_m"})
out = out.sort_values("Windspeed uncertainty", ascending=False).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

# Console print — compact table
print("\n" + "=" * 108)
print(f"{'Location':<26}{'MM':<12}{'WTG':<12}{'Uncertainty':>13}{'Bias':>11}{'Dist(m)':>10}  Excluded?")
print("=" * 108)
for _, r in out.iterrows():
    tag = "  EXCL" if r["excluded"] else ""
    print(f"{str(r['Location'])[:25]:<26}{r['MM']:<12}{r['WTG']:<12}"
          f"{r['Windspeed uncertainty']:>13.4f}{r['Windspeed Bias']:>+11.4f}"
          f"{r['distance_m']:>10.0f}{tag}")

# Summary stats grouped by inclusion status
print("\n" + "=" * 108)
print("SUMMARY (kept vs excluded pairs)")
print("=" * 108)
for label, sub in [("KEPT (in production model)", out[~out["excluded"]]),
                    ("EXCLUDED", out[out["excluded"]])]:
    if len(sub) == 0:
        continue
    unc = sub["Windspeed uncertainty"]
    bias = sub["Windspeed Bias"]
    print(f"\n  {label}: n={len(sub)}")
    print(f"    Uncertainty  min={unc.min():.4f}  median={unc.median():.4f}  max={unc.max():.4f}")
    print(f"    Bias        min={bias.min():+.4f}  median={bias.median():+.4f}  max={bias.max():+.4f}  mean_abs={bias.abs().mean():.4f}")

# Also: which excluded-reason groups have the highest uncertainty?
if out["excluded"].any():
    print("\n  By exclusion-reason group:")
    excl_rows = out[out["excluded"]].copy()
    def reason(f):
        # take the first tag's reason (after the colon)
        first = f.split(";")[0]
        return first.split(":", 1)[1] if ":" in first else first
    excl_rows["reason"] = excl_rows["exclusion_flag"].apply(reason)
    g = excl_rows.groupby("reason").agg(
        n=("Windspeed uncertainty", "size"),
        unc_med=("Windspeed uncertainty", "median"),
        unc_max=("Windspeed uncertainty", "max"),
        bias_med=("Windspeed Bias", "median"),
        bias_absmax=("Windspeed Bias", lambda s: s.abs().max()),
    ).sort_values("unc_med", ascending=False)
    print(g.to_string())

print(f"\nSaved: {OUT_CSV}")
