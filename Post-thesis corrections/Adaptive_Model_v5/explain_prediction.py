"""
Per-feature contribution breakdown for the Adaptive Model v5 (M2c).

Accepts the calculator's own tab-separated output format directly. Paste one
or more prediction rows (with the header line) into the INPUT_DATA string
below, or point INPUT_FILE at a .tsv/.txt export from the calculator.

For every row, prints how each of the 5 features contributes to log(sigma).
If two or more rows are present, also prints pairwise contribution deltas so
you can see which feature is responsible for differences between predictions.

Required columns from the calculator (tab-separated):
  WTG ID, Reference ID, dist_sat, turning, speedup, rough sat_A, rough sat_B, dz_sat
Other columns (Distance, dz, Hub Height, Exp. Uncertainty, Self-pred, ...) are
tolerated and ignored.
"""

import json
import math
import os
import re
from io import StringIO
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, "Results", "model_results.json")

# -------------------------------------------------------------------
# INPUT - paste the calculator output block here (with header line).
# Rows can be tab-separated OR multi-space-separated. Trailing/leading
# whitespace is ignored. Add as many rows as you like.
# -------------------------------------------------------------------
INPUT_DATA = """
WTG ID	Reference ID	Exp. Uncertainty (%)	Self-pred sigma (%)	Overall z	Hub Height (m)	Distance (km)	dz (m)	dist_sat	turning	speedup	rough sat_A	rough sat_B	dz_sat
WEA 2	S03	3.14	0.45	-0.08	179.0	8.7	24.0	0.654	0.031	0.0082	0.388	0.586	0.451
"""

# Alternative: read from a file exported from the calculator. Set to a path,
# or leave as None to use INPUT_DATA above.
INPUT_FILE = None


# -------------------------------------------------------------------
# Model coefficients + scalers (loaded from Adaptive Model v5 fit)
# -------------------------------------------------------------------
def load_model():
    with open(MODEL_JSON) as f:
        r = json.load(f)
    p = r["model_params"]
    s = r["scalers"]
    return {
        "log_sigma0": p["log_sigma0"],
        "gammas": {
            "dist":    p["gamma_dist"],
            "turning": p["gamma_turning"],
            "speedup": p["gamma_speedup"],
            "dz":      p["gamma_dz"],
            "rough":   p["gamma_roughness"],
        },
        "scalers": {
            "dist_sat":            (s["dist_sat_mean"],           s["dist_sat_std"]),
            "turning_sat":         (s["turning_sat_mean"],        s["turning_sat_std"]),
            "wm_abs_log_speedup":  (s["wm_abs_log_speedup_mean"], s["wm_abs_log_speedup_std"]),
            "dz_sat":              (s["dz_sat_mean"],             s["dz_sat_std"]),
            "rough_sat_A":         (s["rough_sat_A_mean"],        s["rough_sat_A_std"]),
            "rough_sat_B":         (s["rough_sat_B_mean"],        s["rough_sat_B_std"]),
        },
    }


# -------------------------------------------------------------------
# Parser - accepts the calculator's TSV format (or multi-space-separated)
# -------------------------------------------------------------------
COLUMN_ALIASES = {
    "wtg id":                 "wtg_id",
    "reference id":           "ref_id",
    "dist_sat":               "dist_sat",
    "turning":                "turning_sat",
    "turning_sat":            "turning_sat",
    "speedup":                "speedup",
    "wm_abs_log_speedup":     "speedup",
    "rough sat_a":            "rough_sat_A",
    "rough sat a":            "rough_sat_A",
    "rough_sat_a":            "rough_sat_A",
    "rough sat_b":            "rough_sat_B",
    "rough sat b":            "rough_sat_B",
    "rough_sat_b":            "rough_sat_B",
    "dz_sat":                 "dz_sat",
    # extra columns kept for the row label
    "exp. uncertainty (%)":   "exp_unc",
    "self-pred sigma (%)":    "self_pred",
    "overall z":              "overall_z",
    "hub height (m)":         "hub_height",
    "distance (km)":          "distance_km",
    "dz (m)":                 "dz_m",
}


def parse_input(text: str) -> pd.DataFrame:
    """Accept tabs OR runs of 2+ spaces as field separators."""
    text = text.strip("\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    normalised = "\n".join(re.sub(r"\t| {2,}", "\t", ln.strip()) for ln in lines)
    df = pd.read_csv(StringIO(normalised), sep="\t")
    new_cols = {}
    for col in df.columns:
        key = col.strip().lower()
        new_cols[col] = COLUMN_ALIASES.get(key, key)
    df = df.rename(columns=new_cols)
    required = ("dist_sat", "turning_sat", "speedup", "dz_sat",
                "rough_sat_A", "rough_sat_B")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Got: {list(df.columns)}")
    return df


# -------------------------------------------------------------------
# Contribution breakdown for a single pair
# -------------------------------------------------------------------
def explain(row: dict, model: dict, label: str):
    s = model["scalers"]
    g = model["gammas"]

    def z(raw, key):
        m, sd = s[key]
        return (raw - m) / sd

    z_dist    = z(row["dist_sat"],    "dist_sat")
    z_turn    = z(row["turning_sat"], "turning_sat")
    z_speedup = z(row["speedup"],     "wm_abs_log_speedup")
    z_dz      = z(row["dz_sat"],      "dz_sat")
    z_A       = z(row["rough_sat_A"], "rough_sat_A")
    z_B       = z(row["rough_sat_B"], "rough_sat_B")
    z_rough   = min(z_A, z_B)
    which_min = "z_A (magnitude)" if z_A <= z_B else "z_B (mismatch)"

    c = {
        "dist":    g["dist"]    * z_dist,
        "turning": g["turning"] * z_turn,
        "speedup": g["speedup"] * z_speedup,
        "dz":      g["dz"]      * z_dz,
        "rough":   g["rough"]   * z_rough,
    }
    z_by = {"dist": z_dist, "turning": z_turn, "speedup": z_speedup,
            "dz": z_dz, "rough": z_rough}
    raw_by = {"dist": row["dist_sat"], "turning": row["turning_sat"],
              "speedup": row["speedup"], "dz": row["dz_sat"], "rough": None}

    total_contrib = sum(c.values())
    log_sigma = model["log_sigma0"] + total_contrib
    sigma = math.exp(log_sigma)

    abs_sum = sum(abs(v) for v in c.values()) or 1.0

    print(f"\n=== {label} ===")
    print(f"{'feature':<12} {'raw':>10} {'z':>10} {'gamma':>8} "
          f"{'contrib':>10}  {'|c|_share':>10}  effect")
    for k in ("dist", "turning", "speedup", "dz", "rough"):
        raw_disp = f"{raw_by[k]:>10.4f}" if raw_by[k] is not None else " " * 10
        effect = "pushes UP" if c[k] > 0 else ("pulls DOWN" if c[k] < 0 else "neutral")
        print(f"{k:<12} {raw_disp} {z_by[k]:>+10.3f} {g[k]:>8.3f} "
              f"{c[k]:>+10.4f}  {abs(c[k])/abs_sum*100:>9.1f}%  {effect}")

    print(f"  rough breakdown: z_A={z_A:+.3f} (rough_sat_A={row['rough_sat_A']:.4f})   "
          f"z_B={z_B:+.3f} (rough_sat_B={row['rough_sat_B']:.4f})   min picks: {which_min}")
    print(f"  log_sigma0={model['log_sigma0']:+.4f}   "
          f"sum_contrib={total_contrib:+.4f}   log_sigma={log_sigma:+.4f}")
    print(f"  ==> sigma = {sigma*100:.2f}%")

    return c, sigma


def compare(res1, res2, label1, label2):
    c1, s1 = res1
    c2, s2 = res2
    print(f"\n=== Difference: [{label2}] minus [{label1}] ===")
    print(f"{'feature':<12} {'contrib_1':>10} {'contrib_2':>10} {'delta':>10}   direction")
    total_delta = 0.0
    diffs = []
    for k in ("dist", "turning", "speedup", "dz", "rough"):
        d = c2[k] - c1[k]
        total_delta += d
        arrow = "raises pair-2 sigma" if d > 0 else \
                ("lowers pair-2 sigma" if d < 0 else "no effect")
        diffs.append((k, d))
        print(f"{k:<12} {c1[k]:>+10.4f} {c2[k]:>+10.4f} {d:>+10.4f}   {arrow}")
    print(f"  total delta log(sigma) = {total_delta:+.4f}  "
          f"->  sigma {s1*100:.2f}% -> {s2*100:.2f}%")

    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  ranked by |delta contribution|:")
    for k, d in diffs:
        print(f"    {k:<10} delta = {d:+.4f}")


def label_of(row: dict) -> str:
    wtg  = str(row.get("wtg_id", "")).strip()
    ref  = str(row.get("ref_id", "")).strip()
    dist = row.get("distance_km", "")
    dz   = row.get("dz_m", "")
    bits = []
    if wtg and ref:
        bits.append(f"{wtg} <- {ref}")
    elif wtg:
        bits.append(wtg)
    if dist != "" and pd.notna(dist):
        bits.append(f"{dist} km")
    if dz != "" and pd.notna(dz):
        bits.append(f"dz={dz} m")
    return "  |  ".join(bits) if bits else "pair"


if __name__ == "__main__":
    model = load_model()

    if INPUT_FILE:
        with open(INPUT_FILE) as f:
            text = f.read()
    else:
        text = INPUT_DATA

    df = parse_input(text)
    print(f"Parsed {len(df)} row(s) from input.")

    results = []
    labels = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        lbl = label_of(rd)
        res = explain(rd, model, lbl)
        results.append(res)
        labels.append(lbl)

    if len(results) >= 2:
        print("\n" + "=" * 78)
        print("PAIRWISE COMPARISONS")
        print("=" * 78)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                compare(results[i], results[j], labels[i], labels[j])
