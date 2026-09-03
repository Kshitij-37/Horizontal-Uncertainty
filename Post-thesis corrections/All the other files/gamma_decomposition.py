"""
Gamma decomposition: pairs vs gate
-----------------------------------
v4 (post-thesis) differs from v3 (thesis-final) in TWO confounded ways:
  (1) +2 close-mast pairs added to the data  (38 -> 40 pairs)
  (2) roughness proximity gate               (un-gated -> gated)

This 2x2 fit isolates each effect on the gammas:

                un-gated roughness      gated roughness
   38 pairs        38u  (~Final)            38g
   40 pairs        40u                      40g  (~Post-thesis)

  pairs effect  = 40u - 38u   (holding the gate OFF)
  gate  effect  = 40g - 40u   (holding pairs at 40)

Uses a MAP fit (scipy) with the SAME priors as the Bayesian model
(HalfNormal(0.3) on gammas, Normal(-3.9,0.5) on log_sigma0, Normal(0,0.05)
on beta_dz, Gamma(2,0.2) on nu) so the point estimates approximate the JSON
posterior means. No pymc required.

RUN:  python gamma_decomposition.py
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

INPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

DZ_SAT_SCALE = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE = 3.0
PROXIMITY_GATE_SCALE = 500.0
EXCLUDED_MASTS = [
    "2015WM018", "2021PA004", "2022PA008", "2022PA018",
    "2011WM011", "2014WM011", "2019HE001", "2019HE002", "2019HE003",
    "2022PA021", "2023PA085", "2024PA014", "2012WM006", "2024PA107",
]

# If you KNOW which pair_ids were the added close-mast pairs, list them here.
# Otherwise leave empty and the script drops the 2 smallest-distance pairs as a proxy.
ADDED_PAIR_IDS = []   # e.g. ["2024PAxxx__2024PAyyy", "2024PAyyy__2024PAxxx"]

FEATURE_LABELS = ["dist", "turn", "spd", "rough", "dz"]


def build_pairs(df):
    df = df.drop_duplicates().copy()
    df["A"] = df["pair_id"].str.split("__").str[0]
    df["B"] = df["pair_id"].str.split("__").str[1]
    df = df[(~df["A"].isin(EXCLUDED_MASTS)) & (~df["B"].isin(EXCLUDED_MASTS))].copy()
    for o, n in {
        "d_turning_deg": "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor": "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac": "rough_speedup_MM_frac_new",
    }.items():
        if n in df:
            m = df[n].notna(); df.loc[m, o] = df.loc[m, n]
    mm = (df["overall_speedup_WTG_factor"] > 0) & (df["overall_speedup_MM_factor"] > 0)
    df["alsr"] = np.nan
    df.loc[mm, "alsr"] = np.abs(np.log(df.loc[mm, "overall_speedup_WTG_factor"] /
                                       df.loc[mm, "overall_speedup_MM_factor"]))
    rows = []
    for pid, g in df.groupby("pair_id"):
        wp = pd.to_numeric(g["Sample_count_pred"], errors="coerce").fillna(0).values.astype(float)
        ws = pd.to_numeric(g["Sample_count_self"], errors="coerce").fillna(0).values.astype(float)
        if wp.sum() == 0 or ws.sum() == 0:
            wp = np.ones(len(g)); ws = np.ones(len(g))
        wp /= wp.sum(); ws /= ws.sum()
        WSs = float(np.sum(ws * g["Mean_windspeed_self"].values))
        e = (float(np.sum(wp * g["Mean_windspeed_predicted"].values)) - WSs) / WSs
        dm = float(g["distance_m"].iloc[0]); dA = g["distance_A"].iloc[0]
        if pd.isna(dA) or dA <= 0:
            continue
        wf = pd.to_numeric(g["freq_MM"], errors="coerce").fillna(0).values
        wf = wf / wf.sum() if wf.sum() > 0 else wp
        turn = 1 - np.exp(-float(np.sum(wf * g["d_turning_deg"].abs().values)) / TURN_SAT_SCALE)
        spd = float(np.sum(wf * g["alsr"].fillna(0).values))
        rW = pd.to_numeric(g["rough_speedup_WTG_frac"], errors="coerce").values
        rM = pd.to_numeric(g["rough_speedup_MM_frac"], errors="coerce").values
        ar = np.abs((rW + rM) / 2); v = ~np.isnan(ar)
        if v.sum() == 0:
            continue
        base = float(np.sum(wf[v] * ar[v]) / wf[v].sum())
        rough_u = 1 - np.exp(-base / ROUGH_SAT_SCALE)
        rough_g = 1 - np.exp(-(base * (1 - np.exp(-dm / PROXIMITY_GATE_SCALE))) / ROUGH_SAT_SCALE)
        dz = float(g["dz"].iloc[0])
        rows.append(dict(pid=pid, e=e, dm=dm,
                         dist=1 - np.exp(-dm / dA), turn=turn, spd=spd,
                         ru=rough_u, rg=rough_g, dzs=1 - np.exp(-abs(dz) / DZ_SAT_SCALE), dz=dz))
    return pd.DataFrame(rows)


def map_fit(sub, gated):
    """MAP point estimate of the 5 gammas (priors match the Bayesian model)."""
    rc = "rg" if gated else "ru"
    feats = ["dist", "turn", "spd", rc, "dzs"]
    Z = np.column_stack([(sub[f] - sub[f].mean()) / (sub[f].std() or 1) for f in feats])
    dzz = (sub["dz"] - sub["dz"].mean()) / (sub["dz"].std() or 1)
    e = sub["e"].values

    def neg_log_post(th):
        ln_nu, ls0, bdz = th[0], th[1], th[2]
        gam = th[3:8]; nu = np.exp(ln_nu)
        sig = np.exp(ls0 + Z @ gam); mu = bdz * dzz.values
        ll = stats.t.logpdf(e, df=nu, loc=mu, scale=sig).sum()
        # negative log priors (HalfNormal gammas via bounds>=0; Normal ls0/bdz; Gamma nu)
        pen = (0.5 * np.sum((gam / 0.3) ** 2)
               + 0.5 * ((ls0 + 3.9) / 0.5) ** 2
               + 0.5 * (bdz / 0.05) ** 2
               - ((2 - 1) * ln_nu - 0.2 * nu))
        return -ll + pen

    bounds = [(-2, 5), (-6, -2), (-1, 1)] + [(0, 3)] * 5
    x0 = [np.log(10), -3.6, 0.0, 0.2, 0.2, 0.1, 0.3, 0.1]
    res = optimize.minimize(neg_log_post, x0, bounds=bounds, method="L-BFGS-B")
    return res.x[3:8]


def main():
    print("Loading data...")
    P = build_pairs(pd.read_excel(INPUT_PATH))
    print(f"Pairs: {len(P)}")

    print("\nSmallest-distance pairs (candidates for the added close-mast pairs):")
    print(P.nsmallest(5, "dm")[["pid", "dm"]].to_string(index=False))

    if ADDED_PAIR_IDS:
        drop_idx = P[P["pid"].isin(ADDED_PAIR_IDS)].index
        print(f"\nUsing ADDED_PAIR_IDS ({len(drop_idx)} rows) to form the ~38-pair set.")
    else:
        drop_idx = P.nsmallest(2, "dm").index
        print("\nNo ADDED_PAIR_IDS given -> dropping the 2 smallest-distance pairs as the ~38 proxy.")
    P38 = P.drop(drop_idx)

    hdr = "".join(f"{l:>9}" for l in FEATURE_LABELS)
    print(f"\n2x2 MAP gamma decomposition:{'':6}{hdr}")
    fits = {}
    for name, sub, gated in [("38u (~Final)", P38, False), ("40u", P, False),
                             ("38g", P38, True), ("40g (~PostThesis)", P, True)]:
        g = map_fit(sub, gated); fits[name] = g
        print(f"  {name:<20}" + "".join(f"{x:>9.3f}" for x in g))

    print("\nEffects (MAP):")
    pairs_eff = fits["40u"] - fits["38u (~Final)"]
    gate_eff = fits["40g (~PostThesis)"] - fits["40u"]
    print(f"  pairs (40u-38u):     " + "".join(f"{x:>+9.3f}" for x in pairs_eff))
    print(f"  gate  (40g-40u):     " + "".join(f"{x:>+9.3f}" for x in gate_eff))

    print("\nSanity vs published Bayesian JSONs:")
    print("  Final       JSON:    " + "".join(f"{x:>9.3f}" for x in [0.187, 0.287, 0.130, 0.285, 0.088]))
    print("  PostThesis  JSON:    " + "".join(f"{x:>9.3f}" for x in [0.283, 0.313, 0.134, 0.352, 0.098]))
    print("\nNote: MAP != full-posterior mean, so absolute values differ slightly from the JSONs;"
          " read the DIRECTION and RELATIVE SIZE of the pairs vs gate effects.")


if __name__ == "__main__":
    main()
