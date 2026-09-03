"""
Leave-one-project-out ablation for the three excluded outliers.

Tests each project's MARGINAL impact rather than scanning all combos (which would
cherry-pick). For each config it runs pair-level LOO under feature A (magnitude,
current) and E (magnitude + mismatch), reporting Pearson / Spearman / bias.

DISCIPLINE: decide inclusion on PHYSICAL grounds, not by picking the max-Pearson
config. Kayislar has a documented reason (energy concentrated in 2 sectors); this
ablation confirms its influence and checks whether the other two fit cleanly under E.

Run in pymc-env (Run button OK — no arguments needed). ~45-60 min (10 LOO fits).
"""
import os, importlib.machinery, importlib.util
import numpy as np, pandas as pd

# import the existing roughness harness (build + fit + LOO)
_HERE = os.path.dirname(os.path.abspath(__file__))
_cmp = os.path.join(_HERE, "roughness_formula_comparison.py")
_loader = importlib.machinery.SourceFileLoader("rfc", _cmp)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _cmp, loader=_loader))
_loader.exec_module(rfc)

# ── the three excluded outlier projects ──
OUTLIER_PROJECTS = {
    "Kayislar":    ["2022PA018"],
    "Hultema":     ["2011WM011", "2014WM011"],
    "Malarberget": ["2012WM006"],
}
# Full production exclusion list (ALL outliers excluded = the clean baseline):
BASE_EXCLUDED = [
    "2015WM018", "2021PA004", "2022PA008", "2022PA018", "2011WM011", "2014WM011",
    "2019HE001", "2019HE002", "2019HE003", "2022PA021", "2023PA085", "2024PA014",
    "2012WM006", "2024PA107",
]
ALL_OUTLIER_MASTS = [m for ms in OUTLIER_PROJECTS.values() for m in ms]

FEATURES = [("A", "magnitude"), ("E", "mag+mismatch")]

# which projects are INCLUDED in each config (rest stay excluded)
CONFIGS = [
    ("clean (all 3 out)",           []),
    ("all 3 in",                    ["Kayislar", "Hultema", "Malarberget"]),
    ("-Kayislar (Hult+Mal in)",     ["Hultema", "Malarberget"]),
    ("-Hultema (Kay+Mal in)",       ["Kayislar", "Malarberget"]),
    ("-Malarberget (Kay+Hult in)",  ["Kayislar", "Hultema"]),
]


def excluded_for(include_projects):
    keep_in = {m for p in include_projects for m in OUTLIER_PROJECTS[p]}
    return [m for m in BASE_EXCLUDED if m not in keep_in]


def main():
    print("=" * 78)
    print("LEAVE-ONE-PROJECT-OUT ABLATION  (features A=magnitude, E=mag+mismatch)")
    print("=" * 78)
    summary = []
    allin_perpair = {}
    for label, include in CONFIGS:
        rfc.pmod.EXCLUDED_MASTS = excluded_for(include)
        sector_df = rfc.load_sector_data()
        for key, desc in FEATURES:
            pair_df, fc = rfc.build_pair_data_with_formula(sector_df, key)
            res_df, r, rho, bias = rfc.run_loo_cv_custom(pair_df, fc)
            n = len(pair_df)
            print(f"  {label:<28} {key} ({desc:<12}) n={n:2d}  "
                  f"Pearson={r:.4f}  Spearman={rho:.4f}  bias={bias:+.4f}")
            summary.append(dict(config=label, feature=key, n=n, pearson=r, spearman=rho, bias=bias))
            if label == "all 3 in":
                allin_perpair[key] = res_df

    # ── summary table ──
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    S = pd.DataFrame(summary)
    for key, _ in FEATURES:
        sub = S[S["feature"] == key]
        print(f"\n  Feature {key}:")
        print(f"    {'config':<30}{'n':>4}{'Pearson':>10}{'Spearman':>10}{'bias':>9}")
        for _, r in sub.iterrows():
            print(f"    {r['config']:<30}{r['n']:>4}{r['pearson']:>10.4f}{r['spearman']:>10.4f}{r['bias']:>+9.4f}")
    S.to_csv(os.path.join(_HERE, "Results", "project_ablation.csv"), index=False)
    print(f"\n  Saved: {os.path.join(_HERE, 'Results', 'project_ablation.csv')}")

    # ── which outlier pair is worst? (all-in config, per-pair predicted vs actual) ──
    print("\n" + "=" * 78)
    print("WHICH OUTLIER PAIR IS WORST (all-3-in config, predicted sigma vs actual |e|)")
    print("=" * 78)
    for key in allin_perpair:
        d = allin_perpair[key].copy()
        d["outlier"] = d["pair_id"].apply(lambda p: any(m in p for m in ALL_OUTLIER_MASTS))
        d = d[d["outlier"]]
        print(f"\n  Feature {key}:")
        print(f"    {'pair_id':<24}{'pred_sigma':>11}{'actual|e|':>11}{'over-pred':>11}")
        for _, r in d.iterrows():
            over = (r["predicted_sigma"] - r["actual_error"]) * 100
            print(f"    {r['pair_id']:<24}{r['predicted_sigma']*100:>10.2f}%{r['actual_error']*100:>10.2f}%{over:>+10.2f}pp")

    print("\nRead: the project whose removal recovers the fit MOST (and whose pairs are the")
    print("biggest over-predictors above) is the influential one. Decide on PHYSICS, not just")
    print("the max-Pearson config. Use Spearman + per-pair over-prediction, not Pearson alone.")


if __name__ == "__main__":
    main()
