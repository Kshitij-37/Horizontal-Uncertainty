"""
Roughness-dominance check: does the A-vs-B roughness formula choice actually
matter for practical accuracy?

Method:
  1. Load per-pair feature contributions from Adaptive M2c (formula A / adaptive min).
  2. Compute each pair's roughness share of TOTAL positive log(sigma) contribution.
  3. Rank pairs by roughness dominance.
  4. Compare adaptive-M2c predictions vs pure-mismatch exp-B predictions on:
     - the top-quartile "roughness-dominated" pairs
     - all pairs
     - complexity-dominated pairs (bottom half)
  5. Print per-pair and aggregate metrics so we can decide empirically.

Answers the question: is formula A doing physically-meaningful work, or would
pure mismatch (B) give equivalent or better accuracy on the pairs where
roughness formula choice actually moves the prediction?
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, ".."))

FEAT_CSV = os.path.join(HERE, "Results", "feature_contributions_per_pair.csv")
ADAPT_LOO_CSV = os.path.join(HERE, "Results", "loo_pair_predictions.csv")
EXPB_LOO_CSV = os.path.join(
    BASE, "sigma_only_refit", "Results", "exp-B__forest-in_loo.csv"
)

OUT_CSV = os.path.join(HERE, "Results", "roughness_dominance_comparison.csv")


def main():
    feat = pd.read_csv(FEAT_CSV)
    adapt = pd.read_csv(ADAPT_LOO_CSV)
    expb = pd.read_csv(EXPB_LOO_CSV).rename(
        columns={"predicted_sigma": "expB_predicted_sigma"}
    )

    df = feat.merge(
        adapt[["pair_id", "predicted_sigma", "actual_error"]].rename(
            columns={"predicted_sigma": "adapt_predicted_sigma"}
        ),
        on="pair_id",
    )
    df = df.merge(expb[["pair_id", "expB_predicted_sigma"]], on="pair_id", how="inner")

    print(f"Pairs in both models (inner join): {len(df)}")
    only_adapt = set(adapt["pair_id"]) - set(expb["pair_id"])
    if only_adapt:
        print(f"  (dropped {len(only_adapt)} pairs not in exp-B: {sorted(only_adapt)})")

    # Roughness dominance: |roughness contribution| / sum(|all feature contributions|)
    # Use absolute values because negative contributions still shape the prediction.
    contrib_cols = [
        "contrib_gamma_dist",
        "contrib_gamma_turning",
        "contrib_gamma_speedup",
        "contrib_gamma_dz",
        "contrib_gamma_roughness",
    ]
    df["abs_total_contrib"] = df[contrib_cols].abs().sum(axis=1)
    df["rough_share_abs"] = df["contrib_gamma_roughness"].abs() / df["abs_total_contrib"]

    # Prediction errors (percentage points)
    df["adapt_err_pp"] = (df["adapt_predicted_sigma"] - df["actual_error"]) * 100
    df["expB_err_pp"] = (df["expB_predicted_sigma"] - df["actual_error"]) * 100
    df["diff_pp"] = (df["adapt_predicted_sigma"] - df["expB_predicted_sigma"]) * 100

    df_sorted = df.sort_values("rough_share_abs", ascending=False).reset_index(drop=True)

    n = len(df_sorted)
    q1 = n // 4

    print("\n" + "=" * 100)
    print("TOP QUARTILE — roughness-dominated pairs (roughness has largest share of |log(sigma)|)")
    print("=" * 100)
    print(
        f"  {'pair_id':<28} {'loc':<20} {'r_share':>7}  "
        f"{'actual':>7} {'adapt':>7} {'expB':>7}  {'a_err':>6} {'b_err':>6}  min_pick"
    )
    for _, r in df_sorted.head(q1).iterrows():
        print(
            f"  {r['pair_id']:<28} {str(r['location'])[:20]:<20} "
            f"{r['rough_share_abs']*100:>6.1f}%  "
            f"{r['actual_error']*100:>6.2f}% "
            f"{r['adapt_predicted_sigma']*100:>6.2f}% "
            f"{r['expB_predicted_sigma']*100:>6.2f}%  "
            f"{r['adapt_err_pp']:>+5.2f} {r['expB_err_pp']:>+5.2f}  "
            f"{r['which_min_wins']}"
        )

    print("\n  aggregate on top-quartile (n=%d):" % q1)
    tq = df_sorted.head(q1)
    print(f"    adapt : mean_bias={tq['adapt_err_pp'].mean():+.2f}pp  "
          f"mean|err|={tq['adapt_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={tq['adapt_err_pp'].abs().max():.2f}pp")
    print(f"    expB  : mean_bias={tq['expB_err_pp'].mean():+.2f}pp  "
          f"mean|err|={tq['expB_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={tq['expB_err_pp'].abs().max():.2f}pp")
    n_adapt_win = int((tq["adapt_err_pp"].abs() < tq["expB_err_pp"].abs()).sum())
    n_expb_win = int((tq["expB_err_pp"].abs() < tq["adapt_err_pp"].abs()).sum())
    print(f"    per-pair wins on top quartile: adapt={n_adapt_win}  expB={n_expb_win}  tie={q1 - n_adapt_win - n_expb_win}")

    print("\n" + "=" * 100)
    print("BOTTOM HALF — complexity-dominated pairs (roughness has small share)")
    print("=" * 100)
    bh_start = n - n // 2
    for _, r in df_sorted.iloc[bh_start:].iterrows():
        print(
            f"  {r['pair_id']:<28} {str(r['location'])[:20]:<20} "
            f"{r['rough_share_abs']*100:>6.1f}%  "
            f"{r['actual_error']*100:>6.2f}% "
            f"{r['adapt_predicted_sigma']*100:>6.2f}% "
            f"{r['expB_predicted_sigma']*100:>6.2f}%  "
            f"{r['adapt_err_pp']:>+5.2f} {r['expB_err_pp']:>+5.2f}  "
            f"{r['which_min_wins']}"
        )
    bh = df_sorted.iloc[bh_start:]
    print("\n  aggregate on bottom half (n=%d):" % len(bh))
    print(f"    adapt : mean_bias={bh['adapt_err_pp'].mean():+.2f}pp  "
          f"mean|err|={bh['adapt_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={bh['adapt_err_pp'].abs().max():.2f}pp")
    print(f"    expB  : mean_bias={bh['expB_err_pp'].mean():+.2f}pp  "
          f"mean|err|={bh['expB_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={bh['expB_err_pp'].abs().max():.2f}pp")

    print("\n" + "=" * 100)
    print("ALL PAIRS (both in adapt-M2c and exp-B, n=%d)" % n)
    print("=" * 100)
    print(f"  adapt : mean_bias={df['adapt_err_pp'].mean():+.2f}pp  "
          f"mean|err|={df['adapt_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={df['adapt_err_pp'].abs().max():.2f}pp")
    print(f"  expB  : mean_bias={df['expB_err_pp'].mean():+.2f}pp  "
          f"mean|err|={df['expB_err_pp'].abs().mean():.2f}pp  "
          f"max|err|={df['expB_err_pp'].abs().max():.2f}pp")

    from scipy.stats import pearsonr, spearmanr
    p_adapt = pearsonr(df["actual_error"], df["adapt_predicted_sigma"])[0]
    p_expb = pearsonr(df["actual_error"], df["expB_predicted_sigma"])[0]
    s_adapt = spearmanr(df["actual_error"], df["adapt_predicted_sigma"])[0]
    s_expb = spearmanr(df["actual_error"], df["expB_predicted_sigma"])[0]
    print(f"\n  Pearson  actual vs adapt: {p_adapt:.4f}   actual vs expB: {p_expb:.4f}")
    print(f"  Spearman actual vs adapt: {s_adapt:.4f}   actual vs expB: {s_expb:.4f}")

    n_adapt = int((df["adapt_err_pp"].abs() < df["expB_err_pp"].abs()).sum())
    n_expb = int((df["expB_err_pp"].abs() < df["adapt_err_pp"].abs()).sum())
    print(f"  per-pair wins on all pairs: adapt={n_adapt}  expB={n_expb}  tie={n - n_adapt - n_expb}")

    # save the merged dataframe for the user to inspect
    out = df_sorted[
        ["pair_id", "location", "rough_share_abs", "which_min_wins",
         "actual_error", "adapt_predicted_sigma", "expB_predicted_sigma",
         "adapt_err_pp", "expB_err_pp", "diff_pp",
         "contrib_gamma_roughness", "z_A", "z_B", "x_rough",
         "abs_total_contrib"]
    ]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-pair comparison: {OUT_CSV}")


if __name__ == "__main__":
    main()
