"""
Pair-level LOO sensitivity ablation.

For each mast in the dataset (both currently included and excluded),
tests the effect of adding/removing it on pair-level LOO metrics.

Uses fast MCMC settings (500 tune, 500 draw, 2 chains) for speed.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import importlib.util
import importlib.machinery

# ─── IMPORT FINAL MODEL ────────────────────────────────────────

_final_model_path = os.path.join(
    os.path.dirname(__file__),
    "ws_uncertainty_model_pairlevel_final"
)
_loader = importlib.machinery.SourceFileLoader("pmod", _final_model_path)
_spec = importlib.util.spec_from_file_location("pmod", _final_model_path, loader=_loader)
pmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmod)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FAST_DRAWS = 500
FAST_TUNE = 500


# ─── HELPERS ────────────────────────────────────────────────────

def get_all_masts(df):
    """Get all unique masts from pair_ids."""
    masts = set()
    for pid in df["pair_id"].unique():
        a, b = pid.split("__")
        masts.add(a)
        masts.add(b)
    return sorted(masts)


def get_mast_location(df, mast_id):
    """Get location name for a mast."""
    mask = (df["pair_id"].str.startswith(mast_id + "__")) | \
           (df["pair_id"].str.endswith("__" + mast_id))
    subset = df[mask]
    if len(subset) > 0 and "location" in subset.columns:
        return subset["location"].iloc[0]
    return "Unknown"


def run_pair_loo(pair_df):
    """Run pair-level LOO and return sigma-only Pearson and Spearman."""
    results_df, r_pearson, r_spearman, bias = pmod.leave_one_pair_out_cv(
        pair_df, draws=FAST_DRAWS, tune=FAST_TUNE, holdout_by="pair"
    )
    if len(results_df) < 2:
        return np.nan, np.nan, len(results_df)

    r = results_df["predicted_sigma"].corr(results_df["actual_error"])
    rho = results_df["predicted_sigma"].corr(results_df["actual_error"], method="spearman")
    return r, rho, len(results_df)


def build_pair_df(raw_df, exclude_masts):
    """Filter raw data by exclusion list and build pair-level dataframe."""
    df = raw_df.copy()
    df["mast_A"] = df["pair_id"].apply(lambda x: x.split("__")[0])
    df["mast_B"] = df["pair_id"].apply(lambda x: x.split("__")[1])
    mask = (~df["mast_A"].isin(exclude_masts)) & (~df["mast_B"].isin(exclude_masts))
    df = df[mask].copy()

    if df["pair_id"].nunique() < 4:
        return None

    data = pmod.build_pair_training_data(df)
    return data["df"]


# ─── MAIN ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PAIR-LEVEL LOO SENSITIVITY ABLATION")
    print("=" * 70)

    print("\nLoading data...")
    raw_df = pd.read_excel(pmod.INPUT_PATH)
    raw_df = raw_df.drop_duplicates()

    current_excluded = list(pmod.EXCLUDED_MASTS)
    all_masts_in_data = get_all_masts(raw_df)
    included_masts = [m for m in all_masts_in_data if m not in current_excluded]
    excluded_masts = [m for m in all_masts_in_data if m in current_excluded]

    print(f"  Total masts in data: {len(all_masts_in_data)}")
    print(f"  Currently included:  {len(included_masts)}")
    print(f"  Currently excluded:  {len(excluded_masts)}")

    # ─── BASELINE ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE (current exclusion list)")
    print("=" * 70)
    baseline_pair_df = build_pair_df(raw_df, current_excluded)
    n_baseline_pairs = len(baseline_pair_df)
    print(f"  {n_baseline_pairs} directional pairs")

    t0 = time.time()
    baseline_r, baseline_rho, baseline_n = run_pair_loo(baseline_pair_df)
    baseline_time = time.time() - t0
    print(f"\n  BASELINE: Pearson={baseline_r:.4f}  Spearman={baseline_rho:.4f}  "
          f"({baseline_n} predictions, {baseline_time/60:.1f} min)")

    results = []

    # ─── DIRECTION 1: REMOVE each included mast ─────────────────
    print("\n" + "=" * 70)
    print(f"DIRECTION 1: REMOVE each included mast ({len(included_masts)} tests)")
    print("=" * 70)

    for i, mast in enumerate(included_masts):
        loc = get_mast_location(raw_df, mast)
        test_excluded = current_excluded + [mast]
        pair_df = build_pair_df(raw_df, test_excluded)

        if pair_df is None or len(pair_df) < 4:
            print(f"\n  [{i+1}/{len(included_masts)}] SKIP {mast} ({loc}) — too few pairs")
            continue

        n_pairs = len(pair_df)
        print(f"\n  [{i+1}/{len(included_masts)}] REMOVE {mast} ({loc}) — {n_pairs} pairs")

        t0 = time.time()
        r, rho, n = run_pair_loo(pair_df)
        elapsed = time.time() - t0

        delta_r = r - baseline_r if not np.isnan(r) else np.nan
        delta_rho = rho - baseline_rho if not np.isnan(rho) else np.nan

        print(f"    Pearson={r:.4f} (Δ={delta_r:+.4f})  "
              f"Spearman={rho:.4f} (Δ={delta_rho:+.4f})  "
              f"({elapsed/60:.1f} min)")

        results.append({
            "mast_id": mast,
            "location": loc,
            "direction": "remove",
            "n_pairs": n_pairs,
            "n_predictions": n,
            "pearson": r,
            "spearman": rho,
            "delta_pearson": delta_r,
            "delta_spearman": delta_rho,
        })

        # Save intermediate results after each test
        pd.DataFrame(results).to_csv(
            os.path.join(RESULTS_DIR, "ablation_pair_sensitivity.csv"),
            index=False
        )

    # ─── DIRECTION 2: ADD BACK each excluded mast ───────────────
    print("\n" + "=" * 70)
    print(f"DIRECTION 2: ADD BACK each excluded mast ({len(excluded_masts)} tests)")
    print("=" * 70)

    for i, mast in enumerate(excluded_masts):
        loc = get_mast_location(raw_df, mast)
        test_excluded = [m for m in current_excluded if m != mast]
        pair_df = build_pair_df(raw_df, test_excluded)

        if pair_df is None or len(pair_df) < 4:
            print(f"\n  [{i+1}/{len(excluded_masts)}] SKIP {mast} ({loc}) — too few pairs")
            continue

        n_pairs = len(pair_df)
        print(f"\n  [{i+1}/{len(excluded_masts)}] ADD BACK {mast} ({loc}) — {n_pairs} pairs")

        t0 = time.time()
        r, rho, n = run_pair_loo(pair_df)
        elapsed = time.time() - t0

        delta_r = r - baseline_r if not np.isnan(r) else np.nan
        delta_rho = rho - baseline_rho if not np.isnan(rho) else np.nan

        print(f"    Pearson={r:.4f} (Δ={delta_r:+.4f})  "
              f"Spearman={rho:.4f} (Δ={delta_rho:+.4f})  "
              f"({elapsed/60:.1f} min)")

        results.append({
            "mast_id": mast,
            "location": loc,
            "direction": "add_back",
            "n_pairs": n_pairs,
            "n_predictions": n,
            "pearson": r,
            "spearman": rho,
            "delta_pearson": delta_r,
            "delta_spearman": delta_rho,
        })

        pd.DataFrame(results).to_csv(
            os.path.join(RESULTS_DIR, "ablation_pair_sensitivity.csv"),
            index=False
        )

    # ─── SUMMARY ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "ablation_pair_sensitivity.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline: Pearson={baseline_r:.4f}  Spearman={baseline_rho:.4f}")
    print(f"\n  Results saved to: {out_path}")

    if len(results_df) > 0:
        print("\n  Top 5 most impactful (by |Δ Spearman|):")
        results_df["abs_delta_rho"] = results_df["delta_spearman"].abs()
        top = results_df.nlargest(5, "abs_delta_rho")
        for _, row in top.iterrows():
            print(f"    {row['direction']:>8s} {row['mast_id']} ({row['location']}): "
                  f"Δρ={row['delta_spearman']:+.4f}  Δr={row['delta_pearson']:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
