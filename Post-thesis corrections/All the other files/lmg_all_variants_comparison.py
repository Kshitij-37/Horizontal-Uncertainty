"""
Full LMG variance-decomposition comparison across all candidate models.

For each variant we compute the proper LMG (Shapley) share of each feature's
contribution to Var(log_sigma). This is the "real" attribution that accounts
for feature correlations — unlike the naive beta^2 attribution.

Reads coefficients from each variant's JSON, rebuilds per-pair standardized
feature values from the corrected input Excel using the appropriate formula
and exclusions, computes the covariance matrix, and reports LMG.

Variants covered:
  - thesis-final              (formula A, z-scored, forest-out, WITH bias)
  - production__forest-out    (formula A, z-scored, forest-out, sigma-only)
  - exp-A__forest-out         (formula A, un-centered, sigma-only)
  - exp-Mc__forest-out        (formula M min-first + centered, sigma-only)
  - exp-M2c__forest-out       (formula M2c per-formula z-score then min, sigma-only)
  - exp-M2c__forest-in        (M2c with forest included)
  - exp-Q__forest-out         (formula A + 500m gate, un-centered, sigma-only)
"""
import os
import json
import math
import itertools
import importlib.machinery
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_RFC = os.path.normpath(os.path.join(HERE, "roughness_formula_comparison.py"))
_loader = importlib.machinery.SourceFileLoader("rfc", _RFC)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _RFC, loader=_loader))
_loader.exec_module(rfc)
pmod = rfc.pmod

REFIT_DIR = os.path.join(HERE, "sigma_only_refit", "Results")
EXPM2C_DIR = os.path.join(HERE, "Results_expM2c")
THESIS_JSON = os.path.normpath(os.path.join(
    HERE, "..", "Bayesian_approach", "Final model", "Results",
    "ws_uncertainty_pairlevel_results.json"))

FOREST_MASTS = ["2011WM011", "2014WM011", "2012WM006"]
BASE_EXCLUDED = list(pmod.EXCLUDED_MASTS)   # includes forest by default
FOREST_OUT = list(BASE_EXCLUDED)
FOREST_IN  = [m for m in BASE_EXCLUDED if m not in FOREST_MASTS]

FEATURE_LABELS = ["distance", "turning", "log_speedup", "dz", "roughness"]


# ─────────────────────────────────────────────────────────────────
# Core LMG computation
# ─────────────────────────────────────────────────────────────────
def compute_lmg(betas, cov_matrix):
    K = len(betas)
    total = float(betas @ cov_matrix @ betas)
    subset_var = {}
    for r in range(K + 1):
        for combo in itertools.combinations(range(K), r):
            idx = list(combo)
            subset_var[frozenset(combo)] = (
                float(betas[idx] @ cov_matrix[np.ix_(idx, idx)] @ betas[idx])
                if idx else 0.0)
    lmg = np.zeros(K)
    for k in range(K):
        for r in range(K):
            weight = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
            for combo in itertools.combinations([j for j in range(K) if j != k], r):
                S = frozenset(combo); Sk = S | {k}
                lmg[k] += weight * (subset_var[Sk] - subset_var[S])
    naive = betas ** 2 * np.diag(cov_matrix)
    return lmg, naive, total


def build_features(formula_key, exclusion_list):
    """Rebuild pair_df with the requested roughness formula and exclusion list.
    Returns pair_df with columns: dist_sat, turning_sat, wm_abs_log_speedup,
    dz_sat, roughness_sat (formula-specific) and additionally rough_sat_A /
    rough_sat_B for M2c formula.
    """
    _saved = list(pmod.EXCLUDED_MASTS)
    try:
        pmod.EXCLUDED_MASTS = exclusion_list
        sector_df = rfc.load_sector_data()
        if formula_key == "M2c":
            # Need both A and B saturated values
            pA, _ = rfc.build_pair_data_with_formula(sector_df, "A")
            pB, _ = rfc.build_pair_data_with_formula(sector_df, "B")
            merged = pA.copy()
            satB_by = pB.set_index("pair_id")["roughness_sat"]
            merged["rough_sat_A"] = merged["roughness_sat"].values
            merged["rough_sat_B"] = merged["pair_id"].map(satB_by).values
            return merged
        pair_df, _ = rfc.build_pair_data_with_formula(sector_df, formula_key)
        return pair_df
    finally:
        pmod.EXCLUDED_MASTS = _saved


def prepare_variant(name, config):
    """Compute LMG for a given variant.
    config keys:
      json_path: str
      formula:   'A', 'M', 'Q', 'M2c', ...  (which roughness formula to build)
      form:      'z_scored', 'un_centered', 'expM2c'
      forest:    'in' or 'out'
    """
    with open(config["json_path"]) as f:
        meta = json.load(f)

    exclusion_list = FOREST_IN if config["forest"] == "in" else FOREST_OUT
    pair_df = build_features(config["formula"], exclusion_list)

    # Compose the beta vector and the standardized feature matrix depending on form
    if config["form"] == "z_scored":
        # Production/thesis: (feature - mean) / std, all 5 features
        mp = meta.get("model_params", meta.get("coeffs", {}))
        sc = meta.get("scalers", {})
        if "gammas" in meta.get("coeffs", {}):
            # sigma_only_refit style
            g = meta["coeffs"]["gammas"]
            betas = np.array([
                g["dist_sat"], g["turning_sat"], g["wm_abs_log_speedup"],
                g["dz_sat"], g["roughness_sat"],
            ])
        else:
            # thesis-final style
            betas = np.array([
                mp["gamma_dist"], mp["gamma_turning"], mp["gamma_speedup"],
                mp["gamma_dz"], mp["gamma_roughness"],
            ])
        cols = ["dist_sat", "turning_sat", "wm_abs_log_speedup", "dz_sat", "roughness_sat"]
        Z_cols = []
        for c in cols:
            m = sc[f"{c}_mean"]; s = sc[f"{c}_std"]
            Z_cols.append((pair_df[c].values - m) / s)
        Z = np.column_stack(Z_cols)

    elif config["form"] == "un_centered":
        # sigma-only refit: divide by std, NO centering
        g = meta["coeffs"]["betas"]
        stds = meta["feature_stds"]
        betas = np.array([
            g["dist_sat"], g["turning_sat"], g["wm_abs_log_speedup"],
            g["dz_sat"], g["roughness_sat"],
        ])
        cols = ["dist_sat", "turning_sat", "wm_abs_log_speedup", "dz_sat", "roughness_sat"]
        Z = np.column_stack([pair_df[c].values / stds[c] for c in cols])

    elif config["form"] == "expM2c":
        # dedicated exp-M2c: 4 non-rough centered + roughness = min(z_A, z_B) with per-formula centering
        mp = meta["model_params"]
        sc = meta["scalers"]
        betas = np.array([
            mp["gamma_dist"], mp["gamma_turning"], mp["gamma_speedup"],
            mp["gamma_dz"], mp["gamma_roughness"],
        ])
        non_rough_cols = ["dist_sat", "turning_sat", "wm_abs_log_speedup", "dz_sat"]
        Z_non_rough = np.column_stack([
            (pair_df[c].values - sc[f"{c}_mean"]) / sc[f"{c}_std"]
            for c in non_rough_cols])
        z_A = (pair_df["rough_sat_A"].values - sc["rough_sat_A_mean"]) / sc["rough_sat_A_std"]
        z_B = (pair_df["rough_sat_B"].values - sc["rough_sat_B_mean"]) / sc["rough_sat_B_std"]
        rough_M2c = np.minimum(z_A, z_B)
        Z = np.column_stack([Z_non_rough, rough_M2c.reshape(-1, 1)])

    elif config["form"] == "sigma_only_refit_M2c":
        # exp-M2c from sigma_only_refit (compressed structure — has "betas" dict + expM2_details)
        g = meta["coeffs"]["gammas"]
        stds = meta["feature_stds"]
        d = meta["expM2_details"]
        betas = np.array([
            g["dist_sat"], g["turning_sat"], g["wm_abs_log_speedup"],
            g["dz_sat"], g["roughness_sat"],
        ])
        non_rough_cols = ["dist_sat", "turning_sat", "wm_abs_log_speedup", "dz_sat"]
        # Refit exp-M2c uses centering on non-rough features via stored feature_means_all
        means = meta.get("feature_means_all", {c: 0.0 for c in non_rough_cols})
        Z_non_rough = np.column_stack([
            (pair_df[c].values - means.get(c, 0.0)) / stds[c]
            for c in non_rough_cols])
        z_A = (pair_df["rough_sat_A"].values - d["mean_A"]) / d["std_A"]
        z_B = (pair_df["rough_sat_B"].values - d["mean_B"]) / d["std_B"]
        rough_M2c = np.minimum(z_A, z_B)
        Z = np.column_stack([Z_non_rough, rough_M2c.reshape(-1, 1)])
    else:
        raise ValueError(f"Unknown form {config['form']}")

    cov = np.cov(Z, rowvar=False, ddof=1)
    lmg, naive, total = compute_lmg(betas, cov)
    return {"name": name, "n": Z.shape[0], "betas": betas,
            "lmg": lmg, "naive": naive, "total": total,
            "naive_sum": float(naive.sum())}


# ─────────────────────────────────────────────────────────────────
# Variant configs
# ─────────────────────────────────────────────────────────────────
VARIANTS = [
    ("thesis-final (formula A, z-scored, with bias)", {
        "json_path": THESIS_JSON,
        "formula": "A",
        "form": "z_scored",
        "forest": "out",
    }),
    ("production__forest-out (formula A, z-scored, sigma-only)", {
        "json_path": os.path.join(REFIT_DIR, "production__forest-out.json"),
        "formula": "A",
        "form": "z_scored",
        "forest": "out",
    }),
    ("exp-A__forest-out (formula A, un-centered, sigma-only)", {
        "json_path": os.path.join(REFIT_DIR, "exp-A__forest-out.json"),
        "formula": "A",
        "form": "un_centered",
        "forest": "out",
    }),
    ("exp-Mc__forest-out (min-first + centered)", {
        "json_path": os.path.join(REFIT_DIR, "exp-Mc__forest-out.json"),
        "formula": "M",   # min-first uses formula-A saturated then min-of-raw
        "form": "z_scored",   # exp-Mc uses production-style z-scoring on the min-based feature
        "forest": "out",
    }),
    ("exp-M2c__forest-out (per-formula z-score then min)", {
        "json_path": os.path.join(EXPM2C_DIR, "ws_uncertainty_expM2c_results.json"),
        "formula": "M2c",
        "form": "expM2c",
        "forest": "out",
    }),
    ("exp-M2c__forest-in (M2c with forest included, from earlier refit)", {
        "json_path": os.path.join(REFIT_DIR, "exp-M2c__forest-in.json"),
        "formula": "M2c",
        "form": "sigma_only_refit_M2c",
        "forest": "in",
    }),
    ("exp-Q__forest-out (magnitude x 500m gate, un-centered)", {
        "json_path": os.path.join(REFIT_DIR, "exp-Q__forest-out.json"),
        "formula": "Q",
        "form": "un_centered",
        "forest": "out",
    }),
]


def main():
    print("=" * 100)
    print("LMG VARIANCE DECOMPOSITION — ALL CANDIDATES (proper attribution accounting for correlations)")
    print("=" * 100)

    results = []
    for name, cfg in VARIANTS:
        try:
            r = prepare_variant(name, cfg)
            results.append(r)
        except Exception as ex:
            print(f"[FAIL] {name}: {ex}")
            continue

    # Print naive %s
    print("\n--- NAIVE decomposition (beta^2 * Var(feature), % of naive-sum) ---")
    print(f"  {'variant':<58}{'n':>4}" + "".join(f"{lab[:9]:>10}" for lab in FEATURE_LABELS))
    print("  " + "-" * 108)
    for r in results:
        pcts = 100 * r["naive"] / r["naive_sum"]
        print(f"  {r['name'][:57]:<58}{r['n']:>4}" + "".join(f"{p:>9.1f}%" for p in pcts))

    # Print LMG %s
    print("\n--- LMG / SHAPLEY decomposition (proper attribution, % of Var(log_sigma)) ---")
    print(f"  {'variant':<58}{'n':>4}" + "".join(f"{lab[:9]:>10}" for lab in FEATURE_LABELS))
    print("  " + "-" * 108)
    for r in results:
        pcts = 100 * r["lmg"] / r["total"]
        print(f"  {r['name'][:57]:<58}{r['n']:>4}" + "".join(f"{p:>9.1f}%" for p in pcts))

    # Rank variants by roughness LMG share (lowest to highest — for the "less-dominant" preference)
    print("\n--- Roughness LMG share, sorted lowest-first (target: LESS dominant) ---")
    rough_idx = FEATURE_LABELS.index("roughness")
    ranked = sorted(results, key=lambda r: r["lmg"][rough_idx] / r["total"])
    print(f"  {'variant':<58}{'roughness LMG %':>18}")
    print("  " + "-" * 76)
    for r in ranked:
        pct = 100 * r["lmg"][rough_idx] / r["total"]
        print(f"  {r['name'][:57]:<58}{pct:>17.1f}%")

    # Rank variants by complexity-features LMG share (dist + turning + speedup + dz)
    print("\n--- 'Complexity' LMG share = dist + turning + log_speedup + dz (higher = more complexity-driven) ---")
    complexity_idx = [FEATURE_LABELS.index(x) for x in ["distance", "turning", "log_speedup", "dz"]]
    ranked = sorted(results, key=lambda r: -sum(r["lmg"][i] / r["total"] for i in complexity_idx))
    print(f"  {'variant':<58}{'complexity LMG %':>19}")
    print("  " + "-" * 77)
    for r in ranked:
        pct = 100 * sum(r["lmg"][i] / r["total"] for i in complexity_idx)
        print(f"  {r['name'][:57]:<58}{pct:>18.1f}%")


if __name__ == "__main__":
    main()
