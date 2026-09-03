"""
Ablation: 4-feature model WITHOUT roughness_sat.

Fits the full model once (no LOO) and prints variance decomposition,
so you can compare against the 5-feature model's decomposition.

Run with pymc-env:
  C:\\Users\\K_Trivedi\\AppData\\Local\\anaconda3\\envs\\pymc-env\\python.exe ablation_no_roughness.py
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import importlib.util
import importlib.machinery

_final_model_path = os.path.join(os.path.dirname(__file__), "ws_uncertainty_model_pairlevel_final")
_loader = importlib.machinery.SourceFileLoader("pmod", _final_model_path)
_spec = importlib.util.spec_from_file_location("pmod", _final_model_path, loader=_loader)
pmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmod)

FEATURE_CONFIG_4 = [
    ("gamma_dist",    "dist_sat",            "Saturating distance (1-exp(-d/dA))"),
    ("gamma_turning", "turning_sat",         "Saturating |turning| (1-exp(-t/s))"),
    ("gamma_speedup", "wm_abs_log_speedup",  "WM |log speedup ratio|"),
    ("gamma_dz",      "dz_sat",              "Saturating |dz| (1-exp(-|dz|/40))"),
]


def main():
    print("Loading data...")
    df = pd.read_excel(pmod.INPUT_PATH)
    df = df.drop_duplicates()

    upgrade_map = {
        "d_turning_deg": "d_turning_deg_new",
        "overall_speedup_WTG_factor": "overall_speedup_WTG_factor_new",
        "overall_speedup_MM_factor": "overall_speedup_MM_factor_new",
        "rough_speedup_WTG_frac": "rough_speedup_WTG_frac_new",
        "rough_speedup_MM_frac": "rough_speedup_MM_frac_new",
    }
    for old_col, new_col in upgrade_map.items():
        if new_col in df.columns:
            mask = df[new_col].notna()
            if mask.sum() > 0:
                df.loc[mask, old_col] = df.loc[mask, new_col]

    df["mast_A"] = df["pair_id"].apply(lambda x: x.split("__")[0])
    df["mast_B"] = df["pair_id"].apply(lambda x: x.split("__")[1])
    mask = (~df["mast_A"].isin(pmod.EXCLUDED_MASTS)) & (~df["mast_B"].isin(pmod.EXCLUDED_MASTS))
    df = df[mask].copy()

    # Build pair data using the full model's function
    data = pmod.build_pair_training_data(df)
    pair_df = data["df"]

    # Re-standardise for 4-feature subset
    scalers = {}
    for _, raw_col, _ in FEATURE_CONFIG_4:
        mean = pair_df[raw_col].mean()
        std = pair_df[raw_col].std()
        std = std if std > 0 else 1.0
        pair_df[f"{raw_col}_z"] = (pair_df[raw_col] - mean) / std
        scalers[f"{raw_col}_mean"] = mean
        scalers[f"{raw_col}_std"] = std

    dz_mean = pair_df["dz"].mean()
    dz_std = pair_df["dz"].std()
    pair_df["dz_z"] = (pair_df["dz"] - dz_mean) / dz_std

    print(f"\n{'='*70}")
    print("4-FEATURE MODEL (no roughness) — single fit")
    print(f"{'='*70}")
    print(f"  {len(pair_df)} directional pairs")

    with pm.Model():
        e_data = pm.Data("e", pair_df["e_overall"].values)
        dist_sat_z = pm.Data("dist_sat_z", pair_df["dist_sat_z"].values)
        turning_z = pm.Data("turning_z", pair_df["turning_sat_z"].values)
        speedup_z = pm.Data("speedup_z", pair_df["wm_abs_log_speedup_z"].values)
        dz_sat_z = pm.Data("dz_sat_z", pair_df["dz_sat_z"].values)
        dz_z = pm.Data("dz_z", pair_df["dz_z"].values)

        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=-3.9, sigma=0.5)

        gamma_dist = pm.HalfNormal("gamma_dist", sigma=0.3)
        gamma_turning = pm.HalfNormal("gamma_turning", sigma=0.3)
        gamma_speedup = pm.HalfNormal("gamma_speedup", sigma=0.3)
        gamma_dz = pm.HalfNormal("gamma_dz", sigma=0.3)

        beta_dz = pm.Normal("beta_dz", mu=0, sigma=0.05)

        log_sigma = (
            log_sigma0
            + gamma_dist * dist_sat_z
            + gamma_turning * turning_z
            + gamma_speedup * speedup_z
            + gamma_dz * dz_sat_z
        )
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        mu = pm.Deterministic("mu", beta_dz * dz_z)

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=e_data)

        print("\n  Sampling...")
        idata = pm.sample(
            draws=2000, tune=2000,
            target_accept=0.95,
            chains=4, cores=4,
            random_seed=42,
        )

    # Variance decomposition
    print(f"\n{'='*70}")
    print("VARIANCE DECOMPOSITION (4 features, no roughness)")
    print(f"{'='*70}")

    gamma_vars = {}
    for gn, _, display in FEATURE_CONFIG_4:
        g_mean = idata.posterior[gn].values.flatten().mean()
        gamma_vars[display] = g_mean ** 2

    baseline_var = idata.posterior["log_sigma0"].values.flatten().var()
    total_var = baseline_var + sum(gamma_vars.values())

    print(f"\n  {'Feature':<45} {'Var contrib':>10} {'% of total':>10}")
    print("  " + "-" * 67)
    for display, v in sorted(gamma_vars.items(), key=lambda x: -x[1]):
        print(f"  {display:<45} {v:>10.5f} {v / total_var * 100:>9.1f}%")
    print("  " + "-" * 67)
    print(f"  {'Baseline (log_sigma0 var)':<45} {baseline_var:>10.5f} {baseline_var / total_var * 100:>9.1f}%")
    print(f"  {'TOTAL':<45} {total_var:>10.5f} {'100.0%':>10}")

    feat_pct = sum(gamma_vars.values()) / total_var * 100
    print(f"\n  Features explain {feat_pct:.1f}% of log(sigma) variance")

    # Posterior summary
    print(f"\n{'='*70}")
    print("POSTERIOR SUMMARY")
    print(f"{'='*70}")

    nu_s = idata.posterior["nu"].values.flatten()
    print(f"  nu = {nu_s.mean():.2f} +/- {nu_s.std():.2f}")

    ls0 = idata.posterior["log_sigma0"].values.flatten()
    print(f"  sigma0 = {np.exp(ls0).mean()*100:.2f}%")

    print(f"\n  {'Feature':<45} {'gamma mean':>10} {'std':>8} {'Multiplier':>12}")
    print("  " + "-" * 77)
    for gn, _, display in FEATURE_CONFIG_4:
        g = idata.posterior[gn].values.flatten()
        print(f"  {display:<45} {g.mean():>10.4f} {g.std():>8.4f} {np.exp(g.mean()):>12.3f}x")

    b = idata.posterior["beta_dz"].values.flatten()
    print(f"\n  beta_dz = {b.mean():+.4f} +/- {b.std():.4f}")

    # Reference: 5-feature model
    print(f"\n{'='*70}")
    print("COMPARE: 5-feature model (from results JSON)")
    print(f"{'='*70}")
    print("  Turning:    34.6%")
    print("  Roughness:  33.9%")
    print("  Distance:   14.6%")
    print("  Speedup:     7.1%")
    print("  dz:          3.3%")
    print("  Baseline:    6.6%")

    print("\nDone.")


if __name__ == "__main__":
    main()
