"""
Additive-floor model + roughness-variant test  (UN-SATURATED features).

Model (per fit):
    e ~ StudentT(nu, mu, sigma)
    mu    = beta_dz * dz                              (bias, RAW dz per-metre)
    sigma = sigma_floor + sum_k beta_k * (f_k / std_k)

f_k are the RAW UN-SATURATED quantities (saturation removed so sigma can rise linearly/unbounded and the
ceiling isn't capped). Divided by std, NO centering -> 0-at-same-site preserved (clean floor), betas
comparable. NOT z-scored (centering would break the floor).

Un-saturated features:
  distance  = -log(1 - dist_sat)   (= d/dA, recovered from the saturated col)
  turning   = wm_abs_turning        speedup = wm_abs_log_speedup
  roughness = wm_abs_roughness[variant]   (+ wm_rough_diff for E)     dz = abs_dz

Compares the current MULTIPLICATIVE model (unchanged) vs the additive floor with roughness variants
A(magnitude)/B(mismatch)/Q(gated)/E(both).

RUN IN pymc-env.  ~30-60 min (5 configs x pair-level LOO).
"""
import os, sys, importlib.machinery, importlib.util, contextlib
import numpy as np, pandas as pd, pymc as pm

_HERE = os.path.dirname(os.path.abspath(__file__))
_RFC = os.path.normpath(os.path.join(_HERE, "..", "roughness_formula_comparison.py"))
_loader = importlib.machinery.SourceFileLoader("rfc", _RFC)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _RFC, loader=_loader))
_loader.exec_module(rfc)

RESULTS_DIR = os.path.join(_HERE, "Results"); os.makedirs(RESULTS_DIR, exist_ok=True)
DRAWS, TUNE, CHAINS, CORES = 500, 500, 2, 1
ROUGH_MAG_RAW = 0.017        # representative RAW wm_abs_roughness magnitude at a rough same-site

# rough_map = the RAW same-site value of the roughness feature (magnitude persists; mismatch/gated vanish).
# NOTE: the multiplicative baseline uses SATURATED cols, so its map uses the saturated 0.83.
CONFIGS = [
    ("baseline-mult", "multiplicative", "A", {"roughness_sat": 0.83}),
    ("add-A (magnitude)", "additive", "A", {"roughness_sat": ROUGH_MAG_RAW}),
    ("add-B (mismatch)",  "additive", "B", {"roughness_sat": 0.0}),
    ("add-Q (gated mag)", "additive", "Q", {"roughness_sat": 0.0}),
    ("add-E (both)",      "additive", "E", {"roughness_sat": ROUGH_MAG_RAW, "rough_diff_sat": 0.0}),
]

# map each feature_config raw_col -> the stored RAW (un-saturated) column
_RAW_OF = {"turning_sat": "wm_abs_turning", "wm_abs_log_speedup": "wm_abs_log_speedup",
           "roughness_sat": "wm_abs_roughness", "rough_diff_sat": "wm_rough_diff", "dz_sat": "abs_dz"}


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        o, e = sys.stdout, sys.stderr
        try: sys.stdout = dn; sys.stderr = dn; yield
        finally: sys.stdout = o; sys.stderr = e


def unsat(df, raw_col):
    """RAW un-saturated series for a feature_config raw_col."""
    if raw_col == "dist_sat":
        return -np.log(1.0 - np.clip(df["dist_sat"].values.astype(float), None, 1 - 1e-9))
    return df[_RAW_OF[raw_col]].values.astype(float)


def _U(pair_df, feats):
    return np.column_stack([unsat(pair_df, rc) for rc in feats])   # (n, k) un-saturated raw


# ── additive model ────────────────────────────────────────────────────────────────
def fit_additive(pair_df, feature_config, draws=DRAWS, tune=TUNE):
    feats = [rc for _, rc, _ in feature_config]
    U = _U(pair_df, feats)
    stds = U.std(0); stds[stds == 0] = 1.0            # scale only (no centering) -> 0 stays 0
    Xs = U / stds
    dz = pair_df["dz"].values.astype(float)
    e = pair_df["e_overall"].values.astype(float)
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        sigma_floor = pm.HalfNormal("sigma_floor", sigma=0.01)
        betas = pm.HalfNormal("betas", sigma=0.1, shape=len(feats))
        beta_dz = pm.Normal("beta_dz", mu=0, sigma=1e-3)
        sigma = sigma_floor + pm.math.dot(Xs, betas)
        mu = beta_dz * dz
        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=e)
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    return idata, feats, stds


def _means(idata, k):
    return (float(idata.posterior["sigma_floor"].values.mean()),
            idata.posterior["betas"].values.reshape(-1, k).mean(0))


def run_loo_additive(pair_df, feature_config):
    pair_df = pair_df.copy()
    pair_df["_grp"] = pair_df["pair_id"].apply(lambda x: "__".join(sorted(x.split("__"))))
    folds = pair_df.groupby("_grp")["pair_id"].unique().to_dict()
    rows = []
    for held in folds.values():
        train = pair_df[~pair_df["pair_id"].isin(held)]
        test = pair_df[pair_df["pair_id"].isin(held)]
        if len(train) < 5:
            continue
        try:
            with _quiet():
                idata, feats, stds = fit_additive(train, feature_config)
        except Exception as ex:
            print(f"   fold failed: {ex}"); continue
        sf, bt = _means(idata, len(feats))
        Xt = _U(test, feats) / stds
        for i in range(len(test)):
            rows.append({"predicted_sigma": sf + float(np.dot(Xt[i], bt)),
                         "actual_error": abs(test.iloc[i]["e_overall"])})
    d = pd.DataFrame(rows)
    if len(d) < 2:
        return d, np.nan, np.nan, np.nan
    return (d, d["predicted_sigma"].corr(d["actual_error"]),
            d["predicted_sigma"].corr(d["actual_error"], method="spearman"),
            (d["predicted_sigma"] - d["actual_error"]).mean())


def additive_floor_report(pair_df, feature_config, rough_map):
    with _quiet():
        idata, feats, stds = fit_additive(pair_df, feature_config)
    sf, bt = _means(idata, len(feats))
    flat = sf                                                            # all raw = 0
    rough = sf + sum(bt[i] * (rough_map.get(f, 0.0) / stds[i]) for i, f in enumerate(feats))
    mx = sf + float(np.dot(_U(pair_df, feats).max(0) / stds, bt))        # over max observed features
    return sf, flat, rough, mx


# ── multiplicative baseline (reuses rfc; SATURATED + z-scored) ──────────────────────
def mult_floor_report(pair_df, feature_config, rough_map):
    data = rfc.rebuild_data_dict_custom(pair_df, feature_config)
    with _quiet():
        _, idata, _ = rfc.fit_model_custom(data, feature_config)
    sc = data["scalers"]; ls0 = float(idata.posterior["log_sigma0"].values.mean())
    gam = {gn: float(idata.posterior[gn].values.mean()) for gn, _, _ in feature_config}
    def sig(raw_vals):
        ls = ls0
        for gn, rc, _ in feature_config:
            ls += gam[gn] * ((raw_vals.get(rc, 0.0) - sc[f"{rc}_mean"]) / sc[f"{rc}_std"])
        return np.exp(ls)
    return np.nan, sig({}), sig(rough_map), sig({rc: pair_df[rc].max() for _, rc, _ in feature_config})


# ── MAIN ───────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78 + "\nADDITIVE-FLOOR (UN-SATURATED) + ROUGHNESS-VARIANT TEST\n" + "=" * 78)
    sector_df = rfc.load_sector_data()
    out = []
    for label, mtype, rkey, rough_map in CONFIGS:
        print(f"[{label}] model={mtype} roughness={rkey} ...")
        pair_df, fc = rfc.build_pair_data_with_formula(sector_df, rkey)
        n = len(pair_df)
        if mtype == "additive":
            _, r, rho, bias = run_loo_additive(pair_df, fc)
            sf, flat, rough, mx = additive_floor_report(pair_df, fc, rough_map)
        else:
            _, r, rho, bias = rfc.run_loo_cv_custom(pair_df, fc)
            sf, flat, rough, mx = mult_floor_report(pair_df, fc, rough_map)
        out.append(dict(config=label, n=n, pearson=r, spearman=rho, bias=bias,
                        sigma_floor=sf, floor_flat=flat*100, floor_rough=rough*100, max_sigma=mx*100))
        print(f"   Pearson={r:.4f} Spearman={rho:.4f} bias={bias:+.4f}  "
              f"floor flat={flat*100:.3f}% rough={rough*100:.3f}%  max={mx*100:.1f}%\n")

    S = pd.DataFrame(out)
    print("=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(f"  {'config':<20}{'n':>4}{'Pearson':>9}{'Spearman':>9}{'bias':>8}"
          f"{'floor_flat':>11}{'floor_rough':>12}{'max_sig':>9}")
    for _, r in S.iterrows():
        print(f"  {r['config']:<20}{r['n']:>4}{r['pearson']:>9.4f}{r['spearman']:>9.4f}"
              f"{r['bias']:>+8.4f}{r['floor_flat']:>10.3f}%{r['floor_rough']:>11.3f}%{r['max_sigma']:>8.1f}%")
    csv = os.path.join(RESULTS_DIR, "additive_floor_roughness_unsat.csv")
    S.to_csv(csv, index=False); print(f"\nSaved: {csv}")
    print("\nRead: did removing saturation lift max_sig (was ~9%) and improve LOO toward the baseline?")
    print("Clean-floor variants (B, Q) should still have floor_flat == floor_rough.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
