"""
Exp-floor (un-centered) model + roughness-variant test.

The "proper" version of the floor decoupling: keep the multiplicative exp (so the ceiling and fit are
preserved), but standardize features by ÷std WITHOUT centering — so the floor decouples from the slope
and stays clean at the same site. NOT the additive/linear form (that capped the ceiling ~12%).

    e ~ StudentT(nu, mu, sigma)
    mu    = beta_dz * dz                              (bias, RAW dz per-metre)
    sigma = exp( log_sigma0 + sum_k beta_k * (f_k / std_k) )   # f_k = SATURATED features, ÷std, NOT centered

At same-site f_k = 0 (except magnitude roughness) -> sigma = exp(log_sigma0) = the floor, independent of
the betas (slopes). Ceiling stays high (exp of saturated features, ~30%). B/Q roughness -> terrain-independent.

Compares: baseline (current z-scored multiplicative) vs exp-A/B/Q/E (un-centered).

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
# priors (may need one tuning pass): log_sigma0 sits at the FLOOR (f=0), so lower than the current -3.5
LS0_MU, LS0_SD, BETA_SD = -5.0, 1.5, 0.3

# rough_map = SATURATED roughness_sat value at a rough same-site (magnitude persists=0.83; mismatch/gated=0)
CONFIGS = [
    ("baseline-mult (z-scored)", "multiplicative", "A", {"roughness_sat": 0.83}),
    ("exp-A (magnitude)", "expfloor", "A", {"roughness_sat": 0.83}),
    ("exp-B (mismatch)",  "expfloor", "B", {"roughness_sat": 0.0}),
    ("exp-Q (gated mag)", "expfloor", "Q", {"roughness_sat": 0.0}),
    ("exp-E (both)",      "expfloor", "E", {"roughness_sat": 0.83, "rough_diff_sat": 0.0}),
]


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        o, e = sys.stdout, sys.stderr
        try: sys.stdout = dn; sys.stderr = dn; yield
        finally: sys.stdout = o; sys.stderr = e


# ── exp-floor, un-centered (SATURATED features / std, NO centering) ──────────────────
def fit_expfloor(pair_df, feature_config, draws=DRAWS, tune=TUNE):
    feats = [rc for _, rc, _ in feature_config]
    F = pair_df[feats].values.astype(float)               # saturated features (0 at same-site, except mag rough)
    stds = F.std(0); stds[stds == 0] = 1.0                 # scale only, NO centering -> 0 stays 0
    Xs = F / stds
    dz = pair_df["dz"].values.astype(float)
    e = pair_df["e_overall"].values.astype(float)
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=LS0_MU, sigma=LS0_SD)
        betas = pm.HalfNormal("betas", sigma=BETA_SD, shape=len(feats))
        beta_dz = pm.Normal("beta_dz", mu=0, sigma=1e-3)
        sigma = pm.math.exp(log_sigma0 + pm.math.dot(Xs, betas))
        mu = beta_dz * dz
        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=e)
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    return idata, feats, stds


def _means(idata, k):
    return (float(idata.posterior["log_sigma0"].values.mean()),
            idata.posterior["betas"].values.reshape(-1, k).mean(0))


def run_loo_expfloor(pair_df, feature_config):
    pair_df = pair_df.copy()
    pair_df["_grp"] = pair_df["pair_id"].apply(lambda x: "__".join(sorted(x.split("__"))))
    folds = pair_df.groupby("_grp")["pair_id"].unique().to_dict()
    feats = [rc for _, rc, _ in feature_config]
    rows = []
    for held in folds.values():
        train = pair_df[~pair_df["pair_id"].isin(held)]
        test = pair_df[pair_df["pair_id"].isin(held)]
        if len(train) < 5:
            continue
        try:
            with _quiet():
                idata, _, stds = fit_expfloor(train, feature_config)
        except Exception as ex:
            print(f"   fold failed: {ex}"); continue
        ls0, bt = _means(idata, len(feats))
        Xt = test[feats].values.astype(float) / stds
        for i in range(len(test)):
            rows.append({"predicted_sigma": float(np.exp(ls0 + np.dot(Xt[i], bt))),
                         "actual_error": abs(test.iloc[i]["e_overall"])})
    d = pd.DataFrame(rows)
    if len(d) < 2:
        return d, np.nan, np.nan, np.nan
    return (d, d["predicted_sigma"].corr(d["actual_error"]),
            d["predicted_sigma"].corr(d["actual_error"], method="spearman"),
            (d["predicted_sigma"] - d["actual_error"]).mean())


def expfloor_report(pair_df, feature_config, rough_map):
    feats = [rc for _, rc, _ in feature_config]
    with _quiet():
        idata, _, stds = fit_expfloor(pair_df, feature_config)
    ls0, bt = _means(idata, len(feats))
    flat = np.exp(ls0)                                                        # all f=0
    rough = np.exp(ls0 + sum(bt[i] * (rough_map.get(f, 0.0) / stds[i]) for i, f in enumerate(feats)))
    mx = np.exp(ls0 + float(np.dot(pair_df[feats].max().values.astype(float) / stds, bt)))
    return np.exp(ls0), flat, rough, mx     # report the fitted floor = exp(log_sigma0)


# ── multiplicative baseline (current z-scored model, reference) ─────────────────────
def mult_report(pair_df, feature_config, rough_map):
    data = rfc.rebuild_data_dict_custom(pair_df, feature_config)
    with _quiet():
        _, idata, _ = rfc.fit_model_custom(data, feature_config)
    sc = data["scalers"]; ls0 = float(idata.posterior["log_sigma0"].values.mean())
    gam = {gn: float(idata.posterior[gn].values.mean()) for gn, _, _ in feature_config}
    def sig(raw):
        ls = ls0
        for gn, rc, _ in feature_config:
            ls += gam[gn] * ((raw.get(rc, 0.0) - sc[f"{rc}_mean"]) / sc[f"{rc}_std"])
        return np.exp(ls)
    return np.nan, sig({}), sig(rough_map), sig({rc: pair_df[rc].max() for _, rc, _ in feature_config})


def main():
    print("=" * 78 + "\nEXP-FLOOR (UN-CENTERED) + ROUGHNESS-VARIANT TEST\n" + "=" * 78)
    sector_df = rfc.load_sector_data()
    out = []
    for label, mtype, rkey, rough_map in CONFIGS:
        print(f"[{label}] model={mtype} roughness={rkey} ...")
        pair_df, fc = rfc.build_pair_data_with_formula(sector_df, rkey)
        n = len(pair_df)
        if mtype == "expfloor":
            _, r, rho, bias = run_loo_expfloor(pair_df, fc)
            fl, flat, rough, mx = expfloor_report(pair_df, fc, rough_map)
        else:
            _, r, rho, bias = rfc.run_loo_cv_custom(pair_df, fc)
            fl, flat, rough, mx = mult_report(pair_df, fc, rough_map)
        out.append(dict(config=label, n=n, pearson=r, spearman=rho, bias=bias,
                        floor=fl*100, floor_flat=flat*100, floor_rough=rough*100, max_sigma=mx*100))
        print(f"   Pearson={r:.4f} Spearman={rho:.4f} bias={bias:+.4f}  "
              f"floor flat={flat*100:.3f}% rough={rough*100:.3f}%  max={mx*100:.1f}%\n")

    S = pd.DataFrame(out)
    print("=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(f"  {'config':<26}{'n':>4}{'Pearson':>9}{'Spearman':>9}{'bias':>8}"
          f"{'floor_flat':>11}{'floor_rough':>12}{'max_sig':>9}")
    for _, r in S.iterrows():
        print(f"  {r['config']:<26}{r['n']:>4}{r['pearson']:>9.4f}{r['spearman']:>9.4f}"
              f"{r['bias']:>+8.4f}{r['floor_flat']:>10.3f}%{r['floor_rough']:>11.3f}%{r['max_sigma']:>8.1f}%")
    csv = os.path.join(RESULTS_DIR, "expfloor_uncentered_roughness.csv")
    S.to_csv(csv, index=False); print(f"\nSaved: {csv}")
    print("\nHOPED-FOR RESULT: exp-Q matches the baseline LOO (~0.93) AND keeps a high ceiling (~30%)")
    print("AND has floor_flat == floor_rough (clean). If so, exp-Q is the model (decoupled clean floor +")
    print("multiplicative fit + ceiling). If LOO lags or the sampler struggles, we retune LS0_MU/BETA_SD.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
