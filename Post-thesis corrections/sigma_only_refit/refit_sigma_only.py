"""
Sigma-only re-fit on corrected data, multiple configurations.

Model change vs production:
  - Target changed from signed e_overall to |e_overall|.
  - Bias term (mu = beta_dz * dz_z) REMOVED. Model is now sigma-only.
  - Likelihood: HalfStudentT(nu, sigma). Sigma retains the "scale of the underlying
    two-sided distribution" interpretation, so downstream P75/P90 multipliers stay valid.

Model forms tested (all sigma = exp(...) on |e|; only the standardization and roughness
formula differ):
  * "production" — z-scored form + magnitude roughness (formula A):
        sigma = exp( log_sigma0 + sum_k gamma_k * (f_k - mean_k) / std_k )
        floor is terrain-dependent (all features drift below sigma0 via negative z-scores)
  * "exp-A" — un-centered + magnitude roughness (formula A, no gate):
        sigma = exp( log_sigma0 + sum_k beta_k * f_k / std_k )
        floor is terrain-dependent (roughness magnitude persists at same-site)
  * "exp-B" — un-centered + MISMATCH roughness (formula B, |rs_W - rs_M|):
        same sigma form as exp-A; roughness feature = weighted_mean(|rs_W - rs_M|)
        At self-pair rs_W = rs_M so mismatch = 0 → floor = exp(log_sigma0), single value
        Tests the hypothesis that forest sites need a mismatch-based roughness
  * "exp-M" — un-centered + ADAPTIVE roughness = min(magnitude, mismatch) per pair:
        Zero new hyperparameters; picks whichever of A or B is smaller for each pair.
        On kept sites (ratio ≈ 1): A and B similar, so behaves like A/B alike.
        On forest sites (ratio ≈ 0.5): mismatch < magnitude, so behaves like B.
        Tests whether "use whichever dissimilarity is smaller" rescues forest fit.
        At self-pair mismatch = 0 → min = 0 → floor = exp(log_sigma0), single value.
        NOTE: min-first, single-std-later — the std used to normalize the roughness
        feature is computed on the mixed distribution of min values.
  * "exp-M2" — same idea as M but STANDARDIZE-FIRST, MIN-LATER:
        Compute std_A and std_B separately from the training pairs, then take
        min(sat_A/std_A, sat_B/std_B) per pair. Physically cleaner because each
        formula is scaled by its own natural spread before being compared.
        Small numerical impact expected (std_A ≈ std_B in this dataset) but the
        interpretation is more defensible.
  * "exp-Q" — un-centered + distance-gated magnitude roughness (formula A * (1-exp(-d/500))):
        Roughness forced to 0 at d=0 via the gate → floor = exp(log_sigma0), single value

Two forest configurations:
  * forest-out — production exclusion list (Hultema, Malarberget out — current default)
  * forest-in  — Hultema + Malarberget re-included, all other exclusions kept

Total fits = 6 forms x 2 forest configs = 12 fits. Each writes:
  * a JSON with fitted coefficients, LOO metrics, per-mast self-prediction sigma,
    and feature_means_all / feature_stds_all (for future centering experiments)
  * exp-M2 JSONs also carry expM2_details with per-formula std_A, std_B, means
  * console table comparing all configs

RUN in pymc-env. Expected runtime: ~90-120 min total for a fresh run; script skips
configs whose JSONs already exist, so if you've run the 10 previous configs only
exp-M2 (2 fits) will actually re-fit — ~15-30 min.
"""
import os, sys, json, importlib.machinery, importlib.util, contextlib
import numpy as np, pandas as pd, pymc as pm

_HERE = os.path.dirname(os.path.abspath(__file__))
_RFC = os.path.normpath(os.path.join(_HERE, "..", "roughness_formula_comparison.py"))
_loader = importlib.machinery.SourceFileLoader("rfc", _RFC)
rfc = importlib.util.module_from_spec(importlib.util.spec_from_file_location("rfc", _RFC, loader=_loader))
_loader.exec_module(rfc)
pmod = rfc.pmod  # final model module (has EXCLUDED_MASTS, INPUT_PATH, etc.)

RESULTS_DIR = os.path.join(_HERE, "Results"); os.makedirs(RESULTS_DIR, exist_ok=True)

DRAWS, TUNE, CHAINS, CORES = 1000, 1000, 2, 1
# exp-Q priors: log_sigma0 sits at the floor (f=0), lower than production's mean-of-training-data
LS0_EXPQ_MU, LS0_EXPQ_SD, BETA_EXPQ_SD = -5.0, 1.5, 0.3
# production priors: log_sigma0 sits at training mean; same values as thesis-final
LS0_PROD_MU, LS0_PROD_SD, GAMMA_PROD_SD = -3.9, 0.5, 0.3

FOREST_MASTS = ["2011WM011", "2014WM011", "2012WM006"]  # Hultema + Malarberget
FOREST_CONFIGS = {
    "forest-out": list(pmod.EXCLUDED_MASTS),                          # as-is
    "forest-in":  [m for m in pmod.EXCLUDED_MASTS if m not in FOREST_MASTS],
}


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        o, e = sys.stdout, sys.stderr
        try: sys.stdout = dn; sys.stderr = dn; yield
        finally: sys.stdout = o; sys.stderr = e


def _build_pair_M(pair_A_tuple, pair_B_tuple):
    """Adaptive formula M: roughness_raw = min(magnitude, mismatch) per pair, THEN saturate.
    Reuses formula A's pair_df structure and replaces the roughness columns.
    At self-pair mismatch = 0, so min = 0 → roughness_sat = 0 → clean floor.

    NOTE: this is the "min-first, std-later" form (single std computed on the mixed
    distribution). The physically-cleaner alternative is exp-M2 (std-first, min-later).
    """
    pair_A, fc = pair_A_tuple
    pair_B, _ = pair_B_tuple
    pair_M = pair_A.copy()
    # merge-by-pair_id so ordering can't silently break
    mag_by = pair_A.set_index("pair_id")["wm_abs_roughness"]
    mm_by  = pair_B.set_index("pair_id")["wm_abs_roughness"]
    combined = pd.concat([mag_by, mm_by], axis=1).min(axis=1)
    pair_M["wm_abs_roughness"] = pair_M["pair_id"].map(combined).values
    pair_M["roughness_sat"] = 1.0 - np.exp(-pair_M["wm_abs_roughness"].values / rfc.ROUGH_SAT_SCALE)
    return pair_M, fc   # feature_config unchanged (same 5 features, just different roughness values)


def _build_pair_M2(pair_A_tuple, pair_B_tuple):
    """Adaptive formula M2: standardize A and B SEPARATELY, THEN take min per pair.

    Each pair carries both saturated roughness values (rough_sat_A and rough_sat_B).
    At fit time (per LOO fold), we compute std of each on training pairs, then
    z_A = rough_sat_A / std_A, z_B = rough_sat_B / std_B, feature = min(z_A, z_B).
    This is more principled than M because each formula is scaled by its own natural
    spread before being compared. At self-pair rough_sat_B = 0 (mismatch=0) so
    z_B = 0 → min = 0 → clean floor (same as M).
    """
    pair_A, fc = pair_A_tuple
    pair_B, _ = pair_B_tuple
    pair_M2 = pair_A.copy()
    # Attach both saturated and raw B values to A's pair_df (structure identical)
    satB_by = pair_B.set_index("pair_id")["roughness_sat"]
    rawB_by = pair_B.set_index("pair_id")["wm_abs_roughness"]
    pair_M2["rough_sat_A"] = pair_M2["roughness_sat"].values   # A's saturated (already present)
    pair_M2["rough_sat_B"] = pair_M2["pair_id"].map(satB_by).values
    pair_M2["wm_abs_roughness_A"] = pair_M2["wm_abs_roughness"].values
    pair_M2["wm_abs_roughness_B"] = pair_M2["pair_id"].map(rawB_by).values
    # roughness_sat / wm_abs_roughness columns will be REPLACED per fold by fit_expM2
    # (we can't precompute since standardization needs per-fold train stds)
    return pair_M2, fc


def load_data(forest_config):
    """Load sector data with the chosen exclusion list. Returns (sector_df, {formula_key: (pair_df, fc), ...})."""
    _saved = list(pmod.EXCLUDED_MASTS)
    try:
        pmod.EXCLUDED_MASTS = FOREST_CONFIGS[forest_config]
        sector_df = rfc.load_sector_data()
        A = rfc.build_pair_data_with_formula(sector_df, "A")   # magnitude
        B = rfc.build_pair_data_with_formula(sector_df, "B")   # mismatch |rs_W - rs_M|
        Q = rfc.build_pair_data_with_formula(sector_df, "Q")   # gated 500m magnitude
        T = rfc.build_pair_data_with_formula(sector_df, "T")   # magnitude-mismatch ||rs_W|-|rs_M||
        M  = _build_pair_M (A, B)                               # adaptive M: min-first, std-later
        M2 = _build_pair_M2(A, B)                               # adaptive M2: std-first, min-later
        formulas = {"A": A, "B": B, "Q": Q, "T": T, "M": M, "M2": M2}
    finally:
        pmod.EXCLUDED_MASTS = _saved
    return sector_df, formulas


# ── fits ──────────────────────────────────────────────────────────────────────
def fit_production(pair_df, feature_config):
    """Production form: z-scored (center + scale), HalfStudentT on |e|."""
    feats = [rc for _, rc, _ in feature_config]
    F = pair_df[feats].values.astype(float)
    means = F.mean(0); stds = F.std(0); stds[stds == 0] = 1.0
    Xz = (F - means) / stds
    abs_e = np.abs(pair_df["e_overall"].values.astype(float))
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=LS0_PROD_MU, sigma=LS0_PROD_SD)
        gammas = pm.HalfNormal("gammas", sigma=GAMMA_PROD_SD, shape=len(feats))
        sigma = pm.math.exp(log_sigma0 + pm.math.dot(Xz, gammas))
        pm.HalfStudentT("obs", nu=nu, sigma=sigma, observed=abs_e)
        idata = pm.sample(draws=DRAWS, tune=TUNE, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    return idata, feats, means, stds


def fit_expQ(pair_df, feature_config):
    """exp-Q form: divide by std, NO centering. HalfStudentT on |e|."""
    feats = [rc for _, rc, _ in feature_config]
    F = pair_df[feats].values.astype(float)
    stds = F.std(0); stds[stds == 0] = 1.0
    Xs = F / stds
    abs_e = np.abs(pair_df["e_overall"].values.astype(float))
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=LS0_EXPQ_MU, sigma=LS0_EXPQ_SD)
        betas = pm.HalfNormal("betas", sigma=BETA_EXPQ_SD, shape=len(feats))
        sigma = pm.math.exp(log_sigma0 + pm.math.dot(Xs, betas))
        pm.HalfStudentT("obs", nu=nu, sigma=sigma, observed=abs_e)
        idata = pm.sample(draws=DRAWS, tune=TUNE, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    return idata, feats, stds


def fit_expM2c(pair_df, feature_config):
    """exp-M2c: fully centered (z-scored) per-formula, then min.

    Non-roughness features: z_k = (f_k - mean_k) / std_k  (production-style)
    Roughness: z_A = (sat_A - mean_A) / std_A, z_B = (sat_B - mean_B) / std_B, then min.
    The resulting x_rough is a min of two z-scored values and can be NEGATIVE for
    below-mean pairs — coefficient stays HalfNormal (positive), so negative feature
    values translate to below-baseline sigma contribution, matching production."""
    non_rough_feats = [rc for _, rc, _ in feature_config if rc != "roughness_sat"]
    F_other = pair_df[non_rough_feats].values.astype(float)
    means_other = F_other.mean(0)
    stds_other = F_other.std(0); stds_other[stds_other == 0] = 1.0
    Xz_other = (F_other - means_other) / stds_other

    sat_A = pair_df["rough_sat_A"].values.astype(float)
    sat_B = pair_df["rough_sat_B"].values.astype(float)
    mean_A = float(sat_A.mean()); std_A = float(sat_A.std()) or 1.0
    mean_B = float(sat_B.mean()); std_B = float(sat_B.std()) or 1.0
    z_A = (sat_A - mean_A) / std_A
    z_B = (sat_B - mean_B) / std_B
    x_rough = np.minimum(z_A, z_B)

    abs_e = np.abs(pair_df["e_overall"].values.astype(float))
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=LS0_PROD_MU, sigma=LS0_PROD_SD)
        gammas_other = pm.HalfNormal("gammas_other", sigma=GAMMA_PROD_SD, shape=len(non_rough_feats))
        beta_rough   = pm.HalfNormal("beta_rough",   sigma=GAMMA_PROD_SD)
        sigma = pm.math.exp(log_sigma0 + pm.math.dot(Xz_other, gammas_other) + beta_rough * x_rough)
        pm.HalfStudentT("obs", nu=nu, sigma=sigma, observed=abs_e)
        idata = pm.sample(draws=DRAWS, tune=TUNE, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    all_feats = [rc for _, rc, _ in feature_config]
    info = {"non_rough_feats": non_rough_feats,
            "means_other": means_other, "stds_other": stds_other,
            "mean_A": mean_A, "std_A": std_A, "mean_B": mean_B, "std_B": std_B}
    return idata, all_feats, info


def coeffs_expM2c(idata, non_rough_feats):
    gammas_other = idata.posterior["gammas_other"].values.reshape(-1, len(non_rough_feats)).mean(0)
    beta_rough   = float(idata.posterior["beta_rough"].values.mean())
    gammas = {f: float(v) for f, v in zip(non_rough_feats, gammas_other)}
    gammas["roughness_sat"] = beta_rough
    return {"log_sigma0": float(idata.posterior["log_sigma0"].values.mean()),
            "gammas":     gammas,   # named "gammas" like production for downstream code
            "nu":         float(idata.posterior["nu"].values.mean())}


def fit_expM2(pair_df, feature_config):
    """exp-M2 form: standardize BOTH roughness formulas separately on the training
    set, then take min per pair. All other features handled as in fit_expQ."""
    non_rough_feats = [rc for _, rc, _ in feature_config if rc != "roughness_sat"]
    F_other = pair_df[non_rough_feats].values.astype(float)
    stds_other = F_other.std(0); stds_other[stds_other == 0] = 1.0
    Xs_other = F_other / stds_other

    sat_A = pair_df["rough_sat_A"].values.astype(float)
    sat_B = pair_df["rough_sat_B"].values.astype(float)
    std_A = float(sat_A.std()) or 1.0
    std_B = float(sat_B.std()) or 1.0
    z_A = sat_A / std_A
    z_B = sat_B / std_B
    x_rough = np.minimum(z_A, z_B)   # per-pair standardized-min (dimensionless)

    abs_e = np.abs(pair_df["e_overall"].values.astype(float))
    with pm.Model():
        nu = pm.Gamma("nu", alpha=2, beta=0.2)
        log_sigma0 = pm.Normal("log_sigma0", mu=LS0_EXPQ_MU, sigma=LS0_EXPQ_SD)
        betas_other = pm.HalfNormal("betas_other", sigma=BETA_EXPQ_SD, shape=len(non_rough_feats))
        beta_rough = pm.HalfNormal("beta_rough", sigma=BETA_EXPQ_SD)
        sigma = pm.math.exp(log_sigma0 + pm.math.dot(Xs_other, betas_other) + beta_rough * x_rough)
        pm.HalfStudentT("obs", nu=nu, sigma=sigma, observed=abs_e)
        idata = pm.sample(draws=DRAWS, tune=TUNE, target_accept=0.95,
                          chains=CHAINS, cores=CORES, progressbar=False, random_seed=42)
    # Return in a shape compatible with the LOO/self-prediction consumers.
    # 'stds' here bundles: 4 non-rough stds + std_A + std_B (roughness split by formula).
    all_feats = [rc for _, rc, _ in feature_config]
    info = {"non_rough_feats": non_rough_feats, "stds_other": stds_other,
            "std_A": std_A, "std_B": std_B}
    return idata, all_feats, info


def coeffs_expM2(idata, non_rough_feats):
    betas_other = idata.posterior["betas_other"].values.reshape(-1, len(non_rough_feats)).mean(0)
    beta_rough  = float(idata.posterior["beta_rough"].values.mean())
    betas = {f: float(v) for f, v in zip(non_rough_feats, betas_other)}
    betas["roughness_sat"] = beta_rough   # keep same key for downstream code
    return {"log_sigma0": float(idata.posterior["log_sigma0"].values.mean()),
            "betas":      betas,
            "nu":         float(idata.posterior["nu"].values.mean())}


def coeffs_production(idata, feats):
    return {"log_sigma0": float(idata.posterior["log_sigma0"].values.mean()),
            "gammas":     {f: float(v) for f, v in zip(feats, idata.posterior["gammas"].values.reshape(-1, len(feats)).mean(0))},
            "nu":         float(idata.posterior["nu"].values.mean())}


def coeffs_expQ(idata, feats):
    return {"log_sigma0": float(idata.posterior["log_sigma0"].values.mean()),
            "betas":      {f: float(v) for f, v in zip(feats, idata.posterior["betas"].values.reshape(-1, len(feats)).mean(0))},
            "nu":         float(idata.posterior["nu"].values.mean())}


# ── LOO (pair-level, physical-pair holdout) ───────────────────────────────────
def _folds(pair_df):
    pair_df = pair_df.copy()
    pair_df["_grp"] = pair_df["pair_id"].apply(lambda x: "__".join(sorted(x.split("__"))))
    return pair_df, pair_df.groupby("_grp")["pair_id"].unique().to_dict()


def loo(pair_df, feature_config, form):
    pair_df, folds = _folds(pair_df)
    rows = []
    for held in folds.values():
        train = pair_df[~pair_df["pair_id"].isin(held)]
        test  = pair_df[pair_df["pair_id"].isin(held)]
        if len(train) < 5:
            continue
        try:
            with _quiet():
                if form in ("production", "exp-Mc"):
                    idata, feats, means, stds = fit_production(train, feature_config)
                    c = coeffs_production(idata, feats)
                    Xt = (test[feats].values.astype(float) - means) / stds
                    log_s = c["log_sigma0"] + Xt @ np.array([c["gammas"][f] for f in feats])
                elif form in ("exp-M2", "exp-M2c"):
                    if form == "exp-M2":
                        idata, feats, info = fit_expM2(train, feature_config)
                        c = coeffs_expM2(idata, info["non_rough_feats"])
                        Xt_other = (test[info["non_rough_feats"]].values.astype(float)
                                    / info["stds_other"])
                        z_A_test = test["rough_sat_A"].values.astype(float) / info["std_A"]
                        z_B_test = test["rough_sat_B"].values.astype(float) / info["std_B"]
                    else:  # exp-M2c
                        idata, feats, info = fit_expM2c(train, feature_config)
                        c = coeffs_expM2c(idata, info["non_rough_feats"])
                        Xt_other = ((test[info["non_rough_feats"]].values.astype(float)
                                     - info["means_other"]) / info["stds_other"])
                        z_A_test = ((test["rough_sat_A"].values.astype(float) - info["mean_A"])
                                    / info["std_A"])
                        z_B_test = ((test["rough_sat_B"].values.astype(float) - info["mean_B"])
                                    / info["std_B"])
                    beta_key = "gammas" if form == "exp-M2c" else "betas"
                    betas_other = np.array([c[beta_key][f] for f in info["non_rough_feats"]])
                    x_rough_test = np.minimum(z_A_test, z_B_test)
                    log_s = (c["log_sigma0"] + Xt_other @ betas_other
                             + c[beta_key]["roughness_sat"] * x_rough_test)
                else:
                    idata, feats, stds = fit_expQ(train, feature_config)
                    c = coeffs_expQ(idata, feats)
                    Xt = test[feats].values.astype(float) / stds
                    log_s = c["log_sigma0"] + Xt @ np.array([c["betas"][f] for f in feats])
        except Exception as ex:
            print(f"   fold failed: {ex}"); continue
        pred = np.exp(log_s)
        for i in range(len(test)):
            rows.append({"pair_id": test.iloc[i]["pair_id"],
                         "predicted_sigma": float(pred[i]),
                         "actual_abs_error": float(abs(test.iloc[i]["e_overall"]))})
    d = pd.DataFrame(rows)
    if len(d) < 2:
        return d, np.nan, np.nan, np.nan
    return (d,
            d["predicted_sigma"].corr(d["actual_abs_error"]),
            d["predicted_sigma"].corr(d["actual_abs_error"], method="spearman"),
            (d["predicted_sigma"] - d["actual_abs_error"]).mean())


# ── per-mast self-prediction sigma ────────────────────────────────────────────
def per_mast_self_sigma(sector_df, form, coeffs, means_or_none, stds, feats,
                          expM2_info=None):
    """Compute self-prediction sigma per mast under the fitted model.

    For a self-pair: dist=0, turning=0, log_speedup=0, dz=0. Roughness handling
    depends on form:
      * production (formula A, z-scored): roughness_sat_self = 1 - exp(-|rs_own|/0.01)
                                          → contribution = gamma * (raw - mean) / std
      * exp-A (formula A, un-centered):   same raw roughness, but contribution = beta * raw / std
                                          (no mean subtraction — roughness magnitude persists)
      * exp-B (formula B, un-centered):   at self-pair rs_WTG = rs_MM → |rs_W - rs_M| = 0
                                          → roughness feature vanishes → sigma = exp(log_sigma0)
      * exp-M (formula M, un-centered):   min(magnitude, mismatch); mismatch = 0 at self-pair
                                          → min = 0 → roughness vanishes → sigma = exp(log_sigma0)
      * exp-M2 (formula M2, un-centered): standardize A and B separately, then min. At self-pair
                                          sat_B = 0 → z_B = 0 → min = 0 → sigma = exp(log_sigma0)
      * exp-Mc (formula M, z-scored):     production-style centering with min-based roughness.
                                          At self all raw features = 0, so each contributes
                                          -mean_k/std_k * gamma_k. Single value across masts.
      * exp-M2c (formula M2, per-formula z-score, then min):
                                          At self sat_B = 0 → z_B_self = -mean_B/std_B (constant);
                                          z_A_self = (sat_A_own - mean_A) / std_A depends on mast.
                                          Terrain-dependent for flat/moderate masts, constant for rough.
      * exp-Q (formula Q, un-centered):   raw roughness × (1 - exp(-0/500)) = 0
                                          → ALL features are 0 → sigma = exp(log_sigma0)
      * exp-T (formula T, un-centered):   ||rs_W|-|rs_M|| = 0 at self-pair
                                          → ALL features are 0 → sigma = exp(log_sigma0)
    """
    rows = []
    for mast, mrows in sector_df.groupby(sector_df["pair_id"].apply(lambda x: x.split("__")[1])):
        first_pair = mrows["pair_id"].iloc[0]
        grp = mrows[mrows["pair_id"] == first_pair]
        freq = pd.to_numeric(grp["freq_MM"], errors="coerce").fillna(0).values
        if freq.sum() == 0:
            continue
        w = freq / freq.sum()
        rs = pd.to_numeric(grp["rough_speedup_MM_frac"], errors="coerce").values
        valid = ~np.isnan(rs)
        rs_self = float(np.sum(w[valid] * np.abs(rs[valid])) / w[valid].sum()) if valid.sum() > 0 else np.nan

        # feature values at self-site
        raw = {f: 0.0 for f in feats}
        # roughness handling depends on form
        if not np.isnan(rs_self):
            if form in ("production", "exp-A"):
                # formula A: roughness magnitude persists at same-site
                raw["roughness_sat"] = 1.0 - np.exp(-rs_self / rfc.ROUGH_SAT_SCALE)
            # exp-B / exp-M / exp-Mc / exp-Q / exp-T: raw["roughness_sat"] stays 0.0
            # exp-M2: raw["roughness_sat"] stays 0.0 (handled via expM2_info below)
            # exp-M2c: raw["roughness_sat"] not used; per-formula z_A/z_B computed below

        log_s = coeffs["log_sigma0"]
        if form in ("production", "exp-Mc"):
            # Centered z-scoring for all 5 features (production style).
            # For exp-Mc, raw["roughness_sat"] is 0 because min(mag, 0) = 0.
            for i, f in enumerate(feats):
                log_s += coeffs["gammas"][f] * (raw[f] - means_or_none[i]) / stds[i]
        elif form == "exp-M2c":
            # Per-formula centered z-scoring on A and B, then min for roughness.
            # Non-roughness features get centered z-scoring like production.
            assert expM2_info is not None, "exp-M2c self-prediction requires expM2_info"
            non_rough = expM2_info["non_rough_feats"]
            for f in non_rough:
                i = feats.index(f)
                log_s += coeffs["gammas"][f] * (raw[f] - means_or_none[i]) / stds[i]
            # Roughness: sat_A_self depends on mast, sat_B_self = 0
            sat_A_self = (1.0 - np.exp(-rs_self / rfc.ROUGH_SAT_SCALE)
                          if not np.isnan(rs_self) else 0.0)
            z_A_self = (sat_A_self - expM2_info["mean_A"]) / expM2_info["std_A"]
            z_B_self = (0.0         - expM2_info["mean_B"]) / expM2_info["std_B"]
            x_rough_self = min(z_A_self, z_B_self)
            log_s += coeffs["gammas"]["roughness_sat"] * x_rough_self
        elif form == "exp-M2":
            # Per-formula ÷std on A and B (no centering), then min for roughness.
            assert expM2_info is not None, "exp-M2 self-prediction requires expM2_info"
            non_rough = expM2_info["non_rough_feats"]
            for f in non_rough:
                i = feats.index(f)
                log_s += coeffs["betas"][f] * raw[f] / stds[i]
            sat_A_self = (1.0 - np.exp(-rs_self / rfc.ROUGH_SAT_SCALE)
                          if not np.isnan(rs_self) else 0.0)
            z_A_self = sat_A_self / expM2_info["std_A"]
            z_B_self = 0.0        / expM2_info["std_B"]
            x_rough_self = min(z_A_self, z_B_self)
            log_s += coeffs["betas"]["roughness_sat"] * x_rough_self
        else:
            # exp-A, exp-B, exp-M, exp-Q: un-centered (÷std, no mean subtraction)
            for i, f in enumerate(feats):
                log_s += coeffs["betas"][f] * raw[f] / stds[i]
        rows.append({"mast": mast, "rs_own_wmean_abs": rs_self,
                     "sigma_self_pct": float(np.exp(log_s)) * 100})
    return pd.DataFrame(rows).sort_values("sigma_self_pct").reset_index(drop=True)


# ── one full configuration (form x forest) ────────────────────────────────────
def run_config(form, forest_config, skip_if_exists=True):
    label = f"{form}__{forest_config}"
    json_path = os.path.join(RESULTS_DIR, f"{label}.json")
    if skip_if_exists and os.path.exists(json_path):
        print(f"\n{'='*78}\n[{label}] JSON already exists — SKIPPING (delete file to re-run)\n{'='*78}")
        with open(json_path) as jf:
            return json.load(jf)
    print(f"\n{'='*78}\n[{label}] loading data ...\n{'='*78}")
    sector_df, formulas = load_data(forest_config)
    # Map each form to the roughness formula it uses:
    #   production, exp-A -> A (magnitude, no gate)
    #   exp-B             -> B (mismatch |rs_W - rs_M|)
    #   exp-M             -> M (adaptive: min(magnitude, mismatch) per pair)
    #   exp-Q             -> Q (magnitude gated by 1-exp(-d/500))
    formula_key = {"production": "A", "exp-A": "A", "exp-B": "B",
                    "exp-M": "M", "exp-Mc": "M",
                    "exp-M2": "M2", "exp-M2c": "M2",
                    "exp-T": "T",
                    "exp-Q": "Q"}[form]
    pair_df, feature_config = formulas[formula_key]
    n = len(pair_df)

    print(f"[{label}] full fit (n={n}) ...")
    with _quiet():
        if form in ("production", "exp-Mc"):
            # Production form (z-scored). For exp-Mc the pair_df is pair_M (min-based
            # roughness), so this is "production applied to the min-based feature".
            idata, feats, means, stds = fit_production(pair_df, feature_config)
            c = coeffs_production(idata, feats)
            expM2_info = None
        elif form in ("exp-M2", "exp-M2c"):
            if form == "exp-M2":
                idata, feats, expM2_info = fit_expM2(pair_df, feature_config)
                c = coeffs_expM2(idata, expM2_info["non_rough_feats"])
                means = None
            else:  # exp-M2c (centered)
                idata, feats, expM2_info = fit_expM2c(pair_df, feature_config)
                c = coeffs_expM2c(idata, expM2_info["non_rough_feats"])
                # means for the non-rough features (they were centered too)
                means = np.array([expM2_info["means_other"][expM2_info["non_rough_feats"].index(f)]
                                  if f in expM2_info["non_rough_feats"] else 0.0
                                  for f in feats])
            # stds array: non-rough stds + effective std of the roughness feature
            if form == "exp-M2":
                z_A = pair_df["rough_sat_A"].values / expM2_info["std_A"]
                z_B = pair_df["rough_sat_B"].values / expM2_info["std_B"]
            else:
                z_A = (pair_df["rough_sat_A"].values - expM2_info["mean_A"]) / expM2_info["std_A"]
                z_B = (pair_df["rough_sat_B"].values - expM2_info["mean_B"]) / expM2_info["std_B"]
            _rough_std_effective = float(np.minimum(z_A, z_B).std()) or 1.0
            stds = np.array([expM2_info["stds_other"][expM2_info["non_rough_feats"].index(f)]
                             if f in expM2_info["non_rough_feats"] else _rough_std_effective
                             for f in feats])
        else:
            # exp-A, exp-B, exp-M, exp-Q — all use the un-centered fit function.
            idata, feats, stds = fit_expQ(pair_df, feature_config); means = None
            c = coeffs_expQ(idata, feats)
            expM2_info = None

    print(f"[{label}] pair-level LOO ...")
    loo_df, r, rho, bias = loo(pair_df, feature_config, form)
    print(f"   Pearson={r:.4f}  Spearman={rho:.4f}  bias={bias:+.4f}")

    self_df = per_mast_self_sigma(sector_df, form, c, means, stds, feats,
                                     expM2_info=expM2_info)
    print(f"   self-prediction sigma: min={self_df['sigma_self_pct'].min():.3f}%  "
          f"median={self_df['sigma_self_pct'].median():.3f}%  max={self_df['sigma_self_pct'].max():.3f}%")

    out = {"config": label, "form": form, "forest_config": forest_config, "n_pairs": n,
           "loo": {"pearson": float(r), "spearman": float(rho), "bias": float(bias)},
           "coeffs": c, "feature_order": feats,
           "feature_stds": {f: float(stds[i]) for i, f in enumerate(feats)},
           "self_prediction_range_pct": {"min": float(self_df["sigma_self_pct"].min()),
                                          "median": float(self_df["sigma_self_pct"].median()),
                                          "max": float(self_df["sigma_self_pct"].max())}}
    if means is not None:
        out["feature_means"] = {f: float(means[i]) for i, f in enumerate(feats)}

    # Always record feature means AND stds on RAW pair_df values — even for un-centered
    # forms — so we can later test centering (z-scoring) with any of these formulas
    # without needing to re-run the full fit.
    for f in feats:
        if f in pair_df.columns:
            out.setdefault("feature_means_all", {})[f] = float(pair_df[f].mean())
            out.setdefault("feature_stds_all",  {})[f] = float(pair_df[f].std())

    # M2-specific extras: record the two per-formula stds and means so someone can
    # reason about the standardize-before-min operation without re-loading the data.
    if expM2_info is not None:
        out["expM2_details"] = {
            "std_A": expM2_info["std_A"], "std_B": expM2_info["std_B"],
            "mean_A": float(pair_df["rough_sat_A"].mean()),
            "mean_B": float(pair_df["rough_sat_B"].mean()),
            "mean_raw_A": float(pair_df["wm_abs_roughness_A"].mean()),
            "mean_raw_B": float(pair_df["wm_abs_roughness_B"].mean()),
            "std_raw_A": float(pair_df["wm_abs_roughness_A"].std()),
            "std_raw_B": float(pair_df["wm_abs_roughness_B"].std()),
        }

    json_path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(json_path, "w") as jf: json.dump(out, jf, indent=2)
    loo_df.to_csv(os.path.join(RESULTS_DIR, f"{label}_loo.csv"), index=False)
    self_df.to_csv(os.path.join(RESULTS_DIR, f"{label}_self_prediction.csv"), index=False)
    print(f"   wrote {json_path}")
    return out


def main():
    print("="*78 + "\nSIGMA-ONLY REFIT ON CORRECTED DATA (4 configurations)\n" + "="*78)
    print("Target: |e_overall|.  Likelihood: HalfStudentT(nu, sigma).  No bias term.\n")
    results = []
    for form in ["production", "exp-A", "exp-B", "exp-T",
                 "exp-M", "exp-Mc", "exp-M2", "exp-M2c", "exp-Q"]:
        for forest_config in ["forest-out", "forest-in"]:
            results.append(run_config(form, forest_config))

    print("\n" + "="*78 + "\nSUMMARY\n" + "="*78)
    print(f"  {'config':<28}{'n':>4}{'Pearson':>9}{'Spearman':>10}{'bias':>9}"
          f"{'self_min':>10}{'self_med':>10}{'self_max':>10}")
    for r in results:
        s = r["self_prediction_range_pct"]
        print(f"  {r['config']:<28}{r['n_pairs']:>4}{r['loo']['pearson']:>9.4f}"
              f"{r['loo']['spearman']:>10.4f}{r['loo']['bias']:>+9.4f}"
              f"{s['min']:>9.3f}%{s['median']:>9.3f}%{s['max']:>9.3f}%")

    print("\nDECISION CRITERIA (in order):")
    print("  1. self_min ~ self_max (terrain-independent floor)")
    print("  2. Pearson within 0.02 of production/forest-out baseline")
    print("  3. self_max <= your defensible max (<~1% ideal)")
    print(f"\nAll outputs: {RESULTS_DIR}")


if __name__ == "__main__":
    import multiprocessing as mp; mp.freeze_support()
    main()
