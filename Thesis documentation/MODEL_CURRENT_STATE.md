# MODEL_CURRENT_STATE.md

**Source of truth:** `Bayesian_approach/Final model/ws_uncertainty_model_pairlevel_final` as of 2026-05-04  
**Results:** `Bayesian_approach/Final model/Results/ws_uncertainty_pairlevel_results.json`

---

## 1. Target Variable

Frequency-weighted omnidirectional signed wind speed error per directional pair:

```
e_overall_i = (WS_pred_overall_i - WS_self_overall_i) / WS_self_overall_i
```

where `WS_pred_overall = sum(w_pred_norm * WS_pred_sector)` and `WS_self_overall = sum(w_self_norm * WS_self_sector)`, with weights from `Sample_count_pred` and `Sample_count_self`. The error is signed (positive = overprediction). Each directional pair produces one scalar observation.

## 2. Likelihood Function

```
e_overall_i ~ StudentT(nu, mu_i, sigma_i)
```

Each directional pair contributes one equally-weighted observation. No energy weighting is applied to the likelihood.

## 3. Model Structure

The model is a flat Bayesian regression — no hierarchical structure, no random effects, no nesting. All pairs are at the same level. The pair random effect from the earlier sector-level model was removed because the model now operates directly at the pair level, so within-pair correlation is no longer an issue.

## 4. Sigma Model (Log-Link)

```
log(sigma_i) = log_sigma0
             + gamma_dist      * dist_sat_z
             + gamma_turning   * turning_sat_z
             + gamma_speedup   * wm_abs_log_speedup_z
             + gamma_roughness * roughness_sat_z
             + gamma_dz        * dz_sat_z

sigma_i = exp(log(sigma_i))
```

Log-link ensures sigma > 0 and gives multiplicative feature effects.

## 5. Mean Model (Bias)

```
mu_i = beta_dz * dz_z_i
```

where `dz_z` is the z-scored signed height difference (WTG minus MM elevation). Captures the systematic WAsP elevation bias: positive dz tends toward underprediction.

## 6. Features

All 5 sigma features and the 1 mu feature are pair-level. Sector values are aggregated using `freq_MM` (MM Weibull frequency distribution) as weights. All features are z-scored before entering the model.

| # | Gamma name | Raw column | Formula | Type |
|---|-----------|-----------|---------|------|
| 1 | gamma_dist | dist_sat | `1 - exp(-d / d_A)` | Pair geometry |
| 2 | gamma_turning | turning_sat | `1 - exp(-t / 3.0)` where `t = sum(freq_MM_norm * \|d_turning_deg\|)` | Sector -> Pair (freq-weighted, saturating) |
| 3 | gamma_speedup | wm_abs_log_speedup | `sum(freq_MM_norm * \|log(speedup_WTG / speedup_MM)\|)` | Sector -> Pair (freq-weighted) |
| 4 | gamma_roughness | roughness_sat | `1 - exp(-r / 0.01)` where `r = sum(freq_MM_norm * \|(rough_WTG + rough_MM) / 2\|)` | Sector -> Pair (freq-weighted, saturating) |
| 5 | gamma_dz | dz_sat | `1 - exp(-\|dz\| / 40)` | Pair geometry |
| mu | beta_dz | dz | Signed height difference (metres) | Pair geometry |

### Feature Design Notes

**Saturating distance (dist_sat):** Uses the WAsP TR6 safe-zone distance (d_A) as the characteristic scale. `1 - exp(-d/d_A)` gives 0 at d=0, 0.63 at d=d_A, and saturates toward 1. Unlike a clamped log transform, this avoids hard discontinuities and gives short-range pairs a non-zero contribution.

**Distance as a conditional feature:** The bivariate Spearman correlation between dist_sat and |e_overall| is only ρ ≈ −0.04, yet the feature contributes 14.6% of explained variance in the multivariate model. This is because distance is a *partial/conditional* effect: it only explains error variance when combined with the terrain complexity encoded in d_A. The highest errors come from close pairs in complex terrain (e.g. Herzhausen: 1.5 km, 18% error), while distant pairs in flat terrain have low error (e.g. Kabbo: 12.2 km, 1.8% error). Raw distance alone cannot capture this — the saturating formulation `1 - exp(-d/d_A)` encodes distance *relative to terrain difficulty*, making it a strong conditional predictor despite weak marginal correlation.

**Turning (turning_sat):** `1 - exp(-t / 3.0)` where `t = sum(freq_MM_norm * |d_turning_deg|)`. The saturating transform bounds the feature to [0, 1], preventing extreme turning values from producing unbounded uncertainty predictions. The scale of 3.0 degrees preserves sensitivity in the 0-3 degree training range while capping extreme values. Always weighted by `freq_MM` regardless of any other setting, because predicted sample counts depend on turning — using them would create circular dependence.

**Speedup (wm_abs_log_speedup):** Uses the log ratio of overall speedup factors (orographic x roughness combined). The log ratio is symmetric: `log(a/b) = -log(b/a)`. Orographic and roughness speedups were tested separately but the combined factor performed better — the overlap is complementary, not redundant.

**Roughness (roughness_sat):** `1 - exp(-r / 0.01)` where `r = sum(freq_MM_norm * |(rough_WTG + rough_MM) / 2|)`. The saturating transform bounds the feature to [0, 1]. The scale of 0.01 was chosen so training mean roughness (0.004) maps to ~0.33 and the training high (~0.009) maps to ~0.59, preserving sensitivity in the training range while bounding extreme values. Both positive (acceleration) and negative (deceleration) roughness corrections indicate large WAsP adjustments, which increase uncertainty regardless of sign.

**Saturating dz (dz_sat):** `1 - exp(-|dz|/40)` where 40 metres is the saturation scale. At |dz|=40m the feature is at 0.63; at |dz|=120m it is at 0.95. The scale of 40m was chosen to give meaningful differentiation across the dataset's typical range (0-70m) while saturating for extreme cases. A LOO sweep of scale values (10, 20, 30, 40, 60, 80, 120) showed minimal sensitivity (Pearson 0.900–0.908); larger scales compress the useful range too much for sites with moderate height differences.

### Sector Aggregation Weights

Sector-to-pair aggregation uses `freq_MM` (MM-site Weibull frequency distribution). This was chosen over `weight_energy_predicted` and `Sample_count_pred` because:
- `freq_MM` is a purely observed MM quantity with no WAsP model dependence
- It is symmetric between A->B and B->A directions
- `Sample_count_pred` was tested as an alternative (FEATURE_WEIGHT_MODE toggle) and gave marginal improvement (Pearson 0.905 -> 0.912) but requires an additional WAsP output at deployment, adding complexity for negligible gain

## 7. Priors

| Parameter | Distribution | Parameters | Rationale |
|-----------|-------------|------------|-----------|
| nu | Gamma | alpha=2, beta=0.2 (mean=10) | Allows heavy tails for outliers |
| log_sigma0 | Normal | mu=-3.9, sigma=0.5 | Baseline ~2-3% uncertainty |
| gamma_dist | HalfNormal | sigma=0.3 | Positive-only: more distance can only increase uncertainty |
| gamma_turning | HalfNormal | sigma=0.3 | Positive-only: more turning mismatch increases uncertainty |
| gamma_speedup | HalfNormal | sigma=0.3 | Positive-only: more speedup mismatch increases uncertainty |
| gamma_roughness | HalfNormal | sigma=0.3 | Positive-only: larger roughness correction increases uncertainty |
| gamma_dz | HalfNormal | sigma=0.3 | Positive-only: more elevation difference increases uncertainty |
| beta_dz | Normal | mu=0, sigma=0.05 | Signed — bias can be positive or negative |

The log_sigma0 prior was changed from mu=-2.5 (sector model, ~8% baseline) to mu=-3.9 (~2% baseline) because the pair-level target has much smaller magnitude after cross-sector cancellation.

## 8. Posterior Estimates (Full Model Fit)

From `ws_uncertainty_pairlevel_results.json` (38 pairs, 4 chains x 2000 draws):

| Parameter | Posterior Mean | Std | Multiplier at z=+1 |
|-----------|---------------|-----|---------------------|
| nu | 14.23 | 7.49 | — |
| log_sigma0 | -3.514 | — | sigma0 = 3.00% |
| gamma_dist | 0.187 | 0.109 | 1.206x |
| gamma_turning | 0.287 | 0.139 | 1.333x |
| gamma_speedup | 0.130 | 0.100 | 1.139x |
| gamma_roughness | 0.285 | 0.148 | 1.329x |
| gamma_dz | 0.088 | 0.073 | 1.092x |
| beta_dz | -0.032 | 0.009 | — |

**nu = 14.23** [95% CI: 4.2, 32.8]: Close to Gaussian, but the Student-t provides robustness against outlier pairs.

**sigma0 = 3.00%**: Baseline uncertainty when all features are at their training-set mean — a "typical" pair.

**beta_dz = -0.057% per metre**: Higher WTG elevation produces slight underprediction bias. dz=+10m -> mu = -0.057% bias.

## 9. Variance Decomposition

Contribution of each feature to `var[log(sigma)]`:

| Feature | % of total |
|---------|-----------|
| Saturating \|turning\| | 34.6% |
| Saturating \|roughness\| | 33.9% |
| Saturating distance | 14.6% |
| WM \|log speedup ratio\| | 7.1% |
| Saturating \|dz\| | 3.3% |
| Baseline (unexplained) | 6.6% |

Features explain 93.4% of log(sigma) variance. Turning and roughness together account for 68.5%.

## 10. LOO Cross-Validation Results

Two holdout strategies, both using sigma-only predictions:

**Pair-level holdout** (19 physical pairs, 38 directions):

| Metric | Value |
|--------|-------|
| Pearson r | 0.926 |
| Spearman rho | 0.746 |
| Bias | -0.98 pp |
| Mean predicted sigma | 3.67% |
| Mean actual \|e\| | 4.65% |

**Site-level holdout** (all pairs from same location removed):

| Metric | Value |
|--------|-------|
| Pearson r | 0.908 |
| Spearman rho | 0.750 |
| Bias | -0.69 pp |
| Mean predicted sigma | 3.96% |
| Mean actual \|e\| | 4.65% |

The model slightly underpredicts uncertainty on average (negative bias), more so in pair-level holdout. The saturating transforms on turning and roughness (added May 2026) improved all LOO metrics compared to the raw-feature model.

## 11. Dataset

| Property | Value |
|----------|-------|
| Directional pairs | 38 |
| Physical pairs | 19 |
| Target e_overall mean | 0.16% |
| Target e_overall std | 6.60% |
| dz range | -116m to +116m (std = 56.1m) |

### Excluded Masts

| Mast IDs | Site | Reason |
|----------|------|--------|
| 2015WM018, 2021PA004, 2022PA008 | Sallachy | Hills and valleys — complex terrain beyond WAsP limits |
| 2022PA018 | Kayislar | Only 2 directions dominate energy |
| 2011WM011, 2014WM011 | Hultema | Persistent outlier despite investigation |
| 2019HE001, 2019HE002, 2019HE003 | Herzhausen CFD | Mixing CFD and WAsP results is not comparable |
| 2022PA021 | Taaibos | 3-mast location, removed 1 mast for model fit |
| 2023PA085 | Ukhanda | 2 mast + 1 LiDAR location, removed 1 mast |
| 2024PA014 | Balver Wald | Removed 1 measurement; site may need full removal |
| 2012WM006 | Malarberget | Persistent outlier despite investigation |
| 2024PA107 | Slovenska East | No displacement height data available |

## 12. MCMC Settings

| Setting | Full model | LOO folds |
|---------|-----------|-----------|
| Chains | 4 | 2 |
| Tune | 2000 | 1000 |
| Draws | 2000 | 1000 |
| target_accept | 0.95 | 0.95 |
| Sampler | NUTS | NUTS |

---

## 13. Development History — Decisions and Experiments

### Model Level: Sector -> Pair

The original model (`ws_uncertainty_model_final.py`) trained on sector-level errors (432 observations from 36 directional pairs). It achieved Pearson 0.903 on sector-level LOO. However, the sector-level target was wrong for EYA: sector errors (~30% spread) partially cancel when aggregated to omnidirectional, and the EYA-relevant quantity is the overall error (~5%). The sector model overestimated uncertainty by ~1.8x on average.

The pair-level model trains directly on the correct target: `e_overall`, the frequency-weighted signed error with cancellation. This reduced the training set from 432 sector observations to 38 pair observations, but the predictions are directly comparable to what matters for EYA.

### Sector Model Aggregation: Clerc et al. (2012)

Before committing to the pair-level model, an attempt was made to salvage the sector model by aggregating sector predictions to pair level using the Clerc et al. (2012) inter-sector correlation structure:

```
rho_ij = max(1 - |delta_theta| / 90, 0)
sigma_overall^2 = sum_i sum_j w_i w_j rho_ij sigma_i sigma_j
```

This hybrid approach (train sectors, aggregate with correlation) achieved Pearson 0.721 / Spearman 0.405 — better than independent quadrature but substantially worse than the pair-level model. Conclusion: the pair-level model is the better approach.

### Feature Evolution

**From sector model (3 features):**
- `log(distance_m / distance_A)` — clamped log distance
- `turning_gradient` — max neighbour difference per sector
- `speedup_diff_std` — std of speedup difference across sectors

**To pair model (5 features, current):**
- `dist_sat` — saturating distance (no hard clamp)
- `turning_sat` — saturating transform of frequency-weighted mean |turning| (1 - exp(-t/3.0))
- `wm_abs_log_speedup` — log ratio of overall speedups (symmetric, naturally compressed)
- `roughness_sat` — saturating transform of frequency-weighted mean |roughness| (1 - exp(-r/0.01))
- `dz_sat` — saturating height difference

**Saturating transforms (May 2026):** Turning and roughness were originally raw values (`wm_abs_turning`, `wm_abs_roughness`). Testing on a flat site at 15 km revealed inflated predictions (~22% uncertainty) caused by roughness z-scores of ~5.8 (training mean 0.004 vs site's 0.017). Applying `1 - exp(-x/scale)` transforms to both features bounded them to [0,1] and improved all LOO metrics (Pair Spearman 0.704 -> 0.746, Site Spearman 0.696 -> 0.750). The log-speedup feature was not saturated because the log transform already compresses its range naturally.

**Features tested and removed:**
- `k_MM` (Weibull shape parameter at MM site): Was active in a 7-feature version. Removed because it added complexity without meaningful improvement and had NaN coverage issues in some datasets.
- `dz_trix` (|dz| x TRIX/100 interaction): Terrain complexity interaction with height difference. Removed — dz_sat alone captured the effect.
- `wstd_turning` (weighted std of |turning|): Redundant with mean |turning|.
- `rough_dissimilarity` (|rough_WTG - rough_MM|): Tested as alternative to average roughness. Less predictive.
- `concentration_ratio` (inverse Simpson index of energy distribution): Did not improve predictions.
- `trix_norm` (TRIX/100): Terrain complexity alone was not predictive after distance and dz captured similar information.

### Orographic vs Overall Speedup

Tested isolating the orographic speedup component (`orog_speedup_WTG_frac_new / orog_speedup_MM_frac_new`) instead of the combined overall speedup (orographic x roughness). Results dropped from Pearson 0.905 to 0.842. The combined overall speedup factor works better — the orographic and roughness components overlap complementarily rather than redundantly.

### Sector Aggregation Weight Experiments

Three weighting schemes were tested for aggregating sector features to pair level:

1. **`weight_energy_predicted`** (energy weights): Original choice. Model-dependent, asymmetric between A->B and B->A.
2. **`freq_MM`** (MM Weibull frequency): Final choice. Purely observed, no model dependence, symmetric.
3. **`Sample_count_pred`** (predicted sample counts): Tested via `FEATURE_WEIGHT_MODE = "pred"` toggle. Gave marginal improvement (Pearson 0.905 -> 0.912) but requires additional WAsP output at deployment.

Turning always uses `freq_MM` regardless of mode — predicted sample counts depend on turning angles, creating circular dependence.

### dz_sat Saturation Scale Sweep

The scale constant in `dz_sat = 1 - exp(-|dz|/scale)` was swept via LOO CV (`sweep_dz_sat_scale.py`):

| Scale | Pearson | Spearman | Bias |
|-------|---------|----------|------|
| 10 | 0.9005 | 0.7183 | -0.90% |
| 20 | 0.9036 | 0.7046 | -0.89% |
| 30 | 0.9051 | 0.7030 | -0.89% |
| **40** | **0.9046** | **0.7046** | **-0.87%** |
| 60 | 0.9070 | 0.7094 | -0.86% |
| 80 | 0.9073 | 0.7094 | -0.85% |
| 120 | 0.9076 | 0.7321 | -0.87% |

Pearson varies by only 0.007 across the full range — the feature is robust to this choice. Scale=40 was kept because larger values compress the useful dz range (0-70m for most sites) into a narrow band where the feature loses discriminating power. At scale=40, a 50m difference gives 0.71; at scale=80, the same 50m gives only 0.47.

### log_sigma0 Prior Change

Changed from `Normal(-2.5, 0.5)` (sector model, ~8% baseline) to `Normal(-3.9, 0.5)` (~2% baseline). The pair-level target has much smaller magnitude after cross-sector cancellation, so the prior needed to shift downward.

### Excluded Masts Evolution

The excluded mast list evolved significantly between the sector and pair models. Key changes:
- Added Sallachy (2021PA004 specifically) — hills and valleys terrain
- Added Malarberget — persistent outlier
- Added Slovenska East (2024PA107) — missing displacement height data
- Removed broad exclusions (Sundern, Doringbaai now included)
- Reduced Balver Wald from 3 masts to 1 (2024PA014 only)

---

## 14. Output Files

| File | Contents |
|------|----------|
| `ws_uncertainty_pairlevel_results.json` | Model params, gamma posteriors, variance decomposition, scalers, LOO summary, excluded masts |
| `ws_uncertainty_pairlevel_idata.nc` | Full MCMC trace (ArviZ InferenceData) |
| `ws_uncertainty_pairlevel_loo_pair_results.csv` | Per-pair LOO predictions (pair-level holdout) |
| `ws_uncertainty_pairlevel_loo_site_results.csv` | Per-pair LOO predictions (site-level holdout) |
| `ws_uncertainty_pairlevel_loo_pair_cv.html` | Interactive LOO scatter plot (pair holdout) |
| `ws_uncertainty_pairlevel_loo_site_cv.html` | Interactive LOO scatter plot (site holdout) |
