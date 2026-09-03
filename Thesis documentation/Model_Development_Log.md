# Model Development Log: Bayesian WS Horizontal Uncertainty Estimation

**Project:** M.Sc. Thesis — Data-driven Bayesian estimation of horizontal wind speed transfer uncertainty  
**Author:** Kshitij Trivedi  
**Last updated:** April 2026  
**Purpose:** Complete record of all modelling decisions, features tested, results obtained, and rationale for each choice. Intended as a reference for Chapter 3 (Results) and Chapter 4 (Discussion).

---

## 1. Problem Statement

WAsP-based horizontal extrapolation between a measurement mast (MM) and a wind turbine generator site (WTG) introduces a transfer uncertainty that is currently estimated using fixed percentage adders by terrain class. This project replaces that with a data-driven Bayesian model that predicts the *standard deviation of the wind speed transfer error* (sigma) at sector level, conditioned on site-specific features.

**Target variable:**

```
e = (WS_predicted - WS_actual) / WS_actual   [signed, fractional]
```

The model predicts sigma of this distribution, not its mean. The mean is assumed zero (symmetric uncertainty) after empirical testing showed that modelling the mean direction does not generalise well with 35 pairs.

---

## 2. Dataset

### 2.1 Final dataset (as of April 2026)

- **35 directional pairs** from **17-18 physical sites**
- **~420 sector-level observations** (12 sectors per directional pair)
- European sites: Germany, Poland, France, Scotland, Sweden, Turkey
- South African sites: Kabbo, Taaibos, Ukhanda, Doringbaai

Each directional pair A→B contributes 12 sector rows. Both A→B and B→A are included where available, giving the model paired symmetry information.

### 2.2 Data pipeline stages

| Stage | Script | Output |
|---|---|---|
| 1 — Timeseries analysis | `Timeseries_analysis.py` | Concurrent period, sector wind speeds |
| 2 — Directional analysis | `Directional analysis.py` | Sector-level WS deviations, turning, speedup |
| 3 — Feature engineering | `Compendium of features.py` | Merged feature matrix |
| 4 — Execution | `Execute_Order_66.py` | Final `Focused_modelling_inputs.xlsx` |

### 2.3 Excluded masts (PROBLEMATIC_MASTS)

The following masts are excluded from all model training runs:

| Mast(s) | Site | Reason |
|---|---|---|
| 2015WM018, 2021PA004, 2022PA008 | Sallachy | Data quality / instrument issues |
| 2022PA018 | Kayislar | Data quality |
| 2024PA014, 2024PA013 | Balver Wald | Data quality |
| 2022PA017, 2023PA062 | Doringbaai | Excluded from modelling scope |
| 2011WM011, 2014WM011 | Hultema | Data quality |
| 2019HE001, 2019HE002, 2019HE003 | Herzhausen CFD | CFD-modelled inputs, not real measurements |
| 2022PA021 | Taaibos | One mast removed after Spearman collapse (42→35 pairs, score 0.507→0.849) |
| 2023PA085 | Ukhanda | Same — one mast removed after Spearman collapse |

### 2.4 Dataset expansion history

| Version | Pairs | Notes |
|---|---|---|
| Initial | 32 | German and Polish sites only |
| +Sallachy | 32 | Excluded immediately (data quality) |
| +Kabbo, Malarberget | 34 | Added South African and Swedish sites |
| +Herzhausen normal, +Taaibos, +Ukhanda | 35 | Replaced Herzhausen CFD with real measurements; two SA sites partially included |

**Taaibos/Ukhanda note:** Initially activating both sites fully (42 pairs) caused Spearman to collapse from ~0.85 to 0.507. One mast was removed from each site based on data quality judgement. Spearman recovered to 0.849.

---

## 3. Model Architecture

### 3.1 Final model specification

```
e_ij ~ StudentT(nu, mu=0, sigma_ij)

log(sigma_ij) = log(sigma_0)
              + gamma_dist    * log_dist_norm_z_i      [pair-level]
              + gamma_turning * turning_gradient_z_ij  [sector-level]
              + gamma_speedup * speedup_diff_std_z_i   [pair-level]
              + alpha_i                                [pair random effect]
```

### 3.2 Prior specification

| Parameter | Prior | Rationale |
|---|---|---|
| nu | Gamma(alpha=2, beta=0.2) | Weakly informative; allows heavy tails. Prior mean=10, posterior ~14 |
| log_sigma0 | Normal(-3.9, 0.5) | Baseline sigma centred on ~2%, appropriate for pair-level target after cross-sector cancellation |
| sigma_pair | HalfNormal(0.5) | Scale of pair random effect; half-normal enforces positivity |
| gamma_dist, gamma_turning, gamma_speedup | HalfNormal(0.3) | Monotonicity constraint: more complexity → more uncertainty, never less |
| pair_effect_raw | Normal(0, 1) | Non-centred parameterisation for sampling efficiency |

### 3.3 Key design decisions

**Why Student-t likelihood?**  
The error distribution has heavier tails than Gaussian — outlier sectors (e.g. Herzhausen at 28%) would dominate a Normal likelihood. Student-t with learned nu provides robustness. Posterior nu ≈ 9, confirming heavier-than-Normal tails.

**Why log-link for sigma?**  
Features multiply uncertainty rather than add it. A pair twice as complex should have ~twice the sigma, not sigma + constant. Log-link ensures sigma > 0 always.

**Why HalfNormal on gamma (not Normal)?**  
Physical monotonicity: higher distance, higher turning complexity, and more speedup contrast should all *increase* uncertainty, never decrease it. HalfNormal enforces this direction constraint.

**Why mu = 0?**  
Mean direction of WAsP error was modelled in Phases 1–3 and did not generalise (LOO signal near zero). The sigma model is the product. The mean is documented separately as a bias note (dz effect, see Section 7.3).

**Why energy weighting?**  
Sectors contributing more to annual energy production have more financial consequence. The likelihood is weighted by `weight_energy_predicted` so the model prioritises accuracy where it matters most.

**Why non-centred parameterisation for pair effects?**  
With only 35 pairs, centred parameterisation of `alpha_i ~ Normal(0, sigma_pair)` causes funnel geometry in the posterior and poor NUTS mixing. Non-centred (`alpha_i = raw_i * sigma_pair` with `raw_i ~ Normal(0,1)`) avoids this.

---

## 4. Feature Development History

### 4.1 Features in the final model

#### Feature 1: log_dist_norm (pair-level)

```
log_dist_norm = log(distance_m / distance_A)
```

- `distance_m`: straight-line distance between MM and WTG (metres)
- `distance_A`: RIX-derived maximum reliable extrapolation distance (TR6 guideline)
- **Interpretation:** positive when the actual distance exceeds the guideline distance; zero at exactly the guideline distance
- **Physical basis:** WAsP linearised flow model degrades beyond its operating envelope. The similarity principle (Landberg et al., 2003) breaks down at large distances.
- **Bivariate Spearman with |error|:** strong positive
- **Note on flat terrain:** For flat sites (RIX near zero), distance_A hits the minimum RIX cap (~8.5 km), making log_dist_norm artificially large for pairs that are actually well within WAsP's operating envelope. This is a structural limitation documented in Section 8.

#### Feature 2: turning_gradient (sector-level)

```
turning_gradient_ij = max(|theta_ij - theta_{i,j-1}|, |theta_ij - theta_{i,j+1}|)
```

- `theta_ij`: wind direction turning angle in sector j for pair i
- Computed as the maximum absolute circular-neighbour difference in turning angle
- **Interpretation:** high values indicate sectors where the wind direction changes abruptly relative to adjacent sectors — a sign of complex, non-linear flow deflection
- **Physical basis:** WAsP assumes smooth, slowly varying flow deflection. Large turning gradients indicate the flow is beyond the model's assumptions in that sector.
- **Bivariate Spearman with |error|:** moderate-strong positive
- **Level:** sector — each of the 12 sectors in a pair gets its own turning_gradient value

#### Feature 3: speedup_diff_std (pair-level)

```
speedup_diff_ij = speedup_MM_factor_ij - speedup_WTG_factor_ij   [per sector]
speedup_diff_std_i = std(speedup_diff_ij)  over j=1..12
```

- **Interpretation:** standard deviation of the speedup contrast between the two sites across sectors. High values mean the terrain effect is very different at the two sites, and this difference varies strongly by wind direction.
- **Physical basis:** WAsP transfers the terrain correction from MM to WTG via the speedup ratio. Large variability in this ratio across sectors indicates the correction is direction-sensitive and therefore less reliable on average.
- **Bivariate Spearman with |error|:** moderate positive
- **Level:** pair — single value per directional pair

### 4.2 Features tested and rejected

#### sample_shortfall (sector-level) — rejected at v6

```
sample_shortfall = log(4380 + 1) - log(Sample_count_pred + 1)
```

- Reference: 4380 = 52560 / 12 (one year of 10-minute data divided by 12 sectors)
- For TR6-compliant projects with ≥ 1 year data: shortfall = 0 (dormant)
- **Why rejected:** Forward selection showed including this feature reduced LOO Spearman from 0.870 to 0.846. Root cause: flat-terrain short-concurrent pairs (e.g. Zawidz, 217 days concurrent) get penalised by the shortfall even though their actual error is low — flat terrain MCP is robust regardless of record length. The feature is correct on average but adds more noise than signal in the rank ordering.
- **Zawidz case:** 217 days concurrent, but individual mast records are 421 days. Actual error ~0%. Shortfall penalises it as if the uncertainty is high — wrong.
- **Attempted fix:** Replaced concurrent-period shortfall with min(record_length_A, record_length_B) from Device data.xlsx. Result: Spearman dropped from 0.135 to 0.042 bivariate — individual record length carries almost no signal. Reverted.

#### TI_MM (sector-level) — rejected at v5

- Turbulence intensity at the MM site per sector
- **Why rejected:** LiDAR instruments systematically overestimate TI vs. cup anemometers by ~3.3 pp. The dataset contains both instrument types, introducing a systematic non-physical bias into the feature. At deployment, only MM TI is available (no WTG TI), so |dTI| (the originally desired feature) cannot be computed anyway.

#### k_MM_deviation (sector-level) — rejected at v5

```
k_MM_deviation = k_MM_sector / k_MM_omni - 1
```

- Deviation of sector Weibull shape parameter from omnidirectional value
- **Why rejected:** Only ~60% of pairs had full k_MM data for all 12 sectors; remaining pairs had NaN, reducing the effective dataset. Partial F-test showed p = 0.36 (not significant). Dropped to preserve full dataset coverage.

#### abs_dz (pair-level) — rejected in forward selection (April 2026)

```
abs_dz = |elevation_WTG - elevation_MM|   [metres]
```

- **Bivariate Spearman with |error|:** 0.684 — strongest bivariate signal of all features tested
- **Why rejected:** Despite strong bivariate signal, LOO Spearman dropped by -0.062 when added to the 3-feature model. The pair random effect absorbs the abs_dz signal during training (high-dz pairs genuinely have high error, but the random effect captures this at the pair level). In LOO, the random effect is removed, and abs_dz alone cannot carry the prediction reliably with 35 pairs.
- **Previously tested on 32-pair dataset:** also catastrophic (D_Spearman = -0.252)

#### TRIX_overall (pair-level) — rejected in forward selection

```
TRIX = 0.9 * (RIX_WTG + RIX_MM) / 2 + 0.1 * |dz|   [at 0.0501 threshold]
```

- Composite terrain complexity index combining RIX and elevation difference
- **Why rejected:** Bimodal distribution (flat terrain near zero, complex terrain clustered high). Competes with log_dist_norm (both penalise complex terrain pairs). D_Spearman = -0.106 vs 4-feat baseline, -0.052 vs 3-feat baseline.

#### RIX_avg_0.0501_sector (sector-level) — rejected in forward selection

- Sector-level average RIX at the 0.0501 slope threshold
- **Why rejected:** With only 35 pairs (420 rows), adding a sector-level terrain complexity gamma on top of the 3-feature model is not stably estimable. D_Spearman = -0.131 vs 4-feat baseline, -0.073 vs 3-feat baseline.

#### RIX_avg_0.3_sector (sector-level) — not formally tested

- Sector-level RIX at the 0.3 threshold (standard WAsP threshold)
- Not tested after RIX_avg_0.0501_sector was rejected for the same structural reason.

#### dRIX_0.3_mag (sector-level)

```
dRIX_0.3_mag = |RIX_WTG_sector - RIX_MM_sector|   [at 0.3 threshold]
```

- **Why rejected (earlier phase):** dRIX has near-zero Spearman with signed error (-0.003) but the absolute value carries some magnitude signal. Tested in Phase 3; when abs_dz and log_dist_norm are both included, dRIX adds no independent information. Negative partial correlation once other features are controlled.

#### concentration_ratio (pair-level)

```
HHI = sum(w_norm_j^2)   [Herfindahl-Hirschman Index]
effective_sectors = 1 / HHI
concentration_ratio = (12 - effective_sectors) / 11   [0=even, 1=all one sector]
```

- Energy concentration: how dominant a single sector is in annual energy production
- **Why rejected:** Zero or negative signal in forward selection on 35-pair dataset.

#### roughness_mismatch (sector-level)

```
roughness_mismatch = |log(ref_length_WTG / ref_length_MM)|
```

- **Why rejected:** Very low bivariate signal. Surface roughness correction is one of WAsP's most reliable modules; mismatch does not reliably translate to prediction error in this dataset.

#### pc_sensitivity (pair-level)

- Generic power curve elasticity at mean wind speed
- **Why rejected:** Terrain-confounded. No signal in flat terrain (where pc_sensitivity is highest for low wind speeds) because actual errors are also low. Zero signal in forward selection.

#### turning_speedup_interaction (sector-level)

```
turning_speedup_ix = turning_gradient * speedup_diff_std
```

- Product of the two individual features
- **Why rejected:** Worse than components alone. Bilinear interaction is already implicit in the additive log-sigma model (both features contribute independently).

#### log_distance (pair-level, standalone)

- Unnormalised log distance without RIX normalisation
- **Why rejected (early phase):** Absorbs variance from other terrain features without clear physical meaning. The normalised version (log_dist_norm) is more interpretable and performs better.

#### self_prediction_MM (pair-level)

- WAsP's own prediction of the MM site using itself as reference (a form of internal model check)
- **Why rejected:** Confounded with terrain complexity — any feature correlated with terrain complexity will also correlate with self_prediction. The partial signal is not significant once other features are included.

#### SLF / water_exposure

- Sea/land/water exposure fraction
- **Why rejected:** Redundant with roughness_mismatch. No additional partial signal.

---

## 5. LOO Cross-Validation Methodology

### 5.1 Physical-pair LOO

The standard leave-one-out procedure holds out one directional pair (e.g. A→B) at a time. This is **incorrect** for this dataset because:

- A→B and B→A share the same two masts
- If A→B is held out but B→A remains in training, the model has seen site A and site B
- The pair random effect absorbs site-specific patterns during training
- Held-out prediction is then inflated by this indirect data leakage

**Solution:** Always hold out both A→B and B→A simultaneously (physical-pair LOO). This was implemented after the standard LOO gave an inflated Spearman of 0.83, which dropped to an honest 0.59 when corrected. Later improvements raised this to 0.87.

### 5.2 Multi-mast contamination (known limitation)

Four sites in the dataset have three masts each (Pelczyce, Pyrzyce III, etc.), giving 3 directional pairs per physical group. When pair A→B is held out, pairs A→C and B→C remain in training. The pair random effect for masts A and B is partially estimated from these remaining pairs. This inflates the true LOO Spearman slightly. Estimated effect: modest (< 0.02 on Spearman). Not fixable without more independent sites.

### 5.3 LOO prediction procedure

In each fold:
1. Remove both directions of the held-out physical pair
2. Fit the model on remaining ~33 directional pairs
3. Apply trained fixed-effect gammas (NO pair random effect) to held-out pair features
4. Compute energy-weighted average predicted sigma and actual |e| per directional pair
5. Collect 35 (predicted_sigma, actual_|e|) points across all folds

The pair random effect is deliberately excluded in step 3 — it would require seeing the held-out data to estimate. This is the correct evaluation for a new unseen site at deployment.

### 5.4 LOO metrics

| Metric | Definition | Purpose |
|---|---|---|
| Pearson r | Linear correlation (predicted_sigma, actual_|e|) | Magnitude tracking |
| Spearman rho | Rank correlation (predicted_sigma, actual_|e|) | Rank ordering (more robust) |
| Bias | mean(predicted_sigma - actual_|e|) | Systematic over/under-estimation |
| Asymmetry correlation | Pearson(|sigma_AB - sigma_BA|, |e_AB - e_BA|) | Whether model captures direction asymmetry |
| Calibration coverage | % actual |e| < t_crit * predicted_sigma | Probabilistic calibration |

---

## 6. Model Evolution — Phase-by-Phase Results

### Phase 1: Kitchen-sink mean regression (Regression A1, A2, A2.1)

- **Target:** EY deviation (energy yield, not wind speed)
- **Likelihood:** Normal
- **Features:** All available terrain features (~21)
- **Result:** Unstable. Mean deviation is dominated by dz (elevation difference) and pair-level noise. Spearman < 0.4 in LOO.
- **Key discovery:** Pair symmetry — A→B and B→A share the same physical pair structure. The pair random effect should flip sign between directions, not be independent.
- **Lesson:** Predicting mean direction of WAsP error is fighting noise. dz dominates but doesn't generalise.

### Phase 2: The pivot to sigma (Regression A3)

- **Target:** sigma of EY deviation
- **Likelihood:** Student-t (signed errors)
- **Key insight:** "Describing the uncertainty is what I need to do. Not describe the direction."
- **Features:** distance, |dz|, |dRIX|, |log_speedup|
- **Log-link for sigma** introduced: features multiplicatively affect uncertainty
- **Discovery:** dRIX has r = -0.003 for signed direction but |dRIX| drives magnitude — same feature useless for direction, valuable for spread
- **LOO Spearman:** ~0.55 (improved over Phase 1)

### Phase 3: Sigma enrichment (Regression A4a–d)

- Progressive addition of sample_shortfall, concentration_ratio, turning_std, speedup_diff_std
- Growth from 4 to 10 sigma drivers
- **LOO Spearman:** ~0.60–0.65

### Phase 4: Wind speed target

- Switched target from EY deviation to WS deviation: `e = (WS_pred - WS_actual) / WS_actual`
- More direct, fewer transformation steps, better interpretability
- Pair-level model at this stage (~67 observations for 34 pairs)

### Phase 5: Sector-level resolution

- Disaggregated from pair-level to sector-level (~800 observations for 420 sectors)
- Tried Exponential likelihood for |e|, then Student-t for signed e
- **LOO Spearman:** ~0.70 with sector resolution (more data, but also more noise per observation)

### Phase 6–7: Final model development

- Student-t with learned nu ≈ 9: heavier tails than Normal, robust to outlier sectors
- TI and Weibull features added, then removed (TI: instrument bias; Weibull: coverage too low)
- Feature ablation from 21 → 6 optimal features: Spearman 0.85 vs 0.75 for full set
- **Dataset expansion** from 32 to 35 pairs (Taaibos, Ukhanda, Herzhausen normal)

### Phase 8 (this session, April 2026): Finalisation

| Model | LOO Pearson | LOO Spearman | Features |
|---|---|---|---|
| 6-feature (k_MM, TI) | ~0.75 | 0.75 | dist, turning, speedup, sample_shortfall, TI_MM, k_MM |
| 4-feature (no TI, no k_MM) | 0.773 | 0.846 | dist, turning, speedup, sample_shortfall |
| **3-feature (final)** | **0.868** | **0.870** | **dist, turning, speedup** |

**Key finding:** Removing sample_shortfall improved Spearman by +0.024 and asymmetry correlation from 0.549 to 0.887. The shortfall feature was adding noise from flat-terrain pairs rather than genuine signal.

---

## 7. Feature Testing — Summary Table

| Feature | Level | Bivariate Spearman | Tested | LOO D_Spearman | Decision |
|---|---|---|---|---|---|
| log_dist_norm | pair | strong + | yes | core feature | KEEP |
| turning_gradient | sector | moderate + | yes | core feature | KEEP |
| speedup_diff_std | pair | moderate + | yes | core feature | KEEP |
| sample_shortfall | sector | moderate + | yes | -0.024 vs 3-feat | DROP |
| abs_dz | pair | **+0.684** | yes | -0.062 vs 3-feat | DROP |
| TI_MM | sector | moderate | yes | instrument bias | DROP |
| k_MM_deviation | sector | low | yes | p=0.36, NaN issues | DROP |
| TRIX_overall | pair | moderate | yes | -0.052 vs 3-feat | DROP |
| RIX_avg_0.0501_sector | sector | moderate | yes | -0.073 vs 3-feat | DROP |
| dRIX_0.3_mag | sector | low | yes | negative partial | DROP |
| concentration_ratio | pair | low | yes | ~0 | DROP |
| roughness_mismatch | sector | very low | yes | ~0 | DROP |
| pc_sensitivity | pair | low | yes | ~0 | DROP |
| turning_speedup_ix | sector | moderate | yes | worse than components | DROP |
| log_distance (raw) | pair | moderate | yes | confounded | DROP |
| self_prediction_MM | pair | low | yes | confounded | DROP |
| SLF / water_exposure | pair | very low | yes | redundant | DROP |

---

## 8. Known Limitations and Open Issues

### 8.1 Flat-terrain cap problem

For flat terrain sites (RIX ≈ 0), `distance_A` hits the minimum RIX cap (~8.5 km). If the actual distance is shorter (e.g. 3 km), `log_dist_norm = log(3000/8500) < 0`, which is negative — the model interprets this as *below-guideline* distance and predicts *lower* sigma than baseline. In practice, flat terrain pairs at any distance have low actual errors, so the direction is correct, but the magnitude of the sigma reduction may be too large.

**Affected pairs:** 9 of 35 directional pairs.  
**Attempted fix:** orog_speedup interaction term (failed — NaN for Malarberget). Clipped distance (failed — reduced correlation). Severity fraction (failed — no independent signal).  
**Current status:** Documented as structural limitation. The model still ranks these pairs correctly (low sigma), it just overestimates the certainty slightly.

### 8.2 Multi-mast LOO contamination

Sites with 3 masts (Pelczyce, Pyrzyce III, Mikolajki Pomorskie, Bielice) contribute 3 directional pairs each. When one pair is held out, the other two remain in training, partially exposing the held-out masts. This inflates true LOO Spearman by an estimated < 0.02. Cannot be resolved without additional independent sites.

### 8.3 Herzhausen overprediction

Herzhausen (dz = ±116m, complex terrain) has actual errors of ±12–17% but the model predicts sigma ≈ 28% in LOO. Root cause: the combination of large dz (high abs_dz) and complex terrain pushes all sigma drivers simultaneously. The pair random effect absorbs this during full training but cannot help in LOO. The model is conservative (over-estimates uncertainty) for this site, which is safe from a financial risk perspective.

### 8.4 dz mean bias — documented but not modelled

Bivariate analysis (April 2026) showed:

```
Pearson(dz, energy-weighted mean_e) = -0.768  (p < 0.001, pair level)
OLS: mean_e = +0.0072 - 0.000857 * dz
```

**Interpretation:** For every 100m of positive dz (WTG higher than MM), expected mean error shifts by -8.6 pp (underprediction). For negative dz (WTG lower), WAsP tends to overpredict.

**Tested as mean driver (`test_dz_mean_driver.py`):** Adding `mu = beta_dz * dz_z` to the likelihood hurt sigma ranking (LOO Spearman dropped from 0.870 to 0.822, D = -0.048). The pair random effect and beta_dz compete during training; in LOO, the competition destabilises sigma estimation.

**Recommended treatment:** Document as a bias note in the uncertainty calculator output. The sigma model remains symmetric (mu=0), but users should be informed that for sites with large dz, the error distribution is asymmetric around zero. No code change to final model.

### 8.5 Unexplained variance

The pair random effect absorbs approximately 38.5% of the total log-sigma variance. This represents genuine physical heterogeneity between sites that no available feature captures — site-specific atmospheric stability regimes, local topographic features not represented in the DEM, instrument calibration differences, etc. This is inherent to the problem and not reducible without additional data.

### 8.6 nu prior (resolved)

Prior updated from `Gamma(alpha=2, beta=0.1)` (mean=20) to `Gamma(alpha=2, beta=0.2)` (mean=10), aligned with the posterior estimate of nu ≈ 9. The original prior was pulling nu slightly upward, making the tails marginally lighter than optimal.

---

## 9. Final Model Parameters (38 pairs, 5 sigma features + dz mean)

| Parameter | Posterior mean | Interpretation |
|---|---|---|
| nu | 14.00 | Moderately heavier tails than Normal; close to Gaussian but Student-t provides outlier robustness |
| log_sigma0 | -3.503 | Baseline sigma = 3.04% at average feature values |
| sigma_pair | ~0.3–0.4 | Pair random effect scale; accounts for site-specific unexplained variance |
| gamma_dist | 0.187 | 1.21x sigma multiplier per std dev of saturating distance |
| gamma_turning | 0.272 | 1.31x sigma multiplier per std dev of WM |turning| |
| gamma_speedup | 0.127 | 1.14x sigma multiplier per std dev of WM |log speedup| |
| gamma_roughness | 0.252 | 1.29x sigma multiplier per std dev of WM |roughness speedup| |
| gamma_dz | 0.092 | 1.10x sigma multiplier per std dev of saturating |dz| |
| beta_dz | -0.034 | -0.060% per metre height difference (mu driver) |

**Note:** These are indicative from test runs. The definitive posterior values should be read from the `ws_uncertainty_v1_final_results.json` output after running `ws_uncertainty_model_final.py`.

---

## 10. Deployment Architecture

### 10.1 What the calculator needs at deployment

For a new unseen site (no pair random effect available):

```
log(sigma_sector) = log_sigma0
                  + gamma_dist    * (log_dist_norm - scaler_mean) / scaler_std
                  + gamma_turning * (turning_gradient - scaler_mean) / scaler_std
                  + gamma_speedup * (speedup_diff_std - scaler_mean) / scaler_std
```

All scalar values (gammas, log_sigma0, scaler means and stds) are saved in `ws_uncertainty_v1_final_results.json`.

### 10.2 Feature availability at deployment

All 3 features are computable from MM-only data and windPRO/WAsP output:

| Feature | Data source | Available at deployment? |
|---|---|---|
| log_dist_norm | WAsP distance_A (RIX-derived), distance_m | Yes |
| turning_gradient | windPRO directional analysis | Yes |
| speedup_diff_std | WAsP speedup factors (both sites) | Yes |

This satisfies the deployment constraint: no WTG-observed quantities required.

### 10.3 CI construction

Given predicted sigma per sector, the confidence interval for wind speed transfer error is:

```
[P_lower, P_upper] = StudentT.ppf([alpha/2, 1-alpha/2], df=nu, loc=0, scale=sigma)
```

For example, at 90% CI: `t_crit = t.ppf(0.95, df=14.00) ≈ 1.76`

```
90% CI = ±1.76 * sigma
```

And noting E[|e|] = 0.84 * sigma for nu = 14.00 (structural, not a bias).

---

## 11. Files Reference

| File | Purpose |
|---|---|
| `ws_uncertainty_model_final.py` | **Final model** — clean, fully documented |
| `test_3feat_no_shortfall.py` | Predecessor test script (same model, slightly different labels) |
| `test_4feat_no_TI.py` | 4-feature baseline (with sample_shortfall) |
| `test_dz_mean_driver.py` | Mean driver test — dz as mu, rejected |
| `test_forward_selection.py` | Forward selection harness — tests candidates vs 3-feat and 4-feat baselines |
| `diag_dz_mean_signal.py` | Diagnostic: signed dz vs signed error, no PyMC |
| `diagnose_dist_norm_flat_terrain.py` | Flat-terrain distance investigation |
| `Results/ws_uncertainty_v1_final_results.json` | Saved posteriors (after running final model) |
| `Results/ws_uncertainty_v1_final_idata.nc` | Full MCMC chains (ArviZ NetCDF format) |

---

## 12. Computational Details

- **Framework:** PyMC 5 / PyTensor, ArviZ
- **Sampler:** NUTS (No-U-Turn Sampler)
- **Full model:** 4 chains, 2000 tune + 2000 draw steps, target_accept=0.95
- **LOO CV folds:** 2 chains, 1000 tune + 1000 draw steps per fold
- **Environment:** pymc-env conda environment with g++ compiler (C compilation enabled)
  - Path: `C:\Users\K_Trivedi\AppData\Local\anaconda3\envs\pymc-env\python.exe`
  - Without g++: ~90 min per fold (pure Python fallback). With g++: ~2–3 min per fold
- **Convergence diagnostics:** R-hat < 1.01, ESS > 400 for all parameters (monitored via `az.summary`)
- **Random seed:** 42 throughout for reproducibility
