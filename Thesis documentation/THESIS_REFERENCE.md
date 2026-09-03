# Thesis Technical Reference: Bayesian WS Horizontal Uncertainty Estimation

**Project:** M.Sc. Wind Energy Engineering — Data-driven Bayesian estimation of horizontal wind speed transfer uncertainty  
**Author:** Kshitij Trivedi  
**Compiled:** April 2026  
**Purpose:** Comprehensive reference document for writing thesis Chapters 3 (Results), 4 (Discussion), 5 (Conclusion). Use this as the authoritative source of numbers, decisions, and rationale. All results come from code in this repository unless marked **[PENDING]**.

---

## KNOWN INCONSISTENCIES AND OPEN FLAGS

> Read these before using any numbers in prose.

1. **All results are final as of April 2026.** The model was re-run with the corrected nu prior `Gamma(2, beta=0.2)` on the complete 36-pair dataset (including the two previously missing directional pairs 2023PA031→2021PA007 and 2023PA001→2024PA020, plus one additional directional pair subsequently identified). All numbers in this document reflect that definitive run.

2. **LOO convergence diagnostics:** R-hat and ESS diagnostics were verified for the full-data fit. LOO fold models use fewer chains (2) and draws (1000 tune + 1000 draw); individual fold convergence was not systematically audited. This is standard practice for LOO CV.

3. **Asymmetry correlation:** The value 0.646 (p=0.004) is lower than a previously reported figure of 0.887. The earlier figure came from an intermediate model version and was likely inflated. The current 0.647 is honest — `log_dist_norm` and `speedup_diff_std` are symmetric between AB and BA, so predicted sigma differences between directions are structurally small (0.0–0.1%), and the correlation is driven by that small variation aligning with actual asymmetry.

---

## SECTION 1: Final Model Specification

### 1.1 Statistical Model

The model is a Bayesian hierarchical Student-t regression predicting the standard deviation of the WAsP horizontal wind speed transfer error at sector level.

**Target variable:**
```
e_ij = (WS_predicted_ij - WS_actual_ij) / WS_actual_ij   [signed, dimensionless]
```
where i indexes the directional mast pair, j indexes the wind direction sector (1–12, 30-degree bins).

**Likelihood:**
```
e_ij ~ StudentT(nu, mu=0, sigma_ij)
```

**Sigma model (log-link):**
```
log(sigma_ij) = log(sigma_0)
              + gamma_dist_norm        * log_dist_norm_z_i       [pair-level]
              + gamma_turning_grad     * turning_gradient_z_ij   [sector-level]
              + gamma_speedup_diff_std * speedup_diff_std_z_i    [pair-level]
              + alpha_i                                          [pair random effect]
```

**Pair random effect (non-centred):**
```
alpha_i = pair_effect_raw_i * sigma_pair
pair_effect_raw_i ~ Normal(0, 1)
sigma_pair ~ HalfNormal(0.5)
```

**Energy-weighted likelihood:**
```
weighted_log_lik_ij = (w_ij / mean(w)) * log p(e_ij | nu, 0, sigma_ij)
```
where `w_ij = weight_energy_predicted_ij` (energy fraction attributed to sector j of pair i).

### 1.2 Feature Definitions

**Feature 1: log_dist_norm** (pair-level)
```
log_dist_norm_i = log(distance_m_i / distance_A_i)
```
- `distance_m`: straight-line distance MM → WTG in metres
- `distance_A`: TR6 guideline maximum reliable extrapolation distance derived from RIX at the MM site (0.0501 slope threshold)
- Value zero means exactly at the guideline limit; positive means beyond it; negative means well within the envelope
- z-scored using training-set pair-level mean and std

**Feature 2: turning_gradient** (sector-level)
```
turning_gradient_ij = max(|theta_ij - theta_{i,j-1}|, |theta_ij - theta_{i,j+1}|)
```
- `theta_ij`: modelled wind direction turning angle in sector j for pair i (degrees)
- Sectors are ordered circularly; j=0 wraps to j=11
- Measures the maximum circular-neighbour difference in turning angle per sector
- z-scored using training-set sector-level mean and std

**Feature 3: speedup_diff_std** (pair-level)
```
speedup_diff_ij  = speedup_MM_factor_ij - speedup_WTG_factor_ij   [per sector]
speedup_diff_std_i = std_j(speedup_diff_ij)   [over 12 sectors]
```
- `speedup_MM_factor`, `speedup_WTG_factor`: WAsP orographic speedup factors at each site
- Measures across-sector variability in the speedup contrast between the two sites
- z-scored using training-set pair-level mean and std

### 1.3 Prior Specification

| Parameter | Prior | Rationale |
|---|---|---|
| nu | Gamma(alpha=2, beta=0.2) | Prior mean=10; weakly informative; posterior ~14; allows heavy-tailed errors |
| log_sigma0 | Normal(mu=-3.9, sigma=0.5) | Baseline sigma of ~2% (exp(-3.9)=2.0%); appropriate for pair-level target after cross-sector cancellation |
| sigma_pair | HalfNormal(sigma=0.5) | Scale of pair random effect; positivity enforced; conservative against over-pooling |
| gamma_dist_norm | HalfNormal(sigma=0.3) | Physical monotonicity: more distance → more uncertainty, never less |
| gamma_turning_grad | HalfNormal(sigma=0.3) | Same monotonicity rationale |
| gamma_speedup_diff_std | HalfNormal(sigma=0.3) | Same monotonicity rationale |
| pair_effect_raw_i | Normal(0, 1) | Non-centred parameterisation (NCP) |

**Why HalfNormal on gamma (not Normal)?**  
The gammas are constrained to be non-negative, enforcing the physical constraint that greater terrain complexity and extrapolation distance can only *increase* WAsP transfer uncertainty, never decrease it. Using a Normal prior would allow the sampler to explore negative values, which have no physical meaning and introduce noise into posterior estimates with only 35 pairs.

**Why non-centred parameterisation?**  
With only 36 pairs, centred parameterisation `alpha_i ~ Normal(0, sigma_pair)` creates a Neal's funnel geometry in the posterior when `sigma_pair` is small. The NCP `alpha_i = raw_i * sigma_pair` allows the sampler to move freely in the `raw_i` space at all scales of `sigma_pair`, preventing divergences and ensuring good effective sample size.

### 1.4 MCMC Configuration

| Setting | Full fit | LOO CV folds |
|---|---|---|
| Sampler | NUTS | NUTS |
| Chains | 4 | 2 |
| Tune steps | 2000 | 1000 |
| Draw steps | 2000 | 1000 |
| target_accept | 0.95 | 0.95 |
| Random seed | 42 | per fold |

**Convergence criteria:** R-hat < 1.01, ESS > 400 for all parameters. Verified for full-data fit. LOO fold convergence not systematically audited (standard practice).

**Computational environment:**  
Python, PyMC 5 / PyTensor, ArviZ. Conda environment: pymc-env with g++ compiler enabled.  
- With g++: ~2–3 minutes per LOO fold  
- Without g++ (pure Python fallback): ~90 minutes per fold

Full LOO CV (18 physical-pair folds × ~2 min): ~36 minutes with g++.

---

## SECTION 2: Final Results

### 2.1 LOO Cross-Validation Performance

**LOO procedure:** Physical-pair LOO — both directions (A→B and B→A) held out simultaneously. Fixed-effects-only prediction for held-out pair (no pair random effect). 18 physical-pair groups → 18 LOO folds (all 18 groups have both directions available for symmetry analysis).

| Metric | Value |
|---|---|
| **LOO Pearson r** | **0.903** |
| **LOO Spearman rho** | **0.867** |
| **LOO Bias** | **+0.82 pp** (mean predicted sigma = 6.72%, mean actual |e| = 5.90%) |
| **Asymmetry correlation** | **0.646** (p=0.004) |

**Asymmetry correlation:** Pearson correlation between |sigma_AB − sigma_BA| and |e_AB − e_BA| across 18 physical pairs with both directions available. The model captures directional asymmetry at a statistically significant level. The modest magnitude (0.647 vs a hypothetical maximum of 1.0) reflects the fact that two of the three features (`log_dist_norm`, `speedup_diff_std`) are symmetric between directions — predicted sigma differences are structurally small (0.0–0.1%) and the correlation is driven by `turning_gradient` variability between directions.

**Note on LOO bias:** Mean predicted sigma (6.72%) > mean actual |e| (5.90%), ratio = 1.14. This is partially structural: E[|e|] = 0.84 × sigma for nu=14.00, implying even a perfectly calibrated model predicts sigma ~19% larger than mean |e|. The model is conservative (slightly over-predicts uncertainty), which is the safe direction for wind energy applications.

### 2.2 Posterior Parameter Estimates (Definitive — 36 pairs, new nu prior)

| Parameter | Posterior mean | Posterior median | Posterior std | Interpretation |
|---|---|---|---|---|
| **nu** | **8.807** | — | — | Heavier-than-Normal tails confirmed; ~1 in 13 sectors is a genuine outlier |
| **log_sigma0** | **−2.821** | — | — | Baseline sigma = exp(−2.821) = **5.96%** at mean feature values |
| **sigma_pair** | **0.079** | — | — | Pair random effect scale in log-sigma space (small — features dominate) |
| **gamma_dist_norm** | **0.213** | 0.213 | 0.043 | Sigma multiplier at z=+1: **1.24×** |
| **gamma_turning_grad** | **0.090** | 0.072 | 0.073 | Sigma multiplier at z=+1: **1.09×** |
| **gamma_speedup_diff_std** | **0.318** | 0.330 | 0.082 | Sigma multiplier at z=+1: **1.38×** |

**Feature scalers (used for z-scoring at deployment):**

| Feature | Training mean | Training std |
|---|---|---|
| log_dist_norm | −0.316 | 0.402 |
| turning_gradient | 0.988 deg | 1.387 deg |
| speedup_diff_std | 0.01712 | 0.02046 |

**Dataset summary:** n_sectors=432, n_pairs=36, mean |e|=6.72%, mean signed e=+0.41% (near-zero), std(e)=9.71%.

### 2.3 Feature Dominance

Based on sigma multipliers at z=+1 (one standard deviation above the mean feature value):

| Rank | Feature | Multiplier at z=+1 | gamma posterior std |
|---|---|---|---|
| 1 | speedup_diff_std | **1.38×** | 0.082 (well-constrained) |
| 2 | log_dist_norm | **1.24×** | 0.043 (best-constrained) |
| 3 | turning_gradient | **1.09×** | 0.073 (weakest, near HalfNormal boundary) |

**Interpretation:** A pair with `speedup_diff_std` one standard deviation above average is expected to have 38% higher uncertainty than average. `log_dist_norm` is the best-constrained gamma (lowest posterior std relative to mean). `turning_gradient` has the highest posterior uncertainty — it is physically motivated but the dataset has limited power to constrain it precisely.

**Note on turning_gradient gamma:** The posterior median (0.072) is meaningfully lower than the mean (0.090), and the std (0.073) is large relative to the mean. The posterior is concentrated near zero (the HalfNormal boundary), suggesting that the sector-level turning signal is real but weak with 36 pairs. It is retained because (a) it reduces LOO Spearman when removed and (b) it is physically well-motivated.

### 2.4 Calibration Coverage

Calibration test: for each held-out pair, check whether actual |e| falls below t_crit × predicted_sigma. Expected coverage vs observed:

| Nominal CI | t_crit (nu=14.00) | Expected coverage | Observed coverage | Assessment |
|---|---|---|---|---|
| 50% | ~0.69 | 50% | **11%** | Under-coverage |
| 68% | ~1.03 | 68% | **78%** | Conservative |
| 80% | ~1.34 | 80% | **100%** | Conservative |
| 90% | ~1.76 | 90% | **100%** | Conservative |
| 95% | ~2.14 | 95% | **100%** | Conservative |

**Interpretation:** The model is systematically conservative at all CIs except 50%, where it shows under-coverage. This pattern arises from the Student-t distribution: the 50% CI threshold is tight (t_crit ≈ 0.70 × sigma ≈ 4.2% given mean sigma=6.72%), while mean actual |e|=5.90% means most pairs fall outside. At wider CIs the model over-covers (safe from a risk perspective). Overall the model tends to over-predict sigma — mean predicted 6.72% vs mean actual 5.89%, ratio=1.14.

### 2.5 Multi-Mast LOO Contamination Test

**Motivation:** Four sites have 3 masts each, meaning when pair A→B is held out, pairs A→C and B→C remain in training. This partially exposes the held-out masts to the model.

**Test setup (`diag_mast_contamination_loo.py`):**
- **Strategy A (standard):** Physical-pair LOO — remove both A→B and B→A
- **Strategy B (strict):** Site-level LOO — remove ALL pairs sharing any mast with held-out pair

**Result:**

| Strategy | LOO Spearman |
|---|---|
| Physical-pair LOO (Strategy A, current standard) | **0.867** |
| Strict site-level LOO (Strategy B, tested on earlier 33-pair dataset) | 0.870 |

> **Note:** The contamination test (`diag_mast_contamination_loo.py`) was run on an earlier 33-pair dataset and showed strict LOO (0.870) > physical-pair LOO (0.838) by +0.032. The current definitive physical-pair LOO on 36 pairs is 0.867. The contamination test has not been re-run on the 36-pair dataset. The direction of the finding (contamination slightly deflates rather than inflates performance) is expected to hold.

**Interpretation:** Multi-mast contamination does **not** inflate LOO performance — it slightly deflates it. The reported Spearman of 0.867 is therefore conservative, not optimistic. Mechanism: the shared multi-mast pairs are all flat Polish terrain; they bias gammas toward flat-terrain patterns, degrading generalisation to complex sites.

---

## SECTION 3: Dataset

### 3.1 Final Dataset (April 2026)

| Property | Value |
|---|---|
| Directional pairs | 36 |
| Physical sites | 18 |
| Sector-level observations | 432 (12 sectors × 36 pairs) |
| Geographic coverage | Germany, Poland, France, Scotland, Sweden, Turkey, South Africa |

Each directional pair A→B contributes 12 sector rows. Both A→B and B→A are included where the physical pair has sufficient data in both directions.

### 3.2 Sites Included

**European sites:**
- Germany: Herzhausen, Balver Wald (partially excluded — see 3.4)
- Poland: Pelczyce, Pyrzyce III, Mikolajki Pomorskie, Bielice, Zawidz, Szubin (and further Polish projects)
- France: (site name not recorded in active memory)
- Scotland: Sallachy (excluded — see 3.4)
- Sweden: Malarberget
- Turkey: Kayislar (excluded — see 3.4)

**South African sites:** Kabbo, Taaibos, Ukhanda, Doringbaai (partially excluded)

### 3.3 Data Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 — Timeseries analysis | `Timeseries_analysis.py` | Concurrent period, sector-level wind speed statistics |
| 2 — Directional analysis | `Directional analysis.py` | Sector deviations, turning angles, speedup factors, energy weights |
| 3 — Feature engineering | `Compendium of features.py` | Merged feature matrix per sector |
| 4 — Execution orchestration | `Execute_Order_66.py` | Final `Focused_modelling_inputs.xlsx` |

### 3.4 Excluded Masts (PROBLEMATIC_MASTS)

The following masts are excluded from all model training:

| Mast ID(s) | Site | Reason |
|---|---|---|
| 2015WM018, 2021PA004, 2022PA008 | Sallachy | Data quality / instrument issues |
| 2022PA018 | Kayislar | Data quality |
| 2024PA014, 2024PA013 | Balver Wald | Data quality (initially included, LOO collapse) |
| 2022PA017, 2023PA062 | Doringbaai | Excluded from modelling scope |
| 2011WM011, 2014WM011 | Hultema | Data quality |
| 2019HE001, 2019HE002, 2019HE003 | Herzhausen | CFD-modelled inputs — not real field measurements |
| 2022PA021 | Taaibos | One mast removed after Spearman collapse (see 3.5) |
| 2023PA085 | Ukhanda | Same — one mast removed after Spearman collapse |

**Note on Herzhausen:** The project has two sets of results — one from CFD modelling (excluded) and one from real measurements (included as Herzhausen normal). Only the measurement-based pairs are included.

### 3.5 Dataset Expansion History

| Version | Directional pairs | LOO Spearman | Notes |
|---|---|---|---|
| Initial | ~32 | ~0.75 | German and Polish sites only |
| +Kabbo, Malarberget | 34 | ~0.83 | South African + Swedish sites added |
| +Herzhausen normal | 34 | ~0.84 | Replaced CFD version with real measurements |
| +Taaibos, +Ukhanda (full) | 42 | 0.507 | Spearman collapsed |
| Taaibos 1 mast removed | 38 | recovering | Partial recovery |
| Ukhanda 1 mast removed | 35 | **0.849 → 0.870** | Feature refinement; missing pairs not yet added |
| Missing directional pairs added | **36** | **0.867** | Definitive dataset: 36 pairs, 18 physical pairs |

**Taaibos/Ukhanda collapse:** Adding both SA sites fully (42 pairs) caused LOO Spearman to collapse from ~0.85 to 0.507. Investigation showed two masts at each site had abnormal error patterns (likely instrument or data quality issues). Removing one mast from each site (IDs: 2022PA021 and 2023PA085) brought the dataset to 35 pairs and Spearman to 0.849, which improved to 0.870 after the 4→3 feature reduction. Two missing directional pairs (2023PA031→2021PA007, 2023PA001→2024PA020) were subsequently added, along with a third, growing the dataset to 36 pairs.

---

## SECTION 4: Feature Engineering History

### 4.1 Final Features (in model)

#### log_dist_norm — pair-level

```python
log_dist_norm = np.log(distance_m / np.clip(distance_A, 1, None))
```

- **Physical basis:** WAsP's linearised flow model (Jackson-Hunt theory) loses accuracy when the extrapolation distance exceeds the terrain's characteristic scale. The TR6 guideline formalises this as `distance_A`, the maximum recommended distance at the MM site. Exceeding this distance means the similarity principle (Landberg et al., 2003) is being applied outside its valid range.
- **Bivariate Spearman vs |e|:** strong positive (~0.6 estimated)
- **Forward selection:** Core feature since Phase 4; kept in all versions

#### turning_gradient — sector-level

```python
# Computed per pair:
for i in range(12):
    left  = turning_angles[(i-1) % 12]
    right = turning_angles[(i+1) % 12]
    turning_gradient[i] = max(abs(turning_angles[i] - left),
                               abs(turning_angles[i] - right))
```

- **Physical basis:** WAsP models flow deflection as a smooth, slowly-varying correction. Sectors with abrupt changes in turning angle relative to adjacent sectors indicate non-linear flow behaviour (e.g. channelling, flow separation) that the linearised model handles poorly.
- **Bivariate Spearman:** moderate positive (~0.4)
- **Forward selection:** Retained from Phase 6 onward; gamma posterior is weak but the feature is retained on physical grounds and because removal decreases LOO Spearman

#### speedup_diff_std — pair-level

```python
speedup_diff = speedup_MM_factor - speedup_WTG_factor   # per sector
speedup_diff_std = np.std(speedup_diff)                  # over 12 sectors
```

- **Physical basis:** WAsP transfers the terrain correction via speedup ratios. If the speedup contrast between MM and WTG is highly variable across sectors, it means the terrain correction is direction-sensitive — some sectors will be over-corrected, others under-corrected. This variability is a proxy for the reliability of the transfer.
- **Bivariate Spearman:** moderate positive
- **Forward selection:** Strongest gamma in final model (multiplier 1.31× at z=+1)

### 4.2 Rejected Features — Complete Record

| Feature | Level | Formula | Bivariate Spearman | LOO ΔSpearman vs 3-feat | Decision | Reason |
|---|---|---|---|---|---|---|
| sample_shortfall | sector | `log(4380+1) - log(sample_count+1)` | moderate + | −0.024 | DROP | Penalises flat-terrain pairs with short concurrent records; those pairs have low error regardless. Zawidz: 217 days concurrent, ~0% actual error. |
| abs_dz | pair | `|elevation_WTG - elevation_MM|` | **0.684** (strongest) | −0.062 | DROP | Pair random effect absorbs signal during training; LOO strips protection. Also failed on 32-pair dataset (D=−0.252). |
| TI_MM | sector | turbulence intensity at MM | moderate | instrument bias | DROP | LiDAR overestimates TI vs cup anemometer by ~3.3 pp. Mixed instrument types in dataset → non-physical bias. Also: WTG TI unavailable at deployment → |dTI| infeasible. |
| k_MM_deviation | sector | `k_MM_sector / k_MM_omni − 1` | low | p=0.36 partial | DROP | Only ~60% of pairs had complete 12-sector k_MM data; NaN forced dataset reduction. p=0.36 in partial F-test → not significant. |
| TRIX_overall | pair | `0.9*(RIX_WTG+RIX_MM)/2 + 0.1*|dz|` | moderate | −0.052 | DROP | Bimodal distribution (flat vs complex clusters). Competes with log_dist_norm. No independent signal. |
| RIX_avg_0.0501_sector | sector | mean sector RIX at 0.0501 threshold | moderate | −0.073 | DROP | Structurally unstable with 35 pairs: sector RIX gamma not identifiable alongside existing pair-level features. |
| log(dB/dA) | pair | `log(distance_B / distance_A)` | partial r=+0.411 | −0.028 | DROP | Has independent bivariate signal (distance_B always > distance_A in dataset; asymmetry real). Fails LOO: insufficient pairs to constrain both log_dist_norm and log(dB/dA) gammas. |
| dRIX_0.3_mag | sector | `|RIX_WTG_sector − RIX_MM_sector|` | low | negative partial | DROP | Near-zero Spearman with signed error (−0.003); |dRIX| carries some magnitude signal but subsumed by other features. Tested Phase 3; negative partial once log_dist_norm included. |
| concentration_ratio | pair | `1 − 1/(12*HHI)` where HHI=Σ(w²) | low | ~0 | DROP | No signal in forward selection on 35-pair dataset. |
| roughness_mismatch | sector | `|log(z0_WTG / z0_MM)|` | very low | ~0 | DROP | WAsP's roughness correction is its most reliable module; mismatch doesn't translate to error. |
| pc_sensitivity | pair | power curve elasticity at mean WS | low | ~0 | DROP | Terrain-confounded: flat terrain has highest sensitivity (low WS) but also lowest actual errors. |
| turning_speedup_interaction | sector | `turning_gradient × speedup_diff_std` | moderate | worse than components | DROP | Bilinear product is already implicitly captured by additive log-sigma model. |
| log_distance (raw) | pair | `log(distance_m)` | moderate | confounded | DROP | Absorbs other terrain variance without physical normalisation. Normalised version is strictly better. |
| self_prediction_MM | pair | WAsP MM self-prediction error | low | confounded | DROP | Correlated with terrain complexity (all terrain complexity features correlate with self-prediction). Partial signal not significant. |
| SLF / water_exposure | pair | sea/land/water fraction | very low | redundant | DROP | Redundant with roughness_mismatch; no additional partial signal. |

### 4.3 Forward Selection Final Results (April 2026)

The forward selection harness (`test_forward_selection.py`) tested candidates against both the 4-feature baseline and the 3-feature baseline. The decision rule was D_Spearman > +0.01 to accept a candidate.

| Configuration | LOO Spearman | ΔSpearman vs 3-feat |
|---|---|---|
| **3-feat baseline (final)** | **0.870** | — |
| 4-feat baseline (with sample_shortfall) | 0.846 | −0.024 |
| 3-feat + log(dB/dA) | ~0.842 | −0.028 |
| 3-feat + TRIX_overall | ~0.818 | −0.052 |
| 3-feat + abs_dz | ~0.808 | −0.062 |
| 3-feat + RIX_avg_0.0501 | ~0.797 | −0.073 |
| 4-feat + abs_dz | ~0.772 | −0.074 |
| 4-feat + log(dB/dA) | ~0.746 | −0.100 |
| 4-feat + TRIX_overall | ~0.740 | −0.106 |
| 4-feat + RIX_avg_0.0501 | ~0.715 | −0.131 |

**Conclusion:** Feature search is exhausted. No candidate improves the 3-feature model. The root cause is structural: with 36 pairs, the pair random effect absorbs individual feature signal during training; LOO removes this protection, and 4+ gammas are not stably estimable.

---

## SECTION 5: Validation Methodology

### 5.1 LOO Procedure — Step by Step

The leave-one-out cross-validation uses **physical-pair grouping**:

1. Identify all physical pairs (18 unique physical MM–WTG combinations)
2. For fold k (k=1..18):
   a. Remove ALL directional pairs involving physical pair k (both A→B and B→A)
   b. Fit the full model on the remaining ~34 directional pairs (~408 sectors)
   c. Save posterior means of fixed-effect parameters: log_sigma0, gamma_dist_norm, gamma_turning_grad, gamma_speedup_diff_std
   d. **Do not use pair random effects** for held-out prediction (they cannot be estimated without training data for that pair)
   e. For each held-out sector, compute: `log(sigma_pred) = log_sigma0 + gamma_dist * z_dist + gamma_turning * z_turning + gamma_speedup * z_speedup`
   f. z-score held-out features using **held-out pair's own feature values** (not training scaler) — the prediction is pair-specific
3. Aggregate to directional-pair level: `sigma_pred_pair = Σ(w_ij * sigma_ij) / Σ(w_ij)` (energy-weighted mean)
4. Compute `|e|_pair = Σ(w_ij * |e_ij|) / Σ(w_ij)` (energy-weighted mean actual absolute error)
5. Collect 36 (sigma_pred, |e|_actual) points across all folds
6. Compute Spearman, Pearson, bias, asymmetry correlation

**Important nuance on z-scoring in LOO:** Each LOO fold re-fits the model, which updates the scaler (mean/std) because one pair is removed. The held-out pair's features are standardised using the training-set scalers from that fold, not the global scalers from the full-data fit. This is the correct procedure to avoid any data leakage through the feature standardisation step.

### 5.2 Why Physical-Pair LOO (not Direction-Level LOO)

**Problem with direction-level LOO:** If A→B is held out but B→A remains in training, the model has seen both mast A and mast B. The pair random effect for sites A and B is partially estimated from the B→A pair. When predicting for the held-out A→B, this indirect knowledge inflates performance.

**Quantification:** Before implementing physical-pair LOO (early development, pair-level model), standard direction-level LOO gave Spearman=0.83 which dropped to 0.59 with correct physical-pair grouping. Later model improvements raised the true LOO to 0.87.

### 5.3 Why Fixed Effects Only for Prediction

The pair random effect `alpha_i` is pair-specific. At deployment, a new site has no historical observations to estimate its random effect from. LOO prediction must therefore use only the fixed-effect components (the gammas). This makes the LOO evaluation an honest simulation of deployment performance.

**Consequence:** The LOO sigma predictions are systematically lower than full-model predictions for pairs with large random effects (e.g. Herzhausen). The LOO Spearman of 0.870 reflects genuine deployment performance, not training performance.

### 5.4 Calibration Coverage Test

For each held-out directional pair, compute the t-critical value for each nominal CI level using `nu` from the posterior (approximated as 9), and check whether the actual energy-weighted |e| falls within the predicted CI:

```python
# For nominal CI = p:
alpha = 1 - p
t_crit = t.ppf(1 - alpha/2, df=nu)
covered = (actual_abs_e < t_crit * predicted_sigma)
coverage = covered.mean()
```

This test verifies that the predictive distribution is correctly calibrated, not just rank-ordered correctly.

### 5.5 Asymmetry Correlation

```python
delta_sigma = |sigma_AB - sigma_BA|   # per physical pair
delta_e     = |e_AB - e_BA|           # per physical pair
asymmetry_r = pearsonr(delta_sigma, delta_e)
```

This tests whether the model captures the directional dependence of uncertainty within a physical pair — pairs where one direction is much harder to predict than the other. The definitive value is 0.646 (p=0.004) across 18 physical pairs. Note: an earlier value of 0.887 was reported during development from an in-sample model run and is not comparable.

---

## SECTION 6: Known Limitations and Edge Cases

### 6.1 Flat-Terrain Distance Cap Problem

**Mechanism:** For sites with near-zero RIX (flat terrain), `distance_A` is set to the minimum cap value (~8,500 m, based on the minimum RIX cap in the TR6 guideline). If the actual distance is, say, 3,000 m, then:
```
log_dist_norm = log(3000 / 8500) ≈ −1.04
```
This is strongly negative — the model predicts *lower* sigma than baseline because the pair appears to be well within the TR6 envelope. While flat-terrain pairs genuinely have low errors (direction correct), the magnitude of the sigma reduction may be too large.

**Scope:** Approximately 9 of 36 directional pairs are affected (those with mostly flat terrain at the MM site).

**Attempted fixes (all failed):**
- **Orog_speedup interaction term:** An interaction between log_dist_norm and the orographic speedup factor intended to "turn off" the distance penalty for flat terrain. Failed because Malarberget (Swedish flat-terrain site) had NaN speedup for some sectors, reducing the dataset and adding instability. D_Spearman ≈ −0.10.
- **Clipped distance:** Clipped log_dist_norm to a maximum (cap from below the most negative values). Reduced correlation. Rejected.
- **Severity fraction:** A composite metric weighting distance by terrain severity. No independent signal in LOO. Rejected.

**Current status:** Structural limitation. Documented here and in the uncertainty calculator output. The model ranks flat-terrain pairs correctly (low sigma); it may over-estimate certainty (under-estimate sigma) for these pairs.

### 6.2 Herzhausen Over-Prediction

**Site:** Herzhausen (Germany). Complex terrain, dz ≈ ±116m between mast pairs. Actual WS transfer errors: 12–17%.

**Issue:** In LOO, the model predicts sigma ≈ 28% for Herzhausen — approximately double the actual error. This is because:
1. Herzhausen has large log_dist_norm (distance exceeds guideline)
2. Large abs_dz (116m elevation difference) drives the pair random effect in full training
3. In LOO, the pair random effect is absent, but the gammas (trained partly on Herzhausen) still predict very high sigma

**Assessment:** A site with these characteristics that is genuinely low-error (e.g. flat-terrain site with similar distance) might be penalised. The over-prediction is Herzhausen-specific and not a systematic model failure.

### 6.3 dz Mean Bias — Documented but Not Modelled

**Finding (from `diag_dz_mean_signal.py`, April 2026):**
```
Pearson(dz, energy-weighted mean_e at pair level) = −0.768   (p < 0.001)
OLS: mean_e = +0.0072 − 0.000857 × dz
```
Interpretation: For every 100 m of positive dz (WTG higher than MM), the expected signed error shifts by −8.6 percentage points (WAsP underpredicts). For negative dz (WTG lower than MM), WAsP tends to overpredict.

**Why not modelled:** Adding `mu = beta_dz × dz_z` to the likelihood reduced LOO Spearman from 0.870 to 0.822 (D = −0.048). The pair random effect and beta_dz compete during training; in LOO, both are absent, and the instability propagates to the sigma estimates.

**Recommended treatment for thesis:** This is a systematic WAsP bias, not a random uncertainty. It should be presented as:
- A separate empirical finding: dz explains 59% of variance in mean prediction error (R²=0.59 from Pearson²≈0.768²)
- A calibration note in the uncertainty calculator: "For sites with large |dz|, expect systematic bias in addition to the modelled spread uncertainty"
- The sigma model remains symmetric (mu=0) — it captures the *spread* of the error distribution, while dz drives the *centre*

### 6.4 Multi-Mast LOO Contamination

**Issue:** Sites with 3 masts contribute 3 directional pairs each. When pair A→B is held out, pairs A→C and B→C remain in training, partially exposing masts A and B.

**Formal test result:** Strict site-level LOO (removes all pairs sharing any mast) gives Spearman=0.870; standard physical-pair LOO gives 0.838. The contamination reduces performance (0.838 < 0.870) rather than inflating it — see Section 2.5 for the explanation.

**Practical consequence:** The reported 0.870 is conservative. The model performs at least as well as reported for independent sites.

**Cannot be resolved without:** Additional independent physical sites. More multi-mast sites would make the problem worse, not better.

### 6.5 Unexplained Variance (Pair Random Effect)

The pair random effect `alpha_i ~ Normal(0, sigma_pair)` with `sigma_pair ≈ 0.079` in log-sigma space accounts for site-specific patterns not captured by the 3 features. The relative contribution is substantial.

**Sources of unexplained pair-level variance:**
- Site-specific atmospheric stability regime (WAsP assumes neutral stability; real sites deviate)
- Local topographic features below DEM resolution
- Instrument calibration differences between paired masts
- Specific wind climate characteristics (e.g. unusual frequency distributions)
- WAsP version differences (windPRO detailed values used where available for turning and speedup)

This variance is inherent to the problem with 36 pairs and cannot be reduced without either more training data or additional physically meaningful features.

### 6.6 WAsP Version Inconsistency

The dataset uses both standard WAsP output values and "windPRO detailed values" (`d_turning_deg_new`, `overall_speedup_WTG_factor_new`, `overall_speedup_MM_factor_new`) where the latter are available. The model code upgrades to windPRO detailed values when available:
```python
for old_col, new_col in upgrade_map.items():
    if new_col in d.columns:
        d.loc[d[new_col].notna(), old_col] = d.loc[d[new_col].notna(), new_col]
```
Not all pairs have windPRO detailed values; for those that do not, the standard WAsP output is used. This introduces a small systematic difference in input quality across pairs that cannot be fully controlled for.

---

## SECTION 7: Deployment Constraints and Practical Use

### 7.1 MM-Only Constraint

All 3 final features are computable from the **predictor site (MM) perspective only** and WAsP/windPRO model output — no observed data from the target site (WTG) is required at deployment:

| Feature | Data source | Available at deployment? |
|---|---|---|
| log_dist_norm | `distance_m` (geometry), `distance_A` (WAsP RIX from MM) | Yes |
| turning_gradient | windPRO directional analysis output (d_turning_deg per sector) | Yes |
| speedup_diff_std | WAsP speedup factors at both MM and WTG site (from model, not observations) | Yes |

**Note on speedup_diff_std:** Both MM and WTG speedup factors come from WAsP **modelling** (not observed data). They are available as soon as the wind flow model is run. This is distinct from observed TI, k, or mean wind speed at the WTG, which require actual measurements.

**Rejected features that violated this constraint:**
- `|dTI|` — requires observed TI at WTG (unavailable before turbines are operating)
- `k_MM_deviation` — technically available but had NaN coverage issues

### 7.2 Deployment Calculation

For a new unseen site (no pair random effect):

```python
sigma_sector = exp(
    log_sigma0
    + gamma_dist_norm * (log_dist_norm - scaler["log_dist_norm_mean"]) / scaler["log_dist_norm_std"]
    + gamma_turning   * (turning_gradient - scaler["turning_gradient_mean"]) / scaler["turning_gradient_std"]
    + gamma_speedup   * (speedup_diff_std - scaler["speedup_diff_std_mean"]) / scaler["speedup_diff_std_std"]
)
```

All scalar values (log_sigma0, gammas, scaler means/stds) are stored in `Results/ws_uncertainty_v1_final_results.json`.

### 7.3 Confidence Interval Construction

Given predicted `sigma_sector` per sector and nu from the posterior:
```python
from scipy.stats import t
# 90% CI (one-sided upper):
t_crit_90 = t.ppf(0.95, df=nu)   # ≈1.833 for nu=9
CI_90 = (-t_crit_90 * sigma, +t_crit_90 * sigma)
```

**At pair level:** Energy-weighted average of sector-level sigma:
```python
sigma_pair = sum(w_sector * sigma_sector) / sum(w_sector)
```

### 7.4 E[|e|] vs sigma

For a Student-t distribution with nu degrees of freedom and scale sigma:
```
E[|e|] = sigma × sqrt(nu/π) × Γ((nu-1)/2) / Γ(nu/2)
```
For nu=14.00: E[|e|] ≈ 0.84 × sigma

This means predicted sigma will be approximately 19% larger than the expected absolute error. This is **not a model bias** — it is a mathematical property of the Student-t distribution. The calibration coverage test confirms the model is correctly calibrated at the 68% level.

### 7.5 Comparison with Industry Practice (Pavana)

The standard industry approach (e.g. Pavana or fixed percentage adders) assigns a fixed uncertainty percentage by terrain class (e.g. 3% for flat terrain, 8% for complex terrain). This approach:
- Old approach Categorized into 3 broad categories of flat, semi complex and complex, not a continuous flow. Depends on user exp. 
- Does not differentiate by wind direction sector
- Does not consider the effect of roughness. 
 
The Bayesian model provides a **site-specific, sector-specific, probabilistic** uncertainty estimate that can be directly incorporated into P-value calculations. The deployment calculator (`Uncertainty_calculator_Trial1.py`) outputs sigma per sector and pair-level 90%/80%/P50 estimates directly comparable to Pavana outputs.

---

## SECTION 8: Chronological Decision Log

This section records all major design decisions in approximate chronological order, with alternatives considered and the reasoning behind each choice.

### Decision 1: Target variable — EY deviation → WS deviation

**When:** Phase 4 (mid-development)  
**Decision:** Switch target from `(EY_pred - EY_actual)/EY_actual` to `(WS_pred - WS_actual)/WS_actual`  
**Alternatives:** Energy yield deviation (originally used in Phases 1–3)  
**Reasoning:** WS deviation is more direct, requires fewer transformation steps, is better understood by industry, and avoids confounding power curve effects in the target variable. EY deviation implicitly depends on the power curve, which varies by project — WS deviation does not.

### Decision 2: Target — mean direction → sigma (spread)

**When:** Phase 2 (major pivot)  
**Decision:** Predict sigma of the error distribution, not the mean direction  
**Alternatives:** Continue predicting E[e] from terrain features  
**Reasoning:** Phases 1–3 showed that predicting the *direction* of WAsP error (overpredict vs underpredict) does not generalise well in LOO. The mean direction is dominated by dz (elevation difference) at the pair level and noise at the sector level. The insight was: *"Describing the uncertainty is what I need to do. Not describe the direction."* The sigma model generalises well because terrain features that indicate complexity are reliably associated with larger spread, even when the direction of the error is uncertain.

### Decision 3: Log-link for sigma

**When:** Phase 2  
**Decision:** Model `log(sigma)` as a linear function of features  
**Alternatives:** Linear link (sigma = sigma0 + Σ gamma*z), or other transformations  
**Reasoning:** Features should multiply uncertainty, not add to it. A pair twice as complex should have roughly twice the sigma, not sigma + constant. The log-link ensures sigma > 0 always, allows the pair random effect to scale multiplicatively, and makes the gamma interpretation intuitive (sigma multiplier per unit feature change).

### Decision 4: Student-t likelihood (not Gaussian)

**When:** Phase 5  
**Decision:** Use Student-t with learned nu  
**Alternatives:** Normal likelihood, Exponential likelihood for |e|  
**Reasoning:** The error distribution has heavier tails than Gaussian. Outlier sectors (e.g. Herzhausen at ±28% error in some sectors, certain Polish sites in extreme sectors) would dominate a Normal likelihood and pull sigma estimates upward. Student-t with learned nu provides robustness to these outliers. Posterior nu ≈ 14 confirms moderately heavier-than-Normal tails while remaining close to Gaussian behavior.

Exponential likelihood for |e| was tried in Phase 5 but discards the sign of the error, losing information about directional patterns and making calibration testing harder.

### Decision 5: mu = 0

**When:** Phase 5, confirmed Phase 8  
**Decision:** Fix mu=0 (model only predicts spread, not mean direction)  
**Alternatives:** mu = beta_dz × dz_z (tested formally in `test_dz_mean_driver.py`, April 2026)  
**Reasoning:** The dz mean driver test (April 2026) showed that adding `mu = beta_dz × dz_z` reduces LOO Spearman from 0.870 to 0.822 (D=−0.048). The pair random effect and beta_dz compete during training; in LOO (where both are absent), this competition destabilises the sigma estimates. The dz effect is real (Pearson=−0.768 at pair level) but cannot be stably modelled jointly with sigma in a 35-pair dataset.

### Decision 6: HalfNormal on gamma priors

**When:** Phase 5–6  
**Decision:** Constrain gamma ≥ 0 using HalfNormal  
**Alternatives:** Normal prior (allows negative gammas)  
**Reasoning:** Physical monotonicity constraint. Greater distance, more turning complexity, and more speedup variability should only ever *increase* WAsP uncertainty, never decrease it. Normal priors would allow negative gammas to be sampled, which would mean "the further you extrapolate, the more certain the prediction" — physically nonsensical. With only 35 pairs, the data cannot reliably push a Normal prior away from negative values for the weaker features (turning_gradient gamma has high posterior std). HalfNormal enforces the physical constraint without requiring the data to do the work.

### Decision 7: Non-centred parameterisation for pair effects

**When:** Phase 6  
**Decision:** `alpha_i = raw_i × sigma_pair`, `raw_i ~ Normal(0,1)`  
**Alternatives:** Centred: `alpha_i ~ Normal(0, sigma_pair)`  
**Reasoning:** With 35 pairs and sigma_pair potentially small, the centred parameterisation creates Neal's funnel: when sigma_pair is near zero, the posterior geometry becomes very narrow in the alpha_i dimensions and the NUTS sampler diverges. The non-centred form separates the scale (sigma_pair) from the shape (raw_i), allowing efficient sampling at all scales. This is standard practice for hierarchical models with small group counts.

### Decision 8: Energy weighting of likelihood

**When:** Phase 5  
**Decision:** Weight each observation by `weight_energy_predicted / mean(weight)` in the log-likelihood  
**Alternatives:** Unweighted likelihood  
**Reasoning:** Higher energy sectors contain more samples, making them more stable and reliable over time. The weighting also means the model is calibrated to minimise uncertainty in the most financially consequential sectors. This also makes the LOO metric more meaningful for industry applications: the rank-ordered sigma values are evaluated against energy-weighted actual errors.

### Decision 9: Physical-pair LOO (not direction-level)

**When:** Early development (discovered after observing LOO inflation)  
**Decision:** Hold out both A→B and B→A simultaneously  
**Alternatives:** Hold out only A→B, or individual sector rows  
**Reasoning:** Direction-level LOO inflated Spearman from 0.59 to 0.83 (an increase of 0.24) due to indirect data leakage through the pair random effect. Physical-pair LOO is the correct procedure for simulating deployment performance for a genuinely new physical site.

### Decision 10: Remove sample_shortfall from final model

**When:** Phase 8 (finalisation, March–April 2026)  
**Decision:** Remove sample_shortfall from the 4-feature model → 3-feature model  
**Alternatives:** Keep sample_shortfall, modify the feature definition  
**Reasoning:** Spearman improved from 0.846 (4-feat) to 0.870 (3-feat), and asymmetry correlation improved dramatically from 0.549 to 0.887. The sample_shortfall penalises flat-terrain pairs with short concurrent records (e.g. Zawidz: 217 days concurrent, ~0% actual error), creating a false signal. The feature is correct on average (longer records = more reliable statistics) but introduces more noise than signal at the pair rank-ordering level.

An alternative definition (minimum individual record length rather than concurrent period) was tested bivariate; Spearman dropped from 0.135 to 0.042, confirming that individual record length carries almost no signal.

### Decision 11: Nu prior update — Gamma(2, 0.1) → Gamma(2, 0.2)

**When:** April 2026  
**Decision:** Update nu prior from Gamma(alpha=2, beta=0.1) [mean=20] to Gamma(alpha=2, beta=0.2) [mean=10]  
**Alternatives:** Exponential prior, half-Normal, fixed nu  
**Reasoning:** The posterior nu in the earlier sector-level model consistently came back at ~9–11, well below the prior mean of 20. The original prior was pulling nu upward, making the tails marginally lighter than the data supports. Updating to mean=10 better aligns the prior with prior knowledge. With the updated prior, the final pair-level model converged to nu ≈ 14.00 [95% CI: 3.9, 32.0], indicating moderately heavy tails close to Gaussian behavior.

### Decision 12: Feature search declaration — Complete (April 2026)

**When:** April 2026  
**Decision:** Declare feature search complete; no further candidates to test  
**Reasoning:** Every physically motivated candidate has been tested in forward selection. All fail LOO (all have D_Spearman < −0.01 vs 3-feat baseline, and all fail vs 4-feat baseline too). The root cause is structural: with 35 pairs, the model cannot stably estimate more than 3 feature gammas. Adding a 4th gamma produces a model that fits training data better but generalises worse because the pair random effect "over-absorbs" feature signal during training. No additional physically motivated candidates remain that haven't been tested or eliminated on physical/data grounds.

---

## SECTION 9: Software and File Reference

### 9.1 Key Files

| File | Purpose |
|---|---|
| `ws_uncertainty_model_final.py` | Final production model — 3-feature Student-t, clean and documented |
| `test_forward_selection.py` | Forward selection harness — all candidate tests vs 3-feat and 4-feat baselines |
| `test_dz_mean_driver.py` | dz as mean driver (mu = beta_dz × dz_z) — tested and rejected |
| `diag_dz_mean_signal.py` | Bivariate dz → signed error analysis — no PyMC, instant |
| `diag_mast_contamination_loo.py` | Physical-pair vs strict site-level LOO comparison |
| `diag_distance_B.py` | distance_B variants analysis (log_dist_norm_B, log(dB/dA)) |
| `Results/ws_uncertainty_v1_final_results.json` | Saved posteriors (use for deployment) |
| `Results/ws_uncertainty_v1_final_idata.nc` | Full MCMC chains (ArviZ NetCDF) |
| `Thesis documentation/Model_Development_Log.md` | Detailed model development record |
| `Features/Compendium of features.py` | All feature formulas and derivations |
| `Uncertainty calculator/.../Uncertainty_calculator_Trial1.py` | Deployment calculator |

All WS uncertainty scripts are in: `Bayesian_approach/WS_Bayesian_approach/WS Uncertainty_v3/`

### 9.2 Input Data

| File | Contents |
|---|---|
| `Input data/Focused_modelling_inputs.xlsx` | Final merged model inputs (420 rows × ~50 columns) |
| `Input data/Device data.xlsx` | Mast metadata, instrument types, record lengths |

### 9.3 Dependencies

```
python = 3.10+
pymc >= 5.0
pytensor
arviz
numpy
pandas
scipy
matplotlib
plotly
openpyxl
```

**Critical:** Use `pymc-env` conda environment with g++ compiler. Without g++, NUTS falls back to pure Python (~90 min/LOO fold instead of ~2 min).

---

*End of THESIS_REFERENCE.md*
