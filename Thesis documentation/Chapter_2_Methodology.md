# Chapter 2: Methodology

## 2.1 Research Design Overview

This chapter presents the complete methodological framework developed for estimating horizontal wind flow modelling uncertainty using a Bayesian hierarchical approach. The research follows an empirical, iterative, data-driven model development philosophy in which each successive modelling phase was motivated by concrete findings -- often surprising -- from the previous phase. The final model did not emerge from a single theoretical blueprint; rather, it crystallised through seven distinct phases of development, each of which refined the modelling target, the feature set, the distributional assumptions, or the validation strategy.

The empirical foundation of this work rests on a dataset of approximately 22 physical measurement pairs drawn from commercial wind resource assessment campaigns across Northern Europe. Each physical pair consists of two meteorological masts (or remote sensing devices) whose concurrent measurements were processed through the WAsP/windPRO wind flow modelling chain to produce cross-predictions: the wind climate observed at mast A is transferred to the location of mast B using the linearised flow model, and the predicted wind speed at B is compared to the actual measurement at B. Each physical pair yields two directional pairs (A predicting B, and B predicting A), producing approximately 41 usable directional pairs after data quality filtering. Each directional pair is further decomposed into 12 wind direction sectors (30-degree compass bins), resulting in approximately 492 sector-level observations that form the primary modelling dataset.

The iterative development proceeded through seven phases, which can be grouped into three conceptual stages:

1. **Phases 1--3 (Mean regression and the pivot to sigma):** Initial attempts to predict the expected direction and magnitude of wind speed deviation, culminating in the critical insight that predicting the spread (uncertainty) of deviations is both more tractable and more useful than predicting their direction.

2. **Phases 4--5 (Target and resolution refinement):** Transition from energy yield to wind speed as the prediction target, and from pair-level to sector-level observations, dramatically increasing statistical power.

3. **Phases 6--7 (Final model architecture):** Adoption of signed errors with a Student-t likelihood, energy-weighted inference, feature ablation, and deployment constraint enforcement, producing the recommended six-feature model.

This iterative structure is not merely a narrative convenience; it reflects the genuine research trajectory and provides essential context for understanding why particular modelling choices were made. Many of the final model's most important design decisions -- HalfNormal priors on sigma drivers, zero-mean likelihood, physical-pair grouping in cross-validation -- arose directly from failed or partially successful earlier attempts.

The data pipeline that feeds the model comprises four sequential stages: timeseries analysis, directional analysis, feature engineering, and execution orchestration. These are described in Section 2.2. The seven modelling phases are then presented in Sections 2.3 through 2.7, followed by the complete final model specification (Section 2.8), validation strategy (Section 2.9), deployment architecture (Section 2.10), and software details (Section 2.11).

*Figure 2.1 provides a schematic overview of the complete data pipeline, from raw measurement files through feature engineering to model training and deployment.*


---

## 2.2 Data Pipeline: From Raw Measurements to Model Inputs

The data pipeline transforms raw wind measurement timeseries into a structured feature matrix suitable for Bayesian modelling. It consists of four stages, each implemented as a standalone Python script with well-defined inputs and outputs. This modular design ensures that any stage can be re-run independently when upstream data changes or when new measurement pairs are added to the dataset.


### 2.2.1 Stage 1: Timeseries Analysis

**Input data.** The primary input to the pipeline consists of raw 10-minute average wind speed and wind direction records exported from windPRO as tab-delimited text files. For each measurement location, three distinct file types are produced:

- **"True" file:** The actual measured wind speed and direction timeseries at the site, as recorded by the physical instrument (cup anemometer or LiDAR). This represents ground truth.
- **"Self" file:** The windPRO self-prediction timeseries, in which the observed wind climate at the site is run through the WAsP flow model and predicted back to the same location. The self-prediction captures any systematic bias introduced by the flow model's terrain representation at that specific site.
- **"Cross" file:** The windPRO cross-prediction timeseries, in which the observed wind climate at a different measurement site (the "predictor" mast) is transferred through the flow model to predict the wind climate at the current site. This is the quantity whose uncertainty the model ultimately aims to characterise.

Each file contains timestamped records at 10-minute resolution, with columns for mean wind speed (m/s) and mean wind direction (degrees).

**Concurrent period identification.** A critical preprocessing step is the identification of concurrent measurement periods between the predictor site and the target site. Only timestamps for which both sites have valid measurements can be used for cross-prediction comparison, because the WAsP transfer assumes concurrent atmospheric conditions. The pipeline performs an inner join on timestamps between the true file at the target site and the cross-prediction file generated from the predictor site. This inner join naturally excludes periods where either mast experienced downtime, instrument failure, or data quality flags.

**Monthly data availability filtering.** After the concurrent period is established, the pipeline applies a monthly completeness criterion: any calendar month with less than 80% data availability (fewer than approximately 3,500 valid 10-minute records out of a theoretical maximum of approximately 4,380) is excluded entirely. This threshold follows standard industry practice for wind resource assessment campaigns [4, 8] and ensures that monthly statistics are not distorted by periods of sparse data that may not be meteorologically representative. The 80% threshold strikes a balance between data retention and statistical reliability: a stricter threshold (e.g., 95%) would discard too many months in campaigns with occasional instrument issues, while a laxer threshold (e.g., 60%) would permit months dominated by a single weather pattern.

**Pair-level statistics.** For each measurement pair (predictor site A, target site B), the following statistics are computed over the filtered concurrent period:

- **Pearson correlation coefficient (r):** Measures the linear association between predicted and measured 10-minute wind speeds. High correlation (r > 0.90) indicates that the flow model captures the temporal structure of the wind climate well; lower correlation suggests that terrain effects, wake effects, or mesoscale phenomena are disrupting the transfer.

- **Mean Absolute Error (MAE):** The average of the absolute differences between predicted and measured wind speeds, expressed in m/s. MAE provides a scale-dependent measure of prediction accuracy.

- **Bias (mean error):** The average signed difference between predicted and measured wind speeds. Positive bias indicates systematic over-prediction; negative bias indicates under-prediction.

- **Standard deviation of errors:** The standard deviation of the signed differences (predicted minus measured), capturing the spread of prediction errors around the mean error.

- **Normalised bias:** The bias divided by the mean measured wind speed, expressing the systematic error as a fraction of the mean wind climate. This enables comparison across sites with different mean wind speeds.

- **Normalised MAE:** The MAE divided by the mean measured wind speed, similarly enabling cross-site comparison of prediction accuracy.

**Energy yield deviation.** The energy yield (EY) deviation is the primary pair-level metric of flow model performance, computed as:

    EY_deviation = (sum(predicted_energy) - sum(measured_energy)) / sum(measured_energy)

where the energy at each 10-minute timestep is computed by applying a representative power curve to the wind speed record. The energy yield deviation captures the financially relevant error: it reflects not just wind speed errors but their amplification through the nonlinear power curve, where errors at high wind speeds (near rated power) have disproportionate impact on energy production.

**Wind speed uncertainty.** The wind speed uncertainty metric is computed as:

    WS_uncertainty = std(predicted - measured) / mean(measured)

This normalised standard deviation of prediction errors provides a scale-independent measure of prediction spread and serves as the precursor to the sigma quantity that the final model predicts.

**Output.** Stage 1 produces a "Collected output" Excel workbook containing one row per measurement pair and columns for all pair-level statistics described above. This workbook serves as the master record of pair-level flow model performance and is consumed by subsequent pipeline stages.


### 2.2.2 Stage 2: Directional Analysis

**Sector decomposition.** The directional analysis stage decomposes each measurement pair's timeseries into 12 wind direction sectors of 30 degrees each, aligned with the compass rose: N (345--015 degrees), NNE (015--045 degrees), ENE (045--075 degrees), E (075--105 degrees), ESE (105--135 degrees), SSE (135--165 degrees), S (165--195 degrees), SSW (195--225 degrees), WSW (225--255 degrees), W (255--285 degrees), WNW (285--315 degrees), and NNW (315--345 degrees). Each 10-minute observation is assigned to a sector based on the measured wind direction at the predictor mast (the measurement mast, or MM).

This 12-sector decomposition matches the standard sectoral resolution used in WAsP and windPRO for wind resource assessment [1, 4] and provides sufficient angular resolution to capture the directional dependence of terrain effects while maintaining adequate sample sizes per sector. A finer resolution (e.g., 36 sectors of 10 degrees) would reduce sample counts to levels where sector-level statistics become unreliable; a coarser resolution (e.g., 4 quadrants) would mask important directional features such as channelling effects, escarpment speedups, or roughness transitions.

**Per-sector computations.** For each of the 12 sectors within each measurement pair, the following quantities are computed:

- **Sample count:** The number of valid 10-minute observations assigned to the sector. Sectors with very low sample counts (fewer than 50 observations) are flagged but retained, as the hierarchical model can partially pool information across sectors.

- **Mean wind speeds:** The arithmetic mean of the measured wind speed (from the true file) and the predicted wind speed (from the cross-prediction file) within the sector.

- **Sector energy yield:** The energy produced within the sector, computed as the sum of power (from the power curve) multiplied by the time fraction represented by each observation. The energy weight of each sector is the sector energy divided by the total energy across all 12 sectors.

- **Sector-level EY deviation:** The relative deviation of predicted energy from self-predicted energy within the sector:

      Sector_EY_deviation = (EY_pred - EY_self) / EY_self * 100%

  Note that the denominator uses the self-prediction energy (EY_self) rather than the true measured energy. This is because the self-prediction represents the flow model's best estimate of the "true" wind climate at the target site, given the terrain model and the WAsP assumptions. Using EY_self as the reference removes any systematic bias that the flow model introduces at the target site itself, isolating the additional error introduced by the horizontal transfer from the predictor site.

- **Sector-level wind speed deviation:** The relative deviation of predicted wind speed from actual measured wind speed:

      WS_deviation = (WS_pred - WS_actual) / WS_actual

  This is the quantity that the final model's sigma parameter describes.

**Deflection metrics.** WAsP computes a wind direction deflection (turning angle) at each site for each sector, representing the angular difference between the geostrophic wind direction and the surface-level wind direction caused by terrain steering. The deflection metrics capture the sensitivity of sector assignment to this turning:

- **Edge fraction:** The fraction of 10-minute observations within a sector whose wind direction falls within the WAsP-computed turning angle of a sector boundary. These "edge" observations are most sensitive to deflection: a small change in the turning angle could reassign them to an adjacent sector. High edge fractions indicate sectors where the energy distribution is sensitive to the flow model's directional deflection estimate.

- **Flip fraction:** The fraction of observations that actually change sector assignment when the WAsP deflection is applied. If the predictor mast observes a wind direction of 14 degrees (assigned to sector N, 345--015 degrees) and the WAsP turning angle at the target site is 5 degrees, the deflected direction would be 19 degrees, falling in sector NNE. This observation has "flipped" from N to NNE. High flip fractions indicate sectors where the flow model's directional treatment actively redistributes energy between sectors.

- **Power-weighted versions:** Both edge fraction and flip fraction are also computed with power weighting, where each observation's contribution is weighted by its power output from the power curve. This ensures that high-wind, high-energy observations receive proportionally more influence in the metric.

- **Differential turning:** The difference in WAsP-computed turning angle between the target site (WTG) and the predictor site (MM) for each sector:

      d_turning_deg = turning_WTG - turning_MM

  Large differential turning indicates that the flow model predicts substantially different directional deflection at the two sites, which may introduce systematic sector-assignment errors.

**Output.** Stage 2 produces a per-pair Excel file containing 12 rows (one per sector) with all sector-level statistics and deflection metrics described above. These files are consumed by the feature engineering stage.


### 2.2.3 Stage 3: Feature Engineering (Compendium of Features)

The feature engineering stage is implemented in a central script (the "Compendium of Features") that merges data from multiple sources into a unified modelling matrix. The input sources include:

- **Device data:** Physical characteristics of each measurement pair, including geographic coordinates, inter-site distance, elevation difference (dz), and terrain slope along the connecting line.
- **Speed-up factors:** WAsP-computed speed-up factors at both the predictor site and the target site for each of the 12 sectors. The speed-up factor quantifies how the local terrain accelerates or decelerates the wind relative to the undisturbed flow.
- **Turbulence Intensity (TI):** The turbulence intensity at the predictor mast for each sector, computed as the standard deviation of 10-minute wind speed divided by the mean wind speed. TI captures the level of atmospheric turbulence, which affects the reliability of the linearised flow model assumptions.
- **Weibull parameters:** The Weibull scale (A) and shape (k) parameters fitted to the wind speed distribution at the predictor mast for each sector. These parameters characterise the statistical distribution of wind speeds.
- **RIX values:** The Ruggedness Index (RIX) at both sites for each sector, computed as the fraction of the terrain surface within a specified radius that exceeds a critical slope threshold (typically 0.3 or 17 degrees) [3, 10]. RIX quantifies terrain complexity from the perspective of the linearised flow model, which assumes gentle slopes.
- **Directional analysis outputs:** All sector-level statistics from Stage 2.

The features are organised into two categories based on their spatial resolution:

**Pair-level features** are constant across all 12 sectors for a given measurement pair. They describe the geometric and statistical relationship between the two sites:

- **distance:** The horizontal distance between the predictor site and the target site, in metres. Greater distances increase the likelihood that the wind flow model must traverse complex terrain features, atmospheric transitions, or mesoscale boundaries.

- **dz:** The elevation difference between the target site and the predictor site, in metres. Height differences introduce systematic wind speed scaling effects (wind shear, acceleration over ridges) that the flow model must capture.

- **slope:** The average terrain slope along the line connecting the two sites. Steep slopes challenge the linearised flow model's fundamental assumption of small perturbations to the mean flow.

- **sample_shortfall:** A measure of how much shorter the measurement campaign is compared to the longest campaign in the dataset, computed on a logarithmic scale:

      sample_shortfall = log(max_pair_samples + 1) - log(this_pair_samples + 1)

  The logarithmic transformation reflects the diminishing returns of additional measurement data: the improvement in statistical certainty from extending a campaign from 3 months to 6 months is much greater than the improvement from 18 months to 21 months. A sample_shortfall of zero indicates the longest campaign in the dataset; positive values indicate shorter campaigns with correspondingly greater statistical uncertainty.

- **concentration_ratio:** A measure of how concentrated the site's energy production is across the wind rose, computed using an inverse Herfindahl index:

      concentration_ratio = (12 - 1 / sum(w_i^2)) / 11

  where w_i is the energy weight of sector i (i.e., the fraction of total energy produced in that sector). This metric ranges from 0 (all energy concentrated in a single sector) to 1 (energy uniformly distributed across all 12 sectors). Low concentration ratios indicate sites where errors in one or two dominant sectors have an outsized impact on the total energy yield uncertainty, because there is limited diversification across the wind rose.

- **speedup_diff_std:** The standard deviation of the difference in speed-up factors between the predictor site and the target site across all 12 sectors:

      speedup_diff_std = std(speedup_MM_i - speedup_WTG_i), for i = 1..12

  A high value indicates that the relationship between the two sites' speed-up factors varies substantially with wind direction, suggesting that the terrain interaction between the sites is directionally complex. If the speed-up difference were constant across all sectors, the flow model could capture it with a simple scaling factor; variable speed-up differences demand that the model correctly capture directionally varying terrain effects.

**Sector-level features** vary across the 12 wind direction sectors within each pair:

- **abs_turning:** The absolute value of the WAsP-computed turning angle at the target site for the given sector. Large turning angles indicate strong directional deflection by the terrain.

- **abs_log_speedup:** The absolute value of the natural logarithm of the speed-up factor at the predictor site. The logarithmic transformation makes the feature symmetric around a speed-up factor of 1.0 (no speed-up): a speed-up of 1.5 and a slow-down of 0.67 both produce |log| values of approximately 0.4.

- **RIX_avg:** The average of the RIX values at the predictor site and the target site for the given sector:

      RIX_avg = (RIX_MM + RIX_WTG) / 2

  This represents the mean terrain ruggedness experienced by the wind flow as it passes between the two sites in that direction.

- **abs_dRIX:** The absolute difference in RIX between the predictor site and the target site:

      abs_dRIX = |RIX_MM - RIX_WTG|

  Large dRIX values indicate a mismatch in terrain complexity between the two sites: one site may be on smooth terrain while the other is in complex terrain. The linearised flow model may perform adequately at both sites individually but struggle to accurately transfer the wind climate between such dissimilar environments.

- **severity_fraction:** The fraction of the terrain surface between the two sites (in the given sector's direction) that exceeds the critical slope threshold. This extends the point-based RIX metric to a path-based measure of terrain difficulty.

- **flip_frac:** The flip fraction computed in Stage 2, representing the fraction of observations that change sector assignment when WAsP deflection is applied.

- **turning_gradient:** The maximum absolute difference in turning angle between the current sector and its two neighbouring sectors:

      turning_gradient = max(|turning_current - turning_left_neighbour|, |turning_current - turning_right_neighbour|)

  Circular wrapping is applied at the 0/360-degree boundary to ensure correct computation for the N and NNW sectors. A high turning gradient indicates a sector where the directional deflection changes rapidly with wind direction, suggesting a terrain feature (such as a valley exit or escarpment edge) that creates a sharp transition in flow behaviour. Such transitions are challenging for the WAsP model because small errors in the incident wind direction can produce large errors in the predicted deflection.

- **speedup_gradient:** Analogous to turning_gradient but for speed-up factors:

      speedup_gradient = max(|speedup_current - speedup_left_neighbour|, |speedup_current - speedup_right_neighbour|)

  A high speedup gradient indicates a sector where the terrain acceleration effect changes rapidly with wind direction.

- **k_MM_deviation:** The deviation of the Weibull shape parameter at the predictor mast from the typical value of 2.0:

      k_MM_deviation = |k_MM - 2.0|

  A Weibull shape parameter of 2.0 corresponds to a Rayleigh distribution, which is the most common wind speed distribution shape observed at most sites [8]. Deviations from k = 2.0 indicate unusual wind speed distribution shapes -- perhaps bimodal (very low k) or very narrow (high k) -- that may interact with the power curve nonlinearity to amplify or attenuate energy yield errors. Importantly, this feature uses only the predictor-mast Weibull parameter, which is available during deployment (the target site has no measurements).

- **roughness_mismatch:** A measure of the difference in surface roughness between the predictor site and the target site for the given sector. Surface roughness affects the vertical wind speed profile and the boundary layer structure; large mismatches between sites may cause the flow model to incorrectly scale wind speeds.

- **TI_MM:** The turbulence intensity at the predictor mast for the given sector. Higher turbulence intensity indicates more gusty, variable wind conditions that may challenge the steady-state assumptions of the linearised flow model.

- **log_energy_weight:** The natural logarithm of the energy weight of the sector:

      log_energy_weight = log(energy_weight_i)

  The logarithmic transformation compresses the range of energy weights, which can span two orders of magnitude between dominant and minor sectors. This feature allows the model to learn whether high-energy or low-energy sectors tend to have different uncertainty characteristics.

- **dist_norm:** A normalised distance metric that accounts for terrain complexity:

      dist_norm = log(distance_m / distance_A)

  where distance_A is the terrain-dependent lower acceptable distance defined in the FGW Technical Guideline TR6 [4]. The TR6 guideline specifies minimum acceptable distances between measurement and turbine locations as a function of terrain complexity: flat terrain permits shorter transfer distances, while complex terrain requires the measurement to be closer to the target. By dividing the actual distance by this terrain-adjusted reference distance, the metric captures how "stretched" the flow model transfer is relative to accepted practice. A dist_norm value of zero indicates a distance exactly at the guideline threshold; positive values indicate distances exceeding the guideline recommendation. The logarithmic transformation reflects the physical expectation that uncertainty grows sub-linearly with distance due to the spatial autocorrelation of terrain effects.

**Output.** Stage 3 produces the file Focused_modelling_inputs.xlsx, containing 12 rows per directional pair (one per sector) and all engineered features as columns. This file is the direct input to the Bayesian model.

*Figure 2.2 illustrates the feature engineering pipeline, showing the flow of data from raw sources through derived features to the modelling matrix. Table 2.1 provides a complete summary of all features, their levels (pair or sector), formulas, and physical interpretations.*


### 2.2.4 Stage 4: Execution Pipeline

The complete four-stage data pipeline is orchestrated by a single execution script (Execute_Order_66.py) that runs all stages in sequence. This script ensures full reproducibility: given the same raw input data, it produces identical feature matrices and, consequently, identical model training data. The execution pipeline handles file path resolution, error logging, and intermediate output validation. It checks that each stage's output conforms to expected dimensions and data types before proceeding to the next stage.

The orchestration script also manages the interface between the data pipeline and the modelling code: after the feature matrix is produced, it can optionally trigger model training, cross-validation, or deployment prediction, depending on command-line arguments. This end-to-end automation was essential during the iterative development process, as each modelling phase required multiple cycles of feature modification, model retraining, and result inspection.


---

## 2.3 Model Evolution: Phase 1 -- Kitchen-Sink Mean Regression

### Starting Point and Motivation

The initial modelling objective was straightforward and, in retrospect, overly ambitious: predict the energy yield (EY) deviation of the wind flow model as a function of measurable terrain and campaign features. The premise was that if the terrain between two measurement sites was complex, the distance was large, or the speed-up factors were extreme, the flow model should systematically over- or under-predict the energy yield by an amount that could be estimated from these features. In other words, the first phase attempted to predict E[y] -- the expected value of the deviation -- as a deterministic function of site characteristics, with residual noise captured by a random error term.

### Regression A1: The Kitchen-Sink Model

The first regression model (designated A1) employed a classical hierarchical linear structure:

    EY_deviation_i = alpha + beta_1 * dRIX_0.3_sector_i + beta_2 * d_overall_speedup_factor_i + beta_3 * TRIX_sector_0.3_i + beta_4 * Mean_windspeed_self_i + pair_effect_p[i] + sector_effect_s[i] + epsilon_i

where:
- alpha is the global intercept
- beta_1 through beta_4 are fixed-effect regression coefficients for the terrain features
- pair_effect_p is a random intercept for each measurement pair, capturing systematic pair-specific deviations not explained by the features
- sector_effect_s is a random intercept for each sector, capturing systematic directional biases not explained by the features
- epsilon_i is the observation-level residual error, assumed Normal(0, sigma)

The random effects were implemented using non-centered parameterisation [5, 6], a standard technique in Bayesian hierarchical modelling that reparameterises the random effects to improve MCMC sampling efficiency. Instead of sampling pair_effect_p directly from Normal(0, sigma_pair), the model samples a standardised variable eta_p from Normal(0, 1) and computes pair_effect_p = sigma_pair * eta_p. This avoids the problematic "funnel geometry" in the posterior -- a pathology where the random effects and their variance become strongly correlated, causing the MCMC sampler to mix poorly.

The model was fitted using a Normal likelihood with sample weighting: the variance of each observation was scaled inversely with the square root of the measurement sample count, so that pairs with longer concurrent measurement periods (and therefore more statistically reliable deviation estimates) received more influence in the likelihood.

### The Discovery of Pair Symmetry

The most consequential finding from Phase 1 was not about the features at all, but about the structure of the data. When predicting the wind climate at site B using observations from site A (the A-to-B direction), the WAsP flow model produces a certain deviation. When the direction is reversed -- predicting at A using observations from B (the B-to-A direction) -- the model produces a different deviation using the same terrain model and the same atmospheric conditions.

A dedicated analysis script (pair_symmetry_check.py) examined the relationship between these two directional deviations by correlating y_total(A-to-B) with -y_total(B-to-A), where y_total is the total deviation including the pair random effect. The hypothesis was that if site A is harder to predict from B (positive deviation), then site B should be correspondingly easier to predict from A (negative deviation), because the terrain transfer errors should reverse direction. This hypothesis was confirmed: the pair-level offsets showed a strong antisymmetric relationship.

This discovery had profound implications for the modelling framework. It meant that the pair random effect for the A-to-B direction should be the negative of the pair random effect for the B-to-A direction. Mathematically, if alpha_p represents the pair-specific offset for the physical pair {A, B}, then:

    pair_effect(A-to-B) = +alpha_p
    pair_effect(B-to-A) = -alpha_p

This antisymmetric structure was implemented using two index variables: pairset_idx (identifying the physical pair, shared by both directions) and dir_sign (+1 for one direction, -1 for the other). The pair effect for observation i then becomes:

    pair_effect_i = dir_sign_i * alpha_pairset[pairset_idx_i]

This structural insight effectively doubled the usable information: instead of treating the 22 physical pairs as 22 independent observations, the model could use the 41 directional pairs as structurally related observations linked by the antisymmetric pair effect. This was a critical data efficiency gain for a dataset of this size.

### Transition to A2: Student-t Likelihood and Heteroscedastic Sigma

Building on the pair symmetry discovery, the model was extended in several directions:

**Student-t likelihood.** The Normal likelihood was replaced with a Student-t likelihood to provide robustness against outliers [5]. Wind flow model deviations occasionally exhibit extreme values -- particularly in complex terrain sectors with low sample counts -- and the heavier tails of the Student-t distribution accommodate such observations without allowing them to unduly influence the parameter estimates. The degrees of freedom parameter (nu) was treated as a free parameter to be learned from the data, with a weakly informative Gamma prior.

**Heteroscedastic sigma.** Instead of a single global sigma, the model introduced a "switchboard" mechanism for testing which features drive the observation-level variance. The log-sigma for each observation was parameterised as a linear combination of candidate features:

    log(sigma_i) = log(sigma_0) + sum(gamma_k * X_ki * switch_k)

where switch_k is a binary indicator (0 or 1) that controls whether feature k enters the sigma model. This switchboard allowed systematic exploration of which features affect the spread of deviations, independent of their effect on the mean.

### A2.1: Joint Likelihood and Sector Filtering

The A2.1 variant introduced two important refinements:

**Joint pair-level and sector-level likelihood.** The same set of beta coefficients was used to predict both the pair-level aggregated deviation and the sector-level deviations within each pair. This joint likelihood encouraged the model to find feature effects that were consistent across both levels of aggregation, reducing the risk of overfitting to sector-level noise.

**Sector sample count filtering.** Sectors with fewer than approximately 1,000 valid 10-minute observations (roughly equivalent to one week of continuous data) were excluded from the modelling dataset. This threshold was chosen empirically: below 1,000 observations, the sector-level deviation estimates became highly volatile, dominated by a few weather events rather than reflecting the long-term wind climate statistics. This filtering improved data retention from approximately 59% to 78% of all sector observations, because it replaced a stricter but less targeted pair-level filter with a more granular sector-level criterion.

**Asinh transform.** An inverse hyperbolic sine (asinh) transform was applied to extreme sector-level deviations. The asinh function behaves linearly near zero and logarithmically for large absolute values, effectively compressing extreme outliers without introducing the singularity at zero that a pure logarithmic transform would create. This stabilised the likelihood contribution of extreme sector observations.

### A2.1: Dz-Gated Self-Prediction

A further variant tested whether the self-prediction deviation (how well the flow model reproduces the wind climate at a site when predicting from the same site) could serve as a predictor of the cross-prediction error. The hypothesis was that a site where the flow model struggles with self-prediction would also be a site where cross-prediction is unreliable. However, this relationship was conditioned on the height difference (|dz|) between the two masts: self-prediction deviation was only included as a predictor for pairs where |dz| <= 20 metres, to avoid conflating terrain effects with simple wind shear scaling errors.

The results were inconclusive: the self-prediction feature showed some correlation with cross-prediction error, but it was heavily confounded with other terrain features. Sites with poor self-prediction tended to be sites with high RIX, high terrain complexity, and large speed-up factors -- all of which were already represented in the feature set. The self-prediction feature was therefore not retained in subsequent model versions.

### The Fundamental Lesson

The overarching lesson from Phase 1 was sobering: **predicting the mean deviation proved extremely difficult.** The expected deviation was overwhelmingly dominated by the height difference (dz) between the two masts. After controlling for dz, the remaining terrain features explained very little of the directional signal. The R-squared values for the mean model (excluding random effects) were disappointingly low, and the random effects absorbed most of the explainable variance -- suggesting that pair-specific idiosyncrasies, not measurable terrain features, determined the direction of the deviation.

This finding was not a failure of the modelling approach but rather an important empirical insight: the direction of flow model error (whether it over- or under-predicts) is largely unpredictable from standard terrain features. The linearised flow model's errors are not systematic in a direction that terrain features can capture; they are, in a practical sense, stochastic. This insight directly motivated the pivot to sigma modelling in Phase 2.


---

## 2.4 Model Evolution: Phase 2 -- The Pivot to Sigma

### The Conceptual Breakthrough

The transition from Phase 1 to Phase 2 represents the most important intellectual turning point in the research. After extensive experimentation with mean regression models, the following insight was recorded in the project notes:

> "Describing the uncertainty is what I need to do. Not describe the direction of uncertainty."

This seemingly simple statement encapsulates a fundamental reframing of the modelling objective. The wind energy industry does not primarily need to know whether a flow model will over-predict or under-predict at a new site (though that would be useful). What it needs is a reliable estimate of **how large** the prediction error might be -- the sigma, or standard deviation, of the deviation distribution. This sigma directly feeds into the P-value calculations that determine project financing:

    P75 = best_estimate * (1 - 0.674 * sigma)
    P90 = best_estimate * (1 - 1.282 * sigma)

A P90 energy yield estimate means the yield that will be exceeded with 90% probability, accounting for all sources of uncertainty. The flow model uncertainty (sigma) is one of the largest contributors to the total uncertainty budget, and improving its estimation directly affects the financial viability assessment of wind energy projects.

### The New Architecture: Sparse Mean, Rich Sigma

Phase 2 introduced a fundamentally different model architecture:

- **Sparse mean model:** The mean of the deviation distribution was modelled with only one feature -- dz (height difference) -- which was the only feature that showed a clear and physically interpretable effect on the expected direction of deviation in Phase 1. All other features were removed from the mean model, accepting that the directional component of the error is largely unpredictable.

- **Rich sigma model:** The standard deviation of the deviation distribution was modelled as a function of multiple terrain and campaign features using a log-linear link:

      log(sigma_i) = log(sigma_0) + gamma_dist * dist_i + gamma_dz * |dz_i| + gamma_dRIX * |dRIX_i| + gamma_speedup * |log_speedup_i|

  The log-link ensures that sigma is always positive (a mathematical necessity for a standard deviation) and implies that features affect sigma multiplicatively: each unit increase in a feature multiplies sigma by exp(gamma), rather than adding to it additively. This multiplicative structure is physically sensible -- a site that is both distant and in complex terrain should have uncertainty that compounds, not merely sums.

### The HalfNormal Prior Innovation

The most critical design decision in the sigma model was the choice of prior distribution for the gamma coefficients: all gamma parameters were assigned HalfNormal priors, constraining them to be non-negative. This encodes the physical principle of **monotonicity**: more distance, more terrain complexity, more speed-up mismatch, more deflection variability can only **increase** uncertainty, never decrease it.

Consider the alternative: if gamma_dist were allowed to be negative, the model could learn that greater distance between sites **reduces** uncertainty. While such a pattern might appear in a small training dataset due to sampling fluctuations (perhaps the most distant pairs in the dataset happen to be in simple terrain), it would be physically nonsensical and would not generalise to new sites. The HalfNormal prior prevents the model from learning such spurious relationships by encoding domain knowledge directly into the prior structure.

This is arguably the single most important modelling choice in the entire framework. It transforms the model from a pure data-driven regression (which would require far more data than 492 observations to reliably estimate 10+ parameters) into a physically informed model that uses domain knowledge to constrain the parameter space. The resulting model is more robust, more interpretable, and requires less training data than an unconstrained alternative.

### The dRIX Exploration: Direction vs. Magnitude

A revealing analysis during Phase 2 examined the dRIX feature (difference in Ruggedness Index between sites) from two perspectives:

- **As a predictor of deviation direction:** The correlation between dRIX and the signed deviation was r = -0.003, effectively zero. Terrain complexity mismatch provides no information about whether the flow model will over-predict or under-predict. This confirmed the Phase 1 finding that directional prediction is intractable.

- **As a predictor of deviation magnitude:** The correlation between |dRIX| and the absolute deviation |y| was positive and meaningful. Sites with greater terrain complexity mismatch exhibited larger absolute errors, regardless of sign.

This duality -- a feature can be useless for predicting direction but valuable for predicting spread -- became a guiding principle for subsequent feature engineering. Features were evaluated not by their ability to predict E[y] but by their ability to predict Var[y] or |y|.

### Phase A3 v2: Refined Sigma Drivers

Based on systematic feature exploration, the A3 v2 model refined the sigma model in several ways:

- **RIX_avg replaced |dRIX|:** The average RIX across both sites (RIX_avg_0.3) showed a stronger correlation with absolute deviation (r = 0.31) than the absolute difference in RIX (|dRIX|, r = 0.12). This makes physical sense: it is the overall terrain complexity encountered by the flow, not the mismatch between the two sites, that primarily determines uncertainty. Even two sites with identical RIX values can have high uncertainty if both are in complex terrain.

- **severity_fraction added:** This path-based terrain complexity measure captured additional information about the terrain between the two sites that neither site's individual RIX value fully reflected.

### Phase A3 v3: Interaction Terms

The A3 v3 model explored interaction effects, specifically the product of flip_fraction and speedup_gradient. The physical hypothesis was that sectors where many observations change sector assignment due to deflection (high flip_fraction) AND where the speed-up factor changes rapidly with direction (high speedup_gradient) should exhibit compounded uncertainty: the observations are being reassigned to sectors with very different speed-up characteristics, amplifying the error. This interaction term showed modest but positive predictive power for sigma.

### The Lesson of Phase 2

Phase 2 established the paradigm that persisted through all subsequent model versions: **the model predicts sigma (uncertainty), not mu (expected deviation).** The features that drive sigma are measures of difficulty, complexity, and mismatch -- they quantify how challenging a particular site configuration is for the flow model, without claiming to know the direction in which the model will err. This framing aligns naturally with the industry's need for uncertainty quantification and avoids the fundamental intractability of directional prediction demonstrated in Phase 1.


---

## 2.5 Model Evolution: Phase 3 -- Sigma Enrichment

### Progressive Feature Addition

With the sigma-modelling paradigm established, Phase 3 (the A4 model variants) focused on systematically enriching the sigma model with additional physically motivated features. Each variant added one or two features and evaluated whether the model's predictive performance improved on held-out data. The progression was deliberately incremental: each new feature was tested in isolation (controlling for the features already in the model) to ensure that its contribution was genuine rather than an artefact of collinearity with existing features.

**A4a: Sample shortfall.** The first addition was sample_shortfall, measuring how much shorter a measurement campaign is compared to the longest campaign in the dataset. The physical rationale is straightforward: shorter measurement campaigns capture fewer complete weather cycles, produce less stable wind statistics, and are therefore inherently more uncertain. A 6-month campaign may miss an entire season's wind pattern, while a 24-month campaign captures two full annual cycles and is far more representative.

The logarithmic scaling of sample_shortfall (computed as the difference in log-sample-counts) reflects diminishing returns: the statistical improvement from adding the 13th month of data is much smaller than the improvement from adding the 4th month. This logarithmic scaling is consistent with standard results from sampling theory, where the standard error of the mean decreases as 1/sqrt(n) -- effectively logarithmically when viewed on a proportional basis.

**A4b: Concentration ratio.** The energy concentration ratio measures how diversified a site's energy production is across the wind rose. Sites where a single sector dominates the energy production (low concentration ratio) are inherently more vulnerable to prediction errors in that sector: if the flow model makes a 10% error in a sector that contributes 40% of the energy, the pair-level error is heavily influenced. Conversely, sites with well-distributed energy production (high concentration ratio) benefit from error diversification -- positive errors in some sectors partially cancel negative errors in others.

The concentration ratio enters the sigma model because it affects how sector-level errors aggregate to pair-level uncertainty. Even if sector-level sigma is the same at two sites, the site with more concentrated energy production will exhibit higher pair-level uncertainty because there is less opportunity for error cancellation.

**A4c: Turning standard deviation (turning_std).** The standard deviation of the WAsP turning angle across all 12 sectors captures the overall variability of directional deflection at a site. High turning_std indicates that the flow model predicts substantially different directional effects across the wind rose, suggesting complex terrain with multiple competing terrain features (valleys, ridges, coastlines) that create sector-dependent flow patterns. Such sites are inherently more challenging for the linearised flow model.

**A4d: Speedup difference standard deviation (speedup_diff_std).** This pair-level feature, described in Section 2.2.3, captures the directional variability of the speed-up relationship between the two sites. A high speedup_diff_std means that the flow model must correctly capture a speed-up ratio that changes substantially with wind direction -- a much harder task than capturing a roughly constant speed-up ratio.

### Growth of the Sigma Model

Through the A4 variants, the sigma model grew from 4 features (distance, |dz|, RIX_avg, speedup) to 10 features. Each addition was accompanied by diagnostic checks: trace plots for convergence, posterior predictive checks for calibration, and comparison of the Widely Applicable Information Criterion (WAIC) or leave-one-out (LOO) cross-validation scores against the previous variant.

### Model Serialisation

Phase 3 also introduced systematic model serialisation. After each model was fitted, the posterior median parameters were saved to a JSON file and the complete InferenceData object (containing all MCMC samples, posterior predictive samples, and model diagnostics) was saved to a NetCDF file using ArviZ [6]. This practice served two purposes:

1. **Reproducibility:** Any model version could be reloaded and inspected without re-running the computationally expensive MCMC sampling (each model fit required approximately 5--10 minutes of computation time).

2. **Deployment:** The JSON file containing posterior medians and z-scoring parameters could be loaded by a lightweight deployment script that required only NumPy and Pandas -- not the full PyMC/ArviZ stack -- making the model accessible to industry practitioners without specialised probabilistic programming expertise.


---

## 2.6 Model Evolution: Phases 4--5 -- Wind Speed Target and Sector Resolution

### Phase 4: Transition to Wind Speed

Phases 1 through 3 used energy yield deviation as the prediction target. While energy yield is the financially relevant quantity, it introduces an additional layer of complexity: the power curve transformation. Energy yield deviation depends not only on wind speed prediction errors but also on where in the power curve those errors occur (errors near rated power have minimal energy impact; errors in the steep part of the curve have maximum impact) and on the specific turbine model's power curve shape.

Phase 4 (WS A1 and WS A2) adapted the entire modelling framework to predict wind speed deviation instead:

    e_i = (WS_predicted_i - WS_actual_i) / WS_actual_i

where WS_predicted is the flow model's cross-prediction of mean wind speed at the target site and WS_actual is the actual measured mean wind speed. This relative wind speed deviation is a dimensionless quantity that is directly comparable across sites with different mean wind speeds.

The transition to wind speed as the target variable offered several advantages:

1. **Directness:** Wind speed is the primary output of the WAsP flow model and the most direct measure of its performance. Energy yield involves an additional transformation (the power curve) that introduces its own uncertainty, which is conceptually separate from the flow model uncertainty.

2. **Universality:** Wind speed deviations are independent of the turbine model. A sigma estimate based on wind speed can be applied to any turbine by propagating through the turbine-specific power curve, whereas an energy-yield-based sigma would need to be re-estimated for each turbine model.

3. **Simplicity:** The relationship between wind speed deviation and terrain features is more direct than the relationship between energy yield deviation and terrain features, because the power curve nonlinearity is removed.

The WS A1 and WS A2 models used the same hierarchical structure as the energy yield models: pair random effects with antisymmetric direction signs, Student-t likelihood, and a log-linear sigma model with 10 features. The results confirmed that the sigma-modelling framework transferred successfully to the wind speed target, with comparable predictive performance.

### Phase 5: Sector-Level Resolution

The most impactful change in Phase 5 was the move from pair-level to sector-level observations. In all previous phases, the primary modelling unit was the measurement pair: each observation was a single deviation value (either EY or WS) computed over the entire concurrent measurement period for a given pair. With approximately 41 directional pairs, this produced roughly 41 observations -- a severely limited dataset for a hierarchical model with 10+ fixed-effect parameters and pair random effects.

Phase 5 decomposed each pair into its 12 constituent sectors, treating each sector as an individual observation. This yielded approximately 492 sector-level observations (after filtering for minimum sample counts), representing a 12-fold increase in dataset size. The benefits were substantial:

1. **Statistical power:** With 492 observations, the model had far more information for estimating the gamma parameters in the sigma model. The posterior distributions became tighter, and the model could reliably distinguish between features with genuine predictive power and those with spurious associations.

2. **Sector-level features:** The sector-level resolution enabled the inclusion of features that vary by direction -- speedup_gradient, turning_gradient, RIX_relative, log_energy_weight -- that could not be meaningfully used in a pair-level model.

3. **Reduced overfitting risk:** The ratio of observations to parameters improved dramatically. A pair-level model with 41 observations and 10 gamma parameters was at high risk of overfitting; a sector-level model with 492 observations and the same 10 parameters was on much firmer statistical ground.

The initial sector-level model (WS Sector v1) used an Exponential likelihood for the absolute errors |e|, with energy-weighted sigma:

    |e_i| ~ Exponential(rate = 1 / sigma_i)

where sigma_i was adjusted by the square root of the predicted energy weight: sigma_adj = sigma_i / sqrt(W_pred_i). This energy weighting downweighted the contribution of low-energy sectors (which tend to have noisy deviation estimates due to low sample counts) and upweighted high-energy sectors (which are more reliable and more financially relevant).

Subsequent variants (WS Sector v2 and v3) refined the likelihood specification and added sector-level features. A notable variant tested Normal priors (instead of HalfNormal) on the gamma parameters, allowing bidirectional effects -- the possibility that some features could decrease sigma. The results were instructive: the posterior distributions for all gammas remained predominantly positive even without the HalfNormal constraint, validating the physical monotonicity assumption. The HalfNormal constraint was therefore retained as a principled regularisation device rather than an arbitrary restriction.


---

## 2.7 Model Evolution: Phases 6--7 -- The Final Model

### Phase 6: Signed Errors with Student-t Distribution

The final model architecture crystallised in Phase 6, which introduced three interconnected changes:

**Signed errors.** Previous sector-level models used absolute errors |e| with an Exponential or half-Normal likelihood. Phase 6 reverted to signed errors e (which can be positive or negative), modelled with a Student-t distribution centred at zero:

    e_i ~ StudentT(nu, mu=0, sigma_i)

The use of signed errors was motivated by two considerations. First, signed errors preserve information about the direction of prediction failure (over- vs. under-prediction), which is relevant for diagnostic analysis even though the model does not attempt to predict the direction. Second, the symmetric Student-t distribution centred at zero naturally produces the plus/minus confidence intervals that the wind energy industry expects:

    CI = best_estimate +/- t_critical(nu, confidence_level) * sigma

This format is immediately interpretable by wind resource engineers and integrates directly into standard uncertainty budgets.

**Student-t distribution.** The Student-t distribution, with degrees of freedom nu learned from the data, provides heavier tails than the Normal distribution. This is physically appropriate because flow model errors occasionally exhibit extreme values -- particularly in sectors where the terrain violates the linearised model's assumptions, where sample counts are low, or where mesoscale effects (sea breezes, gap flows, foehn winds) introduce non-modelled phenomena.

The learned value of nu was approximately 14, indicating moderately heavy tails but close to Gaussian behaviour. For comparison, a Normal distribution corresponds to nu approaching infinity, and a Cauchy distribution has nu = 1. A value of nu = 14 means that extreme deviations (beyond 3 sigma) occur somewhat more frequently than a Normal distribution would predict, but the distribution still has finite variance and well-behaved moments. This is consistent with the empirical observation that most flow model errors are moderate (within 1--2 sigma) but occasional outliers occur.

**Energy-weighted likelihood.** The contribution of each sector observation to the likelihood was weighted by its normalised predicted energy weight:

    w_i = weight_energy_predicted_i / mean(weight_energy_predicted)

This weighting was implemented using a PyMC Potential term that added the weighted log-probability to the model's total log-probability:

    pm.Potential("energy_weight", sum(w_i * logp(StudentT(nu, 0, sigma_i), e_i)))

The energy weighting ensures that the model's parameters are influenced more strongly by sectors that matter more for the annual energy yield. A sector contributing 25% of the site's energy production carries correspondingly more weight in the likelihood than a sector contributing 2%. This is financially appropriate: the uncertainty in the total energy yield is dominated by the uncertainty in the high-energy sectors, and the model should invest its limited information primarily in characterising those sectors well.

**Turbulence Intensity features.** Phase 6 also introduced Turbulence Intensity (TI) features where available:

- **TI_MM:** The turbulence intensity at the predictor mast for each sector. Higher TI indicates more gusty conditions and more energetic turbulent eddies, which challenge the steady-state, time-averaged assumptions of the WAsP flow model.
- **dTI:** The difference in turbulence intensity between the target site and the predictor site, capturing TI mismatch.

The dTI feature was included provisionally, pending the deployment constraint analysis in Phase 7.

### Phase 7: Weibull Integration, Feature Ablation, and Deployment Constraints

Phase 7 represents the final refinement of the model, addressing three issues: additional feature integration, systematic feature selection, and deployment feasibility.

**Weibull shape parameter.** The Weibull shape parameter (k) at the predictor mast was integrated as the feature k_MM_deviation = |k_MM - 2.0|. The physical motivation is that sectors with unusual wind speed distributions (very peaked or very flat) may interact with the flow model's assumptions in ways that increase prediction uncertainty. The WAsP model estimates site-specific Weibull parameters, but the transformation from one site to another involves assumptions about how the distribution shape changes with terrain -- assumptions that may be less reliable when the original distribution is unusual.

**Feature ablation study.** A systematic feature ablation study compared six different feature configurations using physical-pair leave-one-out cross-validation (described in Section 2.9). For each configuration, the model was trained on all but one physical pair, predicted sigma for the held-out pair using only the fixed effects, and the predicted sigma was compared to the actual absolute error.

The six configurations ranged from a minimal model with only distance (1 feature) to the full model with all 21 candidate features. The results revealed a clear pattern of diminishing returns and eventual overfitting:

- The full model (21 features) achieved the highest Pearson correlation (r = 0.82) between predicted sigma and actual |error| but had the worst Spearman rank correlation (rho = 0.75) and the highest bias (+2.1 percentage points). The high Pearson r was driven by correct predictions for a few extreme cases, while the degraded Spearman rho indicated poor ranking performance across the full dataset.

- The recommended model (Top 5 + TI, 6 features) achieved a slightly lower Pearson r (0.79) but the best Spearman rho (0.85) and the lowest bias (+1.0 percentage points). This combination of strong ranking performance and low bias indicates a model that generalises well to new data: it correctly identifies which sites are more uncertain than others (high Spearman rho) without systematically over- or under-estimating the uncertainty level (low bias).

The conclusion was unambiguous: **15 of the 21 candidate features were "passengers" -- they captured noise in the training data that did not generalise to held-out data.** Including them degraded the model's ranking performance and increased its bias. The 6-feature model was selected as the recommended configuration.

**Deployment constraint enforcement.** A critical constraint for the practical usefulness of the model is that all features must be computable from data available at the time of deployment -- that is, before any measurements are taken at the target (WTG) site. At deployment, the user has:

- The location and elevation of the target site (from project planning)
- Measurements from the predictor mast (including wind speed distributions, TI, Weibull parameters)
- WAsP terrain analysis outputs for both sites (speed-up factors, turning angles, RIX)

What the user does **not** have is any observed wind data at the target site. This constraint led to the removal of the |dTI| feature, which required the turbulence intensity at the target site. The TI_MM feature (turbulence intensity at the predictor mast only) was retained, as it requires only predictor-site data.

This deployment constraint was encoded not just as a feature selection criterion but as a fundamental design principle: every feature in the recommended model must be a property of the predictor site, the terrain between the sites, or the geometric relationship between the sites -- never a property of the target site's observed wind climate.

**Data quality exclusions.** Measurement masts from projects with known data quality issues (instrument calibration problems, mast shadowing, or non-standard mounting configurations identified during the wind resource assessment process) were removed from the training dataset. These exclusions were based on project documentation and engineering judgement, not on the deviation values themselves, to avoid introducing selection bias.

*Figure 2.3 presents a timeline of the model evolution across all seven phases, showing the key decisions, feature additions, and performance metrics at each stage.*


---

## 2.8 Final Model Specification

This section provides the complete mathematical specification of the recommended model, including all distributional assumptions, prior choices, and their justifications.

### Target Variable

The model's target variable is the signed relative wind speed deviation at the sector level:

    e_i = (WS_predicted_i - WS_actual_i) / WS_actual_i

where:
- WS_predicted_i is the mean wind speed predicted by the WAsP flow model for sector i of the target site, using observations from the predictor mast
- WS_actual_i is the actual measured mean wind speed at the target site for sector i
- The index i runs over all sector-observations in the dataset (approximately 492 observations: ~41 directional pairs x 12 sectors, minus filtered sectors)

### Likelihood

The observations are modelled with a Student-t distribution:

    e_i ~ StudentT(nu, mu = 0, sigma_i)

where nu is the degrees of freedom (shared across all observations), mu = 0 (zero mean, discussed below), and sigma_i is the observation-specific scale parameter determined by the sigma model.

The likelihood is weighted by the normalised predicted energy weight of each sector:

    w_i = weight_energy_predicted_i / mean(weight_energy_predicted)

Implementation uses a PyMC Potential:

    pm.Potential("energy_weight", sum(w_i * logp(StudentT(nu, 0, sigma_i), e_i)))

### Sigma Model (Log-Linear)

The observation-specific sigma is modelled as a log-linear function of standardised features:

    log(sigma_i) = log(sigma_0) + gamma_1 * z_1i + gamma_2 * z_2i + ... + gamma_6 * z_6i + alpha_p[i]

where:
- sigma_0 is the baseline sigma (the uncertainty when all features are at their training-set mean values)
- z_ji = (X_ji - mean_j) / sd_j is the z-scored (standardised) value of feature j for observation i, using training-set means and standard deviations
- gamma_j is the coefficient for feature j, constrained to be non-negative by the HalfNormal prior
- alpha_p[i] is the pair random effect for the pair to which observation i belongs

Exponentiating both sides:

    sigma_i = sigma_0 * exp(gamma_1 * z_1i) * exp(gamma_2 * z_2i) * ... * exp(gamma_6 * z_6i) * exp(alpha_p[i])

This multiplicative structure means that each feature contributes a multiplier to the baseline sigma. A feature with gamma = 0.1 at its training-set mean (z = 0) contributes a multiplier of exp(0) = 1.0 (no effect); at one standard deviation above the mean (z = 1), it contributes exp(0.1) = 1.105, increasing sigma by 10.5%.

### Prior Specification

The following table summarises all prior distributions and their justifications:

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| nu (degrees of freedom) | Gamma(alpha=2, beta=0.1) | Weakly informative prior that places most mass on moderate values (nu = 5 to 30) while allowing both heavier tails (low nu) and near-Normal behaviour (high nu). The mode is at (alpha-1)/beta = 10, consistent with the expectation of moderately heavy tails. |
| log(sigma_0) (baseline log-sigma) | Normal(-3.9, 0.5) | The baseline sigma should reflect typical pair-level uncertainty when features are at average values. exp(-3.9) = 0.020, or 2.0%, which is appropriate for the pair-level target after cross-sector cancellation reduces the error magnitude relative to sector-level. The standard deviation of 0.5 allows the data to move the baseline between approximately 1.2% and 3.3%. |
| gamma_j (all 6 feature coefficients) | HalfNormal(sigma=0.3) | Constrains gamma >= 0 (physical monotonicity). The scale of 0.3 means that 95% of the prior mass is below 0.6, corresponding to a maximum multiplier of exp(0.6) = 1.82x at one standard deviation above the feature mean. This prevents implausibly large feature effects while allowing meaningful contributions. |
| sigma_pair (pair random effect scale) | HalfNormal(sigma=0.5) | The pair random effect captures systematic pair-specific uncertainty not explained by the features. The scale of 0.5 allows pair effects up to approximately 1.0, corresponding to sigma multipliers up to exp(1.0) = 2.7x. This is generous, reflecting genuine uncertainty about the importance of unmodelled pair-specific factors. |
| eta_p (non-centered pair effects) | Normal(0, 1) | Standard normal prior for the non-centered parameterisation. The actual pair effect is alpha_p = sigma_pair * eta_p, so the effective prior on alpha_p is Normal(0, sigma_pair). |

### The Six Recommended Features

The feature ablation study (Section 2.9) identified the following six features as the recommended configuration:

| # | Feature | Level | Formula | Posterior gamma (median +/- sd) | Sigma multiplier (at z=1) | Variance contribution |
|---|---------|-------|---------|---------------------------------|---------------------------|----------------------|
| 1 | dist_norm | Pair | log(distance_m / distance_A) | 0.365 +/- 0.070 | 1.44x | 47.1% |
| 2 | turning_gradient | Sector | max(\|turn_i - turn_{i-1}\|, \|turn_i - turn_{i+1}\|) | 0.095 +/- 0.061 | 1.10x | 3.2% |
| 3 | speedup_diff_std | Pair | std(speedup_MM - speedup_WTG) across sectors | 0.067 +/- 0.053 | 1.07x | 1.6% |
| 4 | k_MM_deviation | Sector | \|k_MM - 2.0\| | 0.066 +/- 0.038 | 1.07x | 1.5% |
| 5 | sample_shortfall | Pair | log(max_samples + 1) - log(pair_samples + 1) | 0.062 +/- 0.047 | 1.07x | 1.4% |
| 6 | TI_MM | Sector | Turbulence intensity at predictor mast | 0.062 +/- 0.041 | 1.06x | 1.4% |

The dominance of dist_norm (47.1% of variance) reflects the fundamental physical reality that horizontal transfer uncertainty grows with the distance between the measurement site and the prediction site. The remaining five features collectively contribute approximately 9% of the variance, with the pair random effect and baseline sigma accounting for the remainder. While the individual contributions of features 2--6 are modest, their collective inclusion improves the model's Spearman rank correlation by 0.25 (from 0.60 for distance-only to 0.85 for the 6-feature model), demonstrating that they provide meaningful discriminative power for ranking sites by uncertainty.

### Design Decision Explanations

**HalfNormal priors on gamma (physical monotonicity).** All six features in the recommended model are constructed so that larger values indicate more challenging prediction conditions: greater distance, sharper turning gradients, more variable speed-up relationships, more unusual Weibull shapes, shorter measurement campaigns, and higher turbulence. The HalfNormal prior on each gamma coefficient encodes the domain knowledge that these challenging conditions can only increase prediction uncertainty, never decrease it. This is the single most important modelling choice because it transforms the model from a purely data-driven regression into a physically constrained model.

Without this constraint, a model trained on 492 observations with 6 features could easily learn spurious negative associations -- for instance, finding that higher turbulence intensity at the predictor mast is associated with lower uncertainty, simply because the high-TI pairs in the training set happen to have other favourable characteristics. Such a finding would not generalise to new data and would produce physically nonsensical uncertainty estimates. The HalfNormal constraint eliminates this failure mode entirely, at the cost of a small amount of potential model flexibility that, based on the unconstrained-prior experiment in Phase 5, was not needed.

**Non-centered parameterisation.** The pair random effect is parameterised as alpha_p = sigma_pair * eta_p, where eta_p follows a standard Normal(0, 1) distribution and sigma_pair follows a HalfNormal(0.5) distribution. This non-centered parameterisation is a standard technique in Bayesian hierarchical modelling [5, 6] that addresses a well-known pathology of the centered parameterisation (alpha_p directly drawn from Normal(0, sigma_pair)).

In the centered parameterisation, when the data provide little information about sigma_pair (i.e., when the pair random effects are small relative to the observation noise), the posterior develops a "funnel" geometry: sigma_pair and the alpha_p values become strongly correlated, with alpha_p values compressed toward zero when sigma_pair is small. The NUTS sampler [6] struggles to navigate this funnel, producing poor mixing and unreliable posterior estimates.

The non-centered parameterisation breaks this correlation by separating the scale (sigma_pair) from the shape (eta_p) of the random effects. The sampler explores eta_p in a well-behaved standard Normal geometry, regardless of the current value of sigma_pair. The actual pair effects are reconstructed as a deterministic transformation, preserving the same statistical model while dramatically improving sampling efficiency. In practice, this change reduced the number of divergent transitions from several hundred (centered) to zero (non-centered) across all model fits.

**mu = 0 (zero mean).** The model assumes that the expected value of the wind speed deviation is zero for all observations: E[e_i] = 0. This is a strong assumption that deserves careful justification.

The zero-mean assumption reflects two empirical findings from Phases 1 and 2:

First, the direction of flow model error (over- vs. under-prediction) proved to be largely unpredictable from terrain features. After an extensive search involving dozens of candidate features and multiple model architectures, no feature set could reliably predict whether the flow model would over- or under-predict at a new site. The remaining directional signal was absorbed by pair random effects, which are not available for new sites at deployment time.

Second, any systematic directional bias (e.g., if the model consistently over-predicts by 3%) would indicate a deficiency in the WAsP flow model itself, not a property of the site. Such biases should be addressed by calibrating the flow model, not by the uncertainty model. The uncertainty model's role is to characterise the spread of errors around zero, not to correct for systematic bias.

The zero-mean assumption also simplifies the interpretation of sigma: the predicted sigma directly represents the expected magnitude of the error, without needing to disentangle a mean component. For industry users, this means that the uncertainty estimate is symmetric: the flow model is equally likely to over-predict by sigma as to under-predict by sigma, and the confidence interval is centred on the best estimate.

**Energy weighting.** The likelihood is weighted by the normalised predicted energy weight of each sector: w_i = weight_energy_predicted_i / mean(weight_energy_predicted). This weighting scheme ensures that the model's parameters are primarily informed by sectors that matter most for the annual energy yield.

The physical justification is financial: a wind energy project's revenue is proportional to its energy production, and the uncertainty in revenue is dominated by the uncertainty in the high-energy sectors. A sector with 25% of the energy and a 5% wind speed error contributes 1.25 percentage points to the total energy uncertainty, while a sector with 1% of the energy and a 50% wind speed error contributes only 0.5 percentage points. The energy weighting causes the model to prioritise accuracy in the high-energy sectors.

The normalisation by the mean weight ensures that the effective sample size is preserved: the sum of weights across all observations equals the number of observations, so the model does not artificially inflate or deflate the amount of information in the data. This is important for the Bayesian inference: unnormalised weights would distort the posterior by effectively claiming more or less data than actually exists.

**Deployment constraint.** Every feature in the recommended model is computable from data available at deployment time, when no measurements exist at the target (WTG) site. The available data consists of:

- Predictor mast measurements (wind speed distributions, TI, Weibull parameters for all 12 sectors)
- WAsP terrain analysis outputs for both sites (speed-up factors, turning angles, RIX values, which are computed from the digital terrain model and do not require wind measurements)
- The geometric relationship between the sites (distance, elevation difference)

This constraint led to the exclusion of the |dTI| feature (difference in turbulence intensity between sites), which showed modest predictive power but requires TI at the target site -- a quantity that is not available during deployment. The TI_MM feature (turbulence intensity at the predictor mast only) was retained because it requires only predictor-site data.

The deployment constraint is not merely a practical convenience; it is a fundamental requirement for the model's utility. An uncertainty model that requires observed data at the target site would be usable only for retrospective analysis (validating past predictions), not for prospective assessment (estimating uncertainty for a new project). The entire purpose of the model is to estimate uncertainty before the target-site wind climate is known.

### Student-t Expected Absolute Error Relationship

For a Student-t distribution with degrees of freedom nu, location mu = 0, and scale sigma, the expected absolute error has a closed-form expression:

    E[|e|] = sigma * sqrt(nu / pi) * Gamma((nu - 1) / 2) / Gamma(nu / 2)

For the learned value of nu = 14.00:

    E[|e|] = sigma * sqrt(14.00 / pi) * Gamma(6.500) / Gamma(7.000)
           = sigma * 0.844
           approximately 0.84 * sigma

This means that, for a well-calibrated model, the expected absolute wind speed error is structurally 16% below sigma. This is a mathematical property of the Student-t distribution, not a deficiency of the model or a miscalibration. When validating the model by comparing predicted sigma to observed |error|, a well-calibrated model will show:

    sigma approximately |error| / 0.84

This relationship is important for interpreting validation results and for communicating with industry practitioners. The sigma parameter is not the expected absolute error; it is the scale parameter of the Student-t distribution, which relates to the expected absolute error through the formula above.

In the wind energy industry, sigma feeds directly into P-value calculations:

    P75 = best_estimate * (1 - 0.674 * sigma)
    P90 = best_estimate * (1 - 1.282 * sigma)

These multipliers (0.674 for the 75th percentile, 1.282 for the 90th percentile) are derived from the quantile function of the assumed distribution and already account for the distribution shape. Sigma is therefore the correct deliverable to the industry -- not 0.84 * sigma, which would be the expected absolute error. Using 0.84 * sigma would systematically underestimate the P90 exceedance threshold, leading to over-optimistic energy yield estimates and under-estimated financial risk.

*Figure 2.4 presents the Directed Acyclic Graph (DAG) of the final model, showing the dependencies between all parameters, features, and observations. Figure 2.5 visualises the prior distributions for all key parameters. Tables 2.2 and 2.3 provide the complete prior specification and feature summary, respectively.*


---

## 2.9 Validation Strategy

Validation of a Bayesian hierarchical model with limited data requires particular care. Standard validation techniques (random train/test splits, k-fold cross-validation with random folds) can produce misleadingly optimistic performance estimates when the data contain structured dependencies -- as they do in this case, where observations within the same measurement pair share terrain features, meteorological conditions, and systematic flow model biases.

### Leave-One-Out Cross-Validation with Physical-Pair Grouping

The primary validation strategy is leave-one-out (LOO) cross-validation with physical-pair grouping. This approach addresses a subtle but critical source of data leakage that standard LOO would miss.

**The data leakage problem.** Consider a physical pair consisting of masts A and B. This physical pair generates two directional pairs: A predicting B (A-to-B) and B predicting A (B-to-A). These two directional pairs share:
- The same horizontal distance
- The same terrain between the sites
- The same concurrent measurement period
- The same speed-up factors (with only the direction reversed)
- The same RIX values
- The same WAsP terrain model

If standard LOO holds out the A-to-B direction while retaining B-to-A in the training set, the model has access to nearly all the information about this physical pair. The pair random effect learned from B-to-A provides direct information about A-to-B (since they are linked by the antisymmetric relationship alpha_A_to_B = -alpha_B_to_A), and the feature values are nearly identical. The resulting "prediction" for the held-out pair would be unrealistically accurate, inflating the validation performance.

**Physical-pair grouping.** To prevent this leakage, the LOO cross-validation groups directional pairs by their physical pair identity. The grouping key is constructed from sorted mast identifiers:

    pair_key = "__".join(sorted([mast_A, mast_B]))

This ensures that both A-to-B and B-to-A are assigned to the same group. When a physical pair is held out, ALL directional pairs involving those two masts are removed from the training set simultaneously.

**Cross-validation procedure.** For each of the approximately 22 physical pairs (one fold per physical pair):

1. **Data splitting:** All sector observations belonging to the held-out physical pair (approximately 24 observations: 2 directional pairs x 12 sectors) are removed from the training set. The remaining approximately 468 observations form the training set for this fold.

2. **Model retraining:** The full Bayesian model is retrained on the reduced training set using MCMC sampling with reduced computational budget: 2 chains (instead of 4), 1,000 tuning iterations (instead of 2,000), and 1,000 posterior draws (instead of 2,000). This reduced budget was chosen as a compromise between computational feasibility (22 folds x approximately 5 minutes each = approximately 2 hours total) and posterior quality. Diagnostic checks confirmed that R-hat values remained below 1.05 and effective sample sizes exceeded 200 with this reduced budget.

3. **Prediction for held-out pair:** The model predicts sigma for each sector of the held-out pair using only the fixed effects (the gamma coefficients and sigma_0 from the retrained model). The pair random effect is set to zero for the held-out pair, because no training data from that pair is available to estimate the random effect. This is the realistic deployment scenario: when predicting uncertainty for a new site, no site-specific random effect is available.

4. **Aggregation:** The sector-level predicted sigmas are aggregated to a pair-level predicted sigma using energy weights:

       sigma_pair = sqrt(sum(w_i * sigma_i^2))

   where w_i is the normalised energy weight of sector i within the pair. This energy-weighted root-mean-square aggregation reflects how sector-level uncertainties combine to produce the pair-level uncertainty.

5. **Comparison:** The pair-level predicted sigma is compared to the pair-level actual absolute error |e_pair|, where e_pair is the actual wind speed deviation for the held-out pair.

**Performance metrics.** After all 22 folds are completed, the following metrics are computed across all held-out pairs:

- **Pearson correlation (r):** Measures the linear association between predicted sigma and actual |error|. A high Pearson r indicates that the model correctly predicts the magnitude of uncertainty -- sites with high predicted sigma tend to have large actual errors, and vice versa.

- **Spearman rank correlation (rho):** Measures the monotonic (but not necessarily linear) association between predicted sigma and actual |error|. Spearman rho is more robust to outliers and non-linear relationships than Pearson r. A high Spearman rho indicates that the model correctly ranks sites by uncertainty, even if the absolute magnitude of the predictions is imperfect.

- **Bias:** The mean difference between predicted sigma and actual |error| across all held-out pairs, expressed in percentage points. Positive bias indicates systematic over-estimation of uncertainty (conservative); negative bias indicates under-estimation (optimistic).

*Figure 2.6 presents the LOO cross-validation results, showing the scatter plot of predicted sigma versus actual |error| for all held-out physical pairs, along with the Pearson r, Spearman rho, and bias.*

### Calibration Coverage Test

A complementary validation approach assesses the model's calibration -- whether the predicted confidence intervals cover the correct proportion of actual observations. For a well-calibrated model predicting a Student-t(nu, 0, sigma) distribution, X% of the actual deviations should fall within the predicted X% confidence interval.

The procedure computes confidence intervals at multiple confidence levels (50%, 68%, 80%, 90%, 95%) using the Student-t quantile function with the learned nu parameter. For each held-out observation (from the LOO CV), the model predicts sigma_i and checks whether the actual error e_i falls within the confidence interval [-t_crit * sigma_i, +t_crit * sigma_i], where t_crit is the Student-t critical value at the specified confidence level and degrees of freedom nu.

The results are:

| Nominal coverage | Observed coverage | Assessment |
|------------------|-------------------|------------|
| 50% | 29% | Under-coverage |
| 68% | 68% | Near-perfect |
| 80% | 88% | Slightly conservative |
| 90% | 100% | Conservative |
| 95% | 100% | Conservative |

The model is well-calibrated at the 68% level (the "one-sigma" interval) and increasingly conservative at higher confidence levels. The 90% and 95% intervals cover 100% of the held-out observations, meaning that the model never underestimates the uncertainty at these levels -- a desirable property for financial risk assessment, where under-estimating uncertainty is far more damaging than over-estimating it.

The under-coverage at the 50% level is a known issue that likely stems from the treatment of pair random effects in the LOO cross-validation. When a pair is held out, its random effect is set to zero (the prior mean), which may compress the predicted sigma distribution relative to the true posterior that would be obtained if some data from that pair were available. At the 50% confidence level, this compression is large enough relative to the narrow interval width to cause under-coverage; at higher confidence levels, the intervals are wide enough to absorb the compression. Addressing this issue would require a more sophisticated cross-validation procedure that marginalises over possible random effects for the held-out pair, which is computationally prohibitive for a study of this scope.

### Feature Ablation Comparison

The feature ablation study provides the evidence base for selecting the recommended 6-feature model over alternative feature configurations. Six configurations were systematically evaluated using the physical-pair LOO CV procedure described above:

| Feature Set | # Features | Pearson r | Spearman rho | Bias |
|-------------|-----------|-----------|--------------|------|
| Distance only | 1 | 0.65 | 0.60 | +2.8pp |
| Top 5 (no TI) | 5 | 0.77 | 0.84 | +1.0pp |
| Top 5 + TI | 6 | 0.79 | 0.85 | +1.0pp |
| Tier 2 no TI | 8 | 0.78 | 0.82 | +1.5pp |
| All Tier 2 | 9 | 0.80 | 0.83 | +1.3pp |
| Full model | 21 | 0.82 | 0.75 | +2.1pp |

Several patterns emerge from this comparison:

**Distance alone is informative but insufficient.** The single-feature model achieves a Spearman rho of 0.60, confirming that distance is the single strongest predictor of flow model uncertainty. However, a rho of 0.60 means that 40% of site pairs would be incorrectly ranked by a distance-only model. The additional features improve ranking performance by 0.25 (from 0.60 to 0.85), a substantial gain.

**Adding features improves performance up to a point.** Moving from 1 feature to 5 features improves Spearman rho from 0.60 to 0.84 -- the largest marginal gain in the table. Adding TI_MM as the 6th feature provides a further small improvement (0.84 to 0.85). Beyond 6 features, the improvements are minimal or negative.

**The full model overfits.** Despite achieving the highest Pearson r (0.82), the full model has the worst Spearman rho (0.75) and the highest bias (+2.1pp). The high Pearson r is driven by a few pairs where the full model's additional features happen to align with the actual error pattern in the training data; but for most pairs, the additional features add noise that degrades ranking performance. The decline in Spearman rho from 0.85 (6 features) to 0.75 (21 features) is a clear signal of overfitting.

**The 6-feature model is the recommended configuration.** It achieves the best Spearman rho (0.85), tied for the lowest bias (+1.0pp), and a respectable Pearson r (0.79). The small gap in Pearson r relative to the full model (0.79 vs. 0.82) is a worthwhile trade-off for the substantial improvement in ranking performance (0.85 vs. 0.75) and bias reduction (+1.0pp vs. +2.1pp).

The feature ablation results also provide insight into the "passenger features" phenomenon: many terrain features that show statistically significant associations with uncertainty in the full training dataset fail to improve prediction on held-out data. This is a consequence of the limited dataset size (approximately 492 sector observations, approximately 22 physical pairs) and the inherent collinearity among terrain features (complex terrain sites tend to have high RIX, large speed-up factors, large turning angles, and large dRIX simultaneously). In such settings, the model can fit the training data well with many different feature combinations, but only the most robust features generalise.

*Figure 2.7 presents the feature ablation results as a bar chart comparing Pearson r, Spearman rho, and bias across the six feature configurations.*


---

## 2.10 Deployment Architecture

The deployment architecture translates the trained Bayesian model into a practical tool that wind resource engineers can use to estimate flow model uncertainty for new wind energy projects. The architecture is designed for simplicity, transparency, and integration with existing industry workflows.

### Prediction Computation

For a new project with a predictor mast and a target turbine location, the uncertainty calculator takes two inputs:

1. **Trained model parameters:** The posterior median values of all model parameters (log_sigma_0, gamma_1 through gamma_6, and nu), along with the training-set means and standard deviations for each feature (used for z-scoring). These are stored in a JSON file produced during model training.

2. **New site features:** The six feature values for the new project, computed from the WAsP/windPRO terrain analysis and the predictor mast data.

The prediction proceeds in three steps:

**Step 1: Feature standardisation.** Each new-site feature value X_j is standardised using the training-set statistics:

    z_j = (X_j - mean_j_train) / sd_j_train

This ensures that the gamma coefficients, which were learned on standardised features, are applied to equivalently standardised input values. Using training-set statistics (not new-site statistics) is essential: the z-scores represent how the new site compares to the training population, not how the features compare to each other within the new site.

**Step 2: Sigma computation.** The predicted sigma for the new site is computed using the log-linear sigma model:

    sigma_new = exp(log_sigma_0 + gamma_1 * z_1 + gamma_2 * z_2 + ... + gamma_6 * z_6)

No pair random effect is included for new sites. The pair random effect captures systematic site-specific deviations that are not explained by the features, but estimating this effect requires observed error data from the site -- which, by definition, is not available during deployment. Setting the random effect to zero is equivalent to using the population-average sigma, which is the best available estimate for a new, unseen site.

This means that the deployment prediction represents the expected uncertainty for a site with the given feature values, averaged over the population of possible pair-specific effects. For some sites, the actual uncertainty will be higher than predicted (positive random effect); for others, it will be lower (negative random effect). The pair random effect scale (sigma_pair) provides a measure of this residual unpredictability.

**Step 3: Confidence intervals.** The predicted sigma is converted to confidence intervals using the Student-t quantile function:

    CI_lower = -t_critical(nu, confidence_level) * sigma_new
    CI_upper = +t_critical(nu, confidence_level) * sigma_new

Standard confidence levels reported in the industry include:

| Confidence level | t_critical (nu=14.00) | CI as fraction of sigma |
|-----------------|-------------------|------------------------|
| 50% | 0.692 | +/- 0.69 * sigma |
| 68% | 1.031 | +/- 1.03 * sigma |
| 80% | 1.345 | +/- 1.35 * sigma |
| 90% | 1.761 | +/- 1.76 * sigma |
| 95% | 2.145 | +/- 2.14 * sigma |

These confidence intervals represent the range within which the actual wind speed at the target site is expected to fall, relative to the flow model prediction, at the specified confidence level.

### Driver Attribution

A unique feature of the deployment tool is its ability to decompose the predicted sigma into percentage contributions from each feature, identifying which uncertainty drivers dominate for a given project. The decomposition proceeds as follows:

For each feature j, the contribution to log(sigma) is:

    contribution_j = gamma_j * z_j

The total log(sigma) above the baseline is:

    sum(contribution_j) = log(sigma) - log(sigma_0)

The percentage contribution of feature j is:

    pct_j = contribution_j / sum(contribution_k) * 100%

(Only features with positive contributions are included in the percentage calculation; features at or below their training-set mean contribute zero.)

This driver attribution enables actionable recommendations. For example:

- If dist_norm dominates the uncertainty budget, the recommendation is to consider an additional measurement mast closer to the target turbines, or to use a higher-fidelity flow model (e.g., CFD instead of WAsP) for the long-distance transfer.

- If sample_shortfall is a major contributor, the recommendation is to extend the measurement campaign. The logarithmic scaling of sample_shortfall means that the improvement from extending a 6-month campaign to 12 months is approximately the same as the improvement from extending a 12-month campaign to 24 months.

- If turning_gradient is dominant, the recommendation is to pay particular attention to the terrain features causing rapid changes in flow direction, and to consider whether the linear flow model's terrain representation is adequate in those sectors.

- If k_MM_deviation is significant, the recommendation is to verify that the wind speed distribution at the predictor mast is well-characterised and that the Weibull fit is appropriate, as unusual distribution shapes may indicate non-standard atmospheric conditions (e.g., bimodal wind patterns from competing weather systems).

### Excel Output Format

The deployment tool produces an Excel workbook with three sheets:

**Summary sheet:** Contains the project name, predictor mast and target location identifiers, the predicted pair-level sigma (energy-weighted aggregation across sectors), confidence intervals at standard levels (P50, P75, P90, P95), and a traffic-light classification (green: sigma < 5%, yellow: 5% < sigma < 10%, red: sigma > 10%).

**Features sheet:** Contains the six feature values for the project, both raw and z-scored, along with the training-set mean and standard deviation for each feature. This allows the user to see how the project compares to the training population for each feature and to verify the input data.

**Contributions sheet:** Contains the driver attribution breakdown, showing the percentage contribution of each feature to the total sigma, along with the sigma multiplier contributed by each feature. This sheet enables the actionable recommendations described above.

This three-sheet format was designed through consultation with wind resource engineers to integrate naturally into existing project documentation workflows. The Excel format is universally accessible in the industry, requires no specialised software, and can be directly referenced in wind resource assessment reports.

### Input Tool

A companion input tool, implemented as a Tkinter graphical user interface (GUI), assists users in preparing model inputs from windPRO and WAsP output files. The GUI presents fields for each required feature, with tooltips explaining the feature definitions and guidance on where to find the values in the windPRO/WAsP output. The tool validates input ranges against the training data distribution and warns the user if any feature value falls outside the training range (extrapolation), which may reduce the reliability of the prediction.

The input tool also provides a direct path to the uncertainty calculator: after all features are entered, the user can click a "Calculate" button that runs the prediction computation and generates the Excel output workbook in a single step.


---

## 2.11 Software and Computational Details

### Software Stack

The entire modelling framework is implemented in Python 3.x, leveraging the following key libraries:

- **PyMC (version 5.x):** The probabilistic programming framework used for specifying and fitting the Bayesian hierarchical model [6]. PyMC provides a high-level interface for defining probabilistic models, automatic computation of log-probability gradients (via Aesara/PyTensor), and efficient MCMC sampling algorithms.

- **ArviZ:** The Bayesian data analysis library used for posterior diagnostics, convergence checking, and model comparison. ArviZ provides InferenceData objects that store all MCMC output in a standardised format, along with functions for computing R-hat, effective sample sizes, WAIC, and LOO cross-validation scores.

- **NumPy and Pandas:** Standard numerical computing and data manipulation libraries used throughout the data pipeline and modelling code.

- **SciPy:** Used for statistical functions (Weibull fitting, Student-t quantile computation, correlation tests) and for the special functions (Gamma function) needed for the expected absolute error formula.

- **Plotly:** Used for interactive HTML-based visualisations, including the pair-level scatter plots and calibration diagnostics that support exploratory analysis and model validation.

- **Matplotlib:** Used for static publication-quality figures, including trace plots, posterior distributions, prior predictive checks, and the final model diagnostics.

### MCMC Configuration

The primary model is fitted using the No-U-Turn Sampler (NUTS) [5, 6], an adaptive Hamiltonian Monte Carlo algorithm that automatically tunes the step size and trajectory length. The MCMC configuration for the primary model fit is:

| Parameter | Value |
|-----------|-------|
| Number of chains | 4 |
| Tuning iterations per chain | 2,000 |
| Posterior draws per chain | 2,000 |
| Total posterior draws | 8,000 |
| Target acceptance rate | 0.95 |
| Maximum tree depth | 10 (default) |

The target acceptance rate of 0.95 is higher than the default of 0.80, reflecting the model's hierarchical structure with multiple scales. Higher target acceptance rates produce smaller step sizes and longer trajectories, which improve the sampler's ability to navigate the hierarchical geometry (particularly around the pair random effects and their scale parameter). The trade-off is longer computation time per sample, but this is acceptable given the modest dataset size and the importance of reliable posterior estimates.

For the LOO cross-validation folds, a reduced MCMC budget is used to keep the total computation time manageable:

| Parameter | Value |
|-----------|-------|
| Number of chains | 2 |
| Tuning iterations per chain | 1,000 |
| Posterior draws per chain | 1,000 |
| Total posterior draws | 2,000 |
| Target acceptance rate | 0.95 |

With 22 physical-pair folds and approximately 5 minutes per fold, the total LOO CV computation time is approximately 2 hours on the workstation described below.

### Convergence Diagnostics

Model convergence is assessed using three complementary criteria:

1. **R-hat statistic:** The potential scale reduction factor, which compares the between-chain variance to the within-chain variance for each parameter [5]. R-hat values close to 1.0 indicate that all chains have converged to the same posterior distribution. The criterion R-hat < 1.01 is applied to all model parameters; values above 1.01 trigger extended tuning or investigation of model parameterisation issues.

2. **Effective sample size (ESS):** The estimated number of independent posterior samples, accounting for autocorrelation within each chain. The criterion ESS > 400 per parameter ensures that posterior summaries (medians, standard deviations, credible intervals) are estimated with acceptable precision. For a model with approximately 15 parameters (6 gammas, log_sigma_0, nu, sigma_pair, and the eta_p vector), this criterion requires at least 6,000 effectively independent samples across the 8,000 total draws.

3. **Visual trace plot inspection:** Trace plots showing the time series of MCMC samples for each parameter are visually inspected for signs of poor mixing (long autocorrelation, mode-switching, trend) or non-stationarity (drift over the course of the chain). While less formal than R-hat and ESS, visual inspection can detect pathologies that summary statistics miss, such as multimodal posteriors where chains are stuck in different modes.

All three diagnostics are computed automatically by ArviZ and included in the model output. In the final model, all parameters achieved R-hat < 1.005 and ESS > 1,000, indicating excellent convergence.

### Hardware

All computations were performed on a standard workstation running Windows 11, without GPU acceleration. The PyMC NUTS sampler uses CPU-based gradient computation via PyTensor. A typical full model fit (4 chains, 2,000 tuning, 2,000 draws) completes in approximately 8--12 minutes. The complete LOO cross-validation (22 folds) completes in approximately 1.5--2.5 hours.


---

## References

[1] Troen, I. and Petersen, E.L. (1989). *European Wind Atlas*. Riso National Laboratory, Roskilde, Denmark.

[2] Landberg, L., Mylde, L., Petersen, E.L., Rathmann, O., and Mortensen, N.G. (2003). Wind Resource Estimation -- An Overview. *Wind Energy*, 6(3), 261--271.

[3] Mortensen, N.G. and Petersen, E.L. (1997). Influence of Topographical Input Data on the Accuracy of Wind Flow Modelling in Complex Terrain. *Proceedings of the European Wind Energy Conference*, Dublin, Ireland.

[4] FGW -- Foerdergesellschaft Windenergie und andere Dezentrale Energien (2023). *Technical Guideline TR6: Determination of Wind Potential and Energy Yields*. Revision 11.

[5] Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., and Rubin, D.B. (2013). *Bayesian Data Analysis*, 3rd Edition. CRC Press, Boca Raton, FL.

[6] Salvatier, J., Wiecki, T.V., and Fonnesbeck, C. (2016). Probabilistic Programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.

[7] Clifton, A., Kilcher, L., Lundquist, J.K., and Fleming, P. (2016). Using Machine Learning to Predict Wind Turbine Power Output. *Environmental Research Letters*, 11(2), 024009.

[8] Brower, M. (2012). *Wind Resource Assessment: A Practical Guide to Developing a Wind Project*. Wiley, Hoboken, NJ.

[9] IEC 61400-12-1 (2017). *Wind energy generation systems -- Part 12-1: Power performance measurements of electricity producing wind turbines*. International Electrotechnical Commission, Geneva, Switzerland.

[10] Jackson, P.S. and Hunt, J.C.R. (1975). Turbulent wind flow over a low hill. *Quarterly Journal of the Royal Meteorological Society*, 101(430), 929--955.
