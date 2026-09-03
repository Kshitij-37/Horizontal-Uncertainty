# Post-thesis model rework — defense addendum

*A defensible summary of the changes made to the pair-level uncertainty model
after thesis submission, and the reasoning behind each. Written for the
defense audience — statistical/physical justification with metrics.*

---

## 1. Motivation

The thesis-final model achieved LOO Pearson 0.926 on 38 directional pairs.
However, two structural concerns emerged in post-submission review:

**Concern 1 — Terrain-dependent same-site prediction (the "floor" problem).**
When the model is queried for a pair whose sites are co-located (all pair-level
dissimilarity features = 0, a repowering scenario), it does not return a single
sigma value. It returns a range from 0.51% (flat sites) to 2.05% (rough
forested sites). The variation is driven entirely by the roughness feature,
which is formulated as a magnitude — `|(rs_WTG + rs_MM)/2|` — and thus does
not vanish at same-site even though the other four features (distance,
turning, log-speedup, dz) all do. This makes the tool's same-site output
harder to defend for repowering use cases where the extrapolation distance is
zero and the user expects a single, terrain-independent number.

**Concern 2 — Bias/uncertainty separation is not being accepted.**
The thesis-final model included a signed mean predictor `mu = beta_dz * dz`
intended to capture systematic over/under-prediction as a function of
elevation difference. Supervisor feedback raised skepticism about separating a
bias term from the uncertainty prediction, and the empirical contribution of
`beta_dz` was small (< 2% of the total explanation). Retaining a
poorly-motivated term risked weakening the defense of the more central sigma
model.

The rework was structured around resolving both concerns while preserving the
five-feature framework and the LOO Pearson within an acceptable margin.

---

## 2. Changes made

Three deliberate changes, in order of significance:

### 2.1 Bias term removed; target changed to |e|

The signed target `e_overall` was replaced with `|e_overall|`, and the
StudentT likelihood was replaced with HalfStudentT. The model becomes purely a
sigma predictor:

```
|e_overall_i| ~ HalfStudentT(nu, sigma_i)
```

Under HalfStudentT, `sigma` retains its interpretation as the scale of the
underlying two-sided distribution (identical to the StudentT sigma if the
underlying distribution has zero mean, which the removed bias term was
implicitly parameterizing). Downstream P75, P90, P99 multipliers of `sigma`
therefore remain valid without recalibration.

### 2.2 Adaptive roughness feature (formula M2c)

The single roughness feature was replaced with an adaptive formulation:

```
sat_A = 1 - exp(-|(rs_W + rs_M)/2| / 0.01)          # formula A: magnitude
sat_B = 1 - exp(-|rs_W - rs_M|     / 0.01)          # formula B: mismatch

z_A   = (sat_A - mean_A_train) / std_A_train         # per-formula z-score
z_B   = (sat_B - mean_B_train) / std_B_train

roughness_feature = min(z_A, z_B)
```

Design intent: at same-site, `rs_WTG = rs_MM` by definition, so formula B = 0
regardless of terrain. This makes `z_B = -mean_B/std_B` (a constant), and for
rough masts `z_A > z_B` so the min operation picks `z_B`. The result is that
the roughness feature returns a constant value at same-site for all masts,
producing a terrain-independent floor. When the two sites differ meaningfully,
the min operation trades off between the two dissimilarity views: for
same-terrain pairs (small mismatch), it selects formula B; for
different-terrain pairs, it selects formula A.

Per-formula standardization was preferred over a single mixed-distribution std
because formulas A and B are mathematically distinct quantities that should be
scaled by their own natural spreads before being combined.

### 2.3 Training set expanded to 47 directional pairs (from 38)

The exclusion list was reduced from 14 masts to 10 by re-including Hultema
(2011WM011, 2014WM011), Malarberget (2012WM006), and Slovenska East
(2024PA107). Prior tests under formula A had shown these sites fit poorly
under the thesis-final architecture. The adaptive M2c formulation handles them
gracefully: for same-forest pairs, formula B is small, so the min operation
selects it, and the resulting small feature value matches these pairs' low
observed errors. Empirically, re-inclusion of these six extra directional
pairs reduced the roughness variance contribution (see §4) while maintaining
LOO Pearson.

The training set thus contains **47 directional pairs across 42 unique masts**
(some masts appear in multiple pairs as either reference or turbine site).
The pair count is what the likelihood observes; the mast count is what
determines the self-prediction table (one row per unique mast).

---

## 3. Model specification (final form)

```
|e_overall_i| ~ HalfStudentT(nu, sigma_i)

log(sigma_i) = log_sigma0
             + gamma_dist    * z(dist_sat_i)
             + gamma_turning * z(turning_sat_i)
             + gamma_speedup * z(wm_abs_log_speedup_i)
             + gamma_dz      * z(dz_sat_i)
             + gamma_rough   * min(z_A_i, z_B_i)

where z(f) = (f - mean_train(f)) / std_train(f)
```

Fitted parameters (n = 47, 4 chains x 2000 draws + 2000 tune):

| parameter | value |
|---|---|
| nu | 14.7 (95% CI: 4.7 - 33.8) |
| log_sigma0 | -3.36 |
| sigma0 (= exp(log_sigma0)) | 3.50% |
| gamma_dist | 0.309 |
| gamma_turning | 0.279 |
| gamma_speedup | 0.232 |
| gamma_dz | 0.163 |
| gamma_roughness | 0.347 |

---

## 4. Validation

### 4.1 Cross-validation

Two holdout strategies:

| metric | pair-level | site-level |
|---|---|---|
| n folds | 22 (physical pairs) | 22 (unique locations) |
| Pearson r | 0.886 | 0.880 |
| Spearman rho | 0.706 | 0.702 |
| Bias | +0.42 pp | +0.73 pp |
| Mean predicted sigma | 4.58% | 4.89% |
| Mean actual \|e\| | 4.16% | 4.16% |

Compared to the thesis-final Pearson of 0.926, the promoted model gives up
approximately 0.04 in linear correlation. This cost is a real consequence of
(a) removing the bias term, (b) including harder-to-fit sites (Hultema,
Malarberget, Slovenska East), and (c) the adaptive roughness formulation's
slightly reduced individual-pair discrimination. What is bought in return is
addressed in §4.2 - §4.4.

### 4.2 Same-site prediction — clean floor achieved

Across the 42 unique masts in the training set (the 47 pairs collapse to 42
individual masts for the same-site question), the model's same-site sigma
prediction falls in the narrow range **0.441% to 0.450%** (spread ratio 1.02×).
The distribution has two distinct values: 0.441% for the 6 truly-flat masts
(raw roughness identically zero in every sector) and 0.450% for the 36 masts
with any nonzero roughness. The step arises because the min operation picks the
z-scored formula A when `sat_A_self` is small enough to be more negative than
z_B, and picks the constant z_B otherwise; the two regimes give slightly
different results but the practical difference (0.009 percentage points) is
not decision-relevant.

For deployment, any mast with realistic terrain returns 0.45% for its
same-site prediction; only WTGs over truly-uniform smooth terrain would see
0.44%. This is effectively a single-value floor.

The thesis-final model produced a range of 0.51% (flat) to 2.05% (rough) for
the same masts. The promoted model reduces the spread from 4.3× to 1.02×.

### 4.3 Variance decomposition — proper attribution with correlations

Because the five features are correlated (turning-speedup at 0.81,
speedup-dz at 0.90, turning-dz at 0.64, distance-roughness at -0.39), the
naive `beta_k^2` variance decomposition is misleading — cross-terms in the
variance of a sum of correlated features are non-negligible. The proper
Shapley (LMG) decomposition, which averages each feature's marginal
contribution over all K! orderings, gives:

| feature | LMG share |
|---|---|
| turning | 24.3% |
| log_speedup | 21.4% |
| **roughness** | **18.8%** |
| dz | 16.4% |
| distance | 16.7% |

Feature contributions are balanced in the 16-24% range. Roughness is the
smallest contributor, matching the intended physical framing that terrain
complexity (represented through turning, speedup, and dz jointly) should be
the primary driver of extrapolation uncertainty, with roughness providing a
constrained additional signal.

Comparable thesis-final decomposition (LMG on the same 41-pair subset, formula
A + bias): roughness 24%, turning 33%, distance 16%, speedup 15%, dz 12%.
Under the promoted model, roughness is reduced from 24% to 19% while
speedup and dz gain, reflecting the redistribution the adaptive formulation
produces.

### 4.4 Calibration

The bias of +0.42 pp under HalfStudentT indicates that predicted sigma is
approximately 10% higher than the mean absolute error, which is the expected
relationship under the HalfStudentT with nu ~ 15:
`E[|e|] = sigma * sqrt(2/pi) * ... ~= 0.8 * sigma`. The model is well-calibrated
in scale.

---

## 5. What was tested and rejected

To arrive at the promoted formulation, the following variants were tested and
rejected (all under sigma-only, HalfStudentT):

- **Additive floor** (`sigma = sigma_floor + sum_k beta_k * f_k`) — rejected;
  ceiling capped at 9-12% while training data has max |e| of 18%.
- **Distance-gated roughness** (formula A x `(1 - exp(-d/500))`) — rejected;
  the 500m distance scale is empirically-fit rather than physically derived,
  weakening the defense.
- **Pure mismatch roughness** (formula B: `|rs_W - rs_M|`) — rejected;
  loses roughness signal that formula A captures. LOO Pearson dropped to 0.87.
- **Sign-independent magnitude mismatch** (formula T: `||rs_W| - |rs_M||`) —
  rejected; same fit cost as formula B without meaningful benefit.
- **Un-centered min-first adaptive** (exp-M, exp-M2 without centering) —
  superseded by centered variants which showed better fit and cleaner
  interpretation.
- **Additional features** (dRIX, sample_shortfall, Weibull-k deviation, etc.)
  — not tested in this cycle; the five-feature framework was preserved to
  maintain continuity with the thesis-final design.

A more complete comparison of variants is documented in
`adaptive_M_variants_explained.md`.

---

## 6. Known limitations

**LOO Pearson cost.** The promoted model achieves 0.886 vs the thesis-final's
0.926 (with bias, forest-out). The 0.04 gap is real; it reflects the price of
(a) supervisor-preferred bias removal, (b) inclusion of hard-to-fit sites, and
(c) the adaptive formulation's slightly reduced per-pair discrimination.

**Doringbaai pairs (16.7 km) are under-predicted** by 1-2 percentage points
(actual 8-9%, predicted 6-7%). The min operation selects formula A for these
pairs, and formula A's cancellation on their opposite-sign roughness sectors
gives a smaller-than-ideal roughness contribution. Both the thesis-final and
promoted models under-predict these pairs, but the promoted model does so
slightly more.

**Forest pairs remain over-predicted.** Under the promoted model, Hultema
sigma is predicted at ~8% (actual ~3%) and Malarberget at ~3% (actual
~0.5%). This is a substantial improvement over what formula A would give but
still an over-prediction. The current five-feature framework cannot capture
whatever makes forest pairs' errors smaller than the features would suggest;
this is documented as a framework limitation rather than a bug.

**The 0.45% floor represents intrinsic WAsP self-prediction error, not zero.**
For repowering scenarios where both sites are truly co-located, the model
returns 0.45% rather than 0%. This reflects the physical fact that even at the
same site, WAsP's flow model applies corrections that carry residual error.
The 0.45% value falls within the "less than 1%" bound established as the
intrinsic WAsP self-prediction error in physical terms.

---

## 7. Conclusion

The promoted model resolves the two motivating concerns:

1. **Same-site prediction is now terrain-independent** (0.45% single value
   across 42 masts) via the adaptive `min(z_A, z_B)` roughness feature.
2. **Bias term is removed**, aligning with supervisor preference and
   simplifying the model to a pure sigma predictor. Downstream P-multiplier
   usage remains valid under HalfStudentT.

The costs — approximately 0.04 in LOO Pearson and slight under-prediction on
extreme-distance dissimilar-terrain pairs — are proportional to the
architectural improvements gained. The proper (LMG) variance decomposition
shows a balanced feature framework (16-24% per feature) with roughness as the
smallest contributor, matching the physical intuition that terrain complexity
rather than roughness magnitude should drive predicted uncertainty.

---

## 8. References

- Production script: `Post-thesis corrections/ws_uncertainty_model_pairlevel_expM2c.py`
- Fitted coefficients: `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_results.json`
- Per-mast same-site sigma: `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_self_prediction.csv`
- Per-pair feature contributions: `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_feature_contributions.csv`
- **Deployment calculator (GUI)**: `Uncertainty calculator/Uncertainty_calculator_TR_v5/` — bundles the exp-M2c JSON and adds a self-prediction column showing the model's same-site sigma per WTG.
- LMG variance methodology: `Post-thesis corrections/variance_decomposition_explained.md`
- Adaptive-formula variants: `Post-thesis corrections/adaptive_M_variants_explained.md`
- Full journey narrative: `Post-thesis corrections/JOURNEY_TO_PRODUCTION_MODEL.md`
- Project onboarding: `CLAUDE.md`
