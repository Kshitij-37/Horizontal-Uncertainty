# The road to exp-M2c: a story of the last week's model rework

*Written for future-you (or a future collaborator) who needs to understand not
just what the shipping model is, but why it looks the way it does — and what
alternatives were tested along the way.*

---

## Where we started

**The thesis-final model** — a Bayesian pair-level regression predicting the
uncertainty (sigma) of horizontal wind-speed extrapolation. Fit on 38 directional
pairs (19 physical pair sites, minus 14 excluded masts), with:

```
e_overall_i ~ StudentT(nu, mu_i, sigma_i)

log(sigma_i) = log_sigma0
             + gamma_dist    * z(dist_sat)
             + gamma_turning * z(turning_sat)
             + gamma_speedup * z(wm_abs_log_speedup)
             + gamma_rough   * z(roughness_sat)      # formula A: |(rs_W + rs_M)/2|
             + gamma_dz      * z(dz_sat)

mu_i = beta_dz * z(dz_i)         # a small SIGNED bias predictor
```

**Ship-quality LOO metrics:** Pearson 0.926, Spearman 0.746.

**The wart:** if you asked this model "what sigma would you predict when I put a
new turbine at the same location as the mast?" you got different answers for
different masts:
- Flat sites: 0.51%
- Rough forest-like sites: up to 2.05%

That 4× spread came from the fact that roughness_sat is a MAGNITUDE feature (not
a mismatch). At the same site, it doesn't vanish — the mast's own rough_speedup
value still populates the formula. The other 4 features DO vanish at same-site
(they're proper mismatches), but roughness doesn't. Terrain-dependent floor.

Meanwhile, the supervisor was skeptical of the bias term (`mu = beta_dz * dz`).
The user decided to drop it — not worth pushing a concept that isn't landing.

## Two problems to solve

1. **Terrain-dependent floor** — for the tool to be defensible in repowering
   scenarios, same-site sigma should be a single number, not "0.51% or 2.05%
   depending on your terrain."
2. **Bias term** — needs to go, so the model becomes sigma-only. Target
   changes from signed `e_overall` to `|e_overall|`; likelihood becomes
   HalfStudentT (sigma retains its "underlying two-sided scale" meaning, so
   downstream P75/P90 multipliers still work).

## Path 1: the additive-floor detour

First attempt: reparameterize the model as `sigma = sigma_floor + sum_k beta_k * f_k`
where the features are un-saturated raw values. Idea: give the floor an
explicit parameter that's tuneable, decouple it from the slopes.

Tried this with different roughness variants. Result:
- Additive floor with saturated features: ceiling capped at ~9% while actual max
  observed |e| was 18%. Under-predicts the tail.
- Additive floor with un-saturated features: ceiling ~12%. Still under-predicts.

**Rejected as the "lazy fix."** We were trading fit ceiling to get a clean
floor. The multiplicative `exp()` form fits genuinely better on the tail; we
shouldn't give it up.

## Path 2: exp-Q (distance-gated roughness)

Next attempt: keep `sigma = exp(...)` but two changes:
- Un-center the features (divide by std only, no mean subtraction) — this makes
  `log_sigma0` a physically-meaningful "value at all-features-zero" rather than
  a mathematical intercept.
- Multiply the roughness feature by `(1 - exp(-d/500))` — a "distance gate"
  that forces roughness to vanish at d=0.

Result: clean single-value floor of 0.44%, LOO Pearson 0.9347 (basically same
as thesis-final's 0.9294 for a comparable sigma-only fit). Looked like a win.

**User pushback:** the 500m gate is philosophically awkward. A distance scale
chosen empirically to make the math work, not derived from physics. You'd have
to defend "why 500m and not 300m or 700m?" in the thesis. Reasonable concern.

## Path 3: adaptive min-family

Fresh idea: instead of gating with distance, use an ADAPTIVE roughness feature
that inherently vanishes at same-site because of the roughness mathematics.
Combine formula A (magnitude) and formula B (mismatch `|rs_W - rs_M|`) via a min:

```
roughness_M = min( formula_A, formula_B )     # per-pair, zero hyperparameters
```

Why min? At same-site: rs_W = rs_M, so formula B = 0. So min = 0 regardless of
formula A. Clean floor without any distance gate. Physics story: "when the two
sites' roughness treatments are similar (small B), we trust the smaller
dissimilarity measure; when they diverge, min picks whichever is smaller as a
conservative estimate."

But there are TWO independent design choices around this min:

- **When to take the min:** on raw values or after standardization?
- **When to center:** subtract training mean or not?

That gives 4 variants (a 2×2 grid):

| variant | when to min | centering |
|---|---|---|
| exp-M   | min raw values first, then divide by single std | no centering |
| exp-Mc  | same as exp-M but with centering (subtract training mean of the min-values) | yes centering |
| exp-M2  | standardize A and B separately by their own std, THEN min | no centering |
| exp-M2c | z-score A and B separately (center + scale), THEN min | yes centering |

**exp-M2c is the mathematically most principled** — it respects that formulas A
and B have their own natural spreads (different std_A vs std_B), so each should
be scaled by its own distribution before being combined.

All 4 tested via `sigma_only_refit/refit_sigma_only.py`, both forest-in and
forest-out. Full LOO metrics saved.

## Path 4: the mismatch-formula sanity checks

Just to make sure we weren't fooling ourselves with the adaptive family, we also
tested pure formulas:

- **exp-B** (pure signed mismatch): LOO Pearson 0.870 — bad.
- **exp-T** (magnitude-mismatch `||rs_W| - |rs_M||`): also 0.869 — bad.

Both are proper "vanishes at same-site" features, but they lose the magnitude
signal that WAsP's roughness correction magnitude genuinely provides. Conclusion:
**you cannot dodge magnitude — WAsP applies bigger corrections in rougher
terrain and those bigger corrections carry more model error. Magnitude is real
signal, not just double-counting log_speedup.**

Both confirmed: pure-mismatch formulations don't work. The adaptive min family
gets both the magnitude signal (via formula A when it wins the min) AND the
vanishing-at-same-site behavior (via formula B being zero at same-site).

## The Herzhausen scare (and its resolution)

Sign-cancellation analysis: 17.5% of sector rows have opposite-sign `rs_WTG`
vs `rs_MM`. Under formula A, opposite-sign values cancel toward zero in
`|(rs_W + rs_M)/2|`. Concern: if the min operation picks A for a pair with
high sign-cancellation but genuinely high uncertainty, we'd under-predict.

Herzhausen fit that profile — 4 pairs with 6/12 opposite-sign sectors and
observed |e| of 14-18%. Would the min-formula fail here?

**Ran the feature-contribution decomposition** (`herzhausen_decomposition.py`)
to see. Result: for these pairs, roughness contributes only 18-26% of log(sigma);
TURNING contributes 26-31% (the dominant driver). exp-A and exp-M give
essentially identical predictions on Herzhausen (within 0.5 pp), because
turning and other complexity features carry the load and the model rebalances
whatever slack the roughness contribution leaves.

**Complexity of terrain dominates roughness for these hard cases** — a finding
the user had physical intuition for all along. The min-formula's theoretical
weakness on cancellation doesn't manifest as a fit problem.

## The variance-decomposition methodology fix

Along the way, I made a rookie mistake: I said "roughness dropped from 34%
variance under thesis-final to 20% under exp-M2c" — but I was comparing NAIVE
percentages (from the thesis doc's `beta_k²` decomposition) to LMG percentages
(from a proper decomposition script I'd run on exp-M2c). Apples to oranges.

**The user caught it and asked "did you actually run LMG on both models?"** Fair
question. I hadn't. So I wrote `compare_lmg_thesis_vs_expM2c.py` and ran a
proper apples-to-apples comparison.

Result: distance's supposedly-huge share (naive: 26%) collapsed to 15% under
proper LMG — very close to thesis-final's 14.6%. The naive number was inflated
because it uses the sum-of-squared-betas as denominator, which UNDER-counts the
actual variance (which includes 52% of contribution from pairwise cross-terms).

Documented the whole naive-vs-LMG story in `variance_decomposition_explained.md`.
The rule of thumb: **when features are correlated at 0.3+ (ours go up to 0.9),
always cite LMG. The naive decomposition is misleading.**

## The variance-decomp results across all variants

Once I had LMG working, I ran it on all 5 viable candidates
(`lmg_all_variants_comparison.py`):

| variant | n | roughness LMG | complexity LMG |
|---|---|---|---|
| thesis-final | 41 | 23.8% | 76.2% |
| exp-A forest-out | 41 | 24.9% | 75.1% |
| exp-M2c forest-out | 41 | 28.8% | 71.2% |
| exp-M2c forest-in (n=45, no Slovenska) | 45 | 19.3% | 80.7% |
| **exp-M2c n=47 (+ Slovenska East)** | 47 | **18.8%** | **81.2%** |
| exp-Q forest-out | 41 | 30.6% | 69.4% |

**The key surprise:** exp-M2c only delivers "less roughness dominance" when
forest is INCLUDED. Forest-out, exp-M2c is actually more roughness-dominant
than thesis-final. The forest pairs are what activate the min-picks-B behavior
that constrains the roughness coefficient during fitting.

This is a real physical insight: the min operation SHRINKS roughness's role
only in datasets where formula B is meaningfully non-zero for some pairs, which
is exactly the case for forest pairs (both sites in rough terrain, so B is
small relative to A).

## The final expansion to n=47

Last step: added Slovenska East (2024PA107) alongside the forest pairs. It had
been excluded for missing displacement height data, but user judged it
includable for the exp-M2c framework. Result:

- Roughness LMG dropped further: 19.3% → 18.8%
- Complexity LMG rose: 80.7% → 81.2%
- LOO Pearson essentially unchanged: 0.887 → 0.886
- Self-σ still single-value ~0.45%

Every extra reasonable pair we added pushed roughness dominance down. The
pattern is consistent: **more diverse training data → the adaptive formula
relies less on the roughness signal.**

## What the shipping model looks like

```
|e_overall_i| ~ HalfStudentT(nu, sigma_i)         # sigma-only, no bias

log(sigma_i) = log_sigma0
             + gamma_dist    * (dist_sat - mean_dist) / std_dist
             + gamma_turning * (turning_sat - mean_turning) / std_turning
             + gamma_speedup * (wm_abs_log_speedup - mean_speedup) / std_speedup
             + gamma_dz      * (dz_sat - mean_dz) / std_dz
             + gamma_rough   * min(z_A, z_B)

where:
  z_A = (roughness_sat_A - mean_A_train) / std_A_train
  z_B = (roughness_sat_B - mean_B_train) / std_B_train

  roughness_sat_A = 1 - exp( -weighted_mean(|(rs_W + rs_M)/2|) / 0.01 )
  roughness_sat_B = 1 - exp( -weighted_mean(|rs_W - rs_M|)     / 0.01 )
```

Fit on 47 pairs. 10 masts still excluded (Sallachy complex terrain, Kayislar
insufficient directions, Herzhausen CFD, Taaibos multi-mast, Ukhanda multi-mast
+ LiDAR, Balver Wald uncertain). Forest projects and Slovenska East are IN.

Sampler: 4 chains × 2000 draws + 2000 tune. LOO CV holds out each of 47 pairs
(one at a time, with per-fold standardization to avoid data leakage).

## What we did NOT do (and why)

- **Kept dz feature** even though it correlates 0.90 with speedup. User's
  argument: dz is a bulk pair-level scalar; turning and speedup are per-sector.
  The correlation is a data-artifact of "flat = far-apart with low dz". Also:
  dz keeps growing linearly with elevation while turning/speedup saturate — dz
  is extrapolation insurance for out-of-sample sites.
- **Did not add new features** (dRIX, sample_shortfall, TI_MM, etc.). User was
  firm: feature count stays at 5.
- **Did not test z0 (reference roughness length)** as replacement for
  rs (roughness speedup). Physical reason: we're modeling WAsP's extrapolation
  uncertainty, so we need features that capture WAsP's own processing (rs is
  WAsP's OUTPUT). z0 is the INPUT to WAsP; using it directly loses visibility
  into WAsP's own error.
- **Did not reintroduce Hultema and Malarberget with formula A** — that had
  been tested exhaustively in prior sessions and always hurt fit. Only re-included
  now because the adaptive M2c formula handles them differently (min picks B
  for same-forest pairs → smaller roughness feature values → constrained
  coefficient).

## Known limitations

- **LOO Pearson dropped from 0.926 (thesis-final with bias) to 0.886**. Real
  cost. We bought clean floor, forest inclusion, no bias, and LMG-defensible
  attribution.
- **Doringbaai pairs (16.7 km) under-predict** slightly (~0.02 pp). This is
  where the min-formula's "picks A when both are moderate" behavior slightly
  under-fires. Both exp-A and exp-M give similar predictions here — the model
  can't do much better with the current feature set.
- **Forest pairs still over-predict** (Hultema pred ~0.08, actual ~0.03; Malarberget
  pred ~0.03, actual ~0.005). Not as badly as under formula A, but the model
  doesn't fully capture whatever makes forest sites easier to predict than the
  features suggest. Something outside the current 5-feature framework.
- **exp-M2c's self-σ is 0.45%** — a single number for repowering. Cleanly
  defensible as "the intrinsic WAsP self-prediction floor when transfer effects
  vanish."

## Documents to consult

For different questions:

| question | file |
|---|---|
| what is the shipping model? | `Post-thesis corrections/ws_uncertainty_model_pairlevel_expM2c.py` |
| what are the fitted numbers? | `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_results.json` |
| what is each mast's same-site sigma? | `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_self_prediction.csv` |
| what drives each pair's prediction? | `Post-thesis corrections/Results_expM2c/ws_uncertainty_expM2c_feature_contributions.csv` |
| why LMG not naive decomposition? | `Post-thesis corrections/variance_decomposition_explained.md` |
| what are the M-family variants? | `Post-thesis corrections/adaptive_M_variants_explained.md` |
| how did we get here? | this file |
| project-level context and conventions | `CLAUDE.md` (project root) |

## Remaining work

1. **Calculator integration** — none of the shipping calculators
   (`Uncertainty calculator/v3.0`, `v4.0`, `TR_FinalModel`) load the exp-M2c
   JSON yet. They all use the pre-bias-drop, pre-adaptive-roughness prediction
   path. Need to update ONE of them (probably `TR_FinalModel` since it's the
   simplest single-model variant) to load exp-M2c and compute the per-formula
   z-scored min at inference time. Rebuild the exe.

2. **Thesis defense addendum** — a 2-3 page document capturing the story above
   in defense-quality prose. Most raw material is here or in the two other
   `.md` files.

3. **Optional but valuable** — a `expM2c_recalibration.py` script that reads
   the fitted JSON and recomputes P75/P90/P99 quantile predictions for a
   representative set of pairs, verifying HalfStudentT calibration against
   observed |e| quantiles. If well-calibrated, that's a strong defense line.

## A note on the process

We tried a lot of things. Many were dead ends (additive floor, exp-B, exp-T,
exp-Q with the gate defense concern, etc.). We reversed course multiple times
based on either physical intuition or better analysis. The whole enterprise
took about a week of concentrated iteration.

Two moments were pivotal:

1. When the user pointed out the 500m gate in exp-Q was awkward, and reframed
   the goal as "we want an inherently-vanishing roughness feature, not a gated
   one." That opened the adaptive-min family.

2. When the user asked "did you actually run LMG on both models" and forced me
   to redo the variance decomposition properly. That corrected a misleading
   comparison and revealed the true picture — including that exp-M2c only
   satisfies the "less roughness dominance" goal when forest is included.

Both times, catching my sloppiness led to a better final model. Worth
remembering: careful comparison and challenging assumptions matter more than
running one more variant of a fit.

---

**Final status (2026-08):** exp-M2c with n=47 is ready to promote to production.
The model script, results JSON, self-prediction table, feature-contribution
breakdown, LMG decomposition, and diagnostic documentation are all in place.
Remaining task is the calculator integration and thesis defense writeup.
