# CLAUDE.md — Bayesian Wind Speed Uncertainty Project

Onboarding note for any Claude session opening this project. Read this first; verify against current code before asserting anything as fact.

## 1. What this project is

A Bayesian regression model that predicts the uncertainty (sigma) of horizontal wind speed extrapolation from a measurement mast (MM) to a planned wind turbine (WTG) location, using WAsP/windPRO flow-model outputs as features. Written for a wind-energy MSc thesis (submitted; defense pending). The model output feeds an EYA (Energy Yield Assessment) calculator that a project engineer uses to add a horizontal-extrapolation uncertainty component on top of WAsP's P50 prediction.

## 2. Two firm user decisions (do NOT propose reversing without explicit ask)

**A. Drop bias — predict uncertainty only.** The `mu = beta_dz * dz` bias term is being dropped from the model. Reason: the user's supervisor is not accepting the bias concept, and the user is unwilling to push it. Going forward the model is StudentT(nu, mu=0, sigma) — sigma-only, zero-mean assumption. Any rework must remove the beta_dz term entirely; do not include it in new fits or in the calculator JSON. Do NOT propose re-introducing bias unless the user asks.

**B. Forest projects (Hultema, Malarberget) stay excluded.** The user has extensively tested re-including them and performance has dropped in every single case. Do not propose forest re-inclusion as a first-order investigation. The only condition that would justify revisiting is if a proposed model change involves a widely different concept of predicting uncertainty (fundamentally different feature set), where forest data might interact with the new features differently. Do NOT propose "let's try forest-in with the current-ish features again" — that's been done.

**C. Feature set is fixed at the current 5 — do NOT propose adding, dropping, or replacing features.** The 5 features (distance, turning, log_speedup, roughness, dz) were chosen after extensive testing during the thesis. In particular:
- **Do not propose dropping dz** despite its 0.90 Spearman correlation with log_speedup. Two reasons: (1) psychological — assessors expect distance + elevation as primary uncertainty indicators, and (2) physical — turning and log_speedup are per-sector features that may saturate or lose signal at extreme dz beyond the training envelope, whereas dz continues to grow monotonically. dz is extrapolation insurance.
- **Do not propose adding new features** (dRIX, sample_shortfall, k_MM_deviation, gradient features, interactions, etc.). Many were tested during thesis development and rejected. Prior sessions have proposed several of these — the user's answer is stable: no.
- **Do not propose replacing individual features.** log_speedup as ratio, turning as freq-weighted |dθ|, etc. — all considered and settled.

The ONLY tunable choices left inside the model architecture are: (a) the roughness FORMULA (bake-off already done, A/Q/P are the finalists), (b) the parameterization (z-scored vs ÷std-only, i.e. production vs exp-uncentered), and (c) the target (signed e vs |e|).

## 3. Model at a glance (thesis-final / production)

Source: `Bayesian_approach/Final model/ws_uncertainty_model_pairlevel_final`
Results: `Bayesian_approach/Final model/Results/ws_uncertainty_pairlevel_results.json`
Full spec: `Thesis documentation/MODEL_CURRENT_STATE.md`

```
e_overall ~ StudentT(nu, mu, sigma)
mu     = beta_dz * dz_z                        # bias term (signed dz)
sigma  = exp( log_sigma0 + sum_k gamma_k * z_k )   # z_k = full z-score
```

5 sigma features (all z-scored before entering the model):
1. `dist_sat = 1 - exp(-d / d_A)`  — pair-level, saturating; d_A is complexity-adjusted "safe zone" distance
2. `turning_sat = 1 - exp(-t / 3.0)` — freq-weighted mean |d_turning_deg| across 12 sectors, saturated
3. `wm_abs_log_speedup` — freq-weighted mean |log(speedup_WTG / speedup_MM)| across sectors (NO saturation; log is already compressed)
4. `roughness_sat = 1 - exp(-r / 0.01)` — freq-weighted mean |(rs_WTG + rs_MM)/2| across sectors, saturated
5. `dz_sat = 1 - exp(-|dz|/40)` — pair-level, saturating

Fit at n=38 directional pairs (19 physical pairs). LOO Pearson 0.926 / Spearman 0.746.

## 4. The floor problem — RESOLVED (2026-08, promotion candidate: exp-M2c with n=47)

### The original problem

Self-prediction (dist=dz=turning=speedup=0, roughness=mast's own) under the thesis-final model didn't give a single "floor" — it ranged from 0.51% (flat sites) to 2.05% (rough forested sites). Two coupled defects:

1. **Terrain-dependent floor.** roughness_sat (formula A magnitude) doesn't vanish at same-site because it's a magnitude, not a mismatch. The other 4 features DO vanish at same-site.
2. **Floor/slope entanglement.** Under z-scored parameterization, log_sigma0 and every gamma share the centering shift, so adding low-uncertainty training pairs steepens slopes and inflates distant-pair predictions (e.g. WEA 06 W12 rose 11.5% → 13.4%).

### The final answer (promoted, 2026-08)

**Promoted model: exp-M2c, n=47 pairs, sigma-only (no bias), adaptive roughness with per-formula standardization.**

Production folder: `Post-thesis corrections/Adaptive_Model_v5/` (matches `TR_v5` calculator name)
Production script: `Post-thesis corrections/Adaptive_Model_v5/model_script.py`
Results:           `Post-thesis corrections/Adaptive_Model_v5/Results/*`

Full model spec:
```
|e_overall| ~ HalfStudentT(nu, sigma)                       # sigma-only, no bias term
log(sigma) = log_sigma0
           + gamma_dist    * z(dist_sat)          # centered z-score (production style)
           + gamma_turning * z(turning_sat)
           + gamma_speedup * z(wm_abs_log_speedup)
           + gamma_dz      * z(dz_sat)
           + gamma_rough   * min(z_A, z_B)        # ADAPTIVE roughness, per-formula z-score
                                                    z_A = (sat_A - mean_A) / std_A
                                                    z_B = (sat_B - mean_B) / std_B
                                                    sat_A = 1 - exp(-|(rs_W+rs_M)/2| / 0.01)  (formula A)
                                                    sat_B = 1 - exp(-|rs_W-rs_M| / 0.01)      (formula B)
```

Key metrics:
- LOO Pearson 0.886 (pair), 0.880 (site), Spearman 0.706
- Self-σ 0.44-0.45% single value across all 47 masts (**clean floor achieved**)
- LMG variance decomposition: **roughness 18.8%, complexity features 81.2%** (matches user preference)

Excluded masts (10, was 14): kept excluded — Sallachy, Kayislar, Herzhausen CFD, Taaibos, Ukhanda, Balver Wald. Re-INCLUDED — Hultema (2011WM011, 2014WM011), Malarberget (2012WM006), Slovenska East (2024PA107). Under exp-M2c's adaptive roughness, these previously-hard-to-fit sites are handled gracefully — LOO Pearson barely changes when they're added (0.887 → 0.886), and their inclusion is what pushes roughness LMG below 20%.

### The journey — how we got here (Days 1-10 of exploration)

Chronologically:

1. **Started with the observed floor spread (0.51%-2.05%)** and confirmed via `self_prediction_sigma.py` that it was terrain-driven, not a code bug.

2. **Additive-floor experiments** (`Additive floor model/` folder): tried `sigma = sigma_floor + Σ β·f_k`. Rejected — capped ceiling at ~9-12% while training data has max |e| of 18%. Physically the "lazy fix."

3. **exp-Q proposed**: keep multiplicative `exp()` but un-center features (÷std, no mean subtraction) AND multiply roughness by a 500m distance gate `(1 - exp(-d/500))`. This delivered a clean single-value floor (0.44%) but the gate was philosophically awkward — an arbitrary distance scale to defend.

4. **User pushback on the gate**: exp-A tested as the "exp minus the gate" — un-centered but keeps un-gated magnitude roughness. Result: terrain-dependent floor returned (0.33-1.30%), because roughness magnitude persists at same-site without the gate.

5. **Adaptive min-family invented** to combine formula A (magnitude) and formula B (mismatch). The min operation picks whichever is smaller per pair — at same-site B=0 so min=0 → clean floor without a distance gate. Four variants of standardization tested:
   - **exp-M** (min-first, un-centered)
   - **exp-Mc** (min-first, centered)
   - **exp-M2** (per-formula ÷std, then min, un-centered)
   - **exp-M2c** (per-formula z-score, then min, centered)

   All 4 tested via `sigma_only_refit/refit_sigma_only.py`, both forest-in and forest-out. Full LOO metrics saved to `sigma_only_refit/Results/`.

6. **exp-B and exp-T tested as controls** (pure mismatch; magnitude-mismatch |rs_W|-|rs_M|). Both lost fit badly (~0.87 Pearson). Confirms magnitude signal is real, not double-counting log_speedup.

7. **Bias term dropped** across all sigma-only refits per supervisor preference. Target changed from signed `e_overall` to `|e_overall|`; likelihood → HalfStudentT. Sigma retains "scale of underlying two-sided distribution" interpretation, so P75/P90 multipliers stay valid.

8. **Sign-cancellation analysis**: 17.5% of sectors have opposite-sign rs. Concern raised: formula A cancels toward 0, min picks A → could under-predict on high-mismatch pairs. Herzhausen decomposition (`herzhausen_decomposition.py`) confirmed the fear is theoretical — Herzhausen predictions are dominated by turning (30%) and complexity features, not roughness. Both exp-A and exp-M give near-identical predictions there.

9. **The variance-decomposition methodology fix** (major realization): the naive `beta_k²` decomposition is misleading when features are correlated. Full-covariance / LMG / Shapley decomposition is the proper attribution. See `variance_decomposition_explained.md` for detailed derivation. Under proper LMG:
   - Distance's naive 27% share collapsed to 15% (was inflated by the naive-sum denominator)
   - The turning/speedup/dz cluster genuinely shares variance via 0.6-0.9 pairwise correlations

10. **Comprehensive LMG comparison** (`lmg_all_variants_comparison.py`) across all candidates. Only **exp-M2c with forest-in** delivered roughness LMG < 20% while keeping LOO Pearson above 0.87. Forest-out variants all had roughness LMG ≥ 24%. The forest pairs are what activate the min-picking-B behavior that shrinks roughness dominance.

11. **Final expansion to n=47**: Slovenska East (2024PA107) re-included along with the 2 forest projects. Pushed roughness LMG down to 18.8%, complexity LMG up to 81.2%. LOO Pearson stayed at 0.886 (no fit cost). This is the shipping configuration.

### Full options table

| Option | LOO Pearson | Self-σ | Roughness LMG | Verdict |
|---|---|---|---|---|
| Additive floor (saturated) | ~0.88 | 0.29% | — | Rejected — ceiling capped at ~9% |
| Additive floor (un-saturated) | 0.897 | 0.29% | — | Rejected — "lazy fix" |
| exp-Q (un-centered + 500m gate) | 0.916 forest-out | 0.44% | 30.6% | Rejected — gate awkward, roughness dominant |
| exp-A (un-centered, formula A) | 0.919 forest-out | 0.33-1.30% | 24.9% | Rejected — terrain-dep floor |
| exp-B (pure mismatch) | 0.870 | 0.53% | — | Rejected — loses magnitude signal |
| exp-T (magnitude mismatch) | 0.869 | 0.54% | — | Rejected — same as B, sign-independence didn't help |
| exp-M (min-first, un-centered) | 0.901 | 0.47% | — | Runner-up but weaker fit than exp-Mc |
| exp-Mc (min-first, centered) | 0.906 | 0.44% | — | Runner-up — best Spearman but mixed-distribution std concern |
| exp-M2 (per-formula ÷std, uncentered) | 0.875 | 0.50% | — | Superseded by exp-M2c |
| **exp-M2c forest-out (n=41)** | 0.893 | 0.44% | 28.8% | Interim — doesn't reduce roughness dominance |
| **exp-M2c forest-in (n=45)** | 0.887 | 0.46% | 19.3% | Meets preferences |
| **exp-M2c n=47 (+ Slovenska East)** | **0.886** | **0.45%** | **18.8%** | **PROMOTED — best on all criteria** |
| Informative log_sigma0 prior | not tested | — | — | Not pursued (exp-M2c solved the problem) |
| Do nothing | 0.926 with bias | 0.51-2.05% | — | Baseline reference (thesis-final) |

Full test scripts in `Post-thesis corrections/` and its `Additive floor model/`, `sigma_only_refit/` subfolders.

### Diagnostic documents (defense reference)

Everything below is inside `Post-thesis corrections/Adaptive_Model_v5/`:
- `variance_decomposition_explained.md` — the math of LMG vs naive attribution
- `adaptive_variants_explained.md` — walkthrough of M/Mc/M2/M2c with concrete numeric examples
- `Results/model_results.json` — the promoted model's fitted coefficients, LMG decomposition, LOO metrics, self-prediction summary
- `Results/self_prediction_per_mast.csv` — per-mast same-site sigma (42 rows, essentially single-value floor 0.44-0.45%)
- `Results/feature_contributions_per_pair.csv` — per-pair log(σ) decomposition by feature

## 5. Feature framework insights (subtle but important)

Cross-feature Spearman correlations (see `Thesis plots/methodology/feature_correlation.pdf`):
- log_speedup vs elevation_diff: **0.90** (near-redundant statistically)
- turning vs elevation_diff: **0.68**
- turning vs log_speedup: 0.53
- roughness vs distance: -0.39 (moderate)
- roughness vs speedup: 0.40
- everything else: < 0.30

**But do NOT collapse turning/speedup into elevation.** The user has flagged this explicitly: turning and speedup are how WAsP does its per-sector work — they operate at the sector level (12 values per pair) capturing directional flow behavior. dz is a bulk pair-level scalar. Statistical correlation is a data-artifact of how the training set was built (flat = far apart with low dz; complex = close together with high dz), not evidence that the features carry the same information physically. Keeping them separate is defensible; collapsing them loses the sector-level physics.

The correlation IS a real modelling issue — the variance decomposition (34.6% turning + 33.9% roughness + 14.6% distance) is inflated by shared signal. But the fix isn't feature collapse; it's better parameterization (exp-Q, or informative priors, or maybe orthogonalization).

## 6. Physics: what each feature captures

| Feature | Level | Dissimilarity or magnitude? | Vanishes at same-site? |
|---|---|---|---|
| distance | pair | geometry (not either) | YES (d=0) |
| turning | sector-aggregated | mismatch (Δθ between WTG and MM per sector) | YES |
| log_speedup | sector-aggregated | mismatch (log ratio) | YES |
| **roughness** | sector-aggregated | **magnitude** (mean of `|rs|` across sites) | **NO** |
| dz | pair | mismatch (|elevation diff|) | YES |

The odd one out is roughness. Physical reason it's magnitude, not mismatch: `rs = rough_speedup_frac` is already a WAsP-computed CORRECTION applied at each site. Bigger corrections carry more model error (multiplicative error scales with correction size).

**Sign structure of roughness (verified via `Post-thesis corrections/roughness_sign_analysis.py`, results in `rough work/pending issues - KT/new 1.txt`):**
- Positive rs (acceleration): 41.4% of sectors
- Negative rs (deceleration): 32.7%
- Zero: 25.9%
- **Opposite-sign between WTG and MM: 21.1% of sectors, affecting 24/38 pairs**

So sign cancellation in the `|(rs_W+rs_M)/2|` formula is real — NOT rare. Do not assert "sign is almost always same" without checking.

**Empirical: despite 21% sign cancellation, formula A (magnitude of signed mean) still beats F (mean of absolutes) at 0.9278 vs 0.9188 Pearson.** So the cancellation is capturing real physics — when WTG and MM have opposite-sign corrections, they partially cancel in the extrapolation ratio, and formula A represents this correctly while F double-counts them. Don't propose replacing A with F on "no sign cancellation" grounds.

Physical argument for gating roughness by distance (the "Q" form): at same-site, both W and MM sit on the same terrain, so both get identical roughness corrections. Any error in that correction is systematic — it applies equally to both, and cancels out in the extrapolation (which is a ratio). Only when the two sites are far enough apart that their upwind footprints stop overlapping do the correction errors become independent contributors to uncertainty. 500m is roughly the atmospheric boundary layer footprint scale. So the gate is physics, not a math trick.

**Bake-off Pearson on corrected data (from `Post-thesis corrections/Results/roughness_formula_comparison.csv`):**
A (current, magnitude) 0.9357 · S (gated 700m) 0.9355 · R (gated 600m) 0.9353 · **Q (gated 500m) 0.9340** · P (gated 300m) 0.9337 · N (dist_sat gated) 0.888 · F (mean of abs) 0.9188 · B (mismatch) 0.861. **A and Q are within noise on fit; Q's advantage is the clean floor, not the fit.**

## 7. Key files and folders

```
Windflowmodelling/
├── CLAUDE.md                                          # this file
├── Bayesian_approach/
│   └── Final model/
│       ├── ws_uncertainty_model_pairlevel_final       # production model script (Python, no .py ext)
│       └── Results/
│           ├── ws_uncertainty_pairlevel_results.json  # fitted coefficients, scalers, LOO
│           └── ws_uncertainty_pairlevel_idata.nc      # full MCMC trace
├── Post-thesis corrections/
│   ├── roughness_formula_comparison.py                # bake-off framework for roughness/distance variants
│   ├── ws_uncertainty_model_roughness_fix             # v4 gated-roughness model (post-thesis)
│   ├── self_prediction_sigma.py                       # per-mast self-prediction floor table
│   ├── Additive floor model/
│   │   ├── additive_floor_roughness_test.py           # early rework attempt (linear floor)
│   │   └── expfloor_uncentered_roughness_test.py      # exp-Q family — the winning candidate
│   └── Results/
│       ├── roughness_formula_comparison.csv           # A/B/E/Q/etc bake-off
│       ├── distance_formula_comparison.csv            # distance variants bake-off
│       ├── project_ablation.csv                       # leave-one-project-out results
│       └── self_prediction_sigma.csv                  # per-mast floor list
├── Uncertainty calculator/
│   ├── Uncertainty_calculator_TR_v3.0/                # dropdown: conservative vs low-floor
│   ├── Uncertainty_calculator_TR_v4.0/                # gated-roughness model
│   └── Uncertainty_calculator_TR_FinalModel/          # exact thesis-final JSON, single model
├── Thesis documentation/
│   ├── MODEL_CURRENT_STATE.md                         # authoritative model spec (READ THIS)
│   ├── Chapter_2_Methodology.md                       # WAsP context and feature definitions
│   ├── THESIS_REFERENCE.md                            # cross-reference
│   └── Model_Development_Log.md                       # history of design decisions
├── Thesis plots/
│   └── methodology/feature_correlation.pdf            # feature correlation matrix
└── Input data/
    └── Focused_modelling_inputs.xlsx                  # sector-level source data
```

## 8. Environment and workflow rules

**Python:** PyMC scripts must run in the `pymc-env` conda env, NOT the project `.venv`:
`C:\Users\K_Trivedi\AppData\Local\anaconda3\envs\pymc-env\python.exe`
The `.venv` has no g++, so PyTensor falls back to pure Python and each MCMC fold takes ~90 min instead of a few.

**Who runs what:**
- Pure numpy/pandas scripts (no PyMC): Claude may run.
- PyMC / MCMC scripts: **user runs them, Claude does not**. Claude has hit env issues before; user prefers to run and paste results back.

**PyMC + Windows:** Multi-process sampling breaks on Windows. Always use `cores=1, progressbar=False` in `pm.sample(...)`.

## 9. Communication conventions

- **Math in ASCII, not LaTeX.** User's terminal doesn't render `$…$`. Use plain text: `sigma = exp(log_sigma0 + sum_k [beta_k * f_k / std_k])`. Greek letters spelled out, `_` for subscripts, `^` or `exp()` for powers.
- **Do not run PyMC scripts yourself.** Write them, hand off to user, wait for results.
- **File references use markdown links** in the IDE: `[Trial.py:42](path/to/Trial.py#L42)`, not backticks.
- **The user's physics intuition often overrides purely-statistical analysis.** Statistical correlation ≠ physical equivalence. Ask before "collapsing" features that correlate; they may serve distinct roles in WAsP's process model. Recent example: turning and speedup vs elevation are 0.68 / 0.90 correlated, but ARE NOT interchangeable because turning and speedup are per-sector and dz is pair-level scalar.

## 10. Excluded masts (14, hardcoded in the model)

Documented in the production model file's `EXCLUDED_MASTS` list and re-stated in the results JSON. Reasons in the source. Notable ones: Sallachy (complex terrain), Kayislar (only 2 energy-dominant directions), Herzhausen (CFD, not comparable to WAsP), Hultema and Malarberget (persistent outliers). Do not silently un-exclude them — the training set is what it is; if you want a "with-those-masts" fit, do it explicitly and report both.

## 11. Where to look before assuming anything

- Model spec (any question about "what does the model do"): `Thesis documentation/MODEL_CURRENT_STATE.md`
- Feature physics: `Thesis documentation/Chapter_2_Methodology.md`
- Design history: `Thesis documentation/Model_Development_Log.md`
- Fitted numbers: `Bayesian_approach/Final model/Results/ws_uncertainty_pairlevel_results.json`
- Bake-off results: `Post-thesis corrections/Results/*.csv`

If a memory note in `~/.claude/projects/.../memory/` contradicts something in this file or in the source code, the code and this CLAUDE.md are authoritative. Memory notes are point-in-time observations.

## 12. Prior-session context (before assuming anything is "new")

Discussion notes from earlier sessions are in `rough work/pending issues - KT/`. Key established facts as of 2026-07:

- **Data was corrected mid-project.** `Focused_modelling_inputs.xlsx` was regenerated with corrected roughness numbers. Any fitted JSON older than that correction is on wrong data. The shipping calculator (v3) currently bundles a JSON that was fit BEFORE the correction — a known stale-artifact issue.
- **Adding 2 close pairs (38 → 40) already tested.** Floor dropped 0.51% → 0.39%. But γ_dist jumped 0.187 → 0.283 and γ_rough 0.285 → 0.352 — the exact floor/slope entanglement bug, quantified.
- **Distance is closed.** LOO bake-off of 6 variants; current `1 - exp(-d/d_A)` wins. Don't re-test.
- **Extrapolation should NOT be clipped.** Prior claude session initially proposed z-clipping to stay within training range; user rejected, session conceded. The tool is meant to extrapolate monotonically.
- **Plug-in vs posterior-predictive: keep plug-in.** The borderline band where they differ (~z 6-10) is in a range where sites would be rejected on other grounds anyway.
- **"Overall z" column** already delivered in calculator v3.
- **The additive-floor reparameterization has been named as an option but not adopted.** Prior view: only worth doing "if a stakeholder explicitly objects to non-zero at zero distance." exp-Q (which we tested more recently) is close to but not exactly the additive form — it's the un-centered exp with gated roughness.
- **Bias term is being dropped** (see §2A). Prior meeting 06.07 raised the bias/uncertainty separation concern; user has since decided to remove bias entirely rather than push the concept with a skeptical supervisor. All new fits are sigma-only, StudentT(nu, 0, sigma).

Open items still unresolved (as of 2026-08):
- ~~Canonical shipping model~~ **RESOLVED**: exp-M2c with n=47 (see §4). Fitted on corrected data, results at `Post-thesis corrections/Adaptive_Model_v5/Results/`. Deployment calculator at `Uncertainty calculator/Uncertainty_calculator_TR_v5/`.
- ~~exp-Q coefficients not yet written to a promotable JSON~~ **SUPERSEDED**: exp-Q was ruled out in favor of exp-M2c (adaptive avoids the 500m gate defense issue and delivers lower roughness LMG share).
- ~~Whichever model wins, it needs re-fit on the CORRECTED input data~~ **DONE**: exp-M2c fitted on corrected data.
- ~~Calculator integration~~ **DONE**: `Uncertainty calculator/Uncertainty_calculator_TR_v5/` loads the Adaptive Model v5 JSON, computes the per-formula standardization at inference, and adds a self-prediction column. Build the exe via that folder's `build.bat`.
- ~~Thesis defense document~~ **DONE**: `Adaptive_Model_v5/THESIS_DEFENSE_ADDENDUM.md` + `JOURNEY_TO_PRODUCTION_MODEL.md` cover the promoted model and the story of alternatives.

## 13. Section-4 quick reference for future sessions

If someone asks "which model do we ship?": **Adaptive Model v5** (formerly "exp-M2c") — 47 pairs, sigma-only, adaptive roughness via `min(z_A, z_B)` with per-formula centered standardization. Script: `Post-thesis corrections/Adaptive_Model_v5/model_script.py`. Results: `Adaptive_Model_v5/Results/model_results.json`. Deployment calculator: `Uncertainty calculator/Uncertainty_calculator_TR_v5/`.

If someone asks "why not the thesis-final model?": four reasons — (1) supervisor was skeptical of the bias term, so it's gone; (2) thesis-final's terrain-dependent floor (0.51-2.05%) is uncomfortable to defend for repowering scenarios; (3) exp-M2c's LMG-verified 18.8% roughness share matches user's stated physics preference (complexity should dominate over roughness magnitude); (4) exp-M2c uses more data (47 pairs vs 38).

If someone asks "why isn't LOO Pearson higher?": it went from 0.926 (thesis-final, with bias) to 0.886 (exp-M2c). The drop is real. What we bought for that: bias-free, clean single-value floor, LMG-defensible feature attribution, forest inclusion, per-formula mathematical rigor. Trade documented; accepted.

If someone proposes yet another model variant: the design space is exhausted. 7 roughness formulas × 4 standardization strategies × 2 forest configurations tested. exp-M2c dominates on every stated preference criterion. Any further work should target the calculator integration, not the model itself.
