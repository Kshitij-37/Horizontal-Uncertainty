# Adaptive Model v5 — the promoted post-thesis model

This folder contains the final production model that supersedes the thesis-final
regression. It is the reference from which the `TR_v5` deployment calculator is
built.

Internal working name during development was "exp-M2c" (see the story doc for
what each letter meant); everything here uses the cleaner "Adaptive Model v5"
naming.

---

## What's in this folder

### Documentation (read these first)

| file | what it covers |
|---|---|
| `THESIS_DEFENSE_ADDENDUM.md` | Formal defense-quality summary of the model, changes from thesis-final, validation, limitations. ~3 pages. |
| `JOURNEY_TO_PRODUCTION_MODEL.md` | Narrative of the ~1 week of exploration that produced this model — what we tested, rejected, and why. |
| `variance_decomposition_explained.md` | The math of the LMG (Shapley) variance attribution used to report feature importance. Explains why naive `beta²` percentages are misleading and how to properly account for feature correlations. |
| `adaptive_variants_explained.md` | Walkthrough of the four adaptive-min formulations tested (M, Mc, M2, M2c). Concrete numeric examples of what `min(A, B)` does. |

### Code (rerun to regenerate results)

| file | what it does |
|---|---|
| `model_script.py` | The production model. Fits the promoted Bayesian regression on 47 pairs of corrected data, runs pair-level and site-level LOO CV, computes per-mast self-prediction and per-pair feature contributions. **This is the one to run.** |
| `variance_decomposition_analysis.py` | Standalone LMG/Shapley variance-decomposition analysis. Runs against the already-fitted JSON and per-pair CSV to produce naive + proper decompositions. Pure numpy, no PyMC. |
| `comparison_vs_thesis_final.py` | LMG comparison between the promoted model and the thesis-final model on the same n=41 forest-out subset. Historical / defense reference. |

### Fit outputs (`Results/`)

All regenerated each time `model_script.py` runs.

| file | contents |
|---|---|
| `model_results.json` | Fitted coefficients, priors, LOO metrics, naive AND LMG variance decomposition, self-prediction summary, excluded masts. **This is what `TR_v5` calculator loads.** |
| `mcmc_trace.nc` | Full ArviZ InferenceData (posterior samples for all parameters). Load with `arviz.from_netcdf` to inspect posteriors. |
| `loo_pair_predictions.csv` | 47 rows, per-pair predicted sigma vs actual (pair-level holdout) |
| `loo_site_predictions.csv` | 47 rows, per-pair predicted sigma vs actual (site-level holdout — all pairs from a location held out together) |
| `loo_pair_scatter.html` | Interactive plotly scatter of the pair-level LOO |
| `loo_site_scatter.html` | Interactive plotly scatter of the site-level LOO |
| `self_prediction_per_mast.csv` | 42 rows — for each unique mast, the model's predicted sigma at the mast's own site. Should be ~0.44-0.45% (clean floor). |
| `feature_contributions_per_pair.csv` | 47 rows — per-pair breakdown of `log(sigma)` into individual feature contributions. Useful for "why does the model predict X for pair Y?" questions. |

---

## Quick reference — the model in one paragraph

Sigma-only Bayesian regression on 47 directional pairs (42 unique masts).
Target: `|e_overall|`. Likelihood: `HalfStudentT(nu, sigma)`. Five features (all
pair-level): saturating distance, freq-weighted saturating turning, freq-weighted
|log speedup|, saturating |dz|, and an adaptive roughness feature that takes
`min(z_A, z_B)` where `z_A` is the per-formula z-score of magnitude
`|(rs_W+rs_M)/2|` and `z_B` is the per-formula z-score of mismatch
`|rs_W-rs_M|`. LOO Pearson 0.886 (vs thesis-final's 0.926 with bias). Same-site
prediction is essentially a single value (0.44-0.45% across all 42 training
masts) — the "clean floor" property that motivated the whole rework.

For more, read `THESIS_DEFENSE_ADDENDUM.md`.

---

## How to run the model

Uses `pymc-env` conda Python (per project convention):

```
C:\Users\K_Trivedi\AppData\Local\anaconda3\envs\pymc-env\python.exe model_script.py
```

Runtime: ~1.5–3 hours (dominated by 47-fold pair-level + 22-fold site-level LOO).
Faster smoke-test: edit the `FULL_FIT_*` / `LOO_*` constants at the top of
`model_script.py` down to 500 draws × 2 chains — cuts runtime to ~30-45 min.

After the run, `variance_decomposition_analysis.py` can be run standalone
(pandas + numpy only) to get the LMG decomposition without re-fitting.

---

## Downstream consumers

- **`Uncertainty calculator/Uncertainty_calculator_TR_v5/`** — deployment GUI
  that loads `Results/model_results.json` and predicts uncertainty for
  user-pasted windPRO exports. Also shows the per-WTG self-prediction column.
  Build the exe via that folder's `build.bat`.

- **CLAUDE.md** (project root) — describes this model as the shipping model.
  Should be consulted first by any new session.

---

## Excluded masts (10, hardcoded in `model_script.py`)

- Sallachy: 2015WM018, 2021PA004, 2022PA008 (hills and valleys, WAsP limits)
- Kayislar: 2022PA018 (only 2 dominant directions)
- Herzhausen CFD: 2019HE001, 2019HE002, 2019HE003 (CFD, not comparable to WAsP)
- Taaibos: 2022PA021 (3-mast, dropped for fit)
- Ukhanda: 2023PA085 (2 mast + LiDAR)
- Balver Wald: 2024PA014 (uncertain site)

Reincluded (were excluded in thesis-final): Hultema (2011WM011, 2014WM011),
Malarberget (2012WM006), Slovenska East (2024PA107). Under the adaptive
roughness formulation these sites fit well enough to include.
