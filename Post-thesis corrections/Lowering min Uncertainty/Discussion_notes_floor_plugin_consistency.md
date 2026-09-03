# WS Horizontal Uncertainty Model — Discussion Notes

*Captured 2026-06-25. Format: each topic gives **Your side** (Kshitij) and **My side**
(analysis), then **Where we landed**. Numbers are from the actual model JSONs unless noted.*

---

## Model recap (so the notes stand alone)

Pair-level Bayesian model. Each observation = one directional pair (A→B). Target = the
frequency-weighted signed overall wind-speed error.

```
e ~ StudentT(nu, mu, sigma)        nu ≈ 14

log(sigma) = log_sigma0
           + gamma_dist      · z(dist_sat)
           + gamma_turning   · z(turning_sat)
           + gamma_speedup   · z(wm_abs_log_speedup)
           + gamma_roughness · z(roughness_sat)
           + gamma_dz        · z(dz_sat)

mu = beta_dz · z(dz)               (signed height-difference bias)
```

**Key structural facts that drive everything below:**
- The five sigma features are **z-scored** (standardised: `z = (raw − mean)/std`).
- All five features are **saturating and non-negative** (raw ≥ 0).
- All five gammas are **HalfNormal → strictly positive**.
- This is a **horizontal / spatial** extrapolation uncertainty model.

---

## 1. The floor issue (minimum predictable uncertainty)

**Context:** What is the smallest uncertainty the tool can output, and what does it
report when predictor and prediction site are identical (the repowering case)?

**Your side:**
- The least uncertainty the model predicts is **~0.51%**, occurring "when all features
  are 0".
- Worry: at the *same* location (repowering — new turbine on the old turbine's spot),
  the tool should arguably read near-zero, but it can't go below 0.51%.
- Hypothesis: the model never *saw* uncertainty below 0.51% in training, so maybe adding
  low-uncertainty cases would fix it.
- Later action you took: **you added 2 real close-together-mast pairs** (38 → 40 pairs)
  to push the floor down.

**My side:**
- Your premise was half-right. Because features are **z-scored**, "all features = 0"
  *in the equation* means **all features at their training mean**, which gives
  **sigma0 ≈ 2.97–3.00%**, not 0.51%.
- The **0.51% is the same-location case** — all *raw* features = 0, which maps to
  **negative** z-scores. Since gammas are positive and z is negative, sigma drops *below*
  sigma0. **So 0.51% is already exactly the repowering prediction — you weren't hitting a
  training wall, you were hitting the architectural minimum.**
- It is a true floor: features are non-negative and gammas positive, so **raw = 0 is the
  global minimum** of log(sigma). Nothing physical goes lower.
- It is **0.51% not 0%** because `log_sigma0` is a global intercept and `exp(intercept + …)`
  can never reach 0. **That floor is the model's estimate of irreducible, non-horizontal
  uncertainty** (measurement, temporal/interannual, vertical extrapolation). For repowering
  the horizontal component → 0, but these others don't, so a ~0.5% residual is defensible.
  **Do not claim 0%.**
- **Pushback on synthetic self-pairs:** do NOT manufacture same-location training rows.
  A self-pair has error = 0 *by construction* (a tautology / data leak), it corrupts the
  z-scaling (shifts every mean/std), it breaks the Student-t likelihood (pulls sigma→0,
  divergences), and it ignores that in real repowering the **hub height differs → dz ≠ 0**.
- **On the 2 close pairs you actually added (the honest, correct move):**

  |                                   | OLD (38 pairs, bundled) | NEW (40 pairs, re-fit) |
  |-----------------------------------|-------------------------|------------------------|
  | sigma0 (z=0, *average* pair)      | 2.98%                   | 2.65%                  |
  | **Floor (same-location, raw=0)**  | **0.512%**              | **0.385%**             |
  | dist_sat mean                     | 0.552                   | 0.524                  |
  | **z of distance at same-location**| **−3.72 SD**            | **−2.78 SD**           |

  - The floor dropped 0.51% → 0.39% — **but the bigger win is −3.72 → −2.78**: the
    same-location case is now only 2.8 SD outside your data instead of 3.7, so you are
    **extrapolating less far** → the prediction is more *trustworthy*, not just lower.
  - **sigma0 is a misleading diagnostic — don't track it.** It fell partly just because you
    added *similar* pairs, which mechanically lowers the average-pair value. Watch the
    **floor (0.39%)**, not sigma0.
  - **Architectural limit:** you can never reach ~0 by adding data, because the floor is
    `exp(intercept_at_raw0) > 0`. More close pairs is right for *trust*, wrong for *zero*.

**Where we landed:**
- 0.39% vs 0.51% vs 0% **changes no decision** — any same-location pair is already
  "negligible horizontal uncertainty, use it." The floor is a **philosophical** issue, not
  a practical one, *unless a stakeholder explicitly objects to a non-zero number at zero
  distance*.
- If you genuinely want same-location → ~0, that is the **additive-floor reparameterization**
  (see §5), not more data.

---

## 2. Plug-in vs posterior-predictive sigma

**Context:** Far outside the training data (large z), how confident is the single sigma
number, and should we report it differently?

**Your side:**
- The tool's output feeds directly onto the **P50**, then gets multiplied to **P75 / P90 /
  P99** (≈ 0.67·σ, 1.28·σ, 2.33·σ).
- Therefore the consumer needs **one scalar**. Nobody downstream cares about a confidence
  interval on the uncertainty — a band doesn't slot into the P-value machinery.
- Practical threshold point: **a site at ~20% uncertainty won't be used anyway**, so at the
  extreme tail "20% vs 30%" doesn't change the decision.

**My side:**
- Agreed: ship a single number; credible intervals are useless here.
- But you can bake *honesty* into that single number. Today the calculator uses the
  **plug-in** estimate `sigma = exp(mean(gamma)·z)`. The statistically correct single number
  is the **posterior-predictive** mean `E[exp(gamma·z)]`, which by Jensen's inequality is
  **larger**, with the gap growing as you extrapolate:
  `correction ≈ exp(0.5 · s² · z²)` (s = posterior SD of the gamma).
- **Numbers at z = +10 on the speedup feature (current 40-pair fit):**
  **plug-in ≈ 10.1%, posterior-predictive ≈ 17.6%.**

  | z (speedup) | plug-in | posterior-predictive |
  |-------------|---------|----------------------|
  | +3          | 4.4%    | 4.6%                 |
  | +6          | 6.5%    | 7.9%                 |
  | +8          | 8.4%    | 11.8%                |
  | **+10**     | **10.1%** | **17.6%**          |
  | +15         | 21%     | 68%                  |

- The posterior-predictive number **self-inflates at the tail** — exactly "be more
  conservative when extrapolating" — while staying a single scalar the consumer never has
  to interpret. Since this drives P90/P99 (the *downside* numbers), the conservative bias is
  in the safe direction.
- **But** the two methods only differ in a **decision-relevant** way inside a *narrow
  borderline band*. Below it both say "fine" (correction ≈ 1.0). Above it both say "reject"
  — and there **your point is exactly right: 25% vs 40% changes nothing.**

**Where we landed:**
- **Decision rule:** it only matters whether your accept/reject threshold sits inside the
  borderline band (~z 6–10 for speedup).
- If a spatial uncertainty of ~6–8% *alone* already disqualifies a site (likely, given
  total wind-resource budgets ~10–15% all-in), then **keep the simpler plug-in** — the
  divergence lives in a band you'd never accept anyway.
- Posterior-predictive is the *more correct* number but **buys nothing operationally**
  unless the threshold lands in that window. **Leaning: keep plug-in.**

---

## 3. Other problems we discussed (easy to miss)

### 3a. Maximum predictable uncertainty / unbounded speedup feature

**My side (raised it):**
- By the same logic as the floor, max sigma is at the feature *maxima* (gammas positive).
- **4 of 5 features saturate at 1** → bounded ceiling. But **`wm_abs_log_speedup` does NOT
  saturate** — it's unbounded → strictly, **the model has no maximum** (sigma → ∞).
- Empirical max on *real* pairs: **11.6%** (pair holdout) / **14.9%** (site holdout), on
  `2020PA011__2019PA023`. Theoretical cap with the 4 saturating features pinned at 1 ≈ **65%**,
  and speedup can push past 100%.

**Your side:**
- The speedup feature is **self-bounding in practice**: each speedup ≈ 0.5–1.5 around 1, so
  per-sector `|log(WTG/MM)|` maxes ~1.1, and the **frequency-weighted mean across sectors**
  is far smaller. The "infinite" max is theoretical and won't blow up on real data.

**Where we landed:**
- You're right — don't worry about the infinity. **Caveat to remember:** the speedup
  feature's training **std is tiny (≈ 0.043)**, so even a plausible raw value (0.2) maps to
  **z ≈ +3.8** and a ×1.6 multiplier. It won't explode, but high-speedup pairs *are*
  extrapolation beyond support.

### 3b. Should the tool extrapolate at all? (the z-clipping debate)

**My side (initial, later withdrawn):**
- Suggested **clipping z** to the training range (cap ~z+3) so a tiny-std feature can't ride
  an outlier out to a ×4 multiplier.

**Your side:**
- **Rejected it.** The tool *must* extrapolate — training data is always limited, and the
  five features **are dissimilarity measures**, so a +10 SD pair is genuinely more dissimilar
  and **should** read higher. Clamping destroys that monotonicity. Practicality of the tool
  > perfectness of the method.

**Where we landed:**
- **I conceded — clamping is wrong for this tool.** Your current uncapped model *already*
  extrapolates monotonically, which is the desired behaviour. The only real choice is the
  **shape** of extrapolation (exponential, as now, vs saturating), **not whether**.
  Exponential errs conservative, which suits an uncertainty tool. (This is what motivated the
  posterior-predictive idea in §2 — a softer way to handle the tail than clipping.)

### 3c. New feature delivered: "Overall z" column (calculator v3.0)

- Added a single **Overall z** column to the results table.
- Definition chosen (yours): **effect-weighted average** `overall_z = (Σ gammaₖ·zₖ)/(Σ gammaₖ)`.
- Clean identity: `log(sigma) = log_sigma0 + (Σ gamma)·overall_z`. So **overall_z = 0 ⇔
  baseline sigma0**; it is **scale-preserving** (all features at +2 → +2) and **monotone with
  the reported uncertainty**, so it *explains* the sigma column.
- Trade-off: a single extreme feature is **diluted** (speedup at z+10 alone shows ≈ +1.1) —
  acceptable because the per-feature columns still expose the extreme. No separate max|z| flag
  (you declined).
- **Verified:** mean site → 0.000; speedup +1 SD → γ_spd/Σγ (exact); all features +2 SD →
  +2.000 (exact).

---

## 4. Consistency issues (important — resolve before trusting any number)

**The model JSON is not single-sourced. Three files exist with different contents:**

| JSON file | log_sigma0 | sigma0 | gammas (dist/turn/spd/rough/dz) | n_pairs | Role |
|-----------|-----------|--------|----------------------------------|---------|------|
| `Bayesian_approach/Final model/Results/…` | −3.514 | 3.00% | 0.187 / 0.287 / 0.130 / 0.285 / 0.088 | 38 | **BUNDLED in the .exe** |
| `Bayesian_approach/Final model/Results_roughsat/…` | −3.514 | 3.00% | identical to above | 38 | **byte-identical** to Results |
| `Post-thesis corrections/Results/…` | −3.632 | 2.67% | 0.283 / 0.313 / 0.134 / 0.352 / 0.098 | 40 | the **re-fit with your 2 close pairs** |

**What the calculator actually uses:**
- The **`.spec` and `build.bat` both bundle** `Final model/Results/…` into the frozen `.exe`
  (PyInstaller `--add-data`). So the shipped tool runs the **38-pair model** (−3.514).
- **The 40-pair re-fit (Post-thesis) is NOT in the shipped tool.**
- The Python **source fallback path is broken**: `MODEL_JSON`'s fallback resolves to
  `Bayesian_approach/Post-thesis corrections/Results/…`, **which does not exist**. Running the
  `.py` directly raises `FileNotFoundError`; it only works frozen because the JSON is bundled.

**Mid-session confusion this caused (for the record):**
- My *early* analysis (the 0.51% floor, the z-curves, plug-in vs post-pred) read the
  Post-thesis JSON when it still held the **original** values (−3.517 ≈ the bundled −3.514).
  Mid-session that file was **overwritten by your re-fit** (−3.632), which briefly looked like
  the file "changing on its own." **Net: my early numbers are valid for the shipped 38-pair
  model.** The 40-pair numbers are the newer, separate fit.

**The γ-leverage caveat:** the 2 added pairs moved gammas noticeably (`γ_dist` 0.187 → 0.283,
`γ_rough` 0.285 → 0.352), which **raises** predicted uncertainty for distant / rough pairs.
Two points swinging slopes on n = 40 is **high leverage** — though possibly within posterior
SD. **Check posterior SD + LOO on the 40-pair fit before adopting it.**

**Decision you need to make (this is a "pick + rebuild", NOT a refit):**
1. **If the 40-pair (close-pairs / roughness-fix) model is canonical:** re-point all three
   (`.py` fallback, `.spec`, `build.bat`) to `Post-thesis corrections/Results/…` and
   **rebuild the .exe**. No resampling needed — that model is already fit.
2. **If the 38-pair bundled model is canonical:** just **fix the broken `.py` fallback path**
   so source runs match the exe.
- Either way, **all three references must point at one file.** "The JSONs are close" is **not**
  a reason to refit — a refit (re-running PyMC) is only warranted when the *inputs* change.

---

## 5. Point 2 revisited — is the extrapolation tail even reachable? (this session)

**My side:** translated "+N SD of the speedup feature" back into physical terms.
The feature is `wm_abs_log_speedup` = frequency-weighted mean `|log(speedup_WTG/speedup_MM)|`.

| z (speedup) | raw \|log ratio\| | implied speedup ratio |
|---|---|---|
| +3 | 0.163 | 1.18× |
| +10 | 0.464 | 1.59× |
| **+15** | **0.679** | **1.97×** |

A +15 SD pair means one site's frequency-weighted speedup is **~2× the other's** — physically
implausible. **Your observed maximum across 40 real pairs is +3.3 SD (ratio ≈ 1.19×).**

**Where we landed:** the "+15 SD ⇒ 68% vs 21%?" question is **moot — such a pair can't occur.**
The band that actually gets exercised is z ≈ 0–3.3, where plug-in and posterior-predictive barely
differ. This retroactively settles §2: **keep the plug-in.** The posterior-predictive refinement
only mattered in a region you never reach.

---

## 6. Roughness feature & measurement height (investigation, this session)

**Your side:**
- In repowering you extrapolate from *existing* turbines (shorter: 80–120 m) to *planned* ones
  (taller: 150–170 m). To compare like-for-like you extract speedup/turning at the **existing
  (lower) turbine height**. Consequence: an 80→80 m case reads **higher** uncertainty than a
  150→150 m case.
- Your justification: "lower height sits in a rougher flow regime → predictions are poorer →
  uncertainty should be higher." Plausible, but you wanted to know if the data backs it.

**My side (confound analysis on the 40 pairs):**
- The feature is **formula A** = `|(rough_speedup_WTG + rough_speedup_MM)/2|` — the *average
  magnitude* of WAsP roughness-induced speedup across the pair (a terrain-roughness / flow-complexity
  measure, **not** a site-to-site mismatch).
- Correlation with actual |error|: turning 0.84, speedup 0.82, dz_sat 0.62, **roughness 0.455**,
  meas_height −0.28, dist_sat 0.058.
- **Roughness and height are entangled:** `corr(roughness_sat, meas_height) = −0.43`.
- **But roughness survives removing height:** partial corr with error **0.455 → 0.388** — ~85%
  of the roughness signal is genuine, not a height artifact.
- **Height acts mostly *through* roughness:** height's error correlation is −0.28 raw but only
  **−0.10 after controlling for roughness**. Once the model knows roughness, height adds almost nothing.
- **Low vs high mast (split at median 149 m):**
  - ≤149 m (n=20): mean |error| = **5.76%**, roughness_sat = 0.360
  - \>149 m (n=20): mean |error| = **3.11%**, roughness_sat = 0.222

**Where we landed:**
- **Your justification is data-supported.** Lower-height cross-predictions genuinely have ~2× the
  error, roughness is a real signal (not just a height proxy), and it already captures the
  height-related error — so you likely **don't need a separate height feature.**
- **Caveats:** cross-sectional (different sites at different heights — can't fully separate
  "lower height" from "rough sites tend to get short masts"); n=40, wide CIs. The clean causal
  test is the same-site-at-two-heights WAsP re-run.
- **Deployment nuance:** pricing uncertainty at the existing (lower) height means you're likely
  **over-stating** the taller new turbine's horizontal uncertainty (its flow is smoother). That's
  the *safe* direction; a re-run at the taller height would quantify how conservative.
- Scripts: `Roughness height investigation/roughness_vs_error_confound.py` (+ `pair_features_with_height.csv`).

---

## 7. Distance feature & the dA-normalization (investigation, this session)

**Your side:**
- `dist_sat = 1 − exp(−distance_m / distance_A)`, where `distance_A` is complexity-derived
  (complex terrain → small dA, flat → large dA).
- Siting reality: flat terrain → masts far apart (large `distance_m`); complex terrain → masts
  close (small `distance_m`). Hypothesis: the feature ends up **larger in flat, smaller in complex**
  terrain — inverted vs difficulty — which could be a problem. (Also: distance explains *residual*
  variance after the other features.)

**My side (data + LOO bake-off):**
- Hypothesis **confirmed but the effect is *neutralization*, not inversion.** Numerator and
  denominator shrink together, so `dist_sat` is nearly constant across terrain even though error doubles:

  | Terrain (by RIX) | distance_m | distance_A | dist_sat | mean \|error\| |
  |---|---|---|---|---|
  | Flat (n=20) | 6278 m | 7874 m | **0.532** | **2.86%** |
  | Complex (n=20) | 2527 m | 4237 m | **0.517** | **6.00%** |

- The terrain signal *is* inside raw `distance_A` (corr −0.67 with error) but it's **81%
  reconstructable** from turning/speedup/roughness — little unique info.
- Distance still earns its keep on the **residual** (partial corr `dist_sat` 0.244; and you were
  right it mops up residual variance).
- **LOO bake-off of 6 distance parameterizations** (`distance_formula_comparison.py`):

  | variant | Pearson | Spearman | floor | note |
  |---|---|---|---|---|
  | **A_current** (1−exp(−d/dA)) | **0.9315** | 0.797 | 0.380% | **best Pearson** |
  | E_log | 0.9238 | 0.780 | 0.238% | close 2nd |
  | C_fix5k / D_fix8k | ~0.909 | ~0.815 | 0.32–0.39% | best Spearman, worse Pearson |
  | F_drop | 0.9057 | 0.728 | 0.859% | worst rank, floor jumps |
  | B_raw (linear) | 0.8747 | 0.785 | 0.561% | **worst Pearson** |

**Where we landed — CLOSED, keep the current feature:**
- `A_current` wins Pearson (the metric that matters, since output feeds P50→P90/P99) and is
  competitive on Spearman.
- **Key lesson:** `B_raw` had the *best in-sample* residual partial (+0.398) but the *worst
  out-of-sample* Pearson — raw distance overfits, and LOO caught it. The "ugly" normalization is
  protecting you.
- Distance is **not** a dead feature (dropping it worsens Spearman + raises the floor).
- All variants sit in a tight 0.90–0.93 band → the parameterization is **low-stakes**
  (collinearity again). No change to model or calculator.
- *Note:* run was at FAST settings (500 draws / 2 chains); re-run at 2000 for publication decimals,
  but the ranking is very unlikely to change.

---

## 8. Open options on the table (not yet done)

- **Additive-floor reparameterization** (makes the floor a dial, can reach ~0 at
  same-location): `sigma = sigma_floor + Σ betaₖ · fₖ`, with raw saturating features `fₖ`
  (= 0 at same-location) and `betaₖ, sigma_floor ≥ 0`. Floor becomes an explicit, settable
  parameter. **Costs:** full refit; multiplicative (current) may fit scatter better and your
  LOO Pearson is currently 0.93; the z-score / overall_z interpretation would need rework.
  Middle option: `sigma = sigma_floor + exp(log_sigma0 + Σγ·z)` — one line, no refit, but
  same-location reads `sigma_floor + 0.39%` rather than a clean `sigma_floor`.
- **Posterior-predictive switch** in the calculator (§2) — store each gamma's mean *and* SD
  and apply the `exp(0.5·Σ sₖ²·zₖ²)` correction. Small change; only worth it if the
  accept/reject threshold sits in the borderline band.
- **Diagnostics** worth running on the 40-pair fit: posterior SD of the shifted gammas, and
  LOO (pair- and site-level) to confirm the floor-lowering didn't cost calibration elsewhere.
