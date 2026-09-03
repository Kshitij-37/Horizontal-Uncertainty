# WS Uncertainty Model — Status & Action Plan

*Snapshot 2026-07-06. Companion to `Lowering min Uncertainty/Discussion_notes_floor_plugin_consistency.md`
(which has the deeper reasoning). This file is the scannable "where things stand + what to do next".*

---

## 0. The one thing that reframes everything: the input data was corrected

`Focused_modelling_inputs.xlsx` was regenerated with **corrected roughness numbers** (the wind-speed
errors `e_overall` are unchanged — only `rough_speedup_*_frac` changed). **Consequence: every fitted
JSON currently in the repo (Final model + Post-thesis) was fit on the OLD/wrong roughness data**, and
so are the calculators that bundle them. Anything roughness-dependent needs a re-fit. This is the
headline action item; most of the plan below flows from it.

The correction already dissolved one earlier "finding": the dramatic *"magnitude roughness is
backwards on the forest pairs"* result (r = −0.98) was an artifact of the wrong numbers — on corrected
data it's r = +0.99. So the forest/mismatch case is now **weak and open**, not proven.

---

## 1. DECIDED / OPTIMISED (closed — no action needed)

- **Floor (min uncertainty ≈ 0.39–0.51%):** accepted. It's the irreducible WAsP self-prediction error;
  can't reach 0 with this architecture. Adding the 5 m pair lowered it modestly and shrank the
  extrapolation gap. **Closed.**
- **Plug-in vs posterior-predictive:** keep **plug-in**. The extrapolation tail (z+15) is physically
  unreachable (observed max ≈ +3.3 SD → speedup ratio ~1.2×). **Closed.**
- **Unbounded speedup / "extreme over-prediction at high values":** not a real failure mode — the model
  *under*-predicts the tail (shrinkage, slope ≈ 0.5), which is the safe direction. **Closed.**
- **Should the tool extrapolate? / z-clipping:** yes, it should extrapolate monotonically; do **not**
  clip. **Closed.**
- **Distance feature (`dist_sat = 1−exp(−d/dA)`):** LOO bake-off of 6 variants → current form is best on
  Pearson; raw distance overfits. dA-normalisation neutralises the terrain signal but distance still
  earns its keep on the residual. **Closed — keep current.** (Run on OLD data, but distance inputs
  didn't change, so still valid.)
- **`overall_z` column** added to the calculator (effect-weighted avg). **Done + verified.**
- **v3 / v4 calculator split** built and made internally consistent:
  - **v3** = thesis-final, **un-gated** roughness, bundles Final-model JSON, fallback path fixed.
  - **v4** = post-thesis, **gated** roughness, bundles Post-thesis JSON.
  - Fixes the earlier train/serve mismatch (v3 code once gated but bundled an un-gated JSON).
- **`d(fwd-rev)` column** restored to the model's pair printout (A→B vs B→A error delta).
- **Gamma decomposition (pairs vs gate):** the v4 gamma jump was driven **mostly by the added pairs**
  (esp. the 5 m pair), not the gate. Leverage is **benign** — 40-pair LOO *improved* to 0.931/0.797.
- **Roughness = magnitude, and the gate is load-bearing:** the feature is `|(rs_W+rs_M)/2|` (a
  magnitude, not a mismatch), so the gate is what gives close pairs low uncertainty. Mismatch variants
  (B/C/E/H/I/J/K/M) all *lost* the bake-off **on forest-excluded data**.

---

## 2. ACTION PLAN (in priority order)

### ▶ P1 — Re-fit the model on the corrected data  *(everything downstream depends on this)*
- [ ] Run `ws_uncertainty_model_roughness_fix` in **pymc-env** on the corrected Excel → regenerates
      `Post-thesis corrections/Results/…json` with corrected roughness scalers + `gamma_roughness`.
- [ ] (If the Final/thesis model is still a shipping target) re-fit it too on corrected data.
- [ ] Sanity-check LOO vs the old numbers (Pearson was ~0.93; confirm it holds).

### ▶ P2 — Resolve the forest re-inclusion (Hultema ± Malarberget) — *do it inside the P1 re-fit*
- [ ] Run the model **forest-included** (masts commented out in `EXCLUDED_MASTS`) **and**
      **forest-excluded**, both on corrected data. Compare the two LOO summaries.
- [ ] **Decision rule:** if LOO holds with forest pairs in → **re-include them** (free extra data), done.
      If it still drops → run `forest_roughness_test.py loo` (A/B/E/Q) to see if a mismatch/both feature
      rescues them; only then consider a feature change.
- Note: the model file currently has both forest masts un-excluded; `forest_roughness_test.py` currently
  re-excludes Malarberget (your latest edit) — reconcile which set you actually want before deciding.

### ▶ P3 — Regenerate calculator JSONs + rebuild the exes
- [ ] Point v3 (Final) and v4 (Post-thesis) at their **re-fit** JSONs.
- [ ] Rebuild both exes via each folder's `build.bat`. **Current exes are stale** (v3's is the old
      hybrid; v4 was never built) **and now also fit on wrong data.**
- [ ] Decide the **canonical** shipping model: v3 (thesis-final, un-gated) vs v4 (post-thesis, gated).
      This is the long-standing consistency decision (§4 of the discussion notes), now on corrected data.

### ▶ P4 — Documentation / caveats
- [ ] Record the **measurement-selection caveat**: the magnitude roughness feature can rank an
      *inside-forest* mast as more uncertain than an *outside-forest* one, so the tool must not be
      inverted into "pick the lower-σ mast." (Weaker on corrected data, but worth stating.)
- [ ] Update the discussion notes with the corrected-data outcome once P1/P2 land.

---

## 3. FIXES / KNOWN ISSUES

- **Stale, wrong-data artifacts:** all bundled JSONs + both calculator exes are fit on the old roughness
  numbers → superseded once P1 runs. (Highest-impact "fix".)
- **Model-source consistency:** three JSONs existed (Final, Final/Results_roughsat identical, Post-thesis);
  make sure the calculators, `.spec`, and `build.bat` all point at ONE re-fit file per version.
- Already fixed this cycle: v3 fallback path, v3 gate removal, v4 creation, `overall_z`, delta column.

---

## 4. BACKLOG (optional, marginal — only if you want to squeeze more)

- **[FLAGGED IDEA] Find other bias predictors for `mu` beyond dz — e.g. dRIX.** Currently dz is the
  *sole* directional driver, so any other signed/systematic effect is being mislabelled as
  *uncertainty* (leaks into sigma). `RIX_A − RIX_B` (dRIX) is a prime candidate: RIX mismatch is a
  known systematic WAsP prediction bias, and it **flips sign with direction** (antisymmetric), so it
  fits the `mu` structure exactly. Test: add dRIX (signed) to `mu`, check whether it (a) has a
  non-zero coefficient, (b) reduces residual scatter / improves LOO, (c) explains any of the pairs dz
  can't (e.g. large-dz outliers where the linear −5.6%/100 m under-predicts). Data already has
  `dRIX_0.3_sector` / `dRIX_0.0501_sector` / `RIX_avg_*` columns. Screen candidates by
  corr(signed-error, candidate) after removing the dz trend.
- **Marginal accuracy levers** (each worth ~tenths of a point, validate via LOO):
  - more **tail** data (high-dissimilarity pairs) — the only real lever for the under-predicted tail;
  - test a **wider gamma prior** (HalfNormal 0.3 → 0.5) — directly targets the shrinkage, may not survive LOO;
  - test **energy** sector-weighting vs freq weighting.
- **Additive-floor reparameterisation** — only if a stakeholder insists same-location must read ~0%.
  It's a re-architecture; current multiplicative form fits better, so probably skip.
- **Excel version of the calculator** — feasible only as a single-pair manual sheet; full paste-and-parse
  parity needs VBA. You flagged this as informational only.

---

## 5. Script reference (what to run for what)

| script | role | when |
|---|---|---|
| `ws_uncertainty_model_roughness_fix` | **the model** (fit + LOO + writes JSON) | **P1/P2 — run now, both forest ways** |
| `forest_roughness_test.py` | Part 1 diagnostic (no pymc); `… loo` = A/B/E/Q on forest-inclusive | only if P2 shows forest pairs still hurt |
| `roughness_formula_comparison.py` | roughness formula bake-off (uses **Final** model's exclusions = forest-excluded) | done; re-run on corrected data only if revisiting features |
| `distance_formula_comparison.py` | distance feature bake-off | done (concluded: keep current) |
| `gamma_decomposition.py` | gate-vs-pairs MAP decomposition | done (one-off) |
| calculator `build.bat` (v3 & v4) | build the exe from source + JSON | **P3 — after re-fit** |

**Immediate next action:** run `ws_uncertainty_model_roughness_fix` in pymc-env on the corrected data,
forest-included then forest-excluded, and read the two LOO summaries. Everything else waits on that.
