# The M-family: adaptive roughness formulas explained

Read this later when your brain is less fried. Written to be self-contained — walks through
the "why" from the beginning and explains every design choice.

---

## 1. The problem we're trying to solve

The sigma-only model has 5 features. Four of them (distance, turning, log_speedup, dz)
vanish at same-site — meaning if you predict a mast against itself, those features are all
zero. That's exactly what we want for a "clean floor" (small uncertainty at same location).

**Roughness is the odd feature out.** The current formula (called "formula A") is:

```
roughness_A = weighted_mean( |(rs_WTG + rs_MM) / 2| )
```

This does NOT vanish at same-site. If mast is over rough terrain, `rs_WTG` and `rs_MM` are
both nonzero (both equal the mast's own roughness speedup), so the average is nonzero, so
the feature fires. That's why production has a terrain-dependent floor (0.28% flat →
1.21% rough).

**We want a roughness formula that behaves more like the other 4 features** — vanishes at
same-site so all masts get the same clean floor.

---

## 2. The two candidate roughness formulas (A and B)

WAsP produces `rs_WTG` and `rs_MM` per sector — the fractional wind speed change WAsP
applied due to local roughness at each site. There are two natural ways to condense these
into one roughness number per pair:

### Formula A — magnitude of the average correction

```
A = weighted_mean( |(rs_WTG + rs_MM) / 2| )
```

This measures **how much correction WAsP applied on average**. Big value means big
corrections; small value means small corrections. This is the CURRENT thesis/production
formula.

### Formula B — mismatch between the two sites' corrections

```
B = weighted_mean( |rs_WTG - rs_MM| )
```

This measures **how differently WAsP treats the two sites**. If both sites get identical
corrections, B = 0. If they differ a lot, B is large.

### Why neither alone works well

**Formula A alone:**
- On kept sites: works well (LOO Pearson 0.918)
- On forest sites: over-predicts (forest sites have big rs values but low actual error)
- Empirical winner in bake-offs on kept-only data

**Formula B alone:**
- On kept sites: loses signal (LOO Pearson 0.870)
- On forest sites: would work better (mismatch is small for same-forest pairs)
- Empirical loser in bake-offs

Neither is universally right. Enter the adaptive idea.

---

## 3. What "min" actually does — with concrete examples

The M-family uses `min(A, B)` per pair. That means for each pair we:

1. Compute A (magnitude) for that pair
2. Compute B (mismatch) for that pair
3. Take the smaller of the two as the roughness feature

**Why min? Because A and B are two different views of the same phenomenon (WAsP's
roughness treatment), and taking the smaller value is a conservative "we don't want to
double-count" choice.**

### Concrete examples

**Case 1: Both sites in same forest** (rs_WTG = rs_MM = -0.010)
```
A = |(-0.010 + -0.010) / 2| = 0.010    (large — big corrections applied)
B = |-0.010 - -0.010|      = 0.000    (zero — sites agree perfectly)
min(A, B) = 0.000
```
Feature says: "roughness contributes zero uncertainty here." Matches physical reality
(same terrain, corrections cancel in the transfer ratio). This is the forest case where
formula A over-predicts but formula B correctly predicts low.

**Case 2: Both sites at same location** (self-prediction / repowering)
```
rs_WTG = rs_MM by definition
A = |rs_own|                  (nonzero for rough sites)
B = 0                         (mismatch is trivially zero)
min(A, B) = 0
```
This is why min-based formulas give clean same-site floors. Roughness always contributes
zero at self-site because B = 0 there.

**Case 3: One site over rough terrain, other on smooth** (rs_WTG = -0.010, rs_MM = 0)
```
A = |(-0.010 + 0) / 2|  = 0.005   (moderate)
B = |-0.010 - 0|        = 0.010   (larger — sites treated differently)
min(A, B) = 0.005
```
Feature picks A. Model says "moderate roughness contribution — because average correction
size is moderate, we won't double-count the mismatch."

**Case 4: Opposite-sign corrections** (rs_WTG = +0.005, rs_MM = -0.005)
```
A = |(+0.005 + -0.005) / 2| = 0.000  (mean cancels — magnitude looks zero)
B = |+0.005 - -0.005|       = 0.010  (large — sites disagree)
min(A, B) = 0.000
```
Feature picks A (which is zero due to sign cancellation). Model says "zero contribution."
This might under-predict for genuinely-opposite-terrain pairs — a real trade-off of the
min approach.

### Summary of the physical intuition

- **When sites AGREE closely (small mismatch): min picks B** → correctly predicts low
  uncertainty for pairs where both sites share terrain (forest-in-forest, flat-with-flat)
- **When sites DIFFER strongly (large mismatch): min picks A** → uses magnitude as a bound,
  preventing runaway uncertainty from a single large mismatch
- **At self-site: min ALWAYS gives 0** because B = 0 by construction (same rs on both
  sides) → clean single-value floor

The min operation is a "pick whichever suggests less uncertainty" rule. It's conservative
in the sense of "don't count the same physical thing twice from two angles," but it can
UNDER-predict on genuinely-different-terrain pairs.

---

## 4. The four M-family variants

Once we settle on min-based adaptation, there are two independent choices:

- **When to take min**: before or after standardization?
- **When to center**: are the features centered (mean subtracted) or not?

That gives 4 variants in a 2x2 grid:

| | Uncentered (no mean shift) | Centered (mean shifted) |
|---|---|---|
| **Min-first** (take min of raw sat values, then standardize) | **exp-M** | **exp-Mc** |
| **Std-first** (standardize each formula, then take min) | **exp-M2** | **exp-M2c** |

Each addresses a different concern.

---

### exp-M — "min-first, no centering"

```
per pair:
  raw_M = min( A, B )                          # take min of raw magnitudes
  sat_M = 1 - exp( -raw_M / 0.01 )             # saturate
across pairs:
  std_M = std( sat_M values across all pairs )
model:
  log(sigma) = log_sigma0 + ... + beta_rough * sat_M / std_M
```

**Why this design:**
- min is applied to raw values (before standardization) — simplest interpretation
- Single std used for the combined feature — treats mixed distribution as one
- No centering — log_sigma0 keeps its meaning as "value at all-features-zero" = intrinsic floor

**Trade-off:**
- Mathematically inconsistent: single std doesn't reflect that A-values and B-values
  come from formulas with different natural spreads
- But empirically fits well on kept-only data

**Self-σ:** single value = `exp(log_sigma0)`, clean interpretation.

---

### exp-Mc — "min-first, with centering"

```
per pair:
  raw_M = min( A, B )
  sat_M = 1 - exp( -raw_M / 0.01 )
across pairs:
  mean_M = mean( sat_M values )
  std_M  = std( sat_M values )
model:
  log(sigma) = log_sigma0 + ... + gamma_rough * (sat_M - mean_M) / std_M
```

**Why this design:**
- Same min-first structure as exp-M
- Adds centering (subtract training mean) — matches production's z-scoring convention
- log_sigma0 now = "value at training-mean pair" (not the floor number)

**Trade-off:**
- Still has the mixed-distribution std issue
- log_sigma0 loses its direct "this IS the floor" meaning — floor becomes
  `exp(log_sigma0 - Sigma gamma_k * mean_k / std_k)`
- But fit slightly better than exp-M

**Self-σ:** single value (because all features = 0 at self, and each contributes
a constant `-mean_k * gamma_k / std_k`), typically LOWER than exp-M's floor.

---

### exp-M2 — "std-first, no centering"

```
across pairs (from training):
  std_A = std( sat_A values across pairs )    # A's own natural spread
  std_B = std( sat_B values across pairs )    # B's own natural spread
per pair:
  z_A = sat_A / std_A                          # standardize each separately
  z_B = sat_B / std_B
  x_rough = min( z_A, z_B )                    # min of already-standardized values
model:
  log(sigma) = log_sigma0 + ... + beta_rough * x_rough
```

**Why this design:**
- Each formula gets standardized on its OWN natural spread (fixes the mixed-distribution
  concern of exp-M)
- min taken AFTER standardization — compares "how extreme is this pair on A's scale" vs
  "how extreme on B's scale"
- No centering

**Trade-off:**
- Mathematically cleaner than exp-M
- But empirically fits WORSE on kept-only data (Pearson 0.875 vs exp-M's 0.901)
- The per-formula standardization can flip WHICH formula wins the min for some pairs,
  and those flips apparently hurt kept-only fit

**Self-σ:** single value = `exp(log_sigma0)` (same as exp-M because sat_B = 0 at self →
z_B = 0 → min = 0 regardless of standardization).

---

### exp-M2c — "std-first, with centering" (the mathematically strictly-correct version)

```
across pairs (from training):
  mean_A, std_A of sat_A values
  mean_B, std_B of sat_B values
per pair:
  z_A = (sat_A - mean_A) / std_A               # per-formula z-score (center + scale)
  z_B = (sat_B - mean_B) / std_B
  x_rough = min( z_A, z_B )
model:
  log(sigma) = log_sigma0 + ... + gamma_rough * x_rough
```

**Why this design:**
- Combines exp-M2's per-formula standardization with production's centering
- MATHEMATICALLY MOST RIGOROUS variant: each formula gets full z-scoring on its own
  distribution before combining
- No mixed-distribution std problem, no missing centering

**Trade-off:**
- Some fit cost on kept-only data
- Self-σ has small spread (not fully single-value) because for flat masts z_A_self and
  z_B_self are similar (min picks either), for rough masts z_A_self > z_B_self so min
  picks z_B_self (constant negative value)
- BEST forest-in behavior: smallest LOO Pearson drop when forest is added

**Self-σ:** small spread (0.43-0.49% forest-out, 0.45-0.46% forest-in). For rough
masts min picks the constant `z_B_self = -mean_B/std_B`. For very flat masts min picks
`z_A_self = (sat_A_own - mean_A) / std_A` which varies slightly.

---

## 5. Complete results table (forest-out)

| variant | Pearson | Spearman | self-sigma | forest-in penalty |
|---|---|---|---|---|
| exp-A (formula A alone, no adaptive) | **0.9185** | 0.7669 | 0.33-1.30% | -0.036 |
| exp-B (formula B alone, no adaptive) | 0.8706 | 0.5859 | 0.53% (single) | +0.002 |
| exp-Q (formula A × distance gate) | 0.9156 | 0.7331 | 0.44% (single) | -0.039 |
| exp-M | 0.9012 | 0.7725 | 0.47% (single) | -0.020 |
| exp-Mc | 0.9058 | **0.7751** | 0.44% (single) | -0.024 |
| exp-M2 | 0.8748 | 0.7263 | 0.50% (single) | +0.011 |
| **exp-M2c** | 0.8926 | 0.7725 | 0.43-0.49% | **-0.006** (smallest) |

Where "forest-in penalty" = drop in Pearson when adding forest pairs (Hultema + Malarberget).

### What jumps out

- **exp-Mc has the highest Spearman of ANY config.** Best rank-ordering accuracy.
- **exp-M2c has the smallest forest-in penalty of any reasonable config.** Most gracefully
  handles forest inclusion.
- **exp-A has the highest Pearson** but terrain-dependent floor.

---

## 6. How to pick between them

There's no universally best answer. The choice depends on what you prioritize:

### If highest raw fit matters most: exp-A forest-out
- Pearson 0.9185 (highest)
- Terrain-dep floor 0.33-1.30% (unavoidable with pure formula A)
- Uses 41 pairs (excludes forest)
- Simplest formula to explain

### If best rank accuracy + clean floor + centering: exp-Mc forest-out
- Highest Spearman (0.7751)
- Clean single-value self-sigma (0.444%)
- Uses centering like production, so log_sigma0 needs "value at training mean" story
- Still has the mixed-distribution std issue (mathematical inconsistency you flagged)

### If most physical/direct interpretation of intrinsic floor: exp-M forest-out or forest-in
- log_sigma0 = intrinsic floor directly (no centering shift)
- Clean single-value self-sigma
- Still has mixed-distribution std issue
- Slightly lower Pearson than exp-Mc

### If mathematical rigor + uses all data + smallest forest-in cost: exp-M2c forest-in
- Per-formula standardization AND centering — no mathematical inconsistencies
- Uses all 45 pairs
- Smallest forest-in Pearson penalty (-0.006 vs -0.024 for exp-Mc)
- Second-highest Spearman forest-out (0.7725)
- Self-sigma has small spread (0.43-0.49%), not fully single-value

---

## 7. Physical trade-off: what min gives up

**Concrete example where min-based formulas can under-predict:**

Doringbaai pairs (16.7 km apart, different terrain types):
- Actual |e| = 0.08-0.09
- exp-A prediction: 0.09-0.10 (close, slight over-predict)
- exp-M prediction: 0.06-0.07 (under-predict by 0.02)

Why? For these pairs, mismatch (B) is larger than magnitude (A), so min picks A. But the
actual physical uncertainty for a 16 km far-apart pair with different terrain is closer to
what B would suggest. min under-predicts here.

**Concrete example where min-based formulas outperform:**

Sundern close pairs (1.3-1.4 km, similar terrain):
- Actual |e| = 0.03-0.04
- exp-A prediction: 0.05-0.06 (over-predict)
- exp-M prediction: 0.04-0.05 (matches almost exactly)

Here mismatch is small (similar sites), min picks the smaller value, matches better.

**The trade-off is real:** M variants are BETTER on similar-terrain / close / forest-like
pairs, WORSE on genuinely-different-terrain / far-apart pairs.

Whether this trade-off is worth it depends on your typical deployment scenario:
- Mostly close-distance repowering: M variants win
- Mostly far-distance extrapolation to different terrain: exp-A is safer

---

## 8. About dropping the bias term

Historical note: earlier the model had `mu = beta_dz * dz` as a bias predictor (predicts
mean of signed error). We dropped it because:
- Supervisor wasn't accepting the bias/uncertainty separation concept
- The bias term was tiny (contributed ~2% of total explanation)

**What happened to the bias signal:**
- Target changed from signed e_overall to |e_overall|
- Any residual mean-shift in the data now goes into sigma (folded into scale, not
  separated out as mean)
- dz still appears as a sigma feature (dz_sat) — so dz-dependent effects are still
  captured, just as scale rather than mean
- You lose the ability to predict which DIRECTION the error will go (over- vs under-
  predict), but the model was doing this weakly anyway

Practically: sigma predictions increase very slightly (~1-2%) to absorb the previously-
separated bias signal. Not a game-changer either way.

---

## 9. Other known issues (flagged for future work)

1. **Roughness saturation scale (0.01) may be too tight for corrected data.** Corrected
   data has max |rs| = 0.026 in kept pairs, which saturates the feature at 0.92 (near
   ceiling). Might lose discrimination among the highest-roughness pairs. Retesting with
   scale 0.02 or 0.03 might sharpen high-end discrimination.

2. **Speedup feature is unbounded** (`wm_abs_log_speedup` doesn't saturate). Empirically
   self-limits in practice (speedup values ~0.5-1.5 give small log ratios) but at extreme
   deployment sites it could produce unusually high sigmas.

3. **Herzhausen pairs are systematically over-predicted** by all variants (~0.25
   predicted, ~0.18 actual). The current 5-feature framework can't quite match these
   pairs; not a bug, just a limitation of what our features measure.

4. **Small n (41-45)** means coefficients shift moderately (20-30%) when adding just 4
   forest pairs. LOO cross-validation is somewhat protective but the fits are inherently
   sensitive.

---

## 10. The two-line answer if you don't want to read everything

For the current thesis defense with maximum mathematical rigor: **exp-M2c forest-in**.
For the strongest physical-interpretation-of-floor story: **exp-M** or **exp-Mc forest-out**.

If in doubt, exp-Mc forest-out is a middle-ground safe pick: high Spearman, clean single
floor, uses centering like production, near-best Pearson.