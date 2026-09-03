# Variance Decomposition Explained

Reference document for later. Explains what the different variance decomposition
methods are, why they give different numbers, and what the "real" attribution is
for exp-M2c.

Read this whenever you're trying to defend how much each feature contributes to
sigma predictions, or when someone questions why one number is 27% and another
is 15% for the same feature.

---

## 1. The problem in one sentence

When features in a regression are correlated, "how much does feature X contribute
to the prediction?" doesn't have a unique answer — different attribution methods
give different numbers, all mathematically defensible.

Our five features have real correlations (turning-speedup at 0.81 is the biggest),
so this problem bites us. **The naive `beta_k^2` decomposition can be very
misleading.** The proper decomposition (LMG / Shapley) is what should be reported.

---

## 2. The math — where variance comes from

Our model is:

```
log(sigma) = log_sigma0 + sum_k beta_k * z_k
```

where each z_k is a standardized feature. The variance of `log(sigma)`
predictions across the training pairs is:

```
Var(log_sigma) = Var( sum_k beta_k * z_k )
              = sum_k beta_k^2 * Var(z_k)               <- diagonal terms
                + 2 * sum_{i < j} beta_i * beta_j * Cov(z_i, z_j)   <- cross-terms
```

The first sum is what "naive" decomposition reports. The second sum — the
**cross-terms** — is what naive decomposition ignores.

### Concrete example — 2 features

Suppose two features X and Y, both z-scored (variance 1), with correlation `r`:

```
Var(beta_X * z_X + beta_Y * z_Y)
   = beta_X^2 + beta_Y^2 + 2 * beta_X * beta_Y * r
```

- If `r = 0` (independent): total = `beta_X^2 + beta_Y^2` — naive is correct.
- If `r = +1` (perfectly correlated, same direction): total = `(beta_X + beta_Y)^2`.
  Naive would UNDER-count by ignoring the cross-term.
- If `r = -1` (perfectly correlated, opposite): total = `(beta_X - beta_Y)^2`.
  Naive would OVER-count.

**For our 5 features, ratio(actual/naive) = 2.08** — features cooperate on
average, adding 108% to the naive sum via positive cross-terms.

---

## 3. Pairwise cross-terms in exp-M2c

Each term is `2 * beta_i * beta_j * Cov(z_i, z_j)`. Sorted by magnitude:

| Feature pair | correlation | cross-term | % of total variance |
|---|---|---|---|
| turning × log_speedup | +0.81 | +0.100 | **+14.4%** |
| log_speedup × dz | +0.72 | +0.067 | +9.6% |
| turning × dz | +0.64 | +0.062 | +8.8% |
| log_speedup × roughness | +0.33 | +0.047 | +6.8% |
| distance × dz | +0.35 | +0.040 | +5.7% |
| dz × roughness | +0.25 | +0.028 | +4.0% |
| turning × roughness | +0.19 | +0.028 | +4.0% |
| distance × log_speedup | -0.11 | -0.016 | -2.3% |
| distance × roughness | +0.04 | +0.008 | +1.1% |
| distance × turning | 0.00 | -0.000 | 0.0% |

**Total cross-terms: +52% of variance.** Just over half the model's predictive
variance comes from features working together, not features working alone.

The turning-speedup-dz triangle is the big cross-term cluster — each pair has
correlation 0.6-0.8. Distance is the loner — its biggest correlation with the
other features is 0.35 (with dz).

---

## 4. Attribution methods

Once we know the total variance is `beta^T Cov beta`, we need to decide how much
of it to give each feature. Several methods, each with different properties:

### 4.1 Naive: `beta_k^2 * Var(z_k)`

**What it does:** Assumes each feature contributes independently. Ignores cross-terms.

**Pros:** Simple, common in tutorials.

**Cons:**
- Sum can be greater or less than the actual variance (doesn't add up correctly).
- For correlated features, overstates the "sole" role of each.
- Uninformative when correlations are strong.

**When to use:** Only when features are essentially uncorrelated (rare).

### 4.2 Full-covariance / row-attribution: `beta_k * (Cov beta)_k`

**What it does:** Attributes each cross-term to the row-feature. Specifically:

```
share(k) = beta_k * ( sum_j beta_j * Cov(k, j) )
        = beta_k^2 * Var(z_k)           <- own variance
        + beta_k * sum_{j != k} beta_j * Cov(k, j)   <- half of each cross-term
```

**Pros:** Sums to actual variance by construction. Simple to compute.

**Cons:** Not symmetric in "attribution philosophy" — half-splitting cross-terms
is one specific choice; others exist.

### 4.3 LMG / Shapley: average over orderings

**What it does:** For each feature k and each possible ordering of the K features,
compute the marginal variance added when feature k enters at its position.
Average over all K! orderings.

```
LMG(k) = (1/K!) * sum_over_orderings [ Var(features up to k) - Var(features up to just before k) ]
```

**Pros:**
- Game-theoretically fair (Shapley value from cooperative game theory).
- Symmetric: each feature gets equal treatment in the averaging.
- Always non-negative.
- Sums to actual variance.
- Widely accepted as the "gold standard" for correlated-feature regression attribution.

**Cons:** More computation (2^K subset variances). Feasible for K = 5.

### 4.4 Interesting mathematical fact for our case

**For fixed betas, LMG and full-covariance give the SAME numbers.** This isn't
always true — they can differ if you're re-fitting the model for each subset. But
we hold the fitted betas constant, so both methods reduce to:

```
share(k) = beta_k * (Cov beta)_k
```

That's why the "Full-cov %" and "LMG %" columns in the output are identical.

---

## 5. exp-M2c decomposition — final answer

For each feature, three attributions:

| feature | Naive % | Full-cov % | **LMG %** |
|---|---|---|---|
| WM log_speedup | 17.9% | 22.8% | **22.8%** |
| Saturating turning | 19.0% | 22.7% | **22.7%** |
| Adaptive roughness M2c | 25.3% | 20.0% | **20.0%** |
| Saturating dz | 10.9% | 19.3% | **19.3%** |
| Saturating distance | **27.0%** | 15.2% | **15.2%** |

### Is LMG the "real real" decomposition?

**Yes — this is the answer to defend.** It correctly accounts for feature
correlations and sums to the actual variance of the model's predictions.

The naive numbers (which the current model script prints) are useful as a
first-look, but they systematically over-attribute to features that are largely
independent (like distance) and under-attribute to features in correlated
clusters (like turning/speedup/dz).

### Interpretation for defense

Under the LMG (proper) decomposition:

- **Distance carries 15.2%** — very close to the thesis-final's 14.6%. **The
  naive 27% was misleading.** Distance's role is UNCHANGED from thesis-final.
- **Roughness dropped from 34% (thesis-final) to 20%** — this IS a real
  redistribution. Aligns with the "less-dominant roughness" preference.
- **Turning dropped from 35% to 23%** — also a real drop. Turning is no longer
  the dominant feature.
- **Speedup jumped from 7% to 23%** — big real increase. Under exp-M2c, speedup
  is a co-leading feature.
- **dz jumped from 3% to 19%** — bigger role than it had in thesis-final.

The model went from "two-feature-dominated" (turning + roughness at 68% combined)
to "balanced-five-feature" (each in the 15-23% range).

---

## 6. When to use which method

| context | recommended method |
|---|---|
| Quick sanity check | Naive |
| Defense/thesis writeup | **LMG / Shapley** |
| Reporting to non-technical audience | LMG with plain-English framing |
| Debugging why a specific feature matters | Full-covariance + cross-term table |
| Software packages (R's `relaimpo`) | LMG is the default |

**Rule of thumb:** if any pairwise correlation is above 0.3, use LMG (or
equivalently full-covariance). If all correlations are below 0.3, naive is
approximately correct.

Our features have max correlation 0.81 (turning-speedup), so we're firmly in
"must use LMG" territory.

---

## 7. Script location

The decomposition script that produces these numbers:
`Post-thesis corrections/expM2c_variance_decomposition.py`

It:
1. Loads `Results_expM2c/ws_uncertainty_expM2c_results.json` for coefficients
2. Loads `Results_expM2c/ws_uncertainty_expM2c_feature_contributions.csv` for per-pair z-scores
3. Computes correlation and covariance matrices across features
4. Reports naive, full-covariance, and LMG decompositions
5. Prints pairwise cross-terms
6. Zooms in on distance specifically (the historically-questioned feature)

Rerun it any time the model is refit. Takes seconds (pure numpy, no PyMC).

---

## 8. Extra note — total variance changed too

Under exp-M2c: total Var(log_sigma) = 0.698

Under thesis-final (naively summed from published gammas): ~0.223

The model's total predictive variance nearly TRIPLED. Two contributing reasons:
1. Bias term removed — variance previously explained by `mu` now goes into sigma.
2. Adaptive roughness compression — coefficients grew to compensate for smaller
   roughness feature values, and correlations amplified this via cross-terms.

**Practical implication:** the model separates high-uncertainty from low-uncertainty
pairs more strongly than thesis-final. This can be good (better discrimination)
or concerning (potential over-fitting) — but at n=45 with 5 features and LOO
Pearson 0.885, it's not obviously overfitting.

---

## 9. Full script output snapshot (for record)

```
Total Var(log_sigma) predictions: 0.69813
Naive sum (ignoring covariance):  0.33509
Ratio actual/naive:               2.083  (features cooperate on average)

DECOMPOSITION 1 — NAIVE
  Saturating distance                  0.09046       27.0%
  Adaptive roughness M2c               0.08472       25.3%
  Saturating turning                   0.06354       19.0%
  WM |log speedup ratio|               0.05996       17.9%
  Saturating |dz|                      0.03640       10.9%

DECOMPOSITION 2 — FULL COVARIANCE (attributes cross-terms to row-feature)
  WM |log speedup ratio|               0.15938       22.8%
  Saturating turning                   0.15823       22.7%
  Adaptive roughness M2c               0.13990       20.0%
  Saturating |dz|                      0.13444       19.3%
  Saturating distance                  0.10618       15.2%

DECOMPOSITION 3 — LMG / SHAPLEY
  (identical to FULL COVARIANCE for fixed-beta case — same 15-23% each)
```

---

## 10. One-line takeaway

**Distance is 15% (LMG), not 27% (naive). Report the LMG numbers.**
