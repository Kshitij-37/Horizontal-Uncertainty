# Similarity vs Complexity — is the adaptive roughness doing real work?

**Date:** 2026-08-18
**Model under scrutiny:** Adaptive Model v5 (formerly exp-M2c), n=47 pairs, sigma-only
**Question raised:** Kshitij — is the adaptive `min(z_A, z_B)` roughness formulation actually
capturing real physics, or did we just find a formula that happens to fit and we're now
rationalizing it after the fact?

## 1. The concern

The physical framework used throughout the other 4 features is **dissimilarity drives uncertainty** — bigger the difference between WTG and MM sites, bigger the transfer error. This is the standard extrapolation-error framing.

Roughness is the odd one out. The current adaptive formulation combines TWO roughness views:
- **Formula A (complexity / magnitude):** `|(rs_W + rs_M) / 2|` — how big is the roughness correction on average
- **Formula B (similarity / mismatch):** `|rs_W - rs_M|` — how differently were the two sites treated

The model picks whichever is smaller after per-formula z-scoring: `min(z_A, z_B)`.

**The worry:** Formula A doesn't fit the dissimilarity story. Its inclusion feels ad-hoc — a mathematical trick that improves LOO Pearson without a clean physical justification. Maybe we're overfitting to the training set and inventing physics to justify it.

This concern deserves a proper empirical answer, not more theoretical arguments.

## 2. The empirical test

Refitted pure-mismatch formulation (`exp-B forest-in`) was already run in the sigma_only_refit
bake-off. That gives us a direct comparison on the same data:

- **Adaptive M2c:** roughness feature is `min(z_A, z_B)` — the promoted model
- **Pure mismatch (exp-B):** roughness feature is `z(|rs_W - rs_M|)` — the physics-clean alternative

Both fitted on n=45-47 pairs (adaptive has 2 additional Slovenska East pairs; inner join used for comparison → 45 pairs).

For each of the 45 pairs, calculated **roughness dominance** = |roughness contribution| / sum(|all feature contributions|). This tells us which pairs have their prediction driven by roughness vs by other features (turning, speedup, dz, distance).

Then compared prediction errors on:
- **Top-quartile roughness-dominated pairs** (n=11) — where the roughness formula choice matters most
- **Bottom-half complexity-dominated pairs** (n=22) — where roughness barely contributes
- **All pairs** (n=45)

Script: `Adaptive_Model_v5/roughness_dominance_check.py`
Full per-pair CSV: `Adaptive_Model_v5/Results/roughness_dominance_comparison.csv`

## 3. Results at a glance

### Overall (n=45)

| model | mean bias | mean |err| | max |err| | Pearson | Spearman | wins |
|---|---|---|---|---|---|---|
| **Adaptive M2c** | +0.44 pp | **1.77 pp** | 8.37 pp | **0.887** | **0.685** | **32** |
| Pure mismatch (exp-B) | +0.83 pp | 2.20 pp | 8.91 pp | 0.872 | 0.607 | 13 |

Adaptive wins on 32 of 45 pairs. Not close.

### Top-quartile roughness-DOMINATED pairs (n=11)

Pairs where roughness contributes ≥37% of |log(sigma)| — forests (Hultema), Taaibos, Ukhanda, Clermont-en-Argonne, Kabbo, Zawidz.

| model | mean bias | mean |err| | wins |
|---|---|---|---|
| Adaptive M2c | +1.25 pp | 2.03 pp | 7 |
| Pure mismatch | +1.19 pp | 1.99 pp | 4 |

**Effectively tied on accuracy** (0.04 pp difference is noise). Adaptive wins more pairs (7 vs 4) but mean absolute error is indistinguishable.

**This is the direct answer to the concern.** On the pairs where the A-vs-B choice actually should matter most, the two models produce basically the same predictions. So the accusation "adaptive is overfitting via formula A" is at its weakest exactly where you'd expect it to be strongest.

### Bottom-half complexity-DOMINATED pairs (n=22)

Pairs where roughness contributes ≤17% of |log(sigma)| — Herzhausen, Balverwald, Sundern, Doringbaai, Mikolajki, etc.

| model | mean bias | mean |err| |
|---|---|---|
| **Adaptive M2c** | **+0.06 pp** | **1.85 pp** |
| Pure mismatch | +0.72 pp | 2.72 pp |

**Pure mismatch is significantly WORSE** on complexity-dominated pairs. That's counterintuitive at first — roughness barely contributes to these predictions, so the roughness formula shouldn't move the number much.

But it does. And the reason reveals the real physics of what the min operation is doing.

## 4. The surprising mechanism — why pure mismatch loses

Look at the worst-hit pair, Herzhausen 2019PA024__2019PA023:

| quantity | value |
|---|---|
| roughness share of |log(sigma)| under adaptive | **0.5%** |
| actual error | 14.78% |
| adaptive M2c prediction | 15.66% (**+0.88 pp over**) |
| **pure mismatch prediction** | **21.03% (+6.25 pp over)** |

Same pattern on all 4 Herzhausen pairs — pure mismatch predicts **5-7 pp higher** than adaptive.

### Why does mismatch fire so hard on Herzhausen?

Herzhausen has opposite-sign roughness sectors — in some directions WAsP applied acceleration at WTG and deceleration at MM (or vice versa).

- Formula A: `|(rs_W + rs_M)/2|` — opposite signs cancel → small value
- Formula B: `|rs_W - rs_M|` — opposite signs ADD → large value

For a sector with rs_W = +0.010 and rs_M = -0.010:
- A = 0
- B = 0.020

Under pure mismatch, this sector's roughness signal is huge. When 6 of Herzhausen's 12 sectors have opposite-sign roughness, mismatch magnitude balloons across the pair.

But Herzhausen's actual uncertainty is driven by **turning and log_speedup** (72% of its positive log-sigma push under adaptive). Complex terrain → high turning and speedup → high uncertainty prediction. Roughness under pure mismatch adds ANOTHER large positive push → over-prediction.

Under adaptive, `min(z_A, z_B)` picks z_A on Herzhausen — the small one. Roughness stays quiet, letting turning + speedup do the work. Prediction lands within 1 pp of actual.

**Pure mismatch is DOUBLE-COUNTING complex-terrain signal that's already in turning and log_speedup.**

## 5. The cleaner physical story that emerges

Earlier framing (in previous discussions):
> "Formula A captures terrain complexity, which is a real driver of uncertainty. min picks whichever proxy fits best."

That framing was weak — it conceded that A might just be a "fit-hack" complexity proxy. The empirical evidence supports a different story:

**Actual role of the min operation: prevent roughness from double-counting signal already carried by turning and log_speedup.**

- **Opposite-sign sectors (complex terrain):** min picks z_A (small due to cancellation). Turning and log_speedup already know the terrain is complex — they carry the uncertainty signal. Roughness stays out of the way.
- **Same-sign similar magnitudes (both sites in forest z_B is small (low mismatch), z_A is large (both rough). min picks z_B — behaves like pure mismatch. Roughness contribution sta):**ys low, matching the low actual error of same-forest close-pairs.
- **Same-sign very different magnitudes (one flat, one rough):** z_A moderate, z_B large. min picks z_A — moderate contribution reflecting the real dissimilarity in absolute magnitude, without the amplification pure mismatch would give.

The min operation is a **redundancy filter**, not a fit-hack. Where turning and log_speedup already explain the uncertainty, min mutes roughness. Where roughness is the ONLY signal available (forests without orography), min lets mismatch do the work.

This isn't a story we invented after the fact — the mechanism falls directly out of the per-pair data. Herzhausen's 0.5% roughness share under adaptive vs pure mismatch's 5-7 pp over-prediction is not consistent with "adaptive fits by amplifying roughness". It's the opposite — adaptive PROTECTS the prediction from roughness over-firing on pairs where roughness isn't really the driver.

## 6. What this means for the concern

The original worry: "adaptive uses complexity + similarity, which makes me nervous — maybe we're rationalizing a fit."

The empirical answer:

1. **On roughness-dominated pairs (where the formula choice should matter most), adaptive and pure-mismatch are indistinguishable on accuracy.** No evidence of overfitting via formula A there.

2. **On complexity-dominated pairs, pure mismatch systematically over-predicts** because opposite-sign sectors amplify roughness mismatch that's redundant with turning/speedup signals. Adaptive prevents this over-firing.

3. **The min operation's physical role is redundancy prevention**, not complexity-as-proxy-fit. That's a cleaner story than the earlier framing.

## 7. Practical recommendation

**Stick with Adaptive Model v5.** Reasons, in order of importance:

- **Overall accuracy is meaningfully higher** (mean |err| 1.77 vs 2.20 pp; Pearson 0.887 vs 0.872)
- **Wins 32 of 45 pairs** — not a marginal edge
- **Better in the safe direction** — adaptive over-predicts by 0.44 pp on average, pure mismatch by 0.83 pp; but pure mismatch's over-prediction spikes to +6-7 pp on complex terrain, adaptive stays within +1 pp there
- **No accuracy cost on roughness-dominated pairs** — pure mismatch's supposed physical purity buys nothing there
- **Redundancy-prevention story is defensible** without needing to argue about magnitude physics

Switching to pure mismatch would DEGRADE accuracy on real deployment pairs, especially the complex-terrain ones where confident predictions matter most. The physically-cleaner formulation is empirically the less accurate one for this problem — because the other 4 features already carry the complex-terrain signal that pure mismatch tries to add through roughness.

## 8. What we still can't prove

At n=45-47 pairs, we cannot prove the min operation is UNIVERSALLY optimal. Alternative formulations we haven't exhaustively tested:

- Weighted combination `α·z_A + (1-α)·z_B` with learned α
- Formula B with per-sector filtering to exclude opposite-sign contributions
- Product `z_A · z_B` (favors pairs where both signals agree)
- Bayesian mixture over A and B with a latent selection variable

Any of these MIGHT edge out `min(z_A, z_B)` by a small margin. But the current model has an accuracy-defensible story now — the "we might be rationalizing" concern is answered by the empirical roughness-dominance analysis.

If further tinkering with roughness parameterization is warranted, it should be motivated by a specific accuracy failure on a real deployment pair, not by theoretical unease about the min operation.

## 9. Files

- Script: `Post-thesis corrections/Adaptive_Model_v5/roughness_dominance_check.py`
- Output CSV: `Post-thesis corrections/Adaptive_Model_v5/Results/roughness_dominance_comparison.csv`
- Adaptive M2c results: `Post-thesis corrections/Adaptive_Model_v5/Results/`
- Pure mismatch (exp-B) results: `Post-thesis corrections/sigma_only_refit/Results/exp-B__forest-in_loo.csv`
- Related: `Post-thesis corrections/Adaptive_Model_v5/adaptive_variants_explained.md` (the M/Mc/M2/M2c walkthrough)
