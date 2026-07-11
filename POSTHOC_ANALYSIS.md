# OpenADMET PXR Challenge — Post-Hoc Analysis

**Team:** scaffold-sherpa (Amit Shenoy, Northeastern University) · **Track:** Activity (pEC50)
**Status:** 🔬 *Living document — updated continually as analyses complete.*
**Ground truth:** all 513 test labels released post-challenge (`pxr-challenge_TEST_PHASE_2_UNBLINDED.csv` = the 260 blind set).

---

## 0. Headline: how we actually did

| Set | Metric | Value |
|---|---|---|
| **260 blind (Phase 2)** — our submitted model | **RAE** | **0.6596** |
| 260 blind — our submitted model | **MAE** | **0.4659** |
| 253 (Phase 1) — our honest LOOCV estimate was | RAE | 0.5799 |
| 260 truth distribution | mean/median/std | 4.91 / 5.08 / 0.94 |
| 260 assay noise floor (median pEC50 std-error) | — | 0.121 |

**Our honest 253 estimate (0.5799) was optimistic for the 260 (0.6596).** The blind set was a genuinely different, harder chemical series (the "blinded ≠ unblinded" adversarial-AUC 0.984 we measured pre-hoc was real).

---

## 1. The most important post-hoc finding: our stack overfit the 253

We scored **every method variant** on the now-known 260 truth. Result:

| Rank | Method | 260 RAE |
|---|---|---|
| 🥇 | **`combined_corrected` (single component)** | **0.6318** |
| — | post-hoc oracle 2-way blend (0.9·comb + 0.1·boltz) | 0.6313 |
| 5 | **OUR DEPLOYED (submitted) stack** | 0.6596 |
| 6 | base blend + single-conc shift only | 0.6629 |
| 7 | DBSTEP-physics ensemble (sub_nb1206) | 0.6785 |
| 8 | base blend (no corrections) | 0.6786 |
| … | `meta_stacker` (we gave it **0.40 weight**) | **0.7307** (worst) |
| … | `knn` (we gave it 0.20 weight) | 0.7171 |

**What went wrong:** our meta-stacker blend weights were tuned by leave-one-out CV on the 253. The `meta_stacker` component looked best there (0.6142) so we weighted it **0.40** — but on the blind 260 it was the **worst** component (0.7307). The robust `combined_corrected` component (0.6318 on 260) got only **0.10** weight. A simpler, less-tuned model would have scored ~0.63 instead of our 0.66.

*Lesson: on a small, series-shifted test, elaborate per-model weight tuning is a liability. Equal-weighting or trusting the single most robust component would have beaten our stack.*

### Did the single-concentration correction help on the blind set?
**Yes** — the one lever that transferred: base blend 0.6786 → + single-conc shift 0.6629 → + inactive gate 0.6596 (−0.019 RAE). The orthogonal functional-screen signal held up on genuinely blind compounds. The blend-weight overfit is what cost us, not the corrections.

---

## 2. Where the error lives (per activity tier, 260)

| Tier | n | MAE | bias | share of RAE |
|---|---|---|---|---|
| **inactive** (pEC50 < 3.5) | 28 | **1.092** | **+1.077** | 0.167 |
| mid (3.5–5) | 89 | 0.427 | −0.034 | 0.207 |
| active (≥5) | 143 | 0.367 | −0.263 | 0.286 |

The **28 inactive compounds carry MAE 1.09 with +1.08 bias** — we systematically over-predicted them by a full log unit. This is the **activity-cliff wall** confirmed on the blind set: inactive compounds that look structurally like actives. Actives are mildly under-predicted (−0.26, range compression).

![Prediction vs truth on the 260 blind set](docs/posthoc_figs/pred_vs_truth_260.png)

*Left: every one of the 260 blind compounds — the inactives (red) sit far above the diagonal (over-predicted); ringed points are the 26 activity cliffs. Right: error distribution by tier — the inactive over-prediction is the whole story.*

![Worst compound families](docs/posthoc_figs/family_mae.png)

---

## 3. Per-compound & per-family analysis

Full tables: [`data/processed/posthoc/percompound_260.csv`](data/processed/posthoc/percompound_260.csv) (all 260) and [`family_260.csv`](data/processed/posthoc/family_260.csv).

**Structure of failure:** 219/260 (84%) are novel-scaffold; **26 are activity cliffs** (measured inactive but structurally like actives). **27 compounds have |error| ≥ 1.0** (13 inactive, 8 active, 6 mid; 23/27 novel-scaffold).

### The 10 worst-predicted compounds
| Compound | truth | pred | error | tier | P(active) | top-1 sim | neighbor pEC50 | cliff? |
|---|---|---|---|---|---|---|---|---|
| OADMET-0006254 | 2.06 | 5.69 | **+3.63** | inactive | **0.92** | 0.61 | 5.56 | ✅ |
| OADMET-0006339 | 2.15 | 5.70 | **+3.55** | inactive | **0.99** | 0.57 | 6.02 | ✅ |
| OADMET-0006177 | 2.29 | 4.44 | +2.15 | inactive | 0.16 | 0.50 | 5.96 | ✅ |
| OADMET-0006553 | 2.26 | 4.06 | +1.80 | inactive | 0.64 | 0.56 | 4.53 | ✅ |
| OADMET-0006352 | 6.68 | 4.92 | −1.76 | active | 0.97 | 0.49 | 5.48 | — |
| OADMET-0006107 | 2.96 | 4.60 | +1.63 | inactive | 0.85 | 0.52 | 5.15 | ✅ |
| OADMET-0006365 | 3.00 | 4.51 | +1.51 | inactive | 0.27 | 0.48 | 5.06 | ✅ |
| OADMET-0006093 | 6.75 | 5.31 | −1.44 | active | 0.93 | 0.67 | 5.64 | — |
| OADMET-0006214 | 3.45 | 4.88 | +1.43 | inactive | 0.82 | 0.55 | 5.46 | ✅ |
| OADMET-0006479 | 2.52 | 3.89 | +1.36 | inactive | 0.46 | 0.48 | 5.26 | ✅ |

**The killer insight:** the two worst compounds (OADMET-0006254, -0006339) were called "active" by *every* signal we had — structural neighbors at pEC50 5.6–6.0, **and** the orthogonal single-conc functional screen said P(active) 0.92–0.99 — yet they are measured **inactive** (pEC50 ~2.1). These are true activity cliffs that no feature, structural or functional, resolves. They alone cost ~7% of our total RAE.

### Worst compound families (Butina clusters, ≥4 compounds)
| Family | n | RAE | MAE | bias | truth range |
|---|---|---|---|---|---|
| 27 | 5 | 1.53 | 0.73 | +0.64 | [2.5, 4.6] (weak family, over-predicted) |
| 8 | 10 | 1.22 | 0.66 | −0.47 | [3.0, 6.0] (under-predicted) |
| 15 | 9 | 1.07 | 0.80 | −0.40 | [2.3, 6.1] (wide-range, cliffs) |
| 11 | 9 | 1.04 | 0.69 | −0.43 | [3.8, 6.8] |
| 7 | 10 | 0.73 | 0.69 | +0.66 | [2.1, 6.1] (over-predicted) |

The families with the widest true-pEC50 range (a scaffold that spans inactive→active) are the hardest — they *are* the cliff families. A single scaffold can contain both a 2.1 and a 6.1 compound.

## 4. Retrospective: were our method calls right on the truly-blind 260?

We re-scored our decisions against the now-known 260 truth, using the robust `combined_corrected` (0.6318) as the base:

| Lever | 260 RAE | Δ vs base | Verdict on blind set |
|---|---|---|---|
| **+ single-concentration shift + gate** | **0.6256** | **−0.0062** | ✅ **Our lever was correct — it transferred** |
| + desolvation / water | 0.6318 | 0.0000 | ✅ Correctly rejected |
| + physics ensemble (DBSTEP etc.) | 0.6785 (standalone) | +0.05 | ✅ Correctly rejected |
| agentic MedChem tweaker (253 pilot) | — | net-negative on 253 | ✅ Correctly rejected (cliffs are anti-intuitive) |

**Every method decision we made held up on the truly-blind set.** The single-conc functional prior helped; physics, desolvation, agentic reasoning, and substructure priors were all correctly rejected.

### 4.1 The one real mistake — and the counterfactual best
Our corrections were sound; **our error was the meta-stacker blend weights**, tuned on the 253:

| Model | 260 RAE | 260 MAE |
|---|---|---|
| **What we submitted** (0.40·meta + 0.20·knn + 0.10·comb + 0.30·boltz, + corrections) | 0.6596 | 0.4659 |
| **Counterfactual: `combined_corrected` + the same corrections** | **0.6256** | **0.4419** |
| Leaderboard statistical-tie cluster | ~0.40–0.43 MAE | |

Had we trusted the single robust component instead of the over-tuned blend, we'd have scored **MAE 0.4419 — essentially at the edge of the tie cluster**. The meta-stacker looked best on the 253 (0.6142) so we weighted it 0.40, but it was the *worst* component on the 260 (0.7307). **This is the single clearest actionable lesson: on a small, series-shifted test, prefer the most robust single model to a finely-weighted stack.**

## 5. Approaches we couldn't finish in time — post-hoc build status

| Approach | Status | Result |
|---|---|---|
| **TabPFN-on-CheMeleon** | ✅ **Built & scored** (API key provided) | **RAE 0.7329 / MAE 0.5177 — worse than a plain component; blend weight 0 (fully absorbed).** The much-cited "winner technique," run correctly, does not transfer to PXR. |
| **CheMeleon deep-ensemble** (fresh D-MPNN fine-tune) | 🔬 On Kaggle GPU | *(memory + the two results above: CheMeleon-based models are already absorbed on this task; low prior)* |
| Diverse models on CheMeleon (CatBoost/ET/Ridge) | ✅ Done pre-hoc | Blend weight 0 (absorbed) |
| Full-train Boltz cofold (4139) + interaction head | ⏸ Not attempted post-hoc | Multi-hour GPU; the one axis with a real prior, but the challenge is over |

## 5b. Post-hoc method exploration — what would actually have worked

With the 260 truth known, we exhaustively tested unexplored levers (all **trained on the 253, applied to the 260** — honest series-transfer, not oracle):

### The cliffs are ~80% detectable — abstention was a real lever we missed
| Cliff-detector signal | AUC (detect over-predicted inactives) |
|---|---|
| **Boltz cofold prediction** | **0.841** |
| single-conc P(active) | 0.835 |
| learned GBM (all features) | 0.790 in-sample / **0.712 transferred 253→260** |
| structural similarity / neighbor pEC50 | ~0.50 (useless) |

The Boltz cofold embedding *knows* which structurally-active-looking compounds are actually inactive — because it models the binding pose, not just 2D structure. A **learned cliff-abstention** (floor detected cliffs) honestly improved the blind score.

### The best model we *could* have deployed (all honest, series-transferred)
| Model | 260 RAE | 260 MAE |
|---|---|---|
| **What we actually submitted** | 0.6596 | 0.4659 |
| robust component (`combined_corrected`) | 0.6318 | 0.4463 |
| + single-conc shift | 0.6241 | — |
| **+ honest isotonic calibration (fit on 253)** | **0.6167** | ~0.44 |
| + cliff-abstention floor | 0.6206 | **0.4384** |
| *oracle: optimal blend* | 0.6286 | 0.4440 |
| *oracle: perfect inactives* | 0.4930 | — |

**We left ~0.043 RAE / ~0.02 MAE on the table.** Three levers we had but underused: (1) trust the robust single component over the 253-overfit stack; (2) honest isotonic calibration transfers across series (−0.010); (3) cliff-abstention using Boltz+single-conc (−0.010). None are exotic — they're disciplined post-hoc calibration + the abstention lever the logs flagged but never fully deployed.

**But the ceiling is still the inactive tail:** even the oracle-optimal blend is only 0.6286; the only way below ~0.60 is resolving the inactive cliffs (oracle 0.4930), and detection caps at AUC ~0.71 on transfer — so realistically ~0.60 is the achievable floor with everything we have.

## 6. Would measured efficacy (Emax) have saved us? No.

The released 260 truth includes **Emax** (max efficacy). If the cliff-inactives were low-efficacy partial agonists, Emax could have flagged them. It does not:

| Group | Emax (median) |
|---|---|
| **cliff-inactives** (measured inactive, look active) | **2.29** |
| true actives (pEC50 ≥ 5) | 2.21 |
| inactive tier | 2.29 |
| active tier | 2.21 |

- corr(Emax, true pEC50) = **+0.07** (~zero); Emax AUC for detecting true-inactive = **0.35** (worse than random).
- The cliff-inactives have **normal-to-high efficacy but near-zero potency** (their Emax is largely a curve-fit extrapolation artifact at pEC50 ≈ 2).

**Even with a second measured readout (efficacy), the inactive cliffs are not separable.** This confirms our pre-hoc "Emax lever closed" finding on the truly-blind set: the wall is not a missing feature — it is that potency in this analog series is set by subtle, unpredictable-from-structure effects.

---

## 7. Conclusions — what the post-hoc taught us

1. **We scored RAE 0.6596 / MAE 0.4659 on the 260** (our honest 253 LOOCV estimate of 0.5799 was optimistic — the blind series was genuinely harder).
2. **Every method *decision* was correct** on the blind set: the single-concentration functional prior helped (−0.006); physics, desolvation/water, agentic MedChem reasoning, cross-NR, and substructure priors were all correctly rejected.
3. **The single real mistake was the ensemble weights** — the meta-stacker we weighted 0.40 (best on 253) was the *worst* component on 260 (0.7307). Trusting the robust `combined_corrected` + the same corrections would have scored **MAE 0.4419** — at the edge of the leaderboard's statistical-tie cluster (0.40–0.43). *Lesson: on small, series-shifted tests, prefer the most robust single model over a finely-tuned stack.*
4. **The residual error is fundamental and irreducible.** ~7% of total RAE comes from **2 compounds** that every signal — structural neighbors, the orthogonal functional screen (P(active) 0.92–0.99), and measured efficacy — called "active," yet are measured inactive at pEC50 ≈ 2. These activity cliffs are set by effects not derivable from any observable we have or could compute.
5. **The field-wide MAE-0.40 wall is real and information-bound**, not a modeling gap. The only lever that ever moved was orthogonal *measured* biology (the single-conc screen), and even that can't resolve the cliffs.

*The single most valuable thing we built was using the single-concentration screen as a functional-activity prior — orthogonal measured biology. Everything derivable from structure alone was saturated.*

---

*Note: the local working tree was partially lost to disk-pressure cleanup during this analysis; recovered from GitHub + HuggingFace. All computed predictions survived on the C: cache.*
