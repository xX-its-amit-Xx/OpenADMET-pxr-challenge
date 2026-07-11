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

## 4. Retrospective test of our "negative" methods on the 260
*🔬 In progress — did agentic reasoning / physics / substructure / water actually help on the truly-blind set?*

## 5. Approaches we didn't finish in time (now built on GPU)
*🔬 In progress — CheMeleon deep-ensemble, full-train Boltz cofold + interaction head, TabPFN, explicit-water GNN. Running on Kaggle.*

---

*Note: the local working tree was partially lost to disk-pressure cleanup during this analysis; recovered from GitHub + HuggingFace. All computed predictions survived on the C: cache.*
