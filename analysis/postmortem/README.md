# Phase-1 Post-Mortem — OpenADMET PXR Activity Track

Phase 1 closed; **253 of the 513** blinded test compounds were unblinded with true pEC50.
We audited **343 models across 177 notebooks** against ground truth. This folder is the
six-notebook autopsy: what we built, how it failed, and what to do in Phase 2.

The metric is **RAE** (Relative Absolute Error): `sum|y-yhat| / sum|y-mean(y)|`. Mean
predictor = 1.0; lower is better. The honest deployable leaderboard band is **~0.55–0.65**.

## The verified headline picture (every notebook reproduces these)

| Fact | Value |
|---|---|
| Unblind truth | n=253, mean 4.66, std 1.03, range [1.75, 6.72]; bins low/mid/high = 37/174/42 |
| **F1 variance compression** | median model pred_std **0.62** vs truth_std **1.03** (~62% of range), median r ≈ 0.59 |
| **F2 novel-scaffold inactives** | consensus bias_low **+1.23** (MAE 1.28); 8 worst errors all novel dead cpds at nn_sim ~0.5 |
| **F3 rare-active under-recovery** | consensus bias_high **−0.54** (MAE 0.55); inactive tail ~2.4× worse than active tail |
| **F4 CV/stacking overfit** | corr(train-OOF, unblind) = **0.505** (Spearman ≈ 0.03); best-CV-61 median unblind **0.772** |
| Best single on truth | `oof_nb390_pcs_iso` (structure3D) **RAE 0.582** |
| Naive 71-model consensus | **0.648** — *worse* than best single (residuals r ≈ 0.8, no diversity) |
| Oracle / headroom | per-compound oracle ~0.12; forward-ensemble plateau **~0.56 at K≈5** (then overfits) |

## Notebook index

| Notebook | Title | One-line takeaway |
|---|---|---|
| **pm01_landscape.ipynb** | Landscape & headline diagnosis | The four failure modes laid out; CV→unblind collapse is the headline (Spearman ≈ 0.03), and the best-CV cohort is among the *worst* on truth. |
| **pm02_activity_axis.ipynb** | The activity axis | We miss **both** tails, but the inactive side is **2.3×** worse; the single mechanism is variance compression — predictions span only ~64% of the truth range and drain both tails into the populated centre. |
| **pm03_chemistry.ipynb** | Chemistry of the errors | Error is **chemically structured, not noise**: novel scaffolds carry 1.37× the error and are over-predicted; PXR's lipophilicity prior is confirmed (high-logP inactives over-predicted +1.59) — the model reads "greasy ⇒ active." |
| **pm04_taxonomy.ipynb** | Method-family taxonomy | No family escapes the 0.68–0.82 median band; the pathology is **shared, not family-specific**; family residuals correlate ~0.8 so ensembling the pool is near-exhausted (headroom <0.02 RAE). |
| **pm05_why_cv_lied.ipynb** | Why our CV lied | Scaffold-CV is measured on the train manifold; the test is an analog expansion onto a partly-novel rim CV cannot see — budget a **+0.10 optimism shift** on any train-only number; the stacking curve bottoms at **K≈5** then rises. |
| **pm06_phase2_prescriptions.ipynb** | **Synthesis & 5 Phase-2 prescriptions** | The capstone: failure→remedy matrix, the F2 prize (a perfect inactive detector is worth **−0.11 RAE**), and five mechanistically distinct prescriptions retro-tested live on the 253. |

## The five Phase-2 prescriptions (pm06), prioritised

1. **P4 — inject external scaffold-diverse INACTIVES** (ChEMBL NR1I2/PXR + Tox21 negatives) and retrain the base. Attacks F2 (the dominant mode). Proxy test: a detector that surfaces the dead compounds in the novel-scaffold gate captures up to **−0.11 RAE** — the only high-impact lever left, because it brings information the 343-model pool does not contain.
2. **P1 — OOD inactive-gate shrink** (`novel & nn_sim<0.6 & pred>4 ⇒ pred − α`). Honest cross-fit **−0.008**. Cheapest real win; ships as a post-hoc rule on the deploy model.
3. **P3 — similarity-conformal intervals + Tanimoto-kNN fallback + abstain-flag.** Risk-aware deployment (intervals widen as nn_sim drops). Not an automatic RAE win — the dominant F2 errors are *confident*, so no disagreement threshold cleans the core.
4. **P2 — scalar rank-stretch** on the final submission (single global slope, ~−0.003..−0.005 on compressed blends). Must be fit globally, **not** per-fold (per-fold overshoots the already-calibrated best single).
5. **P5 — counter-assay-gated selectivity down-weight** (high predicted PXR-null + high PXR ⇒ likely non-specific/greasy ⇒ shrink). Orthogonal *biological* axis, distinct from P1's chemical-novelty rule.

**Closed dead-ends (do not relitigate):** >5-component SLSQP/30-way stacks, naive consensus,
pure unblind augmentation (0.587), foundation-embedding blends on the pEC50 axis (0-weight).

## Substrate (read-only) — `data/processed/postmortem/`

`pm_compounds.parquet` (253 + truth + consensus + chem), `pm_pred_unblind.npy`/`pm_resid_unblind.npy`
(253×334), `pm_oof_train.npy` (4139×334), `pm_model_meta.parquet` (343), `pm_model_names.txt`
(334 matrix-column names), `pm_unblind_y.npy`, `pm_train_y.npy`, `pm_family_reps.csv`, `pm_meta.json`.
Join meta→matrix by name: `col = names.index(modelname)`. RAE: `from pxr.eval import rae`.

## Reproduce

```bash
.venv/Scripts/python.exe -m jupytext --to notebook analysis/postmortem/pm06_phase2_prescriptions.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
    analysis/postmortem/pm06_phase2_prescriptions.ipynb --ExecutePreprocessor.timeout=1200
```
