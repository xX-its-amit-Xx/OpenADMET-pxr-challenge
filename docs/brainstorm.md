# Notebook 11 — Brainstorm

Ideas for the next modelling step. Captured 2026-05-10 after notebooks 01–10.

---

## Context at time of writing

| Model | 5-fold OOF RAE | Note |
|---|---|---|
| LGBM_aug (07) | 0.5582 | best LGBM |
| Chemprop 2-task (08) | 0.5736 | proper 5-fold |
| Chemprop 11-task (10) | *pending* | + ChEMBL NR + SC + props |
| Best ensemble (08) | ~0.55 | inverse-RAE blend |

Phase 1 closes **2026-05-25** (15 days). Phase 2 starts with ~250 new labels from Analog Set 1 on 2026-05-26.

---

## Path A — Stacked meta-learner *(recommended for nb 11)*

**What**: Collect OOF predictions from every model (LGBM_aug, LGBM_pipeline, Chemprop-08, Chemprop-10, kNN, MMP delta) as a feature matrix. Train a shallow meta-learner — ridge regression or a small LGBM — on those OOF columns to predict true pEC50.

**Why**: Each model has different blind spots. LGBM is good globally; kNN is good for close analogs (test has many); Chemprop captures scaffold-level patterns. A meta-learner can learn which model to trust per region of chemical space. Classical stacking consistently beats any single model or fixed-weight blend in competition settings.

**Implementation sketch**:
```python
# shape: (n_train, n_models)
oof_matrix = np.column_stack([oof_lgbm, oof_lgbm_pipe, oof_chemprop08,
                               oof_chemprop10, oof_knn, oof_mmp_delta])
meta = RidgeCV(alphas=np.logspace(-3, 3, 50))
meta.fit(oof_matrix, y_train)
# test: average model test preds then stack
test_matrix = np.column_stack([lgbm_te, lgbm_pipe_te, cp08_te, cp10_te, knn_te, mmp_te])
final = meta.predict(test_matrix)
```

**Risk**: Meta-training set is small (3,781 rows after SE filter). Use strong regularisation and nested scaffold CV to validate.

**Effort**: 1–2 days.

---

## Path B — Per-compound adaptive blending via similarity

**What**: Instead of a global blend weight (50% Chemprop / 50% LGBM), compute a per-test-compound blend weight from top-1 Tanimoto similarity to training. When sim > 0.6 → upweight kNN and MMP. When sim < 0.4 (scaffold hop) → trust Chemprop more. Secondary signal: test difficulty score from notebook 01.

**Why**: The test set is deliberately heterogeneous. 330/513 compounds have a close training neighbor (sim > 0.5); ~183 don't. A global weight treats them identically. The test difficulty parquet already has all the signals needed.

**Implementation sketch**:
```python
sim = top1_tanimoto  # shape (513,)
# Sigmoid transition centred at 0.5
w_knn = 0.4 / (1 + np.exp(-10 * (sim - 0.5)))   # 0 → 0, 1 → 0.4
w_cp  = 0.5 * (1 - sim / sim.max())               # lower sim → more Chemprop
final = w_knn * knn_te + w_cp * cp_te + (1 - w_knn - w_cp) * lgbm_te
```

**Risk**: Transition hyperparameters (centre, steepness) can only be tuned on training CV, which may not transfer to test. Risk of overtuning.

**Effort**: 1 day. Can be combined with Path A (use similarity as a meta-feature).

---

## Path C — Pretrained molecular foundation model embeddings

**What**: Replace Morgan + RDKit features with embeddings from a pretrained transformer. Options:
- **ChemBERTa-2** (seyonec/ChemBERTa-zinc-base-v1, 77M params) — SMILES-level BERT, easy via `transformers`
- **GROVER** (self-supervised on 10M drug-like molecules, graph-level) — stronger for scaffold generalisation
- **MolBERT** — pre-trained on 1.6B SMILES

Fine-tune only the top layer or last 2 layers; freeze backbone. Or use frozen embeddings as LGBM features.

**Why**: These models have seen orders of magnitude more chemical space than our 3,781 training compounds. The pretrained representations may encode structural nuance (H-bond geometry, ring strain, matched-pair relationships) that ECFP4 and RDKit descriptors miss.

**Risk**: Pretrained on generic drug-like diversity, not nuclear receptor binding. Fine-tuning on 3,781 examples risks overfitting. CPU-only makes fine-tuning slow (~hours per epoch for ChemBERTa). High-variance option: could move the needle by 0.05+ or do nothing.

**Effort**: 3–5 days.

**Best for**: Phase 2 (after Analog Set 1 unblinding), when there are more labelled examples to fine-tune on and the model can be validated against newly revealed labels.

---

## Other ideas (lower priority)

### Rank ensemble
Convert each model's predictions to percentile ranks, average, convert back. More robust to systematic per-model biases (e.g. Chemprop overestimates high-activity compounds). Half a day.

### Multi-cut MMP
Notebook 04 used single-cut fragmentation. Adding double-cut (two R-groups) would expand transform coverage from 66% to ~80% of test compounds. Moderate effort.

### Bayesian / MC-Dropout uncertainty
Use MC Dropout on Chemprop to get per-compound predictive uncertainty. Use uncertainty as a weight: high uncertainty → trust kNN more. Aligns with adaptive blending (Path B) and is a natural extension of the existing Chemprop model.

### Phase 2 fast-refit pipeline
Design a script that takes the 250 newly unblinded labels on 2026-05-26, retrains all models within 2–3 hours, and produces an updated submission. Key: keep all hyperparameters frozen, only refit with new data appended.
