# Notebooks 11–14 — Design Decisions & Paths Forward

Ideas captured 2026-05-10 after notebooks 01–10. All three paths have been implemented as notebooks 11–14.

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

## Path A — Stacked meta-learner → **Notebook 11**

**What**: Collect OOF predictions from every model (LGBM_base, LGBM_aug, kNN) as a feature matrix. Train a RidgeCV meta-learner on those OOF columns to predict true pEC50. Optionally include LGBM_pipeline (nb 09) if its OOF array is available.

**Why**: Each model has different blind spots. LGBM is good globally; kNN is good for close analogs; Chemprop captures scaffold-level patterns. A meta-learner can learn which model to trust per region of chemical space. Classical stacking consistently beats fixed-weight blends.

**Implementation**: `RidgeCV(alphas=np.logspace(-3, 3, 100))` trained on OOF matrix; honest estimate via nested scaffold CV. Final Chemprop blend via inverse-RAE weighting.

**Risk**: Small meta-training set (3,781–4,139 rows). Strong regularisation (RidgeCV) mitigates this.

**Status**: Implemented in `notebooks/11_stacked_ensemble.ipynb`.

---

## Path B — Per-compound adaptive blending → **Notebook 12**

**What**: Instead of a global blend weight, compute per-test-compound weights from top-1 Tanimoto similarity. When sim > 0.6 → upweight kNN. When sim < 0.4 (scaffold hop) → upweight Chemprop. LGBM gets the remainder, clipped to [0.15, 0.65].

**Why**: The test set is deliberately heterogeneous. 330/513 compounds have sim > 0.5; ~183 don't. A global weight treats them identically. Sigmoid transitions are smooth and differentiable.

**Implementation**: sigmoid weights, re-normalised to sum to 1 per compound. Optionally modulated by test difficulty score from nb 01.

**Risk**: Transition hyperparameters (centre=0.55, slope=12) can only be tuned on training CV, which may not transfer. Bug in original walrus-operator expression fixed before running.

**Status**: Implemented in `notebooks/12_adaptive_blend.ipynb`.

---

## Path C — Pretrained molecular transformer embeddings → **Notebooks 13 & 14**

**What**: Replace Morgan + RDKit features with embeddings from pretrained transformers. Two models compared:

| Notebook | Model | Pretraining corpus | Embedding dim |
|---|---|---|---|
| 13 | ChemBERTa-2 (`seyonec/ChemBERTa-zinc-base-v1`) | 77M ZINC SMILES | 768 |
| 14 | MolFormer-XL (`ibm/MolFormer-XL-both-10pct`) | 1.1B ZINC + PubChem SMILES | 768 |

Both use frozen CLS-token embeddings (no fine-tuning) concatenated with Morgan FP as LGBM features. Fine-tuning on 3,781 examples would likely overfit on CPU.

**Why**: These models have seen orders of magnitude more chemical space than our training set. The pretrained representations may encode structural nuance that ECFP4 and RDKit descriptors miss.

**Risk**: Pretrained on generic drug-like diversity, not nuclear receptor binding. High-variance option — could move the needle by 0.05+ or do nothing.

**Status**: Implemented in `notebooks/13_chemberta.ipynb` and `notebooks/14_molformer.ipynb`.

---

## Next paths forward (after notebooks 11–14 results)

These are candidate directions for notebooks 15+, to be evaluated once results from 11–14 are available.

### D — Grand ensemble of all OOF predictions
Pool OOF arrays from notebooks 07, 09, 11, 13, 14 (+ Chemprop) into a single RidgeCV or ElasticNet meta-learner. If pretrained embeddings help even marginally, combining them with structural models may give further gains. ~0.5 day.

### E — Phase 2 fast-refit pipeline
Design a single script that takes ~250 newly unblinded labels from Analog Set 1 (2026-05-26), appends to training data, reruns the best pipeline end-to-end within 2–3 hours, and produces a new submission. Freeze hyperparameters — only refit. Critical for Phase 2 competitiveness.

### F — MolFormer / ChemBERTa fine-tuning with LoRA
After Phase 2 data arrives (~250 new labels → ~4,000 total), fine-tune the last 2 transformer layers using LoRA (low-rank adapters) to reduce parameters. Scaffold holdout on the Phase 2 set for validation. High compute cost on CPU; worth running overnight or switching to GPU index.

### G — Matched molecular pair delta-learning with transformer features
Use MMP deltas (|ΔpEC50|) from notebook 04 as a secondary supervision signal. Transformer embeddings of the core scaffold and the two R-groups could be combined with the MMP delta label to train a specialised cliff-predictor head. Combine with base model via blending.

### H — Rank ensemble for systematic bias correction
Convert all models' predictions to percentile ranks, average, convert back. More robust to per-model systematic biases (Chemprop over-predicts high-activity compounds). Half a day; easy win if bias diagnostics show skew.
