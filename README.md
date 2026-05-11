# OpenADMET PXR Challenge

Submission repository for the [OpenADMET PXR Blind Challenge](https://openadmet.org) — activity prediction track (pEC50 regression on 513 test compounds).

**Primary metric**: RAE (Relative Absolute Error). RAE < 1.0 beats the mean predictor; lower is better.

---

## Environment

```bash
# activate
source .venv/bin/activate      # Unix
.venv\Scripts\activate         # Windows

# install / sync deps
uv sync

# register Jupyter kernel (once)
python -m ipykernel install --user --name pxr-challenge --display-name "pxr-challenge"
```

Python 3.11–3.12. PyTorch is CPU-only by default (see `pyproject.toml`). Switch to GPU by changing the torch index URL to `https://download.pytorch.org/whl/cu124`.

---

## Data

Raw data is gitignored. Re-clone if missing:

```bash
git clone https://huggingface.co/datasets/openadmet/pxr-challenge-train-test data/raw
git clone https://github.com/OpenADMET/PXR-Challenge-Tutorial.git tutorial
```

| File | Rows | Description |
|---|---|---|
| `pxr-challenge_TRAIN.csv` | 4,139 | CRC dose-response: pEC50, Emax, uncertainties |
| `pxr-challenge_TEST_BLINDED.csv` | 513 | Test set — SMILES only |
| `pxr-challenge_counter-assay_TRAIN.csv` | 2,859 | PXR-null counter-screen |
| `pxr-challenge_single_concentration_TRAIN.csv` | 21,003 | Single-point screen: log2FC, FDR |
| `pxr-challenge_structure_TEST_BLINDED.csv` | 184 | Structure track |

---

## Notebook Progression

Run in numbered order. Each notebook saves to `submissions/` and passes results forward.

### 01 — EDA
Question-driven exploratory analysis anchored to modeling decisions.

Key outputs: `data/processed/compounds.parquet`, `cliffs_train.parquet`, `test_difficulty.parquet`

Key numbers:
- Training pEC50 range: 1.61–7.55; mean 4.32; noise floor (median SE) 0.24
- Hit rate (pEC50 ≥ 6): 1.6%
- Test top-1 Tanimoto to train: median 0.52 — test compounds are close analogs
- Train/test InChIKey overlap: 0 (no leakage)
- Activity cliff pairs in train: 10 (Tanimoto ≥ 0.7, |ΔpEC50| ≥ 1.0)

---

### 02 — LGBM Baseline
LightGBM on combined Morgan FP + RDKit descriptor features.

| Feature set | Scaffold 5-fold CV RAE |
|---|---|
| Morgan only | 0.658 |
| RDKit desc only | 0.594 |
| Combined (2,265 feat.) | **0.575** |

Output: `submissions/02_lgbm_baseline.csv`

---

### 03 — Chemprop Multitask GNN
Chemprop MPNN with two output heads: PXR pEC50 + counter-assay pEC50 (NaN-masked loss). Architecture: BondMessagePassing (depth=3, d_h=300) + MeanAgg + FFN (2 layers, dropout=0.1).

| Model | Holdout RAE |
|---|---|
| Chemprop multitask (scaffold fold-0) | 0.517 |
| 50/50 ensemble with LGBM | — |

Note: 0.517 came from fold 0 only — see notebook 08 for proper 5-fold CV.

Outputs: `submissions/03_chemprop_multitask.csv`, `submissions/03_ensemble_lgbm_chemprop.csv`

---

### 04 — Matched Molecular Pair Cliff Corrector
rdMMPA single-cut fragmentation → 1,516 transform pairs; 377 are activity cliffs (|ΔpEC50| > 1.0). Adaptive blend weight = min(0.60, 0.60 × n_analogs / 5).

- Test coverage: 340/513 compounds have training analogs (66.3%)
- Blend improves cliff predictions without hurting global performance

Output: `submissions/04_mmp_corrected.csv`

---

### 05 — Tanimoto k-NN
Similarity-weighted k=20 nearest neighbours on ECFP4 (Tanimoto = Jaccard on bit vectors).

| k | Scaffold CV RAE |
|---|---|
| 20 | 0.740 (underestimate — CV forces scaffold gaps; test set has real neighbors) |

Test top-1 similarity: mean 0.532, median 0.523. 330/513 compounds have sim > 0.5.

Output: `submissions/05_knn_blend.csv` (40% kNN + 60% base ensemble)

---

### 06 — Counter-Assay as Input Feature
pEC50_null used as a LGBM input feature (not just a training target). Imputed via a null-predictor LGBM for the 36% of training compounds without a real null value.

| Model | OOF RAE |
|---|---|
| LGBM baseline | 0.5675 |
| LGBM + null feature | **0.5637** (Δ = −0.0038) |

Output: `submissions/06_counter_lgbm.csv`

---

### 07 — OOF Grid-Search Ensemble
Out-of-fold predictions for LGBM_aug and kNN; grid-search blend weight; then blend with Chemprop.

| Component | OOF RAE |
|---|---|
| LGBM_aug | 0.5582 |
| kNN | 0.7330 |
| Optimal kNN weight | 0.05 |

Chemprop blend weight fixed at 35% (subsequently corrected in 07b to 52% via inverse-RAE).

Outputs: `submissions/07_final_ensemble.csv`, `submissions/07b_chemprop_weighted.csv`

---

### 08 — Chemprop Proper 5-Fold CV
Notebook 03 reported a single-fold RAE of 0.517 (fold 0, seed=0 — optimistically easy fold). This notebook runs proper 5-fold scaffold CV with per-fold scaling to avoid data leakage.

| Fold | RAE |
|---|---|
| 0 | 0.493 |
| 1 | 0.622 |
| 2 | 0.582 |
| 3 | 0.574 |
| 4 | 0.621 |
| **OOF global** | **0.574** |

Chemprop and LGBM_aug are essentially tied at ~0.56–0.57. Inverse-RAE weights: 49% Chemprop / 51% LGBM_aug.

Output: `submissions/08_chemprop_cv_blend.csv`

---

### 09 — Robust Data Preprocessing Pipeline
Systematic feature and label cleaning before LGBM training.

| Step | Details |
|---|---|
| Label outlier removal | Drop SE > 0.5 → removes 358/4,139 compounds (8.6%) |
| Feature outlier clipping | Winsorise RDKit descriptors at Q1/Q3 ± 5×IQR |
| Feature selection | Remove near-zero-variance + >0.95-correlated descriptor pairs |
| Active upsampling | Random oversample actives/moderates to ~10% of training size |
| Pseudo-inactive augmentation | Single-conc high-confidence inactives (|log2FC| < 0.5, FDR > 0.3) assigned N(4.30, 0.24) pEC50 labels |

Saved artefacts: `data/processed/train_clean.parquet`, `data/processed/feature_selector.pkl`

Output: `submissions/09_lgbm_pipeline.csv`

---

### 10 — External Data + Expanded Multitask Chemprop
Pulls bioactivity data from ChEMBL and PubChem for PXR and related nuclear receptors; trains an 11-task Chemprop model.

| Task | Source | # compounds |
|---|---|---|
| pec50_pxr | Challenge train (cleaned) | 3,781 |
| pec50_null | Counter assay | 2,649 |
| log2fc_sp | Single-conc screen | 21,003 |
| chembl_pxr | ChEMBL CHEMBL3401 | ~3,000 |
| chembl_car | ChEMBL CHEMBL3509594 | ~500 |
| chembl_vdr | ChEMBL CHEMBL1977 | ~2,000 |
| chembl_fxr | ChEMBL CHEMBL2047 | ~2,000 |
| chembl_pparg | ChEMBL CHEMBL235 | ~10,000 |
| logP / MW / TPSA | RDKit (all compounds) | all |

Output: `submissions/10_expanded_multitask.csv`

---

## Key Numbers

| Fact | Value |
|---|---|
| Training compounds (raw) | 4,139 |
| Training compounds (SE-filtered) | 3,781 |
| Test compounds | 513 |
| Assay noise floor (median SE) | 0.24 log-units |
| Hit rate in training (pEC50 ≥ 6) | 1.6% |
| Test top-1 Tanimoto to train (median) | 0.52 |
| Activity cliff pairs | 10 |
| Counter-assay overlap with train | 2,649 / 4,139 |
| Mean predictor RAE | ~1.0 |
| Best single-model OOF RAE | ~0.558 (LGBM_aug) |

## Source Library (`src/pxr/`)

| Module | Purpose |
|---|---|
| `data.py` | Loaders: `load_train`, `load_test`, `load_counter`, `load_single_conc` |
| `chem.py` | `standardize`, `morgan_fp_batch`, `to_inchikey`, `bemis_murcko` |
| `eval.py` | `rae`, `compute_metrics`, `scaffold_kfold_indices`, `cv_score` |
| `featurize.py` | `rdkit_desc`, `morgan`, `combined`, `impute`, `FeatureSelector` |
| `preprocess.py` | `remove_label_outliers`, `clip_feature_outliers`, `FeatureSelector`, `upsample_by_category`, `pseudo_inactive_augment` |
| `external.py` | `fetch_chembl_target`, `fetch_all_nr_targets`, `fetch_pubchem_aid`, `standardize_external` |
| `paths.py` | Centralised path constants |

## Phase Timeline

| Date | Event |
|---|---|
| 2026-05-25 | Phase 1 close — submit best ensemble |
| 2026-05-26 | Analog Set 1 unblinded (~250 new labels) |
| 2026-07-01 | Final deadline |
