# OpenADMET PXR Blind Challenge — Activity Track

**Team:** Amit Shenoy (Northeastern University)
**Track:** Activity prediction — pEC50 regression on 513 PXR analogs
**Primary metric:** RAE (Relative Absolute Error); lower is better; mean predictor = ~1.0

Submission for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge). Full details and data: [`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).

---

## Method Overview

A **stacked ensemble of 9 diverse molecular representation models**, meta-learned with scaffold-aware nested cross-validation to prevent analog leakage. The final prediction blends the ElasticNet stack with a separately-trained Chemprop multitask GNN.

### Key Design Choices

- **Scaffold CV throughout**: `scaffold_kfold_indices` keeps all analogs of a scaffold in the same fold — prevents the ~0.1 RAE optimism from random splits on this analog-expansion test set
- **Diverse inductive biases**: Morgan fingerprints, 3D conformer geometry (Uni-Mol), graph topology (GROVER), and SMILES-sequence pretraining (ChemBERTa) cover structurally different signal types
- **ElasticNet stacking**: L1 automatically zeros redundant models; L2 handles collinearity
- **Honest nested CV**: meta-learner is always fit on inner folds, never sees the outer validation fold

---

## Results

**Best submission:** `submissions/25_grand_v5.csv`

| Model | OOF RAE |
|---|---|
| Mean predictor baseline | ~1.0 |
| LGBM baseline (Morgan + RDKit) | 0.5600 |
| LGBM tuned (Optuna) | 0.5394 |
| Chemprop multitask GNN | 0.5170 |
| **Grand Ensemble v5 (nested CV)** | **0.5356** |

---

## Ensemble Architecture

### Base Models

| # | Model | Features / Architecture | OOF RAE |
|---|---|---|---|
| 13 | ChemBERTa-MLM | `ChemBERTa-zinc-base-v1`, 768-dim CLS token | 0.6782 |
| 14 | ChemBERTa-MTR | `ChemBERTa-PubChem-base-v1`, 768-dim CLS token | 0.5993 |
| 16 | LGBM tuned | Morgan + RDKit, Optuna TPE 60-trial search | 0.5491 |
| 19 | Uni-Mol | 3D conformer-aware transformer, 512-dim | 0.7008 |
| 20 | BERT-SMILES | `unikei/bert-base-smiles`, 768-dim CLS token | 0.7150 |
| 21 | SELFormer | `HUBioDataLab/SELFormer`, SELFIES RoBERTa, 768-dim CLS | 0.7691 |
| 22 | GROVER-base | Graph transformer atom FP 1600-dim; pretrained on 10M molecules | 0.6355 |
| 22b | GROVER-large | Graph transformer atom FP 2400-dim | 0.6295 |

### Ensemble Progression

| Version | Notebook | New model added | Nested CV RAE |
|---|---|---|---|
| Grand v2 | nb18 | lgbm_tuned (Optuna) replaces lgbm_aug | 0.5363 |
| Grand v3 | nb23 | v2 + Uni-Mol | 0.5360 |
| Grand v4 | nb24 | v3 + GROVER-base | 0.5358 |
| **Grand v5** | **nb25** | **v4 + GROVER-large** | **0.5356** |

### Final Submission Weights (Grand v5, full-data ElasticNetCV, α=0.00329, l1=0.50)

| Model | Weight |
|---|---|
| LGBM tuned | 74.7% |
| k-NN | 15.6% |
| Uni-Mol | 9.1% |
| ChemBERTa-MLM | 6.3% |
| GROVER-large | 5.9% |
| GROVER-base | 3.4% |
| ChemBERTa-MTR | 2.9% |
| LGBM base | 0% *(zeroed by L1)* |
| BERT-SMILES | 0% *(zeroed by L1)* |

The Grand v5 stack is blended with the standalone Chemprop multitask model (nb08) at 50.9% / 49.1% inverse-RAE weighting for the final submission.

---

## Notebook Progression

Run in numbered order. Each notebook saves OOF arrays to `data/processed/` and predictions to `submissions/`.

### 01 — EDA
Question-driven exploratory analysis anchored to modeling decisions. Key findings:
- pEC50 range: 1.61–7.55; median noise floor (SE) 0.24 log-units
- Hit rate (pEC50 ≥ 6): 1.6%; test/train InChIKey overlap: 0 (no leakage)
- Test top-1 Tanimoto to train: median 0.52 — test is close analogs, not scaffold hops
- 10 activity cliff pairs in train (Tanimoto ≥ 0.7, |ΔpEC50| ≥ 1.0)

Outputs: `compounds.parquet`, `cliffs_train.parquet`, `test_difficulty.parquet`

### 02 — LGBM Baseline
LightGBM on combined Morgan FP + RDKit descriptor features (2,265-dim). Combined features beat Morgan-only (0.658) and RDKit-only (0.594). **Scaffold 5-fold CV RAE: 0.575.**

Output: `submissions/02_lgbm_baseline.csv`

### 03 — Chemprop Multitask GNN
Chemprop 2.x MPNN, dual heads: PXR pEC50 + counter-assay pEC50 (NaN-masked loss on null head). Architecture: BondMessagePassing (depth=3, d_h=300) + MeanAgg + FFN (2 layers, dropout=0.1). **Single-fold holdout RAE: 0.517.**

Output: `submissions/03_chemprop_multitask.csv`

### 04 — Matched Molecular Pair Cliff Corrector
rdMMPA single-cut → 1,516 transforms; 377 are activity cliffs. Adaptive blend weight = min(0.60, 0.60 × n_analogs / 5). Coverage: 340/513 test compounds.

Output: `submissions/04_mmp_corrected.csv`

### 05 — Tanimoto k-NN
Similarity-weighted k=5 NN on ECFP4. **OOF RAE: 0.734** (CV underestimates; test compounds have real neighbors). Used in ensemble for its complementary signal in high-similarity regions.

Output: `submissions/05_knn_blend.csv`

### 06 — Counter-Assay as Input Feature
pEC50_null used as LGBM input feature; imputed for the 36% without real null values. Δ RAE = −0.004 over baseline.

Output: `submissions/06_counter_lgbm.csv`

### 07 — OOF Grid-Search Ensemble
OOF-based grid search for optimal kNN blend weight (5%); combined with Chemprop at 52% inverse-RAE weight.

Outputs: `submissions/07_final_ensemble.csv`, `submissions/07b_chemprop_weighted.csv`

### 08 — Chemprop Proper 5-Fold CV
Proper scaffold 5-fold CV for Chemprop (nb03 reported an easy single fold). **OOF RAE: 0.574.** This is the Chemprop component used in all downstream blends.

Output: `submissions/08_chemprop_cv_blend.csv`

### 09 — Robust Data Preprocessing Pipeline
Label outlier removal (SE > 0.5; drops 358 compounds), RDKit descriptor winsorising, near-zero-variance / high-correlation feature removal, active upsampling, pseudo-inactive augmentation from single-conc screen.

Output: `submissions/09_lgbm_pipeline.csv`

### 10 — External Data + Expanded Multitask Chemprop
ChEMBL + PubChem bioactivity for PXR and related nuclear receptors (CAR, VDR, FXR, PPARγ). 11-task Chemprop.

Output: `submissions/10_expanded_multitask.csv`

### 11 — Stacked Meta-Learner Ensemble
RidgeCV meta-learner on OOF predictions from LGBM_base, LGBM_aug, and k-NN. Saves OOF arrays for downstream stacking.

Output: `submissions/11_stacked_ensemble.csv`

### 12 — Per-Compound Adaptive Blending
Per-compound blend weights via sigmoid of top-1 Tanimoto similarity: kNN upweighted for close analogs, Chemprop for scaffold hops.

Output: `submissions/12_adaptive_blend.csv`

### 13 — ChemBERTa-MLM Embeddings
Frozen CLS-token embeddings (768-dim) from `seyonec/ChemBERTa-zinc-base-v1` + LGBM. **OOF RAE: 0.6782.** Embeddings cached to `data/processed/chemberta_{train,test}_emb.npy`.

Output: `submissions/13_chemberta.csv`

### 14 — ChemBERTa-MTR Embeddings
Same frozen-embedding strategy using `seyonec/ChemBERTa-PubChem-base-v1` (multitask regression pretraining). **OOF RAE: 0.5993.** Embeddings cached to `data/processed/chemberta_mtr_{train,test}_emb.npy`.

Output: `submissions/14_chemberta_mtr.csv`

### 15 — Grand Ensemble v1
First ElasticNet stack: lgbm_aug + knn + chemberta_mlm + chemberta_mtr + chemprop. **Nested CV RAE: 0.5473.**

Output: `submissions/15_grand_ensemble.csv`

### 16 — LGBM Optuna Tuning
Bayesian hyperparameter search (TPE sampler, 60 trials) over n_estimators, num_leaves, learning_rate, subsample, reg_alpha/lambda, min_child_samples. **OOF RAE: 0.5394** (vs 0.5600 baseline).

Output: `data/processed/oof_lgbm_tuned.npy`

### 17 — Rank Ensemble Variants
Rank-averaging over 5–7 model predictions:
- 17a Trio: grand_15 + chemprop_08 + chemberta_13
- 17b Quintet: + chemberta_mtr_14 + meta_11
- 17c Septet: all 7 models

Outputs: `submissions/17a_rank_trio.csv`, `17b_rank_quintet.csv`, `17c_rank_septet.csv`

### 18 — Grand Ensemble v2
Swaps lgbm_aug for lgbm_tuned (Optuna) in 5-model ElasticNet. **Nested CV RAE: 0.5363** (+0.011 over grand_v1). Weights: lgbm_tuned 79.6%, knn 18%, chemberta_mlm 8.3%, chemberta_mtr 7.4%.

Output: `submissions/18_grand_v2.csv`

### 19 — Uni-Mol 3D Embeddings
3D conformer-aware transformer (`unimol_tools.UniMolRepr`), 512-dim CLS. **OOF RAE: 0.7008**; Spearman vs lgbm_tuned = 0.68 (genuinely diverse signal from 3D geometry). Embeddings cached to `data/processed/unimol_{train,test}_emb.npy`.

Output: `submissions/19_unimol.csv`

### 20 — BERT-SMILES Embeddings
`unikei/bert-base-smiles` (BERT pretrained on 1.37M ChEMBL SMILES), 768-dim CLS + LGBM. **OOF RAE: 0.7150.** Zeroed by ElasticNet in all ensembles — insufficient diversity over lgbm_tuned.

Output: `submissions/20_bert_smiles.csv`

### 21 — SELFormer Embeddings
`HUBioDataLab/SELFormer` (RoBERTa on SELFIES strings), 768-dim CLS + LGBM. **OOF RAE: 0.7691.** Excluded from ensembles — marginally worse than BERT-SMILES.

Output: `submissions/21_selformer.csv`

### 22 — GROVER-Base Fingerprints
Graph transformer (tencent-ailab/grover) pretrained on 10M molecules. Atom-level fingerprint, 1600-dim. **OOF RAE: 0.6355**; Spearman vs lgbm_tuned = 0.82. Graph-based inductive bias distinct from SMILES-sequence models. Embeddings cached to `data/processed/grover_{train,test}_emb.npy`.

Output: `submissions/22_grover.csv`

### 22b — GROVER-Large Fingerprints
GROVER-large variant (428MB weights), 2400-dim atom fingerprint. **OOF RAE: 0.6295** (better than base). Spearman(base, large) = 0.903 — not identical, both contribute to ensemble. Embeddings cached to `data/processed/grover_large_{train,test}_emb.npy`.

Output: `submissions/22b_grover_large.csv`

### 23 — Grand Ensemble v3
7-model ElasticNet (v2 + Uni-Mol + BERT-SMILES). **Nested CV RAE: 0.5360**. Uni-Mol gets ~10% weight; BERT-SMILES zeroed.

Output: `submissions/23_grand_v3.csv`

### 24 — Grand Ensemble v4
8-model ElasticNet (v3 + GROVER-base). **Nested CV RAE: 0.5358**. GROVER-base gets 7.1% weight.

Output: `submissions/24_grand_v4.csv`

### 25 — Grand Ensemble v5 *(primary submission)*
9-model ElasticNet (v4 + GROVER-large). **Nested CV RAE: 0.5356.** Both GROVER models retained; lgbm_base and BERT-SMILES zeroed. Blended 49.1/50.9% with Chemprop-08.

Output: `submissions/25_grand_v5.csv`

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
| `pxr-challenge_single_concentration_TRAIN.csv` | 21,003 | Single-point screen |
| `pxr-challenge_structure_TEST_BLINDED.csv` | 184 | Structure track |

---

## Environment

```bash
# Activate
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Unix

# Install / sync deps
uv sync

# Jupyter kernel (once)
python -m ipykernel install --user --name pxr-challenge --display-name "pxr-challenge"
```

Python 3.11–3.12. PyTorch CPU-only by default (see `pyproject.toml`). Switch to GPU: change torch index URL to `https://download.pytorch.org/whl/cu124`.

---

## Source Library (`src/pxr/`)

| Module | Key functions |
|---|---|
| `data.py` | `load_train`, `load_test`, `load_counter`, `load_single_conc` |
| `chem.py` | `standardize`, `morgan_fp_batch`, `bemis_murcko`, `compute_physchem` |
| `eval.py` | `rae`, `compute_metrics`, `scaffold_kfold_indices` |
| `featurize.py` | `rdkit_desc`, `morgan`, `combined`, `impute` |
| `paths.py` | Centralised path constants |

---

## Phase Timeline

| Date | Event |
|---|---|
| 2026-05-25 | Phase 1 close — submit best ensemble |
| 2026-05-26 | Analog Set 1 unblinded (~250 new labels) |
| 2026-07-01 | Final deadline |
