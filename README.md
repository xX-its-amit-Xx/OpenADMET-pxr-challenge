# OpenADMET PXR Blind Challenge — Activity Track

**Team:** Amit Shenoy (Northeastern University)
**Track:** Activity prediction — pEC50 regression on 513 PXR analogs
**Primary metric:** RAE (Relative Absolute Error); lower is better; mean predictor = ~1.0

Submission for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge). Full details and data: [`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).

---

## Method Overview

A **stacked ensemble of 16 diverse molecular representation models**, meta-learned with scaffold-aware nested cross-validation to prevent analog leakage. New models include a 6-head Chemprop auxiliary multitask GNN, multi-NR transfer LGBM with ESM-2/ProtBERT protein embeddings, and cross-attention compound×protein fusion architectures.

### Key Design Choices

- **Scaffold CV throughout**: `scaffold_kfold_indices` keeps all analogs of a scaffold in the same fold — prevents the ~0.1 RAE optimism from random splits on this analog-expansion test set
- **Diverse inductive biases**: Morgan fingerprints, 3D conformer geometry (Uni-Mol), graph topology (GROVER), and SMILES-sequence pretraining (ChemBERTa) cover structurally different signal types
- **ElasticNet stacking**: L1 automatically zeros redundant models; L2 handles collinearity
- **Honest nested CV**: meta-learner is always fit on inner folds, never sees the outer validation fold

---

## Results

**Best submission:** `submissions/36b_grand_v6_no_aux.csv` *(Grand Ensemble v6b — 16 models, OOF RAE 0.5281)*

| Model | OOF RAE |
|---|---|
| Mean predictor baseline | ~1.0 |
| LGBM baseline (Morgan + RDKit) | 0.5600 |
| LGBM tuned (Optuna) | 0.5394 |
| Chemprop multitask GNN | 0.5170 |
| Grand Ensemble v5 (nested CV) | 0.5356 |
| **Grand Ensemble v6b (protein-aware, 16 models)** | **0.5281** |

### New Model Results (nb26–nb36)

| # | Model | OOF RAE | Notes |
|---|---|---|---|
| 26 | Single-conc pseudo-labels LGBM | 0.6003 | Pseudo-labels hurt (noisy; log2FC→pEC50 r=0.52) |
| 27 | NR-weighted LGBM | 0.5964 | Phylogenetic NR transfer marginal |
| 28 | Auxiliary features LGBM | 0.2179 | PXR lig sim + LOO k-NN + Emax/SE — **train-only features; excluded from v6b ensemble** (test preds collapse to std=0.106) |
| 29 | Protein embeddings (ESM-2 + ProtBERT) | — | No OOF; embedding extraction only |
| 30 | Morgan + ESM-2 multi-NR | 0.5763 | Marginal gain; zeroed by ElasticNet in v6b |
| 31 | ChemBERTa-77M-MTR + ESM-2 multi-NR | 0.6186 | Zeroed by ElasticNet |
| 32 | Morgan + ProtBERT multi-NR | 0.5749 | Zeroed by ElasticNet |
| 33 | Cross-attn ChemBERTa-77M tokens × ESM-2 residues | 0.6147 | d_chem=384; zeroed in v6c ensemble |
| 34 | Cross-attn GROVER-large global × ESM-2 residues | 0.6139 | Zeroed by ElasticNet in v6b |
| 35 | Chemprop 6-head auxiliary | 0.5665 | **28.4% weight in v6b** — 6 heads: pEC50, Emax, pEC50_null, logP, TPSA, PXR-sim |
| 36b | Grand Ensemble v6b (no aux) | **0.5281** | 16 models; excludes nb28 (train-only features) |
| 36c | Grand Ensemble v6c (+nb33 cross-attn) | **0.5281** | crossattn_chemberta_esm2 zeroed by L1; identical to v6b |

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

### Final Submission Weights (Grand v6b, full-data ElasticNetCV, 16 models)

| Model | Weight |
|---|---|
| LGBM tuned | 54.6% |
| Chemprop 6-head auxiliary (nb35) | 28.4% |
| Single-conc pseudo-label LGBM (nb26) | 6.1% |
| k-NN | 6.0% |
| Uni-Mol | 2.9% |
| ChemBERTa-MLM | 2.0% |
| All other models | 0% *(zeroed by L1)* |

**Previous best (Grand v5):** lgbm_tuned 74.7%, kNN 15.6%, Uni-Mol 9.1%, ChemBERTa-MLM 6.3%, GROVER-large 5.9%.

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

### 26 — Single-Concentration Pseudo-Labels LGBM
Calibrates 21K single-conc log2FC readouts to pseudo-pEC50 via HuberRegressor on ~300 overlapping compounds. Filters to high-confidence (FDR < 0.1 or |log2FC| > 1). Pseudo-labels added at sample_weight=0.25 to augment LGBM training.

Output: `submissions/26_singleconc_lgbm.csv`

### 27 — Nuclear Receptor Weighted LGBM
PXR CRC train + ChEMBL NR bioactivity (PPARγ, FXR, RXRα, LXRα, VDR, PPARα) with phylogenetic sample weights. NR1I subfamily (PXR, VDR) highest; distal NRs downweighted. Tests whether cross-target NR knowledge transfers.

Output: `submissions/27_nr_weighted_lgbm.csv`

### 28 — Auxiliary Feature Engineering LGBM
Augments Morgan+RDKit features with: Emax/pEC50_SE/CI-width from CRC measurements, predicted pEC50_null (null-predictor LGBM), Tanimoto similarity to 6 known PXR ligands (rifampicin, SR12813, hyperforin, T0901317, taxol, clotrimazole), and k-NN pEC50 (k=3).

Output: `submissions/28_auxiliary_features_lgbm.csv`

### 29 — Protein Embeddings: ESM-2 + ProtBERT
Extracts global protein embeddings for PXR and 6 NR targets from UniProt. ESM-2 (8M, 320-dim) and ProtBERT (1024-dim). Cached to `data/processed/nr_esm2_embeddings.npy` + `nr_protbert_embeddings.npy`. Used as protein features in notebooks 30–32.

### 30 — Multi-NR LGBM: Morgan + ESM-2
Concatenates Morgan+RDKit (2265) + ESM-2 protein embedding (320) = 2585 features per (compound, target) pair. Trains across all 7 NR targets with phylogenetic weights.

Output: `submissions/30_morgan_esm2_multinr.csv`

### 31 — Multi-NR LGBM: ChemBERTa-MTR + ESM-2
ChemBERTa-MTR CLS (768) + ESM-2 (320) = 1088 features. Multi-NR training tests whether sequence-model compound representations transfer across nuclear receptor targets.

Output: `submissions/31_chemberta_esm2_multinr.csv`

### 32 — Multi-NR LGBM: Morgan + ProtBERT
Morgan+RDKit (2265) + ProtBERT global embedding (1024) = 3289 features. Larger protein model vs. ESM-2 — tests whether deeper protein representation improves multi-NR transfer.

Output: `submissions/32_protbert_multinr.csv`

### 33 — Cross-Attention: ChemBERTa Tokens × ESM-2 Residues
Full sequence-level cross-attention: per-token ChemBERTa embeddings (L_c × 768) as Query attend to per-residue ESM-2 embeddings (294 × 320) as Key/Value. The compound learns which protein residues modulate its activity. Architecture: project → multi-head cross-attn → mean-pool (real tokens) → FFN → pEC50. Scaffold 5-fold CV.

Output: `submissions/33_crossattn_chemberta_esm2.csv`

### 34 — Cross-Attention: GROVER-large × ESM-2 Residues
Graph-based cross-attention: GROVER-large global embedding (2400-dim) projected to single-token Query, cross-attends to 294 protein residues as K/V. Learns "which residues does this molecule's graph fingerprint activate?" Complements nb33's SMILES-sequence bias with graph-topology inductive bias.

Output: `submissions/34_crossattn_grover_esm2.csv`

### 35 — Chemprop 6-Head Auxiliary Learning
Chemprop MPNN with 6 output heads: pEC50 (primary), Emax, pEC50_null, logP, TPSA, max-PXR-ligand-similarity. Auxiliary regression tasks regularize the shared molecular encoder toward physically meaningful properties, potentially improving PXR generalization.

Output: `submissions/35_chemprop_auxiliary.csv`

### 36 — Grand Ensemble v6b *(protein-aware, 16 models)*
ElasticNetCV meta-learner with nested scaffold CV over all available models from nb01–nb35. Dynamically loads any OOF arrays that exist — L1 automatically zeros redundant models. **Excludes nb28 auxiliary features** (emax/pec50_se are train-only; mean-imputing for test collapses predictions to std=0.106). v6b **OOF RAE: 0.5281** (−0.0075 vs v5). Key contributors: lgbm_tuned 54.6%, chemprop_aux 28.4%, single-conc 6.1%, kNN 6.0%. Multi-NR protein-aware models zeroed by ElasticNet (not yet complementary enough).

Note: v6 (with aux_features) was also run for analysis — OOF RAE 0.2179 but test std=0.118 (unreliable generalization).

Output: `submissions/36_grand_v6.csv` *(analysis only)*, `submissions/36b_grand_v6_no_aux.csv` *(primary)*

---

## Data

Raw data is gitignored. Re-clone if missing:

```bash
git clone https://huggingface.co/datasets/openadmet/pxr-challenge-train-test data/raw
git clone https://github.com/OpenADMET/PXR-Challenge-Tutorial.git tutorial
```

### Raw Files

| File | Rows | Description |
|---|---|---|
| `pxr-challenge_TRAIN.csv` | 4,139 | CRC dose-response: pEC50, Emax, uncertainties |
| `pxr-challenge_TEST_BLINDED.csv` | 513 | Test set — SMILES only |
| `pxr-challenge_counter-assay_TRAIN.csv` | 2,859 | PXR-null counter-screen |
| `pxr-challenge_single_concentration_TRAIN.csv` | 21,003 | Single-point screen |
| `pxr-challenge_structure_TEST_BLINDED.csv` | 184 | Structure track |

### Schema Details

**TRAIN / counter-assay** share the same schema: `name`, `smiles`, `batch`, `pec50`, `pec50_se`, `pec50_lo/hi`, `emax`, `emax_se`, `emax_lo/hi`, `emax_rel`, `emax_rel_se`, `emax_rel_lo/hi`, `split`, `source`, `ocnt_id`. All renamed to snake_case by `src/pxr/data.py`.

**single_concentration**: `name`, `smiles`, `ocnt_id`, `log2_fc_estimate`, `fdr_bh`, `n_replicates`, `concentration_M`. Note the log2FC readout is a raw fold-change — must be calibrated against CRC data before use as a pseudo-label (see nb26).

**TEST_BLINDED**: only `name` and `smiles` — no labels, no Emax, no pEC50_SE. Any feature that requires CRC measurement fields (emax, pec50_se, pec50_lo/hi) must be mean-imputed at inference, which destroys their signal.

### Data Preparation by Notebook

All notebooks share a common base pipeline:

1. Load via `src/pxr/data.py` loaders (snake_case columns)
2. Compute Bemis-Murcko scaffolds with `chem.bemis_murcko`
3. Build scaffold 5-fold splits with `eval.scaffold_kfold_indices(seed=42)` — all analogs of a scaffold go to the same fold
4. Featurize with `featurize.combined` → Morgan ECFP4 (2048-bit, radius=2) concatenated with ~217 RDKit 2D descriptors = **2,265 features**
5. Impute NaN descriptor values with column medians via `featurize.impute`

Notebook-specific deviations from this base:

| Notebook | Training rows | Feature dim | Key deviation(s) |
|---|---|---|---|
| **02** baseline | 4,139 | 2,265 | None — pure base pipeline |
| **03** Chemprop multitask | 4,139 (+2,858 counter null) | SMILES→MPNN graph | NaN-masked dual head: pEC50 + pEC50_null; counter-assay joined by InChIKey |
| **06** counter feature | 4,139 | 2,266 | pEC50_null added as 1 extra input feature; imputed for ~36% without CRC null data |
| **09** robust pipeline | 3,781 → 8,183 | 985 (selected) | SE filter (drop pec50_se > 0.5, removes 358 rows); IQR-clip descriptors (5× IQR); remove low-variance / high-corr features; upsample actives to 10%; add 4,402 pseudo-inactives from single-conc (pEC50 ≈ 4.3, weight=1.0) |
| **26** single-conc pseudo-labels | 4,139 + 7,309 pseudo | 2,265 | Calibrate log2FC → pEC50 via HuberRegressor on 5,722-compound overlap (slope=0.496, intercept=4.357, r=0.52); keep only FDR<0.1 or \|log2FC\|>1; exclude CRC-train compounds; weight pseudo-labels at 0.25 |
| **27** NR-weighted | 4,139 + 11,496 ChEMBL | 2,265 | ChEMBL NR bioactivity (PPARγ, FXR, RXRα, LXRα, VDR, PPARα) with phylogenetic sample weights: PXR=1.0, VDR=0.5, FXR=0.3, LXRα=RXRα=0.25, PPARγ=PPARα=0.15; quality-filter pEC50 ∈ [3, 10] |
| **28** auxiliary features | 4,139 | 2,277 | Adds 12 extra features on top of combined(2265): Tanimoto to 6 PXR reference ligands (rifampicin, SR12813, hyperforin, T0901317, taxol, clotrimazole); LOO k-NN pEC50 (k=3, exclude self); null-predictor pEC50 (LGBM trained on counter-assay); emax, emax_rel, emax_se, pec50_se from CRC — **test set mean-imputes the 4 CRC fields → predictions collapse; excluded from ensemble** |
| **30** Morgan+ESM-2 multi-NR | 4,139 × 7 NR rows | 2,585 | Morgan+RDKit (2265) + ESM-2 protein embedding (320-dim) concatenated; one (compound, protein) row per NR target; trained with phylogenetic weights |
| **31** ChemBERTa+ESM-2 multi-NR | 4,139 × 7 | 1,088 | ChemBERTa-77M-MTR CLS (768-dim) + ESM-2 (320-dim); otherwise same as nb30 |
| **32** Morgan+ProtBERT multi-NR | 4,139 × 7 | 3,289 | Morgan+RDKit (2265) + ProtBERT global embedding (1024-dim) |
| **33** cross-attn ChemBERTa×ESM-2 | 4,139 | L_c × 768 → scalar | Per-token ChemBERTa embeddings (variable L_c × 768) attend as Query over 294 ESM-2 residue embeddings as K/V; projection + multi-head cross-attn + mean-pool + FFN |
| **34** cross-attn GROVER×ESM-2 | 4,139 | 2400 → 1 token Q | GROVER-large global fingerprint (2400-dim) projected to single token Query, cross-attending to 294 ESM-2 residue K/V |
| **35** Chemprop 6-head auxiliary | 4,139 | SMILES→MPNN graph | 6-task target matrix: pEC50 (primary), Emax, pEC50_null (NaN-masked, 64% coverage), logP, TPSA, max-PXR-ligand-Tanimoto; each fold z-scores targets from fold-train stats and un-scales pEC50 at inference |
| **36** grand ensemble | varies per model | OOF vectors | Loads all cached `data/processed/oof_*.npy` arrays; stacks into meta-feature matrix; ElasticNetCV meta-learner with nested scaffold 5-fold CV; excludes nb28 OOF (train-only features) |

### Feature Dimensionality Reference

| Feature set | Dim | Notes |
|---|---|---|
| Morgan ECFP4 | 2,048 | radius=2, bit vector |
| RDKit 2D descriptors | 217 | via `useful_rdkit_utils.RDKitDescriptors` |
| Combined | 2,265 | Morgan ‖ RDKit; baseline for all LGBM notebooks |
| ChemBERTa-zinc CLS | 768 | frozen from `seyonec/ChemBERTa-zinc-base-v1` |
| ChemBERTa-MTR CLS | 768 | frozen from `seyonec/ChemBERTa-PubChem-base-v1` |
| ChemBERTa-77M-MTR CLS | 384 | frozen from `deepchem/ChemBERTa-77M-MTR` (hidden=384) |
| BERT-SMILES CLS | 768 | `unikei/bert-base-smiles` |
| SELFormer CLS | 768 | `HUBioDataLab/SELFormer`, SELFIES input |
| GROVER-base atom FP | 1,600 | graph-level fingerprint |
| GROVER-large atom FP | 2,400 | graph-level fingerprint |
| Uni-Mol 3D CLS | 512 | 3D conformer-aware transformer |
| ESM-2 8M protein | 320 | mean-pooled over residues; `facebook/esm2_t6_8M_UR50D` |
| ProtBERT protein | 1,024 | mean-pooled; `Rostlab/prot_bert` |

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
