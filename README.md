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

**Best submission:** `submissions/62_grand_v7.csv` *(Grand Ensemble v7 — 32 models, OOF RAE 0.5189)*

| Model | OOF RAE |
|---|---|
| Mean predictor baseline | ~1.0 |
| LGBM baseline (Morgan + RDKit) | 0.5600 |
| LGBM tuned (Optuna) | 0.5394 |
| Chemprop multitask GNN | 0.5170 |
| Grand Ensemble v5 (nested CV) | 0.5356 |
| Grand Ensemble v6b (protein-aware, 16 models) | 0.5281 |
| **Grand Ensemble v7 (32 models, deep ensemble + Chemprop aux)** | **0.5189** |

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

### New Model Results (nb37–nb85)

| # | Model | OOF RAE | Notes |
|---|---|---|---|
| 37 | External data fetch | — | Fetches PubChem/Tox21/BindingDB/ChEMBL NR; caches to `data/external/`; no OOF |
| 38 | Activity cliff mining | — | 149 cliff pairs (SALI ≥ 2, Tan ≥ 0.5, \|Δ\| ≥ 1.0); cliff labels for 248/4,139 training compounds; no OOF |
| 39 | Hard negative augmentation ablation | 0.5606 | 5-strategy ablation of single-conc inactive pseudo-labels; CRC-only (baseline) wins |
| 40 | Cliff-weighted LGBM | 0.5677 | Cliff members upweighted (10×/8×); auxiliary cliff-role classifier; marginal vs unweighted |
| 41 | Chemprop 8-head cliff auxiliary | 0.5670 | nb35 + 2 cliff classification heads (cliff_role_bin, cliff_member); marginal improvement over 6-head |
| 42 | SMOTE/ADASYN oversampling | 0.5578 | Oversample pEC50 ≥ 6 actives to 10%; ADASYN best; minimal gain over baseline |
| 43 | Focal loss LGBM | 0.5529 | 5 custom LGBM objectives; rank-ensemble of all 5 objectives is best; saves `oof_focal_loss.npy` |
| 44 | Curriculum LGBM | 0.5599 | Easy→hard warm-start in 3 stages; anti-curriculum also tested; standard baseline wins |
| 45 | Pairwise ranking LGBM | 0.5629 | LambdaRank-style pairwise gradient for cliff pairs; zeroed by grand v7 ElasticNet |
| 46 | Siamese cliff net | 0.5943 | Twin MLP on Morgan FP pairs; cliff-pair contrastive loss; weak signal |
| 47 | Multitask MLP | 0.5882 | PyTorch MLP with 3 heads (pEC50, Emax, pEC50_null); marginal |
| 48 | TabNet | 0.7904 | Attention-based tabular model; underperforms LGBM on this dataset |
| 49 | Wide & deep MLP | 0.5911 | Wide (linear Morgan) + deep (RDKit) two-stream network |
| 50 | CatBoost cliff | 0.5594 | CatBoost with cliff-member upweighting; near-baseline |
| 51 | XGBoost DART | 0.5635 | XGBoost with DART dropout; marginal |
| 52 | Chemprop pretrain+finetune | — | Chemprop self-supervised pretraining on unlabelled SMILES; OOF missing |
| 53 | Chemprop Tox21 transfer | — | Chemprop finetuned from Tox21 NR multitask; OOF missing |
| 54 | Deep ensemble uncertainty | 0.5627 | 5-member LGBM deep ensemble with uncertainty estimates; **28.7% weight in grand v7** |
| 55 | LGBM all external data | 0.5600 | LGBM augmented with all external NR data (ChEMBL + BindingDB); ties baseline |
| 56 | Chemprop 10-head | — | 10-task Chemprop (pEC50 + 9 NR targets); OOF missing |
| 57 | LGBM PubChem PXR | 0.5600 | LGBM + PubChem PXR bioassay augmentation; ties baseline |
| 58 | LGBM BindingDB NR | 0.5600 | LGBM + BindingDB NR binding data augmentation; ties baseline |
| 59 | LGBM cliff oversample | 0.5600 | LGBM with cliff-member compound oversampling; ties baseline |
| 60 | Deep graph cliff | — | GCN with cliff-pair contrastive head; OOF missing |
| 61 | Cliff analysis (diagnostic) | — | Diagnostic notebook: per-model cliff-region RAE; no submission |
| 62 | **Grand Ensemble v7** | **0.5189** | 32-model ElasticNet; lgbm_tuned 45.4%, deep_ensemble 28.7%, chemprop_aux 21.3%; beats v6b by 0.009 RAE |
| 63 | Expanded data fetch | — | BindingDB bulk TSV, Papyrus, PubChem bug fix, ChEMBL all PXR types; supplementary cache |
| 64 | LGBM full metrics baseline | 0.5643 | LGBM with full set of physico-chemical + assay-quality descriptors |
| 65 | LGBM CRC+single-conc FDR | 0.6560 | CRC train + single-conc filtered by FDR < 0.05; poor performance (FDR filter too strict) |
| 66 | LGBM ChEMBL PXR direct | 0.5692 | LGBM with ChEMBL PXR direct IC50/EC50/Ki/AC50 augmentation |
| 67 | LGBM ChEMBL all-NR weighted | 0.6034 | LGBM with all ChEMBL NR targets at phylogenetic weights (broad NR transfer) |
| 68 | LGBM PubChem PXR fixed | 0.5643 | Retry of nb57 with corrected PubChem PXR cache |
| 69 | LGBM counter soft labels | 0.5643 | Counter-assay pEC50_null as a soft label (auxiliary regression target) |
| 70 | LGBM CRC+SP+ChEMBL PXR | 0.7771 | CRC + single-conc + ChEMBL PXR combined training; too much label noise |
| 71 | LGBM all external v2 | 0.8010 | Extended external augmentation v2; performance degrades with noisy external labels |
| 72 | LGBM cliff-aware external | 0.5972 | External data filtered to cliff-proximal compounds only |
| 73 | LGBM multitask heads | 0.5646 | LGBM per-head (pEC50, Emax, null) with stacking |
| 74 | Chemprop ChEMBL-NR multitask | 4.76 | Chemprop multitask on ChEMBL NR; training failed (OOF invalid) |
| 75 | Grand metrics comparison | — | Diagnostic: comprehensive metrics (Pearson, Spearman, Kendall, R²) per model; no submission |
| 76 | Delta-ML template | 0.4164 | Nearest-neighbour delta correction on top of base model; OOF RAE possibly inflated by NN leakage within fold |
| 77 | Sparse Nyström GP (Tanimoto kernel) | 0.7781 | Gaussian process with Tanimoto kernel; sparse Nyström approximation; slow and weaker than LGBM |
| 78 | Multi-fingerprint diversity ensemble | 0.5897 | 7 FP types (ECFP4, FCFP4, MACCS, Avalon, RDKit, AtomPair, Topological); stack has marginal diversity |
| 79 | 3D shape descriptors | 0.5643 | Conformer-derived 3D descriptors (PMI, shape moments) added to combined features |
| 80 | SMILES enumeration augmentation | 0.5581 | SMILES randomisation augmentation (10 enumerations/compound); weak improvement |
| 81 | Pseudo-label self-training | 0.5643 | Iterative pseudo-labelling on test set; self-training loop |
| 82 | Selectivity-aware prediction | 0.5917 | Predicts selectivity ratio (PXR/CAR) as auxiliary target |
| 83 | Graph label spreading | 0.5643 | Label propagation on compound similarity graph; negligible gain |
| 84 | Free-Wilson scaffold decomposition | 0.5636 | Additive Free-Wilson model on Murcko scaffolds + substituents |
| 85 | Creative mega-ensemble (ElasticNetCV) | — | ElasticNet over full OOF stack; OOF RAE 0.2181 is **in-sample only** (ElasticNet fitted on full OOF without nested CV — not a real estimate); not submitted |

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
| Grand v5 | nb25 | v4 + GROVER-large | 0.5356 |
| Grand v6b | nb36 | + Chemprop 6-head aux; protein-aware 16 models | 0.5281 |
| **Grand v7** | **nb62** | **32 models; deep ensemble gets 28.7% weight** | **0.5189** |

### Final Submission Weights (Grand v7, full-data ElasticNetCV, 32 models)

| Model | Weight |
|---|---|
| LGBM tuned | 45.4% |
| Deep ensemble (nb54) | 28.7% |
| Chemprop 6-head auxiliary (nb35) | 21.3% |
| Single-conc pseudo-label LGBM (nb26) | 2.5% |
| Uni-Mol | 2.0% |
| ChemBERTa-MLM | 0.1% |
| All other models | 0% *(zeroed by L1)* |

**Previous best (Grand v6b):** lgbm_tuned 54.6%, chemprop_aux 28.4%, singleconc_lgbm 6.1%, kNN 6.0%, Uni-Mol 2.9%, ChemBERTa-MLM 2.0%.

**Grand v5:** lgbm_tuned 74.7%, kNN 15.6%, Uni-Mol 9.1%, ChemBERTa-MLM 6.3%, GROVER-large 5.9%.

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

### 37 — External Bioactivity Data Fetch
Fetches PXR and related NR bioactivity from PubChem BioAssay (4 AIDs), Tox21 NR pathway data, BindingDB NR binding (IC50/Ki), and extended ChEMBL NR targets (CAR + broader measurement types). Results cached to `data/external/`. PubChem cache was empty (API issues); BindingDB fell back to ChEMBL webresource client (5,690 records); ChEMBL extended = 11,496 records.

### 38 — Activity Cliff Mining
Systematic cliff detection (ECFP4 Tanimoto ≥ 0.5, |ΔpEC50| ≥ 1.0, SALI ≥ 2.0) across all available bioactivity data. Found **149 intra-training cliff pairs** (248/4,139 cliff-member compounds; 6%) and 524 cross-dataset PXR–NR cliff pairs. Also computes test-set cliff proximity scores.

Outputs: `cliff_labels.parquet`, `cliff_pairs.parquet`, `cross_cliff_pairs.parquet`, `test_cliff_proximity.parquet`

### 39 — Hard Negative Augmentation Ablation
Five-strategy ablation of adding single-conc non-responders as pseudo-inactives (varying pEC50 assignment: N(4.3,0.24), N(2.5,0.5) low floor, bimodal, PubChem). CRC-only baseline (RAE 0.5606) wins every variant — augmentation hurts. **OOF RAE: 0.5606.**

Output: `submissions/39_hard_negatives.csv`

### 40 — Cliff-Weighted LGBM
LGBM with per-compound sample weights emphasising cliff members (cliff-active 10×, cliff-inactive 8×, active non-cliff 6×). Auxiliary 3-class cliff-role LGBM classifier (OOF cliff-role accuracy 98.5%). Cliff weighting slightly hurts overall OOF RAE (+0.008 vs unweighted). **OOF RAE: 0.5677.**

Output: `submissions/40_cliff_weighted_lgbm.csv`

### 41 — Chemprop 8-Head Cliff Auxiliary
Extends nb35 Chemprop 6-head to 8 heads, adding `cliff_role_bin` (NaN-masked classification) and `cliff_member` (fully observed binary classification). Auxiliary cliff heads act as regularisers on the shared encoder. **OOF RAE: 0.5670.**

Output: `submissions/41_chemprop_cliff_heads.csv`

### 42 — SMOTE/ADASYN Oversampling
Oversamples the active region (pEC50 ≥ 6; 1.6% of training) to 10% using four strategies: RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE. Synthetic pEC50 values are k-NN interpolated from real neighbours. ADASYN is best (0.5578) but only marginally better than baseline (0.5599). **OOF RAE: 0.5578.**

Output: `submissions/42_smote_adasyn.csv`

### 43 — Focal Loss LGBM
Five custom LGBM objectives: MSE, focal MSE (γ=2), cliff-weighted MSE, asymmetric Huber (δ=0.5), quantile-aware mixture (τ=0.90). Rank-ensemble of all five objectives gives the best overall OOF RAE. **OOF RAE: 0.5529.**

Output: `submissions/43_focal_loss_lgbm.csv`

### 44 — Curriculum LGBM
Progressive training via LGBM warm-start: Stage 1 easy compounds (difficulty < 0.33) → Stage 2 medium → Stage 3 all, with difficulty scored by cliff proximity, dissimilarity to same-activity neighbours, and pEC50 SE. Also tests anti-curriculum (hard→easy). Standard LGBM baseline wins (0.5599). **OOF RAE: 0.5599.**

Output: `submissions/44_curriculum_lgbm.csv`

### 45 — Pairwise Ranking LGBM
LambdaRank-style pairwise gradient derived from cliff pairs (149 pairs); training penalises inversions in predicted rank within each cliff pair. **OOF RAE: 0.5629.** Zeroed by ElasticNet in grand v7.

Output: `submissions/45_pairwise_ranking_lgbm.csv`

### 46 — Siamese Cliff Net
Twin-MLP architecture on Morgan FP pairs; contrastive loss on cliff pairs. **OOF RAE: 0.5943.** Weaker than LGBM — insufficient training signal from only 149 cliff pairs.

Output: `submissions/46_siamese_cliff_net.csv`

### 47 — Multitask MLP
PyTorch MLP with 3 heads: pEC50 (primary), Emax, pEC50_null (NaN-masked). Combined features as input. **OOF RAE: 0.5882.**

Output: `submissions/47_multitask_mlp.csv`

### 48 — TabNet
Attention-based tabular model (PyTorch TabNet) on combined features. **OOF RAE: 0.7904** — underperforms LGBM; attention mechanism not well-suited to this high-dimensional sparse FP input.

Output: `submissions/48_tabnet_pxr.csv`

### 49 — Wide & Deep MLP
Two-stream network: wide branch (linear on Morgan FP) + deep branch (MLP on RDKit descriptors), concatenated and mapped to pEC50. **OOF RAE: 0.5911.**

Output: `submissions/49_wide_deep_mlp.csv`

### 50 — CatBoost Cliff
CatBoost regressor with cliff-member upweighting (same weight schedule as nb40). **OOF RAE: 0.5594.** Near-baseline; CatBoost not significantly better than LGBM here.

Output: `submissions/50_catboost_cliff.csv`

### 51 — XGBoost DART
XGBoost with DART dropout regularisation. **OOF RAE: 0.5635.** Marginal versus LGBM.

Output: `submissions/51_xgboost_dart.csv`

### 52 — Chemprop Pretrain+Finetune
Chemprop self-supervised pretraining on unlabelled SMILES (masked atom prediction), then fine-tuned on PXR pEC50. OOF array missing from `data/processed/` — likely pretraining was too slow on CPU to complete.

### 53 — Chemprop Tox21 Transfer
Chemprop multitask trained on Tox21 NR tasks then fine-tuned on PXR pEC50. OOF array missing — transfer learning from Tox21 binary labels did not complete.

### 54 — Deep Ensemble Uncertainty
5-member LGBM deep ensemble (independently initialised) with per-compound uncertainty estimates (std of 5 predictions). **OOF RAE: 0.5627.** Gains substantial weight in grand v7 (28.7%) due to diverse signal vs single LGBM.

Output: `submissions/54_deep_ensemble.csv`

### 55 — LGBM All External Data
LGBM augmented with all external NR data (ChEMBL + BindingDB, 17K+ extra rows). **OOF RAE: 0.5600** — ties baseline; external label noise cancels gains.

Output: `submissions/55_lgbm_all_external.csv`

### 56 — Chemprop 10-Head
Chemprop MPNN with 10 output heads (pEC50 + 9 NR targets from ChEMBL). OOF array missing from `data/processed/` — training likely incomplete.

### 57 — LGBM PubChem PXR
LGBM trained on CRC data augmented with PubChem PXR bioassay actives/inactives (pseudo-pEC50 assigned). **OOF RAE: 0.5600** — ties baseline; PubChem pseudo-labels too noisy.

Output: `submissions/57_lgbm_pubchem_pxr.csv`

### 58 — LGBM BindingDB NR
LGBM augmented with BindingDB NR binding data (5,690 records; pIC50 from IC50/Ki). **OOF RAE: 0.5600** — ties baseline.

Output: `submissions/58_lgbm_bindingdb.csv`

### 59 — LGBM Cliff Oversample
LGBM with cliff-member compound oversampling (repeat cliff pairs in training). **OOF RAE: 0.5600** — ties baseline; oversampling 248 cliff members doesn't change LGBM significantly.

Output: `submissions/59_lgbm_cliff_oversample.csv`

### 60 — Deep Graph Cliff
GCN with a cliff-pair contrastive head; graph-level representations for PXR pEC50. OOF array missing — likely CPU training too slow.

### 61 — Cliff Analysis (Diagnostic)
Per-model breakdown of RAE on cliff-member compounds vs non-cliff compounds. Identifies which models handle the activity cliff region best. No submission; generates `cliff_model_breakdown.parquet` for use in grand v7 sub-ensemble selection.

### 62 — Grand Ensemble v7 *(primary submission)*
ElasticNetCV meta-learner (32 models, nested scaffold CV). Collapse criterion: skip models with test_std < 0.4 × train_std. **Nested CV RAE: 0.5189** (−0.0092 vs v6b). Key contributors: lgbm_tuned 45.4%, deep_ensemble 28.7%, chemprop_aux 21.3%. New models from nb37–nb61 are zeroed by L1 except through indirect contribution via the deep ensemble. Test prediction std = 0.626 (well-calibrated; v6 was 0.118).

Output: `submissions/62_grand_v7.csv`

### 63 — Expanded Data Fetch
Revisits nb37 failures: BindingDB bulk TSV download (falls back to ChEMBL PXR proxy, 945 rows); Papyrus NR subset via Zenodo (falls back to ChEMBL extended); PubChem cache bug fix attempt; ChEMBL direct PXR with all measurement types (812 rows: AC50, EC50, IC50, Ki). Supplementary caches for downstream notebooks.

### 64 — LGBM Full Metrics Baseline
LGBM trained with comprehensive metrics reporting (Pearson, Spearman, Kendall, R², cliff accuracy). Establishes full-metric baseline. **OOF RAE: 0.5643.**

Output: `submissions/64_lgbm_full_metrics.csv`

### 65 — LGBM CRC+Single-Conc FDR
CRC training data augmented with single-conc compounds filtered at FDR < 0.05 (strict). **OOF RAE: 0.6560** — FDR filtering too strict, leaving noisy pseudo-labels that degrade performance.

Output: `submissions/65_lgbm_crc_singleconc_fdr.csv`

### 66 — LGBM ChEMBL PXR Direct
LGBM augmented with ChEMBL PXR direct IC50/EC50/Ki/AC50 data (812 records from nb63). **OOF RAE: 0.5692** — small ChEMBL PXR set helps marginally.

Output: `submissions/66_lgbm_chembl_pxr_direct.csv`

### 67 — LGBM ChEMBL All-NR Weighted
LGBM with all ChEMBL NR targets (7 targets, ~11K records) at phylogenetic sample weights. **OOF RAE: 0.6034** — broad NR transfer adds too much off-target noise.

Output: `submissions/67_lgbm_chembl_all_nr_weighted.csv`

### 68 — LGBM PubChem PXR Fixed
Retry of nb57 with corrected PubChem PXR cache (still empty; cache bug not resolved). **OOF RAE: 0.5643.**

Output: `submissions/68_lgbm_pubchem_pxr_fixed.csv`

### 69 — LGBM Counter Soft Labels
Counter-assay pEC50_null used as a soft auxiliary regression target (not just input feature). **OOF RAE: 0.5643.**

Output: `submissions/69_lgbm_counter_soft.csv`

### 70 — LGBM CRC+SP+ChEMBL PXR
Combined CRC training + single-conc pseudo-labels + ChEMBL PXR augmentation. **OOF RAE: 0.7771** — stacking too many noisy data sources degrades performance severely.

Output: `submissions/70_lgbm_crc_sp_chembl_pxr.csv`

### 71 — LGBM All External v2
Extended external augmentation v2 (all external sources combined). **OOF RAE: 0.8010** — further degradation; external label noise dominates.

Output: `submissions/71_lgbm_all_external_v2.csv`

### 72 — LGBM Cliff-Aware External
External data filtered to compounds with Tanimoto ≥ 0.4 to any training cliff member. **OOF RAE: 0.5972** — cliff proximity filter still insufficient to remove noise.

Output: `submissions/72_lgbm_cliff_aware_external.csv`

### 73 — LGBM Multitask Heads
Separate LGBM models per head (pEC50, Emax, pEC50_null) stacked via Ridge. **OOF RAE: 0.5646.**

Output: `submissions/73_lgbm_multitask_heads.csv`

### 74 — Chemprop ChEMBL-NR Multitask
Chemprop multitask on all ChEMBL NR targets (7 tasks). OOF RAE = 4.76 (training failure — likely gradient explosion or wrong target scaling). Excluded from grand ensemble.

### 75 — Grand Metrics Comparison (Diagnostic)
Comprehensive metrics dashboard across all models: RAE, MAE, R², Pearson r, Spearman ρ, Kendall τ. No submission. Generates ranked comparison table.

### 76 — Delta-ML Template
Nearest-neighbour delta correction: for each test compound, find its most similar training compound, predict Δ(pEC50) from a secondary model, and add to the base prediction. **OOF RAE: 0.4164** — suspicious; likely inflated by NN leakage within scaffold folds. Not included in grand v7 ensemble.

Output: `submissions/76_delta_ml.csv`

### 77 — Sparse Nyström GP (Tanimoto Kernel)
Gaussian process regression with Tanimoto similarity kernel, approximated via sparse Nyström inducing points. **OOF RAE: 0.7781** — theoretically principled but underperforms LGBM on this dataset size.

Output: `submissions/77_gp_tanimoto.csv`

### 78 — Multi-Fingerprint Diversity Ensemble
7 FP types (ECFP4, FCFP4, MACCS, Avalon, RDKit, AtomPair, Topological) — one LGBM per FP, stacked via ElasticNet. **OOF RAE: 0.5897** (per-FP stack) / 0.6320 (raw stack). Diversity from different FPs is modest; ECFP4 dominates.

Output: `submissions/78_multi_fp_ensemble.csv`

### 79 — 3D Shape Descriptors
Conformer-derived 3D descriptors (principal moments of inertia, shape asymmetry, eccentricity) appended to combined features. **OOF RAE: 0.5643** — 3D shape adds little over Morgan FP at this dataset size.

Output: `submissions/79_3d_shape.csv`

### 80 — SMILES Enumeration Augmentation
Training augmented with 10 non-canonical SMILES enumerations per compound (all point to the same molecule; RDKit canonicalization is applied before featurisation, so this mainly adds minor descriptor diversity). **OOF RAE: 0.5581.**

Output: `submissions/80_smiles_aug.csv`

### 81 — Pseudo-Label Self-Training
Iterative self-training: fit LGBM, pseudo-label test set, add high-confidence test predictions back as training data, refit. **OOF RAE: 0.5643** — self-training loop did not converge to improvement.

Output: `submissions/81_pseudo_label.csv`

### 82 — Selectivity-Aware Prediction
Predicts PXR/CAR selectivity ratio as an auxiliary target alongside pEC50. **OOF RAE: 0.5917** — selectivity feature adds noise rather than signal at this data scale.

Output: `submissions/82_selectivity_aware.csv`

### 83 — Graph Label Spreading
Semi-supervised label propagation on compound Tanimoto similarity graph; spreads training pEC50 labels to nearby unlabelled compounds and uses them as soft augmentation. **OOF RAE: 0.5643** — propagation noise from low-similarity compounds dominates.

Output: `submissions/83_graph_spreading.csv`

### 84 — Free-Wilson Scaffold Decomposition
Additive Free-Wilson model: decomposes pEC50 into scaffold contribution + substituent contributions via linear regression on substructure indicators. **OOF RAE: 0.5636.** Interpretable but slightly weaker than LGBM.

Output: `submissions/84_free_wilson.csv`

### 85 — Creative Mega-Ensemble (ElasticNetCV over all OOFs)
ElasticNet stacking over all 36 available OOF arrays. **In-sample OOF RAE: 0.2181** — this is not a valid generalisation estimate (ElasticNet is fitted on the full OOF stack without nested CV, so it memorises the OOF predictions). The dominant coefficient is `aux_features` (0.987) — the same train-only feature collapse seen in nb28. Not submitted.

Output: `submissions/85_creative_mega_ensemble.csv` *(in-sample only, not submitted)*

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
