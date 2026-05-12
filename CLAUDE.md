# PXR Challenge — Codebase Guide

OpenADMET PXR Blind Challenge. Two tracks: activity prediction (pEC50 regression on 513 analogs) and structure prediction (184 protein-ligand complexes). Primary metric is RAE (Relative Absolute Error) for activity; LDDT-PLI for structure.

## Environment

```bash
# activate
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Unix

# install / sync deps
uv sync

# jupyter kernel (run once)
python -m ipykernel install --user --name pxr-challenge --display-name "pxr-challenge"
```

Python 3.11–3.12. PyTorch is CPU-only by default (see `pyproject.toml` index config). To switch to GPU, change the torch index URL to `https://download.pytorch.org/whl/cu124`.

## Data

Raw data is gitignored. Re-clone if missing:

```bash
git clone https://huggingface.co/datasets/openadmet/pxr-challenge-train-test data/raw
git clone https://github.com/OpenADMET/PXR-Challenge-Tutorial.git tutorial
```

| File | Rows | Description |
|---|---|---|
| `pxr-challenge_TRAIN.csv` | 4,139 | CRC dose-response: pEC50, Emax, uncertainties |
| `pxr-challenge_TEST_BLINDED.csv` | 513 | Test set — SMILES only, no labels |
| `pxr-challenge_counter-assay_TRAIN.csv` | 2,859 | PXR-null counter-screen, same schema |
| `pxr-challenge_single_concentration_TRAIN.csv` | 21,003 | Single-point screen: log2FC, FDR, concentration_M |
| `pxr-challenge_structure_TEST_BLINDED.csv` | 184 | Structure track: crystal code + SMILES |

Load via `src/pxr/data.py` — all loaders return short snake_case column names.

## Project Layout

```
src/pxr/          core library
  paths.py        centralized path constants (auto-creates processed/figures dirs)
  data.py         dataset loaders (load_train, load_test, load_counter, load_single_conc)
  chem.py         RDKit utilities: standardize, Morgan FP, Tanimoto, physchem, Murcko scaffold
  eval.py         RAE metric, compute_metrics, scaffold_kfold_indices, cv_score
  featurize.py    feature sets: rdkit_desc (~217), morgan (2048), combined (2265)
scripts/
  kernel_launcher.py   Windows asyncio fix for Jupyter kernel
notebooks/        numbered sequence — run in order
data/
  raw/            gitignored — HF dataset clone
  processed/      derived parquets, figures
  external/       PDB structures, external databases
submissions/      versioned CSVs — never overwrite, always new version number
```

## Source Library (src/pxr/)

### `chem.py`
RDKit utilities. Key functions:
- `standardize(smi)` — largest fragment, neutralize, canonical SMILES. Stereo preserved, tautomers NOT canonicalized.
- `morgan_fp_batch(smiles)` → (N, 2048) uint8 — vectorized ECFP4
- `compute_physchem(smi)` → dict: MW, logP, TPSA, HBD, HBA, rotbonds, fsp3, rings, charge
- `add_standard_columns(df)` — adds std_smiles, inchikey, scaffold, physchem cols in-place

### `eval.py`
- `rae(y_true, y_pred)` — primary leaderboard metric. RAE < 1.0 beats mean predictor.
- `scaffold_kfold_indices(scaffolds, n_splits=5)` — scaffold-aware splits; each scaffold entirely in one fold; largest scaffolds assigned first for balanced folds.
- Always use scaffold CV, not random. Test set is analog expansion — random splits are ~0.1 RAE optimistic.

### `featurize.py`
- `rdkit_desc(smiles)` → (N, 217) float32
- `morgan(smiles)` → (N, 2048) float32
- `combined(smiles)` → (N, 2265) float32 — Morgan + RDKit concatenated; best single feature set
- `impute(X)` — median imputation; call after any featurize function before model fit

## Notebooks & Models

Run in numbered order. Each notebook saves to `submissions/` and passes results forward as the "base" for the next.

### `01_eda.ipynb` — Exploratory Data Analysis
Question-driven EDA anchored to modeling decisions. Key outputs:
- `data/processed/compounds.parquet` — 12,889 unique standardized compounds across all configs
- `data/processed/cliffs_train.parquet` — 10 activity cliff pairs (Tanimoto ≥ 0.7, |ΔpEC50| ≥ 1.0)
- `data/processed/test_difficulty.parquet` — per-test-compound difficulty score (low neighbor similarity + high neighbor disagreement = hard)

Key numbers: median pEC50 std error 0.24 (noise floor); 1.6% of train compounds are hits (pEC50 ≥ 6); test top-1 Tanimoto median 0.52; train ∩ test InChIKey overlap = 0 (no leakage).

### `02_baseline.ipynb` — LGBM Baseline
**Output:** `submissions/02_lgbm_baseline.csv`

LightGBM on combined features (Morgan + RDKit). Scaffold 5-fold CV RAE: 0.575. Reproduces and improves tutorial baseline. Combined features beat Morgan-only (0.658) and RDKit-only (0.594). This is the anchor model.

Hyperparams: n_estimators=500, num_leaves=64, learning_rate=0.05.

### `03_multitask.ipynb` — Multi-task Chemprop GNN
**Output:** `submissions/03_chemprop_multitask.csv`, `submissions/03_ensemble_lgbm_chemprop.csv`

Chemprop 2.x MPNN with two heads: PXR pEC50 + counter-assay pEC50. 2,858 compounds have both labels; remainder NaN-masked on null head. Architecture: BondMessagePassing (depth=3, d_h=300) + MeanAgg + FFN (2 layers, dropout=0.1). Scaffold holdout RAE: 0.517.

50/50 ensemble with LGBM becomes the base for notebooks 04–06.

### `04_mmp.ipynb` — Matched Molecular Pair Cliff Corrector
**Output:** `submissions/04_mmp_corrected.csv`

rdMMPA single-cut fragmentation → 1,516 transform pairs; 377 are activity cliffs (|ΔpEC50| > 1.0). At inference: 340/513 test compounds have training analogs. Adaptive blend weight = min(0.60, 0.60 × n_analogs / 5) — more analogs → higher MMP trust. Predictions clipped to training range ± 0.5.

### `05_knn.ipynb` — Tanimoto k-NN
**Output:** `submissions/05_knn_blend.csv`

Similarity-weighted k=5 NN on ECFP4 (Tanimoto). Scaffold CV RAE 0.789 — looks weaker than LGBM but scaffold CV systematically underestimates kNN here because test compounds have real training neighbors (median top-1 sim 0.52). Blended at w=0.40 on top of LGBM+Chemprop base.

### `06_counter_feature.ipynb` — Counter-Assay as Input Feature (in progress)
**Output:** `submissions/06_counter_lgbm.csv`

Uses pEC50_null as an input feature to LGBM (not just a training target). For the 2,858 compounds with real null values, use them directly. For the rest, train a null-predictor LGBM and impute. Retrain main LGBM with this extra feature.

## Submission Convention

- One CSV per notebook, named `NN_descriptor.csv`
- Always two columns: `Molecule Name`, `pEC50`
- Never overwrite — bump the version number
- Validate with `tutorial/validation/activity_validation.py` before submitting

## Key Numbers to Keep in Mind

| Fact | Value |
|---|---|
| Training CRC compounds | 4,139 |
| Test compounds (activity track) | 513 |
| Assay noise floor (median pEC50 SE) | 0.24 log-units |
| Hit rate in training | 1.6% (pEC50 ≥ 6) |
| Test top-1 Tanimoto to train (median) | 0.52 |
| Activity cliff pairs in train | 10 (Tan ≥ 0.7, |Δ| ≥ 1.0) |
| Counter-assay overlap with train | 2,858 / 4,139 |
| Single-conc compounds exclusive to SP | 8,126 |
| Mean predictor RAE | ~1.0 |
| Best single model RAE (Chemprop) | 0.517 |

## Phase Timeline

- Phase 1 close: 2026-05-25 — submit best ensemble
- Analog Set 1 unblinds: 2026-05-26 — ~250 new labels available for refit
- Final deadline: 2026-07-01 — design pipeline for fast Phase 2 refit