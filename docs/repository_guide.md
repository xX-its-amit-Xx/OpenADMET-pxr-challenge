# OpenADMET PXR Challenge - Repository Guide

Good evening, and welcome to **PXR After Dark**, the show where the host is
over-caffeinated, the assay is noisy, and the punchline is usually "the
validation split was lying to us."

This is the clean map of the repository: what matters, why it matters, how the
final submission was assembled, and where all the experimental side quests live.
It uses the SuperCowPowers weekend write-up as a stylistic and scientific
reference: start with a strong baseline, use an honest out-of-distribution
yardstick, and only trust complexity when it is aligned with the biology.

Reference: https://supercowpowers.github.io/workbench/blogs/pxr_weekend_experiments/

---

## The One-Screen Version

**Final submitted file:** `submissions/FINAL_pxr_activity_submission.csv`

**What it contains:**

- 513 activity-track rows total.
- 253 rows from Analog Set 1 use the released true pEC50 labels.
- 260 still-blind rows use the validated `nb1333` deploy predictions.
- Submission succeeded on 2026-06-30 16:25:34 UTC, per
  `data/processed/submission_log.csv`.

**Final 260-blind model, in human language:**

> A 29-model base ensemble plus local analog correction plus a Boltz-2
> protein-ligand interaction signal, then a single-concentration PXR screen
> correction to catch activity cliffs and confident inactives.

**Final 260-blind model, in math:**

```text
base_blend = 0.40 * meta_stacker
           + 0.20 * tanimoto_knn_residual
           + 0.10 * combined_2d_corrected
           + 0.30 * boltz_richz_F5

shifted = base_blend + 0.10 * (P_active_single_conc - 0.5) * 2

if P_active_single_conc < 0.15
   and boltz_richz_F5 < 3.2
   and shifted < 3.5:
       pEC50 = 3.0
else:
       pEC50 = shifted
```

**Validated final numbers:**

| Check | Value |
|---|---:|
| Honest 253 LOOCV RAE for `nb1333` stack | 0.5799 |
| Deploy-mode 253 in-sample RAE | 0.5636 |
| Blind rows predicted by `nb1333` | 260 |
| Gate fires on blind rows | 4 |
| Blind prediction mean / std | 4.873 / 0.715 |
| Clean base models after leakage filter | 29 |

The final CSV was verified against local files:

- 253 final rows match `data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`
  exactly.
- 260 final rows match `submissions/nb1333_final_260.csv` exactly.

Tiny studio applause. Science happened.

---

## Why The Final Recipe Makes Sense

The challenge is not "predict a random split of PXR compounds." That would be a
Tuesday. The hard part is an analog expansion with novel scaffolds, activity
cliffs, and a tiny active tail. A model can look heroic under ordinary CV and
then faceplant on the revealed analog set.

The final recipe exists because each component covers a different failure mode:

| Problem | Final response |
|---|---|
| Base models shrink predictions toward the middle | MAE/RAE-aware stack plus single-conc shift |
| Similar molecules can flip activity sharply | Tanimoto kNN residual corrector |
| 2D structure misses target-context signal | Boltz-2 rich-z interaction embedding |
| Inactive cliffs are over-predicted | Low `P(active)` single-conc gate floors confident inactives |
| Released 253 labels are no longer blind | Use them directly in the final submitted 513 file |
| The 260 labels remain blind | Rebuild deploy components using 4,139 CRC + 253 released labels, never 260 labels |

The most important idea is **orthogonal biology**. Many fancy molecular
descriptors were either redundant with the graph/fingerprint models or too
fragile at n=253. The single-concentration PXR screen measures actual response
behavior on many compounds. It is noisy, but it is a different observation of
the same biological axis. That is why it earned a seat at the desk.

---

## Final Submission, Step By Step

### 1. Define the two test partitions

File: `scripts/nb1333_deploy_260.py`

Inputs:

- `data/raw/pxr-challenge_TEST_BLINDED.csv`
- released 253 truth labels from `C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv`
  during the original run
- equivalent repo copy: `data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`

The 513 test compounds are split by molecule name:

- `ub_idx`: 253 Analog Set 1 rows with released pEC50
- `bl_idx`: 260 rows still blind at final submission time

The final submitted CSV later uses the released truth for `ub_idx`.

### 2. Load the base model prediction pool

File: `C:/pxr_work/meta_stacking/preds_513.pkl`

`nb1333` loads many 513-row prediction vectors and filters out obvious leakers:

```python
if mean_absolute_error(pred_on_253, y253) < 0.05:
    drop_model()
```

That leaves **29 clean base models**. This is not decoration; it is the
checkpoint where the orchestra stops letting the kazoo solo into the microphone.

### 3. Build the meta-stacker component

Files:

- `C:/pxr_work/meta_stacking/meta_stacker_loocv_253.npy`
- `C:/pxr_work/meta_stacking/meta_stacker_te_260.npy`

The meta-stacker learns how to combine base predictions. It is used in two
modes:

- LOOCV mode for honest 253 validation.
- Deploy mode for the 260 blind predictions.

### 4. Build the Tanimoto kNN residual component

Files:

- `C:/pxr_work/meta_stacking/knn_residual_loocv_253_correct.npy`
- `C:/pxr_work/meta_stacking/knn_residual_loocv_te_260_v2.npy`

This component corrects a model using nearby measured analogs. It is most useful
when local chemical neighborhoods carry signal that the global model smooths
away.

### 5. Add the combined 2D corrected component

File: `C:/pxr_work/meta_stacking/combined_corrected_513.npy`

This is the 2D Morgan + RDKit corrected prediction stream. It is lower glamour,
higher utility: the house band of QSAR.

### 6. Add the Boltz rich-z interaction component

Files:

- `data/processed/boltz_z_rich_513.npy`
- `data/processed/boltz_z_rich_train.npy`
- `C:/pxr_work/meta_stacking/loocv_253_F5_base+boltz.npy`

The final deploy component takes a PCA-24 projection of the Boltz-2 rich
protein-ligand interaction embedding, concatenates it with the clean base
prediction matrix, and fits a RidgeCV model.

Important details:

- PCA for the 260 deploy path is fit on the released 253 rows only.
- The honest 253 sanity check uses the precomputed LOOCV file.
- This is target-aware 3D information, not just ligand-only geometry.

Why it helped: Boltz rich-z is both **orthogonal to the base members** and
**aligned with PXR activation biology**. Several other 3D/physics axes were
orthogonal but not activity-aligned, which is a beautiful way to spend compute
and then receive no applause.

### 7. Blend the four primary components

File: `scripts/nb1333_deploy_260.py`

Weights:

```text
0.40 meta_stacker
0.20 tanimoto_knn_residual
0.10 combined_2d_corrected
0.30 boltz_richz_F5
```

These weights are deliberately simple. At n=253, over-tuning blend weights is
how a validation score learns jazz hands and forgets chemistry.

### 8. Train the single-concentration classifier

Files:

- `scripts/nb1320_singleconc_inactive.py`
- `data/raw/pxr-challenge_single_concentration_TRAIN.csv`
- `data/processed/nb1320_singleconc.json`
- `data/processed/nb1320_final.json`

The single-conc screen is aggregated per SMILES:

```text
active = max(log2_fc_estimate) > 0.5 and min(fdr_bh) < 0.1
```

Then an LGBMClassifier is trained on Morgan + RDKit combined features to predict
`P(active)`.

Why classification, not pseudo-label regression? Earlier attempts to pour
single-conc rows directly into pEC50 training usually hurt. The final version
uses the screen as an **activity prior**, not as fake dose-response truth.

### 9. Apply the single-conc shift and inactive gate

File: `scripts/nb1333_deploy_260.py`

First, a small continuous shift:

```text
shifted = base_blend + 0.10 * (P_active - 0.5) * 2
```

Then, a conservative inactive floor:

```text
if P_active < 0.15 and boltz < 3.2 and shifted < 3.5:
    pEC50 = 3.0
```

Validation summary:

- `data/processed/nb1320_final.json`
  - baseline: 0.5977
  - single-conc shift: 0.5883
  - final gate: 0.5799
  - 253 validation flags: 11
- `data/processed/nb1333_deploy.json`
  - 260 gate flags: 4

### 10. Write the `nb1333` deploy outputs

Generated by `scripts/nb1333_deploy_260.py`:

- `submissions/nb1333_final_260.csv`
- `submissions/nb1333_final_513.csv`
- `data/processed/nb1333_deploy.json`
- `data/processed/nb1333_260_annotations.csv`

Important distinction:

`nb1333_final_513.csv` is a model-output artifact. For the 253 released rows it
uses the honest LOOCV stack predictions, not the released truth labels.

`FINAL_pxr_activity_submission.csv` is the actual final submitted artifact. It
uses:

- released truth for the 253 unblinded rows
- `nb1333_final_260.csv` predictions for the 260 blind rows

### 11. Submit the final CSV

File: `scripts/final_submit_watcher.py`

Submission metadata:

- user alias: `scaffold-sherpa`
- track: Activity Prediction
- model tag: GitHub repository URL
- file: `submissions/FINAL_pxr_activity_submission.csv`

Success log:

```text
2026-06-30 16:25:34 UTC, FINAL_pxr_activity_submission.csv,
SUCCESS: Submission received ...
```

That is the final curtain call. Cue house band, fade to molecule names.

---

## Validation Regimes: Do Not Mix These Numbers

This repo contains many good numbers, suspicious numbers, and numbers wearing a
little fake mustache. Compare only within a regime.

| Regime | What it means | Use it for |
|---|---|---|
| Scaffold CV on train | Splits 4,139 CRC rows by scaffold | Early model development |
| PRE-unblind cross-fit | Models trained before Analog Set 1 labels were released | LB-faithful pre-release estimates |
| 253 LOOCV | Leave-one-out validation on released Analog Set 1 | Final correction selection |
| Truth-hybrid final | 253 released truth + 260 blind predictions | Actual final submitted file |
| Late gate validation | Matched holdouts over internal train rows | Deciding whether a new feature axis deserves deploy |

Golden rule: if a score was tuned on the 253 and then scored on the same 253,
the number is a rehearsal, not opening night. The postmortem notebooks exist
largely because the repo learned this lesson the dramatic way.

---

## Repository Map

Snapshot from this workspace, excluding `.git` and `.venv` as places where
clarity goes to nap:

| Area | What lives there | Notes |
|---|---|---|
| `src/pxr/` | Shared Python package | Data loaders, RDKit helpers, featurizers, metrics |
| `scripts/` | Main experiment and deploy engine | Thousands of `nb*.py` scripts plus submitters, gates, fetchers |
| `notebooks/` | Notebook experiments | Early human-readable notebooks plus generated Kaggle notebooks |
| `data/raw/` | Challenge CSVs | Source of truth for train/test/counter/single-conc/structure |
| `data/processed/` | Derived arrays, summaries, ledgers | The archaeology layer and many model sidecars |
| `submissions/` | Versioned CSV and zip outputs | Many historical candidates; final file is explicitly named |
| `analysis/postmortem/` | Phase-1 autopsy notebooks | Best guide to why CV lied and what failed |
| `docs/` | Human docs | This guide plus early brainstorm |
| `trajectory.md`, `trajectory_v2.md` | Auto-generated experiment timelines | Useful, but not the final-submission source of truth |
| `SUBMISSION_SUMMARY.md` | Concise final submission summary | Good companion to this guide |
| `AutogluonModels/`, `catboost_info/`, `checkpoints/`, `logs/` | Generated model/training artifacts | Useful only when debugging runs |
| `tutorial/` | Cloned OpenADMET tutorial repo | Validation/reference, not project-owned code |
| `REINVENT4/` | Cloned generative-model repo | Side quest for analog generation |
| `C:/pxr_work/...` | Sidecar workspace outside repo | Required for exact late-stage final deploy rerun |

File inventory inside the main project areas:

| Extension | Count | What it usually is |
|---|---:|---|
| `.npy` | 3142 | Cached predictions/features |
| `.py` | 1858 | Experiments, deployers, helper scripts |
| `.csv` | 1265 | Data products and submissions |
| `.json` | 1163 | Model summaries and ledgers |
| `.ipynb` | 166 | Notebooks |
| `.md` | 22 | Documentation and runbooks |

Translation: this is a challenge lab notebook that grew legs, learned
automation, and started submitting on a schedule.

---

## Core Library

### `src/pxr/data.py`

Loads every challenge config and renames columns into short snake_case names.
Key loaders:

- `load_train()`: 4,139 CRC dose-response rows.
- `load_test()`: 513 blinded activity rows.
- `load_counter()`: 2,859 PXR-null counter-assay rows.
- `load_single_conc()`: 21,003 single-concentration screen rows.
- `load_structure()`: 184 structure-track rows.
- `load_phase1_unblinded()`: 253 released Analog Set 1 rows.
- `load_crudes()` and `load_semi_pure()`: later HTChem/microscale add-ons.

### `src/pxr/chem.py`

RDKit utility drawer:

- Parse and standardize SMILES.
- Preserve stereochemistry.
- Avoid tautomer canonicalization unless a specific method chooses it.
- Compute Morgan fingerprints, Tanimoto, Murcko scaffolds, and physchem
  descriptors.

### `src/pxr/featurize.py`

Builds the standard model features:

- `morgan`: ECFP4, 2048 bits.
- `rdkit_desc`: roughly 200 RDKit descriptors.
- `combined`: Morgan + RDKit descriptors.
- `impute`: median imputation.

### `src/pxr/eval.py`

Owns scoring and splitting:

- `rae(y_true, y_pred)`: challenge metric.
- `compute_metrics`: RAE, MAE, R2, Spearman, Kendall.
- `scaffold_kfold_indices`: scaffold-aware folds.

The scaffold splitter is central. Random CV was repeatedly optimistic on this
analog-expansion problem.

### `src/pxr/external.py` and `src/pxr/affinity.py`

External-data support:

- ChEMBL target fetches for nuclear receptors.
- PubChem BioAssay fetches for PXR assays.
- Unit normalization onto the p-scale.

This is where the repo tried to turn public potency data into useful signal.
Usually the answer was "nice try, wrong manifold," but it was a necessary test.

---

## Method Families

### 1. Strong Baselines

Representative files:

- `notebooks/02_baseline.ipynb`
- `notebooks/03_multitask.ipynb`
- `scripts/nb950_chemprop_aux_v2.py`

The early story mirrors the reference blog: a plain graph model and sturdy
Morgan/RDKit tree models set a hard baseline. A lot of fancy things failed
because they were fancy, not because they were informative.

Scientific lesson:

> The baseline is not the warm-up act. On this assay, the baseline is a
> bouncer with a clipboard.

### 2. Fingerprints, Tree Models, and SHAP-Pruned Substrates

Representative files:

- `scripts/nb152_lgbm_mae_tuned.py`
- `scripts/nb156_catboost_mae.py`
- `scripts/nb167_xgboost_mae.py`
- `scripts/nb2112_deploy_shap28.py`
- `scripts/nb3204...` candidates summarized in `data/processed/final_ladder_summary.json`

These models are the workhorses: LightGBM, XGBoost, CatBoost, HistGBM, Random
Forest, ExtraTrees, and pruned fingerprints. They are robust, fast, and easy to
bag. They also saturate quickly when all members see the same 2D structure.

### 3. Meta-Stacking and Ensembles

Representative files:

- `scripts/nb109_deep_meta_stack.py`
- `scripts/nb1450_deploy_nb1441_blend.py`
- `scripts/nb1564_final_lb_candidates.py`
- `C:/pxr_work/meta_stacking/run_meta_stacker.py`

Ensembling helped only when the members were genuinely diverse. Large SLSQP or
ElasticNet blends could produce beautiful validation scores and questionable
transfer. The final stack therefore keeps the blend simple and assigns weight
to components with different information sources.

### 4. kNN, Delta-ML, and Matched Molecular Pairs

Representative files:

- `notebooks/04_mmp.ipynb`
- `notebooks/05_knn.ipynb`
- `scripts/nb101_transductive_delta.py`
- `scripts/nb1241_pred_singleconc_residual.py`

These methods help when a test molecule has useful measured neighbors. They
struggle when a scaffold is novel or a cliff jumps in the wrong direction.
Final status: useful as a **component**, dangerous as the whole show.

### 5. Counter-Assay and Selectivity

Representative files:

- `scripts/nb107_assay_decomposition.py`
- `scripts/nb111_selectivity_primary.py`
- `scripts/nb984_counter_impute.py`

The counter-assay is biologically meaningful because PXR activation can be
confounded by non-specific response or assay artifacts. Counter/null predictions
helped as auxiliary heads, selectivity transforms, and decomposition features.

### 6. Single-Concentration Biology

Representative files:

- `scripts/nb99_sc_bio_fingerprint.py`
- `scripts/nb219_singleconc_augment.py`
- `scripts/nb1320_singleconc_inactive.py`

Early attempts to use single-conc data as pseudo-label pEC50 were noisy. The
final useful form was a classifier-derived `P(active)` correction. That is the
difference between asking a blurry photograph to be a ruler and asking it
whether the lights are on.

### 7. External Public Data

Representative files:

- `scripts/chembl_pxr_harvest.py`
- `scripts/nb1040_pxr_direct_chembl.py`
- `scripts/nb130_external_pxr_augment.py`
- `data/processed/research_swarm_ledger.md`

ChEMBL, PubChem, Tox21, BindingDB, NCATS qHTS, nuclear receptor panels, patents,
and literature mining were all explored. Most were either duplicated, too
off-manifold, antagonist-flavored, species-mismatched, or redundant after the
single-conc and GNN axes.

The ledger eventually settled into status-check mode: no new machine-readable
test-adjacent PXR pEC50 source had appeared before final close.

### 8. Foundation Models and Learned Molecular Embeddings

Representative files:

- `notebooks/13_chemberta.ipynb`
- `notebooks/14_molformer.ipynb`
- `scripts/chemeleon_finetune_v2.py`
- `scripts/nb1323_integrate_chemeleon.py`
- `scripts/modal_unimol.py`

Tried families include ChemBERTa, MolFormer, UniMol, GROVER, SELFormer,
CheMeleon, TabPFN, ProtBERT/ESM-style protein embeddings, and cross-attention
compound-protein experiments.

Bottom line: useful occasionally as ensemble diversity, rarely as a dominant
signal. Several models were absorbed by the stronger 2D/GNN stack.

### 9. Physics, 3D, and Structural Proxies

Representative files:

- `scripts/aimnet2_features.py`
- `scripts/fod_xtb_features.py`
- `scripts/dbstep_features.py`
- `scripts/mace_strain_features.py`
- `scripts/boltz_api_cofold.py`
- `scripts/modal_boltz_cofold.py`
- `scripts/combinator_nb1400_gate.py`

Tested axes include xTB/GFN, AIMNet2, ANI2x, DFT-D4, DBSTEP, MACE, SOAP,
PMapper, OrbMol, protonation, surface electrostatics, strain, and Boltz-2
cofold embeddings.

Most ligand-only physics descriptors were absorbed. The exception was structural
signal used in the right role: as a residual/corrector axis, especially
Boltz-rich interaction embeddings and the SOAP/PMapper/rich-z family.

Late note:

- `submissions/nb1400_comp_z_ensemble.csv` is the best late internal gate
  artifact in `C:/pxr_work/search/best_ensemble.json` with RAE 0.4133.
- It is not the same artifact as the final submitted
  `FINAL_pxr_activity_submission.csv`.

### 10. Structure Track

Representative files:

- `scripts/auto_submit_structure_ladder.py`
- `scripts/structure_boltz_multiseed_PLAN.md`
- `scripts/validate_structure_submission.py`
- `submissions/structure_*.zip`

The structure track is separate from activity. It uses LDDT-PLI, PDB/CIF pose
submission files, Boltz/Chai/Vina-style pose generation, and validation scripts.
Some structural outputs became activity features, but the tracks should not be
confused.

### 11. Agentic Medicinal Chemistry and Generative Side Quests

Representative files:

- `scripts/nb1330_agentic_context.py`
- `outputs/pxr_agentic_chemist/`
- `REINVENT4/`

These runs explored med-chem priors, activity-cliff annotations, and generation
near test scaffolds. The scientific result was mostly negative: textbook SAR
reasoning is not enough when cliffs are deliberately adversarial.

### 12. Postmortem and Audits

Representative files:

- `analysis/postmortem/README.md`
- `analysis/postmortem/pm05_why_cv_lied.py`
- `trajectory.md`
- `trajectory_v2.md`
- `scripts/audit_ladder_integrity.py`

This is where the repo grew up. The audits found validation contamination,
anchor drift, and overfit blend selection. They are not optional reading if you
want to extend the work without accidentally re-opening a closed trapdoor.

---

## Reproducing The Final State

### Environment

```bash
uv sync
.venv/Scripts/python.exe -m ipykernel install --user --name pxr-challenge --display-name pxr-challenge
```

The project is Python 3.11 to 3.12. PyTorch is CPU-only by default in
`pyproject.toml`.

### Data

Raw challenge CSVs live in `data/raw/`. If missing, restore from the OpenADMET
Hugging Face dataset:

```bash
git clone https://huggingface.co/datasets/openadmet/pxr-challenge-train-test data/raw
```

### Exact final deploy rerun

The exact late-stage deploy depends on sidecar artifacts in:

```text
C:/pxr_work/meta_stacking/
C:/pxr_work/phase1_unblind/
C:/pxr_work/search/
```

To regenerate the 260 predictions:

```bash
.venv/Scripts/python.exe scripts/nb1333_deploy_260.py
```

Expected side effects:

```text
submissions/nb1333_final_260.csv
submissions/nb1333_final_513.csv
data/processed/nb1333_deploy.json
data/processed/nb1333_260_annotations.csv
```

Then assemble the actual final submission by taking:

- 253 released truth labels from `data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`
- 260 predictions from `submissions/nb1333_final_260.csv`

The assembled, already-submitted file is:

```text
submissions/FINAL_pxr_activity_submission.csv
```

### Submit

```bash
.venv/Scripts/python.exe scripts/final_submit_watcher.py
```

This script handles rate-limit waits and writes to
`data/processed/submission_log.csv`.

---

## Naming Decoder

| Pattern | Meaning |
|---|---|
| `nbNNN_*.py` | Numbered experiment or deploy script |
| `*_summary.json` | Machine-readable result summary |
| `*_gate.py` | A controlled test deciding whether a feature axis deploys |
| `*_deploy_*.py` | Trains on the full allowed data and writes a submission |
| `*_truth.csv` | Usually includes released Analog Set 1 truth rows; inspect carefully |
| `FINAL_*` | Historical "final" for a phase, not necessarily the final submission |
| `FINAL_pxr_activity_submission.csv` | The actual final activity submission |

The repo has many "final" files because the challenge had phases, ladders, and
late manual submissions. The real final gets the full name. Very showbiz.

---

## What To Read Next

If you only have 20 minutes:

1. `SUBMISSION_SUMMARY.md`
2. `docs/repository_guide.md`
3. `scripts/nb1333_deploy_260.py`
4. `scripts/nb1320_singleconc_inactive.py`
5. `data/processed/nb1333_deploy.json`

If you want the cautionary tale:

1. `analysis/postmortem/README.md`
2. `analysis/postmortem/pm05_why_cv_lied.py`
3. `trajectory.md`
4. `trajectory_v2.md`

If you want to extend the science:

1. Pick one validation regime and write it at the top of the experiment.
2. Compare only against candidates in the same regime.
3. Prefer orthogonal, mechanism-aligned data over bigger versions of the same
   molecular representation.
4. Add a summary JSON. Future-you deserves receipts.

---

## Final Scientific Takeaway

The reference blog's weekend lesson still holds: simple, strong, honest baselines
beat fashionable complexity unless the extra signal matches the biology.

This repo's final lesson is the sequel:

> When structure-trained models hit the wall, the winning move is not a louder
> stack. It is a different measurement. Here, the single-concentration screen and
> target-aware Boltz interaction embeddings supplied information the 2D ensemble
> could not fully infer.

And with that, good night from PXR After Dark. Tip your assay controls, validate
your splits, and never let a suspiciously perfect 253-score leave the building
without a chaperone.
