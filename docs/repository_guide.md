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

## The Actual Project Flow

The repo looks chaotic because the strategy was intentionally broad before it
became intentionally narrow. The arc went like this:

### Act I: Try basically everything

The first pass was a full-spectrum model bakeoff:

- 2D fingerprints and RDKit/Mordred-style descriptors.
- LightGBM, XGBoost, CatBoost, Random Forests, ExtraTrees, Ridge, ElasticNet,
  LAD, Huber, quantile, and other robust objectives.
- Chemprop and other graph/message-passing models.
- kNN, matched molecular pairs, delta-ML, scaffold priors, routers, and
  residual correction.
- ChemBERTa, MolFormer, UniMol, GROVER, SELFormer, CheMeleon, TabPFN, and
  protein-aware embeddings.
- External PXR and nuclear receptor data from ChEMBL, PubChem, Tox21/NCATS,
  BindingDB-style sources, and literature/patent sweeps.
- 3D and physics descriptors: xTB, AIMNet2, ANI, DFT-D4, DBSTEP, MACE, SOAP,
  PMapper, OrbMol, strain, surface electrostatics, protonation, and friends.

The point was not elegance. The point was coverage. If there was a reasonable
architecture, data wrangling trick, representation, calibration, or ensemble
strategy, it probably got a tuxedo, a microphone, and at least one chance to
sing.

Scientific readout: most approaches were either weaker than the strong baseline,
absorbed by the existing ensemble, or too fragile under the 253-row honest
Analog Set 1 validation.

### Act II: Stop only making better stacks; find more information

Once the pure model-search surface started saturating, the strategy shifted from
"try a different learner" to "bake in more data." That meant:

- mining public PXR and nuclear receptor activity sources;
- using the 21k single-concentration screen as functional biology;
- testing pseudo-labels, weak labels, synthetic labels, and oracle-like
  auxiliary models;
- checking whether generated or test-adjacent analogs could provide useful
  training signal;
- looking for external assays that measured a different but relevant observable.

The key distinction was this:

> More rows only help if they add the right axis. Noisy pEC50 impostors hurt.
> A noisy but orthogonal functional activity prior helped.

That is why single-conc eventually entered as `P(active)` instead of as fake
dose-response truth.

### Act III: Bake in target structure

After the data-expansion path narrowed, the next useful axis was structural:
Boltz-derived protein-ligand interaction information. Ligand-only 3D descriptors
were often redundant, but Boltz rich-z represented a target-aware interaction
view of PXR activation.

This is the biological reason the final model gives Boltz a real role:

- fingerprints describe the ligand;
- Chemprop/GNN-style models learn ligand patterns;
- single-conc gives functional response evidence;
- Boltz rich-z adds target-context interaction evidence.

The late `nb1400` rich-z corrector win reinforces the same point: structure
helps when it is used as an orthogonal residual axis, not as decorative 3D
confetti.

### Act IV: What I would do next

If there were more time, the next phase would be less "one more blend" and more
"create or find new supervised structure/activity signal":

- Train structure-to-activity predictors from Boltz/Chai/docking interaction
  panels, not just ligand descriptors.
- Generate more protein-ligand structural hypotheses for known active and
  inactive chemotypes, then learn which interaction patterns separate them.
- Use peptide, antibody, or protein-fragment examples with similar functional
  groups as auxiliary structural priors, especially where they expose recurring
  hydrogen-bond, hydrophobic, charge, or aromatic interaction motifs.
- Train on fragment or substructure activity when available, so the model learns
  transferable "this local group tends to activate/silence PXR in this context"
  priors instead of only whole-molecule labels.
- Add more target-family structural contrast: PXR versus CAR, FXR, VDR, PPARs,
  and other nuclear receptors, but only when the endpoint can be tied back to
  activation rather than generic binding.

In late-night terms: once the orchestra has played every arrangement of the same
song, book a new guest. In science terms: the remaining headroom is information,
not another optimizer pass.

---

## Compute Used

This was not one laptop valiantly wheezing under a pile of molecules. The work
used a mixed compute stack:

| Compute venue | Main role |
|---|---|
| Northeastern Explorer cluster | Batch experimentation, larger sweeps, and long-running scientific jobs when local compute was the bottleneck |
| Modal | Cloud execution for heavier model/feature jobs and experiments that benefited from managed remote workers |
| Kaggle / Google Colab | GPU-heavy notebooks, especially Boltz/cofolding, molecular foundation models, and free-GPU exploratory runs |
| GitHub Codespaces | Remote development, repo editing, and reproducible cloud dev sessions |
| AWS WorkSpaces | Persistent remote workstation environment for development, monitoring, and CPU-bound workflow glue |
| Local Windows workstation | File orchestration, RDKit/GBM jobs, submission scripts, and the `C:/pxr_work` sidecar artifact workspace |

The exact final deploy still depends on local/sidecar arrays under
`C:/pxr_work/meta_stacking/`, but the experiment campaign itself was distributed
across those compute venues. The repo is therefore both a codebase and a lab
log: source files here, generated artifacts there, and a great deal of compute
spent asking "does this new signal actually transfer?"

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

## Validation Sets: The Iteration Engine

The validation sets were not a scoreboard stapled onto the end of the project.
They were the engine. Every new idea had to answer a specific scientific
question, and each validation surface was built to catch a different way a model
could fool us while wearing a very convincing lab coat.

Golden rule: compare numbers only inside the same validation regime. A score
from train scaffold CV, a score from the 253 released analogs, and a score from
a matched feature gate are measuring different claims.

### The validation ladder

| Validation surface | Built from | Main artifacts | Question it answered | How it changed decisions |
|---|---|---|---|---|
| Scaffold CV | 4,139 CRC dose-response rows, grouped by Murcko scaffold | `src/pxr/eval.py` | Does a basic model generalize beyond close scaffold neighbors? | Early filter for GBMs, GNNs, fingerprints, descriptors, and multitask heads |
| Tanimoto-OOD train holdout | 10 percent of train rows most dissimilar to the rest | `scripts/b318_tanimoto_holdout.py`, `data/processed/tanimoto_holdout_idx.npy` | Does the method survive low-neighbor-similarity chemistry? | Exposed methods that only worked when a close analog was available |
| Clean analog-expansion holdouts | Five scaffold-disjoint train holdouts chosen to mimic the test similarity profile | `scripts/nb1127_robust_validation.py`, `data/processed/nb1127_robust_validation.json` | Is the 253 score selection-biased or does the idea transfer to never-tuned train chemistry? | Became the honest gate for late feature axes and external-data claims |
| Matched 3-seed holdout gates | Same holdout indices reused across control and treatment models | `C:/pxr_work/mtl/ho_idx_seed*.npy`, `scripts/nb1163_gate.py`, `scripts/nb1166_gate.py`, `scripts/nb1168_gate.py` | Did the new auxiliary head or feature axis beat its exact control? | Promoted only effects that repeated across seeds instead of one lucky split |
| Mirror datasets | FXR, PPARg, RXRa, LXRa, and PXR public analog-expansion simulations | `scripts/nb1090_mirror_datasets.py`, `data/processed/nb1090_mirror.json` | If we had richer public data for a related receptor, would coverage close the gap? | Shifted emphasis from "new architecture" to "new information axis" |
| PRE-unblind cross-fit | Models trained before Analog Set 1 labels were released | `final_lb_report.md`, `data/processed/final_lb_candidates.json` | What would have transferred before seeing the 253 truth? | Anchored LB-faithful estimates and avoided rewriting history after unblinding |
| 253 unblind LOOCV | Released Analog Set 1 labels, one held out at a time | `data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`, `scripts/nb1320_singleconc_inactive.py`, `scripts/nb1333_deploy_260.py` | Does the final correction work on the real analog-expansion distribution? | Selected the final single-conc shift/gate and reported the honest 0.5799 RAE |
| Distance-weighted 253 clusters | Five clusters on the 253 by Tanimoto distance | `scripts/nb1283_distance_weighted_kfold.py`, `data/processed/nb1283_summary.json` | How optimistic is ordinary KFold when chemical neighborhoods are too similar? | Added a pessimistic LB-faithful check; random KFold looked about 0.10 RAE too friendly |
| Leave-series-out checks | Butina-style chemical series held out together | Summarized in `SUBMISSION_SUMMARY.md` and late candidate notes | Do corrections survive whole chemical series being absent? | Kept scaffold and series transfer separate from row-level interpolation |
| Best-of-Bag repeats | Repeated outer and inner folds, then mean/median bagging | `data/processed/nb1191_oof_summary.json`, `final_lb_report.md` | Is a candidate robust or just the funniest split in the room? | Demoted lucky seeds and promoted reproducible families |
| Integrity audit | Prediction files checked against released truth and OOF provenance | `scripts/integrity_audit.py`, `data/processed/integrity_audit.csv`, `scripts/audit_ladder_integrity.py` | Did any `te_*.npy` silently leak the 253 truth or use in-sample predictions? | Removed contaminated anchors and forced deploy artifacts to prove their ancestry |
| Submission and LB ladder | Submitted CSVs, public LB logs, and final submission records | `data/processed/submission_log.csv`, `data/processed/leaderboard_log.csv`, `data/processed/lb_history.csv` | How did internal estimates map onto actual challenge scoring? | Calibrated pre-unblind expectations, then became secondary once the LB froze |

Tiny monologue translation: each validation set was a different heckler in the
audience. If a model could answer all of them without sweating through the suit,
it got to stay in the act.

### Clean holdouts: the no-glamour truth serum

The clean analog-expansion holdouts were built from the training set, not from
the released 253. They were scaffold-disjoint, never tuned against, and chosen
to resemble the test set's train-similarity profile. This mattered because once
the 253 existed, it was very easy to accidentally create a model that learned
the taste of the validation set instead of the chemistry.

`nb1127` quantified that risk:

| Quantity | Value |
|---|---:|
| 253 combined RAE | 0.6987 |
| Clean holdout RAE | 0.4746 |
| Clean holdout std | 0.0206 |
| Gap | 0.2241 |

The interpretation is not "the 253 is bad." The interpretation is that the 253
and the clean train holdouts ask different questions. The 253 is the real
analog-expansion distribution; the clean holdouts are a sanity check against
over-selecting on that real distribution after it becomes visible.

This is why late-stage ideas such as external-data augmentation, multitask GNN
heads, structural features, MACE/physics descriptors, and rich-z correctors were
often routed through matched holdout gates before they were allowed anywhere
near deploy.

### Mirror datasets: rehearsal stages in related biology

The mirror datasets asked a bigger question: if the problem is data coverage,
can we see that effect on related nuclear receptors where richer public data
exists? `nb1090` built small analog-expansion games for FXR, PPARg, RXRa, LXRa,
and PXR, then compared a poor 400-compound training set against a richer public
training pool.

| Target | n compounds | Test size | Poor RAE | Rich RAE | Coverage gap |
|---|---:|---:|---:|---:|---:|
| FXR | 2,816 | 250 | 0.7925 | 0.6324 | 0.1601 |
| PPARg | 2,516 | 250 | 0.8339 | 0.5997 | 0.2342 |
| RXRa | 432 | 216 | 0.7267 | 0.7267 | 0.0000 |
| LXRa | 447 | 224 | 0.8714 | 0.8714 | 0.0000 |
| PXR | 737 | 246 | 0.7831 | 0.7715 | 0.0116 |

The readout was wonderfully annoying, which is how you know it is probably
useful. FXR and PPARg showed that richer coverage can matter a lot. Public PXR
did not show the same rescue, which suggested that more rows were not enough
unless they measured the right activity axis or covered the right chemical
neighborhoods. That helped justify the later pivot toward single-concentration
functional biology and Boltz target-structure information.

### The 253: gold standard, dangerous candy

The 253 unblinded analogs were the best available proxy for the hidden 260:
same challenge design, same analog-expansion pressure, same weird little
activity cliffs. They were also dangerous, because once visible, they could be
overfit by enthusiasm.

So the project used them in three distinct ways:

| Use | Meaning | Guardrail |
|---|---|---|
| LOOCV validation | Fit on 252 released analogs, score the held-out one | Used for final correction selection |
| Deploy-mode sanity check | Fit components with the 253 included, inspect 253 in-sample behavior | Reported separately as deploy-mode, not honest LOOCV |
| Final submitted 513 file | Replace the 253 predictions with released truth | Legal because those labels were public at final submission time |

That distinction is the reason `submissions/nb1333_final_513.csv` and
`submissions/FINAL_pxr_activity_submission.csv` are not the same object. The
first is a model-output artifact with LOOCV predictions on the 253. The second
is the actual submitted truth-hybrid file: 253 released labels plus 260 blind
predictions.

### Distance-weighted 253 clusters

Ordinary random KFold on 253 analogs can place very similar compounds on both
sides of the split. `nb1283` made a harsher five-cluster split by Tanimoto
distance:

| Check | Value |
|---|---:|
| Random KFold mean holdout-to-train similarity | 0.1478 |
| Distance-weighted mean holdout-to-train similarity | 0.1144 |
| Similarity delta | -0.0335 |
| Random-KFold RAE reference | 0.5431 |
| Distance-weighted mean-bag RAE | 0.6406 |
| Estimated optimism gap | 0.0975 |

This was the "dim the stage lights and see who can still sing" validation. It
helped separate interpolation wins from true chemical transfer.

### Integrity audits: because one perfect score is a crime scene

The postmortem audited 343 models across 177 notebooks against the 253 truth.
It found the main failure modes:

| Failure mode | Observed pattern |
|---|---|
| Variance compression | Median prediction std 0.62 versus truth std 1.03 |
| Novel-scaffold inactive miss | Low-activity compounds overpredicted by about +1.23 pEC50 |
| Rare-active under-recovery | High-activity compounds underpredicted by about -0.54 pEC50 |
| CV/stacking overfit | Train-OOF correlation with unblind 0.505, Spearman about 0.03 |

After that, every serious ladder candidate needed a provenance check. The audit
logic looked for suspicious `te_*.npy` files whose 253 subset was much better
than matching OOF predictions, suspiciously close to truth, or otherwise
inconsistent with how the artifact claimed to be trained. Several impressive
anchors were downgraded after this check. The repo did not enjoy that episode,
but it did get healthier.

### The actual iteration loop

This was the practical loop that drove the model search:

1. Propose a new axis: architecture, descriptor, external data source,
   synthetic label, structural feature, or post-hoc correction.
2. Choose the validation surface that matches the claim.
3. Compare against a matched control whenever possible.
4. Require robustness across folds, seeds, or chemical series before promotion.
5. Write the summary JSON or CSV so the result can be audited later.
6. Run integrity checks before treating any deploy prediction as trustworthy.
7. Promote only if the gain survives in the regime it claims to improve.

That is why the final model is not simply "the best score we saw." It is the
survivor of a validation gauntlet: broad architecture search, clean holdouts,
mirror-dataset lessons, postmortem failure analysis, 253 LOOCV, single-conc
biology, Boltz structural signal, and one last provenance check at the door.

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
