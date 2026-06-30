# Real-docking runbook — the last distinct activity lever (Codespace/Linux)

**Why:** cycle-291 proved ChempropEmbed absorbs every *structure-derived* feature (ADMET, 3D-shape,
anchor-fit — all real over fingerprints, all absorbed by the ladder). A **Vina docking score is
physics-based** (a scoring function on a real 3D pose), not a learned structure embedding — so it is
the signal most likely to carry information chempropembed does **not** already have. No GPU needed
(Vina is CPU); runs in the Codespace where the D:-full problem doesn't exist.

## Steps (in the Codespace)

```bash
# 0. bootstrap already ran (devcontainer) -> data + 64 PDBs present
pip install vina meeko          # Linux wheels exist; ADFRsuite optional

# 1. receptor + box (regenerates from 2O9I)
python scripts/nb980_prep_receptor_box.py
#    -> C:/pxr_struct/dock/... locally; in Codespace edit OUT to ./dock_out (or set DOCK_OUT)
mk_prepare_receptor.py -i dock_out/pxr_receptor_2o9i.pdb -o receptor.pdbqt -p   # meeko receptor prep

# 2. dock (checkpointed/resumable; ~1-4 h test, ~10-40 h train on CPU)
DOCK_BOX=dock_out/dock_box.json RECEPTOR_PDBQT=receptor.pdbqt python scripts/dock_pxr.py
#    -> dock_out/scores_{test,train}.npy   (best-pose Vina energy, kcal/mol, lower=better)

# 3. the REAL test — does the vina score add to the chemprop substrate?
#    Reuse scripts/nb982_anchorfit_ladder.py, swapping the anchorfit feature for scores_*.npy:
#    base = combined + chempropembed ; base+vinascore ; chemprop_aux residual + clip ; 253 cross-fit, multi-seed.
#    STABLE-negative delta -> FIRST REAL LADDER BREAK. Absorbed -> docking closes the activity track too.
```

## What to expect
- Start with the **test set only** (513, ~1-4 h) to get the verdict fast; only dock train if the score looks promising.
- A useful enrichment even if the raw score is absorbed: **per-anchor pose contacts** (does the docked pose
  actually H-bond Ser247/His407?) — richer than the score alone. Extend dock_pxr.py to save pose contacts if needed.
- If docking ALSO gets absorbed, the activity ladder is closed on every axis except a better base representation
  (Uni-Mol fine-tune, Kaggle) or external scaffold-diverse data.
