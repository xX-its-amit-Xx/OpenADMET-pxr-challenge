# Staged test — molecule-only Uni-Mol (learned 3D), the definitive 3D close

**When:** spare / next-week Kaggle quota (structure Boltz-2 nb951 has priority).
**Why:** cycle-289 probes closed 2D-SMILES (nb953) and hand-crafted-3D (nb954/nb956 seed-noise).
The one open question: does a *learned* 209M-conformer model (Uni-Mol) extract 3D signal the
hand-crafted descriptors missed? This is the cheapest decisive test — molecule-only, NO docking,
~1 session — before any commitment to the full pocket+docking pipeline.

## Commands

```powershell
# 1. sync the data bundle (includes unimol_*.parquet with folds + max_sim) — one-time
python scripts/kaggle_push.py --data

# 2. push + run on GPU (Internet ON for Uni-Mol weight download)
$env:KAGGLE_ACCEL = "gpu"
python scripts/kaggle_push.py --nb 957
#    open the URL, confirm Accelerator=GPU + Internet ON, Save/Run-All
#    Cell 2 is a FAIL-FAST smoke test: if unimol_tools is broken it aborts in ~2 min.

# 3. pull results
python scripts/kaggle_push.py --nb 957 --pull
#    -> submissions/kaggle_nb957/nb957_result.json
```

## The decision (no moving goalposts — set in advance)

Read `nb957_result.json`:
- **`deep_extrap_mae` < 0.5924** (the nb952 LGBM-combined reference) → learned-3D is REAL;
  the full pocket+docking Uni-Mol pipeline (dock 513 into PXR LBD, pocket-conditioned fine-tune)
  is justified. Build it next.
- **`deep_extrap_mae` >= 0.5924** → the 3D axis is **definitively closed**; activity is at its
  substructure ceiling on every axis; do not pursue Uni-Mol/docking further.
- Secondary: `eval253_deploy_rae` vs chemprop_aux anchor **0.6216** — a standalone-quality check
  (Uni-Mol is unlikely to beat the anchor standalone; the degradation-curve flatness is what matters).

## Resumability / notes
- Per-fold OOF checkpoint (`nb957_oof_ckpt.npy`) + deploy checkpoint on /kaggle/working;
  re-run the push to resume (skips done folds) if a session hits the 12 h wall.
- EPOCHS=40, BS=32 in cell 4 — drop to EPOCHS=25 if a single session can't finish 5 folds + deploy.
- If smoke test fails on numpy: the install cell auto-tries numpy==1.26.4; a kernel restart + re-run
  may be needed (Kaggle base image dependent).
- Runtime estimate: ~5 folds + deploy refit on 4139 mols at conf-augmented 3D ≈ one P100 session.
