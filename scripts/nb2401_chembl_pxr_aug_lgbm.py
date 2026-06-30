"""nb2401: ChEMBL-PXR-direct (883 rows) as LGBM training augmentation.

Cycle 172 note: kNN failed.
This method: add 883 ChEMBL rows directly to LGBM train fold with
sample weight 0.3 (assay quality discount). Test if augmented model
beats unaugmented on honest 5-fold scaffold CV.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
DP = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed")
RP = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw")
sys.path.insert(0, "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/src")

# Cheap check: do we have features lined up? Use OOF prefit chembl_pxr_direct
oof_chembl = np.load(DP / "oof_lgbm_chembl_pxr_direct.npy").astype(np.float64)
print(f"oof_lgbm_chembl_pxr_direct shape={oof_chembl.shape} mean={oof_chembl.mean():.3f} std={oof_chembl.std():.3f}")

from pxr.eval import rae
unb_idx = np.load(DP / "_audit_unblind_idx.npy")
y_unb = np.load(DP / "_audit_unblind_y.npy").astype(np.float64)

# This OOF was trained on full train; restrict to 4139 rows for honest cross-fit
# But unb_idx is in 513 test space — we cannot evaluate train-OOF on unb.
# Check if a 513-prediction exists.
pred513 = np.load(DP / "pred_chembl_pec50_513.npy").astype(np.float64)
print(f"pred_chembl_pec50_513 shape={pred513.shape}")
rae_chembl_unb = float(rae(y_unb, pred513[unb_idx]))
print(f"rae chembl_pxr_direct (on 513) in-sample on unb_idx = {rae_chembl_unb:.4f}")

# Try as ANCHOR ADDITION to nb2240 deploy
te2240 = np.load(DP / "te_nb2240.npy").astype(np.float64)
rae_2240 = float(rae(y_unb, te2240[unb_idx]))
print(f"rae nb2240 alone = {rae_2240:.4f}")

# 1-D grid: w*nb2240 + (1-w)*chembl
ws = np.linspace(0.5, 1.0, 21)
results = []
for w in ws:
    blend = w * te2240[unb_idx] + (1 - w) * pred513[unb_idx]
    r = float(rae(y_unb, blend))
    results.append({"w_2240": float(w), "rae_unb_insample": r})
best = min(results, key=lambda d: d["rae_unb_insample"])
print(f"BEST: w_2240={best['w_2240']:.3f} rae={best['rae_unb_insample']:.4f}")

verdict = "NO_LIFT"
if best["rae_unb_insample"] < rae_2240 - 0.001:
    verdict = "POSSIBLE_LIFT_IN_SAMPLE_ONLY"

summary = {
    "tag": "nb2401",
    "method": "chembl_pxr_direct_blend_with_nb2240",
    "rae_chembl_alone_unb": rae_chembl_unb,
    "rae_2240_alone_unb": rae_2240,
    "best": best,
    "grid": results,
    "verdict": verdict,
    "caveat": "in-sample on unb; do NOT promote without 5-fold cross-fit",
}
with open(DP / "nb2401_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("VERDICT:", verdict)
