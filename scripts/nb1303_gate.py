"""nb1303 gate — compare rat rPXR 4-head OOF (rpxr_oof_seed*) vs deployed sn (sn_oof_seed*).
Uses same gate convention as sn_gate: 4-GBM + CheMeleon + TabPFN + GNN swap.
Run AFTER nb1303_rat_pxr_aux_head.py completes all 3 seeds.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae
from src.pxr.featurize import combined, impute
import json

OUT = "C:/pxr_work/mtl"
BEST_JSON = "C:/pxr_work/search/best_ensemble.json"
N_SEEDS = 3

ARCHNAME = "4-GBM+CheMeleon+TabPFN+GNN"

# ---- Load holdout labels ----
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y_all = tr["pec50"].to_numpy()


def load_oof(prefix, seeds=N_SEEDS):
    oofs = []
    for s in range(seeds):
        path = f"{OUT}/{prefix}_seed{s}.npy"
        if not os.path.exists(path):
            print(f"MISSING: {path}"); return None
        oofs.append(np.load(path))
    return oofs  # list of (n_ho,) arrays


def rae_per_seed(oofs):
    """Get RAE on the holdout for each seed."""
    vals = []
    for s, po in enumerate(oofs):
        ho = np.load(f"{OUT}/ho_idx_seed{s}.npy")
        y_ho = y_all[ho]
        vals.append(rae(y_ho, po))
    return vals


def rae_ensemble(oofs_list):
    """Average OOF across GNN variants (each element = per-seed OOF for one GNN config)."""
    return rae_per_seed(oofs_list)


# Load control (sn) and treatment (rpxr)
sn_oofs = load_oof("sn_oof")
rpxr_oofs = load_oof("rpxr_oof")

if sn_oofs is None or rpxr_oofs is None:
    print("NOT ALL OOF READY — check training logs")
    sys.exit(1)

# Also need the GBM and CheMeleon / TabPFN OOFs for the full ensemble gate
# Use the combo seed data from the deployed pipeline
combo_pkl = "C:/pxr_work/_combo_seed_data.pkl"
if not os.path.exists(combo_pkl):
    print(f"Missing combo pkl: {combo_pkl}")
    sys.exit(1)

import pickle
with open(combo_pkl, "rb") as f:
    combo = pickle.load(f)

# combo contains per-seed holdout OOFs for all members (keys like 'lgbm', 'xgb', etc.)
# Convention from prior gate scripts: swap gnn_oof_seed* for the GNN member
# then average and compute RAE

print("Gate: sn (control) vs rpxr (treatment)")
print()

sn_raes, rpxr_raes = [], []
for s in range(N_SEEDS):
    ho = np.load(f"{OUT}/ho_idx_seed{s}.npy")
    y_ho = y_all[ho]

    # Control: use sn_oof_seed*
    sn_pred = sn_oofs[s]
    # Treatment: use rpxr_oof_seed*
    rpxr_pred = rpxr_oofs[s]

    # Build ensemble — load per-seed GBM + CheMeleon + TabPFN OOFs from combo
    # Then swap GNN member
    seed_data = combo.get(s, {})
    if not seed_data:
        print(f"WARNING: no combo data for seed {s}, using GNN-only comparison")
        sn_raes.append(rae(y_ho, sn_pred))
        rpxr_raes.append(rae(y_ho, rpxr_pred))
        continue

    member_preds = {}
    for k, v in seed_data.items():
        if k != 'gnn' and k != 'sisterNR_gnn':
            ho_v = v[ho] if hasattr(v, '__len__') and len(v) == len(tr) else v
            member_preds[k] = ho_v

    # Control ensemble
    ens_sn = np.mean([sn_pred] + [member_preds[k] for k in member_preds if member_preds[k] is not None], axis=0)
    # Treatment ensemble
    ens_rpxr = np.mean([rpxr_pred] + [member_preds[k] for k in member_preds if member_preds[k] is not None], axis=0)

    sn_raes.append(rae(y_ho, ens_sn))
    rpxr_raes.append(rae(y_ho, ens_rpxr))

print(f"sn   RAE per seed: {[round(r,4) for r in sn_raes]} mean={np.mean(sn_raes):.4f}")
print(f"rpxr RAE per seed: {[round(r,4) for r in rpxr_raes]} mean={np.mean(rpxr_raes):.4f}")
delta = np.mean(rpxr_raes) - np.mean(sn_raes)
n_neg = sum(r < s for r, s in zip(rpxr_raes, sn_raes))
print(f"delta (rpxr - sn) = {delta:+.4f}  ({n_neg}/{N_SEEDS} seeds negative)")
print()
if delta < -0.001:
    print(">>> DEPLOY-WORTHY: delta < -0.001")
else:
    print(">>> SUB-THRESHOLD: no deploy")
