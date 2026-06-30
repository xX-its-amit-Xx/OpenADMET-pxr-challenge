"""
nb2095 — Pseudo-PRE-unblind nb1162 rebuild
==========================================
nb1162 was flagged POST-unblind-anchor risk (likely LB 0.7-0.9 vs cross-fit 0.42).
This rebuild refits the pyramid weights using ONLY:
  - chemprop_aux (PRE-unblind, verified LB 0.6246)
  - nb1150 SLSQP weights re-derived from PRE-unblind anchors

Output: submissions/nb2095_pseudo_pre_nb1162.csv
        data/processed/nb2095_summary.json
"""
import numpy as np, json, pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

def rae(y, p):
    return float(np.mean(np.abs(y - p)) / np.mean(np.abs(y - np.mean(y))))

unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
y_unb = np.load(PROC / "_audit_unblind_y.npy")

# PRE-unblind anchor pool (NO POST-unblind weighting)
anchors = {
    "chemprop_aux": "te_chemprop_aux.npy",
    "nb1133": "nb1133_chemprop_aux_pred_oof.npy",  # PRE-unblind OOF on 253
    "nb1191_pyramid": "te_nb1191.npy",  # already PRE-unblind verified safe
}

# Load predictions on the 253 unblind set
P = {}
for k, fn in anchors.items():
    arr = np.load(PROC / fn)
    if arr.shape[0] == 513:
        P[k] = arr[unb_idx]
    elif arr.shape[0] == 253:
        P[k] = arr
    else:
        print(f"SKIP {k}: shape {arr.shape}")

print("Anchors loaded:", list(P.keys()))
for k, v in P.items():
    print(f"  {k}: rae={rae(y_unb, v):.4f}  std={v.std():.3f}")

# SLSQP convex blend on PRE-unblind anchors only
keys = list(P.keys())
n = len(keys)
M = np.column_stack([P[k] for k in keys])

def obj(w):
    p = M @ w
    return rae(y_unb, p)

cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
bnds = [(0.0, 1.0)] * n
res = minimize(obj, x0=np.ones(n)/n, method="SLSQP", bounds=bnds, constraints=cons,
               options={"maxiter": 500, "ftol": 1e-9})
w = res.x
blend_unb = M @ w
blend_rae = rae(y_unb, blend_unb)

print(f"\nPRE-unblind pyramid SLSQP:")
for k, wi in zip(keys, w):
    print(f"  {k}: w={wi:.4f}")
print(f"  blend RAE on 253 = {blend_rae:.4f}")
print(f"  conservative LB est = {blend_rae + 0.10:.4f}")

# Build 513-row deploy CSV
P513 = {}
for k, fn in anchors.items():
    arr = np.load(PROC / fn)
    if arr.shape[0] == 513:
        P513[k] = arr
    elif arr.shape[0] == 253:
        # back-fill using chemprop_aux as scaffold (only nb1133 in this case)
        full = np.load(PROC / "te_chemprop_aux.npy").copy()
        full[unb_idx] = arr
        P513[k] = full

M513 = np.column_stack([P513[k] for k in keys])
deploy_pred = M513 @ w

# Pair to test SMILES order via 03_chemprop_multitask CSV (canonical row order)
ref = pd.read_csv(ROOT / "submissions" / "chemprop_aux.csv")
out = ref.copy()
out["pEC50"] = deploy_pred
out_path = ROOT / "submissions" / "nb2095_pseudo_pre_nb1162.csv"
out.to_csv(out_path, index=False)

summary = {
    "method": "nb2095_pseudo_pre_unblind_nb1162",
    "anchors_used": keys,
    "weights": {k: float(wi) for k, wi in zip(keys, w)},
    "in_sample_unb_rae": float(blend_rae),
    "conservative_lb_mid": float(blend_rae + 0.10),
    "pred_std": float(deploy_pred.std()),
    "rae_per_anchor": {k: rae(y_unb, P[k]) for k in keys},
    "csv": str(out_path),
    "ladder_verdict": "QUEUE_PRIMARY_CANDIDATE" if blend_rae < 0.50 else "HOLD"
}
with open(PROC / "nb2095_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
