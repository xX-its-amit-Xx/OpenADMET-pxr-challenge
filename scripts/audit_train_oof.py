"""Audit OOF predictor pairs.

Find all train OOF arrays of shape (4139,) (or (4139,1)) and pair with matching
test (513,) arrays. Compute train OOF RAE and test unblind RAE (253 phase-1).
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
PROC = os.path.join(ROOT, "data", "processed")


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    num = np.mean(np.abs(y_true - y_pred))
    den = np.mean(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


# Load train labels (4139 in csv order)
tr_csv = pd.read_csv(os.path.join(ROOT, "data", "raw", "pxr-challenge_TRAIN.csv"))
y_tr = tr_csv["pEC50"].values.astype(float)
assert y_tr.shape == (4139,), y_tr.shape

# Load unblind labels & test indices
unb_idx = np.load(os.path.join(PROC, "_audit_unblind_idx.npy")).astype(int)
y_unb = np.load(os.path.join(PROC, "_audit_unblind_y.npy")).astype(float)


def find_test_match(stem: str) -> str | None:
    """Given a train OOF basename (no ext), find matching te_*.npy.

    Try several name conventions.
    """
    candidates = []
    # convention 1: oof_X.npy -> te_X.npy
    if stem.startswith("oof_"):
        rest = stem[len("oof_"):]
        candidates.append(f"te_{rest}.npy")
        candidates.append(f"te_oof_{rest}.npy")
    # convention 2: X_oof.npy / X_pred_oof.npy / X_resid_oof.npy -> te_X*.npy
    if stem.endswith("_pred_oof"):
        rest = stem[: -len("_pred_oof")]
        candidates.append(f"te_{rest}_pred.npy")
        candidates.append(f"te_{rest}.npy")
        candidates.append(f"{rest}_pred_te.npy")
    if stem.endswith("_resid_oof"):
        rest = stem[: -len("_resid_oof")]
        candidates.append(f"te_{rest}_resid.npy")
        candidates.append(f"{rest}_resid_te.npy")
    if stem.endswith("_oof"):
        rest = stem[: -len("_oof")]
        candidates.append(f"te_{rest}.npy")
        candidates.append(f"{rest}_te.npy")
    # convention 3: nbNNN_pred_oof.npy -> nbNNN_pred_te.npy (already covered)
    # convention 4: te_oof_X.npy is the matching test file in some cases
    candidates.append(f"te_{stem}.npy")
    for c in candidates:
        p = os.path.join(PROC, c)
        if os.path.isfile(p):
            return p
    return None


# Glob candidate train OOF files
patterns = ["oof_*.npy", "*_oof.npy", "*_pred_oof.npy", "*_resid_oof.npy"]
found = set()
for pat in patterns:
    for p in glob.glob(os.path.join(PROC, pat)):
        found.add(p)

print(f"Initial candidates: {len(found)}", flush=True)

manifest = []
skipped = {"wrong_shape": 0, "no_te_match": 0, "te_wrong_shape": 0, "load_fail": 0}

for tr_path in sorted(found):
    stem = os.path.splitext(os.path.basename(tr_path))[0]
    try:
        arr = np.load(tr_path, allow_pickle=False)
    except Exception:
        skipped["load_fail"] += 1
        continue
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape != (4139,):
        skipped["wrong_shape"] += 1
        continue
    te_path = find_test_match(stem)
    if te_path is None:
        skipped["no_te_match"] += 1
        continue
    try:
        te_arr = np.load(te_path, allow_pickle=False)
    except Exception:
        skipped["load_fail"] += 1
        continue
    if te_arr.ndim == 2 and te_arr.shape[1] == 1:
        te_arr = te_arr[:, 0]
    if te_arr.shape != (513,):
        skipped["te_wrong_shape"] += 1
        continue
    tr_rae = rae(y_tr, arr)
    te_rae = rae(y_unb, te_arr[unb_idx])
    manifest.append({
        "name": stem,
        "train_oof_path": tr_path,
        "te_path": te_path,
        "train_oof_rae": tr_rae,
        "te_unblind_rae": te_rae,
        "tr_std": float(np.nanstd(arr)),
        "te_std": float(np.nanstd(te_arr)),
    })

df = pd.DataFrame(manifest)
print(f"\nPaired predictors: {len(df)}", flush=True)
print(f"Skipped: {skipped}", flush=True)

# Sort by train OOF RAE asc
df = df.sort_values("train_oof_rae")
out_csv = os.path.join(PROC, "_audit_oof_pairs.csv")
df.to_csv(out_csv, index=False)
print(f"Saved manifest -> {out_csv}")

# top-15 by lowest train_oof_rae
top = df.head(15).copy()
top["gap_te_minus_tr"] = top["te_unblind_rae"] - top["train_oof_rae"]
print("\nTop-15 by lowest train_oof_rae:")
print(top[["name", "train_oof_rae", "te_unblind_rae", "gap_te_minus_tr", "tr_std", "te_std"]].to_string(index=False))

# flag suspicious pairs: tr_rae much lower than te_rae
df["gap"] = df["te_unblind_rae"] - df["train_oof_rae"]
flagged = df[df["gap"] > 0.30].sort_values("gap", ascending=False)
print(f"\nFlagged (te - tr > 0.30): {len(flagged)}")
print(flagged[["name", "train_oof_rae", "te_unblind_rae", "gap", "tr_std", "te_std"]].head(20).to_string(index=False))
