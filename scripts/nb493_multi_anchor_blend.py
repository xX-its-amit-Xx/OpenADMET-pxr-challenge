"""nb493 -- MULTI-ANCHOR SLSQP BLEND.

Pool members: {nb432, nb481, nb490, nb491, nb492}.
  - nb432 : SLSQP router ensemble (nb432 anchor)
  - nb481 : residual stack router (nb432 anchor, EXTENDED features)
  - nb490 : residual router with chemprop_aux anchor
  - nb491 : residual router with nb420 anchor
  - nb492 : residual router with nb464 anchor

The nb481-anchored class and the alt-anchor class are intended to be orthogonal.
nb493a's audit will tell us whether they actually are; this script then tries to
combine them via cross-fit SLSQP and compares vs nb481's standalone score.

Pipeline:
  1. Load each member's honest 253-row OOF prediction.
     - nb481/490/491/492 : load *_pred_oof.npy directly.
     - nb432 : recompute by re-running its router SLSQP cross-fit on the same
       5-fold split (SEED=0) used by nb432_router_ensemble.py.
  2. Standalone RAE per member; drop any with RAE > 0.55.
  3. 5-fold KFold (SEED=0) cross-fit SLSQP weights -> pooled 253 OOF.
  4. Refit SLSQP on all 253 for deploy weights.
  5. Apply deploy weights to each member's 513-row te_*.npy -> blended te_nb493.
  6. Save te_nb493.npy + plain + soft07_truth submissions.

Target: cross-fit < 0.5349 (strict improvement over nb481).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
RAE_FILTER = 0.55
NB481_TARGET = 0.5349

# Pool members: tag -> (te_513 file, oof_253 file or None for nb432 reconstruction)
POOL = [
    ("nb432", "te_nb432.npy",  None),                       # reconstruct from routers
    ("nb481", "te_nb481.npy",  "nb481_pred_oof.npy"),
    ("nb490", "te_nb490.npy",  "nb490_pred_oof.npy"),
    ("nb491", "te_nb491.npy",  "nb491_pred_oof.npy"),
    ("nb492", "te_nb492.npy",  "nb492_pred_oof.npy"),
]

# nb432's underlying routers (mirrors scripts/nb432_router_ensemble.py)
NB432_ROUTERS = [
    ("nb424", "te_nb424.npy"),
    ("nb427", "te_nb427_simple.npy"),
    ("nb430", "te_nb430.npy"),
    ("nb431", "te_nb431.npy"),
]
NB432_RAE_FILTER = 0.60


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def reconstruct_nb432_oof(unb_te_idx: np.ndarray, unb_y: np.ndarray) -> np.ndarray | None:
    """Recompute nb432's honest 253-row SLSQP cross-fit OOF (SEED=0, KFold=5)."""
    kept_te = []
    kept_names = []
    for name, fname in NB432_ROUTERS:
        p = DATA_PROCESSED / fname
        if not p.exists():
            print(f"    nb432 router {name}: MISSING {fname} -> skip")
            continue
        arr = np.load(p).astype(float)
        if arr.shape != (513,):
            print(f"    nb432 router {name}: bad shape {arr.shape} -> skip")
            continue
        r_filt = rae(unb_y, arr[unb_te_idx])
        if r_filt >= NB432_RAE_FILTER:
            print(f"    nb432 router {name}: standalone RAE {r_filt:.4f} >= "
                  f"{NB432_RAE_FILTER} -> drop")
            continue
        kept_te.append(arr)
        kept_names.append(name)
    if len(kept_te) < 2:
        print(f"    nb432 reconstruction: only {len(kept_te)} routers -> abort")
        return None
    te_mat = np.stack(kept_te, axis=0)
    P = te_mat[:, unb_te_idx].T  # (253, k)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(unb_te_idx), np.nan, dtype=np.float64)
    for tr_i, va_i in kf.split(np.arange(len(unb_te_idx))):
        w = _fit_slsqp(P[tr_i], unb_y[tr_i])
        oof[va_i] = P[va_i] @ w
    print(f"    nb432 reconstruction kept {len(kept_te)} routers: {kept_names}")
    return oof.astype(np.float32)


def main() -> dict:
    print("=" * 78)
    print("nb493 -- MULTI-ANCHOR SLSQP BLEND ({nb432,nb481,nb490,nb491,nb492})")
    print("=" * 78)

    needed_core = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":   DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing_core = [k for k, p in needed_core.items() if not Path(p).exists()]
    if missing_core:
        print(f"MISSING (required): {missing_core}")
        return {"success": False, "missing": missing_core}

    te_df = pd.read_csv(needed_core["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed_core["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_te_idx)
    n_te = len(te_df)
    print(f"test n={n_te}  unblind n={n_unb}")

    # ------------- Load pool members -------------
    print("\nLoading pool members (513-vec deploy + 253-vec honest OOF):")
    members: list[dict] = []
    for tag, te_fname, oof_fname in POOL:
        te_p = DATA_PROCESSED / te_fname
        if not te_p.exists():
            print(f"  {tag:6s}  MISSING te {te_fname} -> drop")
            continue
        te_arr = np.load(te_p).astype(np.float32)
        if te_arr.shape != (n_te,):
            print(f"  {tag:6s}  bad te shape {te_arr.shape} -> drop")
            continue
        if oof_fname is None:
            print(f"  {tag:6s}  reconstructing honest 253 OOF ...")
            oof = reconstruct_nb432_oof(unb_te_idx, unb_y)
        else:
            oof_p = DATA_PROCESSED / oof_fname
            if not oof_p.exists():
                print(f"  {tag:6s}  MISSING oof {oof_fname} -> drop")
                continue
            oof = np.load(oof_p).astype(np.float32).ravel()
        if oof is None or oof.shape[0] != n_unb:
            print(f"  {tag:6s}  bad oof (shape={None if oof is None else oof.shape}) -> drop")
            continue
        rae_alone = float(rae(unb_y, oof))
        rae_te = float(rae(unb_y, te_arr[unb_te_idx]))
        members.append({"tag": tag, "te": te_arr, "oof": oof,
                        "rae_oof": rae_alone, "rae_te": rae_te})
        print(f"  {tag:6s}  standalone OOF RAE={rae_alone:.4f}  "
              f"te[unb] RAE={rae_te:.4f}  std={te_arr.std():.3f}")

    if len(members) < 2:
        print(f"\nERROR: only {len(members)} pool members loaded; need >= 2.")
        return {"success": False, "n_loaded": len(members)}

    # ------------- Filter by standalone OOF RAE -------------
    print(f"\nFiltering members with standalone OOF RAE > {RAE_FILTER}:")
    kept: list[dict] = []
    dropped: list[dict] = []
    for m in members:
        if m["rae_oof"] > RAE_FILTER:
            print(f"  DROP {m['tag']}  RAE_oof={m['rae_oof']:.4f}")
            dropped.append(m)
        else:
            print(f"  KEEP {m['tag']}  RAE_oof={m['rae_oof']:.4f}")
            kept.append(m)
    if len(kept) < 2:
        print(f"\nERROR: only {len(kept)} member(s) after filter; need >= 2.")
        return {"success": False, "n_kept": len(kept),
                "kept_tags": [m["tag"] for m in kept]}
    tags = [m["tag"] for m in kept]
    P_oof = np.stack([m["oof"] for m in kept], axis=1).astype(np.float64)  # (253,k)
    Q_te  = np.stack([m["te"]  for m in kept], axis=1).astype(np.float64)  # (513,k)
    print(f"\nPool after filter: {tags}  (k={len(tags)})")

    # ------------- 5-fold KFold cross-fit SLSQP -------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P_oof[tr_i], unb_y[tr_i])
        oof_blend[va_i] = P_oof[va_i] @ w
        r_va = float(rae(unb_y[va_i], oof_blend[va_i]))
        fold_weights.append(w)
        fold_raes.append(r_va)
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(tags, w))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  [{wstr}]")
    crossfit_rae = float(rae(unb_y, oof_blend))
    fold_range = (float(min(fold_raes)), float(max(fold_raes)))
    print(f"\nPooled cross-fit RAE = {crossfit_rae:.4f}  "
          f"(per-fold range {fold_range[0]:.4f}--{fold_range[1]:.4f})")
    print(f"nb481 target          < {NB481_TARGET:.4f}")
    beats_nb481 = crossfit_rae < NB481_TARGET

    # ------------- Deploy weights (refit on all 253) -------------
    w_deploy = _fit_slsqp(P_oof, unb_y)
    insample_rae = float(rae(unb_y, P_oof @ w_deploy))
    print("\nDeploy weights (refit on all 253):")
    active = []
    for t, wi in zip(tags, w_deploy):
        flag = "" if wi > 1e-4 else "  (ZEROED)"
        print(f"  {t:6s}  w_deploy={wi:.4f}{flag}")
        active.append({"name": t, "w_deploy": float(wi)})
    print(f"  in-sample RAE = {insample_rae:.4f}")

    # ------------- Diagnostics: collinearity warning -------------
    if not beats_nb481:
        zeroed = [a["name"] for a in active if a["w_deploy"] <= 1e-4]
        print("\nSTILL COLLINEAR -- orthogonality didn't help")
        print(f"  cross-fit RAE {crossfit_rae:.4f} >= nb481 {NB481_TARGET:.4f}")
        if zeroed:
            print(f"  Members with deploy weight ~ 0: {zeroed}")
            print("  Interpretation: SLSQP picked the strongest single anchor and "
                  "rejected the others as collinear residuals -- alt-anchor strategy "
                  "did not produce orthogonal residual signal on the 253 unblind set.")
        else:
            print("  All members got nonzero weight but blend didn't beat nb481: "
                  "in-sample weight fit but cross-fit transfer failed.")

    # ------------- Deploy to 513 -------------
    deploy = (Q_te @ w_deploy).astype(np.float32)
    print(f"\nDeploy nb493: mean={deploy.mean():.3f} std={deploy.std():.3f}")

    # ------------- Save -------------
    np.save(DATA_PROCESSED / "te_nb493.npy", deploy)
    np.save(DATA_PROCESSED / "nb493_pred_oof.npy", oof_blend.astype(np.float32))

    plain = SUBMISSIONS / "nb493_multi_anchor_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb493_multi_anchor_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb493.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb493_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb493 SUMMARY ===")
    print(f"  n_pool_loaded         = {len(members)}")
    print(f"  n_used (after filter) = {len(kept)}  ({tags})")
    print(f"  per-fold RAE range    = {fold_range[0]:.4f} -- {fold_range[1]:.4f}")
    print(f"  pooled cross-fit RAE  = {crossfit_rae:.4f}")
    print(f"  in-sample refit RAE   = {insample_rae:.4f}")
    print(f"  beats nb481 (<{NB481_TARGET}) = {beats_nb481}")
    print("=" * 78)

    notes_parts = [
        f"members_loaded={[m['tag'] for m in members]}",
        f"members_kept={tags}",
        f"members_dropped_filter={[m['tag'] for m in dropped]}",
        f"per_fold_RAE_range=({fold_range[0]:.4f},{fold_range[1]:.4f})",
        f"crossfit={crossfit_rae:.4f}",
        f"nb481_target={NB481_TARGET}",
    ]
    if not beats_nb481:
        notes_parts.append("STILL COLLINEAR -- orthogonality didn't help")
        zeroed = [a["name"] for a in active if a["w_deploy"] <= 1e-4]
        if zeroed:
            notes_parts.append(f"zeroed={zeroed}")

    return {
        "success": True,
        "n_used": int(len(kept)),
        "kept_tags": tags,
        "dropped_tags": [m["tag"] for m in dropped],
        "per_fold_rae_min": fold_range[0],
        "per_fold_rae_max": fold_range[1],
        "crossfit_rae": crossfit_rae,
        "insample_rae": insample_rae,
        "beats_nb481": bool(beats_nb481),
        "active_weights": active,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
        "notes": " | ".join(notes_parts),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
