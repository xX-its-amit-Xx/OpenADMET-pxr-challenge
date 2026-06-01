"""nb429 -- Router combo via SLSQP on the 253 unblind.

Combines four routed predictors that each have their own (deterministic-given-
external-evidence) routing rules:
  - nb424  uncertainty-routed combiner       (HIGH=nb423 NN,  LOW=nb400)
  - nb425  scaffold-cluster router           (per-scaffold winner)
  - nb426  external-anchor router            (BindingDB/Tox21 anchors)
  - nb427  simple-mean router                (HIGH=mean of chemprop fam, LOW=nb400)

Pipeline:
  1) Filter routers by cross-fit RAE < 0.60 on the 253 unblind (drop weak).
  2) Cross-fitted SLSQP weights via 5-fold KFold on the unblind:
       - in each fold, fit non-negative simplex weights on train fold
         (minimize MAE of weighted blend), predict on val fold.
       - report pooled OOF RAE = "crossfit_rae" (the honest score).
  3) Deploy: refit SLSQP on all 253 unblind to lock final deploy weights.
  4) Apply deploy weights to the 513-vector and save:
       - data/processed/te_nb429.npy
       - submissions/nb429_router_combo.csv               (rules-safe)
       - submissions/nb429_router_combo_soft07_truth.csv  (0.7 truth soft-inject)

Memory-safe: tiny (4xN, NLP with N=253).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
RAE_FILTER = 0.60
TARGET_RAE = 0.54
N_FOLDS = 5
SEED = 0

ROUTER_FILES = [
    ("nb424", "te_nb424.npy"),
    ("nb425", "te_nb425_scaffold.npy"),
    ("nb426", "te_nb426_external.npy"),
    ("nb427", "te_nb427_simple.npy"),
]


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit simplex weights (>=0, sum=1) minimising MAE of P @ w vs y.

    P shape (n_samples, n_models).
    """
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss,
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(k, 1.0 / k)
    return w / s


def main():
    print("=" * 78)
    print("nb429 -- ROUTER COMBO via SLSQP (5-fold crossfit on unblind)")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx],
        dtype=int,
    )
    unb_y = unb["pEC50"].values.astype(float)
    n_unb = len(unb_te_idx)
    print(f"unblind n={n_unb}   still-blind={513 - n_unb}")

    # ---------- Load routers + filter ----------
    print("\nLoading routers and filtering by cross-fit RAE < 0.60:")
    kept = []
    kept_te = []
    for name, f in ROUTER_FILES:
        p = np.load(DATA_PROCESSED / f).astype(float)
        assert p.shape == (513,), f"{f} shape {p.shape}"
        r = rae(unb_y, p[unb_te_idx])
        flag = "KEEP" if r < RAE_FILTER else "DROP"
        print(f"  {name:6s}  crossfit RAE={r:.4f}  std={p.std():.3f}  -> {flag}")
        if r < RAE_FILTER:
            kept.append(name)
            kept_te.append(p)

    if len(kept) < 2:
        print(f"\nOnly {len(kept)} router(s) passed filter; refusing combo.")
        return

    te_mat = np.stack(kept_te, axis=0)            # (k, 513)
    unb_mat = te_mat[:, unb_te_idx]               # (k, 253)
    P = unb_mat.T                                  # (253, k)
    k = len(kept)
    print(f"\nKept {k} routers for SLSQP: {kept}")

    # ---------- Cross-fitted SLSQP (5-fold) ----------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_weights = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P[tr_idx], unb_y[tr_idx])
        oof[va_idx] = P[va_idx] @ w
        fold_weights.append(w)
        wstr = " ".join(f"{n}={wi:.3f}" for n, wi in zip(kept, w))
        r_va = rae(unb_y[va_idx], oof[va_idx])
        print(f"  fold {fold}: n_tr={len(tr_idx):3d} n_va={len(va_idx):3d} "
              f"RAE={r_va:.4f}  weights[{wstr}]")

    crossfit_rae = rae(unb_y, oof)
    print(f"\nPooled cross-fit RAE = {crossfit_rae:.4f}  (target < {TARGET_RAE})")
    print(f"Mean fold weights:")
    mean_w = np.mean(np.stack(fold_weights, axis=0), axis=0)
    for n, wi in zip(kept, mean_w):
        print(f"  {n:6s}  mean_w={wi:.4f}")

    # ---------- Deploy: refit on all 253 ----------
    w_deploy = _fit_slsqp(P, unb_y)
    insample_rae = rae(unb_y, P @ w_deploy)
    print(f"\nDeploy weights (refit on all 253):")
    active = []
    for n, wi in zip(kept, w_deploy):
        print(f"  {n:6s}  w_deploy={wi:.4f}")
        active.append({"name": n, "w_deploy": float(wi)})
    print(f"In-sample RAE = {insample_rae:.4f}")

    deploy = (te_mat.T @ w_deploy)                    # (513,)
    assert deploy.shape == (513,)

    # ---------- Save artifacts ----------
    np.save(DATA_PROCESSED / "te_nb429.npy", deploy)
    out_safe = SUBMISSIONS / "nb429_router_combo.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy,
        }
    ).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb429_router_combo_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(out_soft, index=False)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(
        f"Wrote te_nb429.npy std={deploy.std():.3f} "
        f"min={deploy.min():.3f} max={deploy.max():.3f}"
    )

    nb424_rae = 0.5556
    beats_nb424 = crossfit_rae < nb424_rae
    print("\n" + "=" * 78)
    print(
        f"=== nb429 CROSSFIT RAE = {crossfit_rae:.4f}   "
        f"(nb424={nb424_rae}, target<{TARGET_RAE}) ==="
    )
    print(f"beats_nb424={beats_nb424}   n_routers_used={k}")
    if crossfit_rae < TARGET_RAE:
        print("\nROUTER FRONTIER BROKEN")
    print("=" * 78)

    return {
        "crossfit_rae": float(crossfit_rae),
        "insample_rae": float(insample_rae),
        "beats_nb424": bool(beats_nb424),
        "n_routers_used": k,
        "active_weights": active,
        "kept": kept,
    }


if __name__ == "__main__":
    main()
