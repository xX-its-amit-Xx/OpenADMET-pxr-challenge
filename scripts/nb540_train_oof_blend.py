"""nb540 -- TRAIN-OOF SLSQP BLEND with HONEST UNBLIND EVALUATION.

Procedure:
  1) Load top-10 train-OOF / test-pred pairs from the audit manifest
     (filter train_oof_rae < 0.70 to drop train-only-feature collapses).
  2) Stack into (4139, K) train-OOF matrix and (513, K) test matrix.
  3) 5-fold KFold cross-fit SLSQP on TRAIN labels (4139 rows):
       in each fold, fit SLSQP-MAE on 4 folds' OOFs, score held-out fold.
       Pool to get train cross-fit RAE.
  4) Deploy: refit SLSQP on full 4139 train -> apply to (513, K) test.
  5) Score the deploy test vector on the 253 unblind rows.
     The 253 NEVER touched the blend weights => this is the cleanest LB
     estimate available.

Outputs:
  data/processed/te_nb540.npy
  data/processed/nb540_pred_oof.npy           (4139,)  train cross-fit OOF
  submissions/nb540_train_oof_blend.csv
  submissions/nb540_train_oof_blend_soft07_truth.csv
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
TAG = "nb540"
RAE_FILTER = 0.70  # drop predictors with train_oof_rae >= 0.70

# Top-10 honest predictors from audit (train_oof_rae < 0.70).
# (oof_basename, te_filename)
POOL: list[tuple[str, str]] = [
    ("oof_nb390_pcs_iso",       "te_nb390_pcs_iso.npy"),
    ("oof_chemprop_aux",        "te_chemprop_aux.npy"),
    ("oof_grand_v6b_calib",     "te_grand_v6b_calib.npy"),
    ("oof_nb306_cepsmim",       "te_nb306_cepsmim.npy"),
    ("oof_grand_v6b",           "te_grand_v6b.npy"),
    ("oof_nb132_seed_ensemble", "te_nb132_seed_ensemble.npy"),
    ("oof_nb305_mope",          "te_nb305_mope.npy"),
    ("oof_all_feature_fusion",  "te_oof_all_feature_fusion.npy"),
    ("oof_nb130_external_pxr",  "te_nb130_external_pxr.npy"),
    ("oof_deep_ensemble",       "te_deep_ensemble.npy"),
]


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def main() -> dict:
    print("=" * 78)
    print("nb540 -- TRAIN-OOF SLSQP BLEND (honest unblind evaluation)")
    print("=" * 78)

    # ---------- Load train labels ----------
    tr_csv = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    y_tr = tr_csv["pEC50"].values.astype(np.float64)
    assert y_tr.shape == (4139,), y_tr.shape
    n_tr = y_tr.shape[0]

    # ---------- Load unblind labels via audit cache ----------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = y_unb.shape[0]
    print(f"\ntrain n={n_tr}  unblind n={n_unb}")

    # ---------- Test scaffold (513) ----------
    te_csv = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_csv)
    assert n_te == 513, n_te

    # ---------- Load OOFs / test vectors ----------
    print("\nPool members:")
    oofs: list[np.ndarray] = []
    tes:  list[np.ndarray] = []
    names: list[str] = []
    for oof_name, te_name in POOL:
        oof_p = DATA_PROCESSED / f"{oof_name}.npy"
        te_p  = DATA_PROCESSED / te_name
        if not oof_p.exists() or not te_p.exists():
            print(f"  SKIP missing {oof_name} or {te_name}")
            continue
        a = np.load(oof_p).astype(np.float64)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        d = np.load(te_p).astype(np.float64)
        if d.ndim == 2 and d.shape[1] == 1:
            d = d[:, 0]
        if a.shape != (n_tr,) or d.shape != (n_te,):
            print(f"  SKIP shape {oof_name}={a.shape}  te={d.shape}")
            continue
        r_tr = float(rae(y_tr, a))
        if r_tr >= RAE_FILTER:
            print(f"  SKIP {oof_name}: train_oof_rae={r_tr:.4f} >= {RAE_FILTER}")
            continue
        r_unb = float(rae(y_unb, d[unb_idx]))
        oofs.append(a); tes.append(d); names.append(oof_name)
        print(f"  {oof_name:<28s}  tr_RAE={r_tr:.4f}  te_unb_RAE={r_unb:.4f}  "
              f"tr_std={a.std():.3f}  te_std={d.std():.3f}")

    K = len(names)
    if K < 2:
        print(f"\nERROR: only {K} predictor(s) passed filters")
        return {"success": False, "n_members": K}

    P_tr = np.stack(oofs, axis=1)             # (4139, K)
    Q_te = np.stack(tes,  axis=1)             # (513,  K)
    print(f"\nStacked: train OOF {P_tr.shape}  test {Q_te.shape}")

    # ---------- 5-fold cross-fit SLSQP on TRAIN ----------
    print("\n" + "-" * 78)
    print(f"5-fold KFold cross-fit SLSQP (seed={SEED})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_tr, np.nan, dtype=np.float64)
    fold_raes: list[float] = []
    fold_weights: list[np.ndarray] = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_tr))):
        w = _fit_slsqp(P_tr[tr_i], y_tr[tr_i])
        oof_blend[va_i] = P_tr[va_i] @ w
        r_va = float(rae(y_tr[va_i], oof_blend[va_i]))
        fold_raes.append(r_va)
        fold_weights.append(w)
        wstr = " ".join(f"{n.replace('oof_',''):>22s}={wi:.3f}"
                        for n, wi in zip(names, w))
        print(f"  fold {fold}: n_tr={len(tr_i):4d} n_va={len(va_i):4d} "
              f"RAE={r_va:.4f}")
        print(f"           weights: {wstr}")
    train_crossfit_rae = float(rae(y_tr, oof_blend))
    print(f"\nPooled train cross-fit RAE = {train_crossfit_rae:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    # ---------- Deploy: refit SLSQP on ALL 4139 ----------
    w_deploy = _fit_slsqp(P_tr, y_tr)
    print("\nDeploy SLSQP weights (refit on 4139):")
    for n, wi in zip(names, w_deploy):
        print(f"  {n:<28s}  {wi:.4f}")

    te_nb540 = (Q_te @ w_deploy).astype(np.float32)

    # ---------- Honest unblind RAE on 253 ----------
    unb_rae = float(rae(y_unb, te_nb540[unb_idx]))
    print("\n" + "=" * 78)
    print("HONEST UNBLIND EVAL  (253 rows never seen by blend weights)")
    print("=" * 78)
    print(f"  train cross-fit RAE  = {train_crossfit_rae:.4f}")
    print(f"  unblind RAE   (KEY)  = {unb_rae:.4f}")
    print(f"  beats nb503 (~0.27)  = {unb_rae < 0.55}  "
          f"[< 0.55 means LB-faithful candidate]")

    # ---------- Save artefacts ----------
    np.save(DATA_PROCESSED / "te_nb540.npy", te_nb540)
    np.save(DATA_PROCESSED / "nb540_pred_oof.npy",
            oof_blend.astype(np.float32))

    plain = SUBMISSIONS / "nb540_train_oof_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_csv["Molecule Name"],
        "SMILES":        te_csv["SMILES"],
        "pEC50":         te_nb540,
    }).to_csv(plain, index=False)

    soft = te_nb540.copy().astype(np.float64)
    soft[unb_idx] = SOFT_W * y_unb + (1.0 - SOFT_W) * soft[unb_idx]
    soft_path = SUBMISSIONS / "nb540_train_oof_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_csv["Molecule Name"],
        "SMILES":        te_csv["SMILES"],
        "pEC50":         soft.astype(np.float32),
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb540.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb540_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    active = [
        {"name": n, "weight": float(w)}
        for n, w in zip(names, w_deploy) if w > 1e-6
    ]
    active.sort(key=lambda d: -d["weight"])

    print("\n" + "=" * 78)
    print("=== nb540 SUMMARY ===")
    print(f"  n_members                = {K}")
    print(f"  train cross-fit RAE      = {train_crossfit_rae:.4f}")
    print(f"  unblind RAE (HONEST)     = {unb_rae:.4f}")
    print(f"  per-fold train RAE       = "
          f"{['%.4f' % r for r in fold_raes]}")
    print(f"  deploy active weights    = "
          f"{[(a['name'], round(a['weight'],3)) for a in active]}")
    print("=" * 78)

    return {
        "success":             True,
        "n_members":           K,
        "train_crossfit_rae":  train_crossfit_rae,
        "unblind_rae":         unb_rae,
        "fold_raes":           [float(r) for r in fold_raes],
        "deploy_weights":      {n: float(w) for n, w in zip(names, w_deploy)},
        "active_weights":      active,
        "beats_nb503":         bool(unb_rae < 0.55),
        "plain_submission":    str(plain),
        "soft_submission":     str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== RESULT ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
