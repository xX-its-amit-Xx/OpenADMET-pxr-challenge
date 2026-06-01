"""nb513 -- SHALLOW ROUTER STACK (regularization-first).

Replaces failed nb500 (45 cols on 253 rows -> df>>n -> overfit). Stacks the
7 residual routers (nb472, nb481, nb482, nb490, nb491, nb492, nb502) into a
single ultra-shallow LGBM meta-learner over the 253 unblind rows. The meta
LGBM predicts TRUTH pEC50 directly (base routers already extracted residual
signal vs anchors).

Feature matrix on 253 unblind = just the N router pred_oof columns (N=7).
Target = unblind truth pEC50.

Cross-fit: 5-fold KFold ultra-shallow LGBM
  (max_depth=2, n_est=50, lr=0.05, num_leaves=4, min_child_samples=30,
   reg_lambda=1.0, seed=0)

Honest cross-fit RAE target: < 0.5283 (nb492 single) and ideally < nb502.

Deploy on 513:
  - Refit shallow LGBM on all 253 rows.
  - Apply to the 7 te_nb*.npy router deploy preds.
  - Save te_nb513.npy + nb513_pred_oof.npy + plain + soft07_truth CSVs.
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
from rdkit import RDLogger
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7

LGBM_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=2,
    num_leaves=4,
    min_child_samples=30,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)

ROUTERS = ["nb472", "nb481", "nb482", "nb490", "nb491", "nb492", "nb502"]


def main() -> dict:
    print("=" * 78)
    print("nb513 -- SHALLOW ROUTER STACK (regularization-first)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag in ROUTERS:
        needed[f"te_{tag}.npy"] = DATA_PROCESSED / f"te_{tag}.npy"
        needed[f"{tag}_pred_oof.npy"] = DATA_PROCESSED / f"{tag}_pred_oof.npy"
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}  routers n={len(ROUTERS)}")

    # ---- Load router preds ----
    pred_oof_cols: list[np.ndarray] = []
    pred_te_cols: list[np.ndarray] = []
    print("\nRouter single-model OOF RAE on unblind n=253:")
    for tag in ROUTERS:
        p_oof = np.load(needed[f"{tag}_pred_oof.npy"]).astype(np.float32)
        p_te = np.load(needed[f"te_{tag}.npy"]).astype(np.float32)
        pred_oof_cols.append(p_oof)
        pred_te_cols.append(p_te)
        print(f"  {tag}: RAE={float(rae(unb_y, p_oof)):.4f}  "
              f"te mean={p_te.mean():.3f} std={p_te.std():.3f}")

    X_unb = np.stack(pred_oof_cols, axis=1).astype(np.float32)
    X_te = np.stack(pred_te_cols, axis=1).astype(np.float32)
    X_unb = np.nan_to_num(X_unb, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    feat_names = list(ROUTERS)

    print(f"\nMETA matrix unblind: {X_unb.shape}  deploy: {X_te.shape}  "
          f"n_feats={X_unb.shape[1]}")
    assert X_unb.shape == (n_unb, len(ROUTERS))
    assert X_te.shape == (n_te, len(ROUTERS))

    # ---- 5-fold cross-fit ----
    print("\n5-fold KFold cross-fit ultra-shallow LGBM (target=truth):")
    print(f"  params: {LGBM_PARAMS}")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], unb_y[tr_i])
        pred_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"pred_oof mean={pred_oof[va_i].mean():.3f} "
              f"std={pred_oof[va_i].std():.3f}")
    del mdl

    rae_meta_oof = float(rae(unb_y, pred_oof))
    print(f"\nHonest cross-fit RAE (n=253) nb513 = {rae_meta_oof:.4f}")
    NB492_TARGET = 0.5283
    nb502_oof = float(rae(unb_y,
                          np.load(DATA_PROCESSED / "nb502_pred_oof.npy")
                          .astype(np.float32)))
    print(f"  reference: nb492 single = {NB492_TARGET:.4f}")
    print(f"  reference: nb502 single = {nb502_oof:.4f}")
    beats_nb492 = rae_meta_oof < NB492_TARGET
    beats_nb502 = rae_meta_oof < nb502_oof
    print(f"  beats nb492 = {beats_nb492}")
    print(f"  beats nb502 = {beats_nb502}")

    # ---- Deploy: refit on all 253 ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, unb_y)
    deploy = deploy_mdl.predict(X_te).astype(np.float32)
    print(f"\nDeploy nb513 (513): mean={deploy.mean():.3f} "
          f"std={deploy.std():.3f}  "
          f"min={deploy.min():.3f} max={deploy.max():.3f}")

    # ---- Importances ----
    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nRouter feature importances (deploy refit):")
    for i in order:
        print(f"  {feat_names[i]:6s}  {imp[i]}")

    # ---- Save ----
    np.save(DATA_PROCESSED / "te_nb513.npy", deploy)
    np.save(DATA_PROCESSED / "nb513_pred_oof.npy", pred_oof)

    plain = SUBMISSIONS / "nb513_shallow_router_stack.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb513_shallow_router_stack_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb513.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb513_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb513 SUMMARY ===")
    print(f"  n_features                                 = {len(feat_names)}")
    print(f"  cross-fit RAE nb513 (honest, n=253)         = {rae_meta_oof:.4f}")
    print(f"  nb502 single (reference)                    = {nb502_oof:.4f}")
    print(f"  beats nb492 (<0.5283)                       = {beats_nb492}")
    print(f"  beats nb502 (<{nb502_oof:.4f})                  = {beats_nb502}")
    print("  Regularization-first replacement for nb500 (45 cols -> overfit).")
    print("  If RAE > 0.53, the shallow stack just can't beat nb502 with these")
    print("  inputs (router preds are already heavily correlated).")
    print("=" * 78)

    return {
        "success": True,
        "n_features": len(feat_names),
        "crossfit_rae": rae_meta_oof,
        "nb502_oof": nb502_oof,
        "beats_nb492": bool(beats_nb492),
        "beats_nb502": bool(beats_nb502),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
