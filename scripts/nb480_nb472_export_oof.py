"""nb480 -- Export nb472 honest cross-fit OOF on the 253 unblind rows.

nb472 saved te_nb472.npy (513-row deploy refit using all 253 unblind for the
residual LGBM). For honest blending we need the OOF predictions that nb472
internally computed (`resid_oof` / `nb472_unb_oof`) but never persisted.

This script re-runs the nb472 protocol exactly (same 18-col feature matrix,
same shallow LGBM, same KFold seed as nb472 -> SEED=0) and persists:

  data/processed/nb472_resid_oof.npy   (253,) cross-fit residual predictions
  data/processed/nb472_pred_oof.npy    (253,) nb432_253 + alpha_253 * resid_oof
  data/processed/nb472_unblind_idx.npy (253,) mapping to the 513-row test index

Verifies cross-fit RAE on nb472_pred_oof ~= 0.5410.
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

# Reuse the nb472 feature builder verbatim so we are guaranteed identical X.
from nb472_residual_stack_router import build_feature_matrix, sigmoid

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW

RDLogger.DisableLog("rdApp.*")

# Match nb472 EXACTLY so the cross-fit RAE reproduces 0.5410.
SEED = 0
N_FOLDS = 5
SCALE = 4.0


def main() -> dict:
    print("=" * 78)
    print("nb480 -- Export nb472 honest cross-fit OOF (253 unblind)")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---------- Same 18-col feature matrix as nb443/nb472 ----------
    print("\nBuilding feature matrix (18 cols, matching nb443/nb472)...")
    X, feat_names = build_feature_matrix(te_df, nb432)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")

    X_unb = X[unb_te_idx]
    # Signed residual target: positive => truth > nb432
    y_resid = (unb_y - nb432[unb_te_idx]).astype(np.float32)
    print(
        f"\nResidual target: mean={y_resid.mean():+.3f}  "
        f"std={y_resid.std():.3f}  |mean|={np.abs(y_resid).mean():.3f}"
    )

    # ---------- 5-fold cross-fit shallow LGBM (identical to nb472) ----------
    params = dict(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=8,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
        n_jobs=2,
    )
    print(f"\n5-fold KFold (shuffle=True, random_state={SEED}) shallow LGBM:")
    print(f"  params = {params}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(
            f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
            f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
            f"std={resid_oof[va_i].std():.3f}"
        )

    # ---------- Soft-gate by err_hat (same recipe as nb472) ----------
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    alpha_253 = alpha_513[unb_te_idx]

    # ---------- Honest blended prediction on the 253 unblind rows ----------
    pred_oof_253 = (nb432[unb_te_idx] + alpha_253 * resid_oof).astype(np.float32)

    rae_nb432_unb = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb472_oof = float(rae(unb_y, pred_oof_253))
    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline           = {rae_nb432_unb:.4f}")
    print(f"  nb472 cross-fit (honest) = {rae_nb472_oof:.4f}  (target ~0.5410)")

    # ---------- Persist OOF artefacts ----------
    out_resid = DATA_PROCESSED / "nb472_resid_oof.npy"
    out_pred = DATA_PROCESSED / "nb472_pred_oof.npy"
    out_idx = DATA_PROCESSED / "nb472_unblind_idx.npy"
    np.save(out_resid, resid_oof)
    np.save(out_pred, pred_oof_253)
    np.save(out_idx, unb_te_idx.astype(np.int64))
    print(f"\nWrote {out_resid}  shape={resid_oof.shape}")
    print(f"Wrote {out_pred}  shape={pred_oof_253.shape}")
    print(f"Wrote {out_idx}  shape={unb_te_idx.shape}")

    # ---------- Verify match to nb472's reported 0.5410 ----------
    target = 0.5410
    delta = abs(rae_nb472_oof - target)
    match = delta < 0.005
    print(
        f"\nCross-fit RAE verify: got {rae_nb472_oof:.4f}  "
        f"target {target}  delta={delta:.4f}  match={match}"
    )

    return {
        "success": True,
        "n_unb": int(n_unb),
        "rae_nb432_unb": rae_nb432_unb,
        "crossfit_rae_nb472": rae_nb472_oof,
        "crossfit_rae_target": target,
        "verify_match": bool(match),
        "resid_oof_path": str(out_resid),
        "pred_oof_path": str(out_pred),
        "unblind_idx_path": str(out_idx),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
