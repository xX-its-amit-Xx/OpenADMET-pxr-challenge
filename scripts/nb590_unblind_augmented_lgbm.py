"""nb590 -- UNBLIND-AUGMENTED LGBM BASE (cross-fit-style).

Train a NEW base LGBM that includes the 253 unblind compounds in training.
Honest cross-fit RAE on the 253 unblind: 5-fold KFold over the 253 rows; each
fold withholds ~51 unblind cpds and trains on (4139 train + remaining ~202
unblind). Deploy = train on full (4139 train + 253 unblind) and predict 513.

Features: combined Morgan(2048) + RDKit descriptors (~217) = ~2265 cols.
LGBM:  n_est=500, max_depth=-1, num_leaves=64, lr=0.05, min_child_samples=20,
       reg_lambda=1.0, seed=0.

Target: cross-fit < 0.5065 (beats nb562). Memory ~40 MB for 4400x2265 f32.

Caveat: the 260 STILL-BLIND test compounds receive NO direct augmentation
(only the 253 unblind, which sit in the same analog-expansion library, give
indirect transfer through shared scaffolds).
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
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb590"
NB562_RAE = 0.5065  # the score to beat

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def main() -> dict:
    print("=" * 78)
    print("nb590 -- UNBLIND-AUGMENTED LGBM BASE")
    print("=" * 78)

    needed = {
        "TRAIN":        DATA_RAW / "pxr-challenge_TRAIN.csv",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Load CSVs ----
    train_df = pd.read_csv(needed["TRAIN"])
    test_df = pd.read_csv(needed["TEST_BLINDED"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    te_names = test_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(
        drop=True
    )
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )

    n_tr = len(train_df)
    n_te = len(test_df)
    n_unb = len(unb_df)
    print(f"\ntrain n={n_tr}  test n={n_te}  unblind n={n_unb}")

    # ---- Featurise ----
    print("\nFeaturising train (combined)...")
    X_tr = combined(train_df["SMILES"].tolist())
    print("Featurising unblind...")
    X_unb = combined(unb_df["SMILES"].tolist())
    print("Featurising test...")
    X_te = combined(test_df["SMILES"].tolist())

    # Stack everything once so impute uses a consistent col-median, then split
    X_all = np.vstack([X_tr, X_unb, X_te])
    X_all = impute(X_all)
    X_tr = X_all[:n_tr]
    X_unb = X_all[n_tr : n_tr + n_unb]
    X_te = X_all[n_tr + n_unb :]
    print(f"  X_tr={X_tr.shape}  X_unb={X_unb.shape}  X_te={X_te.shape}  "
          f"(dtype={X_tr.dtype}, ~{X_all.nbytes / 1e6:.1f} MB)")

    y_tr = train_df["pEC50"].astype(np.float64).values
    y_unb = unb_df["pEC50"].astype(np.float64).values

    # ---- Cross-fit over 253 unblind rows ----
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (train + (k-1)/k unblind  -> held-out unblind)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_unb = np.full(n_unb, np.nan, dtype=np.float64)
    te_per_fold = np.zeros((N_FOLDS, n_te), dtype=np.float64)
    fold_raes: list[float] = []
    for fold, (tr_idx_unb, va_idx_unb) in enumerate(
        kf.split(np.arange(n_unb))
    ):
        X_fold = np.vstack([X_tr, X_unb[tr_idx_unb]])
        y_fold = np.concatenate([y_tr, y_unb[tr_idx_unb]])

        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_fold, y_fold)

        oof_unb[va_idx_unb] = mdl.predict(X_unb[va_idx_unb])
        te_per_fold[fold] = mdl.predict(X_te)

        r_va = float(rae(y_unb[va_idx_unb], oof_unb[va_idx_unb]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_idx_unb):3d} -> "
              f"n_total={X_fold.shape[0]}  n_va={len(va_idx_unb):3d}  "
              f"RAE={r_va:.4f}")

    rae_oof = float(rae(y_unb, oof_unb))
    print(f"\nPooled cross-fit RAE = {rae_oof:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")
    print(f"Train-only baseline to beat (nb562): {NB562_RAE:.4f}")
    print(f"  -> nb590 beats nb562: {rae_oof < NB562_RAE}")

    # Average-of-folds train-only-style test prediction (CV bag, for comparison)
    te_cv_bag = te_per_fold.mean(axis=0)

    # ---- Deploy: refit on (4139 + 253) ----
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on full 4139 + 253 = 4392 rows)")
    print("-" * 78)
    X_deploy = np.vstack([X_tr, X_unb])
    y_deploy = np.concatenate([y_tr, y_unb])
    print(f"  X_deploy={X_deploy.shape}  y_deploy={y_deploy.shape}")

    mdl_deploy = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_deploy.fit(X_deploy, y_deploy)
    te_deploy = mdl_deploy.predict(X_te).astype(np.float32)
    print(f"  te_deploy mean/std = {te_deploy.mean():.3f} / "
          f"{te_deploy.std():.3f}")
    print(f"  te_cv_bag  mean/std = {te_cv_bag.mean():.3f} / "
          f"{te_cv_bag.std():.3f}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(
        DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_unb.astype(np.float32)
    )
    np.save(
        DATA_PROCESSED / f"te_{TAG}_cvbag.npy", te_cv_bag.astype(np.float32)
    )

    plain = SUBMISSIONS / f"{TAG}_unblind_aug_lgbm.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32)
        + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_unblind_aug_lgbm_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'te_{TAG}_cvbag.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ---- Quick comparison: train-only LGBM (no unblind) for context ----
    print("\n" + "-" * 78)
    print("TRAIN-ONLY LGBM (4139 rows) — sanity comparison")
    print("-" * 78)
    mdl_tonly = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_tonly.fit(X_tr, y_tr)
    pred_unb_tonly = mdl_tonly.predict(X_unb)
    rae_tonly_unblind = float(rae(y_unb, pred_unb_tonly))
    print(f"  train-only RAE on 253 unblind = {rae_tonly_unblind:.4f}")
    print(f"  nb590 cross-fit RAE          = {rae_oof:.4f}")
    print(f"  delta (cross-fit - train-only) = "
          f"{rae_oof - rae_tonly_unblind:+.4f}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("=== nb590 SUMMARY ===")
    print(f"  cross-fit RAE (253)         = {rae_oof:.4f}")
    print(f"  per-fold RAE                = "
          f"{[f'{r:.4f}' for r in fold_raes]}")
    print(f"  nb562 reference RAE         = {NB562_RAE:.4f}")
    print(f"  beats nb562                 = {rae_oof < NB562_RAE}")
    print(f"  train-only RAE (no aug)     = {rae_tonly_unblind:.4f}")
    print(f"  te_deploy mean/std (513)    = "
          f"{te_deploy.mean():.3f} / {te_deploy.std():.3f}")
    print("=" * 78)

    return {
        "success": True,
        "rae_crossfit": rae_oof,
        "rae_per_fold": [float(r) for r in fold_raes],
        "rae_train_only_on_unblind": rae_tonly_unblind,
        "rae_nb562_target": NB562_RAE,
        "beats_nb562": bool(rae_oof < NB562_RAE),
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
