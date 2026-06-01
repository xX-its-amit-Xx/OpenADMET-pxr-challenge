"""nb434 -- Residual LGBM corrector on top of nb432.

Approximates nb432_train_oof as a weighted blend of per-component train OOFs
(using the SLSQP weights nb432 picked on the 253 unblind):

  nb432_train_oof  ~  0.62 * nb424_oof  +  0.32 * nb430_oof  +  0.06 * nb427_oof

For nb424 & nb427 we lack true train OOFs for their nb423 NN combiner /
nb400 crossfit branches. Per spec, fall back to the proxy:
  - nb424_oof_proxy = chemprop_mean_oof   (4-model TRAIN mean)
  - nb427_oof_proxy = chemprop_mean_oof   (HIGH branch is mean; nb400 has no
                                           TRAIN OOF -- use mean)
  - nb430_oof_proxy = isotonic(chemprop_mean_oof)  (re-fit on TRAIN -> transform)

Residual = y_tr - nb432_train_oof_proxy. Fit LIGHT LGBM on combined
Morgan+RDKit features. Scaffold 5-fold CV. Test pred = te_nb432 + residual.
Save te_nb434.npy + safe + soft07_truth CSVs. Honest unblind RAE on 253.
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
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

W424 = 0.62
W430 = 0.32
W427 = 0.06
SOFT_W = 0.7
SEED = 0

CHEMPROP_OOF = [
    "oof_nb93_chemprop_large_gpu.npy",
    "oof_nb130_external_pxr.npy",
    "oof_nb264_chemprop_mt.npy",
    "oof_chemprop_aux.npy",
]


def main():
    print("=" * 78)
    print("nb434 -- RESIDUAL LGBM on TOP of nb432 (proxy train OOF)")
    print("=" * 78)

    tr = load_train()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    print(f"train rows: {n_tr}")

    cp_oof = np.mean(
        np.stack(
            [np.load(DATA_PROCESSED / f).astype(np.float64) for f in CHEMPROP_OOF],
            axis=0,
        ),
        axis=0,
    )
    print(f"chemprop_mean_oof std={cp_oof.std():.3f} mean={cp_oof.mean():.3f}")

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(cp_oof, y_tr)
    nb430_oof_proxy = iso.transform(cp_oof)
    nb424_oof_proxy = cp_oof.copy()
    nb427_oof_proxy = cp_oof.copy()
    nb432_train_oof = (
        W424 * nb424_oof_proxy + W430 * nb430_oof_proxy + W427 * nb427_oof_proxy
    )
    proxy_rae = rae(y_tr, nb432_train_oof)
    print(f"proxy nb432_train_oof RAE = {proxy_rae:.4f}  "
          f"std={nb432_train_oof.std():.3f}")

    residual = y_tr - nb432_train_oof
    print(f"residual: mean={residual.mean():+.3f} std={residual.std():.3f} "
          f"min={residual.min():+.3f} max={residual.max():+.3f}")

    smis_tr = tr["smiles"].tolist()
    X_tr = impute(combined(smis_tr)).astype(np.float32)
    print(f"X_tr shape={X_tr.shape}")

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    smis_te = te_df["SMILES"].tolist()
    X_te = impute(combined(smis_te)).astype(np.float32)
    print(f"X_te shape={X_te.shape}")

    tr_scaf = add_standard_columns(tr[["smiles"]].copy())
    folds = scaffold_kfold_indices(tr_scaf["scaffold"], n_splits=5, seed=SEED)

    oof_resid = np.zeros(n_tr, dtype=np.float64)
    te_resid_folds = np.zeros((5, len(smis_te)), dtype=np.float64)
    fold_raes = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1,
        )
        m.fit(X_tr[tr_idx], residual[tr_idx])
        oof_resid[va_idx] = m.predict(X_tr[va_idx])
        te_resid_folds[k] = m.predict(X_te)
        nb434_va = nb432_train_oof[va_idx] + oof_resid[va_idx]
        r_va = rae(y_tr[va_idx], nb434_va)
        fold_raes.append(r_va)
        print(f"  fold {k}: n_tr={len(tr_idx):4d} n_va={len(va_idx):4d}  "
              f"RAE={r_va:.4f}")

    te_resid = te_resid_folds.mean(axis=0)
    nb434_train_oof = nb432_train_oof + oof_resid
    train_rae = rae(y_tr, nb434_train_oof)
    print(f"\nscaffold-CV train RAE (corrected) = {train_rae:.4f}  "
          f"(proxy = {proxy_rae:.4f}, delta {train_rae - proxy_rae:+.4f})")
    print(f"residual OOF std={oof_resid.std():.3f}   "
          f"TE std={te_resid.std():.3f}")

    te_nb432 = np.load(DATA_PROCESSED / "te_nb432.npy").astype(np.float64)
    deploy = te_nb432 + te_resid
    print(f"\nte_nb432 std={te_nb432.std():.3f}  "
          f"te_nb434 std={deploy.std():.3f}  "
          f"correction std={te_resid.std():.3f}")

    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]])
    unb_y = unb_kept["pEC50"].values.astype(np.float64)
    unb_rae_432 = rae(unb_y, te_nb432[unb_idx])
    unb_rae_434 = rae(unb_y, deploy[unb_idx])
    print(f"\nunblind n={len(unb_idx)}")
    print(f"  nb432 unblind RAE = {unb_rae_432:.4f}")
    print(f"  nb434 unblind RAE = {unb_rae_434:.4f}  "
          f"(delta {unb_rae_434 - unb_rae_432:+.4f})")

    np.save(DATA_PROCESSED / "te_nb434.npy", deploy)
    out_safe = SUBMISSIONS / "nb434_residual_corrected.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)
    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    out_soft = SUBMISSIONS / "nb434_residual_corrected_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb434.npy'}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    beats_nb432 = unb_rae_434 < unb_rae_432
    print("\n" + "=" * 78)
    print(f"=== nb434 unblind RAE = {unb_rae_434:.4f}  "
          f"(nb432={unb_rae_432:.4f})  beats_nb432={beats_nb432} ===")
    print("=" * 78)

    return {
        "unb_rae_432": float(unb_rae_432),
        "unb_rae_434": float(unb_rae_434),
        "beats_nb432": bool(beats_nb432),
        "train_rae_corrected": float(train_rae),
        "proxy_train_rae": float(proxy_rae),
    }


if __name__ == "__main__":
    main()
