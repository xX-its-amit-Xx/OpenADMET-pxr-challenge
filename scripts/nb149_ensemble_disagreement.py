"""nb149 -- Ensemble disagreement / uncertainty features.

For each compound, compute the disagreement (std + range + IQR) across all
our base model OOF predictions. High disagreement = high model uncertainty.

These uncertainty features can be input to a meta-learner that learns to
WEIGHT predictions differently when confident vs uncertain.

Concretely:
  - For each compound: std, max-min, q75-q25 across 20 base model OOFs
  - Add these as 3 new features alongside the base feature set
  - Train LGBM; the model can learn 'when uncertain, predict closer to mean'
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

# Models to compute disagreement across
STEMS = [
    "nb167_xgboost_mae", "nb156_catboost_mae", "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool", "nb165_multiseed_162c", "nb149_meta_maeloss",
    "nb183_qreg_poly10", "nb187_diversity_qreg",
    "nb197_dense_grid", "nb224_pool_plus_2",
]


def main():
    print("=== nb149: Ensemble disagreement uncertainty ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    cols_oof, cols_te, names = [], [], []
    for s in STEMS:
        op = DATA_PROCESSED / f"oof_{s}.npy"
        tp = DATA_PROCESSED / f"te_{s}.npy"
        if not op.exists() or not tp.exists(): continue
        o = np.load(op).astype(np.float64).flatten()
        t = np.load(tp).astype(np.float64).flatten()
        if len(o) != len(y_tr) or len(t) != len(te_df): continue
        o = np.where(np.isfinite(o), o, np.nanmean(o))
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        cols_oof.append(o); cols_te.append(t); names.append(s)
    print(f"Loaded {len(names)} base model OOFs: {names}")
    OOF = np.column_stack(cols_oof)
    TE  = np.column_stack(cols_te)

    # Disagreement features per compound: std, range, IQR, mean abs deviation from median
    def disagreement(M):
        std = M.std(axis=1)
        rng = M.max(axis=1) - M.min(axis=1)
        iqr = np.percentile(M, 75, axis=1) - np.percentile(M, 25, axis=1)
        med = np.median(M, axis=1)
        mad = np.mean(np.abs(M - med[:, None]), axis=1)
        return np.column_stack([std, rng, iqr, mad])

    dis_tr = disagreement(OOF)
    dis_te = disagreement(TE)
    print(f"Disagreement shapes: train={dis_tr.shape} test={dis_te.shape}")
    print(f"Train std distribution: median={np.median(dis_tr[:,0]):.3f}  max={dis_tr[:,0].max():.3f}")

    # Correlations of each disagreement feature with PXR pEC50
    for j, name in enumerate(["std","range","iqr","mad"]):
        rho, p = spearmanr(dis_tr[:, j], y_tr)
        print(f"  {name}: ρ={rho:+.3f}  p={p:.2e}")

    # Augmented LGBM
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    # Also include the median prediction itself as a feature
    X_tr_aug = np.hstack([X_tr_base, dis_tr,
                          np.median(OOF, axis=1).reshape(-1,1),
                          np.mean(OOF, axis=1).reshape(-1,1)])
    X_te_aug = np.hstack([X_te_base, dis_te,
                          np.median(TE, axis=1).reshape(-1,1),
                          np.mean(TE, axis=1).reshape(-1,1)])
    print(f"Aug shape: train={X_tr_aug.shape}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                          ("dis_aug", X_tr_aug, X_te_aug)]:
        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:10s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "dis_aug":
            np.save(DATA_PROCESSED / "oof_nb149_dis_aug.npy", oof)
            np.save(DATA_PROCESSED / "te_nb149_dis_aug.npy", te_pred)
            print(f"  Saved")


if __name__ == "__main__":
    main()
