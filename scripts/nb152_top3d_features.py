"""nb152 -- Use ONLY top-K most correlated 3D features (avoid nb151 overfitting).

nb151 had 3176 features (WHIM/MORSE/RDF/GETAWAY/AUTOCORR) and degraded OOF
despite individual feature ρ as high as 0.444. The 3000+ dim was too noisy.

Here: select top-50 3D features by absolute Spearman with PXR pEC50, add ONLY those
as augmentation. Should preserve the strong signal while avoiding the noise.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
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


def main():
    print("=== nb152: Top-50 3D features (selected from nb151) ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    F_tr_all = np.load(DATA_PROCESSED / "nb151_3d_features_train.npy")
    F_te_all = np.load(DATA_PROCESSED / "nb151_3d_features_test.npy")
    print(f"All 3D features: train={F_tr_all.shape}  test={F_te_all.shape}")

    # Rank by |Spearman ρ|
    print("Ranking 3D features by absolute correlation...")
    rho_list = []
    for j in range(F_tr_all.shape[1]):
        col = F_tr_all[:, j]
        if col.std() == 0: continue
        rho, p = spearmanr(col, y_tr)
        if not np.isnan(rho):
            rho_list.append((j, rho, p))
    rho_list.sort(key=lambda x: abs(x[1]), reverse=True)

    for K in [20, 50, 100, 200]:
        top_idx = [r[0] for r in rho_list[:K]]
        F_tr_top = F_tr_all[:, top_idx]
        F_te_top = F_te_all[:, top_idx]

        X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
        X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
        X_tr_aug = np.hstack([X_tr_base, F_tr_top])
        X_te_aug = np.hstack([X_te_base, F_te_top])

        scaffolds = tr["scaffold"].tolist()
        folds = scaffold_kfold_indices(scaffolds, n_splits=5)
        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_aug[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_aug[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_aug[va_idx])
            te_preds.append(m.predict(X_te_aug))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  K={K:3d}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if K == 50:
            np.save(DATA_PROCESSED / "oof_nb152_top3d.npy", oof)
            np.save(DATA_PROCESSED / "te_nb152_top3d.npy", te_pred)
            print("  Saved K=50 as candidate")


if __name__ == "__main__":
    main()
