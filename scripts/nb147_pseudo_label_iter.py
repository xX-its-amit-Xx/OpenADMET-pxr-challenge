"""nb147 -- Pseudo-label iterative refit / self-training.

Idea: use nb224's test predictions as soft labels for test compounds.
Add test compounds + pseudo labels to training data (with low weight).
Refit LGBM on combined data. Get new test predictions. Repeat 2-3 iterations.

This is the classic self-training / pseudo-labeling approach. The risk is
amplifying errors, so we use low weight (0.2-0.3) for the pseudo labels.

If the model can extract additional signal by seeing the test compound
distribution during training (even with soft labels), we may improve.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

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
    print("=== nb147: Pseudo-label iterative refit ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Start with nb224 predictions as initial test labels
    te_init = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy").astype(np.float64)
    print(f"Initial te pseudo-labels: mean={te_init.mean():.3f}  std={te_init.std():.3f}")

    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # Baseline
    oof = np.zeros(len(y_tr)); te_preds = []
    for tr_idx, va_idx in folds:
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_preds.append(m.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    base_rae = rae(y_tr, oof)
    print(f"\n[Iter 0] base only: OOF RAE={base_rae:.4f}  te_std={te_pred.std():.3f}")

    # Iteratively add pseudo labels
    current_te_labels = te_init
    for it in range(1, 4):
        # Combined training set: real CRC (weight 1) + pseudo test (weight 0.2)
        X_combined = np.vstack([X_tr, X_te])
        y_combined = np.concatenate([y_tr, current_te_labels])
        w_combined = np.concatenate([np.ones(len(y_tr)),
                                      np.full(len(current_te_labels), 0.2)])

        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            # Use all of TR + test pseudo (test not in val fold ever)
            tr_mask = np.zeros(len(y_combined), dtype=bool)
            tr_mask[tr_idx] = True
            tr_mask[len(y_tr):] = True  # all test pseudo
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_combined[tr_mask], y_combined[tr_mask],
                  sample_weight=w_combined[tr_mask],
                  eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr[va_idx])
            te_preds.append(m.predict(X_te))
        new_te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        diff = np.mean(np.abs(new_te_pred - current_te_labels))
        print(f"[Iter {it}] OOF RAE={r:.4f}  te_std={new_te_pred.std():.3f}  "
              f"avg label change={diff:.4f}")
        current_te_labels = 0.5 * current_te_labels + 0.5 * new_te_pred  # damped update

    # Save final
    np.save(DATA_PROCESSED / "oof_nb147_pseudo_iter.npy", oof)
    np.save(DATA_PROCESSED / "te_nb147_pseudo_iter.npy", current_te_labels)
    print(f"\nSaved oof_nb147_pseudo_iter.npy + te_nb147_pseudo_iter.npy")


if __name__ == "__main__":
    main()
