"""nb153 -- Multi-output LGBM jointly predicting pec50_pxr + pec50_null.

The counter-assay (PXR-null) has 2,859 compounds with measured pec50_null.
Multi-output regression shares features across both heads; if the inductive
bias helps separate "real PXR" from "general cell" effects, OOF improves.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from pxr.data import load_train, load_test, load_counter
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
    print("=== nb153: Counter-assay multi-output ===\n")
    tr = load_train(); te_df = load_test(); co = load_counter()
    tr = add_standard_columns(tr)
    y_pxr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Build counter-assay lookup
    co_map = dict(zip(co["smiles"], co["pec50"]))
    y_null = np.array([co_map.get(s, np.nan) for s in tr["smiles"]])
    has_null = np.isfinite(y_null)
    print(f"Has counter measurement: {has_null.sum()}/{len(y_pxr)}")

    # Fill missing null with median (so LGBM has values for all rows)
    null_median = np.nanmedian(y_null)
    y_null_filled = np.where(has_null, y_null, null_median)
    print(f"y_null median: {null_median:.3f}")

    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # Joint multi-output via MultiOutputRegressor
    Y = np.column_stack([y_pxr, y_null_filled])

    oof = np.zeros(len(y_pxr))
    te_preds = []
    for tr_idx, va_idx in folds:
        m = MultiOutputRegressor(lgb.LGBMRegressor(**LGBM_BASE), n_jobs=1)
        m.fit(X_tr[tr_idx], Y[tr_idx])
        pred_va = m.predict(X_tr[va_idx])  # shape (n, 2)
        oof[va_idx] = pred_va[:, 0]  # PXR head
        pred_te = m.predict(X_te)
        te_preds.append(pred_te[:, 0])
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_pxr, oof); ratio = te_pred.std() / oof.std()
    print(f"\nMultiOutput LGBM: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
    np.save(DATA_PROCESSED / "oof_nb153_counter_multi.npy", oof)
    np.save(DATA_PROCESSED / "te_nb153_counter_multi.npy", te_pred)
    print("Saved")


if __name__ == "__main__":
    main()
