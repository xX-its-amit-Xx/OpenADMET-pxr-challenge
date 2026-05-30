"""nb154 -- GOSS LightGBM on PXR train + Papyrus 1.5M direct transfer.

Idea: combine 4,139 PXR train + Papyrus compounds with PXR-relevant target
labels as auxiliary training. Use GOSS boost (gradient-one-side sampling)
for memory efficiency on the 1.5M-row dataset.

The auxiliary labels: for each Papyrus compound, map its target activity
to PXR pEC50 space via the per-target calibration we learned in nb144.
Effectively a TRANSFER learning approach where Papyrus extends the training
distribution.

Sample weight: 1.0 for true PXR labels; 0.1-0.3 for transferred Papyrus.
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
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS


def main():
    print("=== nb154: GOSS LightGBM on PXR + Papyrus combined ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Load Papyrus 1.5M
    print("Loading Papyrus...")
    papy = pd.read_parquet(DATA_EXTERNAL / "papyrus_full_filtered.parquet",
                            columns=["SMILES" if True else "SMILES_Stripped",
                                     "accession", "pchembl_value_Mean"])
    smi_col = "SMILES" if "SMILES" in papy.columns else "SMILES_Stripped"
    papy = papy.dropna(subset=[smi_col, "pchembl_value_Mean"])
    # Filter to PXR (O75469) records ONLY for direct transfer
    pxr_papy = papy[papy["accession"] == "O75469"].copy()
    print(f"  Papyrus PXR records: {len(pxr_papy):,}  unique compounds: {pxr_papy[smi_col].nunique():,}")

    if len(pxr_papy) < 100:
        print("Too few PXR records. Using ALL Papyrus with sample weight 0.05.")
        pxr_papy = papy.copy()
        sample_w_aux = 0.05
    else:
        sample_w_aux = 0.30

    # Median-aggregate per compound
    aux_agg = pxr_papy.groupby(smi_col)["pchembl_value_Mean"].median().reset_index()
    aux_agg.columns = ["smiles", "pec50"]

    # Exclude any Papyrus compound that's also in PXR train
    pxr_train_set = set(smiles_tr)
    aux_agg = aux_agg[~aux_agg["smiles"].isin(pxr_train_set)].reset_index(drop=True)
    print(f"  After excluding train overlap: {len(aux_agg):,}")

    # Featurize
    print("Featurizing PXR train + test + aux...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_aux = combined(aux_agg["smiles"].tolist()); X_aux = impute(X_aux)
    y_aux = aux_agg["pec50"].values.astype(np.float64)
    print(f"  Shapes: tr={X_tr.shape}  aux={X_aux.shape}")

    GOSS_LGB = dict(
        boosting_type="goss",  # memory-efficient
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        top_rate=0.2, other_rate=0.1,
        colsample_bytree=0.8, min_child_samples=10,
        objective="mae", n_jobs=4, random_state=42, verbose=-1,
    )

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # CV with transfer learning
    oof = np.zeros(len(y_tr))
    te_preds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        # Combined training: fold train + ALL aux
        X_combined = np.vstack([X_tr[tr_idx], X_aux])
        y_combined = np.concatenate([y_tr[tr_idx], y_aux])
        w_combined = np.concatenate([np.ones(len(tr_idx)),
                                       np.full(len(y_aux), sample_w_aux)])

        m = lgb.LGBMRegressor(**GOSS_LGB)
        m.fit(X_combined, y_combined, sample_weight=w_combined,
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_preds.append(m.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
    print(f"\nGOSS+aux: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")

    np.save(DATA_PROCESSED / "oof_nb154_goss_aux.npy", oof)
    np.save(DATA_PROCESSED / "te_nb154_goss_aux.npy", te_pred)
    print("Saved oof_nb154_goss_aux.npy + te_nb154_goss_aux.npy")


if __name__ == "__main__":
    main()
