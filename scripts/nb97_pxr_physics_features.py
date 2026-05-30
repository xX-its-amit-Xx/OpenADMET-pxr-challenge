"""nb97 — PXR-specific physicochemical feature layer.

Adds 14 domain-informed features derived from RSC 2023 XGBoost PXR study:
  Chi1n, Chi2v, MinAbsPartialCharge (top-3 PXR descriptors in literature)
  LogP/MW distance from activator centroid (MW~314, LogP~3.69)
  Pharmacophore indicators: sulfonation, HBA>HBD, rotbonds, TPSA, FractionCSP3
Trains LGBM on combined() + these 14 PXR features.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)

# PXR activator centroid from RSC 2023 study
PXR_ACT_LOGP = 3.69
PXR_ACT_MW = 314.0

SULFONATE_SMARTS = Chem.MolFromSmarts("[SX4](=O)(=O)")
BIPHENYL_SMARTS  = Chem.MolFromSmarts("c1ccc(-c2ccccc2)cc1")
STEROID_6_5      = Chem.MolFromSmarts("[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]~1.[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]~1")

def pxr_physics_features(smiles_list: list) -> np.ndarray:
    """14 PXR-specific physicochemical + pharmacophore features."""
    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                rows.append([np.nan] * 14)
                continue
            logp      = Descriptors.MolLogP(mol)
            mw        = Descriptors.MolWt(mol)
            tpsa      = Descriptors.TPSA(mol)
            hbd       = rdMolDescriptors.CalcNumHBD(mol)
            hba       = rdMolDescriptors.CalcNumHBA(mol)
            rotbonds  = rdMolDescriptors.CalcNumRotatableBonds(mol)
            chi1n     = Descriptors.Chi1n(mol)
            chi2v     = Descriptors.Chi2v(mol)
            min_chg   = Descriptors.MinAbsPartialCharge(mol)
            frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            logp_dist = abs(logp - PXR_ACT_LOGP)
            mw_dist   = abs(mw - PXR_ACT_MW) / 100.0
            has_sulf  = int(bool(mol.HasSubstructMatch(SULFONATE_SMARTS)))
            hba_hbd_r = hba / max(hbd, 1)
            rows.append([logp_dist, mw_dist, chi1n, chi2v, min_chg,
                         tpsa / 100.0, rotbonds / 10.0, hba_hbd_r, frac_csp3,
                         float(hba > hbd), has_sulf, logp, mw / 300.0, float(hba - hbd)])
        except Exception:
            rows.append([np.nan] * 14)
    return np.array(rows, dtype=np.float32)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  R2={r2:.4f}  "
              f"r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def main():
    print("=== nb97: PXR Physics Feature Layer ===")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing standard combined features...")
    X_tr_base = impute(combined(tr["smiles"].tolist()))
    X_te_base = impute(combined(te["smiles"].tolist()))

    print("Computing PXR physics features...")
    X_tr_pxr = pxr_physics_features(tr["smiles"].tolist())
    X_te_pxr = pxr_physics_features(te["smiles"].tolist())

    # Impute the PXR feature block
    for j in range(X_tr_pxr.shape[1]):
        col = X_tr_pxr[:, j]
        med = np.nanmedian(col)
        X_tr_pxr[np.isnan(col), j] = med if np.isfinite(med) else 0.0
        col2 = X_te_pxr[:, j]
        X_te_pxr[np.isnan(col2), j] = med if np.isfinite(med) else 0.0

    X_tr = np.hstack([X_tr_base, X_tr_pxr])
    X_te = np.hstack([X_te_base, X_te_pxr])
    print(f"Feature shape: train={X_tr.shape}  test={X_te.shape}")

    print("\n=== Scaffold 5-fold CV ===")
    oof = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        fold_rae = rae(y_tr[va_idx], oof[va_idx])
        print(f"  fold {fold+1}  RAE={fold_rae:.4f}", flush=True)

    full_metrics(y_tr, oof, "nb97_pxr_features")

    print("\nTraining final model on all data...")
    m_final = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)]
    )
    te_preds = m_final.predict(X_te)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    np.save(DATA_PROCESSED / "oof_nb97_pxr_features.npy", oof)
    np.save(DATA_PROCESSED / "te_nb97_pxr_features.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "97_pxr_physics_features.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Test  min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  max={te_preds.max():.2f}")


if __name__ == "__main__":
    main()
