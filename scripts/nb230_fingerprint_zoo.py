"""nb230 -- Multiple fingerprint families (Atom Pair, Topological Torsion, MACCS,
extended Morgan radius=4) trained as separate LGBMs, then their OOF/test as
ensemble candidates.

Currently the SLSQP pool is dominated by models built on the same ECFP4
features (which capture small-radius substructures). Different fingerprints
expose different chemistry:
  - Atom Pair: long-range pairwise atom relationships
  - Topological Torsion: 4-atom torsional patterns
  - MACCS keys: 166 hand-crafted structural alerts
  - Morgan radius=4 (ECFP8): wider substructures

Each family produces a distinct LGBM whose OOF errors are decorrelated
from ECFP4-based models — exactly what SLSQP exploits.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

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


def fp_morgan(smiles_list, radius=2, nBits=2048):
    out = np.zeros((len(smiles_list), nBits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits)
        for j in fp.GetOnBits(): out[i, j] = 1.0
    return out


def fp_atom_pair(smiles_list, nBits=2048):
    out = np.zeros((len(smiles_list), nBits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nBits)
        for j in fp.GetOnBits(): out[i, j] = 1.0
    return out


def fp_top_torsion(smiles_list, nBits=2048):
    out = np.zeros((len(smiles_list), nBits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nBits)
        for j in fp.GetOnBits(): out[i, j] = 1.0
    return out


def fp_maccs(smiles_list):
    out = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = MACCSkeys.GenMACCSKeys(m)
        for j in fp.GetOnBits(): out[i, j] = 1.0
    return out


def cv_lgbm(X_tr, X_te, y_tr, splits, label):
    print(f"  {label}: train shape={X_tr.shape}")
    oof = np.zeros(len(y_tr))
    te_preds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_preds.append(m.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
    print(f"    OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
    return oof, te_pred, r, ratio


def main():
    print("=== nb230: Fingerprint zoo (AP, TT, MACCS, ECFP8) ===\n")

    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    fp_funcs = [
        ("ecfp8",      lambda smi: fp_morgan(smi, radius=4, nBits=2048)),
        ("atom_pair",  fp_atom_pair),
        ("top_torsion", fp_top_torsion),
        ("maccs",      fp_maccs),
    ]

    for label, fn in fp_funcs:
        print(f"\n[{label}] computing fingerprints...")
        X_tr = fn(smiles_tr)
        X_te = fn(smiles_te)
        oof, te_pred, r, ratio = cv_lgbm(X_tr, X_te, y_tr, folds, label)
        np.save(DATA_PROCESSED / f"oof_nb230_{label}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb230_{label}.npy", te_pred)
        print(f"    saved oof_nb230_{label}.npy + te_nb230_{label}.npy")


if __name__ == "__main__":
    main()
