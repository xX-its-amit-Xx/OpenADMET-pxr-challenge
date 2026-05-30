"""nb246 -- SMILES enumeration test-time augmentation, but REBUILD nb224 pipeline.

Idea: nb155 tried TTA but on a fresh LGBM (failed). What if we run TTA on
the FULL nb224-style pipeline? The richer the model, the more TTA might help.

For each test compound:
1. Generate 30 randomized SMILES variants via RDKit MolToSmiles(canonical=False, doRandom=True)
2. Featurize each variant
3. Predict via a model that mimics nb224's signature (LGBM on combined features)
4. Take median + IQR

The median across variants reduces fingerprint hash noise (although fingerprints
are atom-order invariant in theory, variant SMILES expose different bit
distributions due to Morgan's bit-hashing).
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def randomized_smiles(smi, n=30, seed=42):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [smi]
    out = [smi]
    n_atoms = mol.GetNumAtoms()
    rng = np.random.default_rng(seed)
    for _ in range(n - 1):
        ans = list(range(n_atoms))
        rng.shuffle(ans)
        try:
            new_mol = Chem.RenumberAtoms(mol, ans)
            s = Chem.MolToSmiles(new_mol, canonical=False, doRandom=True)
            if s and s not in out:
                out.append(s)
        except Exception:
            pass
    return out


def main():
    print("=== nb246: SMILES enumeration TTA on nb224-like model ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Featurizing train...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)

    # Train a 5-fold model ensemble (mimics nb224's underlying base)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("Training 5-fold LGBMs...")
    models = []
    oof = np.zeros(len(y_tr))
    for fold_idx, (ti, vi) in enumerate(folds):
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y_tr[ti], eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        models.append(md)
    print(f"Base LGBM OOF RAE: {rae(y_tr, oof):.4f}")

    K = 30
    print(f"\nTTA on test ({len(smiles_te)} x K={K} variants)...")
    te_pred_tta = np.zeros(len(smiles_te))
    te_var = np.zeros(len(smiles_te))
    t0 = time.time()
    for i, smi in enumerate(smiles_te):
        variants = randomized_smiles(smi, n=K, seed=42 + i)
        # Featurize all variants at once
        X_var = combined(variants); X_var = impute(X_var)
        # Predict with each fold's model, then average
        fold_preds = np.stack([m.predict(X_var) for m in models])  # (5, K)
        fold_avg = fold_preds.mean(axis=0)  # (K,)
        te_pred_tta[i] = np.median(fold_avg)
        te_var[i] = fold_avg.std()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(smiles_te) - i - 1)
            print(f"  {i+1}/{len(smiles_te)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Compare with baseline (no TTA)
    X_te = combined(smiles_te); X_te = impute(X_te)
    te_pred_base = np.mean([m.predict(X_te) for m in models], axis=0)
    delta = np.mean(np.abs(te_pred_tta - te_pred_base))
    print(f"\nTTA-vs-base mean |delta|: {delta:.4f}")
    print(f"  baseline te_std={te_pred_base.std():.4f}, mean={te_pred_base.mean():.3f}")
    print(f"  TTA te_std={te_pred_tta.std():.4f}, mean={te_pred_tta.mean():.3f}")
    print(f"  TTA variance per compound: median={np.median(te_var):.3f}, max={te_var.max():.3f}")

    np.save(DATA_PROCESSED / "te_nb246_tta_median.npy", te_pred_tta)
    np.save(DATA_PROCESSED / "te_nb246_tta_var.npy", te_var)
    np.save(DATA_PROCESSED / "te_nb246_base.npy", te_pred_base)
    np.save(DATA_PROCESSED / "oof_nb246_base.npy", oof)
    print("Saved nb246 arrays")

    # Use TTA variance as a confidence score
    # Compounds with high TTA variance = uncertain → shrink toward median
    median_pred = np.median(te_pred_tta)
    confidence = 1 - (te_var - te_var.min()) / max(te_var.max() - te_var.min(), 1e-6)  # 0-1
    te_shrunk = confidence * te_pred_tta + (1 - confidence) * median_pred
    print(f"Confidence-shrunk: mean={te_shrunk.mean():.3f}, std={te_shrunk.std():.3f}")
    np.save(DATA_PROCESSED / "te_nb246_confidence_shrunk.npy", te_shrunk)


if __name__ == "__main__":
    main()
