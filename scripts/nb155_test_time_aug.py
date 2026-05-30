"""nb155 -- Test-time SMILES augmentation.

For each test compound, generate K=10 randomized SMILES strings via RDKit's
random canonical form (kekulization variants, atom ordering). Predict pEC50
for each variant via the nb224 ensemble's LGBM, then average.

If the model has SMILES-string sensitivity (which fingerprint-based models
shouldn't but might via numerical precision), TTA reduces variance.

This is essentially Monte Carlo on the SMILES representation. Should produce
predictions with same mean but potentially lower variance, which empirically
sometimes helps RAE on noisy test sets.
"""
import os, sys, warnings
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def randomize_smiles(smi, k=10, seed=42):
    """Generate k randomized SMILES strings from the same molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [smi]
    out = [smi]
    n_atoms = mol.GetNumAtoms()
    rng = np.random.default_rng(seed)
    for _ in range(k - 1):
        # Random atom ordering
        ans = list(range(n_atoms))
        rng.shuffle(ans)
        try:
            new_mol = Chem.RenumberAtoms(mol, ans)
            s = Chem.MolToSmiles(new_mol, canonical=False, doRandom=False)
            if s and s not in out:
                out.append(s)
        except Exception:
            pass
    return out


def main():
    print("=== nb155: Test-time SMILES augmentation ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Train the standard LGBM on all train
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    print(f"Training LGBM on {len(smiles_tr)} compounds...")
    m = lgb.LGBMRegressor(**LGBM_BASE)
    m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])

    # For each test compound, generate K SMILES variants, predict each, average
    K = 10
    n_te = len(smiles_te)
    print(f"\nTTA on test ({n_te} compounds × K={K} variants)...")
    te_pred_tta = np.zeros(n_te)
    import time; t0 = time.time()
    for i, smi in enumerate(smiles_te):
        variants = randomize_smiles(smi, k=K, seed=42 + i)
        X_var = combined(variants); X_var = impute(X_var)
        preds = m.predict(X_var)
        te_pred_tta[i] = preds.mean()
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_te}  ({time.time()-t0:.0f}s)")

    # Baseline (no TTA)
    X_te = combined(smiles_te); X_te = impute(X_te)
    te_pred_base = m.predict(X_te)
    diff = np.mean(np.abs(te_pred_tta - te_pred_base))
    print(f"\nTTA vs baseline avg abs diff: {diff:.4f}")
    print(f"  baseline te_std={te_pred_base.std():.4f}")
    print(f"  tta te_std={te_pred_tta.std():.4f}")

    # Compute OOF on train via 5-fold CV WITH TTA (slow but fair comparison)
    print("\nScaffold CV OOF with TTA (slow)...")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    oof_base = np.zeros(len(y_tr))
    oof_tta = np.zeros(len(y_tr))
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        print(f"  fold {fold_idx+1}/5")
        mfold = lgb.LGBMRegressor(**LGBM_BASE)
        mfold.fit(X_tr[tr_idx], y_tr[tr_idx],
                   eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_base[va_idx] = mfold.predict(X_tr[va_idx])
        for j, vi in enumerate(va_idx):
            variants = randomize_smiles(smiles_tr[vi], k=K, seed=42 + vi)
            Xv = combined(variants); Xv = impute(Xv)
            oof_tta[vi] = mfold.predict(Xv).mean()
    print(f"\nBase OOF RAE={rae(y_tr, oof_base):.4f}")
    print(f"TTA  OOF RAE={rae(y_tr, oof_tta):.4f}")

    # Save TTA test predictions
    np.save(DATA_PROCESSED / "oof_nb155_tta.npy", oof_tta)
    np.save(DATA_PROCESSED / "te_nb155_tta.npy", te_pred_tta)
    print("Saved oof_nb155_tta.npy + te_nb155_tta.npy")


if __name__ == "__main__":
    main()
