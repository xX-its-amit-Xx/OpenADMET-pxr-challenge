"""nb257 -- External PXR kNN pseudo-label feature.

Combine new ChEMBL PXR EC50 (1586) + Papyrus PXR (945) + counter-assay
records into a unified external PXR reference set (~2500 unique compounds).

For each train + test compound, find K=5 nearest external PXR compounds via
Tanimoto, compute weighted-average pec50. Use as new feature.

Then: train LGBM with this single new feature + base; stack with 239.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except: return None


def main():
    print("=== nb257: External PXR kNN feature ===\n")

    # Train + test
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}")

    # Build external reference set
    print("\nBuilding external PXR reference set...")
    parts = []
    # 1. New ChEMBL PXR
    chembl = pd.read_parquet("data/processed/chembl_pxr_new_external.parquet")
    chembl_std = chembl.copy()
    chembl_std["std_smiles"] = chembl_std["canonical_smiles"].apply(std_smi)
    chembl_std = chembl_std[chembl_std["std_smiles"].notna()][["std_smiles", "pec50"]]
    print(f"  ChEMBL: {len(chembl_std)}")
    parts.append(chembl_std)

    # 2. Papyrus PXR
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus[papyrus["target_name"].str.contains("PXR", case=False, na=False)].copy()
    papyrus_pxr["std_smiles"] = papyrus_pxr["std_smiles"].apply(std_smi)
    papyrus_pxr = papyrus_pxr[papyrus_pxr["std_smiles"].notna()][["std_smiles", "pec50"]]
    print(f"  Papyrus PXR: {len(papyrus_pxr)}")
    parts.append(papyrus_pxr)

    # Combine, dedupe (keep median per std_smiles)
    ext = pd.concat(parts).groupby("std_smiles")["pec50"].median().reset_index()
    print(f"  After dedupe: {len(ext)}")

    # Exclude train AND test SMILES (no leakage)
    excl = set(smiles_tr) | set(smiles_te)
    ext = ext[~ext["std_smiles"].isin(excl)].reset_index(drop=True)
    print(f"  After train+test exclusion: {len(ext)}")
    print(f"  pec50 stats: mean={ext.pec50.mean():.3f} std={ext.pec50.std():.3f}")

    # Featurize FPS for external
    print("\nComputing Morgan FPs...")
    def morgan(smiles):
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)
        return fps

    fps_ext = morgan(ext["std_smiles"].tolist())
    fps_tr = morgan(smiles_tr)
    fps_te = morgan(smiles_te)
    ext_y = ext["pec50"].values

    # For each train, top-K nearest ext (excluding self)
    K = 5
    print(f"\nComputing kNN features (K={K})...")
    def knn_feat(query_fps, ref_fps, ref_y):
        n = len(query_fps)
        f = np.zeros(n)
        f_count = np.zeros(n)  # how many neighbors above sim threshold
        f_top1 = np.zeros(n)
        for i, qfp in enumerate(query_fps):
            if qfp is None: continue
            sims = np.array(BulkTanimotoSimilarity(qfp, ref_fps))
            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]
            top_y = ref_y[top_idx]
            weights = top_sims + 0.01
            weights = weights / weights.sum()
            f[i] = (weights * top_y).sum()
            f_count[i] = (top_sims >= 0.3).sum()
            f_top1[i] = top_sims[0]
        return f, f_count, f_top1

    knn_tr, count_tr, top1_tr = knn_feat(fps_tr, fps_ext, ext_y)
    knn_te, count_te, top1_te = knn_feat(fps_te, fps_ext, ext_y)

    print(f"\nkNN feature: mean={knn_tr.mean():.3f} std={knn_tr.std():.3f}")
    print(f"Correlation knn_tr with y_tr: {np.corrcoef(knn_tr, y_tr)[0,1]:.4f}")

    # Build feature matrix
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, knn_tr.reshape(-1,1), count_tr.reshape(-1,1), top1_tr.reshape(-1,1)])
    X_te_aug = np.column_stack([X_te, knn_te.reshape(-1,1), count_te.reshape(-1,1), top1_te.reshape(-1,1)])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n=== OOF comparison ===")
    for name, X in [("base", X_tr), ("base+ext_knn", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"  {name:20s}: OOF RAE = {rae(y_tr, oof):.4f}")

    # Final + save
    final_oof = np.zeros(len(y_tr))
    final_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        final_oof[vi] = md.predict(X_tr_aug[vi])
        final_te_preds.append(md.predict(X_te_aug))
    final_te = np.mean(final_te_preds, axis=0)
    print(f"\nFinal nb257 OOF: {rae(y_tr, final_oof):.4f}  te_std={final_te.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb257_ext_knn.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb257_ext_knn.npy", final_te)
    np.save(DATA_PROCESSED / "ext_knn_pec50_tr.npy", knn_tr)
    np.save(DATA_PROCESSED / "ext_knn_pec50_te.npy", knn_te)

    # Stack with 239
    print("\n=== Stack nb257 with 239 ===")
    from scipy.optimize import minimize
    nb239 = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, final_oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb257'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
