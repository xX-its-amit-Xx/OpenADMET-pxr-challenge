"""nb263 -- PubChem PXR-active library kNN feature.

For each train/test compound, compute Tanimoto similarity to top-5 nearest
PXR-active CIDs (from 5739 PubChem actives across multiple AIDs).

Features:
- nb_active_top1_sim
- nb_active_top5_mean_sim
- nb_active_top5_avg_active_rate (how 'active' are similar compounds)
- nb_active_top5_avg_n_active (total active hits)
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
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def morgan(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb263: PXR-active kNN feature ===\n")
    # Load active library
    lib = pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
    print(f"Library: {len(lib)} compounds, active_rate stats: {lib.active_rate.describe()}")
    lib["std_smiles"] = lib["smiles"].apply(std_smi)
    lib = lib.dropna(subset=["std_smiles"]).reset_index(drop=True)
    print(f"After std: {len(lib)}")

    # Train + test
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    # Exclude train+test from library
    excl = set(smiles_tr) | set(smiles_te)
    lib = lib[~lib["std_smiles"].isin(excl)].reset_index(drop=True)
    print(f"After train+test exclusion: {len(lib)}")

    # Compute fingerprints
    print("Computing FPs...")
    fps_lib = morgan(lib["std_smiles"].tolist())
    fps_tr = morgan(smiles_tr)
    fps_te = morgan(smiles_te)

    # KNN features
    K = 5
    lib_rate = lib["active_rate"].values
    lib_nact = lib["n_active"].values

    def knn_feats(qfps):
        n = len(qfps)
        f = np.zeros((n, 4))
        for i, qfp in enumerate(qfps):
            if qfp is None: continue
            sims = np.array(BulkTanimotoSimilarity(qfp, fps_lib))
            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]
            f[i, 0] = top_sims[0]
            f[i, 1] = top_sims.mean()
            w = top_sims + 0.01; w = w / w.sum()
            f[i, 2] = (w * lib_rate[top_idx]).sum()
            f[i, 3] = (w * lib_nact[top_idx]).sum()
        return f

    print("KNN train...")
    X_knn_tr = knn_feats(fps_tr)
    print("KNN test...")
    X_knn_te = knn_feats(fps_te)
    print(f"Feature shape: tr={X_knn_tr.shape}, te={X_knn_te.shape}")

    # Correlations
    feat_names = ["top1_sim", "top5_sim_mean", "weighted_active_rate", "weighted_n_active"]
    print("\nFeature correlations with pec50:")
    for j, n in enumerate(feat_names):
        r = np.corrcoef(X_knn_tr[:, j], y_tr)[0, 1]
        print(f"  {n}: r={r:.4f}")

    # Train LGBM with base + these
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, X_knn_tr])
    X_te_aug = np.column_stack([X_te, X_knn_te])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    print("\n=== OOF ===")
    for name, X in [("base", X_tr), ("base+pxr_active_knn", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"  {name:30s}: OOF RAE = {rae(y_tr, oof):.4f}")

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
    print(f"\nFinal nb263 OOF: {rae(y_tr, final_oof):.4f}")
    np.save(DATA_PROCESSED / "oof_nb263_pxr_active_knn.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb263_pxr_active_knn.npy", final_te)

    # Stack with 239
    print("\n=== 5-way SLSQP w/ nb263 ===")
    from scipy.optimize import minimize
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
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb263'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
