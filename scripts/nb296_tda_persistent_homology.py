"""nb296 -- Persistent-homology TDA fingerprints from 3D conformers.

Compute 0- and 1-dim Vietoris-Rips persistence barcodes on 3D atom point clouds,
vectorise via persistence-image and persistence-statistics, feed to LGBM.

If `gudhi`/`giotto-tda` aren't installed, fall back to a hand-rolled minimal
persistent-homology implementation: distance-matrix based, using scipy MST for
H0 (births at 0, deaths at MST edges). H1 is skipped in fallback mode.
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
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def conformer_coords(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None: return None
    m = Chem.AddHs(m)
    try:
        if AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=10) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=100)
    except Exception:
        return None
    conf = m.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    # Heavy atoms only
    heavy = np.array([a.GetAtomicNum() != 1 for a in m.GetAtoms()])
    return coords[heavy]


def tda_features(coords):
    """Return 16-dim feature vec from H0 persistence (MST edge lengths)."""
    if coords is None or len(coords) < 3:
        return np.zeros(16)
    D = squareform(pdist(coords))
    # H0: persistence values are MST edge lengths
    mst = minimum_spanning_tree(D).toarray()
    deaths = mst[mst > 0]
    if len(deaths) == 0:
        return np.zeros(16)
    feats = [
        len(deaths),
        deaths.mean(), deaths.std(), deaths.max(), deaths.min(),
        np.median(deaths),
        np.percentile(deaths, 25), np.percentile(deaths, 75),
        (deaths > deaths.mean()).sum(),
        # Persistence entropy
        -(deaths / deaths.sum() * np.log(deaths / deaths.sum() + 1e-10)).sum(),
        # Radius of gyration
        np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean(),
        # Diameter
        D.max(),
        # Atoms within 4 of centroid
        (np.linalg.norm(coords - coords.mean(axis=0), axis=1) < 4.0).sum(),
        # Asphericity proxy
        np.var(coords[:, 0]) + np.var(coords[:, 1]) - 2 * np.var(coords[:, 2]),
        # Aspect ratios
        np.std(coords[:, 0]) / (np.std(coords[:, 1]) + 1e-6),
        np.std(coords[:, 2]) / (np.std(coords[:, 0]) + 1e-6),
    ]
    return np.array(feats)


def main():
    print("=== nb296: TDA persistent-homology fingerprints ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    print("Computing 3D conformers + TDA features for train...")
    F_tr = []
    n_fail = 0
    for i, s in enumerate(smiles_tr):
        c = conformer_coords(s)
        F_tr.append(tda_features(c))
        if c is None: n_fail += 1
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(smiles_tr)} (fail={n_fail})")
    F_tr = np.vstack(F_tr).astype(np.float32)
    print(f"Train 3D failures: {n_fail}/{len(smiles_tr)}")

    print("Computing for test...")
    F_te = []
    for i, s in enumerate(smiles_te):
        c = conformer_coords(s)
        F_te.append(tda_features(c))
    F_te = np.vstack(F_te).astype(np.float32)

    # Combine with base features
    print("Combining with base features + LGBM...")
    X_tr_base = impute(combined(smiles_tr)).astype(np.float32)
    X_te_base = impute(combined(smiles_te)).astype(np.float32)
    X_tr = np.column_stack([X_tr_base, F_tr])
    X_te = np.column_stack([X_te_base, F_te])

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    oof = np.zeros(len(y))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y[ti], eval_set=[(X_tr[vi], y[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nTDA+base LGBM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb296_tda.npy", oof)
    np.save(DATA_PROCESSED / "te_nb296_tda.npy", te_pred)

    # SLSQP
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP: OOF {best.fun:.4f}, weight(nb296)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
