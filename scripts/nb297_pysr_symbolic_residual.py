"""nb297 -- Symbolic-regression residual correction over nb239.

Fit a small symbolic expression to the (RDKit descriptors -> nb239 residual)
mapping. Tries pysr if installed; otherwise falls back to a sparse polynomial
LASSO over hand-picked descriptors (low-degree interactions) which is in spirit
the same: low-complexity functional form to discourage overfitting.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb297: Symbolic-regression-style residual correction ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te  = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    resid = y - nb239_oof
    print(f"nb239 residual: mean={resid.mean():.3f}  std={resid.std():.3f}")

    # RDKit descriptors only
    X_tr = impute(rdkit_desc(smiles_tr)).astype(np.float32)
    X_te = impute(rdkit_desc(smiles_te)).astype(np.float32)

    # Standardise + degree-2 polynomial expansion (interactions)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

    # Cap dimensionality: top variance
    var_idx = np.argsort(X_tr_s.var(axis=0))[::-1][:20]
    X_tr_s = X_tr_s[:, var_idx]; X_te_s = X_te_s[:, var_idx]
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_p = poly.fit_transform(X_tr_s).astype(np.float64)
    X_te_p = poly.transform(X_te_s).astype(np.float64)
    print(f"Polynomial (interactions only) expansion: {X_tr_p.shape[1]} terms")

    # Single LASSO with fixed alpha — much faster than LassoCV per fold
    from sklearn.linear_model import Lasso
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof_correction = np.zeros(len(y))
    te_correction_preds = []
    for ti, vi in folds:
        mdl = Lasso(alpha=0.01, max_iter=5000, precompute=False)
        mdl.fit(X_tr_p[ti], resid[ti].astype(np.float64))
        oof_correction[vi] = mdl.predict(X_tr_p[vi])
        te_correction_preds.append(mdl.predict(X_te_p))
    te_correction = np.mean(te_correction_preds, axis=0)

    final_oof = nb239_oof + 0.7 * oof_correction
    final_te  = nb239_te  + 0.7 * te_correction

    r_base = rae(y, nb239_oof)
    r = rae(y, final_oof)
    sp, _ = spearmanr(y, final_oof)
    print(f"\nBase nb239 OOF: {r_base:.4f}")
    print(f"+ symbolic-LASSO correction OOF: {r:.4f}  Spearman={sp:.4f}  te_std={final_te.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb297_pysr.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb297_pysr.npy", final_te)

    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, final_oof])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb297)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
