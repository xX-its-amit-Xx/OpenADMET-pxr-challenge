"""nb288 -- Gaussian Process residual corrector on top of nb239.

GP learns the residual y - nb239_pred on a subsampled per-fold neighborhood
to recalibrate and produce per-compound uncertainty.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.decomposition import TruncatedSVD
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def main():
    print("=== nb288: GP residual corrector on top of nb239 ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    print(f"nb239 OOF RAE: {rae(y_tr, nb239_oof):.4f}")

    print("Featurizing...")
    X_tr = impute(combined(smiles_tr))
    X_te = impute(combined(smiles_te))

    # Reduce dim via SVD for GP scaling (RBF on 2265 dims with N>=300 is slow but feasible).
    print("SVD reducing 2265 -> 64 dims...")
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_r = svd.fit_transform(X_tr.astype(np.float32))
    X_te_r = svd.transform(X_te.astype(np.float32))
    # Standardize for RBF length-scale stability
    mu = X_tr_r.mean(0); sd = X_tr_r.std(0) + 1e-6
    X_tr_r = (X_tr_r - mu) / sd
    X_te_r = (X_te_r - mu) / sd

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof_gp = nb239_oof.copy()
    te_gp_folds = []
    te_std_folds = []
    rng = np.random.default_rng(42)

    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2)) \
             + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e1))
    SAMPLE = 300

    for fi, (ti, vi) in enumerate(folds):
        # Residual on the training fold (using nb239 already-OOF preds for ti
        # would be cleaner, but nb239 OOF preds at ti are produced when ti was
        # in the val role of some other fold — they are valid OOF preds).
        resid_train = y_tr[ti] - nb239_oof[ti]
        # Subsample for GP O(n^3)
        n = min(SAMPLE, len(ti))
        sel = rng.choice(len(ti), n, replace=False)
        Xs = X_tr_r[ti][sel]
        ys = resid_train[sel]

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=42)
        try:
            gp.fit(Xs, ys)
        except Exception as e:
            print(f"  fold {fi}: GP fit failed ({e}); skipping correction.")
            te_gp_folds.append(nb239_te.copy())
            te_std_folds.append(np.zeros(len(smiles_te)))
            continue

        # Val
        resid_val, std_val = gp.predict(X_tr_r[vi], return_std=True)
        oof_gp[vi] = nb239_oof[vi] + resid_val
        # Test
        resid_te, std_te = gp.predict(X_te_r, return_std=True)
        te_gp_folds.append(nb239_te + resid_te)
        te_std_folds.append(std_te)
        print(f"  fold {fi}: GP kernel={gp.kernel_}, val_resid_mean={resid_val.mean():.3f}, "
              f"std_mean={std_val.mean():.3f}")

    te_gp = np.mean(te_gp_folds, axis=0)
    te_std = np.mean(te_std_folds, axis=0)

    r_base = rae(y_tr, nb239_oof); r_gp = rae(y_tr, oof_gp)
    sp_b, _ = spearmanr(y_tr, nb239_oof); sp_g, _ = spearmanr(y_tr, oof_gp)
    print(f"\nBase   OOF RAE={r_base:.4f}  Spearman={sp_b:.4f}")
    print(f"GP-cor OOF RAE={r_gp:.4f}  Spearman={sp_g:.4f}  te_std={te_gp.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb288_gp_corrected.npy", oof_gp)
    np.save(DATA_PROCESSED / "te_nb288_gp_corrected.npy", te_gp)
    np.save(DATA_PROCESSED / "te_nb288_gp_std.npy", te_std)

    # SLSQP 5-way: nb224, nb179_stack, multi_template_delta, delta_loso, nb288_gp
    print("\n=== 5-way SLSQP with nb288 ===")
    try:
        nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
        nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
        mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
        loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
        M = np.column_stack([nb224, nb179s, mtd, loso, oof_gp])
        def loss(w): return rae(y_tr, M @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * 5
        best = None
        for seed in range(80):
            w0 = np.random.default_rng(seed).dirichlet(np.ones(5))
            r = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'ftol': 1e-9, 'maxiter': 200})
            if best is None or r.fun < best.fun: best = r
        print(f"SLSQP OOF: {best.fun:.4f}  weights={np.round(best.x, 4)}")
    except Exception as e:
        print(f"SLSQP skipped: {e}")


if __name__ == "__main__":
    main()
