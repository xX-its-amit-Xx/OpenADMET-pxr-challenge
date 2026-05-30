"""nb267 -- LGBM quantile regression at 5 quantiles.

Train separate LGBM models with quantile objective at q=0.1, 0.25, 0.5, 0.75, 0.9.
For each test compound: get 5 quantile predictions → use as features in meta-LGBM.

This captures UNCERTAINTY in predictions, not just point estimate. May help
distinguish reliable vs unreliable test predictions.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
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
    except: return None


def main():
    print("=== nb267: Quantile regression LGBM ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # Train at each quantile, collect OOF + test preds
    oof_q = np.zeros((len(y_tr), len(quantiles)))
    te_q = np.zeros((len(smiles_te), len(quantiles)))

    for q_i, q in enumerate(quantiles):
        print(f"\nQuantile {q}...")
        LGBM = dict(n_estimators=1000, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=10, objective="quantile", alpha=q, n_jobs=4, random_state=42, verbose=-1)
        te_preds = []
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_tr[ti], y_tr[ti], eval_set=[(X_tr[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof_q[vi, q_i] = md.predict(X_tr[vi])
            te_preds.append(md.predict(X_te))
        te_q[:, q_i] = np.mean(te_preds, axis=0)
        print(f"  q={q}: OOF mean={oof_q[:, q_i].mean():.3f}, std={oof_q[:, q_i].std():.3f}")

    # Median (q=0.5) as standalone
    oof_med = oof_q[:, 2]
    te_med = te_q[:, 2]
    print(f"\nMedian (q=0.5) OOF RAE: {rae(y_tr, oof_med):.4f}")

    # Mean across quantiles (rich aggregation)
    oof_mean_q = oof_q.mean(axis=1)
    te_mean_q = te_q.mean(axis=1)
    print(f"Mean-of-quantiles OOF RAE: {rae(y_tr, oof_mean_q):.4f}")

    # Uncertainty = q0.9 - q0.1 (interquantile range)
    oof_unc = oof_q[:, 4] - oof_q[:, 0]
    te_unc = te_q[:, 4] - te_q[:, 0]
    print(f"Uncertainty (q90-q10): mean={oof_unc.mean():.3f}, te_mean={te_unc.mean():.3f}")

    # Stack: use all 5 quantiles as features in LGBM regression
    print("\n=== Meta-LGBM on quantile features ===")
    LGBM_META = dict(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20,
                     objective="mae", n_jobs=4, random_state=42, verbose=-1)
    oof_meta = np.zeros(len(y_tr))
    te_metas = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM_META)
        md.fit(oof_q[ti], y_tr[ti], eval_set=[(oof_q[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[vi] = md.predict(oof_q[vi])
        te_metas.append(md.predict(te_q))
    te_meta = np.mean(te_metas, axis=0)
    print(f"Meta-LGBM OOF RAE: {rae(y_tr, oof_meta):.4f}")

    np.save(DATA_PROCESSED / "oof_nb267_quantile.npy", oof_meta)
    np.save(DATA_PROCESSED / "te_nb267_quantile.npy", te_meta)
    np.save(DATA_PROCESSED / "oof_nb267_q_features.npy", oof_q)
    np.save(DATA_PROCESSED / "te_nb267_q_features.npy", te_q)

    # Stack with 239
    print("\n=== Stack nb267 with 239 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    M = np.column_stack([nb224, nb179s, mtd, loso, oof_meta])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb267'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
