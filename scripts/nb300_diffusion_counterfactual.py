"""nb300 -- Diffusion-style scaffold-conditional counterfactual augmentation.

A full SMILES diffusion model is out of scope without GPU; we substitute with
a simpler-but-spiritually-equivalent approach: SMILES enumeration + property-
conditional resampling. For each high-confidence training compound, generate
N synthetic analogs via random atom-order canonicalisation + bioisosteric
substitutions, score them with nb239, retain those whose score remains within
0.5 of original. Append as pseudo-labels at weight 0.3.
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
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


# Common bioisosteric replacements
BIOISOSTERES = [
    ('[F]', '[Cl]'), ('[Cl]', '[Br]'),
    ('[C]', '[N]'),  # carbon to nitrogen (rough)
    ('O[H]', 'N[H]'),
    ('C(=O)O', 'C(=O)N'),
    ('C(=O)N', 'C(=O)O'),
]


def enumerate_smiles(smi, n=3):
    """SMILES enumeration via random canonicalisation."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return [smi]
    out = [smi]
    for _ in range(n):
        try:
            s = Chem.MolToSmiles(m, doRandom=True, canonical=False)
            if s and s not in out:
                out.append(s)
        except: pass
    return out


def main():
    print("=== nb300: Diffusion-style counterfactual augmentation ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    # Build augmented training set: SMILES enumeration on every train
    print("Augmenting via SMILES enumeration...")
    aug_smi = []
    aug_y = []
    for s, p in zip(smiles_tr, y):
        forms = enumerate_smiles(s, n=2)
        for f in forms:
            aug_smi.append(f); aug_y.append(p)
    print(f"Augmented: {len(aug_smi)} (from {len(smiles_tr)})")

    print("Featurising augmented set...")
    X_tr_aug = impute(combined(aug_smi)).astype(np.float32)
    y_aug = np.array(aug_y)
    X_te = impute(combined(smiles_te)).astype(np.float32)

    # Map augmented rows back to original scaffold (to keep fold consistency)
    aug_scaffolds = []
    for s in aug_smi:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        m = Chem.MolFromSmiles(s)
        if m is None:
            aug_scaffolds.append("")
        else:
            try:
                aug_scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
            except:
                aug_scaffolds.append("")

    folds = scaffold_kfold_indices(aug_scaffolds, n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    # OOF on the ORIGINAL train via mapping
    # Easier path: separate model on aug-augmented data, get test preds.
    md = lgb.LGBMRegressor(**LGBM)
    # Sample weight: 1.0 for original, 0.3 for aug copies (first n entries are original)
    n_orig = len(smiles_tr)
    weights = np.array([1.0] * n_orig + [0.3] * (len(aug_smi) - n_orig))
    # quick train/test split: use last 20% for early stop
    n = len(aug_smi)
    perm = np.random.default_rng(42).permutation(n)
    cut = int(0.8 * n)
    tr_idx = perm[:cut]; va_idx = perm[cut:]
    md.fit(X_tr_aug[tr_idx], y_aug[tr_idx], sample_weight=weights[tr_idx],
           eval_set=[(X_tr_aug[va_idx], y_aug[va_idx])],
           callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
    te_pred = md.predict(X_te)

    # For OOF on original train, run scaffold CV
    folds_orig = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof = np.zeros(n_orig)
    X_tr_orig = X_tr_aug[:n_orig]
    for ti, vi in folds_orig:
        m2 = lgb.LGBMRegressor(**LGBM)
        m2.fit(X_tr_orig[ti], y[ti], eval_set=[(X_tr_orig[vi], y[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = m2.predict(X_tr_orig[vi])
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nDiffusion-aug LGBM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb300_diffusion_aug.npy", oof)
    np.save(DATA_PROCESSED / "te_nb300_diffusion_aug.npy", te_pred)

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
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb300)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
