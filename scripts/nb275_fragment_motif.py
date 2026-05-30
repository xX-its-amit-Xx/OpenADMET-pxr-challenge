"""nb275 -- Fragment-motif explicit pipeline.

User's idea: don't hope NN learns abstractions; spoon-feed structural priors.

Pipeline:
1. SMILES -> fragment decomposition (BRICS-based)
2. Build fragment library from train compounds
3. For each fragment, compute its pec50 'contribution' via Free-Wilson-like
   regression: pred = beta_0 + sum(beta_i * indicator(fragment i present))
4. For each test compound, decompose, lookup fragment contributions, sum.
5. Optionally weight by PXR pocket residue compatibility:
   - Aromatic fragments -> His407, Phe281/288, Trp299 (hydrophobic stacking)
   - H-bond donors -> Gln285, Ser247 (H-bond accept partners)
   - Bulky lipophilic -> Met243/323, Leu209/308/411 (pocket walls)

Pure interpretable. Each prediction = sum of attributable fragment scores.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def get_brics_fragments(mol):
    """BRICS decomposition: returns set of fragment SMILES."""
    try:
        frags = BRICS.BRICSDecompose(mol, minFragmentSize=3)
        # Clean: remove dummy atoms
        cleaned = set()
        for f in frags:
            mol_f = Chem.MolFromSmiles(f)
            if mol_f is None: continue
            # Strip [*:N] dummies for cleaner SMILES
            for atom in mol_f.GetAtoms():
                atom.SetAtomicNum(atom.GetAtomicNum() if atom.GetAtomicNum() != 0 else 6)
            try:
                Chem.SanitizeMol(mol_f)
                cleaned.add(Chem.MolToSmiles(mol_f))
            except: pass
        return cleaned
    except Exception:
        return set()


def main():
    print("=== nb275: Fragment-motif explicit pipeline ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    # Decompose all train compounds via BRICS
    print("Decomposing train compounds via BRICS...")
    train_frags = []
    for i, s in enumerate(smiles_tr):
        mol = Chem.MolFromSmiles(s)
        if mol:
            train_frags.append(get_brics_fragments(mol))
        else:
            train_frags.append(set())
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(smiles_tr)}")

    # Aggregate fragment frequencies
    frag_counts = Counter()
    for frags in train_frags:
        for f in frags:
            frag_counts[f] += 1
    print(f"\nTotal unique fragments: {len(frag_counts)}")
    print(f"Fragments in >=5 compounds: {sum(1 for v in frag_counts.values() if v >= 5)}")

    # Keep fragments occurring in >=10 compounds (statistical power)
    common_frags = [f for f, c in frag_counts.items() if c >= 10]
    print(f"Common fragments (>=10 occurrences): {len(common_frags)}")
    frag_to_idx = {f: i for i, f in enumerate(common_frags)}

    # Build fragment presence matrix
    print("Building fragment presence matrix...")
    n_frag = len(common_frags)
    X_frag_tr = np.zeros((len(smiles_tr), n_frag), dtype=np.float32)
    for i, frags in enumerate(train_frags):
        for f in frags:
            if f in frag_to_idx:
                X_frag_tr[i, frag_to_idx[f]] = 1.0

    # Decompose test compounds
    print("Decomposing test compounds...")
    X_frag_te = np.zeros((len(smiles_te), n_frag), dtype=np.float32)
    for i, s in enumerate(smiles_te):
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        frags = get_brics_fragments(mol)
        for f in frags:
            if f in frag_to_idx:
                X_frag_te[i, frag_to_idx[f]] = 1.0
    print(f"Test fragment matrix: {X_frag_te.shape}, sparsity {(X_frag_te > 0).mean():.3f}")

    # Free-Wilson-style linear regression
    print("\n=== Linear Free-Wilson model ===")
    from sklearn.linear_model import Ridge
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    oof_fw = np.zeros(len(y_tr))
    te_fw_preds = []
    for ti, vi in folds:
        m = Ridge(alpha=1.0)
        m.fit(X_frag_tr[ti], y_tr[ti])
        oof_fw[vi] = m.predict(X_frag_tr[vi])
        te_fw_preds.append(m.predict(X_frag_te))
    te_fw = np.mean(te_fw_preds, axis=0)
    print(f"Free-Wilson OOF RAE: {rae(y_tr, oof_fw):.4f}")
    print(f"te: mean={te_fw.mean():.3f}, std={te_fw.std():.3f}")

    # LGBM on fragment features
    print("\n=== LGBM on fragment features ===")
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    oof_frag = np.zeros(len(y_tr))
    te_frag_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_frag_tr[ti], y_tr[ti], eval_set=[(X_frag_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_frag[vi] = md.predict(X_frag_tr[vi])
        te_frag_preds.append(md.predict(X_frag_te))
    te_frag = np.mean(te_frag_preds, axis=0)
    print(f"Fragment-LGBM OOF RAE: {rae(y_tr, oof_frag):.4f}")
    print(f"te: mean={te_frag.mean():.3f}, std={te_frag.std():.3f}")

    # Combined: base + fragment features
    print("\n=== Combined: base + fragment features ===")
    X_base_tr = combined(smiles_tr); X_base_tr = impute(X_base_tr)
    X_base_te = combined(smiles_te); X_base_te = impute(X_base_te)
    X_full_tr = np.column_stack([X_base_tr, X_frag_tr])
    X_full_te = np.column_stack([X_base_te, X_frag_te])
    oof_full = np.zeros(len(y_tr))
    te_full_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_full_tr[ti], y_tr[ti], eval_set=[(X_full_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_full[vi] = md.predict(X_full_tr[vi])
        te_full_preds.append(md.predict(X_full_te))
    te_full = np.mean(te_full_preds, axis=0)
    print(f"Combined+fragments OOF RAE: {rae(y_tr, oof_full):.4f}")

    np.save(DATA_PROCESSED / "oof_nb275_fw.npy", oof_fw)
    np.save(DATA_PROCESSED / "te_nb275_fw.npy", te_fw)
    np.save(DATA_PROCESSED / "oof_nb275_frag.npy", oof_frag)
    np.save(DATA_PROCESSED / "te_nb275_frag.npy", te_frag)
    np.save(DATA_PROCESSED / "oof_nb275_combined.npy", oof_full)
    np.save(DATA_PROCESSED / "te_nb275_combined.npy", te_full)

    # Top fragment contributions (Free-Wilson interpretation)
    m_final = Ridge(alpha=1.0)
    m_final.fit(X_frag_tr, y_tr)
    coefs = m_final.coef_
    top_idx = np.argsort(np.abs(coefs))[::-1][:25]
    print("\n=== Top 25 fragments by |coefficient| ===")
    for i in top_idx:
        if coefs[i] > 0.5 or coefs[i] < -0.5:
            n_present = (X_frag_tr[:, i] > 0).sum()
            print(f"  coef={coefs[i]:+.3f}  n_present={n_present:>4}  smarts={common_frags[i][:50]}")

    # Stack with nb239
    print("\n=== 5-way SLSQP w/ nb275_combined ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_full])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way OOF: {best.fun:.4f}, weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
