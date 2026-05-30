"""nb305 -- Mixture of Pharmacophore Experts (MoPE).

Idea C: PXR's promiscuous 1300A^3 pocket binds multiple PHARMACOPHORE
FAMILIES, not one ligand type. Train ONE LGBM per family using only the
matching training compounds. At inference, classify each test compound by
SMARTS family and route to the corresponding expert. Fall back to global
LGBM when a family has too few training members.

7 PXR pharmacophore classes (from Teotico 2008, Cell 2022 review):
  C1: Rifampin-class macrocyclic ansamycin
  C2: Steroid skeleton (cyclopenta-perhydrophenanthrene)
  C3: Statin / fatty-acid-mimetic (long lipophilic chain + carboxyl)
  C4: Taxane / polycyclic-aromatic
  C5: Biphenyl-azole / heterobiaryl
  C6: Sulfonamide / sulfonyl-aryl
  C7: Other (catch-all - uses global model)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# Pharmacophore class SMARTS (rough but distinctive)
CLASS_SMARTS = {
    "C1_ansamycin": ["[OX2H1]C1=C(O)C(=O)c2c(c1=O)C(=O)[#6]2"],   # napthoquinone-naphthol core (rifampicin)
    "C2_steroid":   ["C1CC2CCC3CCCC4CCCC1C2C34", "C1CC2CCC3CCCCC3C2CC1"],  # steran + reduced
    "C3_statin":    ["C(=O)[OH]", "[CX4][CX4][CX4][CX4][CX4][CX4][CX4][CX4]"],  # carboxyl + long chain
    "C4_taxane":    ["C1CC2CCC3(C)C1CC3C(=O)O2"],  # taxane core approximation
    "C5_biaryl_azole": ["c1ccc(-c2ncc[nH]2)cc1", "c1ccc(-c2nnc[nH]2)cc1", "c1ccc(-c2nccs2)cc1"],
    "C6_sulfonamide":  ["S(=O)(=O)N", "c1ccc(S(=O)(=O))cc1"],
    "C7_other":     ["[*]"],  # always matches
}

CLASS_NAMES = list(CLASS_SMARTS.keys())


def classify(smi):
    """Return one-hot 7-d vector of class membership."""
    out = np.zeros(7, dtype=np.float32)
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        out[6] = 1.0
        return out
    for ci, c_name in enumerate(CLASS_NAMES):
        for patt in CLASS_SMARTS[c_name]:
            try:
                p = Chem.MolFromSmarts(patt)
                if p is not None and m.HasSubstructMatch(p):
                    out[ci] = 1.0; break
            except: pass
    if out.sum() == 0:
        out[6] = 1.0
    return out


def main():
    print("=== nb305: Mixture of Pharmacophore Experts ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    print("Classifying compounds into pharmacophore classes...")
    class_tr = np.stack([classify(s) for s in smiles_tr])
    class_te = np.stack([classify(s) for s in smiles_te])
    print("Train class distribution:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n}: {int(class_tr[:, i].sum())} train,  {int(class_te[:, i].sum())} test")

    print("\nFeaturising base features...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)

    # Global model (fallback)
    print("\n--- Training GLOBAL fallback model ---")
    global_oof = np.zeros(len(y))
    global_te_preds = []
    for ti, vi in folds:
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X_tr[ti], y[ti], eval_set=[(X_tr[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        global_oof[vi] = m.predict(X_tr[vi])
        global_te_preds.append(m.predict(X_te))
    global_te = np.mean(global_te_preds, axis=0)
    print(f"  Global OOF RAE: {rae(y, global_oof):.4f}")

    # Per-class models
    print("\n--- Training PER-CLASS expert models ---")
    class_oof_preds = np.zeros((len(y), 7))
    class_te_preds = np.zeros((len(smiles_te), 7))
    class_used = np.zeros(7, dtype=bool)
    for c_idx, c_name in enumerate(CLASS_NAMES):
        members = np.where(class_tr[:, c_idx] > 0)[0]
        if len(members) < 80 or c_name == "C7_other":
            print(f"  {c_name}: SKIP (only {len(members)} members, using global)")
            class_oof_preds[:, c_idx] = global_oof
            class_te_preds[:, c_idx] = global_te
            continue
        print(f"  {c_name}: train on {len(members)} members")
        class_used[c_idx] = True
        # Scaffold-CV within class members
        c_smiles = [smiles_tr[i] for i in members]
        c_scaff  = [tr['scaffold'].tolist()[i] for i in members]
        c_X = X_tr[members]; c_y = y[members]
        c_folds = scaffold_kfold_indices(c_scaff, n_splits=min(5, max(2, len(set(c_scaff)) // 5)))
        oof_c = np.zeros(len(members))
        te_c_preds = []
        for ti, vi in c_folds:
            m = lgb.LGBMRegressor(**LGBM)
            m.fit(c_X[ti], c_y[ti], eval_set=[(c_X[vi], c_y[vi])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof_c[vi] = m.predict(c_X[vi])
            te_c_preds.append(m.predict(X_te))
        te_c = np.mean(te_c_preds, axis=0)
        # Place class-OOF back into full-length array; non-members get global pred
        full_oof_c = global_oof.copy()
        for j, mi in enumerate(members):
            full_oof_c[mi] = oof_c[j]
        class_oof_preds[:, c_idx] = full_oof_c
        class_te_preds[:, c_idx] = te_c
        print(f"    class {c_name} OOF RAE on members: {rae(c_y, oof_c):.4f}")

    # Route: for each compound, use weighted average of expert preds
    # weights = SMARTS match (or all-to-global if no match)
    print("\n--- Routing & blending ---")
    route_oof = np.zeros(len(y))
    route_te  = np.zeros(len(smiles_te))
    for i in range(len(y)):
        ws = class_tr[i] * class_used
        if ws.sum() == 0:
            route_oof[i] = global_oof[i]
        else:
            route_oof[i] = (class_oof_preds[i] * ws).sum() / ws.sum()
    for i in range(len(smiles_te)):
        ws = class_te[i] * class_used
        if ws.sum() == 0:
            route_te[i] = global_te[i]
        else:
            route_te[i] = (class_te_preds[i] * ws).sum() / ws.sum()

    r = rae(y, route_oof)
    sp, _ = spearmanr(y, route_oof)
    print(f"MoPE OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={route_te.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb305_mope.npy", route_oof)
    np.save(DATA_PROCESSED / "te_nb305_mope.npy", route_te)

    # SLSQP blend
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, route_oof])
    def loss_fn(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb305)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
