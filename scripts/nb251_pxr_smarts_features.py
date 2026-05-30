"""nb251 -- Custom PXR-specific SMARTS pattern features.

Define ~50 medicinal-chemistry-relevant patterns observed in PXR ligands:
- Steroid backbone (6-6-6-5 fused)
- Macrocycle (10+ atoms)
- Sulfonyl/sulfonamide
- Tert-butyl groups
- Diaryl heterocycles
- Phenol/aniline
- Carbonyl in ring
- Ester/amide
- ...

For each compound: count matches. Train LGBM on these counts + base features.
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# PXR-relevant SMARTS patterns
PXR_SMARTS = {
    # Functional groups common in PXR ligands
    "phenol": "c1ccc(O)cc1",
    "aniline": "c1ccc(N)cc1",
    "tert_butyl": "C(C)(C)C",
    "tert_amine": "N(C)C",
    "sulfonyl": "S(=O)(=O)",
    "sulfonamide": "S(=O)(=O)N",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "carbonyl_ring": "[#6]1[#6](=O)[#6][#6][#6][#6]1",
    "ether": "[#6]-O-[#6]",
    "trifluoromethyl": "C(F)(F)F",
    "diaryl": "c1ccccc1-c2ccccc2",
    "diaryl_amine": "c1ccc(Nc2ccccc2)cc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "imidazole": "c1ncnc1",
    "thiophene": "c1ccsc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "stilbene": "C=Cc1ccccc1",
    # Sterol-like 6-6-6-5 fused
    "steroid_partial_6_6_6": "C1CCC2CCC3CCCCC3C2C1",
    "isoprenoid_chain": "C(=C)C(C)=CC",
    "hydroxyl_aliphatic": "[CH]([OH])",
    "carboxylic_acid": "C(=O)O[H]",
    "primary_amine": "N[H][H]",
    # PXR pocket-relevant
    "halogen_aromatic": "[F,Cl,Br,I]c1ccccc1",
    "tert_C_aromatic": "C(C)(C)c1ccccc1",
    # Macrocycle indicator (ring with >=10 atoms)
    "ring_10": "C1CCCCCCCCC1",
    # Complex multi-ring sterics
    "fused_3_ring": "C1CCC2(CC1)CCC1(CC2)CCCC1",
    # Specific known motifs
    "trifluoroethyl_aryl_sulfonamide": "C(F)(F)CN(S(=O)(=O)c1ccccc1)c1ccccc1",  # T0901317-ish
    "rifampicin_macrocycle": "C(=O)N1[CH]CCC1",
    # General lipophilic stretches
    "long_alkyl": "CCCCC",
    "very_long_alkyl": "CCCCCCC",
    "isopropyl": "C(C)C",
    "neopentyl": "CC(C)(C)C",
    # H-bond acceptor near hydrophobic
    "ester_near_aryl": "C(=O)Oc",
    "amide_near_aryl": "C(=O)Nc",
    # Reactive/medicinal groups
    "epoxide": "C1OC1",
    "ketone": "C(=O)C",
    "aldehyde": "C=O",
    "nitrile": "C#N",
    # Heterocycle systems
    "morpholine": "O1CCNCC1",
    "piperazine": "N1CCNCC1",
    "piperidine": "N1CCCCC1",
    "pyrrolidine": "N1CCCC1",
    "pyrazine": "c1cnccn1",
    "thiazole": "c1ncsc1",
    "oxazole": "c1ncoc1",
    "tetrazole": "[nH]1nncn1",
    # Drug-like motifs
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "triazole": "c1nncn1",
    "azole_chain": "c1ncn(C)c1",
}


def smarts_count_features(smiles_list):
    """For each SMILES, count occurrences of each SMARTS pattern."""
    patterns = []
    names = []
    for name, smarts in PXR_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                patterns.append(patt)
                names.append(name)
        except Exception:
            pass
    print(f"Compiled {len(patterns)} SMARTS patterns")

    feats = np.zeros((len(smiles_list), len(patterns)), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        for j, p in enumerate(patterns):
            feats[i, j] = len(mol.GetSubstructMatches(p))
    return feats, names


def main():
    print("=== nb251: PXR SMARTS substructure features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    print("Computing SMARTS counts...")
    Xs_tr, feat_names = smarts_count_features(smiles_tr)
    Xs_te, _ = smarts_count_features(smiles_te)
    print(f"Shape: tr={Xs_tr.shape}, te={Xs_te.shape}")

    # Correlations with pec50
    print("\nTop 15 |r| with pec50:")
    df_corr = pd.DataFrame(Xs_tr, columns=feat_names)
    df_corr["pec50"] = y_tr
    corrs = df_corr.corr(numeric_only=True)["pec50"].drop("pec50").sort_values(key=abs, ascending=False)
    print(corrs.head(15))

    # Base + SMARTS
    print("\nFeaturizing base + SMARTS...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, Xs_tr])
    X_te_aug = np.column_stack([X_te, Xs_te])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # OOF for base alone vs base+smarts vs smarts alone
    print("\n=== OOF comparisons ===")
    for name, X in [("base", X_tr), ("smarts_only", Xs_tr), ("base+smarts", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"{name:25s} OOF RAE = {rae(y_tr, oof):.4f}")

    # Final OOF + test on base+smarts
    final_oof = np.zeros(len(y_tr))
    final_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        final_oof[vi] = md.predict(X_tr_aug[vi])
        final_te_preds.append(md.predict(X_te_aug))
    final_te = np.mean(final_te_preds, axis=0)
    print(f"\nFinal nb251 OOF: {rae(y_tr, final_oof):.4f}  te_std={final_te.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb251_smarts.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb251_smarts.npy", final_te)

    # Add to 239 SLSQP and see if it helps
    print("\n=== Stack with 239 ===")
    from scipy.optimize import minimize
    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s_oof = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd_oof = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso_oof = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    # 5-way SLSQP: 239's 4 components + nb251
    M = np.column_stack([nb224_oof, nb179s_oof, mtd_oof, loso_oof, final_oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way w/ nb251: OOF={best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb251'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
