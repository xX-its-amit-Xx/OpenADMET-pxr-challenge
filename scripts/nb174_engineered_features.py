"""nb174 -- Feature engineering inspired by SHAP interpretability (nb171).

Top SHAP features all relate to:
1. Charge-binned surface area (PEOE_VSA1, 4, 5, 11, 12; SlogP_VSA, EState_VSA)
2. Molecular complexity (Chi indices, Kappa, BCUT2D)
3. Heteroatoms (count, types)

PXR LBD biology: ~1300 A^3 plastic hydrophobic pocket, binds via VdW + 4 H-bonds.

Engineered features (PXR-aware):
- pxr_pocket_fit: log(MolWt * LabuteASA / 1000) — proxy for pocket-filling
- lipophilic_share: SlogP_VSA8 / LabuteASA
- charged_surface_total: sum(PEOE_VSA1-5) / LabuteASA
- hbond_density: (NumHBA + NumHBD) / NumHeavyAtoms
- pxr_resi_match: # H-bond acceptors near aromatic ring (proxy for Gln285/His407 interactions)
- hetero_aromatic_ratio: NumAromaticHeterocycles / max(NumAromaticRings, 1)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def pxr_engineered(smiles):
    """Compute PXR-aware engineered features for a list of SMILES."""
    feats = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append([np.nan]*9); continue

        mw = Descriptors.MolWt(mol)
        labute = Descriptors.LabuteASA(mol)
        slogp_vsa8 = Descriptors.SlogP_VSA8(mol)
        peoe1 = Descriptors.PEOE_VSA1(mol)
        peoe2 = Descriptors.PEOE_VSA2(mol)
        peoe3 = Descriptors.PEOE_VSA3(mol)
        peoe4 = Descriptors.PEOE_VSA4(mol)
        peoe5 = Descriptors.PEOE_VSA5(mol)
        n_heavy = max(mol.GetNumHeavyAtoms(), 1)
        n_hba = Lipinski.NumHAcceptors(mol)
        n_hbd = Lipinski.NumHDonors(mol)
        n_arom = Lipinski.NumAromaticRings(mol)
        n_arom_hetero = Lipinski.NumAromaticHeterocycles(mol)

        # Engineered features
        pxr_pocket_fit = np.log(max(mw * labute / 1000, 1))
        lipophilic_share = slogp_vsa8 / max(labute, 1)
        charged_surface_total = (peoe1 + peoe2 + peoe3 + peoe4 + peoe5) / max(labute, 1)
        hbond_density = (n_hba + n_hbd) / n_heavy
        hetero_arom_ratio = n_arom_hetero / max(n_arom, 1)

        # PXR-specific: H-bond acceptors NEAR aromatic rings (Gln285/His407 hydrogen bond context)
        n_aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        n_acceptor_near_arom = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ('N', 'O') and not atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetIsAromatic():
                        n_acceptor_near_arom += 1
                        break
        pxr_residue_match = n_acceptor_near_arom / n_heavy

        # Lipinski "drug-likeness" partial - rifampicin-like big lipophilic
        big_lipophilic = float(mw > 400 and Crippen.MolLogP(mol) > 3)

        # Carbocyclic core (steroid-like, hyperforin-like)
        n_sat_carbo = Lipinski.NumSaturatedCarbocycles(mol)
        carbo_score = n_sat_carbo / max(n_heavy / 10, 1)

        feats.append([pxr_pocket_fit, lipophilic_share, charged_surface_total,
                       hbond_density, hetero_arom_ratio, pxr_residue_match,
                       big_lipophilic, carbo_score, n_acceptor_near_arom])
    return np.array(feats, dtype=np.float32)


FEAT_NAMES = ["pxr_pocket_fit", "lipophilic_share", "charged_surface_total",
              "hbond_density", "hetero_arom_ratio", "pxr_residue_match",
              "big_lipophilic", "carbo_score", "n_acceptor_near_arom"]


def main():
    print("=== nb174: Engineered features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Computing engineered features (train + test)...")
    Xe_tr = pxr_engineered(smiles_tr)
    Xe_te = pxr_engineered(smiles_te)
    print(f"Engineered train: {Xe_tr.shape}, test: {Xe_te.shape}")

    # Stats
    df_eng = pd.DataFrame(Xe_tr, columns=FEAT_NAMES)
    df_eng["pec50"] = y_tr
    corrs = df_eng.corr(numeric_only=True)["pec50"].drop("pec50")
    print("\nPearson r with pec50:")
    print(corrs.sort_values(ascending=False))

    print("\nFeaturizing base (combined Morgan+RDKit)...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    # NaN-fill engineered
    Xe_tr = np.nan_to_num(Xe_tr, nan=0.0)
    Xe_te = np.nan_to_num(Xe_te, nan=0.0)

    X_tr_aug = np.column_stack([X_tr, Xe_tr])
    X_te_aug = np.column_stack([X_te, Xe_te])
    print(f"X_tr_aug={X_tr_aug.shape}, X_te_aug={X_te_aug.shape}")

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # Compare BASE vs AUG OOF
    for name, X in [("base", X_tr), ("base+engineered", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"\n{name}: OOF RAE = {rae(y_tr, oof):.4f}")

    # Train on aug, predict test
    aug_oof = np.zeros(len(y_tr))
    aug_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        aug_oof[vi] = md.predict(X_tr_aug[vi])
        aug_te_preds.append(md.predict(X_te_aug))
    aug_te = np.mean(aug_te_preds, axis=0)
    print(f"\nFinal nb174 OOF RAE: {rae(y_tr, aug_oof):.4f}")
    print(f"te_std: {aug_te.std():.4f}  te_mean: {aug_te.mean():.3f}")

    np.save(DATA_PROCESSED / "oof_nb174_engineered.npy", aug_oof)
    np.save(DATA_PROCESSED / "te_nb174_engineered.npy", aug_te)
    np.save(DATA_PROCESSED / "Xe_tr_nb174.npy", Xe_tr)
    np.save(DATA_PROCESSED / "Xe_te_nb174.npy", Xe_te)
    print("Saved oof/te_nb174_engineered.npy + Xe_tr/te_nb174.npy")


if __name__ == "__main__":
    main()
