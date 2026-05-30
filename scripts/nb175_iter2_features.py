"""nb175 -- Iteration 2 of feature engineering after nb174 findings.

nb174 found:
  pxr_pocket_fit (MW * surface): r=+0.459 (STRONG)
  hbond_density: r=-0.403 (STRONG NEGATIVE - PXR hates polarity)
  charged_surface_total: r=-0.266
  carbo_score: r=+0.153

nb175 enhances with: lipophilic-mass interactions, hetero/carbon ratio inverted,
flexibility, macrolide-like signature, and stacking with nb224 OOF as feature.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def pxr_engineered_v2(smiles):
    feats = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append([np.nan]*15); continue

        mw = Descriptors.MolWt(mol)
        labute = Descriptors.LabuteASA(mol)
        logp = Crippen.MolLogP(mol)
        n_heavy = max(mol.GetNumHeavyAtoms(), 1)
        n_hba = Lipinski.NumHAcceptors(mol)
        n_hbd = Lipinski.NumHDonors(mol)
        n_arom = Lipinski.NumAromaticRings(mol)
        n_arom_hetero = Lipinski.NumAromaticHeterocycles(mol)
        n_rot = Lipinski.NumRotatableBonds(mol)
        n_rings = Lipinski.NumSaturatedCarbocycles(mol) + Lipinski.NumSaturatedHeterocycles(mol) + n_arom
        n_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        n_hetero = n_heavy - n_C
        peoe4 = Descriptors.PEOE_VSA4(mol)
        slogp8 = Descriptors.SlogP_VSA8(mol)
        slogp10 = Descriptors.SlogP_VSA10(mol)
        peoe_sum = sum(getattr(Descriptors, f"PEOE_VSA{i}")(mol) for i in range(1, 14))

        # Original strong feature
        pxr_pocket_fit = np.log(max(mw * labute / 1000, 1))

        # New v2 features
        lipo_mass = mw * logp  # "lipophilic mass" - rifampicin/paclitaxel scale (mw~800, logp~3-5)
        carbon_share = n_C / n_heavy  # fraction carbon (high = lipophilic)
        peoe4_share = peoe4 / max(peoe_sum, 1)  # share of dominant SHAP bin
        lipo_per_atom = slogp8 / n_heavy
        super_lipo_per_atom = slogp10 / n_heavy  # SlogP_VSA10 = very lipophilic surface

        # Polar penalty (since PXR hates polarity)
        hbond_density = (n_hba + n_hbd) / n_heavy  # confirmed negative
        polar_atoms = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ('N', 'O', 'F', 'Cl', 'S')) / n_heavy

        # Flexibility (rifampicin is flexible; macrolides flexible)
        rot_per_heavy = n_rot / n_heavy
        rings_per_heavy = n_rings / n_heavy

        # Macrolide-like: large ring (>10 atoms) present?
        ring_info = mol.GetRingInfo()
        large_ring = float(any(len(r) >= 10 for r in ring_info.AtomRings()))

        # Macrocycle and bridged
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

        # Steroid-like (4 fused rings, low logP penalty)
        steroid_like = float(n_rings >= 4 and labute > 200)

        feats.append([
            pxr_pocket_fit, lipo_mass, carbon_share, peoe4_share,
            lipo_per_atom, super_lipo_per_atom, hbond_density, polar_atoms,
            rot_per_heavy, rings_per_heavy, large_ring, n_spiro,
            steroid_like, mw, logp,
        ])
    return np.array(feats, dtype=np.float32)


FEAT_NAMES = ["pxr_pocket_fit", "lipo_mass", "carbon_share", "peoe4_share",
              "lipo_per_atom", "super_lipo_per_atom", "hbond_density", "polar_atoms",
              "rot_per_heavy", "rings_per_heavy", "large_ring", "n_spiro",
              "steroid_like", "mw", "logp"]


def main():
    print("=== nb175: Iteration 2 engineered features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Computing v2 engineered features...")
    Xe_tr = pxr_engineered_v2(smiles_tr)
    Xe_te = pxr_engineered_v2(smiles_te)
    Xe_tr = np.nan_to_num(Xe_tr, nan=0.0)
    Xe_te = np.nan_to_num(Xe_te, nan=0.0)
    print(f"Shape: {Xe_tr.shape}")

    df_eng = pd.DataFrame(Xe_tr, columns=FEAT_NAMES)
    df_eng["pec50"] = y_tr
    print("\nPearson r with pec50:")
    print(df_eng.corr(numeric_only=True)["pec50"].drop("pec50").sort_values(ascending=False))

    print("\nFeaturizing base...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    X_tr_aug = np.column_stack([X_tr, Xe_tr])
    X_te_aug = np.column_stack([X_te, Xe_te])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n=== OOF comparison ===")
    for name, X in [("base", X_tr), ("base+v2_eng", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"{name}: OOF RAE = {rae(y_tr, oof):.4f}")

    # Final train on aug + save
    final_oof = np.zeros(len(y_tr))
    final_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        final_oof[vi] = md.predict(X_tr_aug[vi])
        final_te_preds.append(md.predict(X_te_aug))
    final_te = np.mean(final_te_preds, axis=0)
    print(f"\nFinal nb175 OOF RAE: {rae(y_tr, final_oof):.4f}")
    np.save(DATA_PROCESSED / "oof_nb175_eng_v2.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb175_eng_v2.npy", final_te)
    np.save(DATA_PROCESSED / "Xe_tr_nb175.npy", Xe_tr)
    np.save(DATA_PROCESSED / "Xe_te_nb175.npy", Xe_te)

    # === STACK with nb224 ===
    print("\n=== Stacking engineered features ONTO nb224 OOF ===")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb224_te  = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")

    # Stack: input = nb224 prediction + engineered features
    X_stk_tr = np.column_stack([nb224_oof.reshape(-1,1), Xe_tr])
    X_stk_te = np.column_stack([nb224_te.reshape(-1,1),  Xe_te])
    print(f"stacked input shape: {X_stk_tr.shape}")
    stk_oof = np.zeros(len(y_tr))
    stk_te_preds = []
    STK_PARAMS = dict(n_estimators=500, num_leaves=15, learning_rate=0.05, min_child_samples=20,
                      objective="mae", n_jobs=4, random_state=42, verbose=-1)
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**STK_PARAMS)
        md.fit(X_stk_tr[ti], y_tr[ti], eval_set=[(X_stk_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        stk_oof[vi] = md.predict(X_stk_tr[vi])
        stk_te_preds.append(md.predict(X_stk_te))
    stk_te = np.mean(stk_te_preds, axis=0)
    r_stk = rae(y_tr, stk_oof)
    r_nb224 = rae(y_tr, nb224_oof)
    print(f"Stacked (nb224 + eng_v2) OOF RAE: {r_stk:.4f}")
    print(f"nb224 alone OOF RAE:              {r_nb224:.4f}")
    print(f"Delta: {r_stk - r_nb224:+.4f}")
    np.save(DATA_PROCESSED / "oof_nb175_stack_nb224_eng.npy", stk_oof)
    np.save(DATA_PROCESSED / "te_nb175_stack_nb224_eng.npy", stk_te)

    if r_stk < r_nb224:
        print("\n*** STACKING HELPED. Saving as candidate submission. ***")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'].tolist() if 'Molecule Name' in te_df.columns else te_df['name'].tolist(),
            'pEC50': stk_te,
        })
        from pxr.paths import SUBMISSIONS
        sub.to_csv(SUBMISSIONS / "234_nb224_eng_stack.csv", index=False)
        print(f"Saved 234_nb224_eng_stack.csv")
    else:
        print("\n--- Stacking did not improve. Saving anyway for reference. ---")


if __name__ == "__main__":
    main()
