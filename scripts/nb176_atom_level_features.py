"""nb176 -- Atom-level + known-PXR-binder similarity features.

Three new feature categories:

A) **Atom-level aggregates** (medicinal chemist's per-atom analysis):
   - Per-atom: PEOE charge, EState, lipophilicity (Crippen), partial charge
   - Aggregate by atom type (C, N, O, F, S, halogen) and ring/non-ring
   - Top-k atom values (most lipophilic atom, most charged atom)

B) **Known PXR binder Tanimoto similarity**:
   - Tanimoto similarity to each of 10 canonical PXR ligands
   - Rifampicin, hyperforin, paclitaxel, SR12813, T0901317, mifepristone,
     pregnenolone-16α-carbonitrile, clotrimazole, lithocholic acid, dexamethasone
   - These span the PXR ligand archetypes (steroid-like, macrolide, lipophilic NR mod)

C) **PXR pocket subpocket scores**:
   - Score for matching three PXR subpockets:
     * Lipophilic core: requires SlogP_VSA8+ heavy
     * Polar anchor (Gln285/Ser247): requires 1-2 H-bond acceptors near aromatic
     * Steric edge (His407/Arg410): requires aromatic ring with H-bond capability
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdPartialCharges
from rdkit.Chem.EState import EStateIndices
from rdkit.DataStructs import TanimotoSimilarity

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# Canonical PXR ligands (literature)
KNOWN_PXR_LIGANDS = {
    "rifampicin": "CC1C(C(C(C=C(C(C(C=CC=C(C(C(C2=C(C3=C(C(=C2O)C)O)C(=O)C=C(N3)C)/C)O)C)OC(=O)C)C)O)C)O)C(=O)O1",
    "hyperforin": "CC(=CCCC(C)(C(C(=O)C(C=C)(C)C)O)CC=C(C)C)C",
    "paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)O",
    "sr12813": "CCOP(=O)(CC(c1cc(c(c(c1)C(C)(C)C)O)C(C)(C)C)P(=O)(OCC)OCC)OCC",
    "t0901317": "OC(C(=O)Nc1ccc(C(F)(F)F)cc1S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "mifepristone": "CC#CC1(CCC2C1(CC(C3=C4CCC(=O)CC4=CCC23)C5=CC=C(C=C5)N(C)C)C)O",
    "pcn": "N#CN1CCC2C1CCC3(C)C2CCC4C3CCC4=O",
    "clotrimazole": "Clc1ccccc1C(c2ccccc2)(c3ccccc3)n4ccnc4",
    "lithocholic_acid": "CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C",
    "dexamethasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",
}


def precompute_known_fps(radius=2, n_bits=2048):
    fps = {}
    for name, smi in KNOWN_PXR_LIGANDS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps[name] = None; continue
        fps[name] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fps


def atom_level_features(mol):
    """Return per-atom features then aggregate."""
    if mol is None:
        return [np.nan] * 20

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms == 0:
        return [np.nan] * 20

    # Compute charges
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for a in mol.GetAtoms():
            try:
                c = float(a.GetDoubleProp("_GasteigerCharge"))
                if np.isfinite(c):
                    charges.append(c)
            except (KeyError, ValueError):
                pass
        charges = np.array(charges) if charges else np.array([0.0])
    except Exception:
        charges = np.array([0.0])

    # Crippen contributions per atom
    try:
        crippen = Crippen._GetAtomContribs(mol)
        logp_contribs = np.array([c[0] for c in crippen])
    except Exception:
        logp_contribs = np.array([0.0])

    # EState
    try:
        estate = EStateIndices(mol)
    except Exception:
        estate = np.array([0.0])

    # Per-atom-type aggregates
    n_aromatic_C = 0
    n_aromatic_hetero = 0
    n_sp3_C = 0
    n_polar_atom = 0
    n_halogen = 0
    n_ring_atom = 0
    sp3_share = 0.0

    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym == "C":
            if a.GetIsAromatic():
                n_aromatic_C += 1
            elif a.GetHybridization() == Chem.HybridizationType.SP3:
                n_sp3_C += 1
        elif sym in ("N", "O", "S"):
            n_polar_atom += 1
            if a.GetIsAromatic():
                n_aromatic_hetero += 1
        elif sym in ("F", "Cl", "Br", "I"):
            n_halogen += 1
        if a.IsInRing():
            n_ring_atom += 1

    sp3_share = n_sp3_C / max(n_atoms, 1)
    aromatic_C_share = n_aromatic_C / max(n_atoms, 1)
    aromatic_hetero_share = n_aromatic_hetero / max(n_atoms, 1)
    polar_share = n_polar_atom / max(n_atoms, 1)
    halogen_share = n_halogen / max(n_atoms, 1)
    ring_share = n_ring_atom / max(n_atoms, 1)

    # Charge aggregates
    max_pos_charge = charges.max() if len(charges) else 0.0
    max_neg_charge = charges.min() if len(charges) else 0.0
    charge_range = max_pos_charge - max_neg_charge

    # LogP atom-level
    top3_lipophilic_atoms = np.sort(logp_contribs)[::-1][:3].sum() if len(logp_contribs) >= 3 else logp_contribs.sum()
    most_lipophilic = logp_contribs.max() if len(logp_contribs) else 0.0
    most_polar = logp_contribs.min() if len(logp_contribs) else 0.0

    # EState aggregates
    estate_max = estate.max() if len(estate) else 0.0
    estate_min = estate.min() if len(estate) else 0.0
    estate_range = estate_max - estate_min

    # Specific PXR pocket scores
    # H-bond acceptor next to aromatic ring (Gln285/His407 anchor)
    hba_arom_anchor = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "O") and not a.GetIsAromatic() and a.GetTotalNumHs() == 0:
            # Acceptor only (no Hs)
            for nbr in a.GetNeighbors():
                if nbr.GetIsAromatic():
                    hba_arom_anchor += 1
                    break

    return [
        max_pos_charge, max_neg_charge, charge_range,
        most_lipophilic, most_polar, top3_lipophilic_atoms,
        estate_max, estate_min, estate_range,
        sp3_share, aromatic_C_share, aromatic_hetero_share,
        polar_share, halogen_share, ring_share,
        n_aromatic_C, n_aromatic_hetero, n_halogen,
        hba_arom_anchor, n_atoms,
    ]


FEAT_NAMES_ATOM = [
    "max_pos_charge", "max_neg_charge", "charge_range",
    "most_lipophilic", "most_polar", "top3_lipophilic_atoms",
    "estate_max", "estate_min", "estate_range",
    "sp3_share", "aromatic_C_share", "aromatic_hetero_share",
    "polar_share", "halogen_share", "ring_share",
    "n_aromatic_C", "n_aromatic_hetero", "n_halogen",
    "hba_arom_anchor", "n_atoms",
]


def known_pxr_similarity(mol, known_fps, radius=2, n_bits=2048):
    """Tanimoto similarity to each known PXR ligand."""
    if mol is None:
        return [0.0] * len(KNOWN_PXR_LIGANDS)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return [TanimotoSimilarity(fp, known_fps[name]) if known_fps[name] is not None else 0.0
            for name in KNOWN_PXR_LIGANDS]


def featurize_atom_and_sim(smiles_list, known_fps):
    feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        atom_f = atom_level_features(mol)
        sim_f = known_pxr_similarity(mol, known_fps)
        feats.append(atom_f + sim_f)
    return np.array(feats, dtype=np.float32)


def main():
    print("=== nb176: Atom-level + known PXR binder similarity ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Precomputing known PXR binder fingerprints...")
    known_fps = precompute_known_fps()
    n_known = sum(1 for v in known_fps.values() if v is not None)
    print(f"  {n_known}/{len(KNOWN_PXR_LIGANDS)} known ligand FPs ready")

    print("Computing atom-level + similarity features...")
    Xf_tr = featurize_atom_and_sim(smiles_tr, known_fps)
    Xf_te = featurize_atom_and_sim(smiles_te, known_fps)
    Xf_tr = np.nan_to_num(Xf_tr, nan=0.0)
    Xf_te = np.nan_to_num(Xf_te, nan=0.0)
    print(f"Shape: train={Xf_tr.shape}, test={Xf_te.shape}")

    all_feat_names = FEAT_NAMES_ATOM + [f"sim_{n}" for n in KNOWN_PXR_LIGANDS]
    print(f"\nFeatures: {len(all_feat_names)}")

    df_f = pd.DataFrame(Xf_tr, columns=all_feat_names)
    df_f["pec50"] = y_tr
    print("\nTop 20 by |Pearson r| with pec50:")
    corrs = df_f.corr(numeric_only=True)["pec50"].drop("pec50").sort_values(key=abs, ascending=False)
    print(corrs.head(20))

    # LGBM on this feature set alone
    print("\nFeaturizing base (combined Morgan+RDKit) for stacking comparison...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, Xf_tr])
    X_te_aug = np.column_stack([X_te, Xf_te])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n=== OOF comparisons ===")
    for name, X in [("base", X_tr), ("atom+sim only", Xf_tr), ("base+atom+sim", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"{name:25s} OOF RAE = {rae(y_tr, oof):.4f}")

    # Final train: base+atom+sim
    final_oof = np.zeros(len(y_tr))
    final_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        final_oof[vi] = md.predict(X_tr_aug[vi])
        final_te_preds.append(md.predict(X_te_aug))
    final_te = np.mean(final_te_preds, axis=0)
    print(f"\nFinal nb176 OOF RAE: {rae(y_tr, final_oof):.4f}")
    np.save(DATA_PROCESSED / "oof_nb176_atom_sim.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb176_atom_sim.npy", final_te)
    np.save(DATA_PROCESSED / "Xf_tr_nb176.npy", Xf_tr)
    np.save(DATA_PROCESSED / "Xf_te_nb176.npy", Xf_te)

    # Stack with nb224
    print("\n=== Stacking nb224 + atom+sim features ===")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb224_te = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    X_stk_tr = np.column_stack([nb224_oof.reshape(-1,1), Xf_tr])
    X_stk_te = np.column_stack([nb224_te.reshape(-1,1),  Xf_te])
    stk_oof = np.zeros(len(y_tr))
    stk_te_preds = []
    STK = dict(n_estimators=500, num_leaves=15, learning_rate=0.05, min_child_samples=20,
               objective="mae", n_jobs=4, random_state=42, verbose=-1)
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**STK)
        md.fit(X_stk_tr[ti], y_tr[ti], eval_set=[(X_stk_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        stk_oof[vi] = md.predict(X_stk_tr[vi])
        stk_te_preds.append(md.predict(X_stk_te))
    stk_te = np.mean(stk_te_preds, axis=0)
    r_stk = rae(y_tr, stk_oof); r_nb224 = rae(y_tr, nb224_oof)
    print(f"Stacked: {r_stk:.4f}  vs nb224: {r_nb224:.4f}  delta: {r_stk-r_nb224:+.4f}")
    if r_stk < r_nb224:
        print("*** STACKING HELPED ***")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'].tolist() if 'Molecule Name' in te_df.columns else te_df['name'].tolist(),
            'pEC50': stk_te,
        })
        sub.to_csv(SUBMISSIONS / "235_nb224_atom_sim_stack.csv", index=False)
        print("Saved 235_nb224_atom_sim_stack.csv")
    np.save(DATA_PROCESSED / "oof_nb176_stack_nb224.npy", stk_oof)
    np.save(DATA_PROCESSED / "te_nb176_stack_nb224.npy", stk_te)


if __name__ == "__main__":
    main()
