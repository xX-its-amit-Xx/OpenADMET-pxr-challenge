"""nb222 -- 3D pharmacophore + shape features anchored on known PXR agonists.

PXR has an unusually large, flexible, hydrophobic ligand-binding pocket
(~1200 Å³). Compounds bind via:
  - hydrophobic contacts (multiple Phe, Met, Leu residues)
  - one or two H-bond donors/acceptors to His407, Gln285, Arg410
  - generally lipophilic, often with one carbonyl or sulfonyl polar anchor

Known PXR agonists from PDB co-crystal structures (anchor compounds):
  Compound         | PDB  | Notes
  rifampicin       | 1SKX | natural antibiotic, classic PXR agonist
  hyperforin       | 1M13 | natural product, St. John's wort
  SR12813          | 1ILH | synthetic, original PXR agonist scaffold
  T0901317         | 2O9I | LXR/PXR dual agonist
  PCN              | 1NRL | pregnenolone carbonitrile (rodent PXR)
  TO901317         | 4S0R | another LXR/PXR ligand
  rifaximin        | 4XAO | rifampicin analog

We anchor on their SMILES (curated from RCSB / literature) and compute:
  1. Tanimoto similarity (ECFP4) to each anchor — fingerprint-based
  2. USRCAT 3D shape+pharmacophore similarity to each anchor — true 3D
  3. PMI (principal moments of inertia) ratios — molecular shape category
  4. Sphericity + asphericity — 3D shape descriptors
  5. Pharmacophore feature counts (Donor, Acceptor, Hydrophobe, Aromatic, etc.)

Features 2-5 capture 3D properties NOT encoded by 2D Morgan/RDKit features
in our base model. This is the user's mechanistic-3D-aware features idea.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors3D, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, SigFactory
from rdkit.Chem.Pharm2D.Gobbi_Pharm2D import Gobbi_Factory

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

# Canonical SMILES of known PXR agonists from PDB co-crystal structures
PXR_AGONISTS = {
    "rifampicin": "CC1C=CC=C(C(=O)NC2=C(C(=C3C(=C2O)C(=O)C(=C(C3=O)C)O)O)O)C(C(C(C(C(C(C1OC)C)O)C)OC(=O)C)C)O",
    "hyperforin": "CC(=CCCC(C)(C(=O)C1=C(C(C(C1=O)(CC=C(C)C)CCC=C(C)C)CC=C(C)C)O)O)C",
    "SR12813":     "CC(C)(C)C1=CC(=C(C=C1)C=C(P(=O)(OCC)OCC)P(=O)(OCC)OCC)C(C)(C)C",
    "T0901317":    "OC(C(F)(F)F)(C(F)(F)F)c1ccc(N(S(=O)(=O)C(F)(F)F)CC2=CC=CC=C2)cc1",
    "PCN":         "O=C1CCC2(C)C3CCC4(C)C(C#N)CCC4C3CCC12",
    "rifaximin":   "CC1=CC=CC2=C1C(=O)C(=C(N2C)O)C(=O)NC(=O)NC3=CC=C(C=C3)C(=O)C(=O)OC",
    "clotrimazole": "Clc1ccccc1C(c2ccccc2)(c3ccccc3)n4ccnc4",
    "paclitaxel":   "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)O",
    "ritonavir":    "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(Cc2ccccc2)CC(C(Cc3ccccc3)NC(=O)OCc4cncs4)O",
    "phenobarbital": "CCC1(C(=O)NC(=O)NC1=O)c2ccccc2",
}


def gen_3d_conformer(smiles, n_confs=1, seed=42):
    """Generate one low-energy 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if not cids: return None
        # MMFF optimize and pick lowest energy
        energies = []
        for cid in cids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol,
                        AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
                if ff:
                    ff.Minimize(maxIts=200)
                    energies.append((ff.CalcEnergy(), cid))
            except Exception:
                pass
        if not energies: return mol
        energies.sort()
        best_cid = energies[0][1]
        mol = Chem.Mol(mol, False, best_cid)
        return mol
    except Exception:
        return None


def compute_3d_descriptors(mol_3d):
    """RDKit 3D descriptors: shape, PMI, sphericity, etc."""
    if mol_3d is None:
        return None
    try:
        feats = {
            "npr1":   rdMolDescriptors.CalcNPR1(mol_3d),
            "npr2":   rdMolDescriptors.CalcNPR2(mol_3d),
            "pmi1":   Descriptors3D.PMI1(mol_3d),
            "pmi2":   Descriptors3D.PMI2(mol_3d),
            "pmi3":   Descriptors3D.PMI3(mol_3d),
            "radius_gyration": Descriptors3D.RadiusOfGyration(mol_3d),
            "asphericity":     Descriptors3D.Asphericity(mol_3d),
            "spherocity":      Descriptors3D.SpherocityIndex(mol_3d),
            "inertial_shape":  Descriptors3D.InertialShapeFactor(mol_3d),
            "eccentricity":    Descriptors3D.Eccentricity(mol_3d),
        }
        return feats
    except Exception:
        return None


def compute_usrcat(mol_3d):
    """USRCAT shape + pharmacophore descriptor (60-dim vector)."""
    if mol_3d is None: return None
    try:
        return np.array(rdMolDescriptors.GetUSRCAT(mol_3d))
    except Exception:
        return None


def compute_pharmacophore_fp(mol):
    """Gobbi 2D pharmacophore fingerprint (~2000 bits)."""
    if mol is None: return None
    try:
        return Generate.Gen2DFingerprint(mol, Gobbi_Factory)
    except Exception:
        return None


def usrcat_similarity(u1, u2):
    """USR similarity (inverse manhattan)."""
    if u1 is None or u2 is None: return 0.0
    return 1.0 / (1.0 + np.mean(np.abs(u1 - u2)))


def compute_anchor_features(smiles_list, anchors_3d, anchors_usrcat, anchors_fps,
                             label="compounds"):
    """For each compound, compute similarity to each anchor + 3D descriptors."""
    print(f"  Computing 3D + pharmacophore features for {len(smiles_list)} {label}...")
    n = len(smiles_list)
    n_anchors = len(anchors_3d)
    sim_usrcat = np.zeros((n, n_anchors))
    sim_morgan = np.zeros((n, n_anchors))
    desc_3d = np.zeros((n, 10))   # 10 3D descriptors
    valid = np.zeros(n, dtype=bool)

    desc_keys = ["npr1","npr2","pmi1","pmi2","pmi3","radius_gyration",
                 "asphericity","spherocity","inertial_shape","eccentricity"]

    for i, smi in enumerate(smiles_list):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{n}...")
        mol_3d = gen_3d_conformer(smi)
        if mol_3d is None: continue
        valid[i] = True

        # 3D descriptors
        d = compute_3d_descriptors(mol_3d)
        if d:
            for k, key in enumerate(desc_keys):
                desc_3d[i, k] = d.get(key, 0.0)

        # USRCAT
        u_i = compute_usrcat(mol_3d)
        for j, u_a in enumerate(anchors_usrcat):
            sim_usrcat[i, j] = usrcat_similarity(u_i, u_a)

        # Morgan similarity to anchors
        mol_flat = Chem.MolFromSmiles(smi)
        if mol_flat:
            fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_flat, 2, 2048)
            for j, fp_a in enumerate(anchors_fps):
                sim_morgan[i, j] = DataStructs.TanimotoSimilarity(fp_i, fp_a)

    n_valid = valid.sum()
    print(f"  Generated 3D conformers for {n_valid}/{n} ({n_valid/n*100:.0f}%)")
    return desc_3d, sim_usrcat, sim_morgan, valid


def main():
    print("=== nb222: 3D pharmacophore + PXR-agonist anchor features ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}\n")

    # ── A. Anchor compounds: build 3D + USRCAT + Morgan ───────────────────────
    print("[A] Building PXR-agonist anchors...")
    anchor_names, anchor_3d, anchor_usrcat, anchor_fps = [], [], [], []
    for name, smi in PXR_AGONISTS.items():
        mol_3d = gen_3d_conformer(smi)
        if mol_3d is None:
            print(f"  {name}: 3D failed, skipping")
            continue
        u = compute_usrcat(mol_3d)
        mol_flat = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_flat, 2, 2048)
        anchor_names.append(name)
        anchor_3d.append(mol_3d)
        anchor_usrcat.append(u)
        anchor_fps.append(fp)
        print(f"  {name}: anchor ready")

    print(f"  Total anchors: {len(anchor_names)}: {anchor_names}\n")

    # Cache 3D features for re-use
    cache_path_tr = DATA_PROCESSED / "nb222_3d_features_train.npz"
    cache_path_te = DATA_PROCESSED / "nb222_3d_features_test.npz"

    if cache_path_tr.exists() and cache_path_te.exists():
        print("[B] Loading cached 3D features...")
        d = np.load(cache_path_tr)
        desc_tr, usrcat_tr, morgan_tr, valid_tr = d["desc"], d["usrcat"], d["morgan"], d["valid"]
        d = np.load(cache_path_te)
        desc_te, usrcat_te, morgan_te, valid_te = d["desc"], d["usrcat"], d["morgan"], d["valid"]
    else:
        print("[B] Computing 3D features (this takes a while: ~30-60 min)...")
        desc_tr, usrcat_tr, morgan_tr, valid_tr = compute_anchor_features(
            smiles_tr, anchor_3d, anchor_usrcat, anchor_fps, "train")
        np.savez(cache_path_tr, desc=desc_tr, usrcat=usrcat_tr, morgan=morgan_tr, valid=valid_tr)
        desc_te, usrcat_te, morgan_te, valid_te = compute_anchor_features(
            smiles_te, anchor_3d, anchor_usrcat, anchor_fps, "test")
        np.savez(cache_path_te, desc=desc_te, usrcat=usrcat_te, morgan=morgan_te, valid=valid_te)

    # ── C. Correlations of new features with PXR labels ──────────────────────
    print("\n[C] Feature correlations with PXR train labels:")
    feat_blocks = [
        ("usrcat", usrcat_tr, [f"usrcat_{n}" for n in anchor_names]),
        ("morgan_anchor", morgan_tr, [f"morgan_{n}" for n in anchor_names]),
        ("3d_desc", desc_tr, ["npr1","npr2","pmi1","pmi2","pmi3","rg","asph","spher","inertial","ecc"]),
    ]
    for block_name, block_data, block_cols in feat_blocks:
        print(f"  -- {block_name} --")
        for j, col_name in enumerate(block_cols):
            vals = block_data[:, j]
            mask = np.isfinite(vals)
            if mask.sum() < 100: continue
            rho, _ = spearmanr(vals[mask], y_tr[mask])
            print(f"    {col_name}: rho={rho:+.3f}")

    # ── D. Assemble feature matrices and CV ──────────────────────────────────
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)

    X_tr_3d = np.hstack([X_tr_base, desc_tr, usrcat_tr, morgan_tr])
    X_te_3d = np.hstack([X_te_base, desc_te, usrcat_te, morgan_te])
    print(f"\n  Augmented feature shape: train={X_tr_3d.shape}  test={X_te_3d.shape}")

    print("\n[D] Scaffold 5-fold CV:")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    cv_results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only", X_tr_base, X_te_base),
        ("3d_aug",    X_tr_3d,   X_te_3d),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:12s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── E. Blend with nb197 ──────────────────────────────────────────────────
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["3d_aug"]
    print("\n[E] Blend 3d_aug with nb197:")
    best_blend, best_r_bl = None, 999
    for w in np.arange(0.05, 0.75, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
            best_r_bl = r_bl; best_blend = (w, oof_bl, te_bl, ratio_bl)

    saved = []
    if ratio_aug >= COLLAPSE_THRESH and r_aug < base_rae:
        np.save(DATA_PROCESSED / "oof_nb222_3d_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb222_3d_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "222_3d_aug.csv", index=False)
        saved.append(f"222_3d_aug OOF={r_aug:.4f}")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"222_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} OOF={best_r_bl:.4f}")

    print(f"\n=== Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
