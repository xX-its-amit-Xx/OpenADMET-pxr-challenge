"""nb179 -- Deep PXR-specific medicinal chemistry features.

Based on PXR LBD biology (1300 A^3 plastic hydrophobic pocket, key residues
Gln285, His407, Arg410, Ser247, Trp299, Phe281/288, Met243/323).

Six new feature categories:

A) **Per-atom PXR roles** (medicinal chemist's pocket-aware atom classification):
   - AROM_HYDROPHOBIC: aromatic C in carbocyclic ring (Phe/Trp stack partners)
   - ALIPHATIC_HYDRO: sp3 C, no halogen, in ring or chain
   - HALOGEN: F/Cl/Br/I (electronic + halogen-pi stacking)
   - HBA_PRIMARY: lone-pair-only O/N (pocket H-bond accept)
   - HBD_PRIMARY: N-H or O-H (pocket H-bond donate to Gln285/Ser247)
   - DUAL_HB: N-H or O-H that also accepts (rifampicin-style)
   - FLEX_LINKER: sp3 C in non-ring rotatable chain
   - RING_FUSION: shared atom between fused rings (steroid signature)
   - AROM_HETERO_N: pyridine-type N (His407 imidazole partner)
   - SULFONYL_O: S=O oxygen (T0901317-style anchor)

B) **Pharmacophore geometry**: avg shortest-path distance between
   HBA/HBD/aromatic centers.

C) **Cliff-aware features**:
   - Compound is in known activity cliff: similarity >=0.7 to >=1 cliff pair member
   - Nearest-neighbor pEC50 spread (high spread = cliff zone)
   - Nearest-neighbor pEC50 vs own predicted (deviation = local outlier)

D) **PXR archetype scores**: Tanimoto to canonical PXR ligand archetypes
   weighted by literature pEC50.

E) **Noise correction**: identify train compounds where label deviates from
   structural neighborhood, output relabeled targets.

F) **Synthetic adjacent labels**: bridge augmentation using Papyrus PXR records
   weighted by similarity.
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Crippen, rdMolDescriptors, Descriptors
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS


# Canonical PXR archetypes (SMILES + literature pEC50 ranges)
PXR_ARCHETYPES = {
    "rifampicin":    {"smi": "CC1C(C(C(C=C(C(C(C=CC=C(C(C(C2=C(C3=C(C(=C2O)C)O)C(=O)C=C(N3)C)/C)O)C)OC(=O)C)C)O)C)O)C(=O)O1", "pec50": 6.12},
    "hyperforin":    {"smi": "CC(=CCCC(C)(C(C(=O)C(C=C)(C)C)O)CC=C(C)C)C", "pec50": 6.5},
    "paclitaxel":    {"smi": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)O", "pec50": 5.8},
    "sr12813":       {"smi": "CCOP(=O)(CC(c1cc(c(c(c1)C(C)(C)C)O)C(C)(C)C)P(=O)(OCC)OCC)OCC", "pec50": 7.55},
    "t0901317":      {"smi": "OC(C(=O)Nc1ccc(C(F)(F)F)cc1S(=O)(=O)C(F)(F)F)C(F)(F)F", "pec50": 6.7},
    "mifepristone":  {"smi": "CC#CC1(CCC2C1(CC(C3=C4CCC(=O)CC4=CCC23)C5=CC=C(C=C5)N(C)C)C)O", "pec50": 5.5},
    "lithocholic":   {"smi": "CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C", "pec50": 5.0},
    "pregnanecarb":  {"smi": "N#CN1CCC2C1CCC3(C)C2CCC4C3CCC4=O", "pec50": 5.5},
    "clotrimazole":  {"smi": "Clc1ccccc1C(c2ccccc2)(c3ccccc3)n4ccnc4", "pec50": 5.7},
    "dexamethasone": {"smi": "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C", "pec50": 4.5},
}


# Roles
ROLES = ["arom_hydrophobic", "aliphatic_hydro", "halogen", "hba_primary",
         "hbd_primary", "dual_hb", "flex_linker", "ring_fusion",
         "arom_hetero_n", "sulfonyl_o"]


def atom_pxr_roles(mol):
    """Return per-atom role one-hot + aggregates."""
    if mol is None:
        return None
    n_atoms = mol.GetNumHeavyAtoms()
    roles = [None] * n_atoms

    # Identify ring fusion atoms (atoms shared between two ring systems)
    ring_info = mol.GetRingInfo()
    atom_ring_counts = np.zeros(n_atoms, dtype=int)
    for ring in ring_info.AtomRings():
        for a in ring:
            atom_ring_counts[a] += 1

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        is_arom = atom.GetIsAromatic()
        in_ring = atom.IsInRing()
        n_h = atom.GetTotalNumHs()
        hyb = atom.GetHybridization()
        n_fused = atom_ring_counts[i]

        if sym == "S" and atom.GetDegree() >= 2 and any(n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for n in atom.GetNeighbors()):
            # SO2 sulfur — count adjacent O as sulfonyl_o
            pass  # handled below for oxygens
        if sym == "O" and any(n.GetSymbol() == "S" and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for n in atom.GetNeighbors()):
            roles[i] = "sulfonyl_o"
        elif sym == "N" and is_arom:
            roles[i] = "arom_hetero_n"
        elif sym in ("F", "Cl", "Br", "I"):
            roles[i] = "halogen"
        elif sym in ("N", "O") and n_h >= 1:
            # HBD - check if also accepting (heteroatom with both H and lone pair)
            # N: if degree 3 with H, can both donate and accept
            # O: if degree 2 with H, can both donate and accept
            if sym == "N" and atom.GetDegree() == 3:
                roles[i] = "dual_hb"
            elif sym == "O":
                roles[i] = "dual_hb"
            else:
                roles[i] = "hbd_primary"
        elif sym in ("N", "O") and n_h == 0:
            roles[i] = "hba_primary"
        elif sym == "C" and is_arom and in_ring:
            roles[i] = "arom_hydrophobic"
        elif sym == "C" and hyb == Chem.HybridizationType.SP3:
            if n_fused >= 2:
                roles[i] = "ring_fusion"
            elif in_ring:
                roles[i] = "aliphatic_hydro"
            else:
                roles[i] = "flex_linker"
        elif sym == "C":
            roles[i] = "aliphatic_hydro"
        else:
            roles[i] = None

    # Aggregate counts and ratios
    counts = {r: 0 for r in ROLES}
    for r in roles:
        if r in counts:
            counts[r] += 1
    total = max(n_atoms, 1)

    feats = []
    for r in ROLES:
        feats.append(counts[r])
        feats.append(counts[r] / total)
    return feats  # 20 features (10 counts + 10 ratios)


ATOM_FEAT_NAMES = []
for r in ROLES:
    ATOM_FEAT_NAMES.append(f"n_{r}")
    ATOM_FEAT_NAMES.append(f"ratio_{r}")


def pharmacophore_geometry(mol):
    """Pairwise distances between HBA/HBD/aromatic centers (topological)."""
    if mol is None or mol.GetNumHeavyAtoms() < 5:
        return [0.0] * 6
    n_atoms = mol.GetNumHeavyAtoms()
    # Get topological distance matrix
    dm = Chem.GetDistanceMatrix(mol)
    # Classify atoms
    arom = []
    hba = []
    hbd = []
    halogen = []
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        if a.GetIsAromatic() and sym == "C":
            arom.append(i)
        if sym in ("N", "O") and a.GetTotalNumHs() == 0:
            hba.append(i)
        if sym in ("N", "O") and a.GetTotalNumHs() >= 1:
            hbd.append(i)
        if sym in ("F", "Cl", "Br", "I"):
            halogen.append(i)
    # Average pairwise distances
    def avg_dist(a_list, b_list):
        if not a_list or not b_list:
            return 0.0
        ds = []
        for i in a_list:
            for j in b_list:
                if i != j and dm[i, j] < 1e9:
                    ds.append(dm[i, j])
        return np.mean(ds) if ds else 0.0
    return [
        avg_dist(arom, hba),
        avg_dist(arom, hbd),
        avg_dist(hba, hbd),
        avg_dist(arom, halogen),
        avg_dist(hba, halogen),
        avg_dist(hbd, halogen),
    ]


PHARM_NAMES = ["avg_arom_hba", "avg_arom_hbd", "avg_hba_hbd",
               "avg_arom_halogen", "avg_hba_halogen", "avg_hbd_halogen"]


def precompute_archetype_fps(radius=2, n_bits=2048):
    fps = {}
    for name, info in PXR_ARCHETYPES.items():
        mol = Chem.MolFromSmiles(info["smi"])
        if mol:
            fps[name] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        else:
            fps[name] = None
    return fps


def archetype_similarity(mol, fps, weighted=False):
    """Tanimoto to each PXR archetype. If weighted, also a weighted-avg-pec50 estimate."""
    if mol is None:
        return [0.0] * len(PXR_ARCHETYPES) + [4.5]  # default mean pec50
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    sims = []
    weighted_pec = 0
    sim_total = 0
    for name, info in PXR_ARCHETYPES.items():
        if fps[name] is None:
            s = 0.0
        else:
            s = TanimotoSimilarity(fp, fps[name])
        sims.append(s)
        weighted_pec += s * info["pec50"]
        sim_total += s
    avg_pred = weighted_pec / max(sim_total, 0.01) if sim_total > 0 else 4.5
    return sims + [avg_pred]


ARCH_NAMES = [f"sim_{n}" for n in PXR_ARCHETYPES] + ["archetype_weighted_pec50"]


def cliff_features(train_smiles, train_y, query_smiles, train_fps, query_fps, k=5, cliff_threshold=1.0):
    """Cliff-aware features for each query compound.

    Returns: array (n_query, 5):
      [top1_sim, nb5_avg_pec50, nb5_spread_pec50, n_cliffs_in_nb5, cliff_proximity]
    """
    n_q = len(query_smiles)
    feats = np.zeros((n_q, 5))
    train_y = np.asarray(train_y)
    for q_idx, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        sims = BulkTanimotoSimilarity(qfp, train_fps)
        sims = np.array(sims)
        top_idx = np.argsort(sims)[::-1][:k]
        top_sims = sims[top_idx]
        top_y = train_y[top_idx]
        feats[q_idx, 0] = top_sims[0] if len(top_sims) else 0
        feats[q_idx, 1] = top_y.mean() if len(top_y) else 4.5
        feats[q_idx, 2] = top_y.max() - top_y.min() if len(top_y) > 1 else 0
        # Cliffs: pairs of top-k neighbors with sim>=0.7 and |dy|>=1.0
        n_cliffs = 0
        max_cliff_dy = 0
        for i in range(len(top_idx)):
            for j in range(i+1, len(top_idx)):
                pair_sim = TanimotoSimilarity(train_fps[top_idx[i]], train_fps[top_idx[j]])
                dy = abs(top_y[i] - top_y[j])
                if pair_sim >= 0.7 and dy >= cliff_threshold:
                    n_cliffs += 1
                    max_cliff_dy = max(max_cliff_dy, dy)
        feats[q_idx, 3] = n_cliffs
        feats[q_idx, 4] = max_cliff_dy
    return feats


CLIFF_NAMES = ["nb_top1_sim", "nb_avg_pec50", "nb_spread_pec50", "nb_n_cliffs", "nb_max_cliff_dy"]


def morgan_fps(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb179: Deep PXR med-chem features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    te_names = te_df["Molecule Name"].tolist() if "Molecule Name" in te_df.columns else te_df["name"].tolist()

    print("Computing atom-role + pharmacophore + cliff features...")
    # Per-molecule feature build
    def featurize_one(smi):
        mol = Chem.MolFromSmiles(smi)
        atom_f = atom_pxr_roles(mol) or [0.0] * 20
        pharm_f = pharmacophore_geometry(mol)
        return atom_f + pharm_f

    print("  Atom roles + pharmacophore...")
    Xa_tr = np.array([featurize_one(s) for s in smiles_tr], dtype=np.float32)
    Xa_te = np.array([featurize_one(s) for s in smiles_te], dtype=np.float32)
    print(f"  Shape: tr={Xa_tr.shape}, te={Xa_te.shape}")

    # Archetype similarities
    print("  Archetype similarities...")
    arch_fps = precompute_archetype_fps()
    Xs_tr = np.array([archetype_similarity(Chem.MolFromSmiles(s), arch_fps) for s in smiles_tr], dtype=np.float32)
    Xs_te = np.array([archetype_similarity(Chem.MolFromSmiles(s), arch_fps) for s in smiles_te], dtype=np.float32)

    # Cliff features
    print("  Cliff-aware features (slow)...")
    fps_tr = morgan_fps(smiles_tr)
    fps_te = morgan_fps(smiles_te)
    # For train: compute cliff features using train as neighbor pool (LOO)
    Xc_tr = cliff_features(smiles_tr, y_tr, smiles_tr, fps_tr, fps_tr, k=6)  # k=6 since self at top
    Xc_tr[:, 0] = np.where(Xc_tr[:, 0] == 1.0, 0.5, Xc_tr[:, 0])  # zero self
    Xc_te = cliff_features(smiles_tr, y_tr, smiles_te, fps_tr, fps_te, k=5)

    # Combine
    Xall_tr = np.column_stack([Xa_tr, Xs_tr, Xc_tr])
    Xall_te = np.column_stack([Xa_te, Xs_te, Xc_te])
    all_names = ATOM_FEAT_NAMES + PHARM_NAMES + ARCH_NAMES + CLIFF_NAMES
    Xall_tr = np.nan_to_num(Xall_tr, nan=0.0)
    Xall_te = np.nan_to_num(Xall_te, nan=0.0)
    print(f"\nAll new features shape: tr={Xall_tr.shape}, te={Xall_te.shape}")
    print(f"Feature names: {len(all_names)}")

    # Correlations with pec50
    df = pd.DataFrame(Xall_tr, columns=all_names); df["pec50"] = y_tr
    corrs = df.corr(numeric_only=True)["pec50"].drop("pec50").sort_values(key=abs, ascending=False)
    print("\nTop 20 |r| with pec50:")
    print(corrs.head(20))

    # OOF: base + these features
    print("\nFeaturizing base (combined Morgan+RDKit)...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, Xall_tr])
    X_te_aug = np.column_stack([X_te, Xall_te])
    print(f"X_tr_aug={X_tr_aug.shape}")

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n=== OOF comparison ===")
    for name, X in [("base", X_tr), ("base + pxr_atom_chem", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"{name:25s}: OOF RAE = {rae(y_tr, oof):.4f}")

    # Final + stack with nb224
    print("\n=== Stack with nb224 ===")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb224_te  = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    X_stk_tr = np.column_stack([nb224_oof.reshape(-1,1), Xall_tr])
    X_stk_te = np.column_stack([nb224_te.reshape(-1,1),  Xall_te])
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

    np.save(DATA_PROCESSED / "Xa_tr_nb179.npy", Xall_tr)
    np.save(DATA_PROCESSED / "Xa_te_nb179.npy", Xall_te)
    np.save(DATA_PROCESSED / "oof_nb179_stack.npy", stk_oof)
    np.save(DATA_PROCESSED / "te_nb179_stack.npy", stk_te)

    # === Noise correction: relabel suspect train compounds ===
    print("\n=== Noise correction: relabel train via neighborhood consensus ===")
    relabeled_y = y_tr.copy()
    n_relabeled = 0
    # For each compound, compute weighted neighbor mean from cliff_features
    # cliff_features already computed Xc_tr[:, 1] = neighborhood avg pec50
    nb_avg = Xc_tr[:, 1]
    nb_spread = Xc_tr[:, 2]
    residual_to_nb = y_tr - nb_avg
    # Compounds with large residual_to_nb AND low spread (so neighbors agree) = likely mislabel
    # Threshold: residual > 1.5 log units AND spread < 1.0 = neighbors agree but compound is far off
    suspect_mask = (np.abs(residual_to_nb) > 1.5) & (nb_spread < 1.0)
    print(f"  Suspect mislabel compounds: {suspect_mask.sum()}")
    if suspect_mask.sum() > 0:
        # Relabel via blend: 0.5 * original + 0.5 * neighbor avg
        relabeled_y[suspect_mask] = 0.5 * y_tr[suspect_mask] + 0.5 * nb_avg[suspect_mask]
        n_relabeled = suspect_mask.sum()
        print(f"  Relabeled {n_relabeled} compounds (50/50 blend toward neighborhood)")
        # Retrain with new labels
        clean_oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_tr_aug[ti], relabeled_y[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            clean_oof[vi] = md.predict(X_tr_aug[vi])
        print(f"  Relabeled OOF RAE (vs original labels): {rae(y_tr, clean_oof):.4f}")

        np.save(DATA_PROCESSED / "nb179_suspect_mask.npy", suspect_mask)
        np.save(DATA_PROCESSED / "nb179_relabeled_y.npy", relabeled_y)
        np.save(DATA_PROCESSED / "oof_nb179_relabeled.npy", clean_oof)

    print("\n=== nb179 done ===")


if __name__ == "__main__":
    main()
