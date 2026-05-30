"""nb291 -- Per-atom biotype + 3D conformer aggregate features for PXR.

User insight: standard FPs encode connectivity but not biological "atom roles"
(donor/acceptor/hydrophobe/...) or 3D shape. Adding both should help.

Pipeline:
1. Embed 1 RDKit conformer per SMILES (MMFFOptimize); fall back to 2D if 3D fails.
2. Per-atom features:
   - 9 binary biotype tags (donor, acceptor, hydrophobe, aromatic, +ionizable,
     -ionizable, halogen, metal/metalloid, chiral)
   - 8 floats (Gasteiger charge, atomic radius, vdW radius, x/y/z centered,
     dist-to-centroid, sasa proxy = # neighbors within 4 A)
3. Aggregate per-molecule:
   - per biotype: count, count/n_atoms, sum_of_charge_where_biotype=1
   - per float: mean, std, min, max
   - extras: n_atoms, n_heavy_atoms, n_aromatic_rings, radius_of_gyration, PMI1/2/3
4. Concat with pxr.featurize.combined() (Morgan + RDKit), train LGBM 5-fold CV.
5. Ablation (combined-only, +biotype, +biotype+3D), SLSQP blend vs nb239 base.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# ---------------- atomic constants ----------------
ATOMIC_RADIUS = {  # covalent radii in Angstrom
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 33: 1.21, 34: 1.20, 35: 1.20, 53: 1.39,
}
VDW_RADIUS = {
    1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80,
    16: 1.80, 17: 1.75, 33: 1.85, 34: 1.90, 35: 1.85, 53: 1.98,
}
HALOGENS = {9, 17, 35, 53}
METALS_METALLOIDS = {3, 4, 5, 11, 12, 13, 14, 19, 20, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 47, 48, 50}

BIOTYPE_NAMES = ["donor", "acceptor", "hydrophobe", "aromatic_atom",
                 "pos_ion", "neg_ion", "halogen", "metal_metalloid", "chiral"]
FLOAT_NAMES = ["charge", "atomic_r", "vdw_r", "x_c", "y_c", "z_c",
               "dist_centroid", "sasa_proxy"]


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _is_donor(atom):
    # Heuristic: N or O with at least one attached H
    z = atom.GetAtomicNum()
    if z not in (7, 8):
        return False
    return atom.GetTotalNumHs() > 0


def _is_acceptor(atom):
    # Heuristic: N or O with lone pair available (not positively charged,
    # not bonded to all positions)
    z = atom.GetAtomicNum()
    if z not in (7, 8):
        return False
    if atom.GetFormalCharge() > 0:
        return False
    return True


def _is_hydrophobe(atom):
    # Heuristic: C bonded only to C/H
    if atom.GetAtomicNum() != 6:
        return False
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() not in (1, 6):
            return False
    return True


def _is_pos_ion(atom):
    z = atom.GetAtomicNum()
    if atom.GetFormalCharge() > 0:
        return True
    # basic N (sp3 N not part of amide)
    if z == 7 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
        for nb in atom.GetNeighbors():
            for b in nb.GetBonds():
                if b.GetBondType() == Chem.rdchem.BondType.DOUBLE and \
                   b.GetOtherAtom(nb).GetAtomicNum() == 8:
                    return False
        return True
    return False


def _is_neg_ion(atom):
    if atom.GetFormalCharge() < 0:
        return True
    # carboxylate-like: O bonded to C that has =O
    if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0:
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 6:
                for b in nb.GetBonds():
                    other = b.GetOtherAtom(nb)
                    if other.GetIdx() != atom.GetIdx() and \
                       other.GetAtomicNum() == 8 and \
                       b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        return True
    return False


def embed_3d(mol):
    """Return mol with one optimized conformer, or None on failure."""
    m = Chem.AddHs(mol)
    try:
        res = AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=False)
        if res == -1:
            res = AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True)
        if res == -1:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        return m
    except Exception:
        return None


def featurize_one(smi):
    """Return 1-D feature vector or None on total failure.
    Layout:
        per-biotype block (9 * 3 = 27) +
        per-float block (8 * 4 = 32) +
        extras (8) = 67 features
    """
    mol_base = Chem.MolFromSmiles(smi)
    if mol_base is None:
        return None, True  # failed, is_3d_fail=True (use flag)

    # try 3D
    mol3d = embed_3d(mol_base)
    used_3d = mol3d is not None

    if used_3d:
        mol = mol3d
        conf = mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float64)
    else:
        # fall back: add Hs without 3D embedding; use 2D coords with z=0
        mol = Chem.AddHs(mol_base)
        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            coords = np.array(conf.GetPositions(), dtype=np.float64)
            coords[:, 2] = 0.0
        except Exception:
            coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float64)

    # Gasteiger charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    atoms = list(mol.GetAtoms())
    n_atoms = len(atoms)
    # heavy atom mask (exclude H for aggregation? user says all atoms.
    # Keep all atoms for aggregations as specified.)
    centroid = coords.mean(axis=0) if n_atoms > 0 else np.zeros(3)
    coords_c = coords - centroid
    dist_to_centroid = np.linalg.norm(coords_c, axis=1)

    # SASA proxy: count of neighbors within 4 A (excluding self)
    if used_3d and n_atoms > 1:
        # pairwise distance
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        sasa_proxy = (d2 < 16.0).sum(axis=1).astype(np.float64)
    else:
        sasa_proxy = np.zeros(n_atoms, dtype=np.float64)

    # per-atom biotype tags + float values
    bio = np.zeros((n_atoms, 9), dtype=np.float64)
    flo = np.zeros((n_atoms, 8), dtype=np.float64)
    for i, a in enumerate(atoms):
        z = a.GetAtomicNum()
        bio[i, 0] = float(_is_donor(a))
        bio[i, 1] = float(_is_acceptor(a))
        bio[i, 2] = float(_is_hydrophobe(a))
        bio[i, 3] = float(a.GetIsAromatic())
        bio[i, 4] = float(_is_pos_ion(a))
        bio[i, 5] = float(_is_neg_ion(a))
        bio[i, 6] = float(z in HALOGENS)
        bio[i, 7] = float(z in METALS_METALLOIDS)
        bio[i, 8] = float(a.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        try:
            ch = float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
            if not np.isfinite(ch):
                ch = 0.0
        except Exception:
            ch = 0.0
        flo[i, 0] = ch
        flo[i, 1] = ATOMIC_RADIUS.get(z, 1.0)
        flo[i, 2] = VDW_RADIUS.get(z, 1.7)
        flo[i, 3] = coords_c[i, 0]
        flo[i, 4] = coords_c[i, 1]
        flo[i, 5] = coords_c[i, 2]
        flo[i, 6] = dist_to_centroid[i]
        flo[i, 7] = sasa_proxy[i]

    # ---- aggregate ----
    feats = []
    # per biotype: count, count/n_atoms, sum_charge_when_bio=1
    for j in range(9):
        cnt = float(bio[:, j].sum())
        feats.append(cnt)
        feats.append(cnt / max(n_atoms, 1))
        feats.append(float((flo[:, 0] * bio[:, j]).sum()))
    # per float: mean, std, min, max
    for j in range(8):
        col = flo[:, j]
        if len(col) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.append(float(col.mean()))
            feats.append(float(col.std()))
            feats.append(float(col.min()))
            feats.append(float(col.max()))
    # extras
    feats.append(float(n_atoms))
    n_heavy = sum(1 for a in atoms if a.GetAtomicNum() != 1)
    feats.append(float(n_heavy))
    try:
        n_arom_rings = CalcNumAromaticRings(mol)
    except Exception:
        n_arom_rings = 0
    feats.append(float(n_arom_rings))
    # radius of gyration
    if n_atoms > 0:
        rg = float(np.sqrt((dist_to_centroid ** 2).mean()))
    else:
        rg = 0.0
    feats.append(rg)
    # principal moments of inertia (PMI1, PMI2, PMI3)
    if used_3d:
        try:
            feats.append(float(Descriptors3D.PMI1(mol)))
            feats.append(float(Descriptors3D.PMI2(mol)))
            feats.append(float(Descriptors3D.PMI3(mol)))
        except Exception:
            feats.extend([0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0])
    # is_3d_success flag
    feats.append(float(used_3d))

    return np.array(feats, dtype=np.float64), (not used_3d)


def featurize_biotype3d(smiles_list, label=""):
    n = len(smiles_list)
    feats = []
    failed_3d = 0
    failed_total = 0
    t0 = time.time()
    for i, s in enumerate(smiles_list):
        if s is None:
            feats.append(None)
            failed_total += 1
            failed_3d += 1
            continue
        v, is_3d_fail = featurize_one(s)
        if v is None:
            feats.append(None)
            failed_total += 1
            failed_3d += 1
        else:
            feats.append(v)
            if is_3d_fail:
                failed_3d += 1
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i+1}/{n}  elapsed={time.time()-t0:.0f}s  3d_fail={failed_3d}")
    # determine width
    width = next((len(v) for v in feats if v is not None), 1)
    out = np.full((n, width), np.nan, dtype=np.float64)
    for i, v in enumerate(feats):
        if v is not None:
            out[i] = v
    print(f"  [{label}] DONE: 3d_fail={failed_3d}/{n}  total_fail={failed_total}/{n}  width={width}")
    return out, failed_3d, failed_total


def cv_lgbm(X_tr, y_tr, X_te, folds, label="model"):
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective='mae', n_jobs=4, random_state=42, verbose=-1)
    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y_tr[ti],
               eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False),
                          lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_tr, oof)
    sp, _ = spearmanr(y_tr, oof)
    mae = mean_absolute_error(y_tr, oof)
    print(f"  [{label}] OOF RAE={r:.4f}  Spearman={sp:.4f}  MAE={mae:.4f}  te_std={te_pred.std():.3f}")
    return oof, te_pred, r, sp, mae


def main():
    print("=== nb291: Biotype + 3D atom features for PXR ===\n")
    tr = load_train()
    tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()

    te_df = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')
    te_df['std_smiles'] = te_df['SMILES'].apply(std_smi)
    smiles_te = te_df['std_smiles'].tolist()
    te_names = te_df['Molecule Name'].tolist()
    te_raw_smi = te_df['SMILES'].tolist()

    print(f"Train: {len(smiles_tr)}  Test: {len(smiles_te)}\n")

    # ---- compute biotype+3D features (cached) ----
    cache_tr = DATA_PROCESSED / "nb291_biotype3d_train.npy"
    cache_te = DATA_PROCESSED / "nb291_biotype3d_test.npy"
    cache_meta = DATA_PROCESSED / "nb291_biotype3d_meta.npy"

    if cache_tr.exists() and cache_te.exists():
        print("Loading cached biotype+3D features ...")
        B_tr = np.load(cache_tr)
        B_te = np.load(cache_te)
        meta = np.load(cache_meta, allow_pickle=True).item()
        fail_tr_3d = meta['fail_tr_3d']; fail_te_3d = meta['fail_te_3d']
        fail_tr_tot = meta['fail_tr_tot']; fail_te_tot = meta['fail_te_tot']
    else:
        print("Computing biotype+3D features for TRAIN ...")
        B_tr, fail_tr_3d, fail_tr_tot = featurize_biotype3d(smiles_tr, label="TRAIN")
        print("Computing biotype+3D features for TEST ...")
        B_te, fail_te_3d, fail_te_tot = featurize_biotype3d(smiles_te, label="TEST")
        np.save(cache_tr, B_tr); np.save(cache_te, B_te)
        np.save(cache_meta, dict(fail_tr_3d=fail_tr_3d, fail_te_3d=fail_te_3d,
                                  fail_tr_tot=fail_tr_tot, fail_te_tot=fail_te_tot),
                allow_pickle=True)

    print(f"\n3D embedding failures (used 2D fallback):")
    print(f"  TRAIN: {fail_tr_3d}/{len(smiles_tr)}  ({100*fail_tr_3d/len(smiles_tr):.1f}%)")
    print(f"  TEST:  {fail_te_3d}/{len(smiles_te)}  ({100*fail_te_3d/len(smiles_te):.1f}%)")
    print(f"Total compound failures (no features at all):")
    print(f"  TRAIN: {fail_tr_tot}  TEST: {fail_te_tot}")
    print(f"Biotype+3D feature width: {B_tr.shape[1]}")

    # Split B into biotype-only block (first 9*3=27) and 3d block (the rest)
    BIO_DIM = 9 * 3
    B_tr_bio = B_tr[:, :BIO_DIM]
    B_te_bio = B_te[:, :BIO_DIM]
    B_tr_3d = B_tr[:, BIO_DIM:]
    B_te_3d = B_te[:, BIO_DIM:]

    # ---- combined features ----
    print("\nComputing combined (Morgan + RDKit) features ...")
    C_tr = combined(smiles_tr); C_tr = impute(C_tr)
    C_te = combined(smiles_te); C_te = impute(C_te)
    print(f"  combined width: {C_tr.shape[1]}")

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)

    # ---- impute biotype+3D nans ----
    def _imp(A):
        A = A.copy()
        med = np.nanmedian(A, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        m = np.isnan(A)
        A[m] = np.take(med, np.where(m)[1])
        return A

    B_tr_full = _imp(B_tr)
    B_te_full = _imp(B_te)
    B_tr_bio_i = _imp(B_tr_bio)
    B_te_bio_i = _imp(B_te_bio)

    # ============================
    # Ablation
    # ============================
    print("\n=== Ablation ===")
    print("\n[A] combined only:")
    oof_A, te_A, rA, spA, maeA = cv_lgbm(C_tr, y_tr, C_te, folds, label="combined")

    print("\n[B] combined + biotype:")
    X_B_tr = np.hstack([C_tr, B_tr_bio_i])
    X_B_te = np.hstack([C_te, B_te_bio_i])
    oof_B, te_B, rB, spB, maeB = cv_lgbm(X_B_tr, y_tr, X_B_te, folds, label="combined+bio")

    print("\n[C] combined + biotype + 3D (full nb291):")
    X_C_tr = np.hstack([C_tr, B_tr_full])
    X_C_te = np.hstack([C_te, B_te_full])
    oof_C, te_C, rC, spC, maeC = cv_lgbm(X_C_tr, y_tr, X_C_te, folds, label="combined+bio+3d")

    # Save the full-model outputs
    np.save(DATA_PROCESSED / "oof_nb291.npy", oof_C)
    np.save(DATA_PROCESSED / "te_nb291.npy", te_C)

    # Submission CSV (nb291 alone)
    out_df = pd.DataFrame({
        "SMILES": te_raw_smi,
        "Molecule Name": te_names,
        "pEC50": te_C,
    })
    sub_path = SUBMISSIONS / "291_biotype_3d_atom_features.csv"
    out_df.to_csv(sub_path, index=False)
    print(f"\nSaved nb291 submission to {sub_path}")

    # ============================
    # SLSQP blend vs nb239 base
    # ============================
    print("\n=== SLSQP blend: nb239_full_slsqp vs nb291 ===")
    oof_239 = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    te_239 = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    print(f"  nb239 base OOF RAE: {rae(y_tr, oof_239):.4f}")
    print(f"  nb291      OOF RAE: {rae(y_tr, oof_C):.4f}")

    M = np.column_stack([oof_239, oof_C])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * 2
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(2))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    w239, w291 = best.x
    print(f"  SLSQP weights: nb239={w239:.4f}  nb291={w291:.4f}")
    print(f"  Blend OOF RAE: {best.fun:.4f}")

    # Save blend submission
    te_blend = w239 * te_239 + w291 * te_C
    out_df2 = pd.DataFrame({
        "SMILES": te_raw_smi,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    })
    sub_path2 = SUBMISSIONS / "291_nb239_nb291_slsqp_blend.csv"
    out_df2.to_csv(sub_path2, index=False)
    print(f"  Saved blend submission to {sub_path2}")

    # ============================
    # Summary
    # ============================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"3D conformer failures: TRAIN={fail_tr_3d}/{len(smiles_tr)}  TEST={fail_te_3d}/{len(smiles_te)}")
    print(f"Ablation OOF RAE:")
    print(f"  [A] combined-only         : RAE={rA:.4f}  Spearman={spA:.4f}  MAE={maeA:.4f}")
    print(f"  [B] combined+biotype      : RAE={rB:.4f}  Spearman={spB:.4f}  MAE={maeB:.4f}")
    print(f"  [C] combined+biotype+3D   : RAE={rC:.4f}  Spearman={spC:.4f}  MAE={maeC:.4f}")
    print(f"SLSQP blend (nb239 + nb291): OOF={best.fun:.4f}  w_nb291={w291:.4f}")


if __name__ == "__main__":
    main()
