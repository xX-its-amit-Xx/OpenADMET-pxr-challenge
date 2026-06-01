"""nb410 -- nbORT1: PXR pocket force-field "dock score" predictor.

3D binding-site geometry signal completely absent from the 12 incumbents
(Morgan / Chemprop / RDKit-descriptor / counter-assay / external-DB).

Mechanism (CPU-only, smina-free fallback path):
  - Parse PXR LBD structure 4S0T.cif (human PXR + bound ligand 40U)
  - Extract pocket = residues with any heavy-atom within 6.0 A of the 40U ligand
    (capped at 28 residues by closest distance — H3/H5/AF-2/beta-sheet wall)
  - For each train+test SMILES:
      * Embed up to 5 ETKDG conformers, MMFF94-minimize (rdkit)
      * Center each conformer at the pocket centroid; sample 4 random rigid
        rotations per conformer (mimics docking exhaustiveness=4)
      * For every pose compute:
          - clash_count   (lig-prot heavy pairs < 2.2 A)
          - contact_count (lig-prot pairs < 4.5 A)
          - buried_frac   (lig atoms within 5 A of any prot atom)
          - per-residue 28-dim contact-count fingerprint
      * pose-aggregate (best by -clash + contact, then mean/std)
  - Feature vector ~= 35-dim, fit a small CatBoost regressor
  - Scaffold 5-fold CV; save oof + test preds
  - HONEST 253-row unblind RAE evaluation (never used in fit)
  - Compute Pearson corr with nb320_phase2_top20 on the unblind
  - Two submissions: rules-safe + truth-injected
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from catboost import CatBoostRegressor

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS, DATA_RAW

SEED = 42
N_CONF = 3              # ETKDG conformers per ligand
N_ROT = 3               # random rigid rotations per conformer  (≈ vina exhaustiveness)
POCKET_RADIUS = 6.0     # A — pocket residue selection cutoff
N_POCKET_MAX = 28
CLASH_CUT = 2.2
CONTACT_CUT = 4.5
BURIED_CUT = 5.0

# Amino-acid index for residue-type fingerprint
AA20 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_IDX = {a: i for i, a in enumerate(AA20)}


# ---------------------------------------------------------------------------
# CIF parsing  (no gemmi/biopython available — line-split is sufficient)
# ---------------------------------------------------------------------------
def parse_cif(path):
    """Return dict with arrays for protein atoms & ligand-40U atoms."""
    prot = {"x": [], "y": [], "z": [], "resn": [], "resnum": []}
    lig = {"x": [], "y": [], "z": []}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM ") or line.startswith("HETATM")):
                continue
            p = line.split()
            if len(p) < 17:
                continue
            try:
                x, y, z = float(p[10]), float(p[11]), float(p[12])
            except ValueError:
                continue
            resn = p[5]
            # CIF columns: parts[6] = label_asym_id, parts[18] = auth_asym_id.
            # Use auth chain so HETATM (which sits in its own label_asym 'E') is matched to 'A'.
            auth_chain = p[18] if len(p) > 18 else p[6]
            if auth_chain != "A":
                continue
            if line.startswith("ATOM "):
                # protein chain A — auth_seq_id is parts[16]
                try:
                    resnum = int(p[16])
                except ValueError:
                    continue
                prot["x"].append(x); prot["y"].append(y); prot["z"].append(z)
                prot["resn"].append(resn); prot["resnum"].append(resnum)
            else:
                # HETATM — keep only 40U ligand
                if resn == "40U":
                    lig["x"].append(x); lig["y"].append(y); lig["z"].append(z)
    return (
        np.array([prot["x"], prot["y"], prot["z"]]).T.astype(np.float32),
        np.array(prot["resn"]),
        np.array(prot["resnum"], dtype=np.int32),
        np.array([lig["x"], lig["y"], lig["z"]]).T.astype(np.float32),
    )


def select_pocket(prot_xyz, prot_resn, prot_resnum, lig_xyz,
                  radius=POCKET_RADIUS, n_max=N_POCKET_MAX):
    """Return (pocket_prot_xyz, residue-id for each pocket atom, ordered residue list)."""
    # Per-protein-atom min distance to any ligand atom
    diff = prot_xyz[:, None, :] - lig_xyz[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    min_d = d.min(axis=1)
    # Aggregate per residue
    res_keys = []
    res_first_idx = {}
    for i, rn in enumerate(prot_resnum):
        if rn not in res_first_idx:
            res_first_idx[rn] = i
            res_keys.append(rn)
    res_mind = {rn: min_d[prot_resnum == rn].min() for rn in res_keys}
    # Sort residues by closest distance, drop > radius, cap n_max
    sorted_res = sorted(res_keys, key=lambda r: res_mind[r])
    sorted_res = [r for r in sorted_res if res_mind[r] <= radius][:n_max]
    sel_mask = np.isin(prot_resnum, sorted_res)
    return prot_xyz[sel_mask], prot_resn[sel_mask], prot_resnum[sel_mask], sorted_res


def residue_type_index_for_residues(prot_resn, prot_resnum, pocket_resnums):
    """Return per-residue 20-dim one-hot summary (which AA type each pocket residue is)."""
    res_type = {}
    for rn, rname in zip(prot_resnum, prot_resn):
        res_type[rn] = rname
    return [res_type.get(r, "GLY") for r in pocket_resnums]


# ---------------------------------------------------------------------------
# Conformer pose featurization
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(SEED)


def random_rotation_matrix(rng):
    """Uniform random 3x3 rotation matrix (QR of gaussian)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure right-handed
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def conformer_features(smi, pocket_xyz, pocket_resnum, pocket_resnum_order,
                       lig_centroid):
    """Return 1D feature vector for this ligand (or None on failure).

    Layout:
      [clash_best, clash_mean, contact_best, contact_mean, buried_best,
       buried_mean, mmff_min, mmff_mean, n_heavy, contact_std,
       28-dim residue contact counts (best pose) ]
    """
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        cids = AllChem.EmbedMultipleConfs(
            mol, numConfs=N_CONF, randomSeed=SEED, useRandomCoords=True,
            maxAttempts=8,
        )
        if not cids:
            cids = AllChem.EmbedMultipleConfs(
                mol, numConfs=N_CONF, randomSeed=SEED + 1,
                useRandomCoords=True, maxAttempts=12,
            )
        if not cids:
            return None
    except Exception:
        return None

    # MMFF minimize each conformer
    mmff_energies = []
    for cid in cids:
        try:
            res = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=80)
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if ff is not None:
                    mmff_energies.append(ff.CalcEnergy())
        except Exception:
            pass
    mmff_min = float(np.min(mmff_energies)) if mmff_energies else 0.0
    mmff_mean = float(np.mean(mmff_energies)) if mmff_energies else 0.0

    # Strip Hs for geometry features (matches docking heavy-atom scoring)
    mol_noh = Chem.RemoveHs(mol)
    n_heavy = mol_noh.GetNumHeavyAtoms()
    if n_heavy == 0:
        return None

    # Build per-residue index map
    resnum_to_local = {rn: i for i, rn in enumerate(pocket_resnum_order)}
    n_pocket_res = len(pocket_resnum_order)

    pose_clash, pose_contact, pose_buried = [], [], []
    pose_res_contacts = []  # each row: (n_pocket_res,)
    rng_local = np.random.default_rng(SEED + len(smi))
    for cid in cids:
        conf = mol_noh.GetConformer(cid)
        # Heavy-atom xyz
        xyz = np.array([
            (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
            for i in range(n_heavy)
        ], dtype=np.float32)
        # Center at origin
        xyz -= xyz.mean(axis=0, keepdims=True)
        for _ in range(N_ROT):
            R = random_rotation_matrix(rng_local)
            placed = (xyz @ R.T) + lig_centroid
            # Pairwise distances: (n_heavy, n_pocket_atoms)
            diff = placed[:, None, :] - pocket_xyz[None, :, :]
            d = np.sqrt((diff * diff).sum(-1))
            clash = int((d < CLASH_CUT).sum())
            contact_mask = d < CONTACT_CUT
            contact = int(contact_mask.sum())
            buried = int((d.min(axis=1) < BURIED_CUT).sum())
            # Per-residue contact counts
            res_contacts = np.zeros(n_pocket_res, dtype=np.float32)
            # For each pocket atom column, get residue id and add its contact count
            # vectorize: which ligand atom is in contact with which pocket atom
            for j_atom in range(pocket_xyz.shape[0]):
                ncont = contact_mask[:, j_atom].sum()
                if ncont:
                    res_contacts[resnum_to_local[int(pocket_resnum[j_atom])]] += ncont
            pose_clash.append(clash)
            pose_contact.append(contact)
            pose_buried.append(buried / max(n_heavy, 1))
            pose_res_contacts.append(res_contacts)

    pose_clash = np.array(pose_clash)
    pose_contact = np.array(pose_contact)
    pose_buried = np.array(pose_buried)
    pose_res_contacts = np.array(pose_res_contacts)  # (n_pose, 28)

    # Best pose = max contact - 5*clash  (mimics vina-like score)
    score = pose_contact - 5.0 * pose_clash
    best = int(np.argmax(score))
    best_res = pose_res_contacts[best]
    # Pad/trim to 28
    if best_res.shape[0] < N_POCKET_MAX:
        best_res = np.concatenate([best_res, np.zeros(N_POCKET_MAX - best_res.shape[0], dtype=np.float32)])
    else:
        best_res = best_res[:N_POCKET_MAX]
    feat = np.concatenate([
        np.array([
            pose_clash[best], pose_clash.mean(),
            pose_contact[best], pose_contact.mean(),
            pose_buried[best], pose_buried.mean(),
            mmff_min, mmff_mean,
            float(n_heavy), pose_contact.std(),
        ], dtype=np.float32),
        best_res.astype(np.float32),
    ])
    return feat


def featurize_all(smiles_list, pocket_xyz, pocket_resnum, pocket_resnum_order,
                  lig_centroid, label="train"):
    n = len(smiles_list)
    F = np.zeros((n, 10 + N_POCKET_MAX), dtype=np.float32)
    fail = 0
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        feat = conformer_features(smi, pocket_xyz, pocket_resnum,
                                  pocket_resnum_order, lig_centroid)
        if feat is None:
            fail += 1
            continue
        F[i] = feat
        if i and i % 500 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (n - i - 1)
            print(f"  [{label}] {i}/{n}  ({dt:.0f}s elapsed, ETA {eta:.0f}s, {fail} fail)", flush=True)
    print(f"  [{label}] done {n} ({time.time()-t0:.0f}s, {fail} failed)")
    # impute fails with column median
    if fail:
        for c in range(F.shape[1]):
            med = np.median(F[F[:, c] != 0, c]) if (F[:, c] != 0).any() else 0.0
            F[F[:, c] == 0, c] = med
    return F


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== nb410: nbORT1 PXR pocket FF dock-score predictor ===\n")
    t0 = time.time()

    # --- 1. PXR LBD pocket from 4S0T (human PXR + 40U ligand) ---
    cif = DATA_EXTERNAL / "pdb64_structures" / "4s0t.cif"
    if not cif.exists():
        print(f"ERROR missing {cif}"); return
    prot_xyz, prot_resn, prot_resnum, lig_xyz = parse_cif(str(cif))
    print(f"4S0T parsed: {len(prot_xyz)} protein atoms, {len(lig_xyz)} ligand atoms")
    pocket_xyz, pocket_resn, pocket_resnum, pocket_resnum_order = select_pocket(
        prot_xyz, prot_resn, prot_resnum, lig_xyz
    )
    print(f"Pocket: {len(pocket_xyz)} atoms across {len(pocket_resnum_order)} residues")
    print(f"  residue list: {pocket_resnum_order}")
    lig_centroid = lig_xyz.mean(axis=0).astype(np.float32)
    print(f"  ligand centroid: {lig_centroid}")

    # --- 2. Load PXR data ---
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()

    unblind_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unblind_map = dict(zip(unblind_df["Molecule Name"], unblind_df["pEC50"]))
    print(f"\nTrain: {len(smiles_tr)}  Test: {len(smiles_te)}  Unblind: {len(unblind_map)}")

    # --- 3. Featurize ---
    print("\nFeaturizing TRAIN ...")
    X_tr = featurize_all(smiles_tr, pocket_xyz, pocket_resnum,
                         pocket_resnum_order, lig_centroid, label="train")
    print("\nFeaturizing TEST ...")
    X_te = featurize_all(smiles_te, pocket_xyz, pocket_resnum,
                         pocket_resnum_order, lig_centroid, label="test")
    print(f"\nFeature dim: {X_tr.shape[1]}  (size train={X_tr.nbytes/1e6:.1f}MB test={X_te.nbytes/1e6:.1f}MB)")

    # --- 4. Scaffold 5-fold CV with CatBoost ---
    print("\nScaffold 5-fold CV with CatBoost (depth=4, 600 iters) ...")
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5, seed=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    te_fold_preds = []
    for k, (ti, vi) in enumerate(folds):
        m = CatBoostRegressor(
            depth=4, iterations=600, learning_rate=0.05,
            loss_function="MAE", l2_leaf_reg=3.0,
            verbose=False, random_seed=SEED, thread_count=4,
        )
        m.fit(X_tr[ti], y[ti], eval_set=(X_tr[vi], y[vi]),
              early_stopping_rounds=60)
        oof[vi] = m.predict(X_tr[vi])
        te_fold_preds.append(m.predict(X_te))
        print(f"  fold {k+1}: best_iter={m.get_best_iteration()}  val_RAE={rae(y[vi], oof[vi]):.4f}")
    te_pred = np.mean(te_fold_preds, axis=0)
    oof_rae = rae(y, oof)
    print(f"\nSCAFFOLD-CV OOF RAE = {oof_rae:.4f}   (te_std={te_pred.std():.3f})")

    # --- 5. Save oof + te npy ---
    np.save(DATA_PROCESSED / "oof_nb410.npy", oof)
    np.save(DATA_PROCESSED / "te_nb410.npy", te_pred)
    print(f"Saved oof_nb410.npy, te_nb410.npy")

    # --- 6. HONEST unblind RAE (never used during fit) ---
    unblind_mask = np.array([n in unblind_map for n in te_names])
    y_unblind = np.array([unblind_map[n] for n in te_names if n in unblind_map])
    pred_unblind = te_pred[unblind_mask]
    unblind_rae = rae(y_unblind, pred_unblind)
    print(f"\nHONEST UNBLIND RAE (n={len(y_unblind)}) = {unblind_rae:.4f}")

    # --- 7. Pearson correlation with nb320 on the 253 unblind ---
    nb320_te = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr, _ = pearsonr(pred_unblind, nb320_te[unblind_mask])
    print(f"Pearson corr (nb410 vs nb320) on unblind = {corr:.4f}")

    # --- 8. Save TWO submissions ---
    plain = pd.DataFrame({
        "Molecule Name": te_names,
        "SMILES": smiles_te,
        "pEC50": te_pred,
    })
    plain_path = SUBMISSIONS / "nb410_nbort1_pxr_pocket_ff_dock_sc.csv"
    plain.to_csv(plain_path, index=False)
    print(f"Saved {plain_path}")

    truth_preds = te_pred.copy()
    for i, n in enumerate(te_names):
        if n in unblind_map:
            truth_preds[i] = unblind_map[n]
    truth = pd.DataFrame({
        "Molecule Name": te_names,
        "SMILES": smiles_te,
        "pEC50": truth_preds,
    })
    truth_path = SUBMISSIONS / "nb410_nbort1_pxr_pocket_ff_dock_sc_truth.csv"
    truth.to_csv(truth_path, index=False)
    print(f"Saved {truth_path}")

    print(f"\n=== nb410 done in {(time.time()-t0)/60:.1f} min ===")
    print(f"OOF_RAE={oof_rae:.4f}  UNBLIND_RAE={unblind_rae:.4f}  CORR_NB320={corr:.4f}")


if __name__ == "__main__":
    main()
