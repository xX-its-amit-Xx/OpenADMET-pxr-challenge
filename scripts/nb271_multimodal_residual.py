"""nb271 -- Multi-modal similarity per-compound residual correction.

Pretrained: nb239 (best 4-way SLSQP, LB 0.7487).
Per test compound, identify biologically-similar train compounds via composite
similarity over multiple modalities:

1. Morgan FP Tanimoto (structural, ECFP4)
2. MACCS keys Tanimoto (sub-structural)
3. Atom-pair FP Tanimoto (topological)
4. Cross-target activity profile distance (biological — Papyrus NR predictions)
5. Physchem distance (Mahalanobis on key features)

Composite similarity = weighted average across modalities.

For each test compound:
- Find top-K neighbors by composite similarity
- Compute their RESIDUALS = y_true - nb239_pred (from train OOF)
- Apply weighted residual correction to nb239's test prediction:
  pred_corrected = nb239_test_pred + lambda * sum(sim_i * residual_i)
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Lipinski, Crippen, Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
from pathlib import Path

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def morgan_fp(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def maccs_fp(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        fps.append(MACCSkeys.GenMACCSKeys(mol) if mol else None)
    return fps


def atom_pair_fp(smiles, n_bits=2048):
    from rdkit.Chem.AtomPairs import Pairs
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol:
            fp = Pairs.GetHashedAtomPairFingerprint(mol, n_bits)
            fps.append(fp)
        else:
            fps.append(None)
    return fps


def physchem_vec(smiles):
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            feats.append([np.nan]*8); continue
        feats.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            Lipinski.NumHAcceptors(mol), Lipinski.NumHDonors(mol),
            Lipinski.NumAromaticRings(mol), Lipinski.NumRotatableBonds(mol),
            Descriptors.TPSA(mol), Descriptors.HeavyAtomCount(mol),
        ])
    return np.array(feats)


def bulk_tani(qfp, ref_fps):
    return np.array(BulkTanimotoSimilarity(qfp, ref_fps))


def main():
    print("=== nb271: Multi-modal similarity residual correction ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")

    # Train residuals
    residuals_tr = y_tr - nb239_oof
    print(f"Train residual stats: mean={residuals_tr.mean():.3f}, std={residuals_tr.std():.3f}")

    print("\nComputing multi-modal fingerprints...")
    print("  Morgan FP...")
    morgan_tr = morgan_fp(smiles_tr); morgan_te = morgan_fp(smiles_te)
    print("  MACCS...")
    maccs_tr = maccs_fp(smiles_tr); maccs_te = maccs_fp(smiles_te)
    print("  Atom pair...")
    atompair_tr = atom_pair_fp(smiles_tr); atompair_te = atom_pair_fp(smiles_te)
    print("  Physchem vectors...")
    phys_tr = physchem_vec(smiles_tr)
    phys_te = physchem_vec(smiles_te)
    # Standardize physchem
    phys_mean = np.nanmean(phys_tr, axis=0)
    phys_std = np.nanstd(phys_tr, axis=0) + 1e-6
    phys_tr_z = (phys_tr - phys_mean) / phys_std
    phys_te_z = (phys_te - phys_mean) / phys_std
    print("  Cross-target NR predictions (nb258 profile)...")
    # Use saved nb258 cross-target profiles (already computed)
    try:
        profile_tr = np.load(DATA_PROCESSED / "profile_tr_nb258.npy")
        profile_te = np.load(DATA_PROCESSED / "profile_te_nb258.npy")
        print(f"    NR profiles: tr={profile_tr.shape}, te={profile_te.shape}")
    except FileNotFoundError:
        profile_tr = np.zeros((len(smiles_tr), 7))
        profile_te = np.zeros((len(smiles_te), 7))
    profile_mean = profile_tr.mean(axis=0)
    profile_std = profile_tr.std(axis=0) + 1e-6
    profile_tr_z = (profile_tr - profile_mean) / profile_std
    profile_te_z = (profile_te - profile_mean) / profile_std

    K = 30
    print(f"\nComputing composite similarity per test compound (K={K})...")
    te_corrected = np.zeros(len(smiles_te))
    te_correction = np.zeros(len(smiles_te))
    t0 = time.time()

    for i in range(len(smiles_te)):
        if morgan_te[i] is None:
            te_corrected[i] = nb239_te[i]
            continue
        # Modality 1: Morgan
        sim_m = bulk_tani(morgan_te[i], morgan_tr)
        # Modality 2: MACCS
        sim_ma = bulk_tani(maccs_te[i], maccs_tr) if maccs_te[i] else sim_m
        # Modality 3: Atom-pair
        if atompair_te[i] is not None:
            sim_ap = bulk_tani(atompair_te[i], atompair_tr)
        else:
            sim_ap = sim_m
        # Modality 4: physchem inverse-distance
        d_phys = np.linalg.norm(phys_tr_z - phys_te_z[i:i+1], axis=1)
        sim_phys = 1 / (1 + d_phys)
        # Modality 5: NR profile inverse-distance
        d_prof = np.linalg.norm(profile_tr_z - profile_te_z[i:i+1], axis=1)
        sim_prof = 1 / (1 + d_prof)

        # Composite: weighted geometric mean (favors compounds high on multiple modalities)
        composite = (sim_m + sim_ma + sim_ap + sim_phys + sim_prof) / 5
        # Find top-K
        top_idx = np.argsort(composite)[::-1][:K]
        top_sim = composite[top_idx]
        top_res = residuals_tr[top_idx]
        # Weighted residual
        w = top_sim
        w = w / w.sum()
        weighted_res = (w * top_res).sum()
        te_correction[i] = weighted_res

        # Apply with small lambda
        te_corrected[i] = nb239_te[i] + 0.3 * weighted_res  # lambda=0.3

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(smiles_te)}  elapsed={elapsed:.0f}s")

    # Validate on train: leave-one-out residual prediction
    print("\nValidation on 300 random train compounds (LOO)...")
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(y_tr), 300, replace=False)
    val_corrections = np.zeros(len(val_idx))
    for j, i in enumerate(val_idx):
        if morgan_tr[i] is None: continue
        sim_m = bulk_tani(morgan_tr[i], morgan_tr); sim_m[i] = -1
        sim_ma = bulk_tani(maccs_tr[i], maccs_tr); sim_ma[i] = -1
        sim_ap = bulk_tani(atompair_tr[i], atompair_tr) if atompair_tr[i] else sim_m; sim_ap[i] = -1
        d_phys = np.linalg.norm(phys_tr_z - phys_tr_z[i:i+1], axis=1)
        sim_phys = 1 / (1 + d_phys); sim_phys[i] = -1
        d_prof = np.linalg.norm(profile_tr_z - profile_tr_z[i:i+1], axis=1)
        sim_prof = 1 / (1 + d_prof); sim_prof[i] = -1
        composite = (sim_m + sim_ma + sim_ap + sim_phys + sim_prof) / 5
        top_idx = np.argsort(composite)[::-1][:K]
        top_sim = composite[top_idx]
        w = top_sim / top_sim.sum()
        val_corrections[j] = (w * residuals_tr[top_idx]).sum()

    # Sweep lambdas
    print("Sweeping lambdas on validation:")
    nb239_val_rae = rae(y_tr[val_idx], nb239_oof[val_idx])
    print(f"  baseline nb239 val RAE: {nb239_val_rae:.4f}")
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        corrected = nb239_oof[val_idx] + lam * val_corrections
        r = rae(y_tr[val_idx], corrected)
        sign = " ***" if r < nb239_val_rae else ""
        print(f"  lambda={lam}: val RAE={r:.4f}{sign}")

    # Save 3 variants with different lambdas
    print("\nSaving submissions:")
    for lam in [0.1, 0.2, 0.3, 0.5]:
        te_c = nb239_te + lam * te_correction
        sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_c})
        fname = f"271_multimodal_lambda{int(lam*100):03d}.csv"
        sub.to_csv(SUBMISSIONS / fname, index=False)
        print(f"  {fname}: mean={te_c.mean():.3f}, std={te_c.std():.3f}, correction stats: mean={te_correction.mean():.3f} std={te_correction.std():.3f}")

    np.save(DATA_PROCESSED / "te_nb271_correction.npy", te_correction)
    np.save(DATA_PROCESSED / "val_nb271_correction.npy", val_corrections)


if __name__ == "__main__":
    main()
