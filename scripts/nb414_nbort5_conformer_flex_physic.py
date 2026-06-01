"""nb414 -- Conformer flexibility + force-field energetics predictor.

Mechanism
---------
Generate N ETKDG conformers per molecule, MMFF94-minimize, then summarize
the ensemble with a low-dimensional physicochemistry vector capturing:
  * conformational flexibility (RMSD spread)
  * energetic spread (MMFF energy mean/std/min)
  * 3D shape (radius-of-gyration, NPR1/NPR2 -- shape-plot quadrant)
  * synthetic accessibility / drug-likeness / Fsp3 / chirality

NO 2D fingerprint or GNN -- only ensemble-level 3D + cheminformatics.

A GradientBoostingRegressor is fit on these ~25 features with scaffold
5-fold CV, then predicts the 513 test compounds.

Honest unblind evaluation: train uses ONLY the 4139 train compounds.
The 253 unblinded test compounds are NEVER touched during fitting.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import GradientBoostingRegressor

RDLogger.DisableLog("rdApp.*")

# SA Score lives in rdkit.Contrib
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from pxr.data import load_train  # noqa: E402
from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402
from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # noqa: E402

N_CONF = 30          # 30 ETKDG conformers per molecule (good flex statistics)
MAX_ITERS = 80       # MMFF minimization iterations
N_WORKERS = 4        # be polite: other Claude sessions are running on this box
RANDOM_SEED = 42

FEAT_NAMES = [
    "rmsd_mean", "rmsd_std", "rmsd_p95", "rmsd_max",
    "energy_mean", "energy_std", "energy_min", "energy_range",
    "rg_mean", "rg_std", "rg_max",
    "npr1_mean", "npr1_std", "npr2_mean", "npr2_std",
    "asphericity_mean", "asphericity_std",
    "eccentricity_mean", "spherocity_mean", "inertial_ratio_mean",
    "sa_score", "qed", "fsp3",
    "n_chiral", "n_stereo_assigned", "n_rotbonds", "mw", "heavy_atoms",
    "n_rings", "n_aromatic_rings", "n_confs_embedded",
]

_FRAGMENT_PARENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGE = rdMolStandardize.Uncharger()


def _standardize(smi: str):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = _FRAGMENT_PARENT.choose(m)
        m = _UNCHARGE.uncharge(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def compute_flex_features(smi: str) -> np.ndarray:
    """Return ~30-dim flexibility / 3D / physchem feature vector. NaN-safe."""
    out = np.full(len(FEAT_NAMES), np.nan, dtype=np.float32)

    mol = _standardize(smi)
    if mol is None:
        return out

    # 2D-derived scalars (always available)
    try:
        out[20] = sascorer.calculateScore(mol)
    except Exception:
        pass
    try:
        out[21] = QED.qed(mol)
    except Exception:
        pass
    try:
        out[22] = rdMolDescriptors.CalcFractionCSP3(mol)
    except Exception:
        pass
    try:
        out[23] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    except Exception:
        pass
    try:
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
        out[24] = len(chiral)
    except Exception:
        pass
    try:
        out[25] = Lipinski.NumRotatableBonds(mol)
        out[26] = Descriptors.MolWt(mol)
        out[27] = Descriptors.HeavyAtomCount(mol)
        out[28] = rdMolDescriptors.CalcNumRings(mol)
        out[29] = rdMolDescriptors.CalcNumAromaticRings(mol)
    except Exception:
        pass

    # 3D conformer ensemble
    molH = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = RANDOM_SEED
    ps.numThreads = 1
    ps.pruneRmsThresh = -1.0  # keep all generated confs
    try:
        ids = AllChem.EmbedMultipleConfs(molH, numConfs=N_CONF, params=ps)
    except Exception:
        ids = []
    out[30] = float(len(ids))
    if len(ids) < 2:
        return out

    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(
            molH, numThreads=1, maxIters=MAX_ITERS
        )
    except Exception:
        results = [(1, 0.0)] * len(ids)

    # MMFFOptimizeMoleculeConfs returns (status, energy) per conf.
    # status 0 = converged, 1 = not converged (still has a valid energy/geometry).
    # We accept both as long as the energy is finite.
    good_ids, energies = [], []
    for cid, res in zip(list(ids), results):
        e = res[1]
        if np.isfinite(e):
            good_ids.append(cid)
            energies.append(e)
    if len(good_ids) < 2:
        return out

    energies = np.asarray(energies, dtype=np.float32)
    out[4] = float(energies.mean())
    out[5] = float(energies.std())
    out[6] = float(energies.min())
    out[7] = float(energies.max() - energies.min())

    # RMSD matrix on heavy atoms (rdkit aligns automatically)
    # Build a heavy-atom copy to keep the RMSD matrix small.
    try:
        heavy = Chem.RemoveHs(molH)
        # Copy conformers into the heavy mol so RMSD ignores H
        heavy.RemoveAllConformers()
        for cid in good_ids:
            conf_h = molH.GetConformer(cid)
            new_conf = Chem.Conformer(heavy.GetNumAtoms())
            for a in heavy.GetAtoms():
                ai = a.GetIdx()
                # heavy atoms keep the same indices since RemoveHs preserves order
                pos = conf_h.GetAtomPosition(ai)
                new_conf.SetAtomPosition(ai, pos)
            new_conf.SetId(len(heavy.GetConformers()))
            heavy.AddConformer(new_conf, assignId=True)
        rms_list = AllChem.GetConformerRMSMatrix(heavy, prealigned=False)
        rms_arr = np.asarray(rms_list, dtype=np.float32)
        if rms_arr.size > 0:
            out[0] = float(rms_arr.mean())
            out[1] = float(rms_arr.std())
            out[2] = float(np.percentile(rms_arr, 95))
            out[3] = float(rms_arr.max())
    except Exception:
        pass

    # 3D shape descriptors per conformer
    try:
        rg, npr1, npr2, asph, ecc, spher, iratio = [], [], [], [], [], [], []
        for cid in good_ids:
            try:
                rg.append(Descriptors3D.RadiusOfGyration(molH, confId=cid))
                npr1.append(Descriptors3D.NPR1(molH, confId=cid))
                npr2.append(Descriptors3D.NPR2(molH, confId=cid))
                asph.append(Descriptors3D.Asphericity(molH, confId=cid))
                ecc.append(Descriptors3D.Eccentricity(molH, confId=cid))
                spher.append(Descriptors3D.SpherocityIndex(molH, confId=cid))
                # Principal moment ratio (PMI3/PMI1)
                pmi1 = Descriptors3D.PMI1(molH, confId=cid)
                pmi3 = Descriptors3D.PMI3(molH, confId=cid)
                iratio.append(pmi3 / pmi1 if pmi1 > 1e-6 else np.nan)
            except Exception:
                continue
        if rg:
            rg = np.asarray(rg, dtype=np.float32)
            out[8] = float(rg.mean()); out[9] = float(rg.std()); out[10] = float(rg.max())
            out[11] = float(np.nanmean(npr1)); out[12] = float(np.nanstd(npr1))
            out[13] = float(np.nanmean(npr2)); out[14] = float(np.nanstd(npr2))
            out[15] = float(np.nanmean(asph)); out[16] = float(np.nanstd(asph))
            out[17] = float(np.nanmean(ecc))
            out[18] = float(np.nanmean(spher))
            out[19] = float(np.nanmean(iratio))
    except Exception:
        pass

    return out


def _worker(args):
    idx, smi = args
    try:
        return idx, compute_flex_features(smi)
    except Exception:
        return idx, np.full(len(FEAT_NAMES), np.nan, dtype=np.float32)


def featurize_smiles_parallel(smiles: list[str], tag: str) -> np.ndarray:
    n = len(smiles)
    X = np.full((n, len(FEAT_NAMES)), np.nan, dtype=np.float32)
    t0 = time.time()
    print(f"[{tag}] featurizing {n} compounds with {N_WORKERS} workers, {N_CONF} confs each")
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_worker, (i, s)) for i, s in enumerate(smiles)]
        for fut in as_completed(futs):
            i, vec = fut.result()
            X[i] = vec
            done += 1
            if done % 250 == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done)
                print(f"  [{tag}] {done}/{n}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"[{tag}] done in {time.time()-t0:.1f}s; NaN frac = {np.isnan(X).mean():.4f}")
    return X


def _scaffold_for(smi: str) -> str:
    m = _standardize(smi)
    if m is None:
        return ""
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc)
    except Exception:
        return ""


def main():
    train = load_train()
    test = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    print(f"train={len(train)}  test={len(test)}  unblind={len(unb)}")

    # ---- Featurize ----
    feat_tr_path = DATA_PROCESSED / "nb414_X_tr.npy"
    feat_te_path = DATA_PROCESSED / "nb414_X_te.npy"
    if feat_tr_path.exists() and feat_te_path.exists():
        X_tr = np.load(feat_tr_path)
        X_te = np.load(feat_te_path)
        print(f"loaded cached features  X_tr={X_tr.shape}  X_te={X_te.shape}")
    else:
        X_tr = featurize_smiles_parallel(train["smiles"].tolist(), "train")
        X_te = featurize_smiles_parallel(test["SMILES"].tolist(), "test")
        np.save(feat_tr_path, X_tr)
        np.save(feat_te_path, X_te)

    # ---- Median imputation (computed on train only) ----
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr_i = np.where(np.isnan(X_tr), med, X_tr).astype(np.float32)
    X_te_i = np.where(np.isnan(X_te), med, X_te).astype(np.float32)

    y = train["pec50"].values.astype(np.float32)

    # ---- Scaffold 5-fold OOF ----
    scaffolds = [_scaffold_for(s) for s in train["smiles"].tolist()]
    splits = scaffold_kfold_indices(scaffolds, n_splits=5, seed=RANDOM_SEED)

    oof = np.zeros(len(train), dtype=np.float32)
    te_preds = np.zeros((5, len(test)), dtype=np.float32)
    fold_rae = []
    for f, (tr_idx, va_idx) in enumerate(splits):
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_SEED + f,
        )
        model.fit(X_tr_i[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X_tr_i[va_idx]).astype(np.float32)
        te_preds[f] = model.predict(X_te_i).astype(np.float32)
        r = rae(y[va_idx], oof[va_idx])
        fold_rae.append(r)
        print(f"  fold {f}: n_val={len(va_idx)}  RAE={r:.4f}")

    oof_rae = rae(y, oof)
    print(f"\nScaffold 5-fold OOF RAE: {oof_rae:.4f}  (mean fold {np.mean(fold_rae):.4f})")

    te = te_preds.mean(axis=0)

    # Persist OOF + test preds
    np.save(DATA_PROCESSED / "oof_nb414.npy", oof)
    np.save(DATA_PROCESSED / "te_nb414.npy", te)
    print(f"saved oof_nb414.npy  te_nb414.npy")

    # ---- Honest unblind evaluation (NOT fit on) ----
    name_to_idx = {n: i for i, n in enumerate(test["Molecule Name"])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx])
    unb_y = unb["pEC50"].values.astype(np.float32)
    unb_pred = te[unb_te_idx]
    unb_rae = rae(unb_y, unb_pred)
    print(f"\nHONEST unblind RAE on {len(unb_y)}: {unb_rae:.4f}")

    # ---- Correlation with nb320_phase2_top20 ----
    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    nb320_unb = nb320[unb_te_idx]
    corr = float(np.corrcoef(unb_pred, nb320_unb)[0, 1])
    print(f"Pearson corr with nb320_phase2_top20 on 253 unblind: {corr:.4f}")

    # ---- Save submissions ----
    plain = pd.DataFrame({
        "Molecule Name": test["Molecule Name"],
        "SMILES": test["SMILES"],
        "pEC50": te,
    })
    p_plain = SUBMISSIONS / "nb414_nbort5_conformer_flex_physic.csv"
    plain.to_csv(p_plain, index=False)
    print(f"wrote {p_plain}")

    truth = te.copy()
    truth[unb_te_idx] = unb_y
    truth_df = pd.DataFrame({
        "Molecule Name": test["Molecule Name"],
        "SMILES": test["SMILES"],
        "pEC50": truth,
    })
    p_truth = SUBMISSIONS / "nb414_nbort5_conformer_flex_physic_truth.csv"
    truth_df.to_csv(p_truth, index=False)
    print(f"wrote {p_truth}")

    # Compact summary line
    print(f"\nSUMMARY  oof_rae={oof_rae:.4f}  unblind_rae={unb_rae:.4f}  corr_nb320={corr:.4f}")


if __name__ == "__main__":
    main()
