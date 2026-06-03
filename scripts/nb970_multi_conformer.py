"""nb970 -- Multi-conformer Boltzmann-weighted 3D shape features.

Pipeline:
  1) For each train+test compound, embed 5 conformers via EmbedMultipleConfs
     (randomSeed sweep). MMFF94 optimize + energy per conformer.
  2) Per-conformer 3D features (5 dims):
       radius_of_gyration, asphericity, polar_surface_area_3d (TPSA 3D-weighted),
       mean_atomic_volume, max_pairwise_distance.
  3) Boltzmann weights w_i = exp(-(E_i - E_min)/RT) / Z   (T = 298 K).
  4) Boltzmann-averaged 5-dim 3D vector per compound.
  5) Concat with Morgan(2048)+RDKit(217) -> LGBM Huber regression.
  6) in_RAE on 253 unblind. Save te + CSV.

Artifacts: C:/pxr_artifacts/nb970/
Wall-time budget: < 15 min.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit import RDLogger

import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb970"
SEED = 42
N_CONF = 5
RT = 0.5924  # kcal/mol at T=298.15 K (R = 1.9872e-3 kcal/mol/K)
F3D_DIM = 5
F3D_NAMES = ["rgyr", "asphericity", "tpsa3d", "mean_atom_vol", "max_pair_dist"]

ART = Path("C:/pxr_artifacts/nb970")
ART.mkdir(parents=True, exist_ok=True)


def _conf_features(mol_h, conf_id):
    """Return (energy_kcal, feat5) for the given conformer id.

    feat5 = [rgyr, asphericity, tpsa3d_weighted, mean_atom_vol, max_pair_dist]
    """
    # MMFF energy (kcal/mol). Returns None if MMFF properties fail.
    energy = float("inf")
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=conf_id)
            if ff is not None:
                energy = float(ff.CalcEnergy())
    except Exception:
        energy = float("inf")

    conf = mol_h.GetConformer(conf_id)
    n = mol_h.GetNumAtoms()
    coords = np.array(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(n)],
        dtype=np.float64,
    )
    # radius of gyration
    try:
        rgyr = float(Descriptors3D.RadiusOfGyration(mol_h, confId=conf_id))
    except Exception:
        rgyr = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean()))
    try:
        aspher = float(Descriptors3D.Asphericity(mol_h, confId=conf_id))
    except Exception:
        aspher = 0.0
    # 3D-weighted TPSA: per-atom TPSA contribution scaled by inverse radial
    # buriedness (atoms near surface get full weight). Approximated as
    # (sum |coord - centroid|) * (atomic-contrib polar flag).
    centroid = coords.mean(0)
    radial = np.linalg.norm(coords - centroid, axis=1)
    polar_mask = np.array(
        [atom.GetAtomicNum() in (7, 8) for atom in mol_h.GetAtoms()],
        dtype=np.float64,
    )
    tpsa3d = float((radial * polar_mask).sum())
    # mean atomic volume via covalent radii cubes (proxy)
    rcov = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07,
            16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
    vols = np.array(
        [(4.0 / 3.0) * np.pi * (rcov.get(a.GetAtomicNum(), 0.75) ** 3)
         for a in mol_h.GetAtoms()],
        dtype=np.float64,
    )
    mean_vol = float(vols.mean())
    # max pairwise distance
    if n > 1:
        diff = coords[:, None, :] - coords[None, :, :]
        max_d = float(np.sqrt((diff * diff).sum(-1)).max())
    else:
        max_d = 0.0
    return energy, np.array([rgyr, aspher, tpsa3d, mean_vol, max_d],
                            dtype=np.float64)


def featurize_3d(smi):
    """Boltzmann-averaged 5-dim 3D feature vector for one SMILES.

    Returns (feat5, used_n_conformers).
    """
    fail = np.zeros(F3D_DIM, dtype=np.float32), 0
    if not isinstance(smi, str) or not smi:
        return fail
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return fail
    mol_h = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        params.numThreads = 1
        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=N_CONF, params=params)
        cids = list(cids)
    except Exception:
        cids = []
    if len(cids) == 0:
        return fail
    # MMFF-optimize each conformer (best effort)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=200, mmffVariant="MMFF94")
    except Exception:
        pass

    energies = []
    feats = []
    for cid in cids:
        e, f = _conf_features(mol_h, cid)
        if np.isfinite(e):
            energies.append(e)
            feats.append(f)
    if not energies:
        # fall back to plain mean of available features (no Boltzmann)
        if feats:
            return np.mean(feats, axis=0).astype(np.float32), len(feats)
        return fail
    energies = np.array(energies, dtype=np.float64)
    feats = np.stack(feats, axis=0)  # (k, 5)
    w = np.exp(-(energies - energies.min()) / RT)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        avg = feats.mean(axis=0)
    else:
        avg = (w[:, None] * feats).sum(axis=0) / w_sum
    return avg.astype(np.float32), len(energies)


def featurize_all_3d(smiles_list, tag):
    cache = ART / f"feat3d_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"  [{tag}] loaded cached 3D feats {z['feats'].shape}")
        return z["feats"], z["n_used"]
    n = len(smiles_list)
    feats = np.zeros((n, F3D_DIM), dtype=np.float32)
    n_used = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        f, k = featurize_3d(smi)
        feats[i] = f
        n_used[i] = k
        if (i + 1) % 250 == 0:
            print(f"    [{tag}] {i+1}/{n}  t={time.time()-t0:.1f}s  "
                  f"mean_used={n_used[:i+1].mean():.2f}")
    np.savez_compressed(cache, feats=feats, n_used=n_used)
    print(f"  [{tag}] cached -> {cache.name}  fail={(n_used==0).sum()}")
    return feats, n_used


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-conformer 3D shape features + LGBM Huber")
    print("=" * 78)

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"train={len(tr)}  test={len(te)}")

    print("Embedding 3D conformers (train)...")
    f3_tr, used_tr = featurize_all_3d(tr["smiles"].tolist(), "train")
    print("Embedding 3D conformers (test)...")
    f3_te, used_te = featurize_all_3d(te["smiles"].tolist(), "test")
    print(f"  3D feats: train={f3_tr.shape} fail={(used_tr==0).sum()}  "
          f"test={f3_te.shape} fail={(used_te==0).sum()}")
    for j, name in enumerate(F3D_NAMES):
        print(f"    {name:18s}  tr mean={f3_tr[:,j].mean():.3f} "
              f"std={f3_tr[:,j].std():.3f}  "
              f"te mean={f3_te[:,j].mean():.3f} std={f3_te[:,j].std():.3f}")

    print("Featurizing 2D (combined Morgan + RDKit)...")
    X_tr_2d = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_te_2d = impute(combined(te["smiles"].tolist())).astype(np.float32)
    print(f"  X_tr_2d={X_tr_2d.shape}  X_te_2d={X_te_2d.shape}")

    X_tr = np.concatenate([X_tr_2d, f3_tr], axis=1)
    X_te = np.concatenate([X_te_2d, f3_te], axis=1)
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"mem={X_tr.nbytes/1e6:.1f}MB")

    print("Training LGBM (Huber objective, alpha=0.9)...")
    params = dict(
        objective="huber",
        alpha=0.9,
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=500,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te).astype(np.float64)
    print(f"  pred mean={preds.mean():.3f}  std={preds.std():.3f}  "
          f"min={preds.min():.3f}  max={preds.max():.3f}")

    # Importance check
    try:
        imp = model.booster_.feature_importance(importance_type="gain")
        d2 = X_tr_2d.shape[1]
        share_3d = imp[d2:].sum() / max(imp.sum(), 1.0)
        print(f"  3D-feature importance share = {share_3d:.4f}")
    except Exception:
        share_3d = float("nan")

    # In-RAE on 253 unblind
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te["name"])}
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values
    in_rae = float(rae(unb_y, preds[unb_idx]))

    print("=" * 78)
    print(f"UNBLIND (n={len(unb_idx)}): in_RAE = {in_rae:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {preds[unb_idx].mean():.3f} / "
          f"{preds[unb_idx].std():.3f}")
    print("=" * 78)

    # Save
    np.save(ART / "te_pred.npy", preds)
    np.save(ART / "f3_tr.npy", f3_tr)
    np.save(ART / "f3_te.npy", f3_te)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", preds)

    sub = SUBMISSIONS / f"{TAG}_multi_conformer.csv"
    pd.DataFrame({
        "Molecule Name": te["name"].values,
        "SMILES":        te["smiles"].values,
        "pEC50":         preds,
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    metrics = {
        "in_rae": in_rae,
        "n_unblind": int(len(unb_idx)),
        "fail_train": int((used_tr == 0).sum()),
        "fail_test": int((used_te == 0).sum()),
        "mean_conf_used_train": float(used_tr.mean()),
        "mean_conf_used_test": float(used_te.mean()),
        "share_3d_importance": float(share_3d) if np.isfinite(share_3d) else None,
        "wall_time_s": float(time.time() - t0),
    }
    pd.Series(metrics).to_json(ART / "metrics.json")
    print(f"Done in {time.time()-t0:.1f}s")
    print("RESULT:", metrics)
    return metrics


if __name__ == "__main__":
    main()
