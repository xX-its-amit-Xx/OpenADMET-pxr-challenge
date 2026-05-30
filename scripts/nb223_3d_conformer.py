"""nb223 -- 3D conformer features (USRCAT + RDKit 3D descriptors).

Hypothesis: 2D fingerprints miss 3D shape, which is critical for PXR's large
flexible binding pocket. Adding 3D shape descriptors could provide signal that
2D models can't see.

Approach:
1. For each compound: embed 1-2 3D conformers (RDKit ETKDG)
2. Compute USRCAT descriptors per conformer (60-dim, atom-type aware shape)
3. Compute additional 3D descriptors: PMI ratios, Asphericity, RadiusOfGyration
4. Aggregate per compound (mean over conformers)
5. Train LGBM with combined + 3D features

If RAE is still 0.5+ but correlation with nb212 is < 0.85, this provides good
diversity for an updated nb219-style blend.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors3D
from rdkit.Chem import rdShapeHelpers
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172
N_CONF = 2

t0 = time.time()


def compute_3d_features(smi):
    """Returns 1D array of 3D features (USRCAT + 3D descriptors), or None if failed."""
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # Embed N_CONF conformers
    try:
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, randomSeed=SEED,
                                          useRandomCoords=True, maxAttempts=10)
        if not ids:
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, randomSeed=SEED+1,
                                              useRandomCoords=True, maxAttempts=20)
        if not ids:
            return None
    except Exception:
        return None

    # Optimize each conformer with MMFF (quick)
    try:
        for cid in ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=50)
    except Exception:
        pass  # if MMFF fails, use unoptimized coords

    # Compute features per conformer, then average
    feats_per_conf = []
    for cid in ids:
        try:
            usrcat = rdMolDescriptors.GetUSRCAT(mol, confId=cid)
            pmi1 = Descriptors3D.PMI1(mol, confId=cid)
            pmi2 = Descriptors3D.PMI2(mol, confId=cid)
            pmi3 = Descriptors3D.PMI3(mol, confId=cid)
            asph = Descriptors3D.Asphericity(mol, confId=cid)
            rog = Descriptors3D.RadiusOfGyration(mol, confId=cid)
            ecc = Descriptors3D.Eccentricity(mol, confId=cid)
            isf = Descriptors3D.InertialShapeFactor(mol, confId=cid)
            spi = Descriptors3D.SpherocityIndex(mol, confId=cid)
            feats_per_conf.append(np.concatenate([
                np.array(usrcat, dtype=np.float64),
                np.array([pmi1, pmi2, pmi3, asph, rog, ecc, isf, spi]),
            ]))
        except Exception:
            continue

    if not feats_per_conf:
        return None
    return np.mean(np.stack(feats_per_conf), axis=0)


def main():
    print("=== nb223: 3D conformer features ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    y_tr = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Compute 3D features for all compounds
    print(f"Computing 3D features for {n_tr} train compounds (this is slow)...", flush=True)
    feat_dim = None
    feats_tr_list = []
    for i, smi in enumerate(tr_df["smiles"]):
        f = compute_3d_features(smi)
        if f is not None and feat_dim is None:
            feat_dim = len(f)
        feats_tr_list.append(f)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{n_tr} ({time.time()-t0:.0f}s)", flush=True)

    if feat_dim is None:
        print("ERROR: no 3D features computed", flush=True)
        return

    feats_tr = np.array([f if f is not None else np.full(feat_dim, np.nan)
                         for f in feats_tr_list], dtype=np.float32)
    n_failed = (np.isnan(feats_tr).all(axis=1)).sum()
    print(f"  Done. feat_dim={feat_dim}, failures={n_failed} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nComputing 3D features for {len(te_df)} test compounds...", flush=True)
    feats_te_list = []
    for i, smi in enumerate(te_df["smiles"]):
        f = compute_3d_features(smi)
        feats_te_list.append(f)
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(te_df)} ({time.time()-t0:.0f}s)", flush=True)

    feats_te = np.array([f if f is not None else np.full(feat_dim, np.nan)
                         for f in feats_te_list], dtype=np.float32)
    n_failed_te = (np.isnan(feats_te).all(axis=1)).sum()
    print(f"  Done. failures={n_failed_te} ({time.time()-t0:.0f}s)", flush=True)

    # Combine with base features
    print("\nBuilding combined feature matrix...", flush=True)
    X_tr_base = impute(feat_combined(tr_df["smiles"].tolist())).astype(np.float32)
    X_te_base = impute(feat_combined(te_df["smiles"].tolist())).astype(np.float32)

    feats_tr = impute(feats_tr)
    feats_te = impute(feats_te)

    X_tr = np.hstack([X_tr_base, feats_tr])
    X_te = np.hstack([X_te_base, feats_te])
    print(f"  shape: {X_tr.shape} (added {feat_dim} 3D features)\n", flush=True)

    # Train LGBM
    print("Training LGBM (5-fold scaffold CV)...", flush=True)
    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(len(te_df))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[va_idx], y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_pred += m.predict(X_te) / N_FOLDS
        print(f"  fold {fi+1}/{N_FOLDS}: best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb223 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb223_3d_conformer"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_pred)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_pred,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
