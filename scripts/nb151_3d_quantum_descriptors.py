"""nb151 -- 3D quantum/topological descriptors (WHIM, MORSE, RDF, GETAWAY, AUTOCORR).

These are 3D conformer-based descriptors that capture:
  - WHIM: weighted holistic invariant molecular descriptors (3D shape + atom property weighting)
  - MORSE: 3D molecule representation of structures based on electron diffraction
  - RDF: radial distribution function on atom pairs
  - GETAWAY: geometry, topology, atom weight assembly
  - AUTOCORR3D: 3D autocorrelation (atom property × spatial separation)

Generated from a single low-energy ETKDGv3 conformer per molecule.

Unlike Morgan/RDKit 2D descriptors we already use, these are TRUE 3D features.
They are an alternative path to the orthogonal physics signal Boltz2/docking
were supposed to provide.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors3D
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def gen_3d(smi, seed=42):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) < 0: return None
    try: AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
    except Exception: pass
    return mol


def compute_3d_descriptors(mol):
    """All RDKit 3D descriptor families. Returns flat array or None."""
    if mol is None: return None
    try:
        feats = []
        # Simple 3D shape descriptors (12 values)
        feats.append(rdMolDescriptors.CalcNPR1(mol))
        feats.append(rdMolDescriptors.CalcNPR2(mol))
        feats.append(Descriptors3D.PMI1(mol))
        feats.append(Descriptors3D.PMI2(mol))
        feats.append(Descriptors3D.PMI3(mol))
        feats.append(Descriptors3D.RadiusOfGyration(mol))
        feats.append(Descriptors3D.Asphericity(mol))
        feats.append(Descriptors3D.SpherocityIndex(mol))
        feats.append(Descriptors3D.InertialShapeFactor(mol))
        feats.append(Descriptors3D.Eccentricity(mol))
        # WHIM (114 values)
        whim = rdMolDescriptors.CalcWHIM(mol)
        feats.extend(whim)
        # MORSE (224 values)
        morse = rdMolDescriptors.CalcMORSE(mol)
        feats.extend(morse)
        # RDF (210 values)
        rdf = rdMolDescriptors.CalcRDF(mol)
        feats.extend(rdf)
        # GETAWAY (273 values)
        getaway = rdMolDescriptors.CalcGETAWAY(mol)
        feats.extend(getaway)
        # AUTOCORR3D (80 values)
        autocorr = rdMolDescriptors.CalcAUTOCORR3D(mol)
        feats.extend(autocorr)
        return np.array(feats, dtype=np.float32)
    except Exception:
        return None


def main():
    print("=== nb151: 3D quantum/topological descriptors ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Test the size of feature vector on first valid compound
    test_mol = gen_3d(smiles_tr[0])
    if test_mol is None:
        for s in smiles_tr[:10]:
            test_mol = gen_3d(s)
            if test_mol is not None: break
    test_feat = compute_3d_descriptors(test_mol)
    n_3d = len(test_feat) if test_feat is not None else 0
    print(f"3D descriptor vector size: {n_3d}")

    def compute_for_set(smiles_list, label):
        print(f"\nComputing 3D for {len(smiles_list)} {label}...")
        n = len(smiles_list)
        feats = np.zeros((n, n_3d), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        t0 = time.time()
        for i, smi in enumerate(smiles_list):
            mol = gen_3d(smi)
            if mol is None: continue
            f = compute_3d_descriptors(mol)
            if f is None or len(f) != n_3d: continue
            feats[i] = f
            valid[i] = True
            if (i+1) % 300 == 0:
                print(f"  {label}: {i+1}/{n}  ({time.time()-t0:.0f}s)")
        print(f"  Valid 3D: {valid.sum()}/{n}")
        # NaN handling: replace inf with NaN, then fill NaN with column median
        feats = np.where(np.isfinite(feats), feats, np.nan)
        col_med = np.nanmedian(feats, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        feats = np.where(np.isnan(feats), np.broadcast_to(col_med, feats.shape), feats)
        return feats

    F_tr = compute_for_set(smiles_tr, "train")
    F_te = compute_for_set(smiles_te, "test")
    np.save(DATA_PROCESSED / "nb151_3d_features_train.npy", F_tr)
    np.save(DATA_PROCESSED / "nb151_3d_features_test.npy", F_te)

    # Correlation analysis of top features
    print("\nTop-15 3D features by |ρ| with PXR pEC50:")
    rho_list = []
    for j in range(F_tr.shape[1]):
        if F_tr[:, j].std() > 0:
            rho, p = spearmanr(F_tr[:, j], y_tr)
            if not np.isnan(rho):
                rho_list.append((j, rho, p))
    rho_list.sort(key=lambda x: abs(x[1]), reverse=True)
    for j, rho, p in rho_list[:15]:
        print(f"  feat[{j}]: ρ={rho:+.3f}  p={p:.2e}")

    # Augmented LGBM CV
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, F_tr])
    X_te_aug = np.hstack([X_te_base, F_te])
    print(f"\nAug shape: train={X_tr_aug.shape}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                          ("3d_quantum", X_tr_aug, X_te_aug)]:
        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:14s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "3d_quantum":
            np.save(DATA_PROCESSED / "oof_nb151_3d_quantum.npy", oof)
            np.save(DATA_PROCESSED / "te_nb151_3d_quantum.npy", te_pred)
            print("  Saved")


if __name__ == "__main__":
    main()
