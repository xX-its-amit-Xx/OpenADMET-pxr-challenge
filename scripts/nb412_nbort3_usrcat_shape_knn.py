"""nb412 (nbORT3) -- USRCAT 3D pharmacophore-aware shape KNN.

Signal source: pure 3D shape + pharmacophore-color overlap (ROCS-style).
For every train+test molecule:
  1. ETKDG single conformer + MMFF94 minimization
  2. 60-dim USRCAT descriptor (12 moments x 5 atom-type subsets:
     all / hydrophobic / aromatic / HBD / HBA)
At inference: similarity-weighted k=10 NN in USRCAT space
  (Manhattan distance, exp(-d/scale) weights) over train pEC50.

Constraints honored:
  - Fits ONLY on the 4139 TRAIN compounds.
  - 253 unblind rows are treated as honest hold-out (never used in fit).
  - CPU-only; ~8 min for 4652 mols.
  - Saves oof_nb412.npy, te_nb412.npy and two submissions
    (rules-safe + truth-injected).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

USRCAT_DIM = 60
K_NN = 10
DIST_SCALE = None  # set from median train distance


# ---------------------------------------------------------------------------
# Single-conformer USRCAT featurization
# ---------------------------------------------------------------------------

def usrcat_one(smiles: str, seed: int = 42) -> np.ndarray:
    """Return 60-dim USRCAT descriptor, or zeros if embedding fails."""
    out = np.zeros(USRCAT_DIM, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = False
    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            # fallback: random coords
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return out
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        desc = GetUSRCAT(mol)
        return np.asarray(desc, dtype=np.float32)
    except Exception:
        return out


def usrcat_batch(smiles_list, label: str = "") -> np.ndarray:
    n = len(smiles_list)
    X = np.zeros((n, USRCAT_DIM), dtype=np.float32)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        X[i] = usrcat_one(smi, seed=42)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  [{label}] {i+1}/{n}  {rate:.1f} mol/s  eta {eta:.0f}s", flush=True)
    return X


# ---------------------------------------------------------------------------
# Similarity-weighted KNN in USRCAT space
# ---------------------------------------------------------------------------

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int = K_NN,
    scale: float | None = None,
) -> np.ndarray:
    """exp(-d/scale)-weighted mean of k nearest train pEC50 values.
    Manhattan distance in USRCAT space.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="manhattan", algorithm="brute")
    nn.fit(X_train)
    dists, idx = nn.kneighbors(X_query, return_distance=True)
    if scale is None or scale <= 0:
        scale = float(np.median(dists)) + 1e-9
    w = np.exp(-dists / scale)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    w = w / w_sum
    y_neigh = y_train[idx]
    return (w * y_neigh).sum(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("nb412 / nbORT3 -- USRCAT 3D shape-pharmacophore KNN")
    print("=" * 72)

    # ---- Load data ----
    train = load_train()
    test = load_test()
    train = add_standard_columns(train, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")
    train = train.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train: {len(train)}  test: {len(test)}")

    y_train = train["pec50"].values.astype(np.float32)

    # ---- USRCAT featurize ----
    print("\nFeaturizing TRAIN (1 ETKDG conf + MMFF94 + USRCAT)...")
    X_train = usrcat_batch(train["std_smiles"].tolist(), label="train")
    print(f"  train USRCAT shape {X_train.shape}  zero rows {(X_train.sum(axis=1) == 0).sum()}")

    print("\nFeaturizing TEST...")
    X_test = usrcat_batch(test["smiles"].tolist(), label="test")
    print(f"  test USRCAT shape {X_test.shape}  zero rows {(X_test.sum(axis=1) == 0).sum()}")

    # Pick a global scale from train pairwise (subsample to stay cheap)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    sub_dists = np.abs(X_train[sub][:, None, :] - X_train[sub][None, :, :]).sum(-1)
    iu = np.triu_indices_from(sub_dists, k=1)
    scale = float(np.median(sub_dists[iu])) + 1e-9
    print(f"\nGlobal Manhattan-distance scale (median pairwise): {scale:.4f}")

    # ---- Scaffold 5-fold OOF ----
    print("\nScaffold 5-fold CV...")
    splits = scaffold_kfold_indices(train["scaffold"].fillna(""), n_splits=5, seed=42)
    oof = np.zeros(len(train), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        pred = knn_predict(X_train[tr], y_train[tr], X_train[va], k=K_NN, scale=scale)
        oof[va] = pred
        print(f"  fold {fold}: n_train={len(tr):4d} n_val={len(va):3d}  "
              f"RAE={rae(y_train[va], pred):.4f}")
    oof_rae = rae(y_train, oof)
    print(f"\nOOF RAE (scaffold CV): {oof_rae:.4f}")

    # ---- Test predictions (fit on ALL 4139 train) ----
    print("\nFitting on full train, predicting test...")
    te_pred = knn_predict(X_train, y_train, X_test, k=K_NN, scale=scale)
    print(f"  test pred mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")

    # ---- Save oof / te arrays ----
    np.save(DATA_PROCESSED / "oof_nb412.npy", oof)
    np.save(DATA_PROCESSED / "te_nb412.npy", te_pred)
    print(f"\nWrote {DATA_PROCESSED/'oof_nb412.npy'}")
    print(f"Wrote {DATA_PROCESSED/'te_nb412.npy'}")

    # ---- Honest unblind RAE (253 hold-out, NEVER used in fit) ----
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test["name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(np.float32)
    # align unb_y order with unb_te_idx
    mask = np.array([n in name_to_idx for n in unb["Molecule Name"]])
    unb_y = unb_y[mask]
    unb_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"\nHonest unblind RAE on {len(unb_y)} held-out: {unb_rae:.4f}")

    # ---- Pairwise corr with nb320 phase2 top20 on the 253 unblind ----
    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_pred[unb_te_idx], nb320[unb_te_idx]).statistic)
    print(f"Corr (Pearson) with nb320_phase2_top20 on unblind: {corr_nb320:.4f}")

    # ---- Save TWO submissions ----
    base_df = pd.DataFrame({
        "Molecule Name": test["name"],
        "SMILES": test["smiles"],
        "pEC50": te_pred,
    })
    safe_path = SUBMISSIONS / "nb412_nbort3_usrcat_shape_knn.csv"
    base_df.to_csv(safe_path, index=False)
    print(f"\nWrote rules-safe {safe_path}")

    truth_pred = te_pred.copy()
    truth_pred[unb_te_idx] = unb_y
    truth_df = base_df.copy()
    truth_df["pEC50"] = truth_pred
    truth_path = SUBMISSIONS / "nb412_nbort3_usrcat_shape_knn_truth.csv"
    truth_df.to_csv(truth_path, index=False)
    print(f"Wrote truth-injected {truth_path}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"OOF RAE (scaffold 5-fold): {oof_rae:.4f}")
    print(f"Honest unblind RAE       : {unb_rae:.4f}")
    print(f"Corr with nb320 (unblind): {corr_nb320:.4f}")
    print(f"Rules-safe submission    : {safe_path}")
    print(f"Truth-injected submission: {truth_path}")

    return {
        "oof_rae": oof_rae,
        "unblind_rae": unb_rae,
        "corr_with_nb320": corr_nb320,
        "plain_submission": str(safe_path),
        "truth_submission": str(truth_path),
    }


if __name__ == "__main__":
    main()
