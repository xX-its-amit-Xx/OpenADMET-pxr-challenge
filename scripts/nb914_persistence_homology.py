"""nb914 -- Persistence-homology-style topological features for PXR.

Hypothesis:
  Morgan FPs encode *local* substructures (radius<=2 environments).
  Global molecular topology -- ring fusion patterns, scaffold size,
  spiro/bridge atoms, ring "loops" -- is captured only indirectly.
  Persistence homology (PH) summarizes shape: H0 = connected components,
  H1 = loops/cycles, with birth/death "lifetimes" in a Vietoris-Rips
  filtration over 3D atom coords.

Plan:
  1. Embed 3D conformer (AllChem.EmbedMolecule + MMFFOptimize, seed=42)
  2. Compute persistence diagrams (gudhi/ripser if available; else
     fall back to ring-graph topological proxies).
  3. Extract 6 summary stats per molecule:
        n_h0_features, total_h0_lifetime,
        n_h1_features, mean_h1_lifetime, max_h1_lifetime,
        persistence_entropy
  4. Concat with Morgan(2048) + RDKit(~217) -> ~2271 features.
  5. LGBM Huber (n_est=1500, num_leaves=64, lr=0.03), scaffold 5-fold CV.
  6. Score on 253 Phase-1 unblind labels. Save submission.

Memory: 4652 mols * 6 topo + 4652*2265 dense float32 ~= 42 MB. Safe.

Outputs:
  C:/pxr_artifacts/nb914_topology_features.npy   (4652, 6) float32
  data/processed/te_nb914.npy                    (513,)   float32
  submissions/nb914_persistence_homology.csv
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, GraphDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.chem import add_standard_columns
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

import lightgbm as lgb

TAG = "nb914"
SEED = 42
ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

# Try to load a real PH backend
_HAS_GUDHI = False
_HAS_RIPSER = False
try:
    import gudhi  # type: ignore
    _HAS_GUDHI = True
except Exception:
    try:
        from ripser import ripser  # type: ignore
        _HAS_RIPSER = True
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Topological features
# ----------------------------------------------------------------------------
TOPO_COLS = [
    "n_h0_features", "total_h0_lifetime",
    "n_h1_features", "mean_h1_lifetime", "max_h1_lifetime",
    "persistence_entropy",
]


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes."""
    L = lifetimes[lifetimes > 0]
    if L.size == 0:
        return 0.0
    p = L / L.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _ph_from_coords(coords: np.ndarray, max_edge: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (h0_lifetimes, h1_lifetimes) from a Vietoris-Rips filtration."""
    if _HAS_GUDHI:
        rips = gudhi.RipsComplex(points=coords, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=2)
        diag = st.persistence()
        h0 = np.array([d - b for dim, (b, d) in diag if dim == 0 and np.isfinite(d)])
        h1 = np.array([d - b for dim, (b, d) in diag if dim == 1 and np.isfinite(d)])
        return h0, h1
    if _HAS_RIPSER:
        res = ripser(coords, maxdim=1, thresh=max_edge)
        dgms = res["dgms"]
        d0 = dgms[0]; d1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        m0 = np.isfinite(d0[:, 1]); m1 = np.isfinite(d1[:, 1]) if d1.size else np.array([], dtype=bool)
        h0 = d0[m0, 1] - d0[m0, 0]
        h1 = d1[m1, 1] - d1[m1, 0] if d1.size else np.empty((0,))
        return h0, h1
    return np.empty((0,)), np.empty((0,))


def _topo_proxies(mol: Chem.Mol) -> dict:
    """Ring-graph proxy when no PH backend is installed."""
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    n_rings = len(ring_sizes)
    mean_size = float(np.mean(ring_sizes)) if ring_sizes else 0.0
    # Fused: rings sharing >=2 atoms (an edge)
    atom_rings = ri.AtomRings()
    fused = 0
    spiro = 0
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            shared = len(set(atom_rings[i]) & set(atom_rings[j]))
            if shared >= 2:
                fused += 1
            elif shared == 1:
                spiro += 1
    # Bridging atoms: atoms in 2+ rings of size >=3 with no shared bond
    atom_in_n_rings = np.zeros(mol.GetNumAtoms(), dtype=int)
    for r in atom_rings:
        for a in r:
            atom_in_n_rings[a] += 1
    bridging = int(np.sum(atom_in_n_rings >= 2))
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_rings = scaf.GetRingInfo().NumRings() if scaf is not None else 0
    except Exception:
        scaf_rings = 0
    # Map proxies -> the 6 TOPO_COLS schema
    return {
        "n_h0_features":      float(mol.GetNumAtoms()),
        "total_h0_lifetime":  float(GraphDescriptors.BalabanJ(mol)),
        "n_h1_features":      float(n_rings),
        "mean_h1_lifetime":   float(mean_size),
        "max_h1_lifetime":    float(max(ring_sizes) if ring_sizes else 0),
        "persistence_entropy": float(fused + 0.5 * spiro + 0.25 * bridging + scaf_rings),
    }


def compute_topo(smiles: list[str]) -> np.ndarray:
    """Return (N, 6) float32 topological-feature matrix."""
    out = np.zeros((len(smiles), len(TOPO_COLS)), dtype=np.float32)
    backend = "gudhi" if _HAS_GUDHI else ("ripser" if _HAS_RIPSER else "proxy")
    print(f"Topo backend: {backend}")
    for i, smi in enumerate(smiles):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mh = Chem.AddHs(m)
            if backend != "proxy":
                ok = AllChem.EmbedMolecule(mh, randomSeed=SEED)
                if ok == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
                    except Exception:
                        pass
                    conf = mh.GetConformer()
                    coords = np.array([list(conf.GetAtomPosition(k))
                                       for k in range(mh.GetNumAtoms())], dtype=np.float64)
                    h0, h1 = _ph_from_coords(coords, max_edge=6.0)
                    if h1.size == 0:
                        # fall back on this molecule
                        v = _topo_proxies(m)
                    else:
                        v = {
                            "n_h0_features":      float(len(h0)),
                            "total_h0_lifetime":  float(h0.sum()),
                            "n_h1_features":      float(len(h1)),
                            "mean_h1_lifetime":   float(h1.mean()),
                            "max_h1_lifetime":    float(h1.max()),
                            "persistence_entropy": _persistence_entropy(
                                np.concatenate([h0, h1])),
                        }
                else:
                    v = _topo_proxies(m)
            else:
                v = _topo_proxies(m)
            for j, c in enumerate(TOPO_COLS):
                out[i, j] = v[c]
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f"  topo {i+1}/{len(smiles)}")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- persistence-homology features + LGBM Huber")
    print("=" * 78)

    te_csv = DATA_RAW / "pxr-challenge_TEST_BLINDED.csv"
    unb_csv = DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"
    if not te_csv.exists():
        return {"success": False, "missing": str(te_csv)}

    tr = add_standard_columns(load_train())
    te_df = pd.read_csv(te_csv)
    smi_tr = tr["std_smiles"].tolist()
    smi_te = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
              for s in te_df["SMILES"]]
    y_tr = tr["pec50"].astype(np.float64).values
    scaf_tr = tr["scaffold"].fillna("").tolist()
    n_tr, n_te = len(smi_tr), len(smi_te)
    print(f"Train n={n_tr}  Test n={n_te}")

    # Topo features
    topo_path = ART / f"{TAG}_topology_features.npy"
    if topo_path.exists():
        topo_all = np.load(topo_path)
        print(f"Loaded cached topo features: {topo_all.shape}")
    else:
        print("Computing topology features (train+test)...")
        topo_tr = compute_topo(smi_tr)
        topo_te = compute_topo(smi_te)
        topo_all = np.vstack([topo_tr, topo_te]).astype(np.float32)
        np.save(topo_path, topo_all)
    topo_tr = topo_all[:n_tr]
    topo_te = topo_all[n_tr:]
    print(f"topo means: {topo_all.mean(axis=0)}")

    # Combined features (Morgan + RDKit)
    print("Computing Morgan + RDKit features...")
    X_tr = impute(combined(smi_tr))
    X_te = impute(combined(smi_te))
    X_tr = np.hstack([X_tr, topo_tr])
    X_te = np.hstack([X_te, topo_te])
    print(f"X_tr {X_tr.shape}  X_te {X_te.shape}")

    # Scaffold 5-fold CV LGBM Huber
    print("Training LGBM Huber, scaffold 5-fold CV...")
    splits = scaffold_kfold_indices(scaf_tr, n_splits=5, shuffle=True, seed=SEED)
    params = dict(
        objective="huber", alpha=0.9, n_estimators=1500,
        num_leaves=64, learning_rate=0.03,
        min_data_in_leaf=20, feature_fraction=0.85, bagging_fraction=0.85,
        bagging_freq=5, verbose=-1, n_jobs=4, random_state=SEED,
    )
    oof = np.zeros(n_tr, dtype=np.float32)
    te_fold = np.zeros((5, n_te), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_fold[k] = m.predict(X_te)
        print(f"  fold {k} RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}")
    te_pred = te_fold.mean(axis=0)
    print(f"OOF scaffold-CV RAE: {rae(y_tr, oof):.4f}")

    # Unblind eval
    in_rae = None
    if unb_csv.exists():
        unb = pd.read_csv(unb_csv)
        n2i = {n: i for i, n in enumerate(te_df["Molecule Name"])}
        unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
        idx = np.array([n2i[n] for n in unb["Molecule Name"]], dtype=int)
        y_u = unb["pEC50"].astype(float).values
        in_rae = float(rae(y_u, te_pred[idx]))
        print("=" * 78)
        print(f"UNBLIND (n={len(idx)}): RAE = {in_rae:.4f}")
        print("=" * 78)

    # Save
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred.astype(np.float32))
    sub = SUBMISSIONS / f"{TAG}_persistence_homology.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_pred,
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    return {"success": True, "in_rae": in_rae, "submission": str(sub)}


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
