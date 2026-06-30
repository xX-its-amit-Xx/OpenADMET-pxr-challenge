"""nb979 — LAST CPU-local lever: pharmacophore-anchor-fit. Can a compound's polar atoms (N/O)
geometrically match the 3D arrangement of the 5 PXR polar anchors? That's the binding determinant
(anchor complementarity), distinct from gross shape (nb977 USR, absorbed) and 2D HBA/HBD counts.

Anchor key-atom 3D geometry from a reference holo (2O9I); compound polar-atom pairwise distances
from an ETKDG conformer; feature = how well the compound can match each anchor-pair distance.
Test combined vs combined+anchorfit on the nb952 degradation curve. All on C:.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct"; CKPT = "C:/pxr_struct/nb979_ckpt.npz"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
# anchor residue -> side-chain functional atom (key H-bonding atom)
ANCHOR_ATOMS = {247: "OG", 285: "NE2", 327: "NE2", 407: "NE2", 410: "NH1"}


def anchor_geometry(pdb="2o9i"):
    st = gemmi.read_structure(f"data/external/pdb64_structures/{pdb}.cif"); st.setup_entities()
    pts = {}
    for ch in st[0]:
        for r in ch:
            if int(r.seqid.num) in ANCHOR_ATOMS:
                want = ANCHOR_ATOMS[int(r.seqid.num)]
                a = next((a for a in r if a.name == want), None) or next((a for a in r if a.name.startswith(want[0])), None)
                if a and int(r.seqid.num) not in pts:
                    pts[int(r.seqid.num)] = [a.pos.x, a.pos.y, a.pos.z]
    P = np.array([pts[k] for k in sorted(pts)])
    dd = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    return dd[np.triu_indices(len(P), 1)]  # the anchor-pair distances


def polar_pair_dists(smi):
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi)); p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0: return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=100)
        xyz = m.GetConformer().GetPositions()
        pol = [i for i, a in enumerate(m.GetAtoms()) if a.GetSymbol() in ("N", "O")]
        if len(pol) < 2: return np.array([])
        P = xyz[pol]
        dd = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        return dd[np.triu_indices(len(P), 1)]
    except Exception:
        return None


def anchorfit_features(smi, anchor_d):
    """For each anchor-pair distance, min |compound polar-pair dist - anchor dist| + match count."""
    pd = polar_pair_dists(smi)
    if pd is None: return None
    if len(pd) == 0: return np.concatenate([np.full(len(anchor_d), 9.9), [0.0]])
    feats = [float(np.min(np.abs(pd - ad))) for ad in anchor_d]            # best match per anchor pair
    nmatch = float(np.sum([(np.abs(pd - ad) < 1.0).any() for ad in anchor_d]))  # anchor pairs matched <1A
    return np.array(feats + [nmatch])


def compute(smiles, anchor_d, tag):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    nf = len(anchor_d) + 1; F = np.full((len(smiles), nf), np.nan)
    print(f"  {tag}: anchorfit for {len(smiles)} ...", flush=True); t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(anchorfit_features, s, anchor_d): i for i, s in enumerate(smiles)}
        for fut in as_completed(futs):
            i = futs[fut]; r = fut.result()
            if r is not None and len(r) == nf: F[i] = r
            done += 1
            if done % 1000 == 0: print(f"    {done}/{len(smiles)} ({time.time()-t0:.0f}s)", flush=True)
    return F


def curve(y, p, sv):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sv >= lo) & (sv < hi); nn = int(m.sum())
        out.append((f"[{lo:.1f},{hi:.1f})", nn, round(float(np.mean(np.abs(y[m]-p[m]))), 4) if nn else None))
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True); te = load_test()
    y = tr["pec50"].to_numpy(float)
    anchor_d = anchor_geometry()
    print(f"anchor-pair distances (A): {np.round(anchor_d,1)}")

    if os.path.exists(CKPT):
        z = np.load(CKPT); F_tr, F_te = z["F_tr"], z["F_te"]; print("resumed")
    else:
        F_tr = compute(tr["smiles"].tolist(), anchor_d, "train")
        F_te = compute(te["smiles"].tolist(), anchor_d, "test")
        np.savez(CKPT, F_tr=F_tr, F_te=F_te)
    np.save(f"{OUT}/nb979_anchorfit_test.npy", F_te)

    from sklearn.impute import SimpleImputer
    Ftr = SimpleImputer(strategy="median").fit_transform(F_tr).astype(np.float32)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in tr["smiles"]]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")
    Xc = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    ref_deep = next(r for r in json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"] if r["bin"] == "[0.0,0.3)")["mae"]

    res = {}
    for tagn, X in [("combined", Xc), ("combined+anchorfit", np.hstack([Xc, Ftr]))]:
        oof = np.full(len(y), np.nan)
        for tri, vai in folds:
            mdl = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], y[tri])
            oof[vai] = mdl.predict(X[vai])
        rows = curve(y, oof, max_sim); deep = next(m for b, n, m in rows if b == "[0.0,0.3)")
        res[tagn] = {"overall": round(float(rae(y, oof)), 4), "deep": deep}
        print(f"{tagn}: overall {res[tagn]['overall']} deep {deep}")
    helps = res["combined+anchorfit"]["deep"] < res["combined"]["deep"] - 0.003
    print(f"\nref {ref_deep}; anchorfit deep {res['combined+anchorfit']['deep']} "
          f"(delta {res['combined+anchorfit']['deep']-res['combined']['deep']:+.4f})")
    print(">>> ANCHOR-FIT ADDS -> multi-seed verify" if helps else
          ">>> anchor-fit absorbed -> CPU-local feature space DEFINITIVELY exhausted; only real lever = docking (Codespace)")
    json.dump({"ref_deep": ref_deep, "results": res, "helps": bool(helps)}, open(f"{OUT}/nb979_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb979_summary.json")


if __name__ == "__main__":
    main()
