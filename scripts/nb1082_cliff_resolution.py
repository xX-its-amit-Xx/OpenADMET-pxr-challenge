"""nb1082 — CLIFF RESOLUTION test (engages the user's pushback directly).

User: chemistry-similar (Tanimoto 0.9) compounds can have very different PXR activity (a rotamer flips it); a
BIOLOGICAL/interaction fingerprint should DISTINGUISH them where chemistry can't. Test, WITHIN the high-Morgan-sim
subset (where chemistry says 'same'), whether each representation's DISTANCE correlates with |dpEC50| (i.e. RESOLVES
the cliffs). And WITHIN the low-Morgan-sim subset, whether any rep gives SMALL distance to the bio-SAME pairs (BRIDGES
distant-but-same). A representation that resolves cliffs AND/OR bridges bio-same captures info chemistry misses.

Reps: Morgan, ErG (pharmacophore), cross-target affinity (7-NR), rich-z (PXR interaction), geom, physchem.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs, Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from src.pxr.chem import morgan_fp_batch

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"
PHYS = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "FractionCSP3",
        "NumAromaticRings", "RingCount", "qed"]


def erg(smiles):
    out = [np.array(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(str(s))), np.float32)
           if Chem.MolFromSmiles(str(s)) else None for s in smiles]
    dim = next(len(v) for v in out if v is not None)
    return np.array([v if v is not None else np.zeros(dim, np.float32) for v in out])


def physchem(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s)); out.append([getattr(Descriptors, n)(m) if m else np.nan for n in PHYS])
    a = np.array(out, np.float32); return np.where(np.isfinite(a), a, np.nanmedian(a, 0))


def tani(A):
    A = (A > 0).astype(np.float32); inter = A @ A.T
    s = A.sum(1)[:, None] + A.sum(1)[None, :] - inter; return inter / np.clip(s, 1, None)


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    te = load_test(); smi = te["smiles"].to_numpy()[unb].tolist()
    n = len(unb); iu = np.triu_indices(n, 1)
    dy = np.abs(y[:, None] - y[None, :])[iu]

    M = morgan_fp_batch(smi).astype(np.float32)
    msim = tani(M); mdist = (1 - msim)[iu]
    reps = {"Morgan": mdist, "ErG": squareform(pdist(StandardScaler().fit_transform(erg(smi))))[iu],
            "physchem": squareform(pdist(StandardScaler().fit_transform(physchem(smi))))[iu]}
    ct = np.load(f"{P}/te_nb258_cross_target.npy")[unb]
    reps["cross_target"] = squareform(pdist(StandardScaler().fit_transform(ct.reshape(-1, 1))))[iu]
    for k, f in [("richz", f"{MO}/test_richz.npy"), ("geom", f"{MO}/test_geom.npy")]:
        if os.path.exists(f):
            a = np.load(f)[unb]; a = np.where(np.isfinite(a), a, np.nanmedian(np.where(np.isfinite(a), a, np.nan), 0))
            reps[k] = squareform(pdist(StandardScaler().fit_transform(a)))[iu]

    msim_f = msim[iu]
    hi = msim_f >= 0.6; lo = msim_f <= 0.35
    cliff = hi & (dy >= 1.0); nonhi = hi & (dy <= 0.3)
    biosame = lo & (dy <= 0.3)
    print(f"high-Morgan-sim pairs: {hi.sum()} (cliffs |d|>=1: {cliff.sum()}, non-cliffs |d|<=0.3: {nonhi.sum()})")
    print(f"low-Morgan-sim pairs: {lo.sum()} (bio-same |d|<=0.3: {biosame.sum()})\n")

    print("=== CLIFF RESOLUTION: within high-Morgan-sim, does rep-distance track |dpEC50|? (resolves cliffs) ===")
    print(f"  {'representation':14s}  Spearman(dist,|dy| | high-sim)   mean-dist(cliff) vs (non-cliff)")
    for name, dist in reps.items():
        rho = spearmanr(dist[hi], dy[hi]).correlation if hi.sum() > 10 else np.nan
        dn = (dist - dist.mean()) / (dist.std() + 1e-9)   # z-normalize the distance
        cliff_d = dn[cliff].mean() if cliff.sum() else np.nan
        non_d = dn[nonhi].mean() if nonhi.sum() else np.nan
        sep = cliff_d - non_d
        flag = "  <-- RESOLVES cliffs" if (rho is not np.nan and rho > 0.15 and sep > 0.1) else ""
        print(f"  {name:14s}  {rho:+.3f}                          {cliff_d:+.2f} vs {non_d:+.2f}  (sep {sep:+.2f}){flag}")

    print("\n=== BRIDGE: within low-Morgan-sim, is rep-distance SMALL for bio-same pairs? (bridges distant-but-same) ===")
    for name, dist in reps.items():
        dn = (dist - dist.mean()) / (dist.std() + 1e-9)
        bs = dn[biosame].mean() if biosame.sum() else np.nan
        other = dn[lo & (dy >= 1.0)].mean() if (lo & (dy >= 1.0)).sum() else np.nan
        flag = "  <-- BRIDGES bio-same" if (bs < other - 0.1) else ""
        print(f"  {name:14s}  mean-dist(bio-same) {bs:+.2f} vs (distant-bio-diff) {other:+.2f}{flag}")
    json.dump({"n_cliffs": int(cliff.sum())}, open(f"{P}/nb1082_cliff.json", "w"), indent=2)


if __name__ == "__main__":
    main()
