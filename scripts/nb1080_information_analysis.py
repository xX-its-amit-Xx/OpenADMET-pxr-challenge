"""nb1080 — INFORMATION-REPRESENTATION analysis (reframe: missing-information, not model-optimization).

Empirically tests the key hypotheses on OUR data (253 unblind labels + 4139 train):
  H9 NOISE FLOOR: from the 253 measurement SEs, the irreducible RAE. Is 0.3 even achievable?
  H1/H2/H10 ALIGNMENT: for each representation, does pairwise DISTANCE predict pairwise |dpEC50| (assay outcome)?
     -> the representation whose geometry aligns with ACTIVITY is "what the target sees". Plus test->train closeness.
  H4 RETRIEVAL: kNN(train->253) per representation -> RAE. Can activity = retrieval + local interpolation?
  H8/H1 CLIFFS: activity cliffs (high sim, big d) + chemically-distant-but-biologically-similar pairs.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_phase1_unblinded, load_train, load_test
from src.pxr.eval import rae
from src.pxr.chem import morgan_fp_batch
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs, MACCSkeys, Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "C:/pxr_struct/boltz/modal"; P = "data/processed"
PHYS = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
        "FractionCSP3", "NumAromaticRings", "NumAliphaticRings", "RingCount", "HeavyAtomCount",
        "NHOHCount", "NOCount", "qed"]


def physchem(smiles):
    fns = {n: getattr(Descriptors, n) for n in PHYS}
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append([fns[n](m) if m else np.nan for n in PHYS])
    a = np.array(out, np.float32); a = np.where(np.isfinite(a), a, np.nanmedian(a, 0)); return a


def maccs(smiles):
    return np.array([np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(str(s))), np.float32)
                     if Chem.MolFromSmiles(str(s)) else np.zeros(167, np.float32) for s in smiles])


def erg(smiles):
    out = [np.array(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(str(s))), np.float32)
           if Chem.MolFromSmiles(str(s)) else None for s in smiles]
    dim = next(len(v) for v in out if v is not None)
    return np.array([v if v is not None else np.zeros(dim, np.float32) for v in out])


def reps_for(smiles, struct=None):
    r = {"Morgan": morgan_fp_batch(smiles).astype(np.float32), "MACCS": maccs(smiles),
         "ErG": erg(smiles), "physchem": StandardScaler().fit_transform(physchem(smiles)).astype(np.float32)}
    if struct is not None:
        for k, v in struct.items():
            r[k] = v
    return r


def tani(A, B):
    A = (A > 0).astype(np.float32); B = (B > 0).astype(np.float32)
    inter = A @ B.T; s = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.clip(s, 1, None)


def main():
    u = load_phase1_unblinded(); y = u["pec50"].to_numpy(); se = u["pec50_se"].to_numpy()
    n = len(y); ybar = y.mean()
    # ---------- H9 NOISE FLOOR ----------
    denom = np.mean(np.abs(y - ybar))
    rae_floor = np.mean(se * np.sqrt(2 / np.pi)) / denom        # E|noise|=SE*sqrt(2/pi)
    # simulate: best model predicts true; observed = true+noise
    sims = []
    rng = np.random.RandomState(0)
    for _ in range(200):
        noise = rng.normal(0, se)
        sims.append(np.mean(np.abs(noise)) / denom)
    print("=== H9 NOISE FLOOR (the 253) ===")
    print(f"  median SE {np.median(se):.3f} | mean SE {np.mean(se):.3f} | pEC50 std {y.std():.3f}")
    print(f"  irreducible RAE floor = {rae_floor:.4f}  (sim {np.mean(sims):.4f} +/- {np.std(sims):.4f})")
    print(f"  -> deploy nb3200 0.4416 is {0.4416/rae_floor:.1f}x the floor; RAE 0.30 is {0.30/rae_floor:.1f}x the floor")

    # ---------- representations (253) ----------
    unb = np.load(f"{P}/_audit_unblind_idx.npy")
    te = load_test(); te_s = te["smiles"].to_numpy()
    smi = te_s[unb].tolist()
    struct = {}
    for k, f in [("richz_struct", f"{D}/test_richz.npy"), ("geom_struct", f"{D}/test_geom.npy")]:
        if os.path.exists(f):
            a = np.load(f)[unb]; a = np.where(np.isfinite(a), a, np.nanmedian(np.where(np.isfinite(a), a, np.nan), 0))
            struct[k] = StandardScaler().fit_transform(a).astype(np.float32)
    R = reps_for(smi, struct)

    # ---------- H1/H2/H10 ALIGNMENT: distance vs |dpEC50| ----------
    print("\n=== H1/H10 ALIGNMENT: Spearman(pairwise distance, pairwise |dpEC50|) on the 253 ===")
    dy = np.abs(y[:, None] - y[None, :]); iu = np.triu_indices(n, 1); dyf = dy[iu]
    align = {}
    for name, X in R.items():
        if name in ("Morgan", "MACCS"):
            dist = 1 - tani(X, X)
        else:
            dist = squareform(pdist(StandardScaler().fit_transform(X)))
        rho = spearmanr(dist[iu], dyf).correlation
        align[name] = float(rho)
        print(f"  {name:14s}: Spearman {rho:+.3f}  (higher = geometry aligns with activity)")

    # ---------- closeness: test->train NN sim per rep (Morgan/MACCS/ErG/physchem) ----------
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True); tr_s = tr["smiles"].tolist()
    Rtr = reps_for(tr_s)
    print("\n=== H4 RETRIEVAL: kNN(train->253, k=5 sim-weighted) RAE per representation ===")
    for name in ["Morgan", "MACCS", "ErG", "physchem"]:
        Xtr, Xte = Rtr[name], R[name]
        if name in ("Morgan", "MACCS"):
            sim = tani(Xte, Xtr)
        else:
            sc = StandardScaler().fit(np.vstack([Xtr, Xte]))
            from sklearn.metrics import pairwise_distances
            sim = 1 / (1 + pairwise_distances(sc.transform(Xte), sc.transform(Xtr)))
        pred = np.zeros(n); ytr = tr["pec50"].to_numpy()
        for i in range(n):
            k = np.argsort(-sim[i])[:5]; w = sim[i][k] ** 2
            pred[i] = np.sum(w * ytr[k]) / np.sum(w)
        med_nn = np.median(sim.max(1))
        print(f"  {name:14s}: retrieval RAE {rae(y, pred):.4f} | median top1 train-sim {med_nn:.3f}")
    print(f"  (reference: nb3200 model RAE 0.4416, noise floor {rae_floor:.3f})")

    # ---------- H8/H1 CLIFFS + distant-but-similar (within 253, Morgan) ----------
    msim = tani(R["Morgan"], R["Morgan"]); np.fill_diagonal(msim, 0)
    hi = (msim[iu] >= 0.6); cliff = hi & (dyf >= 1.0)
    lo = (msim[iu] <= 0.35); biosame = lo & (dyf <= 0.3)
    print("\n=== H8/H1 CLIFFS & DISTANT-BUT-SIMILAR (within 253, Morgan) ===")
    print(f"  high-sim pairs (>=0.6): {hi.sum()} | of those ACTIVITY CLIFFS (|d|>=1.0): {cliff.sum()} ({100*cliff.sum()/max(hi.sum(),1):.0f}%)")
    print(f"  low-sim pairs (<=0.35): {lo.sum()} | of those BIO-SAME (|d|<=0.3): {biosame.sum()} ({100*biosame.sum()/max(lo.sum(),1):.0f}%)")
    print(f"  -> a representation that pulls the {biosame.sum()} distant-but-biosame pairs together while keeping the {cliff.sum()} cliffs apart is the goal")
    json.dump({"rae_floor": float(rae_floor), "alignment": align}, open(f"{P}/nb1080_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
