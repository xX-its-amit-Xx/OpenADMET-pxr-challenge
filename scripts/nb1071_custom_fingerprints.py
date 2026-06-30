"""nb1071 — [Thread 6] atom-based / pharmacophore / 3D fingerprints to BRING TRAIN<->TEST CLOSER + bake in 3D.

User idea: ECFP (Morgan) separates scaffold-diverse train/test; a pharmacophore/reduced-graph (ErG) or 3D-shape
descriptor abstracts the scaffold so different chemotypes with the same pharmacophore look CLOSE -> better OOD
generalization. Panel: ErG (reduced-graph pharmacophore, the scaffold-hopping headliner), AtomPair, TopTorsion,
MACCS, Avalon (2D); USRCAT + AUTOCORR3D + WHIM (3D shape from an ETKDG conformer).

Two decisive tests:
  (A) CLOSENESS: median test->train max-similarity in each FP space vs ECFP (does it pull test toward train?),
      stratified by the novel tail (train-ECFP-sim < 0.5 = where generalization matters).
  (B) SIGNAL: each FP block as a FEATURE on nb3200 (clipped, 30 seeds) + marginal over combined(2265).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs, rdMolDescriptors, MACCSkeys, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def fp2d(smiles, kind):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s)); v = None
        if m:
            try:
                if kind == "erg":
                    v = np.array(rdReducedGraphs.GetErGFingerprint(m), np.float32)
                elif kind == "atompair":
                    v = np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=1024), np.float32)
                elif kind == "toptorsion":
                    v = np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=1024), np.float32)
                elif kind == "maccs":
                    v = np.array(MACCSkeys.GenMACCSKeys(m), np.float32)
                elif kind == "avalon":
                    v = np.array(pyAvalonTools.GetAvalonFP(m, nBits=512), np.float32)
            except Exception:
                v = None
        out.append(v)
    dim = next(len(v) for v in out if v is not None)
    return np.array([v if v is not None else np.zeros(dim, np.float32) for v in out])


def fp3d(smiles):
    """USRCAT(60) + AUTOCORR3D(80) from one ETKDG conformer (WHIM dropped — NaN-prone)."""
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s)); v = None
        if m:
            try:
                mh = Chem.AddHs(m)
                if AllChem.EmbedMolecule(mh, randomSeed=42, maxIterations=400) == 0:
                    AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
                    v = np.concatenate([rdMolDescriptors.GetUSRCAT(mh),
                                        rdMolDescriptors.CalcAUTOCORR3D(mh)]).astype(np.float32)
                    v = np.where(np.isfinite(v), v, np.nan)
            except Exception:
                v = None
        out.append(v)
    dim = next((len(v) for v in out if v is not None), 140)
    arr = np.array([v if v is not None else np.full(dim, np.nan, np.float32) for v in out])
    cm = np.nanmedian(arr, 0); cm = np.where(np.isfinite(cm), cm, 0.0)   # per-column median impute
    idx = np.where(np.isnan(arr)); arr[idx] = np.take(cm, idx[1])
    return arr


def closeness(tr_fp, te_fp, novel, binary):
    """median test->train max-similarity (Tanimoto for binary, 1/(1+euclid) for descriptors)."""
    if binary:
        A = tr_fp.astype(bool); B = te_fp.astype(bool)
        sims = []
        for i in range(len(B)):
            inter = (B[i] & A).sum(1); uni = (B[i] | A).sum(1); sims.append(np.max(inter / np.clip(uni, 1, None)))
        sims = np.array(sims)
    else:
        sc = StandardScaler().fit(np.vstack([tr_fp, te_fp]))
        d = pairwise_distances(sc.transform(te_fp), sc.transform(tr_fp)); sims = 1 / (1 + d.min(1))
    return float(np.median(sims)), float(np.median(sims[novel]))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    tr_s = tr["smiles"].tolist(); te_s = te["smiles"].tolist()

    # ECFP baseline closeness (+ novel mask from ECFP train-sim)
    trE = morgan_fp_batch(tr_s).astype(bool); teE = morgan_fp_batch(te_s).astype(bool)
    top1 = np.array([np.max((teE[i] & trE).sum(1) / np.clip((teE[i] | trE).sum(1), 1, None)) for i in range(len(teE))])
    novel = top1 < 0.5
    ecfp_all, ecfp_nov = float(np.median(top1)), float(np.median(top1[novel]))
    print(f"ECFP closeness: all {ecfp_all:.3f} | novel-tail {ecfp_nov:.3f}  (novel n={novel.sum()})\n")

    print("=== (A) CLOSENESS: test->train median sim (higher = pulls test toward train) ===")
    print(f"  {'ECFP (baseline)':16s}: all {ecfp_all:.3f}  novel {ecfp_nov:.3f}")
    panels = {}
    for kind, binary in [("erg", False), ("atompair", True), ("toptorsion", True), ("maccs", True), ("avalon", True)]:
        trf = fp2d(tr_s, kind); tef = fp2d(te_s, kind)
        a, n = closeness(trf, tef, novel, binary)
        panels[kind] = tef
        flag = "  <-- closer on novel" if n > ecfp_nov + 0.02 else ""
        print(f"  {kind:16s}: all {a:.3f}  novel {n:.3f}{flag}")
    # 3D (subsample train for closeness)
    rng = np.random.RandomState(0); samp = rng.choice(len(tr_s), 800, replace=False)
    tr3 = fp3d([tr_s[i] for i in samp]); te3 = fp3d(te_s)
    a, n = closeness(tr3, te3, novel, False); panels["fp3d"] = te3
    print(f"  {'3D(USRCAT+AC3D+WHIM)':16s}: all {a:.3f}  novel {n:.3f}{'  <-- closer on novel' if n > ecfp_nov + 0.02 else ''}")

    # === (B) SIGNAL: each FP block as feature on nb3200 (marginal over combined 2265) ===
    smiles = te_s; scaf = [murcko(s) for s in np.array(te_s)[unb]]
    base = np.hstack([impute(combined(np.array(te_s)[unb].tolist())).astype(np.float32),
                      np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    def clipped(X, f):
        pred = anchor.copy()
        for tri, vai in f:
            mm = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
            p = anchor[vai] + mm.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
            pred[vai] = np.clip(p, lo, hi)
        return float(rae(y, pred))
    print("\n=== (B) SIGNAL: FP block MARGINAL over combined on nb3200 (30 seeds) ===")
    for name, tef in panels.items():
        sub = tef[unb]
        sub = np.where(np.isnan(sub), np.nanmedian(sub, 0), sub)
        k = min(20, sub.shape[1])
        blk = PCA(k, random_state=0).fit_transform(StandardScaler().fit_transform(sub)).astype(np.float32)
        ds = []
        for s in range(1400, 1430):
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            ds.append(clipped(np.hstack([base, blk]), f) - clipped(base, f))
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"  {name:16s}: {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
    json.dump({"ecfp_novel": ecfp_nov}, open(f"{D}/nb1071_fp.json", "w"), indent=2)


if __name__ == "__main__":
    main()
