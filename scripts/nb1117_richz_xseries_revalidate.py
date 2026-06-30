"""nb1117 — RE-VALIDATE the deployed rich-z/geom gain CROSS-SERIES (cycle-305 action item; decides final submission).

The deploy is nb3200 + rich-z/geom residual (~0.432, honest pooled -0.008). But cycle-305 showed 253-pooled residual
signals can be unblind-SERIES artifacts that DON'T transfer to the blinded 260 (pairwise-SAR looked like -0.0012 pooled
but +0.0029 cross-series). So: does rich-z/geom hold under LEAVE-SERIES-OUT? If the cross-series gain vanishes ->
deploy PLAIN nb3200 (more robust on the blinded LB). If it holds -> rich-z/geom is real. 30 seeds.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def clean(B): return np.where(np.isfinite(B), B, 0.0)


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    te = load_test(); scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]
    richz = clean(np.load(f"{MO}/test_richz.npy")[unb])
    geom = clean(np.load(f"{MO}/test_geom.npy")[unb])
    B = StandardScaler().fit_transform(np.hstack([richz, geom]))

    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(richz)))

    pooled, xseries = [], []
    for seed in range(1200, 1230):
        # residual OOF prediction (scaffold-CV)
        oofr = np.zeros(len(y))
        for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1, random_state=seed)
            m.fit(B[trn], err[trn]); oofr[val] = m.predict(B[val])
        # pooled best blend
        bp = rae(y, anchor)
        for w in np.linspace(0, 1.5, 31): bp = min(bp, rae(y, anchor + w * oofr))
        pooled.append(bp - rae(y, anchor))
        # cross-series: tune blend weight on K-1 series, apply held-out
        oofp = anchor.copy()
        for kk in range(6):
            trn = series != kk; val = series == kk
            if val.sum() < 5: continue
            bw, bb = 0, rae(y[trn], anchor[trn])
            for w in np.linspace(0, 1.5, 31):
                r = rae(y[trn], anchor[trn] + w * oofr[trn])
                if r < bb: bb, bw = r, w
            oofp[val] = anchor[val] + bw * oofr[val]
        xseries.append(rae(y, oofp) - rae(y, anchor))

    print(f"nb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"rich-z+geom residual:")
    print(f"  POOLED blend delta     {np.mean(pooled):+.4f}  (frac improved {np.mean(np.array(pooled)<-1e-6):.2f})")
    print(f"  X-SERIES blend delta   {np.mean(xseries):+.4f}  (frac improved {np.mean(np.array(xseries)<-1e-6):.2f})")
    verdict = "REAL (transfers)" if np.mean(xseries) < -0.002 else "PHANTOM (unblind-series artifact) -> deploy PLAIN nb3200"
    print(f"\nVERDICT: rich-z/geom is {verdict}")
    json.dump({"pooled": float(np.mean(pooled)), "xseries": float(np.mean(xseries))},
              open(f"{P}/nb1117_richz_xseries.json", "w"), indent=2)


if __name__ == "__main__":
    main()
