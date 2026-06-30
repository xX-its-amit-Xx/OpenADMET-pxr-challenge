"""nb1123 — kNN-IMPUTED (read-across) biological fingerprint (user's refined idea: retrieve REAL measured neighbor values).

For each test compound, for each non-PXR NR target, find the K nearest compounds that HAVE real measured pEC50 for that
target (Tanimoto), weighted-average their MEASURED value -> imputed activity. Values are REAL measurements (read-across),
not chemistry-model predictions (the QAFFP failure). Panel = PXR's closest relatives (CAR/VDR/FXR/PPARg/RXRa/LXRa) =
best-case correlated donors. Feature = imputed-NR vector + neighbor-similarities. Honest gate: corr-with-nb3200-ERROR +
blend cross-series. If absorbed even on the best-case NR panel, proteome-scale won't help.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from scipy.stats import spearmanr
import lightgbm as lgb

P = "data/processed"


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    te = load_test(); smte = te["smiles"].to_numpy()[unb].tolist()
    Fte = fpf(smte)

    pap = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    targets = [t for t in pap["target_name"].unique() if t != "PXR"]
    print(f"NR donor panel (excl PXR): {targets}\n")
    K = 5
    cols, sims = [], []
    colnames = []
    for t in targets:
        sub = pap[pap["target_name"] == t].drop_duplicates("inchikey")
        if len(sub) < 50: continue
        Fp = fpf(sub["std_smiles"].tolist()); yp = sub["pec50"].to_numpy()
        # tanimoto test x panel
        inter = Fte @ Fp.T; s = Fp.sum(1)
        sim = inter / np.clip(Fte.sum(1)[:, None] + s[None, :] - inter, 1, None)
        imp = np.zeros(len(smte)); nn = np.zeros(len(smte))
        for i in range(len(smte)):
            k = np.argsort(sim[i])[::-1][:K]; w = sim[i][k] ** 2 + 1e-6
            imp[i] = np.sum(w * yp[k]) / np.sum(w); nn[i] = sim[i][k].max()
        cols.append(imp); sims.append(nn); colnames.append(t)
        print(f"  {t:8s} n={len(sub):5d} | imputed corr-with-PXR-truth {spearmanr(imp,y)[0]:+.3f} | "
              f"median NN-sim {np.median(nn):.2f}")
    B = np.column_stack(cols + sims)
    print(f"\nbio-fingerprint: {B.shape[1]} features ({len(colnames)} imputed-NR + {len(colnames)} similarities)")

    # deploy gate: residual-on-nb3200, 30-seed scaffold-CV, corr-with-error + blend + cross-series
    scaf = [murcko(s) for s in smte]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    Bz = StandardScaler().fit_transform(np.where(np.isfinite(B), B, 0.0))
    series = KMeans(6, n_init=5, random_state=0).fit_predict(PCA(min(8, Bz.shape[1]), random_state=0).fit_transform(Bz))
    corrs, pooled, xser = [], [], []
    for seed in range(1200, 1220):
        oof = np.zeros(len(y))
        for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1, random_state=seed)
            m.fit(Bz[trn], err[trn]); oof[val] = m.predict(Bz[val])
        corrs.append(np.corrcoef(oof, err)[0, 1] if oof.std() > 1e-9 else 0)
        b = rae(y, anchor)
        for w in np.linspace(0, 1.5, 31): b = min(b, rae(y, anchor + w * oof))
        pooled.append(b - rae(y, anchor))
        op = anchor.copy()
        for kk in range(6):
            tr2 = series != kk; va = series == kk
            if va.sum() < 5: continue
            bw, bb = 0, rae(y[tr2], anchor[tr2])
            for w in np.linspace(0, 1.5, 31):
                r = rae(y[tr2], anchor[tr2] + w * oof[tr2])
                if r < bb: bb, bw = r, w
            op[va] = anchor[va] + bw * oof[va]
        xser.append(rae(y, op) - rae(y, anchor))
    print(f"\nnb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"  corr(bio-fp residual pred, nb3200 error) {np.mean(corrs):+.3f}")
    print(f"  blend POOLED delta   {np.mean(pooled):+.4f}")
    print(f"  blend X-SERIES delta {np.mean(xser):+.4f}")
    print("GATE: real lever if corr>0 stable AND blend_xseries < -0.003 (held-out series).")
    json.dump({"corr_err": float(np.mean(corrs)), "pooled": float(np.mean(pooled)), "xseries": float(np.mean(xser))},
              open(f"{P}/nb1123_knn_biofp.json", "w"), indent=2)


if __name__ == "__main__":
    main()
