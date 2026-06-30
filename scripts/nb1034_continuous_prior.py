"""nb1034 — sharpest external prior: continuous neighbor pEC50 (active neighbor -> measured potency; inactive ->
low prior), sim-weighted. Test as targeted directional blend with nb3200 (honest cross-fit) + as feature, focusing
on the covered subset and the novel tail. Last attempt to turn the real-but-weak external signal into a stable gain.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def build():
    m = pd.read_csv(f"{D}/test_pxr_neighbor_matches.csv")
    nsmi = pd.read_csv(f"{D}/test_pxr_neighbor_smiles.csv").dropna(subset=["smiles"])
    pot = pd.read_parquet(f"{D}/pxr_neighbor_potency.parquet")
    cid2smi = dict(zip(nsmi["cid"], nsmi["smiles"]))
    cid2pec = dict(zip(pot["cid"], pot["pec50"]))
    cid2act = dict(zip(pot["cid"], pot["any_active"]))
    LOW = 3.8  # inactive-neighbor prior (qHTS non-responders, pEC50 below ~ assay floor)
    te = load_test().reset_index(drop=True)
    ucids = [c for c in m["cid"].unique() if c in cid2smi]
    nfp = morgan_fp_batch([cid2smi[c] for c in ucids]).astype(np.uint8)
    cidx = {c: i for i, c in enumerate(ucids)}
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(np.uint8); nsum = nfp.sum(1)

    ext_pec = np.full(len(te), np.nan); ext_best = np.zeros(len(te)); ext_n = np.zeros(len(te))
    for pos, grp in m.groupby("test_pos"):
        cids = [c for c in grp["cid"].tolist() if c in cidx]
        if not cids:
            continue
        idx = [cidx[c] for c in cids]
        inter = nfp[idx] @ tefp[pos]; uni = nsum[idx] + tefp[pos].sum() - inter
        sims = inter / np.clip(uni, 1, None)
        npec = []
        for c in cids:
            if cid2act.get(c, False) and np.isfinite(cid2pec.get(c, np.nan)):
                npec.append(cid2pec[c])
            else:
                npec.append(LOW)                 # tested-inactive neighbor -> low prior
        npec = np.array(npec); w = sims ** 2
        keep = sims >= 0.0
        ext_pec[pos] = float(np.sum(w * npec) / np.sum(w)) if w.sum() > 0 else np.nan
        ext_best[pos] = float(sims.max()); ext_n[pos] = len(cids)
    return ext_pec, ext_best, ext_n


def main():
    ext_pec, ext_best, ext_n = build()
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); r_anchor = rae(y, anchor)
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    ep, eb, en = ext_pec[unb], ext_best[unb], ext_n[unb]
    cov = ~np.isnan(ep)
    print(f"continuous prior covers {cov.sum()}/253")
    for thr in [0.0, 0.35, 0.45]:
        c = cov & (eb >= thr)
        if c.sum() >= 10:
            print(f"  sim>={thr} (n={c.sum()}): corr(ext_pec50, true)={np.corrcoef(ep[c],y[c])[0,1]:+.3f}  "
                  f"corr(ext_pec50, nb3200_err)={np.corrcoef(ep[c],(y-anchor)[c])[0,1]:+.3f}")

    # targeted directional blend: pred = (1-g)*anchor + g*ext_pec50 on covered (sim>=thr); g fit per-fold
    print("\ntargeted blend (g fit per-fold by grid on train-covered):")
    for SIM_THR in [0.30, 0.40]:
        conf = cov & (eb >= SIM_THR)
        deltas = []
        for s in range(1400, 1430):
            folds = scaffold_kfold_indices(scaf, 5, seed=s)
            pred = anchor.copy()
            for tri, vai in folds:
                ct = conf & np.isin(np.arange(len(y)), tri); cv = conf & np.isin(np.arange(len(y)), vai)
                if ct.sum() < 8 or cv.sum() == 0:
                    continue
                bg, br = 0.0, rae(y[ct], anchor[ct])
                for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    r = rae(y[ct], (1 - g) * anchor[ct] + g * ep[ct])
                    if r < br:
                        br, bg = r, g
                pred[cv] = (1 - bg) * anchor[cv] + bg * ep[cv]
            deltas.append(rae(y, pred) - r_anchor)
        deltas = np.array(deltas); st = deltas.mean() < 0 and abs(deltas.mean()) > deltas.std()
        print(f"  sim>={SIM_THR} (n={conf.sum()}): delta {deltas.mean():+.5f} +/- {deltas.std():.5f} "
              f"wins {int((deltas<0).sum())}/30 stable={st}")

    # feature test (continuous prior + confidence) on nb3200 substrate
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    F = np.column_stack([np.nan_to_num(ep, nan=np.nanmean(ep)), eb, en]).astype(np.float32)
    resid = y - anchor
    def clipped(X, f):
        pred = anchor.copy()
        for tri, vai in f:
            mm = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
            p = anchor[vai] + mm.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
            pred[vai] = np.clip(p, lo, hi)
        return float(rae(y, pred))
    ds = []
    for s in range(1400, 1430):
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        ds.append(clipped(np.hstack([base, F]), f) - clipped(base, f))
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"\ncontinuous prior as FEATURE on nb3200 (30 seeds): {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
    json.dump({"covered": int(cov.sum()), "feature_delta": float(ds.mean()), "feature_stable": bool(st)},
              open(f"{D}/nb1034_continuous.json", "w"), indent=2)


if __name__ == "__main__":
    main()
