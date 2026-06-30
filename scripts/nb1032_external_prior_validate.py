"""nb1032 — build the PubChem near-neighbor external PXR prior and validate it on the honest 253.

From nb1031c matches (test compound <-> PXR-tested PubChem neighbor + that neighbor's active_rate across PXR assays).
Compute REAL Morgan similarity test<->neighbor, build a sim-weighted external signal per test compound:
  ext_active   = sim-weighted mean neighbor active_rate   (high => neighbors are PXR-active => likely active)
  ext_best_sim = best neighbor Morgan sim                 (confidence)
  ext_n        = number of PXR-tested neighbors            (evidence weight)
The F2-relevant direction: a novel compound whose neighbors are all PXR-INACTIVE (ext_active~0) is itself likely
inactive -> evidence to shrink an over-prediction DOWN.

Validation (honest, no test labels in the signal): (1) corr(ext signals, true pEC50 / nb3200 error) on covered
compounds; (2) ext signals as FEATURES on the nb3200 substrate (clipped residual, 30 seeds); (3) covered-subset RAE.
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


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    m = pd.read_csv(f"{D}/test_pxr_neighbor_matches.csv")
    nsmi = pd.read_csv(f"{D}/test_pxr_neighbor_smiles.csv").dropna(subset=["smiles"])
    cid2smi = dict(zip(nsmi["cid"], nsmi["smiles"]))
    te = load_test().reset_index(drop=True)

    # real Morgan sim test<->neighbor
    ucids = [c for c in m["cid"].unique() if c in cid2smi]
    nfp = morgan_fp_batch([cid2smi[c] for c in ucids]).astype(np.uint8)
    cid_idx = {c: i for i, c in enumerate(ucids)}
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(np.uint8)
    nsum = nfp.sum(1)

    ext_active = np.full(len(te), np.nan); ext_best = np.zeros(len(te)); ext_n = np.zeros(len(te))
    ext_minact = np.full(len(te), np.nan)
    for pos, grp in m.groupby("test_pos"):
        cids = [c for c in grp["cid"].tolist() if c in cid_idx]
        if not cids:
            continue
        idx = [cid_idx[c] for c in cids]
        inter = nfp[idx] @ tefp[pos]; uni = nsum[idx] + tefp[pos].sum() - inter
        sims = inter / np.clip(uni, 1, None)
        ar = grp.set_index("cid").loc[cids, "active_rate"].to_numpy().astype(float)
        keep = sims >= 0.35                      # ignore very weak matches
        if keep.sum() == 0:
            keep = sims >= 0.0
        w = sims[keep] ** 2
        ext_active[pos] = float(np.sum(w * ar[keep]) / np.sum(w)) if w.sum() > 0 else np.nan
        ext_best[pos] = float(sims.max()); ext_n[pos] = int(keep.sum())
        ext_minact[pos] = float(ar[keep].min())

    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    nov = pd.read_csv(f"{D}/novel_targets.csv").set_index("test_pos").loc[unb, "top1_sim"].to_numpy()
    ea, eb, en, em = ext_active[unb], ext_best[unb], ext_n[unb], ext_minact[unb]
    covered = ~np.isnan(ea)
    print(f"external prior covers {covered.sum()}/253 eval  ({(covered & (nov<0.5)).sum()} of the novel-tail eval)")

    if covered.sum() >= 8:
        cc = covered & (eb >= 0.4)
        print(f"\n--- diagnostics on covered (best_sim>=0.4, n={cc.sum()}) ---")
        if cc.sum() >= 8:
            print(f"  corr(ext_active, true pEC50) = {np.corrcoef(ea[cc], y[cc])[0,1]:+.3f}")
            print(f"  corr(ext_active, nb3200 err) = {np.corrcoef(ea[cc], (y-anchor)[cc])[0,1]:+.3f}")
            print(f"  covered mean true pEC50 {y[cc].mean():.2f} vs overall {y.mean():.2f}")

    # feature test on nb3200 substrate (fill uncovered with neutral medians)
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    F = np.column_stack([np.nan_to_num(ea, nan=np.nanmean(ea) if covered.any() else 0.0),
                         eb, en, np.nan_to_num(em, nan=np.nanmean(em) if covered.any() else 0.0)]).astype(np.float32)
    ds = []
    for s in range(1400, 1430):
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        ds.append(clipped(np.hstack([base, F]), resid, anchor, y, f) - clipped(base, resid, anchor, y, f))
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"\next signals as FEATURE on nb3200 (30 seeds): {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")

    # covered-subset targeted: does anchor RAE improve if we trust ext on covered compounds?
    if covered.sum() >= 10:
        cov = covered & (eb >= 0.4)
        print(f"\ncovered-subset (best_sim>=0.4, n={cov.sum()}): anchor RAE {rae(y[cov], anchor[cov]):.4f}")
    json.dump({"covered_eval": int(covered.sum()), "feature_delta": float(ds.mean()), "stable": bool(st)},
              open(f"{D}/nb1032_external_prior.json", "w"), indent=2)


if __name__ == "__main__":
    main()
